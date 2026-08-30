from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .schools import Candidate


@dataclass(frozen=True)
class Strategy:
    candidate: Candidate
    filters: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        return self.candidate.name + ("+" + "+".join(self.filters) if self.filters else "")

    @property
    def complexity(self) -> int:
        return self.candidate.complexity + len(self.filters)

    def signal(self, df: pd.DataFrame) -> pd.Series:
        base = self.candidate.builder(df).astype(bool)
        result = base.copy()
        for item in self.filters:
            if item == "above_sma200":
                sma = df.close.rolling(200, min_periods=200).mean()
                result &= df.close > sma
            elif item == "positive_roc":
                result &= df.close.pct_change(20) > 0
            elif item == "low_volatility":
                vol = df.close.pct_change().rolling(20, min_periods=20).std()
                result &= vol < vol.rolling(100, min_periods=100).median()
            else:
                raise ValueError(f"Unknown filter: {item}")
        return result.fillna(False).astype(int)


def add_filters(candidates: list[Candidate]) -> list[Strategy]:
    strategies = [Strategy(c) for c in candidates]
    for c in candidates:
        for f in ("above_sma200", "positive_roc", "low_volatility"):
            strategies.append(Strategy(c, (f,)))
    return strategies


def mutate_strategy(strategy: Strategy, rng: np.random.Generator) -> Strategy:
    p = dict(strategy.candidate.params)
    changes = {
        "fast": (4, 40, int), "slow": (30, 160, int), "period": (5, 60, int),
        "threshold": (50, 70, float), "width": (1.0, 3.0, float), "relvol": (1.0, 2.0, float),
        "lookback": (3, 30, int), "body_ratio": (0.45, 0.90, float),
    }
    if not p:
        return strategy
    key = str(rng.choice(list(p)))
    if key in changes:
        lo, hi, cast = changes[key]
        center = float(p[key])
        p[key] = cast(np.clip(center * rng.uniform(0.75, 1.25), lo, hi))
    if "fast" in p and "slow" in p and p["fast"] >= p["slow"]:
        p["fast"], p["slow"] = min(p["fast"], p["slow"] - 1), max(p["slow"], p["fast"] + 1)
    # Rebuild known school candidate using the mutated parameters.
    c = replace(strategy.candidate, params=p)
    from .schools import (
        breakout_candidate, mean_reversion_candidate, momentum_candidate,
        price_action_candidate, trend_candidate, volume_breakout_candidate,
    )
    school = c.school
    if school == "trend": c = trend_candidate(p["fast"], p["slow"])
    elif school == "momentum": c = momentum_candidate(p["period"], p["threshold"])
    elif school == "mean_reversion": c = mean_reversion_candidate(p["period"], p["width"])
    elif school == "breakout": c = breakout_candidate(p["period"])
    elif school == "volume": c = volume_breakout_candidate(p["period"], p["relvol"])
    elif school == "price_action": c = price_action_candidate(p["lookback"], p["body_ratio"])
    return Strategy(c, strategy.filters)
