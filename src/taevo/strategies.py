from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .schools import Candidate


@dataclass(frozen=True)
class Strategy:
    candidate: Candidate
    filters: tuple[str, ...] = ()
    directional: bool = False

    @property
    def name(self) -> str:
        suffix = "+" + "+".join(self.filters) if self.filters else ""
        mode = "+directional" if self.directional else ""
        return self.candidate.name + suffix + mode

    @property
    def complexity(self) -> int:
        return self.candidate.complexity + len(self.filters) + int(self.directional)

    def signal(self, df: pd.DataFrame) -> pd.Series:
        if self.directional and self.candidate.directional_builder is not None:
            result = self.candidate.directional_builder(df).copy()
        else:
            result = self.candidate.builder(df).astype(bool).astype(int)

        for item in self.filters:
            if item == "above_sma200":
                sma = df.close.rolling(200, min_periods=200).mean()
                mask = df.close > sma
            elif item == "positive_roc":
                mask = df.close.pct_change(20) > 0
            elif item == "low_volatility":
                vol = df.close.pct_change().rolling(20, min_periods=20).std()
                mask = vol < vol.rolling(100, min_periods=100).median()
            else:
                raise ValueError(f"Unknown filter: {item}")

            if self.directional:
                result = result.where(mask, 0)
            else:
                result = result.astype(bool) & mask

        if self.directional:
            result = pd.to_numeric(result, errors="coerce").fillna(0)
            result = result.where(result.isin([-1, 0, 1]), 0).astype(int)
        else:
            result = result.fillna(False).astype(int)

        return result


def add_filters(candidates: list[Candidate]) -> list[Strategy]:
    strategies: list[Strategy] = []
    filters = ("above_sma200", "positive_roc", "low_volatility")

    for c in candidates:
        strategies.append(Strategy(c, directional=False))

    for c in candidates:
        for f in filters:
            strategies.append(Strategy(c, (f,), directional=False))

    for c in candidates:
        strategies.append(Strategy(c, directional=True))

    for c in candidates:
        for f in filters:
            strategies.append(Strategy(c, (f,), directional=True))

    return strategies


def _candidate_from_parts(school: str, params: dict) -> Candidate:
    from .schools import breakout_candidate, mean_reversion_candidate, momentum_candidate, price_action_candidate, trend_candidate, volume_breakout_candidate
    if school == "trend": return trend_candidate(int(params["fast"]), int(params["slow"]))
    if school == "momentum": return momentum_candidate(int(params["period"]), float(params["threshold"]))
    if school == "mean_reversion": return mean_reversion_candidate(int(params["period"]), float(params["width"]))
    if school == "breakout": return breakout_candidate(int(params["period"]))
    if school == "volume": return volume_breakout_candidate(int(params["period"]), float(params["relvol"]))
    if school == "price_action": return price_action_candidate(int(params["lookback"]), float(params["body_ratio"]))
    raise ValueError(f"Unknown school: {school}")


def strategy_from_record(record: dict) -> Strategy:
    validation = record["validation"]
    base_name, *filters = str(record["strategy"]).split("+")
    candidate = _candidate_from_parts(validation["school"], validation["params"])
    directional = str(record["strategy"]).endswith("+directional")
    filters = [f for f in filters if f != "directional"]
    return Strategy(candidate, tuple(filters), directional=directional)


def mutate_strategy(strategy: Strategy, rng: np.random.Generator) -> Strategy:
    p = dict(strategy.candidate.params)
    changes = {
        "fast": (4, 40, int), "slow": (30, 160, int), "period": (5, 60, int),
        "threshold": (50, 70, float), "width": (1.0, 3.0, float), "relvol": (1.0, 2.0, float),
        "lookback": (3, 30, int), "body_ratio": (0.45, 0.90, float),
    }
    if not p:
        return Strategy(
            _candidate_from_parts(strategy.candidate.school, p),
            strategy.filters,
            directional=not strategy.directional if rng.random() < 0.20 else strategy.directional,
        )

    key = str(rng.choice(list(p)))
    if key in changes:
        lo, hi, cast = changes[key]
        p[key] = cast(np.clip(float(p[key]) * rng.uniform(0.75, 1.25), lo, hi))

    if "fast" in p and "slow" in p and p["fast"] >= p["slow"]:
        p["fast"], p["slow"] = max(4, p["slow"] - 1), min(160, p["fast"] + 1)

    directional = strategy.directional
    if rng.random() < 0.20:
        directional = not directional

    return Strategy(
        _candidate_from_parts(strategy.candidate.school, p),
        strategy.filters,
        directional=directional,
    )
