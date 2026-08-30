from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from . import indicators as ind


SignalBuilder = Callable[[pd.DataFrame], pd.Series]


@dataclass(frozen=True)
class Candidate:
    name: str
    school: str
    params: dict
    complexity: int
    builder: SignalBuilder
    directional_builder: SignalBuilder | None = None


def _clean(signal: pd.Series) -> pd.Series:
    return signal.fillna(False).astype(int)


def _clean_directional(signal: pd.Series) -> pd.Series:
    clean = pd.to_numeric(signal, errors="coerce").fillna(0)
    clean = clean.where(clean.isin([-1, 0, 1]), 0)
    return clean.astype(int)


def trend_candidate(fast: int, slow: int) -> Candidate:
    def build(df):
        a, b = ind.ema(df.close, fast), ind.ema(df.close, slow)
        return _clean((a > b) & (df.close > a))

    def build_directional(df):
        a, b = ind.ema(df.close, fast), ind.ema(df.close, slow)
        signal = pd.Series(0, index=df.index, dtype=int)
        signal.loc[(a > b) & (df.close > a)] = 1
        signal.loc[(a < b) & (df.close < a)] = -1
        return _clean_directional(signal)

    return Candidate(f"trend_ema_{fast}_{slow}", "trend", {"fast": fast, "slow": slow}, 2, build, build_directional)


def momentum_candidate(period: int, threshold: float) -> Candidate:
    def build(df):
        r = ind.rsi(df.close, period)
        return _clean(r > threshold)

    def build_directional(df):
        r = ind.rsi(df.close, period)
        signal = pd.Series(0, index=df.index, dtype=int)
        signal.loc[r > threshold] = 1
        signal.loc[r < (100.0 - threshold)] = -1
        return _clean_directional(signal)

    return Candidate(f"momentum_rsi_{period}_{int(threshold)}", "momentum", {"period": period, "threshold": threshold}, 2, build, build_directional)


def mean_reversion_candidate(period: int, width: float) -> Candidate:
    def build(df):
        low, _, _ = ind.bollinger(df.close, period, width)
        return _clean(df.close < low)

    def build_directional(df):
        lower, _, upper = ind.bollinger(df.close, period, width)
        signal = pd.Series(0, index=df.index, dtype=int)
        signal.loc[df.close < lower] = 1
        signal.loc[df.close > upper] = -1
        return _clean_directional(signal)

    return Candidate(f"meanrev_bb_{period}_{width:.1f}", "mean_reversion", {"period": period, "width": width}, 2, build, build_directional)


def breakout_candidate(period: int) -> Candidate:
    def build(df):
        _, upper = ind.donchian(df, period)
        return _clean(df.close > upper)

    def build_directional(df):
        lower, upper = ind.donchian(df, period)
        signal = pd.Series(0, index=df.index, dtype=int)
        signal.loc[df.close > upper] = 1
        signal.loc[df.close < lower] = -1
        return _clean_directional(signal)

    return Candidate(f"breakout_donchian_{period}", "breakout", {"period": period}, 1, build, build_directional)


def volume_breakout_candidate(period: int, relvol: float) -> Candidate:
    def build(df):
        _, upper = ind.donchian(df, period)
        rv = ind.relative_volume(df.volume, period)
        return _clean((df.close > upper) & (rv > relvol))

    def build_directional(df):
        lower, upper = ind.donchian(df, period)
        rv = ind.relative_volume(df.volume, period)
        active = rv > relvol
        signal = pd.Series(0, index=df.index, dtype=int)
        signal.loc[(df.close > upper) & active] = 1
        signal.loc[(df.close < lower) & active] = -1
        return _clean_directional(signal)

    return Candidate(f"volume_breakout_{period}_{relvol:.1f}", "volume", {"period": period, "relvol": relvol}, 2, build, build_directional)


def price_action_candidate(lookback: int, body_ratio: float) -> Candidate:
    def build(df):
        body = (df.close - df.open).abs()
        rng = (df.high - df.low).replace(0, np.nan)
        prior_high = df.high.rolling(lookback, min_periods=lookback).max().shift(1)
        return _clean((df.close > df.open) & ((body / rng) > body_ratio) & (df.close > prior_high))

    def build_directional(df):
        body = (df.close - df.open).abs()
        rng = (df.high - df.low).replace(0, np.nan)
        prior_high = df.high.rolling(lookback, min_periods=lookback).max().shift(1)
        prior_low = df.low.rolling(lookback, min_periods=lookback).min().shift(1)
        body_ok = (body / rng) > body_ratio
        signal = pd.Series(0, index=df.index, dtype=int)
        signal.loc[(df.close > df.open) & body_ok & (df.close > prior_high)] = 1
        signal.loc[(df.close < df.open) & body_ok & (df.close < prior_low)] = -1
        return _clean_directional(signal)

    return Candidate(f"price_action_{lookback}_{body_ratio:.1f}", "price_action", {"lookback": lookback, "body_ratio": body_ratio}, 2, build, build_directional)


def candidate_library() -> list[Candidate]:
    out = []
    for fast, slow in [(8, 21), (12, 30), (20, 50), (20, 80), (30, 100)]:
        out.append(trend_candidate(fast, slow))
    for p, t in [(7, 55), (14, 55), (14, 60), (21, 55), (21, 60)]:
        out.append(momentum_candidate(p, t))
    for p, w in [(15, 1.5), (20, 2.0), (30, 2.0), (40, 2.5)]:
        out.append(mean_reversion_candidate(p, w))
    for p in [10, 20, 30, 50, 80]:
        out.append(breakout_candidate(p))
        for rv in [1.1, 1.3, 1.5]:
            out.append(volume_breakout_candidate(p, rv))
    for lb in [5, 10, 20]:
        for br in [0.55, 0.65, 0.75]:
            out.append(price_action_candidate(lb, br))
    return out
