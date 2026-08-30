from __future__ import annotations

import numpy as np
import pandas as pd


def sma(s: pd.Series, period: int) -> pd.Series:
    return s.rolling(int(period), min_periods=int(period)).mean()


def ema(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(span=int(period), adjust=False, min_periods=int(period)).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)


def true_range(df: pd.DataFrame) -> pd.Series:
    prev = df["close"].shift(1)
    return pd.concat([df["high"] - df["low"], (df["high"] - prev).abs(), (df["low"] - prev).abs()], axis=1).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def roc(close: pd.Series, period: int = 12) -> pd.Series:
    return close.pct_change(int(period))


def bollinger(close: pd.Series, period: int = 20, width: float = 2.0):
    mid = sma(close, period)
    std = close.rolling(int(period), min_periods=int(period)).std(ddof=0)
    return mid - width * std, mid, mid + width * std


def donchian(df: pd.DataFrame, period: int = 20):
    upper = df["high"].rolling(int(period), min_periods=int(period)).max().shift(1)
    lower = df["low"].rolling(int(period), min_periods=int(period)).min().shift(1)
    return lower, upper


def relative_volume(volume: pd.Series, period: int = 20) -> pd.Series:
    return volume / volume.rolling(int(period), min_periods=int(period)).mean()
