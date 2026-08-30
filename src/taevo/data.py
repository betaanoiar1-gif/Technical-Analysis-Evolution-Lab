from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import ccxt
import pandas as pd

COLUMNS = ["open", "high", "low", "close", "volume"]


@dataclass(frozen=True)
class DataBundle:
    frame: pd.DataFrame
    symbol: str
    timeframe: str
    source: str
    fingerprint: str


def _parse_timestamp(values: pd.Series) -> pd.DatetimeIndex:
    """Parse common Unix timestamp encodings safely.

    Exchange APIs commonly return Unix milliseconds. CSV files may contain
    seconds, milliseconds, microseconds, nanoseconds, or ISO-8601 strings.
    Infer numeric units from magnitude so a millisecond timestamp is never
    silently interpreted as nanoseconds (which would produce dates near 1970).
    """
    numeric = pd.to_numeric(values, errors="coerce")
    numeric_ratio = float(numeric.notna().mean()) if len(values) else 0.0

    if numeric_ratio >= 0.99:
        clean = numeric.dropna()
        if clean.empty:
            return pd.to_datetime(values, utc=True, errors="coerce")
        magnitude = float(clean.abs().median())
        if magnitude >= 1e17:
            unit = "ns"
        elif magnitude >= 1e14:
            unit = "us"
        elif magnitude >= 1e11:
            unit = "ms"
        else:
            unit = "s"
        return pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")

    return pd.to_datetime(values, utc=True, errors="coerce")


def normalize_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    rename = {c.lower().strip(): c.lower().strip() for c in df.columns}
    df = df.rename(columns=rename)
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing OHLCV columns: {sorted(missing)}")
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df.index = _parse_timestamp(df.pop("timestamp"))
        else:
            raise ValueError("Data must use a DatetimeIndex or contain a timestamp column")
    df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    df = df[COLUMNS].apply(pd.to_numeric, errors="coerce")
    valid = df.notna().all(axis=1) & df.index.notna()
    df = df.loc[valid]
    df = df[~df.index.duplicated(keep="last")].sort_index()
    if len(df) < 100:
        raise ValueError("At least 100 valid OHLCV rows are required")
    if (df[["high", "low", "close"]] <= 0).any().any():
        raise ValueError("Prices must be positive")
    if (df["high"] < df[["open", "close"]].max(axis=1)).any():
        raise ValueError("Invalid OHLC: high below open/close")
    if (df["low"] > df[["open", "close"]].min(axis=1)).any():
        raise ValueError("Invalid OHLC: low above open/close")
    return df


def fingerprint(frame: pd.DataFrame) -> str:
    payload = pd.util.hash_pandas_object(frame, index=True).values.tobytes()
    return hashlib.sha256(payload).hexdigest()


def load_csv(path: str | Path, symbol: str = "CSV", timeframe: str = "unknown") -> DataBundle:
    df = normalize_ohlcv(pd.read_csv(path))
    return DataBundle(df, symbol, timeframe, f"csv:{Path(path).name}", fingerprint(df))


def fetch_exchange(symbol: str = "BTC/USDT", timeframe: str = "1h", limit: int = 5000, exchange_id: str = "binance") -> DataBundle:
    exchange_cls = getattr(ccxt, exchange_id, None)
    if exchange_cls is None:
        raise ValueError(f"Unsupported ccxt exchange: {exchange_id}")
    exchange = exchange_cls({"enableRateLimit": True})
    try:
        rows = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=int(limit))
    finally:
        try:
            exchange.close()
        except Exception:
            pass
    if not rows:
        raise RuntimeError("Exchange returned no OHLCV rows")
    frame = pd.DataFrame(rows, columns=["timestamp", *COLUMNS])
    df = normalize_ohlcv(frame)
    return DataBundle(df, symbol, timeframe, f"ccxt:{exchange_id}", fingerprint(df))


def split_three_way(frame: pd.DataFrame, train_ratio: float, validation_ratio: float):
    if not 0 < train_ratio < 1 or not 0 < validation_ratio < 1 or train_ratio + validation_ratio >= 1:
        raise ValueError("train_ratio and validation_ratio must leave a positive holdout")
    n = len(frame)
    a = int(n * train_ratio)
    b = int(n * (train_ratio + validation_ratio))
    return frame.iloc[:a].copy(), frame.iloc[a:b].copy(), frame.iloc[b:].copy()
