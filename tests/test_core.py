import numpy as np
import pandas as pd

from taevo.backtest import run_backtest
from taevo.data import normalize_ohlcv, split_three_way
from taevo.schools import candidate_library
from taevo.strategies import Strategy


def make_df(n=600):
    rng = np.random.default_rng(7)
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    ret = rng.normal(0.0002, 0.01, n)
    close = 100 * np.exp(np.cumsum(ret))
    open_ = close * (1 + rng.normal(0, 0.001, n))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.003, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.003, n))
    volume = rng.uniform(100, 1000, n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


def test_normalize_and_split():
    df = make_df()
    norm = normalize_ohlcv(df.reset_index(names="timestamp"))
    a, b, c = split_three_way(norm, 0.55, 0.20)
    assert len(a) + len(b) + len(c) == len(norm)
    assert a.index[-1] < b.index[0] < c.index[0]


def test_backtest_is_deterministic_and_applies_costs():
    df = make_df()
    signal = pd.Series(1, index=df.index)
    a = run_backtest(df, signal, fee_bps=10, slippage_bps=5)
    b = run_backtest(df, signal, fee_bps=10, slippage_bps=5)
    assert np.isclose(a.metrics["final_equity"], b.metrics["final_equity"])
    assert a.metrics["cost_drag"] > 0


def test_all_school_candidates_build_signal():
    df = make_df()
    for candidate in candidate_library():
        strategy = Strategy(candidate)
        signal = strategy.signal(df)
        assert signal.index.equals(df.index)
        assert set(signal.unique()).issubset({0, 1})
