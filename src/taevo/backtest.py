from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestResult:
    equity: pd.Series
    strategy_returns: pd.Series
    position: pd.Series
    metrics: dict


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    return float((equity.div(peak.replace(0, np.nan)) - 1).min())


def run_backtest(
    df: pd.DataFrame,
    signal: pd.Series,
    initial_capital: float = 1000.0,
    fee_bps: float = 10.0,
    slippage_bps: float = 5.0,
) -> BacktestResult:
    if len(df) == 0:
        raise ValueError("Cannot backtest empty data")

    close = df.close.astype(float)
    pos = signal.reindex(df.index).fillna(0).astype(float).clip(-1, 1)
    executed = pos.shift(1).fillna(0)
    asset_ret = close.pct_change().fillna(0)

    # Turnover is absolute change in signed exposure. A direct flip
    # from +1 to -1 therefore costs 2 units of turnover.
    turnover = pos.diff().abs().fillna(pos.abs())
    cost = turnover * ((fee_bps + slippage_bps) / 10000.0)

    # Signed positions support Long (+1), Flat (0), and Short (-1).
    strategy_ret = executed * asset_ret - cost
    equity = initial_capital * (1 + strategy_ret).cumprod()

    pnl = equity.diff().fillna(0)
    gains = float(pnl.clip(lower=0).sum())
    losses = float(-pnl.clip(upper=0).sum())
    trades = int((turnover > 0).sum())

    years = max(
        (df.index[-1] - df.index[0]).total_seconds()
        / (365.25 * 86400),
        1 / 365.25,
    )

    final_ratio = equity.iloc[-1] / initial_capital
    cagr = float(final_ratio ** (1 / years) - 1)

    vol = float(
        strategy_ret.std(ddof=0) * np.sqrt(24 * 365)
    )

    sharpe = (
        float(
            strategy_ret.mean()
            / strategy_ret.std(ddof=0)
            * np.sqrt(24 * 365)
        )
        if strategy_ret.std(ddof=0) > 0
        else 0.0
    )

    metrics = {
        "total_return": float(final_ratio - 1),
        "cagr": cagr,
        "max_drawdown": max_drawdown(equity),
        "sharpe": sharpe,
        "annualized_volatility": vol,
        "profit_factor": (
            gains / losses
            if losses > 0
            else (float("inf") if gains > 0 else 0.0)
        ),
        "final_equity": float(equity.iloc[-1]),
        "trades": trades,
        "turnover": float(turnover.sum()),
        "exposure": float(executed.abs().mean()),
        "long_exposure": float((executed > 0).mean()),
        "short_exposure": float((executed < 0).mean()),
        "cost_drag": float(cost.sum()),
    }

    return BacktestResult(
        equity,
        strategy_ret,
        executed,
        metrics,
    )


def buy_and_hold(
    df: pd.DataFrame,
    initial_capital: float = 1000.0,
    fee_bps: float = 10.0,
) -> dict:
    ret = df.close.pct_change().fillna(0)
    turnover_cost = pd.Series(0.0, index=df.index)
    turnover_cost.iloc[0] = fee_bps / 10000.0
    equity = initial_capital * (1 + ret - turnover_cost).cumprod()
    return {
        "total_return": float(equity.iloc[-1] / initial_capital - 1),
        "final_equity": float(equity.iloc[-1]),
    }
