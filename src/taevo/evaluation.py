from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import buy_and_hold, run_backtest
from .strategies import Strategy


def score(metrics: dict, complexity: int, min_trades: int = 30, max_drawdown: float = 0.35) -> float:
    trades = metrics["trades"]
    if trades < min_trades:
        return -10.0 - (min_trades - trades) / max(min_trades, 1)
    dd_penalty = max(0.0, abs(metrics["max_drawdown"]) - max_drawdown) * 8
    pf = min(max(metrics["profit_factor"], 0.0), 20.0)
    return float(metrics["sharpe"] + 0.50 * metrics["total_return"] + 0.20 * np.log1p(pf) - dd_penalty - 0.03 * complexity)


def _metrics(strategy: Strategy, df: pd.DataFrame, capital: float, fee_bps: float, slippage_bps: float, min_trades: int) -> dict:
    result = run_backtest(df, strategy.signal(df), capital, fee_bps, slippage_bps)
    m = dict(result.metrics)
    m["score"] = score(m, strategy.complexity, min_trades)
    m["strategy"] = strategy.name
    m["school"] = strategy.candidate.school
    m["complexity"] = strategy.complexity
    m["params"] = strategy.candidate.params
    return m


def robustness(strategy: Strategy, df: pd.DataFrame, capital: float, fee_bps: float, slippage_bps: float, trials: int, rng: np.random.Generator) -> dict:
    values = []
    for _ in range(trials):
        noise = rng.normal(0, 0.0002, len(df))
        perturbed = df.copy()
        perturbed["close"] = (perturbed["close"] * (1 + noise)).clip(lower=1e-9)
        values.append(run_backtest(perturbed, strategy.signal(perturbed), capital, fee_bps, slippage_bps).metrics["sharpe"])
    arr = np.asarray(values, dtype=float)
    return {"median_sharpe": float(np.median(arr)), "p10_sharpe": float(np.quantile(arr, 0.10)), "pass_rate": float(np.mean(arr > 0))}


def walk_forward(strategy: Strategy, df: pd.DataFrame, capital: float, fee_bps: float, slippage_bps: float, windows: int) -> dict:
    n = len(df)
    if windows < 2 or n < windows * 50:
        return {"windows": 0, "mean_sharpe": 0.0, "positive_fraction": 0.0}
    sharpes = []
    for i in range(windows):
        start = int(i * n / windows)
        end = int((i + 1) * n / windows)
        chunk = df.iloc[start:end]
        if len(chunk) < 50:
            continue
        sharpes.append(run_backtest(chunk, strategy.signal(chunk), capital, fee_bps, slippage_bps).metrics["sharpe"])
    arr = np.asarray(sharpes, dtype=float)
    return {"windows": int(len(arr)), "mean_sharpe": float(arr.mean()) if len(arr) else 0.0, "positive_fraction": float(np.mean(arr > 0)) if len(arr) else 0.0}


def evaluate_development(strategy: Strategy, train: pd.DataFrame, validation: pd.DataFrame, cfg: dict, rng: np.random.Generator) -> dict:
    costs = cfg["costs"]
    vcfg = cfg["validation"]
    capital = float(cfg["capital"]["initial_usd"])
    train_m = _metrics(strategy, train, capital, costs["fee_bps"], costs["slippage_bps"], 0)
    val_m = _metrics(strategy, validation, capital, costs["fee_bps"], costs["slippage_bps"], vcfg["min_trades"])
    rob = robustness(strategy, validation, capital, costs["fee_bps"], costs["slippage_bps"], vcfg["robustness_trials"], rng)
    wf = walk_forward(strategy, validation, capital, costs["fee_bps"], costs["slippage_bps"], vcfg["walk_forward_windows"])
    reasons = []
    if val_m["trades"] < vcfg["min_trades"]: reasons.append("insufficient_validation_trades")
    if abs(val_m["max_drawdown"]) > cfg["constraints"]["max_drawdown"]: reasons.append("validation_drawdown_too_high")
    if val_m["profit_factor"] < cfg["constraints"]["min_profit_factor"]: reasons.append("validation_profit_factor_too_low")
    if val_m["turnover"] > cfg["constraints"]["max_turnover"]: reasons.append("turnover_too_high")
    if rob["pass_rate"] < 0.50: reasons.append("robustness_failure")
    if wf["positive_fraction"] < 0.50: reasons.append("walk_forward_failure")
    return {"train": train_m, "validation": val_m, "robustness": rob, "walk_forward": wf, "rejected": bool(reasons), "rejection_reasons": reasons}


def evaluate_holdout(strategy: Strategy, holdout: pd.DataFrame, cfg: dict) -> dict:
    costs = cfg["costs"]
    capital = float(cfg["capital"]["initial_usd"])
    metrics = _metrics(strategy, holdout, capital, costs["fee_bps"], costs["slippage_bps"], 0)
    return {"metrics": metrics, "buy_and_hold": buy_and_hold(holdout, capital, costs["fee_bps"])}
