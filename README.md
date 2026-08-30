# Technical Analysis Evolution Lab

A research-grade laboratory for evolving crypto trading strategies from technical-analysis schools using reproducible experiments, realistic backtests, walk-forward validation, falsification, and an explicit learning loop.

> **Research only.** This project does not promise profitability and is not a live-trading system. Never deploy a strategy with real money without independent validation, risk controls, and operational testing.

## What this project does

1. Loads OHLCV data from CSV or exchange APIs.
2. Computes indicators without look-ahead.
3. Generates hypotheses from multiple technical-analysis schools.
4. Backtests candidates with fees and slippage.
5. Splits development/validation/holdout data.
6. Runs walk-forward validation and robustness perturbations.
7. Detects common failure modes: overfitting, weak sample size, excessive turnover, unstable parameters, and poor out-of-sample behavior.
8. Learns from experiment outcomes by changing the next candidate budget and search regions rather than memorizing a single winner.
9. Writes an auditable experiment ledger and JSON reports.

## Technical-analysis schools in V0.1

- Trend following: SMA/EMA crossovers and slope filters
- Momentum: RSI and rate-of-change filters
- Volatility: Bollinger bands and ATR-based position/risk filters
- Breakout: Donchian-style channel breaks
- Volume: volume confirmation and relative-volume filters
- Price action: candle body/range and breakout confirmation

The architecture is deliberately modular so new schools can be added without changing the backtest engine.

## Repository layout

```text
Technical-Analysis-Evolution-Lab/
├── colab/
│   └── Technical_Analysis_Evolution_Lab.ipynb
├── src/taevo/
│   ├── data.py
│   ├── indicators.py
│   ├── schools.py
│   ├── strategies.py
│   ├── backtest.py
│   ├── evaluation.py
│   ├── learner.py
│   ├── experiment.py
│   └── cli.py
├── configs/default.yaml
├── tests/test_core.py
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Google Colab

Open `colab/Technical_Analysis_Evolution_Lab.ipynb` in Colab, run all cells, choose an asset/timeframe, and inspect the research report. The notebook clones this repository and uses the same Python package as local runs.

## Reproducibility contract

Every run records configuration, random seed, data fingerprint, candidate parameters, metrics, rejection reasons, and the final decision. The final holdout must never be used to tune candidates.

## Suggested first experiment

Start with BTC/USDT spot, hourly candles, no leverage, realistic costs, and a broad but controlled candidate set. Establish a buy-and-hold baseline before evolving strategies.

## Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pytest -q
python -m taevo.cli --symbol BTC/USDT --timeframe 1h --limit 5000
```

The first release is intentionally research-first. Live execution, exchange credentials, and real-money deployment are out of scope for V0.1.
