from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .data import fetch_exchange, load_csv
from .experiment import run_experiment


def main() -> None:
    p = argparse.ArgumentParser(description="Technical Analysis Evolution Lab")
    p.add_argument("--symbol", default="BTC/USDT")
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--limit", type=int, default=5000)
    p.add_argument("--exchange", default="binance")
    p.add_argument("--csv", default=None)
    p.add_argument("--config", default="configs/default.yaml")
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    bundle = load_csv(args.csv, args.symbol, args.timeframe) if args.csv else fetch_exchange(args.symbol, args.timeframe, args.limit, args.exchange)
    report = run_experiment(bundle, cfg)
    candidates = report["best_accepted_development"]
    print(f"Run: {report['manifest']['run_id']}")
    print(f"Data: {bundle.symbol} {bundle.timeframe} rows={len(bundle.frame)}")
    if candidates:
        best = candidates[0]
        print(f"Selected candidate: {best['strategy']}")
        print(f"Validation score: {best['validation']['score']:.4f}")
        print(f"Holdout return: {best['holdout']['total_return']:.4%}")
        print(f"Holdout max drawdown: {best['holdout']['max_drawdown']:.4%}")
    else:
        print("No candidate passed development validation.")


if __name__ == "__main__":
    main()
