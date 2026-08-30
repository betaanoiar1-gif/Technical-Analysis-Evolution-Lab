from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from .data import DataBundle, split_three_way
from .evaluation import evaluate_holdout
from .learner import EvolutionLearner, write_report
from .strategies import strategy_from_record


def run_experiment(bundle: DataBundle, cfg: dict, output_dir: str | Path = "experiments/runs") -> dict:
    train, validation, holdout = split_three_way(bundle.frame, cfg["validation"]["train_ratio"], cfg["validation"]["validation_ratio"])
    learner = EvolutionLearner(cfg, seed=int(cfg["search"]["seed"]))
    report = learner.run(train, validation)
    accepted = report["best_accepted_development"]
    if accepted:
        best = accepted[0]
        final_result = evaluate_holdout(strategy_from_record(best), holdout, cfg)
        best["holdout"] = final_result["metrics"]
        best["holdout_buy_hold"] = final_result["buy_and_hold"]
    run_id = datetime.now(timezone.utc).strftime("RUN-%Y%m%dT%H%M%SZ")
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "symbol": bundle.symbol,
        "timeframe": bundle.timeframe,
        "source": bundle.source,
        "data_fingerprint": bundle.fingerprint,
        "rows": len(bundle.frame),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "holdout_rows": len(holdout),
        "config_fingerprint": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
        "holdout_policy": "blind_after_development_selection",
    }
    report["manifest"] = manifest
    write_report(report, Path(output_dir) / f"{run_id}.json")
    return report
