from __future__ import annotations

from html import escape
from pathlib import Path


def render_html(report: dict, path: str | Path) -> None:
    best = report.get("best_development", [{}])[0]
    manifest = report.get("manifest", {})
    hold = best.get("holdout", {})
    base = best.get("holdout_buy_hold", {})
    rows = {
        "Strategy": best.get("strategy", "n/a"),
        "School": best.get("school", "n/a"),
        "Validation score": best.get("validation", {}).get("score", "n/a"),
        "Validation Sharpe": best.get("validation", {}).get("sharpe", "n/a"),
        "Validation return": best.get("validation", {}).get("total_return", "n/a"),
        "Holdout return": hold.get("total_return", "n/a"),
        "Holdout Sharpe": hold.get("sharpe", "n/a"),
        "Holdout max drawdown": hold.get("max_drawdown", "n/a"),
        "Holdout trades": hold.get("trades", "n/a"),
        "Holdout buy & hold": base.get("total_return", "n/a"),
        "Rejected": best.get("rejected", "n/a"),
        "Reasons": ", ".join(best.get("rejection_reasons", [])),
    }
    body = "\n".join(f"<tr><th>{escape(str(k))}</th><td>{escape(str(v))}</td></tr>" for k, v in rows.items())
    html = f"""<!doctype html><html><head><meta charset='utf-8'><title>TA Evolution Lab Report</title>
    <style>body{{font-family:system-ui;max-width:900px;margin:40px auto;padding:0 20px}}table{{border-collapse:collapse;width:100%}}th,td{{text-align:left;border-bottom:1px solid #ddd;padding:10px}}code{{background:#f3f3f3;padding:2px 4px}}</style></head>
    <body><h1>Technical Analysis Evolution Lab</h1><p>Run <code>{escape(str(manifest.get('run_id','unknown')))}</code> | Data fingerprint <code>{escape(str(manifest.get('data_fingerprint','unknown'))[:16])}</code></p>
    <table>{body}</table><h2>Research integrity</h2><p>The candidate is selected from development/validation evidence. The holdout is reported separately and must not be used for parameter tuning.</p></body></html>"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(html, encoding="utf-8")
