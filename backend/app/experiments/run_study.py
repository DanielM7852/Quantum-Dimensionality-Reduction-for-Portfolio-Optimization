"""
Batch experiment runner for EntangleDR Lab.

Usage:
    python -m app.experiments.run_study --universe sp500_sample --dry-run --limit 10
    python -m app.experiments.run_study --universe tech_10 --ablation --limit 5
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from app.core.data import download_data
from app.core.experiments.ablation import REDUCED_ABLATION_GRID, run_ablation_for_ticker, summarize_ablation_across_tickers
from app.core.experiments.universes import UNIVERSES
from app.core.features import create_features, extract_features
from app.core.targets import prepare_targets
from app.schemas import AnalyzeRequest
from app.services.pipeline import run_analysis


def _results_dir() -> Path:
    root = Path(__file__).resolve().parents[3]
    out = root / "results"
    out.mkdir(exist_ok=True)
    return out


def _format_dates(dates) -> list[str]:
    formatted = []
    for d in dates:
        if isinstance(d, (datetime, np.datetime64)):
            formatted.append(str(np.datetime_as_string(np.datetime64(d), unit="D")))
        else:
            formatted.append(str(d)[:10])
    return formatted


def run_study(
    universe: str,
    start_date: str,
    end_date: str,
    target_days: int = 5,
    n_components: int = 8,
    cv_splits: int = 5,
    transaction_cost_bps: float = 10.0,
    seed: int = 42,
    limit: int = 10,
    dry_run: bool = False,
    ablation: bool = False,
    full_ablation: bool = False,
) -> dict:
    tickers = UNIVERSES.get(universe, UNIVERSES["sp500_sample"])[:limit]

    manifest = {
        "study_id": datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
        "universe": universe,
        "tickers_requested": tickers,
        "start_date": start_date,
        "end_date": end_date,
        "target_days": target_days,
        "n_components": n_components,
        "cv_splits": cv_splits,
        "transaction_cost_bps": transaction_cost_bps,
        "seed": seed,
        "dry_run": dry_run,
        "ablation": ablation,
        "full_ablation": full_ablation,
        "results": [],
        "ablation_by_ticker": {},
        "ablation_summary": [],
        "skipped": [],
    }

    if dry_run:
        manifest["message"] = f"Dry run: would process {len(tickers)} tickers"
        return manifest

    ablation_by_ticker: dict[str, list] = {}

    for ticker in tickers:
        try:
            if ablation:
                raw = download_data(ticker, start_date, end_date)
                if len(raw) < 252:
                    manifest["skipped"].append({"ticker": ticker, "reason": "insufficient data"})
                    continue
                featured = create_features(raw)
                df_valid, y, returns, date_series = prepare_targets(featured, target_days)
                x, _ = extract_features(df_valid)
                dates = _format_dates(date_series.tolist())
                from app.core.experiments.ablation import DEFAULT_ABLATION_GRID

                grid = DEFAULT_ABLATION_GRID if full_ablation else REDUCED_ABLATION_GRID
                ablation_runs = run_ablation_for_ticker(
                    x, y, returns, dates,
                    n_components=n_components,
                    cv_splits=cv_splits,
                    target_days=target_days,
                    transaction_cost_bps=transaction_cost_bps,
                    seed=seed,
                    grid=grid,
                )
                ablation_by_ticker[ticker] = ablation_runs
                best = ablation_runs[0] if ablation_runs else {}
                manifest["results"].append({
                    "ticker": ticker,
                    "best_ablation_sharpe": best.get("sharpe_ratio"),
                    "best_config": {
                        "correlation_threshold": best.get("correlation_threshold"),
                        "iterations": best.get("iterations"),
                        "rotation_scale": best.get("rotation_scale"),
                    },
                })
            else:
                request = AnalyzeRequest(
                    ticker=ticker,
                    start_date=start_date,
                    end_date=end_date,
                    target_days=target_days,
                    n_components=n_components,
                    cv_splits=cv_splits,
                    transaction_cost_bps=transaction_cost_bps,
                    seed=seed,
                )
                result = run_analysis(request)
                best = next((c for c in result.combos if c.combo_id == result.best_combo_id), None)
                entangle = [c for c in result.combos if c.dr_method == "EntangleDR"]
                manifest["results"].append({
                    "ticker": ticker,
                    "best_combo_id": result.best_combo_id,
                    "best_sharpe": best.sharpe_ratio if best else None,
                    "entangle_mean_sharpe": float(np.mean([c.sharpe_ratio for c in entangle])) if entangle else None,
                    "pca_mean_sharpe": float(np.mean([c.sharpe_ratio for c in result.combos if c.dr_method == "PCA"])) if result.combos else None,
                    "n_combos": len(result.combos),
                })
        except Exception as exc:
            manifest["skipped"].append({"ticker": ticker, "reason": str(exc)})

    if ablation:
        manifest["ablation_by_ticker"] = ablation_by_ticker
        manifest["ablation_summary"] = summarize_ablation_across_tickers(ablation_by_ticker)

    return manifest


def save_manifest(manifest: dict) -> Path:
    out_dir = _results_dir()
    study_id = manifest["study_id"]
    json_path = out_dir / f"study_{study_id}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)

    if manifest.get("results"):
        df = pd.DataFrame(manifest["results"])
        parquet_path = out_dir / f"study_{study_id}.parquet"
        df.to_parquet(parquet_path, index=False)

    summary_path = out_dir / "manifest.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "latest_study_id": study_id,
            "universe": manifest.get("universe"),
            "n_results": len(manifest.get("results", [])),
            "n_skipped": len(manifest.get("skipped", [])),
            "dry_run": manifest.get("dry_run"),
            "ablation": manifest.get("ablation"),
        }, f, indent=2)

    return json_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EntangleDR Lab batch study runner")
    parser.add_argument("--universe", default="sp500_sample", choices=list(UNIVERSES.keys()))
    parser.add_argument("--start-date", default="2019-01-01")
    parser.add_argument("--end-date", default="2024-11-01")
    parser.add_argument("--target-days", type=int, default=5)
    parser.add_argument("--n-components", type=int, default=8)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--transaction-cost-bps", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--full-ablation", action="store_true", help="Use full 48-config grid (slow)")
    args = parser.parse_args(argv)

    manifest = run_study(
        universe=args.universe,
        start_date=args.start_date,
        end_date=args.end_date,
        target_days=args.target_days,
        n_components=args.n_components,
        cv_splits=args.cv_splits,
        transaction_cost_bps=args.transaction_cost_bps,
        seed=args.seed,
        limit=args.limit,
        dry_run=args.dry_run,
        ablation=args.ablation,
        full_ablation=args.full_ablation,
    )

    path = save_manifest(manifest)
    print(f"Study complete: {path}")
    print(f"Results: {len(manifest.get('results', []))}, Skipped: {len(manifest.get('skipped', []))}")
    if manifest.get("dry_run"):
        print(manifest.get("message", "Dry run"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
