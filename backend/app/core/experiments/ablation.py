"""Hyperparameter ablation for EntangleDR."""

from dataclasses import dataclass, asdict
from itertools import product

import numpy as np

from app.core.classifiers import LogisticTrader
from app.core.dr_methods import EntangleDRMethod
from app.core.evaluation import evaluate_cv


@dataclass
class AblationConfig:
    correlation_threshold: float = 0.7
    iterations: int = 3
    rotation_scale: float = np.pi / 36


DEFAULT_ABLATION_GRID: list[AblationConfig] = [
    AblationConfig(correlation_threshold=t, iterations=i, rotation_scale=r)
    for t, i, r in product(
        [0.5, 0.6, 0.7, 0.8],
        [1, 3, 5, 10],
        [np.pi / 72, np.pi / 36, np.pi / 18],
    )
]

REDUCED_ABLATION_GRID: list[AblationConfig] = [
    AblationConfig(correlation_threshold=t, iterations=i, rotation_scale=r)
    for t, i, r in product(
        [0.5, 0.7, 0.8],
        [1, 3, 10],
        [np.pi / 72, np.pi / 36, np.pi / 18],
    )
]


def run_ablation_for_ticker(
    x: np.ndarray,
    y: np.ndarray,
    returns: np.ndarray,
    dates: list[str],
    n_components: int = 8,
    cv_splits: int = 5,
    target_days: int = 5,
    transaction_cost_bps: float = 10.0,
    seed: int = 42,
    grid: list[AblationConfig] | None = None,
) -> list[dict]:
    """Run EntangleDR ablation with Logistic classifier; return sorted results."""
    grid = grid or DEFAULT_ABLATION_GRID
    results: list[dict] = []

    for cfg in grid:
        dr = EntangleDRMethod(
            n_components=n_components,
            correlation_threshold=cfg.correlation_threshold,
            iterations=cfg.iterations,
            rotation_scale=cfg.rotation_scale,
        )
        clf = LogisticTrader()
        metrics = evaluate_cv(
            x, y, returns, dr, clf,
            cv_splits=cv_splits,
            target_days=target_days,
            transaction_cost_bps=transaction_cost_bps,
            dates=dates,
            seed=seed,
            n_trials=len(grid),
        )
        if metrics is None:
            continue
        results.append({
            **asdict(cfg),
            "rotation_scale": float(cfg.rotation_scale),
            "sharpe_ratio": metrics["sharpe_ratio"],
            "test_accuracy": metrics["test_accuracy"],
            "total_return": metrics["total_return"],
            "overfit_gap": metrics["overfit_gap"],
            "deflated_sharpe": metrics.get("deflated_sharpe"),
        })

    return sorted(results, key=lambda r: r["sharpe_ratio"], reverse=True)


def summarize_ablation_across_tickers(ticker_results: dict[str, list[dict]]) -> list[dict]:
    """Aggregate ablation configs by mean Sharpe across tickers."""
    if not ticker_results:
        return []

    config_keys = ("correlation_threshold", "iterations", "rotation_scale")
    aggregated: dict[tuple, list[float]] = {}

    for _ticker, runs in ticker_results.items():
        for run in runs:
            key = tuple(run[k] for k in config_keys)
            aggregated.setdefault(key, []).append(run["sharpe_ratio"])

    summary = []
    for key, sharpes in aggregated.items():
        summary.append({
            "correlation_threshold": key[0],
            "iterations": key[1],
            "rotation_scale": key[2],
            "mean_sharpe": float(np.mean(sharpes)),
            "std_sharpe": float(np.std(sharpes)),
            "n_tickers": len(sharpes),
        })

    return sorted(summary, key=lambda r: r["mean_sharpe"], reverse=True)
