from datetime import datetime

import numpy as np

from app.config import DISCLAIMER_TEXT, DISCLAIMER_VERSION, METRICS_CONVENTION
from app.core.classifiers import GradientBoostingTrader, LogisticTrader, RandomForestTrader
from app.core.data import download_data
from app.core.dr_methods import EntangleDRMethod, KernelPCAMethod, NoDRMethod, PCAMethod
from app.core.evaluation import compare_methods_wilcoxon, evaluate_cv
from app.core.features import create_features, extract_features
from app.core.backtest import simulate_buy_and_hold, simulate_random_signal
from app.core.stats import benjamini_hochberg
from app.core.targets import prepare_targets
from app.schemas import (
    AnalyzeMeta,
    AnalyzeRequest,
    AnalyzeResponse,
    BenchmarkResult,
    ComboResult,
    DRSummary,
    EquityPoint,
    SignificanceResult,
)


def _format_dates(dates) -> list[str]:
    formatted = []
    for d in dates:
        if isinstance(d, (datetime, np.datetime64)):
            formatted.append(str(np.datetime_as_string(np.datetime64(d), unit="D")))
        else:
            formatted.append(str(d)[:10])
    return formatted


def _compute_significance_tests(combo_metrics: list[dict]) -> list[SignificanceResult]:
    """Pairwise Wilcoxon tests on fold Sharpe for best combo per DR method."""

    def best_folds(dr_name: str) -> list[float]:
        candidates = [m for m in combo_metrics if m["dr_method"] == dr_name]
        if not candidates:
            return []
        best = max(candidates, key=lambda m: m["sharpe_ratio"])
        return best.get("fold_sharpes", [])

    entangle_folds = best_folds("EntangleDR")
    pca_folds = best_folds("PCA")
    kpca_folds = best_folds("Kernel-PCA")

    results: list[SignificanceResult] = []
    pairs = [
        ("EntangleDR", entangle_folds, "PCA", pca_folds),
        ("EntangleDR", entangle_folds, "Kernel-PCA", kpca_folds),
    ]
    for name_a, folds_a, name_b, folds_b in pairs:
        if len(folds_a) >= 3 and len(folds_b) >= 3:
            min_len = min(len(folds_a), len(folds_b))
            pval = compare_methods_wilcoxon(folds_a[:min_len], folds_b[:min_len])
            results.append(
                SignificanceResult(
                    method_a=name_a,
                    method_b=name_b,
                    metric="sharpe_ratio",
                    p_value=round(pval, 6),
                    significant=False,
                )
            )

    if results:
        pvals = [r.p_value for r in results]
        reject = benjamini_hochberg(pvals)
        for i, r in enumerate(results):
            r.significant = reject[i]

    return results


def run_analysis(request: AnalyzeRequest) -> AnalyzeResponse:
    raw = download_data(request.ticker, request.start_date, request.end_date)
    if len(raw) < 252:
        raise ValueError(f"Insufficient data: {len(raw)} rows (minimum 252 trading days required)")

    featured = create_features(raw)
    df_valid, y, returns, date_series = prepare_targets(featured, request.target_days)
    x, _ = extract_features(df_valid)
    dates = _format_dates(date_series.tolist())

    dr_methods = [
        NoDRMethod(request.n_components),
        PCAMethod(request.n_components),
        KernelPCAMethod(request.n_components),
        EntangleDRMethod(
            request.n_components,
            correlation_threshold=request.entangledr_correlation_threshold,
            iterations=request.entangledr_iterations,
            rotation_scale=request.entangledr_rotation_scale,
        ),
    ]
    classifiers = [LogisticTrader(), RandomForestTrader(), GradientBoostingTrader()]
    n_combos = len(dr_methods) * len(classifiers)

    combo_metrics: list[dict] = []
    combos: list[ComboResult] = []
    equity_curves: dict[str, list[EquityPoint]] = {}

    for dr in dr_methods:
        for clf in classifiers:
            metrics = evaluate_cv(
                x,
                y,
                returns,
                dr,
                clf,
                cv_splits=request.cv_splits,
                target_days=request.target_days,
                transaction_cost_bps=request.transaction_cost_bps,
                dates=dates,
                seed=request.seed,
                n_trials=n_combos,
            )
            if metrics is None:
                continue
            combo_metrics.append(metrics)
            combo = ComboResult(
                dr_method=metrics["dr_method"],
                classifier=metrics["classifier"],
                combo_id=metrics["combo_id"],
                train_accuracy=metrics["train_accuracy"],
                test_accuracy=metrics["test_accuracy"],
                overfit_gap=metrics["overfit_gap"],
                f1=metrics["f1"],
                sharpe_ratio=metrics["sharpe_ratio"],
                total_return=metrics["total_return"],
                max_drawdown=metrics["max_drawdown"],
                win_rate=metrics["win_rate"],
                trade_count=metrics["trade_count"],
                research_grade=metrics["research_grade"],
                accuracy_ci_lower=metrics.get("accuracy_ci_lower"),
                accuracy_ci_upper=metrics.get("accuracy_ci_upper"),
                sharpe_ci_lower=metrics.get("sharpe_ci_lower"),
                sharpe_ci_upper=metrics.get("sharpe_ci_upper"),
                return_ci_lower=metrics.get("return_ci_lower"),
                return_ci_upper=metrics.get("return_ci_upper"),
                brier_score=metrics.get("brier_score"),
                deflated_sharpe=metrics.get("deflated_sharpe"),
            )
            combos.append(combo)
            equity_curves[combo.combo_id] = [
                EquityPoint(date=pt[0], equity=pt[1]) for pt in metrics["equity_curve"]
            ]

    if not combos:
        raise ValueError("Analysis produced no valid model combinations")

    safe = [c for c in combos if c.research_grade != "overfit_risk"]
    pool = safe if safe else combos
    best = max(pool, key=lambda c: c.sharpe_ratio)

    summary_by_dr: dict[str, DRSummary] = {}
    dr_names = sorted({c.dr_method for c in combos})
    for dr_name in dr_names:
        subset = [c for c in combos if c.dr_method == dr_name]
        summary_by_dr[dr_name] = DRSummary(
            accuracy=float(np.mean([c.test_accuracy for c in subset])),
            f1=float(np.mean([c.f1 for c in subset])),
            sharpe_ratio=float(np.mean([c.sharpe_ratio for c in subset])),
            total_return=float(np.mean([c.total_return for c in subset])),
            sharpe_ci_lower=float(np.mean([c.sharpe_ci_lower or 0 for c in subset])),
            sharpe_ci_upper=float(np.mean([c.sharpe_ci_upper or 0 for c in subset])),
        )

    bh_sim = simulate_buy_and_hold(
        returns,
        hold_days=request.target_days,
        transaction_cost_bps=request.transaction_cost_bps,
        dates=dates,
    )
    rand_sim = simulate_random_signal(
        returns,
        hold_days=request.target_days,
        transaction_cost_bps=request.transaction_cost_bps,
        dates=dates,
        seed=request.seed,
    )

    benchmarks = [
        BenchmarkResult(
            name="Buy-and-Hold",
            sharpe_ratio=bh_sim["sharpe_ratio"],
            total_return=bh_sim["total_return"],
            max_drawdown=bh_sim["max_drawdown"],
            win_rate=bh_sim["win_rate"],
            trade_count=bh_sim["trade_count"],
        ),
        BenchmarkResult(
            name="Random-Signal",
            sharpe_ratio=rand_sim["sharpe_ratio"],
            total_return=rand_sim["total_return"],
            max_drawdown=rand_sim["max_drawdown"],
            win_rate=rand_sim["win_rate"],
            trade_count=rand_sim["trade_count"],
        ),
    ]

    benchmark_curves = {
        "Buy-and-Hold": [EquityPoint(date=pt[0], equity=pt[1]) for pt in bh_sim["equity_curve"]],
        "Random-Signal": [EquityPoint(date=pt[0], equity=pt[1]) for pt in rand_sim["equity_curve"]],
    }

    significance_tests = _compute_significance_tests(combo_metrics)

    meta = AnalyzeMeta(
        ticker=request.ticker,
        start_date=request.start_date,
        end_date=request.end_date,
        sample_count=len(df_valid),
        positive_class_pct=float(y.mean() * 100),
        disclaimer_version=DISCLAIMER_VERSION,
        disclaimer_text=DISCLAIMER_TEXT,
        metrics_convention=METRICS_CONVENTION,
        seed=request.seed,
    )

    return AnalyzeResponse(
        meta=meta,
        combos=combos,
        best_combo_id=best.combo_id,
        equity_curves=equity_curves,
        summary_by_dr=summary_by_dr,
        benchmarks=benchmarks,
        benchmark_curves=benchmark_curves,
        significance_tests=significance_tests,
    )
