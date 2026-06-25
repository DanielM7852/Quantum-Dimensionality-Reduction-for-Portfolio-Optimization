import copy

import numpy as np
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from app.core.backtest import assign_research_grade, simulate_long_only
from app.core.stats import bootstrap_ci, deflated_sharpe_ratio


def purged_time_series_splits(n_samples: int, n_splits: int, embargo: int):
    """Time series splits with embargo gap between train and test to reduce label leakage."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_idx, test_idx in tscv.split(np.arange(n_samples)):
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        train_end = train_idx[-1]
        test_start = test_idx[0]
        if test_start - train_end <= embargo:
            gap = embargo - (test_start - train_end) + 1
            test_idx = test_idx[test_idx >= test_start + gap]
            if len(test_idx) < 20:
                continue
        if len(train_idx) < 100 or len(test_idx) < 20:
            continue
        yield train_idx, test_idx


def evaluate_cv(
    x: np.ndarray,
    y: np.ndarray,
    returns: np.ndarray,
    dr_method,
    trading_model,
    cv_splits: int = 5,
    target_days: int = 5,
    transaction_cost_bps: float = 10.0,
    dates: list | None = None,
    seed: int = 42,
    n_trials: int = 1,
) -> dict | None:
    train_accs, test_accs, test_f1s = [], [], []
    test_sharpes, test_returns, test_drawdowns = [], [], []
    test_win_rates, test_trade_counts = [], []
    fold_brier_scores: list[float] = []
    all_predictions: list[np.ndarray] = []
    all_test_returns: list[np.ndarray] = []
    all_test_dates: list = []
    all_fold_trade_returns: list[list[float]] = []

    for train_idx, test_idx in purged_time_series_splits(len(x), cv_splits, embargo=target_days):
        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        y_train = np.nan_to_num(y_train, nan=0).astype(int)
        y_test = np.nan_to_num(y_test, nan=0).astype(int)
        ret_test = np.nan_to_num(returns[test_idx], nan=0)

        scaler = StandardScaler()
        x_train_sc = scaler.fit_transform(x_train)
        x_test_sc = scaler.transform(x_test)

        dr = copy.deepcopy(dr_method)
        model = copy.deepcopy(trading_model)

        try:
            x_train_dr = dr.fit_transform(x_train_sc, y_train)
            x_test_dr = dr.transform(x_test_sc)
        except (ValueError, np.linalg.LinAlgError):
            continue

        model.train(x_train_dr, y_train)
        y_train_pred = model.predict(x_train_dr)
        y_test_pred = model.predict(x_test_dr)

        train_accs.append(accuracy_score(y_train, y_train_pred))
        test_accs.append(accuracy_score(y_test, y_test_pred))
        test_f1s.append(f1_score(y_test, y_test_pred, zero_division=0))

        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(x_test_dr)
                fold_brier_scores.append(brier_score_loss(y_test, proba))
            except (ValueError, AttributeError):
                pass

        fold_dates = None
        if dates is not None:
            fold_dates = [dates[i] for i in test_idx]

        sim = simulate_long_only(
            y_test_pred,
            ret_test,
            hold_days=target_days,
            transaction_cost_bps=transaction_cost_bps,
            dates=fold_dates,
        )
        test_sharpes.append(sim["sharpe_ratio"])
        test_returns.append(sim["total_return"])
        test_drawdowns.append(sim["max_drawdown"])
        test_win_rates.append(sim["win_rate"])
        test_trade_counts.append(sim["trade_count"])
        all_fold_trade_returns.append(sim["trade_returns"])

        all_predictions.append(y_test_pred)
        all_test_returns.append(ret_test)
        if fold_dates:
            all_test_dates.extend(fold_dates)

    if not test_accs:
        return None

    mean_acc = float(np.mean(test_accs))
    mean_train = float(np.mean(train_accs))
    mean_sharpe = float(np.mean(test_sharpes))
    overfit_gap = mean_train - mean_acc

    combined_pred = np.concatenate(all_predictions) if all_predictions else np.array([])
    combined_ret = np.concatenate(all_test_returns) if all_test_returns else np.array([])
    full_sim = simulate_long_only(
        combined_pred,
        combined_ret,
        hold_days=target_days,
        transaction_cost_bps=transaction_cost_bps,
        dates=all_test_dates if all_test_dates else None,
    )

    flat_trades = [r for fold in all_fold_trade_returns for r in fold]
    acc_point, acc_lo, acc_hi = bootstrap_ci(test_accs, seed=seed)
    sharpe_point, sharpe_lo, sharpe_hi = bootstrap_ci(test_sharpes, seed=seed + 1)
    ret_point, ret_lo, ret_hi = bootstrap_ci(test_returns, seed=seed + 2)

    trade_arr = np.array(flat_trades) if flat_trades else np.array([0.0])
    skew = float(stats_skew(trade_arr))
    kurt = float(stats_kurtosis(trade_arr, fisher=False))
    dsr = deflated_sharpe_ratio(
        mean_sharpe,
        n_trials=n_trials,
        n_observations=max(len(flat_trades), 2),
        skew=skew,
        kurtosis=kurt,
    )

    return {
        "dr_method": dr_method.name,
        "classifier": trading_model.name,
        "combo_id": f"{dr_method.name}__{trading_model.name}",
        "train_accuracy": mean_train,
        "test_accuracy": mean_acc,
        "overfit_gap": overfit_gap,
        "f1": float(np.mean(test_f1s)),
        "sharpe_ratio": mean_sharpe,
        "total_return": float(np.mean(test_returns)),
        "max_drawdown": float(np.mean(test_drawdowns)),
        "win_rate": float(np.mean(test_win_rates)),
        "trade_count": int(np.mean(test_trade_counts)),
        "research_grade": assign_research_grade(mean_acc, overfit_gap, mean_sharpe),
        "equity_curve": full_sim["equity_curve"],
        "accuracy_ci_lower": acc_lo,
        "accuracy_ci_upper": acc_hi,
        "sharpe_ci_lower": sharpe_lo,
        "sharpe_ci_upper": sharpe_hi,
        "return_ci_lower": ret_lo,
        "return_ci_upper": ret_hi,
        "brier_score": float(np.mean(fold_brier_scores)) if fold_brier_scores else None,
        "deflated_sharpe": dsr,
        "fold_accuracies": test_accs,
        "fold_sharpes": test_sharpes,
    }


def stats_skew(arr: np.ndarray) -> float:
    if len(arr) < 3:
        return 0.0
    m = arr.mean()
    s = arr.std()
    if s == 0:
        return 0.0
    return float(np.mean(((arr - m) / s) ** 3))


def stats_kurtosis(arr: np.ndarray, fisher: bool = False) -> float:
    if len(arr) < 4:
        return 3.0
    m = arr.mean()
    s = arr.std()
    if s == 0:
        return 3.0
    k = float(np.mean(((arr - m) / s) ** 4))
    return k + 3.0 if fisher else k


def compare_methods_wilcoxon(
    method_a_folds: list[float],
    method_b_folds: list[float],
) -> float:
    """Paired Wilcoxon p-value between two methods' fold metrics."""
    from app.core.stats import wilcoxon_signed_rank_pvalue

    return wilcoxon_signed_rank_pvalue(method_a_folds, method_b_folds)
