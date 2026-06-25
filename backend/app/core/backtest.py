import numpy as np


def apply_transaction_cost(return_pct: float, cost_bps: float) -> float:
    cost = cost_bps / 10_000.0
    return return_pct - cost


def simulate_long_only(
    predictions: np.ndarray,
    forward_returns: np.ndarray,
    hold_days: int = 5,
    transaction_cost_bps: float = 10.0,
    dates: list | None = None,
) -> dict:
    """
    Non-overlapping long-only simulation: enter on signal, hold hold_days, stay flat otherwise.
    """
    equity = [1.0]
    equity_dates: list[str] = []
    trade_returns: list[float] = []
    i = 0
    n = len(predictions)

    while i < n:
        if predictions[i] == 1:
            net_ret = apply_transaction_cost(float(forward_returns[i]), transaction_cost_bps)
            trade_returns.append(net_ret)
            equity.append(equity[-1] * (1.0 + net_ret))
            if dates is not None and i < len(dates):
                equity_dates.append(str(dates[i]))
            i += hold_days
        else:
            i += 1

    if not trade_returns:
        return {
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "trade_count": 0,
            "sharpe_ratio": 0.0,
            "equity_curve": [[str(dates[0]) if dates is not None and len(dates) else "start", 1.0]],
            "trade_returns": [],
        }

    equity_arr = np.array(equity)
    running_max = np.maximum.accumulate(equity_arr)
    drawdowns = (equity_arr - running_max) / running_max
    max_drawdown = float(drawdowns.min())
    total_return = float(equity_arr[-1] - 1.0)
    win_rate = float(np.mean(np.array(trade_returns) > 0))
    sharpe = calculate_trade_sharpe(trade_returns, hold_days)

    curve = [[str(dates[0]) if dates is not None and len(dates) else "start", 1.0]]
    for idx, d in enumerate(equity_dates):
        curve.append([d, float(equity[idx + 1])])

    return {
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "trade_count": len(trade_returns),
        "sharpe_ratio": sharpe,
        "equity_curve": curve,
        "trade_returns": trade_returns,
    }


def simulate_buy_and_hold(
    forward_returns: np.ndarray,
    hold_days: int = 5,
    transaction_cost_bps: float = 10.0,
    dates: list | None = None,
) -> dict:
    """Buy-and-hold benchmark: enter at first bar, hold through all non-overlapping windows."""
    predictions = np.ones(len(forward_returns), dtype=int)
    return simulate_long_only(
        predictions,
        forward_returns,
        hold_days=hold_days,
        transaction_cost_bps=transaction_cost_bps,
        dates=dates,
    )


def simulate_random_signal(
    forward_returns: np.ndarray,
    hold_days: int = 5,
    transaction_cost_bps: float = 10.0,
    dates: list | None = None,
    seed: int = 42,
) -> dict:
    """Random long/flat signal benchmark with fixed seed for reproducibility."""
    rng = np.random.default_rng(seed)
    predictions = rng.integers(0, 2, size=len(forward_returns))
    return simulate_long_only(
        predictions,
        forward_returns,
        hold_days=hold_days,
        transaction_cost_bps=transaction_cost_bps,
        dates=dates,
    )


def calculate_trade_sharpe(trade_returns: list[float], hold_days: int) -> float:
    if len(trade_returns) < 2:
        return 0.0
    arr = np.array(trade_returns)
    std = arr.std()
    if std == 0:
        return 0.0
    trades_per_year = 252 / hold_days
    return float((arr.mean() / std) * np.sqrt(trades_per_year))


def assign_research_grade(
    test_accuracy: float,
    overfit_gap: float,
    sharpe_ratio: float,
) -> str:
    if overfit_gap > 0.10:
        return "overfit_risk"
    if test_accuracy > 0.52 and overfit_gap <= 0.10 and sharpe_ratio > 0:
        if test_accuracy > 0.55 and overfit_gap <= 0.05 and sharpe_ratio > 0.5:
            return "promising"
        return "inconclusive"
    return "inconclusive"
