"""Statistical helpers for research evaluation."""

import numpy as np
from scipy import stats


def bootstrap_ci(
    values: list[float] | np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return (point_estimate, lower, upper) via percentile bootstrap."""
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return 0.0, 0.0, 0.0
    point = float(np.mean(arr))
    if len(arr) < 2:
        return point, point, point

    rng = np.random.default_rng(seed)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boot_means.append(float(np.mean(sample)))

    alpha = (1.0 - confidence) / 2.0
    lower = float(np.percentile(boot_means, 100 * alpha))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha)))
    return point, lower, upper


def wilcoxon_signed_rank_pvalue(
    sample_a: list[float] | np.ndarray,
    sample_b: list[float] | np.ndarray,
) -> float:
    """Paired Wilcoxon signed-rank test (two-sided). Returns 1.0 if insufficient data."""
    a = np.asarray(sample_a, dtype=float)
    b = np.asarray(sample_b, dtype=float)
    if len(a) < 3 or len(b) < 3 or len(a) != len(b):
        return 1.0
    diff = a - b
    if np.allclose(diff, 0):
        return 1.0
    try:
        _, pvalue = stats.wilcoxon(diff, alternative="two-sided")
        return float(pvalue)
    except ValueError:
        return 1.0


def benjamini_hochberg(pvalues: list[float], alpha: float = 0.05) -> list[bool]:
    """Return rejection mask for BH FDR control at level alpha."""
    m = len(pvalues)
    if m == 0:
        return []
    indexed = sorted(enumerate(pvalues), key=lambda x: x[1])
    reject = [False] * m
    max_k = -1
    for rank, (orig_idx, p) in enumerate(indexed, start=1):
        if p <= (rank / m) * alpha:
            max_k = rank
    if max_k >= 0:
        for rank, (orig_idx, p) in enumerate(indexed, start=1):
            if rank <= max_k:
                reject[orig_idx] = True
    return reject


_EULER_GAMMA = 0.5772156649015329


def deflated_sharpe_ratio(
    sharpe: float,
    n_trials: int,
    n_observations: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """
    Bailey & López de Prado deflated Sharpe ratio (simplified).
    Returns probability that observed Sharpe exceeds expected max under null.
    """
    if n_observations < 2 or n_trials < 1:
        return 0.0

    sr = sharpe
    e_max_sr = np.sqrt(2 * np.log(max(n_trials, 1))) * (1 - _EULER_GAMMA) + (
        _EULER_GAMMA / np.sqrt(2 * np.log(max(n_trials, 1)))
    )
    variance_sr = (1 - skew * sr + ((kurtosis - 1) / 4) * sr**2) / max(n_observations - 1, 1)
    if variance_sr <= 0:
        return 0.0
    z = (sr - e_max_sr) / np.sqrt(variance_sr)
    return float(stats.norm.cdf(z))
