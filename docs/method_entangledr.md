# EntangleDR: Correlation-Aware Dimensionality Reduction

## Overview

**EntangleDR** (Entanglement Dimensionality Reduction) is a classical, correlation-aware feature compression method designed for financial signal research. It applies sequential Givens rotations to highly correlated feature pairs, inspired by entanglement structure in quantum state preparation, then selects the top-variance components.

This document specifies the algorithm used in EntangleDR Lab and anchors it against related work.

## Research Questions

1. Under purged time-series cross-validation with transaction costs, does EntangleDR preserve predictive structure better than PCA and Kernel-PCA for short-horizon directional classification?
2. How sensitive is EntangleDR to threshold τ, iteration count, and rotation scale?
3. In which correlation regimes (high vs low feature collinearity) does EntangleDR outperform linear baselines?

## Key Empirical Finding (June 2026)

At d=9 features, all 27 ablation configurations produce **identical Sharpe** per ticker across the Tech 10 universe. Reason: MA ratio features (ma_ratio_10, ma_ratio_20, ma_ratio_50) all have |corr| > 0.5, so τ=0.5 and τ=0.7 activate the same pairs. Once rotated, additional iterations and scale changes produce no change in the top-k variance selection. This confirms EntangleDR is hyperparameter-stable at low d, but its selective rotation design requires **d ≥ 30 features with heterogeneous correlation blocks** to show differential behavior. See `docs/paper_entangledr_draft.md` for full results.

## Related Work

| Approach | Relation to EntangleDR |
|----------|------------------------|
| PCA | Linear orthogonal projection maximizing variance; no pairwise correlation targeting |
| Kernel PCA | Nonlinear DR via RBF kernel; different inductive bias |
| Quantum PCA (qPCA) | True quantum algorithm for PCA; requires simulator/hardware |
| Tensor network compression | Low-rank structure; EntangleDR is a lightweight classical analogue |
| Fama-French factors | Economic latent factors; EntangleDR is unsupervised on technical features |

## Algorithm

**Input:** Feature matrix \(X \in \mathbb{R}^{T \times d}\), hyperparameters \(k\) (components), \(\tau\) (correlation threshold), \(R\) (rotation scale), \(I\) (iterations).

**Output:** Reduced matrix \(Z \in \mathbb{R}^{T \times k}\).

### Steps

1. **Standardize:** \(\tilde{X}_{t,j} = (X_{t,j} - \mu_j) / \sigma_j\)
2. **Entanglement matrix:** \(E = |\text{corr}(\tilde{X})|\)
3. **Initialize:** \(T \leftarrow I_d\)
4. **For** \(i = 1, \ldots, I\):
   - **For** each pair \((p, q)\) with \(p < q\) and \(E_{pq} > \tau\):
     - \(\theta = R \cdot E_{pq}\)
     - Apply Givens rotation \(G_{pq}(\theta)\) to \(T\): \(T \leftarrow G_{pq}(\theta) \cdot T\)
5. **Transform:** \(Y = \tilde{X} \cdot T^\top\)
6. **Select:** Keep columns with top-\(k\) variance from \(Y\)

### Default Hyperparameters

| Parameter | Symbol | Default | Ablation grid |
|-----------|--------|---------|---------------|
| Correlation threshold | τ | 0.7 | {0.5, 0.6, 0.7, 0.8} |
| Iterations | I | 3 | {1, 3, 5, 10} |
| Rotation scale | R | π/36 | {π/72, π/36, π/18} |
| Components | k | 8 | 2–20 |

### Complexity

- Correlation: \(O(d^2 T)\)
- Rotations: \(O(I \cdot d^2)\) per fit
- Suitable for small \(d\) (e.g., 9 technical features in this benchmark)

### Failure Modes

- Near-constant features → degenerate correlation (handled via ε in std)
- Very low τ → excessive rotations, noise amplification
- Very high τ → no rotations, reduces to variance selection on standardized features

## Implementation

See `backend/app/core/dr_methods.py` — class `EntangleDRMethod`.

Legacy alias: `QuantumInspiredMethod` delegates to `EntangleDRMethod` with default hyperparameters.

## Evaluation Protocol

All methods are evaluated under:

- Purged time-series CV with embargo = `target_days`
- StandardScaler before DR
- Long-only non-overlapping backtest with round-trip transaction costs
- Baselines: No-DR (raw features), buy-and-hold, random signal

See `docs/paper_entangledr_draft.md` for full experimental design.
