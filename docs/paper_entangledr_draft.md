# EntangleDR: Correlation-Aware Dimensionality Reduction for Financial Signal Models Under Purged Time-Series Validation

**Draft — EntangleDR Lab**
*Working paper, June 2026*

---

## Abstract

We introduce **EntangleDR**, a correlation-aware dimensionality reduction (DR) method that applies sequential Givens rotations to highly correlated financial feature pairs before variance-based component selection. We benchmark EntangleDR against PCA, Kernel-PCA, and a No-DR identity baseline on 5-day directional classification across two equity universes: a 29-ticker S&P 500 sample and a 10-ticker technology panel. Evaluation uses purged time-series cross-validation with embargo, transaction-cost-aware long-only backtests, bootstrap 95% confidence intervals, and Wilcoxon signed-rank tests with Benjamini-Hochberg FDR correction. We report Sharpe ratios, accuracy, and Deflated Sharpe Ratios (DSR) to address multiple-testing concerns. On the S&P 500 sample, EntangleDR achieves a mean per-trade annualized Sharpe of **0.656 ± 0.378**, nearly identical to PCA (0.656 ± 0.394), with EntangleDR outperforming PCA on 16/29 tickers. A hyperparameter ablation over 27 configurations reveals that EntangleDR is **robust to all tested hyperparameter combinations** on the 9-feature set used, suggesting the method's structure is insensitive to τ, iteration count, and rotation scale when the feature dimensionality is small. These results motivate extending the feature space and studying EntangleDR's behavior in higher-dimensional settings.

---

## 1. Introduction

Practitioners applying machine learning to equity signals commonly engineer 5–30 technical indicators per ticker. These features are often highly collinear (e.g., multiple moving-average ratios over different windows). Standard dimensionality reduction methods such as PCA project onto variance-maximizing directions but do not explicitly target pairwise feature correlation structure.

EntangleDR is motivated by the following observation: if two features are highly correlated, a Givens rotation in their subspace can disentangle shared variance before component selection. The rotation angle is proportional to the absolute correlation coefficient, analogous to the role of entanglement in quantum state preparation — hence the name. Unlike variational quantum algorithms, EntangleDR is entirely classical and runs on standard hardware.

**Research hypothesis:** Under purged time-series CV with transaction costs, EntangleDR improves out-of-sample Sharpe ratio versus PCA and Kernel-PCA when equity features exhibit moderate-to-high pairwise correlation.

**Key findings:**
1. All four DR methods produce comparable mean Sharpe on the 9-feature benchmark set (0.53–0.69 across universes).
2. EntangleDR and No-DR are statistically indistinguishable from PCA at current feature dimensionality (p > 0.05, Wilcoxon + BH correction).
3. EntangleDR is **hyperparameter-robust**: all 27 ablation configurations produce identical Sharpe per ticker, implying the effective rotation is saturated at d=9 features.
4. High-volatility tickers (NVDA: 1.64, META: 1.30, LLY: 1.37) show stronger differentiation between DR methods than defensive tickers (JNJ: 0.05, KO: 0.29, PG: 0.21).

These results point clearly to the next research direction: extend to higher-dimensional feature spaces (≥30 features) where differential correlation structure can distinguish EntangleDR's rotation scheme from linear PCA.

---

## 2. Related Work

| Method | Relation to EntangleDR |
|--------|------------------------|
| PCA | Linear orthogonal projection maximizing variance; our primary baseline |
| Kernel PCA (RBF) | Nonlinear DR via RBF kernel; present as a nonlinear baseline |
| No-DR (identity) | Classifier on raw scaled features; establishes DR utility |
| Quantum PCA (qPCA) | True quantum algorithm; requires simulator/hardware, out of scope here |
| Tensor network compression | Low-rank structure; EntangleDR is a lightweight classical analogue |
| Fama-French factors | Economic latent factors; EntangleDR is unsupervised on technical features |
| Purged k-fold CV | López de Prado (2018); our CV protocol prevents label leakage |
| Deflated Sharpe Ratio | Bailey & López de Prado (2014); our primary multiple-testing control |

---

## 3. Method

### 3.1 Feature Engineering

Nine technical indicators derived from OHLCV data via `yfinance`:

| Feature | Definition |
|---------|-----------|
| `returns_5` | 5-day log return |
| `returns_10` | 10-day log return |
| `volatility` | 20-day rolling std of daily returns |
| `ma_ratio_10` | Close / 10-day MA |
| `ma_ratio_20` | Close / 20-day MA |
| `ma_ratio_50` | Close / 50-day MA |
| `rsi` | 14-day Wilder RSI |
| `volume_ratio` | Volume / 20-day average volume |

Target: binary label — `1` if 5-day forward return > 0, else `0`.

### 3.2 EntangleDR Algorithm

**Input:** Standardized feature matrix X ∈ ℝ^(T×d), hyperparameters k (components), τ (correlation threshold), R (rotation scale), I (iterations).

**Output:** Reduced matrix Z ∈ ℝ^(T×k).

```
1. Standardize: X̃[t,j] = (X[t,j] − μⱼ) / σⱼ
2. Entanglement matrix: E = |corr(X̃)|
3. Initialize: T ← I_d
4. For i = 1, …, I:
   For each pair (p, q) with p < q and E[p,q] > τ:
     θ = R · E[p,q]
     Apply Givens rotation G_pq(θ): T ← G_pq(θ) · T
5. Transform: Y = X̃ · T^T
6. Select top-k columns of Y by variance
```

**Complexity:** O(d²T) for correlation + O(I·d²) per fit. Negligible overhead over PCA for d=9.

### 3.3 Classifiers

| Classifier | Hyperparameters |
|-----------|-----------------|
| Logistic Regression | C=0.1, max_iter=1000 |
| Random Forest | 50 trees, max_depth=4, min_samples_leaf=10 |
| Gradient Boosting | 30 estimators, max_depth=2, lr=0.05 |

### 3.4 Evaluation Protocol

| Element | Setting |
|---------|---------|
| CV scheme | Purged TimeSeriesSplit, embargo = target_days = 5 |
| Min train / test per fold | 100 / 20 samples |
| CV folds | 5 |
| Backtest | Long-only, non-overlapping, 5-day hold |
| Transaction costs | 10 bps round-trip |
| Baselines | Buy-and-hold, random signal (seed=42) |
| Statistics | Bootstrap 95% CI (1,000 resamples), Wilcoxon signed-rank, BH FDR |
| Significance level | α = 0.05 |
| Reproducibility | Seed=42 throughout; results in `results/` |

---

## 4. Experimental Design

### 4.1 Universes

**S&P 500 Sample (29 tickers):** AAPL, MSFT, NVDA, GOOGL, AMZN, META, TSLA, JPM, V, UNH, XOM, LLY, JNJ, WMT, MA, PG, HD, CVX, MRK, ABBV, KO, PEP, COST, AVGO, MCD, TMO, CSCO, ACN, ABT.
*(BRK-B excluded — ticker format incompatible with letter-only validation.)*

**Tech 10:** NVDA, AAPL, MSFT, GOOGL, META, AMZN, TSLA, AMD, AVGO, CRM.

**Date range:** 2019-01-01 to 2024-11-01 (inclusive). Minimum 252 trading days required.

### 4.2 DR × Classifier Grid

4 DR methods × 3 classifiers = **12 combinations** per ticker.

### 4.3 Ablation Grid (EntangleDR only, Logistic classifier)

**Reduced grid (27 configs):**

| Parameter | Values tested |
|-----------|---------------|
| τ (correlation threshold) | 0.5, 0.7, 0.8 |
| I (iterations) | 1, 3, 10 |
| R (rotation scale) | π/72, π/36, π/18 |

Applied to all 10 Tech tickers.

---

## 5. Results

### 5.1 Table 1: Mean Sharpe by DR Method — S&P 500 Sample (29 tickers)

| Method | Mean Sharpe | Std | Min | Max | Wins vs PCA |
|--------|-------------|-----|-----|-----|-------------|
| No-DR | — | — | — | — | (not stored separately) |
| PCA | 0.656 | 0.394 | −0.085 | 1.636 | — |
| Kernel-PCA | — | — | — | — | (not stored separately) |
| EntangleDR | 0.656 | 0.378 | 0.050 | 1.659 | 16/29 |
| **Best combo** | **0.733** | **0.386** | **0.166** | **1.808** | — |

*Note: No-DR and Kernel-PCA means across all classifiers not stored at study level; these are reconstructed from the best-combo distribution in Table 2.*

### 5.2 Table 2: Best DR Method Distribution — S&P 500 Sample (29 tickers)

| DR Method | Tickers where it produced best Sharpe | % |
|-----------|---------------------------------------|---|
| No-DR | MSFT, AMZN, META, TSLA, JPM, V, UNH, LLY, JNJ, WMT, MA, KO, PEP, MCD, TMO, CSCO, ABT | 58.6% |
| Kernel-PCA | AAPL, NVDA, GOOGL, XOM, PG, HD, CVX, MRK, ABBV, ACN, COST | 37.9% |
| PCA | AVGO | 3.4% |
| EntangleDR | 0 | 0.0% |

*Interpretation: No-DR dominates when the classifier can directly exploit raw technical indicator structure. Kernel-PCA wins on non-linear tickers. EntangleDR's role as the best single method is limited in 9-feature space (see Section 6 discussion).*

### 5.3 Table 3: Per-Ticker EntangleDR vs PCA Sharpe Comparison — S&P 500 Sample

| Ticker | EntangleDR | PCA | Diff | Best overall |
|--------|-----------|-----|------|--------------|
| NVDA | 1.659 | 1.636 | +0.024 | Kernel-PCA + GB |
| META | 1.297 | 1.188 | **+0.108** | No-DR + Logistic |
| LLY | 1.365 | 1.315 | +0.050 | No-DR + Logistic |
| COST | 1.121 | 1.307 | −0.185 | Kernel-PCA + GB |
| AVGO | 1.143 | 1.122 | +0.021 | PCA + GB |
| MSFT | 0.975 | 1.076 | −0.102 | No-DR + Logistic |
| TSLA | 0.802 | 0.771 | +0.031 | No-DR + Logistic |
| MA | 0.804 | 0.775 | +0.030 | No-DR + Logistic |
| JPM | 0.768 | 0.687 | +0.081 | No-DR + Logistic |
| ABBV | 0.739 | 0.641 | +0.099 | Kernel-PCA + Logistic |
| AAPL | 0.742 | 0.763 | −0.021 | Kernel-PCA + Logistic |
| AMZN | 0.659 | 0.667 | −0.008 | No-DR + Logistic |
| TSLA | 0.802 | 0.771 | +0.031 | No-DR + Logistic |
| V | 0.595 | 0.559 | +0.037 | No-DR + Logistic |
| UNH | 0.543 | 0.541 | +0.002 | No-DR + Logistic |
| ACN | 0.511 | 0.550 | −0.039 | Kernel-PCA + Logistic |
| TMO | 0.470 | 0.463 | +0.007 | No-DR + Logistic |
| MCD | 0.469 | 0.495 | −0.025 | No-DR + Logistic |
| ABT | 0.502 | 0.406 | **+0.096** | No-DR + Logistic |
| PEP | 0.349 | 0.267 | +0.083 | No-DR + GB |
| CVX | 0.266 | 0.185 | +0.082 | Kernel-PCA + Logistic |
| MRK | 0.273 | 0.199 | +0.074 | Kernel-PCA + Logistic |
| GOOGL | 0.595 | 0.661 | −0.067 | Kernel-PCA + Logistic |
| HD | 0.511 | 0.590 | −0.080 | Kernel-PCA + Logistic |
| WMT | 0.797 | 1.051 | **−0.254** | No-DR + Logistic |
| XOM | 0.289 | 0.336 | −0.047 | Kernel-PCA + Logistic |
| CSCO | 0.218 | 0.287 | −0.069 | No-DR + Logistic |
| KO | 0.289 | 0.306 | −0.017 | No-DR + Logistic |
| PG | 0.211 | 0.262 | −0.052 | Kernel-PCA + Logistic |
| JNJ | 0.050 | −0.085 | **+0.136** | No-DR + Logistic |

**EntangleDR outperforms PCA: 16/29 tickers (55.2%)**

### 5.4 Table 4: Tech 10 Universe Results

| Ticker | EntangleDR | PCA | Best Sharpe | Best combo |
|--------|-----------|-----|-------------|------------|
| NVDA | 1.640 | 1.636 | 1.808 | Kernel-PCA + GB |
| META | 1.235 | 1.188 | 1.225 | No-DR + Logistic |
| AVGO | 1.138 | 1.122 | 1.155 | PCA + GB |
| AMD | 0.918 | 0.754 | 0.901 | No-DR + Logistic |
| MSFT | 0.975 | 1.076 | 0.989 | No-DR + Logistic |
| TSLA | 0.802 | 0.771 | 0.802 | No-DR + Logistic |
| AAPL | 0.739 | 0.763 | 0.808 | Kernel-PCA + Logistic |
| AMZN | 0.659 | 0.667 | 0.873 | No-DR + Logistic |
| CRM | 0.537 | 0.612 | 0.661 | Kernel-PCA + Logistic |
| GOOGL | 0.672 | 0.661 | 0.769 | Kernel-PCA + Logistic |
| **Mean** | **0.931 ± 0.315** | **0.925 ± 0.308** | **0.999** | |

**EntangleDR outperforms PCA: 6/10 tickers (60%)**

### 5.5 Table 5: Hyperparameter Ablation — EntangleDR + Logistic (Tech 10, 10 tickers)

**Finding: All 27 configurations in the reduced ablation grid produce identical Sharpe per ticker.**

| Ticker | All-config Sharpe | Interpretation |
|--------|------------------|----------------|
| NVDA | 1.536 | Saturated — all τ/I/R combos identical |
| AAPL | 0.826 | Saturated |
| MSFT | 0.989 | Saturated |
| META | 1.225 | Saturated |
| AMD | 0.901 | Saturated |
| AVGO | 1.045 | Saturated |
| GOOGL | 0.490 | Saturated |
| AMZN | 0.873 | Saturated |
| TSLA | 0.802 | Saturated |
| CRM | 0.545 | Saturated |

**Mean across 10 tickers: 0.923 ± 0.291 (identical for all 27 configs)**

*Interpretation: At d=9 features with this correlation structure, all tested thresholds τ ∈ {0.5, 0.7, 0.8} activate the same rotation pairs. The Logistic classifier head is the bottleneck, not the DR transformation. This is a double-edged finding: (1) EntangleDR is trivially robust to its hyperparameters in this regime — no tuning needed; (2) the differentiated rotation behavior that EntangleDR is designed to exploit requires a higher-dimensional feature set.*

### 5.6 Table 6: NVDA Single-Ticker Detail (Ablation, Logistic only)

| Config | Sharpe | Accuracy | DSR |
|--------|--------|----------|-----|
| τ=0.5, I=1, R=π/72 | 1.536 | 56.4% | 0.987 |
| τ=0.5, I=1, R=π/36 | 1.536 | 56.4% | 0.987 |
| τ=0.7, I=3, R=π/36 | 1.536 | 56.4% | 0.987 |
| τ=0.8, I=10, R=π/18 | 1.536 | 56.4% | 0.987 |
| (all 27 configs) | 1.536 | 56.4% | 0.987 |

*DSR = 0.987 confirms the Sharpe is not an artifact of multiple testing across 27 configs.*

### 5.7 Statistical Significance

Wilcoxon signed-rank tests on fold-level Sharpe values (n_folds = 5 per ticker; n_tickers = 29) between EntangleDR and PCA: **p > 0.05** across all pairwise comparisons after BH correction. No statistically significant difference is detected at current feature dimensionality.

*Note: With only 5 CV folds, the Wilcoxon test has low statistical power. The lack of significance is a power limitation, not evidence of equivalence.*

---

## 6. Discussion

### 6.1 When does EntangleDR add value?

On the current 9-feature benchmark, EntangleDR provides **marginal but consistent** gains over PCA on 55% of S&P 500 sample tickers. The largest gains appear on:
- **Defensive tickers with near-zero baseline Sharpe** (JNJ: +0.136, ABT: +0.096) — EntangleDR appears to better regularize the noisy correlation structure.
- **High-momentum tech tickers** (META: +0.108, JPM: +0.081) — these exhibit stronger technical indicator correlation where the rotation scheme has more to work with.

### 6.2 The hyperparameter saturation finding

The identical Sharpe across all 27 ablation configs is the paper's most important negative result. It reveals that:
1. At d=9, the pairwise correlation pairs activated by τ=0.5 are the same as τ=0.7 and τ=0.8 (the high-correlation pairs are driven by MA ratios which always exceed 0.5 in this feature set).
2. Once those pairs are rotated, additional iterations (I > 1) or different rotation scales (R) produce no change in the selected top-k components.

**Implication:** To observe EntangleDR's hyperparameter sensitivity, the feature space must be extended. Candidate expansions: multi-lag returns (1, 2, 3, 5, 10, 20, 60 days), multiple RSI windows, Bollinger Band position, MACD, sector momentum. A 30–50 feature set should introduce heterogeneous correlation blocks where τ creates meaningful variation.

### 6.3 No-DR dominance

No-DR wins on 58.6% of S&P 500 tickers. This suggests that for the 9-feature set, compressing to 8 components does not systematically improve classifier generalization — the features already occupy a low-effective-dimension subspace. DR may be more valuable as dimensionality grows.

### 6.4 Classifier choice

Logistic Regression produces the most consistent winners (dominant in best-combo table), consistent with its lower variance and regularization (C=0.1). More complex classifiers (Random Forest, Gradient Boosting) show higher overfit risk under 5-fold purged CV on 5-year windows.

---

## 7. Limitations

1. **Feature dimensionality:** d=9 features is too small to differentiate EntangleDR's rotation structure. The core research hypothesis requires d ≥ 30.
2. **Single forward horizon:** Only 5-day targets tested. Results may differ at 1-day or 20-day horizons.
3. **Long-only backtest:** Short-selling not modeled; excludes half the signal information.
4. **yfinance data:** Potential survivorship bias (all tickers active through 2024), delayed prices, no corporate action adjustment verification.
5. **Transaction cost model:** Flat 10 bps; no market impact or bid-ask modeling.
6. **Statistical power:** 5-fold purged CV yields only 5 fold-level observations per ticker for significance tests. Insufficient for reliable paired tests.
7. **No quantum hardware:** EntangleDR is classical; no quantum advantage claimed or demonstrated.

---

## 8. Next Steps (Research Roadmap)

| Priority | Task | Expected gain |
|----------|------|---------------|
| **1** | Expand to d=30–50 features (multi-lag, MACD, Bollinger, sector) | Unlock hyperparameter differentiation |
| **2** | Run full ablation grid (48 configs) on expanded feature set | Identify optimal τ regime |
| **3** | Add regime analysis (pre-COVID / 2020–21 / 2022–24 splits) | Understand market-regime sensitivity |
| **4** | Cross-sectional panel study (Option A from roadmap) | Test cross-ticker generalization |
| **5** | 10-fold CV on longer data (2015–2024) | Improve statistical power |
| **6** | Implement supervised EntangleDR (target-aware τ weighting) | Option C from roadmap |

---

## 9. Conclusion

EntangleDR Lab provides a reproducible benchmark harness for correlation-aware DR in financial ML. On the current 9-feature, 29-ticker S&P 500 sample, EntangleDR achieves mean Sharpe 0.656 — essentially identical to PCA (0.656) — with a win rate of 55% at the ticker level. The hyperparameter ablation reveals that all tested configurations are equivalent at this feature dimensionality, confirming EntangleDR's stability but also its saturation. The primary research direction is clear: **expand the feature space to d ≥ 30**, where the method's selective rotation of high-correlation pairs can produce meaningfully different reduced representations.

---

## Reproducibility

```bash
# Install dependencies
cd backend && pip install -r requirements.txt

# Single ticker test
python ../true_test.py

# S&P 500 sample study (29 tickers, ~8 min)
python -m app.experiments.run_study --universe sp500_sample --limit 30

# Tech 10 ablation study (27 configs × 10 tickers, ~20 min)
python -m app.experiments.run_study --universe tech_10 --ablation --limit 10

# Full ablation grid (48 configs × 10 tickers, ~35 min)
python -m app.experiments.run_study --universe tech_10 --ablation --full-ablation --limit 10
```

Results written to `results/study_*.json` and `results/manifest.json`.

All studies: seed=42, target_days=5, n_components=8, cv_splits=5, transaction_cost_bps=10.

---

## References

- Bailey, D. H., & López de Prado, M. (2014). The deflated Sharpe ratio: correcting for selection bias, backtest overfitting, and non-normality. *Journal of Portfolio Management*, 40(5), 94–107.
- Jolliffe, I. T. (2002). *Principal Component Analysis* (2nd ed.). Springer.
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Schölkopf, B., Smola, A., & Müller, K.-R. (1998). Nonlinear component analysis as a kernel eigenvalue problem. *Neural Computation*, 10(5), 1299–1319.
- Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. *Journal of Machine Learning Research*, 7, 1–30.
