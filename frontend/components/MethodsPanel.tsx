"use client";

export function MethodsPanel() {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-5 text-sm text-black shadow-sm">
      <h3 className="mb-3 font-semibold text-black">What this tool does</h3>
      <dl className="space-y-3">
        <div>
          <dt className="font-medium text-black">EntangleDR</dt>
          <dd className="mt-1 text-black">
            Correlation-aware dimensionality reduction. It rotates highly correlated feature pairs,
            then keeps the components with the most variance. Your sliders change how aggressively
            that rotation runs.
          </dd>
        </div>
        <div>
          <dt className="font-medium text-black">Baselines</dt>
          <dd className="mt-1 text-black">
            No-DR (raw features), PCA (linear), and Kernel-PCA (nonlinear RBF) are always run
            alongside EntangleDR so you can compare methods on the same ticker and date range.
          </dd>
        </div>
        <div>
          <dt className="font-medium text-black">Purged time-series CV</dt>
          <dd className="mt-1 text-black">
            Train and test splits include an embargo gap equal to the target horizon. This reduces
            label leakage from overlapping forward returns.
          </dd>
        </div>
        <div>
          <dt className="font-medium text-black">Backtest</dt>
          <dd className="mt-1 text-black">
            Long-only, non-overlapping entries with round-trip transaction costs. Buy-and-hold and
            random-signal benchmarks provide context for any strategy Sharpe.
          </dd>
        </div>
        <div>
          <dt className="font-medium text-black">Statistics</dt>
          <dd className="mt-1 text-black">
            Bootstrap 95% confidence intervals on fold metrics. Wilcoxon signed-rank tests compare
            EntangleDR vs PCA and Kernel-PCA with Benjamini-Hochberg correction.
          </dd>
        </div>
      </dl>
      <p className="mt-4 text-xs text-black">
        Full algorithm spec: see{" "}
        <code className="rounded bg-slate-100 px-1 text-black">docs/method_entangledr.md</code> in
        the repository.
      </p>
    </div>
  );
}
