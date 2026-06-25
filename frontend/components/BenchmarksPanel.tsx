"use client";

import type { BenchmarkResult, SignificanceResult } from "@/lib/api";

type Props = {
  benchmarks: BenchmarkResult[];
  significanceTests: SignificanceResult[];
};

export function BenchmarksPanel({ benchmarks, significanceTests }: Props) {
  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <div className="rounded-xl border border-slate-200 bg-white p-4 text-black shadow-sm">
        <h3 className="mb-2 text-sm font-semibold text-black">Benchmarks</h3>
        <p className="mb-4 text-xs text-black">
          Buy-and-hold and random-signal baselines on the same ticker and date range. Use these to
          judge whether any DR plus classifier combo beats simple alternatives.
        </p>
        <table className="min-w-full text-left text-sm text-black">
          <thead className="border-b border-slate-200 text-xs uppercase text-black">
            <tr>
              <th className="px-3 py-2">Strategy</th>
              <th className="px-3 py-2">Sharpe</th>
              <th className="px-3 py-2">Return</th>
              <th className="px-3 py-2">Max DD</th>
            </tr>
          </thead>
          <tbody>
            {benchmarks.map((b) => (
              <tr key={b.name} className="border-b border-slate-100">
                <td className="px-3 py-2 font-medium">{b.name}</td>
                <td className="px-3 py-2">{b.sharpe_ratio.toFixed(2)}</td>
                <td className="px-3 py-2">{(b.total_return * 100).toFixed(1)}%</td>
                <td className="px-3 py-2">{(b.max_drawdown * 100).toFixed(1)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {significanceTests.length > 0 && (
        <div className="rounded-xl border border-slate-200 bg-white p-4 text-black shadow-sm">
          <h3 className="mb-2 text-sm font-semibold text-black">Significance Tests (Sharpe)</h3>
          <p className="mb-4 text-xs text-black">
            Paired Wilcoxon tests on fold-level Sharpe for the best EntangleDR combo vs PCA and
            Kernel-PCA. BH significant means the difference survives multiple-testing correction.
          </p>
          <table className="min-w-full text-left text-sm text-black">
            <thead className="border-b border-slate-200 text-xs uppercase text-black">
              <tr>
                <th className="px-3 py-2">Comparison</th>
                <th className="px-3 py-2">p-value</th>
                <th className="px-3 py-2">BH significant</th>
              </tr>
            </thead>
            <tbody>
              {significanceTests.map((t, i) => (
                <tr key={i} className="border-b border-slate-100">
                  <td className="px-3 py-2">{t.method_a} vs {t.method_b}</td>
                  <td className="px-3 py-2">{t.p_value.toFixed(4)}</td>
                  <td className="px-3 py-2 font-medium">
                    {t.significant ? "Yes" : "No"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
