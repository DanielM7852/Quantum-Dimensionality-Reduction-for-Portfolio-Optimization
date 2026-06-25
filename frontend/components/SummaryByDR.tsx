"use client";

import type { AnalyzeResponse } from "@/lib/api";

type Props = {
  summary: AnalyzeResponse["summary_by_dr"];
};

export function SummaryByDR({ summary }: Props) {
  const entries = Object.entries(summary).sort((a, b) => b[1].sharpe_ratio - a[1].sharpe_ratio);

  if (!entries.length) return null;

  return (
      <div className="rounded-xl border border-slate-200 bg-white p-4 text-black shadow-sm">
        <h3 className="mb-4 text-sm font-semibold text-black">Summary by DR Method</h3>
      <div className="overflow-x-auto">
        <table className="min-w-full text-left text-sm">
          <thead className="border-b border-slate-200 text-xs uppercase text-black">
            <tr>
              <th className="px-3 py-2">Method</th>
              <th className="px-3 py-2">Accuracy</th>
              <th className="px-3 py-2">F1</th>
              <th className="px-3 py-2">Sharpe</th>
              <th className="px-3 py-2">Sharpe 95% CI</th>
              <th className="px-3 py-2">Return</th>
            </tr>
          </thead>
          <tbody>
            {entries.map(([name, s]) => (
              <tr key={name} className="border-b border-slate-100 text-black">
                <td className="px-3 py-2 font-medium">{name}</td>
                <td className="px-3 py-2">{(s.accuracy * 100).toFixed(1)}%</td>
                <td className="px-3 py-2">{s.f1.toFixed(3)}</td>
                <td className="px-3 py-2">{s.sharpe_ratio.toFixed(2)}</td>
                <td className="px-3 py-2 text-black">
                  {s.sharpe_ci_lower != null && s.sharpe_ci_upper != null
                    ? `[${s.sharpe_ci_lower.toFixed(2)}, ${s.sharpe_ci_upper.toFixed(2)}]`
                    : "n/a"}
                </td>
                <td className="px-3 py-2">{(s.total_return * 100).toFixed(1)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
