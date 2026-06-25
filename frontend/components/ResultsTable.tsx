"use client";

import { useMemo, useState } from "react";
import type { ComboResult } from "@/lib/api";

type Props = {
  combos: ComboResult[];
  bestComboId: string | null;
  selectedComboId?: string | null;
  onSelectComboId?: (comboId: string) => void;
};

const gradeColors: Record<string, string> = {
  promising: "bg-emerald-100 text-emerald-800",
  inconclusive: "bg-amber-100 text-amber-800",
  overfit_risk: "bg-red-100 text-red-800",
};

type SortKey = "sharpe_ratio" | "total_return" | "test_accuracy" | "max_drawdown";

export function ResultsTable({
  combos,
  bestComboId,
  selectedComboId,
  onSelectComboId,
}: Props) {
  const [drFilter, setDrFilter] = useState<string>("All");
  const [classifierFilter, setClassifierFilter] = useState<string>("All");
  const [sortKey, setSortKey] = useState<SortKey>("sharpe_ratio");

  const drMethods = useMemo(() => {
    const set = new Set(combos.map((c) => c.dr_method));
    return Array.from(set).sort();
  }, [combos]);

  const classifiers = useMemo(() => {
    const set = new Set(combos.map((c) => c.classifier));
    return Array.from(set).sort();
  }, [combos]);

  const filtered = useMemo(() => {
    return combos.filter((c) => {
      const okDr = drFilter === "All" || c.dr_method === drFilter;
      const okClf = classifierFilter === "All" || c.classifier === classifierFilter;
      return okDr && okClf;
    });
  }, [combos, drFilter, classifierFilter]);

  const sorted = useMemo(() => {
    const arr = [...filtered];
    arr.sort((a, b) => {
      // Higher is always "better" for our displayed metrics.
      // For drawdown, "less negative" is higher, so descending still makes sense.
      const av = a[sortKey];
      const bv = b[sortKey];
      return bv - av;
    });
    return arr;
  }, [filtered, sortKey]);

  return (
    <div className="space-y-4 overflow-x-auto rounded-xl border border-slate-200 bg-white p-4 shadow-sm text-black">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
        <div className="flex flex-wrap gap-3">
          <label className="block text-sm">
            <span className="mr-2 font-medium text-black">DR</span>
            <select
              value={drFilter}
              onChange={(e) => setDrFilter(e.target.value)}
              className="rounded-lg border border-slate-300 px-3 py-2 text-sm"
            >
              <option value="All">All</option>
              {drMethods.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </label>

          <label className="block text-sm">
            <span className="mr-2 font-medium text-black">Classifier</span>
            <select
              value={classifierFilter}
              onChange={(e) => setClassifierFilter(e.target.value)}
              className="rounded-lg border border-slate-300 px-3 py-2 text-sm"
            >
              <option value="All">All</option>
              {classifiers.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </label>

          <label className="block text-sm">
            <span className="mr-2 font-medium text-black">Sort</span>
            <select
              value={sortKey}
              onChange={(e) => setSortKey(e.target.value as SortKey)}
              className="rounded-lg border border-slate-300 px-3 py-2 text-sm"
            >
              <option value="sharpe_ratio">Sharpe</option>
              <option value="total_return">Return</option>
              <option value="test_accuracy">Accuracy</option>
              <option value="max_drawdown">Max drawdown</option>
            </select>
          </label>
        </div>

        <div className="text-xs text-black">
          Showing {sorted.length} of {combos.length} combos
        </div>
      </div>

      <table className="min-w-full text-left text-sm text-black">
        <thead className="border-b border-slate-200 bg-slate-50 text-xs uppercase text-black">
          <tr>
            <th className="px-4 py-3">DR Method</th>
            <th className="px-4 py-3">Classifier</th>
            <th className="px-4 py-3">Accuracy</th>
            <th className="px-4 py-3">Overfit Gap</th>
            <th className="px-4 py-3">Sharpe</th>
            <th className="px-4 py-3">Sharpe CI</th>
            <th className="px-4 py-3">Return</th>
            <th className="px-4 py-3">Trades</th>
            <th className="px-4 py-3">Grade</th>
          </tr>
        </thead>
        <tbody className="text-black">
          {sorted.map((c) => {
            const isBest = c.combo_id === bestComboId;
            const isSelected = selectedComboId && c.combo_id === selectedComboId;
            const rowBg = isSelected ? "bg-indigo-50" : isBest ? "bg-emerald-50" : "";
            const clickable = Boolean(onSelectComboId);

            return (
              <tr
                key={c.combo_id}
                className={`border-b border-slate-100 text-black ${rowBg} ${clickable ? "cursor-pointer" : ""}`}
                onClick={() => {
                  if (onSelectComboId) onSelectComboId(c.combo_id);
                }}
              >
                <td className="px-4 py-3 font-medium text-black">{c.dr_method}</td>
                <td className="px-4 py-3 text-black">{c.classifier}</td>
                <td className="px-4 py-3 text-black">{(c.test_accuracy * 100).toFixed(1)}%</td>
                <td className="px-4 py-3 text-black">{(c.overfit_gap * 100).toFixed(1)}%</td>
                <td className="px-4 py-3 text-black">{c.sharpe_ratio.toFixed(2)}</td>
                <td className="px-4 py-3 text-xs text-black">
                  {c.sharpe_ci_lower != null && c.sharpe_ci_upper != null
                    ? `[${c.sharpe_ci_lower.toFixed(2)}, ${c.sharpe_ci_upper.toFixed(2)}]`
                    : "n/a"}
                </td>
                <td className="px-4 py-3 text-black">{(c.total_return * 100).toFixed(1)}%</td>
                <td className="px-4 py-3 text-black">{c.trade_count}</td>
                <td className="px-4 py-3">
                  <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${gradeColors[c.research_grade]}`}>
                    {c.research_grade.replace("_", " ")}
                  </span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
