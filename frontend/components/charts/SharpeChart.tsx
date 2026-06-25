"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ComboResult } from "@/lib/api";

type Props = {
  combos: ComboResult[];
};

const DR_COLORS: Record<string, string> = {
  "No-DR": "#64748b",
  PCA: "#3b82f6",
  "Kernel-PCA": "#f97316",
  EntangleDR: "#22c55e",
  "Quantum-Inspired": "#16a34a",
};

export function SharpeChart({ combos }: Props) {
  const classifiers = [...new Set(combos.map((c) => c.classifier))];
  const drMethods = [...new Set(combos.map((c) => c.dr_method))];

  const data = classifiers.map((clf) => {
    const row: Record<string, string | number> = { classifier: clf };
    for (const dr of drMethods) {
      const match = combos.find((c) => c.classifier === clf && c.dr_method === dr);
      row[dr] = match ? Number(match.sharpe_ratio.toFixed(2)) : 0;
    }
    return row;
  });

  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4 text-black shadow-sm">
      <h3 className="mb-4 text-sm font-semibold text-black">Sharpe Ratio (per-trade, annualized)</h3>
      <ResponsiveContainer width="100%" height={280}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="classifier" tick={{ fontSize: 11 }} />
          <YAxis tick={{ fontSize: 11 }} />
          <Tooltip />
          <Legend />
          {drMethods.map((dr) => (
            <Bar key={dr} dataKey={dr} fill={DR_COLORS[dr] ?? "#64748b"} />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
