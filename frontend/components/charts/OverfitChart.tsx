"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ReferenceLine,
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
  PCA: "#3b82f6",
  "Kernel-PCA": "#f97316",
  "Quantum-Inspired": "#22c55e",
};

export function OverfitChart({ combos }: Props) {
  const classifiers = [...new Set(combos.map((c) => c.classifier))];
  const drMethods = [...new Set(combos.map((c) => c.dr_method))];

  const data = classifiers.map((clf) => {
    const row: Record<string, string | number> = { classifier: clf };
    for (const dr of drMethods) {
      const match = combos.find((c) => c.classifier === clf && c.dr_method === dr);
      row[dr] = match ? Number((match.overfit_gap * 100).toFixed(1)) : 0;
    }
    return row;
  });

  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4 text-black shadow-sm">
      <h3 className="mb-4 text-sm font-semibold text-black">Overfit Gap (train - test accuracy, %)</h3>
      <ResponsiveContainer width="100%" height={280}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="classifier" tick={{ fontSize: 11 }} />
          <YAxis tick={{ fontSize: 11 }} />
          <Tooltip />
          <Legend />
          <ReferenceLine y={5} stroke="#f59e0b" strokeDasharray="4 4" label="5%" />
          <ReferenceLine y={10} stroke="#ef4444" strokeDasharray="4 4" label="10%" />
          {drMethods.map((dr) => (
            <Bar key={dr} dataKey={dr} fill={DR_COLORS[dr] ?? "#64748b"} />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
