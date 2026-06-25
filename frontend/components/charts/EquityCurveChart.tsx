"use client";

import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

type CurveSeries = {
  key: string;
  data: { date: string; equity: number }[];
  color: string;
  strokeDasharray?: string;
};

type Props = {
  data: { date: string; equity: number }[];
  benchmarks?: { name: string; data: { date: string; equity: number }[] }[];
  title?: string;
};

export function EquityCurveChart({
  data,
  benchmarks = [],
  title = "Equity Curve (non-overlapping trades)",
}: Props) {
  if (!data.length) {
    return (
      <div className="rounded-xl border border-slate-200 bg-white p-4 text-sm text-black shadow-sm">
        No equity curve data available.
      </div>
    );
  }

  const series: CurveSeries[] = [
    { key: "Strategy", data, color: "#4f46e5" },
    ...benchmarks.map((b, i) => ({
      key: b.name,
      data: b.data,
      color: i === 0 ? "#94a3b8" : "#f59e0b",
      strokeDasharray: "6 4",
    })),
  ];

  const dateMap = new Map<string, Record<string, string | number>>();
  for (const s of series) {
    for (const pt of s.data) {
      const d = pt.date.slice(0, 10);
      if (!dateMap.has(d)) dateMap.set(d, { date: d });
      dateMap.get(d)![s.key] = Number(pt.equity.toFixed(4));
    }
  }

  const chartData = Array.from(dateMap.values()).sort((a, b) =>
    String(a.date).localeCompare(String(b.date))
  );

  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4 text-black shadow-sm">
      <h3 className="mb-4 text-sm font-semibold text-black">{title}</h3>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="date" tick={{ fontSize: 10 }} interval="preserveStartEnd" />
          <YAxis tick={{ fontSize: 11 }} domain={["auto", "auto"]} />
          <Tooltip />
          <Legend />
          {series.map((s) => (
            <Line
              key={s.key}
              type="monotone"
              dataKey={s.key}
              stroke={s.color}
              strokeWidth={2}
              strokeDasharray={s.strokeDasharray}
              dot={false}
              connectNulls
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
