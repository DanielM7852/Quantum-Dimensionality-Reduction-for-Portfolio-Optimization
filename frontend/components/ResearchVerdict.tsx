"use client";

import type { ComboResult } from "@/lib/api";

type Props = {
  combo: ComboResult | null;
};

const gradeConfig: Record<string, { label: string; color: string; message: string }> = {
  promising: {
    label: "Promising (research only)",
    color: "border-emerald-300 bg-emerald-50",
    message: "Shows modest edge in backtest. Requires further validation before any real use.",
  },
  inconclusive: {
    label: "Inconclusive",
    color: "border-amber-300 bg-amber-50",
    message: "Results do not clearly support or reject the strategy. Treat as exploratory.",
  },
  overfit_risk: {
    label: "Overfit risk",
    color: "border-red-300 bg-red-50",
    message: "Large train/test gap suggests overfitting. Do not extrapolate to live markets.",
  },
};

export function ResearchVerdict({ combo }: Props) {
  if (!combo) {
    return (
      <div className="rounded-xl border border-slate-200 bg-white p-6 text-sm text-black">
        No combo selected.
      </div>
    );
  }

  const config = gradeConfig[combo.research_grade] ?? gradeConfig.inconclusive;

  return (
    <div className={`rounded-xl border p-6 text-black ${config.color}`}>
      <h3 className="text-lg font-semibold text-black">Selected combo verdict</h3>
      <p className="mt-1 text-sm font-medium text-black">
        {combo.dr_method} + {combo.classifier}
      </p>
      <p className="mt-3 inline-block rounded-full border border-slate-300 bg-white px-3 py-1 text-sm font-semibold text-black">
        {config.label}
      </p>
      <p className="mt-3 text-sm text-black">{config.message}</p>
      <dl className="mt-4 grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
        <div>
          <dt className="text-black">Test accuracy</dt>
          <dd className="font-semibold">{(combo.test_accuracy * 100).toFixed(1)}%</dd>
        </div>
        <div>
          <dt className="text-black">Sharpe (trades)</dt>
          <dd className="font-semibold">{combo.sharpe_ratio.toFixed(2)}</dd>
        </div>
        <div>
          <dt className="text-black">Total return</dt>
          <dd className="font-semibold">{(combo.total_return * 100).toFixed(1)}%</dd>
        </div>
        <div>
          <dt className="text-black">Max drawdown</dt>
          <dd className="font-semibold">{(combo.max_drawdown * 100).toFixed(1)}%</dd>
        </div>
      </dl>
    </div>
  );
}
