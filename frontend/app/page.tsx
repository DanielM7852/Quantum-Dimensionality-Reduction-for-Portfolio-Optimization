"use client";

import { useCallback, useMemo, useState } from "react";
import { AnalyzeForm } from "@/components/AnalyzeForm";
import { BenchmarksPanel } from "@/components/BenchmarksPanel";
import { DisclaimerBanner, DisclaimerFooter } from "@/components/DisclaimerBanner";
import { ExportPanel } from "@/components/ExportPanel";
import { MethodsPanel } from "@/components/MethodsPanel";
import { ResearchVerdict } from "@/components/ResearchVerdict";
import { ResultsTable } from "@/components/ResultsTable";
import { SummaryByDR } from "@/components/SummaryByDR";
import { AccuracyChart } from "@/components/charts/AccuracyChart";
import { EquityCurveChart } from "@/components/charts/EquityCurveChart";
import { OverfitChart } from "@/components/charts/OverfitChart";
import { SharpeChart } from "@/components/charts/SharpeChart";
import { runAnalysis, type AnalyzeRequest, type AnalyzeResponse } from "@/lib/api";

export default function HomePage() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [lastRequest, setLastRequest] = useState<AnalyzeRequest | null>(null);
  const [selectedComboId, setSelectedComboId] = useState<string | null>(null);

  const handleSubmit = useCallback(async (request: AnalyzeRequest) => {
    setLoading(true);
    setError(null);
    try {
      const data = await runAnalysis(request);
      setResult(data);
      setLastRequest(request);
      setSelectedComboId(data.best_combo_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Analysis failed");
      setResult(null);
      setLastRequest(null);
      setSelectedComboId(null);
    } finally {
      setLoading(false);
    }
  }, []);

  const activeComboId = selectedComboId ?? result?.best_combo_id ?? null;
  const activeCombo = useMemo(
    () => result?.combos.find((c) => c.combo_id === activeComboId) ?? null,
    [result, activeComboId]
  );
  const equityCurve = activeComboId && result ? result.equity_curves[activeComboId] ?? [] : [];

  const benchmarkOverlays = useMemo(() => {
    if (!result?.benchmark_curves) return [];
    return Object.entries(result.benchmark_curves).map(([name, data]) => ({ name, data }));
  }, [result]);

  return (
    <>
      <DisclaimerBanner />
      <main className="mx-auto max-w-6xl flex-1 px-4 py-8">
        <header className="mb-8 rounded-xl bg-slate-900 px-6 py-6 text-white">
          <h1 className="text-3xl font-bold tracking-tight">
            {result?.meta.product_name ?? "EntangleDR Lab"}
          </h1>
          <p className="mt-2 max-w-2xl text-slate-300">
            Purged cross-validation benchmark for EntangleDR, PCA, Kernel-PCA, and No-DR baselines
            with buy-and-hold and random-signal comparisons. Research instrument only. Not
            financial advice.
          </p>
        </header>

        <div className="mb-8 grid gap-6 lg:grid-cols-3">
          <div className="lg:col-span-2">
            <AnalyzeForm onSubmit={handleSubmit} loading={loading} />
          </div>
          <MethodsPanel />
        </div>

        {loading && (
          <div className="mt-8 flex items-center justify-center gap-3 text-slate-600">
            <span className="h-5 w-5 animate-spin rounded-full border-2 border-indigo-600 border-t-transparent" />
            Running 12 model combinations. This may take up to 90 seconds...
          </div>
        )}

        {error && (
          <div className="mt-6 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800">
            {error}
          </div>
        )}

        {result && (
          <div className="mt-10 space-y-8">
            <div className="rounded-lg bg-white border border-slate-200 px-4 py-3 text-sm text-black">
              <strong>{result.meta.ticker}</strong> | {result.meta.start_date} to {result.meta.end_date} |{" "}
              {result.meta.sample_count} samples | {result.meta.positive_class_pct.toFixed(1)}% positive class | seed {result.meta.seed ?? 42}
              {lastRequest && (
                <span className="mt-1 block text-black">
                  EntangleDR settings: tau={lastRequest.entangledr_correlation_threshold.toFixed(2)}, I={lastRequest.entangledr_iterations}, R={(lastRequest.entangledr_rotation_scale * 180 / Math.PI).toFixed(1)} deg
                </span>
              )}
            </div>

            <ExportPanel result={result} chartId="equity-chart" />

            <ResearchVerdict combo={activeCombo} />

            <SummaryByDR summary={result.summary_by_dr} />

            <BenchmarksPanel
              benchmarks={result.benchmarks ?? []}
              significanceTests={result.significance_tests ?? []}
            />

            <div className="grid gap-6 lg:grid-cols-2">
              <AccuracyChart combos={result.combos} />
              <SharpeChart combos={result.combos} />
            </div>

            <OverfitChart combos={result.combos} />

            <div>
              <label className="mb-2 block text-sm font-medium text-black">
                Equity curve combo
                <select
                  value={activeComboId ?? ""}
                  onChange={(e) => setSelectedComboId(e.target.value)}
                  className="ml-2 rounded-lg border border-slate-300 px-2 py-1 text-sm text-black"
                >
                  {result.combos.map((c) => (
                    <option key={c.combo_id} value={c.combo_id}>
                      {c.dr_method} + {c.classifier}
                      {c.combo_id === result.best_combo_id ? " (best)" : ""}
                    </option>
                  ))}
                </select>
              </label>
              <div id="equity-chart">
                <EquityCurveChart
                  data={equityCurve}
                  benchmarks={benchmarkOverlays}
                  title={`Equity Curve: ${activeCombo?.dr_method ?? ""} + ${activeCombo?.classifier ?? ""}`}
                />
              </div>
            </div>

            <ResultsTable
              combos={result.combos}
              bestComboId={result.best_combo_id}
              selectedComboId={activeComboId}
              onSelectComboId={(comboId) => setSelectedComboId(comboId)}
            />

            <p className="text-xs text-black">{result.meta.metrics_convention}</p>
          </div>
        )}
      </main>
      <DisclaimerFooter />
    </>
  );
}
