"use client";

import { useEffect, useState } from "react";
import { EntangleDRSliders } from "@/components/EntangleDRSliders";
import type { StudyPreset } from "@/lib/api";
import { STUDY_PRESETS, type AnalyzeRequest } from "@/lib/api";

type Props = {
  onSubmit: (request: AnalyzeRequest) => void;
  loading: boolean;
};

const inputClass =
  "mt-1 w-full rounded-lg border border-slate-300 px-3 py-2 text-sm text-black placeholder:text-slate-500";

export function AnalyzeForm({ onSubmit, loading }: Props) {
  const [preset, setPreset] = useState<StudyPreset>("custom");
  const [presetTickerIndex, setPresetTickerIndex] = useState(0);
  const [ticker, setTicker] = useState("NVDA");
  const [startDate, setStartDate] = useState("2019-01-01");
  const [endDate, setEndDate] = useState("2024-11-01");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [targetDays, setTargetDays] = useState(5);
  const [nComponents, setNComponents] = useState(8);
  const [cvSplits, setCvSplits] = useState(5);
  const [transactionCostBps, setTransactionCostBps] = useState(10);
  const [seed, setSeed] = useState(42);
  const [correlationThreshold, setCorrelationThreshold] = useState(0.7);
  const [entangleIterations, setEntangleIterations] = useState(3);
  const [rotationScale, setRotationScale] = useState(Math.PI / 36);
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);

  useEffect(() => {
    import("@/lib/api").then(({ checkHealth }) =>
      checkHealth().then(setApiOnline)
    );
  }, []);

  useEffect(() => {
    if (preset !== "custom") {
      const tickers = STUDY_PRESETS[preset].tickers;
      setTicker(tickers[presetTickerIndex % tickers.length]);
    }
  }, [preset, presetTickerIndex]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit({
      ticker: ticker.toUpperCase(),
      start_date: startDate,
      end_date: endDate,
      target_days: targetDays,
      n_components: nComponents,
      cv_splits: cvSplits,
      transaction_cost_bps: transactionCostBps,
      seed,
      entangledr_correlation_threshold: correlationThreshold,
      entangledr_iterations: entangleIterations,
      entangledr_rotation_scale: rotationScale,
    });
  };

  const cyclePresetTicker = () => {
    if (preset === "custom") return;
    const len = STUDY_PRESETS[preset].tickers.length;
    setPresetTickerIndex((i) => (i + 1) % len);
  };

  return (
    <form
      onSubmit={handleSubmit}
      className="rounded-xl border border-slate-200 bg-white p-6 text-black shadow-sm"
    >
      <div className="mb-4 flex items-center justify-between">
        <h2 className="text-lg font-semibold text-black">Run DR Comparison</h2>
        {apiOnline === false && (
          <span className="text-xs text-red-700">API offline. Start backend on port 8000.</span>
        )}
        {apiOnline === true && (
          <span className="text-xs font-medium text-emerald-700">API connected</span>
        )}
      </div>

      <p className="mb-4 text-xs leading-relaxed text-black">
        Choose a ticker and date range, tune EntangleDR sliders if desired, then run analysis.
        The backend downloads price data, engineers 9 technical features, applies each DR method,
        trains 3 classifiers under purged time-series CV, and simulates a cost-aware long-only
        backtest.
      </p>

      <label className="mb-4 block">
        <span className="text-sm font-medium text-black">Study preset</span>
        <p className="mt-0.5 text-xs text-black">
          Quick ticker lists used in the paper benchmarks. Custom lets you type any symbol.
        </p>
        <select
          value={preset}
          onChange={(e) => setPreset(e.target.value as StudyPreset)}
          className={inputClass}
        >
          {Object.entries(STUDY_PRESETS).map(([key, { label }]) => (
            <option key={key} value={key}>
              {label}
            </option>
          ))}
        </select>
      </label>

      {preset !== "custom" && (
        <p className="mb-4 text-xs text-black">
          Preset ticker: <strong>{ticker}</strong>{" "}
          <button type="button" onClick={cyclePresetTicker} className="underline">
            (cycle {STUDY_PRESETS[preset].tickers.length} tickers)
          </button>
        </p>
      )}

      <div className="grid gap-4 sm:grid-cols-3">
        <label className="block">
          <span className="text-sm font-medium text-black">Ticker</span>
          <p className="mt-0.5 text-xs text-black">Stock symbol from Yahoo Finance (e.g. NVDA).</p>
          <input
            type="text"
            value={ticker}
            onChange={(e) => {
              setPreset("custom");
              setTicker(e.target.value);
            }}
            disabled={preset !== "custom"}
            className={`${inputClass} uppercase disabled:bg-slate-100 disabled:text-black`}
            required
          />
        </label>
        <label className="block">
          <span className="text-sm font-medium text-black">Start date</span>
          <p className="mt-0.5 text-xs text-black">First day of historical data included.</p>
          <input
            type="date"
            value={startDate}
            onChange={(e) => setStartDate(e.target.value)}
            className={inputClass}
            required
          />
        </label>
        <label className="block">
          <span className="text-sm font-medium text-black">End date</span>
          <p className="mt-0.5 text-xs text-black">Last day of historical data included.</p>
          <input
            type="date"
            value={endDate}
            onChange={(e) => setEndDate(e.target.value)}
            className={inputClass}
            required
          />
        </label>
      </div>

      <div className="mt-6">
        <EntangleDRSliders
          correlationThreshold={correlationThreshold}
          iterations={entangleIterations}
          rotationScale={rotationScale}
          onCorrelationThresholdChange={setCorrelationThreshold}
          onIterationsChange={setEntangleIterations}
          onRotationScaleChange={setRotationScale}
          disabled={loading}
        />
      </div>

      <button
        type="button"
        onClick={() => setShowAdvanced(!showAdvanced)}
        className="mt-4 text-sm text-black underline"
      >
        {showAdvanced ? "Hide" : "Show"} advanced parameters
      </button>

      {showAdvanced && (
        <div className="mt-4 space-y-4">
          <p className="text-xs text-black">
            These settings apply to all DR methods and classifiers, not just EntangleDR.
          </p>
          <div className="grid gap-4 sm:grid-cols-3 lg:grid-cols-5">
            <label className="block">
              <span className="text-sm font-medium text-black">Target days</span>
              <p className="mt-0.5 text-xs text-black">
                Forward return horizon for labels and trade hold length.
              </p>
              <input
                type="number"
                min={1}
                max={20}
                value={targetDays}
                onChange={(e) => setTargetDays(Number(e.target.value))}
                className={inputClass}
              />
            </label>
            <label className="block">
              <span className="text-sm font-medium text-black">Components</span>
              <p className="mt-0.5 text-xs text-black">
                Latent dimensions kept after dimensionality reduction.
              </p>
              <input
                type="number"
                min={2}
                max={20}
                value={nComponents}
                onChange={(e) => setNComponents(Number(e.target.value))}
                className={inputClass}
              />
            </label>
            <label className="block">
              <span className="text-sm font-medium text-black">CV splits</span>
              <p className="mt-0.5 text-xs text-black">
                Number of purged time-series cross-validation folds.
              </p>
              <input
                type="number"
                min={2}
                max={10}
                value={cvSplits}
                onChange={(e) => setCvSplits(Number(e.target.value))}
                className={inputClass}
              />
            </label>
            <label className="block">
              <span className="text-sm font-medium text-black">Cost (bps)</span>
              <p className="mt-0.5 text-xs text-black">
                Round-trip transaction cost subtracted per simulated trade.
              </p>
              <input
                type="number"
                min={0}
                max={100}
                value={transactionCostBps}
                onChange={(e) => setTransactionCostBps(Number(e.target.value))}
                className={inputClass}
              />
            </label>
            <label className="block">
              <span className="text-sm font-medium text-black">Seed</span>
              <p className="mt-0.5 text-xs text-black">
                Random seed for classifiers and the random-signal benchmark.
              </p>
              <input
                type="number"
                min={0}
                value={seed}
                onChange={(e) => setSeed(Number(e.target.value))}
                className={inputClass}
              />
            </label>
          </div>
        </div>
      )}

      <button
        type="submit"
        disabled={loading}
        className="mt-6 w-full rounded-lg bg-indigo-600 px-4 py-3 text-sm font-semibold text-white hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-60 sm:w-auto"
      >
        {loading ? "Running analysis (30-90s)..." : "Run analysis"}
      </button>
    </form>
  );
}
