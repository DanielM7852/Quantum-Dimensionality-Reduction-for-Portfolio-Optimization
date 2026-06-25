"use client";

import { useEffect, useState } from "react";
import { EntangleDRSliders } from "@/components/EntangleDRSliders";
import {
  MAX_CV_SPLITS,
  MAX_ENTANGLEDR_ITERATIONS,
  RANGE_LIMIT_HINT,
  clampCvSplits,
  clampDateRange,
  clampEntangleIterations,
  formatIsoDate,
  getEndDateBounds,
  getStartDateBounds,
  validateAnalysisInputs,
} from "@/lib/analysisLimits";
import type { StudyPreset } from "@/lib/api";
import { DEMO_DEFAULTS, STUDY_PRESETS, type AnalyzeRequest } from "@/lib/api";

type Props = {
  onSubmit: (request: AnalyzeRequest) => void;
  loading: boolean;
};

const inputClass =
  "mt-1 w-full rounded-lg border border-slate-300 px-3 py-2 text-sm text-black placeholder:text-slate-500";

export function AnalyzeForm({ onSubmit, loading }: Props) {
  const [preset, setPreset] = useState<StudyPreset>(DEMO_DEFAULTS.preset);
  const [presetTickerIndex, setPresetTickerIndex] = useState(0);
  const [ticker, setTicker] = useState(DEMO_DEFAULTS.ticker);
  const [startDate, setStartDate] = useState(DEMO_DEFAULTS.start_date);
  const [endDate, setEndDate] = useState(DEMO_DEFAULTS.end_date);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [targetDays, setTargetDays] = useState(5);
  const [nComponents, setNComponents] = useState(8);
  const [cvSplits, setCvSplits] = useState(DEMO_DEFAULTS.cv_splits);
  const [transactionCostBps, setTransactionCostBps] = useState(10);
  const [seed, setSeed] = useState(42);
  const [correlationThreshold, setCorrelationThreshold] = useState(0.7);
  const [entangleIterations, setEntangleIterations] = useState(DEMO_DEFAULTS.entangledr_iterations);
  const [rotationScale, setRotationScale] = useState(Math.PI / 36);
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);
  const [formError, setFormError] = useState<string | null>(null);

  const today = formatIsoDate(new Date());
  const endBounds = getEndDateBounds(startDate);
  const startBounds = getStartDateBounds(endDate);
  const endDateMin = endBounds.min;
  const endDateMax = endBounds.max < today ? endBounds.max : today;
  const startDateMin = startBounds.min;
  const startDateMax = startBounds.max;

  const applyDateRange = (nextStart: string, nextEnd: string) => {
    const clamped = clampDateRange(nextStart, nextEnd, today);
    setStartDate(clamped.startDate);
    setEndDate(clamped.endDate);
    setFormError(null);
  };

  const handleStartDateChange = (value: string) => {
    applyDateRange(value, endDate);
  };

  const handleEndDateChange = (value: string) => {
    applyDateRange(startDate, value);
  };

  const handleCvSplitsChange = (value: number) => {
    setCvSplits(clampCvSplits(value));
    setFormError(null);
  };

  const handleIterationsChange = (value: number) => {
    setEntangleIterations(clampEntangleIterations(value));
    setFormError(null);
  };

  useEffect(() => {
    import("@/lib/api").then(({ checkHealth }) =>
      checkHealth().then(setApiOnline)
    );
  }, []);

  useEffect(() => {
    if (preset === "custom") return;
    const config = STUDY_PRESETS[preset];
    const tickers = config.tickers;
    setTicker(tickers[presetTickerIndex % tickers.length]);
    if ("startDate" in config && config.startDate) setStartDate(config.startDate);
    if ("endDate" in config && config.endDate) setEndDate(config.endDate);
    if ("cvSplits" in config && config.cvSplits) setCvSplits(config.cvSplits);
    if ("entangleIterations" in config && config.entangleIterations) {
      setEntangleIterations(clampEntangleIterations(config.entangleIterations));
    }
  }, [preset, presetTickerIndex]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const clamped = clampDateRange(startDate, endDate, today);
    const safeCvSplits = clampCvSplits(cvSplits);
    const safeIterations = clampEntangleIterations(entangleIterations);
    setStartDate(clamped.startDate);
    setEndDate(clamped.endDate);
    setCvSplits(safeCvSplits);
    setEntangleIterations(safeIterations);

    const validationError = validateAnalysisInputs(
      clamped.startDate,
      clamped.endDate,
      safeCvSplits
    );
    if (validationError) {
      setFormError(validationError);
      return;
    }

    setFormError(null);
    onSubmit({
      ticker: ticker.toUpperCase(),
      start_date: clamped.startDate,
      end_date: clamped.endDate,
      target_days: targetDays,
      n_components: nComponents,
      cv_splits: safeCvSplits,
      transaction_cost_bps: transactionCostBps,
      seed,
      entangledr_correlation_threshold: correlationThreshold,
      entangledr_iterations: safeIterations,
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
          All presets use a 1-year window that finishes in ~15–30s on the hosted API.
          Custom lets you pick any symbol within a 1–2 year range.
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
          Preset ticker: <strong>{ticker}</strong>
          {STUDY_PRESETS[preset].tickers.length > 1 && (
            <>
              {" "}
              <button type="button" onClick={cyclePresetTicker} className="underline">
                (cycle {STUDY_PRESETS[preset].tickers.length} tickers)
              </button>
            </>
          )}
          {preset === "quick_demo" && (
            <span className="mt-1 block text-emerald-800">
              SPY · Jun 2023–Jun 2024 · 3 CV folds · ~15s on hosted API.
            </span>
          )}
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
          <p className="mt-0.5 text-xs text-black">{RANGE_LIMIT_HINT}</p>
          <input
            type="date"
            value={startDate}
            min={startDateMin}
            max={startDateMax}
            onChange={(e) => handleStartDateChange(e.target.value)}
            className={inputClass}
            required
          />
        </label>
        <label className="block">
          <span className="text-sm font-medium text-black">End date</span>
          <p className="mt-0.5 text-xs text-black">
            Must be 1–2 years after the start date (through {today}).
          </p>
          <input
            type="date"
            value={endDate}
            min={endDateMin}
            max={endDateMax}
            onChange={(e) => handleEndDateChange(e.target.value)}
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
          maxIterations={MAX_ENTANGLEDR_ITERATIONS}
          onCorrelationThresholdChange={setCorrelationThreshold}
          onIterationsChange={handleIterationsChange}
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
                max={MAX_CV_SPLITS}
                value={cvSplits}
                onChange={(e) => handleCvSplitsChange(Number(e.target.value))}
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

      {formError && (
        <p className="mt-4 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-900">
          {formError}
        </p>
      )}

      <button
        type="submit"
        disabled={loading}
        className="mt-6 w-full rounded-lg bg-indigo-600 px-4 py-3 text-sm font-semibold text-white hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-60 sm:w-auto"
      >
        {loading ? "Running analysis (15–30s)..." : "Run analysis"}
      </button>
    </form>
  );
}
