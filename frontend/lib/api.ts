import { z } from "zod";

export const STUDY_PRESETS = {
  custom: { label: "Custom ticker", tickers: [] as string[] },
  tech_10: {
    label: "Tech 10",
    tickers: ["NVDA", "AAPL", "MSFT", "GOOGL", "META", "AMZN", "TSLA", "AMD", "AVGO", "CRM"],
  },
  sp500_sample: {
    label: "S&P Sample (10)",
    tickers: ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "JPM", "V", "UNH", "XOM"],
  },
} as const;

export type StudyPreset = keyof typeof STUDY_PRESETS;

export const AnalyzeRequestSchema = z.object({
  ticker: z.string().min(1).max(10),
  start_date: z.string(),
  end_date: z.string(),
  target_days: z.number().min(1).max(20).default(5),
  n_components: z.number().min(2).max(20).default(8),
  cv_splits: z.number().min(2).max(10).default(5),
  transaction_cost_bps: z.number().min(0).max(100).default(10),
  seed: z.number().min(0).default(42),
  entangledr_correlation_threshold: z.number().min(0).max(1).default(0.7),
  entangledr_iterations: z.number().min(1).max(20).default(3),
  entangledr_rotation_scale: z.number().min(0.001).max(1).default(Math.PI / 36),
});

export const ComboResultSchema = z.object({
  dr_method: z.string(),
  classifier: z.string(),
  combo_id: z.string(),
  train_accuracy: z.number(),
  test_accuracy: z.number(),
  overfit_gap: z.number(),
  f1: z.number(),
  sharpe_ratio: z.number(),
  total_return: z.number(),
  max_drawdown: z.number(),
  win_rate: z.number(),
  trade_count: z.number(),
  research_grade: z.enum(["promising", "inconclusive", "overfit_risk"]),
  accuracy_ci_lower: z.number().nullable().optional(),
  accuracy_ci_upper: z.number().nullable().optional(),
  sharpe_ci_lower: z.number().nullable().optional(),
  sharpe_ci_upper: z.number().nullable().optional(),
  return_ci_lower: z.number().nullable().optional(),
  return_ci_upper: z.number().nullable().optional(),
  brier_score: z.number().nullable().optional(),
  deflated_sharpe: z.number().nullable().optional(),
});

export const EquityPointSchema = z.object({
  date: z.string(),
  equity: z.number(),
});

export const BenchmarkResultSchema = z.object({
  name: z.string(),
  sharpe_ratio: z.number(),
  total_return: z.number(),
  max_drawdown: z.number(),
  win_rate: z.number(),
  trade_count: z.number(),
});

export const SignificanceResultSchema = z.object({
  method_a: z.string(),
  method_b: z.string(),
  metric: z.string(),
  p_value: z.number(),
  significant: z.boolean(),
});

export const AnalyzeResponseSchema = z.object({
  meta: z.object({
    ticker: z.string(),
    start_date: z.string(),
    end_date: z.string(),
    sample_count: z.number(),
    positive_class_pct: z.number(),
    disclaimer_version: z.string(),
    disclaimer_text: z.string(),
    metrics_convention: z.string(),
    product_name: z.string().optional(),
    seed: z.number().optional(),
  }),
  combos: z.array(ComboResultSchema),
  best_combo_id: z.string().nullable(),
  equity_curves: z.record(z.string(), z.array(EquityPointSchema)),
  summary_by_dr: z.record(
    z.string(),
    z.object({
      accuracy: z.number(),
      f1: z.number(),
      sharpe_ratio: z.number(),
      total_return: z.number(),
      sharpe_ci_lower: z.number().nullable().optional(),
      sharpe_ci_upper: z.number().nullable().optional(),
    })
  ),
  benchmarks: z.array(BenchmarkResultSchema).optional().default([]),
  benchmark_curves: z.record(z.string(), z.array(EquityPointSchema)).optional().default({}),
  significance_tests: z.array(SignificanceResultSchema).optional().default([]),
});

export type AnalyzeRequest = z.infer<typeof AnalyzeRequestSchema>;
export type AnalyzeResponse = z.infer<typeof AnalyzeResponseSchema>;
export type ComboResult = z.infer<typeof ComboResultSchema>;
export type BenchmarkResult = z.infer<typeof BenchmarkResultSchema>;
export type SignificanceResult = z.infer<typeof SignificanceResultSchema>;

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export async function runAnalysis(
  request: AnalyzeRequest,
  signal?: AbortSignal
): Promise<AnalyzeResponse> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 180_000);

  try {
    const response = await fetch(`${API_URL}/api/v1/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
      signal: signal ?? controller.signal,
    });

    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      const detail = body.detail ?? response.statusText;
      if (response.status === 408) {
        throw new Error("Analysis timed out. Try a shorter date range.");
      }
      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    }

    const data = await response.json();
    return AnalyzeResponseSchema.parse(data);
  } finally {
    clearTimeout(timeout);
  }
}

export async function checkHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${API_URL}/health`, { cache: "no-store" });
    return response.ok;
  } catch {
    return false;
  }
}

export function combosToCsv(combos: ComboResult[]): string {
  const headers = [
    "dr_method", "classifier", "combo_id", "test_accuracy", "overfit_gap",
    "f1", "sharpe_ratio", "total_return", "max_drawdown", "win_rate",
    "trade_count", "research_grade", "sharpe_ci_lower", "sharpe_ci_upper",
  ];
  const rows = combos.map((c) =>
    headers.map((h) => {
      const v = c[h as keyof ComboResult];
      return v === null || v === undefined ? "" : String(v);
    }).join(",")
  );
  return [headers.join(","), ...rows].join("\n");
}
