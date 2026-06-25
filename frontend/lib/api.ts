import { z } from "zod";
import {
  MAX_CV_SPLITS,
  MAX_ENTANGLEDR_ITERATIONS,
  MAX_ANALYSIS_RANGE_DAYS,
  MIN_ANALYSIS_RANGE_DAYS,
  daysBetween,
} from "@/lib/analysisLimits";

/** Defaults tuned for the hosted Render API (~60s request limit on free tier). */
export const DEMO_DEFAULTS = {
  preset: "quick_demo" as const,
  ticker: "SPY",
  start_date: "2023-06-01",
  end_date: "2024-06-01",
  cv_splits: 3,
  entangledr_iterations: 2,
};

export const STUDY_PRESETS = {
  quick_demo: {
    label: "Quick demo (recommended)",
    tickers: ["SPY"],
    startDate: "2023-06-01",
    endDate: "2024-06-01",
    cvSplits: 3,
    entangleIterations: 2,
  },
  custom: { label: "Custom ticker", tickers: [] as string[] },
  tech_10: {
    label: "Tech 10",
    tickers: ["AAPL", "MSFT", "GOOGL", "META", "AMZN", "AMD", "AVGO", "CRM"],
    startDate: "2023-06-01",
    endDate: "2024-06-01",
    cvSplits: 3,
    entangleIterations: 2,
  },
  sp500_sample: {
    label: "S&P Sample (10)",
    tickers: ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "JPM", "V", "UNH", "XOM"],
    startDate: "2023-06-01",
    endDate: "2024-06-01",
    cvSplits: 3,
    entangleIterations: 2,
  },
} as const;

export type StudyPreset = keyof typeof STUDY_PRESETS;

export const AnalyzeRequestSchema = z
  .object({
    ticker: z.string().min(1).max(10),
    start_date: z.string(),
    end_date: z.string(),
    target_days: z.number().min(1).max(20).default(5),
    n_components: z.number().min(2).max(20).default(8),
    cv_splits: z.number().min(2).max(MAX_CV_SPLITS).default(3),
    transaction_cost_bps: z.number().min(0).max(100).default(10),
    seed: z.number().min(0).default(42),
    entangledr_correlation_threshold: z.number().min(0).max(1).default(0.7),
    entangledr_iterations: z.number().min(1).max(MAX_ENTANGLEDR_ITERATIONS).default(2),
    entangledr_rotation_scale: z.number().min(0.001).max(1).default(Math.PI / 36),
  })
  .superRefine((data, ctx) => {
    const span = daysBetween(data.start_date, data.end_date);
    if (span < MIN_ANALYSIS_RANGE_DAYS) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: `Date range must be at least ${MIN_ANALYSIS_RANGE_DAYS} days (~1 year).`,
        path: ["end_date"],
      });
    }
    if (span > MAX_ANALYSIS_RANGE_DAYS) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: `Date range cannot exceed ${MAX_ANALYSIS_RANGE_DAYS} days (~2 years).`,
        path: ["end_date"],
      });
    }
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

const RETRYABLE_STATUS = new Set([502, 503, 504]);

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function wakeHostedApi(): Promise<void> {
  for (let attempt = 0; attempt < 3; attempt++) {
    if (await checkHealth()) return;
    await sleep(2000);
  }
}

export async function runAnalysis(
  request: AnalyzeRequest,
  signal?: AbortSignal
): Promise<AnalyzeResponse> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 120_000);
  const activeSignal = signal ?? controller.signal;

  try {
    for (let attempt = 0; attempt < 2; attempt++) {
      if (attempt > 0) {
        await wakeHostedApi();
        await sleep(2500);
      }

      try {
        const response = await fetch(`${API_URL}/api/v1/analyze`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(request),
          signal: activeSignal,
        });

        if (!response.ok) {
          const body = await response.json().catch(() => ({}));
          const detail = body.detail ?? response.statusText;
          if (RETRYABLE_STATUS.has(response.status) && attempt === 0) {
            continue;
          }
          if (response.status === 408) {
            throw new Error("Analysis timed out. Try the Quick demo preset or a shorter date range.");
          }
          throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
        }

        const data = await response.json();
        return AnalyzeResponseSchema.parse(data);
      } catch (err) {
        const retryable = attempt === 0 && err instanceof TypeError;
        if (retryable) continue;
        if (err instanceof Error && err.name === "AbortError") {
          throw new Error("Analysis timed out. Try the Quick demo preset or a shorter date range.");
        }
        if (err instanceof TypeError) {
          throw new Error(
            "Request failed while waking the hosted API. Wait a few seconds and try Quick demo again."
          );
        }
        throw err;
      }
    }

    throw new Error("Request failed after retry. Wait a few seconds and try Quick demo again.");
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
