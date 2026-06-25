/**
 * Limits tuned for the hosted Render API (~60s request ceiling on free tier).
 * Every UI control and API submission should stay within these bounds.
 */

export const MIN_ANALYSIS_RANGE_DAYS = 365;
export const MAX_ANALYSIS_RANGE_DAYS = 730;
export const MAX_CV_SPLITS = 3;
export const MAX_ENTANGLEDR_ITERATIONS = 3;

export function parseIsoDate(iso: string): Date {
  return new Date(`${iso}T00:00:00`);
}

export function formatIsoDate(date: Date): string {
  return date.toISOString().slice(0, 10);
}

export function addDays(iso: string, days: number): string {
  const date = parseIsoDate(iso);
  date.setDate(date.getDate() + days);
  return formatIsoDate(date);
}

export function daysBetween(start: string, end: string): number {
  const ms = parseIsoDate(end).getTime() - parseIsoDate(start).getTime();
  return Math.floor(ms / 86_400_000);
}

export function clampCvSplits(cvSplits: number): number {
  return Math.min(MAX_CV_SPLITS, Math.max(2, Math.round(cvSplits)));
}

export function clampEntangleIterations(iterations: number): number {
  return Math.min(MAX_ENTANGLEDR_ITERATIONS, Math.max(1, Math.round(iterations)));
}

export function getEndDateBounds(startDate: string): { min: string; max: string } {
  return {
    min: addDays(startDate, MIN_ANALYSIS_RANGE_DAYS),
    max: addDays(startDate, MAX_ANALYSIS_RANGE_DAYS),
  };
}

export function getStartDateBounds(endDate: string): { min: string; max: string } {
  return {
    min: addDays(endDate, -MAX_ANALYSIS_RANGE_DAYS),
    max: addDays(endDate, -MIN_ANALYSIS_RANGE_DAYS),
  };
}

export function clampDateRange(
  startDate: string,
  endDate: string,
  latestEndDate?: string
): { startDate: string; endDate: string } {
  let start = startDate;
  let end = endDate;

  if (latestEndDate && parseIsoDate(end) > parseIsoDate(latestEndDate)) {
    end = latestEndDate;
  }

  if (parseIsoDate(end) < parseIsoDate(start)) {
    end = addDays(start, MIN_ANALYSIS_RANGE_DAYS);
    if (latestEndDate && parseIsoDate(end) > parseIsoDate(latestEndDate)) {
      end = latestEndDate;
      start = addDays(end, -MIN_ANALYSIS_RANGE_DAYS);
    }
  }

  let span = daysBetween(start, end);
  if (span < MIN_ANALYSIS_RANGE_DAYS) {
    end = addDays(start, MIN_ANALYSIS_RANGE_DAYS);
    if (latestEndDate && parseIsoDate(end) > parseIsoDate(latestEndDate)) {
      end = latestEndDate;
      start = addDays(end, -MIN_ANALYSIS_RANGE_DAYS);
    }
  } else if (span > MAX_ANALYSIS_RANGE_DAYS) {
    end = addDays(start, MAX_ANALYSIS_RANGE_DAYS);
  }

  if (latestEndDate && parseIsoDate(end) > parseIsoDate(latestEndDate)) {
    end = latestEndDate;
    span = daysBetween(start, end);
    if (span < MIN_ANALYSIS_RANGE_DAYS) {
      start = addDays(end, -MIN_ANALYSIS_RANGE_DAYS);
    }
  }

  return { startDate: start, endDate: end };
}

export function validateAnalysisInputs(
  startDate: string,
  endDate: string,
  cvSplits: number
): string | null {
  if (parseIsoDate(endDate) < parseIsoDate(startDate)) {
    return "End date must be on or after the start date.";
  }

  const span = daysBetween(startDate, endDate);
  if (span < MIN_ANALYSIS_RANGE_DAYS) {
    return `Date range must be at least ${MIN_ANALYSIS_RANGE_DAYS} days (~1 year).`;
  }
  if (span > MAX_ANALYSIS_RANGE_DAYS) {
    return `Date range cannot exceed ${MAX_ANALYSIS_RANGE_DAYS} days (~2 years) on the hosted API.`;
  }
  if (cvSplits > MAX_CV_SPLITS) {
    return `CV splits cannot exceed ${MAX_CV_SPLITS} on the hosted API.`;
  }
  return null;
}

export const RANGE_LIMIT_HINT =
  "Hosted API: pick a range between 1 and 2 years. Longer studies are available via the local CLI.";
