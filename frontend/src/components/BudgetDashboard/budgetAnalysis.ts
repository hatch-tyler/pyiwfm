/**
 * Pure computation functions for budget analysis charts.
 * No React dependencies — just data in, data out.
 */

const MONTH_LABELS = [
  'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
];

/**
 * Parse year and month directly from an ISO date string (YYYY-MM-...).
 * Avoids `new Date()` to prevent timezone-related month/year shifts.
 */
function parseYearMonth(dateStr: string): { year: number; month: number } {
  const parts = dateStr.split('-');
  if (parts.length >= 2) {
    const year = parseInt(parts[0], 10);
    const month = parseInt(parts[1], 10);
    if (!isNaN(year) && !isNaN(month)) {
      return { year, month };
    }
  }
  const d = new Date(dateStr);
  return { year: d.getFullYear(), month: d.getMonth() + 1 };
}

// ---------------------------------------------------------------------------
// Monthly climatology
// ---------------------------------------------------------------------------

export interface ClimatologySeries {
  name: string;
  min: number[];
  max: number[];
  mean: number[];
  median: number[];
}

export interface MonthlyClimatology {
  months: string[];
  series: ClimatologySeries[];
}

/**
 * Group values by calendar month (1–12) and compute min/max/mean/median
 * per month for each column.
 */
export function computeMonthlyClimatology(
  times: string[],
  columns: Array<{ name: string; values: number[] }>,
): MonthlyClimatology {
  const series: ClimatologySeries[] = columns.map((col) => {
    // Group values by month (1-indexed)
    const buckets: number[][] = Array.from({ length: 12 }, () => []);
    for (let i = 0; i < times.length; i++) {
      const { month } = parseYearMonth(times[i]);
      if (month >= 1 && month <= 12) {
        buckets[month - 1].push(col.values[i]);
      }
    }

    const min: number[] = [];
    const max: number[] = [];
    const mean: number[] = [];
    const median: number[] = [];

    for (let m = 0; m < 12; m++) {
      const vals = buckets[m];
      if (vals.length === 0) {
        min.push(0);
        max.push(0);
        mean.push(0);
        median.push(0);
        continue;
      }
      const sorted = [...vals].sort((a, b) => a - b);
      min.push(sorted[0]);
      max.push(sorted[sorted.length - 1]);
      mean.push(vals.reduce((s, v) => s + v, 0) / vals.length);
      const mid = Math.floor(sorted.length / 2);
      median.push(
        sorted.length % 2 === 0
          ? (sorted[mid - 1] + sorted[mid]) / 2
          : sorted[mid],
      );
    }

    return { name: col.name, min, max, mean, median };
  });

  return { months: MONTH_LABELS, series };
}

// ---------------------------------------------------------------------------
// Component ratio
// ---------------------------------------------------------------------------

export interface RatioSeries {
  name: string;
  times: string[];
  values: number[];
}

/**
 * Compute the ratio of numerator/denominator as a percentage (0–100).
 * Returns NaN for timesteps where denominator is zero.
 */
export function computeComponentRatio(
  times: string[],
  numerator: number[],
  denominator: number[],
  name: string,
): RatioSeries {
  const values = numerator.map((n, i) => {
    const d = denominator[i];
    if (d === 0) return NaN;
    return Math.abs(n / d) * 100;
  });
  return { name, times, values };
}

// ---------------------------------------------------------------------------
// Cumulative departure
// ---------------------------------------------------------------------------

export interface CumulativeDepartureSeries {
  name: string;
  times: string[];
  values: number[];
}

/**
 * Running sum of deviations from the long-term mean.
 * Positive = above average, negative = below average.
 */
export function computeCumulativeDeparture(
  times: string[],
  values: number[],
  name: string,
): CumulativeDepartureSeries {
  if (values.length === 0) return { name, times, values: [] };
  const mean = values.reduce((s, v) => s + v, 0) / values.length;
  const departures: number[] = [];
  let cumulative = 0;
  for (const v of values) {
    cumulative += v - mean;
    departures.push(cumulative);
  }
  return { name, times, values: departures };
}

// ---------------------------------------------------------------------------
// Exceedance probability
// ---------------------------------------------------------------------------

export interface ExceedanceSeries {
  name: string;
  exceedancePct: number[];
  values: number[];
}

/**
 * Sort values descending and assign exceedance probability as rank / (n+1) * 100.
 * Weibull plotting position.
 */
export function computeExceedance(
  values: number[],
  name: string,
): ExceedanceSeries {
  const n = values.length;
  if (n === 0) return { name, exceedancePct: [], values: [] };
  const sorted = [...values].sort((a, b) => b - a);
  const pct = sorted.map((_, i) => ((i + 1) / (n + 1)) * 100);
  return { name, exceedancePct: pct, values: sorted };
}
