/**
 * Unit conversion constants, conversion functions, and time aggregation
 * for the Budget Dashboard.
 *
 * Source-unit-aware: converts FROM the model's actual output units
 * (e.g. TAF, AF, ACRES) rather than assuming raw ft³/ft².
 */

// ---------------------------------------------------------------------------
// Volume units — canonical base is Acre-Feet (AF)
// ---------------------------------------------------------------------------
export interface UnitOption {
  id: string;
  label: string;
}

export const VOLUME_UNITS: UnitOption[] = [
  { id: 'taf', label: 'TAF' },
  { id: 'af', label: 'Acre-feet' },
  { id: 'ft3', label: 'ft\u00B3' },
  { id: 'm3', label: 'm\u00B3' },
];

/** Factor to convert one unit of the given type TO acre-feet. */
const VOLUME_TO_AF: Record<string, number> = {
  'taf': 1000,
  'af': 1,
  'acre-feet': 1,
  'acre-ft': 1,
  'ft3': 1 / 43560,
  'ft^3': 1 / 43560,
  'cubic feet': 1 / 43560,
  'm3': 1 / 1233.48,
  'cubic meters': 1 / 1233.48,
};

/** Factor to convert one AF to the given display unit. */
const AF_TO_DISPLAY: Record<string, number> = {
  'taf': 1 / 1000,
  'af': 1,
  'ft3': 43560,
  'm3': 1233.48,
};

// ---------------------------------------------------------------------------
// Area units — canonical base is Acres
// ---------------------------------------------------------------------------
export const AREA_UNITS: UnitOption[] = [
  { id: 'acres', label: 'Acres' },
  { id: 'ft2', label: 'ft\u00B2' },
  { id: 'm2', label: 'm\u00B2' },
  { id: 'km2', label: 'km\u00B2' },
  { id: 'mi2', label: 'mi\u00B2' },
];

const AREA_TO_ACRES: Record<string, number> = {
  'acres': 1,
  'acre': 1,
  'ft2': 1 / 43560,
  'ft^2': 1 / 43560,
  'sq-ft': 1 / 43560,
  'square feet': 1 / 43560,
  'm2': 1 / 4046.86,
  'square meters': 1 / 4046.86,
  'km2': 247.105,
  'mi2': 640,
};

const ACRES_TO_DISPLAY: Record<string, number> = {
  'acres': 1,
  'ft2': 43560,
  'm2': 4046.86,
  'km2': 1 / 247.105,
  'mi2': 1 / 640,
};

// ---------------------------------------------------------------------------
// Length units
// ---------------------------------------------------------------------------
export const LENGTH_UNITS: UnitOption[] = [
  { id: 'feet', label: 'Feet' },
  { id: 'meters', label: 'Meters' },
];

const LENGTH_TO_FEET: Record<string, number> = {
  'feet': 1,
  'ft': 1,
  'foot': 1,
  'meters': 3.28084,
  'm': 3.28084,
};

const FEET_TO_DISPLAY: Record<string, number> = {
  'feet': 1,
  'meters': 1 / 3.28084,
};

// ---------------------------------------------------------------------------
// Time aggregation options
// ---------------------------------------------------------------------------
export interface TimeAggOption {
  id: string;
  label: string;
}

export const TIME_AGGS: TimeAggOption[] = [
  { id: 'monthly', label: 'Monthly' },
  { id: 'seasonal', label: 'Seasonal' },
  { id: 'water_year', label: 'Water Year' },
  { id: 'calendar_year', label: 'Calendar Year' },
];

// ---------------------------------------------------------------------------
// Rate display options
// ---------------------------------------------------------------------------
export interface RateOption {
  id: string;
  label: string;
}

export const RATE_UNITS: RateOption[] = [
  { id: 'per_month', label: '/month' },
  { id: 'per_year', label: '/year' },
  { id: 'cfs', label: 'ft\u00B3/s (cfs)' },
  { id: 'cfd', label: 'ft\u00B3/day' },
  { id: 'm3_day', label: 'm\u00B3/day' },
  { id: 'm3_yr', label: 'm\u00B3/yr' },
];

// ---------------------------------------------------------------------------
// Source unit resolution (normalize IWFM unit strings)
// ---------------------------------------------------------------------------

/**
 * Normalize an IWFM unit string to a canonical key for lookup tables.
 * Handles case variations and common aliases.
 */
export function resolveSourceUnit(iwfmString: string): string {
  const s = iwfmString.trim().toLowerCase().replace(/[_\-\s.]+/g, ' ').trim();
  // Volume aliases
  if (s === 'taf' || s === 'thousand acre feet' || s === 'thousand acre ft') return 'taf';
  if (s === 'af' || s === 'acre feet' || s === 'acre ft' || s === 'ac ft' || s === 'acre-feet') return 'af';
  if (s === 'ft3' || s === 'ft^3' || s === 'cubic feet' || s === 'cu ft') return 'ft3';
  if (s === 'm3' || s === 'cubic meters' || s === 'cu m') return 'm3';
  // Area aliases
  if (s === 'acres' || s === 'acre') return 'acres';
  if (s === 'ft2' || s === 'ft^2' || s === 'sq ft' || s === 'square feet') return 'ft2';
  if (s === 'm2' || s === 'square meters') return 'm2';
  // Length aliases
  if (s === 'feet' || s === 'ft' || s === 'foot') return 'feet';
  if (s === 'meters' || s === 'm') return 'meters';
  // Fallback: return as-is
  return s;
}

/**
 * Map a source volume unit string to the best matching display unit id,
 * or null if unrecognized.
 */
export function sourceVolumeToDisplayDefault(sourceUnit: string): string | null {
  const canonical = resolveSourceUnit(sourceUnit);
  const match = VOLUME_UNITS.find((u) => u.id === canonical);
  return match ? match.id : null;
}

/**
 * Map a source area unit string to the best matching display unit id,
 * or null if unrecognized.
 */
export function sourceAreaToDisplayDefault(sourceUnit: string): string | null {
  const canonical = resolveSourceUnit(sourceUnit);
  const match = AREA_UNITS.find((u) => u.id === canonical);
  return match ? match.id : null;
}

/**
 * Map a source length unit string to the best matching display unit id,
 * or null if unrecognized.
 */
export function sourceLengthToDisplayDefault(sourceUnit: string): string | null {
  const canonical = resolveSourceUnit(sourceUnit);
  const match = LENGTH_UNITS.find((u) => u.id === canonical);
  return match ? match.id : null;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function findUnitLabel(units: UnitOption[], id: string): string {
  const u = units.find((u) => u.id === id);
  return u ? u.label : id;
}

/** Days in a given month (handles leap years). */
function daysInMonth(year: number, month: number): number {
  return new Date(year, month, 0).getDate();
}

/**
 * Parse year and month directly from an ISO date string (YYYY-MM-...).
 * Avoids `new Date()` to prevent timezone-related month/year shifts
 * (e.g. UTC midnight Oct 1 → Sep 30 in US Pacific).
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
  // Fallback for non-ISO formats
  const d = new Date(dateStr);
  return { year: d.getFullYear(), month: d.getMonth() + 1 };
}

/** Water year: Oct-Sep. Oct 2020 -> WY 2021. */
function waterYear(year: number, month: number): number {
  return month >= 10 ? year + 1 : year;
}

// ---------------------------------------------------------------------------
// Time aggregation with partial year filtering
// ---------------------------------------------------------------------------

export interface AggregatedResult {
  values: number[];
  times: string[];
  partialYearNote?: string;
}

/**
 * Aggregate monthly values into calendar or water year sums.
 * Removes partial first/last years (< 12 months) and returns a note.
 *
 * Water year: Oct-Sep. WY 2025 = Oct 2024 through Sep 2025.
 * Calendar year: Jan-Dec.
 *
 * Output times are plain year strings ("2022", "2023") usable as
 * Plotly date-axis values. The x-axis title (set by the chart) indicates
 * whether values represent water years or calendar years.
 */
export function aggregateToYear(
  values: number[],
  times: string[],
  type: 'calendar' | 'water',
): AggregatedResult {
  const buckets: Record<string, { sum: number; count: number }> = {};
  for (let i = 0; i < values.length; i++) {
    const { year, month } = parseYearMonth(times[i]);
    const key = type === 'water' ? String(waterYear(year, month)) : String(year);
    if (!buckets[key]) buckets[key] = { sum: 0, count: 0 };
    buckets[key].sum += values[i];
    buckets[key].count += 1;
  }

  const sortedKeys = Object.keys(buckets).sort();
  const notes: string[] = [];

  // Filter partial years (< 12 months of data)
  const fullKeys = sortedKeys.filter((key) => {
    const b = buckets[key];
    if (b.count < 12) {
      const label = type === 'water' ? `WY ${key}` : key;
      notes.push(`${label} excluded (${b.count}/12 months)`);
      return false;
    }
    return true;
  });

  return {
    times: fullKeys,
    values: fullKeys.map((k) => buckets[k].sum),
    partialYearNote: notes.length > 0 ? notes.join('; ') : undefined,
  };
}

// ---------------------------------------------------------------------------
// Seasonal aggregation
// ---------------------------------------------------------------------------

/** Season name from month: DJF=Winter, MAM=Spring, JJA=Summer, SON=Fall. */
function seasonOfMonth(month: number): string {
  if (month === 12 || month <= 2) return 'Winter';
  if (month <= 5) return 'Spring';
  if (month <= 8) return 'Summer';
  return 'Fall';
}

/** Season key for grouping: "YYYY Season" using the calendar year of the
 *  season's majority months.  Winter (DJF): Dec of prev year groups with
 *  Jan/Feb of current year → keyed by Jan's year. */
function seasonKey(year: number, month: number): string {
  const season = seasonOfMonth(month);
  const keyYear = (month === 12) ? year + 1 : year;
  return `${keyYear} ${season}`;
}

/**
 * Aggregate monthly values into seasonal sums.
 * Seasons: Winter (DJF), Spring (MAM), Summer (JJA), Fall (SON).
 * Partial seasons (< 3 months) are excluded.
 */
export function aggregateToSeason(
  values: number[],
  times: string[],
): AggregatedResult {
  const buckets: Record<string, { sum: number; count: number }> = {};
  for (let i = 0; i < values.length; i++) {
    const { year, month } = parseYearMonth(times[i]);
    const key = seasonKey(year, month);
    if (!buckets[key]) buckets[key] = { sum: 0, count: 0 };
    buckets[key].sum += values[i];
    buckets[key].count += 1;
  }

  // Sort by year then season order
  const seasonOrder: Record<string, number> = { Winter: 0, Spring: 1, Summer: 2, Fall: 3 };
  const sortedKeys = Object.keys(buckets).sort((a, b) => {
    const [ya, sa] = [parseInt(a), a.split(' ')[1]];
    const [yb, sb] = [parseInt(b), b.split(' ')[1]];
    if (ya !== yb) return ya - yb;
    return (seasonOrder[sa] ?? 0) - (seasonOrder[sb] ?? 0);
  });

  const notes: string[] = [];
  const fullKeys = sortedKeys.filter((key) => {
    const b = buckets[key];
    if (b.count < 3) {
      notes.push(`${key} excluded (${b.count}/3 months)`);
      return false;
    }
    return true;
  });

  return {
    times: fullKeys,
    values: fullKeys.map((k) => buckets[k].sum),
    partialYearNote: notes.length > 0 ? notes.join('; ') : undefined,
  };
}

/**
 * Aggregate area values into seasonal averages.
 * Same season logic but averages instead of sums.
 */
function aggregateAreaToSeason(
  values: number[],
  times: string[],
): AggregatedResult {
  const buckets: Record<string, { sum: number; count: number }> = {};
  for (let i = 0; i < values.length; i++) {
    const { year, month } = parseYearMonth(times[i]);
    const key = seasonKey(year, month);
    if (!buckets[key]) buckets[key] = { sum: 0, count: 0 };
    buckets[key].sum += values[i];
    buckets[key].count += 1;
  }

  const seasonOrder: Record<string, number> = { Winter: 0, Spring: 1, Summer: 2, Fall: 3 };
  const sortedKeys = Object.keys(buckets).sort((a, b) => {
    const [ya, sa] = [parseInt(a), a.split(' ')[1]];
    const [yb, sb] = [parseInt(b), b.split(' ')[1]];
    if (ya !== yb) return ya - yb;
    return (seasonOrder[sa] ?? 0) - (seasonOrder[sb] ?? 0);
  });

  const notes: string[] = [];
  const fullKeys = sortedKeys.filter((key) => {
    const b = buckets[key];
    if (b.count < 3) {
      notes.push(`${key} excluded (${b.count}/3 months)`);
      return false;
    }
    return true;
  });

  return {
    times: fullKeys,
    values: fullKeys.map((k) => buckets[k].sum / buckets[k].count),
    partialYearNote: notes.length > 0 ? notes.join('; ') : undefined,
  };
}

/**
 * Aggregate area values into yearly averages with partial year filtering.
 * Same water year / calendar year logic as aggregateToYear, but averages
 * instead of sums (appropriate for area/length quantities).
 */
function aggregateAreaToYear(
  values: number[],
  times: string[],
  type: 'calendar' | 'water',
): AggregatedResult {
  const buckets: Record<string, { sum: number; count: number }> = {};
  for (let i = 0; i < values.length; i++) {
    const { year, month } = parseYearMonth(times[i]);
    const key = type === 'water' ? String(waterYear(year, month)) : String(year);
    if (!buckets[key]) buckets[key] = { sum: 0, count: 0 };
    buckets[key].sum += values[i];
    buckets[key].count += 1;
  }

  const sortedKeys = Object.keys(buckets).sort();
  const notes: string[] = [];

  const fullKeys = sortedKeys.filter((key) => {
    const b = buckets[key];
    if (b.count < 12) {
      const label = type === 'water' ? `WY ${key}` : key;
      notes.push(`${label} excluded (${b.count}/12 months)`);
      return false;
    }
    return true;
  });

  return {
    times: fullKeys,
    values: fullKeys.map((k) => buckets[k].sum / buckets[k].count),
    partialYearNote: notes.length > 0 ? notes.join('; ') : undefined,
  };
}

// ---------------------------------------------------------------------------
// Rate conversion helpers
// ---------------------------------------------------------------------------

/**
 * Convert monthly volume values to a rate.
 * Input values are in ft3 (cubic feet) for rate-based units.
 */
function applyRate(
  ft3Values: number[],
  times: string[],
  rateId: string,
): number[] {
  switch (rateId) {
    case 'cfs': {
      return ft3Values.map((v, i) => {
        const { year, month } = parseYearMonth(times[i]);
        const seconds = daysInMonth(year, month) * 86400;
        return v / seconds;
      });
    }
    case 'cfd': {
      return ft3Values.map((v, i) => {
        const { year, month } = parseYearMonth(times[i]);
        return v / daysInMonth(year, month);
      });
    }
    case 'm3_day': {
      const ft3ToM3 = 1 / 35.3147;
      return ft3Values.map((v, i) => {
        const { year, month } = parseYearMonth(times[i]);
        return (v * ft3ToM3) / daysInMonth(year, month);
      });
    }
    case 'm3_yr': {
      const ft3ToM3 = 1 / 35.3147;
      return ft3Values.map((v) => v * 12 * ft3ToM3);
    }
    default:
      return ft3Values;
  }
}

// ---------------------------------------------------------------------------
// Main conversion entry points
// ---------------------------------------------------------------------------

/**
 * Convert raw monthly volume values to the user's selected units and
 * time aggregation. Source-unit-aware: uses the actual source unit from
 * the model's metadata rather than assuming ft³.
 */
export function convertVolumeValues(
  rawValues: number[],
  times: string[],
  sourceUnit: string,
  displayUnitId: string,
  rateUnitId: string,
  timeAggId: string,
): AggregatedResult {
  const srcKey = resolveSourceUnit(sourceUnit);
  const srcToAF = VOLUME_TO_AF[srcKey] ?? 1;

  // For rate-based units (cfs, cfd, m3_day, m3_yr), convert to ft3 first
  if (['cfs', 'cfd', 'm3_day', 'm3_yr'].includes(rateUnitId)) {
    const ft3Values = rawValues.map((v) => v * srcToAF * 43560);
    const rateValues = applyRate(ft3Values, times, rateUnitId);

    if (timeAggId === 'seasonal') {
      return aggregateToSeason(rateValues, times);
    } else if (timeAggId === 'water_year') {
      return aggregateToYear(rateValues, times, 'water');
    } else if (timeAggId === 'calendar_year') {
      return aggregateToYear(rateValues, times, 'calendar');
    }
    return { values: rateValues, times };
  }

  // For volume display units, convert source -> AF -> display
  const afToDisplay = AF_TO_DISPLAY[displayUnitId] ?? 1;
  const conversionFactor = srcToAF * afToDisplay;
  let converted = rawValues.map((v) => v * conversionFactor);

  // Apply per_year rate
  if (rateUnitId === 'per_year') {
    converted = converted.map((v) => v * 12);
  }

  // Time aggregation
  if (timeAggId === 'seasonal') {
    return aggregateToSeason(converted, times);
  } else if (timeAggId === 'water_year') {
    return aggregateToYear(converted, times, 'water');
  } else if (timeAggId === 'calendar_year') {
    return aggregateToYear(converted, times, 'calendar');
  }
  return { values: converted, times };
}

/**
 * Convert raw area values to user's selected area unit and aggregate by time.
 * Source-unit-aware.
 */
export function convertAreaValues(
  rawValues: number[],
  times: string[],
  sourceUnit: string,
  areaUnitId: string,
  timeAggId: string,
): AggregatedResult {
  const srcKey = resolveSourceUnit(sourceUnit);
  const srcToAcres = AREA_TO_ACRES[srcKey] ?? 1;
  const acresToDisplay = ACRES_TO_DISPLAY[areaUnitId] ?? 1;
  const factor = srcToAcres * acresToDisplay;
  const converted = rawValues.map((v) => v * factor);

  // For area, yearly/seasonal aggregation takes the average (not sum)
  if (timeAggId === 'seasonal') {
    return aggregateAreaToSeason(converted, times);
  } else if (timeAggId === 'water_year' || timeAggId === 'calendar_year') {
    const type = timeAggId === 'water_year' ? 'water' : 'calendar';
    return aggregateAreaToYear(converted, times, type);
  }
  return { values: converted, times };
}

/**
 * Convert raw length values to user's selected length unit.
 */
export function convertLengthValues(
  rawValues: number[],
  times: string[],
  sourceUnit: string,
  lengthUnitId: string,
  timeAggId: string,
): AggregatedResult {
  const srcKey = resolveSourceUnit(sourceUnit);
  const srcToFeet = LENGTH_TO_FEET[srcKey] ?? 1;
  const feetToDisplay = FEET_TO_DISPLAY[lengthUnitId] ?? 1;
  const factor = srcToFeet * feetToDisplay;
  const converted = rawValues.map((v) => v * factor);

  // For length (subsidence), yearly/seasonal aggregation takes the average
  if (timeAggId === 'seasonal') {
    return aggregateAreaToSeason(converted, times);
  } else if (timeAggId === 'water_year' || timeAggId === 'calendar_year') {
    const type = timeAggId === 'water_year' ? 'water' : 'calendar';
    return aggregateAreaToYear(converted, times, type);
  }
  return { values: converted, times };
}

// ---------------------------------------------------------------------------
// X-axis label builder
// ---------------------------------------------------------------------------

/**
 * Build an x-axis title string from the current time aggregation setting.
 */
export function getXAxisLabel(timeAggId: string): string {
  if (timeAggId === 'seasonal') return 'Season';
  if (timeAggId === 'water_year') return 'Water Year';
  if (timeAggId === 'calendar_year') return 'Calendar Year';
  return 'Date';
}

// ---------------------------------------------------------------------------
// Y-axis label builder
// ---------------------------------------------------------------------------

/**
 * Build a y-axis label from the chart kind and selected units.
 */
export function getYAxisLabel(
  chartKind: 'flow' | 'area' | 'storage' | 'cumulative_subsidence' | 'diversion_balance',
  volumeUnitId: string,
  rateUnitId: string,
  areaUnitId: string,
  _lengthUnitId: string,
  timeAggId: string,
): string {
  if (chartKind === 'area') {
    return `Area (${findUnitLabel(AREA_UNITS, areaUnitId)})`;
  }

  // Cumulative subsidence is volumetric — use volume label with a subsidence prefix
  if (chartKind === 'cumulative_subsidence') {
    const volLabel = findUnitLabel(VOLUME_UNITS, volumeUnitId);
    return `Cumulative Subsidence (${volLabel})`;
  }

  // Volume-based kinds: flow, storage, diversion_balance
  const volLabel = findUnitLabel(VOLUME_UNITS, volumeUnitId);
  const prefix = chartKind === 'storage' ? 'Storage Change' : 'Volume';

  // For rate-based display units, show the rate label directly
  if (['cfs', 'cfd', 'm3_day', 'm3_yr'].includes(rateUnitId)) {
    const rateOpt = RATE_UNITS.find((r) => r.id === rateUnitId);
    return `${prefix} (${rateOpt ? rateOpt.label : volLabel})`;
  }

  // For yearly aggregation, show "volume / year"
  if (timeAggId !== 'monthly') {
    return `${prefix} (${volLabel} / year)`;
  }

  // Monthly + per_month or per_year
  if (rateUnitId === 'per_year') {
    return `${prefix} (${volLabel} / year)`;
  }
  return `${prefix} (${volLabel} / month)`;
}
