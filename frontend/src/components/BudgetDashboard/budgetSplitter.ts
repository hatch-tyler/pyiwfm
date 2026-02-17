/**
 * Classifies budget columns into chart groups for all budget types.
 *
 * - GW: Flow Components, Storage Change, Subsidence
 * - LWU: AG Volumes, AG Areas, URB Volumes, URB Areas
 * - Root Zone: AG Volumes, AG Areas, URB Volumes, URB Areas, NRV Volumes, NRV Areas
 * - Unsaturated: Flow only (storage filtered out)
 * - Stream/Diversion/Lake: All columns as a single chart
 *
 * Storage columns (BEGIN_STORAGE, END_STORAGE, BGN_STOR, END_STOR) are
 * filtered out of display charts for ALL budget types.
 */

import type { BudgetData, BudgetColumnData } from '../../api/client';

export type ChartKind = 'flow' | 'area' | 'storage' | 'cumulative_subsidence' | 'diversion_balance';

export interface ChartGroup {
  title: string;
  data: BudgetData;
  chartKind: ChartKind;
}

export interface ClassifiedBudget {
  charts: ChartGroup[];
}

// ---------------------------------------------------------------------------
// Column classification helpers
// ---------------------------------------------------------------------------

/** True if column name matches any storage pattern. */
function isStorageColumn(name: string): boolean {
  const l = name.toLowerCase();
  return (
    (l.includes('begin') && l.includes('stor')) ||
    (l.includes('bgn') && l.includes('stor')) ||
    (l.includes('end') && l.includes('stor'))
  );
}

/** True if column represents cumulative subsidence (full word or CUM abbreviation). */
function isCumulativeSubsidenceColumn(name: string): boolean {
  const l = name.toLowerCase();
  return (
    (l.includes('cumulative') && l.includes('subsidence')) ||
    (l.includes('cum') && l.includes('subsid'))
  );
}

/** True if column name ends with _AREA (case-insensitive). */
function isAreaColumn(name: string): boolean {
  return name.toLowerCase().endsWith('_area') || name.toLowerCase() === 'area';
}

/** Running sum for cumulative storage change. */
function runningSum(values: number[]): number[] {
  const result: number[] = [];
  let acc = 0;
  for (const v of values) {
    acc += v;
    result.push(acc);
  }
  return result;
}

/** Detect budget type category from the type string. */
function detectCategory(budgetType: string): string {
  const l = budgetType.toLowerCase();
  if (l.includes('groundwater') || l.startsWith('gw')) return 'gw';
  if (l.includes('land') || l === 'lwu') return 'lwu';
  if (l.includes('root') || l === 'rootzone') return 'rootzone';
  if (l.includes('unsat') || l === 'unsaturated') return 'unsaturated';
  // stream_node must come before stream to avoid false match
  if (l === 'stream_node' || l.includes('stream_node') || l.includes('stream node')) return 'stream_node';
  if (l.includes('stream')) return 'stream';
  if (l.includes('diver')) return 'diversion';
  if (l.includes('lake')) return 'lake';
  if (l === 'small_watershed' || l.includes('small_watershed') || l.includes('small watershed')) return 'small_watershed';
  return 'other';
}

// ---------------------------------------------------------------------------
// GW budget classification
// ---------------------------------------------------------------------------

function classifyGW(data: BudgetData): ClassifiedBudget {
  const storageCols: BudgetColumnData[] = [];
  const cumSubsidenceCols: BudgetColumnData[] = [];
  const flowCols: BudgetColumnData[] = [];

  for (const col of data.columns) {
    if (isStorageColumn(col.name)) {
      storageCols.push(col);
    } else if (isCumulativeSubsidenceColumn(col.name)) {
      cumSubsidenceCols.push(col);
    } else {
      // Per-timestep subsidence stays in flow — it is a budget component
      // representing permanent storage loss due to subsidence.
      flowCols.push(col);
    }
  }

  const charts: ChartGroup[] = [];

  // Flow chart (includes per-timestep subsidence)
  if (flowCols.length > 0) {
    charts.push({
      title: 'Flow Components',
      data: { location: data.location, times: data.times, columns: flowCols },
      chartKind: 'flow',
    });
  }

  // Storage Change chart (derived from begin/end pair)
  if (storageCols.length > 0) {
    const beginCol = storageCols.find((c) => {
      const l = c.name.toLowerCase();
      return l.includes('begin') || l.includes('bgn');
    });
    const endCol = storageCols.find((c) => c.name.toLowerCase().includes('end'));
    const changeCol = storageCols.find((c) => c.name.toLowerCase().includes('change'));

    const storageChartCols: BudgetColumnData[] = [];
    const units = storageCols[0].units || '';

    if (beginCol && endCol) {
      const deltaValues = endCol.values.map((v, i) => v - beginCol.values[i]);
      storageChartCols.push({ name: 'Storage Change', values: deltaValues, units });
      storageChartCols.push({
        name: 'Cumulative Storage Change',
        values: runningSum(deltaValues),
        units,
      });
    } else if (changeCol) {
      storageChartCols.push(changeCol);
      storageChartCols.push({
        name: 'Cumulative Storage Change',
        values: runningSum(changeCol.values),
        units: changeCol.units,
      });
    }

    if (storageChartCols.length > 0) {
      charts.push({
        title: 'Storage Change',
        data: { location: data.location, times: data.times, columns: storageChartCols },
        chartKind: 'storage',
      });
    }
  }

  // Cumulative Subsidence chart (volumetric, shown as line)
  if (cumSubsidenceCols.length > 0) {
    charts.push({
      title: 'Cumulative Subsidence',
      data: { location: data.location, times: data.times, columns: cumSubsidenceCols },
      chartKind: 'cumulative_subsidence',
    });
  }

  // Fallback: if we only have storage/subsidence, return all as single chart
  if (charts.length === 0) {
    charts.push({
      title: `${data.location} Budget`,
      data,
      chartKind: 'flow',
    });
  }

  return { charts };
}

// ---------------------------------------------------------------------------
// Prefix-based classification (LWU / Root Zone)
// ---------------------------------------------------------------------------

const CATEGORY_PREFIXES = ['AG_', 'URB_', 'NRV_'] as const;
const CATEGORY_LABELS: Record<string, string> = {
  AG_: 'Agricultural',
  URB_: 'Urban',
  NRV_: 'Native/Riparian',
};

function classifyByPrefix(data: BudgetData): ClassifiedBudget {
  const charts: ChartGroup[] = [];

  for (const prefix of CATEGORY_PREFIXES) {
    const prefixCols = data.columns.filter((c) =>
      c.name.toUpperCase().startsWith(prefix),
    );
    if (prefixCols.length === 0) continue;

    const label = CATEGORY_LABELS[prefix] || prefix;

    // Split into areas vs volumes (filter storage from volumes)
    const areaCols: BudgetColumnData[] = [];
    const volumeCols: BudgetColumnData[] = [];

    for (const col of prefixCols) {
      if (isStorageColumn(col.name)) {
        // Filter out storage columns
        continue;
      } else if (isAreaColumn(col.name)) {
        areaCols.push(col);
      } else {
        volumeCols.push(col);
      }
    }

    if (volumeCols.length > 0) {
      charts.push({
        title: `${label} Volumes`,
        data: { location: data.location, times: data.times, columns: volumeCols },
        chartKind: 'flow',
      });
    }

    if (areaCols.length > 0) {
      charts.push({
        title: `${label} Areas`,
        data: { location: data.location, times: data.times, columns: areaCols },
        chartKind: 'area',
      });
    }
  }

  // Handle any columns that don't match known prefixes
  const unprefixed = data.columns.filter(
    (c) =>
      !CATEGORY_PREFIXES.some((p) => c.name.toUpperCase().startsWith(p)) &&
      !isStorageColumn(c.name),
  );
  if (unprefixed.length > 0) {
    charts.push({
      title: 'Other',
      data: { location: data.location, times: data.times, columns: unprefixed },
      chartKind: 'flow',
    });
  }

  // Fallback if no charts produced
  if (charts.length === 0) {
    charts.push({
      title: `${data.location} Budget`,
      data,
      chartKind: 'flow',
    });
  }

  return { charts };
}

// ---------------------------------------------------------------------------
// Unsaturated budget: filter storage, keep flow
// ---------------------------------------------------------------------------

function classifyUnsaturated(data: BudgetData): ClassifiedBudget {
  const flowCols = data.columns.filter((c) => !isStorageColumn(c.name));

  return {
    charts: [
      {
        title: `${data.location} Unsaturated Zone`,
        data: { location: data.location, times: data.times, columns: flowCols },
        chartKind: 'flow',
      },
    ],
  };
}

// ---------------------------------------------------------------------------
// Simple pass-through (Stream / Lake)
// ---------------------------------------------------------------------------

function classifySimple(data: BudgetData, label: string): ClassifiedBudget {
  // Still filter storage columns if any exist
  const cols = data.columns.filter((c) => !isStorageColumn(c.name));
  return {
    charts: [
      {
        title: `${data.location} ${label}`,
        data: { location: data.location, times: data.times, columns: cols.length > 0 ? cols : data.columns },
        chartKind: 'flow',
      },
    ],
  };
}

// ---------------------------------------------------------------------------
// Diversion budget: flow chart + balance stacked bar
// ---------------------------------------------------------------------------

/** Column names that form the diversion balance equation. */
const DIVERSION_BALANCE_COLS = [
  'actual diversion',
  'delivery',
  'recoverable loss',
  'non-recoverable loss',
];

function isDiversionBalanceCol(name: string): boolean {
  return DIVERSION_BALANCE_COLS.some((pat) => name.toLowerCase().includes(pat));
}

function classifyDiversion(data: BudgetData): ClassifiedBudget {
  const charts: ChartGroup[] = [];

  // All flow columns (no storage)
  const flowCols = data.columns.filter((c) => !isStorageColumn(c.name));
  if (flowCols.length > 0) {
    charts.push({
      title: `${data.location} Diversion`,
      data: { location: data.location, times: data.times, columns: flowCols },
      chartKind: 'flow',
    });
  }

  // Diversion balance chart with the 4 balance columns
  const balanceCols = data.columns.filter((c) => isDiversionBalanceCol(c.name));
  if (balanceCols.length >= 2) {
    charts.push({
      title: 'Diversion Balance',
      data: { location: data.location, times: data.times, columns: balanceCols },
      chartKind: 'diversion_balance',
    });
  }

  if (charts.length === 0) {
    charts.push({
      title: `${data.location} Diversion`,
      data,
      chartKind: 'flow',
    });
  }

  return { charts };
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

/**
 * Classify budget data into chart groups based on budget type.
 * Handles GW, LWU, Root Zone, Unsaturated, Stream, Diversion, Lake.
 */
export function classifyColumns(data: BudgetData, budgetType: string): ClassifiedBudget {
  const category = detectCategory(budgetType);

  switch (category) {
    case 'gw':
      return classifyGW(data);
    case 'lwu':
    case 'rootzone':
      return classifyByPrefix(data);
    case 'unsaturated':
      return classifyUnsaturated(data);
    case 'stream':
      return classifySimple(data, 'Stream');
    case 'stream_node':
      return classifySimple(data, 'Stream Node');
    case 'diversion':
      return classifyDiversion(data);
    case 'lake':
      return classifySimple(data, 'Lake');
    case 'small_watershed':
      return classifySimple(data, 'Small Watershed');
    default:
      return classifySimple(data, 'Budget');
  }
}

// Legacy export for backward compatibility — deprecated, use classifyColumns instead.
export type SplitBudgetData = ClassifiedBudget;
export function splitGWBudget(data: BudgetData, budgetType: string): ClassifiedBudget {
  return classifyColumns(data, budgetType);
}
