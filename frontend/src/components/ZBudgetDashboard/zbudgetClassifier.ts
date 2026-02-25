/**
 * ZBudget-specific column classifier.
 *
 * IWFM zone budget columns encode direction via suffixes:
 *   (+)   = inflow
 *   (-)   = outflow
 *   (+/-) = bidirectional (e.g. storage)
 *
 * This module separates storage columns into their own chart and tags
 * flow groups with `zbudgetKind: 'inflow_outflow'` so the renderer can
 * use stacked inflow/outflow bars instead of grouped bars.
 */

import type { BudgetData } from '../../api/client';
import type { ChartKind, ChartGroup } from '../BudgetDashboard/budgetSplitter';
import { classifyColumns } from '../BudgetDashboard/budgetSplitter';

// ---------------------------------------------------------------------------
// Sign classification
// ---------------------------------------------------------------------------

export type ColumnSign = 'inflow' | 'outflow' | 'bidirectional' | 'neutral';

/** Parse the (+), (-), (+/-) suffix from a column name. */
export function classifyColumnSign(name: string): ColumnSign {
  const trimmed = name.trimEnd();
  if (trimmed.endsWith('(+/-)')) return 'bidirectional';
  if (trimmed.endsWith('(+)')) return 'inflow';
  if (trimmed.endsWith('(-)')) return 'outflow';
  return 'neutral';
}

/** True if the column is a storage term (Storage (+/-), Absolute Storage, or bare "Storage"). */
export function isZBudgetStorageColumn(name: string): boolean {
  const l = name.toLowerCase().trim();
  // Match "Storage (+/-)", "Absolute Storage", or bare "Storage"
  if (l.includes('absolute storage')) return true;
  if (l.includes('storage') && l.includes('(+/-)')) return true;
  // Bare "storage" as the entire name (not partial match like "Storage Change")
  if (l === 'storage') return true;
  // "Storage" with any sign suffix
  if (l.includes('storage') && (l.includes('(+)') || l.includes('(-)') || l.includes('(+/'))) return true;
  return false;
}

/** True if the column is cumulative subsidence. */
function isZBudgetCumulativeSubsidence(name: string): boolean {
  const l = name.toLowerCase();
  return (
    (l.includes('cumulative') && l.includes('subsidence')) ||
    (l.includes('cum') && l.includes('subsid'))
  );
}

// ---------------------------------------------------------------------------
// Color palettes
// ---------------------------------------------------------------------------

/** Cool colors for inflow (+) columns. */
export const INFLOW_COLORS = [
  '#1565c0', // blue 800
  '#2e7d32', // green 800
  '#0097a7', // cyan 700
  '#00695c', // teal 800
  '#1976d2', // blue 700
  '#388e3c', // green 700
  '#0288d1', // light blue 700
  '#43a047', // green 600
];

/** Warm colors for outflow (-) columns. */
export const OUTFLOW_COLORS = [
  '#c62828', // red 800
  '#e65100', // orange 900
  '#bf360c', // deep orange 900
  '#d84315', // deep orange 800
  '#e53935', // red 600
  '#f4511e', // deep orange 600
  '#ff6d00', // orange A700
  '#ff8f00', // amber 800
];

/** Purple/gray colors for bidirectional (+/-) columns. */
export const BIDIRECTIONAL_COLORS = [
  '#6a1b9a', // purple 800
  '#4527a0', // deep purple 800
  '#7b1fa2', // purple 700
  '#5e35b1', // deep purple 600
];

// ---------------------------------------------------------------------------
// Extended ChartGroup type
// ---------------------------------------------------------------------------

export interface ZBudgetChartGroup extends ChartGroup {
  zbudgetKind?: 'inflow_outflow';
}

export interface ZBudgetClassifiedBudget {
  charts: ZBudgetChartGroup[];
}

// ---------------------------------------------------------------------------
// Detect budget category (mirrors budgetSplitter detectCategory)
// ---------------------------------------------------------------------------

function detectCategory(budgetType: string): string {
  const l = budgetType.toLowerCase();
  if (l.includes('groundwater') || l.startsWith('gw')) return 'gw';
  if (l.includes('land') || l === 'lwu') return 'lwu';
  if (l.includes('root') || l === 'rootzone') return 'rootzone';
  if (l.includes('unsat') || l === 'unsaturated') return 'unsaturated';
  return 'other';
}

/** True if any column has a (+), (-), or (+/-) suffix. */
function hasSignSuffixes(data: BudgetData): boolean {
  return data.columns.some((c) => classifyColumnSign(c.name) !== 'neutral');
}

// ---------------------------------------------------------------------------
// GW / Unsaturated zone budget classifier
// ---------------------------------------------------------------------------

function classifyGWZBudget(data: BudgetData): ZBudgetClassifiedBudget {
  const charts: ZBudgetChartGroup[] = [];
  const storageCols = data.columns.filter((c) => isZBudgetStorageColumn(c.name));
  const cumSubCols = data.columns.filter((c) => isZBudgetCumulativeSubsidence(c.name));
  const flowCols = data.columns.filter(
    (c) => !isZBudgetStorageColumn(c.name) && !isZBudgetCumulativeSubsidence(c.name),
  );

  // Only tag as inflow_outflow when columns have sign suffixes
  const hasSigns = flowCols.some((c) => classifyColumnSign(c.name) !== 'neutral');

  if (flowCols.length > 0) {
    charts.push({
      title: 'Flow Components',
      data: { location: data.location, times: data.times, columns: flowCols },
      chartKind: 'flow' as ChartKind,
      ...(hasSigns ? { zbudgetKind: 'inflow_outflow' as const } : {}),
    });
  }

  if (storageCols.length > 0) {
    charts.push({
      title: 'Change in Storage',
      data: { location: data.location, times: data.times, columns: storageCols },
      chartKind: 'storage' as ChartKind,
    });
  }

  if (cumSubCols.length > 0) {
    charts.push({
      title: 'Cumulative Subsidence',
      data: { location: data.location, times: data.times, columns: cumSubCols },
      chartKind: 'cumulative_subsidence' as ChartKind,
    });
  }

  if (charts.length === 0) {
    charts.push({
      title: `${data.location} Budget`,
      data,
      chartKind: 'flow' as ChartKind,
    });
  }

  return { charts };
}

// ---------------------------------------------------------------------------
// Rootzone prefix-based classifier with storage extraction
// ---------------------------------------------------------------------------

const CATEGORY_PREFIXES = ['AG_', 'URB_', 'NRV_'] as const;
const CATEGORY_LABELS: Record<string, string> = {
  AG_: 'Agricultural',
  URB_: 'Urban',
  NRV_: 'Native/Riparian',
};

function isAreaColumn(name: string): boolean {
  return name.toLowerCase().endsWith('_area') || name.toLowerCase() === 'area';
}

function classifyRootZoneZBudget(data: BudgetData): ZBudgetClassifiedBudget {
  const charts: ZBudgetChartGroup[] = [];

  for (const prefix of CATEGORY_PREFIXES) {
    const prefixCols = data.columns.filter((c) =>
      c.name.toUpperCase().startsWith(prefix),
    );
    if (prefixCols.length === 0) continue;

    const label = CATEGORY_LABELS[prefix] || prefix;

    const areaCols = prefixCols.filter((c) => isAreaColumn(c.name));
    const storageCols = prefixCols.filter(
      (c) => !isAreaColumn(c.name) && isZBudgetStorageColumn(c.name),
    );
    const flowCols = prefixCols.filter(
      (c) => !isAreaColumn(c.name) && !isZBudgetStorageColumn(c.name),
    );

    const prefixHasSigns = flowCols.some((c) => classifyColumnSign(c.name) !== 'neutral');

    if (flowCols.length > 0) {
      charts.push({
        title: `${label} Flow Components`,
        data: { location: data.location, times: data.times, columns: flowCols },
        chartKind: 'flow' as ChartKind,
        ...(prefixHasSigns ? { zbudgetKind: 'inflow_outflow' as const } : {}),
      });
    }

    if (storageCols.length > 0) {
      charts.push({
        title: `${label} Change in Storage`,
        data: { location: data.location, times: data.times, columns: storageCols },
        chartKind: 'storage' as ChartKind,
      });
    }

    if (areaCols.length > 0) {
      charts.push({
        title: `${label} Areas`,
        data: { location: data.location, times: data.times, columns: areaCols },
        chartKind: 'area' as ChartKind,
      });
    }
  }

  // Handle unprefixed columns
  const unprefixed = data.columns.filter(
    (c) =>
      !CATEGORY_PREFIXES.some((p) => c.name.toUpperCase().startsWith(p)),
  );
  const unprefixedStorage = unprefixed.filter((c) => isZBudgetStorageColumn(c.name));
  const unprefixedFlow = unprefixed.filter(
    (c) => !isZBudgetStorageColumn(c.name) && !isAreaColumn(c.name),
  );

  const unprefixedHasSigns = unprefixedFlow.some((c) => classifyColumnSign(c.name) !== 'neutral');

  if (unprefixedFlow.length > 0) {
    charts.push({
      title: 'Other Flow Components',
      data: { location: data.location, times: data.times, columns: unprefixedFlow },
      chartKind: 'flow' as ChartKind,
      ...(unprefixedHasSigns ? { zbudgetKind: 'inflow_outflow' as const } : {}),
    });
  }
  if (unprefixedStorage.length > 0) {
    charts.push({
      title: 'Other Change in Storage',
      data: { location: data.location, times: data.times, columns: unprefixedStorage },
      chartKind: 'storage' as ChartKind,
    });
  }

  if (charts.length === 0) {
    charts.push({
      title: `${data.location} Budget`,
      data,
      chartKind: 'flow' as ChartKind,
    });
  }

  return { charts };
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

/**
 * Classify ZBudget columns into chart groups with sign-aware separation.
 *
 * For GW/unsaturated budgets: separates storage and subsidence from flow,
 * tags flow charts with `zbudgetKind: 'inflow_outflow'`.
 *
 * For rootzone: per-prefix grouping with storage extraction per prefix.
 *
 * For LWU/other types without sign suffixes: delegates to the shared
 * `classifyColumns` from budgetSplitter.
 */
export function classifyZBudgetColumns(
  data: BudgetData,
  budgetType: string,
): ZBudgetClassifiedBudget {
  const category = detectCategory(budgetType);

  // Always use ZBudget-aware classifiers for known budget types —
  // they handle storage separation regardless of sign suffixes.
  switch (category) {
    case 'gw':
    case 'unsaturated':
      return classifyGWZBudget(data);
    case 'rootzone':
      return classifyRootZoneZBudget(data);
    default: {
      // For unknown types, check if columns have sign suffixes to decide
      if (hasSignSuffixes(data)) {
        return classifyGWZBudget(data);
      }
      const result = classifyColumns(data, budgetType);
      return { charts: result.charts };
    }
  }
}
