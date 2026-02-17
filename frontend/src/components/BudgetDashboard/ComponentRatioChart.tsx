/**
 * Component ratio chart: shows relationships between budget components
 * as percentages over time (e.g. ET/Precip, Shortage/Demand).
 */

import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Plot from 'react-plotly.js';
import type { BudgetData, BudgetUnitsMetadata } from '../../api/client';
import { computeComponentRatio } from './budgetAnalysis';
import type { ClassifiedBudget } from './budgetSplitter';

const COLORS = [
  '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
  '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
];

/** Find a column by case-insensitive substring match. */
function findCol(
  columns: Array<{ name: string; values: number[] }>,
  ...patterns: string[]
): { name: string; values: number[] } | undefined {
  for (const pat of patterns) {
    const lower = pat.toLowerCase();
    const col = columns.find((c) => c.name.toLowerCase().includes(lower));
    if (col) return col;
  }
  return undefined;
}

interface RatioDef {
  numerator: string[];
  denominator: string[];
  label: string;
}

/** Ratio definitions per budget category. */
const RATIO_DEFS: Record<string, RatioDef[]> = {
  rootzone: [
    { numerator: ['actual_et', 'actual et', '_et'], denominator: ['precip'], label: 'ET / Precip' },
    { numerator: ['runoff'], denominator: ['precip'], label: 'Runoff / Precip' },
    { numerator: ['deep_perc', 'deep perc', 'perc'], denominator: ['precip'], label: 'Deep Perc / Precip' },
  ],
  lwu: [
    { numerator: ['shortage'], denominator: ['demand'], label: 'Shortage / Demand' },
    { numerator: ['pumping'], denominator: ['supply'], label: 'Pumping / Supply' },
  ],
  gw: [
    { numerator: ['recharge'], denominator: ['deep_perc', 'deep perc', 'perc'], label: 'Recharge / Deep Perc' },
    { numerator: ['pumping'], denominator: ['recharge'], label: 'Pumping / Recharge' },
  ],
  stream: [
    { numerator: ['diversion'], denominator: ['upstream', 'inflow'], label: 'Diversion / Upstream' },
    { numerator: ['return'], denominator: ['upstream', 'inflow'], label: 'Return Flow / Upstream' },
  ],
};

/** Detect budget category from type string. */
function detectCategory(budgetType: string): string {
  const l = budgetType.toLowerCase();
  if (l.includes('root') || l === 'rootzone') return 'rootzone';
  if (l.includes('land') || l === 'lwu') return 'lwu';
  if (l.includes('groundwater') || l.startsWith('gw')) return 'gw';
  if (l.includes('stream')) return 'stream';
  return 'other';
}

interface ComponentRatioChartProps {
  budgetData: BudgetData;
  classified: ClassifiedBudget;
  unitsMeta: BudgetUnitsMetadata | undefined;
  budgetType: string;
  contextPrefix?: string;
  budgetLabel?: string;
}

export function ComponentRatioChart({
  budgetData,
  classified,
  budgetType,
  contextPrefix,
  budgetLabel,
}: ComponentRatioChartProps) {
  const category = detectCategory(budgetType);
  const defs = RATIO_DEFS[category];

  if (!defs) {
    return (
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
        <Typography color="text.secondary">
          No ratio analysis available for this budget type.
        </Typography>
      </Box>
    );
  }

  // Collect all columns across all chart groups
  const allCols = classified.charts.flatMap((g) =>
    g.data.columns.map((c) => ({ name: c.name, values: c.values })),
  );
  const times = budgetData.times;

  // For prefix-based budgets (rootzone, lwu), try each prefix
  const prefixes = (category === 'rootzone' || category === 'lwu')
    ? ['AG_', 'URB_', 'NRV_']
    : [''];

  const traces: Plotly.Data[] = [];
  let colorIdx = 0;

  for (const prefix of prefixes) {
    const prefixLabel = prefix ? prefix.replace('_', '') + ' ' : '';
    const prefixCols = prefix
      ? allCols.filter((c) => c.name.toUpperCase().startsWith(prefix))
      : allCols;
    if (prefixCols.length === 0) continue;

    for (const def of defs) {
      const numCol = findCol(prefixCols, ...def.numerator);
      const denCol = findCol(prefixCols, ...def.denominator);
      if (!numCol || !denCol) continue;

      const ratio = computeComponentRatio(times, numCol.values, denCol.values, `${prefixLabel}${def.label}`);
      traces.push({
        x: ratio.times,
        y: ratio.values,
        name: ratio.name,
        type: 'scatter',
        mode: 'lines',
        line: { width: 2, color: COLORS[colorIdx % COLORS.length] },
        connectgaps: false,
      });
      colorIdx++;
    }
  }

  if (traces.length === 0) {
    return (
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
        <Typography color="text.secondary">
          No ratio analysis available for this budget type.
        </Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ height: '100%', p: 1 }}>
      <Plot
        data={traces}
        layout={{
          margin: { l: 70, r: 30, t: 40, b: 50 },
          title: { text: `${(contextPrefix || '') + (budgetLabel ? `${budgetLabel} ` : '')}Component Ratios`, font: { size: 14 } },
          xaxis: { title: { text: 'Date' }, type: 'date' },
          yaxis: { title: { text: 'Ratio (%)' }, range: [0, 100] },
          legend: { orientation: 'h', y: -0.15 },
          autosize: true,
        }}
        config={{ responsive: true, displaylogo: false }}
        style={{ width: '100%', height: '100%' }}
        useResizeHandler
      />
    </Box>
  );
}
