/**
 * Exceedance probability chart: shows the probability that a value is exceeded.
 * X-axis = exceedance probability (0–100%), Y-axis = values in selected units.
 * Useful for understanding frequency distributions of flows, storage, etc.
 */

import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Plot from 'react-plotly.js';
import type { BudgetUnitsMetadata } from '../../api/client';
import { computeExceedance } from './budgetAnalysis';
import {
  convertVolumeValues,
  VOLUME_UNITS,
} from './budgetUnits';
import type { ClassifiedBudget } from './budgetSplitter';

const COLORS = [
  '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
  '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
];

interface ExceedanceChartProps {
  classified: ClassifiedBudget;
  unitsMeta: BudgetUnitsMetadata | undefined;
  volumeUnit: string;
  contextPrefix?: string;
  budgetLabel?: string;
}

export function ExceedanceChart({
  classified,
  unitsMeta,
  volumeUnit,
  contextPrefix,
  budgetLabel,
}: ExceedanceChartProps) {
  const flowGroups = classified.charts.filter((g) => g.chartKind === 'flow');
  if (flowGroups.length === 0) {
    return (
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
        <Typography color="text.secondary">No flow data available for exceedance analysis.</Typography>
      </Box>
    );
  }

  const sourceVolume = unitsMeta?.source_volume_unit ?? 'AF';

  const traces: Plotly.Data[] = [];
  let colorIdx = 0;

  for (const group of flowGroups) {
    for (const col of group.data.columns) {
      const converted = convertVolumeValues(
        col.values, group.data.times, sourceVolume, 'volume', volumeUnit, 'cfs', 'monthly',
      );
      const exc = computeExceedance(converted.values, col.name);

      traces.push({
        x: exc.exceedancePct,
        y: exc.values,
        name: col.name,
        type: 'scatter',
        mode: 'lines',
        line: { width: 2, color: COLORS[colorIdx % COLORS.length] },
      });
      colorIdx++;
    }
  }

  const volLabel = VOLUME_UNITS.find((u) => u.id === volumeUnit)?.label ?? volumeUnit;

  const titlePrefix = (contextPrefix || '') + (budgetLabel ? `${budgetLabel} ` : '');
  const chartTitle = `${titlePrefix}Exceedance Probability`;

  // Reference annotations for common exceedance points
  const annotations = [
    { x: 10, text: '10% (wet)' },
    { x: 50, text: '50% (median)' },
    { x: 90, text: '90% (dry)' },
  ].map((a) => ({
    x: a.x,
    y: 1.02,
    xref: 'x' as const,
    yref: 'paper' as const,
    text: a.text,
    showarrow: false,
    font: { size: 10, color: '#888' },
  }));

  // Vertical reference lines
  const shapes = [10, 50, 90].map((pct) => ({
    type: 'line' as const,
    x0: pct,
    x1: pct,
    y0: 0,
    y1: 1,
    xref: 'x' as const,
    yref: 'paper' as const,
    line: { color: '#ccc', width: 1, dash: 'dot' as const },
  }));

  return (
    <Box sx={{ height: '100%', p: 1 }}>
      <Plot
        data={traces}
        layout={{
          margin: { l: 70, r: 30, t: 50, b: 80 },
          title: { text: chartTitle, font: { size: 14 } },
          xaxis: { title: { text: 'Exceedance Probability (%)' }, range: [0, 100] },
          yaxis: { title: { text: `Volume (${volLabel} / month)` } },
          shapes,
          annotations,
          legend: { orientation: 'h', y: -0.25, xanchor: 'center', x: 0.5 },
          autosize: true,
        }}
        config={{ responsive: true, displaylogo: false }}
        style={{ width: '100%', height: '100%' }}
        useResizeHandler
      />
    </Box>
  );
}
