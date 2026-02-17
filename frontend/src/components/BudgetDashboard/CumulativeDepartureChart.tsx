/**
 * Cumulative departure chart: shows wet/dry trends over time as the
 * running sum of deviations from the long-term mean.
 * Positive = above average (wetter), negative = below average (drier).
 */

import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Plot from 'react-plotly.js';
import type { BudgetUnitsMetadata } from '../../api/client';
import { computeCumulativeDeparture } from './budgetAnalysis';
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

interface CumulativeDepartureChartProps {
  classified: ClassifiedBudget;
  unitsMeta: BudgetUnitsMetadata | undefined;
  volumeUnit: string;
  contextPrefix?: string;
  budgetLabel?: string;
}

export function CumulativeDepartureChart({
  classified,
  unitsMeta,
  volumeUnit,
  contextPrefix,
  budgetLabel,
}: CumulativeDepartureChartProps) {
  // Collect flow columns from all chart groups (skip area, storage, subsidence)
  const flowGroups = classified.charts.filter((g) => g.chartKind === 'flow');
  if (flowGroups.length === 0) {
    return (
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
        <Typography color="text.secondary">No flow data available for cumulative departure analysis.</Typography>
      </Box>
    );
  }

  const sourceVolume = unitsMeta?.source_volume_unit ?? 'AF';

  const traces: Plotly.Data[] = [];
  let colorIdx = 0;

  for (const group of flowGroups) {
    for (const col of group.data.columns) {
      // Convert to display units
      const converted = convertVolumeValues(
        col.values, group.data.times, sourceVolume, volumeUnit, 'per_month', 'monthly',
      );
      const departure = computeCumulativeDeparture(converted.times, converted.values, col.name);

      traces.push({
        x: departure.times,
        y: departure.values,
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
  const chartTitle = `${titlePrefix}Cumulative Departure from Mean`;

  return (
    <Box sx={{ height: '100%', p: 1 }}>
      <Plot
        data={traces}
        layout={{
          margin: { l: 70, r: 30, t: 40, b: 50 },
          title: { text: chartTitle, font: { size: 14 } },
          xaxis: { title: { text: 'Date' }, type: 'date' },
          yaxis: { title: { text: `Cumulative Departure (${volLabel})` } },
          shapes: [
            {
              type: 'line',
              x0: 0,
              x1: 1,
              xref: 'paper',
              y0: 0,
              y1: 0,
              yref: 'y',
              line: { color: '#888', width: 1, dash: 'dash' },
            },
          ],
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
