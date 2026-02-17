/**
 * Monthly pattern (climatology) chart: for each calendar month (Jan–Dec),
 * shows min/max envelope and mean line across all simulation years.
 */

import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Plot from 'react-plotly.js';
import type { BudgetUnitsMetadata } from '../../api/client';
import { computeMonthlyClimatology } from './budgetAnalysis';
import {
  convertVolumeValues,
  convertAreaValues,
  VOLUME_UNITS,
  AREA_UNITS,
} from './budgetUnits';
import type { ClassifiedBudget } from './budgetSplitter';

const COLORS = [
  '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
  '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
];

function hexToRgba(hex: string, alpha: number): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

interface MonthlyPatternChartProps {
  classified: ClassifiedBudget;
  unitsMeta: BudgetUnitsMetadata | undefined;
  volumeUnit: string;
  areaUnit: string;
  contextPrefix?: string;
  budgetLabel?: string;
}

export function MonthlyPatternChart({
  classified,
  unitsMeta,
  volumeUnit,
  areaUnit,
  contextPrefix,
  budgetLabel,
}: MonthlyPatternChartProps) {
  // Build converted columns for the first flow chart group
  const flowGroup = classified.charts.find((g) => g.chartKind === 'flow');
  if (!flowGroup || flowGroup.data.columns.length === 0) {
    return (
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
        <Typography color="text.secondary">No flow data available for monthly pattern analysis.</Typography>
      </Box>
    );
  }

  const sourceVolume = unitsMeta?.source_volume_unit ?? 'AF';
  const sourceArea = unitsMeta?.source_area_unit ?? 'ACRES';
  const isArea = flowGroup.chartKind === 'area';

  // Convert columns to display units (monthly, no time aggregation)
  const convertedCols = flowGroup.data.columns.map((col) => {
    if (isArea) {
      const result = convertAreaValues(col.values, flowGroup.data.times, sourceArea, areaUnit, 'monthly');
      return { name: col.name, values: result.values };
    } else {
      const result = convertVolumeValues(col.values, flowGroup.data.times, sourceVolume, 'volume', volumeUnit, 'cfs', 'monthly');
      return { name: col.name, values: result.values };
    }
  });

  const climatology = computeMonthlyClimatology(flowGroup.data.times, convertedCols);

  const traces: Plotly.Data[] = [];
  climatology.series.forEach((s, i) => {
    const color = COLORS[i % COLORS.length];
    // Min line (invisible, used as base for fill)
    traces.push({
      x: climatology.months,
      y: s.min,
      name: `${s.name} (min)`,
      type: 'scatter',
      mode: 'lines',
      line: { width: 0, color },
      showlegend: false,
    });
    // Max line with fill to min
    traces.push({
      x: climatology.months,
      y: s.max,
      name: `${s.name} (range)`,
      type: 'scatter',
      mode: 'lines',
      fill: 'tonexty',
      fillcolor: hexToRgba(color, 0.15),
      line: { width: 0, color },
      showlegend: false,
    });
    // Mean line
    traces.push({
      x: climatology.months,
      y: s.mean,
      name: s.name,
      type: 'scatter',
      mode: 'lines',
      line: { width: 2.5, color },
    });
  });

  // Build y-axis label
  const unitLabel = isArea
    ? (AREA_UNITS.find((u) => u.id === areaUnit)?.label ?? areaUnit)
    : (VOLUME_UNITS.find((u) => u.id === volumeUnit)?.label ?? volumeUnit);
  const yLabel = isArea ? `Area (${unitLabel})` : `Volume (${unitLabel} / month)`;

  const titlePrefix = (contextPrefix || '') + (budgetLabel ? `${budgetLabel} ` : '');
  const chartTitle = `${titlePrefix}Monthly Pattern (Climatology)`;

  return (
    <Box sx={{ height: '100%', p: 1 }}>
      <Plot
        data={traces}
        layout={{
          margin: { l: 70, r: 30, t: 40, b: 80 },
          title: { text: chartTitle, font: { size: 14 } },
          xaxis: { title: { text: 'Month' }, type: 'category' },
          yaxis: { title: { text: yLabel } },
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
