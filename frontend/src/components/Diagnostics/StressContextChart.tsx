/**
 * Stress context chart — shows budget column time series for a selected
 * element or stream node around the selected timestep.
 */

import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import CircularProgress from '@mui/material/CircularProgress';
import Plot from 'react-plotly.js';

interface BudgetColumn {
  name: string;
  values: number[];
}

interface StressContextChartProps {
  title: string;
  times: string[];
  columns: BudgetColumn[];
  loading?: boolean;
  highlightTimestep?: string;
}

const TRACE_COLORS = [
  '#1976d2', '#d32f2f', '#2e7d32', '#ed6c02', '#9c27b0',
  '#0288d1', '#c2185b', '#558b2f', '#f57c00', '#7b1fa2',
];

export function StressContextChart({
  title,
  times,
  columns,
  loading,
  highlightTimestep,
}: StressContextChartProps) {
  if (loading) {
    return (
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, p: 2 }}>
        <CircularProgress size={20} />
        <Typography variant="body2">Loading budget data...</Typography>
      </Box>
    );
  }

  if (times.length === 0 || columns.length === 0) {
    return (
      <Typography variant="body2" color="text.secondary" sx={{ p: 1 }}>
        No budget data available.
      </Typography>
    );
  }

  // Filter out columns that are all zeros
  const nonZeroCols = columns.filter((col) => col.values.some((v) => Math.abs(v) > 1e-10));
  if (nonZeroCols.length === 0) {
    return (
      <Typography variant="body2" color="text.secondary" sx={{ p: 1 }}>
        All budget columns are zero for this selection.
      </Typography>
    );
  }

  const traces = nonZeroCols.map((col, i) => ({
    x: times,
    y: col.values,
    type: 'scatter' as const,
    mode: 'lines+markers' as const,
    name: col.name,
    marker: { size: 4, color: TRACE_COLORS[i % TRACE_COLORS.length] },
    line: { color: TRACE_COLORS[i % TRACE_COLORS.length], width: 2 },
  }));

  const shapes: Plotly.Shape[] = [];
  if (highlightTimestep) {
    shapes.push({
      type: 'line',
      x0: highlightTimestep,
      x1: highlightTimestep,
      y0: 0,
      y1: 1,
      yref: 'paper',
      line: { color: '#ff9800', width: 2, dash: 'dash' },
    } as Plotly.Shape);
  }

  return (
    <Box>
      <Plot
        data={traces}
        layout={{
          title: { text: title, font: { size: 13 } },
          xaxis: { title: { text: 'Date' }, type: 'date' },
          yaxis: { title: { text: 'Value' } },
          margin: { t: 35, b: 60, l: 70, r: 20 },
          height: 280,
          showlegend: true,
          legend: { orientation: 'h', y: -0.35, xanchor: 'center', x: 0.5, font: { size: 10 } },
          shapes,
        }}
        config={{ responsive: true, displaylogo: false }}
        style={{ width: '100%', height: '100%' }}
        useResizeHandler
      />
    </Box>
  );
}
