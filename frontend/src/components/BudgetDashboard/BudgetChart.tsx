/**
 * Plotly budget chart component.
 * Supports stacked area, grouped bar, and line chart types.
 */

import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import CircularProgress from '@mui/material/CircularProgress';
import Plot from 'react-plotly.js';
import type { BudgetData } from '../../api/client';

interface BudgetChartProps {
  data: BudgetData | null;
  chartType: 'area' | 'bar' | 'line';
  loading: boolean;
  title?: string;
  dualAxis?: boolean;
  yAxisLabel?: string;
  xAxisLabel?: string;
  partialYearNote?: string;
}

/**
 * Detect whether time labels are ISO date strings, bare year strings,
 * or category labels. Returns 'date' for ISO/year strings, 'category' otherwise.
 */
function detectXAxisType(times: string[]): 'date' | 'category' {
  if (times.length === 0) return 'date';
  const first = times[0];
  // ISO date strings contain 'T' or '-' with at least 7 chars (YYYY-MM)
  if (first.includes('T') || (first.includes('-') && first.length >= 7)) return 'date';
  // Bare 4-digit year strings (from yearly aggregation) are valid Plotly dates
  if (/^\d{4}$/.test(first)) return 'date';
  return 'category';
}

// Plotly color palette
const COLORS = [
  '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
  '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
];

export function BudgetChart({ data, chartType, loading, title, dualAxis, yAxisLabel, xAxisLabel, partialYearNote }: BudgetChartProps) {
  if (loading) {
    return (
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
        <CircularProgress />
      </Box>
    );
  }

  if (!data || data.columns.length === 0) {
    return (
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
        <Typography color="text.secondary">
          Select a budget type and location to view data.
        </Typography>
      </Box>
    );
  }

  const traces: Plotly.Data[] = data.columns.map((col, i) => {
    const useY2 = dualAxis && i === 1;
    if (chartType === 'area') {
      return {
        x: data.times,
        y: col.values,
        name: col.name,
        type: 'scatter' as const,
        mode: 'lines' as const,
        fill: 'tonexty' as const,
        stackgroup: useY2 ? 'two' : 'one',
        yaxis: useY2 ? 'y2' : 'y',
        line: { width: 0.5, color: COLORS[i % COLORS.length] },
      };
    } else if (chartType === 'bar') {
      return {
        x: data.times,
        y: col.values,
        name: col.name,
        type: 'bar' as const,
        yaxis: useY2 ? 'y2' : 'y',
        marker: { color: COLORS[i % COLORS.length] },
      };
    } else {
      return {
        x: data.times,
        y: col.values,
        name: col.name,
        type: 'scatter' as const,
        mode: 'lines' as const,
        yaxis: useY2 ? 'y2' : 'y',
        line: { width: 2, color: COLORS[i % COLORS.length] },
      };
    }
  });

  const chartTitle = title ?? `${data.location} Budget`;
  const xAxisType = detectXAxisType(data.times);
  const xAxisTitle = xAxisLabel ?? (xAxisType === 'date' ? 'Date' : 'Year');
  const layoutExtra: Record<string, unknown> = {};
  if (dualAxis && data.columns.length >= 2) {
    layoutExtra.yaxis2 = {
      title: { text: data.columns[1].name },
      overlaying: 'y',
      side: 'right',
    };
    layoutExtra.margin = { l: 70, r: 70, t: 40, b: 50 };
  }

  if (partialYearNote) {
    layoutExtra.annotations = [
      {
        text: partialYearNote,
        xref: 'paper',
        yref: 'paper',
        x: 0.5,
        y: 1.02,
        showarrow: false,
        font: { size: 10, color: '#888' },
      },
    ];
  }

  return (
    <Box sx={{ height: '100%', p: 1 }}>
      <Plot
        data={traces}
        layout={{
          margin: { l: 70, r: 30, t: 40, b: 50 },
          title: { text: chartTitle, font: { size: 14 } },
          xaxis: { title: { text: xAxisTitle }, type: xAxisType },
          yaxis: { title: { text: yAxisLabel ?? data.columns[0]?.name ?? 'Value' } },
          barmode: chartType === 'bar' ? 'group' : undefined,
          legend: { orientation: 'h', y: -0.15 },
          autosize: true,
          ...layoutExtra,
        }}
        config={{ responsive: true, displaylogo: false }}
        style={{ width: '100%', height: '100%' }}
        useResizeHandler
      />
    </Box>
  );
}
