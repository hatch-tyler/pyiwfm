/**
 * Stacked bar chart for diversion balance:
 * Actual Diversion = Delivery + Recoverable Loss + Non-Recoverable Loss
 *
 * Shows a stacked bar of the component parts with a dashed line overlay
 * for the Actual Diversion total.
 */

import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import IconButton from '@mui/material/IconButton';
import OpenInFullIcon from '@mui/icons-material/OpenInFull';
import Plot from 'react-plotly.js';
import type { BudgetData } from '../../api/client';

/**
 * Detect whether time labels are ISO date strings, bare year strings,
 * or category labels. Returns 'date' for ISO/year strings, 'category' otherwise.
 */
function detectXAxisType(times: string[]): 'date' | 'category' {
  if (times.length === 0) return 'date';
  const first = times[0];
  if (first.includes('T') || (first.includes('-') && first.length >= 7)) return 'date';
  if (/^\d{4}$/.test(first)) return 'date';
  return 'category';
}

interface DiversionBalanceChartProps {
  data: BudgetData;
  yAxisLabel: string;
  title?: string;
  xAxisLabel?: string;
  partialYearNote?: string;
  onExpand?: () => void;
}

/** Find a column by partial name match (case-insensitive). */
function findCol(data: BudgetData, pattern: string): { name: string; values: number[] } | null {
  const col = data.columns.find((c) => c.name.toLowerCase().includes(pattern.toLowerCase()));
  return col ? { name: col.name, values: col.values } : null;
}

export function DiversionBalanceChart({ data, yAxisLabel, title, xAxisLabel, partialYearNote, onExpand }: DiversionBalanceChartProps) {
  const delivery = findCol(data, 'delivery');
  const recoverable = findCol(data, 'recoverable loss');
  const nonRecoverable = findCol(data, 'non-recoverable loss');
  const actualDiv = findCol(data, 'actual diversion');

  const traces: Plotly.Data[] = [];

  // Stacked bars for components
  if (delivery) {
    traces.push({
      x: data.times,
      y: delivery.values,
      name: delivery.name,
      type: 'bar',
      marker: { color: '#4caf50' },
    });
  }
  if (recoverable) {
    traces.push({
      x: data.times,
      y: recoverable.values,
      name: recoverable.name,
      type: 'bar',
      marker: { color: '#ff9800' },
    });
  }
  if (nonRecoverable) {
    traces.push({
      x: data.times,
      y: nonRecoverable.values,
      name: nonRecoverable.name,
      type: 'bar',
      marker: { color: '#f44336' },
    });
  }

  // Dashed line overlay for Actual Diversion
  if (actualDiv) {
    traces.push({
      x: data.times,
      y: actualDiv.values,
      name: actualDiv.name,
      type: 'scatter',
      mode: 'lines',
      line: { color: '#1a237e', width: 2, dash: 'dash' },
    });
  }

  const chartTitle = title ?? 'Diversion Balance';
  const xAxisType = detectXAxisType(data.times);
  const xTitle = xAxisLabel ?? (xAxisType === 'date' ? 'Date' : 'Year');

  const annotations: Plotly.Layout['annotations'] = [];
  if (partialYearNote) {
    annotations.push({
      text: partialYearNote,
      xref: 'paper',
      yref: 'paper',
      x: 0.5,
      y: 1.02,
      showarrow: false,
      font: { size: 10, color: '#888' },
    });
  }

  return (
    <Box sx={{ height: '100%', p: 1, position: 'relative' }}>
      {onExpand && (
        <IconButton
          size="small"
          onClick={onExpand}
          title="Expand chart"
          sx={{ position: 'absolute', top: 4, right: 4, zIndex: 5, opacity: 0.6, '&:hover': { opacity: 1 } }}
        >
          <OpenInFullIcon fontSize="small" />
        </IconButton>
      )}
      {traces.length === 0 ? (
        <Typography color="text.secondary" sx={{ textAlign: 'center', mt: 4 }}>
          No diversion balance columns found.
        </Typography>
      ) : (
        <Plot
          data={traces}
          layout={{
            margin: { l: 70, r: 30, t: 50, b: 80 },
            title: { text: chartTitle, font: { size: 14 } },
            xaxis: { title: { text: xTitle }, type: xAxisType },
            yaxis: { title: { text: yAxisLabel } },
            barmode: 'stack',
            legend: { orientation: 'h', y: -0.22, xanchor: 'center', x: 0.5 },
            autosize: true,
            annotations,
          }}
          config={{ responsive: true, displaylogo: false }}
          style={{ width: '100%', height: '100%' }}
          useResizeHandler
        />
      )}
    </Box>
  );
}
