/**
 * Stacked inflow/outflow bar chart for ZBudget flow components.
 *
 * - Inflow (+) columns: stacked above zero in cool colors (blues/greens)
 * - Outflow (-) columns: negated and stacked below zero in warm colors (reds/oranges)
 * - Bidirectional (+/-) / neutral: values as-is, Plotly stacks by sign
 *
 * Uses `barmode: 'relative'` so positive and negative bars stack independently.
 */

import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import IconButton from '@mui/material/IconButton';
import OpenInFullIcon from '@mui/icons-material/OpenInFull';
import Plot from 'react-plotly.js';
import type { BudgetData } from '../../api/client';
import {
  classifyColumnSign,
  INFLOW_COLORS,
  OUTFLOW_COLORS,
  BIDIRECTIONAL_COLORS,
} from './zbudgetClassifier';

function detectXAxisType(times: string[]): 'date' | 'category' {
  if (times.length === 0) return 'date';
  const first = times[0];
  if (first.includes('T') || (first.includes('-') && first.length >= 7)) return 'date';
  if (/^\d{4}$/.test(first)) return 'date';
  return 'category';
}

interface InflowOutflowChartProps {
  data: BudgetData;
  yAxisLabel: string;
  title?: string;
  xAxisLabel?: string;
  partialYearNote?: string;
  onExpand?: () => void;
}

export function InflowOutflowChart({
  data,
  yAxisLabel,
  title,
  xAxisLabel,
  partialYearNote,
  onExpand,
}: InflowOutflowChartProps) {
  let inflowIdx = 0;
  let outflowIdx = 0;
  let bidiIdx = 0;

  const traces: Plotly.Data[] = data.columns.map((col) => {
    const sign = classifyColumnSign(col.name);

    let color: string;
    let values: number[];

    switch (sign) {
      case 'inflow':
        color = INFLOW_COLORS[inflowIdx % INFLOW_COLORS.length];
        inflowIdx++;
        // Ensure positive
        values = col.values.map((v) => Math.abs(v));
        break;
      case 'outflow':
        color = OUTFLOW_COLORS[outflowIdx % OUTFLOW_COLORS.length];
        outflowIdx++;
        // Ensure negative
        values = col.values.map((v) => -Math.abs(v));
        break;
      case 'bidirectional':
        color = BIDIRECTIONAL_COLORS[bidiIdx % BIDIRECTIONAL_COLORS.length];
        bidiIdx++;
        // Values as-is; Plotly relative mode stacks by sign
        values = col.values;
        break;
      default:
        // neutral — use a gray
        color = '#757575';
        values = col.values;
        break;
    }

    return {
      x: data.times,
      y: values,
      name: col.name,
      type: 'bar' as const,
      marker: { color },
    };
  });

  const chartTitle = title ?? 'Flow Components';
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
          sx={{
            position: 'absolute',
            top: 4,
            right: 4,
            zIndex: 5,
            opacity: 0.6,
            '&:hover': { opacity: 1 },
          }}
        >
          <OpenInFullIcon fontSize="small" />
        </IconButton>
      )}
      {traces.length === 0 ? (
        <Typography color="text.secondary" sx={{ textAlign: 'center', mt: 4 }}>
          No flow columns found.
        </Typography>
      ) : (
        <Plot
          data={traces}
          layout={{
            margin: { l: 70, r: 30, t: 50, b: 80 },
            title: { text: chartTitle, font: { size: 14 } },
            xaxis: { title: { text: xTitle }, type: xAxisType },
            yaxis: { title: { text: yAxisLabel } },
            barmode: 'relative',
            bargap: 0.05,
            legend: { orientation: 'h', y: -0.22, xanchor: 'center', x: 0.5 },
            autosize: true,
            annotations,
            shapes: [
              {
                type: 'line',
                xref: 'paper',
                yref: 'y',
                x0: 0,
                x1: 1,
                y0: 0,
                y1: 0,
                line: { color: '#444', width: 1 },
              },
            ],
          }}
          config={{ responsive: true, displaylogo: false }}
          style={{ width: '100%', height: '100%' }}
          useResizeHandler
        />
      )}
    </Box>
  );
}
