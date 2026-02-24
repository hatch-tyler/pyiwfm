/**
 * Storage change bar chart with conditional per-bar coloring.
 *
 * Each bar is colored based on its value:
 *   - Blue (#1565c0) for positive values (storage gain)
 *   - Red (#c62828) for negative values (storage loss)
 */

import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import IconButton from '@mui/material/IconButton';
import OpenInFullIcon from '@mui/icons-material/OpenInFull';
import Plot from 'react-plotly.js';
import type { BudgetData } from '../../api/client';

function detectXAxisType(times: string[]): 'date' | 'category' {
  if (times.length === 0) return 'date';
  const first = times[0];
  if (first.includes('T') || (first.includes('-') && first.length >= 7)) return 'date';
  if (/^\d{4}$/.test(first)) return 'date';
  return 'category';
}

const GAIN_COLOR = '#1565c0';
const LOSS_COLOR = '#c62828';

interface StorageChangeChartProps {
  data: BudgetData;
  yAxisLabel: string;
  title?: string;
  xAxisLabel?: string;
  partialYearNote?: string;
  onExpand?: () => void;
}

export function StorageChangeChart({
  data,
  yAxisLabel,
  title,
  xAxisLabel,
  partialYearNote,
  onExpand,
}: StorageChangeChartProps) {
  const traces: Plotly.Data[] = data.columns.map((col) => {
    const barColors = col.values.map((v) => (v >= 0 ? GAIN_COLOR : LOSS_COLOR));

    return {
      x: data.times,
      y: col.values,
      name: col.name,
      type: 'bar' as const,
      marker: { color: barColors },
    };
  });

  const chartTitle = title ?? 'Change in Storage';
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
          No storage columns found.
        </Typography>
      ) : (
        <Plot
          data={traces}
          layout={{
            margin: { l: 70, r: 30, t: 50, b: 80 },
            title: { text: chartTitle, font: { size: 14 } },
            xaxis: { title: { text: xTitle }, type: xAxisType },
            yaxis: { title: { text: yAxisLabel } },
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
