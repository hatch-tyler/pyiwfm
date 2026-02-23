/**
 * Multi-hydrograph comparison chart.
 * Overlays multiple hydrograph time series on a single Plotly chart.
 */

import { useState } from 'react';
import Box from '@mui/material/Box';
import Paper from '@mui/material/Paper';
import Typography from '@mui/material/Typography';
import IconButton from '@mui/material/IconButton';
import CloseIcon from '@mui/icons-material/Close';
import FullscreenIcon from '@mui/icons-material/Fullscreen';
import FullscreenExitIcon from '@mui/icons-material/FullscreenExit';
import Plot from 'react-plotly.js';
import type { HydrographData } from '../../api/client';

/** Distinct colors for comparison series */
const SERIES_COLORS = [
  '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
  '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
  '#bcbd22', '#17becf',
];

interface ComparisonChartProps {
  series: HydrographData[];
  onClose: () => void;
}

export function ComparisonChart({ series, onClose }: ComparisonChartProps) {
  const [expanded, setExpanded] = useState(false);

  if (series.length === 0) return null;

  const traces: Plotly.Data[] = series.map((s, i) => ({
    x: s.times,
    y: s.values,
    type: 'scatter' as const,
    mode: 'lines' as const,
    name: s.name,
    line: { color: SERIES_COLORS[i % SERIES_COLORS.length], width: 1.5 },
  }));

  const units = series[0]?.units ?? '';
  const yLabel = series[0]?.type === 'gw' ? `Head (${units})` : `Flow (${units})`;

  return (
    <Paper
      elevation={3}
      sx={expanded ? {
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        display: 'flex',
        flexDirection: 'column',
        zIndex: 1300,
      } : {
        position: 'absolute',
        bottom: 0,
        left: 0,
        right: 280,
        height: 300,
        display: 'flex',
        flexDirection: 'column',
        zIndex: 10,
      }}
    >
      <Box sx={{ display: 'flex', alignItems: 'center', px: 2, pt: 1 }}>
        <Typography variant="subtitle2" sx={{ flexGrow: 1 }}>
          Hydrograph Comparison ({series.length} locations)
        </Typography>
        <IconButton size="small" onClick={() => setExpanded(!expanded)} title={expanded ? 'Exit fullscreen' : 'Fullscreen'}>
          {expanded ? <FullscreenExitIcon /> : <FullscreenIcon />}
        </IconButton>
        <IconButton size="small" onClick={onClose}><CloseIcon /></IconButton>
      </Box>
      <Box sx={{ flexGrow: 1, px: 1, pb: 1 }}>
        <Plot
          data={traces}
          layout={{
            margin: { l: 60, r: 20, t: 10, b: 40 },
            xaxis: { title: { text: 'Date' } },
            yaxis: { title: { text: yLabel } },
            showlegend: true,
            legend: { orientation: 'h', y: 1.15, font: { size: 10 } },
            autosize: true,
          }}
          config={{ displayModeBar: false, responsive: true }}
          style={{ width: '100%', height: '100%' }}
        />
      </Box>
    </Paper>
  );
}
