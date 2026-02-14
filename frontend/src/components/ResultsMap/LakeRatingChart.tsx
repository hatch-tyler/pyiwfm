/**
 * Lake rating curve chart: shows elevation vs. area and volume.
 * Uses Plotly with dual y-axes.
 */

import Box from '@mui/material/Box';
import Paper from '@mui/material/Paper';
import Typography from '@mui/material/Typography';
import IconButton from '@mui/material/IconButton';
import CloseIcon from '@mui/icons-material/Close';
import Plot from 'react-plotly.js';
import type { LakeRatingData } from '../../api/client';

interface LakeRatingChartProps {
  data: LakeRatingData;
  onClose: () => void;
}

export function LakeRatingChart({ data, onClose }: LakeRatingChartProps) {
  const traces: Plotly.Data[] = [
    {
      x: data.elevations,
      y: data.areas,
      type: 'scatter',
      mode: 'lines+markers',
      name: 'Area',
      line: { color: '#1f77b4' },
      marker: { size: 4 },
      yaxis: 'y',
    },
    {
      x: data.elevations,
      y: data.volumes,
      type: 'scatter',
      mode: 'lines+markers',
      name: 'Volume',
      line: { color: '#ff7f0e' },
      marker: { size: 4 },
      yaxis: 'y2',
    },
  ];

  return (
    <Paper
      elevation={3}
      sx={{
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
          {data.name} &mdash; Rating Curve ({data.n_points} points)
        </Typography>
        <IconButton size="small" onClick={onClose}><CloseIcon /></IconButton>
      </Box>
      <Box sx={{ flexGrow: 1, px: 1, pb: 1 }}>
        <Plot
          data={traces}
          layout={{
            margin: { l: 60, r: 60, t: 10, b: 40 },
            xaxis: { title: { text: 'Elevation (ft)' } },
            yaxis: { title: { text: 'Area (sq ft)' }, side: 'left' },
            yaxis2: {
              title: { text: 'Volume (cu ft)' },
              side: 'right',
              overlaying: 'y',
            },
            showlegend: true,
            legend: { orientation: 'h', y: 1.12 },
            autosize: true,
          }}
          config={{ displayModeBar: false, responsive: true }}
          style={{ width: '100%', height: '100%' }}
        />
      </Box>
    </Paper>
  );
}
