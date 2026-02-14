/**
 * Stream reach longitudinal profile chart.
 * Shows bed elevation and ground surface elevation along a reach.
 */

import Box from '@mui/material/Box';
import Paper from '@mui/material/Paper';
import Typography from '@mui/material/Typography';
import IconButton from '@mui/material/IconButton';
import CloseIcon from '@mui/icons-material/Close';
import Plot from 'react-plotly.js';
import type { ReachProfileData } from '../../api/client';

interface ReachProfileChartProps {
  data: ReachProfileData;
  onClose: () => void;
}

export function ReachProfileChart({ data, onClose }: ReachProfileChartProps) {
  const distances = data.nodes.map(n => n.distance);
  const bedElevs = data.nodes.map(n => n.bed_elev);
  const gsElevs = data.nodes.map(n => n.ground_surface_elev);

  const traces: Plotly.Data[] = [
    {
      x: distances,
      y: gsElevs,
      type: 'scatter',
      mode: 'lines',
      name: 'Ground Surface',
      line: { color: '#8B4513', width: 2 },
      fill: 'tonexty',
      fillcolor: 'rgba(139, 69, 19, 0.1)',
    },
    {
      x: distances,
      y: bedElevs,
      type: 'scatter',
      mode: 'lines+markers',
      name: 'Bed Elevation',
      line: { color: '#1f77b4', width: 2 },
      marker: { size: 4 },
      fill: 'tozeroy',
      fillcolor: 'rgba(31, 119, 180, 0.15)',
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
        height: 280,
        display: 'flex',
        flexDirection: 'column',
        zIndex: 10,
      }}
    >
      <Box sx={{ display: 'flex', alignItems: 'center', px: 2, pt: 1 }}>
        <Typography variant="subtitle2" sx={{ flexGrow: 1 }}>
          {data.name} &mdash; Profile ({data.total_length.toFixed(0)} ft, {data.n_nodes} nodes)
        </Typography>
        <IconButton size="small" onClick={onClose}><CloseIcon /></IconButton>
      </Box>
      <Box sx={{ flexGrow: 1, px: 1, pb: 1 }}>
        <Plot
          data={traces}
          layout={{
            margin: { l: 60, r: 20, t: 10, b: 40 },
            xaxis: { title: { text: 'Distance from Upstream (ft)' } },
            yaxis: { title: { text: 'Elevation (ft)' } },
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
