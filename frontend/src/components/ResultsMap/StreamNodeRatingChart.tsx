/**
 * Stream node rating curve chart: shows stage (elevation) vs. flow (discharge).
 * Displayed when clicking a stream gage location on the map.
 */

import Box from '@mui/material/Box';
import Paper from '@mui/material/Paper';
import Typography from '@mui/material/Typography';
import IconButton from '@mui/material/IconButton';
import CloseIcon from '@mui/icons-material/Close';
import Plot from 'react-plotly.js';
import type { StreamNodeRatingData } from '../../api/client';

interface StreamNodeRatingChartProps {
  data: StreamNodeRatingData;
  onClose: () => void;
}

export function StreamNodeRatingChart({ data, onClose }: StreamNodeRatingChartProps) {
  const traces: Plotly.Data[] = [
    {
      x: data.flows,
      y: data.stages,
      type: 'scatter',
      mode: 'lines+markers',
      name: 'Rating Curve',
      line: { color: '#1f77b4' },
      marker: { size: 4 },
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
          Stream Node {data.stream_node_id} &mdash; Rating Curve
          ({data.n_points} points, bed elev: {data.bottom_elev.toFixed(1)} ft)
        </Typography>
        <IconButton size="small" onClick={onClose}><CloseIcon /></IconButton>
      </Box>
      <Box sx={{ flexGrow: 1, px: 1, pb: 1 }}>
        <Plot
          data={traces}
          layout={{
            margin: { l: 60, r: 20, t: 10, b: 50 },
            xaxis: { title: { text: 'Flow (cfs)' } },
            yaxis: { title: { text: 'Stage (ft)' } },
            showlegend: false,
            autosize: true,
          }}
          config={{ displayModeBar: false, responsive: true }}
          style={{ width: '100%', height: '100%' }}
        />
      </Box>
    </Paper>
  );
}
