/**
 * Plotly time series chart for hydrograph data.
 * Shows simulated line and optional observed scatter overlay.
 */

import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import IconButton from '@mui/material/IconButton';
import CloseIcon from '@mui/icons-material/Close';
import Plot from 'react-plotly.js';
import type { HydrographData, ObservationData } from '../../api/client';

interface HydrographChartProps {
  data: HydrographData;
  observation?: ObservationData | null;
  onClose: () => void;
}

export function HydrographChart({ data, observation, onClose }: HydrographChartProps) {
  const hasStreamDualAxis = data.type === 'stream' && data.flow_values && data.stage_values;

  const traces: Plotly.Data[] = [];

  if (hasStreamDualAxis) {
    // Dual-axis: flow on left (yaxis), stage on right (yaxis2)
    traces.push({
      x: data.times,
      y: data.flow_values,
      type: 'scatter',
      mode: 'lines',
      name: `Flow (${data.flow_units || 'cfs'})`,
      line: { color: '#1976d2', width: 2 },
      yaxis: 'y',
    });
    traces.push({
      x: data.times,
      y: data.stage_values,
      type: 'scatter',
      mode: 'lines',
      name: `Stage (${data.stage_units || 'ft'})`,
      line: { color: '#2e7d32', width: 2 },
      yaxis: 'y2',
    });
  } else {
    // Single axis (GW head or stream flow-only)
    traces.push({
      x: data.times,
      y: data.values,
      type: 'scatter',
      mode: 'lines',
      name: 'Simulated',
      line: { color: '#1976d2', width: 2 },
    });
  }

  if (observation && observation.times.length > 0) {
    traces.push({
      x: observation.times,
      y: observation.values,
      type: 'scatter',
      mode: 'markers',
      name: 'Observed',
      marker: { color: '#d32f2f', size: 5 },
    });
  }

  const layout: Partial<Plotly.Layout> = {
    margin: { l: 60, r: hasStreamDualAxis ? 60 : 20, t: 10, b: 40 },
    xaxis: { title: { text: 'Date' }, type: 'date' },
    yaxis: {
      title: {
        text: hasStreamDualAxis
          ? (data.flow_units || 'cfs')
          : (data.units || 'Value'),
      },
    },
    legend: { orientation: 'h', y: 1.02, x: 0 },
    autosize: true,
  };

  if (hasStreamDualAxis) {
    layout.yaxis2 = {
      title: { text: data.stage_units || 'ft' },
      overlaying: 'y',
      side: 'right',
    };
  }

  return (
    <Box
      sx={{
        height: 280,
        bgcolor: 'background.paper',
        borderTop: '1px solid',
        borderColor: 'divider',
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      <Box sx={{ display: 'flex', alignItems: 'center', px: 2, pt: 0.5 }}>
        <Typography variant="subtitle2" sx={{ flexGrow: 1 }}>
          {data.name} — {data.type.toUpperCase()} Hydrograph
          {data.layer ? ` (Layer ${data.layer})` : ''}
        </Typography>
        <IconButton size="small" onClick={onClose}>
          <CloseIcon fontSize="small" />
        </IconButton>
      </Box>
      <Box sx={{ flexGrow: 1, px: 1, pb: 1 }}>
        <Plot
          data={traces}
          layout={layout}
          config={{ responsive: true, displaylogo: false }}
          style={{ width: '100%', height: '100%' }}
          useResizeHandler
        />
      </Box>
    </Box>
  );
}
