/**
 * Small Plotly chart for diversion timeseries (max diversion + delivery).
 */

import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Plot from 'react-plotly.js';
import type { DiversionTimeseries } from '../../api/client';

interface DiversionTimeseriesChartProps {
  data: DiversionTimeseries;
  name: string;
}

export function DiversionTimeseriesChart({ data, name }: DiversionTimeseriesChartProps) {
  const traces: Plotly.Data[] = [];

  if (data.max_diversion && data.max_diversion.length > 0) {
    traces.push({
      x: data.times,
      y: data.max_diversion,
      type: 'scatter',
      mode: 'lines',
      name: 'Max Diversion',
      line: { color: '#e65100', width: 2 },
    });
  }

  if (data.delivery && data.delivery.length > 0) {
    traces.push({
      x: data.times,
      y: data.delivery,
      type: 'scatter',
      mode: 'lines',
      name: 'Delivery',
      line: { color: '#1565c0', width: 2, dash: 'dash' },
    });
  }

  if (traces.length === 0) {
    return (
      <Typography variant="body2" color="text.secondary" sx={{ p: 1 }}>
        No timeseries data available.
      </Typography>
    );
  }

  return (
    <Box sx={{ width: '100%' }}>
      <Plot
        data={traces}
        layout={{
          title: { text: name, font: { size: 12 } },
          height: 220,
          margin: { l: 50, r: 20, t: 35, b: 40 },
          xaxis: { title: { text: '' }, type: 'date' },
          yaxis: { title: { text: 'Rate' } },
          legend: { orientation: 'h', y: -0.2, x: 0.5, xanchor: 'center' },
          showlegend: true,
        }}
        config={{ responsive: true, displayModeBar: false }}
        style={{ width: '100%' }}
      />
    </Box>
  );
}
