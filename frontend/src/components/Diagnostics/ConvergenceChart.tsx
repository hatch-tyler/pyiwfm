/**
 * Convergence iteration count chart (timestep vs iterations).
 */

import Box from '@mui/material/Box';
import Plot from 'react-plotly.js';
import type { ConvergenceRecord } from '../../api/client';

export function ConvergenceChart({ records }: { records: ConvergenceRecord[] }) {
  const x = records.map((r) => r.date || String(r.timestep_index));
  const y = records.map((r) => r.iteration_count);
  const colors = records.map((r) => (r.convergence_achieved ? '#1976d2' : '#d32f2f'));

  return (
    <Box sx={{ width: '100%', height: 400 }}>
      <Plot
        data={[
          {
            x,
            y,
            type: 'bar',
            marker: { color: colors },
            name: 'Iterations',
            hovertemplate: 'Timestep: %{x}<br>Iterations: %{y}<extra></extra>',
          },
        ]}
        layout={{
          title: { text: 'Convergence Iterations per Timestep' },
          xaxis: { title: { text: 'Timestep' }, tickangle: -45 },
          yaxis: { title: { text: 'Iteration Count' }, rangemode: 'tozero' },
          margin: { t: 40, b: 80, l: 60, r: 20 },
          height: 380,
          bargap: 0.1,
        }}
        config={{ responsive: true, displaylogo: false }}
        style={{ width: '100%', height: '100%' }}
        useResizeHandler
      />
    </Box>
  );
}
