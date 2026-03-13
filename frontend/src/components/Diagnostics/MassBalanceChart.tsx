/**
 * Mass balance error timeseries chart.
 */

import Box from '@mui/material/Box';
import Plot from 'react-plotly.js';
import type { MassBalanceRecord } from '../../api/client';

export function MassBalanceChart({ records }: { records: MassBalanceRecord[] }) {
  // Group by component
  const byComponent: Record<string, MassBalanceRecord[]> = {};
  for (const r of records) {
    const key = r.component || 'total';
    if (!byComponent[key]) byComponent[key] = [];
    byComponent[key].push(r);
  }

  const traces = Object.entries(byComponent).map(([comp, recs]) => ({
    x: recs.map((r) => r.date || String(r.timestep_index)),
    y: recs.map((r) => r.error_percent ?? r.error_value),
    type: 'scatter' as const,
    mode: 'lines+markers' as const,
    name: comp,
    marker: { size: 3 },
    hovertemplate: `${comp}<br>Timestep: %{x}<br>Error: %{y:.4f}%<extra></extra>`,
  }));

  return (
    <Box sx={{ width: '100%', height: 400 }}>
      <Plot
        data={traces}
        layout={{
          title: { text: 'Mass Balance Error Over Time' },
          xaxis: { title: { text: 'Timestep' }, tickangle: -45 },
          yaxis: { title: { text: 'Error (%)' } },
          margin: { t: 40, b: 80, l: 60, r: 20 },
          height: 380,
          showlegend: Object.keys(byComponent).length > 1,
          legend: { orientation: 'h', y: -0.3 },
        }}
        config={{ responsive: true, displaylogo: false }}
        style={{ width: '100%', height: '100%' }}
        useResizeHandler
      />
    </Box>
  );
}
