/**
 * Water balance Sankey diagram showing inter-component flows.
 */

import { useState, useEffect } from 'react';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import CircularProgress from '@mui/material/CircularProgress';
import Plot from 'react-plotly.js';
import { fetchWaterBalance } from '../../api/client';
import type { WaterBalanceData } from '../../api/client';

export function WaterBalanceSankey() {
  const [data, setData] = useState<WaterBalanceData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    fetchWaterBalance()
      .then((d) => {
        setData(d);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  if (loading) {
    return (
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
        <CircularProgress />
        <Typography sx={{ ml: 2 }}>Loading water balance...</Typography>
      </Box>
    );
  }

  if (error || !data) {
    return (
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
        <Typography color="text.secondary">
          {error || 'No water balance data available.'}
        </Typography>
      </Box>
    );
  }

  if (data.links.length === 0) {
    return (
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
        <Typography color="text.secondary">
          No flow data available for Sankey diagram.
        </Typography>
      </Box>
    );
  }

  const traces: Plotly.Data[] = [
    {
      type: 'sankey',
      orientation: 'h',
      node: {
        pad: 15,
        thickness: 20,
        line: { color: 'black', width: 0.5 },
        label: data.nodes,
      },
      link: {
        source: data.links.map(l => l.source),
        target: data.links.map(l => l.target),
        value: data.links.map(l => l.value),
        label: data.links.map(l => l.label),
      },
    } as Plotly.Data,
  ];

  return (
    <Box sx={{ height: '100%', p: 2 }}>
      <Plot
        data={traces}
        layout={{
          title: { text: 'Water Balance' },
          font: { size: 11 },
          margin: { l: 20, r: 20, t: 40, b: 20 },
          autosize: true,
        }}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: '100%', height: '100%' }}
      />
    </Box>
  );
}
