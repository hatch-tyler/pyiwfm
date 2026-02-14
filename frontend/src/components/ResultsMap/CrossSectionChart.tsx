/**
 * Cross-section chart panel: shows a vertical transect between two map points.
 * Uses Plotly to render layer geometry along the distance axis.
 */

import { useMemo } from 'react';
import Box from '@mui/material/Box';
import Paper from '@mui/material/Paper';
import Typography from '@mui/material/Typography';
import IconButton from '@mui/material/IconButton';
import CloseIcon from '@mui/icons-material/Close';
import Plot from 'react-plotly.js';
import type { CrossSectionData } from '../../api/client';

interface CrossSectionChartProps {
  data: CrossSectionData;
  onClose: () => void;
}

/** Distinct colors for each layer */
const LAYER_COLORS = [
  '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
  '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
  '#bcbd22', '#17becf',
];

export function CrossSectionChart({ data, onClose }: CrossSectionChartProps) {
  const traces = useMemo(() => {
    if (data.n_cells === 0) return [];

    // Extract cell vertices from flat polys array.
    // Format: [nV, v0, v1, ..., nV, v0, v1, ...]
    const cells: number[][] = [];
    let i = 0;
    while (i < data.polys.length) {
      const nv = data.polys[i];
      const verts = data.polys.slice(i + 1, i + 1 + nv);
      cells.push(verts);
      i += nv + 1;
    }

    // Group cells by layer
    const layerCells: Record<number, number[][]> = {};
    cells.forEach((cell, ci) => {
      const layer = ci < data.layer.length ? data.layer[ci] : 1;
      if (!layerCells[layer]) layerCells[layer] = [];
      layerCells[layer].push(cell);
    });

    // Build one scatter trace per layer using cell vertex coordinates
    const result: Plotly.Data[] = [];
    const sortedLayers = Object.keys(layerCells).map(Number).sort((a, b) => a - b);

    for (const layer of sortedLayers) {
      const xVals: (number | null)[] = [];
      const yVals: (number | null)[] = [];

      for (const cell of layerCells[layer]) {
        for (const vi of cell) {
          if (vi >= 0 && vi < data.distance.length) {
            xVals.push(data.distance[vi]);
            // z coordinate is every 3rd value in points array
            yVals.push(data.points[vi * 3 + 2]);
          }
        }
        // Close the polygon and add separator
        if (cell.length > 0 && cell[0] >= 0 && cell[0] < data.distance.length) {
          xVals.push(data.distance[cell[0]]);
          yVals.push(data.points[cell[0] * 3 + 2]);
        }
        xVals.push(null);
        yVals.push(null);
      }

      result.push({
        x: xVals,
        y: yVals,
        type: 'scatter',
        mode: 'lines',
        fill: 'toself',
        name: `Layer ${layer}`,
        line: { color: LAYER_COLORS[(layer - 1) % LAYER_COLORS.length], width: 1 },
        fillcolor: LAYER_COLORS[(layer - 1) % LAYER_COLORS.length] + '40',
      });
    }

    return result;
  }, [data]);

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
          Cross-Section ({data.total_distance.toFixed(0)} ft)
        </Typography>
        <IconButton size="small" onClick={onClose}><CloseIcon /></IconButton>
      </Box>
      <Box sx={{ flexGrow: 1, px: 1, pb: 1 }}>
        {data.n_cells === 0 ? (
          <Typography variant="body2" color="text.secondary" sx={{ p: 2 }}>
            No cells intersected by this cross-section line.
          </Typography>
        ) : (
          <Plot
            data={traces}
            layout={{
              margin: { l: 60, r: 20, t: 10, b: 40 },
              xaxis: { title: { text: 'Distance (ft)' } },
              yaxis: { title: { text: 'Elevation (ft)' } },
              showlegend: true,
              legend: { orientation: 'h', y: 1.12 },
              autosize: true,
            }}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: '100%', height: '100%' }}
          />
        )}
      </Box>
    </Paper>
  );
}
