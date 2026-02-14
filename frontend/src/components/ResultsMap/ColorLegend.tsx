/**
 * Color bar legend overlay for the results map.
 * Supports both sequential (Viridis) and diverging (red-white-blue) scales.
 */

import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import { VIRIDIS_COLORS } from './mapStyles';

/** Diverging palette: red → white → blue */
const DIVERGING_COLORS: [number, number, number][] = [
  [178, 24, 43],
  [214, 96, 77],
  [244, 165, 130],
  [253, 219, 199],
  [247, 247, 247],
  [209, 229, 240],
  [146, 197, 222],
  [67, 147, 195],
  [33, 102, 172],
];

interface ColorLegendProps {
  min: number;
  max: number;
  label?: string;
  diverging?: boolean;
}

export function ColorLegend({ min, max, label = 'Head (ft)', diverging = false }: ColorLegendProps) {
  const palette = diverging ? DIVERGING_COLORS : VIRIDIS_COLORS;

  const stops = palette.map((c, i) => {
    const pct = (i / (palette.length - 1)) * 100;
    return `rgb(${c[0]},${c[1]},${c[2]}) ${pct}%`;
  }).join(', ');

  const midLabel = diverging ? '0' : ((min + max) / 2).toFixed(0);

  return (
    <Box
      sx={{
        position: 'absolute',
        bottom: 24,
        left: 24,
        width: 240,
        bgcolor: 'rgba(255,255,255,0.9)',
        borderRadius: 1,
        p: 1.5,
        boxShadow: 2,
      }}
    >
      <Typography variant="caption" sx={{ fontWeight: 600, display: 'block', mb: 0.5 }}>
        {label}
      </Typography>
      <Box
        sx={{
          height: 12,
          borderRadius: 0.5,
          background: `linear-gradient(to right, ${stops})`,
        }}
      />
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 0.5 }}>
        <Typography variant="caption">{min.toFixed(diverging ? 1 : 0)}</Typography>
        <Typography variant="caption">{midLabel}</Typography>
        <Typography variant="caption">{max.toFixed(diverging ? 1 : 0)}</Typography>
      </Box>
    </Box>
  );
}
