/**
 * Cross-section inset panel showing slice metadata.
 */

import Box from '@mui/material/Box';
import Paper from '@mui/material/Paper';
import Typography from '@mui/material/Typography';
import IconButton from '@mui/material/IconButton';
import CloseIcon from '@mui/icons-material/Close';

import { useViewerStore } from '../../stores/viewerStore';

export default function CrossSectionPanel() {
  const {
    showCrossSection,
    setShowCrossSection,
    sliceAngle,
    slicePosition,
  } = useViewerStore();

  if (!showCrossSection) {
    return null;
  }

  const orientationLabel =
    sliceAngle === 0
      ? 'N-S'
      : sliceAngle === 90
        ? 'E-W'
        : `${sliceAngle}° from N-S`;

  return (
    <Paper
      sx={{
        position: 'absolute',
        bottom: 16,
        right: 16,
        width: 300,
        overflow: 'hidden',
      }}
      elevation={3}
    >
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          px: 1,
          py: 0.5,
          bgcolor: 'primary.main',
          color: 'primary.contrastText',
        }}
      >
        <Typography variant="subtitle2">
          Cross-Section
        </Typography>
        <IconButton
          size="small"
          onClick={() => setShowCrossSection(false)}
          sx={{ color: 'inherit' }}
        >
          <CloseIcon fontSize="small" />
        </IconButton>
      </Box>

      <Box sx={{ p: 1.5 }}>
        <Typography variant="body2">
          Orientation: {orientationLabel}
        </Typography>
        <Typography variant="body2">
          Position: {(slicePosition * 100).toFixed(0)}%
        </Typography>
      </Box>
    </Paper>
  );
}
