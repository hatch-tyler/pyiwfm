/**
 * Information panel showing model metadata.
 */

import Paper from '@mui/material/Paper';
import Typography from '@mui/material/Typography';
import Box from '@mui/material/Box';
import Chip from '@mui/material/Chip';
import Divider from '@mui/material/Divider';

import { useViewerStore } from '../../stores/viewerStore';

export default function InfoPanel() {
  const { modelInfo, bounds, isLoading, error } = useViewerStore();

  if (isLoading) {
    return (
      <Paper sx={{ p: 2 }} elevation={1}>
        <Typography color="text.secondary">Loading model...</Typography>
      </Paper>
    );
  }

  if (error) {
    return (
      <Paper sx={{ p: 2 }} elevation={1}>
        <Typography color="error">{error}</Typography>
      </Paper>
    );
  }

  if (!modelInfo) {
    return null;
  }

  return (
    <Paper sx={{ p: 2 }} elevation={1}>
      <Typography variant="h6" gutterBottom>
        {modelInfo.name}
      </Typography>

      <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', mb: 2 }}>
        <Chip
          label={`${modelInfo.n_nodes.toLocaleString()} nodes`}
          size="small"
          variant="outlined"
        />
        <Chip
          label={`${modelInfo.n_elements.toLocaleString()} elements`}
          size="small"
          variant="outlined"
        />
        <Chip
          label={`${modelInfo.n_layers} layers`}
          size="small"
          variant="outlined"
        />
      </Box>

      {(modelInfo.has_streams || modelInfo.has_lakes) && (
        <>
          <Divider sx={{ my: 1 }} />
          <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
            {modelInfo.has_streams && modelInfo.n_stream_nodes != null && modelInfo.n_stream_nodes > 0 && (
              <Chip
                label={`${modelInfo.n_stream_nodes} stream nodes`}
                size="small"
                color="primary"
                variant="outlined"
              />
            )}
            {modelInfo.has_lakes && modelInfo.n_lakes != null && modelInfo.n_lakes > 0 && (
              <Chip
                label={`${modelInfo.n_lakes} lakes`}
                size="small"
                color="info"
                variant="outlined"
              />
            )}
          </Box>
        </>
      )}

      {bounds && (
        <>
          <Divider sx={{ my: 1 }} />
          <Typography variant="caption" color="text.secondary">
            Bounds: X [{bounds.xmin.toFixed(0)} - {bounds.xmax.toFixed(0)}],
            Y [{bounds.ymin.toFixed(0)} - {bounds.ymax.toFixed(0)}],
            Z [{bounds.zmin.toFixed(0)} - {bounds.zmax.toFixed(0)}]
          </Typography>
        </>
      )}
    </Paper>
  );
}
