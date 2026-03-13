/**
 * Model comparison dialog — compare the loaded model with another model.
 */

import { useState } from 'react';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Dialog from '@mui/material/Dialog';
import DialogTitle from '@mui/material/DialogTitle';
import DialogContent from '@mui/material/DialogContent';
import DialogActions from '@mui/material/DialogActions';
import TextField from '@mui/material/TextField';
import Typography from '@mui/material/Typography';
import CircularProgress from '@mui/material/CircularProgress';
import Alert from '@mui/material/Alert';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Paper from '@mui/material/Paper';
import Chip from '@mui/material/Chip';
import { compareModels } from '../../api/client';
import type { ModelComparisonResult } from '../../api/client';

interface ModelComparisonProps {
  open: boolean;
  onClose: () => void;
}

export function ModelComparison({ open, onClose }: ModelComparisonProps) {
  const [modelPath, setModelPath] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ModelComparisonResult | null>(null);

  const handleCompare = async () => {
    if (!modelPath.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await compareModels(modelPath.trim());
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Comparison failed');
    } finally {
      setLoading(false);
    }
  };

  const handleClose = () => {
    setModelPath('');
    setError(null);
    setResult(null);
    onClose();
  };

  return (
    <Dialog open={open} onClose={handleClose} maxWidth="md" fullWidth>
      <DialogTitle>Compare Models</DialogTitle>
      <DialogContent>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          Compare the currently loaded model with another IWFM model to identify
          differences in mesh, stratigraphy, and parameters.
        </Typography>

        <Box sx={{ display: 'flex', gap: 1, mb: 2 }}>
          <TextField
            fullWidth
            size="small"
            label="Path to other model (Preprocessor.in)"
            value={modelPath}
            onChange={(e) => setModelPath(e.target.value)}
            placeholder="/path/to/other/model/Preprocessor.in"
            disabled={loading}
          />
          <Button
            variant="contained"
            onClick={handleCompare}
            disabled={loading || !modelPath.trim()}
            sx={{ whiteSpace: 'nowrap' }}
          >
            {loading ? <CircularProgress size={20} /> : 'Compare'}
          </Button>
        </Box>

        {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}

        {result && (
          <Box>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
              <Typography variant="subtitle1">Comparison Result</Typography>
              <Chip
                label={result.is_identical ? 'Identical' : 'Differences Found'}
                color={result.is_identical ? 'success' : 'warning'}
                size="small"
              />
            </Box>

            {/* Statistics */}
            {result.statistics && (
              <Box sx={{ display: 'flex', gap: 3, mb: 2 }}>
                <Typography variant="body2">
                  Total: <strong>{result.statistics.total_differences}</strong>
                </Typography>
                <Typography variant="body2" color="success.main">
                  Added: <strong>{result.statistics.added}</strong>
                </Typography>
                <Typography variant="body2" color="error.main">
                  Removed: <strong>{result.statistics.removed}</strong>
                </Typography>
                <Typography variant="body2" color="warning.main">
                  Modified: <strong>{result.statistics.modified}</strong>
                </Typography>
              </Box>
            )}

            {/* Mesh diff */}
            {result.mesh_diff && !result.mesh_diff.is_identical && (
              <Box sx={{ mb: 2 }}>
                <Typography variant="subtitle2" sx={{ mb: 1 }}>Mesh Differences</Typography>
                <TableContainer component={Paper} variant="outlined">
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Category</TableCell>
                        <TableCell align="right">Added</TableCell>
                        <TableCell align="right">Removed</TableCell>
                        <TableCell align="right">Modified</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      <TableRow>
                        <TableCell>Nodes</TableCell>
                        <TableCell align="right">{result.mesh_diff.nodes_added}</TableCell>
                        <TableCell align="right">{result.mesh_diff.nodes_removed}</TableCell>
                        <TableCell align="right">{result.mesh_diff.nodes_modified}</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Elements</TableCell>
                        <TableCell align="right">{result.mesh_diff.elements_added}</TableCell>
                        <TableCell align="right">{result.mesh_diff.elements_removed}</TableCell>
                        <TableCell align="right">{result.mesh_diff.elements_modified}</TableCell>
                      </TableRow>
                    </TableBody>
                  </Table>
                </TableContainer>
              </Box>
            )}

            {/* Stratigraphy diff */}
            {result.stratigraphy_diff && !result.stratigraphy_diff.is_identical && (
              <Box sx={{ mb: 2 }}>
                <Typography variant="subtitle2" sx={{ mb: 1 }}>
                  Stratigraphy Differences ({result.stratigraphy_diff.n_differences})
                </Typography>
                {result.stratigraphy_diff.items && result.stratigraphy_diff.items.length > 0 && (
                  <TableContainer component={Paper} variant="outlined" sx={{ maxHeight: 300 }}>
                    <Table size="small" stickyHeader>
                      <TableHead>
                        <TableRow>
                          <TableCell>Path</TableCell>
                          <TableCell>Type</TableCell>
                          <TableCell>Old Value</TableCell>
                          <TableCell>New Value</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {result.stratigraphy_diff.items.slice(0, 50).map((item, i) => (
                          <TableRow key={i}>
                            <TableCell>
                              <Typography variant="caption" sx={{ fontFamily: 'monospace' }}>
                                {item.path}
                              </Typography>
                            </TableCell>
                            <TableCell>
                              <Chip
                                label={item.diff_type}
                                size="small"
                                color={
                                  item.diff_type === 'ADDED' ? 'success'
                                    : item.diff_type === 'REMOVED' ? 'error'
                                    : 'warning'
                                }
                              />
                            </TableCell>
                            <TableCell>{String(item.old_value ?? '-')}</TableCell>
                            <TableCell>{String(item.new_value ?? '-')}</TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                )}
              </Box>
            )}

            {result.is_identical && (
              <Alert severity="success">
                The two models are structurally identical (mesh and stratigraphy).
              </Alert>
            )}
          </Box>
        )}
      </DialogContent>
      <DialogActions>
        <Button onClick={handleClose}>Close</Button>
      </DialogActions>
    </Dialog>
  );
}
