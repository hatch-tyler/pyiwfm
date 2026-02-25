/**
 * ZoneUploadDialog: two-step dialog for importing zones from shapefile/GeoJSON.
 * Step 1: drag-drop file upload
 * Step 2: select name field + preview zones + confirm
 */

import { useState } from 'react';
import Dialog from '@mui/material/Dialog';
import DialogTitle from '@mui/material/DialogTitle';
import DialogContent from '@mui/material/DialogContent';
import DialogActions from '@mui/material/DialogActions';
import Button from '@mui/material/Button';
import Typography from '@mui/material/Typography';
import Box from '@mui/material/Box';
import MenuItem from '@mui/material/MenuItem';
import Select from '@mui/material/Select';
import FormControl from '@mui/material/FormControl';
import InputLabel from '@mui/material/InputLabel';
import Table from '@mui/material/Table';
import TableHead from '@mui/material/TableHead';
import TableBody from '@mui/material/TableBody';
import TableRow from '@mui/material/TableRow';
import TableCell from '@mui/material/TableCell';
import CircularProgress from '@mui/material/CircularProgress';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import type { ZoneInfo, ZoneUploadPreview } from '../../api/client';
import { uploadZoneFile } from '../../api/client';

interface ZoneUploadDialogProps {
  open: boolean;
  onClose: () => void;
  onZonesImported: (zones: ZoneInfo[]) => void;
}

export function ZoneUploadDialog({ open, onClose, onZonesImported }: ZoneUploadDialogProps) {
  const [step, setStep] = useState<1 | 2>(1);
  const [dragOver, setDragOver] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [preview, setPreview] = useState<ZoneUploadPreview | null>(null);
  const [nameField, setNameField] = useState('');
  const [file, setFile] = useState<File | null>(null);

  const reset = () => {
    setStep(1);
    setDragOver(false);
    setUploading(false);
    setError(null);
    setPreview(null);
    setNameField('');
    setFile(null);
  };

  const handleClose = () => {
    reset();
    onClose();
  };

  const handleUpload = async (f: File) => {
    setFile(f);
    setUploading(true);
    setError(null);
    try {
      const result = await uploadZoneFile(f);
      setPreview(result);
      setNameField(result.default_field);
      setStep(2);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
    } finally {
      setUploading(false);
    }
  };

  const handleFieldChange = async (field: string) => {
    setNameField(field);
    if (!file) return;
    setUploading(true);
    setError(null);
    try {
      const result = await uploadZoneFile(file, field);
      setPreview(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Re-upload failed');
    } finally {
      setUploading(false);
    }
  };

  const handleConfirm = () => {
    if (!preview) return;
    const zones: ZoneInfo[] = preview.zones.map((z) => ({
      id: z.id,
      name: z.name,
      elements: z.elements,
    }));
    onZonesImported(zones);
    handleClose();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const f = e.dataTransfer.files[0];
    if (f) handleUpload(f);
  };

  return (
    <Dialog open={open} onClose={handleClose} maxWidth="sm" fullWidth>
      <DialogTitle>
        {step === 1 ? 'Upload Zone File' : 'Configure Zone Import'}
      </DialogTitle>
      <DialogContent>
        {step === 1 ? (
          <>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              Upload a shapefile (.zip) or GeoJSON file. Each polygon becomes a zone,
              and elements whose centroids fall inside will be assigned.
            </Typography>

            <Box
              onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={handleDrop}
              sx={{
                border: '2px dashed',
                borderColor: dragOver ? 'primary.main' : 'divider',
                borderRadius: 2,
                p: 4,
                textAlign: 'center',
                bgcolor: dragOver ? 'action.hover' : 'transparent',
                cursor: 'pointer',
              }}
            >
              {uploading ? (
                <CircularProgress size={48} />
              ) : (
                <>
                  <CloudUploadIcon sx={{ fontSize: 48, color: 'text.secondary', mb: 1 }} />
                  <Typography>
                    Drop file here or{' '}
                    <Button component="label" size="small">
                      browse
                      <input
                        type="file"
                        accept=".zip,.geojson,.json"
                        hidden
                        onChange={(e) => {
                          const f = e.target.files?.[0];
                          if (f) handleUpload(f);
                          e.target.value = '';
                        }}
                      />
                    </Button>
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    .zip (shapefile), .geojson, .json
                  </Typography>
                </>
              )}
            </Box>
          </>
        ) : preview ? (
          <>
            <FormControl fullWidth size="small" sx={{ mb: 2 }}>
              <InputLabel>Zone name field</InputLabel>
              <Select
                value={nameField}
                label="Zone name field"
                onChange={(e) => handleFieldChange(e.target.value)}
                disabled={uploading}
              >
                {preview.fields.map((f) => (
                  <MenuItem key={f} value={f}>{f}</MenuItem>
                ))}
              </Select>
            </FormControl>

            {uploading ? (
              <Box sx={{ display: 'flex', justifyContent: 'center', py: 2 }}>
                <CircularProgress size={32} />
              </Box>
            ) : (
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Zone</TableCell>
                    <TableCell align="right">Elements</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {preview.zones.map((z) => (
                    <TableRow key={z.id}>
                      <TableCell>{z.name}</TableCell>
                      <TableCell align="right">{z.n_elements}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            )}

            {preview.zones.length === 0 && !uploading && (
              <Typography color="text.secondary" sx={{ mt: 1 }}>
                No zones found. Check that the file contains polygon geometries.
              </Typography>
            )}
          </>
        ) : null}

        {error && (
          <Typography color="error" sx={{ mt: 2 }}>{error}</Typography>
        )}
      </DialogContent>
      <DialogActions>
        {step === 2 && (
          <Button onClick={reset}>Back</Button>
        )}
        <Button onClick={handleClose}>Cancel</Button>
        {step === 2 && (
          <Button
            variant="contained"
            onClick={handleConfirm}
            disabled={!preview || preview.zones.length === 0 || uploading}
          >
            Import Zones
          </Button>
        )}
      </DialogActions>
    </Dialog>
  );
}
