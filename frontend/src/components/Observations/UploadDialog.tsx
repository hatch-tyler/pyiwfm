/**
 * Observation file upload dialog.
 * Allows uploading a CSV file and associating it with a hydrograph location.
 */

import { useState } from 'react';
import Dialog from '@mui/material/Dialog';
import DialogTitle from '@mui/material/DialogTitle';
import DialogContent from '@mui/material/DialogContent';
import DialogActions from '@mui/material/DialogActions';
import Button from '@mui/material/Button';
import Typography from '@mui/material/Typography';
import Box from '@mui/material/Box';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import { uploadObservation, fetchObservations } from '../../api/client';
import { useViewerStore } from '../../stores/viewerStore';

interface UploadDialogProps {
  open: boolean;
  onClose: () => void;
}

export function UploadDialog({ open, onClose }: UploadDialogProps) {
  const { setObservations } = useViewerStore();
  const [dragOver, setDragOver] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState<string | null>(null);

  const handleFile = async (file: File) => {
    setUploading(true);
    setResult(null);
    try {
      const res = await uploadObservation(file);
      setResult(`Uploaded ${res.n_records} records from ${res.observations[0]?.filename ?? file.name}`);
      const obs = await fetchObservations();
      setObservations(obs);
    } catch (err) {
      setResult(`Upload failed: ${err instanceof Error ? err.message : 'Unknown error'}`);
    }
    setUploading(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle>Upload Observation File</DialogTitle>
      <DialogContent>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          Upload a CSV file with columns: datetime, value. The file will be
          stored for the current session and can be overlaid on hydrograph charts.
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
          <CloudUploadIcon sx={{ fontSize: 48, color: 'text.secondary', mb: 1 }} />
          <Typography>
            Drop CSV file here or{' '}
            <Button component="label" size="small">
              browse
              <input
                type="file"
                accept=".csv,.txt"
                hidden
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) handleFile(file);
                  e.target.value = '';
                }}
              />
            </Button>
          </Typography>
        </Box>

        {uploading && (
          <Typography sx={{ mt: 2 }} color="text.secondary">
            Uploading...
          </Typography>
        )}

        {result && (
          <Typography sx={{ mt: 2 }} color={result.startsWith('Upload failed') ? 'error' : 'success.main'}>
            {result}
          </Typography>
        )}
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Close</Button>
      </DialogActions>
    </Dialog>
  );
}
