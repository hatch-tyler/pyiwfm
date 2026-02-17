/**
 * 3-step observation upload wizard.
 *
 * Step 1: Select observation type + drag-drop or browse file
 * Step 2: Column mapping (date, value, optional location)
 * Step 3: Confirm and upload
 */

import { useState } from 'react';
import Dialog from '@mui/material/Dialog';
import DialogTitle from '@mui/material/DialogTitle';
import DialogContent from '@mui/material/DialogContent';
import DialogActions from '@mui/material/DialogActions';
import Button from '@mui/material/Button';
import Typography from '@mui/material/Typography';
import Box from '@mui/material/Box';
import Stepper from '@mui/material/Stepper';
import Step from '@mui/material/Step';
import StepLabel from '@mui/material/StepLabel';
import Select from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem';
import FormControl from '@mui/material/FormControl';
import InputLabel from '@mui/material/InputLabel';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Paper from '@mui/material/Paper';
import Alert from '@mui/material/Alert';
import Tooltip from '@mui/material/Tooltip';
import CircularProgress from '@mui/material/CircularProgress';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import {
  previewObservationFile,
  uploadObservation,
  fetchObservations,
} from '../../api/client';
import type { ObservationPreview, UploadResult } from '../../api/client';
import { useViewerStore } from '../../stores/viewerStore';

const STEPS = ['Type & File', 'Column Mapping', 'Confirm'];

/** Column name patterns for auto-detection. */
const DATE_PATTERNS = ['date', 'datetime', 'time', 'timestamp', 'dt'];
const VALUE_PATTERNS = ['value', 'level', 'head', 'flow', 'stage', 'measurement', 'obs'];
const LOCATION_PATTERNS = ['location', 'station', 'site', 'well', 'id', 'name', 'loc'];

function autoDetectColumn(headers: string[], patterns: string[]): number {
  const lower = headers.map((h) => h.toLowerCase());
  for (const p of patterns) {
    const idx = lower.findIndex((h) => h === p || h.includes(p));
    if (idx >= 0) return idx;
  }
  return -1;
}

interface UploadWizardProps {
  open: boolean;
  onClose: () => void;
}

export function UploadWizard({ open, onClose }: UploadWizardProps) {
  const { setObservations } = useViewerStore();
  const [step, setStep] = useState(0);
  const [obsType, setObsType] = useState('gw');
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<ObservationPreview | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);

  // Column mapping
  const [dateCol, setDateCol] = useState(0);
  const [valueCol, setValueCol] = useState(1);
  const [locationCol, setLocationCol] = useState(-1);

  // Upload result
  const [uploadResult, setUploadResult] = useState<UploadResult | null>(null);

  const reset = () => {
    setStep(0);
    setFile(null);
    setPreview(null);
    setError(null);
    setUploadResult(null);
    setDateCol(0);
    setValueCol(1);
    setLocationCol(-1);
  };

  const handleClose = () => {
    reset();
    onClose();
  };

  const handleFile = async (f: File) => {
    setFile(f);
    setLoading(true);
    setError(null);
    try {
      const prev = await previewObservationFile(f);
      setPreview(prev);

      // Auto-detect columns from headers
      const dc = autoDetectColumn(prev.headers, DATE_PATTERNS);
      const vc = autoDetectColumn(prev.headers, VALUE_PATTERNS);
      const lc = autoDetectColumn(prev.headers, LOCATION_PATTERNS);
      setDateCol(dc >= 0 ? dc : 0);
      setValueCol(vc >= 0 ? vc : (dc === 0 ? 1 : 0));
      setLocationCol(lc);

      setStep(1);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Preview failed');
    }
    setLoading(false);
  };

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const res = await uploadObservation(file, obsType, dateCol, valueCol, locationCol);
      setUploadResult(res);
      const obs = await fetchObservations();
      setObservations(obs);
      setStep(2);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
    }
    setLoading(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  };

  return (
    <Dialog open={open} onClose={handleClose} maxWidth="md" fullWidth>
      <DialogTitle>Upload Observations</DialogTitle>
      <DialogContent>
        <Stepper activeStep={step} sx={{ mb: 3 }}>
          {STEPS.map((label) => (
            <Step key={label}>
              <StepLabel>{label}</StepLabel>
            </Step>
          ))}
        </Stepper>

        {error && (
          <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
            {error}
          </Alert>
        )}

        {/* Step 0: Type & File */}
        {step === 0 && (
          <>
            <FormControl size="small" sx={{ mb: 2, minWidth: 200 }}>
              <InputLabel>Observation Type</InputLabel>
              <Select
                value={obsType}
                label="Observation Type"
                onChange={(e) => setObsType(e.target.value)}
              >
                <MenuItem value="gw">GW Levels</MenuItem>
                <MenuItem value="stream">Stream Gages</MenuItem>
                <MenuItem value="subsidence">Subsidence</MenuItem>
              </Select>
            </FormControl>

            <Alert severity="info" sx={{ mb: 2 }}>
              Upload a CSV or TXT file with columns for date/time and measurement values.
              An optional location column groups records by station or well.
              <Box component="code" sx={{ display: 'block', mt: 0.5, fontSize: 11, whiteSpace: 'pre' }}>
                {'Date,Value,Station\n01/31/2000,125.3,Well-01\n02/28/2000,124.8,Well-01'}
              </Box>
            </Alert>

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
                Drop CSV/TXT file here or{' '}
                <Button component="label" size="small">
                  browse
                  <input
                    type="file"
                    accept=".csv,.txt"
                    hidden
                    onChange={(e) => {
                      const f = e.target.files?.[0];
                      if (f) handleFile(f);
                      e.target.value = '';
                    }}
                  />
                </Button>
              </Typography>
              {loading && <CircularProgress size={24} sx={{ mt: 1 }} />}
            </Box>
          </>
        )}

        {/* Step 1: Column Mapping */}
        {step === 1 && preview && (
          <>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              File: <strong>{file?.name}</strong> ({preview.n_rows} data rows, {preview.headers.length} columns)
            </Typography>

            <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
              <FormControl size="small" sx={{ minWidth: 160 }}>
                <Tooltip title="Column with date/time values (e.g., MM/DD/YYYY, ISO 8601, MM/DD/YYYY HH:MM)" arrow>
                  <InputLabel>Date/Time Column</InputLabel>
                </Tooltip>
                <Select
                  value={dateCol}
                  label="Date/Time Column"
                  onChange={(e) => setDateCol(e.target.value as number)}
                >
                  {preview.headers.map((h: string, i: number) => (
                    <MenuItem key={i} value={i}>{h}</MenuItem>
                  ))}
                </Select>
              </FormControl>

              <FormControl size="small" sx={{ minWidth: 160 }}>
                <Tooltip title="Column with numeric measurement values (e.g., head elevation in ft)" arrow>
                  <InputLabel>Value Column</InputLabel>
                </Tooltip>
                <Select
                  value={valueCol}
                  label="Value Column"
                  onChange={(e) => setValueCol(e.target.value as number)}
                >
                  {preview.headers.map((h: string, i: number) => (
                    <MenuItem key={i} value={i}>{h}</MenuItem>
                  ))}
                </Select>
              </FormControl>

              <FormControl size="small" sx={{ minWidth: 160 }}>
                <Tooltip title="Optional. Groups records by station/well name. Leave as 'None' for single-station files." arrow>
                  <InputLabel>Location Column</InputLabel>
                </Tooltip>
                <Select
                  value={locationCol}
                  label="Location Column"
                  onChange={(e) => setLocationCol(e.target.value as number)}
                >
                  <MenuItem value={-1}><em>None</em></MenuItem>
                  {preview.headers.map((h: string, i: number) => (
                    <MenuItem key={i} value={i}>{h}</MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Box>

            <Typography variant="caption" color="text.secondary" sx={{ mb: 1, display: 'block' }}>
              Preview (first {Math.min(preview.sample_rows.length, 10)} rows):
            </Typography>
            <TableContainer component={Paper} variant="outlined" sx={{ maxHeight: 240 }}>
              <Table size="small" stickyHeader>
                <TableHead>
                  <TableRow>
                    {preview.headers.map((h: string, i: number) => (
                      <TableCell
                        key={i}
                        sx={{
                          fontWeight: 'bold',
                          fontSize: 11,
                          bgcolor: i === dateCol ? 'primary.light'
                            : i === valueCol ? 'success.light'
                            : i === locationCol ? 'warning.light'
                            : undefined,
                          color: (i === dateCol || i === valueCol || i === locationCol) ? 'white' : undefined,
                        }}
                      >
                        {h}
                        {i === dateCol && ' (Date)'}
                        {i === valueCol && ' (Value)'}
                        {i === locationCol && ' (Location)'}
                      </TableCell>
                    ))}
                  </TableRow>
                </TableHead>
                <TableBody>
                  {preview.sample_rows.map((row: string[], ri: number) => (
                    <TableRow key={ri}>
                      {row.map((cell: string, ci: number) => (
                        <TableCell key={ci} sx={{ fontSize: 11, py: 0.5 }}>
                          {cell}
                        </TableCell>
                      ))}
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </>
        )}

        {/* Step 2: Confirm */}
        {step === 2 && uploadResult && (
          <>
            <Alert severity="success" sx={{ mb: 2 }}>
              Uploaded {uploadResult.n_observations} observation set{uploadResult.n_observations > 1 ? 's' : ''}
              {' '}with {uploadResult.n_records} total records.
            </Alert>

            {uploadResult.observations.map((obs) => (
              <Typography key={obs.observation_id} variant="body2" sx={{ mb: 0.5 }}>
                {obs.filename}: {obs.n_records} records
                {obs.location_id ? ` (matched to location #${obs.location_id})` : ''}
                {obs.start_time && ` | ${obs.start_time.slice(0, 10)} - ${obs.end_time?.slice(0, 10)}`}
              </Typography>
            ))}

            {uploadResult.unmatched_locations.length > 0 && (
              <Alert severity="warning" sx={{ mt: 2 }}>
                {uploadResult.unmatched_locations.length} location{uploadResult.unmatched_locations.length > 1 ? 's' : ''} could not be
                matched: {uploadResult.unmatched_locations.join(', ')}
              </Alert>
            )}
          </>
        )}

        {loading && step === 1 && (
          <Box sx={{ display: 'flex', justifyContent: 'center', mt: 2 }}>
            <CircularProgress size={24} />
          </Box>
        )}
      </DialogContent>

      <DialogActions>
        {step > 0 && step < 2 && (
          <Button onClick={() => setStep(step - 1)}>Back</Button>
        )}
        <Box sx={{ flex: 1 }} />
        {step === 1 && (
          <Button variant="contained" onClick={handleUpload} disabled={loading}>
            Upload
          </Button>
        )}
        <Button onClick={handleClose}>
          {step === 2 ? 'Done' : 'Cancel'}
        </Button>
      </DialogActions>
    </Dialog>
  );
}
