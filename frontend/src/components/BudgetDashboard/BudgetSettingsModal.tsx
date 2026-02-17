/**
 * Settings modal for the Budget Dashboard.
 * Shows model native units (read-only) and display unit selectors.
 */

import Dialog from '@mui/material/Dialog';
import DialogTitle from '@mui/material/DialogTitle';
import DialogContent from '@mui/material/DialogContent';
import DialogActions from '@mui/material/DialogActions';
import Button from '@mui/material/Button';
import Typography from '@mui/material/Typography';
import Select from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem';
import InputLabel from '@mui/material/InputLabel';
import FormControl from '@mui/material/FormControl';
import ToggleButtonGroup from '@mui/material/ToggleButtonGroup';
import ToggleButton from '@mui/material/ToggleButton';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableRow from '@mui/material/TableRow';
import Divider from '@mui/material/Divider';
import Box from '@mui/material/Box';
import { useViewerStore } from '../../stores/viewerStore';
import type { BudgetUnitsMetadata } from '../../api/client';
import { VOLUME_UNITS, AREA_UNITS, LENGTH_UNITS, RATE_UNITS, TIME_AGGS } from './budgetUnits';

interface BudgetSettingsModalProps {
  open: boolean;
  onClose: () => void;
  unitsMeta?: BudgetUnitsMetadata;
  hasLengthColumns: boolean;
}

export function BudgetSettingsModal({
  open,
  onClose,
  unitsMeta,
  hasLengthColumns,
}: BudgetSettingsModalProps) {
  const {
    budgetDisplayMode, budgetVolumeUnit, budgetRateUnit, budgetAreaUnit, budgetLengthUnit, budgetTimeAgg,
    budgetAnalysisMode,
    setBudgetDisplayMode, setBudgetVolumeUnit, setBudgetRateUnit, setBudgetAreaUnit, setBudgetLengthUnit, setBudgetTimeAgg,
  } = useViewerStore();

  const isAnalysisMode = budgetAnalysisMode !== 'timeseries';

  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle>Budget Settings</DialogTitle>
      <DialogContent dividers>
        {/* Section 1: Model Units (read-only) */}
        <Typography variant="subtitle2" sx={{ mb: 1 }}>
          Model Native Units
        </Typography>
        <Table size="small" sx={{ mb: 2 }}>
          <TableBody>
            <TableRow>
              <TableCell sx={{ fontWeight: 500, width: 120 }}>Length</TableCell>
              <TableCell>{unitsMeta?.source_length_unit || '—'}</TableCell>
            </TableRow>
            <TableRow>
              <TableCell sx={{ fontWeight: 500 }}>Area (storage)</TableCell>
              <TableCell>{unitsMeta?.source_area_unit || '—'}</TableCell>
            </TableRow>
            {unitsMeta?.source_area_output_unit && (
              <TableRow>
                <TableCell sx={{ fontWeight: 500 }}>Area (output)</TableCell>
                <TableCell>{unitsMeta.source_area_output_unit}</TableCell>
              </TableRow>
            )}
            <TableRow>
              <TableCell sx={{ fontWeight: 500 }}>Volume</TableCell>
              <TableCell>{unitsMeta?.source_volume_unit || '—'}</TableCell>
            </TableRow>
            <TableRow>
              <TableCell sx={{ fontWeight: 500 }}>Timestep</TableCell>
              <TableCell>{unitsMeta?.timestep_unit || '—'}</TableCell>
            </TableRow>
          </TableBody>
        </Table>

        <Divider sx={{ my: 2 }} />

        {/* Section 2: Display Mode & Units */}
        <Typography variant="subtitle2" sx={{ mb: 1 }}>
          Display Mode
        </Typography>
        <ToggleButtonGroup
          value={budgetDisplayMode}
          exclusive
          onChange={(_, val) => { if (val) setBudgetDisplayMode(val); }}
          size="small"
          fullWidth
          sx={{ mb: 2 }}
        >
          <ToggleButton value="volume">Volume</ToggleButton>
          <ToggleButton value="rate">Rate</ToggleButton>
        </ToggleButtonGroup>
        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 2 }}>
          {budgetDisplayMode === 'volume'
            ? 'Values represent total volumes per timestep. Yearly aggregation sums monthly values.'
            : 'Values represent flow rates. Yearly aggregation averages monthly rates.'}
        </Typography>

        <Typography variant="subtitle2" sx={{ mb: 1 }}>
          Display Units
        </Typography>
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          {budgetDisplayMode === 'volume' ? (
            <FormControl fullWidth size="small">
              <InputLabel>Volume Unit</InputLabel>
              <Select
                value={budgetVolumeUnit}
                label="Volume Unit"
                onChange={(e) => setBudgetVolumeUnit(e.target.value)}
              >
                {VOLUME_UNITS.map((u) => (
                  <MenuItem key={u.id} value={u.id}>{u.label}</MenuItem>
                ))}
              </Select>
            </FormControl>
          ) : (
            <FormControl fullWidth size="small">
              <InputLabel>Rate Unit</InputLabel>
              <Select
                value={budgetRateUnit}
                label="Rate Unit"
                onChange={(e) => setBudgetRateUnit(e.target.value)}
              >
                {RATE_UNITS.map((u) => (
                  <MenuItem key={u.id} value={u.id}>{u.label}</MenuItem>
                ))}
              </Select>
            </FormControl>
          )}

          <FormControl fullWidth size="small">
            <InputLabel>Area Unit</InputLabel>
            <Select
              value={budgetAreaUnit}
              label="Area Unit"
              onChange={(e) => setBudgetAreaUnit(e.target.value)}
            >
              {AREA_UNITS.map((u) => (
                <MenuItem key={u.id} value={u.id}>{u.label}</MenuItem>
              ))}
            </Select>
          </FormControl>

          {hasLengthColumns && (
            <FormControl fullWidth size="small">
              <InputLabel>Length Unit</InputLabel>
              <Select
                value={budgetLengthUnit}
                label="Length Unit"
                onChange={(e) => setBudgetLengthUnit(e.target.value)}
              >
                {LENGTH_UNITS.map((u) => (
                  <MenuItem key={u.id} value={u.id}>{u.label}</MenuItem>
                ))}
              </Select>
            </FormControl>
          )}
        </Box>

        <Divider sx={{ my: 2 }} />

        {/* Section 3: Time Aggregation */}
        <Typography variant="subtitle2" sx={{ mb: 1 }}>
          Time Aggregation
        </Typography>
        <FormControl fullWidth size="small" disabled={isAnalysisMode}>
          <InputLabel>Time Aggregation</InputLabel>
          <Select
            value={budgetTimeAgg}
            label="Time Aggregation"
            onChange={(e) => setBudgetTimeAgg(e.target.value)}
          >
            {TIME_AGGS.map((a) => (
              <MenuItem key={a.id} value={a.id}>{a.label}</MenuItem>
            ))}
          </Select>
        </FormControl>
        {isAnalysisMode && (
          <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: 'block' }}>
            Time aggregation is disabled in analysis views.
          </Typography>
        )}
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Close</Button>
      </DialogActions>
    </Dialog>
  );
}
