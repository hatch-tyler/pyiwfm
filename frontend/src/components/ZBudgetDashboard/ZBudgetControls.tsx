/**
 * ZBudget controls sidebar: zone editor + analysis selection.
 */

import { useState, useEffect } from 'react';
import Box from '@mui/material/Box';
import Paper from '@mui/material/Paper';
import Typography from '@mui/material/Typography';
import Button from '@mui/material/Button';
import IconButton from '@mui/material/IconButton';
import TextField from '@mui/material/TextField';
import Select from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem';
import InputLabel from '@mui/material/InputLabel';
import FormControl from '@mui/material/FormControl';
import ToggleButtonGroup from '@mui/material/ToggleButtonGroup';
import ToggleButton from '@mui/material/ToggleButton';
import Divider from '@mui/material/Divider';
import Chip from '@mui/material/Chip';
import Tooltip from '@mui/material/Tooltip';
import StackedLineChartIcon from '@mui/icons-material/StackedLineChart';
import BarChartIcon from '@mui/icons-material/BarChart';
import ShowChartIcon from '@mui/icons-material/ShowChart';
import AddIcon from '@mui/icons-material/Add';
import DeleteIcon from '@mui/icons-material/Delete';
import SettingsIcon from '@mui/icons-material/Settings';
import MenuBookIcon from '@mui/icons-material/MenuBook';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import EditIcon from '@mui/icons-material/Edit';
import VisibilityIcon from '@mui/icons-material/Visibility';
import DeleteSweepIcon from '@mui/icons-material/DeleteSweep';
import { useViewerStore } from '../../stores/viewerStore';
import type { ZoneInfo, ZBudgetPreset, BudgetUnitsMetadata } from '../../api/client';
import { BudgetSettingsModal } from '../BudgetDashboard/BudgetSettingsModal';
import { ZBudgetGlossary } from './ZBudgetGlossary';
import { getZoneColor } from './zoneColors';

const ZBUDGET_LABELS: Record<string, string> = {
  gw: 'Groundwater',
  rootzone: 'Root Zone',
  lwu: 'Land & Water Use',
};

interface ZBudgetControlsProps {
  zbudgetTypes: string[];
  presets: ZBudgetPreset[];
  zones: ZoneInfo[];
  onAddZone: () => void;
  onRemoveZone: (id: number) => void;
  onRenameZone: (id: number, name: string) => void;
  onLoadPreset: (preset: ZBudgetPreset) => void;
  onClearAll: () => void;
  onRunZBudget: () => void;
  loading: boolean;
  unitsMeta?: BudgetUnitsMetadata;
  zoneNames: string[];
}

export function ZBudgetControls({
  zbudgetTypes,
  presets,
  zones,
  onAddZone,
  onRemoveZone,
  onRenameZone,
  onLoadPreset,
  onClearAll,
  onRunZBudget,
  loading,
  unitsMeta,
  zoneNames,
}: ZBudgetControlsProps) {
  const {
    zbudgetActiveType, zbudgetActiveZone, zbudgetEditMode, zbudgetPaintZoneId,
    budgetChartType,
    setZBudgetActiveType, setZBudgetActiveZone, setZBudgetEditMode,
    setZBudgetPaintZoneId, setBudgetChartType,
  } = useViewerStore();

  const [settingsOpen, setSettingsOpen] = useState(false);
  const [glossaryOpen, setGlossaryOpen] = useState(false);
  const [editingName, setEditingName] = useState<number | null>(null);
  const [editValue, setEditValue] = useState('');

  // Auto-select first zbudget type
  useEffect(() => {
    if (zbudgetTypes.length > 0 && !zbudgetActiveType) {
      setZBudgetActiveType(zbudgetTypes[0]);
    }
  }, [zbudgetTypes, zbudgetActiveType, setZBudgetActiveType]);

  const handleStartRename = (z: ZoneInfo) => {
    setEditingName(z.id);
    setEditValue(z.name);
  };

  const handleFinishRename = (id: number) => {
    if (editValue.trim()) {
      onRenameZone(id, editValue.trim());
    }
    setEditingName(null);
  };

  return (
    <>
      <Paper elevation={2} sx={{ width: 260, p: 2, overflowY: 'auto', flexShrink: 0 }}>
        {/* Mode toggle */}
        <Box sx={{ display: 'flex', gap: 1, mb: 2 }}>
          <Button
            variant={zbudgetEditMode ? 'contained' : 'outlined'}
            size="small"
            startIcon={<EditIcon />}
            onClick={() => setZBudgetEditMode(true)}
            sx={{ flex: 1, textTransform: 'none' }}
          >
            Edit Zones
          </Button>
          <Button
            variant={!zbudgetEditMode ? 'contained' : 'outlined'}
            size="small"
            startIcon={<VisibilityIcon />}
            onClick={() => setZBudgetEditMode(false)}
            sx={{ flex: 1, textTransform: 'none' }}
            disabled={zones.length === 0}
          >
            Charts
          </Button>
        </Box>

        <Divider sx={{ mb: 2 }} />

        {/* Zone editor section */}
        {zbudgetEditMode && (
          <>
            <Typography variant="subtitle2" sx={{ mb: 1 }}>Zone Editor</Typography>

            {/* Zone list */}
            {zones.map((z) => (
              <Box
                key={z.id}
                sx={{
                  display: 'flex', alignItems: 'center', gap: 0.5, mb: 0.5,
                  p: 0.5, borderRadius: 1,
                  bgcolor: zbudgetPaintZoneId === z.id ? 'action.selected' : 'transparent',
                  cursor: 'pointer',
                  '&:hover': { bgcolor: 'action.hover' },
                }}
                onClick={() => setZBudgetPaintZoneId(z.id)}
              >
                <Box sx={{ width: 16, height: 16, bgcolor: getZoneColor(z.id), borderRadius: '3px', flexShrink: 0 }} />
                {editingName === z.id ? (
                  <TextField
                    size="small"
                    value={editValue}
                    onChange={(e) => setEditValue(e.target.value)}
                    onBlur={() => handleFinishRename(z.id)}
                    onKeyDown={(e) => { if (e.key === 'Enter') handleFinishRename(z.id); }}
                    autoFocus
                    sx={{ flex: 1, '& input': { py: 0.25, fontSize: 13 } }}
                  />
                ) : (
                  <Typography
                    variant="body2"
                    sx={{ flex: 1, fontSize: 13, overflow: 'hidden', textOverflow: 'ellipsis' }}
                    onDoubleClick={() => handleStartRename(z)}
                  >
                    {z.name}
                  </Typography>
                )}
                <Chip label={z.elements.length} size="small" sx={{ height: 20, fontSize: 11 }} />
                <IconButton size="small" onClick={(e) => { e.stopPropagation(); onRemoveZone(z.id); }}>
                  <DeleteIcon fontSize="small" />
                </IconButton>
              </Box>
            ))}

            <Button
              variant="outlined"
              size="small"
              startIcon={<AddIcon />}
              onClick={onAddZone}
              fullWidth
              sx={{ mb: 1, textTransform: 'none' }}
            >
              Add Zone
            </Button>

            {/* Preset loader */}
            {presets.length > 0 && (
              <FormControl fullWidth size="small" sx={{ mb: 1 }}>
                <InputLabel>Load Preset</InputLabel>
                <Select
                  value=""
                  label="Load Preset"
                  onChange={(e) => {
                    const preset = presets.find((p) => p.name === e.target.value);
                    if (preset) onLoadPreset(preset);
                  }}
                >
                  {presets.map((p) => (
                    <MenuItem key={p.name} value={p.name}>{p.name}</MenuItem>
                  ))}
                </Select>
              </FormControl>
            )}

            <Button
              variant="outlined"
              size="small"
              startIcon={<DeleteSweepIcon />}
              onClick={onClearAll}
              fullWidth
              sx={{ mb: 2, textTransform: 'none' }}
              disabled={zones.length === 0}
            >
              Clear All
            </Button>

            <Divider sx={{ mb: 2 }} />

            {/* Run ZBudget */}
            <Button
              variant="contained"
              color="primary"
              size="small"
              startIcon={<PlayArrowIcon />}
              onClick={onRunZBudget}
              fullWidth
              sx={{ mb: 1, textTransform: 'none' }}
              disabled={zones.length === 0 || loading}
            >
              {loading ? 'Running...' : 'Run ZBudget'}
            </Button>
          </>
        )}

        {/* Chart mode controls */}
        {!zbudgetEditMode && (
          <>
            <Typography variant="subtitle2" sx={{ mb: 1 }}>Analysis</Typography>

            {/* ZBudget type */}
            <FormControl fullWidth size="small" sx={{ mb: 2 }}>
              <InputLabel>ZBudget Type</InputLabel>
              <Select
                value={zbudgetActiveType}
                label="ZBudget Type"
                onChange={(e) => setZBudgetActiveType(e.target.value)}
              >
                {zbudgetTypes.map((t) => (
                  <MenuItem key={t} value={t}>{ZBUDGET_LABELS[t] || t}</MenuItem>
                ))}
              </Select>
            </FormControl>

            {/* Zone selector */}
            <FormControl fullWidth size="small" sx={{ mb: 2 }}>
              <InputLabel>Zone</InputLabel>
              <Select
                value={zbudgetActiveZone}
                label="Zone"
                onChange={(e) => setZBudgetActiveZone(e.target.value)}
              >
                {zoneNames.map((name) => (
                  <MenuItem key={name} value={name}>{name}</MenuItem>
                ))}
              </Select>
            </FormControl>

            {/* Chart type toggle */}
            <ToggleButtonGroup
              value={budgetChartType}
              exclusive
              onChange={(_, val) => { if (val) setBudgetChartType(val); }}
              size="small"
              fullWidth
              sx={{ mb: 1 }}
            >
              <ToggleButton value="area"><StackedLineChartIcon fontSize="small" /></ToggleButton>
              <ToggleButton value="bar"><BarChartIcon fontSize="small" /></ToggleButton>
              <ToggleButton value="line"><ShowChartIcon fontSize="small" /></ToggleButton>
            </ToggleButtonGroup>

            {/* Settings & Glossary */}
            <Tooltip title="Unit Settings">
              <Button
                variant="outlined"
                size="small"
                startIcon={<SettingsIcon />}
                onClick={() => setSettingsOpen(true)}
                fullWidth
                sx={{ mb: 1, textTransform: 'none' }}
              >
                Settings
              </Button>
            </Tooltip>

            <Button
              variant="outlined"
              size="small"
              startIcon={<MenuBookIcon />}
              onClick={() => setGlossaryOpen(true)}
              fullWidth
              sx={{ textTransform: 'none' }}
            >
              Glossary
            </Button>
          </>
        )}
      </Paper>

      <BudgetSettingsModal
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        unitsMeta={unitsMeta}
        hasLengthColumns={false}
      />
      <ZBudgetGlossary
        open={glossaryOpen}
        onClose={() => setGlossaryOpen(false)}
        budgetType={zbudgetActiveType}
      />
    </>
  );
}
