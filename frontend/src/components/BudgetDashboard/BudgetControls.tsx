/**
 * Left sidebar controls for the Budget Dashboard.
 * Budget type selector, location dropdown, chart type toggle,
 * settings button (opens modal), and action buttons.
 */

import { useState, useEffect } from 'react';
import Box from '@mui/material/Box';
import Paper from '@mui/material/Paper';
import Typography from '@mui/material/Typography';
import Select from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem';
import InputLabel from '@mui/material/InputLabel';
import FormControl from '@mui/material/FormControl';
import ToggleButtonGroup from '@mui/material/ToggleButtonGroup';
import ToggleButton from '@mui/material/ToggleButton';
import Button from '@mui/material/Button';
import Divider from '@mui/material/Divider';
import IconButton from '@mui/material/IconButton';
import ShowChartIcon from '@mui/icons-material/ShowChart';
import BarChartIcon from '@mui/icons-material/BarChart';
import StackedLineChartIcon from '@mui/icons-material/StackedLineChart';
import AccountTreeIcon from '@mui/icons-material/AccountTree';
import DownloadIcon from '@mui/icons-material/Download';
import SettingsIcon from '@mui/icons-material/Settings';
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';
import { useViewerStore } from '../../stores/viewerStore';
import { fetchBudgetLocations, getExportBudgetCsvUrl } from '../../api/client';
import type { BudgetLocation, BudgetUnitsMetadata } from '../../api/client';
import { BudgetGlossary } from './BudgetGlossary';
import { BudgetSettingsModal } from './BudgetSettingsModal';

export const BUDGET_LABELS: Record<string, string> = {
  gw: 'Groundwater',
  stream: 'Stream',
  stream_node: 'Stream Node',
  lwu: 'Land & Water Use',
  rootzone: 'Root Zone',
  unsaturated: 'Unsaturated Zone',
  diversion: 'Diversion',
  lake: 'Lake',
  small_watershed: 'Small Watershed',
};

interface BudgetControlsProps {
  budgetTypes: string[];
  hasLengthColumns?: boolean;
  unitsMeta?: BudgetUnitsMetadata;
}

export function BudgetControls({
  budgetTypes,
  hasLengthColumns = false,
  unitsMeta,
}: BudgetControlsProps) {
  const {
    activeBudgetType, activeBudgetLocation, budgetChartType,
    showBudgetSankey, showBudgetGlossary,
    budgetAnalysisMode,
    setActiveBudgetType, setActiveBudgetLocation, setBudgetChartType,
    setShowBudgetSankey, setShowBudgetGlossary,
  } = useViewerStore();

  const isAnalysisMode = budgetAnalysisMode !== 'timeseries';
  const [locations, setLocations] = useState<BudgetLocation[]>([]);
  const [settingsOpen, setSettingsOpen] = useState(false);

  // Load locations when budget type changes
  useEffect(() => {
    if (!activeBudgetType) return;
    fetchBudgetLocations(activeBudgetType)
      .then((data) => {
        setLocations(data.locations);
        if (data.locations.length > 0 && !activeBudgetLocation) {
          setActiveBudgetLocation(data.locations[0].name);
        }
      })
      .catch(console.error);
  }, [activeBudgetType, activeBudgetLocation, setActiveBudgetLocation]);

  // Auto-select first budget type
  useEffect(() => {
    if (budgetTypes.length > 0 && !activeBudgetType) {
      setActiveBudgetType(budgetTypes[0]);
    }
  }, [budgetTypes, activeBudgetType, setActiveBudgetType]);

  return (
    <>
      <Paper elevation={2} sx={{ width: 260, p: 2, overflowY: 'auto', flexShrink: 0 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
          <Typography variant="subtitle2">
            Budget Controls
          </Typography>
          <IconButton
            size="small"
            onClick={() => setShowBudgetGlossary(true)}
            title="Budget term glossary"
          >
            <InfoOutlinedIcon fontSize="small" />
          </IconButton>
        </Box>

        {/* Budget type selector */}
        <FormControl fullWidth size="small" sx={{ mb: 2 }}>
          <InputLabel>Budget Type</InputLabel>
          <Select
            value={activeBudgetType}
            label="Budget Type"
            onChange={(e) => {
              setActiveBudgetType(e.target.value);
              setActiveBudgetLocation('');
              setLocations([]);
            }}
          >
            {budgetTypes.map((t) => (
              <MenuItem key={t} value={t}>
                {BUDGET_LABELS[t] || t}
              </MenuItem>
            ))}
          </Select>
        </FormControl>

        {/* Location selector */}
        <FormControl fullWidth size="small" sx={{ mb: 2 }}>
          <InputLabel>Location</InputLabel>
          <Select
            value={activeBudgetLocation}
            label="Location"
            onChange={(e) => setActiveBudgetLocation(e.target.value)}
          >
            {locations.map((l) => (
              <MenuItem key={l.id} value={l.name}>
                {l.name}
              </MenuItem>
            ))}
          </Select>
        </FormControl>

        {/* Chart type toggle */}
        <Box sx={{ mb: 2 }}>
          <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
            Chart Type
          </Typography>
          <ToggleButtonGroup
            value={budgetChartType}
            exclusive
            onChange={(_, val) => { if (val) setBudgetChartType(val); }}
            size="small"
            fullWidth
            disabled={isAnalysisMode}
          >
            <ToggleButton value="area">
              <StackedLineChartIcon fontSize="small" sx={{ mr: 0.5 }} />
              Area
            </ToggleButton>
            <ToggleButton value="bar">
              <BarChartIcon fontSize="small" sx={{ mr: 0.5 }} />
              Bar
            </ToggleButton>
            <ToggleButton value="line">
              <ShowChartIcon fontSize="small" sx={{ mr: 0.5 }} />
              Line
            </ToggleButton>
          </ToggleButtonGroup>
        </Box>

        <Divider sx={{ my: 1 }} />

        {/* Settings button — opens modal */}
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

        <Divider sx={{ my: 1 }} />

        {/* Water Balance Sankey toggle */}
        <Button
          variant={showBudgetSankey ? 'contained' : 'outlined'}
          size="small"
          startIcon={<AccountTreeIcon />}
          onClick={() => setShowBudgetSankey(!showBudgetSankey)}
          fullWidth
          sx={{ mb: 1, textTransform: 'none' }}
        >
          {showBudgetSankey ? 'Show Time Series' : 'Water Balance'}
        </Button>

        {/* Export CSV */}
        {activeBudgetType && activeBudgetLocation && !showBudgetSankey && (
          <Button
            variant="outlined"
            size="small"
            startIcon={<DownloadIcon />}
            component="a"
            href={getExportBudgetCsvUrl(activeBudgetType, activeBudgetLocation)}
            download
            fullWidth
            sx={{ textTransform: 'none' }}
          >
            Export CSV
          </Button>
        )}
      </Paper>

      {/* Glossary drawer */}
      <BudgetGlossary
        open={showBudgetGlossary}
        onClose={() => setShowBudgetGlossary(false)}
        budgetType={activeBudgetType}
      />

      {/* Settings modal */}
      <BudgetSettingsModal
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        unitsMeta={unitsMeta}
        hasLengthColumns={hasLengthColumns}
      />
    </>
  );
}
