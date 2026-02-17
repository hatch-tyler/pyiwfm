/**
 * Left sidebar controls for the Budget Dashboard.
 * Budget type selector, location dropdown, chart type toggle,
 * unit conversion dropdowns (conditional), and time aggregation.
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
import Chip from '@mui/material/Chip';
import Divider from '@mui/material/Divider';
import IconButton from '@mui/material/IconButton';
import ShowChartIcon from '@mui/icons-material/ShowChart';
import BarChartIcon from '@mui/icons-material/BarChart';
import StackedLineChartIcon from '@mui/icons-material/StackedLineChart';
import AccountTreeIcon from '@mui/icons-material/AccountTree';
import DownloadIcon from '@mui/icons-material/Download';
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';
import { useViewerStore } from '../../stores/viewerStore';
import { fetchBudgetLocations, getExportBudgetCsvUrl } from '../../api/client';
import type { BudgetLocation, BudgetUnitsMetadata } from '../../api/client';
import { VOLUME_UNITS, AREA_UNITS, LENGTH_UNITS, RATE_UNITS, TIME_AGGS } from './budgetUnits';
import { BudgetGlossary } from './BudgetGlossary';

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
  hasVolumeColumns?: boolean;
  hasAreaColumns?: boolean;
  hasLengthColumns?: boolean;
  unitsMeta?: BudgetUnitsMetadata;
}

export function BudgetControls({
  budgetTypes,
  hasVolumeColumns = true,
  hasAreaColumns = false,
  hasLengthColumns = false,
  unitsMeta,
}: BudgetControlsProps) {
  const {
    activeBudgetType, activeBudgetLocation, budgetChartType,
    showBudgetSankey, showBudgetGlossary,
    budgetVolumeUnit, budgetRateUnit, budgetAreaUnit, budgetLengthUnit, budgetTimeAgg,
    budgetAnalysisMode,
    setActiveBudgetType, setActiveBudgetLocation, setBudgetChartType,
    setShowBudgetSankey, setShowBudgetGlossary,
    setBudgetVolumeUnit, setBudgetRateUnit, setBudgetAreaUnit, setBudgetLengthUnit, setBudgetTimeAgg,
    setBudgetAnalysisMode,
  } = useViewerStore();

  const isAnalysisMode = budgetAnalysisMode !== 'timeseries';
  const [locations, setLocations] = useState<BudgetLocation[]>([]);

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

        {/* Settings section */}
        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
          Settings
        </Typography>

        {/* Model native units chip */}
        {unitsMeta && (
          <Chip
            label={`Model: ${unitsMeta.source_volume_unit || '?'}, ${unitsMeta.source_area_unit || '?'} (${unitsMeta.timestep_unit || '?'})`}
            size="small"
            variant="outlined"
            sx={{ mb: 1, fontSize: '0.7rem', height: 22 }}
          />
        )}

        {/* Time Aggregation — grouped with settings */}
        <FormControl fullWidth size="small" sx={{ mb: 1.5 }} disabled={isAnalysisMode}>
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

        {/* Analysis View selector */}
        <FormControl fullWidth size="small" sx={{ mb: 1.5 }}>
          <InputLabel>View</InputLabel>
          <Select
            value={budgetAnalysisMode}
            label="View"
            onChange={(e) => setBudgetAnalysisMode(e.target.value as typeof budgetAnalysisMode)}
          >
            <MenuItem value="timeseries">Time Series</MenuItem>
            <MenuItem value="monthly_pattern">Monthly Pattern</MenuItem>
            <MenuItem value="component_ratios">Component Ratios</MenuItem>
            <MenuItem value="cumulative_departure">Cumulative Departure</MenuItem>
            <MenuItem value="exceedance">Exceedance Probability</MenuItem>
          </Select>
        </FormControl>

        {/* Volume Unit — only when volume columns exist */}
        {hasVolumeColumns && (
          <FormControl fullWidth size="small" sx={{ mb: 1.5 }}>
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
        )}

        {/* Rate Unit — only when volume columns exist */}
        {hasVolumeColumns && (
          <FormControl fullWidth size="small" sx={{ mb: 1.5 }}>
            <InputLabel>Rate Display</InputLabel>
            <Select
              value={budgetRateUnit}
              label="Rate Display"
              onChange={(e) => setBudgetRateUnit(e.target.value)}
            >
              {RATE_UNITS.map((u) => (
                <MenuItem key={u.id} value={u.id}>{u.label}</MenuItem>
              ))}
            </Select>
          </FormControl>
        )}

        {/* Area Unit — only when area columns exist */}
        {hasAreaColumns && (
          <FormControl fullWidth size="small" sx={{ mb: 1.5 }}>
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
        )}

        {/* Length Unit — only when length columns exist (subsidence) */}
        {hasLengthColumns && (
          <FormControl fullWidth size="small" sx={{ mb: 1.5 }}>
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
    </>
  );
}
