/**
 * ZBudget Dashboard: interactive zone definition + zone budget charts.
 * Two modes: zone editor (map) and chart view (reuses BudgetChart).
 */

import { useState, useEffect, useMemo, useCallback, useRef } from 'react';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import CircularProgress from '@mui/material/CircularProgress';
import Dialog from '@mui/material/Dialog';
import DialogContent from '@mui/material/DialogContent';
import IconButton from '@mui/material/IconButton';
import Tabs from '@mui/material/Tabs';
import Tab from '@mui/material/Tab';
import CloseIcon from '@mui/icons-material/Close';
import { useViewerStore } from '../../stores/viewerStore';
import {
  fetchZBudgetTypes,
  fetchZBudgetPresets,
  fetchZBudgetData,
  postZoneDefinition,
} from '../../api/client';
import type {
  ZoneInfo,
  ZBudgetPreset,
  BudgetData,
  BudgetUnitsMetadata,
} from '../../api/client';
import { ZBudgetControls } from './ZBudgetControls';
import { ZoneMap } from './ZoneMap';
import { BudgetChart } from '../BudgetDashboard/BudgetChart';
import { MonthlyPatternChart } from '../BudgetDashboard/MonthlyPatternChart';
import { ComponentRatioChart } from '../BudgetDashboard/ComponentRatioChart';
import { CumulativeDepartureChart } from '../BudgetDashboard/CumulativeDepartureChart';
import { ExceedanceChart } from '../BudgetDashboard/ExceedanceChart';
import { classifyColumns } from '../BudgetDashboard/budgetSplitter';
import type { ChartGroup } from '../BudgetDashboard/budgetSplitter';
import {
  convertVolumeValues,
  convertAreaValues,
  getYAxisLabel,
  getXAxisLabel,
  sourceVolumeToDisplayDefault,
  sourceAreaToDisplayDefault,
} from '../BudgetDashboard/budgetUnits';

interface ConvertedChart {
  data: BudgetData;
  yAxisLabel: string;
  partialYearNote?: string;
}

function convertChartData(
  group: ChartGroup,
  displayMode: 'volume' | 'rate',
  volumeUnit: string,
  rateUnit: string,
  areaUnit: string,
  lengthUnit: string,
  timeAgg: string,
  unitsMeta: BudgetUnitsMetadata | undefined,
): ConvertedChart {
  const sourceVolume = unitsMeta?.source_volume_unit ?? 'AF';
  const sourceArea = unitsMeta?.source_area_unit ?? 'ACRES';
  const isArea = group.chartKind === 'area';
  let firstPartialNote: string | undefined;

  const convertedColumns = group.data.columns.map((col) => {
    if (isArea) {
      const result = convertAreaValues(col.values, group.data.times, sourceArea, areaUnit, timeAgg);
      if (!firstPartialNote && result.partialYearNote) firstPartialNote = result.partialYearNote;
      return { name: col.name, values: result.values, units: col.units };
    } else {
      const result = convertVolumeValues(
        col.values, group.data.times, sourceVolume, displayMode, volumeUnit, rateUnit, timeAgg,
      );
      if (!firstPartialNote && result.partialYearNote) firstPartialNote = result.partialYearNote;
      return { name: col.name, values: result.values, units: col.units };
    }
  });

  let convertedTimes = group.data.times;
  if (group.data.columns.length > 0) {
    if (isArea) {
      const result = convertAreaValues(
        group.data.columns[0].values, group.data.times, sourceArea, areaUnit, timeAgg,
      );
      convertedTimes = result.times;
    } else {
      const result = convertVolumeValues(
        group.data.columns[0].values, group.data.times, sourceVolume, displayMode, volumeUnit, rateUnit, timeAgg,
      );
      convertedTimes = result.times;
    }
  }

  const yAxisLabel = getYAxisLabel(group.chartKind, displayMode, volumeUnit, rateUnit, areaUnit, lengthUnit);

  return {
    data: { location: group.data.location, times: convertedTimes, columns: convertedColumns },
    yAxisLabel,
    partialYearNote: firstPartialNote,
  };
}

const ZBUDGET_LABELS: Record<string, string> = {
  gw: 'Groundwater',
  rootzone: 'Root Zone',
  lwu: 'Land & Water Use',
};

export function ZBudgetView() {
  const {
    zbudgetTypes: storeTypes,
    zbudgetActiveType, zbudgetActiveZone, zbudgetEditMode, zbudgetPaintZoneId,
    zbudgetZones,
    budgetChartType, budgetDisplayMode, budgetVolumeUnit, budgetRateUnit,
    budgetAreaUnit, budgetLengthUnit, budgetTimeAgg,
    budgetAnalysisMode,
    expandedChartIndex, setExpandedChartIndex,
    setZBudgetTypes, setZBudgetZones, setZBudgetActiveZone, setZBudgetEditMode,
    setBudgetVolumeUnit, setBudgetAreaUnit,
    setBudgetAnalysisMode,
  } = useViewerStore();

  const [zbudgetTypes, setLocalTypes] = useState<string[]>(storeTypes);
  const [presets, setPresets] = useState<ZBudgetPreset[]>([]);
  const [geojson, setGeojson] = useState<GeoJSON.FeatureCollection | null>(null);
  const [loading, setLoading] = useState(false);
  const [zbudgetData, setZbudgetData] = useState<BudgetData | null>(null);
  const [unitsMeta, setUnitsMeta] = useState<BudgetUnitsMetadata | undefined>(undefined);
  const [zoneNames, setZoneNames] = useState<string[]>([]);
  const unitsSynced = useRef(false);

  // Load types and presets on mount
  useEffect(() => {
    fetchZBudgetTypes()
      .then((types) => {
        setLocalTypes(types);
        setZBudgetTypes(types);
      })
      .catch(console.error);

    fetchZBudgetPresets()
      .then(setPresets)
      .catch(console.error);
  }, [setZBudgetTypes]);

  // Load mesh GeoJSON for the zone map
  useEffect(() => {
    fetch('/api/mesh/geojson?layer=1')
      .then((r) => { if (r.ok) return r.json(); throw new Error('Failed'); })
      .then(setGeojson)
      .catch(console.error);
  }, []);

  // Auto-sync display units
  useEffect(() => {
    if (unitsSynced.current || !unitsMeta) return;
    if (unitsMeta.source_volume_unit) {
      const d = sourceVolumeToDisplayDefault(unitsMeta.source_volume_unit);
      if (d) setBudgetVolumeUnit(d);
    }
    if (unitsMeta.source_area_unit) {
      const d = sourceAreaToDisplayDefault(unitsMeta.source_area_unit);
      if (d) setBudgetAreaUnit(d);
    }
    unitsSynced.current = true;
  }, [unitsMeta, setBudgetVolumeUnit, setBudgetAreaUnit]);

  // Handle element click: assign to active paint zone
  const handleElementClick = useCallback((elementId: number) => {
    const current = useViewerStore.getState().zbudgetZones;
    const next = current.map((z) => ({
      ...z,
      elements: z.elements.filter((e) => e !== elementId),
    }));
    const target = next.find((z) => z.id === zbudgetPaintZoneId);
    if (target) {
      target.elements = [...target.elements, elementId];
    }
    setZBudgetZones(next);
  }, [zbudgetPaintZoneId, setZBudgetZones]);

  // Zone management callbacks
  const handleAddZone = useCallback(() => {
    const maxId = zbudgetZones.reduce((m, z) => Math.max(m, z.id), 0);
    const newZone: ZoneInfo = {
      id: maxId + 1,
      name: `Zone ${maxId + 1}`,
      elements: [],
    };
    setZBudgetZones([...zbudgetZones, newZone]);
  }, [zbudgetZones, setZBudgetZones]);

  const handleRemoveZone = useCallback((id: number) => {
    setZBudgetZones(zbudgetZones.filter((z) => z.id !== id));
  }, [zbudgetZones, setZBudgetZones]);

  const handleRenameZone = useCallback((id: number, name: string) => {
    setZBudgetZones(zbudgetZones.map((z) => z.id === id ? { ...z, name } : z));
  }, [zbudgetZones, setZBudgetZones]);

  const handleLoadPreset = useCallback((preset: ZBudgetPreset) => {
    setZBudgetZones(preset.zones);
  }, [setZBudgetZones]);

  const handleClearAll = useCallback(() => {
    setZBudgetZones([]);
  }, [setZBudgetZones]);

  // Run ZBudget: post zones then switch to chart view (data is fetched by useEffect)
  const handleRunZBudget = useCallback(async () => {
    if (zbudgetZones.length === 0) return;
    setLoading(true);
    try {
      await postZoneDefinition({
        zones: zbudgetZones,
        extent: 'horizontal',
      });
      // Set zone names for the zone selector
      const names = zbudgetZones.map((z) => z.name);
      setZoneNames(names);
      // Always update active zone to first zone (zones may have changed)
      setZBudgetActiveZone(names.length > 0 ? names[0] : '');
      // Switch to chart view — the useEffect will fetch data
      setZBudgetEditMode(false);
    } catch (err) {
      console.error('Failed to post zone definition:', err);
      setLoading(false);
    }
  }, [zbudgetZones, setZBudgetActiveZone, setZBudgetEditMode]);

  // Reload data when zone or type changes (chart mode)
  useEffect(() => {
    if (zbudgetEditMode || !zbudgetActiveType || !zbudgetActiveZone || zoneNames.length === 0) return;
    setLoading(true);
    fetchZBudgetData(zbudgetActiveType, zbudgetActiveZone)
      .then((data) => {
        setUnitsMeta(data.units_metadata);
        setZbudgetData(data);
        setLoading(false);
      })
      .catch((err) => {
        console.error('Failed to load zbudget data:', err);
        setZbudgetData(null);
        setLoading(false);
      });
  }, [zbudgetActiveType, zbudgetActiveZone, zbudgetEditMode, zoneNames.length]);

  // Classify + convert chart data
  const classified = useMemo(
    () => zbudgetData && zbudgetActiveType
      ? classifyColumns(zbudgetData, zbudgetActiveType)
      : null,
    [zbudgetData, zbudgetActiveType],
  );

  const convertedCharts = useMemo(() => {
    if (!classified) return [];
    return classified.charts.map((group) =>
      convertChartData(
        group, budgetDisplayMode, budgetVolumeUnit, budgetRateUnit,
        budgetAreaUnit, budgetLengthUnit, budgetTimeAgg, unitsMeta,
      ),
    );
  }, [classified, budgetDisplayMode, budgetVolumeUnit, budgetRateUnit, budgetAreaUnit, budgetLengthUnit, budgetTimeAgg, unitsMeta]);

  const chartKinds = useMemo(() => {
    if (!classified) return [];
    return classified.charts.map((g) => g.chartKind);
  }, [classified]);

  const budgetLabel = ZBUDGET_LABELS[zbudgetActiveType] || zbudgetActiveType;
  const contextPrefix = zbudgetActiveZone ? `${zbudgetActiveZone} \u2014 ` : '';
  const xAxisLabel = getXAxisLabel(budgetTimeAgg);

  // Empty state
  if (zbudgetTypes.length === 0 && storeTypes.length === 0) {
    return (
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
        <Typography color="text.secondary">
          No ZBudget data available. Load a model with ZBudget HDF5 output files.
        </Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ display: 'flex', height: '100%' }}>
      {/* Left sidebar */}
      <ZBudgetControls
        zbudgetTypes={zbudgetTypes}
        presets={presets}
        zones={zbudgetZones}
        onAddZone={handleAddZone}
        onRemoveZone={handleRemoveZone}
        onRenameZone={handleRenameZone}
        onLoadPreset={handleLoadPreset}
        onClearAll={handleClearAll}
        onRunZBudget={handleRunZBudget}
        loading={loading}
        unitsMeta={unitsMeta}
        zoneNames={zoneNames}
      />

      {/* Main content */}
      <Box sx={{ flexGrow: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        {zbudgetEditMode ? (
          /* Zone editor mode: show the map */
          <Box sx={{ flexGrow: 1, position: 'relative' }}>
            {geojson ? (
              <ZoneMap
                geojson={geojson}
                zones={zbudgetZones}
                paintZoneId={zbudgetPaintZoneId}
                onElementClick={handleElementClick}
              />
            ) : (
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
                <CircularProgress />
              </Box>
            )}
          </Box>
        ) : (
          /* Chart mode */
          <>
            {/* Analysis mode tabs */}
            <Tabs
              value={budgetAnalysisMode}
              onChange={(_, val) => setBudgetAnalysisMode(val)}
              variant="scrollable"
              scrollButtons="auto"
              sx={{ borderBottom: 1, borderColor: 'divider', minHeight: 36, flexShrink: 0 }}
            >
              <Tab label="Time Series" value="timeseries" sx={{ minHeight: 36, py: 0 }} />
              <Tab label="Monthly Pattern" value="monthly_pattern" sx={{ minHeight: 36, py: 0 }} />
              <Tab label="Component Ratios" value="component_ratios" sx={{ minHeight: 36, py: 0 }} />
              <Tab label="Cumulative Departure" value="cumulative_departure" sx={{ minHeight: 36, py: 0 }} />
              <Tab label="Exceedance" value="exceedance" sx={{ minHeight: 36, py: 0 }} />
            </Tabs>

            <Box sx={{ flexGrow: 1, overflow: 'auto', p: 1 }}>
              {loading ? (
                <Box sx={{ display: 'flex', justifyContent: 'center', pt: 4 }}>
                  <CircularProgress />
                </Box>
              ) : !zbudgetData || !classified ? (
                <Box sx={{ display: 'flex', justifyContent: 'center', pt: 4 }}>
                  <Typography color="text.secondary">
                    Define zones and click "Run ZBudget" to see charts.
                  </Typography>
                </Box>
              ) : budgetAnalysisMode === 'timeseries' ? (
                /* Time series charts */
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                  {convertedCharts.map((chart, i) => (
                    <Box key={i} sx={{ height: 350 }}>
                      <BudgetChart
                        data={chart.data}
                        chartType={chartKinds[i] === 'flow' ? budgetChartType : 'line'}
                        loading={false}
                        title={classified?.charts[i]?.title ?? ''}
                        yAxisLabel={chart.yAxisLabel}
                        xAxisLabel={xAxisLabel}
                        partialYearNote={chart.partialYearNote}
                        onExpand={() => setExpandedChartIndex(i)}
                      />
                    </Box>
                  ))}
                </Box>
              ) : budgetAnalysisMode === 'monthly_pattern' ? (
                <MonthlyPatternChart
                  classified={classified}
                  unitsMeta={unitsMeta}
                  volumeUnit={budgetVolumeUnit}
                  areaUnit={budgetAreaUnit}
                  contextPrefix={contextPrefix}
                  budgetLabel={budgetLabel}
                />
              ) : budgetAnalysisMode === 'component_ratios' ? (
                <ComponentRatioChart
                  budgetData={zbudgetData}
                  classified={classified}
                  unitsMeta={unitsMeta}
                  budgetType={zbudgetActiveType}
                  contextPrefix={contextPrefix}
                  budgetLabel={budgetLabel}
                />
              ) : budgetAnalysisMode === 'cumulative_departure' ? (
                <CumulativeDepartureChart
                  classified={classified}
                  unitsMeta={unitsMeta}
                  volumeUnit={budgetVolumeUnit}
                  contextPrefix={contextPrefix}
                  budgetLabel={budgetLabel}
                />
              ) : budgetAnalysisMode === 'exceedance' ? (
                <ExceedanceChart
                  classified={classified}
                  unitsMeta={unitsMeta}
                  volumeUnit={budgetVolumeUnit}
                  contextPrefix={contextPrefix}
                  budgetLabel={budgetLabel}
                />
              ) : null}
            </Box>
          </>
        )}
      </Box>

      {/* Fullscreen chart dialog */}
      <Dialog fullScreen open={expandedChartIndex !== null} onClose={() => setExpandedChartIndex(null)}>
        <DialogContent sx={{ p: 0, position: 'relative', height: '100vh' }}>
          <IconButton
            onClick={() => setExpandedChartIndex(null)}
            sx={{ position: 'absolute', top: 8, right: 8, zIndex: 10 }}
          >
            <CloseIcon />
          </IconButton>
          {expandedChartIndex !== null && expandedChartIndex < convertedCharts.length && (
            <BudgetChart
              data={convertedCharts[expandedChartIndex].data}
              chartType={chartKinds[expandedChartIndex] === 'flow' ? budgetChartType : 'line'}
              loading={false}
              title={classified?.charts[expandedChartIndex]?.title ?? ''}
              yAxisLabel={convertedCharts[expandedChartIndex].yAxisLabel}
              xAxisLabel={xAxisLabel}
            />
          )}
        </DialogContent>
      </Dialog>
    </Box>
  );
}
