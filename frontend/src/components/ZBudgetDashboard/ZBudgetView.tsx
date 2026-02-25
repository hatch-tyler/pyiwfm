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
import { ZoneUploadDialog } from './ZoneUploadDialog';
import type { SelectionMode } from './ZoneSelectionToolbar';
import { BudgetChart } from '../BudgetDashboard/BudgetChart';
import { MonthlyPatternChart } from '../BudgetDashboard/MonthlyPatternChart';
import { ComponentRatioChart } from '../BudgetDashboard/ComponentRatioChart';
import { CumulativeDepartureChart } from '../BudgetDashboard/CumulativeDepartureChart';
import { ExceedanceChart } from '../BudgetDashboard/ExceedanceChart';
import { classifyZBudgetColumns } from './zbudgetClassifier';
import type { ZBudgetChartGroup } from './zbudgetClassifier';
import { InflowOutflowChart } from './InflowOutflowChart';
import { StorageChangeChart } from './StorageChangeChart';
import {
  getXAxisLabel,
  sourceVolumeToDisplayDefault,
  sourceAreaToDisplayDefault,
} from '../BudgetDashboard/budgetUnits';
import { convertChartData } from '../BudgetDashboard/convertChartData';

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
  const [fetchError, setFetchError] = useState<string | null>(null);
  const [selectionMode, setSelectionMode] = useState<SelectionMode>('point');
  const [uploadDialogOpen, setUploadDialogOpen] = useState(false);
  const unitsSynced = useRef(false);
  const skipNextFetch = useRef(false);

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

  // Precompute element centroids from GeoJSON for spatial selection
  const elementCentroids = useMemo(() => {
    const map = new Map<number, [number, number]>();
    if (!geojson) return map;
    for (const f of geojson.features) {
      const props = f.properties as Record<string, unknown>;
      const elemId = props?.element_id as number;
      if (!elemId) continue;
      const coords = (f.geometry as GeoJSON.Polygon).coordinates[0];
      let sumLng = 0, sumLat = 0;
      for (const [lng, lat] of coords) {
        sumLng += lng;
        sumLat += lat;
      }
      map.set(elemId, [sumLng / coords.length, sumLat / coords.length]);
    }
    return map;
  }, [geojson]);

  // Handle element click: assign to active paint zone (Ctrl+click deselects)
  const handleElementClick = useCallback((elementId: number, ctrlKey: boolean) => {
    const current = useViewerStore.getState().zbudgetZones;
    if (ctrlKey) {
      // Deselect: remove element from whichever zone it belongs to
      const next = current.map((z) => ({
        ...z,
        elements: z.elements.filter((e) => e !== elementId),
      }));
      setZBudgetZones(next);
      return;
    }
    // Normal click: remove from all zones, add to paint zone
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

  // Handle shape selection (rectangle/polygon): batch-assign to paint zone
  const handleShapeSelect = useCallback((elementIds: number[]) => {
    const current = useViewerStore.getState().zbudgetZones;
    const idSet = new Set(elementIds);
    const next = current.map((z) => ({
      ...z,
      elements: z.elements.filter((e) => !idSet.has(e)),
    }));
    const target = next.find((z) => z.id === zbudgetPaintZoneId);
    if (target) {
      target.elements = [...target.elements, ...elementIds];
    }
    setZBudgetZones(next);
  }, [zbudgetPaintZoneId, setZBudgetZones]);

  // Handle zones imported from file upload
  const handleZonesImported = useCallback((zones: ZoneInfo[]) => {
    setZBudgetZones(zones);
  }, [setZBudgetZones]);

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

  // Run ZBudget: post zones, fetch data, then switch to chart view
  const handleRunZBudget = useCallback(async () => {
    if (zbudgetZones.length === 0) return;
    setLoading(true);
    setFetchError(null);
    try {
      await postZoneDefinition({
        zones: zbudgetZones,
        extent: 'horizontal',
      });
      // Set zone names for the zone selector
      const names = zbudgetZones.map((z) => z.name);
      setZoneNames(names);
      // Always update active zone to first zone (zones may have changed)
      const activeZone = names.length > 0 ? names[0] : '';
      setZBudgetActiveZone(activeZone);
      // Fetch data for first zone and active type
      if (zbudgetActiveType && activeZone) {
        const data = await fetchZBudgetData(zbudgetActiveType, activeZone);
        setUnitsMeta(data.units_metadata);
        setZbudgetData(data);
        unitsSynced.current = false;
      }
      // Prevent useEffect from re-fetching the same data
      skipNextFetch.current = true;
      // Switch to chart view so results are visible
      setZBudgetEditMode(false);
    } catch (err) {
      console.error('Failed to run zbudget:', err);
      setFetchError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, [zbudgetZones, zbudgetActiveType, setZBudgetActiveZone, setZBudgetEditMode]);

  // Reload data when zone or type changes (chart mode only)
  useEffect(() => {
    if (zbudgetEditMode || !zbudgetActiveType || !zbudgetActiveZone) return;
    // Skip if data was just loaded by handleRunZBudget
    if (skipNextFetch.current) {
      skipNextFetch.current = false;
      return;
    }
    setLoading(true);
    setFetchError(null);
    fetchZBudgetData(zbudgetActiveType, zbudgetActiveZone)
      .then((data) => {
        setUnitsMeta(data.units_metadata);
        setZbudgetData(data);
      })
      .catch((err) => {
        console.error('Failed to load zbudget data:', err);
        setFetchError(err instanceof Error ? err.message : String(err));
        setZbudgetData(null);
      })
      .finally(() => setLoading(false));
  }, [zbudgetActiveType, zbudgetActiveZone, zbudgetEditMode]);

  // Classify + convert chart data
  const classified = useMemo(
    () => zbudgetData && zbudgetActiveType
      ? classifyZBudgetColumns(zbudgetData, zbudgetActiveType)
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
                selectionMode={selectionMode}
                onSelectionModeChange={setSelectionMode}
                onElementClick={handleElementClick}
                onShapeSelect={handleShapeSelect}
                elementCentroids={elementCentroids}
                onUploadClick={() => setUploadDialogOpen(true)}
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
                <Box sx={{ display: 'flex', justifyContent: 'center', pt: 4, flexDirection: 'column', alignItems: 'center', gap: 1 }}>
                  <Typography color="text.secondary">
                    {fetchError
                      ? 'Failed to load ZBudget data. Check the server logs for details.'
                      : 'Define zones and click "Run ZBudget" to see charts.'}
                  </Typography>
                  {fetchError && (
                    <Typography variant="caption" color="error" sx={{ maxWidth: 500, textAlign: 'center' }}>
                      {fetchError}
                    </Typography>
                  )}
                </Box>
              ) : budgetAnalysisMode === 'timeseries' ? (
                /* Time series charts */
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                  {convertedCharts.map((chart, i) => {
                    const group = classified?.charts[i] as ZBudgetChartGroup | undefined;
                    const kind = chartKinds[i];
                    const isInflowOutflow = group?.zbudgetKind === 'inflow_outflow' && budgetChartType === 'bar';
                    const isStorage = kind === 'storage';

                    return (
                      <Box key={i} sx={{ height: 350 }}>
                        {isInflowOutflow ? (
                          <InflowOutflowChart
                            data={chart.data}
                            yAxisLabel={chart.yAxisLabel}
                            title={group?.title ?? ''}
                            xAxisLabel={xAxisLabel}
                            partialYearNote={chart.partialYearNote}
                            onExpand={() => setExpandedChartIndex(i)}
                          />
                        ) : isStorage ? (
                          <StorageChangeChart
                            data={chart.data}
                            yAxisLabel={chart.yAxisLabel}
                            title={group?.title ?? ''}
                            xAxisLabel={xAxisLabel}
                            partialYearNote={chart.partialYearNote}
                            onExpand={() => setExpandedChartIndex(i)}
                          />
                        ) : (
                          <BudgetChart
                            data={chart.data}
                            chartType={kind === 'flow' ? budgetChartType : 'line'}
                            loading={false}
                            title={group?.title ?? ''}
                            yAxisLabel={chart.yAxisLabel}
                            xAxisLabel={xAxisLabel}
                            partialYearNote={chart.partialYearNote}
                            onExpand={() => setExpandedChartIndex(i)}
                          />
                        )}
                      </Box>
                    );
                  })}
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

      {/* Zone file upload dialog */}
      <ZoneUploadDialog
        open={uploadDialogOpen}
        onClose={() => setUploadDialogOpen(false)}
        onZonesImported={handleZonesImported}
      />

      {/* Fullscreen chart dialog */}
      <Dialog fullScreen open={expandedChartIndex !== null} onClose={() => setExpandedChartIndex(null)}>
        <DialogContent sx={{ p: 0, position: 'relative', height: '100vh' }}>
          <IconButton
            onClick={() => setExpandedChartIndex(null)}
            sx={{ position: 'absolute', top: 8, right: 8, zIndex: 10 }}
          >
            <CloseIcon />
          </IconButton>
          {expandedChartIndex !== null && expandedChartIndex < convertedCharts.length && (() => {
            const eGroup = classified?.charts[expandedChartIndex] as ZBudgetChartGroup | undefined;
            const eKind = chartKinds[expandedChartIndex];
            const eChart = convertedCharts[expandedChartIndex];
            const eIsInflowOutflow = eGroup?.zbudgetKind === 'inflow_outflow' && budgetChartType === 'bar';
            const eIsStorage = eKind === 'storage';

            if (eIsInflowOutflow) {
              return (
                <InflowOutflowChart
                  data={eChart.data}
                  yAxisLabel={eChart.yAxisLabel}
                  title={eGroup?.title ?? ''}
                  xAxisLabel={xAxisLabel}
                />
              );
            }
            if (eIsStorage) {
              return (
                <StorageChangeChart
                  data={eChart.data}
                  yAxisLabel={eChart.yAxisLabel}
                  title={eGroup?.title ?? ''}
                  xAxisLabel={xAxisLabel}
                />
              );
            }
            return (
              <BudgetChart
                data={eChart.data}
                chartType={eKind === 'flow' ? budgetChartType : 'line'}
                loading={false}
                title={eGroup?.title ?? ''}
                yAxisLabel={eChart.yAxisLabel}
                xAxisLabel={xAxisLabel}
              />
            );
          })()}
        </DialogContent>
      </Dialog>
    </Box>
  );
}
