/**
 * Budget Dashboard tab: budget type/location selection + charts.
 * Renders multiple chart groups based on budget type classification,
 * applies source-unit-aware conversions and time aggregation.
 */

import { useState, useEffect, useMemo, useRef } from 'react';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Dialog from '@mui/material/Dialog';
import DialogContent from '@mui/material/DialogContent';
import IconButton from '@mui/material/IconButton';
import Tabs from '@mui/material/Tabs';
import Tab from '@mui/material/Tab';
import CloseIcon from '@mui/icons-material/Close';
import { useViewerStore } from '../../stores/viewerStore';
import { fetchBudgetTypes, fetchBudgetData } from '../../api/client';
import type { BudgetData, BudgetUnitsMetadata } from '../../api/client';
import { BudgetControls, BUDGET_LABELS } from './BudgetControls';
import { BudgetChart } from './BudgetChart';
import { DiversionBalanceChart } from './DiversionBalanceChart';
import { WaterBalanceSankey } from './WaterBalanceSankey';
import { BudgetLocationMap } from './BudgetLocationMap';
import { classifyColumns } from './budgetSplitter';
import type { ChartGroup, ChartKind } from './budgetSplitter';
import {
  convertVolumeValues,
  convertAreaValues,
  getYAxisLabel,
  sourceVolumeToDisplayDefault,
  sourceAreaToDisplayDefault,
  sourceLengthToDisplayDefault,
} from './budgetUnits';
import { MonthlyPatternChart } from './MonthlyPatternChart';
import { ComponentRatioChart } from './ComponentRatioChart';
import { CumulativeDepartureChart } from './CumulativeDepartureChart';
import { ExceedanceChart } from './ExceedanceChart';

interface ConvertedChart {
  data: BudgetData;
  yAxisLabel: string;
  partialYearNote?: string;
}

/** Apply unit conversion to a single chart group's data. */
function convertChartData(
  group: ChartGroup,
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
        col.values, group.data.times, sourceVolume, volumeUnit, rateUnit, timeAgg,
      );
      if (!firstPartialNote && result.partialYearNote) firstPartialNote = result.partialYearNote;
      return { name: col.name, values: result.values, units: col.units };
    }
  });

  // Get the converted time axis from the first column
  let convertedTimes = group.data.times;
  if (group.data.columns.length > 0) {
    if (isArea) {
      const result = convertAreaValues(
        group.data.columns[0].values, group.data.times, sourceArea, areaUnit, timeAgg,
      );
      convertedTimes = result.times;
    } else {
      const result = convertVolumeValues(
        group.data.columns[0].values, group.data.times, sourceVolume, volumeUnit, rateUnit, timeAgg,
      );
      convertedTimes = result.times;
    }
  }

  const yAxisLabel = getYAxisLabel(group.chartKind, volumeUnit, rateUnit, areaUnit, lengthUnit, timeAgg);

  return {
    data: {
      location: group.data.location,
      times: convertedTimes,
      columns: convertedColumns,
    },
    yAxisLabel,
    partialYearNote: firstPartialNote,
  };
}

/** Determine chart type for a given chart kind. */
function getChartTypeForKind(
  kind: ChartKind,
  userChartType: 'area' | 'bar' | 'line',
): 'area' | 'bar' | 'line' {
  // Flow charts use user's selected type; storage/cumulative_subsidence/area use line
  if (kind === 'flow') return userChartType;
  if (kind === 'diversion_balance') return 'bar';
  if (kind === 'cumulative_subsidence') return 'line';
  return 'line';
}

export function BudgetView() {
  const {
    resultsInfo,
    activeBudgetType, activeBudgetLocation, budgetChartType,
    showBudgetSankey,
    budgetVolumeUnit, budgetRateUnit, budgetAreaUnit, budgetLengthUnit, budgetTimeAgg,
    budgetAnalysisMode,
    expandedChartIndex, setExpandedChartIndex,
    setBudgetVolumeUnit, setBudgetAreaUnit, setBudgetLengthUnit,
    setBudgetAnalysisMode,
  } = useViewerStore();

  const [budgetTypes, setBudgetTypes] = useState<string[]>([]);
  const [budgetData, setBudgetData] = useState<BudgetData | null>(null);
  const [unitsMeta, setUnitsMeta] = useState<BudgetUnitsMetadata | undefined>(undefined);
  const [loading, setLoading] = useState(false);
  const unitsSynced = useRef(false);

  // Reset unit sync flag when budget type or location changes
  useEffect(() => {
    unitsSynced.current = false;
  }, [activeBudgetType, activeBudgetLocation]);

  // Auto-sync display units from source metadata
  useEffect(() => {
    if (unitsSynced.current || !unitsMeta) return;
    if (unitsMeta.source_volume_unit) {
      const defaultVol = sourceVolumeToDisplayDefault(unitsMeta.source_volume_unit);
      if (defaultVol) setBudgetVolumeUnit(defaultVol);
    }
    if (unitsMeta.source_area_unit) {
      const defaultArea = sourceAreaToDisplayDefault(unitsMeta.source_area_unit);
      if (defaultArea) setBudgetAreaUnit(defaultArea);
    }
    if (unitsMeta.source_length_unit) {
      const defaultLen = sourceLengthToDisplayDefault(unitsMeta.source_length_unit);
      if (defaultLen) setBudgetLengthUnit(defaultLen);
    }
    unitsSynced.current = true;
  }, [unitsMeta, setBudgetVolumeUnit, setBudgetAreaUnit, setBudgetLengthUnit]);

  // Load available budget types
  useEffect(() => {
    if (resultsInfo?.available_budgets) {
      setBudgetTypes(resultsInfo.available_budgets);
    } else {
      fetchBudgetTypes()
        .then(setBudgetTypes)
        .catch(console.error);
    }
  }, [resultsInfo]);

  // Load budget data when type/location changes
  useEffect(() => {
    if (!activeBudgetType || !activeBudgetLocation) {
      setBudgetData(null);
      setUnitsMeta(undefined);
      return;
    }

    setLoading(true);
    fetchBudgetData(activeBudgetType, activeBudgetLocation)
      .then((data) => {
        setUnitsMeta(data.units_metadata);
        setBudgetData(data);
        setLoading(false);
      })
      .catch((err) => {
        console.error('Failed to load budget data:', err);
        setBudgetData(null);
        setUnitsMeta(undefined);
        setLoading(false);
      });
  }, [activeBudgetType, activeBudgetLocation]);

  // Classify columns into chart groups
  const classified = useMemo(
    () => budgetData && activeBudgetType
      ? classifyColumns(budgetData, activeBudgetType)
      : null,
    [budgetData, activeBudgetType],
  );

  // Apply unit conversions to each chart group
  const convertedCharts = useMemo(() => {
    if (!classified) return [];
    return classified.charts.map((group) =>
      convertChartData(
        group, budgetVolumeUnit, budgetRateUnit, budgetAreaUnit,
        budgetLengthUnit, budgetTimeAgg, unitsMeta,
      ),
    );
  }, [classified, budgetVolumeUnit, budgetRateUnit, budgetAreaUnit, budgetLengthUnit, budgetTimeAgg, unitsMeta]);

  // Keep track of chart kinds for determining chart type
  const chartKinds = useMemo(() => {
    if (!classified) return [];
    return classified.charts.map((g) => g.chartKind);
  }, [classified]);

  // Build contextual titles: "{location} — {budgetType}: {splitterTitle}"
  const budgetLabel = BUDGET_LABELS[activeBudgetType] || activeBudgetType;
  const contextPrefix = activeBudgetLocation ? `${activeBudgetLocation} \u2014 ` : '';

  const chartTitles = useMemo(() => {
    if (!classified) return [];
    return classified.charts.map((g) => {
      const base = g.title;
      // If splitter title already contains the location, just append budget label
      if (activeBudgetLocation && base.includes(activeBudgetLocation)) {
        return `${base} \u2014 ${budgetLabel}`;
      }
      return `${contextPrefix}${budgetLabel}: ${base}`;
    });
  }, [classified, activeBudgetLocation, budgetLabel, contextPrefix]);

  // Compute hasLengthColumns from units metadata (for subsidence length unit selector)
  const hasLengthColumns = unitsMeta?.has_length_columns ?? false;

  if (budgetTypes.length === 0) {
    return (
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
        <Typography color="text.secondary">
          No budget data available. Load a model with simulation results.
        </Typography>
      </Box>
    );
  }

  const showMap = !!(activeBudgetType && activeBudgetLocation && !showBudgetSankey);

  return (
    <Box sx={{ display: 'flex', height: '100%' }}>
      {/* Left sidebar controls */}
      <BudgetControls
        budgetTypes={budgetTypes}
        hasLengthColumns={hasLengthColumns}
        unitsMeta={unitsMeta}
      />

      {/* Main chart area */}
      <Box sx={{ flexGrow: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        {/* Analysis mode tabs */}
        {!showBudgetSankey && (
          <Tabs
            value={budgetAnalysisMode}
            onChange={(_, val) => setBudgetAnalysisMode(val)}
            variant="scrollable"
            scrollButtons="auto"
            sx={{ borderBottom: 1, borderColor: 'divider', flexShrink: 0 }}
          >
            <Tab label="Time Series" value="timeseries" />
            <Tab label="Monthly Pattern" value="monthly_pattern" />
            <Tab label="Component Ratios" value="component_ratios" />
            <Tab label="Cumulative Departure" value="cumulative_departure" />
            <Tab label="Exceedance" value="exceedance" />
          </Tabs>
        )}

        <Box sx={{ flexGrow: 1, overflow: 'hidden' }}>
        {showBudgetSankey ? (
          <WaterBalanceSankey />
        ) : budgetAnalysisMode !== 'timeseries' && budgetData && classified ? (
          budgetAnalysisMode === 'monthly_pattern' ? (
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
              budgetData={budgetData}
              classified={classified}
              unitsMeta={unitsMeta}
              budgetType={activeBudgetType}
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
          ) : null
        ) : convertedCharts.length > 1 ? (
          <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'auto' }}>
            {convertedCharts.map((chart, i) => {
              const kind = chartKinds[i];
              const isStorage = kind === 'storage';
              return (
                <Box
                  key={chartTitles[i]}
                  sx={{
                    flex: kind === 'flow' ? 2 : 1,
                    minHeight: kind === 'flow' ? 250 : 200,
                  }}
                >
                  {kind === 'diversion_balance' ? (
                    <DiversionBalanceChart
                      data={chart.data}
                      yAxisLabel={chart.yAxisLabel}

                      title={chartTitles[i]}
                      partialYearNote={chart.partialYearNote}
                      onExpand={() => setExpandedChartIndex(i)}
                    />
                  ) : (
                    <BudgetChart
                      data={chart.data}
                      chartType={getChartTypeForKind(kind, budgetChartType)}
                      loading={loading && i === 0}
                      title={chartTitles[i]}
                      dualAxis={isStorage}
                      yAxisLabel={chart.yAxisLabel}

                      partialYearNote={chart.partialYearNote}
                      onExpand={() => setExpandedChartIndex(i)}
                    />
                  )}
                </Box>
              );
            })}
          </Box>
        ) : convertedCharts.length === 1 ? (
          chartKinds[0] === 'diversion_balance' ? (
            <DiversionBalanceChart
              data={convertedCharts[0].data}
              yAxisLabel={convertedCharts[0].yAxisLabel}
              title={chartTitles[0]}
              partialYearNote={convertedCharts[0].partialYearNote}
              onExpand={() => setExpandedChartIndex(0)}
            />
          ) : (
            <BudgetChart
              data={convertedCharts[0].data}
              chartType={getChartTypeForKind(chartKinds[0], budgetChartType)}
              loading={loading}
              title={chartTitles[0]}
              yAxisLabel={convertedCharts[0].yAxisLabel}
              partialYearNote={convertedCharts[0].partialYearNote}
              onExpand={() => setExpandedChartIndex(0)}
            />
          )
        ) : (
          <BudgetChart
            data={budgetData}
            chartType={budgetChartType}
            loading={loading}
          />
        )}
        </Box>
      </Box>

      {/* Right-side location map column */}
      {showMap && (
        <Box sx={{
          width: 280, flexShrink: 0, borderLeft: 1, borderColor: 'divider',
          display: 'flex', flexDirection: 'column',
        }}>
          <Box sx={{ width: '100%', height: 280, overflow: 'hidden' }}>
            <BudgetLocationMap budgetType={activeBudgetType} locationName={activeBudgetLocation} />
          </Box>
          <Typography
            variant="caption"
            sx={{ px: 1, py: 0.5, textAlign: 'center', color: 'text.secondary' }}
          >
            {activeBudgetLocation}
          </Typography>
        </Box>
      )}

      {/* Fullscreen chart dialog */}
      <Dialog
        fullScreen
        open={expandedChartIndex !== null}
        onClose={() => setExpandedChartIndex(null)}
      >
        <DialogContent sx={{ p: 0, position: 'relative', height: '100vh' }}>
          <IconButton
            onClick={() => setExpandedChartIndex(null)}
            sx={{ position: 'absolute', top: 8, right: 8, zIndex: 10 }}
          >
            <CloseIcon />
          </IconButton>
          {expandedChartIndex !== null && expandedChartIndex < convertedCharts.length && (
            chartKinds[expandedChartIndex] === 'diversion_balance' ? (
              <DiversionBalanceChart
                data={convertedCharts[expandedChartIndex].data}
                yAxisLabel={convertedCharts[expandedChartIndex].yAxisLabel}
                title={chartTitles[expandedChartIndex]}
                partialYearNote={convertedCharts[expandedChartIndex].partialYearNote}
              />
            ) : (
              <BudgetChart
                data={convertedCharts[expandedChartIndex].data}
                chartType={getChartTypeForKind(chartKinds[expandedChartIndex], budgetChartType)}
                loading={false}
                title={chartTitles[expandedChartIndex]}
                dualAxis={chartKinds[expandedChartIndex] === 'storage'}
                yAxisLabel={convertedCharts[expandedChartIndex].yAxisLabel}
                partialYearNote={convertedCharts[expandedChartIndex].partialYearNote}
              />
            )
          )}
        </DialogContent>
      </Dialog>
    </Box>
  );
}
