/**
 * Shared chart-data conversion: unit-aware volume/area/length transforms
 * used by both BudgetView and ZBudgetView.
 */

import type { BudgetData, BudgetUnitsMetadata } from '../../api/client';
import type { ChartGroup } from './budgetSplitter';
import {
  convertVolumeValues,
  convertAreaValues,
  getYAxisLabel,
} from './budgetUnits';

export interface ConvertedChart {
  data: BudgetData;
  yAxisLabel: string;
  partialYearNote?: string;
}

/** Apply unit conversion to a single chart group's data. */
export function convertChartData(
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
        group.data.columns[0].values, group.data.times, sourceVolume, displayMode, volumeUnit, rateUnit, timeAgg,
      );
      convertedTimes = result.times;
    }
  }

  const yAxisLabel = getYAxisLabel(group.chartKind, displayMode, volumeUnit, rateUnit, areaUnit, lengthUnit);

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
