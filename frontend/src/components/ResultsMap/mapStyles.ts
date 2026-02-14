/**
 * Map style configuration for MapLibre GL JS.
 */

/** Basemap catalog with free tile sources (no API key required) */
export const BASEMAPS: Record<string, { name: string; url: string; isRaster?: boolean }> = {
  positron: {
    name: 'Light',
    url: 'https://basemaps.cartocdn.com/gl/positron-gl-style/style.json',
  },
  dark: {
    name: 'Dark',
    url: 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
  },
  voyager: {
    name: 'Streets',
    url: 'https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json',
  },
  satellite: {
    name: 'Satellite',
    url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    isRaster: true,
  },
};

/** Free CARTO Positron basemap — clean, light background for data overlay */
export const MAP_STYLE = BASEMAPS.positron.url;

/**
 * Build a MapLibre style object for a raster tile source (e.g. satellite).
 */
export function buildRasterStyle(tileUrl: string): Record<string, unknown> {
  return {
    version: 8,
    sources: {
      'raster-tiles': {
        type: 'raster',
        tiles: [tileUrl],
        tileSize: 256,
      },
    },
    layers: [
      {
        id: 'raster-layer',
        type: 'raster',
        source: 'raster-tiles',
      },
    ],
  };
}

/**
 * Get the MapLibre style for a basemap key.
 */
export function getBasemapStyle(key: string): string | Record<string, unknown> {
  const bm = BASEMAPS[key] ?? BASEMAPS.positron;
  if (bm.isRaster) {
    return buildRasterStyle(bm.url);
  }
  return bm.url;
}

/** Viridis-like color scale for head values (blue -> green -> yellow -> red) */
export const VIRIDIS_COLORS: [number, number, number][] = [
  [68, 1, 84],
  [72, 35, 116],
  [64, 67, 135],
  [52, 94, 141],
  [41, 120, 142],
  [32, 144, 140],
  [34, 167, 132],
  [68, 190, 112],
  [121, 209, 81],
  [189, 222, 38],
  [253, 231, 37],
];

/**
 * Interpolate a Viridis-like color from normalized value [0, 1].
 * Returns [r, g, b, a] with alpha 0-255.
 */
export function interpolateColor(
  t: number,
  opacity: number = 180,
): [number, number, number, number] {
  const clamped = Math.max(0, Math.min(1, t));
  const idx = clamped * (VIRIDIS_COLORS.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.min(lo + 1, VIRIDIS_COLORS.length - 1);
  const frac = idx - lo;

  const cLo = VIRIDIS_COLORS[lo];
  const cHi = VIRIDIS_COLORS[hi];

  return [
    Math.round(cLo[0] + (cHi[0] - cLo[0]) * frac),
    Math.round(cLo[1] + (cHi[1] - cLo[1]) * frac),
    Math.round(cLo[2] + (cHi[2] - cLo[2]) * frac),
    opacity,
  ];
}

/** Diverging color scale: red (negative) -> white (zero) -> blue (positive) */
const DIVERGING_COLORS: [number, number, number][] = [
  [178, 24, 43],
  [214, 96, 77],
  [244, 165, 130],
  [253, 219, 199],
  [247, 247, 247],
  [209, 229, 240],
  [146, 197, 222],
  [67, 147, 195],
  [33, 102, 172],
];

/**
 * Interpolate a diverging color from normalized value [0, 1].
 * 0 = red (drawdown), 0.5 = white (no change), 1 = blue (rise).
 */
export function interpolateDivergingColor(
  t: number,
  opacity: number = 180,
): [number, number, number, number] {
  const clamped = Math.max(0, Math.min(1, t));
  const idx = clamped * (DIVERGING_COLORS.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.min(lo + 1, DIVERGING_COLORS.length - 1);
  const frac = idx - lo;

  const cLo = DIVERGING_COLORS[lo];
  const cHi = DIVERGING_COLORS[hi];

  return [
    Math.round(cLo[0] + (cHi[0] - cLo[0]) * frac),
    Math.round(cLo[1] + (cHi[1] - cLo[1]) * frac),
    Math.round(cLo[2] + (cHi[2] - cLo[2]) * frac),
    opacity,
  ];
}

/** Lake blue fill color */
export const LAKE_FILL_COLOR: [number, number, number, number] = [100, 160, 230, 120];
export const LAKE_OUTLINE_COLOR: [number, number, number, number] = [30, 80, 180, 200];

/** Boundary condition type colors */
export const BC_COLORS: Record<string, [number, number, number, number]> = {
  specified_head: [30, 100, 200, 200],   // Blue
  specified_flow: [50, 170, 50, 200],     // Green
  general_head: [230, 150, 30, 200],      // Orange
  constrained_general_head: [180, 50, 180, 200],  // Purple
};

/** Watershed colors */
export const WATERSHED_MARKER_COLOR: [number, number, number, number] = [34, 120, 60, 180];
export const WATERSHED_SELECTED_COLOR: [number, number, number, number] = [50, 180, 80, 255];
export const WATERSHED_GW_PERC_COLOR: [number, number, number, number] = [230, 140, 30, 200];
export const WATERSHED_GW_BASEFLOW_COLOR: [number, number, number, number] = [140, 50, 180, 200];
export const WATERSHED_DEST_COLOR: [number, number, number, number] = [30, 100, 220, 220];

/** Diversion colors */
export const DIVERSION_SOURCE_COLOR: [number, number, number, number] = [220, 120, 20, 180];
export const DIVERSION_SELECTED_COLOR: [number, number, number, number] = [255, 140, 0, 255];
export const DIVERSION_DELIVERY_FILL: [number, number, number, number] = [255, 165, 0, 80];
export const DIVERSION_DELIVERY_OUTLINE: [number, number, number, number] = [255, 140, 0, 200];
