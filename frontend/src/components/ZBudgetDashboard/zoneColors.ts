/**
 * Zone color palette utilities for the ZBudget zone map.
 */

const ZONE_PALETTE = [
  '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
  '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
];

export function getZoneColor(zoneId: number): string {
  return ZONE_PALETTE[(zoneId - 1) % ZONE_PALETTE.length];
}

export function getUnassignedColor(): string {
  return '#e0e0e0';
}

/**
 * Convert hex color to RGBA array for deck.gl.
 */
export function hexToRgba(hex: string, alpha: number = 200): [number, number, number, number] {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return [r, g, b, alpha];
}
