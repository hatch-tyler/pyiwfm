/**
 * Layer color constants for consistent coloring across components.
 */

/** RGB colors (0-1 range) for vtk.js actors — indexed by layer-1. */
export const LAYER_COLORS: [number, number, number][] = [
  [0.12, 0.47, 0.71], // Layer 1 - Steel blue
  [1.0, 0.5, 0.05],   // Layer 2 - Orange
  [0.17, 0.63, 0.17],  // Layer 3 - Forest green
  [0.84, 0.15, 0.16],  // Layer 4 - Crimson
  [0.58, 0.4, 0.74],   // Layer 5 - Purple
  [0.55, 0.34, 0.29],  // Layer 6 - Brown
];

/** CSS hex versions for MUI components — indexed by layer-1. */
export const LAYER_COLORS_HEX = [
  '#1f77b4', // Layer 1
  '#ff7f0e', // Layer 2
  '#2ca02c', // Layer 3
  '#d62728', // Layer 4
  '#9467bd', // Layer 5
  '#8c564b', // Layer 6
];
