/**
 * Pure geometry utilities for spatial selection (no external dependencies).
 */

/** Check if a point [lng, lat] is inside an axis-aligned rectangle. */
export function pointInRect(
  point: [number, number],
  minLng: number,
  minLat: number,
  maxLng: number,
  maxLat: number,
): boolean {
  return point[0] >= minLng && point[0] <= maxLng && point[1] >= minLat && point[1] <= maxLat;
}

/**
 * Ray-casting point-in-polygon test.
 * `polygon` is an array of [lng, lat] vertices (not closed — last != first).
 */
export function pointInPolygon(
  point: [number, number],
  polygon: [number, number][],
): boolean {
  const [px, py] = point;
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const [xi, yi] = polygon[i];
    const [xj, yj] = polygon[j];
    if ((yi > py) !== (yj > py) && px < ((xj - xi) * (py - yi)) / (yj - yi) + xi) {
      inside = !inside;
    }
  }
  return inside;
}

/** Return element IDs whose centroids fall within the given rectangle bounds (geo coords). */
export function elementsInRect(
  centroids: Map<number, [number, number]>,
  minLng: number,
  minLat: number,
  maxLng: number,
  maxLat: number,
): number[] {
  const result: number[] = [];
  centroids.forEach((pt, elemId) => {
    if (pointInRect(pt, minLng, minLat, maxLng, maxLat)) {
      result.push(elemId);
    }
  });
  return result;
}

/** Return element IDs whose centroids fall within the given polygon (geo coords). */
export function elementsInPolygon(
  centroids: Map<number, [number, number]>,
  polygon: [number, number][],
): number[] {
  const result: number[] = [];
  centroids.forEach((pt, elemId) => {
    if (pointInPolygon(pt, polygon)) {
      result.push(elemId);
    }
  });
  return result;
}
