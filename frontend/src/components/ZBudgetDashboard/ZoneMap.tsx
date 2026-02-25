/**
 * ZoneMap: deck.gl element map for interactive zone assignment.
 * Supports point-click, rectangle drag-select, and polygon draw-select.
 * Ctrl+click deselects (removes element from its zone).
 *
 * Architecture: the SVG overlay is purely visual (pointerEvents: 'none').
 * A transparent interaction div sits on top for drawing events and forwards
 * wheel events to the deck.gl canvas so zoom always works.
 */

import { useState, useEffect, useMemo, useCallback, useRef } from 'react';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import DeckGL from '@deck.gl/react';
import { GeoJsonLayer } from '@deck.gl/layers';
import { WebMercatorViewport } from '@deck.gl/core';
import type { PickingInfo } from '@deck.gl/core';
import { Map } from 'react-map-gl/maplibre';
import 'maplibre-gl/dist/maplibre-gl.css';
import { useViewerStore } from '../../stores/viewerStore';
import type { ZoneInfo } from '../../api/client';
import { getZoneColor, getUnassignedColor, hexToRgba } from './zoneColors';
import { ZoneSelectionToolbar } from './ZoneSelectionToolbar';
import type { SelectionMode } from './ZoneSelectionToolbar';
import { elementsInRect, elementsInPolygon } from './spatialUtils';

const BASEMAP = 'https://basemaps.cartocdn.com/gl/positron-gl-style/style.json';

interface ViewState {
  longitude: number;
  latitude: number;
  zoom: number;
  pitch: number;
  bearing: number;
}

interface ZoneMapProps {
  geojson: GeoJSON.FeatureCollection | null;
  zones: ZoneInfo[];
  paintZoneId: number;
  selectionMode: SelectionMode;
  onSelectionModeChange: (mode: SelectionMode) => void;
  onElementClick: (elementId: number, ctrlKey: boolean) => void;
  onShapeSelect: (elementIds: number[]) => void;
  elementCentroids: Map<number, [number, number]>;
  onUploadClick: () => void;
}

export function ZoneMap({
  geojson, zones, paintZoneId,
  selectionMode, onSelectionModeChange,
  onElementClick, onShapeSelect,
  elementCentroids, onUploadClick,
}: ZoneMapProps) {
  const { selectedBasemap } = useViewerStore();

  const [viewState, setViewState] = useState<ViewState>({
    longitude: -120.5,
    latitude: 37.5,
    zoom: 6,
    pitch: 0,
    bearing: 0,
  });

  const [tooltip, setTooltip] = useState<{ x: number; y: number; text: string } | null>(null);

  // Ctrl key tracking
  const ctrlKeyRef = useRef(false);
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => { if (e.key === 'Control') ctrlKeyRef.current = true; };
    const onKeyUp = (e: KeyboardEvent) => { if (e.key === 'Control') ctrlKeyRef.current = false; };
    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);
    return () => {
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup', onKeyUp);
    };
  }, []);

  // ---- Drawing state using refs (synchronous) + state (for visual render) ----
  const containerRef = useRef<HTMLDivElement>(null);

  // Rectangle
  const rectStartRef = useRef<[number, number] | null>(null);
  const rectEndRef = useRef<[number, number] | null>(null);
  const isDrawingRect = useRef(false);
  const [rectVisual, setRectVisual] = useState<{ start: [number, number]; end: [number, number] } | null>(null);

  // Polygon
  const polyVertsRef = useRef<[number, number][]>([]);
  const [polyVisual, setPolyVisual] = useState<[number, number][]>([]);
  const [polyPreviewPt, setPolyPreviewPt] = useState<[number, number] | null>(null);

  // Stable ref for viewState so screenToGeo never uses a stale closure
  const viewStateRef = useRef(viewState);
  viewStateRef.current = viewState;

  // Stable refs for callbacks that need latest values without re-creating
  const elementCentroidsRef = useRef(elementCentroids);
  elementCentroidsRef.current = elementCentroids;
  const onShapeSelectRef = useRef(onShapeSelect);
  onShapeSelectRef.current = onShapeSelect;

  // Reset drawing state when mode changes
  useEffect(() => {
    rectStartRef.current = null;
    rectEndRef.current = null;
    isDrawingRect.current = false;
    setRectVisual(null);
    polyVertsRef.current = [];
    setPolyVisual([]);
    setPolyPreviewPt(null);
  }, [selectionMode]);

  // Build element_id -> zone_id lookup from zones
  const elemToZone = useMemo(() => {
    const map: Record<number, number> = {};
    for (const z of zones) {
      for (const eid of z.elements) {
        map[eid] = z.id;
      }
    }
    return map;
  }, [zones]);

  // Auto-fit to data on first load
  useEffect(() => {
    if (!geojson || geojson.features.length === 0) return;
    let minLng = Infinity, maxLng = -Infinity, minLat = Infinity, maxLat = -Infinity;
    for (const f of geojson.features) {
      const coords = (f.geometry as GeoJSON.Polygon).coordinates[0];
      for (const [lng, lat] of coords) {
        if (lng < minLng) minLng = lng;
        if (lng > maxLng) maxLng = lng;
        if (lat < minLat) minLat = lat;
        if (lat > maxLat) maxLat = lat;
      }
    }
    setViewState((prev) => ({
      ...prev,
      longitude: (minLng + maxLng) / 2,
      latitude: (minLat + maxLat) / 2,
      zoom: 7,
    }));
  }, [geojson]);

  // Screen-to-geo coordinate conversion (reads from ref, never stale)
  const screenToGeo = useCallback((screenX: number, screenY: number): [number, number] => {
    const el = containerRef.current;
    const width = el?.clientWidth ?? 800;
    const height = el?.clientHeight ?? 600;
    const vs = viewStateRef.current;
    const vp = new WebMercatorViewport({ ...vs, width, height });
    const [lng, lat] = vp.unproject([screenX, screenY]);
    return [lng, lat];
  }, []);

  // Forward wheel events from the interaction overlay to the deck.gl canvas
  const handleOverlayWheel = useCallback((e: React.WheelEvent) => {
    const canvas = containerRef.current?.querySelector('canvas');
    if (canvas) {
      canvas.dispatchEvent(new WheelEvent('wheel', {
        deltaX: e.deltaX,
        deltaY: e.deltaY,
        deltaMode: e.deltaMode,
        clientX: e.clientX,
        clientY: e.clientY,
        screenX: e.screenX,
        screenY: e.screenY,
        ctrlKey: e.ctrlKey,
        shiftKey: e.shiftKey,
        altKey: e.altKey,
        metaKey: e.metaKey,
        bubbles: true,
      }));
    }
  }, []);

  // Point-mode click handler for deck.gl
  const handleClick = useCallback((info: PickingInfo) => {
    if (selectionMode !== 'point') return;
    if (info.object) {
      const props = (info.object as GeoJSON.Feature).properties as Record<string, unknown>;
      const elemId = props?.element_id as number;
      if (elemId) {
        onElementClick(elemId, ctrlKeyRef.current);
      }
    }
  }, [onElementClick, selectionMode]);

  // Helper: get mouse position relative to interaction overlay
  const getRelPos = (e: React.MouseEvent | React.PointerEvent): [number, number] => {
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
    return [e.clientX - rect.left, e.clientY - rect.top];
  };

  // ---- Rectangle drawing handlers ----
  const handleRectPointerDown = useCallback((e: React.PointerEvent) => {
    if (selectionMode !== 'rectangle') return;
    const pt = getRelPos(e);
    rectStartRef.current = pt;
    rectEndRef.current = pt;
    isDrawingRect.current = true;
    setRectVisual({ start: pt, end: pt });
  }, [selectionMode]);

  const handleRectPointerMove = useCallback((e: React.PointerEvent) => {
    if (!isDrawingRect.current) return;
    const pt = getRelPos(e);
    rectEndRef.current = pt;
    setRectVisual({ start: rectStartRef.current!, end: pt });
  }, []);

  const handleRectPointerUp = useCallback(() => {
    if (!isDrawingRect.current) return;
    isDrawingRect.current = false;

    const start = rectStartRef.current;
    const end = rectEndRef.current;
    if (!start || !end) { setRectVisual(null); return; }

    // Convert screen corners to geo
    const [lng1, lat1] = screenToGeo(start[0], start[1]);
    const [lng2, lat2] = screenToGeo(end[0], end[1]);

    const minLng = Math.min(lng1, lng2);
    const maxLng = Math.max(lng1, lng2);
    const minLat = Math.min(lat1, lat2);
    const maxLat = Math.max(lat1, lat2);

    const ids = elementsInRect(elementCentroidsRef.current, minLng, minLat, maxLng, maxLat);
    if (ids.length > 0) {
      onShapeSelectRef.current(ids);
    }

    rectStartRef.current = null;
    rectEndRef.current = null;
    setRectVisual(null);
  }, [screenToGeo]);

  // ---- Polygon drawing handlers ----
  const handlePolySvgClick = useCallback((e: React.MouseEvent) => {
    if (selectionMode !== 'polygon') return;
    const pt = getRelPos(e);
    polyVertsRef.current = [...polyVertsRef.current, pt];
    setPolyVisual([...polyVertsRef.current]);
  }, [selectionMode]);

  const handlePolySvgMouseMove = useCallback((e: React.MouseEvent) => {
    if (selectionMode !== 'polygon' || polyVertsRef.current.length === 0) return;
    setPolyPreviewPt(getRelPos(e));
  }, [selectionMode]);

  const handlePolyDoubleClick = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (selectionMode !== 'polygon') return;

    // Double-click fires two clicks first, so the last vertex is a duplicate.
    // We need at least 3 real vertices (the 4th is the duplicate from 2nd click).
    const allVerts = polyVertsRef.current;
    if (allVerts.length < 4) { // 3 real + 1 duplicate
      polyVertsRef.current = [];
      setPolyVisual([]);
      setPolyPreviewPt(null);
      return;
    }

    // Remove the duplicate last vertex from the second click
    const verts = allVerts.slice(0, -1);

    // Convert screen vertices to geo
    const geoVerts = verts.map(([sx, sy]) => screenToGeo(sx, sy));
    const ids = elementsInPolygon(elementCentroidsRef.current, geoVerts);
    if (ids.length > 0) {
      onShapeSelectRef.current(ids);
    }

    polyVertsRef.current = [];
    setPolyVisual([]);
    setPolyPreviewPt(null);
  }, [selectionMode, screenToGeo]);

  const layers = useMemo(() => {
    if (!geojson) return [];

    return [
      new GeoJsonLayer({
        id: 'zone-elements',
        data: geojson,
        filled: true,
        stroked: true,
        lineWidthMinPixels: 0.5,
        getLineColor: [100, 100, 100, 120],
        getFillColor: (f: GeoJSON.Feature) => {
          const props = f.properties as Record<string, unknown>;
          const elemId = props?.element_id as number;
          const zoneId = elemToZone[elemId];
          if (zoneId) {
            return hexToRgba(getZoneColor(zoneId), 180);
          }
          return hexToRgba(getUnassignedColor(), 100);
        },
        pickable: selectionMode === 'point',
        onClick: handleClick,
        onHover: (info: PickingInfo) => {
          if (info.object) {
            const props = (info.object as GeoJSON.Feature).properties as Record<string, unknown>;
            const elemId = (props?.element_id as number) ?? 0;
            const zoneId = elemToZone[elemId];
            const zoneName = zones.find((z) => z.id === zoneId)?.name;
            setTooltip({
              x: info.x ?? 0,
              y: info.y ?? 0,
              text: `Element ${elemId}${zoneName ? ` | ${zoneName}` : ' | Unassigned'}`,
            });
          } else {
            setTooltip(null);
          }
        },
        updateTriggers: {
          getFillColor: [elemToZone, zones],
        },
      }),
    ];
  }, [geojson, elemToZone, zones, handleClick, selectionMode]);

  const mapStyle = useMemo(() => {
    if (selectedBasemap === 'dark') return 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json';
    if (selectedBasemap === 'voyager') return 'https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json';
    return BASEMAP;
  }, [selectedBasemap]);

  // Help text varies by mode
  const helpText = selectionMode === 'point'
    ? `Click to assign, Ctrl+click to remove. Paint zone: ${paintZoneId}`
    : selectionMode === 'rectangle'
      ? `Drag a rectangle to select elements. Paint zone: ${paintZoneId}`
      : `Click to add vertices, double-click to close. Paint zone: ${paintZoneId}`;

  const isDrawingMode = selectionMode !== 'point';
  const noZones = zones.length === 0;

  return (
    <Box ref={containerRef} sx={{ position: 'relative', width: '100%', height: '100%' }}>
      <DeckGL
        viewState={viewState}
        onViewStateChange={({ viewState: vs }) => setViewState(vs as ViewState)}
        layers={layers}
        controller={isDrawingMode
          ? { scrollZoom: true, dragPan: false, doubleClickZoom: false, dragRotate: false }
          : true
        }
        getCursor={() => (selectionMode === 'point' ? 'pointer' : 'crosshair')}
      >
        <Map mapStyle={mapStyle} />
      </DeckGL>

      {/* Interaction overlay: captures drawing events, forwards wheel to deck.gl */}
      {isDrawingMode && (
        <div
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            zIndex: 4,
            cursor: 'crosshair',
          }}
          onWheel={handleOverlayWheel}
          onPointerDown={selectionMode === 'rectangle' ? handleRectPointerDown : undefined}
          onPointerMove={selectionMode === 'rectangle' ? handleRectPointerMove : undefined}
          onPointerUp={selectionMode === 'rectangle' ? handleRectPointerUp : undefined}
          onClick={selectionMode === 'polygon' ? handlePolySvgClick : undefined}
          onMouseMove={selectionMode === 'polygon' ? handlePolySvgMouseMove : undefined}
          onDoubleClick={selectionMode === 'polygon' ? handlePolyDoubleClick : undefined}
        />
      )}

      {/* SVG visual overlay: purely for rendering shapes, no event handling */}
      {isDrawingMode && (
        <svg
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            zIndex: 5,
            pointerEvents: 'none',
          }}
        >
          {/* Rectangle preview */}
          {selectionMode === 'rectangle' && rectVisual && (
            <rect
              x={Math.min(rectVisual.start[0], rectVisual.end[0])}
              y={Math.min(rectVisual.start[1], rectVisual.end[1])}
              width={Math.abs(rectVisual.end[0] - rectVisual.start[0])}
              height={Math.abs(rectVisual.end[1] - rectVisual.start[1])}
              fill="rgba(25, 118, 210, 0.15)"
              stroke="#1976d2"
              strokeWidth={2}
              strokeDasharray="6 3"
            />
          )}

          {/* Polygon preview */}
          {selectionMode === 'polygon' && polyVisual.length > 0 && (
            <>
              <polyline
                points={polyVisual.map(([x, y]) => `${x},${y}`).join(' ')}
                fill="none"
                stroke="#1976d2"
                strokeWidth={2}
              />
              {polyPreviewPt && (
                <line
                  x1={polyVisual[polyVisual.length - 1][0]}
                  y1={polyVisual[polyVisual.length - 1][1]}
                  x2={polyPreviewPt[0]}
                  y2={polyPreviewPt[1]}
                  stroke="#1976d2"
                  strokeWidth={1}
                  strokeDasharray="4 2"
                />
              )}
              {polyPreviewPt && polyVisual.length >= 2 && (
                <line
                  x1={polyPreviewPt[0]}
                  y1={polyPreviewPt[1]}
                  x2={polyVisual[0][0]}
                  y2={polyVisual[0][1]}
                  stroke="#1976d2"
                  strokeWidth={1}
                  strokeDasharray="4 2"
                  opacity={0.5}
                />
              )}
              {polyVisual.length >= 2 && polyPreviewPt && (
                <polygon
                  points={[...polyVisual, polyPreviewPt].map(([x, y]) => `${x},${y}`).join(' ')}
                  fill="rgba(25, 118, 210, 0.1)"
                  stroke="none"
                />
              )}
              {polyVisual.map(([x, y], i) => (
                <circle key={i} cx={x} cy={y} r={4} fill="#1976d2" />
              ))}
            </>
          )}
        </svg>
      )}

      {/* Selection mode toolbar */}
      <ZoneSelectionToolbar
        mode={selectionMode}
        onModeChange={onSelectionModeChange}
        onUploadClick={onUploadClick}
        disabled={noZones}
      />

      {/* Active paint zone indicator */}
      <Box sx={{
        position: 'absolute', top: 8, left: 8, bgcolor: 'rgba(255,255,255,0.9)',
        borderRadius: 1, px: 1.5, py: 0.5, pointerEvents: 'none',
      }}>
        <Typography variant="caption">
          {noZones ? 'Add a zone first, then select elements.' : helpText}
        </Typography>
      </Box>

      {/* Tooltip */}
      {tooltip && (
        <Box sx={{
          position: 'absolute',
          left: tooltip.x + 10,
          top: tooltip.y + 10,
          bgcolor: 'rgba(0,0,0,0.75)',
          color: 'white',
          borderRadius: 1,
          px: 1,
          py: 0.5,
          pointerEvents: 'none',
          fontSize: 12,
          whiteSpace: 'nowrap',
        }}>
          {tooltip.text}
        </Box>
      )}

      {/* Legend */}
      {zones.length > 0 && (
        <Box sx={{
          position: 'absolute', bottom: 8, right: 8, bgcolor: 'rgba(255,255,255,0.9)',
          borderRadius: 1, p: 1, maxHeight: 200, overflowY: 'auto',
        }}>
          <Typography variant="caption" fontWeight={600} sx={{ mb: 0.5, display: 'block' }}>Zones</Typography>
          {zones.map((z) => (
            <Box key={z.id} sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mb: 0.25 }}>
              <Box sx={{ width: 12, height: 12, bgcolor: getZoneColor(z.id), borderRadius: '2px' }} />
              <Typography variant="caption">{z.name} ({z.elements.length})</Typography>
            </Box>
          ))}
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            <Box sx={{ width: 12, height: 12, bgcolor: getUnassignedColor(), borderRadius: '2px' }} />
            <Typography variant="caption">Unassigned</Typography>
          </Box>
        </Box>
      )}
    </Box>
  );
}
