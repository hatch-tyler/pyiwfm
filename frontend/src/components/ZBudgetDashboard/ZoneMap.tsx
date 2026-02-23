/**
 * ZoneMap: deck.gl element map for interactive zone assignment.
 * Elements are colored by their zone assignment; click to assign.
 */

import { useState, useEffect, useMemo, useCallback } from 'react';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import DeckGL from '@deck.gl/react';
import { GeoJsonLayer } from '@deck.gl/layers';
import type { PickingInfo } from '@deck.gl/core';
import { Map } from 'react-map-gl/maplibre';
import 'maplibre-gl/dist/maplibre-gl.css';
import { useViewerStore } from '../../stores/viewerStore';
import type { ZoneInfo } from '../../api/client';
import { getZoneColor, getUnassignedColor, hexToRgba } from './zoneColors';

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
  onElementClick: (elementId: number) => void;
}

export function ZoneMap({ geojson, zones, paintZoneId, onElementClick }: ZoneMapProps) {
  const { selectedBasemap } = useViewerStore();

  const [viewState, setViewState] = useState<ViewState>({
    longitude: -120.5,
    latitude: 37.5,
    zoom: 6,
    pitch: 0,
    bearing: 0,
  });

  const [tooltip, setTooltip] = useState<{ x: number; y: number; text: string } | null>(null);

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

  const handleClick = useCallback((info: PickingInfo) => {
    if (info.object) {
      const props = (info.object as GeoJSON.Feature).properties as Record<string, unknown>;
      const elemId = props?.element_id as number;
      if (elemId) {
        onElementClick(elemId);
      }
    }
  }, [onElementClick]);

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
        pickable: true,
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
  }, [geojson, elemToZone, zones, handleClick]);

  const mapStyle = useMemo(() => {
    if (selectedBasemap === 'dark') return 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json';
    if (selectedBasemap === 'voyager') return 'https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json';
    return BASEMAP;
  }, [selectedBasemap]);

  return (
    <Box sx={{ position: 'relative', width: '100%', height: '100%' }}>
      <DeckGL
        viewState={viewState}
        onViewStateChange={({ viewState: vs }) => setViewState(vs as ViewState)}
        layers={layers}
        controller={true}
        getCursor={() => 'pointer'}
      >
        <Map mapStyle={mapStyle} />
      </DeckGL>

      {/* Active paint zone indicator */}
      <Box sx={{
        position: 'absolute', top: 8, left: 8, bgcolor: 'rgba(255,255,255,0.9)',
        borderRadius: 1, px: 1.5, py: 0.5, pointerEvents: 'none',
      }}>
        <Typography variant="caption">
          Click elements to assign to zone. Active paint zone: {paintZoneId}
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
