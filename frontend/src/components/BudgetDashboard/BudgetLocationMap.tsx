/**
 * Mini-map showing the spatial location of the selected budget entity.
 * Displays a small deck.gl + MapLibre map with context geometry (dimmed)
 * and the selected feature highlighted.
 */

import { useState, useEffect, useRef, useMemo } from 'react';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import { Map } from 'react-map-gl/maplibre';
import DeckGL from '@deck.gl/react';
import { GeoJsonLayer, ScatterplotLayer } from '@deck.gl/layers';
import type { MapRef } from 'react-map-gl/maplibre';
import 'maplibre-gl/dist/maplibre-gl.css';

import {
  fetchBudgetLocationGeometry,
  fetchSubregions,
  fetchStreamGeoJSON,
  fetchLakesGeoJSON,
  fetchSmallWatersheds,
  fetchDiversions,
} from '../../api/client';
import type { BudgetLocationGeometry } from '../../api/client';
import { MAP_STYLE } from '../ResultsMap/mapStyles';

interface BudgetLocationMapProps {
  budgetType: string;
  locationName: string;
}

// Dim context colors
const CONTEXT_FILL: [number, number, number, number] = [180, 180, 180, 60];
const CONTEXT_LINE: [number, number, number, number] = [140, 140, 140, 100];
const HIGHLIGHT_FILL: [number, number, number, number] = [30, 100, 220, 140];
const HIGHLIGHT_LINE: [number, number, number, number] = [20, 60, 180, 220];
const POINT_COLOR: [number, number, number, number] = [220, 40, 40, 220];

// Default initial view (California)
const DEFAULT_VIEW = {
  longitude: -120.5,
  latitude: 37.5,
  zoom: 5,
  pitch: 0,
  bearing: 0,
};

export function BudgetLocationMap({ budgetType, locationName }: BudgetLocationMapProps) {
  const mapRef = useRef<MapRef>(null);

  // Spatial context data (cached per budget type)
  const [contextData, setContextData] = useState<GeoJSON.FeatureCollection | null>(null);
  const [pointData, setPointData] = useState<Array<{ lng: number; lat: number }> | null>(null);
  const [locationGeo, setLocationGeo] = useState<BudgetLocationGeometry | null>(null);
  const [viewState, setViewState] = useState(DEFAULT_VIEW);
  const prevTypeRef = useRef<string>('');

  // Fetch context geometry when budget type changes
  useEffect(() => {
    if (!budgetType || budgetType === prevTypeRef.current) return;
    prevTypeRef.current = budgetType;

    setContextData(null);
    setPointData(null);

    const category = detectCategory(budgetType);

    if (category === 'subregion') {
      fetchSubregions()
        .then(setContextData)
        .catch(() => {});
    } else if (category === 'reach' || category === 'stream_node_context') {
      fetchStreamGeoJSON()
        .then(setContextData)
        .catch(() => {});
    } else if (category === 'lake') {
      fetchLakesGeoJSON()
        .then(setContextData)
        .catch(() => {});
    } else if (category === 'small_watershed') {
      fetchSmallWatersheds()
        .then((data) => {
          const pts = data.watersheds.map((w) => ({
            lng: w.marker_position[0],
            lat: w.marker_position[1],
          }));
          setPointData(pts);
        })
        .catch(() => {});
    } else if (category === 'diversion') {
      fetchDiversions()
        .then((data) => {
          const pts = data.diversions
            .filter((d) => d.source !== null)
            .map((d) => ({ lng: d.source!.lng, lat: d.source!.lat }));
          setPointData(pts);
        })
        .catch(() => {});
    }
  }, [budgetType]);

  // Fetch location geometry when location changes
  useEffect(() => {
    if (!budgetType || !locationName) return;
    fetchBudgetLocationGeometry(budgetType, locationName)
      .then(setLocationGeo)
      .catch(() => {});
  }, [budgetType, locationName]);

  // Center map on selected location whenever relevant data changes
  useEffect(() => {
    if (!locationGeo) return;
    fitToFeature(locationGeo);
  }, [locationGeo, contextData, pointData]);

  function detectCategory(bt: string): string {
    const l = bt.toLowerCase();
    if (l.startsWith('gw') || l.includes('groundwater') || l === 'lwu' || l.includes('land')
        || l.includes('root') || l === 'rootzone' || l.includes('unsat') || l === 'unsaturated') {
      return 'subregion';
    }
    if (l === 'stream_node') return 'stream_node_context';
    if (l.includes('stream')) return 'reach';
    if (l.includes('lake')) return 'lake';
    if (l.includes('diver')) return 'diversion';
    if (l.includes('small_watershed') || l.includes('small watershed')) return 'small_watershed';
    return 'unknown';
  }

  function fitToFeature(_geo: BudgetLocationGeometry) {
    const idx = _geo.location_index;

    // If only a point geometry (stream_node) and no context yet, center on it
    if (_geo.geometry && _geo.geometry.type === 'Point' && !contextData) {
      const [lng, lat] = _geo.geometry.coordinates;
      setViewState({ ...DEFAULT_VIEW, longitude: lng, latitude: lat, zoom: 8 });
      return;
    }

    // For GeoJSON context: zoom to the SELECTED feature
    if (contextData && contextData.features.length > 0 && idx < contextData.features.length) {
      const bounds = getFeatureBounds(contextData.features[idx]);
      if (bounds) {
        const [minLng, minLat, maxLng, maxLat] = bounds;
        const cLng = (minLng + maxLng) / 2;
        const cLat = (minLat + maxLat) / 2;
        const span = Math.max(maxLng - minLng, maxLat - minLat);
        const zoom = span > 0
          ? Math.min(13, Math.max(6, -Math.log2(span / 360) + 1)) - 0.3
          : 10;
        setViewState({ ...DEFAULT_VIEW, longitude: cLng, latitude: cLat, zoom });
        return;
      }
    }

    // For point data (small_watershed, diversion): center on SELECTED point
    if (pointData && pointData.length > 0 && idx < pointData.length) {
      const pt = pointData[idx];
      setViewState({ ...DEFAULT_VIEW, longitude: pt.lng, latitude: pt.lat, zoom: 10 });
      return;
    }

    // Last resort: point geometry (stream_node with context loaded but empty)
    if (_geo.geometry && _geo.geometry.type === 'Point') {
      const [lng, lat] = _geo.geometry.coordinates;
      setViewState({ ...DEFAULT_VIEW, longitude: lng, latitude: lat, zoom: 8 });
    }
  }

  function getFeatureBounds(feature: GeoJSON.Feature): [number, number, number, number] | null {
    const coords: number[][] = [];
    extractCoords(feature.geometry, coords);
    if (coords.length === 0) return null;
    let minLng = Infinity, minLat = Infinity, maxLng = -Infinity, maxLat = -Infinity;
    for (const c of coords) {
      if (c[0] < minLng) minLng = c[0];
      if (c[1] < minLat) minLat = c[1];
      if (c[0] > maxLng) maxLng = c[0];
      if (c[1] > maxLat) maxLat = c[1];
    }
    return [minLng, minLat, maxLng, maxLat];
  }

  function extractCoords(geom: GeoJSON.Geometry, out: number[][]): void {
    if (geom.type === 'Point') {
      out.push(geom.coordinates as number[]);
    } else if (geom.type === 'MultiPoint' || geom.type === 'LineString') {
      for (const c of geom.coordinates) out.push(c as number[]);
    } else if (geom.type === 'MultiLineString' || geom.type === 'Polygon') {
      for (const ring of geom.coordinates) {
        for (const c of ring) out.push(c as number[]);
      }
    } else if (geom.type === 'MultiPolygon') {
      for (const poly of geom.coordinates) {
        for (const ring of poly) {
          for (const c of ring) out.push(c as number[]);
        }
      }
    }
  }

  // Build layers
  const layers = useMemo(() => {
    const result: (GeoJsonLayer | ScatterplotLayer)[] = [];
    if (!locationGeo) return result;

    const idx = locationGeo.location_index;
    const spatialType = locationGeo.spatial_type;

    // GeoJSON-based context (subregions, reaches, lakes)
    if (contextData) {
      // Context layer (all features dimmed)
      result.push(new GeoJsonLayer({
        id: 'budget-context',
        data: contextData,
        filled: spatialType !== 'reach',
        stroked: true,
        getFillColor: CONTEXT_FILL,
        getLineColor: CONTEXT_LINE,
        getLineWidth: spatialType === 'reach' ? 2 : 1,
        lineWidthUnits: 'pixels' as const,
        pickable: false,
      }));

      // Highlight layer (selected feature)
      if (idx < contextData.features.length) {
        const highlighted: GeoJSON.FeatureCollection = {
          type: 'FeatureCollection',
          features: [contextData.features[idx]],
        };
        result.push(new GeoJsonLayer({
          id: 'budget-highlight',
          data: highlighted,
          filled: spatialType !== 'reach',
          stroked: true,
          getFillColor: HIGHLIGHT_FILL,
          getLineColor: HIGHLIGHT_LINE,
          getLineWidth: spatialType === 'reach' ? 4 : 2,
          lineWidthUnits: 'pixels' as const,
          pickable: false,
        }));
      }
    }

    // Point-based context (small watersheds, diversions)
    if (pointData) {
      result.push(new ScatterplotLayer({
        id: 'budget-point-context',
        data: pointData,
        getPosition: (d: { lng: number; lat: number }) => [d.lng, d.lat],
        getRadius: 400,
        radiusUnits: 'meters' as const,
        getFillColor: CONTEXT_FILL,
        pickable: false,
      }));

      if (idx < pointData.length) {
        result.push(new ScatterplotLayer({
          id: 'budget-point-highlight',
          data: [pointData[idx]],
          getPosition: (d: { lng: number; lat: number }) => [d.lng, d.lat],
          getRadius: 800,
          radiusUnits: 'meters' as const,
          getFillColor: HIGHLIGHT_FILL,
          pickable: false,
        }));
      }
    }

    // Stream node point
    if (spatialType === 'point' && locationGeo.geometry) {
      const [lng, lat] = locationGeo.geometry.coordinates;
      result.push(new ScatterplotLayer({
        id: 'budget-stream-node-point',
        data: [{ lng, lat }],
        getPosition: (d: { lng: number; lat: number }) => [d.lng, d.lat],
        getRadius: 600,
        radiusUnits: 'meters' as const,
        getFillColor: POINT_COLOR,
        pickable: false,
      }));
    }

    return result;
  }, [contextData, pointData, locationGeo]);

  return (
    <Box sx={{ width: '100%', height: '100%', position: 'relative' }}>
      <DeckGL
        viewState={viewState}
        onViewStateChange={({ viewState: vs }) => setViewState(vs as typeof DEFAULT_VIEW)}
        layers={layers}
        controller={true}
        style={{ width: '100%', height: '100%' }}
      >
        <Map
          ref={mapRef}
          mapStyle={MAP_STYLE}
          attributionControl={false}
        />
      </DeckGL>
      <Box sx={{
        position: 'absolute', bottom: 2, left: 4,
        bgcolor: 'rgba(255,255,255,0.85)', px: 0.5, borderRadius: 0.5,
      }}>
        <Typography variant="caption" sx={{ fontSize: '0.65rem', color: 'text.secondary' }}>
          {locationName}
        </Typography>
      </Box>
    </Box>
  );
}
