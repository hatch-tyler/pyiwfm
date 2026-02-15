/**
 * Results Map tab: 2D map with head contours, clickable locations, time slider.
 * Uses deck.gl over a MapLibre GL JS basemap.
 *
 * Layers (bottom to top):
 *   1. Mesh choropleth (head values, property, or head diff)
 *   2. Lake polygons
 *   3. Subregion boundary outlines
 *   4. Stream network lines
 *   5. Boundary condition markers
 *   6. Well markers
 *   7. GW node markers
 *   8. Hydrograph location markers (GW, stream, subsidence)
 *   9. Small watershed nodes + arcs
 *  10. Diversion arcs
 *  11. Cross-section line (when tool is active)
 */

import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import CircularProgress from '@mui/material/CircularProgress';
import { Map } from 'react-map-gl/maplibre';
import DeckGL from '@deck.gl/react';
import { GeoJsonLayer, ScatterplotLayer, TextLayer, LineLayer } from '@deck.gl/layers';
import { PathStyleExtension } from '@deck.gl/extensions';
import type { PickingInfo, Layer } from '@deck.gl/core';
import 'maplibre-gl/dist/maplibre-gl.css';

import { useViewerStore } from '../../stores/viewerStore';
import {
  fetchMeshGeoJSON, fetchHeadsByElement, fetchHeadTimes, fetchHeadRange,
  fetchHydrographLocations, fetchHydrograph,
  fetchSubregions, fetchStreamGeoJSON, fetchWells,
  fetchPropertyMap, fetchElementDetail,
  fetchBoundaryConditions, fetchHeadDiff,
  fetchLakesGeoJSON, fetchLakeRating,
  fetchCrossSectionJSON,
  fetchSmallWatersheds, fetchDiversions, fetchDiversionDetail,
  fetchReachProfile, fetchHydrographsMulti,
  fetchMeshNodes,
} from '../../api/client';
import type {
  HydrographLocation, HydrographData, WellInfo,
  BCNodeInfo, SmallWatershedData, SmallWatershed, SmallWatershedGWNode,
  DiversionArc, MeshNodeInfo,
} from '../../api/client';
import {
  interpolateColor, interpolateDivergingColor,
  LAKE_FILL_COLOR, LAKE_OUTLINE_COLOR, BC_COLORS,
  DIVERSION_SOURCE_COLOR, DIVERSION_SELECTED_COLOR,
  DIVERSION_DELIVERY_FILL, DIVERSION_DELIVERY_OUTLINE,
  WATERSHED_MARKER_COLOR, WATERSHED_SELECTED_COLOR,
  WATERSHED_GW_PERC_COLOR, WATERSHED_GW_BASEFLOW_COLOR,
  WATERSHED_DEST_COLOR,
  getBasemapStyle,
} from './mapStyles';
import { ResultsControls } from './ResultsControls';
import { HydrographChart } from './HydrographChart';
import { ColorLegend } from './ColorLegend';
import { ElementDetailPanel } from './ElementDetailPanel';
import { CrossSectionChart } from './CrossSectionChart';
import { LakeRatingChart } from './LakeRatingChart';
import { ReachProfileChart } from './ReachProfileChart';
import { ComparisonChart } from './ComparisonChart';
import { DiversionPanel } from './DiversionPanel';
import { WatershedDetailPanel } from './WatershedDetailPanel';
import { useObservationOverlay } from '../Observations/ObservationOverlay';

interface ViewState {
  longitude: number;
  latitude: number;
  zoom: number;
  pitch: number;
  bearing: number;
}

export function ResultsMapView() {
  const {
    modelInfo, resultsInfo, properties,
    headTimestep, headLayer,
    selectedLocation, selectedHydrograph,
    showGWLocations, showStreamLocations, showSubsidenceLocations,
    showSubregions, showStreamsOnMap, showWells, showNodes,
    showLakes, showBoundaryConditions,
    showSmallWatersheds, showDiversions,
    mapColorProperty, elementDetail,
    headDiffMode, headDiffTimestepA, headDiffTimestepB,
    crossSectionMode, crossSectionPoints, crossSectionData,
    selectedLakeRating,
    compareMode, comparedLocationIds,
    selectedReachProfile,
    selectedWatershedId, selectedWatershedDetail,
    selectedDiversionId, diversionDetail, diversionListOpen,
    isAnimating,
    selectedBasemap,
    headGlobalMin, headGlobalMax,
    setHeadGlobalRange,
    setHeadTimes, setSelectedLocation, setSelectedHydrograph,
    setSelectedElement, setElementDetail,
    setSelectedWatershedId, setSelectedWatershedDetail,
    setCrossSectionPoints, setCrossSectionData,
    setSelectedLakeRating,
    setSelectedReachProfile,
    setSelectedDiversionId, setDiversionDetail, setDiversionListOpen,
    addComparedLocationId,
    setMapColorProperty,
  } = useViewerStore();

  // Observation overlay data for hydrograph chart
  const observationData = useObservationOverlay();

  // Local state
  const [geojson, setGeojson] = useState<GeoJSON.FeatureCollection | null>(null);
  const [headValues, setHeadValues] = useState<number[] | null>(null);
  const [headMin, setHeadMin] = useState(0);
  const [headMax, setHeadMax] = useState(1);
  const [locations, setLocations] = useState<{
    gw: HydrographLocation[];
    stream: HydrographLocation[];
    subsidence: HydrographLocation[];
  }>({ gw: [], stream: [], subsidence: [] });
  const [loading, setLoading] = useState(true);
  const [viewState, setViewState] = useState<ViewState>({
    longitude: -121.5, latitude: 37.5, zoom: 7, pitch: 0, bearing: 0,
  });
  const [tooltip, setTooltip] = useState<{ x: number; y: number; text: string } | null>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Overlay data
  const [subregionGeoJSON, setSubregionGeoJSON] = useState<GeoJSON.FeatureCollection | null>(null);
  const [streamGeoJSON, setStreamGeoJSON] = useState<GeoJSON.FeatureCollection | null>(null);
  const [wellData, setWellData] = useState<WellInfo[]>([]);
  const [nodeData, setNodeData] = useState<MeshNodeInfo[]>([]);
  const [lakeGeoJSON, setLakeGeoJSON] = useState<GeoJSON.FeatureCollection | null>(null);
  const [bcData, setBcData] = useState<BCNodeInfo[]>([]);
  const [watershedData, setWatershedData] = useState<SmallWatershedData | null>(null);
  const [diversionData, setDiversionData] = useState<DiversionArc[]>([]);
  const [comparisonSeries, setComparisonSeries] = useState<HydrographData[]>([]);

  // Property map data (for non-head coloring)
  const [propertyValues, setPropertyValues] = useState<number[] | null>(null);
  const [propertyMin, setPropertyMin] = useState(0);
  const [propertyMax, setPropertyMax] = useState(1);
  const [propertyMeta, setPropertyMeta] = useState<{
    name: string; units: string; log_scale: boolean;
  } | null>(null);

  // Head difference data
  const [diffValues, setDiffValues] = useState<(number | null)[] | null>(null);
  const [diffMin, setDiffMin] = useState(0);
  const [diffMax, setDiffMax] = useState(0);

  // Head data cache for animation prefetching (keyed by "timestep:layer")
  type HeadCacheEntry = { values: number[]; min: number; max: number };
  const headCacheRef = useRef<Record<string, HeadCacheEntry>>({});
  const HEAD_CACHE_LIMIT = 50;

  const nLayers = modelInfo?.n_layers ?? 1;
  const hasHeads = resultsInfo ? resultsInfo.n_head_timesteps > 0 : false;
  const colorByHead = mapColorProperty === 'head' && !headDiffMode;
  const colorByDiff = headDiffMode;

  // Basemap style (string URL for vector, object for raster)
  const mapStyle = useMemo(
    () => getBasemapStyle(selectedBasemap),
    [selectedBasemap],
  );

  // Head fallback: if no head data available and property is 'head', switch to first property
  useEffect(() => {
    if (!hasHeads && mapColorProperty === 'head' && properties.length > 0) {
      const fallback = properties.find(p => p.id !== 'layer');
      if (fallback) {
        setMapColorProperty(fallback.id);
      }
    }
  }, [hasHeads, mapColorProperty, properties, setMapColorProperty]);

  // Load mesh GeoJSON + head times + hydrograph locations on mount
  useEffect(() => {
    const load = async () => {
      setLoading(true);
      try {
        const geo = await fetchMeshGeoJSON(headLayer);
        setGeojson(geo);

        // Center the map on the model extent
        if (geo.features.length > 0) {
          let lngSum = 0, latSum = 0, count = 0;
          for (const f of geo.features) {
            const coords = (f.geometry as GeoJSON.Polygon).coordinates[0];
            for (const c of coords) {
              lngSum += c[0];
              latSum += c[1];
              count++;
            }
          }
          if (count > 0) {
            setViewState(prev => ({
              ...prev,
              longitude: lngSum / count,
              latitude: latSum / count,
              zoom: 8,
            }));
          }
        }

        // Load head times
        if (hasHeads) {
          try {
            const ht = await fetchHeadTimes();
            setHeadTimes(ht.times);
          } catch { /* no head data */ }
        }

        // Load hydrograph locations
        try {
          const locs = await fetchHydrographLocations();
          setLocations(locs);
        } catch { /* no locations */ }
      } catch (err) {
        console.error('Failed to load mesh GeoJSON:', err);
      }
      setLoading(false);
    };
    load();
  }, [headLayer, hasHeads, setHeadTimes]);

  // Load overlay data on mount (with error logging)
  useEffect(() => {
    fetchSubregions().then(setSubregionGeoJSON).catch(e => console.warn('Subregions fetch:', e));
    fetchStreamGeoJSON().then(setStreamGeoJSON).catch(e => console.warn('Streams fetch:', e));
    fetchWells().then(resp => setWellData(resp.wells)).catch(e => console.warn('Wells fetch:', e));
    fetchMeshNodes().then(resp => setNodeData(resp.nodes)).catch(e => console.warn('Nodes fetch:', e));
    fetchLakesGeoJSON().then(setLakeGeoJSON).catch(e => console.warn('Lakes fetch:', e));
    fetchBoundaryConditions().then(resp => setBcData(resp.nodes)).catch(e => console.warn('BCs fetch:', e));
    fetchSmallWatersheds().then(setWatershedData).catch(e => console.warn('Watersheds fetch:', e));
    fetchDiversions().then(resp => setDiversionData(resp.diversions)).catch(e => console.warn('Diversions fetch:', e));
  }, []);

  // Load comparison hydrographs when comparedLocationIds change
  useEffect(() => {
    if (!compareMode || comparedLocationIds.length === 0) {
      setComparisonSeries([]);
      return;
    }
    fetchHydrographsMulti('gw', comparedLocationIds)
      .then(resp => setComparisonSeries(resp.series))
      .catch(() => setComparisonSeries([]));
  }, [compareMode, comparedLocationIds]);

  // Clear head cache and fetch global range when layer changes
  useEffect(() => {
    headCacheRef.current = {};
    setHeadGlobalRange(null, null);
    if (hasHeads) {
      fetchHeadRange(headLayer, 50)
        .then(range => {
          setHeadGlobalRange(range.min, range.max);
        })
        .catch(() => {
          setHeadGlobalRange(null, null);
        });
    }
  }, [headLayer, hasHeads, setHeadGlobalRange]);

  // Load per-element head values when timestep changes — only when coloring by head.
  // Uses fetchHeadsByElement (vertex-averaged) so values align 1:1 with GeoJSON features.
  // Cache avoids refetching, and prefetches next timestep when animating.
  useEffect(() => {
    if (!hasHeads || !colorByHead) return;

    const cacheKey = `${headTimestep}:${headLayer}`;
    const cached = headCacheRef.current[cacheKey];

    if (cached) {
      // Use cached data immediately
      setHeadValues(cached.values);
      setHeadMin(cached.min);
      setHeadMax(cached.max);
    } else {
      // Fetch with short debounce (50ms during animation, 200ms for scrubbing)
      if (debounceRef.current) clearTimeout(debounceRef.current);
      const delay = isAnimating ? 50 : 200;
      debounceRef.current = setTimeout(async () => {
        try {
          const data = await fetchHeadsByElement(headTimestep, headLayer);
          const vals = data.values as number[];

          // Use server-computed min/max (2nd–98th percentile)
          const lo = data.min;
          const hi = data.max;

          // Store in cache (evict oldest if at limit)
          const cache = headCacheRef.current;
          const keys = Object.keys(cache);
          if (keys.length >= HEAD_CACHE_LIMIT) {
            delete cache[keys[0]];
          }
          cache[cacheKey] = { values: vals, min: lo, max: hi };

          setHeadValues(vals);
          setHeadMin(lo);
          setHeadMax(hi);
        } catch {
          // Head data not available for this timestep
        }
      }, delay);
    }

    // Prefetch next timestep when animating
    if (isAnimating && headTimestep < (resultsInfo?.n_head_timesteps ?? 0) - 1) {
      const nextTs = headTimestep + 1;
      const nextKey = `${nextTs}:${headLayer}`;
      if (!headCacheRef.current[nextKey]) {
        fetchHeadsByElement(nextTs, headLayer).then(data => {
          const vals = data.values as number[];
          const lo = data.min;
          const hi = data.max;
          const cache = headCacheRef.current;
          const keys = Object.keys(cache);
          if (keys.length >= HEAD_CACHE_LIMIT) {
            delete cache[keys[0]];
          }
          cache[nextKey] = { values: vals, min: lo, max: hi };
        }).catch(() => {});
      }
    }

    return () => {
      if (debounceRef.current) {
        clearTimeout(debounceRef.current);
        debounceRef.current = null;
      }
    };
  }, [headTimestep, headLayer, hasHeads, colorByHead, isAnimating, resultsInfo]);

  // Load property map when coloring by a non-head property
  useEffect(() => {
    if (colorByHead || colorByDiff) {
      setPropertyValues(null);
      setPropertyMeta(null);
      return;
    }

    const loadProperty = async () => {
      try {
        const resp = await fetchPropertyMap(mapColorProperty, headLayer);
        const vals = resp.features.map(
          f => (f.properties as Record<string, unknown>)?.value as number ?? NaN
        );
        setPropertyValues(vals);
        setPropertyMin(resp.metadata.min);
        setPropertyMax(resp.metadata.max);
        setPropertyMeta({
          name: resp.metadata.name,
          units: resp.metadata.units,
          log_scale: resp.metadata.log_scale,
        });
      } catch (err) {
        console.error('Failed to load property map:', err);
        setPropertyValues(null);
      }
    };
    loadProperty();
  }, [mapColorProperty, headLayer, colorByHead, colorByDiff]);

  // Load head difference data
  useEffect(() => {
    if (!headDiffMode || !hasHeads) {
      setDiffValues(null);
      return;
    }
    const loadDiff = async () => {
      try {
        const resp = await fetchHeadDiff(headDiffTimestepA, headDiffTimestepB, headLayer);
        setDiffValues(resp.values);
        setDiffMin(resp.min);
        setDiffMax(resp.max);
      } catch (err) {
        console.error('Failed to load head diff:', err);
        setDiffValues(null);
      }
    };
    loadDiff();
  }, [headDiffMode, headDiffTimestepA, headDiffTimestepB, headLayer, hasHeads]);

  // Click handler for hydrograph locations
  const handleLocationClick = useCallback(async (loc: HydrographLocation, type: string) => {
    setSelectedLocation({ id: loc.id, type: type as 'gw' | 'stream' | 'subsidence' });
    try {
      const data: HydrographData = await fetchHydrograph(type, loc.id);
      setSelectedHydrograph(data);
    } catch (err) {
      console.error('Failed to load hydrograph:', err);
      setSelectedHydrograph(null);
    }
  }, [setSelectedLocation, setSelectedHydrograph]);

  // Click handler for mesh elements
  const handleElementClick = useCallback(async (elemId: number) => {
    setSelectedElement(elemId);
    try {
      const detail = await fetchElementDetail(elemId);
      setElementDetail(detail);
    } catch (err) {
      console.error('Failed to load element detail:', err);
      setElementDetail(null);
    }
  }, [setSelectedElement, setElementDetail]);

  // Click handler for lake features
  const handleLakeClick = useCallback(async (lakeId: number) => {
    try {
      const rating = await fetchLakeRating(lakeId);
      setSelectedLakeRating(rating);
    } catch {
      // Lake may not have a rating curve
      setSelectedLakeRating(null);
    }
  }, [setSelectedLakeRating]);

  // Click handler for stream reach profiles
  const handleReachClick = useCallback(async (reachId: number) => {
    try {
      const profile = await fetchReachProfile(reachId);
      setSelectedReachProfile(profile);
    } catch {
      setSelectedReachProfile(null);
    }
  }, [setSelectedReachProfile]);

  // Click handler for diversion dots
  const handleDiversionClick = useCallback(async (divId: number) => {
    setSelectedDiversionId(divId);
    setDiversionListOpen(true);
    try {
      const detail = await fetchDiversionDetail(divId);
      setDiversionDetail(detail);
    } catch {
      setDiversionDetail(null);
    }
  }, [setSelectedDiversionId, setDiversionListOpen, setDiversionDetail]);

  // Click handler for watershed markers
  const handleWatershedClick = useCallback((ws: SmallWatershed) => {
    setSelectedWatershedId(ws.id);
    setSelectedWatershedDetail(ws);
  }, [setSelectedWatershedId, setSelectedWatershedDetail]);

  // Map click handler for cross-section tool
  const handleMapClick = useCallback(async (info: PickingInfo) => {
    if (!crossSectionMode) return;
    if (!info.coordinate) return;

    const [lng, lat] = info.coordinate;
    const pts = [...crossSectionPoints, { lng, lat }];

    if (pts.length >= 2) {
      // We have two points — fetch cross-section
      setCrossSectionPoints(pts.slice(0, 2));
      try {
        const data = await fetchCrossSectionJSON(
          pts[0].lng, pts[0].lat,
          pts[1].lng, pts[1].lat,
        );
        setCrossSectionData(data);
      } catch (err) {
        console.error('Failed to load cross-section:', err);
        setCrossSectionData(null);
      }
    } else {
      setCrossSectionPoints(pts);
      setCrossSectionData(null);
    }
  }, [crossSectionMode, crossSectionPoints, setCrossSectionPoints, setCrossSectionData]);

  // Build deck.gl layers
  const layers = useMemo(() => {
    const result: Layer[] = [];

    // Determine which color values/range to use
    const usePropertyColor = !colorByHead && !colorByDiff && propertyValues !== null;
    const useDiffColor = colorByDiff && diffValues !== null;
    const activeValues = useDiffColor ? (diffValues as number[]) : usePropertyColor ? propertyValues : headValues;
    // Use global range when available (fixed scale across animation);
    // fall back to per-frame range.
    const activeMin = useDiffColor ? diffMin : usePropertyColor ? propertyMin : (headGlobalMin ?? headMin);
    const activeMax = useDiffColor ? diffMax : usePropertyColor ? propertyMax : (headGlobalMax ?? headMax);
    const isLogScale = usePropertyColor && propertyMeta?.log_scale;

    // Symmetric range for diverging scale
    const diffAbsMax = Math.max(Math.abs(diffMin), Math.abs(diffMax));

    // Build element_id → feature index map for value lookups.
    // GeoJSON features and per-element value arrays are both in sorted
    // element-ID order, so feature index == value array index.  We also
    // build a reverse map for tooltip hover lookups.
    const elemIdToIdx: Record<number, number> = {};
    if (geojson) {
      geojson.features.forEach((f, i) => {
        const eid = (f.properties as Record<string, unknown>)?.element_id as number;
        if (eid != null) elemIdToIdx[eid] = i;
      });
    }

    // 1. Mesh colored by head values, property, or head difference
    if (geojson) {
      result.push(new GeoJsonLayer({
        id: 'mesh-layer',
        data: geojson,
        filled: true,
        stroked: true,
        lineWidthMinPixels: 0.5,
        getLineColor: [100, 100, 100, 80],
        getFillColor: (_f: GeoJSON.Feature, { index }: { index: number }) => {
          if (!activeValues) return [200, 200, 200, 100] as [number, number, number, number];
          const idx = index;
          if (idx < 0 || idx >= activeValues.length) return [200, 200, 200, 100] as [number, number, number, number];
          const val = activeValues[idx];
          if (val === null || val === undefined || isNaN(val as number)) {
            return [200, 200, 200, 50] as [number, number, number, number];
          }
          if (!useDiffColor && (val as number) < -9000) {
            return [200, 200, 200, 50] as [number, number, number, number];
          }

          if (useDiffColor) {
            // Diverging: map from [-absMax, +absMax] to [0, 1]
            const t = diffAbsMax > 0 ? ((val as number) + diffAbsMax) / (2 * diffAbsMax) : 0.5;
            return interpolateDivergingColor(t);
          }

          let t: number;
          if (isLogScale && activeMin > 0 && activeMax > 0) {
            const logMin = Math.log10(activeMin);
            const logMax = Math.log10(activeMax);
            t = logMax > logMin ? (Math.log10(Math.max(val as number, activeMin)) - logMin) / (logMax - logMin) : 0.5;
          } else {
            t = activeMax > activeMin ? ((val as number) - activeMin) / (activeMax - activeMin) : 0.5;
          }
          return interpolateColor(t);
        },
        pickable: true,
        onHover: (info: PickingInfo) => {
          if (info.object && activeValues) {
            const props = (info.object as GeoJSON.Feature).properties as Record<string, unknown>;
            const elemId = (props?.element_id as number) ?? 0;
            const idx = elemIdToIdx[elemId] ?? -1;
            const val = idx >= 0 && idx < activeValues.length ? activeValues[idx] : null;
            const label = useDiffColor ? 'Head Diff' : colorByHead ? 'Head' : (propertyMeta?.name ?? mapColorProperty);
            const units = useDiffColor ? 'ft' : colorByHead ? 'ft' : (propertyMeta?.units ?? '');
            setTooltip({
              x: info.x ?? 0,
              y: info.y ?? 0,
              text: `Element ${elemId}${val !== null && !isNaN(val as number) ? ` | ${label}: ${(val as number).toFixed(1)} ${units}` : ''}`,
            });
          } else {
            setTooltip(null);
          }
        },
        onClick: (info: PickingInfo) => {
          if (crossSectionMode) return; // Let map click handler handle it
          if (info.object) {
            const props = (info.object as GeoJSON.Feature).properties as Record<string, unknown>;
            const elemId = props?.element_id as number;
            if (elemId) handleElementClick(elemId);
          }
        },
        updateTriggers: {
          getFillColor: [activeValues, activeMin, activeMax, isLogScale, useDiffColor, diffAbsMax],
        },
      }));
    }

    // 2. Lake polygons
    if (showLakes && lakeGeoJSON && lakeGeoJSON.features.length > 0) {
      result.push(new GeoJsonLayer({
        id: 'lake-polygons',
        data: lakeGeoJSON,
        filled: true,
        stroked: true,
        lineWidthMinPixels: 2,
        getFillColor: LAKE_FILL_COLOR,
        getLineColor: LAKE_OUTLINE_COLOR,
        pickable: true,
        onHover: (info: PickingInfo) => {
          if (info.object) {
            const props = (info.object as GeoJSON.Feature).properties as Record<string, unknown>;
            setTooltip({
              x: info.x ?? 0,
              y: info.y ?? 0,
              text: `${props?.name ?? 'Lake'} (${props?.n_elements ?? '?'} elements)`,
            });
          }
        },
        onClick: (info: PickingInfo) => {
          if (info.object) {
            const props = (info.object as GeoJSON.Feature).properties as Record<string, unknown>;
            const lakeId = props?.lake_id as number;
            if (lakeId && props?.has_rating) handleLakeClick(lakeId);
          }
        },
      }));

      // Lake name labels
      const lakeLabelData = lakeGeoJSON.features
        .map(f => {
          const props = f.properties as Record<string, unknown>;
          const centroid = props?.centroid as [number, number];
          if (!centroid) return null;
          return { position: centroid, text: (props?.name as string) ?? '' };
        })
        .filter(Boolean) as Array<{ position: [number, number]; text: string }>;

      if (lakeLabelData.length > 0) {
        result.push(new TextLayer({
          id: 'lake-labels',
          data: lakeLabelData,
          getPosition: (d: { position: [number, number] }) => d.position,
          getText: (d: { text: string }) => d.text,
          getSize: 12,
          getColor: [20, 60, 140, 220],
          fontWeight: 'bold' as unknown as number,
          getTextAnchor: 'middle' as const,
          getAlignmentBaseline: 'center' as const,
          pickable: false,
        }));
      }
    }

    // 3. Subregion boundaries
    if (showSubregions && subregionGeoJSON && subregionGeoJSON.features.length > 0) {
      result.push(new GeoJsonLayer({
        id: 'subregion-boundaries',
        data: subregionGeoJSON,
        filled: false,
        stroked: true,
        lineWidthMinPixels: 3,
        getLineColor: [80, 80, 80, 200],
        getDashArray: [8, 4],
        dashJustified: true,
        extensions: [new PathStyleExtension({ dash: true })],
        pickable: false,
      }));

      // Subregion name labels
      const labelData = subregionGeoJSON.features
        .map(f => {
          const props = f.properties as Record<string, unknown>;
          const centroid = props?.centroid as [number, number];
          if (!centroid) return null;
          return { position: centroid, text: (props?.name as string) ?? '' };
        })
        .filter(Boolean) as Array<{ position: [number, number]; text: string }>;

      if (labelData.length > 0) {
        result.push(new TextLayer({
          id: 'subregion-labels',
          data: labelData,
          getPosition: (d: { position: [number, number] }) => d.position,
          getText: (d: { text: string }) => d.text,
          getSize: 14,
          getColor: [40, 40, 40, 220],
          fontWeight: 'bold' as unknown as number,
          getTextAnchor: 'middle' as const,
          getAlignmentBaseline: 'center' as const,
          pickable: false,
        }));
      }
    }

    // 4. Stream network lines
    if (showStreamsOnMap && streamGeoJSON && streamGeoJSON.features.length > 0) {
      result.push(new GeoJsonLayer({
        id: 'stream-network',
        data: streamGeoJSON,
        filled: false,
        stroked: true,
        lineWidthMinPixels: 2,
        getLineColor: [30, 100, 200, 200],
        pickable: true,
        onHover: (info: PickingInfo) => {
          if (info.object) {
            const props = (info.object as GeoJSON.Feature).properties as Record<string, unknown>;
            setTooltip({
              x: info.x ?? 0,
              y: info.y ?? 0,
              text: `${props?.name ?? 'Stream'} (Reach ${props?.reach_id ?? '?'}) — click for profile`,
            });
          }
        },
        onClick: (info: PickingInfo) => {
          if (info.object) {
            const props = (info.object as GeoJSON.Feature).properties as Record<string, unknown>;
            const reachId = props?.reach_id as number;
            if (reachId) handleReachClick(reachId);
          }
        },
      }));
    }

    // 5. Boundary condition markers
    if (showBoundaryConditions && bcData.length > 0) {
      result.push(new ScatterplotLayer<BCNodeInfo>({
        id: 'bc-markers',
        data: bcData,
        getPosition: (d) => [d.lng, d.lat],
        getRadius: 400,
        getFillColor: (d) => {
          const c = BC_COLORS[d.bc_type];
          return c ?? [150, 150, 150, 180] as [number, number, number, number];
        },
        getLineColor: [255, 255, 255, 200],
        lineWidthMinPixels: 1,
        stroked: true,
        pickable: true,
        radiusMinPixels: 3,
        radiusMaxPixels: 10,
        onHover: (info: PickingInfo<BCNodeInfo>) => {
          if (info.object) {
            const bc = info.object;
            setTooltip({
              x: info.x ?? 0,
              y: info.y ?? 0,
              text: `BC Node ${bc.node_id} | ${bc.bc_type.replace('_', ' ')} | Value: ${bc.value.toFixed(1)} | Layer ${bc.layer}`,
            });
          }
        },
      }));
    }

    // 6. Well markers
    if (showWells && wellData.length > 0) {
      result.push(new ScatterplotLayer<WellInfo>({
        id: 'well-markers',
        data: wellData,
        getPosition: (d) => [d.lng, d.lat],
        getRadius: (d) => Math.max(300, Math.min(2000, Math.sqrt(Math.abs(d.pump_rate)) * 10)),
        getFillColor: (d) => d.pump_rate <= 0
          ? [220, 50, 50, 180] as [number, number, number, number]   // Red = pumping (negative rate)
          : [50, 50, 220, 180] as [number, number, number, number],  // Blue = injection
        getLineColor: [255, 255, 255, 200],
        lineWidthMinPixels: 1,
        stroked: true,
        pickable: true,
        radiusMinPixels: 3,
        radiusMaxPixels: 15,
        onHover: (info: PickingInfo<WellInfo>) => {
          if (info.object) {
            const w = info.object;
            setTooltip({
              x: info.x ?? 0,
              y: info.y ?? 0,
              text: `${w.name} | Rate: ${w.pump_rate.toFixed(1)} | Elem: ${w.element}`,
            });
          }
        },
      }));
    }

    // 7. GW Nodes (small gray dots)
    if (showNodes && nodeData.length > 0) {
      result.push(new ScatterplotLayer<MeshNodeInfo>({
        id: 'gw-nodes',
        data: nodeData,
        getPosition: (d) => [d.lng, d.lat],
        getRadius: 150,
        getFillColor: [120, 120, 120, 140] as [number, number, number, number],
        stroked: false,
        pickable: true,
        radiusMinPixels: 2,
        radiusMaxPixels: 4,
        onHover: (info: PickingInfo<MeshNodeInfo>) => {
          if (info.object) {
            setTooltip({
              x: info.x ?? 0,
              y: info.y ?? 0,
              text: `Node ${info.object.id}`,
            });
          }
        },
      }));
    }

    // 8. GW location markers
    if (showGWLocations && locations.gw.length > 0) {
      result.push(new ScatterplotLayer({
        id: 'gw-locations',
        data: locations.gw,
        getPosition: (d: HydrographLocation) => [d.lng, d.lat],
        getRadius: 500,
        getFillColor: (d: HydrographLocation) => {
          if (compareMode && comparedLocationIds.includes(d.id)) {
            return [255, 165, 0, 255] as [number, number, number, number]; // Orange = in comparison
          }
          return selectedLocation?.id === d.id && selectedLocation?.type === 'gw'
            ? [25, 118, 210, 255] as [number, number, number, number]
            : [25, 118, 210, 180] as [number, number, number, number];
        },
        getLineColor: [255, 255, 255, 255],
        lineWidthMinPixels: 1,
        stroked: true,
        pickable: true,
        radiusMinPixels: 4,
        radiusMaxPixels: 12,
        onClick: (info: PickingInfo<HydrographLocation>) => {
          if (info.object) {
            if (compareMode) {
              addComparedLocationId(info.object.id);
            } else {
              handleLocationClick(info.object, 'gw');
            }
          }
        },
        updateTriggers: {
          getFillColor: [selectedLocation, compareMode, comparedLocationIds],
        },
      }));
    }

    // Stream location markers
    if (showStreamLocations && locations.stream.length > 0) {
      result.push(new ScatterplotLayer({
        id: 'stream-locations',
        data: locations.stream,
        getPosition: (d: HydrographLocation) => [d.lng, d.lat],
        getRadius: 600,
        getFillColor: (d: HydrographLocation) =>
          selectedLocation?.id === d.id && selectedLocation?.type === 'stream'
            ? [46, 125, 50, 255]
            : [46, 125, 50, 180],
        getLineColor: [255, 255, 255, 255],
        lineWidthMinPixels: 1,
        stroked: true,
        pickable: true,
        radiusMinPixels: 5,
        radiusMaxPixels: 14,
        onClick: (info: PickingInfo<HydrographLocation>) => {
          if (info.object) handleLocationClick(info.object, 'stream');
        },
      }));
    }

    // Subsidence location markers
    if (showSubsidenceLocations && locations.subsidence.length > 0) {
      result.push(new ScatterplotLayer({
        id: 'subsidence-locations',
        data: locations.subsidence,
        getPosition: (d: HydrographLocation) => [d.lng, d.lat],
        getRadius: 500,
        getFillColor: [245, 124, 0, 180] as [number, number, number, number],
        getLineColor: [255, 255, 255, 255],
        lineWidthMinPixels: 1,
        stroked: true,
        pickable: true,
        radiusMinPixels: 4,
        radiusMaxPixels: 12,
        onClick: (info: PickingInfo<HydrographLocation>) => {
          if (info.object) handleLocationClick(info.object, 'subsidence');
        },
      }));
    }

    // 9. Small watershed markers + arc arrows to destination stream nodes
    if (showSmallWatersheds && watershedData && watershedData.watersheds.length > 0) {
      // Watershed marker at first GW routing node
      result.push(new ScatterplotLayer<SmallWatershed>({
        id: 'watershed-markers',
        data: watershedData.watersheds,
        getPosition: (d) => d.marker_position,
        getRadius: 600,
        getFillColor: (d) =>
          d.id === selectedWatershedId
            ? WATERSHED_SELECTED_COLOR
            : WATERSHED_MARKER_COLOR,
        getLineColor: [255, 255, 255, 200],
        lineWidthMinPixels: 1,
        stroked: true,
        pickable: true,
        radiusMinPixels: 5,
        radiusMaxPixels: 12,
        onHover: (info: PickingInfo<SmallWatershed>) => {
          if (info.object) {
            const w = info.object;
            setTooltip({
              x: info.x ?? 0,
              y: info.y ?? 0,
              text: `Watershed ${w.id} | Area: ${w.area.toFixed(0)} | CN: ${w.curve_number.toFixed(1)} | ${w.n_gw_nodes} GW nodes`,
            });
          }
        },
        onClick: (info: PickingInfo<SmallWatershed>) => {
          if (info.object) handleWatershedClick(info.object);
        },
        updateTriggers: {
          getFillColor: [selectedWatershedId],
        },
      }));

      // Selected watershed overlay: GW routing nodes + destination node
      if (selectedWatershedDetail) {
        const selWs = selectedWatershedDetail;

        // GW routing nodes colored by type
        result.push(new ScatterplotLayer<SmallWatershedGWNode>({
          id: 'watershed-gw-nodes',
          data: selWs.gw_nodes,
          getPosition: (d) => [d.lng, d.lat],
          getRadius: 400,
          getFillColor: (d) =>
            d.is_baseflow ? WATERSHED_GW_BASEFLOW_COLOR : WATERSHED_GW_PERC_COLOR,
          getLineColor: [255, 255, 255, 220],
          lineWidthMinPixels: 1,
          stroked: true,
          pickable: true,
          radiusMinPixels: 4,
          radiusMaxPixels: 10,
          onHover: (info: PickingInfo<SmallWatershedGWNode>) => {
            if (info.object) {
              const g = info.object;
              const typeLabel = g.is_baseflow ? `Baseflow Layer ${g.layer}` : 'Percolation';
              setTooltip({
                x: info.x ?? 0,
                y: info.y ?? 0,
                text: `GW Node ${g.node_id} | qmaxwb: ${g.raw_qmaxwb.toFixed(2)} | ${typeLabel}`,
              });
            }
          },
        }));

        // Destination stream node (blue marker)
        if (selWs.dest_coords) {
          result.push(new ScatterplotLayer({
            id: 'watershed-dest-node',
            data: [selWs.dest_coords],
            getPosition: (d: { lng: number; lat: number }) => [d.lng, d.lat],
            getRadius: 500,
            getFillColor: WATERSHED_DEST_COLOR,
            getLineColor: [255, 255, 255, 240],
            lineWidthMinPixels: 2,
            stroked: true,
            pickable: true,
            radiusMinPixels: 6,
            radiusMaxPixels: 12,
            onHover: (info: PickingInfo) => {
              if (info.object) {
                setTooltip({
                  x: info.x ?? 0,
                  y: info.y ?? 0,
                  text: `Dest Stream Node ${selWs.dest_stream_node}`,
                });
              }
            },
          }));
        }
      }
    }

    // 10. Diversion source dots + delivery area highlight
    if (showDiversions && diversionData.length > 0) {
      // Source dots (only in-model diversions that have coordinates)
      const inModelDiversions = diversionData.filter(d => d.source !== null);
      if (inModelDiversions.length > 0) {
        result.push(new ScatterplotLayer<DiversionArc>({
          id: 'diversion-dots',
          data: inModelDiversions,
          getPosition: (d) => [d.source!.lng, d.source!.lat],
          getRadius: 500,
          getFillColor: (d) =>
            d.id === selectedDiversionId
              ? DIVERSION_SELECTED_COLOR
              : DIVERSION_SOURCE_COLOR,
          getLineColor: [255, 255, 255, 200],
          lineWidthMinPixels: 1,
          stroked: true,
          pickable: true,
          radiusMinPixels: 4,
          radiusMaxPixels: 12,
          onHover: (info: PickingInfo<DiversionArc>) => {
            if (info.object) {
              const d = info.object;
              setTooltip({
                x: info.x ?? 0,
                y: info.y ?? 0,
                text: `${d.name} | ${d.destination_type} | Max: ${d.max_rate?.toFixed(1) ?? '?'}`,
              });
            }
          },
          onClick: (info: PickingInfo<DiversionArc>) => {
            if (info.object) {
              handleDiversionClick(info.object.id);
            }
          },
          updateTriggers: {
            getFillColor: [selectedDiversionId],
          },
        }));
      }

      // Delivery area polygons (only when a diversion is selected)
      if (diversionDetail?.delivery?.element_polygons) {
        result.push(new GeoJsonLayer({
          id: 'diversion-delivery-area',
          data: diversionDetail.delivery.element_polygons,
          filled: true,
          stroked: true,
          lineWidthMinPixels: 2,
          getFillColor: DIVERSION_DELIVERY_FILL,
          getLineColor: DIVERSION_DELIVERY_OUTLINE,
          pickable: false,
        }));
      }
    }

    // 11. Cross-section line (when tool is active)
    if (crossSectionMode && crossSectionPoints.length > 0) {
      const lineData = crossSectionPoints.length >= 2
        ? [{ sourcePosition: [crossSectionPoints[0].lng, crossSectionPoints[0].lat],
             targetPosition: [crossSectionPoints[1].lng, crossSectionPoints[1].lat] }]
        : [];

      if (lineData.length > 0) {
        result.push(new LineLayer({
          id: 'cross-section-line',
          data: lineData,
          getSourcePosition: (d: { sourcePosition: number[] }) => d.sourcePosition as [number, number],
          getTargetPosition: (d: { targetPosition: number[] }) => d.targetPosition as [number, number],
          getColor: [255, 50, 50, 220],
          getWidth: 3,
          pickable: false,
        }));
      }

      // Start/end point markers
      result.push(new ScatterplotLayer({
        id: 'cross-section-points',
        data: crossSectionPoints,
        getPosition: (d: { lng: number; lat: number }) => [d.lng, d.lat],
        getRadius: 300,
        getFillColor: [255, 50, 50, 220] as [number, number, number, number],
        getLineColor: [255, 255, 255, 255],
        lineWidthMinPixels: 2,
        stroked: true,
        pickable: false,
        radiusMinPixels: 6,
        radiusMaxPixels: 10,
      }));
    }

    return result;
  }, [
    geojson, headValues, headMin, headMax, headGlobalMin, headGlobalMax,
    propertyValues, propertyMin, propertyMax, propertyMeta,
    colorByHead, colorByDiff, mapColorProperty,
    diffValues, diffMin, diffMax,
    subregionGeoJSON, showSubregions,
    streamGeoJSON, showStreamsOnMap,
    wellData, showWells,
    nodeData, showNodes,
    lakeGeoJSON, showLakes, handleLakeClick,
    bcData, showBoundaryConditions,
    watershedData, showSmallWatersheds, selectedWatershedId, selectedWatershedDetail,
    handleWatershedClick,
    diversionData, showDiversions, selectedDiversionId, diversionDetail, handleDiversionClick,
    locations, showGWLocations, showStreamLocations, showSubsidenceLocations,
    selectedLocation, handleLocationClick, handleElementClick,
    handleReachClick,
    compareMode, comparedLocationIds, addComparedLocationId,
    crossSectionMode, crossSectionPoints,
  ]);

  // Legend props
  const legendLabel = colorByDiff
    ? 'Head Difference (ft)'
    : colorByHead
      ? 'Head (ft)'
      : `${propertyMeta?.name ?? mapColorProperty} ${propertyMeta?.units ? `(${propertyMeta.units})` : ''}`;
  const legendMin = colorByDiff ? -Math.max(Math.abs(diffMin), Math.abs(diffMax)) : colorByHead ? (headGlobalMin ?? headMin) : propertyMin;
  const legendMax = colorByDiff ? Math.max(Math.abs(diffMin), Math.abs(diffMax)) : colorByHead ? (headGlobalMax ?? headMax) : propertyMax;
  const showLegend = colorByDiff ? !!diffValues : colorByHead ? !!headValues : !!propertyValues;
  const isDivergingLegend = colorByDiff;

  if (loading) {
    return (
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
        <CircularProgress />
        <Typography sx={{ ml: 2 }}>Loading map data...</Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ display: 'flex', height: '100%' }}>
      {/* Map area */}
      <Box sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column', position: 'relative' }}>
        <Box sx={{ flexGrow: 1, position: 'relative' }}>
          <DeckGL
            viewState={viewState}
            onViewStateChange={({ viewState: vs }) => setViewState(vs as ViewState)}
            layers={layers}
            controller={true}
            onClick={crossSectionMode ? handleMapClick : undefined}
            getCursor={({ isDragging }: { isDragging: boolean }) =>
              crossSectionMode ? 'crosshair' : isDragging ? 'grabbing' : 'grab'
            }
          >
            <Map mapStyle={mapStyle as string} />
          </DeckGL>

          {/* Cross-section mode indicator */}
          {crossSectionMode && (
            <Box
              sx={{
                position: 'absolute',
                top: 12,
                left: '50%',
                transform: 'translateX(-50%)',
                bgcolor: 'rgba(255,50,50,0.85)',
                color: 'white',
                px: 2,
                py: 0.5,
                borderRadius: 1,
                fontSize: 13,
                fontWeight: 600,
                pointerEvents: 'none',
              }}
            >
              {crossSectionPoints.length === 0
                ? 'Click to set start point'
                : crossSectionPoints.length === 1
                  ? 'Click to set end point'
                  : 'Cross-section computed'}
            </Box>
          )}

          {/* Tooltip */}
          {tooltip && (
            <Box
              sx={{
                position: 'absolute',
                left: tooltip.x + 10,
                top: tooltip.y + 10,
                bgcolor: 'rgba(0,0,0,0.75)',
                color: 'white',
                px: 1,
                py: 0.5,
                borderRadius: 0.5,
                fontSize: 12,
                pointerEvents: 'none',
                whiteSpace: 'nowrap',
              }}
            >
              {tooltip.text}
            </Box>
          )}

          {/* Color legend */}
          {showLegend && (
            <ColorLegend
              min={legendMin}
              max={legendMax}
              label={legendLabel}
              diverging={isDivergingLegend}
            />
          )}
        </Box>

        {/* Hydrograph chart panel */}
        {selectedHydrograph && (
          <HydrographChart
            data={selectedHydrograph}
            observation={observationData}
            onClose={() => {
              setSelectedLocation(null);
              setSelectedHydrograph(null);
            }}
          />
        )}

        {/* Cross-section chart panel */}
        {crossSectionData && crossSectionData.n_cells > 0 && (
          <CrossSectionChart
            data={crossSectionData}
            onClose={() => setCrossSectionData(null)}
          />
        )}

        {/* Lake rating chart panel */}
        {selectedLakeRating && (
          <LakeRatingChart
            data={selectedLakeRating}
            onClose={() => setSelectedLakeRating(null)}
          />
        )}

        {/* Reach profile chart panel */}
        {selectedReachProfile && (
          <ReachProfileChart
            data={selectedReachProfile}
            onClose={() => setSelectedReachProfile(null)}
          />
        )}

        {/* Multi-hydrograph comparison chart */}
        {compareMode && comparisonSeries.length > 0 && (
          <ComparisonChart
            series={comparisonSeries}
            onClose={() => setComparisonSeries([])}
          />
        )}
      </Box>

      {/* Right sidebar controls */}
      <ResultsControls nLayers={nLayers} />

      {/* Element detail panel */}
      {elementDetail && (
        <ElementDetailPanel
          detail={elementDetail}
          onClose={() => {
            setSelectedElement(null);
            setElementDetail(null);
          }}
        />
      )}

      {/* Watershed detail panel */}
      {selectedWatershedDetail && (
        <WatershedDetailPanel
          watershed={selectedWatershedDetail}
          onClose={() => {
            setSelectedWatershedId(null);
            setSelectedWatershedDetail(null);
          }}
        />
      )}

      {/* Diversion panel */}
      {diversionListOpen && (
        <DiversionPanel
          diversions={diversionData}
          onClose={() => setDiversionListOpen(false)}
        />
      )}
    </Box>
  );
}
