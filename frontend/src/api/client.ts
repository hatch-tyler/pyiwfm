/**
 * API client for the IWFM viewer backend.
 */

const API_BASE = '/api';

export interface ModelInfo {
  name: string;
  n_nodes: number;
  n_elements: number;
  n_layers: number;
  has_streams: boolean;
  has_lakes: boolean;
  n_stream_nodes: number | null;
  n_lakes: number | null;
}

export interface BoundsInfo {
  xmin: number;
  xmax: number;
  ymin: number;
  ymax: number;
  zmin: number;
  zmax: number;
}

export interface PropertyInfo {
  id: string;
  name: string;
  units: string;
  description: string;
  cmap: string;
  log_scale: boolean;
}

export interface PropertyData {
  property_id: string;
  name: string;
  units: string;
  values: number[];
  min: number;
  max: number;
  mean: number;
}

export interface StreamNode {
  id: number;
  x: number;
  y: number;
  z: number;
  reach_id: number;
}

export interface StreamNetwork {
  n_nodes: number;
  n_reaches: number;
  nodes: StreamNode[];
  reaches: number[][];
}

// ===================================================================
// Model Summary API
// ===================================================================

export interface MeshSummary {
  n_nodes: number;
  n_elements: number;
  n_layers: number;
  n_subregions: number | null;
}

export interface GroundwaterSummary {
  loaded: boolean;
  n_wells: number | null;
  n_hydrograph_locations: number | null;
  n_boundary_conditions: number | null;
  n_tile_drains: number | null;
  has_aquifer_params: boolean;
}

export interface StreamSummary {
  loaded: boolean;
  n_nodes: number | null;
  n_reaches: number | null;
  n_diversions: number | null;
  n_bypasses: number | null;
}

export interface LakeSummary {
  loaded: boolean;
  n_lakes: number | null;
  n_lake_elements: number | null;
}

export interface RootZoneSummary {
  loaded: boolean;
  n_crop_types: number | null;
  n_land_use_types: number | null;
  land_use_type_names: string[] | null;
  n_soil_parameter_sets: number | null;
  n_land_use_elements: number | null;
  n_missing_land_use: number | null;
  land_use_coverage: string | null;
  n_area_timesteps: number | null;
}

export interface SmallWatershedSummary {
  loaded: boolean;
  n_watersheds: number | null;
}

export interface UnsaturatedZoneSummary {
  loaded: boolean;
  n_layers: number | null;
  n_elements: number | null;
}

export interface AvailableResultsSummary {
  has_head_data: boolean;
  n_head_timesteps: number;
  has_gw_hydrographs: boolean;
  has_stream_hydrographs: boolean;
  n_budget_types: number;
  budget_types: string[];
}

export interface ModelSummary {
  name: string;
  source: string | null;
  mesh: MeshSummary;
  groundwater: GroundwaterSummary;
  streams: StreamSummary;
  lakes: LakeSummary;
  rootzone: RootZoneSummary;
  small_watersheds: SmallWatershedSummary;
  unsaturated_zone: UnsaturatedZoneSummary;
  available_results: AvailableResultsSummary;
}

export async function fetchModelSummary(): Promise<ModelSummary> {
  const response = await fetch(`${API_BASE}/model/summary`);
  if (!response.ok) {
    throw new Error(`Failed to fetch model summary: ${response.statusText}`);
  }
  return response.json();
}

export async function fetchModelInfo(): Promise<ModelInfo> {
  const response = await fetch(`${API_BASE}/model/info`);
  if (!response.ok) {
    throw new Error(`Failed to fetch model info: ${response.statusText}`);
  }
  return response.json();
}

export async function fetchBounds(): Promise<BoundsInfo> {
  const response = await fetch(`${API_BASE}/model/bounds`);
  if (!response.ok) {
    throw new Error(`Failed to fetch bounds: ${response.statusText}`);
  }
  return response.json();
}

export interface MeshData {
  n_points: number;
  n_cells: number;
  n_layers: number;
  points: number[];    // Flat: [x0, y0, z0, x1, y1, z1, ...]
  polys: number[];     // Flat VTK cell array: [nV, v0, v1, ..., nV, v0, v1, ...]
  layer: number[];     // Layer number per surface polygon
}

export async function fetchMesh(): Promise<MeshData> {
  const response = await fetch(`${API_BASE}/mesh/json`);
  if (!response.ok) {
    throw new Error(`Failed to fetch mesh: ${response.statusText}`);
  }
  return response.json();
}

export async function fetchMeshLayer(layer: number): Promise<MeshData> {
  const response = await fetch(`${API_BASE}/mesh/json?layer=${layer}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch layer ${layer}: ${response.statusText}`);
  }
  return response.json();
}

export async function fetchSliceJson(
  angle: number,
  position: number
): Promise<MeshData> {
  const response = await fetch(
    `${API_BASE}/slice/json?angle=${angle}&position=${position}`
  );
  if (!response.ok) {
    throw new Error(`Failed to fetch slice: ${response.statusText}`);
  }
  return response.json();
}

export async function fetchSurfaceMesh(): Promise<string> {
  const response = await fetch(`${API_BASE}/mesh/surface`);
  if (!response.ok) {
    throw new Error(`Failed to fetch surface mesh: ${response.statusText}`);
  }
  return response.text();
}

export async function fetchProperties(): Promise<PropertyInfo[]> {
  const response = await fetch(`${API_BASE}/properties`);
  if (!response.ok) {
    throw new Error(`Failed to fetch properties: ${response.statusText}`);
  }
  return response.json();
}

export async function fetchPropertyData(
  propertyId: string,
  layer: number = 0
): Promise<PropertyData> {
  const response = await fetch(
    `${API_BASE}/properties/${propertyId}?layer=${layer}`
  );
  if (!response.ok) {
    throw new Error(`Failed to fetch property data: ${response.statusText}`);
  }
  return response.json();
}

export async function fetchSlice(
  axis: 'x' | 'y' | 'z',
  position: number
): Promise<string> {
  const response = await fetch(
    `${API_BASE}/slice?axis=${axis}&position=${position}`
  );
  if (!response.ok) {
    throw new Error(`Failed to fetch slice: ${response.statusText}`);
  }
  return response.text();
}

export async function fetchCrossSection(
  startX: number,
  startY: number,
  endX: number,
  endY: number
): Promise<string> {
  const response = await fetch(
    `${API_BASE}/slice/cross-section?start_x=${startX}&start_y=${startY}&end_x=${endX}&end_y=${endY}`
  );
  if (!response.ok) {
    throw new Error(`Failed to fetch cross-section: ${response.statusText}`);
  }
  return response.text();
}

export async function fetchStreams(): Promise<StreamNetwork> {
  const response = await fetch(`${API_BASE}/streams`);
  if (!response.ok) {
    throw new Error(`Failed to fetch streams: ${response.statusText}`);
  }
  return response.json();
}

export async function fetchStreamsVTP(): Promise<string> {
  const response = await fetch(`${API_BASE}/streams/vtp`);
  if (!response.ok) {
    throw new Error(`Failed to fetch streams VTP: ${response.statusText}`);
  }
  return response.text();
}

// ===================================================================
// Results API
// ===================================================================

export interface ResultsInfo {
  has_results: boolean;
  available_budgets: string[];
  n_head_timesteps: number;
  head_time_range: { start: string; end: string } | null;
  n_gw_hydrographs: number;
  n_stream_hydrographs: number;
}

export interface HeadData {
  timestep_index: number;
  datetime: string | null;
  layer: number;
  values: number[];
}

export interface HeadTimes {
  times: string[];
  n_timesteps: number;
}

export interface HydrographLocation {
  id: number;
  lng: number;
  lat: number;
  name: string;
  layer?: number;
  reach_id?: number;
  node_id?: number;
}

export interface HydrographLocations {
  gw: HydrographLocation[];
  stream: HydrographLocation[];
  subsidence: HydrographLocation[];
  tile_drain: HydrographLocation[];
}

export interface HydrographData {
  location_id: number;
  name: string;
  type: string;
  layer?: number;
  times: string[];
  values: number[];
  units: string;
  // Stream-specific: optional dual series (flow + stage)
  flow_values?: number[];
  stage_values?: number[];
  flow_units?: string;
  stage_units?: string;
}

export async function fetchResultsInfo(): Promise<ResultsInfo> {
  const response = await fetch(`${API_BASE}/results/info`);
  if (!response.ok) {
    throw new Error(`Failed to fetch results info: ${response.statusText}`);
  }
  return response.json();
}

export async function fetchHeads(timestep: number, layer: number = 1): Promise<HeadData> {
  const response = await fetch(`${API_BASE}/results/heads?timestep=${timestep}&layer=${layer}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch heads: ${response.statusText}`);
  }
  return response.json();
}

// Head global range (for fixed color scale across animation)
export interface HeadRangeData {
  layer: number;
  min: number;
  max: number;
  n_timesteps: number;
  n_frames_scanned: number;
}

export async function fetchHeadRange(layer: number = 1, maxFrames: number = 50): Promise<HeadRangeData> {
  const response = await fetch(`${API_BASE}/results/head-range?layer=${layer}&max_frames=${maxFrames}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch head range: ${response.statusText}`);
  }
  return response.json();
}

export async function fetchHeadTimes(): Promise<HeadTimes> {
  const response = await fetch(`${API_BASE}/results/head-times`);
  if (!response.ok) {
    throw new Error(`Failed to fetch head times: ${response.statusText}`);
  }
  return response.json();
}

export async function fetchHydrographLocations(): Promise<HydrographLocations> {
  const response = await fetch(`${API_BASE}/results/hydrograph-locations`);
  if (!response.ok) {
    throw new Error(`Failed to fetch hydrograph locations: ${response.statusText}`);
  }
  return response.json();
}

export async function fetchHydrograph(type: string, locationId: number): Promise<HydrographData> {
  const response = await fetch(`${API_BASE}/results/hydrograph?type=${type}&location_id=${locationId}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch hydrograph: ${response.statusText}`);
  }
  return response.json();
}

// All-layers GW hydrograph
export interface GWAllLayersLayer {
  layer: number;
  values: (number | null)[];
}

export interface GWAllLayersData {
  location_id: number;
  node_id: number;
  name: string;
  n_layers: number;
  times: string[];
  layers: GWAllLayersLayer[];
}

export async function fetchGWHydrographAllLayers(locationId: number): Promise<GWAllLayersData> {
  const response = await fetch(`${API_BASE}/results/gw-hydrograph-all-layers?location_id=${locationId}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch all-layers hydrograph: ${response.statusText}`);
  }
  return response.json();
}

// All-layers subsidence hydrograph (mirrors GW all-layers)
export interface SubsidenceAllLayersData {
  location_id: number;
  node_id: number;
  name: string;
  n_layers: number;
  times: string[];
  layers: GWAllLayersLayer[];
}

export async function fetchSubsidenceAllLayers(locationId: number): Promise<SubsidenceAllLayersData> {
  const response = await fetch(`${API_BASE}/results/subsidence-all-layers?location_id=${locationId}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch subsidence all-layers: ${response.statusText}`);
  }
  return response.json();
}

// Per-element head values
export interface HeadByElementData {
  timestep_index: number;
  datetime: string | null;
  layer: number;
  values: (number | null)[];
  min: number;
  max: number;
}

export async function fetchHeadsByElement(timestep: number, layer: number = 1): Promise<HeadByElementData> {
  const response = await fetch(`${API_BASE}/results/heads-by-element?timestep=${timestep}&layer=${layer}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch heads by element: ${response.statusText}`);
  }
  return response.json();
}

// ===================================================================
// Mesh GeoJSON API
// ===================================================================

export async function fetchMeshGeoJSON(layer: number = 1): Promise<GeoJSON.FeatureCollection> {
  const response = await fetch(`${API_BASE}/mesh/geojson?layer=${layer}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch mesh GeoJSON: ${response.statusText}`);
  }
  return response.json();
}

export async function fetchHeadMap(
  timestep: number,
  layer: number = 1,
): Promise<GeoJSON.FeatureCollection> {
  const response = await fetch(`${API_BASE}/mesh/head-map?timestep=${timestep}&layer=${layer}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch head map: ${response.statusText}`);
  }
  return response.json();
}

// ===================================================================
// Mesh Nodes API
// ===================================================================

export interface MeshNodeInfo {
  id: number;
  lng: number;
  lat: number;
}

export interface MeshNodesResponse {
  n_nodes: number;
  nodes: MeshNodeInfo[];
}

export async function fetchMeshNodes(layer: number = 1): Promise<MeshNodesResponse> {
  const response = await fetch(`${API_BASE}/mesh/nodes?layer=${layer}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch mesh nodes: ${response.statusText}`);
  }
  return response.json();
}

// ===================================================================
// Subregion API
// ===================================================================

export async function fetchSubregions(): Promise<GeoJSON.FeatureCollection> {
  const response = await fetch(`${API_BASE}/mesh/subregions`);
  if (!response.ok) {
    throw new Error(`Failed to fetch subregions: ${response.statusText}`);
  }
  return response.json();
}

// ===================================================================
// Property Map API
// ===================================================================

export interface PropertyMapMetadata {
  property: string;
  name: string;
  units: string;
  log_scale: boolean;
  layer: number;
  min: number;
  max: number;
}

export interface PropertyMapResponse extends GeoJSON.FeatureCollection {
  metadata: PropertyMapMetadata;
}

export async function fetchPropertyMap(
  property: string,
  layer: number = 1,
): Promise<PropertyMapResponse> {
  const response = await fetch(
    `${API_BASE}/mesh/property-map?property=${encodeURIComponent(property)}&layer=${layer}`
  );
  if (!response.ok) {
    throw new Error(`Failed to fetch property map: ${response.statusText}`);
  }
  return response.json();
}

// ===================================================================
// Stream GeoJSON API
// ===================================================================

export async function fetchStreamGeoJSON(): Promise<GeoJSON.FeatureCollection> {
  const response = await fetch(`${API_BASE}/streams/geojson`);
  if (!response.ok) {
    throw new Error(`Failed to fetch stream GeoJSON: ${response.statusText}`);
  }
  return response.json();
}

// ===================================================================
// Groundwater API
// ===================================================================

export interface WellInfo {
  id: number;
  lng: number;
  lat: number;
  name: string;
  element: number;
  pump_rate: number;
  max_pump_rate: number;
  top_screen: number;
  bottom_screen: number;
  layers: number[];
}

export interface WellsResponse {
  n_wells: number;
  wells: WellInfo[];
}

export async function fetchWells(): Promise<WellsResponse> {
  const response = await fetch(`${API_BASE}/groundwater/wells`);
  if (!response.ok) {
    throw new Error(`Failed to fetch wells: ${response.statusText}`);
  }
  return response.json();
}

// ===================================================================
// Element Detail API
// ===================================================================

export async function fetchElementDetail(elementId: number, timestep?: number): Promise<Record<string, unknown>> {
  const params = timestep !== undefined ? `?timestep=${timestep}` : '';
  const response = await fetch(`${API_BASE}/mesh/element/${elementId}${params}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch element detail: ${response.statusText}`);
  }
  return response.json();
}

// ===================================================================
// Boundary Conditions API
// ===================================================================

export interface BCNodeInfo {
  bc_id: number;
  node_id: number;
  lng: number;
  lat: number;
  bc_type: string;
  value: number;
  layer: number;
  conductance: number | null;
}

export interface BCResponse {
  n_conditions: number;
  nodes: BCNodeInfo[];
}

export async function fetchBoundaryConditions(): Promise<BCResponse> {
  const response = await fetch(`${API_BASE}/groundwater/boundary-conditions`);
  if (!response.ok) {
    throw new Error(`Failed to fetch boundary conditions: ${response.statusText}`);
  }
  return response.json();
}

// ===================================================================
// Head Difference API
// ===================================================================

export interface HeadDiffData {
  timestep_a: number;
  timestep_b: number;
  datetime_a: string | null;
  datetime_b: string | null;
  layer: number;
  values: (number | null)[];
  min: number;
  max: number;
}

export async function fetchHeadDiff(
  timestepA: number,
  timestepB: number,
  layer: number = 1,
): Promise<HeadDiffData> {
  const response = await fetch(
    `${API_BASE}/results/head-diff?timestep_a=${timestepA}&timestep_b=${timestepB}&layer=${layer}`
  );
  if (!response.ok) {
    throw new Error(`Failed to fetch head diff: ${response.statusText}`);
  }
  return response.json();
}

// ===================================================================
// Lake API
// ===================================================================

export async function fetchLakesGeoJSON(): Promise<GeoJSON.FeatureCollection> {
  const response = await fetch(`${API_BASE}/lakes/geojson`);
  if (!response.ok) {
    throw new Error(`Failed to fetch lakes GeoJSON: ${response.statusText}`);
  }
  return response.json();
}

export interface LakeRatingData {
  lake_id: number;
  name: string;
  elevations: number[];
  areas: number[];
  volumes: number[];
  n_points: number;
}

export async function fetchLakeRating(lakeId: number): Promise<LakeRatingData> {
  const response = await fetch(`${API_BASE}/lakes/${lakeId}/rating`);
  if (!response.ok) {
    throw new Error(`Failed to fetch lake rating: ${response.statusText}`);
  }
  return response.json();
}

// ===================================================================
// Cross-Section JSON API
// ===================================================================

export interface CrossSectionData {
  n_points: number;
  n_cells: number;
  points: number[];
  polys: number[];
  layer: number[];
  distance: number[];
  start: { lng: number; lat: number; x: number; y: number };
  end: { lng: number; lat: number; x: number; y: number };
  total_distance: number;
}

export interface CrossSectionHeadLayer {
  layer: number;
  top: (number | null)[];
  bottom: (number | null)[];
  head: (number | null)[];
}

export interface CrossSectionHeadData {
  n_samples: number;
  n_layers: number;
  distance: number[];
  timestep: number;
  datetime: string | null;
  layers: CrossSectionHeadLayer[];
  gs_elev: (number | null)[];
  mask: boolean[];
}

export async function fetchCrossSectionHeads(
  startLng: number,
  startLat: number,
  endLng: number,
  endLat: number,
  timestep: number = 0,
  nSamples: number = 100,
): Promise<CrossSectionHeadData> {
  const response = await fetch(
    `${API_BASE}/slice/cross-section/heads?start_lng=${startLng}&start_lat=${startLat}&end_lng=${endLng}&end_lat=${endLat}&timestep=${timestep}&n_samples=${nSamples}`
  );
  if (!response.ok) {
    throw new Error(`Failed to fetch cross-section heads: ${response.statusText}`);
  }
  return response.json();
}

export async function fetchCrossSectionJSON(
  startLng: number,
  startLat: number,
  endLng: number,
  endLat: number,
): Promise<CrossSectionData> {
  const response = await fetch(
    `${API_BASE}/slice/cross-section/json?start_lng=${startLng}&start_lat=${startLat}&end_lng=${endLng}&end_lat=${endLat}`
  );
  if (!response.ok) {
    throw new Error(`Failed to fetch cross-section: ${response.statusText}`);
  }
  return response.json();
}

// ===================================================================
// Root Zone / Land Use API
// ===================================================================

export interface LandUseElement {
  element_id: number;
  fractions: { agricultural: number; urban: number; native_riparian: number; water: number };
  dominant: string;
  total_area: number;
}

export interface LandUseData {
  n_elements: number;
  elements: LandUseElement[];
}

export interface LandUseTimesteps {
  n_timesteps: number;
  dates: string[];
}

export interface LandUseAreaSeries {
  n_cols: number;
  areas: number[][];
}

export interface ElementLandUseTimeseries {
  element_id: number;
  dates: string[];
  nonponded?: LandUseAreaSeries;
  ponded?: LandUseAreaSeries;
  urban?: LandUseAreaSeries;
  native?: LandUseAreaSeries;
}

export async function fetchLandUse(timestep = 0): Promise<LandUseData> {
  const response = await fetch(`${API_BASE}/rootzone/land-use?timestep=${timestep}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch land use: ${response.statusText}`);
  }
  return response.json();
}

export async function fetchLandUseTimesteps(): Promise<LandUseTimesteps> {
  const response = await fetch(`${API_BASE}/rootzone/timesteps`);
  if (!response.ok) {
    throw new Error(`Failed to fetch land use timesteps: ${response.statusText}`);
  }
  return response.json();
}

export async function fetchLandUseDates(): Promise<LandUseTimesteps> {
  return fetchLandUseTimesteps();
}

export async function fetchElementLandUseTimeseries(elementId: number): Promise<ElementLandUseTimeseries> {
  const response = await fetch(`${API_BASE}/rootzone/land-use/${elementId}/timeseries`);
  if (!response.ok) {
    throw new Error(`Failed to fetch element land use timeseries: ${response.statusText}`);
  }
  return response.json();
}

export async function fetchElementCrops(elementId: number): Promise<Record<string, unknown>> {
  const response = await fetch(`${API_BASE}/rootzone/land-use/${elementId}/crops`);
  if (!response.ok) {
    throw new Error(`Failed to fetch element crops: ${response.statusText}`);
  }
  return response.json();
}

// ===================================================================
// Budget API
// ===================================================================

export interface BudgetLocation {
  id: number;
  name: string;
}

export interface BudgetColumn {
  id: number;
  name: string;
  units: string;
}

export interface BudgetColumnData {
  name: string;
  values: number[];
  units: string;
}

export interface BudgetUnitsMetadata {
  source_volume_unit: string;
  source_area_unit: string;
  source_area_output_unit?: string;
  source_length_unit: string;
  timestep_unit: string;
  has_volume_columns: boolean;
  has_area_columns: boolean;
  has_length_columns: boolean;
}

export interface BudgetData {
  location: string;
  times: string[];
  columns: BudgetColumnData[];
  units_metadata?: BudgetUnitsMetadata;
}

export interface BudgetSummary {
  location: string;
  n_timesteps: number;
  totals: Record<string, number>;
  averages: Record<string, number>;
}

export async function fetchBudgetTypes(): Promise<string[]> {
  const response = await fetch(`${API_BASE}/budgets/types`);
  if (!response.ok) {
    throw new Error(`Failed to fetch budget types: ${response.statusText}`);
  }
  return response.json();
}

export async function fetchBudgetLocations(budgetType: string): Promise<{ locations: BudgetLocation[] }> {
  const response = await fetch(`${API_BASE}/budgets/${budgetType}/locations`);
  if (!response.ok) {
    throw new Error(`Failed to fetch budget locations: ${response.statusText}`);
  }
  return response.json();
}

export async function fetchBudgetColumns(budgetType: string, location: string): Promise<{ columns: BudgetColumn[] }> {
  const response = await fetch(`${API_BASE}/budgets/${budgetType}/columns?location=${encodeURIComponent(location)}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch budget columns: ${response.statusText}`);
  }
  return response.json();
}

export async function fetchBudgetData(budgetType: string, location: string, columns: string = 'all'): Promise<BudgetData> {
  const response = await fetch(
    `${API_BASE}/budgets/${budgetType}/data?location=${encodeURIComponent(location)}&columns=${columns}`
  );
  if (!response.ok) {
    throw new Error(`Failed to fetch budget data: ${response.statusText}`);
  }
  return response.json();
}

export interface BudgetLocationGeometry {
  spatial_type: string;
  location_index: number;
  location_name?: string;
  geometry: GeoJSON.Point | null;
}

export async function fetchBudgetLocationGeometry(
  budgetType: string,
  location: string,
): Promise<BudgetLocationGeometry> {
  const response = await fetch(
    `${API_BASE}/budgets/${budgetType}/location-geometry?location=${encodeURIComponent(location)}`
  );
  if (!response.ok) {
    throw new Error(`Failed to fetch budget location geometry: ${response.statusText}`);
  }
  return response.json();
}

export type BudgetGlossary = Record<string, Record<string, string>>;

export async function fetchBudgetGlossary(): Promise<BudgetGlossary> {
  const response = await fetch(`${API_BASE}/budgets/glossary`);
  if (!response.ok) {
    throw new Error(`Failed to fetch budget glossary: ${response.statusText}`);
  }
  return response.json();
}

export async function fetchBudgetSummary(budgetType: string, location: string): Promise<BudgetSummary> {
  const response = await fetch(
    `${API_BASE}/budgets/${budgetType}/summary?location=${encodeURIComponent(location)}`
  );
  if (!response.ok) {
    throw new Error(`Failed to fetch budget summary: ${response.statusText}`);
  }
  return response.json();
}

// ===================================================================
// Observations API
// ===================================================================

export interface ObservationFile {
  id: string;
  filename: string;
  location_id: number | null;
  type: string;
  n_records: number;
}

export interface ObservationData {
  times: string[];
  values: number[];
  units: string;
}

export interface ObservationPreview {
  headers: string[];
  sample_rows: string[][];
  n_rows: number;
}

export interface UploadResult {
  n_observations: number;
  n_records: number;
  observations: Array<{
    observation_id: string;
    filename: string;
    n_records: number;
    location_id: number | null;
    start_time: string | null;
    end_time: string | null;
  }>;
  unmatched_locations: string[];
}

/** Client-side CSV preview: reads headers and first sample rows. */
export async function previewObservationFile(file: File): Promise<ObservationPreview> {
  const text = await file.text();
  const lines = text.split(/\r?\n/).filter((l) => l.trim().length > 0);
  if (lines.length === 0) throw new Error('File is empty');
  const headers = lines[0].split(',').map((h) => h.trim());
  const dataLines = lines.slice(1);
  const sampleRows = dataLines.slice(0, 10).map((l) => l.split(',').map((c) => c.trim()));
  return { headers, sample_rows: sampleRows, n_rows: dataLines.length };
}

export async function uploadObservation(
  file: File,
  type: string = 'gw',
  dateCol?: number,
  valueCol?: number,
  locationCol?: number,
): Promise<UploadResult> {
  const formData = new FormData();
  formData.append('file', file);
  const params = new URLSearchParams({ type });
  if (dateCol !== undefined) params.set('date_col', String(dateCol));
  if (valueCol !== undefined) params.set('value_col', String(valueCol));
  if (locationCol !== undefined && locationCol >= 0) params.set('location_col', String(locationCol));
  const response = await fetch(`${API_BASE}/observations/upload?${params}`, {
    method: 'POST',
    body: formData,
  });
  if (!response.ok) {
    throw new Error(`Failed to upload observation: ${response.statusText}`);
  }
  return response.json();
}

export async function fetchObservations(): Promise<ObservationFile[]> {
  const response = await fetch(`${API_BASE}/observations`);
  if (!response.ok) {
    throw new Error(`Failed to fetch observations: ${response.statusText}`);
  }
  return response.json();
}

export async function fetchObservationData(obsId: string): Promise<ObservationData> {
  const response = await fetch(`${API_BASE}/observations/${obsId}/data`);
  if (!response.ok) {
    throw new Error(`Failed to fetch observation data: ${response.statusText}`);
  }
  return response.json();
}

export async function deleteObservation(obsId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/observations/${obsId}`, { method: 'DELETE' });
  if (!response.ok) {
    throw new Error(`Failed to delete observation: ${response.statusText}`);
  }
}

// ===================================================================
// Small Watersheds API
// ===================================================================

export interface SmallWatershedGWNode {
  node_id: number;
  lng: number;
  lat: number;
  is_baseflow: boolean;
  layer: number;
  max_perc_rate: number;
  raw_qmaxwb: number;
}

export interface SmallWatershed {
  id: number;
  area: number;
  dest_stream_node: number;
  dest_coords: { lng: number; lat: number } | null;
  marker_position: [number, number];
  n_gw_nodes: number;
  gw_nodes: SmallWatershedGWNode[];
  curve_number: number;
  // Root zone parameters
  wilting_point: number;
  field_capacity: number;
  total_porosity: number;
  lambda_param: number;
  root_depth: number;
  hydraulic_cond: number;
  kunsat_method: number;
  // Aquifer parameters
  gw_threshold: number;
  max_gw_storage: number;
  surface_flow_coeff: number;
  baseflow_coeff: number;
}

export interface SmallWatershedData {
  n_watersheds: number;
  watersheds: SmallWatershed[];
}

export async function fetchSmallWatersheds(): Promise<SmallWatershedData> {
  const response = await fetch(`${API_BASE}/small-watersheds`);
  if (!response.ok) {
    throw new Error(`Failed to fetch small watersheds: ${response.statusText}`);
  }
  return response.json();
}

// ===================================================================
// Diversions API
// ===================================================================

export interface DiversionArc {
  id: number;
  name: string;
  source_node: number;
  source: { lng: number; lat: number } | null;
  destination_type: string;
  destination_id: number;
  destination: { lng: number; lat: number } | null;
  max_rate: number | null;
  priority: number;
}

export interface DiversionsResponse {
  n_diversions: number;
  diversions: DiversionArc[];
}

export interface DiversionDelivery {
  dest_type: string;
  dest_id: number;
  element_ids: number[];
  element_polygons: GeoJSON.FeatureCollection | null;
}

export interface DiversionTimeseries {
  times: string[];
  max_diversion: number[] | null;
  delivery: number[] | null;
}

export interface DiversionDetail {
  id: number;
  name: string;
  source_node: number;
  source: { lng: number; lat: number } | null;
  destination_type: string;
  destination_id: number;
  destination: { lng: number; lat: number } | null;
  max_rate: number | null;
  priority: number;
  delivery: DiversionDelivery;
  timeseries: DiversionTimeseries | null;
}

export async function fetchDiversions(): Promise<DiversionsResponse> {
  const response = await fetch(`${API_BASE}/streams/diversions`);
  if (!response.ok) {
    throw new Error(`Failed to fetch diversions: ${response.statusText}`);
  }
  return response.json();
}

export async function fetchDiversionDetail(divId: number): Promise<DiversionDetail> {
  const response = await fetch(`${API_BASE}/streams/diversions/${divId}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch diversion detail: ${response.statusText}`);
  }
  return response.json();
}

// ===================================================================
// Budget Spatial API
// ===================================================================

export interface BudgetSpatialData {
  budget_type: string;
  column: string;
  stat: string;
  n_locations: number;
  locations: Array<{ id: number; name: string; value: number }>;
  min: number;
  max: number;
  available_columns: string[];
}

export async function fetchBudgetSpatial(
  budgetType: string,
  column: string = '',
  stat: string = 'total',
): Promise<BudgetSpatialData> {
  const params = new URLSearchParams({ stat });
  if (column) params.set('column', column);
  const response = await fetch(`${API_BASE}/budgets/${budgetType}/spatial?${params}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch budget spatial: ${response.statusText}`);
  }
  return response.json();
}

// ===================================================================
// Multi-Hydrograph Comparison API
// ===================================================================

export interface MultiHydrographResponse {
  type: string;
  n_series: number;
  series: HydrographData[];
}

export async function fetchHydrographsMulti(
  type: string,
  ids: number[],
): Promise<MultiHydrographResponse> {
  const response = await fetch(
    `${API_BASE}/results/hydrographs-multi?type=${type}&ids=${ids.join(',')}`
  );
  if (!response.ok) {
    throw new Error(`Failed to fetch multi hydrographs: ${response.statusText}`);
  }
  return response.json();
}

// ===================================================================
// Drawdown API
// ===================================================================

export interface DrawdownData {
  layer: number;
  reference_timestep: number;
  n_timesteps: number;
  timesteps: Array<{
    timestep: number;
    datetime: string | null;
    values: (number | null)[];
    min: number;
    max: number;
  }>;
}

export async function fetchDrawdown(
  layer: number = 1,
  referenceTimestep: number = 0,
): Promise<DrawdownData> {
  const response = await fetch(
    `${API_BASE}/results/drawdown?layer=${layer}&reference_timestep=${referenceTimestep}`
  );
  if (!response.ok) {
    throw new Error(`Failed to fetch drawdown: ${response.statusText}`);
  }
  return response.json();
}

// ===================================================================
// Stream Reach Profile API
// ===================================================================

export interface ReachProfileNode {
  stream_node_id: number;
  gw_node_id: number;
  lng: number;
  lat: number;
  distance: number;
  ground_surface_elev: number;
  bed_elev: number;
  mannings_n: number;
  conductivity: number;
  bed_thickness: number;
  has_rating: boolean;
}

export interface ReachProfileData {
  reach_id: number;
  name: string;
  n_nodes: number;
  total_length: number;
  nodes: ReachProfileNode[];
}

export async function fetchReachProfile(reachId: number): Promise<ReachProfileData> {
  const response = await fetch(`${API_BASE}/streams/reach-profile?reach_id=${reachId}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch reach profile: ${response.statusText}`);
  }
  return response.json();
}

// Stream node rating table (stage-discharge curve)
export interface StreamNodeRatingData {
  stream_node_id: number;
  bottom_elev: number;
  stages: number[];
  flows: number[];
  n_points: number;
}

export async function fetchStreamNodeRating(nodeId: number): Promise<StreamNodeRatingData> {
  const response = await fetch(`${API_BASE}/streams/node-rating?node_id=${nodeId}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch stream node rating: ${response.statusText}`);
  }
  return response.json();
}

// ===================================================================
// Water Balance API
// ===================================================================

export interface WaterBalanceData {
  nodes: string[];
  links: Array<{
    source: number;
    target: number;
    value: number;
    label: string;
  }>;
}

export async function fetchWaterBalance(): Promise<WaterBalanceData> {
  const response = await fetch(`${API_BASE}/budgets/water-balance`);
  if (!response.ok) {
    throw new Error(`Failed to fetch water balance: ${response.statusText}`);
  }
  return response.json();
}

// ===================================================================
// Well Impact API
// ===================================================================

export interface WellImpactData {
  well_id: number;
  name: string;
  center: { lng: number; lat: number };
  pump_rate: number;
  transmissivity: number;
  storativity: number;
  time_days: number;
  n_contours: number;
  contours: Array<{
    radius_ft: number;
    radius_deg: number;
    drawdown_ft: number;
    u: number;
  }>;
}

export async function fetchWellImpact(
  wellId: number,
  time: number = 365,
): Promise<WellImpactData> {
  const response = await fetch(
    `${API_BASE}/groundwater/well-impact?well_id=${wellId}&time=${time}`
  );
  if (!response.ok) {
    throw new Error(`Failed to fetch well impact: ${response.statusText}`);
  }
  return response.json();
}

// ===================================================================
// Export API (download triggers)
// ===================================================================

export function getExportHeadsCsvUrl(timestep: number, layer: number): string {
  return `${API_BASE}/export/heads-csv?timestep=${timestep}&layer=${layer}`;
}

export function getExportMeshGeoJsonUrl(layer: number): string {
  return `${API_BASE}/export/mesh-geojson?layer=${layer}`;
}

export function getExportBudgetCsvUrl(budgetType: string, location: string): string {
  return `${API_BASE}/export/budget-csv?budget_type=${budgetType}&location=${encodeURIComponent(location)}`;
}

export function getExportHydrographCsvUrl(type: string, locationId: number): string {
  return `${API_BASE}/export/hydrograph-csv?type=${type}&location_id=${locationId}`;
}
