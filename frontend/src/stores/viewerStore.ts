/**
 * Zustand store for viewer state management.
 */

import { create } from 'zustand';
import type {
  ModelInfo, BoundsInfo, PropertyInfo,
  ResultsInfo, HydrographData, ObservationFile,
  CrossSectionData, LakeRatingData, ReachProfileData,
  DiversionDetail, SmallWatershed,
} from '../api/client';

export interface ViewerState {
  // Model data
  modelInfo: ModelInfo | null;
  bounds: BoundsInfo | null;
  properties: PropertyInfo[];

  // Loading states
  isLoading: boolean;
  error: string | null;
  meshLoaded: boolean;

  // View settings
  activeProperty: string;
  activeLayer: number;
  opacity: number;
  showEdges: boolean;
  zExaggeration: number;

  // Per-layer visibility
  visibleLayers: boolean[];

  // Visibility toggles
  showMesh: boolean;
  showStreams: boolean;
  showCrossSection: boolean;

  // Cross-section settings
  sliceAngle: number;
  slicePosition: number;

  // === Animation speed ===
  animationSpeed: number;  // ms per frame (100-2000)

  // === Tab Navigation ===
  activeTab: number;  // 0=Overview, 1=3D, 2=Results, 3=Budgets

  // === Results Map state ===
  resultsInfo: ResultsInfo | null;
  headTimestep: number;
  headTimes: string[];
  headLayer: number;
  selectedLocation: { id: number; type: 'gw' | 'stream' | 'subsidence' } | null;
  selectedHydrograph: HydrographData | null;
  showGWLocations: boolean;
  showStreamLocations: boolean;
  showSubsidenceLocations: boolean;
  isAnimating: boolean;

  // === Head global range (for fixed animation color scale) ===
  headGlobalMin: number | null;
  headGlobalMax: number | null;

  // === Map overlay toggles ===
  showSubregions: boolean;
  showStreamsOnMap: boolean;
  showWells: boolean;
  mapColorProperty: string;  // 'head' | property_id from /api/properties

  // === Element detail ===
  selectedElement: number | null;
  elementDetail: Record<string, unknown> | null;

  // === Phase 2: Additional overlays ===
  showLakes: boolean;
  showBoundaryConditions: boolean;
  selectedLakeRating: LakeRatingData | null;

  // === Phase 2: Head difference mode ===
  headDiffMode: boolean;
  headDiffTimestepA: number;
  headDiffTimestepB: number;

  // === Phase 2: Cross-section tool ===
  crossSectionMode: boolean;
  crossSectionPoints: Array<{ lng: number; lat: number }>;
  crossSectionData: CrossSectionData | null;

  // === GW Nodes layer ===
  showNodes: boolean;

  // === Basemap selection ===
  selectedBasemap: string;

  // === Phase 3: Additional overlays ===
  showSmallWatersheds: boolean;
  showDiversions: boolean;

  // === Watershed detail ===
  selectedWatershedId: number | null;
  selectedWatershedDetail: SmallWatershed | null;

  // === Diversion detail ===
  selectedDiversionId: number | null;
  diversionDetail: DiversionDetail | null;
  diversionListOpen: boolean;

  // === Phase 3: Multi-hydrograph comparison ===
  compareMode: boolean;
  comparedLocationIds: number[];

  // === Phase 3: Stream reach profile ===
  selectedReachProfile: ReachProfileData | null;

  // === Phase 3: Budget views ===
  showBudgetSankey: boolean;

  // === Budget state ===
  activeBudgetType: string;
  activeBudgetLocation: string;
  budgetChartType: 'area' | 'bar' | 'line';
  budgetVolumeUnit: string;
  budgetRateUnit: string;
  budgetAreaUnit: string;
  budgetLengthUnit: string;
  budgetTimeAgg: string;
  showBudgetGlossary: boolean;
  budgetAnalysisMode: 'timeseries' | 'monthly_pattern' | 'component_ratios' | 'cumulative_departure' | 'exceedance';

  // === Observation state ===
  observations: ObservationFile[];

  // Actions
  setModelInfo: (info: ModelInfo) => void;
  setBounds: (bounds: BoundsInfo) => void;
  setProperties: (props: PropertyInfo[]) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  setMeshLoaded: (loaded: boolean) => void;
  setActiveProperty: (prop: string) => void;
  setActiveLayer: (layer: number) => void;
  setOpacity: (opacity: number) => void;
  setShowEdges: (show: boolean) => void;
  setZExaggeration: (factor: number) => void;
  setShowMesh: (show: boolean) => void;
  setShowStreams: (show: boolean) => void;
  setShowCrossSection: (show: boolean) => void;
  setSliceAngle: (angle: number) => void;
  setSlicePosition: (pos: number) => void;
  setLayerVisible: (layer: number, visible: boolean) => void;
  setAllLayersVisible: (visible: boolean) => void;

  // Animation speed
  setAnimationSpeed: (speed: number) => void;

  // Tab
  setActiveTab: (tab: number) => void;

  // Results Map
  setResultsInfo: (info: ResultsInfo) => void;
  setHeadTimestep: (ts: number) => void;
  setHeadTimes: (times: string[]) => void;
  setHeadLayer: (layer: number) => void;
  setSelectedLocation: (loc: { id: number; type: 'gw' | 'stream' | 'subsidence' } | null) => void;
  setSelectedHydrograph: (data: HydrographData | null) => void;
  setShowGWLocations: (show: boolean) => void;
  setShowStreamLocations: (show: boolean) => void;
  setShowSubsidenceLocations: (show: boolean) => void;
  setIsAnimating: (a: boolean) => void;

  // Head global range
  setHeadGlobalRange: (min: number | null, max: number | null) => void;

  // Map overlays
  setShowSubregions: (show: boolean) => void;
  setShowStreamsOnMap: (show: boolean) => void;
  setShowWells: (show: boolean) => void;
  setMapColorProperty: (prop: string) => void;

  // Element detail
  setSelectedElement: (id: number | null) => void;
  setElementDetail: (detail: Record<string, unknown> | null) => void;

  // Phase 2 overlays
  setShowLakes: (show: boolean) => void;
  setShowBoundaryConditions: (show: boolean) => void;
  setSelectedLakeRating: (data: LakeRatingData | null) => void;

  // Phase 2 head diff
  setHeadDiffMode: (on: boolean) => void;
  setHeadDiffTimestepA: (ts: number) => void;
  setHeadDiffTimestepB: (ts: number) => void;

  // Phase 2 cross-section
  setCrossSectionMode: (on: boolean) => void;
  setCrossSectionPoints: (pts: Array<{ lng: number; lat: number }>) => void;
  setCrossSectionData: (data: CrossSectionData | null) => void;

  // GW Nodes
  setShowNodes: (show: boolean) => void;

  // Basemap
  setSelectedBasemap: (key: string) => void;

  // Phase 3 overlays
  setShowSmallWatersheds: (show: boolean) => void;
  setShowDiversions: (show: boolean) => void;

  // Watershed detail
  setSelectedWatershedId: (id: number | null) => void;
  setSelectedWatershedDetail: (detail: SmallWatershed | null) => void;

  // Diversion detail
  setSelectedDiversionId: (id: number | null) => void;
  setDiversionDetail: (detail: DiversionDetail | null) => void;
  setDiversionListOpen: (open: boolean) => void;

  // Phase 3 comparison
  setCompareMode: (on: boolean) => void;
  setComparedLocationIds: (ids: number[]) => void;
  addComparedLocationId: (id: number) => void;
  removeComparedLocationId: (id: number) => void;

  // Phase 3 stream reach profile
  setSelectedReachProfile: (data: ReachProfileData | null) => void;

  // Phase 3 budget sankey
  setShowBudgetSankey: (show: boolean) => void;

  // Budget
  setActiveBudgetType: (t: string) => void;
  setActiveBudgetLocation: (l: string) => void;
  setBudgetChartType: (t: 'area' | 'bar' | 'line') => void;
  setBudgetVolumeUnit: (u: string) => void;
  setBudgetRateUnit: (u: string) => void;
  setBudgetAreaUnit: (u: string) => void;
  setBudgetLengthUnit: (u: string) => void;
  setBudgetTimeAgg: (a: string) => void;
  setShowBudgetGlossary: (show: boolean) => void;
  setBudgetAnalysisMode: (mode: 'timeseries' | 'monthly_pattern' | 'component_ratios' | 'cumulative_departure' | 'exceedance') => void;

  // Observations
  setObservations: (obs: ObservationFile[]) => void;
}

export const useViewerStore = create<ViewerState>((set) => ({
  // Model data
  modelInfo: null,
  bounds: null,
  properties: [],

  // Loading states
  isLoading: true,
  error: null,
  meshLoaded: false,

  // View settings
  activeProperty: 'layer',
  activeLayer: 0,
  opacity: 1.0,
  showEdges: false,
  zExaggeration: 1.0,

  // Per-layer visibility
  visibleLayers: [],

  // Visibility toggles
  showMesh: true,
  showStreams: false,
  showCrossSection: false,

  // Cross-section settings
  sliceAngle: 0,
  slicePosition: 0.5,

  // Animation speed
  animationSpeed: 500,

  // Tab navigation
  activeTab: 0,

  // Results Map
  resultsInfo: null,
  headTimestep: 0,
  headTimes: [],
  headLayer: 1,
  selectedLocation: null,
  selectedHydrograph: null,
  showGWLocations: true,
  showStreamLocations: true,
  showSubsidenceLocations: true,
  isAnimating: false,

  // Head global range
  headGlobalMin: null,
  headGlobalMax: null,

  // Map overlays
  showSubregions: false,
  showStreamsOnMap: false,
  showWells: false,
  mapColorProperty: 'head',

  // Element detail
  selectedElement: null,
  elementDetail: null,

  // Phase 2 overlays
  showLakes: false,
  showBoundaryConditions: false,
  selectedLakeRating: null,

  // Phase 2 head diff
  headDiffMode: false,
  headDiffTimestepA: 0,
  headDiffTimestepB: 0,

  // Phase 2 cross-section
  crossSectionMode: false,
  crossSectionPoints: [],
  crossSectionData: null,

  // GW Nodes
  showNodes: false,

  // Basemap
  selectedBasemap: 'positron',

  // Phase 3 overlays
  showSmallWatersheds: false,
  showDiversions: false,

  // Watershed detail
  selectedWatershedId: null,
  selectedWatershedDetail: null,

  // Diversion detail
  selectedDiversionId: null,
  diversionDetail: null,
  diversionListOpen: false,

  // Phase 3 comparison
  compareMode: false,
  comparedLocationIds: [],

  // Phase 3 stream reach profile
  selectedReachProfile: null,

  // Phase 3 budget sankey
  showBudgetSankey: false,

  // Budget
  activeBudgetType: '',
  activeBudgetLocation: '',
  budgetChartType: 'area',
  budgetVolumeUnit: 'af',
  budgetRateUnit: 'per_month',
  budgetAreaUnit: 'acres',
  budgetLengthUnit: 'feet',
  budgetTimeAgg: 'monthly',
  showBudgetGlossary: false,
  budgetAnalysisMode: 'timeseries',

  // Observations
  observations: [],

  // Actions
  setModelInfo: (info) =>
    set({
      modelInfo: info,
      visibleLayers: new Array(info.n_layers).fill(true),
    }),
  setBounds: (bounds) => set({ bounds }),
  setProperties: (props) => set({ properties: props }),
  setLoading: (loading) => set({ isLoading: loading }),
  setError: (error) => set({ error }),
  setMeshLoaded: (loaded) => set({ meshLoaded: loaded }),
  setActiveProperty: (prop) => set({ activeProperty: prop }),
  setActiveLayer: (layer) => set({ activeLayer: layer }),
  setOpacity: (opacity) => set({ opacity }),
  setShowEdges: (show) => set({ showEdges: show }),
  setZExaggeration: (factor) => set({ zExaggeration: factor }),
  setShowMesh: (show) => set({ showMesh: show }),
  setShowStreams: (show) => set({ showStreams: show }),
  setShowCrossSection: (show) => set({ showCrossSection: show }),
  setSliceAngle: (angle) => set({ sliceAngle: angle }),
  setSlicePosition: (pos) => set({ slicePosition: pos }),
  setLayerVisible: (layer, visible) =>
    set((state) => {
      const next = [...state.visibleLayers];
      next[layer] = visible;
      return { visibleLayers: next };
    }),
  setAllLayersVisible: (visible) =>
    set((state) => ({
      visibleLayers: state.visibleLayers.map(() => visible),
    })),

  // Animation speed
  setAnimationSpeed: (speed) => set({ animationSpeed: speed }),

  // Tab
  setActiveTab: (tab) => set({ activeTab: tab }),

  // Results Map
  setResultsInfo: (info) => set({ resultsInfo: info }),
  setHeadTimestep: (ts) => set({ headTimestep: ts }),
  setHeadTimes: (times) => set({ headTimes: times }),
  setHeadLayer: (layer) => set({ headLayer: layer }),
  setSelectedLocation: (loc) => set({ selectedLocation: loc }),
  setSelectedHydrograph: (data) => set({ selectedHydrograph: data }),
  setShowGWLocations: (show) => set({ showGWLocations: show }),
  setShowStreamLocations: (show) => set({ showStreamLocations: show }),
  setShowSubsidenceLocations: (show) => set({ showSubsidenceLocations: show }),
  setIsAnimating: (a) => set({ isAnimating: a }),

  // Head global range
  setHeadGlobalRange: (min, max) => set({ headGlobalMin: min, headGlobalMax: max }),

  // Map overlays
  setShowSubregions: (show) => set({ showSubregions: show }),
  setShowStreamsOnMap: (show) => set({ showStreamsOnMap: show }),
  setShowWells: (show) => set({ showWells: show }),
  setMapColorProperty: (prop) => set({ mapColorProperty: prop }),

  // Element detail (mutual exclusion with watershed detail)
  setSelectedElement: (id) => set(
    id !== null
      ? { selectedElement: id, selectedWatershedId: null, selectedWatershedDetail: null }
      : { selectedElement: id },
  ),
  setElementDetail: (detail) => set(
    detail !== null
      ? { elementDetail: detail, selectedWatershedId: null, selectedWatershedDetail: null }
      : { elementDetail: detail },
  ),

  // Phase 2 overlays
  setShowLakes: (show) => set({ showLakes: show }),
  setShowBoundaryConditions: (show) => set({ showBoundaryConditions: show }),
  setSelectedLakeRating: (data) => set({ selectedLakeRating: data }),

  // Phase 2 head diff
  setHeadDiffMode: (on) => set({ headDiffMode: on }),
  setHeadDiffTimestepA: (ts) => set({ headDiffTimestepA: ts }),
  setHeadDiffTimestepB: (ts) => set({ headDiffTimestepB: ts }),

  // Phase 2 cross-section
  setCrossSectionMode: (on) => set({ crossSectionMode: on }),
  setCrossSectionPoints: (pts) => set({ crossSectionPoints: pts }),
  setCrossSectionData: (data) => set({ crossSectionData: data }),

  // GW Nodes
  setShowNodes: (show) => set({ showNodes: show }),

  // Basemap
  setSelectedBasemap: (key) => set({ selectedBasemap: key }),

  // Phase 3 overlays
  setShowSmallWatersheds: (show) => set({ showSmallWatersheds: show }),
  setShowDiversions: (show) => set({ showDiversions: show }),

  // Watershed detail (mutual exclusion with element detail)
  setSelectedWatershedId: (id) => set(
    id !== null
      ? { selectedWatershedId: id, selectedElement: null, elementDetail: null }
      : { selectedWatershedId: id },
  ),
  setSelectedWatershedDetail: (detail) => set(
    detail !== null
      ? { selectedWatershedDetail: detail, selectedElement: null, elementDetail: null }
      : { selectedWatershedDetail: detail },
  ),

  // Diversion detail
  setSelectedDiversionId: (id) => set({ selectedDiversionId: id }),
  setDiversionDetail: (detail) => set({ diversionDetail: detail }),
  setDiversionListOpen: (open) => set({ diversionListOpen: open }),

  // Phase 3 comparison
  setCompareMode: (on) => set({ compareMode: on }),
  setComparedLocationIds: (ids) => set({ comparedLocationIds: ids }),
  addComparedLocationId: (id) =>
    set((state) => ({
      comparedLocationIds: state.comparedLocationIds.includes(id)
        ? state.comparedLocationIds
        : [...state.comparedLocationIds, id],
    })),
  removeComparedLocationId: (id) =>
    set((state) => ({
      comparedLocationIds: state.comparedLocationIds.filter((x) => x !== id),
    })),

  // Phase 3 stream reach profile
  setSelectedReachProfile: (data) => set({ selectedReachProfile: data }),

  // Phase 3 budget sankey
  setShowBudgetSankey: (show) => set({ showBudgetSankey: show }),

  // Budget
  setActiveBudgetType: (t) => set({ activeBudgetType: t }),
  setActiveBudgetLocation: (l) => set({ activeBudgetLocation: l }),
  setBudgetChartType: (t) => set({ budgetChartType: t }),
  setBudgetVolumeUnit: (u) => set({ budgetVolumeUnit: u }),
  setBudgetRateUnit: (u) => set({ budgetRateUnit: u }),
  setBudgetAreaUnit: (u) => set({ budgetAreaUnit: u }),
  setBudgetLengthUnit: (u) => set({ budgetLengthUnit: u }),
  setBudgetTimeAgg: (a) => set({ budgetTimeAgg: a }),
  setShowBudgetGlossary: (show) => set({ showBudgetGlossary: show }),
  setBudgetAnalysisMode: (mode) => set({ budgetAnalysisMode: mode }),

  // Observations
  setObservations: (obs) => set({ observations: obs }),
}));
