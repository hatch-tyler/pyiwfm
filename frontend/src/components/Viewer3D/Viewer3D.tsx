/**
 * 3D mesh viewer component using vtk.js.
 *
 * Loads each layer as a separate actor with distinct colors.
 * Supports cross-section slice rendering.
 */

import { useEffect, useRef, useState, useCallback } from 'react';
import Box from '@mui/material/Box';
import CircularProgress from '@mui/material/CircularProgress';
import Typography from '@mui/material/Typography';

// vtk.js imports
import '@kitware/vtk.js/Rendering/Profiles/Geometry';
import vtkFullScreenRenderWindow from '@kitware/vtk.js/Rendering/Misc/FullScreenRenderWindow';
import vtkActor from '@kitware/vtk.js/Rendering/Core/Actor';
import vtkMapper from '@kitware/vtk.js/Rendering/Core/Mapper';
import vtkCellArray from '@kitware/vtk.js/Common/Core/CellArray';
import vtkPoints from '@kitware/vtk.js/Common/Core/Points';
import vtkPolyData from '@kitware/vtk.js/Common/DataModel/PolyData';
import { useViewerStore } from '../../stores/viewerStore';
import {
  fetchMeshLayer,
  fetchModelInfo,
  fetchSliceJson,
  fetchStreams,
} from '../../api/client';
import { LAYER_COLORS } from '../../constants/colors';

/* eslint-disable @typescript-eslint/no-explicit-any */
interface ViewerRefs {
  fullScreenRenderer: any;
  renderer: any;
  renderWindow: any;
  layerActors: Map<number, any>;  // layer index (0-based) -> actor
  streamActor: any;
  sliceActors: any[];  // multiple actors for per-layer slice coloring
}

export default function Viewer3D() {
  const containerRef = useRef<HTMLDivElement>(null);
  const vtkRefs = useRef<ViewerRefs>({
    fullScreenRenderer: null,
    renderer: null,
    renderWindow: null,
    layerActors: new Map(),
    streamActor: null,
    sliceActors: [],
  });

  const [loadingMesh, setLoadingMesh] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const sliceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const {
    setMeshLoaded,
    modelInfo,
    opacity,
    showEdges,
    zExaggeration,
    showMesh,
    showStreams,
    showCrossSection,
    sliceAngle,
    slicePosition,
    visibleLayers,
  } = useViewerStore();

  // Helper: create a PolyData actor from flat mesh arrays with a solid color
  const createPolyDataActor = useCallback(
    (
      points: number[],
      polys: number[],
      color: [number, number, number],
    ): any => {
      const pts = vtkPoints.newInstance();
      pts.setData(new Float32Array(points), 3);

      const cells = vtkCellArray.newInstance();
      cells.setData(new Uint32Array(polys));

      const polyData = vtkPolyData.newInstance();
      polyData.setPoints(pts);
      polyData.setPolys(cells);

      const mapper = vtkMapper.newInstance();
      mapper.setInputData(polyData);
      mapper.setScalarVisibility(false);

      const actor = vtkActor.newInstance();
      actor.setMapper(mapper);
      actor.getProperty().setColor(...color);
      actor.getProperty().setEdgeVisibility(false);

      return actor;
    },
    [],
  );

  // Load all layers in parallel
  const loadMesh = useCallback(async () => {
    try {
      setLoadingMesh(true);
      setLoadError(null);

      // We need to know how many layers to fetch
      const info = await fetchModelInfo();
      const nLayers = info.n_layers;

      // Fetch all layers in parallel
      const promises = Array.from({ length: nLayers }, (_, i) =>
        fetchMeshLayer(i + 1),
      );
      const layerDataArray = await Promise.all(promises);

      const renderer = vtkRefs.current.renderer;
      if (!renderer) return;

      // Create an actor for each layer
      layerDataArray.forEach((meshData, i) => {
        if (meshData.n_cells === 0) return;

        const color = LAYER_COLORS[i % LAYER_COLORS.length];
        const actor = createPolyDataActor(
          meshData.points,
          meshData.polys,
          color,
        );

        vtkRefs.current.layerActors.set(i, actor);
        renderer.addActor(actor);
      });

      renderer.resetCamera();
      vtkRefs.current.renderWindow?.render();

      setMeshLoaded(true);
      setLoadingMesh(false);
    } catch (error) {
      console.error('Failed to load mesh:', error);
      setLoadError(
        error instanceof Error ? error.message : 'Failed to load mesh',
      );
      setLoadingMesh(false);
    }
  }, [setMeshLoaded, createPolyDataActor]);

  // Initialize vtk.js renderer
  useEffect(() => {
    if (!containerRef.current) return;

    const fullScreenRenderer = vtkFullScreenRenderWindow.newInstance({
      container: containerRef.current,
      background: [0.9, 0.9, 0.9],
    });

    const renderer = fullScreenRenderer.getRenderer();
    const renderWindow = fullScreenRenderer.getRenderWindow();

    vtkRefs.current.fullScreenRenderer = fullScreenRenderer;
    vtkRefs.current.renderer = renderer;
    vtkRefs.current.renderWindow = renderWindow;

    loadMesh();

    return () => {
      if (vtkRefs.current.fullScreenRenderer) {
        vtkRefs.current.fullScreenRenderer.delete();
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Update layer visibility and opacity
  // When cross-section is active, reduce mesh opacity so the slice plane is visible
  useEffect(() => {
    const meshOpacity = showCrossSection ? Math.min(opacity, 0.3) : opacity;
    vtkRefs.current.layerActors.forEach((actor, i) => {
      const visible = showMesh && (visibleLayers[i] ?? true);
      actor.setVisibility(visible);
      actor.getProperty().setOpacity(meshOpacity);
    });
    vtkRefs.current.renderWindow?.render();
  }, [visibleLayers, showMesh, opacity, showCrossSection]);

  // Update edge visibility on all layer actors
  useEffect(() => {
    vtkRefs.current.layerActors.forEach((actor) => {
      actor.getProperty().setEdgeVisibility(showEdges);
      if (showEdges) {
        actor.getProperty().setEdgeColor(0.2, 0.2, 0.2);
      }
    });
    vtkRefs.current.renderWindow?.render();
  }, [showEdges]);

  // Update Z exaggeration on all actors
  useEffect(() => {
    vtkRefs.current.layerActors.forEach((actor) => {
      actor.setScale(1, 1, zExaggeration);
    });
    for (const sa of vtkRefs.current.sliceActors) {
      sa.setScale(1, 1, zExaggeration);
    }
    if (vtkRefs.current.streamActor) {
      vtkRefs.current.streamActor.setScale(1, 1, zExaggeration);
    }
    vtkRefs.current.renderer?.resetCamera();
    vtkRefs.current.renderWindow?.render();
  }, [zExaggeration]);

  // Helper to remove current slice actors
  const removeSliceActors = useCallback(() => {
    const renderer = vtkRefs.current.renderer;
    if (!renderer) return;
    for (const sa of vtkRefs.current.sliceActors) {
      renderer.removeActor(sa);
    }
    vtkRefs.current.sliceActors = [];
  }, []);

  // Cross-section slice effect (debounced)
  useEffect(() => {
    if (sliceTimerRef.current) {
      clearTimeout(sliceTimerRef.current);
    }

    if (!showCrossSection) {
      removeSliceActors();
      vtkRefs.current.renderWindow?.render();
      return;
    }

    sliceTimerRef.current = setTimeout(async () => {
      try {
        const data = await fetchSliceJson(sliceAngle, slicePosition);
        if (data.n_cells === 0) return;

        const renderer = vtkRefs.current.renderer;
        if (!renderer) return;

        // Remove old slice actors
        removeSliceActors();

        // Group cells by layer so each gets its distinct LAYER_COLOR
        const layerCells: Map<number, number[]> = new Map();
        let idx = 0;
        const polys = data.polys;
        let cellId = 0;
        while (idx < polys.length) {
          const nVerts = polys[idx];
          const layerNum = data.layer[cellId] ?? 1;
          if (!layerCells.has(layerNum)) {
            layerCells.set(layerNum, []);
          }
          for (let k = 0; k <= nVerts; k++) {
            layerCells.get(layerNum)!.push(polys[idx + k]);
          }
          idx += nVerts + 1;
          cellId++;
        }

        const newActors: any[] = [];
        for (const [layerNum, cellPolys] of layerCells.entries()) {
          const colorIdx = ((layerNum - 1) % LAYER_COLORS.length + LAYER_COLORS.length) % LAYER_COLORS.length;
          const color = LAYER_COLORS[colorIdx];

          const actor = createPolyDataActor(data.points, cellPolys, color);
          actor.getProperty().setOpacity(0.95);
          actor.getProperty().setEdgeVisibility(true);
          actor.getProperty().setEdgeColor(0.15, 0.15, 0.15);
          actor.setScale(1, 1, zExaggeration);

          renderer.addActor(actor);
          newActors.push(actor);
        }

        vtkRefs.current.sliceActors = newActors;
        vtkRefs.current.renderWindow?.render();
      } catch (error) {
        console.error('Failed to load slice:', error);
      }
    }, 300);

    return () => {
      if (sliceTimerRef.current) {
        clearTimeout(sliceTimerRef.current);
      }
    };
  }, [showCrossSection, sliceAngle, slicePosition, zExaggeration, removeSliceActors]);

  // Load and toggle streams
  useEffect(() => {
    if (!vtkRefs.current.renderer) return;

    const loadStreams = async () => {
      if (!showStreams) {
        if (vtkRefs.current.streamActor) {
          vtkRefs.current.streamActor.setVisibility(false);
          vtkRefs.current.renderWindow?.render();
        }
        return;
      }

      if (vtkRefs.current.streamActor) {
        vtkRefs.current.streamActor.setVisibility(true);
        vtkRefs.current.renderWindow?.render();
        return;
      }

      try {
        const streamData = await fetchStreams();
        if (streamData.nodes.length === 0) return;

        const streamPoints = vtkPoints.newInstance();
        const pointsArray = new Float32Array(streamData.nodes.length * 3);
        const nodeIdToIdx: Map<number, number> = new Map();

        streamData.nodes.forEach((node, idx) => {
          pointsArray[idx * 3] = node.x;
          pointsArray[idx * 3 + 1] = node.y;
          pointsArray[idx * 3 + 2] = node.z;
          nodeIdToIdx.set(node.id, idx);
        });
        streamPoints.setData(pointsArray, 3);

        // Build lines
        const lineData: number[] = [];
        streamData.reaches.forEach((reach) => {
          const lineIndices = reach
            .map((nodeId) => nodeIdToIdx.get(nodeId))
            .filter((idx): idx is number => idx !== undefined);

          if (lineIndices.length >= 2) {
            lineData.push(lineIndices.length, ...lineIndices);
          }
        });

        const lines = vtkCellArray.newInstance();
        lines.setData(new Uint32Array(lineData));

        const streamPolyData = vtkPolyData.newInstance();
        streamPolyData.setPoints(streamPoints);
        streamPolyData.setLines(lines);

        const mapper = vtkMapper.newInstance();
        mapper.setInputData(streamPolyData);

        const actor = vtkActor.newInstance();
        actor.setMapper(mapper);
        actor.getProperty().setColor(0.2, 0.4, 0.8);
        actor.getProperty().setLineWidth(3);

        vtkRefs.current.streamActor = actor;
        vtkRefs.current.renderer?.addActor(actor);
        vtkRefs.current.renderWindow?.render();
      } catch (error) {
        console.error('Failed to load streams:', error);
      }
    };

    if (modelInfo?.has_streams) {
      loadStreams();
    }
  }, [showStreams, modelInfo?.has_streams]);

  if (loadError) {
    return (
      <Box
        sx={{
          width: '100%',
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          bgcolor: '#f5f5f5',
        }}
      >
        <Typography color="error" variant="h6">
          Error Loading Mesh
        </Typography>
        <Typography color="text.secondary">{loadError}</Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ width: '100%', height: '100%', position: 'relative' }}>
      <div
        ref={containerRef}
        style={{
          width: '100%',
          height: '100%',
        }}
      />
      {loadingMesh && (
        <Box
          sx={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            gap: 2,
          }}
        >
          <CircularProgress />
          <Typography>Loading mesh...</Typography>
        </Box>
      )}
    </Box>
  );
}
