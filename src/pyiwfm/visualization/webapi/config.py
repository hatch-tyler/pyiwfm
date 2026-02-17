"""
Configuration settings for the FastAPI web viewer.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from pyiwfm.core.model import IWFMModel
    from pyiwfm.io.budget import BudgetReader
    from pyiwfm.visualization.webapi.area_loader import AreaDataManager
    from pyiwfm.visualization.webapi.head_loader import LazyHeadDataLoader
    from pyiwfm.visualization.webapi.hydrograph_reader import IWFMHydrographReader

logger = logging.getLogger(__name__)


class ViewerSettings(BaseModel):
    """Settings for the web viewer."""

    host: str = Field(default="127.0.0.1", description="Server host address")
    port: int = Field(default=8080, ge=1, le=65535, description="Server port")
    title: str = Field(default="IWFM Viewer", description="Application title")
    open_browser: bool = Field(default=True, description="Open browser on start")
    debug: bool = Field(default=False, description="Enable debug mode")
    reload: bool = Field(default=False, description="Enable auto-reload (dev mode)")

    model_config = {"extra": "forbid"}


class ModelState:
    """
    Holds the loaded model and derived data for the API.

    This is a singleton-like container that holds the model and
    precomputed meshes for efficient API responses.
    """

    def __init__(self) -> None:
        self._model: IWFMModel | None = None
        self._mesh_3d: bytes | None = None
        self._mesh_surface: bytes | None = None
        self._surface_json_data: dict | None = None
        self._bounds: tuple[float, float, float, float, float, float] | None = None
        self._pv_mesh_3d: object | None = None  # Cached PyVista UnstructuredGrid
        self._layer_surface_cache: dict[int, dict] = {}  # Per-layer surface cache

        # Stream reach boundaries from preprocessor binary
        self._stream_reach_boundaries: list[tuple[int, int, int]] | None = None

        # Diversion timeseries cache
        self._diversion_ts_data: tuple | None = None

        # Cached grid index maps (populated lazily, cleared on set_model)
        self._node_id_to_idx: dict[int, int] | None = None
        self._sorted_elem_ids: list[int] | None = None

        # Results-related state
        self._crs: str = "+proj=utm +zone=10 +datum=NAD83 +units=us-ft +no_defs"
        self._transformer: Any = None  # pyproj Transformer (lazy)
        self._geojson_cache: dict[int, dict] = {}  # layer -> GeoJSON in WGS84
        self._head_loader: LazyHeadDataLoader | None = None
        self._gw_hydrograph_reader: IWFMHydrographReader | None = None
        self._stream_hydrograph_reader: IWFMHydrographReader | None = None
        self._subsidence_reader: IWFMHydrographReader | None = None
        self._budget_readers: dict[str, BudgetReader] = {}
        self._results_dir: Path | None = None
        self._area_manager: AreaDataManager | None = None
        self._observations: dict[str, dict] = {}  # id -> observation data

    @property
    def model(self) -> IWFMModel | None:
        """Get the loaded model."""
        return self._model

    @property
    def is_loaded(self) -> bool:
        """Check if a model is loaded."""
        return self._model is not None

    def set_model(
        self,
        model: IWFMModel,
        crs: str = "+proj=utm +zone=10 +datum=NAD83 +units=us-ft +no_defs",
    ) -> None:
        """Set the model and reset caches."""
        self._model = model
        self._mesh_3d = None
        self._mesh_surface = None
        self._surface_json_data = None
        self._bounds = None
        self._pv_mesh_3d = None
        self._layer_surface_cache = {}

        # Results state
        self._crs = crs
        self._transformer = None
        self._geojson_cache = {}
        self._head_loader = None
        self._gw_hydrograph_reader = None
        self._stream_hydrograph_reader = None
        self._subsidence_reader = None
        self._budget_readers = {}
        self._area_manager = None
        self._observations = {}
        self._stream_reach_boundaries = None
        self._diversion_ts_data = None
        self._node_id_to_idx = None
        self._sorted_elem_ids = None

        # Determine results directory from model metadata
        sim_file = model.metadata.get("simulation_file")
        if sim_file:
            self._results_dir = Path(sim_file).parent
        else:
            self._results_dir = None

    def get_mesh_3d(self) -> bytes:
        """Get the 3D mesh as VTU bytes, computing if needed."""
        if self._mesh_3d is None:
            self._mesh_3d = self._compute_mesh_3d()
        return self._mesh_3d

    def get_mesh_surface(self) -> bytes:
        """Get the surface mesh as VTU bytes, computing if needed."""
        if self._mesh_surface is None:
            self._mesh_surface = self._compute_mesh_surface()
        return self._mesh_surface

    def get_bounds(self) -> tuple[float, float, float, float, float, float]:
        """Get model bounding box."""
        if self._bounds is None:
            self._bounds = self._compute_bounds()
        return self._bounds

    def _compute_mesh_3d(self) -> bytes:
        """Compute 3D mesh as VTU XML string."""
        import vtk

        if self._model is None:
            raise ValueError("No model loaded")

        from pyiwfm.visualization.vtk_export import VTKExporter

        grid = self._model.grid
        if grid is None:
            raise ValueError("No grid loaded")

        exporter = VTKExporter(
            grid=grid,
            stratigraphy=self._model.stratigraphy,
        )
        vtk_grid = exporter.create_3d_mesh()

        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetWriteToOutputString(True)
        writer.SetInputData(vtk_grid)
        writer.Write()

        return cast(bytes, writer.GetOutputString().encode("utf-8"))

    def _compute_mesh_surface(self) -> bytes:
        """Compute surface mesh as VTU XML string."""
        import vtk

        if self._model is None:
            raise ValueError("No model loaded")

        from pyiwfm.visualization.vtk_export import VTKExporter

        grid = self._model.grid
        if grid is None:
            raise ValueError("No grid loaded")

        exporter = VTKExporter(grid=grid)
        vtk_grid = exporter.create_2d_mesh()

        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetWriteToOutputString(True)
        writer.SetInputData(vtk_grid)
        writer.Write()

        return cast(bytes, writer.GetOutputString().encode("utf-8"))

    def get_pyvista_3d(self) -> object:
        """Get the cached PyVista 3D mesh, computing if needed."""
        if self._pv_mesh_3d is None:
            if self._model is None:
                raise ValueError("No model loaded")
            strat = self._model.stratigraphy
            if strat is None:
                raise ValueError("3D mesh requires stratigraphy")
            from pyiwfm.visualization.vtk_export import VTKExporter

            grid = self._model.grid
            if grid is None:
                raise ValueError("No grid loaded")
            exporter = VTKExporter(grid=grid, stratigraphy=strat)
            self._pv_mesh_3d = exporter.to_pyvista_3d()
        return self._pv_mesh_3d

    def get_surface_json(self, layer: int = 0) -> dict:
        """Get the extracted surface mesh as flat JSON-serializable dict.

        Parameters
        ----------
        layer : int
            0 = all layers (default), 1..N = specific layer only.
        """
        if layer in self._layer_surface_cache:
            return self._layer_surface_cache[layer]

        # For layer=0, also check legacy cache
        if layer == 0 and self._surface_json_data is not None:
            return self._surface_json_data

        data = self._compute_surface_json(layer)
        self._layer_surface_cache[layer] = data
        if layer == 0:
            self._surface_json_data = data
        return data

    def _compute_surface_json(self, layer: int = 0) -> dict:
        """Extract the outer surface of the 3D mesh and return flat arrays.

        Parameters
        ----------
        layer : int
            0 = all layers, 1..N = specific layer only.
        """
        import numpy as np

        pv_mesh = self.get_pyvista_3d()

        if layer > 0:
            # Filter to specific layer using threshold
            filtered = pv_mesh.threshold(value=[layer, layer], scalars="layer")  # type: ignore[attr-defined]
            surface = filtered.extract_surface()  # type: ignore[attr-defined]
        else:
            surface = pv_mesh.extract_surface()  # type: ignore[attr-defined]

        # Flat points array: [x0, y0, z0, x1, y1, z1, ...]
        points_flat = surface.points.astype(np.float32).ravel().tolist()

        # Flat polys array in VTK format: [nV, v0, v1, ..., nV, v0, v1, ...]
        polys_flat = surface.faces.tolist()

        # Layer cell data (mapped from volumetric to surface)
        if "layer" in surface.cell_data:
            layer_data = surface.cell_data["layer"].tolist()
        else:
            layer_data = [layer if layer > 0 else 1] * surface.n_cells

        # For single-layer requests, n_layers is the layer number itself
        # For all layers, it's the max layer value
        if layer > 0:
            n_layers = layer
        else:
            n_layers = int(max(layer_data)) if layer_data else 1

        return {
            "n_points": surface.n_points,
            "n_cells": surface.n_cells,
            "n_layers": n_layers,
            "points": points_flat,
            "polys": polys_flat,
            "layer": layer_data,
        }

    def get_slice_json(self, angle: float, position: float) -> dict:
        """Get a cross-section slice as flat JSON-serializable dict.

        Parameters
        ----------
        angle : float
            Angle in degrees from a north-south face.
            0° = north-south cross-section (normal points east),
            90° = east-west cross-section (normal points north).
        position : float
            Normalized position (0-1) along the slice normal.
        """
        import math

        import numpy as np

        from pyiwfm.visualization.webapi.slicing import SlicingController

        pv_mesh = self.get_pyvista_3d()
        slicer = SlicingController(pv_mesh)  # type: ignore[arg-type]

        # Convert angle to normal vector.
        # 0° = N-S face → normal (1,0,0) (east)
        # 90° = E-W face → normal (0,1,0) (north)
        rad = math.radians(angle)
        normal = (math.cos(rad), math.sin(rad), 0.0)

        # Convert normalized position to world-space origin
        origin = slicer.normalized_to_position_along(normal, position)

        slice_mesh = slicer.slice_arbitrary(normal=normal, origin=origin)

        if slice_mesh.n_cells == 0:
            return {
                "n_points": 0,
                "n_cells": 0,
                "n_layers": 0,
                "points": [],
                "polys": [],
                "layer": [],
            }

        # The slice is already a PolyData
        points_flat = slice_mesh.points.astype(np.float32).ravel().tolist()
        polys_flat = slice_mesh.faces.tolist()

        if "layer" in slice_mesh.cell_data:
            layer_data = slice_mesh.cell_data["layer"].tolist()
        else:
            layer_data = [1] * slice_mesh.n_cells

        n_layers = int(max(layer_data)) if layer_data else 1

        return {
            "n_points": slice_mesh.n_points,
            "n_cells": slice_mesh.n_cells,
            "n_layers": n_layers,
            "points": points_flat,
            "polys": polys_flat,
            "layer": layer_data,
        }

    # ------------------------------------------------------------------
    # Stream reach boundaries from preprocessor binary
    # ------------------------------------------------------------------

    def get_stream_reach_boundaries(self) -> list[tuple[int, int, int]] | None:
        """Get reach boundaries from preprocessor binary or streams spec text file.

        Tries two sources in order:
        1. Preprocessor binary output (compiled format)
        2. Preprocessor streams spec text file (ASCII input format)
        """
        if self._stream_reach_boundaries is not None:
            return self._stream_reach_boundaries

        if self._model is None:
            return None

        source_files = getattr(self._model, "source_files", {}) or {}

        # Strategy 1: Try preprocessor binary
        binary_path = source_files.get("binary_preprocessor")
        if binary_path is not None and Path(binary_path).exists():
            try:
                from pyiwfm.io.preprocessor_binary import PreprocessorBinaryReader

                reader = PreprocessorBinaryReader()
                data = reader.read(binary_path)
                if data.streams and data.streams.n_reaches > 0:
                    boundaries: list[tuple[int, int, int]] = []
                    for i in range(data.streams.n_reaches):
                        boundaries.append(
                            (
                                int(data.streams.reach_ids[i]),
                                int(data.streams.reach_upstream_nodes[i]),
                                int(data.streams.reach_downstream_nodes[i]),
                            )
                        )
                    self._stream_reach_boundaries = boundaries
                    logger.info(
                        "Loaded %d reach boundaries from preprocessor binary",
                        len(boundaries),
                    )
                    return boundaries
            except Exception as e:
                logger.debug("Could not read preprocessor binary for reach boundaries: %s", e)

        # Strategy 2: Try streams spec text file (preprocessor input)
        spec_path = source_files.get("streams_spec")
        if spec_path is not None and Path(spec_path).exists():
            try:
                from pyiwfm.io.streams import StreamSpecReader

                spec_reader = StreamSpecReader()
                _nr, _nrt, reach_specs = spec_reader.read(spec_path)
                if reach_specs:
                    boundaries = []
                    for rs in reach_specs:
                        if rs.node_ids:
                            boundaries.append(
                                (
                                    rs.id,
                                    rs.node_ids[0],
                                    rs.node_ids[-1],
                                )
                            )
                    if boundaries:
                        self._stream_reach_boundaries = boundaries
                        logger.info(
                            "Loaded %d reach boundaries from streams spec file",
                            len(boundaries),
                        )
                        return boundaries
            except Exception as e:
                logger.debug("Could not read streams spec for reach boundaries: %s", e)

        # Strategy 3: Try reading preprocessor main to find streams spec
        pp_path = source_files.get("preprocessor_main")
        if pp_path is not None and Path(pp_path).exists():
            try:
                from pyiwfm.io.preprocessor import read_preprocessor_main
                from pyiwfm.io.streams import StreamSpecReader

                pp_config = read_preprocessor_main(pp_path)
                if pp_config.streams_file and pp_config.streams_file.exists():
                    spec_reader = StreamSpecReader()
                    _nr, _nrt, reach_specs = spec_reader.read(pp_config.streams_file)
                    if reach_specs:
                        boundaries = []
                        for rs in reach_specs:
                            if rs.node_ids:
                                boundaries.append(
                                    (
                                        rs.id,
                                        rs.node_ids[0],
                                        rs.node_ids[-1],
                                    )
                                )
                        if boundaries:
                            self._stream_reach_boundaries = boundaries
                            logger.info(
                                "Loaded %d reach boundaries from preprocessor main -> streams spec",
                                len(boundaries),
                            )
                            return boundaries
            except Exception as e:
                logger.debug("Could not read preprocessor main for reach boundaries: %s", e)

        return None

    # ------------------------------------------------------------------
    # Diversion timeseries
    # ------------------------------------------------------------------

    def get_diversion_timeseries(self) -> tuple | None:
        """Get cached diversion timeseries data (times, values, metadata).

        Reads the stream_diversion_ts file once via UnifiedTimeSeriesReader
        and caches the result.
        """
        if self._diversion_ts_data is not None:
            return self._diversion_ts_data

        if self._model is None:
            return None

        source_files = getattr(self._model, "source_files", {}) or {}
        ts_path_raw = source_files.get("stream_diversion_ts")
        if not ts_path_raw:
            return None

        ts_path = Path(ts_path_raw)
        if not ts_path.is_absolute() and self._results_dir:
            ts_path = self._results_dir / ts_path
        if not ts_path.exists():
            logger.warning("Diversion timeseries file not found: %s", ts_path)
            return None

        try:
            from pyiwfm.io.timeseries import UnifiedTimeSeriesReader

            reader = UnifiedTimeSeriesReader()
            times, values, metadata = reader.read_file(ts_path)
            self._diversion_ts_data = (times, values, metadata)
            logger.info(
                "Diversion timeseries loaded: %d timesteps, %d columns",
                len(times),
                values.shape[1] if values.ndim > 1 else 1,
            )
            return self._diversion_ts_data
        except Exception as e:
            logger.error("Failed to load diversion timeseries: %s", e)
            return None

    # ------------------------------------------------------------------
    # Coordinate reprojection
    # ------------------------------------------------------------------

    def _get_transformer(self) -> Any:
        """Get or create the pyproj Transformer for CRS reprojection."""
        if self._transformer is None:
            try:
                from pyproj import Transformer
            except ImportError:
                logger.warning("pyproj not installed; coordinates will not be reprojected")
                return None

            self._transformer = Transformer.from_crs(self._crs, "EPSG:4326", always_xy=True)
        return self._transformer

    def reproject_coords(self, x: float, y: float) -> tuple[float, float]:
        """Reproject model coordinates to WGS84 (lng, lat)."""
        transformer = self._get_transformer()
        if transformer is None:
            return (x, y)
        lng, lat = transformer.transform(x, y)
        return (lng, lat)

    # ------------------------------------------------------------------
    # GeoJSON mesh (for deck.gl 2D map)
    # ------------------------------------------------------------------

    def get_mesh_geojson(self, layer: int = 1) -> dict:
        """Get the mesh as GeoJSON FeatureCollection in WGS84.

        Parameters
        ----------
        layer : int
            Layer number (1-based). Used for property association only;
            the 2D geometry is the same for all layers (plan view).
        """
        if layer in self._geojson_cache:
            return self._geojson_cache[layer]

        geojson = self._compute_mesh_geojson(layer)
        self._geojson_cache[layer] = geojson
        return geojson

    def _compute_mesh_geojson(self, layer: int) -> dict:
        """Build GeoJSON FeatureCollection from the mesh plan view."""
        if self._model is None or self._model.grid is None:
            return {"type": "FeatureCollection", "features": []}

        grid = self._model.grid
        features: list[dict] = []

        for elem in grid.iter_elements():
            ring: list[list[float]] = []
            for nid in elem.vertices:
                node = grid.nodes[nid]
                lng, lat = self.reproject_coords(node.x, node.y)
                ring.append([lng, lat])
            # Close the ring
            if ring:
                ring.append(ring[0])

            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [ring]},
                    "properties": {"element_id": elem.id, "layer": layer},
                }
            )

        return {"type": "FeatureCollection", "features": features}

    # ------------------------------------------------------------------
    # Head data access
    # ------------------------------------------------------------------

    def _get_n_gw_layers(self) -> int:
        """Get the number of groundwater layers from the loaded model."""
        if self._model is None:
            return 1
        # model.n_layers reads from stratigraphy (authoritative)
        n = getattr(self._model, "n_layers", 0)
        if n and n > 0:
            return n
        # Fallback to metadata set by GW reader
        n = self._model.metadata.get("gw_aquifer_n_layers", 1)
        return n if n and n > 0 else 1

    def get_head_loader(self) -> LazyHeadDataLoader | None:
        """Get the lazy head data loader, initializing if needed."""
        if self._head_loader is not None:
            return self._head_loader

        if self._model is None:
            return None

        head_file = self._model.metadata.get("gw_head_all_file")
        if not head_file:
            return None

        head_path = Path(head_file)
        if not head_path.is_absolute() and self._results_dir:
            head_path = self._results_dir / head_path
        if not head_path.exists():
            logger.warning("Head file not found: %s", head_path)
            return None

        n_layers = self._get_n_gw_layers()

        try:
            from pyiwfm.visualization.webapi.head_loader import LazyHeadDataLoader

            suffix = head_path.suffix.lower()
            if suffix in (".hdf", ".h5", ".he5", ".hdf5"):
                loader = LazyHeadDataLoader(head_path)
                # Validate layer count — a pyiwfm-converted file may have been
                # created with the wrong n_layers (e.g. 1 instead of 4).  If so,
                # look for the companion .out text file and re-convert.
                if loader.n_frames > 0 and loader.shape[1] != n_layers:
                    logger.warning(
                        "HDF head file has %d layer(s) but model has %d; "
                        "attempting re-conversion from text source",
                        loader.shape[1],
                        n_layers,
                    )
                    loader = self._reconvert_head_hdf(head_path, n_layers)
                self._head_loader = loader
            elif suffix in (".out", ".txt", ".dat"):
                # Convert text file to HDF on-the-fly, cache alongside the original
                hdf_cache = head_path.with_suffix(".head_cache.hdf")
                if not hdf_cache.exists() or hdf_cache.stat().st_mtime < head_path.stat().st_mtime:
                    from pyiwfm.io.head_all_converter import convert_headall_to_hdf

                    convert_headall_to_hdf(head_path, hdf_cache, n_layers=n_layers)
                self._head_loader = LazyHeadDataLoader(hdf_cache)
            else:
                # Unknown format, try loading directly as HDF
                self._head_loader = LazyHeadDataLoader(head_path)

            if self._head_loader is not None:
                logger.info(
                    "Head loader initialized: %d timesteps, %d nodes, %d layers",
                    self._head_loader.n_frames,
                    self._head_loader.shape[0],
                    self._head_loader.shape[1],
                )
        except Exception as e:
            logger.error("Failed to initialize head loader: %s", e)
            return None

        return self._head_loader

    def _reconvert_head_hdf(self, hdf_path: Path, n_layers: int) -> LazyHeadDataLoader:
        """Re-convert a head HDF that has the wrong layer count.

        Looks for a companion ``.out`` text file (same stem) and converts it
        with the correct ``n_layers``, overwriting the bad HDF.  Falls back
        to the existing file if no text source is found.
        """
        from pyiwfm.visualization.webapi.head_loader import LazyHeadDataLoader

        # Look for companion text file
        text_candidates = [
            hdf_path.with_suffix(".out"),
            hdf_path.with_suffix(".dat"),
            hdf_path.with_suffix(".txt"),
        ]
        text_source = next((p for p in text_candidates if p.exists()), None)

        if text_source is None:
            logger.warning("No text source found for re-conversion; using existing HDF as-is")
            return LazyHeadDataLoader(hdf_path)

        from pyiwfm.io.head_all_converter import convert_headall_to_hdf

        logger.info(
            "Re-converting %s -> %s with n_layers=%d",
            text_source.name,
            hdf_path.name,
            n_layers,
        )
        convert_headall_to_hdf(text_source, hdf_path, n_layers=n_layers)
        return LazyHeadDataLoader(hdf_path)

    # ------------------------------------------------------------------
    # Area data access (land-use area HDF5)
    # ------------------------------------------------------------------

    def get_area_manager(self) -> AreaDataManager | None:
        """Get the area data manager, initializing and converting if needed."""
        if self._area_manager is not None:
            return self._area_manager

        if self._model is None or self._model.rootzone is None:
            return None

        rz = self._model.rootzone
        has_any = any(
            getattr(rz, a, None) is not None
            for a in (
                "nonponded_area_file",
                "ponded_area_file",
                "urban_area_file",
                "native_area_file",
            )
        )
        if not has_any:
            return None

        try:
            from pyiwfm.visualization.webapi.area_loader import AreaDataManager

            mgr = AreaDataManager()
            cache_dir = self._results_dir or Path(".")
            mgr.load_from_rootzone(rz, cache_dir)
            self._area_manager = mgr
            logger.info("Area manager initialized: %d timesteps", mgr.n_timesteps)
        except Exception as e:
            logger.error("Failed to initialize area manager: %s", e)
            return None

        return self._area_manager

    # ------------------------------------------------------------------
    # Hydrograph readers (IWFM text output files)
    # ------------------------------------------------------------------

    def _get_or_convert_hydrograph(self, path: Path) -> IWFMHydrographReader | None:
        """Load a hydrograph file, auto-converting text to HDF5 cache.

        Mirrors the head loader auto-conversion pattern:
        - HDF5 files → ``LazyHydrographDataLoader`` directly
        - Text files → convert to ``{name}.hydrograph_cache.hdf``, then lazy-load
        - Falls back to ``IWFMHydrographReader`` if conversion fails
        """
        suffix = path.suffix.lower()

        if suffix in (".hdf", ".h5", ".he5", ".hdf5"):
            try:
                from pyiwfm.visualization.webapi.hydrograph_loader import (
                    LazyHydrographDataLoader,
                )

                loader = LazyHydrographDataLoader(path)
                if loader.n_timesteps > 0:
                    logger.info(
                        "Hydrograph HDF5 loaded: %d columns, %d timesteps from %s",
                        loader.n_columns,
                        loader.n_timesteps,
                        path.name,
                    )
                    return loader  # type: ignore[return-value]
            except Exception as e:
                logger.warning("Failed to load hydrograph HDF5 %s: %s", path, e)

        if suffix in (".out", ".txt", ".dat"):
            # Try auto-converting to HDF5 cache
            hdf_cache = path.parent / (path.name + ".hydrograph_cache.hdf")
            try:
                if not hdf_cache.exists() or hdf_cache.stat().st_mtime < path.stat().st_mtime:
                    from pyiwfm.io.hydrograph_converter import (
                        convert_hydrograph_to_hdf,
                    )

                    convert_hydrograph_to_hdf(path, hdf_cache)

                from pyiwfm.visualization.webapi.hydrograph_loader import (
                    LazyHydrographDataLoader,
                )

                loader = LazyHydrographDataLoader(hdf_cache)
                if loader.n_timesteps > 0:
                    logger.info(
                        "Hydrograph auto-converted: %d columns, %d timesteps from %s",
                        loader.n_columns,
                        loader.n_timesteps,
                        path.name,
                    )
                    return loader  # type: ignore[return-value]
            except Exception as e:
                logger.warning(
                    "HDF5 auto-conversion failed for %s, falling back to text reader: %s",
                    path.name,
                    e,
                )

            # Fallback: load text file directly
            try:
                from pyiwfm.visualization.webapi.hydrograph_reader import (
                    IWFMHydrographReader,
                )

                reader = IWFMHydrographReader(path)
                logger.info(
                    "Hydrograph text reader: %d columns, %d timesteps from %s",
                    reader.n_columns,
                    reader.n_timesteps,
                    path.name,
                )
                return reader
            except Exception as e:
                logger.error("Failed to load hydrograph text file %s: %s", path, e)

        return None

    def get_gw_hydrograph_reader(self) -> IWFMHydrographReader | None:
        """Get or create the GW hydrograph reader from the output file."""
        if self._gw_hydrograph_reader is not None:
            return self._gw_hydrograph_reader

        if self._model is None:
            return None

        hydrograph_file = self._model.metadata.get("gw_hydrograph_file")
        if not hydrograph_file:
            return None

        p = Path(hydrograph_file)
        if not p.is_absolute() and self._results_dir:
            p = self._results_dir / p
        if not p.exists():
            logger.warning("GW hydrograph file not found: %s", p)
            return None

        self._gw_hydrograph_reader = self._get_or_convert_hydrograph(p)
        return self._gw_hydrograph_reader

    def get_stream_hydrograph_reader(self) -> IWFMHydrographReader | None:
        """Get or create the stream hydrograph reader from the output file."""
        if self._stream_hydrograph_reader is not None:
            return self._stream_hydrograph_reader

        if self._model is None:
            return None

        hydrograph_file = self._model.metadata.get("stream_hydrograph_file")
        if not hydrograph_file:
            return None

        p = Path(hydrograph_file)
        if not p.is_absolute() and self._results_dir:
            p = self._results_dir / p
        if not p.exists():
            logger.warning("Stream hydrograph file not found: %s", p)
            return None

        self._stream_hydrograph_reader = self._get_or_convert_hydrograph(p)
        return self._stream_hydrograph_reader

    def get_subsidence_reader(self) -> IWFMHydrographReader | None:
        """Get or create the subsidence hydrograph reader from the output file."""
        if self._subsidence_reader is not None:
            return self._subsidence_reader

        if self._model is None:
            return None

        # Check subsidence_config for hydrograph_output_file
        subs_config = None
        if self._model.groundwater:
            subs_config = getattr(self._model.groundwater, "subsidence_config", None)

        if subs_config is not None:
            output_file = getattr(subs_config, "hydrograph_output_file", None)
            if output_file:
                p = Path(output_file)
                if not p.is_absolute() and self._results_dir:
                    p = self._results_dir / p
                if p.exists():
                    self._subsidence_reader = self._get_or_convert_hydrograph(p)
                    if self._subsidence_reader is not None:
                        return self._subsidence_reader

        # Fallback: scan model directory for *Subsidence*.out
        if self._results_dir:
            for pattern in ("*Subsidence*.out", "*_Subsidence.out", "*subsidence*.out"):
                matches = list(self._results_dir.glob(pattern))
                if matches:
                    self._subsidence_reader = self._get_or_convert_hydrograph(matches[0])
                    if self._subsidence_reader is not None:
                        return self._subsidence_reader

        return None

    # ------------------------------------------------------------------
    # Hydrograph locations
    # ------------------------------------------------------------------

    def get_hydrograph_locations(self) -> dict[str, list[dict]]:
        """Get all hydrograph locations reprojected to WGS84."""
        result: dict[str, list[dict]] = {"gw": [], "stream": [], "subsidence": []}

        if self._model is None:
            return result

        # GW hydrograph locations — use 1-based index as ID
        # (node_id is 0 for element-based HYDTYP=0 observations)
        if self._model.groundwater:
            for idx, loc in enumerate(self._model.groundwater.hydrograph_locations):
                lng, lat = self.reproject_coords(loc.x, loc.y)
                result["gw"].append(
                    {
                        "id": idx + 1,  # 1-based hydrograph ID
                        "lng": lng,
                        "lat": lat,
                        "name": loc.name or f"GW Hydrograph {idx + 1}",
                        "layer": loc.layer,
                    }
                )

        # Stream hydrograph locations — get coords from associated GW node
        stream_specs = self._model.metadata.get("stream_hydrograph_specs", [])
        if stream_specs and self._model.streams:
            grid = self._model.grid
            for spec in stream_specs:
                nid = spec["node_id"]
                node = self._model.streams.nodes.get(nid)
                if not node:
                    continue

                # Stream nodes often have x=0, y=0; use GW node coords
                x, y = node.x, node.y
                if (x == 0.0 and y == 0.0) and node.gw_node and grid:
                    gw_node = grid.nodes.get(node.gw_node)
                    if gw_node:
                        x, y = gw_node.x, gw_node.y

                if x == 0.0 and y == 0.0:
                    continue  # Skip if still no valid coordinates

                lng, lat = self.reproject_coords(x, y)
                result["stream"].append(
                    {
                        "id": nid,
                        "lng": lng,
                        "lat": lat,
                        "name": spec.get("name", f"Stream Node {nid}"),
                        "reach_id": getattr(node, "reach_id", 0),
                    }
                )

        # Subsidence hydrograph locations
        if self._model.groundwater:
            subs_config = getattr(self._model.groundwater, "subsidence_config", None)
            if subs_config is not None:
                grid = self._model.grid
                specs = getattr(subs_config, "hydrograph_specs", [])
                for spec in specs:
                    x, y = spec.x, spec.y
                    # Node-based specs (hydtyp=1) may have x=0, y=0;
                    # look up coordinates from the associated grid node
                    if x == 0.0 and y == 0.0 and grid:
                        node_id = getattr(spec, "node_id", 0) or getattr(spec, "gw_node", 0)
                        if node_id:
                            gw_node = grid.nodes.get(node_id)
                            if gw_node:
                                x, y = gw_node.x, gw_node.y
                    if x == 0.0 and y == 0.0:
                        continue  # Skip if still no valid coordinates
                    try:
                        lng, lat = self.reproject_coords(x, y)
                    except Exception:
                        continue
                    result["subsidence"].append(
                        {
                            "id": spec.id,
                            "lng": lng,
                            "lat": lat,
                            "name": spec.name or f"Subsidence Obs {spec.id}",
                            "layer": spec.layer,
                        }
                    )

        return result

    # ------------------------------------------------------------------
    # Budget readers
    # ------------------------------------------------------------------

    def get_available_budgets(self) -> list[str]:
        """Return list of budget types that have files available."""
        if self._model is None:
            return []

        budget_keys = {
            "gw": "gw_budget_file",
            "stream": "stream_budget_file",
            "stream_node": "stream_node_budget_file",
            "lwu": "rootzone_lwu_budget_file",
            "rootzone": "rootzone_rz_budget_file",
            "unsaturated": "unsat_zone_budget_file",
            "diversion": "stream_diversion_budget_file",
            "lake": "lake_budget_file",
            "small_watershed": "small_watershed_budget_file",
        }

        available: list[str] = []
        for btype, meta_key in budget_keys.items():
            fpath = self._model.metadata.get(meta_key)
            if fpath:
                p = Path(fpath)
                if not p.is_absolute() and self._results_dir:
                    p = self._results_dir / p
                if p.exists():
                    available.append(btype)

        return available

    def get_budget_reader(self, budget_type: str) -> BudgetReader | None:
        """Get or create a BudgetReader for the given budget type."""
        if budget_type in self._budget_readers:
            return self._budget_readers[budget_type]

        if self._model is None:
            return None

        budget_keys = {
            "gw": "gw_budget_file",
            "stream": "stream_budget_file",
            "stream_node": "stream_node_budget_file",
            "lwu": "rootzone_lwu_budget_file",
            "rootzone": "rootzone_rz_budget_file",
            "unsaturated": "unsat_zone_budget_file",
            "diversion": "stream_diversion_budget_file",
            "lake": "lake_budget_file",
            "small_watershed": "small_watershed_budget_file",
        }

        meta_key = budget_keys.get(budget_type)
        if not meta_key:
            return None

        fpath = self._model.metadata.get(meta_key)
        if not fpath:
            return None

        p = Path(fpath)
        if not p.is_absolute() and self._results_dir:
            p = self._results_dir / p
        if not p.exists():
            logger.warning("Budget file not found: %s", p)
            return None

        try:
            from pyiwfm.io.budget import BudgetReader

            reader = BudgetReader(p)
            self._budget_readers[budget_type] = reader
            logger.info("Budget reader for '%s': %s", budget_type, reader.descriptor)
            return reader
        except Exception as e:
            logger.error("Failed to load budget file %s: %s", p, e)
            return None

    # ------------------------------------------------------------------
    # Observations (session-scoped, in-memory)
    # ------------------------------------------------------------------

    def add_observation(self, obs_id: str, data: dict) -> None:
        """Store an uploaded observation dataset."""
        self._observations[obs_id] = data

    def get_observation(self, obs_id: str) -> dict | None:
        """Get observation data by ID."""
        return self._observations.get(obs_id)

    def list_observations(self) -> list[dict]:
        """List all uploaded observations."""
        return [
            {
                "id": k,
                **{
                    key: v[key]
                    for key in ("filename", "location_id", "type", "n_records")
                    if key in v
                },
            }
            for k, v in self._observations.items()
        ]

    def delete_observation(self, obs_id: str) -> bool:
        """Delete an observation by ID."""
        if obs_id in self._observations:
            del self._observations[obs_id]
            return True
        return False

    # ------------------------------------------------------------------
    # Grid index mappings (cached)
    # ------------------------------------------------------------------

    def get_node_id_to_idx(self) -> dict[int, int]:
        """Get cached node_id -> array index mapping."""
        if self._node_id_to_idx is None:
            if self._model is None or self._model.grid is None:
                return {}
            sorted_ids = sorted(self._model.grid.nodes.keys())
            self._node_id_to_idx = {nid: i for i, nid in enumerate(sorted_ids)}
        return self._node_id_to_idx

    def get_sorted_elem_ids(self) -> list[int]:
        """Get cached sorted element ID list (matching GeoJSON feature order)."""
        if self._sorted_elem_ids is None:
            if self._model is None or self._model.grid is None:
                return []
            self._sorted_elem_ids = sorted(self._model.grid.elements.keys())
        return self._sorted_elem_ids

    # ------------------------------------------------------------------
    # Results info
    # ------------------------------------------------------------------

    def get_results_info(self) -> dict:
        """Get summary of available results."""
        info: dict[str, Any] = {
            "has_results": False,
            "available_budgets": [],
            "n_head_timesteps": 0,
            "head_time_range": None,
            "n_gw_hydrographs": 0,
            "n_stream_hydrographs": 0,
        }

        if self._model is None:
            return info

        budgets = self.get_available_budgets()
        info["available_budgets"] = budgets

        loader = self.get_head_loader()
        if loader and loader.n_frames > 0:
            info["n_head_timesteps"] = loader.n_frames
            times = loader.times
            if times:
                info["head_time_range"] = {
                    "start": times[0].isoformat(),
                    "end": times[-1].isoformat(),
                }

        if self._model.groundwater:
            info["n_gw_hydrographs"] = self._model.groundwater.n_hydrograph_locations

        stream_specs = self._model.metadata.get("stream_hydrograph_specs", [])
        info["n_stream_hydrographs"] = len(stream_specs)

        gw_reader = self.get_gw_hydrograph_reader()
        if gw_reader and gw_reader.n_timesteps > 0:
            info["has_gw_hydrographs"] = True

        stream_reader = self.get_stream_hydrograph_reader()
        if stream_reader and stream_reader.n_timesteps > 0:
            info["has_stream_hydrographs"] = True

        info["has_results"] = bool(
            budgets or (loader and loader.n_frames > 0) or (gw_reader and gw_reader.n_timesteps > 0)
        )

        return info

    def _compute_bounds(self) -> tuple[float, float, float, float, float, float]:
        """Compute model bounding box."""
        if self._model is None:
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        grid = self._model.grid
        strat = self._model.stratigraphy

        if grid is None:
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        xs = [n.x for n in grid.iter_nodes()]
        ys = [n.y for n in grid.iter_nodes()]

        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)

        if strat is not None:
            import numpy as np

            zmin = float(np.min(strat.bottom_elev))
            zmax = float(np.max(strat.top_elev))
        else:
            zmin, zmax = 0.0, 0.0

        return (xmin, xmax, ymin, ymax, zmin, zmax)


# Global model state instance
model_state = ModelState()


def require_model() -> IWFMModel:
    """Return the loaded model or raise HTTPException(404).

    Use this in route handlers to narrow ``model_state.model`` from
    ``IWFMModel | None`` to ``IWFMModel``, satisfying mypy while also
    giving a clean API error when no model is loaded.
    """
    from starlette.exceptions import HTTPException

    m = model_state.model
    if m is None:
        raise HTTPException(status_code=404, detail="No model loaded")
    return m
