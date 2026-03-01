"""Mixin providing results, hydrograph, coordinate, and GeoJSON methods for ModelState."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pyiwfm.core.model import IWFMModel
    from pyiwfm.io.area_loader import AreaDataManager
    from pyiwfm.io.head_loader import LazyHeadDataLoader
    from pyiwfm.io.hydrograph_reader import IWFMHydrographReader

logger = logging.getLogger(__name__)


class ResultsStateMixin:
    """Mixin providing results, hydrograph, and coordinate methods for ModelState."""

    # -- Attributes set by ModelState.__init__ (declared for type checkers) --
    _model: IWFMModel | None
    _results_dir: Path | None
    _crs: str
    _transformer: Any
    _geojson_cache: dict[int, dict]
    _head_loader: LazyHeadDataLoader | None
    _gw_hydrograph_reader: IWFMHydrographReader | None
    _stream_hydrograph_reader: IWFMHydrographReader | None
    _subsidence_reader: IWFMHydrographReader | None
    _tile_drain_reader: IWFMHydrographReader | None
    _area_manager: AreaDataManager | None
    _hydrograph_locations_cache: dict[str, list[dict]] | None
    _stream_reach_boundaries: list[tuple[int, int, int]] | None
    _diversion_ts_data: tuple | None
    _budget_readers: dict

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
            from pyiwfm.io.head_loader import LazyHeadDataLoader

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
        from pyiwfm.io.head_loader import LazyHeadDataLoader

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
            from pyiwfm.io.area_loader import AreaDataManager

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
        - HDF5 files -> ``LazyHydrographDataLoader`` directly
        - Text files -> convert to ``{name}.hydrograph_cache.hdf``, then lazy-load
        - Falls back to ``IWFMHydrographReader`` if conversion fails
        """
        suffix = path.suffix.lower()

        if suffix in (".hdf", ".h5", ".he5", ".hdf5"):
            try:
                from pyiwfm.io.hydrograph_loader import (
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

                from pyiwfm.io.hydrograph_loader import (
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
                from pyiwfm.io.hydrograph_reader import (
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

    def get_tile_drain_reader(self) -> IWFMHydrographReader | None:
        """Get or create the tile drain hydrograph reader from the output file."""
        if self._tile_drain_reader is not None:
            return self._tile_drain_reader

        if self._model is None:
            return None

        # Check td_output_file_raw on the groundwater component
        if self._model.groundwater:
            output_file = getattr(self._model.groundwater, "td_output_file_raw", "")
            if output_file:
                p = Path(output_file)
                if not p.is_absolute() and self._results_dir:
                    p = self._results_dir / p
                if p.exists():
                    self._tile_drain_reader = self._get_or_convert_hydrograph(p)
                    if self._tile_drain_reader is not None:
                        return self._tile_drain_reader

        # Fallback: scan model directory for *TileDrain*.out
        if self._results_dir:
            for pattern in ("*TileDrain*.out", "*tile_drain*.out", "*tiledrain*.out"):
                matches = list(self._results_dir.glob(pattern))
                if matches:
                    self._tile_drain_reader = self._get_or_convert_hydrograph(matches[0])
                    if self._tile_drain_reader is not None:
                        return self._tile_drain_reader

        return None

    # ------------------------------------------------------------------
    # Hydrograph locations
    # ------------------------------------------------------------------

    def get_gw_physical_locations(self) -> list[dict]:
        """Group GW hydrograph specs into unique physical locations.

        IWFM stores one spec per (location, layer) pair, so C2VSimFG with
        4 layers has 54,544 specs for 13,636 physical locations.  We group
        by matching coordinates (for HYDTYP=0) or node_id (for HYDTYP=1).

        Returns a list of dicts, one per physical location::

            {
                "loc": <first HydrographLocation in group>,
                "node_id": int,
                "name": str,          # display name (layer suffix stripped)
                "columns": [(col_idx, layer), ...],
            }

        The ``columns`` list maps directly to hydrograph output file columns.
        """
        groups: dict[tuple, dict] = {}
        order: list[tuple] = []

        if self._model is not None and self._model.groundwater:
            for idx, loc in enumerate(self._model.groundwater.hydrograph_locations):
                raw_nid = getattr(loc, "node_id", 0) or getattr(loc, "gw_node", 0)
                nid = int(raw_nid) if isinstance(raw_nid, (int, float)) else 0
                # Group key: node_id when available, else exact (x, y)
                key: tuple
                if nid > 0:
                    key = ("node", nid)
                else:
                    key = ("xy", loc.x, loc.y)

                if key not in groups:
                    # Strip trailing %layer_number from name if present
                    raw_name = loc.name or ""
                    if "%" in raw_name:
                        raw_name = raw_name.rsplit("%", 1)[0].rstrip()
                    groups[key] = {
                        "loc": loc,
                        "node_id": nid,
                        "name": raw_name,
                        "columns": [],
                    }
                    order.append(key)
                groups[key]["columns"].append((idx, loc.layer))

        return [groups[k] for k in order]

    def get_hydrograph_locations(self) -> dict[str, list[dict]]:
        """Get all hydrograph locations reprojected to WGS84 (cached)."""
        if self._hydrograph_locations_cache is not None:
            return self._hydrograph_locations_cache

        result: dict[str, list[dict]] = {
            "gw": [],
            "stream": [],
            "subsidence": [],
            "tile_drain": [],
        }

        if self._model is None:
            return result

        # GW hydrograph locations -- one entry per physical location
        phys_locs = self.get_gw_physical_locations()
        for phys_idx, group in enumerate(phys_locs):
            loc = group["loc"]
            lng, lat = self.reproject_coords(loc.x, loc.y)
            display_name = group["name"] or f"GW Hydrograph {phys_idx + 1}"
            result["gw"].append(
                {
                    "id": phys_idx + 1,  # 1-based physical location ID
                    "lng": lng,
                    "lat": lat,
                    "name": display_name,
                    "layer": loc.layer,
                    "node_id": group["node_id"],
                }
            )

        # Stream hydrograph locations -- get coords from associated GW node
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
                    sub_node_id = getattr(spec, "node_id", 0) or getattr(spec, "gw_node", 0)
                    result["subsidence"].append(
                        {
                            "id": spec.id,
                            "lng": lng,
                            "lat": lat,
                            "name": spec.name or f"Subsidence Obs {spec.id}",
                            "layer": spec.layer,
                            "node_id": sub_node_id,
                        }
                    )

        # Tile drain hydrograph locations -- get coords from element centroid
        if self._model.groundwater:
            td_specs = getattr(self._model.groundwater, "td_hydro_specs", [])
            if td_specs:
                grid = self._model.grid
                for spec in td_specs:
                    td_id = spec["id"]
                    td_id_type = spec.get("id_type", 1)
                    # Look up the TileDrain or SubIrrigation to get its element/node
                    td_obj: Any = None
                    if td_id_type == 1:
                        td_obj = self._model.groundwater.tile_drains.get(td_id)
                    elif td_id_type == 2:
                        # Sub-irrigation -- find by ID in sub_irrigations list
                        for si in self._model.groundwater.sub_irrigations:
                            if si.id == td_id:
                                td_obj = si
                                break

                    if td_obj is None or grid is None:
                        continue

                    # TileDrain has `element` (actually gw_node); SubIrrigation has `gw_node`
                    node_id = getattr(td_obj, "element", 0) or getattr(td_obj, "gw_node", 0)
                    if not node_id:
                        continue

                    gw_node = grid.nodes.get(node_id)
                    if gw_node is None:
                        continue

                    x, y = gw_node.x, gw_node.y
                    if x == 0.0 and y == 0.0:
                        continue

                    try:
                        lng, lat = self.reproject_coords(x, y)
                    except Exception:
                        continue

                    result["tile_drain"].append(
                        {
                            "id": td_id,
                            "lng": lng,
                            "lat": lat,
                            "name": spec.get("name", f"Tile Drain {td_id}"),
                            "node_id": node_id,
                        }
                    )

        self._hydrograph_locations_cache = result
        return result

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

        budgets = self.get_available_budgets()  # type: ignore[attr-defined]
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
            info["n_gw_hydrographs"] = len(self.get_gw_physical_locations())

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
