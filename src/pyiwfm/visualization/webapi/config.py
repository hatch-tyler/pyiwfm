"""
Configuration settings for the FastAPI web viewer.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from pyiwfm.visualization.webapi._budget_state import BudgetStateMixin
from pyiwfm.visualization.webapi._cache_state import CacheStateMixin
from pyiwfm.visualization.webapi._mesh_state import MeshStateMixin
from pyiwfm.visualization.webapi._results_state import ResultsStateMixin

if TYPE_CHECKING:
    from pyiwfm.core.model import IWFMModel
    from pyiwfm.core.zones import ZoneDefinition
    from pyiwfm.io.area_loader import AreaDataManager
    from pyiwfm.io.budget import BudgetReader
    from pyiwfm.io.cache_loader import SqliteCacheLoader
    from pyiwfm.io.head_loader import LazyHeadDataLoader
    from pyiwfm.io.hydrograph_reader import IWFMHydrographReader
    from pyiwfm.io.zbudget import ZBudgetReader

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


class ModelState(MeshStateMixin, ResultsStateMixin, BudgetStateMixin, CacheStateMixin):
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
        self._elem_id_to_idx: dict[int, int] | None = None

        # Cached hydrograph locations (reprojected to WGS84)
        self._hydrograph_locations_cache: dict[str, list[dict]] | None = None

        # Results-related state
        self._crs: str = "+proj=utm +zone=10 +datum=NAD83 +units=us-ft +no_defs"
        self._transformer: Any = None  # pyproj Transformer (lazy)
        self._geojson_cache: dict[int, dict] = {}  # layer -> GeoJSON in WGS84
        self._head_loader: LazyHeadDataLoader | None = None
        self._gw_hydrograph_reader: IWFMHydrographReader | None = None
        self._stream_hydrograph_reader: IWFMHydrographReader | None = None
        self._subsidence_reader: IWFMHydrographReader | None = None
        self._tile_drain_reader: IWFMHydrographReader | None = None
        self._budget_readers: dict[str, BudgetReader] = {}
        self._zbudget_readers: dict[str, ZBudgetReader] = {}
        self._active_zone_def: ZoneDefinition | None = None
        self._results_dir: Path | None = None
        self._area_manager: AreaDataManager | None = None
        self._observations: dict[str, dict] = {}  # id -> observation data

        # SQLite cache
        self._cache_loader: SqliteCacheLoader | None = None
        self._no_cache: bool = False  # Set True to disable cache
        self._rebuild_cache: bool = False  # Set True to force rebuild

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
        no_cache: bool = False,
        rebuild_cache: bool = False,
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
        self._tile_drain_reader = None
        self._budget_readers = {}
        self._zbudget_readers = {}
        self._active_zone_def = None
        self._area_manager = None
        self._observations = {}
        self._stream_reach_boundaries = None
        self._diversion_ts_data = None
        self._node_id_to_idx = None
        self._sorted_elem_ids = None
        self._elem_id_to_idx = None
        self._hydrograph_locations_cache = None
        # Clear cached physical location grouping
        if hasattr(self, "_gw_phys_locs"):
            del self._gw_phys_locs

        # SQLite cache
        if self._cache_loader is not None:
            self._cache_loader.close()
        self._cache_loader = None
        self._no_cache = no_cache
        self._rebuild_cache = rebuild_cache

        # Determine results directory from model metadata
        sim_file = model.metadata.get("simulation_file")
        if sim_file:
            self._results_dir = Path(sim_file).parent
        else:
            self._results_dir = None

        # Eagerly build the cache at model-load time (before requests).
        # This avoids the cache build running inside a request handler
        # where FastAPI's thread pool can cancel long-running operations.
        if not no_cache:
            self._build_cache_eager()

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
