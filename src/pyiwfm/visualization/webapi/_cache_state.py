"""Mixin providing SQLite cache and grid-index methods for ModelState."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyiwfm.core.model import IWFMModel
    from pyiwfm.io.area_loader import AreaDataManager
    from pyiwfm.io.cache_loader import SqliteCacheLoader

logger = logging.getLogger(__name__)


class CacheStateMixin:
    """Mixin providing SQLite cache and grid-index methods for ModelState."""

    # -- Attributes set by ModelState.__init__ (declared for type checkers) --
    _model: IWFMModel | None
    _results_dir: Path | None
    _node_id_to_idx: dict[int, int] | None
    _sorted_elem_ids: list[int] | None
    _elem_id_to_idx: dict[int, int] | None
    _cache_loader: SqliteCacheLoader | None
    _no_cache: bool
    _rebuild_cache: bool
    _budget_readers: dict
    _area_manager: AreaDataManager | None

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

    def get_elem_id_to_idx(self) -> dict[int, int]:
        """Get cached elem_id -> array index mapping."""
        if self._elem_id_to_idx is None:
            sorted_ids = self.get_sorted_elem_ids()
            self._elem_id_to_idx = {eid: i for i, eid in enumerate(sorted_ids)}
        return self._elem_id_to_idx

    # ------------------------------------------------------------------
    # SQLite cache
    # ------------------------------------------------------------------

    def _build_cache_eager(self) -> None:
        """Build the SQLite cache eagerly at model-load time.

        Called from set_model() so the cache is ready before the server
        starts handling requests.  Long-running builds (10+ minutes for
        large models) must happen here, NOT inside a request handler.
        """
        if self._model is None:
            return

        cache_path = self._get_cache_path()
        if cache_path is None:
            return

        try:
            from pyiwfm.io.cache_builder import (
                SqliteCacheBuilder,
                is_cache_stale,
            )

            if self._rebuild_cache or is_cache_stale(cache_path, self._model):
                logger.info("Building SQLite cache (this may take several minutes)...")
                builder = SqliteCacheBuilder(cache_path)
                builder.build(
                    model=self._model,
                    head_loader=self.get_head_loader(),  # type: ignore[attr-defined]
                    budget_readers=self._budget_readers or None,
                    area_manager=self._area_manager,
                    gw_hydrograph_reader=self.get_gw_hydrograph_reader(),  # type: ignore[attr-defined]
                    stream_hydrograph_reader=self.get_stream_hydrograph_reader(),  # type: ignore[attr-defined]
                    subsidence_reader=self.get_subsidence_reader(),  # type: ignore[attr-defined]
                    tile_drain_reader=self.get_tile_drain_reader(),  # type: ignore[attr-defined]
                )
                self._rebuild_cache = False
                logger.info("SQLite cache build complete.")

            from pyiwfm.io.cache_loader import (
                SqliteCacheLoader,
            )

            self._cache_loader = SqliteCacheLoader(cache_path)
            logger.info("SQLite cache loaded: %s", cache_path)
            stats = self._cache_loader.get_stats()
            logger.info("Cache stats: %s", stats)

        except Exception:
            logger.exception("Failed to build SQLite cache")

    def get_cache_loader(self) -> SqliteCacheLoader | None:
        """Get the SQLite cache loader.

        Returns the pre-built cache (from set_model), or tries to open
        an existing cache file on disk.  Does NOT trigger a build -- that
        happens eagerly in set_model() / _build_cache_eager().
        """
        if self._no_cache:
            return None
        if self._cache_loader is not None:
            return self._cache_loader

        # Try to open an existing cache file on disk.
        cache_path = self._get_cache_path()
        if cache_path is None or not cache_path.exists():
            return None

        try:
            from pyiwfm.io.cache_loader import (
                SqliteCacheLoader,
            )

            self._cache_loader = SqliteCacheLoader(cache_path)
            logger.info("SQLite cache loaded: %s", cache_path)
            stats = self._cache_loader.get_stats()
            logger.info("Cache stats: %s", stats)
            return self._cache_loader

        except Exception as e:
            logger.warning("SQLite cache unavailable: %s", e)
            return None

    def _get_cache_path(self) -> Path | None:
        """Determine the cache file path from model source."""
        if self._results_dir is not None:
            return self._results_dir / "model_cache.db"
        # Fall back to model source directory
        src = self._model.metadata.get("source_dir", "") if self._model else ""
        if src:
            return Path(src) / "model_cache.db"
        return None

    def get_cached_head_by_element(
        self, frame_idx: int, layer: int
    ) -> tuple[list[float | None], float, float] | None:
        """Try to get element-averaged heads from cache."""
        loader = self.get_cache_loader()
        if loader is None:
            return None
        result = loader.get_head_by_element(frame_idx, layer)
        if result is None:
            return None
        arr, min_val, max_val = result
        import numpy as np

        values: list[float | None] = [None if np.isnan(v) else round(float(v), 3) for v in arr]
        return values, min_val, max_val

    def get_cached_head_range(self, layer: int) -> dict | None:
        """Try to get head range from cache."""
        loader = self.get_cache_loader()
        if loader is None:
            return None
        return loader.get_head_range(layer)
