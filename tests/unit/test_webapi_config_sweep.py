"""Sweep tests for webapi config.ModelState — targeting remaining uncovered paths.

Covers:
- set_model: _gw_phys_locs deletion, cache_loader.close()
- _get_n_gw_layers: None model, metadata fallback
- _get_or_convert_hydrograph: HDF5 and text paths
- get_gw_physical_locations: % name stripping
- get_hydrograph_locations: cached return, tile drain locations
- get_available_zbudgets / get_zbudget_reader
- _compute_bounds: grid is None
- get_cache_loader: return cached loader
- _get_cache_path: source_dir fallback
- get_cached_head_by_element / get_cached_head_range
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytest.importorskip("fastapi", reason="FastAPI not available")

from pyiwfm.visualization.webapi.config import ModelState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model_state() -> ModelState:
    return ModelState()


def _mock_model(**kwargs) -> MagicMock:
    model = MagicMock()
    model.name = "TestModel"
    model.metadata = kwargs.get("metadata", {})
    model.source_files = kwargs.get("source_files", {})
    model.groundwater = kwargs.get("groundwater", None)
    model.streams = kwargs.get("streams", None)
    model.grid = kwargs.get("grid", None)
    model.rootzone = kwargs.get("rootzone", None)
    model.stratigraphy = kwargs.get("stratigraphy", None)
    model.n_nodes = kwargs.get("n_nodes", 9)
    model.n_layers = kwargs.get("n_layers", 1)
    return model


# ---------------------------------------------------------------------------
# set_model: _gw_phys_locs deletion and cache_loader.close()
# ---------------------------------------------------------------------------


class TestSetModelEdgeCases:
    """Cover set_model branches: _gw_phys_locs attr deletion, cache_loader close."""

    def test_set_model_deletes_gw_phys_locs(self, tmp_path: Path) -> None:
        """Line 139: When _gw_phys_locs exists, set_model deletes it."""
        ms = _make_model_state()
        ms._gw_phys_locs = {"some": "data"}  # type: ignore[attr-defined]

        model = _mock_model(metadata={"simulation_file": str(tmp_path / "Sim.dat")})
        ms.set_model(model, no_cache=True)

        assert not hasattr(ms, "_gw_phys_locs")

    def test_set_model_closes_cache_loader(self, tmp_path: Path) -> None:
        """Line 143: When _cache_loader is not None, set_model calls close()."""
        ms = _make_model_state()
        mock_loader = MagicMock()
        ms._cache_loader = mock_loader

        model = _mock_model(metadata={"simulation_file": str(tmp_path / "Sim.dat")})
        ms.set_model(model, no_cache=True)

        mock_loader.close.assert_called_once()
        assert ms._cache_loader is None


# ---------------------------------------------------------------------------
# _get_n_gw_layers: None model and metadata fallback
# ---------------------------------------------------------------------------


class TestGetNGwLayers:
    """Cover _get_n_gw_layers branches."""

    def test_no_model_returns_1(self) -> None:
        """Line 607: When model is None, return 1."""
        ms = _make_model_state()
        ms._model = None
        assert ms._get_n_gw_layers() == 1

    def test_metadata_fallback(self) -> None:
        """Lines 613-614: When n_layers is 0, fall back to metadata."""
        ms = _make_model_state()
        model = _mock_model(metadata={"gw_aquifer_n_layers": 3})
        model.n_layers = 0
        ms._model = model
        assert ms._get_n_gw_layers() == 3

    def test_metadata_fallback_returns_1_when_missing(self) -> None:
        """Line 614: When metadata fallback is also 0, return 1."""
        ms = _make_model_state()
        model = _mock_model(metadata={"gw_aquifer_n_layers": 0})
        model.n_layers = 0
        ms._model = model
        assert ms._get_n_gw_layers() == 1


# ---------------------------------------------------------------------------
# _get_or_convert_hydrograph: HDF5 and text paths
# ---------------------------------------------------------------------------


class TestGetOrConvertHydrograph:
    """Cover _get_or_convert_hydrograph HDF5 loading and text fallback."""

    def test_hdf5_path_success(self, tmp_path: Path) -> None:
        """Lines 766-779: HDF5 suffix loads via LazyHydrographDataLoader."""
        hdf_path = tmp_path / "hydro.hdf"
        hdf_path.write_text("dummy")

        ms = _make_model_state()
        ms._model = _mock_model()

        mock_loader = MagicMock()
        mock_loader.n_timesteps = 10
        mock_loader.n_columns = 3

        with patch(
            "pyiwfm.io.hydrograph_loader.LazyHydrographDataLoader",
            return_value=mock_loader,
        ):
            result = ms._get_or_convert_hydrograph(hdf_path)

        assert result is mock_loader

    def test_hdf5_path_exception_returns_none(self, tmp_path: Path) -> None:
        """Lines 780-781: When HDF5 load raises, falls through."""
        hdf_path = tmp_path / "hydro.h5"
        hdf_path.write_text("dummy")

        ms = _make_model_state()
        ms._model = _mock_model()

        with patch(
            "pyiwfm.io.hydrograph_loader.LazyHydrographDataLoader",
            side_effect=Exception("corrupt HDF5"),
        ):
            result = ms._get_or_convert_hydrograph(hdf_path)

        assert result is None

    def test_text_path_auto_convert_success(self, tmp_path: Path) -> None:
        """Lines 785-806: Text file auto-converted to HDF5 cache."""
        txt_path = tmp_path / "hydro.out"
        txt_path.write_text("dummy text")

        ms = _make_model_state()
        ms._model = _mock_model()

        mock_loader = MagicMock()
        mock_loader.n_timesteps = 5
        mock_loader.n_columns = 2

        with (
            patch(
                "pyiwfm.io.hydrograph_converter.convert_hydrograph_to_hdf",
            ) as mock_convert,
            patch(
                "pyiwfm.io.hydrograph_loader.LazyHydrographDataLoader",
                return_value=mock_loader,
            ),
        ):
            result = ms._get_or_convert_hydrograph(txt_path)

        mock_convert.assert_called_once()
        assert result is mock_loader

    def test_text_path_convert_fails_falls_to_text_reader(self, tmp_path: Path) -> None:
        """Lines 807-828: When HDF5 conversion fails, fall back to text reader."""
        txt_path = tmp_path / "hydro.dat"
        txt_path.write_text("dummy text")

        ms = _make_model_state()
        ms._model = _mock_model()

        mock_text_reader = MagicMock()
        mock_text_reader.n_columns = 3
        mock_text_reader.n_timesteps = 8

        with (
            patch(
                "pyiwfm.io.hydrograph_converter.convert_hydrograph_to_hdf",
                side_effect=Exception("convert fail"),
            ),
            patch(
                "pyiwfm.io.hydrograph_reader.IWFMHydrographReader",
                return_value=mock_text_reader,
            ),
        ):
            result = ms._get_or_convert_hydrograph(txt_path)

        assert result is mock_text_reader

    def test_unknown_suffix_returns_none(self, tmp_path: Path) -> None:
        """Unknown suffix that is not HDF5 or text returns None."""
        unknown = tmp_path / "hydro.xyz"
        unknown.write_text("dummy")

        ms = _make_model_state()
        ms._model = _mock_model()

        result = ms._get_or_convert_hydrograph(unknown)
        assert result is None


# ---------------------------------------------------------------------------
# get_gw_physical_locations: % name stripping
# ---------------------------------------------------------------------------


class TestGwPhysicalLocationsNameStripping:
    """Cover the % name stripping in get_gw_physical_locations (line 983)."""

    def test_name_with_percent_stripped(self) -> None:
        """When loc.name has %, strip trailing %layer part."""
        ms = _make_model_state()
        gw = MagicMock()

        loc1 = MagicMock()
        loc1.name = "Well A%1"
        loc1.node_id = 10
        loc1.x = 100.0
        loc1.y = 200.0
        loc1.layer = 1

        loc2 = MagicMock()
        loc2.name = "Well A%2"
        loc2.node_id = 10
        loc2.x = 100.0
        loc2.y = 200.0
        loc2.layer = 2

        gw.hydrograph_locations = [loc1, loc2]
        model = _mock_model(groundwater=gw)
        ms._model = model

        result = ms.get_gw_physical_locations()
        assert len(result) == 1
        assert result[0]["name"] == "Well A"
        assert len(result[0]["columns"]) == 2


# ---------------------------------------------------------------------------
# get_hydrograph_locations: cached return, tile drain locations
# ---------------------------------------------------------------------------


class TestHydrographLocations:
    """Cover get_hydrograph_locations cached return and tile drain path."""

    def test_cached_result_returned(self) -> None:
        """Line 998: When cache is populated, return it directly."""
        ms = _make_model_state()
        ms._model = _mock_model()
        cached = {"gw": [], "stream": [], "subsidence": [], "tile_drain": []}
        ms._hydrograph_locations_cache = cached

        result = ms.get_hydrograph_locations()
        assert result is cached

    def test_tile_drain_locations_via_td_hydro_specs(self) -> None:
        """Lines 1098-1132: tile drain hydrograph locations resolved from specs."""
        from pyiwfm.core.mesh import AppGrid, Element, Node

        nodes = {
            1: Node(id=1, x=500.0, y=600.0),
            2: Node(id=2, x=700.0, y=800.0),
            3: Node(id=3, x=900.0, y=1000.0),
            4: Node(id=4, x=1100.0, y=1200.0),
        }
        elements = {
            1: Element(id=1, vertices=(1, 2, 3, 4), subregion=1),
        }
        grid = AppGrid(nodes=nodes, elements=elements)
        grid.compute_connectivity()
        grid.compute_areas()

        gw = MagicMock()
        gw.hydrograph_locations = []
        gw.subsidence_config = None
        gw.td_hydro_specs = [
            {"id": 1, "id_type": 1, "name": "TD1"},
        ]
        td_obj = MagicMock()
        td_obj.element = 1  # points to gw node 1
        td_obj.gw_node = 0
        gw.tile_drains = {1: td_obj}
        gw.sub_irrigations = []

        model = _mock_model(groundwater=gw, grid=grid)
        model.streams = None
        model.metadata = {}
        ms = _make_model_state()
        ms._model = model

        # Mock reproject_coords to be identity
        ms.reproject_coords = lambda x, y: (x, y)  # type: ignore[assignment]

        result = ms.get_hydrograph_locations()
        assert len(result["tile_drain"]) == 1
        assert result["tile_drain"][0]["id"] == 1
        assert result["tile_drain"][0]["node_id"] == 1

    def test_tile_drain_sub_irrigation_type(self) -> None:
        """Lines 1106-1109: tile drain with id_type==2 (sub-irrigation)."""
        from pyiwfm.core.mesh import AppGrid, Element, Node

        nodes = {
            5: Node(id=5, x=300.0, y=400.0),
            6: Node(id=6, x=500.0, y=400.0),
            7: Node(id=7, x=500.0, y=600.0),
            8: Node(id=8, x=300.0, y=600.0),
        }
        elements = {1: Element(id=1, vertices=(5, 6, 7, 8), subregion=1)}
        grid = AppGrid(nodes=nodes, elements=elements)
        grid.compute_connectivity()
        grid.compute_areas()

        si = MagicMock()
        si.id = 42
        si.element = 0
        si.gw_node = 5

        gw = MagicMock()
        gw.hydrograph_locations = []
        gw.subsidence_config = None
        gw.td_hydro_specs = [
            {"id": 42, "id_type": 2, "name": "SubIrr1"},
        ]
        gw.tile_drains = {}
        gw.sub_irrigations = [si]

        model = _mock_model(groundwater=gw, grid=grid)
        model.streams = None
        model.metadata = {}
        ms = _make_model_state()
        ms._model = model
        ms.reproject_coords = lambda x, y: (x, y)  # type: ignore[assignment]

        result = ms.get_hydrograph_locations()
        assert len(result["tile_drain"]) == 1
        assert result["tile_drain"][0]["id"] == 42
        assert result["tile_drain"][0]["node_id"] == 5

    def test_tile_drain_skipped_when_obj_not_found(self) -> None:
        """Line 1111: tile drain obj is None -> skipped."""
        from pyiwfm.core.mesh import AppGrid, Element, Node

        nodes = {
            1: Node(id=1, x=100.0, y=200.0),
            2: Node(id=2, x=300.0, y=200.0),
            3: Node(id=3, x=300.0, y=400.0),
            4: Node(id=4, x=100.0, y=400.0),
        }
        elements = {1: Element(id=1, vertices=(1, 2, 3, 4), subregion=1)}
        grid = AppGrid(nodes=nodes, elements=elements)
        grid.compute_connectivity()
        grid.compute_areas()

        gw = MagicMock()
        gw.hydrograph_locations = []
        gw.subsidence_config = None
        gw.td_hydro_specs = [
            {"id": 999, "id_type": 1, "name": "TD_missing"},
        ]
        gw.tile_drains = {}  # Not found
        gw.sub_irrigations = []

        model = _mock_model(groundwater=gw, grid=grid)
        model.streams = None
        model.metadata = {}
        ms = _make_model_state()
        ms._model = model
        ms.reproject_coords = lambda x, y: (x, y)  # type: ignore[assignment]

        result = ms.get_hydrograph_locations()
        assert len(result["tile_drain"]) == 0

    def test_tile_drain_skipped_when_node_zero_coords(self) -> None:
        """Line 1124-1125: tile drain gw_node at (0, 0) -> skipped."""
        from pyiwfm.core.mesh import AppGrid, Element, Node

        nodes = {
            1: Node(id=1, x=0.0, y=0.0),
            2: Node(id=2, x=100.0, y=0.0),
            3: Node(id=3, x=100.0, y=100.0),
            4: Node(id=4, x=0.0, y=100.0),
        }
        elements = {1: Element(id=1, vertices=(1, 2, 3, 4), subregion=1)}
        grid = AppGrid(nodes=nodes, elements=elements)
        grid.compute_connectivity()
        grid.compute_areas()

        gw = MagicMock()
        gw.hydrograph_locations = []
        gw.subsidence_config = None
        gw.td_hydro_specs = [
            {"id": 1, "id_type": 1, "name": "TD_zero"},
        ]
        td_obj = MagicMock()
        td_obj.element = 1  # points to node 1 at (0,0)
        td_obj.gw_node = 0
        gw.tile_drains = {1: td_obj}
        gw.sub_irrigations = []

        model = _mock_model(groundwater=gw, grid=grid)
        model.streams = None
        model.metadata = {}
        ms = _make_model_state()
        ms._model = model
        ms.reproject_coords = lambda x, y: (x, y)  # type: ignore[assignment]

        result = ms.get_hydrograph_locations()
        assert len(result["tile_drain"]) == 0


# ---------------------------------------------------------------------------
# get_available_zbudgets / get_zbudget_reader
# ---------------------------------------------------------------------------


class TestZBudgetReaders:
    """Cover get_available_zbudgets and get_zbudget_reader."""

    def test_get_available_zbudgets_with_files(self, tmp_path: Path) -> None:
        """Lines 1244-1248: iterate zbudget keys, find existing files."""
        gw_zbudget = tmp_path / "gw_zbudget.hdf"
        gw_zbudget.write_text("dummy")

        ms = _make_model_state()
        model = _mock_model(metadata={"gw_zbudget_file": str(gw_zbudget)})
        ms._model = model
        ms._results_dir = tmp_path

        result = ms.get_available_zbudgets()
        assert "gw" in result

    def test_get_available_zbudgets_no_model(self) -> None:
        """Line 1231: no model returns empty list."""
        ms = _make_model_state()
        ms._model = None
        assert ms.get_available_zbudgets() == []

    def test_get_zbudget_reader_success(self, tmp_path: Path) -> None:
        """Lines 1275-1290: create and cache a ZBudgetReader."""
        zb_file = tmp_path / "gw_zbudget.hdf"
        zb_file.write_text("dummy")

        ms = _make_model_state()
        model = _mock_model(metadata={"gw_zbudget_file": str(zb_file)})
        ms._model = model
        ms._results_dir = tmp_path

        mock_reader = MagicMock()
        mock_reader.descriptor = "GW ZBudget"

        with patch("pyiwfm.io.zbudget.ZBudgetReader", return_value=mock_reader):
            result = ms.get_zbudget_reader("gw")

        assert result is mock_reader
        assert ms._zbudget_readers["gw"] is mock_reader

    def test_get_zbudget_reader_file_not_found(self, tmp_path: Path) -> None:
        """Lines 1278-1279: file doesn't exist returns None."""
        ms = _make_model_state()
        model = _mock_model(metadata={"gw_zbudget_file": str(tmp_path / "nonexistent.hdf")})
        ms._model = model
        ms._results_dir = tmp_path

        result = ms.get_zbudget_reader("gw")
        assert result is None

    def test_get_zbudget_reader_exception(self, tmp_path: Path) -> None:
        """Lines 1288-1290: reader constructor raises exception."""
        zb_file = tmp_path / "gw_zbudget.hdf"
        zb_file.write_text("corrupt")

        ms = _make_model_state()
        model = _mock_model(metadata={"gw_zbudget_file": str(zb_file)})
        ms._model = model
        ms._results_dir = tmp_path

        with patch(
            "pyiwfm.io.zbudget.ZBudgetReader",
            side_effect=Exception("corrupt file"),
        ):
            result = ms.get_zbudget_reader("gw")

        assert result is None

    def test_get_zbudget_reader_relative_path(self, tmp_path: Path) -> None:
        """Lines 1275-1276: relative path resolved against _results_dir."""
        zb_file = tmp_path / "gw_zbudget.hdf"
        zb_file.write_text("dummy")

        ms = _make_model_state()
        model = _mock_model(metadata={"gw_zbudget_file": "gw_zbudget.hdf"})
        ms._model = model
        ms._results_dir = tmp_path

        mock_reader = MagicMock()
        mock_reader.descriptor = "GW ZBudget"

        with patch("pyiwfm.io.zbudget.ZBudgetReader", return_value=mock_reader):
            result = ms.get_zbudget_reader("gw")

        assert result is mock_reader

    def test_get_zbudget_reader_no_model(self) -> None:
        """Line 1258: no model returns None."""
        ms = _make_model_state()
        ms._model = None
        assert ms.get_zbudget_reader("gw") is None

    def test_get_zbudget_reader_unknown_type(self) -> None:
        """Unknown type returns None."""
        ms = _make_model_state()
        ms._model = _mock_model()
        assert ms.get_zbudget_reader("unknown_type") is None

    def test_get_zbudget_reader_cached(self) -> None:
        """Cached reader returned directly."""
        ms = _make_model_state()
        ms._model = _mock_model()
        cached = MagicMock()
        ms._zbudget_readers["gw"] = cached
        assert ms.get_zbudget_reader("gw") is cached


# ---------------------------------------------------------------------------
# _compute_bounds: grid is None
# ---------------------------------------------------------------------------


class TestComputeBoundsEdgeCases:
    """Cover _compute_bounds when grid is None."""

    def test_grid_is_none(self) -> None:
        """Line 1421: grid is None returns zeros."""
        ms = _make_model_state()
        model = _mock_model(grid=None)
        ms._model = model
        result = ms._compute_bounds()
        assert result == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# get_cache_loader: return already-set loader
# ---------------------------------------------------------------------------


class TestGetCacheLoaderCached:
    """Cover get_cache_loader returning already-set loader (line 1501)."""

    def test_returns_existing_loader(self) -> None:
        """Line 1501: When _cache_loader is set and _no_cache is False, return it."""
        ms = _make_model_state()
        ms._no_cache = False
        mock_loader = MagicMock()
        ms._cache_loader = mock_loader

        result = ms.get_cache_loader()
        assert result is mock_loader


# ---------------------------------------------------------------------------
# _get_cache_path: source_dir fallback
# ---------------------------------------------------------------------------


class TestGetCachePathSourceDirFallback:
    """Cover _get_cache_path source_dir metadata fallback (line 1530)."""

    def test_source_dir_fallback(self, tmp_path: Path) -> None:
        """Line 1530: When results_dir is None, use source_dir from metadata."""
        ms = _make_model_state()
        model = _mock_model(metadata={"source_dir": str(tmp_path)})
        ms._model = model
        ms._results_dir = None

        result = ms._get_cache_path()
        assert result == tmp_path / "model_cache.db"


# ---------------------------------------------------------------------------
# get_cached_head_by_element / get_cached_head_range
# ---------------------------------------------------------------------------


class TestCachedHeadAccess:
    """Cover get_cached_head_by_element and get_cached_head_range."""

    def test_head_by_element_success(self) -> None:
        """Lines 1540-1547: successful cache lookup."""
        ms = _make_model_state()
        ms._no_cache = False
        mock_loader = MagicMock()
        arr = np.array([10.123, float("nan"), 20.456])
        mock_loader.get_head_by_element.return_value = (arr, 10.0, 20.0)
        mock_loader.get_stats.return_value = {}
        ms._cache_loader = mock_loader

        result = ms.get_cached_head_by_element(0, 1)
        assert result is not None
        values, min_val, max_val = result
        assert values[0] == 10.123
        assert values[1] is None  # NaN -> None
        assert values[2] == 20.456
        assert min_val == 10.0
        assert max_val == 20.0

    def test_head_by_element_cache_miss(self) -> None:
        """Lines 1541-1542: cache returns None."""
        ms = _make_model_state()
        ms._no_cache = False
        mock_loader = MagicMock()
        mock_loader.get_head_by_element.return_value = None
        mock_loader.get_stats.return_value = {}
        ms._cache_loader = mock_loader

        result = ms.get_cached_head_by_element(0, 1)
        assert result is None

    def test_head_by_element_no_loader(self) -> None:
        """When cache_loader is None, return None."""
        ms = _make_model_state()
        ms._no_cache = True

        result = ms.get_cached_head_by_element(0, 1)
        assert result is None

    def test_head_range_success(self) -> None:
        """Line 1554: return head range from cache."""
        ms = _make_model_state()
        ms._no_cache = False
        mock_loader = MagicMock()
        mock_loader.get_head_range.return_value = {"min": 5.0, "max": 50.0}
        mock_loader.get_stats.return_value = {}
        ms._cache_loader = mock_loader

        result = ms.get_cached_head_range(1)
        assert result == {"min": 5.0, "max": 50.0}

    def test_head_range_no_loader(self) -> None:
        """When cache_loader is None, return None."""
        ms = _make_model_state()
        ms._no_cache = True

        result = ms.get_cached_head_range(1)
        assert result is None
