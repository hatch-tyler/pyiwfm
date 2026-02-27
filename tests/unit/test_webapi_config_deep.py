"""Deep tests for pyiwfm.visualization.webapi.config.ModelState.

Targets uncovered branches:
- get_stream_reach_boundaries(): strategies 2 and 3
- get_diversion_timeseries(): mock budget reader
- _reconvert_head_hdf(): mock head_all_converter
- get_subsidence_reader() / get_tile_drain_reader(): fallback glob paths
- _build_cache_eager(): stale cache detection
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
    """Create a clean ModelState."""
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
# get_stream_reach_boundaries - Strategy 2 (streams spec text file)
# ---------------------------------------------------------------------------


class TestStreamReachBoundariesStrategy2:
    """Test reach boundaries loaded from streams spec text file."""

    def test_from_streams_spec(self, tmp_path: Path) -> None:
        """Strategy 2: reads from streams_spec source file."""
        spec_path = tmp_path / "Streams.dat"
        spec_path.write_text("dummy")

        ms = _make_model_state()
        model = _mock_model(
            source_files={"streams_spec": str(spec_path)},
        )
        ms._model = model
        ms._results_dir = tmp_path

        mock_reach = MagicMock()
        mock_reach.id = 1
        mock_reach.node_ids = [10, 20, 30]

        mock_reader = MagicMock()
        mock_reader.read.return_value = (1, 3, [mock_reach])

        with patch(
            "pyiwfm.io.streams.StreamSpecReader",
            return_value=mock_reader,
        ):
            result = ms.get_stream_reach_boundaries()

        assert result is not None
        assert len(result) == 1
        assert result[0] == (1, 10, 30)  # (reach_id, upstream, downstream)


class TestStreamReachBoundariesStrategy3:
    """Test reach boundaries loaded from preprocessor main -> streams spec."""

    def test_from_preprocessor_main(self, tmp_path: Path) -> None:
        """Strategy 3: reads preprocessor main to find streams spec."""
        pp_path = tmp_path / "PreProcessor_MAIN.IN"
        pp_path.write_text("dummy")
        streams_file = tmp_path / "Streams.dat"
        streams_file.write_text("dummy")

        ms = _make_model_state()
        model = _mock_model(
            source_files={"preprocessor_main": str(pp_path)},
        )
        ms._model = model
        ms._results_dir = tmp_path

        mock_pp_config = MagicMock()
        mock_pp_config.streams_file = streams_file

        mock_reach = MagicMock()
        mock_reach.id = 2
        mock_reach.node_ids = [100, 200]

        mock_spec_reader = MagicMock()
        mock_spec_reader.read.return_value = (1, 2, [mock_reach])

        with (
            patch(
                "pyiwfm.io.preprocessor.read_preprocessor_main",
                return_value=mock_pp_config,
            ),
            patch(
                "pyiwfm.io.streams.StreamSpecReader",
                return_value=mock_spec_reader,
            ),
        ):
            result = ms.get_stream_reach_boundaries()

        assert result is not None
        assert result[0] == (2, 100, 200)


# ---------------------------------------------------------------------------
# get_diversion_timeseries
# ---------------------------------------------------------------------------


class TestDiversionTimeseries:
    """Test diversion time series loading."""

    def test_load_diversion_ts(self, tmp_path: Path) -> None:
        """Successfully load diversion time series data."""
        ts_file = tmp_path / "DivTS.dat"
        ts_file.write_text("dummy")

        ms = _make_model_state()
        model = _mock_model(
            source_files={"stream_diversion_ts": str(ts_file)},
            metadata={"simulation_file": str(tmp_path / "Sim.dat")},
        )
        ms._model = model
        ms._results_dir = tmp_path

        mock_times = [1, 2, 3]
        mock_values = np.array([[10.0, 20.0], [11.0, 21.0], [12.0, 22.0]])
        mock_meta = {"ncol": 2}

        mock_reader = MagicMock()
        mock_reader.read_file.return_value = (mock_times, mock_values, mock_meta)

        with patch(
            "pyiwfm.io.timeseries.UnifiedTimeSeriesReader",
            return_value=mock_reader,
        ):
            result = ms.get_diversion_timeseries()

        assert result is not None
        times, values, meta = result
        assert len(times) == 3
        assert values.shape == (3, 2)

    def test_no_ts_file_returns_none(self) -> None:
        """Return None when no diversion time series path configured."""
        ms = _make_model_state()
        model = _mock_model(source_files={})
        ms._model = model

        assert ms.get_diversion_timeseries() is None


# ---------------------------------------------------------------------------
# _reconvert_head_hdf
# ---------------------------------------------------------------------------


class TestReconvertHeadHdf:
    """Test _reconvert_head_hdf fallback paths."""

    def test_no_text_source_returns_existing(self, tmp_path: Path) -> None:
        """When no companion text file exists, return existing HDF as-is."""
        hdf_path = tmp_path / "GWHeadAll.hdf"
        hdf_path.write_text("dummy hdf")

        ms = _make_model_state()
        ms._model = _mock_model()

        mock_loader = MagicMock()
        with patch("pyiwfm.io.head_loader.LazyHeadDataLoader", return_value=mock_loader):
            result = ms._reconvert_head_hdf(hdf_path, n_layers=4)

        assert result is mock_loader

    def test_with_text_source_reconverts(self, tmp_path: Path) -> None:
        """When companion .out file exists, re-convert with correct n_layers."""
        hdf_path = tmp_path / "GWHeadAll.hdf"
        hdf_path.write_text("dummy hdf")
        out_path = tmp_path / "GWHeadAll.out"
        out_path.write_text("dummy text")

        ms = _make_model_state()
        ms._model = _mock_model()

        mock_loader = MagicMock()
        with (
            patch("pyiwfm.io.head_all_converter.convert_headall_to_hdf") as mock_convert,
            patch("pyiwfm.io.head_loader.LazyHeadDataLoader", return_value=mock_loader),
        ):
            result = ms._reconvert_head_hdf(hdf_path, n_layers=4)

        mock_convert.assert_called_once_with(out_path, hdf_path, n_layers=4)
        assert result is mock_loader


# ---------------------------------------------------------------------------
# get_subsidence_reader / get_tile_drain_reader - fallback glob
# ---------------------------------------------------------------------------


class TestSubsidenceReaderFallback:
    """Test fallback glob pattern for subsidence reader."""

    def test_fallback_glob_finds_file(self, tmp_path: Path) -> None:
        """Glob pattern finds *Subsidence*.out in results dir."""
        subs_file = tmp_path / "GW_Subsidence.out"
        subs_file.write_text("dummy subsidence output")

        ms = _make_model_state()
        gw = MagicMock()
        gw.subsidence_config = None  # No config -> falls through to glob
        model = _mock_model(groundwater=gw)
        ms._model = model
        ms._results_dir = tmp_path

        mock_reader = MagicMock()
        mock_reader.n_timesteps = 10

        with patch.object(ms, "_get_or_convert_hydrograph", return_value=mock_reader):
            result = ms.get_subsidence_reader()

        assert result is mock_reader


class TestTileDrainReaderFallback:
    """Test fallback glob pattern for tile drain reader."""

    def test_fallback_glob_finds_file(self, tmp_path: Path) -> None:
        """Glob pattern finds *TileDrain*.out in results dir."""
        td_file = tmp_path / "TileDrainHyd.out"
        td_file.write_text("dummy tile drain output")

        ms = _make_model_state()
        gw = MagicMock()
        gw.td_output_file_raw = ""  # No raw path -> falls through to glob
        model = _mock_model(groundwater=gw)
        ms._model = model
        ms._results_dir = tmp_path

        mock_reader = MagicMock()
        mock_reader.n_timesteps = 5

        with patch.object(ms, "_get_or_convert_hydrograph", return_value=mock_reader):
            result = ms.get_tile_drain_reader()

        assert result is mock_reader


# ---------------------------------------------------------------------------
# _build_cache_eager - stale cache detection
# ---------------------------------------------------------------------------


class TestBuildCacheEager:
    """Test eager cache building and stale detection."""

    def test_stale_cache_triggers_rebuild(self, tmp_path: Path) -> None:
        """A stale cache triggers SqliteCacheBuilder.build()."""
        ms = _make_model_state()
        model = _mock_model(metadata={"simulation_file": str(tmp_path / "Sim.dat")})
        ms._model = model
        ms._results_dir = tmp_path
        ms._no_cache = False
        ms._rebuild_cache = False

        mock_builder = MagicMock()
        mock_loader = MagicMock()
        mock_loader.get_stats.return_value = {"n_items": 0}

        with (
            patch("pyiwfm.io.cache_builder.is_cache_stale", return_value=True),
            patch("pyiwfm.io.cache_builder.SqliteCacheBuilder", return_value=mock_builder),
            patch("pyiwfm.io.cache_loader.SqliteCacheLoader", return_value=mock_loader),
        ):
            ms._build_cache_eager()

        mock_builder.build.assert_called_once()
        assert ms._cache_loader is mock_loader

    def test_no_model_skips_cache(self) -> None:
        """_build_cache_eager returns immediately when no model is loaded."""
        ms = _make_model_state()
        ms._model = None
        ms._build_cache_eager()
        assert ms._cache_loader is None
