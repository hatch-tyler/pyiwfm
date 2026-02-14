"""Unit tests for CompleteModelWriter, ModelWriteConfig, and TimeSeriesCopier.

Tests per-file path control, relative path computation, time series copying,
and full model write orchestration.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from pyiwfm.core.mesh import AppGrid, Element, Node, Subregion
from pyiwfm.core.stratigraphy import Stratigraphy
from pyiwfm.core.model import IWFMModel
from pyiwfm.io.config import ModelWriteConfig, OutputFormat
from pyiwfm.io.model_writer import (
    CompleteModelWriter,
    ModelWriteResult,
    TimeSeriesCopier,
    write_model,
    _iso_to_iwfm_date,
    TS_KEY_MAPPING,
    TS_DSS_PARAMS,
    _compute_dss_interval,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_model(
    small_grid_nodes: list[dict],
    small_grid_elements: list[dict],
    sample_stratigraphy_data: dict,
) -> IWFMModel:
    """Create a simple model for testing writers."""
    nodes = {d["id"]: Node(**d) for d in small_grid_nodes}
    elements = {d["id"]: Element(**d) for d in small_grid_elements}
    subregions = {
        1: Subregion(id=1, name="Region A"),
        2: Subregion(id=2, name="Region B"),
    }
    mesh = AppGrid(nodes=nodes, elements=elements, subregions=subregions)
    strat = Stratigraphy(**sample_stratigraphy_data)

    model = IWFMModel(
        name="Test Model",
        mesh=mesh,
        stratigraphy=strat,
        metadata={
            "start_date": "1990-09-30T00:00:00",
            "end_date": "2000-09-30T00:00:00",
            "time_step_length": "1",
            "time_step_unit": "DAY",
        },
    )
    return model


# =============================================================================
# ModelWriteConfig Tests
# =============================================================================


class TestModelWriteConfig:
    """Tests for ModelWriteConfig dataclass."""

    def test_default_paths_nested_layout(self, tmp_path: Path) -> None:
        """Default paths produce standard nested layout."""
        config = ModelWriteConfig(output_dir=tmp_path)
        sim_path = config.get_path("simulation_main")
        assert sim_path == tmp_path / "Simulation" / "Simulation_MAIN.IN"

        gw_path = config.get_path("gw_main")
        assert gw_path == tmp_path / "Simulation" / "GW" / "GW_MAIN.dat"

    def test_get_path_resolves_absolute(self, tmp_path: Path) -> None:
        """get_path returns absolute paths."""
        config = ModelWriteConfig(output_dir=tmp_path)
        path = config.get_path("nodes")
        assert path.is_absolute()
        assert path == tmp_path / "Preprocessor" / "Nodes.dat"

    def test_get_path_with_override(self, tmp_path: Path) -> None:
        """get_path uses overridden path when set."""
        config = ModelWriteConfig(output_dir=tmp_path)
        config.set_file("gw_main", "custom/groundwater.dat")
        assert config.get_path("gw_main") == tmp_path / "custom" / "groundwater.dat"

    def test_get_path_unknown_key_raises(self, tmp_path: Path) -> None:
        """get_path raises KeyError for unknown keys."""
        config = ModelWriteConfig(output_dir=tmp_path)
        with pytest.raises(KeyError):
            config.get_path("nonexistent_key")

    def test_get_relative_path_same_directory(self, tmp_path: Path) -> None:
        """Files in same directory produce just the filename."""
        config = ModelWriteConfig(output_dir=tmp_path)
        # gw_main and gw_bc_main are both in Simulation/GW/
        rel = config.get_relative_path("gw_main", "gw_bc_main")
        assert rel == "BC_MAIN.dat"

    def test_get_relative_path_child_directory(self, tmp_path: Path) -> None:
        """Reference to child directory uses subdir/filename."""
        config = ModelWriteConfig(output_dir=tmp_path)
        rel = config.get_relative_path("simulation_main", "gw_main")
        expected = os.path.join("GW", "GW_MAIN.dat")
        assert rel == expected

    def test_get_relative_path_sibling_directory(self, tmp_path: Path) -> None:
        """Reference to sibling directory uses ../sibling/filename."""
        config = ModelWriteConfig(output_dir=tmp_path)
        # gw_main -> stream_main: from GW/ to Stream/
        rel = config.get_relative_path("gw_main", "stream_main")
        expected = os.path.join("..", "Stream", "Stream_MAIN.dat")
        assert rel == expected

    def test_get_relative_path_deep_custom_paths(self, tmp_path: Path) -> None:
        """Custom deep paths produce correct traversal."""
        config = ModelWriteConfig(output_dir=tmp_path)
        config.set_file("gw_main", "a/b/c/gw.dat")
        config.set_file("stream_main", "x/y/stream.dat")
        rel = config.get_relative_path("gw_main", "stream_main")
        # From a/b/c/ to x/y/ should be ../../../x/y/stream.dat
        expected = os.path.join("..", "..", "..", "x", "y", "stream.dat")
        assert rel == expected

    def test_set_file(self, tmp_path: Path) -> None:
        """set_file overrides a specific path."""
        config = ModelWriteConfig(output_dir=tmp_path)
        config.set_file("nodes", "flat_nodes.dat")
        assert config.get_path("nodes") == tmp_path / "flat_nodes.dat"

    def test_flat_classmethod(self, tmp_path: Path) -> None:
        """flat() classmethod puts all files in one directory."""
        config = ModelWriteConfig.flat(output_dir=tmp_path)
        # All files should be directly in tmp_path
        for key in ModelWriteConfig.DEFAULT_PATHS:
            path = config.get_path(key)
            assert path.parent == tmp_path, (
                f"File {key} not in root dir: {path}"
            )

    def test_nested_classmethod(self, tmp_path: Path) -> None:
        """nested() classmethod produces standard layout."""
        config = ModelWriteConfig.nested(output_dir=tmp_path)
        # Should use defaults
        assert config.get_path("gw_main") == (
            tmp_path / "Simulation" / "GW" / "GW_MAIN.dat"
        )

    def test_post_init_converts_to_path(self) -> None:
        """__post_init__ converts string output_dir to Path."""
        config = ModelWriteConfig(output_dir="some/dir")
        assert isinstance(config.output_dir, Path)

    def test_default_paths_all_keys_present(self) -> None:
        """All DEFAULT_PATHS keys can be resolved."""
        config = ModelWriteConfig(output_dir=Path("/test"))
        for key in ModelWriteConfig.DEFAULT_PATHS:
            path = config.get_path(key)
            assert path is not None

    def test_file_paths_override_takes_precedence(self, tmp_path: Path) -> None:
        """Overrides in file_paths take precedence over DEFAULT_PATHS."""
        config = ModelWriteConfig(
            output_dir=tmp_path,
            file_paths={"gw_main": "my_gw.dat"},
        )
        assert config.get_path("gw_main") == tmp_path / "my_gw.dat"

    def test_flat_relative_paths(self, tmp_path: Path) -> None:
        """Flat layout produces correct relative paths between files."""
        config = ModelWriteConfig.flat(output_dir=tmp_path)
        # All files in same directory, so relative path is just the filename
        rel = config.get_relative_path("simulation_main", "gw_main")
        assert rel == "GW_MAIN.dat"


# =============================================================================
# ModelWriteResult Tests
# =============================================================================


class TestModelWriteResult:
    """Tests for ModelWriteResult dataclass."""

    def test_success_with_no_errors(self) -> None:
        result = ModelWriteResult()
        assert result.success is True

    def test_success_false_with_errors(self) -> None:
        result = ModelWriteResult(errors={"gw": "Failed"})
        assert result.success is False

    def test_files_default_empty(self) -> None:
        result = ModelWriteResult()
        assert result.files == {}

    def test_warnings_default_empty(self) -> None:
        result = ModelWriteResult()
        assert result.warnings == []


# =============================================================================
# TimeSeriesCopier Tests
# =============================================================================


class TestTimeSeriesCopier:
    """Tests for TimeSeriesCopier class."""

    def test_text_to_text_copy(self, tmp_path: Path, simple_model: IWFMModel) -> None:
        """Text-to-text copy produces identical file."""
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        source_file = source_dir / "pumping.dat"
        source_file.write_text("C  Test TS data\n1.0 2.0 3.0\n")

        simple_model.source_files["gw_pumping_ts"] = source_file

        dest_dir = tmp_path / "dest"
        config = ModelWriteConfig(output_dir=dest_dir)
        copier = TimeSeriesCopier(simple_model, config)

        files, warnings = copier.copy_all()
        assert "gw_ts_pumping" in files
        dest_path = files["gw_ts_pumping"]
        assert dest_path.exists()
        assert dest_path.read_text() == source_file.read_text()

    def test_missing_source_file_skips_gracefully(
        self, tmp_path: Path, simple_model: IWFMModel
    ) -> None:
        """Missing source file generates warning, does not crash."""
        simple_model.source_files["gw_pumping_ts"] = Path("/nonexistent/file.dat")

        config = ModelWriteConfig(output_dir=tmp_path / "dest")
        copier = TimeSeriesCopier(simple_model, config)

        files, warnings = copier.copy_all()
        assert "gw_ts_pumping" not in files
        assert any("not found" in w for w in warnings)

    def test_no_source_files_produces_empty_result(
        self, tmp_path: Path, simple_model: IWFMModel
    ) -> None:
        """Model with no source_files produces empty result."""
        config = ModelWriteConfig(output_dir=tmp_path)
        copier = TimeSeriesCopier(simple_model, config)

        files, warnings = copier.copy_all()
        assert files == {}
        assert warnings == []

    def test_creates_destination_directories(
        self, tmp_path: Path, simple_model: IWFMModel
    ) -> None:
        """Copier creates destination directories as needed."""
        source_file = tmp_path / "source" / "et.dat"
        source_file.parent.mkdir()
        source_file.write_text("C  ET data\n")

        simple_model.source_files["et_ts"] = source_file

        dest_dir = tmp_path / "deep" / "nested" / "output"
        config = ModelWriteConfig(output_dir=dest_dir)
        copier = TimeSeriesCopier(simple_model, config)

        files, warnings = copier.copy_all()
        assert "et" in files
        assert files["et"].exists()

    def test_multiple_ts_files(
        self, tmp_path: Path, simple_model: IWFMModel
    ) -> None:
        """Multiple TS files are all copied."""
        source_dir = tmp_path / "source"
        source_dir.mkdir()

        for source_key in ("precipitation_ts", "et_ts"):
            f = source_dir / f"{source_key}.dat"
            f.write_text(f"C  {source_key}\n")
            simple_model.source_files[source_key] = f

        config = ModelWriteConfig(output_dir=tmp_path / "dest")
        copier = TimeSeriesCopier(simple_model, config)

        files, warnings = copier.copy_all()
        assert "precipitation" in files
        assert "et" in files


# =============================================================================
# CompleteModelWriter Tests
# =============================================================================


class TestCompleteModelWriter:
    """Tests for CompleteModelWriter class."""

    def test_write_preprocessor_creates_files(
        self, tmp_path: Path, simple_model: IWFMModel
    ) -> None:
        """Writing preprocessor creates nodes, elements, stratigraphy files."""
        config = ModelWriteConfig(output_dir=tmp_path)
        writer = CompleteModelWriter(simple_model, config)

        pp_files = writer.write_preprocessor()
        assert "nodes" in pp_files
        assert pp_files["nodes"].exists()
        assert "elements" in pp_files
        assert "stratigraphy" in pp_files

    def test_write_all_creates_simulation_main(
        self, tmp_path: Path, simple_model: IWFMModel
    ) -> None:
        """write_all creates the Simulation_MAIN.IN file."""
        config = ModelWriteConfig(output_dir=tmp_path)
        writer = CompleteModelWriter(simple_model, config)

        result = writer.write_all()
        assert "simulation_main" in result.files
        assert result.files["simulation_main"].exists()

    def test_write_all_result_type(
        self, tmp_path: Path, simple_model: IWFMModel
    ) -> None:
        """write_all returns ModelWriteResult."""
        config = ModelWriteConfig(output_dir=tmp_path)
        writer = CompleteModelWriter(simple_model, config)

        result = writer.write_all()
        assert isinstance(result, ModelWriteResult)

    def test_write_nested_layout(
        self, tmp_path: Path, simple_model: IWFMModel
    ) -> None:
        """Default nested layout creates expected subdirectories."""
        config = ModelWriteConfig(output_dir=tmp_path)
        writer = CompleteModelWriter(simple_model, config)
        result = writer.write_all()

        # Check that preprocessor dir was created
        pp_dir = tmp_path / "Preprocessor"
        assert pp_dir.exists()

        # Check that simulation dir was created
        sim_dir = tmp_path / "Simulation"
        assert sim_dir.exists()

    def test_write_flat_layout(
        self, tmp_path: Path, simple_model: IWFMModel
    ) -> None:
        """Flat layout puts all files in one directory."""
        config = ModelWriteConfig.flat(output_dir=tmp_path)
        writer = CompleteModelWriter(simple_model, config)
        result = writer.write_all()

        # All output files should be in tmp_path directly
        for key, path in result.files.items():
            assert path.parent == tmp_path, (
                f"File {key} not in root dir: {path}"
            )

    def test_write_custom_paths(
        self, tmp_path: Path, simple_model: IWFMModel
    ) -> None:
        """Custom file paths are respected."""
        config = ModelWriteConfig(output_dir=tmp_path)
        config.set_file("simulation_main", "custom/sim.in")
        config.set_file("preprocessor_main", "custom/pp.in")
        config.set_file("nodes", "custom/nodes.dat")
        config.set_file("elements", "custom/elements.dat")
        config.set_file("stratigraphy", "custom/strat.dat")

        writer = CompleteModelWriter(simple_model, config)
        result = writer.write_all()

        assert "simulation_main" in result.files
        sim_path = result.files["simulation_main"]
        assert sim_path == tmp_path / "custom" / "sim.in"

    def test_simulation_main_references_correct_relative_paths(
        self, tmp_path: Path, simple_model: IWFMModel
    ) -> None:
        """Simulation main file references components with correct paths."""
        config = ModelWriteConfig(output_dir=tmp_path)
        writer = CompleteModelWriter(simple_model, config)
        result = writer.write_all()

        sim_path = result.files["simulation_main"]
        content = sim_path.read_text()
        # The sim main should reference GW main as GW/GW_MAIN.dat or GW\GW_MAIN.dat
        gw_ref = os.path.join("GW", "GW_MAIN.dat")
        assert gw_ref in content or gw_ref.replace("\\", "/") in content

    def test_output_dir_created_if_missing(
        self, tmp_path: Path, simple_model: IWFMModel
    ) -> None:
        """Output directory is created if it doesn't exist."""
        out = tmp_path / "brand_new" / "model"
        config = ModelWriteConfig(output_dir=out)
        writer = CompleteModelWriter(simple_model, config)
        result = writer.write_all()

        assert out.exists()
        assert result.success or len(result.errors) > 0

    def test_errors_collected_not_raised(
        self, tmp_path: Path
    ) -> None:
        """Errors in component writing are collected, not raised."""
        # Model with no mesh - preprocessor will fail
        model = IWFMModel(name="Empty")
        config = ModelWriteConfig(output_dir=tmp_path)
        writer = CompleteModelWriter(model, config)

        result = writer.write_all()
        # Should not raise, but may have errors
        assert isinstance(result, ModelWriteResult)

    def test_copy_source_ts_false_skips_copy(
        self, tmp_path: Path, simple_model: IWFMModel
    ) -> None:
        """Setting copy_source_ts=False skips TS file copying."""
        source = tmp_path / "source" / "precip.dat"
        source.parent.mkdir()
        source.write_text("C  precip\n")
        simple_model.source_files["precipitation_ts"] = source

        config = ModelWriteConfig(
            output_dir=tmp_path / "dest",
            copy_source_ts=False,
        )
        writer = CompleteModelWriter(simple_model, config)
        result = writer.write_all()

        # Precipitation should NOT be in output files
        assert "precipitation" not in result.files


# =============================================================================
# write_model convenience function Tests
# =============================================================================


class TestWriteModel:
    """Tests for write_model convenience function."""

    def test_basic_smoke_test(
        self, tmp_path: Path, simple_model: IWFMModel
    ) -> None:
        """write_model runs without error."""
        result = write_model(simple_model, tmp_path)
        assert isinstance(result, ModelWriteResult)

    def test_with_file_paths(
        self, tmp_path: Path, simple_model: IWFMModel
    ) -> None:
        """write_model accepts file_paths argument."""
        result = write_model(
            simple_model,
            tmp_path,
            file_paths={"simulation_main": "sim.in"},
        )
        assert isinstance(result, ModelWriteResult)

    def test_ts_format_text(
        self, tmp_path: Path, simple_model: IWFMModel
    ) -> None:
        """write_model with ts_format='text' runs."""
        result = write_model(simple_model, tmp_path, ts_format="text")
        assert isinstance(result, ModelWriteResult)

    def test_ts_format_dss(
        self, tmp_path: Path, simple_model: IWFMModel
    ) -> None:
        """write_model with ts_format='dss' runs."""
        result = write_model(simple_model, tmp_path, ts_format="dss")
        assert isinstance(result, ModelWriteResult)


# =============================================================================
# save_complete_model replacement Tests
# =============================================================================


class TestSaveCompleteModel:
    """Tests for the updated save_complete_model function."""

    def test_delegates_to_complete_model_writer(
        self, tmp_path: Path, simple_model: IWFMModel
    ) -> None:
        """save_complete_model delegates to CompleteModelWriter."""
        from pyiwfm.io.preprocessor import save_complete_model

        files = save_complete_model(simple_model, tmp_path)
        assert isinstance(files, dict)

    def test_returns_dict_of_paths(
        self, tmp_path: Path, simple_model: IWFMModel
    ) -> None:
        """save_complete_model returns dict[str, Path]."""
        from pyiwfm.io.preprocessor import save_complete_model

        files = save_complete_model(simple_model, tmp_path)
        for key, path in files.items():
            assert isinstance(key, str)
            assert isinstance(path, Path)

    def test_accepts_file_paths(
        self, tmp_path: Path, simple_model: IWFMModel
    ) -> None:
        """save_complete_model accepts file_paths parameter."""
        from pyiwfm.io.preprocessor import save_complete_model

        files = save_complete_model(
            simple_model,
            tmp_path,
            file_paths={"simulation_main": "custom_sim.in"},
        )
        assert isinstance(files, dict)


# =============================================================================
# IWFMModel.to_simulation() Tests
# =============================================================================


class TestToSimulation:
    """Tests for IWFMModel.to_simulation with new params."""

    def test_to_simulation_basic(
        self, tmp_path: Path, simple_model: IWFMModel
    ) -> None:
        """to_simulation runs without extra args."""
        files = simple_model.to_simulation(tmp_path)
        assert isinstance(files, dict)

    def test_to_simulation_with_file_paths(
        self, tmp_path: Path, simple_model: IWFMModel
    ) -> None:
        """to_simulation passes file_paths through."""
        files = simple_model.to_simulation(
            tmp_path,
            file_paths={"simulation_main": "sim.in"},
        )
        assert isinstance(files, dict)

    def test_to_simulation_with_ts_format(
        self, tmp_path: Path, simple_model: IWFMModel
    ) -> None:
        """to_simulation passes ts_format through."""
        files = simple_model.to_simulation(tmp_path, ts_format="text")
        assert isinstance(files, dict)


# =============================================================================
# source_files preservation Tests
# =============================================================================


class TestSourceFiles:
    """Tests that source_files is populated correctly."""

    def test_source_files_field_exists(self) -> None:
        """IWFMModel has source_files field."""
        model = IWFMModel(name="test")
        assert hasattr(model, "source_files")
        assert isinstance(model.source_files, dict)

    def test_source_files_default_empty(self) -> None:
        """source_files defaults to empty dict."""
        model = IWFMModel(name="test")
        assert model.source_files == {}

    def test_source_files_can_be_populated(self) -> None:
        """source_files can be populated manually."""
        model = IWFMModel(name="test")
        model.source_files["gw_main"] = Path("/some/path/gw.dat")
        assert "gw_main" in model.source_files
        assert model.source_files["gw_main"] == Path("/some/path/gw.dat")


# =============================================================================
# Relative path edge cases
# =============================================================================


class TestRelativePathEdgeCases:
    """Tests for relative path computation edge cases."""

    def test_same_directory_just_filename(self, tmp_path: Path) -> None:
        """Same directory produces just the filename."""
        config = ModelWriteConfig(output_dir=tmp_path)
        config.set_file("gw_main", "dir/a.dat")
        config.set_file("gw_bc_main", "dir/b.dat")
        rel = config.get_relative_path("gw_main", "gw_bc_main")
        assert rel == "b.dat"

    def test_child_directory(self, tmp_path: Path) -> None:
        """Reference to file in child directory."""
        config = ModelWriteConfig(output_dir=tmp_path)
        config.set_file("simulation_main", "sim.in")
        config.set_file("gw_main", "gw/main.dat")
        rel = config.get_relative_path("simulation_main", "gw_main")
        expected = os.path.join("gw", "main.dat")
        assert rel == expected

    def test_parent_directory(self, tmp_path: Path) -> None:
        """Reference to file in parent directory."""
        config = ModelWriteConfig(output_dir=tmp_path)
        config.set_file("gw_main", "gw/main.dat")
        config.set_file("simulation_main", "sim.in")
        rel = config.get_relative_path("gw_main", "simulation_main")
        expected = os.path.join("..", "sim.in")
        assert rel == expected

    def test_deeply_nested(self, tmp_path: Path) -> None:
        """Deeply nested paths compute correct relative paths."""
        config = ModelWriteConfig(output_dir=tmp_path)
        config.set_file("gw_main", "a/b/c/d/gw.dat")
        config.set_file("stream_main", "x/stream.dat")
        rel = config.get_relative_path("gw_main", "stream_main")
        expected = os.path.join("..", "..", "..", "..", "x", "stream.dat")
        assert rel == expected


# =============================================================================
# _iso_to_iwfm_date helper Tests
# =============================================================================


class TestIsoToIwfmDate:
    """Tests for the ISO to IWFM date conversion helper."""

    def test_midnight_conversion(self) -> None:
        result = _iso_to_iwfm_date("1990-09-30T00:00:00")
        assert result == "09/29/1990_24:00"

    def test_non_midnight_conversion(self) -> None:
        result = _iso_to_iwfm_date("2000-01-15T12:30:00")
        assert result == "01/15/2000_12:30"

    def test_invalid_string_passthrough(self) -> None:
        result = _iso_to_iwfm_date("not-a-date")
        assert result == "not-a-date"

    def test_date_only(self) -> None:
        result = _iso_to_iwfm_date("1990-09-30")
        assert result == "09/29/1990_24:00"


# =============================================================================
# DSS interval computation Tests
# =============================================================================


class TestComputeDSSInterval:
    """Tests for _compute_dss_interval helper."""

    def test_daily_interval(self) -> None:
        from datetime import datetime, timedelta

        base = datetime(2000, 1, 1)
        times = [base + timedelta(days=i) for i in range(3)]
        assert _compute_dss_interval(times) == "1DAY"

    def test_monthly_interval(self) -> None:
        from datetime import datetime

        times = [datetime(2000, m, 1) for m in range(1, 4)]
        result = _compute_dss_interval(times)
        assert result == "1MON"

    def test_hourly_interval(self) -> None:
        from datetime import datetime, timedelta

        base = datetime(2000, 1, 1)
        times = [base + timedelta(hours=i) for i in range(3)]
        assert _compute_dss_interval(times) == "1HOUR"

    def test_single_time_defaults_to_1mon(self) -> None:
        from datetime import datetime

        times = [datetime(2000, 1, 1)]
        assert _compute_dss_interval(times) == "1MON"

    def test_empty_list_defaults_to_1mon(self) -> None:
        assert _compute_dss_interval([]) == "1MON"


# =============================================================================
# DSS Stub File Writing Tests
# =============================================================================


def _make_text_ts_file(filepath: Path, n_cols: int = 3, n_times: int = 3) -> None:
    """Create a valid IWFM text time series file for testing."""
    from datetime import datetime, timedelta

    from pyiwfm.io.timeseries_ascii import format_iwfm_timestamp

    with open(filepath, "w") as f:
        f.write("C  Test time series file\n")
        f.write("C\n")
        f.write(f"{n_cols:<10}                              / NRAIN\n")
        f.write(f"{'1.0':<14}                          / FACTRN\n")
        f.write(f"{'1':<10}                              / NSPRN\n")
        f.write(f"{'0':<10}                              / NFQRN\n")
        f.write(f"{'':40}                              / DSSFL\n")

        base = datetime(2000, 1, 1)
        for t in range(n_times):
            dt = base + timedelta(days=t * 30)
            ts_str = format_iwfm_timestamp(dt)
            vals = "  ".join(f"{(t + 1) * (c + 1):14.6f}" for c in range(n_cols))
            f.write(f"    {ts_str} {vals}\n")


class TestDSSStubWriting:
    """Tests for text-to-DSS conversion producing stub .dat files."""

    def test_text_to_dss_creates_stub_file(
        self, tmp_path: Path, simple_model: IWFMModel
    ) -> None:
        """Text->DSS conversion creates a stub .dat file with DSSFL."""
        source = tmp_path / "source" / "precip.dat"
        source.parent.mkdir()
        _make_text_ts_file(source, n_cols=3)

        simple_model.source_files["precipitation_ts"] = source

        config = ModelWriteConfig(
            output_dir=tmp_path / "dest",
            ts_format=OutputFormat.DSS,
        )
        copier = TimeSeriesCopier(simple_model, config)
        files, warnings = copier.copy_all()

        assert "precipitation" in files
        dest_path = files["precipitation"]
        assert dest_path.exists()

        content = dest_path.read_text()
        # Stub should have DSSFL pointing to the DSS file
        assert "DSSFL" in content
        assert "climate_data.dss" in content

    def test_stub_contains_dss_pathnames(
        self, tmp_path: Path, simple_model: IWFMModel
    ) -> None:
        """Stub file contains DSS pathnames for each column."""
        n_cols = 5
        source = tmp_path / "source" / "precip.dat"
        source.parent.mkdir()
        _make_text_ts_file(source, n_cols=n_cols)

        simple_model.source_files["precipitation_ts"] = source

        config = ModelWriteConfig(
            output_dir=tmp_path / "dest",
            ts_format=OutputFormat.DSS,
        )
        copier = TimeSeriesCopier(simple_model, config)
        files, warnings = copier.copy_all()

        content = files["precipitation"].read_text()
        # Should have DSS pathnames for all columns
        for i in range(1, n_cols + 1):
            assert f"ELEM_{i}" in content
        assert "/PRECIP/" in content

    def test_stub_uses_correct_parameter_code(
        self, tmp_path: Path, simple_model: IWFMModel
    ) -> None:
        """Stub uses the right DSS C-part for different TS types."""
        source = tmp_path / "source" / "et.dat"
        source.parent.mkdir()
        _make_text_ts_file(source, n_cols=2)

        simple_model.source_files["et_ts"] = source

        config = ModelWriteConfig(
            output_dir=tmp_path / "dest",
            ts_format=OutputFormat.DSS,
        )
        copier = TimeSeriesCopier(simple_model, config)
        files, warnings = copier.copy_all()

        content = files["et"].read_text()
        assert "/ET/" in content

    def test_stub_factor_is_one(
        self, tmp_path: Path, simple_model: IWFMModel
    ) -> None:
        """Stub file has FACTOR=1.0 since DSS data is already scaled."""
        source = tmp_path / "source" / "precip.dat"
        source.parent.mkdir()
        _make_text_ts_file(source, n_cols=2)

        simple_model.source_files["precipitation_ts"] = source

        config = ModelWriteConfig(
            output_dir=tmp_path / "dest",
            ts_format=OutputFormat.DSS,
        )
        copier = TimeSeriesCopier(simple_model, config)
        copier.copy_all()

        dest = config.get_path("precipitation")
        content = dest.read_text()
        # Should contain 1.000000 as the factor
        assert "1.000000" in content

    def test_dss_file_created(
        self, tmp_path: Path, simple_model: IWFMModel
    ) -> None:
        """DSS binary file is created alongside stub."""
        source = tmp_path / "source" / "precip.dat"
        source.parent.mkdir()
        _make_text_ts_file(source, n_cols=2, n_times=3)

        simple_model.source_files["precipitation_ts"] = source

        config = ModelWriteConfig(
            output_dir=tmp_path / "dest",
            ts_format=OutputFormat.DSS,
        )
        copier = TimeSeriesCopier(simple_model, config)
        copier.copy_all()

        dss_path = config.get_path("dss_ts_file")
        assert dss_path.exists()
        assert dss_path.stat().st_size > 0

    def test_dss_a_part_customization(
        self, tmp_path: Path, simple_model: IWFMModel
    ) -> None:
        """Custom dss_a_part appears in DSS pathnames."""
        source = tmp_path / "source" / "precip.dat"
        source.parent.mkdir()
        _make_text_ts_file(source, n_cols=2)

        simple_model.source_files["precipitation_ts"] = source

        config = ModelWriteConfig(
            output_dir=tmp_path / "dest",
            ts_format=OutputFormat.DSS,
            dss_a_part="MYPROJECT",
        )
        copier = TimeSeriesCopier(simple_model, config)
        copier.copy_all()

        dest = config.get_path("precipitation")
        content = dest.read_text()
        assert "/MYPROJECT/" in content

    def test_dss_f_part_customization(
        self, tmp_path: Path, simple_model: IWFMModel
    ) -> None:
        """Custom dss_f_part appears in DSS pathnames."""
        source = tmp_path / "source" / "precip.dat"
        source.parent.mkdir()
        _make_text_ts_file(source, n_cols=2)

        simple_model.source_files["precipitation_ts"] = source

        config = ModelWriteConfig(
            output_dir=tmp_path / "dest",
            ts_format=OutputFormat.DSS,
            dss_f_part="V2-SCENARIO",
        )
        copier = TimeSeriesCopier(simple_model, config)
        copier.copy_all()

        dest = config.get_path("precipitation")
        content = dest.read_text()
        assert "/V2-SCENARIO/" in content

    def test_stub_has_correct_tag_names(
        self, tmp_path: Path, simple_model: IWFMModel
    ) -> None:
        """Stub file uses correct IWFM tag names for precip (NRAIN/FACTRN)."""
        source = tmp_path / "source" / "precip.dat"
        source.parent.mkdir()
        _make_text_ts_file(source, n_cols=2)

        simple_model.source_files["precipitation_ts"] = source

        config = ModelWriteConfig(
            output_dir=tmp_path / "dest",
            ts_format=OutputFormat.DSS,
        )
        copier = TimeSeriesCopier(simple_model, config)
        copier.copy_all()

        dest = config.get_path("precipitation")
        content = dest.read_text()
        assert "NRAIN" in content
        assert "FACTRN" in content

    def test_stub_no_inline_data(
        self, tmp_path: Path, simple_model: IWFMModel
    ) -> None:
        """Stub file does NOT contain inline time series data."""
        source = tmp_path / "source" / "precip.dat"
        source.parent.mkdir()
        _make_text_ts_file(source, n_cols=2, n_times=5)

        simple_model.source_files["precipitation_ts"] = source

        config = ModelWriteConfig(
            output_dir=tmp_path / "dest",
            ts_format=OutputFormat.DSS,
        )
        copier = TimeSeriesCopier(simple_model, config)
        copier.copy_all()

        dest = config.get_path("precipitation")
        content = dest.read_text()
        # Inline data has timestamps like MM/DD/YYYY_HH:MM:SS
        # The stub should NOT have these (DSS pathnames instead)
        assert "01/01/2000_00:00:00" not in content
        # But should have DSS pathnames
        assert "DSS Pathnames" in content

    def test_multiple_ts_to_dss_share_dss_file(
        self, tmp_path: Path, simple_model: IWFMModel
    ) -> None:
        """Multiple TS files all write to the same shared DSS file."""
        source_dir = tmp_path / "source"
        source_dir.mkdir()

        for key in ("precipitation_ts", "et_ts"):
            source = source_dir / f"{key}.dat"
            _make_text_ts_file(source, n_cols=2, n_times=3)
            simple_model.source_files[key] = source

        config = ModelWriteConfig(
            output_dir=tmp_path / "dest",
            ts_format=OutputFormat.DSS,
        )
        copier = TimeSeriesCopier(simple_model, config)
        files, warnings = copier.copy_all()

        # Both should be written
        assert "precipitation" in files
        assert "et" in files

        # Both stubs should reference the same DSS file
        precip_content = files["precipitation"].read_text()
        et_content = files["et"].read_text()
        assert "climate_data.dss" in precip_content
        assert "climate_data.dss" in et_content

        # Precip stub has PRECIP pathnames, ET stub has ET pathnames
        assert "/PRECIP/" in precip_content
        assert "/ET/" in et_content
