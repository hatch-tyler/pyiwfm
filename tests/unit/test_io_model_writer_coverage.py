"""Comprehensive tests for io/model_writer.py.

Covers:
- ModelWriteResult dataclass and success property
- TimeSeriesCopier: copy_all, _copy_or_convert, _convert_text_to_dss
- CompleteModelWriter: write_all, component writes, comment metadata
- Convenience functions: _iso_to_iwfm_date, write_model, save_model_with_comments
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pyiwfm.io.config import ModelWriteConfig, OutputFormat
from pyiwfm.io.model_writer import (
    TS_KEY_MAPPING,
    CompleteModelWriter,
    ModelWriteResult,
    TimeSeriesCopier,
    _iso_to_iwfm_date,
    save_model_with_comments,
    write_model,
    write_model_with_comments,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_model(**overrides: object) -> MagicMock:
    """Build a minimal mock IWFMModel with common attributes."""
    model = MagicMock()
    model.groundwater = overrides.get("groundwater", None)
    model.streams = overrides.get("streams", None)
    model.lakes = overrides.get("lakes", None)
    model.rootzone = overrides.get("rootzone", None)
    model.small_watersheds = overrides.get("small_watersheds", None)
    model.unsaturated_zone = overrides.get("unsaturated_zone", None)
    model.supply_adjustment = overrides.get("supply_adjustment", None)
    model.source_files = overrides.get("source_files", {})
    model.metadata = overrides.get("metadata", {})
    return model


def _make_config(tmp_path: Path, **kwargs: object) -> ModelWriteConfig:
    """Build a ModelWriteConfig rooted in *tmp_path*."""
    return ModelWriteConfig(output_dir=tmp_path, **kwargs)


# ===========================================================================
# 5A. ModelWriteResult
# ===========================================================================


class TestModelWriteResult:
    """Tests for the ModelWriteResult dataclass."""

    def test_success_true_when_no_errors(self) -> None:
        result = ModelWriteResult()
        assert result.success is True

    def test_success_false_when_errors_present(self) -> None:
        result = ModelWriteResult(errors={"groundwater": "write failed"})
        assert result.success is False

    def test_default_fields_are_empty(self) -> None:
        result = ModelWriteResult()
        assert result.files == {}
        assert result.errors == {}
        assert result.warnings == []


# ===========================================================================
# 5B. TimeSeriesCopier
# ===========================================================================


class TestTimeSeriesCopierCopyAll:
    """Tests for TimeSeriesCopier.copy_all()."""

    def test_copies_existing_source_files(self, tmp_path: Path) -> None:
        """copy_all copies source files that exist to destination paths."""
        # Create a source file for one TS key
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        source_file = src_dir / "precip.dat"
        source_file.write_text("data")

        model = _make_mock_model(source_files={"precipitation_ts": source_file})
        config = _make_config(tmp_path)

        copier = TimeSeriesCopier(model, config)

        with patch("pyiwfm.io.model_writer.shutil.copy2") as mock_copy:
            files, warnings = copier.copy_all()

        dest_key = TS_KEY_MAPPING["precipitation_ts"]
        expected_dest = config.get_path(dest_key)
        mock_copy.assert_called_once_with(source_file, expected_dest)
        assert dest_key in files
        assert len(warnings) == 0

    def test_skips_when_source_path_is_none(self, tmp_path: Path) -> None:
        """copy_all skips keys whose source_files value is None."""
        model = _make_mock_model(source_files={"precipitation_ts": None})
        config = _make_config(tmp_path)
        copier = TimeSeriesCopier(model, config)

        with patch("pyiwfm.io.model_writer.shutil.copy2") as mock_copy:
            files, warnings = copier.copy_all()

        mock_copy.assert_not_called()
        assert len(files) == 0
        assert len(warnings) == 0

    def test_warns_when_source_file_missing(self, tmp_path: Path) -> None:
        """copy_all warns when the source path does not exist on disk."""
        missing = tmp_path / "does_not_exist.dat"
        model = _make_mock_model(source_files={"precipitation_ts": missing})
        config = _make_config(tmp_path)
        copier = TimeSeriesCopier(model, config)

        files, warnings = copier.copy_all()

        assert len(files) == 0
        assert len(warnings) == 1
        assert "not found" in warnings[0]

    def test_warns_when_dest_key_missing(self, tmp_path: Path) -> None:
        """copy_all warns when config.get_path raises KeyError."""
        src_file = tmp_path / "precip.dat"
        src_file.write_text("data")

        model = _make_mock_model(source_files={"precipitation_ts": src_file})
        config = _make_config(tmp_path)

        # Force KeyError from get_path for the dest key
        original_get_path = config.get_path

        def patched_get_path(key: str) -> Path:
            if key == "precipitation":
                raise KeyError(key)
            return original_get_path(key)

        config.get_path = patched_get_path  # type: ignore[assignment]

        copier = TimeSeriesCopier(model, config)
        files, warnings = copier.copy_all()

        assert len(files) == 0
        assert len(warnings) == 1
        assert "No destination key" in warnings[0]

    def test_warns_on_copy_exception(self, tmp_path: Path) -> None:
        """copy_all records a warning when the copy itself fails."""
        src_file = tmp_path / "precip.dat"
        src_file.write_text("data")

        model = _make_mock_model(source_files={"precipitation_ts": src_file})
        config = _make_config(tmp_path)

        copier = TimeSeriesCopier(model, config)

        with patch(
            "pyiwfm.io.model_writer.shutil.copy2",
            side_effect=OSError("disk full"),
        ):
            files, warnings = copier.copy_all()

        assert len(files) == 0
        assert len(warnings) == 1
        assert "Failed to copy" in warnings[0]


class TestTimeSeriesCopierCopyOrConvert:
    """Tests for TimeSeriesCopier._copy_or_convert()."""

    def test_text_to_text_uses_copy2(self, tmp_path: Path) -> None:
        """Text source + TEXT format config -> shutil.copy2."""
        model = _make_mock_model()
        config = _make_config(tmp_path, ts_format=OutputFormat.TEXT)
        copier = TimeSeriesCopier(model, config)

        src = tmp_path / "input.dat"
        src.write_text("data")
        dest = tmp_path / "output.dat"

        with patch("pyiwfm.io.model_writer.shutil.copy2") as mock_copy:
            copier._copy_or_convert(src, dest, "precipitation_ts")

        mock_copy.assert_called_once_with(src, dest)

    def test_dss_to_dss_uses_copy2(self, tmp_path: Path) -> None:
        """DSS source + DSS format config -> shutil.copy2 (same format)."""
        model = _make_mock_model()
        config = _make_config(tmp_path, ts_format=OutputFormat.DSS)
        copier = TimeSeriesCopier(model, config)

        src = tmp_path / "input.dss"
        src.write_text("dss data")
        dest = tmp_path / "output.dss"

        with patch("pyiwfm.io.model_writer.shutil.copy2") as mock_copy:
            copier._copy_or_convert(src, dest, "precipitation_ts")

        mock_copy.assert_called_once_with(src, dest)

    def test_text_to_dss_calls_convert(self, tmp_path: Path) -> None:
        """Text source + DSS format config -> _convert_text_to_dss."""
        model = _make_mock_model()
        config = _make_config(tmp_path, ts_format=OutputFormat.DSS)
        copier = TimeSeriesCopier(model, config)

        src = tmp_path / "input.dat"
        src.write_text("data")
        dest = tmp_path / "output.dat"

        with patch.object(copier, "_convert_text_to_dss") as mock_convert:
            copier._copy_or_convert(src, dest, "precipitation_ts")

        mock_convert.assert_called_once_with(src, dest, "precipitation_ts")


class TestTimeSeriesCopierConvertTextToDss:
    """Tests for TimeSeriesCopier._convert_text_to_dss()."""

    def test_convert_with_dss_available(self, tmp_path: Path) -> None:
        """When DSS library is importable, writes DSS data and stub file."""
        from datetime import datetime

        import numpy as np

        model = _make_mock_model()
        config = _make_config(tmp_path, ts_format=OutputFormat.DSS)
        copier = TimeSeriesCopier(model, config)

        src = tmp_path / "precip.dat"
        src.write_text("fake data")
        dest = tmp_path / "precip_stub.dat"

        times = [datetime(2020, 1, 1), datetime(2020, 2, 1)]
        values = np.array([[1.0], [2.0]])
        ts_config_val = MagicMock()

        mock_reader_instance = MagicMock()
        mock_reader_instance.read.return_value = (times, values, ts_config_val)

        mock_dss_writer_instance = MagicMock()
        mock_dss_writer_instance.__enter__ = MagicMock(return_value=mock_dss_writer_instance)
        mock_dss_writer_instance.__exit__ = MagicMock(return_value=False)

        mock_ts_writer_instance = MagicMock()

        with (
            patch(
                "pyiwfm.io.timeseries_ascii.TimeSeriesReader",
                return_value=mock_reader_instance,
            ),
            patch(
                "pyiwfm.io.dss.timeseries.DSSTimeSeriesWriter",
                return_value=mock_dss_writer_instance,
            ),
            patch(
                "pyiwfm.io.dss.pathname.DSSPathnameTemplate",
            ),
            patch(
                "pyiwfm.io.dss.pathname.minutes_to_interval",
                return_value="1MON",
            ),
            patch(
                "pyiwfm.io.timeseries_writer.IWFMTimeSeriesDataWriter",
                return_value=mock_ts_writer_instance,
            ),
        ):
            copier._convert_text_to_dss(src, dest, "precipitation_ts")

        # DSS writer should have been called
        mock_dss_writer_instance.write_multiple_timeseries.assert_called_once()
        # Stub writer should have been called
        mock_ts_writer_instance.write.assert_called_once()

    def test_convert_falls_back_on_import_error(self, tmp_path: Path) -> None:
        """When DSS library import fails, falls back to shutil.copy2."""
        from datetime import datetime

        import numpy as np

        model = _make_mock_model()
        config = _make_config(tmp_path, ts_format=OutputFormat.DSS)
        copier = TimeSeriesCopier(model, config)

        src = tmp_path / "precip.dat"
        src.write_text("fake data")
        dest = tmp_path / "precip_stub.dat"

        times = [datetime(2020, 1, 1), datetime(2020, 2, 1)]
        values = np.array([[1.0], [2.0]])
        ts_config_mock = MagicMock()

        mock_reader_instance = MagicMock()
        mock_reader_instance.read.return_value = (times, values, ts_config_mock)

        # Make DSS import fail inside _convert_text_to_dss by patching
        # the module so the local import raises ImportError.
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "pyiwfm.io.dss.timeseries":
                raise ImportError("no pyhecdss")
            return original_import(name, *args, **kwargs)

        with (
            patch(
                "pyiwfm.io.timeseries_ascii.TimeSeriesReader",
                return_value=mock_reader_instance,
            ),
            patch(
                "pyiwfm.io.dss.pathname.minutes_to_interval",
                return_value="1MON",
            ),
            patch("builtins.__import__", side_effect=mock_import),
            patch("pyiwfm.io.model_writer.shutil") as mock_shutil,
        ):
            copier._convert_text_to_dss(src, dest, "precipitation_ts")

        mock_shutil.copy2.assert_called_once_with(src, dest)


# ===========================================================================
# 5C. CompleteModelWriter
# ===========================================================================


class TestCompleteModelWriterWriteAll:
    """Tests for CompleteModelWriter.write_all()."""

    def test_creates_output_directory(self, tmp_path: Path) -> None:
        """write_all creates the output directory if it doesn't exist."""
        out_dir = tmp_path / "new_model"
        model = _make_mock_model()
        config = _make_config(out_dir)
        writer = CompleteModelWriter(model, config)

        with (
            patch.object(writer, "write_preprocessor", return_value={}),
            patch.object(writer, "_write_groundwater"),
            patch.object(writer, "_write_streams"),
            patch.object(writer, "_write_lakes"),
            patch.object(writer, "_write_rootzone"),
            patch.object(writer, "_copy_passthrough_components"),
            patch.object(writer, "_write_supply_adjustment"),
            patch.object(
                writer,
                "_write_simulation_main",
                return_value=out_dir / "sim.in",
            ),
        ):
            writer.write_all()

        assert out_dir.exists()

    def test_writes_preprocessor_files(self, tmp_path: Path) -> None:
        """write_all includes preprocessor files in result."""
        model = _make_mock_model()
        config = _make_config(tmp_path)
        writer = CompleteModelWriter(model, config)

        pp_files = {"nodes": tmp_path / "Nodes.dat"}

        with (
            patch.object(writer, "write_preprocessor", return_value=pp_files),
            patch.object(writer, "_write_groundwater"),
            patch.object(writer, "_write_streams"),
            patch.object(writer, "_write_lakes"),
            patch.object(writer, "_write_rootzone"),
            patch.object(writer, "_copy_passthrough_components"),
            patch.object(writer, "_write_supply_adjustment"),
            patch.object(
                writer,
                "_write_simulation_main",
                return_value=tmp_path / "sim.in",
            ),
        ):
            result = writer.write_all()

        assert "nodes" in result.files
        assert result.success is True

    def test_writes_gw_when_model_has_groundwater(self, tmp_path: Path) -> None:
        """write_all calls _write_groundwater when model.groundwater is set."""
        model = _make_mock_model(groundwater=MagicMock())
        config = _make_config(tmp_path)
        writer = CompleteModelWriter(model, config)

        with (
            patch.object(writer, "write_preprocessor", return_value={}),
            patch.object(writer, "_write_groundwater") as mock_gw,
            patch.object(writer, "_write_streams"),
            patch.object(writer, "_write_lakes"),
            patch.object(writer, "_write_rootzone"),
            patch.object(writer, "_copy_passthrough_components"),
            patch.object(writer, "_write_supply_adjustment"),
            patch.object(
                writer,
                "_write_simulation_main",
                return_value=tmp_path / "sim.in",
            ),
        ):
            writer.write_all()

        mock_gw.assert_called_once()

    def test_skips_gw_when_groundwater_is_none(self, tmp_path: Path) -> None:
        """_write_groundwater returns early when model.groundwater is None."""
        model = _make_mock_model(groundwater=None)
        config = _make_config(tmp_path)
        writer = CompleteModelWriter(model, config)

        result = ModelWriteResult()
        writer._write_groundwater(result)

        # No files written, no errors
        assert len(result.files) == 0
        assert len(result.errors) == 0

    def test_handles_component_write_errors_gracefully(self, tmp_path: Path) -> None:
        """write_all stores component errors in result.errors, not raising."""
        model = _make_mock_model()
        config = _make_config(tmp_path)
        writer = CompleteModelWriter(model, config)

        with (
            patch.object(
                writer,
                "write_preprocessor",
                side_effect=RuntimeError("PP failed"),
            ),
            patch.object(writer, "_write_groundwater"),
            patch.object(writer, "_write_streams"),
            patch.object(writer, "_write_lakes"),
            patch.object(writer, "_write_rootzone"),
            patch.object(writer, "_copy_passthrough_components"),
            patch.object(writer, "_write_supply_adjustment"),
            patch.object(
                writer,
                "_write_simulation_main",
                return_value=tmp_path / "sim.in",
            ),
        ):
            result = writer.write_all()

        assert "preprocessor" in result.errors
        assert "PP failed" in result.errors["preprocessor"]
        assert result.success is False

    def test_copies_timeseries_when_config_copy_source_ts(self, tmp_path: Path) -> None:
        """write_all copies time series when config.copy_source_ts is True."""
        model = _make_mock_model()
        config = _make_config(tmp_path, copy_source_ts=True)
        writer = CompleteModelWriter(model, config)

        with (
            patch.object(writer, "write_preprocessor", return_value={}),
            patch.object(writer, "_write_groundwater"),
            patch.object(writer, "_write_streams"),
            patch.object(writer, "_write_lakes"),
            patch.object(writer, "_write_rootzone"),
            patch.object(writer, "_copy_timeseries") as mock_ts,
            patch.object(writer, "_copy_passthrough_components"),
            patch.object(writer, "_write_supply_adjustment"),
            patch.object(
                writer,
                "_write_simulation_main",
                return_value=tmp_path / "sim.in",
            ),
        ):
            writer.write_all()

        mock_ts.assert_called_once()


class TestCompleteModelWriterSmallWatersheds:
    """Tests for CompleteModelWriter._write_small_watersheds()."""

    def test_write_with_component_data(self, tmp_path: Path) -> None:
        """Writes small watersheds from parsed component data."""
        model = _make_mock_model(
            small_watersheds=MagicMock(),
            metadata={"small_watershed_version": "4.1"},
            source_files={},
        )
        config = _make_config(tmp_path)
        writer = CompleteModelWriter(model, config)
        result = ModelWriteResult()

        mock_sw_writer = MagicMock()
        mock_sw_writer.write_all.return_value = {"main": config.get_path("swshed_main")}

        with patch(
            "pyiwfm.io.small_watershed_writer.SmallWatershedComponentWriter",
            return_value=mock_sw_writer,
        ):
            writer._write_small_watersheds(result)

        assert "swshed_main" in result.files

    def test_fallback_copy_when_writer_fails(self, tmp_path: Path) -> None:
        """Falls back to copy when the writer raises an exception."""
        # Create a source file to be copied
        src_file = tmp_path / "source_sw.dat"
        src_file.write_text("small watershed data")

        model = _make_mock_model(
            small_watersheds=MagicMock(),
            metadata={},
            source_files={"swshed_main": src_file},
        )
        config = _make_config(tmp_path)
        writer = CompleteModelWriter(model, config)
        result = ModelWriteResult()

        with patch(
            "pyiwfm.io.small_watershed_writer.SmallWatershedComponentWriter",
            side_effect=RuntimeError("writer broken"),
        ):
            writer._write_small_watersheds(result)

        # Should have a warning about failure + fallback copy
        assert any("falling back to copy" in w for w in result.warnings)
        assert "swshed_main" in result.files


class TestCompleteModelWriterUnsaturatedZone:
    """Tests for CompleteModelWriter._write_unsaturated_zone()."""

    def test_write_with_component_data(self, tmp_path: Path) -> None:
        """Writes unsaturated zone from parsed component data."""
        model = _make_mock_model(
            unsaturated_zone=MagicMock(),
            metadata={"unsat_zone_version": "4.0"},
            source_files={},
        )
        config = _make_config(tmp_path)
        writer = CompleteModelWriter(model, config)
        result = ModelWriteResult()

        mock_uz_writer = MagicMock()
        mock_uz_writer.write_all.return_value = {"main": config.get_path("unsatzone_main")}

        with patch(
            "pyiwfm.io.unsaturated_zone_writer.UnsatZoneComponentWriter",
            return_value=mock_uz_writer,
        ):
            writer._write_unsaturated_zone(result)

        assert "unsatzone_main" in result.files


class TestCompleteModelWriterSupplyAdjustment:
    """Tests for CompleteModelWriter._write_supply_adjustment()."""

    def test_write_from_parsed_data(self, tmp_path: Path) -> None:
        """Writes supply adjustment from parsed SupplyAdjustment data."""
        model = _make_mock_model(
            supply_adjustment=MagicMock(),
            source_files={},
        )
        config = _make_config(tmp_path)
        writer = CompleteModelWriter(model, config)
        result = ModelWriteResult()

        with patch("pyiwfm.io.supply_adjust.write_supply_adjustment") as mock_write_sa:
            writer._write_supply_adjustment(result)

        mock_write_sa.assert_called_once()
        assert "supply_adjust" in result.files

    def test_fallback_copy_from_source(self, tmp_path: Path) -> None:
        """Falls back to copying source file when no parsed data."""
        src_file = tmp_path / "supply_adjust_src.dat"
        src_file.write_text("supply adjust data")

        model = _make_mock_model(
            supply_adjustment=None,
            source_files={"supply_adjust": src_file},
        )
        config = _make_config(tmp_path)
        writer = CompleteModelWriter(model, config)
        result = ModelWriteResult()

        writer._write_supply_adjustment(result)

        assert "supply_adjust" in result.files
        dest = config.get_path("supply_adjust")
        assert dest.exists()


class TestCompleteModelWriterComments:
    """Tests for CompleteModelWriter.get_file_comments()."""

    def test_returns_metadata_when_preserve_true(self) -> None:
        """get_file_comments returns metadata when preserve_comments=True."""
        mock_comments = MagicMock()
        model = _make_mock_model()
        config = MagicMock(spec=ModelWriteConfig)
        writer = CompleteModelWriter(
            model,
            config,
            comment_metadata={"gw_main": mock_comments},
            preserve_comments=True,
        )

        result = writer.get_file_comments("gw_main")
        assert result is mock_comments

    def test_returns_none_when_preserve_false(self) -> None:
        """get_file_comments returns None when preserve_comments=False."""
        mock_comments = MagicMock()
        model = _make_mock_model()
        config = MagicMock(spec=ModelWriteConfig)
        writer = CompleteModelWriter(
            model,
            config,
            comment_metadata={"gw_main": mock_comments},
            preserve_comments=False,
        )

        result = writer.get_file_comments("gw_main")
        assert result is None


# ===========================================================================
# 5D. Convenience Functions
# ===========================================================================


class TestIsoToIwfmDate:
    """Tests for _iso_to_iwfm_date()."""

    def test_converts_iso_date(self) -> None:
        """Converts ISO date string to IWFM MM/DD/YYYY_HH:MM format."""
        with patch(
            "pyiwfm.io.timeseries_ascii.format_iwfm_timestamp",
            return_value="09/30/1990_24:00",
        ):
            result = _iso_to_iwfm_date("1990-10-01T00:00:00")

        assert result == "09/30/1990_24:00"

    def test_returns_original_on_invalid_input(self) -> None:
        """Returns original string when input is not valid ISO format."""
        result = _iso_to_iwfm_date("not-a-date")
        assert result == "not-a-date"


class TestWriteModelConvenience:
    """Tests for the write_model convenience function."""

    def test_creates_writer_and_calls_write_all(self, tmp_path: Path) -> None:
        """write_model creates config + writer, returns write_all result."""
        model = _make_mock_model(metadata={})

        mock_result = ModelWriteResult(files={"simulation_main": tmp_path / "sim.in"})

        with patch("pyiwfm.io.model_writer.CompleteModelWriter") as MockWriter:
            MockWriter.return_value.write_all.return_value = mock_result
            result = write_model(model, tmp_path)

        MockWriter.assert_called_once()
        assert result is mock_result
        assert result.success is True


class TestSaveModelWithComments:
    """Tests for save_model_with_comments."""

    def test_raises_runtime_error_on_failure(self, tmp_path: Path) -> None:
        """save_model_with_comments raises RuntimeError when write fails."""
        model = _make_mock_model(metadata={})

        failed_result = ModelWriteResult(
            errors={"preprocessor": "boom"},
        )

        with patch(
            "pyiwfm.io.model_writer.write_model_with_comments",
            return_value=failed_result,
        ):
            with pytest.raises(RuntimeError, match="Failed to write model"):
                save_model_with_comments(model, tmp_path)

    def test_returns_files_on_success(self, tmp_path: Path) -> None:
        """save_model_with_comments returns files dict on success."""
        model = _make_mock_model(metadata={})
        expected_files = {"simulation_main": tmp_path / "sim.in"}

        success_result = ModelWriteResult(files=expected_files)

        with patch(
            "pyiwfm.io.model_writer.write_model_with_comments",
            return_value=success_result,
        ):
            files = save_model_with_comments(model, tmp_path)

        assert files == expected_files


# ===========================================================================
# 5E. Additional Coverage Tests
# ===========================================================================


class TestWriteSimulationMain:
    """Tests for CompleteModelWriter._write_simulation_main() (lines 807-913)."""

    def test_simulation_main_with_metadata(self, tmp_path: Path) -> None:
        """_write_simulation_main populates sim config from model metadata."""
        model = _make_mock_model(
            metadata={
                "start_date": "1990-10-01T00:00:00",
                "end_date": "2000-09-30T00:00:00",
                "time_step_length": "1",
                "time_step_unit": "DAY",
                "matrix_solver": 2,
                "relaxation": 1.5,
                "max_iterations": 2000,
                "max_supply_iterations": 75,
                "convergence_tolerance": 0.0001,
                "convergence_volume": 0.01,
                "convergence_supply": 0.005,
                "supply_adjust_option": 11,
                "debug_flag": 1,
                "cache_size": 600000,
                "title_lines": ["Title A", "Title B", "Title C"],
            },
            supply_adjustment=None,
            source_files={},
        )
        config = _make_config(tmp_path)
        writer = CompleteModelWriter(model, config)

        mock_sim_writer = MagicMock()
        mock_sim_writer.write_main.return_value = tmp_path / "Simulation" / "sim.in"

        with (
            patch(
                "pyiwfm.io.simulation_writer.SimulationMainWriter",
                return_value=mock_sim_writer,
            ),
            patch(
                "pyiwfm.io.model_writer._iso_to_iwfm_date",
                side_effect=lambda x: f"converted_{x}",
            ),
        ):
            result = writer._write_simulation_main()

        # Verify the writer was called and produced a path
        mock_sim_writer.write_main.assert_called_once()
        assert result == tmp_path / "Simulation" / "sim.in"

    def test_simulation_main_with_lakes(self, tmp_path: Path) -> None:
        """_write_simulation_main sets lake_main path when model has lakes."""
        mock_lakes = MagicMock()
        mock_lakes.n_lakes = 3

        model = _make_mock_model(
            lakes=mock_lakes,
            metadata={},
            supply_adjustment=None,
            source_files={},
        )
        config = _make_config(tmp_path)
        writer = CompleteModelWriter(model, config)

        mock_sim_writer = MagicMock()
        mock_sim_writer.write_main.return_value = tmp_path / "sim.in"

        with patch(
            "pyiwfm.io.simulation_writer.SimulationMainWriter",
            return_value=mock_sim_writer,
        ) as MockSimWriter:
            writer._write_simulation_main()

        # The SimulationMainConfig should have been created with a non-empty lake_main
        call_args = MockSimWriter.call_args
        sim_config = call_args[0][1]
        assert sim_config.lake_main != ""

    def test_simulation_main_with_supply_adjustment(self, tmp_path: Path) -> None:
        """_write_simulation_main sets supply_adjust when model has supply adjustment."""
        model = _make_mock_model(
            metadata={},
            supply_adjustment=MagicMock(),
            source_files={},
        )
        config = _make_config(tmp_path)
        writer = CompleteModelWriter(model, config)

        mock_sim_writer = MagicMock()
        mock_sim_writer.write_main.return_value = tmp_path / "sim.in"

        with patch(
            "pyiwfm.io.simulation_writer.SimulationMainWriter",
            return_value=mock_sim_writer,
        ) as MockSimWriter:
            writer._write_simulation_main()

        sim_config = MockSimWriter.call_args[0][1]
        assert sim_config.supply_adjust != ""

    def test_simulation_main_error_stored_in_result(self, tmp_path: Path) -> None:
        """write_all stores simulation_main errors in result.errors (lines 414-416)."""
        model = _make_mock_model(metadata={}, source_files={})
        config = _make_config(tmp_path)
        writer = CompleteModelWriter(model, config)

        with (
            patch.object(writer, "write_preprocessor", return_value={}),
            patch.object(writer, "_write_groundwater"),
            patch.object(writer, "_write_streams"),
            patch.object(writer, "_write_lakes"),
            patch.object(writer, "_write_rootzone"),
            patch.object(writer, "_copy_passthrough_components"),
            patch.object(writer, "_write_supply_adjustment"),
            patch.object(
                writer,
                "_write_simulation_main",
                side_effect=RuntimeError("sim write boom"),
            ),
        ):
            result = writer.write_all()

        assert "simulation_main" in result.errors
        assert "sim write boom" in result.errors["simulation_main"]
        assert result.success is False


class TestWriteGroundwaterComponent:
    """Tests for CompleteModelWriter._write_groundwater() (lines 464-512)."""

    def test_write_groundwater_delegates_to_gw_writer(self, tmp_path: Path) -> None:
        """_write_groundwater creates a GWComponentWriter and calls write_all."""
        model = _make_mock_model(
            groundwater=MagicMock(),
            metadata={},
            source_files={},
        )
        config = _make_config(tmp_path)
        writer = CompleteModelWriter(model, config)
        result = ModelWriteResult()

        mock_gw_writer = MagicMock()
        mock_gw_writer.write_all.return_value = {
            "main": tmp_path / "GW" / "GW_MAIN.dat",
            "bc_main": tmp_path / "GW" / "BC_MAIN.dat",
        }

        with patch(
            "pyiwfm.io.gw_writer.GWComponentWriter",
            return_value=mock_gw_writer,
        ):
            writer._write_groundwater(result)

        assert "gw_main" in result.files
        assert "gw_bc_main" in result.files
        assert len(result.errors) == 0

    def test_write_groundwater_error_stored(self, tmp_path: Path) -> None:
        """_write_groundwater stores error in result on exception (line 510-512)."""
        model = _make_mock_model(
            groundwater=MagicMock(),
            metadata={},
            source_files={},
        )
        config = _make_config(tmp_path)
        writer = CompleteModelWriter(model, config)
        result = ModelWriteResult()

        with patch(
            "pyiwfm.io.gw_writer.GWComponentWriter",
            side_effect=RuntimeError("GW write failed"),
        ):
            writer._write_groundwater(result)

        assert "groundwater" in result.errors
        assert "GW write failed" in result.errors["groundwater"]


class TestWriteStreamsComponent:
    """Tests for CompleteModelWriter._write_streams() (lines 519-558)."""

    def test_write_streams_skips_when_none(self, tmp_path: Path) -> None:
        """_write_streams returns early when model.streams is None."""
        model = _make_mock_model(streams=None)
        config = _make_config(tmp_path)
        writer = CompleteModelWriter(model, config)
        result = ModelWriteResult()
        writer._write_streams(result)
        assert len(result.files) == 0

    def test_write_streams_delegates_to_stream_writer(self, tmp_path: Path) -> None:
        """_write_streams creates a StreamComponentWriter and calls write_all."""
        model = _make_mock_model(
            streams=MagicMock(),
            metadata={},
            source_files={},
        )
        config = _make_config(tmp_path)
        writer = CompleteModelWriter(model, config)
        result = ModelWriteResult()

        mock_strm_writer = MagicMock()
        mock_strm_writer.write_all.return_value = {
            "main": tmp_path / "Stream" / "Stream_MAIN.dat",
        }

        with patch(
            "pyiwfm.io.stream_writer.StreamComponentWriter",
            return_value=mock_strm_writer,
        ):
            writer._write_streams(result)

        assert "stream_main" in result.files
        assert len(result.errors) == 0

    def test_write_streams_error_stored(self, tmp_path: Path) -> None:
        """_write_streams stores error in result on exception (lines 556-558)."""
        model = _make_mock_model(
            streams=MagicMock(),
            metadata={},
            source_files={},
        )
        config = _make_config(tmp_path)
        writer = CompleteModelWriter(model, config)
        result = ModelWriteResult()

        with patch(
            "pyiwfm.io.stream_writer.StreamComponentWriter",
            side_effect=RuntimeError("stream writer failed"),
        ):
            writer._write_streams(result)

        assert "streams" in result.errors


class TestWriteRootZoneComponent:
    """Tests for CompleteModelWriter._write_rootzone() (lines 590-636)."""

    def test_write_rootzone_skips_when_none(self, tmp_path: Path) -> None:
        """_write_rootzone returns early when model.rootzone is None."""
        model = _make_mock_model(rootzone=None)
        config = _make_config(tmp_path)
        writer = CompleteModelWriter(model, config)
        result = ModelWriteResult()
        writer._write_rootzone(result)
        assert len(result.files) == 0

    def test_write_rootzone_delegates_to_rz_writer(self, tmp_path: Path) -> None:
        """_write_rootzone creates a RootZoneComponentWriter and calls write_all."""
        model = _make_mock_model(
            rootzone=MagicMock(),
            metadata={},
            source_files={},
        )
        config = _make_config(tmp_path)
        writer = CompleteModelWriter(model, config)
        result = ModelWriteResult()

        mock_rz_writer = MagicMock()
        mock_rz_writer.write_all.return_value = {
            "main": tmp_path / "RootZone" / "RZ_MAIN.dat",
        }

        with patch(
            "pyiwfm.io.rootzone_writer.RootZoneComponentWriter",
            return_value=mock_rz_writer,
        ):
            writer._write_rootzone(result)

        assert "rootzone_main" in result.files

    def test_write_rootzone_error_stored(self, tmp_path: Path) -> None:
        """_write_rootzone stores error in result on exception (lines 634-636)."""
        model = _make_mock_model(
            rootzone=MagicMock(),
            metadata={},
            source_files={},
        )
        config = _make_config(tmp_path)
        writer = CompleteModelWriter(model, config)
        result = ModelWriteResult()

        with patch(
            "pyiwfm.io.rootzone_writer.RootZoneComponentWriter",
            side_effect=RuntimeError("rz writer failed"),
        ):
            writer._write_rootzone(result)

        assert "rootzone" in result.errors


class TestWriteModelConvenienceVersions:
    """Tests for write_model() version propagation (lines 966-971)."""

    def test_write_model_propagates_version_from_metadata(self, tmp_path: Path) -> None:
        """write_model() propagates gw_version from model.metadata to config."""
        model = _make_mock_model(
            metadata={
                "gw_version": "4.2",
                "stream_version": "5.0",
            }
        )
        mock_result = ModelWriteResult(files={"sim": tmp_path / "sim.in"})

        with patch("pyiwfm.io.model_writer.CompleteModelWriter") as MockWriter:
            MockWriter.return_value.write_all.return_value = mock_result
            write_model(model, tmp_path)

        # Verify ModelWriteConfig received the version defaults
        config_arg = MockWriter.call_args[0][1]
        assert config_arg.gw_version == "4.2"
        assert config_arg.stream_version == "5.0"

    def test_write_model_kwargs_override_metadata_version(self, tmp_path: Path) -> None:
        """write_model() explicit kwargs override metadata versions."""
        model = _make_mock_model(metadata={"gw_version": "4.2"})
        mock_result = ModelWriteResult(files={"sim": tmp_path / "sim.in"})

        with patch("pyiwfm.io.model_writer.CompleteModelWriter") as MockWriter:
            MockWriter.return_value.write_all.return_value = mock_result
            write_model(model, tmp_path, gw_version="3.0")

        config_arg = MockWriter.call_args[0][1]
        assert config_arg.gw_version == "3.0"


class TestWriteModelWithCommentsFunction:
    """Tests for write_model_with_comments() (lines 989-1063)."""

    def test_write_model_with_comments_passes_metadata(self, tmp_path: Path) -> None:
        """write_model_with_comments passes comment_metadata to CompleteModelWriter."""
        model = _make_mock_model(metadata={})
        mock_comments = {"gw_main": MagicMock()}
        mock_result = ModelWriteResult(files={"sim": tmp_path / "sim.in"})

        with patch("pyiwfm.io.model_writer.CompleteModelWriter") as MockWriter:
            MockWriter.return_value.write_all.return_value = mock_result
            with patch("pyiwfm.io.model_writer._save_comment_sidecars") as mock_save:
                write_model_with_comments(model, tmp_path, comment_metadata=mock_comments)

        # Verify CompleteModelWriter received comment metadata
        assert MockWriter.call_args[1]["comment_metadata"] is mock_comments
        assert MockWriter.call_args[1]["preserve_comments"] is True
        # Verify sidecars were saved
        mock_save.assert_called_once_with(mock_result.files, mock_comments)

    def test_write_model_with_comments_no_sidecars(self, tmp_path: Path) -> None:
        """write_model_with_comments with save_sidecars=False skips sidecar saving."""
        model = _make_mock_model(metadata={})
        mock_result = ModelWriteResult(files={"sim": tmp_path / "sim.in"})

        with patch("pyiwfm.io.model_writer.CompleteModelWriter") as MockWriter:
            MockWriter.return_value.write_all.return_value = mock_result
            with patch("pyiwfm.io.model_writer._save_comment_sidecars") as mock_save:
                write_model_with_comments(
                    model,
                    tmp_path,
                    comment_metadata={"gw_main": MagicMock()},
                    save_sidecars=False,
                )

        mock_save.assert_not_called()


class TestSaveCommentSidecars:
    """Tests for _save_comment_sidecars() (lines 1066-1100)."""

    def test_saves_sidecar_for_matching_file(self, tmp_path: Path) -> None:
        """_save_comment_sidecars calls save_for_file when output path exists."""
        from pyiwfm.io.model_writer import _save_comment_sidecars

        # Create a real file at the output path
        out_file = tmp_path / "gw_main.dat"
        out_file.write_text("content")

        mock_metadata = MagicMock()
        written_files = {"gw_main": out_file}
        comment_metadata = {"gw_main": mock_metadata}

        _save_comment_sidecars(written_files, comment_metadata)

        mock_metadata.save_for_file.assert_called_once_with(out_file)

    def test_saves_sidecar_via_common_mapping(self, tmp_path: Path) -> None:
        """_save_comment_sidecars tries common mappings when direct key misses."""
        from pyiwfm.io.model_writer import _save_comment_sidecars

        # The file type "simulation_main" is in the common mappings
        out_file = tmp_path / "sim.in"
        out_file.write_text("content")

        mock_metadata = MagicMock()
        written_files = {"simulation_main": out_file}
        comment_metadata = {"simulation_main": mock_metadata}

        _save_comment_sidecars(written_files, comment_metadata)

        mock_metadata.save_for_file.assert_called_once_with(out_file)

    def test_skips_sidecar_when_no_matching_path(self, tmp_path: Path) -> None:
        """_save_comment_sidecars skips when no output path matches."""
        from pyiwfm.io.model_writer import _save_comment_sidecars

        mock_metadata = MagicMock()
        written_files = {}  # No matching files
        comment_metadata = {"some_unknown_key": mock_metadata}

        _save_comment_sidecars(written_files, comment_metadata)

        mock_metadata.save_for_file.assert_not_called()

    def test_handles_save_for_file_exception(self, tmp_path: Path) -> None:
        """_save_comment_sidecars logs warning when save_for_file raises."""
        from pyiwfm.io.model_writer import _save_comment_sidecars

        out_file = tmp_path / "gw_main.dat"
        out_file.write_text("content")

        mock_metadata = MagicMock()
        mock_metadata.save_for_file.side_effect = OSError("disk error")

        written_files = {"gw_main": out_file}
        comment_metadata = {"gw_main": mock_metadata}

        # Should not raise, just log a warning
        _save_comment_sidecars(written_files, comment_metadata)


class TestIsoToIwfmDateAdditional:
    """Additional tests for _iso_to_iwfm_date() edge cases."""

    def test_handles_date_only_iso_string(self) -> None:
        """_iso_to_iwfm_date handles ISO date-only string (no time part)."""
        with patch(
            "pyiwfm.io.timeseries_ascii.format_iwfm_timestamp",
            return_value="01/15/2020_24:00",
        ):
            result = _iso_to_iwfm_date("2020-01-15")
        assert result == "01/15/2020_24:00"

    def test_handles_datetime_with_time(self) -> None:
        """_iso_to_iwfm_date handles ISO datetime with nonzero time."""
        with patch(
            "pyiwfm.io.timeseries_ascii.format_iwfm_timestamp",
            return_value="03/15/2020_12:30",
        ):
            result = _iso_to_iwfm_date("2020-03-15T12:30:00")
        assert result == "03/15/2020_12:30"

    def test_returns_original_on_type_error(self) -> None:
        """_iso_to_iwfm_date returns original value when TypeError occurs."""
        # None would cause TypeError in fromisoformat
        result = _iso_to_iwfm_date(None)  # type: ignore[arg-type]
        assert result is None


class TestUnsaturatedZoneFallbacks:
    """Tests for unsaturated zone fallback paths (lines 703-757)."""

    def test_unsatzone_writer_failure_falls_back_to_copy(self, tmp_path: Path) -> None:
        """_write_unsaturated_zone falls back to copy when writer fails."""
        src_file = tmp_path / "uz_source.dat"
        src_file.write_text("uz data")

        model = _make_mock_model(
            unsaturated_zone=MagicMock(),
            metadata={},
            source_files={"unsatzone_main": src_file},
        )
        config = _make_config(tmp_path)
        writer = CompleteModelWriter(model, config)
        result = ModelWriteResult()

        with patch(
            "pyiwfm.io.unsaturated_zone_writer.UnsatZoneComponentWriter",
            side_effect=RuntimeError("uz writer broken"),
        ):
            writer._write_unsaturated_zone(result)

        assert any("falling back to copy" in w for w in result.warnings)
        assert "unsatzone_main" in result.files

    def test_unsatzone_no_data_no_source(self, tmp_path: Path) -> None:
        """_write_unsaturated_zone does nothing when no data and no source."""
        model = _make_mock_model(
            unsaturated_zone=None,
            source_files={},
        )
        config = _make_config(tmp_path)
        writer = CompleteModelWriter(model, config)
        result = ModelWriteResult()

        writer._write_unsaturated_zone(result)

        assert "unsatzone_main" not in result.files

    def test_unsatzone_source_not_found(self, tmp_path: Path) -> None:
        """_write_unsaturated_zone warns when source file does not exist."""
        model = _make_mock_model(
            unsaturated_zone=None,
            source_files={"unsatzone_main": tmp_path / "nonexistent.dat"},
        )
        config = _make_config(tmp_path)
        writer = CompleteModelWriter(model, config)
        result = ModelWriteResult()

        writer._write_unsaturated_zone(result)

        assert any("Passthrough source not found" in w for w in result.warnings)

    def test_unsatzone_copy_failure(self, tmp_path: Path) -> None:
        """_write_unsaturated_zone warns on copy failure (lines 756-759)."""
        src_file = tmp_path / "uz_source.dat"
        src_file.write_text("uz data")

        model = _make_mock_model(
            unsaturated_zone=None,
            source_files={"unsatzone_main": src_file},
        )
        config = _make_config(tmp_path)
        writer = CompleteModelWriter(model, config)
        result = ModelWriteResult()

        with patch(
            "pyiwfm.io.model_writer.shutil.copy2",
            side_effect=OSError("copy fail"),
        ):
            writer._write_unsaturated_zone(result)

        assert any("Failed to copy passthrough unsatzone_main" in w for w in result.warnings)


class TestSupplyAdjustmentAdditional:
    """Additional supply adjustment coverage (lines 802-803)."""

    def test_supply_adjust_copy_failure(self, tmp_path: Path) -> None:
        """_write_supply_adjustment warns when source copy fails."""
        src_file = tmp_path / "supply_src.dat"
        src_file.write_text("supply data")

        model = _make_mock_model(
            supply_adjustment=None,
            source_files={"supply_adjust": src_file},
        )
        config = _make_config(tmp_path)
        writer = CompleteModelWriter(model, config)
        result = ModelWriteResult()

        with patch(
            "pyiwfm.io.model_writer.shutil.copy2",
            side_effect=OSError("copy fail"),
        ):
            writer._write_supply_adjustment(result)

        assert any("Failed to copy supply adjustment" in w for w in result.warnings)

    def test_supply_adjust_source_not_found(self, tmp_path: Path) -> None:
        """_write_supply_adjustment warns when source does not exist."""
        model = _make_mock_model(
            supply_adjustment=None,
            source_files={"supply_adjust": tmp_path / "missing.dat"},
        )
        config = _make_config(tmp_path)
        writer = CompleteModelWriter(model, config)
        result = ModelWriteResult()

        writer._write_supply_adjustment(result)

        assert any("source not found" in w for w in result.warnings)

    def test_supply_adjust_write_from_data_failure_falls_through(self, tmp_path: Path) -> None:
        """_write_supply_adjustment falls to copy when write_supply_adjustment fails."""
        src_file = tmp_path / "sa_source.dat"
        src_file.write_text("sa data")

        model = _make_mock_model(
            supply_adjustment=MagicMock(),
            source_files={"supply_adjust": src_file},
        )
        config = _make_config(tmp_path)
        writer = CompleteModelWriter(model, config)
        result = ModelWriteResult()

        with patch(
            "pyiwfm.io.supply_adjust.write_supply_adjustment",
            side_effect=RuntimeError("write failed"),
        ):
            writer._write_supply_adjustment(result)

        # Should have a warning about failing to write from data
        assert any("Failed to write supply adjustment from data" in w for w in result.warnings)
        # Should have fallen through to copy
        assert "supply_adjust" in result.files
