"""Tests for pyiwfm.visualization.webapi.services.observation_loader."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from pyiwfm.visualization.webapi.services.observation_loader import (
    _load_single_file,
    guess_obs_type,
    load_csv_file,
    load_smp_file,
    scan_directory,
)


class TestGuessObsType:
    """Test guess_obs_type."""

    def test_groundwater(self) -> None:
        assert guess_obs_type("obs_gw.smp") == "gw"
        assert guess_obs_type("head_data.csv") == "gw"
        assert guess_obs_type("gw_levels.smp") == "gw"

    def test_stream(self) -> None:
        assert guess_obs_type("stream_flow.smp") == "stream"
        assert guess_obs_type("str_obs.csv") == "stream"
        assert guess_obs_type("FLOW_data.smp") == "stream"

    def test_subsidence(self) -> None:
        assert guess_obs_type("subsidence.smp") == "subsidence"
        assert guess_obs_type("insar_data.csv") == "subsidence"

    def test_hdiff(self) -> None:
        assert guess_obs_type("hdiff_pairs.smp") == "hdiff"
        assert guess_obs_type("head_diff.csv") == "hdiff"

    def test_default(self) -> None:
        """Unknown patterns default to gw."""
        assert guess_obs_type("random_data.smp") == "gw"
        assert guess_obs_type("xyz.csv") == "gw"


class TestScanDirectory:
    """Test scan_directory."""

    def test_finds_smp_and_csv(self, tmp_path: Path) -> None:
        (tmp_path / "obs.smp").write_text("data")
        (tmp_path / "data.csv").write_text("data")
        (tmp_path / "readme.txt").write_text("ignore")

        results = scan_directory(tmp_path)
        assert len(results) == 2
        formats = {r["format"] for r in results}
        assert formats == {"smp", "csv"}

    def test_empty_directory(self, tmp_path: Path) -> None:
        results = scan_directory(tmp_path)
        assert len(results) == 0

    def test_recursive(self, tmp_path: Path) -> None:
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "deep.smp").write_text("data")
        (tmp_path / "top.csv").write_text("data")

        results = scan_directory(tmp_path, recursive=True)
        assert len(results) == 2

    def test_non_recursive(self, tmp_path: Path) -> None:
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "deep.smp").write_text("data")
        (tmp_path / "top.csv").write_text("data")

        results = scan_directory(tmp_path, recursive=False)
        assert len(results) == 1

    def test_type_guess_included(self, tmp_path: Path) -> None:
        (tmp_path / "stream_obs.smp").write_text("data")
        results = scan_directory(tmp_path)
        assert results[0]["type_guess"] == "stream"


class TestLoadSmpFile:
    """Test load_smp_file."""

    def test_basic_load(self, tmp_path: Path) -> None:
        smp = tmp_path / "obs.smp"
        smp.write_text(
            "WELL_01                  01/31/2020  12:00:00     105.50\n"
            "WELL_01                  02/29/2020  12:00:00     106.00\n"
        )
        state = MagicMock()
        ids = load_smp_file(smp, "gw", state)
        assert len(ids) == 1
        state.add_observation.assert_called_once()
        call_args = state.add_observation.call_args
        obs_data = call_args[0][1]
        assert obs_data["type"] == "gw"
        assert obs_data["n_records"] == 2


class TestLoadCsvFile:
    """Test load_csv_file."""

    def test_basic_load(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "obs.csv"
        csv_file.write_text("date,value\n2020-01-31,105.5\n2020-02-29,106.0\n")
        state = MagicMock()
        ids = load_csv_file(csv_file, "gw", state)
        assert len(ids) == 1
        state.add_observation.assert_called_once()

    def test_with_location_col(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "obs.csv"
        csv_file.write_text("date,value,location\n2020-01-31,100.0,W1\n2020-01-31,90.0,W2\n")
        state = MagicMock()
        ids = load_csv_file(csv_file, "gw", state, location_col=2)
        assert len(ids) == 2

    def test_empty_csv(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("")
        state = MagicMock()
        ids = load_csv_file(csv_file, "gw", state)
        assert len(ids) == 0


class TestLoadSingleFile:
    """Test _load_single_file."""

    def test_dispatches_smp(self, tmp_path: Path) -> None:
        smp = tmp_path / "obs.smp"
        smp.write_text("WELL_01                  01/31/2020  12:00:00     100.00\n")
        state = MagicMock()
        ids = _load_single_file(smp, "gw", state)
        assert len(ids) == 1

    def test_unsupported_format(self, tmp_path: Path) -> None:
        txt = tmp_path / "data.txt"
        txt.write_text("data")
        state = MagicMock()
        ids = _load_single_file(txt, "gw", state)
        assert len(ids) == 0
