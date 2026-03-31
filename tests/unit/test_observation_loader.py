"""Tests for pyiwfm.visualization.webapi.services.observation_loader."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from pyiwfm.visualization.webapi.services.observation_loader import (
    _load_single_file,
    detect_delimiter,
    guess_obs_type,
    load_delimited_file,
    load_smp_file,
    looks_like_smp,
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

    def test_finds_loadable_files(self, tmp_path: Path) -> None:
        (tmp_path / "obs.smp").write_text("data")
        (tmp_path / "data.csv").write_text("data")
        (tmp_path / "readme.md").write_text("ignore")
        (tmp_path / "extra.tsv").write_text("data")

        results = scan_directory(tmp_path)
        assert len(results) == 3
        formats = {r["format"] for r in results}
        assert formats == {"smp", "csv", "tsv"}

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


class TestLoadDelimitedFile:
    """Test load_delimited_file."""

    def test_basic_load(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "obs.csv"
        csv_file.write_text("date,value\n2020-01-31,105.5\n2020-02-29,106.0\n")
        state = MagicMock()
        ids = load_delimited_file(csv_file, "gw", state)
        assert len(ids) == 1
        state.add_observation.assert_called_once()

    def test_with_location_col(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "obs.csv"
        csv_file.write_text("date,value,location\n2020-01-31,100.0,W1\n2020-01-31,90.0,W2\n")
        state = MagicMock()
        ids = load_delimited_file(csv_file, "gw", state, location_col=2)
        assert len(ids) == 2

    def test_empty_csv(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("")
        state = MagicMock()
        ids = load_delimited_file(csv_file, "gw", state)
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
        f = tmp_path / "data.xyz"
        f.write_text("data")
        state = MagicMock()
        ids = _load_single_file(f, "gw", state)
        assert len(ids) == 0

    def test_txt_with_smp_content_routes_to_smp(self, tmp_path: Path) -> None:
        """A .txt file with SMP-formatted content should be loaded via SMPReader."""
        txt = tmp_path / "obs.txt"
        txt.write_text(
            "WELL_01                  01/31/2020  12:00:00     100.00\n"
            "WELL_01                  02/29/2020  12:00:00     101.00\n"
        )
        state = MagicMock()
        ids = _load_single_file(txt, "gw", state)
        assert len(ids) == 1
        obs_data = state.add_observation.call_args[0][1]
        assert obs_data["n_records"] == 2

    def test_dat_with_csv_content(self, tmp_path: Path) -> None:
        """A .dat file with CSV content should be loaded via delimiter detection."""
        dat = tmp_path / "obs.dat"
        dat.write_text("date,value\n2020-01-31,105.5\n2020-02-29,106.0\n")
        state = MagicMock()
        ids = _load_single_file(dat, "gw", state)
        assert len(ids) == 1


class TestDetectDelimiter:
    """Test detect_delimiter."""

    def test_csv(self) -> None:
        text = "date,value,loc\n2020-01-31,100.0,W1\n2020-02-28,101.0,W1\n"
        assert detect_delimiter(text) == ","

    def test_tsv(self) -> None:
        text = "date\tvalue\tloc\n2020-01-31\t100.0\tW1\n2020-02-28\t101.0\tW1\n"
        assert detect_delimiter(text) == "\t"

    def test_whitespace(self) -> None:
        text = "WELL_01  01/31/2020  12:00:00  100.00\nWELL_01  02/29/2020  12:00:00  101.00\n"
        assert detect_delimiter(text) == "whitespace"

    def test_empty(self) -> None:
        assert detect_delimiter("") == ","


class TestLooksLikeSmp:
    """Test looks_like_smp."""

    def test_smp_content(self) -> None:
        text = (
            "WELL_01                  01/31/2020  12:00:00     100.00\n"
            "WELL_01                  02/29/2020  12:00:00     101.00\n"
        )
        assert looks_like_smp(text) is True

    def test_csv_content(self) -> None:
        text = "date,value\n2020-01-31,100.0\n2020-02-28,101.0\n"
        assert looks_like_smp(text) is False

    def test_tsv_with_headers(self) -> None:
        text = "date\tvalue\n2020-01-31\t100.0\n2020-02-28\t101.0\n"
        assert looks_like_smp(text) is False

    def test_empty(self) -> None:
        assert looks_like_smp("") is False

    def test_tab_delimited_smp(self) -> None:
        """Tab-delimited SMP files (like those from iwfm2obs) should be detected."""
        text = (
            "C_349836N1189228W001\t9/30/1974\t0:00:00\t238.53\n"
            "C_349836N1189228W001\t11/30/1981\t0:00:00\t454.20\n"
        )
        assert looks_like_smp(text) is True


class TestLoadDelimitedFileTsv:
    """Test load_delimited_file with TSV data."""

    def test_tsv_load(self, tmp_path: Path) -> None:
        tsv = tmp_path / "obs.tsv"
        tsv.write_text("date\tvalue\n2020-01-31\t105.5\n2020-02-29\t106.0\n")
        state = MagicMock()
        ids = load_delimited_file(tsv, "gw", state)
        assert len(ids) == 1
        state.add_observation.assert_called_once()

    def test_whitespace_delimited_load(self, tmp_path: Path) -> None:
        ws = tmp_path / "obs.dat"
        ws.write_text("date value\n2020-01-31 105.5\n2020-02-29 106.0\n")
        state = MagicMock()
        ids = load_delimited_file(ws, "gw", state)
        assert len(ids) == 1
