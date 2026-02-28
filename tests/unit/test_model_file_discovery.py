"""Tests for pyiwfm.calibration.model_file_discovery."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_sim_main(
    sim_dir: Path,
    gw_filename: str = "Groundwater.dat",
    stream_filename: str = "Stream.dat",
) -> Path:
    """Write a minimal simulation main file for testing."""
    sim_file = sim_dir / "C2VSimFG.in"
    # Lines 1-4: skip lines, Line 5: GW main, Line 6: Stream main
    # Lines 7-14: skip, Line 15: start date, Line 16: skip, Line 17: time unit
    # The IWFM sim main file format has 4 data lines before the GW main path,
    # then 8 lines between Stream main and start date, then 1 skip before time unit.
    # Total: 4 + 1(GW) + 1(Stream) + 8 + 1(start) + 1(skip) + 1(timeunit) = 17.
    content = textwrap.dedent(f"""\
        C  Header
        C  Comment
        1.0                                     / FACT_LT
        1.0                                     / FACT_AR
        1.0                                     / FACT_VL
        Preprocessor.bin                         / preprocessor
        {gw_filename}                            / GW main file
        {stream_filename}                        / Stream main file
        RZ.dat                                   / Root zone
        SW.dat                                   / Small watersheds
        UZ.dat                                   / Unsaturated zone
        LK.dat                                   / Lakes
        Supp.dat                                 / Supply adjustment
        SurfWater.dat                            / Surface water
        0                                        / Flag
        1.0                                      / Factor
        10/31/1973_24:00                         / Start date
        12/31/2015_24:00                         / End date
        1MON                                     / Time unit
    """)
    sim_file.write_text(content)
    return sim_file


def _write_gw_main(
    gw_dir: Path,
    n_hyd: int = 3,
    out_filename: str = "GW_Hydrographs.out",
) -> Path:
    """Write a minimal GW main file."""
    gw_file = gw_dir / "Groundwater.dat"

    # Build content: 1 skip, then tile drain, pumping, subsidence paths
    # Then 17 skip lines to NOUTH, FACTXY, hydrograph output path, header, entries
    lines: list[str] = []
    lines.append("4.0                                     / Version")
    lines.append("                                        / Tile drain file")
    lines.append("Pumping.dat                             / Pumping file")
    lines.append("                                        / Subsidence file")
    # 17 more non-comment lines to reach NOUTH
    for i in range(16):
        lines.append(f"0.0                                     / Param line {i + 1}")
    lines.append(f"{n_hyd}                                  / NOUTH")
    lines.append("1.0                                     / FACTXY")
    lines.append(f"{out_filename}                           / GWHYDOUTFL")
    # Header line for hydrograph entries
    lines.append("C  ID HYDTYP IOUTHL X Y NAME")
    # Hydrograph entries (HYDTYP=0 means X-Y coords provided)
    for i in range(1, n_hyd + 1):
        lines.append(
            f"{i}  0  {i}  {6300000.0 + i * 100:.1f}  {2160000.0 + i * 100:.1f}  WELL_{i:03d}"
        )

    gw_file.write_text("\n".join(lines))
    return gw_file


def _write_stream_main(
    str_dir: Path,
    n_hyd: int = 2,
    out_filename: str = "Stream_Hydrographs.out",
) -> Path:
    """Write a minimal stream main file."""
    str_file = str_dir / "Stream.dat"

    lines: list[str] = []
    # 7 non-comment lines, the 7th being NOUTR
    for i in range(6):
        lines.append(f"0.0                                     / Stream param {i + 1}")
    lines.append(f"{n_hyd}                                  / NOUTR")
    # 6 lines, the 6th being the output file path
    for i in range(5):
        lines.append(f"0.0                                     / Stream skip {i + 1}")
    lines.append(f"{out_filename}                           / Stream output file")
    # Header + entries
    lines.append("C  ID NAME NODE")
    for i in range(1, n_hyd + 1):
        lines.append(f"{i}  GAGE_{i:03d}  {i * 10}")

    str_file.write_text("\n".join(lines))
    return str_file


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDiscoverHydrographFiles:
    """Test discover_hydrograph_files()."""

    def test_basic_discovery(self, tmp_path: Path) -> None:
        """Test that we can discover GW and stream .out paths."""
        from pyiwfm.calibration.model_file_discovery import discover_hydrograph_files

        sim_file = _write_sim_main(tmp_path)
        _write_gw_main(tmp_path, n_hyd=3, out_filename="GW.out")
        _write_stream_main(tmp_path, n_hyd=2, out_filename="STR.out")

        info = discover_hydrograph_files(sim_file)

        assert info.gw_main_path is not None
        assert info.stream_main_path is not None
        assert info.start_date_str == "10/31/1973"
        assert info.time_unit == "1MON"

    def test_gw_locations_parsed(self, tmp_path: Path) -> None:
        """Test that GW hydrograph locations are parsed correctly."""
        from pyiwfm.calibration.model_file_discovery import discover_hydrograph_files

        _write_sim_main(tmp_path)
        _write_gw_main(tmp_path, n_hyd=3)
        _write_stream_main(tmp_path, n_hyd=2)

        info = discover_hydrograph_files(tmp_path / "C2VSimFG.in")

        assert len(info.gw_locations) == 3
        assert info.gw_locations[0].name == "WELL_001"
        assert info.gw_locations[0].layer == 1
        assert info.gw_locations[1].name == "WELL_002"
        assert info.gw_locations[1].layer == 2

    def test_stream_locations_parsed(self, tmp_path: Path) -> None:
        """Test that stream hydrograph locations are parsed correctly."""
        from pyiwfm.calibration.model_file_discovery import discover_hydrograph_files

        _write_sim_main(tmp_path)
        _write_gw_main(tmp_path, n_hyd=1)
        _write_stream_main(tmp_path, n_hyd=2)

        info = discover_hydrograph_files(tmp_path / "C2VSimFG.in")

        assert len(info.stream_locations) == 2
        assert info.stream_locations[0].name == "GAGE_001"

    def test_missing_sim_file_raises(self, tmp_path: Path) -> None:
        """Test FileNotFoundError for missing sim file."""
        from pyiwfm.calibration.model_file_discovery import discover_hydrograph_files

        with pytest.raises(FileNotFoundError):
            discover_hydrograph_files(tmp_path / "nonexistent.in")

    def test_gw_hydrograph_path_resolved(self, tmp_path: Path) -> None:
        """Test that the .out path is resolved correctly."""
        from pyiwfm.calibration.model_file_discovery import discover_hydrograph_files

        _write_sim_main(tmp_path)
        gw_out = tmp_path / "GW_Hydrographs.out"
        gw_out.touch()
        _write_gw_main(tmp_path, n_hyd=1, out_filename="GW_Hydrographs.out")
        _write_stream_main(tmp_path, n_hyd=0)

        info = discover_hydrograph_files(tmp_path / "C2VSimFG.in")

        assert info.gw_hydrograph_path is not None

    def test_no_stream_file(self, tmp_path: Path) -> None:
        """Test graceful handling when stream main file is missing."""
        from pyiwfm.calibration.model_file_discovery import discover_hydrograph_files

        _write_sim_main(tmp_path, stream_filename="")
        _write_gw_main(tmp_path, n_hyd=1)

        info = discover_hydrograph_files(tmp_path / "C2VSimFG.in")

        assert len(info.stream_locations) == 0

    def test_zero_hydrographs(self, tmp_path: Path) -> None:
        """Test discovery with zero hydrographs."""
        from pyiwfm.calibration.model_file_discovery import discover_hydrograph_files

        _write_sim_main(tmp_path)
        _write_gw_main(tmp_path, n_hyd=0)
        _write_stream_main(tmp_path, n_hyd=0)

        info = discover_hydrograph_files(tmp_path / "C2VSimFG.in")

        assert len(info.gw_locations) == 0
        assert len(info.stream_locations) == 0

    def test_hydtype_1_node_format(self, tmp_path: Path) -> None:
        """Test GW hydrograph entries with HYDTYP=1 (node number format)."""
        from pyiwfm.calibration.model_file_discovery import discover_hydrograph_files

        _write_sim_main(tmp_path)
        _write_stream_main(tmp_path, n_hyd=0)

        # Write GW main with HYDTYP=1 entries
        gw_file = tmp_path / "Groundwater.dat"
        lines: list[str] = []
        lines.append("4.0                                     / Version")
        lines.append("                                        / Tile drain")
        lines.append("Pumping.dat                             / Pumping")
        lines.append("                                        / Subsidence")
        for _ in range(16):
            lines.append("0.0                                     / Param")
        lines.append("2                                       / NOUTH")
        lines.append("1.0                                     / FACTXY")
        lines.append("GW.out                                  / GWHYDOUTFL")
        lines.append("C  Header")
        # HYDTYP=1: ID HYDTYP IOUTHL IOUTH NAME
        lines.append("1  1  1  100  WELL_A")
        lines.append("2  1  2  200  WELL_B")
        gw_file.write_text("\n".join(lines))

        info = discover_hydrograph_files(tmp_path / "C2VSimFG.in")

        assert len(info.gw_locations) == 2
        assert info.gw_locations[0].name == "WELL_A"
        assert info.gw_locations[0].x == 0.0  # HYDTYP=1 → no X/Y
        assert info.gw_locations[1].layer == 2
