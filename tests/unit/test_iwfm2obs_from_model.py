"""Tests for iwfm2obs_from_model and related new functions."""

from __future__ import annotations

import textwrap
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from pyiwfm.calibration.iwfm2obs import (
    InterpolationConfig,
    IWFM2OBSConfig,
    write_multilayer_output,
    write_multilayer_pest_ins,
)
from pyiwfm.calibration.obs_well_spec import ObsWellSpec

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def fixtures_dir() -> Path:
    """Return path to calibration fixtures."""
    return Path(__file__).parent.parent / "fixtures" / "calibration"


# ---------------------------------------------------------------------------
# Tests: write_multilayer_output
# ---------------------------------------------------------------------------


class TestWriteMultilayerOutput:
    """Test write_multilayer_output()."""

    def test_basic_output(self, tmp_path: Path) -> None:
        """Test writing a basic GW_MultiLayer.out file."""
        wells = [
            ObsWellSpec(
                name="WELL_001",
                x=100.0,
                y=200.0,
                element_id=1,
                bottom_of_screen=-50.0,
                top_of_screen=-10.0,
            )
        ]
        weights = [np.array([0.6, 0.4])]
        results: dict[str, list[tuple[datetime, float]]] = {
            "WELL_001": [
                (datetime(2020, 1, 31), 105.5),
                (datetime(2020, 2, 29), 106.0),
            ]
        }
        out_path = tmp_path / "GW_MultiLayer.out"

        write_multilayer_output(results, wells, weights, out_path, n_layers=2)

        assert out_path.exists()
        lines = out_path.read_text().strip().split("\n")
        assert len(lines) == 3  # 1 header + 2 data
        assert "WELL_001" in lines[1]
        assert "105.50" in lines[1]

    def test_output_columns(self, tmp_path: Path) -> None:
        """Test that output has correct number of columns."""
        wells = [
            ObsWellSpec("W1", 0, 0, 1, -100, -50),
        ]
        weights = [np.array([0.3, 0.4, 0.2, 0.1])]
        results = {
            "W1": [(datetime(2020, 6, 15), 50.25)],
        }
        out_path = tmp_path / "ml.out"

        write_multilayer_output(results, wells, weights, out_path, n_layers=4)

        lines = out_path.read_text().strip().split("\n")
        # Data line should have: Name, Date, Time, Simulated, T1-T4, NewTOS, NewBOS
        data_parts = lines[1].split()
        assert len(data_parts) >= 9  # At least Name Date Time Sim T1..T4 TOS BOS

    def test_empty_results(self, tmp_path: Path) -> None:
        """Test writing with empty results."""
        wells = [ObsWellSpec("W1", 0, 0, 1, -100, -50)]
        weights = [np.array([1.0])]
        results: dict[str, list[tuple[datetime, float]]] = {}
        out_path = tmp_path / "ml.out"

        write_multilayer_output(results, wells, weights, out_path, n_layers=1)

        assert out_path.exists()
        lines = out_path.read_text().strip().split("\n")
        assert len(lines) == 1  # header only


# ---------------------------------------------------------------------------
# Tests: write_multilayer_pest_ins
# ---------------------------------------------------------------------------


class TestWriteMultilayerPestIns:
    """Test write_multilayer_pest_ins()."""

    def test_basic_ins(self, tmp_path: Path) -> None:
        """Test writing a basic PEST .ins file."""
        wells = [ObsWellSpec("W1", 0, 0, 1, -100, -50)]
        results = {
            "W1": [
                (datetime(2020, 1, 31), 100.0),
                (datetime(2020, 2, 29), 101.0),
            ],
        }
        ins_path = tmp_path / "ml.ins"

        write_multilayer_pest_ins(results, wells, ins_path)

        assert ins_path.exists()
        lines = ins_path.read_text().strip().split("\n")
        assert lines[0] == "pif #"
        assert lines[1] == "l1"  # skip header
        assert lines[2] == "l1 [WLT00001_00001]50:60"
        assert lines[3] == "l1 [WLT00001_00002]50:60"

    def test_multi_well_ins(self, tmp_path: Path) -> None:
        """Test .ins file with multiple wells."""
        wells = [
            ObsWellSpec("W1", 0, 0, 1, -100, -50),
            ObsWellSpec("W2", 0, 0, 2, -80, -20),
        ]
        results = {
            "W1": [(datetime(2020, 1, 31), 100.0)],
            "W2": [(datetime(2020, 1, 31), 90.0), (datetime(2020, 2, 29), 91.0)],
        }
        ins_path = tmp_path / "ml.ins"

        write_multilayer_pest_ins(results, wells, ins_path)

        lines = ins_path.read_text().strip().split("\n")
        assert lines[2] == "l1 [WLT00001_00001]50:60"
        assert lines[3] == "l1 [WLT00002_00001]50:60"
        assert lines[4] == "l1 [WLT00002_00002]50:60"

    def test_wlt_naming_format(self, tmp_path: Path) -> None:
        """Test WLT naming: 5-digit well seq and 5-digit timestep seq."""
        wells = [ObsWellSpec("W1", 0, 0, 1, -100, -50)]
        results = {"W1": [(datetime(2020, 1, 1), 1.0)]}
        ins_path = tmp_path / "ml.ins"

        write_multilayer_pest_ins(results, wells, ins_path)

        lines = ins_path.read_text().strip().split("\n")
        # Check 5-digit zero-padded format
        assert "WLT00001_00001" in lines[2]


# ---------------------------------------------------------------------------
# Tests: get_columns_as_smp_dict
# ---------------------------------------------------------------------------


class TestGetColumnsAsSmpDict:
    """Test IWFMHydrographReader.get_columns_as_smp_dict()."""

    def test_basic_conversion(self, tmp_path: Path) -> None:
        """Test converting .out columns to SMPTimeSeries dict."""
        from pyiwfm.io.hydrograph_reader import IWFMHydrographReader

        # Write a minimal .out file
        out_file = tmp_path / "test.out"
        lines = [
            "*  HYDROGRAPH ID     1     2",
            "*  LAYER             1     2",
            "*  NODE             10    20",
            "01/31/2020_24:00   100.5  200.3",
            "02/29/2020_24:00   101.0  201.5",
            "03/31/2020_24:00   102.0  202.0",
        ]
        out_file.write_text("\n".join(lines))

        reader = IWFMHydrographReader(out_file)
        bore_ids = {0: "WELL_A", 1: "WELL_B"}

        result = reader.get_columns_as_smp_dict(bore_ids)

        assert "WELL_A" in result
        assert "WELL_B" in result
        assert len(result["WELL_A"].values) == 3
        assert result["WELL_A"].values[0] == pytest.approx(100.5)
        assert result["WELL_B"].values[1] == pytest.approx(201.5)

    def test_empty_reader(self, tmp_path: Path) -> None:
        """Test conversion with no data."""
        from pyiwfm.io.hydrograph_reader import IWFMHydrographReader

        out_file = tmp_path / "empty.out"
        out_file.write_text("")

        reader = IWFMHydrographReader(out_file)
        result = reader.get_columns_as_smp_dict({0: "A"})

        assert len(result) == 0

    def test_invalid_column_index(self, tmp_path: Path) -> None:
        """Test that invalid column indices are skipped."""
        from pyiwfm.io.hydrograph_reader import IWFMHydrographReader

        out_file = tmp_path / "test.out"
        out_file.write_text("*  HYDROGRAPH ID     1\n01/31/2020_24:00   100.0\n")

        reader = IWFMHydrographReader(out_file)
        result = reader.get_columns_as_smp_dict({0: "OK", 5: "BAD"})

        assert "OK" in result
        assert "BAD" not in result

    def test_bore_id_preserved(self, tmp_path: Path) -> None:
        """Test that bore_id is set correctly on SMPTimeSeries."""
        from pyiwfm.io.hydrograph_reader import IWFMHydrographReader

        out_file = tmp_path / "test.out"
        out_file.write_text("*  HYDROGRAPH ID     1\n01/31/2020_24:00   50.0\n")

        reader = IWFMHydrographReader(out_file)
        result = reader.get_columns_as_smp_dict({0: "MY_BORE%1"})

        assert result["MY_BORE%1"].bore_id == "MY_BORE%1"


# ---------------------------------------------------------------------------
# Tests: IWFM2OBSConfig
# ---------------------------------------------------------------------------


class TestIWFM2OBSConfig:
    """Test IWFM2OBSConfig dataclass."""

    def test_defaults(self) -> None:
        """Test default configuration values."""
        cfg = IWFM2OBSConfig()

        assert cfg.date_format == 2
        assert cfg.interpolation.sentinel_value == -999.0

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        from datetime import timedelta

        cfg = IWFM2OBSConfig(
            interpolation=InterpolationConfig(
                max_extrapolation_time=timedelta(days=7),
                sentinel_value=-1.0e38,
            ),
            date_format=1,
        )

        assert cfg.date_format == 1
        assert cfg.interpolation.max_extrapolation_time.days == 7


# ---------------------------------------------------------------------------
# Tests: iwfm2obs_from_model (integration-style with mock data)
# ---------------------------------------------------------------------------


class TestIWFM2OBSFromModel:
    """Test iwfm2obs_from_model() with synthetic model files."""

    def test_gw_interpolation(self, tmp_path: Path) -> None:
        """Test GW interpolation from simulated .out file data."""
        from pyiwfm.calibration.iwfm2obs import iwfm2obs_from_model

        # Create a minimal sim file structure
        sim_dir = tmp_path / "model"
        sim_dir.mkdir()

        # Write sim main
        sim_file = sim_dir / "C2VSimFG.in"
        sim_content = textwrap.dedent("""\
            C  Header
            C  Comment
            1.0                                     / FACT_LT
            1.0                                     / FACT_AR
            1.0                                     / FACT_VL
            Preprocessor.bin                         / preprocessor
            Groundwater.dat                          / GW main file
                                                    / Stream main file
            RZ.dat                                   / Root zone
            SW.dat                                   / Small watersheds
            UZ.dat                                   / Unsaturated zone
            LK.dat                                   / Lakes
            Supp.dat                                 / Supply adjustment
            SurfWater.dat                            / Surface water
            0                                        / Flag
            1.0                                      / Factor
            01/31/2020_24:00                         / Start date
            12/31/2021_24:00                         / End date
            1MON                                     / Time unit
        """)
        sim_file.write_text(sim_content)

        # Write GW main with 1 hydrograph
        gw_lines: list[str] = []
        gw_lines.append("4.0                                     / Version")
        gw_lines.append("                                        / Tile drain")
        gw_lines.append("Pumping.dat                             / Pumping")
        gw_lines.append("                                        / Subsidence")
        for _ in range(16):
            gw_lines.append("0.0                                     / Param")
        gw_lines.append("1                                       / NOUTH")
        gw_lines.append("1.0                                     / FACTXY")
        gw_lines.append("GW_Hydrographs.out                      / GWHYDOUTFL")
        gw_lines.append("C  Header")
        gw_lines.append("1  0  1  100.0  200.0  BORE_A")
        (sim_dir / "Groundwater.dat").write_text("\n".join(gw_lines))

        # Write .out file with test data
        out_lines = [
            "*  HYDROGRAPH ID     1",
            "*  LAYER             1",
            "*  NODE             10",
            "*  NAME            BORE_A",
            "*  ELEMENT          1",
        ]
        # Add monthly data
        dates = [
            "01/31/2020_24:00",
            "02/29/2020_24:00",
            "03/31/2020_24:00",
            "04/30/2020_24:00",
            "05/31/2020_24:00",
            "06/30/2020_24:00",
        ]
        for i, d in enumerate(dates):
            out_lines.append(f"{d}   {100.0 + i * 0.5:.4f}")
        (sim_dir / "GW_Hydrographs.out").write_text("\n".join(out_lines))

        # Write observation SMP with 3 obs times
        obs_lines = [
            " BORE_A               02/15/2020   12:00:00            0.0",
            " BORE_A               04/15/2020   12:00:00            0.0",
            " BORE_A               06/15/2020   12:00:00            0.0",
        ]
        obs_file = sim_dir / "obs_gw.smp"
        obs_file.write_text("\n".join(obs_lines))

        # Run
        output_file = sim_dir / "out_gw.smp"
        results = iwfm2obs_from_model(
            simulation_main_file=sim_file,
            obs_smp_paths={"gw": obs_file},
            output_paths={"gw": output_file},
        )

        assert "gw" in results
        assert output_file.exists()

    def test_no_obs_file_skipped(self, tmp_path: Path) -> None:
        """Test that types without obs files are skipped."""
        from pyiwfm.calibration.iwfm2obs import iwfm2obs_from_model

        sim_dir = tmp_path / "model"
        sim_dir.mkdir()

        # Create minimal sim file pointing to nonexistent files
        sim_file = sim_dir / "sim.in"
        sim_content = textwrap.dedent("""\
            C  Header
            C  Comment
            1.0
            1.0
            1.0
            Preprocessor.bin
                                                    / No GW
                                                    / No Stream
            RZ.dat
            SW.dat
            UZ.dat
            LK.dat
            Supp.dat
            SurfWater.dat
            0
            1.0
            01/31/2020_24:00
            12/31/2021_24:00
            1MON
        """)
        sim_file.write_text(sim_content)

        results = iwfm2obs_from_model(
            simulation_main_file=sim_file,
            obs_smp_paths={},
            output_paths={},
        )

        assert len(results) == 0
