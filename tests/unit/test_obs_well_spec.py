"""Tests for pyiwfm.calibration.obs_well_spec."""

from __future__ import annotations

from pathlib import Path

import pytest

from pyiwfm.calibration.obs_well_spec import read_obs_well_spec


class TestReadObsWellSpec:
    """Test read_obs_well_spec()."""

    def test_fixture_file(self, fixtures_path: Path) -> None:
        """Test reading the fixture obs well spec file."""
        spec_file = fixtures_path / "calibration" / "obs_well_spec.txt"
        if not spec_file.exists():
            pytest.skip("Fixture file not found")

        wells = read_obs_well_spec(spec_file)

        assert len(wells) == 3
        assert wells[0].name == "S_380313N1219426W001"
        assert wells[0].x == pytest.approx(6302184.5)
        assert wells[0].y == pytest.approx(2161430.2)
        assert wells[0].element_id == 1234
        assert wells[0].bottom_of_screen == pytest.approx(-175.44)
        assert wells[0].top_of_screen == pytest.approx(-105.44)
        assert wells[0].overwrite_layer == -1

    def test_overwrite_layer(self, fixtures_path: Path) -> None:
        """Test that overwrite_layer is parsed correctly."""
        spec_file = fixtures_path / "calibration" / "obs_well_spec.txt"
        if not spec_file.exists():
            pytest.skip("Fixture file not found")

        wells = read_obs_well_spec(spec_file)

        assert wells[2].overwrite_layer == 2

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """Test FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            read_obs_well_spec(tmp_path / "nonexistent.txt")

    def test_minimal_file(self, tmp_path: Path) -> None:
        """Test reading a minimal obs well spec file."""
        spec_file = tmp_path / "wells.txt"
        spec_file.write_text("Name X Y Element BOS TOS\nWELL1 100.0 200.0 1 -50.0 -10.0\n")

        wells = read_obs_well_spec(spec_file)

        assert len(wells) == 1
        assert wells[0].name == "WELL1"
        assert wells[0].overwrite_layer == -1  # default

    def test_empty_file(self, tmp_path: Path) -> None:
        """Test reading an empty file (header only)."""
        spec_file = tmp_path / "wells.txt"
        spec_file.write_text("Name X Y Element BOS TOS OverwriteLayer\n")

        wells = read_obs_well_spec(spec_file)

        assert len(wells) == 0
