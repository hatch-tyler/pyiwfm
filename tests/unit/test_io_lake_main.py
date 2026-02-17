"""Tests for LakeMainFileReader.

Covers:
- LakeMainFileReader v4.0 format
- LakeMainFileReader v5.0 format (with outflow rating tables)
- read_lake_main_file convenience function
- LakeMainFileConfig dataclass
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pyiwfm.io.lakes import (
    LakeMainFileConfig,
    LakeMainFileReader,
    LakeOutflowRating,
    LakeParamSpec,
    OutflowRatingPoint,
    read_lake_main_file,
)

# =============================================================================
# LakeMainFileConfig defaults
# =============================================================================


class TestLakeMainFileConfigDefaults:
    """Test LakeMainFileConfig default values."""

    def test_defaults(self) -> None:
        config = LakeMainFileConfig()
        assert config.version == ""
        assert config.max_elev_file is None
        assert config.budget_output_file is None
        assert config.final_elev_file is None
        assert config.conductance_factor == 1.0
        assert config.conductance_time_unit == ""
        assert config.depth_factor == 1.0
        assert config.lake_params == []
        assert config.elev_factor == 1.0
        assert config.outflow_factor == 1.0
        assert config.outflow_time_unit == ""
        assert config.outflow_ratings == []


class TestLakeParamSpec:
    """Test LakeParamSpec dataclass."""

    def test_defaults(self) -> None:
        p = LakeParamSpec()
        assert p.lake_id == 0
        assert p.conductance_coeff == 0.0
        assert p.depth_denom == 1.0
        assert p.max_elev_col == 0
        assert p.et_col == 0
        assert p.precip_col == 0
        assert p.name == ""


# =============================================================================
# LakeMainFileReader v4.0
# =============================================================================


class TestLakeMainFileReaderV40:
    """Tests for LakeMainFileReader with v4.0 format."""

    def _write_lake_main(self, path: Path, content: str) -> Path:
        filepath = path / "lake_main.dat"
        filepath.write_text(content)
        return filepath

    def test_read_version(self, tmp_path: Path) -> None:
        """Read v4.0 version header."""
        content = (
            "C Lake Main File\n"
            "#4.0\n"
            "C *** DO NOT DELETE ABOVE LINE ***\n"
            "                                   / MaxElevFile\n"
            "                                   / BudgetFile\n"
            "                                   / FinalElevFile\n"
            "    1.0                            / FactK\n"
            "    1DAY                           / TimeUnit\n"
            "    1.0                            / FactL\n"
        )
        filepath = self._write_lake_main(tmp_path, content)
        config = LakeMainFileReader().read(filepath)

        assert config.version == "4.0"

    def test_read_parameters(self, tmp_path: Path) -> None:
        """Read conductance and depth factors."""
        content = (
            "#4.0\n"
            "C *** DO NOT DELETE ABOVE LINE ***\n"
            "                                   / MaxElevFile\n"
            "                                   / BudgetFile\n"
            "                                   / FinalElevFile\n"
            "    0.5                            / FactK\n"
            "    1MON                           / TimeUnit\n"
            "    2.0                            / FactL\n"
        )
        filepath = self._write_lake_main(tmp_path, content)
        config = LakeMainFileReader().read(filepath)

        assert config.conductance_factor == pytest.approx(0.5)
        assert config.conductance_time_unit == "1MON"
        assert config.depth_factor == pytest.approx(2.0)

    def test_read_subfile_paths(self, tmp_path: Path) -> None:
        """Read sub-file paths."""
        # Create dummy sub-files
        (tmp_path / "max_elev.dat").write_text("C placeholder\n")
        (tmp_path / "budget.dat").write_text("C placeholder\n")

        content = (
            "#4.0\n"
            "C *** DO NOT DELETE ABOVE LINE ***\n"
            "    max_elev.dat                   / MaxElevFile\n"
            "    budget.dat                     / BudgetFile\n"
            "                                   / FinalElevFile\n"
            "    1.0                            / FactK\n"
            "    1DAY                           / TimeUnit\n"
            "    1.0                            / FactL\n"
        )
        filepath = self._write_lake_main(tmp_path, content)
        config = LakeMainFileReader().read(filepath)

        assert config.max_elev_file is not None
        assert config.max_elev_file.name == "max_elev.dat"
        assert config.budget_output_file is not None
        assert config.budget_output_file.name == "budget.dat"
        assert config.final_elev_file is None

    def test_read_lake_params(self, tmp_path: Path) -> None:
        """Read per-lake parameter lines."""
        content = (
            "#4.0\n"
            "C *** DO NOT DELETE ABOVE LINE ***\n"
            "                                   / MaxElevFile\n"
            "                                   / BudgetFile\n"
            "                                   / FinalElevFile\n"
            "    1.0                            / FactK\n"
            "    1DAY                           / TimeUnit\n"
            "    1.0                            / FactL\n"
            "    1    0.001    10.0    1    1    1    Clear Lake\n"
            "    2    0.002    15.0    2    2    2    Folsom Lake\n"
        )
        filepath = self._write_lake_main(tmp_path, content)
        config = LakeMainFileReader().read(filepath)

        assert len(config.lake_params) == 2
        lp1 = config.lake_params[0]
        assert lp1.lake_id == 1
        assert lp1.conductance_coeff == pytest.approx(0.001)
        assert lp1.depth_denom == pytest.approx(10.0)
        assert lp1.max_elev_col == 1
        assert lp1.et_col == 1
        assert lp1.precip_col == 1
        assert lp1.name == "Clear Lake"

        lp2 = config.lake_params[1]
        assert lp2.lake_id == 2
        assert lp2.name == "Folsom Lake"

    def test_read_no_lakes(self, tmp_path: Path) -> None:
        """Read file with no lake parameter lines."""
        content = (
            "#4.0\n"
            "C *** DO NOT DELETE ABOVE LINE ***\n"
            "                                   / MaxElevFile\n"
            "                                   / BudgetFile\n"
            "                                   / FinalElevFile\n"
            "    1.0                            / FactK\n"
            "    1DAY                           / TimeUnit\n"
            "    1.0                            / FactL\n"
        )
        filepath = self._write_lake_main(tmp_path, content)
        config = LakeMainFileReader().read(filepath)

        assert len(config.lake_params) == 0
        assert len(config.outflow_ratings) == 0


# =============================================================================
# LakeMainFileReader v5.0
# =============================================================================


class TestLakeMainFileReaderV50:
    """Tests for LakeMainFileReader with v5.0 format."""

    def _write_lake_main(self, path: Path, content: str) -> Path:
        filepath = path / "lake_main_v50.dat"
        filepath.write_text(content)
        return filepath

    def test_read_version(self, tmp_path: Path) -> None:
        """Read v5.0 version header."""
        content = (
            "#5.0\n"
            "C *** DO NOT DELETE ABOVE LINE ***\n"
            "                                   / MaxElevFile\n"
            "                                   / BudgetFile\n"
            "                                   / FinalElevFile\n"
            "    1.0                            / FactK\n"
            "    1DAY                           / TimeUnit\n"
            "    1.0                            / FactL\n"
        )
        filepath = self._write_lake_main(tmp_path, content)
        config = LakeMainFileReader().read(filepath)
        assert config.version == "5.0"

    def test_read_with_outflow_ratings(self, tmp_path: Path) -> None:
        """Read v5.0 format with outflow rating tables."""
        content = (
            "#5.0\n"
            "C *** DO NOT DELETE ABOVE LINE ***\n"
            "                                   / MaxElevFile\n"
            "                                   / BudgetFile\n"
            "                                   / FinalElevFile\n"
            "    1.0                            / FactK\n"
            "    1DAY                           / TimeUnit\n"
            "    1.0                            / FactL\n"
            "    1    0.001    10.0    1    1    1    TestLake\n"
            "    1.0                            / ElevFactor\n"
            "    1.0                            / OutflowFactor\n"
            "    1DAY                           / OutflowTimeUnit\n"
            "    1    3    100.0    0.0\n"
            "    110.0   50.0\n"
            "    120.0   200.0\n"
        )
        filepath = self._write_lake_main(tmp_path, content)
        config = LakeMainFileReader().read(filepath)

        assert config.version == "5.0"
        assert len(config.lake_params) == 1
        assert config.elev_factor == pytest.approx(1.0)
        assert config.outflow_factor == pytest.approx(1.0)

        assert len(config.outflow_ratings) == 1
        rt = config.outflow_ratings[0]
        assert rt.lake_id == 1
        assert len(rt.points) == 3
        assert rt.points[0].elevation == pytest.approx(100.0)
        assert rt.points[0].outflow == pytest.approx(0.0)
        assert rt.points[2].elevation == pytest.approx(120.0)
        assert rt.points[2].outflow == pytest.approx(200.0)

    def test_read_v50_factors_applied(self, tmp_path: Path) -> None:
        """Read v5.0 and verify elevation/outflow factors are applied."""
        content = (
            "#5.0\n"
            "C *** DO NOT DELETE ABOVE LINE ***\n"
            "                                   / MaxElevFile\n"
            "                                   / BudgetFile\n"
            "                                   / FinalElevFile\n"
            "    1.0                            / FactK\n"
            "    1DAY                           / TimeUnit\n"
            "    1.0                            / FactL\n"
            "    1    0.001    10.0    1    1    1\n"
            "    0.3048                         / ElevFactor (ft to m)\n"
            "    0.0283168                      / OutflowFactor (cfs to cms)\n"
            "    1DAY                           / OutflowTimeUnit\n"
            "    1    2    100.0    500.0\n"
            "    200.0   1000.0\n"
        )
        filepath = self._write_lake_main(tmp_path, content)
        config = LakeMainFileReader().read(filepath)

        assert config.elev_factor == pytest.approx(0.3048)
        assert config.outflow_factor == pytest.approx(0.0283168)
        rt = config.outflow_ratings[0]
        assert rt.points[0].elevation == pytest.approx(100.0 * 0.3048)
        assert rt.points[0].outflow == pytest.approx(500.0 * 0.0283168)

    def test_read_v50_multiple_lakes(self, tmp_path: Path) -> None:
        """Read v5.0 with multiple lakes and rating tables."""
        content = (
            "#5.0\n"
            "C *** DO NOT DELETE ABOVE LINE ***\n"
            "                                   / MaxElevFile\n"
            "                                   / BudgetFile\n"
            "                                   / FinalElevFile\n"
            "    1.0                            / FactK\n"
            "    1DAY                           / TimeUnit\n"
            "    1.0                            / FactL\n"
            "    1    0.001    10.0    1    1    1\n"
            "    2    0.002    20.0    2    2    2\n"
            "    1.0                            / ElevFactor\n"
            "    1.0                            / OutflowFactor\n"
            "    1DAY                           / OutflowTimeUnit\n"
            "    1    2    100.0    0.0\n"
            "    110.0   50.0\n"
            "    2    2    200.0    0.0\n"
            "    210.0   100.0\n"
        )
        filepath = self._write_lake_main(tmp_path, content)
        config = LakeMainFileReader().read(filepath)

        assert len(config.lake_params) == 2
        assert len(config.outflow_ratings) == 2
        assert config.outflow_ratings[0].lake_id == 1
        assert config.outflow_ratings[1].lake_id == 2
        assert len(config.outflow_ratings[1].points) == 2


# =============================================================================
# Convenience function
# =============================================================================


class TestReadLakeMainFile:
    """Tests for read_lake_main_file convenience function."""

    def test_reads_successfully(self, tmp_path: Path) -> None:
        content = (
            "#4.0\n"
            "C *** DO NOT DELETE ABOVE LINE ***\n"
            "                                   / MaxElevFile\n"
            "                                   / BudgetFile\n"
            "                                   / FinalElevFile\n"
            "    1.0                            / FactK\n"
            "    1DAY                           / TimeUnit\n"
            "    1.0                            / FactL\n"
        )
        filepath = tmp_path / "lake.dat"
        filepath.write_text(content)
        config = read_lake_main_file(filepath)
        assert config.version == "4.0"

    def test_custom_base_dir(self, tmp_path: Path) -> None:
        """Test base_dir parameter for path resolution."""
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "max_elev.dat").write_text("C placeholder\n")

        content = (
            "#4.0\n"
            "C *** DO NOT DELETE ABOVE LINE ***\n"
            "    max_elev.dat                   / MaxElevFile\n"
            "                                   / BudgetFile\n"
            "                                   / FinalElevFile\n"
            "    1.0                            / FactK\n"
            "    1DAY                           / TimeUnit\n"
            "    1.0                            / FactL\n"
        )
        filepath = tmp_path / "lake.dat"
        filepath.write_text(content)
        config = read_lake_main_file(filepath, base_dir=sub)

        assert config.max_elev_file is not None
        assert config.max_elev_file.parent == sub


# =============================================================================
# OutflowRatingPoint and LakeOutflowRating
# =============================================================================


class TestOutflowRatingDataclasses:
    """Tests for outflow rating dataclasses."""

    def test_point_defaults(self) -> None:
        p = OutflowRatingPoint()
        assert p.elevation == 0.0
        assert p.outflow == 0.0

    def test_rating_defaults(self) -> None:
        r = LakeOutflowRating()
        assert r.lake_id == 0
        assert r.points == []

    def test_rating_with_points(self) -> None:
        r = LakeOutflowRating(
            lake_id=1,
            points=[
                OutflowRatingPoint(elevation=100.0, outflow=0.0),
                OutflowRatingPoint(elevation=120.0, outflow=500.0),
            ],
        )
        assert len(r.points) == 2
        assert r.points[1].outflow == 500.0
