"""Tests for groundwater sub-file readers.

Covers:
- TileDrainReader / read_gw_tiledrain
- GWBoundaryReader / read_gw_boundary
- PumpingReader / read_gw_pumping
- SubsidenceReader / read_gw_subsidence
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyiwfm.io.gw_tiledrain import (
    TileDrainReader,
    TileDrainConfig,
    TileDrainSpec,
    SubIrrigationSpec,
    read_gw_tiledrain,
    TD_DEST_OUTSIDE,
    TD_DEST_STREAM,
)
from pyiwfm.io.gw_boundary import (
    GWBoundaryReader,
    GWBoundaryConfig,
    SpecifiedFlowBC,
    SpecifiedHeadBC,
    GeneralHeadBC,
    ConstrainedGeneralHeadBC,
    read_gw_boundary,
)
from pyiwfm.io.gw_pumping import (
    PumpingReader,
    PumpingConfig,
    WellSpec,
    WellPumpingSpec,
    ElementPumpingSpec,
    ElementGroup,
    read_gw_pumping,
)
from pyiwfm.io.gw_subsidence import (
    SubsidenceReader,
    SubsidenceConfig,
    read_gw_subsidence,
)
from pyiwfm.core.exceptions import FileFormatError


# =============================================================================
# TileDrainReader Tests
# =============================================================================


class TestTileDrainReader:
    """Tests for TileDrainReader."""

    def _write_tiledrain_file(self, path: Path, content: str) -> Path:
        filepath = path / "tiledrain.dat"
        filepath.write_text(content)
        return filepath

    def test_read_drains_only(self, tmp_path: Path) -> None:
        """Read file with tile drains and no sub-irrigation."""
        content = (
            "C Tile Drain File\n"
            "C\n"
            "    3                              / NDrain\n"
            "    1.0                            / FACTHD\n"
            "    1.0                            / FACTCDC\n"
            "    1DAY                           / TUNITDR\n"
            "    1    10    100.0    0.5    2    5\n"
            "    2    20    110.0    0.6    1    0\n"
            "    3    30    120.0    0.7    2    8\n"
            "    0                              / NSubIrig\n"
        )
        filepath = self._write_tiledrain_file(tmp_path, content)
        config = TileDrainReader().read(filepath)

        assert config.n_drains == 3
        assert config.drain_height_factor == 1.0
        assert config.drain_conductance_factor == 1.0
        assert config.drain_time_unit == "1DAY"
        assert len(config.tile_drains) == 3

        td = config.tile_drains[0]
        assert td.id == 1
        assert td.gw_node == 10
        assert td.elevation == pytest.approx(100.0)
        assert td.conductance == pytest.approx(0.5)
        assert td.dest_type == TD_DEST_STREAM
        assert td.dest_id == 5

        td2 = config.tile_drains[1]
        assert td2.dest_type == TD_DEST_OUTSIDE
        assert td2.dest_id == 0

        assert config.n_sub_irrigation == 0
        assert len(config.sub_irrigations) == 0

    def test_read_with_sub_irrigation(self, tmp_path: Path) -> None:
        """Read file with both tile drains and sub-irrigation."""
        content = (
            "C Tile Drain and Sub-Irrigation File\n"
            "    2                              / NDrain\n"
            "    1.0                            / FACTHD\n"
            "    2.0                            / FACTCDC\n"
            "    1DAY                           / TUNITDR\n"
            "    1    10    50.0    0.3    2    5\n"
            "    2    20    60.0    0.4    1    0\n"
            "    2                              / NSubIrig\n"
            "    1.0                            / FACTHSI\n"
            "    3.0                            / FACTCDCSI\n"
            "    1MON                           / TUNITSI\n"
            "    1    15    45.0    0.2\n"
            "    2    25    55.0    0.3\n"
        )
        filepath = self._write_tiledrain_file(tmp_path, content)
        config = TileDrainReader().read(filepath)

        assert config.n_drains == 2
        assert config.drain_conductance_factor == 2.0
        # Values are multiplied by factor in reader
        assert config.tile_drains[0].conductance == pytest.approx(0.6)  # 0.3 * 2.0
        assert config.tile_drains[1].conductance == pytest.approx(0.8)  # 0.4 * 2.0

        assert config.n_sub_irrigation == 2
        assert config.subirig_conductance_factor == 3.0
        assert config.subirig_time_unit == "1MON"

        si = config.sub_irrigations[0]
        assert si.id == 1
        assert si.gw_node == 15
        assert si.elevation == pytest.approx(45.0)
        assert si.conductance == pytest.approx(0.6)  # 0.2 * 3.0

    def test_read_zero_drains(self, tmp_path: Path) -> None:
        """Read file with zero drains."""
        content = (
            "C Empty tile drain file\n"
            "    0                              / NDrain\n"
            "    0                              / NSubIrig\n"
        )
        filepath = self._write_tiledrain_file(tmp_path, content)
        config = TileDrainReader().read(filepath)

        assert config.n_drains == 0
        assert len(config.tile_drains) == 0
        assert config.n_sub_irrigation == 0
        assert len(config.sub_irrigations) == 0

    def test_height_factor_applied(self, tmp_path: Path) -> None:
        """Verify height factor is applied to elevation values."""
        content = (
            "    1                              / NDrain\n"
            "    0.3048                         / FACTHD (feet to meters)\n"
            "    1.0                            / FACTCDC\n"
            "    1DAY                           / TUNITDR\n"
            "    1    10    100.0    0.5    1    0\n"
            "    0                              / NSubIrig\n"
        )
        filepath = self._write_tiledrain_file(tmp_path, content)
        config = TileDrainReader().read(filepath)

        assert config.drain_height_factor == pytest.approx(0.3048)
        assert config.tile_drains[0].elevation == pytest.approx(100.0 * 0.3048)

    def test_convenience_function(self, tmp_path: Path) -> None:
        """Test read_gw_tiledrain convenience function."""
        content = (
            "    1                              / NDrain\n"
            "    1.0                            / FACTHD\n"
            "    1.0                            / FACTCDC\n"
            "    1DAY                           / TUNITDR\n"
            "    1    10    100.0    0.5    1    0\n"
            "    0                              / NSubIrig\n"
        )
        filepath = self._write_tiledrain_file(tmp_path, content)
        config = read_gw_tiledrain(filepath)
        assert config.n_drains == 1
        assert len(config.tile_drains) == 1

    def test_comments_skipped(self, tmp_path: Path) -> None:
        """Verify comment lines in all positions are properly skipped."""
        content = (
            "C Header comment\n"
            "c lowercase comment\n"
            "* asterisk comment\n"
            "    2                              / NDrain\n"
            "C  comment between parameters\n"
            "    1.0                            / FACTHD\n"
            "    1.0                            / FACTCDC\n"
            "    1DAY                           / TUNITDR\n"
            "C  comment between data\n"
            "    1    10    100.0    0.5    1    0\n"
            "    2    20    200.0    0.6    2    3\n"
            "    0                              / NSubIrig\n"
        )
        filepath = self._write_tiledrain_file(tmp_path, content)
        config = TileDrainReader().read(filepath)
        assert config.n_drains == 2
        assert len(config.tile_drains) == 2


# =============================================================================
# GWBoundaryReader Tests
# =============================================================================


class TestGWBoundaryReader:
    """Tests for GWBoundaryReader."""

    def _write_bc_main(self, path: Path, content: str) -> Path:
        filepath = path / "gwbc_main.dat"
        filepath.write_text(content)
        return filepath

    def _write_subfile(self, path: Path, name: str, content: str) -> Path:
        filepath = path / name
        filepath.write_text(content)
        return filepath

    def test_read_main_with_no_subfiles(self, tmp_path: Path) -> None:
        """Read main BC file referencing no sub-files (all blank)."""
        content = (
            "C GW Boundary Conditions\n"
            "                                   / SFBC file (blank)\n"
            "                                   / SHBC file (blank)\n"
            "                                   / GHBC file (blank)\n"
            "                                   / CGHBC file (blank)\n"
            "                                   / TS data file (blank)\n"
        )
        filepath = self._write_bc_main(tmp_path, content)
        config = GWBoundaryReader().read(filepath)

        assert config.sp_flow_file is None
        assert config.sp_head_file is None
        assert config.gh_file is None
        assert config.cgh_file is None
        assert config.total_bcs == 0

    def test_read_specified_flow(self, tmp_path: Path) -> None:
        """Read with specified flow BC sub-file."""
        # Create specified flow sub-file
        sf_content = (
            "C Specified flow BCs\n"
            "    2                              / NSFBC\n"
            "    1.0                            / FACTQSF\n"
            "    1DAY                           / TUNITSF\n"
            "    1    1    0    100.0\n"
            "    2    2    1    200.0\n"
        )
        self._write_subfile(tmp_path, "sf_bc.dat", sf_content)

        main_content = (
            "C GW Boundary Conditions\n"
            "    sf_bc.dat                      / SFBC file\n"
            "                                   / SHBC file\n"
            "                                   / GHBC file\n"
            "                                   / CGHBC file\n"
            "                                   / TS data file\n"
        )
        filepath = self._write_bc_main(tmp_path, main_content)
        config = GWBoundaryReader().read(filepath)

        assert config.sp_flow_file is not None
        assert config.n_specified_flow == 2
        assert config.specified_flow_bcs[0].node_id == 1
        assert config.specified_flow_bcs[0].layer == 1
        assert config.specified_flow_bcs[1].base_flow == pytest.approx(200.0)

    def test_read_specified_head(self, tmp_path: Path) -> None:
        """Read with specified head BC sub-file."""
        sh_content = (
            "C Specified head BCs\n"
            "    1                              / NSHBC\n"
            "    1.0                            / FACTHSH\n"
            "    5    1    0    150.0\n"
        )
        self._write_subfile(tmp_path, "sh_bc.dat", sh_content)

        main_content = (
            "C GW Boundary Conditions\n"
            "                                   / SFBC file\n"
            "    sh_bc.dat                      / SHBC file\n"
            "                                   / GHBC file\n"
            "                                   / CGHBC file\n"
            "                                   / TS data file\n"
        )
        filepath = self._write_bc_main(tmp_path, main_content)
        config = GWBoundaryReader().read(filepath)

        assert config.n_specified_head == 1
        assert config.specified_head_bcs[0].node_id == 5
        assert config.specified_head_bcs[0].head_value == pytest.approx(150.0)

    def test_total_bcs_property(self, tmp_path: Path) -> None:
        """Verify total_bcs property counts all BC types."""
        config = GWBoundaryConfig()
        config.specified_flow_bcs = [SpecifiedFlowBC(node_id=1, layer=1)]
        config.specified_head_bcs = [SpecifiedHeadBC(node_id=2, layer=1)]
        config.general_head_bcs = [
            GeneralHeadBC(node_id=3, layer=1, external_head=100.0, conductance=0.5),
            GeneralHeadBC(node_id=4, layer=1, external_head=110.0, conductance=0.6),
        ]
        assert config.total_bcs == 4

    def test_convenience_function(self, tmp_path: Path) -> None:
        """Test read_gw_boundary convenience function."""
        content = (
            "C GW Boundary Conditions\n"
            "                                   / SFBC file\n"
            "                                   / SHBC file\n"
            "                                   / GHBC file\n"
            "                                   / CGHBC file\n"
            "                                   / TS data file\n"
        )
        filepath = self._write_bc_main(tmp_path, content)
        config = read_gw_boundary(filepath)
        assert config.total_bcs == 0


# =============================================================================
# PumpingReader Tests
# =============================================================================


class TestPumpingReader:
    """Tests for PumpingReader."""

    def _write_pump_main(self, path: Path, content: str) -> Path:
        filepath = path / "pump_main.dat"
        filepath.write_text(content)
        return filepath

    def _write_subfile(self, path: Path, name: str, content: str) -> Path:
        filepath = path / name
        filepath.write_text(content)
        return filepath

    def test_read_main_no_subfiles(self, tmp_path: Path) -> None:
        """Read pumping main file with no sub-files."""
        content = (
            "C Pumping Main File\n"
            "#4.0\n"
            "C *** DO NOT DELETE ABOVE LINE ***\n"
            "                                   / Well file\n"
            "                                   / Element pump file\n"
            "                                   / TS data file\n"
            "                                   / Output file\n"
        )
        filepath = self._write_pump_main(tmp_path, content)
        config = PumpingReader().read(filepath)

        assert config.version == "4.0"
        assert config.n_wells == 0
        assert config.n_elem_pumping == 0

    def test_read_with_well_file(self, tmp_path: Path) -> None:
        """Read pumping main with well specification sub-file."""
        well_content = (
            "C Well Specification File\n"
            "    2                              / NWELL\n"
            "    1.0                            / FACTXY\n"
            "    1.0                            / FACTR\n"
            "    1.0                            / FACTLEN\n"
            "    1    100.0    200.0    5.0    -10.0    -50.0\n"
            "    2    150.0    250.0    3.0    -5.0     -40.0\n"
            "C Well Pumping Specs\n"
            "    2                              / NWELLPUMP\n"
            "    1    1    1.0    0    -1    0    0    0    0    0.0\n"
            "    2    2    1.0    0    -1    0    0    0    0    0.0\n"
            "    0                              / NGROUP\n"
        )
        self._write_subfile(tmp_path, "wells.dat", well_content)

        main_content = (
            "#4.0\n"
            "C *** DO NOT DELETE ABOVE LINE ***\n"
            "    wells.dat                      / Well file\n"
            "                                   / Element pump file\n"
            "                                   / TS data file\n"
            "                                   / Output file\n"
        )
        filepath = self._write_pump_main(tmp_path, main_content)
        config = PumpingReader().read(filepath)

        assert config.version == "4.0"
        assert config.n_wells == 2
        assert config.well_specs[0].x == pytest.approx(100.0)
        assert config.well_specs[1].y == pytest.approx(250.0)

    def test_pumping_config_properties(self) -> None:
        """Test PumpingConfig property methods."""
        config = PumpingConfig()
        assert config.n_wells == 0
        assert config.n_elem_pumping == 0

        config.well_specs = [WellSpec(id=1, x=0, y=0)]
        assert config.n_wells == 1

        config.elem_pumping_specs = [
            ElementPumpingSpec(element_id=1),
            ElementPumpingSpec(element_id=2),
        ]
        assert config.n_elem_pumping == 2

    def test_convenience_function(self, tmp_path: Path) -> None:
        """Test read_gw_pumping convenience function."""
        content = (
            "#4.0\n"
            "C *** DO NOT DELETE ABOVE LINE ***\n"
            "                                   / Well file\n"
            "                                   / Element pump file\n"
            "                                   / TS data file\n"
            "                                   / Output file\n"
        )
        filepath = self._write_pump_main(tmp_path, content)
        config = read_gw_pumping(filepath)
        assert config.version == "4.0"
        assert config.n_wells == 0


# =============================================================================
# SubsidenceReader Tests
# =============================================================================


class TestSubsidenceReader:
    """Tests for SubsidenceReader."""

    def _write_subs_file(self, path: Path, content: str) -> Path:
        filepath = path / "subsidence.dat"
        filepath.write_text(content)
        return filepath

    def test_read_v40_minimal(self, tmp_path: Path) -> None:
        """Read minimal v4.0 subsidence file."""
        content = (
            "C Subsidence Parameters\n"
            "#4.0\n"
            "C *** DO NOT DELETE ABOVE LINE ***\n"
            "                                   / IC file\n"
            "                                   / Tecplot file\n"
            "                                   / Final subs file\n"
            "    1.0                            / Output factor\n"
            "    FEET                           / Output unit\n"
            "    0                              / NOUTS\n"
            "    0                              / NGroup\n"
            "    1.0  1.0  1.0  1.0  1.0  1.0  / Conversion factors\n"
        )
        filepath = self._write_subs_file(tmp_path, content)
        config = SubsidenceReader().read(filepath)

        assert config.version == "4.0"
        assert config.output_factor == pytest.approx(1.0)
        assert config.output_unit == "FEET"
        assert config.n_parametric_grids == 0
        assert len(config.conversion_factors) == 6

    def test_read_v50_minimal(self, tmp_path: Path) -> None:
        """Read minimal v5.0 subsidence file with interbed DZ."""
        content = (
            "C Subsidence Parameters\n"
            "#5.0\n"
            "C *** DO NOT DELETE ABOVE LINE ***\n"
            "                                   / IC file\n"
            "                                   / Tecplot file\n"
            "                                   / Final subs file\n"
            "    1.0                            / Output factor\n"
            "    FEET                           / Output unit\n"
            "    0                              / NOUTS\n"
            "    5.0                            / Interbed DZ (v5.0)\n"
            "    0                              / NGroup\n"
            "    1.0  1.0  1.0  1.0  1.0  1.0  1.0  / 7 factors (v5.0)\n"
        )
        filepath = self._write_subs_file(tmp_path, content)
        config = SubsidenceReader().read(filepath)

        assert config.version == "5.0"
        assert config.interbed_dz == pytest.approx(5.0)
        assert len(config.conversion_factors) == 7

    def test_read_with_subfiles(self, tmp_path: Path) -> None:
        """Read file referencing sub-files with paths."""
        ic_content = "C IC placeholder\n"
        (tmp_path / "subs_ic.dat").write_text(ic_content)

        content = (
            "#4.0\n"
            "C *** DO NOT DELETE ABOVE LINE ***\n"
            "    subs_ic.dat                    / IC file\n"
            "                                   / Tecplot file\n"
            "                                   / Final subs file\n"
            "    1.0                            / Output factor\n"
            "    FEET                           / Output unit\n"
            "    0                              / NOUTS\n"
            "    0                              / NGroup\n"
            "    1.0  1.0  1.0  1.0  1.0  1.0  / Conversion factors\n"
        )
        filepath = self._write_subs_file(tmp_path, content)
        config = SubsidenceReader().read(filepath)

        assert config.ic_file is not None
        assert config.ic_file.name == "subs_ic.dat"

    def test_convenience_function(self, tmp_path: Path) -> None:
        """Test read_gw_subsidence convenience function."""
        content = (
            "#4.0\n"
            "C *** DO NOT DELETE ABOVE LINE ***\n"
            "                                   / IC file\n"
            "                                   / Tecplot file\n"
            "                                   / Final subs file\n"
            "    1.0                            / Output factor\n"
            "    FEET                           / Output unit\n"
            "    0                              / NOUTS\n"
            "    0                              / NGroup\n"
            "    1.0  1.0  1.0  1.0  1.0  1.0  / Conversion factors\n"
        )
        filepath = self._write_subs_file(tmp_path, content)
        config = read_gw_subsidence(filepath)
        assert config.version == "4.0"

    def test_config_defaults(self) -> None:
        """Test SubsidenceConfig default values."""
        config = SubsidenceConfig()
        assert config.version == ""
        assert config.output_factor == 1.0
        assert config.output_unit == "FEET"
        assert config.interbed_dz == 0.0
        assert config.n_parametric_grids == 0
        assert config.n_nodes == 0
        assert config.n_layers == 0
