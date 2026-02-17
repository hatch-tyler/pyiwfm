"""Extended coverage tests for groundwater sub-file readers.

Covers edge cases, error paths, and additional branches for:
- PumpingReader / read_gw_pumping / PumpingConfig / data classes
- GWBoundaryReader / read_gw_boundary / GWBoundaryConfig / data classes
- TileDrainReader / read_gw_tiledrain / TileDrainConfig / data classes
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pyiwfm.io.gw_pumping import (
    PumpingReader,
    PumpingConfig,
    WellSpec,
    WellPumpingSpec,
    ElementPumpingSpec,
    ElementGroup,
    read_gw_pumping,
    _is_comment_line,
    _strip_comment,
    DIST_USER_FRAC,
    DIST_TOTAL_AREA,
    DIST_AG_URB_AREA,
    DIST_AG_AREA,
    DIST_URB_AREA,
    DEST_SAME_ELEMENT,
    DEST_OUTSIDE,
    DEST_ELEMENT,
    DEST_SUBREGION,
    DEST_ELEM_GROUP,
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
from pyiwfm.io.gw_tiledrain import (
    TileDrainReader,
    TileDrainConfig,
    TileDrainSpec,
    SubIrrigationSpec,
    read_gw_tiledrain,
    TD_DEST_OUTSIDE,
    TD_DEST_STREAM,
)
from pyiwfm.core.exceptions import FileFormatError


# =============================================================================
# Helper function tests (gw_pumping module)
# =============================================================================


class TestPumpingHelpers:
    """Tests for module-level helper functions in gw_pumping."""

    def test_is_comment_line_empty(self) -> None:
        assert _is_comment_line("") is True

    def test_is_comment_line_blank(self) -> None:
        assert _is_comment_line("   ") is True

    def test_is_comment_line_c_upper(self) -> None:
        assert _is_comment_line("C this is a comment") is True

    def test_is_comment_line_c_lower(self) -> None:
        assert _is_comment_line("c this is a comment") is True

    def test_is_comment_line_asterisk(self) -> None:
        assert _is_comment_line("* this is a comment") is True

    def test_is_comment_line_data(self) -> None:
        assert _is_comment_line("    10    / NCOLADJ") is False

    def test_strip_comment_hash_not_delimiter(self) -> None:
        """'#' is NOT a comment delimiter in IWFM — only '/' is."""
        val, desc = _strip_comment("    10    # comment")
        assert val == "10    # comment"

    def test_strip_comment_with_slash(self) -> None:
        val, desc = _strip_comment("    10    / comment")
        assert val == "10"

    def test_strip_comment_plain(self) -> None:
        val, desc = _strip_comment("   42")
        assert val == "42"
        assert desc == ""


# =============================================================================
# PumpingReader data class tests
# =============================================================================


class TestPumpingDataClasses:
    """Tests for pumping data class constructors and defaults."""

    def test_well_spec_defaults(self) -> None:
        ws = WellSpec(id=1, x=10.0, y=20.0)
        assert ws.radius == 0.0
        assert ws.perf_top == 0.0
        assert ws.perf_bottom == 0.0

    def test_well_pumping_spec_defaults(self) -> None:
        wp = WellPumpingSpec(well_id=1)
        assert wp.pump_column == 0
        assert wp.pump_fraction == 1.0
        assert wp.dist_method == DIST_USER_FRAC
        assert wp.dest_type == DEST_SAME_ELEMENT
        assert wp.dest_id == 0

    def test_element_pumping_spec_defaults(self) -> None:
        ep = ElementPumpingSpec(element_id=5)
        assert ep.pump_column == 0
        assert ep.layer_factors == []
        assert ep.pump_max_fraction == 0.0

    def test_element_group_defaults(self) -> None:
        eg = ElementGroup(id=1)
        assert eg.elements == []

    def test_element_group_with_elements(self) -> None:
        eg = ElementGroup(id=1, elements=[10, 20, 30])
        assert len(eg.elements) == 3

    def test_pumping_config_properties_empty(self) -> None:
        config = PumpingConfig()
        assert config.n_wells == 0
        assert config.n_elem_pumping == 0

    def test_pumping_config_properties_with_data(self) -> None:
        config = PumpingConfig()
        config.well_specs = [WellSpec(id=i, x=0, y=0) for i in range(5)]
        config.elem_pumping_specs = [ElementPumpingSpec(element_id=i) for i in range(3)]
        assert config.n_wells == 5
        assert config.n_elem_pumping == 3

    def test_pumping_constants(self) -> None:
        """Verify distribution and destination constant values."""
        assert DIST_USER_FRAC == 0
        assert DIST_TOTAL_AREA == 1
        assert DIST_AG_URB_AREA == 2
        assert DIST_AG_AREA == 3
        assert DIST_URB_AREA == 4
        assert DEST_SAME_ELEMENT == -1
        assert DEST_OUTSIDE == 0
        assert DEST_ELEMENT == 1
        assert DEST_SUBREGION == 2
        assert DEST_ELEM_GROUP == 3


# =============================================================================
# PumpingReader extended tests
# =============================================================================


class TestPumpingReaderExtended:
    """Extended tests for PumpingReader."""

    def _write_file(self, path: Path, name: str, content: str) -> Path:
        filepath = path / name
        filepath.write_text(content)
        return filepath

    def test_read_with_element_pump_file(self, tmp_path: Path) -> None:
        """Read pumping main with element pumping sub-file."""
        elem_content = (
            "C Element Pumping File\n"
            "    2                              / NSink\n"
            "    1    1    1.0    0    0.5    -1    0    0    0    0    0.0\n"
            "    2    2    0.8    1    0.3    -1    0    0    0    0    0.0\n"
            "    0                              / NGroup\n"
        )
        self._write_file(tmp_path, "elempump.dat", elem_content)

        main_content = (
            "#4.0\n"
            "C Main pumping\n"
            "                                   / Well file\n"
            "    elempump.dat                   / Element pump file\n"
            "                                   / TS data file\n"
            "                                   / Output file\n"
        )
        filepath = self._write_file(tmp_path, "pump_main.dat", main_content)
        config = PumpingReader().read(filepath, n_layers=1)

        assert config.n_elem_pumping == 2
        ep0 = config.elem_pumping_specs[0]
        assert ep0.element_id == 1
        assert ep0.pump_fraction == pytest.approx(1.0)
        assert ep0.layer_factors == [pytest.approx(0.5)]

    def test_read_elem_pump_two_layers(self, tmp_path: Path) -> None:
        """Read element pumping with 2 layers."""
        elem_content = (
            "    1                              / NSink\n"
            "    1    1    1.0    0    0.4    0.6    -1    0    0    0    0    0.0\n"
            "    0                              / NGroup\n"
        )
        self._write_file(tmp_path, "elempump.dat", elem_content)

        main_content = (
            "#4.0\n"
            "                                   / Well file\n"
            "    elempump.dat                   / Element pump file\n"
            "                                   / TS data file\n"
            "                                   / Output file\n"
        )
        filepath = self._write_file(tmp_path, "pump_main.dat", main_content)
        config = PumpingReader().read(filepath, n_layers=2)

        assert config.n_elem_pumping == 1
        ep = config.elem_pumping_specs[0]
        assert len(ep.layer_factors) == 2
        assert ep.layer_factors[0] == pytest.approx(0.4)
        assert ep.layer_factors[1] == pytest.approx(0.6)

    def test_read_with_element_groups(self, tmp_path: Path) -> None:
        """Read well file with element groups for pumping destinations."""
        well_content = (
            "C Well File with groups\n"
            "    1                              / NWell\n"
            "    1.0                            / FACTXY\n"
            "    1.0                            / FACTR\n"
            "    1.0                            / FACTLEN\n"
            "    1    100.0    200.0    4.0    -10.0    -50.0\n"
            "C Pumping specs\n"
            "    1    1    1.0    0    3    1    0    0    0    0.0\n"
            "    1                              / NGroup\n"
            "    1    2    10\n"
            "    11\n"
        )
        self._write_file(tmp_path, "wells.dat", well_content)

        main_content = (
            "#4.0\n"
            "    wells.dat                      / Well file\n"
            "                                   / Element pump file\n"
            "                                   / TS data file\n"
            "                                   / Output file\n"
        )
        filepath = self._write_file(tmp_path, "pump_main.dat", main_content)
        config = PumpingReader().read(filepath)

        assert len(config.well_groups) == 1
        assert config.well_groups[0].id == 1
        assert config.well_groups[0].elements == [10, 11]

    def test_well_radius_halved(self, tmp_path: Path) -> None:
        """Verify well radius is diameter/2."""
        well_content = (
            "    1                              / NWell\n"
            "    1.0                            / FACTXY\n"
            "    1.0                            / FACTR\n"
            "    1.0                            / FACTLEN\n"
            "    1    0.0    0.0    10.0    -5.0    -50.0\n"
            "    1    1    1.0    0    -1    0    0    0    0    0.0\n"
            "    0                              / NGroup\n"
        )
        self._write_file(tmp_path, "wells.dat", well_content)

        main_content = (
            "#4.0\n"
            "    wells.dat                      / Well file\n"
            "                                   / Element pump file\n"
            "                                   / TS data file\n"
            "                                   / Output file\n"
        )
        filepath = self._write_file(tmp_path, "pump_main.dat", main_content)
        config = PumpingReader().read(filepath)

        assert config.well_specs[0].radius == pytest.approx(5.0)

    def test_conversion_factors_applied(self, tmp_path: Path) -> None:
        """Verify FactXY, FactR, FactLen are applied to well specs."""
        well_content = (
            "    1                              / NWell\n"
            "    2.0                            / FACTXY\n"
            "    3.0                            / FACTR\n"
            "    4.0                            / FACTLEN\n"
            "    1    100.0    200.0    6.0    -10.0    -50.0\n"
            "    1    1    1.0    0    -1    0    0    0    0    0.0\n"
            "    0                              / NGroup\n"
        )
        self._write_file(tmp_path, "wells.dat", well_content)

        main_content = (
            "#4.0\n"
            "    wells.dat                      / Well file\n"
            "                                   / Element pump file\n"
            "                                   / TS data file\n"
            "                                   / Output file\n"
        )
        filepath = self._write_file(tmp_path, "pump_main.dat", main_content)
        config = PumpingReader().read(filepath)

        ws = config.well_specs[0]
        assert ws.x == pytest.approx(200.0)      # 100 * 2.0
        assert ws.y == pytest.approx(400.0)      # 200 * 2.0
        assert ws.radius == pytest.approx(9.0)   # (6.0 / 2) * 3.0
        assert ws.perf_top == pytest.approx(-40.0)    # -10 * 4.0
        assert ws.perf_bottom == pytest.approx(-200.0) # -50 * 4.0

    def test_resolve_absolute_path(self, tmp_path: Path) -> None:
        """Absolute paths should not be resolved relative to base_dir."""
        from pyiwfm.io.iwfm_reader import resolve_path

        abs_path = str(tmp_path / "file.dat")
        result = resolve_path(Path("/other"), abs_path)
        assert result == Path(abs_path)

    def test_resolve_relative_path(self, tmp_path: Path) -> None:
        """Relative paths should be resolved relative to base_dir."""
        from pyiwfm.io.iwfm_reader import resolve_path

        result = resolve_path(tmp_path, "subdir/file.dat")
        assert result == tmp_path / "subdir" / "file.dat"

    def test_read_well_file_zero_wells(self, tmp_path: Path) -> None:
        """Well file with zero wells should produce empty config."""
        well_content = (
            "    0                              / NWell\n"
        )
        self._write_file(tmp_path, "wells.dat", well_content)

        main_content = (
            "#4.0\n"
            "    wells.dat                      / Well file\n"
            "                                   / Element pump file\n"
            "                                   / TS data file\n"
            "                                   / Output file\n"
        )
        filepath = self._write_file(tmp_path, "pump_main.dat", main_content)
        config = PumpingReader().read(filepath)

        assert config.n_wells == 0

    def test_elem_pump_zero_sinks(self, tmp_path: Path) -> None:
        """Element pump file with zero sinks should produce empty config."""
        elem_content = (
            "    0                              / NSink\n"
        )
        self._write_file(tmp_path, "elempump.dat", elem_content)

        main_content = (
            "#4.0\n"
            "                                   / Well file\n"
            "    elempump.dat                   / Element pump file\n"
            "                                   / TS data file\n"
            "                                   / Output file\n"
        )
        filepath = self._write_file(tmp_path, "pump_main.dat", main_content)
        config = PumpingReader().read(filepath)

        assert config.n_elem_pumping == 0

    def test_read_gw_pumping_with_base_dir(self, tmp_path: Path) -> None:
        """Test read_gw_pumping convenience function with explicit base_dir."""
        subdir = tmp_path / "sub"
        subdir.mkdir()
        main_content = (
            "#4.0\n"
            "                                   / Well file\n"
            "                                   / Element pump file\n"
            "                                   / TS data file\n"
            "                                   / Output file\n"
        )
        filepath = subdir / "pump_main.dat"
        filepath.write_text(main_content)

        config = read_gw_pumping(filepath, base_dir=subdir, n_layers=2)
        assert config.version == "4.0"


# =============================================================================
# GWBoundaryReader extended tests
# =============================================================================


class TestGWBoundaryReaderExtended:
    """Extended tests for GWBoundaryReader."""

    def _write_file(self, path: Path, name: str, content: str) -> Path:
        filepath = path / name
        filepath.write_text(content)
        return filepath

    def test_read_general_head_bc(self, tmp_path: Path) -> None:
        """Read with general head BC sub-file."""
        gh_content = (
            "C General head BCs\n"
            "    2                              / NGB\n"
            "    1.0                            / FACTH\n"
            "    2.0                            / FACTC\n"
            "    1DAY                           / TimeUnit\n"
            "    1    1    0    100.0    0.5\n"
            "    2    2    1    200.0    0.8\n"
        )
        self._write_file(tmp_path, "gh_bc.dat", gh_content)

        main_content = (
            "C GW BC Main\n"
            "                                   / SFBC\n"
            "                                   / SHBC\n"
            "    gh_bc.dat                      / GHBC\n"
            "                                   / CGHBC\n"
            "                                   / TS data\n"
        )
        filepath = self._write_file(tmp_path, "bc_main.dat", main_content)
        config = GWBoundaryReader().read(filepath)

        assert config.n_general_head == 2
        assert config.gh_head_factor == pytest.approx(1.0)
        assert config.gh_conductance_factor == pytest.approx(2.0)
        assert config.gh_time_unit == "1DAY"

        gh0 = config.general_head_bcs[0]
        assert gh0.node_id == 1
        assert gh0.external_head == pytest.approx(100.0)
        assert gh0.conductance == pytest.approx(1.0)  # 0.5 * 2.0

    def test_read_constrained_gh_bc(self, tmp_path: Path) -> None:
        """Read with constrained general head BC sub-file."""
        cgh_content = (
            "C Constrained GH BCs\n"
            "    1                              / NCGB\n"
            "    1.0                            / FACTH\n"
            "    1.0                            / FACTVL\n"
            "    1DAY                           / TimeUnit Head\n"
            "    1.0                            / FACTC\n"
            "    1DAY                           / TimeUnit Cond\n"
            "    5    1    0    150.0    0.3    120.0    1    500.0\n"
        )
        self._write_file(tmp_path, "cgh_bc.dat", cgh_content)

        main_content = (
            "                                   / SFBC\n"
            "                                   / SHBC\n"
            "                                   / GHBC\n"
            "    cgh_bc.dat                     / CGHBC\n"
            "                                   / TS data\n"
        )
        filepath = self._write_file(tmp_path, "bc_main.dat", main_content)
        config = GWBoundaryReader().read(filepath)

        assert config.n_constrained_gh == 1
        cgh = config.constrained_gh_bcs[0]
        assert cgh.node_id == 5
        assert cgh.external_head == pytest.approx(150.0)
        assert cgh.conductance == pytest.approx(0.3)
        assert cgh.constraining_head == pytest.approx(120.0)
        assert cgh.max_flow_ts_column == 1
        assert cgh.max_flow == pytest.approx(500.0)

    def test_bc_config_count_properties(self) -> None:
        """Verify all count properties on GWBoundaryConfig."""
        config = GWBoundaryConfig()
        assert config.n_specified_flow == 0
        assert config.n_specified_head == 0
        assert config.n_general_head == 0
        assert config.n_constrained_gh == 0
        assert config.total_bcs == 0

        config.specified_flow_bcs = [SpecifiedFlowBC(node_id=1, layer=1)]
        config.specified_head_bcs = [SpecifiedHeadBC(node_id=2, layer=1)]
        config.general_head_bcs = [GeneralHeadBC(node_id=3, layer=1)]
        config.constrained_gh_bcs = [
            ConstrainedGeneralHeadBC(node_id=4, layer=1),
            ConstrainedGeneralHeadBC(node_id=5, layer=1),
        ]
        assert config.n_specified_flow == 1
        assert config.n_specified_head == 1
        assert config.n_general_head == 1
        assert config.n_constrained_gh == 2
        assert config.total_bcs == 5

    def test_specified_flow_factor_applied(self, tmp_path: Path) -> None:
        """Verify flow factor multiplies base_flow values."""
        sf_content = (
            "    1                              / NQB\n"
            "    2.0                            / FACT\n"
            "    1DAY                           / TimeUnit\n"
            "    1    1    0    50.0\n"
        )
        self._write_file(tmp_path, "sf_bc.dat", sf_content)

        main_content = (
            "    sf_bc.dat                      / SFBC\n"
            "                                   / SHBC\n"
            "                                   / GHBC\n"
            "                                   / CGHBC\n"
            "                                   / TS data\n"
        )
        filepath = self._write_file(tmp_path, "bc_main.dat", main_content)
        config = GWBoundaryReader().read(filepath)

        assert config.specified_flow_bcs[0].base_flow == pytest.approx(100.0)

    def test_specified_head_factor_applied(self, tmp_path: Path) -> None:
        """Verify head factor multiplies head_value."""
        sh_content = (
            "    1                              / NHB\n"
            "    0.3048                         / FACT\n"
            "    1    1    0    100.0\n"
        )
        self._write_file(tmp_path, "sh_bc.dat", sh_content)

        main_content = (
            "                                   / SFBC\n"
            "    sh_bc.dat                      / SHBC\n"
            "                                   / GHBC\n"
            "                                   / CGHBC\n"
            "                                   / TS data\n"
        )
        filepath = self._write_file(tmp_path, "bc_main.dat", main_content)
        config = GWBoundaryReader().read(filepath)

        assert config.specified_head_bcs[0].head_value == pytest.approx(100.0 * 0.3048)

    def test_zero_count_subfile_returns_empty(self, tmp_path: Path) -> None:
        """Sub-file with zero count should return empty list."""
        sf_content = (
            "    0                              / NQB\n"
        )
        self._write_file(tmp_path, "sf_bc.dat", sf_content)

        main_content = (
            "    sf_bc.dat                      / SFBC\n"
            "                                   / SHBC\n"
            "                                   / GHBC\n"
            "                                   / CGHBC\n"
            "                                   / TS data\n"
        )
        filepath = self._write_file(tmp_path, "bc_main.dat", main_content)
        config = GWBoundaryReader().read(filepath)

        assert config.n_specified_flow == 0

    def test_data_class_defaults(self) -> None:
        """Test BC data class default attribute values."""
        sf = SpecifiedFlowBC(node_id=1, layer=1)
        assert sf.ts_column == 0
        assert sf.base_flow == 0.0

        sh = SpecifiedHeadBC(node_id=2, layer=1)
        assert sh.ts_column == 0
        assert sh.head_value == 0.0

        gh = GeneralHeadBC(node_id=3, layer=1)
        assert gh.conductance == 0.0
        assert gh.external_head == 0.0

        cgh = ConstrainedGeneralHeadBC(node_id=4, layer=1)
        assert cgh.constraining_head == 0.0
        assert cgh.max_flow == 0.0
        assert cgh.max_flow_ts_column == 0


# =============================================================================
# TileDrainReader extended tests
# =============================================================================


class TestTileDrainReaderExtended:
    """Extended tests for TileDrainReader."""

    def _write_file(self, path: Path, content: str) -> Path:
        filepath = path / "tiledrain.dat"
        filepath.write_text(content)
        return filepath

    def test_drain_conductance_factor_applied(self, tmp_path: Path) -> None:
        """Verify conductance factor is multiplied into drain conductance."""
        content = (
            "    1                              / NDrain\n"
            "    1.0                            / FACTHD\n"
            "    5.0                            / FACTCDC\n"
            "    1DAY                           / TUNITDR\n"
            "    1    10    100.0    0.2    1    0\n"
            "    0                              / NSubIrig\n"
        )
        filepath = self._write_file(tmp_path, content)
        config = TileDrainReader().read(filepath)

        assert config.tile_drains[0].conductance == pytest.approx(1.0)  # 0.2 * 5.0

    def test_subirig_both_factors_applied(self, tmp_path: Path) -> None:
        """Verify both height and conductance factors for sub-irrigation."""
        content = (
            "    0                              / NDrain\n"
            "    1                              / NSubIrig\n"
            "    2.0                            / FACTHSI\n"
            "    3.0                            / FACTCDCSI\n"
            "    1MON                           / TUNITSI\n"
            "    1    15    10.0    0.5\n"
        )
        filepath = self._write_file(tmp_path, content)
        config = TileDrainReader().read(filepath)

        si = config.sub_irrigations[0]
        assert si.elevation == pytest.approx(20.0)   # 10.0 * 2.0
        assert si.conductance == pytest.approx(1.5)  # 0.5 * 3.0

    def test_tiledrain_config_defaults(self) -> None:
        """Test TileDrainConfig default values."""
        config = TileDrainConfig()
        assert config.n_drains == 0
        assert config.drain_height_factor == 1.0
        assert config.drain_conductance_factor == 1.0
        assert config.drain_time_unit == ""
        assert config.n_sub_irrigation == 0
        assert config.subirig_height_factor == 1.0
        assert config.subirig_conductance_factor == 1.0
        assert config.subirig_time_unit == ""

    def test_tile_drain_spec_defaults(self) -> None:
        """Test TileDrainSpec defaults for optional fields."""
        td = TileDrainSpec(id=1, gw_node=10, elevation=50.0, conductance=0.5)
        assert td.dest_type == TD_DEST_OUTSIDE
        assert td.dest_id == 0

    def test_sub_irrigation_spec_constructor(self) -> None:
        """Test SubIrrigationSpec constructor."""
        si = SubIrrigationSpec(id=2, gw_node=20, elevation=60.0, conductance=0.3)
        assert si.id == 2
        assert si.gw_node == 20
        assert si.elevation == 60.0
        assert si.conductance == 0.3

    def test_dest_type_constants(self) -> None:
        """Verify tile drain destination type constants."""
        assert TD_DEST_OUTSIDE == 1
        assert TD_DEST_STREAM == 2

    def test_read_string_path(self, tmp_path: Path) -> None:
        """Reader accepts string paths."""
        content = (
            "    0                              / NDrain\n"
            "    0                              / NSubIrig\n"
        )
        filepath = self._write_file(tmp_path, content)
        config = read_gw_tiledrain(str(filepath))
        assert config.n_drains == 0

    def test_multiple_comments_between_sections(self, tmp_path: Path) -> None:
        """Multiple comment lines between drain and sub-irrigation sections."""
        content = (
            "C Comment block 1\n"
            "C Comment block 2\n"
            "c Comment block 3\n"
            "* Comment block 4\n"
            "    1                              / NDrain\n"
            "    1.0                            / FACTHD\n"
            "    1.0                            / FACTCDC\n"
            "    1DAY                           / TUNITDR\n"
            "C  Data comment\n"
            "    1    10    100.0    0.5    1    0\n"
            "C  Section separator\n"
            "C  Another comment\n"
            "    1                              / NSubIrig\n"
            "    1.0                            / FACTHSI\n"
            "    1.0                            / FACTCDCSI\n"
            "    1DAY                           / TUNITSI\n"
            "    1    20    50.0    0.3\n"
        )
        filepath = self._write_file(tmp_path, content)
        config = TileDrainReader().read(filepath)

        assert config.n_drains == 1
        assert config.n_sub_irrigation == 1
