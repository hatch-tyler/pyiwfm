"""
Tests for groundwater package completeness fixes.

Covers:
- Phase 1: Reader bug fixes (TileDrain version, Subsidence NOUTS, Constrained GH BCs)
- Phase 2: Data accessibility (well names, element pumping, subsidence params, sub-irrigation)
- Phase 3: Writer roundtrips
- Phase 4: Component class enhancements
- Phase 5: WebAPI route updates
"""

from __future__ import annotations

from pathlib import Path

import pytest

# ============================================================================
# Phase 4: Component class tests
# ============================================================================


class TestBoundaryConditionConstrainedGH:
    """Test constrained_general_head BC type."""

    def test_create_constrained_gh_bc(self) -> None:
        from pyiwfm.components.groundwater import BoundaryCondition

        bc = BoundaryCondition(
            id=1,
            bc_type="constrained_general_head",
            nodes=[100],
            values=[50.0],
            layer=1,
            conductance=[0.5],
            constraining_head=30.0,
            max_flow=100.0,
            ts_column=3,
            max_flow_ts_column=5,
        )
        assert bc.bc_type == "constrained_general_head"
        assert bc.constraining_head == 30.0
        assert bc.max_flow == 100.0
        assert bc.ts_column == 3
        assert bc.max_flow_ts_column == 5

    def test_constrained_gh_requires_conductance(self) -> None:
        from pyiwfm.components.groundwater import BoundaryCondition

        with pytest.raises(ValueError, match="constrained_general_head BC requires conductance"):
            BoundaryCondition(
                id=1,
                bc_type="constrained_general_head",
                nodes=[100],
                values=[50.0],
                layer=1,
                conductance=[],  # Missing!
            )

    def test_backward_compat_three_types(self) -> None:
        from pyiwfm.components.groundwater import BoundaryCondition

        for bc_type in ("specified_head", "specified_flow", "general_head"):
            kwargs: dict = {
                "id": 1,
                "bc_type": bc_type,
                "nodes": [1],
                "values": [10.0],
                "layer": 1,
            }
            if bc_type == "general_head":
                kwargs["conductance"] = [0.1]
            bc = BoundaryCondition(**kwargs)
            assert bc.bc_type == bc_type

    def test_invalid_bc_type_raises(self) -> None:
        from pyiwfm.components.groundwater import BoundaryCondition

        with pytest.raises(ValueError, match="bc_type must be one of"):
            BoundaryCondition(
                id=1,
                bc_type="invalid_type",
                nodes=[1],
                values=[10.0],
                layer=1,
            )


class TestWellExpanded:
    """Test expanded Well fields."""

    def test_well_new_fields_defaults(self) -> None:
        from pyiwfm.components.groundwater import Well

        w = Well(id=1, x=100.0, y=200.0, element=5)
        assert w.radius == 0.0
        assert w.pump_column == 0
        assert w.pump_fraction == 1.0
        assert w.dist_method == 0
        assert w.dest_type == -1
        assert w.dest_id == 0

    def test_well_with_all_fields(self) -> None:
        from pyiwfm.components.groundwater import Well

        w = Well(
            id=1,
            x=100.0,
            y=200.0,
            element=5,
            name="Test Well",
            radius=0.5,
            pump_column=3,
            pump_fraction=0.8,
            dist_method=2,
            dest_type=1,
            dest_id=42,
            irig_frac_column=7,
            adjust_column=8,
            pump_max_column=9,
            pump_max_fraction=0.95,
        )
        assert w.name == "Test Well"
        assert w.radius == 0.5
        assert w.pump_column == 3
        assert w.dest_id == 42


class TestElementPumpingExpanded:
    """Test expanded ElementPumping fields."""

    def test_backward_compat(self) -> None:
        from pyiwfm.components.groundwater import ElementPumping

        ep = ElementPumping(element_id=10, layer=1, pump_rate=100.0)
        assert ep.effective_rate == 100.0
        assert ep.pump_column == 0
        assert ep.layer_factors == []

    def test_full_fields(self) -> None:
        from pyiwfm.components.groundwater import ElementPumping

        ep = ElementPumping(
            element_id=10,
            layer=0,
            pump_rate=0.0,
            pump_column=5,
            pump_fraction=0.5,
            dist_method=3,
            layer_factors=[0.3, 0.7],
            dest_type=2,
            dest_id=15,
        )
        assert ep.pump_column == 5
        assert ep.layer_factors == [0.3, 0.7]


class TestSubsidenceExpanded:
    """Test expanded Subsidence and NodeSubsidence."""

    def test_subsidence_new_fields(self) -> None:
        from pyiwfm.components.groundwater import Subsidence

        s = Subsidence(
            element=100,
            layer=1,
            elastic_storage=0.001,
            inelastic_storage=0.01,
            preconsolidation_head=50.0,
            interbed_thick=10.0,
            interbed_thick_min=2.0,
        )
        assert s.interbed_thick == 10.0
        assert s.interbed_thick_min == 2.0
        assert s.node == 100  # alias for element

    def test_node_subsidence(self) -> None:
        from pyiwfm.components.groundwater import NodeSubsidence

        ns = NodeSubsidence(
            node_id=42,
            elastic_sc=[0.001, 0.002],
            inelastic_sc=[0.01, 0.02],
            interbed_thick=[10.0, 5.0],
            interbed_thick_min=[2.0, 1.0],
            precompact_head=[50.0, 45.0],
        )
        assert ns.node_id == 42
        assert len(ns.elastic_sc) == 2

    def test_sub_irrigation(self) -> None:
        from pyiwfm.components.groundwater import SubIrrigation

        si = SubIrrigation(id=1, gw_node=100, elevation=50.0, conductance=0.5)
        assert si.gw_node == 100
        assert repr(si) == "SubIrrigation(id=1, node=100)"


class TestAppGWExpanded:
    """Test expanded AppGW fields."""

    def test_new_collections_empty(self) -> None:
        from pyiwfm.components.groundwater import AppGW

        gw = AppGW(n_nodes=10, n_layers=2, n_elements=5)
        assert gw.node_subsidence == []
        assert gw.sub_irrigations == []
        assert gw.subsidence_config is None
        assert gw.pumping_ts_file is None
        assert gw.bc_ts_file is None

    def test_add_node_subsidence(self) -> None:
        from pyiwfm.components.groundwater import AppGW, NodeSubsidence

        gw = AppGW(n_nodes=10, n_layers=2, n_elements=5)
        ns = NodeSubsidence(node_id=1, elastic_sc=[0.001])
        gw.add_node_subsidence(ns)
        assert gw.n_node_subsidence == 1

    def test_add_sub_irrigation(self) -> None:
        from pyiwfm.components.groundwater import AppGW, SubIrrigation

        gw = AppGW(n_nodes=10, n_layers=2, n_elements=5)
        si = SubIrrigation(id=1, gw_node=5, elevation=100.0, conductance=0.5)
        gw.add_sub_irrigation(si)
        assert gw.n_sub_irrigations == 1

    def test_validate_well_element_zero(self) -> None:
        """Wells with element=0 (not yet assigned) should not cause validation error."""
        from pyiwfm.components.groundwater import AppGW, Well

        gw = AppGW(n_nodes=10, n_layers=2, n_elements=5)
        gw.add_well(Well(id=1, x=0.0, y=0.0, element=0))
        gw.validate()  # Should not raise


# ============================================================================
# Phase 1: Reader bug fix tests
# ============================================================================


class TestTileDrainVersionHeader:
    """Test TileDrainReader handles #4.0 version header."""

    def _write_file(self, tmp_path: Path, content: str) -> Path:
        filepath = tmp_path / "tiledrain.dat"
        filepath.write_text(content)
        return filepath

    def test_with_version_header(self, tmp_path: Path) -> None:
        from pyiwfm.io.gw_tiledrain import TileDrainReader

        content = (
            "C Tile Drain File\n"
            "#4.0\n"
            "C\n"
            "    2                              / NDrain\n"
            "    1.0                            / FACTHD\n"
            "    1.0                            / FACTCDC\n"
            "    1DAY                           / TUNITDR\n"
            "    1    10    100.0    0.5    2    5\n"
            "    2    20    110.0    0.6    1    0\n"
            "    0                              / NSubIrig\n"
        )
        filepath = self._write_file(tmp_path, content)
        config = TileDrainReader().read(filepath)
        assert config.version == "4.0"
        assert config.n_drains == 2
        assert len(config.tile_drains) == 2
        assert config.tile_drains[0].gw_node == 10
        assert config.tile_drains[1].gw_node == 20

    def test_without_version_header(self, tmp_path: Path) -> None:
        """Backward compat: files without version header still parse."""
        from pyiwfm.io.gw_tiledrain import TileDrainReader

        content = (
            "C Tile Drain File (no version)\n"
            "    1                              / NDrain\n"
            "    1.0                            / FACTHD\n"
            "    1.0                            / FACTCDC\n"
            "    1DAY                           / TUNITDR\n"
            "    1    10    100.0    0.5    1    0\n"
            "    0                              / NSubIrig\n"
        )
        filepath = self._write_file(tmp_path, content)
        config = TileDrainReader().read(filepath)
        assert config.version == ""
        assert config.n_drains == 1
        assert len(config.tile_drains) == 1

    def test_with_sub_irrigation(self, tmp_path: Path) -> None:
        from pyiwfm.io.gw_tiledrain import TileDrainReader

        content = (
            "#4.0\n"
            "    0                              / NDrain\n"
            "    2                              / NSubIrig\n"
            "    1.0                            / FACTHSI\n"
            "    1.0                            / FACTCSI\n"
            "    1MON                           / TUNITSI\n"
            "    1    10    50.0    0.3\n"
            "    2    20    55.0    0.4\n"
        )
        filepath = self._write_file(tmp_path, content)
        config = TileDrainReader().read(filepath)
        assert config.version == "4.0"
        assert config.n_drains == 0
        assert config.n_sub_irrigation == 2
        assert len(config.sub_irrigations) == 2
        assert config.sub_irrigations[0].gw_node == 10
        assert config.sub_irrigations[1].conductance == 0.4


class TestSubsidenceNOUTSSection:
    """Test SubsidenceReader handles NOUTS hydrograph section."""

    def _write_file(self, tmp_path: Path, content: str) -> Path:
        filepath = tmp_path / "subsidence.dat"
        filepath.write_text(content)
        return filepath

    def test_with_nouts_section(self, tmp_path: Path) -> None:
        from pyiwfm.io.gw_subsidence import SubsidenceReader

        content = (
            "#4.0\n"
            "C IC file\n"
            "                                    / IC_FILE (empty)\n"
            "                                    / TECPLOT_FILE (empty)\n"
            "                                    / FINAL_SUBS_FILE (empty)\n"
            "    1.0                             / Output factor\n"
            "    FEET                            / Output unit\n"
            "    2                               / NOUTS\n"
            "    3.2808                          / FACTXY\n"
            "    SubsHyd.out                     / SUBHYDOUTFL\n"
            "    1  0  1  100.0  200.0  / InSAR_pt1\n"
            "    2  0  2  150.0  250.0  / Extensometer_1\n"
            "    0                               / NGROUP\n"
            "    1.0  1.0  1.0  1.0  1.0  1.0   / Conversion factors\n"
        )
        filepath = self._write_file(tmp_path, content)
        config = SubsidenceReader().read(filepath, n_nodes=0, n_layers=2)
        assert config.version == "4.0"
        assert config.n_hydrograph_outputs == 2
        assert len(config.hydrograph_specs) == 2
        assert config.hydrograph_specs[0].name == "InSAR_pt1"
        assert config.hydrograph_specs[0].x == pytest.approx(100.0 * 3.2808)
        assert config.hydrograph_specs[1].layer == 2
        assert config.n_parametric_grids == 0

    def test_nouts_zero_skips_factxy(self, tmp_path: Path) -> None:
        """When NOUTS=0, Fortran skips FACTXY and SUBHYDOUTFL lines."""
        from pyiwfm.io.gw_subsidence import SubsidenceReader

        content = (
            "#4.0\n"
            "                                    / IC_FILE\n"
            "                                    / TECPLOT_FILE\n"
            "                                    / FINAL_SUBS_FILE\n"
            "    1.0                             / Output factor\n"
            "    FEET                            / Output unit\n"
            "    0                               / NOUTS\n"
            "    0                               / NGROUP\n"
            "    1.0  1.0  1.0  1.0  1.0  1.0   / Conversion factors\n"
        )
        filepath = self._write_file(tmp_path, content)
        config = SubsidenceReader().read(filepath, n_nodes=0, n_layers=2)
        assert config.n_hydrograph_outputs == 0
        assert config.n_parametric_grids == 0
        assert len(config.conversion_factors) == 6

    def test_nouts_with_node_params(self, tmp_path: Path) -> None:
        """Verify node params still parse correctly after NOUTS section."""
        from pyiwfm.io.gw_subsidence import SubsidenceReader

        content = (
            "#4.0\n"
            "                                    / IC_FILE\n"
            "                                    / TECPLOT_FILE\n"
            "                                    / FINAL_SUBS_FILE\n"
            "    1.0                             / Output factor\n"
            "    FEET                            / Output unit\n"
            "    1                               / NOUTS\n"
            "    1.0                             / FACTXY\n"
            "    SubsHyd.out                     / SUBHYDOUTFL\n"
            "    1  0  1  100.0  200.0  / Obs1\n"
            "    0                               / NGROUP\n"
            "    1.0  1.0  1.0  1.0  1.0  1.0   / Conversion factors\n"
            "    1  0.001  0.01  10.0  2.0  50.0\n"
            "       0.002  0.02  8.0   1.5  45.0\n"
        )
        filepath = self._write_file(tmp_path, content)
        config = SubsidenceReader().read(filepath, n_nodes=1, n_layers=2)
        assert config.n_hydrograph_outputs == 1
        assert len(config.node_params) == 1
        assert config.node_params[0].node_id == 1
        assert config.node_params[0].elastic_sc == [0.001, 0.002]
        assert config.node_params[0].interbed_thick == [10.0, 8.0]


class TestConstrainedGHBCReading:
    """Test that constrained GH BCs are read correctly."""

    def test_read_constrained_gh_bc(self, tmp_path: Path) -> None:
        from pyiwfm.io.gw_boundary import GWBoundaryReader

        # Create the constrained GH sub-file
        cgh_content = (
            "    2                               / NCGB\n"
            "    1.0                             / FACTH\n"
            "    1.0                             / FACTVL\n"
            "    1DAY                            / TUNIT\n"
            "    1.0                             / FACTC\n"
            "    1DAY                            / TUNITC\n"
            "    100  1  3  50.0  0.5  30.0  5  100.0\n"
            "    200  2  4  60.0  0.6  35.0  6  120.0\n"
        )
        cgh_file = tmp_path / "cgh.dat"
        cgh_file.write_text(cgh_content)

        # Create main BC file referencing the CGH sub-file
        main_content = (
            f"                                   / Spec flow file\n"
            f"                                   / Spec head file\n"
            f"                                   / Gen head file\n"
            f"     {'cgh.dat':<30s}  / Constrained GH file\n"
            f"                                   / TS data file\n"
        )
        main_file = tmp_path / "bc_main.dat"
        main_file.write_text(main_content)

        reader = GWBoundaryReader()
        config = reader.read(main_file)
        assert config.n_constrained_gh == 2
        assert config.constrained_gh_bcs[0].node_id == 100
        assert config.constrained_gh_bcs[0].constraining_head == 30.0
        assert config.constrained_gh_bcs[0].max_flow == 100.0
        assert config.constrained_gh_bcs[1].external_head == 60.0


# ============================================================================
# Phase 2: Data accessibility tests
# ============================================================================


class TestWellNameParsing:
    """Test that well names are parsed from / delimiter."""

    def test_well_name_from_spec(self, tmp_path: Path) -> None:
        from pyiwfm.io.gw_pumping import PumpingReader

        well_content = (
            "    2                               / NWELL\n"
            "    1.0                             / FACTXY\n"
            "    1.0                             / FACTR\n"
            "    1.0                             / FACTLT\n"
            "    1  100.0  200.0  1.0  50.0  10.0  / Well_Alpha\n"
            "    2  300.0  400.0  1.0  60.0  20.0  / Well_Beta\n"
            "    1  1  1.0  0  -1  0  0  0  0  0.0\n"
            "    2  2  1.0  0  -1  0  0  0  0  0.0\n"
            "    0                               / NGROUPS\n"
        )
        well_file = tmp_path / "wells.dat"
        well_file.write_text(well_content)

        pump_content = (
            "#4.0\n"
            f"     {'wells.dat':<30s}  / Well spec file\n"
            "                                   / Elem pump file\n"
            "                                   / TS data file\n"
            "                                   / Output file\n"
        )
        pump_file = tmp_path / "pumping.dat"
        pump_file.write_text(pump_content)

        reader = PumpingReader()
        config = reader.read(pump_file, n_layers=1)
        assert config.well_specs[0].name == "Well_Alpha"
        assert config.well_specs[1].name == "Well_Beta"

    def test_well_without_name(self, tmp_path: Path) -> None:
        from pyiwfm.io.gw_pumping import PumpingReader

        well_content = (
            "    1                               / NWELL\n"
            "    1.0                             / FACTXY\n"
            "    1.0                             / FACTR\n"
            "    1.0                             / FACTLT\n"
            "    1  100.0  200.0  1.0  50.0  10.0\n"
            "    1  1  1.0  0  -1  0  0  0  0  0.0\n"
            "    0                               / NGROUPS\n"
        )
        well_file = tmp_path / "wells.dat"
        well_file.write_text(well_content)

        pump_content = (
            "#4.0\n"
            f"     {'wells.dat':<30s}  / Well spec file\n"
            "                                   / Elem pump file\n"
            "                                   / TS data file\n"
            "                                   / Output file\n"
        )
        pump_file = tmp_path / "pumping.dat"
        pump_file.write_text(pump_content)

        reader = PumpingReader()
        config = reader.read(pump_file, n_layers=1)
        assert config.well_specs[0].name == ""


# ============================================================================
# Phase 3: Writer roundtrip tests
# ============================================================================


class TestTileDrainWriterRoundtrip:
    """Test tile drain writer → reader roundtrip."""

    def test_roundtrip(self, tmp_path: Path) -> None:
        from pyiwfm.io.gw_tiledrain import (
            SubIrrigationSpec,
            TileDrainConfig,
            TileDrainReader,
            TileDrainSpec,
        )
        from pyiwfm.io.gw_tiledrain_writer import write_tile_drain_file

        config = TileDrainConfig(
            version="4.0",
            n_drains=2,
            drain_height_factor=1.0,
            drain_conductance_factor=1.0,
            drain_time_unit="1DAY",
            tile_drains=[
                TileDrainSpec(
                    id=1, gw_node=10, elevation=100.0, conductance=0.5, dest_type=2, dest_id=5
                ),
                TileDrainSpec(
                    id=2, gw_node=20, elevation=110.0, conductance=0.6, dest_type=1, dest_id=0
                ),
            ],
            n_sub_irrigation=1,
            subirig_height_factor=1.0,
            subirig_conductance_factor=1.0,
            subirig_time_unit="1MON",
            sub_irrigations=[
                SubIrrigationSpec(id=1, gw_node=30, elevation=50.0, conductance=0.3),
            ],
        )

        filepath = tmp_path / "tiledrain_out.dat"
        write_tile_drain_file(config, filepath)

        # Re-read
        reader = TileDrainReader()
        config2 = reader.read(filepath)
        assert config2.version == "4.0"
        assert config2.n_drains == 2
        assert config2.n_sub_irrigation == 1
        assert config2.tile_drains[0].gw_node == 10
        assert config2.sub_irrigations[0].gw_node == 30


class TestBoundaryWriterRoundtrip:
    """Test BC writer → reader roundtrip for constrained GH."""

    def test_constrained_gh_roundtrip(self, tmp_path: Path) -> None:
        from pyiwfm.io.gw_boundary import (
            ConstrainedGeneralHeadBC,
            GWBoundaryConfig,
            GWBoundaryReader,
        )
        from pyiwfm.io.gw_boundary_writer import (
            write_constrained_gh_bc,
        )

        cgh_file = tmp_path / "cgh.dat"
        config = GWBoundaryConfig(
            cgh_file=cgh_file,
            constrained_gh_bcs=[
                ConstrainedGeneralHeadBC(
                    node_id=100,
                    layer=1,
                    ts_column=3,
                    external_head=50.0,
                    conductance=0.5,
                    constraining_head=30.0,
                    max_flow_ts_column=5,
                    max_flow=100.0,
                ),
            ],
            cgh_head_factor=1.0,
            cgh_max_flow_factor=1.0,
            cgh_head_time_unit="1DAY",
            cgh_conductance_factor=1.0,
            cgh_conductance_time_unit="1DAY",
        )

        write_constrained_gh_bc(config, cgh_file)

        # Re-read directly
        reader = GWBoundaryReader()
        reader._line_num = 0
        config2 = GWBoundaryConfig()
        reader._read_constrained_gh(cgh_file, config2)
        assert config2.n_constrained_gh == 1
        assert config2.constrained_gh_bcs[0].constraining_head == pytest.approx(30.0)
        assert config2.constrained_gh_bcs[0].max_flow == pytest.approx(100.0)


class TestPumpingWriterRoundtrip:
    """Test pumping writer → reader roundtrip."""

    def test_well_spec_roundtrip(self, tmp_path: Path) -> None:
        from pyiwfm.io.gw_pumping import (
            PumpingConfig,
            PumpingReader,
            WellPumpingSpec,
            WellSpec,
        )
        from pyiwfm.io.gw_pumping_writer import write_well_spec_file

        config = PumpingConfig(
            factor_xy=1.0,
            factor_radius=1.0,
            factor_length=1.0,
            well_specs=[
                WellSpec(
                    id=1,
                    x=100.0,
                    y=200.0,
                    radius=0.5,
                    perf_top=50.0,
                    perf_bottom=10.0,
                    name="TestWell",
                ),
            ],
            well_pumping_specs=[
                WellPumpingSpec(well_id=1, pump_column=3, pump_fraction=0.8, dist_method=2),
            ],
            well_groups=[],
        )

        filepath = tmp_path / "wells.dat"
        write_well_spec_file(config, filepath)

        # Read back using raw parsing
        reader = PumpingReader()
        reader._line_num = 0
        config2 = PumpingConfig(factor_xy=1.0, factor_radius=1.0, factor_length=1.0)
        reader._read_well_file(filepath, config2)
        assert config2.n_wells == 1
        assert config2.well_specs[0].name == "TestWell"
        assert config2.well_specs[0].x == pytest.approx(100.0)
        assert config2.well_pumping_specs[0].pump_column == 3


# ============================================================================
# Phase 1B: Subsidence hydrograph spec dataclass
# ============================================================================


class TestSubsidenceHydrographSpec:
    """Test the new SubsidenceHydrographSpec dataclass."""

    def test_create(self) -> None:
        from pyiwfm.io.gw_subsidence import SubsidenceHydrographSpec

        spec = SubsidenceHydrographSpec(id=1, hydtyp=0, layer=2, x=100.0, y=200.0, name="InSAR_1")
        assert spec.id == 1
        assert spec.name == "InSAR_1"
        assert spec.layer == 2


# ============================================================================
# Phase 2F: BC NOUTB section
# ============================================================================


class TestBCNOUTBSection:
    """Test that NOUTB section is read from BC main file."""

    def test_noutb_zero(self, tmp_path: Path) -> None:
        from pyiwfm.io.gw_boundary import GWBoundaryReader

        content = (
            "                                   / Spec flow file\n"
            "                                   / Spec head file\n"
            "                                   / Gen head file\n"
            "                                   / CGH file\n"
            "                                   / TS data file\n"
            "    0                               / NOUTB\n"
        )
        filepath = tmp_path / "bc.dat"
        filepath.write_text(content)
        config = GWBoundaryReader().read(filepath)
        assert config.n_bc_output_nodes == 0

    def test_noutb_with_nodes(self, tmp_path: Path) -> None:
        from pyiwfm.io.gw_boundary import GWBoundaryReader

        content = (
            "                                   / Spec flow file\n"
            "                                   / Spec head file\n"
            "                                   / Gen head file\n"
            "                                   / CGH file\n"
            "                                   / TS data file\n"
            "    2                               / NOUTB\n"
            "    BCFlow.out                      / BHYDOUTFL\n"
            "    100\n"
            "    200\n"
        )
        filepath = tmp_path / "bc.dat"
        filepath.write_text(content)
        config = GWBoundaryReader().read(filepath)
        assert config.n_bc_output_nodes == 2
        assert config.bc_output_specs == [{"id": 100}, {"id": 200}]
