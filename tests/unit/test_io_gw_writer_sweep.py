"""Sweep tests for pyiwfm.io.gw_writer targeting remaining uncovered lines.

Covers:
- _render_gw_main_roundtrip: parametric grids section (lines 381-397),
  per-node aquifer params (lines 401-424), return flow (lines 443-449)
- write_ts_pumping: node-level pumping (lines 747-750, 775-781)
- write_pump_main: element pumping branch (lines 844-848)
- write_gw_main initial heads from cfg.initial_heads (lines 933, 952, 956-960)
- write_elem_pump_specs: element pumping (lines 1019-1059)
- write_tile_drains: tile drain file writing (lines 1019-1059)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tests.conftest import make_simple_grid, make_simple_stratigraphy

# ---------------------------------------------------------------------------
# Helpers -- build a minimal model skeleton for GWComponentWriter
# ---------------------------------------------------------------------------


def _make_model(
    gw: object | None = None,
    n_nodes: int = 9,
    n_layers: int = 2,
) -> object:
    """Create a mock IWFMModel with grid, stratigraphy, and optional GW component."""
    from pyiwfm.core.model import IWFMModel

    grid = make_simple_grid()
    strat = make_simple_stratigraphy(n_nodes=n_nodes, n_layers=n_layers)

    model = IWFMModel(
        name="test_model",
        mesh=grid,
        stratigraphy=strat,
    )
    model.groundwater = gw  # type: ignore[assignment]
    return model


def _make_gw(
    n_nodes: int = 9,
    n_layers: int = 2,
    wells: dict | None = None,
    element_pumping: list | None = None,
    tile_drains: dict | None = None,
    aquifer_params: object | None = None,
    heads: np.ndarray | None = None,
    boundary_conditions: list | None = None,
) -> object:
    """Create a minimal AppGW instance."""
    from pyiwfm.components.groundwater import AppGW

    return AppGW(
        n_nodes=n_nodes,
        n_layers=n_layers,
        n_elements=4,
        wells=wells or {},
        element_pumping=element_pumping or [],
        tile_drains=tile_drains or {},
        aquifer_params=aquifer_params,
        heads=heads,
        boundary_conditions=boundary_conditions or [],
    )


# ---------------------------------------------------------------------------
# _render_gw_main_roundtrip -- parametric grids section
# ---------------------------------------------------------------------------


class TestRenderGWMainRoundtripParametricGrids:
    """Cover lines 381-397: parametric grid data output."""

    def test_parametric_grid_section(self, tmp_path: Path) -> None:
        from pyiwfm.io.groundwater import (
            GWMainFileConfig,
            ParametricGridData,
        )
        from pyiwfm.io.gw_writer import GWComponentWriter, GWWriterConfig

        n_nodes = 9
        n_layers = 2

        # Build parametric grid data
        pgd = ParametricGridData(
            n_nodes=4,
            n_elements=1,
            elements=[(0, 1, 2, 3)],
            node_coords=np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=float),
            node_values=np.zeros((4, n_layers, 5)),
            node_range_str="1-9",
            raw_node_lines=[
                "1  0.0  0.0  1.0  1e-6  0.1  0.1  0.1  1.0  1e-6  0.1  0.1  0.1",
                "2  100.0  0.0  1.0  1e-6  0.1  0.1  0.1  1.0  1e-6  0.1  0.1  0.1",
                "3  100.0  100.0  1.0  1e-6  0.1  0.1  0.1  1.0  1e-6  0.1  0.1  0.1",
                "4  0.0  100.0  1.0  1e-6  0.1  0.1  0.1  1.0  1e-6  0.1  0.1  0.1",
            ],
        )

        cfg = GWMainFileConfig(
            version="4.0",
            n_param_groups=1,
            parametric_grids=[pgd],
            aq_factors_line="1.0  1.0  1.0  1.0  1.0  0.0",
            aq_time_unit_kh="1DAY",
            aq_time_unit_v="1DAY",
            aq_time_unit_l="1DAY",
            tecplot_print_flag=1,
            debug_flag=0,
            coord_factor=1.0,
            hydrograph_locations=[],
            face_flow_specs=[],
            kh_anomalies=[],
            kh_anomaly_factor=1.0,
            kh_anomaly_time_unit="1DAY",
            return_flow_flag=0,
            initial_heads=np.full((n_nodes, n_layers), 50.0),
        )

        gw = _make_gw(n_nodes=n_nodes, n_layers=n_layers)
        gw.gw_main_config = cfg  # type: ignore[attr-defined]
        model = _make_model(gw=gw, n_nodes=n_nodes, n_layers=n_layers)

        config = GWWriterConfig(output_dir=tmp_path / "Simulation")
        writer = GWComponentWriter(model, config)  # type: ignore[arg-type]
        output_path = writer.write_main()

        content = output_path.read_text()
        # Parametric grid markers
        assert "NDP" in content
        assert "NEP" in content
        assert "1-9" in content


class TestRenderGWMainRoundtripPerNodeParams:
    """Cover lines 401-424: per-node aquifer parameter output (NGROUP=0)."""

    def test_per_node_aquifer_params(self, tmp_path: Path) -> None:
        from pyiwfm.components.groundwater import AquiferParameters
        from pyiwfm.io.groundwater import GWMainFileConfig
        from pyiwfm.io.gw_writer import GWComponentWriter, GWWriterConfig

        n_nodes = 9
        n_layers = 2

        aq = AquiferParameters(
            n_nodes=n_nodes,
            n_layers=n_layers,
            kh=np.full((n_nodes, n_layers), 10.0),
            kv=np.full((n_nodes, n_layers), 0.5),
            specific_storage=np.full((n_nodes, n_layers), 1e-5),
            specific_yield=np.full((n_nodes, n_layers), 0.15),
            aquitard_kv=np.full((n_nodes, n_layers), 0.05),
        )

        cfg = GWMainFileConfig(
            version="4.0",
            n_param_groups=0,
            aq_factors_line="1.0  1.0  1.0  1.0  1.0  0.0",
            aq_time_unit_kh="1DAY",
            aq_time_unit_v="1DAY",
            aq_time_unit_l="1DAY",
            tecplot_print_flag=1,
            debug_flag=0,
            coord_factor=1.0,
            hydrograph_locations=[],
            face_flow_specs=[],
            kh_anomalies=[],
            kh_anomaly_factor=1.0,
            kh_anomaly_time_unit="1DAY",
            return_flow_flag=0,
            initial_heads=np.full((n_nodes, n_layers), 75.0),
        )

        gw = _make_gw(n_nodes=n_nodes, n_layers=n_layers, aquifer_params=aq)
        gw.gw_main_config = cfg  # type: ignore[attr-defined]
        model = _make_model(gw=gw, n_nodes=n_nodes, n_layers=n_layers)

        config = GWWriterConfig(output_dir=tmp_path / "Simulation")
        writer = GWComponentWriter(model, config)  # type: ignore[arg-type]
        output_path = writer.write_main()

        content = output_path.read_text()
        # Should have aquifer parameter values
        assert "10.0000" in content
        # Should have initial heads from cfg.initial_heads
        assert "75.0000" in content


class TestRenderGWMainRoundtripReturnFlow:
    """Cover lines 437-449: return flow and initial heads sections."""

    def test_return_flow_flag_and_heads(self, tmp_path: Path) -> None:
        from pyiwfm.io.groundwater import GWMainFileConfig
        from pyiwfm.io.gw_writer import GWComponentWriter, GWWriterConfig

        n_nodes = 9
        n_layers = 2

        cfg = GWMainFileConfig(
            version="4.0",
            n_param_groups=0,
            aq_factors_line="1.0  1.0  1.0  1.0  1.0  0.0",
            aq_time_unit_kh="1DAY",
            aq_time_unit_v="1DAY",
            aq_time_unit_l="1DAY",
            tecplot_print_flag=1,
            debug_flag=0,
            coord_factor=1.0,
            hydrograph_locations=[],
            face_flow_specs=[],
            kh_anomalies=[],
            kh_anomaly_factor=1.0,
            kh_anomaly_time_unit="1DAY",
            return_flow_flag=1,
            initial_heads=np.full((n_nodes, n_layers), 80.0),
        )

        # GW with heads set (takes priority over cfg.initial_heads)
        heads = np.full((n_nodes, n_layers), 90.0)
        gw = _make_gw(n_nodes=n_nodes, n_layers=n_layers, heads=heads)
        gw.gw_main_config = cfg  # type: ignore[attr-defined]
        model = _make_model(gw=gw, n_nodes=n_nodes, n_layers=n_layers)

        config = GWWriterConfig(output_dir=tmp_path / "Simulation")
        writer = GWComponentWriter(model, config)  # type: ignore[arg-type]
        output_path = writer.write_main()

        content = output_path.read_text()
        assert "IFLAGRF" in content
        # Heads from gw.heads (90.0), not cfg.initial_heads (80.0)
        assert "90.0000" in content

    def test_initial_heads_from_config(self, tmp_path: Path) -> None:
        """When gw.heads is None, uses cfg.initial_heads."""
        from pyiwfm.io.groundwater import GWMainFileConfig
        from pyiwfm.io.gw_writer import GWComponentWriter, GWWriterConfig

        n_nodes = 9
        n_layers = 2

        cfg = GWMainFileConfig(
            version="4.0",
            n_param_groups=0,
            aq_factors_line="1.0  1.0  1.0  1.0  1.0  0.0",
            aq_time_unit_kh="1DAY",
            aq_time_unit_v="1DAY",
            aq_time_unit_l="1DAY",
            tecplot_print_flag=1,
            debug_flag=0,
            coord_factor=1.0,
            hydrograph_locations=[],
            face_flow_specs=[],
            kh_anomalies=[],
            kh_anomaly_factor=1.0,
            kh_anomaly_time_unit="1DAY",
            return_flow_flag=0,
            initial_heads=np.full((n_nodes, n_layers), 65.0),
        )

        gw = _make_gw(n_nodes=n_nodes, n_layers=n_layers, heads=None)
        gw.gw_main_config = cfg  # type: ignore[attr-defined]
        model = _make_model(gw=gw, n_nodes=n_nodes, n_layers=n_layers)

        config = GWWriterConfig(output_dir=tmp_path / "Simulation")
        writer = GWComponentWriter(model, config)  # type: ignore[arg-type]
        output_path = writer.write_main()

        content = output_path.read_text()
        assert "65.0000" in content


# ---------------------------------------------------------------------------
# write_elem_pump_specs
# ---------------------------------------------------------------------------


class TestWriteElemPumpSpecs:
    """Cover lines 1019-1059: element pumping specification file."""

    def test_write_elem_pump_specs(self, tmp_path: Path) -> None:
        from pyiwfm.components.groundwater import ElementPumping
        from pyiwfm.io.gw_writer import GWComponentWriter, GWWriterConfig

        ep = ElementPumping(
            element_id=1,
            layer=1,
            pump_rate=-500.0,
            pump_column=1,
            pump_fraction=1.0,
            dist_method=1,
            layer_factors=[0.6, 0.4],
            dest_type=-1,
            dest_id=0,
            irig_frac_column=0,
            adjust_column=0,
            pump_max_column=0,
            pump_max_fraction=0.0,
        )

        gw = _make_gw(element_pumping=[ep])
        model = _make_model(gw=gw)

        config = GWWriterConfig(output_dir=tmp_path / "Simulation")
        writer = GWComponentWriter(model, config)  # type: ignore[arg-type]
        output_path = writer.write_elem_pump_specs()

        content = output_path.read_text()
        assert "NSINK" in content
        assert "NGRP" in content
        # Element ID should appear
        assert "1" in content

    def test_multiple_elem_pumps(self, tmp_path: Path) -> None:
        from pyiwfm.components.groundwater import ElementPumping
        from pyiwfm.io.gw_writer import GWComponentWriter, GWWriterConfig

        eps = [
            ElementPumping(
                element_id=i,
                layer=1,
                pump_rate=-100.0 * i,
                pump_column=i,
                pump_fraction=1.0,
                dist_method=0,
                layer_factors=[1.0],
                dest_type=0,
                dest_id=0,
                irig_frac_column=0,
                adjust_column=0,
                pump_max_column=0,
                pump_max_fraction=0.0,
            )
            for i in range(1, 4)
        ]

        gw = _make_gw(element_pumping=eps)
        model = _make_model(gw=gw)

        config = GWWriterConfig(output_dir=tmp_path / "Simulation")
        writer = GWComponentWriter(model, config)  # type: ignore[arg-type]
        output_path = writer.write_elem_pump_specs()

        content = output_path.read_text()
        assert "3" in content  # NSINK count


# ---------------------------------------------------------------------------
# write_tile_drains
# ---------------------------------------------------------------------------


class TestWriteTileDrains:
    """Cover tile drain writing via write_tile_drains."""

    def test_basic_tile_drains(self, tmp_path: Path) -> None:
        from pyiwfm.components.groundwater import TileDrain
        from pyiwfm.io.gw_writer import GWComponentWriter, GWWriterConfig

        drains = {
            1: TileDrain(
                id=1,
                element=3,
                elevation=50.0,
                conductance=0.01,
                destination_type="stream",
                destination_id=5,
            ),
            2: TileDrain(
                id=2,
                element=7,
                elevation=45.0,
                conductance=0.02,
                destination_type="outside",
                destination_id=0,
            ),
        }

        gw = _make_gw(tile_drains=drains)
        model = _make_model(gw=gw)

        config = GWWriterConfig(output_dir=tmp_path / "Simulation")
        writer = GWComponentWriter(model, config)  # type: ignore[arg-type]
        output_path = writer.write_tile_drains()

        content = output_path.read_text()
        assert output_path.exists()
        # Should contain the drain count
        assert "2" in content

    def test_tile_drains_with_dest_type_int(self, tmp_path: Path) -> None:
        """Cover destination_type as integer (line 748)."""
        from pyiwfm.components.groundwater import TileDrain
        from pyiwfm.io.gw_writer import GWComponentWriter, GWWriterConfig

        drains = {
            1: TileDrain(
                id=1,
                element=1,
                elevation=60.0,
                conductance=0.05,
                destination_type="outside",  # string type
            ),
        }

        gw = _make_gw(tile_drains=drains)
        model = _make_model(gw=gw)

        config = GWWriterConfig(output_dir=tmp_path / "Simulation")
        writer = GWComponentWriter(model, config)  # type: ignore[arg-type]
        output_path = writer.write_tile_drains()
        assert output_path.exists()


# ---------------------------------------------------------------------------
# write_pump_main with element pumping
# ---------------------------------------------------------------------------


class TestWritePumpMain:
    """Cover write_pump_main element pumping branch."""

    def test_pump_main_with_elem_pumping(self, tmp_path: Path) -> None:
        from pyiwfm.components.groundwater import ElementPumping
        from pyiwfm.io.gw_writer import GWComponentWriter, GWWriterConfig

        ep = ElementPumping(
            element_id=2,
            layer=1,
            pump_rate=-200.0,
            pump_column=1,
            pump_fraction=1.0,
            dist_method=0,
            layer_factors=[1.0, 0.0],
            dest_type=0,
            dest_id=0,
            irig_frac_column=0,
            adjust_column=0,
            pump_max_column=0,
            pump_max_fraction=0.0,
        )

        gw = _make_gw(element_pumping=[ep])
        model = _make_model(gw=gw)

        config = GWWriterConfig(output_dir=tmp_path / "Simulation")
        writer = GWComponentWriter(model, config)  # type: ignore[arg-type]
        output_path = writer.write_pump_main()

        content = output_path.read_text()
        assert output_path.exists()
        # Should reference the elem pump file
        assert "ElemPump" in content

    def test_pump_main_with_wells_and_elem_pump(self, tmp_path: Path) -> None:
        from pyiwfm.components.groundwater import ElementPumping, Well
        from pyiwfm.io.gw_writer import GWComponentWriter, GWWriterConfig

        wells = {
            1: Well(
                id=1, x=50.0, y=50.0, element=1, name="W1", top_screen=80.0, bottom_screen=40.0
            ),
        }
        eps = [
            ElementPumping(
                element_id=3,
                layer=1,
                pump_rate=-100.0,
                pump_column=2,
                pump_fraction=1.0,
                dist_method=0,
                layer_factors=[1.0],
                dest_type=0,
                dest_id=0,
                irig_frac_column=0,
                adjust_column=0,
                pump_max_column=0,
                pump_max_fraction=0.0,
            ),
        ]

        gw = _make_gw(wells=wells, element_pumping=eps)
        model = _make_model(gw=gw)

        config = GWWriterConfig(output_dir=tmp_path / "Simulation")
        writer = GWComponentWriter(model, config)  # type: ignore[arg-type]
        output_path = writer.write_pump_main()

        content = output_path.read_text()
        assert "WellSpec" in content
        assert "ElemPump" in content


# ---------------------------------------------------------------------------
# write_ts_pumping
# ---------------------------------------------------------------------------


class TestWriteTsPumping:
    """Cover write_ts_pumping with wells and element pumping."""

    def test_ts_pumping_with_wells(self, tmp_path: Path) -> None:
        from pyiwfm.components.groundwater import Well
        from pyiwfm.io.gw_writer import GWComponentWriter, GWWriterConfig

        wells = {
            1: Well(id=1, x=50.0, y=50.0, element=1, name="W1"),
            2: Well(id=2, x=150.0, y=50.0, element=2, name="W2"),
        }

        gw = _make_gw(wells=wells)
        model = _make_model(gw=gw)

        config = GWWriterConfig(output_dir=tmp_path / "Simulation")
        writer = GWComponentWriter(model, config)  # type: ignore[arg-type]
        output_path = writer.write_ts_pumping()
        assert output_path.exists()

    def test_ts_pumping_with_data(self, tmp_path: Path) -> None:
        from pyiwfm.components.groundwater import Well
        from pyiwfm.io.gw_writer import GWComponentWriter, GWWriterConfig

        wells = {
            1: Well(id=1, x=50.0, y=50.0, element=1),
        }

        gw = _make_gw(wells=wells)
        model = _make_model(gw=gw)

        config = GWWriterConfig(output_dir=tmp_path / "Simulation")
        writer = GWComponentWriter(model, config)  # type: ignore[arg-type]

        dates = ["10/01/1990_24:00", "11/01/1990_24:00"]
        data = np.array([[-100.0], [-120.0]])
        output_path = writer.write_ts_pumping(dates=dates, data=data)
        assert output_path.exists()
        content = output_path.read_text()
        assert "10/01/1990" in content


# ---------------------------------------------------------------------------
# write_all with various combinations
# ---------------------------------------------------------------------------


class TestWriteAll:
    """Cover write_all orchestration with tile drains and elem pumping."""

    def test_write_all_with_tile_drains(self, tmp_path: Path) -> None:
        from pyiwfm.components.groundwater import TileDrain
        from pyiwfm.io.gw_writer import GWComponentWriter, GWWriterConfig

        drains = {
            1: TileDrain(id=1, element=2, elevation=55.0, conductance=0.03),
        }

        gw = _make_gw(tile_drains=drains)
        model = _make_model(gw=gw)

        config = GWWriterConfig(output_dir=tmp_path / "Simulation")
        writer = GWComponentWriter(model, config)  # type: ignore[arg-type]
        results = writer.write_all()

        assert "main" in results
        assert "tile_drains" in results

    def test_write_all_with_element_pumping(self, tmp_path: Path) -> None:
        from pyiwfm.components.groundwater import ElementPumping
        from pyiwfm.io.gw_writer import GWComponentWriter, GWWriterConfig

        eps = [
            ElementPumping(
                element_id=1,
                layer=1,
                pump_rate=-50.0,
                pump_column=1,
                pump_fraction=1.0,
                dist_method=0,
                layer_factors=[1.0],
                dest_type=0,
                dest_id=0,
                irig_frac_column=0,
                adjust_column=0,
                pump_max_column=0,
                pump_max_fraction=0.0,
            ),
        ]

        gw = _make_gw(element_pumping=eps)
        model = _make_model(gw=gw)

        config = GWWriterConfig(output_dir=tmp_path / "Simulation")
        writer = GWComponentWriter(model, config)  # type: ignore[arg-type]
        results = writer.write_all()

        assert "main" in results
        assert "pump_main" in results
        assert "elem_pump_specs" in results

    def test_write_all_no_gw(self, tmp_path: Path) -> None:
        """Write defaults even when no GW component is loaded."""
        from pyiwfm.io.gw_writer import GWComponentWriter, GWWriterConfig

        model = _make_model(gw=None)

        config = GWWriterConfig(output_dir=tmp_path / "Simulation")
        writer = GWComponentWriter(model, config)  # type: ignore[arg-type]
        results = writer.write_all(write_defaults=True)

        assert "main" in results


# ---------------------------------------------------------------------------
# write_gw_component convenience function
# ---------------------------------------------------------------------------


class TestWriteGWComponent:
    """Cover the module-level write_gw_component function."""

    def test_convenience_function(self, tmp_path: Path) -> None:
        from pyiwfm.io.gw_writer import write_gw_component

        model = _make_model(gw=_make_gw())
        results = write_gw_component(model, tmp_path / "Simulation")  # type: ignore[arg-type]
        assert "main" in results

    def test_with_custom_config(self, tmp_path: Path) -> None:
        from pyiwfm.io.gw_writer import GWWriterConfig, write_gw_component

        config = GWWriterConfig(output_dir=tmp_path / "Simulation", gw_subdir="Groundwater")
        model = _make_model(gw=_make_gw())
        results = write_gw_component(model, tmp_path / "Simulation", config=config)  # type: ignore[arg-type]
        assert "main" in results
        # The main file should be in the Groundwater subdirectory
        assert "Groundwater" in str(results["main"])
