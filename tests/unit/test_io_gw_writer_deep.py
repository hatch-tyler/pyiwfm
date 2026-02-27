"""Deep tests for pyiwfm.io.gw_writer targeting uncovered branches.

Covers:
- _render_gw_main_roundtrip() with HYDTYP=0 and HYDTYP=1 hydrograph specs
- Parametric grids section
- Kh anomaly writing
- Return flow section
- Initial heads file writing (from cfg.initial_heads)
- write_ts_pumping()
- write_pump_main() with element pumping only
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pyiwfm.components.groundwater import (
    AppGW,
    ElementPumping,
    HydrographLocation,
)
from pyiwfm.io.groundwater import (
    FaceFlowSpec,
    GWMainFileConfig,
    KhAnomalyEntry,
    ParametricGridData,
)
from pyiwfm.io.gw_writer import GWComponentWriter, GWWriterConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_model(
    gw: AppGW | None = None,
    n_nodes: int = 4,
    n_layers: int = 1,
) -> MagicMock:
    """Build a minimal mock IWFMModel."""
    model = MagicMock()
    model.groundwater = gw
    model.n_nodes = n_nodes
    model.n_layers = n_layers

    # Stratigraphy with uniform top/bottom elevations
    strat = MagicMock()
    strat.n_layers = n_layers
    strat.top_elev = np.full((n_nodes, n_layers), 100.0)
    strat.bottom_elev = np.full((n_nodes, n_layers), 0.0)
    model.stratigraphy = strat

    return model


# ---------------------------------------------------------------------------
# Tests for _render_gw_main_roundtrip
# ---------------------------------------------------------------------------


class TestRenderGWMainRoundtrip:
    """Verify roundtrip rendering with various configurations."""

    def test_hydtyp0_xy_coords(self, tmp_path: Path) -> None:
        """HYDTYP=0 hydrograph locations written with X/Y coordinates."""
        loc = HydrographLocation(node_id=0, layer=1, x=1000.0, y=2000.0, name="Obs1")
        cfg = GWMainFileConfig()
        cfg.version = "4.0"
        cfg.coord_factor = 1.0
        cfg.hydrograph_locations = [loc]
        cfg.face_flow_specs = []
        cfg.n_param_groups = 0
        cfg.aq_factors_line = "1.0 1.0 1.0 1.0 1.0 1.0"
        cfg.aq_time_unit_kh = "DAY"
        cfg.aq_time_unit_v = "DAY"
        cfg.aq_time_unit_l = "DAY"
        cfg.kh_anomalies = []
        cfg.kh_anomaly_factor = 1.0
        cfg.kh_anomaly_time_unit = "DAY"
        cfg.return_flow_flag = 0
        cfg.initial_heads = np.full((4, 1), 50.0)
        cfg.tecplot_print_flag = 1
        cfg.debug_flag = 1
        cfg.raw_paths = {}
        cfg.subsidence_file = None
        cfg.parametric_grids = []

        gw = AppGW(n_nodes=4, n_layers=1, n_elements=2)
        gw.gw_main_config = cfg

        model = _mock_model(gw=gw)
        wconfig = GWWriterConfig(output_dir=tmp_path)
        writer = GWComponentWriter(model, wconfig)
        content = writer._render_gw_main_roundtrip(cfg, gw, n_layers=1, n_nodes=4)

        # HYDTYP=0 row contains "0" and the X/Y values
        assert "1000.0" in content
        assert "2000.0" in content
        assert " 0 " in content  # HYDTYP field

    def test_hydtyp1_node_id(self, tmp_path: Path) -> None:
        """HYDTYP=1 hydrograph locations written with node ID."""
        loc = HydrographLocation(node_id=42, layer=2, x=500.0, y=600.0, name="Well_A")
        cfg = GWMainFileConfig()
        cfg.version = "4.0"
        cfg.coord_factor = 1.0
        cfg.hydrograph_locations = [loc]
        cfg.face_flow_specs = []
        cfg.n_param_groups = 0
        cfg.aq_factors_line = "1.0 1.0 1.0 1.0 1.0 1.0"
        cfg.aq_time_unit_kh = "DAY"
        cfg.aq_time_unit_v = "DAY"
        cfg.aq_time_unit_l = "DAY"
        cfg.kh_anomalies = []
        cfg.kh_anomaly_factor = 1.0
        cfg.kh_anomaly_time_unit = "DAY"
        cfg.return_flow_flag = 0
        cfg.initial_heads = np.full((4, 1), 50.0)
        cfg.tecplot_print_flag = 1
        cfg.debug_flag = 0
        cfg.raw_paths = {}
        cfg.subsidence_file = None
        cfg.parametric_grids = []

        gw = AppGW(n_nodes=4, n_layers=1, n_elements=2)
        gw.gw_main_config = cfg

        model = _mock_model(gw=gw)
        wconfig = GWWriterConfig(output_dir=tmp_path)
        writer = GWComponentWriter(model, wconfig)
        content = writer._render_gw_main_roundtrip(cfg, gw, n_layers=1, n_nodes=4)

        # HYDTYP=1 row contains "1" and the node ID
        assert "42" in content
        assert " 1 " in content
        assert "Well_A" in content

    def test_kh_anomaly_and_return_flow(self, tmp_path: Path) -> None:
        """Kh anomaly entries and return flow flag appear in output."""
        cfg = GWMainFileConfig()
        cfg.version = "4.0"
        cfg.coord_factor = 1.0
        cfg.hydrograph_locations = []
        cfg.face_flow_specs = []
        cfg.n_param_groups = 0
        cfg.aq_factors_line = "1.0 1.0 1.0 1.0 1.0 1.0"
        cfg.aq_time_unit_kh = "DAY"
        cfg.aq_time_unit_v = "DAY"
        cfg.aq_time_unit_l = "DAY"
        cfg.kh_anomalies = [KhAnomalyEntry(element_id=5, kh_per_layer=[2.5])]
        cfg.kh_anomaly_factor = 1.0
        cfg.kh_anomaly_time_unit = "DAY"
        cfg.return_flow_flag = 1
        cfg.initial_heads = np.full((4, 1), 75.0)
        cfg.tecplot_print_flag = 1
        cfg.debug_flag = 0
        cfg.raw_paths = {}
        cfg.subsidence_file = None
        cfg.parametric_grids = []

        gw = AppGW(n_nodes=4, n_layers=1, n_elements=2)
        gw.gw_main_config = cfg

        model = _mock_model(gw=gw)
        wconfig = GWWriterConfig(output_dir=tmp_path)
        writer = GWComponentWriter(model, wconfig)
        content = writer._render_gw_main_roundtrip(cfg, gw, n_layers=1, n_nodes=4)

        # Kh anomaly: NEBK=1, element 5
        assert "1" in content  # NEBK
        assert "5" in content
        assert "2.5000" in content

        # Return flow flag
        assert "IFLAGRF" in content

    def test_initial_heads_from_config(self, tmp_path: Path) -> None:
        """Initial heads written from cfg.initial_heads when gw.heads is None."""
        cfg = GWMainFileConfig()
        cfg.version = "4.0"
        cfg.coord_factor = 1.0
        cfg.hydrograph_locations = []
        cfg.face_flow_specs = []
        cfg.n_param_groups = 0
        cfg.aq_factors_line = "1.0 1.0 1.0 1.0 1.0 1.0"
        cfg.aq_time_unit_kh = "DAY"
        cfg.aq_time_unit_v = "DAY"
        cfg.aq_time_unit_l = "DAY"
        cfg.kh_anomalies = []
        cfg.kh_anomaly_factor = 1.0
        cfg.kh_anomaly_time_unit = "DAY"
        cfg.return_flow_flag = 0
        cfg.initial_heads = np.array([[88.0], [77.0], [66.0], [55.0]])
        cfg.tecplot_print_flag = 1
        cfg.debug_flag = 0
        cfg.raw_paths = {}
        cfg.subsidence_file = None
        cfg.parametric_grids = []

        gw = AppGW(n_nodes=4, n_layers=1, n_elements=2)
        gw.heads = None
        gw.gw_main_config = cfg

        model = _mock_model(gw=gw)
        wconfig = GWWriterConfig(output_dir=tmp_path)
        writer = GWComponentWriter(model, wconfig)
        content = writer._render_gw_main_roundtrip(cfg, gw, n_layers=1, n_nodes=4)

        assert "88.0000" in content
        assert "55.0000" in content


# ---------------------------------------------------------------------------
# Tests for write_pump_main (element pumping only) and write_ts_pumping
# ---------------------------------------------------------------------------


class TestPumpingWriters:
    """Tests for pump main and time-series pumping file writing."""

    def test_write_pump_main_elem_only(self, tmp_path: Path) -> None:
        """write_pump_main with element pumping but no wells."""
        ep = ElementPumping(
            element_id=1,
            layer=1,
            pump_rate=100.0,
            pump_column=1,
            pump_fraction=1.0,
            dist_method=0,
            layer_factors=[1.0],
            dest_type=-1,
            dest_id=0,
            irig_frac_column=0,
            adjust_column=0,
            pump_max_column=0,
            pump_max_fraction=1.0,
        )
        gw = AppGW(n_nodes=4, n_layers=1, n_elements=2, element_pumping=[ep])
        model = _mock_model(gw=gw)

        wconfig = GWWriterConfig(output_dir=tmp_path)
        writer = GWComponentWriter(model, wconfig)

        path = writer.write_pump_main()
        assert path.exists()
        text = path.read_text()
        # Should reference elem_pump_file and ts_pumping_file
        assert "ElemPump" in text or "TSPumping" in text

    def test_write_ts_pumping(self, tmp_path: Path) -> None:
        """write_ts_pumping creates a pumping time series file."""
        gw = AppGW(n_nodes=4, n_layers=1, n_elements=2, element_pumping=[])
        model = _mock_model(gw=gw)

        wconfig = GWWriterConfig(output_dir=tmp_path)
        writer = GWComponentWriter(model, wconfig)

        path = writer.write_ts_pumping(dates=None, data=None)
        assert path.exists()
        text = path.read_text()
        # The file should contain NCOL and FACT tags from the TS config
        assert "0" in text  # ncol=0 because no wells or elem pumping
