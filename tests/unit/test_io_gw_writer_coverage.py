"""Supplementary tests for gw_writer.py targeting uncovered branches.

Covers:
- GWComponentWriter with empty BC lists
- Single-timestep pumping write
- Subsidence-only write operations
- Write_all with various component combinations
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from pyiwfm.components.groundwater import (
    AppGW,
    BoundaryCondition,
    Subsidence,
    TileDrain,
)
from pyiwfm.io.groundwater import (
    GroundwaterReader,
    GroundwaterWriter,
    GWFileConfig,
)

# =============================================================================
# GroundwaterWriter Additional Tests
# =============================================================================


class TestGroundwaterWriterEmptyComponents:
    """Tests for GroundwaterWriter with empty component lists."""

    def test_write_empty_bc_list(self, tmp_path: Path) -> None:
        """Test writing with empty boundary conditions list."""
        gw = AppGW(n_nodes=10, n_layers=2, n_elements=5)
        # No boundary conditions added

        config = GWFileConfig(output_dir=tmp_path)
        writer = GroundwaterWriter(config)

        filepath = writer.write_boundary_conditions(gw)
        assert filepath.exists()
        content = filepath.read_text()
        assert "0" in content  # Zero BCs

    def test_write_empty_wells(self, tmp_path: Path) -> None:
        """Test writing with no wells."""
        gw = AppGW(n_nodes=10, n_layers=2, n_elements=5)

        config = GWFileConfig(output_dir=tmp_path)
        writer = GroundwaterWriter(config)

        filepath = writer.write_wells(gw)
        assert filepath.exists()
        content = filepath.read_text()
        assert "0" in content  # Zero wells

    def test_write_empty_tile_drains(self, tmp_path: Path) -> None:
        """Test writing with no tile drains."""
        gw = AppGW(n_nodes=10, n_layers=2, n_elements=5)

        config = GWFileConfig(output_dir=tmp_path)
        writer = GroundwaterWriter(config)

        filepath = writer.write_tile_drains(gw)
        assert filepath.exists()
        content = filepath.read_text()
        assert "0" in content  # Zero drains


class TestGroundwaterWriterSubsidenceOnly:
    """Tests for subsidence-only groundwater writes."""

    def test_write_subsidence_only_model(self, tmp_path: Path) -> None:
        """Test writing GW model with only subsidence data."""
        gw = AppGW(n_nodes=10, n_layers=2, n_elements=5)
        gw.add_subsidence(
            Subsidence(
                element=1,
                layer=1,
                elastic_storage=1e-5,
                inelastic_storage=1e-4,
                preconsolidation_head=90.0,
            )
        )
        gw.add_subsidence(
            Subsidence(
                element=2,
                layer=1,
                elastic_storage=2e-5,
                inelastic_storage=2e-4,
                preconsolidation_head=85.0,
            )
        )

        config = GWFileConfig(output_dir=tmp_path)
        writer = GroundwaterWriter(config)

        filepath = writer.write_subsidence(gw)

        assert filepath.exists()
        content = filepath.read_text()
        assert "N_SUBSIDENCE" in content
        assert "2" in content


class TestGroundwaterWriterBoundaryConditions:
    """Tests for writing boundary conditions of all types."""

    def test_write_specified_head_bc(self, tmp_path: Path) -> None:
        """Test writing specified head boundary condition."""
        gw = AppGW(n_nodes=10, n_layers=2, n_elements=5)
        gw.add_boundary_condition(
            BoundaryCondition(
                id=1,
                bc_type="specified_head",
                nodes=[1, 2, 3],
                values=[100.0, 95.0, 90.0],
                layer=1,
            )
        )

        config = GWFileConfig(output_dir=tmp_path)
        writer = GroundwaterWriter(config)
        filepath = writer.write_boundary_conditions(gw)

        content = filepath.read_text()
        assert "SPECIFIED HEAD" in content or "specified_head" in content
        assert "100.0" in content

    def test_write_general_head_bc(self, tmp_path: Path) -> None:
        """Test writing general head boundary condition with conductance."""
        gw = AppGW(n_nodes=10, n_layers=2, n_elements=5)
        gw.add_boundary_condition(
            BoundaryCondition(
                id=1,
                bc_type="general_head",
                nodes=[5, 6],
                values=[80.0, 75.0],
                layer=2,
                conductance=[0.01, 0.02],
            )
        )

        config = GWFileConfig(output_dir=tmp_path)
        writer = GroundwaterWriter(config)
        filepath = writer.write_boundary_conditions(gw)

        content = filepath.read_text()
        assert "GENERAL HEAD" in content or "general_head" in content

    def test_write_mixed_bc_types(self, tmp_path: Path) -> None:
        """Test writing multiple boundary condition types."""
        gw = AppGW(n_nodes=20, n_layers=2, n_elements=10)
        gw.add_boundary_condition(
            BoundaryCondition(
                id=1,
                bc_type="specified_head",
                nodes=[1, 2],
                values=[100.0, 95.0],
                layer=1,
            )
        )
        gw.add_boundary_condition(
            BoundaryCondition(
                id=2,
                bc_type="specified_flow",
                nodes=[10, 11],
                values=[-50.0, -60.0],
                layer=1,
            )
        )
        gw.add_boundary_condition(
            BoundaryCondition(
                id=3,
                bc_type="general_head",
                nodes=[20],
                values=[70.0],
                layer=2,
                conductance=[0.05],
            )
        )

        config = GWFileConfig(output_dir=tmp_path)
        writer = GroundwaterWriter(config)
        filepath = writer.write_boundary_conditions(gw)

        content = filepath.read_text()
        assert "3" in content  # 3 BCs


class TestGroundwaterWriterTileDrains:
    """Tests for tile drain writing."""

    def test_write_tile_drain_with_destination(self, tmp_path: Path) -> None:
        """Test writing tile drain with stream destination."""
        gw = AppGW(n_nodes=10, n_layers=2, n_elements=5)
        gw.add_tile_drain(
            TileDrain(
                id=1,
                element=3,
                elevation=50.0,
                conductance=0.01,
                destination_type="stream",
                destination_id=5,
            )
        )

        config = GWFileConfig(output_dir=tmp_path)
        writer = GroundwaterWriter(config)
        filepath = writer.write_tile_drains(gw)

        content = filepath.read_text()
        assert "stream" in content

    def test_write_tile_drain_outside(self, tmp_path: Path) -> None:
        """Test writing tile drain with outside destination."""
        gw = AppGW(n_nodes=10, n_layers=2, n_elements=5)
        gw.add_tile_drain(
            TileDrain(
                id=1,
                element=2,
                elevation=45.0,
                conductance=0.005,
                destination_type="outside",
            )
        )

        config = GWFileConfig(output_dir=tmp_path)
        writer = GroundwaterWriter(config)
        filepath = writer.write_tile_drains(gw)

        content = filepath.read_text()
        assert "outside" in content


class TestGroundwaterWriterInitialHeads:
    """Tests for initial heads writing."""

    def test_write_initial_heads_multi_layer(self, tmp_path: Path) -> None:
        """Test writing initial heads for multiple layers."""
        gw = AppGW(n_nodes=5, n_layers=3, n_elements=3)
        gw.heads = np.array(
            [
                [100.0, 90.0, 80.0],
                [95.0, 85.0, 75.0],
                [90.0, 80.0, 70.0],
                [85.0, 75.0, 65.0],
                [80.0, 70.0, 60.0],
            ]
        )

        config = GWFileConfig(output_dir=tmp_path)
        writer = GroundwaterWriter(config)
        filepath = writer.write_initial_heads(gw)

        content = filepath.read_text()
        assert "5" in content  # n_nodes
        assert "3" in content  # n_layers

    def test_write_read_initial_heads_roundtrip(self, tmp_path: Path) -> None:
        """Test initial heads write-read roundtrip."""
        gw = AppGW(n_nodes=3, n_layers=2, n_elements=2)
        gw.heads = np.array(
            [
                [100.0, 80.0],
                [95.0, 75.0],
                [90.0, 70.0],
            ]
        )

        config = GWFileConfig(output_dir=tmp_path)
        writer = GroundwaterWriter(config)
        filepath = writer.write_initial_heads(gw)

        reader = GroundwaterReader()
        n_nodes, n_layers, heads = reader.read_initial_heads(filepath)

        assert n_nodes == 3
        assert n_layers == 2
        np.testing.assert_array_almost_equal(heads, gw.heads, decimal=2)


# =============================================================================
# GWComponentWriter Additional Tests (targeting gw_writer.py uncovered lines)
# =============================================================================


from types import SimpleNamespace  # noqa: E402
from unittest.mock import patch  # noqa: E402


class TestGWComponentWriterWriteMain:
    """Tests for GWComponentWriter.write_main() (lines 349-373)."""

    def _make_mock_gw_model(self, **overrides):
        """Build a mock model suitable for GWComponentWriter."""
        model = MagicMock()
        gw = MagicMock()
        gw.boundary_conditions = overrides.get("boundary_conditions", [])
        gw.wells = overrides.get("wells", {})
        gw.element_pumping = overrides.get("element_pumping", [])
        gw.tile_drains = overrides.get("tile_drains", {})
        gw.subsidence = overrides.get("subsidence", [])
        gw.aquifer_params = overrides.get("aquifer_params", None)
        gw.heads = overrides.get("heads", None)
        gw.hydrograph_locations = overrides.get("hydrograph_locations", [])
        gw.face_flow_specs = overrides.get("face_flow_specs", [])
        gw.kh_anomalies = overrides.get("kh_anomalies", [])
        gw.return_flow_destinations = overrides.get("return_flow_destinations", {})
        model.groundwater = gw
        model.n_nodes = overrides.get("n_nodes", 3)

        # Stratigraphy mock
        strat = MagicMock()
        strat.n_layers = overrides.get("n_layers", 2)
        strat.top_elev = np.array([[100, 80], [95, 75], [90, 70]], dtype=float)
        strat.bottom_elev = np.array([[80, 60], [75, 55], [70, 50]], dtype=float)
        model.stratigraphy = strat

        model.metadata = overrides.get("metadata", {})
        model.source_files = overrides.get("source_files", {})
        return model

    def test_write_main_creates_file(self, tmp_path: Path) -> None:
        """write_main() creates the GW main file using template."""
        from pyiwfm.io.gw_writer import GWComponentWriter, GWWriterConfig

        model = self._make_mock_gw_model()
        mock_engine = MagicMock()
        mock_engine.render_template.return_value = "C  GW MAIN HEADER\n"

        config = GWWriterConfig(output_dir=tmp_path)
        writer = GWComponentWriter(model, config, template_engine=mock_engine)
        path = writer.write_main()

        assert path.exists()
        content = path.read_text()
        assert "GW MAIN HEADER" in content
        mock_engine.render_template.assert_called_once()

    def test_write_main_with_aquifer_params(self, tmp_path: Path) -> None:
        """write_main() includes aquifer parameter data when present."""
        from pyiwfm.io.gw_writer import GWComponentWriter, GWWriterConfig

        params = MagicMock()
        params.kh = np.array([[5.0, 3.0], [4.0, 2.0], [3.5, 1.5]])
        params.kv = np.array([[0.5, 0.3], [0.4, 0.2], [0.35, 0.15]])
        params.specific_storage = np.array([[1e-5, 1e-6], [2e-5, 2e-6], [3e-5, 3e-6]])
        params.specific_yield = np.array([[0.2, 0.15], [0.18, 0.12], [0.16, 0.1]])

        model = self._make_mock_gw_model(aquifer_params=params)
        mock_engine = MagicMock()
        mock_engine.render_template.return_value = "C  GW HEADER\n"

        config = GWWriterConfig(output_dir=tmp_path)
        writer = GWComponentWriter(model, config, template_engine=mock_engine)
        path = writer.write_main()

        content = path.read_text()
        assert "5.0000" in content  # kh for node 1, layer 1

    def test_write_main_with_kh_anomalies(self, tmp_path: Path) -> None:
        """write_main() includes Kh anomaly section when present (lines 348-357)."""
        from pyiwfm.io.gw_writer import GWComponentWriter, GWWriterConfig

        model = self._make_mock_gw_model(
            kh_anomalies=["1  2  0.5", "3  4  1.2"],
        )
        mock_engine = MagicMock()
        mock_engine.render_template.return_value = "C  GW HEADER\n"

        config = GWWriterConfig(output_dir=tmp_path)
        writer = GWComponentWriter(model, config, template_engine=mock_engine)
        path = writer.write_main()

        content = path.read_text()
        assert "Kh Anomaly Data" in content
        assert "2" in content  # n_kh_anomalies = 2

    def test_write_main_with_return_flow_destinations(self, tmp_path: Path) -> None:
        """write_main() includes return flow destinations when present (lines 358-373)."""
        from pyiwfm.io.gw_writer import GWComponentWriter, GWWriterConfig

        model = self._make_mock_gw_model(
            return_flow_destinations={1: (0, 5), 3: (1, 10)},
        )
        mock_engine = MagicMock()
        mock_engine.render_template.return_value = "C  GW HEADER\n"

        config = GWWriterConfig(output_dir=tmp_path)
        writer = GWComponentWriter(model, config, template_engine=mock_engine)
        path = writer.write_main()

        content = path.read_text()
        assert "GW Return Flows" in content
        assert "1" in content  # IFLAGRF = 1

    def test_write_main_with_heads(self, tmp_path: Path) -> None:
        """write_main() writes initial heads from model data."""
        from pyiwfm.io.gw_writer import GWComponentWriter, GWWriterConfig

        heads = np.array([[100.0, 80.0], [95.0, 75.0], [90.0, 70.0]])
        model = self._make_mock_gw_model(heads=heads)
        mock_engine = MagicMock()
        mock_engine.render_template.return_value = "C  GW HEADER\n"

        config = GWWriterConfig(output_dir=tmp_path)
        writer = GWComponentWriter(model, config, template_engine=mock_engine)
        path = writer.write_main()

        content = path.read_text()
        assert "Initial Groundwater Heads" in content
        assert "100.0000" in content


class TestGWComponentWriterWriteBcMain:
    """Tests for GWComponentWriter.write_bc_main() (lines 528-534)."""

    def test_write_bc_main_creates_file(self, tmp_path: Path) -> None:
        """write_bc_main() creates the BC main file using template."""
        from pyiwfm.io.gw_writer import GWComponentWriter, GWWriterConfig

        model = MagicMock()
        bc1 = SimpleNamespace(bc_type="specified_head", nodes=[1, 2], layer=1)
        bc2 = SimpleNamespace(bc_type="specified_flow", nodes=[5], layer=1)
        bc3 = SimpleNamespace(bc_type="general_head", nodes=[10], layer=2)
        gw = MagicMock()
        gw.boundary_conditions = [bc1, bc2, bc3]
        gw.n_bc_output_nodes = 0
        gw.bc_output_specs = []
        gw.bc_output_file_raw = ""
        model.groundwater = gw
        model.stratigraphy = MagicMock()
        model.stratigraphy.n_layers = 2
        model.n_nodes = 10

        mock_engine = MagicMock()
        mock_engine.render_template.return_value = "C BC MAIN\n"

        config = GWWriterConfig(output_dir=tmp_path)
        writer = GWComponentWriter(model, config, template_engine=mock_engine)
        path = writer.write_bc_main()

        assert path.exists()
        # Verify template was called with correct context keys
        call_kwargs = mock_engine.render_template.call_args[1]
        # write_bc_main now passes spec_head_bc_file/spec_flow_bc_file etc.
        assert "spec_head_bc_file" in call_kwargs
        assert "spec_flow_bc_file" in call_kwargs
        assert "n_bc_output_nodes" in call_kwargs


class TestGWComponentWriterWritePumpMain:
    """Tests for GWComponentWriter.write_pump_main() (lines 605-806)."""

    def test_write_pump_main_with_elem_pumping(self, tmp_path: Path) -> None:
        """write_pump_main() creates pumping main file with element pumping flag."""
        from pyiwfm.io.gw_writer import GWComponentWriter, GWWriterConfig

        model = MagicMock()
        gw = MagicMock()
        gw.wells = {}
        gw.element_pumping = [MagicMock()]
        model.groundwater = gw
        model.stratigraphy = MagicMock()
        model.stratigraphy.n_layers = 2
        model.n_nodes = 5

        mock_engine = MagicMock()
        mock_engine.render_template.return_value = "C PUMP MAIN\n"

        config = GWWriterConfig(output_dir=tmp_path)
        writer = GWComponentWriter(model, config, template_engine=mock_engine)
        path = writer.write_pump_main()

        assert path.exists()
        # write_pump_main now passes well_spec_file/elem_pump_file paths
        call_kwargs = mock_engine.render_template.call_args[1]
        assert call_kwargs["elem_pump_file"] != ""  # element pumping present
        assert call_kwargs["well_spec_file"] == ""  # no wells

    def test_write_pump_main_with_wells_only(self, tmp_path: Path) -> None:
        """write_pump_main() sets well_spec_file for wells-only."""
        from pyiwfm.io.gw_writer import GWComponentWriter, GWWriterConfig

        model = MagicMock()
        gw = MagicMock()
        gw.wells = {1: MagicMock()}
        gw.element_pumping = []
        model.groundwater = gw
        model.stratigraphy = MagicMock()
        model.stratigraphy.n_layers = 2
        model.n_nodes = 5

        mock_engine = MagicMock()
        mock_engine.render_template.return_value = "C PUMP MAIN\n"

        config = GWWriterConfig(output_dir=tmp_path)
        writer = GWComponentWriter(model, config, template_engine=mock_engine)
        writer.write_pump_main()

        call_kwargs = mock_engine.render_template.call_args[1]
        assert call_kwargs["well_spec_file"] != ""  # wells present
        assert call_kwargs["elem_pump_file"] == ""  # no element pumping


class TestGWComponentWriterWriteTileDrains:
    """Tests for GWComponentWriter.write_tile_drains() (lines 605-806)."""

    def test_write_tile_drains_creates_file(self, tmp_path: Path) -> None:
        """write_tile_drains() creates tile drain file using template."""
        from pyiwfm.io.gw_writer import GWComponentWriter, GWWriterConfig

        model = MagicMock()
        drain = SimpleNamespace(
            id=1,
            elevation=50.0,
            conductance=0.01,
            dest_type="stream",
            destination_id=5,
            gw_node=3,
        )
        gw = MagicMock()
        gw.tile_drains = {1: drain}
        gw.td_elev_factor = 1.0
        gw.td_cond_factor = 1.0
        gw.td_time_unit = "1DAY"
        gw.si_elev_factor = 1.0
        gw.si_cond_factor = 1.0
        gw.si_time_unit = "1MON"
        model.groundwater = gw
        model.stratigraphy = MagicMock()
        model.stratigraphy.n_layers = 2
        model.n_nodes = 5

        mock_engine = MagicMock()
        mock_engine.render_template.return_value = "C TILE DRAIN\n"

        config = GWWriterConfig(output_dir=tmp_path)
        writer = GWComponentWriter(model, config, template_engine=mock_engine)
        path = writer.write_tile_drains()

        assert path.exists()
        call_kwargs = mock_engine.render_template.call_args[1]
        assert call_kwargs["n_drains"] == 1
        assert len(call_kwargs["drains"]) == 1
        assert call_kwargs["drains"][0]["elevation"] == 50.0

    def test_write_tile_drains_with_factor_scaling(self, tmp_path: Path) -> None:
        """write_tile_drains() divides by factor to recover raw values."""
        from pyiwfm.io.gw_writer import GWComponentWriter, GWWriterConfig

        model = MagicMock()
        drain = SimpleNamespace(
            id=1,
            elevation=100.0,
            conductance=0.02,
            dest_type=0,
            destination_id=0,
            element=3,
        )
        gw = MagicMock()
        gw.tile_drains = {1: drain}
        gw.td_elev_factor = 2.0  # non-unity
        gw.td_cond_factor = 4.0  # non-unity
        gw.td_time_unit = "1DAY"
        gw.si_elev_factor = 1.0
        gw.si_cond_factor = 1.0
        gw.si_time_unit = "1MON"
        model.groundwater = gw
        model.stratigraphy = MagicMock()
        model.stratigraphy.n_layers = 1
        model.n_nodes = 3

        mock_engine = MagicMock()
        mock_engine.render_template.return_value = "C TILE DRAIN\n"

        config = GWWriterConfig(output_dir=tmp_path)
        writer = GWComponentWriter(model, config, template_engine=mock_engine)
        writer.write_tile_drains()

        call_kwargs = mock_engine.render_template.call_args[1]
        drain_data = call_kwargs["drains"][0]
        # 100.0 / 2.0 = 50.0
        assert drain_data["elevation"] == pytest.approx(50.0)
        # 0.02 / 4.0 = 0.005
        assert drain_data["conductance"] == pytest.approx(0.005)


class TestGWComponentWriterWriteSubsidence:
    """Tests for GWComponentWriter.write_subsidence() (lines 605-806)."""

    def test_write_subsidence_creates_file(self, tmp_path: Path) -> None:
        """write_subsidence() creates subsidence file using template."""
        from pyiwfm.io.gw_writer import GWComponentWriter, GWWriterConfig

        model = MagicMock()
        gw = MagicMock()
        gw.subsidence = [MagicMock(), MagicMock()]
        model.groundwater = gw
        model.stratigraphy = MagicMock()
        model.stratigraphy.n_layers = 2
        model.n_nodes = 5

        mock_engine = MagicMock()
        mock_engine.render_template.return_value = "C SUBSIDENCE\n"

        config = GWWriterConfig(output_dir=tmp_path)
        writer = GWComponentWriter(model, config, template_engine=mock_engine)
        path = writer.write_subsidence()

        assert path.exists()
        call_kwargs = mock_engine.render_template.call_args[1]
        assert call_kwargs["n_subsidence"] == 2


class TestGWComponentWriterWriteAll:
    """Tests for GWComponentWriter.write_all() component presence checks."""

    def test_write_all_with_all_components(self, tmp_path: Path) -> None:
        """write_all() writes all sub-files when all components are present."""
        from pyiwfm.io.gw_writer import GWComponentWriter, GWWriterConfig

        model = MagicMock()
        gw = MagicMock()
        # Use a SimpleNamespace BC so bc_type attribute works for filtering
        bc1 = SimpleNamespace(bc_type="specified_head", nodes=[1], values=[100.0],
                              layer=1, ts_column=0)
        gw.boundary_conditions = [bc1]
        gw.wells = {1: MagicMock()}
        gw.element_pumping = []
        gw.tile_drains = {1: MagicMock()}
        gw.subsidence = [MagicMock()]
        model.groundwater = gw
        model.stratigraphy = MagicMock()
        model.stratigraphy.n_layers = 2
        model.n_nodes = 5

        mock_engine = MagicMock()
        mock_engine.render_template.return_value = "C CONTENT\n"

        config = GWWriterConfig(output_dir=tmp_path)
        writer = GWComponentWriter(model, config, template_engine=mock_engine)

        with (
            patch.object(writer, "write_main", return_value=tmp_path / "main.dat"),
            patch.object(writer, "write_bc_main", return_value=tmp_path / "bc.dat"),
            patch.object(writer, "write_spec_head_bc", return_value=tmp_path / "shbc.dat"),
            patch.object(writer, "write_bc_ts_data", return_value=None),
            patch.object(writer, "write_pump_main", return_value=tmp_path / "pump.dat"),
            patch.object(writer, "write_well_specs", return_value=tmp_path / "ws.dat"),
            patch.object(writer, "write_tile_drains", return_value=tmp_path / "td.dat"),
            patch.object(writer, "write_subsidence", return_value=tmp_path / "sub.dat"),
        ):
            results = writer.write_all()

        assert "main" in results
        assert "bc_main" in results
        assert "pump_main" in results
        assert "tile_drains" in results
        assert "subsidence" in results

    def test_write_all_skips_absent_components(self, tmp_path: Path) -> None:
        """write_all() skips BC/pump/tile/subsidence when not present."""
        from pyiwfm.io.gw_writer import GWComponentWriter, GWWriterConfig

        model = MagicMock()
        gw = MagicMock()
        gw.boundary_conditions = []
        gw.wells = {}
        gw.element_pumping = []
        gw.tile_drains = {}
        gw.subsidence = []
        model.groundwater = gw
        model.stratigraphy = MagicMock()
        model.stratigraphy.n_layers = 2
        model.n_nodes = 5

        mock_engine = MagicMock()
        mock_engine.render_template.return_value = "C CONTENT\n"

        config = GWWriterConfig(output_dir=tmp_path)
        writer = GWComponentWriter(model, config, template_engine=mock_engine)

        with patch.object(writer, "write_main", return_value=tmp_path / "main.dat"):
            results = writer.write_all()

        assert "main" in results
        assert "bc_main" not in results
        assert "pump_main" not in results
        assert "tile_drains" not in results
        assert "subsidence" not in results

    def test_write_all_write_defaults_false_no_gw(self, tmp_path: Path) -> None:
        """write_all(write_defaults=False) returns empty if gw is None."""
        from pyiwfm.io.gw_writer import GWComponentWriter, GWWriterConfig

        model = MagicMock()
        model.groundwater = None
        model.stratigraphy = MagicMock()
        model.stratigraphy.n_layers = 2
        model.n_nodes = 5

        mock_engine = MagicMock()
        mock_engine.render_template.return_value = "C CONTENT\n"

        config = GWWriterConfig(output_dir=tmp_path)
        writer = GWComponentWriter(model, config, template_engine=mock_engine)
        results = writer.write_all(write_defaults=False)

        assert results == {}


class TestGWComponentWriterSpecBCFiles:
    """Tests for write_spec_head_bc / write_spec_flow_bc (lines 605-806)."""

    def test_write_spec_head_bc(self, tmp_path: Path) -> None:
        """write_spec_head_bc() creates specified head BC file."""
        from pyiwfm.io.gw_writer import GWComponentWriter, GWWriterConfig

        model = MagicMock()
        bc = SimpleNamespace(
            bc_type="specified_head", nodes=[1, 2, 3], values=[100.0, 95.0, 90.0],
            layer=1, ts_column=0,
        )
        gw = MagicMock()
        gw.boundary_conditions = [bc]
        gw.bc_config = None  # Prevent MagicMock from being used as factor
        model.groundwater = gw
        model.stratigraphy = MagicMock()
        model.stratigraphy.n_layers = 2
        model.n_nodes = 10

        mock_engine = MagicMock()

        config = GWWriterConfig(output_dir=tmp_path)
        writer = GWComponentWriter(model, config, template_engine=mock_engine)
        path = writer.write_spec_head_bc()

        assert path.exists()
        # write_spec_head_bc now writes directly (no template), check content
        content = path.read_text()
        assert "NHB" in content
        assert "FACT" in content

    def test_write_spec_flow_bc(self, tmp_path: Path) -> None:
        """write_spec_flow_bc() creates specified flow BC file."""
        from pyiwfm.io.gw_writer import GWComponentWriter, GWWriterConfig

        model = MagicMock()
        bc = SimpleNamespace(
            bc_type="specified_flow", nodes=[5, 6], values=[-50.0, -60.0],
            layer=1, ts_column=0,
        )
        gw = MagicMock()
        gw.boundary_conditions = [bc]
        gw.bc_config = None  # Prevent MagicMock from being used as factor
        model.groundwater = gw
        model.stratigraphy = MagicMock()
        model.stratigraphy.n_layers = 2
        model.n_nodes = 10

        mock_engine = MagicMock()

        config = GWWriterConfig(output_dir=tmp_path)
        writer = GWComponentWriter(model, config, template_engine=mock_engine)
        path = writer.write_spec_flow_bc()

        assert path.exists()
        # write_spec_flow_bc now writes directly (no template), check content
        content = path.read_text()
        assert "NQB" in content
        assert "FACT" in content


class TestGWComponentWriterHydrographAndFaceFlow:
    """Tests for write_hydrograph_specs / write_face_flow_specs (lines 820-833)."""

    def test_write_hydrograph_specs_delegates_to_write_main(self, tmp_path: Path) -> None:
        """write_hydrograph_specs() delegates to write_main()."""
        from pyiwfm.io.gw_writer import GWComponentWriter, GWWriterConfig

        model = MagicMock()
        model.groundwater = MagicMock()
        model.stratigraphy = MagicMock()
        model.stratigraphy.n_layers = 1
        model.n_nodes = 2

        mock_engine = MagicMock()
        mock_engine.render_template.return_value = "C  HEADER\n"

        config = GWWriterConfig(output_dir=tmp_path)
        writer = GWComponentWriter(model, config, template_engine=mock_engine)

        with patch.object(writer, "write_main", return_value=tmp_path / "gw.dat") as mock_wm:
            result = writer.write_hydrograph_specs()

        mock_wm.assert_called_once()
        assert result == tmp_path / "gw.dat"

    def test_write_face_flow_specs_delegates_to_write_main(self, tmp_path: Path) -> None:
        """write_face_flow_specs() delegates to write_main()."""
        from pyiwfm.io.gw_writer import GWComponentWriter, GWWriterConfig

        model = MagicMock()
        model.groundwater = MagicMock()
        model.stratigraphy = MagicMock()
        model.stratigraphy.n_layers = 1
        model.n_nodes = 2

        mock_engine = MagicMock()
        mock_engine.render_template.return_value = "C  HEADER\n"

        config = GWWriterConfig(output_dir=tmp_path)
        writer = GWComponentWriter(model, config, template_engine=mock_engine)

        with patch.object(writer, "write_main", return_value=tmp_path / "gw.dat") as mock_wm:
            result = writer.write_face_flow_specs()

        mock_wm.assert_called_once()
        assert result == tmp_path / "gw.dat"


class TestGWComponentWriterConvenience:
    """Tests for write_gw_component convenience function."""

    def test_write_gw_component_default_config(self, tmp_path: Path) -> None:
        """write_gw_component() creates default config when none provided."""
        from pyiwfm.io.gw_writer import write_gw_component

        model = MagicMock()
        model.groundwater = None
        model.stratigraphy = MagicMock()
        model.stratigraphy.n_layers = 1
        model.n_nodes = 2

        mock_writer = MagicMock()
        mock_writer.write_all.return_value = {"main": tmp_path / "GW" / "GW_MAIN.dat"}

        with patch(
            "pyiwfm.io.gw_writer.GWComponentWriter",
            return_value=mock_writer,
        ):
            results = write_gw_component(model, tmp_path)

        assert "main" in results
        mock_writer.write_all.assert_called_once()

    def test_write_gw_component_with_config(self, tmp_path: Path) -> None:
        """write_gw_component() uses provided config."""
        from pyiwfm.io.gw_writer import GWWriterConfig, write_gw_component

        model = MagicMock()
        model.groundwater = None
        model.stratigraphy = MagicMock()
        model.stratigraphy.n_layers = 1
        model.n_nodes = 2

        config = GWWriterConfig(output_dir=tmp_path, version="4.2")

        mock_writer = MagicMock()
        mock_writer.write_all.return_value = {"main": tmp_path / "GW" / "GW_MAIN.dat"}

        with patch(
            "pyiwfm.io.gw_writer.GWComponentWriter",
            return_value=mock_writer,
        ):
            results = write_gw_component(model, tmp_path, config=config)

        assert config.output_dir == tmp_path
        assert "main" in results


class TestGWComponentWriterTsPumping:
    """Tests for write_ts_pumping() (lines 605-806)."""

    def test_write_ts_pumping_creates_file(self, tmp_path: Path) -> None:
        """write_ts_pumping() creates pumping time series file."""
        from pyiwfm.io.gw_writer import GWComponentWriter, GWWriterConfig

        model = MagicMock()
        gw = MagicMock()
        gw.wells = {1: MagicMock()}
        gw.element_pumping = [MagicMock(), MagicMock()]
        model.groundwater = gw
        model.stratigraphy = MagicMock()
        model.stratigraphy.n_layers = 2
        model.n_nodes = 5

        mock_engine = MagicMock()
        mock_ts_writer = MagicMock()
        expected_path = tmp_path / "GW" / "TSPumping.dat"
        mock_ts_writer.write.return_value = expected_path

        config = GWWriterConfig(output_dir=tmp_path)

        with patch(
            "pyiwfm.io.timeseries_writer.IWFMTimeSeriesDataWriter",
            return_value=mock_ts_writer,
        ):
            writer = GWComponentWriter(model, config, template_engine=mock_engine)
            path = writer.write_ts_pumping()

        mock_ts_writer.write.assert_called_once()
        assert path == expected_path


class TestGWWriterConfigProperties:
    """Tests for GWWriterConfig property methods."""

    def test_gw_dir_property(self, tmp_path: Path) -> None:
        """gw_dir returns output_dir / gw_subdir."""
        from pyiwfm.io.gw_writer import GWWriterConfig

        config = GWWriterConfig(output_dir=tmp_path, gw_subdir="Groundwater")
        assert config.gw_dir == tmp_path / "Groundwater"

    def test_main_path_property(self, tmp_path: Path) -> None:
        """main_path returns gw_dir / main_file."""
        from pyiwfm.io.gw_writer import GWWriterConfig

        config = GWWriterConfig(output_dir=tmp_path, main_file="Custom_GW.dat")
        assert config.main_path == tmp_path / "GW" / "Custom_GW.dat"

    def test_bc_main_path_property(self, tmp_path: Path) -> None:
        """bc_main_path returns gw_dir / bc_main_file."""
        from pyiwfm.io.gw_writer import GWWriterConfig

        config = GWWriterConfig(output_dir=tmp_path, bc_main_file="MyBC.dat")
        assert config.bc_main_path == tmp_path / "GW" / "MyBC.dat"

    def test_pump_main_path_property(self, tmp_path: Path) -> None:
        """pump_main_path returns gw_dir / pump_main_file."""
        from pyiwfm.io.gw_writer import GWWriterConfig

        config = GWWriterConfig(output_dir=tmp_path, pump_main_file="MyPump.dat")
        assert config.pump_main_path == tmp_path / "GW" / "MyPump.dat"

    def test_format_property(self, tmp_path: Path) -> None:
        """format property returns 'iwfm_groundwater'."""
        from pyiwfm.io.gw_writer import GWComponentWriter, GWWriterConfig

        model = MagicMock()
        model.groundwater = None
        model.stratigraphy = MagicMock()
        model.stratigraphy.n_layers = 1
        model.n_nodes = 2

        config = GWWriterConfig(output_dir=tmp_path)
        mock_engine = MagicMock()
        writer = GWComponentWriter(model, config, template_engine=mock_engine)
        assert writer.format == "iwfm_groundwater"

    def test_write_method_delegates_to_write_all(self, tmp_path: Path) -> None:
        """write() delegates to write_all()."""
        from pyiwfm.io.gw_writer import GWComponentWriter, GWWriterConfig

        model = MagicMock()
        model.groundwater = None
        model.stratigraphy = MagicMock()
        model.stratigraphy.n_layers = 1
        model.n_nodes = 2

        config = GWWriterConfig(output_dir=tmp_path)
        mock_engine = MagicMock()
        mock_engine.render_template.return_value = "C CONTENT\n"
        writer = GWComponentWriter(model, config, template_engine=mock_engine)

        with patch.object(writer, "write_all") as mock_wa:
            writer.write()

        mock_wa.assert_called_once()
