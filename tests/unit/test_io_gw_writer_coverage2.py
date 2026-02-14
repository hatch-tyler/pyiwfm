"""Coverage tests for pyiwfm.io.gw_writer module (GWComponentWriter).

Note: test_io_gw_writer_coverage.py tests the older GroundwaterWriter from
pyiwfm.io.groundwater. This file tests the newer GWComponentWriter from
pyiwfm.io.gw_writer, targeting uncovered branches.

Covers:
- GWWriterConfig defaults and properties (gw_dir, main_path, bc_main_path,
  pump_main_path)
- write_all() with write_defaults=False and no GW component
- write_all() with full GW component (boundary conditions, wells, tile drains,
  subsidence)
- write_main() creates output file
- write_bc_main() creates boundary condition main file
- write_pump_main() creates pumping main file
- write_tile_drains() creates tile drain file
- write_subsidence() creates subsidence file
- Component absence checks (skip pump when no wells, skip bc when no BCs)
- write_gw_component() convenience function
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pyiwfm.io.gw_writer import (
    GWWriterConfig,
    GWComponentWriter,
    write_gw_component,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_engine():
    """Create a mock TemplateEngine that returns predictable content."""
    engine = MagicMock()
    engine.render_template.return_value = "C  MOCK GW HEADER\n"
    engine.render_string.return_value = "C  MOCK GW STRING\n"
    return engine


@pytest.fixture
def bare_model():
    """Create a model with groundwater=None."""
    model = MagicMock()
    model.groundwater = None
    model.n_nodes = 4
    model.stratigraphy = MagicMock()
    model.stratigraphy.n_layers = 1
    model.stratigraphy.top_elev = np.ones((4, 1)) * 100.0
    model.stratigraphy.bottom_elev = np.ones((4, 1)) * 50.0
    return model


@pytest.fixture
def model_with_empty_gw():
    """Create a model with a GW component but no sub-data."""
    model = MagicMock()
    gw = MagicMock()
    gw.boundary_conditions = []
    gw.wells = {}
    gw.element_pumping = {}
    gw.tile_drains = {}
    gw.subsidence = []
    gw.aquifer_params = None
    gw.heads = None
    model.groundwater = gw
    model.n_nodes = 4
    model.stratigraphy = MagicMock()
    model.stratigraphy.n_layers = 1
    model.stratigraphy.top_elev = np.ones((4, 1)) * 100.0
    model.stratigraphy.bottom_elev = np.ones((4, 1)) * 50.0
    return model


@pytest.fixture
def model_with_full_gw():
    """Create a model with a full GW component."""
    model = MagicMock()
    gw = MagicMock()

    # Boundary conditions
    bc1 = MagicMock()
    bc1.bc_type = "specified_head"
    bc1.nodes = [1, 2]
    bc1.layer = 1
    gw.boundary_conditions = [bc1]

    # Wells
    well1 = MagicMock()
    well1.id = 1
    well1.element = 2
    well1.x = 100.0
    well1.y = 200.0
    well1.bottom_screen = -50.0
    well1.top_screen = 0.0
    well1.name = "Well_1"
    gw.wells = {1: well1}
    gw.element_pumping = {}

    # Tile drains
    drain1 = SimpleNamespace(
        id=1, element=3, gw_node=3, elevation=75.0,
        conductance=0.01, dest_type=1, destination_id=10,
    )
    gw.tile_drains = {1: drain1}
    gw.td_elev_factor = 1.0
    gw.td_cond_factor = 1.0
    gw.td_time_unit = "1DAY"
    gw.si_elev_factor = 1.0
    gw.si_cond_factor = 1.0
    gw.si_time_unit = "1MON"

    # Subsidence
    sub1 = MagicMock()
    sub1.element = 1
    sub1.layer = 1
    gw.subsidence = [sub1]

    # Aquifer parameters
    params = MagicMock()
    params.kh = np.ones((4, 1)) * 10.0
    params.kv = np.ones((4, 1)) * 1.0
    params.specific_storage = np.ones((4, 1)) * 1e-5
    params.specific_yield = np.ones((4, 1)) * 0.15
    gw.aquifer_params = params

    # Initial heads
    gw.heads = np.ones((4, 1)) * 90.0

    model.groundwater = gw
    model.n_nodes = 4
    model.stratigraphy = MagicMock()
    model.stratigraphy.n_layers = 1
    model.stratigraphy.top_elev = np.ones((4, 1)) * 100.0
    model.stratigraphy.bottom_elev = np.ones((4, 1)) * 50.0
    return model


# =============================================================================
# GWWriterConfig Tests
# =============================================================================


class TestGWWriterConfigCoverage:
    """Coverage tests for GWWriterConfig dataclass."""

    def test_gw_dir_property(self, tmp_path: Path) -> None:
        """Test gw_dir combines output_dir and gw_subdir."""
        config = GWWriterConfig(output_dir=tmp_path, gw_subdir="GW")
        assert config.gw_dir == tmp_path / "GW"

    def test_main_path_property(self, tmp_path: Path) -> None:
        """Test main_path combines gw_dir and main_file."""
        config = GWWriterConfig(output_dir=tmp_path)
        assert config.main_path == tmp_path / "GW" / "GW_MAIN.dat"

    def test_bc_main_path_property(self, tmp_path: Path) -> None:
        """Test bc_main_path combines gw_dir and bc_main_file."""
        config = GWWriterConfig(output_dir=tmp_path)
        assert config.bc_main_path == tmp_path / "GW" / "BC_MAIN.dat"

    def test_pump_main_path_property(self, tmp_path: Path) -> None:
        """Test pump_main_path combines gw_dir and pump_main_file."""
        config = GWWriterConfig(output_dir=tmp_path)
        assert config.pump_main_path == tmp_path / "GW" / "Pump_MAIN.dat"

    def test_custom_subdir_property(self, tmp_path: Path) -> None:
        """Test properties with custom subdir."""
        config = GWWriterConfig(output_dir=tmp_path, gw_subdir="Groundwater")
        assert config.gw_dir == tmp_path / "Groundwater"
        assert config.main_path == tmp_path / "Groundwater" / "GW_MAIN.dat"


# =============================================================================
# write_all() Tests
# =============================================================================


class TestGWWriteAll:
    """Tests for GWComponentWriter.write_all()."""

    def test_write_all_no_gw_write_defaults_false(
        self, tmp_path: Path, bare_model: MagicMock, mock_engine: MagicMock
    ) -> None:
        """write_all(write_defaults=False) returns empty dict when no GW."""
        config = GWWriterConfig(output_dir=tmp_path)
        writer = GWComponentWriter(bare_model, config, template_engine=mock_engine)
        results = writer.write_all(write_defaults=False)
        assert results == {}

    def test_write_all_no_gw_write_defaults_true(
        self, tmp_path: Path, bare_model: MagicMock, mock_engine: MagicMock
    ) -> None:
        """write_all(write_defaults=True) writes main even without GW component."""
        config = GWWriterConfig(output_dir=tmp_path)
        writer = GWComponentWriter(bare_model, config, template_engine=mock_engine)
        results = writer.write_all(write_defaults=True)
        assert "main" in results
        assert results["main"].exists()

    def test_write_all_skips_bc_when_empty(
        self, tmp_path: Path, model_with_empty_gw: MagicMock, mock_engine: MagicMock
    ) -> None:
        """write_all() skips bc_main when boundary_conditions is empty."""
        config = GWWriterConfig(output_dir=tmp_path)
        writer = GWComponentWriter(
            model_with_empty_gw, config, template_engine=mock_engine
        )
        results = writer.write_all()
        assert "main" in results
        assert "bc_main" not in results

    def test_write_all_skips_pump_when_no_wells(
        self, tmp_path: Path, model_with_empty_gw: MagicMock, mock_engine: MagicMock
    ) -> None:
        """write_all() skips pump_main when no wells or element pumping."""
        config = GWWriterConfig(output_dir=tmp_path)
        writer = GWComponentWriter(
            model_with_empty_gw, config, template_engine=mock_engine
        )
        results = writer.write_all()
        assert "pump_main" not in results

    def test_write_all_skips_tile_drains_when_empty(
        self, tmp_path: Path, model_with_empty_gw: MagicMock, mock_engine: MagicMock
    ) -> None:
        """write_all() skips tile_drains when tile_drains is empty."""
        config = GWWriterConfig(output_dir=tmp_path)
        writer = GWComponentWriter(
            model_with_empty_gw, config, template_engine=mock_engine
        )
        results = writer.write_all()
        assert "tile_drains" not in results

    def test_write_all_skips_subsidence_when_empty(
        self, tmp_path: Path, model_with_empty_gw: MagicMock, mock_engine: MagicMock
    ) -> None:
        """write_all() skips subsidence when subsidence is empty list."""
        config = GWWriterConfig(output_dir=tmp_path)
        writer = GWComponentWriter(
            model_with_empty_gw, config, template_engine=mock_engine
        )
        results = writer.write_all()
        assert "subsidence" not in results

    def test_write_all_full_gw_includes_all_files(
        self, tmp_path: Path, model_with_full_gw: MagicMock, mock_engine: MagicMock
    ) -> None:
        """write_all() includes all files when full GW data is present."""
        config = GWWriterConfig(output_dir=tmp_path)
        writer = GWComponentWriter(
            model_with_full_gw, config, template_engine=mock_engine
        )
        results = writer.write_all()
        assert "main" in results
        assert "bc_main" in results
        assert "pump_main" in results
        assert "tile_drains" in results
        assert "subsidence" in results


# =============================================================================
# Individual write method tests
# =============================================================================


class TestGWWriteMain:
    """Tests for GWComponentWriter.write_main()."""

    def test_write_main_creates_file(
        self, tmp_path: Path, model_with_empty_gw: MagicMock, mock_engine: MagicMock
    ) -> None:
        """write_main() creates the main output file."""
        config = GWWriterConfig(output_dir=tmp_path)
        writer = GWComponentWriter(
            model_with_empty_gw, config, template_engine=mock_engine
        )
        path = writer.write_main()
        assert path.exists()
        assert path == config.main_path

    def test_write_main_contains_aquifer_params(
        self, tmp_path: Path, model_with_full_gw: MagicMock, mock_engine: MagicMock
    ) -> None:
        """write_main() output contains aquifer parameter data rows."""
        config = GWWriterConfig(output_dir=tmp_path)
        writer = GWComponentWriter(
            model_with_full_gw, config, template_engine=mock_engine
        )
        path = writer.write_main()
        content = path.read_text()
        # Should contain node IDs and aquifer params
        assert "1" in content
        assert "Initial Groundwater Heads" in content


class TestGWWriteBcMain:
    """Tests for GWComponentWriter.write_bc_main()."""

    def test_write_bc_main_creates_file(
        self, tmp_path: Path, model_with_full_gw: MagicMock, mock_engine: MagicMock
    ) -> None:
        """write_bc_main() creates the BC main file."""
        config = GWWriterConfig(output_dir=tmp_path)
        writer = GWComponentWriter(
            model_with_full_gw, config, template_engine=mock_engine
        )
        path = writer.write_bc_main()
        assert path.exists()
        assert path == config.bc_main_path


class TestGWWritePumpMain:
    """Tests for GWComponentWriter.write_pump_main()."""

    def test_write_pump_main_creates_file(
        self, tmp_path: Path, model_with_full_gw: MagicMock, mock_engine: MagicMock
    ) -> None:
        """write_pump_main() creates the pumping main file."""
        config = GWWriterConfig(output_dir=tmp_path)
        writer = GWComponentWriter(
            model_with_full_gw, config, template_engine=mock_engine
        )
        path = writer.write_pump_main()
        assert path.exists()
        assert path == config.pump_main_path


class TestGWWriteTileDrains:
    """Tests for GWComponentWriter.write_tile_drains()."""

    def test_write_tile_drains_creates_file(
        self, tmp_path: Path, model_with_full_gw: MagicMock, mock_engine: MagicMock
    ) -> None:
        """write_tile_drains() creates the tile drain file."""
        config = GWWriterConfig(output_dir=tmp_path)
        writer = GWComponentWriter(
            model_with_full_gw, config, template_engine=mock_engine
        )
        path = writer.write_tile_drains()
        assert path.exists()


class TestGWWriteSubsidence:
    """Tests for GWComponentWriter.write_subsidence()."""

    def test_write_subsidence_creates_file(
        self, tmp_path: Path, model_with_full_gw: MagicMock, mock_engine: MagicMock
    ) -> None:
        """write_subsidence() creates the subsidence file."""
        config = GWWriterConfig(output_dir=tmp_path)
        writer = GWComponentWriter(
            model_with_full_gw, config, template_engine=mock_engine
        )
        path = writer.write_subsidence()
        assert path.exists()


# =============================================================================
# Miscellaneous Tests
# =============================================================================


class TestGWWriterMisc:
    """Miscellaneous coverage tests."""

    def test_format_property(
        self, tmp_path: Path, bare_model: MagicMock, mock_engine: MagicMock
    ) -> None:
        """format property returns 'iwfm_groundwater'."""
        config = GWWriterConfig(output_dir=tmp_path)
        writer = GWComponentWriter(bare_model, config, template_engine=mock_engine)
        assert writer.format == "iwfm_groundwater"

    def test_write_method_delegates(
        self, tmp_path: Path, bare_model: MagicMock, mock_engine: MagicMock
    ) -> None:
        """write() delegates to write_all()."""
        config = GWWriterConfig(output_dir=tmp_path)
        writer = GWComponentWriter(bare_model, config, template_engine=mock_engine)
        writer.write()
        # write() with write_defaults=True should create main file
        assert config.main_path.exists()

    def test_write_gw_component_convenience(
        self, tmp_path: Path, bare_model: MagicMock, mock_engine: MagicMock
    ) -> None:
        """write_gw_component() convenience function works."""
        with patch("pyiwfm.io.gw_writer.TemplateEngine", return_value=mock_engine):
            results = write_gw_component(bare_model, tmp_path)
        assert "main" in results

    def test_write_gw_component_with_config(
        self, tmp_path: Path, bare_model: MagicMock, mock_engine: MagicMock
    ) -> None:
        """write_gw_component() uses provided config and updates output_dir."""
        config = GWWriterConfig(output_dir=tmp_path, version="5.0")
        new_dir = tmp_path / "output2"
        new_dir.mkdir()
        with patch("pyiwfm.io.gw_writer.TemplateEngine", return_value=mock_engine):
            results = write_gw_component(bare_model, new_dir, config=config)
        assert config.output_dir == new_dir
        assert "main" in results
