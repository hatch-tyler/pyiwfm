"""Integration tests for read/write roundtrip with real model files.

Tests use the IWFM Sample Model and C2VSimFG when available.
All tests are skipped if model paths are not accessible.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyiwfm.io.groundwater import (
    GWFileConfig,
    GroundwaterWriter,
    GroundwaterReader,
    read_subsidence,
)
from pyiwfm.components.groundwater import (
    AppGW,
    Well,
    Subsidence,
)


pytestmark = pytest.mark.integration


# =============================================================================
# Groundwater Roundtrip
# =============================================================================


class TestGroundwaterRoundtrip:
    """Roundtrip tests for groundwater I/O."""

    def test_wells_roundtrip(self, tmp_path: Path) -> None:
        """Test wells write -> read roundtrip with realistic data."""
        gw = AppGW(n_nodes=100, n_layers=3, n_elements=50)
        for i in range(20):
            gw.add_well(Well(
                id=i + 1,
                x=1e6 + float(i * 5280),
                y=2e6 + float(i * 2640),
                element=i % 50 + 1,
                name=f"Production Well {i + 1}",
                top_screen=200.0 - i * 5,
                bottom_screen=100.0 - i * 5,
                max_pump_rate=500.0 + i * 10.0,
            ))

        config = GWFileConfig(output_dir=tmp_path)
        writer = GroundwaterWriter(config)
        filepath = writer.write_wells(gw)

        reader = GroundwaterReader()
        wells = reader.read_wells(filepath)

        assert len(wells) == 20
        for wid in gw.wells:
            orig = gw.wells[wid]
            read = wells[wid]
            assert read.x == pytest.approx(orig.x, rel=1e-3)
            assert read.y == pytest.approx(orig.y, rel=1e-3)
            assert read.element == orig.element
            assert read.name == orig.name

    def test_subsidence_roundtrip(self, tmp_path: Path) -> None:
        """Test subsidence write -> read roundtrip with realistic data."""
        gw = AppGW(n_nodes=100, n_layers=3, n_elements=50)

        # Create subsidence entries across elements and layers
        for elem in range(1, 11):
            for layer in range(1, 4):
                gw.add_subsidence(Subsidence(
                    element=elem,
                    layer=layer,
                    elastic_storage=1e-5 * (1 + elem * 0.1),
                    inelastic_storage=1e-4 * (1 + elem * 0.2),
                    preconsolidation_head=100.0 - layer * 10.0 - elem * 2.0,
                ))

        config = GWFileConfig(output_dir=tmp_path)
        writer = GroundwaterWriter(config)
        filepath = writer.write_subsidence(gw)

        reader = GroundwaterReader()
        result = reader.read_subsidence(filepath)

        assert len(result) == 30  # 10 elements x 3 layers
        for orig, read in zip(gw.subsidence, result):
            assert read.element == orig.element
            assert read.layer == orig.layer
            assert read.elastic_storage == pytest.approx(orig.elastic_storage, rel=1e-3)
            assert read.inelastic_storage == pytest.approx(orig.inelastic_storage, rel=1e-3)
            assert read.preconsolidation_head == pytest.approx(
                orig.preconsolidation_head, rel=1e-3
            )

    def test_initial_heads_roundtrip(self, tmp_path: Path) -> None:
        """Test initial heads write -> read roundtrip with realistic data."""
        n_nodes = 50
        n_layers = 3
        gw = AppGW(n_nodes=n_nodes, n_layers=n_layers, n_elements=25)

        # Generate realistic head values
        heads = np.zeros((n_nodes, n_layers))
        for node_idx in range(n_nodes):
            for layer in range(n_layers):
                heads[node_idx, layer] = 200.0 - layer * 30.0 - node_idx * 0.5
        gw.heads = heads

        config = GWFileConfig(output_dir=tmp_path)
        writer = GroundwaterWriter(config)
        filepath = writer.write_initial_heads(gw)

        reader = GroundwaterReader()
        read_n_nodes, read_n_layers, read_heads = reader.read_initial_heads(filepath)

        assert read_n_nodes == n_nodes
        assert read_n_layers == n_layers
        np.testing.assert_array_almost_equal(read_heads, heads, decimal=2)


# =============================================================================
# Sample Model Tests (skipped if path unavailable)
# =============================================================================


class TestSampleModelIO:
    """Tests using the IWFM Sample Model files."""

    def test_sample_model_exists(self, sample_model_path: Path) -> None:
        """Verify sample model directory structure."""
        assert sample_model_path.exists()
        # Check for expected subdirectories
        assert any(sample_model_path.iterdir())

    def test_sample_model_has_simulation_dir(self, sample_model_path: Path) -> None:
        """Check for Simulation subdirectory."""
        sim_dir = sample_model_path / "Simulation"
        if not sim_dir.exists():
            pytest.skip("Simulation directory not found in sample model")
        assert sim_dir.is_dir()


# =============================================================================
# C2VSimFG Tests (skipped if path unavailable)
# =============================================================================


class TestC2VSimFGIO:
    """Tests using C2VSimFG model files."""

    def test_c2vsimfg_exists(self, c2vsimfg_path: Path) -> None:
        """Verify C2VSimFG directory structure."""
        assert c2vsimfg_path.exists()
        assert any(c2vsimfg_path.iterdir())

    def test_c2vsimfg_has_simulation_dir(self, c2vsimfg_path: Path) -> None:
        """Check for Simulation subdirectory in C2VSimFG."""
        sim_dir = c2vsimfg_path / "Simulation"
        if not sim_dir.exists():
            pytest.skip("Simulation directory not found in C2VSimFG")
        assert sim_dir.is_dir()
