"""Tests for PEST++ subsidence parameterization verification.

Verifies subsidence parameter types (ske, skv, pcs), bounds, transforms,
and parameterization strategies work correctly end-to-end.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from pyiwfm.runner.pest_params import (
    IWFMParameterType,
    ParameterTransform,
    Parameter,
    ZoneParameterization,
    PilotPointParameterization,
    DirectParameterization,
)
from pyiwfm.components.groundwater import (
    AppGW,
    Subsidence,
)
from pyiwfm.io.groundwater import (
    GWFileConfig,
    GroundwaterWriter,
    GroundwaterReader,
)


# =============================================================================
# Test Subsidence Parameter Types
# =============================================================================


class TestSubsidenceParameterTypes:
    """Verify subsidence parameter type definitions."""

    def test_elastic_storage_type_exists(self) -> None:
        """Test ELASTIC_STORAGE parameter type exists."""
        assert IWFMParameterType.ELASTIC_STORAGE.value == "ske"

    def test_inelastic_storage_type_exists(self) -> None:
        """Test INELASTIC_STORAGE parameter type exists."""
        assert IWFMParameterType.INELASTIC_STORAGE.value == "skv"

    def test_preconsolidation_type_exists(self) -> None:
        """Test PRECONSOLIDATION parameter type exists."""
        assert IWFMParameterType.PRECONSOLIDATION.value == "pcs"

    def test_elastic_storage_bounds(self) -> None:
        """Test elastic storage default bounds."""
        lb, ub = IWFMParameterType.ELASTIC_STORAGE.default_bounds
        assert lb == pytest.approx(1e-6)
        assert ub == pytest.approx(1e-3)
        assert lb < ub

    def test_inelastic_storage_bounds(self) -> None:
        """Test inelastic storage default bounds."""
        lb, ub = IWFMParameterType.INELASTIC_STORAGE.default_bounds
        assert lb == pytest.approx(1e-5)
        assert ub == pytest.approx(1e-2)
        assert lb < ub

    def test_preconsolidation_bounds(self) -> None:
        """Test preconsolidation head default bounds."""
        lb, ub = IWFMParameterType.PRECONSOLIDATION.default_bounds
        assert lb == pytest.approx(0.0)
        assert ub == pytest.approx(1000.0)
        assert lb < ub

    def test_elastic_storage_log_transform(self) -> None:
        """Test elastic storage uses log transform."""
        assert IWFMParameterType.ELASTIC_STORAGE.default_transform == "log"

    def test_inelastic_storage_log_transform(self) -> None:
        """Test inelastic storage uses log transform."""
        assert IWFMParameterType.INELASTIC_STORAGE.default_transform == "log"

    def test_preconsolidation_no_transform(self) -> None:
        """Test preconsolidation uses no transform (linear)."""
        assert IWFMParameterType.PRECONSOLIDATION.default_transform == "none"

    def test_subsidence_not_multiplier(self) -> None:
        """Test subsidence parameters are not multipliers."""
        assert IWFMParameterType.ELASTIC_STORAGE.is_multiplier is False
        assert IWFMParameterType.INELASTIC_STORAGE.is_multiplier is False
        assert IWFMParameterType.PRECONSOLIDATION.is_multiplier is False


# =============================================================================
# Test Zone Parameterization for Subsidence
# =============================================================================


class TestSubsidenceZoneParameterization:
    """Test zone-based parameterization for subsidence parameters."""

    def test_ske_zone_parameterization(self) -> None:
        """Test elastic storage zone parameterization."""
        strategy = ZoneParameterization(
            param_type=IWFMParameterType.ELASTIC_STORAGE,
            zones=[1, 2, 3],
            layer=1,
            initial_values=1e-5,
        )
        params = strategy.generate_parameters(None)

        assert len(params) == 3
        assert all(p.param_type == IWFMParameterType.ELASTIC_STORAGE for p in params)
        assert all(p.initial_value == pytest.approx(1e-5) for p in params)
        assert all(p.layer == 1 for p in params)

    def test_skv_zone_parameterization(self) -> None:
        """Test inelastic storage zone parameterization."""
        strategy = ZoneParameterization(
            param_type=IWFMParameterType.INELASTIC_STORAGE,
            zones=[1, 2],
            layer=2,
        )
        params = strategy.generate_parameters(None)

        assert len(params) == 2
        assert all(p.param_type == IWFMParameterType.INELASTIC_STORAGE for p in params)
        assert all(p.layer == 2 for p in params)

    def test_pcs_zone_parameterization(self) -> None:
        """Test preconsolidation head zone parameterization."""
        strategy = ZoneParameterization(
            param_type=IWFMParameterType.PRECONSOLIDATION,
            zones=[1],
            initial_values=100.0,
        )
        params = strategy.generate_parameters(None)

        assert len(params) == 1
        assert params[0].param_type == IWFMParameterType.PRECONSOLIDATION
        assert params[0].initial_value == pytest.approx(100.0)

    def test_ske_naming_format(self) -> None:
        """Test elastic storage parameter naming convention."""
        strategy = ZoneParameterization(
            param_type=IWFMParameterType.ELASTIC_STORAGE,
            zones=[1, 2],
            layer=1,
        )
        params = strategy.generate_parameters(None)

        names = [p.name for p in params]
        assert all("ske" in n for n in names)

    def test_multi_layer_ske(self) -> None:
        """Test elastic storage across multiple layers."""
        params_all = []
        for layer in range(1, 4):
            strategy = ZoneParameterization(
                param_type=IWFMParameterType.ELASTIC_STORAGE,
                zones=[1, 2],
                layer=layer,
            )
            params_all.extend(strategy.generate_parameters(None))

        assert len(params_all) == 6  # 2 zones x 3 layers
        layers = [p.layer for p in params_all]
        assert layers.count(1) == 2
        assert layers.count(2) == 2
        assert layers.count(3) == 2

    def test_per_zone_initial_values(self) -> None:
        """Test zone-specific initial values for subsidence."""
        strategy = ZoneParameterization(
            param_type=IWFMParameterType.ELASTIC_STORAGE,
            zones=[1, 2, 3],
            initial_values={1: 1e-5, 2: 5e-5, 3: 1e-4},
        )
        params = strategy.generate_parameters(None)

        values_by_zone = {p.zone: p.initial_value for p in params}
        assert values_by_zone[1] == pytest.approx(1e-5)
        assert values_by_zone[2] == pytest.approx(5e-5)
        assert values_by_zone[3] == pytest.approx(1e-4)


# =============================================================================
# Test Pilot Point Parameterization for Subsidence
# =============================================================================


class TestSubsidencePilotPointParameterization:
    """Test pilot-point parameterization for subsidence."""

    def test_ske_pilot_points(self) -> None:
        """Test elastic storage pilot point parameterization."""
        points = [(0, 0), (500, 0), (0, 500), (500, 500)]
        strategy = PilotPointParameterization(
            param_type=IWFMParameterType.ELASTIC_STORAGE,
            points=points,
            layer=1,
        )
        params = strategy.generate_parameters(None)

        assert len(params) == 4
        assert all(p.location is not None for p in params)
        assert all(p.param_type == IWFMParameterType.ELASTIC_STORAGE for p in params)

    def test_ske_pp_naming(self) -> None:
        """Test pilot point parameter naming for elastic storage."""
        strategy = PilotPointParameterization(
            param_type=IWFMParameterType.ELASTIC_STORAGE,
            points=[(0, 0), (100, 100)],
            layer=1,
        )
        params = strategy.generate_parameters(None)

        names = [p.name for p in params]
        assert all("ske" in n for n in names)
        assert all("pp" in n.lower() for n in names)

    def test_ske_variogram_metadata(self) -> None:
        """Test variogram metadata is preserved for subsidence PP."""
        variogram = {"type": "exponential", "a": 5000, "sill": 1.0}
        strategy = PilotPointParameterization(
            param_type=IWFMParameterType.ELASTIC_STORAGE,
            points=[(0, 0), (100, 100)],
            layer=1,
            variogram=variogram,
        )
        params = strategy.generate_parameters(None)

        assert params[0].metadata["variogram"]["type"] == "exponential"
        assert params[0].metadata["variogram"]["a"] == 5000


# =============================================================================
# Test Direct Parameterization for Subsidence
# =============================================================================


class TestSubsidenceDirectParameterization:
    """Test direct parameterization for subsidence."""

    def test_direct_ske(self) -> None:
        """Test direct elastic storage parameter."""
        strategy = DirectParameterization(
            param_type=IWFMParameterType.ELASTIC_STORAGE,
            name="ske_global",
            initial_value=1e-5,
        )
        params = strategy.generate_parameters(None)

        assert len(params) == 1
        assert params[0].name == "ske_global"
        assert params[0].initial_value == pytest.approx(1e-5)

    def test_direct_skv(self) -> None:
        """Test direct inelastic storage parameter."""
        strategy = DirectParameterization(
            param_type=IWFMParameterType.INELASTIC_STORAGE,
            name="skv_global",
            initial_value=1e-4,
        )
        params = strategy.generate_parameters(None)

        assert len(params) == 1
        assert params[0].name == "skv_global"
        assert params[0].initial_value == pytest.approx(1e-4)

    def test_direct_pcs(self) -> None:
        """Test direct preconsolidation head parameter."""
        strategy = DirectParameterization(
            param_type=IWFMParameterType.PRECONSOLIDATION,
            name="pcs_global",
            initial_value=100.0,
        )
        params = strategy.generate_parameters(None)

        assert len(params) == 1
        assert params[0].name == "pcs_global"
        assert params[0].initial_value == pytest.approx(100.0)


# =============================================================================
# Test End-to-End Subsidence Workflow
# =============================================================================


class TestSubsidenceEndToEnd:
    """End-to-end tests for subsidence parameterization workflow."""

    def test_create_write_parameterize_read(self, tmp_path: Path) -> None:
        """Full workflow: create AppGW -> write -> parameterize -> verify."""
        # Step 1: Create AppGW with subsidence data
        gw = AppGW(n_nodes=20, n_layers=2, n_elements=10)
        subsidence_data = [
            Subsidence(element=1, layer=1, elastic_storage=1e-5,
                      inelastic_storage=1e-4, preconsolidation_head=90.0),
            Subsidence(element=1, layer=2, elastic_storage=2e-5,
                      inelastic_storage=2e-4, preconsolidation_head=85.0),
            Subsidence(element=5, layer=1, elastic_storage=5e-6,
                      inelastic_storage=5e-5, preconsolidation_head=95.0),
        ]
        for sub in subsidence_data:
            gw.add_subsidence(sub)

        # Step 2: Write subsidence to file
        config = GWFileConfig(output_dir=tmp_path)
        writer = GroundwaterWriter(config)
        filepath = writer.write_subsidence(gw)
        assert filepath.exists()

        # Step 3: Generate PEST parameters for subsidence
        ske_strategy = ZoneParameterization(
            param_type=IWFMParameterType.ELASTIC_STORAGE,
            zones=[1, 2],
            layer=1,
            initial_values={1: 1e-5, 2: 5e-6},
        )
        ske_params = ske_strategy.generate_parameters(None)

        skv_strategy = ZoneParameterization(
            param_type=IWFMParameterType.INELASTIC_STORAGE,
            zones=[1, 2],
            layer=1,
            initial_values={1: 1e-4, 2: 5e-5},
        )
        skv_params = skv_strategy.generate_parameters(None)

        pcs_strategy = DirectParameterization(
            param_type=IWFMParameterType.PRECONSOLIDATION,
            name="pcs_uniform",
            initial_value=90.0,
        )
        pcs_params = pcs_strategy.generate_parameters(None)

        all_params = ske_params + skv_params + pcs_params

        # Step 4: Verify parameter bounds
        for p in ske_params:
            lb, ub = IWFMParameterType.ELASTIC_STORAGE.default_bounds
            assert p.lower_bound >= lb
            assert p.upper_bound <= ub

        for p in skv_params:
            lb, ub = IWFMParameterType.INELASTIC_STORAGE.default_bounds
            assert p.lower_bound >= lb
            assert p.upper_bound <= ub

        for p in pcs_params:
            lb, ub = IWFMParameterType.PRECONSOLIDATION.default_bounds
            assert p.lower_bound >= lb
            assert p.upper_bound <= ub

        # Step 5: Read back subsidence and verify
        reader = GroundwaterReader()
        read_sub = reader.read_subsidence(filepath)
        assert len(read_sub) == 3

        # Step 6: Verify PEST line generation
        for param in all_params:
            line = param.to_pest_line()
            assert len(line) > 0
            assert param.name in line

    def test_pest_control_file_format(self) -> None:
        """Verify PEST control file format for subsidence parameters."""
        # Create subsidence parameters
        ske_param = Parameter(
            name="ske_z1_l1",
            initial_value=1e-5,
            lower_bound=1e-6,
            upper_bound=1e-3,
            transform=ParameterTransform.LOG,
            group="ske",
            param_type=IWFMParameterType.ELASTIC_STORAGE,
        )

        skv_param = Parameter(
            name="skv_z1_l1",
            initial_value=1e-4,
            lower_bound=1e-5,
            upper_bound=1e-2,
            transform=ParameterTransform.LOG,
            group="skv",
            param_type=IWFMParameterType.INELASTIC_STORAGE,
        )

        pcs_param = Parameter(
            name="pcs_z1",
            initial_value=90.0,
            lower_bound=0.0,
            upper_bound=1000.0,
            transform=ParameterTransform.NONE,
            group="pcs",
            param_type=IWFMParameterType.PRECONSOLIDATION,
        )

        # Verify PEST line format
        ske_line = ske_param.to_pest_line()
        assert "ske_z1_l1" in ske_line
        assert "log" in ske_line
        assert "ske" in ske_line

        skv_line = skv_param.to_pest_line()
        assert "skv_z1_l1" in skv_line
        assert "log" in skv_line

        pcs_line = pcs_param.to_pest_line()
        assert "pcs_z1" in pcs_line
        assert "none" in pcs_line

        # Verify properties
        assert ske_param.partrans == "log"
        assert ske_param.parval1 == pytest.approx(1e-5)
        assert ske_param.parlbnd == pytest.approx(1e-6)
        assert ske_param.parubnd == pytest.approx(1e-3)

        assert pcs_param.partrans == "none"
        assert pcs_param.parval1 == pytest.approx(90.0)
