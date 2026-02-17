"""Integration tests for PEST++ subsidence parameterization.

Tests verify end-to-end subsidence parameterization workflows including
reading model data, generating PEST parameters, and writing entries.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pyiwfm.components.groundwater import (
    AppGW,
    Subsidence,
)
from pyiwfm.io.groundwater import (
    GroundwaterReader,
    GroundwaterWriter,
    GWFileConfig,
)
from pyiwfm.runner.pest_params import (
    DirectParameterization,
    IWFMParameterType,
    Parameter,
    ParameterGroup,
    ParameterTransform,
    PilotPointParameterization,
    ZoneParameterization,
)

pytestmark = pytest.mark.integration


# =============================================================================
# PEST Subsidence Integration Tests
# =============================================================================


class TestPESTSubsidenceIntegration:
    """Test full PEST subsidence parameterization workflow."""

    def test_zone_parameterization_workflow(self, tmp_path: Path) -> None:
        """Test zone-based subsidence parameterization end-to-end."""
        # Create model with subsidence
        gw = AppGW(n_nodes=100, n_layers=2, n_elements=50)
        for elem in range(1, 11):
            for layer in [1, 2]:
                gw.add_subsidence(
                    Subsidence(
                        element=elem,
                        layer=layer,
                        elastic_storage=1e-5,
                        inelastic_storage=1e-4,
                        preconsolidation_head=90.0,
                    )
                )

        # Write subsidence file
        config = GWFileConfig(output_dir=tmp_path)
        writer = GroundwaterWriter(config)
        filepath = writer.write_subsidence(gw)

        # Create zone parameterizations
        ske_strategy = ZoneParameterization(
            param_type=IWFMParameterType.ELASTIC_STORAGE,
            zones=[1, 2, 3],
            layer=1,
            initial_values={1: 8e-6, 2: 1.2e-5, 3: 2e-5},
        )
        skv_strategy = ZoneParameterization(
            param_type=IWFMParameterType.INELASTIC_STORAGE,
            zones=[1, 2, 3],
            layer=1,
        )
        pcs_strategy = DirectParameterization(
            param_type=IWFMParameterType.PRECONSOLIDATION,
            name="pcs_all",
            initial_value=90.0,
        )

        # Generate parameters
        ske_params = ske_strategy.generate_parameters(None)
        skv_params = skv_strategy.generate_parameters(None)
        pcs_params = pcs_strategy.generate_parameters(None)

        all_params = ske_params + skv_params + pcs_params

        # Verify all parameters have valid PEST lines
        for p in all_params:
            line = p.to_pest_line()
            assert len(line) > 0
            assert p.name in line

        # Read back subsidence data
        reader = GroundwaterReader()
        read_sub = reader.read_subsidence(filepath)
        assert len(read_sub) == 20  # 10 elements x 2 layers

        # Verify bounds are respected
        assert len(ske_params) == 3
        for p in ske_params:
            assert p.lower_bound >= 1e-6
            assert p.upper_bound <= 1e-3

    def test_pilot_point_parameterization_workflow(self, tmp_path: Path) -> None:
        """Test pilot-point subsidence parameterization end-to-end."""
        # Create model with subsidence
        gw = AppGW(n_nodes=50, n_layers=2, n_elements=25)
        for elem in range(1, 6):
            gw.add_subsidence(
                Subsidence(
                    element=elem,
                    layer=1,
                    elastic_storage=1e-5,
                    inelastic_storage=1e-4,
                    preconsolidation_head=90.0,
                )
            )

        # Write subsidence
        config = GWFileConfig(output_dir=tmp_path)
        writer = GroundwaterWriter(config)
        filepath = writer.write_subsidence(gw)

        # Create pilot point parameterization
        points = [(0, 0), (500, 0), (1000, 0), (0, 500), (500, 500), (1000, 500)]
        pp_strategy = PilotPointParameterization(
            param_type=IWFMParameterType.ELASTIC_STORAGE,
            points=points,
            layer=1,
            variogram={"type": "exponential", "a": 5000, "sill": 1.0},
        )

        pp_params = pp_strategy.generate_parameters(None)

        # Verify parameters
        assert len(pp_params) == 6
        assert all(p.location is not None for p in pp_params)
        assert all("ske" in p.name for p in pp_params)

        # Verify PEST line format
        for p in pp_params:
            line = p.to_pest_line()
            assert len(line) > 0

        # Read subsidence back
        reader = GroundwaterReader()
        read_sub = reader.read_subsidence(filepath)
        assert len(read_sub) == 5


# =============================================================================
# PEST Template File Tests
# =============================================================================


class TestSubsidenceTemplateFile:
    """Test PEST template generation for subsidence files."""

    def test_generate_pest_parameter_entries(self, tmp_path: Path) -> None:
        """Test generating PEST parameter entries for subsidence.

        Creates a complete set of parameters and verifies they can be
        formatted as PEST control file entries.
        """
        # Create parameter groups
        ske_group = ParameterGroup(name="ske", inctyp="relative", derinc=0.01)
        skv_group = ParameterGroup(name="skv", inctyp="relative", derinc=0.01)
        pcs_group = ParameterGroup(name="pcs", inctyp="relative", derinc=0.01)

        # Create parameters
        params = [
            Parameter(
                name="ske_z1_l1",
                initial_value=1e-5,
                lower_bound=1e-6,
                upper_bound=1e-3,
                transform=ParameterTransform.LOG,
                group="ske",
                param_type=IWFMParameterType.ELASTIC_STORAGE,
                layer=1,
                zone=1,
            ),
            Parameter(
                name="skv_z1_l1",
                initial_value=1e-4,
                lower_bound=1e-5,
                upper_bound=1e-2,
                transform=ParameterTransform.LOG,
                group="skv",
                param_type=IWFMParameterType.INELASTIC_STORAGE,
                layer=1,
                zone=1,
            ),
            Parameter(
                name="pcs_z1",
                initial_value=90.0,
                lower_bound=0.0,
                upper_bound=1000.0,
                transform=ParameterTransform.NONE,
                group="pcs",
                param_type=IWFMParameterType.PRECONSOLIDATION,
                zone=1,
            ),
        ]

        # Write PEST parameter data section
        pest_lines = []
        for group in [ske_group, skv_group, pcs_group]:
            pest_lines.append(group.to_pest_line())

        for param in params:
            pest_lines.append(param.to_pest_line())

        # Write to file for verification
        pest_file = tmp_path / "pest_params.txt"
        pest_file.write_text("\n".join(pest_lines))

        # Verify file
        content = pest_file.read_text()
        assert "ske" in content
        assert "skv" in content
        assert "pcs" in content
        assert "log" in content
        assert "none" in content
