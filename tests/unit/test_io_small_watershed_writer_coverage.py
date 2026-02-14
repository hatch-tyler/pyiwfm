"""Coverage tests for pyiwfm.io.small_watershed_writer module.

Targets uncovered lines: 55, 91, 95, 118-121, 145-146, 177, 232-238, 285-293.

Tests cover:
- SmallWatershedWriterConfig.swshed_dir with empty/falsy subdir
- SmallWatershedComponentWriter.format property
- write() delegates to write_all()
- write_all() with no component and write_defaults=False
- write_main() with empty watershed list
- _render_main() with baseflow GW nodes (is_baseflow=True)
- _render_main() with component budget_output_file and final_results_file
- write_small_watershed_component() convenience function (config=None and custom)
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from pyiwfm.io.small_watershed_writer import (
    SmallWatershedComponentWriter,
    SmallWatershedWriterConfig,
    write_small_watershed_component,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_engine():
    """Create a mock TemplateEngine that returns predictable content."""
    engine = MagicMock()
    engine.render_template.return_value = "C  MOCK SW HEADER\n"
    engine.render_string.return_value = "C  MOCK STRING\n"
    return engine


@pytest.fixture
def bare_model():
    """Create a model with small_watersheds=None."""
    model = MagicMock()
    model.small_watersheds = None
    return model


@pytest.fixture
def model_with_empty_sw():
    """Create a model with an empty small watersheds component (0 watersheds)."""
    model = MagicMock()
    sw = MagicMock()
    sw.n_watersheds = 0
    sw.iter_watersheds.return_value = iter([])
    sw.budget_output_file = ""
    sw.final_results_file = ""
    sw.area_factor = 1.0
    sw.flow_factor = 1.0
    sw.flow_time_unit = "1DAY"
    sw.rz_solver_tolerance = 1e-8
    sw.rz_max_iterations = 2000
    sw.rz_length_factor = 1.0
    sw.rz_cn_factor = 1.0
    sw.rz_k_factor = 1.0
    sw.rz_k_time_unit = "1DAY"
    sw.aq_gw_factor = 1.0
    sw.aq_time_factor = 1.0
    sw.aq_time_unit = "1DAY"
    sw.ic_factor = 1.0
    model.small_watersheds = sw
    return model


@pytest.fixture
def model_with_watersheds():
    """Create a model with one watershed containing a baseflow GW node."""
    model = MagicMock()
    sw = MagicMock()
    sw.n_watersheds = 1

    # GW node with is_baseflow=True to hit line 177
    gn_baseflow = SimpleNamespace(
        gw_node_id=10,
        max_perc_rate=0.5,
        is_baseflow=True,
        layer=3,
    )
    # GW node without baseflow
    gn_normal = SimpleNamespace(
        gw_node_id=20,
        max_perc_rate=0.8,
        is_baseflow=False,
        layer=1,
    )

    ws = SimpleNamespace(
        id=1,
        area=5000.0,
        dest_stream_node=42,
        n_gw_nodes=2,
        gw_nodes=[gn_baseflow, gn_normal],
        precip_col=1,
        precip_factor=1.0,
        et_col=2,
        wilting_point=0.1,
        field_capacity=0.3,
        total_porosity=0.4,
        lambda_param=0.5,
        kunsat_method=1,
        root_depth=2.0,
        hydraulic_cond=0.01,
        curve_number=75.0,
        gw_threshold=10.0,
        max_gw_storage=100.0,
        surface_flow_coeff=0.5,
        baseflow_coeff=0.3,
        initial_soil_moisture=0.25,
        initial_gw_storage=5.0,
    )
    sw.iter_watersheds.return_value = iter([ws])

    # Factors
    sw.rz_length_factor = 1.0
    sw.rz_k_factor = 1.0
    sw.rz_cn_factor = 1.0
    sw.aq_gw_factor = 1.0
    sw.aq_time_factor = 1.0

    # Component output files (truthy to hit lines 234, 236)
    sw.budget_output_file = "../Results/CustomBud.hdf"
    sw.final_results_file = "../Results/CustomFinal.out"

    # IC factor
    sw.ic_factor = 1.0

    # Other factors for the template context
    sw.area_factor = 43560.0
    sw.flow_factor = 1.0
    sw.flow_time_unit = "1DAY"
    sw.rz_solver_tolerance = 1e-8
    sw.rz_max_iterations = 2000
    sw.rz_k_time_unit = "1DAY"
    sw.aq_time_unit = "1DAY"

    model.small_watersheds = sw
    return model


# =============================================================================
# SmallWatershedWriterConfig Tests
# =============================================================================


class TestSmallWatershedWriterConfigProperties:
    """Tests for SmallWatershedWriterConfig dataclass properties."""

    def test_swshed_dir_default(self, tmp_path: Path) -> None:
        """Default config returns output_dir / 'SmallWatershed'."""
        config = SmallWatershedWriterConfig(output_dir=tmp_path)
        assert config.swshed_dir == tmp_path / "SmallWatershed"

    def test_swshed_dir_empty_subdir(self, tmp_path: Path) -> None:
        """When swshed_subdir is empty (falsy), swshed_dir returns output_dir."""
        config = SmallWatershedWriterConfig(output_dir=tmp_path, swshed_subdir="")
        assert config.swshed_dir == tmp_path


# =============================================================================
# format property
# =============================================================================


class TestSmallWatershedFormatProperty:
    """Tests for SmallWatershedComponentWriter.format property."""

    def test_format_property(
        self, tmp_path: Path, bare_model: MagicMock, mock_engine: MagicMock
    ) -> None:
        """format property returns 'iwfm_small_watershed'."""
        config = SmallWatershedWriterConfig(output_dir=tmp_path)
        writer = SmallWatershedComponentWriter(
            bare_model, config, template_engine=mock_engine
        )
        assert writer.format == "iwfm_small_watershed"


# =============================================================================
# write() delegation
# =============================================================================


class TestSmallWatershedWriteDelegation:
    """Tests for write() -> write_all() delegation."""

    def test_write_delegates_to_write_all(
        self, tmp_path: Path, bare_model: MagicMock, mock_engine: MagicMock
    ) -> None:
        """write() calls write_all() internally."""
        config = SmallWatershedWriterConfig(output_dir=tmp_path)
        writer = SmallWatershedComponentWriter(
            bare_model, config, template_engine=mock_engine
        )
        with patch.object(writer, "write_all") as mock_write_all:
            writer.write()
            mock_write_all.assert_called_once()


# =============================================================================
# write_all() Tests
# =============================================================================


class TestSmallWatershedWriteAll:
    """Tests for SmallWatershedComponentWriter.write_all()."""

    def test_write_all_no_component_no_defaults(
        self, tmp_path: Path, bare_model: MagicMock, mock_engine: MagicMock
    ) -> None:
        """write_all(write_defaults=False) returns empty dict when no component."""
        config = SmallWatershedWriterConfig(output_dir=tmp_path)
        writer = SmallWatershedComponentWriter(
            bare_model, config, template_engine=mock_engine
        )
        results = writer.write_all(write_defaults=False)
        assert results == {}

    def test_write_all_no_component_with_defaults(
        self, tmp_path: Path, bare_model: MagicMock, mock_engine: MagicMock
    ) -> None:
        """write_all(write_defaults=True) writes main file even without component."""
        config = SmallWatershedWriterConfig(output_dir=tmp_path)
        writer = SmallWatershedComponentWriter(
            bare_model, config, template_engine=mock_engine
        )
        results = writer.write_all(write_defaults=True)
        assert "main" in results
        assert results["main"].exists()


# =============================================================================
# write_main() Tests
# =============================================================================


class TestSmallWatershedWriteMain:
    """Tests for SmallWatershedComponentWriter.write_main()."""

    def test_write_main_no_watersheds(
        self, tmp_path: Path, model_with_empty_sw: MagicMock, mock_engine: MagicMock
    ) -> None:
        """write_main() with n_watersheds=0 passes empty ws_list to _render_main."""
        config = SmallWatershedWriterConfig(output_dir=tmp_path)
        writer = SmallWatershedComponentWriter(
            model_with_empty_sw, config, template_engine=mock_engine
        )
        path = writer.write_main()
        assert path.exists()
        # Verify _render_main was called via engine with n_watersheds=0
        render_call = mock_engine.render_template.call_args
        assert render_call is not None
        assert render_call[1]["n_watersheds"] == 0
        assert render_call[1]["watersheds"] == []

    def test_write_main_none_sw_empty_list(
        self, tmp_path: Path, bare_model: MagicMock, mock_engine: MagicMock
    ) -> None:
        """write_main() with sw=None produces ws_list=[] and n_watersheds=0."""
        config = SmallWatershedWriterConfig(output_dir=tmp_path)
        writer = SmallWatershedComponentWriter(
            bare_model, config, template_engine=mock_engine
        )
        path = writer.write_main()
        assert path.exists()
        render_call = mock_engine.render_template.call_args
        assert render_call[1]["n_watersheds"] == 0
        assert render_call[1]["watersheds"] == []


# =============================================================================
# _render_main() Tests - baseflow nodes and component output files
# =============================================================================


class TestSmallWatershedRenderMain:
    """Tests for _render_main() template context building."""

    def test_render_main_baseflow_nodes(
        self,
        tmp_path: Path,
        model_with_watersheds: MagicMock,
        mock_engine: MagicMock,
    ) -> None:
        """Baseflow GW nodes (is_baseflow=True) set perc_rate_raw = -float(layer)."""
        config = SmallWatershedWriterConfig(output_dir=tmp_path)
        writer = SmallWatershedComponentWriter(
            model_with_watersheds, config, template_engine=mock_engine
        )
        writer.write_main()

        render_call = mock_engine.render_template.call_args
        ws_data = render_call[1]["watersheds"]
        assert len(ws_data) == 1

        gw_nodes = ws_data[0]["gw_nodes"]
        assert len(gw_nodes) == 2

        # First node is baseflow: perc_rate_raw should be -float(layer) = -3.0
        bf_node = gw_nodes[0]
        assert bf_node["is_baseflow"] is True
        assert bf_node["perc_rate_raw"] == -3.0

        # Second node is normal: perc_rate_raw should be max_perc_rate = 0.8
        normal_node = gw_nodes[1]
        assert normal_node["is_baseflow"] is False
        assert normal_node["perc_rate_raw"] == 0.8

    def test_render_main_component_output_files(
        self,
        tmp_path: Path,
        model_with_watersheds: MagicMock,
        mock_engine: MagicMock,
    ) -> None:
        """Component with truthy budget_output_file and final_results_file overrides config."""
        config = SmallWatershedWriterConfig(output_dir=tmp_path)
        writer = SmallWatershedComponentWriter(
            model_with_watersheds, config, template_engine=mock_engine
        )
        writer.write_main()

        render_call = mock_engine.render_template.call_args
        # The component's custom files should override the config defaults
        assert render_call[1]["budget_file"] == "../Results/CustomBud.hdf"
        assert render_call[1]["final_results_file"] == "../Results/CustomFinal.out"

    def test_render_main_no_component_output_files(
        self,
        tmp_path: Path,
        model_with_empty_sw: MagicMock,
        mock_engine: MagicMock,
    ) -> None:
        """When component output files are empty, config defaults are used."""
        config = SmallWatershedWriterConfig(
            output_dir=tmp_path,
            budget_file="../Results/DefaultBud.hdf",
            final_results_file="../Results/DefaultFinal.out",
        )
        writer = SmallWatershedComponentWriter(
            model_with_empty_sw, config, template_engine=mock_engine
        )
        writer.write_main()

        render_call = mock_engine.render_template.call_args
        assert render_call[1]["budget_file"] == "../Results/DefaultBud.hdf"
        assert render_call[1]["final_results_file"] == "../Results/DefaultFinal.out"


# =============================================================================
# write_small_watershed_component() convenience function
# =============================================================================


class TestWriteSmallWatershedConvenience:
    """Tests for write_small_watershed_component() module-level function."""

    def test_write_sw_convenience_no_config(
        self, tmp_path: Path, bare_model: MagicMock, mock_engine: MagicMock
    ) -> None:
        """write_small_watershed_component() with config=None creates default config."""
        with patch(
            "pyiwfm.io.small_watershed_writer.TemplateEngine",
            return_value=mock_engine,
        ):
            results = write_small_watershed_component(bare_model, tmp_path)
        assert "main" in results

    def test_write_sw_convenience_with_config(
        self, tmp_path: Path, bare_model: MagicMock, mock_engine: MagicMock
    ) -> None:
        """write_small_watershed_component() with custom config updates output_dir."""
        custom_config = SmallWatershedWriterConfig(
            output_dir=Path("/dummy"),
            version="5.0",
        )
        with patch(
            "pyiwfm.io.small_watershed_writer.TemplateEngine",
            return_value=mock_engine,
        ):
            results = write_small_watershed_component(
                bare_model, tmp_path, config=custom_config
            )
        # output_dir should be overwritten to tmp_path
        assert custom_config.output_dir == tmp_path
        assert "main" in results
