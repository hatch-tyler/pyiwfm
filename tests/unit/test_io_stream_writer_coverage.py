"""Coverage tests for pyiwfm.io.stream_writer module.

Targets uncovered branches and edge cases including:
- StreamWriterConfig default and custom properties
- write_all() with write_defaults=False and various component states
- write_main() file creation and version-dependent output (v4.0 vs v5.0)
- write_diver_specs() and write_bypass_specs() file creation
- _render_bed_params_section() with v5.0 column format
- _render_cross_section() and _render_initial_conditions() for v5.0
- _render_evaporation() with and without evap area file
- Stream nodes from reaches fallback and budget_node_ids fallback
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from pyiwfm.io.stream_writer import (
    StreamComponentWriter,
    StreamWriterConfig,
    write_stream_component,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_engine():
    """Create a mock TemplateEngine that returns predictable content."""
    engine = MagicMock()
    engine.render_template.return_value = "C  MOCK HEADER\n"
    engine.render_string.return_value = "C  MOCK STRING\n"
    return engine


@pytest.fixture
def bare_model():
    """Create a model with streams=None."""
    model = MagicMock()
    model.streams = None
    model.source_files = {}
    return model


@pytest.fixture
def model_with_empty_streams():
    """Create a model with an empty streams component."""
    model = MagicMock()
    streams = MagicMock()
    streams.nodes = {}
    streams.reaches = {}
    streams.diversions = {}
    streams.bypasses = {}
    streams.inflows = []
    streams.budget_node_ids = []
    streams.budget_node_count = 0
    streams.evap_node_specs = []
    streams.evap_area_file = ""
    model.streams = streams
    model.source_files = {}
    return model


@pytest.fixture
def model_with_full_streams():
    """Create a model with streams, diversions, and bypasses."""
    model = MagicMock()
    streams = MagicMock()

    node1 = SimpleNamespace(
        id=1,
        wetted_perimeter=200.0,
        conductivity=15.0,
        bed_thickness=2.0,
        gw_node=5,
        cross_section=None,
        initial_condition=0.0,
    )
    node2 = SimpleNamespace(
        id=2,
        wetted_perimeter=150.0,
        conductivity=10.0,
        bed_thickness=1.0,
        gw_node=8,
        cross_section=None,
        initial_condition=0.0,
    )
    streams.nodes = {1: node1, 2: node2}
    streams.reaches = {}
    streams.budget_node_count = 0
    streams.budget_node_ids = []
    streams.evap_node_specs = []
    streams.evap_area_file = ""

    # Diversions
    div = SimpleNamespace(
        id=1,
        source_node=10,
        max_div_column=1,
        max_div_fraction=1.0,
        recoverable_loss_column=0,
        recoverable_loss_fraction=0.0,
        non_recoverable_loss_column=0,
        non_recoverable_loss_fraction=0.0,
        delivery_dest_type=0,
        delivery_dest_id=5,
        delivery_column=0,
        delivery_fraction=1.0,
        irrigation_fraction_column=0,
        adjustment_column=0,
        name="Div1",
    )
    streams.diversions = {1: div}
    streams.diversion_has_spills = False
    streams.diversion_element_groups = []
    streams.diversion_recharge_zones = []

    # Bypasses
    bp = SimpleNamespace(
        id=1,
        source_node=15,
        destination_node=20,
        dest_type=0,
        flow_factor=1.0,
        flow_time_unit="1DAY",
        spill_factor=1.0,
        spill_time_unit="1DAY",
        diversion_column=1,
        rating_table_flows=[],
        rating_table_spills=[],
        recoverable_loss_fraction=0.0,
        non_recoverable_loss_fraction=0.0,
        name="Bypass1",
        seepage_locations=[],
    )
    streams.bypasses = {1: bp}

    streams.inflows = [MagicMock()]
    model.streams = streams
    model.source_files = {}
    return model


# =============================================================================
# StreamWriterConfig Tests
# =============================================================================


class TestStreamWriterConfigProperties:
    """Tests for StreamWriterConfig dataclass properties."""

    def test_stream_dir_default(self, tmp_path: Path) -> None:
        """Test stream_dir property with default subdir."""
        config = StreamWriterConfig(output_dir=tmp_path)
        assert config.stream_dir == tmp_path / "Stream"

    def test_main_path_default(self, tmp_path: Path) -> None:
        """Test main_path property with default file name."""
        config = StreamWriterConfig(output_dir=tmp_path)
        assert config.main_path == tmp_path / "Stream" / "Stream_MAIN.dat"

    def test_stream_dir_custom_subdir(self, tmp_path: Path) -> None:
        """Test stream_dir property with custom subdir."""
        config = StreamWriterConfig(output_dir=tmp_path, stream_subdir="Streams")
        assert config.stream_dir == tmp_path / "Streams"

    def test_main_path_custom_file(self, tmp_path: Path) -> None:
        """Test main_path with custom main_file and subdir."""
        config = StreamWriterConfig(output_dir=tmp_path, stream_subdir="S", main_file="ctrl.dat")
        assert config.main_path == tmp_path / "S" / "ctrl.dat"

    def test_v50_fields_defaults(self, tmp_path: Path) -> None:
        """Test v5.0 specific fields have correct defaults."""
        config = StreamWriterConfig(output_dir=tmp_path)
        assert config.final_flow_file == ""
        assert config.roughness_factor == 1.0
        assert config.cross_section_length_factor == 1.0
        assert config.ic_type == 0
        assert config.ic_time_unit == ""
        assert config.ic_factor == 1.0


# =============================================================================
# write_all() Tests
# =============================================================================


class TestStreamWriteAll:
    """Tests for StreamComponentWriter.write_all()."""

    def test_write_all_no_streams_write_defaults_false(
        self, tmp_path: Path, bare_model: MagicMock, mock_engine: MagicMock
    ) -> None:
        """write_all(write_defaults=False) returns empty dict when no streams."""
        config = StreamWriterConfig(output_dir=tmp_path)
        writer = StreamComponentWriter(bare_model, config, template_engine=mock_engine)
        results = writer.write_all(write_defaults=False)
        assert results == {}

    def test_write_all_no_streams_write_defaults_true(
        self, tmp_path: Path, bare_model: MagicMock, mock_engine: MagicMock
    ) -> None:
        """write_all(write_defaults=True) writes main file even without streams."""
        config = StreamWriterConfig(output_dir=tmp_path)
        writer = StreamComponentWriter(bare_model, config, template_engine=mock_engine)
        results = writer.write_all(write_defaults=True)
        assert "main" in results
        assert results["main"].exists()

    def test_write_all_with_diversions(
        self, tmp_path: Path, model_with_full_streams: MagicMock, mock_engine: MagicMock
    ) -> None:
        """write_all() includes diver_specs when diversions exist."""
        config = StreamWriterConfig(output_dir=tmp_path)
        writer = StreamComponentWriter(model_with_full_streams, config, template_engine=mock_engine)
        results = writer.write_all()
        assert "main" in results
        assert "diver_specs" in results
        assert "bypass_specs" in results

    def test_write_all_with_empty_streams(
        self, tmp_path: Path, model_with_empty_streams: MagicMock, mock_engine: MagicMock
    ) -> None:
        """write_all() writes main but skips diver/bypass for empty lists."""
        config = StreamWriterConfig(output_dir=tmp_path)
        writer = StreamComponentWriter(
            model_with_empty_streams, config, template_engine=mock_engine
        )
        results = writer.write_all()
        assert "main" in results
        assert "diver_specs" not in results
        assert "bypass_specs" not in results

    def test_write_method_delegates_to_write_all(
        self, tmp_path: Path, bare_model: MagicMock, mock_engine: MagicMock
    ) -> None:
        """write() delegates to write_all()."""
        config = StreamWriterConfig(output_dir=tmp_path)
        writer = StreamComponentWriter(bare_model, config, template_engine=mock_engine)
        # write() should not raise
        writer.write()
        assert config.main_path.exists()


# =============================================================================
# write_main() Tests
# =============================================================================


class TestStreamWriteMain:
    """Tests for StreamComponentWriter.write_main()."""

    def test_write_main_creates_file(
        self, tmp_path: Path, model_with_empty_streams: MagicMock, mock_engine: MagicMock
    ) -> None:
        """write_main() creates the main output file."""
        config = StreamWriterConfig(output_dir=tmp_path)
        writer = StreamComponentWriter(
            model_with_empty_streams, config, template_engine=mock_engine
        )
        path = writer.write_main()
        assert path.exists()
        assert path == config.main_path

    def test_write_main_v40_bed_params(
        self, tmp_path: Path, model_with_full_streams: MagicMock, mock_engine: MagicMock
    ) -> None:
        """v4.0 bed params include WETPR and IRGW columns."""
        config = StreamWriterConfig(output_dir=tmp_path, version="4.0")
        writer = StreamComponentWriter(model_with_full_streams, config, template_engine=mock_engine)
        path = writer.write_main()
        content = path.read_text()
        assert "WETPR" in content
        assert "IRGW" in content

    def test_write_main_v50_bed_params(
        self, tmp_path: Path, model_with_full_streams: MagicMock, mock_engine: MagicMock
    ) -> None:
        """v5.0 bed params omit WETPR/IRGW and include cross-section data."""
        config = StreamWriterConfig(output_dir=tmp_path, version="5.0")
        writer = StreamComponentWriter(model_with_full_streams, config, template_engine=mock_engine)
        path = writer.write_main()
        content = path.read_text()
        # v5.0 format has "IR    CSTRM   DSTRM" not "WETPR"
        assert "Cross-Section Data" in content
        assert "Initial Conditions" in content

    def test_write_main_nodes_from_reaches_fallback(
        self, tmp_path: Path, mock_engine: MagicMock
    ) -> None:
        """When nodes dict is empty, node IDs from reaches are used."""
        model = MagicMock()
        streams = MagicMock()
        streams.nodes = {}
        streams.reaches = {
            1: SimpleNamespace(nodes=[3, 5, 7]),
            2: SimpleNamespace(nodes=[7, 9]),
        }
        streams.budget_node_ids = []
        streams.budget_node_count = 0
        streams.diversions = {}
        streams.bypasses = {}
        streams.inflows = []
        streams.evap_node_specs = []
        streams.evap_area_file = ""
        model.streams = streams
        model.source_files = {}

        config = StreamWriterConfig(output_dir=tmp_path)
        writer = StreamComponentWriter(model, config, template_engine=mock_engine)
        path = writer.write_main()
        content = path.read_text()
        # Should see nodes 3, 5, 7, 9 referenced in hydrograph section
        assert "3" in content
        assert "9" in content

    def test_write_main_nodes_from_budget_ids_fallback(
        self, tmp_path: Path, mock_engine: MagicMock
    ) -> None:
        """When nodes and reaches are empty, budget_node_ids are used."""
        model = MagicMock()
        streams = MagicMock()
        streams.nodes = {}
        streams.reaches = {}
        streams.budget_node_ids = [10, 20, 30]
        streams.budget_node_count = 3
        streams.diversions = {}
        streams.bypasses = {}
        streams.inflows = []
        streams.evap_node_specs = []
        streams.evap_area_file = ""
        model.streams = streams
        model.source_files = {}

        config = StreamWriterConfig(output_dir=tmp_path)
        writer = StreamComponentWriter(model, config, template_engine=mock_engine)
        path = writer.write_main()
        content = path.read_text()
        assert "10" in content
        assert "30" in content


# =============================================================================
# write_diver_specs() Tests
# =============================================================================


class TestStreamWriteDiverSpecs:
    """Tests for StreamComponentWriter.write_diver_specs()."""

    def test_write_diver_specs_creates_file(
        self, tmp_path: Path, model_with_full_streams: MagicMock, mock_engine: MagicMock
    ) -> None:
        """write_diver_specs() creates the diversion specification file."""
        config = StreamWriterConfig(output_dir=tmp_path)
        writer = StreamComponentWriter(model_with_full_streams, config, template_engine=mock_engine)
        path = writer.write_diver_specs()
        assert path.exists()
        content = path.read_text()
        assert "NDIVER" in content
        assert "Div1" in content


# =============================================================================
# write_bypass_specs() Tests
# =============================================================================


class TestStreamWriteBypassSpecs:
    """Tests for StreamComponentWriter.write_bypass_specs()."""

    def test_write_bypass_specs_creates_file(
        self, tmp_path: Path, model_with_full_streams: MagicMock, mock_engine: MagicMock
    ) -> None:
        """write_bypass_specs() creates the bypass specification file."""
        config = StreamWriterConfig(output_dir=tmp_path)
        writer = StreamComponentWriter(model_with_full_streams, config, template_engine=mock_engine)
        path = writer.write_bypass_specs()
        assert path.exists()
        content = path.read_text()
        assert "NBYPASS" in content
        assert "Bypass1" in content

    def test_write_bypass_with_rating_table(self, tmp_path: Path, mock_engine: MagicMock) -> None:
        """Bypasses with inline rating tables (IDIVC < 0) write table rows."""
        model = MagicMock()
        bp = SimpleNamespace(
            id=1,
            source_node=15,
            destination_node=20,
            dest_type=0,
            flow_factor=1.0,
            flow_time_unit="1DAY",
            spill_factor=1.0,
            spill_time_unit="1DAY",
            diversion_column=0,
            rating_table_flows=[100.0, 200.0, 300.0],
            rating_table_spills=[0.0, 50.0, 150.0],
            recoverable_loss_fraction=0.05,
            non_recoverable_loss_fraction=0.02,
            name="RatedBypass",
            seepage_locations=[],
        )
        streams = MagicMock()
        streams.bypasses = {1: bp}
        model.streams = streams

        config = StreamWriterConfig(output_dir=tmp_path)
        writer = StreamComponentWriter(model, config, template_engine=mock_engine)
        path = writer.write_bypass_specs()
        content = path.read_text()
        assert "RatedBypass" in content
        # Inline rating table rows should be present
        assert "100" in content
        assert "200" in content
        assert "300" in content


# =============================================================================
# Format property and convenience function tests
# =============================================================================


class TestStreamWriterMisc:
    """Miscellaneous tests for StreamComponentWriter."""

    def test_format_property(
        self, tmp_path: Path, bare_model: MagicMock, mock_engine: MagicMock
    ) -> None:
        """format property returns 'iwfm_stream'."""
        config = StreamWriterConfig(output_dir=tmp_path)
        writer = StreamComponentWriter(bare_model, config, template_engine=mock_engine)
        assert writer.format == "iwfm_stream"

    def test_write_stream_component_convenience(
        self, tmp_path: Path, bare_model: MagicMock, mock_engine: MagicMock
    ) -> None:
        """write_stream_component() convenience function works."""
        with patch("pyiwfm.io.stream_writer.TemplateEngine", return_value=mock_engine):
            results = write_stream_component(bare_model, tmp_path)
        assert "main" in results

    def test_write_stream_component_with_config(
        self, tmp_path: Path, bare_model: MagicMock, mock_engine: MagicMock
    ) -> None:
        """write_stream_component() uses provided config."""
        config = StreamWriterConfig(output_dir=tmp_path, version="5.0")
        with patch("pyiwfm.io.stream_writer.TemplateEngine", return_value=mock_engine):
            results = write_stream_component(bare_model, tmp_path, config=config)
        assert "main" in results
        # Config output_dir is updated to match the passed output_dir
        assert config.output_dir == tmp_path


# =============================================================================
# Additional Coverage Tests - render methods and v5.0 branches
# =============================================================================


class TestStreamWriteMainRenderContext:
    """Tests for write_main() template rendering context (lines 400-463)."""

    def test_write_main_renders_with_correct_template_v40(
        self, tmp_path: Path, model_with_full_streams: MagicMock, mock_engine: MagicMock
    ) -> None:
        """write_main() renders v4.0 template with inflow/diversion/bypass paths."""
        config = StreamWriterConfig(output_dir=tmp_path, version="4.0")
        writer = StreamComponentWriter(model_with_full_streams, config, template_engine=mock_engine)
        path = writer.write_main()
        content = path.read_text()
        # Should contain bed params header for v4.0 (with WETPR)
        assert "WETPR" in content
        assert "IRGW" in content

    def test_write_main_renders_with_correct_template_v50(
        self, tmp_path: Path, model_with_full_streams: MagicMock, mock_engine: MagicMock
    ) -> None:
        """write_main() renders v5.0 template with v5.0-specific sections."""
        config = StreamWriterConfig(output_dir=tmp_path, version="5.0")
        writer = StreamComponentWriter(model_with_full_streams, config, template_engine=mock_engine)
        # The mock engine returns generic header text via render_template
        mock_engine.render_template.return_value = "C  MOCK HEADER v5.0\n"
        path = writer.write_main()
        content = path.read_text()
        assert "Cross-Section Data" in content
        assert "Initial Conditions" in content
        # v5.0 bed params omit WETPR
        assert "IR    CSTRM   DSTRM" in content


class TestStreamWriteInflowTs:
    """Tests for write_stream_inflow_ts() (lines 509-570)."""

    def test_write_inflow_creates_file(
        self, tmp_path: Path, model_with_full_streams: MagicMock, mock_engine: MagicMock
    ) -> None:
        """write_stream_inflow_ts() creates inflow file via IWFMTimeSeriesDataWriter."""
        config = StreamWriterConfig(output_dir=tmp_path)
        writer = StreamComponentWriter(model_with_full_streams, config, template_engine=mock_engine)

        mock_ts_writer = MagicMock()
        expected_path = config.stream_dir / config.inflow_file
        mock_ts_writer.write.return_value = expected_path

        with patch(
            "pyiwfm.io.timeseries_writer.IWFMTimeSeriesDataWriter",
            return_value=mock_ts_writer,
        ):
            path = writer.write_stream_inflow_ts()

        mock_ts_writer.write.assert_called_once()
        assert path == expected_path


class TestStreamWriteDiverSpecsDetailed:
    """Detailed tests for write_diver_specs() (lines 621-637, 667)."""

    def test_diver_specs_with_spills(self, tmp_path: Path, mock_engine: MagicMock) -> None:
        """write_diver_specs() writes 16-column format with spill columns."""
        model = MagicMock()
        streams = MagicMock()
        div = SimpleNamespace(
            id=1,
            source_node=10,
            max_div_column=1,
            max_div_fraction=1.0,
            recoverable_loss_column=0,
            recoverable_loss_fraction=0.0,
            non_recoverable_loss_column=0,
            non_recoverable_loss_fraction=0.0,
            spill_column=2,
            spill_fraction=0.5,
            delivery_dest_type=0,
            delivery_dest_id=5,
            delivery_column=0,
            delivery_fraction=1.0,
            irrigation_fraction_column=0,
            adjustment_column=0,
            name="DivSpill",
        )
        streams.diversions = {1: div}
        streams.diversion_has_spills = True
        streams.diversion_element_groups = []
        streams.diversion_recharge_zones = []
        streams.diversion_spill_zones = []
        model.streams = streams

        config = StreamWriterConfig(output_dir=tmp_path)
        writer = StreamComponentWriter(model, config, template_engine=mock_engine)
        path = writer.write_diver_specs()
        content = path.read_text()
        assert "ICOLSP" in content
        assert "DivSpill" in content

    def test_diver_specs_with_element_groups(self, tmp_path: Path, mock_engine: MagicMock) -> None:
        """write_diver_specs() writes element groups section."""
        model = MagicMock()
        streams = MagicMock()
        div = SimpleNamespace(
            id=1,
            source_node=10,
            max_div_column=1,
            max_div_fraction=1.0,
            recoverable_loss_column=0,
            recoverable_loss_fraction=0.0,
            non_recoverable_loss_column=0,
            non_recoverable_loss_fraction=0.0,
            delivery_dest_type=0,
            delivery_dest_id=5,
            delivery_column=0,
            delivery_fraction=1.0,
            irrigation_fraction_column=0,
            adjustment_column=0,
            name="DivEG",
        )
        streams.diversions = {1: div}
        streams.diversion_has_spills = False

        # Element group with multiple elements
        eg = SimpleNamespace(id=1, elements=[100, 101, 102])
        streams.diversion_element_groups = [eg]

        # Recharge zone
        rz = SimpleNamespace(
            diversion_id=1, n_zones=2, zone_ids=[10, 20], zone_fractions=[0.6, 0.4]
        )
        streams.diversion_recharge_zones = [rz]
        model.streams = streams

        config = StreamWriterConfig(output_dir=tmp_path)
        writer = StreamComponentWriter(model, config, template_engine=mock_engine)
        path = writer.write_diver_specs()
        content = path.read_text()
        assert "NGRP" in content
        assert "100" in content
        assert "101" in content
        assert "102" in content
        # Recharge zone fractions
        assert "0.6000" in content
        assert "0.4000" in content

    def test_diver_specs_with_spill_zones(self, tmp_path: Path, mock_engine: MagicMock) -> None:
        """write_diver_specs() writes spill zones section when has_spills is True."""
        model = MagicMock()
        streams = MagicMock()
        div = SimpleNamespace(
            id=1,
            source_node=10,
            max_div_column=1,
            max_div_fraction=1.0,
            recoverable_loss_column=0,
            recoverable_loss_fraction=0.0,
            non_recoverable_loss_column=0,
            non_recoverable_loss_fraction=0.0,
            spill_column=2,
            spill_fraction=0.5,
            delivery_dest_type=0,
            delivery_dest_id=5,
            delivery_column=0,
            delivery_fraction=1.0,
            irrigation_fraction_column=0,
            adjustment_column=0,
            name="SpillDiv",
        )
        streams.diversions = {1: div}
        streams.diversion_has_spills = True
        streams.diversion_element_groups = []
        streams.diversion_recharge_zones = []

        # Spill zone
        sz = SimpleNamespace(diversion_id=1, n_zones=1, zone_ids=[50], zone_fractions=[1.0])
        streams.diversion_spill_zones = [sz]
        model.streams = streams

        config = StreamWriterConfig(output_dir=tmp_path)
        writer = StreamComponentWriter(model, config, template_engine=mock_engine)
        path = writer.write_diver_specs()
        content = path.read_text()
        assert "Spill zones" in content
        assert "50" in content


class TestStreamWriteBypassSpecsDetailed:
    """Detailed tests for write_bypass_specs() (lines 782-831)."""

    def test_bypass_with_seepage_locations(self, tmp_path: Path, mock_engine: MagicMock) -> None:
        """write_bypass_specs() writes seepage location data."""
        model = MagicMock()
        sl = SimpleNamespace(
            n_elements=2,
            element_ids=[10, 20],
            element_fractions=[0.7, 0.3],
        )
        bp = SimpleNamespace(
            id=1,
            source_node=15,
            destination_node=20,
            dest_type=0,
            flow_factor=1.0,
            flow_time_unit="1DAY",
            spill_factor=1.0,
            spill_time_unit="1DAY",
            diversion_column=1,
            rating_table_flows=[],
            rating_table_spills=[],
            recoverable_loss_fraction=0.0,
            non_recoverable_loss_fraction=0.0,
            name="SeepBypass",
            seepage_locations=[sl],
        )
        streams = MagicMock()
        streams.bypasses = {1: bp}
        model.streams = streams

        config = StreamWriterConfig(output_dir=tmp_path)
        writer = StreamComponentWriter(model, config, template_engine=mock_engine)
        path = writer.write_bypass_specs()
        content = path.read_text()
        assert "SeepBypass" in content
        assert "0.7000" in content
        assert "0.3000" in content

    def test_bypass_empty_seepage_location(self, tmp_path: Path, mock_engine: MagicMock) -> None:
        """write_bypass_specs() handles seepage location with 0 elements."""
        model = MagicMock()
        sl = SimpleNamespace(
            n_elements=0,
            element_ids=[],
            element_fractions=[],
        )
        bp = SimpleNamespace(
            id=1,
            source_node=15,
            destination_node=20,
            dest_type=0,
            flow_factor=1.0,
            flow_time_unit="1DAY",
            spill_factor=1.0,
            spill_time_unit="1DAY",
            diversion_column=1,
            rating_table_flows=[],
            rating_table_spills=[],
            recoverable_loss_fraction=0.0,
            non_recoverable_loss_fraction=0.0,
            name="EmptySeep",
            seepage_locations=[sl],
        )
        streams = MagicMock()
        streams.bypasses = {1: bp}
        model.streams = streams

        config = StreamWriterConfig(output_dir=tmp_path)
        writer = StreamComponentWriter(model, config, template_engine=mock_engine)
        path = writer.write_bypass_specs()
        content = path.read_text()
        # Should write default "0" for empty seepage
        assert "EmptySeep" in content


class TestStreamBedParamsFromModelData:
    """Tests for _render_bed_params_section using model data (lines 400-463)."""

    def test_bed_params_uses_model_factors(self, tmp_path: Path, mock_engine: MagicMock) -> None:
        """_render_bed_params_section uses model conductivity/time/length factors."""
        model = MagicMock()
        streams = MagicMock()
        streams.nodes = {}
        streams.reaches = {}
        streams.budget_node_ids = [1, 2]
        streams.budget_node_count = 2
        streams.diversions = {}
        streams.bypasses = {}
        streams.inflows = []
        streams.evap_node_specs = []
        streams.evap_area_file = ""
        streams.conductivity_factor = 2.5
        streams.conductivity_time_unit = "1MON"
        streams.length_factor = 3.28
        model.streams = streams
        model.source_files = {}

        config = StreamWriterConfig(output_dir=tmp_path, version="4.0")
        writer = StreamComponentWriter(model, config, template_engine=mock_engine)
        path = writer.write_main()
        content = path.read_text()
        assert "2.5" in content
        assert "1MON" in content
        assert "3.28" in content


class TestStreamV50CrossSection:
    """Tests for _render_cross_section() with model data (lines 509-570)."""

    def test_cross_section_with_data(self, tmp_path: Path, mock_engine: MagicMock) -> None:
        """_render_cross_section writes actual cross-section values from nodes."""
        model = MagicMock()
        cs = SimpleNamespace(bottom_elev=100.0, B0=50.0, s=1.5, n=0.035, max_flow_depth=15.0)
        node1 = SimpleNamespace(
            id=1,
            wetted_perimeter=200.0,
            conductivity=15.0,
            bed_thickness=2.0,
            gw_node=5,
            cross_section=cs,
            initial_condition=5.0,
        )
        streams = MagicMock()
        streams.nodes = {1: node1}
        streams.reaches = {}
        streams.budget_node_ids = []
        streams.budget_node_count = 0
        streams.diversions = {}
        streams.bypasses = {}
        streams.inflows = []
        streams.evap_node_specs = []
        streams.evap_area_file = ""
        streams.roughness_factor = 2.0
        streams.cross_section_length_factor = 3.0
        streams.ic_type = 1
        streams.ic_factor = 0.5
        model.streams = streams
        model.source_files = {}

        config = StreamWriterConfig(output_dir=tmp_path, version="5.0")
        writer = StreamComponentWriter(model, config, template_engine=mock_engine)
        path = writer.write_main()
        content = path.read_text()
        assert "100.00" in content  # bottom_elev
        assert "50.00" in content  # B0
        assert "0.0350" in content  # n
        assert "15.00" in content  # max_flow_depth
        assert "5.0000" in content  # initial_condition


class TestStreamEvaporationSection:
    """Tests for _render_evaporation() with evap specs (lines 621-637, 667)."""

    def test_evaporation_with_specs_and_file(self, tmp_path: Path, mock_engine: MagicMock) -> None:
        """_render_evaporation writes ET column and area column from specs."""
        model = MagicMock()
        spec = SimpleNamespace(node_id=1, et_column=3, area_column=5)
        streams = MagicMock()
        streams.nodes = {
            1: SimpleNamespace(
                id=1,
                wetted_perimeter=100.0,
                conductivity=10.0,
                bed_thickness=1.0,
                gw_node=1,
                cross_section=None,
                initial_condition=0.0,
            )
        }
        streams.reaches = {}
        streams.budget_node_ids = []
        streams.budget_node_count = 0
        streams.diversions = {}
        streams.bypasses = {}
        streams.inflows = []
        streams.evap_node_specs = [spec]
        streams.evap_area_file = "Streams\\SurfArea.dat"
        model.streams = streams
        model.source_files = {}

        config = StreamWriterConfig(output_dir=tmp_path)
        writer = StreamComponentWriter(model, config, template_engine=mock_engine)
        path = writer.write_main()
        content = path.read_text()
        assert "STARFL" in content
        assert "ICETST" in content

    def test_evaporation_without_file_zeroes_columns(
        self, tmp_path: Path, mock_engine: MagicMock
    ) -> None:
        """_render_evaporation zeroes out evap columns when no file is present."""
        model = MagicMock()
        spec = SimpleNamespace(node_id=1, et_column=3, area_column=5)
        streams = MagicMock()
        streams.nodes = {
            1: SimpleNamespace(
                id=1,
                wetted_perimeter=100.0,
                conductivity=10.0,
                bed_thickness=1.0,
                gw_node=1,
                cross_section=None,
                initial_condition=0.0,
            )
        }
        streams.reaches = {}
        streams.budget_node_ids = []
        streams.budget_node_count = 0
        streams.diversions = {}
        streams.bypasses = {}
        streams.inflows = []
        streams.evap_node_specs = [spec]
        streams.evap_area_file = ""  # No evap file
        model.streams = streams
        model.source_files = {}

        config = StreamWriterConfig(output_dir=tmp_path)
        writer = StreamComponentWriter(model, config, template_engine=mock_engine)
        path = writer.write_main()
        content = path.read_text()
        # With no evap file, columns should be zeroed
        lines = content.split("\n")
        evap_data_lines = [
            line
            for line in lines
            if line.strip() and not line.strip().startswith("C") and "0        0" in line
        ]
        assert len(evap_data_lines) >= 1


class TestStreamDiversionDataTs:
    """Tests for write_diversion_data_ts() (lines 933-1069 area)."""

    def test_write_diversion_data_ts(
        self, tmp_path: Path, model_with_full_streams: MagicMock, mock_engine: MagicMock
    ) -> None:
        """write_diversion_data_ts() creates diversion time series file."""
        config = StreamWriterConfig(output_dir=tmp_path)
        writer = StreamComponentWriter(model_with_full_streams, config, template_engine=mock_engine)

        mock_ts_writer = MagicMock()
        expected_path = config.stream_dir / config.diversions_file
        mock_ts_writer.write.return_value = expected_path

        with patch(
            "pyiwfm.io.timeseries_writer.IWFMTimeSeriesDataWriter",
            return_value=mock_ts_writer,
        ):
            path = writer.write_diversion_data_ts()

        mock_ts_writer.write.assert_called_once()
        assert path == expected_path

    def test_write_surface_area_ts(
        self, tmp_path: Path, model_with_full_streams: MagicMock, mock_engine: MagicMock
    ) -> None:
        """write_surface_area_ts() creates surface area time series file."""
        config = StreamWriterConfig(output_dir=tmp_path)
        writer = StreamComponentWriter(model_with_full_streams, config, template_engine=mock_engine)

        mock_ts_writer = MagicMock()
        expected_path = config.stream_dir / config.evap_area_file
        mock_ts_writer.write.return_value = expected_path

        with patch(
            "pyiwfm.io.timeseries_writer.IWFMTimeSeriesDataWriter",
            return_value=mock_ts_writer,
        ):
            writer.write_surface_area_ts()

        mock_ts_writer.write.assert_called_once()


class TestStreamDisconnectionFromModel:
    """Tests for _render_disconnection() using model data (line 487)."""

    def test_disconnection_uses_model_interaction_type(
        self, tmp_path: Path, mock_engine: MagicMock
    ) -> None:
        """_render_disconnection uses interaction_type from model streams."""
        model = MagicMock()
        streams = MagicMock()
        streams.nodes = {}
        streams.reaches = {}
        streams.budget_node_ids = []
        streams.budget_node_count = 0
        streams.diversions = {}
        streams.bypasses = {}
        streams.inflows = []
        streams.evap_node_specs = []
        streams.evap_area_file = ""
        streams.interaction_type = 2
        model.streams = streams
        model.source_files = {}

        config = StreamWriterConfig(output_dir=tmp_path)
        writer = StreamComponentWriter(model, config, template_engine=mock_engine)
        path = writer.write_main()
        content = path.read_text()
        assert "2" in content  # interaction type should be 2


class TestStreamEmptySubdir:
    """Tests for stream_subdir="" path prefix behavior (line 361-363)."""

    def test_empty_subdir_no_prefix(
        self, tmp_path: Path, model_with_full_streams: MagicMock, mock_engine: MagicMock
    ) -> None:
        """When stream_subdir is empty, file paths have no prefix."""
        config = StreamWriterConfig(output_dir=tmp_path, stream_subdir="")
        writer = StreamComponentWriter(model_with_full_streams, config, template_engine=mock_engine)
        # Should render without prefix in the template call
        writer.write_main()
        # The engine's render_template should have been called with
        # inflow_file that has no backslash prefix
        render_call = mock_engine.render_template.call_args
        assert "\\" not in render_call[1].get("inflow_file", "")
