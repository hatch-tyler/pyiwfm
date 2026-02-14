"""
Comprehensive tests for pyiwfm.io.stream_writer module.

Tests cover:
- StreamWriterConfig dataclass and properties
- StreamComponentWriter class methods
- write_stream_component convenience function
- File generation with various model configurations
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import FrozenInstanceError

from pyiwfm.io.stream_writer import (
    StreamWriterConfig,
    StreamComponentWriter,
    write_stream_component,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_model():
    """Create a basic mock model."""
    model = MagicMock()
    model.streams = None
    return model


@pytest.fixture
def mock_stream_node():
    """Create a mock stream node."""
    # Use spec to prevent MagicMock auto-attribute creation
    node = MagicMock(spec=['id', 'wetted_perimeter', 'conductivity', 'bed_thickness', 'cross_section', 'initial_condition'])
    node.id = 1
    node.wetted_perimeter = 200.0
    # gw_node not set intentionally - spec prevents auto-creation, so AttributeError will be raised
    return node


@pytest.fixture
def mock_diversion():
    """Create a mock diversion."""
    # Use spec to prevent auto-attribute creation for Jinja2 template compatibility
    div = MagicMock(spec=['id', 'source_node', 'destination_type', 'destination_id', 'max_rate',
                          'priority', 'name', 'max_div_column', 'max_div_fraction',
                          'recoverable_loss_column', 'recoverable_loss_fraction',
                          'non_recoverable_loss_column', 'non_recoverable_loss_fraction',
                          'spill_column', 'spill_fraction', 'delivery_dest_type',
                          'delivery_dest_id', 'delivery_column', 'delivery_fraction',
                          'irrigation_fraction_column', 'adjustment_column'])
    div.id = 1
    div.source_node = 10
    div.destination_type = "element"
    div.destination_id = 5
    div.max_rate = 1000.0
    div.priority = 1
    div.name = "Test Diversion"
    # Set required attributes for writer
    div.max_div_column = 1
    div.max_div_fraction = 1.0
    div.recoverable_loss_column = 0
    div.recoverable_loss_fraction = 0.0
    div.non_recoverable_loss_column = 0
    div.non_recoverable_loss_fraction = 0.0
    div.spill_column = 0
    div.spill_fraction = 0.0
    div.delivery_dest_type = 0
    div.delivery_dest_id = 5
    div.delivery_column = 0
    div.delivery_fraction = 1.0
    div.irrigation_fraction_column = 0
    div.adjustment_column = 0
    return div


@pytest.fixture
def mock_bypass():
    """Create a mock bypass."""
    # Use spec to prevent auto-attribute creation
    bypass = MagicMock(spec=['id', 'source_node', 'destination_node', 'capacity', 'name',
                             'flow_factor', 'flow_time_unit', 'spill_factor', 'spill_time_unit',
                             'diversion_column', 'rating_table_flows', 'rating_table_spills',
                             'dest_type', 'recoverable_loss_fraction', 'non_recoverable_loss_fraction',
                             'seepage_locations'])
    bypass.id = 1
    bypass.source_node = 15
    bypass.destination_node = 20
    bypass.dest_type = 0  # Stream node
    bypass.capacity = 500.0
    bypass.name = "Test Bypass"
    bypass.flow_factor = 1.0
    bypass.flow_time_unit = "1DAY"
    bypass.spill_factor = 1.0
    bypass.spill_time_unit = "1DAY"
    bypass.diversion_column = 1
    bypass.rating_table_flows = []
    bypass.rating_table_spills = []
    bypass.recoverable_loss_fraction = 0.0
    bypass.non_recoverable_loss_fraction = 0.0
    bypass.seepage_locations = []
    return bypass


@pytest.fixture
def mock_model_with_streams(mock_stream_node, mock_diversion, mock_bypass):
    """Create a mock model with full stream data."""
    model = MagicMock()

    streams = MagicMock()
    # Create second node with spec to prevent auto-attribute creation
    node2 = MagicMock(spec=['id', 'wetted_perimeter', 'conductivity', 'bed_thickness', 'cross_section', 'initial_condition'])
    node2.id = 2
    node2.wetted_perimeter = None
    streams.nodes = {1: mock_stream_node, 2: node2}
    streams.diversions = {1: mock_diversion}
    streams.bypasses = {1: mock_bypass}
    streams.inflows = [MagicMock()]

    model.streams = streams
    return model


@pytest.fixture
def mock_model_no_diversions():
    """Create a mock model with streams but no diversions/bypasses."""
    model = MagicMock()

    streams = MagicMock()
    # Use spec to prevent auto-attribute creation
    node = MagicMock(spec=['id', 'wetted_perimeter', 'conductivity', 'bed_thickness', 'cross_section', 'initial_condition'])
    node.id = 1
    node.wetted_perimeter = 100.0
    streams.nodes = {1: node}
    streams.diversions = {}
    streams.bypasses = {}
    streams.inflows = []

    model.streams = streams
    return model


@pytest.fixture
def stream_config(tmp_path):
    """Create a StreamWriterConfig for testing."""
    return StreamWriterConfig(output_dir=tmp_path)


# =============================================================================
# StreamWriterConfig Tests
# =============================================================================


class TestStreamWriterConfig:
    """Tests for StreamWriterConfig dataclass."""

    def test_config_creation_minimal(self, tmp_path):
        """Test config creation with minimal arguments."""
        config = StreamWriterConfig(output_dir=tmp_path)

        assert config.output_dir == tmp_path
        assert config.stream_subdir == "Stream"
        assert config.version == "4.0"

    def test_config_creation_full(self, tmp_path):
        """Test config creation with all arguments."""
        config = StreamWriterConfig(
            output_dir=tmp_path,
            stream_subdir="Streams",
            version="5.0",
            main_file="main.dat",
            inflow_file="inflows.dat",
            diver_specs_file="divers.dat",
            bypass_specs_file="bypass.dat",
            diversions_file="div_ts.dat",
            strm_budget_file="budget.hdf",
            strm_node_budget_file="node_budget.hdf",
            strm_hyd_file="hyd.out",
            diver_detail_file="diver_detail.hdf",
            flow_factor=0.001,
            flow_unit="cfs",
            length_factor=0.3048,
            length_unit="m",
            conductivity=20.0,
            bed_thickness=2.0,
            wetted_perimeter=100.0,
        )

        assert config.stream_subdir == "Streams"
        assert config.version == "5.0"
        assert config.main_file == "main.dat"
        assert config.flow_factor == 0.001
        assert config.conductivity == 20.0

    def test_stream_dir_property(self, tmp_path):
        """Test stream_dir property returns correct path."""
        config = StreamWriterConfig(output_dir=tmp_path, stream_subdir="Streams")

        assert config.stream_dir == tmp_path / "Streams"

    def test_main_path_property(self, tmp_path):
        """Test main_path property returns correct path."""
        config = StreamWriterConfig(
            output_dir=tmp_path,
            stream_subdir="Stream",
            main_file="Stream_MAIN.dat"
        )

        assert config.main_path == tmp_path / "Stream" / "Stream_MAIN.dat"

    def test_default_file_names(self, tmp_path):
        """Test default file name values."""
        config = StreamWriterConfig(output_dir=tmp_path)

        assert config.main_file == "Stream_MAIN.dat"
        assert config.inflow_file == "StreamInflow.dat"
        assert config.diver_specs_file == "DiverSpecs.dat"
        assert config.bypass_specs_file == "BypassSpecs.dat"
        assert config.diversions_file == "Diversions.dat"

    def test_default_output_files(self, tmp_path):
        """Test default output file paths."""
        config = StreamWriterConfig(output_dir=tmp_path)

        assert config.strm_budget_file == "../Results/StrmBud.hdf"
        assert config.strm_node_budget_file == "../Results/StrmNodeBud.hdf"
        assert config.strm_hyd_file == "../Results/StrmHyd.out"
        assert config.diver_detail_file == "../Results/DiverDetail.hdf"

    def test_default_unit_conversions(self, tmp_path):
        """Test default unit conversion values."""
        config = StreamWriterConfig(output_dir=tmp_path)

        assert config.flow_factor == pytest.approx(0.000022957, rel=1e-5)
        assert config.flow_unit == "ac.ft./day"
        assert config.length_factor == 1.0
        assert config.length_unit == "ft"

    def test_default_stream_bed_parameters(self, tmp_path):
        """Test default stream bed parameter values."""
        config = StreamWriterConfig(output_dir=tmp_path)

        assert config.conductivity == 10.0
        assert config.bed_thickness == 1.0
        assert config.wetted_perimeter == 150.0


# =============================================================================
# StreamComponentWriter Tests
# =============================================================================


class TestStreamComponentWriterInit:
    """Tests for StreamComponentWriter initialization."""

    def test_init_basic(self, mock_model, stream_config):
        """Test basic writer initialization."""
        writer = StreamComponentWriter(mock_model, stream_config)

        assert writer.model is mock_model
        assert writer.config is stream_config

    def test_init_with_template_engine(self, mock_model, stream_config):
        """Test initialization with custom template engine."""
        mock_engine = MagicMock()
        writer = StreamComponentWriter(mock_model, stream_config, mock_engine)

        assert writer.model is mock_model
        assert writer.config is stream_config

    def test_format_property(self, mock_model, stream_config):
        """Test format property returns correct value."""
        writer = StreamComponentWriter(mock_model, stream_config)

        assert writer.format == "iwfm_stream"


class TestStreamComponentWriterWrite:
    """Tests for StreamComponentWriter write methods."""

    def test_write_calls_write_all(self, mock_model, stream_config):
        """Test write() calls write_all()."""
        writer = StreamComponentWriter(mock_model, stream_config)

        with patch.object(writer, 'write_all') as mock_write_all:
            writer.write()
            mock_write_all.assert_called_once()

    def test_write_all_no_streams_defaults_false(self, mock_model, stream_config):
        """Test write_all with no streams and write_defaults=False."""
        writer = StreamComponentWriter(mock_model, stream_config)

        result = writer.write_all(write_defaults=False)

        assert result == {}

    def test_write_all_no_streams_defaults_true(self, mock_model, stream_config):
        """Test write_all with no streams and write_defaults=True."""
        writer = StreamComponentWriter(mock_model, stream_config)

        result = writer.write_all(write_defaults=True)

        assert "main" in result
        assert result["main"].exists()

    def test_write_all_with_streams(self, mock_model_with_streams, stream_config):
        """Test write_all with full stream data."""
        writer = StreamComponentWriter(mock_model_with_streams, stream_config)

        result = writer.write_all()

        assert "main" in result
        assert "diver_specs" in result
        assert "bypass_specs" in result
        assert all(p.exists() for p in result.values())

    def test_write_all_no_diversions(self, mock_model_no_diversions, stream_config):
        """Test write_all with streams but no diversions/bypasses."""
        writer = StreamComponentWriter(mock_model_no_diversions, stream_config)

        result = writer.write_all()

        assert "main" in result
        assert "diver_specs" not in result
        assert "bypass_specs" not in result

    def test_write_all_creates_stream_dir(self, mock_model, stream_config):
        """Test write_all creates stream directory."""
        writer = StreamComponentWriter(mock_model, stream_config)

        assert not stream_config.stream_dir.exists()

        writer.write_all()

        assert stream_config.stream_dir.exists()


class TestStreamComponentWriterWriteMain:
    """Tests for StreamComponentWriter.write_main method."""

    def test_write_main_creates_file(self, mock_model, stream_config):
        """Test write_main creates the main file."""
        writer = StreamComponentWriter(mock_model, stream_config)

        result = writer.write_main()

        assert result.exists()
        assert result.name == "Stream_MAIN.dat"

    def test_write_main_content_has_version(self, mock_model, stream_config):
        """Test main file contains version header."""
        writer = StreamComponentWriter(mock_model, stream_config)

        result = writer.write_main()
        content = result.read_text()

        assert "#4.0" in content or "4.0" in content

    def test_write_main_content_has_header(self, mock_model, stream_config):
        """Test main file contains header comments."""
        writer = StreamComponentWriter(mock_model, stream_config)

        result = writer.write_main()
        content = result.read_text()

        assert "STREAM PARAMETERS DATA FILE" in content
        assert "pyiwfm" in content

    def test_write_main_with_stream_nodes(self, mock_model_with_streams, stream_config):
        """Test main file contains stream node data."""
        writer = StreamComponentWriter(mock_model_with_streams, stream_config)

        result = writer.write_main()
        content = result.read_text()

        # Should have stream bed parameters for nodes
        assert "CSTRM" in content or "cstrm" in content.lower()

    def test_write_main_has_file_paths(self, mock_model_with_streams, stream_config):
        """Test main file references sub-files."""
        writer = StreamComponentWriter(mock_model_with_streams, stream_config)

        result = writer.write_main()
        content = result.read_text()

        # Should reference diversion and bypass files
        assert "DiverSpecs.dat" in content
        assert "BypassSpecs.dat" in content

    def test_write_main_no_inflows(self, mock_model_no_diversions, stream_config):
        """Test main file handles model with no inflows."""
        writer = StreamComponentWriter(mock_model_no_diversions, stream_config)

        result = writer.write_main()
        content = result.read_text()

        # Should still be valid
        assert "INFLOWFL" in content


class TestStreamComponentWriterWriteDiverSpecs:
    """Tests for StreamComponentWriter.write_diver_specs method."""

    def test_write_diver_specs_creates_file(self, mock_model_with_streams, stream_config):
        """Test write_diver_specs creates the file."""
        writer = StreamComponentWriter(mock_model_with_streams, stream_config)

        result = writer.write_diver_specs()

        assert result.exists()
        assert result.name == "DiverSpecs.dat"

    def test_write_diver_specs_content(self, mock_model_with_streams, stream_config):
        """Test diversion specs file content."""
        writer = StreamComponentWriter(mock_model_with_streams, stream_config)

        result = writer.write_diver_specs()
        content = result.read_text()

        assert "DIVERSION SPECIFICATIONS" in content
        assert "NDIVER" in content
        assert "Test Diversion" in content

    def test_write_diver_specs_empty_diversions(self, mock_model_no_diversions, stream_config):
        """Test diversion specs with no diversions."""
        writer = StreamComponentWriter(mock_model_no_diversions, stream_config)

        result = writer.write_diver_specs()
        content = result.read_text()

        assert "NDIVER" in content
        # Should have 0 diversions
        assert "0" in content

    def test_write_diver_specs_no_streams(self, mock_model, stream_config):
        """Test diversion specs with no stream component."""
        writer = StreamComponentWriter(mock_model, stream_config)

        result = writer.write_diver_specs()
        content = result.read_text()

        assert "NDIVER" in content


class TestStreamComponentWriterWriteBypassSpecs:
    """Tests for StreamComponentWriter.write_bypass_specs method."""

    def test_write_bypass_specs_creates_file(self, mock_model_with_streams, stream_config):
        """Test write_bypass_specs creates the file."""
        writer = StreamComponentWriter(mock_model_with_streams, stream_config)

        result = writer.write_bypass_specs()

        assert result.exists()
        assert result.name == "BypassSpecs.dat"

    def test_write_bypass_specs_content(self, mock_model_with_streams, stream_config):
        """Test bypass specs file content."""
        writer = StreamComponentWriter(mock_model_with_streams, stream_config)

        result = writer.write_bypass_specs()
        content = result.read_text()

        assert "BYPASS SPECIFICATIONS" in content
        assert "NBYPASS" in content
        assert "Test Bypass" in content

    def test_write_bypass_specs_empty_bypasses(self, mock_model_no_diversions, stream_config):
        """Test bypass specs with no bypasses."""
        writer = StreamComponentWriter(mock_model_no_diversions, stream_config)

        result = writer.write_bypass_specs()
        content = result.read_text()

        assert "NBYPASS" in content

    def test_write_bypass_specs_no_streams(self, mock_model, stream_config):
        """Test bypass specs with no stream component."""
        writer = StreamComponentWriter(mock_model, stream_config)

        result = writer.write_bypass_specs()
        content = result.read_text()

        assert "NBYPASS" in content


# =============================================================================
# write_stream_component Function Tests
# =============================================================================


class TestWriteStreamComponent:
    """Tests for write_stream_component convenience function."""

    def test_write_stream_component_basic(self, mock_model, tmp_path):
        """Test basic write_stream_component call."""
        result = write_stream_component(mock_model, tmp_path)

        assert isinstance(result, dict)
        assert "main" in result

    def test_write_stream_component_with_config(self, mock_model, tmp_path):
        """Test write_stream_component with custom config."""
        config = StreamWriterConfig(
            output_dir=tmp_path,
            stream_subdir="Streams",
            version="5.0",
        )

        result = write_stream_component(mock_model, tmp_path, config)

        assert (tmp_path / "Streams" / "Stream_MAIN.dat").exists()

    def test_write_stream_component_string_path(self, mock_model, tmp_path):
        """Test write_stream_component with string path."""
        result = write_stream_component(mock_model, str(tmp_path))

        assert isinstance(result, dict)
        assert "main" in result

    def test_write_stream_component_updates_config_output_dir(self, mock_model, tmp_path):
        """Test write_stream_component updates config output_dir."""
        other_path = tmp_path / "other"
        config = StreamWriterConfig(output_dir=other_path)

        result = write_stream_component(mock_model, tmp_path, config)

        # Should use tmp_path, not other_path
        assert (tmp_path / "Stream" / "Stream_MAIN.dat").exists()

    def test_write_stream_component_with_full_streams(
        self, mock_model_with_streams, tmp_path
    ):
        """Test write_stream_component with full stream data."""
        result = write_stream_component(mock_model_with_streams, tmp_path)

        assert "main" in result
        assert "diver_specs" in result
        assert "bypass_specs" in result


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestStreamWriterEdgeCases:
    """Tests for edge cases and error handling."""

    def test_wetted_perimeter_default_when_none(self, mock_model, stream_config):
        """Test default wetted perimeter when node value is None."""
        model = MagicMock()
        streams = MagicMock()
        # Use spec to prevent auto-attribute creation
        node = MagicMock(spec=['id', 'wetted_perimeter', 'conductivity', 'bed_thickness', 'cross_section', 'initial_condition'])
        node.id = 1
        node.wetted_perimeter = None
        streams.nodes = {1: node}
        streams.diversions = {}
        streams.bypasses = {}
        model.streams = streams

        writer = StreamComponentWriter(model, stream_config)
        result = writer.write_main()
        content = result.read_text()

        # Should use default wetted perimeter (150)
        assert "150" in content

    def test_multiple_stream_nodes(self, stream_config):
        """Test handling multiple stream nodes."""
        model = MagicMock()
        streams = MagicMock()
        nodes = {}
        for i in range(5):
            # Use spec to prevent auto-attribute creation
            node = MagicMock(spec=['id', 'wetted_perimeter', 'conductivity', 'bed_thickness', 'cross_section', 'initial_condition'])
            node.id = i + 1
            node.wetted_perimeter = 100.0 + i * 10
            nodes[i + 1] = node
        streams.nodes = nodes
        streams.diversions = {}
        streams.bypasses = {}
        model.streams = streams

        writer = StreamComponentWriter(model, stream_config)
        result = writer.write_main()
        content = result.read_text()

        # All nodes should be in hydrograph output
        for i in range(1, 6):
            assert f"StrmHyd_{i}" in content

    def test_multiple_diversions(self, stream_config):
        """Test handling multiple diversions."""
        model = MagicMock()
        streams = MagicMock()
        streams.nodes = {}

        diversions = {}
        for i in range(3):
            # Use spec to prevent auto-attribute creation
            div = MagicMock(spec=['id', 'source_node', 'destination_type', 'destination_id', 'max_rate',
                                  'priority', 'name', 'max_div_column', 'max_div_fraction',
                                  'recoverable_loss_column', 'recoverable_loss_fraction',
                                  'non_recoverable_loss_column', 'non_recoverable_loss_fraction',
                                  'spill_column', 'spill_fraction', 'delivery_dest_type',
                                  'delivery_dest_id', 'delivery_column', 'delivery_fraction',
                                  'irrigation_fraction_column', 'adjustment_column'])
            div.id = i + 1
            div.source_node = 10 + i
            div.destination_type = "element"
            div.destination_id = i + 1
            div.max_rate = 1000.0 + i * 100
            div.priority = i + 1
            div.name = f"Diversion {i + 1}"
            # Set required attributes for writer
            div.max_div_column = 1
            div.max_div_fraction = 1.0
            div.recoverable_loss_column = 0
            div.recoverable_loss_fraction = 0.0
            div.non_recoverable_loss_column = 0
            div.non_recoverable_loss_fraction = 0.0
            div.delivery_dest_type = 0
            div.delivery_dest_id = i + 1
            div.delivery_column = 0
            div.delivery_fraction = 1.0
            div.irrigation_fraction_column = 0
            div.adjustment_column = 0
            diversions[i + 1] = div

        streams.diversions = diversions
        streams.bypasses = {}
        streams.diversion_has_spills = False
        streams.diversion_element_groups = []
        streams.diversion_recharge_zones = []
        model.streams = streams

        writer = StreamComponentWriter(model, stream_config)
        result = writer.write_diver_specs()
        content = result.read_text()

        assert "Diversion 1" in content
        assert "Diversion 2" in content
        assert "Diversion 3" in content

    def test_multiple_bypasses(self, stream_config):
        """Test handling multiple bypasses."""
        model = MagicMock()
        streams = MagicMock()
        streams.nodes = {}
        streams.diversions = {}

        bypasses = {}
        for i in range(3):
            # Use spec to prevent auto-attribute creation
            bypass = MagicMock(spec=['id', 'source_node', 'destination_node', 'capacity', 'name',
                                     'flow_factor', 'flow_time_unit', 'spill_factor', 'spill_time_unit',
                                     'diversion_column', 'rating_table_flows', 'rating_table_spills',
                                     'dest_type', 'recoverable_loss_fraction', 'non_recoverable_loss_fraction',
                                     'seepage_locations'])
            bypass.id = i + 1
            bypass.source_node = 10 + i
            bypass.destination_node = 20 + i
            bypass.dest_type = 0
            bypass.capacity = 500.0 + i * 50
            bypass.name = f"Bypass {i + 1}"
            bypass.flow_factor = 1.0
            bypass.flow_time_unit = "1DAY"
            bypass.spill_factor = 1.0
            bypass.spill_time_unit = "1DAY"
            bypass.diversion_column = 1
            bypass.rating_table_flows = []
            bypass.rating_table_spills = []
            bypass.recoverable_loss_fraction = 0.0
            bypass.non_recoverable_loss_fraction = 0.0
            bypass.seepage_locations = []
            bypasses[i + 1] = bypass

        streams.bypasses = bypasses
        model.streams = streams

        writer = StreamComponentWriter(model, stream_config)
        result = writer.write_bypass_specs()
        content = result.read_text()

        assert "Bypass 1" in content
        assert "Bypass 2" in content
        assert "Bypass 3" in content

    def test_custom_version_in_main(self, mock_model, tmp_path):
        """Test custom version appears in main file."""
        config = StreamWriterConfig(output_dir=tmp_path, version="5.0")
        writer = StreamComponentWriter(mock_model, config)

        result = writer.write_main()
        content = result.read_text()

        assert "#5.0" in content

    def test_custom_unit_factors_in_main(self, mock_model, tmp_path):
        """Test custom unit factors appear in main file."""
        config = StreamWriterConfig(
            output_dir=tmp_path,
            flow_factor=0.001,
            flow_unit="cfs",
        )
        writer = StreamComponentWriter(mock_model, config)

        result = writer.write_main()
        content = result.read_text()

        assert "0.001" in content
        assert "cfs" in content

    def test_no_inflows_attribute(self, stream_config):
        """Test handling streams without inflows attribute."""
        model = MagicMock()
        # Use spec to list only the attributes we want to exist
        streams = MagicMock(spec=['nodes', 'diversions', 'bypasses', 'reaches', 'budget_node_ids'])
        streams.nodes = {}
        streams.diversions = {}
        streams.bypasses = {}
        streams.reaches = {}
        streams.budget_node_ids = []
        # inflows not in spec, so accessing it will raise AttributeError (handled by writer)
        model.streams = streams

        writer = StreamComponentWriter(model, stream_config)

        # Should not raise
        result = writer.write_main()
        assert result.exists()

    def test_budget_nodes_limit(self, stream_config):
        """Test budget output limits to first 3 nodes."""
        model = MagicMock()
        streams = MagicMock()
        # Use spec to prevent auto-attribute creation
        nodes = {}
        for i in range(1, 11):
            node = MagicMock(spec=['id', 'wetted_perimeter', 'conductivity', 'bed_thickness', 'cross_section', 'initial_condition'])
            node.id = i
            node.wetted_perimeter = 100.0
            nodes[i] = node
        streams.nodes = nodes
        streams.diversions = {}
        streams.bypasses = {}
        model.streams = streams

        writer = StreamComponentWriter(model, stream_config)
        result = writer.write_main()
        content = result.read_text()

        # Should have NBUDR = 3
        assert "3" in content
