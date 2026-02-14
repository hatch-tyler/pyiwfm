"""Supplementary tests for preprocessor_writer.py targeting uncovered branches.

Covers:
- PreProcessorWriter class methods with mock model
- write_stream_config with/without streams
- write_lake_config with/without lakes
- write_all() with all combinations
- write_main() content verification
- _get_output_binary_path
- Convenience function write_preprocessor_files
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, PropertyMock
from typing import Any

import numpy as np
import pytest

from pyiwfm.io.preprocessor_writer import (
    PreProcessorWriter,
    write_preprocessor_files,
    write_nodes_file,
    write_elements_file,
    write_stratigraphy_file,
)
from pyiwfm.io.config import PreProcessorFileConfig
from pyiwfm.templates.engine import TemplateEngine


# =============================================================================
# Mock Model Helper
# =============================================================================


def _make_mock_model(
    has_streams: bool = False,
    has_lakes: bool = False,
    n_nodes: int = 4,
    n_elements: int = 2,
    n_layers: int = 2,
    name: str = "TestModel",
) -> MagicMock:
    """Create a mock IWFMModel for testing.

    Uses SimpleNamespace for objects that need to support format strings,
    since MagicMock.__format__ does not support format specs.
    """
    model = MagicMock()
    model.name = name
    model.n_nodes = n_nodes
    model.n_elements = n_elements
    model.has_streams = has_streams
    model.has_lakes = has_lakes

    # Mock grid with nodes and elements using SimpleNamespace
    nodes = {}
    for i in range(1, n_nodes + 1):
        nodes[i] = SimpleNamespace(x=float(i * 1000), y=float(i * 2000))

    elements = {}
    for i in range(1, n_elements + 1):
        elements[i] = SimpleNamespace(vertices=[1, 2, 3, 4], subregion=1)

    grid = MagicMock()
    grid.nodes = nodes
    grid.elements = elements
    # subregions values need a .name attribute that supports format strings
    grid.subregions = {1: SimpleNamespace(name="Region_1")}
    model.grid = grid

    # Mock stratigraphy
    strat = MagicMock()
    strat.n_layers = n_layers
    strat.gs_elev = np.array([100.0] * n_nodes)
    strat.top_elev = np.tile(np.array([90.0, 70.0])[:n_layers], (n_nodes, 1))
    strat.bottom_elev = np.tile(np.array([70.0, 50.0])[:n_layers], (n_nodes, 1))
    model.stratigraphy = strat

    # Mock streams
    if has_streams:
        streams = MagicMock()
        reach = SimpleNamespace(
            id=1, upstream_node=1, downstream_node=2, outflow_reach=0
        )
        streams.iter_reaches.return_value = [reach]

        node = SimpleNamespace(id=1, gw_node=10, reach_id=1, stage=50.0)
        streams.iter_nodes.return_value = [node]
        model.streams = streams
    else:
        model.streams = None

    # Mock lakes
    if has_lakes:
        lakes = MagicMock()
        lake = SimpleNamespace(id=1, name="Test Lake", elements=[1, 2])
        lakes.iter_lakes.return_value = [lake]
        model.lakes = lakes
    else:
        model.lakes = None

    return model


# =============================================================================
# PreProcessorWriter Tests
# =============================================================================


class TestPreProcessorWriterClass:
    """Tests for PreProcessorWriter class methods."""

    def test_init(self, tmp_path: Path) -> None:
        """Test PreProcessorWriter initialization."""
        model = _make_mock_model()
        config = PreProcessorFileConfig(output_dir=tmp_path)
        writer = PreProcessorWriter(model, config)

        assert writer.model is model
        assert writer.config is config

    def test_format_property(self, tmp_path: Path) -> None:
        """Test format property returns correct value."""
        model = _make_mock_model()
        config = PreProcessorFileConfig(output_dir=tmp_path)
        writer = PreProcessorWriter(model, config)

        assert writer.format == "iwfm_preprocessor"

    def test_write_delegates_to_write_all(self, tmp_path: Path) -> None:
        """Test write() calls write_all()."""
        model = _make_mock_model()
        config = PreProcessorFileConfig(output_dir=tmp_path)
        writer = PreProcessorWriter(model, config)

        writer.write()

        # Verify files were created
        assert config.node_path.exists()
        assert config.element_path.exists()

    def test_write_all_basic(self, tmp_path: Path) -> None:
        """Test write_all creates basic files."""
        model = _make_mock_model()
        config = PreProcessorFileConfig(output_dir=tmp_path)
        writer = PreProcessorWriter(model, config)

        results = writer.write_all()

        assert "nodes" in results
        assert "elements" in results
        assert "stratigraphy" in results
        assert "main" in results
        assert "stream_config" not in results
        assert "lake_config" not in results

    def test_write_all_with_streams(self, tmp_path: Path) -> None:
        """Test write_all includes stream config when has_streams."""
        model = _make_mock_model(has_streams=True)
        config = PreProcessorFileConfig(output_dir=tmp_path)
        writer = PreProcessorWriter(model, config)

        results = writer.write_all()

        assert "stream_config" in results
        assert results["stream_config"].exists()

    def test_write_all_with_lakes(self, tmp_path: Path) -> None:
        """Test write_all includes lake config when has_lakes."""
        model = _make_mock_model(has_lakes=True)
        config = PreProcessorFileConfig(output_dir=tmp_path)
        writer = PreProcessorWriter(model, config)

        results = writer.write_all()

        assert "lake_config" in results
        assert results["lake_config"].exists()

    def test_write_all_with_streams_and_lakes(self, tmp_path: Path) -> None:
        """Test write_all with both streams and lakes."""
        model = _make_mock_model(has_streams=True, has_lakes=True)
        config = PreProcessorFileConfig(output_dir=tmp_path)
        writer = PreProcessorWriter(model, config)

        results = writer.write_all()

        assert "stream_config" in results
        assert "lake_config" in results

    def test_write_main_content(self, tmp_path: Path) -> None:
        """Test write_main content includes expected sections."""
        model = _make_mock_model(name="MainContentTest")
        config = PreProcessorFileConfig(output_dir=tmp_path)
        writer = PreProcessorWriter(model, config)

        filepath = writer.write_main()

        content = filepath.read_text()
        assert "INTEGRATED WATER FLOW MODEL" in content
        assert "MainContentTest" in content
        assert "Generated by pyiwfm" in content
        assert "BINARY OUTPUT FOR SIMULATION" in content
        assert "ELEMENT CONFIGURATION FILE" in content
        assert "NODE X-Y COORDINATE FILE" in content
        assert "STRATIGRAPHIC DATA FILE" in content
        assert "FACTLTOU" in content
        assert "FEET" in content

    def test_write_main_with_streams(self, tmp_path: Path) -> None:
        """Test write_main includes stream file reference."""
        model = _make_mock_model(has_streams=True)
        config = PreProcessorFileConfig(output_dir=tmp_path)
        writer = PreProcessorWriter(model, config)

        filepath = writer.write_main()

        content = filepath.read_text()
        assert "STREAM GEOMETRIC DATA FILE" in content
        assert config.stream_config_file in content

    def test_write_main_without_streams(self, tmp_path: Path) -> None:
        """Test write_main has empty stream reference when no streams."""
        model = _make_mock_model(has_streams=False)
        config = PreProcessorFileConfig(output_dir=tmp_path)
        writer = PreProcessorWriter(model, config)

        filepath = writer.write_main()

        content = filepath.read_text()
        assert "STREAM GEOMETRIC DATA FILE" in content

    def test_get_output_binary_path(self, tmp_path: Path) -> None:
        """Test _get_output_binary_path returns correct format."""
        model = _make_mock_model()
        config = PreProcessorFileConfig(output_dir=tmp_path)
        writer = PreProcessorWriter(model, config)

        path = writer._get_output_binary_path()

        assert "Preprocessor.bin" in path
        assert "\\" not in path  # Backslashes converted to forward slashes

    def test_write_nodes(self, tmp_path: Path) -> None:
        """Test write_nodes creates valid file."""
        model = _make_mock_model(n_nodes=9)
        config = PreProcessorFileConfig(output_dir=tmp_path)
        writer = PreProcessorWriter(model, config)

        filepath = writer.write_nodes()

        content = filepath.read_text()
        assert "NNODES" in content
        assert "9" in content
        assert "FACTXY" in content

    def test_write_elements(self, tmp_path: Path) -> None:
        """Test write_elements creates valid file."""
        model = _make_mock_model(n_elements=4)
        config = PreProcessorFileConfig(output_dir=tmp_path)
        writer = PreProcessorWriter(model, config)

        filepath = writer.write_elements()

        content = filepath.read_text()
        assert "NELEM" in content
        assert "NSUBREGION" in content

    def test_write_stratigraphy(self, tmp_path: Path) -> None:
        """Test write_stratigraphy creates valid file."""
        model = _make_mock_model(n_layers=2)
        config = PreProcessorFileConfig(output_dir=tmp_path)
        writer = PreProcessorWriter(model, config)

        filepath = writer.write_stratigraphy()

        content = filepath.read_text()
        assert "NLAYERS" in content
        assert "2" in content
        assert "FACTEL" in content
        assert "AQT_L1" in content
        assert "AQF_L1" in content

    def test_write_stream_config(self, tmp_path: Path) -> None:
        """Test write_stream_config with streams."""
        model = _make_mock_model(has_streams=True)
        config = PreProcessorFileConfig(output_dir=tmp_path)
        writer = PreProcessorWriter(model, config)

        filepath = writer.write_stream_config()

        content = filepath.read_text()
        assert "STREAM CONFIGURATION FILE" in content
        assert "NREACH" in content
        assert "NSTRMNODE" in content

    def test_write_stream_config_no_streams(self, tmp_path: Path) -> None:
        """Test write_stream_config when streams is None."""
        model = _make_mock_model(has_streams=False)
        model.streams = None
        config = PreProcessorFileConfig(output_dir=tmp_path)
        writer = PreProcessorWriter(model, config)

        filepath = writer.write_stream_config()

        content = filepath.read_text()
        assert "0" in content  # 0 reaches and nodes

    def test_write_lake_config(self, tmp_path: Path) -> None:
        """Test write_lake_config with lakes."""
        model = _make_mock_model(has_lakes=True)
        config = PreProcessorFileConfig(output_dir=tmp_path)
        writer = PreProcessorWriter(model, config)

        filepath = writer.write_lake_config()

        content = filepath.read_text()
        assert "LAKE CONFIGURATION FILE" in content
        assert "NLAKES" in content
        assert "Test Lake" in content

    def test_write_lake_config_no_lakes(self, tmp_path: Path) -> None:
        """Test write_lake_config when lakes is None."""
        model = _make_mock_model(has_lakes=False)
        model.lakes = None
        config = PreProcessorFileConfig(output_dir=tmp_path)
        writer = PreProcessorWriter(model, config)

        filepath = writer.write_lake_config()

        content = filepath.read_text()
        assert "0" in content  # 0 lakes


# =============================================================================
# Convenience Functions
# =============================================================================


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_write_preprocessor_files(self, tmp_path: Path) -> None:
        """Test write_preprocessor_files convenience function."""
        model = _make_mock_model()

        results = write_preprocessor_files(model, tmp_path)

        assert "nodes" in results
        assert "elements" in results
        assert "stratigraphy" in results
        assert "main" in results

    def test_write_preprocessor_files_with_versions(self, tmp_path: Path) -> None:
        """Test write_preprocessor_files with custom versions."""
        model = _make_mock_model(has_streams=True, has_lakes=True)

        results = write_preprocessor_files(
            model, tmp_path,
            stream_version="4.0",
            lake_version="4.0",
        )

        assert "stream_config" in results
        assert "lake_config" in results

    def test_write_preprocessor_files_string_path(self, tmp_path: Path) -> None:
        """Test write_preprocessor_files with string output directory."""
        model = _make_mock_model()

        results = write_preprocessor_files(model, str(tmp_path))

        assert "main" in results


# =============================================================================
# Standalone Writer Functions
# =============================================================================


class TestStandaloneWriterFunctions:
    """Tests for standalone write_nodes_file, write_elements_file, etc."""

    def test_write_nodes_file_basic(self, tmp_path: Path) -> None:
        """Test write_nodes_file standalone function."""
        filepath = write_nodes_file(
            output_path=tmp_path / "nodes.dat",
            node_ids=np.array([1, 2, 3], dtype=np.int32),
            x_coords=np.array([100.0, 200.0, 300.0]),
            y_coords=np.array([400.0, 500.0, 600.0]),
        )

        assert filepath.exists()
        content = filepath.read_text()
        assert "NNODES" in content
        assert "3" in content

    def test_write_nodes_file_with_factor(self, tmp_path: Path) -> None:
        """Test write_nodes_file with custom coordinate factor."""
        filepath = write_nodes_file(
            output_path=tmp_path / "nodes.dat",
            node_ids=np.array([1], dtype=np.int32),
            x_coords=np.array([100.0]),
            y_coords=np.array([200.0]),
            coord_factor=0.3048,
        )

        content = filepath.read_text()
        assert "0.304800" in content
        assert "FACTXY" in content

    def test_write_elements_file_basic(self, tmp_path: Path) -> None:
        """Test write_elements_file standalone function."""
        filepath = write_elements_file(
            output_path=tmp_path / "elements.dat",
            element_ids=np.array([1, 2], dtype=np.int32),
            vertices=np.array([[1, 2, 3, 4], [3, 4, 5, 6]], dtype=np.int32),
            subregions=np.array([1, 1], dtype=np.int32),
        )

        assert filepath.exists()
        content = filepath.read_text()
        assert "NELEM" in content
        assert "NSUBREGION" in content

    def test_write_elements_file_with_names(self, tmp_path: Path) -> None:
        """Test write_elements_file with custom subregion names."""
        filepath = write_elements_file(
            output_path=tmp_path / "elements.dat",
            element_ids=np.array([1, 2], dtype=np.int32),
            vertices=np.array([[1, 2, 3, 4], [3, 4, 5, 6]], dtype=np.int32),
            subregions=np.array([1, 2], dtype=np.int32),
            subregion_names={1: "North", 2: "South"},
        )

        content = filepath.read_text()
        assert "North" in content
        assert "South" in content

    def test_write_stratigraphy_file_basic(self, tmp_path: Path) -> None:
        """Test write_stratigraphy_file standalone function."""
        filepath = write_stratigraphy_file(
            output_path=tmp_path / "strat.dat",
            node_ids=np.array([1, 2, 3], dtype=np.int32),
            ground_surface=np.array([100.0, 100.0, 100.0]),
            layer_tops=np.array([[90.0], [90.0], [90.0]]),
            layer_bottoms=np.array([[70.0], [70.0], [70.0]]),
        )

        assert filepath.exists()
        content = filepath.read_text()
        assert "NLAYERS" in content
        assert "AQT_L1" in content
        assert "AQF_L1" in content

    def test_write_stratigraphy_file_multi_layer(self, tmp_path: Path) -> None:
        """Test write_stratigraphy_file with multiple layers."""
        filepath = write_stratigraphy_file(
            output_path=tmp_path / "strat.dat",
            node_ids=np.array([1, 2], dtype=np.int32),
            ground_surface=np.array([100.0, 100.0]),
            layer_tops=np.array([[90.0, 70.0], [90.0, 70.0]]),
            layer_bottoms=np.array([[70.0, 50.0], [70.0, 50.0]]),
        )

        content = filepath.read_text()
        assert "AQT_L2" in content
        assert "AQF_L2" in content

    def test_write_stratigraphy_file_with_factor(self, tmp_path: Path) -> None:
        """Test write_stratigraphy_file with custom elevation factor."""
        filepath = write_stratigraphy_file(
            output_path=tmp_path / "strat.dat",
            node_ids=np.array([1], dtype=np.int32),
            ground_surface=np.array([100.0]),
            layer_tops=np.array([[90.0]]),
            layer_bottoms=np.array([[70.0]]),
            elev_factor=0.3048,
        )

        content = filepath.read_text()
        assert "0.304800" in content
        assert "FACTEL" in content

    def test_write_nodes_file_creates_parent_dir(self, tmp_path: Path) -> None:
        """Test write_nodes_file creates parent directory."""
        filepath = write_nodes_file(
            output_path=tmp_path / "nested" / "dir" / "nodes.dat",
            node_ids=np.array([1], dtype=np.int32),
            x_coords=np.array([100.0]),
            y_coords=np.array([200.0]),
        )

        assert filepath.exists()
        assert filepath.parent.exists()


# =============================================================================
# Preprocessor Reader Tests (io/preprocessor.py)
# =============================================================================


from pyiwfm.io.preprocessor import (
    _is_comment_line,
    _parse_value_line,
    _resolve_path,
    read_preprocessor_main,
    read_subregions_file,
    load_model_from_preprocessor,
    write_preprocessor_main,
    save_model_to_preprocessor,
    PreProcessorConfig,
)
from pyiwfm.core.exceptions import FileFormatError


class TestHelperFunctions:
    """Tests for preprocessor helper functions."""

    def test_is_comment_line_empty(self) -> None:
        """Empty line is a comment."""
        assert _is_comment_line("") is True
        assert _is_comment_line("   ") is True

    def test_is_comment_line_c(self) -> None:
        """Lines starting with C/c/* are comments."""
        assert _is_comment_line("C  This is a comment") is True
        assert _is_comment_line("c  lowercase comment") is True
        assert _is_comment_line("*  star comment") is True

    def test_is_comment_line_data(self) -> None:
        """Lines not starting with comment chars are data."""
        assert _is_comment_line("10  / VALUE") is False
        assert _is_comment_line("  10  / VALUE") is False  # leading whitespace = data

    def test_parse_value_line_slash(self) -> None:
        """Value line with slash separator."""
        value, desc = _parse_value_line("nodes.dat  / NODES_FILE")
        assert value == "nodes.dat"
        assert desc == "NODES_FILE"

    def test_parse_value_line_hash_not_recognized(self) -> None:
        """Hash is not recognized as an inline comment delimiter."""
        value, desc = _parse_value_line("nodes.dat  # NODES_FILE")
        assert value == "nodes.dat  # NODES_FILE"
        assert desc == ""

    def test_parse_value_line_no_separator(self) -> None:
        """Value line with no separator."""
        value, desc = _parse_value_line("some_value")
        assert value == "some_value"
        assert desc == ""

    def test_resolve_path_absolute(self, tmp_path: Path) -> None:
        """Absolute path returned as-is."""
        abs_path = str(tmp_path / "file.dat")
        result = _resolve_path(Path("/base"), abs_path)
        assert result == Path(abs_path)

    def test_resolve_path_relative(self) -> None:
        """Relative path joined with base dir."""
        result = _resolve_path(Path("/base/dir"), "subdir/file.dat")
        assert result == Path("/base/dir/subdir/file.dat")


class TestReadPreprocessorMain:
    """Tests for read_preprocessor_main()."""

    def test_read_basic_file(self, tmp_path: Path) -> None:
        """Read a basic preprocessor main file."""
        pp_file = tmp_path / "test_pp.in"
        pp_file.write_text(
            "C  Test preprocessor file\n"
            "C\n"
            "TestModel                       / MODEL_NAME\n"
            "nodes.dat                       / NODES_FILE\n"
            "elements.dat                    / ELEMENTS_FILE\n"
            "strat.dat                       / STRATIGRAPHY_FILE\n"
            "2                               / N_LAYERS\n"
            "FT                              / LENGTH_UNIT\n"
        )
        config = read_preprocessor_main(pp_file)
        assert config.model_name == "TestModel"
        assert config.nodes_file is not None
        assert config.elements_file is not None
        assert config.stratigraphy_file is not None
        assert config.n_layers == 2
        assert config.length_unit == "FT"

    def test_read_with_streams_and_lakes(self, tmp_path: Path) -> None:
        """Read preprocessor file with stream and lake references."""
        pp_file = tmp_path / "test_pp.in"
        pp_file.write_text(
            "Model_SL                        / MODEL_NAME\n"
            "nodes.dat                       / NODES_FILE\n"
            "elements.dat                    / ELEMENTS_FILE\n"
            "streams.dat                     / STREAM_FILE\n"
            "lakes.dat                       / LAKE_FILE\n"
        )
        config = read_preprocessor_main(pp_file)
        assert config.streams_file is not None
        assert config.lakes_file is not None

    def test_read_with_all_settings(self, tmp_path: Path) -> None:
        """Read preprocessor file with unit settings."""
        pp_file = tmp_path / "test_pp.in"
        pp_file.write_text(
            "FullModel                       / MODEL_NAME\n"
            "nodes.dat                       / NODES_FILE\n"
            "elements.dat                    / ELEMENTS_FILE\n"
            "METERS                          / LENGTH_UNIT\n"
            "HECTARES                        / AREA_UNIT\n"
            "MCM                             / VOLUME_UNIT\n"
            "output                          / OUTPUT_DIR\n"
        )
        config = read_preprocessor_main(pp_file)
        assert config.length_unit == "METERS"
        assert config.area_unit == "HECTARES"
        assert config.volume_unit == "MCM"
        assert config.output_dir is not None

    def test_read_invalid_nlayers(self, tmp_path: Path) -> None:
        """Non-integer n_layers value is ignored."""
        pp_file = tmp_path / "test_pp.in"
        pp_file.write_text(
            "Model                           / MODEL_NAME\n"
            "abc                             / N_LAYERS\n"
        )
        config = read_preprocessor_main(pp_file)
        assert config.n_layers == 1  # Default

    def test_read_empty_file(self, tmp_path: Path) -> None:
        """Empty file produces defaults."""
        pp_file = tmp_path / "empty_pp.in"
        pp_file.write_text("C  Only comments\nC\n")
        config = read_preprocessor_main(pp_file)
        assert config.model_name == ""
        assert config.nodes_file is None


class TestReadSubregionsFile:
    """Tests for read_subregions_file()."""

    def test_read_valid_subregions(self, tmp_path: Path) -> None:
        """Read valid subregions file."""
        sr_file = tmp_path / "subregions.dat"
        sr_file.write_text(
            "C  Subregion definitions\n"
            "C  ID  NAME\n"
            "3                               / NSUBREGION\n"
            "1  North_Region\n"
            "2  Central_Region\n"
            "3  South_Region\n"
        )
        subregions = read_subregions_file(sr_file)
        assert len(subregions) == 3
        assert subregions[1].name == "North_Region"
        assert subregions[2].name == "Central_Region"
        assert subregions[3].name == "South_Region"

    def test_read_invalid_nsubregion(self, tmp_path: Path) -> None:
        """Non-integer NSUBREGION raises FileFormatError."""
        sr_file = tmp_path / "subregions.dat"
        sr_file.write_text("abc                             / NSUBREGION\n")
        with pytest.raises(FileFormatError, match="Invalid NSUBREGION"):
            read_subregions_file(sr_file)

    def test_read_missing_nsubregion(self, tmp_path: Path) -> None:
        """File with only comments raises FileFormatError."""
        sr_file = tmp_path / "subregions.dat"
        sr_file.write_text("C  Only comments\nC  More comments\n")
        with pytest.raises(FileFormatError, match="Could not find NSUBREGION"):
            read_subregions_file(sr_file)

    def test_read_invalid_subregion_id(self, tmp_path: Path) -> None:
        """Non-integer subregion ID raises FileFormatError."""
        sr_file = tmp_path / "subregions.dat"
        sr_file.write_text(
            "1                               / NSUBREGION\n"
            "abc  Bad_ID_Region\n"
        )
        with pytest.raises(FileFormatError, match="Invalid subregion data"):
            read_subregions_file(sr_file)

    def test_read_subregion_no_name(self, tmp_path: Path) -> None:
        """Subregion with only ID and no name."""
        sr_file = tmp_path / "subregions.dat"
        sr_file.write_text("1                               / NSUBREGION\n5\n")
        subregions = read_subregions_file(sr_file)
        assert len(subregions) == 1
        assert subregions[5].id == 5
        assert subregions[5].name == ""


class TestLoadModelFromPreprocessor:
    """Tests for load_model_from_preprocessor()."""

    def test_load_missing_nodes_file(self, tmp_path: Path) -> None:
        """Missing nodes file ref raises FileFormatError."""
        pp_file = tmp_path / "test_pp.in"
        pp_file.write_text(
            "Model                           / MODEL_NAME\n"
            "elements.dat                    / ELEMENTS_FILE\n"
        )
        with pytest.raises(FileFormatError, match="Nodes file not specified"):
            load_model_from_preprocessor(pp_file)

    def test_load_missing_elements_file(self, tmp_path: Path) -> None:
        """Missing elements file ref raises FileFormatError."""
        pp_file = tmp_path / "test_pp.in"
        pp_file.write_text(
            "Model                           / MODEL_NAME\n"
            "nodes.dat                       / NODES_FILE\n"
        )
        # Create the nodes file so that read_nodes doesn't fail first
        nodes_file = tmp_path / "nodes.dat"
        nodes_file.write_text(
            "C  Node file\n"
            "3                               / NNODES\n"
            "1.0                             / FACTXY\n"
            "     1      0.0      0.0\n"
            "     2   1000.0      0.0\n"
            "     3   1000.0   1000.0\n"
        )
        with pytest.raises(FileFormatError, match="Elements file not specified"):
            load_model_from_preprocessor(pp_file)


class TestWritePreprocessorMain:
    """Tests for write_preprocessor_main()."""

    def test_write_default_header(self, tmp_path: Path) -> None:
        """Write with default header."""
        config = PreProcessorConfig(
            base_dir=tmp_path,
            model_name="TestWrite",
            n_layers=3,
        )
        out_file = tmp_path / "pp_main.in"
        write_preprocessor_main(out_file, config)
        content = out_file.read_text()
        assert "IWFM PreProcessor Main Input File" in content
        assert "Generated by pyiwfm" in content
        assert "TestWrite" in content
        assert "3" in content

    def test_write_custom_header(self, tmp_path: Path) -> None:
        """Write with custom header."""
        config = PreProcessorConfig(
            base_dir=tmp_path,
            model_name="CustomHeader",
        )
        out_file = tmp_path / "pp_main.in"
        write_preprocessor_main(out_file, config, header="My Custom Header\nLine 2")
        content = out_file.read_text()
        assert "My Custom Header" in content
        assert "Line 2" in content
        assert "CustomHeader" in content

    def test_write_with_file_paths(self, tmp_path: Path) -> None:
        """Write with all file paths set."""
        config = PreProcessorConfig(
            base_dir=tmp_path,
            model_name="FullPaths",
            nodes_file=tmp_path / "nodes.dat",
            elements_file=tmp_path / "elements.dat",
            stratigraphy_file=tmp_path / "strat.dat",
            subregions_file=tmp_path / "subreg.dat",
            n_layers=2,
            length_unit="METERS",
            area_unit="SQ_M",
            volume_unit="CU_M",
        )
        out_file = tmp_path / "pp_main.in"
        write_preprocessor_main(out_file, config)
        content = out_file.read_text()
        assert "NODES_FILE" in content
        assert "ELEMENTS_FILE" in content
        assert "STRATIGRAPHY_FILE" in content
        assert "SUBREGIONS_FILE" in content
        assert "METERS" in content


class TestSaveModelToPreprocessor:
    """Tests for save_model_to_preprocessor()."""

    def test_save_minimal_model(self, tmp_path: Path) -> None:
        """Save a model with no mesh or stratigraphy."""
        from pyiwfm.core.model import IWFMModel
        model = IWFMModel(name="minimal_model")
        model.mesh = None
        model.stratigraphy = None

        config = save_model_to_preprocessor(model, tmp_path)
        assert config.model_name == "minimal_model"
        assert config.nodes_file is None
        assert config.elements_file is None


# =============================================================================
# load_complete_model Tests
# =============================================================================

import sys
from unittest.mock import patch, MagicMock
from datetime import datetime

from pyiwfm.io.preprocessor import load_complete_model, save_complete_model


def _make_sim_config(
    tmp_path: Path,
    *,
    preprocessor_file: Path | None = None,
    groundwater_file: str | None = None,
    streams_file: str | None = None,
    lakes_file: str | None = None,
    rootzone_file: str | None = None,
    model_name: str = "TestModel",
) -> MagicMock:
    """Create a mock SimulationConfig for testing load_complete_model."""
    cfg = MagicMock()
    cfg.model_name = model_name
    cfg.preprocessor_file = preprocessor_file
    cfg.groundwater_file = groundwater_file
    cfg.streams_file = streams_file
    cfg.lakes_file = lakes_file
    cfg.rootzone_file = rootzone_file
    cfg.start_date = datetime(2000, 1, 1)
    cfg.end_date = datetime(2000, 12, 31)
    cfg.time_step_length = 1
    cfg.time_step_unit = MagicMock()
    cfg.time_step_unit.value = "DAY"
    return cfg


class TestLoadCompleteModelGroundwater:
    """Tests for load_complete_model groundwater loading (lines 526-544)."""

    @patch("pyiwfm.io.preprocessor.load_model_from_preprocessor")
    def test_gw_component_loaded(self, mock_load_pp, tmp_path):
        """When sim_config.groundwater_file is set and file exists, GW is loaded."""
        sim_file = tmp_path / "simulation.in"
        sim_file.write_text("dummy")

        # Create the GW file so .exists() returns True
        gw_path = tmp_path / "groundwater.dat"
        gw_path.write_text("dummy gw data")

        # Mock model returned by preprocessor loader
        mock_model = MagicMock()
        mock_model.metadata = {}
        mock_model.mesh.n_nodes = 10
        mock_model.mesh.n_elements = 5
        mock_model.n_layers = 2
        mock_load_pp.return_value = mock_model

        sim_config = _make_sim_config(
            tmp_path,
            groundwater_file=str(gw_path),
        )
        # Make the preprocessor file "exist"
        pp_file = tmp_path / "pp.in"
        pp_file.write_text("dummy pp")
        sim_config.preprocessor_file = pp_file

        mock_gw_reader = MagicMock()
        mock_gw_reader.read_wells.return_value = {1: MagicMock(), 2: MagicMock()}
        mock_gw_component = MagicMock()

        with patch("pyiwfm.io.simulation.SimulationReader") as mock_sim_reader_cls, \
             patch("pyiwfm.io.groundwater.GroundwaterReader", return_value=mock_gw_reader), \
             patch("pyiwfm.io.streams.StreamReader"), \
             patch("pyiwfm.io.lakes.LakeReader"), \
             patch("pyiwfm.io.rootzone.RootZoneReader"), \
             patch("pyiwfm.components.groundwater.AppGW", return_value=mock_gw_component):
            mock_sim_reader_cls.return_value.read.return_value = sim_config
            result = load_complete_model(sim_file)

        mock_gw_reader.read_wells.assert_called_once()
        assert mock_model.groundwater == mock_gw_component

    @patch("pyiwfm.io.preprocessor.load_model_from_preprocessor")
    def test_gw_load_error_stored_in_metadata(self, mock_load_pp, tmp_path):
        """When GW loading raises Exception, error is stored in metadata."""
        sim_file = tmp_path / "simulation.in"
        sim_file.write_text("dummy")

        gw_path = tmp_path / "groundwater.dat"
        gw_path.write_text("dummy gw data")

        mock_model = MagicMock()
        mock_model.metadata = {}
        mock_load_pp.return_value = mock_model

        sim_config = _make_sim_config(
            tmp_path,
            groundwater_file=str(gw_path),
        )
        pp_file = tmp_path / "pp.in"
        pp_file.write_text("dummy pp")
        sim_config.preprocessor_file = pp_file

        # Make GW reader raise an exception
        mock_gw_reader = MagicMock()
        mock_gw_reader.read_wells.side_effect = Exception("GW read failed")

        with patch("pyiwfm.io.simulation.SimulationReader") as mock_sim_reader_cls, \
             patch("pyiwfm.io.groundwater.GroundwaterReader", return_value=mock_gw_reader), \
             patch("pyiwfm.io.streams.StreamReader"), \
             patch("pyiwfm.io.lakes.LakeReader"), \
             patch("pyiwfm.io.rootzone.RootZoneReader"):
            mock_sim_reader_cls.return_value.read.return_value = sim_config
            result = load_complete_model(sim_file)

        assert "groundwater_load_error" in mock_model.metadata
        assert "GW read failed" in mock_model.metadata["groundwater_load_error"]


class TestLoadCompleteModelStreams:
    """Tests for load_complete_model stream loading (lines 548-561)."""

    @patch("pyiwfm.io.preprocessor.load_model_from_preprocessor")
    def test_stream_component_loaded(self, mock_load_pp, tmp_path):
        """When sim_config.streams_file is set and file exists, streams are loaded."""
        sim_file = tmp_path / "simulation.in"
        sim_file.write_text("dummy")

        stream_path = tmp_path / "streams.dat"
        stream_path.write_text("dummy stream data")

        mock_model = MagicMock()
        mock_model.metadata = {}
        mock_load_pp.return_value = mock_model

        sim_config = _make_sim_config(
            tmp_path,
            streams_file=str(stream_path),
        )
        pp_file = tmp_path / "pp.in"
        pp_file.write_text("dummy pp")
        sim_config.preprocessor_file = pp_file

        mock_stream_reader = MagicMock()
        mock_stream_reader.read_stream_nodes.return_value = {1: MagicMock()}
        mock_stream_component = MagicMock()

        with patch("pyiwfm.io.simulation.SimulationReader") as mock_sim_reader_cls, \
             patch("pyiwfm.io.groundwater.GroundwaterReader"), \
             patch("pyiwfm.io.streams.StreamReader", return_value=mock_stream_reader), \
             patch("pyiwfm.io.lakes.LakeReader"), \
             patch("pyiwfm.io.rootzone.RootZoneReader"), \
             patch("pyiwfm.components.stream.AppStream", return_value=mock_stream_component):
            mock_sim_reader_cls.return_value.read.return_value = sim_config
            result = load_complete_model(sim_file)

        mock_stream_reader.read_stream_nodes.assert_called_once()
        assert mock_model.streams == mock_stream_component


class TestLoadCompleteModelLakes:
    """Tests for load_complete_model lake loading (lines 565-578)."""

    @patch("pyiwfm.io.preprocessor.load_model_from_preprocessor")
    def test_lake_component_loaded(self, mock_load_pp, tmp_path):
        """When sim_config.lakes_file is set and file exists, lakes are loaded."""
        sim_file = tmp_path / "simulation.in"
        sim_file.write_text("dummy")

        lake_path = tmp_path / "lakes.dat"
        lake_path.write_text("dummy lake data")

        mock_model = MagicMock()
        mock_model.metadata = {}
        mock_load_pp.return_value = mock_model

        sim_config = _make_sim_config(
            tmp_path,
            lakes_file=str(lake_path),
        )
        pp_file = tmp_path / "pp.in"
        pp_file.write_text("dummy pp")
        sim_config.preprocessor_file = pp_file

        mock_lake_reader = MagicMock()
        mock_lake_reader.read_lake_definitions.return_value = {1: MagicMock()}
        mock_lake_component = MagicMock()

        with patch("pyiwfm.io.simulation.SimulationReader") as mock_sim_reader_cls, \
             patch("pyiwfm.io.groundwater.GroundwaterReader"), \
             patch("pyiwfm.io.streams.StreamReader"), \
             patch("pyiwfm.io.lakes.LakeReader", return_value=mock_lake_reader), \
             patch("pyiwfm.io.rootzone.RootZoneReader"), \
             patch("pyiwfm.components.lake.AppLake", return_value=mock_lake_component):
            mock_sim_reader_cls.return_value.read.return_value = sim_config
            result = load_complete_model(sim_file)

        mock_lake_reader.read_lake_definitions.assert_called_once()
        assert mock_model.lakes == mock_lake_component


class TestLoadCompleteModelRootZone:
    """Tests for load_complete_model rootzone loading (lines 582-596)."""

    @patch("pyiwfm.io.preprocessor.load_model_from_preprocessor")
    def test_rootzone_component_loaded(self, mock_load_pp, tmp_path):
        """When sim_config.rootzone_file is set and file exists, rootzone is loaded."""
        sim_file = tmp_path / "simulation.in"
        sim_file.write_text("dummy")

        rz_path = tmp_path / "rootzone.dat"
        rz_path.write_text("dummy rootzone data")

        mock_model = MagicMock()
        mock_model.metadata = {}
        mock_model.mesh.n_elements = 10
        mock_load_pp.return_value = mock_model

        sim_config = _make_sim_config(
            tmp_path,
            rootzone_file=str(rz_path),
        )
        pp_file = tmp_path / "pp.in"
        pp_file.write_text("dummy pp")
        sim_config.preprocessor_file = pp_file

        mock_rz_reader = MagicMock()
        mock_rz_reader.read_crop_types.return_value = {1: MagicMock(), 2: MagicMock()}
        mock_rz_component = MagicMock()

        with patch("pyiwfm.io.simulation.SimulationReader") as mock_sim_reader_cls, \
             patch("pyiwfm.io.groundwater.GroundwaterReader"), \
             patch("pyiwfm.io.streams.StreamReader"), \
             patch("pyiwfm.io.lakes.LakeReader"), \
             patch("pyiwfm.io.rootzone.RootZoneReader", return_value=mock_rz_reader), \
             patch("pyiwfm.components.rootzone.RootZone", return_value=mock_rz_component):
            mock_sim_reader_cls.return_value.read.return_value = sim_config
            result = load_complete_model(sim_file)

        mock_rz_reader.read_crop_types.assert_called_once()
        assert mock_model.rootzone == mock_rz_component


class TestLoadCompleteModelPPCandidates:
    """Tests for load_complete_model pp_candidates branch (line 512)."""

    @patch("pyiwfm.io.preprocessor.load_model_from_preprocessor")
    def test_pp_candidates_glob_path(self, mock_load_pp, tmp_path):
        """When preprocessor_file is not set, find pp candidates by glob."""
        sim_file = tmp_path / "simulation.in"
        sim_file.write_text("dummy")

        # Create a file matching the *_pp.in glob pattern
        pp_candidate = tmp_path / "model_pp.in"
        pp_candidate.write_text("dummy pp")

        mock_model = MagicMock()
        mock_model.metadata = {}
        mock_load_pp.return_value = mock_model

        sim_config = _make_sim_config(tmp_path)
        # No preprocessor_file set
        sim_config.preprocessor_file = None

        with patch("pyiwfm.io.simulation.SimulationReader") as mock_sim_reader_cls, \
             patch("pyiwfm.io.groundwater.GroundwaterReader"), \
             patch("pyiwfm.io.streams.StreamReader"), \
             patch("pyiwfm.io.lakes.LakeReader"), \
             patch("pyiwfm.io.rootzone.RootZoneReader"):
            mock_sim_reader_cls.return_value.read.return_value = sim_config
            result = load_complete_model(sim_file)

        # Should have been called with the pp candidate found by glob
        mock_load_pp.assert_called_once_with(pp_candidate)
        assert result is mock_model


# =============================================================================
# save_complete_model Tests
# =============================================================================


class TestSaveCompleteModelComponents:
    """Tests for save_complete_model component writing branches."""

    @patch("pyiwfm.io.model_writer.CompleteModelWriter")
    def test_save_with_stratigraphy_and_subregions(self, mock_writer_cls, tmp_path):
        """Test that save_complete_model returns file entries from CompleteModelWriter."""
        mock_result = MagicMock()
        mock_result.files = {
            "stratigraphy": tmp_path / "strat.dat",
            "subregions": tmp_path / "subregions.dat",
        }
        mock_writer_instance = MagicMock()
        mock_writer_instance.write_all.return_value = mock_result
        mock_writer_cls.return_value = mock_writer_instance

        model = MagicMock()
        model.name = "strat_model"

        files = save_complete_model(model, tmp_path)

        assert "stratigraphy" in files
        assert files["stratigraphy"] == tmp_path / "strat.dat"
        assert "subregions" in files
        assert files["subregions"] == tmp_path / "subregions.dat"

    @patch("pyiwfm.io.model_writer.CompleteModelWriter")
    def test_save_with_groundwater(self, mock_writer_cls, tmp_path):
        """Test that save_complete_model returns GW file entries from CompleteModelWriter."""
        mock_result = MagicMock()
        mock_result.files = {
            "gw_wells": tmp_path / "gw" / "wells.dat",
        }
        mock_writer_instance = MagicMock()
        mock_writer_instance.write_all.return_value = mock_result
        mock_writer_cls.return_value = mock_writer_instance

        model = MagicMock()
        model.name = "gw_model"

        files = save_complete_model(model, tmp_path)

        assert "gw_wells" in files
        assert files["gw_wells"] == tmp_path / "gw" / "wells.dat"

    @patch("pyiwfm.io.model_writer.CompleteModelWriter")
    def test_save_with_streams(self, mock_writer_cls, tmp_path):
        """Test that save_complete_model returns stream file entries from CompleteModelWriter."""
        mock_result = MagicMock()
        mock_result.files = {
            "stream_stream_nodes": tmp_path / "streams" / "nodes.dat",
        }
        mock_writer_instance = MagicMock()
        mock_writer_instance.write_all.return_value = mock_result
        mock_writer_cls.return_value = mock_writer_instance

        model = MagicMock()
        model.name = "stream_model"

        files = save_complete_model(model, tmp_path)

        assert "stream_stream_nodes" in files
        assert files["stream_stream_nodes"] == tmp_path / "streams" / "nodes.dat"

    @patch("pyiwfm.io.model_writer.CompleteModelWriter")
    def test_save_with_lakes(self, mock_writer_cls, tmp_path):
        """Test that save_complete_model returns lake file entries from CompleteModelWriter."""
        mock_result = MagicMock()
        mock_result.files = {
            "lake_lakes": tmp_path / "lakes" / "lakes.dat",
        }
        mock_writer_instance = MagicMock()
        mock_writer_instance.write_all.return_value = mock_result
        mock_writer_cls.return_value = mock_writer_instance

        model = MagicMock()
        model.name = "lake_model"

        files = save_complete_model(model, tmp_path)

        assert "lake_lakes" in files
        assert files["lake_lakes"] == tmp_path / "lakes" / "lakes.dat"

    @patch("pyiwfm.io.model_writer.CompleteModelWriter")
    def test_save_with_rootzone(self, mock_writer_cls, tmp_path):
        """Test that save_complete_model returns rootzone file entries from CompleteModelWriter."""
        mock_result = MagicMock()
        mock_result.files = {
            "rootzone_crop_types": tmp_path / "rootzone" / "crops.dat",
        }
        mock_writer_instance = MagicMock()
        mock_writer_instance.write_all.return_value = mock_result
        mock_writer_cls.return_value = mock_writer_instance

        model = MagicMock()
        model.name = "rz_model"

        files = save_complete_model(model, tmp_path)

        assert "rootzone_crop_types" in files
        assert files["rootzone_crop_types"] == tmp_path / "rootzone" / "crops.dat"

    @patch("pyiwfm.io.model_writer.CompleteModelWriter")
    def test_save_with_dss_format(self, mock_writer_cls, tmp_path):
        """Test DSS format is passed through to CompleteModelWriter."""
        mock_result = MagicMock()
        mock_result.files = {"simulation_main": tmp_path / "sim.in"}
        mock_writer_instance = MagicMock()
        mock_writer_instance.write_all.return_value = mock_result
        mock_writer_cls.return_value = mock_writer_instance

        model = MagicMock()
        model.name = "dss_model"
        model.groundwater = None
        model.streams = None
        model.lakes = None
        model.rootzone = None

        dss_path = tmp_path / "output.dss"

        files = save_complete_model(
            model, tmp_path, timeseries_format="dss", dss_file=dss_path
        )

        # Verify the writer was called with DSS format in the config
        mock_writer_cls.assert_called_once()
        call_args = mock_writer_cls.call_args
        config = call_args[0][1]  # second positional arg is config
        from pyiwfm.io.config import OutputFormat
        assert config.ts_format == OutputFormat.DSS
        assert config.dss_file == dss_path

    @patch("pyiwfm.io.preprocessor.save_model_to_preprocessor")
    def test_save_dss_import_error_ignored(self, mock_save_pp, tmp_path):
        """Lines 714-715: ImportError when DSS not available is silently caught."""
        mock_pp_config = MagicMock()
        mock_pp_config.nodes_file = None
        mock_pp_config.elements_file = None
        mock_pp_config.stratigraphy_file = None
        mock_pp_config.subregions_file = None
        mock_save_pp.return_value = mock_pp_config

        model = MagicMock()
        model.name = "dss_fail_model"
        model.groundwater = None
        model.streams = None
        model.lakes = None
        model.rootzone = None

        dss_path = tmp_path / "output.dss"

        with patch("pyiwfm.io.simulation.SimulationFileConfig"), \
             patch("pyiwfm.io.simulation.SimulationWriter") as mock_sim_writer_cls, \
             patch("pyiwfm.io.simulation.SimulationConfig"):
            mock_sim_writer = MagicMock()
            mock_sim_writer.write.return_value = tmp_path / "sim.in"
            mock_sim_writer_cls.return_value = mock_sim_writer

            # Ensure pyiwfm.io.dss is NOT in sys.modules so import fails
            saved_module = sys.modules.pop("pyiwfm.io.dss", None)
            try:
                import builtins
                original_import = builtins.__import__

                def mock_import(name, *args, **kwargs):
                    if name == "pyiwfm.io.dss":
                        raise ImportError("No DSS library")
                    return original_import(name, *args, **kwargs)

                with patch("builtins.__import__", side_effect=mock_import):
                    files = save_complete_model(
                        model, tmp_path, timeseries_format="dss", dss_file=dss_path
                    )
            finally:
                if saved_module is not None:
                    sys.modules["pyiwfm.io.dss"] = saved_module

        # DSS not in files since import failed
        assert "dss" not in files


class TestReadSubregionsLineCoverage:
    """Test to hit line 259: continue when parts is empty in subregion reading."""

    def test_subregion_file_with_blank_data_line(self, tmp_path: Path) -> None:
        """A non-comment line that produces empty parts triggers continue (line 259).

        Since _is_comment_line returns True for whitespace-only lines, and
        split(None, 1) on any string with non-whitespace always returns at
        least one part, line 259 is effectively unreachable in normal
        operation. We test the near-boundary case to ensure robustness.
        """
        sr_file = tmp_path / "subregions.dat"
        # Include a line with just a tab that _is_comment_line will catch
        # and a normal subregion line after it
        sr_file.write_text(
            "1                               / NSUBREGION\n"
            "\t\n"
            "1  MyRegion\n"
        )
        subregions = read_subregions_file(sr_file)
        assert len(subregions) == 1
        assert subregions[1].name == "MyRegion"
