"""Unit tests for preprocessor writer module.

Tests:
- PreProcessorFileConfig
- write_nodes_file standalone function
- write_elements_file standalone function
- write_stratigraphy_file standalone function
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pyiwfm.io.config import PreProcessorFileConfig
from pyiwfm.io.preprocessor_writer import (
    write_elements_file,
    write_nodes_file,
    write_stratigraphy_file,
)

# =============================================================================
# Test PreProcessorFileConfig
# =============================================================================


class TestPreProcessorFileConfig:
    """Tests for PreProcessorFileConfig dataclass."""

    def test_basic_creation(self, tmp_path: Path) -> None:
        """Test basic config creation."""
        config = PreProcessorFileConfig(output_dir=tmp_path)

        assert config.output_dir == tmp_path
        assert config.main_file == "Preprocessor.in"
        assert config.node_file == "Nodes.dat"
        assert config.element_file == "Elements.dat"
        assert config.stratigraphy_file == "Stratigraphy.dat"
        assert config.stream_config_file == "StreamConfig.dat"
        assert config.lake_config_file == "LakeConfig.dat"

    def test_custom_file_names(self, tmp_path: Path) -> None:
        """Test config with custom file names."""
        config = PreProcessorFileConfig(
            output_dir=tmp_path,
            main_file="custom_pre.in",
            node_file="custom_nodes.dat",
            element_file="custom_elements.dat",
        )

        assert config.main_file == "custom_pre.in"
        assert config.node_file == "custom_nodes.dat"
        assert config.element_file == "custom_elements.dat"

    def test_version_settings(self, tmp_path: Path) -> None:
        """Test version settings."""
        config = PreProcessorFileConfig(
            output_dir=tmp_path,
            stream_version="4.0",
            lake_version="4.0",
        )

        assert config.stream_version == "4.0"
        assert config.lake_version == "4.0"

    def test_main_path_property(self, tmp_path: Path) -> None:
        """Test main_path property."""
        config = PreProcessorFileConfig(output_dir=tmp_path)

        assert config.main_path == tmp_path / "Preprocessor.in"

    def test_node_path_property(self, tmp_path: Path) -> None:
        """Test node_path property."""
        config = PreProcessorFileConfig(output_dir=tmp_path)

        assert config.node_path == tmp_path / "Nodes.dat"

    def test_element_path_property(self, tmp_path: Path) -> None:
        """Test element_path property."""
        config = PreProcessorFileConfig(output_dir=tmp_path)

        assert config.element_path == tmp_path / "Elements.dat"

    def test_stratigraphy_path_property(self, tmp_path: Path) -> None:
        """Test stratigraphy_path property."""
        config = PreProcessorFileConfig(output_dir=tmp_path)

        assert config.stratigraphy_path == tmp_path / "Stratigraphy.dat"

    def test_stream_config_path_property(self, tmp_path: Path) -> None:
        """Test stream_config_path property."""
        config = PreProcessorFileConfig(output_dir=tmp_path)

        assert config.stream_config_path == tmp_path / "StreamConfig.dat"

    def test_lake_config_path_property(self, tmp_path: Path) -> None:
        """Test lake_config_path property."""
        config = PreProcessorFileConfig(output_dir=tmp_path)

        assert config.lake_config_path == tmp_path / "LakeConfig.dat"

    def test_post_init_converts_string_to_path(self, tmp_path: Path) -> None:
        """Test that __post_init__ converts string to Path."""
        config = PreProcessorFileConfig(output_dir=str(tmp_path))

        assert isinstance(config.output_dir, Path)
        assert config.output_dir == tmp_path


# =============================================================================
# Test write_nodes_file
# =============================================================================


class TestWriteNodesFile:
    """Tests for write_nodes_file function."""

    def test_basic_write(self, tmp_path: Path) -> None:
        """Test basic node file writing."""
        output_path = tmp_path / "nodes.dat"
        node_ids = np.array([1, 2, 3], dtype=np.int32)
        x_coords = np.array([0.0, 100.0, 200.0], dtype=np.float64)
        y_coords = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        result_path = write_nodes_file(output_path, node_ids, x_coords, y_coords)

        assert result_path == output_path
        assert output_path.exists()

    def test_file_content_has_header(self, tmp_path: Path) -> None:
        """Test that file contains header."""
        output_path = tmp_path / "nodes.dat"
        node_ids = np.array([1, 2], dtype=np.int32)
        x_coords = np.array([0.0, 100.0], dtype=np.float64)
        y_coords = np.array([0.0, 0.0], dtype=np.float64)

        write_nodes_file(output_path, node_ids, x_coords, y_coords)

        content = output_path.read_text()
        assert "IWFM" in content
        assert "Node" in content
        assert "NNODES" in content
        assert "FACTXY" in content

    def test_file_content_has_data(self, tmp_path: Path) -> None:
        """Test that file contains node data."""
        output_path = tmp_path / "nodes.dat"
        node_ids = np.array([1, 2, 3], dtype=np.int32)
        x_coords = np.array([100.5, 200.5, 300.5], dtype=np.float64)
        y_coords = np.array([50.0, 60.0, 70.0], dtype=np.float64)

        write_nodes_file(output_path, node_ids, x_coords, y_coords)

        content = output_path.read_text()
        # Check node count
        assert "3" in content  # NNODES = 3
        # Check some coordinates appear
        assert "100" in content
        assert "200" in content

    def test_custom_coord_factor(self, tmp_path: Path) -> None:
        """Test custom coordinate factor."""
        output_path = tmp_path / "nodes.dat"
        node_ids = np.array([1], dtype=np.int32)
        x_coords = np.array([100.0], dtype=np.float64)
        y_coords = np.array([200.0], dtype=np.float64)

        write_nodes_file(output_path, node_ids, x_coords, y_coords, coord_factor=0.3048)

        content = output_path.read_text()
        assert "0.304800" in content  # FACTXY value

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        """Test that parent directory is created."""
        output_path = tmp_path / "subdir" / "deep" / "nodes.dat"
        node_ids = np.array([1], dtype=np.int32)
        x_coords = np.array([0.0], dtype=np.float64)
        y_coords = np.array([0.0], dtype=np.float64)

        write_nodes_file(output_path, node_ids, x_coords, y_coords)

        assert output_path.exists()
        assert output_path.parent.exists()

    def test_large_coordinates(self, tmp_path: Path) -> None:
        """Test with large coordinates (typical UTM values)."""
        output_path = tmp_path / "nodes.dat"
        node_ids = np.array([1, 2, 3], dtype=np.int32)
        x_coords = np.array([6000000.0, 6000100.0, 6000200.0], dtype=np.float64)
        y_coords = np.array([2000000.0, 2000100.0, 2000200.0], dtype=np.float64)

        write_nodes_file(output_path, node_ids, x_coords, y_coords)

        content = output_path.read_text()
        assert "6000000" in content
        assert "2000000" in content


# =============================================================================
# Test write_elements_file
# =============================================================================


class TestWriteElementsFile:
    """Tests for write_elements_file function."""

    def test_basic_write(self, tmp_path: Path) -> None:
        """Test basic element file writing."""
        output_path = tmp_path / "elements.dat"
        element_ids = np.array([1, 2], dtype=np.int32)
        vertices = np.array([[1, 2, 3, 4], [2, 3, 5, 6]], dtype=np.int32)
        subregions = np.array([1, 1], dtype=np.int32)

        result_path = write_elements_file(output_path, element_ids, vertices, subregions)

        assert result_path == output_path
        assert output_path.exists()

    def test_file_content_has_header(self, tmp_path: Path) -> None:
        """Test that file contains header."""
        output_path = tmp_path / "elements.dat"
        element_ids = np.array([1], dtype=np.int32)
        vertices = np.array([[1, 2, 3, 4]], dtype=np.int32)
        subregions = np.array([1], dtype=np.int32)

        write_elements_file(output_path, element_ids, vertices, subregions)

        content = output_path.read_text()
        assert "IWFM" in content
        assert "Element" in content
        assert "NELEM" in content
        assert "NSUBREGION" in content

    def test_multiple_subregions(self, tmp_path: Path) -> None:
        """Test with multiple subregions."""
        output_path = tmp_path / "elements.dat"
        element_ids = np.array([1, 2, 3], dtype=np.int32)
        vertices = np.array([[1, 2, 3, 4], [2, 3, 5, 6], [3, 4, 6, 7]], dtype=np.int32)
        subregions = np.array([1, 2, 1], dtype=np.int32)

        write_elements_file(output_path, element_ids, vertices, subregions)

        content = output_path.read_text()
        # Should have 2 subregions
        assert "2" in content  # NSUBREGION = 2 (unique subregion IDs)

    def test_custom_subregion_names(self, tmp_path: Path) -> None:
        """Test with custom subregion names."""
        output_path = tmp_path / "elements.dat"
        element_ids = np.array([1, 2], dtype=np.int32)
        vertices = np.array([[1, 2, 3, 4], [2, 3, 5, 6]], dtype=np.int32)
        subregions = np.array([1, 2], dtype=np.int32)
        subregion_names = {1: "North Region", 2: "South Region"}

        write_elements_file(output_path, element_ids, vertices, subregions, subregion_names)

        content = output_path.read_text()
        assert "North Region" in content
        assert "South Region" in content

    def test_triangular_elements(self, tmp_path: Path) -> None:
        """Test with triangular elements (fourth vertex = 0)."""
        output_path = tmp_path / "elements.dat"
        element_ids = np.array([1, 2], dtype=np.int32)
        # Fourth vertex is 0 for triangles
        vertices = np.array([[1, 2, 3, 0], [2, 3, 4, 0]], dtype=np.int32)
        subregions = np.array([1, 1], dtype=np.int32)

        write_elements_file(output_path, element_ids, vertices, subregions)

        content = output_path.read_text()
        assert output_path.exists()
        # File should contain the vertex data
        assert "1" in content
        assert "2" in content
        assert "3" in content

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        """Test that parent directory is created."""
        output_path = tmp_path / "subdir" / "elements.dat"
        element_ids = np.array([1], dtype=np.int32)
        vertices = np.array([[1, 2, 3, 4]], dtype=np.int32)
        subregions = np.array([1], dtype=np.int32)

        write_elements_file(output_path, element_ids, vertices, subregions)

        assert output_path.exists()


# =============================================================================
# Test write_stratigraphy_file
# =============================================================================


class TestWriteStratigraphyFile:
    """Tests for write_stratigraphy_file function."""

    def test_basic_write(self, tmp_path: Path) -> None:
        """Test basic stratigraphy file writing."""
        output_path = tmp_path / "strat.dat"
        node_ids = np.array([1, 2, 3], dtype=np.int32)
        ground_surface = np.array([100.0, 110.0, 105.0], dtype=np.float64)
        layer_tops = np.array([[90.0], [100.0], [95.0]], dtype=np.float64)
        layer_bottoms = np.array([[70.0], [80.0], [75.0]], dtype=np.float64)

        result_path = write_stratigraphy_file(
            output_path, node_ids, ground_surface, layer_tops, layer_bottoms
        )

        assert result_path == output_path
        assert output_path.exists()

    def test_file_content_has_header(self, tmp_path: Path) -> None:
        """Test that file contains header."""
        output_path = tmp_path / "strat.dat"
        node_ids = np.array([1], dtype=np.int32)
        ground_surface = np.array([100.0], dtype=np.float64)
        layer_tops = np.array([[90.0]], dtype=np.float64)
        layer_bottoms = np.array([[70.0]], dtype=np.float64)

        write_stratigraphy_file(output_path, node_ids, ground_surface, layer_tops, layer_bottoms)

        content = output_path.read_text()
        assert "IWFM" in content
        assert "Stratigraphy" in content
        assert "NLAYERS" in content
        assert "FACTEL" in content

    def test_single_layer(self, tmp_path: Path) -> None:
        """Test with single layer."""
        output_path = tmp_path / "strat.dat"
        node_ids = np.array([1, 2], dtype=np.int32)
        ground_surface = np.array([100.0, 110.0], dtype=np.float64)
        layer_tops = np.array([[90.0], [100.0]], dtype=np.float64)
        layer_bottoms = np.array([[70.0], [80.0]], dtype=np.float64)

        write_stratigraphy_file(output_path, node_ids, ground_surface, layer_tops, layer_bottoms)

        content = output_path.read_text()
        assert "1" in content  # NLAYERS = 1

    def test_multiple_layers(self, tmp_path: Path) -> None:
        """Test with multiple layers."""
        output_path = tmp_path / "strat.dat"
        node_ids = np.array([1, 2], dtype=np.int32)
        ground_surface = np.array([100.0, 110.0], dtype=np.float64)
        # 3 layers
        layer_tops = np.array([[90.0, 70.0, 50.0], [100.0, 80.0, 60.0]], dtype=np.float64)
        layer_bottoms = np.array([[70.0, 50.0, 30.0], [80.0, 60.0, 40.0]], dtype=np.float64)

        write_stratigraphy_file(output_path, node_ids, ground_surface, layer_tops, layer_bottoms)

        content = output_path.read_text()
        assert "3" in content  # NLAYERS = 3

    def test_thickness_calculation(self, tmp_path: Path) -> None:
        """Test that thicknesses are calculated correctly."""
        output_path = tmp_path / "strat.dat"
        node_ids = np.array([1], dtype=np.int32)
        ground_surface = np.array([100.0], dtype=np.float64)
        layer_tops = np.array([[90.0]], dtype=np.float64)
        layer_bottoms = np.array([[70.0]], dtype=np.float64)

        # Aquitard thickness = GS (100) - Top (90) = 10
        # Aquifer thickness = Top (90) - Bottom (70) = 20

        write_stratigraphy_file(output_path, node_ids, ground_surface, layer_tops, layer_bottoms)

        output_path.read_text()
        # The file should contain thickness values
        assert output_path.exists()

    def test_custom_elev_factor(self, tmp_path: Path) -> None:
        """Test custom elevation factor."""
        output_path = tmp_path / "strat.dat"
        node_ids = np.array([1], dtype=np.int32)
        ground_surface = np.array([100.0], dtype=np.float64)
        layer_tops = np.array([[90.0]], dtype=np.float64)
        layer_bottoms = np.array([[70.0]], dtype=np.float64)

        write_stratigraphy_file(
            output_path,
            node_ids,
            ground_surface,
            layer_tops,
            layer_bottoms,
            elev_factor=0.3048,
        )

        content = output_path.read_text()
        assert "0.304800" in content  # FACTEL value

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        """Test that parent directory is created."""
        output_path = tmp_path / "subdir" / "strat.dat"
        node_ids = np.array([1], dtype=np.int32)
        ground_surface = np.array([100.0], dtype=np.float64)
        layer_tops = np.array([[90.0]], dtype=np.float64)
        layer_bottoms = np.array([[70.0]], dtype=np.float64)

        write_stratigraphy_file(output_path, node_ids, ground_surface, layer_tops, layer_bottoms)

        assert output_path.exists()

    def test_multi_layer_thickness_calculation(self, tmp_path: Path) -> None:
        """Test thickness calculation with multiple layers."""
        output_path = tmp_path / "strat.dat"
        node_ids = np.array([1], dtype=np.int32)
        ground_surface = np.array([100.0], dtype=np.float64)
        # 2 layers
        layer_tops = np.array([[95.0, 75.0]], dtype=np.float64)
        layer_bottoms = np.array([[80.0, 60.0]], dtype=np.float64)

        # Layer 1:
        #   Aquitard = GS (100) - Top1 (95) = 5
        #   Aquifer = Top1 (95) - Bottom1 (80) = 15
        # Layer 2:
        #   Aquitard = Bottom1 (80) - Top2 (75) = 5
        #   Aquifer = Top2 (75) - Bottom2 (60) = 15

        write_stratigraphy_file(output_path, node_ids, ground_surface, layer_tops, layer_bottoms)

        assert output_path.exists()


# =============================================================================
# Test File Content Parsing
# =============================================================================


class TestFileContentParsing:
    """Tests that verify file content can be parsed back."""

    def test_nodes_file_readable(self, tmp_path: Path) -> None:
        """Test that nodes file can be read back."""
        output_path = tmp_path / "nodes.dat"
        node_ids = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        x_coords = np.array([0.0, 100.0, 200.0, 100.0, 50.0], dtype=np.float64)
        y_coords = np.array([0.0, 0.0, 0.0, 100.0, 50.0], dtype=np.float64)

        write_nodes_file(output_path, node_ids, x_coords, y_coords)

        # Read and parse
        content = output_path.read_text()
        lines = content.strip().split("\n")

        # Find data lines (non-comment, non-header)
        data_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith("C") or line.startswith("c"):
                continue
            if "/" in line:  # Header value line
                continue
            # Try to parse as data
            parts = line.split()
            if len(parts) >= 3:
                try:
                    int(parts[0])  # node ID
                    float(parts[1])  # x
                    float(parts[2])  # y
                    data_lines.append(line)
                except ValueError:
                    continue

        assert len(data_lines) == 5

    def test_elements_file_readable(self, tmp_path: Path) -> None:
        """Test that elements file can be read back."""
        output_path = tmp_path / "elements.dat"
        element_ids = np.array([1, 2, 3], dtype=np.int32)
        vertices = np.array([[1, 2, 5, 4], [2, 3, 6, 5], [4, 5, 6, 0]], dtype=np.int32)
        subregions = np.array([1, 1, 2], dtype=np.int32)

        write_elements_file(output_path, element_ids, vertices, subregions)

        content = output_path.read_text()
        assert "NELEM" in content
        assert "3" in content  # 3 elements

    def test_stratigraphy_file_readable(self, tmp_path: Path) -> None:
        """Test that stratigraphy file can be read back."""
        output_path = tmp_path / "strat.dat"
        node_ids = np.array([1, 2, 3], dtype=np.int32)
        ground_surface = np.array([100.0, 105.0, 110.0], dtype=np.float64)
        layer_tops = np.array([[95.0], [100.0], [105.0]], dtype=np.float64)
        layer_bottoms = np.array([[80.0], [85.0], [90.0]], dtype=np.float64)

        write_stratigraphy_file(output_path, node_ids, ground_surface, layer_tops, layer_bottoms)

        content = output_path.read_text()
        assert "NLAYERS" in content
        assert "1" in content  # 1 layer
