"""Unit tests for stream network I/O.

Tests:
- StreamFileConfig
- StreamWriter
- StreamReader
- Convenience functions (write_stream, read_stream_nodes, read_diversions)
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from pyiwfm.io.streams import (
    StreamFileConfig,
    StreamWriter,
    StreamReader,
    write_stream,
    read_stream_nodes,
    read_diversions,
    _is_comment_line,
    _strip_comment,
)
from pyiwfm.components.stream import (
    AppStream,
    StrmNode,
    StrmReach,
    Diversion,
    Bypass,
    StreamRating,
)
from pyiwfm.core.exceptions import FileFormatError


# =============================================================================
# Test Helper Functions
# =============================================================================


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_is_comment_line_c_comment(self) -> None:
        """Test C comment detection."""
        assert _is_comment_line("C This is a comment") is True
        assert _is_comment_line("c lowercase comment") is True

    def test_is_comment_line_asterisk_comment(self) -> None:
        """Test asterisk comment detection."""
        assert _is_comment_line("* This is a comment") is True

    def test_is_comment_line_hash_not_comment(self) -> None:
        """Hash is not a comment character."""
        assert _is_comment_line("# This is a comment") is False

    def test_is_comment_line_empty(self) -> None:
        """Test empty line is treated as comment."""
        assert _is_comment_line("") is True
        assert _is_comment_line("   ") is True

    def test_is_comment_line_data(self) -> None:
        """Test data line is not a comment."""
        assert _is_comment_line("1  2  3  4") is False
        assert _is_comment_line("100                / NSTRNODES") is False

    def test_strip_comment_with_description(self) -> None:
        """Test parsing line with description."""
        value, desc = _strip_comment("100                / NSTRNODES")
        assert value == "100"
        assert desc == "NSTRNODES"

    def test_strip_comment_no_description(self) -> None:
        """Test parsing line without description."""
        value, desc = _strip_comment("100")
        assert value == "100"
        assert desc == ""


# =============================================================================
# Test StreamFileConfig
# =============================================================================


class TestStreamFileConfig:
    """Tests for StreamFileConfig dataclass."""

    def test_basic_creation(self, tmp_path: Path) -> None:
        """Test basic config creation."""
        config = StreamFileConfig(output_dir=tmp_path)

        assert config.output_dir == tmp_path
        assert config.stream_nodes_file == "stream_nodes.dat"
        assert config.reaches_file == "reaches.dat"
        assert config.diversions_file == "diversions.dat"
        assert config.bypasses_file == "bypasses.dat"
        assert config.rating_curves_file == "rating_curves.dat"
        assert config.inflows_file == "stream_inflows.dat"

    def test_custom_file_names(self, tmp_path: Path) -> None:
        """Test config with custom file names."""
        config = StreamFileConfig(
            output_dir=tmp_path,
            stream_nodes_file="custom_nodes.dat",
            reaches_file="custom_reaches.dat",
        )

        assert config.stream_nodes_file == "custom_nodes.dat"
        assert config.reaches_file == "custom_reaches.dat"

    def test_get_stream_nodes_path(self, tmp_path: Path) -> None:
        """Test stream nodes path getter."""
        config = StreamFileConfig(output_dir=tmp_path)
        path = config.get_stream_nodes_path()

        assert path == tmp_path / "stream_nodes.dat"

    def test_get_reaches_path(self, tmp_path: Path) -> None:
        """Test reaches path getter."""
        config = StreamFileConfig(output_dir=tmp_path)
        path = config.get_reaches_path()

        assert path == tmp_path / "reaches.dat"

    def test_get_diversions_path(self, tmp_path: Path) -> None:
        """Test diversions path getter."""
        config = StreamFileConfig(output_dir=tmp_path)
        path = config.get_diversions_path()

        assert path == tmp_path / "diversions.dat"

    def test_get_bypasses_path(self, tmp_path: Path) -> None:
        """Test bypasses path getter."""
        config = StreamFileConfig(output_dir=tmp_path)
        path = config.get_bypasses_path()

        assert path == tmp_path / "bypasses.dat"

    def test_get_rating_curves_path(self, tmp_path: Path) -> None:
        """Test rating curves path getter."""
        config = StreamFileConfig(output_dir=tmp_path)
        path = config.get_rating_curves_path()

        assert path == tmp_path / "rating_curves.dat"

    def test_get_inflows_path(self, tmp_path: Path) -> None:
        """Test inflows path getter."""
        config = StreamFileConfig(output_dir=tmp_path)
        path = config.get_inflows_path()

        assert path == tmp_path / "stream_inflows.dat"


# =============================================================================
# Test StreamWriter
# =============================================================================


class TestStreamWriter:
    """Tests for StreamWriter class."""

    @pytest.fixture
    def sample_stream(self) -> AppStream:
        """Create sample stream component for testing."""
        nodes = {
            1: StrmNode(
                id=1, x=100.0, y=200.0, reach_id=1, gw_node=10,
                bottom_elev=50.0, wetted_perimeter=5.0,
                upstream_node=None, downstream_node=2
            ),
            2: StrmNode(
                id=2, x=150.0, y=250.0, reach_id=1, gw_node=15,
                bottom_elev=45.0, wetted_perimeter=6.0,
                upstream_node=1, downstream_node=None
            ),
        }
        reaches = {
            1: StrmReach(
                id=1, name="Main Channel", upstream_node=1, downstream_node=2,
                nodes=[1, 2], outflow_destination=("boundary", 0)
            ),
        }
        diversions = {
            1: Diversion(
                id=1, source_node=1, destination_type="element",
                destination_id=5, max_rate=100.0, priority=1, name="Div 1"
            ),
        }
        bypasses = {
            1: Bypass(
                id=1, source_node=1, destination_node=2,
                capacity=500.0, name="Bypass 1"
            ),
        }

        return AppStream(
            nodes=nodes,
            reaches=reaches,
            diversions=diversions,
            bypasses=bypasses,
        )

    def test_init_creates_directory(self, tmp_path: Path) -> None:
        """Test writer creates output directory."""
        output_dir = tmp_path / "new_dir" / "subdir"
        config = StreamFileConfig(output_dir=output_dir)

        StreamWriter(config)

        assert output_dir.exists()

    def test_write_stream_nodes(self, sample_stream: AppStream, tmp_path: Path) -> None:
        """Test writing stream nodes file."""
        config = StreamFileConfig(output_dir=tmp_path)
        writer = StreamWriter(config)

        filepath = writer.write_stream_nodes(sample_stream)

        assert filepath.exists()
        content = filepath.read_text()
        assert "NSTRNODES" in content
        assert "2" in content  # Number of nodes
        # Check node data is present
        assert "100.0000" in content
        assert "200.0000" in content

    def test_write_stream_nodes_with_custom_header(
        self, sample_stream: AppStream, tmp_path: Path
    ) -> None:
        """Test writing stream nodes with custom header."""
        config = StreamFileConfig(output_dir=tmp_path)
        writer = StreamWriter(config)

        filepath = writer.write_stream_nodes(sample_stream, header="Custom Header Line")

        content = filepath.read_text()
        assert "Custom Header Line" in content

    def test_write_reaches(self, sample_stream: AppStream, tmp_path: Path) -> None:
        """Test writing reaches file."""
        config = StreamFileConfig(output_dir=tmp_path)
        writer = StreamWriter(config)

        filepath = writer.write_reaches(sample_stream)

        assert filepath.exists()
        content = filepath.read_text()
        assert "NREACHES" in content
        assert "Main Channel" in content

    def test_write_diversions(self, sample_stream: AppStream, tmp_path: Path) -> None:
        """Test writing diversions file."""
        config = StreamFileConfig(output_dir=tmp_path)
        writer = StreamWriter(config)

        filepath = writer.write_diversions(sample_stream)

        assert filepath.exists()
        content = filepath.read_text()
        assert "NDIVERSIONS" in content
        assert "Div 1" in content
        assert "element" in content

    def test_write_bypasses(self, sample_stream: AppStream, tmp_path: Path) -> None:
        """Test writing bypasses file."""
        config = StreamFileConfig(output_dir=tmp_path)
        writer = StreamWriter(config)

        filepath = writer.write_bypasses(sample_stream)

        assert filepath.exists()
        content = filepath.read_text()
        assert "NBYPASSES" in content
        assert "Bypass 1" in content
        assert "500.0000" in content  # Capacity

    def test_write_rating_curves(self, tmp_path: Path) -> None:
        """Test writing rating curves file."""
        # Create stream with rating curve
        rating = StreamRating(
            stages=np.array([0.0, 1.0, 2.0]),
            flows=np.array([0.0, 10.0, 50.0])
        )
        nodes = {
            1: StrmNode(
                id=1, x=100.0, y=200.0, reach_id=1,
                bottom_elev=50.0, wetted_perimeter=5.0,
                rating=rating
            ),
        }
        stream = AppStream(nodes=nodes, reaches={}, diversions={}, bypasses={})

        config = StreamFileConfig(output_dir=tmp_path)
        writer = StreamWriter(config)

        filepath = writer.write_rating_curves(stream)

        assert filepath.exists()
        content = filepath.read_text()
        assert "N_RATING_CURVES" in content
        assert "STAGE" in content
        assert "FLOW" in content

    def test_write_all_files(self, sample_stream: AppStream, tmp_path: Path) -> None:
        """Test writing all stream files."""
        config = StreamFileConfig(output_dir=tmp_path)
        writer = StreamWriter(config)

        files = writer.write(sample_stream)

        assert "stream_nodes" in files
        assert "reaches" in files
        assert "diversions" in files
        assert "bypasses" in files
        # No rating curves in sample stream
        assert "rating_curves" not in files

    def test_write_empty_stream(self, tmp_path: Path) -> None:
        """Test writing empty stream component."""
        stream = AppStream(nodes={}, reaches={}, diversions={}, bypasses={})

        config = StreamFileConfig(output_dir=tmp_path)
        writer = StreamWriter(config)

        files = writer.write(stream)

        assert files == {}  # No files written for empty stream


# =============================================================================
# Test StreamReader
# =============================================================================


class TestStreamReader:
    """Tests for StreamReader class."""

    def test_read_stream_nodes_basic(self, tmp_path: Path) -> None:
        """Test reading basic stream nodes file."""
        node_file = tmp_path / "stream_nodes.dat"
        node_file.write_text("""C Stream nodes file
C
2                               / NSTRNODES
1      100.0000      200.0000     1      10      50.00     5.00       0       2
2      150.0000      250.0000     1      15      45.00     6.00       1       0
""")

        reader = StreamReader()
        nodes = reader.read_stream_nodes(node_file)

        assert len(nodes) == 2
        assert nodes[1].id == 1
        assert nodes[1].x == pytest.approx(100.0)
        assert nodes[1].y == pytest.approx(200.0)
        assert nodes[1].reach_id == 1
        assert nodes[1].gw_node == 10
        assert nodes[1].bottom_elev == pytest.approx(50.0)
        assert nodes[1].downstream_node == 2

        assert nodes[2].id == 2
        assert nodes[2].upstream_node == 1
        assert nodes[2].downstream_node is None

    def test_read_stream_nodes_with_comments(self, tmp_path: Path) -> None:
        """Test reading stream nodes with various comment styles."""
        node_file = tmp_path / "stream_nodes.dat"
        node_file.write_text("""C This is a C comment
* This is an asterisk comment
1                               / NSTRNODES
C Data row
1      100.0000      200.0000     1       0      50.00     5.00       0       0
""")

        reader = StreamReader()
        nodes = reader.read_stream_nodes(node_file)

        assert len(nodes) == 1
        assert nodes[1].id == 1

    def test_read_stream_nodes_missing_nstrnodes(self, tmp_path: Path) -> None:
        """Test error when NSTRNODES is missing."""
        node_file = tmp_path / "stream_nodes.dat"
        node_file.write_text("""C Stream nodes file
C Only comments here
""")

        reader = StreamReader()

        with pytest.raises(FileFormatError, match="NSTRNODES"):
            reader.read_stream_nodes(node_file)

    def test_read_stream_nodes_invalid_nstrnodes(self, tmp_path: Path) -> None:
        """Test error when NSTRNODES is invalid."""
        node_file = tmp_path / "stream_nodes.dat"
        node_file.write_text("""C Stream nodes file
invalid                         / NSTRNODES
""")

        reader = StreamReader()

        with pytest.raises(FileFormatError, match="Invalid NSTRNODES"):
            reader.read_stream_nodes(node_file)

    def test_read_diversions_basic(self, tmp_path: Path) -> None:
        """Test reading basic diversions file."""
        div_file = tmp_path / "diversions.dat"
        div_file.write_text("""C Diversions file
C
2                               / NDIVERSIONS
1        1 element         5      100.0000    1  Irrigation Div
2        2 subregion       3       50.0000    2  Urban Supply
""")

        reader = StreamReader()
        diversions = reader.read_diversions(div_file)

        assert len(diversions) == 2
        assert diversions[1].id == 1
        assert diversions[1].source_node == 1
        assert diversions[1].destination_type == "element"
        assert diversions[1].destination_id == 5
        assert diversions[1].max_rate == pytest.approx(100.0)
        assert diversions[1].priority == 1
        assert diversions[1].name == "Irrigation Div"

        assert diversions[2].id == 2
        assert diversions[2].destination_type == "subregion"
        assert diversions[2].name == "Urban Supply"

    def test_read_diversions_missing_ndiversions(self, tmp_path: Path) -> None:
        """Test error when NDIVERSIONS is missing."""
        div_file = tmp_path / "diversions.dat"
        div_file.write_text("""C Diversions file
C Only comments
""")

        reader = StreamReader()

        with pytest.raises(FileFormatError, match="NDIVERSIONS"):
            reader.read_diversions(div_file)


# =============================================================================
# Test Convenience Functions
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.fixture
    def sample_stream(self) -> AppStream:
        """Create sample stream component for testing."""
        nodes = {
            1: StrmNode(
                id=1, x=100.0, y=200.0, reach_id=1,
                bottom_elev=50.0, wetted_perimeter=5.0,
            ),
        }
        reaches = {
            1: StrmReach(
                id=1, name="Main", upstream_node=1, downstream_node=1,
                nodes=[1], outflow_destination=("boundary", 0)
            ),
        }
        return AppStream(nodes=nodes, reaches=reaches, diversions={}, bypasses={})

    def test_write_stream_basic(self, sample_stream: AppStream, tmp_path: Path) -> None:
        """Test write_stream convenience function."""
        files = write_stream(sample_stream, tmp_path)

        assert "stream_nodes" in files
        assert files["stream_nodes"].exists()

    def test_write_stream_with_config(
        self, sample_stream: AppStream, tmp_path: Path
    ) -> None:
        """Test write_stream with custom config."""
        config = StreamFileConfig(
            output_dir=tmp_path,
            stream_nodes_file="custom_nodes.dat"
        )

        files = write_stream(sample_stream, tmp_path, config=config)

        assert files["stream_nodes"].name == "custom_nodes.dat"

    def test_read_stream_nodes_function(self, tmp_path: Path) -> None:
        """Test read_stream_nodes convenience function."""
        node_file = tmp_path / "nodes.dat"
        node_file.write_text("""C Stream nodes
1                               / NSTRNODES
1      100.0      200.0     1       0      50.0      5.0       0       0
""")

        nodes = read_stream_nodes(node_file)

        assert len(nodes) == 1
        assert nodes[1].x == pytest.approx(100.0)

    def test_read_diversions_function(self, tmp_path: Path) -> None:
        """Test read_diversions convenience function."""
        div_file = tmp_path / "diversions.dat"
        div_file.write_text("""C Diversions
1                               / NDIVERSIONS
1        1 element         5      100.0    1  Test Div
""")

        diversions = read_diversions(div_file)

        assert len(diversions) == 1
        assert diversions[1].name == "Test Div"


# =============================================================================
# Test Roundtrip
# =============================================================================


class TestRoundtrip:
    """Tests for write-read roundtrip."""

    def test_stream_nodes_roundtrip(self, tmp_path: Path) -> None:
        """Test writing and reading stream nodes."""
        # Create original nodes
        original_nodes = {
            1: StrmNode(
                id=1, x=100.0, y=200.0, reach_id=1, gw_node=10,
                bottom_elev=50.0, wetted_perimeter=5.0,
                upstream_node=None, downstream_node=2
            ),
            2: StrmNode(
                id=2, x=150.0, y=250.0, reach_id=1, gw_node=15,
                bottom_elev=45.0, wetted_perimeter=6.0,
                upstream_node=1, downstream_node=None
            ),
        }
        original_stream = AppStream(
            nodes=original_nodes, reaches={}, diversions={}, bypasses={}
        )

        # Write and read back
        config = StreamFileConfig(output_dir=tmp_path)
        writer = StreamWriter(config)
        writer.write_stream_nodes(original_stream)

        reader = StreamReader()
        read_nodes = reader.read_stream_nodes(config.get_stream_nodes_path())

        # Verify
        assert len(read_nodes) == len(original_nodes)
        for node_id in original_nodes:
            orig = original_nodes[node_id]
            read = read_nodes[node_id]
            assert read.id == orig.id
            assert read.x == pytest.approx(orig.x, rel=1e-4)
            assert read.y == pytest.approx(orig.y, rel=1e-4)
            assert read.reach_id == orig.reach_id
            assert read.bottom_elev == pytest.approx(orig.bottom_elev, rel=1e-2)

    def test_diversions_roundtrip(self, tmp_path: Path) -> None:
        """Test writing and reading diversions."""
        # Create original diversions
        original_diversions = {
            1: Diversion(
                id=1, source_node=1, destination_type="element",
                destination_id=5, max_rate=100.0, priority=1, name="Div1"
            ),
            2: Diversion(
                id=2, source_node=2, destination_type="subregion",
                destination_id=3, max_rate=50.0, priority=2, name="Div2"
            ),
        }
        original_stream = AppStream(
            nodes={}, reaches={}, diversions=original_diversions, bypasses={}
        )

        # Write and read back
        config = StreamFileConfig(output_dir=tmp_path)
        writer = StreamWriter(config)
        writer.write_diversions(original_stream)

        reader = StreamReader()
        read_diversions = reader.read_diversions(config.get_diversions_path())

        # Verify
        assert len(read_diversions) == len(original_diversions)
        for div_id in original_diversions:
            orig = original_diversions[div_id]
            read = read_diversions[div_id]
            assert read.id == orig.id
            assert read.source_node == orig.source_node
            assert read.destination_type == orig.destination_type
            assert read.destination_id == orig.destination_id
            assert read.max_rate == pytest.approx(orig.max_rate, rel=1e-4)
            assert read.priority == orig.priority


# =============================================================================
# Additional tests for 95%+ coverage
# =============================================================================


class TestStreamWriterCustomHeaders:
    """Tests for StreamWriter methods with custom headers."""

    @pytest.fixture
    def sample_stream(self) -> AppStream:
        """Create sample stream component for testing."""
        nodes = {
            1: StrmNode(
                id=1, x=100.0, y=200.0, reach_id=1, gw_node=10,
                bottom_elev=50.0, wetted_perimeter=5.0,
                upstream_node=None, downstream_node=2
            ),
            2: StrmNode(
                id=2, x=150.0, y=250.0, reach_id=1, gw_node=15,
                bottom_elev=45.0, wetted_perimeter=6.0,
                upstream_node=1, downstream_node=None
            ),
        }
        reaches = {
            1: StrmReach(
                id=1, name="Main Channel", upstream_node=1, downstream_node=2,
                nodes=[1, 2], outflow_destination=("boundary", 0)
            ),
        }
        diversions = {
            1: Diversion(
                id=1, source_node=1, destination_type="element",
                destination_id=5, max_rate=100.0, priority=1, name="Div 1"
            ),
        }
        bypasses = {
            1: Bypass(
                id=1, source_node=1, destination_node=2,
                capacity=500.0, name="Bypass 1"
            ),
        }
        return AppStream(
            nodes=nodes, reaches=reaches, diversions=diversions, bypasses=bypasses,
        )

    def test_write_reaches_custom_header(
        self, sample_stream: AppStream, tmp_path: Path
    ) -> None:
        """Test writing reaches with custom header."""
        config = StreamFileConfig(output_dir=tmp_path)
        writer = StreamWriter(config)

        filepath = writer.write_reaches(sample_stream, header="Custom Reaches Header")

        content = filepath.read_text()
        assert "Custom Reaches Header" in content

    def test_write_diversions_custom_header(
        self, sample_stream: AppStream, tmp_path: Path
    ) -> None:
        """Test writing diversions with custom header."""
        config = StreamFileConfig(output_dir=tmp_path)
        writer = StreamWriter(config)

        filepath = writer.write_diversions(sample_stream, header="Custom Div Header")

        content = filepath.read_text()
        assert "Custom Div Header" in content

    def test_write_bypasses_custom_header(
        self, sample_stream: AppStream, tmp_path: Path
    ) -> None:
        """Test writing bypasses with custom header."""
        config = StreamFileConfig(output_dir=tmp_path)
        writer = StreamWriter(config)

        filepath = writer.write_bypasses(sample_stream, header="Custom Bypass Header")

        content = filepath.read_text()
        assert "Custom Bypass Header" in content

    def test_write_rating_curves_custom_header(self, tmp_path: Path) -> None:
        """Test writing rating curves with custom header."""
        rating = StreamRating(
            stages=np.array([0.0, 1.0, 2.0]),
            flows=np.array([0.0, 10.0, 50.0])
        )
        nodes = {
            1: StrmNode(
                id=1, x=100.0, y=200.0, reach_id=1,
                bottom_elev=50.0, wetted_perimeter=5.0, rating=rating
            ),
        }
        stream = AppStream(nodes=nodes, reaches={}, diversions={}, bypasses={})

        config = StreamFileConfig(output_dir=tmp_path)
        writer = StreamWriter(config)

        filepath = writer.write_rating_curves(stream, header="Custom Rating Header")

        content = filepath.read_text()
        assert "Custom Rating Header" in content


class TestStreamWriterAllFiles:
    """Tests for write() including all file types."""

    def test_write_with_rating_curves(self, tmp_path: Path) -> None:
        """Test write() produces rating_curves file when nodes have ratings."""
        rating = StreamRating(
            stages=np.array([0.0, 1.0, 2.0]),
            flows=np.array([0.0, 10.0, 50.0])
        )
        nodes = {
            1: StrmNode(
                id=1, x=100.0, y=200.0, reach_id=1, gw_node=10,
                bottom_elev=50.0, wetted_perimeter=5.0,
                upstream_node=None, downstream_node=None,
                rating=rating
            ),
        }
        reaches = {
            1: StrmReach(
                id=1, name="Main", upstream_node=1, downstream_node=1,
                nodes=[1], outflow_destination=("boundary", 0)
            ),
        }
        stream = AppStream(nodes=nodes, reaches=reaches, diversions={}, bypasses={})

        config = StreamFileConfig(output_dir=tmp_path)
        writer = StreamWriter(config)
        files = writer.write(stream)

        assert "stream_nodes" in files
        assert "reaches" in files
        assert "rating_curves" in files


class TestStreamReaderEdgeCases:
    """Tests for StreamReader error paths."""

    def test_read_stream_nodes_invalid_data(self, tmp_path: Path) -> None:
        """Test error when node data contains non-numeric values."""
        node_file = tmp_path / "stream_nodes.dat"
        node_file.write_text("""C Stream nodes file
1                               / NSTRNODES
abc    100.0    200.0    1    0    50.0    5.0    0    0
""")

        reader = StreamReader()
        with pytest.raises(FileFormatError, match="Invalid stream node data"):
            reader.read_stream_nodes(node_file)

    def test_read_stream_nodes_short_lines_skipped(self, tmp_path: Path) -> None:
        """Test that lines with fewer than 6 parts are skipped."""
        node_file = tmp_path / "stream_nodes.dat"
        node_file.write_text("""C Stream nodes file
1                               / NSTRNODES
1 2 3
1      100.0000      200.0000     1      10      50.00     5.00       0       0
""")

        reader = StreamReader()
        nodes = reader.read_stream_nodes(node_file)
        assert len(nodes) == 1
        assert nodes[1].id == 1

    def test_read_diversions_invalid_ndiversions(self, tmp_path: Path) -> None:
        """Test error when NDIVERSIONS value is invalid."""
        div_file = tmp_path / "diversions.dat"
        div_file.write_text("""C Diversions file
invalid                         / NDIVERSIONS
""")

        reader = StreamReader()
        with pytest.raises(FileFormatError, match="Invalid NDIVERSIONS"):
            reader.read_diversions(div_file)

    def test_read_diversions_invalid_data(self, tmp_path: Path) -> None:
        """Test error when diversion data is invalid."""
        div_file = tmp_path / "diversions.dat"
        div_file.write_text("""C Diversions file
1                               / NDIVERSIONS
abc    1 element    5    100.0    1    Test
""")

        reader = StreamReader()
        with pytest.raises(FileFormatError, match="Invalid diversion data"):
            reader.read_diversions(div_file)

    def test_read_diversions_short_lines_skipped(self, tmp_path: Path) -> None:
        """Test that lines with fewer than 6 parts are skipped."""
        div_file = tmp_path / "diversions.dat"
        div_file.write_text("""C Diversions file
1                               / NDIVERSIONS
1 2 3
1        1 element         5      100.0000    1  Div1
""")

        reader = StreamReader()
        diversions = reader.read_diversions(div_file)
        assert len(diversions) == 1

    def test_read_diversions_no_name(self, tmp_path: Path) -> None:
        """Test reading diversion with no name (fewer than 7 parts)."""
        div_file = tmp_path / "diversions.dat"
        div_file.write_text("""C Diversions file
1                               / NDIVERSIONS
1        1 element         5      100.0000    1
""")

        reader = StreamReader()
        diversions = reader.read_diversions(div_file)
        assert len(diversions) == 1
        assert diversions[1].name == ""


class TestStreamWriterManyReachNodes:
    """Test reach node writing with more than 10 nodes (line wrapping)."""

    def test_write_reaches_many_nodes(self, tmp_path: Path) -> None:
        """Test writing reaches with >10 nodes triggers line wrapping."""
        # Create 15 nodes
        node_ids = list(range(1, 16))
        nodes = {
            nid: StrmNode(
                id=nid, x=float(nid * 10), y=float(nid * 20),
                reach_id=1, bottom_elev=50.0, wetted_perimeter=5.0,
            )
            for nid in node_ids
        }
        reaches = {
            1: StrmReach(
                id=1, name="Long Reach", upstream_node=1, downstream_node=15,
                nodes=node_ids, outflow_destination=("boundary", 0)
            ),
        }
        stream = AppStream(nodes=nodes, reaches=reaches, diversions={}, bypasses={})

        config = StreamFileConfig(output_dir=tmp_path)
        writer = StreamWriter(config)
        filepath = writer.write_reaches(stream)

        content = filepath.read_text()
        assert "Long Reach" in content
        # With 15 nodes, should have nodes on multiple lines
        assert "15" in content


class TestStreamWriterReachNoOutflow:
    """Test reach writing when outflow_destination is empty."""

    def test_write_reaches_no_outflow_destination(self, tmp_path: Path) -> None:
        """Test writing reaches when outflow_destination is empty/None."""
        nodes = {
            1: StrmNode(
                id=1, x=100.0, y=200.0, reach_id=1,
                bottom_elev=50.0, wetted_perimeter=5.0,
            ),
        }
        reaches = {
            1: StrmReach(
                id=1, name="Dead End", upstream_node=1, downstream_node=1,
                nodes=[1], outflow_destination=None
            ),
        }
        stream = AppStream(nodes=nodes, reaches=reaches, diversions={}, bypasses={})

        config = StreamFileConfig(output_dir=tmp_path)
        writer = StreamWriter(config)
        filepath = writer.write_reaches(stream)

        content = filepath.read_text()
        assert "boundary" in content
        assert "Dead End" in content


class TestStreamWriterInflows:
    """Tests for write_inflows_timeseries method."""

    def test_write_inflows_basic(self, tmp_path: Path) -> None:
        """Test writing inflows time series with explicit node_ids."""
        config = StreamFileConfig(output_dir=tmp_path)
        writer = StreamWriter(config)

        times = [datetime(2020, 1, 1), datetime(2020, 2, 1)]
        inflows = {
            1: np.array([100.0, 110.0]),
            2: np.array([200.0, 220.0]),
        }

        filepath = writer.write_inflows_timeseries(
            filepath=tmp_path / "inflows.dat",
            times=times,
            inflows=inflows,
            node_ids=[1, 2],
        )

        assert filepath.exists()

    def test_write_inflows_default_node_ids(self, tmp_path: Path) -> None:
        """Test writing inflows with default (sorted) node_ids."""
        config = StreamFileConfig(output_dir=tmp_path)
        writer = StreamWriter(config)

        times = [datetime(2020, 1, 1)]
        inflows = {
            3: np.array([300.0]),
            1: np.array([100.0]),
        }

        filepath = writer.write_inflows_timeseries(
            filepath=tmp_path / "inflows.dat",
            times=times,
            inflows=inflows,
        )

        assert filepath.exists()

    def test_write_inflows_with_missing_node(self, tmp_path: Path) -> None:
        """Test writing inflows where node_ids includes a node not in inflows dict."""
        config = StreamFileConfig(output_dir=tmp_path)
        writer = StreamWriter(config)

        times = [datetime(2020, 1, 1)]
        inflows = {
            1: np.array([100.0]),
        }

        filepath = writer.write_inflows_timeseries(
            filepath=tmp_path / "inflows.dat",
            times=times,
            inflows=inflows,
            node_ids=[1, 99],  # node 99 not in inflows
        )

        assert filepath.exists()
