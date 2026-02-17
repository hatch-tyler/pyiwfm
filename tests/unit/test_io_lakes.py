"""Unit tests for lake component I/O.

Tests:
- LakeFileConfig
- LakeWriter
- LakeReader
- Convenience functions (write_lakes, read_lake_definitions, read_lake_elements)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyiwfm.io.lakes import (
    LakeFileConfig,
    LakeWriter,
    LakeReader,
    write_lakes,
    read_lake_definitions,
    read_lake_elements,
    _is_comment_line,
    _strip_comment,
)
from pyiwfm.components.lake import (
    AppLake,
    Lake,
    LakeElement,
    LakeRating,
    LakeOutflow,
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

    def test_is_comment_line_empty(self) -> None:
        """Test empty line is treated as comment."""
        assert _is_comment_line("") is True
        assert _is_comment_line("   ") is True

    def test_is_comment_line_data(self) -> None:
        """Test data line is not a comment."""
        assert _is_comment_line("1  2  3  4") is False

    def test_strip_comment_with_description(self) -> None:
        """Test parsing line with description."""
        value, desc = _strip_comment("5                / NLAKES")
        assert value == "5"
        assert desc == "NLAKES"

    def test_strip_comment_no_description(self) -> None:
        """Test parsing line without description."""
        value, desc = _strip_comment("5")
        assert value == "5"
        assert desc == ""


# =============================================================================
# Test LakeFileConfig
# =============================================================================


class TestLakeFileConfig:
    """Tests for LakeFileConfig dataclass."""

    def test_basic_creation(self, tmp_path: Path) -> None:
        """Test basic config creation."""
        config = LakeFileConfig(output_dir=tmp_path)

        assert config.output_dir == tmp_path
        assert config.lakes_file == "lakes.dat"
        assert config.lake_elements_file == "lake_elements.dat"
        assert config.rating_curves_file == "lake_rating_curves.dat"
        assert config.outflows_file == "lake_outflows.dat"

    def test_custom_file_names(self, tmp_path: Path) -> None:
        """Test config with custom file names."""
        config = LakeFileConfig(
            output_dir=tmp_path,
            lakes_file="custom_lakes.dat",
            lake_elements_file="custom_elements.dat",
        )

        assert config.lakes_file == "custom_lakes.dat"
        assert config.lake_elements_file == "custom_elements.dat"

    def test_get_lakes_path(self, tmp_path: Path) -> None:
        """Test lakes path getter."""
        config = LakeFileConfig(output_dir=tmp_path)
        path = config.get_lakes_path()

        assert path == tmp_path / "lakes.dat"

    def test_get_lake_elements_path(self, tmp_path: Path) -> None:
        """Test lake elements path getter."""
        config = LakeFileConfig(output_dir=tmp_path)
        path = config.get_lake_elements_path()

        assert path == tmp_path / "lake_elements.dat"

    def test_get_rating_curves_path(self, tmp_path: Path) -> None:
        """Test rating curves path getter."""
        config = LakeFileConfig(output_dir=tmp_path)
        path = config.get_rating_curves_path()

        assert path == tmp_path / "lake_rating_curves.dat"

    def test_get_outflows_path(self, tmp_path: Path) -> None:
        """Test outflows path getter."""
        config = LakeFileConfig(output_dir=tmp_path)
        path = config.get_outflows_path()

        assert path == tmp_path / "lake_outflows.dat"


# =============================================================================
# Test LakeWriter
# =============================================================================


class TestLakeWriter:
    """Tests for LakeWriter class."""

    @pytest.fixture
    def sample_lakes(self) -> AppLake:
        """Create sample lake component for testing."""
        lakes = {
            1: Lake(
                id=1, name="Big Lake", max_elevation=100.0,
                initial_storage=50000.0
            ),
            2: Lake(
                id=2, name="Small Lake", max_elevation=80.0,
                initial_storage=10000.0
            ),
        }
        lake_elements = [
            LakeElement(element_id=10, lake_id=1, fraction=1.0),
            LakeElement(element_id=11, lake_id=1, fraction=1.0),
            LakeElement(element_id=20, lake_id=2, fraction=0.5),
        ]
        return AppLake(lakes=lakes, lake_elements=lake_elements)

    def test_init_creates_directory(self, tmp_path: Path) -> None:
        """Test writer creates output directory."""
        output_dir = tmp_path / "new_dir" / "subdir"
        config = LakeFileConfig(output_dir=output_dir)

        LakeWriter(config)

        assert output_dir.exists()

    def test_write_lake_definitions(self, sample_lakes: AppLake, tmp_path: Path) -> None:
        """Test writing lake definitions file."""
        config = LakeFileConfig(output_dir=tmp_path)
        writer = LakeWriter(config)

        filepath = writer.write_lake_definitions(sample_lakes)

        assert filepath.exists()
        content = filepath.read_text()
        assert "NLAKES" in content
        assert "2" in content  # Number of lakes
        assert "Big Lake" in content
        assert "Small Lake" in content

    def test_write_lake_definitions_with_custom_header(
        self, sample_lakes: AppLake, tmp_path: Path
    ) -> None:
        """Test writing lake definitions with custom header."""
        config = LakeFileConfig(output_dir=tmp_path)
        writer = LakeWriter(config)

        filepath = writer.write_lake_definitions(sample_lakes, header="Custom Header")

        content = filepath.read_text()
        assert "Custom Header" in content

    def test_write_lake_elements(self, sample_lakes: AppLake, tmp_path: Path) -> None:
        """Test writing lake elements file."""
        config = LakeFileConfig(output_dir=tmp_path)
        writer = LakeWriter(config)

        filepath = writer.write_lake_elements(sample_lakes)

        assert filepath.exists()
        content = filepath.read_text()
        assert "NLAKE_ELEMENTS" in content
        assert "3" in content  # Number of elements

    def test_write_rating_curves(self, tmp_path: Path) -> None:
        """Test writing rating curves file."""
        rating = LakeRating(
            elevations=np.array([0.0, 50.0, 100.0]),
            areas=np.array([0.0, 1000.0, 5000.0]),
            volumes=np.array([0.0, 25000.0, 200000.0])
        )
        lakes = {
            1: Lake(id=1, name="Test Lake", rating=rating),
        }
        app_lake = AppLake(lakes=lakes, lake_elements=[])

        config = LakeFileConfig(output_dir=tmp_path)
        writer = LakeWriter(config)

        filepath = writer.write_rating_curves(app_lake)

        assert filepath.exists()
        content = filepath.read_text()
        assert "N_RATING_CURVES" in content
        assert "ELEVATION" in content
        assert "AREA" in content
        assert "VOLUME" in content

    def test_write_outflows(self, tmp_path: Path) -> None:
        """Test writing outflows file."""
        outflow = LakeOutflow(
            lake_id=1,
            destination_type="stream",
            destination_id=5,
            max_rate=1000.0
        )
        lakes = {
            1: Lake(id=1, name="Test Lake", outflow=outflow),
        }
        app_lake = AppLake(lakes=lakes, lake_elements=[])

        config = LakeFileConfig(output_dir=tmp_path)
        writer = LakeWriter(config)

        filepath = writer.write_outflows(app_lake)

        assert filepath.exists()
        content = filepath.read_text()
        assert "N_OUTFLOWS" in content
        assert "stream" in content
        assert "1000.0000" in content

    def test_write_all_files(self, sample_lakes: AppLake, tmp_path: Path) -> None:
        """Test writing all lake files."""
        config = LakeFileConfig(output_dir=tmp_path)
        writer = LakeWriter(config)

        files = writer.write(sample_lakes)

        assert "lakes" in files
        assert "lake_elements" in files
        # No rating curves or outflows in sample lakes
        assert "rating_curves" not in files
        assert "outflows" not in files

    def test_write_empty_lakes(self, tmp_path: Path) -> None:
        """Test writing empty lake component."""
        app_lake = AppLake(lakes={}, lake_elements=[])

        config = LakeFileConfig(output_dir=tmp_path)
        writer = LakeWriter(config)

        files = writer.write(app_lake)

        assert files == {}


# =============================================================================
# Test LakeReader
# =============================================================================


class TestLakeReader:
    """Tests for LakeReader class."""

    def test_read_lake_definitions_basic(self, tmp_path: Path) -> None:
        """Test reading basic lake definitions file."""
        lake_file = tmp_path / "lakes.dat"
        lake_file.write_text("""C Lake definitions file
C
2                               / NLAKES
1      100.0000      50000.0000  Big Lake
2       80.0000      10000.0000  Small Lake
""")

        reader = LakeReader()
        lakes = reader.read_lake_definitions(lake_file)

        assert len(lakes) == 2
        assert lakes[1].id == 1
        assert lakes[1].name == "Big Lake"
        assert lakes[1].max_elevation == pytest.approx(100.0)
        assert lakes[1].initial_storage == pytest.approx(50000.0)

        assert lakes[2].id == 2
        assert lakes[2].name == "Small Lake"

    def test_read_lake_definitions_high_max_elev(self, tmp_path: Path) -> None:
        """Test reading lake with very high max elevation (treated as inf)."""
        lake_file = tmp_path / "lakes.dat"
        lake_file.write_text("""C Lake definitions file
1                               / NLAKES
1      9999.0000      50000.0000  Unbounded Lake
""")

        reader = LakeReader()
        lakes = reader.read_lake_definitions(lake_file)

        assert lakes[1].max_elevation == float("inf")

    def test_read_lake_definitions_missing_nlakes(self, tmp_path: Path) -> None:
        """Test error when NLAKES is missing."""
        lake_file = tmp_path / "lakes.dat"
        lake_file.write_text("""C Lake definitions file
C Only comments
""")

        reader = LakeReader()

        with pytest.raises(FileFormatError, match="NLAKES"):
            reader.read_lake_definitions(lake_file)

    def test_read_lake_definitions_invalid_nlakes(self, tmp_path: Path) -> None:
        """Test error when NLAKES is invalid."""
        lake_file = tmp_path / "lakes.dat"
        lake_file.write_text("""C Lake definitions file
invalid                         / NLAKES
""")

        reader = LakeReader()

        with pytest.raises(FileFormatError, match="Invalid NLAKES"):
            reader.read_lake_definitions(lake_file)

    def test_read_lake_elements_basic(self, tmp_path: Path) -> None:
        """Test reading basic lake elements file."""
        elem_file = tmp_path / "lake_elements.dat"
        elem_file.write_text("""C Lake elements file
C
3                               / NLAKE_ELEMENTS
     10       1   1.000000
     11       1   1.000000
     20       2   0.500000
""")

        reader = LakeReader()
        elements = reader.read_lake_elements(elem_file)

        assert len(elements) == 3
        assert elements[0].element_id == 10
        assert elements[0].lake_id == 1
        assert elements[0].fraction == pytest.approx(1.0)

        assert elements[2].element_id == 20
        assert elements[2].lake_id == 2
        assert elements[2].fraction == pytest.approx(0.5)

    def test_read_lake_elements_default_fraction(self, tmp_path: Path) -> None:
        """Test reading lake elements with default fraction."""
        elem_file = tmp_path / "lake_elements.dat"
        elem_file.write_text("""C Lake elements file
1                               / NLAKE_ELEMENTS
     10       1
""")

        reader = LakeReader()
        elements = reader.read_lake_elements(elem_file)

        assert len(elements) == 1
        assert elements[0].fraction == pytest.approx(1.0)

    def test_read_lake_elements_missing_count(self, tmp_path: Path) -> None:
        """Test error when NLAKE_ELEMENTS is missing."""
        elem_file = tmp_path / "lake_elements.dat"
        elem_file.write_text("""C Lake elements file
C Only comments
""")

        reader = LakeReader()

        with pytest.raises(FileFormatError, match="NLAKE_ELEMENTS"):
            reader.read_lake_elements(elem_file)

    def test_read_rating_curves_basic(self, tmp_path: Path) -> None:
        """Test reading basic rating curves file."""
        rating_file = tmp_path / "rating_curves.dat"
        rating_file.write_text("""C Lake rating curves file
C
1                               / N_RATING_CURVES
C
C  Rating curve for lake 1
1      3  / LAKE_ID, N_POINTS
C  ELEVATION       AREA          VOLUME
      0.0000      0.0000        0.0000
     50.0000   1000.0000    25000.0000
    100.0000   5000.0000   200000.0000
""")

        reader = LakeReader()
        ratings = reader.read_rating_curves(rating_file)

        assert len(ratings) == 1
        assert 1 in ratings

        rating = ratings[1]
        assert len(rating.elevations) == 3
        assert rating.elevations[0] == pytest.approx(0.0)
        assert rating.elevations[2] == pytest.approx(100.0)
        assert rating.areas[1] == pytest.approx(1000.0)
        assert rating.volumes[2] == pytest.approx(200000.0)

    def test_read_rating_curves_multiple(self, tmp_path: Path) -> None:
        """Test reading multiple rating curves."""
        rating_file = tmp_path / "rating_curves.dat"
        rating_file.write_text("""C Lake rating curves file
2                               / N_RATING_CURVES
1      2  / LAKE_ID, N_POINTS
      0.0      0.0        0.0
     50.0   1000.0    25000.0
2      2  / LAKE_ID, N_POINTS
      0.0      0.0        0.0
     30.0    500.0    10000.0
""")

        reader = LakeReader()
        ratings = reader.read_rating_curves(rating_file)

        assert len(ratings) == 2
        assert 1 in ratings
        assert 2 in ratings


# =============================================================================
# Test Convenience Functions
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.fixture
    def sample_lakes(self) -> AppLake:
        """Create sample lake component for testing."""
        lakes = {
            1: Lake(id=1, name="Test Lake", max_elevation=100.0, initial_storage=50000.0),
        }
        lake_elements = [
            LakeElement(element_id=10, lake_id=1, fraction=1.0),
        ]
        return AppLake(lakes=lakes, lake_elements=lake_elements)

    def test_write_lakes_basic(self, sample_lakes: AppLake, tmp_path: Path) -> None:
        """Test write_lakes convenience function."""
        files = write_lakes(sample_lakes, tmp_path)

        assert "lakes" in files
        assert files["lakes"].exists()

    def test_write_lakes_with_config(
        self, sample_lakes: AppLake, tmp_path: Path
    ) -> None:
        """Test write_lakes with custom config."""
        config = LakeFileConfig(
            output_dir=tmp_path,
            lakes_file="custom_lakes.dat"
        )

        files = write_lakes(sample_lakes, tmp_path, config=config)

        assert files["lakes"].name == "custom_lakes.dat"

    def test_read_lake_definitions_function(self, tmp_path: Path) -> None:
        """Test read_lake_definitions convenience function."""
        lake_file = tmp_path / "lakes.dat"
        lake_file.write_text("""C Lake definitions
1                               / NLAKES
1      100.0      50000.0  Test Lake
""")

        lakes = read_lake_definitions(lake_file)

        assert len(lakes) == 1
        assert lakes[1].name == "Test Lake"

    def test_read_lake_elements_function(self, tmp_path: Path) -> None:
        """Test read_lake_elements convenience function."""
        elem_file = tmp_path / "lake_elements.dat"
        elem_file.write_text("""C Lake elements
1                               / NLAKE_ELEMENTS
     10       1   1.0
""")

        elements = read_lake_elements(elem_file)

        assert len(elements) == 1
        assert elements[0].element_id == 10


# =============================================================================
# Test Roundtrip
# =============================================================================


class TestRoundtrip:
    """Tests for write-read roundtrip."""

    def test_lake_definitions_roundtrip(self, tmp_path: Path) -> None:
        """Test writing and reading lake definitions."""
        original_lakes = {
            1: Lake(id=1, name="Big Lake", max_elevation=100.0, initial_storage=50000.0),
            2: Lake(id=2, name="Small Lake", max_elevation=80.0, initial_storage=10000.0),
        }
        original_app = AppLake(lakes=original_lakes, lake_elements=[])

        # Write and read back
        config = LakeFileConfig(output_dir=tmp_path)
        writer = LakeWriter(config)
        writer.write_lake_definitions(original_app)

        reader = LakeReader()
        read_lakes = reader.read_lake_definitions(config.get_lakes_path())

        # Verify
        assert len(read_lakes) == len(original_lakes)
        for lake_id in original_lakes:
            orig = original_lakes[lake_id]
            read = read_lakes[lake_id]
            assert read.id == orig.id
            assert read.name == orig.name
            assert read.max_elevation == pytest.approx(orig.max_elevation, rel=1e-4)
            assert read.initial_storage == pytest.approx(orig.initial_storage, rel=1e-4)

    def test_lake_elements_roundtrip(self, tmp_path: Path) -> None:
        """Test writing and reading lake elements."""
        original_elements = [
            LakeElement(element_id=10, lake_id=1, fraction=1.0),
            LakeElement(element_id=11, lake_id=1, fraction=0.75),
            LakeElement(element_id=20, lake_id=2, fraction=0.5),
        ]
        original_app = AppLake(lakes={}, lake_elements=original_elements)

        # Write and read back
        config = LakeFileConfig(output_dir=tmp_path)
        writer = LakeWriter(config)
        writer.write_lake_elements(original_app)

        reader = LakeReader()
        read_elements = reader.read_lake_elements(config.get_lake_elements_path())

        # Verify
        assert len(read_elements) == len(original_elements)
        for i, orig in enumerate(original_elements):
            read = read_elements[i]
            assert read.element_id == orig.element_id
            assert read.lake_id == orig.lake_id
            assert read.fraction == pytest.approx(orig.fraction, rel=1e-6)

    def test_rating_curves_roundtrip(self, tmp_path: Path) -> None:
        """Test writing and reading rating curves."""
        rating = LakeRating(
            elevations=np.array([0.0, 50.0, 100.0]),
            areas=np.array([0.0, 1000.0, 5000.0]),
            volumes=np.array([0.0, 25000.0, 200000.0])
        )
        original_lakes = {
            1: Lake(id=1, name="Test Lake", rating=rating),
        }
        original_app = AppLake(lakes=original_lakes, lake_elements=[])

        # Write and read back
        config = LakeFileConfig(output_dir=tmp_path)
        writer = LakeWriter(config)
        writer.write_rating_curves(original_app)

        reader = LakeReader()
        read_ratings = reader.read_rating_curves(config.get_rating_curves_path())

        # Verify
        assert len(read_ratings) == 1
        assert 1 in read_ratings

        read_rating = read_ratings[1]
        np.testing.assert_array_almost_equal(read_rating.elevations, rating.elevations, decimal=4)
        np.testing.assert_array_almost_equal(read_rating.areas, rating.areas, decimal=4)
        np.testing.assert_array_almost_equal(read_rating.volumes, rating.volumes, decimal=4)


# =============================================================================
# Additional tests for 95%+ coverage
# =============================================================================


class TestLakeWriterCustomHeaders:
    """Tests for LakeWriter methods with custom headers."""

    def test_write_lake_elements_custom_header(self, tmp_path: Path) -> None:
        """Test writing lake elements with custom header."""
        lake_elements = [
            LakeElement(element_id=10, lake_id=1, fraction=1.0),
        ]
        app_lake = AppLake(lakes={}, lake_elements=lake_elements)

        config = LakeFileConfig(output_dir=tmp_path)
        writer = LakeWriter(config)

        filepath = writer.write_lake_elements(app_lake, header="Custom Elements Header")

        content = filepath.read_text()
        assert "Custom Elements Header" in content

    def test_write_rating_curves_custom_header(self, tmp_path: Path) -> None:
        """Test writing rating curves with custom header."""
        rating = LakeRating(
            elevations=np.array([0.0, 50.0, 100.0]),
            areas=np.array([0.0, 1000.0, 5000.0]),
            volumes=np.array([0.0, 25000.0, 200000.0])
        )
        lakes = {1: Lake(id=1, name="Test Lake", rating=rating)}
        app_lake = AppLake(lakes=lakes, lake_elements=[])

        config = LakeFileConfig(output_dir=tmp_path)
        writer = LakeWriter(config)

        filepath = writer.write_rating_curves(app_lake, header="Custom Rating Header")

        content = filepath.read_text()
        assert "Custom Rating Header" in content

    def test_write_outflows_custom_header(self, tmp_path: Path) -> None:
        """Test writing outflows with custom header."""
        outflow = LakeOutflow(
            lake_id=1, destination_type="stream", destination_id=5, max_rate=1000.0
        )
        lakes = {1: Lake(id=1, name="Test Lake", outflow=outflow)}
        app_lake = AppLake(lakes=lakes, lake_elements=[])

        config = LakeFileConfig(output_dir=tmp_path)
        writer = LakeWriter(config)

        filepath = writer.write_outflows(app_lake, header="Custom Outflows Header")

        content = filepath.read_text()
        assert "Custom Outflows Header" in content


class TestLakeWriterInfValues:
    """Tests for LakeWriter handling of infinite values."""

    def test_write_lake_definitions_inf_max_elevation(self, tmp_path: Path) -> None:
        """Test writing lake with infinite max_elevation writes 9999.0."""
        lakes = {
            1: Lake(id=1, name="Unbounded", max_elevation=float("inf"), initial_storage=0.0),
        }
        app_lake = AppLake(lakes=lakes, lake_elements=[])

        config = LakeFileConfig(output_dir=tmp_path)
        writer = LakeWriter(config)

        filepath = writer.write_lake_definitions(app_lake)

        content = filepath.read_text()
        assert "9999.0000" in content

    def test_write_outflows_inf_max_rate(self, tmp_path: Path) -> None:
        """Test writing outflow with infinite max_rate writes 9999999.0."""
        outflow = LakeOutflow(
            lake_id=1, destination_type="stream", destination_id=5, max_rate=float("inf")
        )
        lakes = {1: Lake(id=1, name="Test Lake", outflow=outflow)}
        app_lake = AppLake(lakes=lakes, lake_elements=[])

        config = LakeFileConfig(output_dir=tmp_path)
        writer = LakeWriter(config)

        filepath = writer.write_outflows(app_lake)

        content = filepath.read_text()
        assert "9999999.0000" in content


class TestLakeWriterFullWrite:
    """Tests for LakeWriter.write() with all component types."""

    def test_write_all_with_ratings_and_outflows(self, tmp_path: Path) -> None:
        """Test write() produces all file types when data is present."""
        rating = LakeRating(
            elevations=np.array([0.0, 50.0]),
            areas=np.array([0.0, 1000.0]),
            volumes=np.array([0.0, 25000.0])
        )
        outflow = LakeOutflow(
            lake_id=1, destination_type="stream", destination_id=5, max_rate=1000.0
        )
        lakes = {
            1: Lake(id=1, name="Full Lake", rating=rating, outflow=outflow),
        }
        lake_elements = [
            LakeElement(element_id=10, lake_id=1, fraction=1.0),
        ]
        app_lake = AppLake(lakes=lakes, lake_elements=lake_elements)

        config = LakeFileConfig(output_dir=tmp_path)
        writer = LakeWriter(config)
        files = writer.write(app_lake)

        assert "lakes" in files
        assert "lake_elements" in files
        assert "rating_curves" in files
        assert "outflows" in files
        # Verify all files exist
        for path in files.values():
            assert path.exists()


class TestLakeReaderEdgeCases:
    """Additional edge case tests for LakeReader."""

    def test_read_lake_definitions_invalid_data(self, tmp_path: Path) -> None:
        """Test error when lake data contains non-numeric values."""
        lake_file = tmp_path / "lakes.dat"
        lake_file.write_text("""C Lake definitions file
1                               / NLAKES
abc    100.0    50000.0  Bad Lake
""")

        reader = LakeReader()
        with pytest.raises(FileFormatError, match="Invalid lake data"):
            reader.read_lake_definitions(lake_file)

    def test_read_lake_definitions_short_lines_skipped(self, tmp_path: Path) -> None:
        """Test that lines with fewer than 3 parts are skipped."""
        lake_file = tmp_path / "lakes.dat"
        lake_file.write_text("""C Lake definitions file
1                               / NLAKES
1 2
1      100.0000      50000.0000  Good Lake
""")

        reader = LakeReader()
        lakes = reader.read_lake_definitions(lake_file)
        assert len(lakes) == 1
        assert lakes[1].name == "Good Lake"

    def test_read_lake_definitions_no_name(self, tmp_path: Path) -> None:
        """Test reading lake definition with no name (exactly 3 parts)."""
        lake_file = tmp_path / "lakes.dat"
        lake_file.write_text("""C Lake definitions file
1                               / NLAKES
1      100.0000      50000.0000
""")

        reader = LakeReader()
        lakes = reader.read_lake_definitions(lake_file)
        assert len(lakes) == 1
        assert lakes[1].name == ""

    def test_read_lake_elements_invalid_count(self, tmp_path: Path) -> None:
        """Test error when NLAKE_ELEMENTS value is invalid."""
        elem_file = tmp_path / "lake_elements.dat"
        elem_file.write_text("""C Lake elements file
invalid                         / NLAKE_ELEMENTS
""")

        reader = LakeReader()
        with pytest.raises(FileFormatError, match="Invalid NLAKE_ELEMENTS"):
            reader.read_lake_elements(elem_file)

    def test_read_lake_elements_invalid_data(self, tmp_path: Path) -> None:
        """Test error when lake element data is invalid."""
        elem_file = tmp_path / "lake_elements.dat"
        elem_file.write_text("""C Lake elements file
1                               / NLAKE_ELEMENTS
abc    1    1.0
""")

        reader = LakeReader()
        with pytest.raises(FileFormatError, match="Invalid lake element data"):
            reader.read_lake_elements(elem_file)

    def test_read_lake_elements_short_lines_skipped(self, tmp_path: Path) -> None:
        """Test that lines with fewer than 2 parts are skipped."""
        elem_file = tmp_path / "lake_elements.dat"
        elem_file.write_text("""C Lake elements file
1                               / NLAKE_ELEMENTS
10
     10       1   1.000000
""")

        reader = LakeReader()
        elements = reader.read_lake_elements(elem_file)
        assert len(elements) == 1

    def test_read_rating_curves_missing_count(self, tmp_path: Path) -> None:
        """Test error when N_RATING_CURVES is missing."""
        rating_file = tmp_path / "rating_curves.dat"
        rating_file.write_text("""C Lake rating curves file
C Only comments
""")

        reader = LakeReader()
        with pytest.raises(FileFormatError, match="N_RATING_CURVES"):
            reader.read_rating_curves(rating_file)

    def test_read_rating_curves_invalid_count(self, tmp_path: Path) -> None:
        """Test error when N_RATING_CURVES value is invalid."""
        rating_file = tmp_path / "rating_curves.dat"
        rating_file.write_text("""C Lake rating curves file
invalid                         / N_RATING_CURVES
""")

        reader = LakeReader()
        with pytest.raises(FileFormatError, match="Invalid N_RATING_CURVES"):
            reader.read_rating_curves(rating_file)

    def test_read_lake_definitions_with_comments_between_data(
        self, tmp_path: Path
    ) -> None:
        """Test reading lake definitions with comments interspersed."""
        lake_file = tmp_path / "lakes.dat"
        lake_file.write_text("""C Lake definitions file
C Header comment
2                               / NLAKES
C First lake
1      100.0000      50000.0000  Big Lake
C Second lake
2       80.0000      10000.0000  Small Lake
""")

        reader = LakeReader()
        lakes = reader.read_lake_definitions(lake_file)
        assert len(lakes) == 2
        assert lakes[1].name == "Big Lake"
        assert lakes[2].name == "Small Lake"


class TestLakeConvenienceWithConfig:
    """Test convenience function write_lakes with explicit config."""

    def test_write_lakes_updates_config_output_dir(self, tmp_path: Path) -> None:
        """Test that write_lakes with config updates the output_dir."""
        lakes = {
            1: Lake(id=1, name="Test Lake", max_elevation=100.0, initial_storage=50000.0),
        }
        lake_elements = [
            LakeElement(element_id=10, lake_id=1, fraction=1.0),
        ]
        app_lake = AppLake(lakes=lakes, lake_elements=lake_elements)

        # Config with a different output_dir initially
        config = LakeFileConfig(output_dir=tmp_path / "old_dir")
        new_dir = tmp_path / "new_dir"
        files = write_lakes(app_lake, new_dir, config=config)

        # Config's output_dir should now be updated
        assert config.output_dir == new_dir
        assert "lakes" in files
