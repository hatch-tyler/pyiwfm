"""Supplementary tests for lakes.py targeting uncovered branches.

Covers:
- LakeWriter additional methods
- LakeReader edge cases
- Roundtrip with rating curves and outflows
- Empty lake component writing
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyiwfm.components.lake import (
    AppLake,
    Lake,
    LakeOutflow,
    LakeRating,
)
from pyiwfm.io.lakes import (
    LakeFileConfig,
    LakeReader,
    LakeWriter,
    write_lakes,
)

# =============================================================================
# LakeWriter Additional Tests
# =============================================================================


class TestLakeWriterAdditional:
    """Additional tests for LakeWriter."""

    @pytest.fixture
    def sample_lake_with_ratings(self) -> AppLake:
        """Create lake component with rating curves."""
        rating = LakeRating(
            elevations=np.array([90.0, 95.0, 100.0]),
            areas=np.array([0.0, 5000.0, 20000.0]),
            volumes=np.array([0.0, 12500.0, 75000.0]),
        )
        lake = Lake(
            id=1,
            name="Test Lake",
            max_elevation=100.0,
            elements=[1, 2],
            rating=rating,
            outflow=LakeOutflow(
                lake_id=1,
                destination_type="stream",
                destination_id=5,
                max_rate=1000.0,
            ),
        )
        return AppLake(lakes={1: lake})

    def test_write_lake_with_ratings(
        self, sample_lake_with_ratings: AppLake, tmp_path: Path
    ) -> None:
        """Test writing lake with rating curves."""
        config = LakeFileConfig(output_dir=tmp_path)
        writer = LakeWriter(config)

        files = writer.write(sample_lake_with_ratings)

        assert "lakes" in files
        assert "rating_curves" in files
        assert "outflows" in files

    def test_write_lake_definitions_content(
        self, sample_lake_with_ratings: AppLake, tmp_path: Path
    ) -> None:
        """Test lake definitions file content."""
        config = LakeFileConfig(output_dir=tmp_path)
        writer = LakeWriter(config)

        filepath = writer.write_lake_definitions(sample_lake_with_ratings)

        content = filepath.read_text()
        assert "NLAKES" in content
        assert "Test Lake" in content
        assert "100.0" in content

    def test_write_multiple_lakes(self, tmp_path: Path) -> None:
        """Test writing multiple lakes."""
        lakes = {
            1: Lake(
                id=1,
                name="Lake A",
                max_elevation=100.0,
                elements=[1],
            ),
            2: Lake(
                id=2,
                name="Lake B",
                max_elevation=95.0,
                elements=[2, 3],
            ),
        }
        app_lake = AppLake(lakes=lakes)

        config = LakeFileConfig(output_dir=tmp_path)
        writer = LakeWriter(config)

        files = writer.write(app_lake)

        assert "lakes" in files
        content = files["lakes"].read_text()
        assert "Lake A" in content
        assert "Lake B" in content

    def test_write_empty_lakes(self, tmp_path: Path) -> None:
        """Test writing empty lake component."""
        app_lake = AppLake(lakes={})

        config = LakeFileConfig(output_dir=tmp_path)
        writer = LakeWriter(config)

        files = writer.write(app_lake)
        assert files == {}


# =============================================================================
# LakeReader Additional Tests
# =============================================================================


class TestLakeReaderAdditional:
    """Additional tests for LakeReader."""

    def test_read_lake_definitions_with_max_elev(self, tmp_path: Path) -> None:
        """Test reading lake definitions with high max elevation."""
        filepath = tmp_path / "lakes.dat"
        # Format: id  max_elev  initial_storage  name
        filepath.write_text("""C  Lake definitions file
C
2                               / NLAKES
1      500.0000    0.0    Lake Alpha
2      999.9999    0.0    Lake Beta
""")

        reader = LakeReader()
        lakes = reader.read_lake_definitions(filepath)

        assert len(lakes) == 2
        assert lakes[1].name == "Lake Alpha"
        assert lakes[1].max_elevation == pytest.approx(500.0)
        assert lakes[2].name == "Lake Beta"
        assert lakes[2].max_elevation == pytest.approx(999.9999)

    def test_read_lake_definitions_with_comments(self, tmp_path: Path) -> None:
        """Test reading lake definitions with interspersed comments."""
        filepath = tmp_path / "lakes.dat"
        # Format: id  max_elev  initial_storage  name
        filepath.write_text("""C  Lake definitions
*  asterisk comment
1                               / NLAKES
C  Lake data
1      100.0    0.0    Test Lake
""")

        reader = LakeReader()
        lakes = reader.read_lake_definitions(filepath)

        assert len(lakes) == 1
        assert lakes[1].name == "Test Lake"


# =============================================================================
# Convenience Functions Additional Tests
# =============================================================================


class TestLakeConvenienceAdditional:
    """Additional convenience function tests."""

    def test_write_lakes_string_path(self, tmp_path: Path) -> None:
        """Test write_lakes with string output directory."""
        lakes = {
            1: Lake(
                id=1,
                name="StringPath Lake",
                max_elevation=100.0,
                elements=[1],
            ),
        }
        app_lake = AppLake(lakes=lakes)

        files = write_lakes(app_lake, str(tmp_path))

        assert "lakes" in files
        assert files["lakes"].exists()

    def test_write_lakes_custom_config(self, tmp_path: Path) -> None:
        """Test write_lakes with custom config."""
        lakes = {
            1: Lake(
                id=1,
                name="Custom Config Lake",
                max_elevation=100.0,
                elements=[1],
            ),
        }
        app_lake = AppLake(lakes=lakes)

        config = LakeFileConfig(
            output_dir=tmp_path,
            lakes_file="custom_lakes.dat",
        )

        files = write_lakes(app_lake, tmp_path, config=config)

        assert files["lakes"].name == "custom_lakes.dat"
