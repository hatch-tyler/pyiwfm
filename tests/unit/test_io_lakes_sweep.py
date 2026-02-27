"""Sweep tests for pyiwfm.io.lakes targeting uncovered lines.

Covers:
- Lines 316-318: Version header (#4.0) skip in read_lake_definitions
- Lines 348-424: Preprocessor format lake reading (single and multi-lake)
- Lines 486: Comment line in read_lake_elements data section
- Lines 776-777: LakeMainFileReader v5.0 elev_factor ValueError
- Lines 791-792: LakeMainFileReader lake_params ValueError/IndexError break
- Lines 827, 831: _read_outflow_rating early returns (None)
- Lines 848: _read_outflow_rating break on empty line
- Lines 866: _read_version blank line skip
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pyiwfm.components.lake import AppLake, Lake, LakeElement, LakeOutflow
from pyiwfm.io.lakes import (
    LakeMainFileConfig,
    LakeMainFileReader,
    LakeReader,
    read_lake_main_file,
)


# =============================================================================
# Preprocessor-format lake definitions
# =============================================================================


class TestLakeReaderPreprocessorFormat:
    """Tests for LakeReader reading preprocessor Lake.dat format."""

    def test_single_lake_preprocessor_format(self, tmp_path: Path) -> None:
        """Read a single lake in preprocessor format (TYPDST DST NELAKE IELAKE)."""
        lake_file = tmp_path / "lakes.dat"
        # Preprocessor format: ID TYPDST DST NELAKE IELAKE
        # Lake 1: dest type 1 (stream), dest 5, 2 elements: 10, 11
        lake_file.write_text(
            "C  Lake preprocessor file\n"
            "1                               / NLAKES\n"
            "1      1       5       2      10\n"
            "11\n"
        )

        reader = LakeReader()
        lakes = reader.read_lake_definitions(lake_file)

        assert len(lakes) == 1
        assert 1 in lakes
        lake = lakes[1]
        assert lake.id == 1
        assert lake.elements == [10, 11]
        # TYPDST=1 means outflow to stream
        assert lake.outflow is not None
        assert lake.outflow.destination_type == "stream"
        assert lake.outflow.destination_id == 5

    def test_single_lake_preprocessor_no_outflow(self, tmp_path: Path) -> None:
        """Preprocessor lake with TYPDST != 1 gets no outflow."""
        lake_file = tmp_path / "lakes.dat"
        # TYPDST=0 means no outflow
        lake_file.write_text(
            "C  Lake preprocessor file\n"
            "1                               / NLAKES\n"
            "1      0       0       1      10\n"
        )

        reader = LakeReader()
        lakes = reader.read_lake_definitions(lake_file)

        assert len(lakes) == 1
        assert lakes[1].outflow is None

    def test_multi_lake_preprocessor_format(self, tmp_path: Path) -> None:
        """Read multiple lakes in preprocessor format with continuation rows."""
        lake_file = tmp_path / "lakes.dat"
        # 2 lakes: Lake 1 has 2 elements (10, 11), Lake 2 has 3 elements (20, 21, 22)
        lake_file.write_text(
            "C  Lake preprocessor file\n"
            "2                               / NLAKES\n"
            "1      1       5       2      10\n"
            "11\n"
            "2      1       8       3      20\n"
            "21\n"
            "22\n"
        )

        reader = LakeReader()
        lakes = reader.read_lake_definitions(lake_file)

        assert len(lakes) == 2
        assert lakes[1].elements == [10, 11]
        assert lakes[1].outflow is not None
        assert lakes[1].outflow.destination_id == 5

        assert lakes[2].elements == [20, 21, 22]
        assert lakes[2].outflow is not None
        assert lakes[2].outflow.destination_id == 8

    def test_preprocessor_with_comments_in_continuation(self, tmp_path: Path) -> None:
        """Preprocessor format with comment lines between continuation rows."""
        lake_file = tmp_path / "lakes.dat"
        lake_file.write_text(
            "C  Lake preprocessor file\n"
            "1                               / NLAKES\n"
            "1      1       3       3      10\n"
            "C  continuation element 2\n"
            "11\n"
            "C  continuation element 3\n"
            "12\n"
        )

        reader = LakeReader()
        lakes = reader.read_lake_definitions(lake_file)

        assert len(lakes) == 1
        assert lakes[1].elements == [10, 11, 12]

    def test_preprocessor_single_element_lake(self, tmp_path: Path) -> None:
        """Preprocessor lake with NELAKE=1 (no continuation rows)."""
        lake_file = tmp_path / "lakes.dat"
        lake_file.write_text(
            "C  Lake preprocessor file\n"
            "1                               / NLAKES\n"
            "1      0       0       1      10\n"
        )

        reader = LakeReader()
        lakes = reader.read_lake_definitions(lake_file)

        assert len(lakes) == 1
        assert lakes[1].elements == [10]

    def test_preprocessor_no_first_elem(self, tmp_path: Path) -> None:
        """Preprocessor lake where first element is 0 (empty element list start)."""
        lake_file = tmp_path / "lakes.dat"
        # Only 4 parts (no IELAKE column), which triggers len(parts) > 4 = False
        lake_file.write_text(
            "C  Lake preprocessor file\n"
            "1                               / NLAKES\n"
            "1      0       0       1\n"
        )

        reader = LakeReader()
        lakes = reader.read_lake_definitions(lake_file)

        assert len(lakes) == 1
        # first_elem would be 0 since len(parts) == 4
        assert lakes[1].elements == []


class TestLakeReaderVersionHeader:
    """Tests for LakeReader handling of version headers (#4.0)."""

    def test_version_header_skipped(self, tmp_path: Path) -> None:
        """Version header line (#4.0) is properly skipped."""
        lake_file = tmp_path / "lakes.dat"
        lake_file.write_text(
            "#4.0\n"
            "C  Lake definitions\n"
            "1                               / NLAKES\n"
            "1      100.0      50000.0  Test Lake\n"
        )

        reader = LakeReader()
        lakes = reader.read_lake_definitions(lake_file)

        assert len(lakes) == 1
        assert lakes[1].name == "Test Lake"

    def test_multiple_version_and_comment_lines(self, tmp_path: Path) -> None:
        """Multiple hash-prefixed lines before data."""
        lake_file = tmp_path / "lakes.dat"
        lake_file.write_text(
            "#4.0\n"
            "#IWFM Lake file\n"
            "C  Some description\n"
            "1                               / NLAKES\n"
            "1      100.0      0.0  My Lake\n"
        )

        reader = LakeReader()
        lakes = reader.read_lake_definitions(lake_file)

        assert len(lakes) == 1
        assert lakes[1].name == "My Lake"


class TestLakeElementsCommentInData:
    """Test comment lines inside element data section (line 486)."""

    def test_comment_between_elements(self, tmp_path: Path) -> None:
        """Comment lines between element data rows are skipped."""
        elem_file = tmp_path / "lake_elements.dat"
        elem_file.write_text(
            "C  Lake elements file\n"
            "2                               / NLAKE_ELEMENTS\n"
            "     10       1   1.000000\n"
            "C  Next element\n"
            "     11       1   0.500000\n"
        )

        reader = LakeReader()
        elements = reader.read_lake_elements(elem_file)

        assert len(elements) == 2
        assert elements[0].element_id == 10
        assert elements[1].element_id == 11


# =============================================================================
# LakeMainFileReader v4.0 and v5.0
# =============================================================================


class TestLakeMainFileReaderV40:
    """Tests for LakeMainFileReader with v4.0 format."""

    def test_basic_v40_read(self, tmp_path: Path) -> None:
        """Read a basic v4.0 lake main file."""
        lake_main = tmp_path / "lake_main.dat"
        lake_main.write_text(
            "C  Lake component main file\n"
            "#4.0\n"
            "max_elev.dat                       / MAXELEVFL\n"
            "lake_budget.dat                    / BUDGETFL\n"
            "lake_final_elev.dat                / FINALFL\n"
            "1.0                                / FACTK\n"
            "1DAY                               / TUNITK\n"
            "1.0                                / FACTL\n"
            "1     0.5     1.0    1    7    2   Lake Alpha\n"
        )

        reader = LakeMainFileReader()
        config = reader.read(lake_main)

        assert config.version == "4.0"
        assert config.max_elev_file is not None
        assert config.budget_output_file is not None
        assert config.final_elev_file is not None
        assert config.conductance_factor == 1.0
        assert config.conductance_time_unit == "1DAY"
        assert config.depth_factor == 1.0
        assert len(config.lake_params) == 1

        param = config.lake_params[0]
        assert param.lake_id == 1
        assert param.conductance_coeff == 0.5
        assert param.depth_denom == 1.0
        assert param.max_elev_col == 1
        assert param.et_col == 7
        assert param.precip_col == 2
        assert param.name == "Lake Alpha"

    def test_v40_multiple_lakes(self, tmp_path: Path) -> None:
        """Read v4.0 lake main file with multiple lake parameter lines."""
        lake_main = tmp_path / "lake_main.dat"
        lake_main.write_text(
            "#4.0\n"
            "max_elev.dat                       / MAXELEVFL\n"
            "                                   / BUDGETFL\n"
            "                                   / FINALFL\n"
            "1.0                                / FACTK\n"
            "1DAY                               / TUNITK\n"
            "1.0                                / FACTL\n"
            "1     0.5     1.0    1    7    2   Lake One\n"
            "2     0.8     2.0    2    8    3   Lake Two\n"
        )

        reader = LakeMainFileReader()
        config = reader.read(lake_main)

        assert len(config.lake_params) == 2
        assert config.lake_params[0].lake_id == 1
        assert config.lake_params[0].name == "Lake One"
        assert config.lake_params[1].lake_id == 2
        assert config.lake_params[1].name == "Lake Two"

    def test_v40_no_optional_files(self, tmp_path: Path) -> None:
        """Read v4.0 where optional file paths are blank."""
        lake_main = tmp_path / "lake_main.dat"
        lake_main.write_text(
            "#4.0\n"
            "                                   / MAXELEVFL\n"
            "                                   / BUDGETFL\n"
            "                                   / FINALFL\n"
            "1.0                                / FACTK\n"
            "1DAY                               / TUNITK\n"
            "1.0                                / FACTL\n"
        )

        reader = LakeMainFileReader()
        config = reader.read(lake_main)

        assert config.max_elev_file is None
        assert config.budget_output_file is None
        assert config.final_elev_file is None
        assert len(config.lake_params) == 0


class TestLakeMainFileReaderV50:
    """Tests for LakeMainFileReader with v5.0 format."""

    def test_basic_v50_with_outflow_ratings(self, tmp_path: Path) -> None:
        """Read v5.0 lake main file with outflow rating tables."""
        lake_main = tmp_path / "lake_main.dat"
        lake_main.write_text(
            "#5.0\n"
            "max_elev.dat                       / MAXELEVFL\n"
            "                                   / BUDGETFL\n"
            "                                   / FINALFL\n"
            "1.0                                / FACTK\n"
            "1DAY                               / TUNITK\n"
            "1.0                                / FACTL\n"
            "1     0.5     1.0    1    7    2   TestLake\n"
            "1.0                                / FACTELEV (breaks lake param loop)\n"
            "1.0                                / FACTQ\n"
            "1DAY                               / TUNITQ\n"
            "1     3     100.0  500.0\n"
            "200.0  1000.0\n"
            "300.0  2000.0\n"
        )

        reader = LakeMainFileReader()
        config = reader.read(lake_main)

        assert config.version == "5.0"
        assert len(config.lake_params) == 1
        assert config.elev_factor == 1.0
        assert config.outflow_factor == 1.0
        assert config.outflow_time_unit == "1DAY"
        assert len(config.outflow_ratings) == 1

        rating = config.outflow_ratings[0]
        assert rating.lake_id == 1
        assert len(rating.points) == 3
        assert rating.points[0].elevation == 100.0
        assert rating.points[0].outflow == 500.0
        assert rating.points[2].elevation == 300.0
        assert rating.points[2].outflow == 2000.0

    def test_v50_outflow_rating_with_factors(self, tmp_path: Path) -> None:
        """v5.0 outflow rating applies elevation and outflow factors."""
        lake_main = tmp_path / "lake_main.dat"
        lake_main.write_text(
            "#5.0\n"
            "                                   / MAXELEVFL\n"
            "                                   / BUDGETFL\n"
            "                                   / FINALFL\n"
            "1.0                                / FACTK\n"
            "1DAY                               / TUNITK\n"
            "1.0                                / FACTL\n"
            "1     0.5     1.0    1    7    2   Lake\n"
            "2.0                                / FACTELEV\n"
            "3.0                                / FACTQ\n"
            "1DAY                               / TUNITQ\n"
            "1     2     10.0  100.0\n"
            "20.0  200.0\n"
        )

        reader = LakeMainFileReader()
        config = reader.read(lake_main)

        assert config.elev_factor == 2.0
        assert config.outflow_factor == 3.0
        assert len(config.outflow_ratings) == 1

        rating = config.outflow_ratings[0]
        # 10.0 * 2.0 = 20.0, 100.0 * 3.0 = 300.0
        assert rating.points[0].elevation == pytest.approx(20.0)
        assert rating.points[0].outflow == pytest.approx(300.0)
        # 20.0 * 2.0 = 40.0, 200.0 * 3.0 = 600.0
        assert rating.points[1].elevation == pytest.approx(40.0)
        assert rating.points[1].outflow == pytest.approx(600.0)


class TestLakeMainFileReaderEdgeCases:
    """Edge cases for LakeMainFileReader targeting uncovered branches."""

    def test_elev_factor_value_error(self, tmp_path: Path) -> None:
        """v5.0: non-numeric single token after lake params triggers ValueError branch."""
        lake_main = tmp_path / "lake_main.dat"
        # After the lake param lines, a single non-numeric token triggers
        # the ValueError in the elev_factor parsing (lines 776-777)
        lake_main.write_text(
            "#5.0\n"
            "                                   / MAXELEVFL\n"
            "                                   / BUDGETFL\n"
            "                                   / FINALFL\n"
            "1.0                                / FACTK\n"
            "1DAY                               / TUNITK\n"
            "1.0                                / FACTL\n"
            "1     0.5     1.0    1    7    2   Lake\n"
            "notanumber                         / breaks elev_factor parse\n"
        )

        reader = LakeMainFileReader()
        config = reader.read(lake_main)

        # The ValueError breaks out of the loop; elev_factor stays default
        assert config.elev_factor == 1.0
        assert len(config.lake_params) == 1

    def test_lake_param_value_error_breaks(self, tmp_path: Path) -> None:
        """Lake param line with bad numeric data triggers ValueError break (lines 791-792)."""
        lake_main = tmp_path / "lake_main.dat"
        lake_main.write_text(
            "#4.0\n"
            "                                   / MAXELEVFL\n"
            "                                   / BUDGETFL\n"
            "                                   / FINALFL\n"
            "1.0                                / FACTK\n"
            "1DAY                               / TUNITK\n"
            "1.0                                / FACTL\n"
            "1     0.5     1.0    1    7    2   Lake Good\n"
            "abc   def     ghi    jkl  mno  pqr  Bad Line\n"
        )

        reader = LakeMainFileReader()
        config = reader.read(lake_main)

        # First lake param parsed fine; second triggers ValueError -> break
        assert len(config.lake_params) == 1
        assert config.lake_params[0].lake_id == 1

    def test_outflow_rating_empty_header_returns_none(self, tmp_path: Path) -> None:
        """_read_outflow_rating returns None when header is empty (line 827)."""
        lake_main = tmp_path / "lake_main.dat"
        # v5.0 with 1 lake but the outflow rating header is empty (EOF)
        lake_main.write_text(
            "#5.0\n"
            "                                   / MAXELEVFL\n"
            "                                   / BUDGETFL\n"
            "                                   / FINALFL\n"
            "1.0                                / FACTK\n"
            "1DAY                               / TUNITK\n"
            "1.0                                / FACTL\n"
            "1     0.5     1.0    1    7    2   Lake\n"
            "1.0                                / FACTELEV\n"
            "1.0                                / FACTQ\n"
            "1DAY                               / TUNITQ\n"
        )
        # No outflow rating data follows -> _next_data_or_empty returns ""

        reader = LakeMainFileReader()
        config = reader.read(lake_main)

        assert len(config.outflow_ratings) == 0

    def test_outflow_rating_short_header_returns_none(self, tmp_path: Path) -> None:
        """_read_outflow_rating returns None when header has < 4 parts (line 831)."""
        lake_main = tmp_path / "lake_main.dat"
        lake_main.write_text(
            "#5.0\n"
            "                                   / MAXELEVFL\n"
            "                                   / BUDGETFL\n"
            "                                   / FINALFL\n"
            "1.0                                / FACTK\n"
            "1DAY                               / TUNITK\n"
            "1.0                                / FACTL\n"
            "1     0.5     1.0    1    7    2   Lake\n"
            "1.0                                / FACTELEV\n"
            "1.0                                / FACTQ\n"
            "1DAY                               / TUNITQ\n"
            "1     3\n"
        )
        # Header has only 2 parts (lake_id, n_points) but needs at least 4

        reader = LakeMainFileReader()
        config = reader.read(lake_main)

        assert len(config.outflow_ratings) == 0

    def test_outflow_rating_empty_line_in_points(self, tmp_path: Path) -> None:
        """_read_outflow_rating breaks when a point line is empty (line 848)."""
        lake_main = tmp_path / "lake_main.dat"
        # Rating with 3 points declared but only 1 continuation line before EOF
        lake_main.write_text(
            "#5.0\n"
            "                                   / MAXELEVFL\n"
            "                                   / BUDGETFL\n"
            "                                   / FINALFL\n"
            "1.0                                / FACTK\n"
            "1DAY                               / TUNITK\n"
            "1.0                                / FACTL\n"
            "1     0.5     1.0    1    7    2   Lake\n"
            "1.0                                / FACTELEV\n"
            "1.0                                / FACTQ\n"
            "1DAY                               / TUNITQ\n"
            "1     3     100.0  500.0\n"
            "200.0  1000.0\n"
        )
        # 3 points expected but only 2 provided (header line + 1 continuation)
        # Third read hits EOF -> empty string -> break

        reader = LakeMainFileReader()
        config = reader.read(lake_main)

        assert len(config.outflow_ratings) == 1
        # Should have 2 points (header line + 1 continuation), not 3
        assert len(config.outflow_ratings[0].points) == 2

    def test_read_version_blank_line_skip(self, tmp_path: Path) -> None:
        """_read_version skips blank lines before version header (line 866)."""
        lake_main = tmp_path / "lake_main.dat"
        lake_main.write_text(
            "\n"
            "\n"
            "C  some comment\n"
            "#4.0\n"
            "                                   / MAXELEVFL\n"
            "                                   / BUDGETFL\n"
            "                                   / FINALFL\n"
            "1.0                                / FACTK\n"
            "1DAY                               / TUNITK\n"
            "1.0                                / FACTL\n"
        )

        reader = LakeMainFileReader()
        config = reader.read(lake_main)

        assert config.version == "4.0"


class TestReadLakeMainFileConvenience:
    """Tests for the read_lake_main_file convenience function."""

    def test_convenience_function(self, tmp_path: Path) -> None:
        """read_lake_main_file delegates to LakeMainFileReader.read."""
        lake_main = tmp_path / "lake_main.dat"
        lake_main.write_text(
            "#4.0\n"
            "                                   / MAXELEVFL\n"
            "                                   / BUDGETFL\n"
            "                                   / FINALFL\n"
            "1.0                                / FACTK\n"
            "1DAY                               / TUNITK\n"
            "1.0                                / FACTL\n"
        )

        config = read_lake_main_file(lake_main)

        assert config.version == "4.0"

    def test_convenience_with_base_dir(self, tmp_path: Path) -> None:
        """read_lake_main_file passes base_dir to reader."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        lake_main = subdir / "lake_main.dat"
        lake_main.write_text(
            "#4.0\n"
            "max_elev.dat                       / MAXELEVFL\n"
            "                                   / BUDGETFL\n"
            "                                   / FINALFL\n"
            "1.0                                / FACTK\n"
            "1DAY                               / TUNITK\n"
            "1.0                                / FACTL\n"
        )

        config = read_lake_main_file(lake_main, base_dir=tmp_path)

        assert config.max_elev_file is not None
        # base_dir is tmp_path, so resolved path should be relative to tmp_path
        assert config.max_elev_file.parent == tmp_path
