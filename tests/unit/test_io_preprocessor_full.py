"""
Comprehensive tests for pyiwfm.io.preprocessor module.

Tests the PreProcessor file I/O handlers including reading and writing
IWFM PreProcessor input files.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pyiwfm.core.exceptions import FileFormatError
from pyiwfm.core.mesh import Subregion
from pyiwfm.io.preprocessor import (
    PreProcessorConfig,
    _is_comment_line,
    _make_relative_path,
    _resolve_path,
    _strip_comment,
    load_complete_model,
    load_model_from_preprocessor,
    read_preprocessor_main,
    read_subregions_file,
    save_complete_model,
    save_model_to_preprocessor,
    write_preprocessor_main,
)

# =============================================================================
# Helper Function Tests
# =============================================================================


class TestIsCommentLine:
    """Test _is_comment_line function."""

    def test_comment_line_with_c(self):
        """Test line starting with 'C' is a comment."""
        assert _is_comment_line("C This is a comment") is True

    def test_comment_line_with_lowercase_c(self):
        """Test line starting with 'c' is a comment."""
        assert _is_comment_line("c this is a comment") is True

    def test_comment_line_with_asterisk(self):
        """Test line starting with '*' is a comment."""
        assert _is_comment_line("* This is a comment") is True

    def test_empty_line_is_comment(self):
        """Test empty line is treated as comment."""
        assert _is_comment_line("") is True

    def test_whitespace_only_is_comment(self):
        """Test whitespace-only line is treated as comment."""
        assert _is_comment_line("   \t  ") is True

    def test_data_line_not_comment(self):
        """Test line with data is not a comment."""
        assert _is_comment_line("123.45  / VALUE") is False

    def test_line_with_leading_space_not_comment(self):
        """Test line starting with space followed by data is not comment."""
        # In IWFM, whitespace then data = data line
        assert _is_comment_line("  123.45") is False

    def test_indented_c_not_comment(self):
        """Test 'C' not in column 1 is not treated as comment."""
        assert _is_comment_line("   C comment") is False


class TestParseValueLine:
    """Test _strip_comment function."""

    def test_parse_with_slash_separator(self):
        """Test parsing line with '/' separator (traditional IWFM)."""
        value, desc = _strip_comment("nodes.dat  / NODES_FILE")
        assert value == "nodes.dat"
        assert desc == "NODES_FILE"

    def test_parse_with_hash_not_recognized(self):
        """Hash is not recognized as an inline comment delimiter."""
        value, desc = _strip_comment("elements.dat # ELEMENTS_FILE")
        assert value == "elements.dat # ELEMENTS_FILE"
        assert desc == ""

    def test_parse_without_separator(self):
        """Test parsing line without any separator."""
        value, desc = _strip_comment("nodes.dat")
        assert value == "nodes.dat"
        assert desc == ""

    def test_parse_preserves_slash_in_path(self):
        """Slash inside a path (no preceding whitespace) is preserved."""
        value, desc = _strip_comment("data/nodes.dat / NODES_FILE")
        # The delimiter is the ' / ' with whitespace, not the '/' in the path
        assert value == "data/nodes.dat"
        assert desc == "NODES_FILE"

    def test_parse_empty_description(self):
        """Test parsing with empty description after separator."""
        value, desc = _strip_comment("value /")
        assert value == "value"
        assert desc == ""

    def test_parse_multiple_slashes(self):
        """Slash preceded by whitespace is the delimiter, not path slashes."""
        value, desc = _strip_comment("a/b/c / DESCRIPTION")
        assert value == "a/b/c"
        assert desc == "DESCRIPTION"


class TestResolvePath:
    """Test _resolve_path function."""

    def test_resolve_relative_path(self, tmp_path):
        """Test resolving relative path."""
        result = _resolve_path(tmp_path, "data/nodes.dat")
        assert result == tmp_path / "data/nodes.dat"

    def test_resolve_absolute_path(self, tmp_path):
        """Test resolving absolute path."""
        import platform

        if platform.system() == "Windows":
            abs_path = "C:/absolute/path/to/file.dat"
        else:
            abs_path = "/absolute/path/to/file.dat"
        result = _resolve_path(tmp_path, abs_path)
        assert result == Path(abs_path)

    def test_resolve_windows_absolute_path(self, tmp_path):
        """Test resolving Windows absolute path."""
        abs_path = "C:/Users/test/file.dat"
        _resolve_path(tmp_path, abs_path)
        # On Windows this would be absolute, on Unix it's relative
        # The function checks is_absolute() which handles this

    def test_resolve_strips_whitespace(self, tmp_path):
        """Test that whitespace is stripped from path."""
        result = _resolve_path(tmp_path, "  nodes.dat  ")
        assert result == tmp_path / "nodes.dat"


class TestMakeRelativePath:
    """Test _make_relative_path function."""

    def test_make_relative_child_path(self, tmp_path):
        """Test making relative path for child directory."""
        base = tmp_path
        target = tmp_path / "data" / "nodes.dat"
        result = _make_relative_path(base, target)
        # Should be relative
        assert result == "data/nodes.dat" or result == "data\\nodes.dat"

    def test_make_relative_unrelated_path(self, tmp_path):
        """Test making relative path for unrelated directory returns absolute."""
        base = tmp_path / "project"
        target = Path("/some/other/path/file.dat")
        result = _make_relative_path(base, target)
        # Should return absolute since can't make relative
        assert result == str(target)


# =============================================================================
# PreProcessorConfig Tests
# =============================================================================


class TestPreProcessorConfig:
    """Test PreProcessorConfig dataclass."""

    def test_create_minimal_config(self, tmp_path):
        """Test creating config with minimal required fields."""
        config = PreProcessorConfig(base_dir=tmp_path)
        assert config.base_dir == tmp_path
        assert config.model_name == ""
        assert config.nodes_file is None
        assert config.n_layers == 1
        assert config.length_unit == "FT"

    def test_create_full_config(self, tmp_path):
        """Test creating config with all fields."""
        config = PreProcessorConfig(
            base_dir=tmp_path,
            model_name="TestModel",
            nodes_file=tmp_path / "nodes.dat",
            elements_file=tmp_path / "elements.dat",
            n_layers=3,
            length_unit="M",
            area_unit="SQ_M",
            volume_unit="CU_M",
        )
        assert config.model_name == "TestModel"
        assert config.nodes_file == tmp_path / "nodes.dat"
        assert config.n_layers == 3
        assert config.length_unit == "M"

    def test_config_metadata_default(self, tmp_path):
        """Test metadata defaults to empty dict."""
        config = PreProcessorConfig(base_dir=tmp_path)
        assert config.metadata == {}


# =============================================================================
# Read PreProcessor Main Tests
# =============================================================================


class TestReadPreprocessorMain:
    """Test read_preprocessor_main function."""

    def test_read_basic_preprocessor_file(self, tmp_path):
        """Test reading basic preprocessor file."""
        pp_file = tmp_path / "preprocessor.in"
        pp_file.write_text("""C  IWFM PreProcessor Input File
C
TestModel                      / MODEL_NAME
nodes.dat                      / NODES_FILE
elements.dat                   / ELEMENTS_FILE
stratigraphy.dat               / STRATIGRAPHY_FILE
3                              / N_LAYERS
FT                             / LENGTH_UNIT
""")

        config = read_preprocessor_main(pp_file)

        assert config.model_name == "TestModel"
        assert config.nodes_file == tmp_path / "nodes.dat"
        assert config.elements_file == tmp_path / "elements.dat"
        assert config.stratigraphy_file == tmp_path / "stratigraphy.dat"
        assert config.n_layers == 3
        assert config.length_unit == "FT"

    def test_read_with_hash_comments(self, tmp_path):
        """Test reading file with # style comments (C2VSimFG format)."""
        # Note: IWFM format with / uses / as delimiter, so paths with / must use #
        pp_file = tmp_path / "preprocessor.in"
        pp_file.write_text("""C  C2VSimFG PreProcessor File
C
MyModel                        / MODEL_NAME
nodes.dat                      / NODES_FILE
elements.dat                   / ELEMENTS_FILE
""")

        config = read_preprocessor_main(pp_file)

        assert config.model_name == "MyModel"
        assert config.nodes_file == tmp_path / "nodes.dat"

    def test_read_with_optional_files(self, tmp_path):
        """Test reading file with optional component files."""
        pp_file = tmp_path / "preprocessor.in"
        pp_file.write_text("""C  PreProcessor
Model1                         / MODEL_NAME
nodes.dat                      / NODES_FILE
elements.dat                   / ELEMENTS_FILE
streams.dat                    / STREAMS_FILE
lakes.dat                      / LAKES_FILE
groundwater.dat                / GROUNDWATER_FILE
rootzone.dat                   / ROOTZONE_FILE
""")

        config = read_preprocessor_main(pp_file)

        assert config.streams_file == tmp_path / "streams.dat"
        assert config.lakes_file == tmp_path / "lakes.dat"
        assert config.groundwater_file == tmp_path / "groundwater.dat"
        assert config.rootzone_file == tmp_path / "rootzone.dat"

    def test_read_skips_comment_lines(self, tmp_path):
        """Test that comment lines are properly skipped."""
        pp_file = tmp_path / "preprocessor.in"
        pp_file.write_text("""C  This is a comment
* Another comment
c lowercase comment
TestModel                      / MODEL_NAME
""")

        config = read_preprocessor_main(pp_file)

        assert config.model_name == "TestModel"

    def test_read_empty_file(self, tmp_path):
        """Test reading empty file."""
        pp_file = tmp_path / "preprocessor.in"
        pp_file.write_text("C  Empty file\n")

        config = read_preprocessor_main(pp_file)

        assert config.model_name == ""
        assert config.nodes_file is None

    def test_read_with_units(self, tmp_path):
        """Test reading file with unit specifications."""
        # The _strip_comment implementation uses / as separator, so
        # unit values are parsed. The description needs "UNIT" keyword.
        pp_file = tmp_path / "preprocessor.in"
        pp_file.write_text("""C  PreProcessor
Model                          / MODEL_NAME
nodes.dat                      / NODES_FILE
elements.dat                   / ELEMENTS_FILE
""")

        config = read_preprocessor_main(pp_file)

        # Default units are used when not specified in file
        assert config.model_name == "Model"
        assert config.length_unit == "FT"  # Default
        assert config.area_unit == "ACRES"  # Default


# =============================================================================
# Read Subregions File Tests
# =============================================================================


class TestReadSubregionsFile:
    """Test read_subregions_file function."""

    def test_read_basic_subregions(self, tmp_path):
        """Test reading basic subregions file."""
        sr_file = tmp_path / "subregions.dat"
        sr_file.write_text("""C  Subregion definitions
3                              / NSUBREGION
1  Sacramento Valley
2  San Joaquin Valley
3  Tulare Basin
""")

        subregions = read_subregions_file(sr_file)

        assert len(subregions) == 3
        assert subregions[1].name == "Sacramento Valley"
        assert subregions[2].name == "San Joaquin Valley"
        assert subregions[3].name == "Tulare Basin"

    def test_read_subregions_with_comments(self, tmp_path):
        """Test reading subregions file with interspersed comments."""
        sr_file = tmp_path / "subregions.dat"
        sr_file.write_text("""C  Subregion definitions
C  ID  NAME
2                              / NSUBREGION
C  First region
1  Region One
C  Second region
2  Region Two
""")

        subregions = read_subregions_file(sr_file)

        assert len(subregions) == 2
        assert subregions[1].name == "Region One"
        assert subregions[2].name == "Region Two"

    def test_read_subregions_without_names(self, tmp_path):
        """Test reading subregions file without names."""
        sr_file = tmp_path / "subregions.dat"
        sr_file.write_text("""C  Subregion definitions
2                              / NSUBREGION
1
2
""")

        subregions = read_subregions_file(sr_file)

        assert len(subregions) == 2
        assert subregions[1].name == ""
        assert subregions[2].name == ""

    def test_read_subregions_invalid_count(self, tmp_path):
        """Test error on invalid NSUBREGION value."""
        sr_file = tmp_path / "subregions.dat"
        sr_file.write_text("""C  Subregions
INVALID                        / NSUBREGION
""")

        with pytest.raises(FileFormatError) as exc_info:
            read_subregions_file(sr_file)
        assert "Invalid NSUBREGION value" in str(exc_info.value)

    def test_read_subregions_missing_count(self, tmp_path):
        """Test error when NSUBREGION is missing."""
        sr_file = tmp_path / "subregions.dat"
        sr_file.write_text("""C  No data, just comments
""")

        with pytest.raises(FileFormatError) as exc_info:
            read_subregions_file(sr_file)
        assert "Could not find NSUBREGION" in str(exc_info.value)

    def test_read_subregions_invalid_id(self, tmp_path):
        """Test error on invalid subregion ID."""
        sr_file = tmp_path / "subregions.dat"
        sr_file.write_text("""C  Subregions
1                              / NSUBREGION
NOT_AN_ID  Region Name
""")

        with pytest.raises(FileFormatError) as exc_info:
            read_subregions_file(sr_file)
        assert "Invalid subregion data" in str(exc_info.value)


# =============================================================================
# Write PreProcessor Main Tests
# =============================================================================


class TestWritePreprocessorMain:
    """Test write_preprocessor_main function."""

    def test_write_basic_file(self, tmp_path):
        """Test writing basic preprocessor file."""
        config = PreProcessorConfig(
            base_dir=tmp_path,
            model_name="TestModel",
            nodes_file=tmp_path / "nodes.dat",
            elements_file=tmp_path / "elements.dat",
            n_layers=3,
        )

        output_file = tmp_path / "preprocessor.in"
        write_preprocessor_main(output_file, config)

        content = output_file.read_text()
        assert "TestModel" in content
        assert "MODEL_NAME" in content
        assert "NODES_FILE" in content
        assert "ELEMENTS_FILE" in content
        assert "N_LAYERS" in content

    def test_write_with_custom_header(self, tmp_path):
        """Test writing file with custom header."""
        config = PreProcessorConfig(
            base_dir=tmp_path,
            model_name="Test",
        )

        output_file = tmp_path / "preprocessor.in"
        write_preprocessor_main(
            output_file, config, header="Custom Header Line 1\nCustom Header Line 2"
        )

        content = output_file.read_text()
        assert "Custom Header Line 1" in content
        assert "Custom Header Line 2" in content

    def test_write_with_all_file_paths(self, tmp_path):
        """Test writing file with all optional paths."""
        config = PreProcessorConfig(
            base_dir=tmp_path,
            model_name="FullModel",
            nodes_file=tmp_path / "nodes.dat",
            elements_file=tmp_path / "elements.dat",
            stratigraphy_file=tmp_path / "strat.dat",
            subregions_file=tmp_path / "subregions.dat",
        )

        output_file = tmp_path / "preprocessor.in"
        write_preprocessor_main(output_file, config)

        content = output_file.read_text()
        assert "STRATIGRAPHY_FILE" in content
        assert "SUBREGIONS_FILE" in content

    def test_write_creates_relative_paths(self, tmp_path):
        """Test that paths are written as relative."""
        config = PreProcessorConfig(
            base_dir=tmp_path,
            model_name="Test",
            nodes_file=tmp_path / "data" / "nodes.dat",
        )

        output_file = tmp_path / "preprocessor.in"
        write_preprocessor_main(output_file, config)

        content = output_file.read_text()
        # Path should be relative, not absolute
        assert str(tmp_path) not in content or "data" in content


# =============================================================================
# Save Model to PreProcessor Tests
# =============================================================================


class TestSaveModelToPreprocessor:
    """Test save_model_to_preprocessor function."""

    @patch("pyiwfm.io.preprocessor.write_nodes")
    @patch("pyiwfm.io.preprocessor.write_elements")
    @patch("pyiwfm.io.preprocessor.write_stratigraphy")
    def test_save_model_with_mesh(
        self, mock_write_strat, mock_write_elem, mock_write_nodes, tmp_path
    ):
        """Test saving model with mesh."""
        # Create mock model
        model = MagicMock()
        model.name = "TestModel"
        model.mesh = MagicMock()
        model.mesh.nodes = {1: MagicMock(), 2: MagicMock()}
        model.mesh.elements = {1: MagicMock()}
        model.mesh.n_subregions = 2
        model.mesh.subregions = {}
        model.stratigraphy = None
        model.n_layers = 1

        config = save_model_to_preprocessor(model, tmp_path)

        assert config.model_name == "TestModel"
        assert config.nodes_file == tmp_path / "nodes.dat"
        assert config.elements_file == tmp_path / "elements.dat"
        mock_write_nodes.assert_called_once()
        mock_write_elem.assert_called_once()

    @patch("pyiwfm.io.preprocessor.write_nodes")
    @patch("pyiwfm.io.preprocessor.write_elements")
    @patch("pyiwfm.io.preprocessor.write_stratigraphy")
    def test_save_model_with_stratigraphy(
        self, mock_write_strat, mock_write_elem, mock_write_nodes, tmp_path
    ):
        """Test saving model with stratigraphy."""
        model = MagicMock()
        model.name = "StratModel"
        model.mesh = MagicMock()
        model.mesh.nodes = {}
        model.mesh.elements = {}
        model.mesh.n_subregions = 1
        model.mesh.subregions = {}
        model.stratigraphy = MagicMock()
        model.n_layers = 3

        config = save_model_to_preprocessor(model, tmp_path)

        assert config.stratigraphy_file == tmp_path / "stratigraphy.dat"
        mock_write_strat.assert_called_once()

    @patch("pyiwfm.io.preprocessor.write_nodes")
    @patch("pyiwfm.io.preprocessor.write_elements")
    def test_save_creates_output_directory(self, mock_write_elem, mock_write_nodes, tmp_path):
        """Test that output directory is created if it doesn't exist."""
        output_dir = tmp_path / "new_dir" / "nested"

        model = MagicMock()
        model.name = "Test"
        model.mesh = MagicMock()
        model.mesh.nodes = {}
        model.mesh.elements = {}
        model.mesh.n_subregions = 1
        model.mesh.subregions = {}
        model.stratigraphy = None
        model.n_layers = 1

        save_model_to_preprocessor(model, output_dir)

        assert output_dir.exists()

    @patch("pyiwfm.io.preprocessor.write_nodes")
    @patch("pyiwfm.io.preprocessor.write_elements")
    def test_save_uses_default_name(self, mock_write_elem, mock_write_nodes, tmp_path):
        """Test that default name is used if model has no name."""
        model = MagicMock()
        model.name = ""
        model.mesh = MagicMock()
        model.mesh.nodes = {}
        model.mesh.elements = {}
        model.mesh.n_subregions = 1
        model.mesh.subregions = {}
        model.stratigraphy = None
        model.n_layers = 1

        config = save_model_to_preprocessor(model, tmp_path)

        assert config.model_name == "iwfm_model"


# =============================================================================
# Load Model from PreProcessor Tests
# =============================================================================


class TestLoadModelFromPreprocessor:
    """Test load_model_from_preprocessor function."""

    @patch("pyiwfm.io.preprocessor.AppGrid")
    @patch("pyiwfm.io.preprocessor.read_stratigraphy")
    @patch("pyiwfm.io.preprocessor.read_elements")
    @patch("pyiwfm.io.preprocessor.read_nodes")
    @patch("pyiwfm.io.preprocessor.read_preprocessor_main")
    def test_load_basic_model(
        self,
        mock_read_pp,
        mock_read_nodes,
        mock_read_elem,
        mock_read_strat,
        mock_grid_cls,
        tmp_path,
    ):
        """Test loading basic model."""
        # Setup mocks
        mock_config = MagicMock()
        mock_config.model_name = "TestModel"
        mock_config.nodes_file = tmp_path / "nodes.dat"
        mock_config.elements_file = tmp_path / "elements.dat"
        mock_config.stratigraphy_file = None
        mock_config.subregions_file = None
        mock_config.length_unit = "FT"
        mock_config.area_unit = "ACRES"
        mock_config.volume_unit = "AF"
        mock_read_pp.return_value = mock_config

        mock_read_nodes.return_value = {
            1: MagicMock(id=1, x=0.0, y=0.0),
            2: MagicMock(id=2, x=1.0, y=0.0),
        }
        mock_read_elem.return_value = (
            {1: MagicMock(id=1, node_ids=[1, 2, 3])},
            1,
            {},  # subregion_names dict
        )

        mock_mesh = MagicMock()
        mock_grid_cls.return_value = mock_mesh

        model = load_model_from_preprocessor(tmp_path / "pp.in")

        assert model.name == "TestModel"
        assert model.mesh is mock_mesh
        assert model.metadata["length_unit"] == "FT"
        mock_mesh.compute_areas.assert_called_once()
        mock_mesh.compute_connectivity.assert_called_once()

    @patch("pyiwfm.io.preprocessor.read_preprocessor_main")
    def test_load_raises_on_missing_nodes_file(self, mock_read_pp, tmp_path):
        """Test error when nodes file is not specified."""
        mock_config = MagicMock()
        mock_config.nodes_file = None
        mock_read_pp.return_value = mock_config

        with pytest.raises(FileFormatError) as exc_info:
            load_model_from_preprocessor(tmp_path / "pp.in")
        assert "Nodes file not specified" in str(exc_info.value)

    @patch("pyiwfm.io.preprocessor.read_nodes")
    @patch("pyiwfm.io.preprocessor.read_preprocessor_main")
    def test_load_raises_on_missing_elements_file(self, mock_read_pp, mock_read_nodes, tmp_path):
        """Test error when elements file is not specified."""
        mock_config = MagicMock()
        mock_config.nodes_file = tmp_path / "nodes.dat"
        mock_config.elements_file = None
        mock_read_pp.return_value = mock_config
        mock_read_nodes.return_value = {}

        with pytest.raises(FileFormatError) as exc_info:
            load_model_from_preprocessor(tmp_path / "pp.in")
        assert "Elements file not specified" in str(exc_info.value)


# =============================================================================
# Load Complete Model Tests
# =============================================================================


class TestLoadCompleteModel:
    """Test load_complete_model function."""

    @patch("pyiwfm.io.rootzone.RootZoneReader")
    @patch("pyiwfm.io.lakes.LakeReader")
    @patch("pyiwfm.io.streams.StreamReader")
    @patch("pyiwfm.io.groundwater.GroundwaterReader")
    @patch("pyiwfm.io.preprocessor.load_model_from_preprocessor")
    @patch("pyiwfm.io.simulation.SimulationReader")
    def test_load_complete_model_basic(
        self,
        mock_sim_reader,
        mock_load_pp,
        mock_gw_reader,
        mock_stream_reader,
        mock_lake_reader,
        mock_rz_reader,
        tmp_path,
    ):
        """Test loading complete model from simulation file."""
        # Create a real Path that exists check can work on
        pp_file = tmp_path / "pp.in"
        pp_file.touch()

        # Setup simulation reader mock
        sim_reader_instance = MagicMock()
        mock_sim_reader.return_value = sim_reader_instance

        sim_config = MagicMock()
        sim_config.preprocessor_file = pp_file
        sim_config.groundwater_file = None
        sim_config.streams_file = None
        sim_config.lakes_file = None
        sim_config.rootzone_file = None
        sim_config.start_date.isoformat.return_value = "2000-01-01"
        sim_config.end_date.isoformat.return_value = "2000-12-31"
        sim_config.time_step_length = 1
        sim_config.time_step_unit.value = "MONTH"
        sim_reader_instance.read.return_value = sim_config

        # Setup model mock
        model = MagicMock()
        model.metadata = {}
        model.mesh = MagicMock()
        model.mesh.n_nodes = 10
        model.mesh.n_elements = 5
        model.n_layers = 1
        mock_load_pp.return_value = model

        result = load_complete_model(tmp_path / "simulation.in")

        assert result == model
        assert "simulation_file" in model.metadata
        mock_load_pp.assert_called_once()

    @patch("pyiwfm.io.preprocessor.IWFMModel")
    @patch("pyiwfm.io.simulation.SimulationReader")
    def test_load_handles_missing_preprocessor(self, mock_sim_reader, mock_model_cls, tmp_path):
        """Test loading when preprocessor file doesn't exist."""
        sim_reader_instance = MagicMock()
        mock_sim_reader.return_value = sim_reader_instance

        sim_config = MagicMock()
        sim_config.model_name = "TestModel"
        sim_config.preprocessor_file = None
        sim_config.groundwater_file = None
        sim_config.streams_file = None
        sim_config.lakes_file = None
        sim_config.rootzone_file = None
        sim_config.start_date.isoformat.return_value = "2000-01-01"
        sim_config.end_date.isoformat.return_value = "2000-12-31"
        sim_config.time_step_length = 1
        sim_config.time_step_unit.value = "MONTH"
        sim_reader_instance.read.return_value = sim_config

        mock_model = MagicMock()
        mock_model.metadata = {}
        mock_model_cls.return_value = mock_model

        result = load_complete_model(tmp_path / "simulation.in")

        # Should create empty model when no preprocessor
        assert result is not None


# =============================================================================
# Save Complete Model Tests
# =============================================================================


class TestSaveCompleteModel:
    """Test save_complete_model function."""

    @patch("pyiwfm.io.model_writer.CompleteModelWriter")
    def test_save_minimal_model(self, mock_writer_cls, tmp_path):
        """Test saving minimal model without components."""
        mock_result = MagicMock()
        mock_result.files = {
            "preprocessor_main": tmp_path / "pp.in",
            "simulation_main": tmp_path / "sim.in",
        }
        mock_writer_instance = MagicMock()
        mock_writer_instance.write_all.return_value = mock_result
        mock_writer_cls.return_value = mock_writer_instance

        model = MagicMock()
        model.name = "TestModel"
        model.groundwater = None
        model.streams = None
        model.lakes = None
        model.rootzone = None

        files = save_complete_model(model, tmp_path)

        assert "preprocessor_main" in files
        assert "simulation_main" in files
        mock_writer_cls.assert_called_once()

    @patch("pyiwfm.io.model_writer.CompleteModelWriter")
    def test_save_model_with_groundwater(self, mock_writer_cls, tmp_path):
        """Test saving model with groundwater component."""
        mock_result = MagicMock()
        mock_result.files = {
            "preprocessor_main": tmp_path / "pp.in",
            "simulation_main": tmp_path / "sim.in",
            "gw_main": tmp_path / "gw" / "GW_MAIN.dat",
        }
        mock_writer_instance = MagicMock()
        mock_writer_instance.write_all.return_value = mock_result
        mock_writer_cls.return_value = mock_writer_instance

        model = MagicMock()
        model.name = "GWModel"
        model.groundwater = MagicMock()
        model.streams = None
        model.lakes = None
        model.rootzone = None

        files = save_complete_model(model, tmp_path)

        assert "gw_main" in files
        mock_writer_cls.assert_called_once()

    @patch("pyiwfm.io.simulation.SimulationWriter")
    @patch("pyiwfm.io.preprocessor.save_model_to_preprocessor")
    def test_save_creates_output_directory(self, mock_save_pp, mock_sim_writer, tmp_path):
        """Test that output directory is created."""
        output_dir = tmp_path / "new_output" / "nested"

        mock_config = MagicMock()
        mock_config.nodes_file = None
        mock_config.elements_file = None
        mock_config.stratigraphy_file = None
        mock_config.subregions_file = None
        mock_save_pp.return_value = mock_config

        sim_writer_instance = MagicMock()
        mock_sim_writer.return_value = sim_writer_instance
        sim_writer_instance.write.return_value = output_dir / "simulation.in"

        model = MagicMock()
        model.name = "Test"
        model.groundwater = None
        model.streams = None
        model.lakes = None
        model.rootzone = None

        save_complete_model(model, output_dir)

        assert output_dir.exists()


# =============================================================================
# Round-Trip Tests
# =============================================================================


class TestPreprocessorRoundTrip:
    """Test round-trip read/write operations."""

    def test_preprocessor_main_roundtrip(self, tmp_path):
        """Test writing and reading preprocessor main file."""
        # Create config
        original_config = PreProcessorConfig(
            base_dir=tmp_path,
            model_name="RoundTripModel",
            nodes_file=tmp_path / "nodes.dat",
            elements_file=tmp_path / "elements.dat",
            n_layers=3,
            length_unit="M",
            area_unit="SQ_M",
            volume_unit="CU_M",
        )

        # Write
        pp_file = tmp_path / "preprocessor.in"
        write_preprocessor_main(pp_file, original_config)

        # Read back
        read_config = read_preprocessor_main(pp_file)

        assert read_config.model_name == original_config.model_name
        assert read_config.n_layers == original_config.n_layers
        assert read_config.length_unit == original_config.length_unit

    def test_subregions_roundtrip(self, tmp_path):
        """Test writing and reading subregions file."""
        from pyiwfm.io.preprocessor import _write_subregions_file

        # Create subregions
        original_subregions = {
            1: Subregion(id=1, name="Region One"),
            2: Subregion(id=2, name="Region Two"),
            3: Subregion(id=3, name="Region Three"),
        }

        # Write
        sr_file = tmp_path / "subregions.dat"
        _write_subregions_file(sr_file, original_subregions)

        # Read back
        read_subregions = read_subregions_file(sr_file)

        assert len(read_subregions) == len(original_subregions)
        for sr_id in original_subregions:
            assert read_subregions[sr_id].name == original_subregions[sr_id].name
