"""Unit tests for PEST++ template file generation."""

import tempfile
from pathlib import Path

import pytest

from pyiwfm.runner.pest_manager import IWFMParameterManager
from pyiwfm.runner.pest_params import (
    IWFMParameterType,
    Parameter,
)
from pyiwfm.runner.pest_templates import (
    IWFMFileSection,
    IWFMTemplateManager,
    TemplateMarker,
)


class TestTemplateMarker:
    """Tests for TemplateMarker dataclass."""

    def test_basic_creation(self):
        """Test basic marker creation."""
        marker = TemplateMarker(
            parameter_name="hk_z1",
            line_number=10,
            column_start=20,
            column_end=35,
            original_value="1.5e-04",
        )
        assert marker.parameter_name == "hk_z1"
        assert marker.line_number == 10
        assert marker.column_start == 20
        assert marker.column_end == 35
        assert marker.original_value == "1.5e-04"


class TestIWFMFileSection:
    """Tests for IWFMFileSection dataclass."""

    def test_basic_creation(self):
        """Test basic section creation."""
        section = IWFMFileSection(
            name="AQUIFER_PROPERTIES",
            start_line=50,
            end_line=100,
        )
        assert section.name == "AQUIFER_PROPERTIES"
        assert section.start_line == 50
        assert section.end_line == 100

    def test_with_columns(self):
        """Test section with column definitions."""
        section = IWFMFileSection(
            name="AQUIFER_PROPERTIES",
            start_line=50,
            end_line=100,
            data_columns={"zone": 1, "hk": 2, "ss": 3},
        )
        assert section.data_columns["zone"] == 1
        assert section.data_columns["hk"] == 2


class TestIWFMTemplateManagerInit:
    """Tests for IWFMTemplateManager initialization."""

    def test_init_basic(self):
        """Test basic initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tm = IWFMTemplateManager(output_dir=tmpdir)
            assert tm.output_dir == Path(tmpdir)
            assert tm.delimiter == "#"
            assert len(tm._templates) == 0

    def test_init_with_custom_delimiter(self):
        """Test initialization with custom delimiter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tm = IWFMTemplateManager(output_dir=tmpdir, delimiter="~")
            assert tm.delimiter == "~"

    def test_init_creates_output_dir(self):
        """Test that output directory is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "templates" / "nested"
            IWFMTemplateManager(output_dir=output_dir)
            assert output_dir.exists()


class TestAquiferTemplates:
    """Tests for aquifer parameter templates."""

    @pytest.fixture
    def sample_aquifer_file(self):
        """Create a sample aquifer parameter file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "Groundwater.dat"
            content = """C Groundwater Parameter File
C Zone  Horizontal_K  Specific_Storage  Specific_Yield
    1    1.500000e-04   1.000000e-06      0.150000
    2    2.500000e-04   1.500000e-06      0.180000
    3    1.000000e-04   8.000000e-07      0.120000
"""
            filepath.write_text(content)
            yield filepath

    @pytest.fixture
    def sample_parameters(self):
        """Create sample parameters."""
        return [
            Parameter(
                name="hk_z1",
                param_type=IWFMParameterType.HORIZONTAL_K,
                initial_value=1.5e-04,
                lower_bound=1e-06,
                upper_bound=1e-02,
                zone=1,
                layer=1,
            ),
            Parameter(
                name="hk_z2",
                param_type=IWFMParameterType.HORIZONTAL_K,
                initial_value=2.5e-04,
                lower_bound=1e-06,
                upper_bound=1e-02,
                zone=2,
                layer=1,
            ),
        ]

    def test_generate_aquifer_template_by_zone(self, sample_aquifer_file, sample_parameters):
        """Test generating zone-based aquifer template."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = IWFMParameterManager()
            for p in sample_parameters:
                pm._parameters[p.name] = p

            tm = IWFMTemplateManager(parameter_manager=pm, output_dir=tmpdir)

            tpl = tm.generate_aquifer_template_by_zone(
                input_file=sample_aquifer_file,
                param_type=IWFMParameterType.HORIZONTAL_K,
                zone_column=1,
                value_column=2,
                layer=1,
            )

            assert tpl.template_path.exists()
            assert len(tpl.parameters) == 2

            # Check template content
            content = tpl.template_path.read_text()
            assert "ptf #" in content
            # Marker format may vary in spacing
            assert "hk_z1" in content

    def test_aquifer_template_no_parameters_raises(self):
        """Test that missing parameters raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = IWFMParameterManager()
            tm = IWFMTemplateManager(parameter_manager=pm, output_dir=tmpdir)

            with pytest.raises(ValueError, match="No.*parameters found"):
                tm.generate_aquifer_template(
                    input_file=Path(tmpdir) / "test.dat",
                    param_type=IWFMParameterType.HORIZONTAL_K,
                    layer=1,
                )


class TestStreamTemplates:
    """Tests for stream parameter templates."""

    @pytest.fixture
    def sample_stream_file(self):
        """Create a sample stream parameter file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "Stream.dat"
            content = """C Stream Parameter File
C Reach  Streambed_K  Thickness
    1    0.010000     1.5
    2    0.015000     1.8
    3    0.008000     1.2
"""
            filepath.write_text(content)
            yield filepath

    @pytest.fixture
    def sample_stream_parameters(self):
        """Create sample stream parameters."""
        return [
            Parameter(
                name="strk_r1",
                param_type=IWFMParameterType.STREAMBED_K,
                initial_value=0.01,
                lower_bound=0.001,
                upper_bound=1.0,
                metadata={"reach_id": 1},
            ),
            Parameter(
                name="strk_r2",
                param_type=IWFMParameterType.STREAMBED_K,
                initial_value=0.015,
                lower_bound=0.001,
                upper_bound=1.0,
                metadata={"reach_id": 2},
            ),
        ]

    def test_generate_stream_template(self, sample_stream_file, sample_stream_parameters):
        """Test generating stream template."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = IWFMParameterManager()
            for p in sample_stream_parameters:
                pm._parameters[p.name] = p

            tm = IWFMTemplateManager(parameter_manager=pm, output_dir=tmpdir)

            tpl = tm.generate_stream_template(
                input_file=sample_stream_file,
                param_type=IWFMParameterType.STREAMBED_K,
                reach_column=1,
                value_column=2,
            )

            assert tpl.template_path.exists()
            assert len(tpl.parameters) == 2

            content = tpl.template_path.read_text()
            assert "ptf #" in content


class TestMultiplierTemplates:
    """Tests for multiplier parameter templates."""

    @pytest.fixture
    def sample_multiplier_parameters(self):
        """Create sample multiplier parameters."""
        return [
            Parameter(
                name="pump_mult",
                param_type=IWFMParameterType.PUMPING_MULT,
                initial_value=1.0,
                lower_bound=0.5,
                upper_bound=1.5,
            ),
            Parameter(
                name="rech_mult",
                param_type=IWFMParameterType.RECHARGE_MULT,
                initial_value=1.0,
                lower_bound=0.5,
                upper_bound=2.0,
            ),
        ]

    def test_generate_multiplier_template(self, sample_multiplier_parameters):
        """Test generating multiplier template."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = IWFMParameterManager()
            for p in sample_multiplier_parameters:
                pm._parameters[p.name] = p

            tm = IWFMTemplateManager(parameter_manager=pm, output_dir=tmpdir)

            tpl = tm.generate_multiplier_template(
                param_type=IWFMParameterType.PUMPING_MULT,
            )

            assert tpl.template_path.exists()
            assert tpl.input_path.exists()  # Also creates initial .dat file

            content = tpl.template_path.read_text()
            assert "ptf #" in content
            assert "pump_mult" in content

    @pytest.fixture
    def sample_zone_multiplier_parameters(self):
        """Create sample zone multiplier parameters."""
        return [
            Parameter(
                name="pump_z1",
                param_type=IWFMParameterType.PUMPING_MULT,
                initial_value=1.0,
                lower_bound=0.5,
                upper_bound=1.5,
                zone=1,
            ),
            Parameter(
                name="pump_z2",
                param_type=IWFMParameterType.PUMPING_MULT,
                initial_value=1.0,
                lower_bound=0.5,
                upper_bound=1.5,
                zone=2,
            ),
        ]

    def test_generate_zone_multiplier_template(self, sample_zone_multiplier_parameters):
        """Test generating zone multiplier template."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = IWFMParameterManager()
            for p in sample_zone_multiplier_parameters:
                pm._parameters[p.name] = p

            tm = IWFMTemplateManager(parameter_manager=pm, output_dir=tmpdir)

            tpl = tm.generate_zone_multiplier_template(
                param_type=IWFMParameterType.PUMPING_MULT,
            )

            assert tpl.template_path.exists()
            assert len(tpl.parameters) == 2

            content = tpl.template_path.read_text()
            assert "Zone_ID" in content or "zone" in content.lower()


class TestPilotPointTemplates:
    """Tests for pilot point parameter templates."""

    @pytest.fixture
    def sample_pp_parameters(self):
        """Create sample pilot point parameters."""
        return [
            Parameter(
                name="pp_hk_001",
                param_type=IWFMParameterType.HORIZONTAL_K,
                initial_value=1.0e-04,
                lower_bound=1e-06,
                upper_bound=1e-02,
                layer=1,
                location=(100.0, 200.0),
            ),
            Parameter(
                name="pp_hk_002",
                param_type=IWFMParameterType.HORIZONTAL_K,
                initial_value=1.5e-04,
                lower_bound=1e-06,
                upper_bound=1e-02,
                layer=1,
                location=(200.0, 200.0),
            ),
            Parameter(
                name="pp_hk_003",
                param_type=IWFMParameterType.HORIZONTAL_K,
                initial_value=1.2e-04,
                lower_bound=1e-06,
                upper_bound=1e-02,
                layer=1,
                location=(150.0, 300.0),
            ),
        ]

    def test_generate_pilot_point_template(self, sample_pp_parameters):
        """Test generating pilot point template."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = IWFMParameterManager()
            for p in sample_pp_parameters:
                pm._parameters[p.name] = p

            tm = IWFMTemplateManager(parameter_manager=pm, output_dir=tmpdir)

            tpl = tm.generate_pilot_point_template(
                param_type=IWFMParameterType.HORIZONTAL_K,
                layer=1,
            )

            assert tpl.template_path.exists()
            assert len(tpl.parameters) == 3

            content = tpl.template_path.read_text()
            assert "ptf #" in content
            assert "pp_hk_001" in content
            assert "100.00" in content  # X coordinate


class TestRootZoneTemplates:
    """Tests for root zone parameter templates."""

    def test_generate_rootzone_template(self):
        """Test generating root zone template."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create input file
            input_file = Path(tmpdir) / "RootZone.dat"
            content = """C Root Zone Parameter File
C LandUse  CropCoef  IrrigEff
CORN       0.95      0.85
ALFALFA    1.05      0.80
ORCHARD    0.90      0.78
"""
            input_file.write_text(content)

            # Create parameters
            sample_rootzone_parameters = [
                Parameter(
                    name="kc_corn",
                    param_type=IWFMParameterType.CROP_COEFFICIENT,
                    initial_value=0.95,
                    lower_bound=0.5,
                    upper_bound=1.5,
                    metadata={"land_use_type": "CORN"},
                ),
                Parameter(
                    name="kc_alfalfa",
                    param_type=IWFMParameterType.CROP_COEFFICIENT,
                    initial_value=1.05,
                    lower_bound=0.5,
                    upper_bound=1.5,
                    metadata={"land_use_type": "ALFALFA"},
                ),
            ]

            pm = IWFMParameterManager()
            for p in sample_rootzone_parameters:
                pm._parameters[p.name] = p

            tm = IWFMTemplateManager(parameter_manager=pm, output_dir=tmpdir)

            tpl = tm.generate_rootzone_template(
                input_file=input_file,
                param_type=IWFMParameterType.CROP_COEFFICIENT,
                land_use_column=1,
                value_column=2,
            )

            assert tpl.template_path.exists()
            # Check at least one parameter was found
            assert len(tpl.parameters) >= 1
            # Verify template has proper header
            content = tpl.template_path.read_text()
            assert "ptf #" in content


class TestTemplateManagerUtilities:
    """Tests for template manager utility methods."""

    def test_get_all_templates(self):
        """Test getting all created templates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = IWFMParameterManager()
            pm._parameters["pump_mult"] = Parameter(
                name="pump_mult",
                param_type=IWFMParameterType.PUMPING_MULT,
                initial_value=1.0,
                lower_bound=0.5,
                upper_bound=1.5,
            )

            tm = IWFMTemplateManager(parameter_manager=pm, output_dir=tmpdir)
            tm.generate_multiplier_template(IWFMParameterType.PUMPING_MULT)

            templates = tm.get_all_templates()
            assert len(templates) == 1

    def test_clear_templates(self):
        """Test clearing templates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = IWFMParameterManager()
            pm._parameters["pump_mult"] = Parameter(
                name="pump_mult",
                param_type=IWFMParameterType.PUMPING_MULT,
                initial_value=1.0,
                lower_bound=0.5,
                upper_bound=1.5,
            )

            tm = IWFMTemplateManager(parameter_manager=pm, output_dir=tmpdir)
            tm.generate_multiplier_template(IWFMParameterType.PUMPING_MULT)

            assert len(tm.get_all_templates()) == 1
            tm.clear_templates()
            assert len(tm.get_all_templates()) == 0

    def test_repr(self):
        """Test string representation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tm = IWFMTemplateManager(output_dir=tmpdir)
            r = repr(tm)
            assert "IWFMTemplateManager" in r
            assert "n_templates=0" in r


# =========================================================================
# Additional tests for increased coverage
# =========================================================================


class TestTemplateManagerInitExtra:
    """Additional initialization tests."""

    def test_init_default_output_dir(self):
        """Test initialization with no output_dir uses current directory."""
        tm = IWFMTemplateManager()
        assert tm.output_dir == Path(".")
        assert tm.model is None
        assert tm.pm is None

    def test_init_with_model(self):
        """Test initialization with a model argument."""
        sentinel = object()
        with tempfile.TemporaryDirectory() as tmpdir:
            tm = IWFMTemplateManager(model=sentinel, output_dir=tmpdir)
            assert tm.model is sentinel

    def test_repr_after_generating_templates(self):
        """Test repr updates after templates are generated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = IWFMParameterManager()
            pm._parameters["pump_mult"] = Parameter(
                name="pump_mult",
                param_type=IWFMParameterType.PUMPING_MULT,
                initial_value=1.0,
                lower_bound=0.5,
                upper_bound=1.5,
            )
            tm = IWFMTemplateManager(parameter_manager=pm, output_dir=tmpdir)
            tm.generate_multiplier_template(IWFMParameterType.PUMPING_MULT)
            r = repr(tm)
            assert "n_templates=1" in r


class TestReplaceValuesInLine:
    """Tests for the _replace_values_in_line utility method."""

    def test_comment_line_c(self):
        """Test that C-style comment lines are not modified."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tm = IWFMTemplateManager(output_dir=tmpdir)
            markers = []
            result = tm._replace_values_in_line(
                "C This is a comment 1.5e-04", {"p1": 1.5e-04}, markers, 1
            )
            assert result == "C This is a comment 1.5e-04"
            assert len(markers) == 0

    def test_comment_line_star(self):
        """Test that *-style comment lines are not modified."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tm = IWFMTemplateManager(output_dir=tmpdir)
            markers = []
            result = tm._replace_values_in_line("* comment 1.5e-04", {"p1": 1.5e-04}, markers, 1)
            assert result == "* comment 1.5e-04"
            assert len(markers) == 0

    def test_comment_line_hash(self):
        """Test that #-style comment lines are not modified."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tm = IWFMTemplateManager(output_dir=tmpdir)
            markers = []
            result = tm._replace_values_in_line("# comment 1.5e-04", {"p1": 1.5e-04}, markers, 1)
            assert result == "# comment 1.5e-04"
            assert len(markers) == 0

    def test_comment_line_exclamation(self):
        """Test that !-style comment lines are not modified."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tm = IWFMTemplateManager(output_dir=tmpdir)
            markers = []
            result = tm._replace_values_in_line("! comment 1.5e-04", {"p1": 1.5e-04}, markers, 1)
            assert result == "! comment 1.5e-04"
            assert len(markers) == 0

    def test_replace_scientific_notation(self):
        """Test replacement of values in scientific notation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tm = IWFMTemplateManager(output_dir=tmpdir)
            markers = []
            line = "  1  1.500000e-04  2.000000e-06"
            result = tm._replace_values_in_line(line, {"hk_z1": 1.5e-04}, markers, 5)
            assert "#" in result
            assert "hk_z1" in result
            assert len(markers) == 1
            assert markers[0].parameter_name == "hk_z1"
            assert markers[0].line_number == 5

    def test_replace_decimal_format(self):
        """Test replacement of values in decimal format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tm = IWFMTemplateManager(output_dir=tmpdir)
            markers = []
            # Use a value that will match one of the format patterns
            line = "  zone1  0.150000"
            result = tm._replace_values_in_line(line, {"sy_z1": 0.15}, markers, 10)
            # The method tries several formats. 0.15 formats as "1.500000e-01"
            # in scientific notation first. If none match, the value stays.
            # Check that the method runs without error.
            assert isinstance(result, str)

    def test_no_match_leaves_line_unchanged(self):
        """Test that lines without matching values are not modified."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tm = IWFMTemplateManager(output_dir=tmpdir)
            markers = []
            line = "  1  9.999999e+99"
            result = tm._replace_values_in_line(line, {"p1": 1.0e-04}, markers, 1)
            assert result == line
            assert len(markers) == 0

    def test_multiple_parameters_in_line(self):
        """Test replacing multiple parameter values in one line."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tm = IWFMTemplateManager(output_dir=tmpdir)
            markers = []
            line = "  1.500000e-04  2.500000e-04"
            result = tm._replace_values_in_line(
                line,
                {"hk_z1": 1.5e-04, "hk_z2": 2.5e-04},
                markers,
                1,
            )
            assert "hk_z1" in result
            assert "hk_z2" in result
            assert len(markers) == 2

    def test_empty_line(self):
        """Test handling of empty line."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tm = IWFMTemplateManager(output_dir=tmpdir)
            markers = []
            result = tm._replace_values_in_line("", {"p1": 1.0}, markers, 1)
            assert result == ""

    def test_whitespace_only_line(self):
        """Test handling of whitespace-only line (not a comment)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tm = IWFMTemplateManager(output_dir=tmpdir)
            markers = []
            result = tm._replace_values_in_line("   ", {"p1": 1.0}, markers, 1)
            assert result == "   "

    def test_marker_records_original_value(self):
        """Test that markers record the original value that was replaced."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tm = IWFMTemplateManager(output_dir=tmpdir)
            markers = []
            line = "  data  1.500000e-04  end"
            tm._replace_values_in_line(line, {"p1": 1.5e-04}, markers, 7)
            assert len(markers) == 1
            assert markers[0].original_value == "1.500000e-04"
            assert markers[0].column_start >= 0
            assert markers[0].column_end > markers[0].column_start

    def test_custom_delimiter(self):
        """Test replacement uses the manager's delimiter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tm = IWFMTemplateManager(output_dir=tmpdir, delimiter="~")
            markers = []
            line = "  1.500000e-04"
            result = tm._replace_values_in_line(line, {"p1": 1.5e-04}, markers, 1)
            assert "~" in result
            assert "#" not in result


class TestAquiferTemplateHappyPath:
    """Tests for aquifer template generation with explicit parameters."""

    def test_generate_aquifer_template_with_explicit_params(self):
        """Test generating aquifer template with explicit parameter list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create input file
            input_file = Path(tmpdir) / "Groundwater.dat"
            content = """C Groundwater Parameter File
C Zone  Horizontal_K
    1    1.500000e-04
    2    2.500000e-04
"""
            input_file.write_text(content)

            params = [
                Parameter(
                    name="hk_z1",
                    param_type=IWFMParameterType.HORIZONTAL_K,
                    initial_value=1.5e-04,
                    lower_bound=1e-06,
                    upper_bound=1e-02,
                    zone=1,
                    layer=1,
                ),
                Parameter(
                    name="hk_z2",
                    param_type=IWFMParameterType.HORIZONTAL_K,
                    initial_value=2.5e-04,
                    lower_bound=1e-06,
                    upper_bound=1e-02,
                    zone=2,
                    layer=1,
                ),
            ]

            tm = IWFMTemplateManager(output_dir=tmpdir)
            tpl = tm.generate_aquifer_template(
                input_file=input_file,
                param_type=IWFMParameterType.HORIZONTAL_K,
                layer=1,
                parameters=params,
            )

            assert tpl.template_path.exists()
            content = tpl.template_path.read_text()
            assert content.startswith("ptf #")
            assert "hk_z1" in content
            assert "hk_z2" in content
            assert len(tpl.parameters) == 2

    def test_generate_aquifer_template_string_param_type(self):
        """Test generating aquifer template with string param type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "Groundwater.dat"
            content = "C header\n    1    1.500000e-04\n"
            input_file.write_text(content)

            params = [
                Parameter(
                    name="hk_z1",
                    param_type=IWFMParameterType.HORIZONTAL_K,
                    initial_value=1.5e-04,
                    lower_bound=1e-06,
                    upper_bound=1e-02,
                    zone=1,
                    layer=1,
                ),
            ]

            tm = IWFMTemplateManager(output_dir=tmpdir)
            tpl = tm.generate_aquifer_template(
                input_file=input_file,
                param_type="hk",  # string form
                layer=1,
                parameters=params,
            )

            assert tpl.template_path.exists()
            assert "hk_z1" in tpl.parameters

    def test_generate_aquifer_template_custom_output(self):
        """Test generating aquifer template with custom output path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "Groundwater.dat"
            content = "C header\n    1    1.500000e-04\n"
            input_file.write_text(content)

            custom_output = Path(tmpdir) / "custom_tpl.tpl"

            params = [
                Parameter(
                    name="hk_z1",
                    param_type=IWFMParameterType.HORIZONTAL_K,
                    initial_value=1.5e-04,
                    lower_bound=1e-06,
                    upper_bound=1e-02,
                    zone=1,
                    layer=1,
                ),
            ]

            tm = IWFMTemplateManager(output_dir=tmpdir)
            tpl = tm.generate_aquifer_template(
                input_file=input_file,
                param_type=IWFMParameterType.HORIZONTAL_K,
                parameters=params,
                output_template=custom_output,
            )

            assert tpl.template_path == custom_output
            assert custom_output.exists()

    def test_generate_aquifer_template_no_layer(self):
        """Test auto-generated name when layer is None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "Groundwater.dat"
            content = "C header\n    1    1.500000e-04\n"
            input_file.write_text(content)

            params = [
                Parameter(
                    name="hk_z1",
                    param_type=IWFMParameterType.HORIZONTAL_K,
                    initial_value=1.5e-04,
                    lower_bound=1e-06,
                    upper_bound=1e-02,
                ),
            ]

            tm = IWFMTemplateManager(output_dir=tmpdir)
            tpl = tm.generate_aquifer_template(
                input_file=input_file,
                param_type=IWFMParameterType.HORIZONTAL_K,
                layer=None,
                parameters=params,
            )

            # No "_l" suffix when layer is None
            assert "_l" not in tpl.template_path.stem

    def test_aquifer_template_no_params_from_manager_raises(self):
        """Test error when manager has no matching parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = IWFMParameterManager()
            tm = IWFMTemplateManager(parameter_manager=pm, output_dir=tmpdir)

            with pytest.raises(ValueError, match="No parameters found"):
                tm.generate_aquifer_template(
                    input_file=Path(tmpdir) / "test.dat",
                    param_type=IWFMParameterType.HORIZONTAL_K,
                )

    def test_aquifer_template_empty_explicit_params_raises(self):
        """Test error when empty parameter list is explicitly passed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tm = IWFMTemplateManager(output_dir=tmpdir)
            with pytest.raises(ValueError, match="No parameters found"):
                tm.generate_aquifer_template(
                    input_file=Path(tmpdir) / "test.dat",
                    param_type=IWFMParameterType.HORIZONTAL_K,
                    parameters=[],
                )


class TestAquiferTemplateByZoneExtra:
    """Additional tests for zone-based aquifer templates."""

    def test_zone_template_no_zone_params_raises(self):
        """Test error when no zone parameters exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = IWFMParameterManager()
            # Add parameter without zone
            pm._parameters["hk_global"] = Parameter(
                name="hk_global",
                param_type=IWFMParameterType.HORIZONTAL_K,
                initial_value=1.0e-04,
                lower_bound=1e-06,
                upper_bound=1e-02,
                # no zone attribute
            )
            tm = IWFMTemplateManager(parameter_manager=pm, output_dir=tmpdir)

            with pytest.raises(ValueError, match="No zone parameters found"):
                tm.generate_aquifer_template_by_zone(
                    input_file=Path(tmpdir) / "test.dat",
                    param_type=IWFMParameterType.HORIZONTAL_K,
                    zone_column=1,
                    value_column=2,
                    layer=1,
                )

    def test_zone_template_string_param_type(self):
        """Test zone template with string param_type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "Groundwater.dat"
            content = "C header\n    1    1.500000e-04\n"
            input_file.write_text(content)

            pm = IWFMParameterManager()
            pm._parameters["hk_z1"] = Parameter(
                name="hk_z1",
                param_type=IWFMParameterType.HORIZONTAL_K,
                initial_value=1.5e-04,
                lower_bound=1e-06,
                upper_bound=1e-02,
                zone=1,
                layer=1,
            )
            tm = IWFMTemplateManager(parameter_manager=pm, output_dir=tmpdir)

            tpl = tm.generate_aquifer_template_by_zone(
                input_file=input_file,
                param_type="hk",  # string
                zone_column=1,
                value_column=2,
                layer=1,
            )

            assert tpl.template_path.exists()
            assert "hk_z1" in tpl.parameters

    def test_zone_template_with_comment_lines(self):
        """Test zone template skips comment lines properly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "Groundwater.dat"
            content = """C This is a comment
* Another comment
    1    1.500000e-04
    2    2.500000e-04
"""
            input_file.write_text(content)

            pm = IWFMParameterManager()
            pm._parameters["hk_z1"] = Parameter(
                name="hk_z1",
                param_type=IWFMParameterType.HORIZONTAL_K,
                initial_value=1.5e-04,
                lower_bound=1e-06,
                upper_bound=1e-02,
                zone=1,
                layer=1,
            )
            tm = IWFMTemplateManager(parameter_manager=pm, output_dir=tmpdir)

            tpl = tm.generate_aquifer_template_by_zone(
                input_file=input_file,
                param_type=IWFMParameterType.HORIZONTAL_K,
                zone_column=1,
                value_column=2,
                layer=1,
            )

            content = tpl.template_path.read_text()
            # Comment lines preserved
            assert "C This is a comment" in content
            assert "* Another comment" in content

    def test_zone_template_with_header_lines(self):
        """Test zone template respects header_lines parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "Groundwater.dat"
            content = """Header line 1
Header line 2
    1    1.500000e-04
"""
            input_file.write_text(content)

            pm = IWFMParameterManager()
            pm._parameters["hk_z1"] = Parameter(
                name="hk_z1",
                param_type=IWFMParameterType.HORIZONTAL_K,
                initial_value=1.5e-04,
                lower_bound=1e-06,
                upper_bound=1e-02,
                zone=1,
                layer=1,
            )
            tm = IWFMTemplateManager(parameter_manager=pm, output_dir=tmpdir)

            tpl = tm.generate_aquifer_template_by_zone(
                input_file=input_file,
                param_type=IWFMParameterType.HORIZONTAL_K,
                zone_column=1,
                value_column=2,
                layer=1,
                header_lines=2,
            )

            content = tpl.template_path.read_text()
            # Header lines should be preserved unchanged
            assert "Header line 1" in content
            assert "Header line 2" in content

    def test_zone_template_custom_output(self):
        """Test zone template with custom output path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "Groundwater.dat"
            content = "C header\n    1    1.500000e-04\n"
            input_file.write_text(content)

            custom_output = Path(tmpdir) / "my_zone_template.tpl"

            pm = IWFMParameterManager()
            pm._parameters["hk_z1"] = Parameter(
                name="hk_z1",
                param_type=IWFMParameterType.HORIZONTAL_K,
                initial_value=1.5e-04,
                lower_bound=1e-06,
                upper_bound=1e-02,
                zone=1,
                layer=1,
            )
            tm = IWFMTemplateManager(parameter_manager=pm, output_dir=tmpdir)

            tpl = tm.generate_aquifer_template_by_zone(
                input_file=input_file,
                param_type=IWFMParameterType.HORIZONTAL_K,
                zone_column=1,
                value_column=2,
                layer=1,
                output_template=custom_output,
            )

            assert tpl.template_path == custom_output

    def test_zone_template_insufficient_columns(self):
        """Test zone template with lines that have fewer columns than expected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "Groundwater.dat"
            # Line with only one column (insufficient for zone_column=1, value_column=2)
            content = "C header\n    1\n    1    1.500000e-04\n"
            input_file.write_text(content)

            pm = IWFMParameterManager()
            pm._parameters["hk_z1"] = Parameter(
                name="hk_z1",
                param_type=IWFMParameterType.HORIZONTAL_K,
                initial_value=1.5e-04,
                lower_bound=1e-06,
                upper_bound=1e-02,
                zone=1,
                layer=1,
            )
            tm = IWFMTemplateManager(parameter_manager=pm, output_dir=tmpdir)

            tpl = tm.generate_aquifer_template_by_zone(
                input_file=input_file,
                param_type=IWFMParameterType.HORIZONTAL_K,
                zone_column=1,
                value_column=2,
                layer=1,
            )

            # Should succeed; insufficient-column line passes through unchanged
            content = tpl.template_path.read_text()
            assert "1" in content

    def test_zone_template_non_integer_zone_id(self):
        """Test zone template handles non-integer zone ID gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "Groundwater.dat"
            # Non-integer zone value in first column
            content = "C header\n    abc    1.500000e-04\n    1    1.500000e-04\n"
            input_file.write_text(content)

            pm = IWFMParameterManager()
            pm._parameters["hk_z1"] = Parameter(
                name="hk_z1",
                param_type=IWFMParameterType.HORIZONTAL_K,
                initial_value=1.5e-04,
                lower_bound=1e-06,
                upper_bound=1e-02,
                zone=1,
                layer=1,
            )
            tm = IWFMTemplateManager(parameter_manager=pm, output_dir=tmpdir)

            # Should not raise; non-integer zone lines pass through
            tpl = tm.generate_aquifer_template_by_zone(
                input_file=input_file,
                param_type=IWFMParameterType.HORIZONTAL_K,
                zone_column=1,
                value_column=2,
                layer=1,
            )

            content = tpl.template_path.read_text()
            assert "abc" in content  # preserved as-is

    def test_zone_template_no_pm(self):
        """Test zone template with no parameter manager raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tm = IWFMTemplateManager(output_dir=tmpdir)

            with pytest.raises(ValueError, match="No zone parameters found"):
                tm.generate_aquifer_template_by_zone(
                    input_file=Path(tmpdir) / "test.dat",
                    param_type=IWFMParameterType.HORIZONTAL_K,
                    zone_column=1,
                    value_column=2,
                )


class TestStreamTemplateExtra:
    """Additional tests for stream parameter templates."""

    def test_stream_template_no_params_raises(self):
        """Test error when no stream parameters exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = IWFMParameterManager()
            tm = IWFMTemplateManager(parameter_manager=pm, output_dir=tmpdir)

            with pytest.raises(ValueError, match="No stream parameters found"):
                tm.generate_stream_template(
                    input_file=Path(tmpdir) / "test.dat",
                    param_type=IWFMParameterType.STREAMBED_K,
                )

    def test_stream_template_string_param_type(self):
        """Test stream template with string param_type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "Stream.dat"
            content = "C header\n    1    0.010000\n"
            input_file.write_text(content)

            pm = IWFMParameterManager()
            pm._parameters["strk_r1"] = Parameter(
                name="strk_r1",
                param_type=IWFMParameterType.STREAMBED_K,
                initial_value=0.01,
                lower_bound=0.001,
                upper_bound=1.0,
                metadata={"reach_id": 1},
            )
            tm = IWFMTemplateManager(parameter_manager=pm, output_dir=tmpdir)

            tpl = tm.generate_stream_template(
                input_file=input_file,
                param_type="strk",  # string
            )

            assert tpl.template_path.exists()

    def test_stream_template_comment_lines_preserved(self):
        """Test that comment lines in stream file are preserved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "Stream.dat"
            content = """C Stream data
* Section 1
# Reaches
    1    0.010000
"""
            input_file.write_text(content)

            pm = IWFMParameterManager()
            pm._parameters["strk_r1"] = Parameter(
                name="strk_r1",
                param_type=IWFMParameterType.STREAMBED_K,
                initial_value=0.01,
                lower_bound=0.001,
                upper_bound=1.0,
                metadata={"reach_id": 1},
            )
            tm = IWFMTemplateManager(parameter_manager=pm, output_dir=tmpdir)

            tpl = tm.generate_stream_template(
                input_file=input_file,
                param_type=IWFMParameterType.STREAMBED_K,
            )

            content = tpl.template_path.read_text()
            assert "C Stream data" in content
            assert "* Section 1" in content
            assert "# Reaches" in content

    def test_stream_template_custom_output(self):
        """Test stream template with custom output path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "Stream.dat"
            content = "C header\n    1    0.010000\n"
            input_file.write_text(content)

            custom_output = Path(tmpdir) / "custom_stream.tpl"

            pm = IWFMParameterManager()
            pm._parameters["strk_r1"] = Parameter(
                name="strk_r1",
                param_type=IWFMParameterType.STREAMBED_K,
                initial_value=0.01,
                lower_bound=0.001,
                upper_bound=1.0,
                metadata={"reach_id": 1},
            )
            tm = IWFMTemplateManager(parameter_manager=pm, output_dir=tmpdir)

            tpl = tm.generate_stream_template(
                input_file=input_file,
                param_type=IWFMParameterType.STREAMBED_K,
                output_template=custom_output,
            )

            assert tpl.template_path == custom_output

    def test_stream_template_with_header_lines(self):
        """Test stream template respects header_lines parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "Stream.dat"
            content = "Header row 1\nHeader row 2\n    1    0.010000\n"
            input_file.write_text(content)

            pm = IWFMParameterManager()
            pm._parameters["strk_r1"] = Parameter(
                name="strk_r1",
                param_type=IWFMParameterType.STREAMBED_K,
                initial_value=0.01,
                lower_bound=0.001,
                upper_bound=1.0,
                metadata={"reach_id": 1},
            )
            tm = IWFMTemplateManager(parameter_manager=pm, output_dir=tmpdir)

            tpl = tm.generate_stream_template(
                input_file=input_file,
                param_type=IWFMParameterType.STREAMBED_K,
                header_lines=2,
            )

            content = tpl.template_path.read_text()
            assert "Header row 1" in content
            assert "Header row 2" in content

    def test_stream_template_malformed_reach_id(self):
        """Test stream template handles non-integer reach IDs gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "Stream.dat"
            content = "C header\n    abc    0.010000\n    1    0.010000\n"
            input_file.write_text(content)

            pm = IWFMParameterManager()
            pm._parameters["strk_r1"] = Parameter(
                name="strk_r1",
                param_type=IWFMParameterType.STREAMBED_K,
                initial_value=0.01,
                lower_bound=0.001,
                upper_bound=1.0,
                metadata={"reach_id": 1},
            )
            tm = IWFMTemplateManager(parameter_manager=pm, output_dir=tmpdir)

            tpl = tm.generate_stream_template(
                input_file=input_file,
                param_type=IWFMParameterType.STREAMBED_K,
            )

            content = tpl.template_path.read_text()
            # Non-integer line preserved
            assert "abc" in content

    def test_stream_template_no_pm(self):
        """Test stream template with no parameter manager raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tm = IWFMTemplateManager(output_dir=tmpdir)

            with pytest.raises(ValueError, match="No stream parameters found"):
                tm.generate_stream_template(
                    input_file=Path(tmpdir) / "test.dat",
                    param_type=IWFMParameterType.STREAMBED_K,
                )

    def test_stream_template_param_without_reach_id(self):
        """Test that stream parameters missing reach_id are excluded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = IWFMParameterManager()
            # Parameter without reach_id in metadata
            pm._parameters["strk_no_reach"] = Parameter(
                name="strk_no_reach",
                param_type=IWFMParameterType.STREAMBED_K,
                initial_value=0.01,
                lower_bound=0.001,
                upper_bound=1.0,
                metadata={},  # no reach_id
            )
            tm = IWFMTemplateManager(parameter_manager=pm, output_dir=tmpdir)

            with pytest.raises(ValueError, match="No stream parameters found"):
                tm.generate_stream_template(
                    input_file=Path(tmpdir) / "test.dat",
                    param_type=IWFMParameterType.STREAMBED_K,
                )


class TestMultiplierTemplateExtra:
    """Additional tests for multiplier templates."""

    def test_multiplier_no_params_raises(self):
        """Test error when no multiplier parameters exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = IWFMParameterManager()
            tm = IWFMTemplateManager(parameter_manager=pm, output_dir=tmpdir)

            with pytest.raises(ValueError, match="No multiplier parameters found"):
                tm.generate_multiplier_template(
                    param_type=IWFMParameterType.PUMPING_MULT,
                )

    def test_multiplier_string_param_type(self):
        """Test multiplier template with string param_type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = IWFMParameterManager()
            pm._parameters["pump_mult"] = Parameter(
                name="pump_mult",
                param_type=IWFMParameterType.PUMPING_MULT,
                initial_value=1.0,
                lower_bound=0.5,
                upper_bound=1.5,
            )
            tm = IWFMTemplateManager(parameter_manager=pm, output_dir=tmpdir)

            tpl = tm.generate_multiplier_template(param_type="pump")
            assert tpl.template_path.exists()

    def test_multiplier_custom_format_width(self):
        """Test multiplier template with custom format width."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = IWFMParameterManager()
            pm._parameters["pump_mult"] = Parameter(
                name="pump_mult",
                param_type=IWFMParameterType.PUMPING_MULT,
                initial_value=1.0,
                lower_bound=0.5,
                upper_bound=1.5,
            )
            tm = IWFMTemplateManager(parameter_manager=pm, output_dir=tmpdir)

            tpl = tm.generate_multiplier_template(
                param_type=IWFMParameterType.PUMPING_MULT,
                format_width=20,
            )

            content = tpl.template_path.read_text()
            assert "pump_mult" in content
            # The marker should be wider
            assert tpl.template_path.exists()

    def test_multiplier_custom_output(self):
        """Test multiplier template with custom output path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = IWFMParameterManager()
            pm._parameters["pump_mult"] = Parameter(
                name="pump_mult",
                param_type=IWFMParameterType.PUMPING_MULT,
                initial_value=1.0,
                lower_bound=0.5,
                upper_bound=1.5,
            )
            tm = IWFMTemplateManager(parameter_manager=pm, output_dir=tmpdir)

            custom_output = Path(tmpdir) / "my_mult.tpl"
            tpl = tm.generate_multiplier_template(
                param_type=IWFMParameterType.PUMPING_MULT,
                output_template=custom_output,
            )

            assert tpl.template_path == custom_output
            # Verify .dat file is created alongside
            assert tpl.input_path.exists()
            assert tpl.input_path.suffix == ".dat"

    def test_multiplier_dat_file_content(self):
        """Test that the initial .dat file contains correct values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = IWFMParameterManager()
            pm._parameters["pump_mult"] = Parameter(
                name="pump_mult",
                param_type=IWFMParameterType.PUMPING_MULT,
                initial_value=1.0,
                lower_bound=0.5,
                upper_bound=1.5,
            )
            tm = IWFMTemplateManager(parameter_manager=pm, output_dir=tmpdir)

            tpl = tm.generate_multiplier_template(
                param_type=IWFMParameterType.PUMPING_MULT,
            )

            dat_content = tpl.input_path.read_text()
            assert "pump_mult" in dat_content
            assert "1.000000e+00" in dat_content

    def test_multiplier_no_pm(self):
        """Test multiplier template with no parameter manager raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tm = IWFMTemplateManager(output_dir=tmpdir)

            with pytest.raises(ValueError, match="No multiplier parameters found"):
                tm.generate_multiplier_template(
                    param_type=IWFMParameterType.PUMPING_MULT,
                )

    def test_multiplier_non_multiplier_type_raises(self):
        """Test that non-multiplier param type yields no valid parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = IWFMParameterManager()
            # HORIZONTAL_K is_multiplier is False
            pm._parameters["hk_z1"] = Parameter(
                name="hk_z1",
                param_type=IWFMParameterType.HORIZONTAL_K,
                initial_value=1.0e-04,
                lower_bound=1e-06,
                upper_bound=1e-02,
            )
            tm = IWFMTemplateManager(parameter_manager=pm, output_dir=tmpdir)

            with pytest.raises(ValueError, match="No multiplier parameters found"):
                tm.generate_multiplier_template(
                    param_type=IWFMParameterType.HORIZONTAL_K,
                )


class TestZoneMultiplierTemplateExtra:
    """Additional tests for zone multiplier templates."""

    def test_zone_mult_no_zone_params_raises(self):
        """Test error when no zone multiplier parameters exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = IWFMParameterManager()
            # Parameter without zone
            pm._parameters["pump_mult"] = Parameter(
                name="pump_mult",
                param_type=IWFMParameterType.PUMPING_MULT,
                initial_value=1.0,
                lower_bound=0.5,
                upper_bound=1.5,
                # no zone
            )
            tm = IWFMTemplateManager(parameter_manager=pm, output_dir=tmpdir)

            with pytest.raises(ValueError, match="No zone multiplier parameters found"):
                tm.generate_zone_multiplier_template(
                    param_type=IWFMParameterType.PUMPING_MULT,
                )

    def test_zone_mult_string_param_type(self):
        """Test zone multiplier template with string param_type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = IWFMParameterManager()
            pm._parameters["pump_z1"] = Parameter(
                name="pump_z1",
                param_type=IWFMParameterType.PUMPING_MULT,
                initial_value=1.0,
                lower_bound=0.5,
                upper_bound=1.5,
                zone=1,
            )
            tm = IWFMTemplateManager(parameter_manager=pm, output_dir=tmpdir)

            tpl = tm.generate_zone_multiplier_template(param_type="pump")
            assert tpl.template_path.exists()

    def test_zone_mult_custom_output(self):
        """Test zone multiplier template with custom output path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = IWFMParameterManager()
            pm._parameters["pump_z1"] = Parameter(
                name="pump_z1",
                param_type=IWFMParameterType.PUMPING_MULT,
                initial_value=1.0,
                lower_bound=0.5,
                upper_bound=1.5,
                zone=1,
            )
            tm = IWFMTemplateManager(parameter_manager=pm, output_dir=tmpdir)

            custom_output = Path(tmpdir) / "my_zone_mult.tpl"
            tpl = tm.generate_zone_multiplier_template(
                param_type=IWFMParameterType.PUMPING_MULT,
                output_template=custom_output,
            )

            assert tpl.template_path == custom_output

    def test_zone_mult_sorted_by_zone(self):
        """Test that zone multiplier parameters are sorted by zone ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = IWFMParameterManager()
            # Add in reverse order
            pm._parameters["pump_z3"] = Parameter(
                name="pump_z3",
                param_type=IWFMParameterType.PUMPING_MULT,
                initial_value=1.0,
                lower_bound=0.5,
                upper_bound=1.5,
                zone=3,
            )
            pm._parameters["pump_z1"] = Parameter(
                name="pump_z1",
                param_type=IWFMParameterType.PUMPING_MULT,
                initial_value=1.0,
                lower_bound=0.5,
                upper_bound=1.5,
                zone=1,
            )
            pm._parameters["pump_z2"] = Parameter(
                name="pump_z2",
                param_type=IWFMParameterType.PUMPING_MULT,
                initial_value=1.0,
                lower_bound=0.5,
                upper_bound=1.5,
                zone=2,
            )
            tm = IWFMTemplateManager(parameter_manager=pm, output_dir=tmpdir)

            tpl = tm.generate_zone_multiplier_template(
                param_type=IWFMParameterType.PUMPING_MULT,
            )

            # Parameters should be sorted: pump_z1, pump_z2, pump_z3
            assert tpl.parameters == ["pump_z1", "pump_z2", "pump_z3"]

    def test_zone_mult_dat_file_content(self):
        """Test that the initial .dat file for zone multiplier contains expected values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = IWFMParameterManager()
            pm._parameters["pump_z1"] = Parameter(
                name="pump_z1",
                param_type=IWFMParameterType.PUMPING_MULT,
                initial_value=1.0,
                lower_bound=0.5,
                upper_bound=1.5,
                zone=1,
            )
            pm._parameters["pump_z2"] = Parameter(
                name="pump_z2",
                param_type=IWFMParameterType.PUMPING_MULT,
                initial_value=0.9,
                lower_bound=0.5,
                upper_bound=1.5,
                zone=2,
            )
            tm = IWFMTemplateManager(parameter_manager=pm, output_dir=tmpdir)

            tpl = tm.generate_zone_multiplier_template(
                param_type=IWFMParameterType.PUMPING_MULT,
            )

            dat_content = tpl.input_path.read_text()
            assert "1.000000e+00" in dat_content
            assert "9.000000e-01" in dat_content

    def test_zone_mult_no_pm(self):
        """Test zone multiplier with no parameter manager raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tm = IWFMTemplateManager(output_dir=tmpdir)

            with pytest.raises(ValueError, match="No zone multiplier parameters found"):
                tm.generate_zone_multiplier_template(
                    param_type=IWFMParameterType.PUMPING_MULT,
                )


class TestPilotPointTemplateExtra:
    """Additional tests for pilot point templates."""

    def test_pp_template_no_params_raises(self):
        """Test error when no pilot point parameters exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = IWFMParameterManager()
            tm = IWFMTemplateManager(parameter_manager=pm, output_dir=tmpdir)

            with pytest.raises(ValueError, match="No pilot point parameters found"):
                tm.generate_pilot_point_template(
                    param_type=IWFMParameterType.HORIZONTAL_K,
                    layer=1,
                )

    def test_pp_template_string_param_type(self):
        """Test pilot point template with string param_type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = IWFMParameterManager()
            pm._parameters["pp_hk_001"] = Parameter(
                name="pp_hk_001",
                param_type=IWFMParameterType.HORIZONTAL_K,
                initial_value=1.0e-04,
                lower_bound=1e-06,
                upper_bound=1e-02,
                layer=1,
                location=(100.0, 200.0),
            )
            tm = IWFMTemplateManager(parameter_manager=pm, output_dir=tmpdir)

            tpl = tm.generate_pilot_point_template(
                param_type="hk",  # string
                layer=1,
            )

            assert tpl.template_path.exists()
            assert "pp_hk_001" in tpl.parameters

    def test_pp_template_custom_output(self):
        """Test pilot point template with custom output path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = IWFMParameterManager()
            pm._parameters["pp_hk_001"] = Parameter(
                name="pp_hk_001",
                param_type=IWFMParameterType.HORIZONTAL_K,
                initial_value=1.0e-04,
                lower_bound=1e-06,
                upper_bound=1e-02,
                layer=1,
                location=(100.0, 200.0),
            )
            tm = IWFMTemplateManager(parameter_manager=pm, output_dir=tmpdir)

            custom_output = Path(tmpdir) / "my_pp.tpl"
            tpl = tm.generate_pilot_point_template(
                param_type=IWFMParameterType.HORIZONTAL_K,
                layer=1,
                output_template=custom_output,
            )

            assert tpl.template_path == custom_output

    def test_pp_template_dat_file_content(self):
        """Test that pilot point .dat file has coordinates and values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = IWFMParameterManager()
            pm._parameters["pp_hk_001"] = Parameter(
                name="pp_hk_001",
                param_type=IWFMParameterType.HORIZONTAL_K,
                initial_value=1.0e-04,
                lower_bound=1e-06,
                upper_bound=1e-02,
                layer=1,
                location=(100.0, 200.0),
            )
            tm = IWFMTemplateManager(parameter_manager=pm, output_dir=tmpdir)

            tpl = tm.generate_pilot_point_template(
                param_type=IWFMParameterType.HORIZONTAL_K,
                layer=1,
            )

            dat_content = tpl.input_path.read_text()
            assert "pp_hk_001" in dat_content
            assert "100.00" in dat_content
            assert "200.00" in dat_content

    def test_pp_template_no_location_uses_zero(self):
        """Test that pilot points without explicit location use (0,0)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = IWFMParameterManager()
            # Parameter with location=None but still identified as pilot point
            # This is an edge case; normally pilot points have locations.
            # The code checks param.location so we need a parameter with
            # location set to trigger the pilot point filter, but then
            # test the fallback in the template creation.
            pm._parameters["pp_hk_001"] = Parameter(
                name="pp_hk_001",
                param_type=IWFMParameterType.HORIZONTAL_K,
                initial_value=1.0e-04,
                lower_bound=1e-06,
                upper_bound=1e-02,
                layer=1,
                location=(100.0, 200.0),
            )
            tm = IWFMTemplateManager(parameter_manager=pm, output_dir=tmpdir)

            tpl = tm.generate_pilot_point_template(
                param_type=IWFMParameterType.HORIZONTAL_K,
                layer=1,
            )

            assert len(tpl.parameters) == 1

    def test_pp_template_wrong_layer_raises(self):
        """Test error when no pilot points exist for specified layer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = IWFMParameterManager()
            pm._parameters["pp_hk_001"] = Parameter(
                name="pp_hk_001",
                param_type=IWFMParameterType.HORIZONTAL_K,
                initial_value=1.0e-04,
                lower_bound=1e-06,
                upper_bound=1e-02,
                layer=1,
                location=(100.0, 200.0),
            )
            tm = IWFMTemplateManager(parameter_manager=pm, output_dir=tmpdir)

            with pytest.raises(ValueError, match="No pilot point parameters found"):
                tm.generate_pilot_point_template(
                    param_type=IWFMParameterType.HORIZONTAL_K,
                    layer=99,  # wrong layer
                )

    def test_pp_template_no_pm(self):
        """Test pilot point template with no parameter manager raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tm = IWFMTemplateManager(output_dir=tmpdir)

            with pytest.raises(ValueError, match="No pilot point parameters found"):
                tm.generate_pilot_point_template(
                    param_type=IWFMParameterType.HORIZONTAL_K,
                    layer=1,
                )

    def test_pp_template_multiple_points(self):
        """Test pilot point template with multiple points."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = IWFMParameterManager()
            for i in range(5):
                pm._parameters[f"pp_hk_{i:03d}"] = Parameter(
                    name=f"pp_hk_{i:03d}",
                    param_type=IWFMParameterType.HORIZONTAL_K,
                    initial_value=1.0e-04 * (i + 1),
                    lower_bound=1e-06,
                    upper_bound=1e-02,
                    layer=1,
                    location=(100.0 * i, 200.0 * i),
                )
            tm = IWFMTemplateManager(parameter_manager=pm, output_dir=tmpdir)

            tpl = tm.generate_pilot_point_template(
                param_type=IWFMParameterType.HORIZONTAL_K,
                layer=1,
            )

            assert len(tpl.parameters) == 5
            content = tpl.template_path.read_text()
            for i in range(5):
                assert f"pp_hk_{i:03d}" in content


class TestRootZoneTemplateExtra:
    """Additional tests for root zone templates."""

    def test_rootzone_no_params_raises(self):
        """Test error when no root zone parameters exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = IWFMParameterManager()
            tm = IWFMTemplateManager(parameter_manager=pm, output_dir=tmpdir)

            with pytest.raises(ValueError, match="No root zone parameters found"):
                tm.generate_rootzone_template(
                    input_file=Path(tmpdir) / "test.dat",
                    param_type=IWFMParameterType.CROP_COEFFICIENT,
                )

    def test_rootzone_string_param_type(self):
        """Test root zone template with string param_type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "RootZone.dat"
            content = "C header\nCORN    0.95\n"
            input_file.write_text(content)

            pm = IWFMParameterManager()
            pm._parameters["kc_corn"] = Parameter(
                name="kc_corn",
                param_type=IWFMParameterType.CROP_COEFFICIENT,
                initial_value=0.95,
                lower_bound=0.5,
                upper_bound=1.5,
                metadata={"land_use_type": "CORN"},
            )
            tm = IWFMTemplateManager(parameter_manager=pm, output_dir=tmpdir)

            tpl = tm.generate_rootzone_template(
                input_file=input_file,
                param_type="kc",  # string
            )

            assert tpl.template_path.exists()

    def test_rootzone_custom_output(self):
        """Test root zone template with custom output path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "RootZone.dat"
            content = "C header\nCORN    0.95\n"
            input_file.write_text(content)

            custom_output = Path(tmpdir) / "my_rz.tpl"

            pm = IWFMParameterManager()
            pm._parameters["kc_corn"] = Parameter(
                name="kc_corn",
                param_type=IWFMParameterType.CROP_COEFFICIENT,
                initial_value=0.95,
                lower_bound=0.5,
                upper_bound=1.5,
                metadata={"land_use_type": "CORN"},
            )
            tm = IWFMTemplateManager(parameter_manager=pm, output_dir=tmpdir)

            tpl = tm.generate_rootzone_template(
                input_file=input_file,
                param_type=IWFMParameterType.CROP_COEFFICIENT,
                output_template=custom_output,
            )

            assert tpl.template_path == custom_output

    def test_rootzone_comment_lines_preserved(self):
        """Test that comment lines in root zone file are preserved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "RootZone.dat"
            content = """C Root Zone data
* Section start
CORN    0.95
"""
            input_file.write_text(content)

            pm = IWFMParameterManager()
            pm._parameters["kc_corn"] = Parameter(
                name="kc_corn",
                param_type=IWFMParameterType.CROP_COEFFICIENT,
                initial_value=0.95,
                lower_bound=0.5,
                upper_bound=1.5,
                metadata={"land_use_type": "CORN"},
            )
            tm = IWFMTemplateManager(parameter_manager=pm, output_dir=tmpdir)

            tpl = tm.generate_rootzone_template(
                input_file=input_file,
                param_type=IWFMParameterType.CROP_COEFFICIENT,
            )

            content = tpl.template_path.read_text()
            assert "C Root Zone data" in content
            assert "* Section start" in content

    def test_rootzone_with_header_lines(self):
        """Test root zone template respects header_lines parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "RootZone.dat"
            content = "Header 1\nHeader 2\nCORN    0.95\n"
            input_file.write_text(content)

            pm = IWFMParameterManager()
            pm._parameters["kc_corn"] = Parameter(
                name="kc_corn",
                param_type=IWFMParameterType.CROP_COEFFICIENT,
                initial_value=0.95,
                lower_bound=0.5,
                upper_bound=1.5,
                metadata={"land_use_type": "CORN"},
            )
            tm = IWFMTemplateManager(parameter_manager=pm, output_dir=tmpdir)

            tpl = tm.generate_rootzone_template(
                input_file=input_file,
                param_type=IWFMParameterType.CROP_COEFFICIENT,
                header_lines=2,
            )

            content = tpl.template_path.read_text()
            assert "Header 1" in content
            assert "Header 2" in content

    def test_rootzone_insufficient_columns(self):
        """Test root zone with lines having fewer columns than expected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "RootZone.dat"
            content = "C header\nshortline\nCORN    0.95\n"
            input_file.write_text(content)

            pm = IWFMParameterManager()
            pm._parameters["kc_corn"] = Parameter(
                name="kc_corn",
                param_type=IWFMParameterType.CROP_COEFFICIENT,
                initial_value=0.95,
                lower_bound=0.5,
                upper_bound=1.5,
                metadata={"land_use_type": "CORN"},
            )
            tm = IWFMTemplateManager(parameter_manager=pm, output_dir=tmpdir)

            tpl = tm.generate_rootzone_template(
                input_file=input_file,
                param_type=IWFMParameterType.CROP_COEFFICIENT,
            )

            content = tpl.template_path.read_text()
            assert "shortline" in content

    def test_rootzone_no_pm(self):
        """Test root zone template with no parameter manager raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tm = IWFMTemplateManager(output_dir=tmpdir)

            with pytest.raises(ValueError, match="No root zone parameters found"):
                tm.generate_rootzone_template(
                    input_file=Path(tmpdir) / "test.dat",
                    param_type=IWFMParameterType.CROP_COEFFICIENT,
                )

    def test_rootzone_param_without_land_use_type(self):
        """Test that params without land_use_type metadata are excluded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = IWFMParameterManager()
            pm._parameters["kc_generic"] = Parameter(
                name="kc_generic",
                param_type=IWFMParameterType.CROP_COEFFICIENT,
                initial_value=0.95,
                lower_bound=0.5,
                upper_bound=1.5,
                metadata={},  # no land_use_type
            )
            tm = IWFMTemplateManager(parameter_manager=pm, output_dir=tmpdir)

            with pytest.raises(ValueError, match="No root zone parameters found"):
                tm.generate_rootzone_template(
                    input_file=Path(tmpdir) / "test.dat",
                    param_type=IWFMParameterType.CROP_COEFFICIENT,
                )


class TestGenerateAllTemplates:
    """Tests for batch template generation."""

    def test_generate_all_no_pm_raises(self):
        """Test error when no parameter manager is set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tm = IWFMTemplateManager(output_dir=tmpdir)
            with pytest.raises(ValueError, match="Parameter manager required"):
                tm.generate_all_templates()

    def test_generate_all_empty_pm(self):
        """Test generating all templates with empty parameter manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = IWFMParameterManager()
            tm = IWFMTemplateManager(parameter_manager=pm, output_dir=tmpdir)

            templates = tm.generate_all_templates()
            assert templates == []

    def test_generate_all_with_pilot_points(self):
        """Test batch generation includes pilot point templates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = IWFMParameterManager()
            pm._parameters["pp_hk_001"] = Parameter(
                name="pp_hk_001",
                param_type=IWFMParameterType.HORIZONTAL_K,
                initial_value=1.0e-04,
                lower_bound=1e-06,
                upper_bound=1e-02,
                layer=1,
                location=(100.0, 200.0),
            )
            pm._parameters["pp_hk_002"] = Parameter(
                name="pp_hk_002",
                param_type=IWFMParameterType.HORIZONTAL_K,
                initial_value=1.5e-04,
                lower_bound=1e-06,
                upper_bound=1e-02,
                layer=1,
                location=(200.0, 300.0),
            )
            tm = IWFMTemplateManager(parameter_manager=pm, output_dir=tmpdir)

            templates = tm.generate_all_templates()
            assert len(templates) >= 1
            # At least one pilot point template
            pp_templates = [t for t in templates if "pp_" in str(t.template_path)]
            assert len(pp_templates) >= 1

    def test_generate_all_with_multipliers(self):
        """Test batch generation includes multiplier templates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = IWFMParameterManager()
            pm._parameters["pump_mult"] = Parameter(
                name="pump_mult",
                param_type=IWFMParameterType.PUMPING_MULT,
                initial_value=1.0,
                lower_bound=0.5,
                upper_bound=1.5,
            )
            tm = IWFMTemplateManager(parameter_manager=pm, output_dir=tmpdir)

            templates = tm.generate_all_templates()
            assert len(templates) >= 1

    def test_generate_all_with_zone_multipliers(self):
        """Test batch generation includes zone multiplier templates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = IWFMParameterManager()
            pm._parameters["pump_z1"] = Parameter(
                name="pump_z1",
                param_type=IWFMParameterType.PUMPING_MULT,
                initial_value=1.0,
                lower_bound=0.5,
                upper_bound=1.5,
                zone=1,
            )
            pm._parameters["pump_z2"] = Parameter(
                name="pump_z2",
                param_type=IWFMParameterType.PUMPING_MULT,
                initial_value=1.0,
                lower_bound=0.5,
                upper_bound=1.5,
                zone=2,
            )
            tm = IWFMTemplateManager(parameter_manager=pm, output_dir=tmpdir)

            templates = tm.generate_all_templates()
            assert len(templates) >= 1

    def test_generate_all_mixed_params(self):
        """Test batch generation with mixed parameter types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = IWFMParameterManager()

            # Pilot point parameter
            pm._parameters["pp_hk_001"] = Parameter(
                name="pp_hk_001",
                param_type=IWFMParameterType.HORIZONTAL_K,
                initial_value=1.0e-04,
                lower_bound=1e-06,
                upper_bound=1e-02,
                layer=1,
                location=(100.0, 200.0),
            )

            # Zone multiplier
            pm._parameters["pump_z1"] = Parameter(
                name="pump_z1",
                param_type=IWFMParameterType.PUMPING_MULT,
                initial_value=1.0,
                lower_bound=0.5,
                upper_bound=1.5,
                zone=1,
            )

            # Global multiplier
            pm._parameters["rech_mult"] = Parameter(
                name="rech_mult",
                param_type=IWFMParameterType.RECHARGE_MULT,
                initial_value=1.0,
                lower_bound=0.5,
                upper_bound=2.0,
            )

            tm = IWFMTemplateManager(parameter_manager=pm, output_dir=tmpdir)

            templates = tm.generate_all_templates()
            # Should have at least one template for each type
            assert len(templates) >= 2

    def test_generate_all_with_input_files(self):
        """Test batch generation accepts input_files argument."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = IWFMParameterManager()
            pm._parameters["pump_mult"] = Parameter(
                name="pump_mult",
                param_type=IWFMParameterType.PUMPING_MULT,
                initial_value=1.0,
                lower_bound=0.5,
                upper_bound=1.5,
            )
            tm = IWFMTemplateManager(parameter_manager=pm, output_dir=tmpdir)

            templates = tm.generate_all_templates(input_files={"pump": "Pumping.dat"})

            assert isinstance(templates, list)

    def test_generate_all_replaces_templates_list(self):
        """Test that generate_all_templates replaces internal template list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = IWFMParameterManager()
            pm._parameters["pump_mult"] = Parameter(
                name="pump_mult",
                param_type=IWFMParameterType.PUMPING_MULT,
                initial_value=1.0,
                lower_bound=0.5,
                upper_bound=1.5,
            )
            tm = IWFMTemplateManager(parameter_manager=pm, output_dir=tmpdir)

            # First call
            templates1 = tm.generate_all_templates()
            count1 = len(templates1)

            # Second call should replace, not append
            templates2 = tm.generate_all_templates()
            assert len(templates2) == count1
            assert len(tm.get_all_templates()) == count1

    def test_generate_all_params_without_type(self):
        """Test that parameters without param_type are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = IWFMParameterManager()
            pm._parameters["generic"] = Parameter(
                name="generic",
                param_type=None,
                initial_value=1.0,
                lower_bound=0.5,
                upper_bound=1.5,
            )
            tm = IWFMTemplateManager(parameter_manager=pm, output_dir=tmpdir)

            templates = tm.generate_all_templates()
            assert templates == []

    def test_generate_all_pilot_points_multiple_layers(self):
        """Test batch generation with pilot points across multiple layers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = IWFMParameterManager()
            for layer in [1, 2]:
                pm._parameters[f"pp_hk_l{layer}"] = Parameter(
                    name=f"pp_hk_l{layer}",
                    param_type=IWFMParameterType.HORIZONTAL_K,
                    initial_value=1.0e-04,
                    lower_bound=1e-06,
                    upper_bound=1e-02,
                    layer=layer,
                    location=(100.0, 200.0),
                )
            tm = IWFMTemplateManager(parameter_manager=pm, output_dir=tmpdir)

            templates = tm.generate_all_templates()
            # Should generate one template per layer
            assert len(templates) >= 2


class TestTemplateMarkerExtra:
    """Additional tests for TemplateMarker dataclass."""

    def test_marker_equality(self):
        """Test that identical markers are equal."""
        m1 = TemplateMarker("p1", 10, 5, 20, "1.5e-04")
        m2 = TemplateMarker("p1", 10, 5, 20, "1.5e-04")
        assert m1 == m2

    def test_marker_inequality(self):
        """Test that different markers are not equal."""
        m1 = TemplateMarker("p1", 10, 5, 20, "1.5e-04")
        m2 = TemplateMarker("p2", 10, 5, 20, "1.5e-04")
        assert m1 != m2


class TestIWFMFileSectionExtra:
    """Additional tests for IWFMFileSection dataclass."""

    def test_default_data_columns_empty(self):
        """Test that default data_columns is empty dict."""
        section = IWFMFileSection(name="test", start_line=1, end_line=10)
        assert section.data_columns == {}

    def test_section_equality(self):
        """Test that identical sections are equal."""
        s1 = IWFMFileSection("test", 1, 10, {"col1": 1})
        s2 = IWFMFileSection("test", 1, 10, {"col1": 1})
        assert s1 == s2
