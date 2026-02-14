"""Tests for PEST++ integration module."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from pyiwfm.runner.pest import (
    Parameter,
    Observation,
    ObservationGroup,
    TemplateFile,
    InstructionFile,
    PESTInterface,
    write_pest_control_file,
)


class TestParameter:
    """Tests for Parameter dataclass."""

    def test_basic_creation(self):
        """Test creating a basic parameter."""
        param = Parameter(
            name="hk_zone1",
            initial_value=10.0,
            lower_bound=0.1,
            upper_bound=1000.0,
        )
        assert param.name == "hk_zone1"
        assert param.initial_value == 10.0
        assert param.lower_bound == 0.1
        assert param.upper_bound == 1000.0
        assert param.group == "default"
        assert param.transform == "none"

    def test_with_transform(self):
        """Test parameter with log transform."""
        param = Parameter(
            name="hk",
            initial_value=10.0,
            lower_bound=0.1,
            upper_bound=1000.0,
            transform="log",
        )
        assert param.transform == "log"

    def test_with_group(self):
        """Test parameter with custom group."""
        param = Parameter(
            name="hk",
            initial_value=10.0,
            lower_bound=0.1,
            upper_bound=1000.0,
            group="hydraulic_conductivity",
        )
        assert param.group == "hydraulic_conductivity"

    def test_invalid_name_length(self):
        """Test that long names raise error."""
        with pytest.raises(ValueError, match="name too long"):
            Parameter(
                name="x" * 201,  # Over 200 char limit
                initial_value=1.0,
                lower_bound=0.0,
                upper_bound=2.0,
            )

    def test_invalid_bounds(self):
        """Test that invalid bounds raise error."""
        with pytest.raises(ValueError, match="Lower bound"):
            Parameter(
                name="test",
                initial_value=1.0,
                lower_bound=10.0,  # Greater than upper
                upper_bound=5.0,
            )

    def test_initial_outside_bounds(self):
        """Test that initial value outside bounds raises error."""
        with pytest.raises(ValueError, match="Initial value"):
            Parameter(
                name="test",
                initial_value=100.0,  # Outside bounds
                lower_bound=0.0,
                upper_bound=10.0,
            )

    def test_to_pest_line(self):
        """Test formatting as PEST control file line."""
        param = Parameter(
            name="hk_zone1",
            initial_value=10.0,
            lower_bound=0.1,
            upper_bound=1000.0,
            group="hk",
        )
        line = param.to_pest_line()

        assert "hk_zone1" in line
        assert "none" in line
        assert "hk" in line


class TestObservation:
    """Tests for Observation dataclass."""

    def test_basic_creation(self):
        """Test creating a basic observation."""
        obs = Observation(
            name="head_well1_t100",
            value=-50.5,
            weight=1.0,
        )
        assert obs.name == "head_well1_t100"
        assert obs.value == -50.5
        assert obs.weight == 1.0
        assert obs.group == "default"

    def test_with_group(self):
        """Test observation with custom group."""
        obs = Observation(
            name="head_well1",
            value=-50.0,
            group="heads",
        )
        assert obs.group == "heads"

    def test_invalid_name_length(self):
        """Test that long names raise error."""
        with pytest.raises(ValueError, match="name too long"):
            Observation(name="x" * 201, value=0.0)

    def test_negative_weight(self):
        """Test that negative weight raises error."""
        with pytest.raises(ValueError, match="Weight must be non-negative"):
            Observation(name="test", value=0.0, weight=-1.0)

    def test_to_pest_line(self):
        """Test formatting as PEST control file line."""
        obs = Observation(
            name="head_well1",
            value=-50.5,
            weight=2.5,
            group="heads",
        )
        line = obs.to_pest_line()

        assert "head_well1" in line
        assert "heads" in line


class TestObservationGroup:
    """Tests for ObservationGroup dataclass."""

    def test_basic_creation(self):
        """Test creating a basic observation group."""
        group = ObservationGroup(name="heads")
        assert group.name == "heads"
        assert group.observations == []

    def test_add_observation(self):
        """Test adding observations to group."""
        group = ObservationGroup(name="heads")

        obs = group.add_observation("head_1", -50.0, 1.0)

        assert len(group.observations) == 1
        assert obs.name == "head_1"
        assert obs.group == "heads"

    def test_multiple_observations(self):
        """Test adding multiple observations."""
        group = ObservationGroup(name="flows")

        group.add_observation("flow_1", 100.0, 0.5)
        group.add_observation("flow_2", 200.0, 0.5)
        group.add_observation("flow_3", 150.0, 0.5)

        assert len(group.observations) == 3


class TestTemplateFile:
    """Tests for TemplateFile dataclass."""

    def test_basic_creation(self):
        """Test creating a basic template file."""
        tpl = TemplateFile(
            template_path=Path("model.tpl"),
            input_path=Path("model.in"),
        )
        assert tpl.template_path == Path("model.tpl")
        assert tpl.input_path == Path("model.in")
        assert tpl.delimiter == "#"
        assert tpl.parameters == []

    def test_path_conversion(self):
        """Test that string paths are converted."""
        tpl = TemplateFile(
            template_path="model.tpl",
            input_path="model.in",
        )
        assert isinstance(tpl.template_path, Path)
        assert isinstance(tpl.input_path, Path)

    def test_create_from_file(self, tmp_path):
        """Test creating template from existing input file."""
        # Create input file with parameter value
        input_file = tmp_path / "model.in"
        input_file.write_text(
            "C Model input\n"
            "HK = 1.5e+01\n"
            "SS = 1.0e-05\n"
        )

        template_file = tmp_path / "model.tpl"

        tpl = TemplateFile.create_from_file(
            input_file,
            template_file,
            parameters={"hk": 1.5e+01},
        )

        assert template_file.exists()
        content = template_file.read_text()
        assert "ptf #" in content
        assert "#" in content  # Parameter marker

    def test_to_pest_line(self):
        """Test formatting as PEST control file line."""
        tpl = TemplateFile(
            template_path=Path("model.tpl"),
            input_path=Path("model.in"),
        )
        line = tpl.to_pest_line()

        assert "model.tpl" in line
        assert "model.in" in line


class TestInstructionFile:
    """Tests for InstructionFile dataclass."""

    def test_basic_creation(self):
        """Test creating a basic instruction file."""
        ins = InstructionFile(
            instruction_path=Path("output.ins"),
            output_path=Path("output.out"),
        )
        assert ins.instruction_path == Path("output.ins")
        assert ins.output_path == Path("output.out")
        assert ins.marker == "@"

    def test_create_for_timeseries(self, tmp_path):
        """Test creating instruction file for time series."""
        output_file = tmp_path / "hydrograph.out"
        instruction_file = tmp_path / "hydrograph.ins"

        ins = InstructionFile.create_for_timeseries(
            output_file,
            instruction_file,
            observations=[
                ("head_t1", 1, 2),  # Line 1, column 2
                ("head_t2", 2, 2),  # Line 2, column 2
            ],
            header_lines=1,
        )

        assert instruction_file.exists()
        assert len(ins.observations) == 2
        assert "head_t1" in ins.observations
        assert "head_t2" in ins.observations

        content = instruction_file.read_text()
        assert "pif @" in content

    def test_create_for_hydrograph(self, tmp_path):
        """Test creating instruction file for hydrograph."""
        output_file = tmp_path / "well_hydrograph.out"
        instruction_file = tmp_path / "well.ins"

        ins = InstructionFile.create_for_hydrograph(
            output_file,
            instruction_file,
            location_name="well1",
            observation_times=[
                (datetime(2000, 10, 1), "oct2000"),
                (datetime(2000, 11, 1), "nov2000"),
            ],
        )

        assert instruction_file.exists()
        assert "well1_oct2000" in ins.observations
        assert "well1_nov2000" in ins.observations

    def test_to_pest_line(self):
        """Test formatting as PEST control file line."""
        ins = InstructionFile(
            instruction_path=Path("output.ins"),
            output_path=Path("output.out"),
        )
        line = ins.to_pest_line()

        assert "output.ins" in line
        assert "output.out" in line


class TestPESTInterface:
    """Tests for PESTInterface class."""

    @pytest.fixture
    def model_dir(self, tmp_path):
        """Create a mock model directory."""
        model = tmp_path / "model"
        model.mkdir()
        (model / "Simulation.in").write_text("C Main file\n")
        return model

    def test_init(self, model_dir):
        """Test interface initialization."""
        pest = PESTInterface(
            model_dir=model_dir,
            case_name="test_cal",
        )

        assert pest.model_dir == model_dir
        assert pest.case_name == "test_cal"
        assert pest.pest_dir.exists()
        assert pest.parameters == []
        assert pest.observations == []

    def test_add_parameter(self, model_dir):
        """Test adding parameters."""
        pest = PESTInterface(model_dir=model_dir, case_name="test")

        param = pest.add_parameter(
            "hk_zone1",
            initial_value=10.0,
            lower_bound=0.1,
            upper_bound=1000.0,
            group="hk",
        )

        assert len(pest.parameters) == 1
        assert param.name == "hk_zone1"
        assert "hk" in pest.parameter_groups

    def test_add_observation(self, model_dir):
        """Test adding observations."""
        pest = PESTInterface(model_dir=model_dir, case_name="test")

        obs = pest.add_observation(
            "head_well1",
            value=-50.0,
            weight=1.0,
            group="heads",
        )

        assert len(pest.observations) == 1
        assert obs.name == "head_well1"
        assert "heads" in pest.observation_groups

    def test_add_observation_group(self, model_dir):
        """Test adding observation groups."""
        pest = PESTInterface(model_dir=model_dir, case_name="test")

        group = pest.add_observation_group(
            "heads",
            observations=[
                ("head_1", -50.0, 1.0),
                ("head_2", -45.0, 1.0),
            ],
        )

        assert group.name == "heads"
        assert len(pest.observations) == 2

    def test_set_model_command(self, model_dir):
        """Test setting model command."""
        pest = PESTInterface(model_dir=model_dir, case_name="test")

        pest.set_model_command("python run_iwfm.py")

        assert pest.model_command == "python run_iwfm.py"

    def test_set_pestpp_option(self, model_dir):
        """Test setting PEST++ options."""
        pest = PESTInterface(model_dir=model_dir, case_name="test")

        pest.set_pestpp_option("svd_pack", "redsvd")
        pest.set_pestpp_option("ies_num_reals", 100)

        assert pest.pestpp_options["svd_pack"] == "redsvd"
        assert pest.pestpp_options["ies_num_reals"] == 100

    def test_write_control_file(self, model_dir):
        """Test writing PEST++ control file."""
        pest = PESTInterface(model_dir=model_dir, case_name="test")

        # Add some parameters and observations
        pest.add_parameter("hk", 10.0, 0.1, 1000.0, group="hk")
        pest.add_observation("head_1", -50.0, 1.0, group="heads")

        # Add template and instruction files
        tpl = TemplateFile(
            template_path=pest.pest_dir / "model.tpl",
            input_path=model_dir / "model.in",
        )
        pest.add_template_file(tpl)

        ins = InstructionFile(
            instruction_path=pest.pest_dir / "output.ins",
            output_path=model_dir / "output.out",
        )
        pest.add_instruction_file(ins)

        # Write control file
        pst_file = pest.write_control_file()

        assert pst_file.exists()
        content = pst_file.read_text()

        assert "pcf" in content
        assert "* control data" in content
        assert "* parameter groups" in content
        assert "* parameter data" in content
        assert "* observation groups" in content
        assert "* observation data" in content
        assert "* model command line" in content
        assert "* model input/output" in content
        assert "hk" in content
        assert "head_1" in content

    def test_write_control_file_with_pestpp_options(self, model_dir):
        """Test control file includes PEST++ options."""
        pest = PESTInterface(model_dir=model_dir, case_name="test")

        pest.add_parameter("p1", 1.0, 0.1, 10.0)
        pest.add_observation("o1", 5.0)
        pest.set_pestpp_option("svd_pack", "redsvd")

        # Add minimal template and instruction
        tpl = TemplateFile(pest.pest_dir / "t.tpl", model_dir / "t.in")
        ins = InstructionFile(pest.pest_dir / "i.ins", model_dir / "o.out")
        pest.add_template_file(tpl)
        pest.add_instruction_file(ins)

        pst_file = pest.write_control_file()

        content = pst_file.read_text()
        assert "++" in content
        assert "svd_pack" in content

    def test_write_model_runner(self, model_dir):
        """Test writing model runner script."""
        pest = PESTInterface(model_dir=model_dir, case_name="test")

        script_path = pest.write_model_runner()

        assert script_path.exists()
        content = script_path.read_text()

        assert "#!/usr/bin/env python" in content
        assert "IWFMRunner" in content
        assert "run_simulation" in content

    def test_repr(self, model_dir):
        """Test string representation."""
        pest = PESTInterface(model_dir=model_dir, case_name="test")
        pest.add_parameter("p1", 1.0, 0.1, 10.0)
        pest.add_observation("o1", 5.0)

        repr_str = repr(pest)
        assert "PESTInterface" in repr_str
        assert "test" in repr_str
        assert "n_parameters=1" in repr_str
        assert "n_observations=1" in repr_str


class TestWritePestControlFile:
    """Tests for write_pest_control_file convenience function."""

    def test_write_control_file(self, tmp_path):
        """Test writing control file with convenience function."""
        pst_path = tmp_path / "test.pst"

        parameters = [
            Parameter("p1", 1.0, 0.1, 10.0, group="pgroup"),
        ]
        observations = [
            Observation("o1", 5.0, 1.0, group="ogroup"),
        ]
        template_files = [
            TemplateFile(tmp_path / "t.tpl", tmp_path / "t.in"),
        ]
        instruction_files = [
            InstructionFile(tmp_path / "i.ins", tmp_path / "o.out"),
        ]

        result = write_pest_control_file(
            pst_path,
            parameters,
            observations,
            template_files,
            instruction_files,
            svd_pack="redsvd",
        )

        assert result == pst_path
        assert pst_path.exists()

        content = pst_path.read_text()
        assert "p1" in content
        assert "o1" in content
        assert "svd_pack" in content
