"""Unit tests for PEST++ instruction file generation."""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from pyiwfm.runner.pest_instructions import (
    IWFM_OUTPUT_FORMATS,
    IWFMInstructionManager,
    OutputFileFormat,
)
from pyiwfm.runner.pest_obs_manager import IWFMObservationManager
from pyiwfm.runner.pest_observations import (
    IWFMObservation,
    IWFMObservationType,
)


class TestOutputFileFormat:
    """Tests for OutputFileFormat dataclass."""

    def test_basic_creation(self):
        """Test basic format creation."""
        fmt = OutputFileFormat(
            name="custom",
            header_lines=5,
            time_column=1,
            value_columns={"value": 2},
        )
        assert fmt.name == "custom"
        assert fmt.header_lines == 5
        assert fmt.time_column == 1

    def test_default_values(self):
        """Test default values."""
        fmt = OutputFileFormat(name="test")
        assert fmt.header_lines == 1
        assert fmt.time_format == "%m/%d/%Y"
        assert fmt.delimiter == "whitespace"


class TestIWFMOutputFormats:
    """Tests for predefined IWFM output formats."""

    def test_head_hydrograph_format(self):
        """Test head hydrograph format."""
        fmt = IWFM_OUTPUT_FORMATS["head_hydrograph"]
        assert fmt.name == "head_hydrograph"
        assert fmt.header_lines == 3
        assert "head" in fmt.value_columns

    def test_stream_hydrograph_format(self):
        """Test stream hydrograph format."""
        fmt = IWFM_OUTPUT_FORMATS["stream_hydrograph"]
        assert "flow" in fmt.value_columns
        assert "stage" in fmt.value_columns

    def test_gw_budget_format(self):
        """Test GW budget format."""
        fmt = IWFM_OUTPUT_FORMATS["gw_budget"]
        assert fmt.header_lines == 4


class TestIWFMInstructionManagerInit:
    """Tests for IWFMInstructionManager initialization."""

    def test_init_basic(self):
        """Test basic initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            im = IWFMInstructionManager(output_dir=tmpdir)
            assert im.output_dir == Path(tmpdir)
            assert im.marker == "@"
            assert len(im._instructions) == 0

    def test_init_with_custom_marker(self):
        """Test initialization with custom marker."""
        with tempfile.TemporaryDirectory() as tmpdir:
            im = IWFMInstructionManager(output_dir=tmpdir, marker="$")
            assert im.marker == "$"

    def test_init_creates_output_dir(self):
        """Test that output directory is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "instructions" / "nested"
            IWFMInstructionManager(output_dir=output_dir)
            assert output_dir.exists()


class TestHeadInstructions:
    """Tests for head observation instructions."""

    @pytest.fixture
    def sample_head_observations(self):
        """Create sample head observations."""
        return [
            IWFMObservation(
                name="h_w1_20200101",
                value=100.0,
                group="head",
                obs_type=IWFMObservationType.HEAD,
                datetime=datetime(2020, 1, 1, 0, 0),
                metadata={"well_id": "W1"},
            ),
            IWFMObservation(
                name="h_w1_20200201",
                value=101.0,
                group="head",
                obs_type=IWFMObservationType.HEAD,
                datetime=datetime(2020, 2, 1, 0, 0),
                metadata={"well_id": "W1"},
            ),
            IWFMObservation(
                name="h_w2_20200101",
                value=150.0,
                group="head",
                obs_type=IWFMObservationType.HEAD,
                datetime=datetime(2020, 1, 1, 0, 0),
                metadata={"well_id": "W2"},
            ),
        ]

    @pytest.fixture
    def sample_head_output_file(self):
        """Create a sample head output file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "GW_Heads.out"
            content = """Well Hydrograph Output
Time             Head
Date_Time        (ft)
01/01/2020_00:00  100.5
02/01/2020_00:00  101.2
03/01/2020_00:00  99.8
"""
            filepath.write_text(content)
            yield filepath

    def test_generate_head_instructions(self, sample_head_observations, sample_head_output_file):
        """Test generating head instructions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            om = IWFMObservationManager()
            for obs in sample_head_observations:
                om._observations[obs.name] = obs
                grp = om.get_observation_group("head")
                grp.observations.append(obs)

            im = IWFMInstructionManager(observation_manager=om, output_dir=tmpdir)

            ins = im.generate_head_instructions(
                output_file=sample_head_output_file,
                observations=sample_head_observations,
                header_lines=3,
                value_column=2,
                time_format="%m/%d/%Y_%H:%M",
            )

            assert ins.instruction_path.exists()
            assert len(ins.observations) == 3

            content = ins.instruction_path.read_text()
            assert "pif @" in content
            assert "h_w1_20200101" in content

    def test_head_instructions_no_observations_raises(self):
        """Test that missing observations raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            im = IWFMInstructionManager(output_dir=tmpdir)

            with pytest.raises(ValueError, match="No.*observations"):
                im.generate_head_instructions(
                    output_file=Path(tmpdir) / "test.out",
                    observations=[],
                )


class TestStreamflowInstructions:
    """Tests for streamflow observation instructions."""

    @pytest.fixture
    def sample_flow_observations(self):
        """Create sample flow observations."""
        return [
            IWFMObservation(
                name="f_g1_20200101",
                value=500.0,
                group="flow",
                obs_type=IWFMObservationType.STREAM_FLOW,
                datetime=datetime(2020, 1, 1, 0, 0),
                metadata={"gage_id": "G1"},
            ),
            IWFMObservation(
                name="f_g1_20200201",
                value=550.0,
                group="flow",
                obs_type=IWFMObservationType.STREAM_FLOW,
                datetime=datetime(2020, 2, 1, 0, 0),
                metadata={"gage_id": "G1"},
            ),
        ]

    @pytest.fixture
    def sample_flow_output_file(self):
        """Create a sample flow output file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "StreamFlow.out"
            content = """Stream Hydrograph Output
Time             Flow      Stage
Date_Time        (cfs)     (ft)
01/01/2020_00:00  505.3     10.5
02/01/2020_00:00  548.2     11.2
"""
            filepath.write_text(content)
            yield filepath

    def test_generate_flow_instructions(self, sample_flow_observations, sample_flow_output_file):
        """Test generating flow instructions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            im = IWFMInstructionManager(output_dir=tmpdir)

            ins = im.generate_flow_instructions(
                output_file=sample_flow_output_file,
                observations=sample_flow_observations,
                variable="flow",
                header_lines=3,
                value_column=2,
                time_format="%m/%d/%Y_%H:%M",
            )

            assert ins.instruction_path.exists()
            assert len(ins.observations) == 2

            content = ins.instruction_path.read_text()
            assert "pif @" in content

    def test_generate_stage_instructions(self, sample_flow_output_file):
        """Test generating stage instructions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            stage_obs = [
                IWFMObservation(
                    name="s_g1_20200101",
                    value=10.5,
                    group="stage",
                    obs_type=IWFMObservationType.STREAM_STAGE,
                    datetime=datetime(2020, 1, 1, 0, 0),
                ),
            ]

            im = IWFMInstructionManager(output_dir=tmpdir)

            ins = im.generate_flow_instructions(
                output_file=sample_flow_output_file,
                observations=stage_obs,
                variable="stage",
                header_lines=3,
                value_column=3,  # Stage is column 3
                time_format="%m/%d/%Y_%H:%M",
            )

            assert ins.instruction_path.exists()


class TestGainLossInstructions:
    """Tests for stream gain/loss instructions."""

    @pytest.fixture
    def sample_gain_loss_observations(self):
        """Create sample gain/loss observations."""
        return [
            IWFMObservation(
                name="sgl_r1_20200101",
                value=10.0,
                group="gain_loss",
                obs_type=IWFMObservationType.STREAM_GAIN_LOSS,
                datetime=datetime(2020, 1, 1),
                metadata={"reach_id": 1},
            ),
            IWFMObservation(
                name="sgl_r2_20200101",
                value=-5.0,
                group="gain_loss",
                obs_type=IWFMObservationType.STREAM_GAIN_LOSS,
                datetime=datetime(2020, 1, 1),
                metadata={"reach_id": 2},
            ),
        ]

    @pytest.fixture
    def sample_gain_loss_file(self):
        """Create a sample gain/loss file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "GainLoss.out"
            content = """Stream Gain/Loss Output
Time        Reach  GainLoss
Date        ID     (cfs)
01/01/2020  1      10.5
01/01/2020  2      -4.8
02/01/2020  1      12.3
"""
            filepath.write_text(content)
            yield filepath

    def test_generate_gain_loss_instructions(
        self, sample_gain_loss_observations, sample_gain_loss_file
    ):
        """Test generating gain/loss instructions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            im = IWFMInstructionManager(output_dir=tmpdir)

            ins = im.generate_gain_loss_instructions(
                output_file=sample_gain_loss_file,
                observations=sample_gain_loss_observations,
                header_lines=3,
                time_format="%m/%d/%Y",
            )

            assert ins.instruction_path.exists()
            assert len(ins.observations) == 2


class TestBudgetInstructions:
    """Tests for budget observation instructions."""

    @pytest.fixture
    def sample_budget_observations(self):
        """Create sample budget observations."""
        return [
            IWFMObservation(
                name="gwbud_rech_20200101",
                value=1000.0,
                group="gwbud",
                obs_type=IWFMObservationType.GW_BUDGET,
                datetime=datetime(2020, 1, 1),
                metadata={"component": "RECHARGE", "budget_type": "gw"},
            ),
            IWFMObservation(
                name="gwbud_pump_20200101",
                value=-500.0,
                group="gwbud",
                obs_type=IWFMObservationType.GW_BUDGET,
                datetime=datetime(2020, 1, 1),
                metadata={"component": "PUMPING", "budget_type": "gw"},
            ),
        ]

    @pytest.fixture
    def sample_budget_file(self):
        """Create a sample budget file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "GW_Budget.out"
            content = """Groundwater Budget Output
Region: ALL
Period     RECHARGE   PUMPING
Date       (AF)       (AF)
01/01/2020  1050.5    -495.2
02/01/2020  980.3     -510.1
"""
            filepath.write_text(content)
            yield filepath

    def test_generate_budget_instructions_gw(self, sample_budget_observations, sample_budget_file):
        """Test generating GW budget instructions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            im = IWFMInstructionManager(output_dir=tmpdir)

            ins = im.generate_budget_instructions(
                budget_file=sample_budget_file,
                budget_type="gw",
                observations=sample_budget_observations,
                header_lines=4,
                time_format="%m/%d/%Y",
            )

            assert ins.instruction_path.exists()
            assert len(ins.observations) == 2

    def test_generate_budget_instructions_invalid_type(self):
        """Test invalid budget type raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            im = IWFMInstructionManager(output_dir=tmpdir)

            with pytest.raises(ValueError, match="Invalid budget type"):
                im.generate_budget_instructions(
                    budget_file=Path(tmpdir) / "test.out",
                    budget_type="invalid",
                    observations=[],
                )


class TestLakeInstructions:
    """Tests for lake observation instructions."""

    @pytest.fixture
    def sample_lake_observations(self):
        """Create sample lake observations."""
        return [
            IWFMObservation(
                name="lak1_lev_20200101",
                value=100.0,
                group="lake_level",
                obs_type=IWFMObservationType.LAKE_LEVEL,
                datetime=datetime(2020, 1, 1, 0, 0),
                metadata={"lake_id": 1},
            ),
        ]

    @pytest.fixture
    def sample_lake_file(self):
        """Create a sample lake output file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "Lake.out"
            content = """Lake Output
Time             Level
Date_Time        (ft)
01/01/2020_00:00  100.5
02/01/2020_00:00  99.8
"""
            filepath.write_text(content)
            yield filepath

    def test_generate_lake_instructions(self, sample_lake_observations, sample_lake_file):
        """Test generating lake instructions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            im = IWFMInstructionManager(output_dir=tmpdir)

            ins = im.generate_lake_instructions(
                output_file=sample_lake_file,
                observations=sample_lake_observations,
                variable="level",
                header_lines=3,
                value_column=2,
                time_format="%m/%d/%Y_%H:%M",
            )

            assert ins.instruction_path.exists()
            assert len(ins.observations) == 1


class TestSubsidenceInstructions:
    """Tests for subsidence observation instructions."""

    @pytest.fixture
    def sample_subsidence_observations(self):
        """Create sample subsidence observations."""
        return [
            IWFMObservation(
                name="sub_n1_20200101",
                value=0.05,
                group="subsidence",
                obs_type=IWFMObservationType.SUBSIDENCE,
                datetime=datetime(2020, 1, 1, 0, 0),
            ),
        ]

    @pytest.fixture
    def sample_subsidence_file(self):
        """Create a sample subsidence output file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "Subsidence.out"
            content = """Subsidence Output
Time             Subsidence
Date_Time        (ft)
01/01/2020_00:00  0.048
02/01/2020_00:00  0.052
"""
            filepath.write_text(content)
            yield filepath

    def test_generate_subsidence_instructions(
        self, sample_subsidence_observations, sample_subsidence_file
    ):
        """Test generating subsidence instructions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            im = IWFMInstructionManager(output_dir=tmpdir)

            ins = im.generate_subsidence_instructions(
                output_file=sample_subsidence_file,
                observations=sample_subsidence_observations,
                header_lines=3,
                value_column=2,
                time_format="%m/%d/%Y_%H:%M",
            )

            assert ins.instruction_path.exists()
            assert len(ins.observations) == 1


class TestCustomInstructions:
    """Tests for custom instruction generation."""

    def test_generate_custom_instructions(self):
        """Test generating custom instructions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "custom.out"
            output_file.write_text("Some custom output content")

            im = IWFMInstructionManager(output_dir=tmpdir)

            observations = [
                ("obs1", "MARKER1", 2),
                ("obs2", "MARKER2", 3),
            ]

            ins = im.generate_custom_instructions(
                output_file=output_file,
                observations=observations,
                header_lines=1,
            )

            assert ins.instruction_path.exists()
            assert len(ins.observations) == 2

            content = ins.instruction_path.read_text()
            assert "@MARKER1@" in content
            assert "obs1" in content

    def test_generate_fixed_format_instructions(self):
        """Test generating fixed-format instructions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "fixed.out"
            output_file.write_text("Header line\n12345678901234567890")

            im = IWFMInstructionManager(output_dir=tmpdir)

            observations = [
                ("obs1", 1, 1, 5),  # Line 1, columns 1-5
                ("obs2", 1, 10, 15),  # Line 1, columns 10-15
            ]

            ins = im.generate_fixed_format_instructions(
                output_file=output_file,
                observations=observations,
                header_lines=1,
            )

            assert ins.instruction_path.exists()
            assert len(ins.observations) == 2

            content = ins.instruction_path.read_text()
            assert "[1:5]" in content
            assert "[10:15]" in content


class TestInstructionManagerUtilities:
    """Tests for instruction manager utility methods."""

    def test_get_all_instructions(self):
        """Test getting all created instructions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test.out"
            output_file.write_text("test content")

            im = IWFMInstructionManager(output_dir=tmpdir)
            im.generate_custom_instructions(
                output_file=output_file,
                observations=[("obs1", "MARKER", 2)],
            )

            instructions = im.get_all_instructions()
            assert len(instructions) == 1

    def test_clear_instructions(self):
        """Test clearing instructions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test.out"
            output_file.write_text("test content")

            im = IWFMInstructionManager(output_dir=tmpdir)
            im.generate_custom_instructions(
                output_file=output_file,
                observations=[("obs1", "MARKER", 2)],
            )

            assert len(im.get_all_instructions()) == 1
            im.clear_instructions()
            assert len(im.get_all_instructions()) == 0

    def test_build_read_instruction(self):
        """Test building read instruction."""
        with tempfile.TemporaryDirectory() as tmpdir:
            im = IWFMInstructionManager(output_dir=tmpdir)
            instruction = im._build_read_instruction(3, "test_obs")
            assert instruction == "w w !test_obs!"

    def test_repr(self):
        """Test string representation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            im = IWFMInstructionManager(output_dir=tmpdir)
            r = repr(im)
            assert "IWFMInstructionManager" in r
            assert "n_instructions=0" in r


# =========================================================================
# Additional Tests for 95%+ Coverage
# =========================================================================


class TestOutputFileFormatExtended:
    """Extended tests for OutputFileFormat."""

    def test_custom_time_format(self):
        """Test custom time format."""
        fmt = OutputFileFormat(name="custom", time_format="%Y-%m-%d %H:%M:%S")
        assert fmt.time_format == "%Y-%m-%d %H:%M:%S"

    def test_custom_delimiter(self):
        """Test custom delimiter."""
        fmt = OutputFileFormat(name="custom", delimiter=",")
        assert fmt.delimiter == ","

    def test_value_columns_default_empty(self):
        """Test default value_columns is empty dict."""
        fmt = OutputFileFormat(name="test")
        assert fmt.value_columns == {}


class TestIWFMOutputFormatsExtended:
    """Extended tests for predefined formats."""

    def test_stream_budget_format(self):
        """Test stream budget format."""
        fmt = IWFM_OUTPUT_FORMATS["stream_budget"]
        assert fmt.header_lines == 4
        assert fmt.name == "stream_budget"

    def test_lake_budget_format(self):
        """Test lake budget format."""
        fmt = IWFM_OUTPUT_FORMATS["lake_budget"]
        assert fmt.header_lines == 4

    def test_subsidence_format(self):
        """Test subsidence format."""
        fmt = IWFM_OUTPUT_FORMATS["subsidence"]
        assert fmt.header_lines == 3
        assert "subsidence" in fmt.value_columns

    def test_all_formats_have_names(self):
        """Test all predefined formats have valid names."""
        for key, fmt in IWFM_OUTPUT_FORMATS.items():
            assert fmt.name == key


class TestIWFMInstructionManagerInitExtended:
    """Extended initialization tests."""

    def test_init_default_output_dir(self):
        """Test initialization with no output_dir uses current directory."""
        im = IWFMInstructionManager()
        assert im.output_dir == Path(".")

    def test_init_with_observation_manager(self):
        """Test initialization with observation manager."""
        om = IWFMObservationManager()
        with tempfile.TemporaryDirectory() as tmpdir:
            im = IWFMInstructionManager(observation_manager=om, output_dir=tmpdir)
            assert im.om is om

    def test_init_with_model(self):
        """Test initialization with model."""

        class MockModel:
            pass

        model = MockModel()
        with tempfile.TemporaryDirectory() as tmpdir:
            im = IWFMInstructionManager(model=model, output_dir=tmpdir)
            assert im.model is model


class TestHeadInstructionsExtended:
    """Extended head instruction tests."""

    def test_head_instructions_from_observation_manager(self):
        """Test generating head instructions from observation manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            om = IWFMObservationManager()
            obs = IWFMObservation(
                name="h1_20200101",
                value=100.0,
                group="head",
                obs_type=IWFMObservationType.HEAD,
                datetime=datetime(2020, 1, 1, 0, 0),
                metadata={"well_id": "W1"},
            )
            om._observations[obs.name] = obs
            grp = om.get_observation_group("head")
            grp.observations.append(obs)

            im = IWFMInstructionManager(observation_manager=om, output_dir=tmpdir)

            output_file = Path(tmpdir) / "GW_Heads.out"
            output_file.write_text("header\ndata\n")

            ins = im.generate_head_instructions(output_file=output_file)
            assert len(ins.observations) == 1
            assert "h1_20200101" in ins.observations

    def test_head_instructions_custom_instruction_file(self):
        """Test specifying custom instruction file path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            obs = [
                IWFMObservation(
                    name="h1",
                    value=100.0,
                    obs_type=IWFMObservationType.HEAD,
                    datetime=datetime(2020, 1, 1, 0, 0),
                ),
            ]
            im = IWFMInstructionManager(output_dir=tmpdir)
            custom_path = Path(tmpdir) / "my_custom.ins"

            ins = im.generate_head_instructions(
                output_file=Path(tmpdir) / "test.out",
                observations=obs,
                instruction_file=custom_path,
            )
            assert ins.instruction_path == custom_path
            assert custom_path.exists()

    def test_head_instructions_no_header(self):
        """Test head instructions with zero header lines."""
        with tempfile.TemporaryDirectory() as tmpdir:
            obs = [
                IWFMObservation(
                    name="h1",
                    value=100.0,
                    obs_type=IWFMObservationType.HEAD,
                    datetime=datetime(2020, 1, 1, 0, 0),
                ),
            ]
            im = IWFMInstructionManager(output_dir=tmpdir)
            ins = im.generate_head_instructions(
                output_file=Path(tmpdir) / "test.out",
                observations=obs,
                header_lines=0,
            )
            content = ins.instruction_path.read_text()
            # Should not have "l3" or any "lN" skip line for header
            lines = content.split("\n")
            assert lines[0] == "pif @"
            # Second line should be a search marker, not lN
            assert not lines[1].startswith("l")

    def test_head_instructions_multiple_obs_same_time(self):
        """Test multiple observations at the same time produce l1 line advances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            obs = [
                IWFMObservation(
                    name="h_w1",
                    value=100.0,
                    obs_type=IWFMObservationType.HEAD,
                    datetime=datetime(2020, 1, 1, 0, 0),
                    metadata={"well_id": "W1"},
                ),
                IWFMObservation(
                    name="h_w2",
                    value=150.0,
                    obs_type=IWFMObservationType.HEAD,
                    datetime=datetime(2020, 1, 1, 0, 0),
                    metadata={"well_id": "W2"},
                ),
            ]
            im = IWFMInstructionManager(output_dir=tmpdir)
            ins = im.generate_head_instructions(
                output_file=Path(tmpdir) / "test.out",
                observations=obs,
            )
            content = ins.instruction_path.read_text()
            assert "l1" in content

    def test_head_instructions_obs_without_datetime_skipped(self):
        """Test observations without datetime are skipped in instruction generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            obs = [
                IWFMObservation(
                    name="h1",
                    value=100.0,
                    obs_type=IWFMObservationType.HEAD,
                    datetime=None,  # No datetime
                ),
                IWFMObservation(
                    name="h2",
                    value=101.0,
                    obs_type=IWFMObservationType.HEAD,
                    datetime=datetime(2020, 1, 1, 0, 0),
                ),
            ]
            im = IWFMInstructionManager(output_dir=tmpdir)
            ins = im.generate_head_instructions(
                output_file=Path(tmpdir) / "test.out",
                observations=obs,
            )
            # Only h2 should be in instructions
            assert "h2" in ins.observations
            assert "h1" not in ins.observations

    def test_head_instructions_default_ins_path(self):
        """Test default instruction file path is derived from output file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            obs = [
                IWFMObservation(
                    name="h1",
                    value=100.0,
                    obs_type=IWFMObservationType.HEAD,
                    datetime=datetime(2020, 1, 1, 0, 0),
                ),
            ]
            im = IWFMInstructionManager(output_dir=tmpdir)
            ins = im.generate_head_instructions(
                output_file=Path(tmpdir) / "GW_Heads.out",
                observations=obs,
            )
            assert ins.instruction_path.name == "GW_Heads.ins"


class TestHeadInstructionsByWell:
    """Tests for per-well head instruction generation."""

    def test_generate_head_instructions_by_well(self):
        """Test generating separate instruction files per well."""
        with tempfile.TemporaryDirectory() as tmpdir:
            obs = [
                IWFMObservation(
                    name="h_w1_t1",
                    value=100.0,
                    obs_type=IWFMObservationType.HEAD,
                    datetime=datetime(2020, 1, 1, 0, 0),
                    metadata={"well_id": "W1"},
                ),
                IWFMObservation(
                    name="h_w2_t1",
                    value=150.0,
                    obs_type=IWFMObservationType.HEAD,
                    datetime=datetime(2020, 1, 1, 0, 0),
                    metadata={"well_id": "W2"},
                ),
            ]

            output_files = {
                "W1": Path(tmpdir) / "W1_heads.out",
                "W2": Path(tmpdir) / "W2_heads.out",
            }
            for f in output_files.values():
                f.write_text("header\ndata\n")

            im = IWFMInstructionManager(output_dir=tmpdir)
            instructions = im.generate_head_instructions_by_well(
                output_files=output_files,
                observations=obs,
            )
            assert len(instructions) == 2

    def test_generate_head_instructions_by_well_from_om(self):
        """Test per-well instructions using observation manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            om = IWFMObservationManager()
            obs1 = IWFMObservation(
                name="h_w1_t1",
                value=100.0,
                obs_type=IWFMObservationType.HEAD,
                datetime=datetime(2020, 1, 1, 0, 0),
                metadata={"well_id": "W1"},
            )
            om._observations[obs1.name] = obs1
            grp = om.get_observation_group("head")
            grp.observations.append(obs1)

            output_files = {"W1": Path(tmpdir) / "W1_heads.out"}
            output_files["W1"].write_text("header\n")

            im = IWFMInstructionManager(observation_manager=om, output_dir=tmpdir)
            instructions = im.generate_head_instructions_by_well(
                output_files=output_files,
            )
            assert len(instructions) == 1

    def test_generate_head_instructions_by_well_missing_output_file(self):
        """Test per-well instructions skip wells without output files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            obs = [
                IWFMObservation(
                    name="h_w1_t1",
                    value=100.0,
                    obs_type=IWFMObservationType.HEAD,
                    datetime=datetime(2020, 1, 1, 0, 0),
                    metadata={"well_id": "W1"},
                ),
                IWFMObservation(
                    name="h_w3_t1",
                    value=200.0,
                    obs_type=IWFMObservationType.HEAD,
                    datetime=datetime(2020, 1, 1, 0, 0),
                    metadata={"well_id": "W3"},  # Not in output_files
                ),
            ]

            output_files = {"W1": Path(tmpdir) / "W1.out"}
            output_files["W1"].write_text("header\n")

            im = IWFMInstructionManager(output_dir=tmpdir)
            instructions = im.generate_head_instructions_by_well(
                output_files=output_files,
                observations=obs,
            )
            # Only W1 should produce instruction file
            assert len(instructions) == 1

    def test_generate_head_instructions_by_well_no_metadata(self):
        """Test per-well instructions skip obs without well_id metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            obs = [
                IWFMObservation(
                    name="h_t1",
                    value=100.0,
                    obs_type=IWFMObservationType.HEAD,
                    datetime=datetime(2020, 1, 1, 0, 0),
                    metadata={},  # No well_id
                ),
            ]
            output_files = {"W1": Path(tmpdir) / "W1.out"}
            output_files["W1"].write_text("header\n")

            im = IWFMInstructionManager(output_dir=tmpdir)
            instructions = im.generate_head_instructions_by_well(
                output_files=output_files,
                observations=obs,
            )
            assert len(instructions) == 0


class TestFlowInstructionsExtended:
    """Extended flow instruction tests."""

    def test_flow_instructions_from_om(self):
        """Test flow instructions using observation manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            om = IWFMObservationManager()
            obs = IWFMObservation(
                name="f1",
                value=500.0,
                obs_type=IWFMObservationType.STREAM_FLOW,
                datetime=datetime(2020, 1, 1, 0, 0),
                group="flow",
            )
            om._observations[obs.name] = obs
            grp = om.get_observation_group("flow")
            grp.observations.append(obs)

            im = IWFMInstructionManager(observation_manager=om, output_dir=tmpdir)
            output_file = Path(tmpdir) / "StreamFlow.out"
            output_file.write_text("header\n")

            ins = im.generate_flow_instructions(output_file=output_file, variable="flow")
            assert len(ins.observations) == 1

    def test_stage_instructions_from_om(self):
        """Test stage instructions using observation manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            om = IWFMObservationManager()
            obs = IWFMObservation(
                name="s1",
                value=10.0,
                obs_type=IWFMObservationType.STREAM_STAGE,
                datetime=datetime(2020, 1, 1, 0, 0),
                group="stage",
            )
            om._observations[obs.name] = obs
            grp = om.get_observation_group("stage")
            grp.observations.append(obs)

            im = IWFMInstructionManager(observation_manager=om, output_dir=tmpdir)
            output_file = Path(tmpdir) / "Stage.out"
            output_file.write_text("header\n")

            ins = im.generate_flow_instructions(output_file=output_file, variable="stage")
            assert len(ins.observations) == 1

    def test_flow_instructions_no_observations_raises(self):
        """Test flow instructions raises when no observations provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            im = IWFMInstructionManager(output_dir=tmpdir)
            with pytest.raises(ValueError, match="No.*observations"):
                im.generate_flow_instructions(
                    output_file=Path(tmpdir) / "test.out",
                    observations=[],
                )

    def test_flow_instructions_custom_file(self):
        """Test custom instruction file path for flow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            obs = [
                IWFMObservation(
                    name="f1",
                    value=100.0,
                    obs_type=IWFMObservationType.STREAM_FLOW,
                    datetime=datetime(2020, 1, 1, 0, 0),
                ),
            ]
            im = IWFMInstructionManager(output_dir=tmpdir)
            custom_path = Path(tmpdir) / "my_flow.ins"
            ins = im.generate_flow_instructions(
                output_file=Path(tmpdir) / "flow.out",
                observations=obs,
                instruction_file=custom_path,
            )
            assert ins.instruction_path == custom_path

    def test_flow_instructions_default_path(self):
        """Test default instruction file path for flow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            obs = [
                IWFMObservation(
                    name="f1",
                    value=100.0,
                    obs_type=IWFMObservationType.STREAM_FLOW,
                    datetime=datetime(2020, 1, 1, 0, 0),
                ),
            ]
            im = IWFMInstructionManager(output_dir=tmpdir)
            ins = im.generate_flow_instructions(
                output_file=Path(tmpdir) / "StreamFlow.out",
                observations=obs,
                variable="flow",
            )
            assert ins.instruction_path.name == "StreamFlow_flow.ins"

    def test_flow_instructions_no_header(self):
        """Test flow instructions with no header lines."""
        with tempfile.TemporaryDirectory() as tmpdir:
            obs = [
                IWFMObservation(
                    name="f1",
                    value=100.0,
                    obs_type=IWFMObservationType.STREAM_FLOW,
                    datetime=datetime(2020, 1, 1, 0, 0),
                ),
            ]
            im = IWFMInstructionManager(output_dir=tmpdir)
            ins = im.generate_flow_instructions(
                output_file=Path(tmpdir) / "flow.out",
                observations=obs,
                header_lines=0,
            )
            content = ins.instruction_path.read_text()
            lines = content.split("\n")
            assert lines[0] == "pif @"
            # No "lN" header skip
            assert not lines[1].startswith("l")


class TestGainLossInstructionsExtended:
    """Extended gain/loss instruction tests."""

    def test_gain_loss_no_observations_raises(self):
        """Test gain/loss raises when no observations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            im = IWFMInstructionManager(output_dir=tmpdir)
            with pytest.raises(ValueError, match="No.*observations"):
                im.generate_gain_loss_instructions(
                    output_file=Path(tmpdir) / "test.out",
                    observations=[],
                )

    def test_gain_loss_from_om(self):
        """Test gain/loss instructions from observation manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            om = IWFMObservationManager()
            obs = IWFMObservation(
                name="sgl1",
                value=10.0,
                obs_type=IWFMObservationType.STREAM_GAIN_LOSS,
                datetime=datetime(2020, 1, 1),
                group="gain_loss",
                metadata={"reach_id": 1},
            )
            om._observations[obs.name] = obs
            grp = om.get_observation_group("gain_loss")
            grp.observations.append(obs)

            im = IWFMInstructionManager(observation_manager=om, output_dir=tmpdir)
            ins = im.generate_gain_loss_instructions(
                output_file=Path(tmpdir) / "gl.out",
            )
            assert len(ins.observations) == 1

    def test_gain_loss_obs_without_datetime_skipped(self):
        """Test gain/loss obs without datetime are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            obs = [
                IWFMObservation(
                    name="sgl1",
                    value=10.0,
                    obs_type=IWFMObservationType.STREAM_GAIN_LOSS,
                    datetime=None,
                    metadata={"reach_id": 1},
                ),
                IWFMObservation(
                    name="sgl2",
                    value=5.0,
                    obs_type=IWFMObservationType.STREAM_GAIN_LOSS,
                    datetime=datetime(2020, 1, 1),
                    metadata={"reach_id": 2},
                ),
            ]
            im = IWFMInstructionManager(output_dir=tmpdir)
            ins = im.generate_gain_loss_instructions(
                output_file=Path(tmpdir) / "gl.out",
                observations=obs,
            )
            assert "sgl2" in ins.observations
            assert "sgl1" not in ins.observations

    def test_gain_loss_custom_instruction_file(self):
        """Test custom instruction file path for gain/loss."""
        with tempfile.TemporaryDirectory() as tmpdir:
            obs = [
                IWFMObservation(
                    name="sgl1",
                    value=10.0,
                    obs_type=IWFMObservationType.STREAM_GAIN_LOSS,
                    datetime=datetime(2020, 1, 1),
                    metadata={"reach_id": 1},
                ),
            ]
            im = IWFMInstructionManager(output_dir=tmpdir)
            custom_path = Path(tmpdir) / "custom_gl.ins"
            ins = im.generate_gain_loss_instructions(
                output_file=Path(tmpdir) / "gl.out",
                observations=obs,
                instruction_file=custom_path,
            )
            assert ins.instruction_path == custom_path

    def test_gain_loss_obs_without_reach_id(self):
        """Test gain/loss obs without reach_id in metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            obs = [
                IWFMObservation(
                    name="sgl1",
                    value=10.0,
                    obs_type=IWFMObservationType.STREAM_GAIN_LOSS,
                    datetime=datetime(2020, 1, 1),
                    metadata={},  # No reach_id
                ),
            ]
            im = IWFMInstructionManager(output_dir=tmpdir)
            ins = im.generate_gain_loss_instructions(
                output_file=Path(tmpdir) / "gl.out",
                observations=obs,
            )
            # Should still create instruction but without reach search marker
            assert "sgl1" in ins.observations

    def test_gain_loss_default_instruction_path(self):
        """Test default instruction file path for gain/loss."""
        with tempfile.TemporaryDirectory() as tmpdir:
            obs = [
                IWFMObservation(
                    name="sgl1",
                    value=10.0,
                    obs_type=IWFMObservationType.STREAM_GAIN_LOSS,
                    datetime=datetime(2020, 1, 1),
                    metadata={"reach_id": 1},
                ),
            ]
            im = IWFMInstructionManager(output_dir=tmpdir)
            ins = im.generate_gain_loss_instructions(
                output_file=Path(tmpdir) / "GainLoss.out",
                observations=obs,
            )
            assert ins.instruction_path.name == "GainLoss_sgl.ins"


class TestBudgetInstructionsExtended:
    """Extended budget instruction tests."""

    def test_budget_no_observations_raises(self):
        """Test budget raises when no observations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            im = IWFMInstructionManager(output_dir=tmpdir)
            with pytest.raises(ValueError, match="No.*budget observations"):
                im.generate_budget_instructions(
                    budget_file=Path(tmpdir) / "test.out",
                    budget_type="gw",
                    observations=[],
                )

    def test_budget_from_om(self):
        """Test budget instructions from observation manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            om = IWFMObservationManager()
            obs = IWFMObservation(
                name="gwb1",
                value=1000.0,
                obs_type=IWFMObservationType.GW_BUDGET,
                datetime=datetime(2020, 1, 1),
                group="gwbud",
                metadata={"component": "RECHARGE", "budget_type": "gw"},
            )
            om._observations[obs.name] = obs
            grp = om.get_observation_group("gwbud")
            grp.observations.append(obs)

            im = IWFMInstructionManager(observation_manager=om, output_dir=tmpdir)
            ins = im.generate_budget_instructions(
                budget_file=Path(tmpdir) / "GW_Budget.out",
                budget_type="gw",
            )
            assert len(ins.observations) == 1

    def test_budget_stream_type(self):
        """Test stream budget instructions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            obs = [
                IWFMObservation(
                    name="strb1",
                    value=500.0,
                    obs_type=IWFMObservationType.STREAM_BUDGET,
                    datetime=datetime(2020, 1, 1),
                    metadata={"component": "INFLOW", "budget_type": "stream"},
                ),
            ]
            im = IWFMInstructionManager(output_dir=tmpdir)
            ins = im.generate_budget_instructions(
                budget_file=Path(tmpdir) / "Str_Budget.out",
                budget_type="stream",
                observations=obs,
            )
            assert len(ins.observations) == 1

    def test_budget_rootzone_type(self):
        """Test rootzone budget instructions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            obs = [
                IWFMObservation(
                    name="rzb1",
                    value=200.0,
                    obs_type=IWFMObservationType.ROOTZONE_BUDGET,
                    datetime=datetime(2020, 1, 1),
                    metadata={"component": "ET", "budget_type": "rootzone"},
                ),
            ]
            im = IWFMInstructionManager(output_dir=tmpdir)
            ins = im.generate_budget_instructions(
                budget_file=Path(tmpdir) / "RZ_Budget.out",
                budget_type="rootzone",
                observations=obs,
            )
            assert len(ins.observations) == 1

    def test_budget_lake_type(self):
        """Test lake budget instructions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            obs = [
                IWFMObservation(
                    name="lkb1",
                    value=300.0,
                    obs_type=IWFMObservationType.LAKE_BUDGET,
                    datetime=datetime(2020, 1, 1),
                    metadata={"component": "INFLOW"},
                ),
            ]
            im = IWFMInstructionManager(output_dir=tmpdir)
            ins = im.generate_budget_instructions(
                budget_file=Path(tmpdir) / "Lake_Budget.out",
                budget_type="lake",
                observations=obs,
            )
            assert len(ins.observations) == 1

    def test_budget_obs_without_datetime_skipped(self):
        """Test budget obs without datetime are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            obs = [
                IWFMObservation(
                    name="gwb_nodt",
                    value=1000.0,
                    obs_type=IWFMObservationType.GW_BUDGET,
                    datetime=None,
                    metadata={"component": "RECHARGE"},
                ),
                IWFMObservation(
                    name="gwb_dt",
                    value=1000.0,
                    obs_type=IWFMObservationType.GW_BUDGET,
                    datetime=datetime(2020, 1, 1),
                    metadata={"component": "RECHARGE"},
                ),
            ]
            im = IWFMInstructionManager(output_dir=tmpdir)
            ins = im.generate_budget_instructions(
                budget_file=Path(tmpdir) / "Budget.out",
                budget_type="gw",
                observations=obs,
            )
            assert "gwb_dt" in ins.observations
            assert "gwb_nodt" not in ins.observations

    def test_budget_obs_without_component(self):
        """Test budget obs without component in metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            obs = [
                IWFMObservation(
                    name="gwb1",
                    value=1000.0,
                    obs_type=IWFMObservationType.GW_BUDGET,
                    datetime=datetime(2020, 1, 1),
                    metadata={},  # No component
                ),
            ]
            im = IWFMInstructionManager(output_dir=tmpdir)
            ins = im.generate_budget_instructions(
                budget_file=Path(tmpdir) / "Budget.out",
                budget_type="gw",
                observations=obs,
            )
            # Should still produce instruction without component search
            assert len(ins.observations) == 1

    def test_budget_custom_instruction_file(self):
        """Test custom instruction file path for budget."""
        with tempfile.TemporaryDirectory() as tmpdir:
            obs = [
                IWFMObservation(
                    name="gwb1",
                    value=1000.0,
                    obs_type=IWFMObservationType.GW_BUDGET,
                    datetime=datetime(2020, 1, 1),
                    metadata={"component": "RECHARGE"},
                ),
            ]
            im = IWFMInstructionManager(output_dir=tmpdir)
            custom_path = Path(tmpdir) / "custom_budget.ins"
            ins = im.generate_budget_instructions(
                budget_file=Path(tmpdir) / "Budget.out",
                budget_type="gw",
                observations=obs,
                instruction_file=custom_path,
            )
            assert ins.instruction_path == custom_path


class TestLakeInstructionsExtended:
    """Extended lake instruction tests."""

    def test_lake_storage_instructions(self):
        """Test lake storage instructions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            obs = [
                IWFMObservation(
                    name="lks1",
                    value=5000.0,
                    obs_type=IWFMObservationType.LAKE_STORAGE,
                    datetime=datetime(2020, 1, 1, 0, 0),
                    group="lake_storage",
                ),
            ]
            im = IWFMInstructionManager(output_dir=tmpdir)
            ins = im.generate_lake_instructions(
                output_file=Path(tmpdir) / "Lake.out",
                observations=obs,
                variable="storage",
            )
            assert len(ins.observations) == 1
            assert ins.instruction_path.name == "Lake_lake_storage.ins"

    def test_lake_no_observations_raises(self):
        """Test lake raises when no observations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            im = IWFMInstructionManager(output_dir=tmpdir)
            with pytest.raises(ValueError, match="No lake.*observations"):
                im.generate_lake_instructions(
                    output_file=Path(tmpdir) / "test.out",
                    observations=[],
                    variable="level",
                )

    def test_lake_from_om_level(self):
        """Test lake level instructions from observation manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            om = IWFMObservationManager()
            obs = IWFMObservation(
                name="lk1",
                value=100.0,
                obs_type=IWFMObservationType.LAKE_LEVEL,
                datetime=datetime(2020, 1, 1, 0, 0),
                group="lake_level",
            )
            om._observations[obs.name] = obs
            grp = om.get_observation_group("lake_level")
            grp.observations.append(obs)

            im = IWFMInstructionManager(observation_manager=om, output_dir=tmpdir)
            ins = im.generate_lake_instructions(
                output_file=Path(tmpdir) / "Lake.out",
                variable="level",
            )
            assert len(ins.observations) == 1

    def test_lake_from_om_storage(self):
        """Test lake storage instructions from observation manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            om = IWFMObservationManager()
            obs = IWFMObservation(
                name="lks1",
                value=5000.0,
                obs_type=IWFMObservationType.LAKE_STORAGE,
                datetime=datetime(2020, 1, 1, 0, 0),
                group="lake_storage",
            )
            om._observations[obs.name] = obs

            # Need to create the storage group since it's not a default
            from pyiwfm.runner.pest_observations import IWFMObservationGroup

            om._observation_groups["lake_storage"] = IWFMObservationGroup(
                name="lake_storage",
                obs_type=IWFMObservationType.LAKE_STORAGE,
            )
            om._observation_groups["lake_storage"].observations.append(obs)

            im = IWFMInstructionManager(observation_manager=om, output_dir=tmpdir)
            ins = im.generate_lake_instructions(
                output_file=Path(tmpdir) / "Lake.out",
                variable="storage",
            )
            assert len(ins.observations) == 1

    def test_lake_custom_instruction_file(self):
        """Test custom instruction file path for lake."""
        with tempfile.TemporaryDirectory() as tmpdir:
            obs = [
                IWFMObservation(
                    name="lk1",
                    value=100.0,
                    obs_type=IWFMObservationType.LAKE_LEVEL,
                    datetime=datetime(2020, 1, 1, 0, 0),
                ),
            ]
            im = IWFMInstructionManager(output_dir=tmpdir)
            custom_path = Path(tmpdir) / "custom_lake.ins"
            ins = im.generate_lake_instructions(
                output_file=Path(tmpdir) / "Lake.out",
                observations=obs,
                instruction_file=custom_path,
            )
            assert ins.instruction_path == custom_path

    def test_lake_obs_without_datetime(self):
        """Test lake obs without datetime are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            obs = [
                IWFMObservation(
                    name="lk_nodt",
                    value=100.0,
                    obs_type=IWFMObservationType.LAKE_LEVEL,
                    datetime=None,
                ),
                IWFMObservation(
                    name="lk_dt",
                    value=101.0,
                    obs_type=IWFMObservationType.LAKE_LEVEL,
                    datetime=datetime(2020, 1, 1, 0, 0),
                ),
            ]
            im = IWFMInstructionManager(output_dir=tmpdir)
            ins = im.generate_lake_instructions(
                output_file=Path(tmpdir) / "Lake.out",
                observations=obs,
            )
            assert "lk_dt" in ins.observations
            assert "lk_nodt" not in ins.observations

    def test_lake_no_header(self):
        """Test lake instructions with no header."""
        with tempfile.TemporaryDirectory() as tmpdir:
            obs = [
                IWFMObservation(
                    name="lk1",
                    value=100.0,
                    obs_type=IWFMObservationType.LAKE_LEVEL,
                    datetime=datetime(2020, 1, 1, 0, 0),
                ),
            ]
            im = IWFMInstructionManager(output_dir=tmpdir)
            ins = im.generate_lake_instructions(
                output_file=Path(tmpdir) / "Lake.out",
                observations=obs,
                header_lines=0,
            )
            content = ins.instruction_path.read_text()
            lines = content.split("\n")
            assert lines[0] == "pif @"
            assert not lines[1].startswith("l")


class TestSubsidenceInstructionsExtended:
    """Extended subsidence instruction tests."""

    def test_subsidence_no_observations_raises(self):
        """Test subsidence raises when no observations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            im = IWFMInstructionManager(output_dir=tmpdir)
            with pytest.raises(ValueError, match="No subsidence"):
                im.generate_subsidence_instructions(
                    output_file=Path(tmpdir) / "test.out",
                    observations=[],
                )

    def test_subsidence_from_om(self):
        """Test subsidence instructions from observation manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            om = IWFMObservationManager()
            obs = IWFMObservation(
                name="sub1",
                value=0.05,
                obs_type=IWFMObservationType.SUBSIDENCE,
                datetime=datetime(2020, 1, 1, 0, 0),
                group="subsidence",
            )
            om._observations[obs.name] = obs
            grp = om.get_observation_group("subsidence")
            grp.observations.append(obs)

            im = IWFMInstructionManager(observation_manager=om, output_dir=tmpdir)
            ins = im.generate_subsidence_instructions(
                output_file=Path(tmpdir) / "Sub.out",
            )
            assert len(ins.observations) == 1

    def test_subsidence_obs_without_datetime_skipped(self):
        """Test subsidence obs without datetime are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            obs = [
                IWFMObservation(
                    name="sub_nodt",
                    value=0.05,
                    obs_type=IWFMObservationType.SUBSIDENCE,
                    datetime=None,
                ),
                IWFMObservation(
                    name="sub_dt",
                    value=0.06,
                    obs_type=IWFMObservationType.SUBSIDENCE,
                    datetime=datetime(2020, 1, 1, 0, 0),
                ),
            ]
            im = IWFMInstructionManager(output_dir=tmpdir)
            ins = im.generate_subsidence_instructions(
                output_file=Path(tmpdir) / "Sub.out",
                observations=obs,
            )
            assert "sub_dt" in ins.observations
            assert "sub_nodt" not in ins.observations

    def test_subsidence_custom_instruction_file(self):
        """Test custom instruction file for subsidence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            obs = [
                IWFMObservation(
                    name="sub1",
                    value=0.05,
                    obs_type=IWFMObservationType.SUBSIDENCE,
                    datetime=datetime(2020, 1, 1, 0, 0),
                ),
            ]
            im = IWFMInstructionManager(output_dir=tmpdir)
            custom_path = Path(tmpdir) / "custom_sub.ins"
            ins = im.generate_subsidence_instructions(
                output_file=Path(tmpdir) / "Sub.out",
                observations=obs,
                instruction_file=custom_path,
            )
            assert ins.instruction_path == custom_path

    def test_subsidence_default_path(self):
        """Test default instruction path for subsidence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            obs = [
                IWFMObservation(
                    name="sub1",
                    value=0.05,
                    obs_type=IWFMObservationType.SUBSIDENCE,
                    datetime=datetime(2020, 1, 1, 0, 0),
                ),
            ]
            im = IWFMInstructionManager(output_dir=tmpdir)
            ins = im.generate_subsidence_instructions(
                output_file=Path(tmpdir) / "Subsidence.out",
                observations=obs,
            )
            assert ins.instruction_path.name == "Subsidence_sub.ins"

    def test_subsidence_no_header(self):
        """Test subsidence with zero header lines."""
        with tempfile.TemporaryDirectory() as tmpdir:
            obs = [
                IWFMObservation(
                    name="sub1",
                    value=0.05,
                    obs_type=IWFMObservationType.SUBSIDENCE,
                    datetime=datetime(2020, 1, 1, 0, 0),
                ),
            ]
            im = IWFMInstructionManager(output_dir=tmpdir)
            ins = im.generate_subsidence_instructions(
                output_file=Path(tmpdir) / "Sub.out",
                observations=obs,
                header_lines=0,
            )
            content = ins.instruction_path.read_text()
            lines = content.split("\n")
            assert lines[0] == "pif @"
            assert not lines[1].startswith("l")


class TestCustomInstructionsExtended:
    """Extended custom instruction tests."""

    def test_custom_no_header(self):
        """Test custom instructions with no header."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "custom.out"
            output_file.write_text("data line")

            im = IWFMInstructionManager(output_dir=tmpdir)
            ins = im.generate_custom_instructions(
                output_file=output_file,
                observations=[("obs1", "MARKER", 2)],
                header_lines=0,
            )
            content = ins.instruction_path.read_text()
            lines = content.split("\n")
            assert lines[0] == "pif @"
            # No header skip line
            assert lines[1].startswith("@")

    def test_custom_instruction_file_path(self):
        """Test custom instruction file with explicit path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "custom.out"
            output_file.write_text("data")

            im = IWFMInstructionManager(output_dir=tmpdir)
            custom_path = Path(tmpdir) / "explicit.ins"
            ins = im.generate_custom_instructions(
                output_file=output_file,
                observations=[("obs1", "MARKER", 2)],
                instruction_file=custom_path,
            )
            assert ins.instruction_path == custom_path

    def test_custom_default_path(self):
        """Test default custom instruction path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "myoutput.out"
            output_file.write_text("data")

            im = IWFMInstructionManager(output_dir=tmpdir)
            ins = im.generate_custom_instructions(
                output_file=output_file,
                observations=[("obs1", "MARKER", 2)],
            )
            assert ins.instruction_path.name == "myoutput_custom.ins"

    def test_custom_multiple_markers(self):
        """Test custom instructions with multiple observations and markers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "multi.out"
            output_file.write_text("data")

            im = IWFMInstructionManager(output_dir=tmpdir, marker="$")
            ins = im.generate_custom_instructions(
                output_file=output_file,
                observations=[
                    ("obs1", "M1", 2),
                    ("obs2", "M2", 3),
                    ("obs3", "M3", 4),
                ],
            )
            content = ins.instruction_path.read_text()
            assert "pif $" in content
            assert "$M1$" in content
            assert "$M2$" in content
            assert "$M3$" in content
            assert len(ins.observations) == 3


class TestFixedFormatInstructionsExtended:
    """Extended fixed-format instruction tests."""

    def test_fixed_format_multiple_lines(self):
        """Test fixed-format with observations on multiple lines."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "fixed.out"
            output_file.write_text("header\nline1data\nline2data\nline3data")

            im = IWFMInstructionManager(output_dir=tmpdir)
            ins = im.generate_fixed_format_instructions(
                output_file=output_file,
                observations=[
                    ("obs1", 1, 1, 5),
                    ("obs2", 3, 1, 5),  # Line 3
                ],
                header_lines=1,
            )
            ins.instruction_path.read_text()
            assert ins.instruction_path.exists()
            assert len(ins.observations) == 2

    def test_fixed_format_no_header(self):
        """Test fixed-format with no header lines."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "fixed.out"
            output_file.write_text("data")

            im = IWFMInstructionManager(output_dir=tmpdir)
            ins = im.generate_fixed_format_instructions(
                output_file=output_file,
                observations=[("obs1", 1, 1, 5)],
                header_lines=0,
            )
            assert len(ins.observations) == 1

    def test_fixed_format_custom_path(self):
        """Test custom instruction file path for fixed format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "fixed.out"
            output_file.write_text("data")

            im = IWFMInstructionManager(output_dir=tmpdir)
            custom_path = Path(tmpdir) / "custom_fixed.ins"
            ins = im.generate_fixed_format_instructions(
                output_file=output_file,
                observations=[("obs1", 1, 1, 5)],
                instruction_file=custom_path,
            )
            assert ins.instruction_path == custom_path

    def test_fixed_format_default_path(self):
        """Test default instruction path for fixed format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "mydata.out"
            output_file.write_text("data")

            im = IWFMInstructionManager(output_dir=tmpdir)
            ins = im.generate_fixed_format_instructions(
                output_file=output_file,
                observations=[("obs1", 1, 1, 5)],
            )
            assert ins.instruction_path.name == "mydata_fixed.ins"

    def test_fixed_format_same_line_observations(self):
        """Test fixed-format with multiple observations on the same line."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "fixed.out"
            output_file.write_text("header\n12345678901234567890")

            im = IWFMInstructionManager(output_dir=tmpdir)
            ins = im.generate_fixed_format_instructions(
                output_file=output_file,
                observations=[
                    ("obs1", 1, 1, 5),
                    ("obs2", 1, 10, 15),
                ],
                header_lines=1,
            )
            content = ins.instruction_path.read_text()
            # Both should be on same line (no l skip between them)
            assert "[1:5]" in content
            assert "[10:15]" in content


class TestGenerateAllInstructions:
    """Tests for batch instruction generation."""

    def test_generate_all_no_om_raises(self):
        """Test batch generation raises without observation manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            im = IWFMInstructionManager(output_dir=tmpdir)
            with pytest.raises(ValueError, match="Observation manager required"):
                im.generate_all_instructions()

    def test_generate_all_with_head_observations(self):
        """Test batch generation with head observations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            om = IWFMObservationManager()
            obs = IWFMObservation(
                name="h1",
                value=100.0,
                obs_type=IWFMObservationType.HEAD,
                datetime=datetime(2020, 1, 1, 0, 0),
                group="head",
            )
            om._observations[obs.name] = obs
            grp = om.get_observation_group("head")
            grp.observations.append(obs)

            output_file = Path(tmpdir) / "GW_Heads.out"
            output_file.write_text("header\n")

            im = IWFMInstructionManager(observation_manager=om, output_dir=tmpdir)
            instructions = im.generate_all_instructions(output_files={"head": output_file})
            assert len(instructions) == 1

    def test_generate_all_empty_output_files(self):
        """Test batch generation with no matching output files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            om = IWFMObservationManager()
            obs = IWFMObservation(
                name="h1",
                value=100.0,
                obs_type=IWFMObservationType.HEAD,
                datetime=datetime(2020, 1, 1, 0, 0),
                group="head",
            )
            om._observations[obs.name] = obs

            im = IWFMInstructionManager(observation_manager=om, output_dir=tmpdir)
            instructions = im.generate_all_instructions(output_files={})
            assert len(instructions) == 0

    def test_generate_all_multiple_types(self):
        """Test batch generation with multiple observation types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            om = IWFMObservationManager()

            head_obs = IWFMObservation(
                name="h1",
                value=100.0,
                obs_type=IWFMObservationType.HEAD,
                datetime=datetime(2020, 1, 1, 0, 0),
                group="head",
            )
            om._observations[head_obs.name] = head_obs
            om.get_observation_group("head").observations.append(head_obs)

            flow_obs = IWFMObservation(
                name="f1",
                value=500.0,
                obs_type=IWFMObservationType.STREAM_FLOW,
                datetime=datetime(2020, 1, 1, 0, 0),
                group="flow",
            )
            om._observations[flow_obs.name] = flow_obs
            om.get_observation_group("flow").observations.append(flow_obs)

            head_file = Path(tmpdir) / "heads.out"
            head_file.write_text("header\n")
            flow_file = Path(tmpdir) / "flow.out"
            flow_file.write_text("header\n")

            im = IWFMInstructionManager(observation_manager=om, output_dir=tmpdir)
            instructions = im.generate_all_instructions(
                output_files={"head": head_file, "flow": flow_file}
            )
            assert len(instructions) == 2

    def test_generate_all_default_output_files(self):
        """Test batch generation with default (None) output files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            om = IWFMObservationManager()
            im = IWFMInstructionManager(observation_manager=om, output_dir=tmpdir)
            instructions = im.generate_all_instructions()
            assert len(instructions) == 0

    def test_generate_all_subsidence(self):
        """Test batch generation includes subsidence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            om = IWFMObservationManager()
            obs = IWFMObservation(
                name="sub1",
                value=0.05,
                obs_type=IWFMObservationType.SUBSIDENCE,
                datetime=datetime(2020, 1, 1, 0, 0),
                group="subsidence",
            )
            om._observations[obs.name] = obs
            om.get_observation_group("subsidence").observations.append(obs)

            sub_file = Path(tmpdir) / "sub.out"
            sub_file.write_text("header\n")

            im = IWFMInstructionManager(observation_manager=om, output_dir=tmpdir)
            instructions = im.generate_all_instructions(output_files={"subsidence": sub_file})
            assert len(instructions) == 1


class TestBuildReadInstructionExtended:
    """Extended tests for _build_read_instruction."""

    def test_column_1(self):
        """Test reading from column 1 (no whitespace skips)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            im = IWFMInstructionManager(output_dir=tmpdir)
            result = im._build_read_instruction(1, "obs1")
            assert result == "!obs1!"

    def test_column_2(self):
        """Test reading from column 2 (one whitespace skip)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            im = IWFMInstructionManager(output_dir=tmpdir)
            result = im._build_read_instruction(2, "obs1")
            assert result == "w !obs1!"

    def test_column_5(self):
        """Test reading from column 5."""
        with tempfile.TemporaryDirectory() as tmpdir:
            im = IWFMInstructionManager(output_dir=tmpdir)
            result = im._build_read_instruction(5, "test_obs")
            assert result == "w w w w !test_obs!"


class TestInstructionManagerReprExtended:
    """Extended repr tests."""

    def test_repr_with_instructions(self):
        """Test repr after creating instructions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test.out"
            output_file.write_text("data")

            im = IWFMInstructionManager(output_dir=tmpdir)
            im.generate_custom_instructions(
                output_file=output_file,
                observations=[("obs1", "MARKER", 2)],
            )
            r = repr(im)
            assert "n_instructions=1" in r
