"""Unit tests for IWFMPestHelper main interface."""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from pyiwfm.runner.pest_helper import (
    IWFMPestHelper,
    RegularizationConfig,
    RegularizationType,
    SVDConfig,
)
from pyiwfm.runner.pest_params import IWFMParameterType


class TestSVDConfig:
    """Tests for SVDConfig."""

    def test_defaults(self):
        """Test default values."""
        cfg = SVDConfig()
        assert cfg.maxsing == 100
        assert cfg.eigthresh == 1e-6

    def test_to_dict(self):
        """Test conversion to PEST++ options."""
        cfg = SVDConfig(maxsing=50, eigthresh=1e-7)
        d = cfg.to_dict()
        assert d["svd_pack"] == "redsvd"
        assert d["max_n_super"] == 50
        assert d["eigthresh"] == 1e-7


class TestRegularizationConfig:
    """Tests for RegularizationConfig."""

    def test_defaults(self):
        """Test default values."""
        cfg = RegularizationConfig()
        assert cfg.reg_type == RegularizationType.PREFERRED_HOMOGENEITY
        assert cfg.weight == 1.0

    def test_custom(self):
        """Test custom values."""
        cfg = RegularizationConfig(
            reg_type=RegularizationType.PREFERRED_VALUE,
            weight=2.0,
            preferred_value=1.0,
        )
        assert cfg.reg_type == RegularizationType.PREFERRED_VALUE
        assert cfg.preferred_value == 1.0


class TestIWFMPestHelperInit:
    """Tests for IWFMPestHelper initialization."""

    def test_basic_init(self):
        """Test basic initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            helper = IWFMPestHelper(pest_dir=tmpdir, case_name="test")
            assert helper.case_name == "test"
            assert helper.pest_dir == Path(tmpdir)
            assert helper.n_parameters == 0
            assert helper.n_observations == 0
            assert helper._is_built is False

    def test_init_creates_pest_dir(self):
        """Test that pest_dir is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pest_dir = Path(tmpdir) / "nested" / "pest"
            IWFMPestHelper(pest_dir=pest_dir)
            assert pest_dir.exists()

    def test_init_with_model_dir(self):
        """Test initialization with model directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model"
            model_dir.mkdir()
            helper = IWFMPestHelper(
                pest_dir=tmpdir,
                model_dir=model_dir,
                case_name="cal",
            )
            assert helper.model_dir == model_dir

    def test_repr(self):
        """Test string representation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            helper = IWFMPestHelper(pest_dir=tmpdir, case_name="test")
            r = repr(helper)
            assert "IWFMPestHelper" in r
            assert "test" in r
            assert "n_params=0" in r
            assert "built=False" in r


class TestAddParameters:
    """Tests for adding parameters through helper."""

    def test_add_zone_parameters(self):
        """Test adding zone-based parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            helper = IWFMPestHelper(pest_dir=tmpdir)
            params = helper.add_zone_parameters(
                param_type=IWFMParameterType.HORIZONTAL_K,
                zones=[1, 2, 3],
                layer=1,
                bounds=(0.1, 100.0),
            )
            assert len(params) == 3
            assert helper.n_parameters == 3

    def test_add_zone_parameters_string_type(self):
        """Test adding zone parameters with string type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            helper = IWFMPestHelper(pest_dir=tmpdir)
            params = helper.add_zone_parameters(
                param_type="hk",
                zones=[1, 2],
                layer=1,
            )
            assert len(params) == 2

    def test_add_multiplier(self):
        """Test adding multiplier parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            helper = IWFMPestHelper(pest_dir=tmpdir)
            params = helper.add_multiplier(
                param_type=IWFMParameterType.PUMPING_MULT,
                bounds=(0.8, 1.2),
            )
            assert len(params) >= 1
            assert helper.n_parameters >= 1

    def test_add_pilot_points(self):
        """Test adding pilot point parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            helper = IWFMPestHelper(pest_dir=tmpdir)
            points = [(100, 200), (300, 400), (500, 600)]
            params = helper.add_pilot_points(
                param_type=IWFMParameterType.HORIZONTAL_K,
                points=points,
                layer=1,
            )
            assert len(params) == 3

    def test_add_stream_parameters(self):
        """Test adding stream parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            helper = IWFMPestHelper(pest_dir=tmpdir)
            params = helper.add_stream_parameters(
                param_type=IWFMParameterType.STREAMBED_K,
                reaches=[1, 2, 3],
            )
            assert len(params) == 3

    def test_add_rootzone_parameters(self):
        """Test adding root zone parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            helper = IWFMPestHelper(pest_dir=tmpdir)
            params = helper.add_rootzone_parameters(
                param_type=IWFMParameterType.CROP_COEFFICIENT,
                land_use_types=["corn", "alfalfa"],
            )
            assert len(params) == 2


class TestAddObservations:
    """Tests for adding observations through helper."""

    def test_add_head_observations(self):
        """Test adding head observations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            helper = IWFMPestHelper(pest_dir=tmpdir)
            times = [datetime(2020, 1, 1), datetime(2020, 2, 1)]
            values = [100.5, 99.8]
            obs = helper.add_head_observations(
                well_id="well1",
                x=100.0,
                y=200.0,
                times=times,
                values=values,
            )
            assert len(obs) == 2
            assert helper.n_observations == 2

    def test_add_streamflow_observations(self):
        """Test adding streamflow observations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            helper = IWFMPestHelper(pest_dir=tmpdir)
            times = [datetime(2020, 1, 1), datetime(2020, 2, 1)]
            values = [500.0, 450.0]
            obs = helper.add_streamflow_observations(
                gage_id="gage1",
                reach_id=1,
                times=times,
                values=values,
            )
            assert len(obs) == 2


class TestConfiguration:
    """Tests for PEST++ configuration."""

    def test_set_svd(self):
        """Test SVD configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            helper = IWFMPestHelper(pest_dir=tmpdir)
            helper.set_svd(maxsing=50, eigthresh=1e-7)
            assert helper._svd_config is not None
            assert helper._svd_config.maxsing == 50

    def test_set_regularization(self):
        """Test regularization configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            helper = IWFMPestHelper(pest_dir=tmpdir)
            helper.set_regularization(
                reg_type="preferred_homogeneity",
                weight=2.0,
            )
            assert helper._regularization is not None
            assert helper._regularization.weight == 2.0

    def test_set_model_command(self):
        """Test setting model command."""
        with tempfile.TemporaryDirectory() as tmpdir:
            helper = IWFMPestHelper(pest_dir=tmpdir)
            helper.set_model_command("python run_iwfm.py")
            assert helper._model_command == "python run_iwfm.py"

    def test_set_pestpp_options(self):
        """Test setting PEST++ options."""
        with tempfile.TemporaryDirectory() as tmpdir:
            helper = IWFMPestHelper(pest_dir=tmpdir)
            helper.set_pestpp_options(
                ies_num_reals=100,
                ies_lambda_mults="0.1,1,10",
            )
            assert helper.get_pestpp_option("ies_num_reals") == 100
            assert helper.get_pestpp_option("ies_lambda_mults") == "0.1,1,10"
            assert helper.get_pestpp_option("nonexistent", "default") == "default"


class TestBuild:
    """Tests for building PEST++ setup."""

    def _create_helper_with_data(self, tmpdir: str) -> IWFMPestHelper:
        """Create a helper with parameters and observations."""
        helper = IWFMPestHelper(pest_dir=tmpdir, case_name="test_build")
        helper.add_zone_parameters(
            param_type=IWFMParameterType.HORIZONTAL_K,
            zones=[1, 2],
            layer=1,
        )
        helper.add_head_observations(
            well_id="w1",
            x=100.0,
            y=200.0,
            times=[datetime(2020, 1, 1)],
            values=[100.0],
        )
        return helper

    def test_build_creates_control_file(self):
        """Test that build creates the control file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            helper = self._create_helper_with_data(tmpdir)
            pst_path = helper.build()
            assert pst_path.exists()
            assert pst_path.name == "test_build.pst"

    def test_build_creates_forward_run_script(self):
        """Test that build creates the forward run script."""
        with tempfile.TemporaryDirectory() as tmpdir:
            helper = self._create_helper_with_data(tmpdir)
            helper.build()
            script = Path(tmpdir) / "forward_run.py"
            assert script.exists()

    def test_build_marks_as_built(self):
        """Test that build sets is_built flag."""
        with tempfile.TemporaryDirectory() as tmpdir:
            helper = self._create_helper_with_data(tmpdir)
            assert not helper._is_built
            helper.build()
            assert helper._is_built

    def test_build_with_svd(self):
        """Test build with SVD configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            helper = self._create_helper_with_data(tmpdir)
            helper.set_svd(maxsing=50)
            pst_path = helper.build()
            content = pst_path.read_text()
            assert "max_n_super" in content

    def test_build_no_parameters_raises(self):
        """Test build with no parameters raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            helper = IWFMPestHelper(pest_dir=tmpdir)
            helper.add_head_observations(
                well_id="w1",
                x=0,
                y=0,
                times=[datetime(2020, 1, 1)],
                values=[100.0],
            )
            with pytest.raises(ValueError, match="No parameters"):
                helper.build()

    def test_build_no_observations_raises(self):
        """Test build with no observations raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            helper = IWFMPestHelper(pest_dir=tmpdir)
            helper.add_zone_parameters("hk", zones=[1], layer=1)
            with pytest.raises(ValueError, match="No observations"):
                helper.build()

    def test_build_custom_pst_path(self):
        """Test build with custom control file path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            helper = self._create_helper_with_data(tmpdir)
            custom_path = Path(tmpdir) / "custom.pst"
            result = helper.build(pst_file=custom_path)
            assert result == custom_path
            assert custom_path.exists()

    def test_build_creates_subdirectories(self):
        """Test that build creates template/instruction subdirs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            helper = self._create_helper_with_data(tmpdir)
            helper.build()
            assert (Path(tmpdir) / "templates").exists()
            assert (Path(tmpdir) / "instructions").exists()


class TestSummary:
    """Tests for summary method."""

    def test_summary(self):
        """Test getting setup summary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            helper = IWFMPestHelper(pest_dir=tmpdir, case_name="test")
            helper.add_zone_parameters("hk", zones=[1, 2], layer=1)
            helper.add_head_observations(
                well_id="w1",
                x=0,
                y=0,
                times=[datetime(2020, 1, 1)],
                values=[100.0],
            )

            summary = helper.summary()
            assert summary["case_name"] == "test"
            assert summary["n_parameters"] == 2
            assert summary["n_observations"] == 1
            assert summary["is_built"] is False

    def test_summary_after_build(self):
        """Test summary after building."""
        with tempfile.TemporaryDirectory() as tmpdir:
            helper = IWFMPestHelper(pest_dir=tmpdir, case_name="test")
            helper.add_zone_parameters("hk", zones=[1], layer=1)
            helper.add_head_observations(
                well_id="w1",
                x=0,
                y=0,
                times=[datetime(2020, 1, 1)],
                values=[100.0],
            )
            helper.build()

            summary = helper.summary()
            assert summary["is_built"] is True


class TestWriteScripts:
    """Tests for script generation."""

    def test_write_forward_run_script(self):
        """Test writing forward run script."""
        with tempfile.TemporaryDirectory() as tmpdir:
            helper = IWFMPestHelper(pest_dir=tmpdir, case_name="test")
            script_path = helper.write_forward_run_script()
            assert script_path.exists()
            content = script_path.read_text()
            assert "forward model runner" in content.lower()
            assert "def main" in content

    def test_write_pp_interpolation_script(self):
        """Test writing pilot point interpolation script."""
        with tempfile.TemporaryDirectory() as tmpdir:
            helper = IWFMPestHelper(pest_dir=tmpdir, case_name="test")
            script_path = helper.write_pp_interpolation_script()
            assert script_path.exists()
            content = script_path.read_text()
            assert "pilot point" in content.lower()


class TestRunPestpp:
    """Tests for PEST++ execution methods."""

    def test_run_before_build_raises(self):
        """Test that running before build raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            helper = IWFMPestHelper(pest_dir=tmpdir)
            with pytest.raises(RuntimeError, match="Must call build"):
                helper.run_pestpp()


class TestPropertyMethods:
    """Tests for property and query methods."""

    def test_parameter_groups(self):
        """Test getting parameter group names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            helper = IWFMPestHelper(pest_dir=tmpdir)
            helper.add_zone_parameters("hk", zones=[1, 2], layer=1, group="aquifer")
            groups = helper.parameter_groups
            assert "aquifer" in groups

    def test_observation_groups(self):
        """Test getting observation group names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            helper = IWFMPestHelper(pest_dir=tmpdir)
            helper.add_head_observations(
                well_id="w1",
                x=0,
                y=0,
                times=[datetime(2020, 1, 1)],
                values=[100.0],
                group="heads",
            )
            groups = helper.observation_groups
            assert "heads" in groups

    def test_add_template_and_instruction(self):
        """Test registering templates and instructions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from pyiwfm.runner.pest import InstructionFile, TemplateFile

            helper = IWFMPestHelper(pest_dir=tmpdir)

            tpl = TemplateFile(
                template_path=Path(tmpdir) / "test.tpl",
                input_path=Path(tmpdir) / "test.dat",
            )
            helper.add_template(tpl)
            assert len(helper._built_templates) == 1

            ins = InstructionFile(
                instruction_path=Path(tmpdir) / "test.ins",
                output_path=Path(tmpdir) / "test.out",
            )
            helper.add_instruction(ins)
            assert len(helper._built_instructions) == 1


# =========================================================================
# Additional tests to increase coverage to 95%+
# =========================================================================


class TestAddParametersEdgeCases:
    """Tests for edge cases in adding parameters through helper."""

    def test_add_multiplier_string_type(self):
        """Test adding multiplier with string param type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            helper = IWFMPestHelper(pest_dir=tmpdir)
            params = helper.add_multiplier(
                param_type="pump",
                bounds=(0.8, 1.2),
            )
            assert len(params) >= 1

    def test_add_multiplier_with_zones(self):
        """Test adding multiplier with zone spatial extent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            helper = IWFMPestHelper(pest_dir=tmpdir)
            params = helper.add_multiplier(
                param_type=IWFMParameterType.RECHARGE_MULT,
                spatial="zone",
                zones=[1, 2, 3],
                bounds=(0.5, 2.0),
                transform="none",
            )
            assert len(params) >= 1

    def test_add_pilot_points_string_type(self):
        """Test adding pilot points with string param type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            helper = IWFMPestHelper(pest_dir=tmpdir)
            points = [(100.0, 200.0), (300.0, 400.0)]
            params = helper.add_pilot_points(
                param_type="hk",
                points=points,
                layer=1,
                prefix="pp",
            )
            assert len(params) == 2

    def test_add_pilot_points_with_variogram_dict(self):
        """Test adding pilot points with variogram provided as dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            helper = IWFMPestHelper(pest_dir=tmpdir)
            points = [(100.0, 200.0)]
            params = helper.add_pilot_points(
                param_type=IWFMParameterType.HORIZONTAL_K,
                points=points,
                layer=1,
                variogram={"variogram_type": "exponential", "a": 10000, "sill": 1.0},
            )
            assert len(params) == 1

    def test_add_stream_parameters_string_type(self):
        """Test adding stream parameters with string param type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            helper = IWFMPestHelper(pest_dir=tmpdir)
            params = helper.add_stream_parameters(
                param_type="strk",
                reaches=[1, 2],
            )
            assert len(params) == 2

    def test_add_rootzone_parameters_string_type(self):
        """Test adding root zone parameters with string param type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            helper = IWFMPestHelper(pest_dir=tmpdir)
            params = helper.add_rootzone_parameters(
                param_type="kc",
                land_use_types=["wheat", "rice"],
            )
            assert len(params) == 2


class TestAddObservationsEdgeCases:
    """Tests for observation edge cases through helper."""

    def test_add_head_observations_custom_group(self):
        """Test adding head observations with a custom group name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            helper = IWFMPestHelper(pest_dir=tmpdir)
            times = [datetime(2020, 1, 1), datetime(2020, 6, 1)]
            values = [100.0, 99.0]
            obs = helper.add_head_observations(
                well_id="test_well",
                x=1000.0,
                y=2000.0,
                times=times,
                values=values,
                layer=2,
                weight=2.0,
                group="custom_heads",
            )
            assert len(obs) == 2
            assert all(o.group == "custom_heads" for o in obs)

    def test_add_head_observations_auto_group(self):
        """Test that auto-generated group name is based on well_id."""
        with tempfile.TemporaryDirectory() as tmpdir:
            helper = IWFMPestHelper(pest_dir=tmpdir)
            times = [datetime(2020, 1, 1)]
            values = [50.0]
            obs = helper.add_head_observations(
                well_id="my_well_001",
                x=0.0,
                y=0.0,
                times=times,
                values=values,
            )
            assert obs[0].group.startswith("head_")

    def test_add_streamflow_observations_custom_group(self):
        """Test adding streamflow observations with custom group."""
        with tempfile.TemporaryDirectory() as tmpdir:
            helper = IWFMPestHelper(pest_dir=tmpdir)
            times = [datetime(2020, 3, 15), datetime(2020, 6, 15)]
            values = [1000.0, 500.0]
            obs = helper.add_streamflow_observations(
                gage_id="usgs_1234",
                reach_id=5,
                times=times,
                values=values,
                weight=0.5,
                transform="log",
                group="custom_flow",
            )
            assert len(obs) == 2
            assert all(o.group == "custom_flow" for o in obs)

    def test_add_streamflow_observations_auto_group(self):
        """Test streamflow observations auto-generated group name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            helper = IWFMPestHelper(pest_dir=tmpdir)
            times = [datetime(2020, 1, 1)]
            values = [250.0]
            obs = helper.add_streamflow_observations(
                gage_id="gage_abc",
                reach_id=1,
                times=times,
                values=values,
            )
            assert obs[0].group.startswith("flow_")

    def test_balance_observation_weights(self):
        """Test balancing observation weights across groups."""
        with tempfile.TemporaryDirectory() as tmpdir:
            helper = IWFMPestHelper(pest_dir=tmpdir)
            # Add observations in two groups
            helper.add_head_observations(
                well_id="w1",
                x=0,
                y=0,
                times=[datetime(2020, 1, 1)],
                values=[100.0],
                group="heads",
            )
            helper.add_streamflow_observations(
                gage_id="g1",
                reach_id=1,
                times=[datetime(2020, 1, 1)],
                values=[500.0],
                group="flows",
            )
            # Should not raise
            helper.balance_observation_weights(contributions={"heads": 0.5, "flows": 0.5})


class TestConfigurationEdgeCases:
    """Tests for additional configuration edge cases."""

    def test_set_regularization_tikhonov(self):
        """Test setting Tikhonov regularization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            helper = IWFMPestHelper(pest_dir=tmpdir)
            helper.set_regularization(
                reg_type="tikhonov",
                weight=1.5,
            )
            assert helper._regularization is not None
            assert helper._regularization.reg_type == RegularizationType.TIKHONOV

    def test_set_regularization_preferred_value(self):
        """Test setting preferred value regularization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            helper = IWFMPestHelper(pest_dir=tmpdir)
            helper.set_regularization(
                reg_type="preferred_value",
                weight=1.0,
                preferred_value=1.0,
            )
            assert helper._regularization.reg_type == RegularizationType.PREFERRED_VALUE
            assert helper._regularization.preferred_value == 1.0

    def test_get_pestpp_option_default(self):
        """Test getting nonexistent option returns default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            helper = IWFMPestHelper(pest_dir=tmpdir)
            assert helper.get_pestpp_option("unknown_option") is None
            assert helper.get_pestpp_option("unknown_option", 42) == 42


class TestBuildEdgeCases:
    """Tests for build edge cases."""

    def _create_helper_with_data(self, tmpdir: str) -> IWFMPestHelper:
        """Create a helper with parameters and observations."""
        helper = IWFMPestHelper(pest_dir=tmpdir, case_name="test_build")
        helper.add_zone_parameters(
            param_type=IWFMParameterType.HORIZONTAL_K,
            zones=[1, 2],
            layer=1,
        )
        helper.add_head_observations(
            well_id="w1",
            x=100.0,
            y=200.0,
            times=[datetime(2020, 1, 1)],
            values=[100.0],
        )
        return helper

    def test_build_with_regularization_homogeneity(self):
        """Test build with preferred homogeneity regularization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            helper = self._create_helper_with_data(tmpdir)
            helper.set_regularization(
                reg_type="preferred_homogeneity",
                weight=2.0,
            )
            pst_path = helper.build()
            content = pst_path.read_text()
            assert "ies_reg_factor" in content

    def test_build_with_regularization_tikhonov(self):
        """Test build with Tikhonov regularization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            helper = self._create_helper_with_data(tmpdir)
            helper.set_regularization(reg_type="tikhonov")
            pst_path = helper.build()
            content = pst_path.read_text()
            assert "use_regul_prior" in content

    def test_build_with_templates_and_instructions(self):
        """Test build with registered templates and instructions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from pyiwfm.runner.pest import InstructionFile, TemplateFile

            helper = self._create_helper_with_data(tmpdir)

            # Create template and instruction files
            tpl_path = Path(tmpdir) / "templates" / "test.tpl"
            tpl_path.parent.mkdir(parents=True, exist_ok=True)
            tpl_path.write_text("ptf ~\n~param1~\n")

            ins_path = Path(tmpdir) / "instructions" / "test.ins"
            ins_path.parent.mkdir(parents=True, exist_ok=True)
            ins_path.write_text("pif @\n@obs1@\n")

            tpl = TemplateFile(
                template_path=tpl_path,
                input_path=Path(tmpdir) / "test.dat",
            )
            ins = InstructionFile(
                instruction_path=ins_path,
                output_path=Path(tmpdir) / "test.out",
            )

            helper.add_template(tpl)
            helper.add_instruction(ins)
            pst_path = helper.build()
            assert pst_path.exists()

    def test_build_with_pestpp_options(self):
        """Test build with custom PEST++ options."""
        with tempfile.TemporaryDirectory() as tmpdir:
            helper = self._create_helper_with_data(tmpdir)
            helper.set_pestpp_options(
                ies_num_reals=200,
                ies_lambda_mults="0.1,1,10",
            )
            pst_path = helper.build()
            content = pst_path.read_text()
            assert "ies_num_reals" in content


class TestWriteScriptsEdgeCases:
    """Tests for script generation edge cases."""

    def test_write_forward_run_script_custom_path(self):
        """Test writing forward run script to custom path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            helper = IWFMPestHelper(pest_dir=tmpdir, case_name="custom")
            custom_path = Path(tmpdir) / "scripts" / "run_model.py"
            custom_path.parent.mkdir(parents=True, exist_ok=True)
            result = helper.write_forward_run_script(filepath=custom_path)
            assert result == custom_path
            assert custom_path.exists()
            content = custom_path.read_text()
            assert "custom" in content

    def test_write_pp_interpolation_script_custom_path(self):
        """Test writing pilot point script to custom path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            helper = IWFMPestHelper(pest_dir=tmpdir, case_name="pp_test")
            custom_path = Path(tmpdir) / "scripts" / "interpolate.py"
            custom_path.parent.mkdir(parents=True, exist_ok=True)
            result = helper.write_pp_interpolation_script(filepath=custom_path)
            assert result == custom_path
            assert custom_path.exists()
            content = custom_path.read_text()
            assert "pp_test" in content


class TestRunPestppEdgeCases:
    """Tests for PEST++ execution edge cases."""

    def test_run_pestpp_not_built_raises(self):
        """Test running PEST++ before build raises RuntimeError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            helper = IWFMPestHelper(pest_dir=tmpdir)
            with pytest.raises(RuntimeError, match="Must call build"):
                helper.run_pestpp("pestpp-glm")

    def test_run_pestpp_pst_missing_raises(self):
        """Test running PEST++ when pst file is missing raises FileNotFoundError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            helper = IWFMPestHelper(pest_dir=tmpdir)
            helper._is_built = True  # Simulate built state
            with pytest.raises(FileNotFoundError, match="Control file not found"):
                helper.run_pestpp("pestpp-glm")

    def test_run_pestpp_exe_not_found_raises(self):
        """Test running PEST++ when executable is not on PATH."""
        with tempfile.TemporaryDirectory() as tmpdir:
            helper = IWFMPestHelper(pest_dir=tmpdir, case_name="test")
            helper._is_built = True
            # Create a dummy .pst file so the file check passes
            pst_file = Path(tmpdir) / "test.pst"
            pst_file.write_text("dummy pst content")
            with pytest.raises(FileNotFoundError, match="PEST\\+\\+ executable not found"):
                helper.run_pestpp("nonexistent_pestpp_program_xyz")


class TestSummaryEdgeCases:
    """Tests for summary with different regularization types."""

    def test_summary_with_regularization(self):
        """Test summary shows regularization type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            helper = IWFMPestHelper(pest_dir=tmpdir, case_name="test")
            helper.set_regularization(reg_type="tikhonov")
            helper.add_zone_parameters("hk", zones=[1], layer=1)
            helper.add_head_observations(
                well_id="w1",
                x=0,
                y=0,
                times=[datetime(2020, 1, 1)],
                values=[100.0],
            )
            summary = helper.summary()
            assert summary["regularization"] == "tikhonov"

    def test_summary_no_regularization(self):
        """Test summary with no regularization configured."""
        with tempfile.TemporaryDirectory() as tmpdir:
            helper = IWFMPestHelper(pest_dir=tmpdir, case_name="test")
            summary = helper.summary()
            assert summary["regularization"] == "none"

    def test_summary_with_pestpp_options(self):
        """Test summary includes PEST++ options."""
        with tempfile.TemporaryDirectory() as tmpdir:
            helper = IWFMPestHelper(pest_dir=tmpdir)
            helper.set_pestpp_options(ies_num_reals=50)
            summary = helper.summary()
            assert summary["pestpp_options"]["ies_num_reals"] == 50


class TestRegularizationTypeEnum:
    """Tests for RegularizationType enum values."""

    def test_none_type(self):
        """Test NONE regularization type."""
        assert RegularizationType.NONE.value == "none"

    def test_preferred_value_type(self):
        """Test PREFERRED_VALUE regularization type."""
        assert RegularizationType.PREFERRED_VALUE.value == "preferred_value"

    def test_tikhonov_type(self):
        """Test TIKHONOV regularization type."""
        assert RegularizationType.TIKHONOV.value == "tikhonov"


class TestInitWithNoModelDir:
    """Tests for initialization when model_dir is None."""

    def test_model_dir_defaults_to_pest_dir(self):
        """Test that model_dir defaults to pest_dir when not provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            helper = IWFMPestHelper(pest_dir=tmpdir)
            assert helper.model_dir == Path(tmpdir)

    def test_init_with_model(self):
        """Test initialization with a model object."""
        from unittest.mock import MagicMock

        with tempfile.TemporaryDirectory() as tmpdir:
            model = MagicMock()
            helper = IWFMPestHelper(pest_dir=tmpdir, model=model)
            assert helper.model is model
