"""Tests for runner/pest_helper.py uncovered branches.

Covers:
- build() SVD/regularization config branches (lines 740-749)
- write_forward_run_script() (lines 784-851)
- write_pp_interpolation_script() (lines 853-916)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
from datetime import datetime

import pytest


def _make_helper_for_build(tmp_path: Path, case_name: str = "test"):
    """Create an IWFMPestHelper ready for build() with mocked params/obs."""
    from pyiwfm.runner.pest_helper import IWFMPestHelper

    helper = IWFMPestHelper(
        model_dir=tmp_path,
        pest_dir=tmp_path / "pest",
        case_name=case_name,
    )

    # Mock parameters and observations so build() doesn't raise
    helper.parameters = MagicMock()
    helper.parameters.n_parameters = 5
    helper.parameters.get_all_parameters.return_value = []
    helper.observations = MagicMock()
    helper.observations.n_observations = 3
    helper.observations.get_all_observations.return_value = []

    # Clear built templates/instructions
    helper._built_templates = []
    helper._built_instructions = []
    helper._model_command = "python forward_run.py"

    return helper


class TestBuildWithSVD:
    """Test build() with SVD configuration."""

    def test_build_with_svd(self, tmp_path: Path) -> None:
        """SVD config applied in build."""
        helper = _make_helper_for_build(tmp_path, "test_svd")

        # Set up SVD config
        svd_config = MagicMock()
        svd_config.to_dict.return_value = {"svd_pack": "redsvd", "max_n_super": 10}
        helper._svd_config = svd_config

        # Mock the PESTInterface
        mock_pest = MagicMock()
        mock_pest.write_control_file.return_value = tmp_path / "pest" / "test.pst"

        with patch("pyiwfm.runner.pest_helper.PESTInterface", return_value=mock_pest):
            (tmp_path / "pest").mkdir(parents=True, exist_ok=True)
            (tmp_path / "pest" / "templates").mkdir(exist_ok=True)
            (tmp_path / "pest" / "instructions").mkdir(exist_ok=True)
            result = helper.build()

        # Verify SVD options were set
        calls = [c for c in mock_pest.set_pestpp_option.call_args_list
                 if c[0][0] in ("svd_pack", "max_n_super")]
        assert len(calls) == 2


class TestBuildWithRegularization:
    """Test build() with regularization config."""

    def test_build_with_preferred_homogeneity(self, tmp_path: Path) -> None:
        """Regularization with PREFERRED_HOMOGENEITY -> ies_reg_factor set."""
        from pyiwfm.runner.pest_helper import RegularizationType

        helper = _make_helper_for_build(tmp_path, "test_reg")

        reg = MagicMock()
        reg.reg_type = RegularizationType.PREFERRED_HOMOGENEITY
        reg.weight = 0.5
        helper._regularization = reg

        mock_pest = MagicMock()
        mock_pest.write_control_file.return_value = tmp_path / "pest" / "test.pst"

        with patch("pyiwfm.runner.pest_helper.PESTInterface", return_value=mock_pest):
            (tmp_path / "pest").mkdir(parents=True, exist_ok=True)
            (tmp_path / "pest" / "templates").mkdir(exist_ok=True)
            (tmp_path / "pest" / "instructions").mkdir(exist_ok=True)
            helper.build()

        mock_pest.set_pestpp_option.assert_any_call("ies_reg_factor", 0.5)

    def test_build_with_tikhonov(self, tmp_path: Path) -> None:
        """Regularization with TIKHONOV -> use_regul_prior set."""
        from pyiwfm.runner.pest_helper import RegularizationType

        helper = _make_helper_for_build(tmp_path, "test_tik")

        reg = MagicMock()
        reg.reg_type = RegularizationType.TIKHONOV
        helper._regularization = reg

        mock_pest = MagicMock()
        mock_pest.write_control_file.return_value = tmp_path / "pest" / "test.pst"

        with patch("pyiwfm.runner.pest_helper.PESTInterface", return_value=mock_pest):
            (tmp_path / "pest").mkdir(parents=True, exist_ok=True)
            (tmp_path / "pest" / "templates").mkdir(exist_ok=True)
            (tmp_path / "pest" / "instructions").mkdir(exist_ok=True)
            helper.build()

        mock_pest.set_pestpp_option.assert_any_call("use_regul_prior", "true")


class TestForwardRunScript:
    """Test write_forward_run_script()."""

    def test_forward_run_script(self, tmp_path: Path) -> None:
        """Script generation with imports."""
        from pyiwfm.runner.pest_helper import IWFMPestHelper

        helper = IWFMPestHelper(
            model_dir=tmp_path,
            pest_dir=tmp_path / "pest",
            case_name="test_fwd",
        )
        (tmp_path / "pest").mkdir(parents=True, exist_ok=True)

        filepath = helper.write_forward_run_script()
        assert filepath.exists()

        content = filepath.read_text()
        assert "forward model runner" in content.lower() or "Forward" in content
        assert "pyiwfm.runner" in content
        assert "IWFMRunner" in content
        assert "test_fwd" in content

    def test_forward_run_script_custom_path(self, tmp_path: Path) -> None:
        """Script with custom output path."""
        from pyiwfm.runner.pest_helper import IWFMPestHelper

        helper = IWFMPestHelper(
            model_dir=tmp_path,
            pest_dir=tmp_path / "pest",
            case_name="custom",
        )
        (tmp_path / "pest").mkdir(parents=True, exist_ok=True)

        custom_path = tmp_path / "custom_run.py"
        filepath = helper.write_forward_run_script(filepath=custom_path)
        assert filepath == custom_path
        assert filepath.exists()


class TestPPInterpolationScript:
    """Test write_pp_interpolation_script()."""

    def test_pp_interpolation_script(self, tmp_path: Path) -> None:
        """Pilot point interpolation script generation."""
        from pyiwfm.runner.pest_helper import IWFMPestHelper

        helper = IWFMPestHelper(
            model_dir=tmp_path,
            pest_dir=tmp_path / "pest",
            case_name="test_pp",
        )
        (tmp_path / "pest").mkdir(parents=True, exist_ok=True)

        filepath = helper.write_pp_interpolation_script()
        assert filepath.exists()

        content = filepath.read_text()
        assert "interpolat" in content.lower()
        assert "pilot" in content.lower()
        assert "test_pp" in content

    def test_pp_interpolation_script_custom_path(self, tmp_path: Path) -> None:
        """Pilot point script with custom output path."""
        from pyiwfm.runner.pest_helper import IWFMPestHelper

        helper = IWFMPestHelper(
            model_dir=tmp_path,
            pest_dir=tmp_path / "pest",
            case_name="pp_custom",
        )
        (tmp_path / "pest").mkdir(parents=True, exist_ok=True)

        custom_path = tmp_path / "interp.py"
        filepath = helper.write_pp_interpolation_script(filepath=custom_path)
        assert filepath == custom_path
        assert filepath.exists()


class TestRunPestpp:
    """Test run_pestpp() command execution (lines 964-968)."""

    def test_run_pestpp_success(self, tmp_path: Path) -> None:
        """run_pestpp executes subprocess.run with correct args."""
        from pyiwfm.runner.pest_helper import IWFMPestHelper

        helper = _make_helper_for_build(tmp_path, "run_test")
        helper._is_built = True

        # Create the expected .pst file
        pest_dir = tmp_path / "pest"
        pest_dir.mkdir(parents=True, exist_ok=True)
        (pest_dir / "run_test.pst").write_text("pst content")

        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch("shutil.which", return_value="/usr/bin/pestpp-glm"), \
             patch("subprocess.run", return_value=mock_result) as mock_run:
            result = helper.run_pestpp("pestpp-glm")

        assert result is mock_result
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert call_args[0][0][0] == "/usr/bin/pestpp-glm"

    def test_run_pestpp_with_extra_args(self, tmp_path: Path) -> None:
        """run_pestpp with extra_args appends them to command."""
        from pyiwfm.runner.pest_helper import IWFMPestHelper

        helper = _make_helper_for_build(tmp_path, "extra_test")
        helper._is_built = True

        pest_dir = tmp_path / "pest"
        pest_dir.mkdir(parents=True, exist_ok=True)
        (pest_dir / "extra_test.pst").write_text("pst content")

        mock_result = MagicMock()
        with patch("shutil.which", return_value="/usr/bin/pestpp-glm"), \
             patch("subprocess.run", return_value=mock_result) as mock_run:
            helper.run_pestpp("pestpp-glm", extra_args=["--verbose", "--restart"])

        call_args = mock_run.call_args[0][0]
        assert "--verbose" in call_args
        assert "--restart" in call_args


class TestRunPestppGlm:
    """Test run_pestpp_glm() with kwargs (lines 990-994)."""

    def test_run_pestpp_glm_with_kwargs(self, tmp_path: Path) -> None:
        """run_pestpp_glm with kwargs rebuilds and runs."""
        from pyiwfm.runner.pest_helper import IWFMPestHelper

        helper = _make_helper_for_build(tmp_path, "glm_test")
        helper._is_built = True

        pest_dir = tmp_path / "pest"
        pest_dir.mkdir(parents=True, exist_ok=True)
        (pest_dir / "glm_test.pst").write_text("pst")

        mock_result = MagicMock()

        with patch.object(helper, "set_pestpp_options") as mock_set, \
             patch.object(helper, "build") as mock_build, \
             patch.object(helper, "run_pestpp", return_value=mock_result) as mock_run:
            result = helper.run_pestpp_glm(n_workers=2, max_n_iter=50)

        mock_set.assert_called_once_with(max_n_iter=50)
        mock_build.assert_called_once()
        mock_run.assert_called_once_with("pestpp-glm", n_workers=2)
        assert result is mock_result


class TestRunPestppIes:
    """Test run_pestpp_ies() (lines 1018-1020)."""

    def test_run_pestpp_ies(self, tmp_path: Path) -> None:
        """run_pestpp_ies sets options, builds, and runs."""
        from pyiwfm.runner.pest_helper import IWFMPestHelper

        helper = _make_helper_for_build(tmp_path, "ies_test")
        helper._is_built = True

        mock_result = MagicMock()

        with patch.object(helper, "set_pestpp_options") as mock_set, \
             patch.object(helper, "build") as mock_build, \
             patch.object(helper, "run_pestpp", return_value=mock_result) as mock_run:
            result = helper.run_pestpp_ies(n_realizations=50, n_workers=4)

        mock_set.assert_called_once_with(ies_num_reals=50)
        mock_build.assert_called_once()
        mock_run.assert_called_once_with("pestpp-ies", n_workers=4)
        assert result is mock_result


class TestRunPestppSen:
    """Test run_pestpp_sen() (lines 1044-1050)."""

    def test_run_pestpp_sen(self, tmp_path: Path) -> None:
        """run_pestpp_sen sets GSA options, builds, and runs."""
        from pyiwfm.runner.pest_helper import IWFMPestHelper

        helper = _make_helper_for_build(tmp_path, "sen_test")
        helper._is_built = True

        mock_result = MagicMock()

        with patch.object(helper, "set_pestpp_options") as mock_set, \
             patch.object(helper, "build") as mock_build, \
             patch.object(helper, "run_pestpp", return_value=mock_result) as mock_run:
            result = helper.run_pestpp_sen(method="morris", n_samples=500)

        mock_set.assert_called_once_with(gsa_method="morris", gsa_sobol_samples=500)
        mock_build.assert_called_once()
        mock_run.assert_called_once_with("pestpp-sen", n_workers=1)
        assert result is mock_result
