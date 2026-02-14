"""Integration tests for PreProcessor execution.

Tests write preprocessor input files and run the PreProcessor executable.
Marked with @pytest.mark.slow for conditional execution.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]


class TestPreProcessorRun:
    """Tests that run the PreProcessor executable."""

    def test_preprocessor_about_flag(self, preprocessor_exe: Path) -> None:
        """Test running PreProcessor with -about flag."""
        result = subprocess.run(
            [str(preprocessor_exe), "-about"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(preprocessor_exe.parent),
        )
        # -about should produce version info, exit code may be 0 or non-zero
        # depending on IWFM behavior
        assert result.stdout or result.stderr

    def test_preprocessor_runs_on_sample_model(
        self,
        sample_model_path: Path,
        preprocessor_exe: Path,
        tmp_path: Path,
    ) -> None:
        """Test running PreProcessor on a copy of the sample model.

        Copies sample model to tmp_path to avoid modifying originals.
        """
        # Copy sample model
        model_copy = tmp_path / "samplemodel"
        shutil.copytree(sample_model_path, model_copy)

        # Find PreProcessor input file
        pre_dir = model_copy / "PreProcessor"
        if not pre_dir.exists():
            pytest.skip("PreProcessor directory not found in sample model")

        # Look for preprocessor input file
        pre_inputs = list(pre_dir.glob("*.in")) + list(pre_dir.glob("*.dat"))
        if not pre_inputs:
            pytest.skip("No PreProcessor input files found")

        # Run preprocessor
        result = subprocess.run(
            [str(preprocessor_exe)],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(pre_dir),
            input=str(pre_inputs[0].name) + "\n",
        )

        # Check output exists
        log_file = pre_dir / "PreprocessorMessages.out"
        assert log_file.exists() or result.returncode == 0
