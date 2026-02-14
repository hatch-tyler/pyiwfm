"""Integration tests for Simulation execution.

Tests run the Simulation executable on model files.
Marked with @pytest.mark.slow for conditional execution.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]


class TestSimulationRun:
    """Tests that run the Simulation executable."""

    def test_simulation_about_flag(self, simulation_exe: Path) -> None:
        """Test running Simulation with -about flag."""
        result = subprocess.run(
            [str(simulation_exe), "-about"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(simulation_exe.parent),
        )
        assert result.stdout or result.stderr

    def test_simulation_runs_on_sample_model(
        self,
        sample_model_path: Path,
        simulation_exe: Path,
        tmp_path: Path,
    ) -> None:
        """Test running Simulation on a copy of the sample model.

        Copies sample model to tmp_path to avoid modifying originals.
        """
        # Copy sample model
        model_copy = tmp_path / "samplemodel"
        shutil.copytree(sample_model_path, model_copy)

        # Find simulation input file
        sim_dir = model_copy / "Simulation"
        if not sim_dir.exists():
            pytest.skip("Simulation directory not found in sample model")

        # Look for simulation input file
        sim_inputs = list(sim_dir.glob("*.in")) + list(sim_dir.glob("*.dat"))
        if not sim_inputs:
            pytest.skip("No Simulation input files found")

        # Run simulation
        result = subprocess.run(
            [str(simulation_exe)],
            capture_output=True,
            text=True,
            timeout=600,
            cwd=str(sim_dir),
            input=str(sim_inputs[0].name) + "\n",
        )

        # Check output exists
        log_file = sim_dir / "SimulationMessages.out"
        assert log_file.exists() or result.returncode == 0
