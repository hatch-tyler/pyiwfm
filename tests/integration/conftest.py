"""Pytest configuration and fixtures for integration tests.

Provides fixtures for locating IWFM sample models and executables.
Tests are skipped if paths are not available.
"""

from __future__ import annotations

from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Model path fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_model_path() -> Path:
    """Return path to the IWFM Sample Model directory.

    Skips the test if the directory does not exist.
    """
    path = Path(r"C:\Users\hatch\OneDrive\Desktop\iwfm-2025.0.1747\samplemodel")
    if not path.exists():
        pytest.skip(f"Sample model directory not found: {path}")
    return path


@pytest.fixture
def c2vsimfg_path() -> Path:
    """Return path to the C2VSimFG model directory.

    Skips the test if the directory does not exist.
    """
    path = Path(r"C:\Users\hatch\OneDrive\Desktop\c2vsimfg")
    if not path.exists():
        pytest.skip(f"C2VSimFG model directory not found: {path}")
    return path


@pytest.fixture
def sample_exe_path(sample_model_path: Path) -> Path:
    """Return path to the IWFM executables directory.

    Skips the test if no executables are found.
    """
    bin_path = sample_model_path / "Bin"
    if not bin_path.exists():
        pytest.skip(f"Executable directory not found: {bin_path}")
    return bin_path


@pytest.fixture
def preprocessor_exe(sample_exe_path: Path) -> Path:
    """Return path to PreProcessor executable."""
    exe = sample_exe_path / "PreProcessor_x64.exe"
    if not exe.exists():
        pytest.skip(f"PreProcessor executable not found: {exe}")
    return exe


@pytest.fixture
def simulation_exe(sample_exe_path: Path) -> Path:
    """Return path to Simulation executable."""
    exe = sample_exe_path / "Simulation_x64.exe"
    if not exe.exists():
        pytest.skip(f"Simulation executable not found: {exe}")
    return exe
