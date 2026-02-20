"""Pytest configuration and fixtures for integration tests.

Provides fixtures for locating IWFM sample models and executables.
Tests are skipped if paths are not available.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Default paths (fallbacks when env vars not set)
# These use Path.home() to avoid hardcoding Windows user paths.
# Set IWFM_SAMPLE_MODEL_DIR / C2VSIMFG_DIR env vars for custom locations.
# ---------------------------------------------------------------------------

_DEFAULT_SAMPLE_MODEL = str(
    Path.home() / "OneDrive" / "Desktop" / "iwfm-2025.0.1747" / "samplemodel"
)
_DEFAULT_C2VSIMFG = str(Path.home() / "OneDrive" / "Desktop" / "c2vsimfg")
_DEFAULT_C2VSIMCG = str(
    Path.home() / "OneDrive" / "Desktop" / "c2vsimcg" / "C2VSimCG_v2025_WY1974-2015"
)


# ---------------------------------------------------------------------------
# Model path fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_model_path() -> Path:
    """Return path to the IWFM Sample Model directory.

    Reads from IWFM_SAMPLE_MODEL_DIR env var, falling back to default.
    Skips the test if the directory does not exist.
    """
    path = Path(os.environ.get("IWFM_SAMPLE_MODEL_DIR", _DEFAULT_SAMPLE_MODEL))
    if not path.exists():
        pytest.skip(f"Sample model directory not found: {path}")
    return path


@pytest.fixture
def c2vsimfg_path() -> Path:
    """Return path to the C2VSimFG model directory.

    Reads from C2VSIMFG_DIR env var, falling back to default.
    Skips the test if the directory does not exist.
    """
    path = Path(os.environ.get("C2VSIMFG_DIR", _DEFAULT_C2VSIMFG))
    if not path.exists():
        pytest.skip(f"C2VSimFG model directory not found: {path}")
    return path


@pytest.fixture
def c2vsimcg_path() -> Path:
    """Return path to the C2VSimCG (Coarse Grid) model directory.

    Reads from C2VSIMCG_DIR env var, falling back to default.
    Skips the test if the directory does not exist.
    """
    path = Path(os.environ.get("C2VSIMCG_DIR", _DEFAULT_C2VSIMCG))
    if not path.exists():
        pytest.skip(f"C2VSimCG model directory not found: {path}")
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
