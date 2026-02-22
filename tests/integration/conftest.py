"""Pytest configuration and fixtures for integration tests.

Provides fixtures for locating IWFM sample models and executables.
Tests are skipped if paths are not available.

The sample model can be obtained three ways (checked in this order):
1. ``IWFM_SAMPLE_MODEL_DIR`` environment variable
2. Default local path ``~/OneDrive/Desktop/iwfm-2025.0.1747/samplemodel``
3. Automatic download from the CNRA data portal (cached at ``~/.cache/pyiwfm/``)
"""

from __future__ import annotations

import logging
import os
import shutil
import zipfile
from pathlib import Path
from urllib import request

import pytest

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default paths (fallbacks when env vars not set)
# ---------------------------------------------------------------------------

_DEFAULT_SAMPLE_MODEL = str(
    Path.home() / "OneDrive" / "Desktop" / "iwfm-2025.0.1747" / "samplemodel"
)

_IWFM_ZIP_URL = (
    "https://data.cnra.ca.gov/dataset/"
    "5c4b82c9-d219-4d71-a6cc-7ea6ccbaa54b/resource/"
    "db90dd10-4080-4bf0-b8d9-8ad83161edd5/download/"
    "iwfm-2025.0.1747.zip"
)
_CACHE_DIR = Path.home() / ".cache" / "pyiwfm" / "iwfm-2025.0.1747"

# Directories to extract from the zip (top-level entries inside the archive)
_EXTRACT_PREFIXES = ("samplemodel/", "Bin/")


# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------


def _download_and_cache_sample_model() -> Path | None:
    """Download the IWFM 2025 sample model zip and cache locally.

    Extracts only ``samplemodel/`` and ``Bin/`` from the archive.
    Returns the path to the ``samplemodel/`` directory, or ``None``
    if the download or extraction fails.
    """
    sample_dir = _CACHE_DIR / "samplemodel"
    if sample_dir.exists() and any(sample_dir.iterdir()):
        return sample_dir

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = _CACHE_DIR / "iwfm-2025.0.1747.zip"

    # Download
    if not zip_path.exists():
        logger.info("Downloading IWFM sample model from %s ...", _IWFM_ZIP_URL)
        try:
            request.urlretrieve(_IWFM_ZIP_URL, zip_path)  # noqa: S310
        except Exception:
            logger.warning("Failed to download IWFM sample model", exc_info=True)
            if zip_path.exists():
                zip_path.unlink()
            return None

    # Extract only the directories we need
    try:
        with zipfile.ZipFile(zip_path) as zf:
            members = [m for m in zf.namelist() if any(m.startswith(p) for p in _EXTRACT_PREFIXES)]
            zf.extractall(_CACHE_DIR, members=members)
    except Exception:
        logger.warning("Failed to extract IWFM sample model zip", exc_info=True)
        # Clean up partial extraction
        for prefix in _EXTRACT_PREFIXES:
            d = _CACHE_DIR / prefix.rstrip("/")
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
        return None

    # Clean up the zip to save disk space
    zip_path.unlink(missing_ok=True)

    if sample_dir.exists():
        return sample_dir
    return None


# ---------------------------------------------------------------------------
# Model path fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_model_path() -> Path:
    """Return path to the IWFM Sample Model directory.

    Resolution order:
    1. ``IWFM_SAMPLE_MODEL_DIR`` environment variable
    2. Default local path
    3. Auto-download from CNRA data portal (cached at ``~/.cache/pyiwfm/``)
    """
    env = os.environ.get("IWFM_SAMPLE_MODEL_DIR")
    if env:
        path = Path(env)
        if path.exists():
            return path
        pytest.skip(f"IWFM_SAMPLE_MODEL_DIR set but not found: {path}")

    default = Path(_DEFAULT_SAMPLE_MODEL)
    if default.exists():
        return default

    downloaded = _download_and_cache_sample_model()
    if downloaded is not None:
        return downloaded

    pytest.skip("Sample model not available (set IWFM_SAMPLE_MODEL_DIR or ensure network access)")


@pytest.fixture
def c2vsimcg_path() -> Path:
    """Return path to the C2VSimCG (Coarse Grid) model directory.

    Requires the ``C2VSIMCG_DIR`` environment variable to be set.
    Skips the test if the env var is not set or the directory does not exist.
    """
    env = os.environ.get("C2VSIMCG_DIR")
    if not env:
        pytest.skip("C2VSIMCG_DIR environment variable not set")
    path = Path(env)
    if not path.exists():
        pytest.skip(f"C2VSimCG model directory not found: {path}")
    return path


@pytest.fixture
def sample_exe_path(sample_model_path: Path) -> Path:
    """Return path to the IWFM executables directory.

    Checks sibling ``Bin/`` directory (works for both local installs and
    the cached download layout).
    """
    # Try sibling Bin/ first (standard layout for both local and cached)
    bin_path = sample_model_path.parent / "Bin"
    if not bin_path.exists():
        # Legacy layout: Bin inside sample model
        bin_path = sample_model_path / "Bin"
    if not bin_path.exists():
        pytest.skip(f"Executable directory not found near: {sample_model_path}")
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
