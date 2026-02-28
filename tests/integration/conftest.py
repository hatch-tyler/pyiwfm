"""Pytest configuration and fixtures for integration tests.

Provides fixtures for locating IWFM sample models and executables.
Tests are skipped if paths are not available.

The sample model can be obtained three ways (checked in this order):
1. ``IWFM_SAMPLE_MODEL_DIR`` environment variable
2. Default local path ``~/OneDrive/Desktop/iwfm-2025.0.1747/samplemodel``
3. Automatic download from the CNRA data portal (cached at ``~/.cache/pyiwfm/``)

Executables are discovered via :class:`IWFMExecutableManager` which searches
local ``Bin/`` directories first and falls back to downloading Linux/Windows
binaries from GitHub releases.
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


def _resolve_sample_model_path() -> Path:
    """Resolve the sample model path (env var → default → download).

    Returns
    -------
    Path
        Path to the sample model directory.

    Raises
    ------
    pytest.skip
        If no sample model is available.
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


# ---------------------------------------------------------------------------
# Model path fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def sample_model_path_session() -> Path:
    """Session-scoped sample model path (downloads once per session)."""
    return _resolve_sample_model_path()


@pytest.fixture
def sample_model_path(sample_model_path_session: Path) -> Path:
    """Return path to the IWFM Sample Model directory.

    Resolution order:
    1. ``IWFM_SAMPLE_MODEL_DIR`` environment variable
    2. Default local path
    3. Auto-download from CNRA data portal (cached at ``~/.cache/pyiwfm/``)
    """
    return sample_model_path_session


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


# ---------------------------------------------------------------------------
# Executable fixtures (unified via IWFMExecutableManager)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def iwfm_executables(sample_model_path_session: Path):  # noqa: ANN201
    """Session-scoped IWFM executables via IWFMExecutableManager.

    Searches the ``Bin/`` sibling of the sample model first (preserves
    CNRA zip layout for local Windows users), then falls back to
    downloading platform-appropriate binaries from GitHub releases.

    Returns
    -------
    IWFMExecutables
        Dataclass with paths to preprocessor, simulation, etc.
    """
    from pyiwfm.runner.executables import IWFMExecutableManager

    search_paths: list[Path] = []
    bin_sibling = sample_model_path_session.parent / "Bin"
    if bin_sibling.exists():
        search_paths.append(bin_sibling)

    mgr = IWFMExecutableManager(search_paths=search_paths)
    try:
        exes = mgr.find_or_download()
    except RuntimeError:
        pytest.skip("IWFM executables not available (local search and GitHub download failed)")

    if not exes.preprocessor or not exes.simulation:
        pytest.skip("IWFM executables incomplete (missing preprocessor or simulation)")

    return exes


@pytest.fixture
def preprocessor_exe(iwfm_executables) -> Path:  # noqa: ANN001
    """Return path to PreProcessor executable."""
    exe = iwfm_executables.preprocessor
    if exe is None or not exe.exists():
        pytest.skip("PreProcessor executable not found")
    return exe


@pytest.fixture
def simulation_exe(iwfm_executables) -> Path:  # noqa: ANN001
    """Return path to Simulation executable."""
    exe = iwfm_executables.simulation
    if exe is None or not exe.exists():
        pytest.skip("Simulation executable not found")
    return exe
