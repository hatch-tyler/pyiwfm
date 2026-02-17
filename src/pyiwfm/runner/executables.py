"""IWFM executable manager for finding, downloading, and placing executables."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from pyiwfm.runner.runner import IWFMExecutables, find_iwfm_executables

logger = logging.getLogger(__name__)

_DEFAULT_GITHUB_REPO = "hatch-tyler/integrated-water-flow-model"
_DEFAULT_VERSION = "2025.0.1747"
_CACHE_DIR_NAME = ".pyiwfm"
_BIN_SUBDIR = "bin"


def _get_cache_dir() -> Path:
    """Return the pyiwfm cache directory (~/.pyiwfm/bin/)."""
    return Path.home() / _CACHE_DIR_NAME / _BIN_SUBDIR


def _detect_platform() -> str:
    """Detect the current platform for asset selection.

    Returns
    -------
    str
        'Windows' or 'Linux'.

    Raises
    ------
    RuntimeError
        If the platform is not supported.
    """
    if sys.platform == "win32":
        return "Windows"
    elif sys.platform.startswith("linux"):
        return "Linux"
    else:
        raise RuntimeError(
            f"Unsupported platform: {sys.platform}. "
            "IWFM executables are only available for Windows and Linux."
        )


def _build_asset_name(version: str, plat: str, config: str = "Release") -> str:
    """Build the GitHub release asset filename.

    Parameters
    ----------
    version : str
        IWFM version string (e.g. '2025.0.1747').
    plat : str
        Platform string ('Windows' or 'Linux').
    config : str
        Build configuration ('Release' or 'Debug').

    Returns
    -------
    str
        Asset filename.
    """
    ext = "zip" if plat == "Windows" else "tar.gz"
    return f"IWFM-{version}-{plat}-x64-{config}.{ext}"


@dataclass
class IWFMExecutableManager:
    """Manage IWFM executable discovery and download.

    Handles finding local executables, downloading from GitHub releases,
    placing executables into model directories, and verification.

    Parameters
    ----------
    github_repo : str
        GitHub repository in 'owner/repo' format.
    version : str
        IWFM version tag to download.
    config : str
        Build configuration: 'Release' or 'Debug'.
    search_paths : list[Path] | None
        Additional paths to search for local executables.

    Examples
    --------
    >>> mgr = IWFMExecutableManager()
    >>> exes = mgr.find_or_download()
    >>> print(exes.available)
    ['preprocessor', 'simulation', 'budget', 'zbudget']
    """

    github_repo: str = ""
    version: str = ""
    config: str = "Release"
    search_paths: list[Path] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Apply defaults from environment variables."""
        if not self.github_repo:
            self.github_repo = os.environ.get("IWFM_GITHUB_REPO", _DEFAULT_GITHUB_REPO)
        if not self.version:
            self.version = os.environ.get("IWFM_VERSION", _DEFAULT_VERSION)

    def find_or_download(self) -> IWFMExecutables:
        """Try to find local executables, download if not found.

        Returns
        -------
        IWFMExecutables
            Discovered or downloaded executables.
        """
        # Try local discovery first
        exes = find_iwfm_executables(search_paths=self.search_paths or None)
        if exes.preprocessor and exes.simulation:
            logger.info("Found local IWFM executables: %s", exes.available)
            return exes

        # Check cache
        cache_dir = _get_cache_dir() / self.version
        if cache_dir.exists():
            cached = find_iwfm_executables(search_paths=[cache_dir])
            if cached.preprocessor and cached.simulation:
                logger.info("Found cached IWFM executables at %s", cache_dir)
                return cached

        # Download from GitHub
        logger.info("Downloading IWFM executables from GitHub...")
        return self.download_from_github(cache_dir)

    def download_from_github(self, dest: Path | None = None) -> IWFMExecutables:
        """Download IWFM executables from a GitHub release.

        Parameters
        ----------
        dest : Path | None
            Destination directory. Defaults to ~/.pyiwfm/bin/{version}/.

        Returns
        -------
        IWFMExecutables
            Downloaded executables.

        Raises
        ------
        RuntimeError
            If the download or extraction fails.
        """
        if dest is None:
            dest = _get_cache_dir() / self.version

        plat = _detect_platform()
        asset_name = _build_asset_name(self.version, plat, self.config)
        tag = f"iwfm-{self.version}"
        url = f"https://github.com/{self.github_repo}/releases/download/{tag}/{asset_name}"

        logger.info("Downloading %s from %s", asset_name, url)

        dest.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = Path(tmpdir) / asset_name

            try:
                urllib.request.urlretrieve(url, str(archive_path))
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to download IWFM executables from {url}: {exc}"
                ) from exc

            # Extract archive
            if asset_name.endswith(".zip"):
                with zipfile.ZipFile(archive_path) as zf:
                    zf.extractall(dest)
            elif asset_name.endswith(".tar.gz"):
                with tarfile.open(archive_path, "r:gz") as tf:
                    tf.extractall(dest)
            else:
                raise RuntimeError(f"Unknown archive format: {asset_name}")

        # Make executables executable on Linux
        if plat == "Linux":
            for exe_path in dest.rglob("*"):
                if exe_path.is_file() and not exe_path.suffix:
                    exe_path.chmod(exe_path.stat().st_mode | 0o755)

        logger.info("Extracted IWFM executables to %s", dest)

        # Find executables in the extracted directory
        exes = find_iwfm_executables(search_paths=[dest])

        # Also search subdirectories (archives may contain nested dirs)
        if not (exes.preprocessor and exes.simulation):
            subdirs = [d for d in dest.iterdir() if d.is_dir()]
            if subdirs:
                exes = find_iwfm_executables(search_paths=subdirs)

        if not exes.preprocessor:
            raise RuntimeError(f"PreProcessor not found after extracting {asset_name} to {dest}")
        if not exes.simulation:
            raise RuntimeError(f"Simulation not found after extracting {asset_name} to {dest}")

        return exes

    def place_executables(
        self,
        exes: IWFMExecutables,
        model_dir: Path,
    ) -> dict[str, Path]:
        """Copy executables into a model directory.

        Parameters
        ----------
        exes : IWFMExecutables
            Source executables.
        model_dir : Path
            Target model directory.

        Returns
        -------
        dict[str, Path]
            Mapping of executable name to placed path.
        """
        placed: dict[str, Path] = {}
        model_dir.mkdir(parents=True, exist_ok=True)

        for attr in ["preprocessor", "simulation", "simulation_parallel", "budget", "zbudget"]:
            src = getattr(exes, attr)
            if src is not None and src.exists():
                dst = model_dir / src.name
                shutil.copy2(str(src), str(dst))
                placed[attr] = dst
                logger.debug("Placed %s -> %s", src.name, dst)

        # Also copy DLL/shared library files alongside executables
        if exes.preprocessor:
            src_dir = exes.preprocessor.parent
            if sys.platform == "win32":
                patterns = ["*.dll", "*.DLL"]
            else:
                patterns = ["*.so", "*.so.*"]
            for pattern in patterns:
                for lib in src_dir.glob(pattern):
                    dst = model_dir / lib.name
                    if not dst.exists():
                        shutil.copy2(str(lib), str(dst))
                        logger.debug("Placed library %s -> %s", lib.name, dst)

        return placed

    def verify_executables(self, exes: IWFMExecutables) -> dict[str, bool]:
        """Verify that executables can run.

        Attempts to run each executable with a quick invocation to check
        that it is a valid binary. On Windows, passing an empty stdin
        typically causes IWFM exes to print usage and exit.

        Parameters
        ----------
        exes : IWFMExecutables
            Executables to verify.

        Returns
        -------
        dict[str, bool]
            Mapping of executable name to verification status.
        """
        results: dict[str, bool] = {}

        for attr in ["preprocessor", "simulation", "budget", "zbudget"]:
            exe_path = getattr(exes, attr)
            if exe_path is None:
                results[attr] = False
                continue

            try:
                proc = subprocess.run(
                    [str(exe_path)],
                    input="\n",
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                # IWFM exes may return non-zero when given empty input,
                # but the fact that they ran at all is verification enough
                results[attr] = True
                logger.debug("Verified %s (return code %d)", attr, proc.returncode)
            except (OSError, subprocess.TimeoutExpired) as exc:
                logger.warning("Failed to verify %s: %s", attr, exc)
                results[attr] = False

        return results
