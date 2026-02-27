"""Tests for pyiwfm.runner.executables: platform detection, asset naming,
executable manager, download, placement, and verification."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pyiwfm.runner.executables import (
    IWFMExecutableManager,
    _build_asset_name,
    _detect_platform,
)

# ---------------------------------------------------------------------------
# _detect_platform
# ---------------------------------------------------------------------------


class TestDetectPlatform:
    """Tests for _detect_platform."""

    def test_win32_returns_windows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("pyiwfm.runner.executables.sys.platform", "win32")
        assert _detect_platform() == "Windows"

    def test_linux_returns_linux(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("pyiwfm.runner.executables.sys.platform", "linux")
        assert _detect_platform() == "Linux"

    def test_darwin_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("pyiwfm.runner.executables.sys.platform", "darwin")
        with pytest.raises(RuntimeError, match="Unsupported platform"):
            _detect_platform()


# ---------------------------------------------------------------------------
# _build_asset_name
# ---------------------------------------------------------------------------


class TestBuildAssetName:
    """Tests for _build_asset_name."""

    def test_windows_zip(self) -> None:
        name = _build_asset_name("2025.0.1747", "Windows")
        assert name == "IWFM-2025.0.1747-Windows-x64-Release.zip"

    def test_linux_tar_gz(self) -> None:
        name = _build_asset_name("2025.0.1747", "Linux")
        assert name == "IWFM-2025.0.1747-Linux-x64-Release.tar.gz"

    def test_debug_config(self) -> None:
        name = _build_asset_name("1.0", "Windows", config="Debug")
        assert name == "IWFM-1.0-Windows-x64-Debug.zip"


# ---------------------------------------------------------------------------
# IWFMExecutableManager.__post_init__
# ---------------------------------------------------------------------------


class TestExecutableManagerPostInit:
    """Tests for IWFMExecutableManager env-var overrides."""

    def test_defaults_applied(self) -> None:
        mgr = IWFMExecutableManager()
        assert mgr.github_repo != ""
        assert mgr.version != ""

    def test_env_var_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("IWFM_GITHUB_REPO", "my-org/my-repo")
        monkeypatch.setenv("IWFM_VERSION", "9.9.9")
        mgr = IWFMExecutableManager()
        assert mgr.github_repo == "my-org/my-repo"
        assert mgr.version == "9.9.9"

    def test_explicit_values_not_overridden(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("IWFM_GITHUB_REPO", "env-org/env-repo")
        mgr = IWFMExecutableManager(github_repo="explicit/repo", version="1.2.3")
        assert mgr.github_repo == "explicit/repo"
        assert mgr.version == "1.2.3"


# ---------------------------------------------------------------------------
# find_or_download
# ---------------------------------------------------------------------------


class TestFindOrDownload:
    """Tests for IWFMExecutableManager.find_or_download."""

    @patch("pyiwfm.runner.executables.find_iwfm_executables")
    def test_local_found(self, mock_find: MagicMock) -> None:
        """When local executables are found, returns them without downloading."""
        exes = MagicMock()
        exes.preprocessor = Path("/usr/bin/iwfm_preproc")
        exes.simulation = Path("/usr/bin/iwfm_sim")
        mock_find.return_value = exes

        mgr = IWFMExecutableManager()
        result = mgr.find_or_download()

        assert result is exes
        mock_find.assert_called_once()


# ---------------------------------------------------------------------------
# download_from_github
# ---------------------------------------------------------------------------


class TestDownloadFromGithub:
    """Tests for IWFMExecutableManager.download_from_github."""

    @patch("pyiwfm.runner.executables._detect_platform", return_value="Windows")
    @patch("pyiwfm.runner.executables.find_iwfm_executables")
    @patch("pyiwfm.runner.executables.urllib.request.urlretrieve")
    @patch("pyiwfm.runner.executables.zipfile.ZipFile")
    def test_windows_download_and_extract(
        self,
        mock_zipfile: MagicMock,
        mock_urlretrieve: MagicMock,
        mock_find: MagicMock,
        mock_platform: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Downloads a .zip archive on Windows and extracts it."""
        exes = MagicMock()
        exes.preprocessor = tmp_path / "PreProcessor.exe"
        exes.simulation = tmp_path / "Simulation.exe"
        mock_find.return_value = exes

        # Make the ZipFile context manager work
        mock_zf_instance = MagicMock()
        mock_zipfile.return_value.__enter__ = MagicMock(return_value=mock_zf_instance)
        mock_zipfile.return_value.__exit__ = MagicMock(return_value=False)

        mgr = IWFMExecutableManager(version="2025.0.1747")
        result = mgr.download_from_github(dest=tmp_path)

        assert result is exes
        mock_urlretrieve.assert_called_once()

    @patch("pyiwfm.runner.executables._detect_platform", return_value="Windows")
    @patch("pyiwfm.runner.executables.urllib.request.urlretrieve")
    def test_download_failure_raises(
        self,
        mock_urlretrieve: MagicMock,
        mock_platform: MagicMock,
        tmp_path: Path,
    ) -> None:
        """RuntimeError is raised when download fails."""
        mock_urlretrieve.side_effect = OSError("Network error")

        mgr = IWFMExecutableManager(version="2025.0.1747")
        with pytest.raises(RuntimeError, match="Failed to download"):
            mgr.download_from_github(dest=tmp_path)


# ---------------------------------------------------------------------------
# place_executables
# ---------------------------------------------------------------------------


class TestPlaceExecutables:
    """Tests for IWFMExecutableManager.place_executables."""

    @patch("pyiwfm.runner.executables.shutil.copy2")
    def test_copies_existing_executables(self, mock_copy2: MagicMock, tmp_path: Path) -> None:
        """Copies each existing executable into the model directory."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        preproc = src_dir / "PreProcessor.exe"
        preproc.touch()
        sim = src_dir / "Simulation.exe"
        sim.touch()

        exes = MagicMock()
        exes.preprocessor = preproc
        exes.simulation = sim
        exes.simulation_parallel = None
        exes.budget = None
        exes.zbudget = None

        model_dir = tmp_path / "model"
        mgr = IWFMExecutableManager()
        placed = mgr.place_executables(exes, model_dir)

        assert "preprocessor" in placed
        assert "simulation" in placed
        assert model_dir.exists()


# ---------------------------------------------------------------------------
# verify_executables
# ---------------------------------------------------------------------------


class TestVerifyExecutables:
    """Tests for IWFMExecutableManager.verify_executables."""

    @patch("pyiwfm.runner.executables.subprocess.run")
    def test_success(self, mock_run: MagicMock) -> None:
        """Verified executables are marked True."""
        mock_run.return_value = MagicMock(returncode=0)
        exes = MagicMock()
        exes.preprocessor = Path("/bin/preproc")
        exes.simulation = Path("/bin/sim")
        exes.budget = None
        exes.zbudget = None

        mgr = IWFMExecutableManager()
        results = mgr.verify_executables(exes)

        assert results["preprocessor"] is True
        assert results["simulation"] is True
        assert results["budget"] is False
        assert results["zbudget"] is False

    @patch("pyiwfm.runner.executables.subprocess.run")
    def test_timeout(self, mock_run: MagicMock) -> None:
        """TimeoutExpired marks the executable as False."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="exe", timeout=10)
        exes = MagicMock()
        exes.preprocessor = Path("/bin/preproc")
        exes.simulation = None
        exes.budget = None
        exes.zbudget = None

        mgr = IWFMExecutableManager()
        results = mgr.verify_executables(exes)

        assert results["preprocessor"] is False

    @patch("pyiwfm.runner.executables.subprocess.run")
    def test_os_error(self, mock_run: MagicMock) -> None:
        """OSError marks the executable as False."""
        mock_run.side_effect = OSError("No such file")
        exes = MagicMock()
        exes.preprocessor = Path("/bin/preproc")
        exes.simulation = None
        exes.budget = None
        exes.zbudget = None

        mgr = IWFMExecutableManager()
        results = mgr.verify_executables(exes)

        assert results["preprocessor"] is False
