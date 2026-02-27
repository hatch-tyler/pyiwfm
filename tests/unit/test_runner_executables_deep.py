"""Deep coverage tests for pyiwfm.runner.executables.

Covers: _get_cache_dir, find_or_download cache/download branches,
download_from_github Linux/error paths, place_executables DLL copying,
verify_executables budget/zbudget paths.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pyiwfm.runner.executables import (
    IWFMExecutableManager,
    _get_cache_dir,
)

# ======================================================================
# _get_cache_dir
# ======================================================================


class TestGetCacheDir:
    def test_returns_pyiwfm_bin_path(self) -> None:
        result = _get_cache_dir()
        assert result == Path.home() / ".pyiwfm" / "bin"

    def test_is_absolute(self) -> None:
        assert _get_cache_dir().is_absolute()


# ======================================================================
# find_or_download: cache branch
# ======================================================================


class TestFindOrDownloadCache:
    @patch("pyiwfm.runner.executables.find_iwfm_executables")
    def test_uses_cache_when_local_not_found(self, mock_find: MagicMock, tmp_path: Path) -> None:
        """When local discovery fails but cache has exes, returns cached."""
        local_exes = MagicMock()
        local_exes.preprocessor = None
        local_exes.simulation = None

        cached_exes = MagicMock()
        cached_exes.preprocessor = Path("/cached/preproc")
        cached_exes.simulation = Path("/cached/sim")

        # First call: local (no exes). Second call: cache (found).
        mock_find.side_effect = [local_exes, cached_exes]

        mgr = IWFMExecutableManager(version="1.0.0")
        cache_dir = _get_cache_dir() / "1.0.0"
        cache_dir.mkdir(parents=True, exist_ok=True)

        result = mgr.find_or_download()
        assert result is cached_exes

    @patch("pyiwfm.runner.executables.find_iwfm_executables")
    def test_downloads_when_cache_missing(self, mock_find: MagicMock, tmp_path: Path) -> None:
        """When local and cache fail, calls download_from_github."""
        no_exes = MagicMock()
        no_exes.preprocessor = None
        no_exes.simulation = None
        mock_find.return_value = no_exes

        mgr = IWFMExecutableManager(version="99.99.99")

        with patch.object(mgr, "download_from_github") as mock_dl:
            dl_exes = MagicMock()
            mock_dl.return_value = dl_exes
            result = mgr.find_or_download()

        assert result is dl_exes
        mock_dl.assert_called_once()


# ======================================================================
# download_from_github: additional paths
# ======================================================================


class TestDownloadFromGithubPaths:
    @patch("pyiwfm.runner.executables._detect_platform", return_value="Linux")
    @patch("pyiwfm.runner.executables.find_iwfm_executables")
    @patch("pyiwfm.runner.executables.urllib.request.urlretrieve")
    @patch("pyiwfm.runner.executables.tarfile.open")
    def test_linux_tar_gz_extract_and_chmod(
        self,
        mock_tarfile: MagicMock,
        mock_urlretrieve: MagicMock,
        mock_find: MagicMock,
        mock_platform: MagicMock,
        tmp_path: Path,
    ) -> None:
        """On Linux: extracts .tar.gz and chmods executables."""
        # Create mock executable files
        exe_file = tmp_path / "Simulation"
        exe_file.touch()
        lib_file = tmp_path / "lib.so"
        lib_file.touch()

        mock_tf = MagicMock()
        mock_tarfile.return_value.__enter__ = MagicMock(return_value=mock_tf)
        mock_tarfile.return_value.__exit__ = MagicMock(return_value=False)

        exes = MagicMock()
        exes.preprocessor = exe_file
        exes.simulation = exe_file
        mock_find.return_value = exes

        mgr = IWFMExecutableManager(version="2025.0.1747")
        result = mgr.download_from_github(dest=tmp_path)

        assert result is exes
        mock_tarfile.assert_called_once()

    @patch("pyiwfm.runner.executables._detect_platform", return_value="Windows")
    @patch("pyiwfm.runner.executables.find_iwfm_executables")
    @patch("pyiwfm.runner.executables.urllib.request.urlretrieve")
    @patch("pyiwfm.runner.executables.zipfile.ZipFile")
    def test_missing_preprocessor_raises(
        self,
        mock_zipfile: MagicMock,
        mock_urlretrieve: MagicMock,
        mock_find: MagicMock,
        mock_platform: MagicMock,
        tmp_path: Path,
    ) -> None:
        """RuntimeError when preprocessor not found after extraction."""
        mock_zf = MagicMock()
        mock_zipfile.return_value.__enter__ = MagicMock(return_value=mock_zf)
        mock_zipfile.return_value.__exit__ = MagicMock(return_value=False)

        no_exes = MagicMock()
        no_exes.preprocessor = None
        no_exes.simulation = None
        mock_find.return_value = no_exes

        mgr = IWFMExecutableManager(version="2025.0.1747")
        with pytest.raises(RuntimeError, match="PreProcessor not found"):
            mgr.download_from_github(dest=tmp_path)

    @patch("pyiwfm.runner.executables._detect_platform", return_value="Windows")
    @patch("pyiwfm.runner.executables.find_iwfm_executables")
    @patch("pyiwfm.runner.executables.urllib.request.urlretrieve")
    @patch("pyiwfm.runner.executables.zipfile.ZipFile")
    def test_missing_simulation_raises(
        self,
        mock_zipfile: MagicMock,
        mock_urlretrieve: MagicMock,
        mock_find: MagicMock,
        mock_platform: MagicMock,
        tmp_path: Path,
    ) -> None:
        """RuntimeError when simulation not found after extraction."""
        mock_zf = MagicMock()
        mock_zipfile.return_value.__enter__ = MagicMock(return_value=mock_zf)
        mock_zipfile.return_value.__exit__ = MagicMock(return_value=False)

        exes = MagicMock()
        exes.preprocessor = Path("/some/preproc")
        exes.simulation = None
        mock_find.return_value = exes

        mgr = IWFMExecutableManager(version="2025.0.1747")
        with pytest.raises(RuntimeError, match="Simulation not found"):
            mgr.download_from_github(dest=tmp_path)

    @patch("pyiwfm.runner.executables._detect_platform", return_value="Windows")
    @patch("pyiwfm.runner.executables.find_iwfm_executables")
    @patch("pyiwfm.runner.executables.urllib.request.urlretrieve")
    @patch("pyiwfm.runner.executables.zipfile.ZipFile")
    def test_searches_subdirectories(
        self,
        mock_zipfile: MagicMock,
        mock_urlretrieve: MagicMock,
        mock_find: MagicMock,
        mock_platform: MagicMock,
        tmp_path: Path,
    ) -> None:
        """When root has no exes but subdir does, searches subdirs."""
        mock_zf = MagicMock()
        mock_zipfile.return_value.__enter__ = MagicMock(return_value=mock_zf)
        mock_zipfile.return_value.__exit__ = MagicMock(return_value=False)

        # Create a subdirectory
        subdir = tmp_path / "IWFM-Release"
        subdir.mkdir()

        no_exes = MagicMock()
        no_exes.preprocessor = None
        no_exes.simulation = None

        found_exes = MagicMock()
        found_exes.preprocessor = Path("/sub/preproc")
        found_exes.simulation = Path("/sub/sim")

        # First find (root) fails, second (subdirs) succeeds
        mock_find.side_effect = [no_exes, found_exes]

        mgr = IWFMExecutableManager(version="2025.0.1747")
        result = mgr.download_from_github(dest=tmp_path)
        assert result is found_exes

    def test_default_dest_when_none(self) -> None:
        """When dest=None, defaults to cache directory."""
        mgr = IWFMExecutableManager(version="1.2.3")
        with patch.object(mgr, "download_from_github", wraps=mgr.download_from_github):
            with patch("pyiwfm.runner.executables._detect_platform", return_value="Windows"):
                with patch("pyiwfm.runner.executables.urllib.request.urlretrieve") as mock_dl:
                    mock_dl.side_effect = OSError("skip actual download")
                    with pytest.raises(RuntimeError):
                        mgr.download_from_github(dest=None)


# ======================================================================
# place_executables: DLL/SO copying
# ======================================================================


class TestPlaceExecutablesDLLs:
    def test_copies_dll_files(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """DLLs next to preprocessor are copied to model_dir."""
        monkeypatch.setattr("pyiwfm.runner.executables.sys.platform", "win32")

        src_dir = tmp_path / "src"
        src_dir.mkdir()
        preproc = src_dir / "PreProcessor.exe"
        preproc.touch()
        sim = src_dir / "Simulation.exe"
        sim.touch()
        dll = src_dir / "libiomp5md.dll"
        dll.write_text("fake dll")

        exes = MagicMock()
        exes.preprocessor = preproc
        exes.simulation = sim
        exes.simulation_parallel = None
        exes.budget = None
        exes.zbudget = None

        model_dir = tmp_path / "model"
        mgr = IWFMExecutableManager()
        mgr.place_executables(exes, model_dir)

        assert (model_dir / "libiomp5md.dll").exists()

    def test_skips_existing_library(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Does not overwrite existing DLL in target."""
        monkeypatch.setattr("pyiwfm.runner.executables.sys.platform", "win32")

        src_dir = tmp_path / "src"
        src_dir.mkdir()
        preproc = src_dir / "PreProcessor.exe"
        preproc.touch()
        sim = src_dir / "Simulation.exe"
        sim.touch()
        dll = src_dir / "existing.dll"
        dll.write_text("source dll")

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        existing = model_dir / "existing.dll"
        existing.write_text("original dll")

        exes = MagicMock()
        exes.preprocessor = preproc
        exes.simulation = sim
        exes.simulation_parallel = None
        exes.budget = None
        exes.zbudget = None

        mgr = IWFMExecutableManager()
        mgr.place_executables(exes, model_dir)

        # Original should be preserved
        assert existing.read_text() == "original dll"

    def test_copies_linux_so_files(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """On Linux, .so files are copied."""
        monkeypatch.setattr("pyiwfm.runner.executables.sys.platform", "linux")

        src_dir = tmp_path / "src"
        src_dir.mkdir()
        preproc = src_dir / "PreProcessor"
        preproc.touch()
        sim = src_dir / "Simulation"
        sim.touch()
        so_file = src_dir / "libgfortran.so"
        so_file.write_text("fake so")

        exes = MagicMock()
        exes.preprocessor = preproc
        exes.simulation = sim
        exes.simulation_parallel = None
        exes.budget = None
        exes.zbudget = None

        model_dir = tmp_path / "model"
        mgr = IWFMExecutableManager()
        mgr.place_executables(exes, model_dir)

        assert (model_dir / "libgfortran.so").exists()

    def test_places_all_optional_executables(self, tmp_path: Path) -> None:
        """Copies budget, zbudget, simulation_parallel when they exist."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()

        exes = MagicMock()
        names = {
            "preprocessor": "PreProcessor.exe",
            "simulation": "Simulation.exe",
            "simulation_parallel": "Simulation_P.exe",
            "budget": "Budget.exe",
            "zbudget": "ZBudget.exe",
        }
        for attr, fname in names.items():
            p = src_dir / fname
            p.touch()
            setattr(exes, attr, p)

        model_dir = tmp_path / "model"
        mgr = IWFMExecutableManager()
        placed = mgr.place_executables(exes, model_dir)

        assert len(placed) == 5
        for attr in names:
            assert attr in placed


# ======================================================================
# verify_executables: additional branches
# ======================================================================


class TestVerifyExecutablesDeep:
    @patch("pyiwfm.runner.executables.subprocess.run")
    def test_budget_and_zbudget_verified(self, mock_run: MagicMock) -> None:
        """Budget and zbudget executables are verified when present."""
        mock_run.return_value = MagicMock(returncode=0)
        exes = MagicMock()
        exes.preprocessor = Path("/bin/preproc")
        exes.simulation = Path("/bin/sim")
        exes.budget = Path("/bin/budget")
        exes.zbudget = Path("/bin/zbudget")

        mgr = IWFMExecutableManager()
        results = mgr.verify_executables(exes)

        assert results["budget"] is True
        assert results["zbudget"] is True
        assert mock_run.call_count == 4

    @patch("pyiwfm.runner.executables.subprocess.run")
    def test_none_executable_marked_false(self, mock_run: MagicMock) -> None:
        """None executables are marked False without calling subprocess."""
        exes = MagicMock()
        exes.preprocessor = None
        exes.simulation = None
        exes.budget = None
        exes.zbudget = None

        mgr = IWFMExecutableManager()
        results = mgr.verify_executables(exes)

        assert all(v is False for v in results.values())
        mock_run.assert_not_called()


# ======================================================================
# IWFMExecutableManager.__post_init__ edge cases
# ======================================================================


class TestPostInitEdges:
    def test_only_repo_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Only version is empty, so env var is used for version only."""
        monkeypatch.setenv("IWFM_VERSION", "5.5.5")
        mgr = IWFMExecutableManager(github_repo="explicit/repo")
        assert mgr.github_repo == "explicit/repo"
        assert mgr.version == "5.5.5"

    def test_only_version_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Only repo is empty, so env var is used for repo only."""
        monkeypatch.setenv("IWFM_GITHUB_REPO", "env-org/env-repo")
        mgr = IWFMExecutableManager(version="1.0.0")
        assert mgr.github_repo == "env-org/env-repo"
        assert mgr.version == "1.0.0"
