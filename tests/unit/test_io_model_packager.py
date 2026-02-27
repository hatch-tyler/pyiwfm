"""Tests for io/model_packager.py.

Covers:
- collect_model_files(): recursive collection with exclusion rules
- package_model(): ZIP creation with manifest
- ModelPackageResult: result dataclass fields
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from pyiwfm.io.model_packager import (
    ModelPackageResult,
    collect_model_files,
    package_model,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _create_model_tree(root: Path) -> None:
    """Create a minimal model directory tree for testing."""
    (root / "Preprocessor").mkdir()
    (root / "Simulation").mkdir()
    (root / "Results").mkdir()
    (root / "__pycache__").mkdir()
    (root / ".git").mkdir()

    # Input files
    (root / "Preprocessor" / "PP_MAIN.in").write_text("PP main")
    (root / "Preprocessor" / "Nodes.dat").write_text("1 100.0 200.0")
    (root / "Simulation" / "Simulation.in").write_text("Sim main")
    (root / "Simulation" / "GW.dat").write_text("gw data")

    # Output / excluded files
    (root / "Results" / "GW_Heads.hdf").write_bytes(b"\x00" * 100)
    (root / "__pycache__" / "mod.cpython-312.pyc").write_bytes(b"\x00")

    # Executable
    (root / "PreProcessor_x64.exe").write_bytes(b"\x00" * 50)

    # A shapefile companion
    (root / "Preprocessor" / "boundary.shp").write_bytes(b"\x00")
    (root / "Preprocessor" / "boundary.dbf").write_bytes(b"\x00")


# ---------------------------------------------------------------------------
# collect_model_files
# ---------------------------------------------------------------------------


class TestCollectModelFiles:
    def test_collects_input_files(self, tmp_path: Path) -> None:
        _create_model_tree(tmp_path)
        files = collect_model_files(tmp_path)
        names = {f.name for f in files}
        assert "PP_MAIN.in" in names
        assert "Nodes.dat" in names
        assert "Simulation.in" in names
        assert "GW.dat" in names

    def test_excludes_results_by_default(self, tmp_path: Path) -> None:
        _create_model_tree(tmp_path)
        files = collect_model_files(tmp_path)
        names = {f.name for f in files}
        assert "GW_Heads.hdf" not in names

    def test_excludes_pycache(self, tmp_path: Path) -> None:
        _create_model_tree(tmp_path)
        files = collect_model_files(tmp_path)
        assert not any("__pycache__" in str(f) for f in files)

    def test_excludes_git(self, tmp_path: Path) -> None:
        _create_model_tree(tmp_path)
        files = collect_model_files(tmp_path)
        assert not any(".git" in str(f) for f in files)

    def test_excludes_executables_by_default(self, tmp_path: Path) -> None:
        _create_model_tree(tmp_path)
        files = collect_model_files(tmp_path)
        names = {f.name for f in files}
        assert "PreProcessor_x64.exe" not in names

    def test_include_executables(self, tmp_path: Path) -> None:
        _create_model_tree(tmp_path)
        files = collect_model_files(tmp_path, include_executables=True)
        names = {f.name for f in files}
        assert "PreProcessor_x64.exe" in names

    def test_include_results(self, tmp_path: Path) -> None:
        _create_model_tree(tmp_path)
        files = collect_model_files(tmp_path, include_results=True)
        names = {f.name for f in files}
        assert "GW_Heads.hdf" in names

    def test_returns_sorted_list(self, tmp_path: Path) -> None:
        _create_model_tree(tmp_path)
        files = collect_model_files(tmp_path)
        assert files == sorted(files)

    def test_gis_files_included(self, tmp_path: Path) -> None:
        _create_model_tree(tmp_path)
        files = collect_model_files(tmp_path)
        names = {f.name for f in files}
        assert "boundary.shp" in names
        assert "boundary.dbf" in names

    def test_empty_dir(self, tmp_path: Path) -> None:
        files = collect_model_files(tmp_path)
        assert files == []


# ---------------------------------------------------------------------------
# package_model
# ---------------------------------------------------------------------------


class TestPackageModel:
    def test_creates_zip(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        _create_model_tree(model_dir)
        result = package_model(model_dir)
        assert result.archive_path.exists()
        assert result.archive_path.suffix == ".zip"

    def test_default_output_path(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "MyModel"
        model_dir.mkdir()
        _create_model_tree(model_dir)
        result = package_model(model_dir)
        assert result.archive_path.name == "MyModel.zip"
        assert result.archive_path.parent == tmp_path

    def test_custom_output_path(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        _create_model_tree(model_dir)
        out = tmp_path / "custom" / "archive.zip"
        result = package_model(model_dir, output_path=out)
        assert result.archive_path == out
        assert out.exists()

    def test_zip_contains_files(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        _create_model_tree(model_dir)
        result = package_model(model_dir)

        with zipfile.ZipFile(result.archive_path) as zf:
            names = zf.namelist()
        assert "Preprocessor/PP_MAIN.in" in names or "Preprocessor\\PP_MAIN.in" in names
        assert "manifest.json" in names

    def test_zip_contains_manifest(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        _create_model_tree(model_dir)
        result = package_model(model_dir)

        with zipfile.ZipFile(result.archive_path) as zf:
            manifest = json.loads(zf.read("manifest.json"))
        assert isinstance(manifest, dict)
        assert len(manifest) > 0

    def test_result_fields(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        _create_model_tree(model_dir)
        result = package_model(model_dir)

        assert isinstance(result, ModelPackageResult)
        assert len(result.files_included) > 0
        assert result.total_size_bytes > 0
        assert len(result.manifest) > 0

    def test_excludes_executables_by_default(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        _create_model_tree(model_dir)
        result = package_model(model_dir)

        with zipfile.ZipFile(result.archive_path) as zf:
            names = zf.namelist()
        assert not any("PreProcessor_x64.exe" in n for n in names)

    def test_include_executables(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        _create_model_tree(model_dir)
        result = package_model(model_dir, include_executables=True)

        with zipfile.ZipFile(result.archive_path) as zf:
            names = zf.namelist()
        assert any("PreProcessor_x64.exe" in n for n in names)

    def test_not_found_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            package_model(tmp_path / "nonexistent")

    def test_preserves_directory_structure(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        _create_model_tree(model_dir)
        result = package_model(model_dir)

        with zipfile.ZipFile(result.archive_path) as zf:
            names = zf.namelist()
        # Check subdirectory structure is preserved
        pp_files = [n for n in names if n.startswith("Preprocessor")]
        sim_files = [n for n in names if n.startswith("Simulation")]
        assert len(pp_files) > 0
        assert len(sim_files) > 0
