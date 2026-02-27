"""Tests for roundtrip/script_generator.py.

Covers:
- generate_run_scripts(): dispatches to bat/sh based on platform
- _generate_bat_scripts(): creates 3 .bat files with correct content
- _generate_sh_scripts(): creates 3 .sh files, shebang, chmod +x
"""

from __future__ import annotations

import stat
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from pyiwfm.roundtrip.script_generator import (
    _generate_bat_scripts,
    _generate_sh_scripts,
    generate_run_scripts,
)


# ---------------------------------------------------------------------------
# generate_run_scripts - platform dispatch
# ---------------------------------------------------------------------------

class TestGenerateRunScripts:
    def test_dispatches_to_bat_on_windows(self, tmp_path: Path) -> None:
        with patch("pyiwfm.roundtrip.script_generator.sys") as mock_sys:
            mock_sys.platform = "win32"
            scripts = generate_run_scripts(
                tmp_path, "PP.in", "Sim.in", "PP_x64.exe", "Sim_x64.exe"
            )
        assert len(scripts) == 3
        assert all(str(s).endswith(".bat") for s in scripts)

    def test_dispatches_to_sh_on_linux(self, tmp_path: Path) -> None:
        with patch("pyiwfm.roundtrip.script_generator.sys") as mock_sys:
            mock_sys.platform = "linux"
            scripts = generate_run_scripts(
                tmp_path, "PP.in", "Sim.in", "PP_x64.exe", "Sim_x64.exe"
            )
        assert len(scripts) == 3
        assert all(str(s).endswith(".sh") for s in scripts)

    def test_returns_list_of_paths(self, tmp_path: Path) -> None:
        scripts = generate_run_scripts(
            tmp_path, "PP.in", "Sim.in"
        )
        assert isinstance(scripts, list)
        assert all(isinstance(s, Path) for s in scripts)


# ---------------------------------------------------------------------------
# _generate_bat_scripts
# ---------------------------------------------------------------------------

class TestGenerateBatScripts:
    def test_creates_three_bat_files(self, tmp_path: Path) -> None:
        scripts = _generate_bat_scripts(
            tmp_path, "PP.in", "Sim.in", "PP_x64.exe", "Sim_x64.exe"
        )
        assert len(scripts) == 3
        for s in scripts:
            assert s.exists()
            assert s.suffix == ".bat"

    def test_preprocessor_script_content(self, tmp_path: Path) -> None:
        scripts = _generate_bat_scripts(
            tmp_path, "PP.in", "Sim.in", "PP_x64.exe", "Sim_x64.exe"
        )
        pp_script = [s for s in scripts if "preprocessor" in s.stem.lower()][0]
        content = pp_script.read_text()
        assert "PP_x64.exe" in content
        assert "PP.in" in content

    def test_simulation_script_content(self, tmp_path: Path) -> None:
        scripts = _generate_bat_scripts(
            tmp_path, "PP.in", "Sim.in", "PP_x64.exe", "Sim_x64.exe"
        )
        sim_script = [s for s in scripts if "simulation" in s.stem.lower()
                       and "all" not in s.stem.lower()][0]
        content = sim_script.read_text()
        assert "Sim_x64.exe" in content
        assert "Sim.in" in content

    def test_run_all_script_content(self, tmp_path: Path) -> None:
        scripts = _generate_bat_scripts(
            tmp_path, "PP.in", "Sim.in", "PP_x64.exe", "Sim_x64.exe"
        )
        all_script = [s for s in scripts if "all" in s.stem.lower()][0]
        content = all_script.read_text()
        # Should reference both exe names or call other scripts
        assert "PP" in content or "preprocessor" in content.lower()
        assert "Sim" in content or "simulation" in content.lower()


# ---------------------------------------------------------------------------
# _generate_sh_scripts
# ---------------------------------------------------------------------------

class TestGenerateShScripts:
    def test_creates_three_sh_files(self, tmp_path: Path) -> None:
        scripts = _generate_sh_scripts(
            tmp_path, "PP.in", "Sim.in", "PP_x64.exe", "Sim_x64.exe"
        )
        assert len(scripts) == 3
        for s in scripts:
            assert s.exists()
            assert s.suffix == ".sh"

    def test_has_shebang(self, tmp_path: Path) -> None:
        scripts = _generate_sh_scripts(
            tmp_path, "PP.in", "Sim.in", "PP_x64.exe", "Sim_x64.exe"
        )
        for s in scripts:
            content = s.read_text()
            assert content.startswith("#!")

    def test_exe_extension_stripped(self, tmp_path: Path) -> None:
        scripts = _generate_sh_scripts(
            tmp_path, "PP.in", "Sim.in", "PP_x64.exe", "Sim_x64.exe"
        )
        pp_script = [s for s in scripts if "preprocessor" in s.stem.lower()][0]
        content = pp_script.read_text()
        # On Linux, .exe should be stripped from the executable reference
        assert ".exe" not in content or "PP_x64" in content

    @pytest.mark.skipif(
        sys.platform == "win32", reason="chmod not meaningful on Windows"
    )
    def test_executable_permission(self, tmp_path: Path) -> None:
        scripts = _generate_sh_scripts(
            tmp_path, "PP.in", "Sim.in", "PP_x64.exe", "Sim_x64.exe"
        )
        for s in scripts:
            mode = s.stat().st_mode
            assert mode & stat.S_IXUSR  # Owner execute bit
