"""Tests for PowerShell .ps1 and budget/zbudget script generation.

Covers:
- _generate_ps1_scripts(): creates PS1 files with correct content
- Budget/ZBudget scripts for all three formats (bat, ps1, sh)
- generate_run_scripts() with formats parameter
- generate_run_scripts() with budget_exe / zbudget_exe
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from pyiwfm.roundtrip.script_generator import (
    _generate_bat_scripts,
    _generate_ps1_scripts,
    _generate_sh_scripts,
    generate_run_scripts,
)

# ---------------------------------------------------------------------------
# _generate_ps1_scripts - basic generation
# ---------------------------------------------------------------------------


class TestGeneratePs1Scripts:
    def test_creates_three_ps1_files(self, tmp_path: Path) -> None:
        scripts = _generate_ps1_scripts(
            tmp_path, "PP.in", "Sim.in", "PP_x64.exe", "Sim_x64.exe"
        )
        assert len(scripts) == 3
        for s in scripts:
            assert s.exists()
            assert s.suffix == ".ps1"

    def test_preprocessor_content(self, tmp_path: Path) -> None:
        scripts = _generate_ps1_scripts(
            tmp_path, "PP.in", "Sim.in", "PP_x64.exe", "Sim_x64.exe"
        )
        pp_script = [s for s in scripts if "preprocessor" in s.stem.lower()][0]
        content = pp_script.read_text()
        assert "PP_x64.exe" in content
        assert "PP.in" in content
        assert "$ErrorActionPreference" in content
        assert "Push-Location" in content
        assert "Pop-Location" in content

    def test_simulation_content(self, tmp_path: Path) -> None:
        scripts = _generate_ps1_scripts(
            tmp_path, "PP.in", "Sim.in", "PP_x64.exe", "Sim_x64.exe"
        )
        sim_script = [
            s for s in scripts if "simulation" in s.stem.lower() and "all" not in s.stem.lower()
        ][0]
        content = sim_script.read_text()
        assert "Sim_x64.exe" in content
        assert "Sim.in" in content
        assert "$LASTEXITCODE" in content

    def test_run_all_content(self, tmp_path: Path) -> None:
        scripts = _generate_ps1_scripts(
            tmp_path, "PP.in", "Sim.in", "PP_x64.exe", "Sim_x64.exe"
        )
        all_script = [s for s in scripts if "all" in s.stem.lower()][0]
        content = all_script.read_text()
        assert "run_preprocessor.ps1" in content
        assert "run_simulation.ps1" in content

    def test_with_subdirectory_paths(self, tmp_path: Path) -> None:
        (tmp_path / "Preprocessor").mkdir()
        (tmp_path / "Simulation").mkdir()
        scripts = _generate_ps1_scripts(
            tmp_path,
            "Preprocessor/PP.in",
            "Simulation/Sim.in",
            "PP_x64.exe",
            "Sim_x64.exe",
        )
        pp_script = [s for s in scripts if "preprocessor" in s.stem.lower()][0]
        content = pp_script.read_text()
        assert "Preprocessor" in content


# ---------------------------------------------------------------------------
# Budget / ZBudget scripts — PS1
# ---------------------------------------------------------------------------


class TestPs1BudgetScripts:
    def test_no_budget_scripts_by_default(self, tmp_path: Path) -> None:
        scripts = _generate_ps1_scripts(
            tmp_path, "PP.in", "Sim.in", "PP_x64.exe", "Sim_x64.exe"
        )
        names = {s.stem for s in scripts}
        assert "run_budget" not in names
        assert "run_zbudget" not in names

    def test_budget_script_created(self, tmp_path: Path) -> None:
        scripts = _generate_ps1_scripts(
            tmp_path,
            "PP.in",
            "Sim.in",
            "PP_x64.exe",
            "Sim_x64.exe",
            budget_exe="Budget_x64.exe",
        )
        budget_scripts = [s for s in scripts if s.stem == "run_budget"]
        assert len(budget_scripts) == 1
        content = budget_scripts[0].read_text()
        assert "Budget_x64.exe" in content
        assert "Budget post-processor" in content

    def test_zbudget_script_created(self, tmp_path: Path) -> None:
        scripts = _generate_ps1_scripts(
            tmp_path,
            "PP.in",
            "Sim.in",
            "PP_x64.exe",
            "Sim_x64.exe",
            zbudget_exe="ZBudget_x64.exe",
        )
        zbudget_scripts = [s for s in scripts if s.stem == "run_zbudget"]
        assert len(zbudget_scripts) == 1
        content = zbudget_scripts[0].read_text()
        assert "ZBudget_x64.exe" in content

    def test_both_budget_scripts(self, tmp_path: Path) -> None:
        scripts = _generate_ps1_scripts(
            tmp_path,
            "PP.in",
            "Sim.in",
            "PP_x64.exe",
            "Sim_x64.exe",
            budget_exe="Budget_x64.exe",
            zbudget_exe="ZBudget_x64.exe",
        )
        # 3 base + 2 budget = 5
        assert len(scripts) == 5

    def test_run_all_includes_budget(self, tmp_path: Path) -> None:
        scripts = _generate_ps1_scripts(
            tmp_path,
            "PP.in",
            "Sim.in",
            "PP_x64.exe",
            "Sim_x64.exe",
            budget_exe="Budget_x64.exe",
            zbudget_exe="ZBudget_x64.exe",
        )
        all_script = [s for s in scripts if s.stem == "run_all"][0]
        content = all_script.read_text()
        assert "run_budget.ps1" in content
        assert "run_zbudget.ps1" in content


# ---------------------------------------------------------------------------
# Budget / ZBudget scripts — BAT
# ---------------------------------------------------------------------------


class TestBatBudgetScripts:
    def test_no_budget_by_default(self, tmp_path: Path) -> None:
        scripts = _generate_bat_scripts(
            tmp_path, "PP.in", "Sim.in", "PP_x64.exe", "Sim_x64.exe"
        )
        names = {s.stem for s in scripts}
        assert "run_budget" not in names

    def test_budget_script_created(self, tmp_path: Path) -> None:
        scripts = _generate_bat_scripts(
            tmp_path,
            "PP.in",
            "Sim.in",
            "PP_x64.exe",
            "Sim_x64.exe",
            budget_exe="Budget_x64.exe",
        )
        budget_scripts = [s for s in scripts if s.stem == "run_budget"]
        assert len(budget_scripts) == 1
        content = budget_scripts[0].read_text()
        assert "Budget_x64.exe" in content

    def test_zbudget_script_created(self, tmp_path: Path) -> None:
        scripts = _generate_bat_scripts(
            tmp_path,
            "PP.in",
            "Sim.in",
            "PP_x64.exe",
            "Sim_x64.exe",
            zbudget_exe="ZBudget_x64.exe",
        )
        zbudget_scripts = [s for s in scripts if s.stem == "run_zbudget"]
        assert len(zbudget_scripts) == 1

    def test_run_all_includes_budget(self, tmp_path: Path) -> None:
        scripts = _generate_bat_scripts(
            tmp_path,
            "PP.in",
            "Sim.in",
            "PP_x64.exe",
            "Sim_x64.exe",
            budget_exe="Budget_x64.exe",
            zbudget_exe="ZBudget_x64.exe",
        )
        all_script = [s for s in scripts if s.stem == "run_all"][0]
        content = all_script.read_text()
        assert "run_budget.bat" in content
        assert "run_zbudget.bat" in content

    def test_five_scripts_with_both(self, tmp_path: Path) -> None:
        scripts = _generate_bat_scripts(
            tmp_path,
            "PP.in",
            "Sim.in",
            "PP_x64.exe",
            "Sim_x64.exe",
            budget_exe="Budget_x64.exe",
            zbudget_exe="ZBudget_x64.exe",
        )
        assert len(scripts) == 5


# ---------------------------------------------------------------------------
# Budget / ZBudget scripts — SH
# ---------------------------------------------------------------------------


class TestShBudgetScripts:
    def test_budget_script_created(self, tmp_path: Path) -> None:
        scripts = _generate_sh_scripts(
            tmp_path,
            "PP.in",
            "Sim.in",
            "PP_x64.exe",
            "Sim_x64.exe",
            budget_exe="Budget_x64.exe",
        )
        budget_scripts = [s for s in scripts if s.stem == "run_budget"]
        assert len(budget_scripts) == 1
        content = budget_scripts[0].read_text()
        assert "Budget" in content
        # .exe should be stripped
        assert ".exe" not in content

    def test_zbudget_script_created(self, tmp_path: Path) -> None:
        scripts = _generate_sh_scripts(
            tmp_path,
            "PP.in",
            "Sim.in",
            "PP_x64.exe",
            "Sim_x64.exe",
            zbudget_exe="ZBudget_x64.exe",
        )
        zbudget_scripts = [s for s in scripts if s.stem == "run_zbudget"]
        assert len(zbudget_scripts) == 1

    def test_run_all_includes_budget(self, tmp_path: Path) -> None:
        scripts = _generate_sh_scripts(
            tmp_path,
            "PP.in",
            "Sim.in",
            "PP_x64.exe",
            "Sim_x64.exe",
            budget_exe="Budget_x64.exe",
            zbudget_exe="ZBudget_x64.exe",
        )
        all_script = [s for s in scripts if s.stem == "run_all"][0]
        content = all_script.read_text()
        assert "run_budget.sh" in content
        assert "run_zbudget.sh" in content

    @pytest.mark.skipif(sys.platform == "win32", reason="chmod not meaningful on Windows")
    def test_budget_scripts_executable(self, tmp_path: Path) -> None:
        import stat

        scripts = _generate_sh_scripts(
            tmp_path,
            "PP.in",
            "Sim.in",
            "PP_x64.exe",
            "Sim_x64.exe",
            budget_exe="Budget_x64.exe",
            zbudget_exe="ZBudget_x64.exe",
        )
        for s in scripts:
            mode = s.stat().st_mode
            assert mode & stat.S_IXUSR


# ---------------------------------------------------------------------------
# generate_run_scripts - formats parameter
# ---------------------------------------------------------------------------


class TestFormatsParameter:
    def test_explicit_bat(self, tmp_path: Path) -> None:
        scripts = generate_run_scripts(
            tmp_path, "PP.in", "Sim.in", formats=["bat"]
        )
        assert all(s.suffix == ".bat" for s in scripts)
        assert len(scripts) == 3

    def test_explicit_ps1(self, tmp_path: Path) -> None:
        scripts = generate_run_scripts(
            tmp_path, "PP.in", "Sim.in", formats=["ps1"]
        )
        assert all(s.suffix == ".ps1" for s in scripts)
        assert len(scripts) == 3

    def test_explicit_sh(self, tmp_path: Path) -> None:
        scripts = generate_run_scripts(
            tmp_path, "PP.in", "Sim.in", formats=["sh"]
        )
        assert all(s.suffix == ".sh" for s in scripts)
        assert len(scripts) == 3

    def test_multiple_formats(self, tmp_path: Path) -> None:
        scripts = generate_run_scripts(
            tmp_path, "PP.in", "Sim.in", formats=["bat", "ps1", "sh"]
        )
        bat = [s for s in scripts if s.suffix == ".bat"]
        ps1 = [s for s in scripts if s.suffix == ".ps1"]
        sh = [s for s in scripts if s.suffix == ".sh"]
        assert len(bat) == 3
        assert len(ps1) == 3
        assert len(sh) == 3

    def test_all_formats_with_budget(self, tmp_path: Path) -> None:
        scripts = generate_run_scripts(
            tmp_path,
            "PP.in",
            "Sim.in",
            budget_exe="Budget_x64.exe",
            zbudget_exe="ZBudget_x64.exe",
            formats=["bat", "ps1", "sh"],
        )
        # 5 scripts per format × 3 formats = 15
        assert len(scripts) == 15

    def test_invalid_format_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Unknown script format"):
            generate_run_scripts(
                tmp_path, "PP.in", "Sim.in", formats=["cmd"]
            )

    def test_budget_via_public_api(self, tmp_path: Path) -> None:
        scripts = generate_run_scripts(
            tmp_path,
            "PP.in",
            "Sim.in",
            budget_exe="Budget_x64.exe",
            formats=["bat"],
        )
        names = {s.stem for s in scripts}
        assert "run_budget" in names
        # 4 scripts: pp, sim, budget, all
        assert len(scripts) == 4
