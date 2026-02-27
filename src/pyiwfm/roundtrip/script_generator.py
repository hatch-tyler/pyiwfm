"""Generate platform-appropriate run scripts for IWFM models."""

from __future__ import annotations

import sys
from pathlib import Path


def generate_run_scripts(
    model_dir: Path,
    preprocessor_main: str,
    simulation_main: str,
    preprocessor_exe: str = "PreProcessor_x64.exe",
    simulation_exe: str = "Simulation_x64.exe",
    budget_exe: str | None = None,
    zbudget_exe: str | None = None,
    formats: list[str] | None = None,
) -> list[Path]:
    """Generate run scripts for an IWFM model.

    Creates scripts in the model directory for each requested format:
    - run_preprocessor (.bat/.ps1/.sh)
    - run_simulation (.bat/.ps1/.sh)
    - run_all (.bat/.ps1/.sh)
    - run_budget (.bat/.ps1/.sh) — if *budget_exe* is provided
    - run_zbudget (.bat/.ps1/.sh) — if *zbudget_exe* is provided

    Parameters
    ----------
    model_dir : Path
        Root directory of the model.
    preprocessor_main : str
        Relative path from model_dir to the preprocessor main file.
    simulation_main : str
        Relative path from model_dir to the simulation main file.
    preprocessor_exe : str
        Name of the preprocessor executable.
    simulation_exe : str
        Name of the simulation executable.
    budget_exe : str | None
        Name of the Budget post-processor executable.  When ``None``
        (default), no budget run script is generated.
    zbudget_exe : str | None
        Name of the ZBudget post-processor executable.  When ``None``
        (default), no zbudget run script is generated.
    formats : list[str] | None
        Script formats to generate.  Accepted values: ``"bat"``,
        ``"ps1"``, ``"sh"``.  ``None`` selects the platform-appropriate
        default (``["bat", "ps1"]`` on Windows, ``["sh"]`` elsewhere).

    Returns
    -------
    list[Path]
        Paths to the generated scripts.
    """
    if formats is None:
        if sys.platform == "win32":
            formats = ["bat", "ps1"]
        else:
            formats = ["sh"]

    scripts: list[Path] = []

    generators = {
        "bat": _generate_bat_scripts,
        "ps1": _generate_ps1_scripts,
        "sh": _generate_sh_scripts,
    }

    for fmt in formats:
        gen = generators.get(fmt)
        if gen is None:
            raise ValueError(f"Unknown script format: {fmt!r} (expected 'bat', 'ps1', or 'sh')")
        scripts.extend(
            gen(
                model_dir,
                preprocessor_main,
                simulation_main,
                preprocessor_exe,
                simulation_exe,
                budget_exe,
                zbudget_exe,
            )
        )

    return scripts


# ---------------------------------------------------------------------------
# .bat scripts (Windows CMD)
# ---------------------------------------------------------------------------


def _generate_bat_scripts(
    model_dir: Path,
    pp_main: str,
    sim_main: str,
    pp_exe: str,
    sim_exe: str,
    budget_exe: str | None = None,
    zbudget_exe: str | None = None,
) -> list[Path]:
    """Generate Windows .bat scripts."""
    scripts: list[Path] = []

    pp_main_path = Path(pp_main)
    sim_main_path = Path(sim_main)

    # Preprocessor script
    pp_dir = (model_dir / pp_main_path).parent
    pp_file = pp_main_path.name
    pp_script = model_dir / "run_preprocessor.bat"
    pp_script.write_text(
        f"@echo off\r\n"
        f"echo Running IWFM PreProcessor...\r\n"
        f'cd /d "%~dp0{pp_dir.relative_to(model_dir)}"\r\n'
        f'echo {pp_file}| "%~dp0{pp_exe}"\r\n'
        f"if %ERRORLEVEL% NEQ 0 (\r\n"
        f"    echo PreProcessor FAILED with error %ERRORLEVEL%\r\n"
        f"    exit /b %ERRORLEVEL%\r\n"
        f")\r\n"
        f"echo PreProcessor completed successfully.\r\n",
        encoding="utf-8",
    )
    scripts.append(pp_script)

    # Simulation script
    sim_dir = (model_dir / sim_main_path).parent
    sim_file = sim_main_path.name
    sim_script = model_dir / "run_simulation.bat"
    sim_script.write_text(
        f"@echo off\r\n"
        f"echo Running IWFM Simulation...\r\n"
        f'cd /d "%~dp0{sim_dir.relative_to(model_dir)}"\r\n'
        f'echo {sim_file}| "%~dp0{sim_exe}"\r\n'
        f"if %ERRORLEVEL% NEQ 0 (\r\n"
        f"    echo Simulation FAILED with error %ERRORLEVEL%\r\n"
        f"    exit /b %ERRORLEVEL%\r\n"
        f")\r\n"
        f"echo Simulation completed successfully.\r\n",
        encoding="utf-8",
    )
    scripts.append(sim_script)

    # Budget script (optional)
    if budget_exe:
        budget_script = model_dir / "run_budget.bat"
        budget_script.write_text(
            f"@echo off\r\n"
            f"echo Running IWFM Budget post-processor...\r\n"
            f'cd /d "%~dp0{sim_dir.relative_to(model_dir)}"\r\n'
            f'echo {sim_file}| "%~dp0{budget_exe}"\r\n'
            f"if %ERRORLEVEL% NEQ 0 (\r\n"
            f"    echo Budget FAILED with error %ERRORLEVEL%\r\n"
            f"    exit /b %ERRORLEVEL%\r\n"
            f")\r\n"
            f"echo Budget completed successfully.\r\n",
            encoding="utf-8",
        )
        scripts.append(budget_script)

    # ZBudget script (optional)
    if zbudget_exe:
        zbudget_script = model_dir / "run_zbudget.bat"
        zbudget_script.write_text(
            f"@echo off\r\n"
            f"echo Running IWFM ZBudget post-processor...\r\n"
            f'cd /d "%~dp0{sim_dir.relative_to(model_dir)}"\r\n'
            f'echo {sim_file}| "%~dp0{zbudget_exe}"\r\n'
            f"if %ERRORLEVEL% NEQ 0 (\r\n"
            f"    echo ZBudget FAILED with error %ERRORLEVEL%\r\n"
            f"    exit /b %ERRORLEVEL%\r\n"
            f")\r\n"
            f"echo ZBudget completed successfully.\r\n",
            encoding="utf-8",
        )
        scripts.append(zbudget_script)

    # Combined script
    all_script = model_dir / "run_all.bat"
    all_lines = [
        "@echo off\r\n",
        "echo ========================================\r\n",
        "echo IWFM Full Model Run\r\n",
        "echo ========================================\r\n",
        'call "%~dp0run_preprocessor.bat"\r\n',
        "if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%\r\n",
        'call "%~dp0run_simulation.bat"\r\n',
        "if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%\r\n",
    ]
    if budget_exe:
        all_lines.append('call "%~dp0run_budget.bat"\r\n')
        all_lines.append("if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%\r\n")
    if zbudget_exe:
        all_lines.append('call "%~dp0run_zbudget.bat"\r\n')
        all_lines.append("if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%\r\n")
    all_lines.extend(
        [
            "echo ========================================\r\n",
            "echo All runs completed successfully.\r\n",
            "echo ========================================\r\n",
        ]
    )
    all_script.write_text("".join(all_lines), encoding="utf-8")
    scripts.append(all_script)

    return scripts


# ---------------------------------------------------------------------------
# .ps1 scripts (PowerShell)
# ---------------------------------------------------------------------------


def _generate_ps1_scripts(
    model_dir: Path,
    pp_main: str,
    sim_main: str,
    pp_exe: str,
    sim_exe: str,
    budget_exe: str | None = None,
    zbudget_exe: str | None = None,
) -> list[Path]:
    """Generate Windows PowerShell .ps1 scripts."""
    scripts: list[Path] = []

    pp_main_path = Path(pp_main)
    sim_main_path = Path(sim_main)

    pp_dir_rel = str((model_dir / pp_main_path).parent.relative_to(model_dir))
    pp_file = pp_main_path.name
    sim_dir_rel = str((model_dir / sim_main_path).parent.relative_to(model_dir))
    sim_file = sim_main_path.name

    # Preprocessor script
    pp_script = model_dir / "run_preprocessor.ps1"
    pp_script.write_text(
        '$ErrorActionPreference = "Stop"\r\n'
        "$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path\r\n"
        'Write-Host "Running IWFM PreProcessor..."\r\n'
        f'Push-Location "$ScriptDir\\{pp_dir_rel}"\r\n'
        "try {{\r\n"
        f'    "{pp_file}" | & "$ScriptDir\\{pp_exe}"\r\n'
        "    if ($LASTEXITCODE -ne 0) {{\r\n"
        '        throw "PreProcessor failed (exit code $LASTEXITCODE)"\r\n'
        "    }}\r\n"
        "}} finally {{\r\n"
        "    Pop-Location\r\n"
        "}}\r\n"
        'Write-Host "PreProcessor completed successfully."\r\n',
        encoding="utf-8",
    )
    scripts.append(pp_script)

    # Simulation script
    sim_script = model_dir / "run_simulation.ps1"
    sim_script.write_text(
        '$ErrorActionPreference = "Stop"\r\n'
        "$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path\r\n"
        'Write-Host "Running IWFM Simulation..."\r\n'
        f'Push-Location "$ScriptDir\\{sim_dir_rel}"\r\n'
        "try {{\r\n"
        f'    "{sim_file}" | & "$ScriptDir\\{sim_exe}"\r\n'
        "    if ($LASTEXITCODE -ne 0) {{\r\n"
        '        throw "Simulation failed (exit code $LASTEXITCODE)"\r\n'
        "    }}\r\n"
        "}} finally {{\r\n"
        "    Pop-Location\r\n"
        "}}\r\n"
        'Write-Host "Simulation completed successfully."\r\n',
        encoding="utf-8",
    )
    scripts.append(sim_script)

    # Budget script (optional)
    if budget_exe:
        budget_script = model_dir / "run_budget.ps1"
        budget_script.write_text(
            '$ErrorActionPreference = "Stop"\r\n'
            "$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path\r\n"
            'Write-Host "Running IWFM Budget post-processor..."\r\n'
            f'Push-Location "$ScriptDir\\{sim_dir_rel}"\r\n'
            "try {{\r\n"
            f'    "{sim_file}" | & "$ScriptDir\\{budget_exe}"\r\n'
            "    if ($LASTEXITCODE -ne 0) {{\r\n"
            '        throw "Budget failed (exit code $LASTEXITCODE)"\r\n'
            "    }}\r\n"
            "}} finally {{\r\n"
            "    Pop-Location\r\n"
            "}}\r\n"
            'Write-Host "Budget completed successfully."\r\n',
            encoding="utf-8",
        )
        scripts.append(budget_script)

    # ZBudget script (optional)
    if zbudget_exe:
        zbudget_script = model_dir / "run_zbudget.ps1"
        zbudget_script.write_text(
            '$ErrorActionPreference = "Stop"\r\n'
            "$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path\r\n"
            'Write-Host "Running IWFM ZBudget post-processor..."\r\n'
            f'Push-Location "$ScriptDir\\{sim_dir_rel}"\r\n'
            "try {{\r\n"
            f'    "{sim_file}" | & "$ScriptDir\\{zbudget_exe}"\r\n'
            "    if ($LASTEXITCODE -ne 0) {{\r\n"
            '        throw "ZBudget failed (exit code $LASTEXITCODE)"\r\n'
            "    }}\r\n"
            "}} finally {{\r\n"
            "    Pop-Location\r\n"
            "}}\r\n"
            'Write-Host "ZBudget completed successfully."\r\n',
            encoding="utf-8",
        )
        scripts.append(zbudget_script)

    # Combined script
    all_script = model_dir / "run_all.ps1"
    all_lines = [
        '$ErrorActionPreference = "Stop"\r\n',
        "$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path\r\n",
        'Write-Host "========================================"\r\n',
        'Write-Host "IWFM Full Model Run"\r\n',
        'Write-Host "========================================"\r\n',
        '& "$ScriptDir\\run_preprocessor.ps1"\r\n',
        '& "$ScriptDir\\run_simulation.ps1"\r\n',
    ]
    if budget_exe:
        all_lines.append('& "$ScriptDir\\run_budget.ps1"\r\n')
    if zbudget_exe:
        all_lines.append('& "$ScriptDir\\run_zbudget.ps1"\r\n')
    all_lines.extend(
        [
            'Write-Host "========================================"\r\n',
            'Write-Host "All runs completed successfully."\r\n',
            'Write-Host "========================================"\r\n',
        ]
    )
    all_script.write_text("".join(all_lines), encoding="utf-8")
    scripts.append(all_script)

    return scripts


# ---------------------------------------------------------------------------
# .sh scripts (Unix/Linux)
# ---------------------------------------------------------------------------


def _generate_sh_scripts(
    model_dir: Path,
    pp_main: str,
    sim_main: str,
    pp_exe: str,
    sim_exe: str,
    budget_exe: str | None = None,
    zbudget_exe: str | None = None,
) -> list[Path]:
    """Generate Unix .sh scripts."""
    scripts: list[Path] = []

    pp_main_path = Path(pp_main)
    sim_main_path = Path(sim_main)

    # Strip .exe suffix for Linux
    if pp_exe.endswith(".exe"):
        pp_exe = pp_exe.replace("_x64.exe", "").replace(".exe", "")
    if sim_exe.endswith(".exe"):
        sim_exe = sim_exe.replace("_x64.exe", "").replace(".exe", "")
    if budget_exe and budget_exe.endswith(".exe"):
        budget_exe = budget_exe.replace("_x64.exe", "").replace(".exe", "")
    if zbudget_exe and zbudget_exe.endswith(".exe"):
        zbudget_exe = zbudget_exe.replace("_x64.exe", "").replace(".exe", "")

    pp_dir_rel = str((model_dir / pp_main_path).parent.relative_to(model_dir))
    pp_file = pp_main_path.name
    sim_dir_rel = str((model_dir / sim_main_path).parent.relative_to(model_dir))
    sim_file = sim_main_path.name

    # Preprocessor script
    pp_script = model_dir / "run_preprocessor.sh"
    pp_script.write_text(
        f"#!/bin/bash\n"
        f"set -e\n"
        f'SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"\n'
        f"echo 'Running IWFM PreProcessor...'\n"
        f'cd "$SCRIPT_DIR/{pp_dir_rel}"\n'
        f'echo "{pp_file}" | "$SCRIPT_DIR/{pp_exe}"\n'
        f"echo 'PreProcessor completed successfully.'\n",
        encoding="utf-8",
    )
    pp_script.chmod(pp_script.stat().st_mode | 0o755)
    scripts.append(pp_script)

    # Simulation script
    sim_script = model_dir / "run_simulation.sh"
    sim_script.write_text(
        f"#!/bin/bash\n"
        f"set -e\n"
        f'SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"\n'
        f"echo 'Running IWFM Simulation...'\n"
        f'cd "$SCRIPT_DIR/{sim_dir_rel}"\n'
        f'echo "{sim_file}" | "$SCRIPT_DIR/{sim_exe}"\n'
        f"echo 'Simulation completed successfully.'\n",
        encoding="utf-8",
    )
    sim_script.chmod(sim_script.stat().st_mode | 0o755)
    scripts.append(sim_script)

    # Budget script (optional)
    if budget_exe:
        budget_script = model_dir / "run_budget.sh"
        budget_script.write_text(
            f"#!/bin/bash\n"
            f"set -e\n"
            f'SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"\n'
            f"echo 'Running IWFM Budget post-processor...'\n"
            f'cd "$SCRIPT_DIR/{sim_dir_rel}"\n'
            f'echo "{sim_file}" | "$SCRIPT_DIR/{budget_exe}"\n'
            f"echo 'Budget completed successfully.'\n",
            encoding="utf-8",
        )
        budget_script.chmod(budget_script.stat().st_mode | 0o755)
        scripts.append(budget_script)

    # ZBudget script (optional)
    if zbudget_exe:
        zbudget_script = model_dir / "run_zbudget.sh"
        zbudget_script.write_text(
            f"#!/bin/bash\n"
            f"set -e\n"
            f'SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"\n'
            f"echo 'Running IWFM ZBudget post-processor...'\n"
            f'cd "$SCRIPT_DIR/{sim_dir_rel}"\n'
            f'echo "{sim_file}" | "$SCRIPT_DIR/{zbudget_exe}"\n'
            f"echo 'ZBudget completed successfully.'\n",
            encoding="utf-8",
        )
        zbudget_script.chmod(zbudget_script.stat().st_mode | 0o755)
        scripts.append(zbudget_script)

    # Combined script
    all_script = model_dir / "run_all.sh"
    all_lines = [
        "#!/bin/bash\n",
        "set -e\n",
        'SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"\n',
        "echo '========================================'\n",
        "echo 'IWFM Full Model Run'\n",
        "echo '========================================'\n",
        '"$SCRIPT_DIR/run_preprocessor.sh"\n',
        '"$SCRIPT_DIR/run_simulation.sh"\n',
    ]
    if budget_exe:
        all_lines.append('"$SCRIPT_DIR/run_budget.sh"\n')
    if zbudget_exe:
        all_lines.append('"$SCRIPT_DIR/run_zbudget.sh"\n')
    all_lines.extend(
        [
            "echo '========================================'\n",
            "echo 'All runs completed successfully.'\n",
            "echo '========================================'\n",
        ]
    )
    all_script.write_text("".join(all_lines), encoding="utf-8")
    all_script.chmod(all_script.stat().st_mode | 0o755)
    scripts.append(all_script)

    return scripts
