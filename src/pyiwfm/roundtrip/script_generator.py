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
) -> list[Path]:
    """Generate run scripts (.bat or .sh) for an IWFM model.

    Creates three scripts in the model directory:
    - run_preprocessor (.bat/.sh)
    - run_simulation (.bat/.sh)
    - run_all (.bat/.sh)

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

    Returns
    -------
    list[Path]
        Paths to the generated scripts.
    """
    scripts: list[Path] = []

    if sys.platform == "win32":
        scripts.extend(
            _generate_bat_scripts(
                model_dir,
                preprocessor_main,
                simulation_main,
                preprocessor_exe,
                simulation_exe,
            )
        )
    else:
        scripts.extend(
            _generate_sh_scripts(
                model_dir,
                preprocessor_main,
                simulation_main,
                preprocessor_exe,
                simulation_exe,
            )
        )

    return scripts


def _generate_bat_scripts(
    model_dir: Path,
    pp_main: str,
    sim_main: str,
    pp_exe: str,
    sim_exe: str,
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

    # Combined script
    all_script = model_dir / "run_all.bat"
    all_script.write_text(
        "@echo off\r\n"
        "echo ========================================\r\n"
        "echo IWFM Full Model Run\r\n"
        "echo ========================================\r\n"
        'call "%~dp0run_preprocessor.bat"\r\n'
        "if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%\r\n"
        'call "%~dp0run_simulation.bat"\r\n'
        "if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%\r\n"
        "echo ========================================\r\n"
        "echo All runs completed successfully.\r\n"
        "echo ========================================\r\n",
        encoding="utf-8",
    )
    scripts.append(all_script)

    return scripts


def _generate_sh_scripts(
    model_dir: Path,
    pp_main: str,
    sim_main: str,
    pp_exe: str,
    sim_exe: str,
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

    # Preprocessor script
    pp_dir = (model_dir / pp_main_path).parent
    pp_file = pp_main_path.name
    pp_script = model_dir / "run_preprocessor.sh"
    pp_script.write_text(
        f"#!/bin/bash\n"
        f"set -e\n"
        f'SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"\n'
        f"echo 'Running IWFM PreProcessor...'\n"
        f'cd "$SCRIPT_DIR/{pp_dir.relative_to(model_dir)}"\n'
        f'echo "{pp_file}" | "$SCRIPT_DIR/{pp_exe}"\n'
        f"echo 'PreProcessor completed successfully.'\n",
        encoding="utf-8",
    )
    pp_script.chmod(pp_script.stat().st_mode | 0o755)
    scripts.append(pp_script)

    # Simulation script
    sim_dir = (model_dir / sim_main_path).parent
    sim_file = sim_main_path.name
    sim_script = model_dir / "run_simulation.sh"
    sim_script.write_text(
        f"#!/bin/bash\n"
        f"set -e\n"
        f'SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"\n'
        f"echo 'Running IWFM Simulation...'\n"
        f'cd "$SCRIPT_DIR/{sim_dir.relative_to(model_dir)}"\n'
        f'echo "{sim_file}" | "$SCRIPT_DIR/{sim_exe}"\n'
        f"echo 'Simulation completed successfully.'\n",
        encoding="utf-8",
    )
    sim_script.chmod(sim_script.stat().st_mode | 0o755)
    scripts.append(sim_script)

    # Combined script
    all_script = model_dir / "run_all.sh"
    all_script.write_text(
        "#!/bin/bash\n"
        "set -e\n"
        'SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"\n'
        "echo '========================================'\n"
        "echo 'IWFM Full Model Run'\n"
        "echo '========================================'\n"
        '"$SCRIPT_DIR/run_preprocessor.sh"\n'
        '"$SCRIPT_DIR/run_simulation.sh"\n'
        "echo '========================================'\n"
        "echo 'All runs completed successfully.'\n"
        "echo '========================================'\n",
        encoding="utf-8",
    )
    all_script.chmod(all_script.stat().st_mode | 0o755)
    scripts.append(all_script)

    return scripts
