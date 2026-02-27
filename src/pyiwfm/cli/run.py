"""
CLI subcommand for generating run scripts and optionally downloading executables.

Usage::

    pyiwfm run --model-dir ./model [--download-executables] [--scripts-only]
               [--format bat] [--format ps1] [--format sh]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def add_run_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the ``pyiwfm run`` subcommand."""
    p = subparsers.add_parser(
        "run",
        help="Generate run scripts (and optionally download executables) for an IWFM model",
    )
    p.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Root directory of the IWFM model",
    )
    p.add_argument(
        "--download-executables",
        action="store_true",
        default=False,
        help="Download IWFM executables from GitHub and place them in the model directory",
    )
    p.add_argument(
        "--scripts-only",
        action="store_true",
        default=False,
        help="Only generate run scripts without executing the model",
    )
    p.add_argument(
        "--format",
        action="append",
        dest="formats",
        choices=["bat", "ps1", "sh"],
        help="Script format(s) to generate (repeatable; default: platform-appropriate)",
    )
    p.set_defaults(func=run_run)


def run_run(args: argparse.Namespace) -> int:
    """Execute the run subcommand."""
    from pyiwfm.cli._model_finder import find_model_files
    from pyiwfm.roundtrip.script_generator import generate_run_scripts

    model_dir = Path(args.model_dir)
    if not model_dir.is_dir():
        print(f"Error: model directory not found: {model_dir}", file=sys.stderr)
        return 1

    # Discover model files
    found = find_model_files(model_dir)
    pp_main = found.get("preprocessor_main")
    sim_main = found.get("simulation_main")

    if pp_main is None:
        print("Error: could not find preprocessor main file", file=sys.stderr)
        return 1
    if sim_main is None:
        print("Error: could not find simulation main file", file=sys.stderr)
        return 1

    pp_main_rel = str(pp_main.relative_to(model_dir))
    sim_main_rel = str(sim_main.relative_to(model_dir))

    # Default executable names
    pp_exe = "PreProcessor_x64.exe"
    sim_exe = "Simulation_x64.exe"
    budget_exe: str | None = None
    zbudget_exe: str | None = None

    # Optionally download and place executables
    if args.download_executables:
        from pyiwfm.runner.executables import IWFMExecutableManager

        mgr = IWFMExecutableManager()
        exes = mgr.find_or_download()
        placed = mgr.place_executables(exes, model_dir)

        if "preprocessor" in placed:
            pp_exe = placed["preprocessor"].name
        if "simulation" in placed:
            sim_exe = placed["simulation"].name
        if "budget" in placed:
            budget_exe = placed["budget"].name
        if "zbudget" in placed:
            zbudget_exe = placed["zbudget"].name

        print(f"Placed executables: {list(placed.keys())}")
    else:
        # Check for existing executables in model_dir
        for candidate in model_dir.glob("Budget*"):
            if candidate.is_file() and candidate.suffix.lower() in {".exe", ""}:
                budget_exe = candidate.name
                break
        for candidate in model_dir.glob("ZBudget*"):
            if candidate.is_file() and candidate.suffix.lower() in {".exe", ""}:
                zbudget_exe = candidate.name
                break

    # Generate scripts
    scripts = generate_run_scripts(
        model_dir,
        preprocessor_main=pp_main_rel,
        simulation_main=sim_main_rel,
        preprocessor_exe=pp_exe,
        simulation_exe=sim_exe,
        budget_exe=budget_exe,
        zbudget_exe=zbudget_exe,
        formats=args.formats,
    )

    for s in scripts:
        print(f"Generated: {s}")

    if not args.scripts_only and not args.download_executables:
        print(
            "\nHint: use --download-executables to also fetch IWFM binaries, "
            "or --scripts-only to skip this message."
        )

    return 0
