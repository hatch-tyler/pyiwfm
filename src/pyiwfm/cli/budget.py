"""
CLI subcommand for IWFM budget Excel export.

Usage::

    pyiwfm budget <control_file> [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def add_budget_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the ``pyiwfm budget`` subcommand."""
    from pyiwfm.cli._parsers import add_control_file_subcommand

    add_control_file_subcommand(
        subparsers,
        name="budget",
        help="Export budget data to Excel from a budget control file",
        file_help="Budget control/input file (.bud/.in)",
        runner=run_budget,
    )


def run_budget(args: argparse.Namespace) -> int:
    """Execute budget export from a control file."""
    from pyiwfm.io.budget.control import read_budget_control
    from pyiwfm.io.budget.excel import budget_control_to_excel

    control_path = Path(args.control_file)
    if not control_path.exists():
        print(f"Error: control file not found: {control_path}", file=sys.stderr)
        return 1

    config = read_budget_control(control_path)

    # Override output directory if requested
    if args.output_dir:
        out_dir = Path(args.output_dir)
        for spec in config.budgets:
            spec.output_file = out_dir / spec.output_file.name

    created = budget_control_to_excel(config)

    if not created:
        print("No budget files were generated.", file=sys.stderr)
        return 1

    for p in created:
        print(f"Wrote: {p}")
    return 0
