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
    p = subparsers.add_parser(
        "budget",
        help="Export budget data to Excel from a budget control file",
    )
    p.add_argument("control_file", type=str, help="Budget control/input file (.bud/.in)")
    p.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    p.set_defaults(func=run_budget)


def run_budget(args: argparse.Namespace) -> int:
    """Execute budget export from a control file."""
    from pyiwfm.io.budget_control import read_budget_control
    from pyiwfm.io.budget_excel import budget_control_to_excel

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
