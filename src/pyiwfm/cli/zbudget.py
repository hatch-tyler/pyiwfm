"""
CLI subcommand for IWFM zone budget Excel export.

Usage::

    pyiwfm zbudget <control_file> [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def add_zbudget_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the ``pyiwfm zbudget`` subcommand."""
    p = subparsers.add_parser(
        "zbudget",
        help="Export zone budget data to Excel from a zbudget control file",
    )
    p.add_argument("control_file", type=str, help="ZBudget control/input file")
    p.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    p.set_defaults(func=run_zbudget)


def run_zbudget(args: argparse.Namespace) -> int:
    """Execute zbudget export from a control file."""
    from pyiwfm.io.zbudget_control import read_zbudget_control
    from pyiwfm.io.zbudget_excel import zbudget_control_to_excel

    control_path = Path(args.control_file)
    if not control_path.exists():
        print(f"Error: control file not found: {control_path}", file=sys.stderr)
        return 1

    config = read_zbudget_control(control_path)

    # Override output directory if requested
    if args.output_dir:
        out_dir = Path(args.output_dir)
        for spec in config.zbudgets:
            spec.output_file = out_dir / spec.output_file.name

    created = zbudget_control_to_excel(config)

    if not created:
        print("No zbudget files were generated.", file=sys.stderr)
        return 1

    for p in created:
        print(f"Wrote: {p}")
    return 0
