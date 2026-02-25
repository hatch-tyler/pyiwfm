"""
pyiwfm command-line interface.

Usage:
    pyiwfm viewer [options]     Launch the interactive web viewer
    pyiwfm export [options]     Export model data to VTK/GeoPackage
    python -m pyiwfm <command>  Same as above
"""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="pyiwfm",
        description="Python tools for IWFM (Integrated Water Flow Model).",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Register subcommands
    from pyiwfm.cli.budget import add_budget_parser
    from pyiwfm.cli.calctyphyd import add_calctyphyd_parser
    from pyiwfm.cli.export import add_export_parser
    from pyiwfm.cli.iwfm2obs import add_iwfm2obs_parser
    from pyiwfm.cli.viewer import add_viewer_parser
    from pyiwfm.cli.zbudget import add_zbudget_parser

    add_viewer_parser(subparsers)
    add_export_parser(subparsers)
    add_budget_parser(subparsers)
    add_zbudget_parser(subparsers)
    add_iwfm2obs_parser(subparsers)
    add_calctyphyd_parser(subparsers)

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    # Dispatch to the subcommand handler
    result: int = args.func(args)
    return result


__all__ = ["main"]
