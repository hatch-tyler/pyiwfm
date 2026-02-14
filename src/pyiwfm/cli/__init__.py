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
    from pyiwfm.cli.export import add_export_parser
    from pyiwfm.cli.viewer import add_viewer_parser

    add_viewer_parser(subparsers)
    add_export_parser(subparsers)

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    # Dispatch to the subcommand handler
    return args.func(args)


__all__ = ["main"]
