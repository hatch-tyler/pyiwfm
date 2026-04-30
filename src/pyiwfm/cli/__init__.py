"""
pyiwfm command-line interface.

Usage:
    pyiwfm viewer [options]     Launch the interactive web viewer
    pyiwfm export [options]     Export model data to VTK/GeoPackage
    python -m pyiwfm <command>  Same as above
"""

from __future__ import annotations

import argparse
import sys

from pyiwfm.core.exceptions import FileFormatError, PyIWFMError


def _format_user_error(exc: BaseException) -> str:
    """Format an expected exception into a one-line user-facing message.

    For ``FileFormatError`` we surface the optional line-number context.
    Other ``PyIWFMError`` subclasses and OS-level errors collapse to
    ``error: <ExceptionType>: <message>``.
    """
    if isinstance(exc, FileFormatError) and exc.line_number is not None:
        return f"error: FileFormatError (line {exc.line_number}): {exc}"
    return f"error: {exc.__class__.__name__}: {exc}"


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
    from pyiwfm.cli.depletion import add_depletion_parser
    from pyiwfm.cli.drawdown import add_drawdown_parser
    from pyiwfm.cli.export import add_export_parser
    from pyiwfm.cli.iwfm2obs import add_iwfm2obs_parser
    from pyiwfm.cli.package import add_package_parser
    from pyiwfm.cli.pest import add_pest_parser
    from pyiwfm.cli.run import add_run_parser
    from pyiwfm.cli.viewer import add_viewer_parser
    from pyiwfm.cli.zbudget import add_zbudget_parser

    add_viewer_parser(subparsers)
    add_export_parser(subparsers)
    add_budget_parser(subparsers)
    add_zbudget_parser(subparsers)
    add_iwfm2obs_parser(subparsers)
    add_calctyphyd_parser(subparsers)
    add_depletion_parser(subparsers)
    add_drawdown_parser(subparsers)
    add_package_parser(subparsers)
    add_run_parser(subparsers)
    add_pest_parser(subparsers)

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    # Top-level exception handler: turn known user-facing errors into
    # clean stderr lines + exit code 1, instead of leaking a Python
    # traceback. ``KeyboardInterrupt`` propagates so users can ^C
    # cleanly. Truly unexpected exceptions also propagate so
    # programmer bugs surface with full tracebacks.
    try:
        result: int = args.func(args)
    except (PyIWFMError, FileNotFoundError, PermissionError, IsADirectoryError) as exc:
        print(_format_user_error(exc), file=sys.stderr)
        return 1
    return result


__all__ = ["main"]
