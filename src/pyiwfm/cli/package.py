"""
CLI subcommand for packaging an IWFM model into a ZIP archive.

Usage::

    pyiwfm package --model-dir ./model [--output model.zip]
                    [--include-executables] [--include-results]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def add_package_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the ``pyiwfm package`` subcommand."""
    p = subparsers.add_parser(
        "package",
        help="Package an IWFM model directory into a ZIP archive",
    )
    p.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Root directory of the IWFM model",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output ZIP file path (default: <model_dir>.zip)",
    )
    p.add_argument(
        "--include-executables",
        action="store_true",
        default=False,
        help="Include .exe/.dll/.so files in the archive",
    )
    p.add_argument(
        "--include-results",
        action="store_true",
        default=False,
        help="Include Results/ directory and HDF5 output files",
    )
    p.set_defaults(func=run_package)


def run_package(args: argparse.Namespace) -> int:
    """Execute model packaging."""
    from pyiwfm.io.model_packager import package_model

    model_dir = Path(args.model_dir)
    if not model_dir.is_dir():
        print(f"Error: model directory not found: {model_dir}", file=sys.stderr)
        return 1

    output_path = Path(args.output) if args.output else None

    result = package_model(
        model_dir,
        output_path=output_path,
        include_executables=args.include_executables,
        include_results=args.include_results,
    )

    n_files = len(result.files_included)
    size_mb = result.total_size_bytes / (1024 * 1024)
    print(f"Packaged {n_files} files ({size_mb:.1f} MB) -> {result.archive_path}")
    return 0
