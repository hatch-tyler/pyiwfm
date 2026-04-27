"""Shared argparse helpers for ``pyiwfm`` CLI subcommands."""

from __future__ import annotations

import argparse
from collections.abc import Callable


def add_control_file_subcommand(
    subparsers: argparse._SubParsersAction,  # type: ignore[type-arg]
    *,
    name: str,
    help: str,  # noqa: A002 — argparse uses ``help`` everywhere
    file_help: str,
    runner: Callable[[argparse.Namespace], int],
) -> None:
    """Register a ``<name> <control_file> [--output-dir DIR]`` subcommand.

    Both ``pyiwfm budget`` and ``pyiwfm zbudget`` share this exact shape
    (positional control file + ``--output-dir`` override). Use this helper
    when adding similar control-file-driven subcommands; for anything with
    additional arguments, register the parser directly.
    """
    p = subparsers.add_parser(name, help=help)
    p.add_argument("control_file", type=str, help=file_help)
    p.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    p.set_defaults(func=runner)
