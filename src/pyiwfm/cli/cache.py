"""``pyiwfm cache`` subcommand.

Manage pyiwfm's on-disk caches (the consolidated ``pyiwfm_cache/`` subfolder
plus orphaned in-place caches left over from before the v2.0 refactor).

Usage::

    pyiwfm cache clear --model-dir DIR [--dry-run]
    pyiwfm cache clear --cache-dir DIR  [--dry-run]
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from pyiwfm.io.cache_paths import (
    AREA_CACHE_LABELS,
    CACHE_SUBDIR,
    ENV_VAR,
    SQLITE_CACHE_NAME,
)

logger = logging.getLogger(__name__)

_LEGACY_CACHE_FILENAMES: frozenset[str] = frozenset(
    [SQLITE_CACHE_NAME, *(f"{lbl}_area_cache.hdf" for lbl in AREA_CACHE_LABELS)]
)
_LEGACY_CACHE_SUFFIXES: tuple[str, ...] = (
    ".head_cache.hdf",
    ".hydrograph_cache.hdf",
)


def add_cache_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the ``pyiwfm cache`` subcommand group."""
    p = subparsers.add_parser(
        "cache",
        help="Manage pyiwfm cache files.",
        description="Inspect or remove pyiwfm cache files written by the web viewer.",
    )
    cache_subs = p.add_subparsers(dest="cache_command", help="Cache operations")

    clear = cache_subs.add_parser(
        "clear",
        help="Delete pyiwfm cache files.",
        description=(
            "Remove pyiwfm cache files. Only files matching known cache "
            "filename patterns are deleted, so it is safe to point this at "
            "any directory."
        ),
    )
    clear.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help=(
            "Model directory to scan. Removes pyiwfm_cache/ subfolders and "
            "any orphaned in-place legacy cache files found beneath it."
        ),
    )
    clear.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=(
            f"Explicit cache directory to clear. Takes precedence over --model-dir and ${ENV_VAR}."
        ),
    )
    clear.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be removed without deleting them.",
    )
    clear.set_defaults(func=run_cache_clear)

    p.set_defaults(func=_no_subcommand)


def _no_subcommand(args: argparse.Namespace) -> int:
    print("Usage: pyiwfm cache clear [--model-dir DIR | --cache-dir DIR] [--dry-run]")
    return 1


def run_cache_clear(args: argparse.Namespace) -> int:
    """Execute ``pyiwfm cache clear``."""
    targets = _collect_targets(args.cache_dir, args.model_dir)

    if not targets:
        print("No pyiwfm cache files found.")
        return 0

    total_bytes = 0
    for path in targets:
        try:
            total_bytes += path.stat().st_size
        except OSError:
            pass

    verb = "Would remove" if args.dry_run else "Removing"
    for path in targets:
        print(f"  {verb}: {path}")

    if args.dry_run:
        print(f"{len(targets)} file(s), {_format_bytes(total_bytes)}. (dry-run, no changes)")
        return 0

    removed = 0
    failed = 0
    for path in targets:
        try:
            path.unlink()
            removed += 1
        except OSError as e:
            print(f"ERROR: could not remove {path}: {e}")
            failed += 1

    # Drop empty cache subdirs we just emptied.
    _prune_empty_cache_dirs(targets)

    print(f"Removed {removed} file(s), {_format_bytes(total_bytes)}.")
    if failed:
        print(f"Failed to remove {failed} file(s).")
        return 1
    return 0


def _collect_targets(cache_dir_arg: Path | None, model_dir_arg: Path | None) -> list[Path]:
    """Find every cache file the clear command should consider."""
    found: list[Path] = []
    seen: set[Path] = set()

    def _add(path: Path) -> None:
        try:
            resolved = path.resolve()
        except OSError:
            return
        if resolved in seen or not path.exists() or not path.is_file():
            return
        seen.add(resolved)
        found.append(path)

    explicit_dir = cache_dir_arg
    if explicit_dir is None:
        env_val = os.environ.get(ENV_VAR)
        if env_val:
            explicit_dir = Path(env_val)

    if explicit_dir is not None:
        if explicit_dir.is_dir():
            for entry in sorted(explicit_dir.iterdir()):
                if _is_cache_file(entry):
                    _add(entry)
        return found

    if model_dir_arg is None:
        return found

    if not model_dir_arg.is_dir():
        print(f"WARNING: model directory not found: {model_dir_arg}")
        return found

    for cache_subdir in sorted(model_dir_arg.rglob(CACHE_SUBDIR)):
        if not cache_subdir.is_dir():
            continue
        for entry in sorted(cache_subdir.iterdir()):
            if _is_cache_file(entry):
                _add(entry)

    # Orphaned in-place files left over from before the refactor.
    for entry in sorted(model_dir_arg.rglob("*")):
        if entry.is_file() and _is_cache_file(entry):
            # Skip files already collected from a pyiwfm_cache/ subdir.
            try:
                if entry.parent.name == CACHE_SUBDIR:
                    continue
            except OSError:
                continue
            _add(entry)

    return found


def _is_cache_file(path: Path) -> bool:
    if not path.is_file():
        return False
    name = path.name
    if name in _LEGACY_CACHE_FILENAMES:
        return True
    return any(name.endswith(suffix) for suffix in _LEGACY_CACHE_SUFFIXES)


def _prune_empty_cache_dirs(removed_files: list[Path]) -> None:
    """Best-effort removal of now-empty ``pyiwfm_cache/`` directories."""
    candidate_dirs: set[Path] = set()
    for f in removed_files:
        if f.parent.name == CACHE_SUBDIR:
            candidate_dirs.add(f.parent)
    for d in candidate_dirs:
        try:
            if d.exists() and not any(d.iterdir()):
                d.rmdir()
        except OSError:
            pass


def _format_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024**2:
        return f"{n / 1024:.1f} KB"
    if n < 1024**3:
        return f"{n / 1024**2:.1f} MB"
    return f"{n / 1024**3:.1f} GB"
