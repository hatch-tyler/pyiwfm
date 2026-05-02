"""Resolve the directory where pyiwfm writes its cache files.

All pyiwfm caches (the SQLite ``model_cache.db`` consumed by the webapi,
plus the intermediate ``*.head_cache.hdf`` / ``*.hydrograph_cache.hdf`` /
``*_area_cache.hdf`` files written by the lazy loaders) land in a single
directory resolved by :func:`resolve_cache_dir`.

Resolution precedence (highest first):

1. ``override`` argument (typically the CLI's ``--cache-dir`` flag).
2. ``PYIWFM_CACHE_DIR`` environment variable.
3. ``<results_dir>/pyiwfm_cache/`` when a results directory is known.
4. ``<source_dir>/pyiwfm_cache/`` as a last resort.

The directory is created if it does not exist.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

CACHE_SUBDIR = "pyiwfm_cache"
ENV_VAR = "PYIWFM_CACHE_DIR"

# Filename patterns used by pyiwfm caches.  Listed here so the migration
# helper can find legacy files written next to source data prior to the
# introduction of the consolidated cache directory.
SQLITE_CACHE_NAME = "model_cache.db"
AREA_CACHE_LABELS = ("nonponded", "ponded", "urban", "native")


def resolve_cache_dir(
    results_dir: Path | None,
    source_dir: Path | None = None,
    override: Path | None = None,
) -> Path | None:
    """Return the directory where pyiwfm caches should live.

    Returns ``None`` only when no override, env var, results dir, or source
    dir is available.
    """
    target = _select_target(results_dir, source_dir, override)
    if target is None:
        return None
    target.mkdir(parents=True, exist_ok=True)
    return target


def _select_target(
    results_dir: Path | None,
    source_dir: Path | None,
    override: Path | None,
) -> Path | None:
    if override is not None:
        return Path(override)
    env_val = os.environ.get(ENV_VAR)
    if env_val:
        return Path(env_val)
    if results_dir is not None:
        return Path(results_dir) / CACHE_SUBDIR
    if source_dir is not None:
        return Path(source_dir) / CACHE_SUBDIR
    return None


def migrate_legacy_caches(
    cache_dir: Path,
    legacy_dirs: list[Path],
    head_source_paths: list[Path] | None = None,
    hydrograph_source_paths: list[Path] | None = None,
) -> int:
    """Move pre-refactor cache files into ``cache_dir``.

    ``legacy_dirs`` are the directories where caches used to land
    (typically the model's results directory and source directory).  For
    each known cache filename, the first match found is moved into
    ``cache_dir``; subsequent duplicates are left in place untouched so
    the user can inspect them.

    ``head_source_paths`` and ``hydrograph_source_paths`` are the IWFM
    text/HDF source files whose intermediate caches need migrating.  The
    legacy filename is derived the same way the loaders compute it
    (``head_path.with_suffix('.head_cache.hdf')`` and
    ``path.parent / (path.name + '.hydrograph_cache.hdf')``).

    Returns the number of files moved.
    """
    moved = 0
    seen_targets: set[Path] = set()

    def _move(src: Path, dst: Path) -> None:
        nonlocal moved
        if dst in seen_targets or dst.exists() or not src.exists():
            return
        # Don't move a file onto itself (cache_dir already inside legacy_dir).
        try:
            if src.resolve() == dst.resolve():
                return
        except OSError:
            return
        try:
            shutil.move(str(src), str(dst))
            seen_targets.add(dst)
            moved += 1
            logger.info("Migrated legacy cache: %s -> %s", src, dst)
        except OSError as e:
            logger.warning("Could not migrate %s -> %s: %s", src, dst, e)

    for legacy in legacy_dirs:
        if legacy is None or not legacy.exists():
            continue
        _move(legacy / SQLITE_CACHE_NAME, cache_dir / SQLITE_CACHE_NAME)
        for label in AREA_CACHE_LABELS:
            name = f"{label}_area_cache.hdf"
            _move(legacy / name, cache_dir / name)

    for head_path in head_source_paths or ():
        legacy_name = head_path.with_suffix(".head_cache.hdf").name
        legacy_path = head_path.parent / legacy_name
        _move(legacy_path, cache_dir / legacy_name)

    for hydro_path in hydrograph_source_paths or ():
        legacy_name = hydro_path.name + ".hydrograph_cache.hdf"
        legacy_path = hydro_path.parent / legacy_name
        _move(legacy_path, cache_dir / legacy_name)

    return moved
