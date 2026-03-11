"""
Shared observation loading logic for the web viewer.

Used by both the API upload/scan endpoints and the CLI ``--observations`` flag.
"""

from __future__ import annotations

import csv
import io
import logging
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pyiwfm.visualization.webapi.config import ModelState

logger = logging.getLogger(__name__)

# Date formats supported for CSV parsing
_DATE_FORMATS = [
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M",
    "%m/%d/%Y",
    "%Y-%m-%dT%H:%M:%S",
]

# Filename patterns for guessing observation type
_TYPE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(str|stream|flow)", re.IGNORECASE), "stream"),
    (re.compile(r"(sub|insar|ext)", re.IGNORECASE), "subsidence"),
    (re.compile(r"(hdiff|head_diff)", re.IGNORECASE), "hdiff"),
    (re.compile(r"(gw|head|level)", re.IGNORECASE), "gw"),
]

_LOADABLE_EXTENSIONS = {".smp", ".csv"}


def guess_obs_type(filename: str) -> str:
    """Guess observation type from filename patterns.

    Returns one of: ``"gw"``, ``"stream"``, ``"subsidence"``, ``"hdiff"``.
    """
    stem = Path(filename).stem
    for pattern, obs_type in _TYPE_PATTERNS:
        if pattern.search(stem):
            return obs_type
    return "gw"


def scan_directory(directory: Path, recursive: bool = False) -> list[dict[str, Any]]:
    """Find loadable observation files in a directory.

    Parameters
    ----------
    directory : Path
        Directory to scan.
    recursive : bool
        If True, scan subdirectories as well.

    Returns
    -------
    list[dict]
        List of ``{path, format, type_guess, size_kb}`` dicts.
    """
    results: list[dict[str, Any]] = []

    if recursive:
        walker: Any = os.walk(directory)
    else:
        walker = [(str(directory), [], os.listdir(directory))]

    for dirpath, _dirs, filenames in walker:
        for fname in filenames:
            ext = Path(fname).suffix.lower()
            if ext not in _LOADABLE_EXTENSIONS:
                continue
            full = Path(dirpath) / fname
            if not full.is_file():
                continue
            results.append(
                {
                    "path": str(full),
                    "format": ext.lstrip("."),
                    "type_guess": guess_obs_type(fname),
                    "size_kb": round(full.stat().st_size / 1024, 1),
                }
            )

    return results


def _parse_datetime(dt_str: str) -> datetime | None:
    """Try parsing a datetime string with multiple formats."""
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(dt_str.strip(), fmt)
        except ValueError:
            continue
    return None


def load_smp_file(
    path: Path,
    obs_type: str,
    state: ModelState,
) -> list[str]:
    """Load a single SMP file into model state.

    Creates one observation per bore ID found in the file.

    Returns list of observation IDs created.
    """
    from pyiwfm.io.smp import SMPReader

    reader = SMPReader(path)
    data = reader.read()
    obs_ids: list[str] = []

    for bore_id, ts in data.items():
        mask = ts.valid_mask
        if not mask.any():
            continue

        times_arr = ts.times[mask]
        values_arr = ts.values[mask]

        times = [str(t.astype("datetime64[s]")).replace("T", " ") for t in times_arr]
        values = [float(v) for v in values_arr]

        # Strip layer suffix for display name
        display_name = re.sub(r"%[1-4]$", "", bore_id)

        obs_id = str(uuid.uuid4())[:8]
        state.add_observation(
            obs_id,
            {
                "filename": path.name,
                "bore_id": bore_id,
                "display_name": display_name,
                "location_id": None,
                "type": obs_type,
                "n_records": len(times),
                "times": times,
                "values": values,
                "units": "",
            },
        )
        obs_ids.append(obs_id)

    logger.info("Loaded %d bores from %s", len(obs_ids), path.name)
    return obs_ids


def load_csv_file(
    path: Path,
    obs_type: str,
    state: ModelState,
    date_col: int = 0,
    value_col: int = 1,
    location_col: int = -1,
) -> list[str]:
    """Load a single CSV file into model state.

    When ``location_col >= 0``, groups rows by location value and creates
    one observation per unique location. Otherwise creates a single observation.

    Returns list of observation IDs created.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    reader = csv.reader(io.StringIO(text))
    header_skipped = False

    # Collect rows grouped by location
    groups: dict[str, tuple[list[str], list[float]]] = {}
    default_key = path.stem

    for row in reader:
        max_col = max(date_col, value_col, location_col if location_col >= 0 else 0)
        if len(row) <= max_col:
            continue

        # Skip header row
        if not header_skipped:
            try:
                float(row[value_col])
            except ValueError:
                header_skipped = True
                continue
            header_skipped = True

        try:
            dt = _parse_datetime(row[date_col])
            val = float(row[value_col].strip())
        except (ValueError, IndexError):
            continue

        if dt is None:
            continue

        loc_key = row[location_col].strip() if location_col >= 0 else default_key
        if loc_key not in groups:
            groups[loc_key] = ([], [])
        groups[loc_key][0].append(dt.isoformat())
        groups[loc_key][1].append(val)

    obs_ids: list[str] = []
    for loc_key, (times, values) in groups.items():
        if not times:
            continue
        obs_id = str(uuid.uuid4())[:8]
        state.add_observation(
            obs_id,
            {
                "filename": path.name,
                "display_name": loc_key,
                "location_id": None,
                "type": obs_type,
                "n_records": len(times),
                "times": times,
                "values": values,
                "units": "",
            },
        )
        obs_ids.append(obs_id)

    logger.info("Loaded %d locations from %s", len(obs_ids), path.name)
    return obs_ids


def load_observation_paths(
    paths: list[Path],
    state: ModelState,
) -> dict[str, int]:
    """Load observation files/directories into model state.

    For each path:
    - File (``.smp``) -> parse via SMPReader, one obs per bore
    - File (``.csv``) -> parse via CSV reader with auto-detect columns
    - Directory -> scan for ``.smp``/``.csv``, load each

    Returns ``{type: n_observations_loaded}``.
    """
    counts: dict[str, int] = {}

    for p in paths:
        p = Path(p)
        if p.is_dir():
            files = scan_directory(p)
            for info in files:
                fp = Path(info["path"])
                obs_type = info["type_guess"]
                ids = _load_single_file(fp, obs_type, state)
                counts[obs_type] = counts.get(obs_type, 0) + len(ids)
        elif p.is_file():
            obs_type = guess_obs_type(p.name)
            ids = _load_single_file(p, obs_type, state)
            counts[obs_type] = counts.get(obs_type, 0) + len(ids)
        else:
            logger.warning("Path not found: %s", p)

    return counts


def _load_single_file(
    path: Path,
    obs_type: str,
    state: ModelState,
) -> list[str]:
    """Load a single file by extension."""
    ext = path.suffix.lower()
    if ext == ".smp":
        return load_smp_file(path, obs_type, state)
    elif ext == ".csv":
        return load_csv_file(path, obs_type, state)
    else:
        logger.warning("Unsupported file format: %s", path)
        return []
