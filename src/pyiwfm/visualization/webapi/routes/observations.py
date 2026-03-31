"""
Observation data upload and retrieval API routes.
"""

from __future__ import annotations

import csv
import io
import logging
import re
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, File, HTTPException, Query, UploadFile

from pyiwfm.visualization.webapi.config import model_state

logger = logging.getLogger(__name__)

# Date formats supported for delimited file parsing
_DATE_FORMATS = [
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M",
    "%m/%d/%Y",
    "%Y-%m-%dT%H:%M:%S",
]

# Regex for SMP-style date: M/DD/YYYY or MM/DD/YYYY
_SMP_DATE_RE = re.compile(r"\d{1,2}/\d{1,2}/\d{4}")


def _validate_safe_path(user_path: str | Path) -> Path:
    """Resolve a user-supplied path and reject obvious traversal attempts.

    Prevents directory traversal by resolving the path and rejecting
    paths that contain ``..`` components after resolution.  Also rejects
    paths pointing to system-critical directories.

    Parameters
    ----------
    user_path : str or Path
        User-supplied file or directory path.

    Returns
    -------
    Path
        Resolved absolute path.

    Raises
    ------
    HTTPException
        If the path appears to be a traversal attempt.
    """
    resolved = Path(user_path).resolve()
    # Reject if the original path contained '..' (even if resolve() cleaned it)
    if ".." in Path(user_path).parts:
        raise HTTPException(status_code=403, detail="Path traversal not allowed")
    return resolved


def _detect_delimiter(text: str) -> str:
    """Detect the delimiter used in a text file by sampling the first lines.

    Returns ``','``, ``'\\t'``, or ``'whitespace'``.
    """
    sample_lines = [line for line in text.split("\n")[:10] if line.strip()]
    if not sample_lines:
        return ","

    tab_counts = [line.count("\t") for line in sample_lines]
    comma_counts = [line.count(",") for line in sample_lines]

    avg_tabs = sum(tab_counts) / len(tab_counts)
    avg_commas = sum(comma_counts) / len(comma_counts)

    if avg_tabs >= 1 and avg_tabs >= avg_commas:
        return "\t"
    if avg_commas >= 1:
        return ","
    return "whitespace"


def _looks_like_smp(text: str) -> bool:
    """Check whether text content looks like SMP format.

    SMP files have no header row and each line contains a bore ID followed
    by a date in ``M/DD/YYYY`` or ``MM/DD/YYYY`` format, a time, and a value.
    """
    lines = [line for line in text.split("\n")[:5] if line.strip()]
    if not lines:
        return False

    matches = 0
    for line in lines:
        parts = line.split()
        if len(parts) < 4:
            continue
        # Check that a date-like field appears somewhere in positions 1-3
        if any(_SMP_DATE_RE.fullmatch(parts[i]) for i in range(1, min(4, len(parts)))):
            matches += 1

    return matches >= len(lines) * 0.8


def _parse_datetime(dt_str: str) -> datetime | None:
    """Try parsing a datetime string with multiple formats."""
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(dt_str.strip(), fmt)
        except ValueError:
            continue
    return None


def _build_upload_response(
    observations: list[dict[str, Any]],
    unmatched_locations: list[str] | None = None,
) -> dict[str, Any]:
    """Build a normalized upload response matching the frontend UploadResult type."""
    total_records = sum(o["n_records"] for o in observations)
    return {
        "n_observations": len(observations),
        "n_records": total_records,
        "observations": observations,
        "unmatched_locations": unmatched_locations or [],
    }


def _split_row(line: str, delimiter: str) -> list[str]:
    """Split a line by the detected delimiter."""
    if delimiter == "whitespace":
        return line.split()
    if delimiter == "\t":
        return [c.strip() for c in line.split("\t")]
    # For comma, use csv.reader for proper quote handling
    return next(csv.reader(io.StringIO(line)))


router = APIRouter(prefix="/api/observations", tags=["observations"])


@router.post("/upload")
async def upload_observation(
    file: UploadFile = File(...),  # noqa: B008
    type: str = Query(default="gw", description="Observation type: gw, stream, or subsidence"),
    date_col: int = Query(default=0, description="0-indexed column for datetime"),
    value_col: int = Query(default=1, description="0-indexed column for values"),
    location_col: int = Query(
        default=-1, description="0-indexed column for location ID, -1 = none"
    ),
) -> dict[str, Any]:
    """Upload an observation file (delimited text or SMP format).

    Supports comma, tab, and whitespace-delimited files. For SMP files
    (detected by extension or content), parsing is automatic. For other
    delimited formats, specify which columns contain dates, values, and
    optionally location identifiers.
    """
    if not model_state.is_loaded:
        raise HTTPException(status_code=404, detail="No model loaded")

    content = await file.read()
    filename = file.filename or "upload.txt"
    obs_type = type if type in ("gw", "stream", "subsidence", "hdiff") else "gw"

    text = content.decode("utf-8", errors="replace")

    # SMP format detection: by extension or by content sniffing
    if filename.lower().endswith(".smp") or _looks_like_smp(text):
        return _handle_smp_upload(content, filename, obs_type)

    # Delimited text format (CSV, TSV, or whitespace)
    return _handle_delimited_upload(text, filename, obs_type, date_col, value_col, location_col)


def _handle_smp_upload(content: bytes, filename: str, obs_type: str) -> dict[str, Any]:
    """Parse an uploaded SMP file and create one observation per bore."""
    from pyiwfm.io.smp import SMPReader

    # SMPReader expects a file path, write to temp file
    with tempfile.NamedTemporaryFile(suffix=".smp", delete=False, mode="wb") as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        reader = SMPReader(tmp_path)
        data = reader.read()
    finally:
        tmp_path.unlink(missing_ok=True)

    observations: list[dict[str, Any]] = []

    for bore_id, ts in data.items():
        mask = ts.valid_mask
        if not mask.any():
            continue

        times_arr = ts.times[mask]
        values_arr = ts.values[mask]

        times = [str(t.astype("datetime64[s]")).replace("T", " ") for t in times_arr]
        values = [float(v) for v in values_arr]

        display_name = re.sub(r"%[1-4]$", "", bore_id)

        obs_id = str(uuid.uuid4())[:8]
        model_state.add_observation(
            obs_id,
            {
                "filename": filename,
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
        observations.append(
            {
                "observation_id": obs_id,
                "filename": filename,
                "n_records": len(times),
                "location_id": None,
                "start_time": times[0] if times else None,
                "end_time": times[-1] if times else None,
            }
        )

    if not observations:
        raise HTTPException(status_code=400, detail="No valid data found in SMP file")

    return _build_upload_response(observations)


def _handle_delimited_upload(
    text: str,
    filename: str,
    obs_type: str,
    date_col: int,
    value_col: int,
    location_col: int,
) -> dict[str, Any]:
    """Parse an uploaded delimited text file using specified column indices.

    Automatically detects the delimiter (comma, tab, or whitespace).
    """
    delimiter = _detect_delimiter(text)
    header_skipped = False

    # Collect rows grouped by location
    groups: dict[str, tuple[list[str], list[float]]] = {}
    default_key = Path(filename).stem

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        row = _split_row(line, delimiter)

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
            dt_str = row[date_col].strip()
            val = float(row[value_col].strip())

            dt = _parse_datetime(dt_str)
            if dt is not None:
                loc_key = row[location_col].strip() if location_col >= 0 else default_key
                if loc_key not in groups:
                    groups[loc_key] = ([], [])
                groups[loc_key][0].append(dt.isoformat())
                groups[loc_key][1].append(val)
        except (ValueError, IndexError):
            continue

    if not groups or all(not t for t, _ in groups.values()):
        raise HTTPException(
            status_code=400,
            detail="No valid data found. Expected a delimited text file with datetime "
            "and value columns.",
        )

    observations: list[dict[str, Any]] = []
    for loc_key, (times, values) in groups.items():
        if not times:
            continue
        obs_id = str(uuid.uuid4())[:8]
        display_name = loc_key if location_col >= 0 else None
        model_state.add_observation(
            obs_id,
            {
                "filename": filename,
                "display_name": display_name or loc_key,
                "location_id": None,
                "type": obs_type,
                "n_records": len(times),
                "times": times,
                "values": values,
                "units": "",
            },
        )
        observations.append(
            {
                "observation_id": obs_id,
                "filename": filename,
                "n_records": len(times),
                "location_id": None,
                "start_time": times[0] if times else None,
                "end_time": times[-1] if times else None,
            }
        )

    return _build_upload_response(observations)


@router.post("/scan-directory")
async def scan_directory_endpoint(
    directory: str = Query(..., description="Path to scan for observation files"),
    load: bool = Query(default=False, description="If true, load all found files"),
    recursive: bool = Query(default=False, description="Scan subdirectories"),
) -> dict[str, Any]:
    """Scan a directory for loadable observation files."""
    if not model_state.is_loaded:
        raise HTTPException(status_code=404, detail="No model loaded")

    from pyiwfm.visualization.webapi.services.observation_loader import (
        scan_directory,
    )

    dir_path = _validate_safe_path(directory)
    if not dir_path.is_dir():
        raise HTTPException(status_code=400, detail=f"Directory not found: {directory}")

    files = scan_directory(dir_path, recursive=recursive)

    if not load:
        return {"files": files}

    # Load all discovered files
    from pyiwfm.visualization.webapi.services.observation_loader import (
        _load_single_file,
    )

    loaded = []
    for info in files:
        fp = Path(info["path"])
        obs_type = info["type_guess"]
        ids = _load_single_file(fp, obs_type, model_state)
        loaded.append(
            {
                "path": info["path"],
                "format": info["format"],
                "type": obs_type,
                "n_observations": len(ids),
            }
        )

    return {"files": files, "loaded": loaded}


@router.post("/load-files")
async def load_files_endpoint(
    files: list[dict[str, Any]] = Body(..., description="List of {path, type} to load"),  # noqa: B008
) -> dict[str, Any]:
    """Load specific observation files by path and type."""
    if not model_state.is_loaded:
        raise HTTPException(status_code=404, detail="No model loaded")

    from pyiwfm.visualization.webapi.services.observation_loader import (
        _load_single_file,
    )

    results: list[dict[str, object]] = []
    for entry in files:
        fp = _validate_safe_path(entry["path"])
        obs_type = entry.get("type", "gw")

        if not fp.is_file():
            results.append({"path": str(fp), "error": "File not found"})
            continue

        ids = _load_single_file(fp, obs_type, model_state)
        results.append(
            {
                "path": str(fp),
                "type": obs_type,
                "n_observations": len(ids),
                "observation_ids": ids,
            }
        )

    return {"results": results}


@router.get("")
def list_observations() -> list[dict[str, Any]]:
    """List all uploaded observations."""
    return model_state.list_observations()


@router.get("/{obs_id}/data")
def get_observation_data(obs_id: str) -> dict[str, Any]:
    """Get observation time series data."""
    obs = model_state.get_observation(obs_id)
    if obs is None:
        raise HTTPException(status_code=404, detail=f"Observation '{obs_id}' not found")

    return {
        "times": obs["times"],
        "values": obs["values"],
        "units": obs.get("units", ""),
    }


@router.put("/{obs_id}/location")
def set_observation_location(
    obs_id: str,
    location_id: int,
    location_type: str = "gw",
) -> dict[str, Any]:
    """Associate an observation with a hydrograph location."""
    obs = model_state.get_observation(obs_id)
    if obs is None:
        raise HTTPException(status_code=404, detail=f"Observation '{obs_id}' not found")

    obs["location_id"] = location_id
    obs["type"] = location_type
    return {"status": "ok", "observation_id": obs_id, "location_id": location_id}


@router.delete("/{obs_id}")
def delete_observation(obs_id: str) -> dict[str, Any]:
    """Delete an observation."""
    if model_state.delete_observation(obs_id):
        return {"status": "deleted", "observation_id": obs_id}
    raise HTTPException(status_code=404, detail=f"Observation '{obs_id}' not found")
