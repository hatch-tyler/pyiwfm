"""
Observation data upload and retrieval API routes.
"""

from __future__ import annotations

import csv
import io
import logging
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, File, HTTPException, Query, UploadFile

from pyiwfm.visualization.webapi.config import model_state

logger = logging.getLogger(__name__)


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
    """Upload an observation file (CSV or SMP format).

    For CSV files, specify which columns contain dates, values, and optionally
    location identifiers. For SMP files, parsing is automatic.
    """
    if not model_state.is_loaded:
        raise HTTPException(status_code=404, detail="No model loaded")

    content = await file.read()
    filename = file.filename or "upload.csv"
    obs_type = type if type in ("gw", "stream", "subsidence", "hdiff") else "gw"

    # SMP format detection
    if filename.lower().endswith(".smp"):
        return _handle_smp_upload(content, filename, obs_type)

    # CSV format
    return _handle_csv_upload(content, filename, obs_type, date_col, value_col, location_col)


def _handle_smp_upload(content: bytes, filename: str, obs_type: str) -> dict[str, Any]:
    """Parse an uploaded SMP file and create one observation per bore."""
    import re

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

    observations = []
    n_records_total = 0

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
                "obs_id": obs_id,
                "bore_id": bore_id,
                "n_records": len(times),
            }
        )
        n_records_total += len(times)

    if not observations:
        raise HTTPException(status_code=400, detail="No valid data found in SMP file")

    return {
        "n_bores": len(observations),
        "n_records_total": n_records_total,
        "filename": filename,
        "observations": observations,
    }


def _handle_csv_upload(
    content: bytes,
    filename: str,
    obs_type: str,
    date_col: int,
    value_col: int,
    location_col: int,
) -> dict[str, Any]:
    """Parse an uploaded CSV file using specified column indices."""
    text = content.decode("utf-8", errors="replace")
    reader = csv.reader(io.StringIO(text))
    header_skipped = False

    # Collect rows grouped by location
    groups: dict[str, tuple[list[str], list[float]]] = {}
    default_key = Path(filename).stem

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
            dt_str = row[date_col].strip()
            val = float(row[value_col].strip())

            # Try multiple datetime formats
            dt = None
            for fmt in [
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %H:%M",
                "%Y-%m-%d",
                "%m/%d/%Y %H:%M:%S",
                "%m/%d/%Y %H:%M",
                "%m/%d/%Y",
                "%Y-%m-%dT%H:%M:%S",
            ]:
                try:
                    dt = datetime.strptime(dt_str, fmt)
                    break
                except ValueError:
                    continue

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
            detail="No valid data found. Expected CSV with datetime and value columns",
        )

    # Single location (no location column or single group)
    if location_col < 0 or len(groups) == 1:
        loc_key = next(iter(groups))
        times, values = groups[loc_key]

        obs_id = str(uuid.uuid4())[:8]
        model_state.add_observation(
            obs_id,
            {
                "filename": filename,
                "location_id": None,
                "type": obs_type,
                "n_records": len(times),
                "times": times,
                "values": values,
                "units": "",
            },
        )
        return {
            "observation_id": obs_id,
            "n_records": len(times),
            "filename": filename,
            "start_time": times[0] if times else None,
            "end_time": times[-1] if times else None,
        }

    # Multiple locations
    observations = []
    n_records_total = 0
    for loc_key, (times, values) in groups.items():
        if not times:
            continue
        obs_id = str(uuid.uuid4())[:8]
        model_state.add_observation(
            obs_id,
            {
                "filename": filename,
                "display_name": loc_key,
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
                "obs_id": obs_id,
                "location_name": loc_key,
                "n_records": len(times),
            }
        )
        n_records_total += len(times)

    return {
        "n_locations": len(observations),
        "n_records_total": n_records_total,
        "filename": filename,
        "observations": observations,
    }


@router.post("/scan-directory")
async def scan_directory_endpoint(
    directory: str = Query(..., description="Path to scan for observation files"),
    load: bool = Query(default=False, description="If true, load all found files"),
    recursive: bool = Query(default=False, description="Scan subdirectories"),
) -> dict[str, Any]:
    """Scan a directory for loadable observation files (.smp, .csv)."""
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
