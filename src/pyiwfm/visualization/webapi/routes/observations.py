"""
Observation data upload and retrieval API routes.
"""

from __future__ import annotations

import csv
import io
import uuid
from datetime import datetime

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from pyiwfm.visualization.webapi.config import model_state

router = APIRouter(prefix="/api/observations", tags=["observations"])


@router.post("/upload")
async def upload_observation(
    file: UploadFile = File(...),  # noqa: B008
    type: str = Query(default="gw", description="Observation type: gw, stream, or subsidence"),
) -> dict:
    """Upload an observation file (CSV: datetime,value)."""
    if not model_state.is_loaded:
        raise HTTPException(status_code=404, detail="No model loaded")

    content = await file.read()
    text = content.decode("utf-8", errors="replace")

    times: list[str] = []
    values: list[float] = []

    # Try CSV parse (datetime, value)
    reader = csv.reader(io.StringIO(text))
    header_skipped = False

    for row in reader:
        if len(row) < 2:
            continue

        # Skip header row
        if not header_skipped:
            try:
                float(row[1])
            except ValueError:
                header_skipped = True
                continue
            header_skipped = True

        try:
            dt_str = row[0].strip()
            val = float(row[1].strip())

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
                times.append(dt.isoformat())
                values.append(val)
        except (ValueError, IndexError):
            continue

    if not times:
        raise HTTPException(
            status_code=400,
            detail="No valid data found. Expected CSV with columns: datetime, value",
        )

    obs_id = str(uuid.uuid4())[:8]
    filename = file.filename or "upload.csv"

    obs_type = type if type in ("gw", "stream", "subsidence") else "gw"

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


@router.get("")
def list_observations() -> list[dict]:
    """List all uploaded observations."""
    return model_state.list_observations()


@router.get("/{obs_id}/data")
def get_observation_data(obs_id: str) -> dict:
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
) -> dict:
    """Associate an observation with a hydrograph location."""
    obs = model_state.get_observation(obs_id)
    if obs is None:
        raise HTTPException(status_code=404, detail=f"Observation '{obs_id}' not found")

    obs["location_id"] = location_id
    obs["type"] = location_type
    return {"status": "ok", "observation_id": obs_id, "location_id": location_id}


@router.delete("/{obs_id}")
def delete_observation(obs_id: str) -> dict:
    """Delete an observation."""
    if model_state.delete_observation(obs_id):
        return {"status": "deleted", "observation_id": obs_id}
    raise HTTPException(status_code=404, detail=f"Observation '{obs_id}' not found")
