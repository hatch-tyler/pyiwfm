"""Unsaturated zone API routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from pyiwfm.visualization.webapi.config import require_model

router = APIRouter(prefix="/api/unsaturated-zone", tags=["unsaturated_zone"])


@router.get("/")
def get_unsaturated_zone() -> dict[str, Any]:
    """Get unsaturated zone component information."""
    model = require_model()
    uz = model.unsaturated_zone
    if uz is None:
        raise HTTPException(status_code=404, detail="Unsaturated zone not loaded")

    return {
        "loaded": True,
        "n_layers": getattr(uz, "n_layers", None),
        "n_elements": getattr(uz, "n_elements", None),
        "n_items": getattr(uz, "n_items", None),
    }


@router.get("/summary")
def get_unsaturated_zone_summary() -> dict[str, Any]:
    """Get summary of unsaturated zone parameters."""
    model = require_model()
    uz = model.unsaturated_zone
    if uz is None:
        raise HTTPException(status_code=404, detail="Unsaturated zone not loaded")

    result: dict[str, Any] = {
        "loaded": True,
        "n_items": getattr(uz, "n_items", None),
    }

    if hasattr(uz, "n_layers"):
        result["n_layers"] = uz.n_layers
    if hasattr(uz, "n_elements"):
        result["n_elements"] = uz.n_elements

    return result
