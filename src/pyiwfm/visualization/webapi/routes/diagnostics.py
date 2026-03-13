"""
Diagnostics API routes for simulation messages, convergence, and mass balance.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Query

from pyiwfm.visualization.webapi.config import model_state

if TYPE_CHECKING:
    from pyiwfm.io.simulation_messages import SimulationMessagesResult

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/diagnostics", tags=["diagnostics"])

# Try to import convergence/mass-balance dataclasses (may not exist yet)
_HAS_CONVERGENCE = False
try:
    from pyiwfm.io.simulation_messages import ConvergenceRecord  # noqa: F401

    _HAS_CONVERGENCE = True
except ImportError:
    pass

_HAS_MASS_BALANCE = False
try:
    from pyiwfm.io.simulation_messages import MassBalanceRecord  # noqa: F401

    _HAS_MASS_BALANCE = True
except ImportError:
    pass


def _find_messages_file() -> Path | None:
    """Locate the SimulationMessages.out file in the results directory.

    Returns
    -------
    Path | None
        Path to the file, or ``None`` if not found.
    """
    results_dir = model_state._results_dir
    if results_dir is None or not results_dir.is_dir():
        return None

    # Try several glob patterns
    patterns = ["*SimulationMessages*", "*Messages*.out", "*messages*.out"]
    for pattern in patterns:
        matches = list(results_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def _get_diagnostics_result() -> SimulationMessagesResult | None:
    """Return cached diagnostics result, parsing on first access.

    Returns
    -------
    SimulationMessagesResult | None
        Parsed result, or ``None`` if the file doesn't exist.
    """
    # Return cached result if available
    cached: SimulationMessagesResult | None = getattr(model_state, "_diagnostics_result", None)
    if cached is not None:
        return cached

    messages_file = _find_messages_file()
    if messages_file is None:
        return None

    try:
        from pyiwfm.io.simulation_messages import SimulationMessagesReader

        reader = SimulationMessagesReader(messages_file)
        result = reader.read()
        model_state._diagnostics_result = result  # type: ignore[attr-defined]
        return result
    except Exception:
        logger.exception("Failed to parse SimulationMessages file: %s", messages_file)
        return None


def _runtime_seconds(result: SimulationMessagesResult | None) -> float | None:
    """Convert total_runtime timedelta to seconds, or None."""
    if result is None or result.total_runtime is None:
        return None
    return result.total_runtime.total_seconds()


@router.get("/messages")
def get_messages(
    severity: str | None = Query(default=None, description="Filter by severity (WARN/FATAL/INFO)"),
    limit: int = Query(default=100, ge=1, le=10000, description="Max messages to return"),
    offset: int = Query(default=0, ge=0, description="Offset for pagination"),
) -> dict[str, Any]:
    """Parse and return simulation messages with optional severity filtering."""
    result = _get_diagnostics_result()
    if result is None:
        return {
            "messages": [],
            "total": 0,
            "warning_count": 0,
            "error_count": 0,
            "total_runtime_seconds": None,
        }

    messages = result.messages

    # Filter by severity if requested
    if severity is not None:
        from pyiwfm.io.simulation_messages import MessageSeverity

        try:
            sev = MessageSeverity[severity.upper()]
        except KeyError:
            # Unknown severity — return empty
            return {
                "messages": [],
                "total": 0,
                "warning_count": result.warning_count,
                "error_count": result.error_count,
                "total_runtime_seconds": _runtime_seconds(result),
            }
        messages = result.filter_by_severity(sev)

    total = len(messages)
    page = messages[offset : offset + limit]

    serialized = [
        {
            "severity": m.severity.name,
            "text": m.text,
            "procedure": m.procedure,
            "line_number": m.line_number,
            "node_ids": m.node_ids,
            "element_ids": m.element_ids,
            "reach_ids": m.reach_ids,
            "layer_ids": m.layer_ids,
        }
        for m in page
    ]

    return {
        "messages": serialized,
        "total": total,
        "warning_count": result.warning_count,
        "error_count": result.error_count,
        "total_runtime_seconds": _runtime_seconds(result),
    }


@router.get("/convergence")
def get_convergence() -> dict[str, Any]:
    """Return convergence iteration data from simulation messages."""
    result = _get_diagnostics_result()

    # Convergence records require the extended dataclasses
    if result is None or not _HAS_CONVERGENCE:
        return {
            "records": [],
            "max_iterations": 0,
            "avg_iterations": 0.0,
            "total_timesteps": 0,
        }

    # Use get_convergence_summary if available on the result
    convergence_summary: dict[str, Any] = {}
    if hasattr(result, "get_convergence_summary"):
        convergence_summary = result.get_convergence_summary()

    records: list[dict[str, Any]] = []
    if hasattr(result, "convergence_records"):
        for rec in result.convergence_records:
            records.append(
                {
                    "timestep_index": rec.timestep_index,
                    "date": rec.date,
                    "iteration_count": rec.iteration_count,
                    "max_residual": rec.max_residual,
                    "convergence_achieved": rec.convergence_achieved,
                }
            )

    iteration_counts = [r["iteration_count"] for r in records]
    max_iter = convergence_summary.get(
        "max_iterations", max(iteration_counts) if iteration_counts else 0
    )
    avg_iter = convergence_summary.get(
        "avg_iterations",
        sum(iteration_counts) / len(iteration_counts) if iteration_counts else 0.0,
    )
    total_ts = convergence_summary.get("total_timesteps", len(records))

    return {
        "records": records,
        "max_iterations": max_iter,
        "avg_iterations": avg_iter,
        "total_timesteps": total_ts,
    }


@router.get("/mass-balance")
def get_mass_balance(
    component: str | None = Query(default=None, description="Filter by component name"),
) -> dict[str, Any]:
    """Return mass balance error timeseries from simulation messages."""
    result = _get_diagnostics_result()

    if result is None or not _HAS_MASS_BALANCE:
        return {"records": [], "components": []}

    records: list[dict[str, Any]] = []
    components: set[str] = set()

    if hasattr(result, "mass_balance_records"):
        for rec in result.mass_balance_records:
            components.add(rec.component)
            if component is not None and rec.component != component:
                continue
            records.append(
                {
                    "timestep_index": rec.timestep_index,
                    "date": rec.date,
                    "component": rec.component,
                    "error_value": rec.error_value,
                    "error_percent": rec.error_percent,
                }
            )

    return {
        "records": records,
        "components": sorted(components),
    }


@router.get("/summary")
def get_summary() -> dict[str, Any]:
    """Combined diagnostics summary."""
    result = _get_diagnostics_result()
    has_diag = result is not None

    if not has_diag:
        return {
            "has_diagnostics": False,
            "message_count": 0,
            "warning_count": 0,
            "error_count": 0,
            "total_runtime_seconds": None,
            "max_iterations": 0,
            "avg_iterations": 0.0,
            "spatial_summary": {"nodes": {}, "elements": {}, "reaches": {}},
        }

    assert result is not None  # for type narrowing

    # Convergence stats
    convergence_data = get_convergence()

    # Spatial summary
    spatial = result.get_spatial_summary()

    return {
        "has_diagnostics": True,
        "message_count": len(result.messages),
        "warning_count": result.warning_count,
        "error_count": result.error_count,
        "total_runtime_seconds": _runtime_seconds(result),
        "max_iterations": convergence_data["max_iterations"],
        "avg_iterations": convergence_data["avg_iterations"],
        "spatial_summary": {
            "nodes": {str(k): v for k, v in spatial["nodes"].items()},
            "elements": {str(k): v for k, v in spatial["elements"].items()},
            "reaches": {str(k): v for k, v in spatial["reaches"].items()},
        },
    }


@router.get("/spatial-summary")
def get_spatial_summary() -> dict[str, dict[str, int]]:
    """Spatial summary of message counts for map overlay."""
    result = _get_diagnostics_result()

    if result is None:
        return {"nodes": {}, "elements": {}, "reaches": {}}

    spatial = result.get_spatial_summary()

    # JSON keys must be strings
    return {
        "nodes": {str(k): v for k, v in spatial["nodes"].items()},
        "elements": {str(k): v for k, v in spatial["elements"].items()},
        "reaches": {str(k): v for k, v in spatial["reaches"].items()},
    }
