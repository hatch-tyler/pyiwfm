"""Stream depletion analysis — compare streamflow between baseline and pumping scenarios."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclasses.dataclass
class StreamDepletionResult:
    """Result of stream depletion analysis at a single reach/node."""

    reach_id: int
    reach_name: str
    times: list[str]
    baseline_flow: NDArray[np.float64]
    scenario_flow: NDArray[np.float64]
    depletion: NDArray[np.float64]  # baseline - scenario (positive = depletion)
    cumulative_depletion: NDArray[np.float64]
    max_depletion: float
    max_depletion_timestep: int
    total_depletion: float  # cumulative at last timestep

    def to_dict(self) -> dict[str, object]:
        """Serialize for API response."""
        return {
            "reach_id": self.reach_id,
            "reach_name": self.reach_name,
            "times": self.times,
            "depletion": [round(float(v), 4) for v in self.depletion],
            "cumulative_depletion": [round(float(v), 4) for v in self.cumulative_depletion],
            "max_depletion": round(float(self.max_depletion), 4),
            "max_depletion_timestep": int(self.max_depletion_timestep),
            "total_depletion": round(float(self.total_depletion), 4),
        }


@dataclasses.dataclass
class StreamDepletionReport:
    """Aggregate depletion results across multiple reaches."""

    results: list[StreamDepletionResult]
    n_reaches: int
    n_timesteps: int
    total_max_depletion: float  # max depletion across all reaches and timesteps
    total_cumulative_depletion: float  # sum of cumulative depletion at last timestep

    def to_dict(self) -> dict[str, object]:
        """Serialize for API response."""
        return {
            "n_reaches": self.n_reaches,
            "n_timesteps": self.n_timesteps,
            "total_max_depletion": round(float(self.total_max_depletion), 4),
            "total_cumulative_depletion": round(float(self.total_cumulative_depletion), 4),
            "reaches": [r.to_dict() for r in self.results],
        }


def compute_stream_depletion(
    baseline_dir: str | Path,
    scenario_dir: str | Path,
    reach_ids: list[int] | None = None,
    budget_type: str = "Stream Reach Budgets",
) -> StreamDepletionReport:
    """Compute stream depletion by comparing two model runs.

    Parameters
    ----------
    baseline_dir : str or Path
        Path to the baseline (no-pumping) model results directory.
    scenario_dir : str or Path
        Path to the pumping scenario model results directory.
    reach_ids : list[int] or None
        Specific reach IDs to analyze. If None, analyze all reaches.
    budget_type : str
        Budget type name to look for in the HDF5 files.

    Returns
    -------
    StreamDepletionReport
        Depletion analysis results.
    """
    from pyiwfm.io.budget import BudgetReader

    baseline_path = Path(baseline_dir)
    scenario_path = Path(scenario_dir)

    # Find stream budget files
    baseline_budget = _find_stream_budget(baseline_path, budget_type)
    scenario_budget = _find_stream_budget(scenario_path, budget_type)

    baseline_reader = BudgetReader(str(baseline_budget))
    scenario_reader = BudgetReader(str(scenario_budget))

    # Get locations (reaches)
    locations = baseline_reader.locations
    if reach_ids is not None:
        location_indices = [i for i, loc in enumerate(locations) if i + 1 in reach_ids]
    else:
        location_indices = list(range(len(locations)))

    results: list[StreamDepletionResult] = []

    for loc_idx in location_indices:
        reach_name = locations[loc_idx] if loc_idx < len(locations) else f"Reach {loc_idx + 1}"

        # Get baseline and scenario data
        base_times, base_values = baseline_reader.get_values(loc_idx)
        scen_times, scen_values = scenario_reader.get_values(loc_idx)

        # Use the total outflow column (typically last non-storage column)
        # or sum inflows - outflows to get net flow
        headers = baseline_reader.get_column_headers(loc_idx)

        # Find gain/loss from GW interaction columns
        # Look for stream-aquifer interaction columns
        base_flow = _extract_stream_flow(headers, base_values)
        scen_flow = _extract_stream_flow(headers, scen_values)

        # Ensure same number of timesteps
        n_ts = min(len(base_flow), len(scen_flow))
        base_flow = base_flow[:n_ts]
        scen_flow = scen_flow[:n_ts]

        # Depletion = baseline - scenario (positive means depletion)
        depletion = base_flow - scen_flow
        cumulative = np.cumsum(depletion)

        # Generate time strings
        times = _format_times(base_times[:n_ts], baseline_reader)

        max_dep_idx = int(np.argmax(np.abs(depletion)))

        results.append(
            StreamDepletionResult(
                reach_id=loc_idx + 1,
                reach_name=reach_name,
                times=times,
                baseline_flow=base_flow,
                scenario_flow=scen_flow,
                depletion=depletion,
                cumulative_depletion=cumulative,
                max_depletion=float(np.max(np.abs(depletion))),
                max_depletion_timestep=max_dep_idx,
                total_depletion=float(cumulative[-1]) if len(cumulative) > 0 else 0.0,
            )
        )

    total_max = max((r.max_depletion for r in results), default=0.0)
    total_cum = sum(r.total_depletion for r in results)
    n_timesteps = results[0].depletion.shape[0] if results else 0

    return StreamDepletionReport(
        results=results,
        n_reaches=len(results),
        n_timesteps=n_timesteps,
        total_max_depletion=total_max,
        total_cumulative_depletion=total_cum,
    )


def _find_stream_budget(results_dir: Path, budget_type: str) -> Path:
    """Find stream budget HDF5 file in a results directory.

    Parameters
    ----------
    results_dir : Path
        Directory to search for stream budget files.
    budget_type : str
        Budget type name (unused in current implementation but reserved for
        future multi-budget disambiguation).

    Returns
    -------
    Path
        Path to the stream budget file.

    Raises
    ------
    FileNotFoundError
        If no stream budget file is found.
    """
    # Look for common stream budget file patterns
    patterns = [
        "*.hdf",
        "*.h5",
        "*Stream*Budget*.hdf5",
        "*Stream*Budget*.hdf",
        "*stream*budget*",
    ]

    for pattern in patterns:
        matches = list(results_dir.glob(pattern))
        for m in matches:
            if "stream" in m.stem.lower() and "budget" in m.stem.lower():
                return m

    # Fallback: try any HDF file
    for pattern in ["*.hdf", "*.h5", "*.hdf5"]:
        matches = list(results_dir.glob(pattern))
        if matches:
            return matches[0]

    msg = f"No stream budget file found in {results_dir}"
    raise FileNotFoundError(msg)


def _extract_stream_flow(
    headers: list[str],
    values: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Extract net stream flow from budget columns.

    Looks for gain/loss from groundwater interaction columns.
    Falls back to summing all inflow (+) and outflow (-) columns.

    Parameters
    ----------
    headers : list[str]
        Column header names from the budget file.
    values : NDArray[np.float64]
        2-D array of shape ``(n_timesteps, n_columns)``.

    Returns
    -------
    NDArray[np.float64]
        1-D array of stream flow values, one per timestep.
    """
    # Look for GW interaction column
    for i, h in enumerate(headers):
        h_lower = h.lower()
        if "gain from gw" in h_lower or "stream-aquifer" in h_lower:
            result: NDArray[np.float64] = values[:, i].copy()
            return result

    # Fallback: sum inflows minus outflows
    total = np.zeros(values.shape[0])
    for i, h in enumerate(headers):
        if "(+)" in h:
            total += values[:, i]
        elif "(-)" in h:
            total += values[:, i]  # already negative

    return total


def _format_times(
    times_array: NDArray[np.float64],
    reader: object,
) -> list[str]:
    """Convert time array to string list.

    Parameters
    ----------
    times_array : NDArray[np.float64]
        Array of time values (Julian days or similar).
    reader : object
        Budget reader instance, checked for header/timestep metadata.

    Returns
    -------
    list[str]
        ISO-format date strings or integer indices as fallback.
    """
    try:
        start_dt = getattr(getattr(reader, "header", None), "timestep", None)
        if start_dt and hasattr(start_dt, "start_datetime") and start_dt.start_datetime:
            from datetime import timedelta

            base = start_dt.start_datetime
            return [(base + timedelta(days=float(t))).isoformat()[:10] for t in times_array]
    except Exception:  # noqa: BLE001
        pass
    return [str(int(t)) for t in times_array]
