"""Water budget mass balance sanity checks.

This module provides utilities for verifying mass balance closure in IWFM
budget output.  Given a :class:`BudgetReader`, it classifies columns as
inflows, outflows, or storage changes and computes per-timestep discrepancies.

Example
-------
>>> from pyiwfm.io.budget import BudgetReader
>>> from pyiwfm.io.budget.checks import check_budget_balance
>>> reader = BudgetReader("GW_Budget.hdf")
>>> report = check_budget_balance(reader, location_index=0)
>>> print(report.max_percent_error)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import timedelta
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pyiwfm.io.budget import BudgetReader

# -------------------------------------------------------------------
# Keyword patterns used to auto-classify budget columns
# -------------------------------------------------------------------
_INFLOW_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"inflow", re.IGNORECASE),
    re.compile(r"recharge", re.IGNORECASE),
    re.compile(r"\bgain\b", re.IGNORECASE),
    re.compile(r"precip", re.IGNORECASE),
    re.compile(r"applied water", re.IGNORECASE),
    re.compile(r"return flow", re.IGNORECASE),
    re.compile(r"tile drain$", re.IGNORECASE),
    re.compile(r"runoff", re.IGNORECASE),
    re.compile(r"deep percolation", re.IGNORECASE),
    re.compile(r"percolation in", re.IGNORECASE),
    re.compile(r"boundary inflow", re.IGNORECASE),
    re.compile(r"subsurface irrigation", re.IGNORECASE),
]

_OUTFLOW_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"outflow", re.IGNORECASE),
    re.compile(r"discharge", re.IGNORECASE),
    re.compile(r"\bloss\b", re.IGNORECASE),
    re.compile(r"pumping", re.IGNORECASE),
    re.compile(r"diversion", re.IGNORECASE),
    re.compile(r"evaporation", re.IGNORECASE),
    re.compile(r"\bET\b"),
    re.compile(r"percolation out", re.IGNORECASE),
    re.compile(r"tile drain outflow", re.IGNORECASE),
    re.compile(r"by-?pass", re.IGNORECASE),
]

_STORAGE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"begin storage", re.IGNORECASE),
    re.compile(r"end storage", re.IGNORECASE),
    re.compile(r"storage", re.IGNORECASE),
    re.compile(r"\bchange\b", re.IGNORECASE),
]

# Columns that should be ignored during mass balance checks
_SKIP_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"discrepancy", re.IGNORECASE),
    re.compile(r"\barea\b", re.IGNORECASE),
    re.compile(r"cumulative", re.IGNORECASE),
    re.compile(r"shortage", re.IGNORECASE),
    re.compile(r"potential", re.IGNORECASE),
    re.compile(r"supply req", re.IGNORECASE),
]


def _matches_any(text: str, patterns: list[re.Pattern[str]]) -> bool:
    """Return True if *text* matches any of the compiled patterns."""
    return any(p.search(text) for p in patterns)


# -------------------------------------------------------------------
# Result data classes
# -------------------------------------------------------------------


@dataclass
class BalanceCheckResult:
    """Result of a mass balance check for one timestep."""

    timestep_index: int
    date: str
    total_inflow: float
    total_outflow: float
    storage_change: float
    discrepancy: float  # inflows - outflows - storage_change
    percent_error: float  # discrepancy / max(|inflows|, |outflows|) * 100
    within_tolerance: bool


@dataclass
class BudgetSanityReport:
    """Full sanity check report for a budget."""

    budget_type: str
    location: str
    n_timesteps: int
    n_violations: int  # timesteps exceeding tolerance
    max_percent_error: float
    mean_percent_error: float
    records: list[BalanceCheckResult] = field(default_factory=list)

    def to_summary_dict(self) -> dict[str, Any]:
        """Serialize summary (without per-timestep records) for API."""
        return {
            "budget_type": self.budget_type,
            "location": self.location,
            "n_timesteps": self.n_timesteps,
            "n_violations": self.n_violations,
            "max_percent_error": self.max_percent_error,
            "mean_percent_error": self.mean_percent_error,
        }


# -------------------------------------------------------------------
# Column classification helpers
# -------------------------------------------------------------------


def _classify_columns(
    headers: list[str],
    inflow_columns: list[str] | None = None,
    outflow_columns: list[str] | None = None,
    storage_column: str | None = None,
) -> tuple[list[int], list[int], list[int]]:
    """Classify columns into inflow, outflow, and storage indices.

    Parameters
    ----------
    headers : list[str]
        Column header names from the budget reader.
    inflow_columns, outflow_columns : list[str] or None
        Explicit column names.  When *None* the function auto-detects by
        matching against keyword patterns.
    storage_column : str or None
        Explicit storage column name.  Auto-detected when *None*.

    Returns
    -------
    tuple of three ``list[int]``
        (inflow_indices, outflow_indices, storage_indices)
    """
    if inflow_columns is not None and outflow_columns is not None:
        lower_map = {h.lower(): i for i, h in enumerate(headers)}
        in_idx = [lower_map[c.lower()] for c in inflow_columns if c.lower() in lower_map]
        out_idx = [lower_map[c.lower()] for c in outflow_columns if c.lower() in lower_map]
        stor_idx: list[int] = []
        if storage_column is not None and storage_column.lower() in lower_map:
            stor_idx = [lower_map[storage_column.lower()]]
        return in_idx, out_idx, stor_idx

    # Auto-detect
    in_idx = []
    out_idx = []
    stor_idx = []
    begin_storage_idx: int | None = None
    end_storage_idx: int | None = None

    for i, h in enumerate(headers):
        if _matches_any(h, _SKIP_PATTERNS):
            continue

        # Check storage first (more specific patterns)
        if re.search(r"begin storage", h, re.IGNORECASE):
            begin_storage_idx = i
            continue
        if re.search(r"end storage", h, re.IGNORECASE):
            end_storage_idx = i
            continue

        # Outflow patterns that are more specific must be checked before
        # the generic inflow patterns (e.g. "Tile Drain Outflow" should be
        # outflow, not inflow via "Tile Drain").
        if _matches_any(h, _OUTFLOW_PATTERNS):
            out_idx.append(i)
            continue

        if _matches_any(h, _INFLOW_PATTERNS):
            in_idx.append(i)
            continue

        if _matches_any(h, _STORAGE_PATTERNS):
            stor_idx.append(i)
            continue

    # Use Begin/End Storage to compute storage change if both are present
    if begin_storage_idx is not None and end_storage_idx is not None:
        stor_idx = [begin_storage_idx, end_storage_idx]
    elif begin_storage_idx is not None:
        stor_idx = [begin_storage_idx]
    elif end_storage_idx is not None:
        stor_idx = [end_storage_idx]

    if storage_column is not None:
        lower_map = {h.lower(): i for i, h in enumerate(headers)}
        if storage_column.lower() in lower_map:
            stor_idx = [lower_map[storage_column.lower()]]

    return in_idx, out_idx, stor_idx


# -------------------------------------------------------------------
# Time-string generation (mirrors budgets route logic)
# -------------------------------------------------------------------


def _make_time_strings(reader: BudgetReader, n_timesteps: int) -> list[str]:
    """Generate ISO date strings for each timestep."""
    ts = reader.header.timestep
    if ts.start_datetime:
        use_months = "MON" in ts.unit.upper() if ts.unit else False
        if use_months:
            from dateutil.relativedelta import relativedelta

            return [
                (ts.start_datetime + relativedelta(months=i)).isoformat()
                for i in range(n_timesteps)
            ]
        delta = timedelta(minutes=ts.delta_t_minutes)
        return [(ts.start_datetime + delta * i).isoformat() for i in range(n_timesteps)]
    return [str(i) for i in range(n_timesteps)]


# -------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------


def check_budget_balance(
    reader: BudgetReader,
    location_index: int = 0,
    tolerance_percent: float = 1.0,
    inflow_columns: list[str] | None = None,
    outflow_columns: list[str] | None = None,
    storage_column: str | None = None,
) -> BudgetSanityReport:
    """Check mass balance for a budget location.

    If column names are not specified, auto-detect by looking for columns
    containing ``'inflow'``/``'recharge'``/``'gain'`` for inflows,
    ``'outflow'``/``'discharge'``/``'loss'`` for outflows, and
    ``'storage'``/``'change'`` for storage.

    Parameters
    ----------
    reader : BudgetReader
        An already-opened budget reader.
    location_index : int
        Zero-based location index.
    tolerance_percent : float
        Maximum acceptable percent error.
    inflow_columns, outflow_columns : list[str] or None
        Explicit column names for inflows/outflows.
    storage_column : str or None
        Explicit storage column name.

    Returns
    -------
    BudgetSanityReport
    """
    headers = reader.get_column_headers(location_index)
    times_arr, values_arr = reader.get_values(location_index)
    n_timesteps = values_arr.shape[0]

    in_idx, out_idx, stor_idx = _classify_columns(
        headers,
        inflow_columns=inflow_columns,
        outflow_columns=outflow_columns,
        storage_column=storage_column,
    )

    time_strings = _make_time_strings(reader, n_timesteps)
    location_name = (
        reader.locations[location_index]
        if location_index < len(reader.locations)
        else str(location_index)
    )

    records: list[BalanceCheckResult] = []
    max_pct = 0.0
    sum_pct = 0.0
    n_violations = 0

    for t in range(n_timesteps):
        total_inflow = float(np.nansum(values_arr[t, in_idx])) if in_idx else 0.0
        total_outflow = float(np.nansum(values_arr[t, out_idx])) if out_idx else 0.0

        # Storage change: End Storage - Begin Storage if both present,
        # otherwise just sum the storage columns.
        if len(stor_idx) == 2:
            begin_val = float(values_arr[t, stor_idx[0]])
            end_val = float(values_arr[t, stor_idx[1]])
            storage_change = end_val - begin_val
        elif stor_idx:
            storage_change = float(np.nansum(values_arr[t, stor_idx]))
        else:
            storage_change = 0.0

        discrepancy = total_inflow - total_outflow - storage_change

        denom = max(abs(total_inflow), abs(total_outflow))
        if denom > 0:
            percent_error = abs(discrepancy) / denom * 100.0
        else:
            percent_error = 0.0

        within_tol = percent_error <= tolerance_percent
        if not within_tol:
            n_violations += 1

        abs_pct = abs(percent_error)
        if abs_pct > max_pct:
            max_pct = abs_pct
        sum_pct += abs_pct

        records.append(
            BalanceCheckResult(
                timestep_index=t,
                date=time_strings[t] if t < len(time_strings) else str(t),
                total_inflow=total_inflow,
                total_outflow=total_outflow,
                storage_change=storage_change,
                discrepancy=discrepancy,
                percent_error=percent_error,
                within_tolerance=within_tol,
            )
        )

    mean_pct = sum_pct / n_timesteps if n_timesteps > 0 else 0.0

    return BudgetSanityReport(
        budget_type=reader.descriptor,
        location=location_name,
        n_timesteps=n_timesteps,
        n_violations=n_violations,
        max_percent_error=max_pct,
        mean_percent_error=mean_pct,
        records=records,
    )


def check_all_budgets(
    budget_readers: dict[str, BudgetReader],
    tolerance_percent: float = 1.0,
) -> dict[str, BudgetSanityReport]:
    """Check mass balance across all available budgets.

    Parameters
    ----------
    budget_readers : dict[str, BudgetReader]
        Mapping of budget type name to reader instance.
    tolerance_percent : float
        Maximum acceptable percent error per timestep.

    Returns
    -------
    dict[str, BudgetSanityReport]
        Mapping of budget type name to its sanity report (first location only).
    """
    results: dict[str, BudgetSanityReport] = {}
    for name, reader in budget_readers.items():
        try:
            results[name] = check_budget_balance(
                reader,
                location_index=0,
                tolerance_percent=tolerance_percent,
            )
        except (KeyError, IndexError, ValueError):
            # Skip budgets that cannot be read
            continue
    return results
