"""
Parser for IWFM SimulationMessages.out files.

SimulationMessages.out is produced by IWFM during simulation and contains
warnings, errors, and informational messages with spatial context (node IDs,
element IDs, reach IDs, layer IDs).  Additionally parses convergence iteration
counts, mass balance errors, and timestep cut / damping factor events.

Example
-------
>>> from pyiwfm.io.simulation_messages import SimulationMessagesReader
>>> reader = SimulationMessagesReader("SimulationMessages.out")
>>> result = reader.read()
>>> print(f"Warnings: {result.warning_count}, Errors: {result.error_count}")
>>> for msg in result.filter_by_severity(MessageSeverity.WARN):
...     print(msg.text[:80])
>>> summary = result.get_convergence_summary()
>>> print(f"Max iterations: {summary['max_iterations']}")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum
from typing import Any

from pyiwfm.io.base import BaseReader


class MessageSeverity(Enum):
    """Severity levels for simulation messages."""

    MESSAGE = 0
    INFO = 1
    WARN = 2
    FATAL = 3


# Regex patterns for spatial ID extraction (case-insensitive)
_NODE_RE = re.compile(r"(?:node|nd)\s*[#=:]?\s*(\d+)", re.IGNORECASE)
_ELEMENT_RE = re.compile(r"(?:element|elem)\s*[#=:]?\s*(\d+)", re.IGNORECASE)
_REACH_RE = re.compile(r"(?:reach)\s*[#=:]?\s*(\d+)", re.IGNORECASE)
_LAYER_RE = re.compile(r"(?:layer)\s*[#=:]?\s*(\d+)", re.IGNORECASE)

# Pattern to detect message block starts
_SEVERITY_RE = re.compile(r"^\*\s*(FATAL|WARN(?:ING)?|INFO)\s*:", re.IGNORECASE)

# Pattern to extract procedure name from parentheses at end of message
_PROCEDURE_RE = re.compile(r"\((\w+)\)\s*$")

# Pattern to extract runtime from summary
_RUNTIME_RE = re.compile(
    r"(?:total\s+)?run(?:\s*-?\s*)time\s*[=:]\s*(\d+)\s*h(?:ours?)?\s*"
    r"(\d+)\s*m(?:in(?:utes?)?)?\s*(\d+(?:\.\d+)?)\s*s(?:ec(?:onds?)?)?",
    re.IGNORECASE,
)

# Pattern to extract warning count from summary
_WARNING_COUNT_RE = re.compile(r"(\d+)\s+warning", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Convergence / mass-balance / timestep-cut patterns
# ---------------------------------------------------------------------------

# Timestep context: "TIME STEP #123" or "TIME STEP 123"
_TIMESTEP_RE = re.compile(r"TIME\s+STEP\s*#?\s*(\d+)", re.IGNORECASE)

# IWFM date in MM/DD/YYYY_HH:MM format
_IWFM_DATE_RE = re.compile(r"\b(\d{1,2}/\d{1,2}/\d{4}_\d{2}:\d{2})\b")

# Iteration line: "Iteration #3" or "ITERATION  5"
_ITERATION_RE = re.compile(r"ITERATION\s*#?\s*(\d+)", re.IGNORECASE)

# Converged: "converged after 4 iterations" or "CONVERGENCE ACHIEVED IN 4 ITERATIONS"
_CONVERGED_RE = re.compile(
    r"CONVERG\w*\s+(?:ACHIEVED\s+IN|after)\s+(\d+)\s+ITERATION",
    re.IGNORECASE,
)

# Convergence failure: "CONVERGENCE NOT ACHIEVED" or "DID NOT CONVERGE"
_NO_CONVERGE_RE = re.compile(
    r"CONVERGENCE\s+NOT\s+ACHIEVED|DID\s+NOT\s+CONVERGE|NOT\s+CONVERGED",
    re.IGNORECASE,
)

# Max residual: "MAX RESIDUAL = 1.234E-03" or "maximum residual: 0.0012"
_MAX_RESIDUAL_RE = re.compile(
    r"MAX(?:IMUM)?\s+RESIDUAL\s*[=:]\s*([+-]?\d+\.?\d*(?:[EeDd][+-]?\d+)?)",
    re.IGNORECASE,
)

# Mass balance error: "MASS BALANCE ERROR" or "mass balance" lines with numeric values
_MASS_BALANCE_RE = re.compile(
    r"MASS\s+BALANCE\s*(?:ERROR)?\s*[=:]\s*([+-]?\d+\.?\d*(?:[EeDd][+-]?\d+)?)",
    re.IGNORECASE,
)

# Mass balance with percentage: "1.234 (0.05%)" or "error = 1.234, 0.05 %"
_MASS_BALANCE_PCT_RE = re.compile(
    r"([+-]?\d+\.?\d*(?:[EeDd][+-]?\d+)?)\s*"
    r"(?:\(|\s*,\s*|\s+)"
    r"([+-]?\d+\.?\d*(?:[EeDd][+-]?\d+)?)\s*%",
    re.IGNORECASE,
)

# Component in mass balance context: "GROUNDWATER mass balance" etc.
_MB_COMPONENT_RE = re.compile(
    r"(GROUNDWATER|STREAM|ROOT\s*ZONE|LAKE|UNSATURATED)\s+MASS\s+BALANCE",
    re.IGNORECASE,
)

# Storage change line (alternative mass balance indicator)
_STORAGE_CHANGE_RE = re.compile(
    r"STORAGE\s+CHANGE\s*[=:]\s*([+-]?\d+\.?\d*(?:[EeDd][+-]?\d+)?)",
    re.IGNORECASE,
)

# Timestep cut: "TIME STEP CUT" or "REDUCING TIME STEP"
_TIMESTEP_CUT_RE = re.compile(
    r"TIME\s*STEP\s*CUT|REDUCING\s+TIME\s*STEP|TIME\s*STEP\s+REDUCED",
    re.IGNORECASE,
)

# New dt after cut: "new dt = 0.5" or "DT = 3600"
_NEW_DT_RE = re.compile(
    r"(?:NEW\s+)?DT\s*[=:]\s*([+-]?\d+\.?\d*(?:[EeDd][+-]?\d+)?)",
    re.IGNORECASE,
)

# Damping factor: "DAMPING FACTOR" or "DAMPING = 0.5"
_DAMPING_RE = re.compile(
    r"DAMPING(?:\s+FACTOR)?\s*[=:]\s*([+-]?\d+\.?\d*(?:[EeDd][+-]?\d+)?)",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# IWFM tabular convergence iteration patterns
# ---------------------------------------------------------------------------

# Supply adjustment iteration header: "*** SUPPLY ADJUSTMENT ITERATION:     1 ***"
_SUPPLY_ADJ_RE = re.compile(
    r"\*{3}\s*SUPPLY\s+ADJUSTMENT\s+ITERATION\s*:\s*(\d+)\s*\*{3}",
    re.IGNORECASE,
)

# Iteration data row from convergence table
# Fortran format: (1X,I6,4X,G13.6,4X,G13.6,4X,A15,2X,G13.6)
# Example: "      1      9167.18          831.113        GW_69_(L1)         1.00000"
_ITER_ROW_RE = re.compile(
    r"^\s+(\d+)\s+"  # ITER number
    r"([+-]?\d+\.?\d*(?:[EeDd][+-]?\d+)?)\s+"  # HEAD CONVERGENCE (L2)
    r"([+-]?\d+\.?\d*(?:[EeDd][+-]?\d+)?)\s+"  # MAX.DIFF
    r"(\S+)\s+"  # VARIABLE (GW_N_(L1), ST_N, LK_N)
    r"([+-]?\d+\.?\d*(?:[EeDd][+-]?\d+)?)",  # VOLUMETRIC CONVERGENCE
)

# Convergence failure: "Desired convergence at GW node was not achieved"
_DESIRED_CONV_RE = re.compile(
    r"Desired\s+convergence\s+at\s+\w+\s+node\s+was\s+not\s+achieved",
    re.IGNORECASE,
)


@dataclass
class SimulationMessage:
    """A single parsed simulation message.

    Attributes
    ----------
    severity : MessageSeverity
        Message severity level.
    text : str
        Full message text (joined from continuation lines).
    procedure : str
        Name of the Fortran procedure that generated the message.
    line_number : int
        Line number in the file where the message starts.
    node_ids : list[int]
        Node IDs extracted from the message text.
    element_ids : list[int]
        Element IDs extracted from the message text.
    reach_ids : list[int]
        Reach IDs extracted from the message text.
    layer_ids : list[int]
        Layer IDs extracted from the message text.
    """

    severity: MessageSeverity
    text: str
    procedure: str
    line_number: int
    node_ids: list[int] = field(default_factory=list)
    element_ids: list[int] = field(default_factory=list)
    reach_ids: list[int] = field(default_factory=list)
    layer_ids: list[int] = field(default_factory=list)


@dataclass
class ConvergenceRecord:
    """Record of convergence iteration data for a single timestep.

    Attributes
    ----------
    timestep_index : int
        1-based timestep number.
    date : str
        IWFM date string (``MM/DD/YYYY_HH:MM``) for the timestep.
    iteration_count : int
        Number of solver iterations for this timestep.
    max_residual : float | None
        Maximum residual at final iteration, if reported.
    convergence_achieved : bool
        Whether the solver converged within the allowed iterations.
    """

    timestep_index: int
    date: str
    iteration_count: int
    max_residual: float | None
    convergence_achieved: bool
    bottleneck_variable: str = ""
    supply_adj_count: int = 0


# Regex for parsing variable identifiers from convergence tables
_GW_VAR_RE = re.compile(r"^GW_(\d+)(?:_\(L(\d+)\))?$")
_ST_VAR_RE = re.compile(r"^ST_(\d+)$")
_LK_VAR_RE = re.compile(r"^LK_(\d+)$")


def _parse_variable_id(variable: str) -> tuple[str, int, int | None]:
    """Parse a convergence table variable identifier.

    Parameters
    ----------
    variable : str
        Variable string like ``"GW_25393_(L1)"``, ``"ST_2620"``, ``"LK_5"``.

    Returns
    -------
    tuple[str, int, int | None]
        ``(entity_type, entity_id, layer)`` where entity_type is
        ``"groundwater"``, ``"stream"``, or ``"lake"``.
    """
    m = _GW_VAR_RE.match(variable)
    if m:
        layer = int(m.group(2)) if m.group(2) else None
        return "groundwater", int(m.group(1)), layer
    m = _ST_VAR_RE.match(variable)
    if m:
        return "stream", int(m.group(1)), None
    m = _LK_VAR_RE.match(variable)
    if m:
        return "lake", int(m.group(1)), None
    return "unknown", 0, None


@dataclass
class ConvergenceHotspot:
    """Aggregated convergence bottleneck information for a single variable.

    Attributes
    ----------
    variable : str
        Variable identifier (e.g. ``"GW_25393_(L1)"``).
    entity_type : str
        Entity type: ``"groundwater"``, ``"stream"``, or ``"lake"``.
    entity_id : int
        Numeric ID of the entity.
    layer : int | None
        Layer number for groundwater nodes.
    occurrence_count : int
        Number of timesteps where this was the bottleneck variable.
    total_iterations : int
        Sum of iteration counts across all timesteps where this was bottleneck.
    worst_timestep_index : int
        Timestep with the most iterations for this variable.
    worst_timestep_date : str
        Date of the worst timestep.
    """

    variable: str
    entity_type: str
    entity_id: int
    layer: int | None
    occurrence_count: int
    total_iterations: int
    worst_timestep_index: int
    worst_timestep_date: str


@dataclass
class MassBalanceRecord:
    """Mass balance error record for a single timestep and component.

    Attributes
    ----------
    timestep_index : int
        1-based timestep number.
    date : str
        IWFM date string for the timestep.
    component : str
        Hydrologic component (e.g. ``"groundwater"``, ``"stream"``,
        ``"rootzone"``).
    error_value : float
        Absolute mass balance error value.
    error_percent : float | None
        Mass balance error as a percentage, if reported.
    """

    timestep_index: int
    date: str
    component: str
    error_value: float
    error_percent: float | None


@dataclass
class TimestepCutRecord:
    """Record of a timestep cut or damping factor adjustment.

    Attributes
    ----------
    timestep_index : int
        1-based timestep number where the cut occurred.
    date : str
        IWFM date string for the timestep.
    reason : str
        Description of why the timestep was cut (e.g. ``"TIME STEP CUT"``,
        ``"DAMPING"``).
    new_dt : float | None
        New timestep size after the cut, if reported.
    """

    timestep_index: int
    date: str
    reason: str
    new_dt: float | None


@dataclass
class SimulationMessagesResult:
    """Result of parsing a SimulationMessages.out file.

    Attributes
    ----------
    messages : list[SimulationMessage]
        All parsed messages.
    total_runtime : timedelta | None
        Total simulation runtime if found in the summary.
    warning_count : int
        Number of warning messages.
    error_count : int
        Number of fatal/error messages.
    convergence_records : list[ConvergenceRecord]
        Per-timestep convergence iteration data.
    mass_balance_records : list[MassBalanceRecord]
        Per-timestep mass balance error data.
    timestep_cuts : list[TimestepCutRecord]
        Timestep cut / damping adjustment events.
    max_iterations : int
        Highest iteration count across all timesteps (0 if none parsed).
    avg_iterations : float
        Average iteration count across all timesteps (0.0 if none parsed).
    """

    messages: list[SimulationMessage]
    total_runtime: timedelta | None
    warning_count: int
    error_count: int
    convergence_records: list[ConvergenceRecord] = field(default_factory=list)
    mass_balance_records: list[MassBalanceRecord] = field(default_factory=list)
    timestep_cuts: list[TimestepCutRecord] = field(default_factory=list)
    max_iterations: int = 0
    avg_iterations: float = 0.0

    def get_convergence_summary(self) -> dict[str, Any]:
        """Return summary statistics for convergence and mass balance.

        Returns
        -------
        dict[str, Any]
            Dictionary with keys:

            - ``max_iterations`` — highest iteration count across all timesteps
            - ``avg_iterations`` — mean iteration count
            - ``total_timesteps`` — number of timesteps with convergence data
            - ``converged_count`` — number of timesteps that converged
            - ``failed_count`` — number of timesteps that did NOT converge
            - ``convergence_rate`` — fraction of timesteps that converged
            - ``timestep_cuts`` — total number of timestep cut events
            - ``mass_balance_records`` — total number of mass balance records
            - ``max_mass_balance_error`` — largest absolute mass balance error
            - ``components_with_errors`` — sorted list of unique component names
        """
        total = len(self.convergence_records)
        converged = sum(1 for r in self.convergence_records if r.convergence_achieved)
        failed = total - converged

        mb_errors = [abs(r.error_value) for r in self.mass_balance_records]
        max_mb = max(mb_errors) if mb_errors else 0.0
        components = sorted({r.component for r in self.mass_balance_records})

        return {
            "max_iterations": self.max_iterations,
            "avg_iterations": self.avg_iterations,
            "total_timesteps": total,
            "converged_count": converged,
            "failed_count": failed,
            "convergence_rate": converged / total if total > 0 else 0.0,
            "timestep_cuts": len(self.timestep_cuts),
            "mass_balance_records": len(self.mass_balance_records),
            "max_mass_balance_error": max_mb,
            "components_with_errors": components,
        }

    def get_hotspots(self) -> list[ConvergenceHotspot]:
        """Return convergence hotspot variables sorted by occurrence count.

        Groups convergence records by ``bottleneck_variable`` and aggregates
        occurrence counts, total iterations, and worst timestep.

        Returns
        -------
        list[ConvergenceHotspot]
            Hotspots sorted by occurrence count (descending).
        """
        groups: dict[str, list[ConvergenceRecord]] = {}
        for rec in self.convergence_records:
            if not rec.bottleneck_variable:
                continue
            groups.setdefault(rec.bottleneck_variable, []).append(rec)

        hotspots: list[ConvergenceHotspot] = []
        for var, recs in groups.items():
            entity_type, entity_id, layer = _parse_variable_id(var)
            worst = max(recs, key=lambda r: r.iteration_count)
            hotspots.append(
                ConvergenceHotspot(
                    variable=var,
                    entity_type=entity_type,
                    entity_id=entity_id,
                    layer=layer,
                    occurrence_count=len(recs),
                    total_iterations=sum(r.iteration_count for r in recs),
                    worst_timestep_index=worst.timestep_index,
                    worst_timestep_date=worst.date,
                )
            )
        hotspots.sort(key=lambda h: h.occurrence_count, reverse=True)
        return hotspots

    def filter_by_severity(self, severity: MessageSeverity) -> list[SimulationMessage]:
        """Return messages matching the given severity.

        Parameters
        ----------
        severity : MessageSeverity
            Severity level to filter by.

        Returns
        -------
        list[SimulationMessage]
            Filtered messages.
        """
        return [m for m in self.messages if m.severity == severity]

    def get_spatial_summary(self) -> dict[str, dict[int, int]]:
        """Summarize message counts by spatial entity.

        Returns
        -------
        dict[str, dict[int, int]]
            Mapping from entity type (``"nodes"``, ``"elements"``,
            ``"reaches"``, ``"layers"``) to a dict of ID → message count.
        """
        summary: dict[str, dict[int, int]] = {
            "nodes": {},
            "elements": {},
            "reaches": {},
            "layers": {},
        }
        for msg in self.messages:
            for nid in msg.node_ids:
                summary["nodes"][nid] = summary["nodes"].get(nid, 0) + 1
            for eid in msg.element_ids:
                summary["elements"][eid] = summary["elements"].get(eid, 0) + 1
            for rid in msg.reach_ids:
                summary["reaches"][rid] = summary["reaches"].get(rid, 0) + 1
            for lid in msg.layer_ids:
                summary["layers"][lid] = summary["layers"].get(lid, 0) + 1
        return summary

    def to_geodataframe(self, grid: Any) -> Any:
        """Convert spatial message summary to a GeoDataFrame.

        Parameters
        ----------
        grid : AppGrid
            The model grid for looking up node/element coordinates.

        Returns
        -------
        GeoDataFrame
            Point geometries at node locations or element centroids
            with message count attributes.
        """
        try:
            import geopandas as gpd
            from shapely.geometry import Point
        except ImportError as exc:
            raise ImportError("geopandas and shapely are required for to_geodataframe()") from exc

        summary = self.get_spatial_summary()
        rows: list[dict[str, Any]] = []

        # Node-based messages
        for node_id, count in summary["nodes"].items():
            idx = node_id - 1  # 1-based to 0-based
            if 0 <= idx < len(grid.nodes):
                node = grid.nodes[idx]
                rows.append(
                    {
                        "entity_type": "node",
                        "entity_id": node_id,
                        "message_count": count,
                        "geometry": Point(node.x, node.y),
                    }
                )

        # Element-based messages (centroids)
        for elem_id, count in summary["elements"].items():
            idx = elem_id - 1
            if 0 <= idx < len(grid.elements):
                elem = grid.elements[idx]
                cx = sum(grid.nodes[n - 1].x for n in elem.node_ids) / len(elem.node_ids)
                cy = sum(grid.nodes[n - 1].y for n in elem.node_ids) / len(elem.node_ids)
                rows.append(
                    {
                        "entity_type": "element",
                        "entity_id": elem_id,
                        "message_count": count,
                        "geometry": Point(cx, cy),
                    }
                )

        if not rows:
            return gpd.GeoDataFrame(
                columns=["entity_type", "entity_id", "message_count", "geometry"]
            )

        return gpd.GeoDataFrame(rows)


def _parse_severity(label: str) -> MessageSeverity:
    """Map a severity label string to MessageSeverity enum."""
    upper = label.upper()
    if upper == "FATAL":
        return MessageSeverity.FATAL
    if upper.startswith("WARN"):
        return MessageSeverity.WARN
    if upper == "INFO":
        return MessageSeverity.INFO
    return MessageSeverity.MESSAGE


def _extract_spatial_ids(text: str) -> tuple[list[int], list[int], list[int], list[int]]:
    """Extract spatial IDs from message text."""
    node_ids = sorted({int(m) for m in _NODE_RE.findall(text)})
    element_ids = sorted({int(m) for m in _ELEMENT_RE.findall(text)})
    reach_ids = sorted({int(m) for m in _REACH_RE.findall(text)})
    layer_ids = sorted({int(m) for m in _LAYER_RE.findall(text)})
    return node_ids, element_ids, reach_ids, layer_ids


def _extract_procedure(text: str) -> str:
    """Extract the procedure name from message text."""
    match = _PROCEDURE_RE.search(text)
    return match.group(1) if match else ""


def _scan_text_for_convergence(
    text: str,
    cur_ts: int,
    cur_date: str,
    cur_max_iter: int,
    cur_max_residual: float | None,
    convergence_records: list[ConvergenceRecord],
    mass_balance_records: list[MassBalanceRecord],
    timestep_cuts: list[TimestepCutRecord],
) -> None:
    """Scan parsed message text for convergence/mass-balance/timestep-cut patterns.

    Called for messages parsed from ``* INFO:``/``* WARN:`` blocks so that
    convergence data embedded in severity messages is not missed.
    """
    # Converged after N iterations
    conv_match = _CONVERGED_RE.search(text)
    if conv_match:
        n_iters = int(conv_match.group(1))
        res_match = _MAX_RESIDUAL_RE.search(text)
        residual = (
            float(res_match.group(1).replace("D", "E").replace("d", "e"))
            if res_match
            else cur_max_residual
        )
        convergence_records.append(
            ConvergenceRecord(
                timestep_index=cur_ts,
                date=cur_date,
                iteration_count=max(cur_max_iter, n_iters),
                max_residual=residual,
                convergence_achieved=True,
            )
        )
        return

    # Convergence NOT achieved (generic or IWFM "Desired convergence" FATAL)
    if _NO_CONVERGE_RE.search(text) or _DESIRED_CONV_RE.search(text):
        # Try to extract iteration count from the text
        iter_m = _ITERATION_RE.search(text)
        n_iters = int(iter_m.group(1)) if iter_m else cur_max_iter
        # Try after N iterations pattern
        after_m = re.search(r"after\s+(\d+)\s+iteration", text, re.IGNORECASE)
        if after_m:
            n_iters = max(n_iters, int(after_m.group(1)))
        # Try "Difference = value" pattern from IWFM FATAL messages
        res_match = _MAX_RESIDUAL_RE.search(text)
        if not res_match:
            res_match = re.search(
                r"Difference\s*=\s*([+-]?\d+\.?\d*(?:[EeDd][+-]?\d+)?)",
                text,
                re.IGNORECASE,
            )
        residual = (
            float(res_match.group(1).replace("D", "E").replace("d", "e"))
            if res_match
            else cur_max_residual
        )
        convergence_records.append(
            ConvergenceRecord(
                timestep_index=cur_ts,
                date=cur_date,
                iteration_count=max(n_iters, 1),
                max_residual=residual,
                convergence_achieved=False,
            )
        )
        return

    # Mass balance error
    mb_match = _MASS_BALANCE_RE.search(text)
    if mb_match:
        error_val = float(mb_match.group(1).replace("D", "E").replace("d", "e"))
        # Try to identify component
        component = "total"
        for comp_name in ("groundwater", "stream", "rootzone", "lake", "unsaturated"):
            if comp_name in text.lower():
                component = comp_name
                break
        # Try to extract percent
        pct_match = re.search(r"(\d+\.?\d*)\s*%", text)
        pct = float(pct_match.group(1)) if pct_match else None
        mass_balance_records.append(
            MassBalanceRecord(
                timestep_index=cur_ts,
                date=cur_date,
                component=component,
                error_value=error_val,
                error_percent=pct,
            )
        )

    # Timestep cut / damping
    if _TIMESTEP_CUT_RE.search(text):
        reason = text[:200]
        dt_match = re.search(r"(?:new|reduced)\s+(?:dt|time\s*step)\s*[=:]\s*([\d.]+)", text, re.I)
        new_dt = float(dt_match.group(1)) if dt_match else None
        timestep_cuts.append(
            TimestepCutRecord(
                timestep_index=cur_ts,
                date=cur_date,
                reason=reason,
                new_dt=new_dt,
            )
        )


class SimulationMessagesReader(BaseReader):
    """Reader for IWFM SimulationMessages.out files.

    Parameters
    ----------
    filepath : Path | str
        Path to the SimulationMessages.out file.
    """

    @property
    def format(self) -> str:
        return "simulation_messages"

    def read(self) -> SimulationMessagesResult:
        """Parse the SimulationMessages.out file.

        Returns
        -------
        SimulationMessagesResult
            Parsed messages with spatial context, convergence records,
            mass balance errors, and timestep cut events.
        """
        messages: list[SimulationMessage] = []
        convergence_records: list[ConvergenceRecord] = []
        mass_balance_records: list[MassBalanceRecord] = []
        timestep_cuts: list[TimestepCutRecord] = []
        total_runtime: timedelta | None = None
        summary_warning_count: int | None = None

        # Running context: current timestep index and date
        cur_ts: int = 0
        cur_date: str = ""
        # Track max iteration seen in current timestep
        cur_max_iter: int = 0
        cur_max_residual: float | None = None

        # Tabular iteration tracking (IWFM supply adjustment tables)
        cur_ts_total_iters: int = 0  # total solver iterations across all supply adjs
        cur_ts_supply_adjs: int = 0  # number of supply adjustment iterations
        last_head_conv: float | None = None  # last HEAD CONVERGENCE value
        last_bottleneck_var: str = ""  # VARIABLE from last iteration row
        in_tabular_block: bool = False  # inside a supply adjustment iteration table

        with open(self.filepath, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.rstrip("\n\r")

            # ----------------------------------------------------------
            # Timestep context: update cur_ts and cur_date
            # ----------------------------------------------------------
            ts_match = _TIMESTEP_RE.search(stripped)
            if ts_match:
                # Flush any pending iteration data from the previous timestep
                if cur_ts > 0 and cur_max_iter > 0:
                    convergence_records.append(
                        ConvergenceRecord(
                            timestep_index=cur_ts,
                            date=cur_date,
                            iteration_count=cur_max_iter,
                            max_residual=cur_max_residual,
                            convergence_achieved=True,
                        )
                    )
                # Flush tabular iteration data from previous timestep
                if cur_ts > 0 and cur_ts_total_iters > 0 and cur_max_iter == 0:
                    convergence_records.append(
                        ConvergenceRecord(
                            timestep_index=cur_ts,
                            date=cur_date,
                            iteration_count=cur_ts_total_iters,
                            max_residual=last_head_conv,
                            convergence_achieved=True,
                            bottleneck_variable=last_bottleneck_var,
                            supply_adj_count=cur_ts_supply_adjs,
                        )
                    )
                cur_ts = int(ts_match.group(1))
                cur_max_iter = 0
                cur_max_residual = None
                cur_ts_total_iters = 0
                cur_ts_supply_adjs = 0
                last_head_conv = None
                last_bottleneck_var = ""
                in_tabular_block = False

            date_match = _IWFM_DATE_RE.search(stripped)
            if date_match:
                cur_date = date_match.group(1)

            # ----------------------------------------------------------
            # Severity block start (existing logic, unchanged)
            # ----------------------------------------------------------
            match = _SEVERITY_RE.match(stripped)
            if match:
                severity = _parse_severity(match.group(1))
                start_line = i + 1  # 1-based line number

                # Collect the rest of this line after the severity label
                text_parts: list[str] = []
                after_label = stripped[match.end() :].strip()
                if after_label:
                    text_parts.append(after_label)

                # Collect continuation lines (start with '*' followed by spaces)
                i += 1
                while i < len(lines):
                    cont = lines[i].rstrip("\n\r")
                    if cont.startswith("*") and (len(cont) < 2 or cont[1] != " "):
                        # New severity block or separator (no space after *)
                        break
                    if cont.startswith("*"):
                        # Check if this is a new severity block
                        if _SEVERITY_RE.match(cont):
                            break
                        # Continuation line: strip leading '* ' or '*   '
                        content = cont.lstrip("*").strip()
                        if content:
                            text_parts.append(content)
                        i += 1
                    else:
                        break

                full_text = " ".join(text_parts)
                procedure = _extract_procedure(full_text)
                node_ids, element_ids, reach_ids, layer_ids = _extract_spatial_ids(full_text)

                messages.append(
                    SimulationMessage(
                        severity=severity,
                        text=full_text,
                        procedure=procedure,
                        line_number=start_line,
                        node_ids=node_ids,
                        element_ids=element_ids,
                        reach_ids=reach_ids,
                        layer_ids=layer_ids,
                    )
                )

                # Also scan parsed message text for convergence/mass-balance
                # patterns since these often appear as INFO/WARN messages.
                _scan_text_for_convergence(
                    full_text,
                    cur_ts,
                    cur_date,
                    cur_max_iter,
                    cur_max_residual,
                    convergence_records,
                    mass_balance_records,
                    timestep_cuts,
                )
                # Update cur_max_iter from convergence scan
                conv_match = _CONVERGED_RE.search(full_text)
                if conv_match:
                    cur_max_iter = 0
                    cur_max_residual = None
                elif _NO_CONVERGE_RE.search(full_text):
                    cur_max_iter = 0
                    cur_max_residual = None
                else:
                    im = _ITERATION_RE.search(full_text)
                    if im:
                        cur_max_iter = max(cur_max_iter, int(im.group(1)))

                # Extract date/timestep context from message text
                ts_match = _TIMESTEP_RE.search(full_text)
                if ts_match:
                    cur_ts = int(ts_match.group(1))
                date_match = _IWFM_DATE_RE.search(full_text)
                if date_match:
                    cur_date = date_match.group(1)

                continue

            # ----------------------------------------------------------
            # IWFM tabular convergence iteration parsing
            # ----------------------------------------------------------
            supply_match = _SUPPLY_ADJ_RE.search(stripped)
            if supply_match:
                cur_ts_supply_adjs += 1
                in_tabular_block = True
                i += 1
                continue

            if in_tabular_block:
                row_match = _ITER_ROW_RE.match(stripped)
                if row_match:
                    cur_ts_total_iters += 1
                    last_head_conv = float(row_match.group(2).replace("D", "E").replace("d", "e"))
                    last_bottleneck_var = row_match.group(4)
                    i += 1
                    continue

            # ----------------------------------------------------------
            # Convergence iteration tracking
            # ----------------------------------------------------------
            iter_match = _ITERATION_RE.search(stripped)
            if iter_match:
                iter_num = int(iter_match.group(1))
                cur_max_iter = max(cur_max_iter, iter_num)
                # Check for residual on the same line
                res_match = _MAX_RESIDUAL_RE.search(stripped)
                if res_match:
                    cur_max_residual = float(res_match.group(1).replace("D", "E").replace("d", "e"))

            # "converged after N iterations"
            conv_match = _CONVERGED_RE.search(stripped)
            if conv_match:
                n_iters = int(conv_match.group(1))
                cur_max_iter = max(cur_max_iter, n_iters)
                res_match = _MAX_RESIDUAL_RE.search(stripped)
                if res_match:
                    cur_max_residual = float(res_match.group(1).replace("D", "E").replace("d", "e"))
                convergence_records.append(
                    ConvergenceRecord(
                        timestep_index=cur_ts,
                        date=cur_date,
                        iteration_count=cur_max_iter,
                        max_residual=cur_max_residual,
                        convergence_achieved=True,
                    )
                )
                cur_max_iter = 0
                cur_max_residual = None

            # "CONVERGENCE NOT ACHIEVED"
            if _NO_CONVERGE_RE.search(stripped):
                res_match = _MAX_RESIDUAL_RE.search(stripped)
                if res_match:
                    cur_max_residual = float(res_match.group(1).replace("D", "E").replace("d", "e"))
                convergence_records.append(
                    ConvergenceRecord(
                        timestep_index=cur_ts,
                        date=cur_date,
                        iteration_count=max(cur_max_iter, 1),
                        max_residual=cur_max_residual,
                        convergence_achieved=False,
                    )
                )
                cur_max_iter = 0
                cur_max_residual = None

            # Standalone max residual (not on an iteration line)
            if not iter_match and not conv_match:
                res_match = _MAX_RESIDUAL_RE.search(stripped)
                if res_match:
                    cur_max_residual = float(res_match.group(1).replace("D", "E").replace("d", "e"))

            # ----------------------------------------------------------
            # Mass balance error parsing
            # ----------------------------------------------------------
            mb_match = _MASS_BALANCE_RE.search(stripped)
            if mb_match:
                error_val = float(mb_match.group(1).replace("D", "E").replace("d", "e"))
                # Determine component
                comp_match = _MB_COMPONENT_RE.search(stripped)
                component = (
                    comp_match.group(1).lower().replace(" ", "") if comp_match else "unknown"
                )
                # Check for percentage
                pct_match = _MASS_BALANCE_PCT_RE.search(stripped)
                error_pct: float | None = None
                if pct_match:
                    error_pct = float(pct_match.group(2).replace("D", "E").replace("d", "e"))
                mass_balance_records.append(
                    MassBalanceRecord(
                        timestep_index=cur_ts,
                        date=cur_date,
                        component=component,
                        error_value=error_val,
                        error_percent=error_pct,
                    )
                )

            # Storage change as alternative mass balance indicator
            sc_match = _STORAGE_CHANGE_RE.search(stripped)
            if sc_match and not mb_match:
                error_val = float(sc_match.group(1).replace("D", "E").replace("d", "e"))
                comp_match = _MB_COMPONENT_RE.search(stripped)
                component = (
                    comp_match.group(1).lower().replace(" ", "") if comp_match else "unknown"
                )
                mass_balance_records.append(
                    MassBalanceRecord(
                        timestep_index=cur_ts,
                        date=cur_date,
                        component=component,
                        error_value=error_val,
                        error_percent=None,
                    )
                )

            # ----------------------------------------------------------
            # Timestep cut / damping detection
            # ----------------------------------------------------------
            if _TIMESTEP_CUT_RE.search(stripped):
                dt_match = _NEW_DT_RE.search(stripped)
                new_dt: float | None = None
                if dt_match:
                    new_dt = float(dt_match.group(1).replace("D", "E").replace("d", "e"))
                timestep_cuts.append(
                    TimestepCutRecord(
                        timestep_index=cur_ts,
                        date=cur_date,
                        reason="TIME STEP CUT",
                        new_dt=new_dt,
                    )
                )

            damp_match = _DAMPING_RE.search(stripped)
            if damp_match:
                damp_val = float(damp_match.group(1).replace("D", "E").replace("d", "e"))
                timestep_cuts.append(
                    TimestepCutRecord(
                        timestep_index=cur_ts,
                        date=cur_date,
                        reason="DAMPING",
                        new_dt=damp_val,
                    )
                )

            # ----------------------------------------------------------
            # Runtime / warning count from summary section
            # ----------------------------------------------------------
            rt_match = _RUNTIME_RE.search(stripped)
            if rt_match:
                hours = int(rt_match.group(1))
                minutes = int(rt_match.group(2))
                seconds = float(rt_match.group(3))
                total_runtime = timedelta(hours=hours, minutes=minutes, seconds=seconds)

            wc_match = _WARNING_COUNT_RE.search(stripped)
            if wc_match:
                summary_warning_count = int(wc_match.group(1))

            i += 1

        # Flush any trailing iteration data from the last timestep
        if cur_ts > 0 and cur_max_iter > 0:
            convergence_records.append(
                ConvergenceRecord(
                    timestep_index=cur_ts,
                    date=cur_date,
                    iteration_count=cur_max_iter,
                    max_residual=cur_max_residual,
                    convergence_achieved=True,
                )
            )
        # Flush trailing tabular iteration data
        if cur_ts > 0 and cur_ts_total_iters > 0 and cur_max_iter == 0:
            convergence_records.append(
                ConvergenceRecord(
                    timestep_index=cur_ts,
                    date=cur_date,
                    iteration_count=cur_ts_total_iters,
                    max_residual=last_head_conv,
                    convergence_achieved=True,
                    bottleneck_variable=last_bottleneck_var,
                    supply_adj_count=cur_ts_supply_adjs,
                )
            )

        # Compute message counts
        warning_count = sum(1 for m in messages if m.severity == MessageSeverity.WARN)
        error_count = sum(1 for m in messages if m.severity == MessageSeverity.FATAL)

        # Use summary warning count if available and larger (some warnings
        # may be collapsed in the output)
        if summary_warning_count is not None and summary_warning_count > warning_count:
            warning_count = summary_warning_count

        # Convergence statistics
        if convergence_records:
            max_iters = max(r.iteration_count for r in convergence_records)
            avg_iters = sum(r.iteration_count for r in convergence_records) / len(
                convergence_records
            )
        else:
            max_iters = 0
            avg_iters = 0.0

        return SimulationMessagesResult(
            messages=messages,
            total_runtime=total_runtime,
            warning_count=warning_count,
            error_count=error_count,
            convergence_records=convergence_records,
            mass_balance_records=mass_balance_records,
            timestep_cuts=timestep_cuts,
            max_iterations=max_iters,
            avg_iterations=avg_iters,
        )
