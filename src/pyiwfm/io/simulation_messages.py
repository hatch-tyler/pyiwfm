"""
Parser for IWFM SimulationMessages.out files.

SimulationMessages.out is produced by IWFM during simulation and contains
warnings, errors, and informational messages with spatial context (node IDs,
element IDs, reach IDs, layer IDs).

Example
-------
>>> from pyiwfm.io.simulation_messages import SimulationMessagesReader
>>> reader = SimulationMessagesReader("SimulationMessages.out")
>>> result = reader.read()
>>> print(f"Warnings: {result.warning_count}, Errors: {result.error_count}")
>>> for msg in result.filter_by_severity(MessageSeverity.WARN):
...     print(msg.text[:80])
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
    """

    messages: list[SimulationMessage]
    total_runtime: timedelta | None
    warning_count: int
    error_count: int

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
            Parsed messages with spatial context and summary statistics.
        """
        messages: list[SimulationMessage] = []
        total_runtime: timedelta | None = None
        summary_warning_count: int | None = None

        with open(self.filepath, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.rstrip("\n\r")

            # Check for severity block start
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
                        # New severity block or separator
                        break
                    if cont.startswith("*"):
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
                continue

            # Check for runtime in summary section
            rt_match = _RUNTIME_RE.search(stripped)
            if rt_match:
                hours = int(rt_match.group(1))
                minutes = int(rt_match.group(2))
                seconds = float(rt_match.group(3))
                total_runtime = timedelta(hours=hours, minutes=minutes, seconds=seconds)

            # Check for warning count in summary
            wc_match = _WARNING_COUNT_RE.search(stripped)
            if wc_match:
                summary_warning_count = int(wc_match.group(1))

            i += 1

        # Compute counts from parsed messages
        warning_count = sum(1 for m in messages if m.severity == MessageSeverity.WARN)
        error_count = sum(1 for m in messages if m.severity == MessageSeverity.FATAL)

        # Use summary warning count if available and larger (some warnings
        # may be collapsed in the output)
        if summary_warning_count is not None and summary_warning_count > warning_count:
            warning_count = summary_warning_count

        return SimulationMessagesResult(
            messages=messages,
            total_runtime=total_runtime,
            warning_count=warning_count,
            error_count=error_count,
        )
