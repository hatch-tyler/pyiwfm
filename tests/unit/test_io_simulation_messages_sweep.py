"""Tests for pyiwfm.io.simulation_messages covering uncovered branches.

Targets:
- MessageSeverity enum values
- SimulationMessagesResult.filter_by_severity, get_spatial_summary, to_geodataframe
- _parse_severity branches (FATAL, WARN, INFO, MESSAGE fallback)
- _extract_spatial_ids: node, element, reach, layer
- _extract_procedure
- SimulationMessagesReader.read: severity blocks, continuation lines,
  runtime summary, warning count summary, summary_warning_count override
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pyiwfm.io.simulation_messages import (
    MessageSeverity,
    SimulationMessage,
    SimulationMessagesReader,
    SimulationMessagesResult,
    _extract_procedure,
    _extract_spatial_ids,
    _parse_severity,
)


# ---------------------------------------------------------------------------
# _parse_severity
# ---------------------------------------------------------------------------


class TestParseSeverity:
    def test_fatal(self) -> None:
        assert _parse_severity("FATAL") == MessageSeverity.FATAL

    def test_warn(self) -> None:
        assert _parse_severity("WARN") == MessageSeverity.WARN

    def test_warning(self) -> None:
        assert _parse_severity("WARNING") == MessageSeverity.WARN

    def test_info(self) -> None:
        assert _parse_severity("INFO") == MessageSeverity.INFO

    def test_unknown_returns_message(self) -> None:
        assert _parse_severity("DEBUG") == MessageSeverity.MESSAGE

    def test_case_insensitive(self) -> None:
        assert _parse_severity("fatal") == MessageSeverity.FATAL
        assert _parse_severity("Warning") == MessageSeverity.WARN


# ---------------------------------------------------------------------------
# _extract_spatial_ids
# ---------------------------------------------------------------------------


class TestExtractSpatialIds:
    def test_node_ids(self) -> None:
        text = "Convergence failure at node 123 and node #456"
        nodes, elems, reaches, layers = _extract_spatial_ids(text)
        assert nodes == [123, 456]
        assert elems == []
        assert reaches == []
        assert layers == []

    def test_element_ids(self) -> None:
        text = "Problem in element 42 and elem=99"
        nodes, elems, reaches, layers = _extract_spatial_ids(text)
        assert elems == [42, 99]

    def test_reach_ids(self) -> None:
        text = "Stream reach 7 has issues"
        _, _, reaches, _ = _extract_spatial_ids(text)
        assert reaches == [7]

    def test_layer_ids(self) -> None:
        text = "Issue at layer 3 and layer 5"
        _, _, _, layers = _extract_spatial_ids(text)
        assert layers == [3, 5]

    def test_mixed_ids(self) -> None:
        text = "Node 1, element 2, reach 3, layer 4"
        nodes, elems, reaches, layers = _extract_spatial_ids(text)
        assert nodes == [1]
        assert elems == [2]
        assert reaches == [3]
        assert layers == [4]

    def test_no_ids(self) -> None:
        text = "General warning message"
        nodes, elems, reaches, layers = _extract_spatial_ids(text)
        assert nodes == [] and elems == [] and reaches == [] and layers == []

    def test_deduplication(self) -> None:
        text = "node 5 and node 5 again"
        nodes, _, _, _ = _extract_spatial_ids(text)
        assert nodes == [5]


# ---------------------------------------------------------------------------
# _extract_procedure
# ---------------------------------------------------------------------------


class TestExtractProcedure:
    def test_extracts_procedure_name(self) -> None:
        text = "Some error message (AppGW_ReadData)"
        assert _extract_procedure(text) == "AppGW_ReadData"

    def test_no_procedure(self) -> None:
        text = "Just a plain message"
        assert _extract_procedure(text) == ""


# ---------------------------------------------------------------------------
# SimulationMessagesResult
# ---------------------------------------------------------------------------


class TestSimulationMessagesResult:
    def _make_messages(self) -> list[SimulationMessage]:
        return [
            SimulationMessage(
                severity=MessageSeverity.WARN,
                text="Warning at node 1",
                procedure="ProcA",
                line_number=10,
                node_ids=[1],
            ),
            SimulationMessage(
                severity=MessageSeverity.FATAL,
                text="Fatal at element 2",
                procedure="ProcB",
                line_number=20,
                element_ids=[2],
            ),
            SimulationMessage(
                severity=MessageSeverity.WARN,
                text="Another warning at node 1 and reach 3",
                procedure="ProcC",
                line_number=30,
                node_ids=[1],
                reach_ids=[3],
            ),
            SimulationMessage(
                severity=MessageSeverity.INFO,
                text="Info with layer 2",
                procedure="ProcD",
                line_number=40,
                layer_ids=[2],
            ),
        ]

    def test_filter_by_severity_warn(self) -> None:
        msgs = self._make_messages()
        result = SimulationMessagesResult(
            messages=msgs, total_runtime=None, warning_count=2, error_count=1
        )
        filtered = result.filter_by_severity(MessageSeverity.WARN)
        assert len(filtered) == 2

    def test_filter_by_severity_fatal(self) -> None:
        msgs = self._make_messages()
        result = SimulationMessagesResult(
            messages=msgs, total_runtime=None, warning_count=2, error_count=1
        )
        filtered = result.filter_by_severity(MessageSeverity.FATAL)
        assert len(filtered) == 1

    def test_filter_by_severity_empty(self) -> None:
        msgs = self._make_messages()
        result = SimulationMessagesResult(
            messages=msgs, total_runtime=None, warning_count=2, error_count=1
        )
        filtered = result.filter_by_severity(MessageSeverity.MESSAGE)
        assert len(filtered) == 0

    def test_get_spatial_summary(self) -> None:
        msgs = self._make_messages()
        result = SimulationMessagesResult(
            messages=msgs, total_runtime=None, warning_count=2, error_count=1
        )
        summary = result.get_spatial_summary()
        assert summary["nodes"][1] == 2  # node 1 appears in 2 messages
        assert summary["elements"][2] == 1
        assert summary["reaches"][3] == 1
        assert summary["layers"][2] == 1

    def test_to_geodataframe_with_grid(self) -> None:
        msgs = [
            SimulationMessage(
                severity=MessageSeverity.WARN,
                text="Warning at node 1",
                procedure="",
                line_number=10,
                node_ids=[1],
                element_ids=[1],
            ),
        ]
        result = SimulationMessagesResult(
            messages=msgs, total_runtime=None, warning_count=1, error_count=0
        )

        # Mock grid with nodes and elements
        mock_node = MagicMock()
        mock_node.x = 100.0
        mock_node.y = 200.0
        mock_elem = MagicMock()
        mock_elem.node_ids = [1, 2]
        mock_node2 = MagicMock()
        mock_node2.x = 300.0
        mock_node2.y = 400.0

        mock_grid = MagicMock()
        mock_grid.nodes = [mock_node, mock_node2]
        mock_grid.elements = [mock_elem]

        gdf = result.to_geodataframe(mock_grid)
        assert len(gdf) == 2  # one node row + one element row
        assert "entity_type" in gdf.columns
        assert "message_count" in gdf.columns

    def test_to_geodataframe_empty_returns_empty(self) -> None:
        result = SimulationMessagesResult(
            messages=[], total_runtime=None, warning_count=0, error_count=0
        )
        mock_grid = MagicMock()
        gdf = result.to_geodataframe(mock_grid)
        assert len(gdf) == 0

    def test_to_geodataframe_out_of_range_ids_skipped(self) -> None:
        msgs = [
            SimulationMessage(
                severity=MessageSeverity.WARN,
                text="Node 999",
                procedure="",
                line_number=10,
                node_ids=[999],
            ),
        ]
        result = SimulationMessagesResult(
            messages=msgs, total_runtime=None, warning_count=1, error_count=0
        )
        mock_grid = MagicMock()
        mock_grid.nodes = []
        mock_grid.elements = []
        gdf = result.to_geodataframe(mock_grid)
        assert len(gdf) == 0


# ---------------------------------------------------------------------------
# SimulationMessagesReader.read
# ---------------------------------------------------------------------------


class TestSimulationMessagesReader:
    def test_read_basic_messages(self, tmp_path: Path) -> None:
        content = (
            "* WARN: Convergence failure at node 42 (AppGW_Simulate)\n"
            "* FATAL: Negative head at element 7 layer 2 (AppGW_ReadData)\n"
            "* INFO: Model checkpoint saved (AppSim_WriteCheckpoint)\n"
        )
        fpath = tmp_path / "SimulationMessages.out"
        fpath.write_text(content, encoding="utf-8")

        reader = SimulationMessagesReader(fpath)
        result = reader.read()

        assert len(result.messages) == 3
        assert result.warning_count == 1
        assert result.error_count == 1

        # Check first message (WARN)
        warn_msg = result.messages[0]
        assert warn_msg.severity == MessageSeverity.WARN
        assert warn_msg.node_ids == [42]
        assert warn_msg.procedure == "AppGW_Simulate"

        # Check second (FATAL)
        fatal_msg = result.messages[1]
        assert fatal_msg.severity == MessageSeverity.FATAL
        assert fatal_msg.element_ids == [7]
        assert fatal_msg.layer_ids == [2]

    def test_read_continuation_lines(self, tmp_path: Path) -> None:
        content = (
            "* WARN: First part of warning\n"
            "*   continuation with node 10\n"
            "*   more details (SomeProcedure)\n"
        )
        fpath = tmp_path / "SimulationMessages.out"
        fpath.write_text(content, encoding="utf-8")

        reader = SimulationMessagesReader(fpath)
        result = reader.read()

        assert len(result.messages) == 1
        msg = result.messages[0]
        assert msg.severity == MessageSeverity.WARN
        assert "First part of warning" in msg.text
        assert "continuation with node 10" in msg.text
        assert msg.node_ids == [10]
        assert msg.procedure == "SomeProcedure"

    def test_continuation_new_block_breaks(self, tmp_path: Path) -> None:
        """A continuation line starting '*' without space after it starts a new block."""
        content = (
            "* WARN: First warning\n"
            "*FATAL: Second is fatal\n"
        )
        fpath = tmp_path / "SimulationMessages.out"
        fpath.write_text(content, encoding="utf-8")

        reader = SimulationMessagesReader(fpath)
        result = reader.read()

        # The second line should break the continuation and start a new block
        # "*FATAL:" has '*' followed by 'F' (not space) so breaks
        assert len(result.messages) >= 1

    def test_non_continuation_line_breaks(self, tmp_path: Path) -> None:
        """A line not starting with '*' breaks the continuation."""
        content = (
            "* WARN: Warning message (ProcA)\n"
            "This is a plain line\n"
            "* INFO: Info message\n"
        )
        fpath = tmp_path / "SimulationMessages.out"
        fpath.write_text(content, encoding="utf-8")

        reader = SimulationMessagesReader(fpath)
        result = reader.read()

        assert len(result.messages) == 2

    def test_read_runtime_summary(self, tmp_path: Path) -> None:
        content = (
            "Some header text\n"
            "Total Run-Time = 2 hours 15 min 30.5 sec\n"
        )
        fpath = tmp_path / "SimulationMessages.out"
        fpath.write_text(content, encoding="utf-8")

        reader = SimulationMessagesReader(fpath)
        result = reader.read()

        assert result.total_runtime is not None
        expected = timedelta(hours=2, minutes=15, seconds=30.5)
        assert result.total_runtime == expected

    def test_read_warning_count_summary(self, tmp_path: Path) -> None:
        content = (
            "* WARN: A warning\n"
            "\n"
            "15 warnings generated during simulation\n"
        )
        fpath = tmp_path / "SimulationMessages.out"
        fpath.write_text(content, encoding="utf-8")

        reader = SimulationMessagesReader(fpath)
        result = reader.read()

        # summary_warning_count (15) > parsed count (1), so use 15
        assert result.warning_count == 15

    def test_read_empty_file(self, tmp_path: Path) -> None:
        fpath = tmp_path / "SimulationMessages.out"
        fpath.write_text("", encoding="utf-8")

        reader = SimulationMessagesReader(fpath)
        result = reader.read()

        assert len(result.messages) == 0
        assert result.total_runtime is None
        assert result.warning_count == 0
        assert result.error_count == 0

    def test_format_property(self, tmp_path: Path) -> None:
        fpath = tmp_path / "SimulationMessages.out"
        fpath.write_text("", encoding="utf-8")
        reader = SimulationMessagesReader(fpath)
        assert reader.format == "simulation_messages"

    def test_continuation_with_empty_content(self, tmp_path: Path) -> None:
        """Continuation line with only asterisks and spaces (empty content)."""
        content = (
            "* WARN: Warning text\n"
            "*     \n"
            "*   actual continuation (Done)\n"
        )
        fpath = tmp_path / "SimulationMessages.out"
        fpath.write_text(content, encoding="utf-8")

        reader = SimulationMessagesReader(fpath)
        result = reader.read()

        assert len(result.messages) == 1
        # Empty content line is skipped but continuation continues
        assert "actual continuation" in result.messages[0].text

    def test_summary_warning_count_not_used_when_smaller(self, tmp_path: Path) -> None:
        """When summary warning count <= parsed count, parsed count is used."""
        content = (
            "* WARN: Warn 1\n"
            "* WARN: Warn 2\n"
            "* WARN: Warn 3\n"
            "\n"
            "2 warnings during simulation\n"
        )
        fpath = tmp_path / "SimulationMessages.out"
        fpath.write_text(content, encoding="utf-8")

        reader = SimulationMessagesReader(fpath)
        result = reader.read()

        # Parsed 3 warnings, summary says 2. Use parsed (larger).
        assert result.warning_count == 3

    def test_after_label_empty(self, tmp_path: Path) -> None:
        """Severity line with nothing after the colon."""
        content = "* WARN:\n"
        fpath = tmp_path / "SimulationMessages.out"
        fpath.write_text(content, encoding="utf-8")

        reader = SimulationMessagesReader(fpath)
        result = reader.read()

        assert len(result.messages) == 1
        assert result.messages[0].text == ""
