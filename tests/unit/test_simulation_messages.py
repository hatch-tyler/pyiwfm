"""Tests for SimulationMessages.out parser."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from pyiwfm.io.simulation_messages import (
    MessageSeverity,
    SimulationMessage,
    SimulationMessagesReader,
    SimulationMessagesResult,
    _extract_spatial_ids,
    _parse_severity,
)


class TestSeverityParsing:
    """Tests for severity label parsing."""

    def test_fatal(self) -> None:
        assert _parse_severity("FATAL") == MessageSeverity.FATAL

    def test_warn(self) -> None:
        assert _parse_severity("WARN") == MessageSeverity.WARN
        assert _parse_severity("WARNING") == MessageSeverity.WARN

    def test_info(self) -> None:
        assert _parse_severity("INFO") == MessageSeverity.INFO

    def test_case_insensitive(self) -> None:
        assert _parse_severity("fatal") == MessageSeverity.FATAL
        assert _parse_severity("Warn") == MessageSeverity.WARN


class TestSpatialExtraction:
    """Tests for spatial ID extraction from message text."""

    def test_node_ids(self) -> None:
        text = "Head at node 123 exceeds surface at node #456"
        nodes, elems, reaches, layers = _extract_spatial_ids(text)
        assert nodes == [123, 456]
        assert elems == []

    def test_element_ids(self) -> None:
        text = "Element 42 has negative storage. Element=99 also."
        nodes, elems, reaches, layers = _extract_spatial_ids(text)
        assert elems == [42, 99]

    def test_reach_ids(self) -> None:
        text = "Reach 7 flow exceeded capacity"
        _, _, reaches, _ = _extract_spatial_ids(text)
        assert reaches == [7]

    def test_layer_ids(self) -> None:
        text = "Dry cell at node 10, layer 3"
        nodes, _, _, layers = _extract_spatial_ids(text)
        assert nodes == [10]
        assert layers == [3]

    def test_no_spatial_ids(self) -> None:
        text = "Generic warning about convergence"
        nodes, elems, reaches, layers = _extract_spatial_ids(text)
        assert nodes == []
        assert elems == []
        assert reaches == []
        assert layers == []

    def test_duplicate_ids_deduplicated(self) -> None:
        text = "node 5 and node 5 again"
        nodes, _, _, _ = _extract_spatial_ids(text)
        assert nodes == [5]


class TestSimulationMessagesReader:
    """Tests for the full SimulationMessages.out parser."""

    def test_parse_warnings(self, tmp_path: Path) -> None:
        content = """\
*
* IWFM Simulation Messages
*
* WARN: Head at node 10 exceeds ground surface elevation
*   at layer 2. Check boundary conditions. (CheckHeads)
*
* WARN: Element 55 has negative groundwater storage
*   (CheckStorage)
*
"""
        msg_file = tmp_path / "SimulationMessages.out"
        msg_file.write_text(content)

        reader = SimulationMessagesReader(msg_file)
        result = reader.read()

        assert result.warning_count == 2
        assert result.error_count == 0
        assert len(result.messages) == 2

        # First warning
        msg0 = result.messages[0]
        assert msg0.severity == MessageSeverity.WARN
        assert 10 in msg0.node_ids
        assert 2 in msg0.layer_ids
        assert msg0.procedure == "CheckHeads"

        # Second warning
        msg1 = result.messages[1]
        assert 55 in msg1.element_ids
        assert msg1.procedure == "CheckStorage"

    def test_parse_fatal(self, tmp_path: Path) -> None:
        content = """\
* FATAL: Simulation failed at node 99 layer 1
*   (SimulationMain)
"""
        msg_file = tmp_path / "SimulationMessages.out"
        msg_file.write_text(content)

        reader = SimulationMessagesReader(msg_file)
        result = reader.read()

        assert result.error_count == 1
        assert result.messages[0].severity == MessageSeverity.FATAL
        assert 99 in result.messages[0].node_ids

    def test_parse_runtime(self, tmp_path: Path) -> None:
        content = """\
* INFO: Simulation started
*   (Main)
*
Total run time = 2 hours 30 min 15.5 sec
"""
        msg_file = tmp_path / "SimulationMessages.out"
        msg_file.write_text(content)

        reader = SimulationMessagesReader(msg_file)
        result = reader.read()

        assert result.total_runtime is not None
        expected = timedelta(hours=2, minutes=30, seconds=15.5)
        assert result.total_runtime == expected

    def test_filter_by_severity(self) -> None:
        messages = [
            SimulationMessage(
                severity=MessageSeverity.WARN,
                text="w1",
                procedure="",
                line_number=1,
            ),
            SimulationMessage(
                severity=MessageSeverity.FATAL,
                text="e1",
                procedure="",
                line_number=2,
            ),
            SimulationMessage(
                severity=MessageSeverity.WARN,
                text="w2",
                procedure="",
                line_number=3,
            ),
        ]
        result = SimulationMessagesResult(
            messages=messages,
            total_runtime=None,
            warning_count=2,
            error_count=1,
        )

        warns = result.filter_by_severity(MessageSeverity.WARN)
        assert len(warns) == 2
        fatals = result.filter_by_severity(MessageSeverity.FATAL)
        assert len(fatals) == 1

    def test_spatial_summary(self) -> None:
        messages = [
            SimulationMessage(
                severity=MessageSeverity.WARN,
                text="",
                procedure="",
                line_number=1,
                node_ids=[10, 20],
            ),
            SimulationMessage(
                severity=MessageSeverity.WARN,
                text="",
                procedure="",
                line_number=2,
                node_ids=[10],
                element_ids=[5],
            ),
        ]
        result = SimulationMessagesResult(
            messages=messages,
            total_runtime=None,
            warning_count=2,
            error_count=0,
        )

        summary = result.get_spatial_summary()
        assert summary["nodes"][10] == 2
        assert summary["nodes"][20] == 1
        assert summary["elements"][5] == 1

    def test_empty_file(self, tmp_path: Path) -> None:
        msg_file = tmp_path / "empty.out"
        msg_file.write_text("")

        reader = SimulationMessagesReader(msg_file)
        result = reader.read()

        assert len(result.messages) == 0
        assert result.warning_count == 0
        assert result.error_count == 0
        assert result.total_runtime is None
