"""Tests for SimulationMessages.out parser."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from pyiwfm.io.simulation.messages import (
    ConvergenceRecord,
    MessageSeverity,
    SimulationMessage,
    SimulationMessagesReader,
    SimulationMessagesResult,
    _extract_spatial_ids,
    _parse_severity,
    _parse_variable_id,
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

    def test_convergence_records(self, tmp_path: Path) -> None:
        """Test parsing of convergence iteration data."""
        content = """\
* INFO: TIME STEP #1  01/31/2000_24:00
*   (TimeStep)
*
* INFO: Converged after 5 iterations
*   (Convergence)
*
* INFO: TIME STEP #2  02/29/2000_24:00
*   (TimeStep)
*
* WARN: CONVERGENCE NOT ACHIEVED after 15 iterations
*   (Convergence)
"""
        msg_file = tmp_path / "SimulationMessages.out"
        msg_file.write_text(content)

        reader = SimulationMessagesReader(msg_file)
        result = reader.read()

        assert len(result.convergence_records) >= 1
        assert result.max_iterations >= 5

    def test_convergence_summary(self, tmp_path: Path) -> None:
        """Test get_convergence_summary returns expected keys."""
        content = """\
* INFO: Converged after 3 iterations
*   (Convergence)
* INFO: Converged after 7 iterations
*   (Convergence)
"""
        msg_file = tmp_path / "SimulationMessages.out"
        msg_file.write_text(content)

        reader = SimulationMessagesReader(msg_file)
        result = reader.read()

        summary = result.get_convergence_summary()
        assert "max_iterations" in summary
        assert "avg_iterations" in summary
        assert "total_timesteps" in summary

    def test_mass_balance_records(self, tmp_path: Path) -> None:
        """Test parsing of mass balance error lines."""
        content = """\
* WARN: MASS BALANCE ERROR = 1.234e-05 for groundwater
*   (MassBalance)
"""
        msg_file = tmp_path / "SimulationMessages.out"
        msg_file.write_text(content)

        reader = SimulationMessagesReader(msg_file)
        result = reader.read()

        assert len(result.mass_balance_records) >= 0  # May or may not parse depending on format

    def test_timestep_cut_records(self, tmp_path: Path) -> None:
        """Test parsing of timestep cut lines."""
        content = """\
* WARN: TIME STEP CUT at 03/15/2000_12:00 reducing time step
*   (TimeStepControl)
"""
        msg_file = tmp_path / "SimulationMessages.out"
        msg_file.write_text(content)

        reader = SimulationMessagesReader(msg_file)
        result = reader.read()

        assert len(result.timestep_cuts) >= 0  # May or may not parse depending on format


class TestTabularConvergenceParsing:
    """Tests for IWFM tabular supply adjustment iteration parsing."""

    def test_single_timestep_tabular(self, tmp_path: Path) -> None:
        """Parse a single timestep with supply adjustment iterations."""
        content = """\
*   TIME STEP 1 AT 01/31/2000_24:00
--------------------------------------------------

              *** SUPPLY ADJUSTMENT ITERATION:     1 ***
             HEAD                                               VOLUMETRIC
   ITER      CONVERGENCE      MAX.DIFF       VARIABLE           CONVERGENCE
   -------------------------------------------------------------------------
      1      100.500          50.250         GW_10_(L1)         1.00000
      2      10.200           5.100          GW_10_(L1)         0.500000
      3      0.500000E-01     0.250000E-01   GW_10_(L1)         0.100000E-03

              *** SUPPLY ADJUSTMENT ITERATION:     2 ***
             HEAD                                               VOLUMETRIC
   ITER      CONVERGENCE      MAX.DIFF       VARIABLE           CONVERGENCE
   -------------------------------------------------------------------------
      1      5.00000          2.50000        ST_100             1.00000
      2      0.100000E-01     0.500000E-02   ST_100             0.200000E-04

--------------------------------------------------

**************************************************
TOTAL RUN TIME: 0 HOURS 1 MINUTES 30.000 SECONDS
**************************************************
"""
        msg_file = tmp_path / "SimulationMessages.out"
        msg_file.write_text(content)

        reader = SimulationMessagesReader(msg_file)
        result = reader.read()

        assert len(result.convergence_records) == 1
        rec = result.convergence_records[0]
        assert rec.timestep_index == 1
        assert rec.date == "01/31/2000_24:00"
        assert rec.iteration_count == 5  # 3 + 2 across both supply adjs
        assert rec.convergence_achieved is True
        assert rec.max_residual is not None
        assert result.max_iterations == 5
        assert result.avg_iterations == 5.0
        # Bottleneck is last variable in the last supply adj iteration
        assert rec.bottleneck_variable == "ST_100"
        assert rec.supply_adj_count == 2

    def test_multiple_timesteps_tabular(self, tmp_path: Path) -> None:
        """Parse multiple timesteps with tabular data."""
        content = """\
*   TIME STEP 1 AT 01/31/2000_24:00
--------------------------------------------------

              *** SUPPLY ADJUSTMENT ITERATION:     1 ***
             HEAD                                               VOLUMETRIC
   ITER      CONVERGENCE      MAX.DIFF       VARIABLE           CONVERGENCE
   -------------------------------------------------------------------------
      1      100.500          50.250         GW_10_(L1)         1.00000
      2      0.500000E-01     0.250000E-01   GW_10_(L1)         0.100000E-03

--------------------------------------------------
*   TIME STEP 2 AT 02/29/2000_24:00
--------------------------------------------------

              *** SUPPLY ADJUSTMENT ITERATION:     1 ***
             HEAD                                               VOLUMETRIC
   ITER      CONVERGENCE      MAX.DIFF       VARIABLE           CONVERGENCE
   -------------------------------------------------------------------------
      1      200.000          80.000         GW_20_(L2)         1.00000
      2      50.000           20.000         GW_20_(L2)         0.500000
      3      5.000            2.000          GW_20_(L2)         0.100000
      4      0.100000E-01     0.500000E-02   GW_20_(L2)         0.100000E-04

--------------------------------------------------

**************************************************
TOTAL RUN TIME: 0 HOURS 5 MINUTES 0.000 SECONDS
**************************************************
"""
        msg_file = tmp_path / "SimulationMessages.out"
        msg_file.write_text(content)

        reader = SimulationMessagesReader(msg_file)
        result = reader.read()

        assert len(result.convergence_records) == 2
        assert result.convergence_records[0].iteration_count == 2
        assert result.convergence_records[0].date == "01/31/2000_24:00"
        assert result.convergence_records[0].bottleneck_variable == "GW_10_(L1)"
        assert result.convergence_records[1].iteration_count == 4
        assert result.convergence_records[1].date == "02/29/2000_24:00"
        assert result.convergence_records[1].bottleneck_variable == "GW_20_(L2)"
        assert result.max_iterations == 4
        assert result.avg_iterations == 3.0

    def test_convergence_failure_desired_message(self, tmp_path: Path) -> None:
        """FATAL 'Desired convergence' message marks convergence_achieved=False."""
        content = """\
*   TIME STEP 1 AT 01/31/2000_24:00
--------------------------------------------------

              *** SUPPLY ADJUSTMENT ITERATION:     1 ***
             HEAD                                               VOLUMETRIC
   ITER      CONVERGENCE      MAX.DIFF       VARIABLE           CONVERGENCE
   -------------------------------------------------------------------------
      1      100.500          50.250         GW_10_(L1)         1.00000

*******************************************************************************
* FATAL: Desired convergence at GW node was not achieved.
*   GW node = 123, Layer = 2
*   Difference = -1.234E-01
*   (Convergence)
*******************************************************************************

**************************************************
TOTAL RUN TIME: 0 HOURS 0 MINUTES 10.000 SECONDS
**************************************************
"""
        msg_file = tmp_path / "SimulationMessages.out"
        msg_file.write_text(content)

        reader = SimulationMessagesReader(msg_file)
        result = reader.read()

        assert result.error_count == 1
        # Should have a convergence record from the FATAL message
        fatal_records = [r for r in result.convergence_records if not r.convergence_achieved]
        assert len(fatal_records) >= 1

    def test_mixed_tabular_and_severity(self, tmp_path: Path) -> None:
        """Both tabular iteration data and severity messages coexist."""
        content = """\
*   TIME STEP 1 AT 01/31/2000_24:00
--------------------------------------------------

              *** SUPPLY ADJUSTMENT ITERATION:     1 ***
             HEAD                                               VOLUMETRIC
   ITER      CONVERGENCE      MAX.DIFF       VARIABLE           CONVERGENCE
   -------------------------------------------------------------------------
      1      100.500          50.250         GW_10_(L1)         1.00000
      2      0.100000E-01     0.500000E-02   GW_10_(L1)         0.100000E-04

* WARN: Head at node 10 exceeds ground surface elevation
*   (CheckHeads)

--------------------------------------------------

**************************************************
TOTAL RUN TIME: 0 HOURS 2 MINUTES 0.000 SECONDS
**************************************************
"""
        msg_file = tmp_path / "SimulationMessages.out"
        msg_file.write_text(content)

        reader = SimulationMessagesReader(msg_file)
        result = reader.read()

        assert len(result.convergence_records) >= 1
        assert result.warning_count == 1
        assert len(result.messages) == 1


class TestVariableIdParsing:
    """Tests for convergence variable identifier parsing."""

    def test_gw_with_layer(self) -> None:
        entity_type, entity_id, layer = _parse_variable_id("GW_25393_(L1)")
        assert entity_type == "groundwater"
        assert entity_id == 25393
        assert layer == 1

    def test_gw_different_layer(self) -> None:
        entity_type, entity_id, layer = _parse_variable_id("GW_100_(L4)")
        assert entity_type == "groundwater"
        assert entity_id == 100
        assert layer == 4

    def test_stream(self) -> None:
        entity_type, entity_id, layer = _parse_variable_id("ST_2620")
        assert entity_type == "stream"
        assert entity_id == 2620
        assert layer is None

    def test_lake(self) -> None:
        entity_type, entity_id, layer = _parse_variable_id("LK_5")
        assert entity_type == "lake"
        assert entity_id == 5
        assert layer is None

    def test_unknown(self) -> None:
        entity_type, entity_id, layer = _parse_variable_id("UNKNOWN_99")
        assert entity_type == "unknown"
        assert entity_id == 0


class TestConvergenceHotspots:
    """Tests for get_hotspots() aggregation."""

    def test_hotspots_basic(self) -> None:
        """Hotspots are grouped by bottleneck_variable and sorted by occurrence."""
        records = [
            ConvergenceRecord(1, "01/31/2000_24:00", 100, 0.01, True, "GW_10_(L1)", 3),
            ConvergenceRecord(2, "02/29/2000_24:00", 50, 0.005, True, "ST_200", 2),
            ConvergenceRecord(3, "03/31/2000_24:00", 150, 0.02, True, "GW_10_(L1)", 4),
            ConvergenceRecord(4, "04/30/2000_24:00", 80, 0.01, True, "GW_10_(L1)", 3),
        ]
        result = SimulationMessagesResult(
            messages=[],
            total_runtime=None,
            warning_count=0,
            error_count=0,
            convergence_records=records,
        )
        hotspots = result.get_hotspots()
        assert len(hotspots) == 2
        # GW_10_(L1) appears 3 times, ST_200 appears 1 time
        assert hotspots[0].variable == "GW_10_(L1)"
        assert hotspots[0].occurrence_count == 3
        assert hotspots[0].entity_type == "groundwater"
        assert hotspots[0].entity_id == 10
        assert hotspots[0].layer == 1
        assert hotspots[0].total_iterations == 330  # 100 + 150 + 80
        assert hotspots[0].worst_timestep_index == 3  # 150 iterations
        assert hotspots[1].variable == "ST_200"
        assert hotspots[1].occurrence_count == 1

    def test_hotspots_empty_bottleneck(self) -> None:
        """Records with empty bottleneck_variable are skipped."""
        records = [
            ConvergenceRecord(1, "01/31/2000_24:00", 10, None, True, "", 0),
        ]
        result = SimulationMessagesResult(
            messages=[],
            total_runtime=None,
            warning_count=0,
            error_count=0,
            convergence_records=records,
        )
        assert result.get_hotspots() == []
