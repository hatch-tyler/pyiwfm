"""Tests for stream sub-writer modules.

Covers:
- pyiwfm.io.stream_bypass_writer.write_bypass_spec
- pyiwfm.io.stream_diversion_writer.write_diversion_spec
- pyiwfm.io.stream_inflow_writer.write_stream_inflow

Each writer is exercised with zero-count configs, populated configs with
various field combinations, and output text verification.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pyiwfm.io.stream_bypass import (
    BypassRatingTable,
    BypassSeepageZone,
    BypassSpec,
    BypassSpecConfig,
)
from pyiwfm.io.stream_bypass_writer import write_bypass_spec
from pyiwfm.io.stream_diversion import (
    DiversionSpec,
    DiversionSpecConfig,
    ElementGroup,
    RechargeZoneDest,
)
from pyiwfm.io.stream_diversion_writer import write_diversion_spec
from pyiwfm.io.stream_inflow import (
    InflowConfig,
    InflowSpec,
)
from pyiwfm.io.stream_inflow_writer import write_stream_inflow

# ===========================================================================
# write_bypass_spec
# ===========================================================================


class TestWriteBypassSpecZero:
    def test_zero_bypasses(self, tmp_path: Path) -> None:
        """Zero bypasses writes only the comment header and NBypass=0."""
        cfg = BypassSpecConfig(n_bypasses=0)
        result = write_bypass_spec(cfg, tmp_path / "bypass.dat")
        assert result.exists()
        text = result.read_text()
        assert "C  IWFM Bypass Specification File" in text
        assert "/ NBypass" in text
        lines = text.strip().splitlines()
        # Only comment + NBypass line
        assert len(lines) == 2

    def test_returns_path_object(self, tmp_path: Path) -> None:
        cfg = BypassSpecConfig(n_bypasses=0)
        result = write_bypass_spec(cfg, str(tmp_path / "bypass.dat"))
        assert isinstance(result, Path)

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        cfg = BypassSpecConfig(n_bypasses=0)
        result = write_bypass_spec(cfg, tmp_path / "a" / "b" / "bypass.dat")
        assert result.exists()


class TestWriteBypassSpecPositiveCol:
    """Bypasses with positive rating_table_col (no inline table)."""

    def test_single_bypass(self, tmp_path: Path) -> None:
        bypass = BypassSpec(
            id=1,
            export_stream_node=10,
            dest_type=1,
            dest_id=20,
            rating_table_col=3,
            frac_recoverable=0.1,
            frac_non_recoverable=0.05,
            name="BP1",
        )
        sz = BypassSeepageZone(
            bypass_id=1,
            n_elements=0,
            element_ids=[],
            element_fractions=[],
        )
        cfg = BypassSpecConfig(
            n_bypasses=1,
            flow_factor=1.0,
            flow_time_unit="1DAY",
            bypass_factor=1.0,
            bypass_time_unit="1DAY",
            bypasses=[bypass],
            seepage_zones=[sz],
        )
        result = write_bypass_spec(cfg, tmp_path / "bypass.dat")
        text = result.read_text()
        # Verify fields appear
        assert "/ NBypass" in text
        assert "/ Flow conversion factor" in text
        assert "/ Flow time unit" in text
        assert "/ Bypass conversion factor" in text
        assert "/ Bypass time unit" in text
        assert "BP1" in text
        # Bypass line contains key fields
        assert "10" in text  # export_stream_node
        assert "20" in text  # dest_id
        assert "0.100000" in text  # frac_recoverable
        assert "0.050000" in text  # frac_non_recoverable

    def test_seepage_zone_no_elements(self, tmp_path: Path) -> None:
        """Seepage zone with n_elements=0 writes 'ID 0 0 0.000000'."""
        bypass = BypassSpec(id=5, export_stream_node=1, dest_type=1, dest_id=2, rating_table_col=1)
        sz = BypassSeepageZone(bypass_id=5, n_elements=0)
        cfg = BypassSpecConfig(
            n_bypasses=1,
            flow_factor=1.0,
            flow_time_unit="1DAY",
            bypass_factor=1.0,
            bypass_time_unit="1DAY",
            bypasses=[bypass],
            seepage_zones=[sz],
        )
        result = write_bypass_spec(cfg, tmp_path / "bypass.dat")
        text = result.read_text()
        # The seepage zone line should have bypass_id, 0, 0, 0.000000
        lines = text.splitlines()
        last_line = lines[-1]
        parts = last_line.split()
        assert int(parts[0]) == 5
        assert int(parts[1]) == 0


class TestWriteBypassSpecInlineRating:
    """Bypasses with negative rating_table_col (inline rating table)."""

    def test_inline_rating_table(self, tmp_path: Path) -> None:
        rating = BypassRatingTable(
            flows=np.array([0.0, 100.0, 200.0]),
            fractions=np.array([0.0, 0.5, 1.0]),
        )
        bypass = BypassSpec(
            id=1,
            export_stream_node=5,
            dest_type=2,
            dest_id=3,
            rating_table_col=-3,  # 3 inline points
            frac_recoverable=0.0,
            frac_non_recoverable=0.0,
            inline_rating=rating,
        )
        sz = BypassSeepageZone(bypass_id=1, n_elements=0)
        cfg = BypassSpecConfig(
            n_bypasses=1,
            flow_factor=2.0,
            flow_time_unit="1MON",
            bypass_factor=1.0,
            bypass_time_unit="1MON",
            bypasses=[bypass],
            seepage_zones=[sz],
        )
        result = write_bypass_spec(cfg, tmp_path / "bypass.dat")
        text = result.read_text()
        # Rating table col appears as -3
        assert "-3" in text
        # Flow values are divided by flow_factor (2.0):
        # 0.0/2.0=0.0, 100.0/2.0=50.0, 200.0/2.0=100.0
        assert "50.0000" in text
        assert "100.0000" in text
        # Fractions written as-is
        assert "0.500000" in text
        assert "1.000000" in text

    def test_inline_rating_flow_factor_zero(self, tmp_path: Path) -> None:
        """When flow_factor=0, fallback to 1.0 for division."""
        rating = BypassRatingTable(
            flows=np.array([50.0]),
            fractions=np.array([0.8]),
        )
        bypass = BypassSpec(
            id=1,
            export_stream_node=1,
            dest_type=1,
            dest_id=2,
            rating_table_col=-1,
            inline_rating=rating,
        )
        sz = BypassSeepageZone(bypass_id=1, n_elements=0)
        cfg = BypassSpecConfig(
            n_bypasses=1,
            flow_factor=0.0,  # zero -> fallback
            flow_time_unit="1DAY",
            bypass_factor=1.0,
            bypass_time_unit="1DAY",
            bypasses=[bypass],
            seepage_zones=[sz],
        )
        result = write_bypass_spec(cfg, tmp_path / "bypass.dat")
        text = result.read_text()
        # flow_factor=0 means flow_factor used for division is still 0.0
        # in the code: flow_factor = config.flow_factor if != 0.0 else 1.0
        # so 50.0 / 1.0 = 50.0
        assert "50.0000" in text
        assert "0.800000" in text


class TestWriteBypassSpecSeepageZones:
    def test_seepage_zone_with_elements(self, tmp_path: Path) -> None:
        bypass = BypassSpec(
            id=1,
            export_stream_node=1,
            dest_type=1,
            dest_id=2,
            rating_table_col=0,
        )
        sz = BypassSeepageZone(
            bypass_id=1,
            n_elements=3,
            element_ids=[10, 20, 30],
            element_fractions=[0.5, 0.3, 0.2],
        )
        cfg = BypassSpecConfig(
            n_bypasses=1,
            flow_factor=1.0,
            flow_time_unit="1DAY",
            bypass_factor=1.0,
            bypass_time_unit="1DAY",
            bypasses=[bypass],
            seepage_zones=[sz],
        )
        result = write_bypass_spec(cfg, tmp_path / "bypass.dat")
        text = result.read_text()
        # Header line: 1 3 10 0.500000
        assert "0.500000" in text
        # Remaining elements: 20 0.300000 and 30 0.200000
        assert "0.300000" in text
        assert "0.200000" in text

    def test_multiple_bypasses_with_seepage(self, tmp_path: Path) -> None:
        bypasses = [
            BypassSpec(id=1, export_stream_node=1, dest_type=1, dest_id=2, rating_table_col=0),
            BypassSpec(id=2, export_stream_node=3, dest_type=2, dest_id=1, rating_table_col=0),
        ]
        szs = [
            BypassSeepageZone(bypass_id=1, n_elements=1, element_ids=[5], element_fractions=[1.0]),
            BypassSeepageZone(bypass_id=2, n_elements=0),
        ]
        cfg = BypassSpecConfig(
            n_bypasses=2,
            flow_factor=1.0,
            flow_time_unit="1DAY",
            bypass_factor=1.0,
            bypass_time_unit="1DAY",
            bypasses=bypasses,
            seepage_zones=szs,
        )
        result = write_bypass_spec(cfg, tmp_path / "bypass.dat")
        text = result.read_text()
        # Both bypass IDs present
        lines = text.splitlines()
        all_text = " ".join(lines)
        assert "1" in all_text
        assert "2" in all_text


# ===========================================================================
# write_diversion_spec
# ===========================================================================


class TestWriteDiversionSpecZero:
    def test_zero_diversions(self, tmp_path: Path) -> None:
        cfg = DiversionSpecConfig(n_diversions=0)
        result = write_diversion_spec(cfg, tmp_path / "diver.dat")
        assert result.exists()
        text = result.read_text()
        assert "C  IWFM Diversion Specification File" in text
        assert "/ NDiver" in text
        lines = text.strip().splitlines()
        assert len(lines) == 2

    def test_returns_path_object(self, tmp_path: Path) -> None:
        cfg = DiversionSpecConfig(n_diversions=0)
        result = write_diversion_spec(cfg, str(tmp_path / "diver.dat"))
        assert isinstance(result, Path)

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        cfg = DiversionSpecConfig(n_diversions=0)
        result = write_diversion_spec(cfg, tmp_path / "x" / "y" / "diver.dat")
        assert result.exists()


class TestWriteDiversionSpec14Col:
    """14-column format (without spill fields)."""

    def test_single_diversion_no_spills(self, tmp_path: Path) -> None:
        div = DiversionSpec(
            id=1,
            stream_node=10,
            max_diver_col=2,
            frac_max_diver=1.0,
            recv_loss_col=0,
            frac_recv_loss=0.0,
            non_recv_loss_col=0,
            frac_non_recv_loss=0.0,
            dest_type=1,
            dest_id=5,
            delivery_col=3,
            frac_delivery=1.0,
            irrig_frac_col=4,
            adjustment_col=0,
            name="DIV_A",
        )
        rz = RechargeZoneDest(diversion_id=1, n_zones=0)
        cfg = DiversionSpecConfig(
            n_diversions=1,
            diversions=[div],
            n_element_groups=0,
            element_groups=[],
            recharge_zones=[rz],
            spill_zones=[],
            has_spills=False,
        )
        result = write_diversion_spec(cfg, tmp_path / "diver.dat")
        text = result.read_text()
        assert "/ NDiver" in text
        assert "/ NGroups" in text
        assert "DIV_A" in text
        # Key field values
        assert "10" in text  # stream_node
        assert "1.0000" in text  # frac_max_diver

    def test_diversion_line_14_columns(self, tmp_path: Path) -> None:
        """Verify exact 14-column layout (no spill cols)."""
        div = DiversionSpec(
            id=1,
            stream_node=2,
            max_diver_col=3,
            frac_max_diver=0.5,
            recv_loss_col=4,
            frac_recv_loss=0.1,
            non_recv_loss_col=5,
            frac_non_recv_loss=0.2,
            dest_type=1,
            dest_id=6,
            delivery_col=7,
            frac_delivery=0.9,
            irrig_frac_col=8,
            adjustment_col=9,
        )
        rz = RechargeZoneDest(diversion_id=1, n_zones=0)
        cfg = DiversionSpecConfig(
            n_diversions=1,
            diversions=[div],
            n_element_groups=0,
            element_groups=[],
            recharge_zones=[rz],
            has_spills=False,
        )
        result = write_diversion_spec(cfg, tmp_path / "diver.dat")
        lines = result.read_text().splitlines()
        # Find diversion data line (after NDiver)
        div_line = None
        for i, ln in enumerate(lines):
            if "/ NDiver" in ln:
                # next non-comment line is the diversion
                div_line = lines[i + 1]
                break
        assert div_line is not None
        parts = div_line.split()
        # 14 numeric fields (no name since it's empty)
        assert len(parts) == 14
        assert parts[0] == "1"  # id
        assert parts[1] == "2"  # stream_node
        assert parts[2] == "3"  # max_diver_col
        assert parts[3] == "0.5000"  # frac_max_diver
        assert parts[10] == "7"  # delivery_col
        assert parts[13] == "9"  # adjustment_col


class TestWriteDiversionSpec16Col:
    """16-column format (with spill fields)."""

    def test_with_spill_fields(self, tmp_path: Path) -> None:
        div = DiversionSpec(
            id=1,
            stream_node=10,
            max_diver_col=2,
            frac_max_diver=1.0,
            recv_loss_col=0,
            frac_recv_loss=0.0,
            non_recv_loss_col=0,
            frac_non_recv_loss=0.0,
            spill_col=5,
            frac_spill=0.25,
            dest_type=1,
            dest_id=3,
            delivery_col=4,
            frac_delivery=0.75,
            irrig_frac_col=6,
            adjustment_col=7,
        )
        rz = RechargeZoneDest(diversion_id=1, n_zones=0)
        sz = RechargeZoneDest(diversion_id=1, n_zones=0)
        cfg = DiversionSpecConfig(
            n_diversions=1,
            diversions=[div],
            n_element_groups=0,
            element_groups=[],
            recharge_zones=[rz],
            spill_zones=[sz],
            has_spills=True,
        )
        result = write_diversion_spec(cfg, tmp_path / "diver.dat")
        text = result.read_text()
        # Spill fields present
        assert "0.2500" in text  # frac_spill
        # Spill zone section written
        lines = text.splitlines()
        # Count non-comment lines after NGroups
        in_post_group = False
        recharge_zone_lines = []
        for ln in lines:
            if "/ NGroups" in ln:
                in_post_group = True
                continue
            if in_post_group and ln.strip() and not ln.startswith("C"):
                recharge_zone_lines.append(ln)
        # 1 recharge zone + 1 spill zone = 2 lines
        assert len(recharge_zone_lines) == 2

    def test_16_col_line_format(self, tmp_path: Path) -> None:
        """Verify 16-column line includes spill_col and frac_spill."""
        div = DiversionSpec(
            id=1,
            stream_node=2,
            max_diver_col=3,
            frac_max_diver=0.5,
            recv_loss_col=4,
            frac_recv_loss=0.1,
            non_recv_loss_col=5,
            frac_non_recv_loss=0.2,
            spill_col=11,
            frac_spill=0.3,
            dest_type=1,
            dest_id=6,
            delivery_col=7,
            frac_delivery=0.9,
            irrig_frac_col=8,
            adjustment_col=9,
        )
        rz = RechargeZoneDest(diversion_id=1, n_zones=0)
        sz = RechargeZoneDest(diversion_id=1, n_zones=0)
        cfg = DiversionSpecConfig(
            n_diversions=1,
            diversions=[div],
            n_element_groups=0,
            element_groups=[],
            recharge_zones=[rz],
            spill_zones=[sz],
            has_spills=True,
        )
        result = write_diversion_spec(cfg, tmp_path / "diver.dat")
        lines = result.read_text().splitlines()
        div_line = None
        for i, ln in enumerate(lines):
            if "/ NDiver" in ln:
                div_line = lines[i + 1]
                break
        assert div_line is not None
        parts = div_line.split()
        # 16 columns (no name)
        assert len(parts) == 16
        assert parts[8] == "11"  # spill_col
        assert parts[9] == "0.3000"  # frac_spill


class TestWriteDiversionElementGroups:
    def test_element_group_with_elements(self, tmp_path: Path) -> None:
        div = DiversionSpec(id=1, stream_node=1, dest_type=4, dest_id=1)
        group = ElementGroup(id=1, elements=[10, 20, 30])
        rz = RechargeZoneDest(diversion_id=1, n_zones=0)
        cfg = DiversionSpecConfig(
            n_diversions=1,
            diversions=[div],
            n_element_groups=1,
            element_groups=[group],
            recharge_zones=[rz],
            has_spills=False,
        )
        result = write_diversion_spec(cfg, tmp_path / "diver.dat")
        text = result.read_text()
        # Group header: ID 3 first_elem
        assert "10" in text  # first element on header
        assert "20" in text  # remaining elements
        assert "30" in text

    def test_empty_element_group(self, tmp_path: Path) -> None:
        div = DiversionSpec(id=1, stream_node=1)
        group = ElementGroup(id=1, elements=[])
        rz = RechargeZoneDest(diversion_id=1, n_zones=0)
        cfg = DiversionSpecConfig(
            n_diversions=1,
            diversions=[div],
            n_element_groups=1,
            element_groups=[group],
            recharge_zones=[rz],
            has_spills=False,
        )
        result = write_diversion_spec(cfg, tmp_path / "diver.dat")
        text = result.read_text()
        # Empty group: "1  0"
        assert "/ NGroups" in text

    def test_multiple_element_groups(self, tmp_path: Path) -> None:
        div = DiversionSpec(id=1, stream_node=1)
        groups = [
            ElementGroup(id=1, elements=[100]),
            ElementGroup(id=2, elements=[200, 300]),
        ]
        rz = RechargeZoneDest(diversion_id=1, n_zones=0)
        cfg = DiversionSpecConfig(
            n_diversions=1,
            diversions=[div],
            n_element_groups=2,
            element_groups=groups,
            recharge_zones=[rz],
            has_spills=False,
        )
        result = write_diversion_spec(cfg, tmp_path / "diver.dat")
        text = result.read_text()
        assert "100" in text
        assert "200" in text
        assert "300" in text


class TestWriteDiversionRechargeZones:
    def test_recharge_zone_with_zones(self, tmp_path: Path) -> None:
        div = DiversionSpec(id=1, stream_node=1)
        rz = RechargeZoneDest(
            diversion_id=1,
            n_zones=2,
            zone_ids=[10, 20],
            zone_fractions=[0.6, 0.4],
        )
        cfg = DiversionSpecConfig(
            n_diversions=1,
            diversions=[div],
            n_element_groups=0,
            element_groups=[],
            recharge_zones=[rz],
            has_spills=False,
        )
        result = write_diversion_spec(cfg, tmp_path / "diver.dat")
        text = result.read_text()
        assert "0.600000" in text
        assert "0.400000" in text
        assert "10" in text
        assert "20" in text

    def test_recharge_zone_empty(self, tmp_path: Path) -> None:
        div = DiversionSpec(id=1, stream_node=1)
        rz = RechargeZoneDest(diversion_id=1, n_zones=0)
        cfg = DiversionSpecConfig(
            n_diversions=1,
            diversions=[div],
            n_element_groups=0,
            element_groups=[],
            recharge_zones=[rz],
            has_spills=False,
        )
        result = write_diversion_spec(cfg, tmp_path / "diver.dat")
        text = result.read_text()
        lines = text.splitlines()
        last_line = lines[-1]
        parts = last_line.split()
        assert int(parts[0]) == 1  # diversion_id
        assert int(parts[1]) == 0  # n_zones


class TestWriteDiversionSpecMultiple:
    def test_multiple_diversions(self, tmp_path: Path) -> None:
        divs = [
            DiversionSpec(id=1, stream_node=5, name="D1"),
            DiversionSpec(id=2, stream_node=10, name="D2"),
            DiversionSpec(id=3, stream_node=15, name="D3"),
        ]
        rzs = [
            RechargeZoneDest(diversion_id=1, n_zones=0),
            RechargeZoneDest(diversion_id=2, n_zones=0),
            RechargeZoneDest(diversion_id=3, n_zones=0),
        ]
        cfg = DiversionSpecConfig(
            n_diversions=3,
            diversions=divs,
            n_element_groups=0,
            element_groups=[],
            recharge_zones=rzs,
            has_spills=False,
        )
        result = write_diversion_spec(cfg, tmp_path / "diver.dat")
        text = result.read_text()
        assert "D1" in text
        assert "D2" in text
        assert "D3" in text


# ===========================================================================
# write_stream_inflow
# ===========================================================================


class TestWriteStreamInflowZero:
    def test_zero_inflows(self, tmp_path: Path) -> None:
        cfg = InflowConfig(
            conversion_factor=1.0,
            time_unit="1DAY",
            n_inflows=0,
        )
        result = write_stream_inflow(cfg, tmp_path / "inflow.dat")
        assert result.exists()
        text = result.read_text()
        assert "C  IWFM Stream Inflow File" in text
        assert "/ Conversion factor" in text
        assert "/ Time unit" in text
        assert "/ NInflow" in text
        lines = text.strip().splitlines()
        # Comment + 3 scalar lines (factor, time_unit, n_inflows)
        assert len(lines) == 4

    def test_returns_path_object(self, tmp_path: Path) -> None:
        cfg = InflowConfig(n_inflows=0)
        result = write_stream_inflow(cfg, str(tmp_path / "inflow.dat"))
        assert isinstance(result, Path)

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        cfg = InflowConfig(n_inflows=0)
        result = write_stream_inflow(cfg, tmp_path / "deep" / "inflow.dat")
        assert result.exists()


class TestWriteStreamInflowPopulated:
    def test_single_inflow(self, tmp_path: Path) -> None:
        specs = [InflowSpec(inflow_id=1, stream_node=42)]
        cfg = InflowConfig(
            conversion_factor=1.0,
            time_unit="1DAY",
            n_inflows=1,
            inflow_specs=specs,
        )
        result = write_stream_inflow(cfg, tmp_path / "inflow.dat")
        text = result.read_text()
        assert "42" in text
        lines = text.strip().splitlines()
        # Comment + factor + time_unit + n_inflows + 1 spec = 5
        assert len(lines) == 5
        # Last line is the spec: InflowID StreamNodeID
        last = lines[-1].split()
        assert int(last[0]) == 1
        assert int(last[1]) == 42

    def test_multiple_inflows(self, tmp_path: Path) -> None:
        specs = [
            InflowSpec(inflow_id=1, stream_node=10),
            InflowSpec(inflow_id=2, stream_node=20),
            InflowSpec(inflow_id=3, stream_node=30),
        ]
        cfg = InflowConfig(
            conversion_factor=2.5,
            time_unit="1MON",
            n_inflows=3,
            inflow_specs=specs,
        )
        result = write_stream_inflow(cfg, tmp_path / "inflow.dat")
        text = result.read_text()
        assert "2.5" in text
        assert "1MON" in text
        lines = text.strip().splitlines()
        # Comment + factor + time_unit + n_inflows + 3 specs = 7
        assert len(lines) == 7
        # Verify each spec line
        spec_lines = lines[-3:]
        for i, ln in enumerate(spec_lines):
            parts = ln.split()
            assert int(parts[0]) == i + 1
            assert int(parts[1]) == (i + 1) * 10

    def test_conversion_factor_written(self, tmp_path: Path) -> None:
        cfg = InflowConfig(
            conversion_factor=0.028317,
            time_unit="1DAY",
            n_inflows=0,
        )
        result = write_stream_inflow(cfg, tmp_path / "inflow.dat")
        text = result.read_text()
        assert "0.028317" in text

    def test_negative_inflows_skips_specs(self, tmp_path: Path) -> None:
        """Negative n_inflows should still write header but no specs."""
        cfg = InflowConfig(
            conversion_factor=1.0,
            time_unit="1DAY",
            n_inflows=-1,
        )
        result = write_stream_inflow(cfg, tmp_path / "inflow.dat")
        text = result.read_text()
        lines = text.strip().splitlines()
        # Comment + factor + time_unit + n_inflows = 4 lines (no specs)
        assert len(lines) == 4


class TestWriteStreamInflowFormat:
    """Verify exact output line formatting."""

    def test_spec_line_format(self, tmp_path: Path) -> None:
        specs = [InflowSpec(inflow_id=123, stream_node=456)]
        cfg = InflowConfig(
            conversion_factor=1.0,
            time_unit="1DAY",
            n_inflows=1,
            inflow_specs=specs,
        )
        result = write_stream_inflow(cfg, tmp_path / "inflow.dat")
        lines = result.read_text().splitlines()
        last = lines[-1]
        parts = last.split()
        assert parts[0] == "123"
        assert parts[1] == "456"

    def test_large_inflow_id(self, tmp_path: Path) -> None:
        specs = [InflowSpec(inflow_id=999999, stream_node=888888)]
        cfg = InflowConfig(
            conversion_factor=1.0,
            time_unit="1DAY",
            n_inflows=1,
            inflow_specs=specs,
        )
        result = write_stream_inflow(cfg, tmp_path / "inflow.dat")
        text = result.read_text()
        assert "999999" in text
        assert "888888" in text
