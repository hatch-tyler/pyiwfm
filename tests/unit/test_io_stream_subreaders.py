"""Tests for stream sub-file readers.

Covers:
- InflowReader / read_stream_inflow
- BypassSpecReader / read_bypass_spec
- DiversionSpecReader / read_diversion_spec
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyiwfm.io.stream_inflow import (
    InflowReader,
    InflowConfig,
    InflowSpec,
    read_stream_inflow,
)
from pyiwfm.io.stream_bypass import (
    BypassSpecReader,
    BypassSpecConfig,
    BypassSpec,
    BypassRatingTable,
    BypassSeepageZone,
    read_bypass_spec,
    BYPASS_DEST_STREAM,
    BYPASS_DEST_LAKE,
)
from pyiwfm.io.stream_diversion import (
    DiversionSpecReader,
    DiversionSpecConfig,
    DiversionSpec,
    ElementGroup,
    RechargeZoneDest,
    read_diversion_spec,
    DEST_ELEMENT,
    DEST_SUBREGION,
    DEST_OUTSIDE,
)
from pyiwfm.core.exceptions import FileFormatError


# =============================================================================
# InflowReader Tests
# =============================================================================


class TestInflowReader:
    """Tests for InflowReader."""

    def _write_inflow_file(self, path: Path, content: str) -> Path:
        filepath = path / "inflow.dat"
        filepath.write_text(content)
        return filepath

    def test_read_basic(self, tmp_path: Path) -> None:
        """Read basic inflow file with conversion factor and node mapping."""
        content = (
            "C Stream Inflow File\n"
            "C\n"
            "    1.0                            # Conversion factor\n"
            "    1DAY                           # Time unit\n"
            "    3                              / NInflow\n"
            "    1    5\n"
            "    2    10\n"
            "    3    15\n"
        )
        filepath = self._write_inflow_file(tmp_path, content)
        config = InflowReader().read(filepath)

        assert config.conversion_factor == pytest.approx(1.0)
        assert config.time_unit == "1DAY"
        assert config.n_inflows == 3
        assert len(config.inflow_specs) == 3

        assert config.inflow_specs[0].inflow_id == 1
        assert config.inflow_specs[0].stream_node == 5
        assert config.inflow_specs[2].stream_node == 15

    def test_read_single_column_format(self, tmp_path: Path) -> None:
        """Read inflow file with single-column format (node ID only)."""
        content = (
            "    1.0                            # Conversion factor\n"
            "    1MON                           # Time unit\n"
            "    2                              / NInflow\n"
            "    5\n"
            "    10\n"
        )
        filepath = self._write_inflow_file(tmp_path, content)
        config = InflowReader().read(filepath)

        assert config.n_inflows == 2
        assert config.inflow_specs[0].inflow_id == 1  # auto-numbered
        assert config.inflow_specs[0].stream_node == 5
        assert config.inflow_specs[1].inflow_id == 2
        assert config.inflow_specs[1].stream_node == 10

    def test_read_with_conversion_factor(self, tmp_path: Path) -> None:
        """Read inflow file with non-unit conversion factor."""
        content = (
            "    0.0283168                      # CFS to CMS\n"
            "    1DAY                           # Time unit\n"
            "    1                              / NInflow\n"
            "    1    5\n"
        )
        filepath = self._write_inflow_file(tmp_path, content)
        config = InflowReader().read(filepath)

        assert config.conversion_factor == pytest.approx(0.0283168)

    def test_read_zero_inflows(self, tmp_path: Path) -> None:
        """Read inflow file with zero inflows."""
        content = (
            "    1.0                            # Conversion factor\n"
            "    1DAY                           # Time unit\n"
            "    0                              / NInflow\n"
        )
        filepath = self._write_inflow_file(tmp_path, content)
        config = InflowReader().read(filepath)

        assert config.n_inflows == 0
        assert len(config.inflow_specs) == 0

    def test_inflow_nodes_property(self) -> None:
        """Test inflow_nodes property filters out zero nodes."""
        config = InflowConfig(n_inflows=3)
        config.inflow_specs = [
            InflowSpec(inflow_id=1, stream_node=5),
            InflowSpec(inflow_id=2, stream_node=0),  # no inflow
            InflowSpec(inflow_id=3, stream_node=10),
        ]
        assert config.inflow_nodes == [5, 10]

    def test_convenience_function(self, tmp_path: Path) -> None:
        """Test read_stream_inflow convenience function."""
        content = (
            "    1.0                            # Conversion factor\n"
            "    1DAY                           # Time unit\n"
            "    1                              / NInflow\n"
            "    1    5\n"
        )
        filepath = self._write_inflow_file(tmp_path, content)
        config = read_stream_inflow(filepath)
        assert config.n_inflows == 1

    def test_comments_skipped(self, tmp_path: Path) -> None:
        """Test that comments between data lines are skipped."""
        content = (
            "C Header\n"
            "c lowercase\n"
            "* asterisk\n"
            "    1.0                            # Conversion factor\n"
            "C  comment in the middle\n"
            "    1DAY                           # Time unit\n"
            "    2                              / NInflow\n"
            "C  comment before data\n"
            "    1    5\n"
            "C  another comment\n"
            "    2    10\n"
        )
        filepath = self._write_inflow_file(tmp_path, content)
        config = InflowReader().read(filepath)
        assert config.n_inflows == 2
        assert len(config.inflow_specs) == 2


# =============================================================================
# BypassSpecReader Tests
# =============================================================================


class TestBypassSpecReader:
    """Tests for BypassSpecReader."""

    def _write_bypass_file(self, path: Path, content: str) -> Path:
        filepath = path / "bypass.dat"
        filepath.write_text(content)
        return filepath

    def test_read_basic(self, tmp_path: Path) -> None:
        """Read bypass spec file with two bypasses, no inline rating."""
        content = (
            "C Bypass Specification File\n"
            "    2                              / NBypass\n"
            "    1.0                            # FlowFactor\n"
            "    1DAY                           # FlowTimeUnit\n"
            "    1.0                            # BypassFactor\n"
            "    1DAY                           # BypassTimeUnit\n"
            "    1    5    1    10    0    0.1    0.05    BYP1\n"
            "    2    8    2    3     0    0.2    0.10    BYP2\n"
            "C Seepage zones\n"
            "    1    0    0    0.0\n"
            "    2    0    0    0.0\n"
        )
        filepath = self._write_bypass_file(tmp_path, content)
        config = BypassSpecReader().read(filepath)

        assert config.n_bypasses == 2
        assert config.flow_factor == pytest.approx(1.0)
        assert config.bypass_factor == pytest.approx(1.0)
        assert len(config.bypasses) == 2

        bp1 = config.bypasses[0]
        assert bp1.id == 1
        assert bp1.export_stream_node == 5
        assert bp1.dest_type == BYPASS_DEST_STREAM
        assert bp1.dest_id == 10
        assert bp1.rating_table_col == 0
        assert bp1.frac_recoverable == pytest.approx(0.1)
        assert bp1.frac_non_recoverable == pytest.approx(0.05)
        assert bp1.name == "BYP1"

        bp2 = config.bypasses[1]
        assert bp2.dest_type == BYPASS_DEST_LAKE
        assert bp2.dest_id == 3

    def test_read_with_inline_rating(self, tmp_path: Path) -> None:
        """Read bypass with inline rating table (negative rating_table_col)."""
        content = (
            "    1                              / NBypass\n"
            "    1.0                            # FlowFactor\n"
            "    1DAY                           # FlowTimeUnit\n"
            "    1.0                            # BypassFactor\n"
            "    1DAY                           # BypassTimeUnit\n"
            "    1    5    1    10    -3    0.1    0.05    RATED\n"
            "    0.0     0.0\n"
            "    100.0   0.5\n"
            "    500.0   1.0\n"
            "C Seepage zones\n"
            "    1    0    0    0.0\n"
        )
        filepath = self._write_bypass_file(tmp_path, content)
        config = BypassSpecReader().read(filepath)

        assert config.n_bypasses == 1
        bp = config.bypasses[0]
        assert bp.rating_table_col == -3
        assert bp.inline_rating is not None
        assert len(bp.inline_rating.flows) == 3
        assert bp.inline_rating.flows[0] == pytest.approx(0.0)
        assert bp.inline_rating.flows[2] == pytest.approx(500.0)
        assert bp.inline_rating.fractions[1] == pytest.approx(0.5)

    def test_read_zero_bypasses(self, tmp_path: Path) -> None:
        """Read file with zero bypasses."""
        content = (
            "C Empty bypass file\n"
            "    0                              / NBypass\n"
        )
        filepath = self._write_bypass_file(tmp_path, content)
        config = BypassSpecReader().read(filepath)

        assert config.n_bypasses == 0
        assert len(config.bypasses) == 0

    def test_read_with_seepage_zones(self, tmp_path: Path) -> None:
        """Read bypass with seepage zone having elements."""
        content = (
            "    1                              / NBypass\n"
            "    1.0                            # FlowFactor\n"
            "    1DAY                           # FlowTimeUnit\n"
            "    1.0                            # BypassFactor\n"
            "    1DAY                           # BypassTimeUnit\n"
            "    1    5    1    10    0    0.1    0.05\n"
            "C Seepage zones\n"
            "    1    2    100    0.6\n"
            "    101    0.4\n"
        )
        filepath = self._write_bypass_file(tmp_path, content)
        config = BypassSpecReader().read(filepath)

        assert len(config.seepage_zones) == 1
        sz = config.seepage_zones[0]
        assert sz.n_elements == 2
        assert sz.element_ids == [100, 101]
        assert sz.element_fractions[0] == pytest.approx(0.6)
        assert sz.element_fractions[1] == pytest.approx(0.4)

    def test_convenience_function(self, tmp_path: Path) -> None:
        """Test read_bypass_spec convenience function."""
        content = (
            "    0                              / NBypass\n"
        )
        filepath = self._write_bypass_file(tmp_path, content)
        config = read_bypass_spec(filepath)
        assert config.n_bypasses == 0


# =============================================================================
# DiversionSpecReader Tests
# =============================================================================


class TestDiversionSpecReader:
    """Tests for DiversionSpecReader."""

    def _write_div_file(self, path: Path, content: str) -> Path:
        filepath = path / "divspec.dat"
        filepath.write_text(content)
        return filepath

    def test_read_14_column_format(self, tmp_path: Path) -> None:
        """Read 14-column (legacy, no spills) diversion spec file."""
        content = (
            "C Diversion Specification File\n"
            "    2                              / NDiver\n"
            "    1    5    1    1.0    0    0.0    0    0.0    1    100    1    1.0    0    0\n"
            "    2    10   2    1.0    0    0.0    0    0.0    3    0      2    1.0    0    0\n"
            "    0                              / NGroup\n"
            "C Recharge zones\n"
            "    1    0    0    0.0\n"
            "    2    0    0    0.0\n"
        )
        filepath = self._write_div_file(tmp_path, content)
        config = DiversionSpecReader().read(filepath)

        assert config.n_diversions == 2
        assert config.has_spills is False
        assert len(config.diversions) == 2

        d1 = config.diversions[0]
        assert d1.id == 1
        assert d1.stream_node == 5
        assert d1.max_diver_col == 1
        assert d1.frac_max_diver == pytest.approx(1.0)
        assert d1.dest_type == DEST_ELEMENT
        assert d1.dest_id == 100

        d2 = config.diversions[1]
        assert d2.dest_type == DEST_OUTSIDE

    def test_read_16_column_format(self, tmp_path: Path) -> None:
        """Read 16-column (with spills) diversion spec file."""
        content = (
            "C Diversion Specification File with spills\n"
            "    1                              / NDiver\n"
            "    1    5    1    1.0    0    0.0    0    0.0    3    0.5    1    100    1    1.0    0    0\n"
            "    0                              / NGroup\n"
            "C Recharge zones\n"
            "    1    0    0    0.0\n"
            "C Spill zones\n"
            "    1    0    0    0.0\n"
        )
        filepath = self._write_div_file(tmp_path, content)
        config = DiversionSpecReader().read(filepath)

        assert config.n_diversions == 1
        assert config.has_spills is True
        d = config.diversions[0]
        assert d.spill_col == 3
        assert d.frac_spill == pytest.approx(0.5)
        assert d.dest_type == DEST_ELEMENT
        assert d.dest_id == 100

    def test_read_with_element_groups(self, tmp_path: Path) -> None:
        """Read file with element groups."""
        content = (
            "    1                              / NDiver\n"
            "    1    5    1    1.0    0    0.0    0    0.0    4    1    1    1.0    0    0\n"
            "    1                              / NGroup\n"
            "    1    3    10\n"
            "    11\n"
            "    12\n"
            "C Recharge zones\n"
            "    1    0    0    0.0\n"
        )
        filepath = self._write_div_file(tmp_path, content)
        config = DiversionSpecReader().read(filepath)

        assert config.n_element_groups == 1
        assert len(config.element_groups) == 1
        g = config.element_groups[0]
        assert g.id == 1
        assert g.elements == [10, 11, 12]

    def test_read_with_recharge_zones(self, tmp_path: Path) -> None:
        """Read file with recharge zone destinations."""
        content = (
            "    1                              / NDiver\n"
            "    1    5    1    1.0    0    0.0    0    0.0    1    100    1    1.0    0    0\n"
            "    0                              / NGroup\n"
            "C Recharge zones\n"
            "    1    2    50    0.7\n"
            "    51    0.3\n"
        )
        filepath = self._write_div_file(tmp_path, content)
        config = DiversionSpecReader().read(filepath)

        assert len(config.recharge_zones) == 1
        rz = config.recharge_zones[0]
        assert rz.n_zones == 2
        assert rz.zone_ids == [50, 51]
        assert rz.zone_fractions[0] == pytest.approx(0.7)
        assert rz.zone_fractions[1] == pytest.approx(0.3)

    def test_read_zero_diversions(self, tmp_path: Path) -> None:
        """Read file with zero diversions."""
        content = (
            "C Empty diversion file\n"
            "    0                              / NDiver\n"
        )
        filepath = self._write_div_file(tmp_path, content)
        config = DiversionSpecReader().read(filepath)

        assert config.n_diversions == 0
        assert len(config.diversions) == 0

    def test_diversion_with_name(self, tmp_path: Path) -> None:
        """Read diversion with trailing name field."""
        content = (
            "    1                              / NDiver\n"
            "    1    5    1    1.0    0    0.0    0    0.0    1    100    1    1.0    0    0    Sacramento River\n"
            "    0                              / NGroup\n"
            "C Recharge zones\n"
            "    1    0    0    0.0\n"
        )
        filepath = self._write_div_file(tmp_path, content)
        config = DiversionSpecReader().read(filepath)

        d = config.diversions[0]
        assert "Sacramento" in d.name

    def test_convenience_function(self, tmp_path: Path) -> None:
        """Test read_diversion_spec convenience function."""
        content = (
            "    0                              / NDiver\n"
        )
        filepath = self._write_div_file(tmp_path, content)
        config = read_diversion_spec(filepath)
        assert config.n_diversions == 0

    def test_config_defaults(self) -> None:
        """Test DiversionSpecConfig default values."""
        config = DiversionSpecConfig()
        assert config.n_diversions == 0
        assert config.has_spills is False
        assert len(config.diversions) == 0
        assert len(config.element_groups) == 0
        assert len(config.recharge_zones) == 0
        assert len(config.spill_zones) == 0
