"""Extended coverage tests for stream sub-file readers.

Covers edge cases, error paths, and additional branches for:
- InflowReader / read_stream_inflow / InflowConfig / InflowSpec
- DiversionSpecReader / read_diversion_spec / DiversionSpecConfig / data classes
- BypassSpecReader / read_bypass_spec / BypassSpecConfig / data classes
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pyiwfm.io.iwfm_reader import strip_inline_comment as inflow_parse_value
from pyiwfm.io.stream_bypass import (
    BYPASS_DEST_LAKE,
    BYPASS_DEST_STREAM,
    BypassRatingTable,
    BypassSeepageZone,
    BypassSpec,
    BypassSpecConfig,
    BypassSpecReader,
    read_bypass_spec,
)
from pyiwfm.io.stream_diversion import (
    DEST_ELEMENT,
    DEST_ELEMENT_SET,
    DEST_OUTSIDE,
    DEST_SUBREGION,
    DiversionSpec,
    DiversionSpecConfig,
    DiversionSpecReader,
    ElementGroup,
    RechargeZoneDest,
    read_diversion_spec,
)
from pyiwfm.io.stream_inflow import (
    InflowConfig,
    InflowReader,
    InflowSpec,
    read_stream_inflow,
)
from pyiwfm.io.stream_inflow import (
    _is_comment_line as inflow_is_comment,
)

# =============================================================================
# InflowReader extended tests
# =============================================================================


class TestInflowHelpers:
    """Tests for module-level helpers in stream_inflow."""

    def test_is_comment_empty(self) -> None:
        assert inflow_is_comment("") is True

    def test_is_comment_whitespace(self) -> None:
        assert inflow_is_comment("   \t  ") is True

    def test_is_comment_c_upper(self) -> None:
        assert inflow_is_comment("C comment") is True

    def test_is_comment_asterisk(self) -> None:
        assert inflow_is_comment("* comment") is True

    def test_is_comment_data_line(self) -> None:
        assert inflow_is_comment("    42") is False

    def test_parse_value_hash_not_delimiter(self) -> None:
        """'#' is NOT a comment delimiter in IWFM — only '/' is."""
        val, _ = inflow_parse_value("    1.0    # factor")
        assert val == "1.0    # factor"

    def test_parse_value_plain(self) -> None:
        val, desc = inflow_parse_value("   hello")
        assert val == "hello"
        assert desc == ""


class TestInflowDataClasses:
    """Tests for stream_inflow data classes."""

    def test_inflow_spec_defaults(self) -> None:
        spec = InflowSpec()
        assert spec.inflow_id == 0
        assert spec.stream_node == 0

    def test_inflow_spec_with_values(self) -> None:
        spec = InflowSpec(inflow_id=5, stream_node=42)
        assert spec.inflow_id == 5
        assert spec.stream_node == 42

    def test_inflow_config_defaults(self) -> None:
        config = InflowConfig()
        assert config.conversion_factor == 1.0
        assert config.time_unit == ""
        assert config.n_inflows == 0
        assert config.inflow_specs == []

    def test_inflow_nodes_empty(self) -> None:
        config = InflowConfig()
        assert config.inflow_nodes == []

    def test_inflow_nodes_filters_zeros(self) -> None:
        config = InflowConfig(
            inflow_specs=[
                InflowSpec(inflow_id=1, stream_node=0),
                InflowSpec(inflow_id=2, stream_node=5),
                InflowSpec(inflow_id=3, stream_node=0),
                InflowSpec(inflow_id=4, stream_node=10),
            ]
        )
        assert config.inflow_nodes == [5, 10]


class TestInflowReaderExtended:
    """Extended tests for InflowReader."""

    def _write(self, path: Path, content: str) -> Path:
        filepath = path / "inflow.dat"
        filepath.write_text(content)
        return filepath

    def test_empty_file(self, tmp_path: Path) -> None:
        """Empty file returns defaults."""
        filepath = self._write(tmp_path, "")
        config = InflowReader().read(filepath)
        assert config.n_inflows == 0
        assert config.inflow_specs == []

    def test_only_comments(self, tmp_path: Path) -> None:
        """File with only comments."""
        content = "C comment 1\nC comment 2\n* comment 3\n"
        filepath = self._write(tmp_path, content)
        config = InflowReader().read(filepath)
        assert config.n_inflows == 0

    def test_negative_inflows_returns_empty(self, tmp_path: Path) -> None:
        """Negative inflow count treated as zero."""
        content = (
            "    1.0                            / factor\n"
            "    1DAY                           / time\n"
            "    -1                             / NInflow\n"
        )
        filepath = self._write(tmp_path, content)
        config = InflowReader().read(filepath)
        assert config.n_inflows == -1
        assert len(config.inflow_specs) == 0

    def test_many_inflows(self, tmp_path: Path) -> None:
        """Read file with many inflow points."""
        lines = [
            "    1.0                            / factor\n",
            "    1DAY                           / time\n",
            "    10                             / NInflow\n",
        ]
        for i in range(1, 11):
            lines.append(f"    {i}    {i * 10}\n")
        content = "".join(lines)
        filepath = self._write(tmp_path, content)
        config = InflowReader().read(filepath)

        assert config.n_inflows == 10
        assert len(config.inflow_specs) == 10
        assert config.inflow_specs[9].stream_node == 100

    def test_read_accepts_string_path(self, tmp_path: Path) -> None:
        """Reader accepts string paths."""
        content = "    1.0\n    1DAY\n    0\n"
        filepath = self._write(tmp_path, content)
        config = read_stream_inflow(str(filepath))
        assert config.n_inflows == 0


# =============================================================================
# DiversionSpecReader extended tests
# =============================================================================


class TestDiversionDataClasses:
    """Tests for stream_diversion data classes."""

    def test_diversion_spec_defaults(self) -> None:
        d = DiversionSpec()
        assert d.id == 0
        assert d.stream_node == 0
        assert d.spill_col == 0
        assert d.frac_spill == 0.0
        assert d.dest_type == DEST_OUTSIDE
        assert d.name == ""

    def test_element_group_defaults(self) -> None:
        eg = ElementGroup()
        assert eg.id == 0
        assert eg.elements == []

    def test_recharge_zone_dest_defaults(self) -> None:
        rz = RechargeZoneDest()
        assert rz.diversion_id == 0
        assert rz.n_zones == 0
        assert rz.zone_ids == []
        assert rz.zone_fractions == []

    def test_diversion_spec_config_defaults(self) -> None:
        config = DiversionSpecConfig()
        assert config.n_diversions == 0
        assert config.has_spills is False
        assert config.diversions == []
        assert config.element_groups == []
        assert config.recharge_zones == []
        assert config.spill_zones == []

    def test_destination_constants(self) -> None:
        assert DEST_ELEMENT == 1
        assert DEST_SUBREGION == 2
        assert DEST_OUTSIDE == 3
        assert DEST_ELEMENT_SET == 4


class TestDiversionReaderExtended:
    """Extended tests for DiversionSpecReader."""

    def _write(self, path: Path, content: str) -> Path:
        filepath = path / "divspec.dat"
        filepath.write_text(content)
        return filepath

    def test_empty_file(self, tmp_path: Path) -> None:
        """Empty file returns default config."""
        filepath = self._write(tmp_path, "")
        config = DiversionSpecReader().read(filepath)
        assert config.n_diversions == 0

    def test_count_numeric_fields(self) -> None:
        """Test the _count_numeric_fields method."""
        reader = DiversionSpecReader()
        assert reader._count_numeric_fields(["1", "2", "3.5", "text"]) == 3
        assert reader._count_numeric_fields(["text"]) == 0
        assert reader._count_numeric_fields([]) == 0
        assert reader._count_numeric_fields(["1.0", "2", "3"]) == 3

    def test_strip_inline_comment(self) -> None:
        """Test the _strip_inline_comment static method."""
        assert DiversionSpecReader._strip_inline_comment("10 / comment") == "10"
        assert DiversionSpecReader._strip_inline_comment("42") == "42"
        assert DiversionSpecReader._strip_inline_comment("") == ""

    def test_16col_with_spill_zones(self, tmp_path: Path) -> None:
        """Read 16-column format with spill zone data."""
        content = (
            "    1                              / NDiver\n"
            "    1    5    1    1.0    0    0.0    0    0.0    3    0.5    1    100    1    1.0    0    0\n"
            "    0                              / NGroup\n"
            "C Recharge zones\n"
            "    1    0    0    0.0\n"
            "C Spill zones\n"
            "    1    1    200    0.8\n"
        )
        filepath = self._write(tmp_path, content)
        config = DiversionSpecReader().read(filepath)

        assert config.has_spills is True
        assert len(config.spill_zones) == 1
        sz = config.spill_zones[0]
        assert sz.n_zones == 1
        assert sz.zone_ids == [200]
        assert sz.zone_fractions[0] == pytest.approx(0.8)

    def test_multiple_element_groups(self, tmp_path: Path) -> None:
        """Read file with multiple element groups."""
        content = (
            "    1                              / NDiver\n"
            "    1    5    1    1.0    0    0.0    0    0.0    4    1    1    1.0    0    0\n"
            "    2                              / NGroup\n"
            "    1    2    10\n"
            "    11\n"
            "    2    1    20\n"
            "C Recharge zones\n"
            "    1    0    0    0.0\n"
        )
        filepath = self._write(tmp_path, content)
        config = DiversionSpecReader().read(filepath)

        assert config.n_element_groups == 2
        assert config.element_groups[0].elements == [10, 11]
        assert config.element_groups[1].elements == [20]

    def test_multiple_diversions_14col(self, tmp_path: Path) -> None:
        """Read multiple diversions in legacy 14-column format."""
        content = (
            "    3                              / NDiver\n"
            "    1    5    1    1.0    0    0.0    0    0.0    1    100    1    1.0    0    0\n"
            "    2    10   2    1.0    0    0.0    0    0.0    2    5      2    1.0    0    0\n"
            "    3    15   3    0.5    1    0.1    1    0.05   3    0      3    0.8    0    0\n"
            "    0                              / NGroup\n"
            "C Recharge zones\n"
            "    1    0    0    0.0\n"
            "    2    0    0    0.0\n"
            "    3    0    0    0.0\n"
        )
        filepath = self._write(tmp_path, content)
        config = DiversionSpecReader().read(filepath)

        assert config.n_diversions == 3
        assert config.has_spills is False
        assert config.diversions[2].frac_max_diver == pytest.approx(0.5)
        assert config.diversions[2].frac_recv_loss == pytest.approx(0.1)

    def test_recharge_zone_with_multiple_elements(self, tmp_path: Path) -> None:
        """Read recharge zone with 3 zone elements."""
        content = (
            "    1                              / NDiver\n"
            "    1    5    1    1.0    0    0.0    0    0.0    1    100    1    1.0    0    0\n"
            "    0                              / NGroup\n"
            "C Recharge zones\n"
            "    1    3    50    0.5\n"
            "    51    0.3\n"
            "    52    0.2\n"
        )
        filepath = self._write(tmp_path, content)
        config = DiversionSpecReader().read(filepath)

        assert len(config.recharge_zones) == 1
        rz = config.recharge_zones[0]
        assert rz.n_zones == 3
        assert rz.zone_ids == [50, 51, 52]
        assert rz.zone_fractions == [pytest.approx(0.5), pytest.approx(0.3), pytest.approx(0.2)]

    def test_read_accepts_string_path(self, tmp_path: Path) -> None:
        """Reader accepts string paths."""
        content = "    0                              / NDiver\n"
        filepath = self._write(tmp_path, content)
        config = read_diversion_spec(str(filepath))
        assert config.n_diversions == 0


# =============================================================================
# BypassSpecReader extended tests
# =============================================================================


class TestBypassDataClasses:
    """Tests for stream_bypass data classes."""

    def test_bypass_spec_defaults(self) -> None:
        bp = BypassSpec()
        assert bp.id == 0
        assert bp.export_stream_node == 0
        assert bp.dest_type == BYPASS_DEST_STREAM
        assert bp.rating_table_col == 0
        assert bp.frac_recoverable == 0.0
        assert bp.frac_non_recoverable == 0.0
        assert bp.name == ""
        assert bp.inline_rating is None

    def test_bypass_rating_table_defaults(self) -> None:
        rt = BypassRatingTable()
        assert len(rt.flows) == 0
        assert len(rt.fractions) == 0

    def test_bypass_seepage_zone_defaults(self) -> None:
        sz = BypassSeepageZone()
        assert sz.bypass_id == 0
        assert sz.n_elements == 0
        assert sz.element_ids == []
        assert sz.element_fractions == []

    def test_bypass_spec_config_defaults(self) -> None:
        config = BypassSpecConfig()
        assert config.n_bypasses == 0
        assert config.flow_factor == 1.0
        assert config.flow_time_unit == ""
        assert config.bypass_factor == 1.0
        assert config.bypass_time_unit == ""
        assert config.bypasses == []
        assert config.seepage_zones == []

    def test_bypass_dest_constants(self) -> None:
        assert BYPASS_DEST_STREAM == 1
        assert BYPASS_DEST_LAKE == 2


class TestBypassReaderExtended:
    """Extended tests for BypassSpecReader."""

    def _write(self, path: Path, content: str) -> Path:
        filepath = path / "bypass.dat"
        filepath.write_text(content)
        return filepath

    def test_empty_file(self, tmp_path: Path) -> None:
        """Empty file returns default config."""
        filepath = self._write(tmp_path, "")
        config = BypassSpecReader().read(filepath)
        assert config.n_bypasses == 0

    def test_flow_factor_applied_to_rating_table(self, tmp_path: Path) -> None:
        """Flow factor should multiply inline rating table flow values."""
        content = (
            "    1                              / NBypass\n"
            "    2.0                            / FlowFactor\n"
            "    1DAY                           / FlowTimeUnit\n"
            "    1.0                            / BypassFactor\n"
            "    1DAY                           / BypassTimeUnit\n"
            "    1    5    1    10    -2    0.1    0.05\n"
            "    100.0   0.5\n"
            "    200.0   1.0\n"
            "C Seepage zones\n"
            "    1    0    0    0.0\n"
        )
        filepath = self._write(tmp_path, content)
        config = BypassSpecReader().read(filepath)

        bp = config.bypasses[0]
        assert bp.inline_rating is not None
        assert bp.inline_rating.flows[0] == pytest.approx(200.0)  # 100 * 2.0
        assert bp.inline_rating.flows[1] == pytest.approx(400.0)  # 200 * 2.0

    def test_lake_destination(self, tmp_path: Path) -> None:
        """Read bypass with lake destination type."""
        content = (
            "    1                              / NBypass\n"
            "    1.0                            / FlowFactor\n"
            "    1DAY\n"
            "    1.0                            / BypassFactor\n"
            "    1DAY\n"
            "    1    5    2    3    0    0.0    0.0    LAKE_BYP\n"
            "    1    0    0    0.0\n"
        )
        filepath = self._write(tmp_path, content)
        config = BypassSpecReader().read(filepath)

        bp = config.bypasses[0]
        assert bp.dest_type == BYPASS_DEST_LAKE
        assert bp.dest_id == 3
        assert bp.name == "LAKE_BYP"

    def test_multiple_bypasses_with_seepage(self, tmp_path: Path) -> None:
        """Read multiple bypasses each with seepage zone data."""
        content = (
            "    2                              / NBypass\n"
            "    1.0\n"
            "    1DAY\n"
            "    1.0\n"
            "    1DAY\n"
            "    1    5    1    10    0    0.1    0.05\n"
            "    2    8    1    12    0    0.2    0.10\n"
            "C Seepage zones\n"
            "    1    1    100    1.0\n"
            "    2    2    200    0.6\n"
            "    201    0.4\n"
        )
        filepath = self._write(tmp_path, content)
        config = BypassSpecReader().read(filepath)

        assert len(config.seepage_zones) == 2
        assert config.seepage_zones[0].element_ids == [100]
        assert config.seepage_zones[1].element_ids == [200, 201]
        assert config.seepage_zones[1].element_fractions[1] == pytest.approx(0.4)

    def test_no_seepage_zones_graceful(self, tmp_path: Path) -> None:
        """File ending without seepage zone data handled gracefully."""
        content = (
            "    1                              / NBypass\n"
            "    1.0\n"
            "    1DAY\n"
            "    1.0\n"
            "    1DAY\n"
            "    1    5    1    10    0    0.1    0.05\n"
        )
        filepath = self._write(tmp_path, content)
        config = BypassSpecReader().read(filepath)

        assert config.n_bypasses == 1
        assert len(config.seepage_zones) == 0

    def test_bypass_name_truncated_to_20(self, tmp_path: Path) -> None:
        """Bypass name should be truncated to 20 characters."""
        content = (
            "    1                              / NBypass\n"
            "    1.0\n"
            "    1DAY\n"
            "    1.0\n"
            "    1DAY\n"
            "    1    5    1    10    0    0.1    0.05    ThisIsAVeryLongBypassName\n"
            "    1    0    0    0.0\n"
        )
        filepath = self._write(tmp_path, content)
        config = BypassSpecReader().read(filepath)

        assert len(config.bypasses[0].name) <= 20

    def test_read_accepts_string_path(self, tmp_path: Path) -> None:
        """Reader accepts string paths."""
        content = "    0                              / NBypass\n"
        filepath = self._write(tmp_path, content)
        config = read_bypass_spec(str(filepath))
        assert config.n_bypasses == 0

    def test_bypass_positive_rating_col(self, tmp_path: Path) -> None:
        """Positive rating_table_col means pre-defined rating (no inline)."""
        content = (
            "    1                              / NBypass\n"
            "    1.0\n"
            "    1DAY\n"
            "    1.0\n"
            "    1DAY\n"
            "    1    5    1    10    2    0.1    0.05\n"
            "    1    0    0    0.0\n"
        )
        filepath = self._write(tmp_path, content)
        config = BypassSpecReader().read(filepath)

        bp = config.bypasses[0]
        assert bp.rating_table_col == 2
        assert bp.inline_rating is None

    def test_seepage_zone_single_element_default_fraction(self, tmp_path: Path) -> None:
        """Seepage zone element without explicit fraction defaults to 1.0."""
        content = (
            "    1                              / NBypass\n"
            "    1.0\n"
            "    1DAY\n"
            "    1.0\n"
            "    1DAY\n"
            "    1    5    1    10    0    0.1    0.05\n"
            "C Seepage\n"
            "    1    2    100    0.7\n"
            "    101\n"
        )
        filepath = self._write(tmp_path, content)
        config = BypassSpecReader().read(filepath)

        sz = config.seepage_zones[0]
        assert sz.element_fractions[1] == pytest.approx(1.0)
