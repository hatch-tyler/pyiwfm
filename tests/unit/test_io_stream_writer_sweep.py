"""Sweep tests for pyiwfm.io.stream_writer targeting specific uncovered lines.

Covers:
- Lines 362-364: budget_node_ids not-a-list and AttributeError fallback
- Lines 428, 480: v4.2 bed params format (WETPR IRGW CSTRM DSTRM)
- Lines 455-456, 461-462, 467-468: AttributeError on node bed-param attributes
- Lines 514-515, 520-521: AttributeError on cross-section factors
- Lines 540-541: AttributeError on node.cross_section
- Lines 562-563, 568-569: AttributeError on ic_type/ic_factor
- Lines 589-590, 592: AttributeError / non-numeric initial_condition
- Lines 619-635: Evaporation absolute path logic (local candidate + relpath)
- Line 787: Empty element group in diver_specs
- Line 828: Spill zone continuation rows in diver_specs
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from pyiwfm.io.stream_writer import (
    StreamComponentWriter,
    StreamWriterConfig,
)

# =============================================================================
# Helpers
# =============================================================================


def _mock_engine() -> MagicMock:
    engine = MagicMock()
    engine.render_template.return_value = "C  MOCK HEADER\n"
    return engine


def _base_streams(**overrides: object) -> MagicMock:
    """Create a MagicMock streams component with sane defaults."""
    streams = MagicMock()
    defaults = {
        "nodes": {},
        "reaches": {},
        "budget_node_ids": [],
        "budget_node_count": 0,
        "diversions": {},
        "bypasses": {},
        "inflows": [],
        "evap_node_specs": [],
        "evap_area_file": "",
    }
    defaults.update(overrides)
    for k, v in defaults.items():
        setattr(streams, k, v)
    return streams


def _model_with_streams(streams: MagicMock) -> MagicMock:
    model = MagicMock()
    model.streams = streams
    model.source_files = {}
    return model


# =============================================================================
# Lines 362-364: budget_node_ids not a list / AttributeError
# =============================================================================


class TestBudgetNodeIdsFallbacks:
    """Cover lines 361-364 where budget_node_ids is not a list or raises."""

    def test_budget_ids_not_a_list(self, tmp_path: Path) -> None:
        """When budget_node_ids is not a list, fall back to empty."""
        streams = _base_streams(
            budget_node_count=2,
            budget_node_ids="not-a-list",  # triggers isinstance check
        )
        model = _model_with_streams(streams)
        engine = _mock_engine()

        config = StreamWriterConfig(output_dir=tmp_path)
        writer = StreamComponentWriter(model, config, template_engine=engine)
        path = writer.write_main()
        content = path.read_text()
        # Should fall back to empty budget_ids; NBUDR written as 0
        assert "NBUDR" in content

    def test_budget_ids_attribute_error(self, tmp_path: Path) -> None:
        """When budget_node_ids raises AttributeError, fall back to empty."""
        streams = _base_streams(budget_node_count=5)
        # Remove budget_node_ids so it raises AttributeError
        del streams.budget_node_ids
        type(streams).budget_node_ids = property(
            lambda self: (_ for _ in ()).throw(AttributeError("no attr"))
        )
        model = _model_with_streams(streams)
        engine = _mock_engine()

        config = StreamWriterConfig(output_dir=tmp_path)
        writer = StreamComponentWriter(model, config, template_engine=engine)
        path = writer.write_main()
        content = path.read_text()
        assert "NBUDR" in content


# =============================================================================
# Lines 428, 480: v4.2 bed params (the else branch)
# =============================================================================


class TestV42BedParams:
    """Cover v4.2 bed params format: IR WETPR IRGW CSTRM DSTRM."""

    def test_v42_column_header(self, tmp_path: Path) -> None:
        """v4.2 writes 5-column header with WETPR and IRGW."""
        node1 = SimpleNamespace(
            id=1,
            wetted_perimeter=200.0,
            conductivity=15.0,
            bed_thickness=2.0,
            gw_node=5,
            cross_section=None,
            initial_condition=0.0,
        )
        streams = _base_streams(nodes={1: node1})
        model = _model_with_streams(streams)
        engine = _mock_engine()

        config = StreamWriterConfig(output_dir=tmp_path, version="4.2")
        writer = StreamComponentWriter(model, config, template_engine=engine)
        path = writer.write_main()
        content = path.read_text()
        # v4.2 header
        assert "WETPR" in content
        assert "IRGW" in content
        assert "CSTRM" in content
        assert "DSTRM" in content
        # Data line should include gw_node=5
        assert "5" in content

    def test_v42_bed_params_data_format(self, tmp_path: Path) -> None:
        """v4.2 data lines contain IR WETPR IRGW CSTRM DSTRM."""
        node = SimpleNamespace(
            id=10,
            wetted_perimeter=300.0,
            conductivity=20.0,
            bed_thickness=3.0,
            gw_node=42,
            cross_section=None,
            initial_condition=0.0,
        )
        streams = _base_streams(nodes={10: node})
        model = _model_with_streams(streams)
        engine = _mock_engine()

        config = StreamWriterConfig(output_dir=tmp_path, version="4.2")
        writer = StreamComponentWriter(model, config, template_engine=engine)
        path = writer.write_main()
        content = path.read_text()
        # Should contain gw_node 42 in the bed params data
        assert "42" in content
        # Should contain wetted perimeter 300
        assert "300" in content


# =============================================================================
# Lines 455-456, 461-462, 467-468: AttributeError on node attrs
# =============================================================================


class TestNodeAttributeErrors:
    """Cover AttributeError fallbacks for conductivity, bed_thickness, wetted_perimeter."""

    def test_node_missing_conductivity(self, tmp_path: Path) -> None:
        """When node has no conductivity attr, use config default."""
        node = SimpleNamespace(id=1, gw_node=1)
        # No conductivity, bed_thickness, or wetted_perimeter
        streams = _base_streams(nodes={1: node})
        model = _model_with_streams(streams)
        engine = _mock_engine()

        config = StreamWriterConfig(output_dir=tmp_path, version="4.0")
        writer = StreamComponentWriter(model, config, template_engine=engine)
        path = writer.write_main()
        content = path.read_text()
        # Should use config defaults (conductivity=10.0, bed_thickness=1.0, wetted_perimeter=150.0)
        assert "10.000" in content
        assert path.exists()

    def test_node_missing_all_bed_attrs_v42(self, tmp_path: Path) -> None:
        """v4.2 format with node missing bed param attributes."""
        node = SimpleNamespace(id=5, gw_node=None)
        # No conductivity, bed_thickness, wetted_perimeter
        streams = _base_streams(nodes={5: node})
        model = _model_with_streams(streams)
        engine = _mock_engine()

        config = StreamWriterConfig(output_dir=tmp_path, version="4.2")
        writer = StreamComponentWriter(model, config, template_engine=engine)
        path = writer.write_main()
        content = path.read_text()
        # Should use defaults and gw_node=0 (since gw_node is None -> guard fails)
        assert "150" in content  # default wetted_perimeter
        assert path.exists()


# =============================================================================
# Lines 514-515, 520-521: AttributeError on cross-section factors
# =============================================================================


class TestCrossSectionFactorErrors:
    """Cover AttributeError on roughness_factor and cross_section_length_factor."""

    def test_missing_roughness_factor(self, tmp_path: Path) -> None:
        """When streams has no roughness_factor, use config default."""
        node = SimpleNamespace(
            id=1,
            wetted_perimeter=100.0,
            conductivity=10.0,
            bed_thickness=1.0,
            gw_node=1,
            cross_section=None,
            initial_condition=0.0,
        )
        streams = _base_streams(nodes={1: node})
        # Remove roughness_factor and cross_section_length_factor
        del streams.roughness_factor
        del streams.cross_section_length_factor
        type(streams).roughness_factor = property(lambda s: (_ for _ in ()).throw(AttributeError))
        type(streams).cross_section_length_factor = property(
            lambda s: (_ for _ in ()).throw(AttributeError)
        )
        model = _model_with_streams(streams)
        engine = _mock_engine()

        config = StreamWriterConfig(output_dir=tmp_path, version="5.0")
        writer = StreamComponentWriter(model, config, template_engine=engine)
        path = writer.write_main()
        content = path.read_text()
        # Should still contain the cross-section section with defaults
        assert "Cross-Section Data" in content
        assert "FACTN" in content


# =============================================================================
# Lines 540-541: AttributeError on node.cross_section
# =============================================================================


class TestNodeCrossSectionError:
    """Cover AttributeError on node.cross_section."""

    def test_node_no_cross_section_attr(self, tmp_path: Path) -> None:
        """When node has no cross_section attribute, write defaults."""
        node = SimpleNamespace(
            id=1,
            wetted_perimeter=100.0,
            conductivity=10.0,
            bed_thickness=1.0,
            gw_node=1,
            initial_condition=0.0,
        )
        # Intentionally no cross_section attribute
        streams = _base_streams(nodes={1: node})
        model = _model_with_streams(streams)
        engine = _mock_engine()

        config = StreamWriterConfig(output_dir=tmp_path, version="5.0")
        writer = StreamComponentWriter(model, config, template_engine=engine)
        path = writer.write_main()
        content = path.read_text()
        # Default cross-section values should appear
        assert "0.0400" in content  # default n


# =============================================================================
# Lines 562-563, 568-569: AttributeError on ic_type/ic_factor
# =============================================================================


class TestInitialConditionFactorErrors:
    """Cover AttributeError on streams.ic_type and streams.ic_factor."""

    def test_missing_ic_type_and_factor(self, tmp_path: Path) -> None:
        """When streams has no ic_type/ic_factor, use config defaults."""
        node = SimpleNamespace(
            id=1,
            wetted_perimeter=100.0,
            conductivity=10.0,
            bed_thickness=1.0,
            gw_node=1,
            cross_section=None,
            initial_condition=0.0,
        )
        streams = _base_streams(nodes={1: node})
        del streams.ic_type
        del streams.ic_factor
        type(streams).ic_type = property(lambda s: (_ for _ in ()).throw(AttributeError))
        type(streams).ic_factor = property(lambda s: (_ for _ in ()).throw(AttributeError))
        model = _model_with_streams(streams)
        engine = _mock_engine()

        config = StreamWriterConfig(output_dir=tmp_path, version="5.0")
        writer = StreamComponentWriter(model, config, template_engine=engine)
        path = writer.write_main()
        content = path.read_text()
        assert "Initial Conditions" in content
        assert "ICTYPE" in content


# =============================================================================
# Lines 589-590, 592: AttributeError / non-numeric initial_condition
# =============================================================================


class TestNodeInitialConditionEdges:
    """Cover AttributeError on node.initial_condition and non-numeric values."""

    def test_node_no_initial_condition_attr(self, tmp_path: Path) -> None:
        """When node has no initial_condition attribute, use 0.0."""
        node = SimpleNamespace(
            id=1,
            wetted_perimeter=100.0,
            conductivity=10.0,
            bed_thickness=1.0,
            gw_node=1,
            cross_section=None,
        )
        # No initial_condition attribute
        streams = _base_streams(nodes={1: node})
        model = _model_with_streams(streams)
        engine = _mock_engine()

        config = StreamWriterConfig(output_dir=tmp_path, version="5.0")
        writer = StreamComponentWriter(model, config, template_engine=engine)
        path = writer.write_main()
        content = path.read_text()
        assert "0.0000" in content

    def test_node_non_numeric_initial_condition(self, tmp_path: Path) -> None:
        """When initial_condition is not numeric, reset to 0.0."""
        node = SimpleNamespace(
            id=1,
            wetted_perimeter=100.0,
            conductivity=10.0,
            bed_thickness=1.0,
            gw_node=1,
            cross_section=None,
            initial_condition="bad_value",  # non-numeric
        )
        streams = _base_streams(nodes={1: node})
        model = _model_with_streams(streams)
        engine = _mock_engine()

        config = StreamWriterConfig(output_dir=tmp_path, version="5.0")
        writer = StreamComponentWriter(model, config, template_engine=engine)
        path = writer.write_main()
        content = path.read_text()
        # Should reset to 0.0
        assert "0.0000" in content


# =============================================================================
# Lines 619-635: Evaporation absolute path resolution
# =============================================================================


class TestEvaporationAbsPath:
    """Cover evap_area_file absolute path resolution logic."""

    def test_evap_abs_path_local_candidate_exists(self, tmp_path: Path) -> None:
        """When evap file exists locally in Streams/ dir, use relative path."""
        # Create local candidate
        streams_dir = tmp_path / "Stream"
        streams_dir.mkdir(parents=True)
        (streams_dir / "SurfArea.dat").write_text("dummy")

        abs_evap_path = str(tmp_path / "elsewhere" / "SurfArea.dat")

        node = SimpleNamespace(
            id=1,
            wetted_perimeter=100.0,
            conductivity=10.0,
            bed_thickness=1.0,
            gw_node=1,
            cross_section=None,
            initial_condition=0.0,
        )
        streams = _base_streams(
            nodes={1: node},
            evap_area_file=abs_evap_path,
            evap_node_specs=[],
        )
        model = _model_with_streams(streams)
        engine = _mock_engine()

        config = StreamWriterConfig(output_dir=tmp_path)
        writer = StreamComponentWriter(model, config, template_engine=engine)
        path = writer.write_main()
        content = path.read_text()
        # Should convert to relative path
        assert "STARFL" in content
        # Should NOT contain the absolute path
        assert "elsewhere" not in content

    def test_evap_abs_path_no_local_candidate(self, tmp_path: Path) -> None:
        """When evap file does not exist locally, compute relative from output_dir."""
        abs_evap_path = str(tmp_path / "other_model" / "Streams" / "SurfArea.dat")

        node = SimpleNamespace(
            id=1,
            wetted_perimeter=100.0,
            conductivity=10.0,
            bed_thickness=1.0,
            gw_node=1,
            cross_section=None,
            initial_condition=0.0,
        )
        streams = _base_streams(
            nodes={1: node},
            evap_area_file=abs_evap_path,
            evap_node_specs=[],
        )
        model = _model_with_streams(streams)
        engine = _mock_engine()

        config = StreamWriterConfig(output_dir=tmp_path)
        writer = StreamComponentWriter(model, config, template_engine=engine)
        path = writer.write_main()
        content = path.read_text()
        assert "STARFL" in content


# =============================================================================
# Line 787: Empty element group in diver_specs
# =============================================================================


class TestDiverSpecsEmptyElementGroup:
    """Cover empty element group (n_elems == 0) at line 787."""

    def test_empty_element_group_writes_zeros(self, tmp_path: Path) -> None:
        """Element group with no elements writes default zeros."""
        model = MagicMock()
        streams = MagicMock()
        div = SimpleNamespace(
            id=1,
            source_node=10,
            max_div_column=1,
            max_div_fraction=1.0,
            recoverable_loss_column=0,
            recoverable_loss_fraction=0.0,
            non_recoverable_loss_column=0,
            non_recoverable_loss_fraction=0.0,
            delivery_dest_type=0,
            delivery_dest_id=5,
            delivery_column=0,
            delivery_fraction=1.0,
            irrigation_fraction_column=0,
            adjustment_column=0,
            name="DivEmpty",
        )
        streams.diversions = {1: div}
        streams.diversion_has_spills = False
        # Empty element group (no elements)
        eg = SimpleNamespace(id=1, elements=[])
        streams.diversion_element_groups = [eg]
        streams.diversion_recharge_zones = []
        model.streams = streams

        engine = _mock_engine()
        config = StreamWriterConfig(output_dir=tmp_path)
        writer = StreamComponentWriter(model, config, template_engine=engine)
        path = writer.write_diver_specs()
        content = path.read_text()
        assert "NGRP" in content
        # Should write the zero-element fallback: "gid 0 0"
        # Find line after NGRP that has the element group
        lines = content.split("\n")
        found_zero_group = False
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith("C"):
                parts = stripped.split()
                if len(parts) >= 3 and parts[1] == "0":
                    found_zero_group = True
                    break
        assert found_zero_group, "Should have a line with element group having 0 elements"


# =============================================================================
# Line 828: Spill zone continuation rows
# =============================================================================


class TestDiverSpecsSpillZoneContinuation:
    """Cover spill zone with multiple zones (continuation rows) at line 828."""

    def test_spill_zone_multiple_zones(self, tmp_path: Path) -> None:
        """Spill zone with >1 zone writes continuation rows."""
        model = MagicMock()
        streams = MagicMock()
        div = SimpleNamespace(
            id=1,
            source_node=10,
            max_div_column=1,
            max_div_fraction=1.0,
            recoverable_loss_column=0,
            recoverable_loss_fraction=0.0,
            non_recoverable_loss_column=0,
            non_recoverable_loss_fraction=0.0,
            spill_column=2,
            spill_fraction=0.5,
            delivery_dest_type=0,
            delivery_dest_id=5,
            delivery_column=0,
            delivery_fraction=1.0,
            irrigation_fraction_column=0,
            adjustment_column=0,
            name="SpillMulti",
        )
        streams.diversions = {1: div}
        streams.diversion_has_spills = True
        streams.diversion_element_groups = []
        streams.diversion_recharge_zones = []

        # Spill zone with 3 zones (first + 2 continuation rows)
        sz = SimpleNamespace(
            diversion_id=1,
            n_zones=3,
            zone_ids=[50, 60, 70],
            zone_fractions=[0.5, 0.3, 0.2],
        )
        streams.diversion_spill_zones = [sz]
        model.streams = streams

        engine = _mock_engine()
        config = StreamWriterConfig(output_dir=tmp_path)
        writer = StreamComponentWriter(model, config, template_engine=engine)
        path = writer.write_diver_specs()
        content = path.read_text()
        assert "Spill zones" in content
        assert "0.5000" in content
        assert "0.3000" in content
        assert "0.2000" in content
        # Check continuation row format (indented, no div_id prefix)
        assert "60" in content
        assert "70" in content
