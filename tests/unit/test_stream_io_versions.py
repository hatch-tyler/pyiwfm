"""Tests for stream I/O across all IWFM versions (v4.0, v4.1, v5.0).

Tests cover:
- Version utility functions
- StreamMainFileReader for v4.0, v4.1, v5.0 main files
- StreamSpecReader for v5.0 (no NRTB)
- StreamComponentWriter for v4.0 and v5.0
- Round-trip read-write consistency
- New data model classes (CrossSectionData, StrmEvapNodeSpec, expanded StrmNode/AppStream)
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from pyiwfm.io.streams import (
    StreamMainFileReader,
    StreamMainFileConfig,
    StreamSpecReader,
    StreamBedParamRow,
    CrossSectionRow,
    StreamInitialConditionRow,
    parse_stream_version,
    stream_version_ge,
)
from pyiwfm.components.stream import (
    AppStream,
    StrmNode,
    StrmReach,
    CrossSectionData,
    StrmEvapNodeSpec,
)


# =============================================================================
# Helpers: write sample stream main files for each version
# =============================================================================


def _write_stream_main_v40(path: Path, n_nodes: int = 3) -> None:
    """Write a minimal v4.0 stream main file with all sections."""
    lines = textwrap.dedent("""\
        #4.0
        C *** DO NOT DELETE ABOVE LINE ***
        C
            StreamInflow.dat               / INFLOWFL
            DiverSpecs.dat                 / DIVSPECFL
            BypassSpecs.dat                / BYPSPECFL
            Diversions.dat                 / DIVFL
            StrmBud.hdf                    / STRMRCHBUDFL
            DiverDetail.hdf                / DIVDTLBUDFL
        C*******************************************************************************
        C                       Hydrograph Output
        C
            2                              / NOUTR
            0                              / IHSQR
            1.0                            / FACTSQOU
            cfs                            / UNITSQOU
            StrmHyd.out                    / STHYDOUTFL
        C
            1     Node_1
            2     Node_2
        C*******************************************************************************
        C                       Stream Node Budget
        C
            2                              / NBUDR
            StrmNodeBud.hdf                / STNDBUDFL
        C
            1
            3
        C*******************************************************************************
        C                       Stream Bed Parameters
        C
            2.0                            / FACTK
            1day                           / TUNITSK
            1.5                            / FACTL
        C
    """)
    # v4.0: 4 columns (IR, K, BedThick, WetPerim)
    for i in range(1, n_nodes + 1):
        lines += f"    {i}    10.0    1.0    150.0\n"
    lines += textwrap.dedent("""\
        C*******************************************************************************
        C                Hydraulic Disconnection
        C
            1                              / INTRCTYPE
        C*******************************************************************************
        C                       Stream Evaporation
        C
            EvapArea.dat                   / STARFL
        C
            1    3    4
            2    5    6
    """)
    path.write_text(lines)


def _write_stream_main_v41(path: Path, n_nodes: int = 3) -> None:
    """Write a minimal v4.1 stream main file (3-col bed params, no WetPerim)."""
    lines = textwrap.dedent("""\
        #4.1
        C *** DO NOT DELETE ABOVE LINE ***
        C
            StreamInflow.dat               / INFLOWFL
            DiverSpecs.dat                 / DIVSPECFL
            BypassSpecs.dat                / BYPSPECFL
            Diversions.dat                 / DIVFL
            StrmBud.hdf                    / STRMRCHBUDFL
            DiverDetail.hdf                / DIVDTLBUDFL
        C*******************************************************************************
            2                              / NOUTR
            0                              / IHSQR
            1.0                            / FACTSQOU
            cfs                            / UNITSQOU
            StrmHyd.out                    / STHYDOUTFL
        C
            1     Node_1
            2     Node_2
        C*******************************************************************************
            1                              / NBUDR
            StrmNodeBud.hdf                / STNDBUDFL
        C
            1
        C*******************************************************************************
            1.0                            / FACTK
            1day                           / TUNITSK
            1.0                            / FACTL
        C
    """)
    # v4.1: 3 columns (IR, K, BedThick) — no WetPerim
    for i in range(1, n_nodes + 1):
        lines += f"    {i}    5.0    0.5\n"
    lines += textwrap.dedent("""\
        C*******************************************************************************
            2                              / INTRCTYPE
        C*******************************************************************************
                                           / STARFL
        C
    """)
    path.write_text(lines)


def _write_stream_main_v50(path: Path, n_nodes: int = 3) -> None:
    """Write a minimal v5.0 stream main file with cross-section and IC."""
    lines = textwrap.dedent("""\
        #5.0
        C *** DO NOT DELETE ABOVE LINE ***
        C
            StreamInflow.dat               / INFLOWFL
            DiverSpecs.dat                 / DIVSPECFL
            BypassSpecs.dat                / BYPSPECFL
            Diversions.dat                 / DIVFL
            StrmBud.hdf                    / STRMRCHBUDFL
            DiverDetail.hdf                / DIVDTLBUDFL
            EndSimFlows.dat                / ENDSIMFL
        C*******************************************************************************
            2                              / NOUTR
            2                              / IHSQR
            1.0                            / FACTSQOU
            cfs                            / UNITSQOU
            1.0                            / FACTLTOU
            ft                             / UNITLTOU
            StrmHyd.out                    / STHYDOUTFL
        C
            1     Node_1
            2     Node_2
        C*******************************************************************************
            1                              / NBUDR
            StrmNodeBud.hdf                / STNDBUDFL
        C
            2
        C*******************************************************************************
            1.0                            / FACTK
            1day                           / TUNITSK
            1.0                            / FACTL
        C
    """)
    # v5.0: 3 columns (IR, K, BedThick)
    for i in range(1, n_nodes + 1):
        lines += f"    {i}    8.0    0.8\n"
    lines += textwrap.dedent("""\
        C*******************************************************************************
            1                              / INTRCTYPE
        C*******************************************************************************
        C                       Cross-Section Data
        C
            1.0                            / FACTN
            1.0                            / FACTLT
        C
    """)
    # Cross-section rows (6 columns)
    for i in range(1, n_nodes + 1):
        lines += f"    {i}    100.0    5.0    0.5    0.035    15.0\n"
    lines += textwrap.dedent("""\
        C*******************************************************************************
        C                       Initial Conditions
        C
            0                              / ICTYPE
            1day                           / TUNITFLOW
            1.0                            / FACTH
        C
    """)
    # Initial condition rows (2 columns)
    for i in range(1, n_nodes + 1):
        lines += f"    {i}    {i * 1.5:.1f}\n"
    lines += textwrap.dedent("""\
        C*******************************************************************************
                                           / STARFL
        C
    """)
    path.write_text(lines)


def _write_stream_spec_v50(path: Path) -> None:
    """Write a v5.0 StreamsSpec file (no NRTB)."""
    lines = textwrap.dedent("""\
        #5.0
        C Stream geometry
        C
            2                              / NRH
        C
        C  Reach definitions (no NRTB for v5.0)
        C
            1    2    0    Reach_1
            1    10
            2    20
            2    1    1    Reach_2
            3    30
    """)
    path.write_text(lines)


def _write_stream_spec_v40(path: Path) -> None:
    """Write a v4.0 StreamsSpec file with NRTB and rating tables.

    Rating tables are in a separate section after all reach definitions.
    Each node's first rating line has 4 columns (node_id, bottom_elev,
    depth, flow); remaining NRTB-1 lines have 2 columns (depth, flow).
    """
    lines = textwrap.dedent("""\
        #4.0
        C Stream geometry
        C
            2                              / NRH
            2                              / NRTB
        C  Reach definitions
            1    2    0    Reach_1
            1    10
            2    20
            2    1    1    Reach_2
            3    30
        C  Rating table section
            1.0                            / FACTLT
            1.0                            / FACTQ
            1min                           / TUNIT
        C  Node 1
            1    100.0    0.0    0.0
                          1.0    10.0
        C  Node 2
            2    90.0     0.0    0.0
                          1.0    10.0
        C  Node 3
            3    80.0     0.0    0.0
                          1.0    10.0
    """)
    path.write_text(lines)


# =============================================================================
# Version Utility Tests
# =============================================================================


class TestStreamVersionUtils:
    """Tests for version parsing utilities."""

    def test_parse_version_40(self) -> None:
        assert parse_stream_version("4.0") == (4, 0)

    def test_parse_version_41(self) -> None:
        assert parse_stream_version("4.1") == (4, 1)

    def test_parse_version_421(self) -> None:
        assert parse_stream_version("4.21") == (4, 21)

    def test_parse_version_50(self) -> None:
        assert parse_stream_version("5.0") == (5, 0)

    def test_parse_version_with_dash(self) -> None:
        assert parse_stream_version("4-1") == (4, 1)

    def test_version_ge_equal(self) -> None:
        assert stream_version_ge("4.1", (4, 1)) is True

    def test_version_ge_greater(self) -> None:
        assert stream_version_ge("5.0", (4, 1)) is True

    def test_version_ge_less(self) -> None:
        assert stream_version_ge("4.0", (4, 1)) is False

    def test_version_ge_minor(self) -> None:
        assert stream_version_ge("4.21", (4, 2)) is True


# =============================================================================
# StreamMainFileReader Tests
# =============================================================================


class TestStreamMainFileReaderV40:
    """Tests for reading v4.0 stream main files."""

    def test_reads_file_paths(self, tmp_path: Path) -> None:
        main_file = tmp_path / "stream.dat"
        _write_stream_main_v40(main_file, n_nodes=3)
        reader = StreamMainFileReader()
        config = reader.read(main_file)

        assert config.version == "4.0"
        assert config.inflow_file is not None
        assert config.inflow_file.name == "StreamInflow.dat"
        assert config.diversion_spec_file is not None
        assert config.bypass_spec_file is not None
        assert config.diversion_file is not None

    def test_reads_hydrograph_specs(self, tmp_path: Path) -> None:
        main_file = tmp_path / "stream.dat"
        _write_stream_main_v40(main_file, n_nodes=3)
        reader = StreamMainFileReader()
        config = reader.read(main_file)

        assert config.hydrograph_count == 2
        assert config.hydrograph_output_type == 0
        assert len(config.hydrograph_specs) == 2
        assert config.hydrograph_specs[0] == (1, "Node_1")

    def test_reads_hydrograph_factors(self, tmp_path: Path) -> None:
        main_file = tmp_path / "stream.dat"
        _write_stream_main_v40(main_file, n_nodes=3)
        reader = StreamMainFileReader()
        config = reader.read(main_file)

        assert config.hydrograph_flow_factor == 1.0
        assert config.hydrograph_flow_unit == "cfs"

    def test_reads_budget_nodes(self, tmp_path: Path) -> None:
        main_file = tmp_path / "stream.dat"
        _write_stream_main_v40(main_file, n_nodes=3)
        reader = StreamMainFileReader()
        config = reader.read(main_file)

        assert config.node_budget_count == 2
        assert config.node_budget_ids == [1, 3]

    def test_reads_bed_params_4_cols(self, tmp_path: Path) -> None:
        main_file = tmp_path / "stream.dat"
        _write_stream_main_v40(main_file, n_nodes=3)
        reader = StreamMainFileReader()
        config = reader.read(main_file)

        assert config.conductivity_factor == 2.0
        assert config.conductivity_time_unit == "1day"
        assert config.length_factor == 1.5
        assert len(config.bed_params) == 3
        assert config.bed_params[0].node_id == 1
        assert config.bed_params[0].conductivity == 10.0
        assert config.bed_params[0].bed_thickness == 1.0
        assert config.bed_params[0].wetted_perimeter == 150.0

    def test_reads_interaction_type(self, tmp_path: Path) -> None:
        main_file = tmp_path / "stream.dat"
        _write_stream_main_v40(main_file, n_nodes=3)
        reader = StreamMainFileReader()
        config = reader.read(main_file)

        assert config.interaction_type == 1

    def test_reads_evaporation(self, tmp_path: Path) -> None:
        main_file = tmp_path / "stream.dat"
        _write_stream_main_v40(main_file, n_nodes=3)
        reader = StreamMainFileReader()
        config = reader.read(main_file)

        assert config.evap_area_file is not None
        assert len(config.evap_node_specs) == 2
        assert config.evap_node_specs[0] == (1, 3, 4)
        assert config.evap_node_specs[1] == (2, 5, 6)


class TestStreamMainFileReaderV41:
    """Tests for reading v4.1 stream main files."""

    def test_reads_bed_params_3_cols(self, tmp_path: Path) -> None:
        main_file = tmp_path / "stream.dat"
        _write_stream_main_v41(main_file, n_nodes=3)
        reader = StreamMainFileReader()
        config = reader.read(main_file)

        assert config.version == "4.1"
        assert len(config.bed_params) == 3
        assert config.bed_params[0].conductivity == 5.0
        assert config.bed_params[0].bed_thickness == 0.5
        assert config.bed_params[0].wetted_perimeter is None

    def test_reads_interaction_type_2(self, tmp_path: Path) -> None:
        main_file = tmp_path / "stream.dat"
        _write_stream_main_v41(main_file, n_nodes=3)
        reader = StreamMainFileReader()
        config = reader.read(main_file)

        assert config.interaction_type == 2


class TestStreamMainFileReaderV50:
    """Tests for reading v5.0 stream main files."""

    def test_reads_final_flow_file(self, tmp_path: Path) -> None:
        main_file = tmp_path / "stream.dat"
        _write_stream_main_v50(main_file, n_nodes=3)
        reader = StreamMainFileReader()
        config = reader.read(main_file)

        assert config.version == "5.0"
        assert config.final_flow_file is not None
        assert config.final_flow_file.name == "EndSimFlows.dat"

    def test_reads_stage_output_factors(self, tmp_path: Path) -> None:
        main_file = tmp_path / "stream.dat"
        _write_stream_main_v50(main_file, n_nodes=3)
        reader = StreamMainFileReader()
        config = reader.read(main_file)

        # IHSQR=2 means both flow and stage
        assert config.hydrograph_output_type == 2
        assert config.hydrograph_flow_factor == 1.0
        assert config.hydrograph_flow_unit == "cfs"
        assert config.hydrograph_elev_factor == 1.0
        assert config.hydrograph_elev_unit == "ft"

    def test_reads_cross_section_data(self, tmp_path: Path) -> None:
        main_file = tmp_path / "stream.dat"
        _write_stream_main_v50(main_file, n_nodes=3)
        reader = StreamMainFileReader()
        config = reader.read(main_file)

        assert config.roughness_factor == 1.0
        assert config.cross_section_length_factor == 1.0
        assert len(config.cross_section_data) == 3
        cs = config.cross_section_data[0]
        assert cs.node_id == 1
        assert cs.bottom_elev == 100.0
        assert cs.B0 == 5.0
        assert cs.s == 0.5
        assert cs.n == 0.035
        assert cs.max_flow_depth == 15.0

    def test_reads_initial_conditions(self, tmp_path: Path) -> None:
        main_file = tmp_path / "stream.dat"
        _write_stream_main_v50(main_file, n_nodes=3)
        reader = StreamMainFileReader()
        config = reader.read(main_file)

        assert config.ic_type == 0
        assert config.ic_time_unit == "1day"
        assert config.ic_factor == 1.0
        assert len(config.initial_conditions) == 3
        assert config.initial_conditions[0].node_id == 1
        assert config.initial_conditions[0].value == 1.5
        assert config.initial_conditions[1].value == 3.0
        assert config.initial_conditions[2].value == 4.5

    def test_reads_bed_params_3_cols(self, tmp_path: Path) -> None:
        main_file = tmp_path / "stream.dat"
        _write_stream_main_v50(main_file, n_nodes=3)
        reader = StreamMainFileReader()
        config = reader.read(main_file)

        assert len(config.bed_params) == 3
        assert config.bed_params[0].conductivity == 8.0
        assert config.bed_params[0].bed_thickness == 0.8
        assert config.bed_params[0].wetted_perimeter is None


# =============================================================================
# StreamSpecReader Tests
# =============================================================================


class TestStreamSpecReaderV50:
    """Tests for StreamSpecReader with v5.0 format (no NRTB)."""

    def test_reads_v50_no_nrtb(self, tmp_path: Path) -> None:
        spec_file = tmp_path / "StreamsSpec.dat"
        _write_stream_spec_v50(spec_file)
        reader = StreamSpecReader()
        n_reaches, n_rtb, reaches = reader.read(spec_file)

        assert n_reaches == 2
        assert n_rtb == 0
        assert len(reaches) == 2
        assert reaches[0].id == 1
        assert reaches[0].n_nodes == 2
        assert reaches[0].node_ids == [1, 2]
        assert reaches[0].node_to_gw_node == {1: 10, 2: 20}

    def test_v40_with_rating_tables(self, tmp_path: Path) -> None:
        spec_file = tmp_path / "StreamsSpec.dat"
        _write_stream_spec_v40(spec_file)
        reader = StreamSpecReader()
        n_reaches, n_rtb, reaches = reader.read(spec_file)

        assert n_reaches == 2
        assert n_rtb == 2
        assert len(reaches) == 2
        assert 1 in reaches[0].node_rating_tables
        assert len(reaches[0].node_rating_tables[1][0]) == 2


# =============================================================================
# Data Model Tests
# =============================================================================


class TestCrossSectionData:
    """Tests for CrossSectionData dataclass."""

    def test_defaults(self) -> None:
        cs = CrossSectionData()
        assert cs.bottom_elev == 0.0
        assert cs.B0 == 0.0
        assert cs.s == 0.0
        assert cs.n == 0.04
        assert cs.max_flow_depth == 10.0

    def test_custom_values(self) -> None:
        cs = CrossSectionData(bottom_elev=50.0, B0=10.0, s=1.5, n=0.035, max_flow_depth=20.0)
        assert cs.bottom_elev == 50.0
        assert cs.B0 == 10.0
        assert cs.s == 1.5
        assert cs.n == 0.035
        assert cs.max_flow_depth == 20.0


class TestStrmEvapNodeSpec:
    """Tests for StrmEvapNodeSpec dataclass."""

    def test_creation(self) -> None:
        spec = StrmEvapNodeSpec(node_id=5, et_column=3, area_column=7)
        assert spec.node_id == 5
        assert spec.et_column == 3
        assert spec.area_column == 7

    def test_defaults(self) -> None:
        spec = StrmEvapNodeSpec(node_id=1)
        assert spec.et_column == 0
        assert spec.area_column == 0


class TestStrmNodeExpansion:
    """Tests for expanded StrmNode fields."""

    def test_backward_compatible_defaults(self) -> None:
        node = StrmNode(id=1, x=0.0, y=0.0)
        assert node.conductivity == 0.0
        assert node.bed_thickness == 0.0
        assert node.cross_section is None
        assert node.initial_condition == 0.0

    def test_with_cross_section(self) -> None:
        cs = CrossSectionData(bottom_elev=100.0, B0=5.0)
        node = StrmNode(id=1, x=0.0, y=0.0, cross_section=cs)
        assert node.cross_section is not None
        assert node.cross_section.bottom_elev == 100.0

    def test_with_all_new_fields(self) -> None:
        node = StrmNode(
            id=1, x=0.0, y=0.0,
            conductivity=10.0,
            bed_thickness=1.5,
            initial_condition=3.0,
        )
        assert node.conductivity == 10.0
        assert node.bed_thickness == 1.5
        assert node.initial_condition == 3.0


class TestAppStreamExpansion:
    """Tests for expanded AppStream fields."""

    def test_backward_compatible_defaults(self) -> None:
        stream = AppStream()
        assert stream.conductivity_factor == 1.0
        assert stream.conductivity_time_unit == ""
        assert stream.length_factor == 1.0
        assert stream.interaction_type == 1
        assert stream.evap_area_file == ""
        assert stream.evap_node_specs == []
        assert stream.ic_type == 0
        assert stream.ic_factor == 1.0
        assert stream.final_flow_file == ""
        assert stream.budget_node_count == 0
        assert stream.budget_node_ids == []
        assert stream.roughness_factor == 1.0
        assert stream.cross_section_length_factor == 1.0

    def test_existing_functionality_preserved(self) -> None:
        stream = AppStream()
        node = StrmNode(id=1, x=100.0, y=200.0, reach_id=1)
        stream.add_node(node)
        assert stream.n_nodes == 1
        assert stream.get_node(1) is node


# =============================================================================
# Writer Tests
# =============================================================================


class TestStreamWriterV40:
    """Tests for stream writer with v4.0 format."""

    def test_write_v40_bed_params_4_cols(self, tmp_path: Path) -> None:
        from pyiwfm.io.stream_writer import StreamComponentWriter, StreamWriterConfig
        from pyiwfm.core.model import IWFMModel

        stream = AppStream()
        for i in range(1, 4):
            stream.add_node(StrmNode(
                id=i, x=float(i * 100), y=float(i * 100),
                reach_id=1, wetted_perimeter=120.0,
            ))

        model = IWFMModel(name="test", streams=stream)
        config = StreamWriterConfig(output_dir=tmp_path, version="4.0")
        writer = StreamComponentWriter(model, config)
        path = writer.write_main()

        content = path.read_text()
        assert "#4.0" in content
        assert "FACTK" in content
        assert "WETPR" in content

    def test_write_v40_interaction_type(self, tmp_path: Path) -> None:
        from pyiwfm.io.stream_writer import StreamComponentWriter, StreamWriterConfig
        from pyiwfm.core.model import IWFMModel

        stream = AppStream()
        stream.add_node(StrmNode(id=1, x=0.0, y=0.0))
        model = IWFMModel(name="test", streams=stream)
        config = StreamWriterConfig(output_dir=tmp_path, version="4.0")
        writer = StreamComponentWriter(model, config)
        path = writer.write_main()

        content = path.read_text()
        assert "INTRCTYPE" in content

    def test_write_v40_evaporation_section(self, tmp_path: Path) -> None:
        from pyiwfm.io.stream_writer import StreamComponentWriter, StreamWriterConfig
        from pyiwfm.core.model import IWFMModel

        stream = AppStream()
        stream.add_node(StrmNode(id=1, x=0.0, y=0.0))
        stream.evap_area_file = "test_area.dat"
        stream.evap_node_specs = [StrmEvapNodeSpec(node_id=1, et_column=2, area_column=3)]

        model = IWFMModel(name="test", streams=stream)
        config = StreamWriterConfig(output_dir=tmp_path, version="4.0")
        writer = StreamComponentWriter(model, config)
        path = writer.write_main()

        content = path.read_text()
        assert "STARFL" in content
        assert "test_area.dat" in content


class TestStreamWriterV50:
    """Tests for stream writer with v5.0 format."""

    def test_write_v50_cross_section(self, tmp_path: Path) -> None:
        from pyiwfm.io.stream_writer import StreamComponentWriter, StreamWriterConfig
        from pyiwfm.core.model import IWFMModel

        stream = AppStream()
        for i in range(1, 4):
            node = StrmNode(id=i, x=float(i * 100), y=float(i * 100))
            node.cross_section = CrossSectionData(
                bottom_elev=100.0 - i * 10,
                B0=5.0, s=0.5, n=0.035, max_flow_depth=15.0,
            )
            stream.add_node(node)

        model = IWFMModel(name="test", streams=stream)
        config = StreamWriterConfig(output_dir=tmp_path, version="5.0")
        writer = StreamComponentWriter(model, config)
        path = writer.write_main()

        content = path.read_text()
        assert "#5.0" in content
        assert "FACTN" in content
        assert "FACTLT" in content
        assert "ENDSIMFL" in content

    def test_write_v50_initial_conditions(self, tmp_path: Path) -> None:
        from pyiwfm.io.stream_writer import StreamComponentWriter, StreamWriterConfig
        from pyiwfm.core.model import IWFMModel

        stream = AppStream()
        for i in range(1, 4):
            node = StrmNode(id=i, x=float(i * 100), y=float(i * 100))
            node.initial_condition = float(i) * 1.5
            stream.add_node(node)

        model = IWFMModel(name="test", streams=stream)
        config = StreamWriterConfig(output_dir=tmp_path, version="5.0")
        writer = StreamComponentWriter(model, config)
        path = writer.write_main()

        content = path.read_text()
        assert "ICTYPE" in content
        assert "FACTH" in content

    def test_write_v50_no_wetted_perimeter(self, tmp_path: Path) -> None:
        from pyiwfm.io.stream_writer import StreamComponentWriter, StreamWriterConfig
        from pyiwfm.core.model import IWFMModel

        stream = AppStream()
        stream.add_node(StrmNode(id=1, x=0.0, y=0.0))
        model = IWFMModel(name="test", streams=stream)
        config = StreamWriterConfig(output_dir=tmp_path, version="5.0")
        writer = StreamComponentWriter(model, config)
        path = writer.write_main()

        content = path.read_text()
        assert "WETPR" not in content

    def test_write_v50_final_flow_file(self, tmp_path: Path) -> None:
        from pyiwfm.io.stream_writer import StreamComponentWriter, StreamWriterConfig
        from pyiwfm.core.model import IWFMModel

        stream = AppStream()
        stream.add_node(StrmNode(id=1, x=0.0, y=0.0))
        model = IWFMModel(name="test", streams=stream)
        config = StreamWriterConfig(
            output_dir=tmp_path, version="5.0",
            final_flow_file="EndSimFlows.dat",
        )
        writer = StreamComponentWriter(model, config)
        path = writer.write_main()

        content = path.read_text()
        assert "EndSimFlows.dat" in content


# =============================================================================
# Round-Trip Tests
# =============================================================================


class TestStreamRoundTrip:
    """Tests that verify read->write->read consistency."""

    def test_roundtrip_v40(self, tmp_path: Path) -> None:
        """Write v4.0 -> read -> verify key fields."""
        main_file = tmp_path / "stream_v40.dat"
        _write_stream_main_v40(main_file, n_nodes=3)

        reader = StreamMainFileReader()
        config = reader.read(main_file)

        assert config.version == "4.0"
        assert len(config.bed_params) == 3
        assert config.bed_params[0].wetted_perimeter == 150.0
        assert config.interaction_type == 1
        assert len(config.evap_node_specs) == 2
        assert config.node_budget_count == 2

    def test_roundtrip_v50(self, tmp_path: Path) -> None:
        """Write v5.0 -> read -> verify key fields."""
        main_file = tmp_path / "stream_v50.dat"
        _write_stream_main_v50(main_file, n_nodes=3)

        reader = StreamMainFileReader()
        config = reader.read(main_file)

        assert config.version == "5.0"
        assert config.final_flow_file is not None
        assert len(config.cross_section_data) == 3
        assert len(config.initial_conditions) == 3
        assert config.hydrograph_output_type == 2
        assert config.hydrograph_elev_factor == 1.0
        assert config.hydrograph_elev_unit == "ft"


# =============================================================================
# StreamBedParamRow / CrossSectionRow / StreamInitialConditionRow Tests
# =============================================================================


class TestStreamBedParamRow:
    """Tests for StreamBedParamRow dataclass."""

    def test_defaults(self) -> None:
        row = StreamBedParamRow(node_id=1)
        assert row.conductivity == 0.0
        assert row.bed_thickness == 0.0
        assert row.wetted_perimeter is None

    def test_with_values(self) -> None:
        row = StreamBedParamRow(
            node_id=5, conductivity=10.0, bed_thickness=1.5, wetted_perimeter=100.0
        )
        assert row.node_id == 5
        assert row.conductivity == 10.0
        assert row.wetted_perimeter == 100.0


class TestCrossSectionRow:
    """Tests for CrossSectionRow dataclass."""

    def test_defaults(self) -> None:
        row = CrossSectionRow(node_id=1)
        assert row.bottom_elev == 0.0
        assert row.B0 == 0.0
        assert row.n == 0.04
        assert row.max_flow_depth == 10.0

    def test_with_values(self) -> None:
        row = CrossSectionRow(
            node_id=3, bottom_elev=50.0, B0=8.0, s=0.5, n=0.035, max_flow_depth=20.0
        )
        assert row.node_id == 3
        assert row.B0 == 8.0


class TestStreamInitialConditionRow:
    """Tests for StreamInitialConditionRow dataclass."""

    def test_defaults(self) -> None:
        row = StreamInitialConditionRow(node_id=1)
        assert row.value == 0.0

    def test_with_value(self) -> None:
        row = StreamInitialConditionRow(node_id=2, value=5.5)
        assert row.value == 5.5
