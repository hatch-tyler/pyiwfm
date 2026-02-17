"""Supplementary tests for streams.py targeting uncovered branches.

Covers:
- StreamReader: read bypass parsing, read inflows
- StreamWriter: write_all with no nodes/reaches/diversions/bypasses
- StreamWriter: inflows with numpy datetime64
- write_stream convenience with existing config
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from pyiwfm.components.stream import (
    AppStream,
    Bypass,
    Diversion,
    StrmNode,
)
from pyiwfm.io.streams import (
    StreamFileConfig,
    StreamReader,
    StreamWriter,
    read_diversions,
    read_stream_nodes,
    write_stream,
)

# =============================================================================
# StreamWriter Edge Cases
# =============================================================================


class TestStreamWriterEdgeCasesAdditional:
    """Additional edge cases for StreamWriter."""

    def test_write_nodes_only(self, tmp_path: Path) -> None:
        """Test writing stream with only nodes (no reaches/diversions/bypasses)."""
        nodes = {
            1: StrmNode(id=1, x=100.0, y=200.0, reach_id=1, bottom_elev=50.0, wetted_perimeter=5.0),
        }
        stream = AppStream(nodes=nodes, reaches={}, diversions={}, bypasses={})

        config = StreamFileConfig(output_dir=tmp_path)
        writer = StreamWriter(config)
        files = writer.write(stream)

        assert "stream_nodes" in files
        assert "reaches" not in files
        assert "diversions" not in files
        assert "bypasses" not in files

    def test_write_nodes_with_no_gw_node(self, tmp_path: Path) -> None:
        """Test writing stream node without gw_node set."""
        nodes = {
            1: StrmNode(
                id=1,
                x=100.0,
                y=200.0,
                reach_id=1,
                bottom_elev=50.0,
                wetted_perimeter=5.0,
                gw_node=None,
            ),
        }
        stream = AppStream(nodes=nodes, reaches={}, diversions={}, bypasses={})

        config = StreamFileConfig(output_dir=tmp_path)
        writer = StreamWriter(config)
        filepath = writer.write_stream_nodes(stream)

        filepath.read_text()
        assert filepath.exists()

    def test_write_node_with_up_and_down_zero(self, tmp_path: Path) -> None:
        """Test writing node with upstream/downstream = None (becomes 0)."""
        nodes = {
            1: StrmNode(
                id=1,
                x=100.0,
                y=200.0,
                reach_id=1,
                bottom_elev=50.0,
                wetted_perimeter=5.0,
                upstream_node=None,
                downstream_node=None,
            ),
        }
        stream = AppStream(nodes=nodes, reaches={}, diversions={}, bypasses={})

        config = StreamFileConfig(output_dir=tmp_path)
        writer = StreamWriter(config)
        filepath = writer.write_stream_nodes(stream)

        content = filepath.read_text()
        # Node should have 0 for upstream and downstream
        assert "      0" in content

    def test_write_with_only_diversions(self, tmp_path: Path) -> None:
        """Test write() with nodes and diversions but no reaches."""
        nodes = {1: StrmNode(id=1, x=0.0, y=0.0, reach_id=1, bottom_elev=0.0, wetted_perimeter=1.0)}
        diversions = {
            1: Diversion(
                id=1,
                source_node=1,
                destination_type="element",
                destination_id=5,
                max_rate=100.0,
                priority=1,
                name="D1",
            ),
        }
        stream = AppStream(nodes=nodes, reaches={}, diversions=diversions, bypasses={})

        config = StreamFileConfig(output_dir=tmp_path)
        writer = StreamWriter(config)
        files = writer.write(stream)

        assert "stream_nodes" in files
        assert "diversions" in files
        assert "reaches" not in files

    def test_write_with_only_bypasses(self, tmp_path: Path) -> None:
        """Test write() with nodes and bypasses but no reaches/diversions."""
        nodes = {1: StrmNode(id=1, x=0.0, y=0.0, reach_id=1, bottom_elev=0.0, wetted_perimeter=1.0)}
        bypasses = {
            1: Bypass(id=1, source_node=1, destination_node=1, capacity=500.0, name="B1"),
        }
        stream = AppStream(nodes=nodes, reaches={}, diversions={}, bypasses=bypasses)

        config = StreamFileConfig(output_dir=tmp_path)
        writer = StreamWriter(config)
        files = writer.write(stream)

        assert "stream_nodes" in files
        assert "bypasses" in files
        assert "diversions" not in files

    def test_write_inflows_with_factor(self, tmp_path: Path) -> None:
        """Test writing inflows with custom units and factor."""
        config = StreamFileConfig(output_dir=tmp_path)
        writer = StreamWriter(config)

        times = [datetime(2020, 1, 1), datetime(2020, 2, 1)]
        inflows = {1: np.array([50.0, 60.0])}

        filepath = writer.write_inflows_timeseries(
            filepath=tmp_path / "inflows.dat",
            times=times,
            inflows=inflows,
            units="AF/DAY",
            factor=43560.0,
            header="Custom inflow header\nSecond line",
        )

        assert filepath.exists()


# =============================================================================
# StreamReader Additional Tests
# =============================================================================


class TestStreamReaderAdditional:
    """Additional reader tests for branch coverage."""

    def test_read_stream_nodes_with_gw_node_zero(self, tmp_path: Path) -> None:
        """Test reading node with gw_node=0 (becomes None)."""
        filepath = tmp_path / "nodes.dat"
        filepath.write_text("""C Stream nodes
1                               / NSTRNODES
1      100.0000      200.0000     1       0      50.00     5.00       0       0
""")

        reader = StreamReader()
        nodes = reader.read_stream_nodes(filepath)

        assert len(nodes) == 1
        assert nodes[1].gw_node is None

    def test_read_stream_nodes_with_up_dn_nodes(self, tmp_path: Path) -> None:
        """Test reading nodes with non-zero upstream/downstream."""
        filepath = tmp_path / "nodes.dat"
        filepath.write_text("""C Stream nodes
2                               / NSTRNODES
1      100.0      200.0     1      10      50.0      5.0       0       2
2      150.0      250.0     1      15      45.0      6.0       1       0
""")

        reader = StreamReader()
        nodes = reader.read_stream_nodes(filepath)

        assert nodes[1].upstream_node is None  # 0 becomes None
        assert nodes[1].downstream_node == 2
        assert nodes[2].upstream_node == 1
        assert nodes[2].downstream_node is None  # 0 becomes None

    def test_read_stream_nodes_minimal_columns(self, tmp_path: Path) -> None:
        """Test reading with exactly 6 columns (minimum required)."""
        filepath = tmp_path / "nodes.dat"
        filepath.write_text("""C Stream nodes
1                               / NSTRNODES
1      100.0000      200.0000     1       0      50.00
""")

        reader = StreamReader()
        nodes = reader.read_stream_nodes(filepath)

        assert len(nodes) == 1
        assert nodes[1].x == pytest.approx(100.0)

    def test_read_many_stream_nodes(self, tmp_path: Path) -> None:
        """Test reading many stream nodes."""
        lines = ["C Stream nodes", "50                              / NSTRNODES"]
        for i in range(1, 51):
            lines.append(f"{i}   {i * 10.0:.4f}   {i * 20.0:.4f}   1   0   50.0   5.0   0   0")

        filepath = tmp_path / "nodes.dat"
        filepath.write_text("\n".join(lines))

        reader = StreamReader()
        nodes = reader.read_stream_nodes(filepath)

        assert len(nodes) == 50
        assert nodes[50].x == pytest.approx(500.0)


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctionEdgeCases:
    """Additional convenience function tests."""

    def test_write_stream_with_custom_config(self, tmp_path: Path) -> None:
        """Test write_stream updates config output_dir."""
        nodes = {1: StrmNode(id=1, x=0.0, y=0.0, reach_id=1, bottom_elev=0.0, wetted_perimeter=1.0)}
        stream = AppStream(nodes=nodes, reaches={}, diversions={}, bypasses={})

        config = StreamFileConfig(
            output_dir=tmp_path / "wrong",
            stream_nodes_file="custom.dat",
        )

        files = write_stream(stream, tmp_path, config=config)

        assert "stream_nodes" in files
        assert files["stream_nodes"].name == "custom.dat"

    def test_read_stream_nodes_string_path(self, tmp_path: Path) -> None:
        """Test read_stream_nodes with string path."""
        filepath = tmp_path / "nodes.dat"
        filepath.write_text("""C Stream nodes
1                               / NSTRNODES
1      100.0      200.0     1       0      50.0      5.0       0       0
""")

        nodes = read_stream_nodes(str(filepath))
        assert len(nodes) == 1

    def test_read_diversions_string_path(self, tmp_path: Path) -> None:
        """Test read_diversions with string path."""
        filepath = tmp_path / "diversions.dat"
        filepath.write_text("""C Diversions
1                               / NDIVERSIONS
1        1 element         5      100.0    1  Div1
""")

        diversions = read_diversions(str(filepath))
        assert len(diversions) == 1
