"""
Stream network I/O handlers for IWFM model files.

This module provides functions for reading and writing IWFM stream network
component files including stream nodes, reaches, diversions, bypasses, and
rating curves.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from pyiwfm.components.stream import (
    AppStream,
    Diversion,
    StrmNode,
)
from pyiwfm.core.exceptions import FileFormatError
from pyiwfm.io.ascii.reader import (
    is_comment_line as _is_comment_line,
)
from pyiwfm.io.ascii.reader import (
    parse_int,
)
from pyiwfm.io.ascii.reader import (
    strip_inline_comment as _strip_comment,
)
from pyiwfm.io.timeseries.ascii import TimeSeriesWriter

# =============================================================================
# New dataclasses for parsed stream sections
# =============================================================================


@dataclass
class StreamBedParamRow:
    """Per-node stream bed parameters from the main file.

    v4.2 column order: IR, WETPR, IRGW, CSTRM, DSTRM (5 columns)
    v4.0 column order: IR, CSTRM, DSTRM, WETPR (4 columns)
    v4.1/v5.0: IR, CSTRM, DSTRM (3 columns)
    """

    node_id: int
    conductivity: float = 0.0
    bed_thickness: float = 0.0
    wetted_perimeter: float | None = None
    gw_node: int = 0


@dataclass
class CrossSectionRow:
    """Per-node v5.0 cross-section data from the main file."""

    node_id: int
    bottom_elev: float = 0.0
    B0: float = 0.0
    s: float = 0.0
    n: float = 0.04
    max_flow_depth: float = 10.0


@dataclass
class StreamInitialConditionRow:
    """Per-node v5.0 initial condition."""

    node_id: int
    value: float = 0.0


@dataclass
class StreamFileConfig:
    """
    Configuration for stream component files.

    Attributes:
        output_dir: Directory for output files
        stream_nodes_file: Stream nodes file name
        reaches_file: Reaches definition file name
        diversions_file: Diversions file name
        bypasses_file: Bypasses file name
        rating_curves_file: Rating curves file name
        inflows_file: Inflow time series file name
    """

    output_dir: Path
    stream_nodes_file: str = "stream_nodes.dat"
    reaches_file: str = "reaches.dat"
    diversions_file: str = "diversions.dat"
    bypasses_file: str = "bypasses.dat"
    rating_curves_file: str = "rating_curves.dat"
    inflows_file: str = "stream_inflows.dat"

    def get_stream_nodes_path(self) -> Path:
        return self.output_dir / self.stream_nodes_file

    def get_reaches_path(self) -> Path:
        return self.output_dir / self.reaches_file

    def get_diversions_path(self) -> Path:
        return self.output_dir / self.diversions_file

    def get_bypasses_path(self) -> Path:
        return self.output_dir / self.bypasses_file

    def get_rating_curves_path(self) -> Path:
        return self.output_dir / self.rating_curves_file

    def get_inflows_path(self) -> Path:
        return self.output_dir / self.inflows_file


class StreamWriter:
    """
    Writer for IWFM stream network component files.

    Writes all stream-related input files including nodes, reaches,
    diversions, bypasses, and rating curves.

    Example:
        >>> config = StreamFileConfig(output_dir=Path("./model"))
        >>> writer = StreamWriter(config)
        >>> files = writer.write(stream_component)
    """

    def __init__(self, config: StreamFileConfig) -> None:
        """
        Initialize the stream writer.

        Args:
            config: File configuration
        """
        self.config = config
        config.output_dir.mkdir(parents=True, exist_ok=True)

    def write(self, stream: AppStream) -> dict[str, Path]:
        """
        Write all stream component files.

        Args:
            stream: AppStream component to write

        Returns:
            Dictionary mapping file type to output path
        """
        files: dict[str, Path] = {}

        # Write stream nodes
        if stream.nodes:
            files["stream_nodes"] = self.write_stream_nodes(stream)

        # Write reaches
        if stream.reaches:
            files["reaches"] = self.write_reaches(stream)

        # Write diversions
        if stream.diversions:
            files["diversions"] = self.write_diversions(stream)

        # Write bypasses
        if stream.bypasses:
            files["bypasses"] = self.write_bypasses(stream)

        # Write rating curves for nodes that have them
        nodes_with_ratings = [n for n in stream.nodes.values() if n.rating is not None]
        if nodes_with_ratings:
            files["rating_curves"] = self.write_rating_curves(stream)

        return files

    def write_stream_nodes(self, stream: AppStream, header: str | None = None) -> Path:
        """
        Write stream nodes file.

        Args:
            stream: AppStream component
            header: Optional header comment

        Returns:
            Path to written file
        """
        filepath = self.config.get_stream_nodes_path()

        with open(filepath, "w", encoding="utf-8") as f:
            # Write header
            if header:
                for line in header.strip().split("\n"):
                    f.write(f"C  {line}\n")
            else:
                f.write("C  Stream nodes file\n")
                f.write("C  Generated by pyiwfm\n")
                f.write("C\n")
                f.write(
                    "C  ID       X              Y       REACH  GW_NODE  BOT_ELEV  WET_PERM  UP_NODE  DN_NODE\n"
                )

            # Write node count
            f.write(f"{len(stream.nodes):<10}                              / NSTRNODES\n")

            # Write nodes in ID order
            for node_id in sorted(stream.nodes.keys()):
                node = stream.nodes[node_id]
                gw_node = node.gw_node if node.gw_node else 0
                up_node = node.upstream_node if node.upstream_node else 0
                dn_node = node.downstream_node if node.downstream_node else 0

                f.write(
                    f"{node.id:<6} {node.x:>14.4f} {node.y:>14.4f} "
                    f"{node.reach_id:>5} {gw_node:>7} {node.bottom_elev:>10.2f} "
                    f"{node.wetted_perimeter:>8.2f} {up_node:>7} {dn_node:>7}\n"
                )

        return filepath

    def write_reaches(self, stream: AppStream, header: str | None = None) -> Path:
        """
        Write reaches definition file.

        Args:
            stream: AppStream component
            header: Optional header comment

        Returns:
            Path to written file
        """
        filepath = self.config.get_reaches_path()

        with open(filepath, "w", encoding="utf-8") as f:
            # Write header
            if header:
                for line in header.strip().split("\n"):
                    f.write(f"C  {line}\n")
            else:
                f.write("C  Stream reaches file\n")
                f.write("C  Generated by pyiwfm\n")
                f.write("C\n")
                f.write("C  ID  UP_NODE  DN_NODE  DEST_TYPE  DEST_ID  NAME\n")

            # Write reach count
            f.write(f"{len(stream.reaches):<10}                              / NREACHES\n")

            # Write reaches in ID order
            for reach_id in sorted(stream.reaches.keys()):
                reach = stream.reaches[reach_id]
                dest_type = (
                    reach.outflow_destination[0] if reach.outflow_destination else "boundary"
                )
                dest_id = reach.outflow_destination[1] if reach.outflow_destination else 0

                f.write(
                    f"{reach.id:<6} {reach.upstream_node:>7} {reach.downstream_node:>7} "
                    f"{dest_type:<12} {dest_id:>6}  {reach.name}\n"
                )

            # Write reach node lists
            f.write("C\n")
            f.write("C  Reach node assignments (REACH_ID followed by node list)\n")
            f.write("C\n")

            for reach_id in sorted(stream.reaches.keys()):
                reach = stream.reaches[reach_id]
                f.write(f"C  Reach {reach_id}: {reach.name}\n")
                f.write(f"{reach_id:<6} {len(reach.nodes):>5}  / REACH_ID, NNODES\n")
                # Write node IDs (10 per line)
                for i, nid in enumerate(reach.nodes):
                    if i > 0 and i % 10 == 0:
                        f.write("\n")
                    f.write(f"{nid:>7}")
                f.write("\n")

        return filepath

    def write_diversions(self, stream: AppStream, header: str | None = None) -> Path:
        """
        Write diversions file.

        Args:
            stream: AppStream component
            header: Optional header comment

        Returns:
            Path to written file
        """
        filepath = self.config.get_diversions_path()

        with open(filepath, "w", encoding="utf-8") as f:
            # Write header
            if header:
                for line in header.strip().split("\n"):
                    f.write(f"C  {line}\n")
            else:
                f.write("C  Stream diversions file\n")
                f.write("C  Generated by pyiwfm\n")
                f.write("C\n")
                f.write("C  ID  SRC_NODE  DEST_TYPE  DEST_ID  MAX_RATE  PRIORITY  NAME\n")

            # Write diversion count
            f.write(f"{len(stream.diversions):<10}                              / NDIVERSIONS\n")

            # Write diversions in ID order
            for div_id in sorted(stream.diversions.keys()):
                div = stream.diversions[div_id]
                f.write(
                    f"{div.id:<6} {div.source_node:>7} {div.destination_type:<12} "
                    f"{div.destination_id:>6} {div.max_rate:>12.4f} {div.priority:>4}  {div.name}\n"
                )

        return filepath

    def write_bypasses(self, stream: AppStream, header: str | None = None) -> Path:
        """
        Write bypasses file.

        Args:
            stream: AppStream component
            header: Optional header comment

        Returns:
            Path to written file
        """
        filepath = self.config.get_bypasses_path()

        with open(filepath, "w", encoding="utf-8") as f:
            # Write header
            if header:
                for line in header.strip().split("\n"):
                    f.write(f"C  {line}\n")
            else:
                f.write("C  Stream bypasses file\n")
                f.write("C  Generated by pyiwfm\n")
                f.write("C\n")
                f.write("C  ID  SRC_NODE  DST_NODE  CAPACITY  NAME\n")

            # Write bypass count
            f.write(f"{len(stream.bypasses):<10}                              / NBYPASSES\n")

            # Write bypasses in ID order
            for bypass_id in sorted(stream.bypasses.keys()):
                bypass = stream.bypasses[bypass_id]
                f.write(
                    f"{bypass.id:<6} {bypass.source_node:>7} {bypass.destination_node:>7} "
                    f"{bypass.capacity:>12.4f}  {bypass.name}\n"
                )

        return filepath

    def write_rating_curves(self, stream: AppStream, header: str | None = None) -> Path:
        """
        Write rating curves file.

        Args:
            stream: AppStream component
            header: Optional header comment

        Returns:
            Path to written file
        """
        filepath = self.config.get_rating_curves_path()

        with open(filepath, "w", encoding="utf-8") as f:
            # Write header
            if header:
                for line in header.strip().split("\n"):
                    f.write(f"C  {line}\n")
            else:
                f.write("C  Stream rating curves file\n")
                f.write("C  Generated by pyiwfm\n")
                f.write("C\n")

            # Count nodes with rating curves
            nodes_with_ratings = [n for n in stream.nodes.values() if n.rating is not None]
            f.write(
                f"{len(nodes_with_ratings):<10}                              / N_RATING_CURVES\n"
            )

            # Write each rating curve
            for node in nodes_with_ratings:
                rating = node.rating
                assert rating is not None
                f.write("C\n")
                f.write(f"C  Rating curve for stream node {node.id}\n")
                f.write(f"{node.id:<6} {len(rating.stages):>5}  / NODE_ID, N_POINTS\n")
                f.write("C  STAGE         FLOW\n")

                for i in range(len(rating.stages)):
                    f.write(f"{rating.stages[i]:>12.4f} {rating.flows[i]:>14.4f}\n")

        return filepath

    def write_inflows_timeseries(
        self,
        filepath: Path | str,
        times: Sequence[datetime],
        inflows: dict[int, NDArray[np.float64]],
        node_ids: list[int] | None = None,
        units: str = "CFS",
        factor: float = 1.0,
        header: str | None = None,
    ) -> Path:
        """
        Write stream inflows time series file.

        Args:
            filepath: Output file path
            times: Sequence of datetime values
            inflows: Dictionary mapping node ID to inflow array
            node_ids: Order of node IDs (default: sorted)
            units: Units string
            factor: Conversion factor
            header: Optional header comment

        Returns:
            Path to written file
        """
        filepath = Path(filepath)

        if node_ids is None:
            node_ids = sorted(inflows.keys())

        n_times = len(times)
        n_nodes = len(node_ids)

        # Build values array
        values = np.zeros((n_times, n_nodes))
        for i, nid in enumerate(node_ids):
            if nid in inflows:
                values[:, i] = inflows[nid]

        writer = TimeSeriesWriter()
        column_ids_mixed: list[str | int] = list(node_ids)
        writer.write(
            filepath=filepath,
            times=times,
            values=values,
            column_ids=column_ids_mixed,
            units=units,
            factor=factor,
            header=header or "Stream inflow time series file\nGenerated by pyiwfm",
        )

        return filepath


class StreamReader:
    """
    Reader for IWFM stream network component files.
    """

    def read_stream_nodes(self, filepath: Path | str) -> dict[int, StrmNode]:
        """
        Read stream nodes from file.

        Args:
            filepath: Path to stream nodes file

        Returns:
            Dictionary mapping node ID to StrmNode object
        """
        filepath = Path(filepath)
        nodes: dict[int, StrmNode] = {}

        with open(filepath, encoding="utf-8") as f:
            line_num = 0
            n_nodes = None

            # Find NSTRNODES
            for line in f:
                line_num += 1
                if _is_comment_line(line):
                    continue

                value, _ = _strip_comment(line)
                n_nodes = parse_int(value, context="NSTRNODES", line_number=line_num)
                break

            if n_nodes is None:
                raise FileFormatError("Required keyword 'NSTRNODES' not found anywhere in file")

            # Read node data
            for line in f:
                line_num += 1
                if _is_comment_line(line):
                    continue

                parts = line.split()
                if len(parts) < 6:
                    continue

                try:
                    node_id = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    reach_id = int(parts[3])
                    gw_node = int(parts[4]) if len(parts) > 4 else 0
                    bottom_elev = float(parts[5]) if len(parts) > 5 else 0.0
                    wetted_perimeter = float(parts[6]) if len(parts) > 6 else 0.0
                    up_node = int(parts[7]) if len(parts) > 7 and parts[7] != "0" else None
                    dn_node = int(parts[8]) if len(parts) > 8 and parts[8] != "0" else None

                    nodes[node_id] = StrmNode(
                        id=node_id,
                        x=x,
                        y=y,
                        reach_id=reach_id,
                        gw_node=gw_node if gw_node != 0 else None,
                        bottom_elev=bottom_elev,
                        wetted_perimeter=wetted_perimeter,
                        upstream_node=up_node,
                        downstream_node=dn_node,
                    )

                except ValueError as e:
                    raise FileFormatError(
                        f"Invalid stream node data: '{line.strip()}'",
                        line_number=line_num,
                    ) from e

        return nodes

    def read_diversions(self, filepath: Path | str) -> dict[int, Diversion]:
        """
        Read diversions from file.

        Args:
            filepath: Path to diversions file

        Returns:
            Dictionary mapping diversion ID to Diversion object
        """
        filepath = Path(filepath)
        diversions: dict[int, Diversion] = {}

        with open(filepath, encoding="utf-8") as f:
            line_num = 0
            n_diversions = None

            # Find NDIVERSIONS
            for line in f:
                line_num += 1
                if _is_comment_line(line):
                    continue

                value, _ = _strip_comment(line)
                n_diversions = parse_int(value, context="NDIVERSIONS", line_number=line_num)
                break

            if n_diversions is None:
                raise FileFormatError("Required keyword 'NDIVERSIONS' not found anywhere in file")

            # Read diversion data
            for line in f:
                line_num += 1
                if _is_comment_line(line):
                    continue

                parts = line.split()
                if len(parts) < 6:
                    continue

                try:
                    div_id = int(parts[0])
                    source_node = int(parts[1])
                    dest_type = parts[2]
                    dest_id = int(parts[3])
                    max_rate = float(parts[4])
                    priority = int(parts[5])
                    name = " ".join(parts[6:]) if len(parts) > 6 else ""

                    diversions[div_id] = Diversion(
                        id=div_id,
                        source_node=source_node,
                        destination_type=dest_type,
                        destination_id=dest_id,
                        max_rate=max_rate,
                        priority=priority,
                        name=name,
                    )

                except ValueError as e:
                    raise FileFormatError(
                        f"Invalid diversion data: '{line.strip()}'",
                        line_number=line_num,
                    ) from e

        return diversions


# Convenience functions


def write_stream(
    stream: AppStream,
    output_dir: Path | str,
    config: StreamFileConfig | None = None,
) -> dict[str, Path]:
    """
    Write stream component to files.

    Args:
        stream: AppStream component to write
        output_dir: Output directory
        config: Optional file configuration

    Returns:
        Dictionary mapping file type to output path
    """
    output_dir = Path(output_dir)

    if config is None:
        config = StreamFileConfig(output_dir=output_dir)
    else:
        config.output_dir = output_dir

    writer = StreamWriter(config)
    return writer.write(stream)


def read_stream_nodes(filepath: Path | str) -> dict[int, StrmNode]:
    """
    Read stream nodes from file.

    Args:
        filepath: Path to stream nodes file

    Returns:
        Dictionary mapping node ID to StrmNode object
    """
    reader = StreamReader()
    return reader.read_stream_nodes(filepath)


def read_diversions(filepath: Path | str) -> dict[int, Diversion]:
    """
    Read diversions from file.

    Args:
        filepath: Path to diversions file

    Returns:
        Dictionary mapping diversion ID to Diversion object
    """
    reader = StreamReader()
    return reader.read_diversions(filepath)
