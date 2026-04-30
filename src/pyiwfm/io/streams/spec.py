"""
StreamsSpec preprocessor file reader.

The IWFM preprocessor StreamsSpec file defines the stream network
geometry: number of reaches, rating-table points, reach-to-node
assignment, and stream-to-GW node mappings. This module hosts the
:class:`StreamReachSpec` dataclass and the :class:`StreamSpecReader`
class.

Split out of :mod:`pyiwfm.io.streams.reader` in v2.0. See
``docs/MIGRATION_v1_to_v2.md`` § 10.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

from pyiwfm.core.exceptions import FileFormatError
from pyiwfm.io.ascii.reader import COMMENT_CHARS
from pyiwfm.io.ascii.reader import (
    is_comment_line as _is_comment_line,
)
from pyiwfm.io.ascii.reader import (
    parse_version as parse_stream_version,
)
from pyiwfm.io.ascii.reader import (
    strip_inline_comment as _strip_comment,
)


@dataclass
class StreamReachSpec:
    """
    Stream reach specification from preprocessor StreamsSpec file.

    Contains the reach definition including node-to-GW-node mappings
    for stream-aquifer interaction.

    Attributes:
        id: Reach ID
        n_nodes: Number of stream nodes in this reach
        outflow_node: Outflow destination (0=boundary, -n=lake n, +n=reach n)
        name: Reach name/description
        node_ids: List of stream node IDs in this reach
        node_to_gw_node: Mapping of stream_node_id -> gw_node_id
        node_rating_tables: Maps stream_node_id -> (stages, flows)
        node_bottom_elevations: Maps stream_node_id -> bottom elevation
    """

    id: int
    n_nodes: int
    outflow_node: int = 0
    name: str = ""
    node_ids: list[int] = field(default_factory=list)
    node_to_gw_node: dict[int, int] = field(default_factory=dict)
    node_rating_tables: dict[int, tuple[list[float], list[float]]] = field(default_factory=dict)
    node_bottom_elevations: dict[int, float] = field(default_factory=dict)


class StreamSpecReader:
    """
    Reader for IWFM preprocessor StreamsSpec file.

    The StreamsSpec file defines the stream network geometry including:
    - Number of reaches and rating table points
    - Reach definitions with node lists
    - Stream-GW node mappings for each stream node
    """

    def __init__(self) -> None:
        self._line_num = 0

    def read(self, filepath: Path | str) -> tuple[int, int, list[StreamReachSpec]]:
        """
        Parse StreamsSpec file.

        Args:
            filepath: Path to the StreamsSpec file

        Returns:
            Tuple of (n_reaches, n_rating_points, list of reach specs)
        """
        filepath = Path(filepath)
        self._line_num = 0

        reach_specs: list[StreamReachSpec] = []
        n_reaches = 0
        n_rating_points = 0

        with open(filepath, encoding="utf-8") as f:
            # Read version header (optional)
            version = self._read_version(f)

            # NRH (number of reaches)
            nrh_str = self._next_data_line(f)
            try:
                n_reaches = int(nrh_str)
            except ValueError as e:
                raise FileFormatError(
                    f"Invalid NRH value: '{nrh_str}'", line_number=self._line_num
                ) from e

            # v5.0 has no NRTB (no rating tables — uses Manning's equation)
            ver = parse_stream_version(version) if version else (4, 0)
            if ver >= (5, 0):
                n_rating_points = 0
            else:
                # NRTB (number of rating table points)
                nrtb_str = self._next_data_line(f)
                try:
                    n_rating_points = int(nrtb_str)
                except ValueError as e:
                    raise FileFormatError(
                        f"Invalid NRTB value: '{nrtb_str}'",
                        line_number=self._line_num,
                    ) from e

            # Read reach specifications (node-GW mappings only; no
            # interleaved rating tables — those are in a separate section)
            for _i in range(n_reaches):
                reach = self._read_reach_spec(f)
                reach_specs.append(reach)

            # Read rating tables (separate section after ALL reaches)
            if n_rating_points > 0:
                self._read_rating_tables(f, n_rating_points, reach_specs)

            # Read optional partial interaction section
            self._read_partial_interaction(f)

        return n_reaches, n_rating_points, reach_specs

    def _read_version(self, f: TextIO) -> str:
        """Read the version header from the file."""
        for line in f:
            self._line_num += 1
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                return stripped[1:].strip()
            if line[0] in COMMENT_CHARS:
                continue
            # First data line - put back by returning empty version
            # and using _next_data_line to re-read
            break
        return ""

    def _next_data_line(self, f: TextIO) -> str:
        """Return next non-comment line with data."""
        for line in f:
            self._line_num += 1
            if _is_comment_line(line):
                continue
            value, _ = _strip_comment(line)
            if value:
                return value
        raise FileFormatError("Unexpected end of file", line_number=self._line_num)

    def _read_reach_spec(self, f: TextIO) -> StreamReachSpec:
        """
        Read a single reach specification.

        Each reach has:
        1. Reach header line: ID, NSNRH, IOUTRH, NAME
        2. Stream node data lines (one per node): stream_node_id, gw_node_id(s)

        Rating tables are in a separate section after ALL reaches (not
        interleaved with node data).
        """
        # Read reach header
        header_line = self._next_data_line(f)
        parts = header_line.split()

        if len(parts) < 3:
            raise FileFormatError(
                f"Invalid reach header: '{header_line}'",
                line_number=self._line_num,
            )

        reach_id = int(parts[0])
        n_nodes = int(parts[1])
        outflow_node = int(parts[2])
        name = " ".join(parts[3:]) if len(parts) > 3 else ""

        reach = StreamReachSpec(
            id=reach_id,
            n_nodes=n_nodes,
            outflow_node=outflow_node,
            name=name,
        )

        # Read node data for this reach
        for _ in range(n_nodes):
            node_line = self._next_data_line(f)
            node_parts = node_line.split()

            if len(node_parts) >= 2:
                stream_node_id = int(node_parts[0])
                gw_node_id = int(node_parts[1])

                reach.node_ids.append(stream_node_id)
                if gw_node_id > 0:
                    reach.node_to_gw_node[stream_node_id] = gw_node_id

        return reach

    def _read_rating_tables(
        self,
        f: TextIO,
        n_rating_points: int,
        reach_specs: list[StreamReachSpec],
    ) -> None:
        """Read the rating table section after all reach definitions.

        IWFM format (v4.x only — v5.0 uses Manning's equation):

            FACTLT                          (length conversion factor)
            FACTQ                           (flow conversion factor)
            TUNIT                           (time unit)
            node_id  bottom_elev  depth  flow   (first point, 4 columns)
                                  depth  flow   (NRTB-1 continuation lines)
            node_id  bottom_elev  depth  flow   (next node)
            ...
        """
        # Build node_id → reach lookup
        node_to_reach: dict[int, StreamReachSpec] = {}
        for rs in reach_specs:
            for nid in rs.node_ids:
                node_to_reach[nid] = rs

        # Read FACTLT, FACTQ, TUNIT header values
        try:
            self._next_data_line(f)  # FACTLT
            self._next_data_line(f)  # FACTQ
            self._next_data_line(f)  # TUNIT
        except FileFormatError:
            return  # No rating table section found

        # Read rating data for each stream node
        total_nodes = sum(rs.n_nodes for rs in reach_specs)
        for _ in range(total_nodes):
            try:
                first_line = self._next_data_line(f)
            except FileFormatError:
                break

            parts = first_line.split()
            if len(parts) < 4:
                break

            try:
                node_id = int(parts[0])
                bottom_elev = float(parts[1])
                stages: list[float] = [float(parts[2])]
                flows: list[float] = [float(parts[3])]
            except (ValueError, IndexError):
                break

            # Read remaining NRTB-1 continuation lines
            for _ in range(n_rating_points - 1):
                try:
                    rt_line = self._next_data_line(f)
                except FileFormatError:
                    break
                rt_parts = rt_line.split()
                if len(rt_parts) >= 2:
                    try:
                        stages.append(float(rt_parts[0]))
                        flows.append(float(rt_parts[1]))
                    except ValueError:
                        pass

            # Assign to the correct reach
            if node_id in node_to_reach:
                rs = node_to_reach[node_id]
                rs.node_rating_tables[node_id] = (stages, flows)
                rs.node_bottom_elevations[node_id] = bottom_elev

    def _read_partial_interaction(self, f: TextIO) -> None:
        """Read optional partial stream-aquifer interaction section.

        Format:
            NSTRPINT  (number of partial interaction nodes; 0 = none)
            node_id  fraction   (per-node entries, if NSTRPINT > 0)
        """
        try:
            nstrpint_str = self._next_data_line(f)
            nstrpint = int(nstrpint_str)
            if nstrpint <= 0:
                return
            for _ in range(nstrpint):
                self._next_data_line(f)
        except (FileFormatError, ValueError):
            return  # Section not present or end of file


# Convenience function


def read_stream_spec(
    filepath: Path | str,
) -> tuple[int, int, list[StreamReachSpec]]:
    """
    Read IWFM preprocessor StreamsSpec file.

    The StreamsSpec file defines the stream network geometry including
    reach definitions and stream-GW node mappings.

    Args:
        filepath: Path to the StreamsSpec file

    Returns:
        Tuple of (n_reaches, n_rating_points, list of StreamReachSpec)

    Example:
        >>> n_reaches, n_rtb, reaches = read_stream_spec("StreamsSpec.dat")
        >>> print(f"Loaded {n_reaches} reaches")
        >>> for reach in reaches:
        ...     print(f"  Reach {reach.id}: {reach.n_nodes} nodes")
    """
    reader = StreamSpecReader()
    return reader.read(filepath)
