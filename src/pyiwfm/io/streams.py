"""
Stream network I/O handlers for IWFM model files.

This module provides functions for reading and writing IWFM stream network
component files including stream nodes, reaches, diversions, bypasses, and
rating curves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence, TextIO

import numpy as np
from numpy.typing import NDArray

from pyiwfm.components.stream import (
    AppStream,
    StrmNode,
    StrmReach,
    Diversion,
    Bypass,
    StreamRating,
    CrossSectionData,
    StrmEvapNodeSpec,
)
from pyiwfm.core.timeseries import TimeSeries, TimeSeriesCollection
from pyiwfm.core.exceptions import FileFormatError
from pyiwfm.io.iwfm_reader import (
    COMMENT_CHARS,
    is_comment_line as _is_comment_line,
    next_data_or_empty as _next_data_or_empty_f,
    parse_version as parse_stream_version,
    resolve_path as _resolve_path_f,
    strip_inline_comment as _parse_value_line,
    version_ge as stream_version_ge,
)
from pyiwfm.io.timeseries_ascii import TimeSeriesWriter


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

        with open(filepath, "w") as f:
            # Write header
            if header:
                for line in header.strip().split("\n"):
                    f.write(f"C  {line}\n")
            else:
                f.write("C  Stream nodes file\n")
                f.write("C  Generated by pyiwfm\n")
                f.write("C\n")
                f.write("C  ID       X              Y       REACH  GW_NODE  BOT_ELEV  WET_PERM  UP_NODE  DN_NODE\n")

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

        with open(filepath, "w") as f:
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
                dest_type = reach.outflow_destination[0] if reach.outflow_destination else "boundary"
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

        with open(filepath, "w") as f:
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

        with open(filepath, "w") as f:
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

        with open(filepath, "w") as f:
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
            f.write(f"{len(nodes_with_ratings):<10}                              / N_RATING_CURVES\n")

            # Write each rating curve
            for node in nodes_with_ratings:
                rating = node.rating
                f.write(f"C\n")
                f.write(f"C  Rating curve for stream node {node.id}\n")
                f.write(f"{node.id:<6} {len(rating.stages):>5}  / NODE_ID, N_POINTS\n")
                f.write(f"C  STAGE         FLOW\n")

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
        writer.write(
            filepath=filepath,
            times=times,
            values=values,
            column_ids=node_ids,
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

        with open(filepath, "r") as f:
            line_num = 0
            n_nodes = None

            # Find NSTRNODES
            for line in f:
                line_num += 1
                if _is_comment_line(line):
                    continue

                value, _ = _parse_value_line(line)
                try:
                    n_nodes = int(value)
                except ValueError as e:
                    raise FileFormatError(
                        f"Invalid NSTRNODES value: '{value}'", line_number=line_num
                    ) from e
                break

            if n_nodes is None:
                raise FileFormatError("Could not find NSTRNODES in file")

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

        with open(filepath, "r") as f:
            line_num = 0
            n_diversions = None

            # Find NDIVERSIONS
            for line in f:
                line_num += 1
                if _is_comment_line(line):
                    continue

                value, _ = _parse_value_line(line)
                try:
                    n_diversions = int(value)
                except ValueError as e:
                    raise FileFormatError(
                        f"Invalid NDIVERSIONS value: '{value}'", line_number=line_num
                    ) from e
                break

            if n_diversions is None:
                raise FileFormatError("Could not find NDIVERSIONS in file")

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


# =============================================================================
# Component Main File Reader (hierarchical dispatcher file)
# =============================================================================


@dataclass
class StreamMainFileConfig:
    """
    Configuration parsed from Stream component main file.

    The stream component main file is a dispatcher that references
    sub-files for inflows, diversions, and bypasses. It also contains
    inline hydrograph output specifications.

    Attributes:
        version: File format version (e.g., "4.2")
        inflow_file: Path to stream inflow time series file
        diversion_spec_file: Path to diversion specifications file
        bypass_spec_file: Path to bypass specifications file
        diversion_file: Path to diversion time series file
        budget_output_file: Path to stream reach budget output
        diversion_budget_file: Path to diversion detail budget output
        hydrograph_count: Number of hydrograph output locations
        hydrograph_output_type: 0=flow, 1=stage, 2=both
        hydrograph_output_file: Path to hydrograph output file
        hydrograph_specs: List of (node_id, name) tuples for output locations
    """

    version: str = ""
    inflow_file: Path | None = None
    diversion_spec_file: Path | None = None
    bypass_spec_file: Path | None = None
    diversion_file: Path | None = None
    budget_output_file: Path | None = None
    diversion_budget_file: Path | None = None
    hydrograph_count: int = 0
    hydrograph_output_type: int = 0
    hydrograph_output_file: Path | None = None
    hydrograph_specs: list[tuple[int, str]] = field(default_factory=list)
    # v5.0 end-of-simulation flows file
    final_flow_file: Path | None = None
    # Hydrograph output factors
    hydrograph_flow_factor: float = 1.0
    hydrograph_flow_unit: str = ""
    hydrograph_elev_factor: float = 1.0
    hydrograph_elev_unit: str = ""
    # Stream node budget section
    node_budget_count: int = 0
    node_budget_output_file: Path | None = None
    node_budget_ids: list[int] = field(default_factory=list)
    # Stream bed parameters
    conductivity_factor: float = 1.0
    conductivity_time_unit: str = ""
    length_factor: float = 1.0
    bed_params: list[StreamBedParamRow] = field(default_factory=list)
    # Hydraulic disconnection
    interaction_type: int | None = None
    # Stream evaporation
    evap_area_file: Path | None = None
    evap_node_specs: list[tuple[int, int, int]] = field(default_factory=list)
    # v5.0 cross-section data
    roughness_factor: float = 1.0
    cross_section_length_factor: float = 1.0
    cross_section_data: list[CrossSectionRow] = field(default_factory=list)
    # v5.0 initial conditions
    ic_type: int = 0
    ic_time_unit: str = ""
    ic_factor: float = 1.0
    initial_conditions: list[StreamInitialConditionRow] = field(default_factory=list)


class StreamMainFileReader:
    """
    Reader for IWFM stream component main file.

    The Stream main file is a hierarchical dispatcher that contains:
    1. Version header (e.g., #4.2)
    2. Paths to sub-files (inflows, diversions, bypasses)
    3. Output file paths
    4. Inline hydrograph output specifications
    """

    def __init__(self) -> None:
        self._line_num = 0
        self._pushback_line: str | None = None

    def read(
        self, filepath: Path | str, base_dir: Path | None = None
    ) -> StreamMainFileConfig:
        """
        Parse Stream main file, extracting config and hydrograph specs.

        Args:
            filepath: Path to the Stream component main file
            base_dir: Base directory for resolving relative paths.
                     If None, uses the parent directory of filepath.

        Returns:
            StreamMainFileConfig with parsed values
        """
        filepath = Path(filepath)
        if base_dir is None:
            base_dir = filepath.parent

        config = StreamMainFileConfig()
        self._line_num = 0
        self._pushback_line = None

        with open(filepath, "r") as f:
            # Read version header
            config.version = self._read_version(f)

            # INFLOWFL (inflow time series file)
            inflow_path = self._next_data_or_empty(f)
            if inflow_path:
                config.inflow_file = self._resolve_path(base_dir, inflow_path)

            # DIVSPECFL (diversion specification file)
            divspec_path = self._next_data_or_empty(f)
            if divspec_path:
                config.diversion_spec_file = self._resolve_path(base_dir, divspec_path)

            # BYPSPECFL (bypass specification file)
            bypspec_path = self._next_data_or_empty(f)
            if bypspec_path:
                config.bypass_spec_file = self._resolve_path(base_dir, bypspec_path)

            # DIVFL (diversion time series file)
            div_path = self._next_data_or_empty(f)
            if div_path:
                config.diversion_file = self._resolve_path(base_dir, div_path)

            # STRMRCHBUDFL (stream reach budget output file)
            budget_path = self._next_data_or_empty(f)
            if budget_path:
                config.budget_output_file = self._resolve_path(base_dir, budget_path)

            # DIVDTLBUDFL (diversion detail budget output file)
            divbud_path = self._next_data_or_empty(f)
            if divbud_path:
                config.diversion_budget_file = self._resolve_path(
                    base_dir, divbud_path
                )

            # v5.0: end-of-simulation flows file (before hydrographs)
            if stream_version_ge(config.version, (5, 0)):
                final_flow_path = self._next_data_or_empty(f)
                if final_flow_path:
                    config.final_flow_file = self._resolve_path(
                        base_dir, final_flow_path
                    )

            # NOUTR (number of hydrograph output nodes)
            noutr_str = self._next_data_or_empty(f)
            if not noutr_str:
                return config

            try:
                config.hydrograph_count = int(noutr_str)
            except ValueError:
                return config

            if config.hydrograph_count <= 0:
                # Still need to read remaining sections
                self._read_post_hydrograph_sections(f, config, base_dir)
                return config

            # IHSQR (hydrograph output type: 0=flow, 1=stage, 2=both)
            ihsqr = self._next_data_or_empty(f)
            if ihsqr:
                try:
                    config.hydrograph_output_type = int(ihsqr)
                except ValueError:
                    pass

            # FACTSQOU (flow output conversion factor)
            factsqou = self._next_data_or_empty(f)
            if factsqou:
                try:
                    config.hydrograph_flow_factor = float(factsqou)
                except ValueError:
                    pass

            # UNITSQOU (flow output units)
            config.hydrograph_flow_unit = self._next_data_or_empty(f)

            # If stage output is included (type 1=stage, 2=both)
            if config.hydrograph_output_type in (1, 2):
                factltou = self._next_data_or_empty(f)
                if factltou:
                    try:
                        config.hydrograph_elev_factor = float(factltou)
                    except ValueError:
                        pass
                config.hydrograph_elev_unit = self._next_data_or_empty(f)

            # STRMHYDOUTFL (hydrograph output file)
            hydout_path = self._next_data_or_empty(f)
            if hydout_path:
                config.hydrograph_output_file = self._resolve_path(
                    base_dir, hydout_path
                )

            # Read inline hydrograph output specifications
            config.hydrograph_specs = self._read_hydrograph_specs(
                f, config.hydrograph_count
            )

            # Read remaining sections after hydrographs
            self._read_post_hydrograph_sections(f, config, base_dir)

        return config

    def _read_post_hydrograph_sections(
        self, f: TextIO, config: StreamMainFileConfig, base_dir: Path
    ) -> None:
        """Read all sections that follow the hydrograph specifications."""
        # Stream node budget section
        self._read_node_budget_section(f, config, base_dir)

        # Stream bed parameters section (version-dependent columns)
        self._read_bed_params_section(f, config)

        # v5.0: cross-section data and initial conditions
        if stream_version_ge(config.version, (5, 0)):
            self._read_cross_section_data(f, config)
            self._read_initial_conditions(f, config)

        # Stream evaporation section (all versions)
        self._read_evaporation_section(f, config, base_dir)

    def _read_node_budget_section(
        self, f: TextIO, config: StreamMainFileConfig, base_dir: Path
    ) -> None:
        """Read stream node budget section: NBUDR, budget file, node IDs."""
        nbudr_str = self._maybe_read_pushback(f)
        if not nbudr_str:
            return
        try:
            config.node_budget_count = int(nbudr_str)
        except ValueError:
            return
        if config.node_budget_count <= 0:
            return
        # Budget output file
        bud_path = self._next_data_or_empty(f)
        if bud_path:
            config.node_budget_output_file = self._resolve_path(base_dir, bud_path)
        # Per-node IDs
        for _ in range(config.node_budget_count):
            node_str = self._next_data_or_empty(f)
            if node_str:
                try:
                    config.node_budget_ids.append(int(node_str))
                except ValueError:
                    break

    def _read_bed_params_section(
        self, f: TextIO, config: StreamMainFileConfig
    ) -> None:
        """Read stream bed parameters: FACTK, TUNITSK, FACTL, per-node rows.

        Column layout depends on version:
        v4.2:  IR  WETPR  IRGW  CSTRM  DSTRM  (5 columns)
        v4.0:  IR  CSTRM  DSTRM  WETPR         (4 columns)
        v4.1:  IR  CSTRM  DSTRM                (3 columns)
        v5.0:  IR  CSTRM  DSTRM                (3 columns)
        """
        # FACTK
        factk_str = self._maybe_read_pushback(f)
        if not factk_str:
            return
        try:
            config.conductivity_factor = float(factk_str)
        except ValueError:
            return

        # TUNITSK
        config.conductivity_time_unit = self._next_data_or_empty(f)

        # FACTL
        factl_str = self._next_data_or_empty(f)
        if factl_str:
            try:
                config.length_factor = float(factl_str)
            except ValueError:
                pass

        # Determine column layout based on version
        # v4.2 uses 5 columns; v5.0+ uses 3 columns (same as v4.1)
        version = parse_stream_version(config.version) if config.version else (4, 0)
        is_v42 = (4, 2) <= version < (5, 0)
        is_v40 = version < (4, 1)
        if is_v42:
            min_cols = 5  # IR, WETPR, IRGW, CSTRM, DSTRM
        elif is_v40:
            min_cols = 4  # IR, CSTRM, DSTRM, WETPR
        else:
            min_cols = 3  # IR, CSTRM, DSTRM

        # Per-node bed parameter rows
        # Auto-detect actual column count from first data row
        detected_ncols = None
        while True:
            line_val = self._next_data_or_empty(f)
            if not line_val:
                break
            parts = line_val.split()

            # Auto-detect on first row: if 5 columns, treat as v4.2
            if detected_ncols is None and len(parts) >= 5:
                detected_ncols = len(parts)
                if not is_v42:
                    is_v42 = True
                    is_v40 = False
                    min_cols = 5
            elif detected_ncols is None:
                detected_ncols = len(parts)

            if len(parts) < min_cols:
                # Likely INTRCTYPE (1 column) — save for next read
                self._pushback_line = line_val
                break
            try:
                row = StreamBedParamRow(node_id=int(parts[0]))
                if is_v42:
                    # v4.2: IR, WETPR, IRGW, CSTRM, DSTRM
                    row.wetted_perimeter = float(parts[1])
                    row.gw_node = int(float(parts[2]))
                    row.conductivity = float(parts[3])
                    row.bed_thickness = float(parts[4])
                elif is_v40:
                    # v4.0: IR, CSTRM, DSTRM, WETPR
                    row.conductivity = float(parts[1])
                    row.bed_thickness = float(parts[2])
                    row.wetted_perimeter = float(parts[3])
                else:
                    # v4.1/v5.0: IR, CSTRM, DSTRM
                    row.conductivity = float(parts[1])
                    row.bed_thickness = float(parts[2])
                config.bed_params.append(row)
            except (ValueError, IndexError):
                self._pushback_line = line_val
                break

        # Read INTRCTYPE
        intrc_str = self._maybe_read_pushback(f)
        if intrc_str:
            try:
                config.interaction_type = int(intrc_str.split()[0])
            except (ValueError, IndexError):
                pass

    def _read_cross_section_data(
        self, f: TextIO, config: StreamMainFileConfig
    ) -> None:
        """Read v5.0 cross-section data: FACTN, FACTLT, per-node 6-col rows."""
        # FACTN (roughness conversion factor)
        factn_str = self._maybe_read_pushback(f)
        if not factn_str:
            return
        try:
            config.roughness_factor = float(factn_str)
        except ValueError:
            return

        # FACTLT (length conversion factor for cross-section)
        factlt_str = self._next_data_or_empty(f)
        if factlt_str:
            try:
                config.cross_section_length_factor = float(factlt_str)
            except ValueError:
                pass

        # Per-node cross-section rows (6 columns: IR BottomElev B0 s n MaxDepth)
        while True:
            line_val = self._next_data_or_empty(f)
            if not line_val:
                break
            parts = line_val.split()
            if len(parts) < 6:
                self._pushback_line = line_val
                break
            try:
                row = CrossSectionRow(
                    node_id=int(parts[0]),
                    bottom_elev=float(parts[1]),
                    B0=float(parts[2]),
                    s=float(parts[3]),
                    n=float(parts[4]),
                    max_flow_depth=float(parts[5]),
                )
                config.cross_section_data.append(row)
            except (ValueError, IndexError):
                self._pushback_line = line_val
                break

    def _read_initial_conditions(
        self, f: TextIO, config: StreamMainFileConfig
    ) -> None:
        """Read v5.0 initial conditions: ICType, TimeUnit, FACTH, per-node rows."""
        # IC Type
        ic_str = self._maybe_read_pushback(f)
        if not ic_str:
            return
        try:
            config.ic_type = int(ic_str.split()[0])
        except (ValueError, IndexError):
            return

        # Time unit (for flow IC)
        config.ic_time_unit = self._next_data_or_empty(f)

        # FACTH (conversion factor)
        facth_str = self._next_data_or_empty(f)
        if facth_str:
            try:
                config.ic_factor = float(facth_str)
            except ValueError:
                pass

        # Per-node IC rows (2 columns: IR value)
        while True:
            line_val = self._next_data_or_empty(f)
            if not line_val:
                break
            parts = line_val.split()
            if len(parts) < 2:
                self._pushback_line = line_val
                break
            try:
                row = StreamInitialConditionRow(
                    node_id=int(parts[0]),
                    value=float(parts[1]),
                )
                config.initial_conditions.append(row)
            except (ValueError, IndexError):
                self._pushback_line = line_val
                break

    def _read_evaporation_section(
        self, f: TextIO, config: StreamMainFileConfig, base_dir: Path
    ) -> None:
        """Read stream evaporation: STARFL (area file), per-node 3-col rows."""
        # STARFL (stream surface area file)
        area_path = self._maybe_read_pushback(f)
        if area_path:
            config.evap_area_file = self._resolve_path(base_dir, area_path)

        # Per-node evap specs (3 columns: IR ICETST ICARST)
        while True:
            line_val = self._next_data_or_empty(f)
            if not line_val:
                break
            parts = line_val.split()
            if len(parts) < 3:
                break
            try:
                config.evap_node_specs.append((
                    int(parts[0]),
                    int(parts[1]),
                    int(parts[2]),
                ))
            except (ValueError, IndexError):
                break

    def _maybe_read_pushback(self, f: TextIO) -> str:
        """Read pushback line if available, otherwise next data line."""
        if self._pushback_line is not None:
            val = self._pushback_line
            self._pushback_line = None
            return val
        return self._next_data_or_empty(f)

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
            break
        return ""

    def _next_data_or_empty(self, f: TextIO) -> str:
        """Return next data value, or empty string for blank lines."""
        lc = [self._line_num]
        val = _next_data_or_empty_f(f, lc)
        self._line_num = lc[0]
        return val

    def _read_hydrograph_specs(
        self, f: TextIO, n_hydrographs: int
    ) -> list[tuple[int, str]]:
        """
        Read inline hydrograph output specifications.

        Format per line: IOUTR  NAME
        - IOUTR: Stream node ID for output
        - NAME: Optional name/description

        Args:
            f: Open file handle
            n_hydrographs: Number of specifications to read

        Returns:
            List of (node_id, name) tuples
        """
        specs: list[tuple[int, str]] = []
        count = 0

        for line in f:
            self._line_num += 1
            if _is_comment_line(line):
                continue

            parts = line.split(None, 1)  # Split on first whitespace
            if not parts:
                continue

            try:
                node_id = int(parts[0])
                name = parts[1].strip() if len(parts) > 1 else ""
                specs.append((node_id, name))

                count += 1
                if count >= n_hydrographs:
                    break

            except ValueError:
                continue

        return specs

    @staticmethod
    def _resolve_path(base_dir: Path, filepath: str) -> Path:
        """Resolve a file path relative to base directory."""
        return _resolve_path_f(base_dir, filepath)


# =============================================================================
# StreamsSpec File Reader (preprocessor stream geometry)
# =============================================================================


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
    node_rating_tables: dict[int, tuple[list[float], list[float]]] = field(
        default_factory=dict
    )
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

    def read(
        self, filepath: Path | str
    ) -> tuple[int, int, list[StreamReachSpec]]:
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

        with open(filepath, "r") as f:
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
            for i in range(n_reaches):
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
            value, _ = _parse_value_line(line)
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


def read_stream_main_file(
    filepath: Path | str, base_dir: Path | None = None
) -> StreamMainFileConfig:
    """
    Read IWFM stream component main file.

    The Stream main file is a hierarchical dispatcher that contains paths
    to sub-files (inflows, diversions, bypasses) and inline hydrograph
    output specifications.

    Args:
        filepath: Path to the Stream component main file
        base_dir: Base directory for resolving relative paths.
                 If None, uses the parent directory of filepath.

    Returns:
        StreamMainFileConfig with parsed values

    Example:
        >>> config = read_stream_main_file("C2VSimFG_Streams.dat")
        >>> print(f"Version: {config.version}")
        >>> print(f"Hydrograph outputs: {config.hydrograph_count}")
    """
    reader = StreamMainFileReader()
    return reader.read(filepath, base_dir)


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
