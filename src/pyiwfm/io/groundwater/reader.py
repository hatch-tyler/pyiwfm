"""
Groundwater component I/O handlers for IWFM model files.

This module provides functions for reading and writing IWFM groundwater
component files including wells, pumping, boundary conditions, aquifer
parameters, tile drains, and subsidence data.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pyiwfm.components.groundwater import (
    AppGW,
    Subsidence,
    Well,
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


@dataclass
class KhAnomalyEntry:
    """Single Kh anomaly overwrite for one element.

    Attributes:
        element_id: 1-based element ID to overwrite.
        kh_per_layer: Kh values per layer, already multiplied by FACT.
    """

    element_id: int
    kh_per_layer: list[float]


@dataclass
class ParametricGridData:
    """Raw parametric grid data parsed from the GW main file.

    Attributes:
        n_nodes: Number of parametric grid nodes.
        n_elements: Number of parametric grid elements.
        elements: Element vertex index tuples (0-based into node arrays).
        node_coords: Parametric node coordinates, shape (n_nodes, 2).
        node_values: Parameter values per node, shape (n_nodes, n_layers, n_params).
            The 5 parameters are: Kh, Ss, Sy, AquitardKv, Kv.
        node_range_str: Raw node range string from file (e.g., "1-441").
        raw_node_lines: Raw text lines for each parametric node (before parsing).
    """

    n_nodes: int
    n_elements: int
    elements: list[tuple[int, ...]]
    node_coords: NDArray[np.float64]
    node_values: NDArray[np.float64]
    node_range_str: str = ""
    raw_node_lines: list[str] = field(default_factory=list)


@dataclass
class FaceFlowSpec:
    """Element face flow output specification.

    Parsed from the inline face flow data in the GW main file.
    Format per line: ID  IOUTFL  IOUTFA  IOUTFB  NAME

    Attributes:
        id: Face flow output ID.
        layer: Aquifer layer for output.
        node_a: First node defining the element face.
        node_b: Second node defining the element face.
        name: Optional description.
    """

    id: int
    layer: int
    node_a: int
    node_b: int
    name: str = ""


@dataclass
class GWFileConfig:
    """
    Configuration for groundwater component files.

    Attributes:
        output_dir: Directory for output files
        wells_file: Wells definition file name
        pumping_file: Pumping time series file name
        aquifer_params_file: Aquifer parameters file name
        boundary_conditions_file: Boundary conditions file name
        tile_drains_file: Tile drains file name
        subsidence_file: Subsidence parameters file name
        initial_heads_file: Initial heads file name
    """

    output_dir: Path
    wells_file: str = "wells.dat"
    pumping_file: str = "pumping.dat"
    aquifer_params_file: str = "aquifer_params.dat"
    boundary_conditions_file: str = "boundary_conditions.dat"
    tile_drains_file: str = "tile_drains.dat"
    subsidence_file: str = "subsidence.dat"
    initial_heads_file: str = "initial_heads.dat"

    def get_wells_path(self) -> Path:
        return self.output_dir / self.wells_file

    def get_pumping_path(self) -> Path:
        return self.output_dir / self.pumping_file

    def get_aquifer_params_path(self) -> Path:
        return self.output_dir / self.aquifer_params_file

    def get_boundary_conditions_path(self) -> Path:
        return self.output_dir / self.boundary_conditions_file

    def get_tile_drains_path(self) -> Path:
        return self.output_dir / self.tile_drains_file

    def get_subsidence_path(self) -> Path:
        return self.output_dir / self.subsidence_file

    def get_initial_heads_path(self) -> Path:
        return self.output_dir / self.initial_heads_file


class GroundwaterWriter:
    """
    Writer for IWFM groundwater component files.

    Writes all groundwater-related input files including wells, pumping
    time series, boundary conditions, aquifer parameters, etc.

    Example:
        >>> config = GWFileConfig(output_dir=Path("./model"))
        >>> writer = GroundwaterWriter(config)
        >>> files = writer.write(gw_component)
    """

    def __init__(self, config: GWFileConfig) -> None:
        """
        Initialize the groundwater writer.

        Args:
            config: File configuration
        """
        self.config = config
        config.output_dir.mkdir(parents=True, exist_ok=True)

    def write(self, gw: AppGW) -> dict[str, Path]:
        """
        Write all groundwater component files.

        Args:
            gw: AppGW component to write

        Returns:
            Dictionary mapping file type to output path
        """
        files: dict[str, Path] = {}

        # Write wells file if there are wells
        if gw.wells:
            files["wells"] = self.write_wells(gw)

        # Write aquifer parameters if available
        if gw.aquifer_params:
            files["aquifer_params"] = self.write_aquifer_params(gw)

        # Write boundary conditions if present
        if gw.boundary_conditions:
            files["boundary_conditions"] = self.write_boundary_conditions(gw)

        # Write tile drains if present
        if gw.tile_drains:
            files["tile_drains"] = self.write_tile_drains(gw)

        # Write subsidence if present
        if gw.subsidence:
            files["subsidence"] = self.write_subsidence(gw)

        # Write initial heads if available
        if gw.heads is not None:
            files["initial_heads"] = self.write_initial_heads(gw)

        return files

    def write_wells(self, gw: AppGW, header: str | None = None) -> Path:
        """
        Write wells definition file.

        Args:
            gw: AppGW component
            header: Optional header comment

        Returns:
            Path to written file
        """
        filepath = self.config.get_wells_path()

        with open(filepath, "w", encoding="utf-8") as f:
            # Write header
            if header:
                for line in header.strip().split("\n"):
                    f.write(f"C  {line}\n")
            else:
                f.write("C  Wells definition file\n")
                f.write("C  Generated by pyiwfm\n")
                f.write("C\n")
                f.write(
                    "C  ID      X              Y       ELEM  TOP_SCR  BOT_SCR  MAX_RATE  NAME\n"
                )

            # Write well count
            f.write(f"{len(gw.wells):<10}                              / NWELLS\n")

            # Write wells in ID order
            for well_id in sorted(gw.wells.keys()):
                well = gw.wells[well_id]
                f.write(
                    f"{well.id:<6} {well.x:>14.4f} {well.y:>14.4f} "
                    f"{well.element:>5} {well.top_screen:>8.2f} {well.bottom_screen:>8.2f} "
                    f"{well.max_pump_rate:>10.2f}  {well.name}\n"
                )

        return filepath

    def write_pumping_timeseries(
        self,
        filepath: Path | str,
        times: Sequence[datetime],
        pumping_rates: dict[int, NDArray[np.float64]],
        well_ids: list[int] | None = None,
        units: str = "TAF",
        factor: float = 1.0,
        header: str | None = None,
    ) -> Path:
        """
        Write pumping time series file.

        Args:
            filepath: Output file path
            times: Sequence of datetime values
            pumping_rates: Dictionary mapping well ID to pumping rate array
            well_ids: Order of well IDs (default: sorted)
            units: Units string
            factor: Conversion factor
            header: Optional header comment

        Returns:
            Path to written file
        """
        filepath = Path(filepath)

        if well_ids is None:
            well_ids = sorted(pumping_rates.keys())

        n_times = len(times)
        n_wells = len(well_ids)

        # Build values array
        values = np.zeros((n_times, n_wells))
        for i, wid in enumerate(well_ids):
            if wid in pumping_rates:
                values[:, i] = pumping_rates[wid]

        writer = TimeSeriesWriter()
        writer.write(
            filepath=filepath,
            times=times,
            values=values,
            column_ids=list(well_ids),
            units=units,
            factor=factor,
            header=header or "Pumping time series file\nGenerated by pyiwfm",
        )

        return filepath

    def write_aquifer_params(self, gw: AppGW, header: str | None = None) -> Path:
        """
        Write aquifer parameters file.

        Args:
            gw: AppGW component
            header: Optional header comment

        Returns:
            Path to written file
        """
        filepath = self.config.get_aquifer_params_path()
        params = gw.aquifer_params

        if params is None:
            raise ValueError("No aquifer parameters to write")

        with open(filepath, "w", encoding="utf-8") as f:
            # Write header
            if header:
                for line in header.strip().split("\n"):
                    f.write(f"C  {line}\n")
            else:
                f.write("C  Aquifer parameters file\n")
                f.write("C  Generated by pyiwfm\n")
                f.write("C\n")

            # Write dimensions
            f.write(f"{params.n_nodes:<10}                              / NNODES\n")
            f.write(f"{params.n_layers:<10}                              / NLAYERS\n")

            # Write parameter headers
            layer_cols = "  ".join(
                [
                    f"KH{i + 1:02d}  KV{i + 1:02d}  SS{i + 1:02d}  SY{i + 1:02d}"
                    for i in range(params.n_layers)
                ]
            )
            f.write(f"C  NODE  {layer_cols}\n")

            # Write parameter data
            for node_idx in range(params.n_nodes):
                node_id = node_idx + 1
                line = f"{node_id:<5}"

                for layer in range(params.n_layers):
                    kh = params.kh[node_idx, layer] if params.kh is not None else 0.0
                    kv = params.kv[node_idx, layer] if params.kv is not None else 0.0
                    ss = (
                        params.specific_storage[node_idx, layer]
                        if params.specific_storage is not None
                        else 0.0
                    )
                    sy = (
                        params.specific_yield[node_idx, layer]
                        if params.specific_yield is not None
                        else 0.0
                    )

                    line += f" {kh:>12.6f} {kv:>12.6f} {ss:>12.6e} {sy:>8.4f}"

                f.write(line + "\n")

        return filepath

    def write_boundary_conditions(self, gw: AppGW, header: str | None = None) -> Path:
        """
        Write boundary conditions file.

        Args:
            gw: AppGW component
            header: Optional header comment

        Returns:
            Path to written file
        """
        filepath = self.config.get_boundary_conditions_path()

        with open(filepath, "w", encoding="utf-8") as f:
            # Write header
            if header:
                for line in header.strip().split("\n"):
                    f.write(f"C  {line}\n")
            else:
                f.write("C  Groundwater boundary conditions file\n")
                f.write("C  Generated by pyiwfm\n")
                f.write("C\n")

            # Group BCs by type in a single pass
            from collections import defaultdict

            bc_groups: dict[str, list[Any]] = defaultdict(list)
            for bc in gw.boundary_conditions:
                bc_groups[bc.bc_type].append(bc)
            specified_head = bc_groups["specified_head"]
            specified_flow = bc_groups["specified_flow"]
            general_head = bc_groups["general_head"]

            # Write specified head BCs
            f.write("C  SPECIFIED HEAD BOUNDARY CONDITIONS\n")
            f.write(f"{len(specified_head):<10}                              / N_SPEC_HEAD_BC\n")

            for bc in specified_head:
                f.write(f"C  BC ID: {bc.id}, Layer: {bc.layer}, N_nodes: {len(bc.nodes)}\n")
                f.write(f"{bc.id:<6} {bc.layer:>3} {len(bc.nodes):>5}  / BC_ID, LAYER, NNODES\n")
                for i, node in enumerate(bc.nodes):
                    f.write(f"  {node:>6} {bc.values[i]:>14.4f}\n")

            # Write specified flow BCs
            f.write("C  SPECIFIED FLOW BOUNDARY CONDITIONS\n")
            f.write(f"{len(specified_flow):<10}                              / N_SPEC_FLOW_BC\n")

            for bc in specified_flow:
                f.write(f"C  BC ID: {bc.id}, Layer: {bc.layer}, N_nodes: {len(bc.nodes)}\n")
                f.write(f"{bc.id:<6} {bc.layer:>3} {len(bc.nodes):>5}  / BC_ID, LAYER, NNODES\n")
                for i, node in enumerate(bc.nodes):
                    f.write(f"  {node:>6} {bc.values[i]:>14.4f}\n")

            # Write general head BCs
            f.write("C  GENERAL HEAD BOUNDARY CONDITIONS\n")
            f.write(f"{len(general_head):<10}                              / N_GEN_HEAD_BC\n")

            for bc in general_head:
                f.write(f"C  BC ID: {bc.id}, Layer: {bc.layer}, N_nodes: {len(bc.nodes)}\n")
                f.write(f"{bc.id:<6} {bc.layer:>3} {len(bc.nodes):>5}  / BC_ID, LAYER, NNODES\n")
                for i, node in enumerate(bc.nodes):
                    cond = bc.conductance[i] if i < len(bc.conductance) else 0.0
                    f.write(f"  {node:>6} {bc.values[i]:>14.4f} {cond:>14.6f}\n")

        return filepath

    def write_tile_drains(self, gw: AppGW, header: str | None = None) -> Path:
        """
        Write tile drains file.

        Args:
            gw: AppGW component
            header: Optional header comment

        Returns:
            Path to written file
        """
        filepath = self.config.get_tile_drains_path()

        with open(filepath, "w", encoding="utf-8") as f:
            # Write header
            if header:
                for line in header.strip().split("\n"):
                    f.write(f"C  {line}\n")
            else:
                f.write("C  Tile drains file\n")
                f.write("C  Generated by pyiwfm\n")
                f.write("C\n")
                f.write("C  ID   ELEM     ELEV     CONDUCTANCE  DEST_TYPE  DEST_ID\n")

            # Write drain count
            f.write(f"{len(gw.tile_drains):<10}                              / NDRAINS\n")

            # Write drains in ID order
            for drain_id in sorted(gw.tile_drains.keys()):
                drain = gw.tile_drains[drain_id]
                dest_id = drain.destination_id if drain.destination_id else 0
                f.write(
                    f"{drain.id:<5} {drain.element:>5} {drain.elevation:>10.2f} "
                    f"{drain.conductance:>14.6f}  {drain.destination_type:<10} {dest_id:>5}\n"
                )

        return filepath

    def write_subsidence(self, gw: AppGW, header: str | None = None) -> Path:
        """
        Write subsidence parameters file.

        Args:
            gw: AppGW component
            header: Optional header comment

        Returns:
            Path to written file
        """
        filepath = self.config.get_subsidence_path()

        with open(filepath, "w", encoding="utf-8") as f:
            # Write header
            if header:
                for line in header.strip().split("\n"):
                    f.write(f"C  {line}\n")
            else:
                f.write("C  Subsidence parameters file\n")
                f.write("C  Generated by pyiwfm\n")
                f.write("C\n")
                f.write("C  ELEM  LAYER  ELASTIC_S  INELASTIC_S  PRECON_HEAD\n")

            # Write subsidence count
            f.write(f"{len(gw.subsidence):<10}                              / N_SUBSIDENCE\n")

            # Write subsidence data
            for sub in gw.subsidence:
                f.write(
                    f"{sub.element:>5} {sub.layer:>5} {sub.elastic_storage:>12.6e} "
                    f"{sub.inelastic_storage:>12.6e} {sub.preconsolidation_head:>12.4f}\n"
                )

        return filepath

    def write_initial_heads(self, gw: AppGW, header: str | None = None) -> Path:
        """
        Write initial heads file.

        Args:
            gw: AppGW component
            header: Optional header comment

        Returns:
            Path to written file
        """
        filepath = self.config.get_initial_heads_path()

        if gw.heads is None:
            raise ValueError("No initial heads to write")

        with open(filepath, "w", encoding="utf-8") as f:
            # Write header
            if header:
                for line in header.strip().split("\n"):
                    f.write(f"C  {line}\n")
            else:
                f.write("C  Initial heads file\n")
                f.write("C  Generated by pyiwfm\n")
                f.write("C\n")

            # Write dimensions
            f.write(f"{gw.n_nodes:<10}                              / NNODES\n")
            f.write(f"{gw.n_layers:<10}                              / NLAYERS\n")

            # Build header for layers
            layer_cols = "  ".join([f"HEAD_L{i + 1:02d}" for i in range(gw.n_layers)])
            f.write(f"C  NODE  {layer_cols}\n")

            # Write head data
            for node_idx in range(gw.n_nodes):
                node_id = node_idx + 1
                line = f"{node_id:<5}"

                for layer in range(gw.n_layers):
                    head = gw.heads[node_idx, layer]
                    line += f" {head:>12.4f}"

                f.write(line + "\n")

        return filepath


class GroundwaterReader:
    """
    Reader for IWFM groundwater component files.
    """

    def read_wells(self, filepath: Path | str) -> dict[int, Well]:
        """
        Read wells from a wells definition file.

        Args:
            filepath: Path to wells file

        Returns:
            Dictionary mapping well ID to Well object
        """
        filepath = Path(filepath)
        wells: dict[int, Well] = {}

        with open(filepath, encoding="utf-8") as f:
            line_num = 0
            n_wells = None

            # Find NWELLS
            for line in f:
                line_num += 1
                if _is_comment_line(line):
                    continue

                value, _ = _strip_comment(line)
                n_wells = parse_int(value, context="NWELLS", line_number=line_num)
                break

            if n_wells is None:
                raise FileFormatError("Required keyword 'NWELLS' not found anywhere in file")

            # Read well data
            for line in f:
                line_num += 1
                if _is_comment_line(line):
                    continue

                parts = line.split()
                if len(parts) < 7:
                    continue

                try:
                    well_id = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    element = int(parts[3])
                    top_screen = float(parts[4])
                    bottom_screen = float(parts[5])
                    max_pump_rate = float(parts[6])
                    name = " ".join(parts[7:]) if len(parts) > 7 else ""

                    wells[well_id] = Well(
                        id=well_id,
                        x=x,
                        y=y,
                        element=element,
                        top_screen=top_screen,
                        bottom_screen=bottom_screen,
                        max_pump_rate=max_pump_rate,
                        name=name,
                    )

                except ValueError as e:
                    raise FileFormatError(
                        f"Invalid well data: '{line.strip()}'", line_number=line_num
                    ) from e

        return wells

    def read_initial_heads(self, filepath: Path | str) -> tuple[int, int, NDArray[np.float64]]:
        """
        Read initial heads from file.

        Args:
            filepath: Path to initial heads file

        Returns:
            Tuple of (n_nodes, n_layers, heads array)
        """
        filepath = Path(filepath)

        with open(filepath, encoding="utf-8") as f:
            line_num = 0
            n_nodes = None
            n_layers = None

            # Find NNODES
            for line in f:
                line_num += 1
                if _is_comment_line(line):
                    continue

                value, _ = _strip_comment(line)
                n_nodes = parse_int(value, context="NNODES", line_number=line_num)
                break

            # Find NLAYERS
            for line in f:
                line_num += 1
                if _is_comment_line(line):
                    continue

                value, _ = _strip_comment(line)
                n_layers = parse_int(value, context="NLAYERS", line_number=line_num)
                break

            if n_nodes is None or n_layers is None:
                raise FileFormatError(
                    "Required keyword 'NNODES or NLAYERS' not found anywhere in file"
                )

            # Initialize heads array
            heads = np.zeros((n_nodes, n_layers))

            # Read head data
            for line in f:
                line_num += 1
                if _is_comment_line(line):
                    continue

                parts = line.split()
                if len(parts) < n_layers + 1:
                    continue

                try:
                    node_id = int(parts[0])
                    node_idx = node_id - 1

                    for layer in range(n_layers):
                        heads[node_idx, layer] = float(parts[layer + 1])

                except (ValueError, IndexError) as e:
                    raise FileFormatError(
                        f"Invalid head data: '{line.strip()}'", line_number=line_num
                    ) from e

        return n_nodes, n_layers, heads

    def read_subsidence(self, filepath: Path | str) -> list[Subsidence]:
        """
        Read subsidence parameters from file.

        Args:
            filepath: Path to subsidence parameters file

        Returns:
            List of Subsidence objects
        """
        filepath = Path(filepath)
        subsidence_list: list[Subsidence] = []

        with open(filepath, encoding="utf-8") as f:
            line_num = 0
            n_subsidence = None

            # Find N_SUBSIDENCE
            for line in f:
                line_num += 1
                if _is_comment_line(line):
                    continue

                value, _ = _strip_comment(line)
                try:
                    n_subsidence = int(value)
                except ValueError as e:
                    raise FileFormatError(
                        f"Invalid N_SUBSIDENCE value: '{value}'",
                        line_number=line_num,
                    ) from e
                break

            if n_subsidence is None:
                raise FileFormatError("Required keyword 'N_SUBSIDENCE' not found anywhere in file")

            # Read subsidence data
            for line in f:
                line_num += 1
                if _is_comment_line(line):
                    continue

                parts = line.split()
                if len(parts) < 5:
                    continue

                try:
                    element = int(parts[0])
                    layer = int(parts[1])
                    elastic_storage = float(parts[2])
                    inelastic_storage = float(parts[3])
                    preconsolidation_head = float(parts[4])

                    subsidence_list.append(
                        Subsidence(
                            element=element,
                            layer=layer,
                            elastic_storage=elastic_storage,
                            inelastic_storage=inelastic_storage,
                            preconsolidation_head=preconsolidation_head,
                        )
                    )

                except ValueError as e:
                    raise FileFormatError(
                        f"Invalid subsidence data: '{line.strip()}'",
                        line_number=line_num,
                    ) from e

        return subsidence_list


# =============================================================================
# Component Main File Reader (hierarchical dispatcher file)
# =============================================================================


def write_groundwater(
    gw: AppGW,
    output_dir: Path | str,
    config: GWFileConfig | None = None,
) -> dict[str, Path]:
    """
    Write groundwater component to files.

    Args:
        gw: AppGW component to write
        output_dir: Output directory
        config: Optional file configuration

    Returns:
        Dictionary mapping file type to output path
    """
    output_dir = Path(output_dir)

    if config is None:
        config = GWFileConfig(output_dir=output_dir)
    else:
        config.output_dir = output_dir

    writer = GroundwaterWriter(config)
    return writer.write(gw)


def read_wells(filepath: Path | str) -> dict[int, Well]:
    """
    Read wells from a wells definition file.

    Args:
        filepath: Path to wells file

    Returns:
        Dictionary mapping well ID to Well object
    """
    reader = GroundwaterReader()
    return reader.read_wells(filepath)


def read_initial_heads(
    filepath: Path | str,
) -> tuple[int, int, NDArray[np.float64]]:
    """
    Read initial heads from file.

    Args:
        filepath: Path to initial heads file

    Returns:
        Tuple of (n_nodes, n_layers, heads array)
    """
    reader = GroundwaterReader()
    return reader.read_initial_heads(filepath)


def read_subsidence(filepath: Path | str) -> list[Subsidence]:
    """
    Read subsidence parameters from file.

    Args:
        filepath: Path to subsidence parameters file

    Returns:
        List of Subsidence objects
    """
    reader = GroundwaterReader()
    return reader.read_subsidence(filepath)
