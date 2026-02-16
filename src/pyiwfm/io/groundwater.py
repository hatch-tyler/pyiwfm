"""
Groundwater component I/O handlers for IWFM model files.

This module provides functions for reading and writing IWFM groundwater
component files including wells, pumping, boundary conditions, aquifer
parameters, tile drains, and subsidence data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence, TextIO

import numpy as np
from numpy.typing import NDArray

from pyiwfm.components.groundwater import (
    AppGW,
    Well,
    BoundaryCondition,
    TileDrain,
    Subsidence,
    AquiferParameters,
    ElementPumping,
    HydrographLocation,
)
from pyiwfm.core.timeseries import TimeSeries, TimeSeriesCollection
from pyiwfm.core.exceptions import FileFormatError
from pyiwfm.io.iwfm_reader import (
    COMMENT_CHARS,
    is_comment_line as _is_comment_line,
    strip_inline_comment as _parse_value_line,
)
from pyiwfm.io.timeseries_ascii import TimeSeriesWriter, format_iwfm_timestamp


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
    """

    n_nodes: int
    n_elements: int
    elements: list[tuple[int, ...]]
    node_coords: NDArray[np.float64]
    node_values: NDArray[np.float64]


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

        with open(filepath, "w") as f:
            # Write header
            if header:
                for line in header.strip().split("\n"):
                    f.write(f"C  {line}\n")
            else:
                f.write("C  Wells definition file\n")
                f.write("C  Generated by pyiwfm\n")
                f.write("C\n")
                f.write("C  ID      X              Y       ELEM  TOP_SCR  BOT_SCR  MAX_RATE  NAME\n")

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
            column_ids=well_ids,
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

        with open(filepath, "w") as f:
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
            layer_cols = "  ".join([f"KH{i+1:02d}  KV{i+1:02d}  SS{i+1:02d}  SY{i+1:02d}" for i in range(params.n_layers)])
            f.write(f"C  NODE  {layer_cols}\n")

            # Write parameter data
            for node_idx in range(params.n_nodes):
                node_id = node_idx + 1
                line = f"{node_id:<5}"

                for layer in range(params.n_layers):
                    kh = params.kh[node_idx, layer] if params.kh is not None else 0.0
                    kv = params.kv[node_idx, layer] if params.kv is not None else 0.0
                    ss = params.specific_storage[node_idx, layer] if params.specific_storage is not None else 0.0
                    sy = params.specific_yield[node_idx, layer] if params.specific_yield is not None else 0.0

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

        with open(filepath, "w") as f:
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
            bc_groups: dict[str, list] = defaultdict(list)
            for bc in gw.boundary_conditions:
                bc_groups[bc.bc_type].append(bc)
            specified_head = bc_groups["specified_head"]
            specified_flow = bc_groups["specified_flow"]
            general_head = bc_groups["general_head"]

            # Write specified head BCs
            f.write(f"C  SPECIFIED HEAD BOUNDARY CONDITIONS\n")
            f.write(f"{len(specified_head):<10}                              / N_SPEC_HEAD_BC\n")

            for bc in specified_head:
                f.write(f"C  BC ID: {bc.id}, Layer: {bc.layer}, N_nodes: {len(bc.nodes)}\n")
                f.write(f"{bc.id:<6} {bc.layer:>3} {len(bc.nodes):>5}  / BC_ID, LAYER, NNODES\n")
                for i, node in enumerate(bc.nodes):
                    f.write(f"  {node:>6} {bc.values[i]:>14.4f}\n")

            # Write specified flow BCs
            f.write(f"C  SPECIFIED FLOW BOUNDARY CONDITIONS\n")
            f.write(f"{len(specified_flow):<10}                              / N_SPEC_FLOW_BC\n")

            for bc in specified_flow:
                f.write(f"C  BC ID: {bc.id}, Layer: {bc.layer}, N_nodes: {len(bc.nodes)}\n")
                f.write(f"{bc.id:<6} {bc.layer:>3} {len(bc.nodes):>5}  / BC_ID, LAYER, NNODES\n")
                for i, node in enumerate(bc.nodes):
                    f.write(f"  {node:>6} {bc.values[i]:>14.4f}\n")

            # Write general head BCs
            f.write(f"C  GENERAL HEAD BOUNDARY CONDITIONS\n")
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

        with open(filepath, "w") as f:
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

        with open(filepath, "w") as f:
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

        with open(filepath, "w") as f:
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
            layer_cols = "  ".join([f"HEAD_L{i+1:02d}" for i in range(gw.n_layers)])
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

        with open(filepath, "r") as f:
            line_num = 0
            n_wells = None

            # Find NWELLS
            for line in f:
                line_num += 1
                if _is_comment_line(line):
                    continue

                value, _ = _parse_value_line(line)
                try:
                    n_wells = int(value)
                except ValueError as e:
                    raise FileFormatError(
                        f"Invalid NWELLS value: '{value}'", line_number=line_num
                    ) from e
                break

            if n_wells is None:
                raise FileFormatError("Could not find NWELLS in file")

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

    def read_initial_heads(
        self, filepath: Path | str
    ) -> tuple[int, int, NDArray[np.float64]]:
        """
        Read initial heads from file.

        Args:
            filepath: Path to initial heads file

        Returns:
            Tuple of (n_nodes, n_layers, heads array)
        """
        filepath = Path(filepath)

        with open(filepath, "r") as f:
            line_num = 0
            n_nodes = None
            n_layers = None

            # Find NNODES
            for line in f:
                line_num += 1
                if _is_comment_line(line):
                    continue

                value, _ = _parse_value_line(line)
                try:
                    n_nodes = int(value)
                except ValueError as e:
                    raise FileFormatError(
                        f"Invalid NNODES value: '{value}'", line_number=line_num
                    ) from e
                break

            # Find NLAYERS
            for line in f:
                line_num += 1
                if _is_comment_line(line):
                    continue

                value, _ = _parse_value_line(line)
                try:
                    n_layers = int(value)
                except ValueError as e:
                    raise FileFormatError(
                        f"Invalid NLAYERS value: '{value}'", line_number=line_num
                    ) from e
                break

            if n_nodes is None or n_layers is None:
                raise FileFormatError("Could not find NNODES or NLAYERS in file")

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

        with open(filepath, "r") as f:
            line_num = 0
            n_subsidence = None

            # Find N_SUBSIDENCE
            for line in f:
                line_num += 1
                if _is_comment_line(line):
                    continue

                value, _ = _parse_value_line(line)
                try:
                    n_subsidence = int(value)
                except ValueError as e:
                    raise FileFormatError(
                        f"Invalid N_SUBSIDENCE value: '{value}'",
                        line_number=line_num,
                    ) from e
                break

            if n_subsidence is None:
                raise FileFormatError("Could not find N_SUBSIDENCE in file")

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


@dataclass
class GWMainFileConfig:
    """
    Configuration parsed from GW component main file.

    The groundwater component main file is a dispatcher that references
    sub-files for boundary conditions, tile drains, pumping, and subsidence.
    It also contains inline hydrograph output location data.

    Attributes:
        version: File format version (e.g., "4.0")
        bc_file: Path to boundary conditions file
        tile_drain_file: Path to tile drains file
        pumping_file: Path to pumping file
        subsidence_file: Path to subsidence file
        overwrite_file: Path to parameter overwrite file (optional)
        head_output_factor: Conversion factor for head output
        head_output_unit: Unit string for head output
        volume_output_factor: Conversion factor for volume output
        volume_output_unit: Unit string for volume output
        debug_flag: Debug output flag
        coord_factor: Coordinate conversion factor for hydrographs
        hydrograph_output_file: Path to hydrograph output file
        hydrograph_locations: List of GW observation point locations
    """

    version: str = ""

    # Sub-file paths (all resolved to absolute)
    bc_file: Path | None = None
    tile_drain_file: Path | None = None
    pumping_file: Path | None = None
    subsidence_file: Path | None = None
    overwrite_file: Path | None = None

    # Conversion factors
    head_output_factor: float = 1.0
    head_output_unit: str = "FEET"
    volume_output_factor: float = 1.0
    volume_output_unit: str = "TAF"
    velocity_output_factor: float = 1.0
    velocity_output_unit: str = ""

    # Output files
    velocity_output_file: Path | None = None
    vertical_flow_output_file: Path | None = None
    head_all_output_file: Path | None = None
    head_tecplot_file: Path | None = None
    velocity_tecplot_file: Path | None = None
    budget_output_file: Path | None = None
    zbudget_output_file: Path | None = None
    final_heads_file: Path | None = None

    # Debug and hydrograph output
    debug_flag: int = 0
    coord_factor: float = 1.0
    hydrograph_output_file: Path | None = None
    hydrograph_locations: list[HydrographLocation] = field(default_factory=list)

    # Element face flow output
    n_face_flow_outputs: int = 0
    face_flow_output_file: Path | None = None
    face_flow_specs: list[FaceFlowSpec] = field(default_factory=list)

    # Aquifer parameters
    aquifer_params: AquiferParameters | None = None

    # Kh anomaly overwrites (parsed but not yet applied to node arrays)
    kh_anomalies: list[KhAnomalyEntry] = field(default_factory=list)

    # Parametric grid data (NGROUP > 0); interpolated later in model.py
    parametric_grids: list[ParametricGridData] = field(default_factory=list)

    # Initial heads
    initial_heads: NDArray[np.float64] | None = field(default=None, repr=False)


class GWMainFileReader:
    """
    Reader for IWFM groundwater component main file.

    The GW main file is a hierarchical dispatcher that contains:
    1. Version header (e.g., #4.0)
    2. Paths to sub-files (BC, tile drains, pumping, subsidence)
    3. Output conversion factors and units
    4. Inline hydrograph location data

    This reader parses the main file to extract configuration and
    hydrograph locations. It does NOT parse the sub-files - use
    the dedicated readers (e.g., GroundwaterReader.read_wells) for those.
    """

    def __init__(self) -> None:
        self._line_num = 0

    def read(
        self, filepath: Path | str, base_dir: Path | None = None
    ) -> GWMainFileConfig:
        """
        Parse GW main file, extracting config and hydrograph locations.

        Args:
            filepath: Path to the GW component main file
            base_dir: Base directory for resolving relative paths.
                     If None, uses the parent directory of filepath.

        Returns:
            GWMainFileConfig with parsed values
        """
        filepath = Path(filepath)
        if base_dir is None:
            base_dir = filepath.parent

        config = GWMainFileConfig()
        self._line_num = 0

        with open(filepath, "r") as f:
            # Read version header (first non-comment line starting with #)
            config.version = self._read_version(f)

            # Read file paths sequentially:
            # BCFL (boundary conditions file)
            bc_path = self._next_data_or_empty(f)
            if bc_path:
                config.bc_file = self._resolve_path(base_dir, bc_path)

            # TDFL (tile drains file)
            td_path = self._next_data_or_empty(f)
            if td_path:
                config.tile_drain_file = self._resolve_path(base_dir, td_path)

            # PUMPFL (pumping file)
            pump_path = self._next_data_or_empty(f)
            if pump_path:
                config.pumping_file = self._resolve_path(base_dir, pump_path)

            # SUBSFL (subsidence file)
            subs_path = self._next_data_or_empty(f)
            if subs_path:
                config.subsidence_file = self._resolve_path(base_dir, subs_path)

            # OVRWRTFL (optional overwrite file, may be empty)
            ovr_path = self._next_data_or_empty(f)
            if ovr_path:
                config.overwrite_file = self._resolve_path(base_dir, ovr_path)

            # FACTLTOU (head output conversion factor)
            factltou = self._next_data_or_empty(f)
            if factltou:
                try:
                    config.head_output_factor = float(factltou)
                except ValueError:
                    pass

            # UNITLTOU (head output unit)
            unitltou = self._next_data_or_empty(f)
            if unitltou:
                config.head_output_unit = unitltou

            # FACTVLOU (volume output conversion factor)
            factvlou = self._next_data_or_empty(f)
            if factvlou:
                try:
                    config.volume_output_factor = float(factvlou)
                except ValueError:
                    pass

            # UNITVLOU (volume output unit)
            unitvlou = self._next_data_or_empty(f)
            if unitvlou:
                config.volume_output_unit = unitvlou

            # FACTVROU (velocity output factor)
            factvrou = self._next_data_or_empty(f)
            if factvrou:
                try:
                    config.velocity_output_factor = float(factvrou)
                except ValueError:
                    pass

            # UNITVROU (velocity output unit)
            unitvrou = self._next_data_or_empty(f)
            if unitvrou:
                config.velocity_output_unit = unitvrou

            # VELOUTFL (velocity output file - optional)
            vel_path = self._next_data_or_empty(f)
            if vel_path:
                config.velocity_output_file = self._resolve_path(base_dir, vel_path)

            # VFLOWOUTFL (vertical flow output file - optional)
            vflow_path = self._next_data_or_empty(f)
            if vflow_path:
                config.vertical_flow_output_file = self._resolve_path(
                    base_dir, vflow_path
                )

            # GWALLOUTFL (GW head all output file - optional)
            headall_path = self._next_data_or_empty(f)
            if headall_path:
                config.head_all_output_file = self._resolve_path(
                    base_dir, headall_path
                )

            # HTPOUTFL (TecPlot head output file - optional)
            htec_path = self._next_data_or_empty(f)
            if htec_path:
                config.head_tecplot_file = self._resolve_path(base_dir, htec_path)

            # VTPOUTFL (TecPlot velocity output file - optional)
            vtec_path = self._next_data_or_empty(f)
            if vtec_path:
                config.velocity_tecplot_file = self._resolve_path(
                    base_dir, vtec_path
                )

            # GWBUDFL (GW budget output file - optional)
            bud_path = self._next_data_or_empty(f)
            if bud_path:
                config.budget_output_file = self._resolve_path(base_dir, bud_path)

            # ZBUDFL (Zone budget output file - optional)
            zbud_path = self._next_data_or_empty(f)
            if zbud_path:
                config.zbudget_output_file = self._resolve_path(base_dir, zbud_path)

            # FNGWFL (final condition output file - optional)
            final_path = self._next_data_or_empty(f)
            if final_path:
                config.final_heads_file = self._resolve_path(base_dir, final_path)

            # KDEB (debug flag)
            kdeb = self._next_data_or_empty(f)
            if kdeb:
                try:
                    config.debug_flag = int(kdeb)
                except ValueError:
                    pass

            # NOUTH (number of hydrograph output locations)
            nouth_str = self._next_data_or_empty(f)
            n_hydrographs = 0
            if nouth_str:
                try:
                    n_hydrographs = int(nouth_str)
                except ValueError:
                    import logging
                    logging.getLogger(__name__).warning(
                        "Could not parse NOUTH value '%s' at line %d, assuming 0",
                        nouth_str, self._line_num,
                    )

            # FACTXY (coordinate conversion factor — read regardless of NOUTH)
            factxy = self._next_data_or_empty(f)
            if factxy:
                try:
                    config.coord_factor = float(factxy)
                except ValueError:
                    pass

            # GWHYDOUTFL (hydrograph output file — read regardless of NOUTH)
            hydout_path = self._next_data_or_empty(f)
            if hydout_path:
                config.hydrograph_output_file = self._resolve_path(
                    base_dir, hydout_path
                )

            # Read inline hydrograph location data (only if NOUTH > 0)
            if n_hydrographs > 0:
                config.hydrograph_locations = self._read_hydrograph_data(
                    f, n_hydrographs, config.coord_factor
                )

            # ── Element Face Flow Output ─────────────────────────────
            # NOUTF (number of element face flow hydrographs)
            noutf_str = self._next_data_or_empty(f)
            if noutf_str:
                try:
                    config.n_face_flow_outputs = int(noutf_str)
                except ValueError:
                    pass

            # FCHYDOUTFL (face flow output file - optional)
            fc_path = self._next_data_or_empty(f)
            if fc_path:
                config.face_flow_output_file = self._resolve_path(
                    base_dir, fc_path
                )

            # Read inline face flow specifications (NOUTF rows)
            if config.n_face_flow_outputs > 0:
                config.face_flow_specs = self._read_face_flow_specs(
                    f, config.n_face_flow_outputs
                )

            # ── Aquifer Parameters ───────────────────────────────────
            try:
                config.aquifer_params = self._read_aquifer_parameters(
                    f, base_dir, config
                )
            except Exception as exc:
                import logging
                logging.getLogger(__name__).warning(
                    "Failed to read aquifer parameters at line %d: %s",
                    self._line_num, exc,
                )

            # ── Anomaly in Hydraulic Conductivity ────────────────────
            try:
                config.kh_anomalies = self._read_kh_anomaly(f)
            except Exception as exc:
                import logging
                logging.getLogger(__name__).warning(
                    "Failed to read Kh anomalies at line %d: %s",
                    self._line_num, exc,
                )

            # ── Initial Heads ────────────────────────────────────────
            try:
                config.initial_heads = self._read_initial_heads(f)
            except Exception as exc:
                import logging
                logging.getLogger(__name__).warning(
                    "Failed to read initial heads at line %d: %s",
                    self._line_num, exc,
                )

        return config

    def _read_version(self, f: TextIO) -> str:
        """Read the version header from the file."""
        for line in f:
            self._line_num += 1
            stripped = line.strip()
            if not stripped:
                continue
            # Version line starts with # followed by version number
            if stripped.startswith("#"):
                return stripped[1:].strip()
            # If we hit a comment line, continue
            if line[0] in COMMENT_CHARS:
                continue
            # If we hit data before version, there's no version header
            break
        return ""

    def _next_data_or_empty(self, f: TextIO) -> str:
        """
        Return next data value, or empty string for blank/comment-only lines.

        This handles the IWFM file format where optional file paths may be
        represented by blank lines.
        """
        for line in f:
            self._line_num += 1
            # Check for comment line (comment char in column 1)
            if line and line[0] in COMMENT_CHARS:
                continue
            # Parse the value
            value, _ = _parse_value_line(line)
            return value
        return ""

    def _read_hydrograph_data(
        self, f: TextIO, n_hydrographs: int, coord_factor: float
    ) -> list[HydrographLocation]:
        """
        Read inline hydrograph location data.

        Format depends on HYDTYP:
        - HYDTYP=0 (x-y coords): ID  HYDTYP  IOUTHL  X  Y  NAME
        - HYDTYP=1 (node number): ID  HYDTYP  IOUTHL  IOUTH  NAME

        Args:
            f: Open file handle
            n_hydrographs: Number of hydrograph locations to read
            coord_factor: Coordinate conversion factor

        Returns:
            List of HydrographLocation objects
        """
        locations: list[HydrographLocation] = []
        count = 0

        for line in f:
            self._line_num += 1
            if _is_comment_line(line):
                continue

            parts = line.split()
            if len(parts) < 4:
                continue

            try:
                # ID, HYDTYP, IOUTHL are always present
                hyd_id = int(parts[0])
                hydtyp = int(parts[1])
                layer = int(parts[2])

                if hydtyp == 0:
                    # x-y coordinates provided: ID HYDTYP IOUTHL X Y NAME
                    if len(parts) < 5:
                        continue
                    x = float(parts[3]) * coord_factor
                    y = float(parts[4]) * coord_factor
                    node_id = 0  # Will need to find nearest node
                    # Name is everything after the Y coordinate
                    name = " ".join(parts[5:]) if len(parts) > 5 else ""
                else:
                    # Node number provided: ID HYDTYP IOUTHL IOUTH NAME
                    x = 0.0
                    y = 0.0
                    node_id = int(parts[3])
                    # Name is everything after the node number
                    name = " ".join(parts[4:]) if len(parts) > 4 else ""

                locations.append(
                    HydrographLocation(
                        node_id=node_id, layer=layer, x=x, y=y, name=name
                    )
                )

                count += 1
                if count >= n_hydrographs:
                    break

            except (ValueError, IndexError):
                # Skip malformed lines
                continue

        return locations

    def _resolve_path(self, base_dir: Path, filepath: str) -> Path:
        """Resolve a file path relative to base directory."""
        path = Path(filepath.strip())
        if path.is_absolute():
            return path
        return base_dir / path

    def _skip_data_lines(self, f: TextIO, count: int) -> None:
        """Skip *count* non-comment data lines."""
        skipped = 0
        for line in f:
            self._line_num += 1
            if _is_comment_line(line):
                continue
            skipped += 1
            if skipped >= count:
                break

    def _read_face_flow_specs(
        self, f: TextIO, count: int
    ) -> list[FaceFlowSpec]:
        """Read inline element face flow specifications.

        Format per line: ID  IOUTFL  IOUTFA  IOUTFB  NAME
        """
        specs: list[FaceFlowSpec] = []
        read_count = 0
        for line in f:
            self._line_num += 1
            if _is_comment_line(line):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                spec = FaceFlowSpec(
                    id=int(parts[0]),
                    layer=int(parts[1]),
                    node_a=int(parts[2]),
                    node_b=int(parts[3]),
                    name=" ".join(parts[4:]) if len(parts) > 4 else "",
                )
                specs.append(spec)
            except (ValueError, IndexError):
                continue
            read_count += 1
            if read_count >= count:
                break
        return specs

    def _read_aquifer_parameters(
        self, f: TextIO, base_dir: Path, config: GWMainFileConfig | None = None
    ) -> AquiferParameters | None:
        """Read the inline Aquifer Parameters section.

        The section layout in the GW main file is:

        1. ``NGROUP`` — number of parametric grid groups.
           ``0`` means Option 2 (per-node parameters).
        2. Conversion factors (one data line):
           ``FX  FKH  FS  FN  FV  FL``
        3. Time units (three data lines):
           ``TUNITKH``, ``TUNITV``, ``TUNITL``
        4. If ``NGROUP > 0``: parametric grid definitions
           (stored in ``config.parametric_grids``).
        5. If ``NGROUP == 0``: per-node parameter data, one node per
           block of ``n_layers`` lines.  First line of each block has
           the node ID; continuation lines have parameters only.

        Per-node columns: ``PKH  PS  PN  PV  PL``
        (horizontal K, specific storage, specific yield,
        aquitard vertical K, aquifer vertical K)

        Returns ``None`` if the section cannot be parsed or if
        parametric grid mode is used (data stored on *config* instead).
        """
        # NGROUP
        ngroup_str = self._next_data_or_empty(f)
        if not ngroup_str:
            return None
        try:
            ngroup = int(ngroup_str)
        except ValueError:
            return None

        # Conversion factors: FX  FKH  FS  FN  FV  FL
        factors_str = self._next_data_or_empty(f)
        if not factors_str:
            return None
        fparts = factors_str.split()
        if len(fparts) < 6:
            return None
        try:
            _fx = float(fparts[0])
            fkh = float(fparts[1])
            fs = float(fparts[2])
            fn = float(fparts[3])
            fv = float(fparts[4])
            fl = float(fparts[5])
        except ValueError:
            return None

        # Time units: TUNITKH, TUNITV, TUNITL (read and discard)
        self._next_data_or_empty(f)  # TUNITKH
        self._next_data_or_empty(f)  # TUNITV
        self._next_data_or_empty(f)  # TUNITL

        if ngroup > 0:
            factors = (_fx, fkh, fs, fn, fv, fl)
            grids = self._read_parametric_aquifer_params(f, ngroup, factors)
            if config is not None:
                config.parametric_grids = grids
            return None

        # Option 2: per-node parameters
        # Read all node data blocks.  The first data line of each node
        # starts with the node ID followed by 5 parameter values.
        # Continuation lines (for layers 2..n_layers) have 5 values only.
        #
        # We don't know n_nodes or n_layers up front, so we collect
        # data dynamically and infer them.

        node_ids: list[int] = []
        # Per-node: list of (kh, ss, sy, aquitard_kv, kv) tuples per layer
        node_layers: list[list[tuple[float, float, float, float, float]]] = []

        current_node_data: list[tuple[float, float, float, float, float]] = []
        current_node_id: int | None = None

        for line in f:
            self._line_num += 1
            if _is_comment_line(line):
                if current_node_id is not None:
                    # A comment line after data started means end of
                    # aquifer params section.  Save last node and stop.
                    node_ids.append(current_node_id)
                    node_layers.append(current_node_data)
                    break
                # Haven't started reading data yet — skip comment.
                continue

            value, _ = _parse_value_line(line)
            parts = value.split()
            if not parts:
                continue

            # Determine if this is a new node line or continuation.
            # A new node line has 6 fields: ID PKH PS PN PV PL
            # A continuation line has 5 fields: PKH PS PN PV PL
            # We distinguish by checking if the first value is an
            # integer that could be a node ID (and field count).
            if len(parts) == 6:
                # New node line
                try:
                    node_id = int(parts[0])
                    pkh = float(parts[1]) * fkh
                    ps = float(parts[2]) * fs
                    pn = float(parts[3]) * fn
                    pv = float(parts[4]) * fv   # aquitard vertical K
                    pl = float(parts[5]) * fl     # aquifer vertical K
                except ValueError:
                    break

                # Save previous node if any
                if current_node_id is not None:
                    node_ids.append(current_node_id)
                    node_layers.append(current_node_data)

                current_node_id = node_id
                current_node_data = [(pkh, ps, pn, pv, pl)]

            elif len(parts) == 5:
                # Continuation line for current node
                try:
                    pkh = float(parts[0]) * fkh
                    ps = float(parts[1]) * fs
                    pn = float(parts[2]) * fn
                    pv = float(parts[3]) * fv
                    pl = float(parts[4]) * fl
                except ValueError:
                    break
                current_node_data.append((pkh, ps, pn, pv, pl))

            else:
                # Unexpected format — end of section
                if current_node_id is not None:
                    node_ids.append(current_node_id)
                    node_layers.append(current_node_data)
                break

        if not node_ids:
            return None

        n_nodes = len(node_ids)
        n_layers = len(node_layers[0])

        # Build arrays (n_nodes, n_layers)
        kh = np.zeros((n_nodes, n_layers), dtype=np.float64)
        ss = np.zeros((n_nodes, n_layers), dtype=np.float64)
        sy = np.zeros((n_nodes, n_layers), dtype=np.float64)
        aquitard_kv = np.zeros((n_nodes, n_layers), dtype=np.float64)
        kv = np.zeros((n_nodes, n_layers), dtype=np.float64)

        for i, layers in enumerate(node_layers):
            for j, (h, s, y, av, v) in enumerate(layers):
                if j < n_layers:
                    kh[i, j] = h
                    ss[i, j] = s
                    sy[i, j] = y
                    aquitard_kv[i, j] = av
                    kv[i, j] = v

        return AquiferParameters(
            n_nodes=n_nodes,
            n_layers=n_layers,
            kh=kh,
            kv=kv,
            specific_storage=ss,
            specific_yield=sy,
            aquitard_kv=aquitard_kv,
        )

    def _read_parametric_aquifer_params(
        self,
        f: TextIO,
        ngroup: int,
        factors: tuple[float, float, float, float, float, float],
    ) -> list[ParametricGridData]:
        """Read NGROUP parametric grid definitions.

        Each parametric grid group contains:
        1. ``NDP NEP`` — number of parametric nodes and elements
        2. ``NEP`` element definition lines: ``ElemID  N1  N2  N3  N4``
        3. ``NDP`` node data lines:
           ``NodeID  X  Y  Param1_L1 ... Param5_LN``

        The 5 parameters per layer are: Kh, Ss, Sy, AquitardKv, Kv.

        Parameters
        ----------
        f : TextIO
            Open file handle positioned after the time-unit lines.
        ngroup : int
            Number of parametric grid groups to read.
        factors : tuple
            ``(FX, FKH, FS, FN, FV, FL)`` conversion factors.

        Returns
        -------
        list[ParametricGridData]
            One entry per parametric grid group.
        """
        fx, fkh, fs, fn, fv, fl = factors
        grids: list[ParametricGridData] = []

        for _ in range(ngroup):
            # NDP  NEP
            ndp_nep_str = self._next_data_or_empty(f)
            if not ndp_nep_str:
                break
            parts = ndp_nep_str.split()
            if len(parts) < 2:
                break
            try:
                ndp = int(parts[0])
                nep = int(parts[1])
            except ValueError:
                break

            # Read NEP element definitions
            elements: list[tuple[int, ...]] = []
            elem_count = 0
            for line in f:
                self._line_num += 1
                if _is_comment_line(line):
                    continue
                value, _ = _parse_value_line(line)
                eparts = value.split()
                if len(eparts) < 4:
                    break
                try:
                    # ElemID, Node1, Node2, Node3, Node4
                    # Node indices are 1-based in the file; convert to
                    # 0-based indices into the node array.
                    verts = [int(p) - 1 for p in eparts[1:]]
                    # Remove trailing -1 entries (Node4=0 means triangle)
                    verts = [v for v in verts if v >= 0]
                    elements.append(tuple(verts))
                except ValueError:
                    break
                elem_count += 1
                if elem_count >= nep:
                    break

            # Read NDP parametric node data lines
            # Each line: NodeID  X  Y  P1_L1 P1_L2 ... P5_LN
            # The parameter values are ordered parameter-major:
            # all layers of Kh, then all layers of Ss, etc.
            node_coords = np.zeros((ndp, 2), dtype=np.float64)
            # We don't know n_layers yet; infer from first line
            all_raw_values: list[list[float]] = []
            node_count = 0
            for line in f:
                self._line_num += 1
                if _is_comment_line(line):
                    continue
                value, _ = _parse_value_line(line)
                nparts = value.split()
                if len(nparts) < 4:
                    break
                try:
                    # NodeID, X, Y, values...
                    node_coords[node_count, 0] = float(nparts[1]) * fx
                    node_coords[node_count, 1] = float(nparts[2]) * fx
                    raw_vals = [float(v) for v in nparts[3:]]
                    all_raw_values.append(raw_vals)
                except (ValueError, IndexError):
                    break
                node_count += 1
                if node_count >= ndp:
                    break

            if not all_raw_values:
                continue

            # Determine n_layers from value count:
            # 5 params * n_layers values per node
            n_values = len(all_raw_values[0])
            n_layers = n_values // 5 if n_values >= 5 else 1

            # Build node_values array: shape (ndp, n_layers, 5)
            # Fortran reshapes with ORDER=[2,1], meaning the data in
            # the file is parameter-major: all layers of param 1, then
            # all layers of param 2, etc.
            node_values = np.zeros((ndp, n_layers, 5), dtype=np.float64)
            param_factors = [fkh, fs, fn, fv, fl]
            for i, raw in enumerate(all_raw_values):
                for p in range(5):
                    for lay in range(n_layers):
                        idx = p * n_layers + lay
                        if idx < len(raw):
                            node_values[i, lay, p] = raw[idx] * param_factors[p]

            grids.append(ParametricGridData(
                n_nodes=ndp,
                n_elements=nep,
                elements=elements,
                node_coords=node_coords[:node_count],
                node_values=node_values[:node_count],
            ))

        return grids

    def _read_kh_anomaly(self, f: TextIO) -> list[KhAnomalyEntry]:
        """Read the Anomaly in Hydraulic Conductivity section.

        Format::

            NEBK        (number of elements to overwrite, 0 = none)
            FACT        (conversion factor for anomaly K)
            TUNITH      (time unit string, e.g. 1DAY)
            IC  IEBK  BK[1]  BK[2] ... BK[n_layers]

        Returns a list of :class:`KhAnomalyEntry` objects with
        Kh values already multiplied by FACT.  The actual overwrite
        onto node arrays is performed later in ``model.py`` once the
        mesh element-to-node connectivity is available.
        """
        # NEBK
        nebk_str = self._next_data_or_empty(f)
        if not nebk_str:
            return []
        try:
            nebk = int(nebk_str)
        except ValueError:
            return []

        if nebk <= 0:
            return []

        # FACT (conversion factor)
        fact_str = self._next_data_or_empty(f)
        try:
            fact = float(fact_str)
        except ValueError:
            fact = 1.0

        # TUNITH (time unit — read and store but no conversion applied)
        self._next_data_or_empty(f)

        # Read NEBK anomaly data lines
        entries: list[KhAnomalyEntry] = []
        count = 0
        for line in f:
            self._line_num += 1
            if _is_comment_line(line):
                continue
            value, _ = _parse_value_line(line)
            parts = value.split()
            if len(parts) < 3:
                break
            try:
                # parts: IC, IEBK, BK[1], ..., BK[NLayers]
                element_id = int(parts[1])
                bk = [float(v) * fact for v in parts[2:]]
                entries.append(KhAnomalyEntry(element_id=element_id, kh_per_layer=bk))
            except (ValueError, IndexError):
                break
            count += 1
            if count >= nebk:
                break

        return entries

    def _read_initial_heads(self, f: TextIO) -> NDArray[np.float64] | None:
        """Read the Initial Groundwater Head Values section.

        Format::

            FACTHP          (conversion factor)
            ID  HP[1] HP[2] ... HP[n_layers]

        Returns an (n_nodes, n_layers) array, or None.
        """
        # FACTHP
        facthp_str = self._next_data_or_empty(f)
        if not facthp_str:
            return None
        try:
            facthp = float(facthp_str)
        except ValueError:
            return None

        rows: list[list[float]] = []
        for line in f:
            self._line_num += 1
            if _is_comment_line(line):
                continue
            value, _ = _parse_value_line(line)
            parts = value.split()
            if len(parts) < 2:
                break
            try:
                _node_id = int(parts[0])
                heads = [float(v) * facthp for v in parts[1:]]
                rows.append(heads)
            except ValueError:
                break

        if not rows:
            return None
        return np.array(rows, dtype=np.float64)


# Convenience functions


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


def read_gw_main_file(
    filepath: Path | str, base_dir: Path | None = None
) -> GWMainFileConfig:
    """
    Read IWFM groundwater component main file.

    The GW main file is a hierarchical dispatcher that contains paths to
    sub-files (boundary conditions, pumping, etc.) and inline hydrograph
    location data.

    Args:
        filepath: Path to the GW component main file
        base_dir: Base directory for resolving relative paths.
                 If None, uses the parent directory of filepath.

    Returns:
        GWMainFileConfig with parsed values including hydrograph locations

    Example:
        >>> config = read_gw_main_file("C2VSimFG_Groundwater.dat")
        >>> print(f"Version: {config.version}")
        >>> print(f"Hydrograph locations: {len(config.hydrograph_locations)}")
        >>> if config.pumping_file:
        ...     wells = read_wells(config.pumping_file)
    """
    reader = GWMainFileReader()
    return reader.read(filepath, base_dir)
