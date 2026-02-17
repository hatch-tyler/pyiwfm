"""
Small Watershed Main File Reader for IWFM.

This module reads the IWFM small watershed component main file, which
defines watershed-level modeling parameters including:
1. Output file paths (budget, final results)
2. Number of small watersheds and their geospatial data
3. Root zone parameters (soil properties, curve numbers)
4. Aquifer parameters (storage, recession coefficients)
5. Initial conditions

Reference: Class_AppSmallWatershed_v40.f90 - Class_AppSmallWatershed_v40_New()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

from pyiwfm.core.exceptions import FileFormatError
from pyiwfm.io.iwfm_reader import (
    COMMENT_CHARS,
)
from pyiwfm.io.iwfm_reader import (
    is_comment_line as _is_comment_line,
)
from pyiwfm.io.iwfm_reader import (
    next_data_or_empty as _next_data_or_empty,
)
from pyiwfm.io.iwfm_reader import (
    resolve_path as _resolve_path_f,
)


@dataclass
class WatershedGWNode:
    """Groundwater node connection for a small watershed.

    Attributes:
        gw_node_id: Groundwater node ID
        max_perc_rate: Max percolation rate (positive) or baseflow layer (negative)
        is_baseflow: Whether this is a baseflow node
        layer: Baseflow layer (if is_baseflow)
    """

    gw_node_id: int = 0
    max_perc_rate: float = 0.0
    is_baseflow: bool = False
    layer: int = 0


@dataclass
class WatershedSpec:
    """Geospatial specification for a single small watershed.

    Attributes:
        id: Watershed ID
        area: Watershed area (converted by area factor)
        dest_stream_node: Destination stream node for outflow
        gw_nodes: List of connected groundwater nodes
    """

    id: int = 0
    area: float = 0.0
    dest_stream_node: int = 0
    gw_nodes: list[WatershedGWNode] = field(default_factory=list)


@dataclass
class WatershedRootZoneParams:
    """Root zone parameters for a single small watershed.

    Attributes:
        id: Watershed ID
        precip_col: Precipitation time-series column index
        precip_factor: Precipitation conversion factor
        et_col: ET time-series column index
        wilting_point: Soil wilting point
        field_capacity: Soil field capacity
        total_porosity: Total porosity
        lambda_param: Pore size distribution parameter
        root_depth: Root zone depth
        hydraulic_cond: Hydraulic conductivity
        kunsat_method: Unsaturated K method code
        curve_number: SCS curve number
    """

    id: int = 0
    precip_col: int = 0
    precip_factor: float = 1.0
    et_col: int = 0
    crop_coeff_col: int = 0
    wilting_point: float = 0.0
    field_capacity: float = 0.0
    total_porosity: float = 0.0
    lambda_param: float = 0.0
    root_depth: float = 0.0
    hydraulic_cond: float = 0.0
    kunsat_method: int = 0
    curve_number: float = 0.0


@dataclass
class WatershedInitialCondition:
    """Initial conditions for a single small watershed.

    Attributes:
        id: Watershed ID
        soil_moisture: Initial soil moisture fraction (0-1)
        gw_storage: Initial groundwater storage (multiplied by ic_factor)
    """

    id: int = 0
    soil_moisture: float = 0.0
    gw_storage: float = 0.0


@dataclass
class WatershedAquiferParams:
    """Aquifer parameters for a single small watershed.

    Attributes:
        id: Watershed ID
        gw_threshold: Groundwater storage threshold
        max_gw_storage: Maximum groundwater storage
        surface_flow_coeff: Surface flow recession coefficient
        baseflow_coeff: Baseflow recession coefficient
    """

    id: int = 0
    gw_threshold: float = 0.0
    max_gw_storage: float = 0.0
    surface_flow_coeff: float = 0.0
    baseflow_coeff: float = 0.0


@dataclass
class SmallWatershedMainConfig:
    """Configuration parsed from Small Watershed component main file.

    Attributes:
        version: File format version
        budget_output_file: Path to budget output file
        final_results_file: Path to final simulation results file
        n_watersheds: Number of small watersheds
        area_factor: Area conversion factor
        flow_factor: Flow rate conversion factor
        flow_time_unit: Time unit for flow rates
        watershed_specs: Geospatial specs per watershed
        rz_solver_tolerance: Root zone solver tolerance
        rz_max_iterations: Root zone solver max iterations
        rz_length_factor: Root zone length conversion factor
        rz_cn_factor: Curve number conversion factor
        rz_k_factor: Hydraulic conductivity conversion factor
        rz_k_time_unit: Time unit for hydraulic conductivity
        rootzone_params: Root zone parameters per watershed
        aq_gw_factor: GW conversion factor
        aq_time_factor: Time conversion factor
        aq_time_unit: Time unit for recession coefficients
        aquifer_params: Aquifer parameters per watershed
    """

    version: str = ""
    budget_output_file: Path | None = None
    final_results_file: Path | None = None
    n_watersheds: int = 0

    # Geospatial factors and data
    area_factor: float = 1.0
    flow_factor: float = 1.0
    flow_time_unit: str = ""
    watershed_specs: list[WatershedSpec] = field(default_factory=list)

    # Root zone factors and data
    rz_solver_tolerance: float = 1e-8
    rz_max_iterations: int = 2000
    rz_length_factor: float = 1.0
    rz_cn_factor: float = 1.0
    rz_k_factor: float = 1.0
    rz_k_time_unit: str = ""
    rootzone_params: list[WatershedRootZoneParams] = field(default_factory=list)

    # Aquifer factors and data
    aq_gw_factor: float = 1.0
    aq_time_factor: float = 1.0
    aq_time_unit: str = ""
    aquifer_params: list[WatershedAquiferParams] = field(default_factory=list)

    # Initial conditions
    ic_factor: float = 1.0
    initial_conditions: list[WatershedInitialCondition] = field(default_factory=list)


class SmallWatershedMainReader:
    """Reader for IWFM small watershed component main file.

    Parses the complete small watershed configuration including
    geospatial data, root zone parameters, aquifer parameters,
    and output file paths.
    """

    def __init__(self) -> None:
        self._line_num = 0

    def read(self, filepath: Path | str, base_dir: Path | None = None) -> SmallWatershedMainConfig:
        """Read small watershed main file.

        Args:
            filepath: Path to the small watershed main file
            base_dir: Base directory for resolving relative paths

        Returns:
            SmallWatershedMainConfig with all configuration data
        """
        filepath = Path(filepath)
        if base_dir is None:
            base_dir = filepath.parent

        config = SmallWatershedMainConfig()
        self._line_num = 0

        with open(filepath) as f:
            # Version header
            config.version = self._read_version(f)

            # Budget output file
            budget_path = _next_data_or_empty(f)
            if budget_path:
                config.budget_output_file = _resolve_path_f(base_dir, budget_path)

            # Final results output file
            final_path = _next_data_or_empty(f)
            if final_path:
                config.final_results_file = _resolve_path_f(base_dir, final_path)

            # Number of small watersheds
            n_ws_str = _next_data_or_empty(f)
            if n_ws_str:
                config.n_watersheds = int(n_ws_str)

            if config.n_watersheds <= 0:
                return config

            # Geospatial data section
            self._read_geospatial_data(f, config)

            # Root zone parameters section
            self._read_rootzone_data(f, config)

            # Aquifer parameters section
            self._read_aquifer_data(f, config)

            # Initial conditions section
            self._read_initial_conditions(f, config)

        return config

    def _read_geospatial_data(self, f: TextIO, config: SmallWatershedMainConfig) -> None:
        """Read geospatial data for all watersheds."""
        # Area factor
        area_str = _next_data_or_empty(f)
        if area_str:
            config.area_factor = float(area_str)

        # Flow rate factor
        flow_str = _next_data_or_empty(f)
        if flow_str:
            config.flow_factor = float(flow_str)

        # Time unit
        config.flow_time_unit = _next_data_or_empty(f)

        # Read per-watershed geospatial specs
        for _ in range(config.n_watersheds):
            line = self._next_data_line(f)
            parts = line.split()

            spec = WatershedSpec()
            spec.id = int(parts[0])
            spec.area = float(parts[1]) * config.area_factor
            spec.dest_stream_node = int(parts[2])
            n_gw_nodes = int(parts[3])

            # First GW node on this line
            if len(parts) >= 6:
                gw_id = int(parts[4])
                rate = float(parts[5])
                node = WatershedGWNode(gw_node_id=gw_id, max_perc_rate=abs(rate))
                if rate < 0:
                    node.is_baseflow = True
                    node.layer = int(abs(rate))
                spec.gw_nodes.append(node)

            # Remaining GW nodes on subsequent lines
            for _ in range(n_gw_nodes - 1):
                gw_line = self._next_data_line(f)
                gw_parts = gw_line.split()
                gw_id = int(gw_parts[0])
                rate = float(gw_parts[1])
                node = WatershedGWNode(gw_node_id=gw_id, max_perc_rate=abs(rate))
                if rate < 0:
                    node.is_baseflow = True
                    node.layer = int(abs(rate))
                spec.gw_nodes.append(node)

            config.watershed_specs.append(spec)

    def _read_rootzone_data(self, f: TextIO, config: SmallWatershedMainConfig) -> None:
        """Read root zone parameters for all watersheds."""
        # Solver tolerance
        tol_str = _next_data_or_empty(f)
        if tol_str:
            config.rz_solver_tolerance = float(tol_str)

        # Max iterations
        iter_str = _next_data_or_empty(f)
        if iter_str:
            config.rz_max_iterations = int(iter_str)

        # Length factor
        len_str = _next_data_or_empty(f)
        if len_str:
            config.rz_length_factor = float(len_str)

        # CN factor
        cn_str = _next_data_or_empty(f)
        if cn_str:
            config.rz_cn_factor = float(cn_str)

        # K factor
        k_str = _next_data_or_empty(f)
        if k_str:
            config.rz_k_factor = float(k_str)

        # K time unit
        config.rz_k_time_unit = _next_data_or_empty(f)

        # Per-watershed root zone parameters
        is_v41 = config.version == "4.1"
        for _ in range(config.n_watersheds):
            line = self._next_data_line(f)
            parts = line.split()

            params = WatershedRootZoneParams()
            try:
                idx = 0
                params.id = int(parts[idx])
                idx += 1
                params.precip_col = int(parts[idx])
                idx += 1
                params.precip_factor = float(parts[idx])
                idx += 1
                params.et_col = int(parts[idx])
                idx += 1
                if is_v41:
                    params.crop_coeff_col = int(parts[idx])
                    idx += 1
                params.wilting_point = float(parts[idx])
                idx += 1
                params.field_capacity = float(parts[idx])
                idx += 1
                params.total_porosity = float(parts[idx])
                idx += 1
                params.lambda_param = float(parts[idx])
                idx += 1
                params.root_depth = float(parts[idx]) * config.rz_length_factor
                idx += 1
                params.hydraulic_cond = float(parts[idx]) * config.rz_k_factor
                idx += 1
                params.kunsat_method = int(parts[idx])
                idx += 1
                params.curve_number = float(parts[idx]) * config.rz_cn_factor
            except (IndexError, ValueError):
                pass

            config.rootzone_params.append(params)

    def _read_aquifer_data(self, f: TextIO, config: SmallWatershedMainConfig) -> None:
        """Read aquifer parameters for all watersheds."""
        # GW factor
        gw_str = _next_data_or_empty(f)
        if gw_str:
            config.aq_gw_factor = float(gw_str)

        # Time factor
        time_str = _next_data_or_empty(f)
        if time_str:
            config.aq_time_factor = float(time_str)

        # Time unit
        config.aq_time_unit = _next_data_or_empty(f)

        # Per-watershed aquifer parameters
        for _ in range(config.n_watersheds):
            line = self._next_data_line(f)
            parts = line.split()

            params = WatershedAquiferParams()
            try:
                params.id = int(parts[0])
                params.gw_threshold = float(parts[1]) * config.aq_gw_factor
                params.max_gw_storage = float(parts[2]) * config.aq_gw_factor
                params.surface_flow_coeff = float(parts[3]) * config.aq_time_factor
                params.baseflow_coeff = float(parts[4]) * config.aq_time_factor
            except (IndexError, ValueError):
                pass

            config.aquifer_params.append(params)

    def _read_initial_conditions(self, f: TextIO, config: SmallWatershedMainConfig) -> None:
        """Read initial conditions for all watersheds.

        This section is optional — older files may end after the aquifer
        parameters. If EOF is reached, the method returns silently with
        default IC values.
        """
        # IC conversion factor
        ic_str = _next_data_or_empty(f)
        if not ic_str:
            return
        config.ic_factor = float(ic_str)

        # Per-watershed initial conditions: ID  SOILMOIST  GWSTOR
        for _ in range(config.n_watersheds):
            try:
                line = self._next_data_line(f)
            except FileFormatError:
                return
            parts = line.split()

            ic = WatershedInitialCondition()
            try:
                ic.id = int(parts[0])
                ic.soil_moisture = float(parts[1])
                ic.gw_storage = float(parts[2]) * config.ic_factor
            except (IndexError, ValueError):
                pass

            config.initial_conditions.append(ic)

    def _read_version(self, f: TextIO) -> str:
        """Read the version header."""
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

    def _next_data_line(self, f: TextIO) -> str:
        """Return the next non-comment data line."""
        for line in f:
            self._line_num += 1
            if _is_comment_line(line):
                continue
            return line.strip()
        raise FileFormatError("Unexpected end of file", line_number=self._line_num)


def read_small_watershed_main(
    filepath: Path | str, base_dir: Path | None = None
) -> SmallWatershedMainConfig:
    """Read IWFM small watershed component main file.

    Args:
        filepath: Path to the small watershed main file
        base_dir: Base directory for resolving relative paths

    Returns:
        SmallWatershedMainConfig with all configuration data
    """
    reader = SmallWatershedMainReader()
    return reader.read(filepath, base_dir)
