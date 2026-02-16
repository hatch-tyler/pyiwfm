"""
Root zone component I/O handlers for IWFM model files.

This module provides functions for reading and writing IWFM root zone
component files including crop types, soil parameters, land use assignments,
and soil moisture data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

from pyiwfm.components.rootzone import (
    CropType,
    LandUseType,
    RootZone,
    SoilParameters,
)
from pyiwfm.core.exceptions import FileFormatError
from pyiwfm.io.iwfm_reader import (
    COMMENT_CHARS,
    is_comment_line as _is_comment_line,
    next_data_line,
    strip_inline_comment as _parse_value_line,
)

logger = logging.getLogger(__name__)


# ── Version utilities ────────────────────────────────────────────────


def parse_version(version: str) -> tuple[int, int]:
    """Parse a version string like ``'4.12'`` into ``(4, 12)``."""
    try:
        parts = version.split(".")
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
        return (major, minor)
    except (ValueError, IndexError):
        return (0, 0)


def version_ge(version: str, target: tuple[int, int]) -> bool:
    """Return True if *version* >= *target*."""
    return parse_version(version) >= target


# ── Per-element soil parameter row ───────────────────────────────────


@dataclass
class ElementSoilParamRow:
    """Per-element soil parameters from the root zone main file.

    Column layout varies by version — see
    :meth:`RootZoneMainFileReader._read_element_soil_params`.
    """

    element_id: int
    wilting_point: float
    field_capacity: float
    total_porosity: float
    lambda_param: float
    hydraulic_conductivity: float
    kunsat_method: int
    precip_column: int
    precip_factor: float
    generic_moisture_column: int
    # v4.0-v4.11: single surface flow destination
    surface_flow_dest_type: int = 0
    surface_flow_dest_id: int = 0
    # v4.0+: K_ponded (optional col 13 in v4.0, explicit in v4.12+)
    k_ponded: float = -1.0
    # v4.1+: capillary rise
    capillary_rise: float = 0.0
    # v4.12+: per-landuse surface flow destinations
    dest_ag: int = 0
    dest_urban_in: int = 0
    dest_urban_out: int = 0
    dest_nvrv: int = 0



@dataclass
class RootZoneFileConfig:
    """
    Configuration for root zone component files.

    Attributes:
        output_dir: Directory for output files
        crop_types_file: Crop types file name
        soil_params_file: Soil parameters file name
        landuse_file: Land use assignments file name
        ag_landuse_file: Agricultural land use file name
        urban_landuse_file: Urban land use file name
        native_landuse_file: Native/riparian land use file name
        soil_moisture_file: Initial soil moisture file name
    """

    output_dir: Path
    crop_types_file: str = "crop_types.dat"
    soil_params_file: str = "soil_params.dat"
    landuse_file: str = "landuse.dat"
    ag_landuse_file: str = "ag_landuse.dat"
    urban_landuse_file: str = "urban_landuse.dat"
    native_landuse_file: str = "native_landuse.dat"
    soil_moisture_file: str = "initial_soil_moisture.dat"

    def get_crop_types_path(self) -> Path:
        return self.output_dir / self.crop_types_file

    def get_soil_params_path(self) -> Path:
        return self.output_dir / self.soil_params_file

    def get_landuse_path(self) -> Path:
        return self.output_dir / self.landuse_file

    def get_ag_landuse_path(self) -> Path:
        return self.output_dir / self.ag_landuse_file

    def get_urban_landuse_path(self) -> Path:
        return self.output_dir / self.urban_landuse_file

    def get_native_landuse_path(self) -> Path:
        return self.output_dir / self.native_landuse_file

    def get_soil_moisture_path(self) -> Path:
        return self.output_dir / self.soil_moisture_file


class RootZoneWriter:
    """
    Writer for IWFM root zone component files.

    Writes all root zone-related input files including crop types, soil
    parameters, and land use assignments.

    Example:
        >>> config = RootZoneFileConfig(output_dir=Path("./model"))
        >>> writer = RootZoneWriter(config)
        >>> files = writer.write(rootzone_component)
    """

    def __init__(self, config: RootZoneFileConfig) -> None:
        """
        Initialize the root zone writer.

        Args:
            config: File configuration
        """
        self.config = config
        config.output_dir.mkdir(parents=True, exist_ok=True)

    def write(self, rootzone: RootZone) -> dict[str, Path]:
        """
        Write all root zone component files.

        Args:
            rootzone: RootZone component to write

        Returns:
            Dictionary mapping file type to output path
        """
        files: dict[str, Path] = {}

        # Write crop types
        if rootzone.crop_types:
            files["crop_types"] = self.write_crop_types(rootzone)

        # Write soil parameters
        if rootzone.soil_params:
            files["soil_params"] = self.write_soil_params(rootzone)

        # Write land use
        if rootzone.element_landuse:
            files["landuse"] = self.write_landuse(rootzone)

        # Write initial soil moisture
        if rootzone.soil_moisture is not None:
            files["soil_moisture"] = self.write_soil_moisture(rootzone)

        return files

    def write_crop_types(self, rootzone: RootZone, header: str | None = None) -> Path:
        """
        Write crop types file.

        Args:
            rootzone: RootZone component
            header: Optional header comment

        Returns:
            Path to written file
        """
        filepath = self.config.get_crop_types_path()

        with open(filepath, "w") as f:
            # Write header
            if header:
                for line in header.strip().split("\n"):
                    f.write(f"C  {line}\n")
            else:
                f.write("C  Crop types definition file\n")
                f.write("C  Generated by pyiwfm\n")
                f.write("C\n")
                f.write("C  ID   ROOT_DEPTH  KC       NAME\n")

            # Write crop count
            f.write(f"{len(rootzone.crop_types):<10}                              / NCROPS\n")

            # Write crops in ID order
            for crop_id in sorted(rootzone.crop_types.keys()):
                crop = rootzone.crop_types[crop_id]
                f.write(
                    f"{crop.id:<6} {crop.root_depth:>10.4f} {crop.kc:>8.4f}  {crop.name}\n"
                )

                # Write monthly Kc if available
                if crop.monthly_kc is not None:
                    f.write("C  Monthly Kc values (Jan-Dec):\n")
                    for i in range(0, 12, 6):
                        kc_vals = " ".join(f"{crop.monthly_kc[j]:>8.4f}" for j in range(i, min(i + 6, 12)))
                        f.write(f"   {kc_vals}\n")

        return filepath

    def write_soil_params(self, rootzone: RootZone, header: str | None = None) -> Path:
        """
        Write soil parameters file.

        Args:
            rootzone: RootZone component
            header: Optional header comment

        Returns:
            Path to written file
        """
        filepath = self.config.get_soil_params_path()

        with open(filepath, "w") as f:
            # Write header
            if header:
                for line in header.strip().split("\n"):
                    f.write(f"C  {line}\n")
            else:
                f.write("C  Soil parameters file\n")
                f.write("C  Generated by pyiwfm\n")
                f.write("C\n")
                f.write("C  ELEM  POROSITY  FIELD_CAP  WILT_PT  SAT_KV\n")

            # Write element count
            f.write(f"{len(rootzone.soil_params):<10}                              / NELEM_SOIL\n")

            # Write soil params in element order
            for elem_id in sorted(rootzone.soil_params.keys()):
                params = rootzone.soil_params[elem_id]
                f.write(
                    f"{elem_id:<6} {params.porosity:>10.6f} {params.field_capacity:>10.6f} "
                    f"{params.wilting_point:>10.6f} {params.saturated_kv:>12.6f}\n"
                )

        return filepath

    def write_landuse(self, rootzone: RootZone, header: str | None = None) -> Path:
        """
        Write land use assignments file.

        Args:
            rootzone: RootZone component
            header: Optional header comment

        Returns:
            Path to written file
        """
        filepath = self.config.get_landuse_path()

        # Group by land use type
        ag_landuse = [elu for elu in rootzone.element_landuse if elu.land_use_type == LandUseType.AGRICULTURAL]
        urban_landuse = [elu for elu in rootzone.element_landuse if elu.land_use_type == LandUseType.URBAN]
        native_landuse = [elu for elu in rootzone.element_landuse if elu.land_use_type == LandUseType.NATIVE_RIPARIAN]
        water_landuse = [elu for elu in rootzone.element_landuse if elu.land_use_type == LandUseType.WATER]

        with open(filepath, "w") as f:
            # Write header
            if header:
                for line in header.strip().split("\n"):
                    f.write(f"C  {line}\n")
            else:
                f.write("C  Land use assignments file\n")
                f.write("C  Generated by pyiwfm\n")
                f.write("C\n")

            # Write total count
            f.write(f"{len(rootzone.element_landuse):<10}                              / NLANDUSE\n")

            # Write agricultural land use
            f.write("C\n")
            f.write("C  AGRICULTURAL LAND USE\n")
            f.write(f"{len(ag_landuse):<10}                              / NAG_LANDUSE\n")

            for elu in ag_landuse:
                crop_str = " ".join(f"{cid}:{frac:.4f}" for cid, frac in elu.crop_fractions.items())
                f.write(f"{elu.element_id:<6} {elu.area:>14.4f}  {crop_str}\n")

            # Write urban land use
            f.write("C\n")
            f.write("C  URBAN LAND USE\n")
            f.write(f"{len(urban_landuse):<10}                              / NURBAN_LANDUSE\n")

            for elu in urban_landuse:
                f.write(
                    f"{elu.element_id:<6} {elu.area:>14.4f} {elu.impervious_fraction:>8.4f}\n"
                )

            # Write native/riparian land use
            f.write("C\n")
            f.write("C  NATIVE/RIPARIAN LAND USE\n")
            f.write(f"{len(native_landuse):<10}                              / NNATIVE_LANDUSE\n")

            for elu in native_landuse:
                f.write(f"{elu.element_id:<6} {elu.area:>14.4f}\n")

            # Write water land use
            f.write("C\n")
            f.write("C  WATER BODIES\n")
            f.write(f"{len(water_landuse):<10}                              / NWATER_LANDUSE\n")

            for elu in water_landuse:
                f.write(f"{elu.element_id:<6} {elu.area:>14.4f}\n")

        return filepath

    def write_soil_moisture(self, rootzone: RootZone, header: str | None = None) -> Path:
        """
        Write initial soil moisture file.

        Args:
            rootzone: RootZone component
            header: Optional header comment

        Returns:
            Path to written file
        """
        filepath = self.config.get_soil_moisture_path()

        if rootzone.soil_moisture is None:
            raise ValueError("No soil moisture data to write")

        with open(filepath, "w") as f:
            # Write header
            if header:
                for line in header.strip().split("\n"):
                    f.write(f"C  {line}\n")
            else:
                f.write("C  Initial soil moisture file\n")
                f.write("C  Generated by pyiwfm\n")
                f.write("C\n")

            # Write dimensions
            f.write(f"{rootzone.n_elements:<10}                              / NELEM\n")
            f.write(f"{rootzone.n_layers:<10}                              / NLAYERS\n")

            # Build header for layers
            layer_cols = "  ".join([f"SM_L{i+1:02d}" for i in range(rootzone.n_layers)])
            f.write(f"C  ELEM  {layer_cols}\n")

            # Write soil moisture data
            for elem_idx in range(rootzone.n_elements):
                elem_id = elem_idx + 1
                line = f"{elem_id:<5}"

                for layer in range(rootzone.n_layers):
                    sm = rootzone.soil_moisture[elem_idx, layer]
                    line += f" {sm:>10.6f}"

                f.write(line + "\n")

        return filepath


class RootZoneReader:
    """
    Reader for IWFM root zone component files.
    """

    def read_crop_types(self, filepath: Path | str) -> dict[int, CropType]:
        """
        Read crop types from file.

        Args:
            filepath: Path to crop types file

        Returns:
            Dictionary mapping crop ID to CropType object
        """
        filepath = Path(filepath)
        crops: dict[int, CropType] = {}

        with open(filepath) as f:
            line_num = 0
            n_crops = None

            # Find NCROPS
            for line in f:
                line_num += 1
                if _is_comment_line(line):
                    continue

                value, _ = _parse_value_line(line)
                try:
                    n_crops = int(value)
                except ValueError as e:
                    raise FileFormatError(
                        f"Invalid NCROPS value: '{value}'", line_number=line_num
                    ) from e
                break

            if n_crops is None:
                raise FileFormatError("Could not find NCROPS in file")

            # Read crop data
            for line in f:
                line_num += 1
                if _is_comment_line(line):
                    continue

                parts = line.split()
                if len(parts) < 3:
                    continue

                try:
                    crop_id = int(parts[0])
                    root_depth = float(parts[1])
                    kc = float(parts[2])
                    name = " ".join(parts[3:]) if len(parts) > 3 else ""

                    crops[crop_id] = CropType(
                        id=crop_id,
                        root_depth=root_depth,
                        kc=kc,
                        name=name,
                    )

                except ValueError as e:
                    raise FileFormatError(
                        f"Invalid crop data: '{line.strip()}'", line_number=line_num
                    ) from e

        return crops

    def read_soil_params(self, filepath: Path | str) -> dict[int, SoilParameters]:
        """
        Read soil parameters from file.

        Args:
            filepath: Path to soil parameters file

        Returns:
            Dictionary mapping element ID to SoilParameters object
        """
        filepath = Path(filepath)
        params: dict[int, SoilParameters] = {}

        with open(filepath) as f:
            line_num = 0
            n_elem = None

            # Find NELEM_SOIL
            for line in f:
                line_num += 1
                if _is_comment_line(line):
                    continue

                value, _ = _parse_value_line(line)
                try:
                    n_elem = int(value)
                except ValueError as e:
                    raise FileFormatError(
                        f"Invalid NELEM_SOIL value: '{value}'", line_number=line_num
                    ) from e
                break

            if n_elem is None:
                raise FileFormatError("Could not find NELEM_SOIL in file")

            # Read parameter data
            for line in f:
                line_num += 1
                if _is_comment_line(line):
                    continue

                parts = line.split()
                if len(parts) < 5:
                    continue

                try:
                    elem_id = int(parts[0])
                    porosity = float(parts[1])
                    field_capacity = float(parts[2])
                    wilting_point = float(parts[3])
                    saturated_kv = float(parts[4])

                    params[elem_id] = SoilParameters(
                        porosity=porosity,
                        field_capacity=field_capacity,
                        wilting_point=wilting_point,
                        saturated_kv=saturated_kv,
                    )

                except ValueError as e:
                    raise FileFormatError(
                        f"Invalid soil parameter data: '{line.strip()}'",
                        line_number=line_num,
                    ) from e

        return params


# =============================================================================
# Component Main File Reader (hierarchical dispatcher file)
# =============================================================================


@dataclass
class RootZoneMainFileConfig:
    """
    Configuration parsed from RootZone component main file.

    The root zone component main file is a dispatcher that contains
    solver parameters and paths to sub-files for crop, urban, and
    native vegetation data.

    Attributes:
        version: File format version (e.g., "4.0", "4.11")
        convergence_tolerance: Root zone solver convergence tolerance
        max_iterations: Maximum solver iterations
        length_conversion: Length unit conversion factor (FACTCN)
        gw_uptake_enabled: Whether ET from GW is enabled (v4.11+ only)
        nonponded_crop_file: Path to non-ponded agricultural crop file
        ponded_crop_file: Path to ponded agricultural crop file
        urban_file: Path to urban land use file
        native_veg_file: Path to native/riparian vegetation file
        return_flow_file: Path to return flow fractions file
        reuse_file: Path to water reuse specifications file
        irrigation_period_file: Path to irrigation period file
        generic_moisture_file: Path to generic moisture data file
        ag_water_demand_file: Path to agricultural water supply requirement file
        lwu_budget_file: Path to land and water use budget output
        rz_budget_file: Path to root zone budget output
        lwu_zone_budget_file: Path to LWU zone budget output (v4.11+ only)
        rz_zone_budget_file: Path to RZ zone budget output (v4.11+ only)
        lu_area_scale_file: Path to land use area scaling output (v4.11+ only)
        final_moisture_file: Path to end-of-simulation soil moisture output
    """

    version: str = ""
    convergence_tolerance: float = 1e-8
    max_iterations: int = 2000
    length_conversion: float = 1.0
    gw_uptake_enabled: bool = False
    nonponded_crop_file: Path | None = None
    ponded_crop_file: Path | None = None
    urban_file: Path | None = None
    native_veg_file: Path | None = None
    return_flow_file: Path | None = None
    reuse_file: Path | None = None
    irrigation_period_file: Path | None = None
    generic_moisture_file: Path | None = None
    ag_water_demand_file: Path | None = None
    lwu_budget_file: Path | None = None
    rz_budget_file: Path | None = None
    lwu_zone_budget_file: Path | None = None  # v4.11+ only
    rz_zone_budget_file: Path | None = None   # v4.11+ only
    lu_area_scale_file: Path | None = None    # v4.11+ only
    final_moisture_file: Path | None = None
    # Soil parameter conversion factors
    k_factor: float = 1.0
    k_exdth_factor: float = 1.0  # FACTEXDTH (v4.1+ only, capillary rise)
    k_time_unit: str = ""
    # v4.12+: surface flow destinations file
    surface_flow_dest_file: Path | None = None
    # Per-element soil parameters
    element_soil_params: list[ElementSoilParamRow] = field(default_factory=list)


class RootZoneMainFileReader:
    """
    Reader for IWFM rootzone component main file.

    The RootZone main file is a hierarchical dispatcher that contains:
    1. Version header (e.g., #4.11)
    2. Solver parameters (convergence, iterations)
    3. Paths to sub-files for different land use types
    4. Output file paths
    """

    def __init__(self) -> None:
        self._line_num = 0

    def read(
        self,
        filepath: Path | str,
        base_dir: Path | None = None,
        n_elements: int = 0,
    ) -> RootZoneMainFileConfig:
        """
        Parse RootZone main file.

        Supports v4.0 through v4.13 formats.  v4.1+ adds capillary
        rise; v4.11+ adds ET-from-GW flag and zone budget files;
        v4.12+ adds per-landuse surface flow destinations.

        Args:
            filepath: Path to the RootZone component main file
            base_dir: Base directory for resolving relative paths.
                     If None, uses the parent directory of filepath.
            n_elements: Expected number of elements (from mesh).
                     If > 0, the soil parameter parser will read
                     exactly this many rows instead of relying on
                     column-count heuristics.

        Returns:
            RootZoneMainFileConfig with parsed values
        """
        filepath = Path(filepath)
        if base_dir is None:
            base_dir = filepath.parent

        config = RootZoneMainFileConfig()
        self._line_num = 0

        with open(filepath) as f:
            # Read version header
            config.version = self._read_version(f)
            is_v411_plus = self._version_ge_411(config.version)
            is_v401_plus = version_ge(config.version, (4, 1))
            is_v412_plus = version_ge(config.version, (4, 12))

            # RZCONV (convergence tolerance)
            rzconv = self._next_data_or_empty(f)
            if rzconv:
                try:
                    config.convergence_tolerance = float(rzconv)
                except ValueError:
                    pass

            # RZITERMX (maximum iterations)
            rzitermx = self._next_data_or_empty(f)
            if rzitermx:
                try:
                    config.max_iterations = int(rzitermx)
                except ValueError:
                    pass

            # FACTCN (length conversion factor)
            factcn = self._next_data_or_empty(f)
            if factcn:
                try:
                    config.length_conversion = float(factcn)
                except ValueError:
                    pass

            # iFlagETFromGW (v4.11+ only: 0=off, 1=on)
            if is_v411_plus:
                gwuptk = self._next_data_or_empty(f)
                if gwuptk:
                    try:
                        config.gw_uptake_enabled = int(gwuptk) > 0
                    except ValueError:
                        pass

            # AGNPFL (non-ponded agricultural crop file)
            agnp_path = self._next_data_or_empty(f)
            if agnp_path:
                config.nonponded_crop_file = self._resolve_path(base_dir, agnp_path)

            # PFL (ponded agricultural crop file)
            p_path = self._next_data_or_empty(f)
            if p_path:
                config.ponded_crop_file = self._resolve_path(base_dir, p_path)

            # URBFL (urban land use file)
            urb_path = self._next_data_or_empty(f)
            if urb_path:
                config.urban_file = self._resolve_path(base_dir, urb_path)

            # NVRVFL (native/riparian vegetation file)
            nvrv_path = self._next_data_or_empty(f)
            if nvrv_path:
                config.native_veg_file = self._resolve_path(base_dir, nvrv_path)

            # Return flow fractions file
            ret_path = self._next_data_or_empty(f)
            if ret_path:
                config.return_flow_file = self._resolve_path(base_dir, ret_path)

            # Reuse file
            reuse_path = self._next_data_or_empty(f)
            if reuse_path:
                config.reuse_file = self._resolve_path(base_dir, reuse_path)

            # Irrigation period file
            irig_path = self._next_data_or_empty(f)
            if irig_path:
                config.irrigation_period_file = self._resolve_path(base_dir, irig_path)

            # Generic moisture file (optional, may be blank)
            gm_path = self._next_data_or_empty(f)
            if gm_path:
                config.generic_moisture_file = self._resolve_path(base_dir, gm_path)

            # Ag water demand file (optional, may be blank)
            agwd_path = self._next_data_or_empty(f)
            if agwd_path:
                config.ag_water_demand_file = self._resolve_path(base_dir, agwd_path)

            # LWU budget output file
            lwu_path = self._next_data_or_empty(f)
            if lwu_path:
                config.lwu_budget_file = self._resolve_path(base_dir, lwu_path)

            # RZ budget output file
            rz_path = self._next_data_or_empty(f)
            if rz_path:
                config.rz_budget_file = self._resolve_path(base_dir, rz_path)

            # v4.11+: Zone budget files
            if is_v411_plus:
                lwu_zb_path = self._next_data_or_empty(f)
                if lwu_zb_path:
                    config.lwu_zone_budget_file = self._resolve_path(base_dir, lwu_zb_path)

                rz_zb_path = self._next_data_or_empty(f)
                if rz_zb_path:
                    config.rz_zone_budget_file = self._resolve_path(base_dir, rz_zb_path)

            # LU area scaling output (v4.11+)
            if is_v411_plus:
                lu_scale_path = self._next_data_or_empty(f)
                if lu_scale_path:
                    config.lu_area_scale_file = self._resolve_path(base_dir, lu_scale_path)

            # Final moisture output file
            final_path = self._next_data_or_empty(f)
            if final_path:
                config.final_moisture_file = self._resolve_path(base_dir, final_path)

            # ── Soil parameter section ────────────────────────────

            # FACTK (K conversion factor)
            factk = self._next_data_or_empty(f)
            if factk:
                try:
                    config.k_factor = float(factk)
                except ValueError:
                    pass

            # FACTEXDTH (capillary rise factor, v4.1+ only)
            if is_v401_plus:
                factexdth = self._next_data_or_empty(f)
                if factexdth:
                    try:
                        config.k_exdth_factor = float(factexdth)
                    except ValueError:
                        pass

            # TUNITK (time unit for K)
            tunitk = self._next_data_or_empty(f)
            if tunitk:
                config.k_time_unit = tunitk

            # v4.12+: Surface flow destinations file
            if is_v412_plus:
                dest_path = self._next_data_or_empty(f)
                if dest_path:
                    config.surface_flow_dest_file = self._resolve_path(
                        base_dir, dest_path
                    )

            # Per-element soil parameters
            config.element_soil_params = self._read_element_soil_params(
                f, config, n_elements=n_elements
            )

        return config

    @staticmethod
    def _version_ge_411(version: str) -> bool:
        """Check if version is >= 4.11."""
        try:
            parts = version.split(".")
            major = int(parts[0])
            minor = int(parts[1]) if len(parts) > 1 else 0
            if major > 4:
                return True
            if major == 4 and minor >= 11:
                return True
            return False
        except (ValueError, IndexError):
            return False

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
        for line in f:
            self._line_num += 1
            if line and line[0] in COMMENT_CHARS:
                continue
            value, _ = _parse_value_line(line)
            return value
        return ""

    def _resolve_path(self, base_dir: Path, filepath: str) -> Path:
        """Resolve a file path relative to base directory."""
        path = Path(filepath.strip())
        if path.is_absolute():
            return path
        return base_dir / path

    def _read_element_soil_params(
        self,
        f: TextIO,
        config: RootZoneMainFileConfig,
        n_elements: int = 0,
    ) -> list[ElementSoilParamRow]:
        """Read per-element soil parameters.

        Column layout depends on version:

        v4.0/v4.01 (12-13 cols):
            IE WP FC TN Lambda K KMethod iPrecCol fPrecFac iGenCol
            iDestType iDest [K_Ponded]

        v4.1/v4.11 (13-14 cols):
            IE WP FC TN Lambda K KMethod CapRise iPrecCol fPrecFac
            iGenCol iDestType iDest [K_Ponded]

        v4.12/v4.13 (16 cols):
            IE WP FC TN Lambda K K_Ponded KMethod CapRise iPrecCol
            fPrecFac iGenCol iDestAg iDestUrbIn iDestUrbOut iDestNVRV

        Args:
            f: Open file handle positioned at start of soil params.
            config: RootZone config with version info.
            n_elements: Expected element count from mesh.  When > 0
                the parser reads exactly this many data rows,
                tolerating short lines as long as the first column
                is a valid integer element ID.
        """
        rows: list[ElementSoilParamRow] = []
        version = parse_version(config.version)
        is_v41 = version >= (4, 1)
        is_v412 = version >= (4, 12)

        if is_v412:
            min_cols = 16
        elif is_v41:
            min_cols = 13
        else:
            min_cols = 12

        blank_count = 0
        max_blanks = 3  # tolerate up to 3 consecutive blank lines
        target = n_elements if n_elements > 0 else 0
        line_counter = [self._line_num]
        while True:
            if target > 0 and len(rows) >= target:
                break
            # Use raw line reader — matches Fortran's free-format READ
            # which reads exactly N values and ignores the rest of the
            # line (including inline '/ description' comments).
            raw_line = next_data_line(f, line_counter=line_counter)
            self._line_num = line_counter[0]
            if not raw_line:
                # Tolerate blank lines within the section
                blank_count += 1
                if blank_count > max_blanks:
                    break
                continue
            blank_count = 0
            parts = raw_line.split()
            if len(parts) < min_cols:
                if target > 0 and len(rows) < target:
                    # When we know the expected count, skip short
                    # lines (section delimiters) and keep reading
                    logger.debug(
                        "Skipping short line (%d cols, need %d) at "
                        "line %d while reading soil params "
                        "(%d/%d rows read)",
                        len(parts),
                        min_cols,
                        self._line_num,
                        len(rows),
                        target,
                    )
                    continue
                break
            # Take only the first min_cols tokens — Fortran reads
            # exactly N values and ignores the rest (inline comments,
            # descriptions).
            parts = parts[:min_cols]
            try:
                row = self._parse_soil_param_row(parts, config)
                rows.append(row)
            except (ValueError, IndexError):
                if target > 0 and len(rows) < target:
                    logger.debug(
                        "Skipping unparseable line at %d while "
                        "reading soil params (%d/%d rows read)",
                        self._line_num,
                        len(rows),
                        target,
                    )
                    continue
                break

        if rows:
            logger.debug(
                "Read %d soil parameter rows (elements %d–%d) from rootzone "
                "main file (version %s)",
                len(rows),
                rows[0].element_id,
                rows[-1].element_id,
                config.version,
            )
        if target > 0 and len(rows) != target:
            logger.warning(
                "Expected %d soil parameter rows but read %d",
                target,
                len(rows),
            )
        return rows

    def _parse_soil_param_row(
        self,
        parts: list[str],
        config: RootZoneMainFileConfig,
    ) -> ElementSoilParamRow:
        """Parse a single soil parameter row based on version."""
        version = parse_version(config.version)
        is_v41 = version >= (4, 1)
        is_v412 = version >= (4, 12)

        if is_v412:
            # v4.12+: IE WP FC TN Lambda K K_Ponded KMethod CapRise
            #         iPrecCol fPrecFac iGenCol iDestAg iDestUrbIn
            #         iDestUrbOut iDestNVRV
            return ElementSoilParamRow(
                element_id=int(parts[0]),
                wilting_point=float(parts[1]),
                field_capacity=float(parts[2]),
                total_porosity=float(parts[3]),
                lambda_param=float(parts[4]),
                hydraulic_conductivity=float(parts[5]),
                k_ponded=float(parts[6]),
                kunsat_method=int(parts[7]),
                capillary_rise=float(parts[8]),
                precip_column=int(parts[9]),
                precip_factor=float(parts[10]),
                generic_moisture_column=int(parts[11]),
                dest_ag=int(parts[12]),
                dest_urban_in=int(parts[13]),
                dest_urban_out=int(parts[14]),
                dest_nvrv=int(parts[15]),
            )
        elif is_v41:
            # v4.1/v4.11: IE WP FC TN Lambda K KMethod CapRise
            #             iPrecCol fPrecFac iGenCol iDestType iDest
            #             [K_Ponded]
            k_ponded = float(parts[13]) if len(parts) > 13 else -1.0
            return ElementSoilParamRow(
                element_id=int(parts[0]),
                wilting_point=float(parts[1]),
                field_capacity=float(parts[2]),
                total_porosity=float(parts[3]),
                lambda_param=float(parts[4]),
                hydraulic_conductivity=float(parts[5]),
                kunsat_method=int(parts[6]),
                capillary_rise=float(parts[7]),
                precip_column=int(parts[8]),
                precip_factor=float(parts[9]),
                generic_moisture_column=int(parts[10]),
                surface_flow_dest_type=int(parts[11]),
                surface_flow_dest_id=int(parts[12]),
                k_ponded=k_ponded,
            )
        else:
            # v4.0/v4.01: IE WP FC TN Lambda K KMethod iPrecCol
            #             fPrecFac iGenCol iDestType iDest [K_Ponded]
            k_ponded = float(parts[12]) if len(parts) > 12 else -1.0
            return ElementSoilParamRow(
                element_id=int(parts[0]),
                wilting_point=float(parts[1]),
                field_capacity=float(parts[2]),
                total_porosity=float(parts[3]),
                lambda_param=float(parts[4]),
                hydraulic_conductivity=float(parts[5]),
                kunsat_method=int(parts[6]),
                precip_column=int(parts[7]),
                precip_factor=float(parts[8]),
                generic_moisture_column=int(parts[9]),
                surface_flow_dest_type=int(parts[10]),
                surface_flow_dest_id=int(parts[11]),
                k_ponded=k_ponded,
            )


# Convenience functions


def write_rootzone(
    rootzone: RootZone,
    output_dir: Path | str,
    config: RootZoneFileConfig | None = None,
) -> dict[str, Path]:
    """
    Write root zone component to files.

    Args:
        rootzone: RootZone component to write
        output_dir: Output directory
        config: Optional file configuration

    Returns:
        Dictionary mapping file type to output path
    """
    output_dir = Path(output_dir)

    if config is None:
        config = RootZoneFileConfig(output_dir=output_dir)
    else:
        config.output_dir = output_dir

    writer = RootZoneWriter(config)
    return writer.write(rootzone)


def read_crop_types(filepath: Path | str) -> dict[int, CropType]:
    """
    Read crop types from file.

    Args:
        filepath: Path to crop types file

    Returns:
        Dictionary mapping crop ID to CropType object
    """
    reader = RootZoneReader()
    return reader.read_crop_types(filepath)


def read_soil_params(filepath: Path | str) -> dict[int, SoilParameters]:
    """
    Read soil parameters from file.

    Args:
        filepath: Path to soil parameters file

    Returns:
        Dictionary mapping element ID to SoilParameters object
    """
    reader = RootZoneReader()
    return reader.read_soil_params(filepath)


def read_rootzone_main_file(
    filepath: Path | str, base_dir: Path | None = None
) -> RootZoneMainFileConfig:
    """
    Read IWFM rootzone component main file.

    The RootZone main file is a hierarchical dispatcher that contains
    solver parameters and paths to sub-files for different land use types.

    Args:
        filepath: Path to the RootZone component main file
        base_dir: Base directory for resolving relative paths.
                 If None, uses the parent directory of filepath.

    Returns:
        RootZoneMainFileConfig with parsed values

    Example:
        >>> config = read_rootzone_main_file("C2VSimFG_RootZone.dat")
        >>> print(f"Version: {config.version}")
        >>> print(f"GW uptake: {config.gw_uptake_enabled}")
        >>> if config.nonponded_crop_file:
        ...     print(f"Crop file: {config.nonponded_crop_file}")
    """
    reader = RootZoneMainFileReader()
    return reader.read(filepath, base_dir)
