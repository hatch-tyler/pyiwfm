"""
Unsaturated Zone Main File Reader for IWFM.

This module reads the IWFM unsaturated zone component main file, which
defines vadose zone modeling parameters including:
1. Number of unsaturated layers
2. Solver parameters
3. Budget output file paths
4. Layer properties (per element)
5. Initial soil moisture conditions

The unsaturated zone is an optional component used for detailed
vadose zone dynamics between the root zone and groundwater.

Reference: Package_AppUnsatZone.f90 - New()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

import numpy as np
from numpy.typing import NDArray

from pyiwfm.io.iwfm_reader import (
    COMMENT_CHARS,
)
from pyiwfm.io.iwfm_reader import (
    next_data_or_empty as _next_data_or_empty,
)
from pyiwfm.io.iwfm_reader import (
    resolve_path as _resolve_path_f,
)

logger = logging.getLogger(__name__)


@dataclass
class UnsatZoneElementData:
    """Per-element unsaturated zone parameters for all layers.

    Fortran format per line: ``ElemID, [ThickMax_L1, Porosity_L1, Lambda_L1,
    HydCond_L1, KMethod_L1, ...(repeat per layer)]``.
    Total values per line: ``1 + 5 * n_layers``.

    Attributes:
        element_id: 1-based element ID.
        thickness_max: Maximum thickness per layer, shape (n_layers,).
        total_porosity: Total porosity per layer, shape (n_layers,).
        lambda_param: Pore size distribution parameter per layer, shape (n_layers,).
        hyd_cond: Saturated hydraulic conductivity per layer, shape (n_layers,).
        kunsat_method: Unsaturated K method per layer, shape (n_layers,).
    """

    element_id: int
    thickness_max: NDArray[np.float64]
    total_porosity: NDArray[np.float64]
    lambda_param: NDArray[np.float64]
    hyd_cond: NDArray[np.float64]
    kunsat_method: NDArray[np.int32]


@dataclass
class UnsatZoneMainConfig:
    """Configuration parsed from Unsaturated Zone component main file.

    Attributes:
        version: File format version
        n_layers: Number of unsaturated zone layers (0=disabled)
        solver_tolerance: Solver convergence tolerance
        max_iterations: Maximum solver iterations
        budget_file: Path to HDF5 budget output file
        zbudget_file: Path to HDF5 zone budget output file
        final_results_file: Path to final simulation results file
        n_parametric_grids: Number of parametric grids (0=direct input)
        coord_factor: Conversion factor for x-y coordinates
        thickness_factor: Conversion factor for layer thickness
        hyd_cond_factor: Conversion factor for hydraulic conductivity
        time_unit: Time unit for hydraulic conductivity
        element_data: Per-element unsaturated zone parameters
        initial_soil_moisture: Initial soil moisture per element per layer.
            Maps element_id -> moisture array. Key 0 means uniform for all.
    """

    version: str = ""
    n_layers: int = 0
    solver_tolerance: float = 1e-8
    max_iterations: int = 2000
    budget_file: Path | None = None
    zbudget_file: Path | None = None
    final_results_file: Path | None = None
    n_parametric_grids: int = 0

    # Conversion factors (from ReadUnsatZoneParameters)
    coord_factor: float = 1.0
    thickness_factor: float = 1.0
    hyd_cond_factor: float = 1.0
    time_unit: str = ""

    # Per-element data (NGROUP == 0 only)
    element_data: list[UnsatZoneElementData] = field(default_factory=list)

    # Initial conditions: element_id -> moisture per layer
    initial_soil_moisture: dict[int, NDArray[np.float64]] = field(default_factory=dict)


class UnsatZoneMainReader:
    """Reader for IWFM unsaturated zone component main file.

    Parses the full file including header configuration, per-element
    parameter data (when NGROUP=0), and initial conditions.
    """

    def __init__(self) -> None:
        self._line_num = 0

    def read(self, filepath: Path | str, base_dir: Path | None = None) -> UnsatZoneMainConfig:
        """Read unsaturated zone main file.

        Args:
            filepath: Path to the unsaturated zone main file
            base_dir: Base directory for resolving relative paths

        Returns:
            UnsatZoneMainConfig with configuration data
        """
        filepath = Path(filepath)
        if base_dir is None:
            base_dir = filepath.parent

        config = UnsatZoneMainConfig()
        self._line_num = 0

        with open(filepath) as f:
            # Version header
            config.version = self._read_version(f)

            # Number of unsaturated zone layers
            n_layers_str = _next_data_or_empty(f)
            if n_layers_str:
                config.n_layers = int(n_layers_str)

            if config.n_layers <= 0:
                return config

            # Solver tolerance
            tol_str = _next_data_or_empty(f)
            if tol_str:
                config.solver_tolerance = float(tol_str)

            # Max iterations
            iter_str = _next_data_or_empty(f)
            if iter_str:
                config.max_iterations = int(iter_str)

            # Budget output file (HDF5)
            budget_path = _next_data_or_empty(f)
            if budget_path:
                config.budget_file = _resolve_path_f(base_dir, budget_path)

            # Zone budget output file (HDF5)
            zbudget_path = _next_data_or_empty(f)
            if zbudget_path:
                config.zbudget_file = _resolve_path_f(base_dir, zbudget_path)

            # Final results output file
            final_path = _next_data_or_empty(f)
            if final_path:
                config.final_results_file = _resolve_path_f(base_dir, final_path)

            # Number of parametric grids (NGROUP)
            ngroup_str = _next_data_or_empty(f)
            if ngroup_str:
                config.n_parametric_grids = int(ngroup_str)

            # ── Parameter section (ReadUnsatZoneParameters) ────────
            # Conversion factors: FX, FThickness, FHydCond (3 values)
            factors_str = _next_data_or_empty(f)
            if factors_str:
                fparts = factors_str.split()
                if len(fparts) >= 1:
                    config.coord_factor = float(fparts[0])
                if len(fparts) >= 2:
                    config.thickness_factor = float(fparts[1])
                if len(fparts) >= 3:
                    config.hyd_cond_factor = float(fparts[2])

            # Time unit for hydraulic conductivity
            time_unit_str = _next_data_or_empty(f)
            if time_unit_str:
                config.time_unit = time_unit_str

            # Per-element data (when NGROUP == 0)
            if config.n_parametric_grids == 0:
                try:
                    config.element_data = self._read_element_params(f, config)
                except Exception as exc:
                    logger.warning(
                        "Failed to read element parameters at line %d: %s",
                        self._line_num,
                        exc,
                    )

            # Initial conditions
            try:
                config.initial_soil_moisture = self._read_initial_conditions(f, config)
            except Exception as exc:
                logger.warning(
                    "Failed to read initial conditions at line %d: %s",
                    self._line_num,
                    exc,
                )

        return config

    def _read_element_params(
        self, f: TextIO, config: UnsatZoneMainConfig
    ) -> list[UnsatZoneElementData]:
        """Read per-element unsaturated zone parameters.

        Fortran format per line: ElemID, [ThickMax_L1, Porosity_L1, Lambda_L1,
                                  HydCond_L1, KMethod_L1, ...(repeat per layer)]
        Total values per line: 1 + 5 * n_layers

        The number of element lines is not explicitly specified in the file;
        the Fortran reads NElements lines (known from the grid). Since we may
        not know NElements, we read until the line no longer matches the
        expected field count.
        """
        elements: list[UnsatZoneElementData] = []
        expected_fields = 1 + 5 * config.n_layers
        # Save position for potential backtrack via _pushback
        self._pushback_line: str | None = None

        while True:
            line_val = _next_data_or_empty(f)
            if not line_val:
                break
            parts = line_val.split()
            if len(parts) < expected_fields:
                # Not enough fields — probably the next section
                # Store this line for the initial conditions reader
                self._pushback_line = line_val
                break
            try:
                elem_id = int(parts[0])
                vals = [float(v) for v in parts[1:]]
                n = config.n_layers
                elements.append(
                    UnsatZoneElementData(
                        element_id=elem_id,
                        thickness_max=np.array(vals[0::5][:n]) * config.thickness_factor,
                        total_porosity=np.array(vals[1::5][:n]),
                        lambda_param=np.array(vals[2::5][:n]),
                        hyd_cond=np.array(vals[3::5][:n]) * config.hyd_cond_factor,
                        kunsat_method=np.array(vals[4::5][:n], dtype=np.int32),
                    )
                )
            except (ValueError, IndexError) as exc:
                logger.warning(
                    "Skipping malformed element data at line %d: %s",
                    self._line_num,
                    exc,
                )
                continue
        return elements

    def _read_initial_conditions(
        self, f: TextIO, config: UnsatZoneMainConfig
    ) -> dict[int, NDArray[np.float64]]:
        """Read initial soil moisture per element per layer.

        Fortran format per line: ElemID, SoilM_L1, SoilM_L2, ...
        Special case: If first ElemID is 0, same values for all elements.
        """
        ic: dict[int, NDArray[np.float64]] = {}
        expected_fields = 1 + config.n_layers

        # Check if we have a pushed-back line from element params
        first_line = getattr(self, "_pushback_line", None)
        self._pushback_line = None

        if first_line is None:
            first_line = _next_data_or_empty(f)
        if not first_line:
            return ic

        parts = first_line.split()
        if len(parts) < expected_fields:
            return ic

        try:
            elem_id = int(parts[0])
            moisture = np.array([float(v) for v in parts[1 : 1 + config.n_layers]])
        except (ValueError, IndexError):
            return ic

        if elem_id == 0:
            # Same values for all elements — store under key 0
            ic[0] = moisture
            return ic

        ic[elem_id] = moisture

        # Read remaining elements
        while True:
            line_val = _next_data_or_empty(f)
            if not line_val:
                break
            parts = line_val.split()
            if len(parts) < expected_fields:
                break
            try:
                elem_id = int(parts[0])
                moisture = np.array([float(v) for v in parts[1 : 1 + config.n_layers]])
                ic[elem_id] = moisture
            except (ValueError, IndexError):
                continue
        return ic

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


def read_unsaturated_zone_main(
    filepath: Path | str, base_dir: Path | None = None
) -> UnsatZoneMainConfig:
    """Read IWFM unsaturated zone component main file.

    Args:
        filepath: Path to the unsaturated zone main file
        base_dir: Base directory for resolving relative paths

    Returns:
        UnsatZoneMainConfig with configuration data
    """
    reader = UnsatZoneMainReader()
    return reader.read(filepath, base_dir)
