"""
IWFM-format Groundwater Main File Writer.

Writes the GW component main file from a GWMainFileConfig, preserving
all sections (sub-file paths, output paths, hydrograph locations, face flow
specs, aquifer parameters, Kh anomalies, and initial heads).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, TextIO

import numpy as np

from pyiwfm.io.groundwater import GWMainFileConfig
from pyiwfm.io.iwfm_writer import (
    ensure_parent_dir as _ensure_parent_dir,
)
from pyiwfm.io.iwfm_writer import (
    write_comment as _write_comment,
)
from pyiwfm.io.iwfm_writer import (
    write_value as _write_value,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from pyiwfm.components.groundwater import AquiferParameters


def _write_path(f: TextIO, path: Path | None, description: str = "") -> None:
    """Write a file path line (or blank if None)."""
    if path is not None:
        _write_value(f, str(path), description)
    else:
        _write_value(f, "", description)


def _format_aquifer_params_block(params: AquiferParameters) -> str:
    """Format the aquifer-parameter rows as a single string.

    Output is byte-identical to the previous per-cell ``f.write`` loop:
    layer 0 rows carry the 1-based node ID, continuation rows are indented.
    Coalescing avoids ``n_nodes * n_layers`` write syscalls on large models.
    """
    n_nodes = params.n_nodes
    n_layers = params.n_layers
    zero = np.zeros((n_nodes, n_layers), dtype=np.float64)
    kh = params.kh if params.kh is not None else zero
    kv = params.kv if params.kv is not None else zero
    ss = params.specific_storage if params.specific_storage is not None else zero
    sy = params.specific_yield if params.specific_yield is not None else zero
    akv = params.aquitard_kv if params.aquitard_kv is not None else zero

    parts: list[str] = []
    append = parts.append
    for i in range(n_nodes):
        for layer in range(n_layers):
            if layer == 0:
                append(f"     {i + 1:>6d}  ")
            else:
                append("             ")
            append(
                f"{kh[i, layer]:>12.6g}  {ss[i, layer]:>12.6g}  "
                f"{sy[i, layer]:>12.6g}  {kv[i, layer]:>12.6g}  "
                f"{akv[i, layer]:>12.6g}\n"
            )
    return "".join(parts)


def _format_initial_heads_block(heads: NDArray[np.float64]) -> str:
    """Format initial-head rows as a single string. Byte-identical to the
    previous per-node ``f.write`` loop."""
    n_nodes, n_layers = heads.shape
    parts: list[str] = []
    for i in range(n_nodes):
        vals = "  ".join(f"{heads[i, j]:>12.4f}" for j in range(n_layers))
        parts.append(f"     {i + 1:>6d}  {vals}\n")
    return "".join(parts)


def write_gw_main_file(config: GWMainFileConfig, filepath: Path | str) -> Path:
    """Write an IWFM-format GW main file from config.

    Args:
        config: Parsed GW main file configuration
        filepath: Output file path

    Returns:
        Path to written file
    """
    filepath = Path(filepath)
    _ensure_parent_dir(filepath)

    with open(filepath, "w", encoding="utf-8") as f:
        _write_comment(f, "IWFM Groundwater Component Main File")
        _write_comment(f, "Written by pyiwfm GWMainFileWriter")
        _write_comment(f, "")

        # Version header
        if config.version:
            f.write(f"#{config.version}\n")

        # Sub-file paths
        _write_path(f, config.bc_file, "BCFL - Boundary conditions file")
        _write_path(f, config.tile_drain_file, "TDFL - Tile drain file")
        _write_path(f, config.pumping_file, "PUMPFL - Pumping file")
        _write_path(f, config.subsidence_file, "SUBSFL - Subsidence file")
        _write_path(f, config.overwrite_file, "OVRWRTFL - Overwrite file")

        # Output conversion factors and units
        _write_value(f, config.head_output_factor, "FACTLTOU")
        _write_value(f, config.head_output_unit, "UNITLTOU")
        _write_value(f, config.volume_output_factor, "FACTVLOU")
        _write_value(f, config.volume_output_unit, "UNITVLOU")
        _write_value(f, config.velocity_output_factor, "FACTVROU")
        _write_value(f, config.velocity_output_unit, "UNITVROU")

        # Output files
        _write_path(f, config.velocity_output_file, "VELOUTFL")
        _write_path(f, config.vertical_flow_output_file, "VFLOWOUTFL")
        _write_path(f, config.head_all_output_file, "GWALLOUTFL")
        _write_path(f, config.head_tecplot_file, "HTPOUTFL")
        _write_path(f, config.velocity_tecplot_file, "VTPOUTFL")
        _write_path(f, config.budget_output_file, "GWBUDFL")
        _write_path(f, config.zbudget_output_file, "ZBUDFL")
        _write_path(f, config.final_heads_file, "FNGWFL")

        # Debug flag
        _write_value(f, config.debug_flag, "KDEB")

        # Hydrograph output section
        _write_value(f, len(config.hydrograph_locations), "NOUTH")
        _write_value(f, config.coord_factor, "FACTXY")
        _write_path(f, config.hydrograph_output_file, "GWHYDOUTFL")

        for loc in config.hydrograph_locations:
            name = loc.name or ""
            f.write(
                f"     {loc.node_id:>6d}  0  {loc.layer:>3d}  "
                f"{loc.x:>15.4f}  {loc.y:>15.4f}  / {name}\n"
            )

        # Face flow output section
        _write_value(f, config.n_face_flow_outputs, "NOUTF")
        _write_path(f, config.face_flow_output_file, "FCHYDOUTFL")
        for spec in config.face_flow_specs:
            f.write(f"     {spec}\n")

        # Aquifer parameters section
        if config.aquifer_params is not None:
            _write_value(f, 0, "NGROUP (direct input)")
            params = config.aquifer_params
            # Write conversion factors (all 1.0 since values are already converted)
            f.write("     1.0  1.0  1.0  1.0  1.0  1.0  / Conversion factors\n")
            f.write("     1DAY                          / Time unit\n")
            # Write per-node data — coalesced into a single f.write to avoid
            # n_nodes * n_layers syscalls on large models (~120k for C2VSimFG).
            f.write(_format_aquifer_params_block(params))

        # Kh anomalies
        _write_value(f, len(config.kh_anomalies), "NEBK")
        for anomaly in config.kh_anomalies:
            f.write(f"     {anomaly}\n")

        # Initial heads
        if config.initial_heads is not None:
            _write_comment(f, "Initial Groundwater Heads")
            _write_value(f, 1.0, "FACTICL")
            f.write(_format_initial_heads_block(config.initial_heads))

    return filepath
