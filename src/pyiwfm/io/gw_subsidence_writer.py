"""
IWFM-format Groundwater Subsidence Writer.

Writes the subsidence parameter file and initial conditions file
from a SubsidenceConfig. Supports both version 4.0 and 5.0 formats.

The writer produces output that is compatible with the SubsidenceReader,
enabling roundtrip read-write fidelity.
"""

from __future__ import annotations

from pathlib import Path
from typing import TextIO

from pyiwfm.io.gw_subsidence import SubsidenceConfig
from pyiwfm.io.iwfm_writer import (
    ensure_parent_dir as _ensure_parent_dir,
)
from pyiwfm.io.iwfm_writer import (
    write_comment as _write_comment,
)
from pyiwfm.io.iwfm_writer import (
    write_value as _write_value,
)


def write_subsidence_main(config: SubsidenceConfig, filepath: Path | str) -> Path:
    """Write the subsidence main parameter file.

    This writes the complete subsidence file including version header,
    sub-file references, output settings, hydrograph output section,
    conversion factors, and node-level parameters.

    Args:
        config: Subsidence configuration with all parameters
        filepath: Output file path

    Returns:
        Path to written file
    """
    filepath = Path(filepath)
    _ensure_parent_dir(filepath)

    is_v50 = config.version.startswith("5")

    with open(filepath, "w") as f:
        _write_comment(f, "IWFM Subsidence Parameter File")

        # Version header
        if config.version:
            f.write(f"#{config.version}\n")

        # IC file (use raw path for roundtrip fidelity)
        ic_str = config._raw_ic_file or str(config.ic_file or "")
        _write_value(f, ic_str, "Initial conditions file")

        # Tecplot output file
        tec_str = config._raw_tecplot_file or str(config.tecplot_file or "")
        _write_value(f, tec_str, "Tecplot output file")

        # Final subsidence output file
        final_str = config._raw_final_subs_file or str(config.final_subs_file or "")
        _write_value(f, final_str, "Final subsidence output file")

        # Output conversion factor
        _write_value(f, config.output_factor, "Output conversion factor")

        # Output unit
        _write_value(f, config.output_unit, "Output unit")

        # Hydrograph output section: NOUTS
        _write_value(f, config.n_hydrograph_outputs, "NOUTS")

        if config.n_hydrograph_outputs > 0:
            # FACTXY (coordinate conversion factor)
            _write_value(f, config.hydrograph_coord_factor, "FACTXY")

            # SUBHYDOUTFL (output file path)
            hyd_str = config._raw_hydrograph_output_file or str(config.hydrograph_output_file or "")
            _write_value(f, hyd_str, "SUBHYDOUTFL")

            # Hydrograph specs: ID HYDTYP ILYR X Y [/ NAME] or ID HYDTYP ILYR IOUTS [NAME]
            for spec in config.hydrograph_specs:
                name_part = f"     {spec.name}" if spec.name else ""
                if spec.hydtyp == 0:
                    # X-Y coordinate format: reverse FACTXY
                    factor = config.hydrograph_coord_factor
                    x = spec.x / factor if factor != 0.0 else spec.x
                    y = spec.y / factor if factor != 0.0 else spec.y
                    f.write(
                        f"    {spec.id:>4d}       {spec.hydtyp}         "
                        f"{spec.layer}         {x:.1f}    {y:.1f}"
                        f"            {name_part}\n"
                    )
                else:
                    # Node number format: x holds node ID
                    node_id = int(spec.x)
                    f.write(
                        f"    {spec.id:>4d}       {spec.hydtyp}         "
                        f"{spec.layer}                                  "
                        f"{node_id}     {name_part}\n"
                    )

        # v5.0: interbed discretization thickness
        if is_v50:
            _write_value(f, config.interbed_dz, "Interbed DZ")

        # Number of parametric grids
        _write_value(f, config.n_parametric_grids, "NGroup")

        # Conversion factors
        if config.conversion_factors:
            factors_str = "  ".join(f"{cf:>12.6f}" for cf in config.conversion_factors)
            _write_value(f, factors_str, "Conversion factors")

        # Write parameter data
        if config.n_parametric_grids == 0 and config.node_params:
            _write_subsidence_params(f, config, is_v50)
        elif config.n_parametric_grids > 0 and config.parametric_grids:
            _write_parametric_grids(f, config)

    return filepath


def _write_subsidence_params(f: TextIO, config: SubsidenceConfig, is_v50: bool) -> None:
    """Write direct subsidence parameter data for all nodes.

    For each node, writes n_layers rows. The first layer row includes
    the node ID; subsequent layers have parameters only.

    Args:
        f: Open file handle
        config: Subsidence configuration
        is_v50: Whether to write v5.0 format (with Kv and NEQ columns)
    """
    factors = config.conversion_factors

    for node_params in config.node_params:
        n_layers = len(node_params.elastic_sc)

        for layer_idx in range(n_layers):
            # Reverse conversion factors applied during reading
            elastic = node_params.elastic_sc[layer_idx] / (factors[1] if len(factors) > 1 else 1.0)
            inelastic = node_params.inelastic_sc[layer_idx] / (
                factors[2] if len(factors) > 2 else 1.0
            )
            thick = node_params.interbed_thick[layer_idx] / (
                factors[3] if len(factors) > 3 else 1.0
            )
            thick_min = node_params.interbed_thick_min[layer_idx] / (
                factors[4] if len(factors) > 4 else 1.0
            )
            precompact = node_params.precompact_head[layer_idx] / (
                factors[5] if len(factors) > 5 else 1.0
            )

            if layer_idx == 0:
                # First layer includes node ID
                line = f"     {node_params.node_id:>6d}"
            else:
                line = "           "

            line += (
                f"  {elastic:>12.6f}  {inelastic:>12.6f}"
                f"  {thick:>12.6f}  {thick_min:>12.6f}"
                f"  {precompact:>12.4f}"
            )

            if is_v50:
                kv = node_params.kv_sub[layer_idx] / (factors[6] if len(factors) > 6 else 1.0)
                neq = node_params.n_eq[layer_idx]
                line += f"  {kv:>12.6f}  {neq:>8.1f}"

            f.write(line + "\n")


def _write_parametric_grids(f: TextIO, config: SubsidenceConfig) -> None:
    """Write parametric grid sections for subsidence parameters.

    Each grid group writes: node range string, NDP, NEP,
    element definitions (if NEP > 0), and node data with
    continuation rows per layer.
    """
    factors = config.conversion_factors
    fx = factors[0] if len(factors) > 0 else 1.0
    param_factors = [factors[i] if i < len(factors) else 1.0 for i in range(1, 6)]

    for grid in config.parametric_grids:
        # Node range string
        f.write(f"   {grid.node_range_str}\n")

        # NDP, NEP
        _write_value(f, grid.n_nodes, "NDP")
        _write_value(f, grid.n_elements, "NEP")

        # Element definitions (if NEP > 0)
        for elem in grid.elements:
            parts = "  ".join(f"{v + 1:>6d}" for v in elem)
            f.write(f"     {parts}\n")

        # Node data: first line has ID X Y P1..P5, continuation lines have P1..P5
        n_layers = grid.node_values.shape[1] if grid.node_values.ndim == 3 else 1
        for i in range(grid.n_nodes):
            for layer_idx in range(n_layers):
                # Reverse conversion factors
                vals = []
                for p in range(5):
                    raw = grid.node_values[i, layer_idx, p] / param_factors[p]
                    vals.append(raw)

                if layer_idx == 0:
                    # First layer: node ID, coordinates, then params
                    px = grid.node_coords[i, 0] / fx if fx != 0.0 else grid.node_coords[i, 0]
                    py = grid.node_coords[i, 1] / fx if fx != 0.0 else grid.node_coords[i, 1]
                    line = f"    {i + 1:>4d}      {px:>10.1f}       {py:>10.1f}"
                    for v in vals:
                        line += f"         {v:.1f}"
                    f.write(line + "\n")
                else:
                    # Continuation line: params only
                    parts_str = "  ".join(f"{v:>12.1f}" for v in vals)
                    f.write(f"                                           {parts_str}\n")


def write_subsidence_ic(config: SubsidenceConfig, filepath: Path | str) -> Path:
    """Write the subsidence initial conditions file.

    Format:
        Line 1: Conversion factor
        Per node: ID InterbedThick_L1..Ln PreCompactHead_L1..Ln

    Args:
        config: Subsidence configuration with IC data
        filepath: Output file path

    Returns:
        Path to written file
    """
    filepath = Path(filepath)
    _ensure_parent_dir(filepath)

    with open(filepath, "w") as f:
        _write_comment(f, "IWFM Subsidence Initial Conditions")

        # Conversion factor
        _write_value(f, config.ic_factor, "IC conversion factor")

        if config.ic_interbed_thick is not None and config.ic_precompact_head is not None:
            n_nodes, n_layers = config.ic_interbed_thick.shape
            ic_factor = config.ic_factor if config.ic_factor != 0.0 else 1.0

            for i in range(n_nodes):
                node_id = i + 1  # 1-based
                parts = [f"{node_id:>6d}"]

                # Interbed thicknesses for all layers
                for layer in range(n_layers):
                    val = config.ic_interbed_thick[i, layer] / ic_factor
                    parts.append(f"{val:>12.4f}")

                # Pre-compaction heads for all layers
                for layer in range(n_layers):
                    val = config.ic_precompact_head[i, layer] / ic_factor
                    parts.append(f"{val:>12.4f}")

                f.write("     " + "  ".join(parts) + "\n")

    return filepath


def write_gw_subsidence(
    config: SubsidenceConfig,
    filepath: Path | str,
    ic_filepath: Path | str | None = None,
) -> Path:
    """Write IWFM GW subsidence parameter file and optional IC file.

    Convenience function that writes both the main parameter file and,
    if IC data is present and an IC filepath is provided, the initial
    conditions file.

    Args:
        config: Subsidence configuration
        filepath: Path to main subsidence parameter file
        ic_filepath: Optional path to IC file (writes if IC data present)

    Returns:
        Path to main written file
    """
    result = write_subsidence_main(config, filepath)

    if (
        ic_filepath is not None
        and config.ic_interbed_thick is not None
        and config.ic_precompact_head is not None
    ):
        write_subsidence_ic(config, ic_filepath)

    return result
