"""
IWFM-format Pumping Writer.

Writes the 3-tier pumping file system from a PumpingConfig:
- Main pumping file (dispatcher)
- Well specification file
- Element pumping specification file
"""

from __future__ import annotations

from pathlib import Path

from pyiwfm.io.gw_pumping import PumpingConfig
from pyiwfm.io.iwfm_writer import (
    ensure_parent_dir as _ensure_parent_dir,
    write_comment as _write_comment,
    write_value as _write_value,
)


def write_pumping_main(config: PumpingConfig, filepath: Path | str) -> Path:
    """Write the pumping main file (dispatcher).

    Args:
        config: Pumping configuration
        filepath: Output file path

    Returns:
        Path to written file
    """
    filepath = Path(filepath)
    _ensure_parent_dir(filepath)

    with open(filepath, "w") as f:
        _write_comment(f, "IWFM Pumping Main File")
        if config.version:
            f.write(f"#{config.version}\n")

        _write_value(f, str(config.well_file or ""), "Well specification file")
        _write_value(f, str(config.elem_pump_file or ""), "Element pumping file")
        _write_value(f, str(config.ts_data_file or ""), "Time series data file")
        _write_value(f, str(config.output_file or ""), "Output file")

    return filepath


def write_well_spec_file(
    config: PumpingConfig, filepath: Path | str
) -> Path:
    """Write the well specification file.

    Args:
        config: Pumping configuration with well specs
        filepath: Output file path

    Returns:
        Path to written file
    """
    filepath = Path(filepath)
    _ensure_parent_dir(filepath)

    with open(filepath, "w") as f:
        _write_comment(f, "IWFM Well Specification File")
        _write_value(f, len(config.well_specs), "NWELL")
        _write_value(f, config.factor_xy, "FACTXY")
        _write_value(f, config.factor_radius, "FACTR")
        _write_value(f, config.factor_length, "FACTLT")

        # Well structural data (raw values — divide by factor to recover file values)
        for ws in config.well_specs:
            x = ws.x / config.factor_xy if config.factor_xy else ws.x
            y = ws.y / config.factor_xy if config.factor_xy else ws.y
            # Radius was stored as diameter/2 * factor, write back as diameter
            diameter = (ws.radius * 2.0 / config.factor_radius
                        if config.factor_radius else ws.radius * 2.0)
            pt = ws.perf_top / config.factor_length if config.factor_length else ws.perf_top
            pb = ws.perf_bottom / config.factor_length if config.factor_length else ws.perf_bottom
            name_part = f"  / {ws.name}" if ws.name else ""
            f.write(
                f"     {ws.id:>6d}  {x:>15.4f}  {y:>15.4f}  "
                f"{diameter:>10.4f}  {pt:>12.4f}  {pb:>12.4f}{name_part}\n"
            )

        # Pumping specifications
        for wps in config.well_pumping_specs:
            f.write(
                f"     {wps.well_id:>6d}  {wps.pump_column:>4d}  "
                f"{wps.pump_fraction:>8.4f}  {wps.dist_method:>3d}  "
                f"{wps.dest_type:>3d}  {wps.dest_id:>6d}  "
                f"{wps.irig_frac_column:>4d}  {wps.adjust_column:>4d}  "
                f"{wps.pump_max_column:>4d}  {wps.pump_max_fraction:>8.4f}\n"
            )

        # Element groups
        _write_value(f, len(config.well_groups), "NGROUPS")
        for grp in config.well_groups:
            for i, elem_id in enumerate(grp.elements):
                if i == 0:
                    f.write(f"     {grp.id:>6d}  {len(grp.elements):>4d}  {elem_id:>6d}\n")
                else:
                    f.write(f"     {elem_id:>6d}\n")

    return filepath


def write_elem_pump_file(
    config: PumpingConfig, filepath: Path | str, n_layers: int = 1
) -> Path:
    """Write the element pumping specification file.

    Args:
        config: Pumping configuration with element pump specs
        filepath: Output file path
        n_layers: Number of aquifer layers

    Returns:
        Path to written file
    """
    filepath = Path(filepath)
    _ensure_parent_dir(filepath)

    with open(filepath, "w") as f:
        _write_comment(f, "IWFM Element Pumping Specification File")
        _write_value(f, len(config.elem_pumping_specs), "NSINK")

        for eps in config.elem_pumping_specs:
            layer_str = "  ".join(f"{lf:>8.4f}" for lf in eps.layer_factors)
            f.write(
                f"     {eps.element_id:>6d}  {eps.pump_column:>4d}  "
                f"{eps.pump_fraction:>8.4f}  {eps.dist_method:>3d}  "
                f"{layer_str}  "
                f"{eps.dest_type:>3d}  {eps.dest_id:>6d}  "
                f"{eps.irig_frac_column:>4d}  {eps.adjust_column:>4d}  "
                f"{eps.pump_max_column:>4d}  {eps.pump_max_fraction:>8.4f}\n"
            )

        # Element groups
        _write_value(f, len(config.elem_groups), "NGROUPS")
        for grp in config.elem_groups:
            for i, elem_id in enumerate(grp.elements):
                if i == 0:
                    f.write(f"     {grp.id:>6d}  {len(grp.elements):>4d}  {elem_id:>6d}\n")
                else:
                    f.write(f"     {elem_id:>6d}\n")

    return filepath
