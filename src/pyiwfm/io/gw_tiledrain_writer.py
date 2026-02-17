"""
IWFM-format Tile Drain Writer.

Writes the tile drain/sub-irrigation file from a TileDrainConfig,
including version header, tile drains section, and sub-irrigation section.
"""

from __future__ import annotations

from pathlib import Path

from pyiwfm.io.gw_tiledrain import TileDrainConfig
from pyiwfm.io.iwfm_writer import (
    ensure_parent_dir as _ensure_parent_dir,
)
from pyiwfm.io.iwfm_writer import (
    write_comment as _write_comment,
)
from pyiwfm.io.iwfm_writer import (
    write_value as _write_value,
)


def write_tile_drain_file(config: TileDrainConfig, filepath: Path | str) -> Path:
    """Write an IWFM-format tile drain/sub-irrigation file.

    Args:
        config: Tile drain configuration
        filepath: Output file path

    Returns:
        Path to written file
    """
    filepath = Path(filepath)
    _ensure_parent_dir(filepath)

    with open(filepath, "w") as f:
        _write_comment(f, "IWFM Tile Drain / Sub-Irrigation File")

        # Version header
        if config.version:
            f.write(f"#{config.version}\n")

        # --- Tile Drains Section ---
        _write_value(f, config.n_drains, "NDrain")
        if config.n_drains > 0:
            _write_value(f, config.drain_height_factor, "FACTHD")
            _write_value(f, config.drain_conductance_factor, "FACTCDC")
            _write_value(f, config.drain_time_unit, "TUNITDR")

            for td in config.tile_drains:
                # Values stored as already-converted; divide by factor to get raw
                elev = td.elevation
                cond = td.conductance
                if config.drain_height_factor not in (0.0, 1.0):
                    elev = elev / config.drain_height_factor
                if config.drain_conductance_factor not in (0.0, 1.0):
                    cond = cond / config.drain_conductance_factor
                f.write(
                    f"     {td.id:>6d}  {td.gw_node:>6d}  "
                    f"{elev:>12.4f}  {cond:>12.6f}  "
                    f"{td.dest_type:>3d}  {td.dest_id:>6d}\n"
                )

        # --- Sub-Irrigation Section ---
        _write_value(f, config.n_sub_irrigation, "NSubIrig")
        if config.n_sub_irrigation > 0:
            _write_value(f, config.subirig_height_factor, "FACTHSI")
            _write_value(f, config.subirig_conductance_factor, "FACTCSI")
            _write_value(f, config.subirig_time_unit, "TUNITSI")

            for si in config.sub_irrigations:
                elev = si.elevation
                cond = si.conductance
                if config.subirig_height_factor not in (0.0, 1.0):
                    elev = elev / config.subirig_height_factor
                if config.subirig_conductance_factor not in (0.0, 1.0):
                    cond = cond / config.subirig_conductance_factor
                f.write(f"     {si.id:>6d}  {si.gw_node:>6d}  {elev:>12.4f}  {cond:>12.6f}\n")

    return filepath
