"""
IWFM-format Stream Bypass Specification Writer.

Writes the bypass specification file from a BypassSpecConfig,
including bypass routes (origin, destination, capacity), optional
inline rating tables, and seepage/recharge zone destinations.

The writer produces output that is compatible with the BypassSpecReader,
enabling roundtrip read-write fidelity.

Reference: Class_Bypass.f90 - Bypass_New()
"""

from __future__ import annotations

from pathlib import Path

from pyiwfm.io.iwfm_writer import (
    ensure_parent_dir as _ensure_parent_dir,
    write_comment as _write_comment,
    write_value as _write_value,
)
from pyiwfm.io.stream_bypass import BypassSpecConfig


def write_bypass_spec(config: BypassSpecConfig, filepath: Path | str) -> Path:
    """Write the bypass specification file.

    Writes the complete bypass spec file including the number of bypasses,
    conversion factors, bypass definitions with optional inline rating
    tables, and seepage/recharge zone data.

    Args:
        config: Bypass specification configuration
        filepath: Output file path

    Returns:
        Path to written file
    """
    filepath = Path(filepath)
    _ensure_parent_dir(filepath)

    with open(filepath, "w") as f:
        _write_comment(f, "IWFM Bypass Specification File")

        # NBypass
        _write_value(f, config.n_bypasses, "NBypass")

        if config.n_bypasses <= 0:
            return filepath

        # Flow conversion factor
        _write_value(f, config.flow_factor, "Flow conversion factor")

        # Stream flow time unit
        _write_value(f, config.flow_time_unit, "Flow time unit")

        # Bypass conversion factor
        _write_value(f, config.bypass_factor, "Bypass conversion factor")

        # Bypass time unit
        _write_value(f, config.bypass_time_unit, "Bypass time unit")

        # Bypass specifications
        for bypass in config.bypasses:
            name_part = f"  {bypass.name}" if bypass.name else ""
            f.write(
                f"     {bypass.id:>6d}  {bypass.export_stream_node:>6d}"
                f"  {bypass.dest_type:>3d}  {bypass.dest_id:>6d}"
                f"  {bypass.rating_table_col:>6d}"
                f"  {bypass.frac_recoverable:>10.6f}"
                f"  {bypass.frac_non_recoverable:>10.6f}"
                f"{name_part}\n"
            )

            # Write inline rating table if rating_table_col < 0
            if bypass.rating_table_col < 0 and bypass.inline_rating is not None:
                flow_factor = config.flow_factor if config.flow_factor != 0.0 else 1.0
                for i in range(len(bypass.inline_rating.flows)):
                    # Reverse the flow factor applied during reading
                    flow = bypass.inline_rating.flows[i] / flow_factor
                    frac = bypass.inline_rating.fractions[i]
                    f.write(f"     {flow:>15.4f}  {frac:>10.6f}\n")

        # Seepage/recharge zones -- one entry per bypass
        for sz in config.seepage_zones:
            if sz.n_elements > 0 and sz.element_ids:
                # Header line: ID NERELS first_IERELS first_FERELS
                f.write(
                    f"     {sz.bypass_id:>6d}  {sz.n_elements:>4d}"
                    f"  {sz.element_ids[0]:>6d}"
                    f"  {sz.element_fractions[0]:>10.6f}\n"
                )
                # Remaining elements
                for j in range(1, len(sz.element_ids)):
                    frac = sz.element_fractions[j] if j < len(sz.element_fractions) else 1.0
                    f.write(f"     {sz.element_ids[j]:>6d}  {frac:>10.6f}\n")
            else:
                # No recharge zones: ID 0 0 0
                f.write(f"     {sz.bypass_id:>6d}  {0:>4d}  {0:>6d}  {0.0:>10.6f}\n")

    return filepath
