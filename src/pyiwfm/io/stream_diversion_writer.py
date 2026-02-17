"""
IWFM-format Stream Diversion Specification Writer.

Writes the diversion specification file from a DiversionSpecConfig,
including diversion specs with delivery destinations, element groups,
recharge zone destinations, and optional spill zone destinations.

Supports both 14-column (legacy, without spill fields) and 16-column
(with spill column and fraction) formats, controlled by the
``has_spills`` flag in the config.

The writer produces output that is compatible with the DiversionSpecReader,
enabling roundtrip read-write fidelity.

Reference: Class_Diversion.f90 - Diversion_New()
"""

from __future__ import annotations

from pathlib import Path
from typing import TextIO

from pyiwfm.io.iwfm_writer import (
    ensure_parent_dir as _ensure_parent_dir,
)
from pyiwfm.io.iwfm_writer import (
    write_comment as _write_comment,
)
from pyiwfm.io.iwfm_writer import (
    write_value as _write_value,
)
from pyiwfm.io.stream_diversion import DiversionSpecConfig


def write_diversion_spec(config: DiversionSpecConfig, filepath: Path | str) -> Path:
    """Write the diversion specification file.

    Writes the complete diversion spec file including the number of
    diversions, diversion definitions, element groups, recharge zone
    destinations, and (if applicable) spill zone destinations.

    Args:
        config: Diversion specification configuration
        filepath: Output file path

    Returns:
        Path to written file
    """
    filepath = Path(filepath)
    _ensure_parent_dir(filepath)

    with open(filepath, "w") as f:
        _write_comment(f, "IWFM Diversion Specification File")

        # NDiver
        _write_value(f, config.n_diversions, "NDiver")

        if config.n_diversions <= 0:
            return filepath

        # Diversion specifications
        for div in config.diversions:
            _write_diversion_line(f, div, config.has_spills)

        # Element groups
        _write_value(f, config.n_element_groups, "NGroups")

        for group in config.element_groups:
            _write_element_group(f, group)

        # Recharge zones -- one entry per diversion
        for rz in config.recharge_zones:
            _write_recharge_zone(f, rz)

        # Spill zones (only if 16-column format) -- one entry per diversion
        if config.has_spills:
            for sz in config.spill_zones:
                _write_recharge_zone(f, sz)

    return filepath


def _write_diversion_line(
    f: TextIO,
    div: object,
    has_spills: bool,
) -> None:
    """Write a single diversion specification line.

    Args:
        f: Open file handle
        div: DiversionSpec instance
        has_spills: Whether to write 16-column format with spill fields
    """
    from pyiwfm.io.stream_diversion import DiversionSpec

    assert isinstance(div, DiversionSpec)

    parts = [
        f"{div.id:>6d}",
        f"{div.stream_node:>6d}",
        f"{div.max_diver_col:>6d}",
        f"{div.frac_max_diver:>10.4f}",
        f"{div.recv_loss_col:>6d}",
        f"{div.frac_recv_loss:>10.4f}",
        f"{div.non_recv_loss_col:>6d}",
        f"{div.frac_non_recv_loss:>10.4f}",
    ]

    if has_spills:
        parts.extend(
            [
                f"{div.spill_col:>6d}",
                f"{div.frac_spill:>10.4f}",
            ]
        )

    parts.extend(
        [
            f"{div.dest_type:>3d}",
            f"{div.dest_id:>6d}",
            f"{div.delivery_col:>6d}",
            f"{div.frac_delivery:>10.4f}",
            f"{div.irrig_frac_col:>6d}",
            f"{div.adjustment_col:>6d}",
        ]
    )

    if div.name:
        parts.append(f"  {div.name}")

    f.write("     " + "  ".join(parts) + "\n")


def _write_element_group(f: TextIO, group: object) -> None:
    """Write a single element group.

    Format:
        GroupID  NElements  FirstElementID
        ElementID  (one per subsequent line)

    Args:
        f: Open file handle
        group: ElementGroup instance
    """
    from pyiwfm.io.stream_diversion import ElementGroup

    assert isinstance(group, ElementGroup)

    n_elements = len(group.elements)

    if n_elements > 0:
        # Header line with first element
        f.write(f"     {group.id:>6d}  {n_elements:>4d}  {group.elements[0]:>6d}\n")
        # Remaining elements
        for elem_id in group.elements[1:]:
            f.write(f"     {elem_id:>6d}\n")
    else:
        f.write(f"     {group.id:>6d}  {0:>4d}\n")


def _write_recharge_zone(f: TextIO, rz: object) -> None:
    """Write recharge zone destination data.

    Format (LossDestination_New in Fortran):
        ID  NERELS  first_IERELS  first_FERELS
        IERELS  FERELS  (for remaining elements)

    When NERELS=0:
        ID  0  0  0

    Args:
        f: Open file handle
        rz: RechargeZoneDest instance
    """
    from pyiwfm.io.stream_diversion import RechargeZoneDest

    assert isinstance(rz, RechargeZoneDest)

    if rz.n_zones > 0 and rz.zone_ids:
        # Header line with first zone
        f.write(
            f"     {rz.diversion_id:>6d}  {rz.n_zones:>4d}"
            f"  {rz.zone_ids[0]:>6d}"
            f"  {rz.zone_fractions[0]:>10.6f}\n"
        )
        # Remaining zones
        for j in range(1, len(rz.zone_ids)):
            frac = rz.zone_fractions[j] if j < len(rz.zone_fractions) else 1.0
            f.write(f"     {rz.zone_ids[j]:>6d}  {frac:>10.6f}\n")
    else:
        # No recharge zones
        f.write(f"     {rz.diversion_id:>6d}  {0:>4d}  {0:>6d}  {0.0:>10.6f}\n")
