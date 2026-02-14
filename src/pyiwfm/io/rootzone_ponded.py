"""
Ponded agricultural crop sub-file reader for IWFM RootZone component.

This module reads the IWFM ponded agricultural crop file (PFL), which
is referenced by the RootZone component main file.  In IWFM v5.0 the
ponded crop file uses the same ``Class_AgLandUse_v50`` format as the
non-ponded crop file, so this module re-exports the non-ponded reader
with ponded-specific type aliases.

Typical ponded crops in C2VSimFG:
    - Rice (flood-decomposed, non-flood-decomposed, no-decomposition)
    - Refuge (seasonal, permanent)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pyiwfm.io.rootzone_nonponded import (
    NonPondedCropConfig,
    NonPondedCropReader,
    CurveNumberRow,
    EtcPointerRow,
    IrrigationPointerRow,
    SoilMoisturePointerRow,
    SupplyReturnReuseRow,
    InitialConditionRow,
)


# ── Ponded-specific aliases ───────────────────────────────────────────
# The binary file format is identical to the non-ponded crop format.
# We provide ponded-specific names for clarity.

PondedCropConfig = NonPondedCropConfig
"""Alias for :class:`NonPondedCropConfig` — same file format."""


class PondedCropReader(NonPondedCropReader):
    """Reader for IWFM ponded agricultural crop sub-file.

    Inherits from :class:`NonPondedCropReader` since IWFM v5.0 uses
    the same ``Class_AgLandUse_v50`` format for both ponded and
    non-ponded crop files.
    """

    pass


# ── convenience function ──────────────────────────────────────────────


def read_ponded_crop(
    filepath: Path | str,
    base_dir: Path | None = None,
    n_subregions: int | None = None,
) -> PondedCropConfig:
    """Read an IWFM ponded agricultural crop sub-file.

    Args:
        filepath: Path to the ponded crop file.
        base_dir: Base directory for resolving relative paths.
        n_subregions: Number of subregions (for exact row counts).

    Returns:
        :class:`PondedCropConfig` with parsed values.
    """
    reader = PondedCropReader(n_subregions=n_subregions)
    return reader.read(filepath, base_dir)
