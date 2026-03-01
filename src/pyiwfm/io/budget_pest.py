"""Budget HDF5 to PEST text export.

Reads IWFM budget data via :class:`~pyiwfm.io.budget.BudgetReader` and
exports observation-ready text files for PEST++ calibration, along with
companion ``.ins`` instruction files.

Functions
---------
- :func:`budget_to_pest_text` — Write budget values as PEST observation text.
- :func:`budget_to_pest_instruction` — Generate a PEST instruction file.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pyiwfm.io.budget import BudgetReader


def budget_to_pest_text(
    budget_path: str | Path,
    output_path: str | Path,
    prefix: str = "gwb",
    locations: list[str | int] | None = None,
    columns: list[str] | None = None,
    *,
    volume_factor: float = 1.0,
) -> Path:
    """Export budget data as whitespace-delimited PEST observation text.

    Each row represents one observation with an auto-generated PEST name
    in the format ``{prefix}{loc_idx:02d}_{timestep:06d}``.

    Parameters
    ----------
    budget_path : str or Path
        Path to an IWFM budget HDF5 (or binary) file.
    output_path : str or Path
        Destination text file.
    prefix : str, default "gwb"
        Prefix for PEST observation names.
    locations : list[str | int] | None
        Subset of locations (names or 0-based indices) to export.
        Exports all locations when ``None``.
    columns : list[str] | None
        Subset of budget columns to export.  Exports all when ``None``.
    volume_factor : float, default 1.0
        Multiplier applied to all values (unit conversion).

    Returns
    -------
    Path
        Path to the written text file.
    """
    budget_path = Path(budget_path)
    output_path = Path(output_path)

    reader = BudgetReader(budget_path)
    loc_names = reader.locations

    if locations is None:
        loc_indices = list(range(len(loc_names)))
    else:
        loc_indices = []
        for loc in locations:
            if isinstance(loc, int):
                loc_indices.append(loc)
            else:
                loc_indices.append(loc_names.index(loc))

    rows: list[str] = []

    for loc_idx in loc_indices:
        df: pd.DataFrame = reader.get_dataframe(location=loc_idx, volume_factor=volume_factor)

        if columns is not None:
            available = [c for c in columns if c in df.columns]
            if not available:
                continue
            df = df[available]

        for ts_idx, (_, row) in enumerate(df.iterrows()):
            for col in df.columns:
                pest_name = f"{prefix}{loc_idx:02d}_{ts_idx:06d}"
                value = float(row[col])
                rows.append(
                    f"{pest_name:<20s} {loc_names[loc_idx]:<20s} {col:<30s} {ts_idx:>8d} {value:>20.8e}"
                )

    output_path.write_text("\n".join(rows) + "\n" if rows else "")
    return output_path


def budget_to_pest_instruction(
    budget_path: str | Path,
    output_path: str | Path,
    prefix: str = "gwb",
    locations: list[str | int] | None = None,
    columns: list[str] | None = None,
) -> Path:
    """Generate a PEST instruction (``.ins``) file for budget observations.

    The instruction file mirrors the structure produced by
    :func:`budget_to_pest_text` so that PEST++ can read the
    corresponding model output.

    Parameters
    ----------
    budget_path : str or Path
        Path to an IWFM budget HDF5 (or binary) file (used to determine
        the number of locations, timesteps, and columns).
    output_path : str or Path
        Destination ``.ins`` file path.
    prefix : str, default "gwb"
        Prefix for PEST observation names (must match the text export).
    locations : list[str | int] | None
        Subset of locations.
    columns : list[str] | None
        Subset of budget columns.

    Returns
    -------
    Path
        Path to the written instruction file.
    """
    budget_path = Path(budget_path)
    output_path = Path(output_path)

    reader = BudgetReader(budget_path)
    loc_names = reader.locations

    if locations is None:
        loc_indices = list(range(len(loc_names)))
    else:
        loc_indices = []
        for loc in locations:
            if isinstance(loc, int):
                loc_indices.append(loc)
            else:
                loc_indices.append(loc_names.index(loc))

    lines: list[str] = ["pif @"]

    for loc_idx in loc_indices:
        df: pd.DataFrame = reader.get_dataframe(location=loc_idx)

        if columns is not None:
            available = [c for c in columns if c in df.columns]
            if not available:
                continue
            df = df[available]

        for ts_idx in range(len(df)):
            pest_name = f"{prefix}{loc_idx:02d}_{ts_idx:06d}"
            lines.append(f"l1 !{pest_name}!")

    output_path.write_text("\n".join(lines) + "\n")
    return output_path
