"""
Budget control file parser for IWFM post-processing.

Reads a ``.bud`` / ``.in`` budget control file that specifies which HDF5
budget output files to process, unit conversion factors, time windows,
and which locations to export.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pyiwfm.io.iwfm_reader import next_data_value as _next_data_value
from pyiwfm.io.iwfm_reader import resolve_path as _resolve_path


@dataclass
class BudgetOutputSpec:
    """Specification for a single budget output in the control file."""

    hdf_file: Path
    """HDF5 budget input file."""

    output_file: Path
    """Output file path (.xlsx or .txt)."""

    output_interval: str | None = None
    """Print interval (e.g. ``"1MON"``).  ``None`` = same as data."""

    location_ids: list[int] = field(default_factory=list)
    """1-based location IDs.  ``[-1]`` = all, ``[]`` = none."""


@dataclass
class BudgetControlConfig:
    """Parsed budget control file."""

    length_factor: float = 1.0
    """FACTLTOU — length unit conversion factor."""

    length_unit: str = "FEET"
    """UNITLTOU — output length unit string."""

    area_factor: float = 1.0
    """FACTAROU — area unit conversion factor."""

    area_unit: str = "AC"
    """UNITAROU — output area unit string."""

    volume_factor: float = 1.0
    """FACTVLOU — volume unit conversion factor."""

    volume_unit: str = "AC.FT."
    """UNITVLOU — output volume unit string."""

    cache_size: int = 100
    """NTIME — cache/block size for reading."""

    begin_date: str | None = None
    """Begin date (``MM/DD/YYYY_HH:MM``) or ``None`` for start-of-data."""

    end_date: str | None = None
    """End date (``MM/DD/YYYY_HH:MM``) or ``None`` for end-of-data."""

    budgets: list[BudgetOutputSpec] = field(default_factory=list)
    """List of budget output specifications."""


def read_budget_control(filepath: Path | str) -> BudgetControlConfig:
    """Parse an IWFM budget control file.

    Parameters
    ----------
    filepath : Path or str
        Path to the budget control / input file.

    Returns
    -------
    BudgetControlConfig
        Parsed configuration.
    """
    filepath = Path(filepath)
    base_dir = filepath.parent
    config = BudgetControlConfig()

    with open(filepath, encoding="ascii", errors="replace") as f:
        # FACTLTOU
        config.length_factor = float(_next_data_value(f))

        # UNITLTOU
        config.length_unit = _next_data_value(f)

        # FACTAROU
        config.area_factor = float(_next_data_value(f))

        # UNITAROU
        config.area_unit = _next_data_value(f)

        # FACTVLOU
        config.volume_factor = float(_next_data_value(f))

        # UNITVLOU
        config.volume_unit = _next_data_value(f)

        # NTIME (cache size)
        config.cache_size = int(_next_data_value(f))

        # NBUDGET — number of budget file groups
        n_budget = int(_next_data_value(f))

        # Begin/end dates
        begin_raw = _next_data_value(f)
        config.begin_date = None if begin_raw.strip() in ("*", "") else begin_raw.strip()

        end_raw = _next_data_value(f)
        config.end_date = None if end_raw.strip() in ("*", "") else end_raw.strip()

        # Read each budget specification
        for _ in range(n_budget):
            spec = BudgetOutputSpec(hdf_file=Path(), output_file=Path())

            # HDF input file
            hdf_raw = _next_data_value(f)
            spec.hdf_file = Path(_resolve_path(base_dir, hdf_raw))

            # Output file
            out_raw = _next_data_value(f)
            spec.output_file = Path(_resolve_path(base_dir, out_raw))

            # Output print interval
            interval_raw = _next_data_value(f)
            spec.output_interval = (
                None if interval_raw.strip() in ("*", "") else interval_raw.strip()
            )

            # NLPRNT — number of locations to print
            nlprnt = int(_next_data_value(f))

            if nlprnt < 0:
                # -1 means all locations
                spec.location_ids = [-1]
            elif nlprnt == 0:
                spec.location_ids = []
            else:
                ids: list[int] = []
                for _ in range(nlprnt):
                    ids.append(int(_next_data_value(f)))
                spec.location_ids = ids

            config.budgets.append(spec)

    return config
