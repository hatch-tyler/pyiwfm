"""
Reader for IWFM hydrograph output text files (.out).

Parses both GW and stream hydrograph output files which share the same
IWFM text format with header metadata and whitespace-delimited time series.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from pyiwfm.io.budget import parse_iwfm_datetime as _parse_iwfm_dt

logger = logging.getLogger(__name__)


class IWFMHydrographReader:
    """Reader for IWFM hydrograph output text files.

    Parses the header for column metadata (hydrograph IDs, layers,
    node/element IDs) and reads the full time series data into a
    NumPy array for efficient column extraction.

    Parameters
    ----------
    filepath : Path or str
        Path to the IWFM hydrograph output file (.out).
    """

    def __init__(self, filepath: Path | str) -> None:
        self._filepath = Path(filepath)
        self._hydrograph_ids: list[int] = []
        self._layers: list[int] = []
        self._node_ids: list[int] = []
        self._times: list[str] = []
        self._data: NDArray[np.float64] = np.empty((0, 0))
        self._parsed = False
        self._parse()

    def _parse(self) -> None:
        """Parse header and data from the hydrograph output file."""
        if not self._filepath.exists():
            logger.warning("Hydrograph file not found: %s", self._filepath)
            return

        try:
            with open(self._filepath) as f:
                lines = f.readlines()
        except Exception as e:
            logger.error("Failed to read hydrograph file %s: %s", self._filepath, e)
            return

        header_lines: list[str] = []
        data_lines: list[str] = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("*"):
                header_lines.append(stripped)
            else:
                data_lines.append(stripped)

        # Parse header: look for HYDROGRAPH ID, LAYER, NODE/ELEMENT lines
        for hline in header_lines:
            content = hline.lstrip("*").strip()
            if content.upper().startswith("HYDROGRAPH ID"):
                # Format: "HYDROGRAPH ID    1    2    3  ...  N"
                parts = content.split()
                # Skip "HYDROGRAPH" and "ID", rest are IDs
                self._hydrograph_ids = [int(x) for x in parts[2:]]
            elif content.upper().startswith("LAYER"):
                parts = content.split()
                self._layers = [int(x) for x in parts[1:]]
            elif content.upper().startswith("NODE") or content.upper().startswith("ELEMENT"):
                parts = content.split()
                # Skip the label word(s), parse remaining integers
                int_vals: list[int] = []
                for p in parts[1:]:
                    try:
                        int_vals.append(int(p))
                    except ValueError:
                        continue
                self._node_ids = int_vals

        if not data_lines:
            logger.warning("No data found in hydrograph file: %s", self._filepath)
            return

        # Parse data lines: "MM/DD/YYYY_HH:MM  val1  val2  ..."
        times: list[str] = []
        rows: list[list[float]] = []

        for dline in data_lines:
            parts = dline.split()
            if len(parts) < 2:
                continue

            time_str = parts[0]
            dt = _parse_iwfm_dt(time_str)
            if dt is None:
                continue

            times.append(dt.isoformat())
            vals = []
            for v in parts[1:]:
                try:
                    vals.append(float(v))
                except ValueError:
                    vals.append(float("nan"))
            rows.append(vals)

        if rows:
            # Ensure all rows have same length (pad with NaN if needed)
            n_cols = max(len(r) for r in rows)
            for r in rows:
                while len(r) < n_cols:
                    r.append(float("nan"))
            self._data = np.array(rows, dtype=np.float64)
            self._times = times
            self._parsed = True

            logger.info(
                "Hydrograph file loaded: %d timesteps, %d columns from %s",
                len(times),
                n_cols,
                self._filepath.name,
            )

    @property
    def n_columns(self) -> int:
        """Number of data columns."""
        return self._data.shape[1] if self._parsed else 0

    @property
    def n_timesteps(self) -> int:
        """Number of timesteps."""
        return self._data.shape[0] if self._parsed else 0

    @property
    def times(self) -> list[str]:
        """ISO 8601 datetime strings for each timestep."""
        return self._times

    @property
    def hydrograph_ids(self) -> list[int]:
        """1-based hydrograph IDs from header."""
        return self._hydrograph_ids

    @property
    def layers(self) -> list[int]:
        """Layer numbers per column (GW only)."""
        return self._layers

    @property
    def node_ids(self) -> list[int]:
        """Node/element IDs per column."""
        return self._node_ids

    def get_time_series(self, column_index: int) -> tuple[list[str], list[float]]:
        """Get time series for a specific column.

        Parameters
        ----------
        column_index : int
            0-based column index.

        Returns
        -------
        tuple[list[str], list[float]]
            (times, values) where times are ISO 8601 strings.
        """
        if not self._parsed or column_index < 0 or column_index >= self.n_columns:
            return [], []
        return self._times, self._data[:, column_index].tolist()

    def find_column_by_node_id(self, node_id: int) -> int | None:
        """Find column index for a given node/element ID.

        Parameters
        ----------
        node_id : int
            Node or element ID to search for.

        Returns
        -------
        int or None
            0-based column index, or None if not found.
        """
        if node_id in self._node_ids:
            return self._node_ids.index(node_id)
        return None
