"""
Converter for IWFM GWALLOUTFL text output files to HDF5.

Converts the fixed-width text file produced by IWFM's ``AllHeadOutFile_PrintResults``
into an HDF5 file compatible with ``LazyHeadDataLoader``.

Text file format (from GWHydrograph.f90 Fortran format specs):
- 4 title lines (decorative box with unit info)
- 2 header lines (``* NODE`` row + ``* TIME  node1 node2 ...`` row)
- Data rows: 21-char timestamp + NNodes values in ``(2X,F10.4)`` format
- Multi-layer: NLayers consecutive rows per timestep (continuation rows
  start with 21 spaces instead of a timestamp)

The converter streams the file line-by-line, writing each timestep directly
to a pre-allocated HDF5 dataset.  Memory usage is O(n_nodes * n_layers)
regardless of the number of timesteps.

Usage::

    python -m pyiwfm.io.head_all_converter C2VSimFG_GW_HeadAll.out --layers 1
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Width of each numeric column: 2 spaces + F10.4 = 12 chars
_COL_WIDTH = 12
# Width of the timestamp field
_TIME_WIDTH = 21
# How many timesteps to grow the HDF5 dataset by when it fills up
_CHUNK_GROW = 256


def _parse_timestamp(text: str) -> datetime:
    """Parse an IWFM timestamp like ``MM/DD/YYYY_HH:MM`` or ``MM/DD/YYYY_24:00``.

    The ``_24:00`` convention means midnight at the *end* of the day,
    which we map to 00:00 of the next day.
    """
    text = text.strip().replace("_", " ")
    if "24:00" in text:
        text = text.replace("24:00", "00:00")
        return datetime.strptime(text, "%m/%d/%Y %H:%M") + timedelta(days=1)
    return datetime.strptime(text, "%m/%d/%Y %H:%M")


def _parse_node_ids(header_line: str) -> list[int]:
    """Extract node IDs from the header line.

    The header line format is: ``*    TIME    node1  node2  ...``
    where each node ID occupies a 12-char field after the 21-char prefix.
    """
    remainder = header_line[_TIME_WIDTH:]
    ids: list[int] = []
    for i in range(0, len(remainder), _COL_WIDTH):
        chunk = remainder[i : i + _COL_WIDTH].strip()
        if chunk:
            try:
                ids.append(int(chunk))
            except ValueError:
                pass
    return ids


def _parse_data_line_numpy(line: str, n_nodes: int) -> np.ndarray:
    """Parse fixed-width numeric values from a data line into a numpy array.

    Uses numpy for vectorised float conversion instead of per-element
    Python ``float()`` calls.
    """
    data_part = line[_TIME_WIDTH:]
    # Split by whitespace and convert in one shot
    parts = data_part.split()
    if len(parts) >= n_nodes:
        return np.array(parts[:n_nodes], dtype=np.float64)
    # Fall back to fixed-width slicing if split gives wrong count
    # (e.g. negative numbers abutting the previous field)
    values = np.empty(n_nodes, dtype=np.float64)
    for i in range(n_nodes):
        start = i * _COL_WIDTH
        end = start + _COL_WIDTH
        chunk = data_part[start:end].strip()
        values[i] = float(chunk) if chunk else np.nan
    return values


def _count_data_lines(fh, header_lines: int) -> int:
    """Cheaply count data lines by scanning newlines without storing content."""
    fh.seek(0)
    total = 0
    for _ in fh:
        total += 1
    fh.seek(0)
    return max(total - header_lines, 0)


def convert_headall_to_hdf(
    text_file: str | Path,
    hdf_file: str | Path | None = None,
    n_layers: int = 1,
) -> Path:
    """Convert an IWFM GWALLOUTFL text file to HDF5.

    Streams the text file line-by-line so memory usage stays proportional
    to a single timestep (n_nodes * n_layers) rather than the whole file.

    Parameters
    ----------
    text_file : str or Path
        Path to the IWFM text output file (e.g. ``C2VSimFG_GW_HeadAll.out``).
    hdf_file : str or Path or None
        Output HDF5 path.  If *None*, uses ``text_file`` with ``.hdf`` extension.
    n_layers : int
        Number of groundwater layers.  Default is 1.

    Returns
    -------
    Path
        Path to the created HDF5 file.
    """
    try:
        import h5py
    except ImportError:
        raise ImportError(
            "h5py is required for HDF5 conversion. Install with: pip install h5py"
        )

    text_path = Path(text_file)
    if hdf_file is None:
        hdf_path = text_path.with_suffix(".hdf")
    else:
        hdf_path = Path(hdf_file)

    logger.info("Converting %s -> %s (n_layers=%d)", text_path, hdf_path, n_layers)

    with open(text_path, "r") as fh:
        # --- Read header (6 lines: 4 title + NODE + TIME) ---
        header_lines_read = 0
        title_count = 0
        while title_count < 4:
            line = fh.readline()
            if not line:
                raise ValueError("Unexpected end of file while reading title lines")
            header_lines_read += 1
            if line.startswith("*"):
                title_count += 1

        header1 = fh.readline()  # "* NODE ..."
        header2 = fh.readline()  # "* TIME  node1  node2 ..."
        header_lines_read += 2

        if not header1 or not header2:
            raise ValueError("Unexpected end of file while reading header lines")

        node_ids = _parse_node_ids(header2.rstrip())
        n_nodes = len(node_ids)
        if n_nodes == 0:
            raise ValueError("Could not parse any node IDs from header line")
        logger.info("Detected %d nodes from header", n_nodes)

        # --- Estimate timestep count from remaining data lines ---
        data_lines = _count_data_lines(fh, header_lines_read)
        estimated_timesteps = max(data_lines // n_layers, 1)
        logger.info(
            "Estimated %d timesteps from %d data lines", estimated_timesteps, data_lines
        )

        # Skip header again after the count pass
        fh.seek(0)
        for _ in range(header_lines_read):
            fh.readline()

        # --- Create HDF5 with resizable dataset ---
        timestamps: list[str] = []
        t_idx = 0

        with h5py.File(hdf_path, "w") as hf:
            ds = hf.create_dataset(
                "head",
                shape=(estimated_timesteps, n_nodes, n_layers),
                maxshape=(None, n_nodes, n_layers),
                dtype=np.float64,
                compression="gzip",
                compression_opts=4,
                chunks=(1, n_nodes, n_layers),
            )

            # --- Stream data lines ---
            row_buf = np.empty((n_nodes, n_layers), dtype=np.float64)

            while True:
                line = fh.readline()
                if not line:
                    break  # EOF

                line = line.rstrip()
                if not line or line.startswith("*"):
                    continue

                # First row of a timestep group has the timestamp
                ts_text = line[:_TIME_WIDTH].strip()
                if not ts_text:
                    continue

                timestamp = _parse_timestamp(ts_text)
                timestamps.append(timestamp.isoformat())

                # Parse n_layers rows for this timestep
                row_buf[:, 0] = _parse_data_line_numpy(line, n_nodes)
                for layer_idx in range(1, n_layers):
                    cont_line = fh.readline()
                    if not cont_line:
                        break
                    row_buf[:, layer_idx] = _parse_data_line_numpy(
                        cont_line.rstrip(), n_nodes
                    )

                # Grow dataset if needed
                if t_idx >= ds.shape[0]:
                    ds.resize(ds.shape[0] + _CHUNK_GROW, axis=0)

                # Write directly to HDF5
                ds[t_idx, :, :] = row_buf
                t_idx += 1

                # Progress reporting
                if t_idx % 100 == 0:
                    logger.info("  %d timesteps written...", t_idx)

            # Trim dataset to actual size
            if t_idx < ds.shape[0]:
                ds.resize(t_idx, axis=0)

            # Write time strings
            str_dt = h5py.string_dtype(encoding="utf-8")
            hf.create_dataset("times", data=timestamps, dtype=str_dt)
            hf.attrs["n_nodes"] = n_nodes
            hf.attrs["n_layers"] = n_layers
            hf.attrs["source"] = str(text_path.name)

    logger.info(
        "Wrote %s: head shape (%d, %d, %d), %d timesteps",
        hdf_path,
        t_idx,
        n_nodes,
        n_layers,
        t_idx,
    )
    return hdf_path


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Convert IWFM GWALLOUTFL text file to HDF5",
    )
    parser.add_argument("text_file", help="Path to the IWFM text output file")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output HDF5 file path (default: same name with .hdf)",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=1,
        help="Number of groundwater layers (default: 1)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    result = convert_headall_to_hdf(args.text_file, args.output, n_layers=args.layers)
    print(f"Created: {result}")


if __name__ == "__main__":
    main()
