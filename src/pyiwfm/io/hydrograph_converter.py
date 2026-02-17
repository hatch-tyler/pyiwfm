"""
Converter for IWFM hydrograph output text files (.out) to HDF5.

Streams the text file line-by-line into an HDF5 cache with metadata
(hydrograph IDs, layers, node IDs) for efficient lazy loading.

The output HDF5 is consumed by ``LazyHydrographDataLoader`` in the
web viewer backend.

Usage::

    python -m pyiwfm.io.hydrograph_converter GW_Hydrographs.out
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path

import h5py
import numpy as np

logger = logging.getLogger(__name__)

_CHUNK_GROW = 256


def _parse_timestamp(text: str) -> str:
    """Parse an IWFM timestamp to ISO 8601 string.

    Handles ``MM/DD/YYYY_HH:MM`` and the ``_24:00`` convention.
    """
    text = text.strip().replace("_", " ")
    if "24:00" in text:
        text = text.replace("24:00", "00:00")
        dt = datetime.strptime(text, "%m/%d/%Y %H:%M") + timedelta(days=1)
    else:
        dt = datetime.strptime(text, "%m/%d/%Y %H:%M")
    return dt.isoformat()


def convert_hydrograph_to_hdf(
    text_file: str | Path,
    hdf_file: str | Path | None = None,
) -> Path:
    """Convert an IWFM hydrograph output text file to HDF5.

    Parameters
    ----------
    text_file : str or Path
        Path to the IWFM hydrograph output file (``.out``).
    hdf_file : str or Path or None
        Output HDF5 path.  If *None*, uses ``text_file`` with
        ``.hydrograph_cache.hdf`` suffix.

    Returns
    -------
    Path
        Path to the created HDF5 file.
    """
    text_path = Path(text_file)
    if hdf_file is None:
        hdf_path = text_path.with_suffix(".hydrograph_cache.hdf")
    else:
        hdf_path = Path(hdf_file)

    logger.info("Converting hydrograph %s -> %s", text_path, hdf_path)

    hydrograph_ids: list[int] = []
    layers: list[int] = []
    node_ids: list[int] = []

    header_lines: list[str] = []
    data_start_line = 0

    with open(text_path) as fh:
        # First pass: parse header and count data lines
        all_lines = fh.readlines()

    for i, line in enumerate(all_lines):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("*"):
            header_lines.append(stripped)
        else:
            data_start_line = i
            break

    # Parse header metadata
    for hline in header_lines:
        content = hline.lstrip("*").strip()
        upper = content.upper()
        if upper.startswith("HYDROGRAPH ID"):
            parts = content.split()
            hydrograph_ids = [int(x) for x in parts[2:]]
        elif upper.startswith("LAYER"):
            parts = content.split()
            layers = [int(x) for x in parts[1:]]
        elif upper.startswith("NODE") or upper.startswith("ELEMENT"):
            parts = content.split()
            int_vals: list[int] = []
            for p in parts[1:]:
                try:
                    int_vals.append(int(p))
                except ValueError:
                    continue
            node_ids = int_vals

    # Parse data lines
    timestamps: list[str] = []
    rows: list[list[float]] = []
    n_cols = 0

    for i in range(data_start_line, len(all_lines)):
        line = all_lines[i].strip()
        if not line or line.startswith("*"):
            continue

        parts = line.split()
        if len(parts) < 2:
            continue

        try:
            iso_time = _parse_timestamp(parts[0])
        except (ValueError, IndexError):
            continue

        timestamps.append(iso_time)
        vals: list[float] = []
        for v in parts[1:]:
            try:
                vals.append(float(v))
            except ValueError:
                vals.append(float("nan"))
        rows.append(vals)
        if len(vals) > n_cols:
            n_cols = len(vals)

    if not rows:
        raise ValueError(f"No data found in hydrograph file: {text_path}")

    n_timesteps = len(rows)
    logger.info("Parsed %d timesteps, %d columns", n_timesteps, n_cols)

    # Pad short rows and build array
    for r in rows:
        while len(r) < n_cols:
            r.append(float("nan"))

    data = np.array(rows, dtype=np.float64)

    # Write HDF5
    str_dt = h5py.string_dtype(encoding="utf-8")
    with h5py.File(hdf_path, "w") as hf:
        hf.create_dataset(
            "data",
            data=data,
            dtype=np.float64,
            compression="gzip",
            compression_opts=4,
            chunks=(1, n_cols),
        )
        hf.create_dataset("times", data=timestamps, dtype=str_dt)

        if hydrograph_ids:
            hf.create_dataset(
                "hydrograph_ids",
                data=np.array(hydrograph_ids, dtype=np.int32),
            )
        if layers:
            hf.create_dataset(
                "layers",
                data=np.array(layers, dtype=np.int32),
            )
        if node_ids:
            hf.create_dataset(
                "node_ids",
                data=np.array(node_ids, dtype=np.int32),
            )

        hf.attrs["n_columns"] = n_cols
        hf.attrs["n_timesteps"] = n_timesteps
        hf.attrs["source"] = str(text_path.name)

    logger.info(
        "Wrote %s: data shape (%d, %d)",
        hdf_path,
        n_timesteps,
        n_cols,
    )
    return hdf_path


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Convert IWFM hydrograph output text file to HDF5",
    )
    parser.add_argument("text_file", help="Path to the IWFM hydrograph .out file")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output HDF5 file path (default: {name}.hydrograph_cache.hdf)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    result = convert_hydrograph_to_hdf(args.text_file, args.output)
    print(f"Created: {result}")


if __name__ == "__main__":
    main()
