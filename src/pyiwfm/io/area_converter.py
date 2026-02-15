"""
Converter for IWFM land-use area text files to HDF5.

Converts the IWFM area time-series text files (non-ponded, ponded, urban,
native/riparian) into HDF5 files for fast random access by the web viewer.

Text file format::

    NFACTARL  NSPRN  NFLL  NWINT  NSPCL  / column pointers
    FACTARL                                / unit conversion factor
    DSSFL                                  / DSS file (blank = none)
    MM/DD/YYYY_HH:MM  IE  A1  A2  ...  AN
                       IE  A1  A2  ...  AN
    ...

The date appears only on the **first** row of each timestep block.
Subsequent rows are continuation rows (element ID + values, no date).

The converter streams the file line-by-line, writing each timestep directly
to a pre-allocated HDF5 dataset.  Memory usage is O(n_elements * n_cols)
regardless of the number of timesteps.

Usage::

    python -m pyiwfm.io.area_converter NonPondedCropArea.dat --label nonponded
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np

from pyiwfm.io.rootzone_area import _has_date, _is_comment, _strip_description

logger = logging.getLogger(__name__)

# How many timesteps to grow the HDF5 dataset by when it fills up
_CHUNK_GROW = 256


def convert_area_to_hdf(
    text_file: str | Path,
    hdf_file: str | Path | None = None,
    label: str = "area",
) -> Path:
    """Convert an IWFM land-use area text file to HDF5 (streaming).

    Creates a dataset named *label* with shape
    ``(n_timesteps, n_elements, n_cols)`` chunked as
    ``(1, n_elements, n_cols)`` for O(1) timestep access.

    If *hdf_file* is ``None``, the output is
    ``text_file.with_suffix('.area_cache.hdf')``.

    Handles both formats: date on every row, or date only on the
    first row of each timestep block (continuation rows).

    Parameters
    ----------
    text_file : str or Path
        Path to the IWFM area time-series text file.
    hdf_file : str or Path or None
        Output HDF5 path.
    label : str
        Dataset name and label attribute (e.g. "nonponded", "ponded").

    Returns
    -------
    Path
        Path to the created HDF5 file.
    """
    text_path = Path(text_file)
    if hdf_file is None:
        hdf_path = text_path.with_suffix(".area_cache.hdf")
    else:
        hdf_path = Path(hdf_file)

    logger.info("Converting %s -> %s (label=%s)", text_path, hdf_path, label)

    with open(text_path) as fh:
        # --- Read header (3 non-comment, non-blank lines) ---
        header_vals: list[str] = []
        while len(header_vals) < 3:
            line = fh.readline()
            if not line:
                raise ValueError(
                    f"Unexpected end of file reading header from {text_path}"
                )
            if _is_comment(line):
                continue
            val = _strip_description(line)
            header_vals.append(val)

        # Parse header
        header_n_cols = len(header_vals[0].split())
        try:
            factor = float(header_vals[1])
        except ValueError:
            factor = 1.0
        dss_file = header_vals[2] if header_vals[2] else ""

        # --- First pass: detect n_elements, n_cols from first timestep ---
        first_data_pos = fh.tell()
        element_ids_first: list[int] = []
        first_date: str | None = None
        n_cols: int | None = None

        for line in fh:
            if _is_comment(line):
                continue
            val = _strip_description(line)
            if not val:
                continue
            parts = val.split()
            if len(parts) < 2:
                continue

            # Determine if this row has a date
            if _has_date(parts[0]):
                date_str = parts[0]
                elem_id = int(parts[1])
                val_tokens = parts[2:]
            else:
                # Continuation row (no date) — belongs to current block
                if first_date is None:
                    # Can't have a continuation before the first date
                    continue
                date_str = first_date
                elem_id = int(parts[0])
                val_tokens = parts[1:]

            if first_date is None:
                first_date = date_str
                n_cols = len(val_tokens)

            if date_str == first_date:
                element_ids_first.append(elem_id)
            else:
                # Hit the next timestep — done reading first block
                break

        if not element_ids_first:
            raise ValueError(f"No data rows found in {text_path}")

        n_elements = len(element_ids_first)
        if n_cols is None:
            n_cols = header_n_cols
        elif n_cols != header_n_cols:
            logger.debug(
                "Header n_cols=%d but data row has %d; using %d",
                header_n_cols,
                n_cols,
                n_cols,
            )

        # Count all data lines for timestep estimate
        fh.seek(first_data_pos)
        total_data_lines = 0
        for line in fh:
            if _is_comment(line):
                continue
            stripped = _strip_description(line)
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) >= 2:
                total_data_lines += 1

        n_timesteps_est = max(total_data_lines // n_elements, 1)
        logger.info(
            "Detected %d elements, %d cols, ~%d timesteps, %d data lines",
            n_elements,
            n_cols,
            n_timesteps_est,
            total_data_lines,
        )

        # --- Second pass: stream data into HDF5 ---
        fh.seek(first_data_pos)

        element_ids_arr = np.array(element_ids_first, dtype=np.int32)
        # Build element_id -> row index mapping
        elem_to_idx = {eid: i for i, eid in enumerate(element_ids_first)}

        timestamps: list[str] = []
        t_idx = 0

        with h5py.File(hdf_path, "w") as hf:
            ds = hf.create_dataset(
                label,
                shape=(n_timesteps_est, n_elements, n_cols),
                maxshape=(None, n_elements, n_cols),
                dtype=np.float64,
                compression="gzip",
                compression_opts=4,
                chunks=(1, n_elements, n_cols),
            )

            row_buf = np.zeros((n_elements, n_cols), dtype=np.float64)
            current_date: str | None = None
            rows_in_block = 0

            for line in fh:
                if _is_comment(line):
                    continue
                val = _strip_description(line)
                if not val:
                    continue
                parts = val.split()
                if len(parts) < 2:
                    continue

                # Parse row: date-bearing or continuation
                if _has_date(parts[0]):
                    date_str = parts[0]
                    elem_id = int(parts[1])
                    val_tokens = parts[2:]
                else:
                    # Continuation row — use current date
                    if current_date is None:
                        continue
                    date_str = current_date
                    elem_id = int(parts[0])
                    val_tokens = parts[1:]

                values = [
                    float(v) * factor for v in val_tokens[:n_cols]
                ]

                if current_date is None:
                    current_date = date_str
                    row_buf[:] = 0.0
                    rows_in_block = 0
                elif date_str != current_date:
                    # Flush completed timestep block
                    if t_idx >= ds.shape[0]:
                        ds.resize(ds.shape[0] + _CHUNK_GROW, axis=0)
                    ds[t_idx, :, :] = row_buf
                    timestamps.append(current_date)
                    t_idx += 1

                    if t_idx % 100 == 0:
                        logger.info("  %d timesteps written...", t_idx)

                    # Start new block
                    current_date = date_str
                    row_buf[:] = 0.0
                    rows_in_block = 0

                idx = elem_to_idx.get(elem_id)
                if idx is not None:
                    row_buf[idx, :len(values)] = values
                    rows_in_block += 1

            # Flush final block
            if current_date is not None and rows_in_block > 0:
                if t_idx >= ds.shape[0]:
                    ds.resize(ds.shape[0] + _CHUNK_GROW, axis=0)
                ds[t_idx, :, :] = row_buf
                timestamps.append(current_date)
                t_idx += 1

            # Trim dataset to actual size
            if t_idx < ds.shape[0]:
                ds.resize(t_idx, axis=0)

            # Write metadata
            str_dt = h5py.string_dtype(encoding="utf-8")
            hf.create_dataset("times", data=timestamps, dtype=str_dt)
            hf.create_dataset("element_ids", data=element_ids_arr)

            hf.attrs["n_elements"] = n_elements
            hf.attrs["n_cols"] = n_cols
            hf.attrs["factor"] = factor
            hf.attrs["source"] = str(text_path.name)
            hf.attrs["label"] = label
            if dss_file:
                hf.attrs["dss_file"] = dss_file

    logger.info(
        "Wrote %s: %s shape (%d, %d, %d), %d timesteps",
        hdf_path,
        label,
        t_idx,
        n_elements,
        n_cols,
        t_idx,
    )
    return hdf_path


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Convert IWFM area text file to HDF5",
    )
    parser.add_argument("text_file", help="Path to the IWFM area text file")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output HDF5 file path (default: same name with .area_cache.hdf)",
    )
    parser.add_argument(
        "--label",
        default="area",
        help="Dataset label (default: area)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    result = convert_area_to_hdf(args.text_file, args.output, label=args.label)
    print(f"Created: {result}")


if __name__ == "__main__":
    main()
