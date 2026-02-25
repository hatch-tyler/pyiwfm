"""
CLI subcommand for CalcTypHyd typical hydrograph computation.

Usage::

    pyiwfm calctyphyd --water-levels wl.smp --weights weights.txt --output typhyd.smp
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def add_calctyphyd_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the ``pyiwfm calctyphyd`` subcommand."""
    p = subparsers.add_parser(
        "calctyphyd",
        help="Compute typical hydrographs from observation data (CalcTypHyd)",
    )
    p.add_argument(
        "--water-levels",
        type=str,
        required=True,
        help="Water level observations SMP file",
    )
    p.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Cluster membership weights file",
    )
    p.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output typical hydrographs SMP file",
    )
    p.set_defaults(func=run_calctyphyd)


def run_calctyphyd(args: argparse.Namespace) -> int:
    """Execute CalcTypHyd computation."""

    import numpy as np

    from pyiwfm.calibration.calctyphyd import compute_typical_hydrographs, read_cluster_weights
    from pyiwfm.io.smp import SMPReader, SMPTimeSeries, SMPWriter

    wl_path = Path(args.water_levels)
    weights_path = Path(args.weights)
    out_path = Path(args.output)

    if not wl_path.exists():
        print(f"Error: water levels file not found: {wl_path}", file=sys.stderr)
        return 1
    if not weights_path.exists():
        print(f"Error: weights file not found: {weights_path}", file=sys.stderr)
        return 1

    reader = SMPReader(wl_path)
    water_levels = reader.read()
    cluster_weights = read_cluster_weights(weights_path)

    result = compute_typical_hydrographs(water_levels, cluster_weights)

    # Convert typical hydrographs to SMP format
    output_data: dict[str, SMPTimeSeries] = {}
    for th in result.hydrographs:
        bore_id = f"CLUSTER_{th.cluster_id}"
        output_data[bore_id] = SMPTimeSeries(
            bore_id=bore_id,
            times=th.times.astype("datetime64[s]"),
            values=th.values,
            excluded=np.zeros(len(th.values), dtype=np.bool_),
        )

    writer = SMPWriter(out_path)
    writer.write(output_data)

    print(f"Wrote {len(result.hydrographs)} typical hydrograph(s) to: {out_path}")
    return 0
