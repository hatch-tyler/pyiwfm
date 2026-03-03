"""
CLI subcommand for CalcTypHyd typical hydrograph computation.

Usage::

    pyiwfm calctyphyd --water-levels wl.smp --weights weights.txt --output typhyd.smp
    pyiwfm calctyphyd --config CalcTypeHyd_Sub1Sub2_sim.in
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
    # Config mode (Fortran .in file)
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Fortran-format CalcTypHyd .in config file (overrides other args)",
    )
    # Direct args mode
    p.add_argument(
        "--water-levels",
        type=str,
        default=None,
        help="Water level observations SMP file",
    )
    p.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Cluster membership weights file",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output typical hydrographs SMP file",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for PEST files (config mode)",
    )
    p.set_defaults(func=run_calctyphyd)


def run_calctyphyd(args: argparse.Namespace) -> int:
    """Execute CalcTypHyd computation."""

    if args.config is not None:
        return _run_config_mode(args)
    return _run_direct_mode(args)


def _run_config_mode(args: argparse.Namespace) -> int:
    """Run CalcTypHyd from a Fortran-format .in config file."""
    from pyiwfm.calibration.calctyphyd import (
        CalcTypHydConfig,
        compute_typical_hydrographs_timeseries,
        read_calctyphyd_config,
        read_cluster_weights,
        write_pest_output,
    )
    from pyiwfm.io.smp import SMPReader

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        return 1

    file_config = read_calctyphyd_config(config_path)

    if not file_config.water_level_path.exists():
        print(
            f"Error: water levels file not found: {file_config.water_level_path}",
            file=sys.stderr,
        )
        return 1
    if not file_config.weights_path.exists():
        print(
            f"Error: weights file not found: {file_config.weights_path}",
            file=sys.stderr,
        )
        return 1

    # Read data
    reader = SMPReader(file_config.water_level_path)
    water_levels = reader.read()
    cluster_weights = read_cluster_weights(
        file_config.weights_path,
        n_clusters=file_config.n_clusters,
    )

    # Compute time-series typical hydrographs
    calc_config = CalcTypHydConfig(
        seasonal_periods=file_config.periods,
    )
    result = compute_typical_hydrographs_timeseries(
        water_levels, cluster_weights, file_config, calc_config
    )

    # Write PEST output
    output_dir = Path(args.output_dir) if args.output_dir else config_path.parent
    written = write_pest_output(result, file_config, output_dir)

    for p in written:
        print(f"  {p}")
    print(f"Wrote {len(written)} file(s) for {len(result.hydrographs)} cluster(s)")
    return 0


def _run_direct_mode(args: argparse.Namespace) -> int:
    """Run CalcTypHyd with direct CLI arguments (original mode)."""
    import numpy as np

    from pyiwfm.calibration.calctyphyd import compute_typical_hydrographs, read_cluster_weights
    from pyiwfm.io.smp import SMPReader, SMPTimeSeries, SMPWriter

    if args.water_levels is None or args.weights is None or args.output is None:
        print(
            "Error: --water-levels, --weights, and --output are required (unless --config is used)",
            file=sys.stderr,
        )
        return 1

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
