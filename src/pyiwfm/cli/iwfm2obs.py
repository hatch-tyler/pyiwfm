"""
CLI subcommand for IWFM2OBS time interpolation.

Usage::

    pyiwfm iwfm2obs --obs obs.smp --sim sim.smp --output interp.smp [--threshold 30]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def add_iwfm2obs_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the ``pyiwfm iwfm2obs`` subcommand."""
    p = subparsers.add_parser(
        "iwfm2obs",
        help="Interpolate simulated heads to observation times (IWFM2OBS)",
    )
    p.add_argument("--obs", type=str, required=True, help="Observed data SMP file")
    p.add_argument("--sim", type=str, required=True, help="Simulated data SMP file")
    p.add_argument("--output", type=str, required=True, help="Output interpolated SMP file")
    p.add_argument(
        "--threshold",
        type=int,
        default=30,
        help="Max extrapolation days (default: 30)",
    )
    p.set_defaults(func=run_iwfm2obs)


def run_iwfm2obs(args: argparse.Namespace) -> int:
    """Execute IWFM2OBS interpolation."""
    from datetime import timedelta

    from pyiwfm.calibration.iwfm2obs import InterpolationConfig, iwfm2obs

    obs_path = Path(args.obs)
    sim_path = Path(args.sim)
    out_path = Path(args.output)

    if not obs_path.exists():
        print(f"Error: observed file not found: {obs_path}", file=sys.stderr)
        return 1
    if not sim_path.exists():
        print(f"Error: simulated file not found: {sim_path}", file=sys.stderr)
        return 1

    config = InterpolationConfig(max_extrapolation_time=timedelta(days=args.threshold))
    result = iwfm2obs(obs_path, sim_path, out_path, config=config)

    print(f"Interpolated {len(result)} bore(s) to: {out_path}")
    return 0
