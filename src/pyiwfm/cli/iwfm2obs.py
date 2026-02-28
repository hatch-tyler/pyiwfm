"""
CLI subcommand for IWFM2OBS time interpolation.

Usage::

    # Explicit SMP mode (original)
    pyiwfm iwfm2obs --obs obs.smp --sim sim.smp --output interp.smp [--threshold 30]

    # Model discovery mode (auto-discovers .out files)
    pyiwfm iwfm2obs --model C2VSimFG.in --obs-gw gw_obs.smp --output-gw gw_out.smp
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyiwfm.calibration.iwfm2obs import InterpolationConfig


def add_iwfm2obs_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the ``pyiwfm iwfm2obs`` subcommand."""
    p = subparsers.add_parser(
        "iwfm2obs",
        help="Interpolate simulated heads to observation times (IWFM2OBS)",
    )

    # Explicit SMP mode (original arguments)
    p.add_argument("--obs", type=str, help="Observed data SMP file (explicit mode)")
    p.add_argument("--sim", type=str, help="Simulated data SMP file (explicit mode)")
    p.add_argument("--output", type=str, help="Output interpolated SMP file (explicit mode)")

    # Model discovery mode
    p.add_argument(
        "--model",
        type=str,
        help="IWFM simulation main file (auto-discovers .out files)",
    )
    p.add_argument("--obs-gw", type=str, help="GW observation SMP file")
    p.add_argument("--output-gw", type=str, help="GW output SMP file")
    p.add_argument("--obs-stream", type=str, help="Stream observation SMP file")
    p.add_argument("--output-stream", type=str, help="Stream output SMP file")

    # Multi-layer options
    p.add_argument("--well-spec", type=str, help="Multi-layer well specification file")
    p.add_argument("--multilayer-out", type=str, help="Multi-layer output file path")
    p.add_argument("--multilayer-ins", type=str, help="Multi-layer PEST .ins file path")

    # Shared options
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

    interp_config = InterpolationConfig(max_extrapolation_time=timedelta(days=args.threshold))

    # Model discovery mode
    if getattr(args, "model", None):
        return _run_model_mode(args, interp_config)

    # Explicit SMP mode (original behavior)
    if not args.obs or not args.sim or not args.output:
        print(
            "Error: --obs, --sim, --output are required in explicit mode.\n"
            "Use --model for auto-discovery mode.",
            file=sys.stderr,
        )
        return 1

    obs_path = Path(args.obs)
    sim_path = Path(args.sim)
    out_path = Path(args.output)

    if not obs_path.exists():
        print(f"Error: observed file not found: {obs_path}", file=sys.stderr)
        return 1
    if not sim_path.exists():
        print(f"Error: simulated file not found: {sim_path}", file=sys.stderr)
        return 1

    result = iwfm2obs(obs_path, sim_path, out_path, config=interp_config)

    print(f"Interpolated {len(result)} bore(s) to: {out_path}")
    return 0


def _run_model_mode(
    args: argparse.Namespace,
    interp_config: InterpolationConfig,
) -> int:
    """Execute IWFM2OBS in model discovery mode."""
    from pyiwfm.calibration.iwfm2obs import IWFM2OBSConfig, iwfm2obs_from_model

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: model file not found: {model_path}", file=sys.stderr)
        return 1

    # Build obs/output path maps
    obs_paths: dict[str, Path] = {}
    out_paths: dict[str, Path] = {}

    if args.obs_gw and args.output_gw:
        obs_paths["gw"] = Path(args.obs_gw)
        out_paths["gw"] = Path(args.output_gw)
    if args.obs_stream and args.output_stream:
        obs_paths["stream"] = Path(args.obs_stream)
        out_paths["stream"] = Path(args.output_stream)

    if not obs_paths:
        print(
            "Error: at least one observation/output pair required (e.g. --obs-gw and --output-gw)",
            file=sys.stderr,
        )
        return 1

    config = IWFM2OBSConfig(interpolation=interp_config)

    well_spec_path = Path(args.well_spec) if args.well_spec else None
    ml_out_path = Path(args.multilayer_out) if args.multilayer_out else None
    ml_ins_path = Path(args.multilayer_ins) if args.multilayer_ins else None

    results = iwfm2obs_from_model(
        simulation_main_file=model_path,
        obs_smp_paths=obs_paths,
        output_paths=out_paths,
        config=config,
        obs_well_spec_path=well_spec_path,
        multilayer_output_path=ml_out_path,
        multilayer_ins_path=ml_ins_path,
    )

    total = sum(len(v) for v in results.values())
    print(f"Interpolated {total} bore(s) across {len(results)} type(s)")
    return 0
