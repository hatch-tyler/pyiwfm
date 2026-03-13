"""CLI subcommand for PEST++ parameter estimation workflows.

Usage::

    pyiwfm pest setup --model-dir ./model [--output-dir ./pest] [--case-name iwfm_cal]
    pyiwfm pest run --pst-file ./pest/iwfm_cal.pst [--executable pestpp-glm]
    pyiwfm pest analyze --pst-file ./pest/iwfm_cal.pst [--format text]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def add_pest_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the ``pest`` subcommand with its sub-subcommands."""
    pest_parser = subparsers.add_parser(
        "pest",
        help="PEST++ parameter estimation workflow",
        description="Generate, run, and analyze PEST++ calibration files.",
    )
    pest_sub = pest_parser.add_subparsers(dest="pest_command", help="PEST++ sub-commands")

    # ---- setup ----
    setup_parser = pest_sub.add_parser(
        "setup",
        help="Generate PEST++ control, template, and instruction files",
    )
    setup_parser.add_argument(
        "--model-dir",
        "-m",
        type=str,
        required=True,
        help="Path to IWFM model directory",
    )
    setup_parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Output directory for PEST++ files (default: <model-dir>/pest)",
    )
    setup_parser.add_argument(
        "--case-name",
        type=str,
        default="iwfm_cal",
        help="Base name for PEST++ files (default: iwfm_cal)",
    )
    setup_parser.add_argument(
        "--zones",
        nargs="+",
        type=int,
        default=None,
        help="Zone IDs for zone-based parameters (e.g., 1 2 3)",
    )
    setup_parser.add_argument(
        "--param-types",
        nargs="+",
        default=None,
        help="Parameter types to include (e.g., hk vk ss sy). Default: hk.",
    )
    setup_parser.add_argument(
        "--obs-file",
        type=str,
        default=None,
        help="Observation well specification file for setting up observations",
    )
    setup_parser.set_defaults(func=run_pest)

    # ---- run ----
    run_parser = pest_sub.add_parser(
        "run",
        help="Execute PEST++ (requires pestpp-glm or pestpp-ies on PATH)",
    )
    run_parser.add_argument(
        "--pst-file",
        "-p",
        type=str,
        required=True,
        help="Path to the PEST++ control (.pst) file",
    )
    run_parser.add_argument(
        "--executable",
        type=str,
        default="pestpp-glm",
        help="PEST++ executable name (default: pestpp-glm)",
    )
    run_parser.add_argument(
        "--num-workers",
        "-n",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)",
    )
    run_parser.add_argument(
        "--working-dir",
        "-w",
        type=str,
        default=None,
        help="Working directory for PEST++ execution (default: directory of .pst file)",
    )
    run_parser.set_defaults(func=run_pest)

    # ---- analyze ----
    analyze_parser = pest_sub.add_parser(
        "analyze",
        help="Post-process PEST++ results",
    )
    analyze_parser.add_argument(
        "--pst-file",
        "-p",
        type=str,
        required=True,
        help="Path to the PEST++ control (.pst) file",
    )
    analyze_parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Output directory for analysis results (default: same as .pst file)",
    )
    analyze_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format for analysis report (default: text)",
    )
    analyze_parser.set_defaults(func=run_pest)

    # Default handler when no sub-subcommand is given
    pest_parser.set_defaults(func=run_pest)


def run_pest(args: argparse.Namespace) -> int:
    """Dispatch to the appropriate pest sub-command."""
    command = getattr(args, "pest_command", None)
    if command is None:
        print("Usage: pyiwfm pest {setup,run,analyze}")
        print("Run 'pyiwfm pest -h' for help.")
        return 1

    if command == "setup":
        return _run_pest_setup(args)
    elif command == "run":
        return _run_pest_run(args)
    elif command == "analyze":
        return _run_pest_analyze(args)
    else:
        print(f"Unknown pest sub-command: {command}")
        return 1


def _run_pest_setup(args: argparse.Namespace) -> int:
    """Generate PEST++ files from an IWFM model."""
    model_dir = Path(args.model_dir)
    if not model_dir.is_dir():
        print(f"Error: Model directory not found: {model_dir}", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir) if args.output_dir else model_dir / "pest"
    case_name: str = args.case_name

    print(f"Setting up PEST++ files from model: {model_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Case name: {case_name}")

    try:
        from pyiwfm.runner.pest_helper import IWFMPestHelper
    except ImportError as e:
        print(f"Error: Missing dependency: {e}", file=sys.stderr)
        print("Install with: pip install pyiwfm[pest]", file=sys.stderr)
        return 1

    # Create helper
    helper = IWFMPestHelper(
        pest_dir=output_dir,
        case_name=case_name,
        model_dir=model_dir,
    )

    # Add parameters
    param_types = args.param_types or ["hk"]
    zones = args.zones or [1]
    print(f"Adding parameter types: {', '.join(param_types)}")
    print(f"Zones: {zones}")

    for ptype in param_types:
        try:
            helper.add_zone_parameters(ptype, zones=zones)
        except (ValueError, KeyError) as e:
            print(f"  Warning: Could not add parameter type '{ptype}': {e}", file=sys.stderr)

    # Add observations if spec file provided
    if args.obs_file:
        obs_path = Path(args.obs_file)
        if not obs_path.exists():
            print(f"Error: Observation file not found: {obs_path}", file=sys.stderr)
            return 1
        print(f"Loading observation spec: {obs_path}")
        try:
            from datetime import datetime

            from pyiwfm.calibration.obs_well_spec import read_obs_well_spec

            wells = read_obs_well_spec(str(obs_path))
            placeholder_time = datetime(2000, 1, 1)
            for well in wells:
                helper.add_head_observations(
                    well_id=well.name,
                    x=well.x,
                    y=well.y,
                    times=[placeholder_time],
                    values=[0.0],
                    layer=max(1, well.overwrite_layer),
                    group="head",
                )
            print(f"  Added {len(wells)} observation wells")
        except Exception as e:
            print(f"  Warning: Could not load observations: {e}", file=sys.stderr)

    # Build PEST++ files
    try:
        pst_path = helper.build()
        print(f"  Control file: {pst_path}")
    except ValueError as e:
        print(f"Error building PEST++ files: {e}", file=sys.stderr)
        print(
            "Ensure both parameters and observations are defined. "
            "Use --obs-file to supply observations.",
            file=sys.stderr,
        )
        return 1
    except Exception as e:
        print(f"Error writing PEST++ files: {e}", file=sys.stderr)
        logger.exception("PEST++ setup failed")
        return 1

    print(f"\nPEST++ setup complete. Files written to: {output_dir}")
    print(f"Run calibration with: pyiwfm pest run --pst-file {pst_path}")
    return 0


def _run_pest_run(args: argparse.Namespace) -> int:
    """Execute PEST++."""
    import shutil
    import subprocess

    pst_file = Path(args.pst_file)
    if not pst_file.exists():
        print(f"Error: PST file not found: {pst_file}", file=sys.stderr)
        return 1

    executable: str = args.executable
    working_dir = Path(args.working_dir) if args.working_dir else pst_file.parent

    # Check if executable is available
    exe_path = shutil.which(executable)
    if exe_path is None:
        print(f"Error: '{executable}' not found on PATH.", file=sys.stderr)
        print("Install PEST++ from: https://github.com/usgs/pestpp", file=sys.stderr)
        return 1

    print(f"Running {executable} with {pst_file.name}")
    print(f"Working directory: {working_dir}")

    cmd: list[str] = [exe_path, str(pst_file)]
    if args.num_workers > 1:
        cmd.extend(["/h", f":{args.num_workers}"])

    try:
        result = subprocess.run(
            cmd,
            cwd=str(working_dir),
            check=False,
            text=True,
        )
        if result.returncode != 0:
            print(f"\n{executable} exited with code {result.returncode}")
            return result.returncode
        else:
            print(f"\n{executable} completed successfully.")
            return 0
    except Exception as e:
        print(f"Error running {executable}: {e}", file=sys.stderr)
        return 1


def _run_pest_analyze(args: argparse.Namespace) -> int:
    """Post-process PEST++ results."""
    pst_file = Path(args.pst_file)
    if not pst_file.exists():
        print(f"Error: PST file not found: {pst_file}", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir) if args.output_dir else pst_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Derive case name from .pst filename
    case_name = pst_file.stem

    print(f"Analyzing PEST++ results for case: {case_name}")
    print(f"Directory: {pst_file.parent}")

    try:
        from pyiwfm.runner.pest_postprocessor import PestPostProcessor
    except ImportError as e:
        print(f"Error: Missing dependency: {e}", file=sys.stderr)
        return 1

    try:
        processor = PestPostProcessor(
            pest_dir=pst_file.parent,
            case_name=case_name,
        )

        if args.format == "json":
            import json

            results = processor.load_results()
            out_path = output_dir / f"{case_name}_analysis.json"
            summary: dict[str, object] = {
                "case_name": results.case_name,
                "n_iterations": results.n_iterations,
                "final_phi": results.final_phi,
                "iteration_phi": results.iteration_phi,
            }
            if results.residuals:
                stats = results.fit_statistics()
                summary["fit_statistics"] = stats
            if results.calibrated_values:
                summary["calibrated_values"] = results.calibrated_values
            with open(out_path, "w") as f:
                json.dump(summary, f, indent=2, default=str)
            print(f"JSON report written to: {out_path}")

        else:  # text
            report = processor.summary_report()
            out_path = output_dir / f"{case_name}_analysis.txt"
            with open(out_path, "w") as f:
                f.write(report)
            print(f"Text report written to: {out_path}")

        print("\nAnalysis complete.")
        return 0

    except Exception as e:
        print(f"Error analyzing results: {e}", file=sys.stderr)
        logger.exception("PEST++ analysis failed")
        return 1
