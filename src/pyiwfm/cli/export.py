"""
``pyiwfm export`` subcommand.

Exports an IWFM model to VTK and/or GeoPackage formats.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def add_export_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``export`` subcommand."""
    p = subparsers.add_parser(
        "export",
        help="Export model data to VTK and/or GeoPackage.",
        description="Export an IWFM model to VTK (.vtu) and/or GeoPackage (.gpkg) formats.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Path to model directory (default: current directory)",
    )
    p.add_argument(
        "--preprocessor",
        type=Path,
        metavar="FILE",
        help="Path to preprocessor file (relative to --model-dir or absolute)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Output directory for exported files (default: ./output)",
    )
    p.add_argument(
        "--format",
        dest="export_format",
        choices=["vtk", "gpkg", "all"],
        default="all",
        help="Export format (default: all)",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    p.set_defaults(func=run_export)


def run_export(args: argparse.Namespace) -> int:
    """Run the ``export`` subcommand."""
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    model_dir: Path
    if args.model_dir is not None:
        model_dir = args.model_dir.resolve()
    elif args.preprocessor and args.preprocessor.is_absolute():
        model_dir = args.preprocessor.parent
    else:
        model_dir = Path.cwd()

    if not model_dir.exists():
        print(f"ERROR: Model directory not found: {model_dir}")
        return 1

    # Load model (reuse viewer's loader)
    from pyiwfm.cli._model_loader import load_model

    try:
        model = load_model(
            model_dir,
            preprocessor_file=args.preprocessor,
        )
    except Exception as exc:
        logger.exception("Failed to load model")
        print(f"ERROR: Failed to load model: {exc}")
        return 1

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fmt = args.export_format
    exported: list[str] = []

    # --- VTK export -------------------------------------------------------
    if fmt in ("vtk", "all"):
        try:
            from pyiwfm.visualization import VTKExporter

            if VTKExporter is None:
                raise ImportError("VTK export requires the 'vtk' package")

            if model.mesh and model.stratigraphy:
                exporter = VTKExporter(model.mesh, model.stratigraphy)

                vtu_3d = output_dir / "model_3d.vtu"
                exporter.export_vtu(vtu_3d, mode="3d")
                print(f"  Exported: {vtu_3d}")
                exported.append(str(vtu_3d))

                vtu_2d = output_dir / "model_2d.vtu"
                exporter.export_vtu(vtu_2d, mode="2d")
                print(f"  Exported: {vtu_2d}")
                exported.append(str(vtu_2d))
            else:
                print("  Skipping VTK export (mesh or stratigraphy not available)")
        except Exception as exc:
            logger.warning("VTK export failed: %s", exc)
            print(f"  VTK export failed: {exc}")

    # --- GeoPackage export ------------------------------------------------
    if fmt in ("gpkg", "all"):
        try:
            from pyiwfm.visualization import GISExporter

            if model.mesh:
                gis_exporter = GISExporter(
                    grid=model.mesh,
                    stratigraphy=model.stratigraphy,
                    streams=model.streams,
                )
                gpkg = output_dir / "model.gpkg"
                gis_exporter.export_geopackage(gpkg)
                print(f"  Exported: {gpkg}")
                exported.append(str(gpkg))
            else:
                print("  Skipping GeoPackage export (mesh not available)")
        except Exception as exc:
            logger.warning("GeoPackage export failed: %s", exc)
            print(f"  GeoPackage export failed: {exc}")

    if exported:
        print()
        print(f"Export complete. {len(exported)} file(s) written to {output_dir}")
    else:
        print()
        print("No files were exported.")

    return 0
