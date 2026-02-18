"""
``pyiwfm viewer`` subcommand.

Launches the FastAPI-based web viewer with client-side vtk.js rendering.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def add_viewer_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``viewer`` subcommand."""
    p = subparsers.add_parser(
        "viewer",
        help="Launch the interactive web viewer for an IWFM model.",
        description=(
            "Launch the web viewer for an IWFM model.\n\n"
            "Uses client-side vtk.js rendering with a FastAPI backend."
        ),
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
        help="Load from preprocessor file (relative to --model-dir or absolute)",
    )
    p.add_argument(
        "--simulation",
        type=Path,
        metavar="FILE",
        help="Load from simulation file (relative to --model-dir or absolute)",
    )
    p.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Server host address (default: 127.0.0.1)",
    )
    p.add_argument(
        "--port",
        type=int,
        default=8080,
        help="HTTP port for the web server (default: 8080)",
    )
    p.add_argument(
        "--title",
        type=str,
        default=None,
        help="Application title (default: auto-detect from model)",
    )
    p.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't automatically open the browser",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging and API docs",
    )
    p.add_argument(
        "--crs",
        type=str,
        default="+proj=utm +zone=10 +datum=NAD83 +units=us-ft +no_defs",
        help="Source coordinate reference system (default: UTM Zone 10N, NAD83, US survey feet)",
    )
    p.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable SQLite cache layer for results data",
    )
    p.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Force rebuild of SQLite cache on startup",
    )

    p.set_defaults(func=run_viewer)


def run_viewer(args: argparse.Namespace) -> int:
    """Run the ``viewer`` subcommand."""
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
    elif args.simulation and args.simulation.is_absolute():
        model_dir = args.simulation.parent
    else:
        model_dir = Path.cwd()

    if not model_dir.exists():
        print(f"ERROR: Model directory not found: {model_dir}")
        return 1

    logger.info("Model directory: %s", model_dir)

    try:
        from pyiwfm.cli._model_loader import load_model

        logger.info("Loading IWFM model...")
        model = load_model(
            model_dir,
            preprocessor_file=args.preprocessor,
            simulation_file=args.simulation,
        )
        logger.info("Model loaded: %s", model.name)
        logger.info("  Nodes: %d", model.n_nodes)
        logger.info("  Elements: %d", model.n_elements)
        logger.info("  Layers: %d", model.n_layers)

        if model.has_streams:
            logger.info("  Stream nodes: %d", model.n_stream_nodes)
        if model.has_lakes:
            logger.info("  Lakes: %d", model.n_lakes)

    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 1
    except Exception as exc:
        logger.exception("Failed to load model")
        print(f"ERROR: Failed to load model: {exc}")
        return 1

    title = args.title
    if title is None:
        from pyiwfm.cli._model_finder import extract_model_name

        ref = args.preprocessor or args.simulation or model_dir
        title = f"{extract_model_name(ref)} - IWFM Viewer"

    try:
        from pyiwfm.visualization.webapi import launch_viewer

        open_browser = not args.no_browser

        print()
        print(f"  Starting web server on http://{args.host}:{args.port}")
        if args.debug:
            print(f"  API docs at http://{args.host}:{args.port}/api/docs")
        print("  Press Ctrl+C to stop the server")
        print()

        launch_viewer(
            model=model,
            host=args.host,
            port=args.port,
            title=title,
            open_browser=open_browser,
            debug=args.debug,
            crs=args.crs,
            no_cache=getattr(args, "no_cache", False),
            rebuild_cache=getattr(args, "rebuild_cache", False),
        )

    except ImportError as exc:
        logger.error("Failed to import web viewer: %s", exc)
        print("ERROR: Web API dependencies not installed.")
        print("Install with: pip install pyiwfm[webapi]")
        return 1
    except KeyboardInterrupt:
        print()
        print("Server stopped.")
        return 0
    except Exception as exc:
        logger.exception("Failed to start web viewer")
        print(f"ERROR: Failed to start web viewer: {exc}")
        return 1

    return 0
