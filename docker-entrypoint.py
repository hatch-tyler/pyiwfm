#!/usr/bin/env python3
"""
Docker entrypoint script for pyiwfm web visualization.

Translates environment variables into pyiwfm CLI arguments and starts
the FastAPI-based web viewer.

Environment Variables:
    MODEL_PATH: Path to model directory (default: /model)
    PORT: Web server port (default: 8080)
    TITLE: Viewer title (default: auto-detect)
    MODE: "web" for web viewer, "export" for VTK/GeoPackage export (default: web)
"""

import os
import sys


def main() -> None:
    model_path = os.environ.get("MODEL_PATH", "/model")
    port = os.environ.get("PORT", "8080")
    title = os.environ.get("TITLE", "")
    mode = os.environ.get("MODE", "web")

    print("=" * 60)
    print("pyiwfm Docker Container")
    print("=" * 60)
    print(f"Model directory: {model_path}")
    print(f"Port: {port}")
    print(f"Mode: {mode}")
    print()

    from pyiwfm.cli import main as cli_main

    if mode == "export":
        argv = ["export", "--model-dir", model_path, "--output-dir", "/output"]
    else:
        argv = [
            "viewer",
            "--model-dir", model_path,
            "--host", "0.0.0.0",
            "--port", port,
            "--no-browser",
        ]
        if title:
            argv += ["--title", title]

    sys.exit(cli_main(argv))


if __name__ == "__main__":
    main()
