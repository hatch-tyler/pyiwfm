"""
FastAPI-based web viewer for IWFM models.

This module provides a modern web viewer using:
- FastAPI for the REST API backend
- React + vtk.js for client-side 3D rendering
"""

from __future__ import annotations

try:
    from pyiwfm.visualization.webapi.server import create_app, launch_viewer
except ImportError:  # fastapi not installed

    def create_app(*args: object, **kwargs: object) -> None:  # type: ignore[misc]
        """Stub — raises ImportError when webapi extras are not installed."""
        raise ImportError("webapi extras required: pip install pyiwfm[webapi]")

    def launch_viewer(*args: object, **kwargs: object) -> None:  # type: ignore[misc]
        """Stub — raises ImportError when webapi extras are not installed."""
        raise ImportError("webapi extras required: pip install pyiwfm[webapi]")


__all__ = ["create_app", "launch_viewer"]
