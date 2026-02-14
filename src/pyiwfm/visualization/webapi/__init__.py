"""
FastAPI-based web viewer for IWFM models.

This module provides a modern web viewer using:
- FastAPI for the REST API backend
- React + vtk.js for client-side 3D rendering
"""

from __future__ import annotations

from pyiwfm.visualization.webapi.server import create_app, launch_viewer

__all__ = ["create_app", "launch_viewer"]
