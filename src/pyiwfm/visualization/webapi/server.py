"""
FastAPI server for the IWFM web viewer.
"""

from __future__ import annotations

import logging
import webbrowser
from pathlib import Path
from threading import Timer
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from pyiwfm.visualization.webapi.config import ViewerSettings, model_state
from pyiwfm.visualization.webapi.routes import (
    budgets_router,
    export_router,
    groundwater_router,
    lakes_router,
    mesh_router,
    model_router,
    observations_router,
    properties_router,
    results_router,
    rootzone_router,
    slices_router,
    small_watersheds_router,
    streams_router,
)

if TYPE_CHECKING:
    from pyiwfm.core.model import IWFMModel

logger = logging.getLogger(__name__)


def create_app(
    model: IWFMModel | None = None,
    settings: ViewerSettings | None = None,
) -> FastAPI:
    """
    Create the FastAPI application.

    Parameters
    ----------
    model : IWFMModel, optional
        Pre-loaded IWFM model
    settings : ViewerSettings, optional
        Viewer settings

    Returns
    -------
    FastAPI
        Configured FastAPI application
    """
    if settings is None:
        settings = ViewerSettings()

    app = FastAPI(
        title=settings.title,
        description="IWFM Model Web Viewer API",
        version="1.0.0",
        docs_url="/api/docs" if settings.debug else None,
        redoc_url="/api/redoc" if settings.debug else None,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(model_router)
    app.include_router(mesh_router)
    app.include_router(properties_router)
    app.include_router(slices_router)
    app.include_router(streams_router)
    app.include_router(results_router)
    app.include_router(budgets_router)
    app.include_router(observations_router)
    app.include_router(groundwater_router)
    app.include_router(rootzone_router)
    app.include_router(lakes_router)
    app.include_router(export_router)
    app.include_router(small_watersheds_router)

    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
    else:

        @app.get("/")
        def root() -> dict:
            return {
                "message": "IWFM Viewer API",
                "docs": "/api/docs",
                "note": "Frontend not built. Run 'npm run build' in frontend/",
            }

    if model is not None:
        model_state.set_model(model)

    return app


def launch_viewer(
    model: IWFMModel,
    host: str = "127.0.0.1",
    port: int = 8080,
    title: str = "IWFM Viewer",
    open_browser: bool = True,
    debug: bool = False,
    crs: str = "+proj=utm +zone=10 +datum=NAD83 +units=us-ft +no_defs",
    no_cache: bool = False,
    rebuild_cache: bool = False,
) -> None:
    """
    Launch the web viewer server.

    Parameters
    ----------
    model : IWFMModel
        The model to visualize
    host : str
        Server host address
    port : int
        Server port
    title : str
        Application title
    open_browser : bool
        Whether to open browser on start
    debug : bool
        Enable debug mode
    crs : str
        Source coordinate reference system (default: UTM Zone 10N, NAD83, US survey feet)
    no_cache : bool
        Disable SQLite cache layer
    rebuild_cache : bool
        Force rebuild of SQLite cache
    """
    try:
        import uvicorn
    except ImportError as e:
        raise ImportError(
            "uvicorn is required for the web viewer. Install with: pip install uvicorn[standard]"
        ) from e

    settings = ViewerSettings(
        host=host,
        port=port,
        title=title,
        open_browser=open_browser,
        debug=debug,
    )

    model_state.set_model(
        model, crs=crs, no_cache=no_cache, rebuild_cache=rebuild_cache,
    )
    app = create_app(model=model, settings=settings)

    url = f"http://{host}:{port}"
    logger.info("Starting server at %s", url)

    if open_browser:
        Timer(1.5, lambda: webbrowser.open(url)).start()

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="debug" if debug else "info",
    )
