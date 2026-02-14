"""
API route modules for the FastAPI web viewer.
"""

from pyiwfm.visualization.webapi.routes.budgets import router as budgets_router
from pyiwfm.visualization.webapi.routes.export import router as export_router
from pyiwfm.visualization.webapi.routes.groundwater import router as groundwater_router
from pyiwfm.visualization.webapi.routes.lakes import router as lakes_router
from pyiwfm.visualization.webapi.routes.mesh import router as mesh_router
from pyiwfm.visualization.webapi.routes.model import router as model_router
from pyiwfm.visualization.webapi.routes.observations import router as observations_router
from pyiwfm.visualization.webapi.routes.properties import router as properties_router
from pyiwfm.visualization.webapi.routes.results import router as results_router
from pyiwfm.visualization.webapi.routes.rootzone import router as rootzone_router
from pyiwfm.visualization.webapi.routes.slices import router as slices_router
from pyiwfm.visualization.webapi.routes.small_watersheds import (
    router as small_watersheds_router,
)
from pyiwfm.visualization.webapi.routes.streams import router as streams_router

__all__ = [
    "model_router",
    "mesh_router",
    "properties_router",
    "slices_router",
    "streams_router",
    "results_router",
    "budgets_router",
    "observations_router",
    "groundwater_router",
    "rootzone_router",
    "lakes_router",
    "export_router",
    "small_watersheds_router",
]
