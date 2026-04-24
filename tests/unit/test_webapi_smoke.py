"""Smoke tests for the FastAPI webapi viewer.

Exercises the backend routes that the 6 frontend tabs (Overview, 3D Mesh,
Results Map, Diagnostics, Budgets, Z-Budgets) depend on, with a specific
focus on the handlers that touch pyvista/vtk at runtime. The existing
per-route tests in `test_webapi_routes_*.py` mock pyvista/vtk via MagicMock
and therefore never exercise the real dependency chain; this file fills
that gap.

Runs on the dedicated `webapi-smoke` CI job which installs
`pyiwfm[webapi,dev]` across the full Python x OS matrix. On legs without
`[webapi]` installed, pyvista/vtk-dependent tests auto-skip via
`pytest.importorskip`.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="FastAPI not available")


BASIC_ROUTES_EXPECT_200 = [
    "/api/model/info",
    "/api/model/summary",
    "/api/model/bounds",
    "/api/mesh/json",
    "/api/mesh/nodes",
    "/api/mesh/subregions",
    "/api/mesh/geojson",
    "/api/streams",
    "/api/streams/geojson",
    "/api/budgets/types",
    "/api/budgets/glossary",
    "/api/zbudgets/types",
    "/api/zbudgets/glossary",
    "/api/properties",
]


@pytest.mark.parametrize("route", BASIC_ROUTES_EXPECT_200)
def test_basic_route_returns_200(webapi_client, route):
    """GET each basic route with a mock model loaded; expect 200."""
    response = webapi_client.get(route)
    assert response.status_code == 200, (
        f"{route} returned {response.status_code}: {response.text[:500]}"
    )


# ---------------------------------------------------------------------------
# pyvista / vtk-dependent routes
#
# These handlers invoke pyvista or vtk at runtime. They are the ones most
# likely to surface dependency conflicts that unit tests with MagicMocks
# can't catch (e.g. pyvista 0.47 API changes, vtk wheel issues on new
# Python versions).
# ---------------------------------------------------------------------------

pv = pytest.importorskip("pyvista", reason="pyvista not installed")
vtk = pytest.importorskip("vtk", reason="vtk not installed")


def test_mesh_3d_vtu_smoke(webapi_client):
    """GET /api/mesh renders via vtk.vtkXMLUnstructuredGridWriter."""
    response = webapi_client.get("/api/mesh")
    assert response.status_code == 200, response.text[:500]
    assert len(response.content) > 0


def test_mesh_surface_vtu_smoke(webapi_client):
    """GET /api/mesh/surface renders surface VTU via vtk."""
    response = webapi_client.get("/api/mesh/surface")
    assert response.status_code == 200, response.text[:500]
    assert len(response.content) > 0


def test_streams_vtp_smoke(webapi_client):
    """GET /api/streams/vtp builds polylines via raw vtk APIs."""
    response = webapi_client.get("/api/streams/vtp")
    assert response.status_code == 200, response.text[:500]
    assert len(response.content) > 0


def test_slice_smoke(webapi_client):
    """GET /api/slice?x=50 slices the 3D mesh via pyvista."""
    response = webapi_client.get("/api/slice?x=50")
    assert response.status_code == 200, response.text[:500]
    assert len(response.content) > 0
