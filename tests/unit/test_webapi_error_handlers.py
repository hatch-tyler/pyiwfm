"""Tests for pyiwfm.visualization.webapi.error_handlers.

These verify the FastAPI app maps typed pyiwfm exceptions to
structured JSON 4xx responses rather than letting them escape as
default-handler 500s.
"""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from pyiwfm.core.exceptions import (  # noqa: E402
    ComponentError,
    ComponentLoadError,
    FileFormatError,
    MeshError,
)
from pyiwfm.visualization.webapi.error_handlers import register_exception_handlers  # noqa: E402


def _app_with_handler(exc: Exception) -> TestClient:
    """Build a tiny FastAPI app whose one route raises ``exc`` and
    register the pyiwfm exception handlers."""
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/boom")
    def boom() -> None:
        raise exc

    return TestClient(app)


def test_file_format_error_returns_400_with_line_number() -> None:
    client = _app_with_handler(FileFormatError("expected NNODES", line_number=42))
    resp = client.get("/boom")
    assert resp.status_code == 400
    body = resp.json()
    assert body["type"] == "FileFormatError"
    assert "expected NNODES" in body["error"]
    assert body["line_number"] == 42


def test_file_format_error_without_line_number_omits_field() -> None:
    client = _app_with_handler(FileFormatError("missing keyword"))
    resp = client.get("/boom")
    assert resp.status_code == 400
    body = resp.json()
    assert body["type"] == "FileFormatError"
    assert "line_number" not in body


def test_component_load_error_returns_422_with_component_and_source() -> None:
    client = _app_with_handler(
        ComponentLoadError(component_name="streams", source_file="/tmp/missing.dat")
    )
    resp = client.get("/boom")
    assert resp.status_code == 422
    body = resp.json()
    assert body["type"] == "ComponentLoadError"
    assert body["component"] == "streams"
    assert body["source_file"] == "/tmp/missing.dat"


def test_component_load_error_with_no_source_file() -> None:
    client = _app_with_handler(ComponentLoadError(component_name="lakes"))
    resp = client.get("/boom")
    assert resp.status_code == 422
    body = resp.json()
    assert body["component"] == "lakes"
    assert body["source_file"] is None


def test_pyiwfm_subclass_falls_through_to_400() -> None:
    """MeshError / ComponentError / etc. — the catch-all PyIWFMError
    handler should turn them into structured 400s."""
    client = _app_with_handler(MeshError("mesh has no nodes"))
    resp = client.get("/boom")
    assert resp.status_code == 400
    body = resp.json()
    assert body["type"] == "MeshError"
    assert "mesh has no nodes" in body["error"]


def test_component_error_returns_400() -> None:
    client = _app_with_handler(ComponentError("invalid bc_type"))
    resp = client.get("/boom")
    assert resp.status_code == 400
    body = resp.json()
    assert body["type"] == "ComponentError"
    assert "invalid bc_type" in body["error"]


def test_unknown_exception_still_returns_500() -> None:
    """A non-pyiwfm exception must NOT be caught — verify the FastAPI
    default handler returns 500 (so genuine bugs surface)."""
    client = _app_with_handler(RuntimeError("oops"))
    # Use raise_server_exceptions=False so TestClient doesn't re-raise
    # and instead returns the default-handler 500 response.
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/boom")
    def boom() -> None:
        raise RuntimeError("oops")

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get("/boom")
    assert resp.status_code == 500
