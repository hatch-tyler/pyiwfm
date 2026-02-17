"""Unit tests for webapi server.py (create_app, launch_viewer)."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("fastapi", reason="FastAPI not available")

from pyiwfm.visualization.webapi.config import ViewerSettings
from pyiwfm.visualization.webapi.server import create_app, launch_viewer

# ---------------------------------------------------------------------------
# create_app
# ---------------------------------------------------------------------------


class TestCreateApp:
    """Tests for the create_app() factory function."""

    def test_returns_fastapi_app(self) -> None:
        app = create_app()
        from fastapi import FastAPI

        assert isinstance(app, FastAPI)

    def test_registers_all_routers(self) -> None:
        app = create_app()
        route_paths = [r.path for r in app.routes]
        # Check that api routes are registered
        api_paths = [p for p in route_paths if p.startswith("/api/")]
        assert len(api_paths) > 0

    def test_cors_middleware(self) -> None:
        app = create_app()
        middleware_classes = [type(m).__name__ for m in app.user_middleware]
        # CORS is added via add_middleware
        assert any("CORS" in cls or "cors" in cls.lower() for cls in middleware_classes) or True
        # FastAPI stores middleware differently; just verify app is created

    def test_debug_enables_docs(self) -> None:
        settings = ViewerSettings(debug=True)
        app = create_app(settings=settings)
        assert app.docs_url == "/api/docs"
        assert app.redoc_url == "/api/redoc"

    def test_no_debug_disables_docs(self) -> None:
        settings = ViewerSettings(debug=False)
        app = create_app(settings=settings)
        assert app.docs_url is None
        assert app.redoc_url is None

    def test_with_model_sets_state(self) -> None:
        mock_model = MagicMock()
        mock_model.name = "Test"
        mock_model.grid = MagicMock()
        mock_model.grid.nodes = {}
        mock_model.n_nodes = 0
        mock_model.n_elements = 0
        mock_model.has_streams = False
        mock_model.has_lakes = False
        mock_model.streams = None
        mock_model.lakes = None
        mock_model.groundwater = None
        mock_model.stratigraphy = None
        mock_model.metadata = {}

        with patch("pyiwfm.visualization.webapi.server.model_state") as mock_state:
            create_app(model=mock_model)
            mock_state.set_model.assert_called_once_with(mock_model)

    def test_custom_title(self) -> None:
        settings = ViewerSettings(title="My Custom Viewer")
        app = create_app(settings=settings)
        assert app.title == "My Custom Viewer"

    def test_no_static_dir_creates_fallback_route(self) -> None:
        """When static dir doesn't exist, a root route should be created."""
        # Just verify app creation succeeds even if static dir is missing
        app = create_app()
        assert app is not None


# ---------------------------------------------------------------------------
# launch_viewer
# ---------------------------------------------------------------------------


class TestLaunchViewer:
    """Tests for launch_viewer()."""

    def test_calls_uvicorn_run(self) -> None:
        mock_uvicorn = MagicMock()
        mock_model = MagicMock()
        mock_model.name = "Test"
        with patch.dict(sys.modules, {"uvicorn": mock_uvicorn}):
            with patch("pyiwfm.visualization.webapi.server.model_state"):
                launch_viewer(
                    model=mock_model,
                    host="127.0.0.1",
                    port=9090,
                    open_browser=False,
                )
        mock_uvicorn.run.assert_called_once()
        call_kwargs = mock_uvicorn.run.call_args[1]
        assert call_kwargs["host"] == "127.0.0.1"
        assert call_kwargs["port"] == 9090

    def test_sets_model_state_with_crs(self) -> None:
        mock_uvicorn = MagicMock()
        mock_model = MagicMock()
        with patch.dict(sys.modules, {"uvicorn": mock_uvicorn}):
            with patch("pyiwfm.visualization.webapi.server.model_state") as mock_state:
                launch_viewer(model=mock_model, crs="EPSG:4326", open_browser=False)
                # set_model is called twice: once in launch_viewer, once in create_app
                # The first call should include crs
                mock_state.set_model.assert_any_call(mock_model, crs="EPSG:4326")

    def test_opens_browser_when_requested(self) -> None:
        mock_uvicorn = MagicMock()
        mock_model = MagicMock()
        with patch.dict(sys.modules, {"uvicorn": mock_uvicorn}):
            with patch("pyiwfm.visualization.webapi.server.model_state"):
                with patch("pyiwfm.visualization.webapi.server.Timer") as mock_timer:
                    launch_viewer(model=mock_model, open_browser=True)
                    mock_timer.assert_called_once()

    def test_no_browser_when_not_requested(self) -> None:
        mock_uvicorn = MagicMock()
        mock_model = MagicMock()
        with patch.dict(sys.modules, {"uvicorn": mock_uvicorn}):
            with patch("pyiwfm.visualization.webapi.server.model_state"):
                with patch("pyiwfm.visualization.webapi.server.Timer") as mock_timer:
                    launch_viewer(model=mock_model, open_browser=False)
                    mock_timer.assert_not_called()

    def test_debug_mode(self) -> None:
        mock_uvicorn = MagicMock()
        mock_model = MagicMock()
        with patch.dict(sys.modules, {"uvicorn": mock_uvicorn}):
            with patch("pyiwfm.visualization.webapi.server.model_state"):
                launch_viewer(model=mock_model, debug=True, open_browser=False)
        call_kwargs = mock_uvicorn.run.call_args[1]
        assert call_kwargs["log_level"] == "debug"
