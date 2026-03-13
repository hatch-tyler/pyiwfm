"""Tests for the diagnostics API routes."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed")

from unittest.mock import patch


class TestDiagnosticsRoutes:
    """Test the diagnostics API endpoint functions."""

    def test_diagnostics_module_imports(self):
        """Verify the diagnostics route module can be imported."""
        from pyiwfm.visualization.webapi.routes import diagnostics  # noqa: F401

    def test_summary_endpoint_no_model(self):
        """Summary should handle missing model gracefully."""
        from pyiwfm.visualization.webapi.routes.diagnostics import _get_diagnostics_result

        with patch("pyiwfm.visualization.webapi.routes.diagnostics.model_state") as ms:
            ms.is_loaded = False
            ms._results_dir = None
            ms._diagnostics_result = None
            result = _get_diagnostics_result()
            assert result is None
