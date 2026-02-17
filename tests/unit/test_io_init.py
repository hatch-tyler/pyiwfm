"""Tests for io/__init__.py import fallback branches.

Covers the except ImportError branches for:
- DSS (optional dependency)
"""

from __future__ import annotations

import importlib
import sys
from unittest.mock import patch


class TestDSSImportFallback:
    """Test DSS import fallback."""

    def test_dss_import_fallback(self) -> None:
        """Force ImportError for dss -> _dss_exports is empty."""
        blocked = {"pyiwfm.io.dss": None}
        with patch.dict(sys.modules, blocked):
            sys.modules.pop("pyiwfm.io", None)
            import pyiwfm.io as io_mod

            importlib.reload(io_mod)
            assert io_mod._dss_exports == []

        sys.modules.pop("pyiwfm.io", None)
