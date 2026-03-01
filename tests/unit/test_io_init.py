"""Tests for io/__init__.py lazy import mechanism.

Covers:
- DSS optional dependency fallback (excluded from __all__ when unavailable)
- Lazy __getattr__ attribute resolution
"""

from __future__ import annotations

import importlib
import sys
from unittest.mock import patch


class TestDSSImportFallback:
    """Test DSS import fallback."""

    def test_dss_import_fallback(self) -> None:
        """When dss is unavailable, DSS names are excluded from __all__."""
        blocked = {"pyiwfm.io.dss": None}
        with patch.dict(sys.modules, blocked):
            sys.modules.pop("pyiwfm.io", None)
            import pyiwfm.io as io_mod

            importlib.reload(io_mod)
            assert "DSSFile" not in io_mod.__all__
            assert "DSSPathname" not in io_mod.__all__

        sys.modules.pop("pyiwfm.io", None)


class TestLazyAttrResolution:
    """Test __getattr__ lazy import."""

    def test_unknown_attr_raises(self) -> None:
        """Accessing a non-existent name raises AttributeError."""
        import pyiwfm.io as io_mod

        try:
            _ = io_mod.no_such_attr_xyz_12345  # type: ignore[attr-defined]
            raise AssertionError("Should have raised AttributeError")
        except AttributeError:
            pass
