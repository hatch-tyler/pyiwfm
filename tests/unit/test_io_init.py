"""Tests for io/__init__.py lazy import mechanism.

Covers:
- DSS is a first-class shipped submodule (always present in __all__)
- Lazy __getattr__ attribute resolution
"""

from __future__ import annotations


class TestDSSExports:
    """DSS ships with pyiwfm (bundled C library on Windows, built from
    source on Linux/macOS via ``dss-build/``). Its public names are
    eagerly re-exported just like any other submodule."""

    def test_dss_names_in_all(self) -> None:
        import pyiwfm.io as io_mod

        for name in (
            "DSSFile",
            "DSSPathname",
            "DSSPathnameTemplate",
            "DSSFileClass",
            "DSSTimeSeriesReader",
            "DSSTimeSeriesWriter",
            "HAS_DSS_LIBRARY",
            "read_timeseries_from_dss",
            "write_collection_to_dss",
            "write_timeseries_to_dss",
        ):
            assert name in io_mod.__all__, f"{name} missing from pyiwfm.io.__all__"
            assert hasattr(io_mod, name)


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

    def test_lazy_heavy_symbol_resolves(self) -> None:
        """Symbols mapped through ``_LAZY_IMPORTS`` (heavy submodules
        like budget/h5py) resolve on first attribute access and are
        cached in ``globals()`` afterward."""
        import pyiwfm.io as io_mod

        cls = io_mod.BudgetReader  # comes from pyiwfm.io.budget (heavy: h5py)
        assert cls.__name__ == "BudgetReader"
        # Cached on second access
        assert io_mod.BudgetReader is cls

    def test_submodule_fallback(self) -> None:
        """Accessing a submodule by name (e.g. for ``mock.patch`` strings)
        works through the PEP 562 fallback even though it isn't in the
        lazy mapping."""
        import types

        import pyiwfm.io as io_mod

        sim = io_mod.simulation
        assert isinstance(sim, types.ModuleType)
        assert sim.__name__ == "pyiwfm.io.simulation"
