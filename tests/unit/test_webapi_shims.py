"""Tests verifying that backward-compat shims re-export the canonical classes."""

from __future__ import annotations


def test_head_loader_shim() -> None:
    from pyiwfm.io.head_loader import LazyHeadDataLoader as Original
    from pyiwfm.visualization.webapi.head_loader import LazyHeadDataLoader

    assert LazyHeadDataLoader is Original


def test_hydrograph_reader_shim() -> None:
    from pyiwfm.io.hydrograph_reader import IWFMHydrographReader as Original
    from pyiwfm.visualization.webapi.hydrograph_reader import IWFMHydrographReader

    assert IWFMHydrographReader is Original


def test_hydrograph_loader_shim() -> None:
    from pyiwfm.io.hydrograph_loader import LazyHydrographDataLoader as Original
    from pyiwfm.visualization.webapi.hydrograph_loader import LazyHydrographDataLoader

    assert LazyHydrographDataLoader is Original


def test_area_loader_lazy_shim() -> None:
    from pyiwfm.io.area_loader import LazyAreaDataLoader as Original
    from pyiwfm.visualization.webapi.area_loader import LazyAreaDataLoader

    assert LazyAreaDataLoader is Original


def test_area_loader_manager_shim() -> None:
    from pyiwfm.io.area_loader import AreaDataManager as Original
    from pyiwfm.visualization.webapi.area_loader import AreaDataManager

    assert AreaDataManager is Original


def test_cache_builder_shim() -> None:
    from pyiwfm.io.cache_builder import SqliteCacheBuilder as Original
    from pyiwfm.visualization.webapi.cache_builder import SqliteCacheBuilder

    assert SqliteCacheBuilder is Original


def test_cache_builder_get_source_mtimes_shim() -> None:
    from pyiwfm.io.cache_builder import get_source_mtimes as original
    from pyiwfm.visualization.webapi.cache_builder import get_source_mtimes

    assert get_source_mtimes is original


def test_cache_builder_is_cache_stale_shim() -> None:
    from pyiwfm.io.cache_builder import is_cache_stale as original
    from pyiwfm.visualization.webapi.cache_builder import is_cache_stale

    assert is_cache_stale is original


def test_cache_loader_shim() -> None:
    from pyiwfm.io.cache_loader import SqliteCacheLoader as Original
    from pyiwfm.visualization.webapi.cache_loader import SqliteCacheLoader

    assert SqliteCacheLoader is Original
