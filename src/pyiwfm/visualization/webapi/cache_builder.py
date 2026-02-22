"""Backward-compatibility shim — module moved to :mod:`pyiwfm.io.cache_builder`."""

from pyiwfm.io.cache_builder import SqliteCacheBuilder, get_source_mtimes, is_cache_stale

__all__ = ["SqliteCacheBuilder", "get_source_mtimes", "is_cache_stale"]
