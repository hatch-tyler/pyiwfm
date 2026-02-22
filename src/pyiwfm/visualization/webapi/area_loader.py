"""Backward-compatibility shim — module moved to :mod:`pyiwfm.io.area_loader`."""

from pyiwfm.io.area_loader import AreaDataManager, LazyAreaDataLoader

__all__ = ["LazyAreaDataLoader", "AreaDataManager"]
