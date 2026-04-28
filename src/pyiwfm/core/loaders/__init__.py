"""
Direct loader functions for :class:`~pyiwfm.core.model.IWFMModel`.

The corresponding ``IWFMModel.from_*`` classmethods are thin dispatchers
over these functions; advanced callers can use the functions directly
when they need to plug into an alternate orchestration (e.g., parallel
multi-model loading) or want to be explicit about which loader path is
in play. The classmethod surface is unchanged.

In v1.x the bodies of these loaders lived inside ``core/model.py``, which
had grown to ~2,500 lines. v2.0 splits them out so the model module
stays focused on the dataclass + I/O methods.

Public functions (mirror the v1.x classmethod names):

- :func:`load_from_preprocessor` — `IWFMModel.from_preprocessor`
- :func:`load_from_preprocessor_binary` — `IWFMModel.from_preprocessor_binary`
- :func:`load_from_simulation` — `IWFMModel.from_simulation`
- :func:`load_from_simulation_with_preprocessor` — `IWFMModel.from_simulation_with_preprocessor`
- :func:`load_from_hdf5` — `IWFMModel.from_hdf5`
"""

from __future__ import annotations

from pyiwfm.core.loaders.from_hdf5 import load_from_hdf5
from pyiwfm.core.loaders.from_preprocessor import load_from_preprocessor
from pyiwfm.core.loaders.from_preprocessor_binary import load_from_preprocessor_binary
from pyiwfm.core.loaders.from_simulation import load_from_simulation
from pyiwfm.core.loaders.from_simulation_with_preprocessor import (
    load_from_simulation_with_preprocessor,
)

__all__ = [
    "load_from_hdf5",
    "load_from_preprocessor",
    "load_from_preprocessor_binary",
    "load_from_simulation",
    "load_from_simulation_with_preprocessor",
]
