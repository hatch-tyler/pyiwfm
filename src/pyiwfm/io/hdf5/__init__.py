"""HDF5 archive format for IWFM models.

This package provides readers, writers, and conveniences for the
HDF5 archive format that pyiwfm uses to round-trip a complete IWFM
model (mesh, stratigraphy, components, time series) through a
single hierarchical file. h5py is required.

Public API:

- :class:`HDF5ModelReader` — read a complete model from HDF5.
- :class:`HDF5ModelWriter` — write a complete model to HDF5.
- :func:`read_model_hdf5` / :func:`write_model_hdf5` — function-style
  conveniences over the classes.
"""

from __future__ import annotations

from pyiwfm.io.hdf5.model import (
    HDF5ModelReader,
    HDF5ModelWriter,
    read_model_hdf5,
    write_model_hdf5,
)

__all__ = [
    "HDF5ModelReader",
    "HDF5ModelWriter",
    "read_model_hdf5",
    "write_model_hdf5",
]
