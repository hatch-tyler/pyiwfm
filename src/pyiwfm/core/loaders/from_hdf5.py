"""Direct loader for ``IWFMModel.from_hdf5``.

In v1.x this body lived as a classmethod in ``core/model.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from pyiwfm.core.model import IWFMModel


def load_from_hdf5(hdf5_file: Path | str) -> IWFMModel:
    """Load a model from HDF5 output file.

    This loads a complete model that was previously saved to HDF5 format
    using ``IWFMModel.to_hdf5()`` or ``write_model_hdf5()``.

    Args:
        hdf5_file: Path to the HDF5 file

    Returns:
        Loaded IWFMModel instance

    Example:
        >>> model = load_from_hdf5("model.h5")
    """
    from pyiwfm.io.hdf5 import read_model_hdf5

    return read_model_hdf5(hdf5_file)
