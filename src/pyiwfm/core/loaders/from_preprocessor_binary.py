"""Direct loader for ``IWFMModel.from_preprocessor_binary``.

In v1.x this body lived as a classmethod in ``core/model.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyiwfm.core.model import IWFMModel


def load_from_preprocessor_binary(
    binary_file: Path | str,
    name: str = "",
) -> IWFMModel:
    """Load a model from the native IWFM PreProcessor binary output.

    The preprocessor binary file (``ACCESS='STREAM'``) contains mesh,
    stratigraphy, stream/lake connectors, and component data compiled
    by the IWFM PreProcessor.

    Args:
        binary_file: Path to the preprocessor binary output file
        name: Model name (optional, defaults to file stem)

    Returns:
        IWFMModel with mesh, stratigraphy, streams, and lakes loaded

    Example:
        >>> model = load_from_preprocessor_binary("PreprocessorOut.bin")
        >>> print(f"Loaded {model.n_nodes} nodes, {model.n_layers} layers")
    """
    # Re-import these helpers from ``core.model`` (not ``core.model_factory``)
    # so tests that patch ``pyiwfm.core.model._binary_data_to_model`` see the
    # patched version at call time. Lazy import avoids a top-level cycle
    # (``core.model`` itself imports from ``core.loaders`` for the dispatchers).
    from pyiwfm.core.model import (
        _binary_data_to_model,
        _resolve_stream_node_coordinates,
    )
    from pyiwfm.io.binary import PreprocessorBinaryReader

    binary_file = Path(binary_file)
    reader = PreprocessorBinaryReader()
    data = reader.read(binary_file)
    model = _binary_data_to_model(data, name=name or binary_file.stem)
    model.metadata["binary_file"] = str(binary_file)
    _resolve_stream_node_coordinates(model)
    return model
