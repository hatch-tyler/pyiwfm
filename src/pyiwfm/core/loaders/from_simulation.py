"""Direct loader for ``IWFMModel.from_simulation``.

In v1.x this body lived as a classmethod in ``core/model.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from pyiwfm.core.model import IWFMModel


def load_from_simulation(simulation_file: Path | str) -> IWFMModel:
    """Load a complete IWFM model from a simulation main input file.

    Delegates to :class:`~pyiwfm.io.model_loader.CompleteModelLoader`
    which auto-detects the simulation file format and loads all
    components (mesh, stratigraphy, groundwater, streams, lakes,
    root zone, etc.).

    Args:
        simulation_file: Path to the simulation main input file

    Returns:
        IWFMModel instance with all components loaded

    Example:
        >>> model = load_from_simulation("Simulation/Simulation.in")
        >>> print(f"Stream nodes: {len(model.streams.nodes)}")
    """
    from pyiwfm.io.model_loader import load_complete_model

    return load_complete_model(simulation_file)
