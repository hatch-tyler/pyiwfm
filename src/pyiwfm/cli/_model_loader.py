"""
Shared model loading logic for CLI subcommands.

Extracted from ``cli/viewer.py`` so both ``viewer`` and ``export``
can use the same loader without depending on the Trame viewer module.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyiwfm.core.model import IWFMModel

logger = logging.getLogger(__name__)


def _resolve_path(base: Path, user_path: Path) -> Path:
    """Return *user_path* resolved against *base* if it is relative."""
    if user_path.is_absolute():
        return user_path
    return base / user_path


def load_model(
    model_dir: Path,
    preprocessor_file: Path | None = None,
    simulation_file: Path | None = None,
) -> IWFMModel:
    """
    Load an IWFM model with automatic fallback.

    Priority:
    1. Explicit simulation file (``from_simulation`` -- loads full model)
    2. Explicit preprocessor file (``from_preprocessor`` -- mesh/stratigraphy only)
    3. Auto-detected: simulation + preprocessor -> ``from_simulation_with_preprocessor``
    4. Auto-detected: simulation only -> ``from_simulation``
    5. Auto-detected: preprocessor only -> ``from_preprocessor``
    6. Auto-detected binary file (``from_preprocessor_binary``)

    The simulation path is preferred because it loads all components including
    groundwater (wells, aquifer parameters, initial heads), stream reaches,
    lakes, and root zone -- not just the mesh and stratigraphy.
    """
    from pyiwfm.cli._model_finder import find_model_files
    from pyiwfm.core.model import IWFMModel

    # --- Explicit file provided -------------------------------------------
    if simulation_file is not None:
        sim_path = _resolve_path(model_dir, simulation_file)
        if preprocessor_file is not None:
            pp_path = _resolve_path(model_dir, preprocessor_file)
            logger.info(
                "Loading full model from simulation + preprocessor: %s, %s",
                sim_path,
                pp_path,
            )
            return IWFMModel.from_simulation_with_preprocessor(sim_path, pp_path)
        logger.info("Loading full model from simulation file: %s", sim_path)
        return IWFMModel.from_simulation(sim_path)

    if preprocessor_file is not None:
        pp_path = _resolve_path(model_dir, preprocessor_file)
        logger.info("Loading model from preprocessor file: %s", pp_path)
        return IWFMModel.from_preprocessor(pp_path)

    # --- Auto-detect ------------------------------------------------------
    files = find_model_files(model_dir)

    sim_file = files["simulation_main"]
    pp_file = files["preprocessor_main"]

    # Prefer simulation (full model) over preprocessor (mesh only)
    if sim_file and pp_file:
        logger.info(
            "Auto-detected simulation + preprocessor files: %s, %s",
            sim_file,
            pp_file,
        )
        try:
            return IWFMModel.from_simulation_with_preprocessor(sim_file, pp_file)
        except Exception as exc:
            logger.warning("Failed to load with both files: %s", exc)
            logger.info("Falling back to simulation-only loading...")

    if sim_file:
        logger.info("Auto-detected simulation file: %s", sim_file)
        try:
            return IWFMModel.from_simulation(sim_file)
        except Exception as exc:
            logger.warning("Failed to load from simulation file: %s", exc)
            logger.info("Falling back to preprocessor or binary...")

    if pp_file:
        logger.info("Auto-detected preprocessor file: %s", pp_file)
        return IWFMModel.from_preprocessor(pp_file)

    if files["preprocessor_binary"]:
        logger.info("Auto-detected binary file: %s", files["preprocessor_binary"])
        return IWFMModel.from_preprocessor_binary(files["preprocessor_binary"])

    raise FileNotFoundError(
        f"No IWFM model files found in {model_dir}. "
        "Expected to find a Simulation_MAIN.IN or PreProcessor_MAIN.IN file."
    )
