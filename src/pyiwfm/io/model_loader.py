"""
Complete IWFM Model Loader.

This module provides a unified entry point for loading a complete IWFM model
from simulation and preprocessor files. It handles the hierarchical file
structure where the simulation main file references component main files,
which in turn reference sub-files for specific data.

The loader supports two simulation file formats:
- Description-based format (pyiwfm writer format with ``/ KEYWORD`` suffixes)
- Positional sequential format (native IWFM Fortran format)

Includes optional comment preservation for round-trip file operations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyiwfm.core.model import IWFMModel
    from pyiwfm.io.comment_extractor import CommentExtractor
    from pyiwfm.io.comment_metadata import CommentMetadata
    from pyiwfm.io.simulation import SimulationConfig

logger = logging.getLogger(__name__)


@dataclass
class ModelLoadResult:
    """Result of loading an IWFM model.

    Attributes:
        model: The loaded IWFMModel instance (or None if loading failed)
        simulation_config: Parsed simulation configuration
        errors: Dictionary of component-level errors encountered
        warnings: List of non-fatal warnings
    """

    model: IWFMModel | None = None
    simulation_config: SimulationConfig | None = None
    errors: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """Whether the model was loaded successfully."""
        return self.model is not None

    @property
    def has_errors(self) -> bool:
        """Whether any component errors occurred."""
        return len(self.errors) > 0


class CompleteModelLoader:
    """Load a complete IWFM model from simulation and preprocessor files.

    This class provides a high-level API for loading IWFM models that
    automatically handles:
    - Detection of simulation file format (description-based vs positional)
    - Hierarchical file resolution (main files -> sub-files)
    - Component loading with graceful error handling
    - Metadata collection from all loaded components

    Example::

        loader = CompleteModelLoader(
            simulation_file="Simulation/C2VSimFG.in",
            preprocessor_file="Preprocessor/C2VSimFG_PreProcessor.in",
        )
        result = loader.load()
        if result.success:
            model = result.model
            print(f"Loaded {model.n_nodes} nodes")
    """

    def __init__(
        self,
        simulation_file: Path | str,
        preprocessor_file: Path | str | None = None,
        use_positional_format: bool | None = None,
    ) -> None:
        """Initialize the model loader.

        Args:
            simulation_file: Path to the simulation main input file
            preprocessor_file: Path to the preprocessor main input file.
                If None, the loader will try to find it from the simulation
                config or use binary preprocessor output.
            use_positional_format: Force positional format reading (True),
                description-based reading (False), or auto-detect (None).
        """
        self.simulation_file = Path(simulation_file)
        self.preprocessor_file = Path(preprocessor_file) if preprocessor_file else None
        self.use_positional_format = use_positional_format
        self.base_dir = self.simulation_file.parent

    def load(self) -> ModelLoadResult:
        """Load the complete IWFM model.

        Returns:
            ModelLoadResult with the loaded model and any errors
        """
        result = ModelLoadResult()

        # Step 1: Read simulation configuration
        try:
            sim_config = self._read_simulation_config()
            result.simulation_config = sim_config
        except Exception as e:
            result.errors["simulation"] = str(e)
            logger.error("Failed to read simulation config: %s", e)
            return result

        # Step 2: Resolve preprocessor file if not provided
        pp_file = self.preprocessor_file
        if pp_file is None and sim_config.preprocessor_file:
            pp_file = self._resolve_path(sim_config.preprocessor_file)
            if not pp_file.exists():
                result.warnings.append(f"Preprocessor file not found: {pp_file}")
                pp_file = None

        if pp_file is None:
            result.errors["preprocessor"] = "No preprocessor file specified or found"
            return result

        # Step 3: Load model using from_simulation_with_preprocessor
        try:
            from pyiwfm.core.model import IWFMModel

            model = IWFMModel.from_simulation_with_preprocessor(self.simulation_file, pp_file)
            result.model = model

            # Collect any component-level errors from metadata
            for key, value in model.metadata.items():
                if key.endswith("_load_error"):
                    component = key.replace("_load_error", "")
                    result.errors[component] = str(value)
                    logger.warning("Component %s load error: %s", component, value)

        except Exception as e:
            result.errors["model"] = str(e)
            logger.error("Failed to load model: %s", e)

        return result

    def load_model(self) -> IWFMModel:
        """Load the complete model, raising on failure.

        Returns:
            IWFMModel instance

        Raises:
            RuntimeError: If model loading fails
        """
        result = self.load()
        if not result.success:
            errors = "; ".join(f"{k}: {v}" for k, v in result.errors.items())
            raise RuntimeError(f"Failed to load IWFM model: {errors}")
        return result.model  # type: ignore[return-value]

    def _read_simulation_config(self) -> SimulationConfig:
        """Read simulation configuration, auto-detecting format."""
        if self.use_positional_format is True:
            return self._read_positional()
        elif self.use_positional_format is False:
            return self._read_description_based()
        else:
            return self._read_auto_detect()

    def _read_positional(self) -> SimulationConfig:
        """Read using positional IWFM format."""
        from pyiwfm.io.simulation import IWFMSimulationReader

        reader = IWFMSimulationReader()
        return reader.read(self.simulation_file, base_dir=self.base_dir)

    def _read_description_based(self) -> SimulationConfig:
        """Read using description-based format."""
        from pyiwfm.io.simulation import SimulationReader

        reader = SimulationReader()
        return reader.read(self.simulation_file)

    def _read_auto_detect(self) -> SimulationConfig:
        """Auto-detect format and read simulation config.

        Heuristic: If the file has numbered description comments
        (``/ 1:``, ``/ 2:``, etc.) or IWFM keywords (``/ BDT``,
        ``/ UNITT``), use description-based reader. Otherwise, use
        positional reader.
        """
        try:
            # Try description-based reader first
            config = self._read_description_based()

            # Check if we got meaningful data
            has_dates = config.start_date.year != 2000 or config.end_date.year != 2000
            has_files = any(
                [
                    config.groundwater_file,
                    config.streams_file,
                    config.rootzone_file,
                ]
            )

            if has_dates or has_files:
                return config

            # Fall back to positional reader
            return self._read_positional()

        except Exception:
            # If description-based fails, try positional
            return self._read_positional()

    def _resolve_path(self, filepath: Path | str) -> Path:
        """Resolve a file path relative to the base directory."""
        path = Path(filepath)
        if path.is_absolute():
            return path
        return self.base_dir / path


def load_complete_model(
    simulation_file: Path | str,
    preprocessor_file: Path | str | None = None,
    use_positional_format: bool | None = None,
) -> IWFMModel:
    """Load a complete IWFM model from simulation and preprocessor files.

    This is a convenience function that creates a CompleteModelLoader
    and loads the model, raising on failure.

    Args:
        simulation_file: Path to the simulation main input file
        preprocessor_file: Path to the preprocessor main input file
        use_positional_format: Force format detection

    Returns:
        IWFMModel instance

    Raises:
        RuntimeError: If model loading fails
    """
    loader = CompleteModelLoader(simulation_file, preprocessor_file, use_positional_format)
    return loader.load_model()


# =============================================================================
# Comment-Aware Model Loading
# =============================================================================


@dataclass
class ModelLoadResultWithComments(ModelLoadResult):
    """Result of loading an IWFM model with comment preservation.

    Extends ModelLoadResult to include extracted comment metadata
    for all loaded files.

    Attributes:
        comment_metadata: Dictionary mapping file type to CommentMetadata.
            Keys include "preprocessor_main", "simulation_main", "gw_main", etc.
    """

    comment_metadata: dict[str, CommentMetadata] = field(default_factory=dict)

    def get_file_comments(self, file_type: str) -> CommentMetadata | None:
        """Get comment metadata for a specific file type.

        Args:
            file_type: File type key (e.g., "preprocessor_main").

        Returns:
            CommentMetadata if available, None otherwise.
        """
        return self.comment_metadata.get(file_type)


class CommentAwareModelLoader(CompleteModelLoader):
    """Load an IWFM model with comment preservation.

    Extends CompleteModelLoader to extract and preserve comments
    from all loaded input files. The preserved comments can be
    used for round-trip file operations.

    Example::

        loader = CommentAwareModelLoader(
            simulation_file="Simulation/Main.in",
            preprocessor_file="Preprocessor/Main.in",
        )
        result = loader.load()
        if result.success:
            model = result.model
            comments = result.comment_metadata
            # Later: write model with preserved comments
            write_model_with_comments(model, "output/", comments)
    """

    def __init__(
        self,
        simulation_file: Path | str,
        preprocessor_file: Path | str | None = None,
        use_positional_format: bool | None = None,
        preserve_comments: bool = True,
    ) -> None:
        """Initialize the comment-aware model loader.

        Args:
            simulation_file: Path to the simulation main input file
            preprocessor_file: Path to the preprocessor main input file
            use_positional_format: Force positional format reading
            preserve_comments: Whether to extract and preserve comments
        """
        super().__init__(simulation_file, preprocessor_file, use_positional_format)
        self.preserve_comments = preserve_comments

    def load(self) -> ModelLoadResultWithComments:
        """Load the complete IWFM model with comment metadata.

        Returns:
            ModelLoadResultWithComments with model, errors, and comments
        """
        # First load the model using parent implementation
        base_result = super().load()

        # Create extended result
        result = ModelLoadResultWithComments(
            model=base_result.model,
            simulation_config=base_result.simulation_config,
            errors=base_result.errors,
            warnings=base_result.warnings,
        )

        # Extract comments if requested and model loaded successfully
        if self.preserve_comments and base_result.success:
            result.comment_metadata = self._extract_all_comments(base_result)

        return result

    def _extract_all_comments(self, base_result: ModelLoadResult) -> dict[str, CommentMetadata]:
        """Extract comments from all model files.

        Args:
            base_result: The base model load result.

        Returns:
            Dictionary mapping file type to CommentMetadata.
        """
        from pyiwfm.io.comment_extractor import CommentExtractor

        comments: dict[str, CommentMetadata] = {}
        extractor = CommentExtractor()

        # Extract from simulation main file
        try:
            sim_comments = extractor.extract(self.simulation_file)
            comments["simulation_main"] = sim_comments
            logger.debug("Extracted comments from simulation main file")
        except Exception as e:
            logger.warning(f"Failed to extract simulation comments: {e}")

        # Extract from preprocessor file
        pp_file = self.preprocessor_file
        if pp_file is None and base_result.simulation_config:
            pp_raw = base_result.simulation_config.preprocessor_file
            if pp_raw is not None:
                pp_file = self._resolve_path(pp_raw)

        if pp_file and pp_file.exists():
            try:
                pp_comments = extractor.extract(pp_file)
                comments["preprocessor_main"] = pp_comments
                logger.debug("Extracted comments from preprocessor main file")
            except Exception as e:
                logger.warning(f"Failed to extract preprocessor comments: {e}")

        # Extract from component files if simulation config is available
        if base_result.simulation_config:
            self._extract_component_comments(base_result.simulation_config, extractor, comments)

        return comments

    def _extract_component_comments(
        self,
        sim_config: SimulationConfig,
        extractor: CommentExtractor,
        comments: dict[str, CommentMetadata],
    ) -> None:
        """Extract comments from component files.

        Args:
            sim_config: Simulation configuration with file paths.
            extractor: CommentExtractor instance.
            comments: Dictionary to populate with extracted comments.
        """
        # Map config attributes to file type keys
        component_files = {
            "gw_main": getattr(sim_config, "groundwater_file", None),
            "stream_main": getattr(sim_config, "streams_file", None),
            "lake_main": getattr(sim_config, "lake_file", None),
            "rootzone_main": getattr(sim_config, "rootzone_file", None),
            "swshed_main": getattr(sim_config, "small_watershed_file", None),
            "unsatzone_main": getattr(sim_config, "unsaturated_zone_file", None),
        }

        for file_type, filepath in component_files.items():
            if filepath:
                resolved = self._resolve_path(filepath)
                if resolved.exists():
                    try:
                        file_comments = extractor.extract(resolved)
                        comments[file_type] = file_comments
                        logger.debug(f"Extracted comments from {file_type}")
                    except Exception as e:
                        logger.warning(f"Failed to extract comments from {file_type}: {e}")


def load_model_with_comments(
    simulation_file: Path | str,
    preprocessor_file: Path | str | None = None,
    use_positional_format: bool | None = None,
) -> tuple[IWFMModel, dict[str, CommentMetadata]]:
    """Load an IWFM model with comment preservation.

    This is a convenience function that loads a model and extracts
    comments from all input files for round-trip preservation.

    Args:
        simulation_file: Path to the simulation main input file
        preprocessor_file: Path to the preprocessor main input file
        use_positional_format: Force positional format reading

    Returns:
        Tuple of (IWFMModel, comment_metadata_dict)

    Raises:
        RuntimeError: If model loading fails

    Example::

        model, comments = load_model_with_comments("Simulation/Main.in")
        # Modify model...
        model.nodes.add_node(...)
        # Write back with preserved comments
        write_model_with_comments(model, "output/", comment_metadata=comments)
    """
    loader = CommentAwareModelLoader(
        simulation_file,
        preprocessor_file,
        use_positional_format,
        preserve_comments=True,
    )
    result = loader.load()

    if not result.success:
        errors = "; ".join(f"{k}: {v}" for k, v in result.errors.items())
        raise RuntimeError(f"Failed to load IWFM model: {errors}")

    return result.model, result.comment_metadata  # type: ignore[return-value]
