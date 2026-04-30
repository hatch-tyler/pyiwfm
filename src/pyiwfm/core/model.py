"""
IWFMModel class - main orchestrator for IWFM model components.

This module provides the central IWFMModel class that orchestrates all
model components including mesh, stratigraphy, groundwater, streams,
lakes, and root zone.

In v2.0 the per-classmethod loader bodies were moved to
:mod:`pyiwfm.core.loaders` to keep this module focused on the dataclass
+ instance methods. The ``IWFMModel.from_*`` classmethods are now thin
5-line dispatchers that delegate to ``pyiwfm.core.loaders.load_from_*``.
External call sites are unchanged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from pyiwfm.core.exceptions import (
    ComponentError,
    ComponentLoadError,
    MeshError,
    StratigraphyError,
    ValidationError,
)
from pyiwfm.core.loaders._common import (
    _COMPONENT_LOAD_EXCEPTIONS,
    StrictMode,
    _record_component_failure,
)
from pyiwfm.core.model_factory import (
    apply_kh_anomalies as _apply_kh_anomalies,
)
from pyiwfm.core.model_factory import (
    apply_parametric_grids as _apply_parametric_grids,
)
from pyiwfm.core.model_factory import (
    apply_parametric_subsidence as _apply_parametric_subsidence,
)
from pyiwfm.core.model_factory import (
    binary_data_to_model as _binary_data_to_model,
)
from pyiwfm.core.model_factory import (
    build_reaches_from_node_reach_ids as _build_reaches_from_node_reach_ids,
)
from pyiwfm.core.model_factory import (
    resolve_stream_node_coordinates as _resolve_stream_node_coordinates,
)

logger = logging.getLogger(__name__)

# Re-export the loader-helper symbols at module scope. v2.0 moved the
# canonical definitions to ``core.loaders._common`` and
# ``core.model_factory``, but tests and external code that imported them
# from ``core.model`` directly keep working through these aliases.
__all__ = [
    "IWFMModel",
    "_COMPONENT_LOAD_EXCEPTIONS",
    "_apply_kh_anomalies",
    "_apply_parametric_grids",
    "_apply_parametric_subsidence",
    "_binary_data_to_model",
    "_build_reaches_from_node_reach_ids",
    "_record_component_failure",
    "_resolve_stream_node_coordinates",
]


if TYPE_CHECKING:
    from numpy.typing import NDArray

    from pyiwfm.components.groundwater import AppGW
    from pyiwfm.components.lake import AppLake
    from pyiwfm.components.rootzone import RootZone
    from pyiwfm.components.small_watershed import AppSmallWatershed
    from pyiwfm.components.stream import AppStream
    from pyiwfm.components.unsaturated_zone import AppUnsatZone
    from pyiwfm.core.mesh import AppGrid
    from pyiwfm.core.stratigraphy import Stratigraphy
    from pyiwfm.io.supply_adjust import SupplyAdjustment


@dataclass
class IWFMModel:
    """
    The main IWFM model container class.

    This class orchestrates all model components and provides methods for
    reading, writing, and validating IWFM models. It mirrors the structure
    of IWFM's Package_Model.

    Attributes:
        name: Model name/identifier
        mesh: Finite element mesh (AppGrid)
        stratigraphy: Vertical layering structure
        groundwater: Groundwater component (AppGW) - wells, pumping, BCs, aquifer params
        streams: Stream network component (AppStream) - nodes, reaches, diversions, bypasses
        lakes: Lake component (AppLake) - lake definitions, elements, rating curves
        rootzone: Root zone component (RootZone) - crop types, soil params, land use
        small_watersheds: Small Watershed component (AppSmallWatershed)
        unsaturated_zone: Unsaturated Zone component (AppUnsatZone)
        supply_adjustment: Parsed supply adjustment specification data
        metadata: Additional model metadata
    """

    name: str
    mesh: AppGrid | None = None
    stratigraphy: Stratigraphy | None = None
    groundwater: AppGW | None = None
    streams: AppStream | None = None
    lakes: AppLake | None = None
    rootzone: RootZone | None = None
    small_watersheds: AppSmallWatershed | None = None
    unsaturated_zone: AppUnsatZone | None = None
    supply_adjustment: SupplyAdjustment | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    source_files: dict[str, Path] = field(default_factory=dict)

    # Components mutated since the last load/save. Populated by the
    # mutation helpers (set_aquifer_parameter, set_stratigraphy_*,
    # add_observation_well, etc.) so save_complete_model can warn or
    # log if a dirty component fails to write. Not user-facing state â€”
    # tests should reach into it directly.
    _dirty: set[str] = field(default_factory=set, repr=False)

    # ========================================================================
    # Class Methods for Loading Models
    # ========================================================================

    @classmethod
    def from_preprocessor(
        cls,
        pp_file: Path | str,
        load_streams: bool = True,
        load_lakes: bool = True,
        *,
        strict: StrictMode = False,
    ) -> IWFMModel:
        """
        Load a model from PreProcessor input files.

        This loads the model structure (mesh, stratigraphy) and optionally
        the stream and lake geometry from the preprocessor input file and
        all files referenced by it.

        Note: This creates a "partial" model with only the static geometry
        defined in the preprocessor. It does not include dynamic components
        like groundwater parameters, pumping, or root zone data which are
        defined in the simulation input files.

        Args:
            pp_file: Path to the main PreProcessor input file
            load_streams: If True, load stream network geometry
            load_lakes: If True, load lake geometry
            strict: ``False`` (default) records errors and continues;
                ``True`` raises :class:`ComponentLoadError` on first
                failure; ``"collect"`` raises a single
                :class:`ValidationError` at the end if any failed.
                See :func:`~pyiwfm.core.loaders.load_from_preprocessor`.

        Returns:
            IWFMModel instance with mesh, stratigraphy, and optionally
            streams/lakes geometry loaded

        Example:
            >>> model = IWFMModel.from_preprocessor("Preprocessor/Preprocessor.in")
            >>> print(f"Loaded {model.n_nodes} nodes, {model.n_elements} elements")
        """
        from pyiwfm.core.loaders import load_from_preprocessor

        return load_from_preprocessor(
            pp_file,
            load_streams=load_streams,
            load_lakes=load_lakes,
            strict=strict,
        )

    @classmethod
    def from_preprocessor_binary(
        cls,
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
            >>> model = IWFMModel.from_preprocessor_binary("PreprocessorOut.bin")
            >>> print(f"Loaded {model.n_nodes} nodes, {model.n_layers} layers")
        """
        from pyiwfm.core.loaders import load_from_preprocessor_binary

        return load_from_preprocessor_binary(binary_file, name=name)

    @classmethod
    def from_simulation(
        cls,
        simulation_file: Path | str,
    ) -> IWFMModel:
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
            >>> model = IWFMModel.from_simulation("Simulation/Simulation.in")
            >>> print(f"Stream nodes: {len(model.streams.nodes)}")
        """
        from pyiwfm.core.loaders import load_from_simulation

        return load_from_simulation(simulation_file)

    @classmethod
    def from_simulation_with_preprocessor(
        cls,
        simulation_file: Path | str,
        preprocessor_file: Path | str,
        load_timeseries: bool = False,
        *,
        strict: StrictMode = False,
    ) -> IWFMModel:
        """
        Load a complete IWFM model using both simulation and preprocessor files.

        This method first loads the mesh and stratigraphy from the preprocessor
        input file (ASCII format), then loads all dynamic components from the
        simulation input file and its referenced component files.

        Use this method when:
        - You have both preprocessor input files and simulation input files
        - You want to load from ASCII preprocessor files rather than binary
        - The binary file path in the simulation file is incorrect or missing

        Args:
            simulation_file: Path to the simulation main input file
            preprocessor_file: Path to the preprocessor main input file
            load_timeseries: If True, also load time series data (slower)
            strict: ``False`` (default) records errors and continues;
                ``True`` raises :class:`ComponentLoadError` on first
                failure; ``"collect"`` raises a single
                :class:`ValidationError` at the end if any failed.
                See :func:`~pyiwfm.core.loaders.load_from_simulation_with_preprocessor`.

        Returns:
            IWFMModel instance with all components loaded

        Example:
            >>> model = IWFMModel.from_simulation_with_preprocessor(
            ...     "Simulation/Simulation.in",
            ...     "Preprocessor/Preprocessor.in"
            ... )
        """
        from pyiwfm.core.loaders import load_from_simulation_with_preprocessor

        return load_from_simulation_with_preprocessor(
            simulation_file,
            preprocessor_file,
            load_timeseries=load_timeseries,
            strict=strict,
        )

    @classmethod
    def from_hdf5(cls, hdf5_file: Path | str) -> IWFMModel:
        """
        Load a model from HDF5 output file.

        This loads a complete model that was previously saved to HDF5 format
        using the to_hdf5() method or write_model_hdf5() function.

        Args:
            hdf5_file: Path to the HDF5 file

        Returns:
            Loaded IWFMModel instance

        Example:
            >>> model = IWFMModel.from_hdf5("model.h5")
        """
        from pyiwfm.core.loaders import load_from_hdf5

        return load_from_hdf5(hdf5_file)

    # ========================================================================
    # Instance Methods for Saving Models
    # ========================================================================

    def to_preprocessor(self, output_dir: Path | str) -> dict[str, Path]:
        """
        Write model to PreProcessor input files.

        Creates all preprocessor input files (nodes, elements, stratigraphy)
        in the specified output directory.

        Args:
            output_dir: Directory to write output files

        Returns:
            Dictionary mapping file type to output path
        """
        from pyiwfm.io.preprocessor import save_model_to_preprocessor

        config = save_model_to_preprocessor(self, output_dir, self.name)

        files: dict[str, Path] = {}
        if config.nodes_file:
            files["nodes"] = config.nodes_file
        if config.elements_file:
            files["elements"] = config.elements_file
        if config.stratigraphy_file:
            files["stratigraphy"] = config.stratigraphy_file
        if config.subregions_file:
            files["subregions"] = config.subregions_file

        return files

    def to_simulation(
        self,
        output_dir: Path | str,
        file_paths: dict[str, str] | None = None,
        ts_format: str = "text",
    ) -> dict[str, Path]:
        """
        Write complete model to simulation input files.

        Creates all input files required for an IWFM simulation, including
        preprocessor files, component files, and the simulation control file.

        Args:
            output_dir: Directory to write output files
            file_paths: Optional dict of {file_key: relative_path} overrides
                for custom directory layouts. If None, uses default nested layout.
            ts_format: Time series format - "text" or "dss"

        Returns:
            Dictionary mapping file type to output path
        """
        from pyiwfm.io.preprocessor import save_complete_model

        return save_complete_model(
            self,
            output_dir,
            timeseries_format=ts_format,
            file_paths=file_paths,
        )

    def to_hdf5(self, output_file: Path | str) -> None:
        """
        Write model to HDF5 file.

        Saves the complete model (mesh, stratigraphy, and all components)
        to a single HDF5 file for efficient storage and later loading.

        Args:
            output_file: Path to the output HDF5 file

        Example:
            >>> model.to_hdf5("model.h5")
        """
        from pyiwfm.io.hdf5 import write_model_hdf5

        write_model_hdf5(output_file, self)

    def to_binary(self, output_file: Path | str) -> None:
        """
        Write model mesh and stratigraphy to binary files.

        Args:
            output_file: Base path for output files (will create .bin and .strat.bin)
        """
        from pyiwfm.io.binary import write_binary_mesh, write_binary_stratigraphy

        output_file = Path(output_file)

        if self.mesh:
            write_binary_mesh(output_file, self.mesh)

        if self.stratigraphy:
            strat_file = output_file.with_suffix(".strat.bin")
            write_binary_stratigraphy(strat_file, self.stratigraphy)

    # ========================================================================
    # Validation Methods
    # ========================================================================

    def validate(self) -> list[str]:
        """
        Validate model structure and data.

        The mesh and stratigraphy validators are contracted to raise
        :class:`MeshError` and :class:`StratigraphyError` respectively
        (with stratigraphy additionally returning a list of non-critical
        warnings). Any other exception is a programmer bug and propagates.

        Returns:
            Empty list. The list-returning shape is preserved for legacy
            callers; the actual error list lives on the
            :class:`ValidationError` raised below.

        Raises:
            ValidationError: If validation finds any error.
        """
        errors: list[str] = []

        # Validate mesh
        if self.mesh is None:
            errors.append("Model has no mesh")
        else:
            try:
                self.mesh.validate()
            except MeshError as e:
                errors.append(f"Mesh validation failed: {e}")

        # Validate stratigraphy
        if self.stratigraphy is None:
            errors.append("Model has no stratigraphy")
        else:
            try:
                warnings = self.stratigraphy.validate()
                errors.extend(warnings)
            except StratigraphyError as e:
                errors.append(f"Stratigraphy validation failed: {e}")

        # Check mesh/stratigraphy consistency
        if self.mesh is not None and self.stratigraphy is not None:
            if self.mesh.n_nodes != self.stratigraphy.n_nodes:
                errors.append(
                    f"Node count mismatch: mesh has {self.mesh.n_nodes}, "
                    f"stratigraphy has {self.stratigraphy.n_nodes}"
                )

        if errors:
            raise ValidationError(
                f"Model validation failed with {len(errors)} error(s)", errors=errors
            )

        return []

    # ========================================================================
    # Properties
    # ========================================================================

    @property
    def n_nodes(self) -> int:
        """Return number of nodes in the mesh."""
        if self.mesh is None:
            return 0
        return self.mesh.n_nodes

    @property
    def n_elements(self) -> int:
        """Return number of elements in the mesh."""
        if self.mesh is None:
            return 0
        return self.mesh.n_elements

    @property
    def n_layers(self) -> int:
        """Return number of layers in the stratigraphy."""
        if self.stratigraphy is None:
            return 0
        return self.stratigraphy.n_layers

    @property
    def grid(self) -> AppGrid | None:
        """Alias for mesh property for compatibility."""
        return self.mesh

    @grid.setter
    def grid(self, value: AppGrid | None) -> None:
        """Set the mesh/grid."""
        self.mesh = value

    # ========================================================================
    # Component Properties
    # ========================================================================

    @property
    def n_wells(self) -> int:
        """Return number of wells in the groundwater component."""
        if self.groundwater is None:
            return 0
        return self.groundwater.n_wells

    @property
    def n_stream_nodes(self) -> int:
        """Return number of stream nodes."""
        if self.streams is None:
            return 0
        return self.streams.n_nodes

    @property
    def n_stream_reaches(self) -> int:
        """Return number of stream reaches."""
        if self.streams is None:
            return 0
        return self.streams.n_reaches

    @property
    def n_diversions(self) -> int:
        """Return number of diversions."""
        if self.streams is None:
            return 0
        return self.streams.n_diversions

    @property
    def n_lakes(self) -> int:
        """Return number of lakes."""
        if self.lakes is None:
            return 0
        return self.lakes.n_lakes

    @property
    def n_crop_types(self) -> int:
        """Return number of crop types in the root zone."""
        if self.rootzone is None:
            return 0
        return self.rootzone.n_crop_types

    @property
    def has_groundwater(self) -> bool:
        """Return True if groundwater component is loaded."""
        return self.groundwater is not None

    @property
    def has_streams(self) -> bool:
        """Return True if stream component is loaded."""
        return self.streams is not None

    @property
    def has_lakes(self) -> bool:
        """Return True if lake component is loaded."""
        return self.lakes is not None

    @property
    def has_rootzone(self) -> bool:
        """Return True if root zone component is loaded."""
        return self.rootzone is not None

    @property
    def has_small_watersheds(self) -> bool:
        """Return True if small watershed component is loaded."""
        return self.small_watersheds is not None

    @property
    def has_unsaturated_zone(self) -> bool:
        """Return True if unsaturated zone component is loaded."""
        return self.unsaturated_zone is not None

    @property
    def load_errors(self) -> list[ComponentLoadError]:
        """List of component load failures recorded during construction.

        Populated when a loader (``IWFMModel.from_preprocessor``,
        ``from_simulation_with_preprocessor``, …) was invoked with
        ``strict=False`` (or ``"collect"`` and no failures occurred to
        trigger an immediate raise). Empty for clean loads or for
        ``strict=True`` (which fails fast and never returns the model).

        Returns a copy so external mutation doesn't corrupt the metadata.
        """
        from pyiwfm.core.loaders._common import _LOAD_ERRORS_KEY

        return list(self.metadata.get(_LOAD_ERRORS_KEY, []))

    @property
    def has_load_errors(self) -> bool:
        """True if any component failed to load during model construction."""
        from pyiwfm.core.loaders._common import _LOAD_ERRORS_KEY

        return bool(self.metadata.get(_LOAD_ERRORS_KEY))

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def summary(self) -> str:
        """
        Return a summary string of the model.

        Returns:
            Multi-line summary of model components
        """
        lines = [
            f"IWFM Model: {self.name}",
            "=" * (len(self.name) + 13),
            "",
            "Mesh & Stratigraphy:",
            f"  Nodes: {self.n_nodes}",
            f"  Elements: {self.n_elements}",
            f"  Layers: {self.n_layers}",
        ]

        if self.mesh is not None:
            lines.append(f"  Subregions: {self.mesh.n_subregions}")

        # Groundwater component
        lines.append("")
        lines.append("Groundwater Component:")
        if self.groundwater is not None:
            lines.append(f"  Wells: {self.groundwater.n_wells}")
            lines.append(f"  Hydrograph Locations: {self.groundwater.n_hydrograph_locations}")
            lines.append(f"  Boundary Conditions: {self.groundwater.n_boundary_conditions}")
            lines.append(f"  Tile Drains: {self.groundwater.n_tile_drains}")
            if self.groundwater.aquifer_params is not None:
                lines.append("  Aquifer Parameters: Loaded")
            else:
                lines.append("  Aquifer Parameters: Not loaded")
        else:
            lines.append("  Not loaded")

        # Stream component
        lines.append("")
        lines.append("Stream Component:")
        if self.streams is not None:
            lines.append(f"  Stream Nodes: {self.streams.n_nodes}")
            lines.append(f"  Reaches: {self.streams.n_reaches}")
            lines.append(f"  Diversions: {self.streams.n_diversions}")
            lines.append(f"  Bypasses: {self.streams.n_bypasses}")
        else:
            lines.append("  Not loaded")

        # Lake component
        lines.append("")
        lines.append("Lake Component:")
        if self.lakes is not None:
            lines.append(f"  Lakes: {self.lakes.n_lakes}")
            lines.append(f"  Lake Elements: {self.lakes.n_lake_elements}")
        else:
            lines.append("  Not loaded")

        # Root zone component
        lines.append("")
        lines.append("Root Zone Component:")
        if self.rootzone is not None:
            lines.append(f"  Crop Types: {self.rootzone.n_crop_types}")
            lines.append(f"  Land Use Assignments: {len(self.rootzone.element_landuse)}")
            lines.append(f"  Soil Parameter Sets: {len(self.rootzone.soil_params)}")
        else:
            lines.append("  Not loaded")

        # Small watershed component
        lines.append("")
        lines.append("Small Watershed Component:")
        if self.small_watersheds is not None:
            lines.append(f"  Watersheds: {self.small_watersheds.n_watersheds}")
        else:
            lines.append("  Not loaded")

        # Unsaturated zone component
        lines.append("")
        lines.append("Unsaturated Zone Component:")
        if self.unsaturated_zone is not None:
            lines.append(f"  Layers: {self.unsaturated_zone.n_layers}")
            lines.append(f"  Elements: {self.unsaturated_zone.n_elements}")
        else:
            lines.append("  Not loaded")

        # Metadata
        lines.append("")
        source = self.metadata.get("source", "unknown")
        lines.append(f"Source: {source}")

        return "\n".join(lines)

    # ========================================================================
    # Model mutation helpers
    # ========================================================================
    #
    # These convenience methods provide an ergonomic, validated path for
    # editing a loaded model in place. The underlying component attributes
    # (``model.groundwater.aquifer_params.kh``, etc.) are still accessible
    # for advanced use, but the helpers below are the documented, supported
    # API for callers building scenarios, calibration runs, or diff tools.
    #
    # Each helper:
    #   - validates inputs (raises ``ValueError`` with helpful messages)
    #   - records the modified component in ``self._dirty`` so save paths
    #     can warn if a dirty component fails to write
    #   - is non-breaking: existing direct-attribute mutation still works
    #
    # See ``docs/user_guide/mutating_models.rst`` for end-to-end examples.

    def mark_dirty(self, component_name: str) -> None:
        """Mark ``component_name`` as modified since the last load/save.

        Call this from custom mutation paths so :meth:`to_simulation` and
        related save methods can surface incomplete writes.
        """
        self._dirty.add(component_name)

    def set_aquifer_parameter(
        self,
        param: str,
        layer: int,
        values: NDArray[np.float64] | list[float],
    ) -> None:
        """Replace an aquifer parameter array for a single layer.

        Parameters
        ----------
        param
            One of ``"kh"``, ``"kv"``, ``"ss"``, ``"sy"``, ``"aquitard_kv"``
            (matches the :class:`AquiferParameters._PARAM_ATTRS` dispatcher).
        layer
            1-based aquifer layer (IWFM convention).
        values
            Array-like of length ``n_nodes`` with the new values for this
            layer.

        Raises
        ------
        ValueError
            If groundwater isn't loaded, the layer is out of range, the
            named parameter array isn't allocated, or the values length
            doesn't match ``n_nodes``.
        KeyError
            If ``param`` isn't a recognized parameter name.

        Examples
        --------
        >>> import numpy as np
        >>> new_kh = np.full(model.groundwater.n_nodes, 1e-4)
        >>> model.set_aquifer_parameter("kh", layer=1, values=new_kh)
        """
        from pyiwfm.core.ids import to_index

        if self.groundwater is None:
            raise ValueError("groundwater component is not loaded")
        params = self.groundwater.aquifer_params
        if params is None:
            raise ValueError("aquifer_params is not set on groundwater component")

        layer_idx = to_index(layer, params.n_layers, kind="layer")

        # Trigger KeyError for unknown param names, ValueError if unset.
        arr = params.get_array(param)

        values_arr = np.asarray(values, dtype=np.float64)
        if values_arr.shape != (params.n_nodes,):
            raise ValueError(
                f"values for {param!r} layer {layer} must have shape "
                f"({params.n_nodes},); got {values_arr.shape}"
            )
        arr[:, layer_idx] = values_arr
        self.mark_dirty("groundwater")

    def set_aquifer_parameter_at(
        self,
        param: str,
        node_id: int,
        layer: int,
        value: float,
    ) -> None:
        """Set a single (node, layer) cell of an aquifer parameter.

        Parameters
        ----------
        param
            See :meth:`set_aquifer_parameter`.
        node_id
            1-based node ID (IWFM convention).
        layer
            1-based aquifer layer.
        value
            New scalar value.

        Raises
        ------
        ValueError
            If groundwater isn't loaded, IDs are out of range, or the
            parameter array isn't allocated.
        """
        from pyiwfm.core.ids import to_index

        if self.groundwater is None:
            raise ValueError("groundwater component is not loaded")
        params = self.groundwater.aquifer_params
        if params is None:
            raise ValueError("aquifer_params is not set on groundwater component")

        node_idx = to_index(node_id, params.n_nodes, kind="node")
        layer_idx = to_index(layer, params.n_layers, kind="layer")
        params.get_array(param)[node_idx, layer_idx] = float(value)
        self.mark_dirty("groundwater")

    def set_stratigraphy_from_thicknesses(
        self,
        gs_elev: NDArray[np.float64] | list[float],
        aquitard_thicknesses: NDArray[np.float64] | list[list[float]],
        aquifer_thicknesses: NDArray[np.float64] | list[list[float]],
        active_node: NDArray[np.bool_] | None = None,
    ) -> None:
        """Replace ``self.stratigraphy`` with one built from thickness arrays.

        Wraps :meth:`Stratigraphy.from_thicknesses` and validates the result
        is consistent with the current mesh (if loaded).

        Parameters
        ----------
        gs_elev
            Ground surface elevations, shape ``(n_nodes,)``.
        aquitard_thicknesses
            Aquitard thicknesses, shape ``(n_nodes, n_layers)``. IWFM
            convention: aquitard ``k`` sits above aquifer layer ``k``.
        aquifer_thicknesses
            Aquifer thicknesses, shape ``(n_nodes, n_layers)``.
        active_node
            Optional active-node flags, shape ``(n_nodes, n_layers)``.

        Raises
        ------
        StratigraphyError
            If thickness shapes are inconsistent or any thickness is
            negative (validation comes from
            :meth:`Stratigraphy.from_thicknesses`).
        ValueError
            If a mesh is loaded and ``gs_elev`` length doesn't match
            ``mesh.n_nodes``.
        """
        from pyiwfm.core.stratigraphy import Stratigraphy

        gs_arr = np.asarray(gs_elev, dtype=np.float64)
        if self.mesh is not None and gs_arr.shape != (self.mesh.n_nodes,):
            raise ValueError(
                f"gs_elev shape {gs_arr.shape} does not match mesh ({self.mesh.n_nodes},)"
            )

        self.stratigraphy = Stratigraphy.from_thicknesses(
            gs_arr,
            np.asarray(aquitard_thicknesses, dtype=np.float64),
            np.asarray(aquifer_thicknesses, dtype=np.float64),
            active_node,
        )
        self.mark_dirty("stratigraphy")

    def add_observation_well(
        self,
        node_id: int,
        layer: int,
        x: float,
        y: float,
        name: str = "",
    ) -> None:
        """Append a groundwater hydrograph observation point.

        Mirrors the IWFM convention where each hydrograph location is
        identified by ``(node_id, layer)`` plus optional ``(x, y)`` for
        rendering and a free-form ``name``.

        Parameters
        ----------
        node_id
            1-based mesh node ID.
        layer
            1-based aquifer layer.
        x, y
            Coordinates (model CRS).
        name
            Optional descriptor (e.g. well name).

        Raises
        ------
        ValueError
            If groundwater isn't loaded, or IDs are out of range against
            the GW component's ``n_nodes`` / ``n_layers``.
        """
        from pyiwfm.components.groundwater import HydrographLocation
        from pyiwfm.core.ids import to_index

        if self.groundwater is None:
            raise ValueError("groundwater component is not loaded")

        # Validate IDs (raises ValueError if out of range)
        to_index(node_id, self.groundwater.n_nodes, kind="node")
        to_index(layer, self.groundwater.n_layers, kind="layer")

        self.groundwater.add_hydrograph_location(
            HydrographLocation(
                node_id=node_id,
                layer=layer,
                x=float(x),
                y=float(y),
                name=name,
            )
        )
        self.mark_dirty("groundwater")

    def remove_observation_well(self, name: str) -> int:
        """Remove all groundwater hydrograph locations whose ``name`` matches.

        Parameters
        ----------
        name
            Exact-match name (case-sensitive). To remove unnamed locations,
            pass an empty string.

        Returns
        -------
        int
            Number of locations removed.

        Raises
        ------
        ValueError
            If groundwater isn't loaded.
        """
        if self.groundwater is None:
            raise ValueError("groundwater component is not loaded")
        before = len(self.groundwater.hydrograph_locations)
        self.groundwater.hydrograph_locations = [
            loc for loc in self.groundwater.hydrograph_locations if loc.name != name
        ]
        removed = before - len(self.groundwater.hydrograph_locations)
        if removed:
            self.mark_dirty("groundwater")
        return removed

    def validate_components(self) -> list[str]:
        """
        Validate all model components.

        Each component validator is contracted (per
        :meth:`pyiwfm.core.base_component.BaseComponent.validate`) to
        raise :class:`ComponentError` on invalid state and return
        ``None`` on success. Any other exception is a programmer bug and
        propagates rather than being silently turned into a warning.

        Returns:
            List of validation warnings (one per component that raised
            ``ComponentError``). Empty if all loaded components validate.
        """
        warnings: list[str] = []
        for label, component in (
            ("Groundwater", self.groundwater),
            ("Stream", self.streams),
            ("Lake", self.lakes),
            ("Root zone", self.rootzone),
            ("Small watershed", self.small_watersheds),
            ("Unsaturated zone", self.unsaturated_zone),
        ):
            if component is None:
                continue
            try:
                component.validate()
            except ComponentError as e:
                warnings.append(f"{label} validation: {e}")

        return warnings

    def __repr__(self) -> str:
        return (
            f"IWFMModel(name='{self.name}', n_nodes={self.n_nodes}, "
            f"n_elements={self.n_elements}, n_layers={self.n_layers})"
        )
