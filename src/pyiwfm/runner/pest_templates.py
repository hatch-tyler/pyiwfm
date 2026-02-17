"""PEST++ template file generation for IWFM models.

This module provides IWFM-aware template file generation for PEST++
calibration and uncertainty analysis. It understands IWFM file formats
and generates appropriate template files for different parameter types.

Template files (.tpl) contain parameter markers that PEST++ replaces
with parameter values during model runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pyiwfm.runner.pest import TemplateFile
from pyiwfm.runner.pest_params import (
    IWFMParameterType,
    Parameter,
)

if TYPE_CHECKING:
    from pyiwfm.runner.pest_manager import IWFMParameterManager


@dataclass
class TemplateMarker:
    """A parameter marker in a template file.

    Attributes
    ----------
    parameter_name : str
        Name of the PEST parameter.
    line_number : int
        Line number in the file (1-based).
    column_start : int
        Starting column position.
    column_end : int
        Ending column position.
    original_value : str
        Original value that was replaced.
    """

    parameter_name: str
    line_number: int
    column_start: int
    column_end: int
    original_value: str


@dataclass
class IWFMFileSection:
    """Represents a section of an IWFM input file.

    IWFM files typically have structured sections with:
    - Comment lines (starting with C or ``*``)
    - Data lines with fixed-format columns
    - Section delimiters

    Attributes
    ----------
    name : str
        Section name or identifier.
    start_line : int
        Starting line number (1-based).
    end_line : int
        Ending line number (1-based).
    data_columns : dict[str, int]
        Mapping of data field names to column indices.
    """

    name: str
    start_line: int
    end_line: int
    data_columns: dict[str, int] = field(default_factory=dict)


class IWFMTemplateManager:
    """Generates PEST++ template files for IWFM input files.

    This class understands IWFM file formats and generates appropriate
    template files for different parameter types. It supports:

    - Aquifer parameter templates (K, Ss, Sy by layer/zone)
    - Stream parameter templates (streambed K, thickness)
    - Multiplier templates (pumping, recharge, ET)
    - Pilot point templates (separate files for kriging)

    Parameters
    ----------
    model : Any
        IWFM model instance (optional, for auto-detection).
    parameter_manager : IWFMParameterManager
        Parameter manager containing parameters to template.
    output_dir : Path | str
        Directory for output template files.
    delimiter : str
        Delimiter character for parameter markers (default: '#').

    Examples
    --------
    >>> tm = IWFMTemplateManager(parameter_manager=pm, output_dir="pest/templates")
    >>> tpl = tm.generate_aquifer_template(
    ...     input_file="Groundwater.dat",
    ...     param_type=IWFMParameterType.HORIZONTAL_K,
    ...     layer=1,
    ... )
    """

    def __init__(
        self,
        model: Any = None,
        parameter_manager: IWFMParameterManager | None = None,
        output_dir: Path | str | None = None,
        delimiter: str = "#",
    ):
        """Initialize the template manager.

        Parameters
        ----------
        model : Any
            IWFM model instance (optional).
        parameter_manager : IWFMParameterManager | None
            Parameter manager with defined parameters.
        output_dir : Path | str | None
            Output directory for templates.
        delimiter : str
            PEST template delimiter character.
        """
        self.model = model
        self.pm = parameter_manager
        self.output_dir = Path(output_dir) if output_dir else Path(".")
        self.delimiter = delimiter
        self._templates: list[TemplateFile] = []

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Aquifer Parameter Templates
    # =========================================================================

    def generate_aquifer_template(
        self,
        input_file: Path | str,
        param_type: IWFMParameterType | str,
        layer: int | None = None,
        parameters: list[Parameter] | None = None,
        output_template: Path | str | None = None,
    ) -> TemplateFile:
        """Generate template for aquifer parameter file.

        Creates a template file for aquifer properties like hydraulic
        conductivity, specific storage, or specific yield.

        Parameters
        ----------
        input_file : Path | str
            Path to the IWFM aquifer parameter file.
        param_type : IWFMParameterType | str
            Type of parameter (e.g., HORIZONTAL_K, SPECIFIC_YIELD).
        layer : int | None
            Model layer for these parameters. If None, applies to all layers.
        parameters : list[Parameter] | None
            Parameters to include. If None, uses parameters from manager.
        output_template : Path | str | None
            Output template path. If None, auto-generates name.

        Returns
        -------
        TemplateFile
            The created template file.
        """
        input_file = Path(input_file)

        # Convert string param_type if needed
        if isinstance(param_type, str):
            param_type = IWFMParameterType(param_type)

        # Get parameters from manager if not provided
        if parameters is None and self.pm is not None:
            parameters = self.pm.get_parameters_by_type(param_type)
            if layer is not None:
                parameters = [p for p in parameters if p.layer == layer]

        if not parameters:
            raise ValueError(f"No parameters found for {param_type.value}")

        # Generate output path
        if output_template is None:
            layer_str = f"_l{layer}" if layer else ""
            output_template = self.output_dir / f"{param_type.value}{layer_str}.tpl"
        else:
            output_template = Path(output_template)

        # Read input file
        content = input_file.read_text()
        lines = content.splitlines()

        # Build parameter value to name mapping
        param_values = {}
        for p in parameters:
            param_values[p.name] = p.initial_value

        # Create template content
        template_lines = [f"ptf {self.delimiter}"]

        # Process each line
        markers: list[TemplateMarker] = []
        for i, line in enumerate(lines):
            new_line = self._replace_values_in_line(line, param_values, markers, i + 1)
            template_lines.append(new_line)

        # Write template
        output_template.write_text("\n".join(template_lines))

        tpl = TemplateFile(
            template_path=output_template,
            input_path=input_file,
            delimiter=self.delimiter,
            parameters=[p.name for p in parameters],
        )
        self._templates.append(tpl)
        return tpl

    def generate_aquifer_template_by_zone(
        self,
        input_file: Path | str,
        param_type: IWFMParameterType | str,
        zone_column: int,
        value_column: int,
        layer: int | None = None,
        header_lines: int = 0,
        output_template: Path | str | None = None,
    ) -> TemplateFile:
        """Generate template for zone-based aquifer parameters.

        For files where each row represents a zone with a parameter value.

        Parameters
        ----------
        input_file : Path | str
            Path to the input file.
        param_type : IWFMParameterType | str
            Type of parameter.
        zone_column : int
            Column containing zone ID (1-based).
        value_column : int
            Column containing parameter value (1-based).
        layer : int | None
            Model layer.
        header_lines : int
            Number of header lines to skip.
        output_template : Path | str | None
            Output template path.

        Returns
        -------
        TemplateFile
            The created template file.
        """
        input_file = Path(input_file)

        if isinstance(param_type, str):
            param_type = IWFMParameterType(param_type)

        # Get zone parameters from manager
        parameters = []
        if self.pm is not None:
            params = self.pm.get_parameters_by_type(param_type)
            if layer is not None:
                params = [p for p in params if p.layer == layer]
            # Filter to zone parameters
            parameters = [p for p in params if p.zone is not None]

        if not parameters:
            raise ValueError(f"No zone parameters found for {param_type.value}")

        # Generate output path
        if output_template is None:
            layer_str = f"_l{layer}" if layer else ""
            output_template = self.output_dir / f"{param_type.value}_zone{layer_str}.tpl"
        else:
            output_template = Path(output_template)

        # Read and process file
        content = input_file.read_text()
        lines = content.splitlines()
        template_lines = [f"ptf {self.delimiter}"]

        # Map zone IDs to parameters
        zone_params = {p.zone: p for p in parameters}
        param_names = []

        for i, line in enumerate(lines):
            if i < header_lines or line.strip().startswith(("C", "*")):
                template_lines.append(line)
                continue

            # Parse line to get zone ID
            parts = line.split()
            if len(parts) >= max(zone_column, value_column):
                try:
                    zone_id = int(parts[zone_column - 1])
                    if zone_id in zone_params:
                        param = zone_params[zone_id]
                        # Replace value column with parameter marker
                        marker = f"{self.delimiter}{param.name:^12s}{self.delimiter}"
                        parts[value_column - 1] = marker
                        template_lines.append("  ".join(parts))
                        param_names.append(param.name)
                        continue
                except (ValueError, IndexError):
                    pass

            template_lines.append(line)

        # Write template
        output_template.write_text("\n".join(template_lines))

        tpl = TemplateFile(
            template_path=output_template,
            input_path=input_file,
            delimiter=self.delimiter,
            parameters=param_names,
        )
        self._templates.append(tpl)
        return tpl

    # =========================================================================
    # Stream Parameter Templates
    # =========================================================================

    def generate_stream_template(
        self,
        input_file: Path | str,
        param_type: IWFMParameterType | str,
        reach_column: int = 1,
        value_column: int = 2,
        header_lines: int = 0,
        output_template: Path | str | None = None,
    ) -> TemplateFile:
        """Generate template for stream parameter file.

        Creates a template for stream parameters like streambed K,
        thickness, or width by reach.

        Parameters
        ----------
        input_file : Path | str
            Path to stream parameter file.
        param_type : IWFMParameterType | str
            Type of parameter (e.g., STREAMBED_K).
        reach_column : int
            Column containing reach ID (1-based).
        value_column : int
            Column containing parameter value (1-based).
        header_lines : int
            Number of header lines to skip.
        output_template : Path | str | None
            Output template path.

        Returns
        -------
        TemplateFile
            The created template file.
        """
        input_file = Path(input_file)

        if isinstance(param_type, str):
            param_type = IWFMParameterType(param_type)

        # Get stream parameters
        parameters = []
        if self.pm is not None:
            params = self.pm.get_parameters_by_type(param_type)
            # Filter to reach parameters (have reach_id in metadata)
            parameters = [p for p in params if p.metadata.get("reach_id") is not None]

        if not parameters:
            raise ValueError(f"No stream parameters found for {param_type.value}")

        # Generate output path
        if output_template is None:
            output_template = self.output_dir / f"{param_type.value}_stream.tpl"
        else:
            output_template = Path(output_template)

        # Read and process file
        content = input_file.read_text()
        lines = content.splitlines()
        template_lines = [f"ptf {self.delimiter}"]

        # Map reach IDs to parameters
        reach_params = {p.metadata["reach_id"]: p for p in parameters}
        param_names = []

        for i, line in enumerate(lines):
            if i < header_lines or line.strip().startswith(("C", "*")):
                template_lines.append(line)
                continue

            parts = line.split()
            if len(parts) >= max(reach_column, value_column):
                try:
                    reach_id = int(parts[reach_column - 1])
                    if reach_id in reach_params:
                        param = reach_params[reach_id]
                        marker = f"{self.delimiter}{param.name:^12s}{self.delimiter}"
                        parts[value_column - 1] = marker
                        template_lines.append("  ".join(parts))
                        param_names.append(param.name)
                        continue
                except (ValueError, IndexError):
                    pass

            template_lines.append(line)

        # Write template
        output_template.write_text("\n".join(template_lines))

        tpl = TemplateFile(
            template_path=output_template,
            input_path=input_file,
            delimiter=self.delimiter,
            parameters=param_names,
        )
        self._templates.append(tpl)
        return tpl

    # =========================================================================
    # Multiplier Templates
    # =========================================================================

    def generate_multiplier_template(
        self,
        param_type: IWFMParameterType | str,
        output_template: Path | str | None = None,
        format_width: int = 15,
    ) -> TemplateFile:
        """Generate template for multiplier parameter file.

        Creates a simple multiplier file with parameter markers.
        Multipliers are applied to base values by a preprocessor.

        Parameters
        ----------
        param_type : IWFMParameterType | str
            Type of multiplier parameter.
        output_template : Path | str | None
            Output template path.
        format_width : int
            Width for parameter markers.

        Returns
        -------
        TemplateFile
            The created template file.
        """
        if isinstance(param_type, str):
            param_type = IWFMParameterType(param_type)

        # Get multiplier parameters
        parameters = []
        if self.pm is not None:
            params = self.pm.get_parameters_by_type(param_type)
            parameters = [p for p in params if param_type.is_multiplier]

        if not parameters:
            raise ValueError(f"No multiplier parameters found for {param_type.value}")

        # Generate paths
        if output_template is None:
            output_template = self.output_dir / f"{param_type.value}_mult.tpl"
        else:
            output_template = Path(output_template)

        mult_file = output_template.with_suffix(".dat")

        # Create multiplier file content
        template_lines = [f"ptf {self.delimiter}"]
        template_lines.append(f"# Multiplier file for {param_type.value}")
        template_lines.append(f"# Generated: {datetime.now().isoformat()}")
        template_lines.append("#")
        template_lines.append("# Parameter_Name   Value")

        param_names = []
        for param in parameters:
            marker = f"{self.delimiter}{param.name:^{format_width}s}{self.delimiter}"
            template_lines.append(f"{param.name:20s}  {marker}")
            param_names.append(param.name)

        # Write template
        output_template.write_text("\n".join(template_lines))

        # Also write initial multiplier file
        mult_lines = [
            f"# Multiplier file for {param_type.value}",
            f"# Generated: {datetime.now().isoformat()}",
            "#",
            "# Parameter_Name   Value",
        ]
        for param in parameters:
            mult_lines.append(f"{param.name:20s}  {param.initial_value:15.6e}")
        mult_file.write_text("\n".join(mult_lines))

        tpl = TemplateFile(
            template_path=output_template,
            input_path=mult_file,
            delimiter=self.delimiter,
            parameters=param_names,
        )
        self._templates.append(tpl)
        return tpl

    def generate_zone_multiplier_template(
        self,
        param_type: IWFMParameterType | str,
        zones: list[int] | None = None,
        output_template: Path | str | None = None,
    ) -> TemplateFile:
        """Generate template for zone-based multipliers.

        Creates a multiplier file with one value per zone.

        Parameters
        ----------
        param_type : IWFMParameterType | str
            Type of multiplier parameter.
        zones : list[int] | None
            Zone IDs. If None, determines from parameters.
        output_template : Path | str | None
            Output template path.

        Returns
        -------
        TemplateFile
            The created template file.
        """
        if isinstance(param_type, str):
            param_type = IWFMParameterType(param_type)

        # Get parameters
        parameters = []
        if self.pm is not None:
            params = self.pm.get_parameters_by_type(param_type)
            parameters = [p for p in params if p.zone is not None]

        if not parameters:
            raise ValueError(f"No zone multiplier parameters found for {param_type.value}")

        # Sort by zone
        parameters = sorted(parameters, key=lambda p: p.zone or 0)

        # Generate paths
        if output_template is None:
            output_template = self.output_dir / f"{param_type.value}_zone_mult.tpl"
        else:
            output_template = Path(output_template)

        mult_file = output_template.with_suffix(".dat")

        # Create template content
        template_lines = [f"ptf {self.delimiter}"]
        template_lines.append(f"# Zone multiplier file for {param_type.value}")
        template_lines.append("# Zone_ID   Multiplier")

        param_names = []
        for param in parameters:
            marker = f"{self.delimiter}{param.name:^12s}{self.delimiter}"
            template_lines.append(f"{param.zone:8d}  {marker}")
            param_names.append(param.name)

        # Write template
        output_template.write_text("\n".join(template_lines))

        # Write initial file
        mult_lines = [
            f"# Zone multiplier file for {param_type.value}",
            "# Zone_ID   Multiplier",
        ]
        for param in parameters:
            mult_lines.append(f"{param.zone:8d}  {param.initial_value:15.6e}")
        mult_file.write_text("\n".join(mult_lines))

        tpl = TemplateFile(
            template_path=output_template,
            input_path=mult_file,
            delimiter=self.delimiter,
            parameters=param_names,
        )
        self._templates.append(tpl)
        return tpl

    # =========================================================================
    # Pilot Point Templates
    # =========================================================================

    def generate_pilot_point_template(
        self,
        param_type: IWFMParameterType | str,
        layer: int = 1,
        output_template: Path | str | None = None,
    ) -> TemplateFile:
        """Generate template for pilot point parameter file.

        Pilot points are separate from IWFM input files. They are
        interpolated to model nodes using kriging by a preprocessor.

        Parameters
        ----------
        param_type : IWFMParameterType | str
            Type of parameter.
        layer : int
            Model layer.
        output_template : Path | str | None
            Output template path.

        Returns
        -------
        TemplateFile
            The created template file.
        """
        if isinstance(param_type, str):
            param_type = IWFMParameterType(param_type)

        # Get pilot point parameters
        parameters = []
        if self.pm is not None:
            params = self.pm.get_pilot_point_parameters()
            parameters = [p for p in params if p.param_type == param_type and p.layer == layer]

        if not parameters:
            raise ValueError(
                f"No pilot point parameters found for {param_type.value} layer {layer}"
            )

        # Generate paths
        if output_template is None:
            output_template = self.output_dir / f"pp_{param_type.value}_l{layer}.tpl"
        else:
            output_template = Path(output_template)

        pp_file = output_template.with_suffix(".dat")

        # Create template content
        template_lines = [f"ptf {self.delimiter}"]
        template_lines.append(f"# Pilot points for {param_type.value} layer {layer}")
        template_lines.append("# Name          X             Y             Value")

        param_names = []
        for param in parameters:
            x = param.location[0] if param.location else 0.0
            y = param.location[1] if param.location else 0.0
            marker = f"{self.delimiter}{param.name:^12s}{self.delimiter}"
            template_lines.append(f"{param.name:12s}  {x:12.2f}  {y:12.2f}  {marker}")
            param_names.append(param.name)

        # Write template
        output_template.write_text("\n".join(template_lines))

        # Write initial pilot point file
        pp_lines = [
            f"# Pilot points for {param_type.value} layer {layer}",
            "# Name          X             Y             Value",
        ]
        for param in parameters:
            x = param.location[0] if param.location else 0.0
            y = param.location[1] if param.location else 0.0
            pp_lines.append(f"{param.name:12s}  {x:12.2f}  {y:12.2f}  {param.initial_value:12.6e}")
        pp_file.write_text("\n".join(pp_lines))

        tpl = TemplateFile(
            template_path=output_template,
            input_path=pp_file,
            delimiter=self.delimiter,
            parameters=param_names,
        )
        self._templates.append(tpl)
        return tpl

    # =========================================================================
    # Root Zone Templates
    # =========================================================================

    def generate_rootzone_template(
        self,
        input_file: Path | str,
        param_type: IWFMParameterType | str,
        land_use_column: int = 1,
        value_column: int = 2,
        header_lines: int = 0,
        output_template: Path | str | None = None,
    ) -> TemplateFile:
        """Generate template for root zone parameter file.

        Creates a template for root zone parameters like crop coefficients,
        irrigation efficiency, etc. by land use type.

        Parameters
        ----------
        input_file : Path | str
            Path to root zone parameter file.
        param_type : IWFMParameterType | str
            Type of parameter.
        land_use_column : int
            Column containing land use type (1-based).
        value_column : int
            Column containing parameter value (1-based).
        header_lines : int
            Number of header lines to skip.
        output_template : Path | str | None
            Output template path.

        Returns
        -------
        TemplateFile
            The created template file.
        """
        input_file = Path(input_file)

        if isinstance(param_type, str):
            param_type = IWFMParameterType(param_type)

        # Get root zone parameters
        parameters = []
        if self.pm is not None:
            params = self.pm.get_parameters_by_type(param_type)
            # Filter to parameters with land_use_type metadata
            parameters = [p for p in params if p.metadata.get("land_use_type") is not None]

        if not parameters:
            raise ValueError(f"No root zone parameters found for {param_type.value}")

        # Generate output path
        if output_template is None:
            output_template = self.output_dir / f"{param_type.value}_rootzone.tpl"
        else:
            output_template = Path(output_template)

        # Read and process file
        content = input_file.read_text()
        lines = content.splitlines()
        template_lines = [f"ptf {self.delimiter}"]

        # Map land use types to parameters
        lu_params = {p.metadata["land_use_type"]: p for p in parameters}
        param_names = []

        for i, line in enumerate(lines):
            if i < header_lines or line.strip().startswith(("C", "*")):
                template_lines.append(line)
                continue

            parts = line.split()
            if len(parts) >= max(land_use_column, value_column):
                try:
                    lu_type = parts[land_use_column - 1]
                    if lu_type in lu_params:
                        param = lu_params[lu_type]
                        marker = f"{self.delimiter}{param.name:^12s}{self.delimiter}"
                        parts[value_column - 1] = marker
                        template_lines.append("  ".join(parts))
                        param_names.append(param.name)
                        continue
                except (ValueError, IndexError):
                    pass

            template_lines.append(line)

        # Write template
        output_template.write_text("\n".join(template_lines))

        tpl = TemplateFile(
            template_path=output_template,
            input_path=input_file,
            delimiter=self.delimiter,
            parameters=param_names,
        )
        self._templates.append(tpl)
        return tpl

    # =========================================================================
    # Batch Generation
    # =========================================================================

    def generate_all_templates(
        self,
        input_files: dict[str, Path | str] | None = None,
    ) -> list[TemplateFile]:
        """Generate all required template files based on parameters.

        Automatically generates templates for all parameters in the
        parameter manager.

        Parameters
        ----------
        input_files : dict[str, Path | str] | None
            Mapping of parameter type values to input file paths.
            E.g., {"hk": "Groundwater.dat", "strk": "Stream.dat"}

        Returns
        -------
        list[TemplateFile]
            List of created template files.
        """
        if self.pm is None:
            raise ValueError("Parameter manager required for batch generation")

        templates = []
        input_files = input_files or {}

        # Group parameters by type
        param_types: dict[IWFMParameterType, list[Any]] = {}
        for param in self.pm.get_all_parameters():
            if param.param_type is not None:
                pt = param.param_type
                if pt not in param_types:
                    param_types[pt] = []
                param_types[pt].append(param)

        for param_type, params in param_types.items():
            # Check for pilot points
            pp_params = [p for p in params if p.location is not None]
            if pp_params:
                layers = {p.layer for p in pp_params if p.layer is not None}
                for layer in layers:
                    try:
                        tpl = self.generate_pilot_point_template(param_type, layer=layer)
                        templates.append(tpl)
                    except ValueError:
                        pass

            # Check for multipliers
            if param_type.is_multiplier:
                zone_params = [p for p in params if p.zone is not None]
                global_params = [p for p in params if p.zone is None]

                if zone_params:
                    try:
                        tpl = self.generate_zone_multiplier_template(param_type)
                        templates.append(tpl)
                    except ValueError:
                        pass

                if global_params:
                    try:
                        tpl = self.generate_multiplier_template(param_type)
                        templates.append(tpl)
                    except ValueError:
                        pass

        self._templates = templates
        return templates

    # =========================================================================
    # Utilities
    # =========================================================================

    def _replace_values_in_line(
        self,
        line: str,
        param_values: dict[str, float],
        markers: list[TemplateMarker],
        line_number: int,
    ) -> str:
        """Replace parameter values with markers in a line.

        Parameters
        ----------
        line : str
            Input line.
        param_values : dict[str, float]
            Mapping of parameter names to values.
        markers : list[TemplateMarker]
            List to append created markers to.
        line_number : int
            Current line number (1-based).

        Returns
        -------
        str
            Line with values replaced by markers.
        """
        # Skip comment lines
        if line.strip().startswith(("C", "*", "!")):
            return line

        result = line
        for param_name, value in param_values.items():
            # Try different numeric formats
            patterns = [
                f"{value:.6e}",
                f"{value:.4e}",
                f"{value:.6f}",
                f"{value:.4f}",
                f"{value:.2f}",
                f"{value:g}",
                str(value),
            ]

            for pattern in patterns:
                if pattern in result:
                    marker = f"{self.delimiter}{param_name:^12s}{self.delimiter}"
                    start = result.find(pattern)
                    end = start + len(pattern)
                    markers.append(
                        TemplateMarker(
                            parameter_name=param_name,
                            line_number=line_number,
                            column_start=start,
                            column_end=end,
                            original_value=pattern,
                        )
                    )
                    result = result.replace(pattern, marker, 1)
                    break

        return result

    def get_all_templates(self) -> list[TemplateFile]:
        """Get all created template files.

        Returns
        -------
        list[TemplateFile]
            All template files created by this manager.
        """
        return list(self._templates)

    def clear_templates(self) -> None:
        """Clear all created templates."""
        self._templates.clear()

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"IWFMTemplateManager(output_dir='{self.output_dir}', "
            f"n_templates={len(self._templates)})"
        )
