"""TemplateFile — extracted from runner/pest.py in v2.0 PR 5.

The class body is verbatim from v1.x ``runner/pest.py``; this module
just gives it a dedicated file so the package no longer has a
1,400-line module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TemplateFile:
    """PEST++ template file (.tpl) definition.

    A template file is an input file with parameters marked by
    delimiters. PEST++ replaces these markers with parameter values.

    Attributes
    ----------
    template_path : Path
        Path to the template file.
    input_path : Path
        Path to the model input file to generate.
    delimiter : str
        Delimiter character for parameter markers (default: '#').
    parameters : list[str]
        List of parameter names in this template.
    """

    template_path: Path
    input_path: Path
    delimiter: str = "#"
    parameters: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Convert paths."""
        self.template_path = Path(self.template_path)
        self.input_path = Path(self.input_path)

    @classmethod
    def create_from_file(
        cls,
        input_file: Path | str,
        template_file: Path | str,
        parameters: dict[str, float],
        delimiter: str = "#",
    ) -> TemplateFile:
        """Create a template file from an existing input file.

        Parameters
        ----------
        input_file : Path | str
            Path to the original input file.
        template_file : Path | str
            Path where template will be written.
        parameters : dict[str, float]
            Dictionary mapping parameter names to their current values
            in the input file. These values will be replaced with markers.
        delimiter : str
            Delimiter character for parameter markers.

        Returns
        -------
        TemplateFile
            The created template file object.
        """
        input_file = Path(input_file)
        template_file = Path(template_file)

        content = input_file.read_text()

        # Replace parameter values with markers
        param_names = []
        for param_name, value in parameters.items():
            # Create marker with fixed width
            marker = f"{delimiter}{param_name:^12s}{delimiter}"

            # Replace the value with the marker
            # Handle different numeric formats
            patterns = [
                f"{value:.6e}",
                f"{value:.6f}",
                f"{value:.4e}",
                f"{value:.4f}",
                f"{value:g}",
                str(value),
            ]

            replaced = False
            for pattern in patterns:
                if pattern in content:
                    content = content.replace(pattern, marker, 1)
                    replaced = True
                    break

            if replaced:
                param_names.append(param_name)

        # Write template file with header
        with open(template_file, "w") as f:
            f.write(f"ptf {delimiter}\n")
            f.write(content)

        return cls(
            template_path=template_file,
            input_path=input_file,
            delimiter=delimiter,
            parameters=param_names,
        )

    def to_pest_line(self) -> str:
        """Format as PEST control file template line."""
        return f"{self.template_path.name}  {self.input_path.name}"
