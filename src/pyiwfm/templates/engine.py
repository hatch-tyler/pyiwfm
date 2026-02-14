"""
Jinja2 template engine for IWFM file generation.

This module provides a hybrid template engine that uses Jinja2 for headers
and structure, with numpy for efficient large data array output.

Supports optional comment preservation for round-trip operations.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, TextIO

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from pyiwfm.io.comment_metadata import CommentMetadata
    from pyiwfm.io.comment_writer import CommentWriter

from jinja2 import Environment, FileSystemLoader, PackageLoader, select_autoescape


# Default template directories
TEMPLATES_DIR = Path(__file__).parent / "iwfm"
LEGACY_TEMPLATES_DIR = Path(__file__).parent / "preprocessor"


class TemplateEngine:
    """
    Hybrid template engine for IWFM file generation.

    Uses Jinja2 for rendering headers and structure, with direct numpy
    output for large data arrays to avoid string concatenation overhead.
    """

    def __init__(
        self,
        template_dir: Path | str | None = None,
        use_package_templates: bool = True,
    ) -> None:
        """
        Initialize the template engine.

        Args:
            template_dir: Custom template directory (optional)
            use_package_templates: If True, also load built-in templates
        """
        loaders = []

        # Add custom template directory if provided
        if template_dir:
            loaders.append(FileSystemLoader(str(template_dir)))

        # Add package templates (both new and legacy directories)
        if use_package_templates:
            if TEMPLATES_DIR.exists():
                loaders.append(FileSystemLoader(str(TEMPLATES_DIR)))
            if LEGACY_TEMPLATES_DIR.exists():
                loaders.append(FileSystemLoader(str(LEGACY_TEMPLATES_DIR)))

        if loaders:
            from jinja2 import ChoiceLoader

            self.env = Environment(
                loader=ChoiceLoader(loaders) if len(loaders) > 1 else loaders[0],
                autoescape=select_autoescape(default=False),
                trim_blocks=True,
                lstrip_blocks=True,
                keep_trailing_newline=True,
            )
        else:
            self.env = Environment(
                autoescape=select_autoescape(default=False),
                trim_blocks=True,
                lstrip_blocks=True,
                keep_trailing_newline=True,
            )

        # Add custom filters
        self._register_filters()

    def _register_filters(self) -> None:
        """Register custom Jinja2 filters for IWFM formatting."""
        # Import and register all filters from the filters module
        from pyiwfm.templates.filters import register_all_filters

        register_all_filters(self.env)

        # Also register legacy filter names for backward compatibility
        self.env.filters["fortran_float"] = _fortran_float
        self.env.filters["fortran_int"] = _fortran_int
        self.env.filters["iwfm_comment"] = _iwfm_comment
        self.env.filters["pad_right"] = _pad_right
        self.env.filters["pad_left"] = _pad_left
        self.env.filters["iwfm_timestamp"] = _iwfm_timestamp
        self.env.filters["dss_pathname"] = _dss_pathname
        self.env.filters["timeseries_ref"] = _timeseries_ref
        self.env.filters["iwfm_array_row"] = _iwfm_array_row

    def render_string(self, template_str: str, **context) -> str:
        """
        Render a template from a string.

        Args:
            template_str: Template string
            **context: Template context variables

        Returns:
            Rendered string
        """
        template = self.env.from_string(template_str)
        return template.render(**context)

    def render_template(self, template_name: str, **context) -> str:
        """
        Render a template from a file.

        Args:
            template_name: Name of the template file
            **context: Template context variables

        Returns:
            Rendered string
        """
        template = self.env.get_template(template_name)
        return template.render(**context)

    def render_with_comments(
        self,
        template_name: str,
        comment_metadata: "CommentMetadata | None" = None,
        section_name: str | None = None,
        inject_header: bool = True,
        **context,
    ) -> str:
        """
        Render a template with preserved comments.

        If comment_metadata is provided and contains preserved comments,
        the header and/or section comments will be injected into the
        rendered output.

        Args:
            template_name: Name of the template file
            comment_metadata: Optional CommentMetadata for comment preservation
            section_name: Optional section name for section-level comments
            inject_header: If True, inject preserved header if available
            **context: Template context variables

        Returns:
            Rendered string with preserved comments
        """
        # First, render the template
        rendered = self.render_template(template_name, **context)

        # If no comment metadata, return as-is
        if comment_metadata is None or not comment_metadata.has_comments():
            return rendered

        # Inject comments using CommentInjector
        from pyiwfm.io.comment_writer import CommentInjector

        injector = CommentInjector(comment_metadata)

        # Inject header if requested
        if inject_header:
            rendered = injector.inject_header(rendered)

        # Inject section comments if section specified
        if section_name:
            # Try to find a section marker in the rendered content
            section = comment_metadata.get_section(section_name)
            if section and section.header_comments:
                # Look for section-identifying text
                rendered = injector.inject_section_comments(
                    rendered, section_name, f"C  {section_name}"
                )

        return rendered

    def render_to_file(
        self,
        template_name: str,
        output_path: Path | str,
        **context,
    ) -> None:
        """
        Render a template to a file.

        Args:
            template_name: Name of the template file
            output_path: Path to output file
            **context: Template context variables
        """
        output = self.render_template(template_name, **context)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output)

    def render_hybrid(
        self,
        header_template: str,
        output_path: Path | str,
        arrays: dict[str, tuple[NDArray, str]] | None = None,
        **context,
    ) -> None:
        """
        Render a file with Jinja2 header and numpy data arrays.

        This hybrid approach uses Jinja2 for the header/structure and
        direct numpy output for large data arrays, which is much faster
        for large models.

        Args:
            header_template: Template name or string for the header
            output_path: Path to output file
            arrays: Dict mapping array name to (data, format_spec) tuples
            **context: Template context variables
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            # Render header with Jinja2
            if header_template.endswith(".j2"):
                header = self.render_template(header_template, **context)
            else:
                header = self.render_string(header_template, **context)

            f.write(header)

            # Write arrays with numpy (fast)
            if arrays:
                for name, (data, fmt) in arrays.items():
                    self._write_array(f, data, fmt)

    def _write_array(self, f: TextIO, data: NDArray, fmt: str) -> None:
        """
        Write a numpy array to file efficiently.

        Args:
            f: File object
            data: Numpy array
            fmt: Format string (e.g., '%10.4f' or '%5d')
        """
        if data.ndim == 1:
            for val in data:
                f.write(fmt % val)
                f.write("\n")
        else:
            # 2D array - write row by row
            for row in data:
                line = " ".join(fmt % val for val in row)
                f.write(line)
                f.write("\n")

    def render_hybrid_with_comments(
        self,
        header_template: str,
        output_path: Path | str,
        comment_metadata: "CommentMetadata | None" = None,
        arrays: dict[str, tuple[NDArray, str]] | None = None,
        **context,
    ) -> None:
        """
        Render a file with Jinja2 header, numpy arrays, and preserved comments.

        This is like render_hybrid but with support for comment preservation.
        If comment_metadata contains a preserved header, it replaces the
        template-generated header.

        Args:
            header_template: Template name or string for the header
            output_path: Path to output file
            comment_metadata: Optional CommentMetadata for preservation
            arrays: Dict mapping array name to (data, format_spec) tuples
            **context: Template context variables
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            # Check for preserved header
            has_preserved = (
                comment_metadata is not None
                and comment_metadata.header_block
            )

            if has_preserved:
                # Use preserved header
                header = "\n".join(comment_metadata.header_block) + "\n"
            else:
                # Render header with Jinja2
                if header_template.endswith(".j2"):
                    header = self.render_template(header_template, **context)
                else:
                    header = self.render_string(header_template, **context)

            f.write(header)

            # Write arrays with numpy (fast)
            if arrays:
                for name, (data, fmt) in arrays.items():
                    self._write_array(f, data, fmt)


class IWFMFileWriter:
    """
    High-level writer for IWFM input files.

    Provides methods for writing specific IWFM file types with
    proper formatting.
    """

    def __init__(self, engine: TemplateEngine | None = None) -> None:
        """
        Initialize the writer.

        Args:
            engine: TemplateEngine instance (creates default if None)
        """
        self.engine = engine or TemplateEngine()

    def write_nodes_file(
        self,
        output_path: Path | str,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        node_ids: NDArray[np.int32] | None = None,
    ) -> None:
        """
        Write an IWFM nodes file.

        Args:
            output_path: Path to output file
            x: X coordinates
            y: Y coordinates
            node_ids: Node IDs (default: 1 to n_nodes)
        """
        n_nodes = len(x)
        if node_ids is None:
            node_ids = np.arange(1, n_nodes + 1, dtype=np.int32)

        header = """C  Node coordinate data file
C  Generated by pyiwfm
C
C  ID             X              Y
{{ n_nodes }}                         / NNODES
"""
        # Combine arrays for efficient output
        data = np.column_stack([node_ids, x, y])

        output_path = Path(output_path)
        with open(output_path, "w") as f:
            f.write(self.engine.render_string(header, n_nodes=n_nodes))
            np.savetxt(f, data, fmt="%5d %14.6f %14.6f")

    def write_elements_file(
        self,
        output_path: Path | str,
        vertices: NDArray[np.int32],
        subregions: NDArray[np.int32],
        element_ids: NDArray[np.int32] | None = None,
        n_subregions: int | None = None,
    ) -> None:
        """
        Write an IWFM elements file.

        Args:
            output_path: Path to output file
            vertices: Vertex array (n_elements, 4), 0 for missing 4th vertex
            subregions: Subregion IDs for each element
            element_ids: Element IDs (default: 1 to n_elements)
            n_subregions: Number of subregions (default: max of subregions)
        """
        n_elements = len(vertices)
        if element_ids is None:
            element_ids = np.arange(1, n_elements + 1, dtype=np.int32)
        if n_subregions is None:
            n_subregions = int(subregions.max())

        header = """C  Element definition data file
C  Generated by pyiwfm
C
C  ID   V1    V2    V3    V4   SR
{{ n_elements }}                         / NELEM
{{ n_subregions }}                         / NSUBREGION
"""
        # Combine arrays
        data = np.column_stack([element_ids, vertices, subregions])

        output_path = Path(output_path)
        with open(output_path, "w") as f:
            f.write(
                self.engine.render_string(
                    header, n_elements=n_elements, n_subregions=n_subregions
                )
            )
            np.savetxt(f, data, fmt="%5d %5d %5d %5d %5d %3d")

    def write_stratigraphy_file(
        self,
        output_path: Path | str,
        gs_elev: NDArray[np.float64],
        top_elev: NDArray[np.float64],
        bottom_elev: NDArray[np.float64],
        node_ids: NDArray[np.int32] | None = None,
    ) -> None:
        """
        Write an IWFM stratigraphy file.

        Args:
            output_path: Path to output file
            gs_elev: Ground surface elevations (n_nodes,)
            top_elev: Layer top elevations (n_nodes, n_layers)
            bottom_elev: Layer bottom elevations (n_nodes, n_layers)
            node_ids: Node IDs (default: 1 to n_nodes)
        """
        n_nodes = len(gs_elev)
        n_layers = top_elev.shape[1]

        if node_ids is None:
            node_ids = np.arange(1, n_nodes + 1, dtype=np.int32)

        # Build header with layer columns
        layer_cols = "  ".join(
            [f"L{i+1}_TOP  L{i+1}_BOT" for i in range(n_layers)]
        )

        header = f"""C  Stratigraphy data file
C  Generated by pyiwfm
C
C  ID        GS  {layer_cols}
{{{{ n_nodes }}}}                         / NNODES
{{{{ n_layers }}}}                         / NLAYERS
"""
        # Build data array: ID, GS, then alternating top/bottom for each layer
        columns = [node_ids.reshape(-1, 1).astype(float), gs_elev.reshape(-1, 1)]
        for layer in range(n_layers):
            columns.append(top_elev[:, layer].reshape(-1, 1))
            columns.append(bottom_elev[:, layer].reshape(-1, 1))

        data = np.hstack(columns)

        # Build format string
        fmt = "%5d %10.4f" + " %10.4f %10.4f" * n_layers

        output_path = Path(output_path)
        with open(output_path, "w") as f:
            f.write(
                self.engine.render_string(header, n_nodes=n_nodes, n_layers=n_layers)
            )
            np.savetxt(f, data, fmt=fmt)


# Jinja2 filters for IWFM formatting


def _fortran_float(value: float, width: int = 14, decimals: int = 6) -> str:
    """Format a float in Fortran style."""
    return f"{value:{width}.{decimals}f}"


def _fortran_int(value: int, width: int = 10) -> str:
    """Format an integer in Fortran style."""
    return f"{value:{width}d}"


def _iwfm_comment(text: str) -> str:
    """Format text as an IWFM comment."""
    return f"C  {text}"


def _pad_right(text: str, width: int) -> str:
    """Pad text on the right to a fixed width."""
    return text.ljust(width)


def _pad_left(text: str, width: int) -> str:
    """Pad text on the left to a fixed width."""
    return text.rjust(width)


def _iwfm_timestamp(dt: datetime | np.datetime64 | str) -> str:
    """
    Format a datetime as an IWFM timestamp string.

    IWFM uses the format MM/DD/YYYY_HH:MM (exactly 16 characters).
    Midnight (00:00) is represented as 24:00 of the previous day.

    Args:
        dt: datetime object, numpy datetime64, or string

    Returns:
        Formatted timestamp string (16 chars)
    """
    if isinstance(dt, str):
        return dt

    from pyiwfm.io.timeseries_ascii import format_iwfm_timestamp
    return format_iwfm_timestamp(dt)


def _dss_pathname(
    a_part: str = "",
    b_part: str = "",
    c_part: str = "",
    d_part: str = "",
    e_part: str = "",
    f_part: str = "",
) -> str:
    """
    Build an HEC-DSS pathname from its parts.

    DSS pathname format: /A/B/C/D/E/F/
    - A: Project/Basin
    - B: Location
    - C: Parameter (FLOW, HEAD, etc.)
    - D: Date window
    - E: Time interval (1DAY, 1HOUR, etc.)
    - F: Version

    Args:
        a_part: Project/Basin part
        b_part: Location part
        c_part: Parameter part
        d_part: Date part
        e_part: Interval part
        f_part: Version part

    Returns:
        Formatted DSS pathname
    """
    return f"/{a_part}/{b_part}/{c_part}/{d_part}/{e_part}/{f_part}/"


def _timeseries_ref(
    filepath: str,
    column: int = 1,
    factor: float = 1.0,
) -> str:
    """
    Format a time series file reference for IWFM input files.

    IWFM uses file references in the format:
    FILEPATH  COLUMN  FACTOR

    Args:
        filepath: Path to the time series file
        column: Column number to read (1-based)
        factor: Conversion factor

    Returns:
        Formatted file reference string
    """
    return f"{filepath:<40}  {column:>3}  {factor:>10.4f}"


def _iwfm_array_row(
    values: list | np.ndarray,
    fmt: str = "%14.6f",
    sep: str = "  ",
) -> str:
    """
    Format a row of array values for IWFM output.

    Args:
        values: List or array of values
        fmt: Printf-style format string
        sep: Separator between values

    Returns:
        Formatted row string
    """
    if isinstance(values, np.ndarray):
        values = values.tolist()
    return sep.join(fmt % v for v in values)
