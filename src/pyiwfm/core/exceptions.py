"""Custom exceptions for pyiwfm package."""

from __future__ import annotations


class PyIWFMError(Exception):
    """Base exception for all pyiwfm errors."""

    pass


class MeshError(PyIWFMError):
    """Error related to mesh operations."""

    pass


class StratigraphyError(PyIWFMError):
    """Error related to stratigraphy operations."""

    pass


class ValidationError(PyIWFMError):
    """Error raised when model validation fails."""

    def __init__(self, message: str, errors: list[str] | None = None) -> None:
        super().__init__(message)
        self.errors = errors or []


class IWFMIOError(PyIWFMError):
    """Error related to file I/O operations."""

    pass


# Backwards-compatible alias
IOError = IWFMIOError  # noqa: A001


class FileFormatError(IWFMIOError):
    """Error raised when file format is invalid."""

    def __init__(self, message: str, line_number: int | None = None) -> None:
        super().__init__(message)
        self.line_number = line_number


class ComponentError(PyIWFMError):
    """Error related to model component operations."""

    pass


class ComponentLoadError(ComponentError):
    """Raised when a model component fails to load.

    Used by :class:`~pyiwfm.core.model.IWFMModel` constructors when
    ``strict=True`` is passed: any failure to parse a component file
    raises this exception with the component name, source file path,
    and original cause.

    Attributes
    ----------
    component_name : str
        Name of the component that failed to load (e.g. ``"streams"``,
        ``"lakes"``, ``"groundwater"``).
    source_file : pathlib.Path | None
        Path to the file that failed to parse, if known.
    """

    def __init__(
        self,
        component_name: str,
        source_file: object = None,
        cause: BaseException | None = None,
    ) -> None:
        self.component_name = component_name
        self.source_file = source_file
        cause_msg = f": {cause.__class__.__name__}: {cause}" if cause is not None else ""
        file_msg = f" from {source_file}" if source_file is not None else ""
        super().__init__(f"Failed to load {component_name} component{file_msg}{cause_msg}")


class ConnectorError(PyIWFMError):
    """Error related to component connector operations."""

    pass
