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


class FileFormatError(IWFMIOError):
    """Error raised when file format is invalid."""

    def __init__(self, message: str, line_number: int | None = None) -> None:
        super().__init__(message)
        self.line_number = line_number


class ComponentError(PyIWFMError):
    """Error related to model component operations."""

    pass


class ConnectorError(PyIWFMError):
    """Error related to component connector operations."""

    pass
