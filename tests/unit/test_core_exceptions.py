"""Unit tests for pyiwfm custom exceptions (core/exceptions.py)."""

from __future__ import annotations

import pytest

from pyiwfm.core.exceptions import (
    ComponentError,
    ConnectorError,
    FileFormatError,
    IOError,
    MeshError,
    PyIWFMError,
    StratigraphyError,
    ValidationError,
)


class TestExceptionHierarchy:
    """Tests for the exception class hierarchy."""

    def test_pyiwfm_error_is_exception(self) -> None:
        assert issubclass(PyIWFMError, Exception)

    def test_mesh_error_inherits(self) -> None:
        assert issubclass(MeshError, PyIWFMError)

    def test_stratigraphy_error_inherits(self) -> None:
        assert issubclass(StratigraphyError, PyIWFMError)

    def test_validation_error_inherits(self) -> None:
        assert issubclass(ValidationError, PyIWFMError)

    def test_io_error_inherits(self) -> None:
        assert issubclass(IOError, PyIWFMError)

    def test_file_format_error_inherits_from_io(self) -> None:
        assert issubclass(FileFormatError, IOError)
        assert issubclass(FileFormatError, PyIWFMError)

    def test_component_error_inherits(self) -> None:
        assert issubclass(ComponentError, PyIWFMError)

    def test_connector_error_inherits(self) -> None:
        assert issubclass(ConnectorError, PyIWFMError)


class TestExceptionInstantiation:
    """Tests for exception creation and attributes."""

    def test_pyiwfm_error(self) -> None:
        exc = PyIWFMError("test error")
        assert str(exc) == "test error"

    def test_mesh_error(self) -> None:
        exc = MeshError("bad mesh")
        assert str(exc) == "bad mesh"

    def test_validation_error_with_errors(self) -> None:
        exc = ValidationError("invalid", errors=["error 1", "error 2"])
        assert str(exc) == "invalid"
        assert exc.errors == ["error 1", "error 2"]

    def test_validation_error_default_errors(self) -> None:
        exc = ValidationError("invalid")
        assert exc.errors == []

    def test_file_format_error_with_line(self) -> None:
        exc = FileFormatError("bad format", line_number=42)
        assert str(exc) == "bad format"
        assert exc.line_number == 42

    def test_file_format_error_no_line(self) -> None:
        exc = FileFormatError("bad format")
        assert exc.line_number is None


class TestExceptionRaising:
    """Tests for raising and catching exceptions."""

    def test_catch_as_pyiwfm_error(self) -> None:
        with pytest.raises(PyIWFMError):
            raise MeshError("test")

    def test_catch_io_catches_file_format(self) -> None:
        with pytest.raises(IOError):
            raise FileFormatError("test")

    def test_catch_specific(self) -> None:
        with pytest.raises(ConnectorError):
            raise ConnectorError("bad connection")
