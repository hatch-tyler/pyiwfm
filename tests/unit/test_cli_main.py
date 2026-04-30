"""Unit tests for CLI main entry point (__init__.py)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from pyiwfm.cli import main
from pyiwfm.core.exceptions import (
    ComponentLoadError,
    FileFormatError,
    MeshError,
)


class TestMain:
    """Tests for main() CLI entry point."""

    def test_no_args_returns_0(self) -> None:
        result = main([])
        assert result == 0

    def test_viewer_help_exits(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["viewer", "--help"])
        assert exc_info.value.code == 0

    def test_export_help_exits(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["export", "--help"])
        assert exc_info.value.code == 0

    def test_unknown_command_exits(self) -> None:
        with pytest.raises(SystemExit):
            main(["nonexistent_command"])

    @patch("pyiwfm.cli.viewer.run_viewer")
    def test_dispatches_to_viewer(self, mock_run_viewer) -> None:
        mock_run_viewer.return_value = 0
        result = main(["viewer", "--model-dir", "/tmp/fake"])
        mock_run_viewer.assert_called_once()
        assert result == 0

    @patch("pyiwfm.cli.export.run_export")
    def test_dispatches_to_export(self, mock_run_export) -> None:
        mock_run_export.return_value = 0
        result = main(["export", "--model-dir", "/tmp/fake"])
        mock_run_export.assert_called_once()
        assert result == 0


class TestTopLevelExceptionHandler:
    """The CLI catches expected user-facing errors and turns them into
    clean stderr lines + exit code 1, instead of leaking a traceback.

    Programmer bugs (AttributeError etc.) propagate so they surface
    with the full traceback during development.
    """

    @patch("pyiwfm.cli.viewer.run_viewer")
    def test_pyiwfm_error_returns_1(self, mock_run_viewer, capsys) -> None:
        mock_run_viewer.side_effect = MeshError("mesh has no nodes")
        result = main(["viewer", "--model-dir", "/tmp/fake"])
        assert result == 1
        captured = capsys.readouterr()
        assert "error: MeshError: mesh has no nodes" in captured.err
        assert "Traceback" not in captured.err

    @patch("pyiwfm.cli.viewer.run_viewer")
    def test_file_format_error_includes_line_number(self, mock_run_viewer, capsys) -> None:
        mock_run_viewer.side_effect = FileFormatError("expected NNODES", line_number=42)
        result = main(["viewer", "--model-dir", "/tmp/fake"])
        assert result == 1
        captured = capsys.readouterr()
        assert "FileFormatError (line 42)" in captured.err
        assert "expected NNODES" in captured.err

    @patch("pyiwfm.cli.viewer.run_viewer")
    def test_component_load_error_returns_1(self, mock_run_viewer, capsys) -> None:
        mock_run_viewer.side_effect = ComponentLoadError(
            component_name="streams", source_file="/tmp/missing.dat"
        )
        result = main(["viewer", "--model-dir", "/tmp/fake"])
        assert result == 1
        captured = capsys.readouterr()
        assert "ComponentLoadError" in captured.err
        assert "streams" in captured.err

    @patch("pyiwfm.cli.viewer.run_viewer")
    def test_file_not_found_error_returns_1(self, mock_run_viewer, capsys) -> None:
        mock_run_viewer.side_effect = FileNotFoundError("/nope/missing.in")
        result = main(["viewer", "--model-dir", "/nope"])
        assert result == 1
        captured = capsys.readouterr()
        assert "FileNotFoundError" in captured.err
        assert "/nope/missing.in" in captured.err

    @patch("pyiwfm.cli.viewer.run_viewer")
    def test_unexpected_exception_propagates(self, mock_run_viewer) -> None:
        # Programmer bug — should NOT be caught.
        mock_run_viewer.side_effect = AttributeError("oops")
        with pytest.raises(AttributeError, match="oops"):
            main(["viewer", "--model-dir", "/tmp/fake"])

    @patch("pyiwfm.cli.viewer.run_viewer")
    def test_keyboard_interrupt_propagates(self, mock_run_viewer) -> None:
        mock_run_viewer.side_effect = KeyboardInterrupt()
        with pytest.raises(KeyboardInterrupt):
            main(["viewer", "--model-dir", "/tmp/fake"])
