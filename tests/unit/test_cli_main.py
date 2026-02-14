"""Unit tests for CLI main entry point (__init__.py)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from pyiwfm.cli import main


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
