"""Unit tests for CLI viewer subcommand (viewer.py)."""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pyiwfm.cli.viewer import add_viewer_parser, run_viewer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(dest="command")
    add_viewer_parser(subs)
    return parser


# ---------------------------------------------------------------------------
# add_viewer_parser
# ---------------------------------------------------------------------------


class TestAddViewerParser:
    """Tests for parser argument registration."""

    def test_parser_creates_viewer_subcommand(self) -> None:
        parser = _make_parser()
        args = parser.parse_args(["viewer", "--model-dir", "/tmp/model"])
        assert args.command == "viewer"

    def test_default_port(self) -> None:
        parser = _make_parser()
        args = parser.parse_args(["viewer"])
        assert args.port == 8080

    def test_default_host(self) -> None:
        parser = _make_parser()
        args = parser.parse_args(["viewer"])
        assert args.host == "127.0.0.1"

    def test_default_crs(self) -> None:
        parser = _make_parser()
        args = parser.parse_args(["viewer"])
        assert "+proj=utm" in args.crs

    def test_custom_port(self) -> None:
        parser = _make_parser()
        args = parser.parse_args(["viewer", "--port", "9000"])
        assert args.port == 9000

    def test_no_browser_flag(self) -> None:
        parser = _make_parser()
        args = parser.parse_args(["viewer", "--no-browser"])
        assert args.no_browser is True

    def test_debug_flag(self) -> None:
        parser = _make_parser()
        args = parser.parse_args(["viewer", "--debug"])
        assert args.debug is True

    def test_title_argument(self) -> None:
        parser = _make_parser()
        args = parser.parse_args(["viewer", "--title", "My Model"])
        assert args.title == "My Model"

    def test_func_set(self) -> None:
        parser = _make_parser()
        args = parser.parse_args(["viewer"])
        assert args.func is run_viewer


# ---------------------------------------------------------------------------
# run_viewer
# ---------------------------------------------------------------------------


class TestRunViewer:
    """Tests for the run_viewer() function."""

    def test_model_dir_not_found(self, tmp_path: Path) -> None:
        args = argparse.Namespace(
            model_dir=tmp_path / "nonexistent",
            preprocessor=None,
            simulation=None,
            host="127.0.0.1",
            port=8080,
            title=None,
            no_browser=True,
            debug=False,
            crs="EPSG:4326",
        )
        result = run_viewer(args)
        assert result == 1

    @patch("pyiwfm.cli._model_loader.load_model")
    def test_file_not_found_returns_1(self, mock_load, tmp_path: Path) -> None:
        mock_load.side_effect = FileNotFoundError("not found")
        args = argparse.Namespace(
            model_dir=tmp_path,
            preprocessor=None,
            simulation=None,
            host="127.0.0.1",
            port=8080,
            title=None,
            no_browser=True,
            debug=False,
            crs="EPSG:4326",
        )
        result = run_viewer(args)
        assert result == 1

    @patch("pyiwfm.cli._model_loader.load_model")
    def test_generic_exception_returns_1(self, mock_load, tmp_path: Path) -> None:
        mock_load.side_effect = RuntimeError("something broke")
        args = argparse.Namespace(
            model_dir=tmp_path,
            preprocessor=None,
            simulation=None,
            host="127.0.0.1",
            port=8080,
            title=None,
            no_browser=True,
            debug=False,
            crs="EPSG:4326",
        )
        result = run_viewer(args)
        assert result == 1

    @patch("pyiwfm.visualization.webapi.launch_viewer")
    @patch("pyiwfm.cli._model_loader.load_model")
    def test_import_error_returns_1(self, mock_load, mock_launch, tmp_path: Path) -> None:
        mock_model = MagicMock()
        mock_model.name = "TestModel"
        mock_model.n_nodes = 10
        mock_model.n_elements = 4
        mock_model.n_layers = 2
        mock_model.has_streams = False
        mock_model.has_lakes = False
        mock_load.return_value = mock_model
        mock_launch.side_effect = ImportError("no webapi")

        args = argparse.Namespace(
            model_dir=tmp_path,
            preprocessor=None,
            simulation=None,
            host="127.0.0.1",
            port=8080,
            title="Test",
            no_browser=True,
            debug=False,
            crs="EPSG:4326",
        )
        result = run_viewer(args)
        assert result == 1

    @patch("pyiwfm.visualization.webapi.launch_viewer")
    @patch("pyiwfm.cli._model_loader.load_model")
    def test_keyboard_interrupt_returns_0(self, mock_load, mock_launch, tmp_path: Path) -> None:
        mock_model = MagicMock()
        mock_model.name = "TestModel"
        mock_model.n_nodes = 10
        mock_model.n_elements = 4
        mock_model.n_layers = 2
        mock_model.has_streams = False
        mock_model.has_lakes = False
        mock_load.return_value = mock_model
        mock_launch.side_effect = KeyboardInterrupt()

        args = argparse.Namespace(
            model_dir=tmp_path,
            preprocessor=None,
            simulation=None,
            host="127.0.0.1",
            port=8080,
            title="Test",
            no_browser=True,
            debug=False,
            crs="EPSG:4326",
        )
        result = run_viewer(args)
        assert result == 0

    @patch("pyiwfm.visualization.webapi.launch_viewer")
    @patch("pyiwfm.cli._model_loader.load_model")
    def test_successful_launch(self, mock_load, mock_launch, tmp_path: Path) -> None:
        mock_model = MagicMock()
        mock_model.name = "TestModel"
        mock_model.n_nodes = 10
        mock_model.n_elements = 4
        mock_model.n_layers = 2
        mock_model.has_streams = False
        mock_model.has_lakes = False
        mock_load.return_value = mock_model

        args = argparse.Namespace(
            model_dir=tmp_path,
            preprocessor=None,
            simulation=None,
            host="127.0.0.1",
            port=8080,
            title="Test",
            no_browser=True,
            debug=False,
            crs="EPSG:4326",
        )
        result = run_viewer(args)

        assert result == 0
        mock_launch.assert_called_once()

    @patch("pyiwfm.cli._model_loader.load_model")
    def test_model_dir_from_simulation_abs(self, mock_load, tmp_path: Path) -> None:
        """When --model-dir is None but --simulation is absolute, derive from it."""
        sim_path = tmp_path / "Sim.in"
        sim_path.touch()

        mock_load.side_effect = FileNotFoundError("test")
        args = argparse.Namespace(
            model_dir=None,
            preprocessor=None,
            simulation=sim_path,
            host="127.0.0.1",
            port=8080,
            title=None,
            no_browser=True,
            debug=False,
            crs="EPSG:4326",
        )
        run_viewer(args)
        # Should have used sim_path.parent as model_dir
        call_args = mock_load.call_args
        assert call_args[0][0] == sim_path.parent
