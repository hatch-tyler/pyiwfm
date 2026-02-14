"""Unit tests for CLI export subcommand (export.py)."""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pyiwfm.cli.export import add_export_parser, run_export


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(dest="command")
    add_export_parser(subs)
    return parser


# ---------------------------------------------------------------------------
# add_export_parser
# ---------------------------------------------------------------------------


class TestAddExportParser:
    """Tests for parser argument registration."""

    def test_parser_creates_export_subcommand(self) -> None:
        parser = _make_parser()
        args = parser.parse_args(["export", "--model-dir", "/tmp/model"])
        assert args.command == "export"

    def test_default_format(self) -> None:
        parser = _make_parser()
        args = parser.parse_args(["export"])
        assert args.export_format == "all"

    def test_format_choices(self) -> None:
        parser = _make_parser()
        for fmt in ("vtk", "gpkg", "all"):
            args = parser.parse_args(["export", "--format", fmt])
            assert args.export_format == fmt

    def test_output_dir_default(self) -> None:
        parser = _make_parser()
        args = parser.parse_args(["export"])
        assert args.output_dir == Path("output")

    def test_func_set(self) -> None:
        parser = _make_parser()
        args = parser.parse_args(["export"])
        assert args.func is run_export


# ---------------------------------------------------------------------------
# run_export
# ---------------------------------------------------------------------------


class TestRunExport:
    """Tests for run_export()."""

    def test_model_dir_not_found(self, tmp_path: Path) -> None:
        args = argparse.Namespace(
            model_dir=tmp_path / "nonexistent",
            preprocessor=None,
            output_dir=tmp_path / "output",
            export_format="all",
            debug=False,
        )
        result = run_export(args)
        assert result == 1

    @patch("pyiwfm.cli._model_loader.load_model")
    def test_load_failure_returns_1(self, mock_load, tmp_path: Path) -> None:
        mock_load.side_effect = RuntimeError("cannot load")
        args = argparse.Namespace(
            model_dir=tmp_path,
            preprocessor=None,
            output_dir=tmp_path / "output",
            export_format="all",
            debug=False,
        )
        result = run_export(args)
        assert result == 1

    @patch("pyiwfm.visualization.VTKExporter")
    @patch("pyiwfm.cli._model_loader.load_model")
    def test_vtk_export_success(self, mock_load, mock_vtk_cls, tmp_path: Path) -> None:
        mock_model = MagicMock()
        mock_model.mesh = MagicMock()
        mock_model.stratigraphy = MagicMock()
        mock_model.streams = None
        mock_load.return_value = mock_model

        mock_exporter = MagicMock()
        mock_vtk_cls.return_value = mock_exporter

        args = argparse.Namespace(
            model_dir=tmp_path,
            preprocessor=None,
            output_dir=tmp_path / "output",
            export_format="vtk",
            debug=False,
        )

        result = run_export(args)
        assert result == 0

    @patch("pyiwfm.cli._model_loader.load_model")
    def test_no_mesh_skips_vtk(self, mock_load, tmp_path: Path) -> None:
        mock_model = MagicMock()
        mock_model.mesh = None
        mock_model.stratigraphy = None
        mock_model.streams = None
        mock_load.return_value = mock_model

        args = argparse.Namespace(
            model_dir=tmp_path,
            preprocessor=None,
            output_dir=tmp_path / "output",
            export_format="vtk",
            debug=False,
        )
        result = run_export(args)
        assert result == 0

    @patch("pyiwfm.cli._model_loader.load_model")
    def test_model_dir_from_preprocessor_abs(self, mock_load, tmp_path: Path) -> None:
        """When --model-dir is None but --preprocessor is absolute."""
        pp_path = tmp_path / "PP.in"
        pp_path.touch()
        mock_load.side_effect = RuntimeError("test stop")

        args = argparse.Namespace(
            model_dir=None,
            preprocessor=pp_path,
            output_dir=tmp_path / "output",
            export_format="all",
            debug=False,
        )
        run_export(args)
        call_args = mock_load.call_args
        assert call_args[0][0] == pp_path.parent
