"""Unit tests for CLI budget and zbudget subcommands."""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

from pyiwfm.cli.budget import add_budget_parser, run_budget
from pyiwfm.cli.zbudget import add_zbudget_parser, run_zbudget

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_budget_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(dest="command")
    add_budget_parser(subs)
    return parser


def _make_zbudget_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(dest="command")
    add_zbudget_parser(subs)
    return parser


# ---------------------------------------------------------------------------
# run_budget
# ---------------------------------------------------------------------------


class TestRunBudget:
    """Tests for run_budget()."""

    def test_missing_control_file_returns_1(self, tmp_path: Path) -> None:
        args = argparse.Namespace(
            control_file=str(tmp_path / "nonexistent.bud"),
            output_dir=None,
        )
        result = run_budget(args)
        assert result == 1

    @patch("pyiwfm.io.budget_excel.budget_control_to_excel")
    @patch("pyiwfm.io.budget_control.read_budget_control")
    def test_success_returns_0(
        self, mock_read: MagicMock, mock_excel: MagicMock, tmp_path: Path
    ) -> None:
        ctl = tmp_path / "budget.bud"
        ctl.touch()

        mock_config = MagicMock()
        mock_read.return_value = mock_config
        mock_excel.return_value = [tmp_path / "out.xlsx"]

        args = argparse.Namespace(control_file=str(ctl), output_dir=None)
        result = run_budget(args)
        assert result == 0
        mock_read.assert_called_once()
        mock_excel.assert_called_once_with(mock_config)

    @patch("pyiwfm.io.budget_excel.budget_control_to_excel")
    @patch("pyiwfm.io.budget_control.read_budget_control")
    def test_no_files_generated_returns_1(
        self, mock_read: MagicMock, mock_excel: MagicMock, tmp_path: Path
    ) -> None:
        ctl = tmp_path / "budget.bud"
        ctl.touch()
        mock_read.return_value = MagicMock()
        mock_excel.return_value = []

        args = argparse.Namespace(control_file=str(ctl), output_dir=None)
        result = run_budget(args)
        assert result == 1

    @patch("pyiwfm.io.budget_excel.budget_control_to_excel")
    @patch("pyiwfm.io.budget_control.read_budget_control")
    def test_output_dir_override(
        self, mock_read: MagicMock, mock_excel: MagicMock, tmp_path: Path
    ) -> None:
        ctl = tmp_path / "budget.bud"
        ctl.touch()
        out_dir = tmp_path / "custom_out"

        spec = MagicMock()
        spec.output_file = Path("original.xlsx")
        mock_config = MagicMock()
        mock_config.budgets = [spec]
        mock_read.return_value = mock_config
        mock_excel.return_value = [out_dir / "original.xlsx"]

        args = argparse.Namespace(control_file=str(ctl), output_dir=str(out_dir))
        result = run_budget(args)
        assert result == 0
        assert spec.output_file == out_dir / "original.xlsx"


# ---------------------------------------------------------------------------
# run_zbudget
# ---------------------------------------------------------------------------


class TestRunZbudget:
    """Tests for run_zbudget()."""

    def test_missing_control_file_returns_1(self, tmp_path: Path) -> None:
        args = argparse.Namespace(
            control_file=str(tmp_path / "nonexistent.zbud"),
            output_dir=None,
        )
        result = run_zbudget(args)
        assert result == 1

    @patch("pyiwfm.io.zbudget_excel.zbudget_control_to_excel")
    @patch("pyiwfm.io.zbudget_control.read_zbudget_control")
    def test_success_returns_0(
        self, mock_read: MagicMock, mock_excel: MagicMock, tmp_path: Path
    ) -> None:
        ctl = tmp_path / "zbudget.zbud"
        ctl.touch()

        mock_config = MagicMock()
        mock_read.return_value = mock_config
        mock_excel.return_value = [tmp_path / "zout.xlsx"]

        args = argparse.Namespace(control_file=str(ctl), output_dir=None)
        result = run_zbudget(args)
        assert result == 0
        mock_read.assert_called_once()
        mock_excel.assert_called_once_with(mock_config)

    @patch("pyiwfm.io.zbudget_excel.zbudget_control_to_excel")
    @patch("pyiwfm.io.zbudget_control.read_zbudget_control")
    def test_no_files_generated_returns_1(
        self, mock_read: MagicMock, mock_excel: MagicMock, tmp_path: Path
    ) -> None:
        ctl = tmp_path / "zbudget.zbud"
        ctl.touch()
        mock_read.return_value = MagicMock()
        mock_excel.return_value = []

        args = argparse.Namespace(control_file=str(ctl), output_dir=None)
        result = run_zbudget(args)
        assert result == 1

    @patch("pyiwfm.io.zbudget_excel.zbudget_control_to_excel")
    @patch("pyiwfm.io.zbudget_control.read_zbudget_control")
    def test_output_dir_override(
        self, mock_read: MagicMock, mock_excel: MagicMock, tmp_path: Path
    ) -> None:
        ctl = tmp_path / "zbudget.zbud"
        ctl.touch()
        out_dir = tmp_path / "custom_out"

        spec = MagicMock()
        spec.output_file = Path("zone_out.xlsx")
        mock_config = MagicMock()
        mock_config.zbudgets = [spec]
        mock_read.return_value = mock_config
        mock_excel.return_value = [out_dir / "zone_out.xlsx"]

        args = argparse.Namespace(control_file=str(ctl), output_dir=str(out_dir))
        result = run_zbudget(args)
        assert result == 0
        assert spec.output_file == out_dir / "zone_out.xlsx"
