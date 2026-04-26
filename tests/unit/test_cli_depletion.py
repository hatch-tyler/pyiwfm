"""Tests for the ``pyiwfm depletion`` CLI subcommand (Phase 2.2.a-v).

Argument parsing and dispatch are tested directly. The end-to-end
loader path (find simulation file, load model, compute depletion) is
tested with mocks so the test doesn't need the IWFM Sample Model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pyiwfm.cli.depletion import (
    _parse_id_list,
    _resolve_plot_kinds,
    add_depletion_parser,
    run_depletion,
)
from pyiwfm.io.stream_depletion import (
    BudgetOutputMissingError,
    StreamDepletionReport,
    StreamDepletionResult,
    StreamNodeDepletionReport,
    StreamNodeDepletionResult,
)


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(dest="command")
    add_depletion_parser(subs)
    return parser


def _make_reach_report() -> StreamDepletionReport:
    times = ["t0", "t1"]
    r = StreamDepletionResult(
        reach_id=1,
        reach_name="Main",
        times=times,
        baseline_flow=np.array([10.0, 11.0]),
        scenario_flow=np.array([8.0, 9.0]),
        depletion=np.array([2.0, 2.0]),
        cumulative_depletion=np.array([2.0, 4.0]),
        max_depletion=2.0,
        max_depletion_timestep=0,
        total_depletion=4.0,
    )
    return StreamDepletionReport(
        results=[r],
        n_reaches=1,
        n_timesteps=2,
        total_max_depletion=2.0,
        total_cumulative_depletion=4.0,
    )


def _make_node_report() -> StreamNodeDepletionReport:
    n = StreamNodeDepletionResult(
        stream_node_id=1,
        times=["t0"],
        baseline_sa_flux=np.array([5.0]),
        scenario_sa_flux=np.array([3.0]),
        depletion=np.array([2.0]),
        cumulative_depletion=np.array([2.0]),
        max_depletion=2.0,
        max_depletion_timestep=0,
        total_depletion=2.0,
    )
    return StreamNodeDepletionReport(
        results=[n],
        n_stream_nodes=1,
        n_timesteps=1,
        total_max_depletion=2.0,
        total_cumulative_depletion=2.0,
    )


class TestArgumentParsing:
    def test_required_args_only(self, tmp_path: Path):
        parser = _make_parser()
        args = parser.parse_args(["depletion", str(tmp_path), str(tmp_path)])
        assert args.command == "depletion"
        assert args.baseline_dir == str(tmp_path)
        assert args.scenario_dir == str(tmp_path)
        assert args.output is None
        assert args.plot is None
        assert args.map is False
        assert args.node_level is False
        assert args.metric == "max"

    def test_all_options(self, tmp_path: Path):
        parser = _make_parser()
        args = parser.parse_args(
            [
                "depletion",
                "/base",
                "/scen",
                "--output",
                "out.xlsx",
                "--plot",
                "cumulative",
                "--plot",
                "summary",
                "--plot-dir",
                "myplots",
                "--map",
                "--node-level",
                "--metric",
                "total",
                "--node-ids",
                "1,2,3",
                "--sa-column",
                "Gain from GW (+)",
                "--crs",
                "EPSG:26910",
            ]
        )
        assert args.plot == ["cumulative", "summary"]
        assert args.map is True
        assert args.node_level is True
        assert args.metric == "total"
        assert args.node_ids == "1,2,3"
        assert args.sa_column == "Gain from GW (+)"
        assert args.crs == "EPSG:26910"

    def test_invalid_metric_rejected(self, tmp_path: Path):
        parser = _make_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["depletion", "/a", "/b", "--metric", "median"])

    def test_invalid_plot_rejected(self):
        parser = _make_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["depletion", "/a", "/b", "--plot", "histogram"])


class TestParseIdList:
    def test_none_returns_none(self):
        assert _parse_id_list(None) is None

    def test_empty_string_returns_none(self):
        assert _parse_id_list("") is None

    def test_comma_separated(self):
        assert _parse_id_list("1,3,5") == [1, 3, 5]

    def test_strips_whitespace(self):
        assert _parse_id_list("1, 2, 3") == [1, 2, 3]

    def test_invalid_raises_systemexit(self):
        with pytest.raises(SystemExit, match="Could not parse"):
            _parse_id_list("1,foo,3")


class TestResolvePlotKinds:
    def test_none(self):
        assert _resolve_plot_kinds(None) == []

    def test_all_expands(self):
        result = _resolve_plot_kinds(["all"])
        assert set(result) == {"cumulative", "timeseries", "summary"}

    def test_preserves_user_order(self):
        assert _resolve_plot_kinds(["summary", "cumulative"]) == ["summary", "cumulative"]

    def test_dedup(self):
        assert _resolve_plot_kinds(["summary", "summary"]) == ["summary"]


class TestRunDepletion:
    def _args(self, tmp_path: Path, **overrides) -> argparse.Namespace:
        defaults: dict = {
            "baseline_dir": str(tmp_path / "baseline"),
            "scenario_dir": str(tmp_path / "scenario"),
            "output": None,
            "plot": None,
            "plot_dir": "depletion_plots",
            "map": False,
            "node_level": False,
            "metric": "max",
            "reach_ids": None,
            "node_ids": None,
            "sa_column": None,
            "crs": None,
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_missing_baseline_dir_returns_1(self, tmp_path: Path):
        args = self._args(tmp_path)
        # Neither directory exists
        result = run_depletion(args)
        assert result == 1

    def test_node_ids_without_node_level_returns_2(self, tmp_path: Path):
        (tmp_path / "baseline").mkdir()
        (tmp_path / "scenario").mkdir()
        args = self._args(tmp_path, node_ids="1,2")
        result = run_depletion(args)
        assert result == 2

    def test_reach_ids_with_node_level_returns_2(self, tmp_path: Path):
        (tmp_path / "baseline").mkdir()
        (tmp_path / "scenario").mkdir()
        args = self._args(tmp_path, reach_ids="1,2", node_level=True)
        result = run_depletion(args)
        assert result == 2

    @patch("pyiwfm.io.stream_depletion.compute_stream_depletion_from_models")
    @patch("pyiwfm.cli.depletion._load_model_from_dir")
    def test_reach_mode_writes_tabular_output(
        self,
        mock_load: MagicMock,
        mock_compute: MagicMock,
        tmp_path: Path,
    ):
        (tmp_path / "baseline").mkdir()
        (tmp_path / "scenario").mkdir()
        mock_load.return_value = MagicMock()
        mock_compute.return_value = _make_reach_report()

        out = tmp_path / "report.csv"
        args = self._args(tmp_path, output=str(out))
        result = run_depletion(args)

        assert result == 0
        assert out.exists()
        # CSV header is the public contract
        first_line = out.read_text().splitlines()[0]
        assert first_line.startswith("reach_id,")

    @patch("pyiwfm.io.stream_depletion.compute_stream_depletion_from_models")
    @patch("pyiwfm.cli.depletion._load_model_from_dir")
    def test_reach_mode_renders_plots(
        self,
        mock_load: MagicMock,
        mock_compute: MagicMock,
        tmp_path: Path,
    ):
        (tmp_path / "baseline").mkdir()
        (tmp_path / "scenario").mkdir()
        mock_load.return_value = MagicMock()
        mock_compute.return_value = _make_reach_report()

        plot_dir = tmp_path / "plots"
        args = self._args(
            tmp_path,
            plot=["all"],
            plot_dir=str(plot_dir),
        )
        result = run_depletion(args)

        assert result == 0
        # All three plot kinds rendered
        assert (plot_dir / "cumulative_depletion.png").exists()
        assert (plot_dir / "depletion_timeseries.png").exists()
        assert (plot_dir / "depletion_summary.png").exists()

    @patch("pyiwfm.io.stream_depletion.compute_stream_node_depletion")
    @patch("pyiwfm.cli.depletion._load_model_from_dir")
    def test_node_level_writes_json_report(
        self,
        mock_load: MagicMock,
        mock_compute: MagicMock,
        tmp_path: Path,
    ):
        (tmp_path / "baseline").mkdir()
        (tmp_path / "scenario").mkdir()
        mock_load.return_value = MagicMock()
        mock_compute.return_value = _make_node_report()

        out = tmp_path / "node_report.json"
        args = self._args(
            tmp_path,
            output=str(out),
            node_level=True,
        )
        result = run_depletion(args)

        assert result == 0
        assert out.exists()
        loaded = json.loads(out.read_text())
        assert loaded["n_stream_nodes"] == 1

    @patch("pyiwfm.io.stream_depletion.compute_stream_node_depletion")
    @patch("pyiwfm.cli.depletion._load_model_from_dir")
    def test_node_level_warns_on_non_json_extension(
        self,
        mock_load: MagicMock,
        mock_compute: MagicMock,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ):
        (tmp_path / "baseline").mkdir()
        (tmp_path / "scenario").mkdir()
        mock_load.return_value = MagicMock()
        mock_compute.return_value = _make_node_report()

        # Asked for .csv but node-level only writes JSON today
        out = tmp_path / "report.csv"
        args = self._args(tmp_path, output=str(out), node_level=True)
        result = run_depletion(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "only supports JSON" in captured.err
        # File still got written (as JSON content)
        loaded = json.loads(out.read_text())
        assert "stream_nodes" in loaded

    @patch("pyiwfm.io.stream_depletion.compute_stream_depletion_from_models")
    @patch("pyiwfm.cli.depletion._load_model_from_dir")
    def test_missing_budget_output_returns_1_with_message(
        self,
        mock_load: MagicMock,
        mock_compute: MagicMock,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ):
        (tmp_path / "baseline").mkdir()
        (tmp_path / "scenario").mkdir()
        mock_load.return_value = MagicMock()
        mock_compute.side_effect = BudgetOutputMissingError(
            model_label="baseline",
            kind="reach",
            metadata_key="stream_budget_file",
            reason="the stream main file did not declare this budget output",
        )

        args = self._args(tmp_path)
        result = run_depletion(args)

        assert result == 1
        err = capsys.readouterr().err
        assert "baseline model" in err
        # The remediation hint mentions the IWFM input lines
        assert "STRMRCHBUDFL" in err

    @patch("pyiwfm.io.stream_depletion.compute_stream_depletion_from_models")
    @patch("pyiwfm.cli.depletion._load_model_from_dir")
    def test_missing_sa_column_returns_1(
        self,
        mock_load: MagicMock,
        mock_compute: MagicMock,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ):
        (tmp_path / "baseline").mkdir()
        (tmp_path / "scenario").mkdir()
        mock_load.return_value = MagicMock()
        mock_compute.side_effect = KeyError(
            "Column 'Foo' not found in budget headers. Available columns: ['A', 'B']"
        )

        args = self._args(tmp_path)
        result = run_depletion(args)

        assert result == 1
        err = capsys.readouterr().err
        assert "not found in budget headers" in err
