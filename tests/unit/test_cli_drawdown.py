"""Tests for the ``pyiwfm drawdown`` CLI subcommand (Phase 2 drawdown).

Argument parsing and dispatch tested directly. End-to-end model load +
heads HDF read is mocked so the test doesn't need real model files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pyiwfm.cli.drawdown import (
    _parse_locations,
    _resolve_ts_plot_kinds,
    add_drawdown_parser,
    run_drawdown,
)
from pyiwfm.io.drawdown import (
    DrawdownAtLocation,
    DrawdownSnapshot,
    DrawdownTimeSeriesReport,
)


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(dest="command")
    add_drawdown_parser(subs)
    return parser


def _make_ts_report(n_locations: int = 1, n_timesteps: int = 3) -> DrawdownTimeSeriesReport:
    times = [f"t{i}" for i in range(n_timesteps)]
    locs = []
    for i in range(n_locations):
        dd = np.array([float(i + 1) * j for j in range(n_timesteps)])
        locs.append(
            DrawdownAtLocation(
                node_id=i + 1,
                layer=1,
                times=times,
                drawdown=dd,
                max_drawdown=float(dd[-1]),
                max_drawdown_timestep=n_timesteps - 1,
                final_drawdown=float(dd[-1]),
            )
        )
    return DrawdownTimeSeriesReport(
        locations=locs,
        n_locations=n_locations,
        n_timesteps=n_timesteps,
        reference_timestep=0,
        times=times,
    )


def _make_snapshot(n_nodes: int = 5) -> DrawdownSnapshot:
    return DrawdownSnapshot(
        timestep=2,
        layer=1,
        reference_timestep=0,
        kind="single",
        time_label="2024-01-03",
        node_ids=np.arange(1, n_nodes + 1, dtype=np.int32),
        drawdown=np.linspace(1.0, float(n_nodes), n_nodes),
        n_nodes=n_nodes,
    )


def _mock_model_with_mesh() -> MagicMock:
    from pyiwfm.core.mesh import AppGrid, Element, Node

    nodes = {i: Node(id=i, x=float(i), y=0.0) for i in range(1, 6)}
    mesh = AppGrid(nodes=nodes, elements={1: Element(id=1, vertices=[1, 2, 3])})
    model = MagicMock()
    model.mesh = mesh
    model.metadata = {}
    model.groundwater = MagicMock()
    model.groundwater.n_layers = 2
    return model


class TestArgumentParsing:
    def test_required_args(self, tmp_path: Path):
        parser = _make_parser()
        args = parser.parse_args(
            ["drawdown", str(tmp_path), "--mode", "snapshot", "--timestep", "5"]
        )
        assert args.mode == "snapshot"
        assert args.timestep == 5
        assert args.layer == 1
        assert args.reference_timestep == 0
        assert args.no_map is False

    def test_mode_required(self, tmp_path: Path):
        parser = _make_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["drawdown", str(tmp_path)])

    def test_invalid_mode_rejected(self, tmp_path: Path):
        parser = _make_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["drawdown", str(tmp_path), "--mode", "histogram"])

    def test_all_options(self, tmp_path: Path):
        parser = _make_parser()
        args = parser.parse_args(
            [
                "drawdown",
                str(tmp_path),
                "--mode",
                "timeseries",
                "--locations",
                "1,1;42,2",
                "--reference-timestep",
                "10",
                "--output",
                "out.csv",
                "--plot",
                "timeseries",
                "--plot-dir",
                "myplots",
                "--no-map",
                "--crs",
                "EPSG:26910",
                "--heads-hdf",
                "/path/to/heads.hdf",
            ]
        )
        assert args.locations == "1,1;42,2"
        assert args.reference_timestep == 10
        assert args.no_map is True


class TestParseLocations:
    def test_none(self):
        assert _parse_locations(None) is None

    def test_single_pair(self):
        assert _parse_locations("1,2") == [(1, 2)]

    def test_multiple_pairs(self):
        assert _parse_locations("1,1;42,2;7,3") == [(1, 1), (42, 2), (7, 3)]

    def test_strips_whitespace(self):
        assert _parse_locations(" 1, 1 ; 2 , 2 ") == [(1, 1), (2, 2)]

    def test_invalid_format_raises(self):
        with pytest.raises(SystemExit, match="expected 'node,layer'"):
            _parse_locations("1")
        with pytest.raises(SystemExit, match="expected 'node,layer'"):
            _parse_locations("1,2,3")

    def test_non_integer_raises(self):
        with pytest.raises(SystemExit, match="Could not parse"):
            _parse_locations("foo,bar")


class TestResolveTsPlotKinds:
    def test_none(self):
        assert _resolve_ts_plot_kinds(None) == []

    def test_all_expands(self):
        assert set(_resolve_ts_plot_kinds(["all"])) == {"timeseries", "summary"}

    def test_dedup(self):
        assert _resolve_ts_plot_kinds(["timeseries", "timeseries"]) == ["timeseries"]


class TestRunDrawdown:
    def _args(self, tmp_path: Path, **overrides) -> argparse.Namespace:
        defaults: dict = {
            "model_dir": str(tmp_path),
            "mode": "snapshot",
            "locations": None,
            "timestep": 0,
            "layer": 1,
            "reference_timestep": 0,
            "output": None,
            "plot": None,
            "plot_dir": str(tmp_path / "plots"),
            "no_map": False,
            "crs": None,
            "heads_hdf": None,
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_missing_model_dir_returns_1(self, tmp_path: Path):
        args = self._args(tmp_path / "nonexistent", timestep=0)
        assert run_drawdown(args) == 1

    def test_timeseries_without_locations_returns_2(self, tmp_path: Path):
        args = self._args(tmp_path, mode="timeseries", timestep=None)
        assert run_drawdown(args) == 2

    def test_snapshot_without_timestep_returns_2(self, tmp_path: Path):
        args = self._args(tmp_path, mode="snapshot", timestep=None)
        assert run_drawdown(args) == 2

    @patch("pyiwfm.cli.drawdown._open_heads_loader")
    @patch("pyiwfm.cli.drawdown._load_model_from_dir")
    @patch("pyiwfm.io.drawdown.DrawdownComputer.build_timeseries_report")
    def test_timeseries_writes_csv(
        self,
        mock_build: MagicMock,
        mock_load: MagicMock,
        mock_loader: MagicMock,
        tmp_path: Path,
    ):
        mock_load.return_value = _mock_model_with_mesh()
        loader = MagicMock()
        loader.n_frames = 10
        loader.n_nodes = 5
        loader.n_layers = 2
        loader._file_path = Path("heads.hdf")
        mock_loader.return_value = loader
        mock_build.return_value = _make_ts_report(n_locations=2, n_timesteps=3)

        out = tmp_path / "drawdown.csv"
        args = self._args(
            tmp_path,
            mode="timeseries",
            locations="1,1;2,1",
            output=str(out),
        )
        result = run_drawdown(args)

        assert result == 0
        assert out.exists()
        first_line = out.read_text().splitlines()[0]
        assert first_line.startswith("node_id,")

    @patch("pyiwfm.cli.drawdown._open_heads_loader")
    @patch("pyiwfm.cli.drawdown._load_model_from_dir")
    @patch("pyiwfm.io.drawdown.DrawdownComputer.build_timeseries_report")
    def test_timeseries_renders_plots(
        self,
        mock_build: MagicMock,
        mock_load: MagicMock,
        mock_loader: MagicMock,
        tmp_path: Path,
    ):
        mock_load.return_value = _mock_model_with_mesh()
        loader = MagicMock()
        loader.n_frames = 10
        loader.n_nodes = 5
        loader.n_layers = 2
        loader._file_path = Path("heads.hdf")
        mock_loader.return_value = loader
        mock_build.return_value = _make_ts_report(n_locations=3)

        plot_dir = tmp_path / "plots"
        args = self._args(
            tmp_path,
            mode="timeseries",
            locations="1,1;2,1;3,1",
            plot=["all"],
            plot_dir=str(plot_dir),
        )
        result = run_drawdown(args)

        assert result == 0
        assert (plot_dir / "drawdown_timeseries.png").exists()
        assert (plot_dir / "drawdown_summary.png").exists()

    @patch("pyiwfm.cli.drawdown._open_heads_loader")
    @patch("pyiwfm.cli.drawdown._load_model_from_dir")
    @patch("pyiwfm.io.drawdown.DrawdownComputer.build_snapshot")
    def test_snapshot_writes_map_and_geojson(
        self,
        mock_build: MagicMock,
        mock_load: MagicMock,
        mock_loader: MagicMock,
        tmp_path: Path,
    ):
        mock_load.return_value = _mock_model_with_mesh()
        loader = MagicMock()
        loader.n_frames = 10
        loader.n_nodes = 5
        loader.n_layers = 2
        loader._file_path = Path("heads.hdf")
        mock_loader.return_value = loader
        mock_build.return_value = _make_snapshot(n_nodes=5)

        plot_dir = tmp_path / "plots"
        args = self._args(
            tmp_path,
            mode="snapshot",
            timestep=2,
            plot_dir=str(plot_dir),
        )
        result = run_drawdown(args)

        assert result == 0
        # Map PNG and GeoJSON written
        assert (plot_dir / "drawdown_snapshot_layer1.png").exists()
        assert (plot_dir / "drawdown_snapshot_layer1.geojson").exists()

    @patch("pyiwfm.cli.drawdown._open_heads_loader")
    @patch("pyiwfm.cli.drawdown._load_model_from_dir")
    @patch("pyiwfm.io.drawdown.DrawdownComputer.build_max_snapshot")
    def test_max_writes_map_and_geojson(
        self,
        mock_build: MagicMock,
        mock_load: MagicMock,
        mock_loader: MagicMock,
        tmp_path: Path,
    ):
        mock_load.return_value = _mock_model_with_mesh()
        loader = MagicMock()
        loader.n_frames = 10
        loader.n_nodes = 5
        loader.n_layers = 2
        loader._file_path = Path("heads.hdf")
        mock_loader.return_value = loader
        max_snap = _make_snapshot(n_nodes=5)
        max_snap.kind = "max"
        max_snap.timestep = -1
        mock_build.return_value = max_snap

        plot_dir = tmp_path / "plots"
        args = self._args(
            tmp_path,
            mode="max",
            timestep=None,
            plot_dir=str(plot_dir),
        )
        result = run_drawdown(args)

        assert result == 0
        assert (plot_dir / "drawdown_max_layer1.png").exists()
        assert (plot_dir / "drawdown_max_layer1.geojson").exists()

    @patch("pyiwfm.cli.drawdown._open_heads_loader")
    @patch("pyiwfm.cli.drawdown._load_model_from_dir")
    @patch("pyiwfm.io.drawdown.DrawdownComputer.build_snapshot")
    def test_snapshot_no_map_skips_outputs(
        self,
        mock_build: MagicMock,
        mock_load: MagicMock,
        mock_loader: MagicMock,
        tmp_path: Path,
    ):
        mock_load.return_value = _mock_model_with_mesh()
        loader = MagicMock()
        loader.n_frames = 10
        loader.n_nodes = 5
        loader.n_layers = 2
        loader._file_path = Path("heads.hdf")
        mock_loader.return_value = loader
        mock_build.return_value = _make_snapshot(n_nodes=5)

        plot_dir = tmp_path / "plots"
        args = self._args(
            tmp_path,
            mode="snapshot",
            timestep=2,
            no_map=True,
            plot_dir=str(plot_dir),
        )
        result = run_drawdown(args)

        assert result == 0
        # No map files written when --no-map
        assert not (plot_dir / "drawdown_snapshot_layer1.png").exists()

    @patch("pyiwfm.cli.drawdown._open_heads_loader")
    @patch("pyiwfm.cli.drawdown._load_model_from_dir")
    @patch("pyiwfm.io.drawdown.DrawdownComputer.build_snapshot")
    def test_snapshot_writes_json_when_requested(
        self,
        mock_build: MagicMock,
        mock_load: MagicMock,
        mock_loader: MagicMock,
        tmp_path: Path,
    ):
        mock_load.return_value = _mock_model_with_mesh()
        loader = MagicMock()
        loader.n_frames = 10
        loader.n_nodes = 5
        loader.n_layers = 2
        loader._file_path = Path("heads.hdf")
        mock_loader.return_value = loader
        mock_build.return_value = _make_snapshot(n_nodes=5)

        out = tmp_path / "snap.json"
        args = self._args(
            tmp_path,
            mode="snapshot",
            timestep=2,
            output=str(out),
            no_map=True,
            plot_dir=str(tmp_path / "p"),
        )
        result = run_drawdown(args)

        assert result == 0
        assert out.exists()
        loaded = json.loads(out.read_text())
        assert loaded["kind"] == "single"
        assert loaded["n_nodes"] == 5
