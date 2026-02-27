"""Unit tests for CLI calctyphyd and iwfm2obs subcommands."""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

from pyiwfm.cli.calctyphyd import add_calctyphyd_parser, run_calctyphyd
from pyiwfm.cli.iwfm2obs import add_iwfm2obs_parser, run_iwfm2obs

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_calctyphyd_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(dest="command")
    add_calctyphyd_parser(subs)
    return parser


def _make_iwfm2obs_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(dest="command")
    add_iwfm2obs_parser(subs)
    return parser


# ---------------------------------------------------------------------------
# run_calctyphyd
# ---------------------------------------------------------------------------


class TestRunCalctyphyd:
    """Tests for run_calctyphyd()."""

    def test_missing_water_levels_returns_1(self, tmp_path: Path) -> None:
        weights = tmp_path / "weights.txt"
        weights.touch()
        args = argparse.Namespace(
            water_levels=str(tmp_path / "nonexistent.smp"),
            weights=str(weights),
            output=str(tmp_path / "out.smp"),
        )
        result = run_calctyphyd(args)
        assert result == 1

    def test_missing_weights_returns_1(self, tmp_path: Path) -> None:
        wl = tmp_path / "wl.smp"
        wl.touch()
        args = argparse.Namespace(
            water_levels=str(wl),
            weights=str(tmp_path / "nonexistent.txt"),
            output=str(tmp_path / "out.smp"),
        )
        result = run_calctyphyd(args)
        assert result == 1

    @patch("pyiwfm.io.smp.SMPWriter")
    @patch("pyiwfm.calibration.calctyphyd.compute_typical_hydrographs")
    @patch("pyiwfm.calibration.calctyphyd.read_cluster_weights")
    @patch("pyiwfm.io.smp.SMPReader")
    def test_success_returns_0(
        self,
        mock_reader_cls: MagicMock,
        mock_read_weights: MagicMock,
        mock_compute: MagicMock,
        mock_writer_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        import numpy as np

        wl = tmp_path / "wl.smp"
        wl.touch()
        weights = tmp_path / "weights.txt"
        weights.touch()

        mock_reader_cls.return_value.read.return_value = {"WELL1": MagicMock()}
        mock_read_weights.return_value = MagicMock()

        # Build a mock typical hydrograph result
        mock_th = MagicMock()
        mock_th.cluster_id = 1
        mock_th.times = np.array(["2020-01-01", "2020-02-01"], dtype="datetime64[s]")
        mock_th.values = np.array([10.0, 12.0])
        mock_result = MagicMock()
        mock_result.hydrographs = [mock_th]
        mock_compute.return_value = mock_result

        args = argparse.Namespace(
            water_levels=str(wl),
            weights=str(weights),
            output=str(tmp_path / "out.smp"),
        )
        result = run_calctyphyd(args)
        assert result == 0
        mock_writer_cls.return_value.write.assert_called_once()


# ---------------------------------------------------------------------------
# run_iwfm2obs
# ---------------------------------------------------------------------------


class TestRunIwfm2obs:
    """Tests for run_iwfm2obs()."""

    def test_missing_obs_file_returns_1(self, tmp_path: Path) -> None:
        sim = tmp_path / "sim.smp"
        sim.touch()
        args = argparse.Namespace(
            obs=str(tmp_path / "nonexistent.smp"),
            sim=str(sim),
            output=str(tmp_path / "out.smp"),
            threshold=30,
        )
        result = run_iwfm2obs(args)
        assert result == 1

    def test_missing_sim_file_returns_1(self, tmp_path: Path) -> None:
        obs = tmp_path / "obs.smp"
        obs.touch()
        args = argparse.Namespace(
            obs=str(obs),
            sim=str(tmp_path / "nonexistent.smp"),
            output=str(tmp_path / "out.smp"),
            threshold=30,
        )
        result = run_iwfm2obs(args)
        assert result == 1

    @patch("pyiwfm.calibration.iwfm2obs.iwfm2obs")
    def test_success_returns_0(self, mock_iwfm2obs: MagicMock, tmp_path: Path) -> None:
        obs = tmp_path / "obs.smp"
        obs.touch()
        sim = tmp_path / "sim.smp"
        sim.touch()
        mock_iwfm2obs.return_value = {"WELL1": MagicMock()}

        args = argparse.Namespace(
            obs=str(obs),
            sim=str(sim),
            output=str(tmp_path / "out.smp"),
            threshold=30,
        )
        result = run_iwfm2obs(args)
        assert result == 0
        mock_iwfm2obs.assert_called_once()
