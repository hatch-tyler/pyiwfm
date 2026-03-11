"""Tests for pyiwfm.calibration.head_differences."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyiwfm.calibration.head_differences import (
    HeadDiffPair,
    compute_head_differences,
    read_pairs_file,
)
from pyiwfm.io.smp import SMPTimeSeries


def _make_ts(bore_id: str, dates: list[str], values: list[float]) -> SMPTimeSeries:
    """Helper to create a SMPTimeSeries."""
    times = np.array([np.datetime64(d) for d in dates], dtype="datetime64[s]")
    return SMPTimeSeries(
        bore_id=bore_id,
        times=times,
        values=np.array(values, dtype=np.float64),
        excluded=np.zeros(len(values), dtype=np.bool_),
    )


class TestComputeHeadDifferences:
    """Test compute_head_differences."""

    def test_basic_diff(self) -> None:
        pairs = [HeadDiffPair("W1%1", "W1%3", "HD_W1")]
        sim_data = {
            "W1%1": _make_ts("W1%1", ["2020-01-31", "2020-02-29"], [100.0, 101.0]),
            "W1%3": _make_ts("W1%3", ["2020-01-31", "2020-02-29"], [90.0, 92.0]),
        }
        result = compute_head_differences(pairs, sim_data)
        assert "HD_W1" in result
        assert len(result["HD_W1"].values) == 2
        assert result["HD_W1"].values[0] == pytest.approx(10.0)
        assert result["HD_W1"].values[1] == pytest.approx(9.0)

    def test_missing_well(self) -> None:
        """Pair skipped if either well is missing."""
        pairs = [HeadDiffPair("W1%1", "W1%3", "HD_W1")]
        sim_data = {
            "W1%1": _make_ts("W1%1", ["2020-01-31"], [100.0]),
        }
        result = compute_head_differences(pairs, sim_data)
        assert len(result) == 0

    def test_partial_date_overlap(self) -> None:
        """Only dates common to both wells are diffed."""
        pairs = [HeadDiffPair("W1%1", "W1%3", "HD_W1")]
        sim_data = {
            "W1%1": _make_ts(
                "W1%1",
                ["2020-01-31", "2020-02-29", "2020-03-31"],
                [100.0, 101.0, 102.0],
            ),
            "W1%3": _make_ts(
                "W1%3",
                ["2020-01-31", "2020-03-31"],
                [90.0, 93.0],
            ),
        }
        result = compute_head_differences(pairs, sim_data)
        assert len(result["HD_W1"].values) == 2

    def test_nan_values_excluded(self) -> None:
        pairs = [HeadDiffPair("W1%1", "W1%3", "HD_W1")]
        sim_data = {
            "W1%1": _make_ts("W1%1", ["2020-01-31", "2020-02-29"], [100.0, float("nan")]),
            "W1%3": _make_ts("W1%3", ["2020-01-31", "2020-02-29"], [90.0, 92.0]),
        }
        result = compute_head_differences(pairs, sim_data)
        assert len(result["HD_W1"].values) == 1

    def test_multiple_pairs(self) -> None:
        pairs = [
            HeadDiffPair("W1%1", "W1%3", "HD_W1"),
            HeadDiffPair("W2%1", "W2%2", "HD_W2"),
        ]
        sim_data = {
            "W1%1": _make_ts("W1%1", ["2020-01-31"], [100.0]),
            "W1%3": _make_ts("W1%3", ["2020-01-31"], [90.0]),
            "W2%1": _make_ts("W2%1", ["2020-01-31"], [80.0]),
            "W2%2": _make_ts("W2%2", ["2020-01-31"], [75.0]),
        }
        result = compute_head_differences(pairs, sim_data)
        assert len(result) == 2
        assert result["HD_W1"].values[0] == pytest.approx(10.0)
        assert result["HD_W2"].values[0] == pytest.approx(5.0)

    def test_bore_id_set_on_result(self) -> None:
        pairs = [HeadDiffPair("W1%1", "W1%3", "HD_W1")]
        sim_data = {
            "W1%1": _make_ts("W1%1", ["2020-01-31"], [100.0]),
            "W1%3": _make_ts("W1%3", ["2020-01-31"], [90.0]),
        }
        result = compute_head_differences(pairs, sim_data)
        assert result["HD_W1"].bore_id == "HD_W1"


class TestReadPairsFile:
    """Test read_pairs_file."""

    def test_basic(self, tmp_path: Path) -> None:
        f = tmp_path / "pairs.dat"
        f.write_text("W1%1  W1%3  HD_W1\nW2%1  W2%2  HD_W2\n")
        pairs = read_pairs_file(f)
        assert len(pairs) == 2
        assert pairs[0].shallow_well == "W1%1"
        assert pairs[0].deep_well == "W1%3"
        assert pairs[0].pair_name == "HD_W1"

    def test_auto_name(self, tmp_path: Path) -> None:
        """Auto-generate pair name if not provided."""
        f = tmp_path / "pairs.dat"
        f.write_text("W1%1  W1%3\n")
        pairs = read_pairs_file(f)
        assert len(pairs) == 1
        assert pairs[0].pair_name.startswith("HD")

    def test_skip_short_lines(self, tmp_path: Path) -> None:
        f = tmp_path / "pairs.dat"
        f.write_text("W1%1\nW2%1  W2%2  HD_W2\n")
        pairs = read_pairs_file(f)
        assert len(pairs) == 1

    def test_empty_file(self, tmp_path: Path) -> None:
        f = tmp_path / "pairs.dat"
        f.write_text("")
        pairs = read_pairs_file(f)
        assert len(pairs) == 0
