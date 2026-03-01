"""Unit tests for calibration residual analysis engine."""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyiwfm.calibration.residuals import (
    WellScreenType,
    compute_residuals,
    export_residual_table,
    filter_residuals,
    max_residuals,
    mean_residuals,
    residual_summary,
)
from pyiwfm.io.smp import SMPTimeSeries


def _make_smp(bore_id: str, values: list[float], start: str = "2020-01-01") -> SMPTimeSeries:
    """Helper: create a simple SMPTimeSeries."""
    n = len(values)
    times = np.array(
        [np.datetime64(start) + np.timedelta64(i * 30, "D") for i in range(n)],
        dtype="datetime64[D]",
    )
    return SMPTimeSeries(
        bore_id=bore_id,
        times=times,
        values=np.array(values, dtype=np.float64),
        excluded=np.zeros(n, dtype=np.bool_),
    )


class TestComputeResiduals:
    """Tests for compute_residuals."""

    def test_basic_join(self) -> None:
        obs = {"W1": _make_smp("W1", [100.0, 110.0, 105.0])}
        sim = {"W1": _make_smp("W1", [102.0, 108.0, 106.0])}
        df = compute_residuals(obs, sim)
        assert len(df) == 3
        assert "residual" in df.columns
        assert df["residual"].iloc[0] == pytest.approx(2.0)

    def test_multiple_wells(self) -> None:
        obs = {
            "W1": _make_smp("W1", [100.0, 110.0]),
            "W2": _make_smp("W2", [200.0, 210.0]),
        }
        sim = {
            "W1": _make_smp("W1", [101.0, 109.0]),
            "W2": _make_smp("W2", [202.0, 208.0]),
        }
        df = compute_residuals(obs, sim)
        assert len(df) == 4
        assert set(df["well_id"].unique()) == {"W1", "W2"}

    def test_unmatched_wells_ignored(self) -> None:
        obs = {"W1": _make_smp("W1", [100.0])}
        sim = {"W2": _make_smp("W2", [200.0])}
        df = compute_residuals(obs, sim)
        assert len(df) == 0

    def test_with_well_info(self) -> None:
        obs = {"W1": _make_smp("W1", [100.0])}
        sim = {"W1": _make_smp("W1", [102.0])}
        info = {"W1": {"layer": 1, "subregion": 3, "screen_type": WellScreenType.KNOWN_SCREENS}}
        df = compute_residuals(obs, sim, well_info=info)
        assert df["layer"].iloc[0] == 1
        assert df["subregion"].iloc[0] == 3
        assert df["screen_type"].iloc[0] == "known_screens"

    def test_nan_values_excluded(self) -> None:
        obs = {"W1": _make_smp("W1", [100.0, float("nan"), 105.0])}
        sim = {"W1": _make_smp("W1", [102.0, 108.0, 106.0])}
        df = compute_residuals(obs, sim)
        assert len(df) == 2


class TestMeanResiduals:
    """Tests for mean_residuals."""

    def test_basic(self) -> None:
        df = pd.DataFrame(
            {
                "well_id": ["W1", "W1", "W2"],
                "residual": [2.0, -2.0, 3.0],
            }
        )
        result = mean_residuals(df)
        assert len(result) == 2
        w1_row = result[result["well_id"] == "W1"]
        assert w1_row["mean_residual"].iloc[0] == pytest.approx(0.0)


class TestMaxResiduals:
    """Tests for max_residuals."""

    def test_basic(self) -> None:
        df = pd.DataFrame(
            {
                "well_id": ["W1", "W1", "W2"],
                "residual": [2.0, -5.0, 3.0],
            }
        )
        result = max_residuals(df)
        w1_row = result[result["well_id"] == "W1"]
        assert w1_row["max_abs_residual"].iloc[0] == pytest.approx(5.0)


class TestFilterResiduals:
    """Tests for filter_residuals."""

    @pytest.fixture()
    def sample_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "well_id": ["W1", "W1", "W2", "W2"],
                "datetime": pd.to_datetime(
                    ["2020-01-15", "2020-06-15", "2020-01-15", "2020-06-15"]
                ),
                "observed": [100.0, 110.0, 200.0, 210.0],
                "simulated": [102.0, 108.0, 202.0, 208.0],
                "residual": [2.0, -2.0, 2.0, -2.0],
                "layer": [1, 1, 2, 2],
                "subregion": [1, 1, 2, 2],
                "screen_type": ["known_screens", "known_screens", "unknown", "unknown"],
            }
        )

    def test_filter_by_layer(self, sample_df: pd.DataFrame) -> None:
        result = filter_residuals(sample_df, layers=[1])
        assert len(result) == 2
        assert all(result["layer"] == 1)

    def test_filter_by_subregion(self, sample_df: pd.DataFrame) -> None:
        result = filter_residuals(sample_df, subregions=[2])
        assert len(result) == 2

    def test_filter_by_date_range(self, sample_df: pd.DataFrame) -> None:
        result = filter_residuals(
            sample_df,
            date_range=(datetime(2020, 3, 1), datetime(2020, 12, 31)),
        )
        assert len(result) == 2

    def test_filter_by_screen_type(self, sample_df: pd.DataFrame) -> None:
        result = filter_residuals(
            sample_df,
            screen_types=[WellScreenType.KNOWN_SCREENS],
        )
        assert len(result) == 2

    def test_filter_combined(self, sample_df: pd.DataFrame) -> None:
        result = filter_residuals(sample_df, layers=[1], subregions=[1])
        assert len(result) == 2

    def test_no_filter(self, sample_df: pd.DataFrame) -> None:
        result = filter_residuals(sample_df)
        assert len(result) == 4


class TestResidualSummary:
    """Tests for residual_summary."""

    def test_basic(self) -> None:
        df = pd.DataFrame(
            {
                "observed": [100.0, 200.0, 150.0],
                "simulated": [102.0, 198.0, 152.0],
                "residual": [2.0, -2.0, 2.0],
            }
        )
        summary = residual_summary(df)
        assert summary["n"] == 3.0
        assert "rmse" in summary
        assert "nash_sutcliffe" in summary
        assert "index_of_agreement" in summary

    def test_empty_df(self) -> None:
        df = pd.DataFrame(columns=["observed", "simulated", "residual"])
        summary = residual_summary(df)
        assert summary["n"] == 0


class TestExportResidualTable:
    """Tests for export_residual_table."""

    def test_writes_csv(self) -> None:
        df = pd.DataFrame(
            {
                "well_id": ["W1"],
                "datetime": [pd.Timestamp("2020-01-15")],
                "observed": [100.0],
                "simulated": [102.0],
                "residual": [2.0],
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out = export_residual_table(df, Path(tmpdir) / "res.csv")
            assert out.exists()
            content = out.read_text()
            assert "well_id" in content
            assert "W1" in content


class TestWellScreenType:
    """Tests for WellScreenType enum."""

    def test_values(self) -> None:
        assert WellScreenType.KNOWN_SCREENS.value == "known_screens"
        assert WellScreenType.INTERPOLATED_TOS.value == "interpolated_tos"
        assert WellScreenType.INTERPOLATED_TOS_BOS.value == "interpolated_tos_bos"
        assert WellScreenType.UNKNOWN.value == "unknown"
