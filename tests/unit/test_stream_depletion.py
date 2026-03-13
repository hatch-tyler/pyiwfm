"""Tests for the stream depletion analysis module."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np


def _make_mock_budget_reader(flow_values: list[list[float]], n_locations: int = 3) -> MagicMock:
    """Create a mock BudgetReader for stream depletion tests."""
    reader = MagicMock()
    reader.n_locations = n_locations
    reader.locations = [f"Reach {i + 1}" for i in range(n_locations)]

    headers = [
        "Upstream Inflow (+)",
        "Gain from GW (+)",
        "Tributary Inflow (+)",
        "Downstream Outflow (-)",
        "Diversion (-)",
    ]
    reader.get_column_headers.return_value = headers

    times = np.arange(len(flow_values), dtype=np.float64)
    values = np.array(flow_values, dtype=np.float64)
    reader.get_values.return_value = (times, values)
    reader.header.timestep.start_datetime = None

    return reader


class TestStreamDepletion:
    """Tests for stream depletion computation."""

    def test_depletion_positive_when_pumping_reduces_flow(self) -> None:
        """Depletion should be positive when scenario has less GW gain."""
        from pyiwfm.io.stream_depletion import _extract_stream_flow

        headers = ["Upstream Inflow (+)", "Gain from GW (+)", "Outflow (-)"]
        # Baseline has more GW gain
        base_vals = np.array([[100, 50, -120], [100, 55, -125]], dtype=np.float64)
        scen_vals = np.array([[100, 30, -100], [100, 35, -105]], dtype=np.float64)

        base_flow = _extract_stream_flow(headers, base_vals)
        scen_flow = _extract_stream_flow(headers, scen_vals)

        depletion = base_flow - scen_flow
        assert all(d > 0 for d in depletion)

    def test_result_to_dict(self) -> None:
        """StreamDepletionResult.to_dict should be serializable."""
        from pyiwfm.io.stream_depletion import StreamDepletionResult

        result = StreamDepletionResult(
            reach_id=1,
            reach_name="Test Reach",
            times=["2020-01-01", "2020-02-01"],
            baseline_flow=np.array([100.0, 110.0]),
            scenario_flow=np.array([80.0, 85.0]),
            depletion=np.array([20.0, 25.0]),
            cumulative_depletion=np.array([20.0, 45.0]),
            max_depletion=25.0,
            max_depletion_timestep=1,
            total_depletion=45.0,
        )

        d = result.to_dict()
        assert d["reach_id"] == 1
        assert len(d["depletion"]) == 2  # type: ignore[arg-type]
        assert d["total_depletion"] == 45.0

    def test_report_to_dict(self) -> None:
        """StreamDepletionReport.to_dict should aggregate correctly."""
        from pyiwfm.io.stream_depletion import StreamDepletionReport, StreamDepletionResult

        r1 = StreamDepletionResult(
            reach_id=1,
            reach_name="R1",
            times=["t1", "t2"],
            baseline_flow=np.array([100.0, 100.0]),
            scenario_flow=np.array([90.0, 85.0]),
            depletion=np.array([10.0, 15.0]),
            cumulative_depletion=np.array([10.0, 25.0]),
            max_depletion=15.0,
            max_depletion_timestep=1,
            total_depletion=25.0,
        )

        report = StreamDepletionReport(
            results=[r1],
            n_reaches=1,
            n_timesteps=2,
            total_max_depletion=15.0,
            total_cumulative_depletion=25.0,
        )

        d = report.to_dict()
        assert d["n_reaches"] == 1
        assert len(d["reaches"]) == 1  # type: ignore[arg-type]

    def test_extract_stream_flow_fallback_to_inflow_outflow(self) -> None:
        """When no GW column exists, should sum inflows and outflows."""
        from pyiwfm.io.stream_depletion import _extract_stream_flow

        headers = ["Upstream Inflow (+)", "Downstream Outflow (-)"]
        values = np.array([[100, -60], [120, -80]], dtype=np.float64)

        flow = _extract_stream_flow(headers, values)
        np.testing.assert_array_almost_equal(flow, [40.0, 40.0])

    def test_extract_stream_flow_gw_column(self) -> None:
        """Should prefer the GW interaction column when available."""
        from pyiwfm.io.stream_depletion import _extract_stream_flow

        headers = ["Upstream Inflow (+)", "Gain from GW (+)", "Outflow (-)"]
        values = np.array([[100, 50, -120], [100, 30, -100]], dtype=np.float64)

        flow = _extract_stream_flow(headers, values)
        np.testing.assert_array_almost_equal(flow, [50.0, 30.0])

    def test_find_stream_budget_raises_when_not_found(self) -> None:
        """Should raise FileNotFoundError for empty directory."""
        import tempfile

        import pytest

        from pyiwfm.io.stream_depletion import _find_stream_budget

        with tempfile.TemporaryDirectory() as tmpdir:
            from pathlib import Path

            with pytest.raises(FileNotFoundError):
                _find_stream_budget(Path(tmpdir), "Stream Reach Budgets")

    def test_cumulative_depletion_is_cumsum(self) -> None:
        """Cumulative depletion should be the running sum of per-timestep depletion."""
        from pyiwfm.io.stream_depletion import StreamDepletionResult

        depletion = np.array([10.0, 20.0, 5.0])
        cumulative = np.cumsum(depletion)

        result = StreamDepletionResult(
            reach_id=1,
            reach_name="R1",
            times=["t1", "t2", "t3"],
            baseline_flow=np.array([100.0, 100.0, 100.0]),
            scenario_flow=np.array([90.0, 80.0, 95.0]),
            depletion=depletion,
            cumulative_depletion=cumulative,
            max_depletion=20.0,
            max_depletion_timestep=1,
            total_depletion=35.0,
        )

        np.testing.assert_array_almost_equal(result.cumulative_depletion, [10.0, 30.0, 35.0])
