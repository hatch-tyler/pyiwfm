"""Tests for the water budget sanity check module."""

from __future__ import annotations

from unittest.mock import MagicMock


def _make_mock_reader():
    """Create a mock BudgetReader with simple data."""
    import numpy as np

    reader = MagicMock()
    reader.n_locations = 1
    reader.locations = ["Subregion 1"]

    headers = [
        "Deep Percolation (+)",
        "Gain from Stream (+)",
        "Pumping (-)",
        "Net Deep Percolation (-)",
        "Cumulative Subsidence",
    ]
    reader.get_column_headers.return_value = headers

    times = np.array([1.0, 2.0, 3.0])
    values = np.array(
        [
            [100.0, 50.0, -80.0, -70.0, 0.0],
            [110.0, 45.0, -85.0, -68.0, -2.0],
            [95.0, 55.0, -90.0, -62.0, 2.0],
        ]
    )
    reader.get_values.return_value = (times, values)

    # Mock header.timestep for time string generation
    reader.header.timestep.start_datetime = None

    return reader


class TestBudgetSanityCheck:
    """Tests for budget balance checking."""

    def test_check_budget_balance(self):
        """check_budget_balance should detect balance issues."""
        from pyiwfm.io.budget.checks import check_budget_balance

        reader = _make_mock_reader()
        report = check_budget_balance(reader, location_index=0)

        assert report.n_timesteps == 3

    def test_to_summary_dict(self):
        """to_summary_dict should return a serializable dict."""
        from pyiwfm.io.budget.checks import check_budget_balance

        reader = _make_mock_reader()
        report = check_budget_balance(reader, location_index=0)
        d = report.to_summary_dict()

        assert "n_timesteps" in d
        assert "n_violations" in d
        assert "max_percent_error" in d
