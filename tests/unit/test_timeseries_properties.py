"""Property-based tests for TimeSeries operations using Hypothesis."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pyiwfm.core.timeseries import TimeSeries

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


@st.composite
def timeseries_strategy(draw: st.DrawFn) -> TimeSeries:
    """Generate a random TimeSeries with sorted dates and matching values."""
    n = draw(st.integers(min_value=2, max_value=100))
    start = datetime(2000, 1, 1)

    # Generate sorted time offsets (in days)
    offsets = sorted(
        draw(
            st.lists(
                st.integers(min_value=0, max_value=36500),
                min_size=n,
                max_size=n,
                unique=True,
            )
        )
    )
    times = [start + timedelta(days=d) for d in offsets]
    np_times = np.array(times, dtype="datetime64[s]")

    values = np.array(
        [
            draw(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False))
            for _ in range(n)
        ]
    )

    name = draw(
        st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("L",)))
    )

    return TimeSeries(times=np_times, values=values, name=name)


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


@pytest.mark.property
class TestTimeSeriesProperties:
    """Property-based tests for TimeSeries invariants."""

    @given(timeseries_strategy())
    @settings(max_examples=30)
    def test_length_consistency(self, ts: TimeSeries) -> None:
        """n_times matches array lengths."""
        assert ts.n_times == len(ts.times)
        assert ts.n_times == ts.values.shape[0]

    @given(timeseries_strategy())
    @settings(max_examples=30)
    def test_start_before_end(self, ts: TimeSeries) -> None:
        """Start time is before or equal to end time."""
        assert ts.start_time <= ts.end_time

    @given(timeseries_strategy())
    @settings(max_examples=30)
    def test_times_are_sorted(self, ts: TimeSeries) -> None:
        """Times should be in ascending order."""
        diffs = np.diff(ts.times.astype(np.int64))
        assert np.all(diffs >= 0)

    @given(timeseries_strategy())
    @settings(max_examples=20)
    def test_to_dataframe_preserves_length(self, ts: TimeSeries) -> None:
        """DataFrame conversion preserves number of rows."""
        df = ts.to_dataframe()
        assert len(df) == ts.n_times

    @given(timeseries_strategy())
    @settings(max_examples=20)
    def test_to_dataframe_preserves_values(self, ts: TimeSeries) -> None:
        """DataFrame conversion preserves values."""
        df = ts.to_dataframe()
        col = df.columns[0]
        np.testing.assert_allclose(df[col].values, ts.values)

    @given(
        st.integers(min_value=2, max_value=50),
    )
    def test_mismatched_lengths_raise(self, n: int) -> None:
        """Creating a TimeSeries with mismatched lengths raises ValueError."""
        times = np.arange(
            np.datetime64("2000-01-01"),
            np.datetime64("2000-01-01") + np.timedelta64(n, "D"),
            dtype="datetime64[D]",
        )
        values = np.zeros(n + 1)
        with pytest.raises(ValueError, match="doesn't match"):
            TimeSeries(times=times, values=values)
