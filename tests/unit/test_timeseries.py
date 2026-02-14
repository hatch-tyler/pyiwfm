"""Unit tests for core timeseries module."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from pyiwfm.core.timeseries import (
    TimeUnit,
    TimeStep,
    SimulationPeriod,
    TimeSeries,
    TimeSeriesCollection,
)


class TestTimeUnit:
    def test_time_unit_values(self) -> None:
        assert TimeUnit.MINUTE.value == "MIN"
        assert TimeUnit.DAY.value == "DAY"

    def test_from_string_day(self) -> None:
        assert TimeUnit.from_string("DAY") == TimeUnit.DAY
        assert TimeUnit.from_string("day") == TimeUnit.DAY

    def test_from_string_hour(self) -> None:
        assert TimeUnit.from_string("HOUR") == TimeUnit.HOUR
        assert TimeUnit.from_string("HR") == TimeUnit.HOUR

    def test_from_string_unknown_raises(self) -> None:
        with pytest.raises(ValueError):
            TimeUnit.from_string("INVALID")

    def test_to_timedelta_day(self) -> None:
        assert TimeUnit.DAY.to_timedelta(3) == timedelta(days=3)

    def test_to_timedelta_hour(self) -> None:
        assert TimeUnit.HOUR.to_timedelta(2) == timedelta(hours=2)


class TestTimeStep:
    def test_basic_creation(self) -> None:
        start = datetime(2020, 1, 1)
        end = datetime(2020, 1, 2)
        ts = TimeStep(start=start, end=end)
        assert ts.start == start
        assert ts.end == end
        assert ts.index == 0

    def test_duration(self) -> None:
        ts = TimeStep(start=datetime(2020, 1, 1), end=datetime(2020, 1, 3))
        assert ts.duration == timedelta(days=2)

    def test_midpoint(self) -> None:
        ts = TimeStep(start=datetime(2020, 1, 1), end=datetime(2020, 1, 3))
        assert ts.midpoint == datetime(2020, 1, 2)


class TestSimulationPeriod:
    def test_basic_creation(self) -> None:
        period = SimulationPeriod(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 12, 31),
            time_step_length=1,
            time_step_unit=TimeUnit.DAY,
        )
        assert period.time_step_length == 1

    def test_n_time_steps(self) -> None:
        period = SimulationPeriod(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 11),
            time_step_length=1,
            time_step_unit=TimeUnit.DAY,
        )
        assert period.n_time_steps == 10

    def test_iter_time_steps(self) -> None:
        period = SimulationPeriod(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 4),
            time_step_length=1,
            time_step_unit=TimeUnit.DAY,
        )
        steps = list(period.iter_time_steps())
        assert len(steps) == 3

    def test_get_time_step(self) -> None:
        period = SimulationPeriod(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 10),
            time_step_length=1,
            time_step_unit=TimeUnit.DAY,
        )
        ts = period.get_time_step(5)
        assert ts.start == datetime(2020, 1, 6)


class TestTimeSeries:
    def test_basic_creation(self) -> None:
        times = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[s]")
        values = np.array([1.0, 2.0])
        ts = TimeSeries(times=times, values=values)
        assert ts.n_times == 2

    def test_validation_length_mismatch(self) -> None:
        times = np.array(["2020-01-01"], dtype="datetime64[s]")
        values = np.array([1.0, 2.0])
        with pytest.raises(ValueError):
            TimeSeries(times=times, values=values)

    def test_from_datetimes(self) -> None:
        dts = [datetime(2020, 1, 1), datetime(2020, 1, 2)]
        values = np.array([1.0, 2.0])
        ts = TimeSeries.from_datetimes(dts, values, name="Test")
        assert ts.n_times == 2
        assert ts.name == "Test"

    def test_getitem(self) -> None:
        times = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[s]")
        values = np.array([10.0, 20.0])
        ts = TimeSeries(times=times, values=values)
        assert ts[0] == 10.0
        assert ts[1] == 20.0

    def test_len(self) -> None:
        times = np.array(["2020-01-01", "2020-01-02", "2020-01-03"], dtype="datetime64[s]")
        values = np.array([1.0, 2.0, 3.0])
        ts = TimeSeries(times=times, values=values)
        assert len(ts) == 3


class TestTimeSeriesCollection:
    def test_basic_creation(self) -> None:
        collection = TimeSeriesCollection()
        assert len(collection) == 0

    def test_add_and_get(self) -> None:
        collection = TimeSeriesCollection()
        times = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[s]")
        ts = TimeSeries(times=times, values=np.array([1.0, 2.0]), location="Node1")
        collection.add(ts)
        assert len(collection) == 1
        assert collection.get("Node1") is ts

    def test_locations(self) -> None:
        collection = TimeSeriesCollection()
        times = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[s]")
        ts1 = TimeSeries(times=times, values=np.array([1.0, 2.0]), location="A")
        ts2 = TimeSeries(times=times, values=np.array([3.0, 4.0]), location="B")
        collection.add(ts1)
        collection.add(ts2)
        assert "A" in collection.locations
        assert "B" in collection.locations

    def test_iter(self) -> None:
        collection = TimeSeriesCollection()
        times = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[s]")
        ts = TimeSeries(times=times, values=np.array([1.0, 2.0]), location="Node1")
        collection.add(ts)
        series_list = list(collection)
        assert len(series_list) == 1


# =============================================================================
# Additional coverage tests
# =============================================================================


class TestTimeUnitAdditional:
    """Additional TimeUnit tests for full coverage."""

    def test_from_string_minute_variants(self) -> None:
        assert TimeUnit.from_string("MIN") == TimeUnit.MINUTE
        assert TimeUnit.from_string("MINUTE") == TimeUnit.MINUTE
        assert TimeUnit.from_string("MINUTES") == TimeUnit.MINUTE
        assert TimeUnit.from_string("minutes") == TimeUnit.MINUTE

    def test_from_string_hour_variants(self) -> None:
        assert TimeUnit.from_string("HOURS") == TimeUnit.HOUR

    def test_from_string_day_variants(self) -> None:
        assert TimeUnit.from_string("DAYS") == TimeUnit.DAY

    def test_from_string_week_variants(self) -> None:
        assert TimeUnit.from_string("WEEK") == TimeUnit.WEEK
        assert TimeUnit.from_string("WEEKS") == TimeUnit.WEEK

    def test_from_string_month_variants(self) -> None:
        assert TimeUnit.from_string("MON") == TimeUnit.MONTH
        assert TimeUnit.from_string("MONTH") == TimeUnit.MONTH
        assert TimeUnit.from_string("MONTHS") == TimeUnit.MONTH

    def test_from_string_year_variants(self) -> None:
        assert TimeUnit.from_string("YEAR") == TimeUnit.YEAR
        assert TimeUnit.from_string("YR") == TimeUnit.YEAR
        assert TimeUnit.from_string("YEARS") == TimeUnit.YEAR

    def test_from_string_whitespace(self) -> None:
        assert TimeUnit.from_string("  DAY  ") == TimeUnit.DAY

    def test_to_timedelta_minute(self) -> None:
        assert TimeUnit.MINUTE.to_timedelta(5) == timedelta(minutes=5)

    def test_to_timedelta_week(self) -> None:
        assert TimeUnit.WEEK.to_timedelta(2) == timedelta(weeks=2)

    def test_to_timedelta_month_approx(self) -> None:
        result = TimeUnit.MONTH.to_timedelta(1)
        assert result == timedelta(days=30)

    def test_to_timedelta_year_approx(self) -> None:
        result = TimeUnit.YEAR.to_timedelta(1)
        assert result == timedelta(days=365)

    def test_to_timedelta_default_n(self) -> None:
        result = TimeUnit.DAY.to_timedelta()
        assert result == timedelta(days=1)

    def test_all_enum_values(self) -> None:
        """Test all enum members exist."""
        assert TimeUnit.MINUTE.value == "MIN"
        assert TimeUnit.HOUR.value == "HOUR"
        assert TimeUnit.DAY.value == "DAY"
        assert TimeUnit.WEEK.value == "WEEK"
        assert TimeUnit.MONTH.value == "MON"
        assert TimeUnit.YEAR.value == "YEAR"


class TestTimeStepAdditional:
    """Additional TimeStep tests for coverage."""

    def test_custom_index(self) -> None:
        ts = TimeStep(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 2),
            index=42,
        )
        assert ts.index == 42

    def test_repr(self) -> None:
        ts = TimeStep(start=datetime(2020, 1, 1), end=datetime(2020, 1, 2))
        result = repr(ts)
        assert "TimeStep" in result
        assert "2020-01-01" in result
        assert "2020-01-02" in result


class TestSimulationPeriodAdditional:
    """Additional SimulationPeriod tests for coverage."""

    def test_duration_property(self) -> None:
        period = SimulationPeriod(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 7, 1),
            time_step_length=1,
            time_step_unit=TimeUnit.DAY,
        )
        assert period.duration == datetime(2020, 7, 1) - datetime(2020, 1, 1)

    def test_time_step_delta(self) -> None:
        period = SimulationPeriod(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 12, 31),
            time_step_length=7,
            time_step_unit=TimeUnit.DAY,
        )
        assert period.time_step_delta == timedelta(days=7)

    def test_n_time_steps_zero_delta(self) -> None:
        """Test n_time_steps when start == end."""
        period = SimulationPeriod(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1),
            time_step_length=1,
            time_step_unit=TimeUnit.DAY,
        )
        assert period.n_time_steps == 0

    def test_iter_time_steps_clamps_end(self) -> None:
        """Test iter_time_steps when final step exceeds end."""
        period = SimulationPeriod(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 5, 12),  # Halfway through day 5
            time_step_length=2,
            time_step_unit=TimeUnit.DAY,
        )
        steps = list(period.iter_time_steps())
        # Last step should be clamped to end
        assert steps[-1].end == datetime(2020, 1, 5, 12)

    def test_get_time_step_clamps_end(self) -> None:
        """Test get_time_step clamps end to simulation end."""
        period = SimulationPeriod(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 3),
            time_step_length=2,
            time_step_unit=TimeUnit.DAY,
        )
        ts = period.get_time_step(1)
        assert ts.end == datetime(2020, 1, 3)

    def test_repr(self) -> None:
        period = SimulationPeriod(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 12, 31),
            time_step_length=1,
            time_step_unit=TimeUnit.MONTH,
        )
        result = repr(period)
        assert "SimulationPeriod" in result
        assert "MON" in result

    def test_iter_time_steps_indices(self) -> None:
        period = SimulationPeriod(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 4),
            time_step_length=1,
            time_step_unit=TimeUnit.DAY,
        )
        steps = list(period.iter_time_steps())
        assert steps[0].index == 0
        assert steps[1].index == 1
        assert steps[2].index == 2

    def test_monthly_period(self) -> None:
        """Test simulation with monthly time steps."""
        period = SimulationPeriod(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 12, 31),
            time_step_length=1,
            time_step_unit=TimeUnit.MONTH,
        )
        assert period.n_time_steps > 0
        assert period.time_step_delta == timedelta(days=30)


class TestTimeSeriesAdditional:
    """Additional TimeSeries tests for full coverage."""

    def test_start_time_property(self) -> None:
        times = np.array(["2020-01-01", "2020-06-01", "2020-12-31"], dtype="datetime64[s]")
        ts = TimeSeries(times=times, values=np.array([1.0, 2.0, 3.0]))
        assert ts.start_time == np.datetime64("2020-01-01", "s")

    def test_end_time_property(self) -> None:
        times = np.array(["2020-01-01", "2020-06-01", "2020-12-31"], dtype="datetime64[s]")
        ts = TimeSeries(times=times, values=np.array([1.0, 2.0, 3.0]))
        assert ts.end_time == np.datetime64("2020-12-31", "s")

    def test_to_dataframe_1d(self) -> None:
        times = np.array(["2020-01-01", "2020-01-02", "2020-01-03"], dtype="datetime64[s]")
        values = np.array([10.0, 20.0, 30.0])
        ts = TimeSeries(times=times, values=values, name="head")
        df = ts.to_dataframe()

        assert len(df) == 3
        assert "head" in df.columns
        assert df["head"].iloc[0] == 10.0

    def test_to_dataframe_2d(self) -> None:
        times = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[s]")
        values = np.array([[1.0, 2.0], [3.0, 4.0]])
        ts = TimeSeries(times=times, values=values, name="data")
        df = ts.to_dataframe()

        assert len(df) == 2
        assert "data_0" in df.columns
        assert "data_1" in df.columns

    def test_to_dataframe_unnamed(self) -> None:
        times = np.array(["2020-01-01"], dtype="datetime64[s]")
        ts = TimeSeries(times=times, values=np.array([5.0]))
        df = ts.to_dataframe()
        assert "value" in df.columns

    def test_resample(self) -> None:
        # Create daily data for 30 days
        dts = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(30)]
        values = np.arange(30, dtype=np.float64)
        ts = TimeSeries.from_datetimes(dts, values, name="test")

        # Resample to weekly
        resampled = ts.resample("W")
        assert resampled.n_times < ts.n_times
        assert resampled.name == "test"

    def test_slice_time_both(self) -> None:
        dts = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(30)]
        values = np.arange(30, dtype=np.float64)
        ts = TimeSeries.from_datetimes(dts, values)

        sliced = ts.slice_time(
            start=datetime(2020, 1, 10),
            end=datetime(2020, 1, 20),
        )
        assert sliced.n_times == 11  # Days 10-20 inclusive

    def test_slice_time_start_only(self) -> None:
        dts = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(10)]
        values = np.arange(10, dtype=np.float64)
        ts = TimeSeries.from_datetimes(dts, values)

        sliced = ts.slice_time(start=datetime(2020, 1, 5))
        assert sliced.n_times == 6  # Days 5-10

    def test_slice_time_end_only(self) -> None:
        dts = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(10)]
        values = np.arange(10, dtype=np.float64)
        ts = TimeSeries.from_datetimes(dts, values)

        sliced = ts.slice_time(end=datetime(2020, 1, 5))
        assert sliced.n_times == 5

    def test_slice_preserves_metadata(self) -> None:
        dts = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(10)]
        values = np.arange(10, dtype=np.float64)
        ts = TimeSeries.from_datetimes(
            dts, values, name="head", units="ft", location="Well1",
            metadata={"source": "obs"},
        )
        sliced = ts.slice_time(start=datetime(2020, 1, 3))
        assert sliced.name == "head"
        assert sliced.units == "ft"
        assert sliced.location == "Well1"
        assert sliced.metadata == {"source": "obs"}

    def test_getitem_slice(self) -> None:
        times = np.array(["2020-01-01", "2020-01-02", "2020-01-03"], dtype="datetime64[s]")
        values = np.array([10.0, 20.0, 30.0])
        ts = TimeSeries(times=times, values=values)
        result = ts[0:2]
        np.testing.assert_array_equal(result, [10.0, 20.0])

    def test_repr(self) -> None:
        times = np.array(["2020-01-01", "2020-12-31"], dtype="datetime64[s]")
        ts = TimeSeries(times=times, values=np.array([1.0, 2.0]), name="head")
        result = repr(ts)
        assert "TimeSeries" in result
        assert "head" in result
        assert "n_times=2" in result

    def test_metadata_default(self) -> None:
        times = np.array(["2020-01-01"], dtype="datetime64[s]")
        ts = TimeSeries(times=times, values=np.array([1.0]))
        assert ts.metadata == {}

    def test_from_datetimes_with_all_kwargs(self) -> None:
        dts = [datetime(2020, 1, 1)]
        ts = TimeSeries.from_datetimes(
            dts, np.array([42.0]),
            name="test", units="m", location="loc1",
            metadata={"key": "val"},
        )
        assert ts.name == "test"
        assert ts.units == "m"
        assert ts.location == "loc1"
        assert ts.metadata == {"key": "val"}


class TestTimeSeriesCollectionAdditional:
    """Additional TimeSeriesCollection tests."""

    def test_getitem(self) -> None:
        collection = TimeSeriesCollection()
        times = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[s]")
        ts = TimeSeries(times=times, values=np.array([1.0, 2.0]), location="Node1")
        collection.add(ts)

        result = collection["Node1"]
        assert result is ts

    def test_repr(self) -> None:
        collection = TimeSeriesCollection(name="Wells")
        result = repr(collection)
        assert "TimeSeriesCollection" in result
        assert "Wells" in result
        assert "n_series=0" in result

    def test_get_missing_returns_none(self) -> None:
        collection = TimeSeriesCollection()
        assert collection.get("nonexistent") is None

    def test_add_by_name_when_no_location(self) -> None:
        """Test add uses name when location is empty."""
        collection = TimeSeriesCollection()
        times = np.array(["2020-01-01"], dtype="datetime64[s]")
        ts = TimeSeries(times=times, values=np.array([1.0]), name="MyName")
        collection.add(ts)
        assert collection.get("MyName") is ts

    def test_variable_attribute(self) -> None:
        collection = TimeSeriesCollection(variable="head")
        assert collection.variable == "head"
