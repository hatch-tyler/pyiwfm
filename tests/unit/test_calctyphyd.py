"""Tests for CalcTypHyd typical hydrograph computation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from pyiwfm.calibration.calctyphyd import (
    CalcTypHydConfig,
    CalcTypHydFileConfig,
    SeasonalPeriod,
    compute_seasonal_averages,
    compute_typical_hydrographs,
    compute_typical_hydrographs_timeseries,
    read_calctyphyd_config,
    read_cluster_weights,
    write_pest_output,
)
from pyiwfm.io.smp import SMPTimeSeries


def _make_annual_ts(bore_id: str, base_value: float = 100.0) -> SMPTimeSeries:
    """Create a year of monthly water level data."""
    dates = [f"2020-{m:02d}-15" for m in range(1, 13)]
    # Simple seasonal pattern
    values = [base_value + 5.0 * np.sin(2 * np.pi * m / 12) for m in range(12)]
    return SMPTimeSeries(
        bore_id=bore_id,
        times=np.array(dates, dtype="datetime64[s]"),
        values=np.array(values, dtype=np.float64),
        excluded=np.zeros(12, dtype=np.bool_),
    )


class TestReadClusterWeights:
    """Tests for cluster weights file reading."""

    def test_read_weights(self, tmp_path: Path) -> None:
        weights_file = tmp_path / "weights.txt"
        weights_file.write_text(
            "# Header comment\nW1 0.8 0.1 0.1\nW2 0.2 0.7 0.1\nW3 0.1 0.2 0.7\n"
        )

        weights = read_cluster_weights(weights_file)
        assert len(weights) == 3
        assert weights["W1"][0] == pytest.approx(0.8)
        assert weights["W2"][1] == pytest.approx(0.7)
        np.testing.assert_allclose(weights["W3"], [0.1, 0.2, 0.7])

    def test_skip_comments_and_blanks(self, tmp_path: Path) -> None:
        weights_file = tmp_path / "weights.txt"
        weights_file.write_text("# Comment\n\nW1 0.5 0.5\n# Another comment\n")

        weights = read_cluster_weights(weights_file)
        assert len(weights) == 1
        assert "W1" in weights


class TestSeasonalAverages:
    """Tests for seasonal average computation."""

    def test_default_4_seasons(self) -> None:
        water_levels = {"W1": _make_annual_ts("W1")}
        avgs = compute_seasonal_averages(water_levels)

        assert "W1" in avgs
        assert len(avgs["W1"]) == 4
        # All seasons should have valid averages
        assert not np.any(np.isnan(avgs["W1"]))

    def test_custom_seasons(self) -> None:
        config = CalcTypHydConfig(
            seasonal_periods=[
                SeasonalPeriod("Wet", [11, 12, 1, 2, 3, 4], "01/15"),
                SeasonalPeriod("Dry", [5, 6, 7, 8, 9, 10], "07/15"),
            ]
        )
        water_levels = {"W1": _make_annual_ts("W1")}
        avgs = compute_seasonal_averages(water_levels, config)

        assert len(avgs["W1"]) == 2

    def test_insufficient_data(self) -> None:
        """Well with too few records gets NaN for some seasons."""
        ts = SMPTimeSeries(
            bore_id="W1",
            times=np.array(["2020-01-15"], dtype="datetime64[s]"),
            values=np.array([100.0]),
            excluded=np.zeros(1, dtype=np.bool_),
        )
        config = CalcTypHydConfig(min_records_per_season=2)
        avgs = compute_seasonal_averages({"W1": ts}, config)

        # Only 1 record in winter, min is 2 → all NaN
        assert np.all(np.isnan(avgs["W1"]))


class TestComputeTypicalHydrographs:
    """Tests for typical hydrograph computation."""

    def test_basic_computation(self, tmp_path: Path) -> None:
        """Compute typical hydrographs with simple weights."""
        water_levels = {
            "W1": _make_annual_ts("W1", 100.0),
            "W2": _make_annual_ts("W2", 200.0),
        }
        weights = {
            "W1": np.array([0.8, 0.2]),
            "W2": np.array([0.2, 0.8]),
        }

        result = compute_typical_hydrographs(water_levels, weights)

        assert len(result.hydrographs) == 2
        assert result.hydrographs[0].cluster_id == 0
        assert result.hydrographs[1].cluster_id == 1
        assert len(result.well_means) == 2

    def test_contributing_wells(self) -> None:
        """Wells with non-zero weights are listed as contributing."""
        water_levels = {
            "W1": _make_annual_ts("W1"),
            "W2": _make_annual_ts("W2"),
        }
        weights = {
            "W1": np.array([1.0, 0.0]),
            "W2": np.array([0.0, 1.0]),
        }

        result = compute_typical_hydrographs(water_levels, weights)

        assert "W1" in result.hydrographs[0].contributing_wells
        assert "W2" not in result.hydrographs[0].contributing_wells
        assert "W2" in result.hydrographs[1].contributing_wells

    def test_demeaned_values(self) -> None:
        """Typical hydrograph values should be de-meaned (centered around 0)."""
        water_levels = {"W1": _make_annual_ts("W1", 100.0)}
        weights = {"W1": np.array([1.0])}

        result = compute_typical_hydrographs(water_levels, weights)

        values = result.hydrographs[0].values
        valid = ~np.isnan(values)
        if np.any(valid):
            # De-meaned values should be centered near 0
            assert abs(float(np.nanmean(values))) < 5.0

    def test_empty_weights(self) -> None:
        """Empty weights produce empty result."""
        water_levels = {"W1": _make_annual_ts("W1")}
        result = compute_typical_hydrographs(water_levels, {})
        assert len(result.hydrographs) == 0


def _make_multiyear_ts(
    bore_id: str,
    start_year: int = 2010,
    end_year: int = 2015,
    base_value: float = 100.0,
) -> SMPTimeSeries:
    """Create multi-year monthly water level data."""
    dates: list[str] = []
    values: list[float] = []
    for yr in range(start_year, end_year + 1):
        for m in range(1, 13):
            dates.append(f"{yr}-{m:02d}-15")
            values.append(base_value + 5.0 * np.sin(2 * np.pi * m / 12))
    return SMPTimeSeries(
        bore_id=bore_id,
        times=np.array(dates, dtype="datetime64[s]"),
        values=np.array(values, dtype=np.float64),
        excluded=np.zeros(len(dates), dtype=np.bool_),
    )


class TestReadCalcTypHydConfig:
    """Tests for Fortran .in config file parsing."""

    def test_parse_config(self, tmp_path: Path) -> None:
        """Parse a synthetic .in file and verify all fields."""
        config_file = tmp_path / "calctyphyd.in"
        wl_file = tmp_path / "water_levels.smp"
        wt_file = tmp_path / "weights.txt"
        wl_file.touch()
        wt_file.touch()

        config_file.write_text(
            f"{wl_file}\n"
            f"{wt_file}\n"
            "3                          / n_clusters\n"
            "50                         / n_wells\n"
            "2                          / n_desired\n"
            "1\n"
            "3\n"
            "01/01/2010                 / start_date\n"
            "12/31/2015                 / end_date\n"
            "TH                         / pest_base_name\n"
            "2                          / n_periods\n"
            "6                          / n_months period 1\n"
            "10 11 12 1 2 3             / months\n"
            "1 15                       / rep_month rep_day\n"
            "6                          / n_months period 2\n"
            "4 5 6 7 8 9               / months\n"
            "7 15                       / rep_month rep_day\n"
        )

        cfg = read_calctyphyd_config(config_file)

        assert cfg.n_clusters == 3
        assert cfg.n_wells == 50
        assert cfg.desired_clusters == [1, 3]
        assert cfg.start_date == "01/01/2010"
        assert cfg.end_date == "12/31/2015"
        assert cfg.pest_base_name == "TH"
        assert len(cfg.periods) == 2
        assert cfg.periods[0].months == [10, 11, 12, 1, 2, 3]
        assert cfg.periods[1].months == [4, 5, 6, 7, 8, 9]
        assert cfg.periods[0].representative_date == "01/15"
        assert cfg.periods[1].representative_date == "07/15"

    def test_relative_paths(self, tmp_path: Path) -> None:
        """Paths in config are resolved relative to config file dir."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        config_file = subdir / "calctyphyd.in"

        config_file.write_text(
            "wl.smp\nwt.txt\n2\n10\n1\n1\n01/01/2010\n12/31/2015\nTH\n1\n3\n1 2 3\n2 15\n"
        )

        cfg = read_calctyphyd_config(config_file)
        assert cfg.water_level_path == subdir / "wl.smp"
        assert cfg.weights_path == subdir / "wt.txt"


class TestDateRangeFiltering:
    """Tests for date-range filtering in compute_seasonal_averages."""

    def test_filter_excludes_out_of_range(self) -> None:
        """Records outside date range are excluded."""
        ts = _make_multiyear_ts("W1", 2010, 2015)
        # Only keep 2012-2013
        config = CalcTypHydConfig(
            start_date=datetime(2012, 1, 1),
            end_date=datetime(2013, 12, 31),
        )
        avgs = compute_seasonal_averages({"W1": ts}, config)
        assert "W1" in avgs
        assert not np.any(np.isnan(avgs["W1"]))

    def test_no_filter(self) -> None:
        """Without date range, all data is used."""
        ts = _make_multiyear_ts("W1", 2010, 2015)
        config = CalcTypHydConfig()
        avgs = compute_seasonal_averages({"W1": ts}, config)
        assert not np.any(np.isnan(avgs["W1"]))

    def test_empty_after_filter(self) -> None:
        """Date range with no data → NaN averages."""
        ts = _make_multiyear_ts("W1", 2010, 2012)
        config = CalcTypHydConfig(
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2021, 12, 31),
        )
        avgs = compute_seasonal_averages({"W1": ts}, config)
        assert np.all(np.isnan(avgs["W1"]))


class TestTimeSeriesOutput:
    """Tests for compute_typical_hydrographs_timeseries."""

    def test_basic_output_dimensions(self) -> None:
        """Output has entries for each period-year with data."""
        water_levels = {
            "W1": _make_multiyear_ts("W1", 2010, 2012),
        }
        weights = {"W1": np.array([1.0, 0.5])}
        file_config = CalcTypHydFileConfig(
            n_clusters=2,
            desired_clusters=[1],
            start_date="01/01/2010",
            end_date="12/31/2012",
            pest_base_name="TH",
            periods=[
                SeasonalPeriod("Winter", [12, 1, 2], "01/15"),
                SeasonalPeriod("Summer", [6, 7, 8], "07/15"),
            ],
        )

        result = compute_typical_hydrographs_timeseries(water_levels, weights, file_config)

        assert len(result.hydrographs) == 1
        th = result.hydrographs[0]
        assert th.cluster_id == 1
        # 3 years × 2 periods = up to 6 entries
        assert len(th.dates) > 0
        assert len(th.values) == len(th.dates)
        assert len(th.pest_names) == len(th.dates)

    def test_pest_naming(self) -> None:
        """PEST names follow {base}{cluster}_{seq} format."""
        water_levels = {"W1": _make_multiyear_ts("W1", 2010, 2011)}
        weights = {"W1": np.array([1.0])}
        file_config = CalcTypHydFileConfig(
            n_clusters=1,
            desired_clusters=[1],
            start_date="01/01/2010",
            end_date="12/31/2011",
            pest_base_name="TH",
            periods=[
                SeasonalPeriod("Summer", [6, 7, 8], "07/15"),
            ],
        )

        result = compute_typical_hydrographs_timeseries(water_levels, weights, file_config)

        th = result.hydrographs[0]
        # Global seq = (year - start_year) * 12 + rep_month
        # 2010: (0)*12 + 7 = 7, 2011: (1)*12 + 7 = 19
        assert th.pest_names[0] == "TH1_7"
        if len(th.pest_names) > 1:
            assert th.pest_names[1] == "TH1_19"


class TestPestOutput:
    """Tests for PEST .out and .ins file writing."""

    def test_write_files(self, tmp_path: Path) -> None:
        """Write .out and .ins files and verify format."""
        from pyiwfm.calibration.calctyphyd import (
            CalcTypHydTimeSeriesResult,
            TypicalHydrographTimeSeries,
        )

        th = TypicalHydrographTimeSeries(
            cluster_id=1,
            dates=[datetime(2010, 1, 15), datetime(2010, 7, 15)],
            values=[1.234, -0.567],
            pest_names=["TH1_1", "TH1_2"],
        )
        result = CalcTypHydTimeSeriesResult(
            hydrographs=[th],
            well_means={"W1": 100.0},
        )
        file_config = CalcTypHydFileConfig(
            weights_path=Path("sub1sub2.txt"),
            pest_base_name="TH",
        )

        written = write_pest_output(result, file_config, tmp_path)

        assert len(written) == 2  # .out + .ins
        out_path = tmp_path / "sim_sub1sub2_cls1.out"
        ins_path = tmp_path / "sim_sub1sub2_cls1.ins"
        assert out_path.exists()
        assert ins_path.exists()

        # Check .out format
        out_lines = out_path.read_text().strip().split("\n")
        assert len(out_lines) == 3  # header + 2 data
        assert "TH1_1" in out_lines[1]
        assert "1.234" in out_lines[1]

        # Check .ins format
        ins_lines = ins_path.read_text().strip().split("\n")
        assert ins_lines[0] == "pif #"
        assert ins_lines[1] == "l1"  # skip header
        assert "[TH1_1]34:46" in ins_lines[2]
        assert "[TH1_2]34:46" in ins_lines[3]

    def test_column_positions(self, tmp_path: Path) -> None:
        """Value at 1-based columns 34-46, matching PEST .ins extraction."""
        from pyiwfm.calibration.calctyphyd import (
            CalcTypHydTimeSeriesResult,
            TypicalHydrographTimeSeries,
        )

        th = TypicalHydrographTimeSeries(
            cluster_id=1,
            dates=[datetime(2010, 1, 15)],
            values=[123.456789],
            pest_names=["TH1_1"],
        )
        result = CalcTypHydTimeSeriesResult(hydrographs=[th], well_means={})
        file_config = CalcTypHydFileConfig(weights_path=Path("wt.txt"))

        write_pest_output(result, file_config, tmp_path)

        out_path = tmp_path / "sim_wt_cls1.out"
        lines = out_path.read_text().split("\n")
        data_line = lines[1]
        # Fortran format: name(16) + date(10) + value(F20.6)
        # PEST .ins extracts at 34:46 (1-based) = Python [33:46]
        value_region = data_line[33:46]
        assert float(value_region.strip()) == pytest.approx(123.456789)


class TestWeightsHeaderSkip:
    """Tests for header detection in read_cluster_weights."""

    def test_header_skipped(self, tmp_path: Path) -> None:
        """Non-numeric header row is detected and skipped."""
        weights_file = tmp_path / "weights.txt"
        weights_file.write_text(
            "well_id  memb1  memb2  cluster  subregion\nW1  0.8  0.2  1  5\nW2  0.3  0.7  2  3\n"
        )

        weights = read_cluster_weights(weights_file, n_clusters=2)
        assert len(weights) == 2
        np.testing.assert_allclose(weights["W1"], [0.8, 0.2])
        np.testing.assert_allclose(weights["W2"], [0.3, 0.7])

    def test_no_header(self, tmp_path: Path) -> None:
        """Files without headers still work."""
        weights_file = tmp_path / "weights.txt"
        weights_file.write_text("W1 0.8 0.2\nW2 0.3 0.7\n")

        weights = read_cluster_weights(weights_file)
        assert len(weights) == 2

    def test_n_clusters_limits_columns(self, tmp_path: Path) -> None:
        """n_clusters=2 ignores extra columns."""
        weights_file = tmp_path / "weights.txt"
        weights_file.write_text("W1 0.8 0.2 1 5\nW2 0.3 0.7 2 3\n")

        weights = read_cluster_weights(weights_file, n_clusters=2)
        assert len(weights["W1"]) == 2
        np.testing.assert_allclose(weights["W1"], [0.8, 0.2])
