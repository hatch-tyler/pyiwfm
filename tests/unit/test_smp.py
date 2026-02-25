"""Tests for SMP file reader/writer."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from pyiwfm.io.smp import (
    SMPReader,
    SMPTimeSeries,
    SMPWriter,
    _is_sentinel,
    _parse_smp_line,
)


class TestSMPParsing:
    """Tests for SMP line parsing."""

    def test_parse_fixed_width_line(self) -> None:
        line = "WELL_01                  01/15/2020   12:00:00    150.25"
        rec = _parse_smp_line(line)
        assert rec is not None
        assert rec.bore_id == "WELL_01"
        assert rec.datetime == datetime(2020, 1, 15, 12, 0, 0)
        assert rec.value == pytest.approx(150.25)
        assert rec.excluded is False

    def test_parse_excluded_record(self) -> None:
        line = "WELL_02                  03/20/2020   00:00:00    200.50  X"
        rec = _parse_smp_line(line)
        assert rec is not None
        assert rec.excluded is True

    def test_parse_sentinel_becomes_nan(self) -> None:
        line = "WELL_03                  06/01/2020   00:00:00  -1.1E+38"
        rec = _parse_smp_line(line)
        assert rec is not None
        assert np.isnan(rec.value)

    def test_parse_whitespace_format(self) -> None:
        line = "WELL_01 01/15/2020 12:00:00 150.25"
        rec = _parse_smp_line(line)
        assert rec is not None
        assert rec.bore_id == "WELL_01"
        assert rec.value == pytest.approx(150.25)

    def test_parse_blank_line_returns_none(self) -> None:
        assert _parse_smp_line("") is None
        assert _parse_smp_line("   ") is None

    def test_is_sentinel(self) -> None:
        assert _is_sentinel(-1.1e38)
        assert _is_sentinel(-9.1e37)
        assert _is_sentinel(float("nan"))
        assert _is_sentinel(float("inf"))
        assert not _is_sentinel(150.0)
        assert not _is_sentinel(0.0)
        assert not _is_sentinel(-50.0)


class TestSMPRoundTrip:
    """Tests for SMP read → write → read round-trip."""

    def test_roundtrip(self, tmp_path: Path) -> None:
        """Write SMP → read SMP → verify bore IDs, times, values match."""
        bore_id = "TEST_WELL_01"
        n = 5
        times = np.array(
            ["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01", "2020-05-01"],
            dtype="datetime64[s]",
        )
        values = np.array([100.0, 101.5, 99.0, 102.3, 100.8], dtype=np.float64)
        excluded = np.array([False, False, True, False, False], dtype=np.bool_)

        ts = SMPTimeSeries(bore_id=bore_id, times=times, values=values, excluded=excluded)

        # Write
        smp_file = tmp_path / "test.smp"
        writer = SMPWriter(smp_file)
        writer.write({bore_id: ts})

        # Read back
        reader = SMPReader(smp_file)
        data = reader.read()

        assert bore_id in data
        result = data[bore_id]
        assert result.bore_id == bore_id
        assert len(result.times) == n
        np.testing.assert_allclose(result.values, values, atol=0.01)
        np.testing.assert_array_equal(result.excluded, excluded)

    def test_roundtrip_multiple_bores(self, tmp_path: Path) -> None:
        """Write multiple bores and read back."""
        smp_file = tmp_path / "multi.smp"
        data: dict[str, SMPTimeSeries] = {}

        for i in range(3):
            bore_id = f"BORE_{i:02d}"
            times = np.array(
                ["2020-01-01", "2020-06-01", "2020-12-01"],
                dtype="datetime64[s]",
            )
            values = np.array([50.0 + i * 10, 55.0 + i * 10, 48.0 + i * 10])
            excluded = np.zeros(3, dtype=np.bool_)
            data[bore_id] = SMPTimeSeries(
                bore_id=bore_id, times=times, values=values, excluded=excluded
            )

        writer = SMPWriter(smp_file)
        writer.write(data)

        reader = SMPReader(smp_file)
        result = reader.read()

        assert len(result) == 3
        assert set(result.keys()) == {"BORE_00", "BORE_01", "BORE_02"}

    def test_roundtrip_nan_values(self, tmp_path: Path) -> None:
        """NaN values round-trip as NaN (sentinel)."""
        smp_file = tmp_path / "nan.smp"
        ts = SMPTimeSeries(
            bore_id="NAN_WELL",
            times=np.array(["2020-01-01", "2020-02-01"], dtype="datetime64[s]"),
            values=np.array([100.0, float("nan")]),
            excluded=np.zeros(2, dtype=np.bool_),
        )

        SMPWriter(smp_file).write({"NAN_WELL": ts})
        result = SMPReader(smp_file).read()

        assert "NAN_WELL" in result
        assert np.isnan(result["NAN_WELL"].values[1])

    def test_bore_ids_property(self, tmp_path: Path) -> None:
        """bore_ids returns unique IDs in order."""
        smp_file = tmp_path / "bores.smp"
        data = {}
        for bid in ["ZZZ", "AAA", "MMM"]:
            data[bid] = SMPTimeSeries(
                bore_id=bid,
                times=np.array(["2020-01-01"], dtype="datetime64[s]"),
                values=np.array([100.0]),
                excluded=np.zeros(1, dtype=np.bool_),
            )
        SMPWriter(smp_file).write(data)

        reader = SMPReader(smp_file)
        ids = reader.bore_ids
        assert len(ids) == 3
        assert set(ids) == {"ZZZ", "AAA", "MMM"}

    def test_read_bore_single(self, tmp_path: Path) -> None:
        """read_bore returns only the requested bore."""
        smp_file = tmp_path / "single.smp"
        data = {}
        for bid in ["W1", "W2"]:
            data[bid] = SMPTimeSeries(
                bore_id=bid,
                times=np.array(["2020-01-01"], dtype="datetime64[s]"),
                values=np.array([100.0 if bid == "W1" else 200.0]),
                excluded=np.zeros(1, dtype=np.bool_),
            )
        SMPWriter(smp_file).write(data)

        reader = SMPReader(smp_file)
        ts = reader.read_bore("W2")
        assert ts is not None
        assert ts.bore_id == "W2"
        assert ts.values[0] == pytest.approx(200.0, abs=0.01)

    def test_read_bore_not_found(self, tmp_path: Path) -> None:
        """read_bore returns None for unknown bore."""
        smp_file = tmp_path / "empty.smp"
        data = {
            "W1": SMPTimeSeries(
                bore_id="W1",
                times=np.array(["2020-01-01"], dtype="datetime64[s]"),
                values=np.array([100.0]),
                excluded=np.zeros(1, dtype=np.bool_),
            )
        }
        SMPWriter(smp_file).write(data)
        assert SMPReader(smp_file).read_bore("MISSING") is None


class TestSMPTimeSeries:
    """Tests for SMPTimeSeries properties."""

    def test_n_records(self) -> None:
        ts = SMPTimeSeries(
            bore_id="W1",
            times=np.array(["2020-01-01", "2020-02-01", "2020-03-01"], dtype="datetime64[s]"),
            values=np.array([100.0, 101.0, 102.0]),
            excluded=np.zeros(3, dtype=np.bool_),
        )
        assert ts.n_records == 3

    def test_valid_mask(self) -> None:
        ts = SMPTimeSeries(
            bore_id="W1",
            times=np.array(["2020-01-01", "2020-02-01", "2020-03-01"], dtype="datetime64[s]"),
            values=np.array([100.0, float("nan"), 102.0]),
            excluded=np.array([False, False, True]),
        )
        expected = np.array([True, False, False])
        np.testing.assert_array_equal(ts.valid_mask, expected)
