"""Unit tests for the IWFM GWALLOUTFL text-to-HDF5 converter.

In v2.0 the converter was folded into :class:`TimeSeriesCache` (in
``pyiwfm.io.timeseries.lazy``); this module's tests cover the public
``TimeSeriesCache.from_iwfm_headall_text`` entrypoint plus the small
internal text-parsing helpers it relies on.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest

# Skip all tests if h5py is not available
h5py = pytest.importorskip("h5py")

from pyiwfm.io.timeseries.lazy import (  # noqa: E402
    _HEADALL_COL_WIDTH,
    _HEADALL_TIME_WIDTH,
    TimeSeriesCache,
    _count_remaining_lines,
    _parse_headall_data_row,
    _parse_headall_node_ids,
)

# ---------------------------------------------------------------------------
# Helpers to build realistic IWFM text content
# ---------------------------------------------------------------------------


def _make_header(node_ids: list[int]) -> str:
    """Build the 6-line header block for a GWALLOUTFL file.

    4 title lines (each starting with ``*``) followed by ``* NODE`` and
    ``* TIME  node1  node2 ...`` lines.
    """
    lines = [
        "* ===================================\n",
        "*  GROUND WATER HEAD AT ALL NODES\n",
        "*  Unit: feet\n",
        "* ===================================\n",
    ]
    lines.append("*            NODE\n")
    time_field = "*            TIME    "
    assert len(time_field) == _HEADALL_TIME_WIDTH
    node_fields = "".join(f"{nid:>{_HEADALL_COL_WIDTH}}" for nid in node_ids)
    lines.append(f"{time_field}{node_fields}\n")
    return "".join(lines)


def _make_data_line(timestamp: str | None, values: list[float]) -> str:
    """Build one data line.

    If *timestamp* is ``None`` the first 21 chars are spaces (continuation row).
    """
    if timestamp is not None:
        ts_field = f" {timestamp}"
        ts_field = ts_field.ljust(_HEADALL_TIME_WIDTH)
    else:
        ts_field = " " * _HEADALL_TIME_WIDTH
    val_fields = "".join(f"{v:>{_HEADALL_COL_WIDTH}.4f}" for v in values)
    return f"{ts_field}{val_fields}\n"


def _make_single_layer_file(
    node_ids: list[int],
    timesteps: list[tuple[str, list[float]]],
) -> str:
    """Build a complete single-layer GWALLOUTFL file."""
    content = _make_header(node_ids)
    for ts, vals in timesteps:
        content += _make_data_line(ts, vals)
    return content


def _make_multi_layer_file(
    node_ids: list[int],
    timesteps: list[tuple[str, list[list[float]]]],
) -> str:
    """Build a complete multi-layer GWALLOUTFL file.

    ``timesteps[i] = (timestamp, [layer0_vals, layer1_vals, ...])``.
    """
    content = _make_header(node_ids)
    for ts, layer_vals in timesteps:
        for layer_idx, vals in enumerate(layer_vals):
            if layer_idx == 0:
                content += _make_data_line(ts, vals)
            else:
                content += _make_data_line(None, vals)
    return content


# ---------------------------------------------------------------------------
# Tests: _parse_headall_node_ids
# ---------------------------------------------------------------------------


class TestParseNodeIds:
    """Tests for the fixed-width node-id header parser."""

    def test_normal_header(self) -> None:
        header = "*            TIME    " + "           1           2           3"
        ids = _parse_headall_node_ids(header)
        assert ids == [1, 2, 3]

    def test_large_node_ids(self) -> None:
        header = "*            TIME    " + "        1001        2002        3003"
        ids = _parse_headall_node_ids(header)
        assert ids == [1001, 2002, 3003]

    def test_single_node(self) -> None:
        header = "*            TIME    " + "          42"
        ids = _parse_headall_node_ids(header)
        assert ids == [42]

    def test_empty_header(self) -> None:
        """A header with no columns after the TIME field returns no IDs."""
        header = "*            TIME    "
        ids = _parse_headall_node_ids(header)
        assert ids == []

    def test_non_integer_chunks_skipped(self) -> None:
        """Non-integer tokens in column positions are silently skipped."""
        header = "*            TIME    " + "           1       abc           3"
        ids = _parse_headall_node_ids(header)
        assert ids == [1, 3]

    def test_all_non_integer(self) -> None:
        header = "*            TIME    " + "        abc1       def2"
        ids = _parse_headall_node_ids(header)
        assert ids == []

    def test_many_nodes(self) -> None:
        n = 50
        header = "*            TIME    " + "".join(
            f"{i:>{_HEADALL_COL_WIDTH}}" for i in range(1, n + 1)
        )
        ids = _parse_headall_node_ids(header)
        assert ids == list(range(1, n + 1))


# ---------------------------------------------------------------------------
# Tests: _parse_headall_data_row
# ---------------------------------------------------------------------------


class TestParseDataRow:
    """Tests for the fixed-width / whitespace data-row parser."""

    def test_whitespace_split_path(self) -> None:
        line = " 01/31/1990_24:00     100.1234     200.5678     300.9012"
        result = _parse_headall_data_row(line, 3)
        np.testing.assert_allclose(result, [100.1234, 200.5678, 300.9012])

    def test_continuation_line(self) -> None:
        """Continuation lines start with spaces instead of a timestamp."""
        line = "                      110.1234     210.5678     310.9012"
        result = _parse_headall_data_row(line, 3)
        np.testing.assert_allclose(result, [110.1234, 210.5678, 310.9012])

    def test_fixed_width_fallback_negative_numbers(self) -> None:
        """Negative numbers abutting previous values force fixed-width fallback."""
        ts_field = " " * _HEADALL_TIME_WIDTH
        col1 = "  -100.12340"
        col2 = "-200000.5678"
        col3 = "   -300.9012"
        assert len(col1) == _HEADALL_COL_WIDTH
        assert len(col2) == _HEADALL_COL_WIDTH
        assert len(col3) == _HEADALL_COL_WIDTH
        line = ts_field + col1 + col2 + col3
        result = _parse_headall_data_row(line, 3)
        np.testing.assert_allclose(result, [-100.12340, -200000.5678, -300.9012])

    def test_empty_chunk_produces_nan(self) -> None:
        """An empty column in fixed-width fallback should produce NaN."""
        ts_field = " " * _HEADALL_TIME_WIDTH
        col1 = "    100.1234"
        col2 = "            "
        col3 = "    300.9012"
        line = ts_field + col1 + col2 + col3
        result = _parse_headall_data_row(line, 3)
        assert result[0] == pytest.approx(100.1234)
        assert np.isnan(result[1])
        assert result[2] == pytest.approx(300.9012)

    def test_extra_parts_ignored(self) -> None:
        line = " 01/31/1990_24:00     100.0000     200.0000     300.0000     400.0000"
        result = _parse_headall_data_row(line, 3)
        np.testing.assert_allclose(result, [100.0, 200.0, 300.0])
        assert result.shape == (3,)

    def test_returns_float64_array(self) -> None:
        line = " 01/31/1990_24:00     100.1234     200.5678     300.9012"
        result = _parse_headall_data_row(line, 3)
        assert result.dtype == np.float64


# ---------------------------------------------------------------------------
# Tests: _count_remaining_lines
# ---------------------------------------------------------------------------


class TestCountRemainingLines:
    """Tests for the line-counting helper."""

    def test_basic_count(self) -> None:
        fh = io.StringIO("data1\ndata2\ndata3\n")
        assert _count_remaining_lines(fh) == 3

    def test_file_position_restored(self) -> None:
        fh = io.StringIO("a\nb\nc\n")
        fh.seek(2)
        _count_remaining_lines(fh)
        assert fh.tell() == 2

    def test_empty_remainder(self) -> None:
        fh = io.StringIO("")
        assert _count_remaining_lines(fh) == 0

    def test_partial_position(self) -> None:
        """Counting should start at the current position, not at byte 0."""
        fh = io.StringIO("h1\nh2\nd1\nd2\nd3\n")
        # Skip past the two "header" lines.
        fh.readline()
        fh.readline()
        assert _count_remaining_lines(fh) == 3


# ---------------------------------------------------------------------------
# Tests: TimeSeriesCache.from_iwfm_headall_text  (integration / end-to-end)
# ---------------------------------------------------------------------------


class TestFromIwfmHeadallText:
    """Tests for the public converter entrypoint."""

    def _write_text_file(self, tmp_path: Path, content: str) -> Path:
        """Write IWFM text content to a file and return its path."""
        path = tmp_path / "GW_HeadAll.out"
        path.write_text(content, encoding="utf-8")
        return path

    def test_single_layer_basic(self, tmp_path: Path) -> None:
        content = _make_single_layer_file(
            node_ids=[1, 2, 3],
            timesteps=[
                ("01/31/1990_24:00", [100.1234, 200.5678, 300.9012]),
                ("02/28/1990_24:00", [101.1234, 201.5678, 301.9012]),
            ],
        )
        text_path = self._write_text_file(tmp_path, content)
        hdf_path = tmp_path / "output.hdf"
        result = TimeSeriesCache.from_iwfm_headall_text(text_path, hdf_path, n_layers=1)

        assert result == hdf_path
        assert hdf_path.exists()

        with h5py.File(hdf_path, "r") as hf:
            head = hf["head"][:]
            assert head.shape == (2, 3, 1)
            np.testing.assert_allclose(head[0, :, 0], [100.1234, 200.5678, 300.9012])
            np.testing.assert_allclose(head[1, :, 0], [101.1234, 201.5678, 301.9012])

            times = [t.decode() if isinstance(t, bytes) else t for t in hf["times"][:]]
            assert len(times) == 2
            assert times[0] == "1990-02-01T00:00:00"
            assert times[1] == "1990-03-01T00:00:00"

            assert hf.attrs["n_nodes"] == 3
            assert hf.attrs["n_layers"] == 1
            assert hf.attrs["source"] == "GW_HeadAll.out"

    def test_multi_layer(self, tmp_path: Path) -> None:
        content = _make_multi_layer_file(
            node_ids=[1, 2, 3],
            timesteps=[
                (
                    "01/31/1990_24:00",
                    [
                        [100.1234, 200.5678, 300.9012],
                        [110.1234, 210.5678, 310.9012],
                    ],
                ),
                (
                    "02/28/1990_24:00",
                    [
                        [101.1234, 201.5678, 301.9012],
                        [111.1234, 211.5678, 311.9012],
                    ],
                ),
            ],
        )
        text_path = self._write_text_file(tmp_path, content)
        hdf_path = tmp_path / "multilayer.hdf"
        TimeSeriesCache.from_iwfm_headall_text(text_path, hdf_path, n_layers=2)

        with h5py.File(hdf_path, "r") as hf:
            head = hf["head"][:]
            assert head.shape == (2, 3, 2)
            np.testing.assert_allclose(head[0, :, 0], [100.1234, 200.5678, 300.9012])
            np.testing.assert_allclose(head[0, :, 1], [110.1234, 210.5678, 310.9012])
            np.testing.assert_allclose(head[1, :, 0], [101.1234, 201.5678, 301.9012])
            np.testing.assert_allclose(head[1, :, 1], [111.1234, 211.5678, 311.9012])
            assert hf.attrs["n_layers"] == 2

    def test_default_output_path(self, tmp_path: Path) -> None:
        content = _make_single_layer_file(
            node_ids=[1],
            timesteps=[("06/15/2020_00:00", [42.0])],
        )
        text_path = self._write_text_file(tmp_path, content)
        result = TimeSeriesCache.from_iwfm_headall_text(text_path, hdf_file=None, n_layers=1)

        expected = text_path.with_suffix(".hdf")
        assert result == expected
        assert expected.exists()

    def test_explicit_output_path(self, tmp_path: Path) -> None:
        content = _make_single_layer_file(
            node_ids=[1, 2],
            timesteps=[("03/01/2010_06:00", [10.0, 20.0])],
        )
        text_path = self._write_text_file(tmp_path, content)
        hdf_path = tmp_path / "subdir" / "custom_name.h5"
        hdf_path.parent.mkdir(parents=True, exist_ok=True)
        result = TimeSeriesCache.from_iwfm_headall_text(text_path, hdf_path)

        assert result == hdf_path
        assert hdf_path.exists()

    def test_dataset_grows_past_estimate(self, tmp_path: Path) -> None:
        """When the initial estimate is too small, the dataset resizes dynamically."""
        n_timesteps = 300  # exceeds the 256-row chunk grow size at least once
        node_ids = [1, 2]
        timesteps = []
        for i in range(n_timesteps):
            month = (i % 12) + 1
            year = 1990 + i // 12
            ts = f"{month:02d}/01/{year}_00:00"
            timesteps.append((ts, [float(i), float(i * 10)]))

        content = _make_single_layer_file(node_ids, timesteps)
        text_path = self._write_text_file(tmp_path, content)
        hdf_path = tmp_path / "grow.hdf"
        result = TimeSeriesCache.from_iwfm_headall_text(text_path, hdf_path, n_layers=1)

        with h5py.File(result, "r") as hf:
            head = hf["head"][:]
            assert head.shape == (n_timesteps, 2, 1)
            assert head[0, 0, 0] == pytest.approx(0.0)
            assert head[-1, 1, 0] == pytest.approx((n_timesteps - 1) * 10.0)

            times = list(hf["times"][:])
            assert len(times) == n_timesteps

    def test_error_empty_file(self, tmp_path: Path) -> None:
        text_path = tmp_path / "empty.out"
        text_path.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="Unexpected end of file while reading title"):
            TimeSeriesCache.from_iwfm_headall_text(text_path)

    def test_error_truncated_title(self, tmp_path: Path) -> None:
        content = "* line1\n* line2\n"
        text_path = tmp_path / "truncated.out"
        text_path.write_text(content, encoding="utf-8")
        with pytest.raises(ValueError, match="Unexpected end of file while reading title"):
            TimeSeriesCache.from_iwfm_headall_text(text_path)

    def test_error_missing_header(self, tmp_path: Path) -> None:
        content = (
            "* ===================================\n"
            "*  GROUND WATER HEAD AT ALL NODES\n"
            "*  Unit: feet\n"
            "* ===================================\n"
        )
        text_path = tmp_path / "no_header.out"
        text_path.write_text(content, encoding="utf-8")
        with pytest.raises(ValueError, match="Unexpected end of file while reading header"):
            TimeSeriesCache.from_iwfm_headall_text(text_path)

    def test_error_no_node_ids(self, tmp_path: Path) -> None:
        content = (
            "* ===================================\n"
            "*  GROUND WATER HEAD AT ALL NODES\n"
            "*  Unit: feet\n"
            "* ===================================\n"
            "*            NODE\n"
            "*            TIME    \n"
        )
        text_path = tmp_path / "no_nodes.out"
        text_path.write_text(content, encoding="utf-8")
        with pytest.raises(ValueError, match="Could not parse any node IDs"):
            TimeSeriesCache.from_iwfm_headall_text(text_path)

    def test_hdf5_compression(self, tmp_path: Path) -> None:
        content = _make_single_layer_file(
            node_ids=[1, 2, 3],
            timesteps=[("01/31/1990_24:00", [1.0, 2.0, 3.0])],
        )
        text_path = self._write_text_file(tmp_path, content)
        hdf_path = tmp_path / "compressed.hdf"
        TimeSeriesCache.from_iwfm_headall_text(text_path, hdf_path)

        with h5py.File(hdf_path, "r") as hf:
            ds = hf["head"]
            assert ds.compression == "gzip"
            assert ds.compression_opts == 4

    def test_hdf5_chunks(self, tmp_path: Path) -> None:
        n_nodes = 5
        content = _make_single_layer_file(
            node_ids=list(range(1, n_nodes + 1)),
            timesteps=[("01/31/1990_24:00", [float(i) for i in range(n_nodes)])],
        )
        text_path = self._write_text_file(tmp_path, content)
        hdf_path = tmp_path / "chunked.hdf"
        TimeSeriesCache.from_iwfm_headall_text(text_path, hdf_path, n_layers=1)

        with h5py.File(hdf_path, "r") as hf:
            ds = hf["head"]
            assert ds.chunks == (1, n_nodes, 1)

    def test_hdf5_times_dataset_dtype(self, tmp_path: Path) -> None:
        content = _make_single_layer_file(
            node_ids=[1],
            timesteps=[("01/01/2000_00:00", [0.0])],
        )
        text_path = self._write_text_file(tmp_path, content)
        hdf_path = tmp_path / "dtypes.hdf"
        TimeSeriesCache.from_iwfm_headall_text(text_path, hdf_path)

        with h5py.File(hdf_path, "r") as hf:
            assert h5py.check_string_dtype(hf["times"].dtype) is not None

    def test_head_dtype_is_float64(self, tmp_path: Path) -> None:
        content = _make_single_layer_file(
            node_ids=[1, 2],
            timesteps=[("01/01/2000_00:00", [1.5, 2.5])],
        )
        text_path = self._write_text_file(tmp_path, content)
        hdf_path = tmp_path / "f64.hdf"
        TimeSeriesCache.from_iwfm_headall_text(text_path, hdf_path)

        with h5py.File(hdf_path, "r") as hf:
            assert hf["head"].dtype == np.float64

    def test_string_path_arguments(self, tmp_path: Path) -> None:
        content = _make_single_layer_file(
            node_ids=[1],
            timesteps=[("01/01/2000_00:00", [42.0])],
        )
        text_path = self._write_text_file(tmp_path, content)
        hdf_path = tmp_path / "strpath.hdf"
        result = TimeSeriesCache.from_iwfm_headall_text(str(text_path), str(hdf_path))

        assert isinstance(result, Path)
        assert result.exists()

    def test_single_timestep(self, tmp_path: Path) -> None:
        content = _make_single_layer_file(
            node_ids=[1, 2],
            timesteps=[("07/04/2020_12:00", [100.0, 200.0])],
        )
        text_path = self._write_text_file(tmp_path, content)
        hdf_path = tmp_path / "single_ts.hdf"
        TimeSeriesCache.from_iwfm_headall_text(text_path, hdf_path)

        with h5py.File(hdf_path, "r") as hf:
            assert hf["head"].shape == (1, 2, 1)

    def test_comment_lines_in_data_skipped(self, tmp_path: Path) -> None:
        header = _make_header([1, 2])
        data = _make_data_line("01/31/1990_24:00", [10.0, 20.0])
        comment = "* This is an inline comment\n"
        data2 = _make_data_line("02/28/1990_24:00", [30.0, 40.0])
        content = header + data + comment + data2

        text_path = self._write_text_file(tmp_path, content)
        hdf_path = tmp_path / "comments.hdf"
        TimeSeriesCache.from_iwfm_headall_text(text_path, hdf_path)

        with h5py.File(hdf_path, "r") as hf:
            head = hf["head"][:]
            assert head.shape == (2, 2, 1)
            np.testing.assert_allclose(head[0, :, 0], [10.0, 20.0])
            np.testing.assert_allclose(head[1, :, 0], [30.0, 40.0])

    def test_blank_lines_in_data_skipped(self, tmp_path: Path) -> None:
        header = _make_header([1])
        data = _make_data_line("01/31/1990_24:00", [10.0])
        blank = "\n"
        data2 = _make_data_line("02/28/1990_24:00", [20.0])
        content = header + data + blank + data2

        text_path = self._write_text_file(tmp_path, content)
        hdf_path = tmp_path / "blanks.hdf"
        TimeSeriesCache.from_iwfm_headall_text(text_path, hdf_path)

        with h5py.File(hdf_path, "r") as hf:
            head = hf["head"][:]
            assert head.shape == (2, 1, 1)

    def test_three_layers(self, tmp_path: Path) -> None:
        content = _make_multi_layer_file(
            node_ids=[1, 2],
            timesteps=[
                (
                    "01/31/1990_24:00",
                    [
                        [100.0, 200.0],
                        [110.0, 210.0],
                        [120.0, 220.0],
                    ],
                ),
            ],
        )
        text_path = self._write_text_file(tmp_path, content)
        hdf_path = tmp_path / "3layer.hdf"
        TimeSeriesCache.from_iwfm_headall_text(text_path, hdf_path, n_layers=3)

        with h5py.File(hdf_path, "r") as hf:
            head = hf["head"][:]
            assert head.shape == (1, 2, 3)
            np.testing.assert_allclose(head[0, :, 0], [100.0, 200.0])
            np.testing.assert_allclose(head[0, :, 1], [110.0, 210.0])
            np.testing.assert_allclose(head[0, :, 2], [120.0, 220.0])

    def test_source_attr_uses_filename_only(self, tmp_path: Path) -> None:
        content = _make_single_layer_file(
            node_ids=[1],
            timesteps=[("01/01/2000_00:00", [0.0])],
        )
        text_path = self._write_text_file(tmp_path, content)
        hdf_path = tmp_path / "source.hdf"
        TimeSeriesCache.from_iwfm_headall_text(text_path, hdf_path)

        with h5py.File(hdf_path, "r") as hf:
            assert hf.attrs["source"] == text_path.name
