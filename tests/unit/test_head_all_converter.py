"""Unit tests for the IWFM GWALLOUTFL text-to-HDF5 converter."""

from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

# Skip all tests if h5py is not available
h5py = pytest.importorskip("h5py")

from pyiwfm.io.head_all_converter import (  # noqa: E402
    _COL_WIDTH,
    _TIME_WIDTH,
    _count_data_lines,
    _parse_data_line_numpy,
    _parse_node_ids,
    _parse_timestamp,
    convert_headall_to_hdf,
    main,
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
    # "* NODE ..." header
    lines.append("*            NODE\n")
    # "* TIME  node1  node2 ..." header -- 21 chars then 12-char fields
    time_field = "*            TIME    "
    assert len(time_field) == _TIME_WIDTH
    node_fields = "".join(f"{nid:>{_COL_WIDTH}}" for nid in node_ids)
    lines.append(f"{time_field}{node_fields}\n")
    return "".join(lines)


def _make_data_line(timestamp: str | None, values: list[float]) -> str:
    """Build one data line.

    If *timestamp* is ``None`` the first 21 chars are spaces (continuation row).
    """
    if timestamp is not None:
        ts_field = f" {timestamp}"
        ts_field = ts_field.ljust(_TIME_WIDTH)
    else:
        ts_field = " " * _TIME_WIDTH
    val_fields = "".join(f"{v:>{_COL_WIDTH}.4f}" for v in values)
    return f"{ts_field}{val_fields}\n"


def _make_single_layer_file(
    node_ids: list[int],
    timesteps: list[tuple[str, list[float]]],
) -> str:
    """Build a complete single-layer GWALLOUTFL file.

    Parameters
    ----------
    node_ids : list[int]
        Node IDs for the header.
    timesteps : list of (timestamp_str, values) tuples
        Each tuple is a timestamp string and a list of head values.
    """
    content = _make_header(node_ids)
    for ts, vals in timesteps:
        content += _make_data_line(ts, vals)
    return content


def _make_multi_layer_file(
    node_ids: list[int],
    timesteps: list[tuple[str, list[list[float]]]],
) -> str:
    """Build a complete multi-layer GWALLOUTFL file.

    Parameters
    ----------
    node_ids : list[int]
        Node IDs for the header.
    timesteps : list of (timestamp_str, layer_values) tuples
        ``layer_values`` is a list-of-lists: one inner list per layer.
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
# Tests: _parse_timestamp
# ---------------------------------------------------------------------------


class TestParseTimestamp:
    """Tests for _parse_timestamp."""

    def test_normal_time(self) -> None:
        result = _parse_timestamp("01/15/2000_12:30")
        assert result == datetime(2000, 1, 15, 12, 30)

    def test_24_00_end_of_day(self) -> None:
        """``24:00`` means midnight at the end of the given day."""
        result = _parse_timestamp("01/31/1990_24:00")
        assert result == datetime(1990, 2, 1, 0, 0)

    def test_24_00_end_of_year(self) -> None:
        """``12/31/1999_24:00`` should roll over to 2000-01-01."""
        result = _parse_timestamp("12/31/1999_24:00")
        assert result == datetime(2000, 1, 1, 0, 0)

    def test_midnight_00_00(self) -> None:
        result = _parse_timestamp("06/15/2020_00:00")
        assert result == datetime(2020, 6, 15, 0, 0)

    def test_leading_trailing_whitespace(self) -> None:
        result = _parse_timestamp("  03/01/2010_06:00  ")
        assert result == datetime(2010, 3, 1, 6, 0)

    def test_leap_day(self) -> None:
        result = _parse_timestamp("02/29/2000_12:00")
        assert result == datetime(2000, 2, 29, 12, 0)

    def test_24_00_on_feb_28_non_leap(self) -> None:
        result = _parse_timestamp("02/28/2001_24:00")
        assert result == datetime(2001, 3, 1, 0, 0)

    def test_24_00_on_feb_29_leap(self) -> None:
        result = _parse_timestamp("02/29/2000_24:00")
        assert result == datetime(2000, 3, 1, 0, 0)


# ---------------------------------------------------------------------------
# Tests: _parse_node_ids
# ---------------------------------------------------------------------------


class TestParseNodeIds:
    """Tests for _parse_node_ids."""

    def test_normal_header(self) -> None:
        header = "*            TIME    " + "           1           2           3"
        ids = _parse_node_ids(header)
        assert ids == [1, 2, 3]

    def test_large_node_ids(self) -> None:
        header = "*            TIME    " + "        1001        2002        3003"
        ids = _parse_node_ids(header)
        assert ids == [1001, 2002, 3003]

    def test_single_node(self) -> None:
        header = "*            TIME    " + "          42"
        ids = _parse_node_ids(header)
        assert ids == [42]

    def test_empty_header(self) -> None:
        """A header with no columns after the TIME field returns no IDs."""
        header = "*            TIME    "
        ids = _parse_node_ids(header)
        assert ids == []

    def test_non_integer_chunks_skipped(self) -> None:
        """Non-integer tokens in column positions are silently skipped."""
        header = "*            TIME    " + "           1       abc           3"
        ids = _parse_node_ids(header)
        assert ids == [1, 3]

    def test_all_non_integer(self) -> None:
        header = "*            TIME    " + "        abc1       def2"
        ids = _parse_node_ids(header)
        assert ids == []

    def test_many_nodes(self) -> None:
        """Works with many nodes."""
        n = 50
        header = "*            TIME    " + "".join(f"{i:>{_COL_WIDTH}}" for i in range(1, n + 1))
        ids = _parse_node_ids(header)
        assert ids == list(range(1, n + 1))


# ---------------------------------------------------------------------------
# Tests: _parse_data_line_numpy
# ---------------------------------------------------------------------------


class TestParseDataLineNumpy:
    """Tests for _parse_data_line_numpy."""

    def test_whitespace_split_path(self) -> None:
        """When values are separated by whitespace, the fast split path is used."""
        line = " 01/31/1990_24:00     100.1234     200.5678     300.9012"
        result = _parse_data_line_numpy(line, 3)
        np.testing.assert_allclose(result, [100.1234, 200.5678, 300.9012])

    def test_continuation_line(self) -> None:
        """Continuation lines start with spaces instead of a timestamp."""
        line = "                      110.1234     210.5678     310.9012"
        result = _parse_data_line_numpy(line, 3)
        np.testing.assert_allclose(result, [110.1234, 210.5678, 310.9012])

    def test_fixed_width_fallback_negative_numbers(self) -> None:
        """Negative numbers abutting previous values force the fixed-width fallback.

        When a negative value fills its entire 12-char column and the next
        column also starts with a minus sign, ``split()`` merges them into
        one token, yielding fewer parts than n_nodes.  The code then falls
        back to fixed-width column slicing at _COL_WIDTH boundaries.

        Example data_part: ``"  -100.12340-200000.5678   -300.9012"``
        - split() gives ``['-100.12340-200000.5678', '-300.9012']`` (2 < 3)
        - Fixed-width: ``[0:12]='  -100.12340'``, ``[12:24]='-200000.5678'``,
          ``[24:36]='   -300.9012'``
        """
        ts_field = " " * _TIME_WIDTH
        # Each column is exactly _COL_WIDTH (12) chars.
        # col1 ends with a digit, col2 starts with '-', so they abut.
        col1 = "  -100.12340"  # 12 chars, last char is a digit
        col2 = "-200000.5678"  # 12 chars, starts with '-' -> no whitespace gap
        col3 = "   -300.9012"  # 12 chars, normal with leading spaces
        assert len(col1) == _COL_WIDTH
        assert len(col2) == _COL_WIDTH
        assert len(col3) == _COL_WIDTH
        data_part = col1 + col2 + col3
        line = ts_field + data_part
        result = _parse_data_line_numpy(line, 3)
        np.testing.assert_allclose(result, [-100.12340, -200000.5678, -300.9012])

    def test_empty_chunk_produces_nan(self) -> None:
        """An empty column in fixed-width mode should produce NaN."""
        ts_field = " " * _TIME_WIDTH
        col1 = "    100.1234"
        col2 = "            "  # 12 spaces -> empty chunk
        col3 = "    300.9012"
        # Force fallback by making split() return fewer tokens than n_nodes.
        # col2 is all spaces, so split gives ['100.1234', '300.9012'] -> 2 < 3
        line = ts_field + col1 + col2 + col3
        result = _parse_data_line_numpy(line, 3)
        assert result[0] == pytest.approx(100.1234)
        assert np.isnan(result[1])
        assert result[2] == pytest.approx(300.9012)

    def test_extra_parts_ignored(self) -> None:
        """Extra columns beyond n_nodes are silently ignored."""
        line = " 01/31/1990_24:00     100.0000     200.0000     300.0000     400.0000"
        result = _parse_data_line_numpy(line, 3)
        np.testing.assert_allclose(result, [100.0, 200.0, 300.0])
        assert result.shape == (3,)

    def test_returns_float64_array(self) -> None:
        line = " 01/31/1990_24:00     100.1234     200.5678     300.9012"
        result = _parse_data_line_numpy(line, 3)
        assert result.dtype == np.float64


# ---------------------------------------------------------------------------
# Tests: _count_data_lines
# ---------------------------------------------------------------------------


class TestCountDataLines:
    """Tests for _count_data_lines."""

    def test_basic_count(self) -> None:
        content = "header1\nheader2\ndata1\ndata2\ndata3\n"
        fh = io.StringIO(content)
        assert _count_data_lines(fh, 2) == 3

    def test_file_position_restored(self) -> None:
        content = "header\ndata1\ndata2\n"
        fh = io.StringIO(content)
        _count_data_lines(fh, 1)
        assert fh.tell() == 0

    def test_zero_data_lines(self) -> None:
        content = "header1\nheader2\n"
        fh = io.StringIO(content)
        assert _count_data_lines(fh, 2) == 0

    def test_header_lines_exceeds_total(self) -> None:
        """If header_lines exceeds total, should return 0 (not negative)."""
        content = "only\n"
        fh = io.StringIO(content)
        assert _count_data_lines(fh, 10) == 0

    def test_empty_file(self) -> None:
        fh = io.StringIO("")
        assert _count_data_lines(fh, 0) == 0

    def test_large_header(self) -> None:
        lines = "".join(f"line{i}\n" for i in range(100))
        fh = io.StringIO(lines)
        assert _count_data_lines(fh, 6) == 94


# ---------------------------------------------------------------------------
# Tests: convert_headall_to_hdf  (integration / end-to-end)
# ---------------------------------------------------------------------------


class TestConvertHeadallToHdf:
    """Tests for the main converter function."""

    def _write_text_file(self, tmp_path: Path, content: str) -> Path:
        """Write IWFM text content to a file and return its path."""
        path = tmp_path / "GW_HeadAll.out"
        path.write_text(content, encoding="utf-8")
        return path

    # -- single layer ---------------------------------------------------

    def test_single_layer_basic(self, tmp_path: Path) -> None:
        """Convert a simple single-layer file and verify HDF5 contents."""
        content = _make_single_layer_file(
            node_ids=[1, 2, 3],
            timesteps=[
                ("01/31/1990_24:00", [100.1234, 200.5678, 300.9012]),
                ("02/28/1990_24:00", [101.1234, 201.5678, 301.9012]),
            ],
        )
        text_path = self._write_text_file(tmp_path, content)
        hdf_path = tmp_path / "output.hdf"
        result = convert_headall_to_hdf(text_path, hdf_path, n_layers=1)

        assert result == hdf_path
        assert hdf_path.exists()

        with h5py.File(hdf_path, "r") as hf:
            head = hf["head"][:]
            assert head.shape == (2, 3, 1)
            np.testing.assert_allclose(head[0, :, 0], [100.1234, 200.5678, 300.9012])
            np.testing.assert_allclose(head[1, :, 0], [101.1234, 201.5678, 301.9012])

            times = [t.decode() if isinstance(t, bytes) else t for t in hf["times"][:]]
            assert len(times) == 2
            # 01/31/1990_24:00 -> 1990-02-01T00:00:00
            assert times[0] == "1990-02-01T00:00:00"
            # 02/28/1990_24:00 -> 1990-03-01T00:00:00
            assert times[1] == "1990-03-01T00:00:00"

            assert hf.attrs["n_nodes"] == 3
            assert hf.attrs["n_layers"] == 1
            assert hf.attrs["source"] == "GW_HeadAll.out"

    # -- multi layer ---------------------------------------------------

    def test_multi_layer(self, tmp_path: Path) -> None:
        """Convert a 2-layer file and verify that layers are stored correctly."""
        content = _make_multi_layer_file(
            node_ids=[1, 2, 3],
            timesteps=[
                (
                    "01/31/1990_24:00",
                    [
                        [100.1234, 200.5678, 300.9012],  # layer 1
                        [110.1234, 210.5678, 310.9012],  # layer 2
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
        result = convert_headall_to_hdf(text_path, hdf_path, n_layers=2)

        assert result == hdf_path

        with h5py.File(hdf_path, "r") as hf:
            head = hf["head"][:]
            assert head.shape == (2, 3, 2)
            # timestep 0, layer 0
            np.testing.assert_allclose(head[0, :, 0], [100.1234, 200.5678, 300.9012])
            # timestep 0, layer 1
            np.testing.assert_allclose(head[0, :, 1], [110.1234, 210.5678, 310.9012])
            # timestep 1, layer 0
            np.testing.assert_allclose(head[1, :, 0], [101.1234, 201.5678, 301.9012])
            # timestep 1, layer 1
            np.testing.assert_allclose(head[1, :, 1], [111.1234, 211.5678, 311.9012])

            assert hf.attrs["n_layers"] == 2

    # -- default output path -------------------------------------------

    def test_default_output_path(self, tmp_path: Path) -> None:
        """When hdf_file is None, the output uses the same name with .hdf extension."""
        content = _make_single_layer_file(
            node_ids=[1],
            timesteps=[("06/15/2020_00:00", [42.0])],
        )
        text_path = self._write_text_file(tmp_path, content)
        result = convert_headall_to_hdf(text_path, hdf_file=None, n_layers=1)

        expected = text_path.with_suffix(".hdf")
        assert result == expected
        assert expected.exists()

    # -- explicit output path ------------------------------------------

    def test_explicit_output_path(self, tmp_path: Path) -> None:
        """An explicit output path is used as-is."""
        content = _make_single_layer_file(
            node_ids=[1, 2],
            timesteps=[("03/01/2010_06:00", [10.0, 20.0])],
        )
        text_path = self._write_text_file(tmp_path, content)
        hdf_path = tmp_path / "subdir" / "custom_name.h5"
        hdf_path.parent.mkdir(parents=True, exist_ok=True)
        result = convert_headall_to_hdf(text_path, hdf_path)

        assert result == hdf_path
        assert hdf_path.exists()

    # -- dataset grows past initial estimate ---------------------------

    def test_dataset_grows_past_estimate(self, tmp_path: Path) -> None:
        """When the initial estimate is too small, the dataset resizes dynamically.

        We create a file with many timesteps to exceed _CHUNK_GROW boundaries.
        The converter should handle the resize transparently.
        """
        n_timesteps = 300  # exceeds _CHUNK_GROW (256) at least once
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
        result = convert_headall_to_hdf(text_path, hdf_path, n_layers=1)

        with h5py.File(result, "r") as hf:
            head = hf["head"][:]
            assert head.shape == (n_timesteps, 2, 1)
            # spot-check first and last
            assert head[0, 0, 0] == pytest.approx(0.0)
            assert head[-1, 1, 0] == pytest.approx((n_timesteps - 1) * 10.0)

            times = list(hf["times"][:])
            assert len(times) == n_timesteps

    # -- error cases ---------------------------------------------------

    def test_error_empty_file(self, tmp_path: Path) -> None:
        """An empty file raises ValueError about unexpected end of file."""
        text_path = tmp_path / "empty.out"
        text_path.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="Unexpected end of file while reading title"):
            convert_headall_to_hdf(text_path)

    def test_error_truncated_title(self, tmp_path: Path) -> None:
        """A file with only 2 title lines raises ValueError."""
        content = "* line1\n* line2\n"
        text_path = tmp_path / "truncated.out"
        text_path.write_text(content, encoding="utf-8")
        with pytest.raises(ValueError, match="Unexpected end of file while reading title"):
            convert_headall_to_hdf(text_path)

    def test_error_missing_header(self, tmp_path: Path) -> None:
        """A file with title lines but no header lines raises ValueError."""
        content = (
            "* ===================================\n"
            "*  GROUND WATER HEAD AT ALL NODES\n"
            "*  Unit: feet\n"
            "* ===================================\n"
        )
        text_path = tmp_path / "no_header.out"
        text_path.write_text(content, encoding="utf-8")
        with pytest.raises(ValueError, match="Unexpected end of file while reading header"):
            convert_headall_to_hdf(text_path)

    def test_error_no_node_ids(self, tmp_path: Path) -> None:
        """A file where the header has no parseable node IDs raises ValueError."""
        content = (
            "* ===================================\n"
            "*  GROUND WATER HEAD AT ALL NODES\n"
            "*  Unit: feet\n"
            "* ===================================\n"
            "*            NODE\n"
            "*            TIME    \n"  # empty after TIME -- no node IDs
        )
        text_path = tmp_path / "no_nodes.out"
        text_path.write_text(content, encoding="utf-8")
        with pytest.raises(ValueError, match="Could not parse any node IDs"):
            convert_headall_to_hdf(text_path)

    # -- HDF5 content verification -------------------------------------

    def test_hdf5_compression(self, tmp_path: Path) -> None:
        """Verify the head dataset uses gzip compression."""
        content = _make_single_layer_file(
            node_ids=[1, 2, 3],
            timesteps=[("01/31/1990_24:00", [1.0, 2.0, 3.0])],
        )
        text_path = self._write_text_file(tmp_path, content)
        hdf_path = tmp_path / "compressed.hdf"
        convert_headall_to_hdf(text_path, hdf_path)

        with h5py.File(hdf_path, "r") as hf:
            ds = hf["head"]
            assert ds.compression == "gzip"
            assert ds.compression_opts == 4

    def test_hdf5_chunks(self, tmp_path: Path) -> None:
        """Verify the dataset chunk shape is (1, n_nodes, n_layers)."""
        n_nodes = 5
        content = _make_single_layer_file(
            node_ids=list(range(1, n_nodes + 1)),
            timesteps=[("01/31/1990_24:00", [float(i) for i in range(n_nodes)])],
        )
        text_path = self._write_text_file(tmp_path, content)
        hdf_path = tmp_path / "chunked.hdf"
        convert_headall_to_hdf(text_path, hdf_path, n_layers=1)

        with h5py.File(hdf_path, "r") as hf:
            ds = hf["head"]
            assert ds.chunks == (1, n_nodes, 1)

    def test_hdf5_times_dataset_dtype(self, tmp_path: Path) -> None:
        """Verify the times dataset uses a string dtype."""
        content = _make_single_layer_file(
            node_ids=[1],
            timesteps=[("01/01/2000_00:00", [0.0])],
        )
        text_path = self._write_text_file(tmp_path, content)
        hdf_path = tmp_path / "dtypes.hdf"
        convert_headall_to_hdf(text_path, hdf_path)

        with h5py.File(hdf_path, "r") as hf:
            times_ds = hf["times"]
            # h5py string dtype
            assert h5py.check_string_dtype(times_ds.dtype) is not None

    def test_head_dtype_is_float64(self, tmp_path: Path) -> None:
        """Verify that head data is stored as float64."""
        content = _make_single_layer_file(
            node_ids=[1, 2],
            timesteps=[("01/01/2000_00:00", [1.5, 2.5])],
        )
        text_path = self._write_text_file(tmp_path, content)
        hdf_path = tmp_path / "f64.hdf"
        convert_headall_to_hdf(text_path, hdf_path)

        with h5py.File(hdf_path, "r") as hf:
            assert hf["head"].dtype == np.float64

    def test_string_path_arguments(self, tmp_path: Path) -> None:
        """Verify that string paths (not Path objects) work correctly."""
        content = _make_single_layer_file(
            node_ids=[1],
            timesteps=[("01/01/2000_00:00", [42.0])],
        )
        text_path = self._write_text_file(tmp_path, content)
        hdf_path = tmp_path / "strpath.hdf"
        result = convert_headall_to_hdf(str(text_path), str(hdf_path))

        assert isinstance(result, Path)
        assert result.exists()

    def test_single_timestep(self, tmp_path: Path) -> None:
        """A file with only one timestep should produce shape (1, n_nodes, n_layers)."""
        content = _make_single_layer_file(
            node_ids=[1, 2],
            timesteps=[("07/04/2020_12:00", [100.0, 200.0])],
        )
        text_path = self._write_text_file(tmp_path, content)
        hdf_path = tmp_path / "single_ts.hdf"
        convert_headall_to_hdf(text_path, hdf_path)

        with h5py.File(hdf_path, "r") as hf:
            assert hf["head"].shape == (1, 2, 1)

    def test_comment_lines_in_data_skipped(self, tmp_path: Path) -> None:
        """Lines starting with ``*`` in the data section are skipped."""
        header = _make_header([1, 2])
        data = _make_data_line("01/31/1990_24:00", [10.0, 20.0])
        comment = "* This is an inline comment\n"
        data2 = _make_data_line("02/28/1990_24:00", [30.0, 40.0])
        content = header + data + comment + data2

        text_path = self._write_text_file(tmp_path, content)
        hdf_path = tmp_path / "comments.hdf"
        convert_headall_to_hdf(text_path, hdf_path)

        with h5py.File(hdf_path, "r") as hf:
            head = hf["head"][:]
            assert head.shape == (2, 2, 1)
            np.testing.assert_allclose(head[0, :, 0], [10.0, 20.0])
            np.testing.assert_allclose(head[1, :, 0], [30.0, 40.0])

    def test_blank_lines_in_data_skipped(self, tmp_path: Path) -> None:
        """Empty lines in the data section are skipped."""
        header = _make_header([1])
        data = _make_data_line("01/31/1990_24:00", [10.0])
        blank = "\n"
        data2 = _make_data_line("02/28/1990_24:00", [20.0])
        content = header + data + blank + data2

        text_path = self._write_text_file(tmp_path, content)
        hdf_path = tmp_path / "blanks.hdf"
        convert_headall_to_hdf(text_path, hdf_path)

        with h5py.File(hdf_path, "r") as hf:
            head = hf["head"][:]
            assert head.shape == (2, 1, 1)

    def test_three_layers(self, tmp_path: Path) -> None:
        """Verify 3-layer files parse correctly."""
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
        convert_headall_to_hdf(text_path, hdf_path, n_layers=3)

        with h5py.File(hdf_path, "r") as hf:
            head = hf["head"][:]
            assert head.shape == (1, 2, 3)
            np.testing.assert_allclose(head[0, :, 0], [100.0, 200.0])
            np.testing.assert_allclose(head[0, :, 1], [110.0, 210.0])
            np.testing.assert_allclose(head[0, :, 2], [120.0, 220.0])

    def test_source_attr_uses_filename_only(self, tmp_path: Path) -> None:
        """The 'source' attr should be the filename, not the full path."""
        content = _make_single_layer_file(
            node_ids=[1],
            timesteps=[("01/01/2000_00:00", [0.0])],
        )
        text_path = self._write_text_file(tmp_path, content)
        hdf_path = tmp_path / "source.hdf"
        convert_headall_to_hdf(text_path, hdf_path)

        with h5py.File(hdf_path, "r") as hf:
            assert hf.attrs["source"] == text_path.name


# ---------------------------------------------------------------------------
# Tests: main()  (CLI entry point)
# ---------------------------------------------------------------------------


class TestMain:
    """Tests for the CLI entry point."""

    def test_main_calls_converter(self, tmp_path: Path) -> None:
        """main() parses args and calls convert_headall_to_hdf."""
        content = _make_single_layer_file(
            node_ids=[1, 2],
            timesteps=[("01/01/2000_00:00", [1.0, 2.0])],
        )
        text_path = tmp_path / "cli_test.out"
        text_path.write_text(content, encoding="utf-8")
        hdf_path = tmp_path / "cli_test.hdf"

        with patch(
            "sys.argv",
            ["head_all_converter", str(text_path), "-o", str(hdf_path), "--layers", "1"],
        ):
            main()

        assert hdf_path.exists()
        with h5py.File(hdf_path, "r") as hf:
            assert hf["head"].shape == (1, 2, 1)

    def test_main_default_output(self, tmp_path: Path) -> None:
        """main() uses default output path when -o is not specified."""
        content = _make_single_layer_file(
            node_ids=[1],
            timesteps=[("06/15/2020_00:00", [42.0])],
        )
        text_path = tmp_path / "default_out.out"
        text_path.write_text(content, encoding="utf-8")

        with patch("sys.argv", ["head_all_converter", str(text_path)]):
            main()

        expected = text_path.with_suffix(".hdf")
        assert expected.exists()

    def test_main_verbose_flag(self, tmp_path: Path) -> None:
        """The -v flag should not cause errors."""
        content = _make_single_layer_file(
            node_ids=[1],
            timesteps=[("01/01/2000_00:00", [0.0])],
        )
        text_path = tmp_path / "verbose.out"
        text_path.write_text(content, encoding="utf-8")

        with patch("sys.argv", ["head_all_converter", str(text_path), "-v"]):
            main()

        expected = text_path.with_suffix(".hdf")
        assert expected.exists()

    def test_main_multi_layer(self, tmp_path: Path) -> None:
        """main() passes --layers to the converter."""
        content = _make_multi_layer_file(
            node_ids=[1, 2],
            timesteps=[
                (
                    "01/31/1990_24:00",
                    [
                        [10.0, 20.0],
                        [30.0, 40.0],
                    ],
                ),
            ],
        )
        text_path = tmp_path / "multilayer_cli.out"
        text_path.write_text(content, encoding="utf-8")
        hdf_path = tmp_path / "multilayer_cli.hdf"

        with patch(
            "sys.argv",
            ["head_all_converter", str(text_path), "-o", str(hdf_path), "--layers", "2"],
        ):
            main()

        with h5py.File(hdf_path, "r") as hf:
            assert hf["head"].shape == (1, 2, 2)
            assert hf.attrs["n_layers"] == 2
