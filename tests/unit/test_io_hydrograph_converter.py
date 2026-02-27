"""Unit tests for pyiwfm.io.hydrograph_converter."""

from __future__ import annotations

from pathlib import Path

import pytest

h5py = pytest.importorskip("h5py")

import numpy as np  # noqa: E402

from pyiwfm.io.hydrograph_converter import (  # noqa: E402
    _parse_timestamp,
    convert_hydrograph_to_hdf,
)

# ---------------------------------------------------------------------------
# _parse_timestamp
# ---------------------------------------------------------------------------


class TestParseTimestamp:
    """Tests for the IWFM timestamp parser."""

    def test_normal_timestamp(self) -> None:
        result = _parse_timestamp("01/31/2020_12:00")
        assert result == "2020-01-31T12:00:00"

    def test_24_00_wraps_to_next_day(self) -> None:
        result = _parse_timestamp("01/31/2020_24:00")
        assert result == "2020-02-01T00:00:00"

    def test_midnight(self) -> None:
        result = _parse_timestamp("06/15/2019_00:00")
        assert result == "2019-06-15T00:00:00"

    def test_with_whitespace(self) -> None:
        result = _parse_timestamp("  03/01/2021_06:30  ")
        assert result == "2021-03-01T06:30:00"

    def test_end_of_year_24_00(self) -> None:
        result = _parse_timestamp("12/31/2020_24:00")
        assert result == "2021-01-01T00:00:00"


# ---------------------------------------------------------------------------
# convert_hydrograph_to_hdf
# ---------------------------------------------------------------------------

_SAMPLE_HYDROGRAPH = """\
* IWFM Groundwater Hydrograph Output
* HYDROGRAPH ID  1  2  3
* LAYER  1  1  1
* NODE  10  20  30
*
01/31/2020_12:00    100.5   200.3   300.1
02/29/2020_12:00    101.0   201.0   301.0
03/31/2020_24:00    102.5   202.5   302.5
"""


class TestConvertHydrographToHdf:
    """Tests for convert_hydrograph_to_hdf()."""

    def test_default_output_path(self, tmp_path: Path) -> None:
        txt_file = tmp_path / "GW_Hydrographs.out"
        txt_file.write_text(_SAMPLE_HYDROGRAPH)

        result = convert_hydrograph_to_hdf(txt_file)
        expected = txt_file.with_suffix(".hydrograph_cache.hdf")
        assert result == expected
        assert expected.exists()

    def test_explicit_output_path(self, tmp_path: Path) -> None:
        txt_file = tmp_path / "GW_Hydrographs.out"
        txt_file.write_text(_SAMPLE_HYDROGRAPH)
        hdf_out = tmp_path / "custom_output.hdf"

        result = convert_hydrograph_to_hdf(txt_file, hdf_out)
        assert result == hdf_out
        assert hdf_out.exists()

    def test_data_shape(self, tmp_path: Path) -> None:
        txt_file = tmp_path / "test.out"
        txt_file.write_text(_SAMPLE_HYDROGRAPH)
        hdf_out = tmp_path / "test.hdf"

        convert_hydrograph_to_hdf(txt_file, hdf_out)

        with h5py.File(hdf_out, "r") as f:
            data = f["data"][:]
            assert data.shape == (3, 3)  # 3 timesteps, 3 columns

    def test_data_values(self, tmp_path: Path) -> None:
        txt_file = tmp_path / "test.out"
        txt_file.write_text(_SAMPLE_HYDROGRAPH)
        hdf_out = tmp_path / "test.hdf"

        convert_hydrograph_to_hdf(txt_file, hdf_out)

        with h5py.File(hdf_out, "r") as f:
            data = f["data"][:]
            np.testing.assert_allclose(data[0], [100.5, 200.3, 300.1])
            np.testing.assert_allclose(data[1], [101.0, 201.0, 301.0])
            np.testing.assert_allclose(data[2], [102.5, 202.5, 302.5])

    def test_times_dataset(self, tmp_path: Path) -> None:
        txt_file = tmp_path / "test.out"
        txt_file.write_text(_SAMPLE_HYDROGRAPH)
        hdf_out = tmp_path / "test.hdf"

        convert_hydrograph_to_hdf(txt_file, hdf_out)

        with h5py.File(hdf_out, "r") as f:
            raw = f["times"][:]
            times = [t.decode() if isinstance(t, bytes) else str(t) for t in raw]
            assert len(times) == 3
            assert times[0] == "2020-01-31T12:00:00"
            assert times[1] == "2020-02-29T12:00:00"
            # 24:00 wraps to next day
            assert times[2] == "2020-04-01T00:00:00"

    def test_hydrograph_ids(self, tmp_path: Path) -> None:
        txt_file = tmp_path / "test.out"
        txt_file.write_text(_SAMPLE_HYDROGRAPH)
        hdf_out = tmp_path / "test.hdf"

        convert_hydrograph_to_hdf(txt_file, hdf_out)

        with h5py.File(hdf_out, "r") as f:
            ids = f["hydrograph_ids"][:].tolist()
            assert ids == [1, 2, 3]

    def test_layers(self, tmp_path: Path) -> None:
        txt_file = tmp_path / "test.out"
        txt_file.write_text(_SAMPLE_HYDROGRAPH)
        hdf_out = tmp_path / "test.hdf"

        convert_hydrograph_to_hdf(txt_file, hdf_out)

        with h5py.File(hdf_out, "r") as f:
            lyrs = f["layers"][:].tolist()
            assert lyrs == [1, 1, 1]

    def test_node_ids(self, tmp_path: Path) -> None:
        txt_file = tmp_path / "test.out"
        txt_file.write_text(_SAMPLE_HYDROGRAPH)
        hdf_out = tmp_path / "test.hdf"

        convert_hydrograph_to_hdf(txt_file, hdf_out)

        with h5py.File(hdf_out, "r") as f:
            nids = f["node_ids"][:].tolist()
            assert nids == [10, 20, 30]

    def test_attrs(self, tmp_path: Path) -> None:
        txt_file = tmp_path / "test.out"
        txt_file.write_text(_SAMPLE_HYDROGRAPH)
        hdf_out = tmp_path / "test.hdf"

        convert_hydrograph_to_hdf(txt_file, hdf_out)

        with h5py.File(hdf_out, "r") as f:
            assert f.attrs["n_columns"] == 3
            assert f.attrs["n_timesteps"] == 3
            assert f.attrs["source"] == "test.out"

    def test_empty_data_raises(self, tmp_path: Path) -> None:
        txt_file = tmp_path / "empty.out"
        txt_file.write_text("* Header only\n* No data\n")

        with pytest.raises(ValueError, match="No data found"):
            convert_hydrograph_to_hdf(txt_file)
