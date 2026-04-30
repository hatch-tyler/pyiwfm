"""Unit tests for the IWFM hydrograph text-to-HDF5 converter.

In v2.0 ``convert_hydrograph_to_hdf`` was folded into
:meth:`TimeSeriesCache.from_iwfm_hydrograph_text`; this module exercises
the new entrypoint plus the shared header-parsing helper.
"""

from __future__ import annotations

from pathlib import Path

import pytest

h5py = pytest.importorskip("h5py")

import numpy as np  # noqa: E402

from pyiwfm.io.timeseries.lazy import TimeSeriesCache, _parse_hydrograph_header  # noqa: E402

# ---------------------------------------------------------------------------
# _parse_hydrograph_header — shared helper used by the converter
# ---------------------------------------------------------------------------


class TestParseHydrographHeader:
    """Tests for the shared header parser."""

    def test_extracts_all_three(self) -> None:
        header = [
            "* HYDROGRAPH ID  1  2  3",
            "* LAYER  1  1  1",
            "* NODE  10  20  30",
        ]
        ids, layers, nodes = _parse_hydrograph_header(header)
        assert ids == [1, 2, 3]
        assert layers == [1, 1, 1]
        assert nodes == [10, 20, 30]

    def test_element_keyword_recognized(self) -> None:
        """Stream hydrographs use ELEMENT instead of NODE."""
        header = ["* ELEMENT  100  200  300"]
        _, _, nodes = _parse_hydrograph_header(header)
        assert nodes == [100, 200, 300]

    def test_missing_sections_default_to_empty(self) -> None:
        header = ["* something else"]
        ids, layers, nodes = _parse_hydrograph_header(header)
        assert ids == []
        assert layers == []
        assert nodes == []

    def test_handles_non_integer_in_node_list(self) -> None:
        """Non-integer tokens in the NODE row are silently skipped."""
        header = ["* NODE  10  abc  30"]
        _, _, nodes = _parse_hydrograph_header(header)
        assert nodes == [10, 30]


# ---------------------------------------------------------------------------
# TimeSeriesCache.from_iwfm_hydrograph_text  (integration)
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


class TestFromIwfmHydrographText:
    """Tests for the public converter entrypoint."""

    def test_default_output_path(self, tmp_path: Path) -> None:
        txt_file = tmp_path / "GW_Hydrographs.out"
        txt_file.write_text(_SAMPLE_HYDROGRAPH)

        result = TimeSeriesCache.from_iwfm_hydrograph_text(txt_file)
        expected = txt_file.with_suffix(".hydrograph_cache.hdf")
        assert result == expected
        assert expected.exists()

    def test_explicit_output_path(self, tmp_path: Path) -> None:
        txt_file = tmp_path / "GW_Hydrographs.out"
        txt_file.write_text(_SAMPLE_HYDROGRAPH)
        hdf_out = tmp_path / "custom_output.hdf"

        result = TimeSeriesCache.from_iwfm_hydrograph_text(txt_file, hdf_out)
        assert result == hdf_out
        assert hdf_out.exists()

    def test_data_shape(self, tmp_path: Path) -> None:
        txt_file = tmp_path / "test.out"
        txt_file.write_text(_SAMPLE_HYDROGRAPH)
        hdf_out = tmp_path / "test.hdf"

        TimeSeriesCache.from_iwfm_hydrograph_text(txt_file, hdf_out)

        with h5py.File(hdf_out, "r") as f:
            data = f["data"][:]
            assert data.shape == (3, 3)

    def test_data_values(self, tmp_path: Path) -> None:
        txt_file = tmp_path / "test.out"
        txt_file.write_text(_SAMPLE_HYDROGRAPH)
        hdf_out = tmp_path / "test.hdf"

        TimeSeriesCache.from_iwfm_hydrograph_text(txt_file, hdf_out)

        with h5py.File(hdf_out, "r") as f:
            data = f["data"][:]
            np.testing.assert_allclose(data[0], [100.5, 200.3, 300.1])
            np.testing.assert_allclose(data[1], [101.0, 201.0, 301.0])
            np.testing.assert_allclose(data[2], [102.5, 202.5, 302.5])

    def test_times_dataset(self, tmp_path: Path) -> None:
        txt_file = tmp_path / "test.out"
        txt_file.write_text(_SAMPLE_HYDROGRAPH)
        hdf_out = tmp_path / "test.hdf"

        TimeSeriesCache.from_iwfm_hydrograph_text(txt_file, hdf_out)

        with h5py.File(hdf_out, "r") as f:
            raw = f["times"][:]
            times = [t.decode() if isinstance(t, bytes) else str(t) for t in raw]
            assert len(times) == 3
            assert times[0] == "2020-01-31T12:00:00"
            assert times[1] == "2020-02-29T12:00:00"
            assert times[2] == "2020-04-01T00:00:00"

    def test_hydrograph_ids(self, tmp_path: Path) -> None:
        txt_file = tmp_path / "test.out"
        txt_file.write_text(_SAMPLE_HYDROGRAPH)
        hdf_out = tmp_path / "test.hdf"

        TimeSeriesCache.from_iwfm_hydrograph_text(txt_file, hdf_out)

        with h5py.File(hdf_out, "r") as f:
            ids = f["hydrograph_ids"][:].tolist()
            assert ids == [1, 2, 3]

    def test_layers(self, tmp_path: Path) -> None:
        txt_file = tmp_path / "test.out"
        txt_file.write_text(_SAMPLE_HYDROGRAPH)
        hdf_out = tmp_path / "test.hdf"

        TimeSeriesCache.from_iwfm_hydrograph_text(txt_file, hdf_out)

        with h5py.File(hdf_out, "r") as f:
            lyrs = f["layers"][:].tolist()
            assert lyrs == [1, 1, 1]

    def test_node_ids(self, tmp_path: Path) -> None:
        txt_file = tmp_path / "test.out"
        txt_file.write_text(_SAMPLE_HYDROGRAPH)
        hdf_out = tmp_path / "test.hdf"

        TimeSeriesCache.from_iwfm_hydrograph_text(txt_file, hdf_out)

        with h5py.File(hdf_out, "r") as f:
            nids = f["node_ids"][:].tolist()
            assert nids == [10, 20, 30]

    def test_attrs(self, tmp_path: Path) -> None:
        txt_file = tmp_path / "test.out"
        txt_file.write_text(_SAMPLE_HYDROGRAPH)
        hdf_out = tmp_path / "test.hdf"

        TimeSeriesCache.from_iwfm_hydrograph_text(txt_file, hdf_out)

        with h5py.File(hdf_out, "r") as f:
            assert f.attrs["n_columns"] == 3
            assert f.attrs["n_timesteps"] == 3
            assert f.attrs["source"] == "test.out"

    def test_empty_data_raises(self, tmp_path: Path) -> None:
        txt_file = tmp_path / "empty.out"
        txt_file.write_text("* Header only\n* No data\n")

        with pytest.raises(ValueError, match="No data found"):
            TimeSeriesCache.from_iwfm_hydrograph_text(txt_file)
