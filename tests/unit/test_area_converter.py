"""
Tests for IWFM area file converter (ASCII -> HDF5) and lazy loader.

Tests cover:
- Streaming ASCII -> HDF5 conversion
- LazyAreaDataLoader random access and caching
- AreaDataManager integration
- Comment char detection fixes (Windows paths)
- Column count validation
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from pyiwfm.io.rootzone_area import (
    _is_comment,
    _iter_data_lines,
    read_area_metadata,
    read_area_timestep,
)

# =====================================================================
# Helper to write test area files
# =====================================================================


def _write_area_file(
    path: Path,
    n_elements: int = 3,
    n_crops: int = 2,
    n_timesteps: int = 2,
    factor: float = 1.0,
    dss_path: str = "",
    date_on_every_row: bool = True,
) -> None:
    """Write a minimal IWFM area time-series file.

    Args:
        date_on_every_row: If True, every row gets a date prefix.
            If False, only the first row of each timestep block gets
            the date (continuation-row format, like C2VSimFG).
    """
    lines: list[str] = ["C  Area data file"]
    # Column pointers
    cols = "  ".join(str(i) for i in range(1, n_crops + 1))
    lines.append(f"   {cols}   / column pointers")
    lines.append(f"   {factor}   / FACTARL")
    lines.append(f"   {dss_path}   / DSSFL")

    dates = [f"{m:02d}/01/2000_24:00" for m in range(10, 10 + n_timesteps)]
    for ts_idx, date in enumerate(dates):
        for eid in range(1, n_elements + 1):
            vals = "  ".join(f"{100.0 + eid + ts_idx * 10 + c:.1f}" for c in range(n_crops))
            if date_on_every_row or eid == 1:
                lines.append(f"   {date}   {eid}   {vals}")
            else:
                # Continuation row: no date, just elem_id + values
                lines.append(f"                     {eid}   {vals}")
    path.write_text("\n".join(lines) + "\n")


# =====================================================================
# Comment char detection fixes
# =====================================================================


class TestIsComment:
    def test_c_comment(self):
        assert _is_comment("C  This is a comment")

    def test_c_lowercase_comment(self):
        assert _is_comment("c  This is a comment")

    def test_star_comment(self):
        assert _is_comment("*  This is a comment")

    def test_blank_line(self):
        assert _is_comment("")
        assert _is_comment("   ")

    def test_data_line_not_comment(self):
        assert not _is_comment("   1   2   3   / data")

    def test_windows_path_not_comment(self):
        """C:\\ paths should NOT be treated as comments."""
        assert not _is_comment("C:\\model\\area.dss")
        assert not _is_comment("c:\\data\\file.dat")

    def test_leading_whitespace_not_comment(self):
        """Column-1 rule: indented C/c/* is NOT a comment."""
        assert not _is_comment("   C  This is a comment")
        assert not _is_comment("  *  asterisk comment")

    def test_leading_whitespace_windows_path(self):
        """Indented Windows paths are not comments."""
        assert not _is_comment("   C:\\model\\area.dss")


class TestIterDataLines:
    def test_filters_comments(self):
        lines = [
            "C  comment\n",
            "   1   2   / pointers\n",
            "c  another comment\n",
            "   1.0   / factor\n",
        ]
        result = _iter_data_lines(lines)
        assert len(result) == 2
        assert result[0] == "1   2"
        assert result[1] == "1.0"

    def test_preserves_windows_path(self):
        lines = [
            "C  comment\n",
            "   1   2   / pointers\n",
            "   1.0   / factor\n",
            "C:\\model\\area.dss   / DSSFL\n",
        ]
        result = _iter_data_lines(lines)
        assert len(result) == 3
        assert "C:\\model\\area.dss" in result[2]


class TestNcolsDetection:
    def test_single_ndata_header(self, tmp_path):
        """When header has single NDATA integer, detect cols from data rows."""
        lines = [
            "C  Area file",
            "   2             / NDATA (single integer, not per-col pointers)",
            "   1.0           / FACTARL",
            "                 / DSSFL",
            "   10/01/2000_24:00   1   100.0   200.0   300.0",
            "   10/01/2000_24:00   2   110.0   210.0   310.0",
        ]
        path = tmp_path / "area.dat"
        path.write_text("\n".join(lines) + "\n")

        # Header says 1 col, but data has 3 value columns
        meta = read_area_metadata(path)
        assert meta.n_columns == 3

        data = read_area_timestep(path, 0)
        assert len(data[1]) == 3


# =====================================================================
# Area converter tests
# =====================================================================


class TestAreaConverter:
    def test_basic_conversion(self, tmp_path):
        """Convert a small area file and verify HDF5 structure."""
        from pyiwfm.io.area_converter import convert_area_to_hdf

        src = tmp_path / "area.dat"
        _write_area_file(src, n_elements=3, n_crops=2, n_timesteps=5)

        hdf = convert_area_to_hdf(src, label="nonponded")
        assert hdf.exists()

        with h5py.File(hdf, "r") as f:
            assert "nonponded" in f
            ds = f["nonponded"]
            assert ds.shape == (5, 3, 2)
            assert ds.dtype == np.float64

            assert "times" in f
            times = [t.decode() for t in f["times"][:]]
            assert len(times) == 5

            assert "element_ids" in f
            eids = f["element_ids"][:]
            assert list(eids) == [1, 2, 3]

            assert f.attrs["n_elements"] == 3
            assert f.attrs["n_cols"] == 2
            assert f.attrs["label"] == "nonponded"

    def test_factor_applied(self, tmp_path):
        """Unit conversion factor is applied during conversion."""
        from pyiwfm.io.area_converter import convert_area_to_hdf

        src = tmp_path / "area.dat"
        _write_area_file(src, n_elements=1, n_crops=1, n_timesteps=1, factor=2.0)

        hdf = convert_area_to_hdf(src, label="test")

        with h5py.File(hdf, "r") as f:
            val = f["test"][0, 0, 0]
            # elem 1, crop 0, ts 0: raw = 101.0, factor = 2.0
            assert val == pytest.approx(202.0)

    def test_custom_output_path(self, tmp_path):
        """Output path can be specified explicitly."""
        from pyiwfm.io.area_converter import convert_area_to_hdf

        src = tmp_path / "area.dat"
        _write_area_file(src, n_elements=2, n_crops=1, n_timesteps=1)

        out = tmp_path / "custom.hdf"
        result = convert_area_to_hdf(src, hdf_file=out, label="area")
        assert result == out
        assert out.exists()

    def test_default_output_path(self, tmp_path):
        """Default output uses .area_cache.hdf suffix."""
        from pyiwfm.io.area_converter import convert_area_to_hdf

        src = tmp_path / "NonPondedArea.dat"
        _write_area_file(src, n_elements=2, n_crops=1, n_timesteps=1)

        result = convert_area_to_hdf(src)
        assert result.suffix == ".hdf"
        assert "area_cache" in result.name

    def test_roundtrip_values(self, tmp_path):
        """Values read from HDF5 match values read from text."""
        from pyiwfm.io.area_converter import convert_area_to_hdf

        src = tmp_path / "area.dat"
        _write_area_file(src, n_elements=3, n_crops=2, n_timesteps=2)

        hdf = convert_area_to_hdf(src, label="test")

        # Read text
        text_ts0 = read_area_timestep(src, 0)
        text_ts1 = read_area_timestep(src, 1)

        # Read HDF5
        with h5py.File(hdf, "r") as f:
            ds = f["test"]
            for eid in [1, 2, 3]:
                idx = eid - 1
                for col in range(2):
                    assert ds[0, idx, col] == pytest.approx(text_ts0[eid][col])
                    assert ds[1, idx, col] == pytest.approx(text_ts1[eid][col])


# =====================================================================
# LazyAreaDataLoader tests
# =====================================================================


class TestLazyAreaDataLoader:
    @pytest.fixture()
    def hdf_file(self, tmp_path):
        """Create a test HDF5 area file."""
        from pyiwfm.io.area_converter import convert_area_to_hdf

        src = tmp_path / "area.dat"
        _write_area_file(src, n_elements=4, n_crops=3, n_timesteps=10)
        return convert_area_to_hdf(src, label="nonponded")

    def test_metadata(self, hdf_file):
        from pyiwfm.io.area_loader import LazyAreaDataLoader

        loader = LazyAreaDataLoader(hdf_file, dataset="nonponded")
        assert loader.n_frames == 10
        assert loader.n_elements == 4
        assert loader.n_cols == 3
        assert len(loader.times) == 10
        assert len(loader.element_ids) == 4

    def test_get_frame(self, hdf_file):
        from pyiwfm.io.area_loader import LazyAreaDataLoader

        loader = LazyAreaDataLoader(hdf_file, dataset="nonponded")
        frame = loader.get_frame(0)
        assert frame.shape == (4, 3)
        assert frame.dtype == np.float64

    def test_get_frame_caching(self, hdf_file):
        from pyiwfm.io.area_loader import LazyAreaDataLoader

        loader = LazyAreaDataLoader(hdf_file, dataset="nonponded", cache_size=3)
        f0 = loader.get_frame(0)
        f0_again = loader.get_frame(0)
        assert np.array_equal(f0, f0_again)

    def test_get_frame_out_of_range(self, hdf_file):
        from pyiwfm.io.area_loader import LazyAreaDataLoader

        loader = LazyAreaDataLoader(hdf_file, dataset="nonponded")
        with pytest.raises(IndexError):
            loader.get_frame(999)

    def test_get_element_timeseries(self, hdf_file):
        from pyiwfm.io.area_loader import LazyAreaDataLoader

        loader = LazyAreaDataLoader(hdf_file, dataset="nonponded")
        ts = loader.get_element_timeseries(0)
        assert ts.shape == (10, 3)

    def test_get_layer_range(self, hdf_file):
        from pyiwfm.io.area_loader import LazyAreaDataLoader

        loader = LazyAreaDataLoader(hdf_file, dataset="nonponded")
        lo, hi, n = loader.get_layer_range(col=-1, max_frames=5)
        assert lo > 0
        assert hi > lo
        assert n <= 5

    def test_cache_eviction(self, hdf_file):
        from pyiwfm.io.area_loader import LazyAreaDataLoader

        loader = LazyAreaDataLoader(hdf_file, dataset="nonponded", cache_size=3)
        for i in range(5):
            loader.get_frame(i)
        # Cache should only have 3 entries
        assert len(loader._cache) == 3


# =====================================================================
# AreaDataManager tests
# =====================================================================


class TestAreaDataManager:
    def test_load_from_rootzone(self, tmp_path):
        from pyiwfm.components.rootzone import RootZone
        from pyiwfm.io.area_loader import AreaDataManager

        np_file = tmp_path / "np_area.dat"
        _write_area_file(np_file, n_elements=3, n_crops=2, n_timesteps=3)

        native_file = tmp_path / "nv_area.dat"
        _write_area_file(native_file, n_elements=3, n_crops=1, n_timesteps=3)

        rz = RootZone(n_elements=3, n_layers=1)
        rz.nonponded_area_file = np_file
        rz.native_area_file = native_file

        mgr = AreaDataManager()
        mgr.load_from_rootzone(rz, tmp_path)

        assert mgr.nonponded is not None
        assert mgr.native is not None
        assert mgr.ponded is None
        assert mgr.urban is None
        assert mgr.n_timesteps == 3

    def test_get_snapshot(self, tmp_path):
        from pyiwfm.components.rootzone import RootZone
        from pyiwfm.io.area_loader import AreaDataManager

        np_file = tmp_path / "np_area.dat"
        _write_area_file(np_file, n_elements=2, n_crops=1, n_timesteps=2)

        rz = RootZone(n_elements=2, n_layers=1)
        rz.nonponded_area_file = np_file

        mgr = AreaDataManager()
        mgr.load_from_rootzone(rz, tmp_path)

        snapshot = mgr.get_snapshot(0)
        assert len(snapshot) == 2
        assert 1 in snapshot
        assert "fractions" in snapshot[1]
        assert "dominant" in snapshot[1]
        assert "total_area" in snapshot[1]
        assert snapshot[1]["total_area"] > 0

    def test_get_element_timeseries(self, tmp_path):
        from pyiwfm.components.rootzone import RootZone
        from pyiwfm.io.area_loader import AreaDataManager

        np_file = tmp_path / "np_area.dat"
        _write_area_file(np_file, n_elements=2, n_crops=2, n_timesteps=5)

        rz = RootZone(n_elements=2, n_layers=1)
        rz.nonponded_area_file = np_file

        mgr = AreaDataManager()
        mgr.load_from_rootzone(rz, tmp_path)

        ts = mgr.get_element_timeseries(1)
        assert ts["element_id"] == 1
        assert len(ts["dates"]) == 5
        assert "nonponded" in ts
        assert len(ts["nonponded"]["areas"]) == 5

    def test_get_dates(self, tmp_path):
        from pyiwfm.components.rootzone import RootZone
        from pyiwfm.io.area_loader import AreaDataManager

        np_file = tmp_path / "np_area.dat"
        _write_area_file(np_file, n_elements=2, n_crops=1, n_timesteps=3)

        rz = RootZone(n_elements=2, n_layers=1)
        rz.nonponded_area_file = np_file

        mgr = AreaDataManager()
        mgr.load_from_rootzone(rz, tmp_path)

        dates = mgr.get_dates()
        assert len(dates) == 3


# =====================================================================
# RootZone.load_land_use_from_arrays tests
# =====================================================================


class TestLoadLandUseFromArrays:
    def test_basic_snapshot(self):
        from pyiwfm.components.rootzone import LandUseType, RootZone

        rz = RootZone(n_elements=2, n_layers=1)
        snapshot = {
            1: {
                "fractions": {
                    "agricultural": 0.6,
                    "urban": 0.3,
                    "native_riparian": 0.1,
                    "water": 0.0,
                },
                "total_area": 1000.0,
                "dominant": "agricultural",
            },
            2: {
                "fractions": {
                    "agricultural": 0.0,
                    "urban": 0.8,
                    "native_riparian": 0.2,
                    "water": 0.0,
                },
                "total_area": 500.0,
                "dominant": "urban",
            },
        }

        rz.load_land_use_from_arrays(snapshot)

        # Element 1: ag (600) + urban (300) + native (100)
        ag = [
            e
            for e in rz.element_landuse
            if e.element_id == 1 and e.land_use_type == LandUseType.AGRICULTURAL
        ]
        assert len(ag) == 1
        assert ag[0].area == pytest.approx(600.0)

        urban = [
            e
            for e in rz.element_landuse
            if e.element_id == 2 and e.land_use_type == LandUseType.URBAN
        ]
        assert len(urban) == 1
        assert urban[0].area == pytest.approx(400.0)

    def test_clears_previous(self):
        from pyiwfm.components.rootzone import ElementLandUse, LandUseType, RootZone

        rz = RootZone(n_elements=2, n_layers=1)
        rz.add_element_landuse(
            ElementLandUse(
                element_id=99,
                land_use_type=LandUseType.AGRICULTURAL,
                area=999.0,
            )
        )

        snapshot = {
            1: {
                "fractions": {
                    "agricultural": 1.0,
                    "urban": 0.0,
                    "native_riparian": 0.0,
                    "water": 0.0,
                },
                "total_area": 100.0,
                "dominant": "agricultural",
            },
        }
        rz.load_land_use_from_arrays(snapshot)

        # Old data should be gone
        old = [e for e in rz.element_landuse if e.element_id == 99]
        assert len(old) == 0


# =====================================================================
# Continuation-row format tests (date only on first row of each block)
# =====================================================================


class TestContinuationRowFormat:
    """Test parsing of area files where the date appears only on the
    first row of each timestep block (C2VSimFG format).
    """

    def test_read_area_timestep_continuation(self, tmp_path):
        """read_area_timestep handles continuation rows."""
        src = tmp_path / "area.dat"
        _write_area_file(
            src,
            n_elements=5,
            n_crops=2,
            n_timesteps=2,
            date_on_every_row=False,
        )

        ts0 = read_area_timestep(src, 0)
        assert len(ts0) == 5, f"Expected 5 elements, got {len(ts0)}"
        for eid in range(1, 6):
            assert eid in ts0, f"Element {eid} missing from timestep 0"
            assert len(ts0[eid]) == 2

        ts1 = read_area_timestep(src, 1)
        assert len(ts1) == 5
        # Values should differ between timesteps
        assert ts0[1] != ts1[1]

    def test_read_all_timesteps_continuation(self, tmp_path):
        """read_all_timesteps handles continuation rows."""
        from pyiwfm.io.rootzone_area import read_all_timesteps

        src = tmp_path / "area.dat"
        _write_area_file(
            src,
            n_elements=4,
            n_crops=3,
            n_timesteps=3,
            date_on_every_row=False,
        )

        result = read_all_timesteps(src)
        assert len(result) == 3
        for date, block in result:
            assert "/" in date
            assert len(block) == 4

    def test_converter_continuation(self, tmp_path):
        """HDF5 converter handles continuation-row format."""
        from pyiwfm.io.area_converter import convert_area_to_hdf

        src = tmp_path / "area.dat"
        _write_area_file(
            src,
            n_elements=5,
            n_crops=2,
            n_timesteps=3,
            date_on_every_row=False,
        )

        hdf = convert_area_to_hdf(src, label="test")

        with h5py.File(hdf, "r") as f:
            ds = f["test"]
            assert ds.shape == (3, 5, 2), f"Expected (3, 5, 2), got {ds.shape}"
            eids = list(f["element_ids"][:])
            assert eids == [1, 2, 3, 4, 5]
            times = [t.decode() for t in f["times"][:]]
            assert len(times) == 3

    def test_roundtrip_continuation(self, tmp_path):
        """Values match between text reader and HDF5 for continuation format."""
        from pyiwfm.io.area_converter import convert_area_to_hdf

        src = tmp_path / "area.dat"
        _write_area_file(
            src,
            n_elements=3,
            n_crops=2,
            n_timesteps=2,
            date_on_every_row=False,
        )

        # Read via text reader
        text_ts0 = read_area_timestep(src, 0)
        text_ts1 = read_area_timestep(src, 1)

        # Read via HDF5
        hdf = convert_area_to_hdf(src, label="rt")
        with h5py.File(hdf, "r") as f:
            ds = f["rt"]
            for eid in [1, 2, 3]:
                idx = eid - 1
                for col in range(2):
                    assert ds[0, idx, col] == pytest.approx(text_ts0[eid][col])
                    assert ds[1, idx, col] == pytest.approx(text_ts1[eid][col])

    def test_mixed_format_both_work(self, tmp_path):
        """Both date-on-every-row and continuation formats produce same results."""
        src_full = tmp_path / "full.dat"
        src_cont = tmp_path / "cont.dat"

        _write_area_file(
            src_full,
            n_elements=4,
            n_crops=2,
            n_timesteps=2,
            date_on_every_row=True,
        )
        _write_area_file(
            src_cont,
            n_elements=4,
            n_crops=2,
            n_timesteps=2,
            date_on_every_row=False,
        )

        full_ts0 = read_area_timestep(src_full, 0)
        cont_ts0 = read_area_timestep(src_cont, 0)

        assert len(full_ts0) == len(cont_ts0) == 4
        for eid in range(1, 5):
            for col in range(2):
                assert full_ts0[eid][col] == pytest.approx(cont_ts0[eid][col])

    def test_metadata_continuation(self, tmp_path):
        """Metadata detection works with continuation format."""
        src = tmp_path / "area.dat"
        _write_area_file(
            src,
            n_elements=3,
            n_crops=4,
            n_timesteps=1,
            date_on_every_row=False,
        )

        meta = read_area_metadata(src)
        assert meta.n_columns == 4
        assert meta.factor == 1.0
