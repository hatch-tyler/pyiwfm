"""Deep coverage tests for pyiwfm.comparison.results_differ.

Covers: all-NaN timesteps, shape mismatches, ValueError in final heads,
budget shape mismatch, budget exception handling, hydrograph < 2 points,
hydrograph exception, nested HDF5, compare_all exception paths,
_parse_head_text_file with non-numeric tokens, _parse_hydrograph_text
edge cases.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from pyiwfm.comparison.results_differ import (  # noqa: E402
    ResultsDiffer,
    _parse_head_text_file,
    _parse_hydrograph_text,
)

# ======================================================================
# compare_heads_hdf5: all-NaN timestep
# ======================================================================


class TestCompareHeadsHDF5AllNaN:
    def test_all_nan_timestep_skipped(self, tmp_path: Path) -> None:
        """Timesteps where all nodes are NaN should be skipped."""
        bd = tmp_path / "baseline"
        wd = tmp_path / "written"
        bd.mkdir()
        wd.mkdir()

        # First timestep: all NaN. Second: valid.
        b_data = np.array([[np.nan, np.nan], [1.0, 2.0]], dtype=np.float64)
        w_data = np.array([[np.nan, np.nan], [1.0, 2.0]], dtype=np.float64)

        with h5py.File(str(bd / "HeadAll.hdf"), "w") as f:
            f.create_dataset("head", data=b_data)
        with h5py.File(str(wd / "HeadAll.hdf"), "w") as f:
            f.create_dataset("head", data=w_data)

        d = ResultsDiffer(bd, wd, head_atol=0.01)
        hc = d.compare_heads_hdf5()
        assert hc is not None
        assert hc.within_tolerance is True
        assert hc.n_timesteps == 2

    def test_no_head_dataset_returns_none(self, tmp_path: Path) -> None:
        """When HDF5 has no head-like dataset, returns None."""
        bd = tmp_path / "baseline"
        wd = tmp_path / "written"
        bd.mkdir()
        wd.mkdir()

        with h5py.File(str(bd / "GWHeadAll.hdf"), "w") as f:
            f.create_dataset("flow_data", data=[1.0])
        with h5py.File(str(wd / "GWHeadAll.hdf"), "w") as f:
            f.create_dataset("flow_data", data=[1.0])

        d = ResultsDiffer(bd, wd)
        assert d.compare_heads_hdf5() is None


# ======================================================================
# compare_heads_text: shape mismatch and all-NaN
# ======================================================================


class TestCompareHeadsTextEdges:
    def _write_head_out(self, path: Path, blocks: list[list[float]]) -> None:
        lines: list[str] = []
        for i, block in enumerate(blocks):
            lines.append(f"01/{i + 1:02d}/2020_24:00  TIME STEP {i + 1}")
            for val in block:
                lines.append(f"  {val}")
        path.write_text("\n".join(lines))

    def test_shape_mismatch(self, tmp_path: Path) -> None:
        """When timestep blocks differ in length, counted as mismatch."""
        bd = tmp_path / "baseline"
        wd = tmp_path / "written"
        bd.mkdir()
        wd.mkdir()

        self._write_head_out(bd / "GW_HeadAll.out", [[1.0, 2.0, 3.0]])
        self._write_head_out(wd / "GW_HeadAll.out", [[1.0, 2.0]])

        d = ResultsDiffer(bd, wd, head_atol=0.01)
        hc = d.compare_heads_text()
        assert hc is not None
        assert hc.n_mismatched_timesteps >= 1

    def test_all_nan_text(self, tmp_path: Path) -> None:
        """Timestep blocks with all NaN values are skipped."""
        bd = tmp_path / "baseline"
        wd = tmp_path / "written"
        bd.mkdir()
        wd.mkdir()

        # Write NaN-like values - use "nan" string which float() handles
        self._write_head_out(bd / "GW_HeadAll.out", [[float("nan")]])
        self._write_head_out(wd / "GW_HeadAll.out", [[float("nan")]])

        d = ResultsDiffer(bd, wd, head_atol=0.01)
        hc = d.compare_heads_text()
        assert hc is not None
        assert hc.within_tolerance is True


# ======================================================================
# compare_final_heads: ValueError path and token count mismatch
# ======================================================================


class TestCompareFinalHeadsEdges:
    def test_non_numeric_tokens_matching(self, tmp_path: Path) -> None:
        """Non-numeric tokens that match should still pass."""
        bd = tmp_path / "baseline"
        wd = tmp_path / "written"
        bd.mkdir()
        wd.mkdir()

        (bd / "FinalGWHeads.dat").write_text("Layer 100.0\n")
        (wd / "FinalGWHeads.dat").write_text("Layer 100.0\n")

        d = ResultsDiffer(bd, wd, head_atol=0.01)
        assert d.compare_final_heads() is True

    def test_non_numeric_tokens_differing(self, tmp_path: Path) -> None:
        """Non-numeric tokens that differ should fail."""
        bd = tmp_path / "baseline"
        wd = tmp_path / "written"
        bd.mkdir()
        wd.mkdir()

        (bd / "FinalGWHeads.dat").write_text("LayerA 100.0\n")
        (wd / "FinalGWHeads.dat").write_text("LayerB 100.0\n")

        d = ResultsDiffer(bd, wd, head_atol=0.01)
        assert d.compare_final_heads() is False

    def test_token_count_mismatch_on_line(self, tmp_path: Path) -> None:
        """Lines with different token counts fail."""
        bd = tmp_path / "baseline"
        wd = tmp_path / "written"
        bd.mkdir()
        wd.mkdir()

        (bd / "FinalGWHeads.dat").write_text("1 100.0 200.0\n")
        (wd / "FinalGWHeads.dat").write_text("1 100.0\n")

        d = ResultsDiffer(bd, wd, head_atol=0.01)
        assert d.compare_final_heads() is False


# ======================================================================
# compare_budgets: shape mismatch and exception
# ======================================================================


class TestCompareBudgetsEdges:
    def test_shape_mismatch_detail(self, tmp_path: Path) -> None:
        """Shape mismatch between datasets adds a detail message."""
        bd = tmp_path / "baseline"
        wd = tmp_path / "written"
        bd.mkdir()
        wd.mkdir()

        with h5py.File(str(bd / "GW_Budget.hdf"), "w") as f:
            f.create_dataset("inflow", data=np.zeros((3, 4)))
        with h5py.File(str(wd / "GW_Budget.hdf"), "w") as f:
            f.create_dataset("inflow", data=np.zeros((3, 5)))

        d = ResultsDiffer(bd, wd)
        results = d.compare_budgets()
        assert len(results) == 1
        assert results[0].within_tolerance is False
        assert any("shape mismatch" in detail for detail in results[0].details)

    def test_exception_during_hdf5_read(self, tmp_path: Path) -> None:
        """Exceptions during HDF5 reads are caught gracefully."""
        bd = tmp_path / "baseline"
        wd = tmp_path / "written"
        bd.mkdir()
        wd.mkdir()

        # Write a valid file for baseline
        with h5py.File(str(bd / "GW_Budget.hdf"), "w") as f:
            f.create_dataset("inflow", data=np.zeros((2, 3)))

        # Write a corrupt/incompatible file for written
        (wd / "GW_Budget.hdf").write_text("not valid hdf5")

        d = ResultsDiffer(bd, wd)
        results = d.compare_budgets()
        assert len(results) == 1
        assert results[0].within_tolerance is False
        assert any("Error" in d for d in results[0].details)


# ======================================================================
# compare_hydrographs: < 2 points and exception
# ======================================================================


class TestCompareHydrographsEdges:
    def test_single_point_location_skipped(self, tmp_path: Path) -> None:
        """Locations with < 2 data points are skipped for NSE."""
        bd = tmp_path / "baseline"
        wd = tmp_path / "written"
        bd.mkdir()
        wd.mkdir()

        # Only 1 data row
        content = "01/01/2020_24:00  10.0\n"
        (bd / "GW_Hydrograph.out").write_text(content)
        (wd / "GW_Hydrograph.out").write_text(content)

        d = ResultsDiffer(bd, wd, nse_threshold=0.99)
        results = d.compare_hydrographs()
        assert len(results) == 1
        # With 0 NSE values, min_nse defaults to 0.0
        assert results[0].min_nse == 0.0
        assert results[0].n_poor_matches == 0

    def test_hydrograph_parse_exception(self, tmp_path: Path) -> None:
        """Exception during hydrograph parsing is caught."""
        bd = tmp_path / "baseline"
        wd = tmp_path / "written"
        bd.mkdir()
        wd.mkdir()

        (bd / "GW_Hydrograph.out").write_text("01/01/2020_24:00  10.0\n02/01/2020_24:00  11.0\n")
        (wd / "GW_Hydrograph.out").write_text("01/01/2020_24:00  10.0\n02/01/2020_24:00  11.0\n")

        d = ResultsDiffer(bd, wd, nse_threshold=0.99)

        with patch(
            "pyiwfm.comparison.results_differ._parse_hydrograph_text",
            side_effect=ValueError("parse error"),
        ):
            results = d.compare_hydrographs()
            assert len(results) == 1
            assert results[0].within_tolerance is False


# ======================================================================
# _list_datasets: nested groups
# ======================================================================


class TestListDatasetsNested:
    def test_deeply_nested(self, tmp_path: Path) -> None:
        """Recursively finds datasets in nested groups."""
        p = tmp_path / "deep.hdf"
        with h5py.File(str(p), "w") as f:
            g1 = f.create_group("level1")
            g2 = g1.create_group("level2")
            g2.create_dataset("deep_data", data=[1.0])
            f.create_dataset("top_data", data=[2.0])

        with h5py.File(str(p), "r") as f:
            ds = ResultsDiffer._list_datasets(f)

        assert "top_data" in ds
        assert "level1/level2/deep_data" in ds


# ======================================================================
# compare_all: exception propagation for all branches
# ======================================================================


class TestCompareAllExceptions:
    def test_final_heads_exception(self, tmp_path: Path) -> None:
        """Exception in compare_final_heads is caught."""
        bd = tmp_path / "baseline"
        wd = tmp_path / "written"
        bd.mkdir()
        wd.mkdir()

        d = ResultsDiffer(bd, wd)
        with patch.object(d, "compare_final_heads", side_effect=OSError("disk fail")):
            result = d.compare_all()
        assert any("Final heads" in e for e in result.errors)

    def test_budgets_exception(self, tmp_path: Path) -> None:
        """Exception in compare_budgets is caught."""
        bd = tmp_path / "baseline"
        wd = tmp_path / "written"
        bd.mkdir()
        wd.mkdir()

        d = ResultsDiffer(bd, wd)
        with patch.object(d, "compare_budgets", side_effect=RuntimeError("corrupt")):
            result = d.compare_all()
        assert any("Budget" in e for e in result.errors)

    def test_hydrographs_exception(self, tmp_path: Path) -> None:
        """Exception in compare_hydrographs is caught."""
        bd = tmp_path / "baseline"
        wd = tmp_path / "written"
        bd.mkdir()
        wd.mkdir()

        d = ResultsDiffer(bd, wd)
        with patch.object(d, "compare_hydrographs", side_effect=RuntimeError("bad")):
            result = d.compare_all()
        assert any("Hydrograph" in e for e in result.errors)

    def test_heads_hdf5_returns_none_falls_through_to_text(self, tmp_path: Path) -> None:
        """When HDF5 returns None, compare_heads_text is tried."""
        bd = tmp_path / "baseline"
        wd = tmp_path / "written"
        bd.mkdir()
        wd.mkdir()

        d = ResultsDiffer(bd, wd)
        with patch.object(d, "compare_heads_hdf5", return_value=None):
            with patch.object(d, "compare_heads_text", return_value=None) as mock_text:
                result = d.compare_all()
                mock_text.assert_called_once()
                assert result.head_comparison is None


# ======================================================================
# _parse_head_text_file: non-numeric tokens
# ======================================================================


class TestParseHeadTextFileEdges:
    def test_non_numeric_tokens_ignored(self, tmp_path: Path) -> None:
        """Non-numeric tokens in data lines are silently skipped."""
        content = "01/01/2020_24:00  TIME STEP 1\n1.5 abc 3.5\n"
        p = tmp_path / "heads.out"
        p.write_text(content)

        blocks = _parse_head_text_file(p)
        assert len(blocks) == 1
        # Only the numeric tokens
        assert 1.5 in blocks[0]
        assert 3.5 in blocks[0]
        assert len(blocks[0]) == 2

    def test_no_data_start(self, tmp_path: Path) -> None:
        """File with no timestep markers returns empty."""
        content = "Some text without date patterns\nAnother line\n"
        p = tmp_path / "heads.out"
        p.write_text(content)
        assert _parse_head_text_file(p) == []


# ======================================================================
# _parse_hydrograph_text: edge cases
# ======================================================================


class TestParseHydrographTextEdges:
    def test_single_token_lines_skipped(self, tmp_path: Path) -> None:
        """Lines with < 2 tokens in data section are skipped."""
        content = "01/01/2020_24:00  10.0\nsingletoken\n02/01/2020_24:00  20.0\n"
        p = tmp_path / "hyd.out"
        p.write_text(content)

        locs = _parse_hydrograph_text(p)
        assert locs["1"] == pytest.approx([10.0, 20.0])

    def test_non_numeric_values_skipped(self, tmp_path: Path) -> None:
        """Non-numeric values in data columns are silently skipped."""
        content = "01/01/2020_24:00  10.0  abc\n02/01/2020_24:00  20.0  30.0\n"
        p = tmp_path / "hyd.out"
        p.write_text(content)

        locs = _parse_hydrograph_text(p)
        assert locs["1"] == pytest.approx([10.0, 20.0])
        assert locs["2"] == pytest.approx([30.0])

    def test_interleaved_comments(self, tmp_path: Path) -> None:
        """Comment lines interleaved with data are skipped."""
        content = (
            "01/01/2020_24:00  5.0\n* comment in data\nC another comment\n02/01/2020_24:00  6.0\n"
        )
        p = tmp_path / "hyd.out"
        p.write_text(content)

        locs = _parse_hydrograph_text(p)
        assert locs["1"] == pytest.approx([5.0, 6.0])
