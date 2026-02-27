"""Tests for pyiwfm.comparison.results_differ module.

Covers dataclass defaults, ResultsComparison.success/summary,
ResultsDiffer construction, head/budget/hydrograph comparisons,
file-discovery helpers, and text-file parsers.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from pyiwfm.comparison.results_differ import (
    BudgetComparison,
    HeadComparison,
    HydrographComparison,
    ResultsComparison,
    ResultsDiffer,
    _parse_head_text_file,
    _parse_hydrograph_text,
)


# ======================================================================
# Dataclass defaults
# ======================================================================


class TestHeadComparisonDefaults:
    def test_defaults(self) -> None:
        hc = HeadComparison()
        assert hc.n_timesteps == 0
        assert hc.n_nodes == 0
        assert hc.max_abs_diff == 0.0
        assert hc.mean_abs_diff == 0.0
        assert hc.n_mismatched_timesteps == 0
        assert hc.within_tolerance is False


class TestBudgetComparisonDefaults:
    def test_defaults(self) -> None:
        bc = BudgetComparison()
        assert bc.name == ""
        assert bc.n_datasets == 0
        assert bc.max_rel_diff == 0.0
        assert bc.within_tolerance is False
        assert bc.details == []

    def test_details_independent(self) -> None:
        """Each instance should get its own details list."""
        a = BudgetComparison()
        b = BudgetComparison()
        a.details.append("x")
        assert b.details == []


class TestHydrographComparisonDefaults:
    def test_defaults(self) -> None:
        hg = HydrographComparison()
        assert hg.name == ""
        assert hg.n_locations == 0
        assert hg.min_nse == 0.0
        assert hg.mean_nse == 0.0
        assert hg.n_poor_matches == 0
        assert hg.within_tolerance is False


class TestResultsComparisonDefaults:
    def test_defaults(self) -> None:
        rc = ResultsComparison()
        assert rc.head_comparison is None
        assert rc.budget_comparisons == []
        assert rc.hydrograph_comparisons == []
        assert rc.final_heads_match is None
        assert rc.errors == []


# ======================================================================
# ResultsComparison.success property
# ======================================================================


class TestResultsComparisonSuccess:
    def test_empty_is_success(self) -> None:
        """No comparisons at all => success (nothing to fail)."""
        assert ResultsComparison().success is True

    def test_errors_mean_failure(self) -> None:
        rc = ResultsComparison(errors=["something broke"])
        assert rc.success is False

    def test_head_failure(self) -> None:
        rc = ResultsComparison(
            head_comparison=HeadComparison(within_tolerance=False),
        )
        assert rc.success is False

    def test_head_pass(self) -> None:
        rc = ResultsComparison(
            head_comparison=HeadComparison(within_tolerance=True),
        )
        assert rc.success is True

    def test_budget_failure(self) -> None:
        rc = ResultsComparison(
            budget_comparisons=[
                BudgetComparison(name="gw", within_tolerance=True),
                BudgetComparison(name="rz", within_tolerance=False),
            ],
        )
        assert rc.success is False

    def test_hydrograph_failure(self) -> None:
        rc = ResultsComparison(
            hydrograph_comparisons=[
                HydrographComparison(name="hyd", within_tolerance=False),
            ],
        )
        assert rc.success is False

    def test_final_heads_false(self) -> None:
        rc = ResultsComparison(final_heads_match=False)
        assert rc.success is False

    def test_final_heads_none_is_ok(self) -> None:
        rc = ResultsComparison(final_heads_match=None)
        assert rc.success is True

    def test_all_pass(self) -> None:
        rc = ResultsComparison(
            head_comparison=HeadComparison(within_tolerance=True),
            budget_comparisons=[BudgetComparison(within_tolerance=True)],
            hydrograph_comparisons=[HydrographComparison(within_tolerance=True)],
            final_heads_match=True,
        )
        assert rc.success is True


# ======================================================================
# ResultsComparison.summary method
# ======================================================================


class TestResultsComparisonSummary:
    def test_summary_header(self) -> None:
        s = ResultsComparison().summary()
        assert "Results Comparison Summary" in s

    def test_summary_with_head(self) -> None:
        rc = ResultsComparison(
            head_comparison=HeadComparison(
                n_timesteps=10,
                max_abs_diff=0.001,
                mean_abs_diff=0.0005,
                n_mismatched_timesteps=0,
                within_tolerance=True,
            ),
        )
        s = rc.summary()
        assert "[PASS] Heads" in s
        assert "0/10 exceed tol" in s

    def test_summary_with_budget_fail(self) -> None:
        rc = ResultsComparison(
            budget_comparisons=[BudgetComparison(name="GW_Budget", within_tolerance=False)],
        )
        s = rc.summary()
        assert "[FAIL] Budget 'GW_Budget'" in s

    def test_summary_with_hydrograph(self) -> None:
        rc = ResultsComparison(
            hydrograph_comparisons=[
                HydrographComparison(
                    name="GW_Hyd.out",
                    n_locations=5,
                    min_nse=0.98,
                    n_poor_matches=1,
                    within_tolerance=False,
                ),
            ],
        )
        s = rc.summary()
        assert "[FAIL] Hydrograph 'GW_Hyd.out'" in s
        assert "1/5 poor" in s

    def test_summary_final_heads(self) -> None:
        rc = ResultsComparison(final_heads_match=True)
        s = rc.summary()
        assert "[PASS] Final heads" in s

    def test_summary_errors(self) -> None:
        rc = ResultsComparison(errors=["disk full"])
        s = rc.summary()
        assert "[ERROR] disk full" in s


# ======================================================================
# ResultsDiffer.__init__
# ======================================================================


class TestResultsDifferInit:
    def test_default_tolerances(self, tmp_path: Path) -> None:
        d = ResultsDiffer(tmp_path / "a", tmp_path / "b")
        assert d.head_atol == 0.01
        assert d.budget_rtol == 1e-3
        assert d.nse_threshold == 0.9999

    def test_custom_tolerances(self, tmp_path: Path) -> None:
        d = ResultsDiffer(
            tmp_path / "a", tmp_path / "b",
            head_atol=0.1, budget_rtol=0.5, nse_threshold=0.8,
        )
        assert d.head_atol == 0.1
        assert d.budget_rtol == 0.5
        assert d.nse_threshold == 0.8

    def test_str_paths_converted(self, tmp_path: Path) -> None:
        d = ResultsDiffer(str(tmp_path / "a"), str(tmp_path / "b"))
        assert isinstance(d.baseline_dir, Path)
        assert isinstance(d.written_dir, Path)


# ======================================================================
# compare_heads_hdf5
# ======================================================================


def _make_head_hdf5(path: Path, data: np.ndarray, ds_name: str = "head") -> None:
    """Helper: write a 2-D or 3-D head dataset to HDF5."""
    with h5py.File(str(path), "w") as f:
        f.create_dataset(ds_name, data=data)


class TestCompareHeadsHDF5:
    def test_matching(self, tmp_path: Path) -> None:
        bd = tmp_path / "baseline"
        wd = tmp_path / "written"
        bd.mkdir()
        wd.mkdir()

        data = np.random.rand(5, 10).astype(np.float64)
        _make_head_hdf5(bd / "GWHeadAll.hdf", data)
        _make_head_hdf5(wd / "GWHeadAll.hdf", data)

        d = ResultsDiffer(bd, wd, head_atol=0.01)
        hc = d.compare_heads_hdf5()
        assert hc is not None
        assert hc.within_tolerance is True
        assert hc.n_timesteps == 5
        assert hc.n_nodes == 10
        assert hc.max_abs_diff == pytest.approx(0.0, abs=1e-12)
        assert hc.n_mismatched_timesteps == 0

    def test_differing(self, tmp_path: Path) -> None:
        bd = tmp_path / "baseline"
        wd = tmp_path / "written"
        bd.mkdir()
        wd.mkdir()

        b_data = np.zeros((3, 4), dtype=np.float64)
        w_data = b_data.copy()
        w_data[1, 2] = 1.0  # big difference

        _make_head_hdf5(bd / "HeadAll.hdf", b_data)
        _make_head_hdf5(wd / "HeadAll.hdf", w_data)

        d = ResultsDiffer(bd, wd, head_atol=0.01)
        hc = d.compare_heads_hdf5()
        assert hc is not None
        assert hc.within_tolerance is False
        assert hc.n_mismatched_timesteps >= 1
        assert hc.max_abs_diff == pytest.approx(1.0)

    def test_missing_files(self, tmp_path: Path) -> None:
        bd = tmp_path / "baseline"
        wd = tmp_path / "written"
        bd.mkdir()
        wd.mkdir()
        # no HDF5 files created
        d = ResultsDiffer(bd, wd)
        assert d.compare_heads_hdf5() is None

    def test_nan_masking(self, tmp_path: Path) -> None:
        """NaN values in both files should be ignored."""
        bd = tmp_path / "baseline"
        wd = tmp_path / "written"
        bd.mkdir()
        wd.mkdir()

        data = np.array([[1.0, np.nan], [np.nan, 3.0]], dtype=np.float64)
        _make_head_hdf5(bd / "GWHeadAll.hdf", data)
        _make_head_hdf5(wd / "GWHeadAll.hdf", data)

        d = ResultsDiffer(bd, wd, head_atol=0.01)
        hc = d.compare_heads_hdf5()
        assert hc is not None
        assert hc.within_tolerance is True


# ======================================================================
# compare_heads_text
# ======================================================================


class TestCompareHeadsText:
    def _write_head_out(self, path: Path, blocks: list[list[float]]) -> None:
        lines: list[str] = []
        for i, block in enumerate(blocks):
            lines.append(f"01/{i + 1:02d}/2020_24:00  TIME STEP {i + 1}")
            for val in block:
                lines.append(f"  {val:.6f}")
        path.write_text("\n".join(lines))

    def test_matching(self, tmp_path: Path) -> None:
        bd = tmp_path / "baseline"
        wd = tmp_path / "written"
        bd.mkdir()
        wd.mkdir()

        vals = [1.0, 2.0, 3.0]
        self._write_head_out(bd / "GW_HeadAll.out", [vals, vals])
        self._write_head_out(wd / "GW_HeadAll.out", [vals, vals])

        d = ResultsDiffer(bd, wd, head_atol=0.01)
        hc = d.compare_heads_text()
        assert hc is not None
        assert hc.within_tolerance is True
        assert hc.n_timesteps == 2

    def test_differing(self, tmp_path: Path) -> None:
        bd = tmp_path / "baseline"
        wd = tmp_path / "written"
        bd.mkdir()
        wd.mkdir()

        self._write_head_out(bd / "GW_HeadAll.out", [[1.0, 2.0]])
        self._write_head_out(wd / "GW_HeadAll.out", [[1.0, 5.0]])

        d = ResultsDiffer(bd, wd, head_atol=0.01)
        hc = d.compare_heads_text()
        assert hc is not None
        assert hc.within_tolerance is False

    def test_missing(self, tmp_path: Path) -> None:
        bd = tmp_path / "baseline"
        wd = tmp_path / "written"
        bd.mkdir()
        wd.mkdir()
        d = ResultsDiffer(bd, wd)
        assert d.compare_heads_text() is None


# ======================================================================
# compare_final_heads
# ======================================================================


class TestCompareFinalHeads:
    def test_matching(self, tmp_path: Path) -> None:
        bd = tmp_path / "baseline"
        wd = tmp_path / "written"
        bd.mkdir()
        wd.mkdir()

        content = "1  100.001\n2  200.002\n"
        (bd / "FinalGWHeads.dat").write_text(content)
        (wd / "FinalGWHeads.dat").write_text(content)

        d = ResultsDiffer(bd, wd, head_atol=0.01)
        assert d.compare_final_heads() is True

    def test_differing(self, tmp_path: Path) -> None:
        bd = tmp_path / "baseline"
        wd = tmp_path / "written"
        bd.mkdir()
        wd.mkdir()

        (bd / "FinalGWHeads.dat").write_text("1  100.0\n")
        (wd / "FinalGWHeads.dat").write_text("1  200.0\n")

        d = ResultsDiffer(bd, wd, head_atol=0.01)
        assert d.compare_final_heads() is False

    def test_within_tolerance(self, tmp_path: Path) -> None:
        bd = tmp_path / "baseline"
        wd = tmp_path / "written"
        bd.mkdir()
        wd.mkdir()

        (bd / "FinalGWHeads.dat").write_text("1  100.000\n")
        (wd / "FinalGWHeads.dat").write_text("1  100.005\n")

        d = ResultsDiffer(bd, wd, head_atol=0.01)
        assert d.compare_final_heads() is True

    def test_missing_files(self, tmp_path: Path) -> None:
        bd = tmp_path / "baseline"
        wd = tmp_path / "written"
        bd.mkdir()
        wd.mkdir()
        d = ResultsDiffer(bd, wd)
        assert d.compare_final_heads() is None

    def test_line_count_mismatch(self, tmp_path: Path) -> None:
        bd = tmp_path / "baseline"
        wd = tmp_path / "written"
        bd.mkdir()
        wd.mkdir()

        (bd / "FinalGWHeads.dat").write_text("1  100.0\n2  200.0\n")
        (wd / "FinalGWHeads.dat").write_text("1  100.0\n")

        d = ResultsDiffer(bd, wd, head_atol=0.01)
        assert d.compare_final_heads() is False


# ======================================================================
# compare_budgets
# ======================================================================


class TestCompareBudgets:
    def test_matching_budgets(self, tmp_path: Path) -> None:
        bd = tmp_path / "baseline"
        wd = tmp_path / "written"
        bd.mkdir()
        wd.mkdir()

        data = np.arange(12, dtype=np.float64).reshape(3, 4)
        with h5py.File(str(bd / "GW_Budget.hdf"), "w") as f:
            f.create_dataset("inflow", data=data)
        with h5py.File(str(wd / "GW_Budget.hdf"), "w") as f:
            f.create_dataset("inflow", data=data)

        d = ResultsDiffer(bd, wd, budget_rtol=1e-3)
        results = d.compare_budgets()
        assert len(results) == 1
        assert results[0].within_tolerance is True

    def test_tolerance_failure(self, tmp_path: Path) -> None:
        bd = tmp_path / "baseline"
        wd = tmp_path / "written"
        bd.mkdir()
        wd.mkdir()

        b_data = np.ones((2, 3), dtype=np.float64)
        w_data = b_data * 1.1  # 10% relative difference

        with h5py.File(str(bd / "GW_Budget.hdf"), "w") as f:
            f.create_dataset("vals", data=b_data)
        with h5py.File(str(wd / "GW_Budget.hdf"), "w") as f:
            f.create_dataset("vals", data=w_data)

        d = ResultsDiffer(bd, wd, budget_rtol=0.001)
        results = d.compare_budgets()
        assert len(results) == 1
        assert results[0].within_tolerance is False
        assert results[0].max_rel_diff > 0.001

    def test_missing_budget_files(self, tmp_path: Path) -> None:
        bd = tmp_path / "baseline"
        wd = tmp_path / "written"
        bd.mkdir()
        wd.mkdir()
        d = ResultsDiffer(bd, wd)
        assert d.compare_budgets() == []


# ======================================================================
# compare_hydrographs
# ======================================================================


class TestCompareHydrographs:
    def _write_hyd_out(self, path: Path, n_ts: int, n_cols: int, offset: float = 0.0) -> None:
        lines: list[str] = []
        for t in range(n_ts):
            date = f"01/{t + 1:02d}/2020_24:00"
            vals = " ".join(f"{float(t + c + 1) + offset:.3f}" for c in range(n_cols))
            lines.append(f"{date}  {vals}")
        path.write_text("\n".join(lines))

    def test_matching_hydrographs(self, tmp_path: Path) -> None:
        bd = tmp_path / "baseline"
        wd = tmp_path / "written"
        bd.mkdir()
        wd.mkdir()

        self._write_hyd_out(bd / "GW_Hydrograph.out", n_ts=20, n_cols=3)
        self._write_hyd_out(wd / "GW_Hydrograph.out", n_ts=20, n_cols=3)

        d = ResultsDiffer(bd, wd, nse_threshold=0.99)
        results = d.compare_hydrographs()
        assert len(results) == 1
        assert results[0].within_tolerance is True
        assert results[0].min_nse == pytest.approx(1.0)

    def test_poor_nse(self, tmp_path: Path) -> None:
        bd = tmp_path / "baseline"
        wd = tmp_path / "written"
        bd.mkdir()
        wd.mkdir()

        self._write_hyd_out(bd / "GW_Hydrograph.out", n_ts=20, n_cols=2)
        self._write_hyd_out(wd / "GW_Hydrograph.out", n_ts=20, n_cols=2, offset=100.0)

        d = ResultsDiffer(bd, wd, nse_threshold=0.99)
        results = d.compare_hydrographs()
        assert len(results) == 1
        assert results[0].within_tolerance is False
        assert results[0].n_poor_matches > 0


# ======================================================================
# compare_all orchestration
# ======================================================================


class TestCompareAll:
    def test_compare_all_empty_dirs(self, tmp_path: Path) -> None:
        bd = tmp_path / "baseline"
        wd = tmp_path / "written"
        bd.mkdir()
        wd.mkdir()

        d = ResultsDiffer(bd, wd)
        result = d.compare_all()
        # No files to compare: head_comparison is None, final_heads None, empty lists
        assert result.head_comparison is None
        assert result.final_heads_match is None
        assert result.budget_comparisons == []
        assert result.hydrograph_comparisons == []
        assert result.success is True

    def test_compare_all_catches_errors(self, tmp_path: Path) -> None:
        bd = tmp_path / "baseline"
        wd = tmp_path / "written"
        bd.mkdir()
        wd.mkdir()

        d = ResultsDiffer(bd, wd)
        # Patch compare_heads_hdf5 to raise
        with patch.object(d, "compare_heads_hdf5", side_effect=RuntimeError("boom")):
            result = d.compare_all()
        assert len(result.errors) >= 1
        assert "boom" in result.errors[0]


# ======================================================================
# _find_head_hdf5, _find_file helpers
# ======================================================================


class TestFileDiscovery:
    def test_find_head_hdf5_present(self, tmp_path: Path) -> None:
        (tmp_path / "GWHeadAll.hdf").touch()
        d = ResultsDiffer(tmp_path, tmp_path)
        assert d._find_head_hdf5(tmp_path) is not None

    def test_find_head_hdf5_absent(self, tmp_path: Path) -> None:
        d = ResultsDiffer(tmp_path, tmp_path)
        assert d._find_head_hdf5(tmp_path) is None

    def test_find_file_with_suffix(self, tmp_path: Path) -> None:
        (tmp_path / "MyBudget.hdf").touch()
        found = ResultsDiffer._find_file(tmp_path, ["*Budget*"], suffix=".hdf")
        assert found is not None
        assert found.name == "MyBudget.hdf"

    def test_find_file_no_match(self, tmp_path: Path) -> None:
        assert ResultsDiffer._find_file(tmp_path, ["*nope*"]) is None

    def test_find_file_ignores_wrong_suffix(self, tmp_path: Path) -> None:
        (tmp_path / "FinalGWHeads.txt").touch()
        found = ResultsDiffer._find_file(tmp_path, ["*FinalGWHeads*"], suffix=".dat")
        assert found is None


# ======================================================================
# _parse_head_text_file
# ======================================================================


class TestParseHeadTextFile:
    def test_basic_parse(self, tmp_path: Path) -> None:
        content = textwrap.dedent("""\
            01/01/2020_24:00  TIME STEP 1
            1.5 2.5 3.5

            02/01/2020_24:00  TIME STEP 2
            4.5 5.5 6.5
        """)
        p = tmp_path / "heads.out"
        p.write_text(content)

        blocks = _parse_head_text_file(p)
        assert len(blocks) == 2
        assert blocks[0] == pytest.approx([1.5, 2.5, 3.5])
        assert blocks[1] == pytest.approx([4.5, 5.5, 6.5])

    def test_empty_file(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.out"
        p.write_text("")
        assert _parse_head_text_file(p) == []


# ======================================================================
# _parse_hydrograph_text
# ======================================================================


class TestParseHydrographText:
    def test_basic_parse(self, tmp_path: Path) -> None:
        content = textwrap.dedent("""\
            01/01/2020_24:00  10.0  20.0
            02/01/2020_24:00  11.0  21.0
            03/01/2020_24:00  12.0  22.0
        """)
        p = tmp_path / "hyd.out"
        p.write_text(content)

        locs = _parse_hydrograph_text(p)
        assert "1" in locs
        assert "2" in locs
        assert locs["1"] == pytest.approx([10.0, 11.0, 12.0])
        assert locs["2"] == pytest.approx([20.0, 21.0, 22.0])

    def test_skips_header_lines(self, tmp_path: Path) -> None:
        content = textwrap.dedent("""\
            * Header line 1
            C Comment line
            # Another comment
            01/01/2020_24:00  5.0
        """)
        p = tmp_path / "hyd.out"
        p.write_text(content)

        locs = _parse_hydrograph_text(p)
        assert "1" in locs
        assert locs["1"] == pytest.approx([5.0])

    def test_empty_file(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.out"
        p.write_text("")
        assert _parse_hydrograph_text(p) == {}


# ======================================================================
# _find_head_dataset
# ======================================================================


class TestFindHeadDataset:
    def test_exact_name(self, tmp_path: Path) -> None:
        p = tmp_path / "test.hdf"
        with h5py.File(str(p), "w") as f:
            f.create_dataset("head", data=[1.0])
        with h5py.File(str(p), "r") as f:
            assert ResultsDiffer._find_head_dataset(f) == "head"

    def test_alternative_names(self, tmp_path: Path) -> None:
        p = tmp_path / "test.hdf"
        with h5py.File(str(p), "w") as f:
            f.create_dataset("GWHeadAtAllNodes", data=[1.0])
        with h5py.File(str(p), "r") as f:
            assert ResultsDiffer._find_head_dataset(f) == "GWHeadAtAllNodes"

    def test_fallback_search(self, tmp_path: Path) -> None:
        p = tmp_path / "test.hdf"
        with h5py.File(str(p), "w") as f:
            f.create_dataset("my_head_data", data=[1.0])
        with h5py.File(str(p), "r") as f:
            assert ResultsDiffer._find_head_dataset(f) == "my_head_data"

    def test_none_when_absent(self, tmp_path: Path) -> None:
        p = tmp_path / "test.hdf"
        with h5py.File(str(p), "w") as f:
            f.create_dataset("flow", data=[1.0])
        with h5py.File(str(p), "r") as f:
            assert ResultsDiffer._find_head_dataset(f) is None


# ======================================================================
# _list_datasets
# ======================================================================


class TestListDatasets:
    def test_flat(self, tmp_path: Path) -> None:
        p = tmp_path / "test.hdf"
        with h5py.File(str(p), "w") as f:
            f.create_dataset("a", data=[1.0])
            f.create_dataset("b", data=[2.0])
        with h5py.File(str(p), "r") as f:
            ds = ResultsDiffer._list_datasets(f)
        assert sorted(ds) == ["a", "b"]

    def test_nested(self, tmp_path: Path) -> None:
        p = tmp_path / "test.hdf"
        with h5py.File(str(p), "w") as f:
            g = f.create_group("grp")
            g.create_dataset("c", data=[3.0])
        with h5py.File(str(p), "r") as f:
            ds = ResultsDiffer._list_datasets(f)
        assert ds == ["grp/c"]
