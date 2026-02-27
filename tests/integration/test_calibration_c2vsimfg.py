"""Integration tests: compare pyiwfm calibration against Fortran C2VSimFG output.

Requires ``C2VSIMFG_DIR`` env var pointing to the C2VSimFG Simulation directory.
Only runs with ``pytest -m integration``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from pyiwfm.calibration.calctyphyd import (
    compute_typical_hydrographs,
    read_cluster_weights,
)
from pyiwfm.calibration.iwfm2obs import iwfm2obs
from pyiwfm.io.smp import SMPReader

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_C2VSIMFG_DIR_VAR = "C2VSIMFG_DIR"


@pytest.fixture()
def c2vsim_dir() -> Path:
    """Resolve C2VSimFG Simulation directory from env var."""
    raw = os.environ.get(_C2VSIMFG_DIR_VAR)
    if not raw:
        pytest.skip(f"{_C2VSIMFG_DIR_VAR} not set")
    d = Path(raw)
    if not d.exists():
        pytest.skip(f"{_C2VSIMFG_DIR_VAR} path does not exist: {d}")
    return d


def _resolve(base: Path, *candidates: str) -> Path:
    """Find the first existing file among candidates."""
    for c in candidates:
        p = base / c
        if p.exists():
            return p
    pytest.skip(f"None of {candidates} found in {base}")
    raise AssertionError("unreachable")  # for type checker


# ---------------------------------------------------------------------------
# TestIWFM2OBSPerLayer — per-layer interpolation vs Fortran output
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.slow
class TestIWFM2OBSPerLayer:
    """Compare pyiwfm iwfm2obs() against Fortran per-layer output.

    Fortran validation baseline: GW_OUT.smp max diff = 0.0 vs old output.
    """

    def test_gw_per_layer(self, c2vsim_dir: Path, tmp_path: Path) -> None:
        """GW per-layer: pyiwfm vs Fortran GW_OUT.smp for 20 sample bores."""
        obs_path = _resolve(c2vsim_dir, "GW_Obs.smp")
        sim_path = _resolve(c2vsim_dir, "gw_temp.smp")
        expected_path = _resolve(c2vsim_dir, "GW_OUT.smp")
        out_path = tmp_path / "gw_out.smp"

        result = iwfm2obs(obs_path, sim_path, out_path)
        expected = SMPReader(expected_path).read()

        # Check a sample of bores (first 20)
        common = sorted(set(result.keys()) & set(expected.keys()))[:20]
        assert len(common) >= 1, "No common bore IDs found"

        for bid in common:
            if bid not in result or bid not in expected:
                continue
            r_vals = result[bid].values
            e_vals = expected[bid].values
            n = min(len(r_vals), len(e_vals))
            np.testing.assert_allclose(
                r_vals[:n],
                e_vals[:n],
                atol=0.01,
                err_msg=f"GW mismatch for bore {bid}",
            )

    def test_str_per_layer(self, c2vsim_dir: Path, tmp_path: Path) -> None:
        """Stream per-layer: pyiwfm vs Fortran STR_OUT.smp."""
        obs_path = _resolve(c2vsim_dir, "STR_Obs.smp")
        sim_path = _resolve(c2vsim_dir, "st_temp.smp")
        expected_path = _resolve(c2vsim_dir, "STR_OUT.smp")
        out_path = tmp_path / "str_out.smp"

        result = iwfm2obs(obs_path, sim_path, out_path)
        expected = SMPReader(expected_path).read()

        common = sorted(set(result.keys()) & set(expected.keys()))[:20]
        assert len(common) >= 1, "No common bore IDs found"

        for bid in common:
            r_vals = result[bid].values
            e_vals = expected[bid].values
            n = min(len(r_vals), len(e_vals))
            np.testing.assert_allclose(
                r_vals[:n],
                e_vals[:n],
                atol=0.01,
                err_msg=f"STR mismatch for bore {bid}",
            )

    def test_output_file_completeness(self, c2vsim_dir: Path, tmp_path: Path) -> None:
        """All bores in obs file appear in output (if also in sim)."""
        obs_path = _resolve(c2vsim_dir, "GW_Obs.smp")
        sim_path = _resolve(c2vsim_dir, "gw_temp.smp")
        out_path = tmp_path / "gw_out.smp"

        result = iwfm2obs(obs_path, sim_path, out_path)
        obs_bores = set(SMPReader(obs_path).bore_ids)
        sim_bores = set(SMPReader(sim_path).bore_ids)
        expected_bores = obs_bores & sim_bores

        assert set(result.keys()) == expected_bores


# ---------------------------------------------------------------------------
# TestCalcTypHydGroundTruth — typical hydrographs vs Fortran output
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.slow
class TestCalcTypHydGroundTruth:
    """Compare pyiwfm CalcTypHyd against Fortran CalcTypeHyd output."""

    @pytest.fixture()
    def calctyphyd_dir(self, c2vsim_dir: Path) -> Path:
        """Locate CalcTypeHyd directory."""
        candidates = [
            c2vsim_dir.parent / "CalcTypeHyd",
            c2vsim_dir / "CalcTypeHyd",
            c2vsim_dir.parent.parent / "CalcTypeHyd",
        ]
        for d in candidates:
            if d.exists():
                return d
        pytest.skip("CalcTypeHyd directory not found")
        raise AssertionError("unreachable")

    def test_cluster4_output(self, c2vsim_dir: Path, calctyphyd_dir: Path) -> None:
        """Cluster 4 typical hydrograph matches Fortran output."""
        wl_path = _resolve(c2vsim_dir, "GW_OUT_ml.smp")
        weights_path = _resolve(calctyphyd_dir, "allsubs.dat")
        expected_path = _resolve(calctyphyd_dir, "sim_allsubs_cls4.out")

        wl_data = SMPReader(wl_path).read()
        weights = read_cluster_weights(weights_path)
        result = compute_typical_hydrographs(wl_data, weights)

        # Load Fortran output
        expected_values = _load_fortran_calctyphyd_output(expected_path)

        # Cluster 4 (0-indexed = 4)
        if len(result.hydrographs) > 4:
            hyd = result.hydrographs[4]
            valid = ~np.isnan(hyd.values) & ~np.isnan(expected_values)
            if np.any(valid):
                np.testing.assert_allclose(
                    hyd.values[valid],
                    expected_values[valid],
                    atol=0.5,
                    err_msg="Cluster 4 typical hydrograph mismatch",
                )

    def test_all_clusters_valid(self, c2vsim_dir: Path, calctyphyd_dir: Path) -> None:
        """All clusters produce non-NaN typical hydrographs."""
        wl_path = _resolve(c2vsim_dir, "GW_OUT_ml.smp")
        weights_path = _resolve(calctyphyd_dir, "allsubs.dat")

        wl_data = SMPReader(wl_path).read()
        weights = read_cluster_weights(weights_path)
        result = compute_typical_hydrographs(wl_data, weights)

        assert len(result.hydrographs) >= 1
        for hyd in result.hydrographs:
            # At least some seasons should have valid data
            assert not np.all(np.isnan(hyd.values)), f"Cluster {hyd.cluster_id} is all-NaN"

    def test_well_means_reasonable(self, c2vsim_dir: Path, calctyphyd_dir: Path) -> None:
        """Well means are within reasonable water level range."""
        wl_path = _resolve(c2vsim_dir, "GW_OUT_ml.smp")
        weights_path = _resolve(calctyphyd_dir, "allsubs.dat")

        wl_data = SMPReader(wl_path).read()
        weights = read_cluster_weights(weights_path)
        result = compute_typical_hydrographs(wl_data, weights)

        for wid, mean in result.well_means.items():
            # C2VSimFG water levels are typically -100 to +1000 ft
            assert -500.0 < mean < 2000.0, f"Well {wid} mean {mean} outside reasonable range"


# ---------------------------------------------------------------------------
# TestRuntimeComparison — benchmarks vs Fortran timing
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.benchmark
class TestRuntimeComparison:
    """Benchmark pyiwfm against Fortran runtime (~3 min for IWFM2OBS)."""

    def test_benchmark_gw_iwfm2obs(
        self, c2vsim_dir: Path, tmp_path: Path, benchmark: object
    ) -> None:
        """Benchmark full GW IWFM2OBS on C2VSimFG data."""
        obs_path = _resolve(c2vsim_dir, "GW_Obs.smp")
        sim_path = _resolve(c2vsim_dir, "gw_temp.smp")
        out_path = tmp_path / "bench_gw_out.smp"

        benchmark(iwfm2obs, obs_path, sim_path, out_path)  # type: ignore[operator]

    def test_benchmark_calctyphyd(self, c2vsim_dir: Path, benchmark: object) -> None:
        """Benchmark CalcTypHyd on C2VSimFG multi-layer data."""
        wl_path = _resolve(c2vsim_dir, "GW_OUT_ml.smp")

        # Find CalcTypeHyd directory
        calctyphyd_dir = None
        for d in [
            c2vsim_dir.parent / "CalcTypeHyd",
            c2vsim_dir / "CalcTypeHyd",
            c2vsim_dir.parent.parent / "CalcTypeHyd",
        ]:
            if d.exists():
                calctyphyd_dir = d
                break
        if calctyphyd_dir is None:
            pytest.skip("CalcTypeHyd directory not found")
            return

        weights_path = _resolve(calctyphyd_dir, "allsubs.dat")

        wl_data = SMPReader(wl_path).read()
        weights = read_cluster_weights(weights_path)

        benchmark(compute_typical_hydrographs, wl_data, weights)  # type: ignore[operator]

    def test_benchmark_str_iwfm2obs(
        self, c2vsim_dir: Path, tmp_path: Path, benchmark: object
    ) -> None:
        """Benchmark stream IWFM2OBS on C2VSimFG data."""
        obs_path = _resolve(c2vsim_dir, "STR_Obs.smp")
        sim_path = _resolve(c2vsim_dir, "st_temp.smp")
        out_path = tmp_path / "bench_str_out.smp"

        benchmark(iwfm2obs, obs_path, sim_path, out_path)  # type: ignore[operator]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_fortran_calctyphyd_output(filepath: Path) -> np.ndarray:
    """Load Fortran CalcTypHyd output file (PEST-format or date+value)."""
    values: list[float] = []
    with open(filepath) as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("!"):
                continue
            parts = stripped.split()
            # Try last column as value
            try:
                val = float(parts[-1])
                values.append(val)
            except ValueError:
                continue
    return np.array(values, dtype=np.float64) if values else np.array([], dtype=np.float64)
