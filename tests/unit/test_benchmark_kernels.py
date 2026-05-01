"""Benchmarks for the calibration FE-interpolation kernel.

Two cases:

- ``test_benchmark_kernel_typical`` — typical PEST-iteration workload
  (100 locations × 4 layers × 4 FE nodes per spec × 365 timesteps).
  Always runs.

- ``test_benchmark_kernel_heavy`` — InSAR-pixel-style workload
  (5,000 locations × same shape per spec). Runs both backends so
  the speedup ratio is visible in CI output.

Both compare the active ``interp_fe_frame`` against the pure-numpy
fallback. When numba isn't installed they're identical and the
benchmark just measures variance.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyiwfm.calibration._kernels import (
    HAS_NUMBA,
    _interp_fe_frame_numpy,
    interp_fe_frame,
)


def _make_workload(n_specs: int, fe_size: int = 4, n_nodes: int = 50_000):
    rng = np.random.default_rng(42)
    frame = rng.uniform(50.0, 100.0, size=(n_nodes, 4)).astype(np.float64)
    node_idx = rng.choice(n_nodes, size=n_specs * fe_size).astype(np.int64)
    coeffs = rng.dirichlet(np.ones(fe_size), size=n_specs).flatten()
    offsets = (np.arange(n_specs + 1) * fe_size).astype(np.int64)
    output = np.empty((n_specs, 4), dtype=np.float64)
    return frame, node_idx, coeffs, offsets, output


@pytest.mark.benchmark(group="kernel-typical")
def test_benchmark_kernel_typical(benchmark) -> None:
    """100 locations × 4 layers per frame — typical PEST-iteration call.

    With ``pyiwfm[fast-calib]`` installed, the active kernel should
    beat the pure-numpy fallback by ~2-5× per frame on this size.
    """
    frame, node_idx, coeffs, offsets, output = _make_workload(n_specs=100)

    # Warm up the JIT (compile + cache writeback) before timing.
    if HAS_NUMBA:
        interp_fe_frame(frame, node_idx, coeffs, offsets, output)

    benchmark(interp_fe_frame, frame, node_idx, coeffs, offsets, output)


@pytest.mark.benchmark(group="kernel-heavy")
def test_benchmark_kernel_heavy(benchmark) -> None:
    """5,000 locations — approximates an InSAR subsidence workload.

    The original triple-Python-loop implementation took seconds per
    frame at this scale; pure-numpy is ~3× faster than that;
    Numba on top is another ~10-30× from eliminating the per-spec
    Python loop overhead.
    """
    frame, node_idx, coeffs, offsets, output = _make_workload(n_specs=5_000)

    if HAS_NUMBA:
        interp_fe_frame(frame, node_idx, coeffs, offsets, output)

    benchmark(interp_fe_frame, frame, node_idx, coeffs, offsets, output)


@pytest.mark.benchmark(group="kernel-heavy-numpy-baseline")
def test_benchmark_kernel_heavy_numpy_baseline(benchmark) -> None:
    """Pure-numpy baseline for the 5,000-location workload.

    Always runs (no numba dependency). Compare against
    ``test_benchmark_kernel_heavy`` to measure the Numba speedup
    when the fast-calib extra is installed.
    """
    frame, node_idx, coeffs, offsets, output = _make_workload(n_specs=5_000)
    benchmark(_interp_fe_frame_numpy, frame, node_idx, coeffs, offsets, output)
