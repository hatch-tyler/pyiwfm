"""Tests for the FE-interpolation kernel in calibration._kernels.

Verifies numerical equivalence between the pure-numpy fallback and the
Numba JIT path (when numba is available). The test runs with both
backends so the fast-calib install matrix is exercised.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyiwfm.calibration._kernels import (
    HAS_NUMBA,
    _interp_fe_frame_numpy,
    get_engine_name,
    interp_fe_frame,
)


def _make_synthetic_workload(
    n_nodes: int = 200,
    n_layers: int = 4,
    n_specs: int = 10,
    fe_size: int = 4,
    seed: int = 42,
):
    """Build a fixed-seed synthetic workload for kernel testing."""
    rng = np.random.default_rng(seed)
    frame = rng.uniform(50.0, 100.0, size=(n_nodes, n_layers))
    node_indices_flat = rng.choice(n_nodes, size=n_specs * fe_size).astype(np.int64)
    coeffs_flat = rng.dirichlet(np.ones(fe_size), size=n_specs).flatten()
    spec_offsets = (np.arange(n_specs + 1) * fe_size).astype(np.int64)
    return frame, node_indices_flat, coeffs_flat, spec_offsets


def test_pure_numpy_kernel_basic() -> None:
    """Pure-numpy kernel produces FE-interpolated values matching the
    by-hand `coeffs @ frame[node_indices, :]` computation."""
    frame, node_idx, coeffs, offsets = _make_synthetic_workload()
    n_specs = len(offsets) - 1
    n_layers = frame.shape[1]
    output = np.full((n_specs, n_layers), np.nan)

    _interp_fe_frame_numpy(frame, node_idx, coeffs, offsets, output)

    # Spot-check a couple specs by hand.
    for s in [0, 5, 9]:
        start, end = int(offsets[s]), int(offsets[s + 1])
        expected = coeffs[start:end] @ frame[node_idx[start:end], :]
        np.testing.assert_allclose(output[s], expected, rtol=1e-12)


def test_kernel_propagates_nan_per_layer() -> None:
    """A NaN in any input node should make the corresponding output
    layer NaN — and only that layer; other layers remain valid."""
    frame, node_idx, coeffs, offsets = _make_synthetic_workload()
    n_specs = len(offsets) - 1
    n_layers = frame.shape[1]

    # Inject a NaN into one input node, layer 1, used by spec 3.
    s = 3
    start = int(offsets[s])
    bad_node = int(node_idx[start])
    frame[bad_node, 1] = np.nan

    output = np.full((n_specs, n_layers), np.nan)
    interp_fe_frame(frame, node_idx, coeffs, offsets, output)

    # Spec 3, layer 1 is NaN.
    assert np.isnan(output[3, 1])
    # Other layers of spec 3 remain valid (other nodes unaffected).
    assert not np.isnan(output[3, 0])
    assert not np.isnan(output[3, 2])
    assert not np.isnan(output[3, 3])


@pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
def test_numba_matches_numpy() -> None:
    """When both backends are available, they must produce
    bit-identical (or float64-round-off-close) output for the same
    workload — that's the contract for `interp_fe_frame` callers."""
    frame, node_idx, coeffs, offsets = _make_synthetic_workload()
    n_specs = len(offsets) - 1
    n_layers = frame.shape[1]

    # Inject some NaN to exercise the per-layer NaN-propagation branch.
    rng = np.random.default_rng(7)
    nan_nodes = rng.choice(frame.shape[0], size=20, replace=False)
    nan_layers = rng.choice(n_layers, size=20)
    for nn, nl in zip(nan_nodes, nan_layers, strict=True):
        frame[nn, nl] = np.nan

    out_numpy = np.full((n_specs, n_layers), np.nan)
    _interp_fe_frame_numpy(frame, node_idx, coeffs, offsets, out_numpy)

    out_numba = np.full((n_specs, n_layers), np.nan)
    interp_fe_frame(frame, node_idx, coeffs, offsets, out_numba)

    # NaN-aware close: same NaN pattern AND close non-NaN values.
    np.testing.assert_array_equal(np.isnan(out_numpy), np.isnan(out_numba))
    np.testing.assert_allclose(
        out_numpy[~np.isnan(out_numpy)],
        out_numba[~np.isnan(out_numba)],
        rtol=1e-13,
    )


def test_engine_name_reports_active_backend() -> None:
    name = get_engine_name()
    assert name in {"numpy", "numba"}
    if HAS_NUMBA:
        assert name == "numba"
    else:
        assert name == "numpy"
