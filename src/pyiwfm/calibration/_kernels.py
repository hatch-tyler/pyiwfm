"""JIT-accelerated FE-interpolation kernels for the calibration pipeline.

When the optional ``numba`` dependency is installed
(``pip install pyiwfm[fast-calib]``), the kernel below is JIT-compiled
to native code on first call and cached in ``~/.numba/`` for
subsequent runs. First-call warmup is ~80ms; cached calls are ~1μs.

Without ``numba`` (the default), the same kernel runs as pure-numpy
inner-loop code that's algorithmically equivalent (~3× per-frame
speedup over the v1.x triple-nested-Python-loop baseline, see
``docs/user_guide/calibration.rst`` § Performance). Numba on top
adds another ~10-50× on the heavy 31k-location subsidence
workload because it eliminates the per-spec Python loop overhead
that pure-numpy can't avoid.

Three engines are available for ``ResultsExtractor.extract``:

1. **Pure-Python (default)** — ships with every install, vectorised
   across layers per spec but with Python overhead in the per-spec
   loop. Fast enough for the typical 10-100-location workload.

2. **Numba** (``pip install pyiwfm[fast-calib]``) — JIT-compiled
   inner kernel; closes most of the gap to the Fortran reference
   implementation. The right default for >1k-location workloads.

3. **Fortran subprocess** (``ResultsExtract.exe`` on PATH) — black-box
   reference implementation, ships with IWFM. Use when the .exe is
   already installed and the workload is >10k locations.

This module's public ``interp_fe_frame`` is the kernel-of-record;
callers don't need to know which backend is active. The active
backend is reported via :data:`HAS_NUMBA`.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


try:
    from numba import njit  # type: ignore[import-not-found,import-untyped]

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


# =============================================================================
# Pure-numpy implementation (always available)
# =============================================================================


def _interp_fe_frame_numpy(
    frame: NDArray[np.float64],
    node_indices_flat: NDArray[np.int64],
    coeffs_flat: NDArray[np.float64],
    spec_offsets: NDArray[np.int64],
    output: NDArray[np.float64],
) -> None:
    """Pure-numpy FE interpolation kernel.

    For each spec ``s``, computes ``coeffs[s] @ frame[node_indices[s], :]``
    over the FE nodes referenced by that spec. NaN propagation: any NaN
    in any input node for a layer leaves the output for that layer NaN.

    Uses a Python for-loop over specs but vectorises across layers via
    a matrix-vector product per spec. Algorithmically identical to the
    inner triple loop of v1.x, factor of ~3 faster from layer
    vectorisation alone.

    Parameters
    ----------
    frame : NDArray, shape ``(n_nodes, n_layers)``
        One timestep slice of the all-node HDF5 output.
    node_indices_flat : NDArray[int64]
        Flat array of FE-node indices (0-based) across all specs.
        Concatenated by spec.
    coeffs_flat : NDArray[float64]
        Flat array of FE-interpolation coefficients across all specs,
        same layout as ``node_indices_flat``.
    spec_offsets : NDArray[int64], shape ``(n_specs+1,)``
        CSR-style offsets: spec ``s`` uses indices
        ``spec_offsets[s]:spec_offsets[s+1]`` of the flat arrays.
    output : NDArray, shape ``(n_specs, n_layers)``
        Pre-allocated output buffer; written in-place. Caller is
        responsible for any pre-fill (e.g. ``np.full(..., np.nan)``).
    """
    n_specs = len(spec_offsets) - 1
    for s in range(n_specs):
        start = int(spec_offsets[s])
        end = int(spec_offsets[s + 1])
        node_indices = node_indices_flat[start:end]
        coeffs = coeffs_flat[start:end]
        sub = frame[node_indices, :]  # (n_fe_nodes, n_layers)
        nan_per_layer = np.any(np.isnan(sub), axis=0)
        interpolated = coeffs @ sub
        output[s, :] = np.where(nan_per_layer, np.nan, interpolated)


# =============================================================================
# Numba JIT implementation (opt-in via fast-calib extra)
# =============================================================================


if HAS_NUMBA:

    @njit(cache=True, fastmath=False)
    def _interp_fe_frame_numba(
        frame: NDArray[np.float64],
        node_indices_flat: NDArray[np.int64],
        coeffs_flat: NDArray[np.float64],
        spec_offsets: NDArray[np.int64],
        output: NDArray[np.float64],
    ) -> None:
        """Numba JIT-compiled FE interpolation kernel.

        Same contract as :func:`_interp_fe_frame_numpy`. The triple
        nested loop runs in compiled C: per-spec, per-layer,
        per-FE-node. NaN check breaks early to skip the rest of the
        accumulation.

        First call triggers JIT compile (~80ms warmup). Subsequent
        calls are cached and run in ~1μs per (spec × layer × fe_node)
        on typical hardware. ``fastmath=False`` is intentional —
        ``fastmath=True`` would let the compiler reorder NaN-producing
        operations and break the bit-identical-to-pure-Python contract.
        """
        n_specs = len(spec_offsets) - 1
        n_layers = frame.shape[1]
        for s in range(n_specs):
            start = spec_offsets[s]
            end = spec_offsets[s + 1]
            for layer in range(n_layers):
                val = 0.0
                has_nan = False
                for i in range(start, end):
                    node_val = frame[node_indices_flat[i], layer]
                    if np.isnan(node_val):
                        has_nan = True
                        break
                    val += coeffs_flat[i] * node_val
                output[s, layer] = np.nan if has_nan else val

    interp_fe_frame = _interp_fe_frame_numba
else:
    interp_fe_frame = _interp_fe_frame_numpy


def get_engine_name() -> str:
    """Return the active backend name for diagnostics / logging."""
    return "numba" if HAS_NUMBA else "numpy"


def log_engine_status() -> None:
    """Log the active backend at INFO level. Called once at import time
    of any module that uses the kernel; safe to call repeatedly."""
    if HAS_NUMBA:
        logger.info("calibration kernel: Numba JIT enabled (pyiwfm[fast-calib] installed)")
    else:
        logger.info(
            "calibration kernel: pure-numpy (default). For >10k-location "
            "workloads consider `pip install pyiwfm[fast-calib]` (Numba JIT) "
            "or the Fortran ResultsExtract.exe backend."
        )
