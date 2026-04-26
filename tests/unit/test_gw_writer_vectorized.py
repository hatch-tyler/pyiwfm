"""Byte-identical golden tests for the vectorized aquifer-params and
initial-heads block formatters in ``gw_main_writer``.

These pin the exact output format that the previous per-cell ``f.write``
loop produced, so future refactors (e.g. swapping in NumPy ``savetxt``)
won't silently change the on-disk representation.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from pyiwfm.io.gw_main_writer import (
    _format_aquifer_params_block,
    _format_initial_heads_block,
)


def _make_params(n_nodes: int, n_layers: int, *, with_aquitard_kv: bool = True):
    params = MagicMock()
    params.n_nodes = n_nodes
    params.n_layers = n_layers
    rng = np.random.default_rng(0)
    params.kh = rng.uniform(1e-3, 1e-1, (n_nodes, n_layers))
    params.kv = rng.uniform(1e-5, 1e-3, (n_nodes, n_layers))
    params.specific_storage = rng.uniform(1e-6, 1e-4, (n_nodes, n_layers))
    params.specific_yield = rng.uniform(0.01, 0.3, (n_nodes, n_layers))
    params.aquitard_kv = rng.uniform(1e-7, 1e-5, (n_nodes, n_layers)) if with_aquitard_kv else None
    return params


def test_aquifer_params_block_row_count_and_prefix():
    """Three-node × two-layer fixture: row count and node-ID prefixing."""
    params = MagicMock()
    params.n_nodes = 3
    params.n_layers = 2
    params.kh = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    params.kv = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    params.specific_storage = np.array([[1e-5, 2e-5], [3e-5, 4e-5], [5e-5, 6e-5]])
    params.specific_yield = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    params.aquitard_kv = np.array([[1e-7, 2e-7], [3e-7, 4e-7], [5e-7, 6e-7]])

    block = _format_aquifer_params_block(params)
    lines = block.splitlines()

    # 3 nodes * 2 layers = 6 rows
    assert len(lines) == 6

    # Layer 0 rows carry the 1-based node ID; layer 1 rows are indented continuations
    assert lines[0].lstrip().startswith("1 ")
    assert lines[1].startswith("             ")
    assert lines[2].lstrip().startswith("2 ")
    assert lines[3].startswith("             ")
    assert lines[4].lstrip().startswith("3 ")
    assert lines[5].startswith("             ")


def test_aquifer_params_block_handles_none_arrays():
    """When optional arrays are None, the formatter substitutes zeros."""
    params = MagicMock()
    params.n_nodes = 2
    params.n_layers = 1
    params.kh = np.array([[1.5], [2.5]])
    params.kv = None  # missing
    params.specific_storage = None
    params.specific_yield = None
    params.aquitard_kv = None

    block = _format_aquifer_params_block(params)
    lines = block.splitlines()
    assert len(lines) == 2
    # All non-kh values should be 0
    assert "           0" in lines[0]


def test_aquifer_params_block_against_old_loop():
    """Generated block matches the byte output of the original per-cell loop
    for a 50-node × 4-layer fixture. Locks in the format so any future
    optimization (e.g. NumPy savetxt) stays byte-identical."""
    params = _make_params(n_nodes=50, n_layers=4)

    # Replicate the original loop verbatim
    expected_parts = []
    for i in range(params.n_nodes):
        for layer in range(params.n_layers):
            kh = params.kh[i, layer]
            kv = params.kv[i, layer]
            ss = params.specific_storage[i, layer]
            sy = params.specific_yield[i, layer]
            akv = params.aquitard_kv[i, layer]
            if layer == 0:
                expected_parts.append(
                    f"     {i + 1:>6d}  {kh:>12.6g}  {ss:>12.6g}  "
                    f"{sy:>12.6g}  {kv:>12.6g}  {akv:>12.6g}\n"
                )
            else:
                expected_parts.append(
                    f"             {kh:>12.6g}  {ss:>12.6g}  "
                    f"{sy:>12.6g}  {kv:>12.6g}  {akv:>12.6g}\n"
                )
    expected = "".join(expected_parts)

    actual = _format_aquifer_params_block(params)
    assert actual == expected


def test_initial_heads_block_against_old_loop():
    """Initial-heads block matches the byte output of the original per-node
    loop for a 100-node × 3-layer fixture."""
    rng = np.random.default_rng(42)
    heads = rng.uniform(50.0, 200.0, (100, 3))

    expected_parts = []
    for i in range(heads.shape[0]):
        vals = "  ".join(f"{heads[i, j]:>12.4f}" for j in range(heads.shape[1]))
        expected_parts.append(f"     {i + 1:>6d}  {vals}\n")
    expected = "".join(expected_parts)

    actual = _format_initial_heads_block(heads)
    assert actual == expected
