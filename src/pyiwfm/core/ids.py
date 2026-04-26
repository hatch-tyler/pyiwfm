"""1-based IWFM ID ↔ 0-based array index helpers.

IWFM uses 1-based IDs for nodes, elements, reaches, lakes, etc. (Fortran
convention). Most internal Python data structures use 0-based array indices.
The conversion is repetitive and error-prone — `idx = element_id - 1` appears
in many places without bounds checks, so an out-of-range ID can silently
write past the end of an array or fall through to a wrong row.

This module centralizes the conversion with mandatory bounds checking. Use
``to_index`` for a single ID and ``to_indices`` for a NumPy array of IDs.
The ``kind`` parameter shows up in the error message ("element 0 is out of
range [1, 12345]") to make failures self-documenting.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


def to_index(one_based_id: int, n_items: int, *, kind: str = "id") -> int:
    """Convert a 1-based IWFM ID to a 0-based array index, validating bounds.

    Parameters
    ----------
    one_based_id : int
        The IWFM ID (1-based, Fortran convention).
    n_items : int
        Number of items in the target array. Valid range is ``[1, n_items]``.
    kind : str, optional
        Label used in error messages, e.g. ``"element"``, ``"node"``,
        ``"reach"``. Default is ``"id"``.

    Returns
    -------
    int
        0-based array index (i.e. ``one_based_id - 1``).

    Raises
    ------
    ValueError
        If ``one_based_id`` is not in the closed interval ``[1, n_items]``.
    """
    if not 1 <= one_based_id <= n_items:
        raise ValueError(f"{kind} {one_based_id!r} is out of range [1, {n_items}]")
    return one_based_id - 1


def to_indices(one_based_ids: ArrayLike, n_items: int, *, kind: str = "id") -> NDArray[np.int64]:
    """Vectorized variant of :func:`to_index` for arrays of IDs.

    Parameters
    ----------
    one_based_ids : array-like
        1-based IDs. Will be cast to ``int64``.
    n_items : int
        Number of items in the target array.
    kind : str, optional
        Label used in error messages.

    Returns
    -------
    numpy.ndarray
        0-based indices, dtype ``int64``, same shape as the input.

    Raises
    ------
    ValueError
        If any element is outside ``[1, n_items]``. The error message lists
        up to the first 5 offending values.
    """
    arr = np.asarray(one_based_ids, dtype=np.int64)
    if arr.size:
        bad_mask = (arr < 1) | (arr > n_items)
        if bad_mask.any():
            bad = arr[bad_mask][:5].tolist()
            extra = "" if bad_mask.sum() <= 5 else f" (+{int(bad_mask.sum()) - 5} more)"
            raise ValueError(f"{kind} value(s) {bad!r}{extra} out of range [1, {n_items}]")
    return arr - 1


__all__ = ["to_index", "to_indices"]
