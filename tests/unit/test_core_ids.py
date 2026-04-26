"""Tests for ``pyiwfm.core.ids`` — 1-based ID → 0-based index helpers."""

from __future__ import annotations

import numpy as np
import pytest

from pyiwfm.core.ids import to_index, to_indices


class TestToIndex:
    def test_valid_range(self):
        assert to_index(1, 10) == 0
        assert to_index(10, 10) == 9
        assert to_index(5, 10) == 4

    def test_zero_raises(self):
        with pytest.raises(ValueError, match=r"id 0 is out of range \[1, 10\]"):
            to_index(0, 10)

    def test_negative_raises(self):
        with pytest.raises(ValueError, match=r"id -3 is out of range"):
            to_index(-3, 10)

    def test_above_n_raises(self):
        with pytest.raises(ValueError, match=r"id 11 is out of range \[1, 10\]"):
            to_index(11, 10)

    def test_kind_label_in_message(self):
        with pytest.raises(ValueError, match="element"):
            to_index(0, 5, kind="element")
        with pytest.raises(ValueError, match="node"):
            to_index(99, 5, kind="node")
        with pytest.raises(ValueError, match="reach"):
            to_index(-1, 5, kind="reach")

    def test_n_items_zero(self):
        # No items at all — every ID is invalid
        with pytest.raises(ValueError):
            to_index(1, 0)


class TestToIndices:
    def test_valid_array(self):
        result = to_indices([1, 5, 10], 10, kind="element")
        assert result.tolist() == [0, 4, 9]
        assert result.dtype == np.int64

    def test_numpy_array_input(self):
        ids = np.array([3, 7], dtype=np.int32)
        result = to_indices(ids, 10)
        assert result.tolist() == [2, 6]

    def test_empty_array(self):
        result = to_indices([], 10)
        assert result.size == 0
        assert result.dtype == np.int64

    def test_out_of_range_lists_offenders(self):
        with pytest.raises(ValueError, match=r"\[0, 11\]"):
            to_indices([0, 5, 11], 10)

    def test_truncates_long_offender_list(self):
        # 6 bad values; message should show up to 5 plus "+1 more"
        with pytest.raises(ValueError, match=r"\+1 more"):
            to_indices([0, -1, -2, -3, -4, -5], 10)

    def test_kind_in_message(self):
        with pytest.raises(ValueError, match="reach"):
            to_indices([99], 10, kind="reach")

    def test_preserves_shape(self):
        ids = np.array([[1, 2], [3, 4]])
        result = to_indices(ids, 10)
        assert result.shape == (2, 2)
        assert result.tolist() == [[0, 1], [2, 3]]
