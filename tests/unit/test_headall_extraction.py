"""Tests for pyiwfm.calibration.headall_extraction."""

from __future__ import annotations

import numpy as np
import pytest

from pyiwfm.calibration.headall_extraction import (
    ExtractionResult,
    WellSpec,
)


class TestWellSpec:
    """Test WellSpec dataclass."""

    def test_defaults(self) -> None:
        w = WellSpec(name="W1", x=100.0, y=200.0)
        assert w.layer == -1
        assert w.element == 0
        assert np.isnan(w.bos)
        assert np.isnan(w.tos)

    def test_custom(self) -> None:
        w = WellSpec(name="W1", x=100.0, y=200.0, layer=2, element=5, bos=-50.0, tos=-10.0)
        assert w.layer == 2
        assert w.element == 5
        assert w.bos == -50.0
        assert w.tos == -10.0


class TestExtractionResult:
    """Test ExtractionResult dataclass."""

    def test_empty(self) -> None:
        times = np.array([], dtype="datetime64[s]")
        r = ExtractionResult(times=times)
        assert len(r.well_names) == 0
        assert len(r.per_layer_heads) == 0
        assert len(r.multi_layer_heads) == 0

    def test_with_data(self) -> None:
        times = np.array(
            [np.datetime64("2020-01-31"), np.datetime64("2020-02-29")],
            dtype="datetime64[D]",
        )
        r = ExtractionResult(times=times, well_names=["W1"])
        r.per_layer_heads["W1"] = np.array([[100.0, 90.0], [101.0, 91.0]])
        r.multi_layer_heads["W1"] = np.array([95.0, 96.0])

        assert len(r.times) == 2
        assert r.per_layer_heads["W1"].shape == (2, 2)
        assert r.multi_layer_heads["W1"][0] == pytest.approx(95.0)
