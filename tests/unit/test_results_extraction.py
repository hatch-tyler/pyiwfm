"""Tests for pyiwfm.calibration.results_extraction."""

from __future__ import annotations

import numpy as np

from pyiwfm.calibration.results_extraction import ExtractionSpec


class TestExtractionSpec:
    """Test ExtractionSpec dataclass."""

    def test_basic(self) -> None:
        spec = ExtractionSpec(
            name="W1",
            x=100.0,
            y=200.0,
            element=5,
            bos=-50.0,
            tos=-10.0,
        )
        assert spec.name == "W1"
        assert spec.x == 100.0
        assert spec.element == 5

    def test_defaults(self) -> None:
        spec = ExtractionSpec(name="W1", x=0.0, y=0.0)
        assert spec.layer == 0
        assert spec.element == 0
        assert np.isnan(spec.bos)
        assert np.isnan(spec.tos)
