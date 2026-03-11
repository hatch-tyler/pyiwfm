"""Tests for pyiwfm.calibration.texture2par dataclasses and helpers."""

from __future__ import annotations

import numpy as np
import pytest

from pyiwfm.calibration.texture2par import (
    MixingConfig,
    PilotPointSet,
    WellLog,
)


class TestWellLog:
    """Test WellLog dataclass."""

    def test_creation(self) -> None:
        wl = WellLog(
            name="Well_A",
            well_id=1,
            x=100.0,
            y=200.0,
            z_land=50.0,
            points=[(0.0, 1.0), (10.0, 0.0), (20.0, 1.0)],
        )
        assert wl.name == "Well_A"
        assert wl.well_id == 1
        assert len(wl.points) == 3
        assert wl.points[0] == (0.0, 1.0)

    def test_empty_points(self) -> None:
        wl = WellLog(name="W", well_id=0, x=0, y=0, z_land=0, points=[])
        assert len(wl.points) == 0


class TestPilotPointSet:
    """Test PilotPointSet dataclass."""

    def test_creation(self) -> None:
        pp = PilotPointSet(
            x=np.array([0.0, 100.0]),
            y=np.array([0.0, 100.0]),
            values={"KCMin": np.array([1.0, 2.0])},
        )
        assert len(pp.x) == 2
        assert "KCMin" in pp.values
        assert pp.zones is None

    def test_with_zones(self) -> None:
        pp = PilotPointSet(
            x=np.array([0.0]),
            y=np.array([0.0]),
            values={"KCMin": np.array([1.0])},
            zones=np.array([1]),
        )
        assert pp.zones is not None
        assert pp.zones[0] == 1


class TestMixingConfig:
    """Test MixingConfig dataclass."""

    def test_defaults(self) -> None:
        mc = MixingConfig()
        assert mc.kh_power == pytest.approx(0.93)
        assert mc.kv_power == pytest.approx(-0.62)
        assert mc.sy_power == pytest.approx(1.0)

    def test_custom(self) -> None:
        mc = MixingConfig(kh_power=1.0, kv_power=-1.0)
        assert mc.kh_power == pytest.approx(1.0)
        assert mc.kv_power == pytest.approx(-1.0)
        # Others should keep defaults
        assert mc.sy_power == pytest.approx(1.0)
