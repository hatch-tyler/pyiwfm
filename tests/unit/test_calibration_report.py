"""Unit tests for calibration report generator."""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from pyiwfm.calibration.report import (
    CalibrationReportConfig,
    generate_calibration_report,
)


def _sample_residuals(n_wells: int = 4, n_timesteps: int = 10) -> pd.DataFrame:
    """Create a synthetic residuals DataFrame for testing."""
    rng = np.random.default_rng(42)
    rows = []
    for w in range(n_wells):
        for t in range(n_timesteps):
            obs = 100.0 + w * 50.0 + rng.normal(0, 2)
            sim = obs + rng.normal(0, 3)
            rows.append(
                {
                    "well_id": f"W{w}",
                    "datetime": pd.Timestamp("2020-01-01") + pd.Timedelta(days=30 * t),
                    "observed": obs,
                    "simulated": sim,
                    "residual": sim - obs,
                    "layer": (w % 2) + 1,
                    "subregion": (w // 2) + 1,
                    "screen_type": "known_screens" if w < 2 else "unknown",
                    "x": 1000.0 + w * 100.0,
                    "y": 2000.0 + w * 100.0,
                }
            )
    return pd.DataFrame(rows)


class TestCalibrationReportConfig:
    """Tests for CalibrationReportConfig dataclass."""

    def test_defaults(self) -> None:
        cfg = CalibrationReportConfig()
        assert cfg.filter_by_subregion is True
        assert cfg.filter_by_layer is True
        assert cfg.include_one_to_one is True
        assert cfg.include_cdf is True
        assert cfg.include_spatial_bias is True
        assert cfg.units == ""
        assert cfg.dpi == 300

    def test_custom(self) -> None:
        cfg = CalibrationReportConfig(
            filter_by_subregion=False,
            include_spatial_bias=False,
        )
        assert cfg.filter_by_subregion is False
        assert cfg.include_spatial_bias is False

    def test_units_and_dpi(self) -> None:
        cfg = CalibrationReportConfig(units="ft", dpi=150)
        assert cfg.units == "ft"
        assert cfg.dpi == 150


class TestGenerateCalibrationReport:
    """Tests for generate_calibration_report."""

    def test_generates_pdf(self) -> None:
        """Test that a PDF file is written."""
        df = _sample_residuals()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = generate_calibration_report(df, Path(tmpdir) / "report.pdf")
            assert out.exists()
            assert out.stat().st_size > 0

    def test_with_default_config(self) -> None:
        """Test default config generates pages for subregions, layers, screen types."""
        df = _sample_residuals()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = generate_calibration_report(df, Path(tmpdir) / "report.pdf")
            assert out.exists()

    def test_minimal_config(self) -> None:
        """Test with all filters and extras disabled."""
        df = _sample_residuals()
        cfg = CalibrationReportConfig(
            filter_by_subregion=False,
            filter_by_layer=False,
            filter_by_screen_type=False,
            include_spatial_bias=False,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out = generate_calibration_report(df, Path(tmpdir) / "report.pdf", config=cfg)
            assert out.exists()

    def test_only_one_to_one(self) -> None:
        """Test with only 1:1 plots."""
        df = _sample_residuals()
        cfg = CalibrationReportConfig(
            filter_by_subregion=False,
            filter_by_layer=False,
            filter_by_screen_type=False,
            include_cdf=False,
            include_spatial_bias=False,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out = generate_calibration_report(df, Path(tmpdir) / "report.pdf", config=cfg)
            assert out.exists()

    def test_only_cdf(self) -> None:
        """Test with only CDF plots."""
        df = _sample_residuals()
        cfg = CalibrationReportConfig(
            filter_by_subregion=False,
            filter_by_layer=False,
            filter_by_screen_type=False,
            include_one_to_one=False,
            include_spatial_bias=False,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out = generate_calibration_report(df, Path(tmpdir) / "report.pdf", config=cfg)
            assert out.exists()

    def test_with_date_range_filter(self) -> None:
        """Test date range filtering."""
        df = _sample_residuals()
        cfg = CalibrationReportConfig(
            filter_by_subregion=False,
            filter_by_layer=False,
            filter_by_screen_type=False,
            filter_by_date_range=[
                (datetime(2020, 1, 1), datetime(2020, 6, 30)),
                (datetime(2020, 7, 1), datetime(2020, 12, 31)),
            ],
            include_spatial_bias=False,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out = generate_calibration_report(df, Path(tmpdir) / "report.pdf", config=cfg)
            assert out.exists()

    def test_empty_dataframe(self) -> None:
        """Test with empty DataFrame produces a valid (empty) PDF."""
        df = pd.DataFrame(columns=["well_id", "datetime", "observed", "simulated", "residual"])
        with tempfile.TemporaryDirectory() as tmpdir:
            out = generate_calibration_report(df, Path(tmpdir) / "report.pdf")
            assert out.exists()

    def test_returns_path(self) -> None:
        """Test that the function returns the output Path."""
        df = _sample_residuals(n_wells=2, n_timesteps=3)
        with tempfile.TemporaryDirectory() as tmpdir:
            result = generate_calibration_report(df, Path(tmpdir) / "report.pdf")
            assert isinstance(result, Path)

    def test_with_units(self) -> None:
        """Test that units parameter is accepted and produces output."""
        df = _sample_residuals(n_wells=2, n_timesteps=5)
        cfg = CalibrationReportConfig(
            filter_by_subregion=False,
            filter_by_layer=False,
            filter_by_screen_type=False,
            include_spatial_bias=False,
            units="ft",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out = generate_calibration_report(df, Path(tmpdir) / "report.pdf", config=cfg)
            assert out.exists()
            assert out.stat().st_size > 0

    def test_custom_dpi(self) -> None:
        """Test that custom DPI is accepted."""
        df = _sample_residuals(n_wells=2, n_timesteps=3)
        cfg = CalibrationReportConfig(
            filter_by_subregion=False,
            filter_by_layer=False,
            filter_by_screen_type=False,
            include_spatial_bias=False,
            include_cdf=False,
            dpi=72,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out = generate_calibration_report(df, Path(tmpdir) / "report.pdf", config=cfg)
            assert out.exists()
