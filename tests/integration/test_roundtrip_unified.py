"""Unified roundtrip tests for IWFM models.

Tests the full pipeline: read -> write -> run -> verify for both the
IWFM Sample Model and C2VSimCG. Each test can be run independently.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pyiwfm.roundtrip.config import RoundtripConfig

# ---------------------------------------------------------------------------
# Shared helpers — one per test assertion
# ---------------------------------------------------------------------------


def _assert_load(config: RoundtripConfig) -> None:
    """Model loads without errors."""
    from pyiwfm.roundtrip.pipeline import RoundtripPipeline

    pipeline = RoundtripPipeline(config)
    result = pipeline.step_load()
    assert result.success, f"Load failed: {result.error}"
    assert result.data is not None


def _assert_write(config: RoundtripConfig) -> None:
    """All expected files are written."""
    from pyiwfm.roundtrip.pipeline import RoundtripPipeline

    pipeline = RoundtripPipeline(config)
    load = pipeline.step_load()
    assert load.success, f"Load failed: {load.error}"

    write = pipeline.step_write()
    assert write.success, f"Write failed: {write.error}"
    assert write.data is not None
    assert len(write.data.files) > 0


def _assert_input_files_match(config: RoundtripConfig) -> None:
    """Written data matches originals (ignoring comments)."""
    from pyiwfm.roundtrip.pipeline import RoundtripPipeline

    pipeline = RoundtripPipeline(config)
    pipeline.step_load()
    pipeline.step_write()
    diff = pipeline.step_diff_inputs()

    assert diff.success, f"Diff failed: {diff.error}"
    if diff.data:
        assert diff.data.files_data_identical == diff.data.files_compared, (
            f"Data differences found:\n{diff.data.summary()}"
        )


def _assert_preprocessor_runs(config: RoundtripConfig) -> None:
    """PreProcessor succeeds on written model."""
    from pyiwfm.roundtrip.pipeline import RoundtripPipeline

    config.run_baseline = False
    config.compare_results = False
    pipeline = RoundtripPipeline(config)
    pipeline.step_load()
    pipeline.step_write()
    pipeline.step_place_executables()

    pipeline.step_run_written()
    if pipeline.result.written_run:
        pp = pipeline.result.written_run.preprocessor
        if pp is not None:
            assert pp.success, (
                f"PP failed (rc={pp.return_code}): {pp.errors[:5]}\nstdout: {pp.stdout[:500]}"
            )


def _assert_simulation_runs(config: RoundtripConfig) -> None:
    """Simulation succeeds on written model."""
    from pyiwfm.roundtrip.pipeline import RoundtripPipeline

    config.run_baseline = False
    config.compare_results = False
    pipeline = RoundtripPipeline(config)
    pipeline.step_load()
    pipeline.step_write()
    pipeline.step_place_executables()

    pipeline.step_run_written()
    if pipeline.result.written_run:
        sim = pipeline.result.written_run.simulation
        if sim is not None:
            assert sim.success, (
                f"Sim failed (rc={sim.return_code}): {sim.errors[:5]}\nstdout: {sim.stdout[:500]}"
            )


def _assert_results_match(config: RoundtripConfig) -> None:
    """Outputs match baseline within tolerance."""
    from pyiwfm.roundtrip.pipeline import RoundtripPipeline

    pipeline = RoundtripPipeline(config)
    result = pipeline.run()

    if result.results_comparison:
        assert result.results_comparison.success, (
            f"Results differ:\n{result.results_comparison.summary()}"
        )


def _assert_full_pipeline(config: RoundtripConfig) -> None:
    """End-to-end: pipeline.run() succeeds."""
    from pyiwfm.roundtrip.pipeline import RoundtripPipeline

    pipeline = RoundtripPipeline(config)
    result = pipeline.run()

    assert result.success, f"Pipeline failed:\n{result.summary()}"


# ---------------------------------------------------------------------------
# Sample Model
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.roundtrip
class TestRoundtripSampleModel:
    """Roundtrip tests for the IWFM Sample Model."""

    @pytest.fixture(autouse=True)
    def setup(self, sample_model_path: Path, tmp_path: Path) -> None:
        """Set up config for sample model tests."""
        from pyiwfm.roundtrip.config import RoundtripConfig

        self.config = RoundtripConfig.for_sample_model(sample_model_path)
        self.config.output_dir = tmp_path / "roundtrip_sample"

    def test_load_model(self) -> None:
        """Model loads without errors."""
        _assert_load(self.config)

    def test_write_model(self) -> None:
        """All expected files are written."""
        _assert_write(self.config)

    def test_input_files_match(self) -> None:
        """Written data matches originals (ignoring comments)."""
        _assert_input_files_match(self.config)

    def test_preprocessor_runs(self) -> None:
        """PreProcessor succeeds on written model."""
        _assert_preprocessor_runs(self.config)

    def test_simulation_runs(self) -> None:
        """Simulation succeeds on written model."""
        _assert_simulation_runs(self.config)

    def test_results_match(self) -> None:
        """Outputs match baseline within tolerance."""
        _assert_results_match(self.config)

    def test_full_pipeline(self) -> None:
        """End-to-end: pipeline.run() succeeds."""
        _assert_full_pipeline(self.config)


# ---------------------------------------------------------------------------
# C2VSimCG
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.roundtrip
class TestRoundtripC2VSimCG:
    """Roundtrip tests for the C2VSimCG (Coarse Grid) model."""

    @pytest.fixture(autouse=True)
    def setup(self, c2vsimcg_path: Path, tmp_path: Path) -> None:
        """Set up config for C2VSimCG tests."""
        from pyiwfm.roundtrip.config import RoundtripConfig

        self.config = RoundtripConfig.for_c2vsimcg(c2vsimcg_path)
        self.config.output_dir = tmp_path / "roundtrip_c2vsimcg"

    def test_load_model(self) -> None:
        """Model loads without errors."""
        _assert_load(self.config)

    def test_write_model(self) -> None:
        """All expected files are written."""
        _assert_write(self.config)

    def test_input_files_match(self) -> None:
        """Written data matches originals (ignoring comments)."""
        _assert_input_files_match(self.config)

    def test_preprocessor_runs(self) -> None:
        """PreProcessor succeeds on written model."""
        _assert_preprocessor_runs(self.config)

    def test_simulation_runs(self) -> None:
        """Simulation succeeds on written model."""
        _assert_simulation_runs(self.config)

    def test_results_match(self) -> None:
        """Outputs match baseline within tolerance."""
        _assert_results_match(self.config)

    def test_full_pipeline(self) -> None:
        """End-to-end: pipeline.run() succeeds."""
        _assert_full_pipeline(self.config)
