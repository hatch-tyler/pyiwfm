"""Tests for roundtrip/pipeline.py.

Covers:
- StepResult, RunPairResult, RoundtripResult: dataclass defaults, success, summary()
- step_load(): success, file not found, exception
- step_write(): success, model not loaded
- step_place_executables(), step_generate_scripts()
- step_diff_inputs(): success, written_dir not set
- step_run_baseline(), step_run_written(): skip, success
- step_compare_results(): skip, success
- run(): full pipeline, early abort
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pyiwfm.roundtrip.config import RoundtripConfig
from pyiwfm.roundtrip.pipeline import (
    RoundtripPipeline,
    RoundtripResult,
    RunPairResult,
    StepResult,
)


# ---------------------------------------------------------------------------
# Dataclass basics
# ---------------------------------------------------------------------------

class TestStepResult:
    def test_defaults(self) -> None:
        r = StepResult()
        assert r.name == ""
        assert r.success is False
        assert r.message == ""
        assert r.error == ""

    def test_fields(self) -> None:
        r = StepResult(name="load", success=True, message="OK")
        assert r.name == "load"
        assert r.success is True


class TestRunPairResult:
    def test_defaults(self) -> None:
        r = RunPairResult()
        assert r.preprocessor is None
        assert r.simulation is None

    def test_success_when_both_succeed(self) -> None:
        pp = MagicMock()
        pp.success = True
        sim = MagicMock()
        sim.success = True
        r = RunPairResult(preprocessor=pp, simulation=sim)
        assert r.success is True

    def test_failure_when_sim_fails(self) -> None:
        pp = MagicMock()
        pp.success = True
        sim = MagicMock()
        sim.success = False
        r = RunPairResult(preprocessor=pp, simulation=sim)
        assert r.success is False

    def test_failure_when_none(self) -> None:
        r = RunPairResult()
        assert r.success is False


class TestRoundtripResult:
    def test_defaults(self) -> None:
        r = RoundtripResult()
        assert r.steps == {}
        assert r.baseline_run is None

    def test_success_all_steps_pass(self) -> None:
        r = RoundtripResult(
            steps={
                "load": StepResult(name="load", success=True),
                "write": StepResult(name="write", success=True),
            }
        )
        assert r.success is True

    def test_failure_any_step_fails(self) -> None:
        r = RoundtripResult(
            steps={
                "load": StepResult(name="load", success=True),
                "write": StepResult(name="write", success=False),
            }
        )
        assert r.success is False

    def test_summary_format(self) -> None:
        r = RoundtripResult(
            steps={"load": StepResult(name="load", success=True, message="OK")}
        )
        s = r.summary()
        assert isinstance(s, str)
        assert len(s) > 0


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def _make_pipeline(tmp_path: Path, **overrides: object) -> RoundtripPipeline:
    """Create a pipeline with a minimal config."""
    model_dir = tmp_path / "model"
    model_dir.mkdir(exist_ok=True)
    sim_dir = model_dir / "Simulation"
    sim_dir.mkdir(exist_ok=True)
    (sim_dir / "Simulation.in").touch()
    pp_dir = model_dir / "Preprocessor"
    pp_dir.mkdir(exist_ok=True)
    (pp_dir / "Preprocessor.in").touch()

    defaults: dict = dict(
        source_model_dir=model_dir,
        simulation_main_file="Simulation/Simulation.in",
        preprocessor_main_file="Preprocessor/Preprocessor.in",
        output_dir=tmp_path / "output",
        run_baseline=False,
        run_written=False,
        compare_results=False,
    )
    defaults.update(overrides)
    cfg = RoundtripConfig(**defaults)
    return RoundtripPipeline(cfg)


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

class TestStepLoad:
    @patch("pyiwfm.core.model.IWFMModel")
    def test_success(self, mock_model_cls: MagicMock, tmp_path: Path) -> None:
        mock_model_cls.from_simulation_with_preprocessor.return_value = MagicMock()
        pipe = _make_pipeline(tmp_path)
        result = pipe.step_load()
        assert result.success

    @patch("pyiwfm.core.model.IWFMModel")
    def test_exception(self, mock_model_cls: MagicMock, tmp_path: Path) -> None:
        mock_model_cls.from_simulation_with_preprocessor.side_effect = ValueError(
            "bad"
        )
        pipe = _make_pipeline(tmp_path)
        result = pipe.step_load()
        assert not result.success

    def test_sim_file_not_found(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "model2"
        model_dir.mkdir()
        pp_dir = model_dir / "Preprocessor"
        pp_dir.mkdir()
        (pp_dir / "Preprocessor.in").touch()

        cfg = RoundtripConfig(
            source_model_dir=model_dir,
            simulation_main_file="NONEXISTENT.in",
            preprocessor_main_file="Preprocessor/Preprocessor.in",
            output_dir=tmp_path / "out",
        )
        pipe = RoundtripPipeline(cfg)
        result = pipe.step_load()
        assert not result.success
        assert "not found" in result.error.lower()


class TestStepWrite:
    @patch("pyiwfm.io.model_writer.CompleteModelWriter")
    @patch("pyiwfm.io.config.ModelWriteConfig")
    def test_success(
        self,
        mock_config_cls: MagicMock,
        mock_writer_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_writer_instance = MagicMock()
        mock_writer_instance.write_all.return_value = MagicMock(
            success=True, files=["a.dat"], errors={}
        )
        mock_writer_cls.return_value = mock_writer_instance

        pipe = _make_pipeline(tmp_path)
        pipe._model = MagicMock()
        result = pipe.step_write()
        assert result.success

    def test_model_not_loaded(self, tmp_path: Path) -> None:
        pipe = _make_pipeline(tmp_path)
        pipe._model = None
        result = pipe.step_write()
        assert not result.success
        assert "not loaded" in result.error.lower()


class TestStepPlaceExecutables:
    def test_with_exe_manager(self, tmp_path: Path) -> None:
        mock_mgr = MagicMock()
        mock_mgr.find_or_download.return_value = MagicMock(available=["sim"])
        mock_mgr.place_executables.return_value = {}
        pipe = _make_pipeline(tmp_path)
        pipe.config.executable_manager = mock_mgr
        result = pipe.step_place_executables()
        assert result.success


class TestStepGenerateScripts:
    def test_success(self, tmp_path: Path) -> None:
        pipe = _make_pipeline(tmp_path)
        written = tmp_path / "written"
        written.mkdir()
        pipe._written_dir = written
        result = pipe.step_generate_scripts()
        assert result.success


class TestStepDiffInputs:
    def test_success(self, tmp_path: Path) -> None:
        pipe = _make_pipeline(tmp_path)
        written = tmp_path / "written"
        written.mkdir()
        pipe._written_dir = written
        result = pipe.step_diff_inputs()
        assert result.success

    def test_written_dir_not_set(self, tmp_path: Path) -> None:
        pipe = _make_pipeline(tmp_path)
        pipe._written_dir = None
        result = pipe.step_diff_inputs()
        assert not result.success


class TestStepRunBaseline:
    def test_skip_when_disabled(self, tmp_path: Path) -> None:
        pipe = _make_pipeline(tmp_path, run_baseline=False)
        result = pipe.step_run_baseline()
        assert result.success
        assert "skip" in result.message.lower()


class TestStepRunWritten:
    def test_skip_when_disabled(self, tmp_path: Path) -> None:
        pipe = _make_pipeline(tmp_path, run_written=False)
        result = pipe.step_run_written()
        assert result.success
        assert "skip" in result.message.lower()

    def test_written_dir_not_set(self, tmp_path: Path) -> None:
        pipe = _make_pipeline(tmp_path, run_written=True)
        pipe._written_dir = None
        result = pipe.step_run_written()
        assert not result.success


class TestStepCompareResults:
    def test_skip_when_disabled(self, tmp_path: Path) -> None:
        pipe = _make_pipeline(tmp_path, compare_results=False)
        result = pipe.step_compare_results()
        assert result.success

    @patch("pyiwfm.comparison.results_differ.ResultsDiffer")
    def test_success(
        self, mock_differ_cls: MagicMock, tmp_path: Path
    ) -> None:
        mock_differ = MagicMock()
        mock_differ.compare_all.return_value = MagicMock(success=True)
        mock_differ_cls.return_value = mock_differ

        pipe = _make_pipeline(tmp_path, compare_results=True)
        pipe._baseline_dir = tmp_path / "baseline"
        pipe._written_dir = tmp_path / "written"
        result = pipe.step_compare_results()
        assert result.success


# ---------------------------------------------------------------------------
# Full pipeline run
# ---------------------------------------------------------------------------

class TestRun:
    @patch("pyiwfm.core.model.IWFMModel")
    @patch("pyiwfm.io.model_writer.CompleteModelWriter")
    @patch("pyiwfm.io.config.ModelWriteConfig")
    def test_full_pipeline(
        self,
        mock_config: MagicMock,
        mock_writer: MagicMock,
        mock_model: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_model.from_simulation_with_preprocessor.return_value = MagicMock()
        mock_writer_inst = MagicMock()
        mock_writer_inst.write_all.return_value = MagicMock(
            success=True, files=["a.dat"], errors={}
        )
        mock_writer.return_value = mock_writer_inst

        pipe = _make_pipeline(tmp_path)
        result = pipe.run()
        assert isinstance(result, RoundtripResult)

    @patch("pyiwfm.core.model.IWFMModel")
    def test_early_abort_on_load_failure(
        self, mock_model: MagicMock, tmp_path: Path
    ) -> None:
        mock_model.from_simulation_with_preprocessor.side_effect = Exception("fail")
        pipe = _make_pipeline(tmp_path)
        result = pipe.run()
        assert not result.success
