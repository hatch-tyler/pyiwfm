"""Deep coverage tests for roundtrip/pipeline.py.

Targets the UNCOVERED lines not exercised by the existing test_roundtrip_pipeline.py:
- Lines 163-165: pp_file not found path
- Lines 214-216: write_result.success=False path
- Lines 232, 244, 259, 266-267: _copy_passthrough_files logic
- Lines 283, 285, 290-293: step_place_executables exception path
- Lines 335-338: step_generate_scripts exception path
- Lines 390-432: step_run_baseline full flow
- Lines 455-481: step_run_written full flow with exes placement
- Lines 519-520, 523-526: step_compare_results with ResultsDiffer
- Lines 554-555, 560, 570-573, 577-580, 584: _run_model() internals
- Lines 606-643: run() full pipeline with all steps
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from pyiwfm.roundtrip.config import RoundtripConfig
from pyiwfm.roundtrip.pipeline import (
    RoundtripPipeline,
    RoundtripResult,
    RunPairResult,
    StepResult,
)
from pyiwfm.runner.results import PreprocessorResult, SimulationResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pipeline(tmp_path: Path, **overrides: object) -> RoundtripPipeline:
    """Create a pipeline with a minimal config and real files on disk."""
    model_dir = tmp_path / "model"
    model_dir.mkdir(exist_ok=True)
    sim_dir = model_dir / "Simulation"
    sim_dir.mkdir(exist_ok=True)
    (sim_dir / "Simulation.in").touch()
    pp_dir = model_dir / "Preprocessor"
    pp_dir.mkdir(exist_ok=True)
    (pp_dir / "Preprocessor.in").touch()

    defaults: dict = {
        "source_model_dir": model_dir,
        "simulation_main_file": "Simulation/Simulation.in",
        "preprocessor_main_file": "Preprocessor/Preprocessor.in",
        "output_dir": tmp_path / "output",
        "run_baseline": False,
        "run_written": False,
        "compare_results": False,
    }
    defaults.update(overrides)
    cfg = RoundtripConfig(**defaults)
    return RoundtripPipeline(cfg)


# ---------------------------------------------------------------------------
# step_load: preprocessor file not found (lines 163-165)
# ---------------------------------------------------------------------------


class TestStepLoadPpNotFound:
    def test_pp_file_not_found(self, tmp_path: Path) -> None:
        """When the preprocessor main file does not exist, step_load should fail."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        sim_dir = model_dir / "Simulation"
        sim_dir.mkdir()
        (sim_dir / "Simulation.in").touch()
        # Preprocessor dir exists but NOT the main file
        pp_dir = model_dir / "Preprocessor"
        pp_dir.mkdir()
        # Deliberately do NOT create Preprocessor.in

        cfg = RoundtripConfig(
            source_model_dir=model_dir,
            simulation_main_file="Simulation/Simulation.in",
            preprocessor_main_file="Preprocessor/Preprocessor.in",
            output_dir=tmp_path / "out",
        )
        pipe = RoundtripPipeline(cfg)
        result = pipe.step_load()
        assert not result.success
        assert "preprocessor" in result.error.lower()
        assert "not found" in result.error.lower()


# ---------------------------------------------------------------------------
# step_write: write_result.success = False (lines 214-216)
# ---------------------------------------------------------------------------


class TestStepWriteErrors:
    @patch("pyiwfm.io.model_writer.CompleteModelWriter")
    @patch("pyiwfm.io.config.ModelWriteConfig")
    def test_write_errors_reported(
        self,
        mock_config_cls: MagicMock,
        mock_writer_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """When write_result.success is False, errors should be joined in step.error."""
        mock_writer_instance = MagicMock()
        mock_writer_instance.write_all.return_value = MagicMock(
            success=False,
            files=[],
            errors={"groundwater": "bad param", "streams": "missing data"},
        )
        mock_writer_cls.return_value = mock_writer_instance

        pipe = _make_pipeline(tmp_path)
        pipe._model = MagicMock()
        result = pipe.step_write()
        assert not result.success
        assert "groundwater" in result.error
        assert "streams" in result.error
        assert result.data is not None


# ---------------------------------------------------------------------------
# _copy_passthrough_files (lines 232, 244, 259, 266-267)
# ---------------------------------------------------------------------------


class TestCopyPassthroughFiles:
    def test_written_dir_none_returns_early(self, tmp_path: Path) -> None:
        """When _written_dir is None, _copy_passthrough_files returns immediately."""
        pipe = _make_pipeline(tmp_path)
        pipe._written_dir = None
        # Should not raise
        pipe._copy_passthrough_files()

    def test_creates_results_dir(self, tmp_path: Path) -> None:
        """If source has a Results dir, ensure written gets an empty one."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        results_dir = model_dir / "Results"
        results_dir.mkdir()
        (results_dir / "output.bin").write_bytes(b"\x00" * 10)

        written_dir = tmp_path / "written"
        written_dir.mkdir()

        pipe = _make_pipeline(tmp_path)
        pipe.config.source_model_dir = model_dir
        pipe._written_dir = written_dir
        pipe._copy_passthrough_files()

        # Results dir should exist but output.bin should NOT be copied
        # (files in Results are skipped)
        assert (written_dir / "Results").exists()
        assert not (written_dir / "Results" / "output.bin").exists()

    def test_copies_binary_files_not_in_results(self, tmp_path: Path) -> None:
        """Binary files (.bin, .hdf, .dss) outside Results should be copied."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "data.bin").write_bytes(b"\x01\x02\x03")
        (model_dir / "input.dat").write_text("some data")

        written_dir = tmp_path / "written"
        written_dir.mkdir()

        pipe = _make_pipeline(tmp_path)
        pipe.config.source_model_dir = model_dir
        pipe._written_dir = written_dir
        pipe._copy_passthrough_files()

        assert (written_dir / "data.bin").exists()
        assert (written_dir / "input.dat").exists()

    def test_skips_files_in_results_directory(self, tmp_path: Path) -> None:
        """Files under Results/ should be skipped even if they match patterns."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        results_dir = model_dir / "Results"
        results_dir.mkdir()
        (results_dir / "heads.hdf").write_bytes(b"\x00")

        written_dir = tmp_path / "written"
        written_dir.mkdir()

        pipe = _make_pipeline(tmp_path)
        pipe.config.source_model_dir = model_dir
        pipe._written_dir = written_dir
        pipe._copy_passthrough_files()

        assert not (written_dir / "Results" / "heads.hdf").exists()

    def test_does_not_overwrite_existing_files(self, tmp_path: Path) -> None:
        """Files already in written dir should not be overwritten."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "keep.dat").write_text("original")

        written_dir = tmp_path / "written"
        written_dir.mkdir()
        (written_dir / "keep.dat").write_text("already written")

        pipe = _make_pipeline(tmp_path)
        pipe.config.source_model_dir = model_dir
        pipe._written_dir = written_dir
        pipe._copy_passthrough_files()

        # Existing file should not have been overwritten
        assert (written_dir / "keep.dat").read_text() == "already written"

    def test_permission_error_silently_skipped(self, tmp_path: Path) -> None:
        """PermissionError during copy is caught and logged, not raised."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "locked.bin").write_bytes(b"\x01")

        written_dir = tmp_path / "written"
        written_dir.mkdir()

        pipe = _make_pipeline(tmp_path)
        pipe.config.source_model_dir = model_dir
        pipe._written_dir = written_dir

        with patch("shutil.copy2", side_effect=PermissionError("locked")):
            # Should not raise
            pipe._copy_passthrough_files()


# ---------------------------------------------------------------------------
# step_place_executables: exception path (lines 290-293)
# ---------------------------------------------------------------------------


class TestStepPlaceExecutablesDeep:
    def test_exception_during_find_or_download(self, tmp_path: Path) -> None:
        """If find_or_download raises, step should fail gracefully."""
        mock_mgr = MagicMock()
        mock_mgr.find_or_download.side_effect = RuntimeError("no executables found")

        pipe = _make_pipeline(tmp_path)
        pipe.config.executable_manager = mock_mgr
        result = pipe.step_place_executables()
        assert not result.success
        assert "no executables found" in result.error

    def test_places_in_baseline_and_written_dirs(self, tmp_path: Path) -> None:
        """When both baseline and written dirs exist, both get executables placed."""
        mock_mgr = MagicMock()
        exes = MagicMock(available=["preprocessor", "simulation"])
        mock_mgr.find_or_download.return_value = exes
        mock_mgr.place_executables.return_value = {}

        pipe = _make_pipeline(tmp_path, run_baseline=True, run_written=True)
        pipe.config.executable_manager = mock_mgr
        baseline_dir = tmp_path / "baseline"
        baseline_dir.mkdir()
        written_dir = tmp_path / "written"
        written_dir.mkdir()
        pipe._baseline_dir = baseline_dir
        pipe._written_dir = written_dir

        result = pipe.step_place_executables()
        assert result.success
        assert mock_mgr.place_executables.call_count == 2


# ---------------------------------------------------------------------------
# step_generate_scripts: exception path (lines 335-338) + both dirs
# ---------------------------------------------------------------------------


class TestStepGenerateScriptsDeep:
    def test_with_both_baseline_and_written_dirs(self, tmp_path: Path) -> None:
        """Scripts generated for both dirs when both exist."""
        pipe = _make_pipeline(tmp_path)
        baseline = tmp_path / "baseline"
        baseline.mkdir()
        written = tmp_path / "written"
        written.mkdir()
        pipe._baseline_dir = baseline
        pipe._written_dir = written

        result = pipe.step_generate_scripts()
        assert result.success
        # Should have generated scripts for both directories
        assert result.data is not None
        assert len(result.data) > 0

    def test_exception_in_generate_scripts(self, tmp_path: Path) -> None:
        """Exception in generate_run_scripts is caught gracefully."""
        pipe = _make_pipeline(tmp_path)
        pipe._written_dir = tmp_path / "written"
        pipe._written_dir.mkdir()

        with patch(
            "pyiwfm.roundtrip.pipeline.generate_run_scripts",
            side_effect=OSError("disk full"),
        ):
            result = pipe.step_generate_scripts()
            assert not result.success
            assert "disk full" in result.error

    def test_with_custom_exes(self, tmp_path: Path) -> None:
        """When _exes is set, custom exe names are used."""
        pipe = _make_pipeline(tmp_path)
        written = tmp_path / "written"
        written.mkdir()
        pipe._written_dir = written

        mock_exes = MagicMock()
        mock_exes.preprocessor = MagicMock()
        mock_exes.preprocessor.name = "PP_custom.exe"
        mock_exes.simulation = MagicMock()
        mock_exes.simulation.name = "Sim_custom.exe"
        pipe._exes = mock_exes

        result = pipe.step_generate_scripts()
        assert result.success


# ---------------------------------------------------------------------------
# step_run_baseline: full flow (lines 390-432)
# ---------------------------------------------------------------------------


class TestStepRunBaselineDeep:
    def test_full_baseline_run_success(self, tmp_path: Path) -> None:
        """Full baseline run: copytree, place exes, _run_model, success path."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        sim_dir = model_dir / "Simulation"
        sim_dir.mkdir()
        (sim_dir / "Simulation.in").write_text("sim")
        pp_dir = model_dir / "Preprocessor"
        pp_dir.mkdir()
        (pp_dir / "Preprocessor.in").write_text("pp")

        pipe = _make_pipeline(tmp_path, run_baseline=True)

        pp_result = PreprocessorResult(success=True, return_code=0)
        sim_result = SimulationResult(success=True, return_code=0)
        pair = RunPairResult(preprocessor=pp_result, simulation=sim_result)

        mock_mgr = MagicMock()
        mock_mgr.find_or_download.return_value = MagicMock(
            preprocessor=MagicMock(), simulation=MagicMock()
        )
        mock_mgr.place_executables.return_value = {}
        pipe.config.executable_manager = mock_mgr

        with patch.object(pipe, "_run_model", return_value=pair):
            result = pipe.step_run_baseline()

        assert result.success
        assert "completed" in result.message.lower()
        assert pipe.result.baseline_run is pair

    def test_baseline_run_failure_reports_errors(self, tmp_path: Path) -> None:
        """When baseline PP or Sim fails, errors are reported."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        sim_dir = model_dir / "Simulation"
        sim_dir.mkdir()
        (sim_dir / "Simulation.in").write_text("sim")
        pp_dir = model_dir / "Preprocessor"
        pp_dir.mkdir()
        (pp_dir / "Preprocessor.in").write_text("pp")

        pipe = _make_pipeline(tmp_path, run_baseline=True)

        pp_result = PreprocessorResult(success=False, return_code=1, errors=["convergence fail"])
        sim_result = SimulationResult(success=True, return_code=0)
        pair = RunPairResult(preprocessor=pp_result, simulation=sim_result)

        mock_mgr = MagicMock()
        mock_mgr.find_or_download.return_value = MagicMock(
            preprocessor=MagicMock(), simulation=MagicMock()
        )
        mock_mgr.place_executables.return_value = {}
        pipe.config.executable_manager = mock_mgr

        with patch.object(pipe, "_run_model", return_value=pair):
            result = pipe.step_run_baseline()

        assert not result.success
        assert "PP:" in result.error

    def test_baseline_exception_caught(self, tmp_path: Path) -> None:
        """If copytree or _run_model raises, step fails gracefully."""
        pipe = _make_pipeline(tmp_path, run_baseline=True)

        with patch("shutil.copytree", side_effect=OSError("copy failed")):
            result = pipe.step_run_baseline()

        assert not result.success
        assert "copy failed" in result.error


# ---------------------------------------------------------------------------
# step_run_written: full flow (lines 455-481)
# ---------------------------------------------------------------------------


class TestStepRunWrittenDeep:
    def test_full_written_run_success(self, tmp_path: Path) -> None:
        """Full written run with exe placement and success."""
        pipe = _make_pipeline(tmp_path, run_written=True)
        written_dir = tmp_path / "written"
        written_dir.mkdir()
        (written_dir / "Simulation").mkdir()
        (written_dir / "Simulation" / "Simulation.in").touch()
        (written_dir / "Preprocessor").mkdir()
        (written_dir / "Preprocessor" / "Preprocessor.in").touch()
        pipe._written_dir = written_dir

        pp_result = PreprocessorResult(success=True, return_code=0)
        sim_result = SimulationResult(success=True, return_code=0)
        pair = RunPairResult(preprocessor=pp_result, simulation=sim_result)

        mock_mgr = MagicMock()
        mock_mgr.place_executables.return_value = {}
        pipe.config.executable_manager = mock_mgr
        pipe._exes = MagicMock()

        with patch.object(pipe, "_run_model", return_value=pair):
            result = pipe.step_run_written()

        assert result.success
        assert "completed" in result.message.lower()

    def test_written_run_failure_with_sim_error(self, tmp_path: Path) -> None:
        """When simulation fails, errors are reported."""
        pipe = _make_pipeline(tmp_path, run_written=True)
        written_dir = tmp_path / "written"
        written_dir.mkdir()
        pipe._written_dir = written_dir

        pp_result = PreprocessorResult(success=True, return_code=0)
        sim_result = SimulationResult(success=False, return_code=1, errors=["mass balance"])
        pair = RunPairResult(preprocessor=pp_result, simulation=sim_result)

        pipe._exes = MagicMock()
        pipe.config.executable_manager = MagicMock()

        with patch.object(pipe, "_run_model", return_value=pair):
            result = pipe.step_run_written()

        assert not result.success
        assert "Sim:" in result.error

    def test_written_run_exception_caught(self, tmp_path: Path) -> None:
        """Exception during _run_model is caught."""
        pipe = _make_pipeline(tmp_path, run_written=True)
        written_dir = tmp_path / "written"
        written_dir.mkdir()
        pipe._written_dir = written_dir
        pipe._exes = MagicMock()
        pipe.config.executable_manager = MagicMock()

        with patch.object(pipe, "_run_model", side_effect=RuntimeError("crash")):
            result = pipe.step_run_written()

        assert not result.success
        assert "crash" in result.error


# ---------------------------------------------------------------------------
# step_compare_results: with ResultsDiffer (lines 519-520, 523-526)
# ---------------------------------------------------------------------------


class TestStepCompareResultsDeep:
    @patch("pyiwfm.comparison.results_differ.ResultsDiffer")
    def test_results_differ_failure(self, mock_differ_cls: MagicMock, tmp_path: Path) -> None:
        """When comparison.success is False, error is set to summary."""
        mock_comparison = MagicMock()
        mock_comparison.success = False
        mock_comparison.summary.return_value = "Heads differ by 0.5 ft"
        mock_differ = MagicMock()
        mock_differ.compare_all.return_value = mock_comparison
        mock_differ_cls.return_value = mock_differ

        pipe = _make_pipeline(tmp_path, compare_results=True)
        pipe._baseline_dir = tmp_path / "baseline"
        pipe._baseline_dir.mkdir()
        pipe._written_dir = tmp_path / "written"
        pipe._written_dir.mkdir()

        result = pipe.step_compare_results()
        assert not result.success
        assert "Heads differ" in result.error

    def test_compare_missing_dirs(self, tmp_path: Path) -> None:
        """When baseline or written dir is None, step fails."""
        pipe = _make_pipeline(tmp_path, compare_results=True)
        pipe._baseline_dir = None
        pipe._written_dir = tmp_path / "written"
        result = pipe.step_compare_results()
        assert not result.success
        assert "both" in result.error.lower()

    @patch("pyiwfm.comparison.results_differ.ResultsDiffer")
    def test_compare_exception_caught(self, mock_differ_cls: MagicMock, tmp_path: Path) -> None:
        """Exception during comparison is caught gracefully."""
        mock_differ_cls.side_effect = ImportError("h5py not available")

        pipe = _make_pipeline(tmp_path, compare_results=True)
        pipe._baseline_dir = tmp_path / "baseline"
        pipe._baseline_dir.mkdir()
        pipe._written_dir = tmp_path / "written"
        pipe._written_dir.mkdir()

        result = pipe.step_compare_results()
        assert not result.success
        assert "h5py" in result.error


# ---------------------------------------------------------------------------
# _run_model internals (lines 554-555, 560, 570-573, 577-580, 584)
# ---------------------------------------------------------------------------


class TestRunModelInternal:
    def test_run_model_pp_and_sim_success(self, tmp_path: Path) -> None:
        """Both preprocessor and simulation run and succeed."""
        pipe = _make_pipeline(tmp_path)
        model_dir = tmp_path / "run_model_test"
        model_dir.mkdir()
        pp_dir = model_dir / "Preprocessor"
        pp_dir.mkdir()
        (pp_dir / "Preprocessor.in").touch()
        sim_dir = model_dir / "Simulation"
        sim_dir.mkdir()
        (sim_dir / "Simulation.in").touch()

        pp_result = PreprocessorResult(success=True, return_code=0)
        sim_result = SimulationResult(success=True, return_code=0)

        mock_runner = MagicMock()
        mock_runner.run_preprocessor.return_value = pp_result
        mock_runner.run_simulation.return_value = sim_result

        mock_exes = MagicMock()
        mock_exes.preprocessor = MagicMock()
        mock_exes.simulation = MagicMock()
        pipe._exes = mock_exes

        with patch("pyiwfm.roundtrip.pipeline.IWFMRunner", return_value=mock_runner):
            pair = pipe._run_model(model_dir)

        assert pair.success
        assert pair.preprocessor is pp_result
        assert pair.simulation is sim_result

    def test_run_model_pp_failure_aborts_sim(self, tmp_path: Path) -> None:
        """If preprocessor fails, simulation is not run."""
        pipe = _make_pipeline(tmp_path)
        model_dir = tmp_path / "run_model_test2"
        model_dir.mkdir()
        pp_dir = model_dir / "Preprocessor"
        pp_dir.mkdir()
        (pp_dir / "Preprocessor.in").touch()
        sim_dir = model_dir / "Simulation"
        sim_dir.mkdir()
        (sim_dir / "Simulation.in").touch()

        pp_result = PreprocessorResult(success=False, return_code=1, errors=["node mismatch"])

        mock_runner = MagicMock()
        mock_runner.run_preprocessor.return_value = pp_result

        mock_exes = MagicMock()
        mock_exes.preprocessor = MagicMock()
        mock_exes.simulation = MagicMock()
        pipe._exes = mock_exes

        with patch("pyiwfm.roundtrip.pipeline.IWFMRunner", return_value=mock_runner):
            pair = pipe._run_model(model_dir)

        assert not pair.success
        assert pair.preprocessor is pp_result
        # Simulation was never called
        mock_runner.run_simulation.assert_not_called()

    def test_run_model_sim_failure(self, tmp_path: Path) -> None:
        """PP succeeds but Sim fails."""
        pipe = _make_pipeline(tmp_path)
        model_dir = tmp_path / "run_model_test3"
        model_dir.mkdir()
        pp_dir = model_dir / "Preprocessor"
        pp_dir.mkdir()
        (pp_dir / "Preprocessor.in").touch()
        sim_dir = model_dir / "Simulation"
        sim_dir.mkdir()
        (sim_dir / "Simulation.in").touch()

        pp_result = PreprocessorResult(success=True, return_code=0)
        sim_result = SimulationResult(success=False, return_code=2, errors=["divergence"])

        mock_runner = MagicMock()
        mock_runner.run_preprocessor.return_value = pp_result
        mock_runner.run_simulation.return_value = sim_result

        mock_exes = MagicMock()
        mock_exes.preprocessor = MagicMock()
        mock_exes.simulation = MagicMock()
        pipe._exes = mock_exes

        with patch("pyiwfm.roundtrip.pipeline.IWFMRunner", return_value=mock_runner):
            pair = pipe._run_model(model_dir)

        assert not pair.success
        assert pair.simulation is sim_result

    def test_run_model_no_exes(self, tmp_path: Path) -> None:
        """When _exes is None, neither PP nor Sim run."""
        pipe = _make_pipeline(tmp_path)
        model_dir = tmp_path / "run_model_no_exes"
        model_dir.mkdir()
        pp_dir = model_dir / "Preprocessor"
        pp_dir.mkdir()
        (pp_dir / "Preprocessor.in").touch()
        sim_dir = model_dir / "Simulation"
        sim_dir.mkdir()
        (sim_dir / "Simulation.in").touch()

        pipe._exes = None

        mock_runner = MagicMock()
        with patch("pyiwfm.roundtrip.pipeline.IWFMRunner", return_value=mock_runner):
            pair = pipe._run_model(model_dir)

        assert pair.preprocessor is None
        assert pair.simulation is None


# ---------------------------------------------------------------------------
# run() full pipeline (lines 606-643)
# ---------------------------------------------------------------------------


class TestRunFullPipeline:
    @patch("pyiwfm.core.model.IWFMModel")
    @patch("pyiwfm.io.model_writer.CompleteModelWriter")
    @patch("pyiwfm.io.config.ModelWriteConfig")
    def test_full_pipeline_with_runs_and_compare(
        self,
        mock_config: MagicMock,
        mock_writer_cls: MagicMock,
        mock_model_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Full pipeline with baseline run, written run, and comparison."""
        mock_model_cls.from_simulation_with_preprocessor.return_value = MagicMock()
        mock_writer_inst = MagicMock()
        mock_writer_inst.write_all.return_value = MagicMock(
            success=True, files=["a.dat"], errors={}
        )
        mock_writer_cls.return_value = mock_writer_inst

        pipe = _make_pipeline(
            tmp_path,
            run_baseline=True,
            run_written=True,
            compare_results=True,
        )

        pp_result = PreprocessorResult(success=True, return_code=0)
        sim_result = SimulationResult(success=True, return_code=0)
        pair = RunPairResult(preprocessor=pp_result, simulation=sim_result)

        mock_mgr = MagicMock()
        mock_mgr.find_or_download.return_value = MagicMock(
            preprocessor=MagicMock(), simulation=MagicMock()
        )
        mock_mgr.place_executables.return_value = {}
        pipe.config.executable_manager = mock_mgr

        with (
            patch.object(pipe, "_run_model", return_value=pair),
            patch.object(pipe, "_copy_passthrough_files"),
            patch("pyiwfm.comparison.results_differ.ResultsDiffer") as mock_differ_cls,
        ):
            mock_comparison = MagicMock()
            mock_comparison.success = True
            mock_differ = MagicMock()
            mock_differ.compare_all.return_value = mock_comparison
            mock_differ_cls.return_value = mock_differ

            result = pipe.run()

        assert isinstance(result, RoundtripResult)
        assert "load" in result.steps
        assert "write" in result.steps

    @patch("pyiwfm.core.model.IWFMModel")
    @patch("pyiwfm.io.model_writer.CompleteModelWriter")
    @patch("pyiwfm.io.config.ModelWriteConfig")
    def test_write_failure_aborts_pipeline(
        self,
        mock_config: MagicMock,
        mock_writer_cls: MagicMock,
        mock_model_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """If step_write fails, pipeline aborts before running."""
        mock_model_cls.from_simulation_with_preprocessor.return_value = MagicMock()
        mock_writer_inst = MagicMock()
        mock_writer_inst.write_all.side_effect = Exception("write crashed")
        mock_writer_cls.return_value = mock_writer_inst

        pipe = _make_pipeline(tmp_path, run_baseline=True, run_written=True)
        result = pipe.run()
        assert not result.success
        assert "write" in result.steps

    @patch("pyiwfm.core.model.IWFMModel")
    @patch("pyiwfm.io.model_writer.CompleteModelWriter")
    @patch("pyiwfm.io.config.ModelWriteConfig")
    def test_baseline_failure_disables_compare(
        self,
        mock_config: MagicMock,
        mock_writer_cls: MagicMock,
        mock_model_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """If baseline run fails, compare_results is set to False."""
        mock_model_cls.from_simulation_with_preprocessor.return_value = MagicMock()
        mock_writer_inst = MagicMock()
        mock_writer_inst.write_all.return_value = MagicMock(
            success=True, files=["a.dat"], errors={}
        )
        mock_writer_cls.return_value = mock_writer_inst

        pipe = _make_pipeline(
            tmp_path,
            run_baseline=True,
            run_written=False,
            compare_results=True,
        )

        # Simulate baseline failure
        failed_pair = RunPairResult(
            preprocessor=PreprocessorResult(success=False, return_code=1, errors=["fail"]),
            simulation=None,
        )

        mock_mgr = MagicMock()
        mock_mgr.find_or_download.return_value = MagicMock(
            preprocessor=MagicMock(), simulation=MagicMock()
        )
        mock_mgr.place_executables.return_value = {}
        pipe.config.executable_manager = mock_mgr

        with (
            patch.object(pipe, "_run_model", return_value=failed_pair),
            patch.object(pipe, "_copy_passthrough_files"),
        ):
            pipe.run()

        # compare_results should have been disabled due to baseline failure
        assert pipe.config.compare_results is False

    @patch("pyiwfm.core.model.IWFMModel")
    @patch("pyiwfm.io.model_writer.CompleteModelWriter")
    @patch("pyiwfm.io.config.ModelWriteConfig")
    def test_written_failure_disables_compare(
        self,
        mock_config: MagicMock,
        mock_writer_cls: MagicMock,
        mock_model_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """If written run fails, compare_results is set to False."""
        mock_model_cls.from_simulation_with_preprocessor.return_value = MagicMock()
        mock_writer_inst = MagicMock()
        mock_writer_inst.write_all.return_value = MagicMock(
            success=True, files=["a.dat"], errors={}
        )
        mock_writer_cls.return_value = mock_writer_inst

        pipe = _make_pipeline(
            tmp_path,
            run_baseline=False,
            run_written=True,
            compare_results=True,
        )

        failed_pair = RunPairResult(
            preprocessor=PreprocessorResult(success=True, return_code=0),
            simulation=SimulationResult(success=False, return_code=1, errors=["diverge"]),
        )

        pipe._exes = MagicMock()
        pipe.config.executable_manager = MagicMock()

        with (
            patch.object(pipe, "_run_model", return_value=failed_pair),
            patch.object(pipe, "_copy_passthrough_files"),
        ):
            pipe.run()

        assert pipe.config.compare_results is False


# ---------------------------------------------------------------------------
# Summary formatting edge cases
# ---------------------------------------------------------------------------


class TestSummaryFormatting:
    def test_summary_with_errors(self) -> None:
        """Summary includes error details for failed steps."""
        r = RoundtripResult(
            steps={
                "load": StepResult(name="load", success=True, message="OK"),
                "write": StepResult(
                    name="write",
                    success=False,
                    message="Write failed",
                    error="groundwater: missing data",
                ),
            }
        )
        s = r.summary()
        assert "FAIL" in s
        assert "PASS" in s
        assert "groundwater" in s
        assert "FAILED" in s

    def test_summary_all_pass(self) -> None:
        """Summary shows PASSED when all steps pass."""
        r = RoundtripResult(
            steps={
                "load": StepResult(name="load", success=True, message="OK"),
                "write": StepResult(name="write", success=True, message="OK"),
            }
        )
        s = r.summary()
        assert "PASSED" in s
