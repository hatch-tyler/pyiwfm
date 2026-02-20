"""Main roundtrip testing pipeline orchestrator."""

from __future__ import annotations

import logging
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pyiwfm.roundtrip.config import RoundtripConfig
from pyiwfm.roundtrip.file_differ import InputDiffResult, diff_all_files
from pyiwfm.roundtrip.script_generator import generate_run_scripts
from pyiwfm.runner.executables import IWFMExecutableManager
from pyiwfm.runner.results import PreprocessorResult, SimulationResult
from pyiwfm.runner.runner import IWFMExecutables, IWFMRunner

logger = logging.getLogger(__name__)


@dataclass
class StepResult:
    """Result of a single pipeline step.

    Attributes
    ----------
    name : str
        Step name.
    success : bool
        Whether the step completed successfully.
    message : str
        Human-readable status message.
    data : Any
        Step-specific result data.
    error : str
        Error message if the step failed.
    """

    name: str = ""
    success: bool = False
    message: str = ""
    data: Any = None
    error: str = ""


@dataclass
class RunPairResult:
    """Result of running preprocessor + simulation.

    Attributes
    ----------
    preprocessor : PreprocessorResult | None
        Result from preprocessing.
    simulation : SimulationResult | None
        Result from simulation.
    """

    preprocessor: PreprocessorResult | None = None
    simulation: SimulationResult | None = None

    @property
    def success(self) -> bool:
        """True if both preprocessor and simulation succeeded."""
        pp_ok = self.preprocessor is not None and self.preprocessor.success
        sim_ok = self.simulation is not None and self.simulation.success
        return pp_ok and sim_ok


@dataclass
class RoundtripResult:
    """Aggregate result of the full roundtrip pipeline.

    Attributes
    ----------
    steps : dict[str, StepResult]
        Results for each pipeline step.
    baseline_run : RunPairResult | None
        Result of running the original model.
    written_run : RunPairResult | None
        Result of running the written model.
    input_diff : InputDiffResult | None
        Input file comparison result.
    results_comparison : Any
        Simulation output comparison result.
    """

    steps: dict[str, StepResult] = field(default_factory=dict)
    baseline_run: RunPairResult | None = None
    written_run: RunPairResult | None = None
    input_diff: InputDiffResult | None = None
    results_comparison: Any = None

    @property
    def success(self) -> bool:
        """True if all executed steps succeeded."""
        return all(s.success for s in self.steps.values())

    def summary(self) -> str:
        """Generate a human-readable summary of all steps."""
        lines = [
            "Roundtrip Test Summary",
            "=" * 50,
        ]
        for name, step in self.steps.items():
            status = "PASS" if step.success else "FAIL"
            lines.append(f"  [{status}] {name}: {step.message}")
            if step.error:
                lines.append(f"         Error: {step.error}")

        lines.append("=" * 50)
        overall = "PASSED" if self.success else "FAILED"
        lines.append(f"Overall: {overall}")
        return "\n".join(lines)


# Patterns to skip when copying the model tree for baseline
_COPY_IGNORE_PATTERNS = shutil.ignore_patterns(
    "*.bak",
    "Results",
    "results",
    "__pycache__",
    "*.pyc",
)


class RoundtripPipeline:
    """Orchestrates the full roundtrip test: load -> write -> run -> verify.

    Parameters
    ----------
    config : RoundtripConfig
        Pipeline configuration.
    """

    def __init__(self, config: RoundtripConfig) -> None:
        self.config = config
        self.result = RoundtripResult()
        self._model: Any = None
        self._exes: IWFMExecutables | None = None
        self._baseline_dir: Path | None = None
        self._written_dir: Path | None = None

    # ------------------------------------------------------------------
    # Step 1: Load model
    # ------------------------------------------------------------------

    def step_load(self) -> StepResult:
        """Load the model via IWFMModel.from_simulation_with_preprocessor()."""
        step = StepResult(name="load")
        try:
            from pyiwfm.core.model import IWFMModel

            cfg = self.config
            sim_file = cfg.source_model_dir / cfg.simulation_main_file
            pp_file = cfg.source_model_dir / cfg.preprocessor_main_file

            if not sim_file.exists():
                step.error = f"Simulation main file not found: {sim_file}"
                self.result.steps["load"] = step
                return step
            if not pp_file.exists():
                step.error = f"Preprocessor main file not found: {pp_file}"
                self.result.steps["load"] = step
                return step

            self._model = IWFMModel.from_simulation_with_preprocessor(
                simulation_file=str(sim_file),
                preprocessor_file=str(pp_file),
            )
            step.success = True
            step.message = "Model loaded successfully"
            step.data = self._model
        except Exception as exc:
            step.error = str(exc)
            step.message = "Model loading failed"
            logger.exception("Failed to load model")

        self.result.steps["load"] = step
        return step

    # ------------------------------------------------------------------
    # Step 2: Write model
    # ------------------------------------------------------------------

    def step_write(self) -> StepResult:
        """Write the model using CompleteModelWriter."""
        step = StepResult(name="write")
        try:
            from pyiwfm.io.config import ModelWriteConfig
            from pyiwfm.io.model_writer import CompleteModelWriter

            if self._model is None:
                step.error = "Model not loaded (run step_load first)"
                self.result.steps["write"] = step
                return step

            self._written_dir = self.config.output_dir / "written"
            self._written_dir.mkdir(parents=True, exist_ok=True)

            write_config = ModelWriteConfig(output_dir=self._written_dir)
            writer = CompleteModelWriter(
                model=self._model,
                config=write_config,
            )
            write_result = writer.write_all()

            if write_result.success:
                step.success = True
                n_files = len(write_result.files)
                step.message = f"Wrote {n_files} files"
                step.data = write_result
            else:
                step.error = "; ".join(f"{k}: {v}" for k, v in write_result.errors.items())
                step.message = "Write completed with errors"
                step.data = write_result

            # Copy files that the writer can't regenerate
            self._copy_passthrough_files()

        except Exception as exc:
            step.error = str(exc)
            step.message = "Write failed"
            logger.exception("Failed to write model")

        self.result.steps["write"] = step
        return step

    def _copy_passthrough_files(self) -> None:
        """Copy files the writer doesn't regenerate from source to written dir."""
        if self._written_dir is None:
            return

        src = self.config.source_model_dir

        # Copy binary files and text data files not already generated by the writer.
        # Binary files (.bin, .hdf, .dss) are always copied.
        # Text files (.dat, .IN) are copied as fallback for sub-files
        # the writer doesn't yet regenerate (e.g. Subsidence.dat, BoundTSD.dat).
        # Skip the Results directory — those are output files IWFM recreates.
        # But ensure the Results directory exists so IWFM can write to it.
        results_dir = src / "Results"
        if results_dir.exists():
            (self._written_dir / "Results").mkdir(parents=True, exist_ok=True)
        for pattern in [
            "**/*.bin", "**/*.BIN",
            "**/*.hdf", "**/*.HDF",
            "**/*.dss", "**/*.DSS",
            "**/*.dat", "**/*.DAT",
            "**/*.IN", "**/*.in",
        ]:
            for f in src.glob(pattern):
                if f.is_relative_to(results_dir):
                    continue
                rel = f.relative_to(src)
                dst = self._written_dir / rel
                if not dst.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        shutil.copy2(str(f), str(dst))
                    except PermissionError:
                        logger.debug("Skipping locked file: %s", rel)

    # ------------------------------------------------------------------
    # Step 3: Place executables
    # ------------------------------------------------------------------

    def step_place_executables(self) -> StepResult:
        """Find/download executables and place them in model directories."""
        step = StepResult(name="place_executables")
        try:
            mgr = self.config.executable_manager or IWFMExecutableManager()
            self._exes = mgr.find_or_download()

            placed: dict[str, dict[str, Path]] = {}

            if self.config.run_baseline and self._baseline_dir:
                placed["baseline"] = mgr.place_executables(self._exes, self._baseline_dir)
            if self.config.run_written and self._written_dir:
                placed["written"] = mgr.place_executables(self._exes, self._written_dir)

            step.success = True
            step.message = f"Placed executables: {self._exes.available}"
            step.data = placed
        except Exception as exc:
            step.error = str(exc)
            step.message = "Failed to place executables"
            logger.exception("Failed to place executables")

        self.result.steps["place_executables"] = step
        return step

    # ------------------------------------------------------------------
    # Step 4: Generate run scripts
    # ------------------------------------------------------------------

    def step_generate_scripts(self) -> StepResult:
        """Generate .bat/.sh run scripts for model directories."""
        step = StepResult(name="generate_scripts")
        try:
            scripts: list[Path] = []
            cfg = self.config

            # Detect exe names
            pp_exe = "PreProcessor_x64.exe" if sys.platform == "win32" else "PreProcessor"
            sim_exe = "Simulation_x64.exe" if sys.platform == "win32" else "Simulation"

            if self._exes and self._exes.preprocessor:
                pp_exe = self._exes.preprocessor.name
            if self._exes and self._exes.simulation:
                sim_exe = self._exes.simulation.name

            for _label, model_dir in [
                ("baseline", self._baseline_dir),
                ("written", self._written_dir),
            ]:
                if model_dir and model_dir.exists():
                    s = generate_run_scripts(
                        model_dir=model_dir,
                        preprocessor_main=cfg.preprocessor_main_file,
                        simulation_main=cfg.simulation_main_file,
                        preprocessor_exe=pp_exe,
                        simulation_exe=sim_exe,
                    )
                    scripts.extend(s)

            step.success = True
            step.message = f"Generated {len(scripts)} scripts"
            step.data = scripts
        except Exception as exc:
            step.error = str(exc)
            step.message = "Failed to generate scripts"
            logger.exception("Failed to generate run scripts")

        self.result.steps["generate_scripts"] = step
        return step

    # ------------------------------------------------------------------
    # Step 5: Diff input files
    # ------------------------------------------------------------------

    def step_diff_inputs(self) -> StepResult:
        """Compare written input files against originals."""
        step = StepResult(name="diff_inputs")
        try:
            if self._written_dir is None:
                step.error = "Written dir not set (run step_write first)"
                self.result.steps["diff_inputs"] = step
                return step

            diff_result = diff_all_files(
                original_dir=self.config.source_model_dir,
                written_dir=self._written_dir,
            )

            self.result.input_diff = diff_result
            step.success = True
            step.message = (
                f"Compared {diff_result.files_compared} files: "
                f"{diff_result.files_data_identical} data-identical"
            )
            step.data = diff_result
        except Exception as exc:
            step.error = str(exc)
            step.message = "Input diff failed"
            logger.exception("Failed to diff input files")

        self.result.steps["diff_inputs"] = step
        return step

    # ------------------------------------------------------------------
    # Step 6: Run baseline
    # ------------------------------------------------------------------

    def step_run_baseline(self) -> StepResult:
        """Copy original model to temp dir and run PP + Sim."""
        step = StepResult(name="run_baseline")

        if not self.config.run_baseline:
            step.success = True
            step.message = "Skipped (run_baseline=False)"
            self.result.steps["run_baseline"] = step
            return step

        try:
            # Copy original model to baseline dir
            self._baseline_dir = self.config.output_dir / "baseline"
            if self._baseline_dir.exists():
                shutil.rmtree(self._baseline_dir)
            shutil.copytree(
                str(self.config.source_model_dir),
                str(self._baseline_dir),
                ignore=_COPY_IGNORE_PATTERNS,
            )

            # Place executables in baseline dir
            if self._exes is None:
                mgr = self.config.executable_manager or IWFMExecutableManager()
                self._exes = mgr.find_or_download()

            mgr_obj = self.config.executable_manager or IWFMExecutableManager()
            mgr_obj.place_executables(self._exes, self._baseline_dir)

            # Run
            pair = self._run_model(self._baseline_dir)
            self.result.baseline_run = pair

            if pair.success:
                step.success = True
                step.message = "Baseline PP + Sim completed"
            else:
                errors: list[str] = []
                if pair.preprocessor and not pair.preprocessor.success:
                    errors.append(f"PP: {'; '.join(pair.preprocessor.errors[:3])}")
                if pair.simulation and not pair.simulation.success:
                    errors.append(f"Sim: {'; '.join(pair.simulation.errors[:3])}")
                step.error = " | ".join(errors)
                step.message = "Baseline run failed"

            step.data = pair
        except Exception as exc:
            step.error = str(exc)
            step.message = "Baseline run failed"
            logger.exception("Failed to run baseline model")

        self.result.steps["run_baseline"] = step
        return step

    # ------------------------------------------------------------------
    # Step 7: Run written model
    # ------------------------------------------------------------------

    def step_run_written(self) -> StepResult:
        """Run PP + Sim on the written model."""
        step = StepResult(name="run_written")

        if not self.config.run_written:
            step.success = True
            step.message = "Skipped (run_written=False)"
            self.result.steps["run_written"] = step
            return step

        try:
            if self._written_dir is None:
                step.error = "Written dir not set (run step_write first)"
                self.result.steps["run_written"] = step
                return step

            # Ensure executables are placed
            if self._exes:
                mgr_obj = self.config.executable_manager or IWFMExecutableManager()
                mgr_obj.place_executables(self._exes, self._written_dir)

            pair = self._run_model(self._written_dir)
            self.result.written_run = pair

            if pair.success:
                step.success = True
                step.message = "Written PP + Sim completed"
            else:
                errors: list[str] = []
                if pair.preprocessor and not pair.preprocessor.success:
                    errors.append(f"PP: {'; '.join(pair.preprocessor.errors[:3])}")
                if pair.simulation and not pair.simulation.success:
                    errors.append(f"Sim: {'; '.join(pair.simulation.errors[:3])}")
                step.error = " | ".join(errors)
                step.message = "Written model run failed"

            step.data = pair
        except Exception as exc:
            step.error = str(exc)
            step.message = "Written model run failed"
            logger.exception("Failed to run written model")

        self.result.steps["run_written"] = step
        return step

    # ------------------------------------------------------------------
    # Step 8: Compare results
    # ------------------------------------------------------------------

    def step_compare_results(self) -> StepResult:
        """Compare simulation outputs between baseline and written runs."""
        step = StepResult(name="compare_results")

        if not self.config.compare_results:
            step.success = True
            step.message = "Skipped (compare_results=False)"
            self.result.steps["compare_results"] = step
            return step

        try:
            if self._baseline_dir is None or self._written_dir is None:
                step.error = "Both baseline and written dirs needed"
                self.result.steps["compare_results"] = step
                return step

            from pyiwfm.comparison.results_differ import ResultsDiffer

            differ = ResultsDiffer(
                baseline_dir=self._baseline_dir,
                written_dir=self._written_dir,
                head_atol=self.config.head_atol,
                budget_rtol=self.config.budget_rtol,
            )

            comparison = differ.compare_all()
            self.result.results_comparison = comparison

            if comparison.success:
                step.success = True
                step.message = "Results match within tolerance"
            else:
                step.error = comparison.summary()
                step.message = "Results differ"

            step.data = comparison
        except Exception as exc:
            step.error = str(exc)
            step.message = "Results comparison failed"
            logger.exception("Failed to compare results")

        self.result.steps["compare_results"] = step
        return step

    # ------------------------------------------------------------------
    # Step 9: Run all
    # ------------------------------------------------------------------

    def run(self) -> RoundtripResult:
        """Execute the full pipeline: load -> write -> run -> verify.

        Returns
        -------
        RoundtripResult
            Aggregate result with all step outcomes.
        """
        logger.info("Starting roundtrip pipeline for %s", self.config.source_model_dir)

        # Step 1: Load
        load = self.step_load()
        if not load.success:
            logger.error("Pipeline aborted: load failed")
            return self.result

        # Step 2: Write
        write = self.step_write()
        if not write.success:
            logger.error("Pipeline aborted: write failed")
            return self.result

        # Step 3: Place executables
        needs_run = self.config.run_baseline or self.config.run_written
        if needs_run:
            self.step_place_executables()

        # Step 4: Generate scripts
        self.step_generate_scripts()

        # Step 5: Diff inputs
        self.step_diff_inputs()

        # Step 6: Run baseline
        if self.config.run_baseline:
            baseline = self.step_run_baseline()
            if not baseline.success:
                logger.warning("Baseline run failed; skipping comparison")
                self.config.compare_results = False

        # Step 7: Run written
        if self.config.run_written:
            written = self.step_run_written()
            if not written.success:
                logger.warning("Written run failed; skipping comparison")
                self.config.compare_results = False

        # Step 8: Compare results
        if self.config.compare_results:
            self.step_compare_results()

        logger.info("Pipeline complete. %s", "PASSED" if self.result.success else "FAILED")
        return self.result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_model(self, model_dir: Path) -> RunPairResult:
        """Run preprocessor + simulation in a model directory.

        Parameters
        ----------
        model_dir : Path
            Model root directory.

        Returns
        -------
        RunPairResult
            Combined PP + Sim results.
        """
        pair = RunPairResult()
        cfg = self.config

        runner = IWFMRunner(
            executables=self._exes,
            working_dir=model_dir,
        )

        # Run preprocessor
        pp_main = model_dir / cfg.preprocessor_main_file
        if pp_main.exists() and self._exes and self._exes.preprocessor:
            logger.info("Running preprocessor in %s", model_dir)
            pair.preprocessor = runner.run_preprocessor(
                main_file=pp_main,
                timeout=cfg.preprocessor_timeout,
            )
            if not pair.preprocessor.success:
                logger.error(
                    "Preprocessor failed: %s",
                    pair.preprocessor.errors[:3],
                )
                return pair

        # Run simulation
        sim_main = model_dir / cfg.simulation_main_file
        if sim_main.exists() and self._exes and self._exes.simulation:
            logger.info("Running simulation in %s", model_dir)
            pair.simulation = runner.run_simulation(
                main_file=sim_main,
                timeout=cfg.simulation_timeout,
            )
            if not pair.simulation.success:
                logger.error(
                    "Simulation failed: %s",
                    pair.simulation.errors[:3],
                )

        return pair
