"""Configuration for the roundtrip testing pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from pyiwfm.runner.executables import IWFMExecutableManager


@dataclass
class RoundtripConfig:
    """Configuration for a roundtrip test run.

    Parameters
    ----------
    source_model_dir : Path
        Root directory of the source IWFM model.
    simulation_main_file : str
        Relative path to the Simulation main file within the model dir.
    preprocessor_main_file : str
        Relative path to the PreProcessor main file within the model dir.
    output_dir : Path
        Directory for written model and test outputs.
    executable_manager : IWFMExecutableManager | None
        Manager for finding/downloading executables. Created automatically
        if not provided.
    run_baseline : bool
        Whether to run the original model for comparison.
    run_written : bool
        Whether to run the written (roundtripped) model.
    compare_results : bool
        Whether to compare simulation results between baseline and written.
    preprocessor_timeout : float
        Timeout in seconds for preprocessor runs.
    simulation_timeout : float
        Timeout in seconds for simulation runs.
    head_atol : float
        Absolute tolerance for head comparisons (ft).
    budget_rtol : float
        Relative tolerance for budget comparisons.
    """

    source_model_dir: Path = field(default_factory=lambda: Path("."))
    simulation_main_file: str = ""
    preprocessor_main_file: str = ""
    output_dir: Path = field(default_factory=lambda: Path("roundtrip_output"))
    executable_manager: IWFMExecutableManager | None = None
    run_baseline: bool = True
    run_written: bool = True
    compare_results: bool = True
    preprocessor_timeout: float = 300
    simulation_timeout: float = 3600
    head_atol: float = 0.01
    budget_rtol: float = 1e-3

    def __post_init__(self) -> None:
        """Convert strings to Path objects."""
        self.source_model_dir = Path(self.source_model_dir)
        self.output_dir = Path(self.output_dir)

    @classmethod
    def from_env(cls) -> RoundtripConfig:
        """Create config from environment variables.

        Reads:
        - IWFM_MODEL_DIR: source model directory
        - IWFM_ROUNDTRIP_OUTPUT: output directory
        - IWFM_BIN: path to executables

        Returns
        -------
        RoundtripConfig
            Configuration populated from environment.
        """
        model_dir = Path(os.environ.get("IWFM_MODEL_DIR", "."))
        output_dir = Path(os.environ.get("IWFM_ROUNDTRIP_OUTPUT", "roundtrip_output"))

        # Auto-detect main files
        sim_main = _find_main_file(model_dir, "Simulation")
        pp_main = _find_main_file(model_dir, "Preprocessor")

        return cls(
            source_model_dir=model_dir,
            simulation_main_file=sim_main,
            preprocessor_main_file=pp_main,
            output_dir=output_dir,
        )

    @classmethod
    def for_sample_model(cls, iwfm_dir: Path | str) -> RoundtripConfig:
        """Create config for the IWFM Sample Model.

        Parameters
        ----------
        iwfm_dir : Path | str
            Root directory containing the sample model.

        Returns
        -------
        RoundtripConfig
            Configuration for sample model roundtrip test.
        """
        iwfm_dir = Path(iwfm_dir)

        # Search sibling Bin/ for executables (standard CNRA zip layout)
        bin_sibling = iwfm_dir.parent / "Bin"
        search_paths = [bin_sibling] if bin_sibling.exists() else []
        exe_mgr = IWFMExecutableManager(search_paths=search_paths) if search_paths else None

        return cls(
            source_model_dir=iwfm_dir,
            simulation_main_file=_find_main_file(iwfm_dir, "Simulation"),
            preprocessor_main_file=_find_main_file(iwfm_dir, "Preprocessor"),
            output_dir=iwfm_dir.parent / "roundtrip_sample",
            executable_manager=exe_mgr,
            preprocessor_timeout=120,
            simulation_timeout=600,
        )

    @classmethod
    def for_c2vsimfg(cls, c2vsimfg_dir: Path | str) -> RoundtripConfig:
        """Create config for the C2VSimFG model.

        Parameters
        ----------
        c2vsimfg_dir : Path | str
            Root directory of the C2VSimFG model.

        Returns
        -------
        RoundtripConfig
            Configuration for C2VSimFG roundtrip test.
        """
        c2vsimfg_dir = Path(c2vsimfg_dir)

        return cls(
            source_model_dir=c2vsimfg_dir,
            simulation_main_file=_find_main_file(c2vsimfg_dir, "Simulation"),
            preprocessor_main_file=_find_main_file(c2vsimfg_dir, "Preprocessor"),
            output_dir=c2vsimfg_dir.parent / "roundtrip_c2vsimfg",
            preprocessor_timeout=600,
            simulation_timeout=3600,
            head_atol=0.01,
            budget_rtol=1e-3,
        )

    @classmethod
    def for_c2vsimcg(cls, c2vsimcg_dir: Path | str) -> RoundtripConfig:
        """Create config for the C2VSimCG (Coarse Grid) model.

        Parameters
        ----------
        c2vsimcg_dir : Path | str
            Root directory of the C2VSimCG model.

        Returns
        -------
        RoundtripConfig
            Configuration for C2VSimCG roundtrip test.
        """
        c2vsimcg_dir = Path(c2vsimcg_dir)

        # Executables ship in !bin/IWFM-2025.0.1747/ inside the model
        bin_dir = c2vsimcg_dir / "!bin" / "IWFM-2025.0.1747"
        exe_mgr = IWFMExecutableManager(search_paths=[bin_dir]) if bin_dir.exists() else None

        return cls(
            source_model_dir=c2vsimcg_dir,
            simulation_main_file=_find_main_file(c2vsimcg_dir, "Simulation"),
            preprocessor_main_file=_find_main_file(c2vsimcg_dir, "Preprocessor"),
            output_dir=c2vsimcg_dir.parent / "roundtrip_c2vsimcg",
            executable_manager=exe_mgr,
            preprocessor_timeout=300,
            simulation_timeout=1800,
            head_atol=0.01,
            budget_rtol=1e-3,
        )


def _find_main_file(model_dir: Path, component: str) -> str:
    """Auto-detect the main input file for a component.

    Searches for common naming conventions in the component subdirectory.

    Parameters
    ----------
    model_dir : Path
        Model root directory.
    component : str
        Component name ('Simulation' or 'Preprocessor').

    Returns
    -------
    str
        Relative path to the main file, or empty string if not found.
    """
    comp_dir = model_dir / component
    if not comp_dir.exists():
        # Try case variations
        for d in model_dir.iterdir():
            if d.is_dir() and d.name.lower() == component.lower():
                comp_dir = d
                break
        else:
            return ""

    # Common main file patterns
    patterns = [
        f"{component}.in",
        f"{component}.dat",
        f"{component}_Main.in",
        f"{component}_Main.dat",
    ]

    for pattern in patterns:
        candidate = comp_dir / pattern
        if candidate.exists():
            return str(candidate.relative_to(model_dir))

    # Fall back to first .in or .dat file
    for ext in [".in", ".dat"]:
        candidates = list(comp_dir.glob(f"*{ext}"))
        if candidates:
            return str(candidates[0].relative_to(model_dir))

    return ""
