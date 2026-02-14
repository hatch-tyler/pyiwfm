"""Main PEST++ helper interface for IWFM models.

This module provides the IWFMPestHelper class - the primary high-level
interface for setting up PEST++ calibration, uncertainty analysis, and
optimization for IWFM models.

It coordinates all PEST++ components:
- Parameter management
- Observation management
- Template file generation
- Instruction file generation
- Geostatistics
- Control file writing
- Execution of PEST++ programs

Examples
--------
>>> helper = IWFMPestHelper(pest_dir="pest_setup", case_name="c2vsim_cal")
>>> helper.add_zone_parameters("hk", zones=[1, 2, 3], layer=1)
>>> helper.add_multiplier("pumping", bounds=(0.8, 1.2))
>>> helper.build()
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, TextIO

import numpy as np

from pyiwfm.runner.pest import (
    PESTInterface,
    TemplateFile,
    InstructionFile,
)
from pyiwfm.runner.pest_params import (
    IWFMParameterType,
    Parameter,
    ParameterGroup,
    ParameterTransform,
)
from pyiwfm.runner.pest_manager import IWFMParameterManager
from pyiwfm.runner.pest_observations import (
    IWFMObservationType,
    IWFMObservation,
    IWFMObservationGroup,
    WeightStrategy,
)
from pyiwfm.runner.pest_obs_manager import IWFMObservationManager
from pyiwfm.runner.pest_templates import IWFMTemplateManager
from pyiwfm.runner.pest_instructions import IWFMInstructionManager
from pyiwfm.runner.pest_geostat import (
    Variogram,
    VariogramType,
    GeostatManager,
)


class RegularizationType(Enum):
    """Types of regularization for PEST++."""

    PREFERRED_HOMOGENEITY = "preferred_homogeneity"
    PREFERRED_VALUE = "preferred_value"
    TIKHONOV = "tikhonov"
    NONE = "none"


@dataclass
class SVDConfig:
    """SVD configuration for PEST++.

    Parameters
    ----------
    maxsing : int
        Maximum number of singular values to use.
    eigthresh : float
        Eigenvalue ratio threshold for truncation.
    """

    maxsing: int = 100
    eigthresh: float = 1e-6

    def to_dict(self) -> dict[str, Any]:
        """Convert to PEST++ options dict."""
        return {
            "svd_pack": "redsvd",
            "max_n_super": self.maxsing,
            "eigthresh": self.eigthresh,
        }


@dataclass
class RegularizationConfig:
    """Regularization configuration.

    Parameters
    ----------
    reg_type : RegularizationType
        Type of regularization.
    weight : float
        Regularization weight multiplier.
    preferred_value : float | None
        Preferred parameter value (for preferred_value type).
    """

    reg_type: RegularizationType = RegularizationType.PREFERRED_HOMOGENEITY
    weight: float = 1.0
    preferred_value: float | None = None


class IWFMPestHelper:
    """Main interface for IWFM PEST++ setup.

    This class provides a high-level interface for setting up PEST++
    calibration, uncertainty analysis, and optimization for IWFM models.
    It coordinates parameter management, observation management,
    template/instruction generation, geostatistics, and control file
    writing.

    Parameters
    ----------
    pest_dir : Path | str
        Directory for PEST++ files.
    case_name : str
        Base name for PEST++ files (e.g., "iwfm_cal" -> iwfm_cal.pst).
    model_dir : Path | str | None
        Directory containing the IWFM model. If None, uses pest_dir.
    model : Any
        IWFM model instance (optional, for mesh/structure queries).

    Examples
    --------
    >>> helper = IWFMPestHelper(pest_dir="pest_setup", case_name="c2vsim_cal")
    >>> helper.add_zone_parameters("hk", zones=[1, 2, 3], layer=1)
    >>> helper.add_head_observations(wells, head_data)
    >>> helper.build()
    """

    def __init__(
        self,
        pest_dir: Path | str,
        case_name: str = "iwfm_cal",
        model_dir: Path | str | None = None,
        model: Any = None,
    ):
        """Initialize the PEST++ helper.

        Parameters
        ----------
        pest_dir : Path | str
            Directory for PEST++ files.
        case_name : str
            Base name for PEST++ files.
        model_dir : Path | str | None
            Directory containing the IWFM model.
        model : Any
            IWFM model instance.
        """
        self.pest_dir = Path(pest_dir)
        self.pest_dir.mkdir(parents=True, exist_ok=True)
        self.case_name = case_name
        self.model_dir = Path(model_dir) if model_dir else self.pest_dir
        self.model = model

        # Initialize managers
        self.parameters = IWFMParameterManager()
        self.observations = IWFMObservationManager()
        self.geostat = GeostatManager(model=model)
        self.templates = IWFMTemplateManager(
            parameter_manager=self.parameters,
            output_dir=self.pest_dir / "templates",
        )
        self.instructions = IWFMInstructionManager(
            observation_manager=self.observations,
            output_dir=self.pest_dir / "instructions",
        )

        # Configuration
        self._svd_config: SVDConfig | None = None
        self._regularization: RegularizationConfig | None = None
        self._pestpp_options: dict[str, Any] = {}
        self._model_command: str = "python forward_run.py"
        self._prior_info: list[str] = []

        # Build artifacts tracking
        self._built_templates: list[TemplateFile] = []
        self._built_instructions: list[InstructionFile] = []
        self._is_built: bool = False

    # --- Convenient parameter methods ---

    def add_zone_parameters(
        self,
        param_type: str | IWFMParameterType,
        zones: list[int],
        layer: int | None = None,
        initial_value: float = 1.0,
        bounds: tuple[float, float] = (0.01, 100.0),
        transform: str = "log",
        group: str | None = None,
    ) -> list[Parameter]:
        """Add zone-based parameters.

        Creates one parameter per zone for the specified property type.

        Parameters
        ----------
        param_type : str | IWFMParameterType
            Parameter type (e.g., "hk", IWFMParameterType.HORIZONTAL_K).
        zones : list[int]
            Zone IDs.
        layer : int | None
            Model layer.
        initial_value : float
            Initial value for all zone parameters.
        bounds : tuple[float, float]
            (lower_bound, upper_bound).
        transform : str
            Parameter transform: "log", "none", "fixed".
        group : str | None
            Parameter group name. Auto-generated if None.

        Returns
        -------
        list[Parameter]
            Created parameters.
        """
        if isinstance(param_type, str):
            param_type = IWFMParameterType(param_type)

        return self.parameters.add_zone_parameters(
            param_type=param_type,
            zones=zones,
            layer=layer,
            initial_values=initial_value,
            bounds=bounds,
            transform=transform,
            group=group,
        )

    def add_pilot_points(
        self,
        param_type: str | IWFMParameterType,
        points: list[tuple[float, float]],
        layer: int = 1,
        initial_value: float = 1.0,
        bounds: tuple[float, float] = (0.01, 100.0),
        variogram: Variogram | dict | None = None,
        transform: str = "log",
        prefix: str | None = None,
    ) -> list[Parameter]:
        """Add pilot point parameters.

        Parameters
        ----------
        param_type : str | IWFMParameterType
            Parameter type.
        points : list[tuple[float, float]]
            Pilot point (x, y) coordinates.
        layer : int
            Model layer.
        initial_value : float
            Initial value.
        bounds : tuple[float, float]
            (lower_bound, upper_bound).
        variogram : Variogram | dict | None
            Variogram for kriging. If dict, creates Variogram from it.
        transform : str
            Parameter transform.
        prefix : str | None
            Name prefix.

        Returns
        -------
        list[Parameter]
            Created parameters.
        """
        if isinstance(param_type, str):
            param_type = IWFMParameterType(param_type)

        if isinstance(variogram, dict):
            variogram = Variogram.from_dict(variogram)

        return self.parameters.add_pilot_points(
            param_type=param_type,
            points=points,
            layer=layer,
            initial_value=initial_value,
            bounds=bounds,
            transform=transform,
            prefix=prefix,
        )

    def add_multiplier(
        self,
        param_type: str | IWFMParameterType,
        spatial: str = "global",
        zones: list[int] | None = None,
        initial_value: float = 1.0,
        bounds: tuple[float, float] = (0.5, 2.0),
        transform: str = "none",
    ) -> list[Parameter]:
        """Add multiplier parameters.

        Parameters
        ----------
        param_type : str | IWFMParameterType
            Parameter type.
        spatial : str
            Spatial extent: "global", "zone".
        zones : list[int] | None
            Zone IDs (required if spatial="zone").
        initial_value : float
            Initial multiplier value.
        bounds : tuple[float, float]
            (lower_bound, upper_bound).
        transform : str
            Parameter transform.

        Returns
        -------
        list[Parameter]
            Created parameters.
        """
        if isinstance(param_type, str):
            param_type = IWFMParameterType(param_type)

        return self.parameters.add_multiplier_parameters(
            param_type=param_type,
            spatial=spatial,
            zones=zones,
            initial_value=initial_value,
            bounds=bounds,
            transform=transform,
        )

    def add_stream_parameters(
        self,
        param_type: str | IWFMParameterType,
        reaches: list[int],
        initial_value: float = 1.0,
        bounds: tuple[float, float] = (0.001, 10.0),
        transform: str = "log",
    ) -> list[Parameter]:
        """Add stream-related parameters.

        Parameters
        ----------
        param_type : str | IWFMParameterType
            Parameter type (e.g., STREAMBED_K).
        reaches : list[int]
            Stream reach IDs.
        initial_value : float
            Initial value.
        bounds : tuple[float, float]
            (lower_bound, upper_bound).
        transform : str
            Parameter transform.

        Returns
        -------
        list[Parameter]
            Created parameters.
        """
        if isinstance(param_type, str):
            param_type = IWFMParameterType(param_type)

        return self.parameters.add_stream_parameters(
            param_type=param_type,
            reaches=reaches,
            initial_values=initial_value,
            bounds=bounds,
            transform=transform,
        )

    def add_rootzone_parameters(
        self,
        param_type: str | IWFMParameterType,
        land_use_types: list[str],
        initial_value: float = 1.0,
        bounds: tuple[float, float] = (0.5, 1.5),
        transform: str = "none",
    ) -> list[Parameter]:
        """Add root zone parameters by land use type.

        Parameters
        ----------
        param_type : str | IWFMParameterType
            Parameter type (e.g., CROP_COEFFICIENT).
        land_use_types : list[str]
            Land use type names.
        initial_value : float
            Initial value.
        bounds : tuple[float, float]
            (lower_bound, upper_bound).
        transform : str
            Parameter transform.

        Returns
        -------
        list[Parameter]
            Created parameters.
        """
        if isinstance(param_type, str):
            param_type = IWFMParameterType(param_type)

        return self.parameters.add_rootzone_parameters(
            param_type=param_type,
            land_use_types=land_use_types,
            initial_values=initial_value,
            bounds=bounds,
            transform=transform,
        )

    # --- Convenient observation methods ---

    def add_head_observations(
        self,
        well_id: str,
        x: float,
        y: float,
        times: list[datetime],
        values: list[float],
        layer: int = 1,
        weight: float = 1.0,
        group: str | None = None,
    ) -> list[IWFMObservation]:
        """Add groundwater head observations for a well.

        This is a simplified convenience method that creates observations
        directly. For more control (DataFrames, weight strategies, etc.),
        use ``self.observations.add_head_observations()`` with WellInfo objects.

        Parameters
        ----------
        well_id : str
            Well identifier.
        x, y : float
            Well coordinates.
        times : list[datetime]
            Observation timestamps.
        values : list[float]
            Observed head values.
        layer : int
            Model layer.
        weight : float
            Observation weight.
        group : str | None
            Observation group name.

        Returns
        -------
        list[IWFMObservation]
            Created observations.
        """
        from pyiwfm.runner.pest_observations import ObservationLocation

        grp_name = group or f"head_{well_id[:8]}"

        # Ensure group exists
        if grp_name not in self.observations._observation_groups:
            self.observations._observation_groups[grp_name] = IWFMObservationGroup(
                name=grp_name,
                obs_type=IWFMObservationType.HEAD,
            )

        created = []
        location = ObservationLocation(x=x, y=y, layer=layer)

        for t, v in zip(times, values):
            date_str = t.strftime("%Y%m%d")
            obs_name = f"{well_id[:12]}_{date_str}"
            # Ensure unique name
            obs_name = self.observations._make_valid_obs_name(obs_name)

            obs = IWFMObservation(
                name=obs_name,
                value=float(v),
                weight=weight,
                group=grp_name,
                obs_type=IWFMObservationType.HEAD,
                datetime=t,
                location=location,
                metadata={"well_id": well_id},
            )
            self.observations._observations[obs_name] = obs
            self.observations._observation_groups[grp_name].observations.append(obs)
            created.append(obs)

        return created

    def add_streamflow_observations(
        self,
        gage_id: str,
        reach_id: int,
        times: list[datetime],
        values: list[float],
        weight: float = 1.0,
        transform: str = "none",
        group: str | None = None,
    ) -> list[IWFMObservation]:
        """Add streamflow observations for a gage.

        This is a simplified convenience method. For more control,
        use ``self.observations.add_streamflow_observations()`` with
        GageInfo objects.

        Parameters
        ----------
        gage_id : str
            Gage identifier.
        reach_id : int
            Stream reach ID.
        times : list[datetime]
            Observation timestamps.
        values : list[float]
            Observed flow values.
        weight : float
            Observation weight.
        transform : str
            Transform: "none", "log".
        group : str | None
            Observation group name.

        Returns
        -------
        list[IWFMObservation]
            Created observations.
        """
        grp_name = group or f"flow_{gage_id[:8]}"

        if grp_name not in self.observations._observation_groups:
            self.observations._observation_groups[grp_name] = IWFMObservationGroup(
                name=grp_name,
                obs_type=IWFMObservationType.STREAM_FLOW,
            )

        created = []
        for t, v in zip(times, values):
            date_str = t.strftime("%Y%m%d")
            obs_name = f"{gage_id[:12]}_{date_str}"
            obs_name = self.observations._make_valid_obs_name(obs_name)

            obs_value = float(v)
            obs = IWFMObservation(
                name=obs_name,
                value=obs_value,
                weight=weight,
                group=grp_name,
                obs_type=IWFMObservationType.STREAM_FLOW,
                datetime=t,
                metadata={"gage_id": gage_id, "reach_id": reach_id},
            )
            self.observations._observations[obs_name] = obs
            self.observations._observation_groups[grp_name].observations.append(obs)
            created.append(obs)

        return created

    # --- Configuration methods ---

    def set_svd(
        self,
        maxsing: int = 100,
        eigthresh: float = 1e-6,
    ) -> None:
        """Configure SVD truncation for parameter estimation.

        Parameters
        ----------
        maxsing : int
            Maximum number of singular values.
        eigthresh : float
            Eigenvalue ratio threshold.
        """
        self._svd_config = SVDConfig(maxsing=maxsing, eigthresh=eigthresh)

    def set_regularization(
        self,
        reg_type: str = "preferred_homogeneity",
        weight: float = 1.0,
        preferred_value: float | None = None,
    ) -> None:
        """Configure regularization for pilot points.

        Parameters
        ----------
        reg_type : str
            Type: "preferred_homogeneity", "preferred_value", "tikhonov".
        weight : float
            Regularization weight multiplier.
        preferred_value : float | None
            Preferred parameter value.
        """
        self._regularization = RegularizationConfig(
            reg_type=RegularizationType(reg_type),
            weight=weight,
            preferred_value=preferred_value,
        )

    def set_model_command(self, command: str) -> None:
        """Set the model forward run command.

        Parameters
        ----------
        command : str
            Command string that PEST++ will execute.
        """
        self._model_command = command

    def set_pestpp_options(self, **options: Any) -> None:
        """Set PEST++ specific options.

        Parameters
        ----------
        **options : Any
            Option name-value pairs (e.g., ies_num_reals=100).
        """
        self._pestpp_options.update(options)

    def get_pestpp_option(self, key: str, default: Any = None) -> Any:
        """Get a PEST++ option value.

        Parameters
        ----------
        key : str
            Option name.
        default : Any
            Default value if not set.

        Returns
        -------
        Any
            Option value.
        """
        return self._pestpp_options.get(key, default)

    def balance_observation_weights(
        self,
        contributions: dict[str, float] | None = None,
    ) -> None:
        """Balance weights across observation groups.

        Parameters
        ----------
        contributions : dict[str, float] | None
            Target contributions per group (e.g., {"head": 0.5, "flow": 0.5}).
            If None, equalizes contributions.
        """
        self.observations.balance_observation_groups(contributions)

    # --- Build methods ---

    def build(
        self,
        pst_file: Path | str | None = None,
    ) -> Path:
        """Build complete PEST++ setup.

        Creates:
        - Control file (.pst)
        - Template files (.tpl)
        - Instruction files (.ins)
        - Forward run script

        Parameters
        ----------
        pst_file : Path | str | None
            Path for control file. Defaults to pest_dir/case_name.pst.

        Returns
        -------
        Path
            Path to the control file.

        Raises
        ------
        ValueError
            If no parameters or observations are defined.
        """
        if self.parameters.n_parameters == 0:
            raise ValueError("No parameters defined. Add parameters before building.")

        if self.observations.n_observations == 0:
            raise ValueError("No observations defined. Add observations before building.")

        if pst_file is None:
            pst_file = self.pest_dir / f"{self.case_name}.pst"
        else:
            pst_file = Path(pst_file)

        # Create subdirectories
        (self.pest_dir / "templates").mkdir(exist_ok=True)
        (self.pest_dir / "instructions").mkdir(exist_ok=True)

        # Build the PEST interface
        pest = PESTInterface(
            model_dir=self.model_dir,
            pest_dir=self.pest_dir,
            case_name=self.case_name,
        )

        # Add parameters
        for param in self.parameters.get_all_parameters():
            pest.add_parameter(
                name=param.name,
                initial_value=param.initial_value,
                lower_bound=param.lower_bound,
                upper_bound=param.upper_bound,
                group=param.group or "default",
                transform=param.transform or "none",
            )

        # Add observations
        for obs in self.observations.get_all_observations():
            pest.add_observation(
                name=obs.name,
                value=obs.value,
                weight=obs.weight,
                group=obs.group or "default",
            )

        # Add template files
        for tpl in self._built_templates:
            pest.add_template_file(tpl)

        # Add instruction files
        for ins in self._built_instructions:
            pest.add_instruction_file(ins)

        # Set model command
        pest.set_model_command(self._model_command)

        # Apply SVD config
        if self._svd_config:
            for key, val in self._svd_config.to_dict().items():
                pest.set_pestpp_option(key, val)

        # Apply regularization
        if self._regularization:
            if self._regularization.reg_type == RegularizationType.PREFERRED_HOMOGENEITY:
                pest.set_pestpp_option("ies_reg_factor", self._regularization.weight)
            elif self._regularization.reg_type == RegularizationType.TIKHONOV:
                pest.set_pestpp_option("use_regul_prior", "true")

        # Apply custom PEST++ options
        for key, val in self._pestpp_options.items():
            pest.set_pestpp_option(key, val)

        # Write control file
        control_path = pest.write_control_file(pst_file)

        # Write forward run script
        self.write_forward_run_script()

        self._is_built = True
        return control_path

    def add_template(self, template: TemplateFile) -> None:
        """Register a template file for the build.

        Parameters
        ----------
        template : TemplateFile
            Template file to register.
        """
        self._built_templates.append(template)

    def add_instruction(self, instruction: InstructionFile) -> None:
        """Register an instruction file for the build.

        Parameters
        ----------
        instruction : InstructionFile
            Instruction file to register.
        """
        self._built_instructions.append(instruction)

    def write_forward_run_script(
        self,
        filepath: Path | str | None = None,
    ) -> Path:
        """Write script that runs IWFM for PEST++.

        Parameters
        ----------
        filepath : Path | str | None
            Output path. Defaults to pest_dir/forward_run.py.

        Returns
        -------
        Path
            Path to the written script.
        """
        if filepath is None:
            filepath = self.pest_dir / "forward_run.py"
        else:
            filepath = Path(filepath)

        script = f'''#!/usr/bin/env python
"""PEST++ forward model runner for IWFM.

Auto-generated by IWFMPestHelper.
Case: {self.case_name}
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

import sys
import os
from pathlib import Path

def main():
    """Run IWFM forward model."""
    model_dir = Path(r"{self.model_dir}")

    # Change to model directory
    os.chdir(model_dir)

    # Run IWFM simulation
    try:
        from pyiwfm.runner import IWFMRunner
        runner = IWFMRunner()
        result = runner.run_simulation(model_dir)

        if not result.success:
            print(f"Simulation failed: {{result.errors}}", file=sys.stderr)
            sys.exit(1)

        sys.exit(0)
    except ImportError:
        # Fall back to direct executable run
        import subprocess
        exe = model_dir / "Simulation_x64.exe"
        if exe.exists():
            proc = subprocess.run([str(exe)], cwd=str(model_dir))
            sys.exit(proc.returncode)
        else:
            print("No IWFM executable found", file=sys.stderr)
            sys.exit(1)

if __name__ == "__main__":
    main()
'''

        filepath.write_text(script)
        return filepath

    def write_pp_interpolation_script(
        self,
        filepath: Path | str | None = None,
    ) -> Path:
        """Write pilot point interpolation script.

        This script is run before the model to interpolate pilot
        point values to model nodes/elements using kriging.

        Parameters
        ----------
        filepath : Path | str | None
            Output path.

        Returns
        -------
        Path
            Path to the written script.
        """
        if filepath is None:
            filepath = self.pest_dir / "interpolate_pp.py"
        else:
            filepath = Path(filepath)

        script = f'''#!/usr/bin/env python
"""Pilot point interpolation script.

Auto-generated by IWFMPestHelper.
Case: {self.case_name}
"""

import sys
import numpy as np
from pathlib import Path

def main():
    """Interpolate pilot point values to model nodes."""
    pest_dir = Path(r"{self.pest_dir}")

    # Read pilot point values from template-generated files
    pp_dir = pest_dir / "pilot_points"
    if not pp_dir.exists():
        print("No pilot point directory found")
        sys.exit(0)

    # Load kriging factors and apply
    for factors_file in pp_dir.glob("pp_factors_*.dat"):
        apply_factors(factors_file)

def apply_factors(factors_file):
    """Apply kriging factors to interpolate pilot point values."""
    # Read factors file
    # Format: target_id n_contributors
    #           pp_name  weight
    lines = factors_file.read_text().strip().split("\\n")
    # Implementation depends on IWFM file format
    pass

if __name__ == "__main__":
    main()
'''

        filepath.write_text(script)
        return filepath

    # --- Execution methods ---

    def run_pestpp(
        self,
        program: str = "pestpp-glm",
        n_workers: int = 1,
        extra_args: list[str] | None = None,
    ) -> subprocess.CompletedProcess:
        """Run a PEST++ program.

        Parameters
        ----------
        program : str
            PEST++ program: "pestpp-glm", "pestpp-ies", "pestpp-sen", etc.
        n_workers : int
            Number of parallel workers.
        extra_args : list[str] | None
            Additional command-line arguments.

        Returns
        -------
        subprocess.CompletedProcess
            Process result.

        Raises
        ------
        FileNotFoundError
            If the PEST++ executable or control file is not found.
        RuntimeError
            If the setup has not been built yet.
        """
        if not self._is_built:
            raise RuntimeError("Must call build() before running PEST++")

        pst_file = self.pest_dir / f"{self.case_name}.pst"
        if not pst_file.exists():
            raise FileNotFoundError(f"Control file not found: {pst_file}")

        # Check for executable
        exe = shutil.which(program)
        if exe is None:
            raise FileNotFoundError(
                f"PEST++ executable not found: {program}. "
                f"Ensure PEST++ is installed and on PATH."
            )

        cmd = [exe, str(pst_file)]
        if extra_args:
            cmd.extend(extra_args)

        return subprocess.run(
            cmd,
            cwd=str(self.pest_dir),
            capture_output=True,
            text=True,
        )

    def run_pestpp_glm(self, n_workers: int = 1, **kwargs: Any) -> subprocess.CompletedProcess:
        """Run pestpp-glm for parameter estimation.

        Parameters
        ----------
        n_workers : int
            Number of parallel workers.
        **kwargs : Any
            Additional PEST++ options set before running.

        Returns
        -------
        subprocess.CompletedProcess
            Process result.
        """
        if kwargs:
            self.set_pestpp_options(**kwargs)
            self.build()  # Rebuild with new options

        return self.run_pestpp("pestpp-glm", n_workers=n_workers)

    def run_pestpp_ies(
        self,
        n_realizations: int = 100,
        n_workers: int = 1,
        **kwargs: Any,
    ) -> subprocess.CompletedProcess:
        """Run pestpp-ies for ensemble calibration.

        Parameters
        ----------
        n_realizations : int
            Number of ensemble members.
        n_workers : int
            Number of parallel workers.
        **kwargs : Any
            Additional PEST++ options.

        Returns
        -------
        subprocess.CompletedProcess
            Process result.
        """
        self.set_pestpp_options(ies_num_reals=n_realizations, **kwargs)
        self.build()
        return self.run_pestpp("pestpp-ies", n_workers=n_workers)

    def run_pestpp_sen(
        self,
        method: str = "sobol",
        n_samples: int = 1000,
        **kwargs: Any,
    ) -> subprocess.CompletedProcess:
        """Run pestpp-sen for sensitivity analysis.

        Parameters
        ----------
        method : str
            Sensitivity method: "sobol", "morris".
        n_samples : int
            Number of samples.
        **kwargs : Any
            Additional PEST++ options.

        Returns
        -------
        subprocess.CompletedProcess
            Process result.
        """
        self.set_pestpp_options(
            gsa_method=method,
            gsa_sobol_samples=n_samples,
            **kwargs,
        )
        self.build()
        return self.run_pestpp("pestpp-sen", n_workers=1)

    # --- Query methods ---

    @property
    def n_parameters(self) -> int:
        """Number of defined parameters."""
        return self.parameters.n_parameters

    @property
    def n_observations(self) -> int:
        """Number of defined observations."""
        return self.observations.n_observations

    @property
    def parameter_groups(self) -> list[str]:
        """List of parameter group names."""
        return [g.name for g in self.parameters.get_all_groups()]

    @property
    def observation_groups(self) -> list[str]:
        """List of observation group names."""
        return [g.name for g in self.observations.get_all_groups()]

    def summary(self) -> dict[str, Any]:
        """Get a summary of the PEST++ setup.

        Returns
        -------
        dict[str, Any]
            Summary information.
        """
        return {
            "case_name": self.case_name,
            "pest_dir": str(self.pest_dir),
            "model_dir": str(self.model_dir),
            "n_parameters": self.n_parameters,
            "n_observations": self.n_observations,
            "n_parameter_groups": len(self.parameter_groups),
            "n_observation_groups": len(self.observation_groups),
            "n_templates": len(self._built_templates),
            "n_instructions": len(self._built_instructions),
            "svd_configured": self._svd_config is not None,
            "regularization": (
                self._regularization.reg_type.value
                if self._regularization
                else "none"
            ),
            "is_built": self._is_built,
            "pestpp_options": dict(self._pestpp_options),
        }

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"IWFMPestHelper(case_name='{self.case_name}', "
            f"n_params={self.n_parameters}, "
            f"n_obs={self.n_observations}, "
            f"built={self._is_built})"
        )
