"""
Unsaturated Zone Component Writer for IWFM models.

This module provides the writer for IWFM unsaturated zone component files,
generating the main file with solver parameters, per-element layer
properties, and initial soil moisture conditions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pyiwfm.io.writer_base import TemplateWriter
from pyiwfm.templates.engine import TemplateEngine

if TYPE_CHECKING:
    from pyiwfm.components.unsaturated_zone import AppUnsatZone
    from pyiwfm.core.model import IWFMModel

logger = logging.getLogger(__name__)


@dataclass
class UnsatZoneWriterConfig:
    """Configuration for unsaturated zone component file writing.

    Attributes:
        output_dir: Base output directory
        unsatzone_subdir: Subdirectory name for unsat zone files
        version: IWFM component version
        main_file: Main file name
        budget_file: Budget output file path
        zbudget_file: Zone budget output file path
        final_results_file: Final results output file path
    """

    output_dir: Path
    unsatzone_subdir: str = "UnsatZone"
    version: str = "4.0"

    # File names
    main_file: str = "UnsatZone_MAIN.dat"

    # Output files (optional)
    budget_file: str = "../Results/UZBud.hdf"
    zbudget_file: str = "../Results/UZZBud.hdf"
    final_results_file: str = "../Results/FinalUZ.out"

    @property
    def unsatzone_dir(self) -> Path:
        """Get the unsaturated zone subdirectory path."""
        if self.unsatzone_subdir:
            return self.output_dir / self.unsatzone_subdir
        return self.output_dir

    @property
    def main_path(self) -> Path:
        """Get the main file path."""
        return self.unsatzone_dir / self.main_file


class UnsatZoneComponentWriter(TemplateWriter):
    """Writer for IWFM Unsaturated Zone Component files.

    Writes the unsaturated zone main file including solver parameters,
    per-element layer properties, and initial soil moisture conditions.

    Example:
        >>> from pyiwfm.io.unsaturated_zone_writer import (
        ...     UnsatZoneComponentWriter, UnsatZoneWriterConfig,
        ... )
        >>> config = UnsatZoneWriterConfig(output_dir=Path("model/Simulation"))
        >>> writer = UnsatZoneComponentWriter(model, config)
        >>> files = writer.write_all()
    """

    def __init__(
        self,
        model: IWFMModel,
        config: UnsatZoneWriterConfig,
        template_engine: TemplateEngine | None = None,
    ) -> None:
        super().__init__(config.output_dir, template_engine)
        self.model = model
        self.config = config

    @property
    def format(self) -> str:
        return "iwfm_unsaturated_zone"

    def write(self, data: Any = None) -> None:
        """Write all unsaturated zone files."""
        self.write_all()

    def write_all(self, write_defaults: bool = True) -> dict[str, Path]:
        """Write all unsaturated zone component files.

        Parameters
        ----------
        write_defaults : bool
            If True, write default files even when no component is loaded.

        Returns
        -------
        dict[str, Path]
            Mapping of file type to output path.
        """
        logger.info(f"Writing unsaturated zone files to {self.config.unsatzone_dir}")

        self.config.unsatzone_dir.mkdir(parents=True, exist_ok=True)

        results: dict[str, Path] = {}

        uz = self.model.unsaturated_zone
        if uz is None and not write_defaults:
            logger.warning("No unsaturated zone component in model and write_defaults=False")
            return results

        results["main"] = self.write_main()

        logger.info(f"Wrote {len(results)} unsaturated zone files")
        return results

    def write_main(self) -> Path:
        """Write the main unsaturated zone control file.

        Returns
        -------
        Path
            Path to written file.
        """
        output_path = self.config.main_path
        self._ensure_dir(output_path)

        uz = self.model.unsaturated_zone
        content = self._render_main(uz)
        output_path.write_text(content)
        logger.info(f"Wrote unsaturated zone main file: {output_path}")
        return output_path

    def _render_main(self, uz: AppUnsatZone | None) -> str:
        """Render the main unsaturated zone file using Jinja2 template."""
        generation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        n_layers = uz.n_layers if uz else 0

        # Build element data for template
        element_data = []
        initial_conditions = []
        uniform_moisture = None

        if uz is not None and uz.n_elements > 0:
            for elem in uz.iter_elements():
                layers_data = []
                for layer in elem.layers:
                    # Reverse conversion factors for writing raw values
                    thickness_raw = (
                        layer.thickness_max / uz.thickness_factor
                        if uz.thickness_factor
                        else layer.thickness_max
                    )
                    hyd_cond_raw = (
                        layer.hyd_cond / uz.hyd_cond_factor
                        if uz.hyd_cond_factor
                        else layer.hyd_cond
                    )
                    layers_data.append(
                        {
                            "thickness_max": thickness_raw,
                            "total_porosity": layer.total_porosity,
                            "lambda_param": layer.lambda_param,
                            "hyd_cond": hyd_cond_raw,
                            "kunsat_method": layer.kunsat_method,
                        }
                    )
                element_data.append(
                    {
                        "element_id": elem.element_id,
                        "layers": layers_data,
                    }
                )

                if elem.initial_moisture is not None:
                    initial_conditions.append(
                        {
                            "element_id": elem.element_id,
                            "moisture": elem.initial_moisture.tolist(),
                        }
                    )

            # Check if all elements share the same initial moisture (uniform)
            if len(initial_conditions) == 1 and initial_conditions[0]["element_id"] == 0:
                uniform_moisture = initial_conditions[0]["moisture"]
                initial_conditions = []

        budget_file = self.config.budget_file
        zbudget_file = self.config.zbudget_file
        final_results_file = self.config.final_results_file
        if uz is not None:
            if uz.budget_file:
                budget_file = uz.budget_file
            if uz.zbudget_file:
                zbudget_file = uz.zbudget_file
            if uz.final_results_file:
                final_results_file = uz.final_results_file

        context = {
            "version": self.config.version,
            "generation_time": generation_time,
            "n_layers": n_layers,
            "solver_tolerance": uz.solver_tolerance if uz else 1e-8,
            "max_iterations": uz.max_iterations if uz else 2000,
            "budget_file": budget_file,
            "zbudget_file": zbudget_file,
            "final_results_file": final_results_file,
            "n_parametric_grids": uz.n_parametric_grids if uz else 0,
            "coord_factor": uz.coord_factor if uz else 1.0,
            "thickness_factor": uz.thickness_factor if uz else 1.0,
            "hyd_cond_factor": uz.hyd_cond_factor if uz else 1.0,
            "time_unit": uz.time_unit if uz else "1DAY",
            "element_data": element_data,
            "uniform_moisture": uniform_moisture,
            "initial_conditions": initial_conditions,
        }

        return self._engine.render_template("unsaturated_zone/unsaturated_zone_main.j2", **context)


def write_unsaturated_zone_component(
    model: IWFMModel,
    output_dir: Path | str,
    config: UnsatZoneWriterConfig | None = None,
) -> dict[str, Path]:
    """Write unsaturated zone component files for a model.

    Parameters
    ----------
    model : IWFMModel
        Model to write.
    output_dir : Path or str
        Output directory.
    config : UnsatZoneWriterConfig, optional
        File configuration.

    Returns
    -------
    dict[str, Path]
        Mapping of file type to output path.
    """
    output_dir = Path(output_dir)

    if config is None:
        config = UnsatZoneWriterConfig(output_dir=output_dir)
    else:
        config.output_dir = output_dir

    writer = UnsatZoneComponentWriter(model, config)
    return writer.write_all()
