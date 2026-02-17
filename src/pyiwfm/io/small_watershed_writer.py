"""
Small Watershed Component Writer for IWFM models.

This module provides the writer for IWFM small watershed component files,
generating the main file with geospatial data, root zone parameters,
and aquifer parameters for all watershed units.
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
    from pyiwfm.components.small_watershed import AppSmallWatershed
    from pyiwfm.core.model import IWFMModel

logger = logging.getLogger(__name__)


@dataclass
class SmallWatershedWriterConfig:
    """Configuration for small watershed component file writing.

    Attributes:
        output_dir: Base output directory
        swshed_subdir: Subdirectory name for small watershed files
        version: IWFM component version
        main_file: Main file name
        budget_file: Budget output file path
        final_results_file: Final results output file path
    """

    output_dir: Path
    swshed_subdir: str = "SmallWatershed"
    version: str = "4.0"

    # File names
    main_file: str = "SmallWatershed_MAIN.dat"

    # Output files (optional)
    budget_file: str = "../Results/SWShedBud.hdf"
    final_results_file: str = "../Results/FinalSWShed.out"

    @property
    def swshed_dir(self) -> Path:
        """Get the small watershed subdirectory path."""
        if self.swshed_subdir:
            return self.output_dir / self.swshed_subdir
        return self.output_dir

    @property
    def main_path(self) -> Path:
        """Get the main file path."""
        return self.swshed_dir / self.main_file


class SmallWatershedComponentWriter(TemplateWriter):
    """Writer for IWFM Small Watershed Component files.

    Writes the small watershed main file including geospatial data,
    root zone parameters, and aquifer parameters.

    Example:
        >>> from pyiwfm.io.small_watershed_writer import (
        ...     SmallWatershedComponentWriter, SmallWatershedWriterConfig,
        ... )
        >>> config = SmallWatershedWriterConfig(output_dir=Path("model/Simulation"))
        >>> writer = SmallWatershedComponentWriter(model, config)
        >>> files = writer.write_all()
    """

    def __init__(
        self,
        model: IWFMModel,
        config: SmallWatershedWriterConfig,
        template_engine: TemplateEngine | None = None,
    ) -> None:
        super().__init__(config.output_dir, template_engine)
        self.model = model
        self.config = config

    @property
    def format(self) -> str:
        return "iwfm_small_watershed"

    def write(self, data: Any = None) -> None:
        """Write all small watershed files."""
        self.write_all()

    def write_all(self, write_defaults: bool = True) -> dict[str, Path]:
        """Write all small watershed component files.

        Parameters
        ----------
        write_defaults : bool
            If True, write default files even when no component is loaded.

        Returns
        -------
        dict[str, Path]
            Mapping of file type to output path.
        """
        logger.info(f"Writing small watershed files to {self.config.swshed_dir}")

        self.config.swshed_dir.mkdir(parents=True, exist_ok=True)

        results: dict[str, Path] = {}

        sw = self.model.small_watersheds
        if sw is None and not write_defaults:
            logger.warning("No small watershed component in model and write_defaults=False")
            return results

        results["main"] = self.write_main()

        logger.info(f"Wrote {len(results)} small watershed files")
        return results

    def write_main(self) -> Path:
        """Write the main small watershed control file.

        Returns
        -------
        Path
            Path to written file.
        """
        output_path = self.config.main_path
        self._ensure_dir(output_path)

        sw = self.model.small_watersheds

        if sw is not None and sw.n_watersheds > 0:
            ws_list = list(sw.iter_watersheds())
            n_watersheds = len(ws_list)
        else:
            ws_list = []
            n_watersheds = 0

        content = self._render_main(sw, ws_list, n_watersheds)
        output_path.write_text(content)
        logger.info(f"Wrote small watershed main file: {output_path}")
        return output_path

    def _render_main(
        self,
        sw: AppSmallWatershed | None,
        ws_list: list,
        n_watersheds: int,
    ) -> str:
        """Render the main small watershed file using Jinja2 template."""
        generation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Build watershed data for template
        ws_data = []
        area_fac = sw.area_factor if sw and sw.area_factor else 1.0
        ic_fac = sw.ic_factor if sw and sw.ic_factor else 1.0
        for ws in ws_list:
            # Compute raw (un-factored) values for writing
            # The reader applies factors on read, so we reverse them for write
            rz_length = sw.rz_length_factor if sw and sw.rz_length_factor else 1.0
            rz_k = sw.rz_k_factor if sw and sw.rz_k_factor else 1.0
            rz_cn = sw.rz_cn_factor if sw and sw.rz_cn_factor else 1.0
            aq_gw = sw.aq_gw_factor if sw and sw.aq_gw_factor else 1.0
            aq_time = sw.aq_time_factor if sw and sw.aq_time_factor else 1.0

            gw_nodes_data = []
            for gn in ws.gw_nodes:
                raw_rate = gn.max_perc_rate
                if gn.is_baseflow:
                    raw_rate = -float(gn.layer)
                gw_nodes_data.append(
                    {
                        "gw_node_id": gn.gw_node_id,
                        "max_perc_rate": gn.max_perc_rate,
                        "is_baseflow": gn.is_baseflow,
                        "layer": gn.layer,
                        "perc_rate_raw": raw_rate,
                    }
                )

            entry = {
                "id": ws.id,
                "area": ws.area,
                "area_raw": ws.area / area_fac if area_fac else ws.area,
                "dest_stream_node": ws.dest_stream_node,
                "n_gw_nodes": ws.n_gw_nodes,
                "gw_nodes": gw_nodes_data,
                "precip_col": ws.precip_col,
                "precip_factor": ws.precip_factor,
                "et_col": ws.et_col,
                "wilting_point": ws.wilting_point,
                "field_capacity": ws.field_capacity,
                "total_porosity": ws.total_porosity,
                "lambda_param": ws.lambda_param,
                "kunsat_method": ws.kunsat_method,
                # Reverse the conversion factors for writing
                "root_depth_raw": (ws.root_depth / rz_length if rz_length else ws.root_depth),
                "hydraulic_cond_raw": (ws.hydraulic_cond / rz_k if rz_k else ws.hydraulic_cond),
                "curve_number_raw": (ws.curve_number / rz_cn if rz_cn else ws.curve_number),
                "gw_threshold_raw": (ws.gw_threshold / aq_gw if aq_gw else ws.gw_threshold),
                "max_gw_storage_raw": (ws.max_gw_storage / aq_gw if aq_gw else ws.max_gw_storage),
                "surface_flow_coeff_raw": (
                    ws.surface_flow_coeff / aq_time if aq_time else ws.surface_flow_coeff
                ),
                "baseflow_coeff_raw": (
                    ws.baseflow_coeff / aq_time if aq_time else ws.baseflow_coeff
                ),
                # Initial conditions
                "initial_soil_moisture": ws.initial_soil_moisture,
                "initial_gw_storage_raw": (
                    ws.initial_gw_storage / ic_fac if ic_fac else ws.initial_gw_storage
                ),
            }
            ws_data.append(entry)

        # Use component data or defaults for factors
        budget_file = self.config.budget_file
        final_results_file = self.config.final_results_file
        if sw is not None:
            if sw.budget_output_file:
                budget_file = sw.budget_output_file
            if sw.final_results_file:
                final_results_file = sw.final_results_file

        context = {
            "version": self.config.version,
            "generation_time": generation_time,
            "budget_file": budget_file,
            "final_results_file": final_results_file,
            "n_watersheds": n_watersheds,
            "area_factor": sw.area_factor if sw else 1.0,
            "flow_factor": sw.flow_factor if sw else 1.0,
            "flow_time_unit": sw.flow_time_unit if sw else "1DAY",
            "watersheds": ws_data,
            "rz_solver_tolerance": sw.rz_solver_tolerance if sw else 1e-8,
            "rz_max_iterations": sw.rz_max_iterations if sw else 2000,
            "rz_length_factor": sw.rz_length_factor if sw else 1.0,
            "rz_cn_factor": sw.rz_cn_factor if sw else 1.0,
            "rz_k_factor": sw.rz_k_factor if sw else 1.0,
            "rz_k_time_unit": sw.rz_k_time_unit if sw else "1DAY",
            "aq_gw_factor": sw.aq_gw_factor if sw else 1.0,
            "aq_time_factor": sw.aq_time_factor if sw else 1.0,
            "aq_time_unit": sw.aq_time_unit if sw else "1DAY",
            "ic_factor": sw.ic_factor if sw else 1.0,
        }

        return self._engine.render_template("small_watershed/small_watershed_main.j2", **context)


def write_small_watershed_component(
    model: IWFMModel,
    output_dir: Path | str,
    config: SmallWatershedWriterConfig | None = None,
) -> dict[str, Path]:
    """Write small watershed component files for a model.

    Parameters
    ----------
    model : IWFMModel
        Model to write.
    output_dir : Path or str
        Output directory.
    config : SmallWatershedWriterConfig, optional
        File configuration.

    Returns
    -------
    dict[str, Path]
        Mapping of file type to output path.
    """
    output_dir = Path(output_dir)

    if config is None:
        config = SmallWatershedWriterConfig(output_dir=output_dir)
    else:
        config.output_dir = output_dir

    writer = SmallWatershedComponentWriter(model, config)
    return writer.write_all()
