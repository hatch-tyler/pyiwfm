"""
Lake Component Writer for IWFM models.

This module provides the main writer for IWFM lake component files,
orchestrating the writing of all lake-related input files including:
- Main lake control file (Lake_MAIN.dat) for v4.0 and v5.0
- Maximum lake elevation time series (v4.0)
- Lake budget output configuration
- v5.0 outflow rating tables
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from numpy.typing import NDArray

from pyiwfm.io.writer_base import TemplateWriter
from pyiwfm.templates.engine import TemplateEngine

if TYPE_CHECKING:
    from pyiwfm.components.lake import AppLake
    from pyiwfm.core.model import IWFMModel

logger = logging.getLogger(__name__)


def _parse_lake_version(version_str: str) -> tuple[int, int]:
    """Parse version string like '4.0' or '5.0' into a tuple."""
    parts = version_str.split(".")
    return (int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)


@dataclass
class LakeWriterConfig:
    """
    Configuration for lake component file writing.

    Attributes
    ----------
    output_dir : Path
        Base output directory for lake files
    lake_subdir : str
        Subdirectory name for lake files (default: "Lake")
    version : str
        IWFM lake component version
    """

    output_dir: Path
    lake_subdir: str = "Lake"
    version: str = "4.0"

    # File names
    main_file: str = "Lake_MAIN.dat"
    max_elev_file: str = "MaxLakeElev.dat"

    # Output files (optional)
    lake_budget_file: str = "../Results/LakeBud.hdf"
    final_elev_file: str = "../Results/FinalLakeElev.out"

    # Unit conversions
    conductivity_factor: float = 1.0
    conductivity_time_unit: str = "1day"
    length_factor: float = 1.0

    # v5.0 rating table factors
    rating_elev_factor: float = 1.0
    rating_flow_factor: float = 1.0
    rating_flow_time_unit: str = "1DAY"

    # Lake bed parameter defaults
    bed_conductivity: float = 2.0
    bed_thickness: float = 1.0

    @property
    def lake_dir(self) -> Path:
        """Get the lake subdirectory path."""
        return self.output_dir / self.lake_subdir

    @property
    def main_path(self) -> Path:
        """Get the main file path."""
        return self.lake_dir / self.main_file


class LakeComponentWriter(TemplateWriter):
    """
    Writer for IWFM Lake Component files.

    Writes all lake-related input files for IWFM simulation.

    Example
    -------
    >>> from pyiwfm.io.lake_writer import LakeComponentWriter, LakeWriterConfig
    >>> config = LakeWriterConfig(output_dir=Path("model/Simulation"))
    >>> writer = LakeComponentWriter(model, config)
    >>> files = writer.write_all()
    """

    def __init__(
        self,
        model: IWFMModel,
        config: LakeWriterConfig,
        template_engine: TemplateEngine | None = None,
    ) -> None:
        """
        Initialize the lake component writer.

        Parameters
        ----------
        model : IWFMModel
            Model to write
        config : LakeWriterConfig
            Output file configuration
        template_engine : TemplateEngine, optional
            Custom template engine
        """
        super().__init__(config.output_dir, template_engine)
        self.model = model
        self.config = config

    @property
    def format(self) -> str:
        return "iwfm_lake"

    def write(self, data: Any = None) -> None:
        """Write all lake files."""
        self.write_all()

    def write_all(self, write_defaults: bool = True) -> dict[str, Path]:
        """
        Write all lake component files.

        Parameters
        ----------
        write_defaults : bool
            If True, write default files even when no lake component
            is loaded (useful for generating simulation skeleton)

        Returns
        -------
        dict[str, Path]
            Mapping of file type to output path
        """
        logger.info(f"Writing lake files to {self.config.lake_dir}")

        # Ensure output directory exists
        self.config.lake_dir.mkdir(parents=True, exist_ok=True)

        results: dict[str, Path] = {}

        # Get lake component
        lakes = self.model.lakes

        if lakes is None and not write_defaults:
            logger.warning("No lake component in model and write_defaults=False")
            return results

        # Write main file
        results["main"] = self.write_main()

        logger.info(f"Wrote {len(results)} lake files")
        return results

    def write_main(self) -> Path:
        """
        Write the main lake control file.

        Returns
        -------
        Path
            Path to written file
        """
        output_path = self.config.main_path
        self._ensure_dir(output_path)

        lakes = self.model.lakes

        # Get lake data
        if lakes is not None and hasattr(lakes, "lakes") and lakes.lakes:
            lake_list = sorted(lakes.lakes.values(), key=lambda lake: lake.id)
            n_lakes = len(lake_list)
        else:
            lake_list = []
            n_lakes = 0

        # _render_lake_main accepts AppLake; when lakes is None we still
        # need to write an empty lake file, so pass a dummy AppLake.
        if lakes is None:
            from pyiwfm.components.lake import AppLake as _AppLake

            lakes = _AppLake()
        content = self._render_lake_main(
            lakes=lakes,
            lake_list=lake_list,
            n_lakes=n_lakes,
        )

        output_path.write_text(content)
        logger.info(f"Wrote lake main file: {output_path}")
        return output_path

    def _render_lake_main(
        self,
        lakes: AppLake,
        lake_list: list,
        n_lakes: int,
    ) -> str:
        """Render the main lake file using Jinja2 template."""
        sep = "\\"
        max_elev_file = (
            f"{self.config.lake_subdir}{sep}{self.config.max_elev_file}" if n_lakes > 0 else ""
        )

        generation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        version = _parse_lake_version(self.config.version)

        # Build lake data for template
        lake_data = []
        for i, lake in enumerate(lake_list):
            conductivity = getattr(lake, "bed_conductivity", self.config.bed_conductivity)
            thickness = getattr(lake, "bed_thickness", self.config.bed_thickness)
            ichlmax = getattr(lake, "max_elev_column", i + 1)
            icetlk = getattr(lake, "et_column", 7)
            icpcplk = getattr(lake, "precip_column", 2)
            name = getattr(lake, "name", f"Lake{lake.id}")
            initial_elev = getattr(lake, "initial_elevation", 280.0)

            entry = {
                "id": lake.id,
                "conductivity": conductivity,
                "thickness": thickness,
                "ichlmax": ichlmax,
                "icetlk": icetlk,
                "icpcplk": icpcplk,
                "name": name,
                "initial_elevation": initial_elev,
            }

            # v5.0 rating table
            if version >= (5, 0):
                rating_elevs = getattr(lake, "outflow_rating_elevations", [])
                rating_flows = getattr(lake, "outflow_rating_flows", [])
                n_pts = min(len(rating_elevs), len(rating_flows))
                entry["n_rating_points"] = n_pts
                entry["rating_points"] = [
                    {"elevation": rating_elevs[j], "flow": rating_flows[j]} for j in range(n_pts)
                ]
            else:
                entry["n_rating_points"] = 0
                entry["rating_points"] = []

            lake_data.append(entry)

        # Choose template based on version
        if version >= (5, 0):
            template_name = "lakes/lake_main_v50.j2"
        else:
            template_name = "lakes/lake_main_v40.j2"

        context = {
            "version": self.config.version,
            "generation_time": generation_time,
            "max_elev_file": max_elev_file,
            "budget_file": self.config.lake_budget_file,
            "final_elev_file": self.config.final_elev_file,
            "factk": self.config.conductivity_factor,
            "tunitk": self.config.conductivity_time_unit,
            "factl": self.config.length_factor,
            "factlkl": self.config.rating_elev_factor,
            "factlkq": self.config.rating_flow_factor,
            "tunitlkq": self.config.rating_flow_time_unit,
            "init_elev_factor": 1.0,
            "lake_data": lake_data,
        }

        return self._engine.render_template(template_name, **context)

    def write_max_lake_elev_ts(
        self,
        dates: list[str] | None = None,
        data: NDArray | None = None,
    ) -> Path:
        """
        Write the maximum lake elevation time series file (v4.0).

        Parameters
        ----------
        dates : list[str], optional
            IWFM timestamps
        data : NDArray, optional
            Max elevation data array (n_times, n_lakes)

        Returns
        -------
        Path
            Path to written file
        """
        from pyiwfm.io.timeseries_writer import (
            IWFMTimeSeriesDataWriter,
            make_max_lake_elev_ts_config,
        )

        lakes = self.model.lakes
        n_cols = 0
        if lakes is not None and hasattr(lakes, "lakes"):
            n_cols = len(lakes.lakes)

        ts_config = make_max_lake_elev_ts_config(
            ncol=n_cols,
            dates=dates,
            data=data,
        )

        output_path = self.config.lake_dir / self.config.max_elev_file
        writer = IWFMTimeSeriesDataWriter(self._engine)
        return writer.write(ts_config, output_path)


def write_lake_component(
    model: IWFMModel,
    output_dir: Path | str,
    config: LakeWriterConfig | None = None,
) -> dict[str, Path]:
    """
    Write lake component files for a model.

    Parameters
    ----------
    model : IWFMModel
        Model to write
    output_dir : Path or str
        Output directory
    config : LakeWriterConfig, optional
        File configuration

    Returns
    -------
    dict[str, Path]
        Mapping of file type to output path
    """
    output_dir = Path(output_dir)

    if config is None:
        config = LakeWriterConfig(output_dir=output_dir)
    else:
        config.output_dir = output_dir

    writer = LakeComponentWriter(model, config)
    return writer.write_all()
