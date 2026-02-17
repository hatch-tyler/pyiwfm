"""
Root Zone Component Writer for IWFM models.

This module provides the main writer for IWFM root zone component files,
orchestrating the writing of all root zone-related input files including:
- Main root zone control file (RootZone_MAIN.dat)
- Soil parameters for each element
- Return flow fractions
- Irrigation periods
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
    from pyiwfm.components.rootzone import RootZone as AppRootZone
    from pyiwfm.core.model import IWFMModel

logger = logging.getLogger(__name__)


@dataclass
class RootZoneWriterConfig:
    """
    Configuration for root zone component file writing.

    Attributes
    ----------
    output_dir : Path
        Base output directory for root zone files
    rootzone_subdir : str
        Subdirectory name for root zone files (default: "RootZone")
    version : str
        IWFM root zone component version
    """

    output_dir: Path
    rootzone_subdir: str = "RootZone"
    version: str = "4.12"

    # File names
    main_file: str = "RootZone_MAIN.dat"
    return_flow_file: str = "ReturnFlowFrac.dat"
    reuse_file: str = "ReuseFrac.dat"
    irig_period_file: str = "IrigPeriod.dat"
    surface_flow_dest_file: str = "SurfaceFlowDest.dat"

    # Output files
    lwu_budget_file: str = "../Results/LWU.hdf"
    rz_budget_file: str = "../Results/RootZone.hdf"
    lwu_zbudget_file: str = "../Results/LWU_ZBud.hdf"
    rz_zbudget_file: str = "../Results/RootZone_ZBud.hdf"

    # Simulation parameters
    convergence: float = 0.001
    max_iterations: int = 150
    inch_to_length_factor: float = 0.0833333  # inches to feet
    gw_uptake: int = 0  # 0 = disabled

    # Soil parameter defaults
    wilting_point: float = 0.0
    field_capacity: float = 0.20
    total_porosity: float = 0.45
    pore_size_index: float = 0.62
    hydraulic_conductivity: float = 2.60
    k_ponded: float = -1.0  # -1.0 means same as K
    rhc_method: int = 2  # 1=Campbell, 2=van Genuchten-Mualem
    capillary_rise: float = 0.0

    # Unit conversions
    k_factor: float = 0.03281  # cm to ft
    cprise_factor: float = 1.0  # FACTEXDTH (v4.1+)
    k_time_unit: str = "1hour"

    # End-of-simulation moisture output
    final_moisture_file: str = ""

    @property
    def rootzone_dir(self) -> Path:
        """Get the root zone subdirectory path."""
        return self.output_dir / self.rootzone_subdir

    @property
    def main_path(self) -> Path:
        """Get the main file path."""
        return self.rootzone_dir / self.main_file


def _sp_val(obj: object, attr: str, default: float | int, alt: str | None = None) -> float | int:
    """Get a numeric attribute from *obj*, trying *attr* first then *alt*.

    Returns *default* if neither attribute yields a real numeric value
    (handles ``MagicMock`` objects that produce non-numeric surrogates).
    """
    for name in (attr, alt):
        if name is None:
            continue
        val = getattr(obj, name, None)
        if isinstance(val, (int, float)):
            return val
    return default


class RootZoneComponentWriter(TemplateWriter):
    """
    Writer for IWFM Root Zone Component files.

    Writes all root zone-related input files for IWFM simulation.

    Example
    -------
    >>> from pyiwfm.io.rootzone_writer import RootZoneComponentWriter, RootZoneWriterConfig
    >>> config = RootZoneWriterConfig(output_dir=Path("model/Simulation"))
    >>> writer = RootZoneComponentWriter(model, config)
    >>> files = writer.write_all()
    """

    def __init__(
        self,
        model: IWFMModel,
        config: RootZoneWriterConfig,
        template_engine: TemplateEngine | None = None,
    ) -> None:
        """
        Initialize the root zone component writer.

        Parameters
        ----------
        model : IWFMModel
            Model to write
        config : RootZoneWriterConfig
            Output file configuration
        template_engine : TemplateEngine, optional
            Custom template engine
        """
        super().__init__(config.output_dir, template_engine)
        self.model = model
        self.config = config

    @property
    def format(self) -> str:
        return "iwfm_rootzone"

    def write(self, data: Any = None) -> None:
        """Write all root zone files."""
        self.write_all()

    def write_all(self, write_defaults: bool = True) -> dict[str, Path]:
        """
        Write all root zone component files.

        Parameters
        ----------
        write_defaults : bool
            If True, write default files even when no root zone component
            is loaded (useful for generating simulation skeleton)

        Returns
        -------
        dict[str, Path]
            Mapping of file type to output path
        """
        logger.info(f"Writing root zone files to {self.config.rootzone_dir}")

        # Ensure output directory exists
        self.config.rootzone_dir.mkdir(parents=True, exist_ok=True)

        results: dict[str, Path] = {}

        # Get root zone component
        rootzone = self.model.rootzone

        if rootzone is None and not write_defaults:
            logger.warning("No root zone component in model and write_defaults=False")
            return results

        # Write main file
        results["main"] = self.write_main()

        # Write v4x sub-files when data is available
        if rootzone is not None:
            try:
                from pyiwfm.io.rootzone_v4x import (
                    NativeRiparianWriterV4x,
                    NonPondedCropWriterV4x,
                    PondedCropWriterV4x,
                    UrbanWriterV4x,
                )

                if rootzone.nonponded_config is not None:
                    writer = NonPondedCropWriterV4x()
                    path = self.config.rootzone_dir / "NonPondedAg.dat"
                    writer.write(rootzone.nonponded_config, path)
                    results["nonponded"] = path

                if rootzone.ponded_config is not None:
                    ponded_writer = PondedCropWriterV4x()
                    path = self.config.rootzone_dir / "PondedAg.dat"
                    ponded_writer.write(rootzone.ponded_config, path)
                    results["ponded"] = path

                if rootzone.urban_config is not None:
                    urban_writer = UrbanWriterV4x()
                    path = self.config.rootzone_dir / "UrbanLandUse.dat"
                    urban_writer.write(rootzone.urban_config, path)
                    results["urban"] = path

                if rootzone.native_riparian_config is not None:
                    nr_writer = NativeRiparianWriterV4x()
                    path = self.config.rootzone_dir / "NativeRiparian.dat"
                    nr_writer.write(rootzone.native_riparian_config, path)
                    results["native_riparian"] = path
            except Exception:
                pass  # sub-file writing is best-effort

        logger.info(f"Wrote {len(results)} root zone files")
        return results

    def write_main(self) -> Path:
        """
        Write the main root zone control file.

        Returns
        -------
        Path
            Path to written file
        """
        output_path = self.config.main_path
        self._ensure_dir(output_path)

        rootzone = self.model.rootzone
        if rootzone is None:
            from pyiwfm.components.rootzone import RootZone

            rootzone = RootZone(n_elements=0, n_layers=1)

        # Get element IDs from mesh
        if self.model.grid is not None:
            element_ids = sorted(self.model.grid.elements.keys())
            n_elements = len(element_ids)
        else:
            element_ids = []
            n_elements = 0

        content = self._render_rootzone_main(
            rootzone=rootzone,
            element_ids=element_ids,
            n_elements=n_elements,
        )

        output_path.write_text(content)
        logger.info(f"Wrote root zone main file: {output_path}")
        return output_path

    def _render_rootzone_main(
        self,
        rootzone: AppRootZone,
        element_ids: list[int],
        n_elements: int,
    ) -> str:
        """Render the main root zone file using inline template.

        Generates version-aware output:
        - v4.0/v4.01: 12-13 col soil params, no GWUPTK, no zone budgets
        - v4.1/v4.11: adds capillary rise, GWUPTK, zone budgets
        - v4.12/v4.13: adds per-landuse surface flow destinations (16 cols)
        """
        from pyiwfm.io.rootzone import version_ge

        generation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ver = self.config.version
        is_v41 = version_ge(ver, (4, 1))
        is_v411 = version_ge(ver, (4, 11))
        is_v412 = version_ge(ver, (4, 12))

        # Build file paths - leave land use files blank for minimal setup
        rz_subdir = self.config.rootzone_subdir
        prefix = (rz_subdir + "\\") if rz_subdir else ""
        rf_file = f"{prefix}{self.config.return_flow_file}"
        ru_file = f"{prefix}{self.config.reuse_file}"
        ip_file = f"{prefix}{self.config.irig_period_file}"
        dest_file = f"{prefix}{self.config.surface_flow_dest_file}"

        # Column header depends on version
        if is_v412:
            col_header = "C  IE      WP      FC      TN      LAMBDA     K   KPonded   RHC  CPRISE  IRNE  FRNE  IMSRC  ICDSTAG  ICDSTURBIN  ICDSTURBOUT  ICDSTNVRV"
        elif is_v41:
            col_header = "C  IE      WP      FC      TN      LAMBDA     K   RHC  CPRISE  IRNE  FRNE  IMSRC  IDEST_TYPE  IDEST  [KPonded]"
        else:
            col_header = "C  IE      WP      FC      TN      LAMBDA     K   RHC  IRNE  FRNE  IMSRC  IDEST_TYPE  IDEST  [KPonded]"

        # Render header via Jinja2 template
        context = {
            "version": ver,
            "generation_time": generation_time,
            "convergence": self.config.convergence,
            "max_iterations": self.config.max_iterations,
            "inch_to_length": self.config.inch_to_length_factor,
            "is_v41": is_v41,
            "is_v411": is_v411,
            "is_v412": is_v412,
            "gw_uptake": self.config.gw_uptake,
            "nonponded_file": "",
            "ponded_file": "",
            "urban_file": "",
            "native_riparian_file": "",
            "return_flow_file": rf_file,
            "reuse_file": ru_file,
            "irig_period_file": ip_file,
            "moisture_src_file": "",
            "ag_water_demand_file": "",
            "lwu_budget_file": self.config.lwu_budget_file,
            "rz_budget_file": self.config.rz_budget_file,
            "lwu_zbudget_file": self.config.lwu_zbudget_file,
            "rz_zbudget_file": self.config.rz_zbudget_file,
            "area_scale_file": "",
            "final_moisture_file": self.config.final_moisture_file,
            "k_factor": self.config.k_factor,
            "cprise_factor": self.config.cprise_factor,
            "k_time_unit": self.config.k_time_unit,
            "dest_file": dest_file,
            "column_header": col_header,
        }

        header = self._engine.render_template("rootzone/rootzone_main.j2", **context)

        lines: list[str] = [header.rstrip()]

        # Add soil parameters for each element
        for elem_id in element_ids:
            sp = None
            if rootzone is not None and hasattr(rootzone, "soil_params") and rootzone.soil_params:
                sp = rootzone.soil_params.get(elem_id, None)

            # Use _sp_val helper to support both real SoilParameters
            # and legacy mocks (which may use old attribute names)
            if sp:
                wp = _sp_val(sp, "wilting_point", self.config.wilting_point)
                fc = _sp_val(sp, "field_capacity", self.config.field_capacity)
                tn = _sp_val(sp, "porosity", self.config.total_porosity, alt="total_porosity")
                lam = _sp_val(
                    sp, "lambda_param", self.config.pore_size_index, alt="pore_size_index"
                )
                k = _sp_val(
                    sp,
                    "saturated_kv",
                    self.config.hydraulic_conductivity,
                    alt="hydraulic_conductivity",
                )
                kp = _sp_val(sp, "k_ponded", self.config.k_ponded)
                rhc = _sp_val(sp, "kunsat_method", self.config.rhc_method, alt="rhc_method")
                cprise = _sp_val(sp, "capillary_rise", self.config.capillary_rise)
                irne = _sp_val(sp, "precip_column", 1)
                frne = _sp_val(sp, "precip_factor", 1.0)
                imsrc = _sp_val(sp, "generic_moisture_column", 0)
            else:
                wp = self.config.wilting_point
                fc = self.config.field_capacity
                tn = self.config.total_porosity
                lam = self.config.pore_size_index
                k = self.config.hydraulic_conductivity
                kp = self.config.k_ponded
                rhc = self.config.rhc_method
                cprise = self.config.capillary_rise
                irne = 1
                frne = 1.0
                imsrc = 0

            if is_v412:
                # Get per-landuse destinations (use getattr for mock compat)
                _dest_ag = getattr(rootzone, "surface_flow_dest_ag", {}) if rootzone else {}
                _dest_ui = getattr(rootzone, "surface_flow_dest_urban_in", {}) if rootzone else {}
                _dest_uo = getattr(rootzone, "surface_flow_dest_urban_out", {}) if rootzone else {}
                _dest_nv = getattr(rootzone, "surface_flow_dest_nvrv", {}) if rootzone else {}
                # Guard against non-dict (e.g. MagicMock)
                dests_ag = _dest_ag.get(elem_id, (1, 0)) if isinstance(_dest_ag, dict) else (1, 0)
                dests_ui = _dest_ui.get(elem_id, (1, 0)) if isinstance(_dest_ui, dict) else (1, 0)
                dests_uo = _dest_uo.get(elem_id, (1, 0)) if isinstance(_dest_uo, dict) else (1, 0)
                dests_nv = _dest_nv.get(elem_id, (1, 0)) if isinstance(_dest_nv, dict) else (1, 0)
                lines.append(
                    f"   {elem_id:<6} {wp:>6.1f}     {fc:>4.2f}    {tn:>4.2f}"
                    f"     {lam:>5.2f}      {k:>4.2f}  {kp:>4.1f}"
                    f"    {rhc}     {cprise:>3.1f}     {irne}    {frne:.1f}"
                    f"     {imsrc}       {dests_ag[0]}        {dests_ui[0]}"
                    f"           {dests_uo[0]}            {dests_nv[0]}"
                )
            elif is_v41:
                _dests = getattr(rootzone, "surface_flow_destinations", {}) if rootzone else {}
                dests = _dests.get(elem_id, (0, 0)) if isinstance(_dests, dict) else (0, 0)
                lines.append(
                    f"   {elem_id:<6} {wp:>6.1f}     {fc:>4.2f}    {tn:>4.2f}"
                    f"     {lam:>5.2f}      {k:>4.2f}    {rhc}"
                    f"     {cprise:>3.1f}     {irne}    {frne:.1f}"
                    f"     {imsrc}       {dests[0]}        {dests[1]}"
                )
            else:
                _dests = getattr(rootzone, "surface_flow_destinations", {}) if rootzone else {}
                dests = _dests.get(elem_id, (0, 0)) if isinstance(_dests, dict) else (0, 0)
                lines.append(
                    f"   {elem_id:<6} {wp:>6.1f}     {fc:>4.2f}    {tn:>4.2f}"
                    f"     {lam:>5.2f}      {k:>4.2f}    {rhc}"
                    f"     {irne}    {frne:.1f}     {imsrc}"
                    f"       {dests[0]}        {dests[1]}"
                )

        return "\n".join(lines) + "\n"

    def write_precip_ts(
        self,
        dates: list[str] | None = None,
        data: NDArray | None = None,
    ) -> Path:
        """Write precipitation time series data file."""
        from pyiwfm.io.timeseries_writer import (
            IWFMTimeSeriesDataWriter,
            make_precip_ts_config,
        )

        n_cols = len(self.model.grid.elements) if self.model.grid else 0
        ts_config = make_precip_ts_config(ncol=n_cols, dates=dates, data=data)
        output_path = self.config.rootzone_dir / "Precip.dat"
        writer = IWFMTimeSeriesDataWriter(self._engine)
        return writer.write(ts_config, output_path)

    def write_et_ts(
        self,
        dates: list[str] | None = None,
        data: NDArray | None = None,
    ) -> Path:
        """Write evapotranspiration time series data file."""
        from pyiwfm.io.timeseries_writer import (
            IWFMTimeSeriesDataWriter,
            make_et_ts_config,
        )

        n_cols = len(self.model.grid.elements) if self.model.grid else 0
        ts_config = make_et_ts_config(ncol=n_cols, dates=dates, data=data)
        output_path = self.config.rootzone_dir / "ET.dat"
        writer = IWFMTimeSeriesDataWriter(self._engine)
        return writer.write(ts_config, output_path)

    def write_crop_coeff_ts(
        self,
        dates: list[str] | None = None,
        data: NDArray | None = None,
    ) -> Path:
        """Write crop coefficient time series data file."""
        from pyiwfm.io.timeseries_writer import (
            IWFMTimeSeriesDataWriter,
            make_crop_coeff_ts_config,
        )

        n_cols = 1  # Placeholder; caller should specify
        ts_config = make_crop_coeff_ts_config(ncol=n_cols, dates=dates, data=data)
        output_path = self.config.rootzone_dir / "CropCoeff.dat"
        writer = IWFMTimeSeriesDataWriter(self._engine)
        return writer.write(ts_config, output_path)

    def write_return_flow_ts(
        self,
        dates: list[str] | None = None,
        data: NDArray | None = None,
    ) -> Path:
        """Write return flow fraction time series data file."""
        from pyiwfm.io.timeseries_writer import (
            IWFMTimeSeriesDataWriter,
            make_return_flow_ts_config,
        )

        n_cols = len(self.model.grid.elements) if self.model.grid else 0
        ts_config = make_return_flow_ts_config(ncol=n_cols, dates=dates, data=data)
        output_path = self.config.rootzone_dir / self.config.return_flow_file
        writer = IWFMTimeSeriesDataWriter(self._engine)
        return writer.write(ts_config, output_path)

    def write_reuse_ts(
        self,
        dates: list[str] | None = None,
        data: NDArray | None = None,
    ) -> Path:
        """Write reuse fraction time series data file."""
        from pyiwfm.io.timeseries_writer import (
            IWFMTimeSeriesDataWriter,
            make_reuse_ts_config,
        )

        n_cols = len(self.model.grid.elements) if self.model.grid else 0
        ts_config = make_reuse_ts_config(ncol=n_cols, dates=dates, data=data)
        output_path = self.config.rootzone_dir / self.config.reuse_file
        writer = IWFMTimeSeriesDataWriter(self._engine)
        return writer.write(ts_config, output_path)

    def write_irig_period_ts(
        self,
        dates: list[str] | None = None,
        data: NDArray | None = None,
    ) -> Path:
        """Write irrigation period time series data file."""
        from pyiwfm.io.timeseries_writer import (
            IWFMTimeSeriesDataWriter,
            make_irig_period_ts_config,
        )

        n_cols = len(self.model.grid.elements) if self.model.grid else 0
        ts_config = make_irig_period_ts_config(ncol=n_cols, dates=dates, data=data)
        output_path = self.config.rootzone_dir / self.config.irig_period_file
        writer = IWFMTimeSeriesDataWriter(self._engine)
        return writer.write(ts_config, output_path)

    def write_ag_water_demand_ts(
        self,
        dates: list[str] | None = None,
        data: NDArray | None = None,
    ) -> Path:
        """Write agricultural water demand time series data file."""
        from pyiwfm.io.timeseries_writer import (
            IWFMTimeSeriesDataWriter,
            make_ag_water_demand_ts_config,
        )

        n_cols = len(self.model.grid.elements) if self.model.grid else 0
        ts_config = make_ag_water_demand_ts_config(ncol=n_cols, dates=dates, data=data)
        output_path = self.config.rootzone_dir / "AgWaterDemand.dat"
        writer = IWFMTimeSeriesDataWriter(self._engine)
        return writer.write(ts_config, output_path)


def write_rootzone_component(
    model: IWFMModel,
    output_dir: Path | str,
    config: RootZoneWriterConfig | None = None,
) -> dict[str, Path]:
    """
    Write root zone component files for a model.

    Parameters
    ----------
    model : IWFMModel
        Model to write
    output_dir : Path or str
        Output directory
    config : RootZoneWriterConfig, optional
        File configuration

    Returns
    -------
    dict[str, Path]
        Mapping of file type to output path
    """
    output_dir = Path(output_dir)

    if config is None:
        config = RootZoneWriterConfig(output_dir=output_dir)
    else:
        config.output_dir = output_dir

    writer = RootZoneComponentWriter(model, config)
    return writer.write_all()
