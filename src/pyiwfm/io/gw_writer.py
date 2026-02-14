"""
Groundwater Component Writer for IWFM models.

This module provides the main writer for IWFM groundwater component files,
orchestrating the writing of all groundwater-related input files including:
- Main groundwater control file (GW_MAIN.dat)
- Boundary conditions (BC_MAIN.dat, SpecHeadBC.dat, SpecFlowBC.dat)
- Pumping files (Pump_MAIN.dat, ElemPump.dat, WellSpec.dat, TSPumping.dat)
- Tile drains (TileDrain.dat)
- Subsidence parameters (Subsidence.dat)
- Aquifer parameters (included in GW_MAIN.dat or separate file)
- Hydrograph specifications
- Face flow output
- Kh anomaly section
- GW return flows
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
from numpy.typing import NDArray

from pyiwfm.io.writer_base import TemplateWriter
from pyiwfm.templates.engine import TemplateEngine

if TYPE_CHECKING:
    from pyiwfm.core.model import IWFMModel
    from pyiwfm.components.groundwater import AppGW

logger = logging.getLogger(__name__)


@dataclass
class GWWriterConfig:
    """
    Configuration for groundwater component file writing.

    Attributes
    ----------
    output_dir : Path
        Base output directory for groundwater files
    gw_subdir : str
        Subdirectory name for groundwater files (default: "GW")
    version : str
        IWFM groundwater component version
    """
    output_dir: Path
    gw_subdir: str = "GW"
    version: str = "4.0"

    # File names
    main_file: str = "GW_MAIN.dat"
    bc_main_file: str = "BC_MAIN.dat"
    pump_main_file: str = "Pump_MAIN.dat"
    tile_drain_file: str = "TileDrain.dat"
    subsidence_file: str = "Subsidence.dat"
    elem_pump_file: str = "ElemPump.dat"
    well_spec_file: str = "WellSpec.dat"
    ts_pumping_file: str = "TSPumping.dat"
    spec_head_bc_file: str = "SpecHeadBC.dat"
    spec_flow_bc_file: str = "SpecFlowBC.dat"
    bound_tsd_file: str = "BoundTSD.dat"

    # Output files (optional)
    gw_budget_file: str = "../Results/GW.hdf"
    gw_zbudget_file: str = "../Results/GW_ZBud.hdf"
    gw_head_file: str = "../Results/GWHeadAll.out"
    gw_hyd_file: str = "../Results/GWHyd.out"
    gw_velocity_file: str = "../Results/GWVelocities.out"
    vertical_flow_file: str = "../Results/VerticalFlow.out"
    final_heads_file: str = "../Results/FinalGWHeads.out"
    tecplot_head_file: str = "../Results/TecPlotGW.out"
    pump_output_file: str = "../Results/Pumping.out"
    face_flow_file: str = "../Results/FaceFlows.out"
    td_output_file: str = "../Results/TileDrainHyd.out"
    bc_output_file: str = "../Results/BCOutput.out"

    # Unit conversions
    length_factor: float = 1.0
    length_unit: str = "ft."
    volume_factor: float = 2.29568e-5  # cu.ft. -> ac.ft.
    volume_unit: str = "ac.ft."
    velocity_factor: float = 1.0
    velocity_unit: str = "fpd"

    @property
    def gw_dir(self) -> Path:
        """Get the groundwater subdirectory path."""
        return self.output_dir / self.gw_subdir

    @property
    def main_path(self) -> Path:
        """Get the main file path."""
        return self.gw_dir / self.main_file

    @property
    def bc_main_path(self) -> Path:
        """Get the boundary conditions main file path."""
        return self.gw_dir / self.bc_main_file

    @property
    def pump_main_path(self) -> Path:
        """Get the pumping main file path."""
        return self.gw_dir / self.pump_main_file


class GWComponentWriter(TemplateWriter):
    """
    Writer for IWFM Groundwater Component files.

    Writes all groundwater-related input files for IWFM simulation.

    Example
    -------
    >>> from pyiwfm.io.gw_writer import GWComponentWriter, GWWriterConfig
    >>> config = GWWriterConfig(output_dir=Path("model/Simulation"))
    >>> writer = GWComponentWriter(model, config)
    >>> files = writer.write_all()
    """

    def __init__(
        self,
        model: "IWFMModel",
        config: GWWriterConfig,
        template_engine: TemplateEngine | None = None,
    ) -> None:
        """
        Initialize the groundwater component writer.

        Parameters
        ----------
        model : IWFMModel
            Model to write
        config : GWWriterConfig
            Output file configuration
        template_engine : TemplateEngine, optional
            Custom template engine
        """
        super().__init__(config.output_dir, template_engine)
        self.model = model
        self.config = config

    @property
    def format(self) -> str:
        return "iwfm_groundwater"

    def write(self, data: Any = None) -> None:
        """Write all groundwater files."""
        self.write_all()

    def write_all(self, write_defaults: bool = True) -> dict[str, Path]:
        """
        Write all groundwater component files.

        Parameters
        ----------
        write_defaults : bool
            If True, write default files even when no groundwater component
            is loaded (useful for generating simulation skeleton)

        Returns
        -------
        dict[str, Path]
            Mapping of file type to output path
        """
        logger.info(f"Writing groundwater files to {self.config.gw_dir}")

        # Ensure output directory exists
        self.config.gw_dir.mkdir(parents=True, exist_ok=True)

        results: dict[str, Path] = {}

        # Get groundwater component
        gw = self.model.groundwater

        # Write component files if data exists
        if gw is not None:
            if gw.boundary_conditions:
                results["bc_main"] = self.write_bc_main()

            if gw.wells or gw.element_pumping:
                results["pump_main"] = self.write_pump_main()

            if gw.tile_drains:
                results["tile_drains"] = self.write_tile_drains()

            if gw.subsidence:
                results["subsidence"] = self.write_subsidence()

        # Always write main file (contains aquifer params and initial heads)
        if write_defaults or gw is not None:
            results["main"] = self.write_main()

        logger.info(f"Wrote {len(results)} groundwater files")
        return results

    def write_main(self) -> Path:
        """
        Write the main groundwater control file.

        Returns
        -------
        Path
            Path to written file
        """
        output_path = self.config.main_path
        self._ensure_dir(output_path)

        gw = self.model.groundwater
        n_layers = self.model.stratigraphy.n_layers
        n_nodes = self.model.n_nodes

        # Determine which files to reference
        has_bc = gw is not None and gw.boundary_conditions
        has_pumping = gw is not None and (gw.wells or gw.element_pumping)
        has_tile_drains = gw is not None and gw.tile_drains
        has_subsidence = gw is not None and gw.subsidence

        content = self._render_gw_main(
            has_bc=has_bc,
            has_pumping=has_pumping,
            has_tile_drains=has_tile_drains,
            has_subsidence=has_subsidence,
            n_layers=n_layers,
            n_nodes=n_nodes,
            gw=gw,
        )

        output_path.write_text(content)
        logger.info(f"Wrote GW main file: {output_path}")
        return output_path

    def _render_gw_main(
        self,
        has_bc: bool,
        has_pumping: bool,
        has_tile_drains: bool,
        has_subsidence: bool,
        n_layers: int,
        n_nodes: int,
        gw: "AppGW",
    ) -> str:
        """Render the main groundwater file using Jinja2 template + numpy."""
        subdir = self.config.gw_subdir
        prefix = (subdir + "\\") if subdir else ""
        bc_file = f"{prefix}{self.config.bc_main_file}" if has_bc else ""
        td_file = f"{prefix}{self.config.tile_drain_file}" if has_tile_drains else ""
        pump_file = f"{prefix}{self.config.pump_main_file}" if has_pumping else ""
        subs_file = f"{prefix}{self.config.subsidence_file}" if has_subsidence else ""

        generation_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Build hydrograph location list
        hydrograph_locations = []
        if gw is not None and hasattr(gw, 'hydrograph_locations'):
            hydrograph_locations = list(gw.hydrograph_locations)

        # Build face flow spec list
        face_flow_specs = []
        if gw is not None and hasattr(gw, 'face_flow_specs'):
            face_flow_specs = getattr(gw, 'face_flow_specs', [])
            if not isinstance(face_flow_specs, list):
                face_flow_specs = []

        context = {
            "version": self.config.version,
            "generation_time": generation_time,
            "bc_file": bc_file,
            "td_file": td_file,
            "pump_file": pump_file,
            "subs_file": subs_file,
            "overwrite_file": "",
            "length_factor": str(self.config.length_factor),
            "length_unit": self.config.length_unit,
            "volume_factor": str(self.config.volume_factor),
            "volume_unit": self.config.volume_unit,
            "velocity_factor": str(self.config.velocity_factor),
            "velocity_unit": self.config.velocity_unit,
            "velocity_file": self.config.gw_velocity_file,
            "vertical_flow_file": self.config.vertical_flow_file,
            "head_all_file": self.config.gw_head_file,
            "tecplot_file": self.config.tecplot_head_file,
            "vtk_file": "",
            "budget_file": self.config.gw_budget_file,
            "zbudget_file": self.config.gw_zbudget_file,
            "final_heads_file": self.config.final_heads_file,
            "head_output_flag": 1,
            "debug_flag": 1,
            "n_hydrographs": len(hydrograph_locations),
            "hydrograph_xy_factor": 1.0,
            "hydrograph_file": self.config.gw_hyd_file,
            "hydrograph_locations": hydrograph_locations,
            "face_flow_specs": face_flow_specs,
            "n_face_flows": len(face_flow_specs),
            "face_flow_file": self.config.face_flow_file,
            "n_param_groups": 0,
            "fact_kh": 1.0,
            "fact_kv": 1.0,
            "fact_ss": 1.0,
            "fact_sy": 1.0,
            "fact_kaq": 1.0,
            "fact_kha": 0.0,
            "time_unit_kh": "DAY",
            "time_unit_kaq": "DAY",
            "time_unit_kv": "DAY",
        }

        header = self._engine.render_template("groundwater/gw_main.j2", **context)

        # Append aquifer parameters (large arrays, use numpy formatting)
        lines = [header]
        if gw and gw.aquifer_params:
            params = gw.aquifer_params
            for node_idx in range(n_nodes):
                node_id = node_idx + 1
                line = f"    {node_id:<5}"
                for layer in range(n_layers):
                    kh = params.kh[node_idx, layer] if params.kh is not None else 1.0
                    kv = params.kv[node_idx, layer] if params.kv is not None else 0.1
                    ss = params.specific_storage[node_idx, layer] if params.specific_storage is not None else 1e-6
                    sy = params.specific_yield[node_idx, layer] if params.specific_yield is not None else 0.1
                    kaq = 0.1
                    line += f"  {kh:10.4f}  {kv:10.4f}  {ss:12.6e}  {sy:8.4f}  {kaq:10.4f}"
                lines.append(line)
        else:
            for node_idx in range(n_nodes):
                node_id = node_idx + 1
                line = f"    {node_id:<5}"
                for layer in range(n_layers):
                    line += f"  {1.0:10.4f}  {0.1:10.4f}  {1e-6:12.6e}  {0.1:8.4f}  {0.1:10.4f}"
                lines.append(line)

        # Kh anomaly section
        n_kh_anomalies = 0
        kh_anomalies = []
        if gw is not None and hasattr(gw, 'kh_anomalies'):
            kh_anomalies = getattr(gw, 'kh_anomalies', [])
            if not isinstance(kh_anomalies, list):
                kh_anomalies = []
            n_kh_anomalies = len(kh_anomalies)

        if n_kh_anomalies > 0:
            lines.append("C*******************************************************************************")
            lines.append("C                       Kh Anomaly Data")
            lines.append("C-------------------------------------------------------------------------------")
            lines.append(f"    {n_kh_anomalies}                           / NEBK")
            lines.append("    1.0                         / FACT")
            lines.append("    DAY                         / TUNITH")
            for row in kh_anomalies:
                lines.append(f"    {row}")

        # GW return flows
        return_flow_flag = 0
        if gw is not None and hasattr(gw, 'return_flow_destinations'):
            return_dests = getattr(gw, 'return_flow_destinations', {})
            if isinstance(return_dests, dict) and return_dests:
                return_flow_flag = 1

        lines.append("C*******************************************************************************")
        lines.append("C                       GW Return Flows")
        lines.append("C-------------------------------------------------------------------------------")
        lines.append(f"    {return_flow_flag}                           / IFLAGRF")

        if return_flow_flag and isinstance(return_dests, dict):
            for node_id in sorted(return_dests.keys()):
                dest_type, dest_id = return_dests[node_id]
                lines.append(f"    {node_id:<6} {dest_type:>4} {dest_id:>6}")

        # Initial heads section
        lines.append("C*******************************************************************************")
        lines.append("C                       Initial Groundwater Heads")
        lines.append("C")
        lines.append("C   FACTHINI ; Conversion factor for initial heads")
        lines.append("C-------------------------------------------------------------------------------")
        lines.append("    1.0                         / FACTHINI")
        lines.append("C-------------------------------------------------------------------------------")
        lines.append("C   ID      ; Node ID")
        lines.append("C   For each layer:")
        lines.append("C     HEAD  ; Initial head [L]")
        lines.append("C-------------------------------------------------------------------------------")

        if gw and gw.heads is not None:
            for node_idx in range(n_nodes):
                node_id = node_idx + 1
                line = f"    {node_id:<5}"
                for layer in range(n_layers):
                    head = gw.heads[node_idx, layer]
                    line += f"  {head:12.4f}"
                lines.append(line)
        else:
            strat = self.model.stratigraphy
            for node_idx in range(n_nodes):
                node_id = node_idx + 1
                line = f"    {node_id:<5}"
                for layer in range(n_layers):
                    head = (strat.top_elev[node_idx, layer] + strat.bottom_elev[node_idx, layer]) / 2
                    line += f"  {head:12.4f}"
                lines.append(line)

        return "\n".join(lines) + "\n"

    def write_bc_main(self) -> Path:
        """
        Write the boundary conditions main file.

        Returns
        -------
        Path
            Path to written file
        """
        output_path = self.config.bc_main_path
        self._ensure_dir(output_path)

        gw = self.model.groundwater
        bcs = gw.boundary_conditions if gw else []

        # Group BCs by type
        spec_head = [bc for bc in bcs if bc.bc_type == "specified_head"]
        spec_flow = [bc for bc in bcs if bc.bc_type == "specified_flow"]
        gen_head = [bc for bc in bcs if bc.bc_type == "general_head"]
        constrained_gh = [bc for bc in bcs if bc.bc_type == "constrained_general_head"]

        generation_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        subdir = self.config.gw_subdir
        prefix = (subdir + "\\") if subdir else ""

        context = {
            "generation_time": generation_time,
            "n_spec_head": len(spec_head),
            "n_spec_flow": len(spec_flow),
            "n_gen_head": len(gen_head),
            "n_constrained_gh": len(constrained_gh),
            "n_tsbc_files": 0,
            "spec_head_bc_file": f"{prefix}{self.config.spec_head_bc_file}" if spec_head else "",
            "spec_head_tsd_file": f"{prefix}{self.config.bound_tsd_file}" if spec_head else "",
            "spec_flow_bc_file": f"{prefix}{self.config.spec_flow_bc_file}" if spec_flow else "",
            "spec_flow_tsd_file": f"{prefix}{self.config.bound_tsd_file}" if spec_flow else "",
            "gen_head_bc_file": "",
            "gen_head_tsd_file": "",
            "constrained_gh_bc_file": "",
            "constrained_gh_tsd_file": "",
            "tsbc_files": [],
            "boundary_node_output": False,
            "n_bc_output_nodes": 0,
            "bc_output_file": self.config.bc_output_file,
            "bc_output_nodes": [],
        }

        content = self._engine.render_template("groundwater/bc_main.j2", **context)

        output_path.write_text(content)
        logger.info(f"Wrote BC main file: {output_path}")
        return output_path

    def write_pump_main(self) -> Path:
        """
        Write the pumping main file.

        Returns
        -------
        Path
            Path to written file
        """
        output_path = self.config.pump_main_path
        self._ensure_dir(output_path)

        gw = self.model.groundwater
        has_wells = gw is not None and gw.wells
        has_elem_pump = gw is not None and gw.element_pumping

        generation_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        subdir = self.config.gw_subdir
        prefix = (subdir + "\\") if subdir else ""

        pump_flag = 1 if has_elem_pump else (2 if has_wells else 0)

        context = {
            "generation_time": generation_time,
            "pump_flag": pump_flag,
            "elem_pump_file": f"{prefix}{self.config.elem_pump_file}" if has_elem_pump else "",
            "well_spec_file": f"{prefix}{self.config.well_spec_file}" if has_wells else "",
            "ts_pumping_file": f"{prefix}{self.config.ts_pumping_file}",
            "pump_output_file": self.config.pump_output_file,
        }

        content = self._engine.render_template("groundwater/pump_main.j2", **context)

        output_path.write_text(content)
        logger.info(f"Wrote pump main file: {output_path}")
        return output_path

    def write_tile_drains(self) -> Path:
        """
        Write the tile drains file.

        Returns
        -------
        Path
            Path to written file
        """
        output_path = self.config.gw_dir / self.config.tile_drain_file
        self._ensure_dir(output_path)

        gw = self.model.groundwater
        drains = gw.tile_drains if gw else {}

        generation_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Get conversion factors from model or use defaults
        td_elev_factor = getattr(gw, 'td_elev_factor', 1.0) if gw else 1.0
        td_cond_factor = getattr(gw, 'td_cond_factor', 1.0) if gw else 1.0
        td_time_unit = getattr(gw, 'td_time_unit', '1DAY') if gw else '1DAY'

        # Build drain data for template.
        # The reader multiplies raw values by the factor, so we divide here
        # to recover the raw values that IWFM expects in the file.
        drain_data = []
        for drain_id in sorted(drains.keys()):
            drain = drains[drain_id]
            # Map destination type to integer (0=outside, 1=stream node)
            dest_type_val = getattr(drain, 'dest_type', 0)
            if isinstance(dest_type_val, str):
                dest_type_val = 0 if dest_type_val.lower() in ('outside', 'none', '') else 1
            elev = drain.elevation
            cond = drain.conductance
            if td_elev_factor != 0.0 and td_elev_factor != 1.0:
                elev = elev / td_elev_factor
            if td_cond_factor != 0.0 and td_cond_factor != 1.0:
                cond = cond / td_cond_factor
            drain_data.append({
                "id": drain.id,
                "gw_node": getattr(drain, 'gw_node', getattr(drain, 'element', 0)),
                "elevation": elev,
                "conductance": cond,
                "dest_type": int(dest_type_val),
                "dest_id": getattr(drain, 'destination_id', getattr(drain, 'dest_id', 0)) or 0,
            })

        # Build sub-irrigation data
        si_elev_factor = getattr(gw, 'si_elev_factor', 1.0) if gw else 1.0
        si_cond_factor = getattr(gw, 'si_cond_factor', 1.0) if gw else 1.0
        si_time_unit = getattr(gw, 'si_time_unit', '1MON') if gw else '1MON'
        sub_irrigations = getattr(gw, 'sub_irrigations', []) if gw else []
        si_data = []
        for si in sub_irrigations:
            elev = si.elevation
            cond = si.conductance
            if si_elev_factor != 0.0 and si_elev_factor != 1.0:
                elev = elev / si_elev_factor
            if si_cond_factor != 0.0 and si_cond_factor != 1.0:
                cond = cond / si_cond_factor
            si_data.append({
                "id": si.id,
                "gw_node": si.gw_node,
                "elevation": elev,
                "conductance": cond,
            })

        context = {
            "generation_time": generation_time,
            "n_drains": len(drains),
            "td_elev_factor": td_elev_factor,
            "td_cond_factor": td_cond_factor,
            "td_time_unit": td_time_unit,
            "td_output_file": self.config.td_output_file,
            "drains": drain_data,
            "n_subirrig": len(si_data),
            "si_elev_factor": si_elev_factor,
            "si_cond_factor": si_cond_factor,
            "si_time_unit": si_time_unit,
            "sub_irrigation": si_data,
            "n_td_hydro": 0,
            "td_hydro_locations": [],
        }

        content = self._engine.render_template("groundwater/tile_drain.j2", **context)

        output_path.write_text(content)
        logger.info(f"Wrote tile drains file: {output_path}")
        return output_path

    def write_subsidence(self) -> Path:
        """
        Write the subsidence parameters file.

        Returns
        -------
        Path
            Path to written file
        """
        output_path = self.config.gw_dir / self.config.subsidence_file
        self._ensure_dir(output_path)

        gw = self.model.groundwater
        subsidence = gw.subsidence if gw else []
        node_subsidence = gw.node_subsidence if gw else []
        subs_config = getattr(gw, 'subsidence_config', None) if gw else None

        generation_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        context = {
            "generation_time": generation_time,
            "n_subsidence": len(subsidence),
            "subsidence_data": subsidence,
            "n_node_subsidence": len(node_subsidence),
            "node_subsidence": node_subsidence,
            "subs_version": getattr(subs_config, 'version', '') if subs_config else '',
            "n_hydrograph_outputs": getattr(subs_config, 'n_hydrograph_outputs', 0) if subs_config else 0,
        }

        content = self._engine.render_template("groundwater/subsidence.j2", **context)

        output_path.write_text(content)
        logger.info(f"Wrote subsidence file: {output_path}")
        return output_path

    def write_spec_head_bc(self) -> Path:
        """
        Write the specified head boundary condition data file.

        Returns
        -------
        Path
            Path to written file
        """
        output_path = self.config.gw_dir / self.config.spec_head_bc_file
        self._ensure_dir(output_path)

        gw = self.model.groundwater
        bcs = gw.boundary_conditions if gw else []
        spec_head = [bc for bc in bcs if bc.bc_type == "specified_head"]

        generation_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        bc_nodes = []
        col = 1
        for bc in spec_head:
            for i, node in enumerate(bc.nodes):
                bc_nodes.append({"node": node, "layer": bc.layer, "column": col})
                col += 1

        context = {
            "generation_time": generation_time,
            "n_nodes": len(bc_nodes),
            "factor": 1.0,
            "bc_nodes": bc_nodes,
        }

        content = self._engine.render_template(
            "groundwater/spec_head_bc.j2", **context
        )

        output_path.write_text(content)
        logger.info(f"Wrote spec head BC file: {output_path}")
        return output_path

    def write_spec_flow_bc(self) -> Path:
        """
        Write the specified flow boundary condition data file.

        Returns
        -------
        Path
            Path to written file
        """
        output_path = self.config.gw_dir / self.config.spec_flow_bc_file
        self._ensure_dir(output_path)

        gw = self.model.groundwater
        bcs = gw.boundary_conditions if gw else []
        spec_flow = [bc for bc in bcs if bc.bc_type == "specified_flow"]

        generation_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        bc_nodes = []
        col = 1
        for bc in spec_flow:
            for i, node in enumerate(bc.nodes):
                bc_nodes.append({"node": node, "layer": bc.layer, "column": col})
                col += 1

        context = {
            "generation_time": generation_time,
            "n_nodes": len(bc_nodes),
            "factor": 1.0,
            "bc_nodes": bc_nodes,
        }

        content = self._engine.render_template(
            "groundwater/spec_flow_bc.j2", **context
        )

        output_path.write_text(content)
        logger.info(f"Wrote spec flow BC file: {output_path}")
        return output_path

    def write_well_specs(self) -> Path:
        """
        Write the well specification file.

        Returns
        -------
        Path
            Path to written file
        """
        output_path = self.config.gw_dir / self.config.well_spec_file
        self._ensure_dir(output_path)

        gw = self.model.groundwater
        wells = gw.wells if gw else {}

        generation_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        lines = [
            "C*******************************************************************************",
            "C                  WELL SPECIFICATION FILE",
            "C",
            f"C             Generated by pyiwfm",
            f"C             {generation_time}",
            "C*******************************************************************************",
            "C",
            f"    {len(wells):<10}                         / NWELL",
            "    1.0                         / FACTWL (well conversion factor)",
            "C-------------------------------------------------------------------------------",
            "C   ID   ELEM   X       Y       RBOT    RTOP    ICOL   NAME",
            "C-------------------------------------------------------------------------------",
        ]

        col = 1
        for well_id in sorted(wells.keys()):
            well = wells[well_id]
            name = well.name or f"Well_{well_id}"
            lines.append(
                f"    {well.id:<5} {well.element:>5}"
                f"  {well.x:>12.2f}  {well.y:>12.2f}"
                f"  {well.bottom_screen:>10.2f}  {well.top_screen:>10.2f}"
                f"  {col:>5}  {name}"
            )
            col += 1

        output_path.write_text("\n".join(lines) + "\n")
        logger.info(f"Wrote well spec file: {output_path}")
        return output_path

    def write_elem_pump_specs(self) -> Path:
        """
        Write the element-based pumping specification file.

        Returns
        -------
        Path
            Path to written file
        """
        output_path = self.config.gw_dir / self.config.elem_pump_file
        self._ensure_dir(output_path)

        gw = self.model.groundwater
        elem_pumping = gw.element_pumping if gw else []

        generation_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        lines = [
            "C*******************************************************************************",
            "C                  ELEMENT PUMPING SPECIFICATION FILE",
            "C",
            f"C             Generated by pyiwfm",
            f"C             {generation_time}",
            "C*******************************************************************************",
            "C",
            f"    {len(elem_pumping):<10}                         / NEPUMP",
            "    1.0                         / FACTEP (pump conversion factor)",
            "C-------------------------------------------------------------------------------",
            "C   ELEM  LAYER  LAYER_FRAC  ICOL",
            "C-------------------------------------------------------------------------------",
        ]

        col = 1
        for ep in elem_pumping:
            lines.append(
                f"    {ep.element_id:<5} {ep.layer:>5}"
                f"  {ep.layer_fraction:>10.4f}  {col:>5}"
            )
            col += 1

        output_path.write_text("\n".join(lines) + "\n")
        logger.info(f"Wrote elem pump spec file: {output_path}")
        return output_path

    def write_ts_pumping(
        self,
        dates: list[str] | None = None,
        data: NDArray | None = None,
    ) -> Path:
        """
        Write the pumping time series data file using IWFMTimeSeriesDataWriter.

        Parameters
        ----------
        dates : list[str], optional
            IWFM timestamps
        data : NDArray, optional
            Pumping data array (n_times, n_cols)

        Returns
        -------
        Path
            Path to written file
        """
        from pyiwfm.io.timeseries_writer import (
            IWFMTimeSeriesDataWriter,
            make_pumping_ts_config,
        )

        gw = self.model.groundwater
        n_cols = 0
        if gw is not None:
            n_cols = len(gw.wells) + len(gw.element_pumping)

        ts_config = make_pumping_ts_config(
            ncol=n_cols,
            dates=dates,
            data=data,
        )

        output_path = self.config.gw_dir / self.config.ts_pumping_file
        writer = IWFMTimeSeriesDataWriter(self._engine)
        return writer.write(ts_config, output_path)

    def write_hydrograph_specs(self) -> Path:
        """
        Write groundwater hydrograph output locations.

        This is typically embedded in the main file, but can also
        be called standalone for verification.

        Returns
        -------
        Path
            Path to main file (hydrograph specs are part of GW_MAIN)
        """
        return self.write_main()

    def write_face_flow_specs(self) -> Path:
        """
        Write face flow output specifications.

        This is typically embedded in the main file.

        Returns
        -------
        Path
            Path to main file (face flow specs are part of GW_MAIN)
        """
        return self.write_main()


def write_gw_component(
    model: "IWFMModel",
    output_dir: Path | str,
    config: GWWriterConfig | None = None,
) -> dict[str, Path]:
    """
    Write groundwater component files for a model.

    Parameters
    ----------
    model : IWFMModel
        Model to write
    output_dir : Path or str
        Output directory
    config : GWWriterConfig, optional
        File configuration

    Returns
    -------
    dict[str, Path]
        Mapping of file type to output path
    """
    output_dir = Path(output_dir)

    if config is None:
        config = GWWriterConfig(output_dir=output_dir)
    else:
        config.output_dir = output_dir

    writer = GWComponentWriter(model, config)
    return writer.write_all()
