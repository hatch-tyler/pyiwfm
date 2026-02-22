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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from numpy.typing import NDArray

from pyiwfm.io.writer_base import TemplateWriter
from pyiwfm.io.writer_config_base import BaseComponentWriterConfig
from pyiwfm.templates.engine import TemplateEngine

if TYPE_CHECKING:
    from pyiwfm.components.groundwater import AppGW
    from pyiwfm.core.model import IWFMModel

logger = logging.getLogger(__name__)


@dataclass
class GWWriterConfig(BaseComponentWriterConfig):
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

    gw_subdir: str = "GW"

    def _get_subdir(self) -> str:
        return self.gw_subdir

    def _get_main_file(self) -> str:
        return self.main_file

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
        return self.component_dir

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
        model: IWFMModel,
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
                spec_head = [bc for bc in gw.boundary_conditions if bc.bc_type == "specified_head"]
                spec_flow = [bc for bc in gw.boundary_conditions if bc.bc_type == "specified_flow"]
                if spec_head:
                    results["spec_head_bc"] = self.write_spec_head_bc()
                if spec_flow:
                    results["spec_flow_bc"] = self.write_spec_flow_bc()
                ts_path = self.write_bc_ts_data()
                if ts_path:
                    results["bc_ts_data"] = ts_path

            if gw.wells or gw.element_pumping:
                results["pump_main"] = self.write_pump_main()
                if gw.wells:
                    results["well_specs"] = self.write_well_specs()
                if gw.element_pumping:
                    results["elem_pump_specs"] = self.write_elem_pump_specs()

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
        strat = self.model.stratigraphy
        assert strat is not None, "Stratigraphy must be loaded to write GW main file"
        n_layers = strat.n_layers
        n_nodes = self.model.n_nodes

        # Use roundtrip path when gw_main_config is available (from reader)
        from pyiwfm.io.groundwater import GWMainFileConfig

        gw_main_cfg = getattr(gw, "gw_main_config", None) if gw else None
        if isinstance(gw_main_cfg, GWMainFileConfig):
            content = self._render_gw_main_roundtrip(gw_main_cfg, gw, n_layers, n_nodes)
        else:
            # Determine which files to reference
            has_bc = bool(gw is not None and gw.boundary_conditions)
            has_pumping = bool(gw is not None and (gw.wells or gw.element_pumping))
            has_tile_drains = bool(gw is not None and gw.tile_drains)
            has_subsidence = bool(gw is not None and gw.subsidence)

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

    def _render_gw_main_roundtrip(
        self,
        cfg: Any,
        gw: AppGW | None,
        n_layers: int,
        n_nodes: int,
    ) -> str:
        """Render GW_MAIN.dat from stored GWMainFileConfig for roundtrip fidelity.

        Generates all data lines to match the original file format exactly.
        Comment lines are decorative and don't affect data comparison.
        """
        subdir = self.config.gw_subdir
        prefix = (subdir + "\\") if subdir else ""
        raw = cfg.raw_paths

        # Helper to pad a value line
        def _val(v: str, desc: str = "") -> str:
            v_str = str(v) if v else ""
            if desc:
                return f"          {v_str:<40} / {desc}"
            return f"          {v_str}"

        lines: list[str] = []

        # Version header
        lines.append(f"#{cfg.version}")

        # Sub-file paths
        has_bc = bool(gw is not None and gw.boundary_conditions)
        has_pumping = bool(gw is not None and (gw.wells or gw.element_pumping))
        has_td = bool(gw is not None and gw.tile_drains)
        has_subs = cfg.subsidence_file is not None

        bc_ref = f"{prefix}{self.config.bc_main_file}" if has_bc else ""
        td_ref = f"{prefix}{self.config.tile_drain_file}" if has_td else ""
        pump_ref = f"{prefix}{self.config.pump_main_file}" if has_pumping else ""
        subs_ref = f"{prefix}{self.config.subsidence_file}" if has_subs else ""

        lines.append(_val(bc_ref, "BCFL"))
        lines.append(_val(td_ref, "TDFL"))
        lines.append(_val(pump_ref, "PUMPFL"))
        lines.append(_val(subs_ref, "SUBSFL"))
        lines.append(_val("", "OVRWRTFL"))

        # Conversion factors and units
        lines.append(_val(str(cfg.head_output_factor), "FACTLTOU"))
        lines.append(_val(cfg.head_output_unit, "UNITLTOU"))
        lines.append(_val(str(cfg.volume_output_factor), "FACTVLOU"))
        lines.append(_val(cfg.volume_output_unit, "UNITVLOU"))
        lines.append(_val(str(cfg.velocity_output_factor), "FACTVROU"))
        lines.append(_val(cfg.velocity_output_unit, "UNITVROU"))

        # Output file paths (use raw paths from reader for roundtrip fidelity)
        lines.append(_val(raw.get("velocity", ""), "VELOUTFL"))
        lines.append(_val(raw.get("vflow", ""), "VFLOWOUTFL"))
        lines.append(_val(raw.get("headall", ""), "GWALLOUTFL"))
        lines.append(_val(raw.get("tecplot", ""), "HTPOUTFL"))
        lines.append(_val(raw.get("vtk", ""), "VTPOUTFL"))
        lines.append(_val(raw.get("budget", ""), "GWBUDFL"))
        lines.append(_val(raw.get("zbudget", ""), "ZBUDFL"))
        lines.append(_val(raw.get("final_heads", ""), "FNGWFL"))

        # IHTPFLAG
        lines.append(_val(str(cfg.aq_head_output_flag), "IHTPFLAG"))

        # Debug flag
        lines.append(f"    {cfg.debug_flag}                           / KDEB")

        # ── Hydrograph output ──────────────────────────────
        hyd_locs = cfg.hydrograph_locations
        lines.append(f"    {len(hyd_locs)}                          / NOUTH")
        lines.append(f"    {cfg.coord_factor}                      / FACTXY")
        lines.append(_val(raw.get("hydout", ""), "GWHYDOUTFL"))

        # Write hydrograph location data with correct HYDTYP format
        for i, loc in enumerate(hyd_locs):
            hid = i + 1
            # Infer HYDTYP: 0=x-y coords (node_id==0), 1=node number
            if loc.node_id == 0:
                # HYDTYP=0: ID  0  IOUTHL  X  Y  NAME
                # Divide x,y by coord_factor to recover original raw values
                x_raw = loc.x / cfg.coord_factor if cfg.coord_factor else loc.x
                y_raw = loc.y / cfg.coord_factor if cfg.coord_factor else loc.y
                lines.append(
                    f"     {hid:<5} 0  {loc.layer:>8}"
                    f"     {x_raw:>12.1f}  {y_raw:>12.1f}"
                    f"              {loc.name}"
                )
            else:
                # HYDTYP=1: ID  1  IOUTHL  [blanks]  IOUTH  NAME
                lines.append(
                    f"     {hid:<5} 1  {loc.layer:>8}"
                    f"                                  {loc.node_id:<8} {loc.name}"
                )

        # ── Face flow output ──────────────────────────────
        ff_specs = cfg.face_flow_specs
        lines.append(f"    {len(ff_specs)}                           / NOUTF")
        lines.append(_val(raw.get("faceflow", ""), "FCHYDOUTFL"))

        for ff in ff_specs:
            lines.append(
                f"    {ff.id:<5}  {ff.layer:>4}  {ff.node_a:>9}  {ff.node_b:>9}      {ff.name}"
            )

        # ── Aquifer parameters ────────────────────────────
        ngroup = cfg.n_param_groups
        lines.append(f"         {ngroup}                      / NGROUP")

        # Conversion factors (single line with 6 values: FX FKH FS FN FV FL)
        lines.append(f"  {cfg.aq_factors_line}")

        # Time units
        lines.append(f"    {cfg.aq_time_unit_kh:<24}/ TUNITKH")
        lines.append(f"    {cfg.aq_time_unit_v:<24}/ TUNITV")
        lines.append(f"    {cfg.aq_time_unit_l:<24}/ TUNITL")

        if ngroup > 0 and cfg.parametric_grids:
            # Parametric grid format: replay raw data
            for grid in cfg.parametric_grids:
                lines.append(f"   {grid.node_range_str}")
                # IWFM's ReadCharacterUntilComment reads data lines until
                # it hits a comment line (C/c/*). Without this separator,
                # the NDP/NEP lines would be parsed as node range data.
                lines.append("C")
                lines.append(f"     {grid.n_nodes}                          / NDP")
                lines.append(f"     {grid.n_elements}                          / NEP")

                # Element definitions (if NEP > 0)
                for elem in grid.elements:
                    parts = "  ".join(str(v + 1) for v in elem)
                    lines.append(f"     {parts}")

                # Parametric node data (raw lines for roundtrip fidelity)
                for raw_line in grid.raw_node_lines:
                    lines.append(f"    {raw_line}")
        else:
            # Per-node format: NGROUP=0
            if gw and gw.aquifer_params:
                params = gw.aquifer_params
                for node_idx in range(n_nodes):
                    node_id = node_idx + 1
                    line_str = f"    {node_id:<5}"
                    for layer in range(n_layers):
                        kh = params.kh[node_idx, layer] if params.kh is not None else 1.0
                        ss = (
                            params.specific_storage[node_idx, layer]
                            if params.specific_storage is not None
                            else 1e-6
                        )
                        sy = (
                            params.specific_yield[node_idx, layer]
                            if params.specific_yield is not None
                            else 0.1
                        )
                        akv = (
                            params.aquitard_kv[node_idx, layer]
                            if params.aquitard_kv is not None
                            else 0.1
                        )
                        kv = params.kv[node_idx, layer] if params.kv is not None else 0.1
                        line_str += f"  {kh:10.4f}  {ss:12.6e}  {sy:8.4f}  {akv:10.4f}  {kv:10.4f}"
                    lines.append(line_str)

        # ── Kh anomaly ────────────────────────────────────
        n_kh = len(cfg.kh_anomalies)
        lines.append(f"     {n_kh}                          / NEBK")
        lines.append(f"     {cfg.kh_anomaly_factor}                        / FACT")
        lines.append(f"     {cfg.kh_anomaly_time_unit:<24}/ TUNITH")

        for entry in cfg.kh_anomalies:
            bk_str = "  ".join(f"{v:.4f}" for v in entry.kh_per_layer)
            lines.append(f"     {entry.element_id}  {bk_str}")

        # ── Return flows ──────────────────────────────────
        lines.append(f"    {cfg.return_flow_flag}                           / IFLAGRF")

        # ── Initial heads ─────────────────────────────────
        lines.append("    1.0                         / FACTHP")

        if gw and gw.heads is not None:
            for node_idx in range(n_nodes):
                node_id = node_idx + 1
                line_str = f"    {node_id:<5}"
                for layer in range(n_layers):
                    head = gw.heads[node_idx, layer]
                    line_str += f"  {head:12.4f}"
                lines.append(line_str)
        elif cfg.initial_heads is not None:
            heads = cfg.initial_heads
            for node_idx in range(heads.shape[0]):
                node_id = node_idx + 1
                line_str = f"    {node_id:<5}"
                for layer in range(heads.shape[1]):
                    line_str += f"  {heads[node_idx, layer]:12.4f}"
                lines.append(line_str)

        return "\n".join(lines) + "\n"

    def _render_gw_main(
        self,
        has_bc: bool,
        has_pumping: bool,
        has_tile_drains: bool,
        has_subsidence: bool,
        n_layers: int,
        n_nodes: int,
        gw: AppGW | None,
    ) -> str:
        """Render the main groundwater file using Jinja2 template + numpy."""
        subdir = self.config.gw_subdir
        prefix = (subdir + "\\") if subdir else ""
        bc_file = f"{prefix}{self.config.bc_main_file}" if has_bc else ""
        td_file = f"{prefix}{self.config.tile_drain_file}" if has_tile_drains else ""
        pump_file = f"{prefix}{self.config.pump_main_file}" if has_pumping else ""
        subs_file = f"{prefix}{self.config.subsidence_file}" if has_subsidence else ""

        generation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Build hydrograph location list
        hydrograph_locations = []
        if gw is not None and hasattr(gw, "hydrograph_locations"):
            hydrograph_locations = list(gw.hydrograph_locations)

        # Build face flow spec list
        face_flow_specs: list[Any] = []
        if gw is not None and hasattr(gw, "face_flow_specs"):
            face_flow_specs = getattr(gw, "face_flow_specs", [])
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
                    ss = (
                        params.specific_storage[node_idx, layer]
                        if params.specific_storage is not None
                        else 1e-6
                    )
                    sy = (
                        params.specific_yield[node_idx, layer]
                        if params.specific_yield is not None
                        else 0.1
                    )
                    kaq = 0.1
                    line += f"  {kh:10.4f}  {kv:10.4f}  {ss:12.6e}  {sy:8.4f}  {kaq:10.4f}"
                lines.append(line)
        else:
            for node_idx in range(n_nodes):
                node_id = node_idx + 1
                line = f"    {node_id:<5}"
                for _layer in range(n_layers):
                    line += f"  {1.0:10.4f}  {0.1:10.4f}  {1e-6:12.6e}  {0.1:8.4f}  {0.1:10.4f}"
                lines.append(line)

        # Kh anomaly section
        lines.append(
            "C*******************************************************************************"
        )
        lines.append("C                       Kh Anomaly Data")
        lines.append(
            "C-------------------------------------------------------------------------------"
        )
        lines.append("    0                           / NEBK")
        lines.append("    1.0                         / FACT")
        lines.append("    DAY                         / TUNITH")

        # GW return flows
        lines.append(
            "C*******************************************************************************"
        )
        lines.append("C                       GW Return Flows")
        lines.append(
            "C-------------------------------------------------------------------------------"
        )
        lines.append("    0                           / IFLAGRF")

        # Initial heads section
        lines.append(
            "C*******************************************************************************"
        )
        lines.append("C                       Initial Groundwater Heads")
        lines.append(
            "C-------------------------------------------------------------------------------"
        )
        lines.append("    1.0                         / FACTHINI")
        lines.append(
            "C-------------------------------------------------------------------------------"
        )

        if gw and gw.heads is not None:
            for node_idx in range(n_nodes):
                node_id = node_idx + 1
                line = f"    {node_id:<5}"
                for layer in range(n_layers):
                    head = gw.heads[node_idx, layer]
                    line += f"  {head:12.4f}"
                lines.append(line)
        else:
            init_strat = self.model.stratigraphy
            assert init_strat is not None, "Stratigraphy must be loaded to write initial heads"
            for node_idx in range(n_nodes):
                node_id = node_idx + 1
                line = f"    {node_id:<5}"
                for layer in range(n_layers):
                    head = (
                        init_strat.top_elev[node_idx, layer]
                        + init_strat.bottom_elev[node_idx, layer]
                    ) / 2
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

        generation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        subdir = self.config.gw_subdir
        prefix = (subdir + "\\") if subdir else ""

        # Get NOUTB data from model's GW component
        n_bc_output_nodes = getattr(gw, "n_bc_output_nodes", 0) if gw else 0
        bc_output_specs = getattr(gw, "bc_output_specs", []) if gw else []
        # Use raw (unresolved) path from original file for roundtrip fidelity
        bc_output_file_val = getattr(gw, "bc_output_file_raw", "") if gw else ""
        if not bc_output_file_val and n_bc_output_nodes > 0:
            bc_output_file_val = self.config.bc_output_file

        context = {
            "generation_time": generation_time,
            # The IWFM BC main file reads 5 file paths first, then NOUTB.
            # File order: SpecFlow, SpecHead, GenHead, ConstrainedGH, TS data.
            "spec_flow_bc_file": f"{prefix}{self.config.spec_flow_bc_file}" if spec_flow else "",
            "spec_head_bc_file": f"{prefix}{self.config.spec_head_bc_file}" if spec_head else "",
            "gen_head_bc_file": "",
            "constrained_gh_bc_file": "",
            "ts_data_file": f"{prefix}{self.config.bound_tsd_file}"
            if (spec_head or spec_flow)
            else "",
            "n_bc_output_nodes": n_bc_output_nodes,
            "bc_output_file": bc_output_file_val,
            "bc_output_nodes": bc_output_specs,
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

        generation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        subdir = self.config.gw_subdir
        prefix = (subdir + "\\") if subdir else ""

        context = {
            "generation_time": generation_time,
            "well_spec_file": f"{prefix}{self.config.well_spec_file}" if has_wells else "",
            "elem_pump_file": f"{prefix}{self.config.elem_pump_file}" if has_elem_pump else "",
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

        generation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Get conversion factors from model or use defaults
        td_elev_factor = getattr(gw, "td_elev_factor", 1.0) if gw else 1.0
        td_cond_factor = getattr(gw, "td_cond_factor", 1.0) if gw else 1.0
        td_time_unit = getattr(gw, "td_time_unit", "1DAY") if gw else "1DAY"

        # Build drain data for template.
        # The reader multiplies raw values by the factor, so we divide here
        # to recover the raw values that IWFM expects in the file.
        drain_data = []
        for drain_id in sorted(drains.keys()):
            drain = drains[drain_id]
            # Map destination type to integer (IWFM: 0=outside, 1=stream node)
            dest_type_raw = getattr(drain, "destination_type", "outside")
            if isinstance(dest_type_raw, str):
                dest_type_val = 1 if dest_type_raw.lower() == "stream" else 0
            elif isinstance(dest_type_raw, (int, float)):
                dest_type_val = int(dest_type_raw)
            else:
                dest_type_val = 0
            elev = drain.elevation
            cond = drain.conductance
            if td_elev_factor != 0.0 and td_elev_factor != 1.0:
                elev = elev / td_elev_factor
            if td_cond_factor != 0.0 and td_cond_factor != 1.0:
                cond = cond / td_cond_factor
            drain_data.append(
                {
                    "id": drain.id,
                    "gw_node": getattr(drain, "gw_node", getattr(drain, "element", 0)),
                    "elevation": elev,
                    "conductance": cond,
                    "dest_type": int(dest_type_val),
                    "dest_id": getattr(drain, "destination_id", getattr(drain, "dest_id", 0)) or 0,
                }
            )

        # Build sub-irrigation data
        si_elev_factor = getattr(gw, "si_elev_factor", 1.0) if gw else 1.0
        si_cond_factor = getattr(gw, "si_cond_factor", 1.0) if gw else 1.0
        si_time_unit = getattr(gw, "si_time_unit", "1MON") if gw else "1MON"
        sub_irrigations = getattr(gw, "sub_irrigations", []) if gw else []
        si_data = []
        for si in sub_irrigations:
            elev = si.elevation
            cond = si.conductance
            if si_elev_factor != 0.0 and si_elev_factor != 1.0:
                elev = elev / si_elev_factor
            if si_cond_factor != 0.0 and si_cond_factor != 1.0:
                cond = cond / si_cond_factor
            si_data.append(
                {
                    "id": si.id,
                    "gw_node": si.gw_node,
                    "elevation": elev,
                    "conductance": cond,
                }
            )

        # Get hydrograph output data from model's GW component
        n_td_hydro = getattr(gw, "td_n_hydro", 0) if gw else 0
        td_hydro_volume_factor = getattr(gw, "td_hydro_volume_factor", 1.0) if gw else 1.0
        td_hydro_volume_unit = getattr(gw, "td_hydro_volume_unit", "") if gw else ""
        td_hydro_specs = getattr(gw, "td_hydro_specs", []) if gw else []

        # Resolve the td_output_file path
        td_output_file_val = getattr(gw, "td_output_file_raw", "") if gw else ""
        if not td_output_file_val:
            td_output_file_val = self.config.td_output_file

        context = {
            "generation_time": generation_time,
            "n_drains": len(drains),
            "td_elev_factor": td_elev_factor,
            "td_cond_factor": td_cond_factor,
            "td_time_unit": td_time_unit,
            "drains": drain_data,
            "n_subirrig": len(si_data),
            "si_elev_factor": si_elev_factor,
            "si_cond_factor": si_cond_factor,
            "si_time_unit": si_time_unit,
            "sub_irrigation": si_data,
            "n_td_hydro": n_td_hydro,
            "td_hydro_volume_factor": td_hydro_volume_factor,
            "td_hydro_volume_unit": td_hydro_volume_unit,
            "td_output_file": td_output_file_val,
            "td_hydro_locations": td_hydro_specs,
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
        subs_config = getattr(gw, "subsidence_config", None) if gw else None

        from pyiwfm.io.gw_subsidence import SubsidenceConfig

        if isinstance(subs_config, SubsidenceConfig):
            from pyiwfm.io.gw_subsidence_writer import write_subsidence_main

            result = write_subsidence_main(subs_config, output_path)
            logger.info(f"Wrote subsidence file: {output_path}")
            return result

        # Fallback: template-based output for programmatic models
        subsidence = gw.subsidence if gw else []
        node_subsidence = gw.node_subsidence if gw else []

        generation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        context = {
            "generation_time": generation_time,
            "n_subsidence": len(subsidence),
            "subsidence_data": subsidence,
            "n_node_subsidence": len(node_subsidence),
            "node_subsidence": node_subsidence,
            "subs_version": "",
            "n_hydrograph_outputs": 0,
        }

        content = self._engine.render_template("groundwater/subsidence.j2", **context)

        output_path.write_text(content)
        logger.info(f"Wrote subsidence file: {output_path}")
        return output_path

    def write_spec_head_bc(self) -> Path:
        """Write the specified head boundary condition data file."""
        output_path = self.config.gw_dir / self.config.spec_head_bc_file
        self._ensure_dir(output_path)

        gw = self.model.groundwater
        bcs = gw.boundary_conditions if gw else []
        spec_head = [bc for bc in bcs if bc.bc_type == "specified_head"]

        # Use bc_config factor if available
        bc_config = getattr(gw, "bc_config", None)
        factor = bc_config.sp_head_factor if bc_config else 1.0

        generation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [
            "C*******************************************************************************",
            "C                  SPECIFIED HEAD BOUNDARY CONDITION DATA",
            "C",
            "C             Generated by pyiwfm",
            f"C             {generation_time}",
            "C*******************************************************************************",
            f"    {len(spec_head):<10}                         / NHB",
            f"    {factor:<14}               / FACT",
            "C   INODE    ILAYER   ITSCOL        BH",
        ]
        for bc in spec_head:
            node = bc.nodes[0]
            head_val = bc.values[0] if bc.values else 0.0
            lines.append(
                f"    {node:>5}    {bc.layer:>5}    {bc.ts_column:>5}    {head_val:>10.1f}"
            )

        output_path.write_text("\n".join(lines) + "\n")
        logger.info(f"Wrote spec head BC file: {output_path}")
        return output_path

    def write_spec_flow_bc(self) -> Path:
        """Write the specified flow boundary condition data file."""
        output_path = self.config.gw_dir / self.config.spec_flow_bc_file
        self._ensure_dir(output_path)

        gw = self.model.groundwater
        bcs = gw.boundary_conditions if gw else []
        spec_flow = [bc for bc in bcs if bc.bc_type == "specified_flow"]

        bc_config = getattr(gw, "bc_config", None)
        factor = bc_config.sp_flow_factor if bc_config else 1.0
        time_unit = bc_config.sp_flow_time_unit if bc_config else ""

        generation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [
            "C*******************************************************************************",
            "C                  SPECIFIED FLOW BOUNDARY CONDITION DATA",
            "C",
            "C             Generated by pyiwfm",
            f"C             {generation_time}",
            "C*******************************************************************************",
            f"    {len(spec_flow):<10}                         / NQB",
            f"    {factor:<14}               / FACT",
        ]
        if time_unit:
            lines.append(f"    {time_unit:<24}/ TUNIT")
        lines.append("C   INODE    ILAYER   ITSCOL   BASEFLOW")
        for bc in spec_flow:
            node = bc.nodes[0]
            flow_val = bc.values[0] if bc.values else 0.0
            lines.append(
                f"    {node:>5}    {bc.layer:>5}    {bc.ts_column:>5}    {flow_val:>10.1f}"
            )

        output_path.write_text("\n".join(lines) + "\n")
        logger.info(f"Wrote spec flow BC file: {output_path}")
        return output_path

    def write_bc_ts_data(self) -> Path | None:
        """Copy the BC time series data file from source if it exists."""
        import shutil

        gw = self.model.groundwater
        if not gw:
            return None
        src = getattr(gw, "bc_ts_file", None)
        if not src or not Path(src).exists():
            return None
        output_path = self.config.gw_dir / self.config.bound_tsd_file
        self._ensure_dir(output_path)
        shutil.copy2(str(src), str(output_path))
        logger.info(f"Copied BC time series file: {output_path}")
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

        generation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines = [
            "C*******************************************************************************",
            "C                  WELL SPECIFICATION FILE",
            "C",
            "C             Generated by pyiwfm",
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

        generation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines = [
            "C*******************************************************************************",
            "C                  ELEMENT PUMPING SPECIFICATION FILE",
            "C",
            "C             Generated by pyiwfm",
            f"C             {generation_time}",
            "C*******************************************************************************",
            "C",
            f"     {len(elem_pumping)}                          / NSINK",
        ]

        for ep in elem_pumping:
            # Format: ELEM ICOL FRAC IDIST LF1..LFn ITYPDST IDEST ICFIRIG ICFADJ ICFMAX FRACMAX
            layer_facs = "".join(f"          {lf}" for lf in ep.layer_factors)
            lines.append(
                f"  {ep.element_id:<5}{ep.pump_column:>5}"
                f"       {ep.pump_fraction}"
                f"       {ep.dist_method}"
                f"{layer_facs}"
                f"          {ep.dest_type}"
                f"         {ep.dest_id}"
                f"        {ep.irig_frac_column}"
                f"          {ep.adjust_column}"
                f"         {ep.pump_max_column}"
                f"       {ep.pump_max_fraction}"
            )

        # Element groups (NGRP)
        lines.append("     0                 / NGRP")

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
    model: IWFMModel,
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
