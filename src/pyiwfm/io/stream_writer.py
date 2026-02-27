"""
Stream Component Writer for IWFM models.

This module provides the main writer for IWFM stream component files,
orchestrating the writing of all stream-related input files including:
- Main stream control file (Stream_MAIN.dat)
- Stream inflows time series
- Diversion specifications
- Bypass specifications
- Stream bed parameters
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from numpy.typing import NDArray

from pyiwfm.io.streams import parse_stream_version
from pyiwfm.io.writer_base import TemplateWriter
from pyiwfm.io.writer_config_base import BaseComponentWriterConfig
from pyiwfm.templates.engine import TemplateEngine

if TYPE_CHECKING:
    from pyiwfm.components.stream import AppStream
    from pyiwfm.core.model import IWFMModel

logger = logging.getLogger(__name__)


@dataclass
class StreamWriterConfig(BaseComponentWriterConfig):
    """
    Configuration for stream component file writing.

    Attributes
    ----------
    output_dir : Path
        Base output directory for stream files
    stream_subdir : str
        Subdirectory name for stream files (default: "Stream")
    version : str
        IWFM stream component version
    """

    stream_subdir: str = "Stream"

    def _get_subdir(self) -> str:
        return self.stream_subdir

    def _get_main_file(self) -> str:
        return self.main_file

    # File names
    main_file: str = "Stream_MAIN.dat"
    inflow_file: str = "StreamInflow.dat"
    diver_specs_file: str = "DiverSpecs.dat"
    bypass_specs_file: str = "BypassSpecs.dat"
    diversions_file: str = "Diversions.dat"

    # Output files (optional)
    strm_budget_file: str = "../Results/StrmBud.hdf"
    strm_node_budget_file: str = "../Results/StrmNodeBud.hdf"
    strm_hyd_file: str = "../Results/StrmHyd.out"
    diver_detail_file: str = "../Results/DiverDetail.hdf"

    # Unit conversions
    flow_factor: float = 0.000022957  # cu.ft./day -> ac.ft./day
    flow_unit: str = "ac.ft./day"
    length_factor: float = 1.0
    length_unit: str = "ft"

    # Stream bed parameters defaults
    conductivity: float = 10.0
    bed_thickness: float = 1.0
    wetted_perimeter: float = 150.0
    conductivity_factor: float = 1.0
    conductivity_time_unit: str = "1day"
    bed_length_factor: float = 1.0
    interaction_type: int = 1

    # v5.0 fields
    final_flow_file: str = ""
    roughness_factor: float = 1.0
    cross_section_length_factor: float = 1.0
    ic_type: int = 0
    ic_time_unit: str = ""
    ic_factor: float = 1.0

    # Evaporation
    evap_area_file: str = ""

    @property
    def stream_dir(self) -> Path:
        """Get the stream subdirectory path."""
        return self.component_dir


class StreamComponentWriter(TemplateWriter):
    """
    Writer for IWFM Stream Component files.

    Writes all stream-related input files for IWFM simulation.

    Example
    -------
    >>> from pyiwfm.io.stream_writer import StreamComponentWriter, StreamWriterConfig
    >>> config = StreamWriterConfig(output_dir=Path("model/Simulation"))
    >>> writer = StreamComponentWriter(model, config)
    >>> files = writer.write_all()
    """

    def __init__(
        self,
        model: IWFMModel,
        config: StreamWriterConfig,
        template_engine: TemplateEngine | None = None,
    ) -> None:
        """
        Initialize the stream component writer.

        Parameters
        ----------
        model : IWFMModel
            Model to write
        config : StreamWriterConfig
            Output file configuration
        template_engine : TemplateEngine, optional
            Custom template engine
        """
        super().__init__(config.output_dir, template_engine)
        self.model = model
        self.config = config

    @property
    def format(self) -> str:
        return "iwfm_stream"

    def write(self, data: Any = None) -> None:
        """Write all stream files."""
        self.write_all()

    def write_all(self, write_defaults: bool = True) -> dict[str, Path]:
        """
        Write all stream component files.

        Parameters
        ----------
        write_defaults : bool
            If True, write default files even when no stream component
            is loaded (useful for generating simulation skeleton)

        Returns
        -------
        dict[str, Path]
            Mapping of file type to output path
        """
        logger.info(f"Writing stream files to {self.config.stream_dir}")

        # Ensure output directory exists
        self.config.stream_dir.mkdir(parents=True, exist_ok=True)

        results: dict[str, Path] = {}

        # Get stream component
        streams = self.model.streams

        if streams is None and not write_defaults:
            logger.warning("No stream component in model and write_defaults=False")
            return results

        # Write main file
        results["main"] = self.write_main()

        # Write component files if stream data exists
        if streams is not None:
            if streams.diversions:
                results["diver_specs"] = self.write_diver_specs()

            if streams.bypasses:
                results["bypass_specs"] = self.write_bypass_specs()

        logger.info(f"Wrote {len(results)} stream files")
        return results

    def write_main(self) -> Path:
        """
        Write the main stream control file.

        Returns
        -------
        Path
            Path to written file
        """
        output_path = self.config.main_path
        self._ensure_dir(output_path)

        streams = self.model.streams

        # Get stream nodes for bed parameters
        if streams is not None and streams.nodes:
            stream_nodes = sorted(streams.nodes.keys())
            n_stream_nodes = len(stream_nodes)
        elif streams is not None and streams.reaches:
            # Fallback: gather node IDs from reaches when nodes dict is empty
            node_id_set: set[int] = set()
            for reach in streams.reaches.values():
                node_id_set.update(reach.nodes)
            stream_nodes = sorted(node_id_set)
            n_stream_nodes = len(stream_nodes)
        elif streams is not None and getattr(streams, "budget_node_ids", None):
            # Second fallback: use budget node IDs
            stream_nodes = sorted(streams.budget_node_ids)
            n_stream_nodes = len(stream_nodes)
        else:
            stream_nodes = []
            n_stream_nodes = 0

        # Determine which files to reference
        has_inflows = (
            streams is not None and hasattr(streams, "inflows") and streams.inflows
        ) or bool(self.model.source_files.get("stream_inflow_ts"))
        has_diversions = streams is not None and streams.diversions
        has_bypasses = streams is not None and streams.bypasses

        if streams is None:
            from pyiwfm.components.stream import AppStream

            streams = AppStream()
        content = self._render_stream_main(
            has_inflows=has_inflows,
            has_diversions=bool(has_diversions),
            has_bypasses=bool(has_bypasses),
            stream_nodes=stream_nodes,
            n_stream_nodes=n_stream_nodes,
            streams=streams,
        )

        output_path.write_text(content)
        logger.info(f"Wrote stream main file: {output_path}")
        return output_path

    def _render_stream_main(
        self,
        has_inflows: bool,
        has_diversions: bool,
        has_bypasses: bool,
        stream_nodes: list[int],
        n_stream_nodes: int,
        streams: AppStream | None,
    ) -> str:
        """Render the main stream file using inline template."""
        version = parse_stream_version(self.config.version)

        content = self._render_header_and_paths(has_inflows, has_diversions, has_bypasses)

        # Hydrograph output section
        content += self._render_hydrograph_section(stream_nodes, n_stream_nodes)

        # Stream node budget section
        content += self._render_node_budget_section(stream_nodes, n_stream_nodes, streams)

        # Stream bed parameters (version-dependent columns)
        content += self._render_bed_params_section(stream_nodes, streams, version)

        # Hydraulic disconnection
        content += self._render_disconnection(streams)

        # v5.0 cross-section data and initial conditions
        if version >= (5, 0):
            content += self._render_cross_section(streams, stream_nodes)
            content += self._render_initial_conditions(streams, stream_nodes)

        # Stream evaporation
        content += self._render_evaporation(streams, stream_nodes)

        return content

    def _render_header_and_paths(
        self, has_inflows: bool, has_diversions: bool, has_bypasses: bool
    ) -> str:
        """Render version header and file path section using Jinja2 template."""
        subdir = self.config.stream_subdir
        if subdir:
            prefix = subdir + "\\"
        else:
            prefix = ""
        inflow_file = f"{prefix}{self.config.inflow_file}" if has_inflows else ""
        diver_file = f"{prefix}{self.config.diver_specs_file}" if has_diversions else ""
        bypass_file = f"{prefix}{self.config.bypass_specs_file}" if has_bypasses else ""
        div_data_file = f"{prefix}{self.config.diversions_file}" if has_diversions else ""

        generation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        version = parse_stream_version(self.config.version)

        template_name = (
            "streams/stream_main_v50.j2" if version >= (5, 0) else "streams/stream_main_v40.j2"
        )

        context = {
            "version": self.config.version,
            "generation_time": generation_time,
            "inflow_file": inflow_file,
            "diver_spec_file": diver_file,
            "bypass_spec_file": bypass_file,
            "div_data_file": div_data_file,
            "strm_budget_file": self.config.strm_budget_file,
            "diver_detail_file": self.config.diver_detail_file,
            "final_flow_file": self.config.final_flow_file or "",
        }

        return self._engine.render_template(template_name, **context)

    def _render_hydrograph_section(self, stream_nodes: list[int], n_stream_nodes: int) -> str:
        """Render hydrograph output section."""
        content = f"""C*******************************************************************************
C                       Stream Flow Hydrograph Output Data
C
C------------------------------------------------------------------------------
C   VALUE                         DESCRIPTION
C------------------------------------------------------------------------------
    {n_stream_nodes:<10}                    / NOUTR
    0                             / IHSQR
    {self.config.flow_factor:<14}           / FACTVROU
    {self.config.flow_unit:<14}             / UNITVROU
    {self.config.length_factor:<14}           / FACTLTOU
    {self.config.length_unit:<14}             / UNITLTOU
    {self.config.strm_hyd_file:<30} / STHYDOUTFL
C-------------------------------------------------------------------------------
C    IOUTR       NAME
C-------------------------------------------------------------------------------
"""
        for i, node_id in enumerate(stream_nodes):
            content += f"       {node_id:<6}      StrmHyd_{i + 1}\n"
        return content

    def _render_node_budget_section(
        self,
        stream_nodes: list[int],
        n_stream_nodes: int,
        streams: AppStream | None,
    ) -> str:
        """Render stream node budget section."""
        # Use model data if available, otherwise default to first 3 nodes
        budget_count = 0
        budget_ids: list[int] = []
        try:
            model_budget_count = streams.budget_node_count if streams is not None else 0
            if not isinstance(model_budget_count, int):
                model_budget_count = 0
        except AttributeError:
            model_budget_count = 0
        if model_budget_count > 0 and streams is not None:
            budget_count = model_budget_count
            try:
                budget_ids = streams.budget_node_ids
                if not isinstance(budget_ids, list):
                    budget_ids = []
            except AttributeError:
                budget_ids = []
        else:
            budget_count = min(3, n_stream_nodes)
            budget_ids = list(stream_nodes[:3])

        content = f"""C*******************************************************************************
C                       Stream Flow Budget at Selected Nodes
C
C------------------------------------------------------------------------------
C   VALUE                         DESCRIPTION
C------------------------------------------------------------------------------
     {budget_count:<10}                    / NBUDR
     {self.config.strm_node_budget_file:<30} / STNDBUDFL
C-------------------------------------------------------------------------------
C    IBUDR
C-------------------------------------------------------------------------------
"""
        for node_id in budget_ids:
            content += f"       {node_id}\n"
        return content

    def _render_bed_params_section(
        self,
        stream_nodes: list[int],
        streams: AppStream | None,
        version: tuple[int, int],
    ) -> str:
        """Render stream bed parameters with correct column count for version."""
        factk = self.config.conductivity_factor
        tunitsk = self.config.conductivity_time_unit
        factl = self.config.bed_length_factor

        # Use model data if available
        if streams is not None:
            try:
                model_factk = streams.conductivity_factor
                if isinstance(model_factk, (int, float)) and model_factk != 1.0:
                    factk = model_factk
            except AttributeError:
                pass
            try:
                model_tunitsk = streams.conductivity_time_unit
                if isinstance(model_tunitsk, str) and model_tunitsk:
                    tunitsk = model_tunitsk
            except AttributeError:
                pass
            try:
                model_factl = streams.length_factor
                if isinstance(model_factl, (int, float)) and model_factl != 1.0:
                    factl = model_factl
            except AttributeError:
                pass

        # Column format depends on version:
        # v4.0: IR  CSTRM  DSTRM  WETPR (4 columns)
        # v4.2: IR  WETPR  IRGW  CSTRM  DSTRM (5 columns)
        # v5.0+: IR  CSTRM  DSTRM (3 columns, no WETPR/IRGW)
        is_v50_plus = version >= (5, 0)
        is_v40 = version < (4, 1)
        if is_v50_plus:
            col_header = "C      IR    CSTRM   DSTRM"
        elif is_v40:
            col_header = "C      IR   CSTRM   DSTRM   WETPR"
        else:
            col_header = "C      IR      WETPR     IRGW   CSTRM   DSTRM"

        content = f"""C*******************************************************************************
C                       Stream Bed Parameters
C
C------------------------------------------------------------------------------
C   VALUE                 DESCRIPTION
C------------------------------------------------------------------------------
    {factk}                   / FACTK
    {tunitsk}                  / TUNITSK
    {factl}                   / FACTL
C------------------------------------------------------------------------------
{col_header}
C------------------------------------------------------------------------------
"""
        for node_id in stream_nodes:
            cstrm = self.config.conductivity
            dstrm = self.config.bed_thickness
            wetpr = self.config.wetted_perimeter
            gw_node = 0

            if streams is not None and node_id in streams.nodes:
                node = streams.nodes[node_id]
                try:
                    node_cond = node.conductivity
                    if isinstance(node_cond, (int, float)) and node_cond > 0:
                        cstrm = node_cond
                except AttributeError:
                    pass
                try:
                    node_thick = node.bed_thickness
                    if isinstance(node_thick, (int, float)) and node_thick > 0:
                        dstrm = node_thick
                except AttributeError:
                    pass
                try:
                    node_wp = node.wetted_perimeter
                    if isinstance(node_wp, (int, float)) and node_wp > 0:
                        wetpr = node_wp
                except AttributeError:
                    pass
                try:
                    if node.gw_node is not None and node.gw_node > 0:
                        gw_node = node.gw_node
                except AttributeError:
                    pass

            if is_v50_plus:
                content += f"{node_id:<8}   {cstrm:.3f}   {dstrm:.0f}\n"
            elif is_v40:
                content += f"{node_id:<8}   {cstrm:.3f}   {dstrm:.0f}{wetpr:>7.0f}\n"
            else:
                content += (
                    f"{node_id:<8}{wetpr:>7.0f}     {gw_node:>5d}   {cstrm:.3f}   {dstrm:.0f}\n"
                )

        return content

    def _render_disconnection(self, streams: AppStream | None) -> str:
        """Render hydraulic disconnection type."""
        intrc = self.config.interaction_type
        if streams is not None:
            try:
                model_intrc = streams.interaction_type
                if isinstance(model_intrc, int) and model_intrc != 1:
                    intrc = model_intrc
            except AttributeError:
                pass
        return f"""C*******************************************************************************
C                Hydraulic Disconnection for Stream-Aquifer Interaction
C
C------------------------------------------------------------------------------
C   VALUE                         DESCRIPTION
C------------------------------------------------------------------------------
     {intrc}                            / INTRCTYPE
"""

    def _render_cross_section(self, streams: AppStream | None, stream_nodes: list[int]) -> str:
        """Render v5.0 cross-section data."""
        roughness_factor = self.config.roughness_factor
        cs_length_factor = self.config.cross_section_length_factor
        if streams is not None:
            try:
                model_rf = streams.roughness_factor
                if isinstance(model_rf, (int, float)) and model_rf != 1.0:
                    roughness_factor = model_rf
            except AttributeError:
                pass
            try:
                model_clf = streams.cross_section_length_factor
                if isinstance(model_clf, (int, float)) and model_clf != 1.0:
                    cs_length_factor = model_clf
            except AttributeError:
                pass

        content = f"""C*******************************************************************************
C                       Stream Cross-Section Data (v5.0)
C
C------------------------------------------------------------------------------
C   VALUE                         DESCRIPTION
C------------------------------------------------------------------------------
    {roughness_factor}                   / FACTN
    {cs_length_factor}                   / FACTLT
C------------------------------------------------------------------------------
C    IR    BottomElev    B0       s        n     MaxDepth
C------------------------------------------------------------------------------
"""
        for node_id in stream_nodes:
            cs = None
            if streams is not None and node_id in streams.nodes:
                try:
                    cs = streams.nodes[node_id].cross_section
                except AttributeError:
                    pass
            if cs is not None:
                content += (
                    f"     {node_id:<6}  {cs.bottom_elev:>10.2f}"
                    f"  {cs.B0:>8.2f}  {cs.s:>6.3f}  {cs.n:>6.4f}"
                    f"  {cs.max_flow_depth:>8.2f}\n"
                )
            else:
                content += f"     {node_id:<6}        0.00      0.00   0.000  0.0400     10.00\n"
        return content

    def _render_initial_conditions(self, streams: AppStream | None, stream_nodes: list[int]) -> str:
        """Render v5.0 initial conditions."""
        ic_type = self.config.ic_type
        ic_time_unit = self.config.ic_time_unit or "1day"
        ic_factor = self.config.ic_factor
        if streams is not None:
            try:
                model_ic_type = streams.ic_type
                if isinstance(model_ic_type, int) and model_ic_type != 0:
                    ic_type = model_ic_type
            except AttributeError:
                pass
            try:
                model_ic_factor = streams.ic_factor
                if isinstance(model_ic_factor, (int, float)) and model_ic_factor != 1.0:
                    ic_factor = model_ic_factor
            except AttributeError:
                pass

        content = f"""C*******************************************************************************
C                       Stream Initial Conditions (v5.0)
C
C------------------------------------------------------------------------------
C   VALUE                         DESCRIPTION
C------------------------------------------------------------------------------
    {ic_type}                             / ICTYPE
    {ic_time_unit}                  / TUNITFLOW
    {ic_factor}                   / FACTH
C------------------------------------------------------------------------------
C    IR       IC_VALUE
C------------------------------------------------------------------------------
"""
        for node_id in stream_nodes:
            ic = 0.0
            if streams is not None and node_id in streams.nodes:
                try:
                    ic = streams.nodes[node_id].initial_condition
                except AttributeError:
                    pass
                if not isinstance(ic, (int, float)):
                    ic = 0.0
            content += f"     {node_id:<6}    {ic:>10.4f}\n"
        return content

    def _render_evaporation(self, streams: AppStream | None, stream_nodes: list[int]) -> str:
        """Render stream evaporation section.

        If no evaporation surface area file (STARFL) is available, all
        evaporation column pointers are forced to zero to prevent IWFM
        from requiring an ET data file.
        """
        import os
        from pathlib import Path as _Path

        evap_file = self.config.evap_area_file
        if streams is not None:
            try:
                model_evap = streams.evap_area_file
                if isinstance(model_evap, str) and model_evap:
                    evap_file = model_evap
            except AttributeError:
                pass

        # Convert absolute path to relative from simulation working dir.
        # First check if the file exists locally (e.g. copied by a previous step),
        # then fall back to computing relative from output_dir.
        if evap_file and os.path.isabs(evap_file):
            evap_path = _Path(evap_file)
            # Look for the same filename in the Streams/ or Stream/ dir
            local_candidates = [
                self.config.output_dir / "Streams" / evap_path.name,
                self.config.stream_dir / evap_path.name,
            ]
            found_local = False
            for candidate in local_candidates:
                if candidate.exists():
                    evap_file = os.path.relpath(candidate, self.config.output_dir)
                    found_local = True
                    break
            if not found_local:
                try:
                    evap_file = os.path.relpath(evap_file, self.config.output_dir)
                except ValueError:
                    pass  # different drive

        # Determine if evaporation is actually active
        has_evap_file = bool(evap_file and evap_file.strip())

        content = f"""C*******************************************************************************
C                            Stream Evaporation
C
C------------------------------------------------------------------------------
C   VALUE                         DESCRIPTION
C------------------------------------------------------------------------------
    {evap_file:<30} / STARFL
C------------------------------------------------------------------------------
C    IR    ICETST    ICARST
C------------------------------------------------------------------------------
"""
        evap_specs: list = []
        if streams is not None:
            try:
                evap_specs = streams.evap_node_specs
                if not isinstance(evap_specs, list):
                    evap_specs = []
            except AttributeError:
                pass
        if evap_specs:
            for spec in evap_specs:
                if has_evap_file:
                    content += (
                        f"     {spec.node_id:<6}    {spec.et_column:>5}    {spec.area_column:>5}\n"
                    )
                else:
                    # Zero out evap columns when no surface area file
                    content += f"     {spec.node_id:<6}        0        0\n"

        return content

    def write_diver_specs(self) -> Path:
        """
        Write the diversion specifications file.

        Writes IWFM 14-column (no spills) diversion specification format
        including NDIVER, per-diversion parameters, element groups, and
        recharge zones.

        Returns
        -------
        Path
            Path to written file
        """
        output_path = self.config.stream_dir / self.config.diver_specs_file
        self._ensure_dir(output_path)

        streams = self.model.streams
        diversions = streams.diversions if streams else {}

        generation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        content = f"""C*******************************************************************************
C                  DIVERSION SPECIFICATIONS FILE
C
C             Generated by pyiwfm
C             {generation_time}
C*******************************************************************************
C
C   NDIVER  ; Number of diversions
C
C-------------------------------------------------------------------------------
    {len(diversions):<10}                    / NDIVER
C-------------------------------------------------------------------------------
"""

        if diversions:
            has_spills = streams.diversion_has_spills if streams else False

            if has_spills:
                content += """C
C   ID  IRDV  ICDVMAX  FDVMAX  ICOLRL  FRACRL  ICOLNL  FRACNL  ICOLSP  FRACSP  TYPDSTDL  DSTDL  ICOLDL  FRACDL  ICFSIRIG  ICADJ  NAME
C-------------------------------------------------------------------------------
"""
            else:
                content += """C
C   ID  IRDV  ICDVMAX  FDVMAX  ICOLRL  FRACRL  ICOLNL  FRACNL  TYPDSTDL  DSTDL  ICOLDL  FRACDL  ICFSIRIG  ICADJ  NAME
C-------------------------------------------------------------------------------
"""

            for div_id in sorted(diversions.keys()):
                div = diversions[div_id]
                if has_spills:
                    content += (
                        f"    {div.id:>5}"
                        f" {div.source_node:>10}"
                        f" {div.max_div_column:>10}"
                        f" {div.max_div_fraction:>10}"
                        f" {div.recoverable_loss_column:>10}"
                        f" {div.recoverable_loss_fraction:>10.4f}"
                        f" {div.non_recoverable_loss_column:>10}"
                        f" {div.non_recoverable_loss_fraction:>10.4f}"
                        f" {div.spill_column:>10}"
                        f" {div.spill_fraction:>10.4f}"
                        f" {div.delivery_dest_type:>10}"
                        f" {div.delivery_dest_id:>10}"
                        f" {div.delivery_column:>10}"
                        f" {div.delivery_fraction:>10.4f}"
                        f" {div.irrigation_fraction_column:>10}"
                        f" {div.adjustment_column:>10}"
                        f"     {div.name}"
                        f"\n"
                    )
                else:
                    content += (
                        f"    {div.id:>5}"
                        f" {div.source_node:>10}"
                        f" {div.max_div_column:>10}"
                        f" {div.max_div_fraction:>10}"
                        f" {div.recoverable_loss_column:>10}"
                        f" {div.recoverable_loss_fraction:>10.4f}"
                        f" {div.non_recoverable_loss_column:>10}"
                        f" {div.non_recoverable_loss_fraction:>10.4f}"
                        f" {div.delivery_dest_type:>10}"
                        f" {div.delivery_dest_id:>10}"
                        f" {div.delivery_column:>10}"
                        f" {div.delivery_fraction:>10.4f}"
                        f" {div.irrigation_fraction_column:>10}"
                        f" {div.adjustment_column:>10}"
                        f"     {div.name}"
                        f"\n"
                    )

            # Element groups section
            elem_groups = streams.diversion_element_groups if streams else []
            content += f"""C-------------------------------------------------------------------------------
C                  Element Groups for Diversion Deliveries
C
C   NGRP    ; Number of element groups
C   ID      ; Element group ID
C   NELEM   ; Number of elements in element group ID
C   IELEM   ; Element numbers in group ID
C
C    ID         NELEM      IELEM
C-------------------------------------------------------------------------------
     {len(elem_groups):<10}           / NGRP
C-------------------------------------------------------------------------------
"""
            for group in elem_groups:
                g_id = group.id
                elems = group.elements
                n_elems = len(elems)
                if n_elems > 0:
                    content += f"    {g_id:>10} {n_elems:>15} {elems[0]:>15}\n"
                    for elem in elems[1:]:
                        content += f"                              {elem:>15}\n"
                else:
                    content += f"    {g_id:>10} {0:>15} {0:>15}\n"

            # Recharge zones section
            content += """C-------------------------------------------------------------------------------
C                  Recharge zone for each diversion point
C
C   ID    NERELS  IERELS     FERELS
C-------------------------------------------------------------------------------
"""
            rz_map = {}
            recharge_zones = streams.diversion_recharge_zones if streams else []
            for rz in recharge_zones:
                rz_map[rz.diversion_id] = rz

            for div_id in sorted(diversions.keys()):
                rz = rz_map.get(div_id)
                if rz and rz.n_zones > 0:
                    content += f"    {div_id:>5} {rz.n_zones:>8} {rz.zone_ids[0]:>8} {rz.zone_fractions[0]:>8.4f}\n"
                    for zid, zfrac in zip(rz.zone_ids[1:], rz.zone_fractions[1:], strict=False):
                        content += f"                    {zid:>8} {zfrac:>8.4f}\n"
                else:
                    content += f"    {div_id:>5}        0         0       0\n"

            # Spill zones section (only if 16-column format)
            if has_spills:
                content += """C-------------------------------------------------------------------------------
C                  Spill zones for each diversion point
C
C   ID    NSPILLS  NODEID     FRACTION
C-------------------------------------------------------------------------------
"""
                sz_map = {}
                spill_zones = streams.diversion_spill_zones if streams else []
                for sz in spill_zones:
                    sz_map[sz.diversion_id] = sz

                for div_id in sorted(diversions.keys()):
                    sz = sz_map.get(div_id)
                    if sz and sz.n_zones > 0:
                        content += f"    {div_id:>5} {sz.n_zones:>8} {sz.zone_ids[0]:>8} {sz.zone_fractions[0]:>8.4f}\n"
                        for zid, zfrac in zip(sz.zone_ids[1:], sz.zone_fractions[1:], strict=False):
                            content += f"                    {zid:>8} {zfrac:>8.4f}\n"
                    else:
                        content += f"    {div_id:>5}        0         0       0\n"

        output_path.write_text(content)
        logger.info(f"Wrote diversion specs file: {output_path}")
        return output_path

    def write_bypass_specs(self) -> Path:
        """
        Write the bypass specifications file.

        IWFM format:
        - NBYPASS, FACTX, TUNITX, FACTY, TUNITY (header)
        - Per bypass: ID, IA, TYPEDEST, DEST, IDIVC, DIVRL, DIVNL, NAME
        - Rating table rows (DIVX, DIVY) if IDIVC < 0
        - Seepage/recharge zone data for each bypass

        Returns
        -------
        Path
            Path to written file
        """
        output_path = self.config.stream_dir / self.config.bypass_specs_file
        self._ensure_dir(output_path)

        streams = self.model.streams
        bypasses = streams.bypasses if streams else {}

        generation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        content = f"""C*******************************************************************************
C                  BYPASS SPECIFICATIONS FILE
C
C             Generated by pyiwfm
C             {generation_time}
C*******************************************************************************
C
C   NBYPASS ; Number of bypasses
C
C-------------------------------------------------------------------------------
    {len(bypasses):<10}                    / NBYPASS
C-------------------------------------------------------------------------------
"""

        if bypasses:
            # Get conversion factors from first bypass
            first_bp = next(iter(bypasses.values()))
            factx = first_bp.flow_factor
            tunitx = first_bp.flow_time_unit or "1DAY"
            facty = first_bp.spill_factor
            tunity = first_bp.spill_time_unit or "1DAY"

            content += f"""    {factx}                        / FACTX
    {tunitx:<14}                / TUNITX
    {facty}                        / FACTY
    {tunity:<14}                / TUNITY
C-------------------------------------------------------------------------------
C
C   ID    IA    TYPEDEST    DEST    IDIVC    DIVRL    DIVNL    NAME
C
C-------------------------------------------------------------------------------
"""
            for bypass_id in sorted(bypasses.keys()):
                bp = bypasses[bypass_id]

                # Determine IDIVC: negative = inline rating table points
                idivc = bp.diversion_column
                if idivc == 0 and bp.rating_table_flows:
                    idivc = -len(bp.rating_table_flows)

                content += (
                    f"    {bp.id:<5}"
                    f" {bp.source_node:>7}"
                    f" {bp.dest_type:>5}"
                    f" {bp.destination_node:>7}"
                    f" {idivc:>7}"
                    f" {bp.recoverable_loss_fraction:>8.4f}"
                    f" {bp.non_recoverable_loss_fraction:>8.4f}"
                    f"    {bp.name}"
                    f"\n"
                )

                # Write inline rating table if IDIVC < 0
                if idivc < 0 and bp.rating_table_flows:
                    for flow_val, spill_val in zip(
                        bp.rating_table_flows, bp.rating_table_spills, strict=False
                    ):
                        content += f"                    {flow_val}     {spill_val}\n"

            # Seepage/recharge zone section
            content += """C-------------------------------------------------------------------------------
C
C               Seepage locations for bypass canals
C
C   ID    NERELS    IERELS    FERELS
C-------------------------------------------------------------------------------
"""
            for bypass_id in sorted(bypasses.keys()):
                bp = bypasses[bypass_id]
                if bp.seepage_locations:
                    # Write first seepage zone (typically only one per bypass)
                    sl = bp.seepage_locations[0]
                    n_elems = getattr(sl, "n_elements", 0)
                    elem_ids = getattr(sl, "element_ids", [])
                    elem_fracs = getattr(sl, "element_fractions", [])
                    if n_elems > 0 and elem_ids:
                        content += (
                            f"    {bp.id:<5} {n_elems:>8} {elem_ids[0]:>8} {elem_fracs[0]:>8.4f}\n"
                        )
                        for eid, efrac in zip(elem_ids[1:], elem_fracs[1:], strict=False):
                            content += f"                    {eid:>8} {efrac:>8.4f}\n"
                    else:
                        content += f"    {bp.id:<5}        0         0       0\n"
                else:
                    # Default: no seepage elements
                    content += f"    {bp.id:<5}        0         0       0\n"

        output_path.write_text(content)
        logger.info(f"Wrote bypass specs file: {output_path}")
        return output_path

    def write_stream_inflow_ts(
        self,
        dates: list[str] | None = None,
        data: NDArray | None = None,
        column_mapping: list[str] | None = None,
    ) -> Path:
        """
        Write the stream inflow time series data file.

        Parameters
        ----------
        dates : list[str], optional
            IWFM timestamps
        data : NDArray, optional
            Inflow data array (n_times, n_cols)
        column_mapping : list[str], optional
            Column mapping rows (ID, IRST)

        Returns
        -------
        Path
            Path to written file
        """
        from pyiwfm.io.timeseries_writer import (
            IWFMTimeSeriesDataWriter,
            make_stream_inflow_ts_config,
        )

        streams = self.model.streams
        n_cols = len(streams.nodes) if streams and streams.nodes else 0

        ts_config = make_stream_inflow_ts_config(
            ncol=n_cols,
            dates=dates,
            data=data,
            column_mapping=column_mapping or [],
            column_header="ID    IRST (Stream node ID, Reach ID)",
        )

        output_path = self.config.stream_dir / self.config.inflow_file
        writer = IWFMTimeSeriesDataWriter(self._engine)
        return writer.write(ts_config, output_path)

    def write_diversion_data_ts(
        self,
        dates: list[str] | None = None,
        data: NDArray | None = None,
    ) -> Path:
        """
        Write the diversion data time series file.

        Parameters
        ----------
        dates : list[str], optional
            IWFM timestamps
        data : NDArray, optional
            Diversion data array (n_times, n_cols)

        Returns
        -------
        Path
            Path to written file
        """
        from pyiwfm.io.timeseries_writer import (
            IWFMTimeSeriesDataWriter,
            make_diversion_ts_config,
        )

        streams = self.model.streams
        n_cols = len(streams.diversions) if streams and streams.diversions else 0

        ts_config = make_diversion_ts_config(
            ncol=n_cols,
            dates=dates,
            data=data,
        )

        output_path = self.config.stream_dir / self.config.diversions_file
        writer = IWFMTimeSeriesDataWriter(self._engine)
        return writer.write(ts_config, output_path)

    def write_surface_area_ts(
        self,
        dates: list[str] | None = None,
        data: NDArray | None = None,
    ) -> Path:
        """
        Write the stream surface area time series file.

        Parameters
        ----------
        dates : list[str], optional
            IWFM timestamps
        data : NDArray, optional
            Surface area data array (n_times, n_cols)

        Returns
        -------
        Path
            Path to written file
        """
        from pyiwfm.io.timeseries_writer import (
            IWFMTimeSeriesDataWriter,
            make_stream_surface_area_ts_config,
        )

        streams = self.model.streams
        n_cols = len(streams.nodes) if streams and streams.nodes else 0

        ts_config = make_stream_surface_area_ts_config(
            ncol=n_cols,
            dates=dates,
            data=data,
        )

        output_path = self.config.stream_dir / self.config.evap_area_file
        writer = IWFMTimeSeriesDataWriter(self._engine)
        return writer.write(ts_config, output_path)


def write_stream_component(
    model: IWFMModel,
    output_dir: Path | str,
    config: StreamWriterConfig | None = None,
) -> dict[str, Path]:
    """
    Write stream component files for a model.

    Parameters
    ----------
    model : IWFMModel
        Model to write
    output_dir : Path or str
        Output directory
    config : StreamWriterConfig, optional
        File configuration

    Returns
    -------
    dict[str, Path]
        Mapping of file type to output path
    """
    output_dir = Path(output_dir)

    if config is None:
        config = StreamWriterConfig(output_dir=output_dir)
    else:
        config.output_dir = output_dir

    writer = StreamComponentWriter(model, config)
    return writer.write_all()
