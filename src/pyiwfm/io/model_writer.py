"""
Complete Model Writer for IWFM models.

This module provides the main writer for exporting a complete IWFM model
to disk with per-file path control and time series format conversion.

Supports optional comment preservation for round-trip operations.

Classes:
    ModelWriteResult: Result of a complete model write operation.
    TimeSeriesCopier: Copies/converts time series files between locations.
    CompleteModelWriter: Orchestrates all component writers.

Functions:
    write_model: Convenience function for writing a complete model.
    write_model_with_comments: Write model with preserved comments.
    save_model_with_comments: High-level API for comment-preserving writes.
"""

from __future__ import annotations

import logging
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pyiwfm.io.config import ModelWriteConfig, OutputFormat

if TYPE_CHECKING:
    from pyiwfm.core.model import IWFMModel
    from pyiwfm.io.comment_metadata import CommentMetadata

logger = logging.getLogger(__name__)


# Maps source_files keys → ModelWriteConfig file_paths keys
TS_KEY_MAPPING: dict[str, str] = {
    "gw_pumping_ts": "gw_ts_pumping",
    "gw_bc_ts": "gw_bound_tsd",
    "stream_inflow_ts": "stream_inflow",
    "stream_diversion_ts": "stream_diversions",
    "lake_max_elev_ts": "lake_max_elev",
    "precipitation_ts": "precipitation",
    "et_ts": "et",
    "irig_frac_ts": "irig_frac",
    "rootzone_return_flow_ts": "rootzone_return_flow",
    "rootzone_reuse_ts": "rootzone_reuse",
    "rootzone_irig_period_ts": "rootzone_irig_period",
    "rootzone_ag_demand_ts": "rootzone_surface_flow_dest",
}

# DSS C-part parameter codes for each TS type
TS_DSS_PARAMS: dict[str, str] = {
    "precipitation_ts": "PRECIP",
    "et_ts": "ET",
    "gw_pumping_ts": "PUMP",
    "gw_bc_ts": "FLOW",
    "stream_inflow_ts": "FLOW",
    "stream_diversion_ts": "DIVERT",
    "lake_max_elev_ts": "ELEV",
    "irig_frac_ts": "FRAC",
    "rootzone_return_flow_ts": "RETURN",
    "rootzone_reuse_ts": "REUSE",
    "rootzone_irig_period_ts": "IRIG-PER",
    "rootzone_ag_demand_ts": "DEMAND",
}


def _compute_dss_interval(times: list) -> str:
    """Compute DSS interval string from a list of datetime objects."""
    if len(times) < 2:
        return "1MON"
    from pyiwfm.io.dss.pathname import minutes_to_interval

    delta = times[1] - times[0]
    minutes = int(delta.total_seconds() / 60)
    return minutes_to_interval(minutes)


@dataclass
class ModelWriteResult:
    """Result of a complete model write operation.

    Attributes:
        files: Mapping of file_key to written output path.
        errors: Mapping of component name to error message.
        warnings: List of non-fatal warning messages.
    """

    files: dict[str, Path] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """True if no errors occurred during writing."""
        return len(self.errors) == 0


class TimeSeriesCopier:
    """Copies or converts time series files from source to destination.

    Handles format conversion between text and DSS formats when the
    target format differs from the source format.
    """

    def __init__(
        self,
        model: IWFMModel,
        config: ModelWriteConfig,
    ) -> None:
        self.model = model
        self.config = config

    def copy_all(self) -> tuple[dict[str, Path], list[str]]:
        """Copy all time series files from source to destination.

        Returns:
            Tuple of (files dict, warnings list).
        """
        files: dict[str, Path] = {}
        warnings: list[str] = []

        for source_key, dest_key in TS_KEY_MAPPING.items():
            source_path = self.model.source_files.get(source_key)
            if source_path is None:
                continue

            source_path = Path(source_path)
            if not source_path.exists():
                warnings.append(f"Source TS file not found: {source_key} -> {source_path}")
                continue

            try:
                dest_path = self.config.get_path(dest_key)
            except KeyError:
                warnings.append(f"No destination key '{dest_key}' for source '{source_key}'")
                continue

            dest_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                self._copy_or_convert(source_path, dest_path, source_key)
                files[dest_key] = dest_path
            except Exception as e:
                warnings.append(f"Failed to copy {source_key}: {e}")

        return files, warnings

    def _copy_or_convert(
        self,
        source: Path,
        dest: Path,
        key: str,
    ) -> None:
        """Copy or convert a single time series file."""
        source_is_dss = source.suffix.lower() == ".dss"
        target_is_dss = self.config.ts_format == OutputFormat.DSS

        if not source_is_dss and target_is_dss:
            # Text -> DSS: read text, write DSS data, write stub .dat file
            self._convert_text_to_dss(source, dest, key)
        else:
            # Text -> Text, DSS -> DSS, or DSS -> Text: copy as-is
            shutil.copy2(source, dest)
            logger.debug("Copied TS file: %s -> %s", source, dest)

    def _convert_text_to_dss(
        self,
        source: Path,
        dest: Path,
        key: str,
    ) -> None:
        """Convert a text time series file to DSS format.

        Reads the source text file, writes data to the shared DSS file,
        and writes a stub ``.dat`` file at *dest* with DSSFL and DSS
        pathnames so that IWFM reads data from the DSS file.

        Falls back to a plain copy if the DSS library is not available.
        """
        from pyiwfm.io.timeseries_ascii import TimeSeriesReader
        from pyiwfm.io.timeseries_writer import (
            DSSPathItem,
            IWFMTimeSeriesDataWriter,
            TimeSeriesDataConfig,
            make_ag_water_demand_ts_config,
            make_diversion_ts_config,
            make_et_ts_config,
            make_irig_period_ts_config,
            make_max_lake_elev_ts_config,
            make_precip_ts_config,
            make_pumping_ts_config,
            make_return_flow_ts_config,
            make_reuse_ts_config,
            make_stream_inflow_ts_config,
        )

        # Factory mapping (keyed by source TS key)
        factories: dict[str, Callable[..., TimeSeriesDataConfig]] = {
            "precipitation_ts": make_precip_ts_config,
            "et_ts": make_et_ts_config,
            "gw_pumping_ts": make_pumping_ts_config,
            "stream_inflow_ts": make_stream_inflow_ts_config,
            "stream_diversion_ts": make_diversion_ts_config,
            "lake_max_elev_ts": make_max_lake_elev_ts_config,
            "rootzone_return_flow_ts": make_return_flow_ts_config,
            "rootzone_reuse_ts": make_reuse_ts_config,
            "rootzone_irig_period_ts": make_irig_period_ts_config,
            "rootzone_ag_demand_ts": make_ag_water_demand_ts_config,
        }

        # Step 1: Read source text file
        reader = TimeSeriesReader()
        times, values, ts_config = reader.read(source)

        import numpy as np

        if values.ndim == 1:
            values = values.reshape(-1, 1)
        actual_cols = values.shape[1]

        # Step 2: Compute DSS parameters
        dss_param = TS_DSS_PARAMS.get(key, "DATA")
        dss_interval = _compute_dss_interval(times)
        a_part = (self.config.dss_a_part or self.config.model_name).upper()
        f_part = self.config.dss_f_part.upper()

        # Step 3: Write data to DSS file
        dss_path = self.config.get_path("dss_ts_file")
        dss_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            from pyiwfm.io.dss.pathname import DSSPathnameTemplate
            from pyiwfm.io.dss.timeseries import (
                DSSTimeSeriesWriter as DSSWriter,
            )

            # Build location -> values mapping
            values_dict: dict[str, np.ndarray] = {}
            for i in range(actual_cols):
                location = f"ELEM_{i + 1}"
                values_dict[location] = values[:, i]

            template = DSSPathnameTemplate(
                a_part=a_part,
                c_part=dss_param,
                e_part=dss_interval,
                f_part=f_part,
            )

            with DSSWriter(dss_path) as dss_writer:
                dss_writer.write_multiple_timeseries(
                    times=times,
                    values_dict=values_dict,
                    template=template,
                )

            logger.info(
                "Wrote %d DSS records for %s to %s",
                actual_cols,
                key,
                dss_path,
            )
        except ImportError:
            logger.warning("DSS library not available; copying raw file for %s", key)
            shutil.copy2(source, dest)
            return

        # Step 4: Build DSS pathnames for stub file (empty D-part)
        dss_paths: list[DSSPathItem] = []
        for i in range(actual_cols):
            location = f"ELEM_{i + 1}"
            pathname = f"/{a_part}/{location}/{dss_param}//{dss_interval}/{f_part}/"
            dss_paths.append(DSSPathItem(index=i + 1, path=pathname))

        # Step 5: Compute relative DSS file path from stub's directory
        dss_rel = os.path.relpath(dss_path, dest.parent)
        dss_rel = str(Path(dss_rel))  # OS-native separators

        # Step 6: Create TimeSeriesDataConfig for stub file.
        # Use factor=1.0 because the data written to DSS already has
        # the original factor applied (the reader applies it on read).
        factory_fn = factories.get(key)
        if factory_fn is not None:
            stub_config = factory_fn(ncol=actual_cols, factor=1.0)
        else:
            stub_config = TimeSeriesDataConfig(ncol=actual_cols, factor=1.0)

        stub_config.dss_file = dss_rel
        stub_config.dss_paths = dss_paths

        # Step 7: Write stub .dat file
        ts_writer = IWFMTimeSeriesDataWriter()
        ts_writer.write(stub_config, dest)

        logger.info("Wrote DSS stub: %s (DSSFL=%s)", dest, dss_rel)


class CompleteModelWriter:
    """Orchestrates all component writers to produce a complete IWFM model.

    Uses ``ModelWriteConfig`` for all path resolution, enabling arbitrary
    directory structures. Each component writer receives paths derived from
    the config, and cross-file references use ``get_relative_path()``.

    Supports optional comment preservation via comment_metadata parameter.
    When provided, preserved comments from the original files will be
    injected into the output files.

    Example::

        config = ModelWriteConfig(output_dir=Path("C:/models/output"))
        writer = CompleteModelWriter(model, config)
        result = writer.write_all()
        if result.success:
            print(f"Wrote {len(result.files)} files")

    Example with comment preservation::

        model, comments = load_model_with_comments("Simulation/Main.in")
        config = ModelWriteConfig(output_dir=Path("output"))
        writer = CompleteModelWriter(model, config, comment_metadata=comments)
        result = writer.write_all()
    """

    def __init__(
        self,
        model: IWFMModel,
        config: ModelWriteConfig,
        comment_metadata: dict[str, CommentMetadata] | None = None,
        preserve_comments: bool = True,
    ) -> None:
        """Initialize the complete model writer.

        Args:
            model: IWFMModel instance to write.
            config: Configuration for output paths and formats.
            comment_metadata: Dictionary mapping file type to CommentMetadata.
                Keys should match file type names (e.g., "preprocessor_main",
                "gw_main", "stream_main").
            preserve_comments: If True and comment_metadata is provided,
                inject preserved comments into output files.
        """
        self.model = model
        self.config = config
        self.comment_metadata = comment_metadata or {}
        self.preserve_comments = preserve_comments

    def get_file_comments(self, file_type: str) -> CommentMetadata | None:
        """Get comment metadata for a specific file type.

        Args:
            file_type: File type key (e.g., "preprocessor_main").

        Returns:
            CommentMetadata if available and preservation is enabled.
        """
        if self.preserve_comments:
            return self.comment_metadata.get(file_type)
        return None

    def write_all(self) -> ModelWriteResult:
        """Write the complete model to disk.

        Returns:
            ModelWriteResult with files written, errors, and warnings.
        """
        result = ModelWriteResult()

        # Ensure base output directory exists
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Phase 1: Write preprocessor files
        try:
            pp_files = self.write_preprocessor()
            result.files.update(pp_files)
        except Exception as e:
            result.errors["preprocessor"] = str(e)
            logger.error(f"Preprocessor write failed: {e}")

        # Phase 2: Write component files
        self._write_groundwater(result)
        self._write_streams(result)
        self._write_lakes(result)
        self._write_rootzone(result)

        # Phase 3: Copy time series files
        if self.config.copy_source_ts:
            self._copy_timeseries(result)

        # Phase 3b: Copy component files the writer doesn't regenerate
        self._copy_passthrough_components(result)

        # Phase 3c: Write supply adjustment file
        self._write_supply_adjustment(result)

        # Phase 4: Write simulation main file (references all above)
        try:
            sim_path = self._write_simulation_main()
            result.files["simulation_main"] = sim_path
        except Exception as e:
            result.errors["simulation_main"] = str(e)
            logger.error(f"Simulation main write failed: {e}")

        return result

    def write_preprocessor(self) -> dict[str, Path]:
        """Write preprocessor files (nodes, elements, stratigraphy).

        Returns:
            Mapping of file type to output path.
        """
        from pyiwfm.io.config import PreProcessorFileConfig
        from pyiwfm.io.preprocessor_writer import PreProcessorWriter

        pp_main_path = self.config.get_path("preprocessor_main")
        pp_dir = pp_main_path.parent

        pp_config = PreProcessorFileConfig(
            output_dir=pp_dir,
            main_file=pp_main_path.name,
            node_file=self.config.get_path("nodes").name,
            element_file=self.config.get_path("elements").name,
            stratigraphy_file=self.config.get_path("stratigraphy").name,
            stream_config_file=self.config.get_path("stream_config").name,
            lake_config_file=self.config.get_path("lake_config").name,
            binary_output_file=self.config.get_relative_path(
                "preprocessor_main", "preprocessor_bin"
            ),
        )

        # Handle nodes/elements/stratigraphy in different dirs than PP main
        # by ensuring each file's parent dir exists
        for key in ("nodes", "elements", "stratigraphy"):
            self.config.get_path(key).parent.mkdir(parents=True, exist_ok=True)

        writer = PreProcessorWriter(self.model, pp_config)
        raw_files = writer.write_all()

        files: dict[str, Path] = {}
        for k, v in raw_files.items():
            files[k] = v
        files["preprocessor_main"] = raw_files.get("main", pp_main_path)
        return files

    def _write_groundwater(self, result: ModelWriteResult) -> None:
        """Write groundwater component files."""
        if self.model.groundwater is None:
            return

        try:
            from pyiwfm.io.gw_writer import GWComponentWriter, GWWriterConfig

            gw_main_path = self.config.get_path("gw_main")
            gw_dir = gw_main_path.parent
            sim_dir = self.config.get_path("simulation_main").parent

            # gw_subdir controls both output directory nesting AND path
            # references.  Set output_dir to the Simulation directory so
            # the writer creates files in gw_dir and references them with
            # the correct relative prefix (e.g. "GW\TileDrain.dat").
            gw_subdir = os.path.relpath(gw_dir, sim_dir)
            if gw_subdir == ".":
                gw_subdir = ""

            gw_config = GWWriterConfig(
                output_dir=sim_dir,
                gw_subdir=gw_subdir,
                version=self.config.gw_version,
                main_file=gw_main_path.name,
                bc_main_file=self.config.get_path("gw_bc_main").name,
                pump_main_file=self.config.get_path("gw_pump_main").name,
                tile_drain_file=self.config.get_path("gw_tile_drain").name,
                subsidence_file=self.config.get_path("gw_subsidence").name,
                elem_pump_file=self.config.get_path("gw_elem_pump").name,
                well_spec_file=self.config.get_path("gw_well_spec").name,
                ts_pumping_file=self.config.get_path("gw_ts_pumping").name,
                spec_head_bc_file=self.config.get_path("gw_spec_head_bc").name,
                spec_flow_bc_file=self.config.get_path("gw_spec_flow_bc").name,
                bound_tsd_file=self.config.get_path("gw_bound_tsd").name,
                # Output file paths relative from simulation working dir
                gw_budget_file=self.config.get_relative_path(
                    "simulation_main", "results_gw_budget"
                ),
                gw_zbudget_file=self.config.get_relative_path(
                    "simulation_main", "results_gw_zbudget"
                ),
                gw_head_file=self.config.get_relative_path("simulation_main", "results_gw_head"),
            )

            writer = GWComponentWriter(self.model, gw_config)
            gw_files = writer.write_all()
            for k, v in gw_files.items():
                result.files[f"gw_{k}"] = v
        except Exception as e:
            result.errors["groundwater"] = str(e)
            logger.error(f"Groundwater write failed: {e}")

    def _write_streams(self, result: ModelWriteResult) -> None:
        """Write stream component files."""
        if self.model.streams is None:
            return

        try:
            from pyiwfm.io.stream_writer import (
                StreamComponentWriter,
                StreamWriterConfig,
            )

            stream_main_path = self.config.get_path("stream_main")
            stream_dir = stream_main_path.parent
            sim_dir = self.config.get_path("simulation_main").parent

            # stream_subdir controls both the output directory nesting AND
            # the path prefix in the stream main file.  Set output_dir to
            # the Simulation directory and stream_subdir to the relative
            # path from Simulation to Stream so the writer creates files
            # in the right place and references them correctly.
            stream_subdir = os.path.relpath(stream_dir, sim_dir)
            if stream_subdir == ".":
                stream_subdir = ""

            stream_config = StreamWriterConfig(
                output_dir=sim_dir,
                stream_subdir=stream_subdir,
                version=self.config.stream_version,
                main_file=stream_main_path.name,
                inflow_file=self.config.get_path("stream_inflow").name,
                diver_specs_file=self.config.get_path("stream_diver_specs").name,
                bypass_specs_file=self.config.get_path("stream_bypass_specs").name,
                diversions_file=self.config.get_path("stream_diversions").name,
                strm_budget_file=self.config.get_relative_path(
                    "simulation_main", "results_strm_budget"
                ),
            )

            writer = StreamComponentWriter(self.model, stream_config)
            stream_files = writer.write_all()
            for k, v in stream_files.items():
                result.files[f"stream_{k}"] = v
        except Exception as e:
            result.errors["streams"] = str(e)
            logger.error(f"Stream write failed: {e}")

    def _write_lakes(self, result: ModelWriteResult) -> None:
        """Write lake component files."""
        if self.model.lakes is None:
            return

        try:
            from pyiwfm.io.lake_writer import LakeComponentWriter, LakeWriterConfig

            lake_main_path = self.config.get_path("lake_main")
            lake_dir = lake_main_path.parent

            lake_config = LakeWriterConfig(
                output_dir=lake_dir,
                lake_subdir="",
                version=self.config.lake_version,
                main_file=lake_main_path.name,
                max_elev_file=self.config.get_path("lake_max_elev").name,
                lake_budget_file=self.config.get_relative_path(
                    "simulation_main", "results_lake_budget"
                ),
            )

            writer = LakeComponentWriter(self.model, lake_config)
            lake_files = writer.write_all()
            for k, v in lake_files.items():
                result.files[f"lake_{k}"] = v
        except Exception as e:
            result.errors["lakes"] = str(e)
            logger.error(f"Lake write failed: {e}")

    def _write_rootzone(self, result: ModelWriteResult) -> None:
        """Write root zone component files."""
        if self.model.rootzone is None:
            return

        try:
            from pyiwfm.io.rootzone_writer import (
                RootZoneComponentWriter,
                RootZoneWriterConfig,
            )

            rz_main_path = self.config.get_path("rootzone_main")
            rz_dir = rz_main_path.parent
            sim_dir = self.config.get_path("simulation_main").parent

            # rootzone_subdir controls both output nesting AND path
            # references, same as gw_subdir and stream_subdir.
            rz_subdir = os.path.relpath(rz_dir, sim_dir)
            if rz_subdir == ".":
                rz_subdir = ""

            rz_config = RootZoneWriterConfig(
                output_dir=sim_dir,
                rootzone_subdir=rz_subdir,
                version=self.config.rootzone_version,
                main_file=rz_main_path.name,
                return_flow_file=self.config.get_path("rootzone_return_flow").name,
                reuse_file=self.config.get_path("rootzone_reuse").name,
                irig_period_file=self.config.get_path("rootzone_irig_period").name,
                surface_flow_dest_file=self.config.get_path("rootzone_surface_flow_dest").name,
                lwu_budget_file=self.config.get_relative_path(
                    "simulation_main", "results_lwu_budget"
                ),
                rz_budget_file=self.config.get_relative_path(
                    "simulation_main", "results_rz_budget"
                ),
            )

            writer = RootZoneComponentWriter(self.model, rz_config)
            rz_files = writer.write_all()
            for k, v in rz_files.items():
                result.files[f"rootzone_{k}"] = v
        except Exception as e:
            result.errors["rootzone"] = str(e)
            logger.error(f"Root zone write failed: {e}")

    def _copy_timeseries(self, result: ModelWriteResult) -> None:
        """Copy time series files from source to destination."""
        copier = TimeSeriesCopier(self.model, self.config)
        ts_files, ts_warnings = copier.copy_all()
        result.files.update(ts_files)
        result.warnings.extend(ts_warnings)

    def _copy_passthrough_components(self, result: ModelWriteResult) -> None:
        """Write or copy supplemental component files.

        For Small Watersheds and Unsaturated Zone, uses dedicated writers
        when component data is available on the model, otherwise falls back
        to copying the source file.
        """
        self._write_small_watersheds(result)
        self._write_unsaturated_zone(result)

    def _write_small_watersheds(self, result: ModelWriteResult) -> None:
        """Write or copy small watershed component files."""
        if self.model.small_watersheds is not None:
            try:
                from pyiwfm.io.small_watershed_writer import (
                    SmallWatershedComponentWriter,
                    SmallWatershedWriterConfig,
                )

                sw_main_path = self.config.get_path("swshed_main")
                sw_dir = sw_main_path.parent

                sw_config = SmallWatershedWriterConfig(
                    output_dir=sw_dir,
                    swshed_subdir="",
                    version=self.model.metadata.get("small_watershed_version", "4.0"),
                    main_file=sw_main_path.name,
                )

                writer = SmallWatershedComponentWriter(self.model, sw_config)
                sw_files = writer.write_all()
                for k, v in sw_files.items():
                    result.files[f"swshed_{k}"] = v
                return
            except Exception as e:
                result.warnings.append(
                    f"Failed to write small watershed from data, falling back to copy: {e}"
                )

        # Fallback: copy source file
        source_path = self.model.source_files.get("swshed_main")
        if source_path is None:
            return
        source_path = Path(source_path)
        if not source_path.exists():
            result.warnings.append(f"Passthrough source not found: swshed_main -> {source_path}")
            return
        try:
            dest_path = self.config.get_path("swshed_main")
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, dest_path)
            result.files["swshed_main"] = dest_path
            logger.info(f"Copied passthrough component: swshed_main -> {dest_path}")
        except Exception as e:
            result.warnings.append(f"Failed to copy passthrough swshed_main: {e}")

    def _write_unsaturated_zone(self, result: ModelWriteResult) -> None:
        """Write or copy unsaturated zone component files."""
        if self.model.unsaturated_zone is not None:
            try:
                from pyiwfm.io.unsaturated_zone_writer import (
                    UnsatZoneComponentWriter,
                    UnsatZoneWriterConfig,
                )

                uz_main_path = self.config.get_path("unsatzone_main")
                uz_dir = uz_main_path.parent

                uz_config = UnsatZoneWriterConfig(
                    output_dir=uz_dir,
                    unsatzone_subdir="",
                    version=self.model.metadata.get("unsat_zone_version", "4.0"),
                    main_file=uz_main_path.name,
                )

                writer = UnsatZoneComponentWriter(self.model, uz_config)
                uz_files = writer.write_all()
                for k, v in uz_files.items():
                    result.files[f"unsatzone_{k}"] = v
                return
            except Exception as e:
                result.warnings.append(
                    f"Failed to write unsaturated zone from data, falling back to copy: {e}"
                )

        # Fallback: copy source file
        source_path = self.model.source_files.get("unsatzone_main")
        if source_path is None:
            return
        source_path = Path(source_path)
        if not source_path.exists():
            result.warnings.append(f"Passthrough source not found: unsatzone_main -> {source_path}")
            return
        try:
            dest_path = self.config.get_path("unsatzone_main")
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, dest_path)
            result.files["unsatzone_main"] = dest_path
            logger.info(f"Copied passthrough component: unsatzone_main -> {dest_path}")
        except Exception as e:
            result.warnings.append(f"Failed to copy passthrough unsatzone_main: {e}")

    def _write_supply_adjustment(self, result: ModelWriteResult) -> None:
        """Write the supply adjustment file.

        Uses parsed SupplyAdjustment data if available on the model.
        Falls back to copying the source file if parsing wasn't done.
        """
        # Check if model has parsed supply adjustment data
        if self.model.supply_adjustment is not None:
            try:
                from pyiwfm.io.supply_adjust import write_supply_adjustment

                dest_path = self.config.get_path("supply_adjust")
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                write_supply_adjustment(self.model.supply_adjustment, dest_path)
                result.files["supply_adjust"] = dest_path
                logger.info("Wrote supply adjustment: %s", dest_path)
                return
            except Exception as e:
                result.warnings.append(f"Failed to write supply adjustment from data: {e}")
                # Fall through to passthrough copy

        # Fallback: copy source file if available
        source_path = self.model.source_files.get("supply_adjust")
        if source_path is None:
            return

        source_path = Path(source_path)
        if not source_path.exists():
            result.warnings.append(f"Supply adjustment source not found: {source_path}")
            return

        try:
            dest_path = self.config.get_path("supply_adjust")
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, dest_path)
            result.files["supply_adjust"] = dest_path
            logger.info("Copied supply adjustment: %s -> %s", source_path, dest_path)
        except Exception as e:
            result.warnings.append(f"Failed to copy supply adjustment: {e}")

    def _write_simulation_main(self) -> Path:
        """Write the Simulation_MAIN.IN file referencing all components."""
        from pyiwfm.io.simulation_writer import (
            SimulationMainConfig,
            SimulationMainWriter,
        )

        sim_main_path = self.config.get_path("simulation_main")
        sim_dir = sim_main_path.parent

        # Build relative paths from simulation_main to each component
        sim_config = SimulationMainConfig(
            output_dir=sim_dir,
            main_file=sim_main_path.name,
            preprocessor_bin=self.config.get_relative_path("simulation_main", "preprocessor_bin"),
            gw_main=self.config.get_relative_path("simulation_main", "gw_main"),
            stream_main=self.config.get_relative_path("simulation_main", "stream_main"),
            rootzone_main=self.config.get_relative_path("simulation_main", "rootzone_main"),
            irig_frac=self.config.get_relative_path("simulation_main", "irig_frac"),
            precip_file=self.config.get_relative_path("simulation_main", "precipitation"),
            et_file=self.config.get_relative_path("simulation_main", "et"),
        )

        # Lake: only set path if model has lakes
        if self.model.lakes and self.model.lakes.n_lakes > 0:
            sim_config.lake_main = self.config.get_relative_path("simulation_main", "lake_main")
        else:
            sim_config.lake_main = ""

        # Supply adjustment: only set path if model has supply adjustment data
        if self.model.supply_adjustment is not None or self.model.source_files.get("supply_adjust"):
            sim_config.supply_adjust = self.config.get_relative_path(
                "simulation_main", "supply_adjust"
            )
        else:
            sim_config.supply_adjust = ""

        # Small watershed and unsaturated zone (pass-through, not regenerated)
        if self.model.source_files.get("swshed_main"):
            sim_config.swshed_main = self.config.get_relative_path("simulation_main", "swshed_main")
        if self.model.source_files.get("unsatzone_main"):
            sim_config.unsatzone_main = self.config.get_relative_path(
                "simulation_main", "unsatzone_main"
            )

        # Populate simulation period from model metadata
        meta = self.model.metadata
        if meta.get("start_date"):
            # Convert ISO date to IWFM format
            sim_config.begin_date = _iso_to_iwfm_date(meta["start_date"])
        if meta.get("end_date"):
            sim_config.end_date = _iso_to_iwfm_date(meta["end_date"])
        if meta.get("time_step_length") and meta.get("time_step_unit"):
            sim_config.time_step = f"{meta['time_step_length']}{meta['time_step_unit']}"

        # Solver parameters
        if meta.get("matrix_solver") is not None:
            sim_config.matrix_solver = meta["matrix_solver"]
        if meta.get("relaxation") is not None:
            sim_config.relaxation = meta["relaxation"]
        if meta.get("max_iterations") is not None:
            sim_config.max_iterations = meta["max_iterations"]
        if meta.get("max_supply_iterations") is not None:
            sim_config.max_supply_iter = meta["max_supply_iterations"]
        if meta.get("convergence_tolerance") is not None:
            sim_config.convergence_head = meta["convergence_tolerance"]
        if meta.get("convergence_volume") is not None:
            sim_config.convergence_volume = meta["convergence_volume"]
        if meta.get("convergence_supply") is not None:
            sim_config.convergence_supply = meta["convergence_supply"]
        if meta.get("supply_adjust_option") is not None:
            sim_config.supply_adjust_option = meta["supply_adjust_option"]
        if meta.get("debug_flag") is not None:
            sim_config.kdeb = meta["debug_flag"]
        if meta.get("cache_size") is not None:
            sim_config.cache_size = meta["cache_size"]

        # Title lines
        title_lines = meta.get("title_lines", [])
        if len(title_lines) > 0:
            sim_config.title1 = title_lines[0]
        if len(title_lines) > 1:
            sim_config.title2 = title_lines[1]
        if len(title_lines) > 2:
            sim_config.title3 = title_lines[2]

        writer = SimulationMainWriter(self.model, sim_config)
        return writer.write_main()


def _iso_to_iwfm_date(iso_str: str) -> str:
    """Convert ISO date string to IWFM date format.

    IWFM uses MM/DD/YYYY_HH:MM (16 chars) with 24:00 for midnight.
    Midnight is represented as 24:00 of the *previous* day, e.g.
    '1990-10-01T00:00:00' -> '09/30/1990_24:00' (end of Sep 30).
    """
    from datetime import datetime

    try:
        dt = datetime.fromisoformat(iso_str)
        from pyiwfm.io.timeseries_ascii import format_iwfm_timestamp

        return format_iwfm_timestamp(dt)
    except (ValueError, TypeError):
        return iso_str


def write_model(
    model: IWFMModel,
    output_dir: Path | str,
    file_paths: dict[str, str] | None = None,
    ts_format: str = "text",
    **kwargs: Any,
) -> ModelWriteResult:
    """Write a complete IWFM model to disk.

    This is the main convenience function for writing a model. It creates
    a ``ModelWriteConfig`` and delegates to ``CompleteModelWriter``.

    Args:
        model: IWFMModel instance to write.
        output_dir: Base output directory.
        file_paths: Optional dict of {file_key: relative_path} overrides.
            If None, uses default nested layout.
        ts_format: Time series format - ``"text"`` or ``"dss"``.
        **kwargs: Additional arguments passed to ``ModelWriteConfig``.

    Returns:
        ModelWriteResult with written files, errors, and warnings.

    Example::

        result = write_model(model, "C:/models/output")
        if result.success:
            print(f"Model written to {len(result.files)} files")
    """
    fmt = OutputFormat.DSS if ts_format.lower() == "dss" else OutputFormat.TEXT

    # Propagate component versions from model metadata unless the caller
    # explicitly overrides them via **kwargs.
    version_defaults: dict[str, str] = {}
    for key in ("gw_version", "stream_version", "lake_version", "rootzone_version"):
        if key not in kwargs:
            val = model.metadata.get(key)
            if val:
                version_defaults[key] = val

    config = ModelWriteConfig(
        output_dir=Path(output_dir),
        file_paths=file_paths or {},
        ts_format=fmt,
        **{**version_defaults, **kwargs},  # type: ignore[arg-type]
    )
    writer = CompleteModelWriter(model, config)
    return writer.write_all()


# =============================================================================
# Comment-Preserving Model Writing
# =============================================================================


def write_model_with_comments(
    model: IWFMModel,
    output_dir: Path | str,
    comment_metadata: dict[str, CommentMetadata] | None = None,
    file_paths: dict[str, str] | None = None,
    ts_format: str = "text",
    save_sidecars: bool = True,
    **kwargs: Any,
) -> ModelWriteResult:
    """Write a complete IWFM model with preserved comments.

    This is like write_model() but with support for comment preservation.
    When comment_metadata is provided, preserved comments from the original
    files are injected into the output files.

    Args:
        model: IWFMModel instance to write.
        output_dir: Base output directory.
        comment_metadata: Dictionary mapping file type to CommentMetadata.
            Typically obtained from load_model_with_comments().
        file_paths: Optional dict of {file_key: relative_path} overrides.
        ts_format: Time series format - ``"text"`` or ``"dss"``.
        save_sidecars: If True, save comment metadata as sidecar files
            alongside the output files for future round-trips.
        **kwargs: Additional arguments passed to ``ModelWriteConfig``.

    Returns:
        ModelWriteResult with written files, errors, and warnings.

    Example::

        # Load model with comments
        model, comments = load_model_with_comments("Simulation/Main.in")

        # Modify the model
        model.nodes.add_node(...)

        # Write back with preserved comments
        result = write_model_with_comments(
            model,
            "output/",
            comment_metadata=comments,
        )
    """
    fmt = OutputFormat.DSS if ts_format.lower() == "dss" else OutputFormat.TEXT

    # Propagate component versions from model metadata
    version_defaults: dict[str, str] = {}
    for key in ("gw_version", "stream_version", "lake_version", "rootzone_version"):
        if key not in kwargs:
            val = model.metadata.get(key)
            if val:
                version_defaults[key] = val

    config = ModelWriteConfig(
        output_dir=Path(output_dir),
        file_paths=file_paths or {},
        ts_format=fmt,
        **{**version_defaults, **kwargs},  # type: ignore[arg-type]
    )

    writer = CompleteModelWriter(
        model,
        config,
        comment_metadata=comment_metadata,
        preserve_comments=True,
    )
    result = writer.write_all()

    # Save sidecar files for future round-trips
    if save_sidecars and comment_metadata:
        _save_comment_sidecars(result.files, comment_metadata)

    return result


def _save_comment_sidecars(
    written_files: dict[str, Path],
    comment_metadata: dict[str, CommentMetadata],
) -> None:
    """Save comment metadata as sidecar files.

    Args:
        written_files: Dictionary of file type to written path.
        comment_metadata: Dictionary of file type to CommentMetadata.
    """

    for file_type, metadata in comment_metadata.items():
        # Map file type to written path
        output_path = written_files.get(file_type)
        if output_path is None:
            # Try common mappings
            mappings = {
                "preprocessor_main": "preprocessor_main",
                "simulation_main": "simulation_main",
                "gw_main": "gw_main",
                "stream_main": "stream_main",
                "lake_main": "lake_main",
                "rootzone_main": "rootzone_main",
            }
            mapped_key = mappings.get(file_type)
            if mapped_key:
                output_path = written_files.get(mapped_key)

        if output_path and output_path.exists():
            try:
                metadata.save_for_file(output_path)
                logger.debug(f"Saved comment sidecar for {file_type}")
            except Exception as e:
                logger.warning(f"Failed to save comment sidecar for {file_type}: {e}")


def save_model_with_comments(
    model: IWFMModel,
    output_dir: Path | str,
    comment_metadata: dict[str, CommentMetadata] | None = None,
) -> dict[str, Path]:
    """High-level API for writing a model with preserved comments.

    This is the simplest API for round-trip model operations with
    comment preservation.

    Args:
        model: IWFMModel instance to write.
        output_dir: Base output directory.
        comment_metadata: Dictionary mapping file type to CommentMetadata.
            If None, writes without comment preservation (uses templates).

    Returns:
        Dictionary mapping file type to written path.

    Raises:
        RuntimeError: If writing fails with errors.

    Example::

        # Load model with comments
        model, comments = load_model_with_comments("Simulation/Main.in")

        # Modify the model
        model.nodes.add_node(...)

        # Write back with preserved comments
        files = save_model_with_comments(model, "output/", comments)
        print(f"Wrote files: {list(files.keys())}")

        # Or write NEW model without preserved comments (uses templates)
        new_files = save_model_with_comments(new_model, "output/")
    """
    result = write_model_with_comments(
        model,
        output_dir,
        comment_metadata=comment_metadata,
    )

    if not result.success:
        errors = "; ".join(f"{k}: {v}" for k, v in result.errors.items())
        raise RuntimeError(f"Failed to write model: {errors}")

    return result.files
