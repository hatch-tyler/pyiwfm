"""
PreProcessor file I/O handlers for IWFM models.

This module provides functions for reading and writing IWFM PreProcessor
input files, which define the model structure and input file paths.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pyiwfm.core.mesh import AppGrid, Node, Element, Subregion
from pyiwfm.core.stratigraphy import Stratigraphy
from pyiwfm.core.model import IWFMModel
from pyiwfm.core.exceptions import FileFormatError
from pyiwfm.io.ascii import (
    read_nodes,
    read_elements,
    read_stratigraphy,
    write_nodes,
    write_elements,
    write_stratigraphy,
)


# IWFM comment characters - these must be in column 1 (first character)
COMMENT_CHARS = ("C", "c", "*")


def _is_comment_line(line: str) -> bool:
    """
    Check if a line is a comment line.

    In IWFM Fortran format, a comment line has the comment character
    in column 1 (the very first character of the line), not after
    leading whitespace. Lines starting with whitespace followed by
    data are data lines.
    """
    # Empty or whitespace-only lines are treated as comments
    if not line.strip():
        return True
    # Check first character of the original line (column 1)
    # Note: lines starting with whitespace followed by data are NOT comments
    return line[0] in COMMENT_CHARS


def _parse_value_line(line: str) -> tuple[str, str]:
    """
    Parse an IWFM value line with optional description.

    Format: VALUE / DESCRIPTION  or just VALUE

    Uses "/" as the inline comment delimiter.  The separator is
    always preceded by whitespace so we look for ``whitespace + /``
    to avoid splitting on ``/`` inside dates.
    """
    import re

    # Find the first occurrence of whitespace followed by '/'
    m = re.search(r"\s+/", line)
    if m:
        return line[: m.start()].strip(), line[m.end() :].strip()

    return line.strip(), ""


def _resolve_path(base_dir: Path, filepath: str) -> Path:
    """
    Resolve a file path relative to a base directory.

    IWFM paths can be:
    - Absolute paths
    - Relative paths (relative to the main input file)
    """
    path = Path(filepath.strip())
    if path.is_absolute():
        return path
    return base_dir / path


@dataclass
class PreProcessorConfig:
    """
    Configuration from an IWFM PreProcessor main input file.

    Contains paths to all component input files and model settings.
    """

    base_dir: Path
    model_name: str = ""

    # Core file paths
    nodes_file: Path | None = None
    elements_file: Path | None = None
    stratigraphy_file: Path | None = None
    subregions_file: Path | None = None

    # Optional component file paths
    streams_file: Path | None = None
    lakes_file: Path | None = None
    groundwater_file: Path | None = None
    rootzone_file: Path | None = None
    pumping_file: Path | None = None

    # Output settings
    output_dir: Path | None = None
    budget_output_file: Path | None = None
    heads_output_file: Path | None = None

    # Model settings
    n_layers: int = 1
    length_unit: str = "FT"
    area_unit: str = "ACRES"
    volume_unit: str = "AF"

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)


def read_preprocessor_main(filepath: Path | str) -> PreProcessorConfig:
    """
    Read an IWFM PreProcessor main input file.

    The main input file contains paths to all component input files
    and basic model configuration.

    Expected format (simplified):
        Model Name                / Description
        nodes.dat                 / NODES_FILE
        elements.dat              / ELEMENTS_FILE
        stratigraphy.dat          / STRAT_FILE
        ...

    Args:
        filepath: Path to the main PreProcessor input file

    Returns:
        PreProcessorConfig with parsed values
    """
    filepath = Path(filepath)
    base_dir = filepath.parent

    config = PreProcessorConfig(base_dir=base_dir)

    with open(filepath, "r") as f:
        line_num = 0
        data_lines: list[tuple[str, str]] = []  # (value, description)

        for line in f:
            line_num += 1
            if _is_comment_line(line):
                continue

            value, desc = _parse_value_line(line)
            if value:
                data_lines.append((value, desc.upper()))

    # Parse the data lines based on position and description hints
    # IWFM PreProcessor files have a fairly standard structure
    idx = 0

    # First non-comment line is typically the model name or a version
    if idx < len(data_lines):
        value, desc = data_lines[idx]
        if "NAME" in desc or "TITLE" in desc or idx == 0:
            config.model_name = value
            idx += 1

    # Look for file paths by description
    for value, desc in data_lines[idx:]:
        value_path = _resolve_path(base_dir, value) if not value.isdigit() else None

        if "NODE" in desc and "FILE" in desc:
            config.nodes_file = value_path
        elif "ELEM" in desc and "FILE" in desc:
            config.elements_file = value_path
        elif "STRAT" in desc and "FILE" in desc:
            config.stratigraphy_file = value_path
        elif "SUBREG" in desc and "FILE" in desc:
            config.subregions_file = value_path
        elif "STREAM" in desc and "FILE" in desc:
            config.streams_file = value_path
        elif "LAKE" in desc and "FILE" in desc:
            config.lakes_file = value_path
        elif "GROUND" in desc and "FILE" in desc:
            config.groundwater_file = value_path
        elif "ROOT" in desc and "FILE" in desc:
            config.rootzone_file = value_path
        elif "PUMP" in desc and "FILE" in desc:
            config.pumping_file = value_path
        elif "LAYER" in desc:
            try:
                config.n_layers = int(value)
            except ValueError:
                pass
        elif "LENGTH" in desc and "UNIT" in desc:
            config.length_unit = value.upper()
        elif "AREA" in desc and "UNIT" in desc:
            config.area_unit = value.upper()
        elif "VOLUME" in desc and "UNIT" in desc:
            config.volume_unit = value.upper()
        elif "OUTPUT" in desc and "DIR" in desc:
            config.output_dir = value_path

    return config


def read_subregions_file(filepath: Path | str) -> dict[int, Subregion]:
    """
    Read subregion definitions from a subregions file.

    Expected format:
        NSUBREGION                / Number of subregions
        ID  NAME                  (one line per subregion)

    Args:
        filepath: Path to the subregions file

    Returns:
        Dictionary mapping subregion ID to Subregion object
    """
    filepath = Path(filepath)
    subregions: dict[int, Subregion] = {}

    with open(filepath, "r") as f:
        line_num = 0
        n_subregions = None

        # Find NSUBREGION
        for line in f:
            line_num += 1
            if _is_comment_line(line):
                continue

            value, _ = _parse_value_line(line)
            try:
                n_subregions = int(value)
            except ValueError as e:
                raise FileFormatError(
                    f"Invalid NSUBREGION value: '{value}'", line_number=line_num
                ) from e
            break

        if n_subregions is None:
            raise FileFormatError("Could not find NSUBREGION in file")

        # Read subregion data
        for line in f:
            line_num += 1
            if _is_comment_line(line):
                continue

            parts = line.split(None, 1)  # Split on first whitespace
            if len(parts) < 1:
                continue

            try:
                sr_id = int(parts[0])
                sr_name = parts[1].strip() if len(parts) > 1 else ""
            except ValueError as e:
                raise FileFormatError(
                    f"Invalid subregion data: '{line.strip()}'", line_number=line_num
                ) from e

            subregions[sr_id] = Subregion(id=sr_id, name=sr_name)

    return subregions


def load_model_from_preprocessor(
    pp_filepath: Path | str,
    load_components: bool = False,
) -> IWFMModel:
    """
    Load an IWFMModel from PreProcessor input files.

    This reads the main PreProcessor file and all referenced input files
    to construct a complete model.

    Args:
        pp_filepath: Path to the main PreProcessor input file
        load_components: If True, also load stream, lake, etc. components

    Returns:
        IWFMModel instance
    """
    pp_filepath = Path(pp_filepath)
    config = read_preprocessor_main(pp_filepath)

    # Read nodes
    if config.nodes_file is None:
        raise FileFormatError("Nodes file not specified in PreProcessor file")
    nodes = read_nodes(config.nodes_file)

    # Read elements
    if config.elements_file is None:
        raise FileFormatError("Elements file not specified in PreProcessor file")
    elements, n_subregions, subregion_names = read_elements(config.elements_file)

    # Read subregions: prefer separate file, fall back to names from element file
    subregions: dict[int, Subregion] = {}
    if config.subregions_file and config.subregions_file.exists():
        subregions = read_subregions_file(config.subregions_file)
    elif subregion_names:
        subregions = {
            sr_id: Subregion(id=sr_id, name=name)
            for sr_id, name in subregion_names.items()
        }

    # Create mesh
    mesh = AppGrid(nodes=nodes, elements=elements, subregions=subregions)
    mesh.compute_areas()
    mesh.compute_connectivity()

    # Read stratigraphy
    stratigraphy = None
    if config.stratigraphy_file and config.stratigraphy_file.exists():
        stratigraphy = read_stratigraphy(config.stratigraphy_file)

    # Create model
    model = IWFMModel(
        name=config.model_name or pp_filepath.stem,
        mesh=mesh,
        stratigraphy=stratigraphy,
        metadata={
            "preprocessor_file": str(pp_filepath),
            "length_unit": config.length_unit,
            "area_unit": config.area_unit,
            "volume_unit": config.volume_unit,
        },
    )

    return model


def write_preprocessor_main(
    filepath: Path | str,
    config: PreProcessorConfig,
    header: str | None = None,
) -> None:
    """
    Write an IWFM PreProcessor main input file.

    Args:
        filepath: Path to the output file
        config: PreProcessorConfig with file paths and settings
        header: Optional header comment
    """
    filepath = Path(filepath)

    with open(filepath, "w") as f:
        # Write header
        if header:
            for line in header.strip().split("\n"):
                f.write(f"C  {line}\n")
        else:
            f.write("C  IWFM PreProcessor Main Input File\n")
            f.write("C  Generated by pyiwfm\n")
            f.write("C\n")

        # Write model name
        f.write(f"{config.model_name:<40} / MODEL_NAME\n")

        # Write core files
        if config.nodes_file:
            rel_path = _make_relative_path(filepath.parent, config.nodes_file)
            f.write(f"{rel_path:<40} / NODES_FILE\n")

        if config.elements_file:
            rel_path = _make_relative_path(filepath.parent, config.elements_file)
            f.write(f"{rel_path:<40} / ELEMENTS_FILE\n")

        if config.stratigraphy_file:
            rel_path = _make_relative_path(filepath.parent, config.stratigraphy_file)
            f.write(f"{rel_path:<40} / STRATIGRAPHY_FILE\n")

        if config.subregions_file:
            rel_path = _make_relative_path(filepath.parent, config.subregions_file)
            f.write(f"{rel_path:<40} / SUBREGIONS_FILE\n")

        # Write settings
        f.write(f"{config.n_layers:<40} / N_LAYERS\n")
        f.write(f"{config.length_unit:<40} / LENGTH_UNIT\n")
        f.write(f"{config.area_unit:<40} / AREA_UNIT\n")
        f.write(f"{config.volume_unit:<40} / VOLUME_UNIT\n")


def _make_relative_path(base_dir: Path, target_path: Path) -> str:
    """Make a path relative to a base directory if possible."""
    try:
        return str(target_path.relative_to(base_dir))
    except ValueError:
        return str(target_path)


def save_model_to_preprocessor(
    model: IWFMModel,
    output_dir: Path | str,
    model_name: str | None = None,
) -> PreProcessorConfig:
    """
    Save an IWFMModel to PreProcessor input files.

    Creates all necessary input files in the output directory.

    Args:
        model: IWFMModel to save
        output_dir: Directory for output files
        model_name: Model name (defaults to model.name)

    Returns:
        PreProcessorConfig with paths to created files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    name = model_name or model.name or "iwfm_model"

    # Create config
    config = PreProcessorConfig(
        base_dir=output_dir,
        model_name=name,
        n_layers=model.n_layers,
    )

    # Write nodes
    if model.mesh:
        config.nodes_file = output_dir / "nodes.dat"
        write_nodes(config.nodes_file, model.mesh.nodes)

        # Write elements
        config.elements_file = output_dir / "elements.dat"
        n_sr = model.mesh.n_subregions or 1
        write_elements(config.elements_file, model.mesh.elements, n_subregions=n_sr)

        # Write subregions if present
        if model.mesh.subregions:
            config.subregions_file = output_dir / "subregions.dat"
            _write_subregions_file(config.subregions_file, model.mesh.subregions)

    # Write stratigraphy
    if model.stratigraphy:
        config.stratigraphy_file = output_dir / "stratigraphy.dat"
        write_stratigraphy(config.stratigraphy_file, model.stratigraphy)

    # Write main input file
    main_file = output_dir / f"{name}_pp.in"
    write_preprocessor_main(main_file, config)

    return config


def _write_subregions_file(
    filepath: Path | str,
    subregions: dict[int, Subregion],
) -> None:
    """Write subregion definitions to a file."""
    filepath = Path(filepath)

    with open(filepath, "w") as f:
        f.write("C  Subregion definitions\n")
        f.write("C  ID  NAME\n")
        f.write(f"{len(subregions):<10}                    / NSUBREGION\n")

        for sr_id in sorted(subregions.keys()):
            sr = subregions[sr_id]
            f.write(f"{sr.id:<5} {sr.name}\n")


def load_complete_model(
    simulation_file: Path | str,
    load_timeseries: bool = False,
) -> IWFMModel:
    """
    Load a complete IWFM model from a simulation main input file.

    This reads the simulation main file and all referenced component input files
    to construct a complete model with groundwater, streams, lakes, and rootzone
    components.

    Args:
        simulation_file: Path to the simulation main input file
        load_timeseries: If True, also load time series data (can be slow)

    Returns:
        IWFMModel instance with all components loaded

    Example:
        >>> from pyiwfm.io import load_complete_model
        >>> model = load_complete_model("simulation.in")
        >>> print(f"Loaded model with {model.groundwater.n_wells} wells")
    """
    from pyiwfm.io.simulation import SimulationReader
    from pyiwfm.io.groundwater import GroundwaterReader
    from pyiwfm.io.streams import StreamReader
    from pyiwfm.io.lakes import LakeReader
    from pyiwfm.io.rootzone import RootZoneReader

    simulation_file = Path(simulation_file)
    base_dir = simulation_file.parent

    # Read simulation config
    sim_reader = SimulationReader()
    sim_config = sim_reader.read(simulation_file)

    # First load the preprocessor/mesh data
    if sim_config.preprocessor_file and sim_config.preprocessor_file.exists():
        model = load_model_from_preprocessor(sim_config.preprocessor_file)
    else:
        # Try to find preprocessor file in same directory
        pp_candidates = list(base_dir.glob("*_pp.in")) + list(base_dir.glob("*preprocessor*.in"))
        if pp_candidates:
            model = load_model_from_preprocessor(pp_candidates[0])
        else:
            # Create empty model
            model = IWFMModel(name=sim_config.model_name)

    # Update model metadata with simulation settings
    model.metadata["simulation_file"] = str(simulation_file)
    model.metadata["start_date"] = sim_config.start_date.isoformat()
    model.metadata["end_date"] = sim_config.end_date.isoformat()
    model.metadata["time_step_length"] = sim_config.time_step_length
    model.metadata["time_step_unit"] = sim_config.time_step_unit.value

    # Load groundwater component
    if sim_config.groundwater_file:
        gw_file = _resolve_path(base_dir, str(sim_config.groundwater_file))
        if gw_file.exists():
            try:
                gw_reader = GroundwaterReader()
                wells = gw_reader.read_wells(gw_file)

                # Create AppGW component
                from pyiwfm.components.groundwater import AppGW
                n_nodes = model.mesh.n_nodes if model.mesh else 0
                n_elements = model.mesh.n_elements if model.mesh else 0
                n_layers = model.n_layers

                gw = AppGW(n_nodes=n_nodes, n_layers=n_layers, n_elements=n_elements)
                for well in wells.values():
                    gw.add_well(well)

                # Also try hierarchical reader for aquifer parameters,
                # hydrographs, and initial heads
                try:
                    from pyiwfm.io.groundwater import GWMainFileReader

                    gw_main_reader = GWMainFileReader()
                    gw_config = gw_main_reader.read(gw_file, base_dir=base_dir)

                    for loc in gw_config.hydrograph_locations:
                        gw.add_hydrograph_location(loc)

                    if gw_config.aquifer_params is not None:
                        try:
                            gw.set_aquifer_parameters(gw_config.aquifer_params)
                        except ValueError:
                            gw.aquifer_params = gw_config.aquifer_params
                    elif gw_config.parametric_grids and model.mesh:
                        try:
                            from pyiwfm.core.model import _apply_parametric_grids
                            _apply_parametric_grids(
                                gw, gw_config.parametric_grids, model.mesh,
                            )
                        except Exception:
                            pass

                    # Apply Kh anomaly overwrites
                    if (gw_config.kh_anomalies
                            and gw.aquifer_params is not None
                            and model.mesh):
                        try:
                            from pyiwfm.core.model import _apply_kh_anomalies
                            _apply_kh_anomalies(
                                gw.aquifer_params,
                                gw_config.kh_anomalies,
                                model.mesh,
                            )
                        except Exception:
                            pass

                    if gw_config.initial_heads is not None:
                        try:
                            gw.set_heads(gw_config.initial_heads)
                        except ValueError:
                            pass
                except Exception:
                    pass

                model.groundwater = gw
            except Exception as e:
                model.metadata["groundwater_load_error"] = str(e)

    # Load streams component
    if sim_config.streams_file:
        stream_file = _resolve_path(base_dir, str(sim_config.streams_file))
        if stream_file.exists():
            try:
                stream_reader = StreamReader()
                nodes = stream_reader.read_stream_nodes(stream_file)

                from pyiwfm.components.stream import AppStream
                stream = AppStream()
                for node in nodes.values():
                    stream.add_node(node)

                model.streams = stream
            except Exception as e:
                model.metadata["streams_load_error"] = str(e)

    # Load lakes component
    if sim_config.lakes_file:
        lake_file = _resolve_path(base_dir, str(sim_config.lakes_file))
        if lake_file.exists():
            try:
                lake_reader = LakeReader()
                lakes_dict = lake_reader.read_lake_definitions(lake_file)

                from pyiwfm.components.lake import AppLake
                lakes = AppLake()
                for lake in lakes_dict.values():
                    lakes.add_lake(lake)

                model.lakes = lakes
            except Exception as e:
                model.metadata["lakes_load_error"] = str(e)

    # Load rootzone component
    if sim_config.rootzone_file:
        rz_file = _resolve_path(base_dir, str(sim_config.rootzone_file))
        if rz_file.exists():
            try:
                rz_reader = RootZoneReader()
                crops = rz_reader.read_crop_types(rz_file)

                from pyiwfm.components.rootzone import RootZone
                n_elements = model.mesh.n_elements if model.mesh else 0
                rootzone = RootZone(n_elements=n_elements, n_layers=1)
                for crop in crops.values():
                    rootzone.add_crop_type(crop)

                model.rootzone = rootzone
            except Exception as e:
                model.metadata["rootzone_load_error"] = str(e)

    model.metadata["source"] = "simulation"
    return model


def save_complete_model(
    model: IWFMModel,
    output_dir: Path | str,
    timeseries_format: str = "ascii",
    dss_file: Path | str | None = None,
    file_paths: dict[str, str] | None = None,
) -> dict[str, Path]:
    """
    Save a complete IWFM model to all input files.

    This is the main function for exporting a complete model, including
    all component files (groundwater, streams, lakes, rootzone), time series,
    and the preprocessor/simulation control files.

    Delegates to :class:`~pyiwfm.io.model_writer.CompleteModelWriter` for
    the actual writing.

    Args:
        model: IWFMModel to save
        output_dir: Directory for output files
        timeseries_format: Format for time series data ("ascii" or "dss")
        dss_file: Path to DSS file for time series (if format is "dss")
        file_paths: Optional dict of {file_key: relative_path} overrides.
            If None, uses default nested layout.

    Returns:
        Dictionary mapping file type to output path

    Example:
        >>> from pyiwfm.io import save_complete_model
        >>> files = save_complete_model(model, Path("./model_output"))
        >>> print(f"Wrote {len(files)} files")
    """
    from pyiwfm.io.config import ModelWriteConfig, OutputFormat
    from pyiwfm.io.model_writer import CompleteModelWriter

    ts_fmt = OutputFormat.DSS if timeseries_format == "dss" else OutputFormat.TEXT

    # Propagate component versions from loaded model metadata so the
    # writer produces files matching the original model's version tags.
    version_kwargs: dict[str, str] = {}
    for key in ("gw_version", "stream_version", "lake_version", "rootzone_version"):
        val = model.metadata.get(key)
        if val:
            version_kwargs[key] = val

    config = ModelWriteConfig(
        output_dir=Path(output_dir),
        ts_format=ts_fmt,
        dss_file=dss_file or "model.dss",
        file_paths=file_paths or {},
        **version_kwargs,
    )
    writer = CompleteModelWriter(model, config)
    result = writer.write_all()
    return result.files
