"""
Discover hydrograph output files from an IWFM simulation main file.

Parses the simulation main file → component main files → .out file paths
and hydrograph metadata (bore IDs, layer numbers).  Ports the model-file-
discovery logic from the old ``iwfm2obs_2015`` Fortran program.

Reuses existing pyiwfm readers wherever possible:

- :func:`pyiwfm.io.iwfm_reader.next_data_value` for line parsing
- :func:`pyiwfm.io.iwfm_reader.resolve_path` for path resolution
- :func:`pyiwfm.io.iwfm_reader.is_comment_line` for comment detection
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

from pyiwfm.components.groundwater import HydrographLocation
from pyiwfm.io.iwfm_reader import (
    is_comment_line as _is_comment_line,
)
from pyiwfm.io.iwfm_reader import (
    resolve_path as _resolve_path,
)
from pyiwfm.io.iwfm_reader import (
    strip_inline_comment as _strip_inline_comment,
)

logger = logging.getLogger(__name__)


@dataclass
class HydrographFileInfo:
    """Paths and metadata for discovered hydrograph output files.

    Attributes
    ----------
    gw_hydrograph_path : Path or None
        Path to the GW hydrograph .out file.
    stream_hydrograph_path : Path or None
        Path to the stream hydrograph .out file.
    subsidence_hydrograph_path : Path or None
        Path to the subsidence hydrograph .out file.
    tiledrain_hydrograph_path : Path or None
        Path to the tile drain hydrograph .out file.
    gw_locations : list[HydrographLocation]
        GW hydrograph locations (with layer info).
    stream_locations : list[HydrographLocation]
        Stream hydrograph locations.
    n_model_layers : int
        Number of model layers (from stratigraphy, if discoverable).
    gw_main_path : Path or None
        Path to the GW main file.
    stream_main_path : Path or None
        Path to the stream main file.
    start_date_str : str
        Simulation start date string from the main file.
    time_unit : str
        Simulation time step unit (e.g. ``"1MON"``).
    """

    gw_hydrograph_path: Path | None = None
    stream_hydrograph_path: Path | None = None
    subsidence_hydrograph_path: Path | None = None
    tiledrain_hydrograph_path: Path | None = None
    gw_locations: list[HydrographLocation] = field(default_factory=list)
    stream_locations: list[HydrographLocation] = field(default_factory=list)
    n_model_layers: int = 1
    gw_main_path: Path | None = None
    stream_main_path: Path | None = None
    start_date_str: str = ""
    time_unit: str = ""


def _read_data_line(f: TextIO, line_counter: list[int] | None = None) -> str:
    """Read next non-comment, non-blank line from *f*.

    Returns the full raw line (no inline-comment stripping).
    Raises ``StopIteration`` on EOF.
    """
    for raw in f:
        if line_counter is not None:
            line_counter[0] += 1
        if _is_comment_line(raw):
            continue
        stripped = raw.strip()
        if stripped:
            return stripped
    raise StopIteration("Unexpected end of file")


def _read_data_value(f: TextIO, line_counter: list[int] | None = None) -> str:
    """Read next data value (non-comment line with inline comment stripped)."""
    raw = _read_data_line(f, line_counter)
    value, _ = _strip_inline_comment(raw)
    return value.strip()


def _skip_lines(f: TextIO, n: int, line_counter: list[int] | None = None) -> str:
    """Skip *n* non-comment lines, returning the last one read."""
    line = ""
    for _ in range(n):
        line = _read_data_line(f, line_counter)
    return line


def discover_hydrograph_files(
    simulation_main_file: Path | str,
) -> HydrographFileInfo:
    """Parse an IWFM simulation main file to discover hydrograph output paths.

    Parameters
    ----------
    simulation_main_file : Path or str
        Path to the IWFM simulation main file (e.g. ``C2VSimFG.in``).

    Returns
    -------
    HydrographFileInfo
        Discovered paths and metadata.

    Raises
    ------
    FileNotFoundError
        If the simulation main file does not exist.
    """
    sim_path = Path(simulation_main_file)
    if not sim_path.exists():
        raise FileNotFoundError(f"Simulation main file not found: {sim_path}")

    sim_dir = sim_path.parent
    info = HydrographFileInfo()

    # ------------------------------------------------------------------
    # 1. Parse simulation main file
    # ------------------------------------------------------------------
    with open(sim_path) as f:
        lc: list[int] = [0]

        # Skip 4 lines to reach component file paths
        _skip_lines(f, 4, lc)

        # Line 5: GW main file
        gw_main_raw = _read_data_value(f, lc)
        gw_main_path = _resolve_path(sim_dir, gw_main_raw, allow_empty=True)
        if gw_main_path and gw_main_path.exists():
            info.gw_main_path = gw_main_path
        elif gw_main_raw:
            logger.warning("GW main file not found: %s", gw_main_raw)

        # Line 6: Stream main file
        stream_main_raw = _read_data_value(f, lc)
        stream_main_path = _resolve_path(sim_dir, stream_main_raw, allow_empty=True)
        if stream_main_path and stream_main_path.exists():
            info.stream_main_path = stream_main_path

        # Skip to start date (skip 8 lines, then read start date)
        _skip_lines(f, 8, lc)
        start_date_raw = _read_data_value(f, lc)
        # Extract date portion before '_'
        if "_" in start_date_raw:
            info.start_date_str = start_date_raw.split("_")[0]
        else:
            info.start_date_str = start_date_raw

        # Skip 1, then read time unit
        _skip_lines(f, 1, lc)
        time_unit_raw = _read_data_value(f, lc)
        info.time_unit = time_unit_raw.strip().upper()

    logger.info("Simulation file: %s", sim_path)
    logger.info("Start date: %s, time unit: %s", info.start_date_str, info.time_unit)

    # ------------------------------------------------------------------
    # 2. Parse GW main file
    # ------------------------------------------------------------------
    if info.gw_main_path:
        _parse_gw_main_file(info)

    # ------------------------------------------------------------------
    # 3. Parse Stream main file
    # ------------------------------------------------------------------
    if info.stream_main_path:
        _parse_stream_main_file(info)

    return info


def _parse_gw_main_file(info: HydrographFileInfo) -> None:
    """Parse GW main file to extract hydrograph .out path and locations."""
    gw_path = info.gw_main_path
    if gw_path is None:
        return

    gw_dir = gw_path.parent

    try:
        with open(gw_path) as f:
            lc: list[int] = [0]

            # Line 1: version/debug (skip)
            _read_data_line(f, lc)

            # Line 2: Tile drain file path
            td_raw = _read_data_value(f, lc)
            td_path = _resolve_path(gw_dir, td_raw, allow_empty=True)
            if td_path and td_path.exists():
                info.tiledrain_hydrograph_path = None  # Will parse TD main later
                # Store for later parsing
                _td_main_path = td_path
            else:
                _td_main_path = None

            # Line 3: Pumping file (skip)
            _read_data_line(f, lc)

            # Line 4: Subsidence file path
            sub_raw = _read_data_value(f, lc)
            sub_path = _resolve_path(gw_dir, sub_raw, allow_empty=True)
            _sub_main_path = sub_path if sub_path and sub_path.exists() else None

            # Skip 17 lines to reach NOUTH (GW main lines 5-21)
            last = _skip_lines(f, 17, lc)
            val, _ = _strip_inline_comment(last)
            n_hyd = int(val.split()[0])

            # FACTXY
            factxy_raw = _read_data_value(f, lc)
            try:
                factxy = float(factxy_raw.split()[0])
            except (ValueError, IndexError):
                factxy = 1.0

            # GWHYDOUTFL (hydrograph output file path)
            out_raw = _read_data_value(f, lc)
            out_path = _resolve_path(gw_dir, out_raw, allow_empty=True)
            if out_path:
                info.gw_hydrograph_path = out_path

            # Read NOUTH hydrograph entries (comment headers auto-skipped)
            locations: list[HydrographLocation] = []
            for _ in range(n_hyd):
                try:
                    line = _read_data_line(f, lc)
                except StopIteration:
                    break
                parts = line.split()
                if len(parts) < 4:
                    continue

                hyd_id = int(parts[0])
                hyd_type = int(parts[1])
                layer = int(parts[2])

                if hyd_type == 0:
                    # X-Y format: ID HYDTYP IOUTHL X Y NAME
                    x = float(parts[3]) * factxy
                    y = float(parts[4]) * factxy
                    name = parts[5] if len(parts) > 5 else f"HYD{hyd_id}"
                else:
                    # Node format: ID HYDTYP IOUTHL IOUTH NAME
                    x = 0.0
                    y = 0.0
                    name = parts[4] if len(parts) > 4 else f"HYD{hyd_id}"

                locations.append(
                    HydrographLocation(
                        node_id=hyd_id,
                        layer=layer,
                        x=x,
                        y=y,
                        name=name,
                    )
                )

            info.gw_locations = locations

    except Exception as e:
        logger.warning("Error parsing GW main file %s: %s", gw_path, e)

    logger.info(
        "GW: %d hydrographs, .out=%s",
        len(info.gw_locations),
        info.gw_hydrograph_path,
    )


def _parse_stream_main_file(info: HydrographFileInfo) -> None:
    """Parse stream main file to extract hydrograph .out path and locations."""
    str_path = info.stream_main_path
    if str_path is None:
        return

    str_dir = str_path.parent

    try:
        with open(str_path) as f:
            lc: list[int] = [0]

            # Skip 7 lines to reach NOUTR
            last = _skip_lines(f, 7, lc)
            val, _ = _strip_inline_comment(last)
            n_hyd = int(val.split()[0])

            # Skip 6 lines, the last being the output file path
            last = _skip_lines(f, 6, lc)
            val, _ = _strip_inline_comment(last)
            out_path = _resolve_path(str_dir, val.strip(), allow_empty=True)
            if out_path:
                info.stream_hydrograph_path = out_path

            # Read NOUTR hydrograph entries (comment headers auto-skipped)
            locations: list[HydrographLocation] = []
            for _ in range(n_hyd):
                try:
                    line = _read_data_line(f, lc)
                except StopIteration:
                    break
                parts = line.split()
                if len(parts) < 2:
                    continue
                hyd_id = int(parts[0])
                name = parts[1] if len(parts) > 1 else f"STR{hyd_id}"
                locations.append(
                    HydrographLocation(node_id=hyd_id, layer=1, x=0.0, y=0.0, name=name)
                )

            info.stream_locations = locations

    except Exception as e:
        logger.warning("Error parsing stream main file %s: %s", str_path, e)

    logger.info(
        "Stream: %d hydrographs, .out=%s",
        len(info.stream_locations),
        info.stream_hydrograph_path,
    )
