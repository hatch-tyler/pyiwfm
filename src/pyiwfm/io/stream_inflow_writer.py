"""
IWFM-format Stream Inflow Writer.

Writes the stream inflow file header from an InflowConfig, including
the conversion factor, time unit, number of inflow series, and the
per-series stream node mapping.

The writer produces output that is compatible with the InflowReader,
enabling roundtrip read-write fidelity. Note that this writes only
the header section; actual time-series data is handled separately
by the time-series writer modules.

Reference: Class_StrmInflow.f90 - New()
"""

from __future__ import annotations

from pathlib import Path

from pyiwfm.io.iwfm_writer import (
    ensure_parent_dir as _ensure_parent_dir,
)
from pyiwfm.io.iwfm_writer import (
    write_comment as _write_comment,
)
from pyiwfm.io.iwfm_writer import (
    write_value as _write_value,
)
from pyiwfm.io.stream_inflow import InflowConfig


def write_stream_inflow(config: InflowConfig, filepath: Path | str) -> Path:
    """Write the stream inflow file header.

    Writes the conversion factor, time unit, number of inflow series,
    and per-series inflow-ID to stream-node mapping.

    Args:
        config: Inflow configuration with header data
        filepath: Output file path

    Returns:
        Path to written file
    """
    filepath = Path(filepath)
    _ensure_parent_dir(filepath)

    with open(filepath, "w") as f:
        _write_comment(f, "IWFM Stream Inflow File")

        # Conversion factor
        _write_value(f, config.conversion_factor, "Conversion factor")

        # Time unit
        _write_value(f, config.time_unit, "Time unit")

        # Number of inflow series
        _write_value(f, config.n_inflows, "NInflow")

        if config.n_inflows <= 0:
            return filepath

        # Inflow specifications: InflowID StreamNodeID
        for spec in config.inflow_specs:
            f.write(f"     {spec.inflow_id:>6d}  {spec.stream_node:>6d}\n")

    return filepath
