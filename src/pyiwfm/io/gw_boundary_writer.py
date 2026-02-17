"""
IWFM-format Boundary Conditions Writer.

Writes the BC dispatcher file and sub-files from a GWBoundaryConfig.
"""

from __future__ import annotations

from pathlib import Path

from pyiwfm.io.gw_boundary import GWBoundaryConfig
from pyiwfm.io.iwfm_writer import (
    ensure_parent_dir as _ensure_parent_dir,
)
from pyiwfm.io.iwfm_writer import (
    write_comment as _write_comment,
)
from pyiwfm.io.iwfm_writer import (
    write_value as _write_value,
)


def write_bc_main(config: GWBoundaryConfig, filepath: Path | str) -> Path:
    """Write the BC main dispatcher file.

    Args:
        config: BC configuration
        filepath: Output file path

    Returns:
        Path to written file
    """
    filepath = Path(filepath)
    _ensure_parent_dir(filepath)

    with open(filepath, "w") as f:
        _write_comment(f, "IWFM Boundary Conditions Main File")

        _write_value(f, str(config.sp_flow_file or ""), "Specified flow BC file")
        _write_value(f, str(config.sp_head_file or ""), "Specified head BC file")
        _write_value(f, str(config.gh_file or ""), "General head BC file")
        _write_value(f, str(config.cgh_file or ""), "Constrained GH BC file")
        _write_value(f, str(config.ts_data_file or ""), "Time series data file")

        # NOUTB section
        _write_value(f, config.n_bc_output_nodes, "NOUTB")
        if config.n_bc_output_nodes > 0:
            _write_value(f, str(config.bc_output_file or ""), "BHYDOUTFL")
            for node_id in config.bc_output_specs:
                _write_value(f, node_id)

    return filepath


def write_specified_flow_bc(config: GWBoundaryConfig, filepath: Path | str) -> Path:
    """Write specified flow BC sub-file."""
    filepath = Path(filepath)
    _ensure_parent_dir(filepath)

    with open(filepath, "w") as f:
        _write_comment(f, "IWFM Specified Flow Boundary Conditions")
        _write_value(f, len(config.specified_flow_bcs), "NQB")
        if config.specified_flow_bcs:
            _write_value(f, config.sp_flow_factor, "FACT")
            _write_value(f, config.sp_flow_time_unit, "TUNIT")
            for bc in config.specified_flow_bcs:
                flow = (
                    bc.base_flow / config.sp_flow_factor if config.sp_flow_factor else bc.base_flow
                )
                f.write(
                    f"     {bc.node_id:>6d}  {bc.layer:>3d}  {bc.ts_column:>4d}  {flow:>12.4f}\n"
                )

    return filepath


def write_specified_head_bc(config: GWBoundaryConfig, filepath: Path | str) -> Path:
    """Write specified head BC sub-file."""
    filepath = Path(filepath)
    _ensure_parent_dir(filepath)

    with open(filepath, "w") as f:
        _write_comment(f, "IWFM Specified Head Boundary Conditions")
        _write_value(f, len(config.specified_head_bcs), "NHB")
        if config.specified_head_bcs:
            _write_value(f, config.sp_head_factor, "FACT")
            for bc in config.specified_head_bcs:
                head = (
                    bc.head_value / config.sp_head_factor
                    if config.sp_head_factor
                    else bc.head_value
                )
                f.write(
                    f"     {bc.node_id:>6d}  {bc.layer:>3d}  {bc.ts_column:>4d}  {head:>12.4f}\n"
                )

    return filepath


def write_general_head_bc(config: GWBoundaryConfig, filepath: Path | str) -> Path:
    """Write general head BC sub-file."""
    filepath = Path(filepath)
    _ensure_parent_dir(filepath)

    with open(filepath, "w") as f:
        _write_comment(f, "IWFM General Head Boundary Conditions")
        _write_value(f, len(config.general_head_bcs), "NGB")
        if config.general_head_bcs:
            _write_value(f, config.gh_head_factor, "FACTH")
            _write_value(f, config.gh_conductance_factor, "FACTC")
            _write_value(f, config.gh_time_unit, "TUNIT")
            for bc in config.general_head_bcs:
                head = (
                    bc.external_head / config.gh_head_factor
                    if config.gh_head_factor
                    else bc.external_head
                )
                cond = (
                    bc.conductance / config.gh_conductance_factor
                    if config.gh_conductance_factor
                    else bc.conductance
                )
                f.write(
                    f"     {bc.node_id:>6d}  {bc.layer:>3d}  "
                    f"{bc.ts_column:>4d}  {head:>12.4f}  {cond:>12.6f}\n"
                )

    return filepath


def write_constrained_gh_bc(config: GWBoundaryConfig, filepath: Path | str) -> Path:
    """Write constrained general head BC sub-file."""
    filepath = Path(filepath)
    _ensure_parent_dir(filepath)

    with open(filepath, "w") as f:
        _write_comment(f, "IWFM Constrained General Head Boundary Conditions")
        _write_value(f, len(config.constrained_gh_bcs), "NCGB")
        if config.constrained_gh_bcs:
            _write_value(f, config.cgh_head_factor, "FACTH")
            _write_value(f, config.cgh_max_flow_factor, "FACTVL")
            _write_value(f, config.cgh_head_time_unit, "TUNIT")
            _write_value(f, config.cgh_conductance_factor, "FACTC")
            _write_value(f, config.cgh_conductance_time_unit, "TUNITC")
            for bc in config.constrained_gh_bcs:
                head = (
                    bc.external_head / config.cgh_head_factor
                    if config.cgh_head_factor
                    else bc.external_head
                )
                cond = (
                    bc.conductance / config.cgh_conductance_factor
                    if config.cgh_conductance_factor
                    else bc.conductance
                )
                ch = (
                    bc.constraining_head / config.cgh_head_factor
                    if config.cgh_head_factor
                    else bc.constraining_head
                )
                mf = (
                    bc.max_flow / config.cgh_max_flow_factor
                    if config.cgh_max_flow_factor
                    else bc.max_flow
                )
                f.write(
                    f"     {bc.node_id:>6d}  {bc.layer:>3d}  "
                    f"{bc.ts_column:>4d}  {head:>12.4f}  {cond:>12.6f}  "
                    f"{ch:>12.4f}  {bc.max_flow_ts_column:>4d}  {mf:>12.4f}\n"
                )

    return filepath
