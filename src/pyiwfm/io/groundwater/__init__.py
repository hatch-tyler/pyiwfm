"""Groundwater reader, writer, and per-feature sub-files.

The v1.x flat modules ``pyiwfm.io.groundwater`` (the main reader),
``pyiwfm.io.gw_writer`` (the Jinja2 writer),
``pyiwfm.io.gw_main_writer`` (the GW main-file writer), and the eight
per-feature reader/writer pairs (``gw_boundary``,
``gw_boundary_writer``, ``gw_pumping``, ``gw_pumping_writer``,
``gw_subsidence``, ``gw_subsidence_writer``, ``gw_tiledrain``,
``gw_tiledrain_writer``) are now collapsed into one subpackage:

- :mod:`pyiwfm.io.groundwater.reader` — was ``groundwater.py``
  (`GroundwaterReader`, `GroundwaterWriter` (legacy free-form),
  `GWFileConfig`, the `read_*` / `write_groundwater` per-file
  functions, and the dataclasses `KhAnomalyEntry`,
  `ParametricGridData`, `FaceFlowSpec`).
- :mod:`pyiwfm.io.groundwater.main_reader` — `GWMainFileConfig` +
  `GWMainFileReader` (the hierarchical main-file dispatcher). Split
  out of ``reader.py`` once the package layout was in place; now
  ~1075 LOC instead of 1875.
- :mod:`pyiwfm.io.groundwater.writer` — was ``gw_writer.py``
  (`GWComponentWriter`, `GWWriterConfig`, `write_gw_component`).
- :mod:`pyiwfm.io.groundwater.main_writer` — was ``gw_main_writer.py``
  (`write_gw_main_file`).
- :mod:`pyiwfm.io.groundwater.boundary` /
  :mod:`pyiwfm.io.groundwater.boundary_writer` — was
  ``gw_boundary[_writer].py``.
- :mod:`pyiwfm.io.groundwater.pumping` /
  :mod:`pyiwfm.io.groundwater.pumping_writer` — was
  ``gw_pumping[_writer].py``.
- :mod:`pyiwfm.io.groundwater.subsidence` /
  :mod:`pyiwfm.io.groundwater.subsidence_writer` — was
  ``gw_subsidence[_writer].py``.
- :mod:`pyiwfm.io.groundwater.tiledrain` /
  :mod:`pyiwfm.io.groundwater.tiledrain_writer` — was
  ``gw_tiledrain[_writer].py``.

The package re-exports every public symbol. The ten v1.x sibling
paths (``pyiwfm.io.gw_writer``, ``pyiwfm.io.gw_boundary``, …) are gone
in v2.0; use ``from pyiwfm.io.groundwater import X`` instead. See
``docs/MIGRATION_v1_to_v2.md`` § 10.
"""

from __future__ import annotations

from pyiwfm.io.groundwater.boundary import (
    ConstrainedGeneralHeadBC,
    GeneralHeadBC,
    GWBoundaryConfig,
    GWBoundaryReader,
    SpecifiedFlowBC,
    SpecifiedHeadBC,
    read_gw_boundary,
)
from pyiwfm.io.groundwater.boundary_writer import (
    write_bc_main,
    write_constrained_gh_bc,
    write_general_head_bc,
    write_specified_flow_bc,
    write_specified_head_bc,
)
from pyiwfm.io.groundwater.main_reader import (
    GWMainFileConfig,
    GWMainFileReader,
    read_gw_main_file,
)
from pyiwfm.io.groundwater.main_writer import write_gw_main_file
from pyiwfm.io.groundwater.pumping import (
    ElementGroup,
    ElementPumpingSpec,
    PumpingConfig,
    PumpingReader,
    WellPumpingSpec,
    WellSpec,
    read_gw_pumping,
)
from pyiwfm.io.groundwater.pumping_writer import (
    write_elem_pump_file,
    write_pumping_main,
    write_well_spec_file,
)
from pyiwfm.io.groundwater.reader import (
    FaceFlowSpec,
    GroundwaterReader,
    GroundwaterWriter,
    GWFileConfig,
    KhAnomalyEntry,
    ParametricGridData,
    read_initial_heads,
    read_subsidence,
    read_wells,
    write_groundwater,
)
from pyiwfm.io.groundwater.subsidence import (
    ParametricSubsidenceData,
    SubsidenceConfig,
    SubsidenceHydrographSpec,
    SubsidenceNodeParams,
    SubsidenceReader,
    read_gw_subsidence,
)
from pyiwfm.io.groundwater.subsidence_writer import (
    write_gw_subsidence,
    write_subsidence_ic,
    write_subsidence_main,
)
from pyiwfm.io.groundwater.tiledrain import (
    SubIrrigationSpec,
    TileDrainConfig,
    TileDrainHydroSpec,
    TileDrainReader,
    TileDrainSpec,
    read_gw_tiledrain,
)
from pyiwfm.io.groundwater.tiledrain_writer import write_tile_drain_file
from pyiwfm.io.groundwater.writer import (
    GWComponentWriter,
    GWWriterConfig,
    write_gw_component,
)

__all__ = [
    # reader.py
    "FaceFlowSpec",
    "GroundwaterReader",
    "GroundwaterWriter",
    "GWFileConfig",
    "GWMainFileConfig",
    "GWMainFileReader",
    "KhAnomalyEntry",
    "ParametricGridData",
    "read_gw_main_file",
    "read_initial_heads",
    "read_subsidence",
    "read_wells",
    "write_groundwater",
    # writer.py
    "GWComponentWriter",
    "GWWriterConfig",
    "write_gw_component",
    # main_writer.py
    "write_gw_main_file",
    # boundary.py + boundary_writer.py
    "ConstrainedGeneralHeadBC",
    "GeneralHeadBC",
    "GWBoundaryConfig",
    "GWBoundaryReader",
    "SpecifiedFlowBC",
    "SpecifiedHeadBC",
    "read_gw_boundary",
    "write_bc_main",
    "write_constrained_gh_bc",
    "write_general_head_bc",
    "write_specified_flow_bc",
    "write_specified_head_bc",
    # pumping.py + pumping_writer.py
    "ElementGroup",
    "ElementPumpingSpec",
    "PumpingConfig",
    "PumpingReader",
    "WellPumpingSpec",
    "WellSpec",
    "read_gw_pumping",
    "write_elem_pump_file",
    "write_pumping_main",
    "write_well_spec_file",
    # subsidence.py + subsidence_writer.py
    "ParametricSubsidenceData",
    "SubsidenceConfig",
    "SubsidenceHydrographSpec",
    "SubsidenceNodeParams",
    "SubsidenceReader",
    "read_gw_subsidence",
    "write_gw_subsidence",
    "write_subsidence_ic",
    "write_subsidence_main",
    # tiledrain.py + tiledrain_writer.py
    "SubIrrigationSpec",
    "TileDrainConfig",
    "TileDrainHydroSpec",
    "TileDrainReader",
    "TileDrainSpec",
    "read_gw_tiledrain",
    "write_tile_drain_file",
]
