"""Root-zone reader, writer, and per-land-use sub-files.

The v1.x flat modules ``pyiwfm.io.rootzone`` (the main reader),
``pyiwfm.io.rootzone_writer`` (the Jinja2 writer),
``pyiwfm.io._rootzone_base``, and the six per-land-use sub-files
(``rootzone_area``, ``rootzone_native``, ``rootzone_nonponded``,
``rootzone_ponded``, ``rootzone_urban``, plus the v4.x variant
``rootzone_v4x``) are now collapsed into one subpackage:

- :mod:`pyiwfm.io.rootzone.reader` — reader for the root-zone main file.
- :mod:`pyiwfm.io.rootzone.writer` — Jinja2 component writer.
- :mod:`pyiwfm.io.rootzone._base` — shared ``_RootzoneReaderBase``.
- :mod:`pyiwfm.io.rootzone.area` — element-area metadata reader.
- :mod:`pyiwfm.io.rootzone.native` — native/riparian sub-file (v5+).
- :mod:`pyiwfm.io.rootzone.nonponded` — non-ponded crop sub-file (v5+).
- :mod:`pyiwfm.io.rootzone.ponded` — ponded crop sub-file (v5+).
- :mod:`pyiwfm.io.rootzone.urban` — urban land-use sub-file (v5+).
- :mod:`pyiwfm.io.rootzone.v4x` — v4.x readers and writers for all four
  land-use sub-files.

The package re-exports the public symbols from each submodule. The
v1.x paths ``pyiwfm.io.rootzone_writer``, ``pyiwfm.io.rootzone_area``,
``pyiwfm.io.rootzone_native``, etc. are gone in v2.0; use
``from pyiwfm.io.rootzone import X`` instead. See
``docs/MIGRATION_v1_to_v2.md`` § 10.
"""

from __future__ import annotations

from pyiwfm.io.rootzone.area import (
    AreaFileMetadata,
    read_all_timesteps,
    read_area_metadata,
    read_area_timestep,
)
from pyiwfm.io.rootzone.native import (
    NativeRiparianCNRow,
    NativeRiparianConfig,
    NativeRiparianEtcRow,
    NativeRiparianInitialRow,
    NativeRiparianReader,
    read_native_riparian,
)
from pyiwfm.io.rootzone.nonponded import (
    CurveNumberRow,
    EtcPointerRow,
    InitialConditionRow,
    IrrigationPointerRow,
    NonPondedCropConfig,
    NonPondedCropReader,
    SoilMoisturePointerRow,
    SupplyReturnReuseRow,
    read_nonponded_crop,
)
from pyiwfm.io.rootzone.ponded import (
    PondedCropConfig,
    PondedCropReader,
    read_ponded_crop,
)
from pyiwfm.io.rootzone.reader import (
    ElementSoilParamRow,
    RootZoneFileConfig,
    RootZoneMainFileConfig,
    RootZoneMainFileReader,
    RootZoneReader,
    RootZoneWriter,
    parse_version,
    read_crop_types,
    read_rootzone_main_file,
    read_soil_params,
    version_ge,
    write_rootzone,
)
from pyiwfm.io.rootzone.urban import (
    SurfaceFlowDestRow,
    UrbanCurveNumberRow,
    UrbanInitialConditionRow,
    UrbanLandUseConfig,
    UrbanLandUseReader,
    UrbanManagementRow,
    read_urban_landuse,
)
from pyiwfm.io.rootzone.v4x import (
    AgInitialConditionRow,
    ElementCropRow,
    NativeRiparianConfigV4x,
    NativeRiparianElementRowV4x,
    NativeRiparianInitialRowV4x,
    NativeRiparianReaderV4x,
    NativeRiparianWriterV4x,
    NonPondedCropConfigV4x,
    NonPondedCropReaderV4x,
    NonPondedCropWriterV4x,
    PondedCropConfigV4x,
    PondedCropReaderV4x,
    PondedCropWriterV4x,
    RootDepthRow,
    UrbanConfigV4x,
    UrbanElementRowV4x,
    UrbanInitialRowV4x,
    UrbanReaderV4x,
    UrbanWriterV4x,
    read_native_riparian_v4x,
    read_nonponded_v4x,
    read_ponded_v4x,
    read_urban_v4x,
)
from pyiwfm.io.rootzone.writer import (
    RootZoneComponentWriter,
    RootZoneWriterConfig,
    write_rootzone_component,
)

__all__ = [
    # reader.py
    "ElementSoilParamRow",
    "RootZoneFileConfig",
    "RootZoneMainFileConfig",
    "RootZoneMainFileReader",
    "RootZoneReader",
    "RootZoneWriter",
    "parse_version",
    "read_crop_types",
    "read_rootzone_main_file",
    "read_soil_params",
    "version_ge",
    "write_rootzone",
    # writer.py
    "RootZoneComponentWriter",
    "RootZoneWriterConfig",
    "write_rootzone_component",
    # area.py
    "AreaFileMetadata",
    "read_all_timesteps",
    "read_area_metadata",
    "read_area_timestep",
    # native.py
    "NativeRiparianCNRow",
    "NativeRiparianConfig",
    "NativeRiparianEtcRow",
    "NativeRiparianInitialRow",
    "NativeRiparianReader",
    "read_native_riparian",
    # nonponded.py
    "CurveNumberRow",
    "EtcPointerRow",
    "InitialConditionRow",
    "IrrigationPointerRow",
    "NonPondedCropConfig",
    "NonPondedCropReader",
    "SoilMoisturePointerRow",
    "SupplyReturnReuseRow",
    "read_nonponded_crop",
    # ponded.py
    "PondedCropConfig",
    "PondedCropReader",
    "read_ponded_crop",
    # urban.py
    "SurfaceFlowDestRow",
    "UrbanCurveNumberRow",
    "UrbanInitialConditionRow",
    "UrbanLandUseConfig",
    "UrbanLandUseReader",
    "UrbanManagementRow",
    "read_urban_landuse",
    # v4x.py
    "AgInitialConditionRow",
    "ElementCropRow",
    "NativeRiparianConfigV4x",
    "NativeRiparianElementRowV4x",
    "NativeRiparianInitialRowV4x",
    "NativeRiparianReaderV4x",
    "NativeRiparianWriterV4x",
    "NonPondedCropConfigV4x",
    "NonPondedCropReaderV4x",
    "NonPondedCropWriterV4x",
    "PondedCropConfigV4x",
    "PondedCropReaderV4x",
    "PondedCropWriterV4x",
    "RootDepthRow",
    "UrbanConfigV4x",
    "UrbanElementRowV4x",
    "UrbanInitialRowV4x",
    "UrbanReaderV4x",
    "UrbanWriterV4x",
    "read_native_riparian_v4x",
    "read_nonponded_v4x",
    "read_ponded_v4x",
    "read_urban_v4x",
]
