"""Binary file formats used by IWFM.

Two distinct binary formats live under this package:

- :mod:`pyiwfm.io.binary.fortran` — the Fortran unformatted-sequential
  format used for mesh and stratigraphy round-trip
  (``FortranBinaryReader``, ``FortranBinaryWriter``,
  ``StreamAccessBinaryReader``, plus convenience writers).

- :mod:`pyiwfm.io.binary.preprocessor` — the IWFM preprocessor's
  ``ACCESS='STREAM'`` binary output that the simulation phase consumes
  (``PreprocessorBinaryReader``, ``read_preprocessor_binary``, and the
  ``*Data`` record dataclasses).

The package re-exports the public symbols from both submodules. The
v1.x path ``from pyiwfm.io.preprocessor_binary import X`` is gone in
v2.0; use ``from pyiwfm.io.binary import X`` (or the deeper
``pyiwfm.io.binary.preprocessor.X``) instead. See
``docs/MIGRATION_v1_to_v2.md`` § 10.
"""

from __future__ import annotations

from pyiwfm.io.binary.fortran import (
    FortranBinaryReader,
    FortranBinaryWriter,
    StreamAccessBinaryReader,
    read_fortran_record,
    write_binary_mesh,
    write_binary_stratigraphy,
)
from pyiwfm.io.binary.preprocessor import (
    AppElementData,
    AppFaceData,
    AppNodeData,
    LakeData,
    LakeGWConnectorData,
    PreprocessorBinaryData,
    PreprocessorBinaryReader,
    StratigraphyData,
    StreamData,
    StreamGWConnectorData,
    StreamLakeConnectorData,
    SubregionData,
    read_preprocessor_binary,
)

__all__ = [
    # Fortran-binary primitives
    "FortranBinaryReader",
    "FortranBinaryWriter",
    "StreamAccessBinaryReader",
    "read_fortran_record",
    "write_binary_mesh",
    "write_binary_stratigraphy",
    # Preprocessor binary records
    "AppElementData",
    "AppFaceData",
    "AppNodeData",
    "LakeData",
    "LakeGWConnectorData",
    "PreprocessorBinaryData",
    "PreprocessorBinaryReader",
    "StratigraphyData",
    "StreamData",
    "StreamGWConnectorData",
    "StreamLakeConnectorData",
    "SubregionData",
    "read_preprocessor_binary",
]
