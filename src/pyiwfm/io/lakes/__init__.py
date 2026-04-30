"""Lakes reader and writer.

The v1.x flat modules ``pyiwfm.io.lakes`` (the reader) and
``pyiwfm.io.lake_writer`` (the writer) are now collapsed into one
subpackage:

- :mod:`pyiwfm.io.lakes.reader` — readers for the lake main file,
  lake definitions, lake elements, and outflow rating tables.
- :mod:`pyiwfm.io.lakes.writer` — Jinja2 component writer.

The package re-exports the public symbols from both submodules. The
v1.x path ``from pyiwfm.io.lake_writer import X`` is gone in v2.0; use
``from pyiwfm.io.lakes import X`` instead. The plural/singular
mismatch in the original module names dissolves in this layout. See
``docs/MIGRATION_v1_to_v2.md`` § 10.
"""

from __future__ import annotations

from pyiwfm.io.lakes.reader import (
    LakeFileConfig,
    LakeMainFileConfig,
    LakeMainFileReader,
    LakeOutflowRating,
    LakeParamSpec,
    LakeReader,
    LakeWriter,
    OutflowRatingPoint,
    read_lake_definitions,
    read_lake_elements,
    read_lake_main_file,
    write_lakes,
)
from pyiwfm.io.lakes.writer import (
    LakeComponentWriter,
    LakeWriterConfig,
    write_lake_component,
)

__all__ = [
    # reader.py
    "LakeFileConfig",
    "LakeMainFileConfig",
    "LakeMainFileReader",
    "LakeOutflowRating",
    "LakeParamSpec",
    "LakeReader",
    "LakeWriter",
    "OutflowRatingPoint",
    "read_lake_definitions",
    "read_lake_elements",
    "read_lake_main_file",
    "write_lakes",
    # writer.py
    "LakeComponentWriter",
    "LakeWriterConfig",
    "write_lake_component",
]
