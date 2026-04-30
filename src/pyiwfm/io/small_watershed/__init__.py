"""Small-watershed reader and writer.

The v1.x flat modules ``pyiwfm.io.small_watershed`` (the reader) and
``pyiwfm.io.small_watershed_writer`` (the writer) are now collapsed
into one subpackage:

- :mod:`pyiwfm.io.small_watershed.reader` — reader for the
  small-watershed main file.
- :mod:`pyiwfm.io.small_watershed.writer` — Jinja2 component writer.

The package re-exports the public symbols from both submodules. The
v1.x path ``from pyiwfm.io.small_watershed_writer import X`` is gone
in v2.0; use ``from pyiwfm.io.small_watershed import X`` instead. See
``docs/MIGRATION_v1_to_v2.md`` § 10.
"""

from __future__ import annotations

from pyiwfm.io.small_watershed.reader import (
    SmallWatershedMainConfig,
    SmallWatershedMainReader,
    WatershedAquiferParams,
    WatershedGWNode,
    WatershedInitialCondition,
    WatershedRootZoneParams,
    WatershedSpec,
    read_small_watershed_main,
)
from pyiwfm.io.small_watershed.writer import (
    SmallWatershedComponentWriter,
    SmallWatershedWriterConfig,
    write_small_watershed_component,
)

__all__ = [
    # reader.py
    "SmallWatershedMainConfig",
    "SmallWatershedMainReader",
    "WatershedAquiferParams",
    "WatershedGWNode",
    "WatershedInitialCondition",
    "WatershedRootZoneParams",
    "WatershedSpec",
    "read_small_watershed_main",
    # writer.py
    "SmallWatershedComponentWriter",
    "SmallWatershedWriterConfig",
    "write_small_watershed_component",
]
