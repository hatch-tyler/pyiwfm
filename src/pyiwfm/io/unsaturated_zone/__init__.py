"""Unsaturated-zone reader and writer.

The v1.x flat modules ``pyiwfm.io.unsaturated_zone`` (the reader) and
``pyiwfm.io.unsaturated_zone_writer`` (the writer) are now collapsed
into one subpackage:

- :mod:`pyiwfm.io.unsaturated_zone.reader` — reader for the
  unsaturated-zone main file.
- :mod:`pyiwfm.io.unsaturated_zone.writer` — Jinja2 component writer.

The package re-exports the public symbols from both submodules. The
v1.x path ``from pyiwfm.io.unsaturated_zone_writer import X`` is gone
in v2.0; use ``from pyiwfm.io.unsaturated_zone import X`` instead. See
``docs/MIGRATION_v1_to_v2.md`` § 10.
"""

from __future__ import annotations

from pyiwfm.io.unsaturated_zone.reader import (
    UnsatZoneElementData,
    UnsatZoneMainConfig,
    UnsatZoneMainReader,
    read_unsaturated_zone_main,
)
from pyiwfm.io.unsaturated_zone.writer import (
    UnsatZoneComponentWriter,
    UnsatZoneWriterConfig,
    write_unsaturated_zone_component,
)

__all__ = [
    # reader.py
    "UnsatZoneElementData",
    "UnsatZoneMainConfig",
    "UnsatZoneMainReader",
    "read_unsaturated_zone_main",
    # writer.py
    "UnsatZoneComponentWriter",
    "UnsatZoneWriterConfig",
    "write_unsaturated_zone_component",
]
