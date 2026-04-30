"""Preprocessor reader, writer, and mesh subfile helpers.

The v1.x flat modules ``pyiwfm.io.preprocessor`` (the main reader),
``pyiwfm.io.preprocessor_writer`` (the Jinja2 writer), and
``pyiwfm.io.mesh`` (the nodes/elements/stratigraphy preprocessor
subfiles) are now collapsed into one subpackage:

- :mod:`pyiwfm.io.preprocessor.reader` — readers for the IWFM
  preprocessor main file and subregions.
- :mod:`pyiwfm.io.preprocessor.writer` — Jinja2 writer
  (``PreProcessorWriter``).
- :mod:`pyiwfm.io.preprocessor.mesh` — canonical readers and writers
  for the nodes, elements, and stratigraphy preprocessor subfiles.

The preprocessor binary format lives in
:mod:`pyiwfm.io.binary.preprocessor` (see § 10 of
``docs/MIGRATION_v1_to_v2.md``).

The package re-exports the public symbols from all three submodules.
The v1.x paths ``pyiwfm.io.preprocessor_writer`` and
``pyiwfm.io.mesh`` are gone in v2.0; use
``from pyiwfm.io.preprocessor import X`` instead. See
``docs/MIGRATION_v1_to_v2.md`` § 10.
"""

from __future__ import annotations

from pyiwfm.io.preprocessor.mesh import (
    read_elements,
    read_nodes,
    read_stratigraphy,
    write_elements,
    write_nodes,
    write_stratigraphy,
)
from pyiwfm.io.preprocessor.reader import (
    PreProcessorConfig,
    _make_relative_path,
    _resolve_path,
    _write_subregions_file,
    read_preprocessor_main,
    read_subregions_file,
    save_complete_model,
    save_model_to_preprocessor,
    write_preprocessor_main,
)
from pyiwfm.io.preprocessor.writer import (
    PreProcessorWriter,
    write_preprocessor_files,
)

__all__ = [
    # reader.py
    "PreProcessorConfig",
    "read_preprocessor_main",
    "read_subregions_file",
    "save_complete_model",
    "save_model_to_preprocessor",
    "write_preprocessor_main",
    # writer.py
    "PreProcessorWriter",
    "write_preprocessor_files",
    # mesh.py
    "read_elements",
    "read_nodes",
    "read_stratigraphy",
    "write_elements",
    "write_nodes",
    "write_stratigraphy",
]

# Private helpers re-exported for cross-module callers (still importable
# via ``from pyiwfm.io.preprocessor import _resolve_path``) but kept out of
# ``__all__`` because they remain implementation detail.
_ = (_make_relative_path, _resolve_path, _write_subregions_file)
