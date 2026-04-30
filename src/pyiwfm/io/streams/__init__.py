"""Stream reader, writer, and per-feature sub-files.

The v1.x flat modules ``pyiwfm.io.streams`` (the reader),
``pyiwfm.io.stream_writer`` (the Jinja2 writer), and the seven
per-feature modules (``stream_bypass``, ``stream_bypass_writer``,
``stream_diversion``, ``stream_diversion_writer``, ``stream_inflow``,
``stream_inflow_writer``, ``stream_depletion``) are now collapsed
into one subpackage:

- :mod:`pyiwfm.io.streams.reader` — was ``streams.py`` (`StreamReader`,
  the legacy `StreamWriter`, plus `StreamFileConfig`,
  `read_stream_nodes`, `read_diversions`, `write_stream`).

- :mod:`pyiwfm.io.streams.main_reader` — `StreamMainFileReader` +
  `StreamMainFileConfig` (the hierarchical main-file dispatcher).
  Split out of ``streams/reader.py`` once the package layout was in
  place; now ~600 LOC instead of 1500.

- :mod:`pyiwfm.io.streams.spec` — `StreamSpecReader` +
  `StreamReachSpec` (the preprocessor StreamsSpec geometry file).

- :mod:`pyiwfm.io.streams.writer` — was ``stream_writer.py``
  (`StreamComponentWriter`, `StreamWriterConfig`,
  `write_stream_component`).

- :mod:`pyiwfm.io.streams.bypass`, :mod:`pyiwfm.io.streams.bypass_writer`
  — was ``stream_bypass[_writer].py``.

- :mod:`pyiwfm.io.streams.diversion`,
  :mod:`pyiwfm.io.streams.diversion_writer` — was
  ``stream_diversion[_writer].py``.

- :mod:`pyiwfm.io.streams.inflow`, :mod:`pyiwfm.io.streams.inflow_writer`
  — was ``stream_inflow[_writer].py``.

- :mod:`pyiwfm.io.streams.depletion` — was ``stream_depletion.py``.

The package re-exports every public symbol. The eight v1.x sibling
paths are gone in v2.0; use ``from pyiwfm.io.streams import X``
instead. See ``docs/MIGRATION_v1_to_v2.md`` § 10.
"""

from __future__ import annotations

from pyiwfm.io.ascii.reader import (
    parse_version as parse_stream_version,
)
from pyiwfm.io.ascii.reader import (
    version_ge as stream_version_ge,
)
from pyiwfm.io.streams.bypass import (
    BypassRatingTable,
    BypassSeepageZone,
    BypassSpec,
    BypassSpecConfig,
    BypassSpecReader,
    read_bypass_spec,
)
from pyiwfm.io.streams.bypass_writer import write_bypass_spec
from pyiwfm.io.streams.depletion import (
    BudgetOutputMissingError,
    StreamDepletionReport,
    StreamDepletionResult,
    StreamNodeDepletionReport,
    StreamNodeDepletionResult,
    compute_stream_depletion,
    compute_stream_depletion_from_models,
    compute_stream_node_depletion,
    write_stream_depletion_csv,
    write_stream_depletion_excel,
    write_stream_depletion_json,
)
from pyiwfm.io.streams.diversion import (
    DiversionSpec,
    DiversionSpecConfig,
    DiversionSpecReader,
    ElementGroup,
    RechargeZoneDest,
    read_diversion_spec,
)
from pyiwfm.io.streams.diversion_writer import write_diversion_spec
from pyiwfm.io.streams.inflow import (
    InflowConfig,
    InflowReader,
    InflowSpec,
    read_stream_inflow,
)
from pyiwfm.io.streams.inflow_writer import write_stream_inflow
from pyiwfm.io.streams.main_reader import (
    StreamMainFileConfig,
    StreamMainFileReader,
    read_stream_main_file,
)
from pyiwfm.io.streams.reader import (
    CrossSectionRow,
    StreamBedParamRow,
    StreamFileConfig,
    StreamInitialConditionRow,
    StreamReader,
    StreamWriter,
    read_diversions,
    read_stream_nodes,
    write_stream,
)
from pyiwfm.io.streams.spec import (
    StreamReachSpec,
    StreamSpecReader,
    read_stream_spec,
)
from pyiwfm.io.streams.writer import (
    StreamComponentWriter,
    StreamWriterConfig,
    write_stream_component,
)

__all__ = [
    # reader.py
    "CrossSectionRow",
    "StreamBedParamRow",
    "StreamFileConfig",
    "StreamInitialConditionRow",
    "StreamMainFileConfig",
    "StreamMainFileReader",
    "StreamReachSpec",
    "StreamReader",
    "StreamSpecReader",
    "StreamWriter",
    "parse_stream_version",
    "read_diversions",
    "read_stream_main_file",
    "read_stream_nodes",
    "read_stream_spec",
    "stream_version_ge",
    "write_stream",
    # writer.py
    "StreamComponentWriter",
    "StreamWriterConfig",
    "write_stream_component",
    # bypass.py + bypass_writer.py
    "BypassRatingTable",
    "BypassSeepageZone",
    "BypassSpec",
    "BypassSpecConfig",
    "BypassSpecReader",
    "read_bypass_spec",
    "write_bypass_spec",
    # diversion.py + diversion_writer.py
    "DiversionSpec",
    "DiversionSpecConfig",
    "DiversionSpecReader",
    "ElementGroup",
    "RechargeZoneDest",
    "read_diversion_spec",
    "write_diversion_spec",
    # inflow.py + inflow_writer.py
    "InflowConfig",
    "InflowReader",
    "InflowSpec",
    "read_stream_inflow",
    "write_stream_inflow",
    # depletion.py
    "BudgetOutputMissingError",
    "StreamDepletionReport",
    "StreamDepletionResult",
    "StreamNodeDepletionReport",
    "StreamNodeDepletionResult",
    "compute_stream_depletion",
    "compute_stream_depletion_from_models",
    "compute_stream_node_depletion",
    "write_stream_depletion_csv",
    "write_stream_depletion_excel",
    "write_stream_depletion_json",
]
