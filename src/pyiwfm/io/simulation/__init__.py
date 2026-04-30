"""Simulation reader, writer, and messages parser.

The v1.x flat modules ``pyiwfm.io.simulation`` (the reader),
``pyiwfm.io.simulation_writer`` (the writer), and
``pyiwfm.io.simulation_messages`` (the message-log parser) are now
collapsed into one subpackage:

- :mod:`pyiwfm.io.simulation.reader` — readers for the IWFM simulation
  main control file (``IWFMSimulationReader``, ``SimulationReader``,
  ``SimulationConfig``).
- :mod:`pyiwfm.io.simulation.writer` — Jinja2 writer for the
  simulation main file (``SimulationMainWriter``,
  ``SimulationMainConfig``).
- :mod:`pyiwfm.io.simulation.messages` — parser for the IWFM
  ``Message.out`` log (convergence, mass balance, timestep-cut
  records).

The package re-exports the public symbols from all three submodules.
The v1.x paths ``pyiwfm.io.simulation_writer`` and
``pyiwfm.io.simulation_messages`` are gone in v2.0; use
``from pyiwfm.io.simulation import X`` instead. See
``docs/MIGRATION_v1_to_v2.md`` § 10.
"""

from __future__ import annotations

from pyiwfm.io.simulation.messages import (
    ConvergenceHotspot,
    ConvergenceRecord,
    MassBalanceRecord,
    MessageSeverity,
    SimulationMessage,
    SimulationMessagesReader,
    SimulationMessagesResult,
    TimestepCutRecord,
)
from pyiwfm.io.simulation.reader import (
    IWFMSimulationReader,
    SimulationConfig,
    SimulationFileConfig,
    SimulationReader,
    SimulationWriter,
    read_iwfm_simulation,
    read_simulation,
    write_simulation,
)
from pyiwfm.io.simulation.writer import (
    SimulationMainConfig,
    SimulationMainWriter,
    write_simulation_main,
)

__all__ = [
    # reader.py
    "IWFMSimulationReader",
    "SimulationConfig",
    "SimulationFileConfig",
    "SimulationReader",
    "SimulationWriter",
    "read_iwfm_simulation",
    "read_simulation",
    "write_simulation",
    # writer.py
    "SimulationMainConfig",
    "SimulationMainWriter",
    "write_simulation_main",
    # messages.py
    "ConvergenceHotspot",
    "ConvergenceRecord",
    "MassBalanceRecord",
    "MessageSeverity",
    "SimulationMessage",
    "SimulationMessagesReader",
    "SimulationMessagesResult",
    "TimestepCutRecord",
]
