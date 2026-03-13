Simulation Diagnostics
======================

This tutorial walks through parsing IWFM simulation logs to identify
convergence issues, track mass balance errors, and locate problematic
model regions.

.. contents:: Table of Contents
   :local:
   :depth: 2

Learning Objectives
-------------------

- Parse ``SimulationMessages.out`` using ``SimulationMessagesReader``
- Filter messages by severity
- Plot convergence iteration counts over time
- Identify spatial hotspots from message entity references

Prerequisites
-------------

.. code-block:: bash

    pip install pyiwfm[all]

You need a ``SimulationMessages.out`` file from a completed IWFM run.

Step 1: Parse Simulation Messages
-----------------------------------

.. code-block:: python

    from pyiwfm.io.simulation_messages import SimulationMessagesReader

    reader = SimulationMessagesReader("SimulationMessages.out")
    result = reader.read()

    print(f"Messages: {len(result.messages)}")
    print(f"Warnings: {result.warning_count}")
    print(f"Errors: {result.error_count}")
    print(f"Runtime: {result.total_runtime}")
    print(f"Convergence records: {len(result.convergence_records)}")
    print(f"Mass balance records: {len(result.mass_balance_records)}")
    print(f"Timestep cuts: {len(result.timestep_cuts)}")

Step 2: Filter by Severity
----------------------------

.. code-block:: python

    from pyiwfm.io.simulation_messages import MessageSeverity

    # Get all fatal errors
    fatals = result.filter_by_severity(MessageSeverity.FATAL)
    print(f"\n{len(fatals)} fatal errors:")
    for msg in fatals[:10]:
        print(f"  Line {msg.line_number}: [{msg.procedure}] {msg.text[:80]}")

    # Get warnings
    warnings = result.filter_by_severity(MessageSeverity.WARN)
    print(f"\n{len(warnings)} warnings")

Step 3: Plot Convergence
--------------------------

.. code-block:: python

    import matplotlib.pyplot as plt

    iters = [rec.iteration_count for rec in result.convergence_records]
    converged = [rec.convergence_achieved for rec in result.convergence_records]

    fig, ax = plt.subplots(figsize=(14, 4))
    colors = ["green" if c else "red" for c in converged]
    ax.bar(range(len(iters)), iters, color=colors, width=1.0, alpha=0.7)
    ax.set_xlabel("Timestep Index")
    ax.set_ylabel("Iterations")
    ax.set_title(f"Convergence Iterations (avg={result.avg_iterations:.1f}, max={result.max_iterations})")

    # Add legend
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="green", alpha=0.7, label="Converged"),
        Patch(color="red", alpha=0.7, label="Not converged"),
    ])
    plt.tight_layout()
    plt.show()

Step 4: Mass Balance Error Analysis
-------------------------------------

.. code-block:: python

    # Group by component
    from collections import defaultdict

    by_component = defaultdict(list)
    for rec in result.mass_balance_records:
        by_component[rec.component].append(rec)

    fig, ax = plt.subplots(figsize=(14, 5))
    for component, records in by_component.items():
        errors = [r.error_value for r in records]
        ax.plot(range(len(errors)), errors, label=component, linewidth=0.8)

    ax.set_xlabel("Record Index")
    ax.set_ylabel("Mass Balance Error")
    ax.set_title("Mass Balance Error by Component")
    ax.legend()
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()

Step 5: Spatial Hotspot Analysis
---------------------------------

Identify which nodes, elements, or reaches are mentioned most frequently
in diagnostic messages:

.. code-block:: python

    spatial = result.get_spatial_summary()

    for entity_type in ["nodes", "elements", "reaches"]:
        counts = spatial.get(entity_type, {})
        if counts:
            top = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"\nTop 5 {entity_type} by message count:")
            for entity_id, count in top:
                print(f"  {entity_type[:-1].title()} {entity_id}: {count} messages")

Step 6: Convergence Summary
-----------------------------

.. code-block:: python

    summary = result.get_convergence_summary()
    print("Convergence Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

Web Viewer Diagnostics
-----------------------

For an interactive experience, launch the web viewer and navigate to the
**Diagnostics** tab:

.. code-block:: bash

    pyiwfm viewer --model-dir /path/to/model

The Diagnostics tab provides:

- Searchable/filterable message table
- Interactive convergence chart with zoom
- Mass balance error chart with component toggles
- Spatial summary overlay on the model mesh

See :doc:`/user_guide/diagnostics` for more details on the web viewer
diagnostics interface.
