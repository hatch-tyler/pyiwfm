Simulation Diagnostics
======================

pyiwfm can parse IWFM ``SimulationMessages.out`` files to extract structured
diagnostic information — message severities, convergence tracking, mass balance
errors, and spatial entity references.

.. contents:: Table of Contents
   :local:
   :depth: 2

Reading Simulation Messages
---------------------------

Use :class:`~pyiwfm.io.simulation_messages.SimulationMessagesReader` to parse
a ``SimulationMessages.out`` file:

.. code-block:: python

    from pyiwfm.io.simulation_messages import SimulationMessagesReader

    reader = SimulationMessagesReader("SimulationMessages.out")
    result = reader.read()

    print(f"Total messages: {len(result.messages)}")
    print(f"Warnings: {result.warning_count}")
    print(f"Errors: {result.error_count}")
    print(f"Total runtime: {result.total_runtime}")

Filtering by Severity
~~~~~~~~~~~~~~~~~~~~~

Messages are classified into four severity levels: ``MESSAGE``, ``INFO``,
``WARN``, and ``FATAL``:

.. code-block:: python

    from pyiwfm.io.simulation_messages import MessageSeverity

    warnings = result.filter_by_severity(MessageSeverity.WARN)
    for msg in warnings[:5]:
        print(f"[{msg.procedure}] {msg.text}")

Convergence Tracking
--------------------

The parser extracts per-timestep convergence iteration counts and residuals:

.. code-block:: python

    for rec in result.convergence_records[:5]:
        print(f"Timestep {rec.timestep_index}: {rec.iteration_count} iterations, "
              f"converged={rec.convergence_achieved}")

    summary = result.get_convergence_summary()
    print(f"Max iterations: {summary['max_iterations']}")
    print(f"Avg iterations: {summary['avg_iterations']:.1f}")

Plotting convergence over time:

.. code-block:: python

    import matplotlib.pyplot as plt

    dates = [rec.date for rec in result.convergence_records]
    iters = [rec.iteration_count for rec in result.convergence_records]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(range(len(iters)), iters, linewidth=0.5)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Iteration Count")
    ax.set_title("Convergence Iterations Over Time")
    plt.tight_layout()
    plt.show()

Mass Balance Errors
-------------------

Mass balance error records track per-component errors at each timestep:

.. code-block:: python

    for rec in result.mass_balance_records[:5]:
        print(f"Timestep {rec.timestep_index} [{rec.component}]: "
              f"error={rec.error_value:.4f}, {rec.error_percent:.2f}%")

Spatial Summary
---------------

Map message counts to spatial entities (nodes, elements, reaches) for
identifying problem areas:

.. code-block:: python

    spatial = result.get_spatial_summary()

    # Top 5 nodes with most messages
    node_counts = spatial.get("nodes", {})
    top_nodes = sorted(node_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    for node_id, count in top_nodes:
        print(f"Node {node_id}: {count} messages")

If you have a loaded mesh, convert to a GeoDataFrame for GIS analysis:

.. code-block:: python

    gdf = result.to_geodataframe(model.grid)
    gdf.to_file("diagnostics.gpkg", driver="GPKG")

Web Viewer Diagnostics Tab
--------------------------

The web viewer includes a **Diagnostics** tab that provides an interactive
interface for exploring simulation diagnostic data:

- **Messages table**: Filterable by severity with pagination
- **Convergence chart**: Iteration counts plotted over simulation timesteps
- **Mass balance error chart**: Per-component error timeseries
- **Spatial summary**: Message counts overlaid on the model mesh

Launch the viewer and navigate to the Diagnostics tab:

.. code-block:: bash

    pyiwfm viewer --model-dir /path/to/model

The diagnostics data is served by the ``/api/diagnostics/`` endpoints:

- ``GET /api/diagnostics/messages`` — Paginated message list with severity filtering
- ``GET /api/diagnostics/convergence`` — Convergence iteration timeseries
- ``GET /api/diagnostics/mass-balance`` — Mass balance error timeseries by component
- ``GET /api/diagnostics/summary`` — Combined diagnostics summary
- ``GET /api/diagnostics/spatial-summary`` — Spatial message counts for map overlay
