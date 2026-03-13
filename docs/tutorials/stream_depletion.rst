Stream Depletion Analysis
=========================

This tutorial demonstrates how to compare baseline and pumping-scenario model
runs to quantify stream flow depletion using
:func:`~pyiwfm.io.stream_depletion.compute_stream_depletion`.

.. contents:: Table of Contents
   :local:
   :depth: 2

Learning Objectives
-------------------

- Set up baseline and pumping scenario model directories
- Compute stream depletion using ``compute_stream_depletion()``
- Interpret ``StreamDepletionResult`` and ``StreamDepletionReport`` fields
- Plot depletion timeseries and cumulative depletion

Prerequisites
-------------

.. code-block:: bash

    pip install pyiwfm[all]

You need two completed model runs with stream reach budget HDF5 output:

- **Baseline**: Model run without additional pumping
- **Scenario**: Model run with pumping stress applied

Step 1: Compute Stream Depletion
---------------------------------

.. code-block:: python

    from pyiwfm.io.stream_depletion import compute_stream_depletion

    report = compute_stream_depletion(
        baseline_dir="baseline_model/Results",
        scenario_dir="pumping_model/Results",
    )

    print(f"Reaches analyzed: {report.n_reaches}")
    print(f"Timesteps: {report.n_timesteps}")
    print(f"Total max depletion: {report.total_max_depletion:.2f}")
    print(f"Total cumulative depletion: {report.total_cumulative_depletion:.2f}")

To analyze only specific reaches:

.. code-block:: python

    report = compute_stream_depletion(
        baseline_dir="baseline_model/Results",
        scenario_dir="pumping_model/Results",
        reach_ids=[1, 5, 10],
    )

Step 2: Inspect Individual Reaches
------------------------------------

Each :class:`~pyiwfm.io.stream_depletion.StreamDepletionResult` contains
per-reach depletion data:

.. code-block:: python

    for result in report.results:
        print(f"Reach {result.reach_id} ({result.reach_name}):")
        print(f"  Max depletion: {result.max_depletion:.2f} at timestep {result.max_depletion_timestep}")
        print(f"  Total depletion: {result.total_depletion:.2f}")

Key fields:

- ``baseline_flow`` — Flow values from the baseline run (list of floats)
- ``scenario_flow`` — Flow values from the pumping scenario
- ``depletion`` — Per-timestep depletion (baseline - scenario)
- ``cumulative_depletion`` — Running total of depletion over time
- ``max_depletion`` — Peak depletion value
- ``total_depletion`` — Sum of all depletion values

Step 3: Plot Depletion Timeseries
----------------------------------

.. code-block:: python

    import matplotlib.pyplot as plt

    result = report.results[0]  # First reach

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Per-timestep depletion
    ax1.plot(result.depletion, color="tab:red", linewidth=1)
    ax1.set_ylabel("Depletion (flow units)")
    ax1.set_title(f"Stream Depletion — Reach {result.reach_id} ({result.reach_name})")
    ax1.axhline(0, color="gray", linestyle="--", linewidth=0.5)

    # Cumulative depletion
    ax2.fill_between(range(len(result.cumulative_depletion)),
                     result.cumulative_depletion, alpha=0.3, color="tab:red")
    ax2.plot(result.cumulative_depletion, color="tab:red", linewidth=1)
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("Cumulative Depletion")

    plt.tight_layout()
    plt.show()

Step 4: Compare Baseline vs Scenario Flows
--------------------------------------------

.. code-block:: python

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(result.baseline_flow, label="Baseline", color="tab:blue")
    ax.plot(result.scenario_flow, label="With Pumping", color="tab:orange")
    ax.fill_between(range(len(result.depletion)),
                    result.baseline_flow, result.scenario_flow,
                    alpha=0.2, color="tab:red", label="Depletion")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Stream Flow")
    ax.set_title(f"Reach {result.reach_id}: Baseline vs Pumping Scenario")
    ax.legend()
    plt.tight_layout()
    plt.show()

Step 5: Multi-Reach Summary
----------------------------

.. code-block:: python

    # Bar chart of max depletion by reach
    fig, ax = plt.subplots(figsize=(10, 6))

    reach_names = [r.reach_name for r in report.results]
    max_depletions = [r.max_depletion for r in report.results]

    ax.barh(reach_names, max_depletions, color="tab:red", alpha=0.7)
    ax.set_xlabel("Maximum Depletion (flow units)")
    ax.set_title("Maximum Stream Depletion by Reach")
    plt.tight_layout()
    plt.show()

Serialization
-------------

Convert results to dictionaries for JSON export or API responses:

.. code-block:: python

    report_dict = report.to_dict()
    import json
    print(json.dumps(report_dict, indent=2, default=str))
