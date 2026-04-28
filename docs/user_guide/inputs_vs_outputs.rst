Editable inputs vs. computed outputs
====================================

pyiwfm reads two distinct categories of model data, and they have
different round-trip stories:

- **Editable model inputs** — files the user authored (mesh, stratigraphy,
  aquifer parameters, time-series boundary conditions, etc.). pyiwfm
  reads them, lets you mutate them in Python, and writes them back so
  IWFM can re-run the model. Every input has both a reader and a writer.

- **Computed outputs / analysis modules** — files IWFM produces (heads,
  hydrographs, subsidence, budgets) and Python-side analysis results
  (drawdown reports, stream depletion). These are read-only by design:
  if you want different outputs, edit the inputs and re-run IWFM.

The audit on which this distinction is based is documented in CHANGELOG
v1.3.0; the table below is the canonical reference.

Editable model inputs
---------------------

Every component that loads from disk also writes back to disk. The
mutation helpers on :class:`~pyiwfm.core.model.IWFMModel` are the
documented Python API for editing them; see :doc:`mutating_models` for
end-to-end examples.

.. list-table::
   :header-rows: 1
   :widths: 22 26 26 26

   * - Component
     - Reader / Writer
     - Mutation helper
     - One-liner
   * - Mesh (nodes, elements)
     - ``io.ascii.read_nodes`` / ``read_elements`` ↔ ``io.preprocessor_writer``
     - direct on ``model.mesh``
     - ``model.mesh.nodes[1].x = 0.0``
   * - Stratigraphy
     - ``io.ascii.read_stratigraphy`` ↔ same writer
     - :meth:`IWFMModel.set_stratigraphy_from_thicknesses`
     - ``model.set_stratigraphy_from_thicknesses(...)``
   * - Aquifer parameters (Kh, Kv, Ss, Sy, aquitard_kv)
     - ``io.gw_main_writer`` ↔ same reader
     - :meth:`IWFMModel.set_aquifer_parameter` / :meth:`~IWFMModel.set_aquifer_parameter_at`
     - ``model.set_aquifer_parameter("kh", layer=1, values=arr)``
   * - Initial heads
     - ``io.gw_main_writer`` (embedded in GW main file)
     - direct on ``model.groundwater.aquifer_params``
     - ``params.heads[i, layer] = h``
   * - GW boundary conditions / pumping / tile drains
     - ``io.gw_boundary_writer``, ``gw_pumping_writer``, ``gw_tiledrain_writer``
     - direct on ``model.groundwater``
     - ``gw.add_boundary_condition(bc)``
   * - GW observation wells (hydrograph locations)
     - ``io.gw_main_writer`` (embedded section)
     - :meth:`IWFMModel.add_observation_well` / :meth:`~IWFMModel.remove_observation_well`
     - ``model.add_observation_well(node_id, layer, x, y, name)``
   * - Stream network (reaches, nodes, diversions, bypasses, inflows)
     - ``io.stream_writer``, ``stream_diversion_writer``, ``stream_bypass_writer``, ``stream_inflow_writer``
     - direct on ``model.streams``
     - ``stream.add_reach(reach)``
   * - Lakes
     - ``io.lake_writer``
     - direct on ``model.lakes``
     - ``lakes.add_lake(lake)``
   * - Root zone (crop types, soil, land use)
     - ``io.rootzone_writer`` (+ v4.x / native / nonponded / ponded / urban variants)
     - direct on ``model.rootzone``
     - ``rootzone.crop_types[id] = ct``
   * - Small watersheds
     - ``io.small_watershed_writer``
     - direct on ``model.small_watersheds``
     - ``sw.add_watershed(ws)``
   * - Unsaturated zone
     - ``io.unsaturated_zone_writer``
     - direct on ``model.unsaturated_zone``
     - ``uz.add_node(n)``
   * - Time series (pumping schedules, BCs, inflows, climate)
     - ``io.timeseries_writer``, ``timeseries_ascii``
     - direct on the relevant time-series object
     - ``ts.values[i] = v``
   * - Supply adjustment specification
     - ``io.supply_adjust.read_supply_adjustment`` ↔ ``write_supply_adjustment``
     - direct on ``model.supply_adjustment``
     - ``model.supply_adjustment.codes[t] = "10"``
   * - Simulation main file (dates, solver options)
     - ``io.simulation_writer``
     - direct on the simulation config
     - (configure before re-running with ``IWFMRunner``)

To save a mutated model:

.. code-block:: python

    from pyiwfm.io import save_complete_model
    save_complete_model(model, output_dir="modified_model")

Computed outputs / analysis modules
-----------------------------------

These modules read data IWFM produced or analysis pyiwfm computes. They
have **no writer for the canonical output format** — that would be a
parallel write path that drifts from the source of truth. Where users
need a stakeholder-friendly export (CSV, JSON, Excel, plot, GeoJSON),
the module exposes dedicated **report writers** that don't pretend to be
IWFM input.

If you want different outputs, edit the inputs (above) and re-run IWFM
via :class:`~pyiwfm.runner.IWFMRunner`.

.. list-table::
   :header-rows: 1
   :widths: 28 28 44

   * - Module
     - Source
     - Available report exports
   * - ``io.timeseries_io.LazyNodalLoader``
     - IWFM HDF5 (``GWHeadAtAllNodes``)
     - none — see web-viewer download endpoints in ``visualization/webapi``
   * - ``io.timeseries_io.LazyTabularLoader`` / ``io.hydrograph_reader``
     - IWFM hydrograph output (.out)
     - same
   * - ``io.gw_subsidence`` (read side)
     - IWFM ``SubsidenceAtAllNodes`` HDF5
     - same
   * - ``io.budget``, ``io.zbudget``
     - IWFM ``*_Budget.hdf``
     - ``io.budget_excel``, ``io.zbudget_excel`` (exposed via ``pyiwfm budget`` / ``pyiwfm zbudget`` CLI)
   * - ``io.simulation_messages``
     - IWFM ``SimulationMessages.out`` log
     - none (read for diagnostics)
   * - ``io.area_loader``
     - cached HDF5 produced by the rootzone pipeline
     - none — the canonical input path is ``rootzone_writer``
   * - ``io.cache_builder`` / ``io.cache_loader``
     - internal SQLite cache for the web viewer
     - not user-facing
   * - ``io.drawdown``
     - computes drawdown by comparing two model runs
     - tabular CSV / JSON / Excel writers (planned, Phase 2.2.a)
   * - ``io.stream_depletion``
     - computes stream depletion by comparing two model runs
     - tabular CSV / JSON / Excel writers + plots + maps (planned, Phase 2.2.a)

Why this distinction matters
----------------------------

A user editing ``LazyNodalLoader``-loaded heads on disk and saving
them would see **no change** in the next IWFM run — IWFM would overwrite
the heads HDF5 with whatever new heads the simulation produces. The
model-update path is always: edit the inputs, re-run IWFM, read the new
outputs.
