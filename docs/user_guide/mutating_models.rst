Mutating loaded models
======================

After loading an :class:`~pyiwfm.core.model.IWFMModel`, you can edit its
inputs in place using the **mutation helpers** on the model object. These
methods validate inputs, mutate the underlying component, and record what
changed in ``model._dirty`` so save paths can warn about partial writes.

The helpers are the documented, supported API for editing a loaded model
(scenario analysis, calibration scripts, batch perturbation studies). The
underlying component attributes (``model.groundwater.aquifer_params.kh``,
etc.) remain accessible if you need finer-grained control.

For an explanation of what data is editable input versus computed output,
see :doc:`inputs_vs_outputs`.

Editing aquifer parameters
--------------------------

Replace an entire layer's worth of values for a named parameter:

.. code-block:: python

    import numpy as np
    from pyiwfm import IWFMModel

    model = IWFMModel.from_simulation("Simulation/Simulation.in")

    # New horizontal-K array for layer 1, one value per node
    new_kh = np.full(model.groundwater.n_nodes, 1.5e-4)
    model.set_aquifer_parameter("kh", layer=1, values=new_kh)

Or update a single ``(node, layer)`` cell:

.. code-block:: python

    model.set_aquifer_parameter_at("kh", node_id=1234, layer=2, value=2.0e-4)

Recognized parameter names match the :class:`AquiferParameters` short-name
dispatcher:

.. list-table::
   :header-rows: 1

   * - Short name
     - Stored as
   * - ``"kh"``
     - ``aquifer_params.kh``
   * - ``"kv"``
     - ``aquifer_params.kv``
   * - ``"ss"``
     - ``aquifer_params.specific_storage``
   * - ``"sy"``
     - ``aquifer_params.specific_yield``
   * - ``"aquitard_kv"``
     - ``aquifer_params.aquitard_kv``

Both helpers validate the layer (1-based, must be in ``[1, n_layers]``),
the node ID (1-based, must be in ``[1, n_nodes]``), and the values shape.
Out-of-range IDs raise :class:`ValueError`; unknown parameter names raise
:class:`KeyError`.

Editing stratigraphy
--------------------

Replace the entire :class:`~pyiwfm.core.stratigraphy.Stratigraphy` from
thickness arrays. This wraps
:meth:`~pyiwfm.core.stratigraphy.Stratigraphy.from_thicknesses` and
additionally checks that ``gs_elev`` matches the loaded mesh:

.. code-block:: python

    model.set_stratigraphy_from_thicknesses(
        gs_elev=ground_surface_elevations,            # (n_nodes,)
        aquitard_thicknesses=aquitard_array,          # (n_nodes, n_layers)
        aquifer_thicknesses=aquifer_array,            # (n_nodes, n_layers)
    )

IWFM convention: aquitard ``k`` sits above aquifer layer ``k``; aquitard
``0`` is between the ground surface and aquifer layer 0. Negative
thicknesses raise
:class:`~pyiwfm.core.exceptions.StratigraphyError`.

Adding and removing observation wells
-------------------------------------

Append a groundwater hydrograph location:

.. code-block:: python

    model.add_observation_well(
        node_id=42,
        layer=1,
        x=614000.0,
        y=4271000.0,
        name="MW-7",
    )

Remove all locations matching a name:

.. code-block:: python

    n_removed = model.remove_observation_well("MW-7")

Saving changes
--------------

After mutation, save the model to a new directory using
:func:`~pyiwfm.io.save_complete_model`:

.. code-block:: python

    from pathlib import Path
    from pyiwfm.io import save_complete_model

    save_complete_model(model, output_dir=Path("scenario_high_kh"))

The set of components mutated since the last load is available on
``model._dirty``:

.. code-block:: python

    if model._dirty:
        print("Modified:", sorted(model._dirty))

Calibration workflow example
----------------------------

A typical calibration loop reads parameter perturbations from PEST++ and
applies them via the mutation API:

.. code-block:: python

    from pyiwfm import IWFMModel
    from pyiwfm.io import save_complete_model

    model = IWFMModel.from_simulation("Simulation/Simulation.in")

    # Apply perturbations from PEST parameter file (illustrative)
    for param_name, layer, values in load_pest_parameters("params.par"):
        model.set_aquifer_parameter(param_name, layer=layer, values=values)

    save_complete_model(model, output_dir="run_iter_42")
    # ... invoke IWFMRunner to run the new model ...
