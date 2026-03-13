Drawdown Analysis
=================

This tutorial demonstrates how to compute and visualize groundwater drawdown
using pyiwfm's :class:`~pyiwfm.io.drawdown.DrawdownComputer`.

.. contents:: Table of Contents
   :local:
   :depth: 2

Learning Objectives
-------------------

- Load head data from HDF5 output
- Create a ``DrawdownComputer`` and compute drawdown at specific timesteps
- Compute maximum drawdown maps across all timesteps
- Visualize drawdown spatially using matplotlib
- Use the web viewer's drawdown mode

Prerequisites
-------------

.. code-block:: bash

    pip install pyiwfm[all]

You need a model with HDF5 head output (e.g., ``GW_HeadAll.hdf``).

Step 1: Load Head Data
----------------------

.. code-block:: python

    from pyiwfm.io.head_loader import LazyHeadDataLoader

    loader = LazyHeadDataLoader("Results/GW_HeadAll.hdf")
    print(f"Timesteps: {loader.n_timesteps}")
    print(f"Nodes: {loader.n_nodes}")
    print(f"Layers: {loader.n_layers}")

Step 2: Compute Drawdown
-------------------------

Create a :class:`~pyiwfm.io.drawdown.DrawdownComputer` and compute drawdown
at a specific timestep relative to the first timestep:

.. code-block:: python

    from pyiwfm.io.drawdown import DrawdownComputer

    computer = DrawdownComputer(loader)

    # Drawdown at timestep 100 relative to timestep 0 (initial conditions)
    dd = computer.compute_drawdown(timestep=100, reference_timestep=0, layer=1)
    print(f"Drawdown range: {dd.min():.2f} to {dd.max():.2f} ft")

Positive values indicate water level decline (drawdown); negative values
indicate water level rise.

Step 3: Element-Averaged Drawdown
----------------------------------

For element-centered visualization, compute element-averaged drawdown:

.. code-block:: python

    from pyiwfm.io import load_complete_model

    model = load_complete_model("Simulation/Simulation.in")

    dd_elem = computer.compute_drawdown_by_element(
        timestep=100, reference_timestep=0, layer=1, grid=model.grid
    )
    print(f"Elements: {len(dd_elem)}")

Step 4: Maximum Drawdown Map
-----------------------------

Compute the maximum drawdown at each node across all timesteps:

.. code-block:: python

    max_dd = computer.compute_max_drawdown_map(reference_timestep=0, layer=1)
    print(f"Max drawdown anywhere: {max_dd.max():.2f} ft")

Step 5: Visualize with Matplotlib
----------------------------------

.. code-block:: python

    import matplotlib.pyplot as plt
    from pyiwfm.visualization.plotting import plot_scalar_field

    fig, ax = plot_scalar_field(
        model.grid, max_dd, field_type="node",
        cmap="RdBu", show_mesh=False,
    )
    ax.set_title("Maximum Drawdown (ft)")
    ax.set_xlabel("Easting (ft)")
    ax.set_ylabel("Northing (ft)")
    plt.tight_layout()
    plt.show()

The ``RdBu`` (Red-Blue) diverging colormap highlights areas of drawdown (red)
and recovery (blue) relative to the reference timestep.

Step 6: Drawdown Range for Color Scaling
-----------------------------------------

Use :meth:`~pyiwfm.io.drawdown.DrawdownComputer.compute_drawdown_range` to
get robust min/max values (2nd–98th percentile) for consistent color scaling
across animation frames:

.. code-block:: python

    vmin, vmax = computer.compute_drawdown_range(reference_timestep=0, layer=1)
    print(f"Robust range: {vmin:.2f} to {vmax:.2f} ft")

Web Viewer: Drawdown Mode
--------------------------

The web viewer's Results Map tab supports drawdown visualization:

1. Launch the viewer: ``pyiwfm viewer --model-dir /path/to/model``
2. Navigate to the **Results Map** tab
3. In the controls panel, change **Color By** to **Drawdown**
4. A **Reference Timestep** slider appears — set the baseline timestep
5. Use the timestep slider to scrub through time and see drawdown evolve
6. The color scale automatically uses a diverging palette (blue = rising,
   red = falling water levels)

The drawdown computation uses the same ``DrawdownComputer`` on the backend,
with pagination support (``offset``/``limit``/``skip``) for smooth animation.
