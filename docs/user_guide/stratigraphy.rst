Working with Stratigraphy
=========================

This guide covers creating and manipulating model stratigraphy (layer structure).

Stratigraphy Fundamentals
-------------------------

IWFM stratigraphy defines the vertical structure of aquifer layers:

- **Ground surface elevation** at each node
- **Layer top elevations** for each node and layer
- **Layer bottom elevations** for each node and layer

Because IWFM uses a finite element formulation, all nodes in the mesh
participate in the solution — there is no concept of "inactive cells" as
in finite difference models.

Creating Stratigraphy
---------------------

Basic stratigraphy with uniform layer thicknesses:

.. code-block:: python

    import numpy as np
    from pyiwfm.core.stratigraphy import Stratigraphy

    # Model dimensions
    n_nodes = 100
    n_layers = 3

    # Ground surface elevation (e.g., from DEM)
    gs_elev = np.full(n_nodes, 100.0)  # 100 ft elevation

    # Layer structure: 3 layers of 30 ft each
    layer_thickness = 30.0

    # Top elevations for each layer
    top_elev = np.column_stack([
        gs_elev,                           # Layer 1 top = ground surface
        gs_elev - layer_thickness,         # Layer 2 top
        gs_elev - 2 * layer_thickness,     # Layer 3 top
    ])

    # Bottom elevations for each layer
    bottom_elev = np.column_stack([
        gs_elev - layer_thickness,         # Layer 1 bottom
        gs_elev - 2 * layer_thickness,     # Layer 2 bottom
        gs_elev - 3 * layer_thickness,     # Layer 3 bottom (aquifer base)
    ])

    # All nodes active in all layers
    active_node = np.ones((n_nodes, n_layers), dtype=bool)

    # Create stratigraphy
    strat = Stratigraphy(
        n_layers=n_layers,
        n_nodes=n_nodes,
        gs_elev=gs_elev,
        top_elev=top_elev,
        bottom_elev=bottom_elev,
        active_node=active_node,
    )

    print(f"Stratigraphy: {strat.n_layers} layers, {strat.n_nodes} nodes")

Variable Layer Thicknesses
--------------------------

Create stratigraphy with spatially varying thicknesses:

.. code-block:: python

    import numpy as np
    from pyiwfm.core.stratigraphy import Stratigraphy

    # Assume we have node coordinates
    x = np.array([node.x for node in grid.iter_nodes()])
    y = np.array([node.y for node in grid.iter_nodes()])
    n_nodes = len(x)
    n_layers = 2

    # Ground surface varies with topography
    gs_elev = 100.0 - 0.01 * x + 0.005 * y  # Slope towards +x

    # Layer 1 thickness varies spatially
    layer1_thickness = 30.0 + 0.02 * x  # Thicker to the east

    # Layer 2 is uniform thickness
    layer2_thickness = 50.0

    # Calculate elevations
    top_elev = np.column_stack([
        gs_elev,                         # Layer 1 top
        gs_elev - layer1_thickness,      # Layer 2 top
    ])

    bottom_elev = np.column_stack([
        gs_elev - layer1_thickness,                        # Layer 1 bottom
        gs_elev - layer1_thickness - layer2_thickness,     # Layer 2 bottom
    ])

    active_node = np.ones((n_nodes, n_layers), dtype=bool)

    strat = Stratigraphy(
        n_layers=n_layers,
        n_nodes=n_nodes,
        gs_elev=gs_elev,
        top_elev=top_elev,
        bottom_elev=bottom_elev,
        active_node=active_node,
    )

Accessing Stratigraphy Data
---------------------------

Query stratigraphy properties via the convenience accessors (preferred) or
the raw arrays:

.. code-block:: python

    # Aquifer layer thickness at all nodes, layer 0 (0-based)
    layer1_thickness = strat.get_layer_thickness(layer=0)
    print(f"Layer 1 thickness: min={layer1_thickness.min():.1f}, "
          f"max={layer1_thickness.max():.1f}")

    # Total aquifer-column thickness (summed across all layers) at each node
    total_thickness = strat.get_total_thickness()

    # Per-node elevations (ground surface, layer tops, layer bottoms)
    gs, tops, bottoms = strat.get_node_elevations(node_idx=50)
    print(f"Node 51 GS elev: {gs:.1f}, layer 1 top: {tops[0]:.1f}")

Accessing Aquitard Thicknesses
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The IWFM stratigraphy format stores alternating
*aquitard–aquifer* thicknesses per node. The ``Stratigraphy`` dataclass
exposes the computed thicknesses directly — aquitard ``k`` sits above
aquifer layer ``k`` (aquitard ``0`` is the uppermost one, between the ground
surface and the top of aquifer layer 0):

.. code-block:: python

    # Thickness of the top aquitard (above aquifer layer 0) at every node
    aqt0 = strat.get_aquitard_thickness(aquitard=0)
    # == strat.gs_elev - strat.top_elev[:, 0]

    # Thickness of the aquitard between aquifer layers 0 and 1
    aqt1 = strat.get_aquitard_thickness(aquitard=1)
    # == strat.bottom_elev[:, 0] - strat.top_elev[:, 1]

    # Full (n_nodes, n_aquitards) array in one vectorised pass
    all_aqt = strat.get_all_aquitard_thicknesses()
    print(all_aqt.shape)  # (n_nodes, n_aquitards)

    # All aquitards for a specific node (as a Python list)
    node_aqt = strat.get_node_aquitards(node_idx=50)

By IWFM convention ``strat.n_aquitards == strat.n_layers``; a thickness of
zero is valid and simply indicates no aquitard at that layer boundary.

Loading Stratigraphy from Arrays
--------------------------------

Load stratigraphy from existing data arrays:

.. code-block:: python

    import numpy as np

    # Load from files (example paths)
    gs_elev = np.loadtxt("ground_surface.txt")
    top_elev = np.loadtxt("layer_tops.txt")
    bottom_elev = np.loadtxt("layer_bottoms.txt")

    n_nodes = len(gs_elev)
    n_layers = top_elev.shape[1]

    # Create active node array (all active by default)
    active_node = np.ones((n_nodes, n_layers), dtype=bool)

    strat = Stratigraphy(
        n_layers=n_layers,
        n_nodes=n_nodes,
        gs_elev=gs_elev,
        top_elev=top_elev,
        bottom_elev=bottom_elev,
        active_node=active_node,
    )

Validating Stratigraphy
-----------------------

Check for common stratigraphy issues:

.. code-block:: python

    def validate_stratigraphy(strat: Stratigraphy) -> list[str]:
        """Check stratigraphy for common issues."""
        issues = []

        # Check that layer tops are above bottoms
        for layer in range(strat.n_layers):
            bad_thickness = strat.top_elev[:, layer] < strat.bottom_elev[:, layer]
            if np.any(bad_thickness):
                n_bad = np.sum(bad_thickness)
                issues.append(f"Layer {layer + 1}: {n_bad} nodes with negative thickness")

        # Check that layers don't overlap
        for layer in range(strat.n_layers - 1):
            overlap = strat.bottom_elev[:, layer] < strat.top_elev[:, layer + 1]
            if np.any(overlap):
                n_overlap = np.sum(overlap)
                issues.append(f"Layers {layer + 1}-{layer + 2}: {n_overlap} nodes overlap")

        # Check that ground surface is at or above layer 1 top
        above_gs = strat.top_elev[:, 0] > strat.gs_elev + 0.01
        if np.any(above_gs):
            n_above = np.sum(above_gs)
            issues.append(f"Layer 1 top above ground surface at {n_above} nodes")

        return issues

    # Validate
    issues = validate_stratigraphy(strat)
    if issues:
        for issue in issues:
            print(f"WARNING: {issue}")
    else:
        print("Stratigraphy validation passed")
