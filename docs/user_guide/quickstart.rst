Quickstart Guide
================

This guide walks you through the basic features of pyiwfm.

Loading an Existing IWFM Model
------------------------------

The easiest way to work with IWFM is to load an existing model:

.. code-block:: python

    from pyiwfm.core.model import IWFMModel

    # Load from preprocessor input file (mesh, stratigraphy, streams, lakes geometry)
    model = IWFMModel.from_preprocessor("Preprocessor/Preprocessor.in")
    print(model.summary())

    # Load complete model from simulation input file (all components)
    model = IWFMModel.from_simulation("Simulation/Simulation.in")
    print(f"Loaded model with {model.n_nodes} nodes")
    print(f"Groundwater wells: {len(model.groundwater.wells) if model.groundwater else 0}")

    # Load using both preprocessor and simulation files
    model = IWFMModel.from_simulation_with_preprocessor(
        "Simulation/Simulation.in",
        "Preprocessor/Preprocessor.in"
    )

    # Load from HDF5 file
    model = IWFMModel.from_hdf5("model.h5")

    # Save model to different formats
    model.to_hdf5("backup.h5")
    model.to_preprocessor("output/")
    model.to_simulation("full_output/")

Creating a Simple Mesh
----------------------

Create a mesh manually by defining nodes and elements:

.. code-block:: python

    from pyiwfm.core.mesh import AppGrid, Node, Element

    # Define nodes (id, x, y coordinates)
    nodes = {
        1: Node(id=1, x=0.0, y=0.0, is_boundary=True),
        2: Node(id=2, x=100.0, y=0.0, is_boundary=True),
        3: Node(id=3, x=200.0, y=0.0, is_boundary=True),
        4: Node(id=4, x=0.0, y=100.0, is_boundary=True),
        5: Node(id=5, x=100.0, y=100.0, is_boundary=False),
        6: Node(id=6, x=200.0, y=100.0, is_boundary=True),
    }

    # Define elements (id, vertices as tuple, subregion)
    elements = {
        1: Element(id=1, vertices=(1, 2, 5, 4), subregion=1),
        2: Element(id=2, vertices=(2, 3, 6, 5), subregion=1),
    }

    # Create the grid
    grid = AppGrid(nodes=nodes, elements=elements)

    # Compute mesh connectivity
    grid.compute_connectivity()

    print(f"Mesh has {grid.n_nodes} nodes and {grid.n_elements} elements")

Generating a Mesh from a Boundary
---------------------------------

Use the Triangle mesh generator to create a mesh from a boundary polygon:

.. code-block:: python

    import numpy as np
    from pyiwfm.mesh_generation import TriangleMeshGenerator
    from pyiwfm.mesh_generation.constraints import BoundaryConstraint

    # Define a square boundary
    boundary_coords = np.array([
        [0.0, 0.0],
        [1000.0, 0.0],
        [1000.0, 1000.0],
        [0.0, 1000.0],
    ])
    boundary = BoundaryConstraint(coordinates=boundary_coords)

    # Generate mesh
    generator = TriangleMeshGenerator()
    result = generator.generate(boundary, max_area=10000.0, min_angle=25.0)

    print(f"Generated {result.n_elements} elements")

    # Convert to AppGrid for use with other pyiwfm modules
    grid = generator.to_appgrid(result)

Creating Stratigraphy
---------------------

Define the vertical layer structure for your model:

.. code-block:: python

    import numpy as np
    from pyiwfm.core.stratigraphy import Stratigraphy

    n_nodes = grid.n_nodes
    n_layers = 3

    # Ground surface elevation (same for all nodes in this example)
    gs_elev = np.full(n_nodes, 100.0)

    # Layer top elevations (n_nodes x n_layers)
    top_elev = np.column_stack([
        np.full(n_nodes, 100.0),   # Layer 1 top
        np.full(n_nodes, 70.0),    # Layer 2 top
        np.full(n_nodes, 40.0),    # Layer 3 top
    ])

    # Layer bottom elevations
    bottom_elev = np.column_stack([
        np.full(n_nodes, 70.0),    # Layer 1 bottom
        np.full(n_nodes, 40.0),    # Layer 2 bottom
        np.full(n_nodes, 0.0),     # Layer 3 bottom
    ])

    # All nodes active in all layers
    active_node = np.ones((n_nodes, n_layers), dtype=bool)

    stratigraphy = Stratigraphy(
        n_layers=n_layers,
        n_nodes=n_nodes,
        gs_elev=gs_elev,
        top_elev=top_elev,
        bottom_elev=bottom_elev,
        active_node=active_node,
    )

    print(f"Created {stratigraphy.n_layers}-layer stratigraphy")

Exporting to GIS Formats
------------------------

Export your mesh to GeoPackage for use in GIS software:

.. code-block:: python

    from pyiwfm.visualization import GISExporter

    # Create exporter with coordinate reference system
    exporter = GISExporter(
        grid=grid,
        stratigraphy=stratigraphy,
        crs="EPSG:26910"  # NAD83 / UTM zone 10N
    )

    # Export to GeoPackage (includes nodes, elements, boundary)
    exporter.export_geopackage("model.gpkg")

    # Or export to Shapefiles
    exporter.export_shapefiles("shapefiles/")

    # Or export individual layers as GeoJSON
    exporter.export_geojson("nodes.geojson", layer="nodes")
    exporter.export_geojson("elements.geojson", layer="elements")

Plotting the Mesh
-----------------

Visualize your mesh using matplotlib:

.. code-block:: python

    from pyiwfm.visualization.plotting import plot_mesh, plot_scalar_field
    import numpy as np

    # Plot the mesh
    fig, ax = plot_mesh(grid, show_edges=True, show_node_ids=True)
    fig.savefig("mesh.png", dpi=150)

    # Plot a scalar field (e.g., head values)
    head_values = np.random.uniform(50, 100, grid.n_nodes)
    fig, ax = plot_scalar_field(
        grid, head_values,
        field_type="node",
        cmap="viridis",
        show_colorbar=True
    )
    fig.savefig("heads.png", dpi=150)

Exporting to VTK for 3D Visualization
-------------------------------------

Export 3D mesh for visualization in ParaView:

.. code-block:: python

    from pyiwfm.visualization import VTKExporter

    # Create VTK exporter
    vtk_exporter = VTKExporter(grid=grid, stratigraphy=stratigraphy)

    # Export 2D surface mesh
    vtk_exporter.export_vtu("mesh_2d.vtu", mode="2d")

    # Export 3D volumetric mesh
    vtk_exporter.export_vtu("mesh_3d.vtu", mode="3d")

    # Export with scalar data
    vtk_exporter.export_vtu(
        "mesh_with_data.vtu",
        mode="3d",
        node_scalars={"head": head_values},
    )

Comparing Models
----------------

Compare two models and generate a report:

.. code-block:: python

    from pyiwfm.comparison import ModelDiffer, ComparisonReport
    from pyiwfm.comparison.metrics import ComparisonMetrics

    # Compare two meshes
    differ = ModelDiffer()
    mesh_diff = differ.diff_meshes(grid1, grid2)

    print(f"Meshes are identical: {mesh_diff.is_identical}")
    print(f"Nodes added: {mesh_diff.nodes_added}")
    print(f"Elements modified: {mesh_diff.elements_modified}")

    # Compute comparison metrics
    observed_heads = np.array([50.0, 55.0, 60.0, 65.0, 70.0])
    simulated_heads = np.array([51.0, 54.5, 61.0, 64.0, 71.0])

    metrics = ComparisonMetrics.compute(observed_heads, simulated_heads)
    print(metrics.summary())
    print(f"Rating: {metrics.rating()}")

    # Generate a report
    from pyiwfm.comparison import ModelDiff

    model_diff = ModelDiff(mesh_diff=mesh_diff)
    report = ComparisonReport(
        title="Model Comparison",
        model_diff=model_diff,
        head_metrics=metrics,
    )

    # Save as HTML
    report.save("comparison_report.html")

Next Steps
----------

- See the :doc:`/tutorials/index` for detailed workflows
- Check the :doc:`/api/index` for complete API reference
- Explore :doc:`mesh` for advanced mesh operations
- Learn about :doc:`io` for reading/writing IWFM files
