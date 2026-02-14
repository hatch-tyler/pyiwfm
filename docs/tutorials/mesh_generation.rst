Tutorial: Mesh Generation
=========================

This tutorial demonstrates how to create a finite element mesh for an IWFM model,
starting from a boundary polygon and adding constraints for streams and wells.

Learning Objectives
-------------------

By the end of this tutorial, you will be able to:

1. Define a model boundary from coordinates
2. Add stream network constraints
3. Add well location constraints
4. Create refinement zones
5. Generate a triangular mesh
6. Export the mesh to GIS formats

Step 1: Import Libraries
------------------------

.. code-block:: python

    import numpy as np
    from pathlib import Path

    from pyiwfm.core.mesh import AppGrid, Node, Element
    from pyiwfm.mesh_generation import TriangleMeshGenerator
    from pyiwfm.mesh_generation.constraints import (
        BoundaryConstraint,
        LineConstraint,
        PointConstraint,
        RefinementZone,
    )
    from pyiwfm.visualization import GISExporter
    from pyiwfm.visualization.plotting import plot_mesh

Step 2: Define the Model Boundary
---------------------------------

Define the boundary of your model domain. This can come from a shapefile or
be defined manually with coordinates.

.. code-block:: python

    # Define an irregular polygon boundary (coordinates in feet)
    boundary_coords = np.array([
        [0.0, 0.0],
        [5000.0, 0.0],
        [6000.0, 1000.0],
        [6000.0, 4000.0],
        [5000.0, 5000.0],
        [3000.0, 5500.0],
        [1000.0, 5000.0],
        [0.0, 3000.0],
    ])

    # Create boundary constraint
    boundary = BoundaryConstraint(coordinates=boundary_coords)

    print(f"Boundary has {len(boundary_coords)} vertices")
    print(f"Approximate area: {boundary.area:.0f} sq ft")

Step 3: Add Stream Constraints
------------------------------

Define stream reaches that must be honored by the mesh. The mesh generator
will place nodes along the stream and create elements that follow the stream.

.. code-block:: python

    # Main river (flows from NE to SW)
    river_coords = np.array([
        [5500.0, 4500.0],  # Upstream
        [4000.0, 3500.0],
        [3000.0, 3000.0],
        [2000.0, 2000.0],
        [500.0, 1000.0],   # Downstream
    ])
    river = LineConstraint(coordinates=river_coords, name="Main River")

    # Tributary (joins main river)
    tributary_coords = np.array([
        [4500.0, 1000.0],  # Upstream
        [3500.0, 1500.0],
        [2000.0, 2000.0],  # Confluence
    ])
    tributary = LineConstraint(coordinates=tributary_coords, name="Tributary")

    stream_constraints = [river, tributary]
    print(f"Added {len(stream_constraints)} stream constraints")

Step 4: Add Well Locations
--------------------------

Add point constraints for well locations. The mesh generator will ensure
nodes exist at these exact locations.

.. code-block:: python

    # Define well locations
    well_locations = [
        (1500.0, 3500.0, "Well-1"),
        (3500.0, 4000.0, "Well-2"),
        (4500.0, 2500.0, "Well-3"),
        (2000.0, 1000.0, "Well-4"),
    ]

    well_constraints = [
        PointConstraint(x=x, y=y, name=name)
        for x, y, name in well_locations
    ]

    print(f"Added {len(well_constraints)} well locations")

Step 5: Define Refinement Zones
-------------------------------

Create areas where you want finer mesh resolution, such as near pumping wells
or in areas of interest.

.. code-block:: python

    # Refinement zone around wells 1 and 2 (northern area)
    refine_north = RefinementZone(
        coordinates=np.array([
            [1000.0, 3000.0],
            [4500.0, 3000.0],
            [4500.0, 4500.0],
            [1000.0, 4500.0],
        ]),
        max_area=10000.0,  # Smaller elements
        name="North Refinement",
    )

    # Refinement zone along main river corridor
    refine_river = RefinementZone(
        coordinates=np.array([
            [500.0, 500.0],
            [4500.0, 2500.0],
            [4000.0, 4000.0],
            [1500.0, 3500.0],
            [0.0, 2500.0],
        ]),
        max_area=15000.0,
        name="River Corridor",
    )

    refinement_zones = [refine_north, refine_river]
    print(f"Added {len(refinement_zones)} refinement zones")

Step 6: Generate the Mesh
-------------------------

Use the Triangle mesh generator to create the mesh.

.. code-block:: python

    # Create mesh generator
    generator = TriangleMeshGenerator()

    # Generate mesh with all constraints
    result = generator.generate(
        boundary=boundary,
        line_constraints=stream_constraints,
        point_constraints=well_constraints,
        refinement_zones=refinement_zones,
        max_area=50000.0,    # Maximum element area (coarse areas)
        min_angle=25.0,      # Minimum angle for quality
    )

    print(f"Generated mesh:")
    print(f"  Nodes: {result.n_nodes}")
    print(f"  Elements: {result.n_elements}")

Step 7: Convert to AppGrid
--------------------------

Convert the mesh result to an AppGrid for use with other pyiwfm modules.

.. code-block:: python

    # Convert to AppGrid
    grid = generator.to_appgrid(result)

    # Verify the conversion
    print(f"AppGrid created:")
    print(f"  Nodes: {grid.n_nodes}")
    print(f"  Elements: {grid.n_elements}")
    print(f"  Boundary nodes: {grid.n_boundary_nodes}")

Step 8: Visualize the Mesh
--------------------------

Plot the mesh to verify it looks correct.

.. code-block:: python

    # Create mesh plot
    fig, ax = plot_mesh(
        grid,
        show_edges=True,
        edge_color="gray",
        fill_color="lightblue",
        alpha=0.3,
        figsize=(12, 10),
    )

    # Add stream lines to the plot
    for constraint in stream_constraints:
        coords = constraint.coordinates
        ax.plot(coords[:, 0], coords[:, 1], 'b-', linewidth=2, label=constraint.name)

    # Add well locations
    for x, y, name in well_locations:
        ax.plot(x, y, 'ro', markersize=8)
        ax.annotate(name, (x, y), xytext=(5, 5), textcoords='offset points')

    ax.set_title("Generated Finite Element Mesh")
    ax.legend()
    fig.savefig("mesh_with_constraints.png", dpi=150, bbox_inches="tight")
    print("Saved mesh plot to mesh_with_constraints.png")

Step 9: Export to GIS
---------------------

Export the mesh to a GeoPackage for use in GIS software.

.. code-block:: python

    # Create GIS exporter with coordinate reference system
    exporter = GISExporter(
        grid=grid,
        crs="EPSG:2227",  # NAD83 / California Zone 3 (feet)
    )

    # Export to GeoPackage
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    exporter.export_geopackage(
        output_dir / "model_mesh.gpkg",
        include_boundary=True,
    )
    print(f"Exported mesh to {output_dir / 'model_mesh.gpkg'}")

    # Also export as shapefiles
    exporter.export_shapefiles(output_dir / "shapefiles")
    print(f"Exported shapefiles to {output_dir / 'shapefiles'}")

Step 10: Analyze Mesh Quality
-----------------------------

Check the quality of the generated mesh.

.. code-block:: python

    # Calculate element areas
    areas = []
    for elem in grid.iter_elements():
        coords = np.array([
            [grid.nodes[v].x, grid.nodes[v].y]
            for v in elem.vertices
        ])
        n = len(coords)
        area = 0.5 * abs(sum(
            coords[i, 0] * coords[(i + 1) % n, 1] -
            coords[(i + 1) % n, 0] * coords[i, 1]
            for i in range(n)
        ))
        areas.append(area)

    areas = np.array(areas)

    print("Mesh Quality Statistics:")
    print(f"  Total elements: {len(areas)}")
    print(f"  Minimum area: {areas.min():.1f} sq ft")
    print(f"  Maximum area: {areas.max():.1f} sq ft")
    print(f"  Mean area: {areas.mean():.1f} sq ft")
    print(f"  Median area: {np.median(areas):.1f} sq ft")
    print(f"  Area ratio (max/min): {areas.max() / areas.min():.1f}")

Complete Script
---------------

Here's the complete script combining all steps:

.. code-block:: python

    """Complete mesh generation example."""

    import numpy as np
    from pathlib import Path

    from pyiwfm.mesh_generation import TriangleMeshGenerator
    from pyiwfm.mesh_generation.constraints import (
        BoundaryConstraint, LineConstraint, PointConstraint, RefinementZone
    )
    from pyiwfm.visualization import GISExporter
    from pyiwfm.visualization.plotting import plot_mesh

    # 1. Define boundary
    boundary = BoundaryConstraint(coordinates=np.array([
        [0, 0], [5000, 0], [6000, 1000], [6000, 4000],
        [5000, 5000], [3000, 5500], [1000, 5000], [0, 3000]
    ]))

    # 2. Define streams
    river = LineConstraint(coordinates=np.array([
        [5500, 4500], [4000, 3500], [3000, 3000], [2000, 2000], [500, 1000]
    ]))

    # 3. Define wells
    wells = [PointConstraint(x=1500, y=3500), PointConstraint(x=3500, y=4000)]

    # 4. Define refinement
    refine = RefinementZone(
        coordinates=np.array([[1000, 3000], [4500, 3000], [4500, 4500], [1000, 4500]]),
        max_area=10000
    )

    # 5. Generate mesh
    generator = TriangleMeshGenerator()
    result = generator.generate(
        boundary=boundary,
        line_constraints=[river],
        point_constraints=wells,
        refinement_zones=[refine],
        max_area=50000,
        min_angle=25
    )

    # 6. Convert and export
    grid = generator.to_appgrid(result)
    exporter = GISExporter(grid=grid, crs="EPSG:2227")
    exporter.export_geopackage("model.gpkg")

    print(f"Generated {grid.n_nodes} nodes, {grid.n_elements} elements")

Next Steps
----------

- Add stratigraphy to create a complete 3D model
- Export to VTK for 3D visualization in ParaView
- Try the Gmsh generator for quadrilateral elements
