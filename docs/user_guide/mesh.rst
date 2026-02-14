Working with Meshes
===================

This guide covers creating, manipulating, and analyzing finite element meshes.

Mesh Fundamentals
-----------------

An IWFM mesh consists of three main components:

- **Nodes**: Points with x, y coordinates
- **Elements**: Triangular (3 vertices) or quadrilateral (4 vertices) cells
- **Faces**: Edges shared between elements

Node Class
~~~~~~~~~~

.. code-block:: python

    from pyiwfm.core.mesh import Node

    # Create a node
    node = Node(
        id=1,
        x=100.0,
        y=200.0,
        is_boundary=True,  # Node on domain boundary
    )

    # Access properties
    print(f"Node {node.id} at ({node.x}, {node.y})")
    print(f"Is boundary: {node.is_boundary}")

Element Class
~~~~~~~~~~~~~

.. code-block:: python

    from pyiwfm.core.mesh import Element

    # Create a quadrilateral element
    quad = Element(
        id=1,
        vertices=(1, 2, 5, 4),  # Node IDs, counter-clockwise
        subregion=1,
    )

    print(f"Element {quad.id} is a quad: {quad.is_quad}")
    print(f"Number of vertices: {quad.n_vertices}")

    # Create a triangular element
    tri = Element(
        id=2,
        vertices=(2, 3, 5),  # 3 vertices for triangle
        subregion=1,
    )

    print(f"Element {tri.id} is a triangle: {tri.is_triangle}")

Creating a Mesh Manually
------------------------

Create a mesh by defining nodes and elements:

.. code-block:: python

    from pyiwfm.core.mesh import AppGrid, Node, Element

    # Define a 3x3 grid of nodes (9 nodes total)
    nodes = {}
    node_id = 1
    for j in range(3):
        for i in range(3):
            is_boundary = (i == 0 or i == 2 or j == 0 or j == 2)
            nodes[node_id] = Node(
                id=node_id,
                x=float(i * 100),
                y=float(j * 100),
                is_boundary=is_boundary,
            )
            node_id += 1

    # Define 4 quadrilateral elements
    elements = {
        1: Element(id=1, vertices=(1, 2, 5, 4), subregion=1),
        2: Element(id=2, vertices=(2, 3, 6, 5), subregion=1),
        3: Element(id=3, vertices=(4, 5, 8, 7), subregion=2),
        4: Element(id=4, vertices=(5, 6, 9, 8), subregion=2),
    }

    # Create the grid
    grid = AppGrid(nodes=nodes, elements=elements)

    # IMPORTANT: Compute connectivity after creating the grid
    grid.compute_connectivity()

    # Mesh statistics
    print(f"Nodes: {grid.n_nodes}")
    print(f"Elements: {grid.n_elements}")
    print(f"Boundary nodes: {grid.n_boundary_nodes}")

Mesh Connectivity
-----------------

After calling ``compute_connectivity()``, the mesh has additional information:

.. code-block:: python

    # Node connectivity
    for node in grid.iter_nodes():
        print(f"Node {node.id}:")
        print(f"  Connected nodes: {node.connected_nodes}")
        print(f"  Surrounding elements: {node.surrounding_elements}")

    # Element neighbors
    for elem in grid.iter_elements():
        print(f"Element {elem.id}:")
        print(f"  Vertices: {elem.vertices}")
        print(f"  Subregion: {elem.subregion}")

Iterating Over Mesh Components
------------------------------

.. code-block:: python

    # Iterate over nodes
    for node in grid.iter_nodes():
        print(f"Node {node.id}: ({node.x}, {node.y})")

    # Iterate over elements
    for elem in grid.iter_elements():
        print(f"Element {elem.id}: {elem.vertices}")

    # Iterate over boundary nodes only
    for node in grid.iter_boundary_nodes():
        print(f"Boundary node {node.id}")

    # Get specific node or element
    node = grid.nodes[5]
    elem = grid.elements[1]

Generating Meshes with Triangle
-------------------------------

For more complex geometries, use the Triangle mesh generator:

.. code-block:: python

    import numpy as np
    from pyiwfm.mesh_generation import TriangleMeshGenerator
    from pyiwfm.mesh_generation.constraints import (
        BoundaryConstraint,
        LineConstraint,
        PointConstraint,
        RefinementZone,
    )

    # Define boundary (L-shaped domain)
    boundary_coords = np.array([
        [0.0, 0.0],
        [1000.0, 0.0],
        [1000.0, 500.0],
        [500.0, 500.0],
        [500.0, 1000.0],
        [0.0, 1000.0],
    ])
    boundary = BoundaryConstraint(coordinates=boundary_coords)

    # Add a stream constraint (forces mesh edges along stream)
    stream_coords = np.array([
        [100.0, 0.0],
        [100.0, 500.0],
        [300.0, 800.0],
    ])
    stream = LineConstraint(coordinates=stream_coords)

    # Add point constraints (forces nodes at specific locations)
    wells = [
        PointConstraint(x=200.0, y=200.0),
        PointConstraint(x=400.0, y=300.0),
    ]

    # Add refinement zone (smaller elements in this area)
    refine_zone = RefinementZone(
        coordinates=np.array([
            [0.0, 0.0],
            [300.0, 0.0],
            [300.0, 300.0],
            [0.0, 300.0],
        ]),
        max_area=1000.0,  # Smaller elements in this zone
    )

    # Generate mesh
    generator = TriangleMeshGenerator()
    result = generator.generate(
        boundary=boundary,
        line_constraints=[stream],
        point_constraints=wells,
        refinement_zones=[refine_zone],
        max_area=5000.0,   # Maximum element area
        min_angle=25.0,    # Minimum angle (improves quality)
    )

    print(f"Generated {result.n_nodes} nodes, {result.n_elements} elements")

    # Convert to AppGrid
    grid = generator.to_appgrid(result)

Generating Meshes with Gmsh
---------------------------

Gmsh supports quad and mixed element meshes:

.. code-block:: python

    from pyiwfm.mesh_generation import GmshMeshGenerator

    # Create generator for quad elements
    generator = GmshMeshGenerator(element_type="quad")

    # Generate mesh
    result = generator.generate(
        boundary=boundary,
        line_constraints=[stream],
        max_area=5000.0,
    )

    # Convert to AppGrid
    grid = generator.to_appgrid(result)

    # Check element types
    n_quads = sum(1 for e in grid.iter_elements() if e.is_quad)
    n_tris = sum(1 for e in grid.iter_elements() if e.is_triangle)
    print(f"Quads: {n_quads}, Triangles: {n_tris}")

Mesh Quality
------------

Check mesh quality metrics:

.. code-block:: python

    import numpy as np

    # Calculate element areas
    areas = []
    for elem in grid.iter_elements():
        # Get vertex coordinates
        coords = np.array([
            [grid.nodes[v].x, grid.nodes[v].y]
            for v in elem.vertices
        ])

        # Calculate area using shoelace formula
        n = len(coords)
        area = 0.5 * abs(sum(
            coords[i, 0] * coords[(i + 1) % n, 1] -
            coords[(i + 1) % n, 0] * coords[i, 1]
            for i in range(n)
        ))
        areas.append(area)

    areas = np.array(areas)
    print(f"Min area: {areas.min():.2f}")
    print(f"Max area: {areas.max():.2f}")
    print(f"Mean area: {areas.mean():.2f}")
    print(f"Area ratio (max/min): {areas.max() / areas.min():.2f}")

Adding Holes to Meshes
----------------------

Create meshes with internal holes (e.g., for islands or exclusion zones):

.. code-block:: python

    # Define hole boundary
    hole_coords = np.array([
        [400.0, 400.0],
        [600.0, 400.0],
        [600.0, 600.0],
        [400.0, 600.0],
    ])

    # Generate mesh with hole
    generator = TriangleMeshGenerator()
    result = generator.generate(
        boundary=boundary,
        holes=[hole_coords],
        max_area=5000.0,
    )
