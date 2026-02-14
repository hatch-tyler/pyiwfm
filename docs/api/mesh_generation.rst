Mesh Generation
===============

The mesh generation modules provide tools for creating finite element meshes
from boundary polygons and constraint features.

Generators Module
-----------------

The generators module contains the abstract base class for mesh generators.

.. automodule:: pyiwfm.mesh_generation.generators
   :members:
   :undoc-members:
   :show-inheritance:

Constraints Module
------------------

The constraints module contains classes for representing mesh constraints
such as boundaries, internal features, and refinement zones.

.. automodule:: pyiwfm.mesh_generation.constraints
   :members:
   :undoc-members:
   :show-inheritance:

Triangle Wrapper
----------------

The Triangle wrapper provides an interface to the Triangle mesh generation library
for creating triangular meshes.

.. automodule:: pyiwfm.mesh_generation.triangle_wrapper
   :members:
   :undoc-members:
   :show-inheritance:

Gmsh Wrapper
------------

The Gmsh wrapper provides an interface to the Gmsh mesh generation library
for creating triangular, quadrilateral, or mixed meshes.

.. automodule:: pyiwfm.mesh_generation.gmsh_wrapper
   :members:
   :undoc-members:
   :show-inheritance:
