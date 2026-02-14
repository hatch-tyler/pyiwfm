Installation
============

Requirements
------------

pyiwfm requires Python 3.10 or later.

Basic Installation
------------------

Install pyiwfm using pip:

.. code-block:: bash

    pip install pyiwfm

This installs the core package with required dependencies:

- numpy >= 1.21
- pandas >= 1.3
- h5py >= 3.0
- jinja2 >= 3.0

Optional Dependencies
---------------------

pyiwfm has several optional dependency groups for additional functionality:

GIS Support
~~~~~~~~~~~

For exporting to GIS formats (GeoPackage, Shapefile, GeoJSON):

.. code-block:: bash

    pip install pyiwfm[gis]

This installs:

- geopandas >= 0.10
- shapely >= 2.0

Mesh Generation
~~~~~~~~~~~~~~~

For creating finite element meshes:

.. code-block:: bash

    pip install pyiwfm[mesh]

This installs:

- triangle >= 20220202 (triangular meshes)
- gmsh >= 4.11 (triangular, quadrilateral, or mixed meshes)

Visualization
~~~~~~~~~~~~~

For 3D visualization and plotting:

.. code-block:: bash

    pip install pyiwfm[viz]

This installs:

- vtk >= 9.0 (3D export for ParaView)
- matplotlib >= 3.5 (2D plotting)

All Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~

To install all optional dependencies:

.. code-block:: bash

    pip install pyiwfm[all]

Development Installation
------------------------

For development, clone the repository and install in editable mode:

.. code-block:: bash

    git clone https://github.com/CADWRDeltaModeling/pyiwfm.git
    cd pyiwfm
    pip install -e ".[dev]"

The ``[dev]`` extra includes testing and documentation tools:

- pytest
- pytest-cov
- sphinx
- pydata-sphinx-theme

Verifying Installation
----------------------

Verify your installation by importing pyiwfm:

.. code-block:: python

    >>> import pyiwfm
    >>> print(pyiwfm.__version__)
    0.1.0

Check available modules:

.. code-block:: python

    >>> from pyiwfm.core.mesh import AppGrid, Node, Element
    >>> from pyiwfm.visualization import GISExporter

    # Check if optional modules are available
    >>> try:
    ...     from pyiwfm.mesh_generation import TriangleMeshGenerator
    ...     print("Triangle mesh generation available")
    ... except ImportError:
    ...     print("Install pyiwfm[mesh] for mesh generation")
