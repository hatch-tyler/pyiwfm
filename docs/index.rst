.. pyiwfm documentation master file

pyiwfm: Python Interface for IWFM
=================================

**pyiwfm** is a Python package for working with IWFM (Integrated Water Flow Model)
models developed by the California Department of Water Resources.

.. grid:: 2

    .. grid-item-card:: Getting Started
        :link: user_guide/installation
        :link-type: doc

        New to pyiwfm? Start here to learn how to install and use the package.

    .. grid-item-card:: User Guide
        :link: user_guide/index
        :link-type: doc

        Learn how to use pyiwfm's core features for reading, writing, and
        manipulating IWFM models.

    .. grid-item-card:: Tutorials
        :link: tutorials/index
        :link-type: doc

        Step-by-step tutorials for common workflows like mesh generation,
        visualization, and model comparison.

    .. grid-item-card:: API Reference
        :link: api/index
        :link-type: doc

        Complete API documentation for all modules, classes, and functions.

    .. grid-item-card:: Visualization Gallery
        :link: gallery/index
        :link-type: doc

        Browse examples showcasing pyiwfm's visualization capabilities
        for meshes, scalar fields, streams, and water budgets.

Features
--------

- **Read/Write IWFM Files**: Support for ASCII, binary, HDF5, and HEC-DSS formats
- **Mesh Generation**: Create finite element meshes using Triangle or Gmsh
- **GIS Export**: Export to GeoPackage, Shapefile, and GeoJSON formats
- **Interactive Web Viewer**: Browser-based 3D visualization with Trame and PyVista
- **Multi-Scale Viewing**: View data at element, subregion, or custom zone scales
- **Zone Editor**: Interactively create and edit spatial zones
- **PEST++ Calibration**: Complete interface for parameter estimation with PEST++
- **Ensemble Methods**: Prior/posterior ensemble generation for pestpp-ies
- **Subprocess Runner**: Run IWFM executables and manage scenarios
- **Plotting**: Matplotlib-based visualization of meshes and scalar fields
- **Model Comparison**: Compare models with metrics and generate reports

Quick Example
-------------

.. code-block:: python

    from pyiwfm.core.mesh import AppGrid, Node, Element
    from pyiwfm.visualization import GISExporter

    # Create a simple mesh
    nodes = {
        1: Node(id=1, x=0.0, y=0.0),
        2: Node(id=2, x=100.0, y=0.0),
        3: Node(id=3, x=50.0, y=100.0),
    }
    elements = {
        1: Element(id=1, vertices=(1, 2, 3), subregion=1),
    }
    grid = AppGrid(nodes=nodes, elements=elements)

    # Export to GeoPackage
    exporter = GISExporter(grid=grid, crs="EPSG:26910")
    exporter.export_geopackage("model.gpkg")

Installation
------------

Install pyiwfm using pip:

.. code-block:: bash

    pip install pyiwfm

For optional dependencies:

.. code-block:: bash

    # GIS support
    pip install pyiwfm[gis]

    # Mesh generation
    pip install pyiwfm[mesh]

    # Visualization
    pip install pyiwfm[viz]

    # All optional dependencies
    pip install pyiwfm[all]

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/index
   user_guide/installation
   user_guide/quickstart
   user_guide/mesh
   user_guide/stratigraphy
   user_guide/io
   user_guide/web_viewer
   user_guide/pest

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/index
   tutorials/mesh_generation
   tutorials/visualization
   tutorials/model_comparison

.. toctree::
   :maxdepth: 2
   :caption: Gallery

   gallery/index
   gallery/mesh_visualization
   gallery/scalar_fields
   gallery/stream_networks
   gallery/stratigraphy
   gallery/timeseries
   gallery/budget_plots

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index
   api/core
   api/components
   api/io
   api/mesh_generation
   api/visualization
   api/comparison
   api/runner

.. toctree::
   :maxdepth: 1
   :caption: Development

   changelog
   contributing

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
