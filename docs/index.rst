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

.. grid:: 3

    .. grid-item-card:: Drawdown Analysis
        :link: tutorials/drawdown_analysis
        :link-type: doc

        Compute and visualize groundwater drawdown relative to a
        reference timestep with diverging colormaps.

    .. grid-item-card:: Simulation Diagnostics
        :link: tutorials/simulation_diagnostics
        :link-type: doc

        Parse simulation logs to track convergence, mass balance errors,
        and identify problem areas.

    .. grid-item-card:: Stream Depletion
        :link: tutorials/stream_depletion
        :link-type: doc

        Compare baseline and pumping scenarios to quantify impact on
        stream flows.

Features
--------

- **Read/Write IWFM Files**: Support for ASCII, binary, HDF5, and HEC-DSS formats
- **Mesh Generation**: Create finite element meshes using Triangle or Gmsh
- **GIS Export**: Export to GeoPackage, Shapefile, and GeoJSON formats
- **Interactive Web Viewer**: Browser-based visualization with FastAPI, React, vtk.js, and deck.gl — includes data export (CSV, GeoPackage, matplotlib plots), model comparison, drawdown animation, and head statistics
- **Calibration Tools**: SMP observation file I/O, IWFM2OBS time interpolation with automatic model file discovery, multi-layer T-weighted observation well processing, fuzzy c-means well clustering, typical hydrograph computation (CalcTypHyd), SimulationMessages.out parser, and publication-quality calibration figures
- **PEST++ Integration**: Complete interface for parameter estimation with PEST++
- **Ensemble Methods**: Prior/posterior ensemble generation for pestpp-ies
- **Subprocess Runner**: Run IWFM executables and manage scenarios
- **Plotting**: Matplotlib-based visualization of meshes and scalar fields, including server-side plot generation via the web API
- **Model Comparison**: Compare models with metrics and generate reports, accessible via web viewer endpoint
- **Consistent Component Interface**: BaseComponent ABC provides ``validate()`` and ``n_items`` across all 6 model components
- **Drawdown Analysis**: Compute drawdown relative to reference timesteps with per-node, per-element, max-map, and robust range calculations
- **Stream Depletion**: Compare baseline and pumping model runs to quantify stream flow depletion at individual reaches
- **Budget Checks**: Mass balance sanity checks to detect timesteps with inflow/outflow/storage imbalance
- **Mesh Quality**: Element-level quality metrics (aspect ratio, skewness, min/max angle) with aggregate statistics
- **Simulation Diagnostics**: Parse SimulationMessages.out for structured messages, convergence tracking, mass balance errors, and spatial hotspots
- **PEST++ CLI**: ``pyiwfm pest setup/run/analyze`` commands for end-to-end calibration from the command line

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

    # Mesh generation (triangle, gmsh)
    pip install pyiwfm[mesh]

    # VTK 3D export
    pip install pyiwfm[viz]

    # Web viewer (FastAPI + vtk.js + deck.gl)
    pip install pyiwfm[webapi]

    # PEST++ integration (scipy)
    pip install pyiwfm[pest]

    # Development (pytest, mypy, ruff)
    pip install -e ".[dev]"

    # All optional dependencies
    pip install pyiwfm[all]

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/index

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: Gallery

   gallery/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 1
   :caption: Development

   changelog
   MIGRATION_v1_to_v2
   contributing

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
