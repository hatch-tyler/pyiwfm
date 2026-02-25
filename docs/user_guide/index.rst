User Guide
==========

This user guide provides comprehensive documentation for using pyiwfm.

.. toctree::
   :maxdepth: 2

   installation
   quickstart
   mesh
   stratigraphy
   io
   calibration
   web_viewer
   pest

Overview
--------

pyiwfm is organized into several main modules:

**Core Modules** (``pyiwfm.core``)
    Fundamental data structures including meshes, stratigraphy, time series,
    zones, data aggregation, and a high-level query API.

**Component Modules** (``pyiwfm.components``)
    Model components like groundwater, streams, lakes, and root zone.

**I/O Modules** (``pyiwfm.io``)
    Reading and writing IWFM model files in ASCII, binary, HDF5, HEC-DSS, and
    zone definition formats.

**Mesh Generation** (``pyiwfm.mesh_generation``)
    Tools for creating finite element meshes from boundaries and constraints.

**Visualization** (``pyiwfm.visualization``)
    Static export (GIS, VTK, matplotlib) and interactive web-based 3D viewer
    with multi-scale viewing, zone editing, and property visualization.

**Runner** (``pyiwfm.runner``)
    Subprocess execution of IWFM executables, scenario management, and complete
    PEST++ calibration interface with parameter estimation, geostatistics,
    ensemble methods, and post-processing.

**Calibration** (``pyiwfm.calibration``)
    Observation well clustering (fuzzy c-means), time interpolation of
    simulated heads to observation times (IWFM2OBS), typical hydrograph
    computation (CalcTypHyd), and publication-quality calibration figures.

**Comparison** (``pyiwfm.comparison``)
    Tools for comparing models and generating reports.

Key Concepts
------------

Meshes
~~~~~~

IWFM uses an unstructured finite element mesh. The mesh consists of:

- **Nodes**: Points with x, y coordinates
- **Elements**: Triangular or quadrilateral cells defined by node vertices
- **Faces**: Edges shared between elements

Stratigraphy
~~~~~~~~~~~~

The stratigraphy defines the vertical layer structure:

- Ground surface elevation at each node
- Top and bottom elevations for each layer
- Active/inactive flags for each node-layer combination

Time Series
~~~~~~~~~~~

Time series data is used for:

- Boundary conditions (e.g., specified heads)
- Pumping rates
- Diversions
- Model outputs (heads, flows, budgets)
