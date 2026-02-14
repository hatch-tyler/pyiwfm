Visualization
=============

The visualization modules provide tools for exporting model data to various
formats for visualization in external tools.

GIS Export
----------

The GIS export module provides functionality for exporting model data to
GIS formats including GeoPackage, Shapefile, and GeoJSON.

.. automodule:: pyiwfm.visualization.gis_export
   :members:
   :undoc-members:
   :show-inheritance:

VTK Export
----------

The VTK export module provides functionality for exporting 3D model data
to VTK formats for visualization in ParaView.

.. automodule:: pyiwfm.visualization.vtk_export
   :members:
   :undoc-members:
   :show-inheritance:

Plotting
--------

The plotting module provides matplotlib-based visualization tools for
creating 2D plots of model meshes and data.

.. automodule:: pyiwfm.visualization.plotting
   :members:
   :undoc-members:
   :show-inheritance:

Interactive Web Viewer
----------------------

The web viewer is a FastAPI backend + React SPA frontend with four tabs:
Overview, 3D Mesh (vtk.js), Results Map (deck.gl + MapLibre), and Budgets
(Plotly).

Configuration
~~~~~~~~~~~~~

The ``ModelState`` singleton that holds the loaded ``IWFMModel`` and provides
lazy getters for head data, budget data, stream reach boundaries, and
coordinate reprojection.

.. automodule:: pyiwfm.visualization.webapi.config
   :members:
   :undoc-members:
   :show-inheritance:

Server
~~~~~~

FastAPI application creation, CRS configuration, and static file serving.

.. automodule:: pyiwfm.visualization.webapi.server
   :members:
   :undoc-members:
   :show-inheritance:

Head Data Loader
~~~~~~~~~~~~~~~~

Lazy HDF5 head results reader with per-frame caching.

.. automodule:: pyiwfm.visualization.webapi.head_loader
   :members:
   :undoc-members:
   :show-inheritance:

Hydrograph Reader
~~~~~~~~~~~~~~~~~

Parser for IWFM ``.out`` text hydrograph files.

.. automodule:: pyiwfm.visualization.webapi.hydrograph_reader
   :members:
   :undoc-members:
   :show-inheritance:

Properties
~~~~~~~~~~

Property extraction and caching for the web viewer.

.. automodule:: pyiwfm.visualization.webapi.properties
   :members:
   :undoc-members:
   :show-inheritance:

Slicing
~~~~~~~

Cross-section slice plane computation.

.. automodule:: pyiwfm.visualization.webapi.slicing
   :members:
   :undoc-members:
   :show-inheritance:
