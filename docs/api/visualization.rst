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
(Moved to :mod:`pyiwfm.io.head_loader`; see :doc:`io` for full docs.)

.. automodule:: pyiwfm.io.head_loader
   :members:
   :undoc-members:
   :show-inheritance:

Hydrograph Reader
~~~~~~~~~~~~~~~~~

Parser for IWFM ``.out`` text hydrograph files.
(Moved to :mod:`pyiwfm.io.hydrograph_reader`; see :doc:`io` for full docs.)

.. automodule:: pyiwfm.io.hydrograph_reader
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

API Routes
~~~~~~~~~~

The web viewer backend exposes a REST API under ``/api/``. Key route groups:

- **Model** (``/api/model``): Load model info, compare with a second model
- **Mesh** (``/api/mesh``): GeoJSON mesh, element details, node lookups
- **Results** (``/api/results``): Head values, drawdown with pagination (``offset``/``limit``/``skip``), head statistics (min/max/mean/std across timesteps)
- **Budgets** (``/api/budgets``): Water budget time series by type and location
- **Export** (``/api/export``): CSV downloads (heads, budgets, hydrographs), GeoJSON mesh, GeoPackage (multi-layer), and matplotlib plot generation (PNG/SVG for mesh, elements, streams, heads)
- **Observations** (``/api/observations``): Upload observed data for hydrograph comparison
- **Groundwater, Streams, Lakes, Root Zone** (``/api/groundwater``, ``/api/streams``, etc.): Component-specific endpoints
