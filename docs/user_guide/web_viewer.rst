Interactive Web Viewer
======================

pyiwfm includes a browser-based interactive viewer for exploring IWFM
models. The viewer is built with `FastAPI <https://fastapi.tiangolo.com/>`_
on the backend and `React <https://react.dev/>`_ +
`vtk.js <https://kitware.github.io/vtk-js/>`_ +
`deck.gl <https://deck.gl/>`_ on the frontend.

.. contents:: Table of Contents
   :local:
   :depth: 2

Quick Start
-----------

Launch the viewer from the command line:

.. code-block:: bash

    # Auto-detect model in current directory
    pyiwfm viewer

    # Specify model directory
    pyiwfm viewer --model-dir /path/to/model

    # Specify CRS for coordinate reprojection (optional)
    pyiwfm viewer --model-dir /path/to/model --crs "+proj=utm +zone=10 +datum=NAD83 +units=us-ft +no_defs"

    # Configure port (default: 8080)
    pyiwfm viewer --model-dir /path/to/model --port 9000

Or launch from Python:

.. code-block:: python

    from pyiwfm.visualization.webapi.server import create_app
    import uvicorn

    app = create_app(model_dir="/path/to/model")
    uvicorn.run(app, host="0.0.0.0", port=8080)

Model Auto-Detection
--------------------

The launcher automatically searches for IWFM input files:

1. Looks for ``Preprocessor/*.in`` or ``PreProcessor_MAIN.IN``
2. Looks for ``Simulation/*.in`` or ``Simulation_MAIN.IN``
3. Extracts model name from input files
4. Loads stream specification files if available

Missing components are handled gracefully:

- No stratigraphy: 2D mesh only
- No streams: stream layer omitted
- No simulation results: static property display

Viewer Tabs
-----------

The viewer provides four tabs, each focused on a different aspect of the model.

Overview
~~~~~~~~

Model summary and metadata including node/element counts, component
availability, simulation time range, and coordinate reference system.

3D Mesh
~~~~~~~

Interactive 3D rendering of the model mesh using vtk.js.

**Controls:**

- Layer slider to filter which layer is displayed
- Show all layers or a single layer
- Opacity slider
- Edge visibility toggle
- Stream network overlay
- Vertical exaggeration slider

**Cross-Section:**

- Enable/disable slice plane
- Select axis (X, Y, Z)
- Position slider

Results Map
~~~~~~~~~~~

2D map view using deck.gl and MapLibre GL for head contour visualization.

**Features:**

- Head values displayed as color-coded elements
- Timestep and layer selection
- Head difference (change) between two timesteps
- Hydrograph locations displayed as markers
- Click a hydrograph marker to view time series chart
- Upload observed data (CSV) for comparison overlay
- Stream network overlay on map

Budgets
~~~~~~~

Plotly-based charts for water budget time series.

**Features:**

- Groundwater, stream, lake, root zone, and other budget types
- Location and column selection
- Monthly budget timestep support

Docker Deployment
-----------------

The viewer can be deployed using Docker:

.. code-block:: bash

    # Build and run with Docker
    docker run -p 8080:8080 -v /path/to/model:/model pyiwfm

    # Or use docker-compose
    docker-compose up --build

See ``DOCKER.md`` for full configuration including environment variables
(``PORT``, ``TITLE``, ``MODE``, ``MODEL_PATH``).

Coordinate Reprojection
-----------------------

The backend reprojects model coordinates to WGS84 for map display using
``pyproj``. Use the ``--crs`` flag to specify the model's coordinate
reference system:

.. code-block:: bash

    pyiwfm viewer --crs "+proj=utm +zone=10 +datum=NAD83 +units=us-ft +no_defs"

For C2VSimFG models, the CRS defaults to UTM Zone 10N, NAD83, US survey feet.

Performance
-----------

The viewer includes optimizations for large models:

- **Lazy loading**: Head data loaded on demand from HDF5
- **Caching**: Property arrays and reprojected coordinates computed once
- **Vectorized computation**: NumPy-based operations for sub-100ms response
- **Pre-built frontend**: Static React SPA served directly by FastAPI

For C2VSimFG-scale models (~130,000 cells), the viewer remains interactive
with these optimizations.

Frontend Development
--------------------

The React frontend source is in ``frontend/``. To rebuild:

.. code-block:: bash

    cd frontend
    npm install
    npm run build

This compiles TypeScript and builds the SPA to
``src/pyiwfm/visualization/webapi/static/``.

For development with hot reload:

.. code-block:: bash

    cd frontend
    npm run dev

The Vite dev server proxies ``/api`` requests to ``http://localhost:8080``,
so the FastAPI backend must be running separately.
