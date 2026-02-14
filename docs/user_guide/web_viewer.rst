Interactive Web Viewer
=====================

pyiwfm includes a browser-based interactive 3D viewer for exploring IWFM
models at multiple spatial scales. The viewer is built on
`Trame <https://kitware.github.io/trame/>`_ and
`PyVista <https://docs.pyvista.org/>`_.

.. contents:: Table of Contents
   :local:
   :depth: 2

Quick Start
-----------

Launch the viewer from the command line:

.. code-block:: bash

    # Auto-detect model in current directory
    python run_iwfm_viewer.py

    # Specify model directory
    python run_iwfm_viewer.py --model-dir /path/to/model

    # Specify individual files
    python run_iwfm_viewer.py --preprocessor PreProcessor_MAIN.IN

    # Load with simulation results
    python run_iwfm_viewer.py --simulation Simulation_MAIN.IN --load-results

    # Configure port and theme
    python run_iwfm_viewer.py --port 8080 --theme dark

Or launch from Python:

.. code-block:: python

    from pyiwfm.core.model import IWFMModel
    from pyiwfm.visualization.web.app import IWFMWebViewer

    model = IWFMModel.from_preprocessor("PreProcessor_MAIN.IN")
    viewer = IWFMWebViewer(model)
    viewer.start()

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

Viewer Controls
---------------

The viewer provides controls in a side drawer:

**Layer Controls:**

- Layer slider to filter which layer is displayed
- Show all layers or a single layer

**View Scale:**

- **Element**: Color each element individually
- **Subregion**: Aggregate values to model subregions
- **Custom Zone**: Load custom zone definitions for aggregation

**Aggregation Method** (when scale is not element):

- Sum
- Mean
- Area-weighted mean (default)
- Min / Max
- Median

**Zone Controls:**

- Load zone file (.txt, .dat, .geojson)
- Click zones in the list to highlight
- Toggle zone boundary lines

**Display Controls:**

- Opacity slider
- Edge visibility toggle
- Stream network toggle
- Vertical exaggeration slider

**Cross-Section:**

- Enable/disable slice plane
- Select axis (X, Y, Z)
- Position slider

Multi-Scale Viewing
-------------------

The multi-scale system enables viewing model data at different spatial
resolutions. Data is aggregated from element-level values to zone-level
statistics using the ``DataAggregator``.

Architecture
~~~~~~~~~~~~

.. code-block:: text

    Element Data (mesh) --> ZoneDefinition (mapping) --> DataAggregator (compute)
      --> ModelQueryAPI (expose) --> MultiScaleVisualizer (render) --> Web UI

Subregion View
~~~~~~~~~~~~~~

Subregions are predefined groupings from IWFM model input files. When
switching to "subregion" scale, elements are colored by their subregion
assignment and values are aggregated per subregion.

Custom Zone View
~~~~~~~~~~~~~~~~

Load zone definition files in IWFM ZBudget or GeoJSON format:

.. code-block:: python

    from pyiwfm.io.zones import read_zone_file

    zones = read_zone_file("my_zones.txt")

IWFM zone file format:

.. code-block:: text

    C Zone definitions for water budget analysis
    1                           # ZExtent: 1=horizontal
    1  Sacramento Valley        # Zone ID and name
    2  San Joaquin Valley
    3  Tulare Basin
    /                           # Separator
    1    1                      # Element 1 in Zone 1
    2    1                      # Element 2 in Zone 1
    3    2                      # Element 3 in Zone 2

Zone Editor
-----------

The interactive zone editor allows creating custom zones directly in the
viewer by selecting elements.

**Selection Modes:**

- **Single click**: Toggle individual elements
- **Box select**: Select elements in a rectangular region
- **Paint**: Select elements by dragging

**Operations:**

- Create zone from selection
- Add selected elements to an existing zone
- Remove elements from a zone
- Delete or rename zones
- Undo/redo (20-level stack)

**Persistence:**

Zones can be saved to and loaded from files:

.. code-block:: python

    from pyiwfm.visualization.web.widgets.zone_editor import ZoneEditor
    from pyiwfm.core.zones import ZoneDefinition

    editor = ZoneEditor(grid=model.mesh)
    editor.start_edit_mode()

    # Programmatic element selection
    editor.toggle_element_selection(element_id=42)
    editor.select_adjacent(element_id=42)

    # Create zone from selection
    zone_id = editor.create_zone_from_selection(name="Custom Area")

    # Save zones
    editor.save_zones("custom_zones.geojson")

Data Query and Export
---------------------

The ``ModelQueryAPI`` provides programmatic access to model data at any
spatial scale:

.. code-block:: python

    from pyiwfm.core.query import ModelQueryAPI

    api = ModelQueryAPI(model=model)

    # Get values at different scales
    element_values = api.get_values("kh", scale="element")
    subregion_values = api.get_values("kh", scale="subregion",
                                       aggregation="area_weighted_mean")

    # Register custom zones and query
    api.register_zones("basin", zone_definition)
    zone_values = api.get_values("kh", scale="basin")

    # Export to DataFrame or CSV
    df = api.export_to_dataframe(["kh", "sy"], scale="subregion")
    api.export_to_csv(["kh", "sy"], "output.csv", scale="element")

    # Time series for a specific zone
    ts = api.get_timeseries("head", zone_id=1)

Available Variables
~~~~~~~~~~~~~~~~~~~

The query API dynamically discovers available properties:

- ``kh`` - Horizontal hydraulic conductivity
- ``kv`` - Vertical hydraulic conductivity
- ``ss`` - Specific storage
- ``sy`` - Specific yield
- ``head`` - Groundwater head (if results loaded)
- ``thickness`` - Layer thickness
- ``top_elevation`` - Layer top elevation
- ``bottom_elevation`` - Layer bottom elevation
- ``area`` - Element area
- ``subregion`` - Subregion assignment

Property Visualization
----------------------

The ``PropertyVisualizer`` manages display of model properties with
appropriate colormaps:

- ``kh``, ``kv``: "turbo" colormap (log scale)
- ``ss``, ``sy``: "plasma" colormap
- ``head``: "coolwarm" colormap
- ``thickness``, elevations: "terrain" or "viridis"

Properties are cached and support layer filtering (view a single layer
or all layers simultaneously).

Performance
-----------

The viewer includes optimizations for large models:

- **Level of Detail**: Automatic mesh decimation for models with >10,000 nodes
- **Surface extraction**: Render only the surface for very large 3D meshes
- **Caching**: Property arrays and zone boundary meshes are computed once
- **Vectorized aggregation**: NumPy-based aggregation for sub-100ms response

For C2VSimFG-scale models (~130,000 cells), the viewer remains interactive
with these optimizations.
