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

The web visualization modules provide an interactive 3D model viewer built on
Trame and PyVista, with multi-scale viewing, zone editing, and property
visualization.

Web Application
~~~~~~~~~~~~~~~

The main web application class that creates the Trame-based interactive viewer.

.. automodule:: pyiwfm.visualization.web.app
   :members:
   :undoc-members:
   :show-inheritance:

Mesh Viewer
~~~~~~~~~~~

PyVista mesh creation, rendering, and interaction management.

.. automodule:: pyiwfm.visualization.web.viewer
   :members:
   :undoc-members:
   :show-inheritance:

Property Visualizer
~~~~~~~~~~~~~~~~~~~

Property display with colormaps, layer filtering, and value range management.

.. automodule:: pyiwfm.visualization.web.properties
   :members:
   :undoc-members:
   :show-inheritance:

Multi-Scale Visualizer
~~~~~~~~~~~~~~~~~~~~~~

Scale-aware rendering that supports element, subregion, and custom zone views
with spatial aggregation.

.. automodule:: pyiwfm.visualization.web.multi_scale
   :members:
   :undoc-members:
   :show-inheritance:

Zone Editor Widget
~~~~~~~~~~~~~~~~~~

Interactive zone creation and editing with element selection and undo/redo.

.. automodule:: pyiwfm.visualization.web.widgets.zone_editor
   :members:
   :undoc-members:
   :show-inheritance:
