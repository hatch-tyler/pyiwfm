Tutorial: Reading an Existing IWFM Model
=========================================

This tutorial demonstrates how to load existing IWFM models with pyiwfm,
inspect their components, read simulation results, and create visualizations.

Learning Objectives
-------------------

By the end of this tutorial, you will be able to:

1. Load a complete IWFM model with a single function call
2. Inspect mesh, stratigraphy, and component data
3. Handle loading errors gracefully
4. Load individual components separately
5. Read simulation results (heads, budgets, hydrographs)
6. Preserve comments for roundtrip editing
7. Export loaded models to GIS and VTK formats

Overview
--------

pyiwfm provides three primary entry points for loading IWFM models:

- **``load_complete_model()``** — simplest, one-line loading
- **``CompleteModelLoader``** — more control, detailed error diagnostics
- **``IWFMModel.from_*()`` classmethods** — object-oriented API

Choose based on your needs: quick exploration, production workflows with
error handling, or integration into larger Python applications.

Quick Load
----------

The fastest way to load a model is the one-liner ``load_complete_model()``:

.. code-block:: python

   from pyiwfm.io import load_complete_model

   # Load from simulation main input file
   model = load_complete_model("Simulation/Simulation.in")

   # Print summary
   print(model.summary())

This reads the simulation main file, follows file references to the
preprocessor and all component files, and returns a fully populated
``IWFMModel`` object.

**Expected output:**

.. code-block:: text

   IWFM Model Summary
   ==================
   Nodes:         32537
   Elements:      31517
   Layers:        4
   Groundwater:   Yes
   Streams:       Yes
   Lakes:         Yes
   Root Zone:     Yes

Inspecting the Model
--------------------

Once loaded, access all model components through attributes:

.. code-block:: python

   # Mesh geometry
   print(f"Nodes:    {model.n_nodes}")
   print(f"Elements: {model.n_elements}")

   # Stratigraphy
   print(f"Layers:   {model.n_layers}")

   # Check which components are loaded
   components = {
       'Groundwater':      model.groundwater is not None,
       'Streams':          model.streams is not None,
       'Lakes':            model.lakes is not None,
       'Root Zone':        model.rootzone is not None,
       'Small Watersheds': model.small_watersheds is not None,
       'Unsaturated Zone': model.unsaturated_zone is not None,
   }
   for name, loaded in components.items():
       status = "Loaded" if loaded else "Not present"
       print(f"  {name}: {status}")

Visualize the mesh immediately after loading. Here we use a sample mesh to
demonstrate what the plot looks like:

.. plot::
   :include-source:

   import matplotlib.pyplot as plt
   from pyiwfm.sample_models import create_sample_mesh
   from pyiwfm.visualization.plotting import plot_mesh

   # In practice: plot_mesh(model.mesh, ...)
   mesh = create_sample_mesh(nx=15, ny=12, n_subregions=4)

   fig, ax = plot_mesh(mesh, show_edges=True, edge_color='gray',
                       fill_color='lightblue', alpha=0.3)
   ax.set_title(f'Loaded Model Mesh ({mesh.n_nodes} nodes, {mesh.n_elements} elements)')
   ax.set_xlabel('X (feet)')
   ax.set_ylabel('Y (feet)')
   plt.show()

Loading with Error Handling
---------------------------

For production code, use ``CompleteModelLoader`` which provides detailed
diagnostics via ``ModelLoadResult``:

.. code-block:: python

   from pyiwfm.io import CompleteModelLoader

   loader = CompleteModelLoader(
       simulation_file="Simulation/Simulation.in",
       preprocessor_file="Preprocessor/Preprocessor.in",
   )
   result = loader.load()

   # Check overall success
   print(f"Success: {result.success}")

   # Check for component-level errors
   if result.has_errors:
       print("Component errors:")
       for component, error_msg in result.errors.items():
           print(f"  {component}: {error_msg}")

   # Check warnings
   if result.warnings:
       print("Warnings:")
       for warning in result.warnings:
           print(f"  - {warning}")

   # Use the model (may have partial data if some components failed)
   if result.model is not None:
       model = result.model
       print(f"Loaded {model.n_nodes} nodes, {model.n_elements} elements")

The loader continues loading remaining components even if one fails,
storing errors in ``result.errors`` rather than raising exceptions.
This allows you to work with the parts of the model that loaded
successfully.

Loading Individual Components
-----------------------------

You can also load individual components directly using reader classes.
This is useful when you only need specific parts of a model:

.. code-block:: python

   from pyiwfm.io import read_nodes, read_elements, read_stratigraphy

   # Load just the mesh
   nodes = read_nodes("Preprocessor/Nodal.dat")
   elements = read_elements("Preprocessor/Element.dat")

   from pyiwfm.core.mesh import AppGrid
   grid = AppGrid(nodes=nodes, elements=elements)
   grid.compute_connectivity()

   print(f"Loaded mesh: {grid.n_nodes} nodes, {grid.n_elements} elements")

   # Load stratigraphy separately
   stratigraphy = read_stratigraphy("Preprocessor/Stratigraphy.dat")
   print(f"Stratigraphy: {stratigraphy.n_layers} layers")

Or load just the preprocessor portion (mesh + stratigraphy + stream/lake geometry):

.. code-block:: python

   from pyiwfm.core.model import IWFMModel

   model = IWFMModel.from_preprocessor("Preprocessor/Preprocessor.in")
   print(f"Loaded mesh and stratigraphy only")
   print(f"Nodes: {model.n_nodes}, Layers: {model.n_layers}")

Visualize the loaded mesh with a stream network overlay:

.. plot::
   :include-source:

   import matplotlib.pyplot as plt
   from pyiwfm.sample_models import create_sample_mesh, create_sample_stream_network
   from pyiwfm.visualization.plotting import plot_mesh

   # In practice: plot_mesh(model.mesh, ...) + plot_streams(model.streams, ...)
   mesh = create_sample_mesh(nx=15, ny=12, n_subregions=4)
   stream_nodes, reaches = create_sample_stream_network(mesh)

   fig, ax = plot_mesh(mesh, show_edges=True, edge_color='lightgray', alpha=0.2)

   # Overlay stream reaches
   for from_idx, to_idx in reaches:
       x1, y1 = stream_nodes[from_idx]
       x2, y2 = stream_nodes[to_idx]
       ax.plot([x1, x2], [y1, y2], 'b-', linewidth=2, zorder=3)
   sx, sy = zip(*stream_nodes)
   ax.scatter(sx, sy, c='blue', s=40, zorder=4, label='Stream Nodes')

   ax.set_title('Loaded Model with Streams')
   ax.set_xlabel('X (feet)')
   ax.set_ylabel('Y (feet)')
   ax.legend()
   plt.show()

Reading Simulation Results
--------------------------

After running an IWFM simulation, load results for visualization:

**Head data from output files:**

.. code-block:: python

   from pyiwfm.visualization.webapi.hydrograph_reader import read_hydrograph_file

   # Read head hydrograph output
   times, heads = read_hydrograph_file("Results/GW_Heads.out")
   print(f"Head data: {len(times)} timesteps")

**Budget data:**

.. code-block:: python

   from pyiwfm.io import BudgetReader

   reader = BudgetReader("Results/GW_Budget.hdf")
   budget_data = reader.read()
   print(f"Budget components: {list(budget_data.keys())}")

Visualize head distribution at a single timestep:

.. plot::
   :include-source:

   import matplotlib.pyplot as plt
   from pyiwfm.sample_models import create_sample_mesh, create_sample_scalar_field
   from pyiwfm.visualization.plotting import plot_scalar_field

   # In practice: plot_scalar_field(model.mesh, heads[-1, :], ...)
   mesh = create_sample_mesh(nx=15, ny=12, n_subregions=4)
   head = create_sample_scalar_field(mesh, field_type='head')

   fig, ax = plot_scalar_field(mesh, head, field_type='node', cmap='viridis',
                               show_mesh=True, edge_color='white')
   ax.set_title('Groundwater Head (Final Timestep)')
   ax.set_xlabel('X (feet)')
   ax.set_ylabel('Y (feet)')
   plt.show()

Plot head time series at selected nodes:

.. plot::
   :include-source:

   import matplotlib.pyplot as plt
   from pyiwfm.sample_models import create_sample_timeseries
   from pyiwfm.visualization.plotting import plot_timeseries

   # In practice: create TimeSeries from heads[:, node_id] arrays
   series_list = [
       create_sample_timeseries(name="Node 100", n_years=10, trend=-0.3, noise_level=0.05),
       create_sample_timeseries(name="Node 500", n_years=10, trend=-0.6, noise_level=0.08),
       create_sample_timeseries(name="Node 1000", n_years=10, trend=-0.4, noise_level=0.06),
   ]

   fig, ax = plot_timeseries(series_list, title='Head at Selected Nodes',
                              ylabel='Head (ft)')
   plt.show()

Comment-Preserving Load
-----------------------

When you need to modify an IWFM model and write it back with all original
comments intact, use the comment-preserving workflow:

.. code-block:: python

   from pyiwfm.io import load_model_with_comments

   # Load model AND comment metadata
   model, comments = load_model_with_comments("Simulation/Simulation.in")

   # Inspect preserved comments
   for file_key, metadata in comments.items():
       n_header = len(metadata.header_block) if metadata.header_block else 0
       n_sections = len(metadata.sections)
       print(f"  {file_key}: {n_header} header lines, {n_sections} sections")

   # Make modifications to the model...
   # model.groundwater.wells[5].pumping_rate *= 1.2

   # Write back with preserved comments
   from pyiwfm.io import save_complete_model
   save_complete_model(model, "output_modified/")

This ensures that user comments, header blocks, and inline descriptions
from the original files are retained in the output.

Visualizing the Loaded Model
-----------------------------

Quick visualization gallery using pyiwfm helpers. These examples use
sample data to show what each plot looks like:

**Mesh with element IDs:**

.. plot::
   :include-source:

   import matplotlib.pyplot as plt
   from pyiwfm.sample_models import create_sample_mesh
   from pyiwfm.visualization.plotting import plot_mesh

   mesh = create_sample_mesh(nx=6, ny=6, n_subregions=2)
   fig, ax = plot_mesh(mesh, show_element_ids=True, fill_color='lightyellow', alpha=0.5)
   ax.set_title('Mesh with Element IDs')
   ax.set_xlabel('X (feet)')
   ax.set_ylabel('Y (feet)')
   plt.show()

**Layer thickness as scalar field:**

.. plot::
   :include-source:

   import matplotlib.pyplot as plt
   import numpy as np
   from pyiwfm.sample_models import create_sample_mesh, create_sample_stratigraphy
   from pyiwfm.visualization.plotting import plot_scalar_field

   mesh = create_sample_mesh(nx=15, ny=12, n_subregions=4)
   strat = create_sample_stratigraphy(mesh, n_layers=3, surface_base=100.0,
                                       layer_thickness=50.0)

   thickness = strat.top_elev[:, 0] - strat.bottom_elev[:, 0]
   fig, ax = plot_scalar_field(mesh, thickness, field_type='node', cmap='YlOrRd')
   ax.set_title('Layer 1 Thickness (ft)')
   ax.set_xlabel('X (feet)')
   ax.set_ylabel('Y (feet)')
   plt.show()

**Budget bar chart:**

.. plot::
   :include-source:

   import matplotlib.pyplot as plt
   from pyiwfm.visualization.plotting import plot_budget_bar

   budget_components = {
       'Recharge': 15000, 'Pumping': -18500,
       'Stream Seepage': 8500, 'Baseflow': -7200,
       'Subsurface Inflow': 5200, 'GW ET': -3100,
   }
   fig, ax = plot_budget_bar(budget_components, title='GW Budget Summary',
                              units='AF/year')
   plt.show()

**Budget stacked over time:**

.. plot::
   :include-source:

   import matplotlib.pyplot as plt
   import numpy as np
   from pyiwfm.visualization.plotting import plot_budget_stacked

   np.random.seed(42)
   n_years = 10
   times = np.arange('2010-01-01', '2020-01-01', dtype='datetime64[Y]')
   components = {
       'Recharge': 15000 + np.random.normal(0, 1000, n_years),
       'Stream Seepage': 8500 + np.random.normal(0, 500, n_years),
       'Pumping': -(18500 + np.arange(n_years) * 200 + np.random.normal(0, 500, n_years)),
       'Baseflow': -(7200 + np.random.normal(0, 400, n_years)),
   }
   fig, ax = plot_budget_stacked(times, components,
                                  title='GW Budget Over Time', units='AF/year')
   plt.show()

For interactive visualization, launch the web viewer from the command line:

.. code-block:: bash

   pyiwfm viewer --model-dir /path/to/model/

Exporting the Loaded Model
--------------------------

Export to GIS formats for use in QGIS/ArcGIS:

.. code-block:: python

   from pyiwfm.visualization import GISExporter

   exporter = GISExporter(
       grid=model.mesh,
       stratigraphy=model.stratigraphy,
       crs="EPSG:26910",  # NAD83 / UTM zone 10N
   )

   # Export to GeoPackage (nodes + elements + boundary)
   exporter.export_geopackage("model_export.gpkg")
   print("Exported to GeoPackage")

   # Export to shapefiles
   exporter.export_shapefiles("shapefiles/")

Export to VTK for 3D visualization in ParaView:

.. code-block:: python

   from pyiwfm.visualization import VTKExporter

   vtk_exporter = VTKExporter(grid=model.mesh, stratigraphy=model.stratigraphy)

   # 2D surface mesh
   vtk_exporter.export_vtu("mesh_2d.vtu", mode="2d")

   # Full 3D volumetric mesh
   vtk_exporter.export_vtu("mesh_3d.vtu", mode="3d")

   # With scalar data attached
   vtk_exporter.export_vtu("mesh_with_heads.vtu", mode="3d",
                            node_scalars={"head": head_values})

Complete Script
---------------

Here is a complete example combining all the steps above:

.. code-block:: python

   """Load an existing IWFM model, inspect it, and create visualizations."""

   from pathlib import Path
   from pyiwfm.io import CompleteModelLoader
   from pyiwfm.visualization.plotting import (
       plot_mesh, plot_scalar_field, plot_streams,
       plot_budget_bar,
   )
   from pyiwfm.visualization import GISExporter
   import numpy as np

   # --- Load the model ---
   loader = CompleteModelLoader(
       simulation_file="Simulation/Simulation.in",
       preprocessor_file="Preprocessor/Preprocessor.in",
   )
   result = loader.load()

   if not result.success:
       print(f"Load failed: {result.errors}")
       raise SystemExit(1)

   model = result.model
   print(model.summary())

   # Print any warnings
   for w in result.warnings:
       print(f"Warning: {w}")

   # --- Visualize the mesh ---
   output = Path("output_plots")
   output.mkdir(exist_ok=True)

   fig, ax = plot_mesh(model.mesh, show_edges=True, edge_color='gray', alpha=0.3)
   ax.set_title(f'Model Mesh ({model.n_nodes} nodes)')
   fig.savefig(output / "mesh.png", dpi=150)

   # --- Plot layer thickness ---
   thickness = model.stratigraphy.top_elev[:, 0] - model.stratigraphy.bottom_elev[:, 0]
   fig, ax = plot_scalar_field(model.mesh, thickness, cmap='YlOrRd')
   ax.set_title('Layer 1 Thickness (ft)')
   fig.savefig(output / "layer1_thickness.png", dpi=150)

   # --- Plot streams ---
   if model.streams:
       fig, ax = plot_mesh(model.mesh, edge_color='lightgray', alpha=0.15)
       plot_streams(model.streams, ax=ax, line_width=2)
       ax.set_title('Stream Network')
       fig.savefig(output / "streams.png", dpi=150)

   # --- Export to GIS ---
   exporter = GISExporter(grid=model.mesh, crs="EPSG:26910")
   exporter.export_geopackage(output / "model.gpkg")

   print(f"All outputs saved to {output}/")

Next Steps
----------

- See :doc:`building_sample_model` for constructing a model from scratch
- See :doc:`visualization` for advanced plotting techniques
- See :doc:`/user_guide/io` for all supported file formats
- Launch the web viewer with ``pyiwfm viewer --model-dir /path/to/model/``
