Tutorial: Reading an Existing IWFM Model
=========================================

This tutorial demonstrates how to load an existing IWFM model with pyiwfm,
inspect its components, and create visualizations.  All examples use the
**C2VSimCG** (California Central Valley Simulation -- Coarse Grid) model, a
real-world IWFM model maintained by the California Department of Water
Resources.

.. note::

   The figures in this tutorial were pre-generated from C2VSimCG and checked
   into the repository so the documentation builds without requiring access
   to the model files.  To regenerate them, run::

       python docs/scripts/generate_tutorial_figures.py /path/to/C2VSimCG

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

- **``load_complete_model()``** -- simplest, one-line loading
- **``CompleteModelLoader``** -- more control, detailed error diagnostics
- **``IWFMModel.from_*()`` classmethods** -- object-oriented API

Choose based on your needs: quick exploration, production workflows with
error handling, or integration into larger Python applications.

Quick Load
----------

The fastest way to load a model is the one-liner ``load_complete_model()``:

.. code-block:: python

   from pyiwfm.io import load_complete_model

   model = load_complete_model("C2VSimCG/Simulation/Simulation.in")

   # Print summary
   print(model.summary())

This reads the simulation main file, follows file references to the
preprocessor and all component files, and returns a fully populated
``IWFMModel`` object.

**Expected output (C2VSimCG):**

.. code-block:: text

   IWFM Model: C2VSimCG
   =====================

   Mesh & Stratigraphy:
     Nodes: 1393
     Elements: 1392
     Layers: 4
     Subregions: 21

   Groundwater Component:
     Wells: ...
     Hydrograph Locations: ...
     Boundary Conditions: ...
     Tile Drains: ...
     Aquifer Parameters: Loaded

   Stream Component:
     Stream Nodes: ...
     Reaches: ...
     Diversions: ...
     Bypasses: ...

   Lake Component:
     Lakes: ...
     Lake Elements: ...

   Root Zone Component:
     Crop Types: ...

Inspecting the Model
--------------------

Once loaded, access all model components through attributes:

.. code-block:: python

   # Mesh geometry
   print(f"Nodes:    {model.n_nodes}")       # 1393
   print(f"Elements: {model.n_elements}")     # 1392

   # Stratigraphy
   print(f"Layers:   {model.n_layers}")       # 4

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

Visualize the mesh immediately after loading:

.. code-block:: python

   from pyiwfm.visualization.plotting import plot_mesh

   fig, ax = plot_mesh(model.mesh, show_edges=True, edge_color='gray',
                       fill_color='lightblue', alpha=0.3, figsize=(10, 10))
   ax.set_title(f'C2VSimCG Mesh ({model.n_nodes} nodes, {model.n_elements} elements)')
   ax.set_xlabel('Easting (ft)')
   ax.set_ylabel('Northing (ft)')

.. image:: /_static/tutorials/reading_models/mesh.png
   :alt: C2VSimCG finite element mesh
   :width: 100%

Color elements by subregion to see how the Central Valley is partitioned:

.. code-block:: python

   from pyiwfm.visualization.plotting import plot_elements

   fig, ax = plot_elements(model.mesh, color_by='subregion',
                           cmap='Set3', alpha=0.7, figsize=(10, 10))
   ax.set_title(f'C2VSimCG Subregions ({model.mesh.n_subregions} subregions)')

.. image:: /_static/tutorials/reading_models/subregions.png
   :alt: C2VSimCG elements colored by subregion
   :width: 100%

Stream Network
--------------

Access the stream component to inspect reaches, stream nodes, diversions,
and bypasses:

.. code-block:: python

   streams = model.streams
   print(f"Stream Nodes: {streams.n_nodes}")
   print(f"Reaches:      {streams.n_reaches}")
   print(f"Diversions:   {streams.n_diversions}")
   print(f"Bypasses:     {streams.n_bypasses}")

Overlay the stream network on the mesh:

.. code-block:: python

   from pyiwfm.visualization.plotting import plot_mesh, plot_streams

   fig, ax = plot_mesh(model.mesh, show_edges=True, edge_color='lightgray',
                       fill_color='white', alpha=0.15, figsize=(10, 10))
   plot_streams(model.streams, ax=ax, line_color='blue', line_width=1.5)
   ax.set_title('C2VSimCG Stream Network')

.. image:: /_static/tutorials/reading_models/streams.png
   :alt: C2VSimCG stream network overlaid on mesh
   :width: 100%

Lakes
-----

Access the lake component:

.. code-block:: python

   lakes = model.lakes
   print(f"Number of lakes: {lakes.n_lakes}")
   print(f"Lake elements:   {lakes.n_lake_elements}")

Visualize lake boundaries on the mesh:

.. code-block:: python

   from pyiwfm.visualization.plotting import plot_lakes, plot_mesh

   fig, ax = plot_mesh(model.mesh, show_edges=True, edge_color='lightgray',
                       fill_color='white', alpha=0.15, figsize=(10, 10))
   plot_lakes(model.lakes, model.mesh, ax=ax, fill_color='cyan',
              edge_color='blue', alpha=0.5)
   ax.set_title('C2VSimCG Lakes')

.. image:: /_static/tutorials/reading_models/lakes.png
   :alt: C2VSimCG lake boundaries on mesh
   :width: 100%

Stratigraphy and Ground Surface
-------------------------------

The stratigraphy contains ground surface elevation and layer top/bottom
elevations at every node:

.. code-block:: python

   strat = model.stratigraphy
   print(f"Layers:          {strat.n_layers}")
   print(f"Nodes:           {strat.n_nodes}")
   print(f"GS elev range:   {strat.gs_elev.min():.0f} - {strat.gs_elev.max():.0f} ft")

Plot the ground surface elevation as a scalar field:

.. code-block:: python

   from pyiwfm.visualization.plotting import plot_scalar_field

   fig, ax = plot_scalar_field(model.mesh, strat.gs_elev,
                               field_type='node', cmap='terrain',
                               show_mesh=False, figsize=(10, 10))
   ax.set_title('C2VSimCG Ground Surface Elevation (ft)')

.. image:: /_static/tutorials/reading_models/ground_surface.png
   :alt: C2VSimCG ground surface elevation
   :width: 100%

Compute and plot Layer 1 thickness:

.. code-block:: python

   thickness = strat.get_layer_thickness(0)  # Layer 1 (0-indexed)
   fig, ax = plot_scalar_field(model.mesh, thickness,
                               field_type='node', cmap='YlOrRd',
                               show_mesh=False, figsize=(10, 10))
   ax.set_title('C2VSimCG Layer 1 Thickness (ft)')

.. image:: /_static/tutorials/reading_models/layer_thickness.png
   :alt: C2VSimCG Layer 1 thickness
   :width: 100%

Extract and plot a vertical cross-section that cuts east-west through the
center of the model domain:

.. code-block:: python

   from pyiwfm.core.cross_section import CrossSectionExtractor
   from pyiwfm.visualization.plotting import plot_cross_section

   extractor = CrossSectionExtractor(model.mesh, model.stratigraphy)
   xs = extractor.extract(
       start=(x_min, y_mid),   # western edge, midpoint latitude
       end=(x_max, y_mid),     # eastern edge
       n_samples=200,
   )
   fig, ax = plot_cross_section(xs, title='C2VSimCG East-West Cross-Section')

.. image:: /_static/tutorials/reading_models/cross_section.png
   :alt: C2VSimCG east-west stratigraphic cross-section
   :width: 100%

Loading with Error Handling
---------------------------

For production code, use ``CompleteModelLoader`` which provides detailed
diagnostics via ``ModelLoadResult``:

.. code-block:: python

   from pyiwfm.io import CompleteModelLoader

   loader = CompleteModelLoader(
       simulation_file="C2VSimCG/Simulation/Simulation.in",
       preprocessor_file="C2VSimCG/Preprocessor/Preprocessor.in",
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
   nodes = read_nodes("C2VSimCG/Preprocessor/Nodal.dat")
   elements = read_elements("C2VSimCG/Preprocessor/Element.dat")

   from pyiwfm.core.mesh import AppGrid
   grid = AppGrid(nodes=nodes, elements=elements)
   grid.compute_connectivity()

   print(f"Loaded mesh: {grid.n_nodes} nodes, {grid.n_elements} elements")

   # Load stratigraphy separately
   stratigraphy = read_stratigraphy("C2VSimCG/Preprocessor/Stratigraphy.dat")
   print(f"Stratigraphy: {stratigraphy.n_layers} layers")

Or load just the preprocessor portion (mesh + stratigraphy + stream/lake geometry):

.. code-block:: python

   from pyiwfm.core.model import IWFMModel

   model = IWFMModel.from_preprocessor("C2VSimCG/Preprocessor/Preprocessor.in")
   print(f"Loaded mesh and stratigraphy only")
   print(f"Nodes: {model.n_nodes}, Layers: {model.n_layers}")

Reading Simulation Results
--------------------------

After running an IWFM simulation, load results for visualization:

**Head data from output files:**

.. code-block:: python

   from pyiwfm.visualization.webapi.hydrograph_reader import read_hydrograph_file

   # Read head hydrograph output
   times, heads = read_hydrograph_file("C2VSimCG/Results/GW_Heads.out")
   print(f"Head data: {len(times)} timesteps")

**Budget data:**

.. code-block:: python

   from pyiwfm.io import BudgetReader

   reader = BudgetReader("C2VSimCG/Results/GW_Budget.hdf")
   budget_data = reader.read()
   print(f"Budget components: {list(budget_data.keys())}")

The following figures show what C2VSimCG-scale budget plots look like
(illustrative values -- actual values depend on your simulation results):

**Budget bar chart:**

.. code-block:: python

   from pyiwfm.visualization.plotting import plot_budget_bar

   budget_components = {
       'Deep Percolation': 5_800_000,
       'Stream Seepage': 2_400_000,
       'Subsurface Inflow': 800_000,
       'Pumping': -7_500_000,
       'Outflow to Streams': -1_200_000,
       'Subsurface Outflow': -300_000,
   }
   fig, ax = plot_budget_bar(budget_components,
                              title='C2VSimCG Groundwater Budget',
                              units='AF/year')

.. image:: /_static/tutorials/reading_models/budget_bar.png
   :alt: C2VSimCG groundwater budget bar chart
   :width: 100%

**Budget stacked over time:**

.. code-block:: python

   import numpy as np
   from pyiwfm.visualization.plotting import plot_budget_stacked

   times = np.arange('2005-01-01', '2015-01-01', dtype='datetime64[Y]')
   components = {
       'Deep Percolation': 5_800_000 + np.random.normal(0, 400_000, 10),
       'Stream Seepage': 2_400_000 + np.random.normal(0, 200_000, 10),
       'Pumping': -(7_500_000 + np.arange(10) * 50_000),
       'Outflow to Streams': -(1_200_000 + np.random.normal(0, 100_000, 10)),
   }
   fig, ax = plot_budget_stacked(times, components,
                                  title='C2VSimCG GW Budget Over Time',
                                  units='AF/year')

.. image:: /_static/tutorials/reading_models/budget_stacked.png
   :alt: C2VSimCG groundwater budget stacked over time
   :width: 100%

Comment-Preserving Load
-----------------------

When you need to modify an IWFM model and write it back with all original
comments intact, use the comment-preserving workflow:

.. code-block:: python

   from pyiwfm.io import load_model_with_comments

   # Load model AND comment metadata
   model, comments = load_model_with_comments("C2VSimCG/Simulation/Simulation.in")

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

For interactive visualization, launch the web viewer from the command line:

.. code-block:: bash

   pyiwfm viewer --model-dir C2VSimCG/

Complete Script
---------------

Here is a complete example combining all the steps above:

.. code-block:: python

   """Load C2VSimCG, inspect it, and create visualizations."""

   from pathlib import Path
   from pyiwfm.io import CompleteModelLoader
   from pyiwfm.core.cross_section import CrossSectionExtractor
   from pyiwfm.visualization.plotting import (
       plot_mesh, plot_elements, plot_scalar_field,
       plot_streams, plot_lakes, plot_cross_section,
       plot_budget_bar,
   )
   from pyiwfm.visualization import GISExporter
   import numpy as np

   # --- Load the model ---
   loader = CompleteModelLoader(
       simulation_file="C2VSimCG/Simulation/Simulation.in",
       preprocessor_file="C2VSimCG/Preprocessor/Preprocessor.in",
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

   # --- Output directory ---
   output = Path("output_plots")
   output.mkdir(exist_ok=True)

   # --- Mesh ---
   fig, ax = plot_mesh(model.mesh, show_edges=True, edge_color='gray', alpha=0.3)
   ax.set_title(f'C2VSimCG Mesh ({model.n_nodes} nodes)')
   fig.savefig(output / "mesh.png", dpi=150)

   # --- Subregions ---
   fig, ax = plot_elements(model.mesh, color_by='subregion', cmap='Set3')
   ax.set_title(f'Subregions ({model.mesh.n_subregions})')
   fig.savefig(output / "subregions.png", dpi=150)

   # --- Ground surface elevation ---
   fig, ax = plot_scalar_field(model.mesh, model.stratigraphy.gs_elev,
                               cmap='terrain', show_mesh=False)
   ax.set_title('Ground Surface Elevation (ft)')
   fig.savefig(output / "ground_surface.png", dpi=150)

   # --- Layer 1 thickness ---
   thickness = model.stratigraphy.get_layer_thickness(0)
   fig, ax = plot_scalar_field(model.mesh, thickness, cmap='YlOrRd')
   ax.set_title('Layer 1 Thickness (ft)')
   fig.savefig(output / "layer1_thickness.png", dpi=150)

   # --- Streams ---
   if model.streams:
       fig, ax = plot_mesh(model.mesh, edge_color='lightgray', alpha=0.15)
       plot_streams(model.streams, ax=ax, line_width=1.5)
       ax.set_title('Stream Network')
       fig.savefig(output / "streams.png", dpi=150)

   # --- Lakes ---
   if model.lakes:
       fig, ax = plot_mesh(model.mesh, edge_color='lightgray', alpha=0.15)
       plot_lakes(model.lakes, model.mesh, ax=ax)
       ax.set_title('Lakes')
       fig.savefig(output / "lakes.png", dpi=150)

   # --- Cross-section ---
   all_x = [n.x for n in model.mesh.nodes.values()]
   all_y = [n.y for n in model.mesh.nodes.values()]
   extractor = CrossSectionExtractor(model.mesh, model.stratigraphy)
   xs = extractor.extract(
       start=(min(all_x), (min(all_y) + max(all_y)) / 2),
       end=(max(all_x), (min(all_y) + max(all_y)) / 2),
       n_samples=200,
   )
   fig, ax = plot_cross_section(xs, title='East-West Cross-Section')
   fig.savefig(output / "cross_section.png", dpi=150)

   # --- Export to GIS ---
   exporter = GISExporter(grid=model.mesh, crs="EPSG:26910")
   exporter.export_geopackage(output / "model.gpkg")

   print(f"All outputs saved to {output}/")

Next Steps
----------

- See :doc:`building_sample_model` for constructing a model from scratch
- See :doc:`visualization` for advanced plotting techniques
- See :doc:`/user_guide/io` for all supported file formats
- Launch the web viewer with ``pyiwfm viewer --model-dir C2VSimCG/``
