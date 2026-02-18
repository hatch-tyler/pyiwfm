Tutorial: Building the IWFM Sample Model from Scratch
======================================================

This tutorial demonstrates how to build the official IWFM sample model
programmatically with pyiwfm and then run it. The sample model is a
uniform grid with homogeneous aquifer properties, streams, a lake, root
zone land use, and a 10-year daily simulation.

Learning Objectives
-------------------

By the end of this tutorial, you will be able to:

1. Create mesh nodes and elements programmatically
2. Define stratigraphy layers
3. Configure groundwater, stream, lake, and root zone components
4. Assemble an ``IWFMModel`` and write all input files
5. Run the IWFM preprocessor and simulation
6. Visualize simulation results

Overview
--------

The IWFM sample model we are building has these properties:

- **Mesh**: 21 x 21 rectangular grid (441 nodes, 400 quad elements)
- **Subregions**: 2 (left half and right half)
- **Stratigraphy**: 2 aquifer layers
- **Streams**: 3 reaches with 23 stream nodes
- **Lakes**: 1 lake
- **Root Zone**: Non-ponded crops, ponded (rice), urban, native vegetation
- **Simulation period**: 10/01/1990 -- 09/30/2000 (daily timestep)

Section 1: Create the Mesh
---------------------------

Build the 21 x 21 node grid (441 nodes) and 400 quadrilateral elements:

.. code-block:: python

   import numpy as np
   from pyiwfm.core.mesh import AppGrid, Node, Element

   # Grid dimensions
   nx, ny = 21, 21
   x0, y0 = 1_804_440.0, 14_435_520.0   # Lower-left corner (ft)
   dx, dy = 6_561.6, 2_296.56             # Node spacing (ft)

   # Create nodes
   nodes = {}
   node_id = 1
   for j in range(ny):
       for i in range(nx):
           is_boundary = (i == 0 or i == nx - 1 or j == 0 or j == ny - 1)
           nodes[node_id] = Node(
               id=node_id,
               x=x0 + i * dx,
               y=y0 + j * dy,
               is_boundary=is_boundary,
           )
           node_id += 1

   print(f"Created {len(nodes)} nodes")

   # Create elements (20 x 20 = 400 quads)
   elements = {}
   elem_id = 1
   for j in range(ny - 1):
       for i in range(nx - 1):
           n1 = j * nx + i + 1
           n2 = n1 + 1
           n3 = n2 + nx
           n4 = n1 + nx
           # Left half = subregion 1, right half = subregion 2
           subregion = 1 if i < (nx - 1) // 2 else 2
           elements[elem_id] = Element(
               id=elem_id,
               vertices=(n1, n2, n3, n4),
               subregion=subregion,
           )
           elem_id += 1

   print(f"Created {len(elements)} elements")

   # Assemble the grid
   grid = AppGrid(nodes=nodes, elements=elements)
   grid.compute_connectivity()

   print(f"Grid: {grid.n_nodes} nodes, {grid.n_elements} elements")
   print(f"Subregions: {sorted(grid.subregions)}")

Visualize the mesh:

.. code-block:: python

   from pyiwfm.visualization.plotting import plot_mesh, plot_elements

   fig, ax = plot_elements(grid, color_by='subregion', cmap='Set2', alpha=0.6)
   ax.set_title(f'Sample Model Mesh ({grid.n_nodes} nodes, {grid.n_elements} elements)')
   ax.set_xlabel('X (feet)')
   ax.set_ylabel('Y (feet)')
   fig.savefig("mesh.png", dpi=150)

Section 2: Define Stratigraphy
-------------------------------

Create a 2-layer stratigraphy with sloping ground surface:

.. code-block:: python

   from pyiwfm.core.stratigraphy import Stratigraphy

   n_nodes = grid.n_nodes
   n_layers = 2

   # Ground surface elevation: slopes from 400 ft (south) to 200 ft (north)
   gs_elev = np.array([
       400.0 - 200.0 * (nodes[i].y - y0) / ((ny - 1) * dy)
       for i in range(1, n_nodes + 1)
   ])

   # Layer 1: top 120 ft
   top_elev = np.column_stack([
       gs_elev,
       gs_elev - 120.0,
   ])
   bottom_elev = np.column_stack([
       gs_elev - 120.0,
       gs_elev - 240.0,
   ])

   # All nodes active in all layers
   active_node = np.ones((n_nodes, n_layers), dtype=bool)

   stratigraphy = Stratigraphy(
       n_layers=n_layers,
       n_nodes=n_nodes,
       gs_elev=gs_elev,
       top_elev=top_elev,
       bottom_elev=bottom_elev,
       active_node=active_node,
   )

   print(f"Stratigraphy: {stratigraphy.n_layers} layers")
   print(f"Ground surface range: {gs_elev.min():.0f} - {gs_elev.max():.0f} ft")

Visualize ground surface and layer thickness:

.. code-block:: python

   from pyiwfm.visualization.plotting import plot_scalar_field

   fig, ax = plot_scalar_field(grid, gs_elev, cmap='terrain',
                               show_mesh=True, edge_color='white')
   ax.set_title('Ground Surface Elevation (ft)')
   fig.savefig("ground_surface.png", dpi=150)

   thickness = top_elev[:, 0] - bottom_elev[:, 0]
   fig, ax = plot_scalar_field(grid, thickness, cmap='YlOrRd')
   ax.set_title('Layer 1 Thickness (ft)')
   fig.savefig("layer1_thickness.png", dpi=150)

Section 3: Groundwater Component
---------------------------------

Configure the groundwater component with aquifer parameters, initial
heads, and boundary conditions:

.. code-block:: python

   from pyiwfm.components.groundwater import (
       AppGW, AquiferParameters, BoundaryCondition, Well,
   )

   # Aquifer parameters (uniform)
   aquifer_params = AquiferParameters(
       pkh=np.full((n_nodes, n_layers), 50.0),     # Horiz. hydraulic conductivity (ft/day)
       ps=np.full((n_nodes, n_layers), 1e-6),       # Specific storage (1/ft)
       pn=np.full((n_nodes, n_layers), 0.25),       # Porosity
       pv=np.full((n_nodes, n_layers), 0.2),        # Vertical anisotropy (Kv/Kh)
   )

   # Initial heads: approximates ground surface with slight offset
   initial_heads = np.column_stack([
       gs_elev - 20.0,   # Layer 1: 20 ft below ground
       gs_elev - 40.0,   # Layer 2: 40 ft below ground
   ])

   # Specified head boundary conditions at upstream nodes (south boundary)
   boundary_conditions = []
   for node_id in range(1, nx + 1):  # Row 1 (south)
       bc = BoundaryCondition(
           node_id=node_id,
           layer=1,
           bc_type="specified_head",
           head=gs_elev[node_id - 1] - 20.0,
       )
       boundary_conditions.append(bc)

   # Create groundwater component
   gw = AppGW(
       n_nodes=n_nodes,
       n_layers=n_layers,
       n_elements=grid.n_elements,
       aquifer_params=aquifer_params,
       heads=initial_heads,
       boundary_conditions=boundary_conditions,
   )

   print(f"GW: {len(gw.boundary_conditions)} boundary conditions")

Visualize initial head distribution:

.. code-block:: python

   fig, ax = plot_scalar_field(grid, initial_heads[:, 0],
                               cmap='viridis', show_mesh=True, edge_color='white')
   ax.set_title('Initial Head - Layer 1 (ft)')
   fig.savefig("initial_heads.png", dpi=150)

Section 4: Stream Component
----------------------------

Create the stream network with 3 reaches flowing south to north:

.. code-block:: python

   from pyiwfm.components.stream import AppStream, StrmNode, StrmReach

   stream = AppStream()

   # Create stream nodes along a central channel
   # Main channel runs from south to north through column 11
   center_col = nx // 2  # column index 10 (node x index)
   strm_id = 1
   for j in range(ny):
       gw_node_id = j * nx + center_col + 1
       node = StrmNode(
           id=strm_id,
           gw_node=gw_node_id,
           x=nodes[gw_node_id].x,
           y=nodes[gw_node_id].y,
       )
       stream.add_node(node)
       strm_id += 1

   # Create 3 reaches
   # Reach 1: stream nodes 1-7 (southern third)
   reach1 = StrmReach(id=1, nodes=list(range(1, 8)))
   stream.add_reach(reach1)

   # Reach 2: stream nodes 7-14 (middle third)
   reach2 = StrmReach(id=2, nodes=list(range(7, 15)))
   stream.add_reach(reach2)

   # Reach 3: stream nodes 14-21 (northern third)
   reach3 = StrmReach(id=3, nodes=list(range(14, 22)))
   stream.add_reach(reach3)

   print(f"Streams: {len(stream.nodes)} nodes, {len(stream.reaches)} reaches")

Visualize the stream network overlaid on the mesh:

.. code-block:: python

   fig, ax = plot_mesh(grid, show_edges=True, edge_color='lightgray', alpha=0.2)

   from pyiwfm.visualization.plotting import plot_streams
   plot_streams(stream, ax=ax, show_nodes=True, line_color='blue', line_width=2)
   ax.set_title('Stream Network (3 Reaches)')
   fig.savefig("streams.png", dpi=150)

Section 5: Lake Component
--------------------------

Define a lake occupying a cluster of elements in the northeast:

.. code-block:: python

   from pyiwfm.components.lake import AppLake, Lake, LakeElement

   lake_component = AppLake()

   # Lake 1 occupies a 3x3 block of elements in the northeast corner
   lake_elem_ids = []
   for j in range(17, 20):       # Top 3 rows
       for i in range(17, 20):   # Right 3 columns
           eid = j * (nx - 1) + i + 1
           lake_elem_ids.append(eid)

   lake = Lake(
       id=1,
       name="Sample Lake",
       max_elevation=350.0,
       bed_conductance=0.1,
       outflow_destination=0,  # No outflow destination
   )
   lake_component.add_lake(lake)

   for eid in lake_elem_ids:
       lake_component.add_lake_element(LakeElement(lake_id=1, element_id=eid))

   print(f"Lakes: {len(lake_component.lakes)} lake(s), "
         f"{len(lake_component.lake_elements)} elements")

Visualize lake elements:

.. code-block:: python

   from pyiwfm.visualization.plotting import plot_elements

   fig, ax = plot_mesh(grid, show_edges=True, edge_color='lightgray', alpha=0.2)

   # Highlight lake elements
   for eid in lake_elem_ids:
       elem = elements[eid]
       verts = [(nodes[v].x, nodes[v].y) for v in elem.vertices]
       from matplotlib.patches import Polygon as MplPolygon
       ax.add_patch(MplPolygon(verts, facecolor='cyan', edgecolor='blue',
                               alpha=0.7, linewidth=1))

   ax.set_title('Lake Elements')
   fig.savefig("lake_elements.png", dpi=150)

Section 6: Root Zone Component
-------------------------------

Configure root zone with non-ponded crops, ponded agriculture (rice),
urban, and native vegetation:

.. code-block:: python

   from pyiwfm.components.rootzone import (
       RootZone, CropType, SoilParameters, ElementLandUse,
   )

   rz = RootZone(
       n_elements=grid.n_elements,
       n_layers=n_layers,
   )

   # Define crop types
   crops = [
       CropType(id=1, name="Tomato"),
       CropType(id=2, name="Alfalfa"),
       CropType(id=3, name="Rice"),        # ponded
       CropType(id=4, name="Urban"),
       CropType(id=5, name="Native Veg"),
       CropType(id=6, name="Riparian"),
   ]
   for crop in crops:
       rz.add_crop_type(crop)

   # Set soil parameters for all elements
   for eid in range(1, grid.n_elements + 1):
       rz.set_soil_parameters(eid, SoilParameters(
           wilting_point=0.10,
           field_capacity=0.30,
           total_porosity=0.40,
           root_depth=4.0,           # ft
           saturated_kh=10.0,        # ft/day
       ))

   print(f"Root Zone: {len(rz.crop_types)} crop types, "
         f"{len(rz.soil_params)} element soil params")

Section 7: Assemble and Write the Model
-----------------------------------------

Combine all components into an ``IWFMModel`` and write to disk:

.. code-block:: python

   from pathlib import Path
   from pyiwfm.core.model import IWFMModel
   from pyiwfm.io import save_complete_model

   # Assemble the model
   model = IWFMModel(
       name="IWFM Sample Model",
       mesh=grid,
       stratigraphy=stratigraphy,
       groundwater=gw,
       streams=stream,
       lakes=lake_component,
       rootzone=rz,
       metadata={
           "start_date": "10/01/1990",
           "end_date": "09/30/2000",
           "timestep": "1DAY",
           "description": "IWFM sample model built with pyiwfm",
       },
   )

   print(model.summary())

   # Write all input files
   output_dir = Path("sample_model_output")
   output_dir.mkdir(exist_ok=True)

   files_written = save_complete_model(model, output_dir)

   print(f"\nFiles written ({len(files_written)}):")
   for name, path in sorted(files_written.items()):
       print(f"  {name}: {path}")

Section 8: Run the Preprocessor
---------------------------------

Use ``IWFMRunner`` to execute the IWFM preprocessor:

.. code-block:: python

   from pyiwfm.runner.runner import IWFMRunner
   from pyiwfm.runner.executables import find_iwfm_executables

   # Find IWFM executables on the system
   executables = find_iwfm_executables()
   print(f"Preprocessor: {executables.preprocessor}")
   print(f"Simulation:   {executables.simulation}")

   # Run preprocessor
   runner = IWFMRunner(executables=executables, working_dir=output_dir)
   pp_result = runner.run_preprocessor("Preprocessor.in")

   print(f"Preprocessor success: {pp_result.success}")
   if not pp_result.success:
       print(f"Error: {pp_result.stderr}")
   else:
       print("Preprocessor completed successfully")

Section 9: Run the Simulation
-------------------------------

Execute the IWFM simulation:

.. code-block:: python

   sim_result = runner.run_simulation("Simulation.in", timeout=600)

   print(f"Simulation success: {sim_result.success}")
   if sim_result.success:
       print(f"Runtime: {sim_result.runtime:.1f} seconds")
       print("Output files generated")
   else:
       print(f"Error: {sim_result.stderr}")

Section 10: Visualize Results
------------------------------

Load and visualize the simulation results:

.. code-block:: python

   from pyiwfm.visualization.plotting import (
       plot_scalar_field, plot_timeseries, plot_budget_bar,
       plot_budget_stacked, plot_budget_pie,
   )
   from pyiwfm.core.timeseries import TimeSeries

   # --- Groundwater heads at final timestep ---
   # (Assuming head results are available in HDF5 or output files)
   # final_heads = load_head_results(output_dir / "Results" / "GW_Heads.hdf")

   # For demonstration with synthetic final heads:
   final_heads = initial_heads[:, 0] - 5.0 * np.random.uniform(0, 1, n_nodes)
   head_change = final_heads - initial_heads[:, 0]

   fig, ax = plot_scalar_field(grid, final_heads, cmap='viridis',
                               show_mesh=True, edge_color='white')
   ax.set_title('Final Groundwater Head (ft)')
   fig.savefig("final_heads.png", dpi=150)

   fig, ax = plot_scalar_field(grid, head_change, cmap='RdBu',
                               show_mesh=False)
   ax.set_title('Head Change: Final - Initial (ft)')
   fig.savefig("head_change.png", dpi=150)

   # --- Head time series at selected nodes ---
   selected_nodes = [111, 221, 331]  # Center of grid at different rows
   times = np.arange('1990-10-01', '2000-10-01', dtype='datetime64[M]')
   n_times = len(times)

   series_list = []
   for nid in selected_nodes:
       synthetic_head = (
           initial_heads[nid - 1, 0]
           - 0.5 * np.arange(n_times) / n_times * 10
           + 2 * np.sin(2 * np.pi * np.arange(n_times) / 12)
           + np.random.normal(0, 0.5, n_times)
       )
       ts = TimeSeries(times=times, values=synthetic_head,
                       name=f"Node {nid}", units="ft")
       series_list.append(ts)

   fig, ax = plot_timeseries(series_list, title='Head at Selected Nodes',
                              ylabel='Head (ft)')
   fig.savefig("head_timeseries.png", dpi=150)

   # --- Budget visualizations ---
   budget_components = {
       'Recharge': 5200,
       'Stream Seepage': 3100,
       'Subsurface Inflow': 1800,
       'Pumping': -7500,
       'Stream Baseflow': -2100,
       'GW ET': -900,
   }

   fig, ax = plot_budget_bar(budget_components, title='GW Budget Summary',
                              units='AF/year')
   fig.savefig("budget_bar.png", dpi=150)

   # Budget over time
   n_years = 10
   budget_times = np.arange('1991-01-01', '2001-01-01', dtype='datetime64[Y]')
   np.random.seed(42)
   time_budgets = {
       'Recharge': 5200 + np.random.normal(0, 500, n_years),
       'Stream Seepage': 3100 + np.random.normal(0, 300, n_years),
       'Pumping': -(7500 + np.arange(n_years) * 100 + np.random.normal(0, 200, n_years)),
       'Baseflow': -(2100 + np.random.normal(0, 200, n_years)),
   }

   fig, ax = plot_budget_stacked(budget_times, time_budgets,
                                  title='GW Budget Over Time', units='AF/year')
   fig.savefig("budget_stacked.png", dpi=150)

   # Root zone water balance pie chart
   rz_components = {
       'Precipitation': 12000,
       'Applied Water': 8500,
       'ET': -15000,
       'Deep Percolation': -4000,
       'Runoff': -1500,
   }
   fig, ax = plot_budget_pie({k: abs(v) for k, v in rz_components.items()},
                              title='Root Zone Water Balance',
                              budget_type='both')
   fig.savefig("rz_pie.png", dpi=150)

   print("All result plots saved!")

Interactive Viewing
-------------------

For interactive 3D visualization and map exploration, launch the web viewer:

.. code-block:: bash

   pyiwfm viewer --model-dir sample_model_output/

This opens a browser with 4 tabs: Overview, 3D Mesh, Results Map, and Budgets.

Complete Script
---------------

Here is the complete script combining all steps:

.. code-block:: python

   """Build the IWFM sample model from scratch and visualize results."""

   import numpy as np
   from pathlib import Path
   from pyiwfm.core.mesh import AppGrid, Node, Element
   from pyiwfm.core.stratigraphy import Stratigraphy
   from pyiwfm.core.model import IWFMModel
   from pyiwfm.components.groundwater import (
       AppGW, AquiferParameters, BoundaryCondition,
   )
   from pyiwfm.components.stream import AppStream, StrmNode, StrmReach
   from pyiwfm.components.lake import AppLake, Lake, LakeElement
   from pyiwfm.components.rootzone import RootZone, CropType, SoilParameters
   from pyiwfm.io import save_complete_model
   from pyiwfm.visualization.plotting import (
       plot_mesh, plot_elements, plot_scalar_field, plot_streams,
       plot_timeseries, plot_budget_bar,
   )
   from pyiwfm.core.timeseries import TimeSeries

   # ---- 1. Mesh ----
   nx, ny = 21, 21
   x0, y0 = 1_804_440.0, 14_435_520.0
   dx, dy = 6_561.6, 2_296.56

   nodes = {}
   nid = 1
   for j in range(ny):
       for i in range(nx):
           is_boundary = (i == 0 or i == nx - 1 or j == 0 or j == ny - 1)
           nodes[nid] = Node(id=nid, x=x0 + i * dx, y=y0 + j * dy,
                             is_boundary=is_boundary)
           nid += 1

   elements = {}
   eid = 1
   for j in range(ny - 1):
       for i in range(nx - 1):
           n1 = j * nx + i + 1
           elements[eid] = Element(
               id=eid,
               vertices=(n1, n1 + 1, n1 + 1 + nx, n1 + nx),
               subregion=1 if i < 10 else 2,
           )
           eid += 1

   grid = AppGrid(nodes=nodes, elements=elements)
   grid.compute_connectivity()

   # ---- 2. Stratigraphy ----
   n_nodes = grid.n_nodes
   gs_elev = np.array([
       400.0 - 200.0 * (nodes[i].y - y0) / ((ny - 1) * dy)
       for i in range(1, n_nodes + 1)
   ])
   top_elev = np.column_stack([gs_elev, gs_elev - 120.0])
   bottom_elev = np.column_stack([gs_elev - 120.0, gs_elev - 240.0])
   active_node = np.ones((n_nodes, 2), dtype=bool)

   strat = Stratigraphy(
       n_layers=2, n_nodes=n_nodes, gs_elev=gs_elev,
       top_elev=top_elev, bottom_elev=bottom_elev, active_node=active_node,
   )

   # ---- 3. Groundwater ----
   initial_heads = np.column_stack([gs_elev - 20.0, gs_elev - 40.0])
   bcs = [
       BoundaryCondition(
           node_id=i, layer=1, bc_type="specified_head",
           head=gs_elev[i - 1] - 20.0,
       )
       for i in range(1, nx + 1)
   ]
   gw = AppGW(
       n_nodes=n_nodes, n_layers=2, n_elements=grid.n_elements,
       aquifer_params=AquiferParameters(
           pkh=np.full((n_nodes, 2), 50.0),
           ps=np.full((n_nodes, 2), 1e-6),
           pn=np.full((n_nodes, 2), 0.25),
           pv=np.full((n_nodes, 2), 0.2),
       ),
       heads=initial_heads,
       boundary_conditions=bcs,
   )

   # ---- 4. Streams ----
   stream = AppStream()
   center_col = nx // 2
   for j in range(ny):
       gw_nid = j * nx + center_col + 1
       stream.add_node(StrmNode(id=j + 1, gw_node=gw_nid,
                                x=nodes[gw_nid].x, y=nodes[gw_nid].y))
   stream.add_reach(StrmReach(id=1, nodes=list(range(1, 8))))
   stream.add_reach(StrmReach(id=2, nodes=list(range(7, 15))))
   stream.add_reach(StrmReach(id=3, nodes=list(range(14, 22))))

   # ---- 5. Lake ----
   lake_comp = AppLake()
   lake_comp.add_lake(Lake(id=1, name="Sample Lake",
                           max_elevation=350.0, bed_conductance=0.1,
                           outflow_destination=0))
   for j in range(17, 20):
       for i in range(17, 20):
           lake_comp.add_lake_element(
               LakeElement(lake_id=1, element_id=j * (nx - 1) + i + 1))

   # ---- 6. Root Zone ----
   rz = RootZone(n_elements=grid.n_elements, n_layers=2)
   for crop in [CropType(id=1, name="Tomato"), CropType(id=2, name="Alfalfa"),
                CropType(id=3, name="Rice"), CropType(id=4, name="Urban"),
                CropType(id=5, name="Native Veg"), CropType(id=6, name="Riparian")]:
       rz.add_crop_type(crop)
   for e in range(1, grid.n_elements + 1):
       rz.set_soil_parameters(e, SoilParameters(
           wilting_point=0.10, field_capacity=0.30,
           total_porosity=0.40, root_depth=4.0, saturated_kh=10.0))

   # ---- 7. Assemble and Write ----
   model = IWFMModel(
       name="IWFM Sample Model",
       mesh=grid, stratigraphy=strat,
       groundwater=gw, streams=stream,
       lakes=lake_comp, rootzone=rz,
       metadata={
           "start_date": "10/01/1990",
           "end_date": "09/30/2000",
           "timestep": "1DAY",
       },
   )
   print(model.summary())

   output_dir = Path("sample_model_output")
   output_dir.mkdir(exist_ok=True)
   files = save_complete_model(model, output_dir)
   print(f"Wrote {len(files)} files to {output_dir}")

   # ---- 8. Visualize ----
   fig, ax = plot_elements(grid, color_by='subregion', cmap='Set2')
   ax.set_title('Sample Model Mesh')
   fig.savefig(output_dir / "mesh.png", dpi=150)

   fig, ax = plot_scalar_field(grid, gs_elev, cmap='terrain')
   ax.set_title('Ground Surface Elevation (ft)')
   fig.savefig(output_dir / "ground_surface.png", dpi=150)

   fig, ax = plot_mesh(grid, edge_color='lightgray', alpha=0.2)
   plot_streams(stream, ax=ax, show_nodes=True, line_width=2)
   ax.set_title('Stream Network')
   fig.savefig(output_dir / "streams.png", dpi=150)

   fig, ax = plot_scalar_field(grid, initial_heads[:, 0], cmap='viridis')
   ax.set_title('Initial Head - Layer 1 (ft)')
   fig.savefig(output_dir / "initial_heads.png", dpi=150)

   print("Done! All plots saved.")

Next Steps
----------

- See :doc:`reading_models` for loading existing models
- See :doc:`/user_guide/io` for the full I/O API
- Use ``pyiwfm viewer --model-dir sample_model_output/`` for interactive viewing
- See :doc:`/user_guide/quickstart` for other pyiwfm workflows
