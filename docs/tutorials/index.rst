Tutorials
=========

Step-by-step tutorials for common pyiwfm workflows.

.. toctree::
   :maxdepth: 2

   reading_models
   building_sample_model
   mesh_generation
   visualization
   model_comparison

Overview
--------

These tutorials walk you through complete workflows using pyiwfm:

**Reading an Existing Model**
    Load an existing IWFM model, inspect its components, read simulation
    results, and create visualizations. Start here if you already have a
    model and want to work with it in Python.

**Building the Sample Model from Scratch**
    Build the official IWFM sample model programmatically -- mesh,
    stratigraphy, groundwater, streams, lakes, root zone -- then run the
    preprocessor and simulation and visualize results.

**Mesh Generation**
    Create a finite element mesh from scratch, including defining boundaries,
    adding stream constraints, and refining areas of interest.

**Visualization**
    Export model data for visualization in GIS software, ParaView, and
    matplotlib. Create publication-quality figures.

**Model Comparison**
    Compare two model versions, compute performance metrics, and generate
    comparison reports.

Prerequisites
-------------

Make sure you have pyiwfm installed with the required optional dependencies:

.. code-block:: bash

    pip install pyiwfm[all]

The tutorials assume basic familiarity with:

- Python and NumPy arrays
- IWFM model concepts (mesh, stratigraphy, layers)
- Basic GIS concepts (coordinate systems, shapefiles)
