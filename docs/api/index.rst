API Reference
=============

This section contains the complete API documentation for pyiwfm.

.. toctree::
   :maxdepth: 2

   core
   components
   io
   mesh_generation
   visualization
   comparison
   runner

Module Overview
---------------

Core Modules
~~~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :recursive:

   pyiwfm.core.mesh
   pyiwfm.core.model
   pyiwfm.core.stratigraphy
   pyiwfm.core.timeseries
   pyiwfm.core.zones
   pyiwfm.core.aggregation
   pyiwfm.core.query

Component Modules
~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :recursive:

   pyiwfm.components.groundwater
   pyiwfm.components.stream
   pyiwfm.components.lake
   pyiwfm.components.rootzone
   pyiwfm.components.connectors

I/O Modules
~~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :recursive:

   pyiwfm.io.ascii
   pyiwfm.io.binary
   pyiwfm.io.hdf5
   pyiwfm.io.preprocessor
   pyiwfm.io.simulation
   pyiwfm.io.groundwater
   pyiwfm.io.streams
   pyiwfm.io.lakes
   pyiwfm.io.rootzone
   pyiwfm.io.timeseries_ascii
   pyiwfm.io.dss
   pyiwfm.io.zones

Mesh Generation
~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :recursive:

   pyiwfm.mesh_generation.generators
   pyiwfm.mesh_generation.constraints
   pyiwfm.mesh_generation.triangle_wrapper
   pyiwfm.mesh_generation.gmsh_wrapper

Visualization
~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :recursive:

   pyiwfm.visualization.gis_export
   pyiwfm.visualization.vtk_export
   pyiwfm.visualization.plotting
   pyiwfm.visualization.web.app
   pyiwfm.visualization.web.viewer
   pyiwfm.visualization.web.properties
   pyiwfm.visualization.web.multi_scale

Comparison
~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :recursive:

   pyiwfm.comparison.differ
   pyiwfm.comparison.metrics
   pyiwfm.comparison.report

Runner and PEST++ Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :recursive:

   pyiwfm.runner.runner
   pyiwfm.runner.results
   pyiwfm.runner.scenario
   pyiwfm.runner.pest
   pyiwfm.runner.pest_params
   pyiwfm.runner.pest_manager
   pyiwfm.runner.pest_observations
   pyiwfm.runner.pest_obs_manager
   pyiwfm.runner.pest_templates
   pyiwfm.runner.pest_instructions
   pyiwfm.runner.pest_geostat
   pyiwfm.runner.pest_helper
   pyiwfm.runner.pest_ensemble
   pyiwfm.runner.pest_postprocessor

Sample Models
~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :recursive:

   pyiwfm.sample_models
