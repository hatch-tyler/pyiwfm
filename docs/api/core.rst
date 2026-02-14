Core Modules
============

The core modules provide the fundamental data structures for representing
IWFM models, including meshes, stratigraphy, and time series data.

Mesh Module
-----------

The mesh module contains classes for representing finite element meshes.

.. automodule:: pyiwfm.core.mesh
   :members:
   :undoc-members:
   :show-inheritance:

Node
~~~~

.. autoclass:: pyiwfm.core.mesh.Node
   :members:
   :undoc-members:
   :show-inheritance:

Element
~~~~~~~

.. autoclass:: pyiwfm.core.mesh.Element
   :members:
   :undoc-members:
   :show-inheritance:

Face
~~~~

.. autoclass:: pyiwfm.core.mesh.Face
   :members:
   :undoc-members:
   :show-inheritance:

AppGrid
~~~~~~~

.. autoclass:: pyiwfm.core.mesh.AppGrid
   :members:
   :undoc-members:
   :show-inheritance:


Stratigraphy Module
-------------------

The stratigraphy module contains classes for representing model layer structure.

.. automodule:: pyiwfm.core.stratigraphy
   :members:
   :undoc-members:
   :show-inheritance:


Time Series Module
------------------

The time series module provides classes for working with temporal data.

.. automodule:: pyiwfm.core.timeseries
   :members:
   :undoc-members:
   :show-inheritance:


Model Module
------------

The model module provides the central IWFMModel class that orchestrates all
model components.

.. automodule:: pyiwfm.core.model
   :members:
   :undoc-members:
   :show-inheritance:

IWFMModel
~~~~~~~~~

The IWFMModel class is the main container for IWFM models, providing methods
for loading models from various sources and saving to different formats.

**Loading Models:**

- ``IWFMModel.from_preprocessor(pp_file)`` - Load from preprocessor input files (mesh, stratigraphy, geometry)
- ``IWFMModel.from_preprocessor_binary(binary_file)`` - Load from preprocessor binary output
- ``IWFMModel.from_simulation(sim_file)`` - Load complete model from simulation input file
- ``IWFMModel.from_simulation_with_preprocessor(sim_file, pp_file)`` - Load using both files
- ``IWFMModel.from_hdf5(hdf5_file)`` - Load from HDF5 file

**Saving Models:**

- ``model.to_preprocessor(output_dir)`` - Save to preprocessor input files
- ``model.to_simulation(output_dir)`` - Save complete model to simulation files
- ``model.to_hdf5(output_file)`` - Save to HDF5 format
- ``model.to_binary(output_file)`` - Save mesh/stratigraphy to binary

.. autoclass:: pyiwfm.core.model.IWFMModel
   :members:
   :undoc-members:
   :show-inheritance:


Zones Module
------------

The zones module provides data structures for defining spatial zones
(subregions, custom zones for ZBudget analysis) and mapping elements to zones.

.. automodule:: pyiwfm.core.zones
   :members:
   :undoc-members:
   :show-inheritance:

Zone
~~~~

.. autoclass:: pyiwfm.core.zones.Zone
   :members:
   :undoc-members:
   :show-inheritance:

ZoneDefinition
~~~~~~~~~~~~~~

.. autoclass:: pyiwfm.core.zones.ZoneDefinition
   :members:
   :undoc-members:
   :show-inheritance:


Aggregation Module
------------------

The aggregation module provides spatial data aggregation from element-level
values to zone-level statistics using configurable methods.

.. automodule:: pyiwfm.core.aggregation
   :members:
   :undoc-members:
   :show-inheritance:

DataAggregator
~~~~~~~~~~~~~~

.. autoclass:: pyiwfm.core.aggregation.DataAggregator
   :members:
   :undoc-members:
   :show-inheritance:


Query Module
------------

The query module provides a high-level API for accessing model data at
multiple spatial scales with aggregation and export capabilities.

.. automodule:: pyiwfm.core.query
   :members:
   :undoc-members:
   :show-inheritance:

ModelQueryAPI
~~~~~~~~~~~~~

.. autoclass:: pyiwfm.core.query.ModelQueryAPI
   :members:
   :undoc-members:
   :show-inheritance:
