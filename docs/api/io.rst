I/O Modules
===========

The I/O modules provide functionality for reading and writing IWFM model files
in various formats.

.. contents:: Table of Contents
   :local:
   :depth: 2

Core I/O
--------

ASCII Module
~~~~~~~~~~~~

The ASCII module provides readers and writers for IWFM's text-based file formats.

.. automodule:: pyiwfm.io.ascii
   :members:
   :undoc-members:
   :show-inheritance:

Binary Module
~~~~~~~~~~~~~

The binary module provides readers and writers for IWFM's Fortran binary formats.

.. automodule:: pyiwfm.io.binary
   :members:
   :undoc-members:
   :show-inheritance:

HDF5 Module
~~~~~~~~~~~

The HDF5 module provides readers and writers for HDF5-based storage.

.. automodule:: pyiwfm.io.hdf5
   :members:
   :undoc-members:
   :show-inheritance:

Base Classes
~~~~~~~~~~~~

Abstract base classes for readers and writers.

.. automodule:: pyiwfm.io.base
   :members:
   :undoc-members:
   :show-inheritance:

Writer Base
~~~~~~~~~~~

Template-based writer base classes used by all component writers.

.. automodule:: pyiwfm.io.writer_base
   :members:
   :undoc-members:
   :show-inheritance:

Writer Configuration
~~~~~~~~~~~~~~~~~~~~

Configuration dataclasses for the model writer system.

.. automodule:: pyiwfm.io.config
   :members:
   :undoc-members:
   :show-inheritance:

Writer Configuration Base
~~~~~~~~~~~~~~~~~~~~~~~~~

Shared base dataclass (``BaseComponentWriterConfig``) for all component writer
configs (groundwater, streams, lakes, root zone, small watersheds, unsaturated
zone). Provides common fields: ``output_dir``, ``version``, ``length_factor``,
``length_unit``, ``volume_factor``, ``volume_unit``, ``subdir``.

.. automodule:: pyiwfm.io.writer_config_base
   :members:
   :undoc-members:
   :show-inheritance:

PreProcessor I/O
----------------

PreProcessor Module
~~~~~~~~~~~~~~~~~~~

The preprocessor module provides functions for loading and saving complete IWFM models.

.. automodule:: pyiwfm.io.preprocessor
   :members:
   :undoc-members:
   :show-inheritance:

PreProcessor Binary
~~~~~~~~~~~~~~~~~~~

Reader for IWFM preprocessor binary output files.

.. automodule:: pyiwfm.io.preprocessor_binary
   :members:
   :undoc-members:
   :show-inheritance:

PreProcessor Writer
~~~~~~~~~~~~~~~~~~~

Writer for IWFM preprocessor input files (nodes, elements, stratigraphy).

.. automodule:: pyiwfm.io.preprocessor_writer
   :members:
   :undoc-members:
   :show-inheritance:

Simulation I/O
--------------

Simulation Module
~~~~~~~~~~~~~~~~~

The simulation module provides readers and writers for IWFM simulation control files.

.. automodule:: pyiwfm.io.simulation
   :members:
   :undoc-members:
   :show-inheritance:

Simulation Writer
~~~~~~~~~~~~~~~~~

Writer for IWFM simulation main control files.

.. automodule:: pyiwfm.io.simulation_writer
   :members:
   :undoc-members:
   :show-inheritance:

Groundwater I/O
---------------

Groundwater Module
~~~~~~~~~~~~~~~~~~

The groundwater module provides readers and writers for IWFM groundwater component files.

.. automodule:: pyiwfm.io.groundwater
   :members:
   :undoc-members:
   :show-inheritance:

Groundwater Writer
~~~~~~~~~~~~~~~~~~

Writer for IWFM groundwater component files using Jinja2 templates.

.. automodule:: pyiwfm.io.gw_writer
   :members:
   :undoc-members:
   :show-inheritance:

Groundwater Boundary Conditions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reader for groundwater boundary condition files (specified head, general head).

.. automodule:: pyiwfm.io.gw_boundary
   :members:
   :undoc-members:
   :show-inheritance:

Groundwater Pumping
~~~~~~~~~~~~~~~~~~~

Reader for groundwater pumping specification files.

.. automodule:: pyiwfm.io.gw_pumping
   :members:
   :undoc-members:
   :show-inheritance:

Groundwater Tile Drains
~~~~~~~~~~~~~~~~~~~~~~~

Reader for groundwater tile drain specification files.

.. automodule:: pyiwfm.io.gw_tiledrain
   :members:
   :undoc-members:
   :show-inheritance:

Groundwater Subsidence
~~~~~~~~~~~~~~~~~~~~~~

Reader for groundwater subsidence parameter files.

.. automodule:: pyiwfm.io.gw_subsidence
   :members:
   :undoc-members:
   :show-inheritance:

Stream I/O
----------

Streams Module
~~~~~~~~~~~~~~

The streams module provides readers and writers for IWFM stream network component files.

.. automodule:: pyiwfm.io.streams
   :members:
   :undoc-members:
   :show-inheritance:

Stream Writer
~~~~~~~~~~~~~

Writer for IWFM stream component files using Jinja2 templates.

.. automodule:: pyiwfm.io.stream_writer
   :members:
   :undoc-members:
   :show-inheritance:

Stream Diversion
~~~~~~~~~~~~~~~~

Reader for stream diversion specification files.

.. automodule:: pyiwfm.io.stream_diversion
   :members:
   :undoc-members:
   :show-inheritance:

Stream Bypass
~~~~~~~~~~~~~

Reader for stream bypass specification files.

.. automodule:: pyiwfm.io.stream_bypass
   :members:
   :undoc-members:
   :show-inheritance:

Stream Inflow
~~~~~~~~~~~~~

Reader for stream inflow data files.

.. automodule:: pyiwfm.io.stream_inflow
   :members:
   :undoc-members:
   :show-inheritance:

Lake I/O
--------

Lakes Module
~~~~~~~~~~~~

The lakes module provides readers and writers for IWFM lake component files.

.. automodule:: pyiwfm.io.lakes
   :members:
   :undoc-members:
   :show-inheritance:

Lake Writer
~~~~~~~~~~~

Writer for IWFM lake component files using Jinja2 templates.

.. automodule:: pyiwfm.io.lake_writer
   :members:
   :undoc-members:
   :show-inheritance:

Root Zone I/O
-------------

Root Zone Module
~~~~~~~~~~~~~~~~

The root zone module provides readers and writers for IWFM root zone component files.

.. automodule:: pyiwfm.io.rootzone
   :members:
   :undoc-members:
   :show-inheritance:

Root Zone Writer
~~~~~~~~~~~~~~~~

Writer for IWFM root zone component files using Jinja2 templates.

.. automodule:: pyiwfm.io.rootzone_writer
   :members:
   :undoc-members:
   :show-inheritance:

Root Zone v4.x
~~~~~~~~~~~~~~

Readers and writers specific to IWFM v4.x root zone format.

.. automodule:: pyiwfm.io.rootzone_v4x
   :members:
   :undoc-members:
   :show-inheritance:

Root Zone Non-Ponded Crops
~~~~~~~~~~~~~~~~~~~~~~~~~~

Reader for non-ponded (dry) crop land use parameters.

.. automodule:: pyiwfm.io.rootzone_nonponded
   :members:
   :undoc-members:
   :show-inheritance:

Root Zone Ponded Crops
~~~~~~~~~~~~~~~~~~~~~~

Reader for ponded (rice/wetland) crop land use parameters.

.. automodule:: pyiwfm.io.rootzone_ponded
   :members:
   :undoc-members:
   :show-inheritance:

Root Zone Urban
~~~~~~~~~~~~~~~

Reader for urban land use parameters.

.. automodule:: pyiwfm.io.rootzone_urban
   :members:
   :undoc-members:
   :show-inheritance:

Root Zone Native/Riparian
~~~~~~~~~~~~~~~~~~~~~~~~~

Reader for native and riparian vegetation parameters.

.. automodule:: pyiwfm.io.rootzone_native
   :members:
   :undoc-members:
   :show-inheritance:

Supplemental Package I/O
-------------------------

Small Watershed Module
~~~~~~~~~~~~~~~~~~~~~~

Reader for IWFM small watershed component files.

.. automodule:: pyiwfm.io.small_watershed
   :members:
   :undoc-members:
   :show-inheritance:

Small Watershed Writer
~~~~~~~~~~~~~~~~~~~~~~

Writer for IWFM small watershed component files using Jinja2 templates.

.. automodule:: pyiwfm.io.small_watershed_writer
   :members:
   :undoc-members:
   :show-inheritance:

Unsaturated Zone Module
~~~~~~~~~~~~~~~~~~~~~~~

Reader for IWFM unsaturated zone component files.

.. automodule:: pyiwfm.io.unsaturated_zone
   :members:
   :undoc-members:
   :show-inheritance:

Unsaturated Zone Writer
~~~~~~~~~~~~~~~~~~~~~~~

Writer for IWFM unsaturated zone component files using Jinja2 templates.

.. automodule:: pyiwfm.io.unsaturated_zone_writer
   :members:
   :undoc-members:
   :show-inheritance:

Supply Adjustment
~~~~~~~~~~~~~~~~~

Reader and writer for IWFM supply adjustment (irrigation) files.

.. automodule:: pyiwfm.io.supply_adjust
   :members:
   :undoc-members:
   :show-inheritance:

Time Series I/O
---------------

Time Series ASCII Module
~~~~~~~~~~~~~~~~~~~~~~~~

The time series ASCII module provides readers and writers for IWFM ASCII time series files.

.. automodule:: pyiwfm.io.timeseries_ascii
   :members:
   :undoc-members:
   :show-inheritance:

Time Series Module
~~~~~~~~~~~~~~~~~~

Unified time series reader supporting multiple formats.

.. automodule:: pyiwfm.io.timeseries
   :members:
   :undoc-members:
   :show-inheritance:

Time Series Writer
~~~~~~~~~~~~~~~~~~

Writer for IWFM time series data files.

.. automodule:: pyiwfm.io.timeseries_writer
   :members:
   :undoc-members:
   :show-inheritance:

Budget / Results I/O
--------------------

Budget Module
~~~~~~~~~~~~~

Reader for IWFM budget output files (HDF5 and binary formats).

.. automodule:: pyiwfm.io.budget
   :members:
   :undoc-members:
   :show-inheritance:

ZBudget Module
~~~~~~~~~~~~~~

Reader for IWFM zone budget output files.

.. automodule:: pyiwfm.io.zbudget
   :members:
   :undoc-members:
   :show-inheritance:

Model I/O
---------

Model Loader
~~~~~~~~~~~~~

Complete model loading from simulation and preprocessor files.

.. automodule:: pyiwfm.io.model_loader
   :members:
   :undoc-members:
   :show-inheritance:

Model Writer
~~~~~~~~~~~~

Complete model writer that orchestrates all component writers.

.. automodule:: pyiwfm.io.model_writer
   :members:
   :undoc-members:
   :show-inheritance:

Infrastructure
--------------

IWFM Reader Utilities
~~~~~~~~~~~~~~~~~~~~~

Central module for IWFM file line-reading, comment handling, version parsing,
and path resolution.  All ``io/`` reader modules import helpers from here
rather than defining their own copies.

.. automodule:: pyiwfm.io.iwfm_reader
   :members:
   :undoc-members:
   :show-inheritance:

Comment Extractor
~~~~~~~~~~~~~~~~~

Extracts comments from IWFM files for preservation during roundtrip I/O.

.. automodule:: pyiwfm.io.comment_extractor
   :members:
   :undoc-members:
   :show-inheritance:

Comment Metadata
~~~~~~~~~~~~~~~~

Stores extracted comment metadata for use during file writing.

.. automodule:: pyiwfm.io.comment_metadata
   :members:
   :undoc-members:
   :show-inheritance:

Comment Writer
~~~~~~~~~~~~~~

Injects preserved comments back into written IWFM files.

.. automodule:: pyiwfm.io.comment_writer
   :members:
   :undoc-members:
   :show-inheritance:

Parametric Grid
~~~~~~~~~~~~~~~

Parametric grid interpolation utilities.

.. automodule:: pyiwfm.io.parametric_grid
   :members:
   :undoc-members:
   :show-inheritance:

HEC-DSS Module
--------------

The DSS module provides support for reading and writing HEC-DSS 7 files.

.. note::

   HEC-DSS support requires the HEC-DSS C library to be installed. Set the
   ``HECDSS_LIB`` environment variable to point to the library location.

DSS Package
~~~~~~~~~~~

.. automodule:: pyiwfm.io.dss
   :members:
   :undoc-members:
   :show-inheritance:

DSS Pathname Utilities
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: pyiwfm.io.dss.pathname
   :members:
   :undoc-members:
   :show-inheritance:

DSS Wrapper
~~~~~~~~~~~

.. automodule:: pyiwfm.io.dss.wrapper
   :members:
   :undoc-members:
   :show-inheritance:

DSS Time Series
~~~~~~~~~~~~~~~

.. automodule:: pyiwfm.io.dss.timeseries
   :members:
   :undoc-members:
   :show-inheritance:

Data Loaders
-------------

These modules provide lazy, cached access to IWFM output data (heads,
hydrographs, land-use areas).  They were moved from ``visualization.webapi``
so that CLI tools, notebooks, and scripts can use them without importing the
web viewer.

Head Data Loader
~~~~~~~~~~~~~~~~

Lazy HDF5 reader for time-varying groundwater head data.

.. automodule:: pyiwfm.io.head_loader
   :members:
   :undoc-members:
   :show-inheritance:

Hydrograph Reader
~~~~~~~~~~~~~~~~~

Parser for IWFM ``.out`` text hydrograph files.

.. automodule:: pyiwfm.io.hydrograph_reader
   :members:
   :undoc-members:
   :show-inheritance:

Hydrograph Loader
~~~~~~~~~~~~~~~~~

Lazy HDF5-backed hydrograph loader (same interface as ``IWFMHydrographReader``).

.. automodule:: pyiwfm.io.hydrograph_loader
   :members:
   :undoc-members:
   :show-inheritance:

Area Data Loader
~~~~~~~~~~~~~~~~

Lazy HDF5 reader for land-use area data and multi-type area manager.

.. automodule:: pyiwfm.io.area_loader
   :members:
   :undoc-members:
   :show-inheritance:

SQLite Cache Builder
~~~~~~~~~~~~~~~~~~~~

Pre-computes aggregates from HDF5/text loaders into a single SQLite database.

.. automodule:: pyiwfm.io.cache_builder
   :members:
   :undoc-members:
   :show-inheritance:

SQLite Cache Loader
~~~~~~~~~~~~~~~~~~~

Read-only access to the viewer SQLite cache.

.. automodule:: pyiwfm.io.cache_loader
   :members:
   :undoc-members:
   :show-inheritance:

Zone File I/O
-------------

The zones module provides readers and writers for IWFM zone definition files
(used by ZBudget) and GeoJSON zone files.

.. automodule:: pyiwfm.io.zones
   :members:
   :undoc-members:
   :show-inheritance:

Supported formats:

- **IWFM ZBudget format**: Text-based element-to-zone assignments with zone names
  and extents (horizontal/vertical)
- **GeoJSON**: Standard geospatial format with geometry and properties for
  interoperability with GIS tools
