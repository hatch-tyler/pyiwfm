Calibration
===========

The calibration modules provide tools for observation well clustering,
time interpolation of simulated heads to observation times, typical
hydrograph computation, model file discovery, multi-layer observation well
processing, and publication-quality calibration figures.

.. contents:: Table of Contents
   :local:
   :depth: 2

IWFM2OBS
--------

Interpolate simulated heads to observation timestamps and compute
multi-layer transmissivity-weighted composite heads.  Mirrors the
Fortran IWFM2OBS utility.  Includes the integrated ``iwfm2obs_from_model()``
workflow that auto-discovers ``.out`` files from the simulation main file.

.. automodule:: pyiwfm.calibration.iwfm2obs
   :members:
   :undoc-members:
   :show-inheritance:

Model File Discovery
---------------------

Parse an IWFM simulation main file to auto-discover hydrograph ``.out``
file paths and observation metadata (bore IDs, layers, coordinates).
Ports the model-file-discovery logic from the old Fortran IWFM2OBS program.

.. automodule:: pyiwfm.calibration.model_file_discovery
   :members:
   :undoc-members:
   :show-inheritance:

Observation Well Specification
-------------------------------

Read observation well specification files for multi-layer target
processing with screen depth intervals and element locations.

.. automodule:: pyiwfm.calibration.obs_well_spec
   :members:
   :undoc-members:
   :show-inheritance:

Typical Hydrographs (CalcTypHyd)
---------------------------------

Compute typical hydrograph curves by cluster using seasonal averaging
and membership-weighted combination.

.. automodule:: pyiwfm.calibration.calctyphyd
   :members:
   :undoc-members:
   :show-inheritance:

Fuzzy C-Means Clustering
-------------------------

NumPy-only fuzzy c-means clustering of observation wells using combined
spatial and temporal features.

.. automodule:: pyiwfm.calibration.clustering
   :members:
   :undoc-members:
   :show-inheritance:

Calibration Plots
-----------------

Publication-quality composite figures for calibration reports: 1:1 plots,
residual histograms, hydrograph panels, metrics tables, water budget
summaries, cluster maps, and typical hydrograph curves.

.. automodule:: pyiwfm.visualization.calibration_plots
   :members:
   :undoc-members:
   :show-inheritance:
