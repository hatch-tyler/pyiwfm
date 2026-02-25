Calibration
===========

The calibration modules provide tools for observation well clustering,
time interpolation of simulated heads to observation times, typical
hydrograph computation, and publication-quality calibration figures.

.. contents:: Table of Contents
   :local:
   :depth: 2

IWFM2OBS
--------

Interpolate simulated heads to observation timestamps and compute
multi-layer transmissivity-weighted composite heads.  Mirrors the
Fortran IWFM2OBS utility.

.. automodule:: pyiwfm.calibration.iwfm2obs
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
