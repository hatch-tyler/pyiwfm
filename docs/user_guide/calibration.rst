Calibration
===========

pyiwfm provides a suite of calibration tools that mirror and extend
IWFM's Fortran utilities (IWFM2OBS, CalcTypHyd) with additional
capabilities for observation well clustering and publication-quality
calibration figures.

Modules
-------

``pyiwfm.io.smp``
    Read and write SMP (Sample/Bore) observation files. The SMP format
    is IWFM's standard observation file with bore ID, date/time, value,
    and optional exclusion flags.

``pyiwfm.io.simulation_messages``
    Parse ``SimulationMessages.out`` to extract warnings, errors, and
    their spatial locations (node, element, reach, layer IDs via regex).

``pyiwfm.calibration.iwfm2obs``
    Linearly interpolate simulated time series to observation timestamps.
    For multi-layer wells, compute transmissivity-weighted composite heads
    using finite element shape functions from ``core/interpolation.py``.

``pyiwfm.calibration.clustering``
    NumPy-only fuzzy c-means clustering of observation wells. Features
    combine normalized spatial coordinates with temporal characteristics
    (amplitude, trend slope, seasonal strength). Configurable
    spatial/temporal weighting.

``pyiwfm.calibration.calctyphyd``
    Compute typical hydrograph curves by cluster. Seasonal averaging,
    de-meaning per well, and membership-weighted combination produce
    representative seasonal patterns for each cluster.

``pyiwfm.visualization.calibration_plots``
    Publication-quality composite figures: 1:1 observed-vs-simulated
    plots, residual histograms, hydrograph comparison panels, metrics
    tables, water budget summaries, cluster maps, and typical hydrograph
    curves. All use the ``pyiwfm-publication.mplstyle`` style sheet.

``pyiwfm.visualization.plotting``
    Individual plot functions: ``plot_one_to_one()`` for scatter plots
    with identity line, regression, and metrics text box;
    ``plot_spatial_bias()`` for diverging-colorbar bias maps with mesh
    background.

CLI Commands
------------

.. code-block:: bash

   # Interpolate simulated heads to observation times
   pyiwfm iwfm2obs --obs observed.smp --sim simulated.smp --output interp.smp

   # Compute typical hydrographs
   pyiwfm calctyphyd --water-levels wl.smp --weights weights.txt --output typhyd.smp

See the :doc:`/tutorials/calibration` tutorial for a complete walkthrough.
