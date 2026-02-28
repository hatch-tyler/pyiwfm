Calibration
===========

pyiwfm provides a suite of calibration tools that mirror and extend
IWFM's Fortran utilities (IWFM2OBS, CalcTypHyd) with additional
capabilities for model file discovery, multi-layer observation well
processing, observation well clustering, and publication-quality
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

``pyiwfm.io.hydrograph_reader``
    Read IWFM hydrograph output ``.out`` files (GW, stream, subsidence,
    tile drain). ``IWFMHydrographReader`` parses the ``*``-prefixed header
    for column metadata and loads time series into NumPy arrays.
    ``get_columns_as_smp_dict()`` bridges ``.out`` data to the
    interpolation pipeline.

``pyiwfm.calibration.model_file_discovery``
    Parse an IWFM simulation main file to auto-discover hydrograph
    ``.out`` file paths and observation metadata (bore IDs, layers,
    coordinates). Ports the discovery logic from the old Fortran IWFM2OBS
    program. Uses ``iwfm_reader`` utilities for comment handling and
    path resolution.

``pyiwfm.calibration.obs_well_spec``
    Read observation well specification files for multi-layer target
    processing. Each well has a name, coordinates, element ID, and
    screen top/bottom depths.

``pyiwfm.calibration.iwfm2obs``
    Linearly interpolate simulated time series to observation timestamps.
    For multi-layer wells, compute transmissivity-weighted composite heads
    using finite element shape functions from ``core/interpolation.py``.
    ``iwfm2obs_from_model()`` provides an integrated workflow that
    auto-discovers ``.out`` files, reads them directly, interpolates, and
    optionally computes multi-layer T-weighted averages.

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

   # Explicit SMP mode: interpolate simulated heads to observation times
   pyiwfm iwfm2obs --obs observed.smp --sim simulated.smp --output interp.smp

   # Model discovery mode: auto-discover .out files from simulation main file
   pyiwfm iwfm2obs --model C2VSimFG.in --obs-gw gw_obs.smp --output-gw gw_out.smp

   # Model mode with stream observations
   pyiwfm iwfm2obs --model C2VSimFG.in \
       --obs-gw gw_obs.smp --output-gw gw_out.smp \
       --obs-stream str_obs.smp --output-stream str_out.smp

   # Model mode with multi-layer target processing
   pyiwfm iwfm2obs --model C2VSimFG.in \
       --obs-gw gw_obs.smp --output-gw gw_out.smp \
       --well-spec obs_wells.txt \
       --multilayer-out GW_MultiLayer.out \
       --multilayer-ins GWHMultiLayer.ins

   # Compute typical hydrographs
   pyiwfm calctyphyd --water-levels wl.smp --weights weights.txt --output typhyd.smp

See the :doc:`/tutorials/calibration` tutorial for a complete walkthrough.
