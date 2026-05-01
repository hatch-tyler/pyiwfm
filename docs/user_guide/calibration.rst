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

``pyiwfm.io.simulation.messages``
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
    representative seasonal patterns for each cluster. Includes Fortran
    ``.in`` config file parsing (``read_calctyphyd_config``), time-series
    output matching the Fortran algorithm (``compute_typical_hydrographs_timeseries``),
    PEST ``.out``/``.ins`` file generation (``write_pest_output``), and
    date-range filtering. Verified byte-identical vs Fortran on C2VSimFG.

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

   # CalcTypHyd with Fortran config file (produces PEST .out/.ins files)
   pyiwfm calctyphyd --config CalcTypeHyd.in

   # Deduplicate per-layer SMP output (strip %N suffixes)
   pyiwfm iwfm2obs --deduplicate-smp GW_OUT.smp --output GW_OUT_dedup.smp

See the :doc:`/tutorials/calibration` tutorial for a complete walkthrough.

Performance
-----------

The calibration pipeline runs once per PEST iteration, so its speed is
the wall-clock bottleneck for parameter-estimation runs that drive the
model 50–500 times.

The Python implementations in ``calibration/iwfm2obs.py``,
``calibration/results_extraction.py``, and
``calibration/headall_extraction.py`` were vectorised in v2.0.0a2:

- **FE interpolation per timestep** — the inner per-layer loop
  (``for layer in range(n_layers): np.dot(coeffs, vals)``) was hoisted
  into a single matrix-vector product (``coeffs @ frame[node_indices, :]``)
  that computes all layers at once. Typical speedup: **~3× per frame**.

- **Multi-layer aggregation** (``_aggregate_layers``) — the per-timestep
  loop over (n_times) timesteps applying T-weighted averaging or
  layer-summation was replaced with a single broadcasted numpy
  expression. Typical speedup on a 365-timestep × 4-layer workload:
  **~80× on the aggregation pass alone**.

- **Composite-head computation** (``iwfm2obs_from_model``) — the
  triple-nested loop building per-(timestep, layer) Python work was
  replaced with one batched matrix-vector product per well. Typical
  speedup: **~5×**.

- **T-weighted layer-weight computation** (``compute_multilayer_weights``)
  — the per-layer dict-build for FE interpolation of layer top/bottom
  elevations and hydraulic conductivity was replaced with array slicing
  + matrix-vector products. Typical speedup: **~2-3×**.

Three engines are available for the FE-interpolation hot path
(``ResultsExtractor.extract`` and ``HeadAllExtractor.extract``):

1. **Pure-numpy (default)** — vectorised across layers per spec; ships
   with every pyiwfm install, no extra deps. Fast enough for typical
   PEST-iteration workloads (10–100 locations).

2. **Numba JIT** — install with ``pip install pyiwfm[fast-calib]``.
   Triggers a ~80ms one-time JIT compile on first call (cached in
   ``~/.numba/`` for future runs), then closes most of the gap to
   the Fortran reference implementation. Benchmarks on a synthetic
   5,000-location workload: pure-numpy 32 ms/frame, Numba 78 μs/frame
   — about **400× faster**. Recommended for >1,000-location runs
   (e.g. InSAR-pixel subsidence calibration).

3. **Fortran subprocess** — ``ResultsExtract.exe`` from the IWFM
   distribution wrapped via ``calibration.results_extraction.FortranBackend``.
   Use when the .exe is already on PATH and the workload is so heavy
   that even Numba's per-call setup cost matters (10k+ locations,
   3650+ timesteps). Black-box reference implementation; pyiwfm
   parses its output and presents the same ``ExtractionResult`` shape.

Engine selection at runtime: ``calibration.results_extraction`` and
``calibration.headall_extraction`` automatically use the Numba kernel
when the ``fast-calib`` extra is installed; no code change is needed.
The selected engine is logged at INFO level via the
``pyiwfm.calibration._kernels`` logger when the kernel first loads.
``FortranBackend`` is a separate class that callers instantiate
explicitly when they want it.
