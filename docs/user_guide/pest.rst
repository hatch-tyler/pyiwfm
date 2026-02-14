PEST++ Calibration
==================

pyiwfm provides a complete interface for calibrating IWFM models using
`PEST++ <https://github.com/usgs/pestpp>`_, the USGS parameter estimation
suite. The interface handles parameter definition, observation management,
template/instruction file generation, geostatistics, ensemble analysis,
and post-processing.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
--------

The PEST++ integration is built around the ``IWFMPestHelper`` class, which
coordinates all calibration components:

.. code-block:: python

    from pyiwfm.runner import IWFMPestHelper, IWFMParameterType
    from datetime import datetime

    # Create a calibration setup
    helper = IWFMPestHelper(
        pest_dir="./pest_setup",
        case_name="iwfm_cal",
        model_dir="./model",
    )

    # Add parameters
    helper.add_zone_parameters(
        param_type=IWFMParameterType.HORIZONTAL_K,
        zones=[1, 2, 3],
        layer=1,
        bounds=(0.01, 100.0),
    )

    # Add observations
    helper.add_head_observations(
        well_id="MW-01",
        x=500000.0, y=4200000.0,
        times=[datetime(2020, 1, 1), datetime(2020, 7, 1)],
        values=[120.5, 118.3],
    )

    # Configure and build
    helper.set_svd(maxsing=50)
    helper.set_model_command("python forward_run.py")
    pst_path = helper.build()

Architecture
------------

The calibration interface consists of these components:

- **IWFMPestHelper**: Main entry point coordinating all components
- **IWFMParameterManager**: Parameter registration and group management
- **IWFMObservationManager**: Observation targets with weights
- **IWFMTemplateManager**: Template file (.tpl) generation
- **IWFMInstructionManager**: Instruction file (.ins) generation
- **GeostatManager**: Variogram modeling and spatial correlation
- **IWFMEnsembleManager**: Prior/posterior ensemble generation
- **PestPostProcessor**: Calibration result analysis

Parameter Types
---------------

pyiwfm defines IWFM-specific parameter types through the ``IWFMParameterType`` enum:

**Aquifer Parameters:**

- ``HORIZONTAL_K`` - Horizontal hydraulic conductivity
- ``VERTICAL_K`` - Vertical hydraulic conductivity
- ``SPECIFIC_STORAGE`` - Specific storage
- ``SPECIFIC_YIELD`` - Specific yield

**Stream Parameters:**

- ``STREAMBED_K`` - Streambed hydraulic conductivity
- ``STREAMBED_THICKNESS`` - Streambed thickness
- ``STREAM_WIDTH`` - Stream channel width

**Lake Parameters:**

- ``LAKEBED_K`` - Lakebed hydraulic conductivity

**Root Zone Parameters:**

- ``CROP_COEFFICIENT`` - Crop coefficient
- ``IRRIGATION_EFFICIENCY`` - Irrigation efficiency
- ``ROOT_DEPTH`` - Root zone depth
- ``SOIL_MOISTURE_CAPACITY`` - Soil moisture capacity

**Flux Multipliers:**

- ``PUMPING_MULT`` - Pumping rate multiplier
- ``RECHARGE_MULT`` - Recharge rate multiplier
- ``DIVERSION_MULT`` - Diversion rate multiplier
- ``PRECIP_MULT`` - Precipitation multiplier
- ``ET_MULT`` - Evapotranspiration multiplier

Parameterization Strategies
---------------------------

Zone-Based Parameters
~~~~~~~~~~~~~~~~~~~~~

Assign one parameter value per zone or subregion:

.. code-block:: python

    params = helper.add_zone_parameters(
        param_type=IWFMParameterType.HORIZONTAL_K,
        zones=[1, 2, 3],
        layer=1,
        bounds=(0.01, 100.0),
        group="aquifer_k",
    )

Pilot Point Parameters
~~~~~~~~~~~~~~~~~~~~~~

Spatially distributed parameters interpolated via kriging:

.. code-block:: python

    points = [(100, 200), (300, 400), (500, 600)]
    params = helper.add_pilot_points(
        param_type=IWFMParameterType.HORIZONTAL_K,
        points=points,
        layer=1,
    )

Multiplier Parameters
~~~~~~~~~~~~~~~~~~~~~

Adjust existing model values by a factor:

.. code-block:: python

    params = helper.add_multiplier(
        param_type=IWFMParameterType.PUMPING_MULT,
        bounds=(0.8, 1.2),
    )

Stream and Root Zone Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Stream parameters by reach
    params = helper.add_stream_parameters(
        param_type=IWFMParameterType.STREAMBED_K,
        reaches=[1, 2, 3],
    )

    # Root zone parameters by land use type
    params = helper.add_rootzone_parameters(
        param_type=IWFMParameterType.CROP_COEFFICIENT,
        land_use_types=["corn", "alfalfa", "pasture"],
    )

Observation Types
-----------------

Head Observations
~~~~~~~~~~~~~~~~~

.. code-block:: python

    obs = helper.add_head_observations(
        well_id="MW-01",
        x=500000.0,
        y=4200000.0,
        times=[datetime(2020, 1, 1), datetime(2020, 7, 1)],
        values=[120.5, 118.3],
        weight=1.0,
        group="heads",
    )

Streamflow Observations
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    obs = helper.add_streamflow_observations(
        gage_id="USGS-11303500",
        reach_id=5,
        times=[datetime(2020, 1, 1), datetime(2020, 7, 1)],
        values=[500.0, 350.0],
        weight=0.5,
        group="flows",
    )

Configuration
-------------

SVD Truncation
~~~~~~~~~~~~~~

Singular Value Decomposition controls the parameter solution space:

.. code-block:: python

    helper.set_svd(
        maxsing=50,       # Maximum singular values
        eigthresh=1e-6,   # Eigenvalue threshold
    )

Regularization
~~~~~~~~~~~~~~

Add prior information to constrain the calibration:

.. code-block:: python

    helper.set_regularization(
        reg_type="preferred_homogeneity",
        weight=2.0,
    )

PEST++ Options
~~~~~~~~~~~~~~

Pass any PEST++ option directly:

.. code-block:: python

    helper.set_pestpp_options(
        ies_num_reals=100,
        ies_lambda_mults="0.1,1,10",
        ies_subset_size=10,
    )

Building and Running
--------------------

.. code-block:: python

    # Build generates all files
    pst_path = helper.build()

    # Summary of the setup
    summary = helper.summary()
    print(f"Parameters: {summary['n_parameters']}")
    print(f"Observations: {summary['n_observations']}")

    # Run PEST++ (requires pestpp-ies or pestpp-glm on PATH)
    helper.run_pestpp()

The ``build()`` method creates:

- Control file (``.pst``)
- Template files (``.tpl``)
- Instruction files (``.ins``)
- Forward run script (``forward_run.py``)
- Template and instruction subdirectories

Geostatistics
-------------

The ``GeostatManager`` supports variogram modeling for pilot point
parameterization and spatially correlated ensemble generation.

.. code-block:: python

    from pyiwfm.runner import GeostatManager, Variogram, VariogramType

    geo = GeostatManager()

    # Define a variogram
    vario = Variogram(
        vtype=VariogramType.SPHERICAL,
        sill=1.0,
        range_a=5000.0,
        nugget=0.1,
    )

    # Compute empirical variogram from data
    from pyiwfm.runner import compute_empirical_variogram
    empirical = compute_empirical_variogram(
        x=x_coords, y=y_coords, values=measured_values,
        n_lags=15, max_lag=10000.0,
    )

Ensemble Management
-------------------

The ``IWFMEnsembleManager`` supports pestpp-ies ensemble workflows:

.. code-block:: python

    from pyiwfm.runner import IWFMEnsembleManager, Parameter, IWFMParameterType

    params = [
        Parameter(name="hk_z1", param_type=IWFMParameterType.HORIZONTAL_K,
                  initial_value=1.0, lower_bound=0.1, upper_bound=10.0),
        Parameter(name="sy_z1", param_type=IWFMParameterType.SPECIFIC_YIELD,
                  initial_value=0.15, lower_bound=0.05, upper_bound=0.3),
    ]

    em = IWFMEnsembleManager(parameters=params)

    # Generate prior ensemble
    prior = em.generate_prior_ensemble(n_realizations=100, seed=42)

    # Write for PEST++
    em.write_parameter_ensemble(prior, "prior.csv")

    # After calibration, load posterior
    posterior = em.load_posterior_ensemble("posterior.csv")

    # Analyze uncertainty reduction
    reduction = em.compute_reduction_factor(prior, posterior)

    # Get ensemble statistics
    stats = em.analyze_ensemble(posterior)
    print(f"Mean: {stats.mean}")
    print(f"Std: {stats.std}")

Post-Processing
---------------

The ``PestPostProcessor`` loads and analyzes PEST++ output files:

.. code-block:: python

    from pyiwfm.runner import PestPostProcessor

    pp = PestPostProcessor(pest_dir="./pest_setup", case_name="iwfm_cal")
    results = pp.load_results()

    # Fit statistics
    stats = results.fit_statistics()
    print(f"RMSE: {stats['rmse']:.3f}")
    print(f"R-squared: {stats['r_squared']:.3f}")
    print(f"Nash-Sutcliffe: {stats['nse']:.3f}")

    # Per-group statistics
    head_stats = results.fit_statistics(group="heads")
    flow_stats = results.fit_statistics(group="flows")

    # Sensitivity analysis
    if results.sensitivities:
        top = results.sensitivities.most_sensitive(5)
        for name, css in top:
            print(f"  {name}: {css:.2f}")

    # Export calibrated parameters
    pp.export_calibrated_parameters("calibrated.csv", format="csv")

    # Summary report
    report = pp.summary_report()
    print(report)

Complete Workflow Example
-------------------------

.. code-block:: python

    from datetime import datetime
    from pyiwfm.runner import (
        IWFMPestHelper,
        IWFMParameterType,
        PestPostProcessor,
        IWFMEnsembleManager,
    )

    # 1. Setup
    helper = IWFMPestHelper(
        pest_dir="./pest_cal",
        case_name="c2vsim_cal",
        model_dir="./c2vsim_model",
    )

    # 2. Parameters
    helper.add_zone_parameters("hk", zones=range(1, 22), layer=1)
    helper.add_zone_parameters("sy", zones=range(1, 22), layer=1)
    helper.add_multiplier("pumping_mult", bounds=(0.5, 1.5))

    # 3. Observations
    for well_id, x, y, times, values in well_data:
        helper.add_head_observations(well_id, x, y, times, values)

    for gage_id, reach, times, values in gage_data:
        helper.add_streamflow_observations(gage_id, reach, times, values)

    # 4. Configure
    helper.set_svd(maxsing=100)
    helper.set_pestpp_options(ies_num_reals=200)

    # 5. Build and run
    helper.build()
    helper.run_pestpp()

    # 6. Post-process
    pp = PestPostProcessor("./pest_cal", "c2vsim_cal")
    results = pp.load_results()
    print(pp.summary_report())
