Time Series Visualization
=========================

This page demonstrates how to visualize temporal data from
IWFM models, including groundwater levels, streamflow, and
comparison between observed and simulated values. Each example
shows the pyiwfm helper function first, then optionally the raw
matplotlib equivalent for customization.

Basic Time Series Plot
----------------------

Plot a single time series:

.. plot::
   :include-source:

   import matplotlib.pyplot as plt
   from pyiwfm.sample_models import create_sample_timeseries
   from pyiwfm.visualization.plotting import plot_timeseries

   # Create sample groundwater head time series
   ts = create_sample_timeseries(
       name="Well MW-001",
       n_years=10,
       seasonal=True,
       trend=-0.5,
       noise_level=0.05
   )

   fig, ax = plot_timeseries(ts, title='Groundwater Level', ylabel='Head (ft)')
   plt.show()

Multiple Time Series
--------------------

Compare time series from multiple locations:

.. plot::
   :include-source:

   import matplotlib.pyplot as plt
   from pyiwfm.sample_models import create_sample_timeseries_collection
   from pyiwfm.visualization.plotting import plot_timeseries_collection

   # Create collection of well time series
   collection = create_sample_timeseries_collection(n_locations=5, n_years=10)

   fig, ax = plot_timeseries_collection(collection, title='Multiple Well Hydrographs',
                                        ylabel='Groundwater Head (ft)')
   plt.show()

Time Series Statistics
----------------------

Use :func:`~pyiwfm.visualization.plotting.plot_timeseries_statistics` for ensemble
statistics with min/max or standard deviation bands:

.. plot::
   :include-source:

   import matplotlib.pyplot as plt
   from pyiwfm.sample_models import create_sample_timeseries_collection
   from pyiwfm.visualization.plotting import plot_timeseries_statistics

   collection = create_sample_timeseries_collection(n_locations=8, n_years=10)

   fig, ax = plot_timeseries_statistics(collection, band='minmax',
                                         show_individual=True,
                                         title='Ensemble Statistics',
                                         ylabel='Groundwater Head (ft)')
   plt.show()

Standard deviation band variant:

.. plot::
   :include-source:

   import matplotlib.pyplot as plt
   from pyiwfm.sample_models import create_sample_timeseries_collection
   from pyiwfm.visualization.plotting import plot_timeseries_statistics

   collection = create_sample_timeseries_collection(n_locations=8, n_years=10)

   fig, ax = plot_timeseries_statistics(collection, band='std',
                                         mean_color='darkgreen',
                                         title='Ensemble with Std Dev Bands',
                                         ylabel='Groundwater Head (ft)')
   plt.show()

Observed vs Simulated Comparison
--------------------------------

Use :func:`~pyiwfm.visualization.plotting.plot_timeseries_comparison` for
calibration plots:

.. plot::
   :include-source:

   import matplotlib.pyplot as plt
   from pyiwfm.sample_models import create_sample_timeseries
   from pyiwfm.visualization.plotting import plot_timeseries_comparison

   simulated = create_sample_timeseries(
       name="Simulated", n_years=5, seasonal=True, trend=-0.4, noise_level=0.02
   )
   observed = create_sample_timeseries(
       name="Observed", n_years=5, seasonal=True, trend=-0.6, noise_level=0.15
   )

   fig, ax = plot_timeseries_comparison(observed, simulated,
                                         title='Model Calibration',
                                         show_residuals=True,
                                         show_metrics=True)
   plt.show()

Raw matplotlib alternative for more control:

.. plot::
   :include-source:

   import matplotlib.pyplot as plt
   import numpy as np
   from pyiwfm.sample_models import create_sample_timeseries

   simulated = create_sample_timeseries(
       name="Simulated", n_years=5, seasonal=True, trend=-0.4, noise_level=0.02
   )
   observed = create_sample_timeseries(
       name="Observed", n_years=5, seasonal=True, trend=-0.6, noise_level=0.15
   )

   fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [3, 1]})

   times_obs = [t.item() for t in observed.times]
   times_sim = [t.item() for t in simulated.times]

   ax1 = axes[0]
   ax1.plot(times_obs, observed.values, 'ko', markersize=3, alpha=0.6, label='Observed')
   ax1.plot(times_sim, simulated.values, 'b-', linewidth=1.5, label='Simulated')
   ax1.set_ylabel('Head (ft)')
   ax1.set_title('Model Calibration: Observed vs Simulated')
   ax1.legend()
   ax1.grid(True, alpha=0.3)

   ax2 = axes[1]
   residuals = observed.values - simulated.values
   ax2.bar(times_obs, residuals, width=5, alpha=0.7, color='gray')
   ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
   ax2.axhline(y=np.mean(residuals), color='red', linestyle='--',
               label=f'Mean Bias: {np.mean(residuals):.2f}')
   ax2.set_ylabel('Residual (ft)')
   ax2.set_xlabel('Date')
   ax2.legend(loc='upper right')
   ax2.grid(True, alpha=0.3)

   plt.tight_layout()
   plt.show()

Streamflow Hydrograph
---------------------

Use :func:`~pyiwfm.visualization.plotting.plot_streamflow_hydrograph` for
streamflow with optional baseflow separation:

.. plot::
   :include-source:

   import matplotlib.pyplot as plt
   import numpy as np
   from pyiwfm.visualization.plotting import plot_streamflow_hydrograph

   np.random.seed(42)
   n_days = 365 * 3
   times = np.arange('2020-01-01', '2022-12-31', dtype='datetime64[D]')[:n_days]
   t = np.arange(n_days)

   baseflow = 100 + 50 * np.sin(2 * np.pi * t / 365)
   storms = np.zeros(n_days)
   storm_days = np.random.choice(n_days, 30, replace=False)
   for sd in storm_days:
       peak = np.random.uniform(200, 800)
       decay = np.exp(-np.arange(30) / 5)
       end_idx = min(sd + 30, n_days)
       storms[sd:end_idx] += peak * decay[:end_idx - sd]

   total_flow = np.maximum(baseflow + storms + np.random.normal(0, 10, n_days), 0)

   fig, ax = plot_streamflow_hydrograph(times, total_flow, baseflow=baseflow,
                                         title='Stream Hydrograph with Baseflow Separation',
                                         units='cfs')
   plt.show()

Dual Axis Comparison
--------------------

Use :func:`~pyiwfm.visualization.plotting.plot_dual_axis` to compare related
variables with different scales on two y-axes:

.. plot::
   :include-source:

   import matplotlib.pyplot as plt
   import numpy as np
   from pyiwfm.sample_models import create_sample_timeseries
   from pyiwfm.visualization.plotting import plot_dual_axis

   pumping = create_sample_timeseries(
       name="Pumping", n_years=3, seasonal=True, trend=0.2, noise_level=0.05
   )
   head = create_sample_timeseries(
       name="GW Level", n_years=3, seasonal=True, trend=-0.8, noise_level=0.03
   )

   fig, (ax1, ax2) = plot_dual_axis(
       pumping, head,
       color1='tab:red', color2='tab:blue',
       ylabel1='Pumping (AF/month)', ylabel2='Groundwater Level (ft)',
       title='Pumping and Groundwater Level Response'
   )
   plt.show()

Time Series Collection with Custom Styling
-------------------------------------------

Use :func:`~pyiwfm.visualization.plotting.plot_timeseries_collection` with
extra styling options:

.. plot::
   :include-source:

   import matplotlib.pyplot as plt
   from pyiwfm.sample_models import create_sample_timeseries_collection
   from pyiwfm.visualization.plotting import plot_timeseries_collection

   collection = create_sample_timeseries_collection(n_locations=4, n_years=5)

   fig, ax = plot_timeseries_collection(
       collection,
       title='Well Hydrographs (Custom Styling)',
       colors=['#e41a1c', '#377eb8', '#4daf4a', '#984ea3'],
       linestyles=['-', '--', '-.', ':'],
       ylabel='Head (ft)',
       grid=True,
   )
   plt.show()
