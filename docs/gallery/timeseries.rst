Time Series Visualization
=========================

This page demonstrates how to visualize temporal data from
IWFM models, including groundwater levels, streamflow, and
comparison between observed and simulated values.

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

Display time series with statistical summaries:

.. plot::
   :include-source:

   import matplotlib.pyplot as plt
   import numpy as np
   from pyiwfm.sample_models import create_sample_timeseries_collection

   collection = create_sample_timeseries_collection(n_locations=8, n_years=10)

   fig, ax = plt.subplots(figsize=(14, 6))

   # Get time series data
   series_list = list(collection.series.values())
   times = [t.item() for t in series_list[0].times]  # Convert to Python datetime
   n_times = len(times)

   values_matrix = np.zeros((len(series_list), n_times))
   for i, ts in enumerate(series_list):
       values_matrix[i, :] = ts.values

   # Calculate statistics
   mean_vals = np.mean(values_matrix, axis=0)
   std_vals = np.std(values_matrix, axis=0)
   min_vals = np.min(values_matrix, axis=0)
   max_vals = np.max(values_matrix, axis=0)

   # Plot individual series (light)
   for ts in series_list:
       ax.plot(times, ts.values, 'lightblue', alpha=0.5, linewidth=0.8)

   # Plot envelope
   ax.fill_between(times, min_vals, max_vals, alpha=0.2, color='blue', label='Range')
   ax.fill_between(times, mean_vals - std_vals, mean_vals + std_vals,
                   alpha=0.3, color='blue', label='+/- 1 Std Dev')

   # Plot mean
   ax.plot(times, mean_vals, 'navy', linewidth=2, label='Mean')

   ax.set_ylabel('Groundwater Head (ft)')
   ax.set_title('Ensemble Statistics')
   ax.legend(loc='upper right')
   ax.grid(True, alpha=0.3)
   plt.show()

Observed vs Simulated Comparison
--------------------------------

Compare model results against observations:

.. plot::
   :include-source:

   import matplotlib.pyplot as plt
   import numpy as np
   from pyiwfm.sample_models import create_sample_timeseries

   # Create simulated time series
   simulated = create_sample_timeseries(
       name="Simulated",
       n_years=5,
       seasonal=True,
       trend=-0.4,
       noise_level=0.02
   )

   # Create observed with different characteristics
   observed = create_sample_timeseries(
       name="Observed",
       n_years=5,
       seasonal=True,
       trend=-0.6,
       noise_level=0.15
   )

   fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [3, 1]})

   # Convert times to Python datetime for plotting
   times_obs = [t.item() for t in observed.times]
   times_sim = [t.item() for t in simulated.times]

   # Main comparison plot
   ax1 = axes[0]
   ax1.plot(times_obs, observed.values, 'ko', markersize=3, alpha=0.6, label='Observed')
   ax1.plot(times_sim, simulated.values, 'b-', linewidth=1.5, label='Simulated')
   ax1.set_ylabel('Head (ft)')
   ax1.set_title('Model Calibration: Observed vs Simulated')
   ax1.legend()
   ax1.grid(True, alpha=0.3)

   # Residual plot
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

Display streamflow with characteristics common to surface water:

.. plot::
   :include-source:

   import matplotlib.pyplot as plt
   import numpy as np
   from datetime import datetime, timedelta

   # Generate synthetic streamflow
   np.random.seed(42)
   start = datetime(2020, 1, 1)
   n_days = 365 * 3
   times = [start + timedelta(days=i) for i in range(n_days)]

   # Base flow + seasonal + storm events
   t = np.arange(n_days)
   baseflow = 100 + 50 * np.sin(2 * np.pi * t / 365)

   # Add storm events
   storms = np.zeros(n_days)
   storm_days = np.random.choice(n_days, 30, replace=False)
   for sd in storm_days:
       peak = np.random.uniform(200, 800)
       decay = np.exp(-np.arange(30) / 5)
       end_idx = min(sd + 30, n_days)
       storms[sd:end_idx] += peak * decay[:end_idx - sd]

   flow = baseflow + storms + np.random.normal(0, 10, n_days)
   flow = np.maximum(flow, 0)

   fig, ax = plt.subplots(figsize=(14, 6))

   # Plot with fill
   ax.fill_between(times, 0, flow, alpha=0.5, color='steelblue')
   ax.plot(times, flow, 'b-', linewidth=0.5)

   ax.set_xlabel('Date')
   ax.set_ylabel('Streamflow (cfs)')
   ax.set_title('Streamflow Hydrograph')
   ax.grid(True, alpha=0.3)
   ax.set_ylim(bottom=0)

   # Add annotation for peak
   peak_idx = np.argmax(flow)
   ax.annotate(f'Peak: {flow[peak_idx]:.0f} cfs',
               xy=(times[peak_idx], flow[peak_idx]),
               xytext=(times[peak_idx], flow[peak_idx] + 100),
               arrowprops=dict(arrowstyle='->', color='red'),
               fontsize=10)

   plt.show()

Dual Axis Comparison
--------------------

Compare related variables with different scales:

.. plot::
   :include-source:

   import matplotlib.pyplot as plt
   import numpy as np
   from datetime import datetime, timedelta

   # Generate synthetic data
   np.random.seed(42)
   start = datetime(2020, 1, 1)
   n_months = 36
   times = [start + timedelta(days=30*i) for i in range(n_months)]

   # Groundwater level and pumping
   t = np.arange(n_months)
   pumping = 1000 + 300 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 50, n_months)
   pumping = np.maximum(pumping, 0)

   # Head responds to pumping with lag
   head = 50 - 0.5 * t
   for i in range(1, n_months):
       head[i] -= 0.005 * pumping[i-1]
   head += np.random.normal(0, 1, n_months)

   fig, ax1 = plt.subplots(figsize=(14, 6))

   # Pumping on primary axis
   color1 = 'tab:red'
   ax1.set_xlabel('Date')
   ax1.set_ylabel('Pumping (AF/month)', color=color1)
   bars = ax1.bar(times, pumping, width=25, alpha=0.7, color=color1, label='Pumping')
   ax1.tick_params(axis='y', labelcolor=color1)

   # Head on secondary axis
   ax2 = ax1.twinx()
   color2 = 'tab:blue'
   ax2.set_ylabel('Groundwater Level (ft)', color=color2)
   line, = ax2.plot(times, head, color=color2, linewidth=2, marker='o', markersize=4)
   ax2.tick_params(axis='y', labelcolor=color2)

   # Combined legend
   ax1.legend([bars, line], ['Pumping', 'GW Level'], loc='upper right')

   plt.title('Pumping and Groundwater Level Response')
   fig.tight_layout()
   plt.show()
