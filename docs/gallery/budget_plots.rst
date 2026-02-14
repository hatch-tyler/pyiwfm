Water Budget Visualization
==========================

This page demonstrates how to visualize water budget components
from IWFM models, including bar charts, pie charts, stacked plots,
and water balance diagrams.

Bar Chart Budget
----------------

Display budget components as a horizontal bar chart:

.. plot::
   :include-source:

   import matplotlib.pyplot as plt
   import numpy as np
   from pyiwfm.sample_models import create_sample_budget_data

   budget = create_sample_budget_data()

   fig, ax = plt.subplots(figsize=(10, 8))

   # Combine inflows and outflows
   components = []
   values = []
   colors = []

   for name, val in budget['Inflows'].items():
       components.append(name)
       values.append(val)
       colors.append('steelblue')

   for name, val in budget['Outflows'].items():
       components.append(name)
       values.append(val)
       colors.append('coral')

   for name, val in budget['Storage Change'].items():
       components.append(name)
       values.append(val)
       colors.append('gold' if val >= 0 else 'orange')

   # Create horizontal bar chart
   y_pos = np.arange(len(components))
   ax.barh(y_pos, values, color=colors, edgecolor='black', linewidth=0.5)

   # Add zero line
   ax.axvline(x=0, color='black', linewidth=1)

   # Labels
   ax.set_yticks(y_pos)
   ax.set_yticklabels(components)
   ax.set_xlabel('Flow (AF/year)')
   ax.set_title('Annual Water Budget Components')

   # Add value labels
   for i, (v, c) in enumerate(zip(values, colors)):
       offset = 500 if v >= 0 else -500
       ha = 'left' if v >= 0 else 'right'
       ax.text(v + offset, i, f'{v:,.0f}', va='center', ha=ha, fontsize=9)

   ax.grid(True, axis='x', alpha=0.3)
   plt.tight_layout()
   plt.show()

Pie Chart Budget
----------------

Show budget components as pie charts:

.. plot::
   :include-source:

   import matplotlib.pyplot as plt
   import numpy as np
   from pyiwfm.sample_models import create_sample_budget_data

   budget = create_sample_budget_data()

   fig, axes = plt.subplots(1, 2, figsize=(14, 6))

   # Inflows pie
   ax1 = axes[0]
   inflow_names = list(budget['Inflows'].keys())
   inflow_values = list(budget['Inflows'].values())
   colors1 = plt.cm.Blues(np.linspace(0.3, 0.8, len(inflow_names)))

   wedges1, texts1, autotexts1 = ax1.pie(
       inflow_values, labels=inflow_names, autopct='%1.1f%%',
       colors=colors1, startangle=90, explode=[0.02] * len(inflow_names)
   )
   ax1.set_title(f'Inflows\n(Total: {sum(inflow_values):,.0f} AF/year)')

   # Outflows pie
   ax2 = axes[1]
   outflow_names = list(budget['Outflows'].keys())
   outflow_values = [-v for v in budget['Outflows'].values()]  # Make positive
   colors2 = plt.cm.Oranges(np.linspace(0.3, 0.8, len(outflow_names)))

   wedges2, texts2, autotexts2 = ax2.pie(
       outflow_values, labels=outflow_names, autopct='%1.1f%%',
       colors=colors2, startangle=90, explode=[0.02] * len(outflow_names)
   )
   ax2.set_title(f'Outflows\n(Total: {sum(outflow_values):,.0f} AF/year)')

   plt.tight_layout()
   plt.show()

Stacked Budget Over Time
------------------------

Show budget evolution over time:

.. plot::
   :include-source:

   import matplotlib.pyplot as plt
   import numpy as np
   from datetime import datetime

   # Generate synthetic time-varying budget
   years = list(range(2010, 2025))
   n_years = len(years)

   # Base values with trends
   np.random.seed(42)
   recharge = 15000 + np.random.normal(0, 1500, n_years) + np.arange(n_years) * 100
   stream_seepage = 8500 + np.random.normal(0, 800, n_years)
   subsurface = 5200 + np.random.normal(0, 500, n_years)

   pumping = 18500 + np.random.normal(0, 1000, n_years) + np.arange(n_years) * 200
   baseflow = 7200 + np.random.normal(0, 600, n_years)
   et = 3100 + np.random.normal(0, 300, n_years)

   fig, ax = plt.subplots(figsize=(14, 7))

   # Stack inflows (positive)
   ax.fill_between(years, 0, recharge, alpha=0.8, label='Recharge', color='steelblue')
   ax.fill_between(years, recharge, recharge + stream_seepage, alpha=0.8,
                   label='Stream Seepage', color='cornflowerblue')
   ax.fill_between(years, recharge + stream_seepage,
                   recharge + stream_seepage + subsurface, alpha=0.8,
                   label='Subsurface Inflow', color='lightsteelblue')

   # Stack outflows (negative)
   ax.fill_between(years, 0, -pumping, alpha=0.8, label='Pumping', color='coral')
   ax.fill_between(years, -pumping, -pumping - baseflow, alpha=0.8,
                   label='Stream Baseflow', color='lightsalmon')
   ax.fill_between(years, -pumping - baseflow, -pumping - baseflow - et, alpha=0.8,
                   label='GW ET', color='peachpuff')

   # Zero line
   ax.axhline(y=0, color='black', linewidth=1)

   # Net change line
   total_in = recharge + stream_seepage + subsurface
   total_out = pumping + baseflow + et
   net = total_in - total_out
   ax.plot(years, net, 'k--', linewidth=2, marker='o', label='Net Change')

   ax.set_xlabel('Year')
   ax.set_ylabel('Flow (AF/year)')
   ax.set_title('Annual Water Budget (2010-2024)')
   ax.legend(loc='upper left', ncol=2)
   ax.grid(True, alpha=0.3)

   plt.tight_layout()
   plt.show()

Water Balance Diagram
---------------------

Create a conceptual water balance diagram:

.. plot::
   :include-source:

   import matplotlib.pyplot as plt
   import matplotlib.patches as mpatches
   from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
   import numpy as np

   fig, ax = plt.subplots(figsize=(14, 10))

   # Main aquifer box
   aquifer = FancyBboxPatch((3, 2), 8, 4, boxstyle="round,pad=0.05",
                            facecolor='lightblue', edgecolor='blue', linewidth=2)
   ax.add_patch(aquifer)
   ax.text(7, 4, 'AQUIFER\nStorage Change: +500 AF/yr', ha='center', va='center',
           fontsize=12, fontweight='bold')

   # Inflow arrows and labels
   inflows = [
       ('Recharge\n15,000', (7, 8), (7, 6.2)),
       ('Stream\nSeepage\n8,500', (0.5, 5), (2.8, 4.5)),
       ('Subsurface\nInflow\n5,200', (0.5, 3), (2.8, 3.5)),
   ]

   for label, start, end in inflows:
       arrow = FancyArrowPatch(start, end, arrowstyle='->', mutation_scale=20,
                               color='green', linewidth=3)
       ax.add_patch(arrow)
       ax.text(start[0], start[1] + 0.5, label, ha='center', va='bottom',
               fontsize=10, color='green')

   # Outflow arrows and labels
   outflows = [
       ('Pumping\n-18,500', (7, 1.8), (7, -0.5)),
       ('Stream\nBaseflow\n-7,200', (11.2, 4.5), (13.5, 5)),
       ('GW ET\n-3,100', (11.2, 3.5), (13.5, 3)),
   ]

   for label, start, end in outflows:
       arrow = FancyArrowPatch(start, end, arrowstyle='->', mutation_scale=20,
                               color='red', linewidth=3)
       ax.add_patch(arrow)
       ax.text(end[0] + 0.3, end[1], label, ha='left', va='center',
               fontsize=10, color='red')

   # Stream (simplified)
   stream_x = [13, 13.5, 14, 13.5, 14]
   stream_y = [6, 5.5, 5, 4, 3.5]
   ax.plot(stream_x, stream_y, 'b-', linewidth=4, alpha=0.5)
   ax.text(13.5, 6.5, 'Stream', ha='center', fontsize=10, color='blue')

   # Land surface
   ax.plot([0, 14], [6.5, 6.5], 'g-', linewidth=2)
   ax.fill_between([0, 14], [6.5, 6.5], [7, 7], color='green', alpha=0.3)
   ax.text(2, 6.8, 'Land Surface', fontsize=10, color='darkgreen')

   # Summary box
   summary = '''Water Balance Summary
   ─────────────────────
   Total Inflow:  28,700 AF/yr
   Total Outflow: 28,800 AF/yr
   Net Change:      -100 AF/yr
   '''
   ax.text(0.5, 0.5, summary, fontsize=10, family='monospace',
           bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange'))

   ax.set_xlim(-1, 15)
   ax.set_ylim(-1, 9)
   ax.set_aspect('equal')
   ax.axis('off')
   ax.set_title('Conceptual Water Balance Diagram', fontsize=14, fontweight='bold')

   plt.tight_layout()
   plt.show()

Zone Budget Comparison
----------------------

Compare budgets across different zones:

.. plot::
   :include-source:

   import matplotlib.pyplot as plt
   import numpy as np

   # Synthetic zone budget data
   zones = ['Zone 1\n(Urban)', 'Zone 2\n(Agricultural)', 'Zone 3\n(Native)',
            'Zone 4\n(Mixed)']
   n_zones = len(zones)

   # Budget components by zone
   np.random.seed(42)
   recharge = [5000, 8000, 3000, 4000]
   pumping = [8000, 12000, 1000, 5000]
   stream = [2000, 3000, 1500, 2500]
   et = [1000, 2000, 1800, 1200]

   x = np.arange(n_zones)
   width = 0.2

   fig, ax = plt.subplots(figsize=(12, 7))

   bars1 = ax.bar(x - 1.5*width, recharge, width, label='Recharge', color='steelblue')
   bars2 = ax.bar(x - 0.5*width, [-p for p in pumping], width, label='Pumping', color='coral')
   bars3 = ax.bar(x + 0.5*width, stream, width, label='Stream Exchange', color='teal')
   bars4 = ax.bar(x + 1.5*width, [-e for e in et], width, label='ET', color='gold')

   ax.axhline(y=0, color='black', linewidth=0.5)

   ax.set_ylabel('Flow (AF/year)')
   ax.set_title('Zone Budget Comparison')
   ax.set_xticks(x)
   ax.set_xticklabels(zones)
   ax.legend()
   ax.grid(True, axis='y', alpha=0.3)

   plt.tight_layout()
   plt.show()

Monthly Budget Pattern
----------------------

Show seasonal variation in budget components:

.. plot::
   :include-source:

   import matplotlib.pyplot as plt
   import numpy as np

   months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
             'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

   # Seasonal patterns
   recharge = [2000, 2500, 2000, 1500, 500, 200, 100, 100, 300, 800, 1500, 2000]
   pumping = [800, 900, 1200, 1800, 2500, 3000, 3200, 3000, 2500, 1500, 900, 800]
   stream = [1000, 1200, 1100, 800, 500, 300, 200, 200, 400, 600, 800, 1000]
   et = [200, 250, 400, 600, 900, 1100, 1200, 1100, 800, 500, 300, 200]

   fig, ax = plt.subplots(figsize=(14, 6))

   x = np.arange(12)
   width = 0.2

   ax.bar(x - 1.5*width, recharge, width, label='Recharge', color='steelblue')
   ax.bar(x - 0.5*width, pumping, width, label='Pumping', color='coral')
   ax.bar(x + 0.5*width, stream, width, label='Stream Seepage', color='teal')
   ax.bar(x + 1.5*width, et, width, label='GW ET', color='gold')

   ax.set_ylabel('Flow (AF/month)')
   ax.set_title('Monthly Budget Components')
   ax.set_xticks(x)
   ax.set_xticklabels(months)
   ax.legend(loc='upper right')
   ax.grid(True, axis='y', alpha=0.3)

   # Add season backgrounds
   ax.axvspan(-0.5, 1.5, alpha=0.1, color='blue', label='_Winter')
   ax.axvspan(1.5, 4.5, alpha=0.1, color='green', label='_Spring')
   ax.axvspan(4.5, 7.5, alpha=0.1, color='yellow', label='_Summer')
   ax.axvspan(7.5, 10.5, alpha=0.1, color='orange', label='_Fall')
   ax.axvspan(10.5, 11.5, alpha=0.1, color='blue')

   plt.tight_layout()
   plt.show()

Cumulative Budget
-----------------

Show cumulative budget over simulation period:

.. plot::
   :include-source:

   import matplotlib.pyplot as plt
   import numpy as np
   from datetime import datetime, timedelta

   # Generate monthly data
   start = datetime(2015, 1, 1)
   n_months = 120  # 10 years
   dates = [start + timedelta(days=30*i) for i in range(n_months)]

   np.random.seed(42)
   t = np.arange(n_months)

   # Monthly flows
   recharge = 1200 + 500 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 100, n_months)
   pumping = 1500 + 300 * np.sin(2 * np.pi * (t + 6) / 12) + np.random.normal(0, 80, n_months)

   net_monthly = recharge - pumping
   cumulative = np.cumsum(net_monthly)

   fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

   # Monthly net
   ax1 = axes[0]
   colors = ['green' if v >= 0 else 'red' for v in net_monthly]
   ax1.bar(dates, net_monthly, width=25, color=colors, alpha=0.7)
   ax1.axhline(y=0, color='black', linewidth=0.5)
   ax1.set_ylabel('Monthly Net (AF)')
   ax1.set_title('Monthly Net Budget')
   ax1.grid(True, alpha=0.3)

   # Cumulative
   ax2 = axes[1]
   ax2.fill_between(dates, 0, cumulative, where=(cumulative >= 0),
                    color='green', alpha=0.3, label='Surplus')
   ax2.fill_between(dates, 0, cumulative, where=(cumulative < 0),
                    color='red', alpha=0.3, label='Deficit')
   ax2.plot(dates, cumulative, 'k-', linewidth=1.5)
   ax2.axhline(y=0, color='black', linewidth=0.5)
   ax2.set_xlabel('Date')
   ax2.set_ylabel('Cumulative (AF)')
   ax2.set_title('Cumulative Storage Change')
   ax2.legend()
   ax2.grid(True, alpha=0.3)

   plt.tight_layout()
   plt.show()
