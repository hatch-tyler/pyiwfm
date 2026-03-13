Mesh Quality Visualization
==========================

Visualize element quality metrics to identify problematic areas in an IWFM mesh.

Aspect Ratio Map
-----------------

Color elements by aspect ratio to find elongated cells:

.. plot::
   :include-source:

   import matplotlib.pyplot as plt
   import numpy as np
   from pyiwfm.sample_models import create_sample_mesh
   from pyiwfm.core.mesh_quality import compute_mesh_quality
   from pyiwfm.visualization.plotting import plot_scalar_field

   mesh = create_sample_mesh(nx=12, ny=12, n_subregions=4)
   report = compute_mesh_quality(mesh)

   aspect_ratios = np.array([eq.aspect_ratio for eq in report.element_qualities])

   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

   # Spatial map
   plot_scalar_field(mesh, aspect_ratios, field_type='cell',
                     ax=ax1, cmap='YlOrRd', show_mesh=True,
                     edge_color='gray', edge_width=0.3)
   ax1.set_title(f'Aspect Ratio (mean={report.aspect_ratio_mean:.2f})')
   ax1.set_xlabel('X (feet)')
   ax1.set_ylabel('Y (feet)')

   # Histogram
   ax2.hist(aspect_ratios, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
   ax2.axvline(5.0, color='red', linestyle='--', linewidth=1.5, label='Threshold (AR=5)')
   ax2.set_xlabel('Aspect Ratio')
   ax2.set_ylabel('Count')
   ax2.set_title('Aspect Ratio Distribution')
   ax2.legend()

   plt.show()

Skewness Map
-------------

Skewness measures deviation from ideal element shape (0 = perfect, 1 = degenerate):

.. plot::
   :include-source:

   import matplotlib.pyplot as plt
   import numpy as np
   from pyiwfm.sample_models import create_sample_mesh
   from pyiwfm.core.mesh_quality import compute_mesh_quality
   from pyiwfm.visualization.plotting import plot_scalar_field

   mesh = create_sample_mesh(nx=15, ny=15, n_subregions=4)
   report = compute_mesh_quality(mesh)

   skewness = np.array([eq.skewness for eq in report.element_qualities])

   fig, ax = plot_scalar_field(mesh, skewness, field_type='cell',
                               cmap='RdYlGn_r', show_mesh=True,
                               edge_color='gray', edge_width=0.3)
   ax.set_title(f'Element Skewness (mean={report.skewness_mean:.3f})')
   ax.set_xlabel('X (feet)')
   ax.set_ylabel('Y (feet)')
   plt.show()

Minimum Angle Distribution
----------------------------

Elements with very small angles can cause numerical instability:

.. plot::
   :include-source:

   import matplotlib.pyplot as plt
   import numpy as np
   from pyiwfm.sample_models import create_sample_mesh
   from pyiwfm.core.mesh_quality import compute_mesh_quality

   mesh = create_sample_mesh(nx=20, ny=20, n_subregions=4)
   report = compute_mesh_quality(mesh)

   min_angles = [eq.min_angle for eq in report.element_qualities]
   max_angles = [eq.max_angle for eq in report.element_qualities]

   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

   ax1.hist(min_angles, bins=30, edgecolor='black', alpha=0.7, color='coral')
   ax1.axvline(15.0, color='red', linestyle='--', label='Min threshold (15°)')
   ax1.set_xlabel('Minimum Angle (degrees)')
   ax1.set_ylabel('Count')
   ax1.set_title(f'Minimum Angles (global min={report.min_angle_global:.1f}°)')
   ax1.legend()

   ax2.hist(max_angles, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
   ax2.axvline(165.0, color='red', linestyle='--', label='Max threshold (165°)')
   ax2.set_xlabel('Maximum Angle (degrees)')
   ax2.set_ylabel('Count')
   ax2.set_title(f'Maximum Angles (global max={report.max_angle_global:.1f}°)')
   ax2.legend()

   plt.show()
