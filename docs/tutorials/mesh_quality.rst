Mesh Quality Analysis
=====================

This tutorial demonstrates how to compute and interpret mesh quality metrics
using :func:`~pyiwfm.core.mesh_quality.compute_mesh_quality`.

.. contents:: Table of Contents
   :local:
   :depth: 2

Learning Objectives
-------------------

- Compute element-level quality metrics (aspect ratio, skewness, angles)
- Identify problematic elements that may cause numerical issues
- Visualize mesh quality spatially and as histograms

Prerequisites
-------------

.. code-block:: bash

    pip install pyiwfm[all]

Step 1: Load a Mesh
--------------------

.. code-block:: python

    from pyiwfm.io import load_complete_model

    model = load_complete_model("Simulation/Simulation.in")
    grid = model.grid
    print(f"Mesh: {grid.n_nodes} nodes, {grid.n_elements} elements")

Or create a sample mesh for experimentation:

.. code-block:: python

    from pyiwfm.sample_models import create_sample_mesh

    grid = create_sample_mesh(nx=15, ny=15, n_subregions=4)

Step 2: Compute Mesh Quality
------------------------------

.. code-block:: python

    from pyiwfm.core.mesh_quality import compute_mesh_quality

    report = compute_mesh_quality(grid)

    print(f"Elements: {report.n_elements}")
    print(f"  Triangles: {report.n_triangles}")
    print(f"  Quads: {report.n_quads}")
    print(f"Area: {report.area_min:.1f} – {report.area_max:.1f} (mean {report.area_mean:.1f})")
    print(f"Aspect ratio: {report.aspect_ratio_min:.2f} – {report.aspect_ratio_max:.2f} "
          f"(mean {report.aspect_ratio_mean:.2f})")
    print(f"Skewness (mean): {report.skewness_mean:.3f}")
    print(f"Angle range: {report.min_angle_global:.1f}° – {report.max_angle_global:.1f}°")
    print(f"Poor quality elements: {report.poor_quality_count}")

Step 3: Inspect Individual Elements
-------------------------------------

.. code-block:: python

    # Find worst elements by aspect ratio
    worst = sorted(report.element_qualities, key=lambda e: e.aspect_ratio, reverse=True)[:5]
    print("Top 5 worst aspect ratios:")
    for eq in worst:
        print(f"  Element {eq.element_id}: AR={eq.aspect_ratio:.2f}, "
              f"skew={eq.skewness:.3f}, angles={eq.min_angle:.1f}°–{eq.max_angle:.1f}°")

Step 4: Histogram of Aspect Ratios
------------------------------------

.. code-block:: python

    import matplotlib.pyplot as plt
    import numpy as np

    aspect_ratios = [eq.aspect_ratio for eq in report.element_qualities]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(aspect_ratios, bins=50, edgecolor="black", alpha=0.7)
    ax.axvline(5.0, color="red", linestyle="--", label="Quality threshold (AR=5)")
    ax.set_xlabel("Aspect Ratio")
    ax.set_ylabel("Number of Elements")
    ax.set_title("Distribution of Element Aspect Ratios")
    ax.legend()
    plt.tight_layout()
    plt.show()

Step 5: Spatial Visualization of Skewness
-------------------------------------------

.. code-block:: python

    from pyiwfm.visualization.plotting import plot_scalar_field

    skewness = np.array([eq.skewness for eq in report.element_qualities])

    fig, ax = plot_scalar_field(
        grid, skewness, field_type="cell",
        cmap="YlOrRd", show_mesh=True, edge_color="gray",
    )
    ax.set_title("Element Skewness")
    ax.set_xlabel("Easting (ft)")
    ax.set_ylabel("Northing (ft)")
    plt.tight_layout()
    plt.show()

Step 6: Quality Summary for API
---------------------------------

The report can be serialized for use in web APIs or JSON export:

.. code-block:: python

    summary = report.to_dict()
    import json
    print(json.dumps(summary, indent=2))

This is the same data shown in the web viewer's Overview tab mesh quality card.

Quality Thresholds
------------------

General guidelines for IWFM mesh quality:

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Metric
     - Threshold
     - Impact
   * - Aspect Ratio
     - < 5
     - High ratios cause anisotropic numerical error
   * - Skewness
     - < 0.8
     - High skewness degrades element stiffness matrix accuracy
   * - Minimum Angle
     - > 15°
     - Very small angles lead to ill-conditioned elements
   * - Maximum Angle
     - < 165°
     - Very obtuse angles reduce solution accuracy
