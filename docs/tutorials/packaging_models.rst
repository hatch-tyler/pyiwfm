Tutorial: Packaging and Running IWFM Models
=============================================

This tutorial demonstrates how to package an IWFM model directory into a
distributable ZIP archive and generate platform-appropriate run scripts
using pyiwfm.

Learning Objectives
-------------------

By the end of this tutorial, you will be able to:

1. Collect model files from an IWFM model directory
2. Package a complete model into a ZIP archive with a manifest
3. Generate run scripts for Windows (``.bat`` / ``.ps1``) and Linux (``.sh``)
4. Use the ``pyiwfm package`` and ``pyiwfm run`` CLI commands

Prerequisites
-------------

Install pyiwfm with basic dependencies:

.. code-block:: bash

    pip install pyiwfm

No additional extras are required for packaging and script generation.

Collecting Model Files
----------------------

The :func:`~pyiwfm.io.model_packager.collect_model_files` function walks an
IWFM model directory and returns all files relevant to the model, filtering
out output files, caches, and version-control directories:

.. code-block:: python

   from pathlib import Path
   from pyiwfm.io import collect_model_files

   model_dir = Path("C2VSimCG")
   files = collect_model_files(model_dir)

   print(f"Found {len(files)} model files")
   for f in files[:10]:
       print(f"  {f.relative_to(model_dir)}")

**Default exclusions:**

- ``Results/``, ``__pycache__/``, ``.git/``, ``.svn/`` directories
- ``.hdf`` / ``.h5`` output files
- ``.exe`` / ``.dll`` / ``.so`` executables

**Including executables or results:**

.. code-block:: python

   # Include executables for a self-contained package
   files_with_exes = collect_model_files(model_dir, include_executables=True)

   # Include results directory and HDF5 output files
   files_with_results = collect_model_files(model_dir, include_results=True)

   # Include everything
   all_files = collect_model_files(
       model_dir,
       include_executables=True,
       include_results=True,
   )

Creating a ZIP Archive
----------------------

Use :func:`~pyiwfm.io.model_packager.package_model` to create a ZIP archive
that preserves the directory structure and includes a ``manifest.json``:

.. code-block:: python

   from pyiwfm.io import package_model

   result = package_model(
       model_dir,
       output_path=Path("C2VSimCG_v1.0.zip"),
   )

   print(f"Archive:    {result.archive_path}")
   print(f"Files:      {len(result.files_included)}")
   print(f"Size:       {result.total_size_bytes / 1024 / 1024:.1f} MB")

The archive preserves the standard IWFM directory layout:

.. code-block:: text

   C2VSimCG_v1.0.zip
   ├── manifest.json
   ├── Preprocessor/
   │   ├── Preprocessor.in
   │   ├── Nodal.dat
   │   ├── Element.dat
   │   ├── Stratigraphy.dat
   │   └── ...
   ├── Simulation/
   │   ├── Simulation.in
   │   ├── Groundwater/
   │   ├── Streams/
   │   └── ...
   └── ...

**Packaging with executables (self-contained):**

.. code-block:: python

   result = package_model(
       model_dir,
       output_path=Path("C2VSimCG_standalone.zip"),
       include_executables=True,
   )

**Inspecting the manifest:**

The manifest maps each file in the archive to a category (``input``,
``binary``, ``script``, ``gis``, ``executable``, ``data``):

.. code-block:: python

   import json

   for rel_path, category in sorted(result.manifest.items()):
       print(f"  [{category:10s}] {rel_path}")

Generating Run Scripts
----------------------

pyiwfm generates platform-appropriate run scripts for executing the
IWFM preprocessor, simulation, and optional post-processors (Budget,
ZBudget).

Basic Script Generation
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pyiwfm.roundtrip.script_generator import generate_run_scripts

   scripts = generate_run_scripts(
       model_dir=Path("C2VSimCG"),
       preprocessor_main="Preprocessor/Preprocessor.in",
       simulation_main="Simulation/Simulation.in",
   )

   for s in scripts:
       print(f"Generated: {s.name}")

On Windows this creates ``.bat`` and ``.ps1`` scripts by default; on
Linux/macOS it creates ``.sh`` scripts.

**Generated scripts:**

- ``run_preprocessor`` -- runs the IWFM PreProcessor
- ``run_simulation`` -- runs the IWFM Simulation
- ``run_all`` -- runs preprocessor then simulation in sequence

Choosing Script Formats
~~~~~~~~~~~~~~~~~~~~~~~~

Use the ``formats`` parameter to control which script types are generated:

.. code-block:: python

   # Generate all three formats
   scripts = generate_run_scripts(
       model_dir=Path("C2VSimCG"),
       preprocessor_main="Preprocessor/Preprocessor.in",
       simulation_main="Simulation/Simulation.in",
       formats=["bat", "ps1", "sh"],
   )
   # -> 9 scripts (3 per format)

Adding Budget and ZBudget Scripts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pass ``budget_exe`` and/or ``zbudget_exe`` to also generate post-processor
run scripts:

.. code-block:: python

   scripts = generate_run_scripts(
       model_dir=Path("C2VSimCG"),
       preprocessor_main="Preprocessor/Preprocessor.in",
       simulation_main="Simulation/Simulation.in",
       budget_exe="Budget_x64.exe",
       zbudget_exe="ZBudget_x64.exe",
   )

   for s in scripts:
       print(f"Generated: {s.name}")

This adds ``run_budget`` and ``run_zbudget`` scripts, and the ``run_all``
script is updated to include them after the simulation step.

CLI Commands
------------

Both packaging and script generation are available from the command line.

``pyiwfm package``
~~~~~~~~~~~~~~~~~~

Package a model directory into a ZIP archive:

.. code-block:: bash

   # Basic packaging (inputs only)
   pyiwfm package --model-dir ./C2VSimCG

   # Custom output path
   pyiwfm package --model-dir ./C2VSimCG --output C2VSimCG_v1.0.zip

   # Include executables for a self-contained archive
   pyiwfm package --model-dir ./C2VSimCG --include-executables

   # Include results/output files
   pyiwfm package --model-dir ./C2VSimCG --include-results

**Example output:**

.. code-block:: text

   Packaged 147 files (23.4 MB) -> C:\models\C2VSimCG.zip

``pyiwfm run``
~~~~~~~~~~~~~~

Generate run scripts and optionally download executables:

.. code-block:: bash

   # Generate run scripts only (auto-detects platform)
   pyiwfm run --model-dir ./C2VSimCG --scripts-only

   # Generate specific format(s)
   pyiwfm run --model-dir ./C2VSimCG --scripts-only --format bat --format ps1

   # Download executables from GitHub and generate scripts
   pyiwfm run --model-dir ./C2VSimCG --download-executables --scripts-only

**Example output:**

.. code-block:: text

   Generated: C:\models\C2VSimCG\run_preprocessor.bat
   Generated: C:\models\C2VSimCG\run_simulation.bat
   Generated: C:\models\C2VSimCG\run_all.bat
   Generated: C:\models\C2VSimCG\run_preprocessor.ps1
   Generated: C:\models\C2VSimCG\run_simulation.ps1
   Generated: C:\models\C2VSimCG\run_all.ps1

Complete Workflow
-----------------

Here is a complete example that packages a model, downloads executables,
and generates run scripts:

.. code-block:: python

   """Package an IWFM model for distribution."""

   from pathlib import Path
   from pyiwfm.io import package_model
   from pyiwfm.roundtrip.script_generator import generate_run_scripts

   model_dir = Path("C2VSimCG")

   # 1. Generate run scripts
   scripts = generate_run_scripts(
       model_dir=model_dir,
       preprocessor_main="Preprocessor/Preprocessor.in",
       simulation_main="Simulation/Simulation.in",
       budget_exe="Budget_x64.exe",
       formats=["bat", "ps1", "sh"],
   )
   print(f"Generated {len(scripts)} run scripts")

   # 2. Package the model (scripts are now included in the ZIP)
   result = package_model(
       model_dir,
       output_path=Path("C2VSimCG_distribution.zip"),
       include_executables=True,
   )

   print(f"Archive: {result.archive_path}")
   print(f"Files:   {len(result.files_included)}")
   print(f"Size:    {result.total_size_bytes / 1024 / 1024:.1f} MB")

   # 3. Print manifest summary
   categories = {}
   for category in result.manifest.values():
       categories[category] = categories.get(category, 0) + 1
   for cat, count in sorted(categories.items()):
       print(f"  {cat}: {count} files")

Next Steps
----------

- See :doc:`reading_models` for loading and inspecting IWFM models
- See :doc:`building_sample_model` for constructing a model from scratch
- See :doc:`/user_guide/io` for all supported file formats
- See :doc:`/api/roundtrip` for the full API reference
