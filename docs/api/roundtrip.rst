Roundtrip and CLI Modules
=========================

The roundtrip modules provide read-write-read verification pipelines,
run script generation, model packaging, and CLI subcommands for IWFM models.

.. contents:: Table of Contents
   :local:
   :depth: 2

Roundtrip Pipeline
------------------

The pipeline module orchestrates a full read-write-read verification cycle:
load a model, write it to a new directory, optionally run both versions,
and compare results.

.. automodule:: pyiwfm.roundtrip.pipeline
   :members:
   :undoc-members:
   :show-inheritance:

Roundtrip Configuration
-----------------------

Configuration dataclass for the roundtrip pipeline.

.. automodule:: pyiwfm.roundtrip.config
   :members:
   :undoc-members:
   :show-inheritance:

Run Script Generator
--------------------

Generates platform-appropriate run scripts (``.bat``, ``.ps1``, ``.sh``)
for IWFM preprocessor, simulation, and optional Budget/ZBudget
post-processors.

.. automodule:: pyiwfm.roundtrip.script_generator
   :members:
   :undoc-members:
   :show-inheritance:

Model Packager
--------------

Packages an IWFM model directory into a distributable ZIP archive with
an embedded ``manifest.json``.

.. automodule:: pyiwfm.io.model_packager
   :members:
   :undoc-members:
   :show-inheritance:

Executable Manager
------------------

Downloads, locates, and places IWFM executables (PreProcessor, Simulation,
Budget, ZBudget) for model execution.

.. automodule:: pyiwfm.runner.executables
   :members:
   :undoc-members:
   :show-inheritance:

Results Differ
--------------

Compares simulation results between baseline and written model runs.

.. automodule:: pyiwfm.comparison.results_differ
   :members:
   :undoc-members:
   :show-inheritance:

CLI Subcommands
---------------

Package Command
~~~~~~~~~~~~~~~

The ``pyiwfm package`` subcommand packages a model directory into a ZIP archive.

.. automodule:: pyiwfm.cli.package
   :members:
   :undoc-members:
   :show-inheritance:

Run Command
~~~~~~~~~~~

The ``pyiwfm run`` subcommand generates run scripts and optionally downloads
executables.

.. automodule:: pyiwfm.cli.run
   :members:
   :undoc-members:
   :show-inheritance:
