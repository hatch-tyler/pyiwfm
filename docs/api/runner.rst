Runner and PEST++ Integration
==============================

The runner modules provide subprocess execution, scenario management, and
complete PEST++ calibration interface for IWFM models.

.. contents:: Table of Contents
   :local:
   :depth: 2

Runner Module
-------------

The runner module provides the core subprocess interface for executing
IWFM executables, including ``IWFMRunner`` and ``IWFMExecutables``.

.. automodule:: pyiwfm.runner.runner
   :members:
   :undoc-members:
   :show-inheritance:

Results Module
--------------

Typed result classes for IWFM executable runs.

.. automodule:: pyiwfm.runner.results
   :members:
   :undoc-members:
   :show-inheritance:

Scenario Module
---------------

Scenario management for running and comparing multiple model configurations.

.. automodule:: pyiwfm.runner.scenario
   :members:
   :undoc-members:
   :show-inheritance:

PEST++ Interface
-----------------

Low-level PEST++ control file interface.

.. automodule:: pyiwfm.runner.pest
   :members:
   :undoc-members:
   :show-inheritance:

Parameter Management
--------------------

IWFM-specific parameter types, transforms, and parameterization strategies.

Parameter Types and Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: pyiwfm.runner.pest_params
   :members:
   :undoc-members:
   :show-inheritance:

Parameter Manager
~~~~~~~~~~~~~~~~~

.. automodule:: pyiwfm.runner.pest_manager
   :members:
   :undoc-members:
   :show-inheritance:

Observation Management
----------------------

Observation types, weights, and management for PEST++ calibration targets.

Observation Types
~~~~~~~~~~~~~~~~~

.. automodule:: pyiwfm.runner.pest_observations
   :members:
   :undoc-members:
   :show-inheritance:

Observation Manager
~~~~~~~~~~~~~~~~~~~

.. automodule:: pyiwfm.runner.pest_obs_manager
   :members:
   :undoc-members:
   :show-inheritance:

Template and Instruction Files
------------------------------

Automatic generation of PEST++ template (.tpl) and instruction (.ins) files
from IWFM input/output files.

Template Manager
~~~~~~~~~~~~~~~~

.. automodule:: pyiwfm.runner.pest_templates
   :members:
   :undoc-members:
   :show-inheritance:

Instruction Manager
~~~~~~~~~~~~~~~~~~~

.. automodule:: pyiwfm.runner.pest_instructions
   :members:
   :undoc-members:
   :show-inheritance:

Geostatistics
-------------

Variogram modeling and spatial correlation for pilot point and ensemble
parameterization.

.. automodule:: pyiwfm.runner.pest_geostat
   :members:
   :undoc-members:
   :show-inheritance:

Main Helper Interface
---------------------

The ``IWFMPestHelper`` class is the primary entry point for setting up
PEST++ calibration. It coordinates parameters, observations, templates,
instructions, and control file generation.

.. automodule:: pyiwfm.runner.pest_helper
   :members:
   :undoc-members:
   :show-inheritance:

Ensemble Management
-------------------

Prior and posterior ensemble generation for pestpp-ies iterative ensemble
smoother workflows.

.. automodule:: pyiwfm.runner.pest_ensemble
   :members:
   :undoc-members:
   :show-inheritance:

Post-Processing
---------------

Load and analyze PEST++ output files for calibration assessment.

.. automodule:: pyiwfm.runner.pest_postprocessor
   :members:
   :undoc-members:
   :show-inheritance:
