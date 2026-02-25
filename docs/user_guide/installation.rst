Installation
============

Requirements
------------

pyiwfm requires Python 3.10 or later.

Basic Installation
------------------

Install pyiwfm using pip:

.. code-block:: bash

    pip install pyiwfm

This installs the core package with required dependencies:

- numpy >= 1.23
- pandas >= 2.0
- h5py >= 3.7
- jinja2 >= 3.0
- matplotlib >= 3.8
- geopandas >= 1.0
- shapely >= 2.0
- pyogrio >= 0.7

Optional Dependencies
---------------------

pyiwfm has several optional dependency groups for additional functionality:

Mesh Generation
~~~~~~~~~~~~~~~

For creating finite element meshes:

.. code-block:: bash

    pip install pyiwfm[mesh]

This installs:

- triangle >= 20220202 (triangular meshes)
- gmsh >= 4.11 (triangular, quadrilateral, or mixed meshes)

VTK 3D Export
~~~~~~~~~~~~~

For 3D visualization export (ParaView):

.. code-block:: bash

    pip install pyiwfm[viz]

This installs:

- vtk >= 9.0

Web Viewer
~~~~~~~~~~

For the interactive web viewer (FastAPI + React + vtk.js + deck.gl):

.. code-block:: bash

    pip install pyiwfm[webapi]

This installs:

- fastapi >= 0.104
- uvicorn >= 0.24
- pydantic >= 2.0
- pyvista >= 0.43
- vtk >= 9.0
- pyproj >= 3.4
- python-multipart >= 0.0.6

PEST++ Integration
~~~~~~~~~~~~~~~~~~

For parameter estimation workflows:

.. code-block:: bash

    pip install pyiwfm[pest]

This installs:

- scipy >= 1.7

Documentation
~~~~~~~~~~~~~

For building the Sphinx documentation:

.. code-block:: bash

    pip install pyiwfm[docs]

This installs:

- sphinx
- pydata-sphinx-theme
- myst-parser
- sphinx-design
- sphinx-copybutton

All Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~

To install all optional dependencies:

.. code-block:: bash

    pip install pyiwfm[all]

Development Installation
------------------------

For development, clone the repository and install in editable mode:

.. code-block:: bash

    git clone https://github.com/hatch-tyler/pyiwfm.git
    cd pyiwfm
    pip install -e ".[dev]"

The ``[dev]`` extra includes testing and linting tools:

- pytest, pytest-cov, hypothesis
- mypy
- ruff, pre-commit
- httpx

Verifying Installation
----------------------

Verify your installation by importing pyiwfm:

.. code-block:: python

    >>> import pyiwfm
    >>> print(pyiwfm.__version__)
    1.0.0

Check available modules:

.. code-block:: python

    >>> from pyiwfm.core.mesh import AppGrid, Node, Element
    >>> from pyiwfm.visualization import GISExporter

    # Check if optional modules are available
    >>> try:
    ...     from pyiwfm.mesh_generation import TriangleMeshGenerator
    ...     print("Triangle mesh generation available")
    ... except ImportError:
    ...     print("Install pyiwfm[mesh] for mesh generation")
