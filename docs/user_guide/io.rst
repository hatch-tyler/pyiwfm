Reading and Writing Files
=========================

This guide covers reading and writing IWFM model files in various formats.

Supported Formats
-----------------

pyiwfm supports multiple file formats:

- **ASCII**: Human-readable text files (IWFM's native format)
- **Binary**: Fortran unformatted binary files (faster I/O)
- **HDF5**: Hierarchical data format (efficient for large datasets)
- **HEC-DSS 7**: Time series data format (optional, requires external library)

Complete Model I/O
------------------

The easiest way to work with IWFM models is to use the complete model I/O functions.

Loading a Complete Model
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from pyiwfm.io import load_complete_model

    # Load model from simulation main file
    model = load_complete_model("Simulation/Simulation.in")

    # Access model components
    print(f"Grid: {model.grid.n_nodes} nodes, {model.grid.n_elements} elements")
    print(f"Stratigraphy layers: {model.stratigraphy.n_layers}")

    # Access component data (if loaded)
    if model.groundwater:
        print(f"Wells: {len(model.groundwater.wells)}")
    if model.stream:
        print(f"Stream nodes: {len(model.stream.nodes)}")

Saving a Complete Model
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from pathlib import Path
    from pyiwfm.io import save_complete_model

    # Save model to a new directory
    files_written = save_complete_model(
        model,
        output_dir=Path("output_model"),
        timeseries_format="ascii",  # or "dss"
    )

    print(f"Written files: {list(files_written.keys())}")

ASCII Files
-----------

Reading ASCII Node Files
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from pyiwfm.io import read_nodes

    # Read nodes from IWFM node file
    nodes = read_nodes("Preprocessor/Nodal.dat")

    print(f"Read {len(nodes)} nodes")
    for node_id, node in list(nodes.items())[:5]:
        print(f"  Node {node_id}: ({node.x}, {node.y})")

Writing ASCII Node Files
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from pyiwfm.io import write_nodes
    from pyiwfm.core.mesh import Node

    # Prepare nodes
    nodes = {
        1: Node(id=1, x=0.0, y=0.0),
        2: Node(id=2, x=100.0, y=0.0),
        3: Node(id=3, x=50.0, y=100.0),
    }

    # Write to file
    write_nodes("output/nodes.dat", nodes)

Reading and Writing Elements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from pyiwfm.io import read_elements, write_elements
    from pyiwfm.core.mesh import Element

    # Read elements from IWFM element file
    elements = read_elements("Preprocessor/Element.dat")

    # Write elements
    write_elements("output/elements.dat", elements)

Reading and Writing Stratigraphy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from pyiwfm.io import read_stratigraphy, write_stratigraphy

    # Read stratigraphy
    stratigraphy = read_stratigraphy("Preprocessor/Stratigraphy.dat")
    print(f"Layers: {stratigraphy.n_layers}")

    # Write stratigraphy
    write_stratigraphy("output/stratigraphy.dat", stratigraphy)

Binary Files
------------

Binary files are faster for large datasets but not human-readable.

Reading Native IWFM Binary Files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from pyiwfm.core.model import IWFMModel

    # Load model from IWFM PreProcessor binary output (ACCESS='STREAM' format)
    model = IWFMModel.from_preprocessor_binary("PreprocessorOut.bin")
    print(f"Read mesh: {model.n_nodes} nodes, {model.n_elements} elements")

HDF5 Files
----------

HDF5 is recommended for large models and result storage.

Writing Complete Model to HDF5
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from pyiwfm.io import write_model_hdf5

    # Write complete model to HDF5
    write_model_hdf5("model.h5", model)

Reading Model from HDF5
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from pyiwfm.io import read_model_hdf5

    # Read complete model from HDF5
    model = read_model_hdf5("model.h5")

HDF5 File Structure
~~~~~~~~~~~~~~~~~~~

The HDF5 file structure follows this hierarchy:

.. code-block:: text

    model.h5
    ├── mesh/
    │   ├── nodes/
    │   │   ├── id
    │   │   ├── x
    │   │   ├── y
    │   │   └── is_boundary
    │   └── elements/
    │       ├── id
    │       ├── vertices
    │       └── subregion
    ├── stratigraphy/
    │   ├── n_layers
    │   ├── gs_elev
    │   ├── top_elev
    │   ├── bottom_elev
    │   └── active_node
    └── timeseries/
        └── heads/
            ├── times
            └── values

Component File Writers
----------------------

pyiwfm provides specialized writers for each IWFM component.

Groundwater Component
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from pathlib import Path
    from pyiwfm.io.groundwater import GroundwaterWriter, GWFileConfig

    # Configure output files
    config = GWFileConfig(
        output_dir=Path("output/Groundwater"),
        wells_file="Wells.dat",
        pumping_file="Pumping.dat",
    )

    # Write groundwater files
    with GroundwaterWriter(config) as writer:
        files = writer.write(model.groundwater)
        print(f"Written: {files}")

Stream Component
~~~~~~~~~~~~~~~~

.. code-block:: python

    from pathlib import Path
    from pyiwfm.io.streams import StreamWriter, StreamFileConfig

    config = StreamFileConfig(
        output_dir=Path("output/Streams"),
        stream_nodes_file="StreamNodes.dat",
        reaches_file="Reaches.dat",
    )

    with StreamWriter(config) as writer:
        files = writer.write(model.stream)

Lake Component
~~~~~~~~~~~~~~

.. code-block:: python

    from pathlib import Path
    from pyiwfm.io.lakes import LakeWriter, LakeFileConfig

    config = LakeFileConfig(
        output_dir=Path("output/Lakes"),
    )

    with LakeWriter(config) as writer:
        files = writer.write(model.lakes)

Root Zone Component
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from pathlib import Path
    from pyiwfm.io.rootzone import RootZoneWriter, RootZoneFileConfig

    config = RootZoneFileConfig(
        output_dir=Path("output/RootZone"),
    )

    with RootZoneWriter(config) as writer:
        files = writer.write(model.rootzone)

Simulation Control File
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from pathlib import Path
    from pyiwfm.io.simulation import SimulationWriter, SimulationConfig

    config = SimulationConfig(
        title="My IWFM Model",
        start_date="10/01/1990",
        end_date="09/30/2020",
        time_step="1DAY",
    )

    writer = SimulationWriter(Path("output"))
    writer.write(config)

ASCII Time Series
-----------------

IWFM uses a specific 21-character timestamp format for time series files.

Writing Time Series
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    from datetime import datetime
    from pyiwfm.io import TimeSeriesWriter, format_iwfm_timestamp

    # Create times and values
    times = np.array([datetime(2020, 1, i) for i in range(1, 32)], dtype="datetime64[s]")
    values = np.random.rand(31)

    # Write to file
    with TimeSeriesWriter("pumping.dat") as writer:
        writer.write(times, {"WELL_1": values})

    # Or use the convenience function
    from pyiwfm.io import write_timeseries
    write_timeseries("pumping.dat", times, {"WELL_1": values, "WELL_2": values * 0.5})

Reading Time Series
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from pyiwfm.io import TimeSeriesReader, read_timeseries

    # Read time series from file
    with TimeSeriesReader("pumping.dat") as reader:
        times, values_dict = reader.read()

    # Or use the convenience function
    times, values_dict = read_timeseries("pumping.dat")
    print(f"Locations: {list(values_dict.keys())}")

Timestamp Format
~~~~~~~~~~~~~~~~

IWFM uses a 21-character timestamp format:

.. code-block:: python

    from pyiwfm.io import format_iwfm_timestamp, parse_iwfm_timestamp
    from datetime import datetime

    dt = datetime(2020, 6, 15, 12, 30, 0)

    # Format to IWFM string: "06/15/2020_12:30:00"
    ts_str = format_iwfm_timestamp(dt)

    # Parse back to datetime
    dt_parsed = parse_iwfm_timestamp(ts_str)

HEC-DSS Support
---------------

pyiwfm supports reading and writing HEC-DSS 7 files for time series data.
This requires the HEC-DSS C library to be installed.

.. note::

    HEC-DSS support is optional. Set the ``HECDSS_LIB`` environment variable
    to point to the HEC-DSS library location.

Checking DSS Availability
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from pyiwfm.io.dss import HAS_DSS_LIBRARY

    if HAS_DSS_LIBRARY:
        print("HEC-DSS library is available")
    else:
        print("HEC-DSS library not found")

DSS Pathnames
~~~~~~~~~~~~~

HEC-DSS uses a 6-part pathname: ``/A/B/C/D/E/F/``

.. code-block:: python

    from pyiwfm.io.dss import DSSPathname, DSSPathnameTemplate

    # Create a pathname directly
    pathname = DSSPathname(
        a_part="PROJECT",
        b_part="LOCATION",
        c_part="FLOW",
        d_part="01JAN2020-31DEC2020",
        e_part="1DAY",
        f_part="SIM",
    )
    print(str(pathname))  # /PROJECT/LOCATION/FLOW/01JAN2020-31DEC2020/1DAY/SIM/

    # Use a template for multiple locations
    template = DSSPathnameTemplate(
        a_part="IWFM_MODEL",
        c_part="HEAD",
        e_part="1DAY",
        f_part="BASELINE",
    )

    # Create pathnames for different locations
    path1 = template.make_pathname(location="WELL_001")
    path2 = template.make_pathname(location="WELL_002")

Writing Time Series to DSS
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from pyiwfm.io.dss import (
        DSSTimeSeriesWriter,
        DSSPathnameTemplate,
        write_timeseries_to_dss,
        write_collection_to_dss,
    )
    from pyiwfm.core.timeseries import TimeSeries

    # Write a single time series
    ts = TimeSeries(times=times, values=values, name="HEAD", location="WELL_1", units="ft")
    template = DSSPathnameTemplate(a_part="MODEL", c_part="HEAD", e_part="1DAY")

    result = write_timeseries_to_dss(
        "output.dss",
        ts,
        template.make_pathname(location="WELL_1"),
    )
    print(f"Success: {result.success}")

    # Write a collection of time series
    from pyiwfm.core.timeseries import TimeSeriesCollection

    collection = TimeSeriesCollection(variable="HEAD")
    collection.add(ts)

    result = write_collection_to_dss("output.dss", collection, template, units="ft")

Reading Time Series from DSS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from pyiwfm.io.dss import DSSTimeSeriesReader, read_timeseries_from_dss

    # Read a single time series
    ts = read_timeseries_from_dss(
        "output.dss",
        "/MODEL/WELL_1/HEAD/01JAN2020-31DEC2020/1DAY//"
    )
    print(f"Values: {ts.values}")

    # Read multiple time series
    with DSSTimeSeriesReader("output.dss") as reader:
        collection = reader.read_collection([
            "/MODEL/WELL_1/HEAD//1DAY//",
            "/MODEL/WELL_2/HEAD//1DAY//",
        ])

Budget Excel Export
-------------------

pyiwfm can parse IWFM budget and zone budget control files and export the
results to formatted Excel workbooks -- one sheet per location (or zone) with
title lines, bold headers, unit conversion, and auto-fitted column widths.

Control File Workflow
~~~~~~~~~~~~~~~~~~~~~

The standard IWFM workflow uses a control file that references one or more HDF5
budget files with unit conversion factors and output paths:

.. code-block:: python

    from pyiwfm.io import read_budget_control, budget_control_to_excel

    # Parse the control file
    config = read_budget_control("C2VSimFG_Budget_xlsx.in")

    # Export all budgets to Excel (one .xlsx per budget spec)
    created_files = budget_control_to_excel(config)
    for f in created_files:
        print(f"Created: {f}")

Zone budget control files follow the same pattern:

.. code-block:: python

    from pyiwfm.io import read_zbudget_control, zbudget_control_to_excel

    config = read_zbudget_control("C2VSimFG_ZBudget_xlsx.in")
    created_files = zbudget_control_to_excel(config)

Direct Export
~~~~~~~~~~~~~

You can also export directly from a ``BudgetReader`` or ``ZBudgetReader``
without a control file:

.. code-block:: python

    from pyiwfm.io import BudgetReader, budget_to_excel

    reader = BudgetReader("GW_Budget.hdf")
    budget_to_excel(
        reader,
        "GW_Budget.xlsx",
        volume_factor=2.29568e-05,
        volume_unit="AC.FT.",
        area_factor=2.29568e-05,
        area_unit="AC",
    )

Unit-Converted DataFrames
~~~~~~~~~~~~~~~~~~~~~~~~~

The ``get_dataframe()`` method on both ``BudgetReader`` and ``ZBudgetReader``
accepts optional conversion factors for direct use in scripts and notebooks:

.. code-block:: python

    from pyiwfm.io import BudgetReader

    reader = BudgetReader("GW_Budget.hdf")

    # Raw data (no conversion)
    df_raw = reader.get_dataframe(0)

    # With unit conversion applied
    df_converted = reader.get_dataframe(
        0,
        length_factor=1.0,
        area_factor=2.29568e-05,
        volume_factor=2.29568e-05,
    )

CLI Commands
~~~~~~~~~~~~

Budget export is also available from the command line:

.. code-block:: bash

    # Export budgets from a control file
    pyiwfm budget C2VSimFG_Budget_xlsx.in

    # Export zone budgets from a control file
    pyiwfm zbudget C2VSimFG_ZBudget_xlsx.in

    # Override output directory
    pyiwfm budget C2VSimFG_Budget_xlsx.in --output-dir /path/to/output

PreProcessor Integration
------------------------

Loading Models from PreProcessor Files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from pyiwfm.core.model import IWFMModel
    from pyiwfm.io import load_complete_model

    # Load just mesh and stratigraphy from preprocessor
    model = IWFMModel.from_preprocessor("Preprocessor/Preprocessor.in")

    # Load complete model including all components
    model = load_complete_model("Simulation/Simulation.in")

Saving Models to PreProcessor Format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from pathlib import Path
    from pyiwfm.io import save_model_to_preprocessor, save_complete_model

    # Save mesh and stratigraphy
    save_model_to_preprocessor(model, Path("output/Preprocessor"))

    # Save complete model with all components
    files = save_complete_model(model, Path("output"))
    print(f"Written: {list(files.keys())}")
