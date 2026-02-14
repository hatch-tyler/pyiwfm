# pyiwfm

Python package for reading, writing, visualizing, and comparing IWFM (Integrated Water Flow Model) models.

## Installation

```bash
# Basic installation
pip install -e .

# With GIS support
pip install -e ".[gis]"

# With mesh generation
pip install -e ".[mesh]"

# With web viewer (FastAPI + React + vtk.js + deck.gl)
pip install -e ".[webapi]"

# With all optional dependencies
pip install -e ".[all]"

# Development installation
pip install -e ".[dev]"
```

## Quick Start

```python
from pyiwfm import AppGrid, Node, Element, Stratigraphy
import numpy as np

# Create a simple mesh
nodes = {
    1: Node(id=1, x=0.0, y=0.0),
    2: Node(id=2, x=100.0, y=0.0),
    3: Node(id=3, x=100.0, y=100.0),
    4: Node(id=4, x=0.0, y=100.0),
}

elements = {
    1: Element(id=1, vertices=(1, 2, 3, 4), subregion=1),
}

grid = AppGrid(nodes=nodes, elements=elements)
grid.compute_areas()
grid.compute_connectivity()

print(f"Grid: {grid.n_nodes} nodes, {grid.n_elements} elements")
```

## Model I/O Example

```python
from pathlib import Path
from pyiwfm.io import load_complete_model, save_complete_model

# Load a complete IWFM model from simulation main file
model = load_complete_model("Simulation/Simulation.in")

print(f"Loaded model with {model.grid.n_nodes} nodes")
print(f"Groundwater wells: {len(model.groundwater.wells) if model.groundwater else 0}")

# Save model to new directory
save_complete_model(model, Path("output_model"))

# Write time series to HEC-DSS (requires bundled C library or HECDSS_LIB env var)
from pyiwfm.io.dss import DSSTimeSeriesWriter, DSSPathnameTemplate, HAS_DSS_LIBRARY

if HAS_DSS_LIBRARY:
    template = DSSPathnameTemplate(a_part="IWFM", c_part="HEAD", e_part="1DAY")
    with DSSTimeSeriesWriter("output.dss") as writer:
        writer.write_timeseries(head_timeseries, template.make_pathname(location="WELL_1"))
```

## Web Visualization

pyiwfm includes an interactive web viewer built with FastAPI (backend) and React + vtk.js + deck.gl (frontend). Launch it with:

```bash
# CRS is optional — defaults to C2VSimFG UTM Zone 10N for most models
pyiwfm viewer --model-dir /path/to/model

# Specify CRS explicitly if needed
pyiwfm viewer --model-dir /path/to/model --crs "+proj=utm +zone=10 +datum=NAD83 +units=us-ft +no_defs"
```

The viewer provides four tabs:
- **Overview**: Model summary and metadata
- **3D Mesh**: Interactive vtk.js 3D rendering with layer visibility, cross-section slicing, stream network overlay, and z-exaggeration
- **Results Map**: deck.gl + MapLibre map showing head contours, hydrograph locations, and observation upload/comparison
- **Budgets**: Plotly charts of water budget time series with location/column selection

The frontend is pre-built into `src/pyiwfm/visualization/webapi/static/`. To rebuild from source:

```bash
cd frontend && npm install && npm run build
```

## Features

- **Core Data Structures**: Node, Element, Face, AppGrid, Stratigraphy, TimeSeries
- **Complete Model I/O**: Full roundtrip support for reading and writing IWFM models
  - ASCII files (nodes, elements, stratigraphy, time series)
  - Binary files (Fortran unformatted)
  - HDF5 files (efficient large model storage)
  - HEC-DSS 7 files (time series with optional library support)
- **Component Writers**: Write complete IWFM input files
  - Groundwater: wells, pumping, boundary conditions, aquifer parameters
  - Streams: nodes, reaches, diversions, bypasses, rating curves
  - Lakes: definitions, elements, rating curves, outflows
  - Root Zone: crop types, soil parameters, land use
  - Simulation: main control file
- **PreProcessor Integration**: Load/save complete models from IWFM file structure
- **Mesh Generation**: Triangle and Gmsh wrappers
- **Visualization**: GIS export, VTK 3D export, interactive web viewer with budget charts, head maps, and hydrograph comparison
- **Model Comparison**: Diff and comparison metrics

## License

GPL-2.0 - Same as IWFM
