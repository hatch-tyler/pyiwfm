# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

pyiwfm is a Python package for working with IWFM (Integrated Water Flow Model) models developed by the California Department of Water Resources. It provides tools for reading/writing IWFM model files (ASCII, binary, HDF5, HEC-DSS formats), mesh manipulation, simulation execution, PEST++ parameter estimation, and interactive web visualization.

## Commands

### Installation
```bash
pip install -e .              # Basic install
pip install -e ".[dev]"       # Development (pytest, mypy, ruff)
pip install -e ".[webapi]"    # Web viewer (FastAPI + vtk.js + deck.gl)
pip install -e ".[all]"       # All optional dependencies
```

### Testing
```bash
pytest tests/                          # Run all tests
pytest tests/ --cov=pyiwfm             # With coverage
pytest tests/unit/test_mesh.py -v      # Single test file
pytest tests/ -m slow                  # Run slow-marked tests
pytest tests/ -m integration           # Run integration tests
```

### Code Quality
```bash
ruff format src/ tests/                # Format code
ruff check src/ tests/ --fix           # Lint with auto-fix
mypy src/pyiwfm/                       # Type checking
pre-commit run --all-files             # Run all pre-commit hooks
```

### CLI
```bash
pyiwfm viewer /path/to/model           # Launch the web viewer
pyiwfm export /path/to/model           # Export to VTK/GeoPackage
```

### Frontend
```bash
cd frontend && npm install && npm run build   # Build React frontend to webapi/static/
cd frontend && npm run dev                    # Vite dev server with hot reload
```

## Architecture

### Source Layout
```
src/pyiwfm/
├── core/              # Mesh (Node, Element, AppGrid), Stratigraphy, TimeSeries, IWFMModel
├── components/        # Groundwater, Stream, Lake, RootZone, SmallWatershed, UnsaturatedZone
├── io/                # 50+ file type readers/writers (ASCII, binary, HDF5, HEC-DSS)
├── runner/            # IWFMRunner (subprocess execution), PEST++ integration
├── visualization/
│   ├── webapi/        # FastAPI viewer: config.py, server.py, routes/, static/
│   │                  # Also contains head_loader, hydrograph_reader, slicing, properties
│   ├── vtk_export.py  # VTKExporter (2D/3D mesh, PyVista)
│   └── ...            # Matplotlib plots, GIS export
├── mesh_generation/   # Triangle and Gmsh wrappers
├── comparison/        # Model diffing and metrics
└── cli/               # Command-line interface entry points

frontend/              # React + TypeScript + vtk.js + deck.gl (builds to webapi/static/)
```

### Key Classes
- **AppGrid**: Main mesh container with nodes, elements, faces, subregions (mirrors IWFM's Class_AppGrid)
- **IWFMModel**: Orchestrates all components (grid, stratigraphy, groundwater, streams, lakes, rootzone, small watersheds, unsaturated zone)
- **BaseReader/BaseWriter**: Abstract I/O classes in `io/base.py` and `io/writer_base.py`

### I/O System
The `io/` module handles 50+ IWFM file formats. Key patterns:
- Readers parse IWFM ASCII/binary files into Python objects
- Writers use Jinja2 templates from `templates/` directory
- Comment preservation during roundtrip (read → write) via `comment_extractor.py`
- `load_complete_model()` / `save_complete_model()` for full model I/O
- Each component (groundwater, streams, lakes, rootzone, etc.) has dedicated reader and writer modules

### Web Viewer Architecture
The viewer is a FastAPI backend + React SPA frontend with 4 tabs: Overview, 3D Mesh (vtk.js), Results Map (deck.gl + MapLibre), and Budgets (Plotly).

**Backend** (`visualization/webapi/`):
- `config.py` — `ModelState` singleton that holds the loaded `IWFMModel` and provides lazy getters for head data, budget data, stream reach boundaries, etc.
- `server.py` — FastAPI app creation with CRS configuration and static file serving
- `routes/` — 14 route modules: model, mesh, results, groundwater, streams, lakes, rootzone, small_watersheds, budgets, export, observations, slices, properties
- `head_loader.py` — `LazyHeadDataLoader` reads HDF5 head results on demand
- `hydrograph_reader.py` — Parses IWFM `.out` text hydrograph files
- Coordinate reprojection: server-side via `pyproj` (model CRS → WGS84), `--crs` CLI flag

**Frontend** (`frontend/`):
- State management: Zustand store in `stores/viewerStore.ts`
- API client: `api/client.ts` (typed fetch wrappers)
- 3D rendering: vtk.js in `components/Viewer3D/`
- 2D map: deck.gl + MapLibre in `components/ResultsMap/`
- Charts: Plotly in `components/BudgetDashboard/`
- URL hash routing for tab navigation (#overview, #3d, #results, #budgets)
- Builds to `src/pyiwfm/visualization/webapi/static/` via Vite

### PEST++ Integration
`runner/pest*.py` modules provide parameter estimation workflow:
- Template/instruction file generation
- Parameter bounds and groups
- Observation management
- Ensemble methods (prior/posterior)

## Code Style

- Python 3.10+ with type hints required (mypy strict mode)
- Line length: 100 characters
- NumPy-style docstrings
- Uses 1-based IDs to match IWFM Fortran conventions
- Pre-commit hooks: trailing whitespace, ruff lint/format, mypy

## IWFM Domain Conventions

- All node/element/reach IDs are 1-based (Fortran convention)
- Datetime format in IWFM files: `MM/DD/YYYY_HH:MM` (note `_24:00` means end of day → next day 00:00)
- C2VSimFG CRS: UTM Zone 10N, NAD83, US survey feet (`+proj=utm +zone=10 +datum=NAD83 +units=us-ft +no_defs`). This is NOT EPSG:2227.
- Monthly budget timesteps (`1MON`) vary 28–31 days; use `dateutil.relativedelta`
- Stream `StrmNode` often has `x=0, y=0`; look up coordinates via `node.gw_node` → `grid.nodes[gw_node].x/y`

## TypeScript / Frontend Notes

- TypeScript strict mode enabled: unused imports/variables cause build failures (TS6133)
- Target is ES2020 — `new Map<K,V>()` with generics may not work; use `Record<string, T>` for caches
- Plotly title objects must use `{ text: 'Title' }` format, not bare strings
- All API responses should be typed; see existing patterns in `api/client.ts`

## Optional Dependencies

Many modules have optional imports. Handle ImportError gracefully:
- `gis`: geopandas, shapely
- `mesh`: triangle, gmsh
- `viz`: vtk, matplotlib
- `webapi`: fastapi, uvicorn, pydantic, pyvista, vtk, pyproj, python-multipart
- `dss`: pyhecdss (HEC-DSS 7 support)
