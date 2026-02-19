# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

pyiwfm is a Python package for working with IWFM (Integrated Water Flow Model) models developed by the California Department of Water Resources. It provides tools for reading/writing IWFM model files (ASCII, binary, HDF5, HEC-DSS formats), mesh manipulation, simulation execution, PEST++ parameter estimation, and interactive web visualization.

## Commands

### Installation
```bash
pip install -e .              # Basic install (includes matplotlib, geopandas, shapely)
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
pytest tests/ -m roundtrip             # Run roundtrip read→write→read tests
pytest tests/ -m property              # Run property-based (Hypothesis) tests
```

### Code Quality
```bash
ruff format src/ tests/                # Format code
ruff check src/ tests/ --fix           # Lint with auto-fix
mypy src/pyiwfm/                       # Type checking
pre-commit run --all-files             # Run all pre-commit hooks
```

### Documentation
```bash
sphinx-build docs docs/_build          # Build Sphinx docs (requires .[docs] extra)
```

### CLI
```bash
pyiwfm viewer --model-dir /path/to/model   # Launch the web viewer
pyiwfm export --model-dir /path/to/model   # Export to VTK/GeoPackage
```

### Frontend
```bash
cd frontend && npm install && npm run build   # Build React frontend to webapi/static/
cd frontend && npm run dev                    # Vite dev server with hot reload
cd frontend && npm run lint                   # ESLint check
```

### Docker
```bash
docker-compose up --build               # Build and start web viewer
docker run -p 8080:8080 -v /path/to/model:/model pyiwfm  # Run with model mounted
docker build -f dss-build/Dockerfile -t pyiwfm-dss .      # Full image with HEC-DSS support
docker-compose --profile dss up --build viewer-dss        # Compose with HEC-DSS
```
See `DOCKER.md` for full configuration (env vars: PORT, TITLE, MODE, MODEL_PATH).

## Architecture

### Source Layout
```
src/pyiwfm/
├── core/              # Mesh (Node, Element, AppGrid), Stratigraphy, TimeSeries, IWFMModel
├── components/        # Groundwater, Stream, Lake, RootZone, SmallWatershed, UnsaturatedZone
├── io/                # 50+ file type readers/writers (ASCII, binary, HDF5, HEC-DSS)
├── runner/            # IWFMRunner (subprocess execution), PEST++ integration, Scenario manager
├── visualization/
│   ├── webapi/        # FastAPI viewer: config.py, server.py, routes/, services/, static/
│   │                  # Also contains head_loader, hydrograph_reader, slicing, properties
│   ├── vtk_export.py  # VTKExporter (2D/3D mesh, PyVista)
│   └── ...            # Matplotlib plots, GIS export
├── templates/         # Jinja2 templates for IWFM file generation
│   ├── engine.py      # Hybrid Jinja2 + NumPy template engine
│   ├── filters.py     # Custom Jinja2 filters (fortran_float, fortran_int, etc.)
│   └── iwfm/          # Subdirs per component: groundwater, streams, lakes, rootzone, etc.
├── mesh_generation/   # Triangle and Gmsh wrappers
├── comparison/        # Model diffing and metrics
└── cli/               # CLI entry points + _model_finder.py, _model_loader.py helpers

frontend/              # React + TypeScript + vtk.js + deck.gl (builds to webapi/static/)
```

### Key Classes
- **AppGrid**: Main mesh container with nodes, elements, faces, subregions (mirrors IWFM's Class_AppGrid)
- **IWFMModel**: Orchestrates all components (grid, stratigraphy, groundwater, streams, lakes, rootzone, small watersheds, unsaturated zone)
- **BaseReader/BaseWriter**: Abstract I/O classes in `io/base.py` and `io/writer_base.py`
- **CommentAwareReader/CommentAwareWriter**: Extended base classes for roundtrip comment preservation

### I/O System
The `io/` module handles 50+ IWFM file formats. Key patterns:
- **`iwfm_reader.py`** is the central module for all IWFM line-reading utilities. It provides: `COMMENT_CHARS`, `is_comment_line()`, `strip_inline_comment()`, `next_data_value()`, `next_data_line()`, `next_data_or_empty()`, `resolve_path()`, `parse_version()`, `version_ge()`, and `LineBuffer`. All 19+ io/ readers import from this module — never duplicate these helpers.
- Readers parse IWFM ASCII/binary files into Python objects
- Writers use Jinja2 templates via `templates/engine.py` (hybrid Jinja2 headers + NumPy array output)
- Comment preservation during roundtrip (read → write) via `comment_extractor.py` and `comment_writer.py`
- `load_complete_model()` / `save_complete_model()` for full model I/O (in `io/preprocessor.py`)
- `CompleteModelLoader` / `CompleteModelWriter` for advanced loading (in `io/model_loader.py` / `io/model_writer.py`)
- Each component (groundwater, streams, lakes, rootzone, etc.) has dedicated reader and writer modules
- `head_all_converter` is intentionally NOT in `io/__init__.py`; import directly: `from pyiwfm.io.head_all_converter import convert_headall_to_hdf`

### Web Viewer Architecture
The viewer is a FastAPI backend + React SPA frontend with 4 tabs: Overview, 3D Mesh (vtk.js), Results Map (deck.gl + MapLibre), and Budgets (Plotly).

**Backend** (`visualization/webapi/`):
- `config.py` — `ModelState` singleton that holds the loaded `IWFMModel` and provides lazy getters for head data, budget data, stream reach boundaries, etc.
- `server.py` — FastAPI app creation with CRS configuration and static file serving
- `routes/` — 13 route modules: model, mesh, results, groundwater, streams, lakes, rootzone, small_watersheds, budgets, export, observations, slices, properties
- `head_loader.py` — `LazyHeadDataLoader` reads HDF5 head results on demand
- `hydrograph_reader.py` — Parses IWFM `.out` text hydrograph files
- Coordinate reprojection: server-side via `pyproj` (model CRS → WGS84), `--crs` CLI flag

**Frontend** (`frontend/`):
- State management: Zustand store in `stores/viewerStore.ts`
- API client: `api/client.ts` (typed fetch wrappers)
- 3D rendering: vtk.js in `components/Viewer3D/`
- 2D map: deck.gl + MapLibre in `components/ResultsMap/`
- Charts: Plotly in `components/BudgetDashboard/`
- UI: MUI (Material UI) components
- URL hash routing for tab navigation (#overview, #3d, #results, #budgets)
- Builds to `src/pyiwfm/visualization/webapi/static/` via Vite
- Path alias: `@` → `frontend/src/` (configured in `vite.config.ts`)
- Dev proxy: `/api` → `http://localhost:8080` (backend must be running separately)

### Test Fixtures
`tests/conftest.py` provides shared fixtures and helpers:
- `make_simple_grid()` — creates a 2x2 quad AppGrid (9 nodes, 4 elements) for unit tests
- `make_simple_stratigraphy()` — creates uniform-layer Stratigraphy for unit tests
- `fixtures_path` / `small_model_path` — paths to test fixture data in `tests/fixtures/`
- Integration tests (roundtrip, preprocessor/simulation runs) live in `tests/integration/`

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
- Pre-commit hooks: trailing whitespace, end-of-file-fixer, check-yaml, check-toml, check-added-large-files, ruff lint/format, mypy
- Ruff rules: E, W, F, I (isort), B (bugbear), C4 (comprehensions), UP (pyupgrade), NPY

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
- Build step is `tsc && vite build` — TypeScript errors block the build

## Optional Dependencies

Core dependencies include matplotlib, geopandas, shapely, and pyogrio (always installed). The following extras have optional imports — handle ImportError gracefully:
- `mesh`: triangle, gmsh
- `viz`: vtk
- `webapi`: fastapi, uvicorn, pydantic, pyvista, vtk, pyproj, python-multipart
- `dss`: bundled HEC-DSS 7 C library (`io/dss/lib/hecdss.dll`) with ctypes wrapper; no extra install needed. On Linux, build `libhecdss.so` from source via `dss-build/` (see `dss-build/build_hecdss.py`). Set `HECDSS_LIB` env var to override library path.
- `pest`: scipy
