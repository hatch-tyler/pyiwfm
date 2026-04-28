# Migrating from pyiwfm v1.x to v2.0

> **Status:** Template / work in progress. v2.0 is being developed on
> the [`next` branch](V2_ROADMAP.md). This document is currently
> excluded from the user-facing docs build via `docs/conf.py`. **Move
> it into the user-guide toctree when `v2.0.0a1` is tagged**, and
> announce it from the v2.0.0a1 PyPI release notes.

This guide walks you through upgrading a project from pyiwfm v1.x to
v2.0. v2.0 is a major version with intentional breaking changes; see
the [v2 roadmap](V2_ROADMAP.md) for the design rationale.

## TL;DR

- **Most code keeps working unchanged in v2.x** thanks to deprecation
  shims at every removed import path.
- **You'll see `DeprecationWarning`s** on the first use of any moved
  or renamed API. Each warning includes the new path.
- **Shims are removed in v3.0**, so plan to migrate before then.
- **One subclassing change is a hard break** (no shim possible): if
  you subclass `BaseComponent`, you must implement four new abstract
  methods. See [Section 3](#3-broaden-the-basecomponent-contract).

## Try v2 today

```bash
# Install the latest v2 alpha (pip won't pick this up by default)
pip install --pre 'pyiwfm>=2.0.0a1,<3'

# Run your existing v1 code
python your_script.py
# → Watch for DeprecationWarnings; each one points to the new API.
```

To make warnings visible in scripts that suppress them:

```python
import warnings
warnings.filterwarnings("default", category=DeprecationWarning, module=r"pyiwfm.*")
```

## How to read this guide

The five subsections below correspond one-to-one with the five PRs
that land in v2.0 (see `V2_ROADMAP.md` § 3 for the breakdown):

1. [Head/hydrograph cluster consolidation](#1-headhydrograph-cluster-consolidation)
2. [`core/model.py` constructor split](#2-coremodelpy-constructor-split)
3. [Broaden the `BaseComponent` contract](#3-broaden-the-basecomponent-contract)
4. [Move `webapi/slicing.py` and `webapi/properties.py` to `io/`](#4-move-webapislicingpy-and-webapipropertiespy-to-io)
5. [Split the 1,000+-line writers](#5-split-the-1000-line-writers)

Within each section, every individual API change uses this template:

> ### `<old qualified name>`
>
> **Status:** _shimmed_ (continues working with `DeprecationWarning` through
> v2.x, removed in v3.0) **or** _hard break_ (no shim; v2.0 requires the
> code change).
>
> **Why:** one-sentence reason for the change. Link to the PR/commit
> if it's not obvious from `V2_ROADMAP.md`.
>
> **v1.x:**
> ```python
> # the old code
> ```
>
> **v2.x (preferred):**
> ```python
> # the new code
> ```

The maintainer adding the change is responsible for filling in their
section's entries when their PR lands. PRs that introduce a public-API
change but don't update this file should be requested-changes by the
reviewer.

---

## 1. Head/hydrograph cluster consolidation

Consolidates 1,384 lines across five near-duplicate modules into a
single `pyiwfm.io.timeseries_io` abstraction. The five removed modules
keep working as deprecation shims through v2.x; entries below document
the canonical replacement for each public name.

> _Entries below are placeholders; they get filled in when PR 1 lands._

### `pyiwfm.io.head_loader.LazyHeadDataLoader` → `pyiwfm.io.timeseries_io.LazyNodalLoader`

**Status:** _hard break_ — module deleted, class renamed. No deprecation
shim (v2.0 is the breaking-release window — see the design notes in PR 1
for the rationale on why a clean rename was preferred over shims).

**Why:** The data shape (3-D nodal `(t, n, layer)`) is what the class
actually represents; the new name reflects that. The shared LRU + HDF5
scaffolding now lives in a private mixin used by both
`LazyNodalLoader` and `LazyTabularLoader`.

**v1.x:**
```python
from pyiwfm.io.head_loader import LazyHeadDataLoader
loader = LazyHeadDataLoader("Results/HeadAll.hdf", n_layers=4)
heads_at_t0 = loader[loader.times[0]]
```

**v2.x:**
```python
from pyiwfm.io.timeseries_io import LazyNodalLoader
loader = LazyNodalLoader("Results/HeadAll.hdf", n_layers=4)
heads_at_t0 = loader[loader.times[0]]
```

The constructor signature, `__getitem__`, `times`, `n_frames`,
`n_nodes`, `n_layers`, `shape`, `data_type`, `get_frame`, `get_head`,
`get_composite_subsidence`, `get_layer_range`, `to_dict`, and
`clear_cache` are all preserved verbatim.

### `pyiwfm.io.hydrograph_loader.LazyHydrographDataLoader` → `pyiwfm.io.timeseries_io.LazyTabularLoader`

**Status:** _hard break_ — module deleted, class renamed.

**Why:** Same shape-naming argument as above — this loader handles
flat `(t, columns)` data, hence "tabular".

**v1.x:**
```python
from pyiwfm.io.hydrograph_loader import LazyHydrographDataLoader
loader = LazyHydrographDataLoader("hydrograph_cache.hdf")
times, vals = loader.get_time_series(col_idx=0)
```

**v2.x:**
```python
from pyiwfm.io.timeseries_io import LazyTabularLoader
loader = LazyTabularLoader("hydrograph_cache.hdf")
times, vals = loader.get_time_series(col_idx=0)
```

All public methods (`get_row`, `get_time_series`,
`find_column_by_node_id`, plus `n_columns`, `n_timesteps`, `times`,
`hydrograph_ids`, `layers`, `node_ids`) are preserved verbatim.

### `pyiwfm.io.head_all_converter.convert_headall_to_hdf` → `pyiwfm.io.timeseries_io.TimeSeriesCache.from_iwfm_headall_text`

**Status:** _hard break_ — module deleted, function moved to a static
method on `TimeSeriesCache`.

**v1.x:**
```python
from pyiwfm.io.head_all_converter import convert_headall_to_hdf
convert_headall_to_hdf("GWALLOUTFL.out", "HeadAll.hdf", n_layers=4)
```

**v2.x:**
```python
from pyiwfm.io.timeseries_io import TimeSeriesCache
TimeSeriesCache.from_iwfm_headall_text("GWALLOUTFL.out", "HeadAll.hdf", n_layers=4)
```

The argument order and semantics are unchanged. The CLI entrypoint
(`python -m pyiwfm.io.head_all_converter ...`) was removed; if you used
it, write a small driver script that calls the static method directly.

### `pyiwfm.io.hydrograph_converter.convert_hydrograph_to_hdf` → `pyiwfm.io.timeseries_io.TimeSeriesCache.from_iwfm_hydrograph_text`

**Status:** _hard break_ — module deleted, function moved to a static
method on `TimeSeriesCache`.

**v1.x:**
```python
from pyiwfm.io.hydrograph_converter import convert_hydrograph_to_hdf
convert_hydrograph_to_hdf("GW_Hydrograph.out", "hydrograph_cache.hdf")
```

**v2.x:**
```python
from pyiwfm.io.timeseries_io import TimeSeriesCache
TimeSeriesCache.from_iwfm_hydrograph_text("GW_Hydrograph.out", "hydrograph_cache.hdf")
```

### `pyiwfm.io.hydrograph_reader.IWFMHydrographReader`

**Status:** _kept_ — name and import path unchanged. The class is
already separate from the converter (which lived in
`hydrograph_converter.py`); v2.0 just deletes the converter module —
the reader's API is identical.

---

## 2. `core/model.py` constructor split

`IWFMModel.from_*` classmethods are now thin dispatchers that delegate
to `pyiwfm.core.loaders.*`. **The classmethods themselves are
unchanged** — `IWFMModel.from_simulation(...)` etc. continue to work
identically. The new namespace is available for advanced users who
need direct access to the loader pipelines.

There is **no breaking change in this PR**, only a new optional
namespace. Entries below document the new direct-import paths in case
your code needs them.

### `pyiwfm.core.loaders` (new)

**Status:** _new addition_, not a removal.

**v1.x:** Loader implementations live in `core/model.py` as
classmethods. Direct access requires reading the source.

**v2.x:**
```python
from pyiwfm.core.loaders import (
    load_from_preprocessor,
    load_from_simulation,
    load_from_simulation_with_preprocessor,
    load_from_hdf5,
)

# Equivalent to IWFMModel.from_simulation_with_preprocessor(...)
model = load_from_simulation_with_preprocessor(
    simulation_file="Simulation/Simulation.in",
    preprocessor_file="Preprocessor/Preprocessor.in",
    strict=True,
)
```

The `IWFMModel.from_*` classmethods continue to work exactly as
before; this is purely an additional surface for callers who want it.

---

## 3. `BaseComponent` docstring fix (PR 3)

### `BaseComponent.validate()` contract

**Status:** _docstring clarified_; **no behavioral change**.

**Why:** A v2.0 PR 3 audit found the v1.x class docstring described
`validate()` as returning "a list of validation error strings" while
its method signature was `-> None` and the docstring said "Raises
ValidationError." All six shipped component implementations
(`AppGW`, `AppStream`, `AppLake`, `RootZone`, `AppSmallWatershed`,
`AppUnsatZone`) consistently raise
:class:`~pyiwfm.core.exceptions.ComponentError` on invalid state and
return ``None`` on success — the docstring was simply wrong about
the list-of-strings shape. v2.0 corrects the docstring to match.

**v1.x:**
```python
errors = component.validate()  # docstring lied — actually raises
                               # ComponentError or returns None
```

**v2.x (the contract every implementation has always followed):**
```python
try:
    component.validate()
except ComponentError as e:
    print(f"{component} failed validation: {e}")
```

### What was deferred and why

The original v2.0 roadmap also proposed adding `to_dict` / `from_dict`
/ `clone` / `validate_against(grid)` as new abstract methods on
`BaseComponent`. A concrete-call audit during PR 3 found **no current
caller** invokes any of those methods on a `BaseComponent` subclass —
the `.to_dict()` calls in the codebase are all on unrelated dataclasses
(reports, metrics). Adding 4 abstract methods × 6 component
implementations for code that nothing calls would violate the
project's "don't design for hypothetical future requirements" rule
(see CLAUDE.md "Doing tasks"), so these additions are deferred.

If a real caller emerges later (HDF5 scenario diffs, JSON snapshots,
generic deep-copy), the methods can be added in a single non-breaking
PR by giving them default implementations on `BaseComponent` itself
— they'd only become abstract if the project decides every subclass
must override them, which is a stronger constraint that needs
justification at the time.

---

## 4. Move `webapi/slicing.py` and `webapi/properties.py` to `io/`

Both modules are full implementations (600+ lines each), not
webapi-specific shims. They have no web-only dependency and belong
in the `io/` namespace alongside other domain logic. Old paths
continue to work as forwarding shims through v2.x.

### `pyiwfm.visualization.webapi.slicing`

**Status:** _shimmed_ (forwarding `import *` from new location with
DeprecationWarning).

**v1.x:**
```python
from pyiwfm.visualization.webapi.slicing import SlicingController
```

**v2.x:**
```python
from pyiwfm.io.slicing import SlicingController
```

### `pyiwfm.visualization.webapi.properties`

**Status:** _shimmed_.

**v1.x:**
```python
from pyiwfm.visualization.webapi.properties import PropertyVisualizer
```

**v2.x:**
```python
from pyiwfm.io.properties import PropertyVisualizer
```

---

## 5. Split the 1,000+-line writers

Pure file reorganization. **No public API changes** — the package's
`__init__.py` re-exports every previously-public name, so all your
existing imports continue to work without warnings.

The reorganized layout is below for reference; you only need to update
your imports if you were importing from internal submodules (rare):

| v1 | v2 |
|---|---|
| `pyiwfm.io.gw_writer` (one 1,161-line module) | `pyiwfm.io.gw_writer` (package; submodules `main`, `parameters`, `boundary_conditions`, `pumping`, `subsidence`) |
| `pyiwfm.io.stream_writer` | `pyiwfm.io.stream_writer` (package; submodules `main`, `nodes`, `reaches`, `diversions`, `bypasses`) |
| `pyiwfm.io.cache_builder` | `pyiwfm.io.cache_builder` (package; submodules `mesh`, `heads`, `budgets`, `metadata`) |
| `pyiwfm.runner.pest` | `pyiwfm.runner.pest` (package; submodules `manager`, `templates`, `instructions`, `params`, `observations`, `postprocessor`) |

If your code accesses internal helpers (rare, e.g. some test fixtures
in downstream projects), the import path within the package may have
changed:

**v1.x (internal helper access):**
```python
from pyiwfm.io.gw_writer import _format_aquifer_params_block  # internal
```

**v2.x:**
```python
from pyiwfm.io.gw_writer.parameters import _format_aquifer_params_block
```

---

## Migration checklist

Run through this list when bumping your project's pinned `pyiwfm`
version from `<2` to `>=2,<3`:

- [ ] Read [§ TL;DR](#tldr) and [§ Try v2 today](#try-v2-today)
- [ ] Pin to `pyiwfm>=2.0.0a1,<3` in your test environment
- [ ] Enable `DeprecationWarning` filtering for `pyiwfm.*` (snippet above)
- [ ] Run your full test suite under v2 alpha; capture every warning
- [ ] For each warning, follow the link to the canonical v2 path; update
      the import. Most warnings name the exact module/symbol to use.
- [ ] If you subclass `BaseComponent`, implement the four new abstract
      methods (see [§ 3](#3-broaden-the-basecomponent-contract))
- [ ] Re-run your test suite; warnings should be gone
- [ ] Bump your project's pinned dependency to a real `>=2.0.0,<3`
      version once v2.0.0 final ships
- [ ] (Optional) Open an issue at the [pyiwfm tracker](https://github.com/hatch-tyler/pyiwfm/issues)
      if any v2 API change is missing from this guide or unclear

## Getting help

- **Docs:** [user guide](user_guide/index.rst), [API reference](api/index.rst)
- **Issue tracker:** https://github.com/hatch-tyler/pyiwfm/issues —
  prefix v2-related issues with `[v2]` in the title for triage
- **Roadmap context:** [V2_ROADMAP.md](V2_ROADMAP.md)
