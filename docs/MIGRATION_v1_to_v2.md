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

Both modules were full implementations (600+ lines each), not
webapi-specific shims — `slicing.py` is pure PyVista mesh slicing,
`properties.py` is pure metadata/lookup tables. Neither has any
web-only dependency. v2.0 PR 4 moved both to the `io/` namespace
where they belong alongside other domain logic. **The old paths no
longer exist (no shim).**

### `pyiwfm.visualization.webapi.slicing` → `pyiwfm.io.slicing`

**Status:** _hard rename_ — module deleted from `webapi/`, recreated
under `io/`. ``git mv`` was used so the file history follows the move.

**v1.x:**
```python
from pyiwfm.visualization.webapi.slicing import SlicingController
```

**v2.x:**
```python
from pyiwfm.io.slicing import SlicingController
```

The `SlicingController` class API itself is unchanged.

### `pyiwfm.visualization.webapi.properties` → `pyiwfm.io.properties`

**Status:** _hard rename_ — module moved.

**v1.x:**
```python
from pyiwfm.visualization.webapi.properties import PropertyVisualizer, PROPERTY_INFO
```

**v2.x:**
```python
from pyiwfm.io.properties import PropertyVisualizer, PROPERTY_INFO
```

---

## 5. Split `runner/pest.py` into a package (writer/cache splits deferred)

### `pyiwfm.runner.pest` — single module → package

**Status:** _internal restructuring_; **public API unchanged**.

The 1,456-line `runner/pest.py` is now a package whose `__init__.py`
re-exports every previously-public name. All your existing imports
continue to work without warnings:

```python
# Still works in v2.x:
from pyiwfm.runner.pest import (
    Parameter,
    Observation,
    ObservationGroup,
    TemplateFile,
    InstructionFile,
    PESTInterface,
    write_pest_control_file,
)
```

Internal layout (only relevant if you imported from internal helpers,
which is rare):

| v1 | v2 |
|---|---|
| `pyiwfm.runner.pest` (single 1,456-line module) | `pyiwfm.runner.pest` (package: `parameter`, `observation`, `template`, `instruction`, `interface`, `write_control_file`) |

The `PESTInterface` class itself stays whole — its 850 lines are tightly
coupled around shared instance state and would require either a mixin
chain or extracting methods to module-level helpers, both of which add
indirection without functional gain.

### What was deferred and why

The original v2.0 roadmap also called for splitting three other large
modules:

- `io/gw_writer.py` (1,161 lines) — single `GWComponentWriter` class
- `io/stream_writer.py` (1,102 lines) — single `StreamComponentWriter` class
- `io/cache_builder.py` (964 lines, sub-1,000 — was at the threshold)

A v2.0 PR 5 audit confirmed all three are single-class modules with
methods tightly coupled to shared instance state (the writer
configuration, file handles, format flags). Splitting would require
either:

1. **Mixin classes** — `class GWComponentWriter(MainMixin, BCMixin, ...)`,
   adding MRO indirection without changing behavior.
2. **Free functions** — extracting methods as module-level functions
   that take `self` as their first argument, which is the same as
   methods called externally.

Both add abstractions for purely cosmetic gains (file boundaries
inside one logical unit) and risk subtle behavior changes in
attribute-access ordering. The CLAUDE.md guidance — "don't add
abstractions beyond what the task requires" — pushes toward keeping
these modules whole until a real second consumer of one of these
sections appears (which would be the architectural seam that justifies
the split). When that happens, the refactor can ship in a non-major
release alongside the new caller.

For navigation, all three modules already use comment-banner section
headers so a reader can jump to "BC writing" or "subsidence section"
without `cd`-ing into a subdirectory.

---

## 6. Rootzone v5+ reader consolidation

### `pyiwfm.io._rootzone_base._RootzoneReaderBase` (new internal base)

**Status:** _internal restructuring_; **public API unchanged**.

In v1.x the three v5+ rootzone variant readers
(:class:`~pyiwfm.io.rootzone_native.NativeRiparianReader`,
:class:`~pyiwfm.io.rootzone_nonponded.NonPondedCropReader`,
:class:`~pyiwfm.io.rootzone_urban.UrbanLandUseReader`) each carried
their own copies of:

- `_resolve(base_dir, filepath)` — byte-identical 4-line method, 3
  copies.
- `_read_rows(buf, min_cols, [n_expected])` — ~30-line tabular-section
  reader, 3 copies (urban differed only in accepting a per-call
  ``n_expected`` override).
- Inline `try: float(val); except ValueError: raise FileFormatError(...)`
  blocks at every scalar-read site (~6 places per reader × 3 readers).

PR 6 extracts this scaffolding into ``pyiwfm.io._rootzone_base._RootzoneReaderBase``
(168 lines) with shared `_resolve`, `_read_rows`, `_parse_float`, and
`_parse_int` helpers. The three concrete readers now inherit from it.

**Per-module line reductions:**

| Module | v1.x lines | v2.x lines | Δ |
|---|---|---|---|
| `rootzone_native.py` | 318 | 255 | −63 |
| `rootzone_nonponded.py` | 458 | 389 | −69 |
| `rootzone_urban.py` | 367 | 299 | −68 |
| `_rootzone_base.py` (new) | — | 168 | +168 |
| **Net** | **1,143** | **1,111** | **−32** |

The bigger win isn't the line count — it's that the readers no longer
have triplicate copies of the same parsing scaffolding to keep in sync,
and a future v5+ rootzone variant slots in by inheriting the same base.

**v4.x readers are unchanged.** ``rootzone_v4x.py`` already had its
own ``_V4xReaderBase`` shipped before v2.0; that base remains and was
not touched. The original v2.0 roadmap proposed unifying v4.x and v5+
under one common ABC, but the two file-format generations have
sufficiently different shapes (different scalar-field orders, different
section types) that a single base would just push the variation into a
config object — adding indirection without functional gain.

**No public API changes.** All three concrete reader classes keep the
same constructor signatures, public ``read()`` method, and return
types. External callers don't see the inheritance change.

If you imported one of the now-removed private helpers
(`_is_comment_line`, `_LineBuffer`, `_strip_comment` re-exports) from
``pyiwfm.io.rootzone_nonponded`` directly (rare; the test suite did
this in two places before PR 6), update to import from the canonical
:mod:`pyiwfm.io.iwfm_reader`:

```python
# v1.x:
from pyiwfm.io.rootzone_nonponded import _is_comment_line

# v2.x:
from pyiwfm.io.iwfm_reader import is_comment_line as _is_comment_line
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
