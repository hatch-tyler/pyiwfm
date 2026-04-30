# Migrating from pyiwfm v1.x to v2.0

> **Status:** v2.0 implementation complete on the
> [`next` branch](V2_ROADMAP.md) as of 2026-04-27. Seven PRs landed
> (see [§ Implementation status](#implementation-status) below). This
> guide is ready to ship with the `v2.0.0a1` tag. The maintainer
> tagging the alpha should also remove this file from `exclude_patterns`
> in `docs/conf.py` so it appears in the published docs.

This guide walks you through upgrading a project from pyiwfm v1.x to
v2.0. See the [v2 roadmap](V2_ROADMAP.md) for the design rationale and
the per-PR audit notes.

## TL;DR

- **No deprecation shims.** v2.0 is a clean break — every renamed
  module / class / function landed as a hard rename. The migration
  guide (this document) carries the rename mapping.
- **What changed in practice is small.** PR 1 renamed two loader
  classes and consolidated converter functions; PR 4 moved two files
  from `webapi/` to `io/`. Everything else (PRs 2, 5, 6, 7) was
  internal restructuring with public APIs unchanged.
- **`IWFMModel.from_*` classmethods are unchanged** — PR 2 split their
  bodies into `core/loaders/` but the dispatchers stay.
- **`BaseComponent` users:** the v1.x docstring was misleading about
  `validate()` returning a list of error strings. PR 3 corrected the
  docstring; the actual contract (raise `ComponentError` on invalid
  state, return `None` on success) was already what every shipped
  component did. **No new abstract methods were added** — the
  speculative `to_dict`/`from_dict`/`clone`/`validate_against`
  proposals were deferred (see [Section 3](#3-basecomponent-docstring-fix-pr-3)).

## Try v2 today

```bash
# Install the latest v2 alpha (pip won't pick this up by default)
pip install --pre 'pyiwfm>=2.0.0a1,<3'

# Run your existing v1 code
python your_script.py
```

If your code is affected by the renames in [Sections 1](#1-headhydrograph-cluster-consolidation)
or [4](#4-move-webapislicingpy-and-webapipropertiespy-to-io), you'll
see `ImportError` on first run. The error message names the missing
module — match it against the corresponding section below to find the
new path.

## Implementation status

All seven PRs landed on the `next` branch between 2026-04-27 and
the v2.0.0a1 tag. The "Outcome" column flags what differed from the
original [v2 roadmap](V2_ROADMAP.md) targets:

| § | PR | Topic | Outcome |
|---|---|---|---|
| [1](#1-headhydrograph-cluster-consolidation) | 1 | Head/hydrograph cluster | Two classes (`LazyNodalLoader`, `LazyTabularLoader`) instead of one — divergent data shapes |
| [2](#2-coremodelpy-constructor-split) | 2 | `core/model.py` constructor split | As designed: 2,498 → 958 lines |
| [3](#3-basecomponent-docstring-fix-pr-3) | 3 | `BaseComponent` contract | **Deferred** speculative `to_dict`/`from_dict`/`clone`/`validate_against` (no current caller); shipped docstring fix only |
| [4](#4-move-webapislicingpy-and-webapipropertiespy-to-io) | 4 | webapi → io move | As designed: pure rename |
| [5](#5-split-runnerpestpy-into-a-package-writercache-splits-deferred) | 5 | Large writer splits | **Partial:** `runner/pest.py` split as designed; `gw_writer`/`stream_writer`/`cache_builder` deferred (single-class methods, would need mixins for cosmetic gain) |
| [6](#6-rootzone-v5-reader-consolidation) | 6 | Rootzone v5+ ABC | **Smaller than projected:** plan's −1,575 line target was based on assuming all 5 modules duplicated; actual was 3 modules and netted −32 lines, but real win is the architectural seam |
| [7](#7-component-writer-scaffolding-pr-7) | 7 | BaseComponentWriter ABC | **Partial:** shipped `open_iwfm_file` + `write_element_group` shared helpers; deferred the speculative `BaseComponentWriter` strategy class (per-writer format diversity makes a uniform `WriteSpec` schema awkward) |

The "deferred" items in PRs 3, 5, and 7 follow the same pattern: an
audit found no concrete caller would benefit from the proposed
abstraction, and CLAUDE.md's "don't add abstractions beyond what the
task requires" pushes back. Each deferral is documented in its
section below with a recovery path if a real consumer appears later.

## How to read this guide

The seven subsections below correspond one-to-one with the seven PRs
that shipped to v2.0:

1. [Head/hydrograph cluster consolidation (PR 1)](#1-headhydrograph-cluster-consolidation)
2. [`core/model.py` constructor split (PR 2)](#2-coremodelpy-constructor-split)
3. [`BaseComponent` docstring fix (PR 3)](#3-basecomponent-docstring-fix-pr-3)
4. [Move `webapi/slicing.py` and `webapi/properties.py` to `io/` (PR 4)](#4-move-webapislicingpy-and-webapipropertiespy-to-io)
5. [Split `runner/pest.py` into a package (PR 5)](#5-split-runnerpestpy-into-a-package-writercache-splits-deferred)
6. [Rootzone v5+ reader consolidation (PR 6)](#6-rootzone-v5-reader-consolidation)
7. [Component-writer scaffolding (PR 7)](#7-component-writer-scaffolding-pr-7)

Within each section, every individual API change uses this template:

> ### `<old qualified name>` → `<new qualified name>`
>
> **Status:** _hard rename_ (v2.0 requires the code change) **or**
> _internal restructuring_ (no public-API change).
>
> **v1.x:**
> ```python
> # the old code
> ```
>
> **v2.x:**
> ```python
> # the new code
> ```

---

## 1. Head/hydrograph cluster consolidation

Consolidates 1,402 lines across five near-duplicate modules into two:
``pyiwfm.io.timeseries_io`` (the new lazy-loader + cache namespace)
and ``pyiwfm.io.hydrograph_reader`` (kept as the eager text reader).
The five v1.x modules are **deleted in v2.0** — there are no
deprecation shims. Entries below document the canonical replacement
for each public name; update your imports accordingly.

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
:mod:`pyiwfm.io.ascii.reader`:

```python
# v1.x:
from pyiwfm.io.rootzone_nonponded import _is_comment_line

# v2.x:
from pyiwfm.io.ascii.reader import is_comment_line as _is_comment_line
```

---

## 7. Component-writer scaffolding (PR 7)

### `pyiwfm.io.ascii.writer.open_iwfm_file` and `write_element_group` (new helpers)

**Status:** _internal restructuring_; **no public API change**.

Six small component-writer modules
(``gw_boundary_writer``, ``gw_pumping_writer``, ``gw_tiledrain_writer``,
``stream_diversion_writer``, ``stream_bypass_writer``,
``stream_inflow_writer``) all opened a similar wrapper at the top of
every ``write_*`` function: convert ``filepath`` to ``Path``, call
``ensure_parent_dir``, ``open(...)`` for write, write the standard
``C  IWFM ...`` header. PR 7 collapses that 4-line pattern into a
single ``with open_iwfm_file(filepath, header) as f:`` context
manager (in :mod:`pyiwfm.io.ascii.writer`).

In addition, the element-group block format
(``GroupID  N_elements  FirstElement`` followed by element-per-line)
appeared inline three times across the writers (twice in
``gw_pumping_writer.py``, once as a private helper in
``stream_diversion_writer.py``). PR 7 extracts a shared
:func:`pyiwfm.io.ascii.writer.write_element_group` and routes all three
sites through it.

**Per-module line reductions:**

| Module | v1.x lines | v2.x lines | Δ |
|---|---|---|---|
| ``gw_boundary_writer.py`` | 172 | 144 | −28 |
| ``gw_pumping_writer.py`` | 148 | 122 | −26 |
| ``gw_tiledrain_writer.py`` | 81 | 69 | −12 |
| ``stream_diversion_writer.py`` | 193 | 172 | −21 |
| ``stream_bypass_writer.py`` | 106 | 94 | −12 |
| ``stream_inflow_writer.py`` | 67 | 55 | −12 |
| **Net (consolidated)** | **767** | **656** | **−111** |
| ``iwfm_writer.py`` (shared, +helpers) | 70 | 142 | +72 |
| **Total replacement** | **837** | **798** | **−39** |

The bigger win again isn't the line count — it's that the boilerplate
no longer sits at the top of every writer where it can drift, and
the element-group format change can be made in one place.

### What was deferred and why

The original v2.0 roadmap proposed a strategy-based
``BaseComponentWriter`` class hierarchy where each writer module would
become a small config dataclass (``WriteSpec``: header, sections,
formatters) consumed by a base class. A PR 7 audit found this would
not produce the projected savings: each writer has unique format
strings, conditional logic for empty vs populated sections, and
factor-application patterns that don't fit a uniform ``WriteSpec``
schema. Pushing that variation into config objects would just
relocate the code, not reduce it.

The same audit-and-defer pattern as PR 5 (writer/cache splits) and
PR 3 (BaseComponent abstract methods) applies here: the duplication
that did warrant extraction (the ``open_iwfm_file`` boilerplate and
the element-group block format) was extracted; the wholesale
class-hierarchy refactor was not, because no current caller would
benefit and the project rule (CLAUDE.md "don't add abstractions
beyond what the task requires") pushes back on speculative
indirection.

### Element-group helper rename within `stream_diversion_writer`

The private ``_write_element_group(f, group)`` helper inside
``stream_diversion_writer.py`` is now a thin adapter that unwraps the
``ElementGroup`` dataclass and delegates to the shared
:func:`~pyiwfm.io.ascii.writer.write_element_group`. External callers
of ``stream_diversion_writer._write_element_group`` are unaffected
(the function signature is unchanged).

---

## 8. Mesh + stratigraphy writer unification

Two changes that landed together after the original seven v2.0 PRs but
before the v2.0.0a1 tag.

### `pyiwfm.io.ascii` → `pyiwfm.io.mesh`

**Status:** _hard rename_.

`pyiwfm.io.ascii` was the only module under `pyiwfm/io/` named after a
format rather than a domain. It contained only the six preprocessor
mesh + stratigraphy functions (`read_nodes`, `read_elements`,
`read_stratigraphy`, and the matching writers) — not generic ASCII
utilities. The generic-reader role is filled by
`pyiwfm.io.ascii.reader`. The module was renamed to align with the
domain-by-component pattern used everywhere else in `pyiwfm/io/`.

Top-level re-exports through `pyiwfm.io` keep the same names, so most
callers are unaffected.

**v1.x:**

```python
from pyiwfm.io.ascii import read_nodes, write_nodes
```

**v2.x:**

```python
from pyiwfm.io.mesh import read_nodes, write_nodes

# Or — preferred — the unchanged top-level re-export:
from pyiwfm.io import read_nodes, write_nodes
```

`mock.patch` strings targeting the old path must update:

```python
# v1.x
patch("pyiwfm.io.ascii.read_nodes", ...)

# v2.x
patch("pyiwfm.io.mesh.read_nodes", ...)
```

### `write_nodes_file` / `write_elements_file` / `write_stratigraphy_file` removed

**Status:** _hard removal_ — the array-shape standalone writers in
`pyiwfm.io.preprocessor_writer` are deleted. Use the canonical
`pyiwfm.io.mesh.write_*` functions, which take the domain types
directly.

Three parallel writer surfaces previously coexisted: `mesh.write_*`
took `dict[int, Node]`/`dict[int, Element]`/`Stratigraphy`,
`PreProcessorWriter.write_*` took an `IWFMModel`, and the standalone
`write_*_file` functions took raw NumPy arrays — each with its own
slightly-different output format. They are now collapsed onto a single
canonical implementation in `pyiwfm.io.mesh`. The class methods on
`PreProcessorWriter` are thin orchestration delegates; the array-shape
standalones are gone (zero production callers existed).

**v1.x:**

```python
import numpy as np
from pyiwfm.io.preprocessor_writer import write_nodes_file

write_nodes_file(
    "output/nodes.dat",
    node_ids=np.array([1, 2, 3], dtype=np.int32),
    x_coords=np.array([0.0, 100.0, 200.0]),
    y_coords=np.array([0.0, 100.0, 100.0]),
    coord_factor=0.3048,
)
```

**v2.x:**

```python
from pyiwfm.io.mesh import write_nodes
from pyiwfm.core.mesh import Node

nodes = {
    int(nid): Node(id=int(nid), x=float(x), y=float(y))
    for nid, x, y in zip(node_ids, x_coords, y_coords)
}
write_nodes("output/nodes.dat", nodes, factor=0.3048)
```

The same construct-then-call pattern applies to elements
(`{int(eid): Element(id=int(eid), vertices=tuple(verts), subregion=int(sr))
for eid, verts, sr in zip(...)}`) and stratigraphy (build a
`pyiwfm.core.stratigraphy.Stratigraphy` from your arrays via its
`from_thicknesses` classmethod and pass to `mesh.write_stratigraphy`).

The canonical writer always emits `FACTXY` (nodes) / `FACTEL`
(stratigraphy) factor lines — pass `factor=` to override the default of
`1.0`. Subregion names round-trip through `mesh.write_elements` via
`subregion_names=`; `n_subregions` is inferred from the element
subregion IDs when omitted. Custom headers via `header=` continue to
work as before.

---

## 10. `io/` restructure — format-primitive subpackages

**Status:** _internal restructuring_ for top-level callers; _hard rename_
for callers that imported submodule paths directly.

`pyiwfm/io/` is being reorganised so each on-disk format gets its own
subpackage (`hdf5/`, `binary/`, `ascii/`, `dss/`) and each IWFM model
domain gets its own subpackage (`groundwater/`, `streams/`, etc.). The
flat 84-file directory was hard to navigate; the new shape mirrors the
structure of an IWFM model.

The restructure ships **one cluster per PR**. Each section below
documents the cluster's old → new path mapping.

### `pyiwfm.io.hdf5` (was a module, now a package)

The module `pyiwfm/io/hdf5.py` is now the package `pyiwfm/io/hdf5/`
with a `model.py` submodule. The package's `__init__.py` re-exports the
public API, so the existing path `from pyiwfm.io.hdf5 import X`
continues to work unchanged for `HDF5ModelReader`, `HDF5ModelWriter`,
`read_model_hdf5`, `write_model_hdf5`.

**v1.x and v2.x — both work:**

```python
from pyiwfm.io.hdf5 import HDF5ModelReader, write_model_hdf5
```

**New direct path (optional):**

```python
from pyiwfm.io.hdf5.model import HDF5ModelReader  # v2.x only
```

`mock.patch("pyiwfm.io.hdf5.write_model_hdf5")` continues to work —
the patch target resolves through the package's re-export.

### `pyiwfm.io.binary` (was a module, now a package) and `pyiwfm.io.preprocessor_binary` (gone)

`pyiwfm/io/binary.py` and `pyiwfm/io/preprocessor_binary.py` were both
binary-format modules but lived as siblings in the flat `io/` directory.
They're now the single package `pyiwfm/io/binary/`:

- `pyiwfm/io/binary/fortran.py` — was `binary.py` (Fortran
  unformatted-sequential primitives: `FortranBinaryReader`,
  `FortranBinaryWriter`, `StreamAccessBinaryReader`,
  `write_binary_mesh`, `write_binary_stratigraphy`,
  `read_fortran_record`).
- `pyiwfm/io/binary/preprocessor.py` — was `preprocessor_binary.py`
  (IWFM preprocessor's `ACCESS='STREAM'` binary output and the
  `*Data` record dataclasses: `PreprocessorBinaryReader`,
  `read_preprocessor_binary`, `AppNodeData`, `AppElementData`,
  `LakeData`, `StreamData`, etc.).

The package `__init__.py` re-exports both submodules' public symbols,
so `from pyiwfm.io.binary import X` works for any of them.

**v1.x:**

```python
from pyiwfm.io.binary import FortranBinaryReader
from pyiwfm.io.preprocessor_binary import PreprocessorBinaryReader, AppNodeData
```

**v2.x — preferred:**

```python
from pyiwfm.io.binary import (
    FortranBinaryReader,
    PreprocessorBinaryReader,
    AppNodeData,
)
```

**v2.x — explicit submodule paths:**

```python
from pyiwfm.io.binary.fortran import FortranBinaryReader
from pyiwfm.io.binary.preprocessor import PreprocessorBinaryReader
```

The `from pyiwfm.io.preprocessor_binary import …` path is **gone**;
update to `from pyiwfm.io.binary import …`.

`mock.patch("pyiwfm.io.preprocessor_binary.X")` strings need updating
to `mock.patch("pyiwfm.io.binary.X")` (re-exported on the package) —
not `pyiwfm.io.binary.preprocessor.X`, because the consumers import
through the package re-export, not the deep submodule.

### `pyiwfm.io.ascii` (new package) — `iwfm_reader`, `iwfm_writer`, and the comment cluster (gone)

`pyiwfm/io/iwfm_reader.py`, `pyiwfm/io/iwfm_writer.py`, and the three
comment-preservation modules (`comment_extractor.py`,
`comment_metadata.py`, `comment_writer.py`) were five flat ASCII helper
modules. They're now the single package `pyiwfm/io/ascii/`:

- `pyiwfm/io/ascii/reader.py` — was `iwfm_reader.py` (line-reading
  helpers: `parse_int`, `parse_float`, `next_data_value`,
  `next_data_line`, `next_data_or_empty`, `is_comment_line`,
  `strip_inline_comment`, `resolve_path`, `parse_version`,
  `version_ge`, `ReaderMixin`, `LineBuffer`, `COMMENT_CHARS`).

- `pyiwfm/io/ascii/writer.py` — was `iwfm_writer.py` (line-writing
  helpers: `write_comment`, `write_value`, `ensure_parent_dir`,
  `open_iwfm_file`, `write_element_group`).

- `pyiwfm/io/ascii/comment_extractor.py`,
  `pyiwfm/io/ascii/comment_metadata.py`,
  `pyiwfm/io/ascii/comment_writer.py` — round-trip comment
  preservation. Public symbols unchanged: `LineType`, `ParsedLine`,
  `CommentExtractor`, `extract_comments`, `extract_and_save_comments`,
  `PreserveMode`, `SectionComments`, `CommentMetadata`,
  `FileCommentMetadata`, `CommentWriter`, `CommentInjector`.

`pyiwfm/io/ascii/__init__.py` re-exports every public symbol, so the
canonical v2.x path is `from pyiwfm.io.ascii import …`.

**v1.x:**

```python
from pyiwfm.io.iwfm_reader import parse_int, next_data_value
from pyiwfm.io.iwfm_writer import open_iwfm_file, write_comment
from pyiwfm.io.comment_extractor import CommentExtractor
from pyiwfm.io.comment_metadata import CommentMetadata
from pyiwfm.io.comment_writer import CommentWriter
```

**v2.x:**

```python
from pyiwfm.io.ascii import parse_int, next_data_value
from pyiwfm.io.ascii import open_iwfm_file, write_comment
from pyiwfm.io.ascii import CommentExtractor, CommentMetadata, CommentWriter

# Or the deeper paths if you want to be explicit about which file the
# symbol lives in:
from pyiwfm.io.ascii.reader import parse_int
from pyiwfm.io.ascii.writer import open_iwfm_file
```

The five v1.x paths (`pyiwfm.io.iwfm_reader`, `pyiwfm.io.iwfm_writer`,
`pyiwfm.io.comment_extractor`, `pyiwfm.io.comment_metadata`,
`pyiwfm.io.comment_writer`) are **gone**; update to
`from pyiwfm.io.ascii import …`.

`mock.patch("pyiwfm.io.iwfm_reader.X")` and the four sister patch
strings need updating to `mock.patch("pyiwfm.io.ascii.X")` — patch the
package re-export, not the deep submodule, because the consumers
import through the package.

### `pyiwfm.io.unsaturated_zone` (was a module, now a package) and `pyiwfm.io.unsaturated_zone_writer` (gone)

`pyiwfm/io/unsaturated_zone.py` and `pyiwfm/io/unsaturated_zone_writer.py`
were two flat modules for the same domain. They're now the single
package `pyiwfm/io/unsaturated_zone/`:

- `pyiwfm/io/unsaturated_zone/reader.py` — was `unsaturated_zone.py`
  (`UnsatZoneMainReader`, `UnsatZoneMainConfig`,
  `UnsatZoneElementData`, `read_unsaturated_zone_main`).

- `pyiwfm/io/unsaturated_zone/writer.py` — was
  `unsaturated_zone_writer.py` (`UnsatZoneComponentWriter`,
  `UnsatZoneWriterConfig`, `write_unsaturated_zone_component`).

The package `__init__.py` re-exports both submodules' public API so
the existing path `from pyiwfm.io.unsaturated_zone import X` keeps
working for every reader symbol and now also resolves every writer
symbol (which previously needed
`from pyiwfm.io.unsaturated_zone_writer import …`).

**v1.x:**

```python
from pyiwfm.io.unsaturated_zone import UnsatZoneMainReader
from pyiwfm.io.unsaturated_zone_writer import (
    UnsatZoneComponentWriter,
    write_unsaturated_zone_component,
)
```

**v2.x:**

```python
from pyiwfm.io.unsaturated_zone import (
    UnsatZoneMainReader,
    UnsatZoneComponentWriter,
    write_unsaturated_zone_component,
)
```

The `from pyiwfm.io.unsaturated_zone_writer import …` path is **gone**;
update to `from pyiwfm.io.unsaturated_zone import …`.

`mock.patch("pyiwfm.io.unsaturated_zone_writer.X")` strings need
updating to `mock.patch("pyiwfm.io.unsaturated_zone.X")` (re-exported
on the package) — not `pyiwfm.io.unsaturated_zone.writer.X`, because
consumers import through the package re-export, not the deep
submodule.

---

## 9. Strict-by-default loading at user-facing surfaces

**Status:** _behaviour change_ at the CLI; opt-out flag provided.

In v1.x and the early v2.0 alphas, CLI invocations like
`pyiwfm viewer --model-dir ./model` would silently succeed even when
some component files (streams, lakes, root zone) failed to parse. The
broken components were dropped, the partially-loaded model was
returned, and the only signal was a `WARNING` log line that most users
never noticed. A user could spend an hour wondering why streams
weren't visible in the viewer.

v2.0 makes the user-facing CLI surfaces strict by default:

- `pyiwfm viewer` and `pyiwfm export` now pass `strict="collect"` to
  the loader. If any component fails, the loader runs through every
  remaining component, then raises a single `ValidationError` listing
  all failures. The CLI top-level handler (see [§ 8 Tighten exception
  handling](#8-mesh--stratigraphy-writer-unification)) prints a clean
  one-line error per failure and exits with code 1.
- A new `--allow-partial-load` flag opts back into the historical
  permissive behaviour: components that fail to parse are recorded on
  `model.load_errors` and the CLI prints a stderr banner reminding the
  user they're running degraded.

**v1.x:**

```console
$ pyiwfm viewer --model-dir broken-model/
INFO: Loading IWFM model...
INFO: Server started on http://127.0.0.1:8080
# Streams component silently dropped — only the WARNING log line
# (often suppressed by default logging config) records it.
```

**v2.x:**

```console
$ pyiwfm viewer --model-dir broken-model/
error: ValidationError: 2 component(s) failed to load
$ echo $?
1

# Opt out if you really want a degraded server:
$ pyiwfm viewer --model-dir broken-model/ --allow-partial-load
warning: model loaded with 2 component error(s); drop --allow-partial-load to see the full report.
INFO: Server started on http://127.0.0.1:8080
```

The library API is **unchanged**: `IWFMModel.from_preprocessor(...)`
and `IWFMModel.from_simulation_with_preprocessor(...)` still default
to `strict=False`. Only the CLI's `cli/_model_loader.load_model()`
helper changed. Scripts that wrap pyiwfm continue to work.

If you want the CLI's strict-by-default behaviour from a Python script,
pass `strict="collect"` explicitly:

```python
from pyiwfm.core.model import IWFMModel

model = IWFMModel.from_simulation_with_preprocessor(
    "Simulation/Simulation.in",
    "Preprocessor/Preprocessor.in",
    strict="collect",  # raise one ValidationError listing every failure
)
```

To inspect partial loads programmatically (when you stay with
`strict=False`):

```python
model = IWFMModel.from_preprocessor("model/Preprocessor.in")
if model.has_load_errors:
    for err in model.load_errors:
        print(f"  {err.component_name}: {err}")
```

---

## Migration checklist

Run through this list when bumping your project's pinned `pyiwfm`
version from `<2` to `>=2,<3`:

- [ ] Read [§ TL;DR](#tldr) and [§ Implementation status](#implementation-status)
- [ ] Pin to `pyiwfm>=2.0.0a1,<3` in your test environment
- [ ] Run your full test suite under v2 alpha; the failures will be
      `ImportError`s pointing at one of the renamed paths in
      [§ 1](#1-headhydrograph-cluster-consolidation) or
      [§ 4](#4-move-webapislicingpy-and-webapipropertiespy-to-io)
- [ ] For each `ImportError`, look up the new path in the corresponding
      section below. The renames are mechanical:
      - `pyiwfm.io.head_loader.LazyHeadDataLoader` → `pyiwfm.io.timeseries_io.LazyNodalLoader`
      - `pyiwfm.io.hydrograph_loader.LazyHydrographDataLoader` → `pyiwfm.io.timeseries_io.LazyTabularLoader`
      - `convert_headall_to_hdf(...)` → `TimeSeriesCache.from_iwfm_headall_text(...)`
      - `convert_hydrograph_to_hdf(...)` → `TimeSeriesCache.from_iwfm_hydrograph_text(...)`
      - `pyiwfm.visualization.webapi.{slicing,properties}` → `pyiwfm.io.{slicing,properties}`
      - `pyiwfm.io.ascii` → `pyiwfm.io.mesh` (see [§ 8](#8-mesh--stratigraphy-writer-unification))
      - `pyiwfm.io.preprocessor_writer.write_{nodes,elements,stratigraphy}_file`
        — removed; build domain objects and call
        `pyiwfm.io.mesh.write_{nodes,elements,stratigraphy}` instead
- [ ] **CLI behaviour:** `pyiwfm viewer` / `pyiwfm export` now exit
      with code 1 when any component fails to load (see [§ 9](#9-strict-by-default-loading-at-user-facing-surfaces)).
      If your CI / scripts depended on the silent partial-load
      behaviour, add `--allow-partial-load` to the invocation.
- [ ] Re-run your test suite; should be green
- [ ] Bump your project's pinned dependency to a real `>=2.0.0,<3`
      version once v2.0.0 final ships
- [ ] (Optional) Open an issue at the [pyiwfm tracker](https://github.com/hatch-tyler/pyiwfm/issues)
      if any v2 API change is missing from this guide or unclear

If you have an out-of-tree subclass of `BaseComponent` (rare —
internal use), no changes are required. PR 3 deferred the speculative
addition of `to_dict`/`from_dict`/`clone`/`validate_against` abstract
methods because no concrete caller would benefit from them; see
[§ 3](#3-basecomponent-docstring-fix-pr-3) for details.

## Getting help

- **Docs:** [user guide](user_guide/index.rst), [API reference](api/index.rst)
- **Issue tracker:** https://github.com/hatch-tyler/pyiwfm/issues —
  prefix v2-related issues with `[v2]` in the title for triage
- **Roadmap context:** [V2_ROADMAP.md](V2_ROADMAP.md)
