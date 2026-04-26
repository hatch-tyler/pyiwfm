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

### `pyiwfm.io.head_loader.LazyHeadDataLoader`

**Status:** _shimmed_ (DeprecationWarning in v2.x, removed in v3.0).

**Why:** Replaced by the source-agnostic `LazyTimeSeriesLoader`, which
also covers hydrograph data and any future per-node time-series
formats.

**v1.x:**
```python
from pyiwfm.io.head_loader import LazyHeadDataLoader
loader = LazyHeadDataLoader("Results/HeadAll.hdf")
heads_at_t0 = loader[loader.times[0]]
```

**v2.x:**
```python
from pyiwfm.io.timeseries_io import LazyTimeSeriesLoader
loader = LazyTimeSeriesLoader("Results/HeadAll.hdf", source="iwfm_head_hdf5")
heads_at_t0 = loader[loader.times[0]]
```

The `__getitem__`, `times`, `n_frames`, `n_nodes`, `n_layers` interface
is unchanged.

### `pyiwfm.io.head_all_converter.convert_headall_to_hdf`

**Status:** _shimmed_ (DeprecationWarning in v2.x, removed in v3.0).

**Why:** Replaced by `TimeSeriesCache.from_text()`, which handles head
and hydrograph text outputs through one path.

**v1.x:**
```python
from pyiwfm.io.head_all_converter import convert_headall_to_hdf
convert_headall_to_hdf("GWALLOUTFL.out", "HeadAll.hdf")
```

**v2.x:**
```python
from pyiwfm.io.timeseries_io import TimeSeriesCache
TimeSeriesCache.from_text(
    "GWALLOUTFL.out",
    output="HeadAll.hdf",
    source="iwfm_head_text",
)
```

### `pyiwfm.io.hydrograph_loader.HydrographLoader`

**Status:** _shimmed_.

> _Fill in v1 → v2 example when PR 1 lands._

### `pyiwfm.io.hydrograph_converter.convert_hydrograph_to_hdf`

**Status:** _shimmed_.

> _Fill in v1 → v2 example when PR 1 lands._

### `pyiwfm.io.hydrograph_reader.IWFMHydrographReader`

**Status:** _kept_ — name and import path unchanged. Loses its
conversion responsibility (use `TimeSeriesCache` for that), but the
text-reading interface is identical.

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

## 3. Broaden the `BaseComponent` contract

> **⚠ Hard break.** This is the only v2.0 change without a shim. If
> you subclass :class:`~pyiwfm.core.base_component.BaseComponent` (rare
> — it's intended for internal pyiwfm use), your subclass must
> implement four new abstract methods. Code that only **uses**
> components (without subclassing) is unaffected.

### `BaseComponent` — new abstract methods

**Status:** _hard break_ for subclassers.

**Why:** Phase-1 verification confirmed `BaseComponent` is already
an ABC with `validate()` and `n_items`. The four new methods enable
generic JSON / HDF5 serialization, scenario diff, deep-copy, and
mesh-aware validation across all components — without per-class
bespoke code.

**v1.x:**
```python
from pyiwfm.core.base_component import BaseComponent

class MyCustomComponent(BaseComponent):
    def validate(self) -> list[str]:
        return []

    @property
    def n_items(self) -> int:
        return 0
```

**v2.x:**
```python
from pyiwfm.core.base_component import BaseComponent
from pyiwfm.core.serialization import default_to_dict, default_from_dict

class MyCustomComponent(BaseComponent):
    def validate(self) -> list[str]:
        return []

    @property
    def n_items(self) -> int:
        return 0

    # NEW required methods — for simple dataclass-style components,
    # the helpers in pyiwfm.core.serialization implement these for you:
    def to_dict(self) -> dict[str, object]:
        return default_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "MyCustomComponent":
        return default_from_dict(cls, data)

    def clone(self) -> "MyCustomComponent":
        return self.from_dict(self.to_dict())

    def validate_against(self, grid) -> list[str]:
        # Mesh-aware validation: e.g. check that all node references
        # exist in the supplied grid. Return [] if no mesh-cross-checks
        # apply.
        return []
```

For component classes that already define `to_dict` / `from_dict`
informally (most of pyiwfm's built-in components do), you can simply
remove the `default_*` helpers and the existing methods satisfy the
contract.

If you need to interact with a `BaseComponent` polymorphically (across
multiple subclasses), use the new contract:

```python
# v2 generic snapshot:
snapshot = {name: comp.to_dict() for name, comp in model.iter_components()}

# v2 generic diff:
def diff(a: BaseComponent, b: BaseComponent) -> dict:
    return {k: (a.to_dict()[k], b.to_dict()[k])
            for k in a.to_dict()
            if a.to_dict().get(k) != b.to_dict().get(k)}
```

### `BaseComponent.validate()` return type

**Status:** _signature clarified_, not a hard break for most callers.

**Why:** Phase-1 verification found the v1 docstring was internally
inconsistent — the class docstring said `validate()` returns "a list
of validation error strings", but the method signature said
`-> None`. v2.0 commits to the list-returning convention.

**v1.x:**
```python
errors = component.validate()  # returned None on some subclasses,
                               # list[str] on others — caller had to guess
```

**v2.x:**
```python
errors: list[str] = component.validate()
if errors:
    raise ValidationError(f"{component} failed validation: {errors}")
```

Callers that only checked truthiness (`if not errors:`) continue to
work because `None` and `[]` are both falsy — but the new contract is
unambiguous.

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
