# Roadmap: pyiwfm v2.0

> **Status (2026-04-27):** All seven implementation PRs landed on the
> `next` branch. See [§ Implementation status](#implementation-status)
> for the per-PR outcomes. The migration guide
> ([`docs/MIGRATION_v1_to_v2.md`](MIGRATION_v1_to_v2.md)) is finalized.
> Awaiting `v2.0.0a1` tag.

## Executive Summary

v2.0 collects the breaking refactors from the comprehensive review that
intentionally were **not** shipped in v1.2.x / v1.3.x because they
remove or rename public API. They landed on the long-running `next`
branch and will ship as a single major release once the v1 line is
stable enough to leave behind.

The original roadmap targeted five Phase-3 items. Mid-implementation,
two additional items from Phase 4.B were promoted in (rootzone variant
ABC, BaseComponentWriter), bringing the actual scope to **seven PRs**.
Three of the seven (PRs 3, 5, 7) deferred speculative scope after
audit, with deferral reasoning recorded in
[`docs/MIGRATION_v1_to_v2.md`](MIGRATION_v1_to_v2.md).

None of these changes are user-functional improvements. They are
architectural cleanups that pay down accumulated debt and unblock
future work.

---

## Implementation status

| PR | Topic | Commit | Outcome vs. plan |
|---|---|---|---|
| 1 | Head/hydrograph cluster | `f37a96a` | **As designed** (with two thin classes instead of one — divergent data shapes). 5 modules → 2 (`io/timeseries_io.py` + `io/hydrograph_reader.py`). Net 1,402 → 1,275 lines. |
| 2 | `core/model.py` constructor split | `094d383` | **Exceeded.** `core/model.py` 2,498 → 958 lines (−61%). Loader bodies moved to `core/loaders/` package. |
| 3 | `BaseComponent` contract | `7a284c1` | **Partial.** Docstring fix shipped; speculative `to_dict`/`from_dict`/`clone`/`validate_against` abstract methods deferred (no current caller). |
| 4 | webapi → io move | `426faea` | **As designed.** `slicing.py` + `properties.py` moved with `git mv` (history preserved). |
| 5 | Large writer splits | `2fb3b44` | **Partial.** `runner/pest.py` 1,456 → package with 6 files. Splits of `gw_writer`/`stream_writer`/`cache_builder` deferred (single-class methods, would need mixins for cosmetic gain). |
| 6 | Rootzone v5+ ABC | `dabb865` | **Smaller than projected.** Plan's −1,575 line target assumed all 5 modules duplicated; actual was 3 v5+ modules and netted −32 lines. The architectural seam (shared `_RootzoneReaderBase`) is the real win. |
| 7 | Component-writer scaffolding | `bbde3c2` | **Partial.** Shipped `open_iwfm_file` + `write_element_group` shared helpers (−111 lines across 6 writers). Deferred the speculative `BaseComponentWriter` strategy class. |
| 8 | Migration guide finalization | (this PR) | Pure docs polish. |

The "deferred" items in PRs 3, 5, and 7 follow the same pattern: an
audit during implementation found no concrete caller would benefit
from the proposed abstraction, and CLAUDE.md's "don't add abstractions
beyond what the task requires" pushed back. Each deferral is documented
in `docs/MIGRATION_v1_to_v2.md` with a recovery path if a real consumer
appears later.

The original five-item plan, kept below for historical context (the
descriptions in §3 still describe the originally-proposed scope, not
the final shipped scope — see the per-PR notes in
`docs/MIGRATION_v1_to_v2.md` for what actually landed):

1. **Consolidate the head/hydrograph cluster** (1,384 lines of
   near-duplicate readers/loaders/converters → ~500 lines of one
   source-agnostic abstraction)
2. **Split `core/model.py` constructors** (2,153-line file → per-loader
   modules in `core/loaders/`)
3. **Broaden the `BaseComponent` contract** (`to_dict` / `from_dict` /
   `clone` / `validate_against` as `@abstractmethod`)
4. **Move `webapi/slicing.py` and `webapi/properties.py` to `io/`**
   (they're full implementations, not webapi-specific shims)
5. **Split the 1,000+-line writers** (`gw_writer.py` 1,161,
   `stream_writer.py` 1,102, `cache_builder.py` 964, `runner/pest.py`
   1,456 → focused submodules)

The two added Phase-4.B items shipped as PRs 6 and 7.

---

## 1. Branch Naming & Setup

### 1.1 Branch name: `next`

Reasons for `next` (vs. `v2-dev`, `2.x`, `develop`, `next-major`):

- **Industry-standard for major-version development.** Used by Node.js,
  React, Vue, Vite, semantic-release, and the Anthropic SDKs. New
  contributors recognize the convention immediately.
- **Doesn't lock in the version number.** If scope expands and we cut
  v3 instead, the branch name still describes its purpose without
  being misleading.
- **Pairs cleanly with `master` as the trunk.** No risk of confusion
  with feature branches (which are named after their feature).
- **Short and easy to type.** `git checkout next` is one short word.

The branch is **long-running** — it does not get merged back into
`master` and rebased; it eventually replaces `master` at the v2.0.0
release.

### 1.2 Initial setup

```bash
# Cut the branch from the latest stable v1 trunk
git checkout master
git pull --ff-only
git checkout -b next
git push -u origin next

# Set the branch up so PRs can target it
gh api -X POST "repos/<owner>/pyiwfm/branches/next/protection" \
    --input branch-protection.json
```

Branch-protection rules for `next` should mirror `master`'s:
required reviewers, status checks (lint, mypy, tests, docs, integration
on schedule), no direct pushes.

### 1.3 Keeping `next` in sync with `master`

While `next` is in flight, all v1.x bug-fix and feature work continues
on `master`. We forward-port `master` into `next` weekly so `next`
doesn't drift unmaintainably:

```bash
# Weekly sync (or after each v1.x patch release)
git checkout next
git pull --ff-only
git merge --no-ff master -m "Merge master into next (sync for v1.x.y)"
# Resolve conflicts; the v2 refactors will conflict often with v1 work.
# Conflicts are expected; resolve in favor of the v2 architecture.
git push
```

When a v1.x bug fix is genuinely critical for v2 development too, cherry-pick it
in the **other** direction (master → next) immediately rather than waiting for
the weekly sync.

---

## 2. Versioning & Pre-Releases (PEP 440)

The project versions via [hatch-vcs](https://github.com/ofek/hatch-vcs)
from git tags. PEP 440 pre-release identifiers (matching how PyPI
sorts and pip resolves) are:

- `v2.0.0a1`, `v2.0.0a2`, ... — alphas, breaking changes still landing
- `v2.0.0b1`, `v2.0.0b2`, ... — betas, API frozen, integration testing
- `v2.0.0rc1`, `v2.0.0rc2`, ... — release candidates, only blocker fixes
- `v2.0.0` — final release, branch is merged to master and `next` is
  re-cut from the new master for v3 work (or deleted until needed)

`pip install pyiwfm` does **not** install pre-releases unless the user
passes `--pre` or pins explicitly. So tagging a `v2.0.0a1` is safe —
existing users on v1.x are unaffected.

---

## 3. PR Breakdown

Each of the five refactors is sized as **one reviewable PR** against
`next`. They are independent — the order below is recommended (low-risk
first) but not strictly required. Each PR ships **with deprecation
shims** that keep v1.x import paths working with a `DeprecationWarning`,
so users can run their v1 code under a v2 alpha and see warnings before
hard breakage at v2.0.0 final.

### PR 1: Consolidate the head/hydrograph cluster

**Files removed (after deprecation period):**
`io/head_loader.py` (477), `io/head_all_converter.py` (270),
`io/hydrograph_loader.py` (170), `io/hydrograph_reader.py` (250),
`io/hydrograph_converter.py` (217). Total: 1,384 lines.

**Files added:**
- `io/timeseries_io.py` — new module with three classes:
  - `TimeSeriesSource` (ABC) — interface for "stream of nodal
    time-series data" with subclasses for text, HDF5, binary
  - `TimeSeriesCache` (HDF5-backed, replaces both `head_all_converter`
    and `hydrograph_converter`)
  - `LazyTimeSeriesLoader` parameterized by source kind (replaces both
    `head_loader` and `hydrograph_loader`)
- `io/hydrograph_reader.py` — kept as the *text* reader only; loses its
  conversion responsibility

**Deprecation shims (kept for v2.0–v2.x, removed in v3.0):**
- `io/head_loader.py` re-exports `LazyHeadDataLoader` as a thin alias
  for `LazyTimeSeriesLoader(source="iwfm_head_hdf5")` with a
  `DeprecationWarning` on import
- Same for `head_all_converter`, `hydrograph_loader`, `hydrograph_converter`

**Test strategy:** Existing `test_io_head_loader.py`, `test_webapi_head_loader.py`,
`test_io_hydrograph_*.py` keep running unchanged via the shims. Add
`tests/unit/test_timeseries_io.py` covering the new abstraction directly.

**Estimated size:** ~600 lines of new code, ~200 lines of shims, deletes
1,384 lines. Net **−784 lines**.

**Risk:** Low. The lazy/converter pattern is well-established in the
existing modules; this is mostly extracting a common ABC.

---

### PR 2: Split `core/model.py` constructors

**Why:** `core/model.py` is 2,153 lines and contains six classmethods
(`from_preprocessor`, `from_preprocessor_binary`, `from_simulation`,
`from_simulation_with_preprocessor`, `from_hdf5`, `from_binary`) that
each implement a complete loader pipeline. The longest
(`from_simulation_with_preprocessor`) is 1,397 lines on its own. They
share little except the final `IWFMModel(...)` construction.

**Files added:**
- `core/loaders/__init__.py` (re-exports for stable paths)
- `core/loaders/from_preprocessor.py`
- `core/loaders/from_simulation.py`
- `core/loaders/from_simulation_with_preprocessor.py`
- `core/loaders/from_hdf5.py`

**Files changed:**
- `core/model.py` shrinks to ~400 lines: dataclass definition, mutation
  helpers (Phase 2.1), validation, `__repr__`, and 5-line
  classmethod dispatchers that delegate to `core/loaders/`

**No public-API break.** `IWFMModel.from_simulation(...)` etc. still
work. The new `core.loaders` namespace is also available for advanced
users who want direct access.

**Test strategy:** All existing `test_model_*.py` keep passing
unchanged. Move loader-specific tests to `tests/unit/loaders/`.

**Estimated size:** Mostly file moves with minor refactoring at
seams. **~0 net lines** (same code, reorganized).

**Risk:** Medium. Easy to introduce subtle behavior changes when
splitting tightly-coupled code. Mitigation: integration tests
(`test_roundtrip.py`, `test_roundtrip_unified.py`) catch any drift
against the IWFM Sample Model.

---

### PR 3: Broaden the `BaseComponent` contract

**Why:** `core/base_component.py` is already an ABC with `validate()`
and `n_items` (Phase-1 verification corrected the audit's claim that
it wasn't). What it's missing is the wider contract that enables
generic operations across components:

```python
class BaseComponent(ABC):
    @abstractmethod
    def to_dict(self) -> dict[str, Any]: ...
    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[str, Any]) -> Self: ...
    @abstractmethod
    def clone(self) -> Self: ...
    @abstractmethod
    def validate_against(self, grid: AppGrid) -> list[str]: ...
```

These unblock:
- Generic JSON / HDF5 serialization across all components (no per-class
  bespoke code)
- Scenario diff tools (`a.to_dict() == b.to_dict()`)
- Mutation API in `IWFMModel` operating on `BaseComponent`
  polymorphically

**Files changed:** `core/base_component.py` plus the six component
implementations (`components/groundwater.py`,
`components/stream.py`, `components/lake.py`, `components/rootzone.py`,
`components/small_watershed.py`, `components/unsaturated_zone.py`).

**Why this is breaking:** Any out-of-tree subclass of `BaseComponent`
must implement the new abstract methods. There's no clean shim
strategy for adding required methods to an ABC, so this is a clean
break at v2.0.0.

**Migration guide entry:** "If you subclass `BaseComponent` (rare —
this is intended for internal use), implement `to_dict`, `from_dict`,
`clone`, and `validate_against`. Default implementations for
`to_dict` / `from_dict` are provided as helpers in `core.serialization`."

**Test strategy:** Six component-roundtrip tests
(`tests/unit/test_<component>_serialization.py`) verifying that
`from_dict(to_dict(c)) == c` for each. Plus a meta-test that every
concrete subclass of `BaseComponent` actually implements the new
abstract methods.

**Estimated size:** ~600 new lines (helpers + per-component
implementations), no deletions.

**Risk:** Low-medium. The methods are additive; the risk is
implementation correctness for each component (especially nested
fields like time series, observation wells).

---

### PR 4: Move `webapi/slicing.py` and `webapi/properties.py` to `io/`

**Why:** Phase-1 verification found these are full implementations
(602 and 626 lines), not webapi-specific shims. They're imported by
`routes/slices.py` and `_mesh_state.py` respectively. They have no
web-only dependency — they're domain logic that happens to live in
the wrong directory.

**Files moved (with deprecation shims):**
- `src/pyiwfm/visualization/webapi/slicing.py` → `src/pyiwfm/io/slicing.py`
- `src/pyiwfm/visualization/webapi/properties.py` → `src/pyiwfm/io/properties.py`

**Shims at the old paths:**
```python
# src/pyiwfm/visualization/webapi/slicing.py (v2.0–v2.x only)
import warnings
warnings.warn(
    "pyiwfm.visualization.webapi.slicing has moved to pyiwfm.io.slicing; "
    "the old path will be removed in v3.0",
    DeprecationWarning,
    stacklevel=2,
)
from pyiwfm.io.slicing import *  # noqa: F401,F403
```

**Files changed:** Update internal imports in `routes/slices.py`
and `_mesh_state.py` to use the new paths. Update CLAUDE.md to
remove the (corrected) note that they're misplaced.

**Test strategy:** Move `tests/unit/test_webapi_properties.py` to
`tests/unit/test_io_properties.py`; same for slicing. Add a one-line
test asserting that importing from the old path triggers a
`DeprecationWarning`.

**Estimated size:** Pure reorganization. **0 net lines** plus ~30
lines of shims.

**Risk:** Very low.

---

### PR 5: Split the 1,000+-line writers

**Files restructured:**

| Current file | Lines | Split into |
|---|---|---|
| `io/gw_writer.py` | 1,161 | `io/gw_writer/__init__.py` (re-exports), `main.py`, `parameters.py`, `boundary_conditions.py`, `pumping.py`, `subsidence.py` |
| `io/stream_writer.py` | 1,102 | `io/stream_writer/__init__.py`, `main.py`, `nodes.py`, `reaches.py`, `diversions.py`, `bypasses.py` |
| `io/cache_builder.py` | 964 | `io/cache_builder/__init__.py`, `mesh.py`, `heads.py`, `budgets.py`, `metadata.py` |
| `runner/pest.py` | 1,456 | `runner/pest/__init__.py`, `manager.py`, `templates.py`, `instructions.py`, `params.py`, `observations.py`, `postprocessor.py` (most submodules already exist as siblings — this PR cleans the entry-point file) |

**Public API:** Unchanged. The `__init__.py` files re-export every
public name so `from pyiwfm.io.groundwater.writer import GroundwaterWriter`
still works.

**Test strategy:** Existing tests pass unchanged. No new tests needed
beyond verifying that the re-export shape matches.

**Estimated size:** Pure reorganization. **0 net lines.**

**Risk:** Very low. File moves only.

---

## 4. Deprecation & Migration Policy

> **Note (2026-04-27):** the policy below describes the original plan
> assumption that v2.0 would ship with deprecation shims. During
> implementation we landed on a different policy: **clean rename, no
> shims** — see [`docs/MIGRATION_v1_to_v2.md`](MIGRATION_v1_to_v2.md)
> § TL;DR for the rationale (small audience, every user reads the
> changelog, shims defer pain rather than eliminate it). The original
> shim policy is kept below for historical context.

The original contract proposal for v1 → v2:

1. **Every removed public name** (functions, classes, module paths)
   gets a deprecation shim in v2.0.0–v2.x that:
   - Emits `DeprecationWarning` once on first use (use
     `warnings.warn(..., stacklevel=2)`)
   - Forwards to the new implementation
2. **Every renamed module** gets a forwarding module that re-exports
   the new location, with a single `DeprecationWarning` on import
3. **Every changed signature** is either:
   - Made backward-compatible by adding optional kwargs (preferred), or
   - Wrapped in a shim function that translates old calls to new
4. **Hard breaks** (where no shim is feasible — e.g. the `BaseComponent`
   ABC contract) are documented in `CHANGELOG.md` under
   `## [2.0.0]` → `### Removed` with a one-paragraph migration note.

Deprecation shims persist through the entire v2.x line and are
removed in v3.0.

`docs/MIGRATION_v1_to_v2.md` (added in PR 1, expanded in each PR)
collects the migration notes in one place. Each entry has the form:

```markdown
### `pyiwfm.io.head_loader.LazyHeadDataLoader`

**v1.x:**
```python
from pyiwfm.io.head_loader import LazyHeadDataLoader
loader = LazyHeadDataLoader("heads.hdf", n_layers=4)
```

**v2.x (hard rename — no shim):**
```python
from pyiwfm.io.timeseries.lazy import LazyNodalLoader
loader = LazyNodalLoader("heads.hdf", n_layers=4)
```

PR 1 documented why the original "one source-agnostic loader" idea was
dropped in favor of two thin shape-specific classes (`LazyNodalLoader`
for `(t, n, layer)`, `LazyTabularLoader` for `(t, columns)`).
```

---

## 5. Release Cadence

The original cadence anticipated one alpha per PR. In practice all
seven implementation PRs (1–7) plus the migration guide finalization
(PR 8) landed back-to-back on `next` between 2026-04-27 and the
v2.0.0a1 tag, so we collapse to one alpha for the bundle:

| Tag | Trigger | What's in it |
|---|---|---|
| `v2.0.0a1` | All 8 PRs merged, migration guide finalized | All v2.0 refactors (PRs 1–7) + migration guide |
| `v2.0.0a2`+ | Only if a follow-up audit lands a behavior fix | Iterate based on alpha user feedback |
| `v2.0.0b1` | API frozen, no `[v2]` issues open >1 week | First beta |
| `v2.0.0b2`+ | Bug fixes only | Iterate |
| `v2.0.0rc1` | After ≥30 days of beta with no API changes | Release candidate |
| `v2.0.0rc2`+ | Only for blocker fixes | |
| `v2.0.0` | After ≥14 days of rc with no blockers | Final |

After v2.0.0:
- v1.x line continues for ≥6 months as `master-1.x` (rename current
  `master` to `master-1.x`, then rename `next` to `master`)
- Critical security / data-corruption fixes get backported to v1.x
- New features go to v2.x only

---

## 6. CI on `next`

The CI workflow (`.github/workflows/ci.yml`) needs three small changes
when `next` is created:

1. **Trigger on `next` push/PR:**
   ```yaml
   on:
     push:
       branches: [master, next]
     pull_request:
       branches: [master, next]
   ```
2. **Publish workflow** (`.github/workflows/publish.yml`) must accept
   PEP 440 pre-release tags (`v2.0.0a1`, `b1`, `rc1`). The current
   regex matches `v*` so this works as-is, but verify before tagging.
3. **Codecov uploads** from `next` should go to a separate "next"
   flag so coverage on the v2 branch doesn't pollute v1's history:
   ```yaml
   - uses: codecov/codecov-action@v6
     with:
       files: coverage.xml
       flags: ${{ github.ref == 'refs/heads/next' && 'next' || 'master' }}
   ```

---

## 7. CHANGELOG strategy

The `[Unreleased]` section in `docs/changelog.rst` continues to track
master (v1.x) work. Add a parallel section for v2 alphas at the top:

```rst
[2.0.0a1] - YYYY-MM-DD
----------------------

**Major version with breaking changes.** See ``docs/MIGRATION_v1_to_v2.md``
for the upgrade guide. Pre-release: install with ``pip install --pre pyiwfm``.

Removed (with deprecation shims through v2.x; hard removal in v3.0)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``pyiwfm.io.head_loader.LazyHeadDataLoader`` — replaced by
  ``pyiwfm.io.timeseries.lazy.LazyTimeSeriesLoader``.
- ``pyiwfm.io.head_all_converter`` (entire module) — replaced by
  ``pyiwfm.io.timeseries.lazy.TimeSeriesCache``.
- ...

Changed
~~~~~~~

- (file-only reorganizations land here)

Added
~~~~~

- ``pyiwfm.io.timeseries.lazy`` — unified abstraction over all IWFM
  time-series outputs.
```

---

## 8. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Long-running `next` branch drifts from `master` | High | Medium | Weekly merge-from-master cadence; conflicts are expected and acceptable |
| Deprecation shims accidentally do nothing (silent breakage) | Medium | High | Each PR includes one test asserting the old path emits `DeprecationWarning` AND the forwarded behavior matches |
| Users install `v2.0.0a1` accidentally and get burned | Low | Medium | PEP 440 + pip's default `--no-pre` behavior protects this; mention pre-release status prominently in PyPI description |
| `BaseComponent` ABC changes break unknown out-of-tree subclasses | Medium | Medium | Document in `MIGRATION_v1_to_v2.md`; can't shim around required abstract methods |
| Integration tests don't cover a refactored code path | Low | High | Run the IWFM Sample Model + (when available) C2VSimCG / C2VSimFG end-to-end against each alpha |
| v1.x bug fix is needed but conflicts hard with v2 | Low | Medium | Cherry-pick fix to both branches; the PR template includes a checkbox for "needs cherry-pick to next" |

---

## 9. Out of scope for v2.0

Explicitly **NOT** in v2.0 (defer to v2.x or v3.0):

- New web write/edit endpoints (the viewer stays read-only by design)
- Editing UI in the React frontend
- Replacing pyproject extras structure
- Migrating off Zustand or vtk.js
- Adding a typed schema layer (Pydantic / attrs / msgspec) over the
  existing dataclasses
- Switching from `hatch` to a different build backend

These are valuable but each is its own scoping conversation. Putting
them in v2.0 would balloon the scope and delay every refactor that's
already designed.
