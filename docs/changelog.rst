Changelog
=========

All notable changes to pyiwfm will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

[Unreleased]
------------

Added
~~~~~

- **Numba JIT kernel for the calibration FE-interpolation hot path**.
  New ``pyiwfm.calibration._kernels`` module hosts both a pure-numpy
  fallback (always available) and a ``@numba.njit(cache=True)`` JIT
  kernel that activates when ``numba`` is installed. ``ResultsExtractor``
  and ``HeadAllExtractor`` route through the kernel automatically;
  no caller code change is needed. Synthetic benchmark on a
  5,000-location workload: pure-numpy 32 ms/frame → Numba 78 μs/frame
  (~400× speedup). Closes most of the gap to the Fortran
  ``ResultsExtract.exe`` reference implementation for the heaviest
  pyiwfm-native workloads. Opt-in via ``pip install pyiwfm[fast-calib]``
  (the ~50MB ``numba`` install is not in the default install path so
  users who never run results_extraction don't pay the cost).
  Numerical equivalence between numpy and Numba paths is verified by
  ``tests/unit/test_calibration_kernels.py``. See
  ``docs/user_guide/calibration.rst`` § Performance for the
  three-engine story (pure-numpy / Numba / FortranBackend).

Changed
~~~~~~~

- **Vectorised the calibration FE-interpolation hot paths** in
  ``calibration/results_extraction.py``,
  ``calibration/headall_extraction.py``, and
  ``calibration/iwfm2obs.py``. The triple-nested
  ``for ti in timesteps: for spec: for layer: np.dot(coeffs, vals)``
  loop was replaced with one matrix-vector product per
  (location, timestep) that computes all layers at once
  (``coeffs @ frame[node_indices, :]``). The per-timestep
  ``_aggregate_layers`` T-weighted/sum loops were replaced with
  broadcasted numpy expressions. The ``iwfm2obs_from_model``
  composite-head computation was reduced from a triple-nested
  Python loop with per-(timestep, layer) f-string + dict lookup to
  one batched matrix-vector product per well. The
  ``compute_multilayer_weights`` per-layer dict-build for FE
  interpolation of layer top/bottom elevations and hydraulic
  conductivity was replaced with array slicing + matrix-vector
  products. Numerical results are bit-identical to the loop
  versions; benchmarks show ~3× per frame on FE interpolation,
  ~80× on aggregation, ~5× on composite-head, ~2-3× on
  multilayer-weights. See ``docs/user_guide/calibration.rst``
  Performance section.

[2.0.0a1] - 2026-05-01
----------------------

First v2.0 alpha release. Cuts the line between the v1.x series (which
remains supported on ``master``) and the v2.0 series on ``next``. Every
entry below was previously labelled ``[Unreleased]``; reclassified to
``[2.0.0a1]`` at tag-cut time. See ``docs/MIGRATION_v1_to_v2.md`` for
the full guide to upgrading.

Added
~~~~~

- ``IWFMModel.load_errors`` (returns ``list[ComponentLoadError]``) and
  ``IWFMModel.has_load_errors`` properties for introspecting partial
  loads. Populated by the ``from_*`` loaders when ``strict=False`` (the
  default) or when ``strict="collect"`` recorded errors before its
  finalize raise. Backward-compatible: the existing scalar
  ``metadata['{component}_load_error']`` keys still get set.

- ``strict="collect"`` mode for ``IWFMModel.from_preprocessor`` and
  ``from_simulation_with_preprocessor`` (and their underlying
  ``load_from_*`` functions). Loads every component, then raises a
  single ``ValidationError`` at the end enumerating every component
  that failed — best UX for one-shot CLI / Web API surfaces. Existing
  ``strict=False`` (lenient) and ``strict=True`` (fail-fast) modes are
  unchanged.

- ``pyiwfm.cli._model_loader.warn_if_partial_load(model)`` helper:
  prints a one-line stderr banner when a CLI load succeeded with
  ``--allow-partial-load`` but some components still failed.

- ``pyiwfm.io.preprocessor.mesh.write_nodes`` / ``write_elements`` / ``write_stratigraphy``
  gained keyword-only parameters for the canonical preprocessor format:
  ``factor=`` (FACTXY for nodes, FACTEL for stratigraphy) and, for
  ``write_elements``, ``subregion_names=`` / ``n_subregions=`` (the latter
  is inferred from element ``.subregion`` values when omitted). All three
  emit the IWFM-conventional ASCII-art title block by default and use the
  ``AQT_L*/AQF_L*`` column legend.

Changed
~~~~~~~

- **``pyiwfm.io.groundwater.reader``** split: ``GWMainFileReader`` /
  ``GWMainFileConfig`` moved to a new ``groundwater/main_reader.py``
  (~1075 LOC). The package ``__init__.py`` re-exports both symbols so
  callers that import from ``pyiwfm.io.groundwater`` are unaffected.
  Two intra-package importers (``main_writer.py``, ``writer.py``)
  updated to point at the new submodule for ``GWMainFileConfig``.
  ``groundwater/reader.py`` drops from 1876 to ~837 LOC. Internal-only
  follow-up to the io restructure (no migration-guide entry needed).

- **``pyiwfm.io.streams.reader``** split: ``StreamMainFileReader`` /
  ``StreamMainFileConfig`` moved to a new ``streams/main_reader.py``
  (~590 LOC), and ``StreamSpecReader`` / ``StreamReachSpec`` moved to
  a new ``streams/spec.py`` (~325 LOC). The package ``__init__.py``
  re-exports every symbol unchanged, so callers that import from
  ``pyiwfm.io.streams`` are unaffected. ``streams/reader.py`` drops
  from 1503 to ~666 LOC. Internal-only follow-up to the io
  restructure (no migration-guide entry needed).

- **``pyiwfm.io.groundwater``** is now a package (was a module),
  the largest cluster in the io restructure. Eleven flat
  modules (``groundwater.py``, ``gw_writer.py``, ``gw_main_writer.py``,
  ``gw_boundary[_writer].py``, ``gw_pumping[_writer].py``,
  ``gw_subsidence[_writer].py``, ``gw_tiledrain[_writer].py``)
  collapse into one subpackage. The naming inconsistency between the
  long-form ``groundwater.py`` reader and the short-form ``gw_*.py``
  siblings dissolves: the directory carries the full name, and
  submodules inside drop the redundant ``gw_`` prefix
  (e.g. ``gw_boundary.py`` → ``groundwater/boundary.py``,
  ``gw_writer.py`` → ``groundwater/writer.py``). The package
  ``__init__.py`` re-exports every public symbol; reader imports
  (``from pyiwfm.io.groundwater import GroundwaterReader``) keep
  working unchanged. The ten v1.x sibling paths are **gone** in
  v2.0 — update imports and ``mock.patch`` strings. This is the
  thirteenth and final PR of the io restructure plan; the eight
  format-primitive and domain packages
  (``ascii``, ``binary``, ``hdf5``, plus ``unsaturated_zone``,
  ``small_watershed``, ``lakes``, ``simulation``, ``preprocessor``,
  ``rootzone``, ``budget``, ``timeseries``, ``streams``,
  ``groundwater``) replace 84 flat modules with 13 cohesive
  subpackages plus the cross-cutting flat helpers. See
  ``docs/MIGRATION_v1_to_v2.md`` § 10.

- **``pyiwfm.io.streams``** is now a package (was a module),
  collapsing the nine flat stream modules (``streams.py``,
  ``stream_writer.py``, ``stream_bypass.py`` /
  ``stream_bypass_writer.py``, ``stream_diversion.py`` /
  ``stream_diversion_writer.py``, ``stream_inflow.py`` /
  ``stream_inflow_writer.py``, ``stream_depletion.py``) into one
  subpackage. The plural/singular asymmetry between ``streams.py`` and
  the ``stream_*.py`` siblings dissolves: ``pyiwfm/io/streams/`` carries
  the plural, submodules inside drop the redundant ``stream_`` prefix
  (e.g. ``stream_bypass.py`` → ``streams/bypass.py``). The package
  ``__init__.py`` re-exports every public symbol; reader imports keep
  working unchanged. The eight v1.x sibling paths are **gone** in
  v2.0 — update imports and ``mock.patch`` strings. See
  ``docs/MIGRATION_v1_to_v2.md`` § 10.

- **``pyiwfm.io.timeseries``** is now a package (was a module),
  collapsing five flat modules (``timeseries.py``,
  ``timeseries_ascii.py``, ``timeseries_writer.py``,
  ``timeseries_io.py``, ``timeseries_reader.py``) into one subpackage:
  ``pyiwfm/io/timeseries/{reader,ascii,writer,lazy,compat}.py``. The
  package ``__init__.py`` re-exports every public symbol. The four
  v1.x sibling paths (``pyiwfm.io.timeseries_ascii``,
  ``pyiwfm.io.timeseries_writer``, ``pyiwfm.io.timeseries_io``,
  ``pyiwfm.io.timeseries_reader``) are **gone** in v2.0 — update
  imports and ``mock.patch`` strings (e.g. the ``h5py.File`` patch
  target moved from ``pyiwfm.io.timeseries.h5py.File`` to
  ``pyiwfm.io.timeseries.reader.h5py.File``). See
  ``docs/MIGRATION_v1_to_v2.md`` § 10.

- **``pyiwfm.io.budget``** is now a package (was a module),
  collapsing the nine flat budget / zone-budget modules (``budget.py``,
  ``zbudget.py``, ``budget_checks.py``, ``budget_control.py``,
  ``zbudget_control.py``, ``budget_excel.py``, ``zbudget_excel.py``,
  ``budget_pest.py``, ``budget_utils.py``) into one subpackage with
  consistent naming (e.g. ``zbudget.py`` → ``budget/zone_reader.py``).
  The package ``__init__.py`` re-exports every public symbol —
  including the leaky ``parse_iwfm_datetime`` / ``iwfm_date_to_iso``
  re-exports that callers had been importing from the flat
  ``pyiwfm.io.budget``. The eight v1.x sibling paths
  (``pyiwfm.io.zbudget``, ``pyiwfm.io.budget_checks``, etc.) are
  **gone** in v2.0 — update imports and ``mock.patch`` strings. See
  ``docs/MIGRATION_v1_to_v2.md`` § 10.

- **``pyiwfm.io.rootzone``** is now a package (was a module),
  collapsing the nine flat root-zone modules (``rootzone.py``,
  ``rootzone_writer.py``, ``_rootzone_base.py``, ``rootzone_area.py``,
  ``rootzone_native.py``, ``rootzone_nonponded.py``,
  ``rootzone_ponded.py``, ``rootzone_urban.py``, ``rootzone_v4x.py``)
  into one subpackage:
  ``pyiwfm/io/rootzone/{reader,writer,_base,area,native,nonponded,
  ponded,urban,v4x}.py``. The package ``__init__.py`` re-exports
  every public symbol; reader imports
  (``from pyiwfm.io.rootzone import RootZoneReader``) keep working
  unchanged. The eight v1.x flat sibling paths
  (``pyiwfm.io.rootzone_writer``, ``pyiwfm.io.rootzone_native``, …)
  are **gone** in v2.0 — update imports and ``mock.patch`` strings.
  See ``docs/MIGRATION_v1_to_v2.md`` § 10.

- **``pyiwfm.io.preprocessor``** is now a package (was a module),
  collapsing ``preprocessor.py`` (main-file reader),
  ``preprocessor_writer.py`` (Jinja2 writer), and ``mesh.py`` (the
  canonical nodes/elements/stratigraphy reader-writer pair, renamed
  from ``ascii.py`` earlier in v2.0) into one subpackage:
  ``pyiwfm/io/preprocessor/{reader,writer,mesh}.py``. The package
  ``__init__.py`` re-exports all three submodules' public API. The
  preprocessor *binary* format remains a separate module under
  ``pyiwfm.io.binary.preprocessor`` (PR β). The
  ``pyiwfm.io.preprocessor_writer`` and ``pyiwfm.io.mesh`` paths are
  **gone** in v2.0 — update imports and ``mock.patch`` strings. See
  ``docs/MIGRATION_v1_to_v2.md`` § 10.

- **``pyiwfm.io.simulation``** is now a package (was a module),
  collapsing ``simulation.py`` (reader), ``simulation_writer.py``
  (Jinja2 writer), and ``simulation_messages.py`` (``Message.out``
  parser) into one subpackage:
  ``pyiwfm/io/simulation/{reader,writer,messages}.py``. The package
  ``__init__.py`` re-exports all three submodules' public API; reader
  imports (``from pyiwfm.io.simulation import IWFMSimulationReader``)
  keep working unchanged. Writer imports moved from
  ``pyiwfm.io.simulation_writer`` and message-log imports from
  ``pyiwfm.io.simulation_messages`` to ``pyiwfm.io.simulation`` (or
  the deeper ``pyiwfm.io.simulation.writer`` /
  ``pyiwfm.io.simulation.messages``); the v1.x flat paths are
  **gone** in v2.0. See ``docs/MIGRATION_v1_to_v2.md`` § 10.

- **``pyiwfm.io.lakes``** is now a package (was a module). The
  plural/singular mismatch between ``lakes.py`` (reader) and
  ``lake_writer.py`` (writer) dissolves: both are now submodules of
  ``pyiwfm/io/lakes/`` (``reader.py`` and ``writer.py``). Reader
  imports (``from pyiwfm.io.lakes import LakeReader``) keep working
  unchanged because the package re-exports those names. Writer imports
  moved from ``pyiwfm.io.lake_writer`` to ``pyiwfm.io.lakes`` (or the
  deeper ``pyiwfm.io.lakes.writer``); the v1.x flat path is **gone**
  in v2.0. See ``docs/MIGRATION_v1_to_v2.md`` § 10.

- **``pyiwfm.io.small_watershed``** is now a package (was a module),
  collapsing ``pyiwfm/io/small_watershed.py`` and
  ``pyiwfm/io/small_watershed_writer.py`` into one subpackage:
  ``pyiwfm/io/small_watershed/reader.py`` and
  ``pyiwfm/io/small_watershed/writer.py``. The package ``__init__.py``
  re-exports both submodules' public API; reader imports
  (``from pyiwfm.io.small_watershed import SmallWatershedMainReader``)
  keep working unchanged. The ``from pyiwfm.io.small_watershed_writer
  import …`` path is **gone** in v2.0 — update imports and
  ``mock.patch`` strings. See ``docs/MIGRATION_v1_to_v2.md`` § 10.

- **``pyiwfm.io.unsaturated_zone``** is now a package (was a module),
  collapsing ``pyiwfm/io/unsaturated_zone.py`` and
  ``pyiwfm/io/unsaturated_zone_writer.py`` into one subpackage:
  ``pyiwfm/io/unsaturated_zone/reader.py`` (was ``unsaturated_zone.py``)
  and ``pyiwfm/io/unsaturated_zone/writer.py`` (was
  ``unsaturated_zone_writer.py``). The package ``__init__.py``
  re-exports both submodules' public API so every reader and writer
  symbol is now importable from ``pyiwfm.io.unsaturated_zone``. The
  ``from pyiwfm.io.unsaturated_zone_writer import …`` path is **gone**
  in v2.0 — update imports and ``mock.patch`` strings. First domain
  package in the io restructure plan; ``small_watershed/``, ``lakes/``,
  etc. follow. See ``docs/MIGRATION_v1_to_v2.md`` § 10.

- **``pyiwfm.io.ascii``** is now a format-primitive package collecting
  every IWFM ASCII helper. ``pyiwfm/io/iwfm_reader.py`` →
  ``pyiwfm/io/ascii/reader.py``, ``pyiwfm/io/iwfm_writer.py`` →
  ``pyiwfm/io/ascii/writer.py``, and the comment cluster
  (``comment_extractor.py``, ``comment_metadata.py``,
  ``comment_writer.py``) moved into ``pyiwfm/io/ascii/`` as well. The
  package ``__init__.py`` re-exports the full public API surface, so
  ``from pyiwfm.io.ascii import parse_int, write_comment,
  CommentExtractor`` works regardless of which submodule each symbol
  lives in. The five v1.x paths
  (``pyiwfm.io.iwfm_reader``/``iwfm_writer``/``comment_extractor``/
  ``comment_metadata``/``comment_writer``) are **gone** in v2.0 — update
  imports and ``mock.patch`` strings. See
  ``docs/MIGRATION_v1_to_v2.md`` § 10.

- **``pyiwfm.io.binary``** is now a package (was a module). The file
  ``pyiwfm/io/binary.py`` moved to ``pyiwfm/io/binary/fortran.py``
  (Fortran unformatted-sequential primitives), and
  ``pyiwfm/io/preprocessor_binary.py`` moved to
  ``pyiwfm/io/binary/preprocessor.py`` (IWFM preprocessor binary
  format). The package ``__init__.py`` re-exports both submodules'
  public API so ``from pyiwfm.io.binary import X`` works for either.
  The ``from pyiwfm.io.preprocessor_binary import …`` path is **gone**
  in v2.0 — update imports and ``mock.patch`` strings. See
  ``docs/MIGRATION_v1_to_v2.md`` § 10.

- **``pyiwfm.io.hdf5``** is now a package (was a module): the file moved
  to ``pyiwfm/io/hdf5/model.py`` with the package ``__init__.py``
  re-exporting the public API. Top-level callers
  (``from pyiwfm.io import HDF5ModelReader``) and direct-submodule
  callers (``from pyiwfm.io.hdf5 import HDF5ModelReader``) are
  unaffected. Optional new direct path:
  ``from pyiwfm.io.hdf5.model import HDF5ModelReader``. First step of
  a broader io restructure into format-primitive
  (``ascii/``/``binary/``/``hdf5/``/``dss/``) and domain
  (``groundwater/``/``streams/``/…) subpackages — see
  ``docs/MIGRATION_v1_to_v2.md`` § 10.

- **All text-mode file I/O across the codebase now uses explicit
  ``encoding="utf-8"``** (183 sites). Previously, ``open(path)``,
  ``Path.read_text()``, and ``Path.write_text()`` calls inherited the
  platform-default encoding (``cp1252`` on Windows, ``utf-8`` elsewhere),
  so an IWFM file with non-ASCII characters (e.g. Spanish-language
  subregion names, °, smart quotes) would hit silent decoder errors on
  Windows that surfaced as confusing format-error tracebacks. UTF-8 is
  a strict superset of ASCII, so existing fixtures are unaffected.
  The Pylint ``PLW1514`` rule (preview) is now enabled in ruff to
  prevent regression.

- **CLI** subcommands ``viewer`` and ``export`` now load IWFM models
  with ``strict="collect"`` by default. A model with broken or missing
  component files now produces a single ``ValidationError`` listing
  every failed component and exits with code 1, instead of silently
  starting up with a partially-loaded model. Pass ``--allow-partial-load``
  to opt back into the historical permissive behaviour; a stderr banner
  is printed on startup so you're aware you're running degraded.

- **Renamed** ``pyiwfm.io.ascii`` → ``pyiwfm.io.preprocessor.mesh``. The module
  contained only the six mesh + stratigraphy preprocessor-subfile
  functions (``read_nodes``, ``read_elements``, ``read_stratigraphy``,
  and the ``write_*`` counterparts), not generic ASCII utilities; the
  generic-reader role is filled by ``pyiwfm.io.ascii.reader``. All
  re-exports through ``pyiwfm.io`` keep the same names, so callers that
  use ``from pyiwfm.io import read_nodes`` are unaffected. Direct
  submodule imports must update ``from pyiwfm.io.ascii import ...`` →
  ``from pyiwfm.io.preprocessor.mesh import ...``.

- **Unified the IWFM ASCII writer for nodes / elements / stratigraphy**
  on a single canonical implementation in ``pyiwfm.io.preprocessor.mesh``. The three
  parallel writer surfaces that previously coexisted (``mesh.write_*``,
  ``PreProcessorWriter.write_nodes/_elements/_stratigraphy``,
  ``preprocessor_writer.write_*_file``) are now collapsed:
  ``PreProcessorWriter`` mesh methods are thin orchestration delegates,
  and the standalone array-shape functions are removed (see *Removed*
  below). The canonical format always emits FACTXY/FACTEL factor lines,
  uses the ``AQT_L*/AQF_L*`` column legend, and renders the IWFM
  ASCII-art title block; ``header=`` continues to override.

Removed
~~~~~~~

- ``pyiwfm.utils`` package — was empty (`__all__ = []`) and had no
  candidate helpers to populate it after a codebase audit. The
  empty package was creating discoverability confusion.

- ``pyiwfm.io.preprocessor.writer.write_nodes_file`` /
  ``write_elements_file`` / ``write_stratigraphy_file`` (and their
  ``pyiwfm.io`` re-exports) — replaced by the unified
  ``pyiwfm.io.preprocessor.mesh.write_nodes`` / ``write_elements`` /
  ``write_stratigraphy`` taking the canonical ``dict[int, Node]`` /
  ``dict[int, Element]`` / ``Stratigraphy`` domain types. Callers
  holding raw NumPy arrays should construct the domain objects (see
  the migration guide).

Fixed
~~~~~

[1.3.0] - 2026-04-26
--------------------

Quick-win patch series surfaced by a fresh post-v1.2.0 audit. All
non-breaking; bigger refactors (Phase 3 + 4.B) bundle into v2.0 on the
``next`` branch.

Changed
~~~~~~~

- **Vectorize webapi route hot paths.** Three ``for i in range(...)``
  loops over NumPy arrays were pure overhead in the webapi serving path:

  - ``GET /api/results/head-diff`` (``routes/results.py:110``) now
    rounds and masks dry cells in one ``np.round`` + list comprehension
    over Python primitives instead of indexing element-by-element.
  - ``GET /api/budgets/{type}/summary`` (``routes/budgets.py:486``)
    computes per-column totals and means via ``axis=0`` reductions
    instead of looping with ``np.nansum`` / ``np.nanmean`` per column.
  - ``GET /api/export/budget-csv`` and ``GET /api/export/timeseries/budget``
    pre-round the entire matrix once with ``np.round`` and zip rows
    instead of rounding each cell inside two nested Python loops.

  Added a byte-for-byte equivalence test (``test_diff_values_match_legacy_per_index_formula``)
  asserting the new ``head-diff`` rounds identically to the legacy
  ``round(float(diff[i]), 3)`` formula on banker's-rounding ties and
  dry-cell edge cases.

- **Replace ``"No model loaded"`` boilerplate with ``ModelState`` methods.**
  Add ``ModelState.require_loaded()`` and ``ModelState.require_model()``
  methods that pair with the existing module-level ``require_model()``
  helper but resolve through the *caller's* ``model_state`` binding (so
  ``patch("...routes.foo.model_state", state)`` tests substitute their
  fixture transparently). Replace 23 instances of the two-line
  ``if not model_state.is_loaded: raise HTTPException(...)`` pattern
  across results / budgets / zbudgets / export / mesh / model /
  observations / properties routes; net –74/+59 lines.

- **Consolidate ``_make_config_v40`` and ``_make_config_v50`` test
  fixtures** in ``test_io_gw_subsidence_writer.py`` into a single
  version-parameterized factory. The two original helpers shared 22 of
  25 default fields verbatim.

- **Extract ``cli/_parsers.py::add_control_file_subcommand``** for the
  shared ``budget`` / ``zbudget`` parser-registration pattern.

Notes
~~~~~

- ``@cached_property`` audit found no conversions are warranted in
  v1.3.x: the expensive properties (``AppGrid.x``, ``y``, ``vertex``,
  ``element_centroids``) are already hand-cached with explicit
  ``_invalidate_cache()`` hooks that the test suite asserts directly.
  ``n_*`` count properties cannot safely be cached because the underlying
  collections are mutated by the v1.2.0 mutation helpers. The remaining
  ``@property`` declarations are trivial accessors where caching adds
  bookkeeping overhead exceeding the saved work.

- Cross-test-file fixture consolidation (broader than the within-file
  v40/v50 case above) was rejected after audit: each ``_make_config``
  helper across ``test_io_*_writer.py`` builds a different writer-config
  dataclass with unique defaults, so moving them to a shared
  ``_writer_fixtures.py`` would relocate lines rather than delete them.

[1.2.0] - 2026-04-25
--------------------

Major minor release from a comprehensive code review (see
``docs/V2_ROADMAP.md`` for what's deferred to v2.0). Non-breaking;
existing v1.1.x code keeps working without modification.

Added
~~~~~

**Python API for editing loaded models** (``pyiwfm.core.model.IWFMModel``)

- ``IWFMModel.set_aquifer_parameter(param, layer, values)`` — replace
  an aquifer-parameter array (``"kh"``, ``"kv"``, ``"ss"``, ``"sy"``,
  ``"aquitard_kv"``) for a single layer.
- ``IWFMModel.set_aquifer_parameter_at(param, node_id, layer, value)`` —
  set a single ``(node, layer)`` cell.
- ``IWFMModel.set_stratigraphy_from_thicknesses(gs_elev,
  aquitard_thicknesses, aquifer_thicknesses, active_node=None)`` —
  rebuild stratigraphy from thickness arrays with mesh-consistency check.
- ``IWFMModel.add_observation_well(node_id, layer, x, y, name="")``
  and ``remove_observation_well(name)`` — manage groundwater hydrograph
  output locations.
- ``IWFMModel.mark_dirty(component_name)`` plus internal ``_dirty: set[str]``
  field — track which components have been mutated since the last load/save.
- New user guide page ``docs/user_guide/mutating_models.rst`` walks
  through the calibration workflow.

**Stream depletion analysis suite** (``pyiwfm.io.streams.depletion``,
``pyiwfm.visualization.{plot,map}_depletion``, ``pyiwfm depletion`` CLI)

- ``BudgetOutputMissingError`` raised when a model required for
  comparative analysis didn't declare or didn't produce a stream
  reach/node budget; message tells the operator which IWFM input line
  to fix (``STRMRCHBUDFL`` / ``STNDBUDFL``).
- ``compute_stream_depletion_from_models(baseline_model,
  scenario_model, *, reach_ids=None, sa_column=...)`` — model-driven
  reach-level depletion; resolves budget paths from
  ``model.metadata['stream_budget_file']``.
- ``compute_stream_node_depletion(baseline_model, scenario_model, *,
  node_ids=None, sa_column=...)`` and
  ``StreamNodeDepletionResult`` / ``StreamNodeDepletionReport`` —
  per-stream-node analysis using the stream node budget HDF.
- ``DEFAULT_SA_COLUMN`` constant (``"Stream-Aquifer Interaction Within Model"``,
  the IWFM v5+ canonical column name); override via ``sa_column=...``
  for older models that emit ``"Gain from GW (+)"``.
- Tabular writers: ``write_stream_depletion_csv`` /
  ``write_stream_depletion_json`` / ``write_stream_depletion_excel``,
  plus ``StreamDepletionReport.write(path, format=None)`` dispatcher.
- Time-series plots in ``pyiwfm.visualization.plot_depletion``:
  ``plot_cumulative_depletion`` (with optional ``pumping_timeseries``
  overlay), ``plot_depletion_timeseries``, ``plot_depletion_summary_bar``.
- Spatial maps in ``pyiwfm.visualization.map_depletion``:
  ``plot_depletion_map`` (reach-level color polylines),
  ``plot_stream_node_depletion_map`` (per-node scatter),
  ``plot_depletion_along_reach`` (longitudinal profile),
  ``export_depletion_geojson`` and
  ``export_stream_node_depletion_geojson``.
- ``pyiwfm depletion <baseline> <scenario>`` CLI subcommand wires the
  above together: ``--output``, ``--plot``, ``--map``, ``--node-level``,
  ``--metric``, ``--reach-ids`` / ``--node-ids``, ``--sa-column``,
  ``--crs``.

**Drawdown analysis suite** (``pyiwfm.io.drawdown``,
``pyiwfm.visualization.{plot,map}_drawdown``, ``pyiwfm drawdown`` CLI)

- ``DrawdownAtLocation`` / ``DrawdownTimeSeriesReport`` — drawdown vs
  time at a chosen set of ``(node, layer)`` locations.
- ``DrawdownSnapshot`` — per-node drawdown at a single timestep
  (``kind="single"``) or per-node max across all timesteps
  (``kind="max"``).
- ``DrawdownComputer.build_timeseries_report(locations,
  reference_timestep)``, ``build_snapshot(timestep, layer,
  reference_timestep)``, ``build_max_snapshot(layer, reference_timestep)``.
- Tabular writers: ``write_drawdown_timeseries_csv`` / ``_json`` /
  ``_excel`` plus ``DrawdownTimeSeriesReport.write()`` dispatcher.
- Plots in ``pyiwfm.visualization.plot_drawdown``:
  ``plot_drawdown_timeseries``, ``plot_drawdown_summary_bar``.
- Maps in ``pyiwfm.visualization.map_drawdown``: ``plot_drawdown_map``
  (cone-of-depression scatter), ``export_drawdown_geojson``.
- ``pyiwfm drawdown <model_dir>`` CLI subcommand with
  ``--mode {timeseries,snapshot,max}``, ``--locations 1,1;42,2``,
  ``--reference-timestep``, ``--output``, ``--plot``, ``--no-map``,
  ``--crs``, ``--heads-hdf``.

**Documentation: editable inputs vs. computed outputs**

- New user guide page ``docs/user_guide/inputs_vs_outputs.rst``
  enumerates every model component (with its writer module and the
  Phase 2.1 mutation helper) versus every read-only analysis module
  (heads, hydrographs, subsidence, budgets, etc.).
- "Read-only by design" docstring annotations on ``head_loader``,
  ``hydrograph_loader``, ``hydrograph_reader``, ``area_loader``,
  ``budget``, ``zbudget``, ``simulation_messages``, ``drawdown``,
  ``stream_depletion``.

**Structured exception handling for model loading** (``pyiwfm.core.exceptions``,
``pyiwfm.core.model``)

- New ``ComponentLoadError(component_name, source_file, cause)`` —
  raised by ``IWFMModel.from_preprocessor`` and
  ``from_simulation_with_preprocessor`` when called with the new
  ``strict=True`` kwarg and a component fails to parse.
- ``IWFMModel.from_preprocessor(..., *, strict=False)`` and
  ``IWFMModel.from_simulation_with_preprocessor(..., *, strict=False)``
  — pass ``strict=True`` from calibration / analysis pipelines that
  require a complete model. The default lenient mode (``strict=False``)
  preserves the historical behavior: log a structured warning, record
  the error in ``model.metadata``, and continue with whatever components
  loaded.
- ``_COMPONENT_LOAD_EXCEPTIONS`` tuple replaces the previous
  ``except Exception`` catch-all at every component-load boundary in
  ``core/model.py`` (47 sites: 2 in ``from_preprocessor``, 35 in
  ``from_simulation_with_preprocessor``, plus the inner format-
  detection fallbacks). Programmer errors (``TypeError``,
  ``AttributeError``, ``NameError``, ``RuntimeError``) now bubble up
  instead of being silently swallowed.

**Centralized 1-based ID validation** (``pyiwfm.core.ids``)

- New ``to_index(one_based_id, n_items, *, kind="id") -> int`` and
  ``to_indices(one_based_ids, n_items, *, kind="id") -> NDArray[int64]``
  helpers replace inline ``id - 1`` arithmetic. Both validate bounds
  and raise ``ValueError`` with a self-documenting message that
  includes ``kind`` (``"element"`` / ``"node"`` / ``"reach"``) and the
  offending value(s).
- Adopted in ``core/zones.py`` (4 sites) and ``core/aggregation.py``,
  closing a class of silent-overwrite bugs where invalid IDs would
  write past the end of preallocated arrays.

**Frontend error containment** (``frontend/src/components/common/ErrorBoundary.tsx``)

- New React class component wraps each viewer tab body in
  ``frontend/src/App.tsx`` so a render-time crash in one tab
  (Plotly, vtk.js, deck.gl) leaves the other tabs usable. Renders a
  fallback UI with a "Try again" button.

**CI hardening** (``.github/workflows/ci.yml``)

- ``mypy src/pyiwfm/`` now runs across the full Python 3.10–3.14 ×
  Ubuntu/Windows test matrix instead of only Python 3.12 / Ubuntu.
- New ``docs`` job runs ``sphinx-build -W docs docs/_build`` so
  documentation rot (broken refs, missing modules, malformed RST)
  fails CI instead of silently shipping.

Changed
~~~~~~~

- ``pyiwfm.io.streams.depletion._extract_stream_flow`` now requires an
  exact column-name match (default ``"Stream-Aquifer Interaction
  Within Model"``). The previous substring matching against
  ``"gain from gw"`` / ``"stream-aquifer"`` and the ``(+)/(-)`` column
  sum-fallback were removed because they could silently misclassify
  columns. Older models that emit ``"Gain from GW (+)"`` should pass
  ``sa_column="Gain from GW (+)"`` explicitly.

Fixed
~~~~~

- ``pyiwfm.io.groundwater.main_writer.write_gw_main_file`` now coalesces the
  per-cell ``f.write`` loop for aquifer parameters (and similarly for
  initial heads) into a single write per block. On a C2VSimFG-class
  model (~30k nodes × 4 layers) this collapses ~120k syscalls into one,
  making ``save_complete_model`` substantially faster. Output is
  byte-identical to v1.1.x; locked in by ``tests/unit/test_gw_writer_vectorized.py``.
- Same coalescing applied to ``gw_subsidence_writer``'s parametric grid
  block.
- ``pyiwfm.visualization.webapi.routes.export`` — five ``json.dumps``
  call sites (mesh GeoJSON export, model report, three records-export
  endpoints) now use a shared ``_json_default`` callable that handles
  ``np.integer``, ``np.floating`` (NaN → null), ``np.ndarray``,
  ``pd.Timestamp``, and ``bytes``. Mesh GeoJSON exports no longer fail
  with ``TypeError: Object of type int64 is not JSON serializable``
  on models with NumPy ``int64`` element indices.
- ``docs/user_guide/io.rst`` — example code uses ``model.mesh.n_nodes``
  instead of the alias ``model.grid.n_nodes`` (``mesh`` is the canonical
  attribute per CLAUDE.md).
- ``CLAUDE.md`` line 141 — corrected stale comment claiming
  ``webapi/slicing.py`` and ``webapi/properties.py`` are shims (they
  are 600-line full implementations imported by ``routes/slices.py``
  and ``_mesh_state.py`` respectively).

[1.1.3] - 2026-04-23
--------------------

Added
~~~~~

- Python 3.14 support: CI now tests Ubuntu + Windows × 3.14, and
  ``Programming Language :: Python :: 3.14`` added to the classifiers.

Fixed
~~~~~

- Tighten ``[viz]`` and ``[webapi]`` extras to ``vtk>=9.6`` and
  (for ``[webapi]``) ``pyvista>=0.47``. vtk 9.6.0 is the first
  version with Python 3.14 wheels, so the previous ``vtk>=9.0`` pin
  let pip backtrack through older vtk versions during resolution and
  emit a misleading ResolutionImpossible error in some user
  environments on Python 3.14. The new minimums keep pip on vtk
  versions that have 3.14 wheels and align with pyvista 0.47's
  transitive ``vtk>=9.2.2,<9.7.0`` constraint.

[1.1.2] - 2026-04-23
--------------------

Added
~~~~~

**Stratigraphy Aquitard Accessors and Builder** (``pyiwfm.core.stratigraphy``)

- ``Stratigraphy.n_aquitards`` property (equals ``n_layers``)
- ``Stratigraphy.get_aquitard_thickness(aquitard)`` — single aquitard
  thickness at every node (aquitard 0 = top aquitard above aquifer layer 0;
  aquitard *k* = between layer *k-1* bottom and layer *k* top)
- ``Stratigraphy.get_all_aquitard_thicknesses()`` — vectorised
  ``(n_nodes, n_aquitards)`` array
- ``Stratigraphy.get_node_aquitards(node_idx)`` — list of aquitard thicknesses
  at a specific node, mirroring ``get_node_elevations``
- ``Stratigraphy.from_thicknesses(gs_elev, aquitard_thicknesses,
  aquifer_thicknesses, active_node=None)`` classmethod — construct a
  Stratigraphy directly from per-layer aquitard + aquifer thickness
  arrays (vectorised via ``np.cumsum``). Matches the IWFM file-format
  convention so users no longer need to hand-roll cumulative elevation
  math when programmatically generating stratigraphies with aquitards.
- The stratigraphy writer and ``read_stratigraphy`` now both delegate
  the aquitard math to these helpers so it has a single owner.

**AquiferParameters Dispatcher** (``pyiwfm.components.groundwater``)

- ``AquiferParameters.get_array(param)`` / ``.get_layer(param, layer)`` /
  ``.get_at(param, node_idx, layer)`` — unified accessors keyed by short
  parameter names (``"kh"``, ``"kv"``, ``"ss"``, ``"sy"``, ``"aquitard_kv"``);
  unknown names raise ``KeyError``, unset arrays raise ``ValueError``
- Existing ``get_layer_kh`` / ``get_layer_kv`` are now thin shims over the
  dispatcher (no breaking change)

**Stream Reach → Groundwater Node Mapping** (``pyiwfm.components.stream``)

- ``AppStream.get_gw_nodes_in_reach(reach_id)`` returns
  ``list[int | None]`` of gw node IDs in upstream-to-downstream order

**Element Centroid Cache** (``pyiwfm.core.mesh``)

- ``AppGrid.element_centroids`` lazy property — vectorised
  ``(n_elements, 2)`` array built in a single pass, cached until mesh
  mutation invalidates
- ``AppGrid.get_element_centroid(element_id)`` now an O(1) dict + array
  lookup after first call (previously re-computed per call)

**Observation Upload Format Expansion**

- Web viewer observation upload now supports CSV, TSV, SMP, and generic
  whitespace-delimited text files (previously CSV only)
- Automatic delimiter detection (comma, tab, whitespace) from file content
- Content-based SMP format sniffing — SMP data is recognized regardless of
  file extension (``.txt``, ``.dat``, ``.smp``, etc.)
- Frontend upload wizard auto-detects format and skips column mapping for SMP
- File extensions ``.csv``, ``.tsv``, ``.txt``, ``.dat``, ``.smp`` all accepted
- Normalized upload API response shape for consistent frontend handling

**Calibration Enhancements**

- ``compute_composite_continuous()``: T-weighted composite heads at all
  simulation timesteps (full temporal resolution for CalcTypHyd)
- ``average_to_seasonal()``: Aggregate continuous monthly series to seasonal
  window averages for sim/obs comparison
- ``compute_obs_type_hydrographs()``: Convenience wrapper for observed type
  hydrograph computation with PEST output
- ``BIANNUAL_SEASONS``: Spring/Fall seasonal period definitions for clustering

Changed
~~~~~~~

- Budget plot colors: improved contrast for Ice Blue and Cream palette entries
- Budget table: colored squares before component names, adjusted column widths,
  moved discrepancy percentage to its own row

**Simulation Reader Refactor**

- ``IWFMSimulationReader.read()`` split into focused section methods
  (``_read_titles``, ``_read_input_files``, ``_read_time_settings``,
  ``_read_processing_options``, ``_read_solver_settings``,
  ``_read_convergence_tail``) for easier piecewise testing. No behavior
  change; 11 fixed input-file slots now driven by the
  ``_INPUT_FILE_FIELDS`` table.

**Test & Dev-Env Improvements**

- ``[dev]`` extra now pulls in ``[mesh]`` so ``pip install -e ".[dev]"``
  yields a full test environment (mesh-wrapper unit tests no longer silently
  skip); triangle pin relaxed to ``>=20200424`` with a Python 3.14 compat
  marker.
- Graceful module-/class-level ``importorskip`` guards for ``hypothesis``
  property-test files and ``pytest-benchmark``-dependent classes.
- ``tests/unit/test_results_extraction_subsidence.py`` resolves the C2VSimFG
  subsidence HDF5 via ``C2VSIMFG_SUBS_HDF`` or ``C2VSIMFG_DIR`` env vars
  (previously hard-coded) so CI and contributors can unskip those 3 tests
  by configuring the existing data location.
- New integration test ``tests/integration/test_stratigraphy_roundtrip.py``
  exercises aquitard helpers and read → write → read against the IWFM
  Sample Model and C2VSimFG.

Fixed
~~~~~

- Release workflow now uses ``python -m build`` instead of
  ``hatch build``, so the wheel is built from the sdist (isolated)
  rather than from the source tree. This prevents the sdist-then-wheel
  version-mismatch that affected v1.1.0 and v1.1.1, where
  ``hatch_build.py``'s ``npm run build`` could dirty tracked files
  between packaging steps and cause hatch-vcs to bump the wheel to the
  next dev version. The workflow also pins the version from the git
  tag via ``SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PYIWFM`` and asserts
  sdist/wheel filenames match before uploading, as a safety net.
- ``pyiwfm.io.ascii.read_stratigraphy`` now correctly handles stratigraphy
  files whose node IDs are not 1-based contiguous. The previous dead-code
  ``idx = node_id - 1`` branch silently scrambled row assignments; now
  builds a ``node_id → row`` mapping from the file and raises
  ``FileFormatError`` on duplicate IDs.
- ``DSSFile.__init__`` validates the ``mode`` argument (raises
  ``ValueError`` for anything other than ``"r"`` / ``"w"`` / ``"rw"``) and
  the three dead ``pass``-only mode-dispatch branches in ``open()`` have
  been removed — HEC-DSS 7's ``zopenExtended`` does not expose per-mode
  open semantics, so the branches had no effect.
- ``pyiwfm.io.supply_adjust`` now imports ``COMMENT_CHARS`` and
  ``strip_inline_comment`` from ``pyiwfm.io.ascii.reader`` instead of
  duplicating them locally (per CLAUDE.md's "never duplicate these
  helpers" rule). Blank-line-as-data semantics for the DSSFL slot are
  preserved via a small local ``_is_fortran_comment`` wrapper.

[1.1.0] - 2026-03-13
--------------------

Drawdown, Diagnostics, Stream Depletion, Mesh Quality, Budget Checks, and PEST++ CLI

Added
~~~~~

**Drawdown Analysis** (``pyiwfm.io.drawdown``)

- ``DrawdownComputer``: Compute per-node and per-element drawdown relative to
  a reference timestep
- ``compute_drawdown_range()``: Robust min/max (2nd–98th percentile) for
  consistent color scaling across animation frames
- ``compute_max_drawdown_map()``: Maximum drawdown at each node across all
  timesteps
- Web viewer: *Color By: Drawdown* mode in Results Map with diverging color
  scale and reference timestep slider

**Stream Depletion** (``pyiwfm.io.streams.depletion``)

- ``compute_stream_depletion()``: Compare baseline and pumping-scenario model
  runs to quantify per-reach stream flow depletion
- ``StreamDepletionResult``: Per-reach timeseries (baseline, scenario,
  depletion, cumulative, max, total)
- ``StreamDepletionReport``: Aggregate report across multiple reaches

**Budget Checks** (``pyiwfm.io.budget.checks``)

- ``check_budget_balance()``: Per-location mass balance sanity check with
  configurable tolerance
- ``check_all_budgets()``: Check mass balance across all available budget types
- ``BudgetSanityReport``: Summary of violations, max/mean percent error

**Mesh Quality** (``pyiwfm.core.mesh_quality``)

- ``compute_mesh_quality()``: Element-level quality metrics (aspect ratio,
  skewness, min/max angle, area) with aggregate statistics
- ``MeshQualityReport``: Summary with poor-quality element count and
  serialization to dict for API responses
- Web viewer: Mesh quality card on Overview tab

**Simulation Diagnostics** (``pyiwfm.visualization.webapi.routes.diagnostics``)

- ``/api/diagnostics/messages``: Paginated messages with severity filtering
- ``/api/diagnostics/convergence``: Iteration count timeseries
- ``/api/diagnostics/mass-balance``: Per-component mass balance error
- ``/api/diagnostics/summary``: Combined diagnostics summary
- ``/api/diagnostics/spatial-summary``: Spatial message counts for map overlay
- Web viewer: Diagnostics tab with interactive charts and message table

**Unsaturated Zone Routes** (``pyiwfm.visualization.webapi.routes.unsaturated_zone``)

- ``/api/unsaturated-zone/``: Component info endpoint
- ``/api/unsaturated-zone/summary``: Parameter summary endpoint

**PEST++ CLI** (``pyiwfm.cli.pest``)

- ``pyiwfm pest setup``: Generate PEST++ control, template, and instruction
  files from an IWFM model directory
- ``pyiwfm pest run``: Execute PEST++ with configurable executable and workers
- ``pyiwfm pest analyze``: Post-process results in text or JSON format

**Web Viewer Enhancements**

- Subsidence surface visualization on Results Map
- Model comparison dialog with mesh/stratigraphy diff
- HTML/JSON report export endpoints
- Small watersheds and unsaturated zone API routes

**Documentation**

- New tutorials: Drawdown Analysis, Stream Depletion, Simulation Diagnostics,
  Mesh Quality
- New gallery pages: Mesh Quality, Drawdown Maps, Convergence
- Updated API reference with new modules
- Updated web viewer guide with 6-tab layout and new features
- PEST++ CLI section in user guide
- Diagnostics user guide

[1.0.4] - 2026-03-04
--------------------

IWFM2OBS Timestamp Alignment and CalcTypHyd Fortran-Matching Features

Added
~~~~~

**IWFM2OBS Timestamp Alignment** (``pyiwfm.calibration.iwfm2obs``)

- ``_compute_model_dates()``: Replicate Fortran ``ComputeDate`` for 1MON/1DAY/1WEEK/1YEAR
- ``_replace_timestamps()``: Replace parsed ``.out`` timestamps with computed dates
- ``expand_obs_to_layers()``: Auto-expand base observation IDs to per-layer variants
- ``deduplicate_smp()``: Strip ``%N`` layer suffixes from SMP files
- CLI: ``--deduplicate-smp`` mode
- Verified: max ``|diff|`` = 0.000060 ft vs Fortran on C2VSimFG (48,816 GW bore IDs)

**CalcTypHyd Fortran-Matching Features** (``pyiwfm.calibration.calctyphyd``)

- ``read_calctyphyd_config()`` / ``CalcTypHydFileConfig``: Parse Fortran ``.in`` config files
- ``compute_typical_hydrographs_timeseries()``: Per-period-year output matching Fortran
  algorithm (period-year slot averaging, mean from slot averages)
- ``write_pest_output()``: Write PEST ``.out``/``.ins`` files with exact Fortran column format
- ``read_cluster_weights(n_clusters=)``: Header detection and column limiting
- ``CalcTypHydConfig.start_date/end_date``: Date-range filtering
- CLI: ``--config`` mode for Fortran ``.in`` files, ``--output-dir`` for PEST files
- Verified: byte-identical output vs Fortran on C2VSimFG (6 clusters × 62 entries)

[1.0.2] - 2026-02-27
--------------------

IWFM2OBS Model Discovery and Multi-Layer Output

Added
~~~~~

**Model File Discovery** (``pyiwfm.calibration.model_file_discovery``)

- ``discover_hydrograph_files()``: Parse IWFM simulation main file to auto-discover
  GW and stream hydrograph ``.out`` file paths and hydrograph metadata (bore IDs,
  layers, coordinates)
- ``HydrographFileInfo``: Dataclass with discovered paths, hydrograph locations,
  start date, and time unit

**Observation Well Specification** (``pyiwfm.calibration.obs_well_spec``)

- ``read_obs_well_spec()``: Read observation well specification files for multi-layer
  target processing (name, coordinates, element, screen top/bottom)
- ``ObsWellSpec``: Dataclass for multi-layer well screen geometry

**IWFM2OBS Model-Discovery Mode** (``pyiwfm.calibration.iwfm2obs``)

- ``iwfm2obs_from_model()``: Full IWFM2OBS workflow that reads ``.out`` files directly
  via simulation main file discovery — combines the old Fortran IWFM2OBS's direct
  file reading with the new multi-layer T-weighted averaging
- ``IWFM2OBSConfig``: Configuration dataclass for the integrated workflow
- ``write_multilayer_output()``: Write ``GW_MultiLayer.out`` format (Name, Date, Time,
  Simulated, T1-T4, NewTOS, NewBOS)
- ``write_multilayer_pest_ins()``: Write PEST instruction file with ``WLT{well:05d}_{timestep:05d}``
  naming at columns 50:60

**Hydrograph Reader Enhancement** (``pyiwfm.io.hydrograph_reader``)

- ``IWFMHydrographReader.get_columns_as_smp_dict()``: Extract ``.out`` columns as
  ``SMPTimeSeries`` dict, bridging the hydrograph reader to the interpolation pipeline

**CLI Model-Discovery Mode** (``pyiwfm.cli.iwfm2obs``)

- ``--model`` flag for automatic model file discovery from simulation main file
- ``--obs-gw``, ``--output-gw``, ``--obs-stream``, ``--output-stream`` for per-type
  observation/output SMP paths
- ``--well-spec``, ``--multilayer-out``, ``--multilayer-ins`` for multi-layer processing

[1.0.0] - 2026-02-24
--------------------

Calibration Tools, Clustering, and Publication-Quality Plotting

Added
~~~~~

**SMP Observation File I/O** (``pyiwfm.io.smp``)

- ``SMPReader`` / ``SMPWriter``: Read and write IWFM SMP (Sample/Bore) observation files
- ``SMPRecord``, ``SMPTimeSeries``: Typed containers for bore ID, datetime, value, exclusion flag
- Fixed-width parsing with sentinel value (NaN) handling

**SimulationMessages.out Parser** (``pyiwfm.io.simulation.messages``)

- ``SimulationMessagesReader``: Parse IWFM simulation diagnostic output files
- ``SimulationMessage``: Structured message with severity, procedure, spatial IDs
- Regex-based extraction of node, element, reach, and layer IDs
- ``to_geodataframe()``: Map messages to spatial locations for GIS analysis

**IWFM2OBS Time Interpolation** (``pyiwfm.calibration.iwfm2obs``)

- ``interpolate_to_obs_times()``: Linear/nearest interpolation of simulated to observed times
- ``compute_multilayer_weights()``: Transmissivity-weighted averaging for multi-layer wells
- ``compute_composite_head()``: Composite head from layer heads using T-weights
- ``iwfm2obs()``: Complete workflow function matching Fortran IWFM2OBS utility

**Typical Hydrograph Computation** (``pyiwfm.calibration.calctyphyd``)

- ``compute_typical_hydrographs()``: Seasonal averaging + de-meaning + weighted combination
- ``compute_seasonal_averages()``: Per-well seasonal period averaging
- ``read_cluster_weights()``: Parse cluster weight files for CalcTypHyd input

**Fuzzy C-Means Clustering** (``pyiwfm.calibration.clustering``)

- ``fuzzy_cmeans_cluster()``: NumPy-only fuzzy c-means with spatial + temporal features
- ``ClusteringResult``: Membership matrix, cluster centers, fuzzy partition coefficient
- ``to_weights_file()``: Export weights in CalcTypHyd-compatible format
- Feature extraction: cross-correlation, amplitude, trend, seasonal strength

**Calibration Plots** (``pyiwfm.visualization.calibration_plots``)

- ``plot_calibration_summary()``: Multi-panel publication figure (1:1, spatial bias, histogram, metrics)
- ``plot_hydrograph_panel()``: Grid of observed vs simulated hydrographs
- ``plot_metrics_table()``: Matplotlib table of per-well statistics
- ``plot_residual_histogram()``: Residual distribution with optional normal fit
- ``plot_water_budget_summary()`` / ``plot_zbudget_summary()``: Stacked bar budget charts
- ``plot_cluster_map()``: Spatial cluster membership visualization
- ``plot_typical_hydrographs()``: Overlay of typical hydrographs by cluster

**New Plot Functions** (``pyiwfm.visualization.plotting``)

- ``plot_one_to_one()``: Scatter with 1:1 line, regression, and metrics text box
- ``plot_spatial_bias()``: Diverging colormap of observation bias on mesh background

**Publication Matplotlib Style** (``visualization/styles/pyiwfm-publication.mplstyle``)

- Journal-quality defaults: serif fonts, no top/right spines, 300 DPI, constrained layout

**Scaled RMSE Metric** (``pyiwfm.comparison.metrics``)

- ``scaled_rmse()``: Dimensionless RMSE / (max - min) for cross-site comparison
- Added ``scaled_rmse`` field to ``ComparisonMetrics`` dataclass

**CLI Subcommands**

- ``pyiwfm iwfm2obs``: Time interpolation of simulated to observed SMP times
- ``pyiwfm calctyphyd``: Compute typical hydrographs from clustered observation wells

[0.4.0] - 2026-01-15
--------------------

Supplemental Package Support and Web Viewer Enhancements

Fixed
~~~~~

**Root Zone Version-Dependent Parsing Bugs**

- Fixed ARSCLFL (land use area scaling) version guard: only read for v4.12+, not v4.11+
- Fixed FinalMoistureOutFile (FMFL) read: skip for v4.12+ where it was removed
- Fixed root zone soil parameter table parsing when ``n_elements`` is known

Changed
~~~~~~~

**I/O Reader Deduplication** (``pyiwfm.io.ascii.reader``)

- Centralized ``resolve_path()``, ``next_data_or_empty()``, ``parse_version()``,
  and ``version_ge()`` into ``iwfm_reader.py`` — the canonical module for all
  IWFM file-reading utilities
- Replaced 14 identical ``_next_data_or_empty`` method copies across reader modules
  with thin wrappers delegating to the central function
- Replaced 11 identical ``_resolve_path`` copies (10 methods + 1 module-level function
  in ``preprocessor.py``) with delegations to ``iwfm_reader.resolve_path()``
- Unified version parsing: ``rootzone.parse_version`` and ``streams.parse_stream_version``
  merged into ``iwfm_reader.parse_version()`` (handles both ``.`` and ``-`` separators)
- Net reduction of ~42 lines across 16 files with no behavior changes

Added
~~~~~

**Small Watershed Component** (``pyiwfm.components.small_watershed``)

- ``AppSmallWatershed``: Container for small watershed model units
- ``WatershedUnit``: Individual watershed with root zone and aquifer parameters
- ``WatershedGWNode``: Groundwater node connection with percolation rates
- ``from_config()``: Build component from reader config
- ``validate()``: Check areas, stream node references, GW node connectivity

**Unsaturated Zone Component** (``pyiwfm.components.unsaturated_zone``)

- ``AppUnsatZone``: Container for unsaturated zone elements
- ``UnsatZoneElement``: Per-element layer data with initial soil moisture
- ``UnsatZoneLayer``: Layer-level soil hydraulic properties
- ``from_config()``: Build component from reader config
- ``validate()``: Check layer counts and element consistency

**Small Watershed Writer** (``pyiwfm.io.small_watershed.writer``)

- ``SmallWatershedComponentWriter``: Template-based writer for small watershed files
- ``SmallWatershedWriterConfig``: Writer configuration with output paths
- Jinja2 template for IWFM v4.0 format with geospatial, root zone, and aquifer sections

**Unsaturated Zone Writer** (``pyiwfm.io.unsaturated_zone.writer``)

- ``UnsatZoneComponentWriter``: Template-based writer for unsaturated zone files
- ``UnsatZoneWriterConfig``: Writer configuration with output paths
- Jinja2 template for IWFM v4.0 format with element data and initial moisture sections

**Model Integration**

- ``IWFMModel.small_watersheds``: Optional small watershed component attribute
- ``IWFMModel.unsaturated_zone``: Optional unsaturated zone component attribute
- ``CompleteModelWriter``: Dedicated writers replace passthrough file copy for both packages
- Full roundtrip support for small watershed and unsaturated zone files

**BaseComponent ABC** (``pyiwfm.core.base_component``)

- ``BaseComponent``: Abstract base class with ``validate()`` and ``n_items`` interface
- All 6 components (AppGW, AppStream, AppLake, RootZone, AppSmallWatershed,
  AppUnsatZone) inherit from ``BaseComponent``

**Model Factory Extraction** (``pyiwfm.core.model_factory``)

- Extracted 6 helper functions (~420 lines) from ``IWFMModel`` to reduce
  the God Object pattern
- Public functions: ``build_reaches_from_node_reach_ids``,
  ``apply_kh_anomalies``, ``apply_parametric_grids``,
  ``apply_parametric_subsidence``, ``binary_data_to_model``,
  ``resolve_stream_node_coordinates``
- ``IWFMModel`` classmethods delegate to factory functions (backward compatible)

**Writer Config Consolidation** (``pyiwfm.io.writer_config_base``)

- ``BaseComponentWriterConfig``: Shared base dataclass for all 6 component
  writer configs (common fields: output_dir, version, length/volume
  factors/units, subdir)

**I/O Reader Deduplication**

- Consolidated ``_resolve_path()`` and ``_parse_version()`` duplicates across
  readers to use canonical implementations from ``iwfm_reader.py``
- Consolidated Fortran binary record I/O between ``base.py`` and ``binary.py``

**Web Viewer Performance Improvements**

- Cached ``node_id_to_idx`` and ``elem_id_to_idx`` mappings in ``ModelState``
  instead of rebuilding per request
- Cached hydrograph location data (GW and stream) in ``ModelState``
- Drawdown endpoint now supports pagination: ``offset``, ``limit``, ``skip``
  parameters for frame-by-frame animation playback

**New Web Viewer API Endpoints**

- ``GET /api/export/geopackage``: Download multi-layer GeoPackage (nodes,
  elements, streams, subregions, boundary) via ``GISExporter``
- ``GET /api/export/plot/{plot_type}``: Generate publication-quality matplotlib
  figures (mesh, elements, streams, heads) as PNG or SVG
- ``POST /api/model/compare``: Load a second model and compare meshes/stratigraphy
  via ``ModelDiffer``
- ``GET /api/results/statistics``: Time-aggregated head statistics
  (min, max, mean, std per node across all timesteps)

**FastAPI Web Viewer Enhancements** (2026-02)

- Results Map tab with deck.gl + MapLibre GL for head contour visualization
- Budgets tab with Plotly charts for GW, stream, and other budget types
- Hydrograph overlay with observed vs simulated data
- Server-side coordinate reprojection via pyproj (model CRS to WGS84)
- Stream node z elevation from stratigraphy ground surface
- Monthly budget timestep support using ``relativedelta``
- Budget units populated from column type codes
- CRS default corrected to proj string for C2VSimFG

**Component Exports**

- Top-level ``pyiwfm`` package now exports AppGW, AppStream, AppLake, RootZone,
  AppSmallWatershed, AppUnsatZone for convenient ``from pyiwfm import AppGW``

**Budget & Zone Budget Excel Export** (``pyiwfm.io.budget.excel``, ``pyiwfm.io.budget.zone_excel``)

- ``budget_to_excel()``: Export budget HDF5 data to formatted Excel workbooks
  (one sheet per location, bold titles/headers, auto-fit columns)
- ``zbudget_to_excel()``: Same for zone budget data (one sheet per zone)
- ``budget_control_to_excel()`` / ``zbudget_control_to_excel()``: Batch export
  from control file configuration (one ``.xlsx`` per budget spec)
- Unit conversion factors (FACTLTOU/FACTAROU/FACTVLOU) applied per column type
  using codes from ``Budget_Parameters.f90``

**Budget & Zone Budget Control File Parsers** (``pyiwfm.io.budget.control``, ``pyiwfm.io.budget.zone_control``)

- ``read_budget_control()``: Parse IWFM budget post-processor control files
  (FACTLTOU, UNITLTOU, FACTAROU, UNITAROU, FACTVLOU, UNITVLOU, dates,
  per-budget HDF5 paths, output paths, location IDs)
- ``read_zbudget_control()``: Parse IWFM zone budget control files
  (same pattern with zone definition file support)
- ``BudgetControlConfig`` / ``ZBudgetControlConfig``: Typed dataclasses

**Budget Utilities** (``pyiwfm.io.budget._utils``)

- ``apply_unit_conversion()``: Apply IWFM conversion factors per column type
- ``format_title_lines()``: Substitute ``@UNITVL@``, ``@UNITAR@``, ``@LOCNAME@``,
  ``@AREA@`` markers in title templates
- ``filter_time_range()``: Filter DataFrames by IWFM date range

**Budget CLI Commands** (``pyiwfm.cli.budget``, ``pyiwfm.cli.zbudget``)

- ``pyiwfm budget <control_file>``: Export budgets to Excel from control file
- ``pyiwfm zbudget <control_file>``: Export zone budgets to Excel from control file
- ``--output-dir`` flag to override output directory

**Budget Unit Conversion in Readers**

- ``BudgetReader.get_dataframe()`` now accepts keyword-only ``length_factor``,
  ``area_factor``, ``volume_factor`` for on-the-fly unit conversion
- ``ZBudgetReader.get_dataframe()`` now accepts keyword-only ``volume_factor``
- Backward compatible: all factors default to 1.0

**Budget Excel Download Endpoints**

- ``GET /api/budgets/{budget_type}/excel``: Download formatted budget workbook
- ``GET /api/export/budget-excel``: Download budget Excel from the export routes

**Documentation**

- Added ~30 missing module entries to API docs (``docs/api/io.rst``)
- Added Small Watershed and Unsaturated Zone to component docs
- Reorganized I/O docs into logical sections (Core, GW, Stream, Lake, RZ, Supplemental, etc.)
- Added BaseComponent and model_factory to core API docs
- Added writer_config_base to I/O API docs
- Added API routes summary to visualization docs

[0.2.0] - 2025-06-01
--------------------

Complete IWFM File Writers Implementation

Added
~~~~~

**IWFMModel Class Methods**

- ``IWFMModel.from_preprocessor(pp_file)``: Load from preprocessor input files (mesh, stratigraphy, geometry)
- ``IWFMModel.from_preprocessor_binary(binary_file)``: Load from native IWFM preprocessor binary (``ACCESS='STREAM'``)
- ``IWFMModel.from_simulation(sim_file)``: Load complete model from simulation input file
- ``IWFMModel.from_simulation_with_preprocessor(sim_file, pp_file)``: Load using both files
- ``IWFMModel.from_hdf5(hdf5_file)``: Load from HDF5 file
- ``model.to_preprocessor(output_dir)``: Save to preprocessor input files
- ``model.to_simulation(output_dir)``: Save complete model to simulation files
- ``model.to_hdf5(output_file)``: Save to HDF5 format
- ``model.to_binary(output_file)``: Save mesh/stratigraphy to binary format
- ``model.grid`` property: Alias for ``mesh`` for compatibility

**Complete Model I/O**

- ``pyiwfm.io.load_complete_model``: Load complete IWFM model from simulation main file
- ``pyiwfm.io.save_complete_model``: Save complete IWFM model to all input files
- Full roundtrip support for reading, modifying, and writing IWFM models

**Time Series ASCII I/O**

- ``pyiwfm.io.timeseries.ascii``: ASCII time series reader/writer module
- ``TimeSeriesWriter``: Write IWFM ASCII time series files with 21-char timestamp format
- ``TimeSeriesReader``: Read IWFM ASCII time series files
- ``format_iwfm_timestamp``: Format datetime to IWFM 21-character timestamp
- ``parse_iwfm_timestamp``: Parse IWFM timestamp string to datetime

**Groundwater Component I/O**

- ``pyiwfm.io.groundwater``: Complete groundwater component file I/O
- ``GroundwaterWriter``: Write wells, pumping, boundary conditions, aquifer parameters
- ``GroundwaterReader``: Read groundwater component files
- ``GWFileConfig``: Configuration for groundwater file paths

**Stream Component I/O**

- ``pyiwfm.io.streams``: Complete stream network component file I/O
- ``StreamWriter``: Write stream nodes, reaches, diversions, bypasses, rating curves
- ``StreamReader``: Read stream component files
- ``StreamFileConfig``: Configuration for stream file paths

**Lake Component I/O**

- ``pyiwfm.io.lakes``: Complete lake component file I/O
- ``LakeWriter``: Write lake definitions, elements, rating curves, outflows
- ``LakeReader``: Read lake component files
- ``LakeFileConfig``: Configuration for lake file paths

**Root Zone Component I/O**

- ``pyiwfm.io.rootzone``: Complete root zone component file I/O
- ``RootZoneWriter``: Write crop types, soil parameters, land use
- ``RootZoneReader``: Read root zone component files
- ``RootZoneFileConfig``: Configuration for root zone file paths

**Simulation Control I/O**

- ``pyiwfm.io.simulation``: Simulation control file I/O
- ``SimulationWriter``: Write simulation main control file
- ``SimulationReader``: Read simulation control file
- ``SimulationConfig``: Simulation configuration dataclass

**HEC-DSS 7 Support**

- ``pyiwfm.io.dss``: Complete HEC-DSS 7 time series I/O package
- ``DSSFile``: Context manager for DSS file operations using ctypes
- ``DSSPathname``: DSS pathname representation (/A/B/C/D/E/F/)
- ``DSSPathnameTemplate``: Template for generating pathnames
- ``DSSTimeSeriesWriter``: High-level time series writer
- ``DSSTimeSeriesReader``: High-level time series reader
- ``write_timeseries_to_dss``: Convenience function for single time series
- ``read_timeseries_from_dss``: Convenience function for reading time series
- ``write_collection_to_dss``: Write TimeSeriesCollection to DSS
- ``HAS_DSS_LIBRARY``: Flag indicating DSS library availability

**Template Engine Updates**

- ``iwfm_timestamp`` filter: Format datetime for IWFM files
- ``dss_pathname`` filter: Format DSS pathnames
- ``timeseries_ref`` filter: Reference time series files
- ``iwfm_array_row`` filter: Format array rows for IWFM files

Changed
~~~~~~~

- Updated ``pyiwfm.io.__init__.py`` with all new exports
- Extended ``pyiwfm.io.preprocessor`` with ``load_complete_model()`` and ``save_complete_model()``

[0.3.0] - 2025-10-01
--------------------

PEST++ Calibration Interface, Multi-Scale Viewing, and Subprocess Runner

Added
~~~~~

**Subprocess Runner** (``pyiwfm.runner``)

- ``IWFMRunner``: Run IWFM executables (Preprocessor, Simulation, Budget, ZBudget) via subprocess
- ``IWFMExecutables``: Locate and manage IWFM executable paths
- ``find_iwfm_executables()``: Auto-detect executables in PATH or specified directories
- ``RunResult``, ``PreprocessorResult``, ``SimulationResult``, ``BudgetResult``, ``ZBudgetResult``: Typed result classes

**Scenario Management** (``pyiwfm.runner.scenario``)

- ``Scenario``: Define named model scenarios with parameter overrides
- ``ScenarioManager``: Manage, run, and compare multiple scenarios
- ``ScenarioResult``: Collect and compare scenario outputs

**PEST++ Integration** (``pyiwfm.runner.pest``)

- ``PESTInterface``: Low-level PEST++ control file interface
- ``TemplateFile``, ``InstructionFile``: PEST++ template/instruction file management
- ``ObservationGroup``: Observation group definitions
- ``write_pest_control_file()``: Generate PEST++ control files (.pst)

**PEST++ Parameter Management** (``pyiwfm.runner.pest_params``)

- ``IWFMParameterType``: Enum of all IWFM parameter types (aquifer, stream, lake, rootzone, flux multipliers)
- ``ParameterTransform``: Log/none/tied/fixed transform types
- ``ParameterGroup``, ``Parameter``: Parameter definitions with bounds and transforms
- Parameterization strategies: ``ZoneParameterization``, ``MultiplierParameterization``,
  ``PilotPointParameterization``, ``DirectParameterization``, ``StreamParameterization``,
  ``RootZoneParameterization``
- ``IWFMParameterManager``: Central parameter registry and management

**PEST++ Observation Management** (``pyiwfm.runner.pest_observations``)

- ``IWFMObservationType``: Enum of observation types (head, drawdown, flow, stage, etc.)
- ``IWFMObservation``, ``IWFMObservationGroup``: Observation definitions with weights
- ``ObservationLocation``: Spatial location for observations
- ``WeightStrategy``: Equal, inverse variance, decay, and group contribution weighting
- ``DerivedObservation``: Computed observations (gradients, differences)
- ``IWFMObservationManager``: Central observation registry
- ``WellInfo``, ``GageInfo``: Monitoring point metadata

**PEST++ Template/Instruction Generation** (``pyiwfm.runner.pest_templates``, ``pest_instructions``)

- ``IWFMTemplateManager``: Generate PEST++ template files (.tpl) from IWFM input files
- ``TemplateMarker``: Define parameter marker locations in templates
- ``IWFMFileSection``: Track file sections for template generation
- ``IWFMInstructionManager``: Generate PEST++ instruction files (.ins) from IWFM output
- ``OutputFileFormat``: Define output file parsing rules
- ``IWFM_OUTPUT_FORMATS``: Predefined formats for standard IWFM outputs

**Geostatistics** (``pyiwfm.runner.pest_geostat``)

- ``VariogramType``: Spherical, exponential, Gaussian, power variogram models
- ``Variogram``: Variogram definition with nugget, sill, range parameters
- ``GeostatManager``: Manage spatial correlation structures and pilot point kriging
- ``compute_empirical_variogram()``: Compute empirical variograms from spatial data

**Main PEST++ Helper Interface** (``pyiwfm.runner.pest_helper``)

- ``IWFMPestHelper``: High-level interface coordinating all PEST++ components
- Convenience methods: ``add_zone_parameters()``, ``add_multiplier()``, ``add_pilot_points()``,
  ``add_stream_parameters()``, ``add_rootzone_parameters()``
- ``add_head_observations()``, ``add_streamflow_observations()``
- ``set_svd()``, ``set_regularization()``, ``set_model_command()``, ``set_pestpp_options()``
- ``build()``: Generate complete PEST++ setup (control file, templates, instructions, scripts)
- ``run_pestpp()``: Execute PEST++ from within Python
- ``SVDConfig``, ``RegularizationConfig``, ``RegularizationType``: Configuration classes

**Ensemble Management** (``pyiwfm.runner.pest_ensemble``)

- ``IWFMEnsembleManager``: Prior/posterior ensemble generation for pestpp-ies
- ``EnsembleStatistics``: Statistical analysis of parameter ensembles
- Latin Hypercube Sampling and geostatistical realization generation
- CSV I/O compatible with PEST++ ensemble format
- Uncertainty reduction analysis between prior and posterior ensembles

**Post-Processing** (``pyiwfm.runner.pest_postprocessor``)

- ``PestPostProcessor``: Load and analyze PEST++ output files (.rei, .sen, .iobj, .par)
- ``CalibrationResults``: Container for all calibration output data
- ``ResidualData``: Observation residual analysis with weighted statistics and group phi
- ``SensitivityData``: Parameter sensitivity rankings
- Fit statistics: RMSE, MAE, R-squared, Nash-Sutcliffe efficiency, bias, percent bias
- Parameter identifiability analysis
- Summary report generation
- Export to CSV and PEST format

**Zone Management** (``pyiwfm.core.zones``)

- ``Zone``: Dataclass for zone definition with id, name, elements, area
- ``ZoneDefinition``: Element-to-zone mapping with query and validation
- Factory methods: ``from_subregions()``, ``from_element_list()``
- Zone CRUD operations (add, remove, rename)

**Data Aggregation** (``pyiwfm.core.aggregation``)

- ``DataAggregator``: Spatial aggregation engine with 6 methods
- ``AggregationMethod``: Enum (sum, mean, area_weighted_mean, min, max, median)
- ``aggregate_to_array()``: Expand zone values back to elements for visualization
- ``aggregate_timeseries()``: Multi-timestep aggregation
- ``create_aggregator_from_grid()``: Factory from AppGrid

**Model Query API** (``pyiwfm.core.query``)

- ``ModelQueryAPI``: High-level unified query interface for multi-scale data access
- ``get_values()``: Fetch data at any spatial scale with configurable aggregation
- ``get_timeseries()``: Retrieve temporal data for locations or zones
- ``export_to_dataframe()``, ``export_to_csv()``: Data export to Pandas and CSV
- Dynamic variable and scale discovery

**Zone File I/O** (``pyiwfm.io.zones``)

- ``read_iwfm_zone_file()``, ``write_iwfm_zone_file()``: IWFM ZBudget zone format
- ``read_geojson_zones()``, ``write_geojson_zones()``: GeoJSON with geometry
- ``read_zone_file()``, ``write_zone_file()``: Auto-detecting universal I/O
- ``auto_detect_zone_file()``: Format detection by extension and content

**Interactive Web Viewer** (``pyiwfm.visualization.webapi``)

- FastAPI backend + React SPA frontend with 4 tabs (Overview, 3D Mesh, Results Map, Budgets)
- ``ModelState`` singleton for lazy model and results loading
- vtk.js-based 3D mesh rendering with layer visibility, cross-section slicing, and z-exaggeration
- deck.gl + MapLibre Results Map with head contours and hydrograph markers
- Plotly budget charts with location/column selection
- Stream network overlay on both 3D and 2D views
- Server-side coordinate reprojection via ``pyproj``
- ``pyiwfm viewer`` CLI launcher with ``--model-dir``, ``--crs``, ``--port`` options
- Auto-detection of preprocessor and simulation files
- Graceful degradation for missing components

[0.1.0] - 2024-07-01
--------------------

Initial release of pyiwfm.

Added
~~~~~

**Core Modules**

- ``pyiwfm.core.mesh``: AppGrid, Node, Element, Face classes for mesh representation
- ``pyiwfm.core.stratigraphy``: Stratigraphy class for layer structure
- ``pyiwfm.core.timeseries``: TimeSeries and TimeSeriesCollection classes

**Component Modules**

- ``pyiwfm.components.groundwater``: AppGW, AquiferParameters, Well classes
- ``pyiwfm.components.stream``: AppStream, StrmNode, StrmReach classes
- ``pyiwfm.components.lake``: AppLake, Lake, LakeElement classes
- ``pyiwfm.components.rootzone``: RootZone, LandUse, Crop classes
- ``pyiwfm.components.connectors``: StreamGWConnector, LakeGWConnector

**I/O Modules**

- ``pyiwfm.io.ascii``: ASCII file readers and writers
- ``pyiwfm.io.binary``: Fortran binary file handlers
- ``pyiwfm.io.hdf5``: HDF5 file handlers

**Mesh Generation**

- ``pyiwfm.mesh_generation.generators``: MeshGenerator ABC and MeshResult
- ``pyiwfm.mesh_generation.constraints``: BoundaryConstraint, LineConstraint, PointConstraint, RefinementZone
- ``pyiwfm.mesh_generation.triangle_wrapper``: TriangleMeshGenerator
- ``pyiwfm.mesh_generation.gmsh_wrapper``: GmshMeshGenerator

**Visualization**

- ``pyiwfm.visualization.gis_export``: GISExporter for GeoPackage, Shapefile, GeoJSON
- ``pyiwfm.visualization.vtk_export``: VTKExporter for 2D and 3D VTK files
- ``pyiwfm.visualization.plotting``: matplotlib-based visualization functions

**Comparison**

- ``pyiwfm.comparison.differ``: ModelDiffer, MeshDiff, StratigraphyDiff
- ``pyiwfm.comparison.metrics``: ComparisonMetrics, TimeSeriesComparison, SpatialComparison
- ``pyiwfm.comparison.report``: ReportGenerator with text, JSON, and HTML output

Documentation
~~~~~~~~~~~~~

- Comprehensive Sphinx documentation with PyData theme
- User guide with installation, quickstart, and detailed guides
- Tutorials for mesh generation, visualization, and model comparison
- Full API reference with examples
