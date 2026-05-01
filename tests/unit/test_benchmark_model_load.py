"""Benchmarks for end-to-end model load time.

Tracks ``IWFMModel.from_preprocessor`` against the bundled small_model
fixture, so CI can flag regressions in the most user-visible code path
(loading a model is what every workflow starts with).

Two benchmarks ship:

- ``test_benchmark_load_small_model`` — uses the fixture in
  ``tests/fixtures/small_model/`` (~ a dozen nodes, a handful of
  elements). Always runs in CI; sub-millisecond expected.

- ``test_benchmark_load_c2vsimfg`` — gated on the ``C2VSIMFG_DIR``
  environment variable so it doesn't break CI for users who don't
  have the model. Loads the C2VSimFG model via
  ``from_simulation_with_preprocessor`` (the more realistic surface;
  full simulation directory + preprocessor + components). Skipped
  cleanly when the env var is unset.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from pyiwfm.core.model import IWFMModel


@pytest.fixture
def small_preprocessor_in(small_model_path: Path) -> Path:
    return small_model_path / "Preprocessor" / "Preprocessor.in"


@pytest.mark.benchmark(group="model-load")
def test_benchmark_load_small_model(benchmark, small_preprocessor_in: Path) -> None:
    """Benchmark loading the bundled small_model fixture.

    This is the only model-load benchmark that always runs in CI.
    The fixture is tiny (~10 nodes, ~4 elements) so the benchmark
    measures parser overhead, not throughput. A regression here
    means parsing got slower per-line, which is what we want to
    catch.
    """
    model = benchmark(IWFMModel.from_preprocessor, small_preprocessor_in)
    assert model.n_nodes > 0


@pytest.mark.integration
@pytest.mark.benchmark(group="model-load")
@pytest.mark.skipif(
    not os.environ.get("C2VSIMFG_DIR"),
    reason="C2VSIMFG_DIR not set; install the C2VSimFG model and export the env var to run",
)
def test_benchmark_load_c2vsimfg(benchmark) -> None:
    """Benchmark loading C2VSimFG end-to-end.

    The realistic perf metric — typical C2VSimFG load is 20-60 seconds
    on first parse. A regression that doubles this would be the kind
    of thing a benchmark like this is meant to catch. Skipped on CI
    by default because the model is ~5GB and isn't bundled.
    """
    model_dir = Path(os.environ["C2VSIMFG_DIR"])
    sim_main = next(model_dir.glob("**/C2VSimFG.in"), None)
    if sim_main is None:
        pytest.skip("C2VSimFG.in not found under C2VSIMFG_DIR")

    model = benchmark(IWFMModel.from_simulation_with_preprocessor, sim_main)
    assert model.n_nodes > 0
