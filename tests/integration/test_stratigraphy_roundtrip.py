"""Integration tests for Stratigraphy aquitard helpers and file roundtrip.

Validates that the new aquitard accessors on :class:`Stratigraphy` behave
correctly against real IWFM models, and that a read -> write -> read cycle
of the stratigraphy file preserves elevations and aquitard thicknesses.

Runs against:
    * IWFM Sample Model (auto-downloaded via ``sample_model_path`` fixture)
    * C2VSimFG (requires ``C2VSIMFG_DIR`` env var)

Each test skips cleanly if the corresponding model is unavailable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyiwfm.core.model import IWFMModel
from pyiwfm.core.stratigraphy import Stratigraphy
from pyiwfm.io.mesh import read_stratigraphy, write_stratigraphy


def _find_stratigraphy_file(model_dir: Path) -> Path | None:
    """Locate the stratigraphy input file inside a model directory.

    IWFM conventionally uses ``*Strat*.dat`` under ``Preprocessor/``.
    """
    for candidate in model_dir.rglob("*[Ss]trat*.dat"):
        # Skip binary / output files
        if candidate.is_file():
            return candidate
    return None


def _check_aquitard_identity(strat: Stratigraphy) -> None:
    """aquitards + aquifers must equal gs - bottom_of_last_layer."""
    aquitard_sum = strat.get_all_aquitard_thicknesses().sum(axis=1)
    aquifer_sum = (strat.top_elev - strat.bottom_elev).sum(axis=1)
    np.testing.assert_allclose(
        aquitard_sum + aquifer_sum,
        strat.gs_elev - strat.bottom_elev[:, -1],
        atol=1e-6,
        rtol=1e-9,
    )


def _check_roundtrip(strat_file: Path, tmp_path: Path) -> None:
    """Read -> write -> read the stratigraphy file and compare."""
    original = read_stratigraphy(strat_file)

    written = tmp_path / "strat_roundtrip.dat"
    write_stratigraphy(written, original)
    reread = read_stratigraphy(written)

    assert reread.n_layers == original.n_layers
    assert reread.n_nodes == original.n_nodes
    # write_stratigraphy writes %.4f precision; allow ~1e-3 tolerance
    np.testing.assert_allclose(reread.gs_elev, original.gs_elev, atol=1e-3)
    np.testing.assert_allclose(reread.top_elev, original.top_elev, atol=1e-3)
    np.testing.assert_allclose(reread.bottom_elev, original.bottom_elev, atol=1e-3)
    np.testing.assert_allclose(
        reread.get_all_aquitard_thicknesses(),
        original.get_all_aquitard_thicknesses(),
        atol=1e-3,
    )


@pytest.mark.integration
class TestSampleModelStratigraphy:
    """Aquitard helpers and roundtrip against the IWFM Sample Model."""

    @pytest.fixture(autouse=True)
    def _setup(self, sample_model_path: Path) -> None:
        self.model = IWFMModel.from_simulation_with_preprocessor(
            simulation_file=sample_model_path / "Simulation" / "Simulation_MAIN.IN",
            preprocessor_file=sample_model_path / "Preprocessor" / "PreProcessor_MAIN.IN",
        )
        self.model_dir = sample_model_path

    def test_aquitard_helpers_match_elevation_arithmetic(self) -> None:
        strat = self.model.stratigraphy
        assert strat is not None
        _check_aquitard_identity(strat)

        # Spot-check helper vs direct arithmetic
        for k in range(strat.n_aquitards):
            via_helper = strat.get_aquitard_thickness(k)
            if k == 0:
                expected = strat.gs_elev - strat.top_elev[:, 0]
            else:
                expected = strat.bottom_elev[:, k - 1] - strat.top_elev[:, k]
            np.testing.assert_allclose(via_helper, expected)

    def test_stratigraphy_roundtrip(self, tmp_path: Path) -> None:
        strat_file = _find_stratigraphy_file(self.model_dir / "Preprocessor")
        if strat_file is None:
            pytest.skip("Stratigraphy file not found in sample model")
        _check_roundtrip(strat_file, tmp_path)


@pytest.mark.integration
class TestC2VSimFGStratigraphy:
    """Aquitard helpers and roundtrip against C2VSimFG.

    The ``C2VSIMFG_DIR`` env var may point at either the model root or the
    ``Simulation`` subdirectory; we search both.
    """

    @pytest.fixture(autouse=True)
    def _setup(self, c2vsimfg_path: Path) -> None:
        self.model_dir = c2vsimfg_path

    def test_aquitard_helpers_on_c2vsimfg_strat(self, tmp_path: Path) -> None:
        # Search both the provided dir and its parent for a stratigraphy file.
        search_roots = [self.model_dir, self.model_dir.parent]
        strat_file: Path | None = None
        for root in search_roots:
            strat_file = _find_stratigraphy_file(root)
            if strat_file is not None:
                break
        if strat_file is None:
            pytest.skip("Stratigraphy file not found under C2VSimFG directory")

        strat = read_stratigraphy(strat_file)
        _check_aquitard_identity(strat)
        _check_roundtrip(strat_file, tmp_path)
