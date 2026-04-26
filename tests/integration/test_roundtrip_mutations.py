"""Integration tests for the IWFMModel mutation helpers (Phase 2.3).

These exercise the user's #1 quality bar: load a real model, mutate via
the new ergonomic helpers (Phase 2.1), save to a tmp dir, reload, and
verify the mutation survived end-to-end.

Runs against:
    * IWFM Sample Model (auto-downloaded via ``sample_model_path`` fixture)

Tests skip cleanly if the sample model is unavailable.

Why this matters: ``test_roundtrip.py`` covers load→write→load identity
for *unmodified* models. This file covers load→mutate→write→load
identity for *modified* models — proving that every mutation path the
Python API exposes actually persists through the full I/O stack.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyiwfm.core.model import IWFMModel
from pyiwfm.io import save_complete_model


@pytest.fixture
def loaded_sample_model(sample_model_path: Path) -> IWFMModel:
    """Load the IWFM Sample Model with both simulation and preprocessor files."""
    return IWFMModel.from_simulation_with_preprocessor(
        simulation_file=sample_model_path / "Simulation" / "Simulation_MAIN.IN",
        preprocessor_file=sample_model_path / "Preprocessor" / "PreProcessor_MAIN.IN",
    )


def _roundtrip_save_and_reload(
    model: IWFMModel,
    output_dir: Path,
) -> IWFMModel:
    """Save the model and reload it — used to assert mutations persist."""
    save_complete_model(model, output_dir)
    # Find the simulation file in the output dir
    sim_candidates = list(output_dir.rglob("Simulation_MAIN.IN")) or list(
        output_dir.rglob("Simulation*.in")
    )
    pp_candidates = list(output_dir.rglob("PreProcessor_MAIN.IN")) or list(
        output_dir.rglob("PreProcessor*.in")
    )
    assert sim_candidates, f"No simulation file written to {output_dir}"
    assert pp_candidates, f"No preprocessor file written to {output_dir}"
    return IWFMModel.from_simulation_with_preprocessor(
        simulation_file=sim_candidates[0],
        preprocessor_file=pp_candidates[0],
    )


@pytest.mark.integration
class TestAquiferParameterMutationRoundtrip:
    """Mutating an aquifer parameter array must persist through save/reload."""

    def test_set_kh_layer_persists(
        self,
        loaded_sample_model: IWFMModel,
        tmp_path: Path,
    ):
        model = loaded_sample_model
        if model.groundwater is None or model.groundwater.aquifer_params is None:
            pytest.skip("sample model does not expose aquifer parameters")

        n_nodes = model.groundwater.aquifer_params.n_nodes
        # Halve all Kh values in layer 1
        original_kh = model.groundwater.aquifer_params.kh
        if original_kh is None:
            pytest.skip("sample model has no Kh array")
        new_kh_layer = original_kh[:, 0] * 0.5

        model.set_aquifer_parameter("kh", layer=1, values=new_kh_layer)
        assert "groundwater" in model._dirty

        # Save + reload
        reloaded = _roundtrip_save_and_reload(model, tmp_path / "modified")

        assert reloaded.groundwater is not None
        assert reloaded.groundwater.aquifer_params is not None
        assert reloaded.groundwater.aquifer_params.kh is not None

        # Mutation survived (allow %.6g rounding the writer applies)
        np.testing.assert_allclose(
            reloaded.groundwater.aquifer_params.kh[:, 0],
            new_kh_layer,
            rtol=1e-5,
        )
        # Other layers untouched
        if model.groundwater.aquifer_params.n_layers > 1:
            np.testing.assert_allclose(
                reloaded.groundwater.aquifer_params.kh[:, 1],
                original_kh[:, 1],
                rtol=1e-5,
            )
        assert reloaded.groundwater.aquifer_params.n_nodes == n_nodes

    def test_set_single_cell_persists(
        self,
        loaded_sample_model: IWFMModel,
        tmp_path: Path,
    ):
        model = loaded_sample_model
        if model.groundwater is None or model.groundwater.aquifer_params is None:
            pytest.skip("sample model does not expose aquifer parameters")
        if model.groundwater.aquifer_params.kh is None:
            pytest.skip("no Kh array")

        # Pick a node in the middle of the mesh and bump its Kh value
        target_node = max(1, model.groundwater.aquifer_params.n_nodes // 2)
        new_value = 1.234e-3
        model.set_aquifer_parameter_at("kh", node_id=target_node, layer=1, value=new_value)

        reloaded = _roundtrip_save_and_reload(model, tmp_path / "modified")
        assert reloaded.groundwater is not None
        assert reloaded.groundwater.aquifer_params is not None
        assert reloaded.groundwater.aquifer_params.kh is not None
        assert reloaded.groundwater.aquifer_params.kh[target_node - 1, 0] == pytest.approx(
            new_value, rel=1e-5
        )


@pytest.mark.integration
class TestObservationWellMutationRoundtrip:
    """Adding and removing observation wells must persist through save/reload."""

    def test_add_observation_well_persists(
        self,
        loaded_sample_model: IWFMModel,
        tmp_path: Path,
    ):
        model = loaded_sample_model
        if model.groundwater is None:
            pytest.skip("sample model has no groundwater component")

        before = len(model.groundwater.hydrograph_locations)
        # Pick a real node from the mesh for the new observation
        target_node = next(iter(model.groundwater.n_nodes and range(1, 2)), 1)

        model.add_observation_well(
            node_id=target_node,
            layer=1,
            x=12345.0,
            y=67890.0,
            name="PYIWFM-TEST-WELL",
        )
        assert len(model.groundwater.hydrograph_locations) == before + 1

        reloaded = _roundtrip_save_and_reload(model, tmp_path / "modified")
        assert reloaded.groundwater is not None

        names = [loc.name for loc in reloaded.groundwater.hydrograph_locations]
        # The new well should be present after save/reload
        assert "PYIWFM-TEST-WELL" in names

    def test_remove_observation_well_persists(
        self,
        loaded_sample_model: IWFMModel,
        tmp_path: Path,
    ):
        model = loaded_sample_model
        if model.groundwater is None:
            pytest.skip("sample model has no groundwater component")
        if not model.groundwater.hydrograph_locations:
            pytest.skip("sample model has no hydrograph locations to remove")

        # Add a well, save, reload, remove it, save, reload — verify gone.
        model.add_observation_well(node_id=1, layer=1, x=0.0, y=0.0, name="TO-BE-REMOVED")
        intermediate = _roundtrip_save_and_reload(model, tmp_path / "added")
        # In the intermediate model, the well is present
        assert intermediate.groundwater is not None
        assert any(
            loc.name == "TO-BE-REMOVED" for loc in intermediate.groundwater.hydrograph_locations
        )

        n_removed = intermediate.remove_observation_well("TO-BE-REMOVED")
        assert n_removed == 1
        final = _roundtrip_save_and_reload(intermediate, tmp_path / "removed")
        assert final.groundwater is not None
        names = [loc.name for loc in final.groundwater.hydrograph_locations]
        assert "TO-BE-REMOVED" not in names


@pytest.mark.integration
class TestStratigraphyMutationRoundtrip:
    """``set_stratigraphy_from_thicknesses`` must produce a stratigraphy that
    survives save/reload."""

    def test_uniform_thickening_persists(
        self,
        loaded_sample_model: IWFMModel,
        tmp_path: Path,
    ):
        model = loaded_sample_model
        if model.stratigraphy is None or model.mesh is None:
            pytest.skip("sample model lacks stratigraphy or mesh")

        original_thicknesses = model.stratigraphy.top_elev - model.stratigraphy.bottom_elev
        gs = model.stratigraphy.gs_elev.copy()
        # Increase every aquifer thickness by 1 ft
        new_aquifer = original_thicknesses + 1.0
        # Aquitard thicknesses unchanged from original
        original_aquitards = model.stratigraphy.get_all_aquitard_thicknesses()

        model.set_stratigraphy_from_thicknesses(
            gs_elev=gs,
            aquitard_thicknesses=original_aquitards,
            aquifer_thicknesses=new_aquifer,
        )
        assert "stratigraphy" in model._dirty

        reloaded = _roundtrip_save_and_reload(model, tmp_path / "modified")
        assert reloaded.stratigraphy is not None

        new_actual = reloaded.stratigraphy.top_elev - reloaded.stratigraphy.bottom_elev
        # write_stratigraphy uses %.4f precision; allow ~1e-3 tolerance
        np.testing.assert_allclose(new_actual, new_aquifer, atol=1e-3)
