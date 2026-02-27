"""Tests for pyiwfm.io.gw_subsidence_writer module.

Covers write_subsidence_main (v4.0 and v5.0), write_subsidence_ic,
write_gw_subsidence wrapper, _write_subsidence_params (v4.0 vs v5.0
format differences), hydrograph output section (NOUTS=0, hydtyp=0,
hydtyp=1), and parametric grid writing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pyiwfm.io.gw_subsidence import (
    ParametricSubsidenceData,
    SubsidenceConfig,
    SubsidenceHydrographSpec,
    SubsidenceNodeParams,
)
from pyiwfm.io.gw_subsidence_writer import (
    write_gw_subsidence,
    write_subsidence_ic,
    write_subsidence_main,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_node_params(
    node_id: int = 1,
    n_layers: int = 1,
    *,
    elastic: float = 0.001,
    inelastic: float = 0.01,
    thick: float = 10.0,
    thick_min: float = 1.0,
    precompact: float = 50.0,
    kv: float = 0.0,
    neq: float = 0.0,
) -> SubsidenceNodeParams:
    """Build a SubsidenceNodeParams with uniform layer values."""
    return SubsidenceNodeParams(
        node_id=node_id,
        elastic_sc=[elastic] * n_layers,
        inelastic_sc=[inelastic] * n_layers,
        interbed_thick=[thick] * n_layers,
        interbed_thick_min=[thick_min] * n_layers,
        precompact_head=[precompact] * n_layers,
        kv_sub=[kv] * n_layers,
        n_eq=[neq] * n_layers,
    )


def _make_config_v40(**overrides: object) -> SubsidenceConfig:
    """Build a minimal v4.0 SubsidenceConfig with sensible defaults."""
    defaults: dict[str, object] = {
        "version": "4.0",
        "ic_file": None,
        "tecplot_file": None,
        "final_subs_file": None,
        "output_factor": 1.0,
        "output_unit": "FEET",
        "n_hydrograph_outputs": 0,
        "hydrograph_coord_factor": 1.0,
        "hydrograph_output_file": None,
        "hydrograph_specs": [],
        "interbed_dz": 0.0,
        "n_parametric_grids": 0,
        "conversion_factors": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "node_params": [],
        "n_nodes": 0,
        "n_layers": 0,
        "ic_factor": 1.0,
        "ic_interbed_thick": None,
        "ic_precompact_head": None,
        "parametric_grids": [],
        "_raw_ic_file": "",
        "_raw_tecplot_file": "",
        "_raw_final_subs_file": "",
        "_raw_hydrograph_output_file": "",
    }
    defaults.update(overrides)
    return SubsidenceConfig(**defaults)  # type: ignore[arg-type]


def _make_config_v50(**overrides: object) -> SubsidenceConfig:
    """Build a minimal v5.0 SubsidenceConfig."""
    defaults: dict[str, object] = {
        "version": "5.0",
        "ic_file": None,
        "tecplot_file": None,
        "final_subs_file": None,
        "output_factor": 1.0,
        "output_unit": "FEET",
        "n_hydrograph_outputs": 0,
        "hydrograph_coord_factor": 1.0,
        "hydrograph_output_file": None,
        "hydrograph_specs": [],
        "interbed_dz": 5.0,
        "n_parametric_grids": 0,
        "conversion_factors": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "node_params": [],
        "n_nodes": 0,
        "n_layers": 0,
        "ic_factor": 1.0,
        "ic_interbed_thick": None,
        "ic_precompact_head": None,
        "parametric_grids": [],
        "_raw_ic_file": "",
        "_raw_tecplot_file": "",
        "_raw_final_subs_file": "",
        "_raw_hydrograph_output_file": "",
    }
    defaults.update(overrides)
    return SubsidenceConfig(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# write_subsidence_main — v4.0
# ---------------------------------------------------------------------------


class TestWriteSubsidenceMainV40:
    def test_minimal_config(self, tmp_path: Path) -> None:
        """Empty v4.0 config writes header, version, and scalar fields."""
        cfg = _make_config_v40()
        result = write_subsidence_main(cfg, tmp_path / "subs.dat")
        assert result.exists()
        text = result.read_text()
        assert "C  IWFM Subsidence Parameter File" in text
        assert "#4.0" in text
        assert "FEET" in text
        assert "/ NGroup" in text

    def test_returns_path_object(self, tmp_path: Path) -> None:
        cfg = _make_config_v40()
        result = write_subsidence_main(cfg, str(tmp_path / "subs.dat"))
        assert isinstance(result, Path)

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        cfg = _make_config_v40()
        deep = tmp_path / "a" / "b" / "subs.dat"
        result = write_subsidence_main(cfg, deep)
        assert result.exists()

    def test_raw_paths_written_when_present(self, tmp_path: Path) -> None:
        cfg = _make_config_v40(
            _raw_ic_file="..\\IC\\ic.dat",
            _raw_tecplot_file="..\\OUT\\tec.dat",
            _raw_final_subs_file="..\\OUT\\final.dat",
        )
        result = write_subsidence_main(cfg, tmp_path / "subs.dat")
        text = result.read_text()
        assert "..\\IC\\ic.dat" in text
        assert "..\\OUT\\tec.dat" in text
        assert "..\\OUT\\final.dat" in text

    def test_fallback_to_path_objects(self, tmp_path: Path) -> None:
        """When raw paths are empty, str(path) is used."""
        cfg = _make_config_v40(
            ic_file=Path("ic.dat"),
            tecplot_file=Path("tec.dat"),
            final_subs_file=Path("final.dat"),
        )
        result = write_subsidence_main(cfg, tmp_path / "subs.dat")
        text = result.read_text()
        assert "ic.dat" in text
        assert "tec.dat" in text
        assert "final.dat" in text

    def test_conversion_factors_written(self, tmp_path: Path) -> None:
        factors = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        cfg = _make_config_v40(conversion_factors=factors)
        result = write_subsidence_main(cfg, tmp_path / "subs.dat")
        text = result.read_text()
        for f in factors:
            assert f"{f:>12.6f}" in text

    def test_no_interbed_dz_for_v40(self, tmp_path: Path) -> None:
        """v4.0 should NOT write the Interbed DZ line."""
        cfg = _make_config_v40(interbed_dz=10.0)
        result = write_subsidence_main(cfg, tmp_path / "subs.dat")
        text = result.read_text()
        assert "Interbed DZ" not in text

    def test_with_node_params(self, tmp_path: Path) -> None:
        """v4.0 with direct node params writes 5 columns per layer."""
        params = [
            _make_node_params(node_id=1, n_layers=2),
            _make_node_params(node_id=2, n_layers=2),
        ]
        cfg = _make_config_v40(
            node_params=params,
            n_nodes=2,
            n_layers=2,
            n_parametric_grids=0,
        )
        result = write_subsidence_main(cfg, tmp_path / "subs.dat")
        text = result.read_text()
        # Node IDs should appear
        lines = text.splitlines()
        data_lines = [
            ln
            for ln in lines
            if ln.strip() and not ln.startswith("C") and "/" not in ln and not ln.startswith("#")
        ]
        # 2 nodes x 2 layers = 4 data lines
        assert len(data_lines) >= 4


# ---------------------------------------------------------------------------
# write_subsidence_main — v5.0
# ---------------------------------------------------------------------------


class TestWriteSubsidenceMainV50:
    def test_minimal_config(self, tmp_path: Path) -> None:
        cfg = _make_config_v50()
        result = write_subsidence_main(cfg, tmp_path / "subs.dat")
        text = result.read_text()
        assert "#5.0" in text
        assert "/ Interbed DZ" in text

    def test_interbed_dz_written(self, tmp_path: Path) -> None:
        cfg = _make_config_v50(interbed_dz=7.5)
        result = write_subsidence_main(cfg, tmp_path / "subs.dat")
        text = result.read_text()
        assert "7.5" in text
        assert "Interbed DZ" in text

    def test_v50_node_params_include_kv_neq(self, tmp_path: Path) -> None:
        """v5.0 writes Kv and NEQ columns after the base 5 params."""
        params = [
            _make_node_params(
                node_id=1,
                n_layers=1,
                elastic=0.001,
                inelastic=0.01,
                thick=10.0,
                thick_min=1.0,
                precompact=50.0,
                kv=0.5,
                neq=3.0,
            ),
        ]
        cfg = _make_config_v50(
            node_params=params,
            n_nodes=1,
            n_layers=1,
            n_parametric_grids=0,
        )
        result = write_subsidence_main(cfg, tmp_path / "subs.dat")
        text = result.read_text()
        # Kv and NEQ should appear on the data line
        lines = text.splitlines()
        data_lines = [
            ln
            for ln in lines
            if ln.strip() and not ln.startswith("C") and "/" not in ln and not ln.startswith("#")
        ]
        assert len(data_lines) >= 1
        # The data line should have at least 8 tokens: node_id + 5 params + kv + neq
        parts = data_lines[-1].split()
        assert len(parts) >= 8

    def test_v50_conversion_factors_seven(self, tmp_path: Path) -> None:
        factors = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0]
        cfg = _make_config_v50(conversion_factors=factors)
        result = write_subsidence_main(cfg, tmp_path / "subs.dat")
        text = result.read_text()
        assert "2.000000" in text


# ---------------------------------------------------------------------------
# _write_subsidence_params — conversion factor division
# ---------------------------------------------------------------------------


class TestWriteSubsidenceParams:
    def test_v40_factor_division(self, tmp_path: Path) -> None:
        """Parameters are divided by conversion factors when writing."""
        params = [
            _make_node_params(
                node_id=1,
                n_layers=1,
                elastic=2.0,
                inelastic=6.0,
                thick=12.0,
                thick_min=4.0,
                precompact=10.0,
            ),
        ]
        # factors: [FX, elastic, inelastic, thick, thick_min, precompact]
        factors = [1.0, 2.0, 3.0, 4.0, 2.0, 5.0]
        cfg = _make_config_v40(
            node_params=params,
            n_nodes=1,
            n_layers=1,
            n_parametric_grids=0,
            conversion_factors=factors,
        )
        result = write_subsidence_main(cfg, tmp_path / "subs.dat")
        text = result.read_text()
        # elastic: 2.0 / 2.0 = 1.0
        assert "1.000000" in text
        # inelastic: 6.0 / 3.0 = 2.0
        assert "2.000000" in text
        # thick: 12.0 / 4.0 = 3.0
        assert "3.000000" in text
        # thick_min: 4.0 / 2.0 = 2.0 (already covered)
        # precompact: 10.0 / 5.0 = 2.0 (already covered)

    def test_v50_factor_division_with_kv(self, tmp_path: Path) -> None:
        """v5.0 divides kv by factors[6]."""
        params = [
            _make_node_params(
                node_id=1,
                n_layers=1,
                elastic=1.0,
                inelastic=1.0,
                thick=1.0,
                thick_min=1.0,
                precompact=1.0,
                kv=6.0,
                neq=2.0,
            ),
        ]
        factors = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.0]
        cfg = _make_config_v50(
            node_params=params,
            n_nodes=1,
            n_layers=1,
            n_parametric_grids=0,
            conversion_factors=factors,
        )
        result = write_subsidence_main(cfg, tmp_path / "subs.dat")
        text = result.read_text()
        # kv: 6.0 / 3.0 = 2.0
        assert "2.000000" in text
        # neq is NOT divided by a factor; it should appear as 2.0
        assert "2.0" in text

    def test_multi_layer_first_row_has_node_id(self, tmp_path: Path) -> None:
        """First layer row includes node ID; second layer row does not."""
        params = [
            _make_node_params(node_id=42, n_layers=2, elastic=0.5),
        ]
        cfg = _make_config_v40(
            node_params=params,
            n_nodes=1,
            n_layers=2,
            n_parametric_grids=0,
        )
        result = write_subsidence_main(cfg, tmp_path / "subs.dat")
        text = result.read_text()
        assert "42" in text
        # Find data lines (non-comment, non-descriptor)
        lines = text.splitlines()
        data_lines = [
            ln
            for ln in lines
            if ln.strip() and not ln.startswith("C") and "/" not in ln and not ln.startswith("#")
        ]
        # First data line has node id, second does not
        assert len(data_lines) >= 2
        first_parts = data_lines[-2].split()
        second_parts = data_lines[-1].split()
        # First row has node_id so more tokens
        assert int(first_parts[0]) == 42
        # Second row has only param columns (no integer node id as first)
        assert len(second_parts) == 5  # 5 params for v4.0


# ---------------------------------------------------------------------------
# Hydrograph section
# ---------------------------------------------------------------------------


class TestHydrographSection:
    def test_nouts_zero_skips_hyd_section(self, tmp_path: Path) -> None:
        """When n_hydrograph_outputs=0, FACTXY / SUBHYDOUTFL are skipped."""
        cfg = _make_config_v40(n_hydrograph_outputs=0)
        result = write_subsidence_main(cfg, tmp_path / "subs.dat")
        text = result.read_text()
        assert "/ NOUTS" in text
        assert "FACTXY" not in text
        assert "SUBHYDOUTFL" not in text

    def test_hydtyp_zero_xy_format(self, tmp_path: Path) -> None:
        """hydtyp=0 writes x-y coordinate format with FACTXY reversal."""
        spec = SubsidenceHydrographSpec(id=1, hydtyp=0, layer=1, x=100.0, y=200.0, name="WELL_A")
        cfg = _make_config_v40(
            n_hydrograph_outputs=1,
            hydrograph_coord_factor=2.0,
            hydrograph_output_file=Path("hyd_out.dat"),
            hydrograph_specs=[spec],
        )
        result = write_subsidence_main(cfg, tmp_path / "subs.dat")
        text = result.read_text()
        assert "/ FACTXY" in text
        assert "/ SUBHYDOUTFL" in text or "hyd_out.dat" in text
        # x = 100.0 / 2.0 = 50.0; y = 200.0 / 2.0 = 100.0
        assert "50.0" in text
        assert "100.0" in text
        assert "WELL_A" in text

    def test_hydtyp_one_node_format(self, tmp_path: Path) -> None:
        """hydtyp=1 writes node number format."""
        spec = SubsidenceHydrographSpec(id=2, hydtyp=1, layer=3, x=99.0, y=0.0, name="NODE_B")
        cfg = _make_config_v40(
            n_hydrograph_outputs=1,
            hydrograph_coord_factor=1.0,
            hydrograph_output_file=Path("hyd_out.dat"),
            hydrograph_specs=[spec],
        )
        result = write_subsidence_main(cfg, tmp_path / "subs.dat")
        text = result.read_text()
        # Node ID is int(spec.x) = 99
        assert "99" in text
        assert "NODE_B" in text

    def test_hydtyp_zero_factor_zero_fallback(self, tmp_path: Path) -> None:
        """When factor is 0, x and y are written as-is."""
        spec = SubsidenceHydrographSpec(id=1, hydtyp=0, layer=1, x=300.0, y=400.0)
        cfg = _make_config_v40(
            n_hydrograph_outputs=1,
            hydrograph_coord_factor=0.0,
            hydrograph_output_file=Path("hyd.dat"),
            hydrograph_specs=[spec],
        )
        result = write_subsidence_main(cfg, tmp_path / "subs.dat")
        text = result.read_text()
        assert "300.0" in text
        assert "400.0" in text

    def test_multiple_hydrograph_specs(self, tmp_path: Path) -> None:
        specs = [
            SubsidenceHydrographSpec(id=1, hydtyp=0, layer=1, x=10.0, y=20.0),
            SubsidenceHydrographSpec(id=2, hydtyp=1, layer=2, x=55.0),
        ]
        cfg = _make_config_v40(
            n_hydrograph_outputs=2,
            hydrograph_coord_factor=1.0,
            hydrograph_output_file=Path("hyd.dat"),
            hydrograph_specs=specs,
        )
        result = write_subsidence_main(cfg, tmp_path / "subs.dat")
        text = result.read_text()
        assert "10.0" in text
        assert "20.0" in text
        assert "55" in text

    def test_raw_hyd_output_path(self, tmp_path: Path) -> None:
        """Raw hydrograph output path is preferred over Path object."""
        spec = SubsidenceHydrographSpec(id=1, hydtyp=1, layer=1, x=1.0)
        cfg = _make_config_v40(
            n_hydrograph_outputs=1,
            hydrograph_coord_factor=1.0,
            hydrograph_output_file=Path("hyd.dat"),
            _raw_hydrograph_output_file="..\\Results\\hyd.dat",
            hydrograph_specs=[spec],
        )
        result = write_subsidence_main(cfg, tmp_path / "subs.dat")
        text = result.read_text()
        assert "..\\Results\\hyd.dat" in text


# ---------------------------------------------------------------------------
# Parametric grids
# ---------------------------------------------------------------------------


class TestParametricGrids:
    def test_parametric_grid_writing(self, tmp_path: Path) -> None:
        """Single parametric grid with elements and node data."""
        node_coords = np.array([[100.0, 200.0], [300.0, 400.0], [500.0, 600.0]])
        node_values = np.array(
            [
                [[1.0, 2.0, 3.0, 4.0, 5.0]],
                [[6.0, 7.0, 8.0, 9.0, 10.0]],
                [[11.0, 12.0, 13.0, 14.0, 15.0]],
            ]
        )  # shape (3, 1, 5)
        grid = ParametricSubsidenceData(
            node_range_str="1-100",
            n_nodes=3,
            n_elements=1,
            elements=[(0, 1, 2)],
            node_coords=node_coords,
            node_values=node_values,
        )
        cfg = _make_config_v40(
            n_parametric_grids=1,
            parametric_grids=[grid],
            conversion_factors=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        )
        result = write_subsidence_main(cfg, tmp_path / "subs.dat")
        text = result.read_text()
        assert "1-100" in text
        assert "/ NDP" in text
        assert "/ NEP" in text
        # Node coordinates (reversed by FX=1.0)
        assert "100.0" in text
        assert "200.0" in text

    def test_parametric_grid_element_indices_one_based(self, tmp_path: Path) -> None:
        """Element vertex indices are written as 1-based (v + 1)."""
        node_coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        node_values = np.ones((4, 1, 5))
        grid = ParametricSubsidenceData(
            node_range_str="1-50",
            n_nodes=4,
            n_elements=2,
            elements=[(0, 1, 2), (1, 2, 3)],
            node_coords=node_coords,
            node_values=node_values,
        )
        cfg = _make_config_v40(
            n_parametric_grids=1,
            parametric_grids=[grid],
        )
        result = write_subsidence_main(cfg, tmp_path / "subs.dat")
        text = result.read_text()
        lines = text.splitlines()
        # Find element definition lines (after NEP, before node data)
        # Elements are 0-based in data, written as v+1 -> 1-based
        # First elem: (0,1,2) -> "1  2  3"
        # At least one element line with 1-based indices
        found = False
        for ln in lines:
            parts = ln.split()
            if len(parts) == 3:
                try:
                    vals = [int(p) for p in parts]
                    if vals == [1, 2, 3]:
                        found = True
                        break
                except ValueError:
                    pass
        assert found

    def test_parametric_grid_multi_layer_continuation(self, tmp_path: Path) -> None:
        """Multi-layer parametric grid writes continuation lines."""
        node_coords = np.array([[10.0, 20.0]])
        node_values = np.array(
            [
                [[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]],
            ]
        )  # shape (1, 2, 5)
        grid = ParametricSubsidenceData(
            node_range_str="1-10",
            n_nodes=1,
            n_elements=0,
            elements=[],
            node_coords=node_coords,
            node_values=node_values,
        )
        cfg = _make_config_v40(
            n_parametric_grids=1,
            parametric_grids=[grid],
        )
        result = write_subsidence_main(cfg, tmp_path / "subs.dat")
        text = result.read_text()
        # Should have a continuation line for layer 2 with only params
        lines = text.splitlines()
        # Count non-empty non-comment lines after "/ NEP"
        in_node_section = False
        data_line_count = 0
        for ln in lines:
            stripped = ln.strip()
            if "/ NEP" in ln:
                in_node_section = True
                continue
            if in_node_section and stripped and not stripped.startswith("C"):
                data_line_count += 1
        # 1 node x 2 layers = 2 lines
        assert data_line_count == 2


# ---------------------------------------------------------------------------
# write_subsidence_ic
# ---------------------------------------------------------------------------


class TestWriteSubsidenceIC:
    def test_basic_ic_file(self, tmp_path: Path) -> None:
        ic_thick = np.array([[10.0, 20.0], [30.0, 40.0]])
        ic_head = np.array([[50.0, 60.0], [70.0, 80.0]])
        cfg = _make_config_v40(
            ic_factor=1.0,
            ic_interbed_thick=ic_thick,
            ic_precompact_head=ic_head,
        )
        result = write_subsidence_ic(cfg, tmp_path / "ic.dat")
        assert result.exists()
        text = result.read_text()
        assert "C  IWFM Subsidence Initial Conditions" in text
        assert "/ IC conversion factor" in text
        # Node 1: 10, 20, 50, 60
        assert "10.0000" in text
        assert "20.0000" in text
        assert "50.0000" in text
        assert "60.0000" in text
        # Node 2: 30, 40, 70, 80
        assert "30.0000" in text
        assert "40.0000" in text
        assert "70.0000" in text
        assert "80.0000" in text

    def test_ic_factor_division(self, tmp_path: Path) -> None:
        """IC values are divided by ic_factor before writing."""
        ic_thick = np.array([[20.0]])
        ic_head = np.array([[100.0]])
        cfg = _make_config_v40(
            ic_factor=2.0,
            ic_interbed_thick=ic_thick,
            ic_precompact_head=ic_head,
        )
        result = write_subsidence_ic(cfg, tmp_path / "ic.dat")
        text = result.read_text()
        # 20.0 / 2.0 = 10.0; 100.0 / 2.0 = 50.0
        assert "10.0000" in text
        assert "50.0000" in text

    def test_ic_factor_zero_fallback(self, tmp_path: Path) -> None:
        """When ic_factor=0, fallback to 1.0 (no division by zero)."""
        ic_thick = np.array([[7.0]])
        ic_head = np.array([[13.0]])
        cfg = _make_config_v40(
            ic_factor=0.0,
            ic_interbed_thick=ic_thick,
            ic_precompact_head=ic_head,
        )
        result = write_subsidence_ic(cfg, tmp_path / "ic.dat")
        text = result.read_text()
        # 0.0 factor -> ic_factor set to 1.0 in writer, so raw values used
        assert "7.0000" in text
        assert "13.0000" in text

    def test_node_ids_one_based(self, tmp_path: Path) -> None:
        """Node IDs are written as 1-based (i+1)."""
        ic_thick = np.array([[1.0], [2.0], [3.0]])
        ic_head = np.array([[4.0], [5.0], [6.0]])
        cfg = _make_config_v40(
            ic_factor=1.0,
            ic_interbed_thick=ic_thick,
            ic_precompact_head=ic_head,
        )
        result = write_subsidence_ic(cfg, tmp_path / "ic.dat")
        text = result.read_text()
        lines = [
            ln
            for ln in text.splitlines()
            if ln.strip() and not ln.startswith("C") and "/" not in ln
        ]
        assert len(lines) == 3
        # First column should be 1, 2, 3
        for i, ln in enumerate(lines):
            parts = ln.split()
            assert int(parts[0]) == i + 1

    def test_returns_path(self, tmp_path: Path) -> None:
        ic_thick = np.array([[1.0]])
        ic_head = np.array([[2.0]])
        cfg = _make_config_v40(
            ic_interbed_thick=ic_thick,
            ic_precompact_head=ic_head,
        )
        result = write_subsidence_ic(cfg, str(tmp_path / "ic.dat"))
        assert isinstance(result, Path)

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        ic_thick = np.array([[1.0]])
        ic_head = np.array([[2.0]])
        cfg = _make_config_v40(
            ic_interbed_thick=ic_thick,
            ic_precompact_head=ic_head,
        )
        result = write_subsidence_ic(cfg, tmp_path / "deep" / "ic.dat")
        assert result.exists()

    def test_empty_ic_data_writes_header_only(self, tmp_path: Path) -> None:
        """When IC arrays are None, only the header is written."""
        cfg = _make_config_v40(
            ic_interbed_thick=None,
            ic_precompact_head=None,
        )
        result = write_subsidence_ic(cfg, tmp_path / "ic.dat")
        text = result.read_text()
        lines = text.strip().splitlines()
        # Only comment + factor line
        assert len(lines) == 2


# ---------------------------------------------------------------------------
# write_gw_subsidence (wrapper)
# ---------------------------------------------------------------------------


class TestWriteGwSubsidence:
    def test_main_only(self, tmp_path: Path) -> None:
        """When ic_filepath is None, only main file is written."""
        cfg = _make_config_v40()
        result = write_gw_subsidence(cfg, tmp_path / "subs.dat")
        assert result.exists()
        assert not (tmp_path / "ic.dat").exists()

    def test_main_and_ic(self, tmp_path: Path) -> None:
        """When ic_filepath is given and IC data is present, both files are written."""
        ic_thick = np.array([[5.0]])
        ic_head = np.array([[10.0]])
        cfg = _make_config_v40(
            ic_interbed_thick=ic_thick,
            ic_precompact_head=ic_head,
        )
        main_path = tmp_path / "subs.dat"
        ic_path = tmp_path / "ic.dat"
        result = write_gw_subsidence(cfg, main_path, ic_filepath=ic_path)
        assert result == main_path
        assert main_path.exists()
        assert ic_path.exists()

    def test_ic_not_written_when_data_missing(self, tmp_path: Path) -> None:
        """When IC data is None, IC file is NOT written even if path given."""
        cfg = _make_config_v40(
            ic_interbed_thick=None,
            ic_precompact_head=None,
        )
        ic_path = tmp_path / "ic.dat"
        write_gw_subsidence(cfg, tmp_path / "subs.dat", ic_filepath=ic_path)
        assert not ic_path.exists()

    def test_ic_not_written_when_only_thick_present(self, tmp_path: Path) -> None:
        """Both ic_interbed_thick and ic_precompact_head must be present."""
        cfg = _make_config_v40(
            ic_interbed_thick=np.array([[1.0]]),
            ic_precompact_head=None,
        )
        ic_path = tmp_path / "ic.dat"
        write_gw_subsidence(cfg, tmp_path / "subs.dat", ic_filepath=ic_path)
        assert not ic_path.exists()

    def test_returns_main_path(self, tmp_path: Path) -> None:
        cfg = _make_config_v40()
        result = write_gw_subsidence(cfg, tmp_path / "subs.dat")
        assert isinstance(result, Path)
        assert result.name == "subs.dat"
