"""Tests for unsaturated zone and small watershed IO and component classes.

Covers:
- io/unsaturated_zone.py: UnsatZoneMainReader, data classes, read function
- io/small_watershed.py: SmallWatershedMainReader, data classes, read function
- components/unsaturated_zone.py: AppUnsatZone, UnsatZoneElement, UnsatZoneLayer
- components/small_watershed.py: AppSmallWatershed, WatershedUnit, WatershedGWNode
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyiwfm.io.unsaturated_zone import (
    UnsatZoneMainReader,
    UnsatZoneMainConfig,
    UnsatZoneElementData,
    read_unsaturated_zone_main,
    _is_comment_line as uz_is_comment,
    _strip_comment as uz_parse_value,
)
from pyiwfm.io.small_watershed import (
    SmallWatershedMainReader,
    SmallWatershedMainConfig,
    WatershedSpec,
    WatershedGWNode as IOWatershedGWNode,
    WatershedRootZoneParams,
    WatershedAquiferParams,
    read_small_watershed_main,
)
from pyiwfm.components.unsaturated_zone import (
    AppUnsatZone,
    UnsatZoneElement,
    UnsatZoneLayer,
)
from pyiwfm.components.small_watershed import (
    AppSmallWatershed,
    WatershedUnit,
    WatershedGWNode as CompWatershedGWNode,
)
from pyiwfm.core.exceptions import ComponentError, FileFormatError


# =============================================================================
# IO helpers tests
# =============================================================================


class TestUnsatZoneIOHelpers:
    """Tests for module-level helpers in io/unsaturated_zone."""

    def test_is_comment_empty(self) -> None:
        assert uz_is_comment("") is True

    def test_is_comment_c(self) -> None:
        assert uz_is_comment("C comment") is True

    def test_is_comment_data(self) -> None:
        assert uz_is_comment("    42") is False

    def test_parse_value_with_slash(self) -> None:
        val, _ = uz_parse_value("    1.0    / factor")
        assert val == "1.0"

    def test_parse_value_plain(self) -> None:
        val, desc = uz_parse_value("   hello")
        assert val == "hello"
        assert desc == ""


# =============================================================================
# UnsatZoneMainConfig data class tests
# =============================================================================


class TestUnsatZoneMainConfig:
    """Tests for UnsatZoneMainConfig defaults and attributes."""

    def test_defaults(self) -> None:
        config = UnsatZoneMainConfig()
        assert config.version == ""
        assert config.n_layers == 0
        assert config.solver_tolerance == 1e-8
        assert config.max_iterations == 2000
        assert config.budget_file is None
        assert config.zbudget_file is None
        assert config.final_results_file is None
        assert config.n_parametric_grids == 0
        assert config.coord_factor == 1.0
        assert config.thickness_factor == 1.0
        assert config.hyd_cond_factor == 1.0
        assert config.time_unit == ""
        assert config.element_data == []
        assert config.initial_soil_moisture == {}


class TestUnsatZoneElementData:
    """Tests for UnsatZoneElementData."""

    def test_constructor(self) -> None:
        ed = UnsatZoneElementData(
            element_id=1,
            thickness_max=np.array([5.0, 10.0]),
            total_porosity=np.array([0.3, 0.35]),
            lambda_param=np.array([0.5, 0.6]),
            hyd_cond=np.array([1e-5, 2e-5]),
            kunsat_method=np.array([1, 1], dtype=np.int32),
        )
        assert ed.element_id == 1
        assert len(ed.thickness_max) == 2


# =============================================================================
# UnsatZoneMainReader tests
# =============================================================================


class TestUnsatZoneMainReader:
    """Tests for UnsatZoneMainReader."""

    def _write(self, path: Path, content: str) -> Path:
        filepath = path / "uzf_main.dat"
        filepath.write_text(content)
        return filepath

    def test_read_disabled(self, tmp_path: Path) -> None:
        """Zero layers means unsaturated zone is disabled."""
        content = (
            "C Unsaturated Zone Main File\n"
            "#4.0\n"
            "C *** DO NOT DELETE ABOVE LINE ***\n"
            "    0                              / NLayers\n"
        )
        filepath = self._write(tmp_path, content)
        config = UnsatZoneMainReader().read(filepath)

        assert config.version == "4.0"
        assert config.n_layers == 0

    def test_read_basic_config(self, tmp_path: Path) -> None:
        """Read basic unsaturated zone configuration without element data."""
        content = (
            "#4.0\n"
            "C Main file\n"
            "    2                              / NLayers\n"
            "    1e-6                           / Tolerance\n"
            "    1000                           / MaxIter\n"
            "    budget.hdf                     / BudgetFile\n"
            "    zbudget.hdf                    / ZBudgetFile\n"
            "    final.dat                      / FinalResults\n"
            "    1                              / NGroup (parametric)\n"
            "    1.0  1.0  1.0                 / Factors\n"
            "    1DAY                           / TimeUnit\n"
        )
        filepath = self._write(tmp_path, content)
        config = UnsatZoneMainReader().read(filepath)

        assert config.version == "4.0"
        assert config.n_layers == 2
        assert config.solver_tolerance == pytest.approx(1e-6)
        assert config.max_iterations == 1000
        assert config.budget_file is not None
        assert config.budget_file.name == "budget.hdf"
        assert config.n_parametric_grids == 1

    def test_read_with_element_data_single_layer(self, tmp_path: Path) -> None:
        """Read with direct element data (NGROUP=0), single layer."""
        content = (
            "#4.0\n"
            "    1                              / NLayers\n"
            "    1e-8                           / Tolerance\n"
            "    2000                           / MaxIter\n"
            "                                   / BudgetFile\n"
            "                                   / ZBudgetFile\n"
            "                                   / FinalResults\n"
            "    0                              / NGroup\n"
            "    1.0  1.0  1.0                 / Factors\n"
            "    1DAY                           / TimeUnit\n"
            "    1    5.0    0.35    0.5    1e-5    1\n"
            "    2    6.0    0.30    0.4    2e-5    1\n"
            "C End of element data - initial conditions\n"
            "    0    0.20\n"
        )
        filepath = self._write(tmp_path, content)
        config = UnsatZoneMainReader().read(filepath)

        assert config.n_layers == 1
        assert config.n_parametric_grids == 0
        assert len(config.element_data) == 2
        assert config.element_data[0].element_id == 1
        assert config.element_data[0].thickness_max[0] == pytest.approx(5.0)
        assert config.element_data[1].total_porosity[0] == pytest.approx(0.30)

    def test_read_initial_conditions_uniform(self, tmp_path: Path) -> None:
        """Read initial conditions with element ID=0 (uniform)."""
        content = (
            "#4.0\n"
            "    1                              / NLayers\n"
            "    1e-8\n"
            "    2000\n"
            "                                   / BudgetFile\n"
            "                                   / ZBudgetFile\n"
            "                                   / FinalResults\n"
            "    0                              / NGroup\n"
            "    1.0  1.0  1.0\n"
            "    1DAY\n"
            "    1    5.0    0.35    0.5    1e-5    1\n"
            "C IC section\n"
            "    0    0.25\n"
        )
        filepath = self._write(tmp_path, content)
        config = UnsatZoneMainReader().read(filepath)

        assert 0 in config.initial_soil_moisture
        assert config.initial_soil_moisture[0][0] == pytest.approx(0.25)

    def test_read_conversion_factors(self, tmp_path: Path) -> None:
        """Verify conversion factors are parsed correctly."""
        content = (
            "#4.0\n"
            "    1                              / NLayers\n"
            "    1e-8\n"
            "    2000\n"
            "                                   / BudgetFile\n"
            "                                   / ZBudgetFile\n"
            "                                   / FinalResults\n"
            "    0                              / NGroup\n"
            "    2.0  3.0  4.0                 / Factors\n"
            "    1MON                           / TimeUnit\n"
        )
        filepath = self._write(tmp_path, content)
        config = UnsatZoneMainReader().read(filepath)

        assert config.coord_factor == pytest.approx(2.0)
        assert config.thickness_factor == pytest.approx(3.0)
        assert config.hyd_cond_factor == pytest.approx(4.0)
        assert config.time_unit == "1MON"

    def test_convenience_function(self, tmp_path: Path) -> None:
        """Test read_unsaturated_zone_main convenience function."""
        content = (
            "#4.0\n"
            "    0                              / NLayers\n"
        )
        filepath = self._write(tmp_path, content)
        config = read_unsaturated_zone_main(filepath)
        assert config.n_layers == 0

    def test_resolve_relative_path(self, tmp_path: Path) -> None:
        from pyiwfm.io.iwfm_reader import resolve_path

        result = resolve_path(tmp_path, "subdir/file.dat")
        assert result == tmp_path / "subdir" / "file.dat"

    def test_resolve_absolute_path(self, tmp_path: Path) -> None:
        from pyiwfm.io.iwfm_reader import resolve_path

        abs_path = str(tmp_path / "file.dat")
        result = resolve_path(Path("/other"), abs_path)
        assert result == Path(abs_path)


# =============================================================================
# SmallWatershedMainConfig data class tests
# =============================================================================


class TestSmallWatershedMainConfig:
    """Tests for SmallWatershedMainConfig defaults."""

    def test_defaults(self) -> None:
        config = SmallWatershedMainConfig()
        assert config.version == ""
        assert config.n_watersheds == 0
        assert config.budget_output_file is None
        assert config.final_results_file is None
        assert config.area_factor == 1.0
        assert config.flow_factor == 1.0
        assert config.flow_time_unit == ""
        assert config.watershed_specs == []
        assert config.rootzone_params == []
        assert config.aquifer_params == []

    def test_io_watershed_gw_node_defaults(self) -> None:
        gn = IOWatershedGWNode()
        assert gn.gw_node_id == 0
        assert gn.max_perc_rate == 0.0
        assert gn.is_baseflow is False
        assert gn.layer == 0

    def test_watershed_spec_defaults(self) -> None:
        ws = WatershedSpec()
        assert ws.id == 0
        assert ws.area == 0.0
        assert ws.dest_stream_node == 0
        assert ws.gw_nodes == []

    def test_rootzone_params_defaults(self) -> None:
        rz = WatershedRootZoneParams()
        assert rz.id == 0
        assert rz.curve_number == 0.0
        assert rz.hydraulic_cond == 0.0

    def test_aquifer_params_defaults(self) -> None:
        aq = WatershedAquiferParams()
        assert aq.id == 0
        assert aq.gw_threshold == 0.0
        assert aq.baseflow_coeff == 0.0


# =============================================================================
# SmallWatershedMainReader tests
# =============================================================================


class TestSmallWatershedMainReader:
    """Tests for SmallWatershedMainReader."""

    def _write(self, path: Path, content: str) -> Path:
        filepath = path / "sw_main.dat"
        filepath.write_text(content)
        return filepath

    def test_read_zero_watersheds(self, tmp_path: Path) -> None:
        """Read file with zero watersheds."""
        content = (
            "#4.0\n"
            "C Main file\n"
            "    budget.hdf                     / BudgetFile\n"
            "    final.dat                      / FinalResults\n"
            "    0                              / NWatersheds\n"
        )
        filepath = self._write(tmp_path, content)
        config = SmallWatershedMainReader().read(filepath)

        assert config.version == "4.0"
        assert config.n_watersheds == 0

    def test_read_complete_watershed(self, tmp_path: Path) -> None:
        """Read complete watershed with all sections."""
        content = (
            "#4.0\n"
            "    budget.hdf                     / BudgetFile\n"
            "    final.dat                      / FinalResults\n"
            "    1                              / NWatersheds\n"
            "C Geospatial data\n"
            "    1.0                            / AreaFactor\n"
            "    1.0                            / FlowFactor\n"
            "    1DAY                           / FlowTimeUnit\n"
            "    1    1000.0    5    2    10    50.0\n"
            "    11    -1.0\n"
            "C Root zone params\n"
            "    1e-6                           / Tolerance\n"
            "    500                            / MaxIter\n"
            "    1.0                            / LengthFactor\n"
            "    1.0                            / CNFactor\n"
            "    1.0                            / KFactor\n"
            "    1DAY                           / KTimeUnit\n"
            "    1    1    1.0    2    0.1    0.3    0.4    0.5    2.0    1e-5    1    75.0\n"
            "C Aquifer params\n"
            "    1.0                            / GWFactor\n"
            "    1.0                            / TimeFactor\n"
            "    1DAY                           / TimeUnit\n"
            "    1    100.0    500.0    0.05    0.02\n"
        )
        filepath = self._write(tmp_path, content)
        config = SmallWatershedMainReader().read(filepath)

        assert config.version == "4.0"
        assert config.n_watersheds == 1
        assert config.budget_output_file is not None

        # Geospatial
        assert len(config.watershed_specs) == 1
        ws = config.watershed_specs[0]
        assert ws.id == 1
        assert ws.area == pytest.approx(1000.0)
        assert ws.dest_stream_node == 5
        assert len(ws.gw_nodes) == 2
        assert ws.gw_nodes[0].gw_node_id == 10
        assert ws.gw_nodes[0].max_perc_rate == pytest.approx(50.0)
        assert ws.gw_nodes[1].is_baseflow is True
        assert ws.gw_nodes[1].layer == 1

        # Root zone
        assert len(config.rootzone_params) == 1
        rz = config.rootzone_params[0]
        assert rz.id == 1
        assert rz.curve_number == pytest.approx(75.0)
        assert rz.total_porosity == pytest.approx(0.4)

        # Aquifer
        assert len(config.aquifer_params) == 1
        aq = config.aquifer_params[0]
        assert aq.id == 1
        assert aq.gw_threshold == pytest.approx(100.0)
        assert aq.max_gw_storage == pytest.approx(500.0)

    def test_area_factor_applied(self, tmp_path: Path) -> None:
        """Verify area_factor is multiplied into watershed area."""
        content = (
            "#4.0\n"
            "                                   / BudgetFile\n"
            "                                   / FinalResults\n"
            "    1                              / NWatersheds\n"
            "    2.0                            / AreaFactor\n"
            "    1.0                            / FlowFactor\n"
            "    1DAY\n"
            "    1    500.0    5    1    10    50.0\n"
            "C Root zone\n"
            "    1e-6\n"
            "    500\n"
            "    1.0\n"
            "    1.0\n"
            "    1.0\n"
            "    1DAY\n"
            "    1    1    1.0    2    0.1    0.3    0.4    0.5    2.0    1e-5    1    75.0\n"
            "C Aquifer\n"
            "    1.0\n"
            "    1.0\n"
            "    1DAY\n"
            "    1    100.0    500.0    0.05    0.02\n"
        )
        filepath = self._write(tmp_path, content)
        config = SmallWatershedMainReader().read(filepath)

        assert config.watershed_specs[0].area == pytest.approx(1000.0)  # 500 * 2.0

    def test_convenience_function(self, tmp_path: Path) -> None:
        """Test read_small_watershed_main convenience function."""
        content = (
            "#4.0\n"
            "                                   / BudgetFile\n"
            "                                   / FinalResults\n"
            "    0                              / NWatersheds\n"
        )
        filepath = self._write(tmp_path, content)
        config = read_small_watershed_main(filepath)
        assert config.n_watersheds == 0


# =============================================================================
# Component: AppUnsatZone tests
# =============================================================================


class TestUnsatZoneLayer:
    """Tests for UnsatZoneLayer."""

    def test_defaults(self) -> None:
        layer = UnsatZoneLayer()
        assert layer.thickness_max == 0.0
        assert layer.total_porosity == 0.0
        assert layer.lambda_param == 0.0
        assert layer.hyd_cond == 0.0
        assert layer.kunsat_method == 0

    def test_with_values(self) -> None:
        layer = UnsatZoneLayer(
            thickness_max=5.0, total_porosity=0.35,
            lambda_param=0.5, hyd_cond=1e-5, kunsat_method=1
        )
        assert layer.thickness_max == 5.0


class TestUnsatZoneElement:
    """Tests for UnsatZoneElement."""

    def test_defaults(self) -> None:
        elem = UnsatZoneElement()
        assert elem.element_id == 0
        assert elem.layers == []
        assert elem.initial_moisture is None
        assert elem.n_layers == 0

    def test_n_layers_property(self) -> None:
        elem = UnsatZoneElement(
            element_id=1,
            layers=[UnsatZoneLayer(), UnsatZoneLayer(), UnsatZoneLayer()],
        )
        assert elem.n_layers == 3

    def test_repr(self) -> None:
        elem = UnsatZoneElement(element_id=42, layers=[UnsatZoneLayer()])
        r = repr(elem)
        assert "42" in r
        assert "1" in r


class TestAppUnsatZone:
    """Tests for AppUnsatZone component class."""

    def test_defaults(self) -> None:
        comp = AppUnsatZone()
        assert comp.n_layers == 0
        assert comp.n_elements == 0
        assert comp.solver_tolerance == 1e-8

    def test_add_and_get_element(self) -> None:
        comp = AppUnsatZone(n_layers=1)
        elem = UnsatZoneElement(element_id=5, layers=[UnsatZoneLayer()])
        comp.add_element(elem)
        assert comp.n_elements == 1
        assert comp.get_element(5).element_id == 5

    def test_iter_elements_sorted(self) -> None:
        comp = AppUnsatZone(n_layers=1)
        for eid in [3, 1, 2]:
            comp.add_element(UnsatZoneElement(
                element_id=eid, layers=[UnsatZoneLayer()]
            ))
        ids = [e.element_id for e in comp.iter_elements()]
        assert ids == [1, 2, 3]

    def test_validate_success(self) -> None:
        comp = AppUnsatZone(n_layers=1)
        comp.add_element(UnsatZoneElement(
            element_id=1, layers=[UnsatZoneLayer()]
        ))
        comp.validate()  # Should not raise

    def test_validate_zero_layers_raises(self) -> None:
        comp = AppUnsatZone(n_layers=0)
        with pytest.raises(ComponentError, match="non-positive"):
            comp.validate()

    def test_validate_layer_mismatch_raises(self) -> None:
        comp = AppUnsatZone(n_layers=2)
        comp.add_element(UnsatZoneElement(
            element_id=1, layers=[UnsatZoneLayer()]  # only 1 layer
        ))
        with pytest.raises(ComponentError, match="1 layers"):
            comp.validate()

    def test_from_config(self, tmp_path: Path) -> None:
        """Test from_config class method with parsed config."""
        content = (
            "#4.0\n"
            "    1                              / NLayers\n"
            "    1e-8\n"
            "    2000\n"
            "                                   / BudgetFile\n"
            "                                   / ZBudgetFile\n"
            "                                   / FinalResults\n"
            "    0                              / NGroup\n"
            "    1.0  1.0  1.0\n"
            "    1DAY\n"
            "    1    5.0    0.35    0.5    1e-5    1\n"
            "    2    6.0    0.30    0.4    2e-5    1\n"
            "C IC\n"
            "    0    0.20\n"
        )
        filepath = tmp_path / "uzf.dat"
        filepath.write_text(content)
        config = read_unsaturated_zone_main(filepath)
        comp = AppUnsatZone.from_config(config)

        assert comp.n_layers == 1
        assert comp.n_elements == 2
        elem1 = comp.get_element(1)
        assert elem1.layers[0].thickness_max == pytest.approx(5.0)
        assert elem1.initial_moisture is not None
        assert elem1.initial_moisture[0] == pytest.approx(0.20)

    def test_repr(self) -> None:
        comp = AppUnsatZone(n_layers=2)
        r = repr(comp)
        assert "n_layers=2" in r
        assert "n_elements=0" in r


# =============================================================================
# Component: AppSmallWatershed tests
# =============================================================================


class TestCompWatershedGWNode:
    """Tests for component WatershedGWNode."""

    def test_defaults(self) -> None:
        gn = CompWatershedGWNode()
        assert gn.gw_node_id == 0
        assert gn.max_perc_rate == 0.0
        assert gn.is_baseflow is False
        assert gn.layer == 0


class TestWatershedUnit:
    """Tests for WatershedUnit."""

    def test_defaults(self) -> None:
        ws = WatershedUnit()
        assert ws.id == 0
        assert ws.area == 0.0
        assert ws.n_gw_nodes == 0
        assert ws.curve_number == 0.0

    def test_n_gw_nodes_property(self) -> None:
        ws = WatershedUnit(
            gw_nodes=[CompWatershedGWNode(), CompWatershedGWNode()]
        )
        assert ws.n_gw_nodes == 2

    def test_repr(self) -> None:
        ws = WatershedUnit(id=3, area=1000.0)
        r = repr(ws)
        assert "3" in r
        assert "1000" in r


class TestAppSmallWatershed:
    """Tests for AppSmallWatershed component class."""

    def test_defaults(self) -> None:
        comp = AppSmallWatershed()
        assert comp.n_watersheds == 0
        assert comp.area_factor == 1.0

    def test_add_and_get_watershed(self) -> None:
        comp = AppSmallWatershed()
        ws = WatershedUnit(id=5, area=100.0, dest_stream_node=1,
                           gw_nodes=[CompWatershedGWNode(gw_node_id=1)])
        comp.add_watershed(ws)
        assert comp.n_watersheds == 1
        assert comp.get_watershed(5).area == 100.0

    def test_iter_watersheds_sorted(self) -> None:
        comp = AppSmallWatershed()
        for wid in [3, 1, 2]:
            comp.add_watershed(WatershedUnit(
                id=wid, area=float(wid),
                dest_stream_node=1,
                gw_nodes=[CompWatershedGWNode(gw_node_id=1)],
            ))
        ids = [ws.id for ws in comp.iter_watersheds()]
        assert ids == [1, 2, 3]

    def test_validate_success(self) -> None:
        comp = AppSmallWatershed()
        comp.add_watershed(WatershedUnit(
            id=1, area=100.0, dest_stream_node=5,
            gw_nodes=[CompWatershedGWNode(gw_node_id=10)],
        ))
        comp.validate()  # Should not raise

    def test_validate_zero_area_raises(self) -> None:
        comp = AppSmallWatershed()
        comp.add_watershed(WatershedUnit(
            id=1, area=0.0, dest_stream_node=5,
            gw_nodes=[CompWatershedGWNode(gw_node_id=10)],
        ))
        with pytest.raises(ComponentError, match="non-positive area"):
            comp.validate()

    def test_validate_invalid_stream_node_raises(self) -> None:
        comp = AppSmallWatershed()
        comp.add_watershed(WatershedUnit(
            id=1, area=100.0, dest_stream_node=0,
            gw_nodes=[CompWatershedGWNode(gw_node_id=10)],
        ))
        with pytest.raises(ComponentError, match="invalid destination"):
            comp.validate()

    def test_validate_no_gw_nodes_raises(self) -> None:
        comp = AppSmallWatershed()
        comp.add_watershed(WatershedUnit(
            id=1, area=100.0, dest_stream_node=5, gw_nodes=[],
        ))
        with pytest.raises(ComponentError, match="no connected GW nodes"):
            comp.validate()

    def test_from_config(self, tmp_path: Path) -> None:
        """Test from_config class method with parsed config."""
        content = (
            "#4.0\n"
            "    budget.hdf                     / BudgetFile\n"
            "    final.dat                      / FinalResults\n"
            "    1                              / NWatersheds\n"
            "C Geospatial\n"
            "    1.0                            / AreaFactor\n"
            "    1.0                            / FlowFactor\n"
            "    1DAY\n"
            "    1    1000.0    5    1    10    50.0\n"
            "C Root zone\n"
            "    1e-6\n"
            "    500\n"
            "    1.0\n"
            "    1.0\n"
            "    1.0\n"
            "    1DAY\n"
            "    1    1    1.0    2    0.1    0.3    0.4    0.5    2.0    1e-5    1    75.0\n"
            "C Aquifer\n"
            "    1.0\n"
            "    1.0\n"
            "    1DAY\n"
            "    1    100.0    500.0    0.05    0.02\n"
        )
        filepath = tmp_path / "sw_main.dat"
        filepath.write_text(content)
        config = read_small_watershed_main(filepath)
        comp = AppSmallWatershed.from_config(config)

        assert comp.n_watersheds == 1
        ws = comp.get_watershed(1)
        assert ws.area == pytest.approx(1000.0)
        assert ws.curve_number == pytest.approx(75.0)
        assert ws.gw_threshold == pytest.approx(100.0)
        assert ws.baseflow_coeff == pytest.approx(0.02)

    def test_repr(self) -> None:
        comp = AppSmallWatershed()
        comp.add_watershed(WatershedUnit(id=1, area=1.0))
        r = repr(comp)
        assert "n_watersheds=1" in r
