"""Tests for Small Watershed component, reader, and writer."""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pyiwfm.components.small_watershed import (
    AppSmallWatershed,
    WatershedGWNode,
    WatershedUnit,
)
from pyiwfm.io.small_watershed import (
    SmallWatershedMainConfig,
    SmallWatershedMainReader,
    WatershedAquiferParams,
    WatershedGWNode as ReaderGWNode,
    WatershedInitialCondition,
    WatershedRootZoneParams,
    WatershedSpec,
)
from pyiwfm.io.small_watershed_writer import (
    SmallWatershedComponentWriter,
    SmallWatershedWriterConfig,
    write_small_watershed_component,
)


# ---------------------------------------------------------------------------
# Component class tests
# ---------------------------------------------------------------------------

class TestWatershedUnit:
    def test_basic_creation(self):
        ws = WatershedUnit(id=1, area=100.0, dest_stream_node=5)
        assert ws.id == 1
        assert ws.area == 100.0
        assert ws.dest_stream_node == 5
        assert ws.n_gw_nodes == 0

    def test_gw_nodes(self):
        gn = WatershedGWNode(gw_node_id=10, max_perc_rate=5.0)
        ws = WatershedUnit(id=1, area=50.0, dest_stream_node=3, gw_nodes=[gn])
        assert ws.n_gw_nodes == 1
        assert ws.gw_nodes[0].gw_node_id == 10

    def test_repr(self):
        ws = WatershedUnit(id=2, area=200.5)
        assert "id=2" in repr(ws)


class TestAppSmallWatershed:
    def _make_component(self) -> AppSmallWatershed:
        comp = AppSmallWatershed()
        gn1 = WatershedGWNode(gw_node_id=10, max_perc_rate=5.0)
        gn2 = WatershedGWNode(gw_node_id=20, max_perc_rate=3.0)
        ws1 = WatershedUnit(
            id=1, area=100.0, dest_stream_node=5, gw_nodes=[gn1]
        )
        ws2 = WatershedUnit(
            id=2, area=200.0, dest_stream_node=8, gw_nodes=[gn2]
        )
        comp.add_watershed(ws1)
        comp.add_watershed(ws2)
        return comp

    def test_add_and_count(self):
        comp = self._make_component()
        assert comp.n_watersheds == 2

    def test_get_watershed(self):
        comp = self._make_component()
        ws = comp.get_watershed(1)
        assert ws.area == 100.0

    def test_iter_watersheds_sorted(self):
        comp = self._make_component()
        ids = [ws.id for ws in comp.iter_watersheds()]
        assert ids == [1, 2]

    def test_validate_valid(self):
        comp = self._make_component()
        comp.validate()  # Should not raise

    def test_validate_bad_area(self):
        comp = AppSmallWatershed()
        comp.add_watershed(
            WatershedUnit(
                id=1,
                area=-1.0,
                dest_stream_node=5,
                gw_nodes=[WatershedGWNode(gw_node_id=1)],
            )
        )
        with pytest.raises(Exception, match="non-positive area"):
            comp.validate()

    def test_validate_no_gw_nodes(self):
        comp = AppSmallWatershed()
        comp.add_watershed(
            WatershedUnit(id=1, area=10.0, dest_stream_node=5)
        )
        with pytest.raises(Exception, match="no connected GW nodes"):
            comp.validate()

    def test_repr(self):
        comp = self._make_component()
        assert "n_watersheds=2" in repr(comp)


class TestFromConfig:
    def test_from_config(self):
        config = SmallWatershedMainConfig(
            version="4.0",
            n_watersheds=1,
            area_factor=1.0,
            flow_factor=1.0,
            flow_time_unit="1DAY",
            rz_length_factor=1.0,
            rz_cn_factor=1.0,
            rz_k_factor=1.0,
            aq_gw_factor=1.0,
            aq_time_factor=1.0,
            ic_factor=1.0,
        )
        config.watershed_specs = [
            WatershedSpec(
                id=1,
                area=500.0,
                dest_stream_node=3,
                gw_nodes=[ReaderGWNode(gw_node_id=10, max_perc_rate=2.5)],
            ),
        ]
        config.rootzone_params = [
            WatershedRootZoneParams(
                id=1,
                precip_col=2,
                wilting_point=0.1,
                field_capacity=0.3,
                total_porosity=0.4,
                curve_number=75.0,
            ),
        ]
        config.aquifer_params = [
            WatershedAquiferParams(
                id=1,
                gw_threshold=10.0,
                max_gw_storage=100.0,
                surface_flow_coeff=0.01,
                baseflow_coeff=0.005,
            ),
        ]
        config.initial_conditions = [
            WatershedInitialCondition(id=1, soil_moisture=0.3, gw_storage=8.0),
        ]

        comp = AppSmallWatershed.from_config(config)
        assert comp.n_watersheds == 1
        assert comp.ic_factor == 1.0

        ws = comp.get_watershed(1)
        assert ws.area == 500.0
        assert ws.dest_stream_node == 3
        assert ws.wilting_point == 0.1
        assert ws.curve_number == 75.0
        assert ws.gw_threshold == 10.0
        assert ws.baseflow_coeff == 0.005
        assert ws.initial_soil_moisture == 0.3
        assert ws.initial_gw_storage == 8.0


# ---------------------------------------------------------------------------
# Reader tests
# ---------------------------------------------------------------------------

class TestSmallWatershedReader:
    def test_read_empty_file(self, tmp_path):
        content = textwrap.dedent("""\
            #4.0
            C  Budget output file
                                                         / SWBUDFL
            C  Final results file
                                                         / FNSWFL
            C  Number of watersheds
                 0                                        / NSWShed
        """)
        filepath = tmp_path / "SmallWS_MAIN.dat"
        filepath.write_text(content)

        reader = SmallWatershedMainReader()
        config = reader.read(filepath)
        assert config.version == "4.0"
        assert config.n_watersheds == 0

    def test_read_one_watershed(self, tmp_path):
        content = textwrap.dedent("""\
            #4.0
            C  Budget output file
            ../Results/SWShedBud.hdf                      / SWBUDFL
            C  Final results file
            ../Results/FinalSWShed.out                    / FNSWFL
            C  Number of watersheds
                 1                                        / NSWShed
            C  Area factor
                 1.0                                      / FACTAROU
            C  Flow factor
                 1.0                                      / FACTLTOU
            C  Flow time unit
                 1DAY                                     / UNITLTOU
            C  Watershed geospatial data
                 1  500.0  3  1  10  2.5
            C  RZ solver tolerance
                 1.0E-08                                  / RZTOLERANCE
            C  RZ max iterations
                 2000                                     / RZMAXITER
            C  RZ length factor
                 1.0                                      / FACTLN
            C  RZ CN factor
                 1.0                                      / FACTCN
            C  RZ K factor
                 1.0                                      / FACTKU
            C  RZ K time unit
                 1DAY                                     / TUNITKU
            C  Root zone parameters
                 1  2  1.0  3  0.1  0.3  0.4  0.5  10.0  0.01  1  75.0
            C  Aquifer GW factor
                 1.0                                      / FACTGW
            C  Aquifer time factor
                 1.0                                      / FACTTM
            C  Aquifer time unit
                 1DAY                                     / TUNITTM
            C  Aquifer parameters
                 1  10.0  100.0  0.01  0.005
            C  Initial conditions factor
                 1.0                                      / FACTIC
            C  Initial conditions
                 1  0.25  5.0
        """)
        filepath = tmp_path / "SmallWS_MAIN.dat"
        filepath.write_text(content)

        reader = SmallWatershedMainReader()
        config = reader.read(filepath)

        assert config.n_watersheds == 1
        assert len(config.watershed_specs) == 1
        assert config.watershed_specs[0].dest_stream_node == 3
        assert len(config.rootzone_params) == 1
        assert len(config.aquifer_params) == 1
        assert config.aquifer_params[0].baseflow_coeff == 0.005
        # Initial conditions
        assert len(config.initial_conditions) == 1
        assert config.initial_conditions[0].soil_moisture == 0.25
        assert config.initial_conditions[0].gw_storage == 5.0


class TestSmallWatershedReaderIC:
    """Tests for initial conditions parsing."""

    def test_read_initial_conditions_with_factor(self, tmp_path):
        """IC factor is applied to gw_storage but not soil_moisture."""
        content = textwrap.dedent("""\
            #4.0
            C  Budget output file
                                                         / SWBUDFL
            C  Final results file
                                                         / FNSWFL
            C  Number of watersheds
                 1                                        / NSWShed
            C  Area factor
                 1.0                                      / FACTAROU
            C  Flow factor
                 1.0                                      / FACTLTOU
            C  Flow time unit
                 1DAY                                     / UNITLTOU
            C  Watershed geospatial data
                 1  500.0  3  1  10  2.5
            C  RZ solver tolerance
                 1.0E-08                                  / RZTOLERANCE
            C  RZ max iterations
                 2000                                     / RZMAXITER
            C  RZ length factor
                 1.0                                      / FACTLN
            C  RZ CN factor
                 1.0                                      / FACTCN
            C  RZ K factor
                 1.0                                      / FACTKU
            C  RZ K time unit
                 1DAY                                     / TUNITKU
            C  Root zone parameters
                 1  2  1.0  3  0.1  0.3  0.4  0.5  10.0  0.01  1  75.0
            C  Aquifer GW factor
                 1.0                                      / FACTGW
            C  Aquifer time factor
                 1.0                                      / FACTTM
            C  Aquifer time unit
                 1DAY                                     / TUNITTM
            C  Aquifer parameters
                 1  10.0  100.0  0.01  0.005
            C  IC factor
                 0.3048                                   / FACTIC
            C  Initial conditions
                 1  0.5  10.0
        """)
        filepath = tmp_path / "SmallWS_MAIN.dat"
        filepath.write_text(content)

        reader = SmallWatershedMainReader()
        config = reader.read(filepath)

        assert config.ic_factor == pytest.approx(0.3048)
        assert len(config.initial_conditions) == 1
        ic = config.initial_conditions[0]
        assert ic.id == 1
        assert ic.soil_moisture == pytest.approx(0.5)
        # gw_storage = raw (10.0) * ic_factor (0.3048)
        assert ic.gw_storage == pytest.approx(10.0 * 0.3048)

    def test_read_two_watersheds_ic(self, tmp_path):
        """IC section with multiple watersheds."""
        content = textwrap.dedent("""\
            #4.0
            C  Budget output file
                                                         / SWBUDFL
            C  Final results file
                                                         / FNSWFL
            C  Number of watersheds
                 2                                        / NSWShed
            C  Area factor
                 1.0                                      / FACTAROU
            C  Flow factor
                 1.0                                      / FACTLTOU
            C  Flow time unit
                 1DAY                                     / UNITLTOU
            C  Watershed geospatial data
                 1  500.0  3  1  10  2.5
                 2  600.0  4  1  20  3.0
            C  RZ solver tolerance
                 1.0E-08                                  / RZTOLERANCE
            C  RZ max iterations
                 2000                                     / RZMAXITER
            C  RZ length factor
                 1.0                                      / FACTLN
            C  RZ CN factor
                 1.0                                      / FACTCN
            C  RZ K factor
                 1.0                                      / FACTKU
            C  RZ K time unit
                 1DAY                                     / TUNITKU
            C  Root zone parameters
                 1  2  1.0  3  0.1  0.3  0.4  0.5  10.0  0.01  1  75.0
                 2  2  1.0  3  0.1  0.3  0.4  0.5  10.0  0.01  1  75.0
            C  Aquifer GW factor
                 1.0                                      / FACTGW
            C  Aquifer time factor
                 1.0                                      / FACTTM
            C  Aquifer time unit
                 1DAY                                     / TUNITTM
            C  Aquifer parameters
                 1  10.0  100.0  0.01  0.005
                 2  12.0  120.0  0.02  0.008
            C  IC factor
                 1.0                                      / FACTIC
            C  Initial conditions
                 1  0.25  5.0
                 2  0.35  8.0
        """)
        filepath = tmp_path / "SmallWS_MAIN.dat"
        filepath.write_text(content)

        reader = SmallWatershedMainReader()
        config = reader.read(filepath)

        assert len(config.initial_conditions) == 2
        assert config.initial_conditions[0].soil_moisture == 0.25
        assert config.initial_conditions[0].gw_storage == 5.0
        assert config.initial_conditions[1].soil_moisture == 0.35
        assert config.initial_conditions[1].gw_storage == 8.0


class TestSmallWatershedReaderV41:
    """Tests for v4.1 root zone parsing (crop_coeff_col)."""

    def test_read_v41_root_zone(self, tmp_path):
        """v4.1 root zone line has 13 fields with crop_coeff_col after et_col."""
        content = textwrap.dedent("""\
            #4.1
            C  Budget output file
                                                         / SWBUDFL
            C  Final results file
                                                         / FNSWFL
            C  Number of watersheds
                 1                                        / NSWShed
            C  Area factor
                 1.0                                      / FACTAROU
            C  Flow factor
                 1.0                                      / FACTLTOU
            C  Flow time unit
                 1DAY                                     / UNITLTOU
            C  Watershed geospatial data
                 1  500.0  3  1  10  2.5
            C  RZ solver tolerance
                 1.0E-08                                  / RZTOLERANCE
            C  RZ max iterations
                 2000                                     / RZMAXITER
            C  RZ length factor
                 1.0                                      / FACTLN
            C  RZ CN factor
                 1.0                                      / FACTCN
            C  RZ K factor
                 1.0                                      / FACTKU
            C  RZ K time unit
                 1DAY                                     / TUNITKU
            C  Root zone parameters (v4.1: ID ICOLPREC FACTPREC ICOLET ICOLCROPCOEF WP FC TP LBD RZDPTH HYDCOND KUNM CN)
                 1  2  1.0  3  7  0.1  0.3  0.4  0.5  10.0  0.01  1  75.0
            C  Aquifer GW factor
                 1.0                                      / FACTGW
            C  Aquifer time factor
                 1.0                                      / FACTTM
            C  Aquifer time unit
                 1DAY                                     / TUNITTM
            C  Aquifer parameters
                 1  10.0  100.0  0.01  0.005
            C  IC factor
                 1.0                                      / FACTIC
            C  Initial conditions
                 1  0.2  3.0
        """)
        filepath = tmp_path / "SmallWS_MAIN.dat"
        filepath.write_text(content)

        reader = SmallWatershedMainReader()
        config = reader.read(filepath)

        assert config.version == "4.1"
        assert len(config.rootzone_params) == 1
        rz = config.rootzone_params[0]
        assert rz.precip_col == 2
        assert rz.et_col == 3
        assert rz.crop_coeff_col == 7
        assert rz.wilting_point == pytest.approx(0.1)
        assert rz.curve_number == pytest.approx(75.0)


# ---------------------------------------------------------------------------
# Writer tests
# ---------------------------------------------------------------------------

class TestSmallWatershedWriter:
    def test_write_main(self, tmp_path):
        from pyiwfm.io.small_watershed_writer import (
            SmallWatershedComponentWriter,
            SmallWatershedWriterConfig,
        )

        # Build mock model
        model = MagicMock()
        model.name = "TestModel"

        comp = AppSmallWatershed(
            area_factor=1.0,
            flow_factor=1.0,
            flow_time_unit="1DAY",
            rz_length_factor=1.0,
            rz_cn_factor=1.0,
            rz_k_factor=1.0,
            rz_k_time_unit="1DAY",
            aq_gw_factor=1.0,
            aq_time_factor=1.0,
            aq_time_unit="1DAY",
            ic_factor=1.0,
        )
        gn = WatershedGWNode(gw_node_id=10, max_perc_rate=2.5)
        ws = WatershedUnit(
            id=1,
            area=500.0,
            dest_stream_node=3,
            gw_nodes=[gn],
            precip_col=2,
            wilting_point=0.1,
            field_capacity=0.3,
            total_porosity=0.4,
            lambda_param=0.5,
            root_depth=10.0,
            hydraulic_cond=0.01,
            curve_number=75.0,
            gw_threshold=10.0,
            max_gw_storage=100.0,
            surface_flow_coeff=0.01,
            baseflow_coeff=0.005,
            initial_soil_moisture=0.3,
            initial_gw_storage=5.0,
        )
        comp.add_watershed(ws)
        model.small_watersheds = comp

        config = SmallWatershedWriterConfig(
            output_dir=tmp_path,
            swshed_subdir="",
        )

        writer = SmallWatershedComponentWriter(model, config)
        files = writer.write_all()

        assert "main" in files
        main_path = files["main"]
        assert main_path.exists()

        content = main_path.read_text()
        assert "#4.0" in content
        assert "SMALL WATERSHED" in content.upper()
        # IC section should be present
        assert "FACTIC" in content
        assert "SOILMOIST" in content


# ---------------------------------------------------------------------------
# Round-trip test
# ---------------------------------------------------------------------------

class TestSmallWatershedRoundTrip:
    def test_read_write_read(self, tmp_path):
        """Read a file, build component, write it back, re-read."""
        from pyiwfm.io.small_watershed_writer import (
            SmallWatershedComponentWriter,
            SmallWatershedWriterConfig,
        )

        # Step 1: Build a config from scratch
        config = SmallWatershedMainConfig(
            version="4.0",
            n_watersheds=1,
            area_factor=1.0,
            flow_factor=1.0,
            flow_time_unit="1DAY",
            rz_length_factor=1.0,
            rz_cn_factor=1.0,
            rz_k_factor=1.0,
            rz_k_time_unit="1DAY",
            aq_gw_factor=1.0,
            aq_time_factor=1.0,
            aq_time_unit="1DAY",
            ic_factor=1.0,
        )
        config.watershed_specs = [
            WatershedSpec(
                id=1,
                area=500.0,
                dest_stream_node=3,
                gw_nodes=[ReaderGWNode(gw_node_id=10, max_perc_rate=2.5)],
            ),
        ]
        config.rootzone_params = [
            WatershedRootZoneParams(
                id=1,
                precip_col=2,
                precip_factor=1.0,
                et_col=3,
                wilting_point=0.1,
                field_capacity=0.3,
                total_porosity=0.4,
                lambda_param=0.5,
                root_depth=10.0,
                hydraulic_cond=0.01,
                kunsat_method=1,
                curve_number=75.0,
            ),
        ]
        config.aquifer_params = [
            WatershedAquiferParams(
                id=1,
                gw_threshold=10.0,
                max_gw_storage=100.0,
                surface_flow_coeff=0.01,
                baseflow_coeff=0.005,
            ),
        ]
        config.initial_conditions = [
            WatershedInitialCondition(id=1, soil_moisture=0.25, gw_storage=5.0),
        ]

        # Step 2: Build component
        comp = AppSmallWatershed.from_config(config)
        assert comp.n_watersheds == 1

        # Step 3: Write via writer
        model = MagicMock()
        model.name = "TestModel"
        model.small_watersheds = comp

        writer_config = SmallWatershedWriterConfig(
            output_dir=tmp_path,
            swshed_subdir="",
        )
        writer = SmallWatershedComponentWriter(model, writer_config)
        files = writer.write_all()

        # Step 4: Re-read the written file
        reader = SmallWatershedMainReader()
        re_config = reader.read(files["main"])

        assert re_config.n_watersheds == 1
        assert len(re_config.watershed_specs) == 1
        assert re_config.watershed_specs[0].dest_stream_node == 3
        # Verify IC section round-tripped
        assert len(re_config.initial_conditions) == 1
        assert re_config.initial_conditions[0].soil_moisture == pytest.approx(0.25)
        assert re_config.initial_conditions[0].gw_storage == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Additional writer coverage tests
# ---------------------------------------------------------------------------

class TestSmallWatershedWriterConfig:
    def test_swshed_dir_empty_subdir(self, tmp_path):
        """Config with empty swshed_subdir returns output_dir directly."""
        config = SmallWatershedWriterConfig(
            output_dir=tmp_path,
            swshed_subdir="",
        )
        assert config.swshed_dir == tmp_path

    def test_swshed_dir_with_subdir(self, tmp_path):
        """Config with non-empty swshed_subdir returns subdirectory."""
        config = SmallWatershedWriterConfig(
            output_dir=tmp_path,
            swshed_subdir="SmallWatershed",
        )
        assert config.swshed_dir == tmp_path / "SmallWatershed"


class TestSmallWatershedWriterCoverage:
    def test_format_property(self, tmp_path):
        """Verify format property returns 'iwfm_small_watershed'."""
        model = MagicMock()
        model.small_watersheds = None
        config = SmallWatershedWriterConfig(
            output_dir=tmp_path,
            swshed_subdir="",
        )
        writer = SmallWatershedComponentWriter(model, config)
        assert writer.format == "iwfm_small_watershed"

    def test_write_delegates_to_write_all(self, tmp_path):
        """Call write(), verify write_all is called."""
        model = MagicMock()
        model.small_watersheds = None
        config = SmallWatershedWriterConfig(
            output_dir=tmp_path,
            swshed_subdir="",
        )
        writer = SmallWatershedComponentWriter(model, config)
        with patch.object(writer, "write_all") as mock_write_all:
            writer.write()
            mock_write_all.assert_called_once()

    def test_write_all_no_component_no_defaults(self, tmp_path):
        """Model without small_watersheds, write_defaults=False => early return."""
        model = MagicMock()
        model.small_watersheds = None
        config = SmallWatershedWriterConfig(
            output_dir=tmp_path,
            swshed_subdir="",
        )
        writer = SmallWatershedComponentWriter(model, config)
        results = writer.write_all(write_defaults=False)
        assert results == {}

    def test_write_main_no_watersheds(self, tmp_path):
        """Small watershed component with empty watershed list."""
        model = MagicMock()
        comp = AppSmallWatershed(
            area_factor=1.0,
            flow_factor=1.0,
            flow_time_unit="1DAY",
        )
        # No watersheds added
        model.small_watersheds = comp

        config = SmallWatershedWriterConfig(
            output_dir=tmp_path,
            swshed_subdir="",
        )
        writer = SmallWatershedComponentWriter(model, config)
        files = writer.write_all()

        assert "main" in files
        content = files["main"].read_text()
        # n_watersheds should be 0
        assert "0" in content

    def test_render_main_baseflow_nodes(self, tmp_path):
        """Watershed with baseflow GW nodes (is_baseflow=True) => negative layer."""
        model = MagicMock()
        comp = AppSmallWatershed(
            area_factor=1.0,
            flow_factor=1.0,
            flow_time_unit="1DAY",
            rz_length_factor=1.0,
            rz_cn_factor=1.0,
            rz_k_factor=1.0,
            rz_k_time_unit="1DAY",
            aq_gw_factor=1.0,
            aq_time_factor=1.0,
            aq_time_unit="1DAY",
            ic_factor=1.0,
        )
        # Create a baseflow GW node
        gn_baseflow = WatershedGWNode(
            gw_node_id=11, max_perc_rate=5.0, is_baseflow=True, layer=2
        )
        gn_normal = WatershedGWNode(
            gw_node_id=10, max_perc_rate=2.5, is_baseflow=False, layer=0
        )
        ws = WatershedUnit(
            id=1,
            area=500.0,
            dest_stream_node=3,
            gw_nodes=[gn_normal, gn_baseflow],
            precip_col=2,
            wilting_point=0.1,
            field_capacity=0.3,
            total_porosity=0.4,
            lambda_param=0.5,
            root_depth=10.0,
            hydraulic_cond=0.01,
            curve_number=75.0,
            gw_threshold=10.0,
            max_gw_storage=100.0,
            surface_flow_coeff=0.01,
            baseflow_coeff=0.005,
        )
        comp.add_watershed(ws)
        model.small_watersheds = comp

        config = SmallWatershedWriterConfig(
            output_dir=tmp_path,
            swshed_subdir="",
        )
        writer = SmallWatershedComponentWriter(model, config)
        files = writer.write_all()

        assert "main" in files
        assert files["main"].exists()
        content = files["main"].read_text()
        # The baseflow node should produce a negative layer value in the raw output
        # (perc_rate_raw = -float(gn.layer) = -2.0)
        assert "-2" in content

    def test_render_main_component_output_files(self, tmp_path):
        """Component with budget_output_file and final_results_file set."""
        model = MagicMock()
        comp = AppSmallWatershed(
            area_factor=1.0,
            flow_factor=1.0,
            flow_time_unit="1DAY",
            rz_length_factor=1.0,
            rz_cn_factor=1.0,
            rz_k_factor=1.0,
            rz_k_time_unit="1DAY",
            aq_gw_factor=1.0,
            aq_time_factor=1.0,
            aq_time_unit="1DAY",
            ic_factor=1.0,
            budget_output_file="../CustomResults/SWShedBud.hdf",
            final_results_file="../CustomResults/FinalSW.out",
        )
        gn = WatershedGWNode(gw_node_id=10, max_perc_rate=2.5)
        ws = WatershedUnit(
            id=1,
            area=500.0,
            dest_stream_node=3,
            gw_nodes=[gn],
            precip_col=2,
            wilting_point=0.1,
            field_capacity=0.3,
            total_porosity=0.4,
            lambda_param=0.5,
            root_depth=10.0,
            hydraulic_cond=0.01,
            curve_number=75.0,
            gw_threshold=10.0,
            max_gw_storage=100.0,
            surface_flow_coeff=0.01,
            baseflow_coeff=0.005,
        )
        comp.add_watershed(ws)
        model.small_watersheds = comp

        config = SmallWatershedWriterConfig(
            output_dir=tmp_path,
            swshed_subdir="",
        )
        writer = SmallWatershedComponentWriter(model, config)
        files = writer.write_all()

        content = files["main"].read_text()
        # The custom output files from the component should override config defaults
        assert "CustomResults/SWShedBud.hdf" in content
        assert "CustomResults/FinalSW.out" in content

    def test_write_sw_convenience_no_config(self, tmp_path):
        """write_small_watershed_component() with config=None."""
        model = MagicMock()
        comp = AppSmallWatershed(
            area_factor=1.0,
            flow_factor=1.0,
            flow_time_unit="1DAY",
            rz_length_factor=1.0,
            rz_cn_factor=1.0,
            rz_k_factor=1.0,
            rz_k_time_unit="1DAY",
            aq_gw_factor=1.0,
            aq_time_factor=1.0,
            aq_time_unit="1DAY",
            ic_factor=1.0,
        )
        gn = WatershedGWNode(gw_node_id=10, max_perc_rate=2.5)
        ws = WatershedUnit(
            id=1,
            area=500.0,
            dest_stream_node=3,
            gw_nodes=[gn],
            precip_col=2,
            wilting_point=0.1,
            field_capacity=0.3,
            total_porosity=0.4,
            lambda_param=0.5,
            root_depth=10.0,
            hydraulic_cond=0.01,
            curve_number=75.0,
            gw_threshold=10.0,
            max_gw_storage=100.0,
            surface_flow_coeff=0.01,
            baseflow_coeff=0.005,
        )
        comp.add_watershed(ws)
        model.small_watersheds = comp

        results = write_small_watershed_component(model, tmp_path, config=None)
        assert "main" in results
        assert results["main"].exists()

    def test_write_sw_convenience_with_config(self, tmp_path):
        """write_small_watershed_component() with custom config."""
        model = MagicMock()
        comp = AppSmallWatershed(
            area_factor=1.0,
            flow_factor=1.0,
            flow_time_unit="1DAY",
            rz_length_factor=1.0,
            rz_cn_factor=1.0,
            rz_k_factor=1.0,
            rz_k_time_unit="1DAY",
            aq_gw_factor=1.0,
            aq_time_factor=1.0,
            aq_time_unit="1DAY",
            ic_factor=1.0,
        )
        gn = WatershedGWNode(gw_node_id=10, max_perc_rate=2.5)
        ws = WatershedUnit(
            id=1,
            area=500.0,
            dest_stream_node=3,
            gw_nodes=[gn],
            precip_col=2,
            wilting_point=0.1,
            field_capacity=0.3,
            total_porosity=0.4,
            lambda_param=0.5,
            root_depth=10.0,
            hydraulic_cond=0.01,
            curve_number=75.0,
            gw_threshold=10.0,
            max_gw_storage=100.0,
            surface_flow_coeff=0.01,
            baseflow_coeff=0.005,
        )
        comp.add_watershed(ws)
        model.small_watersheds = comp

        custom_config = SmallWatershedWriterConfig(
            output_dir=Path("/dummy"),  # Will be overridden
            swshed_subdir="CustomSW",
        )
        results = write_small_watershed_component(
            model, tmp_path, config=custom_config
        )
        assert "main" in results
        # Verify the config's output_dir was overridden to tmp_path
        assert custom_config.output_dir == tmp_path
        assert results["main"].exists()
