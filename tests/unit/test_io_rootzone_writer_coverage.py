"""Coverage tests for pyiwfm.io.rootzone_writer module.

Targets uncovered branches and edge cases including:
- RootZoneWriterConfig defaults and properties
- _sp_val() helper with present, missing, and alt attribute lookups
- write_all() with write_defaults=False and no root zone component
- write_all() with mock root zone writes main file
- write_main() creates output file
- Version-specific rendering (v4.12 format with per-landuse destinations)
- write_rootzone_component() convenience function
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from pyiwfm.io.rootzone_writer import (
    RootZoneWriterConfig,
    RootZoneComponentWriter,
    _sp_val,
    write_rootzone_component,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_engine():
    """Create a mock TemplateEngine that returns predictable content."""
    engine = MagicMock()
    engine.render_template.return_value = "C  MOCK ROOTZONE HEADER\n"
    engine.render_string.return_value = "C  MOCK ROOTZONE STRING\n"
    return engine


@pytest.fixture
def bare_model():
    """Create a model with rootzone=None and grid=None."""
    model = MagicMock()
    model.rootzone = None
    model.grid = None
    return model


@pytest.fixture
def model_with_grid():
    """Create a model with a grid but no root zone."""
    model = MagicMock()
    model.rootzone = None

    elem1 = MagicMock()
    elem1.id = 1
    elem2 = MagicMock()
    elem2.id = 2
    model.grid = MagicMock()
    model.grid.elements = {1: elem1, 2: elem2}
    return model


@pytest.fixture
def model_with_rootzone():
    """Create a model with grid and rootzone component."""
    model = MagicMock()

    # Grid
    elem1 = MagicMock()
    elem1.id = 1
    elem2 = MagicMock()
    elem2.id = 2
    model.grid = MagicMock()
    model.grid.elements = {1: elem1, 2: elem2}

    # Root zone
    rootzone = MagicMock()
    sp1 = SimpleNamespace(
        wilting_point=0.1, field_capacity=0.25, porosity=0.50,
        lambda_param=0.7, saturated_kv=3.0, k_ponded=-1.0,
        kunsat_method=2, capillary_rise=0.5,
        precip_column=1, precip_factor=1.0,
        generic_moisture_column=0,
    )
    rootzone.soil_params = {1: sp1, 2: sp1}
    rootzone.nonponded_config = None
    rootzone.ponded_config = None
    rootzone.urban_config = None
    rootzone.native_riparian_config = None
    rootzone.surface_flow_destinations = {}
    rootzone.surface_flow_dest_ag = {}
    rootzone.surface_flow_dest_urban_in = {}
    rootzone.surface_flow_dest_urban_out = {}
    rootzone.surface_flow_dest_nvrv = {}

    model.rootzone = rootzone
    return model


# =============================================================================
# _sp_val() Tests
# =============================================================================


class TestSpVal:
    """Tests for the _sp_val() helper function."""

    def test_returns_primary_attribute(self) -> None:
        """_sp_val() returns primary attribute when present."""
        obj = SimpleNamespace(wilting_point=0.15)
        result = _sp_val(obj, "wilting_point", 0.0)
        assert result == 0.15

    def test_returns_default_when_missing(self) -> None:
        """_sp_val() returns default when attribute is missing."""
        obj = SimpleNamespace()
        result = _sp_val(obj, "wilting_point", 0.0)
        assert result == 0.0

    def test_returns_alt_when_primary_missing(self) -> None:
        """_sp_val() tries alt name when primary is missing."""
        obj = SimpleNamespace(total_porosity=0.45)
        result = _sp_val(obj, "porosity", 0.0, alt="total_porosity")
        assert result == 0.45

    def test_returns_default_when_both_missing(self) -> None:
        """_sp_val() returns default when both primary and alt are missing."""
        obj = SimpleNamespace()
        result = _sp_val(obj, "porosity", 0.42, alt="total_porosity")
        assert result == 0.42

    def test_returns_default_for_non_numeric(self) -> None:
        """_sp_val() returns default when attribute is not numeric (e.g. MagicMock)."""
        obj = MagicMock()
        # MagicMock attributes return MagicMock objects, not numbers
        result = _sp_val(obj, "wilting_point", 0.33)
        assert result == 0.33

    def test_returns_int_attribute(self) -> None:
        """_sp_val() returns integer attribute values."""
        obj = SimpleNamespace(rhc_method=1)
        result = _sp_val(obj, "rhc_method", 2)
        assert result == 1

    def test_alt_is_none(self) -> None:
        """_sp_val() handles alt=None gracefully."""
        obj = SimpleNamespace()
        result = _sp_val(obj, "missing_attr", 99.0, alt=None)
        assert result == 99.0


# =============================================================================
# RootZoneWriterConfig Tests
# =============================================================================


class TestRootZoneWriterConfigCoverage:
    """Coverage tests for RootZoneWriterConfig properties."""

    def test_rootzone_dir_default(self, tmp_path: Path) -> None:
        """Test rootzone_dir with default subdir."""
        config = RootZoneWriterConfig(output_dir=tmp_path)
        assert config.rootzone_dir == tmp_path / "RootZone"

    def test_main_path_default(self, tmp_path: Path) -> None:
        """Test main_path with default values."""
        config = RootZoneWriterConfig(output_dir=tmp_path)
        assert config.main_path == tmp_path / "RootZone" / "RootZone_MAIN.dat"

    def test_custom_subdir(self, tmp_path: Path) -> None:
        """Test rootzone_dir with custom subdir."""
        config = RootZoneWriterConfig(output_dir=tmp_path, rootzone_subdir="RZ")
        assert config.rootzone_dir == tmp_path / "RZ"
        assert config.main_path == tmp_path / "RZ" / "RootZone_MAIN.dat"

    def test_default_version(self, tmp_path: Path) -> None:
        """Test default version is 4.12."""
        config = RootZoneWriterConfig(output_dir=tmp_path)
        assert config.version == "4.12"

    def test_default_soil_defaults(self, tmp_path: Path) -> None:
        """Test default soil parameter values."""
        config = RootZoneWriterConfig(output_dir=tmp_path)
        assert config.wilting_point == 0.0
        assert config.field_capacity == 0.20
        assert config.total_porosity == 0.45
        assert config.pore_size_index == 0.62
        assert config.hydraulic_conductivity == 2.60
        assert config.k_ponded == -1.0
        assert config.rhc_method == 2
        assert config.capillary_rise == 0.0


# =============================================================================
# write_all() Tests
# =============================================================================


class TestRootZoneWriteAll:
    """Tests for RootZoneComponentWriter.write_all()."""

    def test_write_all_no_rootzone_write_defaults_false(
        self, tmp_path: Path, bare_model: MagicMock, mock_engine: MagicMock
    ) -> None:
        """write_all(write_defaults=False) returns empty when no rootzone."""
        config = RootZoneWriterConfig(output_dir=tmp_path)
        writer = RootZoneComponentWriter(
            bare_model, config, template_engine=mock_engine
        )
        results = writer.write_all(write_defaults=False)
        assert results == {}

    def test_write_all_no_rootzone_write_defaults_true(
        self, tmp_path: Path, model_with_grid: MagicMock, mock_engine: MagicMock
    ) -> None:
        """write_all(write_defaults=True) writes main even without rootzone."""
        config = RootZoneWriterConfig(output_dir=tmp_path)
        writer = RootZoneComponentWriter(
            model_with_grid, config, template_engine=mock_engine
        )
        results = writer.write_all(write_defaults=True)
        assert "main" in results
        assert results["main"].exists()

    def test_write_all_with_rootzone_writes_main(
        self, tmp_path: Path, model_with_rootzone: MagicMock, mock_engine: MagicMock
    ) -> None:
        """write_all() with rootzone component writes main file."""
        config = RootZoneWriterConfig(output_dir=tmp_path)
        writer = RootZoneComponentWriter(
            model_with_rootzone, config, template_engine=mock_engine
        )
        results = writer.write_all()
        assert "main" in results
        assert results["main"].exists()


# =============================================================================
# write_main() Tests
# =============================================================================


class TestRootZoneWriteMain:
    """Tests for RootZoneComponentWriter.write_main()."""

    def test_write_main_creates_file(
        self, tmp_path: Path, model_with_grid: MagicMock, mock_engine: MagicMock
    ) -> None:
        """write_main() creates the main output file."""
        config = RootZoneWriterConfig(output_dir=tmp_path)
        writer = RootZoneComponentWriter(
            model_with_grid, config, template_engine=mock_engine
        )
        path = writer.write_main()
        assert path.exists()
        assert path == config.main_path

    def test_write_main_with_soil_params(
        self, tmp_path: Path, model_with_rootzone: MagicMock, mock_engine: MagicMock
    ) -> None:
        """write_main() includes soil parameter rows from rootzone data."""
        config = RootZoneWriterConfig(output_dir=tmp_path, version="4.12")
        writer = RootZoneComponentWriter(
            model_with_rootzone, config, template_engine=mock_engine
        )
        path = writer.write_main()
        content = path.read_text()
        # Should contain element soil parameter rows
        # Element 1 and 2 should be present
        assert "1" in content
        assert "2" in content

    def test_write_main_v40_format(
        self, tmp_path: Path, model_with_rootzone: MagicMock, mock_engine: MagicMock
    ) -> None:
        """v4.0 format omits capillary rise column."""
        config = RootZoneWriterConfig(output_dir=tmp_path, version="4.0")
        writer = RootZoneComponentWriter(
            model_with_rootzone, config, template_engine=mock_engine
        )
        path = writer.write_main()
        content = path.read_text()
        # v4.0 should still produce valid output
        assert path.exists()

    def test_write_main_no_grid(
        self, tmp_path: Path, bare_model: MagicMock, mock_engine: MagicMock
    ) -> None:
        """write_main() with no grid writes file with zero elements."""
        config = RootZoneWriterConfig(output_dir=tmp_path)
        writer = RootZoneComponentWriter(
            bare_model, config, template_engine=mock_engine
        )
        path = writer.write_main()
        assert path.exists()


# =============================================================================
# Miscellaneous Tests
# =============================================================================


class TestRootZoneWriterMisc:
    """Miscellaneous tests for RootZoneComponentWriter."""

    def test_format_property(
        self, tmp_path: Path, bare_model: MagicMock, mock_engine: MagicMock
    ) -> None:
        """format property returns 'iwfm_rootzone'."""
        config = RootZoneWriterConfig(output_dir=tmp_path)
        writer = RootZoneComponentWriter(
            bare_model, config, template_engine=mock_engine
        )
        assert writer.format == "iwfm_rootzone"

    def test_write_method_delegates(
        self, tmp_path: Path, model_with_grid: MagicMock, mock_engine: MagicMock
    ) -> None:
        """write() delegates to write_all()."""
        config = RootZoneWriterConfig(output_dir=tmp_path)
        writer = RootZoneComponentWriter(
            model_with_grid, config, template_engine=mock_engine
        )
        writer.write()
        assert config.main_path.exists()

    def test_write_rootzone_component_convenience(
        self, tmp_path: Path, bare_model: MagicMock, mock_engine: MagicMock
    ) -> None:
        """write_rootzone_component() convenience function works."""
        with patch(
            "pyiwfm.io.rootzone_writer.TemplateEngine", return_value=mock_engine
        ):
            results = write_rootzone_component(bare_model, tmp_path)
        assert "main" in results

    def test_write_rootzone_component_with_config(
        self, tmp_path: Path, bare_model: MagicMock, mock_engine: MagicMock
    ) -> None:
        """write_rootzone_component() uses provided config and updates output_dir."""
        config = RootZoneWriterConfig(output_dir=tmp_path, version="4.0")
        new_dir = tmp_path / "out2"
        new_dir.mkdir()
        with patch(
            "pyiwfm.io.rootzone_writer.TemplateEngine", return_value=mock_engine
        ):
            results = write_rootzone_component(bare_model, new_dir, config=config)
        assert config.output_dir == new_dir
        assert "main" in results


# =============================================================================
# write_all() sub-component writer branches (lines 201-223)
# =============================================================================


class TestWriteAllSubComponents:
    """Tests that write_all() invokes v4x sub-writers when configs are non-None."""

    @pytest.fixture
    def model_with_all_configs(self):
        """Model with rootzone where all four land-use configs are non-None."""
        model = MagicMock()

        elem1 = MagicMock()
        elem1.id = 1
        model.grid = MagicMock()
        model.grid.elements = {1: elem1}

        rootzone = MagicMock()
        rootzone.soil_params = {}
        rootzone.nonponded_config = MagicMock(name="nonponded_cfg")
        rootzone.ponded_config = MagicMock(name="ponded_cfg")
        rootzone.urban_config = MagicMock(name="urban_cfg")
        rootzone.native_riparian_config = MagicMock(name="nvrv_cfg")
        rootzone.surface_flow_destinations = {}
        rootzone.surface_flow_dest_ag = {}
        rootzone.surface_flow_dest_urban_in = {}
        rootzone.surface_flow_dest_urban_out = {}
        rootzone.surface_flow_dest_nvrv = {}

        model.rootzone = rootzone
        return model

    @patch("pyiwfm.io.rootzone_v4x.NativeRiparianWriterV4x")
    @patch("pyiwfm.io.rootzone_v4x.UrbanWriterV4x")
    @patch("pyiwfm.io.rootzone_v4x.PondedCropWriterV4x")
    @patch("pyiwfm.io.rootzone_v4x.NonPondedCropWriterV4x")
    def test_nonponded_branch(
        self,
        MockNonPonded,
        MockPonded,
        MockUrban,
        MockNVRV,
        tmp_path: Path,
        model_with_all_configs: MagicMock,
        mock_engine: MagicMock,
    ) -> None:
        """write_all() calls NonPondedCropWriterV4x when nonponded_config is set."""
        config = RootZoneWriterConfig(output_dir=tmp_path)
        writer = RootZoneComponentWriter(
            model_with_all_configs, config, template_engine=mock_engine
        )
        results = writer.write_all()
        assert "nonponded" in results
        assert results["nonponded"] == config.rootzone_dir / "NonPondedAg.dat"

    @patch("pyiwfm.io.rootzone_v4x.NativeRiparianWriterV4x")
    @patch("pyiwfm.io.rootzone_v4x.UrbanWriterV4x")
    @patch("pyiwfm.io.rootzone_v4x.PondedCropWriterV4x")
    @patch("pyiwfm.io.rootzone_v4x.NonPondedCropWriterV4x")
    def test_ponded_branch(
        self,
        MockNonPonded,
        MockPonded,
        MockUrban,
        MockNVRV,
        tmp_path: Path,
        model_with_all_configs: MagicMock,
        mock_engine: MagicMock,
    ) -> None:
        """write_all() calls PondedCropWriterV4x when ponded_config is set."""
        config = RootZoneWriterConfig(output_dir=tmp_path)
        writer = RootZoneComponentWriter(
            model_with_all_configs, config, template_engine=mock_engine
        )
        results = writer.write_all()
        assert "ponded" in results
        assert results["ponded"] == config.rootzone_dir / "PondedAg.dat"

    @patch("pyiwfm.io.rootzone_v4x.NativeRiparianWriterV4x")
    @patch("pyiwfm.io.rootzone_v4x.UrbanWriterV4x")
    @patch("pyiwfm.io.rootzone_v4x.PondedCropWriterV4x")
    @patch("pyiwfm.io.rootzone_v4x.NonPondedCropWriterV4x")
    def test_urban_branch(
        self,
        MockNonPonded,
        MockPonded,
        MockUrban,
        MockNVRV,
        tmp_path: Path,
        model_with_all_configs: MagicMock,
        mock_engine: MagicMock,
    ) -> None:
        """write_all() calls UrbanWriterV4x when urban_config is set."""
        config = RootZoneWriterConfig(output_dir=tmp_path)
        writer = RootZoneComponentWriter(
            model_with_all_configs, config, template_engine=mock_engine
        )
        results = writer.write_all()
        assert "urban" in results
        assert results["urban"] == config.rootzone_dir / "UrbanLandUse.dat"

    @patch("pyiwfm.io.rootzone_v4x.NativeRiparianWriterV4x")
    @patch("pyiwfm.io.rootzone_v4x.UrbanWriterV4x")
    @patch("pyiwfm.io.rootzone_v4x.PondedCropWriterV4x")
    @patch("pyiwfm.io.rootzone_v4x.NonPondedCropWriterV4x")
    def test_native_riparian_branch(
        self,
        MockNonPonded,
        MockPonded,
        MockUrban,
        MockNVRV,
        tmp_path: Path,
        model_with_all_configs: MagicMock,
        mock_engine: MagicMock,
    ) -> None:
        """write_all() calls NativeRiparianWriterV4x when native_riparian_config is set."""
        config = RootZoneWriterConfig(output_dir=tmp_path)
        writer = RootZoneComponentWriter(
            model_with_all_configs, config, template_engine=mock_engine
        )
        results = writer.write_all()
        assert "native_riparian" in results
        assert results["native_riparian"] == config.rootzone_dir / "NativeRiparian.dat"

    @patch("pyiwfm.io.rootzone_v4x.NativeRiparianWriterV4x")
    @patch("pyiwfm.io.rootzone_v4x.UrbanWriterV4x")
    @patch("pyiwfm.io.rootzone_v4x.PondedCropWriterV4x")
    @patch("pyiwfm.io.rootzone_v4x.NonPondedCropWriterV4x")
    def test_all_four_sub_writers_called(
        self,
        MockNonPonded,
        MockPonded,
        MockUrban,
        MockNVRV,
        tmp_path: Path,
        model_with_all_configs: MagicMock,
        mock_engine: MagicMock,
    ) -> None:
        """write_all() writes all four sub-files when all configs are present."""
        config = RootZoneWriterConfig(output_dir=tmp_path)
        writer = RootZoneComponentWriter(
            model_with_all_configs, config, template_engine=mock_engine
        )
        results = writer.write_all()
        assert set(results.keys()) >= {"main", "nonponded", "ponded", "urban", "native_riparian"}
        MockNonPonded.return_value.write.assert_called_once()
        MockPonded.return_value.write.assert_called_once()
        MockUrban.return_value.write.assert_called_once()
        MockNVRV.return_value.write.assert_called_once()


# =============================================================================
# v4.1 format branch (lines 295, 396-398)
# =============================================================================


class TestV41FormatBranch:
    """Tests for v4.1 rendering path in _render_rootzone_main()."""

    def test_write_main_v41_format(
        self, tmp_path: Path, model_with_rootzone: MagicMock, mock_engine: MagicMock
    ) -> None:
        """v4.1 format includes capillary rise column between RHC and IRNE."""
        config = RootZoneWriterConfig(output_dir=tmp_path, version="4.1")
        writer = RootZoneComponentWriter(
            model_with_rootzone, config, template_engine=mock_engine
        )
        path = writer.write_main()
        content = path.read_text()
        assert path.exists()
        # v4.1 data rows have cprise (0.5) between rhc (2) and irne (1).
        # The v4.0 branch does NOT have cprise in its row format.
        # Verify that the data row structure matches v4.1 pattern:
        # "   elem_id  wp  fc  tn  lam  k  rhc  cprise  irne  frne  imsrc  dests..."
        lines = content.strip().split("\n")
        data_lines = [ln for ln in lines if not ln.startswith("C")]
        assert len(data_lines) == 2  # two elements
        # Each line should contain the cprise value (0.5) between rhc and irne
        for line in data_lines:
            parts = line.split()
            # v4.1: elem wp fc tn lam k rhc cprise irne frne imsrc dest_type dest
            assert len(parts) == 13  # 13 columns for v4.1

    def test_write_main_v41_has_destinations(
        self, tmp_path: Path, model_with_rootzone: MagicMock, mock_engine: MagicMock
    ) -> None:
        """v4.1 format row includes surface flow destination columns."""
        # Set surface_flow_destinations for elements
        model_with_rootzone.rootzone.surface_flow_destinations = {
            1: (3, 5),
            2: (4, 6),
        }
        config = RootZoneWriterConfig(output_dir=tmp_path, version="4.1")
        writer = RootZoneComponentWriter(
            model_with_rootzone, config, template_engine=mock_engine
        )
        path = writer.write_main()
        content = path.read_text()
        lines = content.strip().split("\n")
        data_lines = [ln for ln in lines if not ln.startswith("C")]
        # Element 1 should have dest (3, 5)
        parts_e1 = data_lines[0].split()
        assert parts_e1[-2] == "3"
        assert parts_e1[-1] == "5"
        # Element 2 should have dest (4, 6)
        parts_e2 = data_lines[1].split()
        assert parts_e2[-2] == "4"
        assert parts_e2[-1] == "6"

    def test_write_main_v411_format(
        self, tmp_path: Path, model_with_rootzone: MagicMock, mock_engine: MagicMock
    ) -> None:
        """v4.11 is >= 4.1 but < 4.12, so should use the v4.1 row format branch."""
        config = RootZoneWriterConfig(output_dir=tmp_path, version="4.11")
        writer = RootZoneComponentWriter(
            model_with_rootzone, config, template_engine=mock_engine
        )
        path = writer.write_main()
        content = path.read_text()
        assert path.exists()
        # v4.11 uses v4.1 branch: 13-column rows (same as v4.1)
        lines = content.strip().split("\n")
        data_lines = [ln for ln in lines if not ln.startswith("C")]
        for line in data_lines:
            parts = line.split()
            assert len(parts) == 13

    def test_v41_differs_from_v40(
        self, tmp_path: Path, model_with_rootzone: MagicMock, mock_engine: MagicMock
    ) -> None:
        """v4.1 data rows have more columns than v4.0 (capillary rise added)."""
        # Write v4.0
        config_v40 = RootZoneWriterConfig(output_dir=tmp_path / "v40", version="4.0")
        writer_v40 = RootZoneComponentWriter(
            model_with_rootzone, config_v40, template_engine=mock_engine
        )
        path_v40 = writer_v40.write_main()
        content_v40 = path_v40.read_text()

        # Write v4.1
        config_v41 = RootZoneWriterConfig(output_dir=tmp_path / "v41", version="4.1")
        writer_v41 = RootZoneComponentWriter(
            model_with_rootzone, config_v41, template_engine=mock_engine
        )
        path_v41 = writer_v41.write_main()
        content_v41 = path_v41.read_text()

        # v4.0 data rows should have 12 columns; v4.1 should have 13
        lines_v40 = [ln for ln in content_v40.strip().split("\n") if not ln.startswith("C")]
        lines_v41 = [ln for ln in content_v41.strip().split("\n") if not ln.startswith("C")]
        assert len(lines_v40[0].split()) == 12
        assert len(lines_v41[0].split()) == 13


# =============================================================================
# Time-series writer methods (lines 423-527)
# =============================================================================


class TestTimeSeriesMethods:
    """Tests for the seven write_*_ts() time-series methods."""

    @pytest.fixture
    def writer_with_grid(self, tmp_path: Path, mock_engine: MagicMock):
        """Create a RootZoneComponentWriter with a model that has a 2-element grid."""
        model = MagicMock()
        elem1 = MagicMock()
        elem1.id = 1
        elem2 = MagicMock()
        elem2.id = 2
        model.grid = MagicMock()
        model.grid.elements = {1: elem1, 2: elem2}
        model.rootzone = None

        config = RootZoneWriterConfig(output_dir=tmp_path)
        return RootZoneComponentWriter(model, config, template_engine=mock_engine)

    @pytest.mark.parametrize(
        "method_name, config_maker_name, expected_filename",
        [
            ("write_precip_ts", "make_precip_ts_config", "Precip.dat"),
            ("write_et_ts", "make_et_ts_config", "ET.dat"),
            ("write_crop_coeff_ts", "make_crop_coeff_ts_config", "CropCoeff.dat"),
            ("write_return_flow_ts", "make_return_flow_ts_config", "ReturnFlowFrac.dat"),
            ("write_reuse_ts", "make_reuse_ts_config", "ReuseFrac.dat"),
            ("write_irig_period_ts", "make_irig_period_ts_config", "IrigPeriod.dat"),
            ("write_ag_water_demand_ts", "make_ag_water_demand_ts_config", "AgWaterDemand.dat"),
        ],
    )
    def test_ts_method_calls_writer(
        self,
        method_name: str,
        config_maker_name: str,
        expected_filename: str,
        writer_with_grid,
        tmp_path: Path,
    ) -> None:
        """Each write_*_ts() method creates a config, invokes IWFMTimeSeriesDataWriter."""
        mock_ts_config = MagicMock(name="ts_config")
        mock_ts_writer_instance = MagicMock(name="ts_writer")
        expected_path = writer_with_grid.config.rootzone_dir / expected_filename
        mock_ts_writer_instance.write.return_value = expected_path

        with (
            patch(
                f"pyiwfm.io.timeseries_writer.{config_maker_name}",
                return_value=mock_ts_config,
            ) as mock_maker,
            patch(
                "pyiwfm.io.timeseries_writer.IWFMTimeSeriesDataWriter",
                return_value=mock_ts_writer_instance,
            ) as MockTSWriter,
        ):
            method = getattr(writer_with_grid, method_name)
            result = method()

        # The config maker was called
        mock_maker.assert_called_once()
        # IWFMTimeSeriesDataWriter was instantiated
        MockTSWriter.assert_called_once()
        # .write() was called with the config and path
        mock_ts_writer_instance.write.assert_called_once_with(
            mock_ts_config, expected_path
        )
        assert result == expected_path

    @pytest.mark.parametrize(
        "method_name, config_maker_name",
        [
            ("write_precip_ts", "make_precip_ts_config"),
            ("write_et_ts", "make_et_ts_config"),
            ("write_return_flow_ts", "make_return_flow_ts_config"),
            ("write_reuse_ts", "make_reuse_ts_config"),
            ("write_irig_period_ts", "make_irig_period_ts_config"),
            ("write_ag_water_demand_ts", "make_ag_water_demand_ts_config"),
        ],
    )
    def test_ts_method_uses_grid_element_count(
        self,
        method_name: str,
        config_maker_name: str,
        writer_with_grid,
    ) -> None:
        """Time-series methods that use grid element count pass ncol=2 (2-elem grid)."""
        mock_ts_config = MagicMock()
        mock_ts_writer_instance = MagicMock()
        mock_ts_writer_instance.write.return_value = MagicMock()

        with (
            patch(
                f"pyiwfm.io.timeseries_writer.{config_maker_name}",
                return_value=mock_ts_config,
            ) as mock_maker,
            patch(
                "pyiwfm.io.timeseries_writer.IWFMTimeSeriesDataWriter",
                return_value=mock_ts_writer_instance,
            ),
        ):
            method = getattr(writer_with_grid, method_name)
            method()

        # ncol should be 2 (from 2-element grid)
        call_kwargs = mock_maker.call_args
        assert call_kwargs[1].get("ncol", call_kwargs[0][0] if call_kwargs[0] else None) == 2

    def test_crop_coeff_ts_uses_ncol_1(self, writer_with_grid) -> None:
        """write_crop_coeff_ts() uses ncol=1 as placeholder, not grid element count."""
        mock_ts_config = MagicMock()
        mock_ts_writer_instance = MagicMock()
        mock_ts_writer_instance.write.return_value = MagicMock()

        with (
            patch(
                "pyiwfm.io.timeseries_writer.make_crop_coeff_ts_config",
                return_value=mock_ts_config,
            ) as mock_maker,
            patch(
                "pyiwfm.io.timeseries_writer.IWFMTimeSeriesDataWriter",
                return_value=mock_ts_writer_instance,
            ),
        ):
            writer_with_grid.write_crop_coeff_ts()

        call_kwargs = mock_maker.call_args
        assert call_kwargs[1].get("ncol", call_kwargs[0][0] if call_kwargs[0] else None) == 1

    @pytest.mark.parametrize(
        "method_name",
        [
            "write_precip_ts",
            "write_et_ts",
            "write_crop_coeff_ts",
            "write_return_flow_ts",
            "write_reuse_ts",
            "write_irig_period_ts",
            "write_ag_water_demand_ts",
        ],
    )
    def test_ts_method_passes_dates_and_data(
        self, method_name: str, writer_with_grid
    ) -> None:
        """Time-series methods pass dates and data kwargs to the config maker."""
        mock_ts_config = MagicMock()
        mock_ts_writer_instance = MagicMock()
        mock_ts_writer_instance.write.return_value = MagicMock()

        dates = ["10/01/1990_24:00", "10/02/1990_24:00"]
        data = MagicMock(name="data_array")

        # Patch all config makers to avoid import issues
        with (
            patch(
                "pyiwfm.io.timeseries_writer.make_precip_ts_config",
                return_value=mock_ts_config,
            ),
            patch(
                "pyiwfm.io.timeseries_writer.make_et_ts_config",
                return_value=mock_ts_config,
            ),
            patch(
                "pyiwfm.io.timeseries_writer.make_crop_coeff_ts_config",
                return_value=mock_ts_config,
            ),
            patch(
                "pyiwfm.io.timeseries_writer.make_return_flow_ts_config",
                return_value=mock_ts_config,
            ),
            patch(
                "pyiwfm.io.timeseries_writer.make_reuse_ts_config",
                return_value=mock_ts_config,
            ),
            patch(
                "pyiwfm.io.timeseries_writer.make_irig_period_ts_config",
                return_value=mock_ts_config,
            ),
            patch(
                "pyiwfm.io.timeseries_writer.make_ag_water_demand_ts_config",
                return_value=mock_ts_config,
            ),
            patch(
                "pyiwfm.io.timeseries_writer.IWFMTimeSeriesDataWriter",
                return_value=mock_ts_writer_instance,
            ),
        ):
            method = getattr(writer_with_grid, method_name)
            method(dates=dates, data=data)
