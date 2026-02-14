"""Coverage tests for pyiwfm.io.unsaturated_zone_writer module.

Targets uncovered lines and branches including:
- UnsatZoneWriterConfig.unsatzone_dir with empty subdir (line 57)
- UnsatZoneComponentWriter.format property (line 93)
- UnsatZoneComponentWriter.write() delegates to write_all() (line 97)
- write_all() with no component and write_defaults=False (lines 122-125)
- _render_main() with elements via iter_elements() (lines 161+)
- _render_main() initial moisture conditions (lines 187, 198-199)
- _render_main() component-level file overrides (lines 204-210)
- write_unsaturated_zone_component() convenience function (lines 257-265)
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pyiwfm.io.unsaturated_zone_writer import (
    UnsatZoneComponentWriter,
    UnsatZoneWriterConfig,
    write_unsaturated_zone_component,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_engine():
    """Create a mock TemplateEngine that returns predictable content."""
    engine = MagicMock()
    engine.render_template.return_value = "C  MOCK UNSATZONE HEADER\n"
    engine.render_string.return_value = "C  MOCK UNSATZONE STRING\n"
    return engine


@pytest.fixture
def bare_model():
    """Create a model with unsaturated_zone=None."""
    model = MagicMock()
    model.unsaturated_zone = None
    return model


@pytest.fixture
def model_with_uz():
    """Create a model with an unsaturated zone component containing elements."""
    model = MagicMock()

    # Build layers for element 1
    layer1_e1 = SimpleNamespace(
        thickness_max=10.0,
        total_porosity=0.40,
        lambda_param=0.5,
        hyd_cond=2.0,
        kunsat_method=1,
    )
    layer2_e1 = SimpleNamespace(
        thickness_max=20.0,
        total_porosity=0.35,
        lambda_param=0.6,
        hyd_cond=1.5,
        kunsat_method=2,
    )

    elem1 = SimpleNamespace(
        element_id=1,
        layers=[layer1_e1, layer2_e1],
        initial_moisture=None,
    )

    # Build layers for element 2
    layer1_e2 = SimpleNamespace(
        thickness_max=15.0,
        total_porosity=0.38,
        lambda_param=0.55,
        hyd_cond=1.8,
        kunsat_method=1,
    )
    layer2_e2 = SimpleNamespace(
        thickness_max=25.0,
        total_porosity=0.32,
        lambda_param=0.65,
        hyd_cond=1.2,
        kunsat_method=2,
    )

    elem2 = SimpleNamespace(
        element_id=2,
        layers=[layer1_e2, layer2_e2],
        initial_moisture=None,
    )

    uz = MagicMock()
    uz.n_layers = 2
    uz.n_elements = 2
    uz.iter_elements.return_value = [elem1, elem2]
    uz.thickness_factor = 1.0
    uz.hyd_cond_factor = 1.0
    uz.solver_tolerance = 1e-8
    uz.max_iterations = 2000
    uz.n_parametric_grids = 0
    uz.coord_factor = 1.0
    uz.time_unit = "1DAY"
    uz.budget_file = ""
    uz.zbudget_file = ""
    uz.final_results_file = ""

    model.unsaturated_zone = uz
    return model


# =============================================================================
# UnsatZoneWriterConfig Tests
# =============================================================================


class TestUnsatZoneWriterConfig:
    """Tests for UnsatZoneWriterConfig properties."""

    def test_unsatzone_dir_default(self, tmp_path: Path) -> None:
        """Config with default subdir returns output_dir / 'UnsatZone'."""
        config = UnsatZoneWriterConfig(output_dir=tmp_path)
        assert config.unsatzone_dir == tmp_path / "UnsatZone"

    def test_unsatzone_dir_empty_subdir(self, tmp_path: Path) -> None:
        """Config with empty/falsy unsatzone_subdir returns output_dir directly.

        Covers line 57 (the falsy branch of the if self.unsatzone_subdir check).
        """
        config = UnsatZoneWriterConfig(output_dir=tmp_path, unsatzone_subdir="")
        assert config.unsatzone_dir == tmp_path

    def test_main_path(self, tmp_path: Path) -> None:
        """main_path uses unsatzone_dir plus main_file."""
        config = UnsatZoneWriterConfig(output_dir=tmp_path)
        assert config.main_path == tmp_path / "UnsatZone" / "UnsatZone_MAIN.dat"


# =============================================================================
# format Property Test
# =============================================================================


class TestUnsatZoneFormatProperty:
    """Test for UnsatZoneComponentWriter.format property."""

    def test_format_property(
        self, tmp_path: Path, bare_model: MagicMock, mock_engine: MagicMock
    ) -> None:
        """format property returns 'iwfm_unsaturated_zone'.

        Covers line 93.
        """
        config = UnsatZoneWriterConfig(output_dir=tmp_path)
        writer = UnsatZoneComponentWriter(
            bare_model, config, template_engine=mock_engine
        )
        assert writer.format == "iwfm_unsaturated_zone"


# =============================================================================
# write() Delegation Test
# =============================================================================


class TestUnsatZoneWriteDelegation:
    """Test that write() delegates to write_all()."""

    def test_write_delegates_to_write_all(
        self, tmp_path: Path, bare_model: MagicMock, mock_engine: MagicMock
    ) -> None:
        """write() calls write_all() internally.

        Covers line 97.
        """
        config = UnsatZoneWriterConfig(output_dir=tmp_path)
        writer = UnsatZoneComponentWriter(
            bare_model, config, template_engine=mock_engine
        )
        with patch.object(writer, "write_all") as mock_write_all:
            writer.write()
            mock_write_all.assert_called_once()


# =============================================================================
# write_all() Tests
# =============================================================================


class TestUnsatZoneWriteAll:
    """Tests for UnsatZoneComponentWriter.write_all()."""

    def test_write_all_no_component_no_defaults(
        self, tmp_path: Path, bare_model: MagicMock, mock_engine: MagicMock
    ) -> None:
        """write_all(write_defaults=False) returns empty when no uz component.

        Covers lines 122-125.
        """
        config = UnsatZoneWriterConfig(output_dir=tmp_path)
        writer = UnsatZoneComponentWriter(
            bare_model, config, template_engine=mock_engine
        )
        results = writer.write_all(write_defaults=False)
        assert results == {}

    def test_write_all_no_component_defaults_true(
        self, tmp_path: Path, bare_model: MagicMock, mock_engine: MagicMock
    ) -> None:
        """write_all(write_defaults=True) writes main file even without uz component."""
        config = UnsatZoneWriterConfig(output_dir=tmp_path)
        writer = UnsatZoneComponentWriter(
            bare_model, config, template_engine=mock_engine
        )
        results = writer.write_all(write_defaults=True)
        assert "main" in results
        assert results["main"].exists()

    def test_write_all_with_component(
        self, tmp_path: Path, model_with_uz: MagicMock, mock_engine: MagicMock
    ) -> None:
        """write_all() with uz component writes main file."""
        config = UnsatZoneWriterConfig(output_dir=tmp_path)
        writer = UnsatZoneComponentWriter(
            model_with_uz, config, template_engine=mock_engine
        )
        results = writer.write_all()
        assert "main" in results
        assert results["main"].exists()


# =============================================================================
# _render_main() Tests
# =============================================================================


class TestRenderMain:
    """Tests for UnsatZoneComponentWriter._render_main()."""

    def test_render_main_with_elements(
        self, tmp_path: Path, model_with_uz: MagicMock, mock_engine: MagicMock
    ) -> None:
        """_render_main() with multiple elements calls iter_elements() and passes
        element_data to the template.

        Covers the loop at line 161+ (iter_elements branch).
        """
        config = UnsatZoneWriterConfig(output_dir=tmp_path)
        writer = UnsatZoneComponentWriter(
            model_with_uz, config, template_engine=mock_engine
        )
        uz = model_with_uz.unsaturated_zone

        writer._render_main(uz)

        # Verify the template engine was called with element_data
        call_kwargs = mock_engine.render_template.call_args[1]
        element_data = call_kwargs["element_data"]
        assert len(element_data) == 2
        assert element_data[0]["element_id"] == 1
        assert element_data[1]["element_id"] == 2
        # Each element should have 2 layers
        assert len(element_data[0]["layers"]) == 2
        assert len(element_data[1]["layers"]) == 2

    def test_render_main_initial_conditions(
        self, tmp_path: Path, model_with_uz: MagicMock, mock_engine: MagicMock
    ) -> None:
        """_render_main() with initial_moisture on elements populates
        initial_conditions list.

        Covers line 187 (if elem.initial_moisture is not None branch).
        """
        config = UnsatZoneWriterConfig(output_dir=tmp_path)
        writer = UnsatZoneComponentWriter(
            model_with_uz, config, template_engine=mock_engine
        )

        uz = model_with_uz.unsaturated_zone

        # Set initial moisture on both elements
        elems = list(uz.iter_elements.return_value)
        elems[0].initial_moisture = np.array([0.20, 0.25])
        elems[1].initial_moisture = np.array([0.18, 0.22])
        uz.iter_elements.return_value = elems

        writer._render_main(uz)

        call_kwargs = mock_engine.render_template.call_args[1]
        initial_conditions = call_kwargs["initial_conditions"]
        assert len(initial_conditions) == 2
        assert initial_conditions[0]["element_id"] == 1
        assert initial_conditions[0]["moisture"] == [0.20, 0.25]
        assert initial_conditions[1]["element_id"] == 2
        assert initial_conditions[1]["moisture"] == [0.18, 0.22]
        # uniform_moisture should remain None when multiple elements
        assert call_kwargs["uniform_moisture"] is None

    def test_render_main_uniform_moisture(
        self, tmp_path: Path, mock_engine: MagicMock
    ) -> None:
        """_render_main() with a single element_id=0 having initial moisture
        triggers the uniform_moisture path.

        Covers lines 198-199 (uniform_moisture = initial_conditions[0]["moisture"]).
        """
        model = MagicMock()

        # Build a single element with element_id=0 and initial moisture
        layer = SimpleNamespace(
            thickness_max=10.0,
            total_porosity=0.40,
            lambda_param=0.5,
            hyd_cond=2.0,
            kunsat_method=1,
        )
        elem = SimpleNamespace(
            element_id=0,
            layers=[layer],
            initial_moisture=np.array([0.30]),
        )

        uz = MagicMock()
        uz.n_layers = 1
        uz.n_elements = 1
        uz.iter_elements.return_value = [elem]
        uz.thickness_factor = 1.0
        uz.hyd_cond_factor = 1.0
        uz.solver_tolerance = 1e-8
        uz.max_iterations = 2000
        uz.n_parametric_grids = 0
        uz.coord_factor = 1.0
        uz.time_unit = "1DAY"
        uz.budget_file = ""
        uz.zbudget_file = ""
        uz.final_results_file = ""

        model.unsaturated_zone = uz

        config = UnsatZoneWriterConfig(output_dir=tmp_path)
        writer = UnsatZoneComponentWriter(
            model, config, template_engine=mock_engine
        )

        writer._render_main(uz)

        call_kwargs = mock_engine.render_template.call_args[1]
        # uniform_moisture should be set (list from tolist)
        assert call_kwargs["uniform_moisture"] == [0.30]
        # initial_conditions should be emptied
        assert call_kwargs["initial_conditions"] == []

    def test_render_main_component_files(
        self, tmp_path: Path, model_with_uz: MagicMock, mock_engine: MagicMock
    ) -> None:
        """_render_main() overrides config file paths with component-level
        budget_file, zbudget_file, and final_results_file when they are truthy.

        Covers lines 204->212, 206, 208, 210.
        """
        config = UnsatZoneWriterConfig(output_dir=tmp_path)
        writer = UnsatZoneComponentWriter(
            model_with_uz, config, template_engine=mock_engine
        )

        uz = model_with_uz.unsaturated_zone
        # Set component-level file overrides
        uz.budget_file = "../Results/CustomBud.hdf"
        uz.zbudget_file = "../Results/CustomZBud.hdf"
        uz.final_results_file = "../Results/CustomFinal.out"

        writer._render_main(uz)

        call_kwargs = mock_engine.render_template.call_args[1]
        assert call_kwargs["budget_file"] == "../Results/CustomBud.hdf"
        assert call_kwargs["zbudget_file"] == "../Results/CustomZBud.hdf"
        assert call_kwargs["final_results_file"] == "../Results/CustomFinal.out"

    def test_render_main_no_component(
        self, tmp_path: Path, bare_model: MagicMock, mock_engine: MagicMock
    ) -> None:
        """_render_main(None) uses config defaults for all values."""
        config = UnsatZoneWriterConfig(output_dir=tmp_path)
        writer = UnsatZoneComponentWriter(
            bare_model, config, template_engine=mock_engine
        )

        writer._render_main(None)

        call_kwargs = mock_engine.render_template.call_args[1]
        assert call_kwargs["n_layers"] == 0
        assert call_kwargs["solver_tolerance"] == 1e-8
        assert call_kwargs["max_iterations"] == 2000
        assert call_kwargs["element_data"] == []
        assert call_kwargs["uniform_moisture"] is None
        assert call_kwargs["initial_conditions"] == []
        # Should use config defaults for file paths
        assert call_kwargs["budget_file"] == config.budget_file
        assert call_kwargs["zbudget_file"] == config.zbudget_file
        assert call_kwargs["final_results_file"] == config.final_results_file

    def test_render_main_thickness_and_hyd_cond_factors(
        self, tmp_path: Path, model_with_uz: MagicMock, mock_engine: MagicMock
    ) -> None:
        """_render_main() applies reverse conversion factors for thickness and hyd_cond."""
        config = UnsatZoneWriterConfig(output_dir=tmp_path)
        writer = UnsatZoneComponentWriter(
            model_with_uz, config, template_engine=mock_engine
        )

        uz = model_with_uz.unsaturated_zone
        uz.thickness_factor = 2.0
        uz.hyd_cond_factor = 0.5

        writer._render_main(uz)

        call_kwargs = mock_engine.render_template.call_args[1]
        elem0_layer0 = call_kwargs["element_data"][0]["layers"][0]
        # thickness_max=10.0, factor=2.0 => raw = 10.0 / 2.0 = 5.0
        assert elem0_layer0["thickness_max"] == pytest.approx(5.0)
        # hyd_cond=2.0, factor=0.5 => raw = 2.0 / 0.5 = 4.0
        assert elem0_layer0["hyd_cond"] == pytest.approx(4.0)

    def test_render_main_zero_factors_use_raw(
        self, tmp_path: Path, model_with_uz: MagicMock, mock_engine: MagicMock
    ) -> None:
        """When thickness_factor or hyd_cond_factor is 0 (falsy), raw values are used."""
        config = UnsatZoneWriterConfig(output_dir=tmp_path)
        writer = UnsatZoneComponentWriter(
            model_with_uz, config, template_engine=mock_engine
        )

        uz = model_with_uz.unsaturated_zone
        uz.thickness_factor = 0
        uz.hyd_cond_factor = 0

        writer._render_main(uz)

        call_kwargs = mock_engine.render_template.call_args[1]
        elem0_layer0 = call_kwargs["element_data"][0]["layers"][0]
        # With factor=0, should use raw values directly
        assert elem0_layer0["thickness_max"] == 10.0
        assert elem0_layer0["hyd_cond"] == 2.0


# =============================================================================
# write_unsaturated_zone_component() Convenience Function Tests
# =============================================================================


class TestWriteUnsatZoneConvenience:
    """Tests for the write_unsaturated_zone_component() convenience function."""

    def test_write_uz_convenience_no_config(
        self, tmp_path: Path, bare_model: MagicMock, mock_engine: MagicMock
    ) -> None:
        """write_unsaturated_zone_component() with config=None creates default config.

        Covers lines 257-260, 264-265.
        """
        with patch(
            "pyiwfm.io.unsaturated_zone_writer.TemplateEngine",
            return_value=mock_engine,
        ):
            results = write_unsaturated_zone_component(bare_model, tmp_path)
        assert "main" in results

    def test_write_uz_convenience_with_config(
        self, tmp_path: Path, bare_model: MagicMock, mock_engine: MagicMock
    ) -> None:
        """write_unsaturated_zone_component() with custom config updates output_dir.

        Covers lines 261-262.
        """
        config = UnsatZoneWriterConfig(
            output_dir=tmp_path, version="5.0"
        )
        new_dir = tmp_path / "custom_out"
        new_dir.mkdir()
        with patch(
            "pyiwfm.io.unsaturated_zone_writer.TemplateEngine",
            return_value=mock_engine,
        ):
            results = write_unsaturated_zone_component(
                bare_model, new_dir, config=config
            )
        # output_dir should have been updated on the config
        assert config.output_dir == new_dir
        assert "main" in results

    def test_write_uz_convenience_string_path(
        self, tmp_path: Path, bare_model: MagicMock, mock_engine: MagicMock
    ) -> None:
        """write_unsaturated_zone_component() accepts string path and converts to Path.

        Covers line 257 (output_dir = Path(output_dir)).
        """
        with patch(
            "pyiwfm.io.unsaturated_zone_writer.TemplateEngine",
            return_value=mock_engine,
        ):
            results = write_unsaturated_zone_component(
                bare_model, str(tmp_path)
            )
        assert "main" in results
