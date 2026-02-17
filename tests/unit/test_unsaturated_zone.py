"""Tests for Unsaturated Zone component, reader, and writer."""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pyiwfm.components.unsaturated_zone import (
    AppUnsatZone,
    UnsatZoneElement,
    UnsatZoneLayer,
)
from pyiwfm.io.unsaturated_zone import (
    UnsatZoneElementData,
    UnsatZoneMainConfig,
    UnsatZoneMainReader,
)
from pyiwfm.io.unsaturated_zone_writer import (
    UnsatZoneComponentWriter,
    UnsatZoneWriterConfig,
    write_unsaturated_zone_component,
)

# ---------------------------------------------------------------------------
# Component class tests
# ---------------------------------------------------------------------------


class TestUnsatZoneLayer:
    def test_defaults(self):
        layer = UnsatZoneLayer()
        assert layer.thickness_max == 0.0
        assert layer.kunsat_method == 0


class TestUnsatZoneElement:
    def test_basic(self):
        elem = UnsatZoneElement(
            element_id=1,
            layers=[
                UnsatZoneLayer(thickness_max=10.0, total_porosity=0.35),
                UnsatZoneLayer(thickness_max=20.0, total_porosity=0.30),
            ],
        )
        assert elem.element_id == 1
        assert elem.n_layers == 2

    def test_repr(self):
        elem = UnsatZoneElement(element_id=5, layers=[UnsatZoneLayer()])
        assert "id=5" in repr(elem)


class TestAppUnsatZone:
    def _make_component(self) -> AppUnsatZone:
        comp = AppUnsatZone(n_layers=2)
        for eid in [1, 2, 3]:
            elem = UnsatZoneElement(
                element_id=eid,
                layers=[
                    UnsatZoneLayer(
                        thickness_max=10.0,
                        total_porosity=0.35,
                        lambda_param=0.5,
                        hyd_cond=0.01,
                        kunsat_method=1,
                    ),
                    UnsatZoneLayer(
                        thickness_max=20.0,
                        total_porosity=0.30,
                        lambda_param=0.4,
                        hyd_cond=0.005,
                        kunsat_method=1,
                    ),
                ],
                initial_moisture=np.array([0.2, 0.25]),
            )
            comp.add_element(elem)
        return comp

    def test_add_and_count(self):
        comp = self._make_component()
        assert comp.n_elements == 3

    def test_get_element(self):
        comp = self._make_component()
        elem = comp.get_element(2)
        assert elem.element_id == 2

    def test_iter_elements_sorted(self):
        comp = self._make_component()
        ids = [e.element_id for e in comp.iter_elements()]
        assert ids == [1, 2, 3]

    def test_validate_valid(self):
        comp = self._make_component()
        comp.validate()

    def test_validate_zero_layers(self):
        comp = AppUnsatZone(n_layers=0)
        with pytest.raises(Exception, match="non-positive layer count"):
            comp.validate()

    def test_validate_layer_mismatch(self):
        comp = AppUnsatZone(n_layers=2)
        comp.add_element(UnsatZoneElement(element_id=1, layers=[UnsatZoneLayer()]))
        with pytest.raises(Exception, match="1 layers.*expects 2"):
            comp.validate()

    def test_repr(self):
        comp = self._make_component()
        assert "n_layers=2" in repr(comp)
        assert "n_elements=3" in repr(comp)


class TestFromConfig:
    def test_from_config(self):
        config = UnsatZoneMainConfig(
            version="4.0",
            n_layers=2,
            solver_tolerance=1e-6,
            max_iterations=1000,
            thickness_factor=1.0,
            hyd_cond_factor=1.0,
        )
        config.element_data = [
            UnsatZoneElementData(
                element_id=1,
                thickness_max=np.array([10.0, 20.0]),
                total_porosity=np.array([0.35, 0.30]),
                lambda_param=np.array([0.5, 0.4]),
                hyd_cond=np.array([0.01, 0.005]),
                kunsat_method=np.array([1, 1], dtype=np.int32),
            ),
        ]
        config.initial_soil_moisture = {
            0: np.array([0.2, 0.25]),  # uniform for all
        }

        comp = AppUnsatZone.from_config(config)
        assert comp.n_layers == 2
        assert comp.n_elements == 1
        assert comp.solver_tolerance == 1e-6

        elem = comp.get_element(1)
        assert elem.layers[0].thickness_max == 10.0
        assert elem.layers[1].hyd_cond == 0.005
        assert elem.initial_moisture is not None
        np.testing.assert_array_almost_equal(elem.initial_moisture, [0.2, 0.25])


# ---------------------------------------------------------------------------
# Reader tests
# ---------------------------------------------------------------------------


class TestUnsatZoneReader:
    def test_read_disabled(self, tmp_path):
        content = textwrap.dedent("""\
            #4.0
            C  Number of unsaturated zone layers
                 0                                / NUnsatLayers
        """)
        filepath = tmp_path / "UnsatZone_MAIN.dat"
        filepath.write_text(content)

        reader = UnsatZoneMainReader()
        config = reader.read(filepath)
        assert config.version == "4.0"
        assert config.n_layers == 0

    def test_read_with_layers(self, tmp_path):
        content = textwrap.dedent("""\
            #4.0
            C  Number of layers
                 1                                / NUnsatLayers
            C  Solver tolerance
                 1.0E-06                          / CONVERGENCE
            C  Max iterations
                 1000                             / MAXITER
            C  Budget file
            ../Results/UZBud.hdf                  / UZBUDFL
            C  Zone budget file
            ../Results/UZZBud.hdf                 / UZZBUDFL
            C  Final results file
            ../Results/FinalUZ.out                / FNUZFL
            C  NGROUP
                 0                                / NGROUP
            C  Conversion factors
                 1.0  1.0  1.0                    / FX FThickness FHydCond
            C  Time unit
                 1DAY                             / TUNITK
            C  Element data
                 1  10.0  0.35  0.5  0.01  1
                 2  15.0  0.30  0.4  0.005 1
            C  Initial conditions
                 0  0.20
        """)
        filepath = tmp_path / "UnsatZone_MAIN.dat"
        filepath.write_text(content)

        reader = UnsatZoneMainReader()
        config = reader.read(filepath)

        assert config.n_layers == 1
        assert config.solver_tolerance == 1e-6
        assert config.max_iterations == 1000
        assert len(config.element_data) == 2
        assert config.element_data[0].element_id == 1
        assert 0 in config.initial_soil_moisture


# ---------------------------------------------------------------------------
# Writer tests
# ---------------------------------------------------------------------------


class TestUnsatZoneWriter:
    def test_write_main(self, tmp_path):
        from pyiwfm.io.unsaturated_zone_writer import (
            UnsatZoneComponentWriter,
            UnsatZoneWriterConfig,
        )

        model = MagicMock()
        model.name = "TestModel"

        comp = AppUnsatZone(
            n_layers=1,
            solver_tolerance=1e-6,
            max_iterations=1000,
            thickness_factor=1.0,
            hyd_cond_factor=1.0,
            time_unit="1DAY",
        )
        elem = UnsatZoneElement(
            element_id=1,
            layers=[
                UnsatZoneLayer(
                    thickness_max=10.0,
                    total_porosity=0.35,
                    lambda_param=0.5,
                    hyd_cond=0.01,
                    kunsat_method=1,
                ),
            ],
            initial_moisture=np.array([0.2]),
        )
        comp.add_element(elem)
        model.unsaturated_zone = comp

        config = UnsatZoneWriterConfig(
            output_dir=tmp_path,
            unsatzone_subdir="",
        )

        writer = UnsatZoneComponentWriter(model, config)
        files = writer.write_all()

        assert "main" in files
        main_path = files["main"]
        assert main_path.exists()

        content = main_path.read_text()
        assert "#4.0" in content
        assert "UNSATURATED ZONE" in content.upper()

    def test_write_disabled(self, tmp_path):
        """Write with n_layers=0 should produce a minimal file."""
        from pyiwfm.io.unsaturated_zone_writer import (
            UnsatZoneComponentWriter,
            UnsatZoneWriterConfig,
        )

        model = MagicMock()
        model.name = "TestModel"
        model.unsaturated_zone = AppUnsatZone(n_layers=0)

        config = UnsatZoneWriterConfig(
            output_dir=tmp_path,
            unsatzone_subdir="",
        )

        writer = UnsatZoneComponentWriter(model, config)
        files = writer.write_all()
        assert "main" in files
        content = files["main"].read_text()
        assert "0" in content  # n_layers = 0


# ---------------------------------------------------------------------------
# Round-trip test
# ---------------------------------------------------------------------------


class TestUnsatZoneRoundTrip:
    def test_read_write_read(self, tmp_path):
        """Build config -> component -> write -> re-read."""
        from pyiwfm.io.unsaturated_zone_writer import (
            UnsatZoneComponentWriter,
            UnsatZoneWriterConfig,
        )

        config = UnsatZoneMainConfig(
            version="4.0",
            n_layers=1,
            solver_tolerance=1e-6,
            max_iterations=1000,
            thickness_factor=1.0,
            hyd_cond_factor=1.0,
            time_unit="1DAY",
        )
        config.element_data = [
            UnsatZoneElementData(
                element_id=1,
                thickness_max=np.array([10.0]),
                total_porosity=np.array([0.35]),
                lambda_param=np.array([0.5]),
                hyd_cond=np.array([0.01]),
                kunsat_method=np.array([1], dtype=np.int32),
            ),
        ]
        config.initial_soil_moisture = {1: np.array([0.2])}

        comp = AppUnsatZone.from_config(config)
        assert comp.n_elements == 1

        model = MagicMock()
        model.name = "TestModel"
        model.unsaturated_zone = comp

        writer_config = UnsatZoneWriterConfig(
            output_dir=tmp_path,
            unsatzone_subdir="",
        )
        writer = UnsatZoneComponentWriter(model, writer_config)
        files = writer.write_all()

        reader = UnsatZoneMainReader()
        re_config = reader.read(files["main"])

        assert re_config.n_layers == 1
        assert len(re_config.element_data) == 1
        assert re_config.element_data[0].element_id == 1


# ---------------------------------------------------------------------------
# Additional writer coverage tests
# ---------------------------------------------------------------------------


class TestUnsatZoneWriterConfig:
    def test_unsatzone_dir_empty_subdir(self, tmp_path):
        """Config with empty unsatzone_subdir returns output_dir directly."""
        config = UnsatZoneWriterConfig(
            output_dir=tmp_path,
            unsatzone_subdir="",
        )
        assert config.unsatzone_dir == tmp_path

    def test_unsatzone_dir_with_subdir(self, tmp_path):
        """Config with non-empty unsatzone_subdir returns subdirectory."""
        config = UnsatZoneWriterConfig(
            output_dir=tmp_path,
            unsatzone_subdir="UnsatZone",
        )
        assert config.unsatzone_dir == tmp_path / "UnsatZone"


class TestUnsatZoneWriterCoverage:
    def test_format_property(self, tmp_path):
        """Verify format property returns 'iwfm_unsaturated_zone'."""
        model = MagicMock()
        model.unsaturated_zone = None
        config = UnsatZoneWriterConfig(
            output_dir=tmp_path,
            unsatzone_subdir="",
        )
        writer = UnsatZoneComponentWriter(model, config)
        assert writer.format == "iwfm_unsaturated_zone"

    def test_write_delegates_to_write_all(self, tmp_path):
        """Call write(), verify write_all is called."""
        model = MagicMock()
        model.unsaturated_zone = None
        config = UnsatZoneWriterConfig(
            output_dir=tmp_path,
            unsatzone_subdir="",
        )
        writer = UnsatZoneComponentWriter(model, config)
        with patch.object(writer, "write_all") as mock_write_all:
            writer.write()
            mock_write_all.assert_called_once()

    def test_write_all_no_component_no_defaults(self, tmp_path):
        """Model without unsaturated_zone, write_defaults=False => early return."""
        model = MagicMock()
        model.unsaturated_zone = None
        config = UnsatZoneWriterConfig(
            output_dir=tmp_path,
            unsatzone_subdir="",
        )
        writer = UnsatZoneComponentWriter(model, config)
        results = writer.write_all(write_defaults=False)
        assert results == {}

    def test_render_main_with_elements(self, tmp_path):
        """Component with multiple elements."""
        model = MagicMock()
        comp = AppUnsatZone(
            n_layers=1,
            solver_tolerance=1e-6,
            max_iterations=1000,
            thickness_factor=1.0,
            hyd_cond_factor=1.0,
            time_unit="1DAY",
        )
        for eid in [1, 2, 3]:
            elem = UnsatZoneElement(
                element_id=eid,
                layers=[
                    UnsatZoneLayer(
                        thickness_max=10.0 * eid,
                        total_porosity=0.35,
                        lambda_param=0.5,
                        hyd_cond=0.01,
                        kunsat_method=1,
                    ),
                ],
            )
            comp.add_element(elem)
        model.unsaturated_zone = comp

        config = UnsatZoneWriterConfig(
            output_dir=tmp_path,
            unsatzone_subdir="",
        )
        writer = UnsatZoneComponentWriter(model, config)
        files = writer.write_all()

        assert "main" in files
        content = files["main"].read_text()
        # All three element IDs should appear in the output
        assert "1" in content
        assert "2" in content
        assert "3" in content

    def test_render_main_initial_conditions(self, tmp_path):
        """Element with initial_moisture data (per-element conditions)."""
        model = MagicMock()
        comp = AppUnsatZone(
            n_layers=2,
            solver_tolerance=1e-8,
            max_iterations=2000,
            thickness_factor=1.0,
            hyd_cond_factor=1.0,
            time_unit="1DAY",
        )
        # Element 1 with initial moisture
        elem1 = UnsatZoneElement(
            element_id=1,
            layers=[
                UnsatZoneLayer(
                    thickness_max=10.0,
                    total_porosity=0.35,
                    lambda_param=0.5,
                    hyd_cond=0.01,
                    kunsat_method=1,
                ),
                UnsatZoneLayer(
                    thickness_max=20.0,
                    total_porosity=0.30,
                    lambda_param=0.4,
                    hyd_cond=0.005,
                    kunsat_method=1,
                ),
            ],
            initial_moisture=np.array([0.2, 0.25]),
        )
        # Element 2 with initial moisture
        elem2 = UnsatZoneElement(
            element_id=2,
            layers=[
                UnsatZoneLayer(
                    thickness_max=12.0,
                    total_porosity=0.33,
                    lambda_param=0.5,
                    hyd_cond=0.01,
                    kunsat_method=1,
                ),
                UnsatZoneLayer(
                    thickness_max=18.0,
                    total_porosity=0.28,
                    lambda_param=0.4,
                    hyd_cond=0.005,
                    kunsat_method=1,
                ),
            ],
            initial_moisture=np.array([0.22, 0.27]),
        )
        comp.add_element(elem1)
        comp.add_element(elem2)
        model.unsaturated_zone = comp

        config = UnsatZoneWriterConfig(
            output_dir=tmp_path,
            unsatzone_subdir="",
        )
        writer = UnsatZoneComponentWriter(model, config)
        files = writer.write_all()

        assert "main" in files
        content = files["main"].read_text()
        # Initial moisture values should appear in the output
        assert "0.2" in content

    def test_render_main_uniform_moisture(self, tmp_path):
        """Single element_id=0 with initial moisture => uniform moisture."""
        model = MagicMock()
        comp = AppUnsatZone(
            n_layers=1,
            solver_tolerance=1e-8,
            max_iterations=2000,
            thickness_factor=1.0,
            hyd_cond_factor=1.0,
            time_unit="1DAY",
        )
        # Element with element_id=0 and initial moisture triggers uniform path
        elem = UnsatZoneElement(
            element_id=0,
            layers=[
                UnsatZoneLayer(
                    thickness_max=10.0,
                    total_porosity=0.35,
                    lambda_param=0.5,
                    hyd_cond=0.01,
                    kunsat_method=1,
                ),
            ],
            initial_moisture=np.array([0.30]),
        )
        comp.add_element(elem)
        model.unsaturated_zone = comp

        config = UnsatZoneWriterConfig(
            output_dir=tmp_path,
            unsatzone_subdir="",
        )
        writer = UnsatZoneComponentWriter(model, config)
        files = writer.write_all()

        assert "main" in files
        content = files["main"].read_text()
        # The uniform moisture value should appear
        assert "0.3" in content

    def test_render_main_component_files(self, tmp_path):
        """Component with budget_file, zbudget_file, final_results_file set."""
        model = MagicMock()
        comp = AppUnsatZone(
            n_layers=1,
            solver_tolerance=1e-8,
            max_iterations=2000,
            thickness_factor=1.0,
            hyd_cond_factor=1.0,
            time_unit="1DAY",
            budget_file="../CustomResults/UZBud.hdf",
            zbudget_file="../CustomResults/UZZBud.hdf",
            final_results_file="../CustomResults/FinalUZ.out",
        )
        elem = UnsatZoneElement(
            element_id=1,
            layers=[
                UnsatZoneLayer(
                    thickness_max=10.0,
                    total_porosity=0.35,
                    lambda_param=0.5,
                    hyd_cond=0.01,
                    kunsat_method=1,
                ),
            ],
        )
        comp.add_element(elem)
        model.unsaturated_zone = comp

        config = UnsatZoneWriterConfig(
            output_dir=tmp_path,
            unsatzone_subdir="",
        )
        writer = UnsatZoneComponentWriter(model, config)
        files = writer.write_all()

        content = files["main"].read_text()
        # The custom output files from the component should override config defaults
        assert "CustomResults/UZBud.hdf" in content
        assert "CustomResults/UZZBud.hdf" in content
        assert "CustomResults/FinalUZ.out" in content

    def test_write_uz_convenience_no_config(self, tmp_path):
        """write_unsaturated_zone_component() with config=None."""
        model = MagicMock()
        comp = AppUnsatZone(
            n_layers=1,
            solver_tolerance=1e-8,
            max_iterations=2000,
            thickness_factor=1.0,
            hyd_cond_factor=1.0,
            time_unit="1DAY",
        )
        elem = UnsatZoneElement(
            element_id=1,
            layers=[
                UnsatZoneLayer(
                    thickness_max=10.0,
                    total_porosity=0.35,
                    lambda_param=0.5,
                    hyd_cond=0.01,
                    kunsat_method=1,
                ),
            ],
        )
        comp.add_element(elem)
        model.unsaturated_zone = comp

        results = write_unsaturated_zone_component(model, tmp_path, config=None)
        assert "main" in results
        assert results["main"].exists()

    def test_write_uz_convenience_with_config(self, tmp_path):
        """write_unsaturated_zone_component() with custom config."""
        model = MagicMock()
        comp = AppUnsatZone(
            n_layers=1,
            solver_tolerance=1e-8,
            max_iterations=2000,
            thickness_factor=1.0,
            hyd_cond_factor=1.0,
            time_unit="1DAY",
        )
        elem = UnsatZoneElement(
            element_id=1,
            layers=[
                UnsatZoneLayer(
                    thickness_max=10.0,
                    total_porosity=0.35,
                    lambda_param=0.5,
                    hyd_cond=0.01,
                    kunsat_method=1,
                ),
            ],
        )
        comp.add_element(elem)
        model.unsaturated_zone = comp

        custom_config = UnsatZoneWriterConfig(
            output_dir=Path("/dummy"),  # Will be overridden
            unsatzone_subdir="CustomUZ",
        )
        results = write_unsaturated_zone_component(model, tmp_path, config=custom_config)
        assert "main" in results
        # Verify the config's output_dir was overridden to tmp_path
        assert custom_config.output_dir == tmp_path
        assert results["main"].exists()
