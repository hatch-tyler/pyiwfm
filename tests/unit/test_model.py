"""Unit tests for IWFMModel class.

Tests:
- IWFMModel dataclass basics
- Property methods
- Boolean has_* properties
- validate method
- summary method
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock

import numpy as np
import pytest

from pyiwfm.core.model import IWFMModel
from pyiwfm.core.exceptions import ValidationError


# =============================================================================
# Test IWFMModel Basics
# =============================================================================


class TestIWFMModelBasics:
    """Tests for IWFMModel dataclass basics."""

    def test_basic_creation(self) -> None:
        """Test basic model creation with just a name."""
        model = IWFMModel(name="TestModel")

        assert model.name == "TestModel"
        assert model.mesh is None
        assert model.stratigraphy is None
        assert model.groundwater is None
        assert model.streams is None
        assert model.lakes is None
        assert model.rootzone is None
        assert model.metadata == {}

    def test_metadata_default_factory(self) -> None:
        """Test that metadata uses independent default dict."""
        model1 = IWFMModel(name="Model1")
        model2 = IWFMModel(name="Model2")

        model1.metadata["key"] = "value"

        assert "key" not in model2.metadata

    def test_repr(self) -> None:
        """Test __repr__ method."""
        model = IWFMModel(name="TestModel")
        repr_str = repr(model)

        assert "IWFMModel" in repr_str
        assert "TestModel" in repr_str
        assert "n_nodes=0" in repr_str
        assert "n_elements=0" in repr_str
        assert "n_layers=0" in repr_str


# =============================================================================
# Test Properties - No Components
# =============================================================================


class TestPropertiesNoComponents:
    """Tests for properties when components are None."""

    def test_n_nodes_no_mesh(self) -> None:
        """Test n_nodes returns 0 when mesh is None."""
        model = IWFMModel(name="Test")
        assert model.n_nodes == 0

    def test_n_elements_no_mesh(self) -> None:
        """Test n_elements returns 0 when mesh is None."""
        model = IWFMModel(name="Test")
        assert model.n_elements == 0

    def test_n_layers_no_stratigraphy(self) -> None:
        """Test n_layers returns 0 when stratigraphy is None."""
        model = IWFMModel(name="Test")
        assert model.n_layers == 0

    def test_grid_property_alias(self) -> None:
        """Test grid property is alias for mesh."""
        model = IWFMModel(name="Test")
        assert model.grid is None

        mock_mesh = MagicMock()
        model.grid = mock_mesh
        assert model.mesh is mock_mesh
        assert model.grid is mock_mesh

    def test_n_wells_no_groundwater(self) -> None:
        """Test n_wells returns 0 when groundwater is None."""
        model = IWFMModel(name="Test")
        assert model.n_wells == 0

    def test_n_stream_nodes_no_streams(self) -> None:
        """Test n_stream_nodes returns 0 when streams is None."""
        model = IWFMModel(name="Test")
        assert model.n_stream_nodes == 0

    def test_n_stream_reaches_no_streams(self) -> None:
        """Test n_stream_reaches returns 0 when streams is None."""
        model = IWFMModel(name="Test")
        assert model.n_stream_reaches == 0

    def test_n_diversions_no_streams(self) -> None:
        """Test n_diversions returns 0 when streams is None."""
        model = IWFMModel(name="Test")
        assert model.n_diversions == 0

    def test_n_lakes_no_lakes(self) -> None:
        """Test n_lakes returns 0 when lakes is None."""
        model = IWFMModel(name="Test")
        assert model.n_lakes == 0

    def test_n_crop_types_no_rootzone(self) -> None:
        """Test n_crop_types returns 0 when rootzone is None."""
        model = IWFMModel(name="Test")
        assert model.n_crop_types == 0


# =============================================================================
# Test has_* Boolean Properties
# =============================================================================


class TestHasProperties:
    """Tests for has_* boolean properties."""

    def test_has_groundwater_false(self) -> None:
        """Test has_groundwater returns False when None."""
        model = IWFMModel(name="Test")
        assert model.has_groundwater is False

    def test_has_groundwater_true(self) -> None:
        """Test has_groundwater returns True when set."""
        model = IWFMModel(name="Test")
        model.groundwater = MagicMock()
        assert model.has_groundwater is True

    def test_has_streams_false(self) -> None:
        """Test has_streams returns False when None."""
        model = IWFMModel(name="Test")
        assert model.has_streams is False

    def test_has_streams_true(self) -> None:
        """Test has_streams returns True when set."""
        model = IWFMModel(name="Test")
        model.streams = MagicMock()
        assert model.has_streams is True

    def test_has_lakes_false(self) -> None:
        """Test has_lakes returns False when None."""
        model = IWFMModel(name="Test")
        assert model.has_lakes is False

    def test_has_lakes_true(self) -> None:
        """Test has_lakes returns True when set."""
        model = IWFMModel(name="Test")
        model.lakes = MagicMock()
        assert model.has_lakes is True

    def test_has_rootzone_false(self) -> None:
        """Test has_rootzone returns False when None."""
        model = IWFMModel(name="Test")
        assert model.has_rootzone is False

    def test_has_rootzone_true(self) -> None:
        """Test has_rootzone returns True when set."""
        model = IWFMModel(name="Test")
        model.rootzone = MagicMock()
        assert model.has_rootzone is True


# =============================================================================
# Test Properties with Mock Components
# =============================================================================


class TestPropertiesWithMocks:
    """Tests for properties with mock components."""

    def test_n_nodes_with_mesh(self) -> None:
        """Test n_nodes with mock mesh."""
        model = IWFMModel(name="Test")
        mock_mesh = MagicMock()
        mock_mesh.n_nodes = 100
        model.mesh = mock_mesh

        assert model.n_nodes == 100

    def test_n_elements_with_mesh(self) -> None:
        """Test n_elements with mock mesh."""
        model = IWFMModel(name="Test")
        mock_mesh = MagicMock()
        mock_mesh.n_elements = 50
        model.mesh = mock_mesh

        assert model.n_elements == 50

    def test_n_layers_with_stratigraphy(self) -> None:
        """Test n_layers with mock stratigraphy."""
        model = IWFMModel(name="Test")
        mock_strat = MagicMock()
        mock_strat.n_layers = 4
        model.stratigraphy = mock_strat

        assert model.n_layers == 4

    def test_n_wells_with_groundwater(self) -> None:
        """Test n_wells with mock groundwater."""
        model = IWFMModel(name="Test")
        mock_gw = MagicMock()
        mock_gw.n_wells = 25
        model.groundwater = mock_gw

        assert model.n_wells == 25

    def test_n_stream_nodes_with_streams(self) -> None:
        """Test n_stream_nodes with mock streams."""
        model = IWFMModel(name="Test")
        mock_streams = MagicMock()
        mock_streams.n_nodes = 200
        model.streams = mock_streams

        assert model.n_stream_nodes == 200

    def test_n_stream_reaches_with_streams(self) -> None:
        """Test n_stream_reaches with mock streams."""
        model = IWFMModel(name="Test")
        mock_streams = MagicMock()
        mock_streams.n_reaches = 15
        model.streams = mock_streams

        assert model.n_stream_reaches == 15

    def test_n_diversions_with_streams(self) -> None:
        """Test n_diversions with mock streams."""
        model = IWFMModel(name="Test")
        mock_streams = MagicMock()
        mock_streams.n_diversions = 10
        model.streams = mock_streams

        assert model.n_diversions == 10

    def test_n_lakes_with_lakes(self) -> None:
        """Test n_lakes with mock lakes."""
        model = IWFMModel(name="Test")
        mock_lakes = MagicMock()
        mock_lakes.n_lakes = 3
        model.lakes = mock_lakes

        assert model.n_lakes == 3

    def test_n_crop_types_with_rootzone(self) -> None:
        """Test n_crop_types with mock rootzone."""
        model = IWFMModel(name="Test")
        mock_rz = MagicMock()
        mock_rz.n_crop_types = 12
        model.rootzone = mock_rz

        assert model.n_crop_types == 12


# =============================================================================
# Test Validate Method
# =============================================================================


class TestValidate:
    """Tests for validate method."""

    def test_validate_no_mesh_raises_error(self) -> None:
        """Test that validate raises error when mesh is None."""
        model = IWFMModel(name="Test")
        model.stratigraphy = MagicMock()
        model.stratigraphy.validate.return_value = []
        model.stratigraphy.n_nodes = 0

        with pytest.raises(ValidationError) as excinfo:
            model.validate()

        # Check that 'no mesh' is in one of the error messages
        assert any("no mesh" in str(e) for e in excinfo.value.errors)

    def test_validate_no_stratigraphy_raises_error(self) -> None:
        """Test that validate raises error when stratigraphy is None."""
        model = IWFMModel(name="Test")
        mock_mesh = MagicMock()
        mock_mesh.validate.return_value = None
        mock_mesh.n_nodes = 10
        model.mesh = mock_mesh

        with pytest.raises(ValidationError) as excinfo:
            model.validate()

        # Check that 'no stratigraphy' is in one of the error messages
        assert any("no stratigraphy" in str(e) for e in excinfo.value.errors)

    def test_validate_node_count_mismatch_raises_error(self) -> None:
        """Test that validate raises error when node counts mismatch."""
        model = IWFMModel(name="Test")

        mock_mesh = MagicMock()
        mock_mesh.validate.return_value = None
        mock_mesh.n_nodes = 100
        model.mesh = mock_mesh

        mock_strat = MagicMock()
        mock_strat.validate.return_value = []
        mock_strat.n_nodes = 50  # Different from mesh
        model.stratigraphy = mock_strat

        with pytest.raises(ValidationError) as excinfo:
            model.validate()

        # Check that 'mismatch' is in one of the error messages
        assert any("mismatch" in str(e) for e in excinfo.value.errors)

    def test_validate_success(self) -> None:
        """Test successful validation."""
        model = IWFMModel(name="Test")

        mock_mesh = MagicMock()
        mock_mesh.validate.return_value = None
        mock_mesh.n_nodes = 100
        model.mesh = mock_mesh

        mock_strat = MagicMock()
        mock_strat.validate.return_value = []
        mock_strat.n_nodes = 100  # Same as mesh
        model.stratigraphy = mock_strat

        errors = model.validate()
        assert errors == []

    def test_validate_with_mesh_error(self) -> None:
        """Test validation when mesh validation fails."""
        model = IWFMModel(name="Test")

        mock_mesh = MagicMock()
        mock_mesh.validate.side_effect = Exception("Mesh error")
        mock_mesh.n_nodes = 100
        model.mesh = mock_mesh

        mock_strat = MagicMock()
        mock_strat.validate.return_value = []
        mock_strat.n_nodes = 100
        model.stratigraphy = mock_strat

        with pytest.raises(ValidationError) as excinfo:
            model.validate()

        # Check that mesh error is in the error messages
        assert any("Mesh validation failed" in str(e) for e in excinfo.value.errors)


# =============================================================================
# Test Summary Method
# =============================================================================


class TestSummary:
    """Tests for summary method."""

    def test_summary_empty_model(self) -> None:
        """Test summary for empty model."""
        model = IWFMModel(name="EmptyModel")
        summary = model.summary()

        assert "EmptyModel" in summary
        assert "Nodes: 0" in summary
        assert "Elements: 0" in summary
        assert "Layers: 0" in summary
        assert "Not loaded" in summary

    def test_summary_with_mesh(self) -> None:
        """Test summary with mesh."""
        model = IWFMModel(name="MeshModel")
        mock_mesh = MagicMock()
        mock_mesh.n_nodes = 100
        mock_mesh.n_elements = 50
        mock_mesh.n_subregions = 3
        model.mesh = mock_mesh

        summary = model.summary()

        assert "Nodes: 100" in summary
        assert "Elements: 50" in summary
        assert "Subregions: 3" in summary

    def test_summary_with_groundwater(self) -> None:
        """Test summary with groundwater component."""
        model = IWFMModel(name="GWModel")

        mock_gw = MagicMock()
        mock_gw.n_wells = 25
        mock_gw.n_boundary_conditions = 10
        mock_gw.n_tile_drains = 5
        mock_gw.aquifer_params = None
        model.groundwater = mock_gw

        summary = model.summary()

        assert "Wells: 25" in summary
        assert "Boundary Conditions: 10" in summary
        assert "Tile Drains: 5" in summary

    def test_summary_with_streams(self) -> None:
        """Test summary with stream component."""
        model = IWFMModel(name="StreamModel")

        mock_streams = MagicMock()
        mock_streams.n_nodes = 200
        mock_streams.n_reaches = 15
        mock_streams.n_diversions = 10
        mock_streams.n_bypasses = 3
        model.streams = mock_streams

        summary = model.summary()

        assert "Stream Nodes: 200" in summary
        assert "Reaches: 15" in summary
        assert "Diversions: 10" in summary
        assert "Bypasses: 3" in summary

    def test_summary_with_lakes(self) -> None:
        """Test summary with lake component."""
        model = IWFMModel(name="LakeModel")

        mock_lakes = MagicMock()
        mock_lakes.n_lakes = 3
        mock_lakes.n_lake_elements = 150
        model.lakes = mock_lakes

        summary = model.summary()

        assert "Lakes: 3" in summary
        assert "Lake Elements: 150" in summary

    def test_summary_with_rootzone(self) -> None:
        """Test summary with root zone component."""
        model = IWFMModel(name="RZModel")

        mock_rz = MagicMock()
        mock_rz.n_crop_types = 12
        mock_rz.element_landuse = {1: "ag", 2: "urban"}
        mock_rz.soil_params = {1: "sandy", 2: "clay"}
        model.rootzone = mock_rz

        summary = model.summary()

        assert "Crop Types: 12" in summary
        assert "Land Use Assignments: 2" in summary
        assert "Soil Parameter Sets: 2" in summary

    def test_summary_source_metadata(self) -> None:
        """Test summary includes source metadata."""
        model = IWFMModel(name="SourceModel")
        model.metadata["source"] = "preprocessor"

        summary = model.summary()

        assert "Source: preprocessor" in summary


# =============================================================================
# Test Validate Components Method
# =============================================================================


class TestValidateComponents:
    """Tests for validate_components method."""

    def test_validate_components_empty(self) -> None:
        """Test validate_components with no components."""
        model = IWFMModel(name="Test")
        warnings = model.validate_components()
        assert warnings == []

    def test_validate_components_success(self) -> None:
        """Test validate_components when all components validate."""
        model = IWFMModel(name="Test")

        mock_gw = MagicMock()
        mock_gw.validate.return_value = None
        model.groundwater = mock_gw

        mock_streams = MagicMock()
        mock_streams.validate.return_value = None
        model.streams = mock_streams

        warnings = model.validate_components()
        assert warnings == []

    def test_validate_components_with_errors(self) -> None:
        """Test validate_components when components have errors."""
        model = IWFMModel(name="Test")

        mock_gw = MagicMock()
        mock_gw.validate.side_effect = Exception("GW error")
        model.groundwater = mock_gw

        mock_streams = MagicMock()
        mock_streams.validate.side_effect = Exception("Stream error")
        model.streams = mock_streams

        warnings = model.validate_components()

        assert len(warnings) == 2
        assert any("Groundwater" in w for w in warnings)
        assert any("Stream" in w for w in warnings)


# =============================================================================
# Test from_preprocessor_binary Class Method
# =============================================================================


class TestFromPreprocessorBinary:
    """Tests for from_preprocessor_binary classmethod."""

    def test_from_preprocessor_binary_basic(self, tmp_path: Path) -> None:
        """Test loading model from binary file using mocked readers."""
        binary_file = tmp_path / "model.bin"
        binary_file.write_bytes(b"\x00")  # Dummy file

        mock_mesh = MagicMock()
        mock_mesh.n_nodes = 50
        mock_mesh.n_elements = 30

        from unittest.mock import patch

        with patch("pyiwfm.core.model.Path", wraps=Path):
            with patch(
                "pyiwfm.io.binary.FortranBinaryReader"
            ) as mock_reader_cls, patch(
                "pyiwfm.io.binary.read_binary_mesh", return_value=mock_mesh
            ) as mock_read_mesh, patch(
                "pyiwfm.io.binary.read_binary_stratigraphy"
            ):
                # Mock context manager
                mock_reader_inst = MagicMock()
                mock_reader_cls.return_value.__enter__ = MagicMock(
                    return_value=mock_reader_inst
                )
                mock_reader_cls.return_value.__exit__ = MagicMock(
                    return_value=False
                )

                model = IWFMModel.from_preprocessor_binary(
                    binary_file, name="TestBinary"
                )

                assert model.name == "TestBinary"
                assert model.mesh is mock_mesh
                assert model.metadata["source"] == "binary"
                assert model.metadata["binary_file"] == str(binary_file)
                mock_read_mesh.assert_called_once_with(binary_file)

    def test_from_preprocessor_binary_default_name(self, tmp_path: Path) -> None:
        """Test that name defaults to file stem when not provided."""
        binary_file = tmp_path / "my_model.bin"
        binary_file.write_bytes(b"\x00")

        mock_mesh = MagicMock()

        from unittest.mock import patch

        with patch(
            "pyiwfm.io.binary.FortranBinaryReader"
        ) as mock_reader_cls, patch(
            "pyiwfm.io.binary.read_binary_mesh", return_value=mock_mesh
        ), patch(
            "pyiwfm.io.binary.read_binary_stratigraphy"
        ):
            mock_reader_cls.return_value.__enter__ = MagicMock()
            mock_reader_cls.return_value.__exit__ = MagicMock(return_value=False)

            model = IWFMModel.from_preprocessor_binary(binary_file)

            assert model.name == "my_model"

    def test_from_preprocessor_binary_with_stratigraphy(
        self, tmp_path: Path
    ) -> None:
        """Test loading with companion stratigraphy binary file."""
        binary_file = tmp_path / "model.bin"
        binary_file.write_bytes(b"\x00")
        strat_file = tmp_path / "model.strat.bin"
        strat_file.write_bytes(b"\x00")

        mock_mesh = MagicMock()
        mock_strat = MagicMock()

        from unittest.mock import patch

        with patch(
            "pyiwfm.io.binary.FortranBinaryReader"
        ) as mock_reader_cls, patch(
            "pyiwfm.io.binary.read_binary_mesh", return_value=mock_mesh
        ), patch(
            "pyiwfm.io.binary.read_binary_stratigraphy",
            return_value=mock_strat,
        ):
            mock_reader_cls.return_value.__enter__ = MagicMock()
            mock_reader_cls.return_value.__exit__ = MagicMock(return_value=False)

            model = IWFMModel.from_preprocessor_binary(binary_file)

            assert model.stratigraphy is mock_strat

    def test_from_binary_is_alias(self) -> None:
        """Test that from_binary delegates to from_preprocessor_binary."""
        from unittest.mock import patch

        mock_model = IWFMModel(name="alias_test")
        with patch.object(
            IWFMModel, "from_preprocessor_binary", return_value=mock_model
        ) as mock_method:
            result = IWFMModel.from_binary("some_path.bin")
            mock_method.assert_called_once_with("some_path.bin")
            assert result is mock_model


# =============================================================================
# Test from_preprocessor Class Method
# =============================================================================


class TestFromPreprocessor:
    """Tests for from_preprocessor classmethod."""

    def test_from_preprocessor_raises_on_missing_nodes_file(self) -> None:
        """Test that from_preprocessor raises FileFormatError when nodes file is missing."""
        from unittest.mock import patch

        mock_config = MagicMock()
        mock_config.nodes_file = None
        mock_config.elements_file = "elems.dat"
        mock_config.model_name = "test"

        with patch(
            "pyiwfm.io.preprocessor.read_preprocessor_main",
            return_value=mock_config,
        ), patch(
            "pyiwfm.io.ascii.read_nodes"
        ), patch(
            "pyiwfm.io.ascii.read_elements"
        ):
            from pyiwfm.core.exceptions import FileFormatError

            with pytest.raises(FileFormatError, match="Nodes file"):
                IWFMModel.from_preprocessor("fake_pp.in")

    def test_from_preprocessor_raises_on_missing_elements_file(self) -> None:
        """Test that from_preprocessor raises FileFormatError when elements file is missing."""
        from unittest.mock import patch

        mock_config = MagicMock()
        mock_config.nodes_file = "nodes.dat"
        mock_config.elements_file = None
        mock_config.model_name = "test"

        mock_nodes = {1: MagicMock(), 2: MagicMock()}

        with patch(
            "pyiwfm.io.preprocessor.read_preprocessor_main",
            return_value=mock_config,
        ), patch(
            "pyiwfm.io.ascii.read_nodes",
            return_value=mock_nodes,
        ), patch(
            "pyiwfm.io.ascii.read_elements"
        ):
            from pyiwfm.core.exceptions import FileFormatError

            with pytest.raises(FileFormatError, match="Elements file"):
                IWFMModel.from_preprocessor("fake_pp.in")


# =============================================================================
# Test Model Metadata
# =============================================================================


class TestModelMetadata:
    """Tests for metadata property and dict structure."""

    def test_metadata_empty_by_default(self) -> None:
        """Test metadata is empty dict by default."""
        model = IWFMModel(name="Test")
        assert isinstance(model.metadata, dict)
        assert len(model.metadata) == 0

    def test_metadata_complete_structure(self) -> None:
        """Test metadata with complete dict structure typical of preprocessor load."""
        metadata = {
            "source": "preprocessor",
            "preprocessor_file": "/path/to/pp.in",
            "length_unit": "FT",
            "area_unit": "AC",
            "volume_unit": "AF",
        }
        model = IWFMModel(name="FullMeta", metadata=metadata)

        assert model.metadata["source"] == "preprocessor"
        assert model.metadata["preprocessor_file"] == "/path/to/pp.in"
        assert model.metadata["length_unit"] == "FT"
        assert model.metadata["area_unit"] == "AC"
        assert model.metadata["volume_unit"] == "AF"

    def test_metadata_binary_source(self) -> None:
        """Test metadata structure for binary source."""
        metadata = {
            "source": "binary",
            "binary_file": "/path/to/model.bin",
        }
        model = IWFMModel(name="BinaryMeta", metadata=metadata)

        assert model.metadata["source"] == "binary"
        assert "binary_file" in model.metadata

    def test_metadata_with_load_errors(self) -> None:
        """Test metadata tracks component load errors."""
        model = IWFMModel(name="ErrorMeta")
        model.metadata["streams_load_error"] = "Could not parse stream file"
        model.metadata["lakes_load_error"] = "Lake file not found"

        assert "streams_load_error" in model.metadata
        assert "lakes_load_error" in model.metadata

    def test_metadata_with_simulation_fields(self) -> None:
        """Test metadata structure with simulation-level fields."""
        metadata = {
            "source": "simulation_with_preprocessor",
            "simulation_file": "/path/to/sim.in",
            "preprocessor_file": "/path/to/pp.in",
            "start_date": "1990-10-01",
            "end_date": "2015-09-30",
            "time_step_length": 1,
            "time_step_unit": "MON",
        }
        model = IWFMModel(name="SimMeta", metadata=metadata)

        assert model.metadata["start_date"] == "1990-10-01"
        assert model.metadata["end_date"] == "2015-09-30"
        assert model.metadata["time_step_length"] == 1
        assert model.metadata["time_step_unit"] == "MON"

    def test_metadata_with_various_components_set(self) -> None:
        """Test metadata is independent of component state."""
        model = IWFMModel(
            name="CompMeta",
            metadata={"source": "test"},
        )
        model.groundwater = MagicMock()
        model.streams = MagicMock()
        model.lakes = MagicMock()
        model.rootzone = MagicMock()

        # Setting components should not affect metadata
        assert model.metadata == {"source": "test"}


# =============================================================================
# Test Model Export / Serialization
# =============================================================================


class TestModelExport:
    """Tests for export and serialization methods."""

    def test_to_preprocessor_delegates(self, tmp_path: Path) -> None:
        """Test to_preprocessor delegates to save_model_to_preprocessor."""
        from unittest.mock import patch

        model = IWFMModel(name="ExportTest")

        mock_config = MagicMock()
        mock_config.nodes_file = tmp_path / "nodes.dat"
        mock_config.elements_file = tmp_path / "elements.dat"
        mock_config.stratigraphy_file = tmp_path / "strat.dat"
        mock_config.subregions_file = None

        with patch(
            "pyiwfm.io.preprocessor.save_model_to_preprocessor",
            return_value=mock_config,
        ) as mock_save:
            files = model.to_preprocessor(tmp_path)

            mock_save.assert_called_once_with(model, tmp_path, "ExportTest")
            assert "nodes" in files
            assert "elements" in files
            assert "stratigraphy" in files
            assert "subregions" not in files

    def test_to_preprocessor_with_subregions(self, tmp_path: Path) -> None:
        """Test to_preprocessor includes subregions when present."""
        from unittest.mock import patch

        model = IWFMModel(name="SubregTest")

        mock_config = MagicMock()
        mock_config.nodes_file = tmp_path / "nodes.dat"
        mock_config.elements_file = tmp_path / "elements.dat"
        mock_config.stratigraphy_file = tmp_path / "strat.dat"
        mock_config.subregions_file = tmp_path / "subregions.dat"

        with patch(
            "pyiwfm.io.preprocessor.save_model_to_preprocessor",
            return_value=mock_config,
        ):
            files = model.to_preprocessor(tmp_path)

            assert "subregions" in files
            assert files["subregions"] == tmp_path / "subregions.dat"

    def test_to_simulation_delegates(self, tmp_path: Path) -> None:
        """Test to_simulation delegates to save_complete_model."""
        from unittest.mock import patch

        model = IWFMModel(name="SimExport")

        expected_files = {
            "simulation": tmp_path / "sim.in",
            "preprocessor": tmp_path / "pp.in",
        }

        with patch(
            "pyiwfm.io.preprocessor.save_complete_model",
            return_value=expected_files,
        ) as mock_save:
            files = model.to_simulation(tmp_path)

            mock_save.assert_called_once_with(
                model, tmp_path,
                timeseries_format="text",
                file_paths=None,
            )
            assert files == expected_files

    def test_to_hdf5_delegates(self, tmp_path: Path) -> None:
        """Test to_hdf5 delegates to write_model_hdf5."""
        from unittest.mock import patch

        model = IWFMModel(name="HDF5Export")
        output_file = tmp_path / "model.h5"

        with patch(
            "pyiwfm.io.hdf5.write_model_hdf5"
        ) as mock_write:
            model.to_hdf5(output_file)

            mock_write.assert_called_once_with(output_file, model)

    def test_to_binary_with_mesh_and_stratigraphy(self, tmp_path: Path) -> None:
        """Test to_binary writes both mesh and stratigraphy files."""
        from unittest.mock import patch

        model = IWFMModel(name="BinaryExport")
        model.mesh = MagicMock()
        model.stratigraphy = MagicMock()

        output_file = tmp_path / "model.bin"

        with patch(
            "pyiwfm.io.binary.write_binary_mesh"
        ) as mock_write_mesh, patch(
            "pyiwfm.io.binary.write_binary_stratigraphy"
        ) as mock_write_strat:
            model.to_binary(output_file)

            mock_write_mesh.assert_called_once_with(output_file, model.mesh)
            mock_write_strat.assert_called_once_with(
                output_file.with_suffix(".strat.bin"),
                model.stratigraphy,
            )

    def test_to_binary_mesh_only(self, tmp_path: Path) -> None:
        """Test to_binary with mesh but no stratigraphy."""
        from unittest.mock import patch

        model = IWFMModel(name="MeshOnly")
        model.mesh = MagicMock()
        model.stratigraphy = None

        output_file = tmp_path / "model.bin"

        with patch(
            "pyiwfm.io.binary.write_binary_mesh"
        ) as mock_write_mesh, patch(
            "pyiwfm.io.binary.write_binary_stratigraphy"
        ) as mock_write_strat:
            model.to_binary(output_file)

            mock_write_mesh.assert_called_once()
            mock_write_strat.assert_not_called()

    def test_to_binary_stratigraphy_only(self, tmp_path: Path) -> None:
        """Test to_binary with stratigraphy but no mesh."""
        from unittest.mock import patch

        model = IWFMModel(name="StratOnly")
        model.mesh = None
        model.stratigraphy = MagicMock()

        output_file = tmp_path / "model.bin"

        with patch(
            "pyiwfm.io.binary.write_binary_mesh"
        ) as mock_write_mesh, patch(
            "pyiwfm.io.binary.write_binary_stratigraphy"
        ) as mock_write_strat:
            model.to_binary(output_file)

            mock_write_mesh.assert_not_called()
            mock_write_strat.assert_called_once()

    def test_to_binary_empty_model(self, tmp_path: Path) -> None:
        """Test to_binary with neither mesh nor stratigraphy."""
        from unittest.mock import patch

        model = IWFMModel(name="Empty")
        output_file = tmp_path / "model.bin"

        with patch(
            "pyiwfm.io.binary.write_binary_mesh"
        ) as mock_write_mesh, patch(
            "pyiwfm.io.binary.write_binary_stratigraphy"
        ) as mock_write_strat:
            model.to_binary(output_file)

            mock_write_mesh.assert_not_called()
            mock_write_strat.assert_not_called()

    def test_repr_with_all_components(self) -> None:
        """Test __repr__ with all components set."""
        model = IWFMModel(name="FullModel")

        mock_mesh = MagicMock()
        mock_mesh.n_nodes = 500
        mock_mesh.n_elements = 400
        model.mesh = mock_mesh

        mock_strat = MagicMock()
        mock_strat.n_layers = 3
        model.stratigraphy = mock_strat

        model.groundwater = MagicMock()
        model.streams = MagicMock()
        model.lakes = MagicMock()
        model.rootzone = MagicMock()

        repr_str = repr(model)

        assert "FullModel" in repr_str
        assert "n_nodes=500" in repr_str
        assert "n_elements=400" in repr_str
        assert "n_layers=3" in repr_str

    def test_repr_format(self) -> None:
        """Test __repr__ exact format string."""
        model = IWFMModel(name="Fmt")
        expected = "IWFMModel(name='Fmt', n_nodes=0, n_elements=0, n_layers=0)"
        assert repr(model) == expected


# =============================================================================
# Test Model Components Interaction
# =============================================================================


class TestModelComponents:
    """Tests for component interaction methods."""

    def test_grid_setter_updates_mesh(self) -> None:
        """Test that setting grid updates mesh."""
        model = IWFMModel(name="GridTest")
        mock_mesh = MagicMock()
        mock_mesh.n_nodes = 42

        model.grid = mock_mesh

        assert model.mesh is mock_mesh
        assert model.grid is mock_mesh
        assert model.n_nodes == 42

    def test_grid_setter_to_none(self) -> None:
        """Test setting grid to None."""
        model = IWFMModel(name="GridNone")
        model.mesh = MagicMock()

        model.grid = None

        assert model.mesh is None
        assert model.grid is None
        assert model.n_nodes == 0

    def test_multiple_components_coexist(self) -> None:
        """Test that all components can be set simultaneously."""
        model = IWFMModel(name="AllComponents")

        mock_mesh = MagicMock()
        mock_mesh.n_nodes = 100
        mock_mesh.n_elements = 80
        model.mesh = mock_mesh

        mock_strat = MagicMock()
        mock_strat.n_layers = 4
        model.stratigraphy = mock_strat

        mock_gw = MagicMock()
        mock_gw.n_wells = 20
        model.groundwater = mock_gw

        mock_streams = MagicMock()
        mock_streams.n_nodes = 150
        mock_streams.n_reaches = 10
        mock_streams.n_diversions = 5
        model.streams = mock_streams

        mock_lakes = MagicMock()
        mock_lakes.n_lakes = 2
        model.lakes = mock_lakes

        mock_rz = MagicMock()
        mock_rz.n_crop_types = 8
        model.rootzone = mock_rz

        # Verify all counts are accessible simultaneously
        assert model.n_nodes == 100
        assert model.n_elements == 80
        assert model.n_layers == 4
        assert model.n_wells == 20
        assert model.n_stream_nodes == 150
        assert model.n_stream_reaches == 10
        assert model.n_diversions == 5
        assert model.n_lakes == 2
        assert model.n_crop_types == 8

        # Verify all has_* are True
        assert model.has_groundwater is True
        assert model.has_streams is True
        assert model.has_lakes is True
        assert model.has_rootzone is True

    def test_summary_with_all_components(self) -> None:
        """Test summary output with all components populated."""
        model = IWFMModel(name="Complete")
        model.metadata["source"] = "simulation"

        mock_mesh = MagicMock()
        mock_mesh.n_nodes = 100
        mock_mesh.n_elements = 80
        mock_mesh.n_subregions = 5
        model.mesh = mock_mesh

        mock_strat = MagicMock()
        mock_strat.n_layers = 3
        model.stratigraphy = mock_strat

        mock_gw = MagicMock()
        mock_gw.n_wells = 10
        mock_gw.n_boundary_conditions = 4
        mock_gw.n_tile_drains = 2
        mock_gw.aquifer_params = MagicMock()  # Not None => "Loaded"
        model.groundwater = mock_gw

        mock_streams = MagicMock()
        mock_streams.n_nodes = 200
        mock_streams.n_reaches = 15
        mock_streams.n_diversions = 8
        mock_streams.n_bypasses = 1
        model.streams = mock_streams

        mock_lakes = MagicMock()
        mock_lakes.n_lakes = 3
        mock_lakes.n_lake_elements = 45
        model.lakes = mock_lakes

        mock_rz = MagicMock()
        mock_rz.n_crop_types = 12
        mock_rz.element_landuse = {i: f"use_{i}" for i in range(20)}
        mock_rz.soil_params = {i: f"soil_{i}" for i in range(10)}
        model.rootzone = mock_rz

        mock_sw = MagicMock()
        mock_sw.n_watersheds = 2
        model.small_watersheds = mock_sw

        mock_uz = MagicMock()
        mock_uz.n_layers = 3
        mock_uz.n_elements = 50
        model.unsaturated_zone = mock_uz

        summary = model.summary()

        # Verify all sections are present
        assert "Complete" in summary
        assert "Subregions: 5" in summary
        assert "Wells: 10" in summary
        assert "Aquifer Parameters: Loaded" in summary
        assert "Stream Nodes: 200" in summary
        assert "Lakes: 3" in summary
        assert "Crop Types: 12" in summary
        assert "Land Use Assignments: 20" in summary
        assert "Soil Parameter Sets: 10" in summary
        assert "Source: simulation" in summary
        # Verify "Not loaded" is NOT present since all components are set
        assert "Not loaded" not in summary

    def test_summary_aquifer_params_not_loaded(self) -> None:
        """Test summary shows 'Not loaded' for aquifer params when None."""
        model = IWFMModel(name="NoAqParams")

        mock_gw = MagicMock()
        mock_gw.n_wells = 5
        mock_gw.n_boundary_conditions = 0
        mock_gw.n_tile_drains = 0
        mock_gw.aquifer_params = None
        model.groundwater = mock_gw

        summary = model.summary()

        assert "Aquifer Parameters: Not loaded" in summary

    def test_summary_unknown_source(self) -> None:
        """Test summary shows 'unknown' when source is not in metadata."""
        model = IWFMModel(name="NoSource")

        summary = model.summary()

        assert "Source: unknown" in summary


# =============================================================================
# Test Model Query-Related Properties
# =============================================================================


class TestModelQuery:
    """Tests for query-related properties and data retrieval."""

    def test_n_stream_nodes_zero_without_streams(self) -> None:
        """Test stream node count is 0 without streams component."""
        model = IWFMModel(name="NoStreams")
        assert model.n_stream_nodes == 0

    def test_n_stream_reaches_zero_without_streams(self) -> None:
        """Test stream reach count is 0 without streams component."""
        model = IWFMModel(name="NoStreams")
        assert model.n_stream_reaches == 0

    def test_n_diversions_zero_without_streams(self) -> None:
        """Test diversion count is 0 without streams component."""
        model = IWFMModel(name="NoStreams")
        assert model.n_diversions == 0

    def test_n_wells_zero_without_groundwater(self) -> None:
        """Test well count is 0 without groundwater component."""
        model = IWFMModel(name="NoGW")
        assert model.n_wells == 0

    def test_n_lakes_zero_without_lakes(self) -> None:
        """Test lake count is 0 without lakes component."""
        model = IWFMModel(name="NoLakes")
        assert model.n_lakes == 0

    def test_n_crop_types_zero_without_rootzone(self) -> None:
        """Test crop type count is 0 without rootzone component."""
        model = IWFMModel(name="NoRZ")
        assert model.n_crop_types == 0

    def test_from_simulation_delegates(self) -> None:
        """Test from_simulation delegates to load_complete_model."""
        from unittest.mock import patch

        mock_model = IWFMModel(name="loaded")

        with patch(
            "pyiwfm.io.preprocessor.load_complete_model",
            return_value=mock_model,
        ) as mock_load:
            result = IWFMModel.from_simulation("sim.in", load_timeseries=True)

            mock_load.assert_called_once_with("sim.in", load_timeseries=True)
            assert result is mock_model

    def test_from_simulation_default_args(self) -> None:
        """Test from_simulation default argument for load_timeseries."""
        from unittest.mock import patch

        mock_model = IWFMModel(name="loaded")

        with patch(
            "pyiwfm.io.preprocessor.load_complete_model",
            return_value=mock_model,
        ) as mock_load:
            IWFMModel.from_simulation("sim.in")

            mock_load.assert_called_once_with("sim.in", load_timeseries=False)

    def test_from_hdf5_delegates(self) -> None:
        """Test from_hdf5 delegates to read_model_hdf5."""
        from unittest.mock import patch

        mock_model = IWFMModel(name="hdf5_loaded")

        with patch(
            "pyiwfm.io.hdf5.read_model_hdf5",
            return_value=mock_model,
        ) as mock_read:
            result = IWFMModel.from_hdf5("model.h5")

            mock_read.assert_called_once_with("model.h5")
            assert result is mock_model


# =============================================================================
# Test Model Validation Edge Cases
# =============================================================================


class TestModelValidation:
    """Tests for additional validation edge cases."""

    def test_validate_stratigraphy_node_mismatch(self) -> None:
        """Test validation catches node count mismatch between mesh and stratigraphy."""
        model = IWFMModel(name="Mismatch")

        mock_mesh = MagicMock()
        mock_mesh.validate.return_value = None
        mock_mesh.n_nodes = 200
        model.mesh = mock_mesh

        mock_strat = MagicMock()
        mock_strat.validate.return_value = []
        mock_strat.n_nodes = 150  # Mismatch
        model.stratigraphy = mock_strat

        with pytest.raises(ValidationError) as excinfo:
            model.validate()

        errors = excinfo.value.errors
        assert any("mismatch" in e.lower() for e in errors)
        assert any("200" in e for e in errors)
        assert any("150" in e for e in errors)

    def test_validate_stratigraphy_returns_warnings(self) -> None:
        """Test that stratigraphy warnings are included in validation errors."""
        model = IWFMModel(name="StratWarn")

        mock_mesh = MagicMock()
        mock_mesh.validate.return_value = None
        mock_mesh.n_nodes = 100
        model.mesh = mock_mesh

        mock_strat = MagicMock()
        mock_strat.validate.return_value = [
            "Layer 2 has zero thickness at node 5",
            "Layer 3 pinches out at node 12",
        ]
        mock_strat.n_nodes = 100
        model.stratigraphy = mock_strat

        # Even though mesh and strat match on node count, the warnings from
        # stratigraphy.validate() are included in errors list and cause
        # ValidationError to be raised.
        with pytest.raises(ValidationError) as excinfo:
            model.validate()

        errors = excinfo.value.errors
        assert any("zero thickness" in e for e in errors)
        assert any("pinches out" in e for e in errors)

    def test_validate_stratigraphy_exception(self) -> None:
        """Test validation when stratigraphy.validate() raises an exception."""
        model = IWFMModel(name="StratErr")

        mock_mesh = MagicMock()
        mock_mesh.validate.return_value = None
        mock_mesh.n_nodes = 100
        model.mesh = mock_mesh

        mock_strat = MagicMock()
        mock_strat.validate.side_effect = Exception("Corrupt stratigraphy data")
        mock_strat.n_nodes = 100
        model.stratigraphy = mock_strat

        with pytest.raises(ValidationError) as excinfo:
            model.validate()

        errors = excinfo.value.errors
        assert any("Stratigraphy validation failed" in e for e in errors)

    def test_validate_both_mesh_and_stratigraphy_none(self) -> None:
        """Test validation with neither mesh nor stratigraphy."""
        model = IWFMModel(name="Empty")

        with pytest.raises(ValidationError) as excinfo:
            model.validate()

        errors = excinfo.value.errors
        assert any("no mesh" in e for e in errors)
        assert any("no stratigraphy" in e for e in errors)
        assert len(errors) == 2

    def test_validate_mesh_exception_and_stratigraphy_none(self) -> None:
        """Test validation when mesh raises exception and stratigraphy is None."""
        model = IWFMModel(name="MeshExc")

        mock_mesh = MagicMock()
        mock_mesh.validate.side_effect = Exception("Duplicate nodes found")
        mock_mesh.n_nodes = 50
        model.mesh = mock_mesh

        with pytest.raises(ValidationError) as excinfo:
            model.validate()

        errors = excinfo.value.errors
        assert any("Mesh validation failed" in e for e in errors)
        assert any("no stratigraphy" in e for e in errors)

    def test_validate_components_lake_failure(self) -> None:
        """Test validate_components captures lake validation failure."""
        model = IWFMModel(name="LakeFail")

        mock_lakes = MagicMock()
        mock_lakes.validate.side_effect = Exception("Lake has no elements")
        model.lakes = mock_lakes

        warnings = model.validate_components()

        assert len(warnings) == 1
        assert "Lake validation" in warnings[0]
        assert "Lake has no elements" in warnings[0]

    def test_validate_components_rootzone_failure(self) -> None:
        """Test validate_components captures rootzone validation failure."""
        model = IWFMModel(name="RZFail")

        mock_rz = MagicMock()
        mock_rz.validate.side_effect = Exception("No crop types defined")
        model.rootzone = mock_rz

        warnings = model.validate_components()

        assert len(warnings) == 1
        assert "Root zone validation" in warnings[0]
        assert "No crop types defined" in warnings[0]

    def test_validate_components_mixed_success_failure(self) -> None:
        """Test validate_components with some components passing and some failing."""
        model = IWFMModel(name="Mixed")

        # Groundwater passes
        mock_gw = MagicMock()
        mock_gw.validate.return_value = None
        model.groundwater = mock_gw

        # Streams fail
        mock_streams = MagicMock()
        mock_streams.validate.side_effect = Exception("Disconnected reach")
        model.streams = mock_streams

        # Lakes pass
        mock_lakes = MagicMock()
        mock_lakes.validate.return_value = None
        model.lakes = mock_lakes

        # Root zone fails
        mock_rz = MagicMock()
        mock_rz.validate.side_effect = Exception("Invalid soil params")
        model.rootzone = mock_rz

        warnings = model.validate_components()

        # Only the two failing components should produce warnings
        assert len(warnings) == 2
        assert any("Stream" in w for w in warnings)
        assert any("Root zone" in w for w in warnings)
        # Passing components should NOT be in warnings
        assert not any("Groundwater" in w for w in warnings)
        assert not any("Lake" in w for w in warnings)

    def test_validate_components_all_four_fail(self) -> None:
        """Test validate_components when all four components fail."""
        model = IWFMModel(name="AllFail")

        mock_gw = MagicMock()
        mock_gw.validate.side_effect = Exception("GW error")
        model.groundwater = mock_gw

        mock_streams = MagicMock()
        mock_streams.validate.side_effect = Exception("Stream error")
        model.streams = mock_streams

        mock_lakes = MagicMock()
        mock_lakes.validate.side_effect = Exception("Lake error")
        model.lakes = mock_lakes

        mock_rz = MagicMock()
        mock_rz.validate.side_effect = Exception("RZ error")
        model.rootzone = mock_rz

        warnings = model.validate_components()

        assert len(warnings) == 4
        assert any("Groundwater" in w for w in warnings)
        assert any("Stream" in w for w in warnings)
        assert any("Lake" in w for w in warnings)
        assert any("Root zone" in w for w in warnings)

    def test_validate_error_count_in_message(self) -> None:
        """Test that ValidationError message includes error count."""
        model = IWFMModel(name="ErrorCount")

        with pytest.raises(ValidationError) as excinfo:
            model.validate()

        # Both "no mesh" and "no stratigraphy" => 2 errors
        assert "2 error(s)" in str(excinfo.value)

    def test_validate_success_returns_empty_list(self) -> None:
        """Test that successful validation returns an empty list."""
        model = IWFMModel(name="Valid")

        mock_mesh = MagicMock()
        mock_mesh.validate.return_value = None
        mock_mesh.n_nodes = 50
        model.mesh = mock_mesh

        mock_strat = MagicMock()
        mock_strat.validate.return_value = []
        mock_strat.n_nodes = 50
        model.stratigraphy = mock_strat

        result = model.validate()
        assert result == []
        assert isinstance(result, list)
