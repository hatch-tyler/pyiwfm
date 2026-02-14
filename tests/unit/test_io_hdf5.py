"""Unit tests for HDF5 I/O handlers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from pyiwfm.core.mesh import AppGrid, Element, Node, Subregion
from pyiwfm.core.stratigraphy import Stratigraphy
from pyiwfm.core.model import IWFMModel
from pyiwfm.core.timeseries import TimeSeries
from pyiwfm.core.exceptions import FileFormatError

# Skip all tests if h5py is not available
pytest.importorskip("h5py")

from pyiwfm.io.hdf5 import (
    HDF5ModelWriter,
    HDF5ModelReader,
    write_model_hdf5,
    read_model_hdf5,
)


class TestHDF5MeshIO:
    """Tests for HDF5 mesh I/O."""

    def test_write_read_mesh_roundtrip(
        self,
        tmp_path: Path,
        small_grid_nodes: list[dict],
        small_grid_elements: list[dict],
    ) -> None:
        """Test mesh write/read roundtrip."""
        # Create original mesh
        nodes = {d["id"]: Node(**d) for d in small_grid_nodes}
        elements = {d["id"]: Element(**d) for d in small_grid_elements}
        subregions = {
            1: Subregion(id=1, name="Region A"),
            2: Subregion(id=2, name="Region B"),
        }
        grid = AppGrid(nodes=nodes, elements=elements, subregions=subregions)
        grid.compute_areas()
        grid.compute_connectivity()

        # Write to HDF5
        filepath = tmp_path / "mesh.h5"
        with HDF5ModelWriter(filepath) as writer:
            writer.write_mesh(grid)

        # Read back
        with HDF5ModelReader(filepath) as reader:
            grid_back = reader.read_mesh()

        # Verify
        assert grid_back.n_nodes == grid.n_nodes
        assert grid_back.n_elements == grid.n_elements
        assert grid_back.n_subregions == grid.n_subregions

        # Check coordinates
        for nid in grid.nodes:
            assert grid_back.nodes[nid].x == pytest.approx(grid.nodes[nid].x)
            assert grid_back.nodes[nid].y == pytest.approx(grid.nodes[nid].y)
            assert grid_back.nodes[nid].area == pytest.approx(grid.nodes[nid].area)

        # Check elements
        for eid in grid.elements:
            assert grid_back.elements[eid].vertices == grid.elements[eid].vertices
            assert grid_back.elements[eid].subregion == grid.elements[eid].subregion

        # Check subregions
        assert grid_back.subregions[1].name == "Region A"
        assert grid_back.subregions[2].name == "Region B"

    def test_write_read_triangular_mesh(self, tmp_path: Path) -> None:
        """Test roundtrip with triangular elements."""
        nodes = {
            1: Node(id=1, x=0.0, y=0.0),
            2: Node(id=2, x=100.0, y=0.0),
            3: Node(id=3, x=50.0, y=86.6),
        }
        elements = {
            1: Element(id=1, vertices=(1, 2, 3), subregion=1),
        }
        grid = AppGrid(nodes=nodes, elements=elements)

        filepath = tmp_path / "tri_mesh.h5"
        with HDF5ModelWriter(filepath) as writer:
            writer.write_mesh(grid)

        with HDF5ModelReader(filepath) as reader:
            grid_back = reader.read_mesh()

        assert grid_back.elements[1].is_triangle
        assert grid_back.elements[1].vertices == (1, 2, 3)


class TestHDF5StratigraphyIO:
    """Tests for HDF5 stratigraphy I/O."""

    def test_write_read_stratigraphy_roundtrip(
        self, tmp_path: Path, sample_stratigraphy_data: dict
    ) -> None:
        """Test stratigraphy write/read roundtrip."""
        strat = Stratigraphy(**sample_stratigraphy_data)

        filepath = tmp_path / "strat.h5"
        with HDF5ModelWriter(filepath) as writer:
            writer.write_stratigraphy(strat)

        with HDF5ModelReader(filepath) as reader:
            strat_back = reader.read_stratigraphy()

        assert strat_back.n_nodes == strat.n_nodes
        assert strat_back.n_layers == strat.n_layers
        np.testing.assert_allclose(strat_back.gs_elev, strat.gs_elev)
        np.testing.assert_allclose(strat_back.top_elev, strat.top_elev)
        np.testing.assert_allclose(strat_back.bottom_elev, strat.bottom_elev)
        np.testing.assert_array_equal(strat_back.active_node, strat.active_node)


class TestHDF5TimeSeriesIO:
    """Tests for HDF5 time series I/O."""

    def test_write_read_timeseries_roundtrip(self, tmp_path: Path) -> None:
        """Test time series write/read roundtrip."""
        times = np.array(
            ["2020-01-01", "2020-01-02", "2020-01-03"], dtype="datetime64[D]"
        )
        values = np.array([100.0, 101.0, 102.0])

        ts = TimeSeries(
            times=times,
            values=values,
            name="test_head",
            units="ft",
            location="node_42",
        )

        filepath = tmp_path / "ts.h5"
        with HDF5ModelWriter(filepath) as writer:
            writer.write_timeseries(ts, variable="head")

        with HDF5ModelReader(filepath) as reader:
            ts_back = reader.read_timeseries("head", "node_42")

        assert ts_back.n_times == 3
        assert ts_back.name == "test_head"
        assert ts_back.units == "ft"
        assert ts_back.location == "node_42"
        np.testing.assert_allclose(ts_back.values, values)

    def test_list_timeseries(self, tmp_path: Path) -> None:
        """Test listing available time series."""
        times = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[D]")
        values = np.array([1.0, 2.0])

        filepath = tmp_path / "ts_list.h5"
        with HDF5ModelWriter(filepath) as writer:
            writer.write_timeseries(
                TimeSeries(times=times, values=values, location="node_1"),
                variable="head",
            )
            writer.write_timeseries(
                TimeSeries(times=times, values=values, location="node_2"),
                variable="head",
            )
            writer.write_timeseries(
                TimeSeries(times=times, values=values, location="reach_1"),
                variable="flow",
            )

        with HDF5ModelReader(filepath) as reader:
            ts_list = reader.list_timeseries()

        assert "head" in ts_list
        assert "flow" in ts_list
        assert set(ts_list["head"]) == {"node_1", "node_2"}
        assert ts_list["flow"] == ["reach_1"]


class TestHDF5ModelIO:
    """Tests for complete model I/O."""

    def test_write_read_model_roundtrip(
        self,
        tmp_path: Path,
        small_grid_nodes: list[dict],
        small_grid_elements: list[dict],
        sample_stratigraphy_data: dict,
    ) -> None:
        """Test complete model write/read roundtrip."""
        # Create model
        nodes = {d["id"]: Node(**d) for d in small_grid_nodes}
        elements = {d["id"]: Element(**d) for d in small_grid_elements}
        grid = AppGrid(nodes=nodes, elements=elements)

        strat = Stratigraphy(**sample_stratigraphy_data)

        model = IWFMModel(
            name="Test Model",
            mesh=grid,
            stratigraphy=strat,
            metadata={"description": "Test model for HDF5 roundtrip"},
        )

        # Write and read
        filepath = tmp_path / "model.h5"
        write_model_hdf5(filepath, model)
        model_back = read_model_hdf5(filepath)

        # Verify
        assert model_back.name == "Test Model"
        assert model_back.n_nodes == model.n_nodes
        assert model_back.n_elements == model.n_elements
        assert model_back.n_layers == model.n_layers

    def test_model_metadata(self, tmp_path: Path) -> None:
        """Test metadata is preserved."""
        model = IWFMModel(
            name="Metadata Test",
            metadata={
                "author": "Test Author",
                "version": "1.0",
            },
        )

        filepath = tmp_path / "meta.h5"
        with HDF5ModelWriter(filepath) as writer:
            writer.write_metadata(model.metadata)
            writer.write_metadata({"name": model.name})

        with HDF5ModelReader(filepath) as reader:
            meta = reader.read_metadata()

        assert meta["name"] == "Metadata Test"
        assert meta["author"] == "Test Author"


class TestHDF5Compression:
    """Tests for HDF5 compression options."""

    def test_compressed_mesh(
        self,
        tmp_path: Path,
        small_grid_nodes: list[dict],
        small_grid_elements: list[dict],
    ) -> None:
        """Test mesh with compression."""
        nodes = {d["id"]: Node(**d) for d in small_grid_nodes}
        elements = {d["id"]: Element(**d) for d in small_grid_elements}
        grid = AppGrid(nodes=nodes, elements=elements)

        filepath = tmp_path / "compressed.h5"
        with HDF5ModelWriter(filepath, compression="gzip") as writer:
            writer.write_mesh(grid)

        with HDF5ModelReader(filepath) as reader:
            grid_back = reader.read_mesh()

        assert grid_back.n_nodes == grid.n_nodes

    def test_no_compression(
        self,
        tmp_path: Path,
        small_grid_nodes: list[dict],
        small_grid_elements: list[dict],
    ) -> None:
        """Test mesh without compression."""
        nodes = {d["id"]: Node(**d) for d in small_grid_nodes}
        elements = {d["id"]: Element(**d) for d in small_grid_elements}
        grid = AppGrid(nodes=nodes, elements=elements)

        filepath = tmp_path / "uncompressed.h5"
        with HDF5ModelWriter(filepath, compression=None) as writer:
            writer.write_mesh(grid)

        with HDF5ModelReader(filepath) as reader:
            grid_back = reader.read_mesh()

        assert grid_back.n_nodes == grid.n_nodes


# =============================================================================
# Additional tests for 95%+ coverage
# =============================================================================


class TestHDF5WriterErrorPaths:
    """Tests for HDF5ModelWriter when file is not open."""

    def test_write_mesh_not_open(self) -> None:
        """Test write_mesh raises RuntimeError when file not open."""
        writer = HDF5ModelWriter("dummy.h5")
        with pytest.raises(RuntimeError, match="File not open"):
            writer.write_mesh(AppGrid(nodes={}, elements={}))

    def test_write_stratigraphy_not_open(self) -> None:
        """Test write_stratigraphy raises RuntimeError when file not open."""
        writer = HDF5ModelWriter("dummy.h5")
        strat = Stratigraphy(
            n_layers=1, n_nodes=2,
            gs_elev=np.array([100.0, 100.0]),
            top_elev=np.array([[90.0], [90.0]]),
            bottom_elev=np.array([[80.0], [80.0]]),
            active_node=np.array([[True], [True]]),
        )
        with pytest.raises(RuntimeError, match="File not open"):
            writer.write_stratigraphy(strat)

    def test_write_timeseries_not_open(self) -> None:
        """Test write_timeseries raises RuntimeError when file not open."""
        writer = HDF5ModelWriter("dummy.h5")
        ts = TimeSeries(
            times=np.array(["2020-01-01"], dtype="datetime64[D]"),
            values=np.array([1.0]),
        )
        with pytest.raises(RuntimeError, match="File not open"):
            writer.write_timeseries(ts, "head")

    def test_write_metadata_not_open(self) -> None:
        """Test write_metadata raises RuntimeError when file not open."""
        writer = HDF5ModelWriter("dummy.h5")
        with pytest.raises(RuntimeError, match="File not open"):
            writer.write_metadata({"key": "value"})


class TestHDF5ReaderErrorPaths:
    """Tests for HDF5ModelReader when file is not open or data is missing."""

    def test_read_mesh_not_open(self) -> None:
        """Test read_mesh raises RuntimeError when file not open."""
        reader = HDF5ModelReader("dummy.h5")
        with pytest.raises(RuntimeError, match="File not open"):
            reader.read_mesh()

    def test_read_stratigraphy_not_open(self) -> None:
        """Test read_stratigraphy raises RuntimeError when file not open."""
        reader = HDF5ModelReader("dummy.h5")
        with pytest.raises(RuntimeError, match="File not open"):
            reader.read_stratigraphy()

    def test_read_timeseries_not_open(self) -> None:
        """Test read_timeseries raises RuntimeError when file not open."""
        reader = HDF5ModelReader("dummy.h5")
        with pytest.raises(RuntimeError, match="File not open"):
            reader.read_timeseries("head", "node_1")

    def test_list_timeseries_not_open(self) -> None:
        """Test list_timeseries raises RuntimeError when file not open."""
        reader = HDF5ModelReader("dummy.h5")
        with pytest.raises(RuntimeError, match="File not open"):
            reader.list_timeseries()

    def test_read_metadata_not_open(self) -> None:
        """Test read_metadata raises RuntimeError when file not open."""
        reader = HDF5ModelReader("dummy.h5")
        with pytest.raises(RuntimeError, match="File not open"):
            reader.read_metadata()

    def test_read_mesh_missing(self, tmp_path: Path) -> None:
        """Test read_mesh raises FileFormatError when no mesh data."""
        import h5py
        filepath = tmp_path / "empty.h5"
        with h5py.File(filepath, "w") as f:
            f.create_group("other")

        with HDF5ModelReader(filepath) as reader:
            with pytest.raises(FileFormatError, match="No mesh data"):
                reader.read_mesh()

    def test_read_stratigraphy_missing(self, tmp_path: Path) -> None:
        """Test read_stratigraphy raises FileFormatError when no stratigraphy."""
        import h5py
        filepath = tmp_path / "empty.h5"
        with h5py.File(filepath, "w") as f:
            f.create_group("other")

        with HDF5ModelReader(filepath) as reader:
            with pytest.raises(FileFormatError, match="No stratigraphy data"):
                reader.read_stratigraphy()

    def test_read_timeseries_no_group(self, tmp_path: Path) -> None:
        """Test read_timeseries raises when no timeseries group exists."""
        import h5py
        filepath = tmp_path / "empty.h5"
        with h5py.File(filepath, "w") as f:
            f.create_group("other")

        with HDF5ModelReader(filepath) as reader:
            with pytest.raises(FileFormatError, match="No timeseries"):
                reader.read_timeseries("head", "node_1")

    def test_read_timeseries_missing_variable(self, tmp_path: Path) -> None:
        """Test read_timeseries raises when variable not found."""
        import h5py
        filepath = tmp_path / "ts.h5"
        with h5py.File(filepath, "w") as f:
            ts_grp = f.create_group("timeseries")
            ts_grp.create_group("flow")

        with HDF5ModelReader(filepath) as reader:
            with pytest.raises(FileFormatError, match="Variable 'head' not found"):
                reader.read_timeseries("head", "node_1")

    def test_read_timeseries_missing_location(self, tmp_path: Path) -> None:
        """Test read_timeseries raises when location not found."""
        import h5py
        filepath = tmp_path / "ts.h5"
        with h5py.File(filepath, "w") as f:
            ts_grp = f.create_group("timeseries")
            var_grp = ts_grp.create_group("head")
            var_grp.create_group("node_1")

        with HDF5ModelReader(filepath) as reader:
            with pytest.raises(FileFormatError, match="Location 'node_99' not found"):
                reader.read_timeseries("head", "node_99")

    def test_list_timeseries_empty(self, tmp_path: Path) -> None:
        """Test list_timeseries returns empty dict when no timeseries group."""
        import h5py
        filepath = tmp_path / "empty.h5"
        with h5py.File(filepath, "w") as f:
            f.create_group("other")

        with HDF5ModelReader(filepath) as reader:
            result = reader.list_timeseries()
        assert result == {}


class TestHDF5MetadataEdgeCases:
    """Tests for metadata edge cases."""

    def test_write_metadata_datetime_value(self, tmp_path: Path) -> None:
        """Test writing metadata with datetime value."""
        filepath = tmp_path / "meta.h5"
        with HDF5ModelWriter(filepath) as writer:
            writer.write_metadata({
                "created": datetime(2020, 6, 15, 12, 0, 0),
                "name": "test",
                "count": 42,
                "ratio": 3.14,
                "active": True,
            })

        with HDF5ModelReader(filepath) as reader:
            meta = reader.read_metadata()

        assert meta["name"] == "test"
        assert meta["count"] == 42
        assert "2020-06-15" in meta["created"]

    def test_write_metadata_to_existing_group(self, tmp_path: Path) -> None:
        """Test writing metadata multiple times reuses existing group."""
        filepath = tmp_path / "meta.h5"
        with HDF5ModelWriter(filepath) as writer:
            writer.write_metadata({"key1": "value1"})
            writer.write_metadata({"key2": "value2"})

        with HDF5ModelReader(filepath) as reader:
            meta = reader.read_metadata()

        assert meta["key1"] == "value1"
        assert meta["key2"] == "value2"

    def test_read_metadata_no_metadata_group(self, tmp_path: Path) -> None:
        """Test reading metadata returns empty dict when no metadata group."""
        import h5py
        filepath = tmp_path / "empty.h5"
        with h5py.File(filepath, "w") as f:
            f.create_group("other")

        with HDF5ModelReader(filepath) as reader:
            meta = reader.read_metadata()
        assert meta == {}


class TestHDF5ModelIOEdgeCases:
    """Tests for model I/O edge cases."""

    def test_read_model_with_name_override(self, tmp_path: Path) -> None:
        """Test read_model with explicit name override."""
        model = IWFMModel(name="Original Name")
        filepath = tmp_path / "model.h5"
        with HDF5ModelWriter(filepath) as writer:
            writer.write_model(model)

        model_back = read_model_hdf5(filepath, name="Override Name")
        assert model_back.name == "Override Name"

    def test_write_model_no_mesh_no_strat(self, tmp_path: Path) -> None:
        """Test writing model with no mesh and no stratigraphy."""
        model = IWFMModel(name="Bare Model")
        filepath = tmp_path / "bare.h5"
        with HDF5ModelWriter(filepath) as writer:
            writer.write_model(model)

        model_back = read_model_hdf5(filepath)
        assert model_back.name == "Bare Model"
        assert model_back.mesh is None
        assert model_back.stratigraphy is None

    def test_read_model_no_mesh_no_strat(self, tmp_path: Path) -> None:
        """Test reading model when HDF5 has no mesh/strat groups."""
        import h5py
        filepath = tmp_path / "minimal.h5"
        with h5py.File(filepath, "w") as f:
            meta = f.create_group("metadata")
            meta.attrs["name"] = "Minimal"

        with HDF5ModelReader(filepath) as reader:
            model = reader.read_model()

        assert model.name == "Minimal"
        assert model.mesh is None
        assert model.stratigraphy is None


class TestHDF5MeshWithoutOptionalFields:
    """Test reading mesh when optional fields are missing."""

    def test_read_mesh_without_area_and_boundary(self, tmp_path: Path) -> None:
        """Test reading mesh with no area/is_boundary datasets."""
        import h5py
        filepath = tmp_path / "mesh_minimal.h5"
        with h5py.File(filepath, "w") as f:
            mesh_grp = f.create_group("mesh")
            nodes_grp = mesh_grp.create_group("nodes")
            nodes_grp.create_dataset("id", data=np.array([1, 2, 3], dtype=np.int32))
            nodes_grp.create_dataset("x", data=np.array([0.0, 100.0, 50.0]))
            nodes_grp.create_dataset("y", data=np.array([0.0, 0.0, 86.6]))

            elem_grp = mesh_grp.create_group("elements")
            elem_grp.create_dataset("id", data=np.array([1], dtype=np.int32))
            elem_grp.create_dataset("vertices", data=np.array([[1, 2, 3, 0]], dtype=np.int32))
            elem_grp.create_dataset("subregion", data=np.array([1], dtype=np.int32))

            mesh_grp.attrs["n_nodes"] = 3
            mesh_grp.attrs["n_elements"] = 1
            mesh_grp.attrs["n_subregions"] = 0

        with HDF5ModelReader(filepath) as reader:
            grid = reader.read_mesh()

        assert grid.n_nodes == 3
        assert grid.nodes[1].area == 0.0
        assert grid.nodes[1].is_boundary is False
        assert grid.n_elements == 1
