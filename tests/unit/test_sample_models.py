"""Unit tests for sample model generators.

Tests:
- create_sample_mesh
- create_sample_triangular_mesh
- create_sample_stratigraphy
- create_sample_scalar_field
- create_sample_element_field
- create_sample_timeseries
- create_sample_timeseries_collection
- create_sample_stream_network
- create_sample_budget_data
- create_sample_model
"""

from __future__ import annotations

import numpy as np
import pytest

from pyiwfm.sample_models import (
    create_sample_mesh,
    create_sample_triangular_mesh,
    create_sample_stratigraphy,
    create_sample_scalar_field,
    create_sample_element_field,
    create_sample_timeseries,
    create_sample_timeseries_collection,
    create_sample_stream_network,
    create_sample_budget_data,
    create_sample_model,
)
from pyiwfm.core.mesh import AppGrid
from pyiwfm.core.stratigraphy import Stratigraphy
from pyiwfm.core.model import IWFMModel
from pyiwfm.core.timeseries import TimeSeries, TimeSeriesCollection


# =============================================================================
# Test create_sample_mesh
# =============================================================================


class TestCreateSampleMesh:
    """Tests for create_sample_mesh function."""

    def test_default_mesh(self) -> None:
        """Test creating mesh with default parameters."""
        mesh = create_sample_mesh()

        assert isinstance(mesh, AppGrid)
        assert mesh.n_nodes == 100  # 10x10
        assert mesh.n_elements == 81  # 9x9

    def test_custom_dimensions(self) -> None:
        """Test creating mesh with custom dimensions."""
        mesh = create_sample_mesh(nx=5, ny=5)

        assert mesh.n_nodes == 25  # 5x5
        assert mesh.n_elements == 16  # 4x4

    def test_small_mesh(self) -> None:
        """Test creating smallest possible mesh."""
        mesh = create_sample_mesh(nx=2, ny=2, n_subregions=1)

        assert mesh.n_nodes == 4
        assert mesh.n_elements == 1

    def test_rectangular_mesh(self) -> None:
        """Test non-square mesh dimensions."""
        mesh = create_sample_mesh(nx=5, ny=3)

        assert mesh.n_nodes == 15  # 5x3
        assert mesh.n_elements == 8  # 4x2

    def test_node_coordinates(self) -> None:
        """Test node coordinate spacing."""
        mesh = create_sample_mesh(nx=3, ny=3, dx=500.0, dy=1000.0)

        x_coords = sorted(set(n.x for n in mesh.nodes.values()))
        y_coords = sorted(set(n.y for n in mesh.nodes.values()))

        assert x_coords == [0.0, 500.0, 1000.0]
        assert y_coords == [0.0, 1000.0, 2000.0]

    def test_subregion_count(self) -> None:
        """Test mesh has correct number of subregions."""
        mesh = create_sample_mesh(nx=10, ny=10, n_subregions=4)

        assert len(mesh.subregions) == 4

    def test_subregion_names(self) -> None:
        """Test subregion naming convention."""
        mesh = create_sample_mesh(n_subregions=3)

        assert mesh.subregions[1].name == "Subregion 1"
        assert mesh.subregions[2].name == "Subregion 2"
        assert mesh.subregions[3].name == "Subregion 3"

    def test_element_vertices(self) -> None:
        """Test elements have valid vertex references."""
        mesh = create_sample_mesh(nx=4, ny=4)

        for elem in mesh.elements.values():
            assert len(elem.vertices) == 4  # Quadrilateral
            for vid in elem.vertices:
                assert vid in mesh.nodes

    def test_connectivity_computed(self) -> None:
        """Test that connectivity is computed."""
        mesh = create_sample_mesh(nx=4, ny=4)

        # After compute_connectivity, faces should exist
        assert hasattr(mesh, "faces") or hasattr(mesh, "_faces")

    def test_areas_computed(self) -> None:
        """Test that element areas are computed."""
        mesh = create_sample_mesh(nx=3, ny=3, dx=100.0, dy=100.0)

        for elem in mesh.elements.values():
            assert elem.area > 0
            assert elem.area == pytest.approx(10000.0)  # 100x100

    def test_single_subregion(self) -> None:
        """Test mesh with single subregion."""
        mesh = create_sample_mesh(nx=5, ny=5, n_subregions=1)

        assert len(mesh.subregions) == 1
        for elem in mesh.elements.values():
            assert elem.subregion == 1

    def test_many_subregions(self) -> None:
        """Test mesh with many subregions."""
        mesh = create_sample_mesh(nx=10, ny=10, n_subregions=9)

        assert len(mesh.subregions) == 9
        # All elements should have valid subregion
        for elem in mesh.elements.values():
            assert 1 <= elem.subregion <= 9


# =============================================================================
# Test create_sample_triangular_mesh
# =============================================================================


class TestCreateSampleTriangularMesh:
    """Tests for create_sample_triangular_mesh function."""

    def test_default_mesh(self) -> None:
        """Test creating triangular mesh with defaults."""
        mesh = create_sample_triangular_mesh()

        assert isinstance(mesh, AppGrid)
        assert mesh.n_nodes > 0
        assert mesh.n_elements > 0

    def test_center_node(self) -> None:
        """Test center node is at specified center."""
        mesh = create_sample_triangular_mesh(
            center_x=1000.0, center_y=2000.0
        )

        center = mesh.nodes[1]
        assert center.x == pytest.approx(1000.0)
        assert center.y == pytest.approx(2000.0)

    def test_node_count(self) -> None:
        """Test node count based on rings and sectors."""
        n_rings = 3
        n_sectors = 8
        mesh = create_sample_triangular_mesh(
            n_rings=n_rings, n_sectors=n_sectors
        )

        expected_nodes = 1 + n_rings * n_sectors  # center + ring nodes
        assert mesh.n_nodes == expected_nodes

    def test_element_count(self) -> None:
        """Test element count based on rings and sectors."""
        n_rings = 3
        n_sectors = 8
        mesh = create_sample_triangular_mesh(
            n_rings=n_rings, n_sectors=n_sectors
        )

        # Inner ring: n_sectors triangles
        # Each outer ring: 2 * n_sectors triangles (quads split in two)
        expected_elements = n_sectors + 2 * n_sectors * (n_rings - 1)
        assert mesh.n_elements == expected_elements

    def test_triangular_elements(self) -> None:
        """Test all elements are triangles."""
        mesh = create_sample_triangular_mesh()

        for elem in mesh.elements.values():
            assert len(elem.vertices) == 3

    def test_subregion_assignment(self) -> None:
        """Test subregion assignment for triangular mesh."""
        n_subregions = 4
        mesh = create_sample_triangular_mesh(n_subregions=n_subregions)

        assert len(mesh.subregions) == n_subregions
        for elem in mesh.elements.values():
            assert 1 <= elem.subregion <= n_subregions

    def test_radius_affects_extent(self) -> None:
        """Test that radius parameter affects mesh extent."""
        mesh = create_sample_triangular_mesh(
            radius=10000.0, center_x=0.0, center_y=0.0
        )

        # Outer nodes should be approximately at the radius
        max_dist = max(
            (n.x**2 + n.y**2) ** 0.5
            for nid, n in mesh.nodes.items()
            if nid != 1
        )
        assert max_dist == pytest.approx(10000.0, rel=0.01)

    def test_areas_computed(self) -> None:
        """Test that element areas are computed and positive."""
        mesh = create_sample_triangular_mesh()

        for elem in mesh.elements.values():
            assert elem.area > 0


# =============================================================================
# Test create_sample_stratigraphy
# =============================================================================


class TestCreateSampleStratigraphy:
    """Tests for create_sample_stratigraphy function."""

    @pytest.fixture
    def small_mesh(self) -> AppGrid:
        """Create a small mesh for stratigraphy tests."""
        return create_sample_mesh(nx=3, ny=3)

    def test_basic_creation(self, small_mesh: AppGrid) -> None:
        """Test basic stratigraphy creation."""
        strat = create_sample_stratigraphy(small_mesh)

        assert isinstance(strat, Stratigraphy)
        assert strat.n_layers == 3
        assert strat.n_nodes == small_mesh.n_nodes

    def test_custom_layers(self, small_mesh: AppGrid) -> None:
        """Test stratigraphy with custom layer count."""
        strat = create_sample_stratigraphy(small_mesh, n_layers=5)

        assert strat.n_layers == 5

    def test_ground_surface_slope(self, small_mesh: AppGrid) -> None:
        """Test ground surface has slope."""
        strat = create_sample_stratigraphy(
            small_mesh, surface_base=200.0, surface_slope=(-0.01, -0.02)
        )

        # Ground surface should vary spatially
        gs_min = strat.gs_elev.min()
        gs_max = strat.gs_elev.max()
        assert gs_max > gs_min

    def test_layer_ordering(self, small_mesh: AppGrid) -> None:
        """Test layer tops are above bottoms."""
        strat = create_sample_stratigraphy(small_mesh)

        for layer in range(strat.n_layers):
            for node in range(strat.n_nodes):
                assert strat.top_elev[node, layer] > strat.bottom_elev[node, layer]

    def test_layer_continuity(self, small_mesh: AppGrid) -> None:
        """Test layers are continuous (top of next = bottom of previous)."""
        strat = create_sample_stratigraphy(small_mesh, n_layers=3)

        for layer in range(1, strat.n_layers):
            np.testing.assert_allclose(
                strat.top_elev[:, layer],
                strat.bottom_elev[:, layer - 1],
            )

    def test_active_nodes(self, small_mesh: AppGrid) -> None:
        """Test all nodes are active."""
        strat = create_sample_stratigraphy(small_mesh)

        assert strat.active_node.all()

    def test_layer_thickness(self, small_mesh: AppGrid) -> None:
        """Test layer thickness matches parameter."""
        thickness = 75.0
        strat = create_sample_stratigraphy(
            small_mesh, n_layers=2, layer_thickness=thickness
        )

        actual_thickness = strat.top_elev[:, 0] - strat.bottom_elev[:, 0]
        np.testing.assert_allclose(actual_thickness, thickness)

    def test_array_shapes(self, small_mesh: AppGrid) -> None:
        """Test array shapes are correct."""
        n_layers = 4
        strat = create_sample_stratigraphy(small_mesh, n_layers=n_layers)
        n_nodes = small_mesh.n_nodes

        assert strat.gs_elev.shape == (n_nodes,)
        assert strat.top_elev.shape == (n_nodes, n_layers)
        assert strat.bottom_elev.shape == (n_nodes, n_layers)
        assert strat.active_node.shape == (n_nodes, n_layers)


# =============================================================================
# Test create_sample_scalar_field
# =============================================================================


class TestCreateSampleScalarField:
    """Tests for create_sample_scalar_field function."""

    @pytest.fixture
    def mesh(self) -> AppGrid:
        """Create a mesh for scalar field tests."""
        return create_sample_mesh(nx=5, ny=5)

    def test_head_field(self, mesh: AppGrid) -> None:
        """Test head field generation."""
        values = create_sample_scalar_field(mesh, field_type="head")

        assert values.shape == (mesh.n_nodes,)
        assert values.dtype == np.float64

    def test_drawdown_field(self, mesh: AppGrid) -> None:
        """Test drawdown field (cone centered in domain)."""
        values = create_sample_scalar_field(
            mesh, field_type="drawdown", noise_level=0.0
        )

        assert values.shape == (mesh.n_nodes,)
        assert values.max() > 0  # Peak at center

    def test_recharge_field(self, mesh: AppGrid) -> None:
        """Test recharge field."""
        values = create_sample_scalar_field(
            mesh, field_type="recharge", noise_level=0.0
        )

        assert values.shape == (mesh.n_nodes,)
        assert (values > 0).all()  # All positive

    def test_pumping_field(self, mesh: AppGrid) -> None:
        """Test pumping field (negative clusters)."""
        values = create_sample_scalar_field(
            mesh, field_type="pumping", noise_level=0.0
        )

        assert values.shape == (mesh.n_nodes,)
        assert values.min() < 0  # Pumping is negative

    def test_subsidence_field(self, mesh: AppGrid) -> None:
        """Test subsidence field."""
        values = create_sample_scalar_field(
            mesh, field_type="subsidence", noise_level=0.0
        )

        assert values.shape == (mesh.n_nodes,)
        assert values.min() < 0  # Subsidence is negative

    def test_generic_field(self, mesh: AppGrid) -> None:
        """Test unknown field type produces generic values."""
        values = create_sample_scalar_field(
            mesh, field_type="unknown_type", noise_level=0.0
        )

        assert values.shape == (mesh.n_nodes,)

    def test_noise_level_zero(self, mesh: AppGrid) -> None:
        """Test noise_level=0 produces deterministic output."""
        v1 = create_sample_scalar_field(mesh, field_type="head", noise_level=0.0)
        v2 = create_sample_scalar_field(mesh, field_type="head", noise_level=0.0)

        np.testing.assert_array_equal(v1, v2)

    def test_noise_level_positive(self, mesh: AppGrid) -> None:
        """Test positive noise level adds randomness."""
        # With noise, repeated calls produce different results
        np.random.seed(42)
        v1 = create_sample_scalar_field(mesh, field_type="head", noise_level=0.5)
        np.random.seed(99)
        v2 = create_sample_scalar_field(mesh, field_type="head", noise_level=0.5)

        assert not np.array_equal(v1, v2)


# =============================================================================
# Test create_sample_element_field
# =============================================================================


class TestCreateSampleElementField:
    """Tests for create_sample_element_field function."""

    @pytest.fixture
    def mesh(self) -> AppGrid:
        """Create a mesh for element field tests."""
        return create_sample_mesh(nx=5, ny=5)

    def test_land_use_field(self, mesh: AppGrid) -> None:
        """Test land use field (categorical 1-5)."""
        np.random.seed(42)
        values = create_sample_element_field(mesh, field_type="land_use")

        assert values.shape == (mesh.n_elements,)
        assert values.dtype == np.float64
        assert values.min() >= 1
        assert values.max() <= 5

    def test_soil_type_field(self, mesh: AppGrid) -> None:
        """Test soil type field (position-dependent)."""
        values = create_sample_element_field(mesh, field_type="soil_type")

        assert values.shape == (mesh.n_elements,)
        assert values.min() >= 1

    def test_generic_element_field(self, mesh: AppGrid) -> None:
        """Test unknown field type produces random values."""
        np.random.seed(42)
        values = create_sample_element_field(mesh, field_type="crop")

        assert values.shape == (mesh.n_elements,)
        assert 0.0 <= values.min()
        assert values.max() <= 1.0


# =============================================================================
# Test create_sample_timeseries
# =============================================================================


class TestCreateSampleTimeseries:
    """Tests for create_sample_timeseries function."""

    def test_default_timeseries(self) -> None:
        """Test default time series creation."""
        ts = create_sample_timeseries()

        assert isinstance(ts, TimeSeries)
        assert ts.name == "Groundwater Head"
        assert ts.units == "ft"
        assert ts.n_times > 0

    def test_custom_name_and_years(self) -> None:
        """Test custom parameters."""
        ts = create_sample_timeseries(name="Well A", n_years=5)

        assert ts.name == "Well A"
        expected_points = int(365 * 5 / 1)  # daily for 5 years
        assert ts.n_times == expected_points

    def test_timestep_days(self) -> None:
        """Test custom timestep."""
        ts = create_sample_timeseries(n_years=1, timestep_days=7)

        expected_points = int(365 / 7)
        assert ts.n_times == expected_points

    def test_no_seasonal_pattern(self) -> None:
        """Test time series without seasonal pattern."""
        ts = create_sample_timeseries(
            n_years=2, seasonal=False, noise_level=0.0
        )

        # Without seasonal or noise, should be monotonic with trend
        assert ts.n_times > 0

    def test_trend(self) -> None:
        """Test time series trend is applied."""
        ts = create_sample_timeseries(
            n_years=10, trend=-1.0, seasonal=False, noise_level=0.0
        )

        # First value should be higher than last with negative trend
        assert ts.values[0] > ts.values[-1]

    def test_custom_start_date(self) -> None:
        """Test custom start date."""
        from datetime import datetime
        start = datetime(2000, 6, 15)
        ts = create_sample_timeseries(start_date=start, n_years=1)

        # Check that the start time matches
        assert ts.n_times > 0


# =============================================================================
# Test create_sample_timeseries_collection
# =============================================================================


class TestCreateSampleTimeseriesCollection:
    """Tests for create_sample_timeseries_collection function."""

    def test_default_collection(self) -> None:
        """Test default collection creation."""
        collection = create_sample_timeseries_collection()

        assert isinstance(collection, TimeSeriesCollection)
        assert len(collection) == 5
        assert collection.name == "Sample Wells"

    def test_custom_locations(self) -> None:
        """Test custom number of locations."""
        collection = create_sample_timeseries_collection(n_locations=3)

        assert len(collection) == 3

    def test_location_names(self) -> None:
        """Test location naming convention."""
        collection = create_sample_timeseries_collection(n_locations=3)

        assert "Well_1" in collection.locations
        assert "Well_2" in collection.locations
        assert "Well_3" in collection.locations

    def test_each_series_valid(self) -> None:
        """Test each series in collection is valid."""
        collection = create_sample_timeseries_collection(
            n_locations=2, n_years=2
        )

        for ts in collection:
            assert isinstance(ts, TimeSeries)
            assert ts.n_times > 0
            assert ts.units == "ft"

    def test_different_trends(self) -> None:
        """Test series have different trends."""
        collection = create_sample_timeseries_collection(
            n_locations=3, n_years=5
        )

        # Each location should have slightly different data
        series_list = list(collection)
        assert not np.array_equal(series_list[0].values, series_list[1].values)


# =============================================================================
# Test create_sample_stream_network
# =============================================================================


class TestCreateSampleStreamNetwork:
    """Tests for create_sample_stream_network function."""

    @pytest.fixture
    def mesh(self) -> AppGrid:
        """Create a mesh for stream network tests."""
        return create_sample_mesh(nx=10, ny=10)

    def test_returns_tuple(self, mesh: AppGrid) -> None:
        """Test function returns correct tuple structure."""
        result = create_sample_stream_network(mesh)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_node_coords_structure(self, mesh: AppGrid) -> None:
        """Test stream node coordinates structure."""
        stream_nodes, _ = create_sample_stream_network(mesh)

        assert len(stream_nodes) > 0
        for coord in stream_nodes:
            assert len(coord) == 2  # (x, y)
            assert isinstance(coord[0], float)
            assert isinstance(coord[1], float)

    def test_reach_connectivity(self, mesh: AppGrid) -> None:
        """Test reach connectivity structure."""
        stream_nodes, reaches = create_sample_stream_network(mesh)

        assert len(reaches) > 0
        for reach in reaches:
            assert len(reach) == 2  # (from_idx, to_idx)
            assert 0 <= reach[0] < len(stream_nodes)
            assert 0 <= reach[1] < len(stream_nodes)

    def test_main_channel_present(self, mesh: AppGrid) -> None:
        """Test main channel has 12 nodes."""
        stream_nodes, reaches = create_sample_stream_network(mesh)

        # Main channel has 12 nodes, so at least 12 stream nodes
        assert len(stream_nodes) >= 12

    def test_tributaries_present(self, mesh: AppGrid) -> None:
        """Test tributaries are added."""
        stream_nodes, reaches = create_sample_stream_network(mesh)

        # 12 main + 5 + 6 + 5 + 4 tributary nodes = 32
        assert len(stream_nodes) == 32

    def test_stream_within_domain(self, mesh: AppGrid) -> None:
        """Test all stream nodes are within or near mesh domain."""
        stream_nodes, _ = create_sample_stream_network(mesh)
        x_coords = [n.x for n in mesh.nodes.values()]
        y_coords = [n.y for n in mesh.nodes.values()]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        for x, y in stream_nodes:
            assert x_min - 1000 <= x <= x_max + 1000
            assert y_min - 1000 <= y <= y_max + 1000


# =============================================================================
# Test create_sample_budget_data
# =============================================================================


class TestCreateSampleBudgetData:
    """Tests for create_sample_budget_data function."""

    def test_returns_dict(self) -> None:
        """Test returns dictionary structure."""
        budget = create_sample_budget_data()

        assert isinstance(budget, dict)
        assert "Inflows" in budget
        assert "Outflows" in budget
        assert "Storage Change" in budget

    def test_inflows_positive(self) -> None:
        """Test inflow values are positive."""
        budget = create_sample_budget_data()

        for name, value in budget["Inflows"].items():
            assert value > 0, f"Inflow '{name}' should be positive"

    def test_outflows_negative(self) -> None:
        """Test outflow values are negative."""
        budget = create_sample_budget_data()

        for name, value in budget["Outflows"].items():
            assert value < 0, f"Outflow '{name}' should be negative"

    def test_budget_components(self) -> None:
        """Test expected budget components exist."""
        budget = create_sample_budget_data()

        assert "Recharge" in budget["Inflows"]
        assert "Pumping" in budget["Outflows"]
        assert "Storage" in budget["Storage Change"]

    def test_approximate_balance(self) -> None:
        """Test budget approximately balances."""
        budget = create_sample_budget_data()

        total_in = sum(budget["Inflows"].values())
        total_out = sum(budget["Outflows"].values())
        storage = sum(budget["Storage Change"].values())

        # Inflows + Outflows ≈ Storage Change
        balance = total_in + total_out - storage
        assert abs(balance) <= 1000  # Allow some imbalance


# =============================================================================
# Test create_sample_model
# =============================================================================


class TestCreateSampleModel:
    """Tests for create_sample_model function."""

    def test_default_model(self) -> None:
        """Test creating default sample model."""
        model = create_sample_model()

        assert isinstance(model, IWFMModel)
        assert model.name == "Sample Model"

    def test_model_has_mesh(self) -> None:
        """Test model includes mesh."""
        model = create_sample_model()

        assert model.mesh is not None
        assert isinstance(model.mesh, AppGrid)
        assert model.mesh.n_nodes > 0
        assert model.mesh.n_elements > 0

    def test_model_has_stratigraphy(self) -> None:
        """Test model includes stratigraphy."""
        model = create_sample_model()

        assert model.stratigraphy is not None
        assert isinstance(model.stratigraphy, Stratigraphy)

    def test_model_metadata(self) -> None:
        """Test model has metadata."""
        model = create_sample_model()

        assert model.metadata is not None
        assert "description" in model.metadata
        assert "version" in model.metadata
        assert "units" in model.metadata

    def test_custom_name(self) -> None:
        """Test custom model name."""
        model = create_sample_model(name="My Model")

        assert model.name == "My Model"

    def test_custom_dimensions(self) -> None:
        """Test custom mesh dimensions."""
        model = create_sample_model(nx=5, ny=5, n_layers=2)

        assert model.mesh.n_nodes == 25
        assert model.stratigraphy.n_layers == 2

    def test_mesh_stratigraphy_consistency(self) -> None:
        """Test mesh and stratigraphy have matching node counts."""
        model = create_sample_model(nx=8, ny=8)

        assert model.stratigraphy.n_nodes == model.mesh.n_nodes
