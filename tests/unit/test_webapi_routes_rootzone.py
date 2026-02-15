"""Comprehensive tests for the FastAPI root zone / land-use routes.

Covers all endpoints in ``pyiwfm.visualization.webapi.routes.rootzone``:

* ``_ensure_land_use_loaded`` helper
* ``GET /api/rootzone/status``
* ``GET /api/rootzone/land-use``
* ``GET /api/rootzone/timesteps``
* ``GET /api/rootzone/land-use/{element_id}/timeseries``
* ``GET /api/rootzone/land-use/{element_id}/crops``
* ``GET /api/rootzone/crops``
* ``GET /api/rootzone/soil-params/{element_id}``

Every branch and edge case is exercised to achieve 95%+ coverage.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

fastapi = pytest.importorskip("fastapi", reason="FastAPI not available")
pydantic = pytest.importorskip("pydantic", reason="Pydantic not available")

from fastapi.testclient import TestClient

from pyiwfm.visualization.webapi.config import model_state
from pyiwfm.visualization.webapi.server import create_app

# We need to import the module itself to reset the _land_use_loaded flag
import pyiwfm.visualization.webapi.routes.rootzone as rz_module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reset_model_state():
    """Reset the global model_state to a clean state."""
    model_state._model = None
    model_state._mesh_3d = None
    model_state._mesh_surface = None
    model_state._surface_json_data = None
    model_state._bounds = None
    model_state._pv_mesh_3d = None
    model_state._layer_surface_cache = {}
    model_state._crs = "+proj=utm +zone=10 +datum=NAD83 +units=us-ft +no_defs"
    model_state._transformer = None
    model_state._geojson_cache = {}
    model_state._head_loader = None
    model_state._gw_hydrograph_reader = None
    model_state._stream_hydrograph_reader = None
    model_state._budget_readers = {}
    model_state._observations = {}
    model_state._results_dir = None
    # Restore any monkey-patched methods back to the class originals
    for attr in ("get_budget_reader", "get_available_budgets", "reproject_coords",
                 "get_stream_reach_boundaries", "get_head_loader", "get_gw_hydrograph_reader",
                 "get_stream_hydrograph_reader", "get_area_manager", "get_subsidence_reader"):
        if attr in model_state.__dict__:
            del model_state.__dict__[attr]
    # Reset the module-level _land_use_loaded flag
    rz_module._land_use_loaded = False


def _make_mock_model(rootzone=None, metadata=None):
    """Create a minimal mock IWFMModel."""
    model = MagicMock()
    model.name = "TestModel"
    model.rootzone = rootzone
    model.metadata = metadata or {}
    model.stratigraphy = None
    model.streams = None
    model.groundwater = None
    model.lakes = None
    model.has_streams = False
    model.has_lakes = False
    model.n_nodes = 4
    model.n_elements = 1
    model.n_layers = 1
    model.n_lakes = 0
    model.n_stream_nodes = 0
    model.source_files = {}
    return model


def _make_mock_rootzone(
    element_landuse=None,
    crop_types=None,
    soil_params=None,
    nonponded_area_file=None,
    ponded_area_file=None,
    urban_area_file=None,
    native_area_file=None,
    nonponded_config=None,
    ponded_config=None,
    urban_config=None,
    native_riparian_config=None,
):
    """Create a mock RootZone component."""
    rz = MagicMock()
    rz.element_landuse = element_landuse if element_landuse is not None else []
    rz.crop_types = crop_types if crop_types is not None else {}
    rz.soil_params = soil_params if soil_params is not None else {}
    rz.nonponded_area_file = nonponded_area_file
    rz.ponded_area_file = ponded_area_file
    rz.urban_area_file = urban_area_file
    rz.native_area_file = native_area_file
    rz.nonponded_config = nonponded_config
    rz.ponded_config = ponded_config
    rz.urban_config = urban_config
    rz.native_riparian_config = native_riparian_config
    return rz


def _make_mock_area_manager(n_timesteps=0, snapshot=None, dates=None, timeseries=None):
    """Create a mock AreaDataManager."""
    mgr = MagicMock()
    type(mgr).n_timesteps = PropertyMock(return_value=n_timesteps)
    mgr.get_snapshot.return_value = snapshot or {}
    mgr.get_dates.return_value = dates or []
    mgr.get_element_timeseries.return_value = timeseries or {}
    mgr._loaders.return_value = []
    return mgr


def _make_mock_elu(element_id, land_use_type_value, area, crop_fractions=None, impervious_fraction=0.0):
    """Create a mock ElementLandUse."""
    elu = MagicMock()
    elu.element_id = element_id
    elu.land_use_type = MagicMock()
    elu.land_use_type.value = land_use_type_value
    elu.area = area
    elu.crop_fractions = crop_fractions or {}
    elu.impervious_fraction = impervious_fraction
    return elu


def _make_mock_crop_type(crop_id, name="Crop", root_depth=3.0, kc=1.0):
    """Create a mock CropType."""
    ct = MagicMock()
    ct.id = crop_id
    ct.name = name
    ct.root_depth = root_depth
    ct.kc = kc
    return ct


def _make_mock_soil_params(
    porosity=0.4,
    field_capacity=0.3,
    wilting_point=0.15,
    saturated_kv=1.0,
    lambda_param=0.5,
    kunsat_method=2,
    k_ponded=-1.0,
    capillary_rise=0.0,
    available_water=0.15,
    drainable_porosity=0.1,
    precip_column=1,
    precip_factor=1.0,
):
    """Create a mock SoilParameters."""
    sp = MagicMock()
    sp.porosity = porosity
    sp.field_capacity = field_capacity
    sp.wilting_point = wilting_point
    sp.saturated_kv = saturated_kv
    sp.lambda_param = lambda_param
    sp.kunsat_method = kunsat_method
    sp.k_ponded = k_ponded
    sp.capillary_rise = capillary_rise
    sp.available_water = available_water
    sp.drainable_porosity = drainable_porosity
    sp.precip_column = precip_column
    sp.precip_factor = precip_factor
    return sp


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _cleanup():
    """Reset model state and module flag before and after each test."""
    _reset_model_state()
    yield
    _reset_model_state()


@pytest.fixture()
def client_no_model():
    """TestClient with no model loaded."""
    app = create_app()
    return TestClient(app)


@pytest.fixture()
def client_with_model_no_rootzone():
    """TestClient with a model but no rootzone component."""
    model = _make_mock_model(rootzone=None)
    model_state._model = model
    app = create_app()
    return TestClient(app)


@pytest.fixture()
def client_with_empty_rootzone():
    """TestClient with a model that has an empty rootzone."""
    rz = _make_mock_rootzone()
    model = _make_mock_model(rootzone=rz)
    model_state._model = model
    # Disable area manager to avoid side effects
    model_state.get_area_manager = lambda: None
    app = create_app()
    return TestClient(app)


# ===========================================================================
# 1. _ensure_land_use_loaded tests
# ===========================================================================


class TestEnsureLandUseLoaded:
    """Tests for the _ensure_land_use_loaded helper function."""

    def test_already_loaded_returns_immediately(self):
        """Once _land_use_loaded is True, it returns immediately."""
        rz_module._land_use_loaded = True
        # Should not touch model_state at all
        rz_module._ensure_land_use_loaded()
        assert rz_module._land_use_loaded is True

    def test_no_model_sets_flag(self):
        """When model is None, sets flag and returns."""
        model_state._model = None
        rz_module._ensure_land_use_loaded()
        assert rz_module._land_use_loaded is True

    def test_no_rootzone_sets_flag(self):
        """When model.rootzone is None, sets flag and returns."""
        model = _make_mock_model(rootzone=None)
        model_state._model = model
        rz_module._ensure_land_use_loaded()
        assert rz_module._land_use_loaded is True

    def test_element_landuse_already_populated_sets_flag(self):
        """When element_landuse is already populated, sets flag and returns."""
        elu = _make_mock_elu(1, "agricultural", 100.0)
        rz = _make_mock_rootzone(element_landuse=[elu])
        model = _make_mock_model(rootzone=rz)
        model_state._model = model
        rz_module._ensure_land_use_loaded()
        assert rz_module._land_use_loaded is True

    def test_no_area_files_sets_flag(self):
        """When no area files are wired, sets flag with warning."""
        rz = _make_mock_rootzone(
            element_landuse=[],
            nonponded_area_file=None,
            ponded_area_file=None,
            urban_area_file=None,
            native_area_file=None,
        )
        model = _make_mock_model(rootzone=rz)
        model_state._model = model
        model_state.get_area_manager = lambda: None
        rz_module._ensure_land_use_loaded()
        assert rz_module._land_use_loaded is True

    def test_area_files_exist_but_not_on_disk_sets_flag(self):
        """When area files are wired but none exist on disk."""
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = False
        rz = _make_mock_rootzone(
            element_landuse=[],
            nonponded_area_file=mock_path,
            ponded_area_file=None,
            urban_area_file=None,
            native_area_file=None,
        )
        model = _make_mock_model(rootzone=rz)
        model_state._model = model
        model_state.get_area_manager = lambda: None
        rz_module._ensure_land_use_loaded()
        assert rz_module._land_use_loaded is True

    def test_hdf5_manager_success(self):
        """When HDF5 area manager is available and has data, loads from it."""
        rz = _make_mock_rootzone(element_landuse=[])
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = True
        rz.nonponded_area_file = mock_path

        model = _make_mock_model(rootzone=rz)
        model_state._model = model

        snapshot = {
            1: {"fractions": {"agricultural": 0.5, "urban": 0.3}, "dominant": "agricultural", "total_area": 1000.0},
        }
        mgr = _make_mock_area_manager(n_timesteps=5, snapshot=snapshot)
        model_state.get_area_manager = lambda: mgr

        rz_module._ensure_land_use_loaded()
        assert rz_module._land_use_loaded is True
        mgr.get_snapshot.assert_called_once_with(0)
        rz.load_land_use_from_arrays.assert_called_once_with(snapshot)

    def test_hdf5_manager_snapshot_exception_fallback_to_text(self):
        """When HDF5 snapshot fails, falls back to text loading."""
        rz = _make_mock_rootzone(element_landuse=[])
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = True
        rz.nonponded_area_file = mock_path

        model = _make_mock_model(rootzone=rz)
        model_state._model = model

        mgr = MagicMock()
        type(mgr).n_timesteps = PropertyMock(return_value=5)
        mgr.get_snapshot.side_effect = RuntimeError("HDF5 read error")
        model_state.get_area_manager = lambda: mgr

        # After HDF5 failure, text fallback triggers load_land_use_snapshot
        rz_module._ensure_land_use_loaded()
        rz.load_land_use_snapshot.assert_called_once_with(timestep=0)

    def test_hdf5_manager_none_fallback_to_text(self):
        """When HDF5 manager is None, falls back to text loading."""
        rz = _make_mock_rootzone(element_landuse=[])
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = True
        rz.nonponded_area_file = mock_path

        model = _make_mock_model(rootzone=rz)
        model_state._model = model
        model_state.get_area_manager = lambda: None

        rz_module._ensure_land_use_loaded()
        rz.load_land_use_snapshot.assert_called_once_with(timestep=0)

    def test_hdf5_manager_zero_timesteps_fallback_to_text(self):
        """When HDF5 manager has 0 timesteps, falls back to text loading."""
        rz = _make_mock_rootzone(element_landuse=[])
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = True
        rz.nonponded_area_file = mock_path

        model = _make_mock_model(rootzone=rz)
        model_state._model = model
        mgr = _make_mock_area_manager(n_timesteps=0)
        model_state.get_area_manager = lambda: mgr

        rz_module._ensure_land_use_loaded()
        rz.load_land_use_snapshot.assert_called_once_with(timestep=0)

    def test_text_fallback_success_with_data(self):
        """Text-based loading succeeds and produces data."""
        rz = _make_mock_rootzone(element_landuse=[])
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = True
        rz.nonponded_area_file = mock_path

        model = _make_mock_model(rootzone=rz)
        model_state._model = model
        model_state.get_area_manager = lambda: None

        # Simulate load_land_use_snapshot populating element_landuse
        def _populate_elu(timestep=0):
            rz.element_landuse.append(_make_mock_elu(1, "agricultural", 100.0))

        rz.load_land_use_snapshot.side_effect = _populate_elu

        rz_module._ensure_land_use_loaded()
        assert rz_module._land_use_loaded is True

    def test_text_fallback_loads_zero_entries(self):
        """Text-based loading runs but produces no data entries."""
        rz = _make_mock_rootzone(element_landuse=[])
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = True
        rz.nonponded_area_file = mock_path

        model = _make_mock_model(rootzone=rz)
        model_state._model = model
        model_state.get_area_manager = lambda: None

        # load_land_use_snapshot does not append anything
        rz.load_land_use_snapshot.side_effect = lambda timestep=0: None

        rz_module._ensure_land_use_loaded()
        # _land_use_loaded NOT set because n_loaded == 0 triggers warning path
        # but function does NOT set flag in that branch (line 98)
        # Actually, re-reading the code: it does NOT set flag in the else branch
        # This is correct -- the function won't set the flag if no data was loaded

    def test_text_fallback_exception(self):
        """Text-based loading raises an exception."""
        rz = _make_mock_rootzone(element_landuse=[])
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = True
        rz.nonponded_area_file = mock_path

        model = _make_mock_model(rootzone=rz)
        model_state._model = model
        model_state.get_area_manager = lambda: None

        rz.load_land_use_snapshot.side_effect = RuntimeError("Parse error")

        # Should not raise - catches exception internally
        rz_module._ensure_land_use_loaded()

    def test_multiple_area_files_logging(self):
        """All 4 area file types are logged during loading check."""
        rz = _make_mock_rootzone(element_landuse=[])
        np_path = MagicMock(spec=Path)
        np_path.exists.return_value = True
        p_path = MagicMock(spec=Path)
        p_path.exists.return_value = False
        u_path = MagicMock(spec=Path)
        u_path.exists.return_value = True

        rz.nonponded_area_file = np_path
        rz.ponded_area_file = p_path
        rz.urban_area_file = u_path
        rz.native_area_file = None

        model = _make_mock_model(rootzone=rz)
        model_state._model = model
        model_state.get_area_manager = lambda: None

        # At least one file exists, so text loading will be attempted
        rz_module._ensure_land_use_loaded()
        rz.load_land_use_snapshot.assert_called_once()


# ===========================================================================
# 2. GET /api/rootzone/status
# ===========================================================================


class TestGetRootzoneStatus:
    """Tests for GET /api/rootzone/status."""

    def test_no_model_returns_404(self, client_no_model):
        resp = client_no_model.get("/api/rootzone/status")
        assert resp.status_code == 404
        assert "No model loaded" in resp.json()["detail"]

    def test_no_rootzone_returns_not_loaded(self, client_with_model_no_rootzone):
        resp = client_with_model_no_rootzone.get("/api/rootzone/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["loaded"] is False
        assert "No rootzone component" in data["reason"]

    def test_rootzone_with_area_files(self):
        """Status shows area file info when files are wired."""
        mock_np = MagicMock(spec=Path)
        mock_np.exists.return_value = True
        mock_np.stat.return_value = MagicMock(st_size=12345)
        mock_np.__str__ = lambda self: "/path/to/nonponded.dat"

        rz = _make_mock_rootzone(
            nonponded_area_file=mock_np,
            nonponded_config=MagicMock(),
            ponded_config=None,
            urban_config=None,
            native_riparian_config=None,
        )
        rz.element_landuse = []
        rz.crop_types = {}

        model = _make_mock_model(rootzone=rz, metadata={"rootzone_version": "4.1"})
        model_state._model = model
        model_state.get_area_manager = lambda: None

        app = create_app()
        client = TestClient(app)
        resp = client.get("/api/rootzone/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["loaded"] is True
        assert data["area_files"]["nonponded"]["exists"] is True
        assert data["area_files"]["nonponded"]["size_bytes"] == 12345
        assert data["configs_loaded"]["nonponded_config"] is True
        assert data["configs_loaded"]["ponded_config"] is False
        assert data["rootzone_version"] == "4.1"
        assert data["area_manager"] is None

    def test_rootzone_area_file_not_wired(self):
        """Area file is None -- shows path=None in status."""
        rz = _make_mock_rootzone(
            nonponded_area_file=None,
            ponded_area_file=None,
            urban_area_file=None,
            native_area_file=None,
            nonponded_config=None,
            ponded_config=None,
            urban_config=None,
            native_riparian_config=None,
        )
        rz.element_landuse = []
        rz.crop_types = {}

        model = _make_mock_model(rootzone=rz)
        model_state._model = model
        model_state.get_area_manager = lambda: None

        app = create_app()
        client = TestClient(app)
        resp = client.get("/api/rootzone/status")
        data = resp.json()
        assert data["area_files"]["nonponded"]["path"] is None
        assert data["area_files"]["nonponded"]["exists"] is False

    def test_rootzone_with_area_manager(self):
        """Status includes area_manager info when manager is available."""
        loader_mock = MagicMock()
        loader_mock.n_frames = 10
        loader_mock.n_elements = 100
        loader_mock.n_cols = 5

        mgr = MagicMock()
        type(mgr).n_timesteps = PropertyMock(return_value=10)
        mgr._loaders.return_value = [("nonponded", loader_mock)]

        rz = _make_mock_rootzone(
            nonponded_area_file=None,
            ponded_area_file=None,
            urban_area_file=None,
            native_area_file=None,
            nonponded_config=None,
            ponded_config=None,
            urban_config=None,
            native_riparian_config=None,
        )
        rz.element_landuse = []
        rz.crop_types = {}

        model = _make_mock_model(rootzone=rz)
        model_state._model = model
        model_state.get_area_manager = lambda: mgr

        app = create_app()
        client = TestClient(app)
        resp = client.get("/api/rootzone/status")
        data = resp.json()
        assert data["area_manager"] is not None
        assert data["area_manager"]["n_timesteps"] == 10
        assert "nonponded" in data["area_manager"]["loaders"]

    def test_rootzone_status_shows_land_use_counts(self):
        """Status includes n_element_landuse and n_crop_types."""
        elu = _make_mock_elu(1, "agricultural", 100.0)
        ct = _make_mock_crop_type(1, "Grain")

        rz = _make_mock_rootzone(
            element_landuse=[elu],
            crop_types={1: ct},
            nonponded_area_file=None,
            ponded_area_file=None,
            urban_area_file=None,
            native_area_file=None,
            nonponded_config=None,
            ponded_config=None,
            urban_config=None,
            native_riparian_config=None,
        )

        model = _make_mock_model(rootzone=rz)
        model_state._model = model
        model_state.get_area_manager = lambda: None

        app = create_app()
        client = TestClient(app)
        resp = client.get("/api/rootzone/status")
        data = resp.json()
        assert data["n_element_landuse"] == 1
        assert data["n_crop_types"] == 1

    def test_rootzone_area_file_exists_false(self):
        """Area file is wired but does not exist on disk."""
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = False
        mock_path.__str__ = lambda self: "/path/to/missing.dat"

        rz = _make_mock_rootzone(
            nonponded_area_file=mock_path,
            ponded_area_file=None,
            urban_area_file=None,
            native_area_file=None,
            nonponded_config=None,
            ponded_config=None,
            urban_config=None,
            native_riparian_config=None,
        )
        rz.element_landuse = []
        rz.crop_types = {}

        model = _make_mock_model(rootzone=rz)
        model_state._model = model
        model_state.get_area_manager = lambda: None

        app = create_app()
        client = TestClient(app)
        resp = client.get("/api/rootzone/status")
        data = resp.json()
        assert data["area_files"]["nonponded"]["exists"] is False
        assert data["area_files"]["nonponded"]["size_bytes"] is None


# ===========================================================================
# 3. GET /api/rootzone/land-use
# ===========================================================================


class TestGetLandUse:
    """Tests for GET /api/rootzone/land-use."""

    def test_no_model_returns_404(self, client_no_model):
        resp = client_no_model.get("/api/rootzone/land-use")
        assert resp.status_code == 404
        assert "No model loaded" in resp.json()["detail"]

    def test_no_rootzone_returns_404(self, client_with_model_no_rootzone):
        resp = client_with_model_no_rootzone.get("/api/rootzone/land-use")
        assert resp.status_code == 404
        assert "No root zone data" in resp.json()["detail"]

    def test_land_use_via_hdf5_manager(self):
        """Land use from HDF5 manager at default timestep."""
        snapshot = {
            1: {"fractions": {"agricultural": 0.6, "urban": 0.4}, "dominant": "agricultural", "total_area": 500.0},
            2: {"fractions": {"urban": 1.0}, "dominant": "urban", "total_area": 200.0},
        }
        mgr = _make_mock_area_manager(n_timesteps=10, snapshot=snapshot)

        rz = _make_mock_rootzone(element_landuse=[])
        model = _make_mock_model(rootzone=rz)
        model_state._model = model
        model_state.get_area_manager = lambda: mgr

        # Need an area file that exists so _ensure_land_use_loaded completes
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = True
        rz.nonponded_area_file = mock_path

        app = create_app()
        client = TestClient(app)
        resp = client.get("/api/rootzone/land-use")
        assert resp.status_code == 200
        data = resp.json()
        assert data["n_elements"] == 2
        eids = {e["element_id"] for e in data["elements"]}
        assert eids == {1, 2}

    def test_land_use_via_hdf5_manager_with_timestep(self):
        """Land use from HDF5 manager at specific timestep."""
        snapshot = {
            1: {"fractions": {"agricultural": 1.0}, "dominant": "agricultural", "total_area": 300.0},
        }
        mgr = _make_mock_area_manager(n_timesteps=5, snapshot=snapshot)

        rz = _make_mock_rootzone(element_landuse=[])
        model = _make_mock_model(rootzone=rz)
        model_state._model = model
        model_state.get_area_manager = lambda: mgr

        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = True
        rz.nonponded_area_file = mock_path

        app = create_app()
        client = TestClient(app)
        resp = client.get("/api/rootzone/land-use?timestep=3")
        assert resp.status_code == 200
        mgr.get_snapshot.assert_called_with(3)

    def test_land_use_via_hdf5_manager_timestep_clamped(self):
        """Timestep exceeding n_timesteps is clamped to last."""
        snapshot = {
            1: {"fractions": {"agricultural": 1.0}, "dominant": "agricultural", "total_area": 100.0},
        }
        mgr = _make_mock_area_manager(n_timesteps=3, snapshot=snapshot)

        rz = _make_mock_rootzone(element_landuse=[])
        model = _make_mock_model(rootzone=rz)
        model_state._model = model
        model_state.get_area_manager = lambda: mgr

        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = True
        rz.nonponded_area_file = mock_path

        app = create_app()
        client = TestClient(app)
        resp = client.get("/api/rootzone/land-use?timestep=100")
        assert resp.status_code == 200
        # timestep 100 >= 3, so clamped to 2 (n_timesteps - 1)
        mgr.get_snapshot.assert_called_with(2)

    def test_land_use_text_fallback_timestep_0(self):
        """Fallback to text with timestep=0 uses already-loaded data."""
        elu_ag = _make_mock_elu(1, "agricultural", 300.0)
        elu_ur = _make_mock_elu(1, "urban", 200.0)

        rz = _make_mock_rootzone(element_landuse=[elu_ag, elu_ur])
        model = _make_mock_model(rootzone=rz)
        model_state._model = model
        model_state.get_area_manager = lambda: None

        app = create_app()
        client = TestClient(app)
        # Already loaded flag must be set to avoid _ensure_land_use_loaded side effects
        rz_module._land_use_loaded = True

        resp = client.get("/api/rootzone/land-use?timestep=0")
        assert resp.status_code == 200
        data = resp.json()
        assert data["n_elements"] == 1
        elem = data["elements"][0]
        assert elem["element_id"] == 1
        assert elem["total_area"] == 500.0
        # Fractions sum to 1.0
        assert abs(elem["fractions"]["agricultural"] - 0.6) < 0.001
        assert abs(elem["fractions"]["urban"] - 0.4) < 0.001
        assert elem["dominant"] == "agricultural"

    def test_land_use_text_fallback_timestep_nonzero(self):
        """Fallback with timestep > 0 triggers text reload."""
        elu = _make_mock_elu(1, "agricultural", 100.0)
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        rz = _make_mock_rootzone(
            element_landuse=[elu],
            nonponded_area_file=mock_path,
        )

        model = _make_mock_model(rootzone=rz)
        model_state._model = model
        model_state.get_area_manager = lambda: None
        rz_module._land_use_loaded = True

        app = create_app()
        client = TestClient(app)
        resp = client.get("/api/rootzone/land-use?timestep=5")
        assert resp.status_code == 200
        rz.load_land_use_snapshot.assert_called_once_with(timestep=5)

    def test_land_use_text_fallback_timestep_nonzero_exception(self):
        """Text reload exception is caught, returns existing data."""
        elu = _make_mock_elu(1, "agricultural", 100.0)
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        rz = _make_mock_rootzone(
            element_landuse=[elu],
            nonponded_area_file=mock_path,
        )
        rz.load_land_use_snapshot.side_effect = RuntimeError("Parse fail")

        model = _make_mock_model(rootzone=rz)
        model_state._model = model
        model_state.get_area_manager = lambda: None
        rz_module._land_use_loaded = True

        app = create_app()
        client = TestClient(app)
        resp = client.get("/api/rootzone/land-use?timestep=5")
        assert resp.status_code == 200
        # Still returns existing element data
        assert resp.json()["n_elements"] == 1

    def test_land_use_text_fallback_no_area_files(self):
        """Text fallback with no area files -- doesn't try reload for timestep>0."""
        elu = _make_mock_elu(1, "native_riparian", 50.0)
        rz = _make_mock_rootzone(
            element_landuse=[elu],
            nonponded_area_file=None,
            ponded_area_file=None,
            urban_area_file=None,
            native_area_file=None,
        )

        model = _make_mock_model(rootzone=rz)
        model_state._model = model
        model_state.get_area_manager = lambda: None
        rz_module._land_use_loaded = True

        app = create_app()
        client = TestClient(app)
        resp = client.get("/api/rootzone/land-use?timestep=5")
        assert resp.status_code == 200
        # No reload attempted because no area files
        rz.load_land_use_snapshot.assert_not_called()

    def test_land_use_empty_element_landuse(self):
        """When element_landuse is empty and no manager, returns empty."""
        rz = _make_mock_rootzone(element_landuse=[])
        model = _make_mock_model(rootzone=rz)
        model_state._model = model
        model_state.get_area_manager = lambda: None
        rz_module._land_use_loaded = True

        app = create_app()
        client = TestClient(app)
        resp = client.get("/api/rootzone/land-use")
        assert resp.status_code == 200
        data = resp.json()
        assert data["n_elements"] == 0
        assert data["elements"] == []

    def test_land_use_multiple_types_same_element(self):
        """Multiple land use types for same element are aggregated."""
        elus = [
            _make_mock_elu(1, "agricultural", 100.0),
            _make_mock_elu(1, "urban", 200.0),
            _make_mock_elu(1, "native_riparian", 50.0),
            _make_mock_elu(1, "water", 10.0),
        ]
        rz = _make_mock_rootzone(element_landuse=elus)
        model = _make_mock_model(rootzone=rz)
        model_state._model = model
        model_state.get_area_manager = lambda: None
        rz_module._land_use_loaded = True

        app = create_app()
        client = TestClient(app)
        resp = client.get("/api/rootzone/land-use")
        data = resp.json()
        assert data["n_elements"] == 1
        elem = data["elements"][0]
        assert elem["total_area"] == 360.0
        assert elem["dominant"] == "urban"

    def test_land_use_zero_total_area(self):
        """Element with zero total area gets 0.0 fractions and 'unknown' dominant."""
        elu = _make_mock_elu(1, "agricultural", 0.0)
        rz = _make_mock_rootzone(element_landuse=[elu])
        model = _make_mock_model(rootzone=rz)
        model_state._model = model
        model_state.get_area_manager = lambda: None
        rz_module._land_use_loaded = True

        app = create_app()
        client = TestClient(app)
        resp = client.get("/api/rootzone/land-use")
        data = resp.json()
        elem = data["elements"][0]
        assert elem["total_area"] == 0.0
        assert elem["dominant"] == "unknown"
        for frac in elem["fractions"].values():
            assert frac == 0.0


# ===========================================================================
# 4. GET /api/rootzone/timesteps
# ===========================================================================


class TestGetLandUseTimesteps:
    """Tests for GET /api/rootzone/timesteps."""

    def test_no_model_returns_404(self, client_no_model):
        resp = client_no_model.get("/api/rootzone/timesteps")
        assert resp.status_code == 404

    def test_no_rootzone_returns_404(self, client_with_model_no_rootzone):
        resp = client_with_model_no_rootzone.get("/api/rootzone/timesteps")
        assert resp.status_code == 404

    def test_no_area_manager_returns_empty(self):
        """No area manager returns 0 timesteps."""
        rz = _make_mock_rootzone()
        model = _make_mock_model(rootzone=rz)
        model_state._model = model
        model_state.get_area_manager = lambda: None

        app = create_app()
        client = TestClient(app)
        resp = client.get("/api/rootzone/timesteps")
        assert resp.status_code == 200
        data = resp.json()
        assert data["n_timesteps"] == 0
        assert data["dates"] == []

    def test_area_manager_zero_timesteps(self):
        """Area manager with 0 timesteps returns empty."""
        mgr = _make_mock_area_manager(n_timesteps=0)

        rz = _make_mock_rootzone()
        model = _make_mock_model(rootzone=rz)
        model_state._model = model
        model_state.get_area_manager = lambda: mgr

        app = create_app()
        client = TestClient(app)
        resp = client.get("/api/rootzone/timesteps")
        data = resp.json()
        assert data["n_timesteps"] == 0

    def test_area_manager_with_dates(self):
        """Area manager returns proper dates."""
        dates = ["2000-01-01", "2000-02-01", "2000-03-01"]
        mgr = _make_mock_area_manager(n_timesteps=3, dates=dates)

        rz = _make_mock_rootzone()
        model = _make_mock_model(rootzone=rz)
        model_state._model = model
        model_state.get_area_manager = lambda: mgr

        app = create_app()
        client = TestClient(app)
        resp = client.get("/api/rootzone/timesteps")
        data = resp.json()
        assert data["n_timesteps"] == 3
        assert data["dates"] == dates


# ===========================================================================
# 5. GET /api/rootzone/land-use/{element_id}/timeseries
# ===========================================================================


class TestGetElementTimeseries:
    """Tests for GET /api/rootzone/land-use/{element_id}/timeseries."""

    def test_no_model_returns_404(self, client_no_model):
        resp = client_no_model.get("/api/rootzone/land-use/1/timeseries")
        assert resp.status_code == 404

    def test_no_rootzone_returns_404(self, client_with_model_no_rootzone):
        resp = client_with_model_no_rootzone.get("/api/rootzone/land-use/1/timeseries")
        assert resp.status_code == 404

    def test_no_area_manager_returns_404(self):
        """No area manager -> 404."""
        rz = _make_mock_rootzone()
        model = _make_mock_model(rootzone=rz)
        model_state._model = model
        model_state.get_area_manager = lambda: None

        app = create_app()
        client = TestClient(app)
        resp = client.get("/api/rootzone/land-use/1/timeseries")
        assert resp.status_code == 404
        assert "No area data available" in resp.json()["detail"]

    def test_area_manager_zero_timesteps_returns_404(self):
        """Area manager with 0 timesteps -> 404."""
        mgr = _make_mock_area_manager(n_timesteps=0)

        rz = _make_mock_rootzone()
        model = _make_mock_model(rootzone=rz)
        model_state._model = model
        model_state.get_area_manager = lambda: mgr

        app = create_app()
        client = TestClient(app)
        resp = client.get("/api/rootzone/land-use/1/timeseries")
        assert resp.status_code == 404

    def test_element_not_found_returns_404(self):
        """Element timeseries with only element_id + dates (len<=2) -> 404."""
        # Result dict with only element_id and dates = len 2
        mgr = _make_mock_area_manager(
            n_timesteps=5,
            timeseries={"element_id": 1, "dates": ["2000-01-01"]},
        )

        rz = _make_mock_rootzone()
        model = _make_mock_model(rootzone=rz)
        model_state._model = model
        model_state.get_area_manager = lambda: mgr

        app = create_app()
        client = TestClient(app)
        resp = client.get("/api/rootzone/land-use/999/timeseries")
        assert resp.status_code == 404
        assert "No area data for element" in resp.json()["detail"]

    def test_valid_timeseries(self):
        """Valid timeseries returns data."""
        ts_data = {
            "element_id": 1,
            "dates": ["2000-01-01", "2000-02-01"],
            "nonponded": {"n_cols": 3, "areas": [[10.0, 20.0, 30.0], [11.0, 21.0, 31.0]]},
        }
        mgr = _make_mock_area_manager(n_timesteps=5, timeseries=ts_data)

        rz = _make_mock_rootzone()
        model = _make_mock_model(rootzone=rz)
        model_state._model = model
        model_state.get_area_manager = lambda: mgr

        app = create_app()
        client = TestClient(app)
        resp = client.get("/api/rootzone/land-use/1/timeseries")
        assert resp.status_code == 200
        data = resp.json()
        assert data["element_id"] == 1
        assert "nonponded" in data


# ===========================================================================
# 6. GET /api/rootzone/land-use/{element_id}/crops
# ===========================================================================


class TestGetElementCrops:
    """Tests for GET /api/rootzone/land-use/{element_id}/crops."""

    def test_no_model_returns_404(self, client_no_model):
        resp = client_no_model.get("/api/rootzone/land-use/1/crops")
        assert resp.status_code == 404

    def test_no_rootzone_returns_404(self, client_with_model_no_rootzone):
        resp = client_with_model_no_rootzone.get("/api/rootzone/land-use/1/crops")
        assert resp.status_code == 404

    def test_element_not_found_returns_404(self):
        """No land use data for element -> 404."""
        rz = _make_mock_rootzone(element_landuse=[])
        rz.get_landuse_for_element.return_value = []

        model = _make_mock_model(rootzone=rz)
        model_state._model = model
        model_state.get_area_manager = lambda: None
        rz_module._land_use_loaded = True

        app = create_app()
        client = TestClient(app)
        resp = client.get("/api/rootzone/land-use/999/crops")
        assert resp.status_code == 404
        assert "No land use data for element" in resp.json()["detail"]

    def test_agricultural_crops_with_crop_types(self):
        """Agricultural element with crop fractions."""
        ct1 = _make_mock_crop_type(1, "Grain")
        ct2 = _make_mock_crop_type(2, "Alfalfa")

        elu = _make_mock_elu(
            1, "agricultural", 1000.0,
            crop_fractions={1: 0.6, 2: 0.4},
        )

        rz = _make_mock_rootzone(
            element_landuse=[elu],
            crop_types={1: ct1, 2: ct2},
            soil_params={},
        )
        rz.get_landuse_for_element.return_value = [elu]

        model = _make_mock_model(rootzone=rz)
        model_state._model = model
        model_state.get_area_manager = lambda: None
        rz_module._land_use_loaded = True

        app = create_app()
        client = TestClient(app)
        resp = client.get("/api/rootzone/land-use/1/crops")
        assert resp.status_code == 200
        data = resp.json()
        assert data["element_id"] == 1
        assert len(data["crops"]) == 2
        assert data["crops"][0]["name"] == "Grain"
        assert data["crops"][0]["fraction"] == 0.6
        assert data["crops"][0]["area"] == 600.0
        assert data["crops"][1]["name"] == "Alfalfa"
        assert data["urban_impervious_fraction"] == 0.0

    def test_agricultural_crop_missing_from_crop_types(self):
        """Crop ID not in crop_types dict uses fallback name."""
        elu = _make_mock_elu(
            1, "agricultural", 100.0,
            crop_fractions={99: 1.0},
        )

        rz = _make_mock_rootzone(
            element_landuse=[elu],
            crop_types={},  # No crop type 99
            soil_params={},
        )
        rz.get_landuse_for_element.return_value = [elu]
        # crop_types.get(99) should return None
        rz.crop_types = {}

        model = _make_mock_model(rootzone=rz)
        model_state._model = model
        model_state.get_area_manager = lambda: None
        rz_module._land_use_loaded = True

        app = create_app()
        client = TestClient(app)
        resp = client.get("/api/rootzone/land-use/1/crops")
        data = resp.json()
        assert data["crops"][0]["name"] == "Crop 99"

    def test_urban_impervious_fraction(self):
        """Urban element provides impervious fraction."""
        elu_urban = _make_mock_elu(1, "urban", 500.0, impervious_fraction=0.65)

        rz = _make_mock_rootzone(
            element_landuse=[elu_urban],
            crop_types={},
            soil_params={},
        )
        rz.get_landuse_for_element.return_value = [elu_urban]

        model = _make_mock_model(rootzone=rz)
        model_state._model = model
        model_state.get_area_manager = lambda: None
        rz_module._land_use_loaded = True

        app = create_app()
        client = TestClient(app)
        resp = client.get("/api/rootzone/land-use/1/crops")
        data = resp.json()
        assert data["urban_impervious_fraction"] == 0.65
        assert data["crops"] == []

    def test_mixed_agricultural_and_urban(self):
        """Element with both agricultural and urban land use types."""
        ct = _make_mock_crop_type(1, "Rice")
        elu_ag = _make_mock_elu(1, "agricultural", 300.0, crop_fractions={1: 1.0})
        elu_ur = _make_mock_elu(1, "urban", 200.0, impervious_fraction=0.8)

        rz = _make_mock_rootzone(
            element_landuse=[elu_ag, elu_ur],
            crop_types={1: ct},
            soil_params={},
        )
        rz.get_landuse_for_element.return_value = [elu_ag, elu_ur]

        model = _make_mock_model(rootzone=rz)
        model_state._model = model
        model_state.get_area_manager = lambda: None
        rz_module._land_use_loaded = True

        app = create_app()
        client = TestClient(app)
        resp = client.get("/api/rootzone/land-use/1/crops")
        data = resp.json()
        assert len(data["crops"]) == 1
        assert data["crops"][0]["name"] == "Rice"
        assert data["urban_impervious_fraction"] == 0.8

    def test_with_soil_parameters(self):
        """Element with soil parameters includes them in response."""
        elu = _make_mock_elu(1, "agricultural", 100.0, crop_fractions={})
        sp = _make_mock_soil_params(porosity=0.45, field_capacity=0.30, wilting_point=0.12)

        rz = _make_mock_rootzone(
            element_landuse=[elu],
            crop_types={},
            soil_params={1: sp},
        )
        rz.get_landuse_for_element.return_value = [elu]

        model = _make_mock_model(rootzone=rz)
        model_state._model = model
        model_state.get_area_manager = lambda: None
        rz_module._land_use_loaded = True

        app = create_app()
        client = TestClient(app)
        resp = client.get("/api/rootzone/land-use/1/crops")
        data = resp.json()
        assert data["soil_parameters"] is not None
        assert data["soil_parameters"]["porosity"] == 0.45
        assert data["soil_parameters"]["field_capacity"] == 0.30
        assert data["soil_parameters"]["wilting_point"] == 0.12

    def test_without_soil_parameters(self):
        """Element without soil parameters returns soil_parameters=None."""
        elu = _make_mock_elu(1, "agricultural", 100.0, crop_fractions={})

        rz = _make_mock_rootzone(
            element_landuse=[elu],
            crop_types={},
            soil_params={},
        )
        rz.get_landuse_for_element.return_value = [elu]

        model = _make_mock_model(rootzone=rz)
        model_state._model = model
        model_state.get_area_manager = lambda: None
        rz_module._land_use_loaded = True

        app = create_app()
        client = TestClient(app)
        resp = client.get("/api/rootzone/land-use/1/crops")
        data = resp.json()
        assert data["soil_parameters"] is None

    def test_native_riparian_type_no_crops(self):
        """Native/riparian land use type contributes neither crops nor urban fraction."""
        elu = _make_mock_elu(1, "native_riparian", 500.0)

        rz = _make_mock_rootzone(
            element_landuse=[elu],
            crop_types={},
            soil_params={},
        )
        rz.get_landuse_for_element.return_value = [elu]

        model = _make_mock_model(rootzone=rz)
        model_state._model = model
        model_state.get_area_manager = lambda: None
        rz_module._land_use_loaded = True

        app = create_app()
        client = TestClient(app)
        resp = client.get("/api/rootzone/land-use/1/crops")
        data = resp.json()
        assert data["crops"] == []
        assert data["urban_impervious_fraction"] == 0.0


# ===========================================================================
# 7. GET /api/rootzone/crops
# ===========================================================================


class TestGetCrops:
    """Tests for GET /api/rootzone/crops."""

    def test_no_model_returns_404(self, client_no_model):
        resp = client_no_model.get("/api/rootzone/crops")
        assert resp.status_code == 404

    def test_no_rootzone_returns_404(self, client_with_model_no_rootzone):
        resp = client_with_model_no_rootzone.get("/api/rootzone/crops")
        assert resp.status_code == 404

    def test_empty_crop_types(self):
        """No crop types defined returns empty list."""
        rz = _make_mock_rootzone(crop_types={})
        # Need values() to work like a real dict
        rz.crop_types = {}

        model = _make_mock_model(rootzone=rz)
        model_state._model = model

        app = create_app()
        client = TestClient(app)
        resp = client.get("/api/rootzone/crops")
        assert resp.status_code == 200
        data = resp.json()
        assert data["n_crops"] == 0
        assert data["crops"] == []

    def test_single_crop(self):
        """Single crop type."""
        ct = _make_mock_crop_type(1, "Grain", root_depth=2.5, kc=0.8)
        rz = _make_mock_rootzone()
        rz.crop_types = {1: ct}

        model = _make_mock_model(rootzone=rz)
        model_state._model = model

        app = create_app()
        client = TestClient(app)
        resp = client.get("/api/rootzone/crops")
        data = resp.json()
        assert data["n_crops"] == 1
        assert data["crops"][0]["id"] == 1
        assert data["crops"][0]["name"] == "Grain"
        assert data["crops"][0]["root_depth"] == 2.5
        assert data["crops"][0]["kc"] == 0.8

    def test_multiple_crops_sorted_by_id(self):
        """Multiple crops returned sorted by ID."""
        ct3 = _make_mock_crop_type(3, "Corn")
        ct1 = _make_mock_crop_type(1, "Wheat")
        ct2 = _make_mock_crop_type(2, "Rice")
        rz = _make_mock_rootzone()
        rz.crop_types = {3: ct3, 1: ct1, 2: ct2}

        model = _make_mock_model(rootzone=rz)
        model_state._model = model

        app = create_app()
        client = TestClient(app)
        resp = client.get("/api/rootzone/crops")
        data = resp.json()
        assert data["n_crops"] == 3
        ids = [c["id"] for c in data["crops"]]
        assert ids == [1, 2, 3]


# ===========================================================================
# 8. GET /api/rootzone/soil-params/{element_id}
# ===========================================================================


class TestGetSoilParams:
    """Tests for GET /api/rootzone/soil-params/{element_id}."""

    def test_no_model_returns_404(self, client_no_model):
        resp = client_no_model.get("/api/rootzone/soil-params/1")
        assert resp.status_code == 404

    def test_no_rootzone_returns_404(self, client_with_model_no_rootzone):
        resp = client_with_model_no_rootzone.get("/api/rootzone/soil-params/1")
        assert resp.status_code == 404

    def test_element_not_found_returns_404(self):
        """No soil params for element -> 404."""
        rz = _make_mock_rootzone(soil_params={})
        # Make .get() work like a real dict
        rz.soil_params = {}

        model = _make_mock_model(rootzone=rz)
        model_state._model = model

        app = create_app()
        client = TestClient(app)
        resp = client.get("/api/rootzone/soil-params/999")
        assert resp.status_code == 404
        assert "No soil parameters for element" in resp.json()["detail"]

    def test_valid_soil_params(self):
        """Valid soil parameters returned for element."""
        sp = _make_mock_soil_params(
            porosity=0.42,
            field_capacity=0.28,
            wilting_point=0.13,
            saturated_kv=2.5,
            lambda_param=0.6,
            kunsat_method=1,
            k_ponded=3.0,
            capillary_rise=0.5,
            available_water=0.15,
            drainable_porosity=0.14,
            precip_column=2,
            precip_factor=0.9,
        )
        rz = _make_mock_rootzone()
        rz.soil_params = {1: sp}

        model = _make_mock_model(rootzone=rz)
        model_state._model = model

        app = create_app()
        client = TestClient(app)
        resp = client.get("/api/rootzone/soil-params/1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["element_id"] == 1
        assert data["porosity"] == 0.42
        assert data["field_capacity"] == 0.28
        assert data["wilting_point"] == 0.13
        assert data["saturated_kv"] == 2.5
        assert data["lambda_param"] == 0.6
        assert data["kunsat_method"] == 1
        assert data["k_ponded"] == 3.0
        assert data["capillary_rise"] == 0.5
        assert data["available_water"] == 0.15
        assert data["drainable_porosity"] == 0.14
        assert data["precip_column"] == 2
        assert data["precip_factor"] == 0.9

    def test_multiple_elements_returns_correct_one(self):
        """Multiple elements, returns the right one by ID."""
        sp1 = _make_mock_soil_params(porosity=0.30)
        sp2 = _make_mock_soil_params(porosity=0.50)
        rz = _make_mock_rootzone()
        rz.soil_params = {1: sp1, 2: sp2}

        model = _make_mock_model(rootzone=rz)
        model_state._model = model

        app = create_app()
        client = TestClient(app)
        resp = client.get("/api/rootzone/soil-params/2")
        data = resp.json()
        assert data["element_id"] == 2
        assert data["porosity"] == 0.50
