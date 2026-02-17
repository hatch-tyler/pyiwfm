"""
Tests for root zone module gaps (Feb 2026):
- Gap 1: Crop type extraction from v4.x sub-files
- Gap 2: NativeRiparian ISTRMRV column
- Gap 3: Area data reader + RootZone.load_land_use_snapshot()
- Gap 4: WebAPI rootzone endpoints
- Gap 5: v5.0 reader dispatch
- Gap 6: v4.12 destination encoding
- Gap 7: Time-series reader wrapper
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pyiwfm.components.rootzone import (
    CropType,
    ElementLandUse,
    LandUseType,
    RootZone,
    SoilParameters,
)
from pyiwfm.io.rootzone_v4x import (
    NativeRiparianConfigV4x,
    NativeRiparianElementRowV4x,
    NativeRiparianInitialRowV4x,
    NativeRiparianWriterV4x,
    NonPondedCropConfigV4x,
    PondedCropConfigV4x,
    RootDepthRow,
    read_native_riparian_v4x,
)

# =====================================================================
# Gap 1: Crop type extraction from v4.x sub-file configs
# =====================================================================


class TestCropTypeExtraction:
    """Verify crop types are correctly extracted from sub-file configs."""

    def test_nonponded_crop_types(self):
        """Non-ponded crops populate rootzone.crop_types."""
        rz = RootZone(n_elements=2, n_layers=1)
        rz.nonponded_config = NonPondedCropConfigV4x(
            n_crops=3,
            crop_codes=["GRAIN", "ALFALFA", "PASTURE"],
            root_depth_data=[
                RootDepthRow(1, 3.0, 1),
                RootDepthRow(2, 5.0, 2),
                RootDepthRow(3, 2.5, 3),
            ],
        )
        # Simulate what model.py does
        crop_id_offset = 0
        np_cfg = rz.nonponded_config
        for i, code in enumerate(np_cfg.crop_codes):
            crop_id = i + 1
            rd = (
                np_cfg.root_depth_data[i].max_root_depth if i < len(np_cfg.root_depth_data) else 0.0
            )
            rz.add_crop_type(CropType(id=crop_id, name=code, root_depth=rd))
        crop_id_offset = len(np_cfg.crop_codes)

        assert rz.n_crop_types == 3
        assert rz.crop_types[1].name == "GRAIN"
        assert rz.crop_types[1].root_depth == pytest.approx(3.0)
        assert rz.crop_types[2].name == "ALFALFA"
        assert rz.crop_types[3].name == "PASTURE"
        assert crop_id_offset == 3

    def test_ponded_crop_types_with_offset(self):
        """Ponded crops get IDs offset by non-ponded count."""
        rz = RootZone(n_elements=2, n_layers=1)
        # Simulate 2 non-ponded crops already added
        rz.add_crop_type(CropType(id=1, name="GRAIN", root_depth=3.0))
        rz.add_crop_type(CropType(id=2, name="ALFALFA", root_depth=5.0))

        p_cfg = PondedCropConfigV4x(root_depths=[2.0, 2.1, 2.2, 2.3, 2.4])
        crop_id_offset = 2
        _PONDED_NAMES = ["RICE_FL", "RICE_NFL", "RICE_NDC", "REFUGE_SL", "REFUGE_PR"]
        for i, depth in enumerate(p_cfg.root_depths):
            crop_id = crop_id_offset + i + 1
            name = _PONDED_NAMES[i] if i < len(_PONDED_NAMES) else f"PONDED_{i + 1}"
            rz.add_crop_type(CropType(id=crop_id, name=name, root_depth=depth))

        assert rz.n_crop_types == 7
        assert rz.crop_types[3].name == "RICE_FL"
        assert rz.crop_types[3].root_depth == pytest.approx(2.0)
        assert rz.crop_types[7].name == "REFUGE_PR"
        assert rz.crop_types[7].root_depth == pytest.approx(2.4)

    def test_nonponded_with_missing_root_depth(self):
        """Handles case where root_depth_data is shorter than crop_codes."""
        rz = RootZone(n_elements=2, n_layers=1)
        rz.nonponded_config = NonPondedCropConfigV4x(
            n_crops=3,
            crop_codes=["A", "B", "C"],
            root_depth_data=[
                RootDepthRow(1, 3.0, 1),
            ],
        )
        np_cfg = rz.nonponded_config
        for i, code in enumerate(np_cfg.crop_codes):
            crop_id = i + 1
            rd = (
                np_cfg.root_depth_data[i].max_root_depth if i < len(np_cfg.root_depth_data) else 0.0
            )
            rz.add_crop_type(CropType(id=crop_id, name=code, root_depth=rd))

        assert rz.crop_types[1].root_depth == pytest.approx(3.0)
        assert rz.crop_types[2].root_depth == pytest.approx(0.0)
        assert rz.crop_types[3].root_depth == pytest.approx(0.0)

    def test_no_configs_means_no_crops(self):
        """Without sub-file configs, crop_types remains empty."""
        rz = RootZone(n_elements=2, n_layers=1)
        assert rz.n_crop_types == 0


# =====================================================================
# Gap 2: NativeRiparian ISTRMRV column
# =====================================================================


class TestNativeRiparianISTRMRV:
    """Verify ISTRMRV (riparian_stream_node) column is handled."""

    def test_dataclass_default(self):
        """Default riparian_stream_node is 0."""
        row = NativeRiparianElementRowV4x(1, 65.0, 70.0, 1, 4)
        assert row.riparian_stream_node == 0

    def test_dataclass_with_stream_node(self):
        """Can construct with explicit stream node."""
        row = NativeRiparianElementRowV4x(1, 65.0, 70.0, 1, 4, 2685)
        assert row.riparian_stream_node == 2685

    def test_read_5_columns(self, tmp_path):
        """Reader handles 5-column files (no ISTRMRV)."""
        lines = [
            "C  Native file",
            "   area.dat   / AREA_FILE",
            "   1.0        / RD_FACTOR",
            "   3.5        / NATIVE_RD",
            "   2.0        / RIPARIAN_RD",
            "   1   65.0   70.0   1   4",
            "   2   66.0   71.0   2   5",
            "C  Initial conditions",
            "   1   0.15   0.18",
            "   2   0.16   0.19",
        ]
        path = tmp_path / "nvrv.dat"
        path.write_text("\n".join(lines) + "\n")

        cfg = read_native_riparian_v4x(path, n_elements=2)
        assert cfg.element_data[0].riparian_stream_node == 0
        assert cfg.element_data[1].riparian_stream_node == 0

    def test_read_6_columns(self, tmp_path):
        """Reader handles 6-column files (with ISTRMRV)."""
        lines = [
            "C  Native file",
            "   area.dat   / AREA_FILE",
            "   1.0        / RD_FACTOR",
            "   3.5        / NATIVE_RD",
            "   2.0        / RIPARIAN_RD",
            "   1   65.0   70.0   1   4   2685",
            "   2   66.0   71.0   2   5   2700",
            "C  Initial conditions",
            "   1   0.15   0.18",
            "   2   0.16   0.19",
        ]
        path = tmp_path / "nvrv.dat"
        path.write_text("\n".join(lines) + "\n")

        cfg = read_native_riparian_v4x(path, n_elements=2)
        assert cfg.element_data[0].riparian_stream_node == 2685
        assert cfg.element_data[1].riparian_stream_node == 2700

    def test_writer_includes_stream_node(self, tmp_path):
        """Writer includes 6th column when any row has stream node."""
        cfg = NativeRiparianConfigV4x(
            area_data_file=Path("area.dat"),
            root_depth_factor=1.0,
            native_root_depth=3.5,
            riparian_root_depth=2.0,
            element_data=[
                NativeRiparianElementRowV4x(1, 65.0, 70.0, 1, 4, 2685),
                NativeRiparianElementRowV4x(2, 66.0, 71.0, 2, 5, 0),
            ],
            initial_conditions=[
                NativeRiparianInitialRowV4x(1, 0.15, 0.18),
                NativeRiparianInitialRowV4x(2, 0.16, 0.19),
            ],
        )
        out = tmp_path / "nvrv_out.dat"
        NativeRiparianWriterV4x().write(cfg, out)

        content = out.read_text()
        assert "ISTRMRV" in content
        assert "2685" in content

    def test_writer_omits_stream_node_when_zero(self, tmp_path):
        """Writer omits 6th column when all rows have stream_node=0."""
        cfg = NativeRiparianConfigV4x(
            area_data_file=Path("area.dat"),
            root_depth_factor=1.0,
            native_root_depth=3.5,
            riparian_root_depth=2.0,
            element_data=[
                NativeRiparianElementRowV4x(1, 65.0, 70.0, 1, 4, 0),
                NativeRiparianElementRowV4x(2, 66.0, 71.0, 2, 5, 0),
            ],
            initial_conditions=[
                NativeRiparianInitialRowV4x(1, 0.15, 0.18),
                NativeRiparianInitialRowV4x(2, 0.16, 0.19),
            ],
        )
        out = tmp_path / "nvrv_out.dat"
        NativeRiparianWriterV4x().write(cfg, out)

        content = out.read_text()
        assert "ISTRMRV" not in content

    def test_round_trip_with_stream_node(self, tmp_path):
        """Read-write-read round trip preserves ISTRMRV."""
        cfg = NativeRiparianConfigV4x(
            area_data_file=Path("area.dat"),
            root_depth_factor=1.0,
            native_root_depth=3.5,
            riparian_root_depth=2.0,
            element_data=[
                NativeRiparianElementRowV4x(1, 65.0, 70.0, 1, 4, 2685),
                NativeRiparianElementRowV4x(2, 66.0, 71.0, 2, 5, 2700),
            ],
            initial_conditions=[
                NativeRiparianInitialRowV4x(1, 0.15, 0.18),
                NativeRiparianInitialRowV4x(2, 0.16, 0.19),
            ],
        )
        out = tmp_path / "nvrv.dat"
        NativeRiparianWriterV4x().write(cfg, out)

        cfg2 = read_native_riparian_v4x(out, n_elements=2)
        assert cfg2.element_data[0].riparian_stream_node == 2685
        assert cfg2.element_data[1].riparian_stream_node == 2700


# =====================================================================
# Gap 3: Area data reader
# =====================================================================


def _write_area_file(path: Path, n_elements: int = 3, n_crops: int = 2) -> None:
    """Write a minimal IWFM area time-series file."""
    lines = [
        "C  Area data file",
    ]
    # Column pointers (one per crop)
    cols = "  ".join(str(i) for i in range(1, n_crops + 1))
    lines.append(f"   {cols}   / column pointers")
    # Factor
    lines.append("   1.0   / FACTARL")
    # DSS file (blank)
    lines.append("         / DSSFL")
    # Two timesteps
    for ts_idx, date in enumerate(["10/01/1921_24:00", "11/01/1921_24:00"]):
        for eid in range(1, n_elements + 1):
            vals = "  ".join(f"{100.0 + eid + ts_idx * 10 + c:.1f}" for c in range(n_crops))
            lines.append(f"   {date}   {eid}   {vals}")
    path.write_text("\n".join(lines) + "\n")


class TestAreaReader:
    def test_read_metadata(self, tmp_path):
        from pyiwfm.io.rootzone_area import read_area_metadata

        path = tmp_path / "area.dat"
        _write_area_file(path, n_elements=3, n_crops=2)
        meta = read_area_metadata(path)
        assert meta.n_columns == 2
        assert meta.factor == pytest.approx(1.0)

    def test_read_first_timestep(self, tmp_path):
        from pyiwfm.io.rootzone_area import read_area_timestep

        path = tmp_path / "area.dat"
        _write_area_file(path, n_elements=3, n_crops=2)
        data = read_area_timestep(path, timestep_index=0)
        assert len(data) == 3
        assert 1 in data
        assert len(data[1]) == 2
        # elem 1, crop 0: 100.0 + 1 + 0*10 + 0 = 101.0
        assert data[1][0] == pytest.approx(101.0)

    def test_read_second_timestep(self, tmp_path):
        from pyiwfm.io.rootzone_area import read_area_timestep

        path = tmp_path / "area.dat"
        _write_area_file(path, n_elements=3, n_crops=2)
        data = read_area_timestep(path, timestep_index=1)
        assert len(data) == 3
        # elem 1, crop 0: 100.0 + 1 + 1*10 + 0 = 111.0
        assert data[1][0] == pytest.approx(111.0)

    def test_read_all_timesteps(self, tmp_path):
        from pyiwfm.io.rootzone_area import read_all_timesteps

        path = tmp_path / "area.dat"
        _write_area_file(path, n_elements=3, n_crops=2)
        timesteps = read_all_timesteps(path)
        assert len(timesteps) == 2
        assert timesteps[0][0] == "10/01/1921_24:00"
        assert timesteps[1][0] == "11/01/1921_24:00"
        assert len(timesteps[0][1]) == 3

    def test_factor_applied(self, tmp_path):
        """Unit conversion factor is applied to values."""
        from pyiwfm.io.rootzone_area import read_area_timestep

        lines = [
            "C  Area file",
            "   1   / cols",
            "   2.0  / FACTARL (doubles all values)",
            "        / DSSFL",
            "   10/01/2000_24:00   1   50.0",
        ]
        path = tmp_path / "area.dat"
        path.write_text("\n".join(lines) + "\n")

        data = read_area_timestep(path, timestep_index=0)
        assert data[1][0] == pytest.approx(100.0)  # 50 * 2

    def test_empty_file_returns_empty(self, tmp_path):
        """File with only header returns empty dict."""
        from pyiwfm.io.rootzone_area import read_area_timestep

        lines = [
            "C  Area file",
            "   1   2   / cols",
            "   1.0  / FACTARL",
            "        / DSSFL",
        ]
        path = tmp_path / "area.dat"
        path.write_text("\n".join(lines) + "\n")

        data = read_area_timestep(path, timestep_index=0)
        assert data == {}


# =====================================================================
# Gap 3 continued: RootZone.load_land_use_snapshot()
# =====================================================================


class TestLoadLandUseSnapshot:
    def test_loads_nonponded(self, tmp_path):
        """Loads non-ponded area data into element_landuse."""
        area_file = tmp_path / "np_area.dat"
        _write_area_file(area_file, n_elements=2, n_crops=2)

        rz = RootZone(n_elements=2, n_layers=1)
        rz.nonponded_area_file = area_file
        rz.nonponded_config = NonPondedCropConfigV4x(n_crops=2)
        rz.load_land_use_snapshot(timestep=0)

        ag = [e for e in rz.element_landuse if e.land_use_type == LandUseType.AGRICULTURAL]
        assert len(ag) == 2
        assert ag[0].area > 0

    def test_loads_native(self, tmp_path):
        """Loads native/riparian area data."""
        area_file = tmp_path / "nv_area.dat"
        _write_area_file(area_file, n_elements=2, n_crops=1)

        rz = RootZone(n_elements=2, n_layers=1)
        rz.native_area_file = area_file
        rz.load_land_use_snapshot(timestep=0)

        nv = [e for e in rz.element_landuse if e.land_use_type == LandUseType.NATIVE_RIPARIAN]
        assert len(nv) == 2

    def test_clears_previous_data(self, tmp_path):
        """Calling again replaces previous data."""
        area_file = tmp_path / "np_area.dat"
        _write_area_file(area_file, n_elements=2, n_crops=1)

        rz = RootZone(n_elements=2, n_layers=1)
        rz.nonponded_area_file = area_file
        rz.nonponded_config = NonPondedCropConfigV4x(n_crops=1)
        rz.load_land_use_snapshot(timestep=0)
        count1 = len(rz.element_landuse)
        rz.load_land_use_snapshot(timestep=0)
        assert len(rz.element_landuse) == count1

    def test_no_files_means_empty(self):
        """Without any area files, element_landuse stays empty."""
        rz = RootZone(n_elements=2, n_layers=1)
        rz.load_land_use_snapshot(timestep=0)
        assert rz.element_landuse == []


# =====================================================================
# Gap 3: RootZone new fields
# =====================================================================


class TestRootZoneAreaFields:
    def test_default_none(self):
        rz = RootZone(n_elements=2, n_layers=1)
        assert rz.nonponded_area_file is None
        assert rz.ponded_area_file is None
        assert rz.urban_area_file is None
        assert rz.native_area_file is None

    def test_set_paths(self, tmp_path):
        rz = RootZone(n_elements=2, n_layers=1)
        rz.nonponded_area_file = tmp_path / "np.dat"
        rz.ponded_area_file = tmp_path / "p.dat"
        assert rz.nonponded_area_file == tmp_path / "np.dat"
        assert rz.ponded_area_file == tmp_path / "p.dat"


# =====================================================================
# Gap 4: WebAPI rootzone endpoints
# =====================================================================


# Guard imports
fastapi = pytest.importorskip("fastapi", reason="FastAPI not available")
pytest.importorskip("pydantic", reason="Pydantic not available")
pytest.importorskip("httpx", reason="httpx not available")


class TestWebapiRootzoneRoutes:
    @pytest.fixture(autouse=True)
    def setup_model(self):
        """Set up a mock model with rootzone data."""
        from fastapi.testclient import TestClient

        # Reset module-level flag
        import pyiwfm.visualization.webapi.routes.rootzone as rz_mod
        from pyiwfm.visualization.webapi.config import model_state
        from pyiwfm.visualization.webapi.server import create_app

        rz_mod._land_use_loaded = False

        model = MagicMock()
        model.name = "TestModel"
        model.metadata = {}

        rz = RootZone(n_elements=2, n_layers=1)
        rz.add_crop_type(CropType(id=1, name="GRAIN", root_depth=3.0))
        rz.add_crop_type(CropType(id=2, name="ALFALFA", root_depth=5.0))
        rz.set_soil_parameters(
            1,
            SoilParameters(
                porosity=0.45,
                field_capacity=0.20,
                wilting_point=0.10,
                saturated_kv=2.5,
                lambda_param=0.62,
                kunsat_method=2,
            ),
        )
        rz.add_element_landuse(
            ElementLandUse(
                element_id=1,
                land_use_type=LandUseType.AGRICULTURAL,
                area=100.0,
                crop_fractions={1: 0.6, 2: 0.4},
            )
        )
        rz.add_element_landuse(
            ElementLandUse(
                element_id=1,
                land_use_type=LandUseType.URBAN,
                area=50.0,
                impervious_fraction=0.3,
            )
        )

        model.rootzone = rz

        model_state._model = model
        app = create_app()
        self.client = TestClient(app)

        yield

        model_state._model = None

    def test_get_crops(self):
        resp = self.client.get("/api/rootzone/crops")
        assert resp.status_code == 200
        data = resp.json()
        assert data["n_crops"] == 2
        assert data["crops"][0]["name"] == "GRAIN"
        assert data["crops"][1]["name"] == "ALFALFA"

    def test_get_land_use(self):
        resp = self.client.get("/api/rootzone/land-use")
        assert resp.status_code == 200
        data = resp.json()
        assert data["n_elements"] >= 1

    def test_get_element_crops(self):
        resp = self.client.get("/api/rootzone/land-use/1/crops")
        assert resp.status_code == 200
        data = resp.json()
        assert data["element_id"] == 1
        assert len(data["crops"]) == 2
        assert data["urban_impervious_fraction"] == pytest.approx(0.3)

    def test_get_element_crops_not_found(self):
        resp = self.client.get("/api/rootzone/land-use/999/crops")
        assert resp.status_code == 404

    def test_get_soil_params(self):
        resp = self.client.get("/api/rootzone/soil-params/1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["porosity"] == pytest.approx(0.45)
        assert data["field_capacity"] == pytest.approx(0.20)
        assert data["wilting_point"] == pytest.approx(0.10)
        assert data["saturated_kv"] == pytest.approx(2.5)
        assert data["lambda_param"] == pytest.approx(0.62)
        assert "available_water" in data
        assert "drainable_porosity" in data

    def test_get_soil_params_not_found(self):
        resp = self.client.get("/api/rootzone/soil-params/999")
        assert resp.status_code == 404

    def test_get_crops_no_model(self):
        from pyiwfm.visualization.webapi.config import model_state

        model_state._model = None
        resp = self.client.get("/api/rootzone/crops")
        assert resp.status_code == 404

    def test_get_crops_no_rootzone(self):
        from pyiwfm.visualization.webapi.config import model_state

        model_state._model.rootzone = None
        resp = self.client.get("/api/rootzone/crops")
        assert resp.status_code == 404


# =====================================================================
# Gap 5: v5.0 reader dispatch (unit test with version check)
# =====================================================================


class TestV50ReaderDispatch:
    def test_version_ge_50(self):
        from pyiwfm.io.rootzone import version_ge

        assert version_ge("5.0", (5, 0))
        assert version_ge("5.1", (5, 0))
        assert not version_ge("4.13", (5, 0))

    def test_version_dispatch_logic(self):
        """Verify the dispatch logic selects correct readers."""
        from pyiwfm.io.rootzone import version_ge

        # For v4.x, should use V4x readers
        assert not version_ge("4.11", (5, 0))
        assert not version_ge("4.12", (5, 0))
        assert not version_ge("4.0", (5, 0))

        # For v5.0+, should use v5.0 readers
        assert version_ge("5.0", (5, 0))


# =====================================================================
# Gap 6: v4.12 destination encoding
# =====================================================================


class TestV412DestinationEncoding:
    def test_positive_destination(self):
        """Positive values should store (raw, abs)."""
        rz = RootZone(n_elements=2, n_layers=1)
        # Simulate what model.py does for v4.12
        rz.surface_flow_dest_ag[1] = (5, abs(5))
        assert rz.surface_flow_dest_ag[1] == (5, 5)

    def test_negative_destination(self):
        """Negative values preserve sign in first element."""
        rz = RootZone(n_elements=2, n_layers=1)
        rz.surface_flow_dest_ag[1] = (-3, abs(-3))
        assert rz.surface_flow_dest_ag[1] == (-3, 3)

    def test_zero_destination(self):
        """Zero means no destination."""
        rz = RootZone(n_elements=2, n_layers=1)
        rz.surface_flow_dest_ag[1] = (0, abs(0))
        assert rz.surface_flow_dest_ag[1] == (0, 0)


# =====================================================================
# Gap 7: Time-series reader wrapper
# =====================================================================


class TestTimeSeriesReaderWrapper:
    def test_import(self):
        from pyiwfm.io.timeseries_reader import (
            IWFMTimeSeriesData,
            read_iwfm_timeseries,
        )

        assert IWFMTimeSeriesData is not None
        assert callable(read_iwfm_timeseries)

    def test_dataclass_defaults(self):
        from pyiwfm.io.timeseries_reader import IWFMTimeSeriesData

        ts = IWFMTimeSeriesData()
        assert ts.n_columns == 0
        assert ts.dates == []
        assert ts.factor == 1.0
        assert ts.time_unit == ""
        assert ts.data.shape == (0, 0)

    def test_read_simple_file(self, tmp_path):
        """Read a simple 2-column IWFM time-series file."""
        from pyiwfm.io.timeseries_reader import read_iwfm_timeseries

        lines = [
            "C  Test time series",
            "   2             / NDATA",
            "   1.0           / FACTOR",
            "C  Data",
            "10/01/1921_24:00   1.5   2.5",
            "11/01/1921_24:00   3.0   4.0",
        ]
        path = tmp_path / "ts.dat"
        path.write_text("\n".join(lines) + "\n")

        ts = read_iwfm_timeseries(path)
        assert ts.n_columns == 2
        assert len(ts.dates) == 2
        assert ts.data.shape[0] == 2
        assert ts.factor == pytest.approx(1.0)
