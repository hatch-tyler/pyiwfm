"""
Tests for v4.x root zone I/O: main file reader (all versions),
sub-file readers/writers, round-trips, and model loading integration.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from pyiwfm.io.rootzone import (
    RootZoneMainFileReader,
    RootZoneMainFileConfig,
    ElementSoilParamRow,
    parse_version as _parse_version,
    version_ge as _version_ge,
)
from pyiwfm.io.rootzone_v4x import (
    # Data classes
    NonPondedCropConfigV4x,
    PondedCropConfigV4x,
    UrbanConfigV4x,
    NativeRiparianConfigV4x,
    RootDepthRow,
    ElementCropRow,
    AgInitialConditionRow,
    UrbanElementRowV4x,
    UrbanInitialRowV4x,
    NativeRiparianElementRowV4x,
    NativeRiparianInitialRowV4x,
    # Readers
    NonPondedCropReaderV4x,
    PondedCropReaderV4x,
    UrbanReaderV4x,
    NativeRiparianReaderV4x,
    # Writers
    NonPondedCropWriterV4x,
    PondedCropWriterV4x,
    UrbanWriterV4x,
    NativeRiparianWriterV4x,
    # Convenience functions
    read_nonponded_v4x,
    read_ponded_v4x,
    read_urban_v4x,
    read_native_riparian_v4x,
)
from pyiwfm.components.rootzone import SoilParameters, RootZone


# =====================================================================
# Version utility tests
# =====================================================================


class TestVersionUtilities:
    def test_parse_version_40(self):
        assert _parse_version("4.0") == (4, 0)

    def test_parse_version_412(self):
        assert _parse_version("4.12") == (4, 12)

    def test_parse_version_50(self):
        assert _parse_version("5.0") == (5, 0)

    def test_parse_version_invalid(self):
        assert _parse_version("abc") == (0, 0)

    def test_version_ge_true(self):
        assert _version_ge("4.12", (4, 12))
        assert _version_ge("4.13", (4, 12))
        assert _version_ge("5.0", (4, 12))

    def test_version_ge_false(self):
        assert not _version_ge("4.0", (4, 1))
        assert not _version_ge("4.11", (4, 12))


# =====================================================================
# SoilParameters expansion tests
# =====================================================================


class TestSoilParametersExpansion:
    def test_backwards_compatible_defaults(self):
        """New fields have defaults, so existing code still works."""
        sp = SoilParameters(
            porosity=0.45,
            field_capacity=0.20,
            wilting_point=0.10,
            saturated_kv=2.5,
        )
        assert sp.lambda_param == 0.5
        assert sp.kunsat_method == 2
        assert sp.k_ponded == -1.0
        assert sp.capillary_rise == 0.0
        assert sp.precip_column == 1
        assert sp.precip_factor == 1.0
        assert sp.generic_moisture_column == 0

    def test_full_construction(self):
        sp = SoilParameters(
            porosity=0.45,
            field_capacity=0.20,
            wilting_point=0.10,
            saturated_kv=2.5,
            lambda_param=0.62,
            kunsat_method=1,
            k_ponded=3.0,
            capillary_rise=1.5,
            precip_column=3,
            precip_factor=0.9,
            generic_moisture_column=2,
        )
        assert sp.lambda_param == 0.62
        assert sp.kunsat_method == 1
        assert sp.k_ponded == 3.0
        assert sp.capillary_rise == 1.5


# =====================================================================
# RootZone expansion tests
# =====================================================================


class TestRootZoneExpansion:
    def test_new_fields_have_defaults(self):
        rz = RootZone(n_elements=10, n_layers=1)
        assert rz.nonponded_config is None
        assert rz.ponded_config is None
        assert rz.urban_config is None
        assert rz.native_riparian_config is None
        assert rz.surface_flow_destinations == {}
        assert rz.surface_flow_dest_ag == {}
        assert rz.surface_flow_dest_urban_in == {}
        assert rz.surface_flow_dest_urban_out == {}
        assert rz.surface_flow_dest_nvrv == {}


# =====================================================================
# Main file reader tests
# =====================================================================


def _write_main_file_v40(path: Path, n_elements: int = 3) -> None:
    """Write a minimal v4.0 main file."""
    lines = [
        "#4.0",
        "C  test file",
        "   0.001                                       / RZCONV",
        "   150                                         / RZITERMX",
        "   0.0833333                                   / FACTCN",
        "C  No GWUPTK for v4.0",
        "                                               / AGNPFL",
        "                                               / PFL",
        "                                               / URBFL",
        "                                               / NVRVFL",
        "                                               / RFFL",
        "                                               / RUFL",
        "                                               / IPFL",
        "                                               / MSRCFL",
        "                                               / AGWDFL",
        "                                               / LWUBUDFL",
        "                                               / RZBUDFL",
        "C  No zone budgets for v4.0",
        "                                               / FMFL",
        "C  Soil parameters section",
        "   0.03281                                     / FACTK",
        "   1hour                                       / TUNITK",
        "C  IE  WP  FC  TN  Lambda  K  KMethod  iPrecCol  fPrecFac  iGenCol  iDestType  iDest",
    ]
    for i in range(1, n_elements + 1):
        lines.append(
            f"   {i}   0.10  0.20  0.45  0.62  2.60  2  {i}  1.0  0  0  0"
        )
    path.write_text("\n".join(lines) + "\n")


def _write_main_file_v411(path: Path, n_elements: int = 3) -> None:
    """Write a minimal v4.11 main file (v4.11+ has GWUPTK and zone budgets)."""
    lines = [
        "#4.11",
        "C  test file",
        "   0.001                                       / RZCONV",
        "   150                                         / RZITERMX",
        "   0.0833333                                   / FACTCN",
        "   0                                           / GWUPTK",
        "                                               / AGNPFL",
        "                                               / PFL",
        "                                               / URBFL",
        "                                               / NVRVFL",
        "                                               / RFFL",
        "                                               / RUFL",
        "                                               / IPFL",
        "                                               / MSRCFL",
        "                                               / AGWDFL",
        "                                               / LWUBUDFL",
        "                                               / RZBUDFL",
        "                                               / ZLWUBUDFL",
        "                                               / ZRZBUDFL",
        "                                               / ARSCLFL",
        "                                               / FMFL",
        "C  Soil parameters section",
        "   0.03281                                     / FACTK",
        "   1.0                                         / FACTEXDTH",
        "   1hour                                       / TUNITK",
        "C  IE WP FC TN Lambda K KMethod CapRise iPrecCol fPrecFac iGenCol iDestType iDest",
    ]
    for i in range(1, n_elements + 1):
        lines.append(
            f"   {i}   0.10  0.20  0.45  0.62  2.60  2  0.0  {i}  1.0  0  0  0"
        )
    path.write_text("\n".join(lines) + "\n")


def _write_main_file_v412(path: Path, n_elements: int = 3) -> None:
    """Write a minimal v4.12 main file."""
    lines = [
        "#4.12",
        "C  test file",
        "   0.001                                       / RZCONV",
        "   150                                         / RZITERMX",
        "   0.0833333                                   / FACTCN",
        "   0                                           / GWUPTK",
        "                                               / AGNPFL",
        "                                               / PFL",
        "                                               / URBFL",
        "                                               / NVRVFL",
        "                                               / RFFL",
        "                                               / RUFL",
        "                                               / IPFL",
        "                                               / MSRCFL",
        "                                               / AGWDFL",
        "                                               / LWUBUDFL",
        "                                               / RZBUDFL",
        "                                               / ZLWUBUDFL",
        "                                               / ZRZBUDFL",
        "                                               / ARSCLFL",
        "                                               / FMFL",
        "C  Soil parameters section",
        "   0.03281                                     / FACTK",
        "   1.0                                         / FACTEXDTH",
        "   1hour                                       / TUNITK",
        "                                               / DESTFL",
        "C  IE WP FC TN Lambda K KPonded KMethod CapRise iPrecCol fPrecFac iGenCol iDestAg iDestUrbIn iDestUrbOut iDestNVRV",
    ]
    for i in range(1, n_elements + 1):
        lines.append(
            f"   {i}   0.10  0.20  0.45  0.62  2.60  -1.0  2  0.0  {i}  1.0  0  1  1  1  1"
        )
    path.write_text("\n".join(lines) + "\n")


class TestMainFileReaderV40:
    def test_reads_version(self, tmp_path):
        path = tmp_path / "rz.dat"
        _write_main_file_v40(path)
        reader = RootZoneMainFileReader()
        cfg = reader.read(path)
        assert cfg.version == "4.0"

    def test_reads_solver_params(self, tmp_path):
        path = tmp_path / "rz.dat"
        _write_main_file_v40(path)
        cfg = RootZoneMainFileReader().read(path)
        assert cfg.convergence_tolerance == pytest.approx(0.001)
        assert cfg.max_iterations == 150

    def test_no_gw_uptake(self, tmp_path):
        path = tmp_path / "rz.dat"
        _write_main_file_v40(path)
        cfg = RootZoneMainFileReader().read(path)
        assert cfg.gw_uptake_enabled is False

    def test_reads_soil_params(self, tmp_path):
        path = tmp_path / "rz.dat"
        _write_main_file_v40(path, n_elements=3)
        cfg = RootZoneMainFileReader().read(path)
        assert len(cfg.element_soil_params) == 3
        row = cfg.element_soil_params[0]
        assert row.element_id == 1
        assert row.wilting_point == pytest.approx(0.10)
        assert row.field_capacity == pytest.approx(0.20)
        assert row.total_porosity == pytest.approx(0.45)
        assert row.lambda_param == pytest.approx(0.62)
        assert row.hydraulic_conductivity == pytest.approx(2.60)
        assert row.kunsat_method == 2
        assert row.capillary_rise == 0.0  # not in v4.0

    def test_k_factor_applied(self, tmp_path):
        path = tmp_path / "rz.dat"
        _write_main_file_v40(path)
        cfg = RootZoneMainFileReader().read(path)
        assert cfg.k_factor == pytest.approx(0.03281)


class TestMainFileReaderV411:
    def test_reads_version(self, tmp_path):
        path = tmp_path / "rz.dat"
        _write_main_file_v411(path)
        cfg = RootZoneMainFileReader().read(path)
        assert cfg.version == "4.11"

    def test_capillary_rise_present(self, tmp_path):
        path = tmp_path / "rz.dat"
        _write_main_file_v411(path)
        cfg = RootZoneMainFileReader().read(path)
        assert len(cfg.element_soil_params) == 3
        row = cfg.element_soil_params[0]
        assert row.capillary_rise == pytest.approx(0.0)

    def test_exdth_factor(self, tmp_path):
        path = tmp_path / "rz.dat"
        _write_main_file_v411(path)
        cfg = RootZoneMainFileReader().read(path)
        assert cfg.k_exdth_factor == pytest.approx(1.0)


class TestMainFileReaderV412:
    def test_reads_version(self, tmp_path):
        path = tmp_path / "rz.dat"
        _write_main_file_v412(path)
        cfg = RootZoneMainFileReader().read(path)
        assert cfg.version == "4.12"

    def test_per_landuse_destinations(self, tmp_path):
        path = tmp_path / "rz.dat"
        _write_main_file_v412(path)
        cfg = RootZoneMainFileReader().read(path)
        assert len(cfg.element_soil_params) == 3
        row = cfg.element_soil_params[0]
        assert row.dest_ag == 1
        assert row.dest_urban_in == 1
        assert row.dest_urban_out == 1
        assert row.dest_nvrv == 1

    def test_16_columns(self, tmp_path):
        path = tmp_path / "rz.dat"
        _write_main_file_v412(path)
        cfg = RootZoneMainFileReader().read(path)
        row = cfg.element_soil_params[0]
        assert row.k_ponded == pytest.approx(-1.0)
        assert row.kunsat_method == 2


# =====================================================================
# Native/riparian sub-file tests
# =====================================================================


def _write_native_riparian_file(path: Path, n_elements: int = 3) -> None:
    lines = [
        "C  Native/riparian sub-file",
        "   area_data.dat                               / AREA_FILE",
        "   1.0                                         / ROOT_DEPTH_FACTOR",
        "   3.5                                         / NATIVE_ROOT_DEPTH",
        "   2.0                                         / RIPARIAN_ROOT_DEPTH",
        "C  Element data: IE NativeCN RiparianCN NativeEtcCol RiparianEtcCol",
    ]
    for i in range(1, n_elements + 1):
        lines.append(f"   {i}   65.0   70.0   {i}   {i + n_elements}")
    lines.append("C  Initial conditions: IE NativeMC RiparianMC")
    for i in range(1, n_elements + 1):
        lines.append(f"   {i}   0.15   0.18")
    path.write_text("\n".join(lines) + "\n")


class TestNativeRiparianReaderV4x:
    def test_read_basic(self, tmp_path):
        path = tmp_path / "nvrv.dat"
        _write_native_riparian_file(path, n_elements=3)
        reader = NativeRiparianReaderV4x(n_elements=3)
        cfg = reader.read(path)
        assert cfg.native_root_depth == pytest.approx(3.5)
        assert cfg.riparian_root_depth == pytest.approx(2.0)
        assert len(cfg.element_data) == 3
        assert len(cfg.initial_conditions) == 3

    def test_element_data_values(self, tmp_path):
        path = tmp_path / "nvrv.dat"
        _write_native_riparian_file(path, n_elements=2)
        reader = NativeRiparianReaderV4x(n_elements=2)
        cfg = reader.read(path)
        row = cfg.element_data[0]
        assert row.element_id == 1
        assert row.native_cn == pytest.approx(65.0)
        assert row.riparian_cn == pytest.approx(70.0)
        assert row.native_etc_column == 1
        assert row.riparian_etc_column == 3

    def test_initial_conditions(self, tmp_path):
        path = tmp_path / "nvrv.dat"
        _write_native_riparian_file(path, n_elements=2)
        cfg = read_native_riparian_v4x(path, n_elements=2)
        ic = cfg.initial_conditions[0]
        assert ic.element_id == 1
        assert ic.native_moisture == pytest.approx(0.15)
        assert ic.riparian_moisture == pytest.approx(0.18)


class TestNativeRiparianWriterV4x:
    def test_round_trip(self, tmp_path):
        cfg = NativeRiparianConfigV4x(
            area_data_file=Path("area.dat"),
            root_depth_factor=1.0,
            native_root_depth=3.5,
            riparian_root_depth=2.0,
            element_data=[
                NativeRiparianElementRowV4x(1, 65.0, 70.0, 1, 4),
                NativeRiparianElementRowV4x(2, 66.0, 71.0, 2, 5),
            ],
            initial_conditions=[
                NativeRiparianInitialRowV4x(1, 0.15, 0.18),
                NativeRiparianInitialRowV4x(2, 0.16, 0.19),
            ],
        )
        out = tmp_path / "nvrv_out.dat"
        NativeRiparianWriterV4x().write(cfg, out)

        reader = NativeRiparianReaderV4x(n_elements=2)
        cfg2 = reader.read(out)
        assert cfg2.native_root_depth == pytest.approx(3.5)
        assert cfg2.riparian_root_depth == pytest.approx(2.0)
        assert len(cfg2.element_data) == 2
        assert cfg2.element_data[0].native_cn == pytest.approx(65.0)
        assert cfg2.initial_conditions[1].riparian_moisture == pytest.approx(0.19)


# =====================================================================
# Urban sub-file tests
# =====================================================================


def _write_urban_file(path: Path, n_elements: int = 3) -> None:
    lines = [
        "C  Urban sub-file",
        "   urban_area.dat                              / AREA_FILE",
        "   1.0                                         / ROOT_DEPTH_FACTOR",
        "   2.5                                         / ROOT_DEPTH",
        "   population.dat                              / POP_FILE",
        "   percap.dat                                  / PERCAP_FILE",
        "   wateruse.dat                                / WUSE_FILE",
        "C  Element data",
    ]
    for i in range(1, n_elements + 1):
        lines.append(
            f"   {i}   0.6   72.0   {i}   {i}   1.0   {i}   {i}   {i}   {i}"
        )
    lines.append("C  Initial conditions")
    for i in range(1, n_elements + 1):
        lines.append(f"   {i}   0.5   0.12")
    path.write_text("\n".join(lines) + "\n")


class TestUrbanReaderV4x:
    def test_read_basic(self, tmp_path):
        path = tmp_path / "urban.dat"
        _write_urban_file(path, n_elements=3)
        cfg = read_urban_v4x(path, n_elements=3)
        assert cfg.root_depth == pytest.approx(2.5)
        assert len(cfg.element_data) == 3
        assert len(cfg.initial_conditions) == 3

    def test_element_data_values(self, tmp_path):
        path = tmp_path / "urban.dat"
        _write_urban_file(path, n_elements=2)
        cfg = read_urban_v4x(path, n_elements=2)
        row = cfg.element_data[0]
        assert row.pervious_fraction == pytest.approx(0.6)
        assert row.curve_number == pytest.approx(72.0)


class TestUrbanWriterV4x:
    def test_round_trip(self, tmp_path):
        cfg = UrbanConfigV4x(
            area_data_file=Path("area.dat"),
            root_depth_factor=1.0,
            root_depth=2.5,
            population_file=Path("pop.dat"),
            per_capita_water_use_file=Path("percap.dat"),
            water_use_specs_file=Path("wuse.dat"),
            element_data=[
                UrbanElementRowV4x(1, 0.6, 72.0, 1, 1, 1.0, 1, 1, 1, 1),
                UrbanElementRowV4x(2, 0.7, 73.0, 2, 2, 1.0, 2, 2, 2, 2),
            ],
            initial_conditions=[
                UrbanInitialRowV4x(1, 0.5, 0.12),
                UrbanInitialRowV4x(2, 0.5, 0.13),
            ],
        )
        out = tmp_path / "urban_out.dat"
        UrbanWriterV4x().write(cfg, out)

        cfg2 = read_urban_v4x(out, n_elements=2)
        assert cfg2.root_depth == pytest.approx(2.5)
        assert len(cfg2.element_data) == 2
        assert cfg2.element_data[1].curve_number == pytest.approx(73.0)
        assert cfg2.initial_conditions[0].moisture_content == pytest.approx(0.12)


# =====================================================================
# Non-ponded crop sub-file tests
# =====================================================================


def _write_nonponded_file(path: Path, n_elements: int = 2, n_crops: int = 2) -> None:
    """Write a minimal non-ponded crop file."""
    lines = [
        "C  Non-ponded crop file",
        f"   {n_crops}                                  / NCROPS",
        "   1                                           / DEMAND_FROM_MOISTURE",
    ]
    # Crop codes
    for i in range(1, n_crops + 1):
        lines.append(f"   CR{i:02d}")
    # Area data file
    lines.append("   crop_areas.dat                              / AREA_FILE")
    # NBudgetCrops
    lines.append("   0                                           / NBUDGETCROPS")
    # Root depth fractions file
    lines.append("   rootfrac.dat                                / ROOT_FRAC_FILE")
    # Root depth factor
    lines.append("   1.0                                         / ROOT_DEPTH_FACTOR")
    # Root depth data
    for i in range(1, n_crops + 1):
        lines.append(f"   {i}   3.0   {i}")
    # Curve numbers table
    lines.append("C  Curve numbers")
    for i in range(1, n_elements + 1):
        vals = " ".join(f"  {70.0 + i:.4f}" for _ in range(n_crops))
        lines.append(f"   {i} {vals}")
    # ETc pointers
    lines.append("C  ETc pointers")
    for i in range(1, n_elements + 1):
        vals = " ".join(f"  {float(i):.4f}" for _ in range(n_crops))
        lines.append(f"   {i} {vals}")
    # Supply req pointers
    lines.append("C  Supply req pointers")
    for i in range(1, n_elements + 1):
        vals = " ".join(f"  {float(i):.4f}" for _ in range(n_crops))
        lines.append(f"   {i} {vals}")
    # Irrigation pointers
    lines.append("C  Irrigation pointers")
    for i in range(1, n_elements + 1):
        vals = " ".join(f"  {float(i):.4f}" for _ in range(n_crops))
        lines.append(f"   {i} {vals}")
    # Min soil moisture file
    lines.append("   minsm.dat                                   / MIN_SM_FILE")
    # Min moisture pointers
    lines.append("C  Min moisture pointers")
    for i in range(1, n_elements + 1):
        vals = " ".join(f"  {0.05:.4f}" for _ in range(n_crops))
        lines.append(f"   {i} {vals}")
    # Target soil moisture file (blank = skip)
    lines.append("                                               / TARGET_SM_FILE")
    # Return flow pointers
    lines.append("C  Return flow pointers")
    for i in range(1, n_elements + 1):
        vals = " ".join(f"  {0.3:.4f}" for _ in range(n_crops))
        lines.append(f"   {i} {vals}")
    # Reuse pointers
    lines.append("C  Reuse pointers")
    for i in range(1, n_elements + 1):
        vals = " ".join(f"  {0.1:.4f}" for _ in range(n_crops))
        lines.append(f"   {i} {vals}")
    # Leaching factors file (blank = skip)
    lines.append("                                               / LEACH_FILE")
    # Initial conditions
    lines.append("C  Initial conditions")
    for i in range(1, n_elements + 1):
        mcs = " ".join(f"  {0.15:.6f}" for _ in range(n_crops))
        lines.append(f"   {i}     0.5000 {mcs}")
    path.write_text("\n".join(lines) + "\n")


class TestNonPondedCropReaderV4x:
    def test_read_basic(self, tmp_path):
        path = tmp_path / "agnp.dat"
        _write_nonponded_file(path, n_elements=2, n_crops=2)
        cfg = read_nonponded_v4x(path, n_elements=2)
        assert cfg.n_crops == 2
        assert cfg.demand_from_moisture_flag == 1
        assert len(cfg.crop_codes) == 2
        assert cfg.crop_codes[0] == "CR01"

    def test_curve_numbers(self, tmp_path):
        path = tmp_path / "agnp.dat"
        _write_nonponded_file(path, n_elements=2, n_crops=2)
        cfg = read_nonponded_v4x(path, n_elements=2)
        assert len(cfg.curve_numbers) == 2
        assert cfg.curve_numbers[0].element_id == 1
        assert len(cfg.curve_numbers[0].values) == 2

    def test_root_depth_data(self, tmp_path):
        path = tmp_path / "agnp.dat"
        _write_nonponded_file(path, n_elements=2, n_crops=2)
        cfg = read_nonponded_v4x(path, n_elements=2)
        assert len(cfg.root_depth_data) == 2
        assert cfg.root_depth_data[0].max_root_depth == pytest.approx(3.0)

    def test_initial_conditions(self, tmp_path):
        path = tmp_path / "agnp.dat"
        _write_nonponded_file(path, n_elements=2, n_crops=2)
        cfg = read_nonponded_v4x(path, n_elements=2)
        assert len(cfg.initial_conditions) == 2
        ic = cfg.initial_conditions[0]
        assert ic.element_id == 1
        assert ic.precip_fraction == pytest.approx(0.5)
        assert len(ic.moisture_contents) == 2

    def test_no_target_sm(self, tmp_path):
        """When target SM file is blank, no target pointers are read."""
        path = tmp_path / "agnp.dat"
        _write_nonponded_file(path, n_elements=2, n_crops=2)
        cfg = read_nonponded_v4x(path, n_elements=2)
        assert cfg.target_soil_moisture_file is None
        assert cfg.target_moisture_pointers == []

    def test_no_leaching(self, tmp_path):
        """When leaching file is blank, no leaching pointers are read."""
        path = tmp_path / "agnp.dat"
        _write_nonponded_file(path, n_elements=2, n_crops=2)
        cfg = read_nonponded_v4x(path, n_elements=2)
        assert cfg.leaching_factors_file is None
        assert cfg.leaching_pointers == []


class TestNonPondedCropWriterV4x:
    def test_round_trip(self, tmp_path):
        cfg = NonPondedCropConfigV4x(
            n_crops=2,
            demand_from_moisture_flag=1,
            crop_codes=["CR01", "CR02"],
            area_data_file=Path("area.dat"),
            n_budget_crops=0,
            root_depth_fractions_file=Path("rootfrac.dat"),
            root_depth_factor=1.0,
            root_depth_data=[
                RootDepthRow(1, 3.0, 1),
                RootDepthRow(2, 3.0, 2),
            ],
            curve_numbers=[
                ElementCropRow(1, [71.0, 72.0]),
                ElementCropRow(2, [73.0, 74.0]),
            ],
            etc_pointers=[
                ElementCropRow(1, [1.0, 2.0]),
                ElementCropRow(2, [1.0, 2.0]),
            ],
            supply_req_pointers=[
                ElementCropRow(1, [1.0, 2.0]),
                ElementCropRow(2, [1.0, 2.0]),
            ],
            irrigation_pointers=[
                ElementCropRow(1, [1.0, 2.0]),
                ElementCropRow(2, [1.0, 2.0]),
            ],
            min_soil_moisture_file=Path("minsm.dat"),
            min_moisture_pointers=[
                ElementCropRow(1, [0.05, 0.05]),
                ElementCropRow(2, [0.05, 0.05]),
            ],
            return_flow_pointers=[
                ElementCropRow(1, [0.3, 0.3]),
                ElementCropRow(2, [0.3, 0.3]),
            ],
            reuse_pointers=[
                ElementCropRow(1, [0.1, 0.1]),
                ElementCropRow(2, [0.1, 0.1]),
            ],
            initial_conditions=[
                AgInitialConditionRow(1, 0.5, [0.15, 0.15]),
                AgInitialConditionRow(2, 0.5, [0.15, 0.15]),
            ],
        )
        out = tmp_path / "agnp_out.dat"
        NonPondedCropWriterV4x().write(cfg, out)

        cfg2 = read_nonponded_v4x(out, n_elements=2)
        assert cfg2.n_crops == 2
        assert cfg2.crop_codes == ["CR01", "CR02"]
        assert len(cfg2.curve_numbers) == 2
        assert cfg2.curve_numbers[0].values[0] == pytest.approx(71.0)
        assert cfg2.initial_conditions[0].precip_fraction == pytest.approx(0.5)


# =====================================================================
# Ponded crop sub-file tests
# =====================================================================


def _write_ponded_file(path: Path, n_elements: int = 2) -> None:
    nc = 5
    lines = [
        "C  Ponded crop file",
        "   ponded_areas.dat                            / AREA_FILE",
        "   0                                           / NBUDGETCROPS",
        "   1.0                                         / ROOT_DEPTH_FACTOR",
    ]
    # 5 root depths
    for _ in range(nc):
        lines.append("   2.0")
    # Curve numbers
    lines.append("C  Curve numbers")
    for i in range(1, n_elements + 1):
        vals = " ".join(f"  {75.0:.4f}" for _ in range(nc))
        lines.append(f"   {i} {vals}")
    # ETc pointers
    lines.append("C  ETc pointers")
    for i in range(1, n_elements + 1):
        vals = " ".join(f"  {float(i):.4f}" for _ in range(nc))
        lines.append(f"   {i} {vals}")
    # Supply req pointers
    lines.append("C  Supply req")
    for i in range(1, n_elements + 1):
        vals = " ".join(f"  {float(i):.4f}" for _ in range(nc))
        lines.append(f"   {i} {vals}")
    # Irrigation pointers
    lines.append("C  Irrigation")
    for i in range(1, n_elements + 1):
        vals = " ".join(f"  {float(i):.4f}" for _ in range(nc))
        lines.append(f"   {i} {vals}")
    # Ponding depth file
    lines.append("   ponding.dat                                 / PONDING_FILE")
    # Operations flow file
    lines.append("   operations.dat                              / OPERATIONS_FILE")
    # Ponding depth pointers
    lines.append("C  Ponding depth pointers")
    for i in range(1, n_elements + 1):
        vals = " ".join(f"  {0.5:.4f}" for _ in range(nc))
        lines.append(f"   {i} {vals}")
    # Decomp water pointers
    lines.append("C  Decomp water pointers")
    for i in range(1, n_elements + 1):
        vals = " ".join(f"  {0.0:.4f}" for _ in range(nc))
        lines.append(f"   {i} {vals}")
    # Return flow pointers
    lines.append("C  Return flow")
    for i in range(1, n_elements + 1):
        vals = " ".join(f"  {0.3:.4f}" for _ in range(nc))
        lines.append(f"   {i} {vals}")
    # Reuse pointers
    lines.append("C  Reuse")
    for i in range(1, n_elements + 1):
        vals = " ".join(f"  {0.1:.4f}" for _ in range(nc))
        lines.append(f"   {i} {vals}")
    # Initial conditions
    lines.append("C  Initial conditions")
    for i in range(1, n_elements + 1):
        mcs = " ".join(f"  {0.15:.6f}" for _ in range(nc))
        lines.append(f"   {i}     0.5000 {mcs}")
    path.write_text("\n".join(lines) + "\n")


class TestPondedCropReaderV4x:
    def test_read_basic(self, tmp_path):
        path = tmp_path / "ponded.dat"
        _write_ponded_file(path, n_elements=2)
        cfg = read_ponded_v4x(path, n_elements=2)
        assert len(cfg.root_depths) == 5
        assert cfg.root_depths[0] == pytest.approx(2.0)
        assert len(cfg.curve_numbers) == 2

    def test_initial_conditions(self, tmp_path):
        path = tmp_path / "ponded.dat"
        _write_ponded_file(path, n_elements=2)
        cfg = read_ponded_v4x(path, n_elements=2)
        assert len(cfg.initial_conditions) == 2
        ic = cfg.initial_conditions[0]
        assert ic.precip_fraction == pytest.approx(0.5)
        assert len(ic.moisture_contents) == 5


class TestPondedCropWriterV4x:
    def test_round_trip(self, tmp_path):
        cfg = PondedCropConfigV4x(
            area_data_file=Path("area.dat"),
            n_budget_crops=0,
            root_depth_factor=1.0,
            root_depths=[2.0, 2.0, 2.0, 2.0, 2.0],
            curve_numbers=[
                ElementCropRow(1, [75.0] * 5),
                ElementCropRow(2, [75.0] * 5),
            ],
            etc_pointers=[
                ElementCropRow(1, [1.0] * 5),
                ElementCropRow(2, [2.0] * 5),
            ],
            supply_req_pointers=[
                ElementCropRow(1, [1.0] * 5),
                ElementCropRow(2, [2.0] * 5),
            ],
            irrigation_pointers=[
                ElementCropRow(1, [1.0] * 5),
                ElementCropRow(2, [2.0] * 5),
            ],
            ponding_depth_file=Path("ponding.dat"),
            operations_flow_file=Path("ops.dat"),
            ponding_depth_pointers=[
                ElementCropRow(1, [0.5] * 5),
                ElementCropRow(2, [0.5] * 5),
            ],
            decomp_water_pointers=[
                ElementCropRow(1, [0.0] * 5),
                ElementCropRow(2, [0.0] * 5),
            ],
            return_flow_pointers=[
                ElementCropRow(1, [0.3] * 5),
                ElementCropRow(2, [0.3] * 5),
            ],
            reuse_pointers=[
                ElementCropRow(1, [0.1] * 5),
                ElementCropRow(2, [0.1] * 5),
            ],
            initial_conditions=[
                AgInitialConditionRow(1, 0.5, [0.15] * 5),
                AgInitialConditionRow(2, 0.5, [0.15] * 5),
            ],
        )
        out = tmp_path / "ponded_out.dat"
        PondedCropWriterV4x().write(cfg, out)

        cfg2 = read_ponded_v4x(out, n_elements=2)
        assert len(cfg2.root_depths) == 5
        assert cfg2.root_depths[0] == pytest.approx(2.0)
        assert len(cfg2.curve_numbers) == 2


# =====================================================================
# ElementSoilParamRow dataclass tests
# =====================================================================


class TestElementSoilParamRow:
    def test_defaults(self):
        row = ElementSoilParamRow(
            element_id=1,
            wilting_point=0.1,
            field_capacity=0.2,
            total_porosity=0.45,
            lambda_param=0.62,
            hydraulic_conductivity=2.6,
            kunsat_method=2,
            precip_column=1,
            precip_factor=1.0,
            generic_moisture_column=0,
        )
        assert row.surface_flow_dest_type == 0
        assert row.k_ponded == -1.0
        assert row.capillary_rise == 0.0
        assert row.dest_ag == 0


# =====================================================================
# Backward compatibility tests
# =====================================================================


class TestBackwardCompatibility:
    def test_rootzone_old_construction_still_works(self):
        """RootZone constructed without new fields still works."""
        rz = RootZone(n_elements=5, n_layers=1)
        assert rz.n_elements == 5
        assert rz.nonponded_config is None
        assert rz.surface_flow_destinations == {}

    def test_soil_params_old_construction_still_works(self):
        """SoilParameters with only 4 args still works."""
        sp = SoilParameters(0.45, 0.20, 0.10, 2.5)
        assert sp.porosity == 0.45
        assert sp.available_water == pytest.approx(0.10)
        assert sp.drainable_porosity == pytest.approx(0.25)
