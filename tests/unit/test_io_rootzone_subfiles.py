"""Tests for IWFM RootZone sub-file readers.

Covers:
- NonPondedCropReader / read_nonponded_crop
- PondedCropReader / read_ponded_crop  (same format as non-ponded)
- UrbanLandUseReader / read_urban_landuse
- NativeRiparianReader / read_native_riparian
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pyiwfm.io.rootzone_nonponded import (
    NonPondedCropConfig,
    NonPondedCropReader,
    CurveNumberRow,
    EtcPointerRow,
    SupplyReturnReuseRow,
    InitialConditionRow,
    read_nonponded_crop,
)
from pyiwfm.io.rootzone_ponded import (
    PondedCropConfig,
    PondedCropReader,
    read_ponded_crop,
)
from pyiwfm.io.rootzone_urban import (
    UrbanLandUseConfig,
    UrbanLandUseReader,
    UrbanManagementRow,
    SurfaceFlowDestRow,
    UrbanInitialConditionRow,
    read_urban_landuse,
)
from pyiwfm.io.rootzone_native import (
    NativeRiparianConfig,
    NativeRiparianReader,
    NativeRiparianCNRow,
    NativeRiparianEtcRow,
    NativeRiparianInitialRow,
    read_native_riparian,
)
from pyiwfm.core.exceptions import FileFormatError


# =====================================================================
# Fixtures – minimal sample files
# =====================================================================


NONPONDED_SAMPLE = """\
C  Non-ponded agricultural crop file
C  Generated for test
3                                               / NCrops
SubRegion\\CropArea.dat                          / Subregional area file
Element\\AgArea.dat                              / Elemental area file
1.0                                             / Output factor
ACRES                                           / Output unit
..\\Results\\AvgCrop.out                          / Output file
0.3048                                          / Root depth factor
3.5                                             / Root depth crop 1
4.0                                             / Root depth crop 2
2.8                                             / Root depth crop 3
C  Curve numbers (subregion_id  CN_soil1  CN_soil2)
1   75.0   80.0
2   70.0   78.0
C  ETc column pointers (subregion_id  col_crop1  col_crop2  col_crop3)
1   1   2   3
2   4   5   6
C  Irrigation period data file
IrrigPeriod.dat
C  Irrigation period pointers
1   1   2   3
2   4   5   6
C  Minimum soil moisture data file
MinMoist.dat
C  Min soil moisture pointers
1   1   2   3
2   4   5   6
C  Target soil moisture data file (optional)
TargetMoist.dat
C  Target soil moisture pointers
1   1   1   1
2   1   1   1
C  Water demand data file (optional)
WaterDemand.dat
C  Demand from moisture flag
1
C  Supply / return flow / reuse (subregion_id  supply  return  reuse)
1   1   1   1
2   2   2   2
C  Initial conditions (subregion_id  pf1 mc1  pf2 mc2)
1   0.5   0.3   0.4   0.25
2   0.6   0.35  0.5   0.28
"""


URBAN_SAMPLE = """\
C  Urban land use file
C  Test data
UrbanArea.dat                                   / Area data file
0.3048                                          / Root depth factor
3.0                                             / Root depth value
C  Curve numbers (subregion_id  CN_soil1  CN_soil2)
1   85.0   90.0
2   82.0   88.0
C  Urban water demand file
UrbanDemand.dat
C  Water use specs file
WaterUseSpecs.dat
C  Management data (sub_id  perv_frac  demand  wuse  etc  retflow  reuse)
1   0.35  1   1   1   1   1
2   0.40  2   2   2   2   2
C  Surface flow destinations (elem_id  dest_type  dest_id)
1   2   10
2   2   15
3   1   0
C  Initial conditions (sub_id  pf1 mc1  pf2 mc2)
1   0.4   0.20   0.3   0.18
2   0.5   0.25   0.4   0.22
"""


NATIVE_RIPARIAN_SAMPLE = """\
C  Native/Riparian vegetation file
C  Test data
NativeRipArea.dat                               / Area data file
0.3048                                          / Root depth factor
5.0                                             / Native root depth
8.0                                             / Riparian root depth
C  Curve numbers (sub_id  native_cn1 native_cn2  riparian_cn1 riparian_cn2)
1   60.0   65.0   55.0   58.0
2   62.0   67.0   57.0   60.0
C  ETc pointers (sub_id  native_col  riparian_col)
1   1   2
2   3   4
C  Initial conditions (sub_id  native_soil1 riparian_soil1  native_soil2 riparian_soil2)
1   0.30   0.45   0.28   0.42
2   0.32   0.48   0.30   0.44
"""


# =====================================================================
# NonPondedCropReader tests
# =====================================================================


class TestNonPondedCropReader:
    """Tests for NonPondedCropReader."""

    def test_read_basic_config(self, tmp_path: Path) -> None:
        """Read a complete non-ponded crop file."""
        f = tmp_path / "nonponded.dat"
        f.write_text(NONPONDED_SAMPLE)

        config = read_nonponded_crop(f)

        assert config.n_crops == 3
        assert config.subregional_area_file is not None
        assert config.elemental_area_file is not None
        assert config.area_output_factor == 1.0
        assert config.area_output_unit == "ACRES"
        assert config.area_output_file is not None

    def test_root_depths_with_factor(self, tmp_path: Path) -> None:
        """Root depths are multiplied by the conversion factor."""
        f = tmp_path / "nonponded.dat"
        f.write_text(NONPONDED_SAMPLE)

        config = read_nonponded_crop(f)

        assert len(config.root_depths) == 3
        assert config.root_depth_factor == pytest.approx(0.3048)
        # 3.5 * 0.3048 = 1.0668
        assert config.root_depths[0] == pytest.approx(3.5 * 0.3048)
        assert config.root_depths[1] == pytest.approx(4.0 * 0.3048)
        assert config.root_depths[2] == pytest.approx(2.8 * 0.3048)

    def test_curve_numbers(self, tmp_path: Path) -> None:
        """Parse curve-number rows."""
        f = tmp_path / "nonponded.dat"
        f.write_text(NONPONDED_SAMPLE)

        config = read_nonponded_crop(f)

        assert len(config.curve_numbers) == 2
        assert config.curve_numbers[0].subregion_id == 1
        assert config.curve_numbers[0].cn_values == [75.0, 80.0]
        assert config.curve_numbers[1].subregion_id == 2

    def test_etc_pointers(self, tmp_path: Path) -> None:
        """Parse ETc column pointers."""
        f = tmp_path / "nonponded.dat"
        f.write_text(NONPONDED_SAMPLE)

        config = read_nonponded_crop(f)

        assert len(config.etc_pointers) == 2
        assert config.etc_pointers[0].etc_columns == [1, 2, 3]
        assert config.etc_pointers[1].etc_columns == [4, 5, 6]

    def test_irrigation_period(self, tmp_path: Path) -> None:
        """Parse irrigation period file and pointers."""
        f = tmp_path / "nonponded.dat"
        f.write_text(NONPONDED_SAMPLE)

        config = read_nonponded_crop(f)

        assert config.irrigation_period_file is not None
        assert "IrrigPeriod.dat" in str(config.irrigation_period_file)
        assert len(config.irrigation_pointers) == 2

    def test_soil_moisture_files(self, tmp_path: Path) -> None:
        """Parse minimum and target soil moisture files."""
        f = tmp_path / "nonponded.dat"
        f.write_text(NONPONDED_SAMPLE)

        config = read_nonponded_crop(f)

        assert config.min_soil_moisture_file is not None
        assert config.target_soil_moisture_file is not None
        assert len(config.min_moisture_pointers) == 2
        assert len(config.target_moisture_pointers) == 2

    def test_water_demand_and_flag(self, tmp_path: Path) -> None:
        """Parse water demand file and demand-from-moisture flag."""
        f = tmp_path / "nonponded.dat"
        f.write_text(NONPONDED_SAMPLE)

        config = read_nonponded_crop(f)

        assert config.water_demand_file is not None
        assert config.demand_from_moisture_flag == 1

    def test_supply_return_reuse(self, tmp_path: Path) -> None:
        """Parse supply / return-flow / reuse rows."""
        f = tmp_path / "nonponded.dat"
        f.write_text(NONPONDED_SAMPLE)

        config = read_nonponded_crop(f)

        assert len(config.supply_return_reuse) == 2
        assert config.supply_return_reuse[0].subregion_id == 1
        assert config.supply_return_reuse[0].supply_column == 1
        assert config.supply_return_reuse[1].reuse_column == 2

    def test_initial_conditions(self, tmp_path: Path) -> None:
        """Parse initial soil-moisture conditions."""
        f = tmp_path / "nonponded.dat"
        f.write_text(NONPONDED_SAMPLE)

        config = read_nonponded_crop(f)

        assert len(config.initial_conditions) == 2
        assert config.initial_conditions[0].subregion_id == 1
        assert config.initial_conditions[0].precip_fractions == pytest.approx(
            [0.5, 0.4]
        )
        assert config.initial_conditions[0].moisture_contents == pytest.approx(
            [0.3, 0.25]
        )

    def test_invalid_ncrops_raises(self, tmp_path: Path) -> None:
        """Non-integer NCrops raises FileFormatError."""
        f = tmp_path / "bad.dat"
        f.write_text("C bad file\nabc\n")

        reader = NonPondedCropReader()
        with pytest.raises(FileFormatError, match="Invalid NCrops"):
            reader.read(f)

    def test_convenience_function(self, tmp_path: Path) -> None:
        """read_nonponded_crop convenience function works."""
        f = tmp_path / "nonponded.dat"
        f.write_text(NONPONDED_SAMPLE)

        config = read_nonponded_crop(str(f))
        assert config.n_crops == 3

    def test_minimal_file(self, tmp_path: Path) -> None:
        """Minimal file with just NCrops and no sub-sections."""
        f = tmp_path / "minimal.dat"
        f.write_text("C Minimal\n0\n")

        config = read_nonponded_crop(f)
        assert config.n_crops == 0
        assert config.root_depths == []

    def test_resolve_relative_paths(self, tmp_path: Path) -> None:
        """Sub-file paths are resolved relative to base_dir."""
        f = tmp_path / "subdir" / "nonponded.dat"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(NONPONDED_SAMPLE)

        config = read_nonponded_crop(f)

        assert config.subregional_area_file is not None
        assert str(config.subregional_area_file).startswith(str(tmp_path))


# =====================================================================
# PondedCropReader tests
# =====================================================================


class TestPondedCropReader:
    """Tests for PondedCropReader (same format as non-ponded)."""

    def test_read_same_format(self, tmp_path: Path) -> None:
        """PondedCropReader reads the same format as NonPondedCropReader."""
        f = tmp_path / "ponded.dat"
        f.write_text(NONPONDED_SAMPLE)

        config = read_ponded_crop(f)

        assert config.n_crops == 3
        assert len(config.root_depths) == 3

    def test_type_alias(self) -> None:
        """PondedCropConfig is an alias for NonPondedCropConfig."""
        assert PondedCropConfig is NonPondedCropConfig

    def test_ponded_reader_subclass(self) -> None:
        """PondedCropReader is a subclass of NonPondedCropReader."""
        assert issubclass(PondedCropReader, NonPondedCropReader)


# =====================================================================
# UrbanLandUseReader tests
# =====================================================================


class TestUrbanLandUseReader:
    """Tests for UrbanLandUseReader."""

    def test_read_basic_config(self, tmp_path: Path) -> None:
        """Read a complete urban land-use file."""
        f = tmp_path / "urban.dat"
        f.write_text(URBAN_SAMPLE)

        config = read_urban_landuse(f)

        assert config.area_data_file is not None
        assert "UrbanArea.dat" in str(config.area_data_file)

    def test_root_depth(self, tmp_path: Path) -> None:
        """Root depth is multiplied by conversion factor."""
        f = tmp_path / "urban.dat"
        f.write_text(URBAN_SAMPLE)

        config = read_urban_landuse(f)

        assert config.root_depth_factor == pytest.approx(0.3048)
        assert config.root_depth == pytest.approx(3.0 * 0.3048)

    def test_curve_numbers(self, tmp_path: Path) -> None:
        """Parse urban curve-number rows."""
        f = tmp_path / "urban.dat"
        f.write_text(URBAN_SAMPLE)

        config = read_urban_landuse(f)

        assert len(config.curve_numbers) == 2
        assert config.curve_numbers[0].cn_values == [85.0, 90.0]

    def test_demand_and_specs_files(self, tmp_path: Path) -> None:
        """Parse demand and water-use specification file paths."""
        f = tmp_path / "urban.dat"
        f.write_text(URBAN_SAMPLE)

        config = read_urban_landuse(f)

        assert config.demand_file is not None
        assert "UrbanDemand.dat" in str(config.demand_file)
        assert config.water_use_specs_file is not None
        assert "WaterUseSpecs.dat" in str(config.water_use_specs_file)

    def test_management_rows(self, tmp_path: Path) -> None:
        """Parse per-subregion management data (7 columns)."""
        f = tmp_path / "urban.dat"
        f.write_text(URBAN_SAMPLE)

        config = read_urban_landuse(f)

        assert len(config.management) == 2
        m1 = config.management[0]
        assert m1.subregion_id == 1
        assert m1.pervious_fraction == pytest.approx(0.35)
        assert m1.demand_column == 1
        assert m1.etc_column == 1
        m2 = config.management[1]
        assert m2.pervious_fraction == pytest.approx(0.40)
        assert m2.demand_column == 2

    def test_surface_flow_destinations(self, tmp_path: Path) -> None:
        """Parse per-element surface-flow destinations."""
        f = tmp_path / "urban.dat"
        f.write_text(URBAN_SAMPLE)

        config = read_urban_landuse(f)

        assert len(config.surface_flow_destinations) == 3
        d1 = config.surface_flow_destinations[0]
        assert d1.element_id == 1
        assert d1.dest_type == 2  # stream node
        assert d1.dest_id == 10
        d3 = config.surface_flow_destinations[2]
        assert d3.dest_type == 1  # outside domain
        assert d3.dest_id == 0

    def test_initial_conditions(self, tmp_path: Path) -> None:
        """Parse initial soil-moisture conditions."""
        f = tmp_path / "urban.dat"
        f.write_text(URBAN_SAMPLE)

        config = read_urban_landuse(f)

        assert len(config.initial_conditions) == 2
        ic1 = config.initial_conditions[0]
        assert ic1.precip_fractions == pytest.approx([0.4, 0.3])
        assert ic1.moisture_contents == pytest.approx([0.20, 0.18])

    def test_invalid_root_depth_raises(self, tmp_path: Path) -> None:
        """Non-numeric root depth raises FileFormatError."""
        f = tmp_path / "bad.dat"
        f.write_text("C bad\nArea.dat\nabc\n")

        reader = UrbanLandUseReader()
        with pytest.raises(FileFormatError, match="Invalid root depth factor"):
            reader.read(f)

    def test_convenience_function(self, tmp_path: Path) -> None:
        """read_urban_landuse convenience function works."""
        f = tmp_path / "urban.dat"
        f.write_text(URBAN_SAMPLE)

        config = read_urban_landuse(str(f))
        assert config.area_data_file is not None


# =====================================================================
# NativeRiparianReader tests
# =====================================================================


class TestNativeRiparianReader:
    """Tests for NativeRiparianReader."""

    def test_read_basic_config(self, tmp_path: Path) -> None:
        """Read a complete native/riparian vegetation file."""
        f = tmp_path / "native.dat"
        f.write_text(NATIVE_RIPARIAN_SAMPLE)

        config = read_native_riparian(f)

        assert config.area_data_file is not None
        assert "NativeRipArea.dat" in str(config.area_data_file)

    def test_root_depths(self, tmp_path: Path) -> None:
        """Root depths are multiplied by conversion factor."""
        f = tmp_path / "native.dat"
        f.write_text(NATIVE_RIPARIAN_SAMPLE)

        config = read_native_riparian(f)

        assert config.root_depth_factor == pytest.approx(0.3048)
        assert config.native_root_depth == pytest.approx(5.0 * 0.3048)
        assert config.riparian_root_depth == pytest.approx(8.0 * 0.3048)

    def test_curve_numbers_split(self, tmp_path: Path) -> None:
        """Curve numbers are split between native and riparian."""
        f = tmp_path / "native.dat"
        f.write_text(NATIVE_RIPARIAN_SAMPLE)

        config = read_native_riparian(f)

        assert len(config.curve_numbers) == 2
        cn1 = config.curve_numbers[0]
        assert cn1.subregion_id == 1
        assert cn1.native_cn == [60.0, 65.0]
        assert cn1.riparian_cn == [55.0, 58.0]

    def test_etc_pointers(self, tmp_path: Path) -> None:
        """Parse ETc column pointers for native and riparian."""
        f = tmp_path / "native.dat"
        f.write_text(NATIVE_RIPARIAN_SAMPLE)

        config = read_native_riparian(f)

        assert len(config.etc_pointers) == 2
        assert config.etc_pointers[0].native_etc_column == 1
        assert config.etc_pointers[0].riparian_etc_column == 2
        assert config.etc_pointers[1].native_etc_column == 3
        assert config.etc_pointers[1].riparian_etc_column == 4

    def test_initial_conditions_split(self, tmp_path: Path) -> None:
        """Initial conditions are split between native and riparian."""
        f = tmp_path / "native.dat"
        f.write_text(NATIVE_RIPARIAN_SAMPLE)

        config = read_native_riparian(f)

        assert len(config.initial_conditions) == 2
        ic1 = config.initial_conditions[0]
        assert ic1.native_moisture == pytest.approx([0.30, 0.28])
        assert ic1.riparian_moisture == pytest.approx([0.45, 0.42])

    def test_invalid_native_root_depth_raises(self, tmp_path: Path) -> None:
        """Non-numeric native root depth raises FileFormatError."""
        f = tmp_path / "bad.dat"
        f.write_text("C bad\nArea.dat\n1.0\nabc\n")

        reader = NativeRiparianReader()
        with pytest.raises(FileFormatError, match="Invalid native root depth"):
            reader.read(f)

    def test_invalid_riparian_root_depth_raises(self, tmp_path: Path) -> None:
        """Non-numeric riparian root depth raises FileFormatError."""
        f = tmp_path / "bad.dat"
        f.write_text("C bad\nArea.dat\n1.0\n5.0\nabc\n")

        reader = NativeRiparianReader()
        with pytest.raises(
            FileFormatError, match="Invalid riparian root depth"
        ):
            reader.read(f)

    def test_convenience_function(self, tmp_path: Path) -> None:
        """read_native_riparian convenience function works."""
        f = tmp_path / "native.dat"
        f.write_text(NATIVE_RIPARIAN_SAMPLE)

        config = read_native_riparian(str(f))
        assert config.native_root_depth > 0

    def test_single_soil_type(self, tmp_path: Path) -> None:
        """Handle files with a single soil type."""
        content = """\
C  Native/Riparian single soil
NativeArea.dat
1.0
4.0
6.0
C  CN (sub_id  native_cn  riparian_cn)
1   65.0   58.0
C  ETc (sub_id  native  riparian)
1   1   2
C  Initial (sub_id  native  riparian)
1   0.35   0.50
"""
        f = tmp_path / "native_1soil.dat"
        f.write_text(content)

        config = read_native_riparian(f)

        assert config.native_root_depth == pytest.approx(4.0)
        assert config.riparian_root_depth == pytest.approx(6.0)
        assert len(config.curve_numbers) == 1
        assert config.curve_numbers[0].native_cn == [65.0]
        assert config.curve_numbers[0].riparian_cn == [58.0]
        assert len(config.initial_conditions) == 1
        assert config.initial_conditions[0].native_moisture == [
            pytest.approx(0.35)
        ]
        assert config.initial_conditions[0].riparian_moisture == [
            pytest.approx(0.50)
        ]


# =====================================================================
# DataClass construction tests
# =====================================================================


class TestDataClassDefaults:
    """Test that data classes initialise with sensible defaults."""

    def test_nonponded_config_defaults(self) -> None:
        config = NonPondedCropConfig()
        assert config.n_crops == 0
        assert config.root_depths == []
        assert config.curve_numbers == []

    def test_urban_config_defaults(self) -> None:
        config = UrbanLandUseConfig()
        assert config.root_depth == 0.0
        assert config.management == []
        assert config.surface_flow_destinations == []

    def test_native_config_defaults(self) -> None:
        config = NativeRiparianConfig()
        assert config.native_root_depth == 0.0
        assert config.riparian_root_depth == 0.0
        assert config.curve_numbers == []


# =====================================================================
# Import / export tests
# =====================================================================


class TestImportExports:
    """Verify all new symbols are importable from pyiwfm.io."""

    def test_nonponded_imports(self) -> None:
        from pyiwfm.io import (
            NonPondedCropConfig,
            NonPondedCropReader,
            CurveNumberRow,
            EtcPointerRow,
            IrrigationPointerRow,
            SoilMoisturePointerRow,
            SupplyReturnReuseRow,
            InitialConditionRow,
            read_nonponded_crop,
        )

        assert NonPondedCropConfig is not None

    def test_ponded_imports(self) -> None:
        from pyiwfm.io import (
            PondedCropConfig,
            PondedCropReader,
            read_ponded_crop,
        )

        assert PondedCropConfig is not None

    def test_urban_imports(self) -> None:
        from pyiwfm.io import (
            UrbanLandUseConfig,
            UrbanLandUseReader,
            UrbanCurveNumberRow,
            UrbanManagementRow,
            SurfaceFlowDestRow,
            UrbanInitialConditionRow,
            read_urban_landuse,
        )

        assert UrbanLandUseConfig is not None

    def test_native_imports(self) -> None:
        from pyiwfm.io import (
            NativeRiparianConfig,
            NativeRiparianReader,
            NativeRiparianCNRow,
            NativeRiparianEtcRow,
            NativeRiparianInitialRow,
            read_native_riparian,
        )

        assert NativeRiparianConfig is not None


# =====================================================================
# Coverage gap tests
# =====================================================================


class TestLineBufferEdgeCases:
    """Tests for _LineBuffer edge cases."""

    def test_empty_line_is_comment(self) -> None:
        """Empty string and blank lines are treated as comments."""
        from pyiwfm.io.rootzone_nonponded import _is_comment_line

        assert _is_comment_line("") is True
        assert _is_comment_line("   ") is True
        assert _is_comment_line("\n") is True

    def test_hash_not_comment_line(self) -> None:
        """Lines starting with # (after whitespace) are not comments."""
        from pyiwfm.io.rootzone_nonponded import _is_comment_line

        assert _is_comment_line("  # version") is False

    def test_buffer_pushback_at_zero(self) -> None:
        """Pushback at position 0 does not go negative."""
        from pyiwfm.io.rootzone_nonponded import _LineBuffer

        buf = _LineBuffer(["data\n"])
        buf.pushback()  # pos is still 0
        assert buf.line_num == 0

    def test_next_data_eof_raises(self) -> None:
        """next_data raises on empty buffer."""
        from pyiwfm.io.rootzone_nonponded import _LineBuffer

        buf = _LineBuffer([])
        with pytest.raises(FileFormatError, match="Unexpected end"):
            buf.next_data()

    def test_next_data_skips_blank_value(self) -> None:
        """next_data skips lines with only a comment delimiter."""
        from pyiwfm.io.rootzone_nonponded import _LineBuffer

        buf = _LineBuffer(["C comment\n", "   \n", "42\n"])
        val = buf.next_data()
        assert val == "42"

    def test_next_data_or_empty_at_eof(self) -> None:
        """next_data_or_empty returns empty string at EOF."""
        from pyiwfm.io.rootzone_nonponded import _LineBuffer

        buf = _LineBuffer([])
        assert buf.next_data_or_empty() == ""


class TestNSubregionsParameter:
    """Tests for explicit n_subregions count-based reading."""

    def test_nonponded_with_n_subregions(self, tmp_path: Path) -> None:
        """n_subregions forces exact row count, skipping mid-section comments."""
        content = """\
C  Test with n_subregions
2
Area1.dat
Area2.dat
1.0
ACRES
Output.out
1.0
3.5
4.0
C  CN data with comment inside
1   75.0   80.0
C  mid-section comment (should be skipped)
2   70.0   78.0
C  ETc pointers
1   1   2
2   4   5
"""
        f = tmp_path / "np.dat"
        f.write_text(content)

        config = read_nonponded_crop(f, n_subregions=2)

        assert len(config.curve_numbers) == 2
        assert config.curve_numbers[1].cn_values == [70.0, 78.0]
        assert len(config.etc_pointers) == 2

    def test_native_with_n_subregions(self, tmp_path: Path) -> None:
        """NativeRiparianReader with explicit n_subregions."""
        content = """\
C  Test
Area.dat
1.0
5.0
8.0
1   60.0  65.0  55.0  58.0
C  mid-section comment
2   62.0  67.0  57.0  60.0
1   1   2
2   3   4
1   0.30   0.45   0.28   0.42
2   0.32   0.48   0.30   0.44
"""
        f = tmp_path / "nr.dat"
        f.write_text(content)

        config = read_native_riparian(f, n_subregions=2)

        assert len(config.curve_numbers) == 2
        assert len(config.etc_pointers) == 2
        assert len(config.initial_conditions) == 2

    def test_urban_with_n_subregions(self, tmp_path: Path) -> None:
        """UrbanLandUseReader with explicit n_subregions and n_elements."""
        content = """\
C  Test
Area.dat
1.0
3.0
1   85.0   90.0
C  mid-section comment
2   82.0   88.0
Demand.dat
WaterUse.dat
1   0.35  1   1   1   1   1
2   0.40  2   2   2   2   2
1   2   10
2   1   0
1   0.4   0.20   0.3   0.18
2   0.5   0.25   0.4   0.22
"""
        f = tmp_path / "urb.dat"
        f.write_text(content)

        config = read_urban_landuse(f, n_subregions=2, n_elements=2)

        assert len(config.curve_numbers) == 2
        assert len(config.management) == 2
        assert len(config.surface_flow_destinations) == 2
        assert len(config.initial_conditions) == 2


class TestMalformedTabularData:
    """Tests for ValueError handling in tabular readers."""

    def test_nonponded_bad_cn_row_stops(self, tmp_path: Path) -> None:
        """Malformed CN row stops reading that section."""
        content = """\
C  Bad CN
1
Area1.dat
Area2.dat
1.0
ACRES
Out.out
1.0
3.5
C  CN with bad second row
1   75.0   80.0
abc  xyz  bad
"""
        f = tmp_path / "bad_cn.dat"
        f.write_text(content)

        config = read_nonponded_crop(f)

        # First CN row should parse; second is malformed
        assert len(config.curve_numbers) == 1

    def test_nonponded_bad_value_factor(self, tmp_path: Path) -> None:
        """Non-numeric output factor is silently ignored."""
        content = """\
C  Bad factor
2
Area1.dat
Area2.dat
abc
ACRES
Out.out
1.0
3.5
4.0
"""
        f = tmp_path / "bad_factor.dat"
        f.write_text(content)

        config = read_nonponded_crop(f)

        # Default factor preserved
        assert config.area_output_factor == 1.0
        assert config.n_crops == 2

    def test_nonponded_bad_root_depth_factor(self, tmp_path: Path) -> None:
        """Non-numeric root depth factor is silently ignored."""
        content = """\
C  Bad root depth factor
1
Area1.dat
Area2.dat
1.0
ACRES
Out.out
notanumber
3.5
"""
        f = tmp_path / "bad_rdf.dat"
        f.write_text(content)

        config = read_nonponded_crop(f)

        # Default factor preserved
        assert config.root_depth_factor == 1.0

    def test_urban_bad_cn_stops(self, tmp_path: Path) -> None:
        """Malformed urban CN row stops reading."""
        content = """\
C  Bad
Area.dat
1.0
3.0
C  CN
1   85.0
abc  bad
"""
        f = tmp_path / "bad_urban.dat"
        f.write_text(content)

        config = read_urban_landuse(f)
        assert len(config.curve_numbers) == 1

    def test_native_bad_cn_stops(self, tmp_path: Path) -> None:
        """Malformed native/riparian CN row stops reading."""
        content = """\
C  Bad
Area.dat
1.0
5.0
8.0
C  CN
1   60.0  65.0  55.0  58.0
abc  bad  data  here
"""
        f = tmp_path / "bad_native.dat"
        f.write_text(content)

        config = read_native_riparian(f)
        assert len(config.curve_numbers) == 1

    def test_urban_bad_management_stops(self, tmp_path: Path) -> None:
        """Malformed management row stops reading."""
        content = """\
C  Test
Area.dat
1.0
3.0
C  CN
1   85.0
C  Demand
Demand.dat
C  WaterUse
WaterUse.dat
C  Management
1   0.35  1   1   1   1   1
abc 0.40  2   2   2   2   2
"""
        f = tmp_path / "bad_mgmt.dat"
        f.write_text(content)

        config = read_urban_landuse(f)
        assert len(config.management) == 1

    def test_urban_bad_dest_stops(self, tmp_path: Path) -> None:
        """Malformed destination row stops reading."""
        content = """\
C  Test
Area.dat
1.0
3.0
C  CN
1   85.0
C  Demand
Demand.dat
C  WaterUse
WaterUse.dat
C  Management
1   0.35  1   1   1   1   1
C  Destinations
1   2   10
abc 1   0
"""
        f = tmp_path / "bad_dest.dat"
        f.write_text(content)

        config = read_urban_landuse(f)
        assert len(config.surface_flow_destinations) == 1

    def test_nonponded_bad_supply_stops(self, tmp_path: Path) -> None:
        """Malformed supply/return/reuse row stops reading."""
        # Use n_subregions=1 to get through CN and ETc sections
        content = """\
C  Minimal with bad supply
1
Area1.dat
Area2.dat
1.0
ACRES
Out.out
1.0
3.5
C  CN
1   75.0
C  ETc
1   1
C  Irrig file
Irrig.dat
C  Irrig pointers
1   1
C  Min moist file
MinM.dat
C  Min moist pointers
1   1
C  Target moist (optional - empty)

C  Water demand (optional - empty)

C  Demand flag
1
C  Supply/return/reuse
1   1   1   1
abc 2   2   2
"""
        f = tmp_path / "bad_supply.dat"
        f.write_text(content)

        config = read_nonponded_crop(f, n_subregions=1)
        assert len(config.supply_return_reuse) == 1

    def test_nonponded_bad_initial_stops(self, tmp_path: Path) -> None:
        """Malformed initial condition row stops reading."""
        content = """\
C  Minimal with bad initial
1
Area1.dat
Area2.dat
1.0
ACRES
Out.out
1.0
3.5
C  CN
1   75.0
C  ETc
1   1
C  Irrig file
Irrig.dat
C  Irrig pointers
1   1
C  Min moist
MinM.dat
C  Min moist pointers
1   1
C  No target

C  No demand

C  Demand flag
1
C  Supply
1   1   1   1
C  Initial conditions
1   0.5   0.3
abc 0.6   0.4
"""
        f = tmp_path / "bad_init.dat"
        f.write_text(content)

        config = read_nonponded_crop(f, n_subregions=1)
        assert len(config.initial_conditions) == 1

    def test_native_bad_etc_stops(self, tmp_path: Path) -> None:
        """Malformed ETc pointer row stops reading."""
        content = """\
C  Native with bad ETc
Area.dat
1.0
5.0
8.0
C  CN
1   60.0  65.0  55.0  58.0
C  ETc
1   1   2
abc 3   4
"""
        f = tmp_path / "bad_etc.dat"
        f.write_text(content)

        config = read_native_riparian(f)
        assert len(config.etc_pointers) == 1

    def test_native_bad_initial_stops(self, tmp_path: Path) -> None:
        """Malformed initial condition row stops reading."""
        content = """\
C  Native bad initial
Area.dat
1.0
5.0
8.0
C  CN
1   60.0  65.0  55.0  58.0
C  ETc
1   1   2
C  Init
1   0.30   0.45   0.28   0.42
abc 0.32   0.48   0.30   0.44
"""
        f = tmp_path / "bad_init_native.dat"
        f.write_text(content)

        config = read_native_riparian(f)
        assert len(config.initial_conditions) == 1

    def test_urban_bad_initial_stops(self, tmp_path: Path) -> None:
        """Malformed urban initial condition row stops reading."""
        content = """\
C  Urban bad initial
Area.dat
1.0
3.0
C  CN
1   85.0
C  Demand
Demand.dat
C  WaterUse
WaterUse.dat
C  Management
1   0.35  1   1   1   1   1
C  Destinations
1   2   10
C  Initial
1   0.4   0.20
abc 0.5   0.25
"""
        f = tmp_path / "bad_urban_init.dat"
        f.write_text(content)

        config = read_urban_landuse(f)
        assert len(config.initial_conditions) == 1


class TestMinColsPushback:
    """Tests for the min_cols pushback path in _read_rows."""

    def test_short_line_stops_section(self, tmp_path: Path) -> None:
        """A line with fewer columns than expected ends the section."""
        content = """\
C  NonPonded with short line
1
Area1.dat
Area2.dat
1.0
ACRES
Out.out
1.0
3.5
C  CN - only one row, then a short line
1   75.0   80.0
99
"""
        f = tmp_path / "short.dat"
        f.write_text(content)

        config = read_nonponded_crop(f)
        assert len(config.curve_numbers) == 1
