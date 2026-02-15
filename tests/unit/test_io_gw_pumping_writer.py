"""Tests for pyiwfm.io.gw_pumping_writer module."""

from __future__ import annotations

from pathlib import Path

import pytest

from pyiwfm.io.gw_pumping import (
    ElementGroup,
    ElementPumpingSpec,
    PumpingConfig,
    WellPumpingSpec,
    WellSpec,
)
from pyiwfm.io.gw_pumping_writer import (
    _write_comment,
    _write_value,
    write_elem_pump_file,
    write_pumping_main,
    write_well_spec_file,
)


# ---------------------------------------------------------------------------
# Helper: in-memory StringIO tests for low-level helpers
# ---------------------------------------------------------------------------

import io


class TestWriteComment:
    def test_basic_comment(self) -> None:
        buf = io.StringIO()
        _write_comment(buf, "hello world")
        assert buf.getvalue() == "C  hello world\n"

    def test_empty_comment(self) -> None:
        buf = io.StringIO()
        _write_comment(buf, "")
        assert buf.getvalue() == "C  \n"


class TestWriteValue:
    def test_with_description(self) -> None:
        buf = io.StringIO()
        _write_value(buf, 42, "number of wells")
        result = buf.getvalue()
        assert "42" in result
        assert "/ number of wells" in result
        assert result.endswith("\n")

    def test_without_description(self) -> None:
        buf = io.StringIO()
        _write_value(buf, "some_file.dat")
        result = buf.getvalue()
        assert "some_file.dat" in result
        assert "/" not in result
        assert result.endswith("\n")

    def test_with_empty_description(self) -> None:
        buf = io.StringIO()
        _write_value(buf, 99, "")
        result = buf.getvalue()
        # Empty description treated same as no description
        assert "/" not in result
        assert "99" in result


# ---------------------------------------------------------------------------
# write_pumping_main
# ---------------------------------------------------------------------------


class TestWritePumpingMain:
    def test_basic_main_file(self, tmp_path: Path) -> None:
        config = PumpingConfig(
            version="4.0",
            well_file=Path("wells.dat"),
            elem_pump_file=Path("elem_pump.dat"),
            ts_data_file=Path("pump_ts.dat"),
            output_file=Path("pump_out.dat"),
        )
        out = tmp_path / "pump_main.dat"
        result = write_pumping_main(config, out)

        assert result == out
        assert out.exists()

        text = out.read_text()
        assert "C  IWFM Pumping Main File" in text
        assert "#4.0" in text
        assert "wells.dat" in text
        assert "elem_pump.dat" in text
        assert "pump_ts.dat" in text
        assert "pump_out.dat" in text
        assert "/ Well specification file" in text
        assert "/ Element pumping file" in text
        assert "/ Time series data file" in text
        assert "/ Output file" in text

    def test_no_version(self, tmp_path: Path) -> None:
        config = PumpingConfig(version="")
        out = tmp_path / "pump_main.dat"
        write_pumping_main(config, out)

        text = out.read_text()
        assert "#" not in text

    def test_none_paths(self, tmp_path: Path) -> None:
        config = PumpingConfig()
        out = tmp_path / "pump_main.dat"
        write_pumping_main(config, out)

        text = out.read_text()
        # None becomes empty string via (config.well_file or "")
        assert "/ Well specification file" in text

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        config = PumpingConfig(version="4.0")
        out = tmp_path / "subdir" / "deep" / "pump_main.dat"
        result = write_pumping_main(config, out)

        assert result == out
        assert out.exists()

    def test_string_filepath(self, tmp_path: Path) -> None:
        config = PumpingConfig()
        out = str(tmp_path / "pump_main.dat")
        result = write_pumping_main(config, out)

        assert isinstance(result, Path)
        assert result.exists()

    def test_returns_path_object(self, tmp_path: Path) -> None:
        config = PumpingConfig()
        out = tmp_path / "pump_main.dat"
        result = write_pumping_main(config, out)
        assert isinstance(result, Path)


# ---------------------------------------------------------------------------
# write_well_spec_file
# ---------------------------------------------------------------------------


class TestWriteWellSpecFile:
    def _make_config(self) -> PumpingConfig:
        return PumpingConfig(
            factor_xy=1.0,
            factor_radius=1.0,
            factor_length=1.0,
            well_specs=[
                WellSpec(
                    id=1,
                    x=100.0,
                    y=200.0,
                    radius=0.5,
                    perf_top=50.0,
                    perf_bottom=10.0,
                    name="Well A",
                ),
                WellSpec(
                    id=2,
                    x=300.0,
                    y=400.0,
                    radius=0.75,
                    perf_top=60.0,
                    perf_bottom=20.0,
                    name="",
                ),
            ],
            well_pumping_specs=[
                WellPumpingSpec(
                    well_id=1,
                    pump_column=1,
                    pump_fraction=1.0,
                    dist_method=0,
                    dest_type=-1,
                    dest_id=0,
                    irig_frac_column=0,
                    adjust_column=0,
                    pump_max_column=0,
                    pump_max_fraction=0.0,
                ),
                WellPumpingSpec(
                    well_id=2,
                    pump_column=2,
                    pump_fraction=0.5,
                    dist_method=1,
                    dest_type=1,
                    dest_id=5,
                    irig_frac_column=3,
                    adjust_column=4,
                    pump_max_column=5,
                    pump_max_fraction=0.8,
                ),
            ],
            well_groups=[
                ElementGroup(id=1, elements=[10, 20, 30]),
                ElementGroup(id=2, elements=[40]),
            ],
        )

    def test_basic_well_file(self, tmp_path: Path) -> None:
        config = self._make_config()
        out = tmp_path / "wells.dat"
        result = write_well_spec_file(config, out)

        assert result == out
        assert out.exists()

        text = out.read_text()
        assert "C  IWFM Well Specification File" in text
        assert "/ NWELL" in text
        assert "/ FACTXY" in text
        assert "/ FACTR" in text
        assert "/ FACTLT" in text

    def test_well_structural_data_with_factors(self, tmp_path: Path) -> None:
        config = self._make_config()
        config.factor_xy = 2.0
        config.factor_radius = 0.5
        config.factor_length = 3.0
        # WellSpec x=100 => file x=100/2.0=50
        # WellSpec radius=0.5 => diameter = 0.5*2/0.5 = 2.0
        # WellSpec perf_top=50 => file pt = 50/3.0 = 16.6667
        out = tmp_path / "wells.dat"
        write_well_spec_file(config, out)

        text = out.read_text()
        lines = text.strip().split("\n")

        # Find structural data lines (after header lines)
        # Header: comment, NWELL, FACTXY, FACTR, FACTLT = 5 lines
        struct_line_1 = lines[5]
        assert "50.0000" in struct_line_1  # x / factor_xy = 100 / 2 = 50
        assert "100.0000" in struct_line_1  # y / factor_xy = 200 / 2 = 100
        assert "/ Well A" in struct_line_1

    def test_well_name_present(self, tmp_path: Path) -> None:
        config = self._make_config()
        out = tmp_path / "wells.dat"
        write_well_spec_file(config, out)

        text = out.read_text()
        assert "/ Well A" in text

    def test_well_name_absent(self, tmp_path: Path) -> None:
        config = self._make_config()
        out = tmp_path / "wells.dat"
        write_well_spec_file(config, out)

        text = out.read_text()
        lines = text.strip().split("\n")
        # Second well has no name - should not have "/ " at end
        struct_line_2 = lines[6]
        assert "/ " not in struct_line_2

    def test_zero_factor_xy(self, tmp_path: Path) -> None:
        """When factor_xy is 0 (falsy), x and y should be written as-is."""
        config = PumpingConfig(
            factor_xy=0.0,
            factor_radius=1.0,
            factor_length=1.0,
            well_specs=[
                WellSpec(id=1, x=100.0, y=200.0, radius=0.5, perf_top=50.0, perf_bottom=10.0),
            ],
            well_pumping_specs=[],
            well_groups=[],
        )
        out = tmp_path / "wells.dat"
        write_well_spec_file(config, out)

        text = out.read_text()
        assert "100.0000" in text
        assert "200.0000" in text

    def test_zero_factor_radius(self, tmp_path: Path) -> None:
        """When factor_radius is 0 (falsy), diameter = radius * 2.0 (no division)."""
        config = PumpingConfig(
            factor_xy=1.0,
            factor_radius=0.0,
            factor_length=1.0,
            well_specs=[
                WellSpec(id=1, x=0.0, y=0.0, radius=0.5, perf_top=0.0, perf_bottom=0.0),
            ],
            well_pumping_specs=[],
            well_groups=[],
        )
        out = tmp_path / "wells.dat"
        write_well_spec_file(config, out)

        text = out.read_text()
        # diameter = 0.5 * 2.0 = 1.0
        assert "1.0000" in text

    def test_zero_factor_length(self, tmp_path: Path) -> None:
        """When factor_length is 0 (falsy), perf_top and perf_bottom are written as-is."""
        config = PumpingConfig(
            factor_xy=1.0,
            factor_radius=1.0,
            factor_length=0.0,
            well_specs=[
                WellSpec(id=1, x=0.0, y=0.0, radius=0.5, perf_top=50.0, perf_bottom=10.0),
            ],
            well_pumping_specs=[],
            well_groups=[],
        )
        out = tmp_path / "wells.dat"
        write_well_spec_file(config, out)

        text = out.read_text()
        assert "50.0000" in text
        assert "10.0000" in text

    def test_pumping_specs_written(self, tmp_path: Path) -> None:
        config = self._make_config()
        out = tmp_path / "wells.dat"
        write_well_spec_file(config, out)

        text = out.read_text()
        # Check pump_fraction values
        assert "1.0000" in text
        assert "0.5000" in text

    def test_groups_with_multiple_elements(self, tmp_path: Path) -> None:
        config = self._make_config()
        out = tmp_path / "wells.dat"
        write_well_spec_file(config, out)

        text = out.read_text()
        assert "/ NGROUPS" in text
        # Group 1 has 3 elements; first line has group_id, count, first element
        # Subsequent lines have one element each
        assert "10" in text
        assert "20" in text
        assert "30" in text

    def test_groups_single_element(self, tmp_path: Path) -> None:
        config = self._make_config()
        out = tmp_path / "wells.dat"
        write_well_spec_file(config, out)

        text = out.read_text()
        # Group 2 has 1 element: [40]
        assert "40" in text

    def test_empty_well_specs(self, tmp_path: Path) -> None:
        config = PumpingConfig(
            well_specs=[],
            well_pumping_specs=[],
            well_groups=[],
        )
        out = tmp_path / "wells.dat"
        write_well_spec_file(config, out)

        text = out.read_text()
        assert "/ NWELL" in text
        # NWELL should be 0
        lines = text.strip().split("\n")
        nwell_line = [l for l in lines if "NWELL" in l][0]
        assert "0" in nwell_line

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        config = PumpingConfig(well_specs=[], well_pumping_specs=[], well_groups=[])
        out = tmp_path / "deep" / "dir" / "wells.dat"
        result = write_well_spec_file(config, out)
        assert result.exists()

    def test_string_filepath(self, tmp_path: Path) -> None:
        config = PumpingConfig(well_specs=[], well_pumping_specs=[], well_groups=[])
        out = str(tmp_path / "wells.dat")
        result = write_well_spec_file(config, out)
        assert isinstance(result, Path)
        assert result.exists()


# ---------------------------------------------------------------------------
# write_elem_pump_file
# ---------------------------------------------------------------------------


class TestWriteElemPumpFile:
    def _make_config(self) -> PumpingConfig:
        return PumpingConfig(
            elem_pumping_specs=[
                ElementPumpingSpec(
                    element_id=1,
                    pump_column=1,
                    pump_fraction=1.0,
                    dist_method=0,
                    layer_factors=[0.5, 0.3, 0.2],
                    dest_type=-1,
                    dest_id=0,
                    irig_frac_column=0,
                    adjust_column=0,
                    pump_max_column=0,
                    pump_max_fraction=0.0,
                ),
                ElementPumpingSpec(
                    element_id=2,
                    pump_column=3,
                    pump_fraction=0.75,
                    dist_method=2,
                    layer_factors=[1.0],
                    dest_type=3,
                    dest_id=10,
                    irig_frac_column=5,
                    adjust_column=6,
                    pump_max_column=7,
                    pump_max_fraction=0.9,
                ),
            ],
            elem_groups=[
                ElementGroup(id=1, elements=[100, 200]),
                ElementGroup(id=2, elements=[300]),
            ],
        )

    def test_basic_elem_pump_file(self, tmp_path: Path) -> None:
        config = self._make_config()
        out = tmp_path / "elem_pump.dat"
        result = write_elem_pump_file(config, out)

        assert result == out
        assert out.exists()

        text = out.read_text()
        assert "C  IWFM Element Pumping Specification File" in text
        assert "/ NSINK" in text

    def test_elem_pump_specs_written(self, tmp_path: Path) -> None:
        config = self._make_config()
        out = tmp_path / "elem_pump.dat"
        write_elem_pump_file(config, out)

        text = out.read_text()
        # Check layer factors are formatted
        assert "0.5000" in text
        assert "0.3000" in text
        assert "0.2000" in text
        # Check second spec
        assert "0.7500" in text
        assert "0.9000" in text

    def test_elem_groups_written(self, tmp_path: Path) -> None:
        config = self._make_config()
        out = tmp_path / "elem_pump.dat"
        write_elem_pump_file(config, out)

        text = out.read_text()
        assert "/ NGROUPS" in text
        assert "100" in text
        assert "200" in text
        assert "300" in text

    def test_groups_multi_element_format(self, tmp_path: Path) -> None:
        """First element line has group_id, count, first_elem. Rest have just elem_id."""
        config = PumpingConfig(
            elem_pumping_specs=[],
            elem_groups=[
                ElementGroup(id=5, elements=[10, 20, 30]),
            ],
        )
        out = tmp_path / "elem_pump.dat"
        write_elem_pump_file(config, out)

        text = out.read_text()
        lines = text.strip().split("\n")
        # Find the group lines after NGROUPS
        ngroups_idx = next(i for i, l in enumerate(lines) if "NGROUPS" in l)
        group_line = lines[ngroups_idx + 1]
        # Should contain group_id=5, count=3, first element=10
        parts = group_line.split()
        assert parts[0] == "5"
        assert parts[1] == "3"
        assert parts[2] == "10"
        # Next lines should be continuation elements
        assert lines[ngroups_idx + 2].strip() == "20"
        assert lines[ngroups_idx + 3].strip() == "30"

    def test_empty_elem_specs(self, tmp_path: Path) -> None:
        config = PumpingConfig(elem_pumping_specs=[], elem_groups=[])
        out = tmp_path / "elem_pump.dat"
        write_elem_pump_file(config, out)

        text = out.read_text()
        assert "/ NSINK" in text
        nsink_line = [l for l in text.strip().split("\n") if "NSINK" in l][0]
        assert "0" in nsink_line

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        config = PumpingConfig(elem_pumping_specs=[], elem_groups=[])
        out = tmp_path / "sub" / "dir" / "elem_pump.dat"
        result = write_elem_pump_file(config, out)
        assert result.exists()

    def test_string_filepath(self, tmp_path: Path) -> None:
        config = PumpingConfig(elem_pumping_specs=[], elem_groups=[])
        out = str(tmp_path / "elem_pump.dat")
        result = write_elem_pump_file(config, out)
        assert isinstance(result, Path)
        assert result.exists()

    def test_single_layer_factor(self, tmp_path: Path) -> None:
        config = PumpingConfig(
            elem_pumping_specs=[
                ElementPumpingSpec(
                    element_id=1,
                    pump_column=1,
                    pump_fraction=1.0,
                    dist_method=0,
                    layer_factors=[1.0],
                    dest_type=-1,
                    dest_id=0,
                    irig_frac_column=0,
                    adjust_column=0,
                    pump_max_column=0,
                    pump_max_fraction=0.0,
                ),
            ],
            elem_groups=[],
        )
        out = tmp_path / "elem_pump.dat"
        write_elem_pump_file(config, out, n_layers=1)

        text = out.read_text()
        assert "1.0000" in text

    def test_empty_groups(self, tmp_path: Path) -> None:
        config = PumpingConfig(
            elem_pumping_specs=[],
            elem_groups=[],
        )
        out = tmp_path / "elem_pump.dat"
        write_elem_pump_file(config, out)

        text = out.read_text()
        assert "/ NGROUPS" in text
        ngroups_line = [l for l in text.strip().split("\n") if "NGROUPS" in l][0]
        assert "0" in ngroups_line


# ---------------------------------------------------------------------------
# Roundtrip-style integration test
# ---------------------------------------------------------------------------


class TestWriterRoundtrip:
    """Verify that data written can be parsed back via the reader."""

    def test_well_file_roundtrip(self, tmp_path: Path) -> None:
        """Write a well spec file and read it back, checking key values."""
        from pyiwfm.io.gw_pumping import PumpingReader

        config = PumpingConfig(
            factor_xy=1.0,
            factor_radius=1.0,
            factor_length=1.0,
            well_specs=[
                WellSpec(
                    id=1,
                    x=100.0,
                    y=200.0,
                    radius=0.5,
                    perf_top=50.0,
                    perf_bottom=10.0,
                    name="Test Well",
                ),
            ],
            well_pumping_specs=[
                WellPumpingSpec(
                    well_id=1,
                    pump_column=1,
                    pump_fraction=1.0,
                    dist_method=0,
                    dest_type=-1,
                    dest_id=0,
                    irig_frac_column=0,
                    adjust_column=0,
                    pump_max_column=0,
                    pump_max_fraction=0.0,
                ),
            ],
            well_groups=[
                ElementGroup(id=1, elements=[10, 20]),
            ],
        )

        out = tmp_path / "wells.dat"
        write_well_spec_file(config, out)

        # Read it back
        reader = PumpingReader()
        result = PumpingConfig()
        reader._read_well_file(out, result)

        assert len(result.well_specs) == 1
        ws = result.well_specs[0]
        assert ws.id == 1
        assert ws.name == "Test Well"
        assert abs(ws.x - 100.0) < 0.01
        assert abs(ws.y - 200.0) < 0.01

        assert len(result.well_pumping_specs) == 1
        wps = result.well_pumping_specs[0]
        assert wps.well_id == 1
        assert wps.pump_column == 1

        assert len(result.well_groups) == 1
        grp = result.well_groups[0]
        assert grp.id == 1
        assert grp.elements == [10, 20]

    def test_elem_pump_file_roundtrip(self, tmp_path: Path) -> None:
        """Write an elem pump file and read it back."""
        from pyiwfm.io.gw_pumping import PumpingReader

        config = PumpingConfig(
            elem_pumping_specs=[
                ElementPumpingSpec(
                    element_id=5,
                    pump_column=2,
                    pump_fraction=0.8,
                    dist_method=1,
                    layer_factors=[0.6, 0.4],
                    dest_type=2,
                    dest_id=3,
                    irig_frac_column=4,
                    adjust_column=5,
                    pump_max_column=6,
                    pump_max_fraction=0.9,
                ),
            ],
            elem_groups=[
                ElementGroup(id=1, elements=[100]),
            ],
        )

        out = tmp_path / "elem_pump.dat"
        write_elem_pump_file(config, out, n_layers=2)

        # Read it back
        reader = PumpingReader()
        result = PumpingConfig()
        reader._read_elem_pump_file(out, result, n_layers=2)

        assert len(result.elem_pumping_specs) == 1
        eps = result.elem_pumping_specs[0]
        assert eps.element_id == 5
        assert eps.pump_column == 2
        assert abs(eps.pump_fraction - 0.8) < 0.001
        assert len(eps.layer_factors) == 2
        assert abs(eps.layer_factors[0] - 0.6) < 0.001
        assert abs(eps.layer_factors[1] - 0.4) < 0.001
        assert eps.dest_type == 2
        assert eps.dest_id == 3

        assert len(result.elem_groups) == 1
        assert result.elem_groups[0].elements == [100]
