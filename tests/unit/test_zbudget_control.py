"""Tests for pyiwfm.io.zbudget_control -- zbudget control file parser."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from pyiwfm.io.zbudget_control import ZBudgetControlConfig, read_zbudget_control


@pytest.fixture()
def fixture_control(tmp_path: Path) -> Path:
    """Create a minimal zbudget control file."""
    content = textwrap.dedent("""\
        C ZBudget control test file
        C
             2.29568E-05                   / FACTAROU
             AC                            / UNITAROU
             2.29568E-05                   / FACTVLOU
             AC.FT.                        / UNITVLOU
             100                           / NTIME
             2                             / NZBUDGET
             *                             / TIMBGN
             *                             / TIMEND
        C
        C ZBudget #1 -- all zones, with zone def file
        C
             Zones.dat                     / Zone definition file
             GWZBud.hdf                    / ZBudget HDF5 file
             GWZBud.xlsx                   / Output file
             1MON                          / Print interval
             -1                            / NZPRNT
        C
        C ZBudget #2 -- specific zones, no zone def
        C
                                           / Zone definition file (blank)
             RZZBud.hdf                    / ZBudget HDF5 file
             RZZBud.xlsx                   / Output file
             *                             / Print interval
             2                             / NZPRNT
             1                             / Zone 1
             3                             / Zone 3
    """)
    p = tmp_path / "test_zbudget.in"
    p.write_text(content, encoding="ascii")
    return p


def test_parse_basic_fields(fixture_control: Path) -> None:
    config = read_zbudget_control(fixture_control)

    assert isinstance(config, ZBudgetControlConfig)
    assert config.area_factor == pytest.approx(2.29568e-05)
    assert config.area_unit == "AC"
    assert config.volume_factor == pytest.approx(2.29568e-05)
    assert config.volume_unit == "AC.FT."
    assert config.cache_size == 100


def test_parse_two_zbudgets(fixture_control: Path) -> None:
    config = read_zbudget_control(fixture_control)
    assert len(config.zbudgets) == 2


def test_zbudget_all_zones(fixture_control: Path) -> None:
    config = read_zbudget_control(fixture_control)
    spec = config.zbudgets[0]
    assert spec.zone_def_file is not None
    assert spec.zone_def_file.name == "Zones.dat"
    assert spec.hdf_file.name == "GWZBud.hdf"
    assert spec.output_file.name == "GWZBud.xlsx"
    assert spec.output_interval == "1MON"
    assert spec.zone_ids == [-1]


def test_zbudget_specific_zones(fixture_control: Path) -> None:
    config = read_zbudget_control(fixture_control)
    spec = config.zbudgets[1]
    assert spec.zone_def_file is None  # blank line
    assert spec.hdf_file.name == "RZZBud.hdf"
    assert spec.output_interval is None
    assert spec.zone_ids == [1, 3]


def test_dates_star(fixture_control: Path) -> None:
    config = read_zbudget_control(fixture_control)
    assert config.begin_date is None
    assert config.end_date is None


def test_fixture_file_parses(tmp_path: Path) -> None:
    """Verify the shipped fixture file parses without error."""
    fixture = Path(__file__).parent.parent / "fixtures" / "zbudget_control.in"
    if fixture.exists():
        config = read_zbudget_control(fixture)
        assert len(config.zbudgets) == 1
        assert config.zbudgets[0].zone_ids == [-1]
