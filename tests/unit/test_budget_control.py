"""Tests for pyiwfm.io.budget_control -- budget control file parser."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from pyiwfm.io.budget_control import BudgetControlConfig, read_budget_control


@pytest.fixture()
def fixture_control(tmp_path: Path) -> Path:
    """Create a minimal budget control file."""
    content = textwrap.dedent("""\
        C Budget control test file
        C
             2.29568E-05                   / FACTLTOU
             FEET                          / UNITLTOU
             2.29568E-05                   / FACTAROU
             AC                            / UNITAROU
             2.29568E-05                   / FACTVLOU
             AC.FT.                        / UNITVLOU
             100                           / NTIME
             2                             / NBUDGET
             *                             / TIMBGN
             *                             / TIMEND
        C
        C Budget #1 -- all locations
        C
             GW_Budget.hdf                 / Budget HDF5 file
             GW_Budget.xlsx                / Output file
             1MON                          / Print interval
             -1                            / NLPRNT
        C
        C Budget #2 -- specific locations
        C
             Stream_Budget.hdf             / Budget HDF5 file
             Stream_Budget.xlsx            / Output file
             *                             / Print interval (same as data)
             3                             / NLPRNT
             1                             / Location 1
             5                             / Location 5
             10                            / Location 10
    """)
    p = tmp_path / "test_budget.in"
    p.write_text(content, encoding="ascii")
    return p


def test_parse_basic_fields(fixture_control: Path) -> None:
    config = read_budget_control(fixture_control)

    assert isinstance(config, BudgetControlConfig)
    assert config.length_factor == pytest.approx(2.29568e-05)
    assert config.length_unit == "FEET"
    assert config.area_factor == pytest.approx(2.29568e-05)
    assert config.area_unit == "AC"
    assert config.volume_factor == pytest.approx(2.29568e-05)
    assert config.volume_unit == "AC.FT."
    assert config.cache_size == 100


def test_parse_dates_star(fixture_control: Path) -> None:
    config = read_budget_control(fixture_control)
    assert config.begin_date is None
    assert config.end_date is None


def test_parse_two_budgets(fixture_control: Path) -> None:
    config = read_budget_control(fixture_control)
    assert len(config.budgets) == 2


def test_budget_all_locations(fixture_control: Path) -> None:
    config = read_budget_control(fixture_control)
    spec = config.budgets[0]
    assert spec.hdf_file.name == "GW_Budget.hdf"
    assert spec.output_file.name == "GW_Budget.xlsx"
    assert spec.output_interval == "1MON"
    assert spec.location_ids == [-1]  # All locations


def test_budget_specific_locations(fixture_control: Path) -> None:
    config = read_budget_control(fixture_control)
    spec = config.budgets[1]
    assert spec.hdf_file.name == "Stream_Budget.hdf"
    assert spec.output_file.name == "Stream_Budget.xlsx"
    assert spec.output_interval is None  # * → None
    assert spec.location_ids == [1, 5, 10]


def test_parse_dates_explicit(tmp_path: Path) -> None:
    content = textwrap.dedent("""\
             1.0                           / FACTLTOU
             FT                            / UNITLTOU
             1.0                           / FACTAROU
             AC                            / UNITAROU
             1.0                           / FACTVLOU
             AF                            / UNITVLOU
             50                            / NTIME
             1                             / NBUDGET
             10/01/1921_24:00              / TIMBGN
             09/30/2015_24:00              / TIMEND
        C
             data.hdf                      / Budget HDF5 file
             output.xlsx                   / Output file
             *                             / Print interval
             0                             / NLPRNT (none)
    """)
    p = tmp_path / "dates.in"
    p.write_text(content, encoding="ascii")

    config = read_budget_control(p)
    assert config.begin_date == "10/01/1921_24:00"
    assert config.end_date == "09/30/2015_24:00"
    assert config.budgets[0].location_ids == []  # NLPRNT=0


def test_fixture_file_parses(tmp_path: Path) -> None:
    """Verify the shipped fixture file parses without error."""
    fixture = Path(__file__).parent.parent / "fixtures" / "budget_control.in"
    if fixture.exists():
        config = read_budget_control(fixture)
        assert len(config.budgets) == 1
        assert config.budgets[0].location_ids == [-1]
