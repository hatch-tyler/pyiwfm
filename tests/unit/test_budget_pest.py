"""Unit tests for budget PEST text export."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from pyiwfm.io.budget_pest import budget_to_pest_instruction, budget_to_pest_text


def _mock_reader(location_names: list[str], dfs: dict[int, pd.DataFrame]) -> MagicMock:
    """Create a mock BudgetReader."""
    reader = MagicMock()
    reader.locations = location_names
    reader.get_dataframe = MagicMock(side_effect=lambda location=0, **kw: dfs[location])
    return reader


class TestBudgetToPestText:
    """Tests for budget_to_pest_text."""

    @patch("pyiwfm.io.budget_pest.BudgetReader")
    def test_basic_export(self, mock_cls: MagicMock) -> None:
        """Test basic text export with one location."""
        df = pd.DataFrame({"Inflow": [100.0, 200.0], "Outflow": [-50.0, -80.0]})
        mock_cls.return_value = _mock_reader(["Loc1"], {0: df})

        with tempfile.TemporaryDirectory() as tmpdir:
            out = budget_to_pest_text(Path(tmpdir) / "fake.hdf", Path(tmpdir) / "obs.txt")
            assert out.exists()
            content = out.read_text()
            assert "gwb00_000000" in content
            assert "gwb00_000001" in content

    @patch("pyiwfm.io.budget_pest.BudgetReader")
    def test_prefix(self, mock_cls: MagicMock) -> None:
        """Test custom prefix."""
        df = pd.DataFrame({"Inflow": [100.0]})
        mock_cls.return_value = _mock_reader(["Loc1"], {0: df})

        with tempfile.TemporaryDirectory() as tmpdir:
            out = budget_to_pest_text(
                Path(tmpdir) / "fake.hdf",
                Path(tmpdir) / "obs.txt",
                prefix="rzb",
            )
            content = out.read_text()
            assert "rzb00_000000" in content

    @patch("pyiwfm.io.budget_pest.BudgetReader")
    def test_location_subset(self, mock_cls: MagicMock) -> None:
        """Test exporting only selected locations."""
        df0 = pd.DataFrame({"Inflow": [100.0]})
        df1 = pd.DataFrame({"Inflow": [200.0]})
        mock_cls.return_value = _mock_reader(["Loc1", "Loc2"], {0: df0, 1: df1})

        with tempfile.TemporaryDirectory() as tmpdir:
            out = budget_to_pest_text(
                Path(tmpdir) / "fake.hdf",
                Path(tmpdir) / "obs.txt",
                locations=[1],
            )
            content = out.read_text()
            assert "gwb01_000000" in content
            assert "gwb00" not in content

    @patch("pyiwfm.io.budget_pest.BudgetReader")
    def test_column_filter(self, mock_cls: MagicMock) -> None:
        """Test exporting only selected columns."""
        df = pd.DataFrame({"Inflow": [100.0], "Outflow": [-50.0]})
        mock_cls.return_value = _mock_reader(["Loc1"], {0: df})

        with tempfile.TemporaryDirectory() as tmpdir:
            out = budget_to_pest_text(
                Path(tmpdir) / "fake.hdf",
                Path(tmpdir) / "obs.txt",
                columns=["Inflow"],
            )
            content = out.read_text()
            assert "Inflow" in content
            assert "Outflow" not in content

    @patch("pyiwfm.io.budget_pest.BudgetReader")
    def test_location_by_name(self, mock_cls: MagicMock) -> None:
        """Test selecting location by name string."""
        df = pd.DataFrame({"Inflow": [100.0]})
        mock_cls.return_value = _mock_reader(["RegionA", "RegionB"], {1: df})

        with tempfile.TemporaryDirectory() as tmpdir:
            out = budget_to_pest_text(
                Path(tmpdir) / "fake.hdf",
                Path(tmpdir) / "obs.txt",
                locations=["RegionB"],
            )
            content = out.read_text()
            assert "gwb01_000000" in content


class TestBudgetToPestInstruction:
    """Tests for budget_to_pest_instruction."""

    @patch("pyiwfm.io.budget_pest.BudgetReader")
    def test_basic_instruction(self, mock_cls: MagicMock) -> None:
        """Test basic instruction file generation."""
        df = pd.DataFrame({"Inflow": [100.0, 200.0]})
        mock_cls.return_value = _mock_reader(["Loc1"], {0: df})

        with tempfile.TemporaryDirectory() as tmpdir:
            out = budget_to_pest_instruction(Path(tmpdir) / "fake.hdf", Path(tmpdir) / "obs.ins")
            assert out.exists()
            content = out.read_text()
            assert content.startswith("pif @")
            assert "!gwb00_000000!" in content
            assert "!gwb00_000001!" in content

    @patch("pyiwfm.io.budget_pest.BudgetReader")
    def test_instruction_with_column_filter(self, mock_cls: MagicMock) -> None:
        """Test instruction file with column filter."""
        df = pd.DataFrame({"Inflow": [100.0], "Outflow": [-50.0]})
        mock_cls.return_value = _mock_reader(["Loc1"], {0: df})

        with tempfile.TemporaryDirectory() as tmpdir:
            out = budget_to_pest_instruction(
                Path(tmpdir) / "fake.hdf",
                Path(tmpdir) / "obs.ins",
                columns=["Inflow"],
            )
            content = out.read_text()
            assert "!gwb00_000000!" in content
