"""Unit tests for CLI model loading (_model_loader.py)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pyiwfm.cli._model_loader import _resolve_path, load_model

# ---------------------------------------------------------------------------
# _resolve_path
# ---------------------------------------------------------------------------


class TestResolvePath:
    """Tests for _resolve_path()."""

    def test_relative_path(self, tmp_path: Path) -> None:
        result = _resolve_path(tmp_path, Path("Simulation/Sim.in"))
        assert result == tmp_path / "Simulation" / "Sim.in"

    def test_absolute_path(self, tmp_path: Path) -> None:
        # Use a platform-appropriate absolute path
        abs_path = tmp_path / "other_place" / "Sim.in"
        result = _resolve_path(tmp_path / "base", abs_path)
        assert result == abs_path

    def test_current_dir_relative(self, tmp_path: Path) -> None:
        result = _resolve_path(tmp_path, Path("test.in"))
        assert result == tmp_path / "test.in"


# ---------------------------------------------------------------------------
# load_model
# ---------------------------------------------------------------------------


class TestLoadModel:
    """Tests for load_model()."""

    @patch("pyiwfm.core.model.IWFMModel")
    @patch("pyiwfm.cli._model_finder.find_model_files")
    def test_explicit_simulation_file(self, mock_find, mock_model_cls, tmp_path: Path) -> None:
        mock_model = MagicMock()
        mock_model_cls.from_simulation.return_value = mock_model
        sim_file = Path("Simulation/Sim.in")

        result = load_model(tmp_path, simulation_file=sim_file)

        mock_model_cls.from_simulation.assert_called_once_with(tmp_path / sim_file)
        assert result is mock_model

    @patch("pyiwfm.core.model.IWFMModel")
    @patch("pyiwfm.cli._model_finder.find_model_files")
    def test_explicit_preprocessor_file(self, mock_find, mock_model_cls, tmp_path: Path) -> None:
        mock_model = MagicMock()
        mock_model_cls.from_preprocessor.return_value = mock_model
        pp_file = Path("Preprocessor/PP.in")

        result = load_model(tmp_path, preprocessor_file=pp_file)

        mock_model_cls.from_preprocessor.assert_called_once_with(tmp_path / pp_file)
        assert result is mock_model

    @patch("pyiwfm.core.model.IWFMModel")
    @patch("pyiwfm.cli._model_finder.find_model_files")
    def test_explicit_both_files(self, mock_find, mock_model_cls, tmp_path: Path) -> None:
        mock_model = MagicMock()
        mock_model_cls.from_simulation_with_preprocessor.return_value = mock_model
        sim_file = Path("Sim.in")
        pp_file = Path("PP.in")

        result = load_model(tmp_path, simulation_file=sim_file, preprocessor_file=pp_file)

        mock_model_cls.from_simulation_with_preprocessor.assert_called_once_with(
            tmp_path / sim_file, tmp_path / pp_file
        )
        assert result is mock_model

    @patch("pyiwfm.core.model.IWFMModel")
    @patch("pyiwfm.cli._model_finder.find_model_files")
    def test_auto_detect_sim_and_pp(self, mock_find, mock_model_cls, tmp_path: Path) -> None:
        mock_find.return_value = {
            "simulation_main": tmp_path / "Sim.in",
            "preprocessor_main": tmp_path / "PP.in",
            "preprocessor_binary": None,
        }
        mock_model = MagicMock()
        mock_model_cls.from_simulation_with_preprocessor.return_value = mock_model

        result = load_model(tmp_path)

        mock_model_cls.from_simulation_with_preprocessor.assert_called_once()
        assert result is mock_model

    @patch("pyiwfm.core.model.IWFMModel")
    @patch("pyiwfm.cli._model_finder.find_model_files")
    def test_auto_detect_sim_only(self, mock_find, mock_model_cls, tmp_path: Path) -> None:
        mock_find.return_value = {
            "simulation_main": tmp_path / "Sim.in",
            "preprocessor_main": None,
            "preprocessor_binary": None,
        }
        mock_model = MagicMock()
        mock_model_cls.from_simulation.return_value = mock_model

        result = load_model(tmp_path)

        mock_model_cls.from_simulation.assert_called_once()
        assert result is mock_model

    @patch("pyiwfm.core.model.IWFMModel")
    @patch("pyiwfm.cli._model_finder.find_model_files")
    def test_auto_detect_pp_only(self, mock_find, mock_model_cls, tmp_path: Path) -> None:
        mock_find.return_value = {
            "simulation_main": None,
            "preprocessor_main": tmp_path / "PP.in",
            "preprocessor_binary": None,
        }
        mock_model = MagicMock()
        mock_model_cls.from_preprocessor.return_value = mock_model

        result = load_model(tmp_path)

        mock_model_cls.from_preprocessor.assert_called_once()
        assert result is mock_model

    @patch("pyiwfm.core.model.IWFMModel")
    @patch("pyiwfm.cli._model_finder.find_model_files")
    def test_auto_detect_binary(self, mock_find, mock_model_cls, tmp_path: Path) -> None:
        mock_find.return_value = {
            "simulation_main": None,
            "preprocessor_main": None,
            "preprocessor_binary": tmp_path / "PP.bin",
        }
        mock_model = MagicMock()
        mock_model_cls.from_preprocessor_binary.return_value = mock_model

        result = load_model(tmp_path)

        mock_model_cls.from_preprocessor_binary.assert_called_once()
        assert result is mock_model

    @patch("pyiwfm.core.model.IWFMModel")
    @patch("pyiwfm.cli._model_finder.find_model_files")
    def test_fallback_chain_sim_pp_fails_to_sim_only(
        self, mock_find, mock_model_cls, tmp_path: Path
    ) -> None:
        mock_find.return_value = {
            "simulation_main": tmp_path / "Sim.in",
            "preprocessor_main": tmp_path / "PP.in",
            "preprocessor_binary": None,
        }
        mock_model = MagicMock()
        mock_model_cls.from_simulation_with_preprocessor.side_effect = RuntimeError("fail")
        mock_model_cls.from_simulation.return_value = mock_model

        result = load_model(tmp_path)

        mock_model_cls.from_simulation.assert_called_once()
        assert result is mock_model

    @patch("pyiwfm.core.model.IWFMModel")
    @patch("pyiwfm.cli._model_finder.find_model_files")
    def test_nothing_found_raises(self, mock_find, mock_model_cls, tmp_path: Path) -> None:
        mock_find.return_value = {
            "simulation_main": None,
            "preprocessor_main": None,
            "preprocessor_binary": None,
        }

        with pytest.raises(FileNotFoundError, match="No IWFM model files found"):
            load_model(tmp_path)

    @patch("pyiwfm.core.model.IWFMModel")
    @patch("pyiwfm.cli._model_finder.find_model_files")
    def test_sim_fails_falls_to_pp(self, mock_find, mock_model_cls, tmp_path: Path) -> None:
        mock_find.return_value = {
            "simulation_main": tmp_path / "Sim.in",
            "preprocessor_main": tmp_path / "PP.in",
            "preprocessor_binary": None,
        }
        mock_model = MagicMock()
        mock_model_cls.from_simulation_with_preprocessor.side_effect = RuntimeError("fail")
        mock_model_cls.from_simulation.side_effect = RuntimeError("fail")
        mock_model_cls.from_preprocessor.return_value = mock_model

        result = load_model(tmp_path)

        mock_model_cls.from_preprocessor.assert_called_once()
        assert result is mock_model
