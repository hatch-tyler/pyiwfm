"""Unit tests for CLI model file discovery (_model_finder.py)."""

from __future__ import annotations

from pathlib import Path

from pyiwfm.cli._model_finder import (
    extract_model_name,
    find_binary_file,
    find_model_files,
    find_preprocessor_file,
    find_simulation_file,
)

# ---------------------------------------------------------------------------
# find_preprocessor_file
# ---------------------------------------------------------------------------


class TestFindPreprocessorFile:
    """Tests for find_preprocessor_file()."""

    def test_known_candidate_match(self, tmp_path: Path) -> None:
        pp_dir = tmp_path / "Preprocessor"
        pp_dir.mkdir()
        pp_file = pp_dir / "C2VSimFG_Preprocessor.in"
        pp_file.touch()
        result = find_preprocessor_file(tmp_path)
        assert result == pp_file

    def test_known_candidate_generic(self, tmp_path: Path) -> None:
        pp_dir = tmp_path / "Preprocessor"
        pp_dir.mkdir()
        pp_file = pp_dir / "PreProcessor_MAIN.IN"
        pp_file.touch()
        result = find_preprocessor_file(tmp_path)
        assert result == pp_file

    def test_candidate_at_root(self, tmp_path: Path) -> None:
        pp_file = tmp_path / "PreProcessor_MAIN.IN"
        pp_file.touch()
        result = find_preprocessor_file(tmp_path)
        assert result == pp_file

    def test_glob_fallback(self, tmp_path: Path) -> None:
        pp_dir = tmp_path / "Preprocessor"
        pp_dir.mkdir()
        pp_file = pp_dir / "custom_model.in"
        pp_file.touch()
        result = find_preprocessor_file(tmp_path)
        assert result == pp_file

    def test_glob_with_main_priority(self, tmp_path: Path) -> None:
        pp_dir = tmp_path / "Preprocessor"
        pp_dir.mkdir()
        (pp_dir / "other.in").touch()
        main_file = pp_dir / "custom_main.in"
        main_file.touch()
        result = find_preprocessor_file(tmp_path)
        assert result == main_file

    def test_no_match_returns_none(self, tmp_path: Path) -> None:
        result = find_preprocessor_file(tmp_path)
        assert result is None

    def test_empty_directory(self, tmp_path: Path) -> None:
        result = find_preprocessor_file(tmp_path)
        assert result is None

    def test_preprocessor_dir_case_variation(self, tmp_path: Path) -> None:
        pp_dir = tmp_path / "PreProcessor"
        pp_dir.mkdir()
        pp_file = pp_dir / "PreProcessor_MAIN.IN"
        pp_file.touch()
        result = find_preprocessor_file(tmp_path)
        assert result == pp_file


# ---------------------------------------------------------------------------
# find_simulation_file
# ---------------------------------------------------------------------------


class TestFindSimulationFile:
    """Tests for find_simulation_file()."""

    def test_known_candidate_match(self, tmp_path: Path) -> None:
        sim_dir = tmp_path / "Simulation"
        sim_dir.mkdir()
        sim_file = sim_dir / "C2VSimFG.in"
        sim_file.touch()
        result = find_simulation_file(tmp_path)
        assert result == sim_file

    def test_known_candidate_generic(self, tmp_path: Path) -> None:
        sim_dir = tmp_path / "Simulation"
        sim_dir.mkdir()
        sim_file = sim_dir / "Simulation_MAIN.IN"
        sim_file.touch()
        result = find_simulation_file(tmp_path)
        assert result == sim_file

    def test_candidate_at_root(self, tmp_path: Path) -> None:
        sim_file = tmp_path / "Simulation_MAIN.IN"
        sim_file.touch()
        result = find_simulation_file(tmp_path)
        assert result == sim_file

    def test_glob_fallback(self, tmp_path: Path) -> None:
        sim_dir = tmp_path / "Simulation"
        sim_dir.mkdir()
        sim_file = sim_dir / "custom_sim.in"
        sim_file.touch()
        result = find_simulation_file(tmp_path)
        assert result == sim_file

    def test_glob_with_main_priority(self, tmp_path: Path) -> None:
        sim_dir = tmp_path / "Simulation"
        sim_dir.mkdir()
        (sim_dir / "other.in").touch()
        main_file = sim_dir / "custom_main.in"
        main_file.touch()
        result = find_simulation_file(tmp_path)
        assert result == main_file

    def test_no_match_returns_none(self, tmp_path: Path) -> None:
        result = find_simulation_file(tmp_path)
        assert result is None


# ---------------------------------------------------------------------------
# find_binary_file
# ---------------------------------------------------------------------------


class TestFindBinaryFile:
    """Tests for find_binary_file()."""

    def test_candidate_match(self, tmp_path: Path) -> None:
        sim_dir = tmp_path / "Simulation"
        sim_dir.mkdir()
        bin_file = sim_dir / "C2VSimFG_Preprocessor.bin"
        bin_file.touch()
        result = find_binary_file(tmp_path)
        assert result == bin_file

    def test_no_match(self, tmp_path: Path) -> None:
        result = find_binary_file(tmp_path)
        assert result is None

    def test_preprocessor_dir(self, tmp_path: Path) -> None:
        pp_dir = tmp_path / "Preprocessor"
        pp_dir.mkdir()
        bin_file = pp_dir / "PreProcessor.bin"
        bin_file.touch()
        result = find_binary_file(tmp_path)
        assert result == bin_file

    def test_root_level(self, tmp_path: Path) -> None:
        bin_file = tmp_path / "PreProcessor.bin"
        bin_file.touch()
        result = find_binary_file(tmp_path)
        assert result == bin_file


# ---------------------------------------------------------------------------
# find_model_files
# ---------------------------------------------------------------------------


class TestFindModelFiles:
    """Tests for find_model_files()."""

    def test_returns_all_keys(self, tmp_path: Path) -> None:
        result = find_model_files(tmp_path)
        assert "preprocessor_main" in result
        assert "simulation_main" in result
        assert "preprocessor_binary" in result

    def test_all_found(self, tmp_path: Path) -> None:
        sim_dir = tmp_path / "Simulation"
        sim_dir.mkdir()
        pp_dir = tmp_path / "Preprocessor"
        pp_dir.mkdir()
        (sim_dir / "C2VSimFG.in").touch()
        (pp_dir / "C2VSimFG_Preprocessor.in").touch()
        (sim_dir / "C2VSimFG_Preprocessor.bin").touch()

        result = find_model_files(tmp_path)
        assert result["preprocessor_main"] is not None
        assert result["simulation_main"] is not None
        assert result["preprocessor_binary"] is not None

    def test_all_none(self, tmp_path: Path) -> None:
        result = find_model_files(tmp_path)
        assert result["preprocessor_main"] is None
        assert result["simulation_main"] is None
        assert result["preprocessor_binary"] is None


# ---------------------------------------------------------------------------
# extract_model_name
# ---------------------------------------------------------------------------


class TestExtractModelName:
    """Tests for extract_model_name()."""

    def test_tag_c2vsim(self) -> None:
        """Path parts containing 'c2vsim' are detected."""
        path = Path("C2VSimFG_v2") / "Simulation" / "C2VSimFG.in"
        result = extract_model_name(path)
        assert "C2VSimFG" in result

    def test_tag_cvhm(self) -> None:
        path = Path("CVHM_Model") / "Simulation" / "Sim.in"
        result = extract_model_name(path)
        assert "CVHM" in result

    def test_tag_iwfm(self) -> None:
        path = Path("my_iwfm_run") / "Simulation" / "Sim.in"
        result = extract_model_name(path)
        assert "iwfm" in result.lower()

    def test_file_input(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "models" / "test_run"
        model_dir.mkdir(parents=True)
        filepath = model_dir / "Sim.in"
        filepath.touch()
        result = extract_model_name(filepath)
        assert result == "models"

    def test_directory_input(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "my_project"
        model_dir.mkdir()
        result = extract_model_name(model_dir)
        assert result == "my_project"

    def test_model_keyword_in_path(self, tmp_path: Path) -> None:
        d = tmp_path / "my_model_data" / "Sim"
        d.mkdir(parents=True)
        path = d / "run.in"
        result = extract_model_name(path)
        assert "model" in result.lower()
