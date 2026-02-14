"""Unit tests for io/simulation_writer.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pyiwfm.io.simulation_writer import (
    SimulationMainConfig,
    SimulationMainWriter,
    write_simulation_main,
)


# ---------------------------------------------------------------------------
# SimulationMainConfig
# ---------------------------------------------------------------------------


class TestSimulationMainConfig:
    """Tests for the configuration dataclass."""

    def test_default_values(self, tmp_path: Path) -> None:
        config = SimulationMainConfig(output_dir=tmp_path)
        assert config.main_file == "Simulation_MAIN.IN"
        assert config.begin_date == "09/30/1990_24:00"
        assert config.end_date == "09/30/2000_24:00"
        assert config.time_step == "1DAY"
        assert config.matrix_solver == 2

    def test_main_path(self, tmp_path: Path) -> None:
        config = SimulationMainConfig(output_dir=tmp_path)
        assert config.main_path == tmp_path / "Simulation_MAIN.IN"

    def test_custom_values(self, tmp_path: Path) -> None:
        config = SimulationMainConfig(
            output_dir=tmp_path,
            begin_date="01/01/2020_00:00",
            end_date="12/31/2025_24:00",
            time_step="1MON",
            max_iterations=3000,
        )
        assert config.begin_date == "01/01/2020_00:00"
        assert config.time_step == "1MON"
        assert config.max_iterations == 3000

    def test_custom_title(self, tmp_path: Path) -> None:
        config = SimulationMainConfig(
            output_dir=tmp_path,
            title1="C2VSimFG",
            title2="Version 2.0",
        )
        assert config.title1 == "C2VSimFG"
        assert config.title2 == "Version 2.0"


# ---------------------------------------------------------------------------
# SimulationMainWriter
# ---------------------------------------------------------------------------


class TestSimulationMainWriter:
    """Tests for the writer class."""

    def test_format_property(self, tmp_path: Path) -> None:
        config = SimulationMainConfig(output_dir=tmp_path)
        model = MagicMock()
        writer = SimulationMainWriter(model, config)
        assert writer.format == "iwfm_simulation"

    @patch.object(SimulationMainWriter, "_render_simulation_main")
    def test_write_main_creates_file(self, mock_render, tmp_path: Path) -> None:
        mock_render.return_value = "C SIMULATION MAIN FILE\n"
        config = SimulationMainConfig(output_dir=tmp_path)
        model = MagicMock()
        writer = SimulationMainWriter(model, config)
        path = writer.write_main()
        assert path.exists()
        assert path.read_text() == "C SIMULATION MAIN FILE\n"

    @patch.object(SimulationMainWriter, "_render_simulation_main")
    def test_write_calls_write_main(self, mock_render, tmp_path: Path) -> None:
        mock_render.return_value = "content"
        config = SimulationMainConfig(output_dir=tmp_path)
        model = MagicMock()
        writer = SimulationMainWriter(model, config)
        writer.write()
        assert (tmp_path / "Simulation_MAIN.IN").exists()

    def test_render_uses_template(self, tmp_path: Path) -> None:
        config = SimulationMainConfig(output_dir=tmp_path)
        model = MagicMock()

        mock_engine = MagicMock()
        mock_engine.render_template.return_value = "rendered content"
        writer = SimulationMainWriter(model, config, template_engine=mock_engine)

        result = writer._render_simulation_main()
        mock_engine.render_template.assert_called_once()
        assert result == "rendered content"

    def test_render_passes_config_values(self, tmp_path: Path) -> None:
        config = SimulationMainConfig(
            output_dir=tmp_path,
            begin_date="01/01/2020_00:00",
            time_step="1MON",
        )
        model = MagicMock()

        mock_engine = MagicMock()
        mock_engine.render_template.return_value = ""
        writer = SimulationMainWriter(model, config, template_engine=mock_engine)
        writer._render_simulation_main()

        call_kwargs = mock_engine.render_template.call_args[1]
        assert call_kwargs["begin_date"] == "01/01/2020_00:00"
        assert call_kwargs["time_step"] == "1MON"


# ---------------------------------------------------------------------------
# write_simulation_main convenience function
# ---------------------------------------------------------------------------


class TestWriteSimulationMain:
    """Tests for the convenience function."""

    @patch.object(SimulationMainWriter, "write_main")
    def test_creates_writer_and_calls_write(self, mock_write, tmp_path: Path) -> None:
        mock_write.return_value = tmp_path / "Simulation_MAIN.IN"
        model = MagicMock()
        result = write_simulation_main(model, tmp_path)
        mock_write.assert_called_once()

    @patch.object(SimulationMainWriter, "write_main")
    def test_with_custom_config(self, mock_write, tmp_path: Path) -> None:
        mock_write.return_value = tmp_path / "Simulation_MAIN.IN"
        model = MagicMock()
        config = SimulationMainConfig(
            output_dir=tmp_path,
            time_step="1MON",
        )
        write_simulation_main(model, tmp_path, config=config)
        mock_write.assert_called_once()

    @patch.object(SimulationMainWriter, "write_main")
    def test_default_config(self, mock_write, tmp_path: Path) -> None:
        mock_write.return_value = tmp_path / "Sim.IN"
        model = MagicMock()
        write_simulation_main(model, tmp_path)
        mock_write.assert_called_once()
