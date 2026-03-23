"""Tests for FortranBackend (ResultsExtract subprocess wrapper).

Covers: generate_input format, _parse_output with edge cases,
subprocess error handling.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pyiwfm.calibration.results_extraction import ExtractionSpec


class TestGenerateInput:
    """Test FortranBackend.generate_input() format correctness."""

    def _make_backend(self, data_type: str = "HEAD") -> object:
        from pyiwfm.calibration.results_extraction import FortranBackend

        return FortranBackend(
            exe_path=Path("ResultsExtract_x64.exe"),
            sim_file=Path("Simulation.in"),
            data_type=data_type,
        )

    def test_basic_format(self) -> None:
        backend = self._make_backend()
        specs = [
            ExtractionSpec(name="W1", x=100.0, y=200.0, layer=1),
            ExtractionSpec(name="W2", x=300.0, y=400.0, layer=0),
        ]
        content, names = backend.generate_input(specs)
        assert names == ["W1", "W2"]
        lines = content.strip().split("\n")
        assert len(lines) == 7  # SIMFILE, DATATYPE, OUTFILE, NHYD, FACTXY, 2 specs
        assert "Simulation.in" in lines[0]
        assert "HEAD" in lines[1]
        assert "2" in lines[3]  # NHYD

    def test_subsidence_data_type(self) -> None:
        backend = self._make_backend("SUBSIDENCE")
        specs = [ExtractionSpec(name="S1", x=50.0, y=60.0, layer=0)]
        content, names = backend.generate_input(specs)
        assert "SUBSIDENCE" in content

    def test_negative_layer_clamped_to_zero(self) -> None:
        """Layer=-1 (T-weighted) should be written as 0 for Fortran."""
        backend = self._make_backend()
        specs = [ExtractionSpec(name="W1", x=100.0, y=200.0, layer=-1)]
        content, _ = backend.generate_input(specs)
        # Find the spec line (last line)
        spec_line = content.strip().split("\n")[-1]
        parts = spec_line.split()
        layer_val = int(parts[2])
        assert layer_val == 0

    def test_factxy_applied(self) -> None:
        backend = self._make_backend()
        specs = [ExtractionSpec(name="W1", x=1.0, y=2.0, layer=1)]
        content, _ = backend.generate_input(specs, factxy=3.28084)
        lines = content.strip().split("\n")
        factxy_line = lines[4]
        assert "3.28084" in factxy_line

    def test_empty_specs(self) -> None:
        backend = self._make_backend()
        content, names = backend.generate_input([])
        assert names == []
        assert "0" in content.split("\n")[3]  # NHYD = 0


class TestParseOutput:
    """Test FortranBackend._parse_output() with synthetic output files."""

    def _make_backend(self, data_type: str = "HEAD") -> object:
        from pyiwfm.calibration.results_extraction import FortranBackend

        return FortranBackend(
            exe_path=Path("dummy"),
            sim_file=Path("dummy"),
            data_type=data_type,
        )

    def test_basic_parse(self, tmp_path: Path) -> None:
        out = tmp_path / "test.out"
        out.write_text(
            "* Header line\n"
            "* Another header\n"
            "01/31/2020_12:00     10.5     20.3\n"
            "02/29/2020_12:00     11.0     21.0\n"
        )
        backend = self._make_backend()
        result = backend._parse_output(out, ["W1", "W2"])
        assert len(result.times) == 2
        assert len(result.names) == 2
        assert_allclose(result.values["W1"], [10.5, 11.0])
        assert_allclose(result.values["W2"], [20.3, 21.0])

    def test_24_00_timestamp(self, tmp_path: Path) -> None:
        """24:00 should be treated as 00:00 (end of day)."""
        out = tmp_path / "test.out"
        out.write_text("01/31/2020_24:00     5.0\n")
        backend = self._make_backend()
        result = backend._parse_output(out, ["W1"])
        assert len(result.times) == 1
        assert_allclose(result.values["W1"], [5.0])

    def test_malformed_values_become_nan(self, tmp_path: Path) -> None:
        out = tmp_path / "test.out"
        out.write_text("01/31/2020_12:00     10.5     BADVAL\n")
        backend = self._make_backend()
        result = backend._parse_output(out, ["W1", "W2"])
        assert_allclose(result.values["W1"], [10.5])
        assert np.isnan(result.values["W2"][0])

    def test_missing_columns_padded_with_nan(self, tmp_path: Path) -> None:
        """If a row has fewer values than expected, pad with NaN."""
        out = tmp_path / "test.out"
        out.write_text("01/31/2020_12:00     10.5\n")
        backend = self._make_backend()
        result = backend._parse_output(out, ["W1", "W2"])
        assert_allclose(result.values["W1"], [10.5])
        assert np.isnan(result.values["W2"][0])

    def test_empty_output_file(self, tmp_path: Path) -> None:
        out = tmp_path / "test.out"
        out.write_text("* Header only\n")
        backend = self._make_backend()
        result = backend._parse_output(out, ["W1"])
        assert len(result.times) == 0

    def test_incremental_flag_preserved(self, tmp_path: Path) -> None:
        out = tmp_path / "test.out"
        out.write_text("01/31/2020_12:00     1.0\n")
        backend = self._make_backend("SUBSIDENCE")
        backend._incremental = True
        result = backend._parse_output(out, ["S1"])
        assert result.incremental is True


class TestRunErrorHandling:
    """Test FortranBackend.run() error handling (mocked subprocess)."""

    def test_missing_executable_raises(self) -> None:
        from pyiwfm.calibration.results_extraction import FortranBackend

        backend = FortranBackend(
            exe_path=Path("/nonexistent/ResultsExtract_x64.exe"),
            sim_file=Path("dummy"),
        )
        specs = [ExtractionSpec(name="W1", x=1.0, y=2.0, layer=1)]
        with pytest.raises(FileNotFoundError, match="not found"):
            backend.run(specs)

    @patch("subprocess.run")
    def test_nonzero_returncode_raises(self, mock_run: MagicMock, tmp_path: Path) -> None:
        from pyiwfm.calibration.results_extraction import FortranBackend

        exe = tmp_path / "ResultsExtract_x64.exe"
        exe.write_text("dummy")

        mock_run.return_value = MagicMock(returncode=1, stderr="Segfault", stdout="")

        backend = FortranBackend(
            exe_path=exe,
            sim_file=Path("dummy"),
        )
        specs = [ExtractionSpec(name="W1", x=1.0, y=2.0, layer=1)]
        with pytest.raises(RuntimeError, match="exited with code 1"):
            backend.run(specs, work_dir=tmp_path)

    @patch("subprocess.run")
    def test_timeout_raises(self, mock_run: MagicMock, tmp_path: Path) -> None:
        import subprocess as sp

        from pyiwfm.calibration.results_extraction import FortranBackend

        exe = tmp_path / "ResultsExtract_x64.exe"
        exe.write_text("dummy")

        mock_run.side_effect = sp.TimeoutExpired(cmd="test", timeout=300)

        backend = FortranBackend(
            exe_path=exe,
            sim_file=Path("dummy"),
        )
        specs = [ExtractionSpec(name="W1", x=1.0, y=2.0, layer=1)]
        with pytest.raises(RuntimeError, match="timed out"):
            backend.run(specs, work_dir=tmp_path)
