"""Tests for the structured error handling added to ``IWFMModel.from_preprocessor``.

Covers:

- ``ComponentLoadError`` carries the component name, source path, and cause
- ``_record_component_failure`` logs and records the error in metadata under
  ``strict=False`` (default)
- ``_record_component_failure`` raises ``ComponentLoadError`` chained from the
  cause under ``strict=True``
- The set of expected exception types caught at component-load boundaries
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pyiwfm.core.exceptions import ComponentLoadError
from pyiwfm.core.model import (
    _COMPONENT_LOAD_EXCEPTIONS,
    _record_component_failure,
)


class TestComponentLoadError:
    def test_message_includes_name_file_and_cause(self):
        cause = ValueError("bad token at line 12")
        err = ComponentLoadError("streams", Path("model/Stream.dat"), cause)
        msg = str(err)
        assert "streams" in msg
        assert "Stream.dat" in msg
        assert "ValueError" in msg
        assert "bad token at line 12" in msg

    def test_attributes_round_trip(self):
        path = Path("model/Lake.dat")
        cause = OSError(2, "No such file")
        err = ComponentLoadError("lakes", path, cause)
        assert err.component_name == "lakes"
        assert err.source_file == path

    def test_cause_chain_via_raise_from(self):
        try:
            try:
                raise ValueError("inner")
            except ValueError as e:
                raise ComponentLoadError("streams", None, e) from e
        except ComponentLoadError as outer:
            assert isinstance(outer.__cause__, ValueError)
            assert str(outer.__cause__) == "inner"


class TestRecordComponentFailure:
    def _model(self) -> MagicMock:
        m = MagicMock()
        m.metadata = {}
        return m

    def test_lenient_mode_logs_and_records_metadata(self, caplog):
        model = self._model()
        cause = ValueError("malformed")
        with caplog.at_level(logging.WARNING):
            _record_component_failure(model, "streams", Path("Stream.dat"), cause, strict=False)
        # Metadata key follows existing convention: "<name>_load_error"
        assert "streams_load_error" in model.metadata
        assert "ValueError" in model.metadata["streams_load_error"]
        assert "malformed" in model.metadata["streams_load_error"]
        # Warning was logged with structured fields
        assert any("streams" in r.message for r in caplog.records)
        # exc_info=True means the traceback got attached
        assert any(r.exc_info is not None for r in caplog.records)

    def test_strict_mode_raises_component_load_error(self):
        model = self._model()
        cause = OSError(2, "Stream.dat not found")
        with pytest.raises(ComponentLoadError) as exc_info:
            _record_component_failure(model, "streams", Path("Stream.dat"), cause, strict=True)
        err = exc_info.value
        assert err.component_name == "streams"
        assert err.source_file == Path("Stream.dat")
        # Original exception is chained
        assert isinstance(err.__cause__, OSError)

    def test_strict_mode_still_records_metadata_before_raising(self):
        # Even when raising, the metadata is populated so the caller can see
        # what failed if they catch ComponentLoadError.
        model = self._model()
        try:
            _record_component_failure(
                model, "lakes", Path("Lake.dat"), KeyError("missing key"), strict=True
            )
        except ComponentLoadError:
            pass
        assert "lakes_load_error" in model.metadata


class TestExpectedExceptionTuple:
    """The catch-tuple should cover real parser failures but not programmer
    errors (TypeError, AttributeError, NameError) — those should bubble up."""

    def test_includes_oserror_valueerror_keyerror(self):
        for cls in (OSError, ValueError, KeyError, IndexError, UnicodeDecodeError):
            assert cls in _COMPONENT_LOAD_EXCEPTIONS or any(
                issubclass(cls, exc) for exc in _COMPONENT_LOAD_EXCEPTIONS
            )

    def test_excludes_programmer_errors(self):
        # These should NOT be caught — they indicate bugs in pyiwfm itself
        for cls in (TypeError, AttributeError, NameError, RuntimeError):
            assert cls not in _COMPONENT_LOAD_EXCEPTIONS
            # And not a subclass of any caught exception
            assert not any(issubclass(cls, exc) for exc in _COMPONENT_LOAD_EXCEPTIONS)

    def test_filenotfounderror_is_caught_via_oserror(self):
        # FileNotFoundError is the most common real failure; verify it gets
        # caught (it inherits from OSError).
        assert issubclass(FileNotFoundError, OSError)
        assert OSError in _COMPONENT_LOAD_EXCEPTIONS

    def test_importerror_is_caught(self):
        # pyiwfm has many optional dependencies (triangle, gmsh, dss, vtk,
        # etc.) — a missing optional dep raised through a lazy import
        # should be a graceful "feature unavailable" failure, not a crash.
        assert ImportError in _COMPONENT_LOAD_EXCEPTIONS or any(
            issubclass(ImportError, exc) for exc in _COMPONENT_LOAD_EXCEPTIONS
        )


class TestFromSimulationWithPreprocessorStrict:
    """End-to-end strict mode on the big classmethod.

    The Phase-2 follow-up migrated 35 bare ``except Exception`` sites in
    ``from_simulation_with_preprocessor`` to the same
    ``_record_component_failure`` + ``strict`` contract used in
    ``from_preprocessor``. These tests prove the contract holds at the
    integration level — when ``strict=True``, a component-load failure
    raises :class:`ComponentLoadError`; when ``strict=False`` (default),
    the failure is recorded in metadata and loading continues.
    """

    def _setup_files(self, tmp_path: Path) -> tuple[Path, Path]:
        """Create stub simulation + preprocessor files. The mocks below
        bypass actual parsing, but the files have to exist."""
        sim_dir = tmp_path / "Simulation"
        pp_dir = tmp_path / "Preprocessor"
        sim_dir.mkdir()
        pp_dir.mkdir()
        sim_file = sim_dir / "Simulation.in"
        pp_file = pp_dir / "Preprocessor.in"
        sim_file.write_text("fake")
        pp_file.write_text("fake")
        return sim_file, pp_file

    def _mock_pp_model(self) -> MagicMock:
        from unittest.mock import MagicMock

        from pyiwfm.core.model import IWFMModel

        m = MagicMock(spec=IWFMModel)
        m.metadata = {}
        m.source_files = {}
        m.streams = None
        m.lakes = None
        m.rootzone = None
        m.small_watersheds = None
        m.unsaturated_zone = None
        m.groundwater = None
        m.supply_adjustment = None
        m.mesh = MagicMock()
        m.mesh.n_nodes = 1
        m.stratigraphy = MagicMock()
        return m

    def _mock_sim_config_with_unsat_zone(self, tmp_path: Path) -> MagicMock:
        """SimulationConfig with one unsaturated-zone file and nothing else.

        We pick unsat_zone because its outer try/except is the simplest in
        the classmethod (no inner fallback) — patching its reader to raise
        directly triggers the outer ``_record_component_failure`` path.
        """
        from datetime import datetime
        from unittest.mock import MagicMock

        from pyiwfm.io.simulation import TimeUnit

        cfg = MagicMock()
        cfg.groundwater_file = None
        cfg.streams_file = None
        cfg.lakes_file = None
        cfg.rootzone_file = None
        cfg.small_watershed_file = None
        cfg.unsaturated_zone_file = "uz.dat"
        cfg.supply_adjust_file = None
        cfg.precipitation_file = None
        cfg.et_file = None
        cfg.binary_preprocessor_file = None
        cfg.irrigation_fractions_file = None
        cfg.title_lines = []
        cfg.start_date = datetime(2024, 1, 1)
        cfg.end_date = datetime(2024, 12, 31)
        cfg.time_step_length = 1
        cfg.time_step_unit = TimeUnit.DAY
        cfg.matrix_solver = 2
        cfg.relaxation = 1.0
        cfg.max_iterations = 50
        cfg.max_supply_iterations = 50
        cfg.convergence_tolerance = 1e-6
        cfg.convergence_volume = 0.0
        cfg.convergence_supply = 1e-3
        cfg.supply_adjust_option = 0
        cfg.debug_flag = 0
        cfg.cache_size = 500000
        # Create the UZ file so .exists() is True
        (tmp_path / "Simulation" / "uz.dat").write_text("fake")
        return cfg

    def test_strict_false_records_metadata(self, tmp_path: Path):
        """Default lenient mode: failure recorded in metadata, no raise."""
        from unittest.mock import patch

        from pyiwfm.core.model import IWFMModel

        sim_file, pp_file = self._setup_files(tmp_path)
        sim_cfg = self._mock_sim_config_with_unsat_zone(tmp_path)
        pp_model = self._mock_pp_model()

        with (
            patch.object(IWFMModel, "from_preprocessor", return_value=pp_model),
            patch("pyiwfm.io.simulation.SimulationReader.read", return_value=sim_cfg),
            patch(
                "pyiwfm.io.preprocessor._resolve_path",
                side_effect=lambda base, p: Path(base) / p,
            ),
            patch(
                "pyiwfm.io.unsaturated_zone.UnsatZoneMainReader.read",
                side_effect=ValueError("simulated UZ main parse failure"),
            ),
        ):
            model = IWFMModel.from_simulation_with_preprocessor(sim_file, pp_file)

        assert "unsat_zone_load_error" in model.metadata
        assert "ValueError" in model.metadata["unsat_zone_load_error"]
        assert "simulated UZ main parse failure" in model.metadata["unsat_zone_load_error"]

    def test_strict_true_raises_component_load_error(self, tmp_path: Path):
        """Strict mode: failure raises ComponentLoadError chained from cause."""
        from unittest.mock import patch

        from pyiwfm.core.model import IWFMModel

        sim_file, pp_file = self._setup_files(tmp_path)
        sim_cfg = self._mock_sim_config_with_unsat_zone(tmp_path)
        pp_model = self._mock_pp_model()

        with (
            patch.object(IWFMModel, "from_preprocessor", return_value=pp_model),
            patch("pyiwfm.io.simulation.SimulationReader.read", return_value=sim_cfg),
            patch(
                "pyiwfm.io.preprocessor._resolve_path",
                side_effect=lambda base, p: Path(base) / p,
            ),
            patch(
                "pyiwfm.io.unsaturated_zone.UnsatZoneMainReader.read",
                side_effect=ValueError("simulated UZ main parse failure"),
            ),
            pytest.raises(ComponentLoadError) as exc_info,
        ):
            IWFMModel.from_simulation_with_preprocessor(sim_file, pp_file, strict=True)

        err = exc_info.value
        assert err.component_name == "unsat_zone"
        assert isinstance(err.__cause__, ValueError)
        assert "simulated UZ main parse failure" in str(err.__cause__)
