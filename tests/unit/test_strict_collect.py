"""Tests for the ``strict="collect"`` loader mode and
:attr:`IWFMModel.load_errors` / :attr:`has_load_errors` introspection
properties (Fix 8 of the exception-handling audit).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from pyiwfm.core.exceptions import (
    ComponentLoadError,
    IWFMIOError,
    ValidationError,
)
from pyiwfm.core.loaders._common import (
    _LOAD_ERRORS_KEY,
    _finalize_collected_errors,
    _record_component_failure,
)
from pyiwfm.core.model import IWFMModel

# ---------------------------------------------------------------------------
# IWFMModel.load_errors / has_load_errors
# ---------------------------------------------------------------------------


class TestLoadErrorsProperties:
    def test_clean_model_has_no_load_errors(self) -> None:
        model = IWFMModel(name="t")
        assert model.load_errors == []
        assert model.has_load_errors is False

    def test_load_errors_returns_typed_list(self) -> None:
        """``_record_component_failure`` populates a typed list that the
        property exposes; backward-compatible scalar key stays too."""
        model = IWFMModel(name="t")
        _record_component_failure(
            model,
            "streams",
            Path("/tmp/streams.dat"),
            ValueError("bad row"),
            strict=False,
        )

        # New typed path
        assert model.has_load_errors is True
        errors = model.load_errors
        assert len(errors) == 1
        assert isinstance(errors[0], ComponentLoadError)
        assert errors[0].component_name == "streams"
        assert errors[0].source_file == Path("/tmp/streams.dat")
        # Backward-compat scalar key
        assert "streams_load_error" in model.metadata

    def test_load_errors_returns_copy(self) -> None:
        """The property returns a copy so callers can't corrupt metadata."""
        model = IWFMModel(name="t")
        _record_component_failure(model, "lakes", None, ValueError("oops"), strict=False)
        snapshot = model.load_errors
        snapshot.clear()  # mutate the returned list
        # Internal state survives
        assert model.has_load_errors is True
        assert len(model.load_errors) == 1

    def test_multiple_failures_accumulate(self) -> None:
        model = IWFMModel(name="t")
        _record_component_failure(model, "streams", None, ValueError("a"), strict=False)
        _record_component_failure(model, "lakes", None, ValueError("b"), strict=False)
        assert len(model.load_errors) == 2
        names = [e.component_name for e in model.load_errors]
        assert names == ["streams", "lakes"]


# ---------------------------------------------------------------------------
# strict=True (fail-fast)
# ---------------------------------------------------------------------------


class TestStrictTrue:
    def test_strict_true_raises_immediately(self) -> None:
        model = IWFMModel(name="t")
        with pytest.raises(ComponentLoadError) as excinfo:
            _record_component_failure(
                model, "streams", Path("/tmp/s.dat"), ValueError("bad"), strict=True
            )
        assert excinfo.value.component_name == "streams"
        assert excinfo.value.source_file == Path("/tmp/s.dat")


# ---------------------------------------------------------------------------
# strict="collect"
# ---------------------------------------------------------------------------


class TestStrictCollect:
    def test_collect_records_without_raising_during_load(self) -> None:
        """``_record_component_failure`` with ``strict="collect"`` does not
        raise; the loader's end-of-load finalize call handles the raise."""
        model = IWFMModel(name="t")
        _record_component_failure(model, "streams", None, ValueError("a"), strict="collect")
        _record_component_failure(model, "lakes", None, IWFMIOError("b"), strict="collect")
        assert len(model.load_errors) == 2  # recorded but no raise

    def test_finalize_raises_validation_error_with_full_list(self) -> None:
        model = IWFMModel(name="t")
        _record_component_failure(
            model, "streams", None, ValueError("bad row 17"), strict="collect"
        )
        _record_component_failure(
            model, "lakes", None, IWFMIOError("missing keyword"), strict="collect"
        )
        with pytest.raises(ValidationError) as excinfo:
            _finalize_collected_errors(model, strict="collect")
        assert "2 component(s) failed to load" in str(excinfo.value)
        # Each error message appears in the typed errors list
        joined = " | ".join(excinfo.value.errors)
        assert "streams" in joined
        assert "lakes" in joined

    def test_finalize_no_raise_when_clean(self) -> None:
        model = IWFMModel(name="t")
        # Should be a no-op
        _finalize_collected_errors(model, strict="collect")
        assert model.has_load_errors is False

    def test_finalize_no_raise_when_strict_false(self) -> None:
        """With ``strict=False`` the finalize is a no-op even if errors
        were recorded — that's the partial-load mode."""
        model = IWFMModel(name="t")
        _record_component_failure(model, "streams", None, ValueError("x"), strict=False)
        _finalize_collected_errors(model, strict=False)  # no-op
        assert model.has_load_errors is True  # still recorded for introspection

    def test_finalize_no_raise_when_strict_true(self) -> None:
        """``strict=True`` would have raised in ``_record_component_failure``
        already; finalize is a no-op."""
        model = IWFMModel(name="t")
        _finalize_collected_errors(model, strict=True)  # no-op


# ---------------------------------------------------------------------------
# CLI integration: load_model honors allow_partial_load
# ---------------------------------------------------------------------------


class TestCliLoadModel:
    """Verify ``cli/_model_loader.load_model`` passes ``strict="collect"``
    by default and ``strict=False`` when ``allow_partial_load=True``."""

    def test_load_model_default_passes_strict_collect(self, tmp_path: Path) -> None:
        from pyiwfm.cli._model_loader import load_model

        with patch("pyiwfm.core.model.IWFMModel.from_simulation_with_preprocessor") as mock_load:
            mock_load.return_value = IWFMModel(name="t")
            sim = tmp_path / "Simulation.in"
            sim.write_text("")
            pp = tmp_path / "Preprocessor.in"
            pp.write_text("")
            load_model(tmp_path, preprocessor_file=pp, simulation_file=sim)

        kwargs = mock_load.call_args.kwargs
        assert kwargs.get("strict") == "collect"

    def test_load_model_allow_partial_passes_strict_false(self, tmp_path: Path) -> None:
        from pyiwfm.cli._model_loader import load_model

        with patch("pyiwfm.core.model.IWFMModel.from_simulation_with_preprocessor") as mock_load:
            mock_load.return_value = IWFMModel(name="t")
            sim = tmp_path / "Simulation.in"
            sim.write_text("")
            pp = tmp_path / "Preprocessor.in"
            pp.write_text("")
            load_model(
                tmp_path,
                preprocessor_file=pp,
                simulation_file=sim,
                allow_partial_load=True,
            )

        kwargs = mock_load.call_args.kwargs
        assert kwargs.get("strict") is False

    def test_warn_if_partial_load_silent_when_clean(self, capsys) -> None:
        from pyiwfm.cli._model_loader import warn_if_partial_load

        model = IWFMModel(name="t")
        warn_if_partial_load(model)
        captured = capsys.readouterr()
        assert captured.err == ""

    def test_warn_if_partial_load_prints_when_errors(self, capsys) -> None:
        from pyiwfm.cli._model_loader import warn_if_partial_load

        model = IWFMModel(name="t")
        _record_component_failure(model, "streams", None, ValueError("a"), strict=False)
        _record_component_failure(model, "lakes", None, ValueError("b"), strict=False)
        warn_if_partial_load(model)
        captured = capsys.readouterr()
        assert "2 component error" in captured.err
        assert "--allow-partial-load" in captured.err


# ---------------------------------------------------------------------------
# Integration: backward compat key still set
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    def test_legacy_metadata_scalar_key_still_set(self) -> None:
        """External code may still read ``metadata['{component}_load_error']``;
        the new typed list lives alongside it under ``__load_errors__``."""
        model = IWFMModel(name="t")
        _record_component_failure(model, "rootzone", None, ValueError("crop missing"), strict=False)
        assert "rootzone_load_error" in model.metadata
        assert "crop missing" in model.metadata["rootzone_load_error"]
        # And the typed list
        assert _LOAD_ERRORS_KEY in model.metadata
        assert isinstance(model.metadata[_LOAD_ERRORS_KEY], list)
