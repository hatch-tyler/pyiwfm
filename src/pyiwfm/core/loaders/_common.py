"""Shared helpers for the ``IWFMModel`` loader functions.

These were defined at module scope in v1.x's ``core/model.py``. v2.0
relocates them here so the loader functions in
``pyiwfm.core.loaders.from_*`` can import them without depending on
``core/model.py`` (which itself uses them only for the classmethod
dispatchers).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

from pyiwfm.core.exceptions import (
    ComponentError,
    ComponentLoadError,
    IWFMIOError,
    ValidationError,
)

if TYPE_CHECKING:
    from pathlib import Path

    from pyiwfm.core.model import IWFMModel

logger = logging.getLogger(__name__)

# Type alias for the strict-loading behaviour. Three values:
#
# - ``False``: log + record + continue. The model is returned partially
#   loaded; introspect via :attr:`IWFMModel.load_errors`.
# - ``True``: raise :class:`ComponentLoadError` on the first failure.
# - ``"collect"``: load every component, then raise a single
#   :class:`ValidationError` at the end if any failed (best UX: users
#   see all problems at once and don't have to re-run after each fix).
StrictMode = bool | Literal["collect"]

# Key under which load failures are accumulated on
# ``IWFMModel.metadata`` for introspection.
_LOAD_ERRORS_KEY = "__load_errors__"

# Exceptions we expect during component-file parsing. Anything outside this
# tuple is a programmer error (TypeError, AttributeError, NameError, etc.)
# and should bubble up rather than be swallowed.
#
# Includes:
#   - Python builtins for I/O / parsing failures (OSError covers
#     FileNotFoundError + permission errors; ValueError covers bad numeric
#     conversions; KeyError / IndexError cover malformed dicts/arrays).
#   - ImportError: pyiwfm has many optional dependencies (triangle, gmsh,
#     dss, vtk, etc.) and component readers may import them lazily. A
#     missing optional dep is a legitimate "feature unavailable" condition,
#     not a programmer error.
#   - pyiwfm's own IWFMIOError (parent of FileFormatError) and
#     ComponentError, raised by domain readers for format-specific issues.
_COMPONENT_LOAD_EXCEPTIONS: tuple[type[BaseException], ...] = (
    OSError,
    ValueError,
    KeyError,
    IndexError,
    UnicodeDecodeError,
    ImportError,
    IWFMIOError,
    ComponentError,
)


def _record_component_failure(
    model: IWFMModel,
    component_name: str,
    source_file: Path | None,
    exc: BaseException,
    *,
    strict: StrictMode,
) -> None:
    """Common handling for a failed component load.

    With ``strict=False`` (default): logs a structured warning with full
    traceback and stores the error in ``model.metadata`` for backward
    compatibility. The model retains whatever components loaded
    successfully. Callers can introspect via
    :attr:`IWFMModel.load_errors`.

    With ``strict=True``: logs the warning, then raises
    :class:`ComponentLoadError` chained from ``exc``. Callers that need a
    complete model (calibration, analysis pipelines) should pass this.

    With ``strict="collect"``: same recording behaviour as ``False``;
    the loader function calls :func:`_finalize_collected_errors` at the
    end of its run to raise a single :class:`ValidationError` aggregating
    every component that failed. This gives users the full picture in
    one shot instead of fail-on-first-then-rerun.
    """
    logger.warning(
        "Failed to load %s component from %s: %s: %s",
        component_name,
        source_file,
        type(exc).__name__,
        exc,
        exc_info=True,
    )
    # Backward-compatible scalar entry per component.
    model.metadata[f"{component_name}_load_error"] = f"{type(exc).__name__}: {exc}"
    # Typed list — the source of truth for IWFMModel.load_errors.
    typed = ComponentLoadError(component_name, source_file, exc)
    typed.__cause__ = exc
    model.metadata.setdefault(_LOAD_ERRORS_KEY, []).append(typed)
    if strict is True:
        raise typed from exc


def _finalize_collected_errors(model: IWFMModel, strict: StrictMode) -> None:
    """If ``strict="collect"`` and any component failed during the load,
    raise a single :class:`ValidationError` enumerating them.

    Callers (the ``load_from_*`` functions) invoke this once after all
    component sections have been processed.
    """
    if strict != "collect":
        return
    errors = model.metadata.get(_LOAD_ERRORS_KEY, [])
    if not errors:
        return
    raise ValidationError(
        f"{len(errors)} component(s) failed to load",
        errors=[str(e) for e in errors],
    )
