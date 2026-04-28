"""Shared helpers for the ``IWFMModel`` loader functions.

These were defined at module scope in v1.x's ``core/model.py``. v2.0
relocates them here so the loader functions in
``pyiwfm.core.loaders.from_*`` can import them without depending on
``core/model.py`` (which itself uses them only for the classmethod
dispatchers).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pyiwfm.core.exceptions import (
    ComponentError,
    ComponentLoadError,
    IWFMIOError,
)

if TYPE_CHECKING:
    from pathlib import Path

    from pyiwfm.core.model import IWFMModel

logger = logging.getLogger(__name__)

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
    strict: bool,
) -> None:
    """Common handling for a failed component load.

    With ``strict=False`` (default): logs a structured warning with full
    traceback and stores the error in ``model.metadata`` for backward
    compatibility. The model retains whatever components loaded
    successfully.

    With ``strict=True``: logs the warning, then raises
    :class:`ComponentLoadError` chained from ``exc``. Callers that need a
    complete model (calibration, analysis pipelines) should pass this.
    """
    logger.warning(
        "Failed to load %s component from %s: %s: %s",
        component_name,
        source_file,
        type(exc).__name__,
        exc,
        exc_info=True,
    )
    model.metadata[f"{component_name}_load_error"] = f"{type(exc).__name__}: {exc}"
    if strict:
        raise ComponentLoadError(component_name, source_file, exc) from exc
