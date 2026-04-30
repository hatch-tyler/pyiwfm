"""FastAPI exception handlers for the IWFM web API.

Maps the typed :mod:`pyiwfm.core.exceptions` family to JSON responses
with appropriate HTTP status codes and structured detail. Without
these handlers, every IWFM-typed exception that escapes a route would
return ``HTTP 500`` with the raw stringified exception in the body —
indistinguishable to a frontend from a server bug.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi.responses import JSONResponse

from pyiwfm.core.exceptions import (
    ComponentLoadError,
    FileFormatError,
    PyIWFMError,
)

if TYPE_CHECKING:
    from fastapi import FastAPI, Request

logger = logging.getLogger(__name__)


async def file_format_error_handler(_request: Request, exc: FileFormatError) -> JSONResponse:
    """Map :class:`FileFormatError` (e.g. a malformed user upload) to ``400``."""
    detail: dict[str, object] = {
        "error": str(exc),
        "type": "FileFormatError",
    }
    if exc.line_number is not None:
        detail["line_number"] = exc.line_number
    logger.warning("FileFormatError: %s (line=%s)", exc, exc.line_number)
    return JSONResponse(status_code=400, content=detail)


async def component_load_error_handler(_request: Request, exc: ComponentLoadError) -> JSONResponse:
    """Map :class:`ComponentLoadError` to ``422 Unprocessable Entity``.

    422 (rather than 400) signals "the request was syntactically valid
    but the referenced model component couldn't be loaded" — useful
    for clients that want to distinguish a genuinely malformed
    request from a load-time data problem.
    """
    detail = {
        "error": str(exc),
        "type": "ComponentLoadError",
        "component": exc.component_name,
        "source_file": str(exc.source_file) if exc.source_file is not None else None,
    }
    logger.warning(
        "ComponentLoadError: component=%s source=%s msg=%s",
        exc.component_name,
        exc.source_file,
        exc,
    )
    return JSONResponse(status_code=422, content=detail)


async def pyiwfm_error_handler(_request: Request, exc: PyIWFMError) -> JSONResponse:
    """Catch-all for the remaining :class:`PyIWFMError` subclasses (``400``).

    Registered after the specific handlers above; FastAPI dispatches on
    the most specific registered handler first, so this only fires for
    ``MeshError`` / ``StratigraphyError`` / ``ComponentError`` /
    ``ValidationError`` / ``IWFMIOError`` / ``ConnectorError``.
    """
    detail = {
        "error": str(exc),
        "type": exc.__class__.__name__,
    }
    logger.warning("%s: %s", exc.__class__.__name__, exc)
    return JSONResponse(status_code=400, content=detail)


def register_exception_handlers(app: FastAPI) -> None:
    """Register all pyiwfm exception handlers on the given FastAPI app.

    Order matters only loosely: FastAPI selects the handler whose
    registered exception class is the closest base of the raised
    exception. We register the most specific (``FileFormatError``,
    ``ComponentLoadError``) first for clarity.
    """
    app.add_exception_handler(FileFormatError, file_format_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(ComponentLoadError, component_load_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(PyIWFMError, pyiwfm_error_handler)  # type: ignore[arg-type]
