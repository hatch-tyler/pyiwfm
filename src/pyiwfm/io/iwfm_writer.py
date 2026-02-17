"""
Unified IWFM file line-writing utilities.

Mirrors ``iwfm_reader.py`` on the output side: every ``io/`` writer should
import helpers from this module rather than defining its own copy.

Canonical helpers
-----------------
- ``write_comment``   -- write an IWFM full-line comment (``C  ...``)
- ``write_value``     -- write a data value with optional ``/ description``
- ``ensure_parent_dir`` -- create parent directories for an output path
"""

from __future__ import annotations

from pathlib import Path
from typing import TextIO


def write_comment(f: TextIO, text: str) -> None:
    """Write an IWFM comment line.

    Produces ``C  <text>\\n``, matching the Fortran convention where
    ``C`` in column 1 marks a comment.

    Parameters
    ----------
    f : TextIO
        Open file handle to write to.
    text : str
        Comment text (without leading ``C`` or trailing newline).
    """
    f.write(f"C  {text}\n")


def write_value(f: TextIO, value: object, description: str = "") -> None:
    """Write a data value line with optional inline description.

    With *description*::

        <5 spaces><value padded to 30 chars>  / <description>\\n

    Without *description*::

        <5 spaces><value>\\n

    Parameters
    ----------
    f : TextIO
        Open file handle to write to.
    value : object
        The value to write (converted via ``str()``).
    description : str, optional
        Inline comment after ``/`` separator.
    """
    if description:
        f.write(f"     {value!s:<30s}  / {description}\n")
    else:
        f.write(f"     {value}\n")


def ensure_parent_dir(filepath: Path) -> None:
    """Create parent directories for *filepath* if they do not exist.

    Parameters
    ----------
    filepath : Path
        Target file path whose parent directory tree will be created.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
