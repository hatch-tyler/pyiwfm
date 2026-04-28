"""
Unified IWFM file line-writing utilities.

Mirrors ``iwfm_reader.py`` on the output side: every ``io/`` writer should
import helpers from this module rather than defining its own copy.

Canonical helpers
-----------------
- ``write_comment``       -- write an IWFM full-line comment (``C  ...``)
- ``write_value``         -- write a data value with optional ``/ description``
- ``ensure_parent_dir``   -- create parent directories for an output path
- ``open_iwfm_file``      -- context manager: ensure parent dir, open file,
                             write the standard ``C  IWFM ...`` header
- ``write_element_group`` -- write an IWFM element-group block (group ID,
                             count, element IDs)
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
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


@contextmanager
def open_iwfm_file(filepath: Path | str, header: str) -> Iterator[TextIO]:
    """Context manager combining the standard IWFM writer setup.

    Replaces the v1.x boilerplate that appeared at the top of every
    ``write_*`` function across the small ``io/`` component-writer
    modules:

    .. code-block:: python

        # v1.x:
        filepath = Path(filepath)
        ensure_parent_dir(filepath)
        with open(filepath, "w") as f:
            write_comment(f, "IWFM ... File")
            ...

        # v2.x:
        with open_iwfm_file(filepath, "IWFM ... File") as f:
            ...

    Parameters
    ----------
    filepath : Path or str
        Output file path.
    header : str
        Title text written via :func:`write_comment` as the first line.

    Yields
    ------
    TextIO
        Open file handle, positioned just after the header comment.
    """
    p = Path(filepath)
    ensure_parent_dir(p)
    with open(p, "w") as f:
        write_comment(f, header)
        yield f


def write_element_group(f: TextIO, group_id: int, element_ids: list[int]) -> None:
    """Write an IWFM element-group block.

    Format produced::

        <group_id>  <n_elements>  <first_element_id>
            <element_id>          # one per subsequent line

    For an empty group, writes ``<group_id>  0`` on a single line.

    Parameters
    ----------
    f : TextIO
        Open file handle to write to.
    group_id : int
        Group identifier (1-based, IWFM convention).
    element_ids : list[int]
        Element IDs in this group. May be empty.
    """
    n = len(element_ids)
    if n == 0:
        f.write(f"     {group_id:>6d}  {0:>4d}\n")
        return

    f.write(f"     {group_id:>6d}  {n:>4d}  {element_ids[0]:>6d}\n")
    for elem_id in element_ids[1:]:
        f.write(f"     {elem_id:>6d}\n")
