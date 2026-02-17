"""
Supply adjustment file reader and writer for IWFM.

IWFM supply adjustment files use the IntTSDataInFileType format:
    - Header comments
    - NCOLADJ  (number of columns)
    - NSPADJ   (time step update frequency)
    - NFQADJ   (data repetition frequency)
    - DSSFL    (DSS filename, blank for inline)
    - Data lines: timestamp + integer adjustment codes (00, 01, 10)

The adjustment codes are two-digit integers:
    - 1st digit: 0 = no agriculture adjustment, 1 = adjust agriculture
    - 2nd digit: 0 = no urban adjustment, 1 = adjust urban
    - Combined: 00=none, 01=urban only, 10=ag only, 11=both (deprecated)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from pyiwfm.io.timeseries_ascii import (
    _strip_inline_comment,
    format_iwfm_timestamp,
    parse_iwfm_timestamp,
)

logger = logging.getLogger(__name__)

# IWFM line-comment characters (must appear in column 1).
_LINE_COMMENT_CHARS = ("C", "c", "*")


def _is_fortran_comment(line: str) -> bool:
    """Check if a line is a Fortran-style IWFM comment.

    Only C, c, or * in column 1 count as line comments.
    This distinction matters for blank DSSFL lines like
    ``'                   / DSSFL'`` which are data lines (empty value).
    """
    if not line or not line.strip():
        return False  # blank lines are not comments — caller decides handling
    return line[0] in _LINE_COMMENT_CHARS


@dataclass
class SupplyAdjustment:
    """Parsed supply adjustment specification data.

    Attributes:
        n_columns: Number of adjustment columns (NCOLADJ).
        nsp: Time step update frequency (NSPADJ).
        nfq: Data repetition frequency (NFQADJ).
        dss_file: DSS filename (empty string if inline data).
        times: List of timestamps for each data row.
        values: List of rows, each row is a list of integer adjustment codes.
        header_lines: Original header comment lines.
    """

    n_columns: int = 0
    nsp: int = 1
    nfq: int = 0
    dss_file: str = ""
    times: list[datetime] = field(default_factory=list)
    values: list[list[int]] = field(default_factory=list)
    header_lines: list[str] = field(default_factory=list)


def read_supply_adjustment(filepath: Path | str) -> SupplyAdjustment:
    """Read a supply adjustment file.

    Parses the IWFM integer time series format (NCOL, NSP, NFQ, DSSFL)
    followed by timestamp + integer data rows.

    Args:
        filepath: Path to the supply adjustment file.

    Returns:
        SupplyAdjustment with parsed data.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If file format is invalid.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Supply adjustment file not found: {filepath}")

    result = SupplyAdjustment()

    with open(filepath, errors="replace") as f:
        # Phase 1: Read NCOLADJ (first non-comment value)
        for line in f:
            if _is_fortran_comment(line):
                result.header_lines.append(line.rstrip("\n"))
                continue
            value_str = _strip_inline_comment(line)
            if not value_str:
                continue
            result.n_columns = int(value_str)
            break

        # Phase 2: Read NSPADJ (second non-comment value)
        for line in f:
            if _is_fortran_comment(line):
                continue
            value_str = _strip_inline_comment(line)
            if not value_str:
                continue
            result.nsp = int(value_str)
            break

        # Phase 3: Read NFQADJ (third non-comment value)
        for line in f:
            if _is_fortran_comment(line):
                continue
            value_str = _strip_inline_comment(line)
            if not value_str:
                continue
            result.nfq = int(value_str)
            break

        # Phase 4: Read DSSFL (may be blank)
        # Use _is_fortran_comment (not _is_comment_line) because a blank
        # DSSFL line like "     / DSSFL" must NOT be treated as a comment.
        for line in f:
            if _is_fortran_comment(line):
                continue
            if not line.strip():
                result.dss_file = ""
                break
            value_str = _strip_inline_comment(line)
            result.dss_file = value_str  # May be empty string
            break

        # Phase 5: Read data lines (timestamp + integer codes)
        # Use token-based parsing (split on whitespace) instead of
        # fixed-width slicing.
        for line in f:
            if _is_fortran_comment(line):
                continue
            stripped = line.strip()
            if not stripped:
                continue

            # Try to parse as a timestamp data line
            try:
                tokens = stripped.split()
                if not tokens:
                    continue
                dt = parse_iwfm_timestamp(tokens[0])

                # Remaining tokens are integer adjustment codes
                int_values = [int(v) for v in tokens[1:]]

                result.times.append(dt)
                result.values.append(int_values)
            except ValueError:
                # Skip non-data lines (DSS pathnames, extra comments, etc.)
                continue

    logger.info(
        "Read supply adjustment: %d columns, %d rows from %s",
        result.n_columns,
        len(result.times),
        filepath,
    )
    return result


def write_supply_adjustment(
    data: SupplyAdjustment,
    filepath: Path | str,
) -> Path:
    """Write a supply adjustment file.

    Writes the IWFM integer time series format: NCOL, NSP, NFQ, DSSFL,
    followed by timestamp + integer data rows.

    Args:
        data: SupplyAdjustment data to write.
        filepath: Output file path.

    Returns:
        Path to the written file.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w") as f:
        # Header
        f.write(
            "C*******************************************************************************\n"
        )
        f.write("C\n")
        f.write("C                     SUPPLY ADJUSTMENT SPECIFICATIONS\n")
        f.write("C                           for IWFM Simulation\n")
        f.write("C\n")
        f.write("C    Generated by pyiwfm\n")
        f.write(f"C    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(
            "C*******************************************************************************\n"
        )
        f.write("C\n")
        f.write("C   NCOLADJ:  Number of columns in the supply adjustment specifications data\n")
        f.write(
            "C   NSPADJ :  Number of time steps to update the supply adjustment specifications\n"
        )
        f.write("C   NFQADJ :  Repetition frequency of the supply adjustment specifications data\n")
        f.write("C   DSSFL  :  DSS filename (blank = inline data)\n")
        f.write("C\n")
        f.write(
            "C-------------------------------------------------------------------------------\n"
        )
        f.write("C         VALUE                                      DESCRIPTION\n")
        f.write(
            "C-------------------------------------------------------------------------------\n"
        )

        # Parameters (no FACTOR for integer TS)
        f.write(f"          {data.n_columns:<38}/ NCOLADJ\n")
        f.write(f"          {data.nsp:<38}/ NSPADJ\n")
        f.write(f"          {data.nfq:<38}/ NFQADJ\n")
        dss_str = data.dss_file if data.dss_file else ""
        f.write(f"    {dss_str:<44}/ DSSFL\n")

        # Data section header
        f.write(
            "C*******************************************************************************\n"
        )
        f.write("C                    Supply Adjustment Specifications Data\n")
        f.write("C\n")
        f.write("C   ITADJ:  Time\n")
        f.write("C   KADJ :  Supply adjustment code (2-digit: 1st=ag, 2nd=urban)\n")
        f.write("C           00=None, 01=Urban, 10=Ag\n")
        f.write("C\n")
        f.write(
            "C-------------------------------------------------------------------------------\n"
        )

        # Data rows
        for i, dt in enumerate(data.times):
            ts_str = format_iwfm_timestamp(dt)
            vals = data.values[i] if i < len(data.values) else []
            val_strs = [f"\t{v:02d}" for v in vals]
            f.write(f"    {ts_str}{''.join(val_strs)}\n")

    logger.info("Wrote supply adjustment: %s", filepath)
    return filepath
