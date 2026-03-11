"""Generate PEST++ instruction files from model output files.

This module creates INS files by scanning actual simulated output (SMP files,
CalcTypeHyd .out files, etc.) — guaranteeing that instruction file structure
matches model output exactly.

INS files should be generated once after a setup forward run, not during
every forward run or from observation data pipelines.

Usage::

    from pyiwfm.calibration.pest_files import generate_smp_ins, generate_thyd_ins

    # From SMP output (streams, GW, subsidence, head diffs)
    generate_smp_ins("SQRTSTR_OUT.smp", "pest/STR_PEST.ins", col_range="50:60")

    # From CalcTypeHyd .out files
    generate_thyd_ins("sim_sub1sub2_cls1.out", "pest/sim_sub1sub2_cls1.ins")
"""

from __future__ import annotations

from pathlib import Path

# PEST++ v5+ supports up to 200-char obs names; we cap at 50 for readability.
_MAX_OBS_NAME = 50


def _safe_pest_name(raw: str, max_len: int = _MAX_OBS_NAME) -> str:
    """Sanitize a string for use as part of a PEST++ observation name.

    Replaces spaces, ``%``, and other problematic characters with ``_``.
    Truncates to *max_len* characters.
    """
    safe = raw.replace(" ", "_").replace("%", "_")
    return safe[:max_len] if len(safe) > max_len else safe


def generate_smp_ins(
    smp_path: Path | str,
    ins_path: Path | str,
    col_range: str = "50:60",
) -> int:
    """Generate a PEST instruction file from an SMP output file.

    Each non-empty line in the SMP file becomes an observation. The observation
    name is ``{station}_{seq:04d}`` where seq resets per station.

    Parameters
    ----------
    smp_path : Path
        Path to the SMP output file (e.g., SQRTSTR_OUT.smp, HD_OUT.smp).
    ins_path : Path
        Path for the generated instruction file.
    col_range : str
        Column range for value extraction (default "50:60" for standard
        A25+A12+A12+A11 SMP format).

    Returns
    -------
    int
        Number of observations written.
    """
    smp_path = Path(smp_path)
    ins_path = Path(ins_path)
    ins_path.parent.mkdir(parents=True, exist_ok=True)

    n_obs = 0
    prev_station = None
    seq = 0

    with open(smp_path) as fin, open(ins_path, "w") as fout:
        fout.write("pif #\n")

        for line in fin:
            if not line.strip():
                continue
            # SMP format: station(25) date(12) time(12) value(11+)
            station = line[:25].strip()
            if not station:
                continue

            # Sanitize for PEST++ (preserve case, replace problematic chars)
            safe_station = _safe_pest_name(station, max_len=44)

            if station != prev_station:
                seq = 1
                prev_station = station
            else:
                seq += 1

            obs_name = f"{safe_station}_{seq:04d}"
            fout.write(f"l1  [{obs_name}]{col_range}\n")
            n_obs += 1

    return n_obs


def generate_thyd_ins(
    out_path: Path | str,
    ins_path: Path | str,
    col_range: str = "34:46",
) -> int:
    """Generate a PEST instruction file from a CalcTypeHyd .out file.

    Skips the header line, then reads PEST_NAME from columns 1-14.
    CalcTypeHyd output format: ``(A14, A12, F20.6)`` — name, date, value.

    Parameters
    ----------
    out_path : Path
        Path to the CalcTypeHyd .out file.
    ins_path : Path
        Path for the generated instruction file.
    col_range : str
        Column range for value extraction (default "34:46" matches F20.6
        starting at col 27, capturing the significant digits).

    Returns
    -------
    int
        Number of observations written.
    """
    out_path = Path(out_path)
    ins_path = Path(ins_path)
    ins_path.parent.mkdir(parents=True, exist_ok=True)

    n_obs = 0
    with open(out_path) as fin, open(ins_path, "w") as fout:
        fout.write("pif #\n")
        fout.write("l1\n")  # skip header

        fin.readline()  # skip header
        for line in fin:
            if not line.strip():
                continue
            pest_name = line[:14].strip()
            if not pest_name or pest_name == "PEST_NAME":
                continue
            fout.write(f"l1 [{pest_name}]{col_range}\n")
            n_obs += 1

    return n_obs


def generate_multilayer_ins(
    out_path: Path | str,
    ins_path: Path | str,
    col_range: str = "50:60",
) -> int:
    """Generate a PEST instruction file from GW_OUT_ml.smp.

    The multi-layer SMP output uses fixed-format columns:
    ``(A25, date, 2X, time, 4X, F11.2, ...)``.
    The simulated value occupies columns 50-60.

    If the first line looks like a header (contains "Name" or "Date" as
    text), it is skipped.  Otherwise all lines are treated as data.

    Parameters
    ----------
    out_path : Path
        Path to the multi-layer output SMP file.
    ins_path : Path
        Path for the generated instruction file.
    col_range : str
        Column range for the simulated value (default "50:60").

    Returns
    -------
    int
        Number of observations written.
    """
    out_path = Path(out_path)
    ins_path = Path(ins_path)
    ins_path.parent.mkdir(parents=True, exist_ok=True)

    n_obs = 0
    prev_well = None
    seq = 0

    with open(out_path) as fin, open(ins_path, "w") as fout:
        fout.write("pif #\n")

        # Peek at first line to detect header
        first_line = fin.readline()
        if not first_line:
            return 0
        first_stripped = first_line.strip().lower()
        has_header = "name" in first_stripped[:30] and "date" in first_stripped
        if has_header:
            fout.write("l1\n")  # skip header in INS
        else:
            # No header — rewind to process first line as data
            fin.seek(0)

        for line in fin:
            if not line.strip():
                continue
            well_name = line[:25].strip()
            if not well_name:
                continue

            # Sanitize for PEST (preserve case, replace spaces/% with _)
            # Reserve 6 chars for "_NNNNN" suffix
            safe_name = _safe_pest_name(well_name, max_len=_MAX_OBS_NAME - 6)

            if well_name != prev_well:
                seq = 1
                prev_well = well_name
            else:
                seq += 1

            obs_name = f"{safe_name}_{seq:05d}"
            fout.write(f"l1 [{obs_name}]{col_range}\n")
            n_obs += 1

    return n_obs
