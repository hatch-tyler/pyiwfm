"""
Reader for observation well specification files (multi-layer target input).

The obs well spec file defines wells with screen intervals for
transmissivity-weighted depth averaging of groundwater heads.

File format (whitespace-delimited, one header line)::

    Name                X           Y    Element   BOS    TOS  OverwriteLayer
    S_380313N1219426W  6302184.5  2161430.2   1234  -175.44  -105.44  -1

``OverwriteLayer = -1`` means use the screen interval; a positive value
forces all weight to that single layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ObsWellSpec:
    """Specification for one observation well with screen interval.

    Attributes
    ----------
    name : str
        Well identifier (up to 25 characters).
    x : float
        X coordinate of the well.
    y : float
        Y coordinate of the well.
    element_id : int
        Element containing the well (1-based).
    bottom_of_screen : float
        Bottom elevation of the well screen.
    top_of_screen : float
        Top elevation of the well screen.
    overwrite_layer : int
        Layer override (-1 = use screen interval, >0 = force single layer).
    """

    name: str
    x: float
    y: float
    element_id: int
    bottom_of_screen: float
    top_of_screen: float
    overwrite_layer: int = -1


def read_obs_well_spec(filepath: Path | str) -> list[ObsWellSpec]:
    """Read an observation well specification file.

    Parameters
    ----------
    filepath : Path or str
        Path to the obs well spec file.

    Returns
    -------
    list[ObsWellSpec]
        Parsed well specifications.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If a data line cannot be parsed.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Obs well spec file not found: {path}")

    wells: list[ObsWellSpec] = []

    with open(path) as f:
        # Skip header line
        header = f.readline()
        if not header:
            return wells

        for line_num, line in enumerate(f, start=2):
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("C"):
                continue

            parts = stripped.split()
            if len(parts) < 6:
                continue

            try:
                name = parts[0]
                x = float(parts[1])
                y = float(parts[2])
                elem = int(parts[3])
                bos = float(parts[4])
                tos = float(parts[5])
                overwrite = int(parts[6]) if len(parts) > 6 else -1

                wells.append(
                    ObsWellSpec(
                        name=name,
                        x=x,
                        y=y,
                        element_id=elem,
                        bottom_of_screen=bos,
                        top_of_screen=tos,
                        overwrite_layer=overwrite,
                    )
                )
            except (ValueError, IndexError) as e:
                raise ValueError(f"Error parsing obs well spec at line {line_num}: {e}") from e

    return wells
