"""Parameter — extracted from runner/pest.py in v2.0 PR 5.

The class body is verbatim from v1.x ``runner/pest.py``; this module
just gives it a dedicated file so the package no longer has a
1,400-line module.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Parameter:
    """PEST++ parameter definition.

    Attributes
    ----------
    name : str
        Parameter name (up to 200 chars in PEST++).
    initial_value : float
        Initial parameter value.
    lower_bound : float
        Lower bound for parameter.
    upper_bound : float
        Upper bound for parameter.
    group : str
        Parameter group name.
    transform : str
        Transformation type: 'none', 'log', 'fixed', 'tied'.
    change_limit : str
        Parameter change limit type: 'factor', 'relative', 'absolute'.
    scale : float
        Scale factor for parameter.
    offset : float
        Offset for parameter.
    dercom : int
        Derivative command index (1-based).
    tied_to : str
        Name of parent parameter if transform is 'tied'.
    """

    name: str
    initial_value: float
    lower_bound: float
    upper_bound: float
    group: str = "default"
    transform: str = "none"
    change_limit: str = "factor"
    scale: float = 1.0
    offset: float = 0.0
    dercom: int = 1
    tied_to: str = ""

    def __post_init__(self) -> None:
        """Validate parameter."""
        if len(self.name) > 200:
            raise ValueError(f"Parameter name too long: {self.name}")
        if self.lower_bound > self.upper_bound:
            raise ValueError(f"Lower bound ({self.lower_bound}) > upper bound ({self.upper_bound})")
        if self.transform == "tied" and not self.tied_to:
            raise ValueError(f"Tied parameter '{self.name}' must specify tied_to")
        # Skip bounds check for tied/fixed parameters (values may not be meaningful)
        if self.transform not in ("tied", "fixed"):
            if not self.lower_bound <= self.initial_value <= self.upper_bound:
                raise ValueError(
                    f"Initial value ({self.initial_value}) not within bounds "
                    f"[{self.lower_bound}, {self.upper_bound}]"
                )

    def to_pest_line(self) -> str:
        """Format as v1 PEST control file parameter line."""
        return (
            f"{self.name:<20s} {self.transform:>5s}   {self.change_limit:<8s}"
            f"{self.initial_value:>15.7g}  {self.lower_bound:>13.7g}  "
            f"{self.upper_bound:>13.7g}  {self.group:<12s}"
            f"{self.scale:>10.3g}  {self.offset:>6.3g}  {self.dercom}"
        )

    def to_csv_dict(self) -> dict[str, str]:
        """Return dict for CSV export (v2 external format)."""
        return {
            "parnme": self.name,
            "partrans": self.transform,
            "parchglim": self.change_limit,
            "parval1": str(self.initial_value),
            "parlbnd": str(self.lower_bound),
            "parubnd": str(self.upper_bound),
            "pargp": self.group,
            "scale": str(self.scale),
            "offset": str(self.offset),
            "dercom": str(self.dercom),
            "partied": self.tied_to,
        }
