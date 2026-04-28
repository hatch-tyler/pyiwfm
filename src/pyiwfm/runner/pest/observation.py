"""Observation and ObservationGroup — extracted from runner/pest.py in v2.0 PR 5.

The class body is verbatim from v1.x ``runner/pest.py``; this module
just gives it a dedicated file so the package no longer has a
1,400-line module.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Observation:
    """PEST++ observation definition.

    Attributes
    ----------
    name : str
        Observation name (up to 200 chars in PEST++).
    value : float
        Observed value.
    weight : float
        Observation weight (inverse of standard deviation).
    group : str
        Observation group name.
    """

    name: str
    value: float
    weight: float = 1.0
    group: str = "default"
    extra_columns: dict[str, str] | None = None

    def __post_init__(self) -> None:
        """Validate observation."""
        if len(self.name) > 200:
            raise ValueError(f"Observation name too long: {self.name}")
        if self.weight < 0:
            raise ValueError(f"Weight must be non-negative: {self.weight}")
        if self.extra_columns is None:
            self.extra_columns = {}

    def to_pest_line(self) -> str:
        """Format as PEST control file observation line."""
        return f"{self.name:20s} {self.value:15.7e} {self.weight:10.4e} {self.group:20s}"

    def to_csv_dict(self, extra_fieldnames: list[str] | None = None) -> dict[str, str]:
        """Return dict for CSV export (v2 external format).

        Parameters
        ----------
        extra_fieldnames : list[str] | None
            If provided, include these extra columns from ``extra_columns``.
            Columns not present in ``extra_columns`` are written as empty strings.
        """
        d = {
            "obsnme": self.name,
            "obsval": str(self.value),
            "weight": str(self.weight),
            "obgnme": self.group,
        }
        if extra_fieldnames:
            for col in extra_fieldnames:
                d[col] = self.extra_columns.get(col, "") if self.extra_columns else ""
        return d


@dataclass
class ObservationGroup:
    """Group of observations with shared properties.

    Attributes
    ----------
    name : str
        Group name.
    observations : list[Observation]
        Observations in this group.
    covariance_matrix : str | None
        Path to covariance matrix file for this group.
    """

    name: str
    observations: list[Observation] = field(default_factory=list)
    covariance_matrix: str | None = None

    def add_observation(
        self,
        name: str,
        value: float,
        weight: float = 1.0,
    ) -> Observation:
        """Add an observation to this group.

        Parameters
        ----------
        name : str
            Observation name.
        value : float
            Observed value.
        weight : float
            Observation weight.

        Returns
        -------
        Observation
            The created observation.
        """
        obs = Observation(name=name, value=value, weight=weight, group=self.name)
        self.observations.append(obs)
        return obs
