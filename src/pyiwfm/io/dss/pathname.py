"""
HEC-DSS pathname utilities.

This module provides classes and functions for working with HEC-DSS pathnames
which are used to identify data records in DSS files.

DSS Pathname Format: /A/B/C/D/E/F/
- A: Project or Basin name
- B: Location (e.g., stream gage, well ID)
- C: Parameter (e.g., FLOW, STAGE, HEAD)
- D: Date range (e.g., 01JAN2000)
- E: Time interval (e.g., 1DAY, 1HOUR)
- F: Version or scenario
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterator


# Valid DSS time intervals
VALID_INTERVALS = {
    "1MIN", "2MIN", "3MIN", "4MIN", "5MIN", "6MIN", "10MIN", "12MIN", "15MIN",
    "20MIN", "30MIN", "1HOUR", "2HOUR", "3HOUR", "4HOUR", "6HOUR", "8HOUR",
    "12HOUR", "1DAY", "1WEEK", "1MON", "1YEAR", "IR-DAY", "IR-MON", "IR-YEAR",
}

# Mapping from common names to DSS interval codes
INTERVAL_MAPPING = {
    "minute": "1MIN",
    "hourly": "1HOUR",
    "hour": "1HOUR",
    "daily": "1DAY",
    "day": "1DAY",
    "weekly": "1WEEK",
    "week": "1WEEK",
    "monthly": "1MON",
    "month": "1MON",
    "yearly": "1YEAR",
    "year": "1YEAR",
    "annual": "1YEAR",
}

# DSS parameter codes
PARAMETER_CODES = {
    "flow": "FLOW",
    "streamflow": "FLOW",
    "discharge": "FLOW",
    "stage": "STAGE",
    "head": "HEAD",
    "groundwater_head": "GW-HEAD",
    "elevation": "ELEV",
    "precipitation": "PRECIP",
    "evaporation": "EVAP",
    "evapotranspiration": "ET",
    "storage": "STOR",
    "volume": "VOLUME",
    "area": "AREA",
    "velocity": "VEL",
    "temperature": "TEMP",
    "pumping": "PUMP",
    "diversion": "DIVERT",
    "return_flow": "RETURN",
}


@dataclass
class DSSPathname:
    """
    HEC-DSS pathname with structured parts.

    DSS pathnames identify data records using six parts:
    /A/B/C/D/E/F/

    Attributes:
        a_part: Project/Basin name
        b_part: Location identifier
        c_part: Parameter type (FLOW, HEAD, etc.)
        d_part: Date/time window
        e_part: Time interval (1DAY, 1HOUR, etc.)
        f_part: Version/scenario name
    """

    a_part: str = ""
    b_part: str = ""
    c_part: str = ""
    d_part: str = ""
    e_part: str = ""
    f_part: str = ""

    def __post_init__(self) -> None:
        """Validate and normalize pathname parts."""
        # Ensure all parts are uppercase (DSS convention)
        self.a_part = self.a_part.upper() if self.a_part else ""
        self.b_part = self.b_part.upper() if self.b_part else ""
        self.c_part = self.c_part.upper() if self.c_part else ""
        self.d_part = self.d_part.upper() if self.d_part else ""
        self.e_part = self.e_part.upper() if self.e_part else ""
        self.f_part = self.f_part.upper() if self.f_part else ""

    def __str__(self) -> str:
        """Return the full pathname string."""
        return f"/{self.a_part}/{self.b_part}/{self.c_part}/{self.d_part}/{self.e_part}/{self.f_part}/"

    @classmethod
    def from_string(cls, pathname: str) -> "DSSPathname":
        """
        Parse a pathname from string.

        Args:
            pathname: DSS pathname string (e.g., "/PROJECT/LOC/FLOW//1DAY/V1/")

        Returns:
            DSSPathname object

        Raises:
            ValueError: If pathname format is invalid
        """
        # Remove leading/trailing whitespace
        pathname = pathname.strip()

        # Check for valid format (starts and ends with /)
        if not pathname.startswith("/") or not pathname.endswith("/"):
            raise ValueError(f"Invalid DSS pathname format: {pathname}")

        # Split by /
        parts = pathname[1:-1].split("/")  # Remove first and last /

        if len(parts) != 6:
            raise ValueError(
                f"DSS pathname must have exactly 6 parts, got {len(parts)}: {pathname}"
            )

        return cls(
            a_part=parts[0],
            b_part=parts[1],
            c_part=parts[2],
            d_part=parts[3],
            e_part=parts[4],
            f_part=parts[5],
        )

    @classmethod
    def build(
        cls,
        project: str = "",
        location: str = "",
        parameter: str = "",
        date_range: str = "",
        interval: str = "1DAY",
        version: str = "",
    ) -> "DSSPathname":
        """
        Build a pathname with more descriptive parameter names.

        Args:
            project: Project or basin name (A part)
            location: Location identifier (B part)
            parameter: Parameter name (C part) - will be converted to DSS code
            date_range: Date range (D part)
            interval: Time interval (E part) - will be converted to DSS code
            version: Version or scenario (F part)

        Returns:
            DSSPathname object
        """
        # Convert parameter to DSS code if needed
        param_lower = parameter.lower()
        c_part = PARAMETER_CODES.get(param_lower, parameter.upper())

        # Convert interval to DSS code if needed
        interval_lower = interval.lower()
        e_part = INTERVAL_MAPPING.get(interval_lower, interval.upper())

        return cls(
            a_part=project,
            b_part=location,
            c_part=c_part,
            d_part=date_range,
            e_part=e_part,
            f_part=version,
        )

    def with_location(self, location: str) -> "DSSPathname":
        """Return a new pathname with a different location (B part)."""
        return DSSPathname(
            a_part=self.a_part,
            b_part=location.upper(),
            c_part=self.c_part,
            d_part=self.d_part,
            e_part=self.e_part,
            f_part=self.f_part,
        )

    def with_parameter(self, parameter: str) -> "DSSPathname":
        """Return a new pathname with a different parameter (C part)."""
        param_lower = parameter.lower()
        c_part = PARAMETER_CODES.get(param_lower, parameter.upper())
        return DSSPathname(
            a_part=self.a_part,
            b_part=self.b_part,
            c_part=c_part,
            d_part=self.d_part,
            e_part=self.e_part,
            f_part=self.f_part,
        )

    def with_date_range(self, date_range: str) -> "DSSPathname":
        """Return a new pathname with a different date range (D part)."""
        return DSSPathname(
            a_part=self.a_part,
            b_part=self.b_part,
            c_part=self.c_part,
            d_part=date_range.upper(),
            e_part=self.e_part,
            f_part=self.f_part,
        )

    def with_version(self, version: str) -> "DSSPathname":
        """Return a new pathname with a different version (F part)."""
        return DSSPathname(
            a_part=self.a_part,
            b_part=self.b_part,
            c_part=self.c_part,
            d_part=self.d_part,
            e_part=self.e_part,
            f_part=version.upper(),
        )

    def matches(self, pattern: "DSSPathname | str") -> bool:
        """
        Check if this pathname matches a pattern.

        Pattern parts can be empty to match any value.

        Args:
            pattern: DSSPathname pattern or string

        Returns:
            True if pathname matches pattern
        """
        if isinstance(pattern, str):
            pattern = DSSPathname.from_string(pattern)

        return (
            (not pattern.a_part or self.a_part == pattern.a_part)
            and (not pattern.b_part or self.b_part == pattern.b_part)
            and (not pattern.c_part or self.c_part == pattern.c_part)
            and (not pattern.d_part or self.d_part == pattern.d_part)
            and (not pattern.e_part or self.e_part == pattern.e_part)
            and (not pattern.f_part or self.f_part == pattern.f_part)
        )

    @property
    def is_regular_interval(self) -> bool:
        """Check if this is a regular-interval time series."""
        return not self.e_part.startswith("IR-")

    @property
    def is_irregular_interval(self) -> bool:
        """Check if this is an irregular-interval time series."""
        return self.e_part.startswith("IR-")


@dataclass
class DSSPathnameTemplate:
    """
    Template for generating multiple DSS pathnames.

    Allows creating pathnames with variable parts.

    Example:
        >>> template = DSSPathnameTemplate(
        ...     a_part="PROJECT",
        ...     c_part="FLOW",
        ...     e_part="1DAY",
        ...     f_part="OBS",
        ... )
        >>> pathname = template.make_pathname(location="STREAM_01")
    """

    a_part: str = ""
    b_part: str = ""  # Usually variable
    c_part: str = ""
    d_part: str = ""  # Usually variable
    e_part: str = "1DAY"
    f_part: str = ""

    def make_pathname(
        self,
        location: str | None = None,
        date_range: str | None = None,
        **kwargs,
    ) -> DSSPathname:
        """
        Create a pathname from the template.

        Args:
            location: Location to use (overrides b_part)
            date_range: Date range to use (overrides d_part)
            **kwargs: Additional parts to override (a_part, c_part, etc.)

        Returns:
            DSSPathname object
        """
        return DSSPathname(
            a_part=kwargs.get("a_part", self.a_part),
            b_part=(location or kwargs.get("b_part", self.b_part)).upper(),
            c_part=kwargs.get("c_part", self.c_part),
            d_part=(date_range or kwargs.get("d_part", self.d_part)).upper(),
            e_part=kwargs.get("e_part", self.e_part),
            f_part=kwargs.get("f_part", self.f_part),
        )

    def make_pathnames(
        self,
        locations: list[str],
        date_range: str = "",
    ) -> Iterator[DSSPathname]:
        """
        Generate pathnames for multiple locations.

        Args:
            locations: List of location identifiers
            date_range: Date range to use

        Yields:
            DSSPathname objects
        """
        for loc in locations:
            yield self.make_pathname(location=loc, date_range=date_range)


def format_dss_date(dt: datetime) -> str:
    """
    Format a datetime for DSS D-part.

    Args:
        dt: datetime object

    Returns:
        Date string in DSS format (e.g., "01JAN2000")
    """
    return dt.strftime("%d%b%Y").upper()


def format_dss_date_range(start: datetime, end: datetime) -> str:
    """
    Format a date range for DSS D-part.

    Args:
        start: Start datetime
        end: End datetime

    Returns:
        Date range string (e.g., "01JAN2000-31DEC2000")
    """
    return f"{format_dss_date(start)}-{format_dss_date(end)}"


def parse_dss_date(date_str: str) -> datetime:
    """
    Parse a DSS date string.

    Args:
        date_str: Date string in DSS format (e.g., "01JAN2000")

    Returns:
        datetime object
    """
    return datetime.strptime(date_str.upper(), "%d%b%Y")


def interval_to_minutes(interval: str) -> int:
    """
    Convert DSS interval to minutes.

    Args:
        interval: DSS interval string (e.g., "1DAY", "1HOUR")

    Returns:
        Interval in minutes
    """
    interval = interval.upper()

    if interval.endswith("MIN"):
        return int(interval[:-3])
    elif interval.endswith("HOUR"):
        return int(interval[:-4]) * 60
    elif interval.endswith("DAY"):
        return int(interval[:-3]) * 60 * 24
    elif interval.endswith("WEEK"):
        return int(interval[:-4]) * 60 * 24 * 7
    elif interval.endswith("MON"):
        return int(interval[:-3]) * 60 * 24 * 30  # Approximate
    elif interval.endswith("YEAR"):
        return int(interval[:-4]) * 60 * 24 * 365  # Approximate
    else:
        raise ValueError(f"Unknown DSS interval: {interval}")


def minutes_to_interval(minutes: int) -> str:
    """
    Convert minutes to the nearest DSS interval.

    Args:
        minutes: Interval in minutes

    Returns:
        DSS interval string
    """
    if minutes < 60:
        if minutes in [1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30]:
            return f"{minutes}MIN"
        else:
            # Find nearest valid minute interval
            valid_mins = [1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30]
            nearest = min(valid_mins, key=lambda x: abs(x - minutes))
            return f"{nearest}MIN"
    elif minutes < 60 * 24:
        hours = minutes // 60
        if hours in [1, 2, 3, 4, 6, 8, 12]:
            return f"{hours}HOUR"
        else:
            return "1HOUR"
    elif minutes < 60 * 24 * 7:
        return "1DAY"
    elif minutes < 60 * 24 * 28:
        return "1WEEK"
    elif minutes < 60 * 24 * 365:
        return "1MON"
    else:
        return "1YEAR"
