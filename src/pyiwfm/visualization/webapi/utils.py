"""Shared utilities for the webapi package."""

from __future__ import annotations

import math


def sanitize_values(values: list) -> list:
    """Replace NaN/Inf with None in a list of numeric values for JSON safety."""
    return [
        None if (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) else v
        for v in values
    ]
