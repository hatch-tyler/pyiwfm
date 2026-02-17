"""Core data structures for pyiwfm."""

from __future__ import annotations

from pyiwfm.core.aggregation import AggregationMethod, DataAggregator, create_aggregator_from_grid

# Cross-section support
from pyiwfm.core.cross_section import CrossSection, CrossSectionExtractor
from pyiwfm.core.exceptions import (
    IWFMIOError,
    MeshError,
    PyIWFMError,
    StratigraphyError,
    ValidationError,
)

# Interpolation support
from pyiwfm.core.interpolation import (
    FEInterpolator,
    InterpolationResult,
    ParametricGrid,
    fe_interpolate,
    fe_interpolate_at_element,
    find_containing_element,
    interpolation_coefficients,
    point_in_element,
)
from pyiwfm.core.mesh import AppGrid, Element, Face, Node, Subregion
from pyiwfm.core.model import IWFMModel
from pyiwfm.core.query import ModelQueryAPI, TimeSeries
from pyiwfm.core.stratigraphy import Stratigraphy

# Zone and aggregation support
from pyiwfm.core.zones import Zone, ZoneDefinition

__all__ = [
    # Mesh classes
    "Node",
    "Element",
    "Face",
    "Subregion",
    "AppGrid",
    # Model classes
    "Stratigraphy",
    "IWFMModel",
    # Zone and aggregation
    "Zone",
    "ZoneDefinition",
    "DataAggregator",
    "AggregationMethod",
    "create_aggregator_from_grid",
    "ModelQueryAPI",
    "TimeSeries",
    # Interpolation
    "FEInterpolator",
    "ParametricGrid",
    "InterpolationResult",
    "point_in_element",
    "find_containing_element",
    "interpolation_coefficients",
    "fe_interpolate",
    "fe_interpolate_at_element",
    # Cross-section
    "CrossSection",
    "CrossSectionExtractor",
    # Exceptions
    "PyIWFMError",
    "MeshError",
    "StratigraphyError",
    "ValidationError",
    "IWFMIOError",
]
