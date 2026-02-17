"""Model components for pyiwfm (mirrors IWFM Fortran packages)."""

from __future__ import annotations

from pyiwfm.components.connectors import (
    LakeGWConnection,
    LakeGWConnector,
    StreamGWConnection,
    StreamGWConnector,
    StreamLakeConnection,
    StreamLakeConnector,
)
from pyiwfm.components.groundwater import (
    AppGW,
    AquiferParameters,
    BoundaryCondition,
    ElementPumping,
    HydrographLocation,
    Subsidence,
    TileDrain,
    Well,
)
from pyiwfm.components.lake import (
    AppLake,
    Lake,
    LakeElement,
    LakeOutflow,
    LakeRating,
)
from pyiwfm.components.rootzone import (
    CropType,
    ElementLandUse,
    LandUseType,
    RootZone,
    SoilParameters,
)
from pyiwfm.components.small_watershed import (
    AppSmallWatershed,
    WatershedGWNode,
    WatershedUnit,
)
from pyiwfm.components.stream import (
    AppStream,
    Bypass,
    Diversion,
    StreamRating,
    StrmNode,
    StrmReach,
)
from pyiwfm.components.unsaturated_zone import (
    AppUnsatZone,
    UnsatZoneElement,
    UnsatZoneLayer,
)

__all__ = [
    # Stream
    "StreamRating",
    "StrmNode",
    "StrmReach",
    "Diversion",
    "Bypass",
    "AppStream",
    # Groundwater
    "Well",
    "ElementPumping",
    "BoundaryCondition",
    "TileDrain",
    "Subsidence",
    "AquiferParameters",
    "HydrographLocation",
    "AppGW",
    # Lake
    "LakeRating",
    "LakeElement",
    "LakeOutflow",
    "Lake",
    "AppLake",
    # Root Zone
    "LandUseType",
    "CropType",
    "SoilParameters",
    "ElementLandUse",
    "RootZone",
    # Connectors
    "StreamGWConnection",
    "StreamGWConnector",
    "LakeGWConnection",
    "LakeGWConnector",
    "StreamLakeConnection",
    "StreamLakeConnector",
    # Small Watershed
    "WatershedGWNode",
    "WatershedUnit",
    "AppSmallWatershed",
    # Unsaturated Zone
    "UnsatZoneLayer",
    "UnsatZoneElement",
    "AppUnsatZone",
]
