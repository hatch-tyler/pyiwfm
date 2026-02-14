"""Model components for pyiwfm (mirrors IWFM Fortran packages)."""

from __future__ import annotations

from pyiwfm.components.stream import (
    StreamRating,
    StrmNode,
    StrmReach,
    Diversion,
    Bypass,
    AppStream,
)
from pyiwfm.components.groundwater import (
    Well,
    ElementPumping,
    BoundaryCondition,
    TileDrain,
    Subsidence,
    AquiferParameters,
    HydrographLocation,
    AppGW,
)
from pyiwfm.components.lake import (
    LakeRating,
    LakeElement,
    LakeOutflow,
    Lake,
    AppLake,
)
from pyiwfm.components.rootzone import (
    LandUseType,
    CropType,
    SoilParameters,
    ElementLandUse,
    RootZone,
)
from pyiwfm.components.connectors import (
    StreamGWConnection,
    StreamGWConnector,
    LakeGWConnection,
    LakeGWConnector,
    StreamLakeConnection,
    StreamLakeConnector,
)
from pyiwfm.components.small_watershed import (
    WatershedGWNode,
    WatershedUnit,
    AppSmallWatershed,
)
from pyiwfm.components.unsaturated_zone import (
    UnsatZoneLayer,
    UnsatZoneElement,
    AppUnsatZone,
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
