"""PEST++ observation manager for IWFM models.

This module provides the IWFMObservationManager class for managing observations
in PEST++ calibration, uncertainty analysis, and optimization setups.

The manager supports:
- Groundwater head and drawdown observations
- Stream flow and stage observations
- Lake level and storage observations
- Water budget component observations
- Derived observations from expressions
- Flexible weight calculation strategies
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

import pandas as pd

from pyiwfm.runner.pest_observations import (
    IWFMObservationType,
    IWFMObservation,
    IWFMObservationGroup,
    ObservationLocation,
    WeightStrategy,
    DerivedObservation,
)


@dataclass
class WellInfo:
    """Information about an observation well.

    Attributes
    ----------
    well_id : str
        Unique well identifier.
    x : float
        X coordinate.
    y : float
        Y coordinate.
    screen_top : float | None
        Top of well screen elevation.
    screen_bottom : float | None
        Bottom of well screen elevation.
    layer : int | None
        Model layer (if known).
    node_id : int | None
        Associated model node.
    name : str | None
        Well name.
    """

    well_id: str
    x: float
    y: float
    screen_top: float | None = None
    screen_bottom: float | None = None
    layer: int | None = None
    node_id: int | None = None
    name: str | None = None

    def to_location(self) -> ObservationLocation:
        """Convert to ObservationLocation.

        Returns
        -------
        ObservationLocation
            Location object for this well.
        """
        z = None
        if self.screen_top is not None and self.screen_bottom is not None:
            z = (self.screen_top + self.screen_bottom) / 2
        return ObservationLocation(
            x=self.x,
            y=self.y,
            z=z,
            node_id=self.node_id,
            layer=self.layer,
        )


@dataclass
class GageInfo:
    """Information about a stream gage.

    Attributes
    ----------
    gage_id : str
        Unique gage identifier.
    reach_id : int | None
        Stream reach ID.
    node_id : int | None
        Stream node ID.
    x : float | None
        X coordinate.
    y : float | None
        Y coordinate.
    name : str | None
        Gage name.
    """

    gage_id: str
    reach_id: int | None = None
    node_id: int | None = None
    x: float | None = None
    y: float | None = None
    name: str | None = None

    def to_location(self) -> ObservationLocation | None:
        """Convert to ObservationLocation.

        Returns
        -------
        ObservationLocation | None
            Location object for this gage, or None if no coordinates.
        """
        if self.x is None or self.y is None:
            return None
        return ObservationLocation(
            x=self.x,
            y=self.y,
            node_id=self.node_id,
            reach_id=self.reach_id,
        )


class IWFMObservationManager:
    """Manages all observations for an IWFM PEST++ setup.

    This class provides methods for adding various types of observations,
    managing observation weights, and exporting to PEST++ format.

    Parameters
    ----------
    model : Any
        IWFM model instance (optional, for auto-detection of locations).

    Examples
    --------
    >>> om = IWFMObservationManager()
    >>> # Add head observations from files
    >>> om.add_head_observations(
    ...     wells="observation_wells.csv",
    ...     observed_data="head_timeseries.csv",
    ...     weight_strategy="inverse_variance",
    ... )
    >>> # Add streamflow observations
    >>> om.add_streamflow_observations(
    ...     gages="stream_gages.csv",
    ...     observed_data="flow_timeseries.csv",
    ...     transform="log",
    ... )
    >>> # Balance weights
    >>> om.balance_observation_groups({"head": 0.5, "flow": 0.5})
    >>> # Export to dataframe
    >>> df = om.to_dataframe()
    """

    def __init__(self, model: Any = None):
        """Initialize the observation manager.

        Parameters
        ----------
        model : Any
            IWFM model instance (optional).
        """
        self.model = model
        self._observations: dict[str, IWFMObservation] = {}
        self._observation_groups: dict[str, IWFMObservationGroup] = {}
        self._derived_observations: dict[str, DerivedObservation] = {}

        # Create default groups
        self._create_default_groups()

    def _create_default_groups(self) -> None:
        """Create default observation groups for each type."""
        default_groups = [
            ("head", IWFMObservationType.HEAD),
            ("drawdown", IWFMObservationType.DRAWDOWN),
            ("hdiff", IWFMObservationType.HEAD_DIFFERENCE),
            ("flow", IWFMObservationType.STREAM_FLOW),
            ("stage", IWFMObservationType.STREAM_STAGE),
            ("gain_loss", IWFMObservationType.STREAM_GAIN_LOSS),
            ("lake_level", IWFMObservationType.LAKE_LEVEL),
            ("gwbud", IWFMObservationType.GW_BUDGET),
            ("strbud", IWFMObservationType.STREAM_BUDGET),
            ("rzbud", IWFMObservationType.ROOTZONE_BUDGET),
            ("subsidence", IWFMObservationType.SUBSIDENCE),
        ]
        for name, obs_type in default_groups:
            self._observation_groups[name] = IWFMObservationGroup(
                name=name,
                obs_type=obs_type,
            )

    # =========================================================================
    # Head Observations
    # =========================================================================

    def add_head_observations(
        self,
        wells: "pd.DataFrame | Path | str | list[WellInfo]",
        observed_data: "pd.DataFrame | Path | str",
        layers: int | list[int] | str = "auto",
        weight_strategy: str | WeightStrategy = WeightStrategy.EQUAL,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        frequency: str | None = None,
        group_by: str = "well",
        group_name: str | None = None,
        obs_name_format: str = "{well}_{date}",
        error_std: float | None = None,
    ) -> list[IWFMObservation]:
        """Add groundwater head observations.

        Parameters
        ----------
        wells : pd.DataFrame | Path | str | list[WellInfo]
            Well information with columns: well_id, x, y, (optional) screen_top,
            screen_bottom, layer. Or list of WellInfo objects.
        observed_data : pd.DataFrame | Path | str
            Observed head data with columns: well_id, datetime, head.
            Or path to CSV file.
        layers : int | list[int] | str
            Layer(s) for observations. "auto" determines from screen depth.
        weight_strategy : str | WeightStrategy
            Weight calculation strategy.
        start_date : datetime | None
            Filter observations to start at this date.
        end_date : datetime | None
            Filter observations to end at this date.
        frequency : str | None
            Resample frequency (e.g., "MS" for monthly start).
        group_by : str
            How to group observations: "well", "layer", "time", "all".
        group_name : str | None
            Custom group name. Defaults based on group_by.
        obs_name_format : str
            Format string for observation names.
        error_std : float | None
            Measurement error standard deviation.

        Returns
        -------
        list[IWFMObservation]
            List of created observations.
        """
        # Parse inputs
        well_list = self._parse_wells(wells)
        obs_df = self._parse_timeseries_data(observed_data, "head")

        # Filter by date range
        if start_date is not None:
            obs_df = obs_df[obs_df["datetime"] >= start_date]
        if end_date is not None:
            obs_df = obs_df[obs_df["datetime"] <= end_date]

        # Resample if requested
        if frequency is not None:
            obs_df = self._resample_timeseries(obs_df, "well_id", "head", frequency)

        # Determine weight strategy
        if isinstance(weight_strategy, str):
            weight_strategy = WeightStrategy(weight_strategy)

        # Create observations
        created_obs = []
        well_dict = {w.well_id: w for w in well_list}

        for _, row in obs_df.iterrows():
            well_id = row["well_id"]
            if well_id not in well_dict:
                continue

            well = well_dict[well_id]
            obs_time = row["datetime"]
            value = row["head"]

            # Determine layer
            if layers == "auto":
                layer = self._determine_layer_from_screen(well)
            elif isinstance(layers, int):
                layer = layers
            else:
                layer = well.layer

            # Create observation name
            date_str = obs_time.strftime("%Y%m%d") if isinstance(obs_time, datetime) else str(obs_time)
            obs_name = obs_name_format.format(
                well=well_id[:12],  # Truncate for PEST name limits
                date=date_str,
                layer=layer or 1,
            )
            # Ensure unique and valid name
            obs_name = self._make_valid_obs_name(obs_name)

            # Determine group
            if group_name:
                grp_name = group_name
            elif group_by == "well":
                grp_name = f"head_{well_id[:8]}"
            elif group_by == "layer":
                grp_name = f"head_l{layer or 1}"
            elif group_by == "time":
                grp_name = f"head_{date_str[:6]}"
            else:
                grp_name = "head"

            # Ensure group exists
            if grp_name not in self._observation_groups:
                self._observation_groups[grp_name] = IWFMObservationGroup(
                    name=grp_name,
                    obs_type=IWFMObservationType.HEAD,
                )

            # Create observation
            location = well.to_location()
            if layer:
                location.layer = layer

            obs = IWFMObservation(
                name=obs_name,
                value=float(value),
                weight=1.0,
                group=grp_name,
                obs_type=IWFMObservationType.HEAD,
                datetime=obs_time if isinstance(obs_time, datetime) else None,
                location=location,
                error_std=error_std or IWFMObservationType.HEAD.typical_error,
                metadata={"well_id": well_id},
            )

            # Calculate weight
            obs.weight = obs.calculate_weight(weight_strategy)

            self._observations[obs_name] = obs
            self._observation_groups[grp_name].observations.append(obs)
            created_obs.append(obs)

        return created_obs

    def add_drawdown_observations(
        self,
        wells: "pd.DataFrame | Path | str | list[WellInfo]",
        observed_data: "pd.DataFrame | Path | str",
        reference_date: datetime | None = None,
        reference_values: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> list[IWFMObservation]:
        """Add drawdown observations (change from reference).

        Parameters
        ----------
        wells : pd.DataFrame | Path | str | list[WellInfo]
            Well information.
        observed_data : pd.DataFrame | Path | str
            Observed head data.
        reference_date : datetime | None
            Date to use as reference (drawdown = head - head_at_reference).
        reference_values : dict[str, float] | None
            Dictionary of well_id -> reference head values.
        **kwargs : Any
            Additional arguments passed to add_head_observations.

        Returns
        -------
        list[IWFMObservation]
            List of created drawdown observations.
        """
        # Parse observed data
        obs_df = self._parse_timeseries_data(observed_data, "head")

        # Determine reference values
        if reference_values is None:
            reference_values = {}
            if reference_date is not None:
                # Get values closest to reference date
                for well_id in obs_df["well_id"].unique():
                    well_data = obs_df[obs_df["well_id"] == well_id]
                    well_data = well_data.sort_values("datetime")
                    # Find closest to reference date
                    idx = (well_data["datetime"] - reference_date).abs().idxmin()
                    reference_values[well_id] = well_data.loc[idx, "head"]
            else:
                # Use first value as reference
                for well_id in obs_df["well_id"].unique():
                    well_data = obs_df[obs_df["well_id"] == well_id].sort_values("datetime")
                    if len(well_data) > 0:
                        reference_values[well_id] = well_data.iloc[0]["head"]

        # Calculate drawdown
        def calc_drawdown(row):
            ref = reference_values.get(row["well_id"], 0)
            return ref - row["head"]

        obs_df = obs_df.copy()
        obs_df["head"] = obs_df.apply(calc_drawdown, axis=1)

        # Update kwargs for drawdown
        kwargs.setdefault("group_name", "drawdown")
        kwargs.setdefault("obs_name_format", "{well}_dd_{date}")

        # Create observations
        obs_list = self.add_head_observations(wells, obs_df, **kwargs)

        # Update observation type
        for obs in obs_list:
            obs.obs_type = IWFMObservationType.DRAWDOWN
            obs.metadata["reference_value"] = reference_values.get(
                obs.metadata.get("well_id"), 0
            )

        return obs_list

    def add_head_difference_observations(
        self,
        well_pairs: list[tuple[str, str]],
        observed_data: "pd.DataFrame | Path | str",
        weight: float = 1.0,
        group_name: str = "hdiff",
        **kwargs: Any,
    ) -> list[IWFMObservation]:
        """Add head difference observations between well pairs.

        Parameters
        ----------
        well_pairs : list[tuple[str, str]]
            List of (well_id_1, well_id_2) pairs. Difference = head1 - head2.
        observed_data : pd.DataFrame | Path | str
            Observed head data with columns: well_id, datetime, head.
        weight : float
            Observation weight.
        group_name : str
            Observation group name.
        **kwargs : Any
            Additional arguments.

        Returns
        -------
        list[IWFMObservation]
            List of created head difference observations.
        """
        obs_df = self._parse_timeseries_data(observed_data, "head")

        # Ensure group exists
        if group_name not in self._observation_groups:
            self._observation_groups[group_name] = IWFMObservationGroup(
                name=group_name,
                obs_type=IWFMObservationType.HEAD_DIFFERENCE,
            )

        created_obs = []

        # Get unique times
        times = obs_df["datetime"].unique()

        for well1, well2 in well_pairs:
            data1 = obs_df[obs_df["well_id"] == well1].set_index("datetime")
            data2 = obs_df[obs_df["well_id"] == well2].set_index("datetime")

            # Find common times
            common_times = set(data1.index) & set(data2.index)

            for obs_time in sorted(common_times):
                head1 = data1.loc[obs_time, "head"]
                head2 = data2.loc[obs_time, "head"]
                diff = head1 - head2

                date_str = obs_time.strftime("%Y%m%d") if isinstance(obs_time, datetime) else str(obs_time)
                obs_name = f"hdif_{well1[:6]}_{well2[:6]}_{date_str}"
                obs_name = self._make_valid_obs_name(obs_name)

                obs = IWFMObservation(
                    name=obs_name,
                    value=float(diff),
                    weight=weight,
                    group=group_name,
                    obs_type=IWFMObservationType.HEAD_DIFFERENCE,
                    datetime=obs_time if isinstance(obs_time, datetime) else None,
                    metadata={"well_1": well1, "well_2": well2},
                )

                self._observations[obs_name] = obs
                self._observation_groups[group_name].observations.append(obs)
                created_obs.append(obs)

        return created_obs

    # =========================================================================
    # Stream Observations
    # =========================================================================

    def add_streamflow_observations(
        self,
        gages: "pd.DataFrame | Path | str | list[GageInfo]",
        observed_data: "pd.DataFrame | Path | str",
        weight_strategy: str | WeightStrategy = WeightStrategy.EQUAL,
        transform: str = "none",
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        frequency: str | None = None,
        group_name: str | None = None,
        obs_name_format: str = "{gage}_{date}",
        error_std: float | None = None,
    ) -> list[IWFMObservation]:
        """Add stream discharge observations.

        Parameters
        ----------
        gages : pd.DataFrame | Path | str | list[GageInfo]
            Gage information with columns: gage_id, reach_id or node_id.
        observed_data : pd.DataFrame | Path | str
            Observed flow data with columns: gage_id, datetime, flow.
        weight_strategy : str | WeightStrategy
            Weight calculation strategy.
        transform : str
            Transform for flow: 'none', 'log', 'sqrt'.
        start_date : datetime | None
            Filter start date.
        end_date : datetime | None
            Filter end date.
        frequency : str | None
            Resample frequency.
        group_name : str | None
            Custom group name.
        obs_name_format : str
            Format for observation names.
        error_std : float | None
            Measurement error.

        Returns
        -------
        list[IWFMObservation]
            List of created observations.
        """
        # Parse inputs
        gage_list = self._parse_gages(gages)
        obs_df = self._parse_timeseries_data(observed_data, "flow")

        # Filter by date range
        if start_date is not None:
            obs_df = obs_df[obs_df["datetime"] >= start_date]
        if end_date is not None:
            obs_df = obs_df[obs_df["datetime"] <= end_date]

        # Resample if requested
        if frequency is not None:
            obs_df = self._resample_timeseries(obs_df, "gage_id", "flow", frequency)

        # Determine weight strategy
        if isinstance(weight_strategy, str):
            weight_strategy = WeightStrategy(weight_strategy)

        # Determine group name
        grp_name = group_name or "flow"

        # Ensure group exists
        if grp_name not in self._observation_groups:
            self._observation_groups[grp_name] = IWFMObservationGroup(
                name=grp_name,
                obs_type=IWFMObservationType.STREAM_FLOW,
            )

        # Create observations
        created_obs = []
        gage_dict = {g.gage_id: g for g in gage_list}

        for _, row in obs_df.iterrows():
            gage_id = row["gage_id"]
            gage = gage_dict.get(gage_id)

            obs_time = row["datetime"]
            value = row["flow"]

            # Skip non-positive values for log transform
            if transform == "log" and value <= 0:
                continue

            # Create observation name
            date_str = obs_time.strftime("%Y%m%d") if isinstance(obs_time, datetime) else str(obs_time)
            obs_name = obs_name_format.format(
                gage=gage_id[:12] if gage_id else "gage",
                date=date_str,
            )
            obs_name = self._make_valid_obs_name(obs_name)

            # Get location
            location = gage.to_location() if gage else None

            obs = IWFMObservation(
                name=obs_name,
                value=float(value),
                weight=1.0,
                group=grp_name,
                obs_type=IWFMObservationType.STREAM_FLOW,
                datetime=obs_time if isinstance(obs_time, datetime) else None,
                location=location,
                transform=transform,
                error_std=error_std,
                metadata={"gage_id": gage_id},
            )

            # Calculate weight
            obs.weight = obs.calculate_weight(weight_strategy)

            self._observations[obs_name] = obs
            self._observation_groups[grp_name].observations.append(obs)
            created_obs.append(obs)

        return created_obs

    def add_stream_stage_observations(
        self,
        gages: "pd.DataFrame | Path | str | list[GageInfo]",
        observed_data: "pd.DataFrame | Path | str",
        **kwargs: Any,
    ) -> list[IWFMObservation]:
        """Add stream stage observations.

        Parameters
        ----------
        gages : pd.DataFrame | Path | str | list[GageInfo]
            Gage information.
        observed_data : pd.DataFrame | Path | str
            Observed stage data with columns: gage_id, datetime, stage.
        **kwargs : Any
            Additional arguments.

        Returns
        -------
        list[IWFMObservation]
            List of created observations.
        """
        # Parse data - expect "stage" column
        obs_df = self._parse_timeseries_data(observed_data, "stage")

        # Rename column for generic flow processing
        obs_df = obs_df.rename(columns={"stage": "flow"})

        kwargs.setdefault("group_name", "stage")
        kwargs.setdefault("transform", "none")

        obs_list = self.add_streamflow_observations(gages, obs_df, **kwargs)

        # Update observation type
        for obs in obs_list:
            obs.obs_type = IWFMObservationType.STREAM_STAGE

        return obs_list

    def add_gain_loss_observations(
        self,
        reaches: list[int],
        observed_data: "pd.DataFrame | Path | str",
        weight: float = 1.0,
        group_name: str = "gain_loss",
        **kwargs: Any,
    ) -> list[IWFMObservation]:
        """Add stream gain/loss observations.

        Parameters
        ----------
        reaches : list[int]
            List of reach IDs.
        observed_data : pd.DataFrame | Path | str
            Observed data with columns: reach_id, datetime, gain_loss.
        weight : float
            Observation weight.
        group_name : str
            Observation group name.
        **kwargs : Any
            Additional arguments.

        Returns
        -------
        list[IWFMObservation]
            List of created observations.
        """
        obs_df = self._parse_timeseries_data(observed_data, "gain_loss")

        # Filter to specified reaches
        obs_df = obs_df[obs_df["reach_id"].isin(reaches)]

        # Ensure group exists
        if group_name not in self._observation_groups:
            self._observation_groups[group_name] = IWFMObservationGroup(
                name=group_name,
                obs_type=IWFMObservationType.STREAM_GAIN_LOSS,
            )

        created_obs = []

        for _, row in obs_df.iterrows():
            reach_id = row["reach_id"]
            obs_time = row["datetime"]
            value = row["gain_loss"]

            date_str = obs_time.strftime("%Y%m%d") if isinstance(obs_time, datetime) else str(obs_time)
            obs_name = f"sgl_r{reach_id}_{date_str}"
            obs_name = self._make_valid_obs_name(obs_name)

            location = ObservationLocation(x=0, y=0, reach_id=reach_id)

            obs = IWFMObservation(
                name=obs_name,
                value=float(value),
                weight=weight,
                group=group_name,
                obs_type=IWFMObservationType.STREAM_GAIN_LOSS,
                datetime=obs_time if isinstance(obs_time, datetime) else None,
                location=location,
                metadata={"reach_id": reach_id},
            )

            self._observations[obs_name] = obs
            self._observation_groups[group_name].observations.append(obs)
            created_obs.append(obs)

        return created_obs

    # =========================================================================
    # Lake Observations
    # =========================================================================

    def add_lake_observations(
        self,
        lakes: list[int] | str = "all",
        observed_data: "pd.DataFrame | Path | str | None" = None,
        obs_type: str = "level",
        weight: float = 1.0,
        group_name: str | None = None,
        **kwargs: Any,
    ) -> list[IWFMObservation]:
        """Add lake level or storage observations.

        Parameters
        ----------
        lakes : list[int] | str
            Lake IDs or "all".
        observed_data : pd.DataFrame | Path | str | None
            Observed data with columns: lake_id, datetime, value.
        obs_type : str
            "level" or "storage".
        weight : float
            Observation weight.
        group_name : str | None
            Custom group name.
        **kwargs : Any
            Additional arguments.

        Returns
        -------
        list[IWFMObservation]
            List of created observations.
        """
        if observed_data is None:
            return []

        # Determine observation type
        iwfm_obs_type = (
            IWFMObservationType.LAKE_LEVEL if obs_type == "level"
            else IWFMObservationType.LAKE_STORAGE
        )

        grp_name = group_name or f"lake_{obs_type}"

        # Ensure group exists
        if grp_name not in self._observation_groups:
            self._observation_groups[grp_name] = IWFMObservationGroup(
                name=grp_name,
                obs_type=iwfm_obs_type,
            )

        # Parse data
        obs_df = self._parse_timeseries_data(observed_data, "value")

        # Filter lakes if needed
        if lakes != "all":
            obs_df = obs_df[obs_df["lake_id"].isin(lakes)]

        created_obs = []

        for _, row in obs_df.iterrows():
            lake_id = row["lake_id"]
            obs_time = row["datetime"]
            value = row["value"]

            date_str = obs_time.strftime("%Y%m%d") if isinstance(obs_time, datetime) else str(obs_time)
            obs_name = f"lak{lake_id}_{obs_type[:3]}_{date_str}"
            obs_name = self._make_valid_obs_name(obs_name)

            location = ObservationLocation(x=0, y=0, lake_id=lake_id)

            obs = IWFMObservation(
                name=obs_name,
                value=float(value),
                weight=weight,
                group=grp_name,
                obs_type=iwfm_obs_type,
                datetime=obs_time if isinstance(obs_time, datetime) else None,
                location=location,
                metadata={"lake_id": lake_id},
            )

            self._observations[obs_name] = obs
            self._observation_groups[grp_name].observations.append(obs)
            created_obs.append(obs)

        return created_obs

    # =========================================================================
    # Budget Observations
    # =========================================================================

    def add_budget_observations(
        self,
        budget_type: str,
        components: list[str] | None = None,
        locations: list[int] | str = "all",
        aggregate: str = "sum",
        observed_data: "pd.DataFrame | Path | str | None" = None,
        weight: float = 1.0,
        group_name: str | None = None,
        **kwargs: Any,
    ) -> list[IWFMObservation]:
        """Add water budget component observations.

        Parameters
        ----------
        budget_type : str
            Budget type: "gw", "stream", "rootzone", "lake".
        components : list[str] | None
            Budget components to observe.
        locations : list[int] | str
            Location IDs or "all".
        aggregate : str
            Aggregation method: "sum", "mean", "by_location".
        observed_data : pd.DataFrame | Path | str | None
            Observed budget data.
        weight : float
            Observation weight.
        group_name : str | None
            Custom group name.
        **kwargs : Any
            Additional arguments.

        Returns
        -------
        list[IWFMObservation]
            List of created observations.
        """
        # Map budget type to observation type
        budget_obs_types = {
            "gw": IWFMObservationType.GW_BUDGET,
            "stream": IWFMObservationType.STREAM_BUDGET,
            "rootzone": IWFMObservationType.ROOTZONE_BUDGET,
            "lake": IWFMObservationType.LAKE_BUDGET,
        }

        if budget_type not in budget_obs_types:
            raise ValueError(f"Invalid budget type: {budget_type}. "
                           f"Must be one of: {list(budget_obs_types.keys())}")

        iwfm_obs_type = budget_obs_types[budget_type]
        grp_name = group_name or f"{budget_type}bud"

        # Ensure group exists
        if grp_name not in self._observation_groups:
            self._observation_groups[grp_name] = IWFMObservationGroup(
                name=grp_name,
                obs_type=iwfm_obs_type,
            )

        if observed_data is None:
            return []

        # Parse data
        obs_df = self._parse_timeseries_data(observed_data, "value")

        # Filter components if specified
        if components is not None and "component" in obs_df.columns:
            obs_df = obs_df[obs_df["component"].isin(components)]

        # Filter locations
        if locations != "all" and "location_id" in obs_df.columns:
            obs_df = obs_df[obs_df["location_id"].isin(locations)]

        created_obs = []

        # Aggregate if needed
        if aggregate == "sum" and "location_id" in obs_df.columns:
            # Sum across locations for each time and component
            group_cols = ["datetime"]
            if "component" in obs_df.columns:
                group_cols.append("component")
            obs_df = obs_df.groupby(group_cols)["value"].sum().reset_index()
        elif aggregate == "mean" and "location_id" in obs_df.columns:
            group_cols = ["datetime"]
            if "component" in obs_df.columns:
                group_cols.append("component")
            obs_df = obs_df.groupby(group_cols)["value"].mean().reset_index()

        for _, row in obs_df.iterrows():
            obs_time = row["datetime"]
            value = row["value"]
            component = row.get("component", "total")
            location_id = row.get("location_id")

            date_str = obs_time.strftime("%Y%m%d") if isinstance(obs_time, datetime) else str(obs_time)

            if location_id is not None:
                obs_name = f"{budget_type}_{component[:6]}_{location_id}_{date_str}"
            else:
                obs_name = f"{budget_type}_{component[:6]}_{date_str}"
            obs_name = self._make_valid_obs_name(obs_name)

            metadata = {"budget_type": budget_type, "component": component}
            if location_id is not None:
                metadata["location_id"] = location_id

            obs = IWFMObservation(
                name=obs_name,
                value=float(value),
                weight=weight,
                group=grp_name,
                obs_type=iwfm_obs_type,
                datetime=obs_time if isinstance(obs_time, datetime) else None,
                metadata=metadata,
            )

            self._observations[obs_name] = obs
            self._observation_groups[grp_name].observations.append(obs)
            created_obs.append(obs)

        return created_obs

    # =========================================================================
    # Derived Observations
    # =========================================================================

    def add_derived_observation(
        self,
        expression: str,
        obs_names: list[str],
        result_name: str,
        target_value: float = 0.0,
        weight: float = 1.0,
        group: str = "derived",
    ) -> DerivedObservation:
        """Add derived observation from expression.

        Parameters
        ----------
        expression : str
            Mathematical expression using observation names.
        obs_names : list[str]
            Names of observations used in the expression.
        result_name : str
            Name for the derived observation.
        target_value : float
            Target value for the derived quantity.
        weight : float
            Observation weight.
        group : str
            Observation group name.

        Returns
        -------
        DerivedObservation
            The created derived observation.

        Examples
        --------
        >>> # Mass balance closure
        >>> om.add_derived_observation(
        ...     expression="inflow - outflow - storage_change",
        ...     obs_names=["total_inflow", "total_outflow", "delta_storage"],
        ...     result_name="mass_balance_error",
        ...     target_value=0.0,
        ...     weight=10.0,
        ... )
        """
        # Validate observation names exist
        for name in obs_names:
            if name not in self._observations:
                raise ValueError(f"Observation not found: {name}")

        derived = DerivedObservation(
            name=result_name,
            expression=expression,
            source_observations=obs_names,
            target_value=target_value,
            weight=weight,
            group=group,
        )

        self._derived_observations[result_name] = derived
        return derived

    # =========================================================================
    # Weight Management
    # =========================================================================

    def set_group_weights(
        self,
        group: str,
        weight: float | str = "auto",
        contribution: float | None = None,
    ) -> None:
        """Set weights for an observation group.

        Parameters
        ----------
        group : str
            Group name.
        weight : float | str
            Weight value or "auto" to calculate from contribution.
        contribution : float | None
            Target contribution to objective function (0-1).
        """
        if group not in self._observation_groups:
            raise ValueError(f"Group not found: {group}")

        grp = self._observation_groups[group]

        if weight == "auto" and contribution is not None:
            grp.target_contribution = contribution
            # Will be calculated in balance_observation_groups
        elif isinstance(weight, (int, float)):
            for obs in grp.observations:
                obs.weight = float(weight)

    def balance_observation_groups(
        self,
        target_contributions: dict[str, float] | None = None,
    ) -> None:
        """Balance weights so groups contribute equally or as specified.

        Parameters
        ----------
        target_contributions : dict[str, float] | None
            Dictionary mapping group names to target contributions (0-1).
            If None, groups contribute equally.
        """
        # Get groups with observations
        active_groups = {
            name: grp for name, grp in self._observation_groups.items()
            if len(grp.observations) > 0
        }

        if len(active_groups) == 0:
            return

        # Determine target contributions
        if target_contributions is None:
            # Equal contribution
            n_groups = len(active_groups)
            target_contributions = {name: 1.0 / n_groups for name in active_groups}
        else:
            # Fill in missing groups with remaining contribution
            specified_total = sum(target_contributions.values())
            unspecified = set(active_groups.keys()) - set(target_contributions.keys())
            if unspecified and specified_total < 1.0:
                remaining = (1.0 - specified_total) / len(unspecified)
                for name in unspecified:
                    target_contributions[name] = remaining

        # Normalize to sum to 1
        total = sum(target_contributions.values())
        if total > 0:
            target_contributions = {k: v / total for k, v in target_contributions.items()}

        # Calculate scaling factors
        # Total phi = sum of (weight * residual)^2
        # For balancing, assume residual = 1
        # Current contribution_i = sum(weight_i^2) / sum_all(weight^2)
        # Want: sum(scale * weight_i)^2 / total = target_contribution_i

        total_contribution = sum(grp.contribution for grp in active_groups.values())

        for name, grp in active_groups.items():
            if name not in target_contributions:
                continue

            target = target_contributions[name]
            current = grp.contribution / total_contribution if total_contribution > 0 else 0

            if current > 0 and target > 0:
                # Scale factor to achieve target contribution
                scale = np.sqrt(target / current)
                grp.scale_weights(scale)
            elif target > 0:
                # Set uniform weight to achieve target
                n_obs = len(grp.observations)
                if n_obs > 0:
                    weight = np.sqrt(target * total_contribution / n_obs)
                    for obs in grp.observations:
                        obs.weight = weight

    def apply_temporal_weights(
        self,
        decay_factor: float = 0.95,
        reference_date: datetime | None = None,
    ) -> None:
        """Apply temporal decay to observation weights.

        Recent observations are weighted higher than older ones.

        Parameters
        ----------
        decay_factor : float
            Annual decay factor (0-1). Weight = decay_factor^(years_from_reference).
        reference_date : datetime | None
            Reference date. If None, uses most recent observation date.
        """
        # Find reference date if not specified
        if reference_date is None:
            max_date = None
            for obs in self._observations.values():
                if obs.datetime is not None:
                    if max_date is None or obs.datetime > max_date:
                        max_date = obs.datetime
            reference_date = max_date

        if reference_date is None:
            return

        # Apply decay
        for obs in self._observations.values():
            if obs.datetime is not None:
                years = (reference_date - obs.datetime).days / 365.0
                decay = decay_factor ** max(0, years)
                obs.weight *= decay

    # =========================================================================
    # Access Methods
    # =========================================================================

    def get_observation(self, name: str) -> IWFMObservation | None:
        """Get observation by name.

        Parameters
        ----------
        name : str
            Observation name.

        Returns
        -------
        IWFMObservation | None
            The observation, or None if not found.
        """
        return self._observations.get(name)

    def get_observations_by_type(
        self,
        obs_type: IWFMObservationType,
    ) -> list[IWFMObservation]:
        """Get all observations of a specific type.

        Parameters
        ----------
        obs_type : IWFMObservationType
            Observation type.

        Returns
        -------
        list[IWFMObservation]
            Matching observations.
        """
        return [
            obs for obs in self._observations.values()
            if obs.obs_type == obs_type
        ]

    def get_observations_by_group(self, group: str) -> list[IWFMObservation]:
        """Get all observations in a group.

        Parameters
        ----------
        group : str
            Group name.

        Returns
        -------
        list[IWFMObservation]
            Observations in the group.
        """
        if group in self._observation_groups:
            return list(self._observation_groups[group].observations)
        return []

    def get_all_observations(self) -> list[IWFMObservation]:
        """Get all observations.

        Returns
        -------
        list[IWFMObservation]
            All observations.
        """
        return list(self._observations.values())

    def get_observation_group(self, name: str) -> IWFMObservationGroup | None:
        """Get observation group by name.

        Parameters
        ----------
        name : str
            Group name.

        Returns
        -------
        IWFMObservationGroup | None
            The group, or None if not found.
        """
        return self._observation_groups.get(name)

    def get_all_groups(self) -> list[IWFMObservationGroup]:
        """Get all observation groups with observations.

        Returns
        -------
        list[IWFMObservationGroup]
            All groups that have at least one observation.
        """
        return [
            grp for grp in self._observation_groups.values()
            if len(grp.observations) > 0
        ]

    # =========================================================================
    # Export Methods
    # =========================================================================

    def to_dataframe(self) -> "pd.DataFrame":
        """Export all observations to a DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with observation data.

        Raises
        ------
        ImportError
            If pandas is not available.
        """
        records = []
        for obs in self._observations.values():
            record = obs.to_dict()
            records.append(record)

        return pd.DataFrame(records)

    def from_dataframe(self, df: "pd.DataFrame") -> None:
        """Load observations from a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with observation data.
        """
        for _, row in df.iterrows():
            obs_type = None
            if "obs_type" in row and pd.notna(row["obs_type"]):
                obs_type = IWFMObservationType(row["obs_type"])

            obs_datetime = None
            if "datetime" in row and pd.notna(row["datetime"]):
                obs_datetime = pd.to_datetime(row["datetime"])

            location = None
            if "location" in row and isinstance(row["location"], dict):
                location = ObservationLocation(**row["location"])

            obs = IWFMObservation(
                name=row["name"],
                value=row["value"],
                weight=row.get("weight", 1.0),
                group=row.get("group", "default"),
                obs_type=obs_type,
                datetime=obs_datetime,
                location=location,
                transform=row.get("transform", "none"),
            )

            self._observations[obs.name] = obs

            # Add to group
            grp_name = obs.group
            if grp_name not in self._observation_groups:
                self._observation_groups[grp_name] = IWFMObservationGroup(
                    name=grp_name,
                    obs_type=obs_type,
                )
            self._observation_groups[grp_name].observations.append(obs)

    def write_observation_file(self, filepath: Path | str) -> None:
        """Write observations to a CSV file.

        Parameters
        ----------
        filepath : Path | str
            Output file path.
        """
        df = self.to_dataframe()
        df.to_csv(filepath, index=False)

    def read_observation_file(self, filepath: Path | str) -> None:
        """Read observations from a CSV file.

        Parameters
        ----------
        filepath : Path | str
            Input file path.
        """
        df = pd.read_csv(filepath)
        self.from_dataframe(df)

    # =========================================================================
    # Statistics
    # =========================================================================

    @property
    def n_observations(self) -> int:
        """Total number of observations."""
        return len(self._observations)

    @property
    def n_groups(self) -> int:
        """Number of observation groups with observations."""
        return len([g for g in self._observation_groups.values() if len(g) > 0])

    def summary(self) -> dict[str, Any]:
        """Get summary of all observations.

        Returns
        -------
        dict
            Summary statistics.
        """
        return {
            "n_observations": self.n_observations,
            "n_groups": self.n_groups,
            "groups": {
                name: grp.summary()
                for name, grp in self._observation_groups.items()
                if len(grp) > 0
            },
            "n_derived": len(self._derived_observations),
        }

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _parse_wells(
        self,
        wells: "pd.DataFrame | Path | str | list[WellInfo]",
    ) -> list[WellInfo]:
        """Parse well information from various input formats."""
        if isinstance(wells, list) and all(isinstance(w, WellInfo) for w in wells):
            return wells

        if isinstance(wells, (str, Path)):
            wells = pd.read_csv(wells)

        # Convert DataFrame to WellInfo list
        well_list = []
        for _, row in wells.iterrows():
            well = WellInfo(
                well_id=str(row["well_id"]),
                x=row["x"],
                y=row["y"],
                screen_top=row.get("screen_top"),
                screen_bottom=row.get("screen_bottom"),
                layer=row.get("layer"),
                node_id=row.get("node_id"),
                name=row.get("name"),
            )
            well_list.append(well)

        return well_list

    def _parse_gages(
        self,
        gages: "pd.DataFrame | Path | str | list[GageInfo]",
    ) -> list[GageInfo]:
        """Parse gage information from various input formats."""
        if isinstance(gages, list) and all(isinstance(g, GageInfo) for g in gages):
            return gages

        if isinstance(gages, (str, Path)):
            gages = pd.read_csv(gages)

        # Convert DataFrame to GageInfo list
        gage_list = []
        for _, row in gages.iterrows():
            gage = GageInfo(
                gage_id=str(row["gage_id"]),
                reach_id=row.get("reach_id"),
                node_id=row.get("node_id"),
                x=row.get("x"),
                y=row.get("y"),
                name=row.get("name"),
            )
            gage_list.append(gage)

        return gage_list

    def _parse_timeseries_data(
        self,
        data: "pd.DataFrame | Path | str",
        value_column: str,
    ) -> "pd.DataFrame":
        """Parse time series data from various input formats."""
        if isinstance(data, (str, Path)):
            data = pd.read_csv(data, parse_dates=["datetime"])

        # Ensure datetime column
        if "datetime" in data.columns:
            data["datetime"] = pd.to_datetime(data["datetime"])

        return data

    def _resample_timeseries(
        self,
        df: "pd.DataFrame",
        group_col: str,
        value_col: str,
        frequency: str,
    ) -> "pd.DataFrame":
        """Resample time series data to specified frequency."""
        result_dfs = []
        for group_id in df[group_col].unique():
            group_df = df[df[group_col] == group_id].copy()
            group_df = group_df.set_index("datetime")
            resampled = group_df[value_col].resample(frequency).mean()
            resampled = resampled.dropna().reset_index()
            resampled[group_col] = group_id
            resampled = resampled.rename(columns={value_col: value_col})
            result_dfs.append(resampled)

        return pd.concat(result_dfs, ignore_index=True)

    def _determine_layer_from_screen(self, well: WellInfo) -> int | None:
        """Determine model layer from well screen depths."""
        if well.layer is not None:
            return well.layer

        # Would need model stratigraphy to determine layer
        # For now, return None
        return None

    def _make_valid_obs_name(self, name: str) -> str:
        """Make a valid PEST observation name."""
        # Remove invalid characters
        name = "".join(c if c.isalnum() or c == "_" else "_" for c in name)

        # Truncate to 200 chars (PEST++ limit)
        if len(name) > 200:
            name = name[:200]

        # Ensure unique
        base_name = name
        counter = 1
        while name in self._observations:
            suffix = f"_{counter}"
            name = base_name[:200 - len(suffix)] + suffix
            counter += 1

        return name

    # =========================================================================
    # Iteration
    # =========================================================================

    def __iter__(self):
        """Iterate over observations."""
        return iter(self._observations.values())

    def __len__(self) -> int:
        """Return number of observations."""
        return len(self._observations)

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"IWFMObservationManager(n_observations={self.n_observations}, "
            f"n_groups={self.n_groups})"
        )
