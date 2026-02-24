"""
Zone budget (ZBudget) file reader for IWFM simulation output.

This module provides classes for reading IWFM zone budget HDF5 files,
which contain spatially aggregated water balance data by user-defined zones.

Example
-------
Read a zone budget file:

>>> from pyiwfm.io.zbudget import ZBudgetReader
>>> reader = ZBudgetReader("GWZBud.hdf")
>>> print(reader.zones)
['Zone 1', 'Zone 2', 'Zone 3']
>>> df = reader.get_dataframe(zone="Zone 1")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# ZBudget data type codes (from ZBudget_Parameters.f90)
ZBUDGET_DATA_TYPES = {
    1: "Storage",
    2: "VerticalFlow",
    3: "FaceFlow",
    4: "ElementData",
}


@dataclass
class ZBudgetHeader:
    """Zone budget file header information."""

    software_version: str = ""
    descriptor: str = ""
    vert_flows_at_node: bool = False
    face_flows_defined: bool = False
    storages_defined: bool = False
    compute_error: bool = False
    n_data: int = 0
    data_types: list[int] = field(default_factory=list)
    data_names: list[str] = field(default_factory=list)
    data_hdf_paths: list[str] = field(default_factory=list)
    n_layers: int = 0
    n_elements: int = 0
    n_timesteps: int = 0
    start_datetime: datetime | None = None
    delta_t_minutes: int = 1440
    time_unit: str = "1DAY"
    elem_data_columns: dict[int, NDArray[np.int32]] = field(default_factory=dict)


@dataclass
class ZoneInfo:
    """Information about a zone."""

    id: int
    name: str
    n_elements: int = 0
    element_ids: list[int] = field(default_factory=list)
    area: float = 0.0
    adjacent_zones: list[int] = field(default_factory=list)


class ZBudgetReader:
    """
    Reader for IWFM zone budget HDF5 files.

    Zone budget files contain spatially aggregated water balance data
    for user-defined zones (groups of elements).

    Parameters
    ----------
    filepath : str or Path
        Path to the zone budget HDF5 file.

    Attributes
    ----------
    filepath : Path
        Path to the zone budget file.
    header : ZBudgetHeader
        Parsed header information.
    zones : list[str]
        List of zone names.
    data_names : list[str]
        List of budget data column names.

    Examples
    --------
    >>> reader = ZBudgetReader("GWZBud.hdf")
    >>> print(reader.descriptor)
    'GROUNDWATER ZONE BUDGET'
    >>> print(reader.zones)
    ['Zone 1', 'Zone 2', 'Zone 3']
    >>>
    >>> # Get data column names
    >>> print(reader.data_names)
    ['Deep Percolation', 'Pumping', 'Subsurface Inflow', ...]
    >>>
    >>> # Read data for a zone
    >>> df = reader.get_dataframe(zone="Zone 1", layer=1)
    """

    def __init__(self, filepath: str | Path) -> None:
        self.filepath = Path(filepath)

        if not self.filepath.exists():
            raise FileNotFoundError(f"ZBudget file not found: {self.filepath}")

        # Read header
        self.header = self._read_header()

        # Read zone information
        self._zone_info: dict[str, ZoneInfo] = {}
        self._read_zone_info()

    @staticmethod
    def _h5_get(group: h5py.File | h5py.Group, key: str) -> Any:
        """Read a value from an HDF5 group.

        Checks child datasets/groups first, then HDF5 group attributes.
        IWFM stores most ZBudget metadata as HDF5 attrs, not datasets.
        Returns the raw value (scalar or array) or ``None`` if not found.
        """
        if key in group:
            obj = group[key]
            if isinstance(obj, h5py.Dataset):
                if obj.ndim == 0:
                    return obj[()]
                return obj[:]
            return obj
        if key in group.attrs:
            return group.attrs[key]
        return None

    def _read_header(self) -> ZBudgetHeader:
        """Read header from HDF5 file."""
        header = ZBudgetHeader()
        _get = self._h5_get

        with h5py.File(self.filepath, "r") as f:
            # Find attributes group (try /Attributes/ first, then root)
            if "Attributes" in f:
                ag = f["Attributes"]
            else:
                ag = f

            # Software version
            val = _get(ag, "Software_Version")
            if val is not None:
                header.software_version = val.decode() if isinstance(val, bytes) else str(val)

            # Descriptor
            val = _get(ag, "Descriptor")
            if val is not None:
                header.descriptor = val.decode() if isinstance(val, bytes) else str(val)

            # Boolean flags
            val = _get(ag, "lVertFlows_DefinedAtNode")
            if val is not None:
                header.vert_flows_at_node = bool(val)
            val = _get(ag, "lFaceFlows_Defined")
            if val is not None:
                header.face_flows_defined = bool(val)
            val = _get(ag, "lStorages_Defined")
            if val is not None:
                header.storages_defined = bool(val)
            val = _get(ag, "lComputeError")
            if val is not None:
                header.compute_error = bool(val)

            # Scalar metadata
            val = _get(ag, "NData")
            if val is not None:
                header.n_data = int(val)
            val = _get(ag, "NTimeSteps")
            if val is not None:
                header.n_timesteps = int(val)
            val = _get(ag, "SystemData%NElements")
            if val is not None:
                header.n_elements = int(val)
            val = _get(ag, "SystemData%NLayers")
            if val is not None:
                header.n_layers = int(val)

            # Timestep info
            val = _get(ag, "TimeStep%BeginDateAndTime")
            if val is not None:
                from pyiwfm.io.budget import parse_iwfm_datetime

                dt_str = val.decode() if isinstance(val, bytes) else str(val)
                header.start_datetime = parse_iwfm_datetime(dt_str)
            val = _get(ag, "TimeStep%Unit")
            if val is not None:
                header.time_unit = val.decode() if isinstance(val, bytes) else str(val)
            val = _get(ag, "TimeStep%DeltaT_InMinutes")
            if val is not None:
                header.delta_t_minutes = int(val)

            # Array datasets
            val = _get(ag, "DataTypes")
            if val is not None:
                header.data_types = list(np.asarray(val).flat)
            val = _get(ag, "FullDataNames")
            if val is not None:
                header.data_names = [
                    n.decode().strip() if isinstance(n, bytes) else str(n).strip()
                    for n in np.asarray(val).flat
                ]
            val = _get(ag, "DataHDFPaths")
            if val is not None:
                header.data_hdf_paths = [
                    p.decode().strip() if isinstance(p, bytes) else str(p).strip()
                    for p in np.asarray(val).flat
                ]

            # ElemDataColumns — layer mapping arrays.
            # Keys: "Layer1_ElemDataColumns" or "Layer_1_ElemDataColumns".
            all_keys: set[str] = set()
            if isinstance(ag, (h5py.File, h5py.Group)):
                all_keys.update(ag.keys())
                all_keys.update(ag.attrs.keys())
            for key in sorted(all_keys):
                if "ElemDataColumns" not in key:
                    continue
                prefix = key.split("ElemDataColumns")[0].rstrip("_")
                num_str = prefix.replace("Layer", "").replace("_", "")
                if not num_str.isdigit():
                    continue
                layer_num = int(num_str)
                edc = _get(ag, key)
                if edc is not None:
                    header.elem_data_columns[layer_num] = np.asarray(edc, dtype=np.int32)
                    if header.n_layers < layer_num:
                        header.n_layers = layer_num

            # Infer n_timesteps / n_elements from dataset shapes if still zero
            if header.n_timesteps == 0 or header.n_elements == 0:
                self._infer_shape_from_data(f, header)

        logger.debug(
            "ZBudget header: n_timesteps=%d, n_elements=%d, n_layers=%d, "
            "n_data=%d, descriptor='%s', edc_layers=%s",
            header.n_timesteps,
            header.n_elements,
            header.n_layers,
            header.n_data,
            header.descriptor,
            sorted(header.elem_data_columns.keys()),
        )
        return header

    @staticmethod
    def _infer_shape_from_data(f: h5py.File, header: ZBudgetHeader) -> None:
        """Try to infer n_timesteps and n_elements from actual dataset shapes."""
        # Strategy 1: Layer_N/data_name datasets (real C2VSimFG layout)
        for layer_key in sorted(f.keys()):
            if not layer_key.startswith("Layer"):
                continue
            layer_group = f[layer_key]
            if not isinstance(layer_group, h5py.Group):
                continue
            for ds_name in layer_group.keys():
                ds = layer_group[ds_name]
                if isinstance(ds, h5py.Dataset) and ds.ndim == 2:
                    if header.n_timesteps == 0:
                        header.n_timesteps = ds.shape[0]
                    if ds.shape[1] > header.n_elements:
                        header.n_elements = ds.shape[1]
                    return

        # Strategy 2: data_name/Layer_N datasets (old test fixture layout)
        for path_str in header.data_hdf_paths:
            path_clean = path_str.strip("/")
            if path_clean in f:
                data_group = f[path_clean]
                if isinstance(data_group, h5py.Group):
                    for key in data_group.keys():
                        if key.startswith("Layer"):
                            ds = data_group[key]
                            if isinstance(ds, h5py.Dataset) and ds.ndim == 2:
                                if header.n_timesteps == 0:
                                    header.n_timesteps = ds.shape[0]
                                if ds.shape[1] > header.n_elements:
                                    header.n_elements = ds.shape[1]
                                return

    def _read_zone_info(self) -> None:
        """Read zone definitions from file."""
        with h5py.File(self.filepath, "r") as f:
            # Check for ZoneList group
            if "ZoneList" in f:
                zone_list = f["ZoneList"]

                if "NZones" in zone_list.attrs:
                    n_zones = int(zone_list.attrs["NZones"])
                elif "NZones" in zone_list:
                    n_zones = int(zone_list["NZones"][()])
                else:
                    n_zones = 0

                # Read zone names
                if "ZoneNames" in zone_list:
                    zone_names = zone_list["ZoneNames"][:]
                    zone_names = [
                        n.decode().strip() if isinstance(n, bytes) else str(n).strip()
                        for n in zone_names
                    ]
                else:
                    zone_names = [f"Zone {i + 1}" for i in range(n_zones)]

                # Read zone element assignments
                for i, name in enumerate(zone_names):
                    zone_info = ZoneInfo(id=i + 1, name=name)

                    # Try to read element list
                    zone_key = f"Zone_{i + 1}"
                    if zone_key in zone_list:
                        zone_group = zone_list[zone_key]
                        if "Elements" in zone_group:
                            zone_info.element_ids = list(zone_group["Elements"][:])
                            zone_info.n_elements = len(zone_info.element_ids)
                        if "Area" in zone_group:
                            zone_info.area = float(zone_group["Area"][()])
                        if "AdjacentZones" in zone_group:
                            zone_info.adjacent_zones = list(zone_group["AdjacentZones"][:])

                    self._zone_info[name] = zone_info

            # No ZoneList — zones will be injected by _sync_active_zones()

    @property
    def descriptor(self) -> str:
        """Return budget descriptor."""
        return self.header.descriptor

    @property
    def zones(self) -> list[str]:
        """Return list of zone names."""
        return list(self._zone_info.keys())

    @property
    def n_zones(self) -> int:
        """Return number of zones."""
        return len(self._zone_info)

    @property
    def data_names(self) -> list[str]:
        """Return list of budget data column names."""
        return self.header.data_names

    @property
    def n_layers(self) -> int:
        """Return number of model layers."""
        return self.header.n_layers

    @property
    def n_timesteps(self) -> int:
        """Return number of time steps."""
        return self.header.n_timesteps

    def get_zone_info(self, zone: str | int) -> ZoneInfo:
        """
        Get information about a zone.

        Parameters
        ----------
        zone : str or int
            Zone name or index (1-based).

        Returns
        -------
        ZoneInfo
            Zone information including elements and area.
        """
        if isinstance(zone, int):
            zone_names = list(self._zone_info.keys())
            if 1 <= zone <= len(zone_names):
                return self._zone_info[zone_names[zone - 1]]
            raise IndexError(f"Zone index {zone} out of range [1, {len(zone_names)}]")

        if zone in self._zone_info:
            return self._zone_info[zone]

        # Case-insensitive search
        for name, info in self._zone_info.items():
            if name.lower() == zone.lower():
                return info

        raise KeyError(f"Zone '{zone}' not found. Available: {self.zones}")

    def _resolve_data_index(self, data_name: str | int) -> int:
        """Resolve a data column name or index to a 0-based index."""
        if isinstance(data_name, int):
            if 0 <= data_name < self.header.n_data:
                return data_name
            raise IndexError(f"Data index {data_name} out of range")
        try:
            return self.header.data_names.index(data_name)
        except ValueError:
            lower_names = [n.lower() for n in self.header.data_names]
            try:
                return lower_names.index(data_name.lower())
            except ValueError:
                raise KeyError(
                    f"Data '{data_name}' not found. Available: {self.data_names}"
                ) from None

    def get_element_data(
        self,
        data_name: str | int,
        layer: int = 1,
    ) -> NDArray[np.float64]:
        """
        Read raw element-level data for a budget component.

        Parameters
        ----------
        data_name : str or int
            Data column name or index (0-based).
        layer : int
            Layer number (1-based).

        Returns
        -------
        NDArray[np.float64]
            2D array of shape ``(n_timesteps, n_items)``.  For datasets that
            cover all elements, ``n_items == n_elements``.  For sparse datasets
            (e.g. Streams), ``n_items < n_elements`` — use
            ``header.elem_data_columns`` to map element IDs to column indices.
        """
        data_idx = self._resolve_data_index(data_name)

        # Resolve the human-readable data name for Layer_N/data_name lookup
        resolved_name = self.header.data_names[data_idx]

        # Get HDF5 path for old-style data_name/Layer_N lookup
        if data_idx < len(self.header.data_hdf_paths):
            hdf_path = self.header.data_hdf_paths[data_idx].strip("/")
        else:
            hdf_path = resolved_name.replace(" ", "_")

        with h5py.File(self.filepath, "r") as f:
            # Strategy 1: Layer_N/data_name (real C2VSimFG structure)
            layer_key = f"Layer_{layer}"
            if layer_key in f:
                layer_group = f[layer_key]
                if isinstance(layer_group, h5py.Group) and resolved_name in layer_group:
                    return np.array(layer_group[resolved_name][:], dtype=np.float64)

            # Strategy 2: data_name/Layer_N (old test fixture structure)
            if hdf_path in f:
                data_group = f[hdf_path]
                if isinstance(data_group, h5py.Group):
                    if layer_key in data_group:
                        return np.array(data_group[layer_key][:], dtype=np.float64)
                    # Try without underscore
                    alt_key = f"Layer{layer}"
                    if alt_key in data_group:
                        return np.array(data_group[alt_key][:], dtype=np.float64)

            raise KeyError(
                f"Data '{resolved_name}' (path='{hdf_path}') for Layer_{layer} not found in file"
            )

    def get_zone_data(
        self,
        zone: str | int,
        data_name: str | int | None = None,
        layer: int = 1,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Read aggregated zone budget data.

        Parameters
        ----------
        zone : str or int
            Zone name or index.
        data_name : str or int, optional
            Data column name or index. If None, returns all data columns.
        layer : int
            Layer number (1-based).

        Returns
        -------
        tuple[NDArray, NDArray]
            Tuple of (times, values) where:
            - times: 1D array of time values
            - values: 1D or 2D array of zone budget values
        """
        zone_info = self.get_zone_info(zone)

        # Get element indices for this zone
        elem_ids = zone_info.element_ids
        zone_idx = 0
        if not elem_ids:
            # If no element IDs, assume zone index maps directly
            if isinstance(zone, int):
                zone_idx = zone - 1
            else:
                zone_idx = self.zones.index(zone)

        # Determine which data columns to read
        if data_name is not None:
            if isinstance(data_name, int):
                data_indices = [data_name]
            else:
                try:
                    data_indices = [self.header.data_names.index(data_name)]
                except ValueError:
                    lower_names = [n.lower() for n in self.header.data_names]
                    data_indices = [lower_names.index(data_name.lower())]
        else:
            data_indices = list(range(self.header.n_data))

        # Read and aggregate data
        n_timesteps = self.header.n_timesteps
        values = np.zeros((n_timesteps, len(data_indices)))

        # Get ElemDataColumns for this layer (if available)
        edc = self.header.elem_data_columns.get(layer)

        for i, data_idx in enumerate(data_indices):
            try:
                raw = self.get_element_data(data_idx, layer)
                # raw shape: (n_timesteps, n_items)

                if elem_ids:
                    if edc is not None and data_idx < edc.shape[0]:
                        # Sparse aggregation: use ElemDataColumns mapping
                        col_indices = []
                        for eid in elem_ids:
                            if eid - 1 < edc.shape[-1]:
                                compressed_col = int(edc[data_idx, eid - 1])
                                if 0 < compressed_col <= raw.shape[1]:
                                    col_indices.append(compressed_col - 1)
                        if col_indices:
                            values[:, i] = raw[:, col_indices].sum(axis=1)
                    else:
                        # Fallback: assume 1:1 element-to-column mapping
                        col_indices = [eid - 1 for eid in elem_ids if 0 <= eid - 1 < raw.shape[1]]
                        if col_indices:
                            values[:, i] = raw[:, col_indices].sum(axis=1)
                else:
                    # Use zone index directly (pre-aggregated data)
                    if zone_idx < raw.shape[1]:
                        values[:, i] = raw[:, zone_idx]
            except (KeyError, IndexError):
                # Data not available for this column
                pass

        # Generate time array
        ts = self.header
        if ts.start_datetime:
            start = ts.start_datetime
            use_months = "MON" in ts.time_unit.upper() if ts.time_unit else False
            if use_months:
                from dateutil.relativedelta import relativedelta

                times = np.array(
                    [(start + relativedelta(months=i)).timestamp() for i in range(n_timesteps)]
                )
            else:
                delta = timedelta(minutes=ts.delta_t_minutes)
                times = np.array([(start + i * delta).timestamp() for i in range(n_timesteps)])
        else:
            times = np.arange(n_timesteps, dtype=np.float64)

        # Squeeze if single column
        if len(data_indices) == 1:
            values = values.squeeze()

        return times, values

    def get_dataframe(
        self,
        zone: str | int,
        layer: int = 1,
        data_columns: list[str] | None = None,
        *,
        volume_factor: float = 1.0,
    ) -> pd.DataFrame:
        """
        Read zone budget data as a pandas DataFrame.

        When *volume_factor* is left at its default of ``1.0`` the raw
        simulation values are returned unchanged.  Pass the FACTVLOU
        value to get unit-converted output.

        Parameters
        ----------
        zone : str or int
            Zone name or index.
        layer : int
            Layer number (1-based).
        data_columns : list[str], optional
            Specific data columns to include. If None, includes all.
        volume_factor : float
            Multiplier for all data columns (zone budget columns are
            volumetric).  Default ``1.0``.

        Returns
        -------
        pd.DataFrame
            DataFrame with datetime index and budget columns.
        """
        # Get data
        times, values = self.get_zone_data(zone, layer=layer)

        # Ensure 2D
        if values.ndim == 1:
            values = values.reshape(-1, 1)

        # Apply unit conversion when factor is non-unity
        if volume_factor != 1.0:
            values = values * volume_factor

        # Create datetime index
        ts = self.header
        index: pd.Index  # type: ignore[type-arg]
        if ts.start_datetime:
            index = pd.to_datetime(times, unit="s")
        else:
            index = pd.Index(times, name="Time")

        # Column names
        col_names = self.data_names[: values.shape[1]]

        df = pd.DataFrame(values, index=index, columns=col_names)

        # Filter columns if specified
        if data_columns is not None:
            available = [c for c in data_columns if c in df.columns]
            df = df[available]

        return df

    def get_all_zones_dataframe(
        self,
        data_name: str,
        layer: int = 1,
    ) -> pd.DataFrame:
        """
        Get data for all zones as a DataFrame.

        Parameters
        ----------
        data_name : str
            Data column name.
        layer : int
            Layer number (1-based).

        Returns
        -------
        pd.DataFrame
            DataFrame with datetime index and zone columns.
        """
        # Read data for each zone
        zone_data = {}
        for zone_name in self.zones:
            times, values = self.get_zone_data(zone_name, data_name, layer)
            zone_data[zone_name] = values

        # Create datetime index
        ts = self.header
        zindex: pd.Index  # type: ignore[type-arg]
        if ts.start_datetime:
            zindex = pd.to_datetime(times, unit="s")
        else:
            zindex = pd.Index(times, name="Time")

        return pd.DataFrame(zone_data, index=zindex)

    def get_monthly_averages(
        self,
        zone: str | int,
        layer: int = 1,
    ) -> pd.DataFrame:
        """
        Calculate monthly averages for zone budget data.

        Parameters
        ----------
        zone : str or int
            Zone name or index.
        layer : int
            Layer number (1-based).

        Returns
        -------
        pd.DataFrame
            DataFrame with monthly averaged values.
        """
        df = self.get_dataframe(zone, layer)

        if isinstance(df.index, pd.DatetimeIndex):
            result: pd.DataFrame = df.resample("MS").mean()
            return result
        else:
            return df

    def get_annual_totals(
        self,
        zone: str | int,
        layer: int = 1,
    ) -> pd.DataFrame:
        """
        Calculate annual totals for zone budget data.

        Parameters
        ----------
        zone : str or int
            Zone name or index.
        layer : int
            Layer number (1-based).

        Returns
        -------
        pd.DataFrame
            DataFrame with annual total values.
        """
        df = self.get_dataframe(zone, layer)

        if isinstance(df.index, pd.DatetimeIndex):
            annual_result: pd.DataFrame = df.resample("YS").sum()
            return annual_result
        else:
            n_years = len(df) // 12
            if n_years == 0:
                total: pd.DataFrame = df.sum().to_frame().T
                return total
            annual = []
            for i in range(n_years):
                annual.append(df.iloc[i * 12 : (i + 1) * 12].sum())
            return pd.DataFrame(annual)

    def get_water_balance(
        self,
        zone: str | int,
        layer: int = 1,
        inflow_columns: list[str] | None = None,
        outflow_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate water balance for a zone.

        Parameters
        ----------
        zone : str or int
            Zone name or index.
        layer : int
            Layer number (1-based).
        inflow_columns : list[str], optional
            Columns to sum as inflows.
        outflow_columns : list[str], optional
            Columns to sum as outflows (will be negated).

        Returns
        -------
        pd.DataFrame
            DataFrame with inflows, outflows, and balance.
        """
        df = self.get_dataframe(zone, layer)

        # Auto-detect inflow/outflow columns if not specified
        if inflow_columns is None:
            inflow_columns = [
                c
                for c in df.columns
                if any(kw in c.lower() for kw in ["inflow", "recharge", "percolation", "precip"])
            ]
        if outflow_columns is None:
            outflow_columns = [
                c
                for c in df.columns
                if any(kw in c.lower() for kw in ["outflow", "pump", "et", "evap", "discharge"])
            ]

        result = pd.DataFrame(index=df.index)

        if inflow_columns:
            result["Total Inflow"] = df[inflow_columns].sum(axis=1)
        else:
            result["Total Inflow"] = 0.0

        if outflow_columns:
            result["Total Outflow"] = df[outflow_columns].abs().sum(axis=1)
        else:
            result["Total Outflow"] = 0.0

        result["Net Balance"] = result["Total Inflow"] - result["Total Outflow"]

        return result

    def to_csv(
        self,
        output_dir: str | Path,
        zones: list[str] | None = None,
        layer: int = 1,
    ) -> list[Path]:
        """
        Export zone budget data to CSV files.

        Parameters
        ----------
        output_dir : str or Path
            Output directory for CSV files.
        zones : list[str], optional
            Zones to export. If None, exports all zones.
        layer : int
            Layer number (1-based).

        Returns
        -------
        list[Path]
            List of created CSV file paths.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if zones is None:
            zones = self.zones

        created_files = []
        for zone in zones:
            df = self.get_dataframe(zone, layer)
            safe_name = zone.replace(" ", "_").replace("/", "_")
            output_path = output_dir / f"zbudget_{safe_name}_layer{layer}.csv"
            df.to_csv(output_path)
            created_files.append(output_path)

        return created_files

    def __repr__(self) -> str:
        return (
            f"ZBudgetReader('{self.filepath.name}', "
            f"descriptor='{self.descriptor}', "
            f"n_zones={self.n_zones}, "
            f"n_layers={self.n_layers}, "
            f"n_timesteps={self.n_timesteps})"
        )
