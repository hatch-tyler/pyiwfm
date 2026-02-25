"""
HDF5 file I/O handlers for IWFM model files.

This module provides functions for reading and writing IWFM model data
in HDF5 format, which provides efficient storage for large datasets
and supports compression.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from numpy.typing import NDArray

from pyiwfm import __version__
from pyiwfm.core.exceptions import FileFormatError
from pyiwfm.core.mesh import AppGrid, Element, Node, Subregion
from pyiwfm.core.model import IWFMModel
from pyiwfm.core.stratigraphy import Stratigraphy
from pyiwfm.core.timeseries import TimeSeries


class HDF5ModelWriter:
    """
    Writer for IWFM model data in HDF5 format.

    The HDF5 file structure:
        /mesh/
            nodes/x, y, area, is_boundary
            elements/vertices, subregion, area
            subregions/id, name
        /stratigraphy/
            gs_elev, top_elev, bottom_elev, active_node
        /timeseries/
            {variable}/{location}/times, values
        /metadata/
            name, version, created, ...
    """

    def __init__(self, filepath: Path | str, compression: str | None = "gzip") -> None:
        """
        Initialize the writer.

        Args:
            filepath: Path to the output HDF5 file
            compression: Compression algorithm ('gzip', 'lzf', or None)
        """
        self.filepath = Path(filepath)
        self.compression = compression
        self._file: h5py.File | None = None

    def __enter__(self) -> HDF5ModelWriter:
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self._file = h5py.File(self.filepath, "w")
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: object
    ) -> None:
        if self._file:
            self._file.close()

    def _create_dataset(
        self, group: h5py.Group, name: str, data: NDArray, **kwargs: Any
    ) -> h5py.Dataset:
        """Create a dataset with optional compression."""
        if self.compression and data.size > 100:
            return group.create_dataset(name, data=data, compression=self.compression, **kwargs)
        return group.create_dataset(name, data=data, **kwargs)

    def write_mesh(self, mesh: AppGrid) -> None:
        """Write mesh data to the HDF5 file."""
        if self._file is None:
            raise RuntimeError("File not open")

        mesh_grp = self._file.create_group("mesh")

        # Write nodes
        nodes_grp = mesh_grp.create_group("nodes")
        n_nodes = mesh.n_nodes

        # Collect node data into arrays in a single pass
        node_ids = sorted(mesh.nodes.keys())
        x = np.empty(n_nodes, dtype=np.float64)
        y = np.empty(n_nodes, dtype=np.float64)
        area = np.empty(n_nodes, dtype=np.float64)
        is_boundary = np.empty(n_nodes, dtype=bool)
        for i, nid in enumerate(node_ids):
            node = mesh.nodes[nid]
            x[i] = node.x
            y[i] = node.y
            area[i] = node.area
            is_boundary[i] = node.is_boundary

        self._create_dataset(nodes_grp, "id", np.array(node_ids, dtype=np.int32))
        self._create_dataset(nodes_grp, "x", x)
        self._create_dataset(nodes_grp, "y", y)
        self._create_dataset(nodes_grp, "area", area)
        self._create_dataset(nodes_grp, "is_boundary", is_boundary)

        # Write elements
        elem_grp = mesh_grp.create_group("elements")
        n_elements = mesh.n_elements

        elem_ids = sorted(mesh.elements.keys())
        # Store vertices as (n_elements, 4) with 0 padding for triangles
        vertices = np.zeros((n_elements, 4), dtype=np.int32)
        subregions = np.zeros(n_elements, dtype=np.int32)
        areas = np.zeros(n_elements)

        for i, eid in enumerate(elem_ids):
            elem = mesh.elements[eid]
            for j, v in enumerate(elem.vertices):
                vertices[i, j] = v
            subregions[i] = elem.subregion
            areas[i] = elem.area

        self._create_dataset(elem_grp, "id", np.array(elem_ids, dtype=np.int32))
        self._create_dataset(elem_grp, "vertices", vertices)
        self._create_dataset(elem_grp, "subregion", subregions)
        self._create_dataset(elem_grp, "area", areas)

        # Write subregions if present
        if mesh.subregions:
            sr_grp = mesh_grp.create_group("subregions")
            sr_ids = sorted(mesh.subregions.keys())
            sr_names = [mesh.subregions[sid].name for sid in sr_ids]

            self._create_dataset(sr_grp, "id", np.array(sr_ids, dtype=np.int32))
            # Store names as variable-length strings
            dt = h5py.special_dtype(vlen=str)
            names_ds = sr_grp.create_dataset("name", (len(sr_names),), dtype=dt)
            for i, name in enumerate(sr_names):
                names_ds[i] = name

        # Store counts as attributes
        mesh_grp.attrs["n_nodes"] = n_nodes
        mesh_grp.attrs["n_elements"] = n_elements
        mesh_grp.attrs["n_subregions"] = mesh.n_subregions

    def write_stratigraphy(self, strat: Stratigraphy) -> None:
        """Write stratigraphy data to the HDF5 file."""
        if self._file is None:
            raise RuntimeError("File not open")

        strat_grp = self._file.create_group("stratigraphy")

        self._create_dataset(strat_grp, "gs_elev", strat.gs_elev)
        self._create_dataset(strat_grp, "top_elev", strat.top_elev)
        self._create_dataset(strat_grp, "bottom_elev", strat.bottom_elev)
        self._create_dataset(strat_grp, "active_node", strat.active_node)

        strat_grp.attrs["n_layers"] = strat.n_layers
        strat_grp.attrs["n_nodes"] = strat.n_nodes

    def write_timeseries(self, ts: TimeSeries, variable: str) -> None:
        """
        Write a time series to the HDF5 file.

        Args:
            ts: TimeSeries to write
            variable: Variable name (e.g., 'head', 'flow')
        """
        if self._file is None:
            raise RuntimeError("File not open")

        # Create timeseries group if needed
        if "timeseries" not in self._file:
            self._file.create_group("timeseries")

        ts_grp = self._file["timeseries"]

        # Create variable group if needed
        if variable not in ts_grp:
            ts_grp.create_group(variable)

        var_grp = ts_grp[variable]

        # Use location as key, or name, or auto-generate
        location = ts.location or ts.name or f"series_{len(var_grp)}"

        loc_grp = var_grp.create_group(location)

        # Convert datetime64 to ISO strings for storage
        times_str = np.array([str(t) for t in ts.times], dtype=h5py.special_dtype(vlen=str))
        loc_grp.create_dataset("times", data=times_str)
        self._create_dataset(loc_grp, "values", ts.values)

        loc_grp.attrs["name"] = ts.name
        loc_grp.attrs["units"] = ts.units
        loc_grp.attrs["location"] = ts.location

    def write_metadata(self, metadata: dict[str, Any]) -> None:
        """Write model metadata to the HDF5 file."""
        if self._file is None:
            raise RuntimeError("File not open")

        if "metadata" not in self._file:
            meta_grp = self._file.create_group("metadata")
        else:
            meta_grp = self._file["metadata"]

        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                meta_grp.attrs[key] = value
            elif isinstance(value, datetime):
                meta_grp.attrs[key] = value.isoformat()

    def write_model(self, model: IWFMModel) -> None:
        """
        Write a complete IWFMModel to the HDF5 file.

        Args:
            model: IWFMModel instance to write
        """
        # Write metadata
        self.write_metadata(
            {
                "name": model.name,
                "created": datetime.now().isoformat(),
                "pyiwfm_version": __version__,
            }
        )

        # Write mesh
        if model.mesh is not None:
            self.write_mesh(model.mesh)

        # Write stratigraphy
        if model.stratigraphy is not None:
            self.write_stratigraphy(model.stratigraphy)


class HDF5ModelReader:
    """Reader for IWFM model data from HDF5 format."""

    def __init__(self, filepath: Path | str) -> None:
        """
        Initialize the reader.

        Args:
            filepath: Path to the HDF5 file
        """
        self.filepath = Path(filepath)
        self._file: h5py.File | None = None

    def __enter__(self) -> HDF5ModelReader:
        self._file = h5py.File(self.filepath, "r")
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: object
    ) -> None:
        if self._file:
            self._file.close()

    def read_mesh(self) -> AppGrid:
        """Read mesh data from the HDF5 file."""
        if self._file is None:
            raise RuntimeError("File not open")

        if "mesh" not in self._file:
            raise FileFormatError("No mesh data found in HDF5 file")

        mesh_grp = self._file["mesh"]
        nodes_grp = mesh_grp["nodes"]
        elem_grp = mesh_grp["elements"]

        # Read nodes
        node_ids = nodes_grp["id"][:]
        x = nodes_grp["x"][:]
        y = nodes_grp["y"][:]
        area = nodes_grp["area"][:] if "area" in nodes_grp else np.zeros_like(x)
        is_boundary = (
            nodes_grp["is_boundary"][:]
            if "is_boundary" in nodes_grp
            else np.zeros(len(x), dtype=bool)
        )

        nodes: dict[int, Node] = {}
        for i, nid in enumerate(node_ids):
            nodes[int(nid)] = Node(
                id=int(nid),
                x=float(x[i]),
                y=float(y[i]),
                area=float(area[i]),
                is_boundary=bool(is_boundary[i]),
            )

        # Read elements
        elem_ids = elem_grp["id"][:]
        vertices = elem_grp["vertices"][:]
        subregions = elem_grp["subregion"][:]
        elem_areas = elem_grp["area"][:] if "area" in elem_grp else np.zeros(len(elem_ids))

        elements: dict[int, Element] = {}
        for i, eid in enumerate(elem_ids):
            v = vertices[i]
            # Remove trailing zeros (triangle handling)
            v_list = [int(vi) for vi in v if vi != 0]
            elements[int(eid)] = Element(
                id=int(eid),
                vertices=tuple(v_list),
                subregion=int(subregions[i]),
                area=float(elem_areas[i]),
            )

        # Read subregions if present
        subregion_dict: dict[int, Subregion] = {}
        if "subregions" in mesh_grp:
            sr_grp = mesh_grp["subregions"]
            sr_ids = sr_grp["id"][:]
            sr_names = sr_grp["name"][:]
            for i, sid in enumerate(sr_ids):
                name = sr_names[i] if isinstance(sr_names[i], str) else sr_names[i].decode()
                subregion_dict[int(sid)] = Subregion(id=int(sid), name=name)

        return AppGrid(nodes=nodes, elements=elements, subregions=subregion_dict)

    def read_stratigraphy(self) -> Stratigraphy:
        """Read stratigraphy data from the HDF5 file."""
        if self._file is None:
            raise RuntimeError("File not open")

        if "stratigraphy" not in self._file:
            raise FileFormatError("No stratigraphy data found in HDF5 file")

        strat_grp = self._file["stratigraphy"]

        gs_elev = strat_grp["gs_elev"][:]
        top_elev = strat_grp["top_elev"][:]
        bottom_elev = strat_grp["bottom_elev"][:]
        active_node = strat_grp["active_node"][:]

        n_layers = int(strat_grp.attrs["n_layers"])
        n_nodes = int(strat_grp.attrs["n_nodes"])

        return Stratigraphy(
            n_layers=n_layers,
            n_nodes=n_nodes,
            gs_elev=gs_elev,
            top_elev=top_elev,
            bottom_elev=bottom_elev,
            active_node=active_node,
        )

    def read_timeseries(self, variable: str, location: str) -> TimeSeries:
        """
        Read a time series from the HDF5 file.

        Args:
            variable: Variable name
            location: Location identifier

        Returns:
            TimeSeries instance
        """
        if self._file is None:
            raise RuntimeError("File not open")

        if "timeseries" not in self._file:
            raise FileFormatError("No timeseries data found in HDF5 file")

        ts_grp = self._file["timeseries"]

        if variable not in ts_grp:
            raise FileFormatError(f"Variable '{variable}' not found")

        var_grp = ts_grp[variable]

        if location not in var_grp:
            raise FileFormatError(f"Location '{location}' not found for variable '{variable}'")

        loc_grp = var_grp[location]

        times_str = loc_grp["times"][:]
        times = np.array([np.datetime64(t) for t in times_str])
        values = loc_grp["values"][:]

        name = str(loc_grp.attrs.get("name", ""))
        units = str(loc_grp.attrs.get("units", ""))
        loc = str(loc_grp.attrs.get("location", location))

        return TimeSeries(
            times=times,
            values=values,
            name=name,
            units=units,
            location=loc,
        )

    def list_timeseries(self) -> dict[str, list[str]]:
        """
        List all available time series.

        Returns:
            Dictionary mapping variable names to lists of location names
        """
        if self._file is None:
            raise RuntimeError("File not open")

        result: dict[str, list[str]] = {}

        if "timeseries" not in self._file:
            return result

        ts_grp = self._file["timeseries"]
        for var_name in ts_grp:
            var_grp = ts_grp[var_name]
            result[var_name] = list(var_grp.keys())

        return result

    def read_metadata(self) -> dict[str, Any]:
        """Read model metadata from the HDF5 file."""
        if self._file is None:
            raise RuntimeError("File not open")

        result: dict[str, Any] = {}

        if "metadata" in self._file:
            meta_grp = self._file["metadata"]
            for key in meta_grp.attrs:
                result[key] = meta_grp.attrs[key]

        return result

    def read_model(self, name: str | None = None) -> IWFMModel:
        """
        Read a complete IWFMModel from the HDF5 file.

        Args:
            name: Model name (optional, read from metadata if not provided)

        Returns:
            IWFMModel instance
        """
        metadata = self.read_metadata()
        model_name = name or metadata.get("name", "unnamed")

        f = self._file
        mesh = self.read_mesh() if f is not None and "mesh" in f else None
        strat = self.read_stratigraphy() if f is not None and "stratigraphy" in f else None

        return IWFMModel(
            name=model_name,
            mesh=mesh,
            stratigraphy=strat,
            metadata=metadata,
        )


# Convenience functions


def write_model_hdf5(
    filepath: Path | str,
    model: IWFMModel,
    compression: str | None = "gzip",
) -> None:
    """
    Write an IWFMModel to an HDF5 file.

    Args:
        filepath: Path to the output file
        model: IWFMModel to write
        compression: Compression algorithm
    """
    with HDF5ModelWriter(filepath, compression=compression) as writer:
        writer.write_model(model)


def read_model_hdf5(filepath: Path | str, name: str | None = None) -> IWFMModel:
    """
    Read an IWFMModel from an HDF5 file.

    Args:
        filepath: Path to the HDF5 file
        name: Model name (optional)

    Returns:
        IWFMModel instance
    """
    with HDF5ModelReader(filepath) as reader:
        return reader.read_model(name)
