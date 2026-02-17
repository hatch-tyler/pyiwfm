"""
PreProcessor Binary File Reader for IWFM.

This module reads the binary output file produced by the IWFM PreProcessor.
The binary file contains pre-processed mesh, stratigraphy, and component data
that enables faster simulation startup.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from pyiwfm.io.binary import FortranBinaryReader


@dataclass
class AppNodeData:
    """Pre-processed application node data."""

    id: int
    area: float
    boundary_node: bool
    n_connected_node: int
    n_face_id: int
    surrounding_elements: NDArray[np.int32]
    connected_nodes: NDArray[np.int32]
    face_ids: NDArray[np.int32]
    elem_id_on_ccw_side: NDArray[np.int32]
    irrotational_coeff: NDArray[np.float64]


@dataclass
class AppElementData:
    """Pre-processed application element data."""

    id: int
    subregion: int
    area: float
    face_ids: NDArray[np.int32]
    vertex_areas: NDArray[np.float64]
    vertex_area_fractions: NDArray[np.float64]
    integral_del_shp_i_del_shp_j: NDArray[np.float64]
    integral_rot_del_shp_i_del_shp_j: NDArray[np.float64]


@dataclass
class AppFaceData:
    """Pre-processed application face data."""

    nodes: NDArray[np.int32]  # (n_faces, 2) - node indices for each face
    elements: NDArray[np.int32]  # (n_faces, 2) - element indices on each side
    boundary: NDArray[np.bool_]  # (n_faces,) - True if boundary face
    lengths: NDArray[np.float64]  # (n_faces,) - face lengths


@dataclass
class SubregionData:
    """Pre-processed subregion data."""

    id: int
    name: str
    n_elements: int
    n_neighbor_regions: int
    area: float
    region_elements: NDArray[np.int32]
    neighbor_region_ids: NDArray[np.int32]
    neighbor_n_boundary_faces: NDArray[np.int32]
    neighbor_boundary_faces: list[NDArray[np.int32]]


@dataclass
class StratigraphyData:
    """Pre-processed stratigraphy data."""

    n_layers: int
    ground_surface_elev: NDArray[np.float64]  # (n_nodes,)
    top_elev: NDArray[np.float64]  # (n_nodes, n_layers)
    bottom_elev: NDArray[np.float64]  # (n_nodes, n_layers)
    active_node: NDArray[np.bool_]  # (n_nodes, n_layers)
    active_layer_above: NDArray[np.int32]  # (n_nodes, n_layers)
    active_layer_below: NDArray[np.int32]  # (n_nodes, n_layers)
    top_active_layer: NDArray[np.int32]  # (n_nodes,)
    bottom_active_layer: NDArray[np.int32]  # (n_nodes,)


@dataclass
class StreamGWConnectorData:
    """Pre-processed stream-GW connector data."""

    n_stream_nodes: int
    gw_nodes: NDArray[np.int32]  # GW node index for each stream node
    layers: NDArray[np.int32]  # Layer for each connection


@dataclass
class LakeGWConnectorData:
    """Pre-processed lake-GW connector data."""

    n_lakes: int
    lake_elements: list[NDArray[np.int32]]  # Elements for each lake
    lake_nodes: list[NDArray[np.int32]]  # GW nodes for each lake


@dataclass
class StreamLakeConnectorData:
    """Pre-processed stream-lake connector data."""

    n_connections: int
    stream_nodes: NDArray[np.int32]  # Stream node indices
    lake_ids: NDArray[np.int32]  # Connected lake IDs


@dataclass
class StreamData:
    """Pre-processed stream data."""

    n_reaches: int
    n_stream_nodes: int
    reach_ids: NDArray[np.int32]
    reach_names: list[str]
    reach_upstream_nodes: NDArray[np.int32]
    reach_downstream_nodes: NDArray[np.int32]
    reach_outflow_dest: NDArray[np.int32]  # 0=boundary, -n=lake n, +n=reach n


@dataclass
class LakeData:
    """Pre-processed lake data."""

    n_lakes: int
    lake_ids: NDArray[np.int32]
    lake_names: list[str]
    lake_max_elevations: NDArray[np.float64]
    lake_elements: list[NDArray[np.int32]]


@dataclass
class PreprocessorBinaryData:
    """Complete preprocessor binary output structure."""

    # Grid dimensions
    n_nodes: int = 0
    n_elements: int = 0
    n_faces: int = 0
    n_subregions: int = 0
    n_boundary_faces: int = 0

    # Coordinates
    x: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    y: NDArray[np.float64] = field(default_factory=lambda: np.array([]))

    # Element connectivity
    n_vertex: NDArray[np.int32] = field(default_factory=lambda: np.array([]))
    vertex: NDArray[np.int32] = field(default_factory=lambda: np.array([]))

    # App grid data
    app_nodes: list[AppNodeData] = field(default_factory=list)
    app_elements: list[AppElementData] = field(default_factory=list)
    app_faces: AppFaceData | None = None
    boundary_face_list: NDArray[np.int32] = field(default_factory=lambda: np.array([]))
    subregions: list[SubregionData] = field(default_factory=list)

    # Stratigraphy
    stratigraphy: StratigraphyData | None = None

    # Component connectors
    stream_lake_connector: StreamLakeConnectorData | None = None
    stream_gw_connector: StreamGWConnectorData | None = None
    lake_gw_connector: LakeGWConnectorData | None = None

    # Components
    lakes: LakeData | None = None
    streams: StreamData | None = None

    # Matrix data (sparse matrix structure)
    matrix_n_equations: int = 0
    matrix_connectivity: NDArray[np.int32] = field(default_factory=lambda: np.array([]))


class PreprocessorBinaryReader:
    """
    Reader for IWFM PreProcessor binary output files.

    The preprocessor binary file contains all pre-computed data needed by the
    simulation, including mesh geometry, stratigraphy, component connectors,
    and sparse matrix structure.
    """

    def __init__(self, endian: str = "<"):
        """
        Initialize the reader.

        Args:
            endian: Byte order ('<' = little-endian, '>' = big-endian)
        """
        self.endian = endian

    def read(self, filepath: Path | str) -> PreprocessorBinaryData:
        """
        Read preprocessor binary file.

        Args:
            filepath: Path to the binary file

        Returns:
            PreprocessorBinaryData with all preprocessed data
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Binary file not found: {filepath}")

        data = PreprocessorBinaryData()

        with FortranBinaryReader(filepath, self.endian) as f:
            # Read grid data
            self._read_grid_data(f, data)

            # Read stratigraphy
            self._read_stratigraphy(f, data)

            # Read stream-lake connector
            self._read_stream_lake_connector(f, data)

            # Read stream-GW connector
            self._read_stream_gw_connector(f, data)

            # Read lake-GW connector
            self._read_lake_gw_connector(f, data)

            # Read lake data
            self._read_lake_data(f, data)

            # Read stream data
            self._read_stream_data(f, data)

            # Read matrix data
            self._read_matrix_data(f, data)

        return data

    def _read_grid_data(self, f: FortranBinaryReader, data: PreprocessorBinaryData) -> None:
        """Read grid dimensions and geometry."""
        # Grid dimensions
        data.n_nodes = f.read_int()
        data.n_elements = f.read_int()
        data.n_faces = f.read_int()
        data.n_subregions = f.read_int()
        data.n_boundary_faces = f.read_int()

        # Node coordinates
        data.x = f.read_double_array()
        data.y = f.read_double_array()

        # Element connectivity
        data.n_vertex = f.read_int_array()  # Number of vertices per element
        data.vertex = f.read_int_array()  # Flattened vertex indices

        # Read app node data
        for _ in range(data.n_nodes):
            node_data = self._read_app_node(f)
            data.app_nodes.append(node_data)

        # Read app element data
        for _ in range(data.n_elements):
            elem_data = self._read_app_element(f)
            data.app_elements.append(elem_data)

        # Read app face data
        data.app_faces = self._read_app_faces(f, data.n_faces)

        # Read boundary face list
        if data.n_boundary_faces > 0:
            data.boundary_face_list = f.read_int_array()

        # Read subregion data
        for _ in range(data.n_subregions):
            sub_data = self._read_subregion(f)
            data.subregions.append(sub_data)

    def _read_app_node(self, f: FortranBinaryReader) -> AppNodeData:
        """Read single app node record."""
        node_id = f.read_int()
        area = f.read_double()
        boundary_node = f.read_int() != 0
        n_connected = f.read_int()
        n_face_id = f.read_int()
        n_surround = f.read_int()
        n_connected_arr = f.read_int()

        surrounding = f.read_int_array() if n_surround > 0 else np.array([], dtype=np.int32)
        connected = f.read_int_array() if n_connected_arr > 0 else np.array([], dtype=np.int32)
        face_ids = f.read_int_array() if n_face_id > 0 else np.array([], dtype=np.int32)
        elem_ccw = f.read_int_array() if n_face_id > 0 else np.array([], dtype=np.int32)
        irrot_coeff = f.read_double_array() if n_face_id > 0 else np.array([], dtype=np.float64)

        return AppNodeData(
            id=node_id,
            area=area,
            boundary_node=boundary_node,
            n_connected_node=n_connected,
            n_face_id=n_face_id,
            surrounding_elements=surrounding,
            connected_nodes=connected,
            face_ids=face_ids,
            elem_id_on_ccw_side=elem_ccw,
            irrotational_coeff=irrot_coeff,
        )

    def _read_app_element(self, f: FortranBinaryReader) -> AppElementData:
        """Read single app element record."""
        elem_id = f.read_int()
        subregion = f.read_int()
        area = f.read_double()
        n_faces = f.read_int()
        n_vert_area = f.read_int()
        n_del_shp = f.read_int()
        n_rot_shp = f.read_int()

        face_ids = f.read_int_array() if n_faces > 0 else np.array([], dtype=np.int32)
        vert_areas = f.read_double_array() if n_vert_area > 0 else np.array([], dtype=np.float64)
        vert_fracs = f.read_double_array() if n_vert_area > 0 else np.array([], dtype=np.float64)
        del_shp = f.read_double_array() if n_del_shp > 0 else np.array([], dtype=np.float64)
        rot_shp = f.read_double_array() if n_rot_shp > 0 else np.array([], dtype=np.float64)

        return AppElementData(
            id=elem_id,
            subregion=subregion,
            area=area,
            face_ids=face_ids,
            vertex_areas=vert_areas,
            vertex_area_fractions=vert_fracs,
            integral_del_shp_i_del_shp_j=del_shp,
            integral_rot_del_shp_i_del_shp_j=rot_shp,
        )

    def _read_app_faces(self, f: FortranBinaryReader, n_faces: int) -> AppFaceData:
        """Read app face data."""
        if n_faces == 0:
            return AppFaceData(
                nodes=np.array([], dtype=np.int32).reshape(0, 2),
                elements=np.array([], dtype=np.int32).reshape(0, 2),
                boundary=np.array([], dtype=np.bool_),
                lengths=np.array([], dtype=np.float64),
            )

        # Read face node pairs
        nodes_flat = f.read_int_array()
        nodes = nodes_flat.reshape((n_faces, 2))

        # Read face element pairs (elements on each side)
        elements_flat = f.read_int_array()
        elements = elements_flat.reshape((n_faces, 2))

        # Read boundary flags
        boundary_int = f.read_int_array()
        boundary = boundary_int.astype(np.bool_)

        # Read face lengths
        lengths = f.read_double_array()

        return AppFaceData(
            nodes=nodes,
            elements=elements,
            boundary=boundary,
            lengths=lengths,
        )

    def _read_subregion(self, f: FortranBinaryReader) -> SubregionData:
        """Read single subregion record."""
        sub_id = f.read_int()
        name = f.read_string()
        n_elements = f.read_int()
        n_neighbors = f.read_int()
        area = f.read_double()

        elements = f.read_int_array() if n_elements > 0 else np.array([], dtype=np.int32)
        neighbor_ids = f.read_int_array() if n_neighbors > 0 else np.array([], dtype=np.int32)
        neighbor_n_faces = f.read_int_array() if n_neighbors > 0 else np.array([], dtype=np.int32)

        # Read boundary faces for each neighbor
        neighbor_faces: list[NDArray[np.int32]] = []
        if n_neighbors > 0:
            for n_faces in neighbor_n_faces:
                if n_faces > 0:
                    faces = f.read_int_array()
                    neighbor_faces.append(faces)
                else:
                    neighbor_faces.append(np.array([], dtype=np.int32))

        return SubregionData(
            id=sub_id,
            name=name,
            n_elements=n_elements,
            n_neighbor_regions=n_neighbors,
            area=area,
            region_elements=elements,
            neighbor_region_ids=neighbor_ids,
            neighbor_n_boundary_faces=neighbor_n_faces,
            neighbor_boundary_faces=neighbor_faces,
        )

    def _read_stratigraphy(self, f: FortranBinaryReader, data: PreprocessorBinaryData) -> None:
        """Read stratigraphy data."""
        n_layers = f.read_int()
        n_nodes = data.n_nodes

        gs_elev = f.read_double_array()
        top_flat = f.read_double_array()
        bottom_flat = f.read_double_array()
        active_flat = f.read_int_array()

        # Additional computed data
        active_above_flat = f.read_int_array()
        active_below_flat = f.read_int_array()
        top_active = f.read_int_array()
        bottom_active = f.read_int_array()

        data.stratigraphy = StratigraphyData(
            n_layers=n_layers,
            ground_surface_elev=gs_elev,
            top_elev=top_flat.reshape((n_nodes, n_layers)),
            bottom_elev=bottom_flat.reshape((n_nodes, n_layers)),
            active_node=active_flat.reshape((n_nodes, n_layers)).astype(np.bool_),
            active_layer_above=active_above_flat.reshape((n_nodes, n_layers)),
            active_layer_below=active_below_flat.reshape((n_nodes, n_layers)),
            top_active_layer=top_active,
            bottom_active_layer=bottom_active,
        )

    def _read_stream_lake_connector(
        self, f: FortranBinaryReader, data: PreprocessorBinaryData
    ) -> None:
        """Read stream-lake connector data."""
        n_connections = f.read_int()

        if n_connections == 0:
            data.stream_lake_connector = StreamLakeConnectorData(
                n_connections=0,
                stream_nodes=np.array([], dtype=np.int32),
                lake_ids=np.array([], dtype=np.int32),
            )
            return

        stream_nodes = f.read_int_array()
        lake_ids = f.read_int_array()

        data.stream_lake_connector = StreamLakeConnectorData(
            n_connections=n_connections,
            stream_nodes=stream_nodes,
            lake_ids=lake_ids,
        )

    def _read_stream_gw_connector(
        self, f: FortranBinaryReader, data: PreprocessorBinaryData
    ) -> None:
        """Read stream-GW connector data."""
        n_stream_nodes = f.read_int()

        if n_stream_nodes == 0:
            data.stream_gw_connector = StreamGWConnectorData(
                n_stream_nodes=0,
                gw_nodes=np.array([], dtype=np.int32),
                layers=np.array([], dtype=np.int32),
            )
            return

        gw_nodes = f.read_int_array()
        layers = f.read_int_array()

        data.stream_gw_connector = StreamGWConnectorData(
            n_stream_nodes=n_stream_nodes,
            gw_nodes=gw_nodes,
            layers=layers,
        )

    def _read_lake_gw_connector(self, f: FortranBinaryReader, data: PreprocessorBinaryData) -> None:
        """Read lake-GW connector data."""
        n_lakes = f.read_int()

        if n_lakes == 0:
            data.lake_gw_connector = LakeGWConnectorData(
                n_lakes=0,
                lake_elements=[],
                lake_nodes=[],
            )
            return

        lake_elements: list[NDArray[np.int32]] = []
        lake_nodes: list[NDArray[np.int32]] = []

        for _ in range(n_lakes):
            n_elems = f.read_int()
            elements = f.read_int_array() if n_elems > 0 else np.array([], dtype=np.int32)
            lake_elements.append(elements)

            n_nodes_lake = f.read_int()
            nodes = f.read_int_array() if n_nodes_lake > 0 else np.array([], dtype=np.int32)
            lake_nodes.append(nodes)

        data.lake_gw_connector = LakeGWConnectorData(
            n_lakes=n_lakes,
            lake_elements=lake_elements,
            lake_nodes=lake_nodes,
        )

    def _read_lake_data(self, f: FortranBinaryReader, data: PreprocessorBinaryData) -> None:
        """Read lake component data."""
        n_lakes = f.read_int()

        if n_lakes == 0:
            data.lakes = LakeData(
                n_lakes=0,
                lake_ids=np.array([], dtype=np.int32),
                lake_names=[],
                lake_max_elevations=np.array([], dtype=np.float64),
                lake_elements=[],
            )
            return

        lake_ids = f.read_int_array()
        lake_names: list[str] = []
        lake_max_elev: list[float] = []
        lake_elements: list[NDArray[np.int32]] = []

        for _ in range(n_lakes):
            name = f.read_string()
            lake_names.append(name)
            max_elev = f.read_double()
            lake_max_elev.append(max_elev)
            n_elems = f.read_int()
            elements = f.read_int_array() if n_elems > 0 else np.array([], dtype=np.int32)
            lake_elements.append(elements)

        data.lakes = LakeData(
            n_lakes=n_lakes,
            lake_ids=lake_ids,
            lake_names=lake_names,
            lake_max_elevations=np.array(lake_max_elev),
            lake_elements=lake_elements,
        )

    def _read_stream_data(self, f: FortranBinaryReader, data: PreprocessorBinaryData) -> None:
        """Read stream component data."""
        n_reaches = f.read_int()
        n_stream_nodes = f.read_int()

        if n_reaches == 0:
            data.streams = StreamData(
                n_reaches=0,
                n_stream_nodes=0,
                reach_ids=np.array([], dtype=np.int32),
                reach_names=[],
                reach_upstream_nodes=np.array([], dtype=np.int32),
                reach_downstream_nodes=np.array([], dtype=np.int32),
                reach_outflow_dest=np.array([], dtype=np.int32),
            )
            return

        reach_ids = f.read_int_array()
        reach_names: list[str] = []
        upstream_nodes = f.read_int_array()
        downstream_nodes = f.read_int_array()
        outflow_dest = f.read_int_array()

        for _ in range(n_reaches):
            name = f.read_string()
            reach_names.append(name)

        data.streams = StreamData(
            n_reaches=n_reaches,
            n_stream_nodes=n_stream_nodes,
            reach_ids=reach_ids,
            reach_names=reach_names,
            reach_upstream_nodes=upstream_nodes,
            reach_downstream_nodes=downstream_nodes,
            reach_outflow_dest=outflow_dest,
        )

    def _read_matrix_data(self, f: FortranBinaryReader, data: PreprocessorBinaryData) -> None:
        """Read sparse matrix structure data."""
        try:
            data.matrix_n_equations = f.read_int()
            if data.matrix_n_equations > 0:
                data.matrix_connectivity = f.read_int_array()
        except EOFError:
            # Matrix data may not be present in all binary files
            pass


def read_preprocessor_binary(filepath: Path | str, endian: str = "<") -> PreprocessorBinaryData:
    """
    Convenience function to read preprocessor binary file.

    Args:
        filepath: Path to the binary file
        endian: Byte order

    Returns:
        PreprocessorBinaryData with all preprocessed data
    """
    reader = PreprocessorBinaryReader(endian=endian)
    return reader.read(filepath)
