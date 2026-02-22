"""
PreProcessor Binary File Reader for IWFM.

This module reads the binary output file produced by the IWFM PreProcessor.
The binary file uses ``ACCESS='STREAM'`` (raw bytes, no Fortran record
markers) and contains pre-processed mesh, stratigraphy, and component data
that enables faster simulation startup.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from pyiwfm.io.binary import StreamAccessBinaryReader

# Fixed-length string sizes used by IWFM binary output
_SUBREGION_NAME_LEN = 50
_REACH_NAME_LEN = 30


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
    """Reader for IWFM PreProcessor binary output files.

    The preprocessor binary uses ``ACCESS='STREAM'`` (raw bytes, no Fortran
    record markers).  All arrays are written with explicit lengths that must
    be derived from dimension counts read earlier in the file.

    Section read order (matching IWFM Fortran ``WritePreprocessedData``):

    1. AppGrid — dimensions, coordinates, connectivity, per-node/element
       data, face data, boundary faces, subregions
    2. Stratigraphy — NLayers, TopActiveLayer, ActiveNode, GSElev,
       TopElev, BottomElev  (2-D arrays in Fortran column-major order)
    3. StrmLakeConnector — 3 sub-connectors (stream→lake, lake→stream,
       lake→lake), each with count + source/dest arrays
    4. StrmGWConnector — version int, then gw_node/layer arrays
    5. LakeGWConnector — per-lake element/node arrays
    6. AppLake — version int, per-lake data with rating tables
    7. AppStream — version int, per-node rating tables, per-reach metadata
    8. Matrix — sparse structure (optional)
    """

    def __init__(self, endian: str = "<") -> None:
        self.endian = endian

    def read(self, filepath: Path | str) -> PreprocessorBinaryData:
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Binary file not found: {filepath}")

        data = PreprocessorBinaryData()

        with StreamAccessBinaryReader(filepath, self.endian) as f:
            self._read_grid_data(f, data)
            self._read_stratigraphy(f, data)
            self._read_stream_lake_connector(f, data)
            self._read_stream_gw_connector(f, data)
            self._read_lake_gw_connector(f, data)
            self._read_lake_data(f, data)
            self._read_stream_data(f, data)
            self._read_matrix_data(f, data)

        return data

    # -- Section 1: AppGrid ------------------------------------------------

    def _read_grid_data(
        self, f: StreamAccessBinaryReader, data: PreprocessorBinaryData
    ) -> None:
        # 5 dimension ints
        data.n_nodes = f.read_int()
        data.n_elements = f.read_int()
        data.n_faces = f.read_int()
        data.n_subregions = f.read_int()
        data.n_boundary_faces = f.read_int()

        # Coordinates
        data.x = f.read_doubles(data.n_nodes)
        data.y = f.read_doubles(data.n_nodes)

        # Element connectivity: NVertex(NElements), Vertex(MaxNV, NElements)
        data.n_vertex = f.read_ints(data.n_elements)
        max_nv = int(data.n_vertex.max()) if data.n_elements > 0 else 4
        data.vertex = f.read_ints(max_nv * data.n_elements)

        # Per-node app data
        for _ in range(data.n_nodes):
            data.app_nodes.append(self._read_app_node(f))

        # Per-element app data
        for _ in range(data.n_elements):
            data.app_elements.append(self._read_app_element(f))

        # Face data
        data.app_faces = self._read_app_faces(f, data.n_faces)

        # Boundary face list
        if data.n_boundary_faces > 0:
            data.boundary_face_list = f.read_ints(data.n_boundary_faces)

        # Subregion data
        for _ in range(data.n_subregions):
            data.subregions.append(self._read_subregion(f))

    def _read_app_node(self, f: StreamAccessBinaryReader) -> AppNodeData:
        node_id = f.read_int()
        area = f.read_double()
        boundary_node = f.read_logical()
        n_connected = f.read_int()
        n_face_id = f.read_int()
        n_surround = f.read_int()
        n_connected_arr = f.read_int()

        surrounding = f.read_ints(n_surround)
        connected = f.read_ints(n_connected_arr)
        face_ids = f.read_ints(n_face_id)
        elem_ccw = f.read_ints(n_face_id)
        irrot_coeff = f.read_doubles(n_face_id)

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

    def _read_app_element(self, f: StreamAccessBinaryReader) -> AppElementData:
        elem_id = f.read_int()
        subregion = f.read_int()
        area = f.read_double()
        n_faces = f.read_int()
        n_vert_area = f.read_int()
        n_del_shp = f.read_int()
        n_rot_shp = f.read_int()

        face_ids = f.read_ints(n_faces)
        vert_areas = f.read_doubles(n_vert_area)
        vert_fracs = f.read_doubles(n_vert_area)
        del_shp = f.read_doubles(n_del_shp)
        rot_shp = f.read_doubles(n_rot_shp)

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

    def _read_app_faces(
        self, f: StreamAccessBinaryReader, n_faces: int
    ) -> AppFaceData:
        if n_faces == 0:
            return AppFaceData(
                nodes=np.array([], dtype=np.int32).reshape(0, 2),
                elements=np.array([], dtype=np.int32).reshape(0, 2),
                boundary=np.array([], dtype=np.bool_),
                lengths=np.array([], dtype=np.float64),
            )

        nodes = f.read_ints(2 * n_faces).reshape((n_faces, 2))
        elements = f.read_ints(2 * n_faces).reshape((n_faces, 2))
        boundary = f.read_logicals(n_faces)
        lengths = f.read_doubles(n_faces)

        return AppFaceData(
            nodes=nodes, elements=elements, boundary=boundary, lengths=lengths
        )

    def _read_subregion(self, f: StreamAccessBinaryReader) -> SubregionData:
        sub_id = f.read_int()
        name = f.read_string(_SUBREGION_NAME_LEN)
        n_elements = f.read_int()
        n_neighbors = f.read_int()
        area = f.read_double()

        region_elements = f.read_ints(n_elements)
        neighbor_ids = f.read_ints(n_neighbors)
        neighbor_n_faces = f.read_ints(n_neighbors)

        neighbor_faces: list[NDArray[np.int32]] = []
        for nf in neighbor_n_faces:
            neighbor_faces.append(f.read_ints(int(nf)))

        return SubregionData(
            id=sub_id,
            name=name,
            n_elements=n_elements,
            n_neighbor_regions=n_neighbors,
            area=area,
            region_elements=region_elements,
            neighbor_region_ids=neighbor_ids,
            neighbor_n_boundary_faces=neighbor_n_faces,
            neighbor_boundary_faces=neighbor_faces,
        )

    # -- Section 2: Stratigraphy -------------------------------------------

    def _read_stratigraphy(
        self, f: StreamAccessBinaryReader, data: PreprocessorBinaryData
    ) -> None:
        n_layers = f.read_int()
        n = data.n_nodes

        # Fortran write order: NLayers, TopActiveLayer, ActiveNode,
        #   GSElev, TopElev, BottomElev
        # 2-D arrays stored column-major (Fortran order).
        top_active = f.read_ints(n)
        active_flat = f.read_logicals(n * n_layers)
        gs_elev = f.read_doubles(n)
        top_flat = f.read_doubles(n * n_layers)
        bottom_flat = f.read_doubles(n * n_layers)

        active_2d = active_flat.reshape((n, n_layers), order="F")

        data.stratigraphy = StratigraphyData(
            n_layers=n_layers,
            ground_surface_elev=gs_elev,
            top_elev=top_flat.reshape((n, n_layers), order="F"),
            bottom_elev=bottom_flat.reshape((n, n_layers), order="F"),
            active_node=active_2d,
            # Derived arrays — not stored in binary; fill with zeros.
            active_layer_above=np.zeros((n, n_layers), dtype=np.int32),
            active_layer_below=np.zeros((n, n_layers), dtype=np.int32),
            top_active_layer=top_active,
            bottom_active_layer=np.zeros(n, dtype=np.int32),
        )

    # -- Section 3: StrmLakeConnector (3 sub-connectors) -------------------

    def _read_stream_lake_connector(
        self, f: StreamAccessBinaryReader, data: PreprocessorBinaryData
    ) -> None:
        # Sub-connector 1: Stream inflow → Lake
        n1 = f.read_int()
        strm_nodes = f.read_ints(n1) if n1 > 0 else np.array([], dtype=np.int32)
        lake_ids = f.read_ints(n1) if n1 > 0 else np.array([], dtype=np.int32)

        data.stream_lake_connector = StreamLakeConnectorData(
            n_connections=n1, stream_nodes=strm_nodes, lake_ids=lake_ids
        )

        # Sub-connector 2: Lake outflow → Stream (read & discard)
        n2 = f.read_int()
        if n2 > 0:
            f.read_ints(n2)  # lake IDs
            f.read_ints(n2)  # stream nodes

        # Sub-connector 3: Lake outflow → Lake (read & discard)
        n3 = f.read_int()
        if n3 > 0:
            f.read_ints(n3)  # source lake IDs
            f.read_ints(n3)  # dest lake IDs

    # -- Section 4: StrmGWConnector ----------------------------------------

    def _read_stream_gw_connector(
        self, f: StreamAccessBinaryReader, data: PreprocessorBinaryData
    ) -> None:
        version = f.read_int()

        if version == 0:
            data.stream_gw_connector = StreamGWConnectorData(
                n_stream_nodes=0,
                gw_nodes=np.array([], dtype=np.int32),
                layers=np.array([], dtype=np.int32),
            )
            return

        n_strm = f.read_int()
        gw_nodes = f.read_ints(n_strm)
        layers = f.read_ints(n_strm)

        data.stream_gw_connector = StreamGWConnectorData(
            n_stream_nodes=n_strm, gw_nodes=gw_nodes, layers=layers
        )

    # -- Section 5: LakeGWConnector ----------------------------------------

    def _read_lake_gw_connector(
        self, f: StreamAccessBinaryReader, data: PreprocessorBinaryData
    ) -> None:
        n_lakes = f.read_int()

        if n_lakes == 0:
            data.lake_gw_connector = LakeGWConnectorData(
                n_lakes=0, lake_elements=[], lake_nodes=[]
            )
            return

        lake_elements: list[NDArray[np.int32]] = []
        lake_nodes: list[NDArray[np.int32]] = []

        for _ in range(n_lakes):
            ne = f.read_int()
            lake_elements.append(f.read_ints(ne))
            nn = f.read_int()
            lake_nodes.append(f.read_ints(nn))

        data.lake_gw_connector = LakeGWConnectorData(
            n_lakes=n_lakes,
            lake_elements=lake_elements,
            lake_nodes=lake_nodes,
        )

    # -- Section 6: AppLake (version-prefixed) -----------------------------

    def _read_lake_data(
        self, f: StreamAccessBinaryReader, data: PreprocessorBinaryData
    ) -> None:
        version = f.read_int()

        if version == 0:
            data.lakes = LakeData(
                n_lakes=0,
                lake_ids=np.array([], dtype=np.int32),
                lake_names=[],
                lake_max_elevations=np.array([], dtype=np.float64),
                lake_elements=[],
            )
            return

        n_lakes = f.read_int()
        lake_ids = np.zeros(n_lakes, dtype=np.int32)
        lake_max_elev = np.zeros(n_lakes, dtype=np.float64)
        lake_elements: list[NDArray[np.int32]] = []

        for i in range(n_lakes):
            lake_ids[i] = f.read_int()
            lake_max_elev[i] = f.read_double()
            # PairedData rating table
            n_pts = f.read_int()
            if n_pts > 0:
                f.read_doubles(n_pts)  # elevation points (skip)
                f.read_doubles(n_pts)  # outflow points (skip)
            ne = f.read_int()
            lake_elements.append(f.read_ints(ne) if ne > 0 else np.array([], dtype=np.int32))

        data.lakes = LakeData(
            n_lakes=n_lakes,
            lake_ids=lake_ids,
            lake_names=[f"Lake {int(lake_ids[i])}" for i in range(n_lakes)],
            lake_max_elevations=lake_max_elev,
            lake_elements=lake_elements,
        )

    # -- Section 7: AppStream (version-prefixed) ---------------------------

    def _read_stream_data(
        self, f: StreamAccessBinaryReader, data: PreprocessorBinaryData
    ) -> None:
        version = f.read_int()

        if version == 0:
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

        n_reaches = f.read_int()
        n_strm_nodes = f.read_int()

        # Per-stream-node: ID + rating table
        for _ in range(n_strm_nodes):
            _node_id = f.read_int()
            n_pts = f.read_int()
            if n_pts > 0:
                f.read_doubles(n_pts)  # stage points (skip)
                f.read_doubles(n_pts)  # flow points (skip)

        # Per-reach metadata
        reach_ids = np.zeros(n_reaches, dtype=np.int32)
        reach_names: list[str] = []
        upstream = np.zeros(n_reaches, dtype=np.int32)
        downstream = np.zeros(n_reaches, dtype=np.int32)
        outflow = np.zeros(n_reaches, dtype=np.int32)

        for i in range(n_reaches):
            reach_ids[i] = f.read_int()
            reach_names.append(f.read_string(_REACH_NAME_LEN))
            upstream[i] = f.read_int()
            downstream[i] = f.read_int()
            outflow[i] = f.read_int()

        data.streams = StreamData(
            n_reaches=n_reaches,
            n_stream_nodes=n_strm_nodes,
            reach_ids=reach_ids,
            reach_names=reach_names,
            reach_upstream_nodes=upstream,
            reach_downstream_nodes=downstream,
            reach_outflow_dest=outflow,
        )

    # -- Section 8: Matrix (optional) --------------------------------------

    def _read_matrix_data(
        self, f: StreamAccessBinaryReader, data: PreprocessorBinaryData
    ) -> None:
        try:
            data.matrix_n_equations = f.read_int()
            if data.matrix_n_equations > 0 and not f.at_eof():
                # Connectivity: NJCOLs(NEquations) then JCOL values
                n_jcols = f.read_ints(data.matrix_n_equations)
                total = int(n_jcols.sum())
                if total > 0 and not f.at_eof():
                    data.matrix_connectivity = f.read_ints(total)
        except EOFError:
            pass


def read_preprocessor_binary(
    filepath: Path | str, endian: str = "<"
) -> PreprocessorBinaryData:
    """Convenience function to read preprocessor binary file.

    Args:
        filepath: Path to the binary file
        endian: Byte order

    Returns:
        PreprocessorBinaryData with all preprocessed data
    """
    reader = PreprocessorBinaryReader(endian=endian)
    return reader.read(filepath)
