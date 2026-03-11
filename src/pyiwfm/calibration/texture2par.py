"""
Python replacement for the Fortran Texture2Par executable.

Replicates the full T2P pipeline:
1. Read well log data (% coarse with depth)
2. Compute % coarse per well per model layer
3. Krige % coarse from wells to model nodes (PcNode)
4. Krige pilot point end-member values to model nodes/elements
5. Apply power-law mixing model to compute final parameters
6. Write IWFM simulation input files via pyiwfm native I/O

Uses pyiwfm's GeostatManager for kriging and native readers/writers for
format-aware roundtrip I/O of IWFM simulation files.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from pyiwfm.core.model import IWFMModel
from pyiwfm.runner.pest_geostat import GeostatManager, Variogram

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes for well logs and pilot points
# ---------------------------------------------------------------------------


@dataclass
class WellLog:
    """A single well log with % coarse data at depth intervals."""

    name: str
    well_id: int
    x: float  # feet
    y: float  # feet
    z_land: float  # land surface elevation (feet)
    points: list[tuple[float, float]]  # (depth_ft, pc) pairs where pc is 0 or 1


@dataclass
class PilotPointSet:
    """Set of pilot points for a parameter domain."""

    x: NDArray  # (n_pp,) feet
    y: NDArray  # (n_pp,) feet
    values: dict[str, NDArray]  # param_name -> (n_pp,) or (n_pp, n_layers)
    zones: NDArray | None = None  # (n_pp,) zone IDs


@dataclass
class MixingConfig:
    """Power-law mixing exponents for each parameter domain."""

    kh_power: float = 0.93  # horizontal conductivity
    kv_power: float = -0.62  # vertical conductivity
    sy_power: float = 1.0  # specific yield
    sce_power: float = 1.0  # elastic storage coefficient
    uzpk_power: float = -0.62  # UZ hydraulic conductivity
    rzk_power: float = -0.62  # RZ conductivity
    rzpk_power: float = -0.62  # RZ ponded conductivity


# ---------------------------------------------------------------------------
# Main Texture2Par class
# ---------------------------------------------------------------------------


class Texture2Par:
    """Python replacement for Fortran Texture2Par.

    Replicates the full T2P pipeline:
    1. Read well log data (% coarse with depth)
    2. Compute % coarse per well per model layer
    3. Krige % coarse from wells to model nodes (PcNode)
    4. Krige pilot point end-member values to model nodes
    5. Apply power-law mixing model to compute final parameters
    6. Write IWFM simulation input files

    Parameters
    ----------
    model : IWFMModel
        Loaded IWFM model with mesh and stratigraphy.
    variogram : Variogram
        Variogram model for kriging.
    n_nearest : int
        Number of nearest wells/points to use in kriging.
        If 0, uses all points (full covariance matrix).
    mixing : MixingConfig
        Power-law mixing exponents.
    """

    def __init__(
        self,
        model: IWFMModel,
        variogram: Variogram,
        n_nearest: int = 0,
        mixing: MixingConfig | None = None,
    ):
        self.model = model
        self.variogram = variogram
        self.n_nearest = n_nearest
        self.mixing = mixing or MixingConfig()
        self.geostat = GeostatManager()

        # Mesh geometry (feet)
        self._node_x = model.mesh.x  # type: ignore[union-attr]
        self._node_y = model.mesh.y  # type: ignore[union-attr]
        self._n_nodes = model.n_nodes
        self._n_layers = model.n_layers
        self._n_elements = model.n_elements

        # Compute element centroids
        self._elem_cx, self._elem_cy = self._compute_element_centroids()

        # Well log data
        self._well_logs: list[WellLog] = []
        self._layer_texture: dict[int, NDArray] = {}  # well_idx -> (n_layers,) pct_coarse

        # Kriged texture field (populated by krige_texture_to_nodes)
        self._pc_node: NDArray | None = None  # (n_nodes, n_layers)
        self._pc_elem: NDArray | None = None  # (n_elements, n_layers)

        # Pilot point sets
        self._pilot_points: dict[str, PilotPointSet] = {}

        # Kriging factors (cached)
        self._factors_node: NDArray | None = None  # (n_nodes, n_pp)
        self._factors_elem: NDArray | None = None  # (n_elements, n_pp)

        # Global params (KCk, KFk from original T2P)
        self.kck: float = 0.007
        self.kfk: float = 0.007

    # ------------------------------------------------------------------
    # Element centroid computation
    # ------------------------------------------------------------------

    def _compute_element_centroids(self) -> tuple[NDArray, NDArray]:
        """Compute element centroid coordinates in feet."""
        elements = self.model.mesh.elements  # type: ignore[union-attr]
        nodes = self.model.mesh.nodes  # type: ignore[union-attr]
        sorted_elem_ids = sorted(elements.keys())

        cx = np.zeros(len(sorted_elem_ids))
        cy = np.zeros(len(sorted_elem_ids))

        for i, eid in enumerate(sorted_elem_ids):
            elem = elements[eid]
            verts = [v for v in elem.vertices if v > 0]
            cx[i] = np.mean([nodes[v].x for v in verts])
            cy[i] = np.mean([nodes[v].y for v in verts])

        return cx, cy

    # ------------------------------------------------------------------
    # Well log processing (replicates readIWFM.f90:inputwells_IWFM)
    # ------------------------------------------------------------------

    def read_well_logs(self, wellfile: Path) -> None:
        """Read well log data from T2P-format well log file.

        Format: TSV with header, columns:
            WellName, Well, Point, PC, X, Y, Zland, Depth, [reserved cols]

        PC values: 0 = fine, 1 = coarse, -99 = no data

        Coordinates in the file may be in feet (matching T2P convention).
        """
        wellfile = Path(wellfile)
        logger.info("Reading well logs from %s", wellfile)

        wells: dict[int, WellLog] = {}

        with open(wellfile) as f:
            f.readline()  # skip header
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 8:
                    continue
                try:
                    name = parts[0].strip()
                    well_id = int(parts[1])
                    # point = int(parts[2])  # not needed
                    pc = float(parts[3])
                    x = float(parts[4])
                    y = float(parts[5])
                    z_land = float(parts[6])
                    depth = float(parts[7])
                except (ValueError, IndexError):
                    continue

                if pc == -99:
                    continue  # no data

                if well_id not in wells:
                    wells[well_id] = WellLog(
                        name=name,
                        well_id=well_id,
                        x=x,
                        y=y,
                        z_land=z_land,
                        points=[],
                    )
                wells[well_id].points.append((depth, pc))

        self._well_logs = list(wells.values())
        logger.info(
            "Read %d wells with %d total log points",
            len(self._well_logs),
            sum(len(w.points) for w in self._well_logs),
        )

    def compute_layer_texture(self) -> None:
        """Compute % coarse per well per model layer.

        Replicates MakePar.f90:layertexture() — for each well, intersects
        the depth intervals with model layer boundaries at the nearest node,
        computing the fraction of coarse material within each layer.
        """
        if not self._well_logs:
            raise RuntimeError("No well logs loaded. Call read_well_logs() first.")

        strat = self.model.stratigraphy
        if strat is None:
            raise RuntimeError("Model has no stratigraphy loaded.")

        from scipy.spatial import KDTree

        node_tree = KDTree(np.column_stack([self._node_x, self._node_y]))

        for wi, well in enumerate(self._well_logs):
            # Find nearest node for layer boundaries
            _, nearest_idx = node_tree.query([well.x, well.y])

            # Layer boundaries at this node (elevations in feet)
            layer_top = strat.top_elev[nearest_idx, :]  # (n_layers,)
            layer_bot = strat.bottom_elev[nearest_idx, :]  # (n_layers,)

            # Sort well log points by depth
            sorted_pts = sorted(well.points, key=lambda p: p[0])

            # Compute % coarse per layer
            pc_layer = np.full(self._n_layers, 0.5)  # default 50% if no data

            for layer in range(self._n_layers):
                ltop = layer_top[layer]
                lbot = layer_bot[layer]

                if ltop <= lbot:
                    continue  # degenerate layer

                # Convert depth to elevation: elev = z_land - depth
                coarse_thickness = 0.0
                total_thickness = 0.0

                for pi in range(len(sorted_pts)):
                    depth_top = sorted_pts[pi][0]
                    depth_bot = sorted_pts[pi + 1][0] if pi + 1 < len(sorted_pts) else depth_top
                    pc_val = sorted_pts[pi][1]

                    elev_top = well.z_land - depth_top
                    elev_bot = well.z_land - depth_bot

                    # Intersection with layer
                    int_top = min(elev_top, ltop)
                    int_bot = max(elev_bot, lbot)

                    if int_top > int_bot:
                        thickness = int_top - int_bot
                        total_thickness += thickness
                        if pc_val > 0.5:  # coarse
                            coarse_thickness += thickness

                if total_thickness > 0:
                    pc_layer[layer] = coarse_thickness / total_thickness

            self._layer_texture[wi] = pc_layer

        logger.info("Computed layer texture for %d wells", len(self._layer_texture))

    def krige_texture_to_nodes(self) -> NDArray:
        """Krige % coarse from wells to all model nodes.

        Returns PcNode array of shape (n_nodes, n_layers).
        Also computes PcElem for element centroids.
        """
        if not self._layer_texture:
            raise RuntimeError("No layer texture computed. Call compute_layer_texture() first.")

        # Extract well coordinates and texture values
        well_x = np.array([w.x for w in self._well_logs])
        well_y = np.array([w.y for w in self._well_logs])

        pc_node = np.zeros((self._n_nodes, self._n_layers))
        pc_elem = np.zeros((self._n_elements, self._n_layers))

        for layer in range(self._n_layers):
            # Get wells with texture data for this layer
            well_vals = np.array(
                [
                    self._layer_texture[wi][layer]
                    for wi in range(len(self._well_logs))
                    if wi in self._layer_texture
                ]
            )

            if len(well_vals) == 0:
                pc_node[:, layer] = 0.5
                pc_elem[:, layer] = 0.5
                continue

            # Krige to nodes
            if self.n_nearest > 0:
                # Local kriging with nearest neighbors
                pc_node[:, layer] = self._krige_local(
                    well_x,
                    well_y,
                    well_vals,
                    self._node_x,
                    self._node_y,
                )
            else:
                pc_node[:, layer] = self.geostat.krige(
                    well_x,
                    well_y,
                    well_vals,
                    self._node_x,
                    self._node_y,
                    self.variogram,
                )

            # Krige to element centroids
            if self.n_nearest > 0:
                pc_elem[:, layer] = self._krige_local(
                    well_x,
                    well_y,
                    well_vals,
                    self._elem_cx,
                    self._elem_cy,
                )
            else:
                pc_elem[:, layer] = self.geostat.krige(
                    well_x,
                    well_y,
                    well_vals,
                    self._elem_cx,
                    self._elem_cy,
                    self.variogram,
                )

        # Clip to [0, 1]
        pc_node = np.clip(pc_node, 0.0, 1.0)
        pc_elem = np.clip(pc_elem, 0.0, 1.0)

        self._pc_node = pc_node
        self._pc_elem = pc_elem

        logger.info("Kriged texture to %d nodes and %d elements", self._n_nodes, self._n_elements)
        return pc_node

    def _krige_local(
        self,
        source_x: NDArray,
        source_y: NDArray,
        source_vals: NDArray,
        target_x: NDArray,
        target_y: NDArray,
    ) -> NDArray:
        """Krige using only the nearest n_nearest source points for each target.

        Replicates T2P's spkrige_tree approach with KD-tree local neighborhood.
        Includes negative weight correction (Deutsch 1995).
        """
        from scipy import linalg
        from scipy.spatial import KDTree

        n_near = min(self.n_nearest, len(source_x))
        source_tree = KDTree(np.column_stack([source_x, source_y]))
        n_target = len(target_x)
        result = np.zeros(n_target)

        for i in range(n_target):
            dists, indices = source_tree.query(
                [target_x[i], target_y[i]],
                k=n_near,
            )
            if n_near == 1:
                dists = np.array([dists])
                indices = np.array([indices])

            local_x = source_x[indices]
            local_y = source_y[indices]
            local_vals = source_vals[indices]

            # Build local kriging system (ordinary kriging)
            n = len(local_x)
            C_pp = np.zeros((n, n))
            for a in range(n):
                for b in range(n):
                    d = np.sqrt((local_x[a] - local_x[b]) ** 2 + (local_y[a] - local_y[b]) ** 2)
                    C_pp[a, b] = self.variogram.covariance(d)

            # Augmented system
            K = np.zeros((n + 1, n + 1))
            K[:n, :n] = C_pp
            K[:n, n] = 1.0
            K[n, :n] = 1.0

            # RHS
            b_vec = np.zeros(n + 1)
            for a in range(n):
                d = np.sqrt((local_x[a] - target_x[i]) ** 2 + (local_y[a] - target_y[i]) ** 2)
                b_vec[a] = self.variogram.covariance(d)
            b_vec[n] = 1.0

            try:
                w = linalg.solve(K, b_vec)
            except linalg.LinAlgError:
                K[: n + 1, : n + 1] += np.eye(n + 1) * 1e-10
                w = linalg.solve(K, b_vec)

            weights = w[:n]

            # Negative weight correction (Deutsch 1995)
            weights = self._correct_negative_weights(weights)

            result[i] = np.dot(weights, local_vals)

        return result

    @staticmethod
    def _correct_negative_weights(weights: NDArray) -> NDArray:
        """Correct negative kriging weights (Deutsch 1995).

        Sets negative weights to 0 and renormalizes remaining to sum to 1.
        """
        corrected = np.maximum(weights, 0.0)
        total = corrected.sum()
        if total > 0:
            corrected /= total
        else:
            # All weights were negative — use uniform
            corrected = np.ones_like(weights) / len(weights)
        return corrected  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # Pilot point I/O
    # ------------------------------------------------------------------

    def read_pilot_points(self, pp_file: Path, domain: str) -> None:
        """Read pilot point values from a TSV file.

        Parameters
        ----------
        pp_file : Path
            Path to pilot point file (TSV with header).
        domain : str
            Domain name: "aquifer", "aquitard", "subsidence", "unsaturated",
            "rootzone".
        """
        pp_file = Path(pp_file)
        logger.info("Reading pilot points from %s (domain=%s)", pp_file, domain)

        with open(pp_file) as f:
            f.readline()  # skip header
            data = []
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    data.append([float(p) for p in parts])

        if not data:
            raise ValueError(f"No data found in {pp_file}")

        arr = np.array(data)

        if domain == "aquifer":
            # Columns: X, Y, KCMin, DeltaKC, KFMin, DeltaKF, SsC, SsF, SyC, SyF, AnisoC, AnisoF, Zone
            pp = PilotPointSet(
                x=arr[:, 0],
                y=arr[:, 1],
                values={
                    "KCMin": arr[:, 2],
                    "DeltaKC": arr[:, 3],
                    "KFMin": arr[:, 4],
                    "DeltaKF": arr[:, 5],
                    "SsC": arr[:, 6],
                    "SsF": arr[:, 7],
                    "SyC": arr[:, 8],
                    "SyF": arr[:, 9],
                    "AnisoC": arr[:, 10],
                    "AnisoF": arr[:, 11],
                },
                zones=arr[:, 12].astype(int) if arr.shape[1] > 12 else None,
            )
        elif domain == "aquitard":
            # Columns: X, Y, KCMin, DeltaKC, KFMin, DeltaKF, AnisoC, AnisoF, Zone
            pp = PilotPointSet(
                x=arr[:, 0],
                y=arr[:, 1],
                values={
                    "KCMin": arr[:, 2],
                    "DeltaKC": arr[:, 3],
                    "KFMin": arr[:, 4],
                    "DeltaKF": arr[:, 5],
                    "AnisoC": arr[:, 6],
                    "AnisoF": arr[:, 7],
                },
                zones=arr[:, 8].astype(int) if arr.shape[1] > 8 else None,
            )
        elif domain == "subsidence":
            # Columns: X, Y, SCE_C_L1..L4, SCE_F_L1..L4, SCI_F_L1..L4
            pp = PilotPointSet(
                x=arr[:, 0],
                y=arr[:, 1],
                values={},
            )
            col = 2
            for lyr in range(1, 5):
                pp.values[f"SCE_C_L{lyr}"] = arr[:, col]
                col += 1
            for lyr in range(1, 5):
                pp.values[f"SCE_F_L{lyr}"] = arr[:, col]
                col += 1
            for lyr in range(1, 5):
                pp.values[f"SCI_F_L{lyr}"] = arr[:, col]
                col += 1
        elif domain == "unsaturated":
            # Columns: X, Y, PK_C_L1, PK_F_L1, PK_C_L2, PK_F_L2, PN_L1, PN_L2, PI_L1, PI_L2
            pp = PilotPointSet(
                x=arr[:, 0],
                y=arr[:, 1],
                values={
                    "PK_C_L1": arr[:, 2],
                    "PK_F_L1": arr[:, 3],
                    "PK_C_L2": arr[:, 4],
                    "PK_F_L2": arr[:, 5],
                    "PN_L1": arr[:, 6],
                    "PN_L2": arr[:, 7],
                    "PI_L1": arr[:, 8],
                    "PI_L2": arr[:, 9],
                },
            )
        elif domain == "rootzone":
            # Columns: X, Y, K_C, K_F, PondedK_C, PondedK_F, FC, WP
            pp = PilotPointSet(
                x=arr[:, 0],
                y=arr[:, 1],
                values={
                    "K_C": arr[:, 2],
                    "K_F": arr[:, 3],
                    "PondedK_C": arr[:, 4],
                    "PondedK_F": arr[:, 5],
                    "FC": arr[:, 6],
                    "WP": arr[:, 7],
                },
            )
        else:
            raise ValueError(f"Unknown domain: {domain}")

        self._pilot_points[domain] = pp
        logger.info(
            "Loaded %d pilot points for domain %s with params %s",
            len(pp.x),
            domain,
            list(pp.values.keys()),
        )

    def read_pilot_points_t2p(self, t2p_file: Path) -> None:
        """Read pilot points from existing Texture2Par.in format.

        Parses the aquifer and aquitard pilot point sections from the
        original T2P input file.
        """
        t2p_file = Path(t2p_file)
        logger.info("Reading pilot points from T2P file: %s", t2p_file)

        with open(t2p_file) as f:
            lines = f.readlines()

        # Parse variogram settings and global params
        data_lines = [
            line.rstrip() for line in lines if line.strip() and not line.strip().startswith("*")
        ]

        # Parse global params (KCk, KFk, KHp, KVp, Syp)
        global_idx = 0
        for i, line in enumerate(data_lines):
            if "/" in line and "KCk" in line:
                self.kck = float(line.split()[0])
                global_idx = i
                break

        if global_idx > 0:
            self.kfk = float(data_lines[global_idx + 1].split()[0])
            self.mixing.kh_power = float(data_lines[global_idx + 2].split()[0])
            self.mixing.kv_power = float(data_lines[global_idx + 3].split()[0])
            self.mixing.sy_power = float(data_lines[global_idx + 4].split()[0])

        # Find aquifer and aquitard sections by looking for * marker lines
        aq_start = -1
        aq_end = -1
        aqt_start = -1
        aqt_end = -1

        for i, line in enumerate(lines):
            stripped = line.strip()
            if "Pilot Points - X, Y, KCMin" in stripped and "Aquitard" not in stripped:
                aq_start = i + 1
            elif "Aquitard Pilot Points" in stripped:
                aq_end = i
                aqt_start = i + 1
            elif stripped.startswith("* EOF"):
                if aqt_start > 0:
                    aqt_end = i
                elif aq_start > 0:
                    aq_end = i

        # Parse aquifer pilot points
        if aq_start > 0:
            aq_data = []
            for i in range(aq_start, aq_end if aq_end > 0 else len(lines)):
                line = lines[i].strip()
                if line.startswith("*") or not line:
                    continue
                parts = line.split()
                if len(parts) >= 13:
                    aq_data.append([float(p) for p in parts[:13]])

            if aq_data:
                arr = np.array(aq_data)
                self._pilot_points["aquifer"] = PilotPointSet(
                    x=arr[:, 0],
                    y=arr[:, 1],
                    values={
                        "KCMin": arr[:, 2],
                        "DeltaKC": arr[:, 3],
                        "KFMin": arr[:, 4],
                        "DeltaKF": arr[:, 5],
                        "SsC": arr[:, 6],
                        "SsF": arr[:, 7],
                        "SyC": arr[:, 8],
                        "SyF": arr[:, 9],
                        "AnisoC": arr[:, 10],
                        "AnisoF": arr[:, 11],
                    },
                    zones=arr[:, 12].astype(int),
                )
                logger.info("Parsed %d aquifer pilot points", len(arr))

        # Parse aquitard pilot points
        if aqt_start > 0:
            aqt_data = []
            for i in range(aqt_start, aqt_end if aqt_end > 0 else len(lines)):
                line = lines[i].strip()
                if line.startswith("*") or not line:
                    continue
                parts = line.split()
                if len(parts) >= 9:
                    aqt_data.append([float(p) for p in parts[:9]])

            if aqt_data:
                arr = np.array(aqt_data)
                self._pilot_points["aquitard"] = PilotPointSet(
                    x=arr[:, 0],
                    y=arr[:, 1],
                    values={
                        "KCMin": arr[:, 2],
                        "DeltaKC": arr[:, 3],
                        "KFMin": arr[:, 4],
                        "DeltaKF": arr[:, 5],
                        "AnisoC": arr[:, 6],
                        "AnisoF": arr[:, 7],
                    },
                    zones=arr[:, 8].astype(int),
                )
                logger.info("Parsed %d aquitard pilot points", len(arr))

    # ------------------------------------------------------------------
    # Kriging factors (precompute for reuse across PEST++ iterations)
    # ------------------------------------------------------------------

    def precompute_kriging_factors(
        self,
        domain: str,
        target: str = "nodes",
    ) -> NDArray:
        """Precompute kriging factors for a pilot point domain.

        Parameters
        ----------
        domain : str
            Pilot point domain name.
        target : str
            "nodes" or "elements" — determines target locations.

        Returns
        -------
        NDArray
            Kriging factor matrix (n_target x n_pilot).
        """
        pp = self._pilot_points.get(domain)
        if pp is None:
            raise RuntimeError(f"No pilot points loaded for domain '{domain}'")

        if target == "nodes":
            tx, ty = self._node_x, self._node_y
        elif target == "elements":
            tx, ty = self._elem_cx, self._elem_cy
        else:
            raise ValueError(f"target must be 'nodes' or 'elements', got '{target}'")

        factors = self.geostat.compute_kriging_factors(
            pp.x,
            pp.y,
            tx,
            ty,
            self.variogram,
        )

        # Apply negative weight correction row-by-row
        for i in range(factors.shape[0]):
            factors[i, :] = self._correct_negative_weights(factors[i, :])

        if target == "nodes":
            self._factors_node = factors
        else:
            self._factors_elem = factors

        logger.info("Computed kriging factors: %s → %d %s", domain, factors.shape[0], target)
        return factors

    def krige_endmembers(
        self,
        domain: str,
        target: str = "nodes",
    ) -> dict[str, NDArray]:
        """Krige all pilot point parameters for a domain to target locations.

        Parameters
        ----------
        domain : str
            Pilot point domain name.
        target : str
            "nodes" or "elements".

        Returns
        -------
        dict mapping param_name to interpolated values at target locations.
        """
        pp = self._pilot_points.get(domain)
        if pp is None:
            raise RuntimeError(f"No pilot points for domain '{domain}'")

        # Use cached factors if available, otherwise compute
        factors = self._factors_node if target == "nodes" else self._factors_elem
        if factors is None or factors.shape[1] != len(pp.x):
            factors = self.precompute_kriging_factors(domain, target)

        result = {}
        for name, values in pp.values.items():
            if values.ndim == 1:
                result[name] = factors @ values
            else:
                # Multi-layer: krige each layer separately
                n_layers = values.shape[1]
                interp = np.zeros((factors.shape[0], n_layers))
                for layer in range(n_layers):
                    interp[:, layer] = factors @ values[:, layer]
                result[name] = interp

        return result

    # ------------------------------------------------------------------
    # Mixing models
    # ------------------------------------------------------------------

    @staticmethod
    def mix_power_law(
        pc: NDArray,
        coarse: NDArray,
        fine: NDArray,
        power: float,
    ) -> NDArray:
        """Power-law mixing model.

        param = (pc * coarse^p + (1-pc) * fine^p) ^ (1/p)

        Handles p=0 as geometric mean: coarse^pc * fine^(1-pc).
        """
        if abs(power) < 1e-10:
            # Geometric mean
            return np.where(  # type: ignore[no-any-return]
                (coarse > 0) & (fine > 0),
                coarse**pc * fine ** (1.0 - pc),
                pc * coarse + (1.0 - pc) * fine,  # fallback to linear
            )

        # Ensure positive values for power-law
        coarse_safe = np.maximum(coarse, 1e-30)
        fine_safe = np.maximum(fine, 1e-30)

        mixed = pc * coarse_safe**power + (1.0 - pc) * fine_safe**power
        mixed = np.maximum(mixed, 1e-30)
        return mixed ** (1.0 / power)  # type: ignore[no-any-return]

    @staticmethod
    def mix_fine_only(pc: NDArray, fine_val: NDArray) -> NDArray:
        """Fine-only mixing: param = fine_val * (1 - pc).

        Used for inelastic compaction (SCI) which only occurs in fine sediment.
        """
        return fine_val * (1.0 - pc)

    # ------------------------------------------------------------------
    # Domain-specific parameter computation
    # ------------------------------------------------------------------

    def compute_aquifer_params(self) -> dict[str, NDArray]:
        """Compute aquifer parameters via texture mixing at nodes.

        Returns dict with keys: Kh, Kv, Ss, Sy, Aniso — each (n_nodes, n_layers).
        Uses the T2P mixing model:
            Kh = KCk * (pc * KCMin^p + (1-pc) * KFMin^p)^(1/p) * (1 + pc*DeltaKC + (1-pc)*DeltaKF)
        """
        if self._pc_node is None:
            raise RuntimeError("Texture field not computed. Call krige_texture_to_nodes() first.")

        aq = self._pilot_points.get("aquifer")
        if aq is None:
            raise RuntimeError("No aquifer pilot points loaded.")

        # Krige aquifer end-members to nodes
        endmembers = self.krige_endmembers("aquifer", "nodes")

        pc = self._pc_node  # (n_nodes, n_layers)
        result = {}

        # Horizontal conductivity: Kh = KCk * mix(KCMin, KFMin, KHp) * (1 + mix(DeltaKC, DeltaKF, KHp))
        kc_min = endmembers["KCMin"][:, np.newaxis] * np.ones((1, self._n_layers))
        delta_kc = endmembers["DeltaKC"][:, np.newaxis] * np.ones((1, self._n_layers))
        kf_min = endmembers["KFMin"][:, np.newaxis] * np.ones((1, self._n_layers))
        delta_kf = endmembers["DeltaKF"][:, np.newaxis] * np.ones((1, self._n_layers))

        kh_base = self.mix_power_law(pc, kc_min, kf_min, self.mixing.kh_power)
        kh_delta = pc * delta_kc + (1.0 - pc) * delta_kf
        result["Kh"] = self.kck * kh_base * (1.0 + kh_delta)

        # Vertical conductivity: Kv = KFk * mix(KCMin, KFMin, KVp)
        result["Kv"] = self.kfk * self.mix_power_law(pc, kc_min, kf_min, self.mixing.kv_power)

        # Specific storage
        ssc = endmembers["SsC"][:, np.newaxis] * np.ones((1, self._n_layers))
        ssf = endmembers["SsF"][:, np.newaxis] * np.ones((1, self._n_layers))
        result["Ss"] = self.mix_power_law(pc, ssc, ssf, 1.0)  # linear mixing for Ss

        # Specific yield
        syc = endmembers["SyC"][:, np.newaxis] * np.ones((1, self._n_layers))
        syf = endmembers["SyF"][:, np.newaxis] * np.ones((1, self._n_layers))
        result["Sy"] = self.mix_power_law(pc, syc, syf, self.mixing.sy_power)

        # Anisotropy ratio
        aniso_c = endmembers["AnisoC"][:, np.newaxis] * np.ones((1, self._n_layers))
        aniso_f = endmembers["AnisoF"][:, np.newaxis] * np.ones((1, self._n_layers))
        result["Aniso"] = pc * aniso_c + (1.0 - pc) * aniso_f

        # Aquitard parameters (if present)
        if "aquitard" in self._pilot_points:
            aqt = self.krige_endmembers("aquitard", "nodes")
            aqt_kc = aqt["KCMin"][:, np.newaxis] * np.ones((1, self._n_layers))
            aqt_kf = aqt["KFMin"][:, np.newaxis] * np.ones((1, self._n_layers))

            result["AqtKv"] = self.kfk * self.mix_power_law(
                pc,
                aqt_kc,
                aqt_kf,
                self.mixing.kv_power,
            )
            aqt_aniso_c = aqt["AnisoC"][:, np.newaxis] * np.ones((1, self._n_layers))
            aqt_aniso_f = aqt["AnisoF"][:, np.newaxis] * np.ones((1, self._n_layers))
            result["AqtAniso"] = pc * aqt_aniso_c + (1.0 - pc) * aqt_aniso_f

        logger.info(
            "Computed aquifer parameters at %d nodes x %d layers", self._n_nodes, self._n_layers
        )
        return result

    def compute_subsidence_params(self) -> dict[str, NDArray]:
        """Compute subsidence parameters via texture mixing at nodes.

        Returns dict with keys: SCE, SCI — each (n_nodes, n_layers).
        SCE uses power-law mixing (coarse + fine end-members).
        SCI uses fine-only mixing.
        """
        if self._pc_node is None:
            raise RuntimeError("Texture field not computed.")

        sub = self._pilot_points.get("subsidence")
        if sub is None:
            raise RuntimeError("No subsidence pilot points loaded.")

        endmembers = self.krige_endmembers("subsidence", "nodes")
        pc = self._pc_node
        result = {}

        n_layers = min(4, self._n_layers)
        sce = np.zeros((self._n_nodes, n_layers))
        sci = np.zeros((self._n_nodes, n_layers))

        for lyr in range(n_layers):
            sce_c = endmembers.get(f"SCE_C_L{lyr + 1}")
            sce_f = endmembers.get(f"SCE_F_L{lyr + 1}")
            sci_f = endmembers.get(f"SCI_F_L{lyr + 1}")

            if sce_c is not None and sce_f is not None:
                sce[:, lyr] = self.mix_power_law(
                    pc[:, lyr],
                    sce_c,
                    sce_f,
                    self.mixing.sce_power,
                )
            if sci_f is not None:
                sci[:, lyr] = self.mix_fine_only(pc[:, lyr], sci_f)

        result["SCE"] = sce
        result["SCI"] = sci

        logger.info(
            "Computed subsidence parameters at %d nodes x %d layers", self._n_nodes, n_layers
        )
        return result

    def compute_unsat_params(self) -> dict[str, NDArray]:
        """Compute unsaturated zone parameters at element centroids.

        Returns dict with keys: PK, PN, PI — each (n_elements, n_uz_layers).
        PK uses texture mixing; PN, PI use direct kriging.
        """
        if self._pc_elem is None:
            raise RuntimeError("Texture field not computed.")

        uz = self._pilot_points.get("unsaturated")
        if uz is None:
            raise RuntimeError("No unsaturated pilot points loaded.")

        # Compute kriging factors for elements
        self.precompute_kriging_factors("unsaturated", "elements")
        endmembers = self.krige_endmembers("unsaturated", "elements")

        pc = self._pc_elem  # Use layers 0 and 1 for UZ layers
        result = {}

        n_uz = 2  # typically 2 UZ layers
        pk = np.zeros((self._n_elements, n_uz))
        pn = np.zeros((self._n_elements, n_uz))
        pi = np.zeros((self._n_elements, n_uz))

        for lyr in range(n_uz):
            pk_c = endmembers.get(f"PK_C_L{lyr + 1}")
            pk_f = endmembers.get(f"PK_F_L{lyr + 1}")
            pn_val = endmembers.get(f"PN_L{lyr + 1}")
            pi_val = endmembers.get(f"PI_L{lyr + 1}")

            if pk_c is not None and pk_f is not None:
                pc_lyr = pc[:, lyr] if lyr < pc.shape[1] else pc[:, 0]
                pk[:, lyr] = self.mix_power_law(
                    pc_lyr,
                    pk_c,
                    pk_f,
                    self.mixing.uzpk_power,
                )
            if pn_val is not None:
                pn[:, lyr] = pn_val
            if pi_val is not None:
                pi[:, lyr] = pi_val

        result["PK"] = pk
        result["PN"] = pn
        result["PI"] = pi

        logger.info("Computed UZ parameters at %d elements x %d layers", self._n_elements, n_uz)
        return result

    def compute_rootzone_params(self) -> dict[str, NDArray]:
        """Compute root zone parameters at element centroids.

        Returns dict with keys: K, PondedK, FC, WP — each (n_elements,).
        K and PondedK use texture mixing; FC and WP use direct kriging.
        """
        if self._pc_elem is None:
            raise RuntimeError("Texture field not computed.")

        rz = self._pilot_points.get("rootzone")
        if rz is None:
            raise RuntimeError("No rootzone pilot points loaded.")

        self.precompute_kriging_factors("rootzone", "elements")
        endmembers = self.krige_endmembers("rootzone", "elements")

        pc = self._pc_elem[:, 0]  # Use top layer texture for root zone
        result = {}

        # Texture-mixed
        result["K"] = self.mix_power_law(
            pc,
            endmembers["K_C"],
            endmembers["K_F"],
            self.mixing.rzk_power,
        )
        result["PondedK"] = self.mix_power_law(
            pc,
            endmembers["PondedK_C"],
            endmembers["PondedK_F"],
            self.mixing.rzpk_power,
        )

        # Direct kriging
        result["FC"] = endmembers["FC"]
        result["WP"] = endmembers["WP"]

        logger.info("Computed RZ parameters at %d elements", self._n_elements)
        return result

    # ------------------------------------------------------------------
    # IWFM file writing (uses pyiwfm native readers/writers)
    # ------------------------------------------------------------------

    def write_groundwater_file(
        self,
        input_path: Path,
        output_path: Path,
    ) -> None:
        """Read GW file, replace aquifer params with computed values, write.

        Parameters
        ----------
        input_path : Path
            Original (template) groundwater file to read.
        output_path : Path
            Output path for modified groundwater file.
        """
        from pyiwfm.io.groundwater import GWMainFileReader

        logger.info("Reading GW template: %s", input_path)
        reader = GWMainFileReader(str(input_path))  # type: ignore[call-arg]
        config = reader.read()  # type: ignore[call-arg]

        # Compute aquifer params
        params = self.compute_aquifer_params()

        # Update config with computed values
        if config.aquifer_params is not None:
            aq = config.aquifer_params
            aq.kh[:] = params["Kh"]  # type: ignore[index]
            aq.kv[:] = params["Kv"]  # type: ignore[index]
            aq.specific_storage[:] = params["Ss"]  # type: ignore[index]
            aq.specific_yield[:] = params["Sy"]  # type: ignore[index]

            if "AqtKv" in params and hasattr(aq, "aquitard_kv"):
                aq.aquitard_kv[:] = params["AqtKv"]  # type: ignore[index]

        # Write modified config
        from pyiwfm.io.gw_main_writer import write_gw_main_file

        write_gw_main_file(config, str(output_path))
        logger.info("Wrote GW file: %s", output_path)

    def write_subsidence_file(
        self,
        input_path: Path,
        output_path: Path,
    ) -> None:
        """Read subsidence file, replace SCE/SCI with computed values, write."""
        from pyiwfm.io.gw_subsidence import SubsidenceReader

        logger.info("Reading subsidence template: %s", input_path)
        reader = SubsidenceReader(str(input_path))  # type: ignore[call-arg]
        config = reader.read()  # type: ignore[call-arg]

        params = self.compute_subsidence_params()

        # Update node params
        if hasattr(config, "node_params") and config.node_params:
            for i, np_obj in enumerate(config.node_params):
                if i < params["SCE"].shape[0]:
                    np_obj.elastic_sc = params["SCE"][i, :]  # type: ignore[assignment]
                    np_obj.inelastic_sc = params["SCI"][i, :]  # type: ignore[assignment]

        from pyiwfm.io.gw_subsidence_writer import write_subsidence_main

        write_subsidence_main(config, str(output_path))
        logger.info("Wrote subsidence file: %s", output_path)

    def write_unsaturated_file(
        self,
        input_path: Path,
        output_path: Path,
    ) -> None:
        """Read UZ file, replace PK/PN/PI with computed values, write."""
        from pyiwfm.io.unsaturated_zone import UnsatZoneMainReader

        logger.info("Reading UZ template: %s", input_path)
        reader = UnsatZoneMainReader(str(input_path))  # type: ignore[call-arg]
        config = reader.read()  # type: ignore[call-arg]

        params = self.compute_unsat_params()

        # Update element data
        if hasattr(config, "element_data"):
            for i, elem_data in enumerate(config.element_data):
                if i < params["PK"].shape[0]:
                    for lyr in range(min(len(elem_data.hyd_cond), params["PK"].shape[1])):
                        elem_data.hyd_cond[lyr] = params["PK"][i, lyr]
                        elem_data.total_porosity[lyr] = params["PN"][i, lyr]
                        elem_data.lambda_param[lyr] = params["PI"][i, lyr]

        from pyiwfm.io.unsaturated_zone_writer import UnsatZoneComponentWriter

        writer = UnsatZoneComponentWriter(config)  # type: ignore[call-arg, arg-type]
        writer.write_main(str(output_path))  # type: ignore[call-arg]
        logger.info("Wrote UZ file: %s", output_path)

    def write_rootzone_file(
        self,
        input_path: Path,
        output_path: Path,
    ) -> None:
        """Read RZ file, replace K/PondedK/FC/WP with computed values, write."""
        from pyiwfm.io.rootzone import RootZoneMainFileReader

        logger.info("Reading RZ template: %s", input_path)
        reader = RootZoneMainFileReader(str(input_path))  # type: ignore[call-arg]
        config = reader.read()  # type: ignore[call-arg]

        params = self.compute_rootzone_params()

        # Update soil params
        if hasattr(config, "soil_params"):
            sorted_elem_ids = sorted(config.soil_params.keys())
            for i, eid in enumerate(sorted_elem_ids):
                if i < len(params["K"]):
                    sp = config.soil_params[eid]
                    sp.saturated_kv = params["K"][i]
                    sp.k_ponded = params["PondedK"][i]
                    sp.field_capacity = params["FC"][i]
                    sp.wilting_point = params["WP"][i]

        from pyiwfm.io.rootzone_writer import RootZoneComponentWriter

        writer = RootZoneComponentWriter(config)  # type: ignore[call-arg, arg-type]
        writer.write_main(str(output_path))  # type: ignore[call-arg]
        logger.info("Wrote RZ file: %s", output_path)

    def write_streams_file(
        self,
        input_path: Path,
        output_path: Path,
        reach_multipliers: dict[int, float] | None = None,
    ) -> None:
        """Read streams file, apply CSTRM multipliers, write."""
        from pyiwfm.io.streams import StreamsReader  # type: ignore[attr-defined]

        logger.info("Reading streams template: %s", input_path)
        reader = StreamsReader(str(input_path))
        config = reader.read()

        if reach_multipliers and hasattr(config, "nodes"):
            for node_id, node in config.nodes.items():
                # Find which reach this node belongs to
                for reach_id, reach in config.reaches.items():
                    if node_id in range(reach.upstream_node, reach.downstream_node + 1):
                        if reach_id in reach_multipliers:
                            node.conductivity *= reach_multipliers[reach_id]
                        break

        from pyiwfm.io.stream_writer import StreamComponentWriter

        writer = StreamComponentWriter(config)  # type: ignore[call-arg]
        writer.write_main(str(output_path))  # type: ignore[call-arg]
        logger.info("Wrote streams file: %s", output_path)

    def write_small_watersheds_file(
        self,
        input_path: Path,
        output_path: Path,
        watershed_params: dict[int, dict[str, float]] | None = None,
    ) -> None:
        """Read small watersheds file, apply SWKS/GWKS values, write."""
        from pyiwfm.io.small_watershed import SmallWatershedMainReader

        logger.info("Reading small watersheds template: %s", input_path)
        reader = SmallWatershedMainReader(str(input_path))  # type: ignore[call-arg]
        config = reader.read()  # type: ignore[call-arg]

        if watershed_params and hasattr(config, "aquifer_params"):
            for i, aq_param in enumerate(config.aquifer_params):
                ws_id = i + 1  # 1-based
                if ws_id in watershed_params:
                    wp = watershed_params[ws_id]
                    if "SWKS" in wp:
                        aq_param.surface_flow_coeff = wp["SWKS"]
                    if "GWKS" in wp:
                        aq_param.baseflow_coeff = wp["GWKS"]

        from pyiwfm.io.small_watershed_writer import SmallWatershedComponentWriter

        writer = SmallWatershedComponentWriter(config)  # type: ignore[call-arg, arg-type]
        writer.write_main(str(output_path))  # type: ignore[call-arg]
        logger.info("Wrote small watersheds file: %s", output_path)

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def save_texture_cache(self, path: Path) -> None:
        """Save PcNode and PcElem to numpy files for reuse."""
        path = Path(path)
        if self._pc_node is not None:
            np.save(str(path / "pcnode_cache.npy"), self._pc_node)
        if self._pc_elem is not None:
            np.save(str(path / "pcelem_cache.npy"), self._pc_elem)
        logger.info("Saved texture cache to %s", path)

    def load_texture_cache(self, path: Path) -> bool:
        """Load PcNode and PcElem from numpy files. Returns True if loaded."""
        path = Path(path)
        pc_node_path = path / "pcnode_cache.npy"
        pc_elem_path = path / "pcelem_cache.npy"

        if pc_node_path.exists() and pc_elem_path.exists():
            self._pc_node = np.load(str(pc_node_path))
            self._pc_elem = np.load(str(pc_elem_path))
            logger.info("Loaded texture cache from %s", path)
            return True
        return False

    def save_kriging_factors(self, path: Path) -> None:
        """Save kriging factor matrices for reuse."""
        path = Path(path)
        if self._factors_node is not None:
            np.save(str(path / "factors_node.npy"), self._factors_node)
        if self._factors_elem is not None:
            np.save(str(path / "factors_elem.npy"), self._factors_elem)
        logger.info("Saved kriging factors to %s", path)

    def load_kriging_factors(self, path: Path) -> bool:
        """Load kriging factor matrices. Returns True if loaded."""
        path = Path(path)
        fn = path / "factors_node.npy"
        fe = path / "factors_elem.npy"

        loaded = False
        if fn.exists():
            self._factors_node = np.load(str(fn))
            loaded = True
        if fe.exists():
            self._factors_elem = np.load(str(fe))
            loaded = True
        if loaded:
            logger.info("Loaded kriging factors from %s", path)
        return loaded
