"""Spatial drawdown maps and GIS export (Phase 2 drawdown).

Renders a :class:`~pyiwfm.io.drawdown.DrawdownSnapshot` (drawdown at
every node at a single timestep, or per-node max across all timesteps)
as a colored map, and exports the same data as GeoJSON points.

Time-series plots over a per-location report live in
:mod:`pyiwfm.visualization.plot_drawdown`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402

if TYPE_CHECKING:
    from pyiwfm.core.model import IWFMModel
    from pyiwfm.io.drawdown import DrawdownSnapshot


def _node_coords(
    snapshot: DrawdownSnapshot,
    model: IWFMModel,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (x, y, drawdown) arrays for nodes that match the model mesh.

    Filters out nodes whose drawdown is NaN (dry cells) and nodes whose
    ID isn't present in ``model.mesh.nodes``. Coordinates come from
    ``Node.x`` / ``Node.y``.
    """
    if model.mesh is None:
        raise ValueError(
            "model must have .mesh loaded; use IWFMModel.from_simulation_with_preprocessor"
        )
    nodes = model.mesh.nodes
    xs: list[float] = []
    ys: list[float] = []
    vals: list[float] = []
    for node_id, dd in zip(snapshot.node_ids, snapshot.drawdown, strict=True):
        if not np.isfinite(dd):
            continue
        node = nodes.get(int(node_id))
        if node is None:
            continue
        xs.append(float(node.x))
        ys.append(float(node.y))
        vals.append(float(dd))
    return np.asarray(xs), np.asarray(ys), np.asarray(vals)


def plot_drawdown_map(
    snapshot: DrawdownSnapshot,
    model: IWFMModel,
    *,
    ax: Axes | None = None,
    cmap: str = "RdBu_r",
    point_size: float = 25.0,
    show_colorbar: bool = True,
    title: str | None = None,
) -> Axes:
    """Render a per-node drawdown map (cone of depression).

    Each model node is a colored point; the symmetric ``RdBu_r`` colormap
    reads positive drawdown (water-level decline) as red and negative
    drawdown (water-level rise) as blue. Dry cells (NaN drawdown) are
    omitted.

    Parameters
    ----------
    snapshot
        A :class:`DrawdownSnapshot` from
        :meth:`DrawdownComputer.build_snapshot` or
        :meth:`DrawdownComputer.build_max_snapshot`.
    model
        Loaded :class:`IWFMModel` — must have ``.mesh`` for node coordinates.
    ax
        Existing matplotlib Axes. When ``None``, creates one.
    cmap
        Matplotlib colormap. Default ``"RdBu_r"`` is symmetric.
    point_size
        Marker size in points². Default 25.
    show_colorbar
        Whether to attach a colorbar.
    title
        Plot title; defaults to ``"Drawdown map (<kind>, layer <n>, <time>)"``.

    Returns
    -------
    matplotlib.axes.Axes

    Raises
    ------
    ValueError
        ``model.mesh`` is missing or no snapshot nodes match the mesh.
    """
    xs, ys, vals = _node_coords(snapshot, model)
    if vals.size == 0:
        raise ValueError(
            "no nodes in the drawdown snapshot could be matched against the "
            "model mesh — every node was either dry or not in mesh.nodes"
        )

    vmax = max(1e-12, float(np.max(np.abs(vals))))
    vmin = -vmax

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    sc = ax.scatter(
        xs,
        ys,
        c=vals,
        s=point_size,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        edgecolors="black",
        linewidths=0.2,
    )
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(
        title or f"Drawdown map ({snapshot.kind}, layer {snapshot.layer}, {snapshot.time_label})"
    )
    ax.grid(True, alpha=0.2)

    if show_colorbar:
        cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("drawdown (positive = decline)")

    return ax


def export_drawdown_geojson(
    snapshot: DrawdownSnapshot,
    model: IWFMModel,
    output_path: str | Path,
    *,
    crs: str | None = None,
) -> Path:
    """Write a GeoJSON FeatureCollection: one Point per node with drawdown.

    Each feature carries ``node_id``, ``layer``, ``drawdown``, and the
    snapshot metadata (``kind``, ``timestep``, ``reference_timestep``,
    ``time_label``). Dry-cell (NaN) nodes are omitted.

    Parameters
    ----------
    snapshot
        Drawdown snapshot.
    model
        Loaded model — must have ``.mesh`` for node coordinates.
    output_path
        Destination ``.geojson`` path.
    crs
        Optional CRS string (e.g. ``"EPSG:26910"``).

    Returns
    -------
    pathlib.Path
        Path written.

    Raises
    ------
    ValueError
        Model lacks ``.mesh``.
    """
    if model.mesh is None:
        raise ValueError(
            "model must have .mesh loaded; use IWFMModel.from_simulation_with_preprocessor"
        )

    nodes = model.mesh.nodes
    features = []
    for node_id, dd in zip(snapshot.node_ids, snapshot.drawdown, strict=True):
        if not np.isfinite(dd):
            continue
        node = nodes.get(int(node_id))
        if node is None:
            continue
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(node.x), float(node.y)],
                },
                "properties": {
                    "node_id": int(node_id),
                    "layer": int(snapshot.layer),
                    "drawdown": float(dd),
                },
            }
        )

    payload: dict[str, object] = {
        "type": "FeatureCollection",
        "features": features,
        "properties": {
            "n_nodes": len(features),
            "kind": snapshot.kind,
            "timestep": int(snapshot.timestep),
            "reference_timestep": int(snapshot.reference_timestep),
            "time_label": snapshot.time_label,
        },
    }
    if crs is not None:
        payload["crs"] = {"type": "name", "properties": {"name": crs}}

    output_path = Path(output_path)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return output_path
