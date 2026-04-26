"""Spatial maps and GIS export for stream depletion (Phase 2.2.a-iii / 2.2.a-iv).

Renders the stream network of an :class:`~pyiwfm.core.model.IWFMModel`
colored by depletion, and exports the same data as GeoJSON for QGIS /
web maps.

Two granularities:

- :func:`plot_depletion_map` / :func:`export_depletion_geojson` —
  reach-level (one polyline per reach, colored by reach metric) over
  a :class:`~pyiwfm.io.stream_depletion.StreamDepletionReport`.
- :func:`plot_stream_node_depletion_map` /
  :func:`export_stream_node_depletion_geojson` — node-level (one
  point per stream node, sized/colored by node metric) over a
  :class:`~pyiwfm.io.stream_depletion.StreamNodeDepletionReport`.

Time-series plots over the same reports live in
:mod:`pyiwfm.visualization.plot_depletion`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

if TYPE_CHECKING:
    from pyiwfm.core.model import IWFMModel
    from pyiwfm.io.stream_depletion import (
        StreamDepletionReport,
        StreamNodeDepletionReport,
    )


_METRIC_ATTR = {
    "max": "max_depletion",
    "total": "total_depletion",
}


def _reach_polylines(
    report: StreamDepletionReport,
    model: IWFMModel,
) -> list[tuple[int, list[tuple[float, float]]]]:
    """For each reach in ``report``, return its (reach_id, [(x, y), ...]) polyline.

    Coordinates come from the linked groundwater node in
    :class:`~pyiwfm.components.stream.AppStream`. Stream nodes without a
    linked GW node are skipped. Reaches that resolve to fewer than 2
    points are skipped (can't draw a line).

    Reaches in the report but missing from the stream component are also
    skipped silently — the caller's stream model may have been loaded
    differently. The list of dropped reaches is logged via the warnings
    mechanism.
    """
    if model.streams is None or model.mesh is None:
        raise ValueError(
            "model must have both .streams and .mesh loaded; "
            "use IWFMModel.from_simulation_with_preprocessor"
        )

    polylines: list[tuple[int, list[tuple[float, float]]]] = []
    nodes = model.mesh.nodes
    for r in report.results:
        if r.reach_id not in model.streams.reaches:
            continue
        coords: list[tuple[float, float]] = []
        for stream_node in model.streams.get_nodes_in_reach(r.reach_id):
            gw_id = stream_node.gw_node
            if gw_id is None:
                continue
            gw_node = nodes.get(gw_id)
            if gw_node is None:
                continue
            coords.append((float(gw_node.x), float(gw_node.y)))
        if len(coords) >= 2:
            polylines.append((r.reach_id, coords))
    return polylines


def plot_depletion_map(
    report: StreamDepletionReport,
    model: IWFMModel,
    *,
    metric: Literal["max", "total"] = "max",
    ax: Axes | None = None,
    cmap: str = "RdBu_r",
    linewidth: float = 2.5,
    show_colorbar: bool = True,
    title: str | None = None,
) -> Axes:
    """Render the stream network colored by per-reach depletion metric.

    Parameters
    ----------
    report
        The depletion report.
    model
        Loaded model — must have both ``.streams`` and ``.mesh``.
    metric
        ``"max"`` or ``"total"`` depletion. Sets both the color values
        and the colorbar label.
    ax
        Existing matplotlib Axes. When ``None``, creates one.
    cmap
        Matplotlib colormap name. Default ``"RdBu_r"`` is symmetric so
        positive depletion (red) and negative gain (blue) read clearly.
    linewidth
        Width of the rendered reach polylines.
    show_colorbar
        Whether to attach a colorbar.
    title
        Optional title; defaults to ``"Stream depletion map (<metric>)"``.

    Returns
    -------
    matplotlib.axes.Axes
        The axes the network was drawn on.

    Raises
    ------
    ValueError
        If ``metric`` is unknown, if the model is missing required
        components, or if no reaches in the report could be matched
        against the stream network.
    """
    if metric not in _METRIC_ATTR:
        raise ValueError(f"metric must be 'max' or 'total', got {metric!r}")

    polylines = _reach_polylines(report, model)
    if not polylines:
        raise ValueError(
            "no reaches in the depletion report could be matched against the "
            "stream network — check that stream nodes have linked GW nodes"
        )

    # Map reach_id -> metric value
    attr = _METRIC_ATTR[metric]
    values_by_id = {r.reach_id: float(getattr(r, attr)) for r in report.results}
    segments = [coords for _, coords in polylines]
    values = np.array([values_by_id[rid] for rid, _ in polylines])

    # Symmetric color range about zero so the diverging colormap reads clearly
    vmax = max(1e-12, float(np.max(np.abs(values))))
    vmin = -vmax

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    lc = LineCollection(segments, cmap=cmap, linewidths=linewidth)
    lc.set_array(values)
    lc.set_clim(vmin, vmax)
    ax.add_collection(lc)
    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title or f"Stream depletion map ({metric})")
    ax.grid(True, alpha=0.2)

    if show_colorbar:
        cbar = plt.colorbar(lc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(f"{metric} depletion")

    return ax


def export_depletion_geojson(
    report: StreamDepletionReport,
    model: IWFMModel,
    output_path: str | Path,
    *,
    metric: Literal["max", "total"] = "max",
    crs: str | None = None,
) -> Path:
    """Write a GeoJSON FeatureCollection: one LineString per reach.

    Each feature carries ``reach_id``, ``reach_name``, ``max_depletion``,
    ``total_depletion``, and ``max_depletion_timestep`` properties so
    downstream tools (QGIS, the web viewer's GeoJSON ingest) can color or
    filter by any of them.

    Parameters
    ----------
    report
        The depletion report.
    model
        Loaded model — must have both ``.streams`` and ``.mesh``.
    output_path
        Destination ``.geojson`` path.
    metric
        Currently only used for the message in the file's top-level
        ``properties.metric_emphasized`` field; all metrics are always
        included in each feature so consumers can pick.
    crs
        Optional CRS string (e.g. ``"EPSG:26910"``) to record in the
        GeoJSON. When ``None``, no CRS is written and consumers should
        treat coordinates as model-native. The model's
        :attr:`~pyiwfm.core.model.IWFMModel.metadata` may contain a
        ``"crs"`` entry to use here.

    Returns
    -------
    pathlib.Path
        The path written.
    """
    if metric not in _METRIC_ATTR:
        raise ValueError(f"metric must be 'max' or 'total', got {metric!r}")

    polylines = _reach_polylines(report, model)
    by_id = {r.reach_id: r for r in report.results}

    features = []
    for reach_id, coords in polylines:
        r = by_id[reach_id]
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [list(c) for c in coords],
                },
                "properties": {
                    "reach_id": r.reach_id,
                    "reach_name": r.reach_name,
                    "max_depletion": float(r.max_depletion),
                    "total_depletion": float(r.total_depletion),
                    "max_depletion_timestep": int(r.max_depletion_timestep),
                },
            }
        )

    payload: dict[str, object] = {
        "type": "FeatureCollection",
        "features": features,
        "properties": {
            "n_reaches": len(features),
            "metric_emphasized": metric,
            "n_timesteps": report.n_timesteps,
        },
    }
    if crs is not None:
        payload["crs"] = {
            "type": "name",
            "properties": {"name": crs},
        }

    output_path = Path(output_path)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return output_path


# ---------------------------------------------------------------------------
# Node-level (Phase 2.2.a-iv)
# ---------------------------------------------------------------------------


def _stream_node_points(
    report: StreamNodeDepletionReport,
    model: IWFMModel,
) -> list[tuple[int, float, float, int | None]]:
    """For each node in ``report``, return ``(stream_node_id, x, y, gw_node_id)``.

    Coordinates come from the linked groundwater node on the AppStream
    component. Stream nodes without a linked GW node are skipped — there's
    nowhere to plot them.

    Returns an empty list silently when the model lacks streams/mesh; the
    caller is responsible for producing a useful error.
    """
    points: list[tuple[int, float, float, int | None]] = []
    if model.streams is None or model.mesh is None:
        return points
    nodes = model.mesh.nodes
    stream_nodes = model.streams.nodes
    for r in report.results:
        sn = stream_nodes.get(r.stream_node_id)
        if sn is None:
            continue
        gw_id = sn.gw_node
        if gw_id is None:
            continue
        gw_node = nodes.get(gw_id)
        if gw_node is None:
            continue
        points.append((r.stream_node_id, float(gw_node.x), float(gw_node.y), gw_id))
    return points


def plot_stream_node_depletion_map(
    report: StreamNodeDepletionReport,
    model: IWFMModel,
    *,
    metric: Literal["max", "total"] = "max",
    ax: Axes | None = None,
    cmap: str = "RdBu_r",
    size_by_magnitude: bool = True,
    base_size: float = 30.0,
    show_colorbar: bool = True,
    title: str | None = None,
) -> Axes:
    """Scatter map: one point per stream node, sized/colored by depletion.

    Parameters
    ----------
    report
        The :class:`StreamNodeDepletionReport`.
    model
        Loaded model — must have both ``.streams`` and ``.mesh``.
    metric
        ``"max"`` (largest absolute depletion across timesteps) or
        ``"total"`` (cumulative depletion at the last timestep).
    ax
        Existing matplotlib Axes. When ``None``, creates one.
    cmap
        Matplotlib colormap. Default ``"RdBu_r"`` reads positive depletion
        as red, negative (gain) as blue.
    size_by_magnitude
        If ``True`` (default), point area scales with ``|metric|`` so
        hot-spots draw the eye. If ``False``, all points are the same size
        (``base_size``).
    base_size
        Base marker area in points² when ``size_by_magnitude=False``, or
        the size at maximum magnitude when ``True``.
    show_colorbar
        Whether to attach a colorbar.
    title
        Optional title; defaults to ``"Stream node depletion map (<metric>)"``.

    Returns
    -------
    matplotlib.axes.Axes
        The axes the points were drawn on.

    Raises
    ------
    ValueError
        If ``metric`` is unknown, the model lacks streams/mesh, or no
        nodes in the report could be matched against the stream
        component (typically because no stream node has a linked GW
        node — re-check the model's stream-GW connector).
    """
    if metric not in _METRIC_ATTR:
        raise ValueError(f"metric must be 'max' or 'total', got {metric!r}")
    if model.streams is None or model.mesh is None:
        raise ValueError(
            "model must have both .streams and .mesh loaded; "
            "use IWFMModel.from_simulation_with_preprocessor"
        )

    points = _stream_node_points(report, model)
    if not points:
        raise ValueError(
            "no stream nodes in the depletion report could be matched "
            "against the model's stream network — check the stream-GW "
            "connector and that stream node IDs in the budget match the "
            "model's stream nodes"
        )

    attr = _METRIC_ATTR[metric]
    by_id = {r.stream_node_id: r for r in report.results}
    xs = np.array([p[1] for p in points])
    ys = np.array([p[2] for p in points])
    values = np.array([float(getattr(by_id[p[0]], attr)) for p in points])

    vmax = max(1e-12, float(np.max(np.abs(values))))
    vmin = -vmax

    if size_by_magnitude:
        # Scale 0..vmax → small..base_size (keep a floor so zero-depletion
        # points are still visible)
        sizes = base_size * (0.2 + 0.8 * np.abs(values) / vmax)
    else:
        sizes = np.full_like(values, base_size, dtype=np.float64)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    sc = ax.scatter(
        xs,
        ys,
        c=values,
        s=sizes,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        edgecolors="black",
        linewidths=0.3,
    )
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title or f"Stream node depletion map ({metric})")
    ax.grid(True, alpha=0.2)

    if show_colorbar:
        cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(f"{metric} depletion")

    return ax


def export_stream_node_depletion_geojson(
    report: StreamNodeDepletionReport,
    model: IWFMModel,
    output_path: str | Path,
    *,
    crs: str | None = None,
) -> Path:
    """Write a GeoJSON FeatureCollection: one Point per stream node.

    Each feature carries:

    - ``stream_node_id``
    - ``gw_node_id`` (the linked groundwater node, or ``None`` if
      unconnected — but unconnected nodes are skipped before reaching
      this point so this is always populated in practice)
    - ``max_depletion``, ``total_depletion``, ``max_depletion_timestep``

    Compatible with QGIS and the web viewer's existing GeoJSON ingest.
    Use the ``crs`` argument to record the model CRS if known (e.g.
    ``"EPSG:26910"``); otherwise consumers should treat coordinates as
    model-native.

    Parameters
    ----------
    report
        The :class:`StreamNodeDepletionReport`.
    model
        Loaded model.
    output_path
        Destination ``.geojson`` path.
    crs
        Optional CRS string.

    Returns
    -------
    pathlib.Path
        The path written.

    Raises
    ------
    ValueError
        Model lacks streams/mesh.
    """
    if model.streams is None or model.mesh is None:
        raise ValueError(
            "model must have both .streams and .mesh loaded; "
            "use IWFMModel.from_simulation_with_preprocessor"
        )

    points = _stream_node_points(report, model)
    by_id = {r.stream_node_id: r for r in report.results}

    features = []
    for stream_node_id, x, y, gw_id in points:
        r = by_id[stream_node_id]
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [x, y]},
                "properties": {
                    "stream_node_id": stream_node_id,
                    "gw_node_id": gw_id,
                    "max_depletion": float(r.max_depletion),
                    "total_depletion": float(r.total_depletion),
                    "max_depletion_timestep": int(r.max_depletion_timestep),
                },
            }
        )

    payload: dict[str, object] = {
        "type": "FeatureCollection",
        "features": features,
        "properties": {
            "n_stream_nodes": len(features),
            "n_timesteps": report.n_timesteps,
        },
    }
    if crs is not None:
        payload["crs"] = {"type": "name", "properties": {"name": crs}}

    output_path = Path(output_path)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return output_path


def plot_depletion_along_reach(
    report: StreamNodeDepletionReport,
    model: IWFMModel,
    reach_id: int,
    *,
    metric: Literal["max", "total"] = "max",
    ax: Axes | None = None,
    title: str | None = None,
) -> Axes:
    """Longitudinal depletion profile within a single reach.

    Walks the reach's stream nodes in upstream-to-downstream order and
    plots the chosen depletion metric vs. node index. Useful for
    showing where along a reach the pumping signal manifests as
    stream-aquifer interaction change.

    Parameters
    ----------
    report
        The :class:`StreamNodeDepletionReport` (per-stream-node).
    model
        Loaded model — must have ``.streams`` so the reach's node
        ordering is known.
    reach_id
        1-based reach ID.
    metric
        ``"max"`` or ``"total"`` depletion.
    ax
        Existing axes.
    title
        Optional title; defaults to ``"Depletion along reach <id> (<metric>)"``.

    Returns
    -------
    matplotlib.axes.Axes

    Raises
    ------
    ValueError
        Model lacks streams, ``reach_id`` not present, or no stream
        nodes from the reach are present in the report.
    """
    if metric not in _METRIC_ATTR:
        raise ValueError(f"metric must be 'max' or 'total', got {metric!r}")
    if model.streams is None:
        raise ValueError("model must have .streams loaded")
    if reach_id not in model.streams.reaches:
        raise ValueError(
            f"reach_id {reach_id} not in model.streams.reaches "
            f"(available: {sorted(model.streams.reaches.keys())})"
        )

    reach_nodes = model.streams.get_nodes_in_reach(reach_id)
    by_id = {r.stream_node_id: r for r in report.results}
    attr = _METRIC_ATTR[metric]

    indices: list[int] = []
    values: list[float] = []
    sn_ids: list[int] = []
    for i, sn in enumerate(reach_nodes):
        if sn.id in by_id:
            indices.append(i)
            values.append(float(getattr(by_id[sn.id], attr)))
            sn_ids.append(sn.id)
    if not values:
        raise ValueError(
            f"no stream nodes from reach {reach_id} are present in the depletion report"
        )

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    ax.plot(indices, values, marker="o", linestyle="-", color="steelblue")
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("stream node index along reach (upstream → downstream)")
    ax.set_ylabel(f"{metric} depletion")
    ax.set_title(title or f"Depletion along reach {reach_id} ({metric}, {len(values)} nodes)")
    ax.grid(True, alpha=0.3)
    return ax
