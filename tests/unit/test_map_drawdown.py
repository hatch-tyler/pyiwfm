"""Tests for ``pyiwfm.visualization.map_drawdown`` (Phase 2 drawdown)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection

from pyiwfm.core.mesh import AppGrid, Element, Node
from pyiwfm.core.model import IWFMModel
from pyiwfm.io.drawdown import DrawdownSnapshot
from pyiwfm.visualization.map_drawdown import (
    export_drawdown_geojson,
    plot_drawdown_map,
)


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


def _make_model(n_nodes: int = 5) -> IWFMModel:
    """Build a minimal model with ``n_nodes`` mesh nodes on a row."""
    nodes = {i: Node(id=i, x=float(i * 10), y=0.0) for i in range(1, n_nodes + 1)}
    elements = {1: Element(id=1, vertices=[1, 2, 3])}
    mesh = AppGrid(nodes=nodes, elements=elements)
    return IWFMModel(name="test", mesh=mesh)


def _make_snapshot(
    n_nodes: int = 5,
    *,
    kind: str = "single",
    timestep: int = 1,
    layer: int = 1,
    nan_indices: list[int] | None = None,
) -> DrawdownSnapshot:
    drawdown = np.linspace(1.0, float(n_nodes), n_nodes)
    if nan_indices:
        for i in nan_indices:
            drawdown[i] = np.nan
    return DrawdownSnapshot(
        timestep=timestep,
        layer=layer,
        reference_timestep=0,
        kind=kind,
        time_label=f"2024-01-{timestep + 1:02d}",
        node_ids=np.arange(1, n_nodes + 1, dtype=np.int32),
        drawdown=drawdown,
        n_nodes=n_nodes,
    )


class TestPlotDrawdownMap:
    def test_returns_axes_with_scatter(self):
        model = _make_model()
        snap = _make_snapshot()
        ax = plot_drawdown_map(snap, model, show_colorbar=False)
        assert isinstance(ax, Axes)
        scs = [c for c in ax.collections if isinstance(c, PathCollection)]
        assert len(scs) == 1
        # 5 finite-drawdown nodes
        assert len(scs[0].get_offsets()) == 5

    def test_dry_cells_omitted(self):
        model = _make_model(n_nodes=5)
        snap = _make_snapshot(n_nodes=5, nan_indices=[1, 3])
        ax = plot_drawdown_map(snap, model, show_colorbar=False)
        sc = [c for c in ax.collections if isinstance(c, PathCollection)][0]
        # 5 nodes - 2 dry = 3
        assert len(sc.get_offsets()) == 3

    def test_no_mesh_raises(self):
        model = IWFMModel(name="bare")
        snap = _make_snapshot()
        with pytest.raises(ValueError, match=r"\.mesh"):
            plot_drawdown_map(snap, model, show_colorbar=False)

    def test_no_matching_nodes_raises(self):
        model = _make_model(n_nodes=3)
        # Snapshot references nodes 100..104 not in mesh; everything dropped
        snap = DrawdownSnapshot(
            timestep=0,
            layer=1,
            reference_timestep=0,
            kind="single",
            time_label="t0",
            node_ids=np.array([100, 101, 102], dtype=np.int32),
            drawdown=np.array([1.0, 2.0, 3.0]),
            n_nodes=3,
        )
        with pytest.raises(ValueError, match="could be matched"):
            plot_drawdown_map(snap, model, show_colorbar=False)

    def test_title_includes_metadata(self):
        model = _make_model()
        snap = _make_snapshot(kind="max", timestep=-1, layer=2)
        ax = plot_drawdown_map(snap, model, show_colorbar=False)
        title = ax.get_title()
        assert "max" in title
        assert "layer 2" in title

    def test_uses_supplied_ax(self):
        model = _make_model()
        snap = _make_snapshot()
        fig, ax = plt.subplots()
        result = plot_drawdown_map(snap, model, ax=ax, show_colorbar=False)
        assert result is ax


class TestExportDrawdownGeojson:
    def test_writes_one_point_per_finite_node(self, tmp_path: Path):
        model = _make_model(n_nodes=5)
        snap = _make_snapshot(n_nodes=5)
        out = tmp_path / "drawdown.geojson"

        result = export_drawdown_geojson(snap, model, out)

        assert result == out
        payload = json.loads(out.read_text())
        assert payload["type"] == "FeatureCollection"
        assert len(payload["features"]) == 5
        for feat in payload["features"]:
            assert feat["geometry"]["type"] == "Point"

    def test_dry_cells_omitted_from_geojson(self, tmp_path: Path):
        model = _make_model(n_nodes=5)
        snap = _make_snapshot(n_nodes=5, nan_indices=[0, 2])
        out = tmp_path / "out.geojson"
        export_drawdown_geojson(snap, model, out)
        payload = json.loads(out.read_text())
        assert len(payload["features"]) == 3
        # NaN node IDs (1 and 3) are NOT in the output
        ids = [f["properties"]["node_id"] for f in payload["features"]]
        assert 1 not in ids
        assert 3 not in ids

    def test_feature_properties(self, tmp_path: Path):
        model = _make_model()
        snap = _make_snapshot(kind="max", timestep=-1, layer=2)
        out = tmp_path / "out.geojson"
        export_drawdown_geojson(snap, model, out)
        payload = json.loads(out.read_text())
        f0 = payload["features"][0]
        # Per-feature: node_id, layer, drawdown
        assert "node_id" in f0["properties"]
        assert "layer" in f0["properties"]
        assert "drawdown" in f0["properties"]
        assert f0["properties"]["layer"] == 2
        # Top-level metadata reflects the snapshot kind
        assert payload["properties"]["kind"] == "max"
        assert payload["properties"]["timestep"] == -1  # max-snapshot sentinel

    def test_crs_recorded(self, tmp_path: Path):
        model = _make_model()
        snap = _make_snapshot()
        out = tmp_path / "out.geojson"
        export_drawdown_geojson(snap, model, out, crs="EPSG:26910")
        payload = json.loads(out.read_text())
        assert payload["crs"]["properties"]["name"] == "EPSG:26910"

    def test_no_mesh_raises(self):
        model = IWFMModel(name="bare")
        snap = _make_snapshot()
        with pytest.raises(ValueError, match=r"\.mesh"):
            export_drawdown_geojson(snap, model, "/tmp/x.geojson")
