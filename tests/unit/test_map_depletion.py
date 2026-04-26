"""Tests for ``pyiwfm.visualization.map_depletion`` (Phase 2.2.a-iii / 2.2.a-iv)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, PathCollection

from pyiwfm.components.stream import AppStream, StrmNode, StrmReach
from pyiwfm.core.mesh import AppGrid, Element, Node
from pyiwfm.core.model import IWFMModel
from pyiwfm.io.stream_depletion import (
    StreamDepletionReport,
    StreamDepletionResult,
    StreamNodeDepletionReport,
    StreamNodeDepletionResult,
)
from pyiwfm.visualization.map_depletion import (
    export_depletion_geojson,
    export_stream_node_depletion_geojson,
    plot_depletion_along_reach,
    plot_depletion_map,
    plot_stream_node_depletion_map,
)


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


def _make_model_with_streams() -> IWFMModel:
    """Build a 6-node mesh + a 2-reach stream network where each stream
    node is linked to a GW node with a known coordinate."""
    nodes = {
        1: Node(id=1, x=0.0, y=0.0),
        2: Node(id=2, x=10.0, y=0.0),
        3: Node(id=3, x=20.0, y=0.0),
        4: Node(id=4, x=0.0, y=10.0),
        5: Node(id=5, x=10.0, y=10.0),
        6: Node(id=6, x=20.0, y=10.0),
    }
    elements = {
        1: Element(id=1, vertices=[1, 2, 5, 4]),
        2: Element(id=2, vertices=[2, 3, 6, 5]),
    }
    mesh = AppGrid(nodes=nodes, elements=elements)

    streams = AppStream()
    # Reach 1: stream nodes 101, 102, 103 mapped to GW nodes 1, 2, 3
    for sn_id, gw_id in [(101, 1), (102, 2), (103, 3)]:
        streams.add_node(StrmNode(id=sn_id, x=0.0, y=0.0, gw_node=gw_id, reach_id=1))
    streams.add_reach(
        StrmReach(
            id=1,
            upstream_node=101,
            downstream_node=103,
            nodes=[101, 102, 103],
            name="Main",
        )
    )
    # Reach 2: stream nodes 201, 202 mapped to GW nodes 4, 5
    for sn_id, gw_id in [(201, 4), (202, 5)]:
        streams.add_node(StrmNode(id=sn_id, x=0.0, y=0.0, gw_node=gw_id, reach_id=2))
    streams.add_reach(
        StrmReach(
            id=2,
            upstream_node=201,
            downstream_node=202,
            nodes=[201, 202],
            name="Tributary",
        )
    )

    return IWFMModel(name="test", mesh=mesh, streams=streams)


def _make_report() -> StreamDepletionReport:
    times = ["t0", "t1", "t2"]
    reaches = []
    for ri, (max_d, total_d) in enumerate([(5.0, 12.0), (1.5, 3.0)], start=1):
        depletion = np.array([max_d * 0.5, max_d, max_d * 0.7])
        cumulative = np.cumsum(depletion)
        reaches.append(
            StreamDepletionResult(
                reach_id=ri,
                reach_name=f"Reach {ri}",
                times=times,
                baseline_flow=np.array([100.0, 100.0, 100.0]),
                scenario_flow=np.array([100.0, 100.0, 100.0]) - depletion,
                depletion=depletion,
                cumulative_depletion=cumulative,
                max_depletion=max_d,
                max_depletion_timestep=1,
                total_depletion=total_d,
            )
        )
    return StreamDepletionReport(
        results=reaches,
        n_reaches=2,
        n_timesteps=3,
        total_max_depletion=5.0,
        total_cumulative_depletion=15.0,
    )


class TestPlotDepletionMap:
    def test_returns_axes(self):
        model = _make_model_with_streams()
        report = _make_report()
        ax = plot_depletion_map(report, model, show_colorbar=False)
        assert isinstance(ax, Axes)

    def test_one_line_collection_with_two_reaches(self):
        model = _make_model_with_streams()
        report = _make_report()
        ax = plot_depletion_map(report, model, show_colorbar=False)
        # The map adds exactly one LineCollection containing both reaches
        lcs = [c for c in ax.collections if isinstance(c, LineCollection)]
        assert len(lcs) == 1
        assert len(lcs[0].get_segments()) == 2

    def test_metric_max_vs_total(self):
        model = _make_model_with_streams()
        report = _make_report()
        ax_max = plot_depletion_map(report, model, metric="max", show_colorbar=False)
        ax_total = plot_depletion_map(report, model, metric="total", show_colorbar=False)
        # Color values come from the chosen metric
        lc_max = [c for c in ax_max.collections if isinstance(c, LineCollection)][0]
        lc_total = [c for c in ax_total.collections if isinstance(c, LineCollection)][0]
        assert lc_max.get_array().max() == pytest.approx(5.0)
        assert lc_total.get_array().max() == pytest.approx(12.0)

    def test_unknown_metric_raises(self):
        model = _make_model_with_streams()
        report = _make_report()
        with pytest.raises(ValueError, match="metric must be"):
            plot_depletion_map(report, model, metric="mean", show_colorbar=False)  # type: ignore[arg-type]

    def test_missing_streams_raises(self):
        model = IWFMModel(name="bare")  # no mesh, no streams
        report = _make_report()
        with pytest.raises(ValueError, match=r"\.streams"):
            plot_depletion_map(report, model, show_colorbar=False)

    def test_no_matching_reaches_raises(self):
        # A report with reach IDs that don't exist in the model's stream component
        model = _make_model_with_streams()
        bogus = _make_report()
        bogus.results[0].reach_id = 99
        bogus.results[1].reach_id = 100
        with pytest.raises(ValueError, match="could be matched"):
            plot_depletion_map(bogus, model, show_colorbar=False)


class TestExportGeojson:
    def test_writes_feature_collection(self, tmp_path: Path):
        model = _make_model_with_streams()
        report = _make_report()
        out = tmp_path / "depletion.geojson"

        result = export_depletion_geojson(report, model, out)

        assert result == out
        payload = json.loads(out.read_text())
        assert payload["type"] == "FeatureCollection"
        assert len(payload["features"]) == 2

    def test_feature_properties(self, tmp_path: Path):
        model = _make_model_with_streams()
        report = _make_report()
        out = tmp_path / "out.geojson"
        export_depletion_geojson(report, model, out)

        payload = json.loads(out.read_text())
        f0 = payload["features"][0]
        # All metrics included so consumers can pick
        assert "max_depletion" in f0["properties"]
        assert "total_depletion" in f0["properties"]
        assert "max_depletion_timestep" in f0["properties"]
        assert f0["geometry"]["type"] == "LineString"
        # Reach 1 has 3 nodes -> 3 coordinates
        assert len(f0["geometry"]["coordinates"]) == 3

    def test_crs_is_recorded_when_supplied(self, tmp_path: Path):
        model = _make_model_with_streams()
        report = _make_report()
        out = tmp_path / "out.geojson"
        export_depletion_geojson(report, model, out, crs="EPSG:26910")

        payload = json.loads(out.read_text())
        assert payload["crs"]["properties"]["name"] == "EPSG:26910"

    def test_no_crs_when_not_supplied(self, tmp_path: Path):
        model = _make_model_with_streams()
        report = _make_report()
        out = tmp_path / "out.geojson"
        export_depletion_geojson(report, model, out)

        payload = json.loads(out.read_text())
        assert "crs" not in payload


# ---------------------------------------------------------------------------
# Node-level fixtures + tests (Phase 2.2.a-iv)
# ---------------------------------------------------------------------------


def _make_node_report() -> StreamNodeDepletionReport:
    """Build a node-level fixture matching ``_make_model_with_streams``.

    The model has stream nodes 101, 102, 103 (reach 1) and 201, 202 (reach 2).
    But ``StreamNodeDepletionReport.results[i].stream_node_id`` is the
    1-based index from the budget (i.e. 1..5), not the model's stream node ID.
    To make the spatial join meaningful, we pretend the budget enumerated
    nodes 101–103 and 201–202 (i.e. node IDs match the model). This is the
    common case when each model node is a stream-budget location.
    """
    rng = np.random.default_rng(0)
    times = ["t0", "t1", "t2"]
    results = []
    # Use the actual stream node IDs from the model fixture
    for sn_id, max_d in [(101, 5.0), (102, 4.0), (103, 3.0), (201, 1.5), (202, 1.0)]:
        depletion = rng.uniform(0.5, max_d, 3)
        cumulative = np.cumsum(depletion)
        results.append(
            StreamNodeDepletionResult(
                stream_node_id=sn_id,
                times=times,
                baseline_sa_flux=np.array([10.0, 11.0, 12.0]),
                scenario_sa_flux=np.array([10.0, 11.0, 12.0]) - depletion,
                depletion=depletion,
                cumulative_depletion=cumulative,
                max_depletion=float(np.max(depletion)),
                max_depletion_timestep=int(np.argmax(depletion)),
                total_depletion=float(cumulative[-1]),
            )
        )
    return StreamNodeDepletionReport(
        results=results,
        n_stream_nodes=len(results),
        n_timesteps=3,
        total_max_depletion=max(r.max_depletion for r in results),
        total_cumulative_depletion=sum(r.total_depletion for r in results),
    )


class TestPlotStreamNodeDepletionMap:
    def test_returns_axes_with_scatter(self):
        model = _make_model_with_streams()
        report = _make_node_report()
        ax = plot_stream_node_depletion_map(report, model, show_colorbar=False)
        assert isinstance(ax, Axes)
        # One scatter PathCollection
        scs = [c for c in ax.collections if isinstance(c, PathCollection)]
        assert len(scs) == 1
        # 5 stream nodes, all linked → 5 points
        offsets = scs[0].get_offsets()
        assert len(offsets) == 5

    def test_metric_max_vs_total(self):
        model = _make_model_with_streams()
        report = _make_node_report()
        ax_max = plot_stream_node_depletion_map(report, model, metric="max", show_colorbar=False)
        ax_total = plot_stream_node_depletion_map(
            report, model, metric="total", show_colorbar=False
        )
        sc_max = [c for c in ax_max.collections if isinstance(c, PathCollection)][0]
        sc_total = [c for c in ax_total.collections if isinstance(c, PathCollection)][0]
        # Different metrics produce different value arrays
        assert not np.allclose(sc_max.get_array(), sc_total.get_array())

    def test_unknown_metric_raises(self):
        model = _make_model_with_streams()
        report = _make_node_report()
        with pytest.raises(ValueError, match="metric must be"):
            plot_stream_node_depletion_map(report, model, metric="mean", show_colorbar=False)  # type: ignore[arg-type]

    def test_missing_streams_raises(self):
        model = IWFMModel(name="bare")
        report = _make_node_report()
        with pytest.raises(ValueError, match=r"\.streams"):
            plot_stream_node_depletion_map(report, model, show_colorbar=False)

    def test_no_matching_nodes_raises(self):
        # Report references node IDs that don't exist in the model
        model = _make_model_with_streams()
        bogus = _make_node_report()
        for r in bogus.results:
            r.stream_node_id += 9000
        with pytest.raises(ValueError, match="could be matched"):
            plot_stream_node_depletion_map(bogus, model, show_colorbar=False)

    def test_size_by_magnitude_off_uses_uniform_size(self):
        model = _make_model_with_streams()
        report = _make_node_report()
        ax = plot_stream_node_depletion_map(
            report, model, size_by_magnitude=False, base_size=42.0, show_colorbar=False
        )
        sc = [c for c in ax.collections if isinstance(c, PathCollection)][0]
        sizes = sc.get_sizes()
        # All sizes equal to base_size
        assert np.allclose(sizes, 42.0)


class TestExportStreamNodeDepletionGeojson:
    def test_writes_one_point_per_node(self, tmp_path: Path):
        model = _make_model_with_streams()
        report = _make_node_report()
        out = tmp_path / "node_depletion.geojson"

        result = export_stream_node_depletion_geojson(report, model, out)

        assert result == out
        payload = json.loads(out.read_text())
        assert payload["type"] == "FeatureCollection"
        assert len(payload["features"]) == 5
        for feat in payload["features"]:
            assert feat["geometry"]["type"] == "Point"

    def test_feature_properties(self, tmp_path: Path):
        model = _make_model_with_streams()
        report = _make_node_report()
        out = tmp_path / "out.geojson"
        export_stream_node_depletion_geojson(report, model, out)

        payload = json.loads(out.read_text())
        f0 = payload["features"][0]
        for key in (
            "stream_node_id",
            "gw_node_id",
            "max_depletion",
            "total_depletion",
            "max_depletion_timestep",
        ):
            assert key in f0["properties"]

    def test_crs_recorded(self, tmp_path: Path):
        model = _make_model_with_streams()
        report = _make_node_report()
        out = tmp_path / "out.geojson"
        export_stream_node_depletion_geojson(report, model, out, crs="EPSG:26910")
        payload = json.loads(out.read_text())
        assert payload["crs"]["properties"]["name"] == "EPSG:26910"

    def test_unconnected_nodes_skipped(self, tmp_path: Path):
        # Add a stream node with no GW link → should not appear in output
        model = _make_model_with_streams()
        model.streams.add_node(StrmNode(id=999, x=0, y=0, gw_node=None, reach_id=2))
        report = _make_node_report()
        # Append a node that exists in the report but has no GW link
        report.results.append(
            StreamNodeDepletionResult(
                stream_node_id=999,
                times=["t0"],
                baseline_sa_flux=np.array([10.0]),
                scenario_sa_flux=np.array([5.0]),
                depletion=np.array([5.0]),
                cumulative_depletion=np.array([5.0]),
                max_depletion=5.0,
                max_depletion_timestep=0,
                total_depletion=5.0,
            )
        )
        report.n_stream_nodes = 6

        out = tmp_path / "out.geojson"
        export_stream_node_depletion_geojson(report, model, out)
        payload = json.loads(out.read_text())
        # Original 5 connected nodes only; node 999 dropped
        assert len(payload["features"]) == 5
        ids = [f["properties"]["stream_node_id"] for f in payload["features"]]
        assert 999 not in ids


class TestPlotDepletionAlongReach:
    def test_plots_reach_nodes_in_order(self):
        model = _make_model_with_streams()
        report = _make_node_report()
        ax = plot_depletion_along_reach(report, model, reach_id=1)
        assert isinstance(ax, Axes)
        # Reach 1 has 3 stream nodes (101, 102, 103), all in the report
        lines = [line for line in ax.get_lines() if len(line.get_xdata()) >= 2]
        assert any(len(line.get_xdata()) == 3 for line in lines)

    def test_unknown_reach_raises(self):
        model = _make_model_with_streams()
        report = _make_node_report()
        with pytest.raises(ValueError, match="not in model.streams.reaches"):
            plot_depletion_along_reach(report, model, reach_id=99)

    def test_no_overlap_raises(self):
        # Report doesn't include any of reach 2's nodes
        model = _make_model_with_streams()
        report = _make_node_report()
        # Drop node 201 and 202
        report.results = [r for r in report.results if r.stream_node_id < 200]
        with pytest.raises(ValueError, match="no stream nodes from reach"):
            plot_depletion_along_reach(report, model, reach_id=2)

    def test_unknown_metric_raises(self):
        model = _make_model_with_streams()
        report = _make_node_report()
        with pytest.raises(ValueError, match="metric must be"):
            plot_depletion_along_reach(report, model, reach_id=1, metric="mean")  # type: ignore[arg-type]

    def test_missing_streams_raises(self):
        model = IWFMModel(name="bare")
        report = _make_node_report()
        with pytest.raises(ValueError, match=r"\.streams"):
            plot_depletion_along_reach(report, model, reach_id=1)
