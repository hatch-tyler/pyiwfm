"""
CLI subcommand for stream-depletion analysis (Phase 2.2.a-v).

Wires together the depletion computation, tabular writers, time-series
plots, and spatial maps so non-Python users can produce a stakeholder-
ready report from two model runs with a single command.

Usage::

    pyiwfm depletion <baseline_dir> <scenario_dir> [options]

The baseline and scenario directories must each contain a loadable IWFM
model (the simulation main file is auto-discovered via
``cli._model_finder.find_simulation_file``). Both models must have
declared the required budget output (stream reach budget for the
default reach-level analysis, stream node budget for ``--node-level``);
:class:`~pyiwfm.io.stream_depletion.BudgetOutputMissingError` is raised
with an actionable message if not.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyiwfm.core.model import IWFMModel
    from pyiwfm.io.stream_depletion import (
        StreamDepletionReport,
        StreamNodeDepletionReport,
    )

logger = logging.getLogger(__name__)

# Plot kinds the CLI knows how to render. ``all`` is a meta-value that
# expands to every other entry.
_PLOT_KINDS = ("cumulative", "timeseries", "summary")


def add_depletion_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the ``pyiwfm depletion`` subcommand."""
    p = subparsers.add_parser(
        "depletion",
        help="Stream depletion analysis between two IWFM model runs",
        description=(
            "Compute stream depletion by comparing baseline vs scenario "
            "model results. Supports tabular export (CSV/JSON/Excel), "
            "time-series plots, and spatial maps. Default analysis is "
            "reach-level using the stream reach budget; pass --node-level "
            "to use the stream node budget instead."
        ),
    )
    p.add_argument(
        "baseline_dir",
        type=str,
        help="Path to the baseline (no-pumping) model directory.",
    )
    p.add_argument(
        "scenario_dir",
        type=str,
        help="Path to the pumping-scenario model directory.",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Tabular report path. Format chosen by extension: .csv, .json, "
            ".xlsx. If omitted, no tabular file is written."
        ),
    )
    p.add_argument(
        "--plot",
        action="append",
        choices=[*_PLOT_KINDS, "all"],
        default=None,
        help=(
            "Plot kind to render (PNG). Repeat to render multiple, or pass "
            "'all'. Files written to --plot-dir."
        ),
    )
    p.add_argument(
        "--plot-dir",
        type=str,
        default="depletion_plots",
        help="Directory for plot PNG output (default: ./depletion_plots).",
    )
    p.add_argument(
        "--map",
        action="store_true",
        help=(
            "Render the spatial depletion map: PNG + GeoJSON. For "
            "reach-level (default): one polyline per reach. With "
            "--node-level: one point per stream node. Written to --plot-dir."
        ),
    )
    p.add_argument(
        "--node-level",
        action="store_true",
        help=(
            "Use the stream NODE budget instead of the stream REACH budget. "
            "Both models must declare STNDBUDFL in their stream main file."
        ),
    )
    p.add_argument(
        "--metric",
        choices=("max", "total"),
        default="max",
        help="Metric used for map color scale (default: max).",
    )
    p.add_argument(
        "--reach-ids",
        type=str,
        default=None,
        help=(
            "Comma-separated list of 1-based reach IDs to include "
            "(reach mode only). Default: all reaches."
        ),
    )
    p.add_argument(
        "--node-ids",
        type=str,
        default=None,
        help=(
            "Comma-separated list of 1-based stream node IDs to include "
            "(--node-level only). Default: all nodes."
        ),
    )
    p.add_argument(
        "--sa-column",
        type=str,
        default=None,
        help=(
            "Override the stream-aquifer interaction column name. Default: "
            "the IWFM v5+ canonical 'Stream-Aquifer Interaction Within "
            "Model'. Use 'Gain from GW (+)' for older models."
        ),
    )
    p.add_argument(
        "--crs",
        type=str,
        default=None,
        help="CRS string to record in the GeoJSON (e.g. 'EPSG:26910').",
    )
    p.set_defaults(func=run_depletion)


def _parse_id_list(text: str | None) -> list[int] | None:
    if not text:
        return None
    try:
        return [int(part.strip()) for part in text.split(",") if part.strip()]
    except ValueError as e:
        raise SystemExit(
            f"Could not parse ID list {text!r}: {e}. Expected comma-"
            f"separated integers, e.g. --reach-ids 1,3,5"
        ) from e


def _resolve_plot_kinds(plot_args: list[str] | None) -> list[str]:
    if not plot_args:
        return []
    if "all" in plot_args:
        return list(_PLOT_KINDS)
    # Preserve user order, drop duplicates
    seen: set[str] = set()
    out: list[str] = []
    for k in plot_args:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def _load_model_from_dir(model_dir: Path, label: str) -> IWFMModel:
    """Discover and load an IWFM model from a directory.

    Loads with both simulation and preprocessor files when both exist
    so the stream component carries the GW-node connector that
    :func:`plot_depletion_map` needs to render reach polylines. Falls
    back to simulation-only when the preprocessor file isn't found.
    """
    from pyiwfm.cli._model_finder import find_preprocessor_file, find_simulation_file
    from pyiwfm.core.model import IWFMModel

    sim_file = find_simulation_file(model_dir)
    if sim_file is None:
        raise SystemExit(
            f"Could not locate the simulation main file in the {label} "
            f"directory: {model_dir}. Expected a path like "
            f"<dir>/Simulation/Simulation_MAIN.IN."
        )

    pp_file = find_preprocessor_file(model_dir)
    if pp_file is not None:
        return IWFMModel.from_simulation_with_preprocessor(
            simulation_file=sim_file, preprocessor_file=pp_file
        )
    return IWFMModel.from_simulation(sim_file)


def run_depletion(args: argparse.Namespace) -> int:
    """Execute the depletion analysis end-to-end."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from pyiwfm.io.stream_depletion import (
        DEFAULT_SA_COLUMN,
        BudgetOutputMissingError,
        compute_stream_depletion_from_models,
        compute_stream_node_depletion,
    )

    baseline_dir = Path(args.baseline_dir)
    scenario_dir = Path(args.scenario_dir)
    if not baseline_dir.is_dir():
        print(f"Error: baseline directory not found: {baseline_dir}", file=sys.stderr)
        return 1
    if not scenario_dir.is_dir():
        print(f"Error: scenario directory not found: {scenario_dir}", file=sys.stderr)
        return 1

    # ID filters / column override
    reach_ids = _parse_id_list(args.reach_ids)
    node_ids = _parse_id_list(args.node_ids)
    sa_column = args.sa_column or DEFAULT_SA_COLUMN

    # Plot/map setup
    plot_kinds = _resolve_plot_kinds(args.plot)
    want_map = bool(args.map)
    want_plot_dir = bool(plot_kinds) or want_map
    plot_dir = Path(args.plot_dir) if want_plot_dir else None
    if plot_dir is not None:
        plot_dir.mkdir(parents=True, exist_ok=True)

    # Validate flag combinations early so users don't wait for model loading
    if reach_ids and args.node_level:
        print(
            "Error: --reach-ids is only valid for reach-level analysis. "
            "Use --node-ids with --node-level.",
            file=sys.stderr,
        )
        return 2
    if node_ids and not args.node_level:
        print(
            "Error: --node-ids requires --node-level.",
            file=sys.stderr,
        )
        return 2

    # Load both models. Use a structured error path so missing budget
    # outputs surface with a helpful remediation hint.
    print(f"Loading baseline model from {baseline_dir} ...", file=sys.stderr)
    baseline_model = _load_model_from_dir(baseline_dir, "baseline")
    print(f"Loading scenario model from {scenario_dir} ...", file=sys.stderr)
    scenario_model = _load_model_from_dir(scenario_dir, "scenario")

    # Compute the report (reach-level or node-level)
    try:
        if args.node_level:
            print("Computing per-stream-node depletion ...", file=sys.stderr)
            node_report = compute_stream_node_depletion(
                baseline_model,
                scenario_model,
                node_ids=node_ids,
                sa_column=sa_column,
            )
            return _emit_node_outputs(
                node_report,
                baseline_model,
                args=args,
                plot_dir=plot_dir,
            )
        else:
            print("Computing per-reach depletion ...", file=sys.stderr)
            reach_report = compute_stream_depletion_from_models(
                baseline_model,
                scenario_model,
                reach_ids=reach_ids,
                sa_column=sa_column,
            )
            return _emit_reach_outputs(
                reach_report,
                baseline_model,
                args=args,
                plot_kinds=plot_kinds,
                want_map=want_map,
                plot_dir=plot_dir,
            )
    except BudgetOutputMissingError as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1
    except KeyError as e:
        # Likely the sa_column name didn't match an actual budget column;
        # the message already includes the available column list.
        print(f"\nError: {e}", file=sys.stderr)
        return 1


def _emit_reach_outputs(
    report: StreamDepletionReport,
    baseline_model: IWFMModel,
    *,
    args: argparse.Namespace,
    plot_kinds: list[str],
    want_map: bool,
    plot_dir: Path | None,
) -> int:
    """Write tabular / plot / map outputs for a reach-level report."""
    print(
        f"  Computed depletion for {report.n_reaches} reaches × {report.n_timesteps} timesteps.",
        file=sys.stderr,
    )

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        report.write(out)  # type: ignore[attr-defined]
        print(f"Wrote tabular report: {out}")

    if plot_kinds:
        from pyiwfm.visualization.plot_depletion import (
            plot_cumulative_depletion,
            plot_depletion_summary_bar,
            plot_depletion_timeseries,
        )

        assert plot_dir is not None
        for kind in plot_kinds:
            if kind == "cumulative":
                ax = plot_cumulative_depletion(report)
                filename = "cumulative_depletion.png"
            elif kind == "timeseries":
                ax = plot_depletion_timeseries(report)
                filename = "depletion_timeseries.png"
            else:  # "summary"
                ax = plot_depletion_summary_bar(report)
                filename = "depletion_summary.png"
            _save_axes_figure(ax, plot_dir / filename)
            print(f"Wrote plot: {plot_dir / filename}")

    if want_map:
        from pyiwfm.visualization.map_depletion import (
            export_depletion_geojson,
            plot_depletion_map,
        )

        assert plot_dir is not None
        png = plot_dir / f"depletion_map_{args.metric}.png"
        geojson = plot_dir / f"depletion_map_{args.metric}.geojson"
        ax = plot_depletion_map(report, baseline_model, metric=args.metric)
        _save_axes_figure(ax, png)
        export_depletion_geojson(report, baseline_model, geojson, metric=args.metric, crs=args.crs)
        print(f"Wrote map: {png}")
        print(f"Wrote GeoJSON: {geojson}")

    return 0


def _save_axes_figure(ax: object, output_path: Path) -> None:
    """Save the figure containing ``ax`` to ``output_path``.

    Indirection so callers can pass a ``matplotlib.axes.Axes`` without
    mypy complaining that ``ax.figure`` could be a ``SubFigure`` (the
    Figure | SubFigure union type doesn't expose ``savefig``). Going
    through ``plt.gcf()`` ensures we always have a real Figure.
    """
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    fig = getattr(ax, "figure", None)
    if not isinstance(fig, Figure):
        fig = plt.gcf()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")


def _emit_node_outputs(
    report: StreamNodeDepletionReport,
    baseline_model: IWFMModel,
    *,
    args: argparse.Namespace,
    plot_dir: Path | None,
) -> int:
    """Write tabular / map outputs for a node-level report.

    Node-level output: tabular JSON via ``report.to_dict()`` (CSV/Excel
    writers haven't been added for node-level yet — they'd follow the
    same pattern as the reach writers; left for a focused follow-up
    when we know what columns stakeholders actually want), plus the
    spatial node map and GeoJSON.
    """
    import json

    print(
        f"  Computed depletion for {report.n_stream_nodes} stream nodes "
        f"× {report.n_timesteps} timesteps.",
        file=sys.stderr,
    )

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix.lower() != ".json":
            print(
                f"Warning: --node-level only supports JSON tabular output today; "
                f"writing JSON to {out} (extension {out.suffix!r} ignored).",
                file=sys.stderr,
            )
        with out.open("w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"Wrote tabular report: {out}")

    if args.map:
        from pyiwfm.visualization.map_depletion import (
            export_stream_node_depletion_geojson,
            plot_stream_node_depletion_map,
        )

        assert plot_dir is not None
        png = plot_dir / f"node_depletion_map_{args.metric}.png"
        geojson = plot_dir / f"node_depletion_map_{args.metric}.geojson"
        ax = plot_stream_node_depletion_map(report, baseline_model, metric=args.metric)
        _save_axes_figure(ax, png)
        export_stream_node_depletion_geojson(report, baseline_model, geojson, crs=args.crs)
        print(f"Wrote node map: {png}")
        print(f"Wrote GeoJSON: {geojson}")

    return 0
