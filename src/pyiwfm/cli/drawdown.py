"""
CLI subcommand for drawdown analysis (Phase 2 drawdown).

Wires the drawdown computation, tabular writers, time-series plots,
and spatial maps into a single command for stakeholder reports from a
loaded IWFM model run.

Usage::

    pyiwfm drawdown <model_dir> --mode timeseries --locations 1,1;42,2 \\
        --output drawdown.xlsx --plot all
    pyiwfm drawdown <model_dir> --mode snapshot --timestep 100 --layer 1
    pyiwfm drawdown <model_dir> --mode max --layer 1

Drawdown is computed against ``--reference-timestep`` (default 0) on
the GW heads HDF declared in the model's GW main file (the
``gw_head_all_file`` entry in :attr:`IWFMModel.metadata`). Override
the HDF path with ``--heads-hdf`` if needed.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyiwfm.core.model import IWFMModel
    from pyiwfm.io.drawdown import DrawdownComputer, DrawdownSnapshot
    from pyiwfm.io.head_loader import LazyHeadDataLoader

logger = logging.getLogger(__name__)

# Plot kinds for ``--mode timeseries``. ``all`` expands to every entry.
_TS_PLOT_KINDS = ("timeseries", "summary")


def add_drawdown_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the ``pyiwfm drawdown`` subcommand."""
    p = subparsers.add_parser(
        "drawdown",
        help="Drawdown analysis at observation locations or as spatial maps",
        description=(
            "Compute drawdown (head change vs a reference timestep) "
            "from a loaded IWFM model. Three modes: timeseries (per-"
            "location time series + plots + tabular export), snapshot "
            "(map + GeoJSON at a single timestep), max (per-node "
            "max-across-time map + GeoJSON). The heads HDF is "
            "discovered from the model's stream main file by default."
        ),
    )
    p.add_argument(
        "model_dir",
        type=str,
        help="Path to the IWFM model directory (auto-discovers Simulation_MAIN.IN).",
    )
    p.add_argument(
        "--mode",
        choices=("timeseries", "snapshot", "max"),
        required=True,
        help=(
            "Analysis mode. timeseries: per-location time series. "
            "snapshot: per-node map at one timestep. max: per-node max "
            "drawdown across all timesteps."
        ),
    )
    p.add_argument(
        "--locations",
        type=str,
        default=None,
        help=(
            "(timeseries mode) Semicolon-separated list of node,layer "
            "pairs, e.g. '1,1;42,2'. Both 1-based."
        ),
    )
    p.add_argument(
        "--timestep",
        type=int,
        default=None,
        help="(snapshot mode) 0-based timestep index for the snapshot.",
    )
    p.add_argument(
        "--layer",
        type=int,
        default=1,
        help="1-based aquifer layer (snapshot/max modes; default 1).",
    )
    p.add_argument(
        "--reference-timestep",
        type=int,
        default=0,
        help="0-based reference timestep (drawdown = head(ref) - head(t)).",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Tabular report path. timeseries: CSV/JSON/Excel by extension. "
            "snapshot/max: JSON only (snapshots are dense node arrays)."
        ),
    )
    p.add_argument(
        "--plot",
        action="append",
        choices=[*_TS_PLOT_KINDS, "all"],
        default=None,
        help=(
            "(timeseries mode) Plot kind to render: timeseries, summary, "
            "or all. Repeat to render multiple. Files written to --plot-dir."
        ),
    )
    p.add_argument(
        "--plot-dir",
        type=str,
        default="drawdown_plots",
        help="Directory for plot/map output (default: ./drawdown_plots).",
    )
    p.add_argument(
        "--no-map",
        action="store_true",
        help=(
            "(snapshot/max modes) Suppress the map PNG + GeoJSON. By "
            "default these modes always render a map since that's their "
            "primary purpose."
        ),
    )
    p.add_argument(
        "--crs",
        type=str,
        default=None,
        help="CRS string to record in GeoJSON output (e.g. 'EPSG:26910').",
    )
    p.add_argument(
        "--heads-hdf",
        type=str,
        default=None,
        help=("Override the heads HDF path. By default, uses model.metadata['gw_head_all_file']."),
    )
    p.set_defaults(func=run_drawdown)


def _parse_locations(text: str | None) -> list[tuple[int, int]] | None:
    """Parse '1,1;42,2' into ``[(1, 1), (42, 2)]``. ``None`` → ``None``."""
    if not text:
        return None
    out: list[tuple[int, int]] = []
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [p.strip() for p in chunk.split(",")]
        if len(parts) != 2:
            raise SystemExit(
                f"Could not parse location {chunk!r}: expected 'node,layer' "
                f"format, e.g. --locations 1,1;42,2"
            )
        try:
            out.append((int(parts[0]), int(parts[1])))
        except ValueError as e:
            raise SystemExit(f"Could not parse location {chunk!r}: {e}") from e
    return out


def _resolve_ts_plot_kinds(plot_args: list[str] | None) -> list[str]:
    if not plot_args:
        return []
    if "all" in plot_args:
        return list(_TS_PLOT_KINDS)
    seen: set[str] = set()
    out: list[str] = []
    for k in plot_args:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def _save_axes_figure(ax: object, output_path: Path) -> None:
    """Save the figure containing ``ax`` to ``output_path``.

    Same indirection used by :mod:`pyiwfm.cli.depletion` to satisfy
    mypy and ensure we save a real ``Figure`` (not a ``SubFigure``).
    """
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    fig = getattr(ax, "figure", None)
    if not isinstance(fig, Figure):
        fig = plt.gcf()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")


def _load_model_from_dir(model_dir: Path) -> IWFMModel:
    """Discover and load an IWFM model from a directory.

    Loads with the preprocessor when available so the mesh is populated
    (the drawdown maps need ``model.mesh`` for node coordinates).
    """
    from pyiwfm.cli._model_finder import find_preprocessor_file, find_simulation_file
    from pyiwfm.core.model import IWFMModel

    sim_file = find_simulation_file(model_dir)
    if sim_file is None:
        raise SystemExit(
            f"Could not locate the simulation main file in: {model_dir}. "
            f"Expected a path like <dir>/Simulation/Simulation_MAIN.IN."
        )

    pp_file = find_preprocessor_file(model_dir)
    if pp_file is not None:
        return IWFMModel.from_simulation_with_preprocessor(
            simulation_file=sim_file, preprocessor_file=pp_file
        )
    return IWFMModel.from_simulation(sim_file)


def _open_heads_loader(model: IWFMModel, heads_hdf_override: str | None) -> LazyHeadDataLoader:
    """Return a :class:`LazyHeadDataLoader` for the model's heads HDF.

    Resolution order:
      1. ``heads_hdf_override`` (CLI ``--heads-hdf`` argument)
      2. ``model.metadata['gw_head_all_file']`` (declared in GW main file)

    Raises :class:`SystemExit` with a remediation hint if neither is set
    or the file doesn't exist.
    """
    from pyiwfm.io.head_loader import LazyHeadDataLoader

    if heads_hdf_override:
        path = Path(heads_hdf_override)
        if not path.exists():
            raise SystemExit(f"--heads-hdf path not found: {path}")
    else:
        raw = model.metadata.get("gw_head_all_file")
        if not raw:
            raise SystemExit(
                "Model didn't declare a heads HDF output. Either set the "
                "HEADALL output in the GW main file (and re-run the "
                "simulation), or pass --heads-hdf <path> on the command line."
            )
        path = Path(raw)
        if not path.exists():
            raise SystemExit(
                f"Heads HDF declared by the model not found on disk: {path}. "
                f"Run the simulation, or pass --heads-hdf <path> to override."
            )

    n_layers = model.groundwater.n_layers if model.groundwater else None
    return LazyHeadDataLoader(path, n_layers=n_layers)


def run_drawdown(args: argparse.Namespace) -> int:
    """Execute the drawdown analysis end-to-end."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    model_dir = Path(args.model_dir)
    if not model_dir.is_dir():
        print(f"Error: model directory not found: {model_dir}", file=sys.stderr)
        return 1

    # Validate mode-specific args early
    if args.mode == "timeseries":
        if not args.locations:
            print(
                "Error: --mode timeseries requires --locations 'node,layer[;node,layer...]'",
                file=sys.stderr,
            )
            return 2
        if args.timestep is not None:
            print(
                "Warning: --timestep is ignored in timeseries mode "
                "(use --reference-timestep to set the comparison baseline).",
                file=sys.stderr,
            )
    elif args.mode == "snapshot":
        if args.timestep is None:
            print("Error: --mode snapshot requires --timestep N", file=sys.stderr)
            return 2
        if args.locations:
            print("Warning: --locations is ignored in snapshot mode.", file=sys.stderr)
        if args.plot:
            print("Warning: --plot is ignored in snapshot mode.", file=sys.stderr)
    elif args.mode == "max":
        if args.timestep is not None:
            print("Warning: --timestep is ignored in max mode.", file=sys.stderr)
        if args.locations:
            print("Warning: --locations is ignored in max mode.", file=sys.stderr)
        if args.plot:
            print("Warning: --plot is ignored in max mode.", file=sys.stderr)

    # Setup output directory if we'll write any files there
    plot_dir = Path(args.plot_dir)
    needs_plot_dir = (args.mode == "timeseries" and args.plot) or (
        args.mode in ("snapshot", "max") and not args.no_map
    )
    if needs_plot_dir:
        plot_dir.mkdir(parents=True, exist_ok=True)

    # Load the model + heads loader
    print(f"Loading model from {model_dir} ...", file=sys.stderr)
    model = _load_model_from_dir(model_dir)
    loader = _open_heads_loader(model, args.heads_hdf)
    print(
        f"  Heads HDF: {loader._file_path.name} "  # type: ignore[attr-defined]
        f"({loader.n_frames} timesteps × {loader.n_nodes} nodes × "
        f"{loader.n_layers} layers).",
        file=sys.stderr,
    )

    from pyiwfm.io.drawdown import DrawdownComputer

    computer = DrawdownComputer(loader)

    if args.mode == "timeseries":
        return _run_timeseries(args, model, computer, plot_dir)
    if args.mode == "snapshot":
        return _run_snapshot(args, model, computer, plot_dir)
    return _run_max(args, model, computer, plot_dir)


def _run_timeseries(
    args: argparse.Namespace,
    model: IWFMModel,  # noqa: ARG001  (kept for symmetry with other run_* helpers)
    computer: DrawdownComputer,
    plot_dir: Path,
) -> int:
    locations = _parse_locations(args.locations)
    assert locations is not None  # validated upstream

    print(
        f"Building time-series report for {len(locations)} locations "
        f"(reference timestep {args.reference_timestep}) ...",
        file=sys.stderr,
    )
    try:
        report = computer.build_timeseries_report(
            locations=locations,
            reference_timestep=args.reference_timestep,
        )
    except IndexError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            report.write(out)  # type: ignore[attr-defined]
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        print(f"Wrote tabular report: {out}")

    plot_kinds = _resolve_ts_plot_kinds(args.plot)
    if plot_kinds:
        from pyiwfm.visualization.plot_drawdown import (
            plot_drawdown_summary_bar,
            plot_drawdown_timeseries,
        )

        for kind in plot_kinds:
            if kind == "timeseries":
                ax = plot_drawdown_timeseries(report)
                filename = "drawdown_timeseries.png"
            else:  # "summary"
                ax = plot_drawdown_summary_bar(report)
                filename = "drawdown_summary.png"
            _save_axes_figure(ax, plot_dir / filename)
            print(f"Wrote plot: {plot_dir / filename}")

    return 0


def _run_snapshot(
    args: argparse.Namespace,
    model: IWFMModel,
    computer: DrawdownComputer,
    plot_dir: Path,
) -> int:
    print(
        f"Building snapshot at timestep {args.timestep}, layer {args.layer} "
        f"(reference timestep {args.reference_timestep}) ...",
        file=sys.stderr,
    )
    try:
        snapshot = computer.build_snapshot(
            timestep=args.timestep,
            layer=args.layer,
            reference_timestep=args.reference_timestep,
        )
    except IndexError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return _emit_snapshot_outputs(snapshot, args, model, plot_dir, prefix="snapshot")


def _run_max(
    args: argparse.Namespace,
    model: IWFMModel,
    computer: DrawdownComputer,
    plot_dir: Path,
) -> int:
    print(
        f"Building max-across-time snapshot, layer {args.layer} "
        f"(reference timestep {args.reference_timestep}) ...",
        file=sys.stderr,
    )
    try:
        snapshot = computer.build_max_snapshot(
            layer=args.layer,
            reference_timestep=args.reference_timestep,
        )
    except IndexError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return _emit_snapshot_outputs(snapshot, args, model, plot_dir, prefix="max")


def _emit_snapshot_outputs(
    snapshot: DrawdownSnapshot,
    args: argparse.Namespace,
    model: IWFMModel,
    plot_dir: Path,
    *,
    prefix: str,
) -> int:
    """Write JSON + map PNG + GeoJSON for a snapshot or max-snapshot."""
    import json

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix.lower() != ".json":
            print(
                f"Warning: snapshot/max output is JSON only; writing JSON to "
                f"{out} (extension {out.suffix!r} ignored).",
                file=sys.stderr,
            )
        with out.open("w", encoding="utf-8") as f:
            json.dump(snapshot.to_dict(), f, indent=2)
        print(f"Wrote snapshot JSON: {out}")

    if not args.no_map:
        from pyiwfm.visualization.map_drawdown import (
            export_drawdown_geojson,
            plot_drawdown_map,
        )

        png = plot_dir / f"drawdown_{prefix}_layer{snapshot.layer}.png"
        geojson = plot_dir / f"drawdown_{prefix}_layer{snapshot.layer}.geojson"
        try:
            ax = plot_drawdown_map(snapshot, model)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        _save_axes_figure(ax, png)
        export_drawdown_geojson(snapshot, model, geojson, crs=args.crs)
        print(f"Wrote map: {png}")
        print(f"Wrote GeoJSON: {geojson}")

    return 0
