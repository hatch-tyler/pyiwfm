#!/usr/bin/env python
"""Generate pre-rendered PNG figures for the Reading Models tutorial.

Usage::

    python docs/scripts/generate_tutorial_figures.py /path/to/C2VSimCG

Or set the ``C2VSIMCG_DIR`` environment variable::

    C2VSIMCG_DIR=/path/to/C2VSimCG python docs/scripts/generate_tutorial_figures.py

The script expects to find a ``Simulation.in`` file inside the given
directory.  All figures are saved to
``docs/_static/tutorials/reading_models/``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from pyiwfm.core.model import IWFMModel

matplotlib.use("Agg")

# Resolve output directory relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
OUTPUT_DIR = REPO_ROOT / "docs" / "_static" / "tutorials" / "reading_models"


def _find_simulation_file(model_dir: Path) -> Path:
    """Locate the simulation main file inside *model_dir*."""
    candidates = [
        model_dir / "Simulation.in",
        model_dir / "Simulation" / "Simulation.in",
        model_dir / "C2VSimCG.in",
        model_dir / "Simulation" / "C2VSimCG.in",
    ]
    for c in candidates:
        if c.exists():
            return c
    # Fallback: glob for simulation or C2VSim main files
    for p in model_dir.rglob("*imulation*.in"):
        return p
    for p in model_dir.rglob("C2VSim*.in"):
        return p
    raise FileNotFoundError(f"Could not find a simulation .in file inside {model_dir}")


def _find_preprocessor_file(model_dir: Path) -> Path | None:
    """Locate the preprocessor main .in file near *model_dir*."""
    candidates = [
        model_dir / "Preprocessor" / "C2VSimCG_Preprocessor.in",
        model_dir / "Preprocessor" / "Preprocessor.in",
    ]
    for c in candidates:
        if c.exists():
            return c
    # Check sibling Preprocessor/ directory (if model_dir is Simulation/)
    sibling = model_dir.parent / "Preprocessor"
    if sibling.is_dir():
        for p in sibling.glob("*reprocessor*.in"):
            return p
    # Check within model_dir itself
    for p in model_dir.rglob("*reprocessor*.in"):
        # Skip binary files
        if p.suffix == ".in":
            return p
    return None


def generate_mesh(model: IWFMModel, output_dir: Path) -> None:
    """mesh.png -- full C2VSimCG mesh with element edges."""
    from pyiwfm.visualization.plotting import plot_mesh

    assert model.mesh is not None
    fig, ax = plot_mesh(
        model.mesh,
        show_edges=True,
        edge_color="gray",
        fill_color="lightblue",
        alpha=0.3,
        figsize=(10, 10),
    )
    ax.set_title(f"C2VSimCG Mesh ({model.n_nodes} nodes, {model.n_elements} elements)")
    ax.set_xlabel("Easting (ft)")
    ax.set_ylabel("Northing (ft)")
    fig.savefig(output_dir / "mesh.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved mesh.png")


def generate_subregions(model: IWFMModel, output_dir: Path) -> None:
    """subregions.png -- elements colored by subregion."""
    from pyiwfm.visualization.plotting import plot_elements

    assert model.mesh is not None
    fig, ax = plot_elements(
        model.mesh,
        color_by="subregion",
        cmap="tab20",
        alpha=0.7,
        figsize=(10, 10),
    )
    ax.set_title(f"C2VSimCG Subregions ({model.mesh.n_subregions} subregions)")
    ax.set_xlabel("Easting (ft)")
    ax.set_ylabel("Northing (ft)")
    fig.savefig(output_dir / "subregions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved subregions.png")


def generate_streams(model: IWFMModel, output_dir: Path) -> None:
    """streams.png -- stream network overlaid on mesh."""
    from pyiwfm.visualization.plotting import plot_mesh, plot_streams

    if model.streams is None:
        print("  Skipping streams.png -- no stream component loaded")
        return

    assert model.mesh is not None

    fig, ax = plot_mesh(
        model.mesh,
        show_edges=True,
        edge_color="lightgray",
        fill_color="white",
        alpha=0.15,
        figsize=(10, 10),
    )
    plot_streams(model.streams, ax=ax, line_color="blue", line_width=1.5)
    ax.set_title(
        f"C2VSimCG Stream Network ({model.streams.n_reaches} reaches, "
        f"{model.streams.n_nodes} stream nodes)"
    )
    ax.set_xlabel("Easting (ft)")
    ax.set_ylabel("Northing (ft)")
    fig.savefig(output_dir / "streams.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved streams.png")


def generate_lakes(model: IWFMModel, output_dir: Path) -> None:
    """lakes.png -- lake boundaries on mesh."""
    from pyiwfm.visualization.plotting import plot_lakes, plot_mesh

    if model.lakes is None:
        print("  Skipping lakes.png -- no lake component loaded")
        return

    assert model.mesh is not None
    fig, ax = plot_mesh(
        model.mesh,
        show_edges=True,
        edge_color="lightgray",
        fill_color="white",
        alpha=0.15,
        figsize=(10, 10),
    )
    plot_lakes(
        model.lakes,
        model.mesh,
        ax=ax,
        fill_color="cyan",
        edge_color="blue",
        alpha=0.5,
    )
    ax.set_title(f"C2VSimCG Lakes ({model.lakes.n_lakes} lakes)")
    ax.set_xlabel("Easting (ft)")
    ax.set_ylabel("Northing (ft)")
    fig.savefig(output_dir / "lakes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved lakes.png")


def generate_ground_surface(model: IWFMModel, output_dir: Path) -> None:
    """ground_surface.png -- ground surface elevation scalar field."""
    from pyiwfm.visualization.plotting import plot_scalar_field

    if model.stratigraphy is None:
        print("  Skipping ground_surface.png -- no stratigraphy loaded")
        return

    assert model.mesh is not None
    gs_elev = model.stratigraphy.gs_elev
    fig, ax = plot_scalar_field(
        model.mesh,
        gs_elev,
        field_type="node",
        cmap="terrain",
        show_mesh=False,
        figsize=(10, 10),
    )
    ax.set_title("C2VSimCG Ground Surface Elevation (ft)")
    ax.set_xlabel("Easting (ft)")
    ax.set_ylabel("Northing (ft)")
    fig.savefig(output_dir / "ground_surface.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved ground_surface.png")


def generate_layer_thickness(model: IWFMModel, output_dir: Path) -> None:
    """layer_thickness.png -- Layer 1 thickness."""
    from pyiwfm.visualization.plotting import plot_scalar_field

    if model.stratigraphy is None:
        print("  Skipping layer_thickness.png -- no stratigraphy loaded")
        return

    assert model.mesh is not None
    thickness = model.stratigraphy.get_layer_thickness(0)
    fig, ax = plot_scalar_field(
        model.mesh,
        thickness,
        field_type="node",
        cmap="YlOrRd",
        show_mesh=False,
        figsize=(10, 10),
    )
    ax.set_title("C2VSimCG Layer 1 Thickness (ft)")
    ax.set_xlabel("Easting (ft)")
    ax.set_ylabel("Northing (ft)")
    fig.savefig(output_dir / "layer_thickness.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved layer_thickness.png")


def generate_cross_section(model: IWFMModel, output_dir: Path) -> None:
    """cross_section.png -- stratigraphy cross-section through the model."""
    from pyiwfm.core.cross_section import CrossSectionExtractor
    from pyiwfm.visualization.plotting import plot_cross_section

    if model.stratigraphy is None or model.mesh is None:
        print("  Skipping cross_section.png -- mesh or stratigraphy missing")
        return

    # Determine a line that cuts roughly east-west through the model center
    all_x = [n.x for n in model.mesh.nodes.values()]
    all_y = [n.y for n in model.mesh.nodes.values()]
    x_min, x_max = min(all_x), max(all_x)
    y_mid = (min(all_y) + max(all_y)) / 2.0

    extractor = CrossSectionExtractor(model.mesh, model.stratigraphy)
    xs = extractor.extract(
        start=(x_min, y_mid),
        end=(x_max, y_mid),
        n_samples=200,
    )

    fig, ax = plot_cross_section(
        xs,
        title="C2VSimCG East-West Cross-Section",
        figsize=(14, 6),
    )
    ax.set_xlabel("Distance (ft)")
    ax.set_ylabel("Elevation (ft)")
    fig.savefig(output_dir / "cross_section.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved cross_section.png")


def _find_budget_file(model_dir: Path) -> Path | None:
    """Locate a GW budget HDF5 file in the model results."""
    candidates = [
        model_dir / "Results" / "GW_Budget.hdf",
        model_dir / "Results" / "GW_Budget.hdf5",
    ]
    for c in candidates:
        if c.exists():
            return c
    # Glob for any GW budget file
    for p in model_dir.rglob("*GW*Budget*.hdf*"):
        return p
    return None


def generate_budget_bar(model_dir: Path, output_dir: Path) -> None:
    """budget_bar.png -- GW budget bar chart from real C2VSimCG data."""
    from pyiwfm.visualization.plotting import plot_budget_bar

    budget_file = _find_budget_file(model_dir)
    if budget_file is not None:
        from pyiwfm.io import BudgetReader

        reader = BudgetReader(budget_file)
        df = reader.get_dataframe(0, volume_factor=2.29568e-05)
        budget = df.drop(columns=["Time"], errors="ignore").mean().to_dict()
        title = "C2VSimCG Groundwater Budget"
        units = "AF/month"
    else:
        print("  WARNING: No GW budget HDF5 found -- using illustrative values")
        budget = {
            "Deep Percolation": 5_800_000.0,
            "Stream Seepage": 2_400_000.0,
            "Subsurface Inflow": 800_000.0,
            "Pumping": -7_500_000.0,
            "Outflow to Streams": -1_200_000.0,
            "Subsurface Outflow": -300_000.0,
        }
        title = "C2VSimCG Groundwater Budget (Illustrative)"
        units = "AF/year"

    fig, ax = plot_budget_bar(budget, title=title, units=units)
    fig.savefig(output_dir / "budget_bar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved budget_bar.png")


def generate_budget_stacked(model_dir: Path, output_dir: Path) -> None:
    """budget_stacked.png -- budget components over time from real C2VSimCG data."""
    from pyiwfm.visualization.plotting import plot_budget_stacked

    budget_file = _find_budget_file(model_dir)
    if budget_file is not None:
        from pyiwfm.io import BudgetReader

        reader = BudgetReader(budget_file)
        df = reader.get_dataframe(0, volume_factor=2.29568e-05)
        times = df["Time"].values
        components = {col: df[col].values for col in df.columns if col != "Time"}
        title = "C2VSimCG GW Budget Over Time"
        units = "AF/month"
    else:
        print("  WARNING: No GW budget HDF5 found -- using illustrative values")
        rng = np.random.default_rng(42)
        n_years = 10
        times = np.array([np.datetime64(f"{y}-01-01") for y in range(2005, 2015)])
        components = {
            "Deep Percolation": 5_800_000 + rng.normal(0, 400_000, n_years),
            "Stream Seepage": 2_400_000 + rng.normal(0, 200_000, n_years),
            "Pumping": -(
                7_500_000 + np.arange(n_years) * 50_000 + rng.normal(0, 300_000, n_years)
            ),
            "Outflow to Streams": -(1_200_000 + rng.normal(0, 100_000, n_years)),
        }
        title = "C2VSimCG Groundwater Budget Over Time (Illustrative)"
        units = "AF/year"

    fig, ax = plot_budget_stacked(times, components, title=title, units=units)
    fig.savefig(output_dir / "budget_stacked.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved budget_stacked.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate tutorial figures from a C2VSimCG model.")
    parser.add_argument(
        "model_dir",
        nargs="?",
        default=os.environ.get("C2VSIMCG_DIR"),
        help="Path to C2VSimCG model directory (or set C2VSIMCG_DIR env var)",
    )
    args = parser.parse_args()

    if args.model_dir is None:
        parser.error("Provide a model directory as an argument or set C2VSIMCG_DIR")

    model_dir = Path(args.model_dir).resolve()
    if not model_dir.is_dir():
        print(f"ERROR: {model_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sim_file = _find_simulation_file(model_dir)
    pp_file = _find_preprocessor_file(model_dir)
    print(f"Loading C2VSimCG from: {sim_file}")
    if pp_file:
        print(f"Preprocessor file: {pp_file}")

    if pp_file:
        from pyiwfm.io.model_loader import CompleteModelLoader

        loader = CompleteModelLoader(simulation_file=sim_file, preprocessor_file=pp_file)
        result = loader.load()
        if not result.success:
            print(f"ERROR: Failed to load model: {result.errors}", file=sys.stderr)
            sys.exit(1)
        model = result.model
    else:
        from pyiwfm.io import load_complete_model

        model = load_complete_model(sim_file)
    print(model.summary())
    print()

    print("Generating figures...")

    generate_mesh(model, OUTPUT_DIR)
    generate_subregions(model, OUTPUT_DIR)
    generate_streams(model, OUTPUT_DIR)
    generate_lakes(model, OUTPUT_DIR)
    generate_ground_surface(model, OUTPUT_DIR)
    generate_layer_thickness(model, OUTPUT_DIR)
    generate_cross_section(model, OUTPUT_DIR)

    # Budget figures use real C2VSimCG data when available
    generate_budget_bar(model_dir, OUTPUT_DIR)
    generate_budget_stacked(model_dir, OUTPUT_DIR)

    print(f"\nAll figures saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
