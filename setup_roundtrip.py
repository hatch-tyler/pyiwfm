"""
Setup roundtrip test: read C2VSimFG via pyiwfm, write two versions,
and copy IWFM executables for each.

Usage:
    python setup_roundtrip.py [--output-dir C:\temp\iwfm_roundtrip]

Output structure:
    <output_dir>/
        text/                       # All-text time series version
            Preprocessor/
                Preprocessor.in
                ...
            Simulation/
                Simulation_MAIN.IN
                ...
            Results/
            PreProcessor_x64.exe
            Simulation_x64.exe
        dss/                        # DSS precipitation version
            Preprocessor/
                Preprocessor.in
                ...
            Simulation/
                Simulation_MAIN.IN
                climate_data.dss
                ...
            Results/
            PreProcessor_x64.exe
            Simulation_x64.exe
"""

import argparse
import shutil
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Setup pyiwfm roundtrip test for C2VSimFG"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"C:\temp\iwfm_roundtrip"),
        help="Base output directory (default: C:\\temp\\iwfm_roundtrip)",
    )
    parser.add_argument(
        "--c2vsimfg-dir",
        type=Path,
        default=Path(r"C:\Users\hatch\OneDrive\Desktop\c2vsimfg"),
        help="Path to C2VSimFG model directory",
    )
    parser.add_argument(
        "--bin-dir",
        type=Path,
        default=Path(r"C:\Users\hatch\OneDrive\Desktop\iwfm-2025.0.1747\src\Bin"),
        help="Path to IWFM executables directory",
    )
    parser.add_argument(
        "--skip-text",
        action="store_true",
        help="Skip writing text version",
    )
    parser.add_argument(
        "--skip-dss",
        action="store_true",
        help="Skip writing DSS version",
    )
    args = parser.parse_args()

    # Validate paths
    sim_file = args.c2vsimfg_dir / "Simulation" / "C2VSimFG.in"
    pp_file = args.c2vsimfg_dir / "Preprocessor" / "C2VSimFG_Preprocessor.in"
    sim_exe = args.bin_dir / "Simulation_x64.exe"
    pp_exe = args.bin_dir / "PreProcessor_x64.exe"

    for path, desc in [
        (sim_file, "Simulation main file"),
        (pp_file, "Preprocessor main file"),
        (sim_exe, "Simulation executable"),
        (pp_exe, "PreProcessor executable"),
    ]:
        if not path.exists():
            print(f"ERROR: {desc} not found: {path}")
            sys.exit(1)

    text_dir = args.output_dir / "text"
    dss_dir = args.output_dir / "dss"

    # ----------------------------------------------------------------
    # Step 1: Load C2VSimFG model
    # ----------------------------------------------------------------
    print("=" * 60)
    print("Step 1: Loading C2VSimFG model via pyiwfm")
    print("=" * 60)
    print(f"  Simulation file: {sim_file}")
    print(f"  Preprocessor file: {pp_file}")

    from pyiwfm import IWFMModel

    t0 = time.perf_counter()
    model = IWFMModel.from_simulation_with_preprocessor(
        sim_file, pp_file, load_timeseries=True
    )
    elapsed = time.perf_counter() - t0
    print(f"  Loaded in {elapsed:.1f}s")
    print(f"  Nodes: {model.n_nodes}")
    print(f"  Elements: {model.n_elements}")
    print(f"  Layers: {model.n_layers}")
    print(f"  Has GW: {model.has_groundwater}")
    print(f"  Has Streams: {model.has_streams}")
    print(f"  Has RootZone: {model.has_rootzone}")
    print()

    # ----------------------------------------------------------------
    # Step 2: Write text version
    # ----------------------------------------------------------------
    if not args.skip_text:
        print("=" * 60)
        print("Step 2: Writing TEXT version")
        print("=" * 60)
        print(f"  Output: {text_dir}")

        if text_dir.exists():
            print("  Removing existing text directory...")
            shutil.rmtree(text_dir)

        t0 = time.perf_counter()
        files_text = model.to_simulation(text_dir, ts_format="text")
        elapsed = time.perf_counter() - t0
        print(f"  Wrote {len(files_text)} files in {elapsed:.1f}s")
        for key, path in sorted(files_text.items()):
            print(f"    {key}: {path.name}")
        print()

        # Create Results directory
        results_dir = text_dir / "Results"
        results_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Created: {results_dir}")

        # Copy executables to model root (next to Preprocessor/ and Simulation/)
        for exe in [pp_exe, sim_exe]:
            dst = text_dir / exe.name
            shutil.copy2(exe, dst)
            print(f"  Copied: {exe.name} -> {dst}")
        print()

    # ----------------------------------------------------------------
    # Step 3: Write DSS version
    # ----------------------------------------------------------------
    if not args.skip_dss:
        print("=" * 60)
        print("Step 3: Writing DSS version (precipitation via HEC-DSS)")
        print("=" * 60)
        print(f"  Output: {dss_dir}")

        if dss_dir.exists():
            print("  Removing existing dss directory...")
            shutil.rmtree(dss_dir)

        t0 = time.perf_counter()
        files_dss = model.to_simulation(dss_dir, ts_format="dss")
        elapsed = time.perf_counter() - t0
        print(f"  Wrote {len(files_dss)} files in {elapsed:.1f}s")
        for key, path in sorted(files_dss.items()):
            print(f"    {key}: {path.name}")
        print()

        # Create Results directory
        results_dir = dss_dir / "Results"
        results_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Created: {results_dir}")

        # Copy executables to model root (next to Preprocessor/ and Simulation/)
        for exe in [pp_exe, sim_exe]:
            dst = dss_dir / exe.name
            shutil.copy2(exe, dst)
            print(f"  Copied: {exe.name} -> {dst}")
        print()

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print("=" * 60)
    print("Setup complete!")
    print("=" * 60)
    if not args.skip_text:
        print(f"  Text model:  {text_dir}")
        print(f"    Preprocessor: {text_dir / 'Preprocessor' / 'Preprocessor.in'}")
        print(f"    Simulation:   {text_dir / 'Simulation' / 'Simulation_MAIN.IN'}")
    if not args.skip_dss:
        print(f"  DSS model:   {dss_dir}")
        print(f"    Preprocessor: {dss_dir / 'Preprocessor' / 'Preprocessor.in'}")
        print(f"    Simulation:   {dss_dir / 'Simulation' / 'Simulation_MAIN.IN'}")
    print()
    print("Next: run the models with run_roundtrip.ps1")
    print("  .\\run_roundtrip.ps1 -Model text -RunPreprocessor -RunSimulation")
    print("  .\\run_roundtrip.ps1 -Model dss -RunPreprocessor -RunSimulation")
    print("  .\\run_roundtrip.ps1 -Model both -RunAll")


if __name__ == "__main__":
    main()
