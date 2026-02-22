#!/usr/bin/env python
"""Generate the small_model fixture files for unit tests.

Uses ``build_tutorial_model()`` to create synthetic IWFM model data,
wraps it in an ``IWFMModel``, and writes all component files to
``tests/fixtures/small_model/`` via ``save_complete_model()``.

Run once (or re-run when model format changes) and commit the output::

    python tests/fixtures/generate_small_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the package is importable when running from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def main() -> None:
    from pyiwfm.core.model import IWFMModel
    from pyiwfm.io.preprocessor import save_complete_model
    from pyiwfm.sample_models import build_tutorial_model

    output_dir = Path(__file__).resolve().parent / "small_model"

    # Clean previous output (on Windows/OneDrive, rmtree can fail with
    # permission errors, so we just ensure the directory exists and let
    # the writer overwrite files in place).
    if output_dir.exists():
        import shutil

        shutil.rmtree(output_dir, ignore_errors=True)

    # Build the tutorial model data
    tut = build_tutorial_model()

    # Wrap in IWFMModel
    model = IWFMModel(
        name="SmallModel",
        mesh=tut.grid,
        stratigraphy=tut.stratigraphy,
        groundwater=tut.groundwater,
        streams=tut.stream,
        lakes=tut.lakes,
        rootzone=tut.rootzone,
        metadata={
            "description": "Small fixture model for unit tests",
            "version": "1.0",
        },
    )

    # Write all component files
    files = save_complete_model(model, output_dir)

    print(f"Generated {len(files)} files in {output_dir}:")
    for key, path in sorted(files.items()):
        rel = path.relative_to(output_dir)
        print(f"  {key:30s} -> {rel}")


if __name__ == "__main__":
    main()
