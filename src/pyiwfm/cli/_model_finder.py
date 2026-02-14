"""
Consolidated IWFM model file discovery.

Merges all duplicate file-finding logic from the former launcher scripts into
a single module.  Discovery strategy: check known candidate paths first (fast),
then fall back to glob patterns.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Known candidate paths (checked in order, first match wins)
# ---------------------------------------------------------------------------

_PREPROCESSOR_CANDIDATES = [
    # C2VSimFG specific
    "Preprocessor/C2VSimFG_Preprocessor.in",
    "Preprocessor/C2VSimFG_Preprocessor.IN",
    # Generic patterns
    "Preprocessor/PreProcessor_MAIN.IN",
    "Preprocessor/PreProcessor_MAIN.in",
    "Preprocessor/PreProcessor.in",
    "Preprocessor/Preprocessor_MAIN.IN",
    "Preprocessor/Preprocessor_MAIN.in",
    "Preprocessor/Preprocessor_MAIN.dat",
    "Preprocessor/Preprocessor.in",
    "PreProcessor/PreProcessor_MAIN.IN",
    "PreProcessor/PreProcessor_MAIN.in",
    "PreProcessor/PreProcessor.in",
    "PreProcessor_MAIN.IN",
    "PreProcessor_MAIN.in",
    "PreProcessor.in",
    "Preprocessor.in",
    "PP_Main.in",
    "PP_Main.IN",
]

_SIMULATION_CANDIDATES = [
    # C2VSimFG specific
    "Simulation/C2VSimFG.in",
    "Simulation/C2VSimFG.IN",
    # Generic patterns
    "Simulation/Simulation_MAIN.IN",
    "Simulation/Simulation_MAIN.in",
    "Simulation/Simulation.in",
    "Simulation/Simulation_MAIN.dat",
    "Simulation_MAIN.IN",
    "Simulation_MAIN.in",
    "Simulation.in",
    "Sim_Main.in",
    "Sim_Main.IN",
]

_BINARY_CANDIDATES = [
    "Simulation/C2VSimFG_Preprocessor.bin",
    "Simulation/PreProcessor.bin",
    "Preprocessor/C2VSimFG_Preprocessor.bin",
    "Preprocessor/PreProcessor.bin",
    "PreProcessor.bin",
]

# Glob patterns used as fallback when candidate paths don't match
_PREPROCESSOR_GLOBS = [
    "Preprocessor/*.in",
    "Preprocessor/*.IN",
    "PreProcessor/*.in",
    "PreProcessor/*.IN",
]

_SIMULATION_GLOBS = [
    "Simulation/*.in",
    "Simulation/*.IN",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def find_preprocessor_file(model_dir: Path) -> Path | None:
    """Find preprocessor main input file in *model_dir*."""
    # Fast path: check known candidates
    for candidate in _PREPROCESSOR_CANDIDATES:
        path = model_dir / candidate
        if path.exists():
            return path

    # Fallback: glob for .in / .IN files in preprocessor directories
    for pattern in _PREPROCESSOR_GLOBS:
        matches = list(model_dir.glob(pattern))
        if matches:
            main_files = [f for f in matches if "main" in f.name.lower()]
            return main_files[0] if main_files else matches[0]

    return None


def find_simulation_file(model_dir: Path) -> Path | None:
    """Find simulation main input file in *model_dir*."""
    for candidate in _SIMULATION_CANDIDATES:
        path = model_dir / candidate
        if path.exists():
            return path

    for pattern in _SIMULATION_GLOBS:
        matches = list(model_dir.glob(pattern))
        if matches:
            main_files = [f for f in matches if "main" in f.name.lower()]
            return main_files[0] if main_files else matches[0]

    return None


def find_binary_file(model_dir: Path) -> Path | None:
    """Find preprocessor binary file in *model_dir*."""
    for candidate in _BINARY_CANDIDATES:
        path = model_dir / candidate
        if path.exists():
            return path
    return None


def find_model_files(model_dir: Path) -> dict[str, Path | None]:
    """
    Return a dict with discovered model file paths.

    Keys: ``preprocessor_main``, ``simulation_main``, ``preprocessor_binary``.
    Values are ``Path`` objects or ``None`` if not found.
    """
    return {
        "preprocessor_main": find_preprocessor_file(model_dir),
        "simulation_main": find_simulation_file(model_dir),
        "preprocessor_binary": find_binary_file(model_dir),
    }


def extract_model_name(filepath: Path) -> str:
    """Derive a human-readable model name from *filepath*."""
    for part in filepath.parts:
        if any(tag in part.lower() for tag in ("c2vsim", "cvhm", "iwfm", "model")):
            return part

    if filepath.is_file():
        return filepath.parent.parent.name
    return filepath.name
