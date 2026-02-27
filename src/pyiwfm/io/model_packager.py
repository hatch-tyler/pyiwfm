"""Package IWFM model directories into distributable ZIP archives."""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

# Directories excluded by default (case-insensitive comparison)
_EXCLUDED_DIRS: frozenset[str] = frozenset(
    {
        "results",
        "__pycache__",
        ".git",
        ".svn",
        ".hg",
        ".pyiwfm_cache",
        "node_modules",
    }
)

# File extensions excluded by default (output files that can be regenerated)
_EXCLUDED_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".hdf",
        ".h5",
        ".pyc",
        ".pyo",
    }
)

# Extensions to include (input / config / data files)
_INCLUDED_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".dat",
        ".in",
        ".bin",
        ".dss",
        ".txt",
        ".csv",
        ".json",
        ".geojson",
        ".shp",
        ".shx",
        ".dbf",
        ".prj",
        ".cpg",
        ".sbn",
        ".sbx",
        ".gpkg",
        ".tif",
        ".tiff",
        ".asc",
        ".bat",
        ".sh",
        ".ps1",
        ".xml",
        ".yaml",
        ".yml",
        ".toml",
        ".cfg",
        ".ini",
        ".out",
    }
)

# Executable extensions (only included when include_executables is True)
_EXECUTABLE_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".exe",
        ".dll",
        ".so",
    }
)


@dataclass
class ModelPackageResult:
    """Result of a model packaging operation.

    Attributes
    ----------
    archive_path : Path
        Path to the created ZIP archive.
    files_included : list[Path]
        Absolute paths of all files that were added to the archive.
    total_size_bytes : int
        Total uncompressed size of included files in bytes.
    manifest : dict[str, str]
        Mapping of relative path (inside ZIP) to the file-type category.
    """

    archive_path: Path
    files_included: list[Path] = field(default_factory=list)
    total_size_bytes: int = 0
    manifest: dict[str, str] = field(default_factory=dict)


def _categorize_file(path: Path) -> str:
    """Return a human-readable category for a file based on its extension."""
    ext = path.suffix.lower()
    if ext in {".exe", ".dll", ".so"}:
        return "executable"
    if ext in {".dat", ".in"}:
        return "input"
    if ext in {".bin"}:
        return "binary"
    if ext in {".dss"}:
        return "dss"
    if ext in {".bat", ".sh", ".ps1"}:
        return "script"
    if ext in {".shp", ".shx", ".dbf", ".prj", ".cpg", ".gpkg", ".geojson"}:
        return "gis"
    if ext in {".hdf", ".h5"}:
        return "output"
    return "data"


def _is_excluded_dir(dir_name: str) -> bool:
    """Check if a directory name should be excluded (case-insensitive)."""
    return dir_name.lower() in _EXCLUDED_DIRS


def collect_model_files(
    model_dir: Path,
    *,
    include_executables: bool = False,
    include_results: bool = False,
) -> list[Path]:
    """Collect all model files from a directory tree.

    Walks *model_dir* recursively and returns files relevant to an IWFM model,
    excluding output/cache directories by default.

    Parameters
    ----------
    model_dir : Path
        Root directory of the IWFM model.
    include_executables : bool
        If ``True``, include ``.exe``, ``.dll``, and ``.so`` files.
    include_results : bool
        If ``True``, include the ``Results/`` directory and HDF5 output files.

    Returns
    -------
    list[Path]
        Sorted list of absolute paths to included files.
    """
    model_dir = model_dir.resolve()
    collected: list[Path] = []

    excluded_dirs = set(_EXCLUDED_DIRS)
    if include_results:
        excluded_dirs.discard("results")

    allowed_extensions = set(_INCLUDED_EXTENSIONS)
    if include_executables:
        allowed_extensions |= _EXECUTABLE_EXTENSIONS
    if include_results:
        allowed_extensions |= {".hdf", ".h5"}

    for item in model_dir.rglob("*"):
        if not item.is_file():
            continue

        # Skip files in excluded directories
        rel = item.relative_to(model_dir)
        if any(part.lower() in excluded_dirs for part in rel.parts[:-1]):
            continue

        ext = item.suffix.lower()

        # Skip executables unless requested
        if ext in _EXECUTABLE_EXTENSIONS and not include_executables:
            continue

        # Skip HDF5 output files unless results included
        if ext in _EXCLUDED_EXTENSIONS and not include_results:
            continue

        # Include files with known extensions, or extensionless files
        # (some IWFM models have config files without extensions)
        if ext in allowed_extensions or ext == "":
            collected.append(item)

    return sorted(collected)


def package_model(
    model_dir: Path,
    output_path: Path | None = None,
    *,
    include_executables: bool = False,
    include_results: bool = False,
    compression: int = zipfile.ZIP_DEFLATED,
    compresslevel: int = 6,
) -> ModelPackageResult:
    """Package an IWFM model directory into a ZIP archive.

    Creates a ZIP file preserving the directory structure (``Preprocessor/``,
    ``Simulation/``, etc.) with a ``manifest.json`` embedded in the archive.

    Parameters
    ----------
    model_dir : Path
        Root directory of the IWFM model.
    output_path : Path | None
        Path for the output ZIP file. If ``None``, defaults to
        ``<model_dir_name>.zip`` in the parent of *model_dir*.
    include_executables : bool
        If ``True``, include ``.exe``, ``.dll``, and ``.so`` files.
    include_results : bool
        If ``True``, include the ``Results/`` directory and HDF5 output files.
    compression : int
        ZIP compression method (default ``ZIP_DEFLATED``).
    compresslevel : int
        Compression level (0-9, default 6).

    Returns
    -------
    ModelPackageResult
        Result with archive path, file list, size, and manifest.

    Raises
    ------
    FileNotFoundError
        If *model_dir* does not exist or is not a directory.
    """
    model_dir = Path(model_dir).resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    if output_path is None:
        output_path = model_dir.parent / f"{model_dir.name}.zip"
    output_path = Path(output_path).resolve()

    # Collect files
    files = collect_model_files(
        model_dir,
        include_executables=include_executables,
        include_results=include_results,
    )

    result = ModelPackageResult(archive_path=output_path)
    manifest: dict[str, str] = {}

    # Create ZIP
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(
        output_path, "w", compression=compression, compresslevel=compresslevel
    ) as zf:
        for file_path in files:
            rel_path = file_path.relative_to(model_dir)
            arcname = str(rel_path)
            zf.write(file_path, arcname)

            result.files_included.append(file_path)
            result.total_size_bytes += file_path.stat().st_size
            manifest[arcname] = _categorize_file(file_path)

        # Write manifest inside the archive
        result.manifest = manifest
        manifest_json = json.dumps(manifest, indent=2, sort_keys=True)
        zf.writestr("manifest.json", manifest_json)

    return result
