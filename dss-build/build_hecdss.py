#!/usr/bin/env python3
"""
Build script for HEC-DSS shared library for Python.

This script builds the HEC-DSS library as a shared library (DLL/SO) that can be
used with pyiwfm via ctypes.

Usage:
    python build_hecdss.py [--heclib-source PATH] [--install]

Options:
    --heclib-source PATH  Path to existing heclib source directory
    --install            Install library to pyiwfm package
    --debug              Build debug configuration
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def find_cmake() -> Path:
    """Find CMake executable."""
    cmake_path = shutil.which("cmake")
    if cmake_path is None:
        raise RuntimeError(
            "CMake not found. Please install CMake 3.16 or later.\n"
            "Windows: https://cmake.org/download/\n"
            "Linux: sudo apt install cmake\n"
            "macOS: brew install cmake"
        )
    return Path(cmake_path)


def find_heclib_source() -> Path | None:
    """Find heclib source in IWFM build directory."""
    # Check common locations relative to this script
    script_dir = Path(__file__).parent
    possible_paths = [
        # pyiwfm/dss-build/ -> pyiwfm/ -> repo root -> src/build/...
        script_dir.parent.parent / "src" / "build" / "_deps" / "heclib-src" / "heclib" / "heclib_c",
        # Keep old patterns as fallbacks
        script_dir.parent.parent / "build" / "_deps" / "heclib-src" / "heclib" / "heclib_c",
        Path.cwd() / "build" / "_deps" / "heclib-src" / "heclib" / "heclib_c",
        Path.cwd() / "src" / "build" / "_deps" / "heclib-src" / "heclib" / "heclib_c",
    ]

    for path in possible_paths:
        if path.exists():
            return path

    return None


def run_cmake(
    source_dir: Path,
    build_dir: Path,
    heclib_source: Path | None = None,
    install: bool = False,
    debug: bool = False,
) -> None:
    """Run CMake to configure and build."""
    cmake = find_cmake()

    # Create build directory
    build_dir.mkdir(parents=True, exist_ok=True)

    # Configure options
    build_type = "Debug" if debug else "Release"
    cmake_args = [
        str(cmake),
        "-S", str(source_dir),
        "-B", str(build_dir),
        f"-DCMAKE_BUILD_TYPE={build_type}",
    ]

    if heclib_source:
        cmake_args.append(f"-DHECLIB_SOURCE_DIR={heclib_source}")

    # Platform-specific generator
    if platform.system() == "Windows":
        # Try to find Visual Studio or use Ninja
        if shutil.which("ninja"):
            cmake_args.extend(["-G", "Ninja"])
        else:
            # Default to Visual Studio
            cmake_args.extend(["-G", "Visual Studio 17 2022", "-A", "x64"])

    # Configure
    print(f"Configuring with CMake...")
    print(f"  Source: {source_dir}")
    print(f"  Build: {build_dir}")
    if heclib_source:
        print(f"  Heclib source: {heclib_source}")

    result = subprocess.run(cmake_args, cwd=build_dir)
    if result.returncode != 0:
        raise RuntimeError("CMake configuration failed")

    # Build
    print(f"\nBuilding ({build_type})...")
    build_args = [str(cmake), "--build", str(build_dir), "--config", build_type]
    result = subprocess.run(build_args)
    if result.returncode != 0:
        raise RuntimeError("Build failed")

    # Install if requested
    if install:
        print(f"\nInstalling...")
        install_args = [str(cmake), "--install", str(build_dir), "--config", build_type]
        result = subprocess.run(install_args)
        if result.returncode != 0:
            raise RuntimeError("Installation failed")

    print("\nBuild completed successfully!")

    # Show output location
    if platform.system() == "Windows":
        lib_name = "hecdss.dll"
        lib_path = build_dir / "bin" / lib_name
        if not lib_path.exists():
            lib_path = build_dir / build_type / lib_name
    elif platform.system() == "Darwin":
        lib_name = "libhecdss.dylib"
        lib_path = build_dir / "lib" / lib_name
    else:
        lib_name = "libhecdss.so"
        lib_path = build_dir / "lib" / lib_name

    if lib_path.exists():
        print(f"\nLibrary built: {lib_path}")
        print(f"\nTo use with pyiwfm, set environment variable:")
        print(f"  export HECDSS_LIB={lib_path}")
    else:
        # Search for the library
        for ext in [".dll", ".so", ".dylib"]:
            for f in build_dir.rglob(f"*hecdss*{ext}"):
                print(f"\nLibrary built: {f}")
                print(f"\nTo use with pyiwfm, set environment variable:")
                print(f"  export HECDSS_LIB={f}")
                break


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build HEC-DSS shared library for Python"
    )
    parser.add_argument(
        "--heclib-source",
        type=Path,
        help="Path to heclib source directory",
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="Install library to pyiwfm package",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Build debug configuration",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=None,
        help="Build directory (default: ./build)",
    )

    args = parser.parse_args()

    # Find source directory (where CMakeLists.txt is)
    source_dir = Path(__file__).parent

    # Find heclib source
    heclib_source = args.heclib_source
    if heclib_source is None:
        heclib_source = find_heclib_source()
        if heclib_source:
            print(f"Found heclib source: {heclib_source}")
        else:
            print("No local heclib source found. Will fetch from GitHub.")

    # Determine build directory
    build_dir = args.build_dir
    if build_dir is None:
        build_dir = source_dir / "build"

    try:
        run_cmake(
            source_dir=source_dir,
            build_dir=build_dir,
            heclib_source=heclib_source,
            install=args.install,
            debug=args.debug,
        )
        return 0
    except RuntimeError as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\nUnexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
