"""
Full round-trip test using the IWFM Sample Model.

This script:
1. Reads the existing sample model using pyiwfm comment extraction
2. Extracts comments from all input files
3. Writes two versions:
   - Version with preserved comments (original + sidecar JSON files)
   - Version with default template headers (generic headers)
4. Runs IWFM preprocessor + simulation on both versions to verify they work
5. Compares results to verify both produce successful simulations

Prerequisites:
- Sample model at samplemodel/ with inline comments converted to '#'
- Executables at src/Bin/ (Jan 2025+ builds with '#' comment support)
"""

import subprocess
import sys
import shutil
from pathlib import Path
from datetime import datetime

# Paths
BASE_DIR = Path(r"C:\Users\hatch\OneDrive\Desktop\iwfm-2025.0.1747")
SAMPLE_MODEL_DIR = BASE_DIR / "samplemodel"
PYIWFM_SRC = BASE_DIR / "src" / "pyiwfm" / "src"
SRC_BIN = BASE_DIR / "src" / "Bin"
SIMULATION_EXE = SRC_BIN / "Simulation_x64.exe"
PREPROCESSOR_EXE = SRC_BIN / "PreProcessor_x64.exe"

# Output directories (within samplemodel to keep paths manageable)
TEST_DIR = SAMPLE_MODEL_DIR / "_roundtrip_test"
PRESERVED_DIR = TEST_DIR / "preserved"
DEFAULT_DIR = TEST_DIR / "default"

# Add pyiwfm to path
sys.path.insert(0, str(PYIWFM_SRC))

# Timeouts in seconds
PP_TIMEOUT = 120       # 2 minutes for preprocessor
SIM_TIMEOUT = 900      # 15 minutes for simulation

# Files/patterns to skip when copying
SKIP_SUFFIXES = {".bak", ".out", ".iwfm_comments.json"}
SKIP_DIRS = {"Results", "Budget", "ZBudget", "Bin", "Sample Model Documentation",
             "_roundtrip_test"}


def log(msg: str) -> None:
    """Print a timestamped log message."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


def check_prerequisites() -> bool:
    """Check that all required files and executables exist."""
    log("Checking prerequisites...")

    checks = [
        ("Sample model directory", SAMPLE_MODEL_DIR.exists()),
        ("Simulation executable", SIMULATION_EXE.exists()),
        ("PreProcessor executable", PREPROCESSOR_EXE.exists()),
        ("Preprocessor main file", (SAMPLE_MODEL_DIR / "Preprocessor" / "PreProcessor_MAIN.IN").exists()),
        ("Simulation main file", (SAMPLE_MODEL_DIR / "Simulation" / "Simulation_MAIN.IN").exists()),
        ("PreProcessor.bin", (SAMPLE_MODEL_DIR / "Simulation" / "PreProcessor.bin").exists()),
        ("TSDATA_IN.DSS", (SAMPLE_MODEL_DIR / "Simulation" / "TSDATA_IN.DSS").exists()),
    ]

    all_ok = True
    for name, ok in checks:
        status = "OK" if ok else "MISSING"
        print(f"  - {name}: {status}")
        if not ok:
            all_ok = False

    return all_ok


def copy_model(src_dir: Path, dest_dir: Path) -> None:
    """Copy model files to destination, skipping backups and outputs."""
    log(f"Copying model to {dest_dir.name}...")

    if dest_dir.exists():
        shutil.rmtree(dest_dir, ignore_errors=True)
        # Wait briefly for filesystem/OneDrive to release handles
        import time
        time.sleep(1)
    dest_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for item in src_dir.iterdir():
        if item.name in SKIP_DIRS:
            if item.name == "Results":
                # Create empty Results directory (simulation writes here)
                (dest_dir / item.name).mkdir(exist_ok=True)
            continue

        if item.is_dir():
            # Copy subdirectory, filtering out unwanted files
            _copy_dir_filtered(item, dest_dir / item.name)
            count += 1
        elif item.suffix not in SKIP_SUFFIXES:
            shutil.copy2(item, dest_dir / item.name)
            count += 1

    log(f"  Copied {count} items")


def _copy_dir_filtered(src: Path, dest: Path) -> None:
    """Recursively copy directory, skipping backup and output files."""
    dest.mkdir(parents=True, exist_ok=True)

    for item in src.iterdir():
        if item.is_dir():
            _copy_dir_filtered(item, dest / item.name)
        elif item.suffix not in SKIP_SUFFIXES:
            shutil.copy2(item, dest / item.name)


def extract_comments_from_model(model_dir: Path) -> dict:
    """Extract comments from all key input files."""
    from pyiwfm.io.comment_extractor import CommentExtractor

    log("Extracting comments from input files...")

    extractor = CommentExtractor()
    comments = {}

    # All input files to extract comments from
    files_to_extract = [
        # Preprocessor files
        ("preprocessor_main", model_dir / "Preprocessor" / "PreProcessor_MAIN.IN"),
        ("elements", model_dir / "Preprocessor" / "Element.dat"),
        ("nodes", model_dir / "Preprocessor" / "NodeXY.dat"),
        ("stratigraphy", model_dir / "Preprocessor" / "Strata.dat"),
        ("streams_pp", model_dir / "Preprocessor" / "Stream.dat"),
        ("lakes_pp", model_dir / "Preprocessor" / "Lake.dat"),
        # Simulation files
        ("simulation_main", model_dir / "Simulation" / "Simulation_MAIN.IN"),
        ("gw_main", model_dir / "Simulation" / "GW" / "GW_MAIN.dat"),
        ("stream_main", model_dir / "Simulation" / "Stream" / "Stream_MAIN.dat"),
        ("lake_main", model_dir / "Simulation" / "Lake" / "Lake_MAIN.dat"),
        ("rootzone_main", model_dir / "Simulation" / "RootZone" / "RootZone_MAIN.dat"),
        ("swshed", model_dir / "Simulation" / "SWShed.dat"),
        ("unsatzone", model_dir / "Simulation" / "UnsatZone.dat"),
    ]

    for file_type, filepath in files_to_extract:
        if filepath.exists():
            try:
                metadata = extractor.extract(filepath)
                comments[file_type] = metadata
                n_header = len(metadata.header_block)
                n_sections = len(metadata.sections)
                log(f"  - {file_type}: {n_header} header lines, {n_sections} sections")
            except Exception as e:
                log(f"  - {file_type}: ERROR - {e}")
        else:
            log(f"  - {file_type}: not found")

    return comments


def write_preserved_version(model_dir: Path, output_dir: Path, comments: dict) -> None:
    """Write model with preserved comments (copy + save sidecar files)."""
    log("Writing preserved-comments version...")

    # Copy all model files
    copy_model(model_dir, output_dir)

    # Save sidecar JSON files alongside each input file
    file_mapping = {
        "preprocessor_main": output_dir / "Preprocessor" / "PreProcessor_MAIN.IN",
        "elements": output_dir / "Preprocessor" / "Element.dat",
        "nodes": output_dir / "Preprocessor" / "NodeXY.dat",
        "stratigraphy": output_dir / "Preprocessor" / "Strata.dat",
        "streams_pp": output_dir / "Preprocessor" / "Stream.dat",
        "lakes_pp": output_dir / "Preprocessor" / "Lake.dat",
        "simulation_main": output_dir / "Simulation" / "Simulation_MAIN.IN",
        "gw_main": output_dir / "Simulation" / "GW" / "GW_MAIN.dat",
        "stream_main": output_dir / "Simulation" / "Stream" / "Stream_MAIN.dat",
        "lake_main": output_dir / "Simulation" / "Lake" / "Lake_MAIN.dat",
        "rootzone_main": output_dir / "Simulation" / "RootZone" / "RootZone_MAIN.dat",
        "swshed": output_dir / "Simulation" / "SWShed.dat",
        "unsatzone": output_dir / "Simulation" / "UnsatZone.dat",
    }

    sidecar_count = 0
    for file_type, metadata in comments.items():
        if file_type in file_mapping:
            target = file_mapping[file_type]
            if target.exists():
                metadata.save_for_file(target)
                sidecar_count += 1

    log(f"  Saved {sidecar_count} sidecar files")

    # Verify original comments are still in files
    pp_main = output_dir / "Preprocessor" / "PreProcessor_MAIN.IN"
    if pp_main.exists():
        content = pp_main.read_text()
        if "IWFM Public Release" in content:
            log("  Preprocessor file: original comments confirmed")


def write_default_version(model_dir: Path, output_dir: Path) -> None:
    """Write model with default/generic template headers."""
    log("Writing default-templates version...")

    # Copy all model files
    copy_model(model_dir, output_dir)

    # Replace headers in key files with generic versions
    _replace_header(
        output_dir / "Preprocessor" / "PreProcessor_MAIN.IN",
        "MAIN INPUT FILE for IWFM Pre-Processing",
    )
    _replace_header(
        output_dir / "Simulation" / "Simulation_MAIN.IN",
        "MAIN INPUT FILE for IWFM Simulation",
    )
    _replace_header(
        output_dir / "Simulation" / "GW" / "GW_MAIN.dat",
        "Groundwater Component Main File",
    )

    log("  Replaced headers in 3 key files")


def _replace_header(filepath: Path, description: str) -> None:
    """Replace a file's header block with a generic version.

    Preserves version markers (e.g. '#4.0') and 'DO NOT DELETE' lines
    that appear before the header banner, as IWFM requires these.
    """
    if not filepath.exists():
        return

    content = filepath.read_text(encoding="utf-8", errors="replace")
    lines = content.split("\n")

    # Identify pre-header lines that must be preserved (version markers, etc.)
    # These are lines before the first banner (C***...) that are NOT comment-only
    pre_header = []
    first_banner_idx = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("C*") and len(stripped) > 10:
            first_banner_idx = i
            break
        # Preserve version markers like '#4.0' and 'DO NOT DELETE' lines
        pre_header.append(line)

    # Generic header (replaces the banner blocks and descriptive comments)
    generic_header = [
        "C" + "*" * 78,
        "C",
        "C                  INTEGRATED WATER FLOW MODEL (IWFM)",
        "C                      Generated by pyiwfm round-trip test",
        "C",
        f"C    {description}",
        "C",
        "C" + "*" * 78,
    ]

    # Find where the header comment block ends and data begins
    # Look for the first non-comment data line after the header banners
    header_end = 0
    banner_count = 0
    for i in range(first_banner_idx, len(lines)):
        stripped = lines[i].strip()
        if stripped.startswith("C*") and len(stripped) > 10:
            banner_count += 1
        # After seeing at least 2 banners, find the first data line
        if banner_count >= 2:
            for j in range(i + 1, len(lines)):
                sj = lines[j].strip()
                if sj and not sj.startswith(("C", "c", "*")):
                    header_end = j
                    break
            if header_end > 0:
                break

    if header_end == 0:
        # Fallback: find "Titles Printed" section
        for i, line in enumerate(lines):
            if "Titles Printed" in line:
                header_end = i
                break

    if header_end > 0:
        parts = []
        if pre_header:
            parts.append("\n".join(pre_header))
        parts.append("\n".join(generic_header))
        parts.append("\n".join(lines[header_end:]))
        new_content = "\n".join(parts)
        filepath.write_text(new_content, encoding="utf-8")


def run_preprocessor(model_dir: Path, name: str) -> tuple:
    """Run IWFM preprocessor. Returns (success, output_text)."""
    log(f"Running preprocessor ({name})...")

    pp_dir = model_dir / "Preprocessor"
    batch_file = pp_dir / "_run_pp.bat"

    batch_content = f'@echo off\ncd /d "{pp_dir}"\necho PreProcessor_MAIN.IN | "{PREPROCESSOR_EXE}"\n'
    batch_file.write_text(batch_content)

    try:
        result = subprocess.run(
            [str(batch_file)],
            cwd=str(pp_dir),
            capture_output=True,
            text=True,
            timeout=PP_TIMEOUT,
            shell=True,
        )

        output = result.stdout + result.stderr

        # Check messages file and stdout for FATAL errors
        msg_file = pp_dir / "PreprocessorMessages.out"
        messages = ""
        if msg_file.exists():
            messages = msg_file.read_text(errors="replace")

        combined = output + messages

        # FATAL errors mean failure even if TOTAL RUN TIME is printed
        if "FATAL" in combined.upper():
            log(f"  Preprocessor encountered FATAL error")
            # Extract the fatal message
            for line in combined.split("\n"):
                if "FATAL" in line.upper() or "Error" in line:
                    log(f"    {line.strip()}")
            return False, combined

        if "TOTAL RUN TIME" in combined:
            log(f"  Preprocessor completed successfully")
            return True, output

        log(f"  Preprocessor: no clear success/failure indicator")
        log(f"  Exit code: {result.returncode}")
        return result.returncode == 0, output

    except subprocess.TimeoutExpired:
        log(f"  Preprocessor timed out")
        return False, "Timeout"
    except Exception as e:
        log(f"  Preprocessor error: {e}")
        return False, str(e)


def run_simulation(model_dir: Path, name: str) -> tuple:
    """Run IWFM simulation. Returns (success, output_text)."""
    log(f"Running simulation ({name})...")
    log(f"  (This will take approximately 10-12 minutes)")

    sim_dir = model_dir / "Simulation"
    batch_file = sim_dir / "_run_sim.bat"

    batch_content = f'@echo off\ncd /d "{sim_dir}"\necho Simulation_MAIN.IN | "{SIMULATION_EXE}"\n'
    batch_file.write_text(batch_content)

    try:
        result = subprocess.run(
            [str(batch_file)],
            cwd=str(sim_dir),
            capture_output=True,
            text=True,
            timeout=SIM_TIMEOUT,
            shell=True,
        )

        output = result.stdout + result.stderr

        # Check messages file too
        msg_file = sim_dir / "SimulationMessages.out"
        messages = ""
        if msg_file.exists():
            messages = msg_file.read_text(errors="replace")

        combined = output + messages

        # FATAL errors mean failure even if TOTAL RUN TIME is printed
        if "FATAL" in combined.upper():
            log(f"  Simulation encountered FATAL error")
            for line in combined.split("\n"):
                if "FATAL" in line.upper() or ("Error" in line and "FATAL" not in line.upper()):
                    log(f"    {line.strip()}")
            return False, combined

        if "TOTAL RUN TIME" in combined:
            log(f"  Simulation completed successfully")
            return True, output

        log(f"  Simulation: no clear success indicator")
        log(f"  Exit code: {result.returncode}")
        # Print last 10 lines of output for debugging
        out_lines = output.strip().split("\n")
        for line in out_lines[-10:]:
            log(f"    {line}")
        return False, output

    except subprocess.TimeoutExpired:
        log(f"  Simulation timed out after {SIM_TIMEOUT}s")
        return False, "Timeout"
    except Exception as e:
        log(f"  Simulation error: {e}")
        return False, str(e)


def verify_comment_preservation(preserved_dir: Path, default_dir: Path, comments: dict) -> dict:
    """Verify that comment preservation worked correctly."""
    log("Verifying comment preservation...")

    results = {}

    # Check preserved version has original comments
    pp_preserved = preserved_dir / "Preprocessor" / "PreProcessor_MAIN.IN"
    if pp_preserved.exists():
        content = pp_preserved.read_text(errors="replace")
        results["preserved_has_original_header"] = "IWFM Public Release" in content
        log(f"  Preserved version has original header: {results['preserved_has_original_header']}")

    # Check default version has generic header
    pp_default = default_dir / "Preprocessor" / "PreProcessor_MAIN.IN"
    if pp_default.exists():
        content = pp_default.read_text(errors="replace")
        results["default_has_generic_header"] = "pyiwfm round-trip test" in content
        results["default_lacks_original"] = "IWFM Public Release" not in content
        log(f"  Default version has generic header: {results['default_has_generic_header']}")
        log(f"  Default version lacks original header: {results['default_lacks_original']}")

    # Check sidecar files exist in preserved version
    sidecar = pp_preserved.parent / (pp_preserved.name + ".iwfm_comments.json")
    results["sidecar_exists"] = sidecar.exists()
    log(f"  Sidecar file exists: {results['sidecar_exists']}")

    # Verify sidecar round-trip
    if sidecar.exists():
        from pyiwfm.io.comment_metadata import CommentMetadata
        loaded = CommentMetadata.load(sidecar)
        results["sidecar_loadable"] = loaded is not None
        if loaded:
            results["sidecar_has_header"] = len(loaded.header_block) > 0
            results["sidecar_has_sections"] = len(loaded.sections) > 0
            log(f"  Sidecar loadable: True ({len(loaded.header_block)} header lines, "
                f"{len(loaded.sections)} sections)")
        else:
            log(f"  Sidecar loadable: False")

    # Check that data content is identical (comments shouldn't affect data)
    pp_orig_data = _extract_data_lines(SAMPLE_MODEL_DIR / "Preprocessor" / "PreProcessor_MAIN.IN")
    pp_pres_data = _extract_data_lines(pp_preserved)
    pp_def_data = _extract_data_lines(pp_default)

    results["preserved_data_matches_original"] = pp_orig_data == pp_pres_data
    results["default_data_matches_original"] = pp_def_data == pp_orig_data
    log(f"  Preserved data matches original: {results['preserved_data_matches_original']}")
    log(f"  Default data matches original: {results['default_data_matches_original']}")

    return results


def _extract_data_lines(filepath: Path) -> list:
    """Extract non-comment data lines from a file (for comparison)."""
    if not filepath.exists():
        return []
    lines = []
    for line in filepath.read_text(errors="replace").split("\n"):
        stripped = line.strip()
        if stripped and not stripped.startswith(("C", "c", "*")):
            # Strip inline comments for comparison
            if "#" in line:
                data_part = line.split("#")[0].strip()
            else:
                data_part = stripped
            if data_part:
                lines.append(data_part)
    return lines


def cleanup(test_dir: Path) -> None:
    """Clean up test directories."""
    if test_dir.exists():
        log(f"Cleaning up {test_dir}...")
        shutil.rmtree(test_dir, ignore_errors=True)


def run_roundtrip_test() -> bool:
    """Run the complete round-trip test."""
    print("=" * 70)
    print("IWFM SAMPLE MODEL ROUND-TRIP TEST WITH COMMENT PRESERVATION")
    print("=" * 70)
    print()

    # Check prerequisites
    if not check_prerequisites():
        log("Prerequisites check failed. Exiting.")
        return False
    print()

    # Clean up any previous test
    cleanup(TEST_DIR)
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # Phase 1: Extract comments from original model
        print("-" * 70)
        print("PHASE 1: EXTRACT COMMENTS")
        print("-" * 70)
        comments = extract_comments_from_model(SAMPLE_MODEL_DIR)
        log(f"Total files with comments extracted: {len(comments)}")
        print()

        # Phase 2: Write two model versions
        print("-" * 70)
        print("PHASE 2: WRITE MODEL VERSIONS")
        print("-" * 70)
        write_preserved_version(SAMPLE_MODEL_DIR, PRESERVED_DIR, comments)
        print()
        write_default_version(SAMPLE_MODEL_DIR, DEFAULT_DIR)
        print()

        # Phase 3: Run preprocessor on both
        print("-" * 70)
        print("PHASE 3: RUN PREPROCESSOR")
        print("-" * 70)
        pp_success_1, pp_out_1 = run_preprocessor(PRESERVED_DIR, "preserved")
        pp_success_2, pp_out_2 = run_preprocessor(DEFAULT_DIR, "default")
        print()

        if not pp_success_1:
            log("ERROR: Preprocessor failed on preserved version")
            log("Output:")
            for line in pp_out_1.strip().split("\n")[-20:]:
                print(f"  {line}")
            return False

        if not pp_success_2:
            log("ERROR: Preprocessor failed on default version")
            log("Output:")
            for line in pp_out_2.strip().split("\n")[-20:]:
                print(f"  {line}")
            return False

        # Phase 4: Run simulation on both
        print("-" * 70)
        print("PHASE 4: RUN SIMULATIONS")
        print("-" * 70)
        sim_success_1, sim_out_1 = run_simulation(PRESERVED_DIR, "preserved")
        print()
        sim_success_2, sim_out_2 = run_simulation(DEFAULT_DIR, "default")
        print()

        # Phase 5: Verify comment preservation
        print("-" * 70)
        print("PHASE 5: VERIFY COMMENT PRESERVATION")
        print("-" * 70)
        verification = verify_comment_preservation(PRESERVED_DIR, DEFAULT_DIR, comments)
        print()

        # Final Report
        print("=" * 70)
        print("TEST RESULTS SUMMARY")
        print("=" * 70)
        print()

        print("Comment Extraction:")
        for file_type, metadata in comments.items():
            n_header = len(metadata.header_block)
            n_sections = len(metadata.sections)
            total_inline = sum(len(s.inline_comments) for s in metadata.sections.values())
            print(f"  {file_type}: {n_header} header lines, {n_sections} sections, "
                  f"{total_inline} inline comments")
        print()

        print("Preprocessor:")
        print(f"  Preserved comments version: {'PASS' if pp_success_1 else 'FAIL'}")
        print(f"  Default templates version:  {'PASS' if pp_success_2 else 'FAIL'}")
        print()

        print("Simulation:")
        print(f"  Preserved comments version: {'PASS' if sim_success_1 else 'FAIL'}")
        print(f"  Default templates version:  {'PASS' if sim_success_2 else 'FAIL'}")
        print()

        print("Comment Preservation:")
        for check, passed in verification.items():
            status = "PASS" if passed else "FAIL"
            print(f"  {check}: {status}")
        print()

        # Overall
        all_passed = (
            pp_success_1 and pp_success_2
            and sim_success_1 and sim_success_2
            and all(verification.values())
        )

        print("=" * 70)
        if all_passed:
            print("OVERALL: ALL TESTS PASSED")
        else:
            print("OVERALL: SOME TESTS FAILED")
        print("=" * 70)

        return all_passed

    except Exception as e:
        log(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_roundtrip_test()
    sys.exit(0 if success else 1)
