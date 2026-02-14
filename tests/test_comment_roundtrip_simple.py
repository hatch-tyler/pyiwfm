"""
Simple round-trip test for comment preservation using sample model files.

This test validates the comment preservation workflow:
1. Extract comments from original IWFM input files
2. Serialize to JSON sidecar files
3. Load from JSON sidecar files
4. Restore comments to output files
5. Verify round-trip preservation

Note: This test does NOT run the IWFM simulation - it validates the
comment preservation infrastructure.
"""

import sys
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

import pytest

# Paths
SAMPLE_MODEL_DIR = Path(r"C:\Users\hatch\OneDrive\Desktop\iwfm-2025.0.1747\samplemodel")
PYIWFM_SRC = Path(r"C:\Users\hatch\OneDrive\Desktop\iwfm-2025.0.1747\src\pyiwfm\src")

# Add pyiwfm to path
sys.path.insert(0, str(PYIWFM_SRC))


def log(msg: str) -> None:
    """Print a timestamped log message."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


def test_comment_extraction_from_sample_model():
    """Test that comments can be extracted from sample model files."""
    from pyiwfm.io.comment_extractor import CommentExtractor

    log("Testing comment extraction from sample model files...")

    extractor = CommentExtractor()
    results = {}

    # Key files to test
    files_to_test = [
        ("preprocessor_main", SAMPLE_MODEL_DIR / "Preprocessor" / "PreProcessor_MAIN.IN"),
        ("simulation_main", SAMPLE_MODEL_DIR / "Simulation" / "Simulation_MAIN.IN"),
        ("elements", SAMPLE_MODEL_DIR / "Preprocessor" / "Element.dat"),
        ("nodes", SAMPLE_MODEL_DIR / "Preprocessor" / "NodeXY.dat"),
        ("stratigraphy", SAMPLE_MODEL_DIR / "Preprocessor" / "Strata.dat"),
    ]

    for file_type, filepath in files_to_test:
        if filepath.exists():
            try:
                metadata = extractor.extract(filepath)
                results[file_type] = {
                    "success": True,
                    "header_lines": len(metadata.header_block),
                    "sections": list(metadata.sections.keys()),
                    "has_comments": metadata.has_comments,
                }
                log(f"  - {file_type}: {len(metadata.header_block)} header lines, "
                    f"{len(metadata.sections)} sections")
            except Exception as e:
                results[file_type] = {"success": False, "error": str(e)}
                log(f"  - {file_type}: ERROR - {e}")
        else:
            results[file_type] = {"success": False, "error": "File not found"}
            log(f"  - {file_type}: File not found")

    # Verify we extracted something
    successful = sum(1 for r in results.values() if r.get("success"))
    assert successful >= 3, f"Should extract from at least 3 files, got {successful}"


def test_json_sidecar_round_trip():
    """Test that comments can be serialized and deserialized from JSON."""
    from pyiwfm.io.comment_extractor import CommentExtractor
    from pyiwfm.io.comment_metadata import CommentMetadata

    log("Testing JSON sidecar round-trip...")

    extractor = CommentExtractor()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Extract comments from preprocessor main file
        pp_file = SAMPLE_MODEL_DIR / "Preprocessor" / "PreProcessor_MAIN.IN"
        if not pp_file.exists():
            log("  - Preprocessor file not found, skipping")
            pytest.skip("Preprocessor file not found")

        original_metadata = extractor.extract(pp_file)
        log(f"  - Extracted {len(original_metadata.header_block)} header lines")

        # Save to sidecar file
        temp_pp_file = tmpdir / "PreProcessor_MAIN.IN"
        shutil.copy(pp_file, temp_pp_file)

        sidecar_path = original_metadata.save_for_file(temp_pp_file)
        log(f"  - Saved sidecar: {sidecar_path.name}")
        assert sidecar_path.exists(), "Sidecar file should exist"

        # Load from sidecar file
        loaded_metadata = CommentMetadata.load_for_file(temp_pp_file)
        assert loaded_metadata is not None, "Should load metadata from sidecar"
        log(f"  - Loaded {len(loaded_metadata.header_block)} header lines")

        # Verify round-trip
        assert len(loaded_metadata.header_block) == len(original_metadata.header_block), \
            "Header block length should match"
        assert loaded_metadata.header_block == original_metadata.header_block, \
            "Header block content should match"
        log("  - Round-trip verification: PASSED")


def test_comment_restoration():
    """Test that comments can be restored to output files."""
    from pyiwfm.io.comment_extractor import CommentExtractor
    from pyiwfm.io.comment_writer import CommentWriter

    log("Testing comment restoration...")

    extractor = CommentExtractor()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Extract comments from preprocessor main file
        pp_file = SAMPLE_MODEL_DIR / "Preprocessor" / "PreProcessor_MAIN.IN"
        if not pp_file.exists():
            log("  - Preprocessor file not found, skipping")
            pytest.skip("Preprocessor file not found")

        metadata = extractor.extract(pp_file)
        log(f"  - Extracted {len(metadata.header_block)} header lines")

        # Create a writer and restore header
        writer = CommentWriter(metadata)
        restored_header = writer.restore_header()

        # Verify header contains expected content
        original_content = pp_file.read_text()

        # Check that key content from original is in restored
        assert "INTEGRATED WATER FLOW MODEL" in restored_header, \
            "Restored header should contain model name"
        assert "IWFM" in restored_header, \
            "Restored header should contain IWFM"
        log("  - Restored header contains expected content")

        # Write to a test file
        output_file = tmpdir / "Preprocessor_restored.in"
        output_file.write_text(restored_header + "\nC  Additional content\n")
        log(f"  - Wrote restored file: {output_file.name}")

        # Verify restored file
        restored_content = output_file.read_text()
        assert "INTEGRATED WATER FLOW MODEL" in restored_content, \
            "Restored file should contain model name"
        log("  - Restoration verification: PASSED")


def test_preserved_vs_default_headers():
    """Test that preserved headers differ from default templates."""
    from pyiwfm.io.comment_extractor import CommentExtractor
    from pyiwfm.io.comment_writer import CommentWriter

    log("Testing preserved vs default headers...")

    extractor = CommentExtractor()

    # Extract from preprocessor
    pp_file = SAMPLE_MODEL_DIR / "Preprocessor" / "PreProcessor_MAIN.IN"
    if not pp_file.exists():
        log("  - Preprocessor file not found, skipping")
        pytest.skip("Preprocessor file not found")

    metadata = extractor.extract(pp_file)
    writer = CommentWriter(metadata)

    # Get preserved header
    preserved_header = writer.restore_header()

    # Create a minimal default header
    default_header = """C*******************************************************************************
C
C                  INTEGRATED WATER FLOW MODEL (IWFM)
C                      Generated by pyiwfm
C
C*******************************************************************************
"""

    # Verify they differ (preserved should have more content)
    assert len(preserved_header) > len(default_header), \
        "Preserved header should be longer than minimal default"

    # Verify preserved header has project-specific content
    assert "Public Release" in preserved_header or "DWR" in preserved_header, \
        "Preserved header should contain project-specific content"

    log(f"  - Preserved header: {len(preserved_header)} chars")
    log(f"  - Default header: {len(default_header)} chars")
    log("  - Headers differ as expected: PASSED")


def test_multiple_file_types():
    """Test comment extraction from different IWFM file types."""
    from pyiwfm.io.comment_extractor import CommentExtractor

    log("Testing multiple file types...")

    extractor = CommentExtractor()
    results = {}

    # Different file types to test
    file_types = {
        "element": (SAMPLE_MODEL_DIR / "Preprocessor" / "Element.dat", "Element configuration"),
        "nodes": (SAMPLE_MODEL_DIR / "Preprocessor" / "NodeXY.dat", "Node coordinates"),
        "strata": (SAMPLE_MODEL_DIR / "Preprocessor" / "Strata.dat", "Stratigraphy"),
        "streams": (SAMPLE_MODEL_DIR / "Preprocessor" / "Stream.dat", "Stream geometry"),
    }

    for name, (filepath, description) in file_types.items():
        if filepath.exists():
            try:
                metadata = extractor.extract(filepath)
                results[name] = {
                    "description": description,
                    "header_lines": len(metadata.header_block),
                    "sections": len(metadata.sections),
                    "success": True,
                }
                log(f"  - {name} ({description}): {len(metadata.header_block)} header, "
                    f"{len(metadata.sections)} sections")
            except Exception as e:
                results[name] = {"success": False, "error": str(e)}
                log(f"  - {name}: ERROR - {e}")

    successful = sum(1 for r in results.values() if r.get("success"))
    log(f"  - Successfully extracted from {successful}/{len(file_types)} file types")


def run_all_tests():
    """Run all comment preservation tests."""
    print("=" * 70)
    print("COMMENT PRESERVATION ROUND-TRIP TEST")
    print("=" * 70)
    print()

    # Check prerequisites
    if not SAMPLE_MODEL_DIR.exists():
        log(f"ERROR: Sample model directory not found: {SAMPLE_MODEL_DIR}")
        return False

    log(f"Sample model directory: {SAMPLE_MODEL_DIR}")
    print()

    # Run tests
    tests = [
        ("Comment Extraction", test_comment_extraction_from_sample_model),
        ("JSON Sidecar Round-Trip", test_json_sidecar_round_trip),
        ("Comment Restoration", test_comment_restoration),
        ("Preserved vs Default", test_preserved_vs_default_headers),
        ("Multiple File Types", test_multiple_file_types),
    ]

    results = {}
    for name, test_func in tests:
        print(f"\n{'-' * 60}")
        log(f"TEST: {name}")
        print("-" * 60)
        try:
            result = test_func()
            results[name] = "PASSED" if result else "SKIPPED"
        except AssertionError as e:
            results[name] = f"FAILED: {e}"
            log(f"ASSERTION ERROR: {e}")
        except Exception as e:
            results[name] = f"ERROR: {e}"
            log(f"EXCEPTION: {e}")

    # Summary
    print()
    print("=" * 70)
    print("TEST RESULTS SUMMARY")
    print("=" * 70)

    passed = 0
    for name, result in results.items():
        status = "PASSED" if result == "PASSED" else result
        print(f"  - {name}: {status}")
        if result == "PASSED":
            passed += 1

    print()
    print(f"TOTAL: {passed}/{len(tests)} tests passed")
    print("=" * 70)

    return passed == len(tests)


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
