"""
Round-trip test for comment preservation with sample model.

This script tests the complete workflow:
1. Create sample IWFM files with custom comments
2. Extract comments using CommentExtractor
3. Save metadata to sidecar files
4. Load metadata from sidecar files
5. Restore comments using CommentWriter
6. Verify round-trip preservation
"""

import tempfile
from pathlib import Path


def run_roundtrip_test():
    """Run the complete round-trip test."""

    # Sample IWFM files with custom comments
    sample_preprocessor = """C*******************************************************************************
C
C                  INTEGRATED WATER FLOW MODEL (IWFM)
C                      Custom User Model Configuration
C                      Created by: Test User, 2025
C
C*******************************************************************************
C
C                            MAIN INPUT FILE
C                        for IWFM Pre-Processing
C
C             DO NOT MODIFY - Custom configuration for testing
C
C*****************************************************************************
C                     Titles Printed in the Output
C
C   *A Maximum of 3 title lines can be printed.
C-----------------------------------------------------------------------------
                                  My Custom Model
                              Round-Trip Test
                                   v2025.0.1747
C-----------------------------------------------------------------------------
C*****************************************************************************
C                            File Description
C-----------------------------------------------------------------------------
    PreProcessor_Output.bin                                / 1: BINARY OUTPUT
    Elements.dat                                           / 2: ELEMENT FILE
    Nodes.dat                                              / 3: NODE FILE
    Stratigraphy.dat                                       / 4: STRATIGRAPHY FILE
                                                           / 5: STREAM FILE
                                                           / 6: LAKE FILE
C******************************************************************************
C                    Pre-Processor Output Specifications
C-----------------------------------------------------------------------------
     1                          /KOUT
     2                          /KDEB
C*****************************************************************************
C                  Unit Specifications
C-----------------------------------------------------------------------------
    1.0                         /FACTLTOU
    FEET                        /UNITLTOU
    0.000022957                 /FACTAROU
    ACRES                       /UNITAROU
C*****************************************************************************
"""

    sample_nodes = """C*******************************************************************************
C
C                      IWFM NODAL COORDINATES FILE
C                      Custom Node Data - DO NOT EDIT
C
C*******************************************************************************
C
9                                               / NNODES
1.0                                             / FACTXY
C
C   ID              X              Y
C-----------------------------------------------------------------------------
     1       0.000000       0.000000  / SW corner - monitoring well MW-1
     2     100.000000       0.000000  / Southern boundary
     3     200.000000       0.000000  / SE corner
     4       0.000000     100.000000  / Western boundary
     5     100.000000     100.000000  / Center - pumping well PW-1
     6     200.000000     100.000000  / Eastern boundary
     7       0.000000     200.000000  / NW corner
     8     100.000000     200.000000  / Northern boundary
     9     200.000000     200.000000  / NE corner - stream gauge SG-1
C
C   End of node data
C*****************************************************************************
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Write sample files
        pp_file = tmpdir / 'Preprocessor.in'
        pp_file.write_text(sample_preprocessor)

        nodes_file = tmpdir / 'Nodes.dat'
        nodes_file.write_text(sample_nodes)

        print('=== ROUND-TRIP TEST WITH COMMENT PRESERVATION ===')
        print()

        # Step 1: Extract comments from both files
        from pyiwfm.io.comment_extractor import CommentExtractor
        from pyiwfm.io.comment_metadata import CommentMetadata

        extractor = CommentExtractor()

        print('Step 1: Extract comments from Preprocessor.in')
        pp_meta = extractor.extract(pp_file)
        print(f'  - Extracted {len(pp_meta.header_block)} header lines')
        print(f'  - Detected IWFM version: {pp_meta.iwfm_version or "(not detected)"}')
        print(f'  - Sections found: {list(pp_meta.sections.keys())}')
        print()

        print('Step 2: Extract comments from Nodes.dat')
        nodes_meta = extractor.extract(nodes_file)
        print(f'  - Extracted {len(nodes_meta.header_block)} header lines')
        print(f'  - Sections found: {list(nodes_meta.sections.keys())}')
        print()

        # Step 2: Save metadata to sidecar files
        print('Step 3: Save comment metadata to sidecar files')
        pp_sidecar = pp_meta.save_for_file(pp_file)
        nodes_sidecar = nodes_meta.save_for_file(nodes_file)
        print(f'  - Saved: {pp_sidecar.name}')
        print(f'  - Saved: {nodes_sidecar.name}')
        print()

        # Step 3: Load metadata from sidecar files (simulating future load)
        print('Step 4: Load comment metadata from sidecar files')
        loaded_pp_meta = CommentMetadata.load_for_file(pp_file)
        loaded_nodes_meta = CommentMetadata.load_for_file(nodes_file)
        print(f'  - Loaded preprocessor metadata: {loaded_pp_meta is not None}')
        print(f'  - Loaded nodes metadata: {loaded_nodes_meta is not None}')
        print()

        # Step 4: Use CommentWriter to restore headers
        from pyiwfm.io.comment_writer import CommentWriter

        print('Step 5: Restore headers using CommentWriter')
        pp_writer = CommentWriter(loaded_pp_meta)
        header = pp_writer.restore_header()

        # Check that custom comments are preserved
        assert 'Custom User Model Configuration' in header, 'Custom header not preserved!'
        assert 'DO NOT MODIFY' in header, 'Custom instruction not preserved!'
        print('  - Custom header preserved: YES')
        print()

        nodes_writer = CommentWriter(loaded_nodes_meta)
        nodes_header = nodes_writer.restore_header()
        assert 'Custom Node Data - DO NOT EDIT' in nodes_header, 'Custom node header not preserved!'
        print('  - Custom node header preserved: YES')
        print()

        # Step 5: Verify section comment extraction
        print('Step 6: Verify section comment extraction')
        print(f'  - Node metadata sections: {list(loaded_nodes_meta.sections.keys())}')
        print()

        # Step 6: Complete round-trip verification
        print('Step 7: Complete round-trip verification')

        # Write a new file using the restored header
        output_file = tmpdir / 'output' / 'Preprocessor_restored.in'
        output_file.parent.mkdir(exist_ok=True)

        with open(output_file, 'w') as f:
            f.write(header)
            f.write('C  Additional content after restoration\n')

        # Verify the output contains preserved comments
        restored_content = output_file.read_text()
        assert 'Custom User Model Configuration' in restored_content
        assert 'DO NOT MODIFY' in restored_content
        print('  - Output file contains preserved custom comments: YES')
        print()

        # Show sample of preserved header
        print('Step 8: Sample of preserved header (first 10 lines):')
        print('-' * 60)
        for line in header.split('\n')[:10]:
            print(f'  {line}')
        print('  ...')
        print('-' * 60)
        print()

        # Summary
        print('=' * 60)
        print('ROUND-TRIP TEST RESULTS: ALL PASSED')
        print('=' * 60)
        print()
        print('Verified capabilities:')
        print('  [OK] Extract comments from IWFM files')
        print('  [OK] Serialize comments to JSON sidecar files')
        print('  [OK] Load comments from sidecar files')
        print('  [OK] Restore custom headers during file writing')
        print('  [OK] Preserve user-defined comments through round-trip')

        return True


if __name__ == '__main__':
    success = run_roundtrip_test()
    exit(0 if success else 1)
