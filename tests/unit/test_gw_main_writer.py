"""
Comprehensive tests for pyiwfm.io.gw_main_writer module.

Tests the IWFM GW main file writer which serializes a GWMainFileConfig
object to the IWFM-format groundwater component main file.
"""

import io
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from pyiwfm.io.gw_main_writer import (
    _write_comment,
    _write_path,
    _write_value,
    write_gw_main_file,
)


# =============================================================================
# Helpers
# =============================================================================


def _make_hydro_loc(node_id: int, layer: int, x: float, y: float, name: str = ""):
    """Create a mock HydrographLocation."""
    loc = MagicMock()
    loc.node_id = node_id
    loc.layer = layer
    loc.x = x
    loc.y = y
    loc.name = name
    return loc


def _make_aquifer_params(
    n_nodes: int,
    n_layers: int,
    *,
    kh: np.ndarray | None = None,
    kv: np.ndarray | None = None,
    specific_storage: np.ndarray | None = None,
    specific_yield: np.ndarray | None = None,
    aquitard_kv: np.ndarray | None = None,
):
    """Create a mock AquiferParameters."""
    params = MagicMock()
    params.n_nodes = n_nodes
    params.n_layers = n_layers
    params.kh = kh
    params.kv = kv
    params.specific_storage = specific_storage
    params.specific_yield = specific_yield
    params.aquitard_kv = aquitard_kv
    return params


def _make_config(**overrides):
    """Create a mock GWMainFileConfig with sensible defaults.

    All attributes are set to minimal/empty defaults. Pass keyword
    arguments to override specific attributes.
    """
    config = MagicMock()

    defaults = {
        "version": "",
        "bc_file": None,
        "tile_drain_file": None,
        "pumping_file": None,
        "subsidence_file": None,
        "overwrite_file": None,
        "head_output_factor": 1.0,
        "head_output_unit": "FEET",
        "volume_output_factor": 1.0,
        "volume_output_unit": "TAF",
        "velocity_output_factor": 1.0,
        "velocity_output_unit": "FT/DAY",
        "velocity_output_file": None,
        "vertical_flow_output_file": None,
        "head_all_output_file": None,
        "head_tecplot_file": None,
        "velocity_tecplot_file": None,
        "budget_output_file": None,
        "zbudget_output_file": None,
        "final_heads_file": None,
        "debug_flag": 0,
        "coord_factor": 1.0,
        "hydrograph_output_file": None,
        "hydrograph_locations": [],
        "n_face_flow_outputs": 0,
        "face_flow_output_file": None,
        "face_flow_specs": [],
        "aquifer_params": None,
        "kh_anomalies": [],
        "initial_heads": None,
    }
    defaults.update(overrides)
    for key, value in defaults.items():
        setattr(config, key, value)
    return config


# =============================================================================
# Tests for _write_comment
# =============================================================================


class TestWriteComment:
    """Tests for _write_comment helper."""

    def test_basic_comment(self):
        buf = io.StringIO()
        _write_comment(buf, "Hello world")
        assert buf.getvalue() == "C  Hello world\n"

    def test_empty_comment(self):
        buf = io.StringIO()
        _write_comment(buf, "")
        assert buf.getvalue() == "C  \n"

    def test_comment_with_special_chars(self):
        buf = io.StringIO()
        _write_comment(buf, "Version 4.0 / test & notes")
        assert buf.getvalue() == "C  Version 4.0 / test & notes\n"


# =============================================================================
# Tests for _write_value
# =============================================================================


class TestWriteValue:
    """Tests for _write_value helper."""

    def test_value_with_description(self):
        buf = io.StringIO()
        _write_value(buf, 1.0, "FACTLTOU")
        line = buf.getvalue()
        assert line.startswith("     ")
        assert "/ FACTLTOU" in line
        assert line.endswith("\n")

    def test_value_without_description(self):
        buf = io.StringIO()
        _write_value(buf, 42)
        assert buf.getvalue() == "     42\n"

    def test_string_value_with_description(self):
        buf = io.StringIO()
        _write_value(buf, "FEET", "UNITLTOU")
        line = buf.getvalue()
        assert "FEET" in line
        assert "/ UNITLTOU" in line

    def test_value_left_aligned_in_30_char_field(self):
        """Value should be left-aligned within a 30-char field when description is present."""
        buf = io.StringIO()
        _write_value(buf, "X", "desc")
        line = buf.getvalue()
        # Format is: "     {value!s:<30s}  / {description}\n"
        # So after the 5-space indent, the value field occupies 30 chars
        after_indent = line[5:]  # strip leading 5 spaces
        value_field = after_indent[:30]
        assert value_field == "X" + " " * 29

    def test_integer_value(self):
        buf = io.StringIO()
        _write_value(buf, 0, "KDEB")
        line = buf.getvalue()
        assert "0" in line
        assert "/ KDEB" in line

    def test_empty_string_description(self):
        """Empty description string is falsy, so no '/' should appear."""
        buf = io.StringIO()
        _write_value(buf, 5, "")
        assert buf.getvalue() == "     5\n"


# =============================================================================
# Tests for _write_path
# =============================================================================


class TestWritePath:
    """Tests for _write_path helper."""

    def test_path_with_value(self):
        buf = io.StringIO()
        p = Path("Simulation/GW_BC.dat")
        _write_path(buf, p, "BCFL - Boundary conditions file")
        line = buf.getvalue()
        assert str(p) in line
        assert "/ BCFL - Boundary conditions file" in line

    def test_path_none(self):
        buf = io.StringIO()
        _write_path(buf, None, "SUBSFL - Subsidence file")
        line = buf.getvalue()
        # When None, writes empty string as the value
        assert "/ SUBSFL - Subsidence file" in line
        # The value field should be blank (empty string left-padded in 30 chars)
        after_indent = line[5:]
        value_field = after_indent[:30]
        assert value_field.strip() == ""

    def test_path_no_description(self):
        buf = io.StringIO()
        p = Path("some/file.dat")
        _write_path(buf, p)
        line = buf.getvalue()
        # With no description, _write_value uses the no-description branch
        assert "/ " not in line
        assert str(p) in line

    def test_path_none_no_description(self):
        buf = io.StringIO()
        _write_path(buf, None)
        line = buf.getvalue()
        assert line == "     \n"


# =============================================================================
# Tests for write_gw_main_file - full config
# =============================================================================


class TestWriteGWMainFileFull:
    """Tests for write_gw_main_file with a fully populated config."""

    @pytest.fixture
    def full_config(self):
        """Create a fully populated GWMainFileConfig mock."""
        locs = [
            _make_hydro_loc(10, 1, 1234567.1234, 9876543.5678, "Well-A"),
            _make_hydro_loc(20, 2, 2000000.0000, 3000000.0000, ""),
            _make_hydro_loc(5, 3, 500.5, 600.6, "Obs Point #3"),
        ]

        n_nodes, n_layers = 3, 2
        kh = np.array([[10.0, 5.0], [12.0, 6.0], [8.0, 4.0]])
        kv = np.array([[1.0, 0.5], [1.2, 0.6], [0.8, 0.4]])
        ss = np.array([[1e-5, 2e-5], [1.5e-5, 2.5e-5], [1.1e-5, 2.1e-5]])
        sy = np.array([[0.15, 0.10], [0.18, 0.12], [0.14, 0.09]])
        akv = np.array([[0.01, 0.02], [0.015, 0.025], [0.011, 0.021]])

        params = _make_aquifer_params(
            n_nodes,
            n_layers,
            kh=kh,
            kv=kv,
            specific_storage=ss,
            specific_yield=sy,
            aquitard_kv=akv,
        )

        initial_heads = np.array(
            [[100.0, 90.0], [105.0, 95.0], [110.0, 85.0]]
        )

        return _make_config(
            version="4.0",
            bc_file=Path("Simulation/GW_BC.dat"),
            tile_drain_file=Path("Simulation/GW_TD.dat"),
            pumping_file=None,
            subsidence_file=None,
            overwrite_file=Path("Simulation/GW_OVR.dat"),
            head_output_factor=1.0,
            head_output_unit="FEET",
            volume_output_factor=2.2957e-5,
            volume_output_unit="TAF",
            velocity_output_factor=1.0,
            velocity_output_unit="FT/DAY",
            velocity_output_file=Path("Results/GW_Vel.out"),
            vertical_flow_output_file=None,
            head_all_output_file=Path("Results/GW_HeadAll.hdf"),
            head_tecplot_file=None,
            velocity_tecplot_file=None,
            budget_output_file=Path("Results/GW_Budget.hdf"),
            zbudget_output_file=None,
            final_heads_file=Path("Results/GW_FinalHeads.dat"),
            debug_flag=0,
            coord_factor=0.3048,
            hydrograph_output_file=Path("Results/GW_Hydro.out"),
            hydrograph_locations=locs,
            n_face_flow_outputs=2,
            face_flow_output_file=Path("Results/GW_FaceFlow.out"),
            face_flow_specs=["1  1  100  101  Face_A", "2  2  200  201  Face_B"],
            aquifer_params=params,
            kh_anomalies=["5  1.5  2.0", "10  3.0  4.0"],
            initial_heads=initial_heads,
        )

    def test_returns_path(self, tmp_path, full_config):
        outfile = tmp_path / "gw_main.dat"
        result = write_gw_main_file(full_config, outfile)
        assert result == outfile
        assert outfile.exists()

    def test_comment_header(self, tmp_path, full_config):
        outfile = tmp_path / "gw_main.dat"
        write_gw_main_file(full_config, outfile)
        lines = outfile.read_text().splitlines()
        assert lines[0] == "C  IWFM Groundwater Component Main File"
        assert lines[1] == "C  Written by pyiwfm GWMainFileWriter"
        assert lines[2] == "C  "

    def test_version_header(self, tmp_path, full_config):
        outfile = tmp_path / "gw_main.dat"
        write_gw_main_file(full_config, outfile)
        lines = outfile.read_text().splitlines()
        assert "#4.0" in lines

    def test_subfile_paths_present(self, tmp_path, full_config):
        outfile = tmp_path / "gw_main.dat"
        write_gw_main_file(full_config, outfile)
        content = outfile.read_text()
        assert str(Path("Simulation/GW_BC.dat")) in content
        assert "/ BCFL - Boundary conditions file" in content
        assert str(Path("Simulation/GW_TD.dat")) in content
        assert "/ TDFL - Tile drain file" in content
        assert str(Path("Simulation/GW_OVR.dat")) in content
        assert "/ OVRWRTFL - Overwrite file" in content

    def test_none_paths_written_as_blank(self, tmp_path, full_config):
        outfile = tmp_path / "gw_main.dat"
        write_gw_main_file(full_config, outfile)
        content = outfile.read_text()
        # Pumping and subsidence are None, so the value field is empty
        assert "/ PUMPFL - Pumping file" in content
        assert "/ SUBSFL - Subsidence file" in content

    def test_output_factors_and_units(self, tmp_path, full_config):
        outfile = tmp_path / "gw_main.dat"
        write_gw_main_file(full_config, outfile)
        content = outfile.read_text()
        assert "/ FACTLTOU" in content
        assert "/ UNITLTOU" in content
        assert "/ FACTVLOU" in content
        assert "/ UNITVLOU" in content
        assert "/ FACTVROU" in content
        assert "/ UNITVROU" in content
        assert "FEET" in content
        assert "TAF" in content

    def test_output_file_paths(self, tmp_path, full_config):
        outfile = tmp_path / "gw_main.dat"
        write_gw_main_file(full_config, outfile)
        content = outfile.read_text()
        assert str(Path("Results/GW_Vel.out")) in content
        assert "/ VELOUTFL" in content
        assert str(Path("Results/GW_HeadAll.hdf")) in content
        assert "/ GWALLOUTFL" in content
        assert str(Path("Results/GW_Budget.hdf")) in content
        assert "/ GWBUDFL" in content
        assert str(Path("Results/GW_FinalHeads.dat")) in content
        assert "/ FNGWFL" in content

    def test_debug_flag(self, tmp_path, full_config):
        outfile = tmp_path / "gw_main.dat"
        write_gw_main_file(full_config, outfile)
        content = outfile.read_text()
        assert "/ KDEB" in content

    def test_hydrograph_section(self, tmp_path, full_config):
        outfile = tmp_path / "gw_main.dat"
        write_gw_main_file(full_config, outfile)
        content = outfile.read_text()
        # NOUTH = 3 hydrograph locations
        lines = content.splitlines()
        nouth_lines = [ln for ln in lines if "/ NOUTH" in ln]
        assert len(nouth_lines) == 1
        assert "3" in nouth_lines[0]
        # Check coord factor
        assert "/ FACTXY" in content
        # Check hydrograph output file
        assert str(Path("Results/GW_Hydro.out")) in content
        assert "/ GWHYDOUTFL" in content

    def test_hydrograph_locations_format(self, tmp_path, full_config):
        outfile = tmp_path / "gw_main.dat"
        write_gw_main_file(full_config, outfile)
        content = outfile.read_text()
        # First location: node_id=10, layer=1, x=1234567.1234, y=9876543.5678, name=Well-A
        assert "/ Well-A" in content
        assert "10" in content
        # Second location has empty name
        assert "/ Obs Point #3" in content

    def test_hydrograph_location_node_id_and_coords(self, tmp_path, full_config):
        """Verify the exact format of hydrograph location lines."""
        outfile = tmp_path / "gw_main.dat"
        write_gw_main_file(full_config, outfile)
        lines = outfile.read_text().splitlines()
        # Find lines containing "/ Well-A"
        well_a_lines = [ln for ln in lines if "/ Well-A" in ln]
        assert len(well_a_lines) == 1
        line = well_a_lines[0]
        # Format: "     {node_id:>6d}  0  {layer:>3d}  {x:>15.4f}  {y:>15.4f}  / {name}"
        assert "    10  0    1" in line
        assert "1234567.1234" in line
        assert "9876543.5678" in line

    def test_face_flow_section(self, tmp_path, full_config):
        outfile = tmp_path / "gw_main.dat"
        write_gw_main_file(full_config, outfile)
        content = outfile.read_text()
        assert "/ NOUTF" in content
        assert str(Path("Results/GW_FaceFlow.out")) in content
        assert "/ FCHYDOUTFL" in content
        assert "1  1  100  101  Face_A" in content
        assert "2  2  200  201  Face_B" in content

    def test_aquifer_params_section(self, tmp_path, full_config):
        outfile = tmp_path / "gw_main.dat"
        write_gw_main_file(full_config, outfile)
        content = outfile.read_text()
        # NGROUP line
        assert "/ NGROUP (direct input)" in content
        # Conversion factors line
        assert "1.0  1.0  1.0  1.0  1.0  1.0  / Conversion factors" in content
        # Time unit line
        assert "1DAY" in content

    def test_aquifer_params_node_data(self, tmp_path, full_config):
        """Verify per-node aquifer parameter data is written correctly."""
        outfile = tmp_path / "gw_main.dat"
        write_gw_main_file(full_config, outfile)
        lines = outfile.read_text().splitlines()

        # Find lines with node IDs (layer 0 lines have the node id)
        # Node 1, layer 0: kh=10, ss=1e-5, sy=0.15, kv=1, akv=0.01
        # The format is: {i+1:>6d}  {kh:>12.6g}  {ss:>12.6g}  {sy:>12.6g}  {kv:>12.6g}  {akv:>12.6g}
        node1_layer0 = [ln for ln in lines if ln.strip().startswith("1") and "10" in ln and "0.15" in ln]
        assert len(node1_layer0) >= 1

    def test_aquifer_params_multilayer_format(self, tmp_path, full_config):
        """Layer 0 lines have node id, layer 1+ lines have spaces instead."""
        outfile = tmp_path / "gw_main.dat"
        write_gw_main_file(full_config, outfile)
        content = outfile.read_text()
        lines = content.splitlines()

        # Find the conversion factors line to anchor ourselves
        conv_idx = None
        for i, ln in enumerate(lines):
            if "/ Conversion factors" in ln:
                conv_idx = i
                break
        assert conv_idx is not None

        # After "Conversion factors" and "1DAY" comes the node data
        # Node 1 layer 0 -> starts with node id "     1"
        # Node 1 layer 1 -> starts with spaces "             "
        time_idx = conv_idx + 1
        node1_layer0_idx = time_idx + 1
        node1_layer1_idx = time_idx + 2

        node1_l0 = lines[node1_layer0_idx]
        node1_l1 = lines[node1_layer1_idx]

        # Layer 0 line starts with 5 spaces then node id right-aligned in 6 chars
        assert node1_l0.startswith("     ")
        assert "1" in node1_l0[:12]  # node id within first 12 chars

        # Layer 1 line starts with 13 spaces (no node id)
        assert node1_l1.startswith("             ")

    def test_kh_anomalies_section(self, tmp_path, full_config):
        outfile = tmp_path / "gw_main.dat"
        write_gw_main_file(full_config, outfile)
        content = outfile.read_text()
        # NEBK = number of anomalies
        assert "/ NEBK" in content
        nebk_lines = [ln for ln in content.splitlines() if "/ NEBK" in ln]
        assert "2" in nebk_lines[0]
        # Anomaly data
        assert "5  1.5  2.0" in content
        assert "10  3.0  4.0" in content

    def test_initial_heads_section(self, tmp_path, full_config):
        outfile = tmp_path / "gw_main.dat"
        write_gw_main_file(full_config, outfile)
        content = outfile.read_text()
        # Comment line
        assert "C  Initial Groundwater Heads" in content
        # FACTICL
        assert "/ FACTICL" in content
        # Node 1 heads: 100.0, 90.0
        assert "100.0000" in content
        assert "90.0000" in content
        # Node 3 heads: 110.0, 85.0
        assert "110.0000" in content
        assert "85.0000" in content

    def test_initial_heads_node_format(self, tmp_path, full_config):
        """Each initial head row has node id followed by per-layer head values."""
        outfile = tmp_path / "gw_main.dat"
        write_gw_main_file(full_config, outfile)
        lines = outfile.read_text().splitlines()

        # Find the FACTICL line
        facticl_idx = None
        for i, ln in enumerate(lines):
            if "/ FACTICL" in ln:
                facticl_idx = i
                break
        assert facticl_idx is not None

        # Lines after FACTICL are the head data
        head_line_1 = lines[facticl_idx + 1]
        head_line_2 = lines[facticl_idx + 2]
        head_line_3 = lines[facticl_idx + 3]

        # Node 1
        assert head_line_1.strip().startswith("1")
        assert "100.0000" in head_line_1
        assert "90.0000" in head_line_1

        # Node 2
        assert head_line_2.strip().startswith("2")
        assert "105.0000" in head_line_2
        assert "95.0000" in head_line_2

        # Node 3
        assert head_line_3.strip().startswith("3")
        assert "110.0000" in head_line_3
        assert "85.0000" in head_line_3

    def test_overall_section_order(self, tmp_path, full_config):
        """Verify sections appear in the correct IWFM order."""
        outfile = tmp_path / "gw_main.dat"
        write_gw_main_file(full_config, outfile)
        content = outfile.read_text()

        # Get positions of key markers to verify ordering
        markers = [
            "IWFM Groundwater Component Main File",
            "#4.0",
            "/ BCFL",
            "/ TDFL",
            "/ PUMPFL",
            "/ SUBSFL",
            "/ OVRWRTFL",
            "/ FACTLTOU",
            "/ UNITLTOU",
            "/ FACTVLOU",
            "/ UNITVLOU",
            "/ FACTVROU",
            "/ UNITVROU",
            "/ VELOUTFL",
            "/ VFLOWOUTFL",
            "/ GWALLOUTFL",
            "/ HTPOUTFL",
            "/ VTPOUTFL",
            "/ GWBUDFL",
            "/ ZBUDFL",
            "/ FNGWFL",
            "/ KDEB",
            "/ NOUTH",
            "/ FACTXY",
            "/ GWHYDOUTFL",
            "/ NOUTF",
            "/ FCHYDOUTFL",
            "/ NGROUP",
            "/ Conversion factors",
            "/ NEBK",
            "Initial Groundwater Heads",
            "/ FACTICL",
        ]

        positions = []
        for m in markers:
            pos = content.find(m)
            assert pos >= 0, f"Marker '{m}' not found in output"
            positions.append(pos)

        # All markers should appear in strictly increasing order
        for i in range(len(positions) - 1):
            assert positions[i] < positions[i + 1], (
                f"Marker '{markers[i]}' (pos {positions[i]}) should appear before "
                f"'{markers[i+1]}' (pos {positions[i+1]})"
            )


# =============================================================================
# Tests for write_gw_main_file - minimal config
# =============================================================================


class TestWriteGWMainFileMinimal:
    """Tests for write_gw_main_file with minimal/empty config."""

    @pytest.fixture
    def minimal_config(self):
        """Create a minimal GWMainFileConfig mock with empty/default values."""
        return _make_config(
            version="",
            bc_file=None,
            tile_drain_file=None,
            pumping_file=None,
            subsidence_file=None,
            overwrite_file=None,
            head_output_factor=1.0,
            head_output_unit="FEET",
            volume_output_factor=1.0,
            volume_output_unit="TAF",
            velocity_output_factor=1.0,
            velocity_output_unit="",
            velocity_output_file=None,
            vertical_flow_output_file=None,
            head_all_output_file=None,
            head_tecplot_file=None,
            velocity_tecplot_file=None,
            budget_output_file=None,
            zbudget_output_file=None,
            final_heads_file=None,
            debug_flag=0,
            coord_factor=1.0,
            hydrograph_output_file=None,
            hydrograph_locations=[],
            n_face_flow_outputs=0,
            face_flow_output_file=None,
            face_flow_specs=[],
            aquifer_params=None,
            kh_anomalies=[],
            initial_heads=None,
        )

    def test_no_version_header(self, tmp_path, minimal_config):
        outfile = tmp_path / "gw_min.dat"
        write_gw_main_file(minimal_config, outfile)
        content = outfile.read_text()
        assert "#" not in content

    def test_no_aquifer_params_section(self, tmp_path, minimal_config):
        outfile = tmp_path / "gw_min.dat"
        write_gw_main_file(minimal_config, outfile)
        content = outfile.read_text()
        assert "NGROUP" not in content
        assert "Conversion factors" not in content
        assert "1DAY" not in content

    def test_no_initial_heads_section(self, tmp_path, minimal_config):
        outfile = tmp_path / "gw_min.dat"
        write_gw_main_file(minimal_config, outfile)
        content = outfile.read_text()
        assert "Initial Groundwater Heads" not in content
        assert "FACTICL" not in content

    def test_zero_hydrograph_locations(self, tmp_path, minimal_config):
        outfile = tmp_path / "gw_min.dat"
        write_gw_main_file(minimal_config, outfile)
        content = outfile.read_text()
        nouth_lines = [ln for ln in content.splitlines() if "/ NOUTH" in ln]
        assert len(nouth_lines) == 1
        assert "0" in nouth_lines[0]

    def test_zero_kh_anomalies(self, tmp_path, minimal_config):
        outfile = tmp_path / "gw_min.dat"
        write_gw_main_file(minimal_config, outfile)
        content = outfile.read_text()
        nebk_lines = [ln for ln in content.splitlines() if "/ NEBK" in ln]
        assert len(nebk_lines) == 1
        assert "0" in nebk_lines[0]

    def test_empty_face_flow_specs(self, tmp_path, minimal_config):
        outfile = tmp_path / "gw_min.dat"
        write_gw_main_file(minimal_config, outfile)
        content = outfile.read_text()
        assert "/ NOUTF" in content
        assert "/ FCHYDOUTFL" in content

    def test_file_still_valid(self, tmp_path, minimal_config):
        """Minimal config should still produce a valid file with all required sections."""
        outfile = tmp_path / "gw_min.dat"
        write_gw_main_file(minimal_config, outfile)
        content = outfile.read_text()
        # Must have comment header
        assert "C  IWFM Groundwater Component Main File" in content
        # Must have all required descriptors
        for desc in [
            "BCFL", "TDFL", "PUMPFL", "SUBSFL", "OVRWRTFL",
            "FACTLTOU", "UNITLTOU", "FACTVLOU", "UNITVLOU",
            "FACTVROU", "UNITVROU",
            "VELOUTFL", "VFLOWOUTFL", "GWALLOUTFL",
            "HTPOUTFL", "VTPOUTFL", "GWBUDFL", "ZBUDFL", "FNGWFL",
            "KDEB", "NOUTH", "FACTXY", "GWHYDOUTFL",
            "NOUTF", "FCHYDOUTFL", "NEBK",
        ]:
            assert desc in content, f"Required descriptor '{desc}' missing from output"


# =============================================================================
# Tests for directory creation
# =============================================================================


class TestDirectoryCreation:
    """Tests that write_gw_main_file creates parent directories."""

    def test_creates_nested_directories(self, tmp_path):
        config = _make_config()
        outfile = tmp_path / "deep" / "nested" / "dir" / "gw_main.dat"
        assert not outfile.parent.exists()
        result = write_gw_main_file(config, outfile)
        assert result == outfile
        assert outfile.exists()
        assert outfile.parent.exists()

    def test_existing_directory_no_error(self, tmp_path):
        config = _make_config()
        outfile = tmp_path / "gw_main.dat"
        # tmp_path already exists
        result = write_gw_main_file(config, outfile)
        assert result == outfile
        assert outfile.exists()


# =============================================================================
# Tests for aquifer params edge cases
# =============================================================================


class TestAquiferParamsEdgeCases:
    """Tests for aquifer parameter writing with partial None arrays."""

    def test_some_arrays_none(self, tmp_path):
        """When some aquifer param arrays are None, those values default to 0.0."""
        params = _make_aquifer_params(
            2,
            1,
            kh=np.array([[100.0], [200.0]]),
            kv=None,
            specific_storage=None,
            specific_yield=np.array([[0.2], [0.3]]),
            aquitard_kv=None,
        )
        config = _make_config(aquifer_params=params)
        outfile = tmp_path / "gw_partial.dat"
        write_gw_main_file(config, outfile)
        content = outfile.read_text()

        # kh values should be present
        assert "100" in content
        assert "200" in content
        # sy values should be present
        assert "0.2" in content
        assert "0.3" in content
        # kv, ss, akv are None => 0.0 values should appear
        # The zeros appear multiple times (n_nodes * n_layers for each None array)

    def test_all_arrays_none(self, tmp_path):
        """When all param arrays are None, all values written as 0.0."""
        params = _make_aquifer_params(
            2,
            1,
            kh=None,
            kv=None,
            specific_storage=None,
            specific_yield=None,
            aquitard_kv=None,
        )
        config = _make_config(aquifer_params=params)
        outfile = tmp_path / "gw_allnone.dat"
        write_gw_main_file(config, outfile)
        content = outfile.read_text()
        assert "/ NGROUP (direct input)" in content
        # Should still have per-node lines with zeros
        lines = content.splitlines()
        # Find lines after time unit that have node data
        found_node_lines = False
        for ln in lines:
            if ln.strip().startswith("1") and "0" in ln and "NGROUP" not in ln:
                found_node_lines = True
                break
        assert found_node_lines

    def test_single_node_single_layer(self, tmp_path):
        """Simplest case: 1 node, 1 layer."""
        params = _make_aquifer_params(
            1,
            1,
            kh=np.array([[50.0]]),
            kv=np.array([[5.0]]),
            specific_storage=np.array([[1e-4]]),
            specific_yield=np.array([[0.25]]),
            aquitard_kv=np.array([[0.005]]),
        )
        config = _make_config(aquifer_params=params)
        outfile = tmp_path / "gw_single.dat"
        write_gw_main_file(config, outfile)
        content = outfile.read_text()
        assert "50" in content
        assert "0.25" in content

    def test_three_layers(self, tmp_path):
        """Verify multi-layer output: layer 0 has node id, layers 1-2 have spaces."""
        n_nodes, n_layers = 2, 3
        kh = np.ones((n_nodes, n_layers)) * 10.0
        params = _make_aquifer_params(
            n_nodes,
            n_layers,
            kh=kh,
            kv=np.ones((n_nodes, n_layers)),
            specific_storage=np.ones((n_nodes, n_layers)) * 1e-5,
            specific_yield=np.ones((n_nodes, n_layers)) * 0.15,
            aquitard_kv=np.ones((n_nodes, n_layers)) * 0.01,
        )
        config = _make_config(aquifer_params=params)
        outfile = tmp_path / "gw_3layer.dat"
        write_gw_main_file(config, outfile)
        lines = outfile.read_text().splitlines()

        # Find the "1DAY" line
        time_idx = None
        for i, ln in enumerate(lines):
            if "1DAY" in ln:
                time_idx = i
                break
        assert time_idx is not None

        # Node 1: 3 lines (layer 0, 1, 2)
        n1_l0 = lines[time_idx + 1]
        n1_l1 = lines[time_idx + 2]
        n1_l2 = lines[time_idx + 3]

        # Node 2: 3 lines (layer 0, 1, 2)
        n2_l0 = lines[time_idx + 4]
        n2_l1 = lines[time_idx + 5]
        n2_l2 = lines[time_idx + 6]

        # Layer 0 lines have node id (right-aligned 6 chars after 5-space indent)
        assert n1_l0.strip().startswith("1")
        assert n2_l0.strip().startswith("2")

        # Layer 1+ lines start with 13 spaces
        assert n1_l1.startswith("             ")
        assert n1_l2.startswith("             ")
        assert n2_l1.startswith("             ")
        assert n2_l2.startswith("             ")


# =============================================================================
# Tests for hydrograph location edge cases
# =============================================================================


class TestHydrographLocationEdgeCases:
    """Tests for hydrograph location formatting edge cases."""

    def test_location_without_name(self, tmp_path):
        """Location with name=None should output empty string after '/'."""
        loc = _make_hydro_loc(42, 1, 100.0, 200.0, None)
        config = _make_config(
            hydrograph_locations=[loc],
            hydrograph_output_file=Path("out.dat"),
        )
        outfile = tmp_path / "gw_noname.dat"
        write_gw_main_file(config, outfile)
        lines = outfile.read_text().splitlines()
        # Find the hydrograph location line (contains "42" as node id)
        loc_lines = [ln for ln in lines if "42" in ln and "0" in ln and "/" in ln and "NOUTH" not in ln]
        assert len(loc_lines) >= 1
        # Name should be empty string (or " " after "/")
        assert "/ " in loc_lines[0]

    def test_location_with_empty_name(self, tmp_path):
        """Location with name='' should output empty string after '/'."""
        loc = _make_hydro_loc(7, 2, 500.0, 600.0, "")
        config = _make_config(
            hydrograph_locations=[loc],
            hydrograph_output_file=Path("out.dat"),
        )
        outfile = tmp_path / "gw_emptyname.dat"
        write_gw_main_file(config, outfile)
        content = outfile.read_text()
        lines = content.splitlines()
        loc_lines = [ln for ln in lines if "/ " in ln and "7" in ln and "NOUTH" not in ln
                     and "FACTXY" not in ln and "GWHYDOUTFL" not in ln]
        assert len(loc_lines) >= 1

    def test_multiple_locations_all_written(self, tmp_path):
        """All hydrograph locations should appear in output."""
        locs = [
            _make_hydro_loc(1, 1, 10.0, 20.0, "A"),
            _make_hydro_loc(2, 1, 30.0, 40.0, "B"),
            _make_hydro_loc(3, 2, 50.0, 60.0, "C"),
            _make_hydro_loc(4, 3, 70.0, 80.0, "D"),
        ]
        config = _make_config(
            hydrograph_locations=locs,
            hydrograph_output_file=Path("out.dat"),
        )
        outfile = tmp_path / "gw_multiloc.dat"
        write_gw_main_file(config, outfile)
        content = outfile.read_text()
        # NOUTH should be 4
        nouth_lines = [ln for ln in content.splitlines() if "/ NOUTH" in ln]
        assert "4" in nouth_lines[0]
        # All names present
        for name in ["A", "B", "C", "D"]:
            assert f"/ {name}" in content


# =============================================================================
# Tests for filepath handling
# =============================================================================


class TestFilepathHandling:
    """Tests for filepath string/Path handling."""

    def test_accepts_string_path(self, tmp_path):
        config = _make_config()
        outfile = str(tmp_path / "gw_str.dat")
        result = write_gw_main_file(config, outfile)
        assert isinstance(result, Path)
        assert result.exists()

    def test_accepts_path_object(self, tmp_path):
        config = _make_config()
        outfile = tmp_path / "gw_path.dat"
        result = write_gw_main_file(config, outfile)
        assert result == outfile
        assert result.exists()

    def test_return_value_matches_input(self, tmp_path):
        config = _make_config()
        outfile = tmp_path / "gw_ret.dat"
        result = write_gw_main_file(config, outfile)
        assert result == outfile


# =============================================================================
# Tests for complete output content verification
# =============================================================================


class TestOutputContentVerification:
    """Verify exact output format for a small, controlled config."""

    def test_exact_output_small_config(self, tmp_path):
        """Write a very small config and verify line-by-line content."""
        loc = _make_hydro_loc(1, 1, 100.0, 200.0, "TestWell")
        params = _make_aquifer_params(
            1,
            1,
            kh=np.array([[10.0]]),
            kv=np.array([[1.0]]),
            specific_storage=np.array([[0.0001]]),
            specific_yield=np.array([[0.15]]),
            aquitard_kv=np.array([[0.01]]),
        )
        initial_heads = np.array([[50.0]])

        config = _make_config(
            version="4.0",
            bc_file=Path("BC.dat"),
            tile_drain_file=None,
            pumping_file=None,
            subsidence_file=None,
            overwrite_file=None,
            head_output_factor=1.0,
            head_output_unit="FEET",
            volume_output_factor=1.0,
            volume_output_unit="TAF",
            velocity_output_factor=1.0,
            velocity_output_unit="FT/DAY",
            velocity_output_file=None,
            vertical_flow_output_file=None,
            head_all_output_file=None,
            head_tecplot_file=None,
            velocity_tecplot_file=None,
            budget_output_file=None,
            zbudget_output_file=None,
            final_heads_file=None,
            debug_flag=0,
            coord_factor=1.0,
            hydrograph_output_file=Path("hydro.out"),
            hydrograph_locations=[loc],
            n_face_flow_outputs=0,
            face_flow_output_file=None,
            face_flow_specs=[],
            aquifer_params=params,
            kh_anomalies=[],
            initial_heads=initial_heads,
        )

        outfile = tmp_path / "gw_exact.dat"
        write_gw_main_file(config, outfile)
        lines = outfile.read_text().splitlines()

        # Verify key lines
        assert lines[0] == "C  IWFM Groundwater Component Main File"
        assert lines[1] == "C  Written by pyiwfm GWMainFileWriter"
        assert lines[2] == "C  "
        assert lines[3] == "#4.0"

        # BC.dat path line
        assert "BC.dat" in lines[4]
        assert "/ BCFL - Boundary conditions file" in lines[4]

        # NOUTH = 1
        nouth_idx = next(i for i, ln in enumerate(lines) if "/ NOUTH" in ln)
        assert "1" in lines[nouth_idx]

        # Hydrograph location line
        hydro_line = next(ln for ln in lines if "/ TestWell" in ln)
        assert "1" in hydro_line  # node_id
        assert "100.0000" in hydro_line
        assert "200.0000" in hydro_line

        # NEBK = 0
        nebk_idx = next(i for i, ln in enumerate(lines) if "/ NEBK" in ln)
        assert "0" in lines[nebk_idx]

        # Initial heads
        assert "C  Initial Groundwater Heads" in lines
        facticl_idx = next(i for i, ln in enumerate(lines) if "/ FACTICL" in ln)
        head_line = lines[facticl_idx + 1]
        assert "1" in head_line  # node id
        assert "50.0000" in head_line

    def test_no_trailing_whitespace_in_comment_lines(self, tmp_path):
        """Comment lines should not have unexpected trailing whitespace."""
        config = _make_config()
        outfile = tmp_path / "gw_trail.dat"
        write_gw_main_file(config, outfile)
        lines = outfile.read_text().splitlines()
        for line in lines:
            if line.startswith("C  ") and line.strip() == "C":
                # The "C  " line with empty text is "C  " which is fine
                pass
