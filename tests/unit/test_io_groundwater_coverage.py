"""Coverage tests for io/groundwater.py module.

Tests _is_comment_line, _strip_comment, KhAnomalyEntry,
ParametricGridData, GWFileConfig path methods,
GroundwaterReader.read_wells, GroundwaterReader.read_initial_heads,
and GWMainFileReader for version header, file paths, aquifer
parameters section, and hydrograph locations.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyiwfm.core.exceptions import FileFormatError
from pyiwfm.io.groundwater import (
    GroundwaterReader,
    GWFileConfig,
    GWMainFileReader,
    KhAnomalyEntry,
    ParametricGridData,
    _is_comment_line,
    _strip_comment,
)

# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestIsCommentLine:
    """Tests for _is_comment_line helper."""

    def test_c_uppercase(self) -> None:
        assert _is_comment_line("C  This is a comment") is True

    def test_c_lowercase(self) -> None:
        assert _is_comment_line("c  lowercase") is True

    def test_asterisk(self) -> None:
        assert _is_comment_line("* asterisk") is True

    def test_empty_string(self) -> None:
        assert _is_comment_line("") is True

    def test_whitespace_only(self) -> None:
        assert _is_comment_line("   ") is True

    def test_data_line(self) -> None:
        assert _is_comment_line("10  20  30") is False

    def test_hash_line_not_comment(self) -> None:
        assert _is_comment_line("# 4.0") is False

    def test_indented_c_not_comment(self) -> None:
        """A 'C' that is NOT in column 1 is NOT a comment line."""
        assert _is_comment_line("  C indented") is False


class TestParseValueLine:
    """Tests for _strip_comment helper."""

    def test_slash_delimiter(self) -> None:
        value, desc = _strip_comment("100  / NWELLS")
        assert value == "100"
        assert desc == "NWELLS"

    def test_hash_not_delimiter(self) -> None:
        """'#' is NOT a comment delimiter in IWFM — only '/' is."""
        value, desc = _strip_comment("4.0 # version")
        assert value == "4.0 # version"
        assert desc == ""

    def test_no_description(self) -> None:
        value, desc = _strip_comment("42")
        assert value == "42"
        assert desc == ""

    def test_path_with_slashes_not_split(self) -> None:
        """Date-style slashes (09/30/1990) should not be split."""
        value, desc = _strip_comment("09/30/1990")
        assert value == "09/30/1990"
        assert desc == ""


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------


class TestKhAnomalyEntry:
    """Tests for KhAnomalyEntry dataclass."""

    def test_construction(self) -> None:
        entry = KhAnomalyEntry(element_id=5, kh_per_layer=[0.01, 0.02])
        assert entry.element_id == 5
        assert entry.kh_per_layer == [0.01, 0.02]


class TestParametricGridData:
    """Tests for ParametricGridData dataclass."""

    def test_construction(self) -> None:
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
        values = np.zeros((3, 2, 5))
        pgd = ParametricGridData(
            n_nodes=3,
            n_elements=1,
            elements=[(0, 1, 2)],
            node_coords=coords,
            node_values=values,
        )
        assert pgd.n_nodes == 3
        assert pgd.n_elements == 1
        assert len(pgd.elements) == 1
        assert pgd.node_coords.shape == (3, 2)
        assert pgd.node_values.shape == (3, 2, 5)


# ---------------------------------------------------------------------------
# GWFileConfig path methods
# ---------------------------------------------------------------------------


class TestGWFileConfigPaths:
    """Tests for GWFileConfig path accessor methods."""

    def test_all_path_methods(self, tmp_path: Path) -> None:
        config = GWFileConfig(output_dir=tmp_path)
        assert config.get_wells_path() == tmp_path / "wells.dat"
        assert config.get_pumping_path() == tmp_path / "pumping.dat"
        assert config.get_aquifer_params_path() == tmp_path / "aquifer_params.dat"
        assert config.get_boundary_conditions_path() == tmp_path / "boundary_conditions.dat"
        assert config.get_tile_drains_path() == tmp_path / "tile_drains.dat"
        assert config.get_subsidence_path() == tmp_path / "subsidence.dat"
        assert config.get_initial_heads_path() == tmp_path / "initial_heads.dat"

    def test_custom_filenames(self, tmp_path: Path) -> None:
        config = GWFileConfig(
            output_dir=tmp_path,
            wells_file="my_wells.in",
        )
        assert config.get_wells_path() == tmp_path / "my_wells.in"


# ---------------------------------------------------------------------------
# GroundwaterReader.read_wells
# ---------------------------------------------------------------------------


class TestReadWells:
    """Tests for GroundwaterReader.read_wells from temp files."""

    def test_basic_wells(self, tmp_path: Path) -> None:
        filepath = tmp_path / "wells.dat"
        filepath.write_text(
            "C  Wells file\n"
            "2                        / NWELLS\n"
            "1  1000.0  2000.0  1  50.0  100.0  500.0  Well_A\n"
            "2  1500.0  2500.0  2  60.0  120.0  750.0  Well_B\n"
        )
        reader = GroundwaterReader()
        wells = reader.read_wells(filepath)
        assert len(wells) == 2
        assert wells[1].x == pytest.approx(1000.0)
        assert wells[2].name == "Well_B"

    def test_wells_with_comments_between_rows(self, tmp_path: Path) -> None:
        filepath = tmp_path / "wells.dat"
        filepath.write_text(
            "C  Header\n1  / NWELLS\nC  Data follows\n1  100.0  200.0  1  10.0  20.0  300.0  W1\n"
        )
        reader = GroundwaterReader()
        wells = reader.read_wells(filepath)
        assert len(wells) == 1

    def test_wells_invalid_nwells_raises(self, tmp_path: Path) -> None:
        filepath = tmp_path / "wells.dat"
        filepath.write_text("abc  / NWELLS\n")
        with pytest.raises(FileFormatError, match="Invalid NWELLS"):
            GroundwaterReader().read_wells(filepath)


# ---------------------------------------------------------------------------
# GroundwaterReader.read_initial_heads
# ---------------------------------------------------------------------------


class TestReadInitialHeads:
    """Tests for GroundwaterReader.read_initial_heads."""

    def test_basic(self, tmp_path: Path) -> None:
        filepath = tmp_path / "heads.dat"
        filepath.write_text(
            "C  Initial heads\n"
            "3  / NNODES\n"
            "2  / NLAYERS\n"
            "C  ID  HEAD_L01  HEAD_L02\n"
            "1  100.0  90.0\n"
            "2  105.0  95.0\n"
            "3  110.0  100.0\n"
        )
        reader = GroundwaterReader()
        n_nodes, n_layers, heads = reader.read_initial_heads(filepath)
        assert n_nodes == 3
        assert n_layers == 2
        assert heads.shape == (3, 2)
        assert heads[0, 0] == pytest.approx(100.0)
        assert heads[2, 1] == pytest.approx(100.0)

    def test_missing_nnodes_raises(self, tmp_path: Path) -> None:
        filepath = tmp_path / "heads.dat"
        filepath.write_text("C  only comments\n")
        with pytest.raises(FileFormatError):
            GroundwaterReader().read_initial_heads(filepath)


# ---------------------------------------------------------------------------
# GWMainFileReader
# ---------------------------------------------------------------------------


def _write_gw_main(path: Path, version: str = "4.0") -> None:
    """Write a minimal GW main file for testing."""
    lines = [
        "C  Groundwater component main file\n",
        f"# {version}\n",
        # File paths (5 fields)
        "bc_file.dat  / BCFL\n",
        "tile_drain.dat  / TDFL\n",
        "pumping.dat  / PUMPFL\n",
        "subsidence.dat  / SUBSFL\n",
        "  / OVRWRTFL (empty)\n",
        # Conversion factors and units
        "1.0  / FACTLTOU\n",
        "FEET  / UNITLTOU\n",
        "1.0  / FACTVLOU\n",
        "TAF  / UNITVLOU\n",
        "1.0  / FACTVROU\n",
        "FT/DAY  / UNITVROU\n",
        # Output files (8 optional)
        "  / VELOUTFL\n",
        "  / VFLOWOUTFL\n",
        "  / GWALLOUTFL\n",
        "  / HTPOUTFL\n",
        "  / VTPOUTFL\n",
        "  / GWBUDFL\n",
        "  / ZBUDFL\n",
        "  / FNGWFL\n",
        # ITECPLOTFLAG then KDEB
        "1  / ITECPLOTFLAG\n",
        "0  / KDEB\n",
        # Hydrographs
        "2  / NOUTH\n",
        "1.0  / FACTXY\n",
        "hydout.dat  / GWHYDOUTFL\n",
        # 2 hydrograph locations: HYDTYP=1 (node-based)
        "1  1  1  10  Station_A\n",
        "2  1  2  20  Station_B\n",
        # Face flow output
        "0  / NOUTF\n",
        "  / FCHYDOUTFL\n",
        # Aquifer parameters (NGROUP=0, per-node)
        "0  / NGROUP\n",
        "1.0 1.0 1.0 1.0 1.0 1.0  / FX FKH FS FN FV FL\n",
        "1DAY  / TUNITKH\n",
        "1DAY  / TUNITV\n",
        "1DAY  / TUNITL\n",
        # Two nodes, 1 layer each (ID + 5 params = 6 fields)
        "1  10.0  0.001  0.15  0.01  5.0\n",
        "2  12.0  0.002  0.18  0.02  6.0\n",
        # End of aquifer params: comment line triggers end
        "C  End of aquifer parameters\n",
        # Kh anomaly section (NEBK=0 still needs FACT and TUNITH)
        "0  / NEBK\n",
        "1.0  / FACT\n",
        "1DAY  / TUNITH\n",
        # Return flow flag
        "0  / IFLAGRF\n",
        # Initial heads section
        "1.0  / FACTHP\n",
        "1  50.0\n",
        "2  55.0\n",
    ]
    path.write_text("".join(lines))


class TestGWMainFileReaderVersion:
    """Tests for version header reading."""

    def test_reads_version(self, tmp_path: Path) -> None:
        filepath = tmp_path / "gw_main.dat"
        _write_gw_main(filepath, version="4.0")
        config = GWMainFileReader().read(filepath)
        assert config.version == "4.0"

    def test_reads_v50_version(self, tmp_path: Path) -> None:
        filepath = tmp_path / "gw_main.dat"
        filepath.write_text(
            "C  GW Main\n"
            "# 5.0\n"
            # Minimum required fields to avoid EOF
            "  / BCFL\n"
            "  / TDFL\n"
            "  / PUMPFL\n"
            "  / SUBSFL\n"
            "  / OVRWRTFL\n"
            "1.0  / FACTLTOU\n"
            "FT\n"
            "1.0\n"
            "TAF\n"
            "1.0\n"
            "FT/DAY\n"
            "  / VELOUTFL\n"
            "  / VFLOWOUTFL\n"
            "  / GWALLOUTFL\n"
            "  / HTPOUTFL\n"
            "  / VTPOUTFL\n"
            "  / GWBUDFL\n"
            "  / ZBUDFL\n"
            "  / FNGWFL\n"
            "1  / ITECPLOTFLAG\n"
            "0  / KDEB\n"
            "0  / NOUTH\n"
            "1.0  / FACTXY\n"
            "  / GWHYDOUTFL\n"
            "0  / NOUTF\n"
            "  / FCHYDOUTFL\n"
        )
        config = GWMainFileReader().read(filepath)
        assert config.version == "5.0"


class TestGWMainFileReaderFilePaths:
    """Tests for reading sub-file paths."""

    def test_reads_file_paths(self, tmp_path: Path) -> None:
        filepath = tmp_path / "gw_main.dat"
        _write_gw_main(filepath)
        config = GWMainFileReader().read(filepath)
        assert config.bc_file is not None
        assert config.bc_file.name == "bc_file.dat"
        assert config.tile_drain_file is not None
        assert config.pumping_file is not None
        assert config.subsidence_file is not None

    def test_reads_output_factors(self, tmp_path: Path) -> None:
        filepath = tmp_path / "gw_main.dat"
        _write_gw_main(filepath)
        config = GWMainFileReader().read(filepath)
        assert config.head_output_factor == pytest.approx(1.0)
        assert config.head_output_unit == "FEET"
        assert config.volume_output_factor == pytest.approx(1.0)
        assert config.volume_output_unit == "TAF"


class TestGWMainFileReaderHydrographs:
    """Tests for reading hydrograph location data."""

    def test_reads_hydrograph_locations(self, tmp_path: Path) -> None:
        filepath = tmp_path / "gw_main.dat"
        _write_gw_main(filepath)
        config = GWMainFileReader().read(filepath)
        assert len(config.hydrograph_locations) == 2
        assert config.hydrograph_locations[0].node_id == 10
        assert config.hydrograph_locations[0].layer == 1
        assert config.hydrograph_locations[0].name == "Station_A"
        assert config.hydrograph_locations[1].node_id == 20
        assert config.hydrograph_locations[1].name == "Station_B"

    def test_zero_hydrographs(self, tmp_path: Path) -> None:
        filepath = tmp_path / "gw_main.dat"
        filepath.write_text(
            "C  GW Main\n"
            "# 4.0\n"
            "  / BCFL\n"
            "  / TDFL\n"
            "  / PUMPFL\n"
            "  / SUBSFL\n"
            "  / OVRWRTFL\n"
            "1.0\n"
            "FT\n"
            "1.0\n"
            "TAF\n"
            "1.0\n"
            "FT/DAY\n"
            "  / VELOUTFL\n"
            "  / VFLOWOUTFL\n"
            "  / GWALLOUTFL\n"
            "  / HTPOUTFL\n"
            "  / VTPOUTFL\n"
            "  / GWBUDFL\n"
            "  / ZBUDFL\n"
            "  / FNGWFL\n"
            "1  / ITECPLOTFLAG\n"
            "0  / KDEB\n"
            "0  / NOUTH\n"
            "1.0\n"
            "  / GWHYDOUTFL\n"
            "0  / NOUTF\n"
            "  / FCHYDOUTFL\n"
        )
        config = GWMainFileReader().read(filepath)
        assert len(config.hydrograph_locations) == 0


class TestGWMainFileReaderAquiferParams:
    """Tests for reading inline aquifer parameters (NGROUP=0)."""

    def test_reads_aquifer_parameters(self, tmp_path: Path) -> None:
        filepath = tmp_path / "gw_main.dat"
        _write_gw_main(filepath)
        config = GWMainFileReader().read(filepath)
        assert config.aquifer_params is not None
        assert config.aquifer_params.n_nodes == 2
        assert config.aquifer_params.n_layers == 1
        assert config.aquifer_params.kh[0, 0] == pytest.approx(10.0)
        assert config.aquifer_params.kh[1, 0] == pytest.approx(12.0)


class TestGWMainFileReaderInitialHeads:
    """Tests for reading initial heads section."""

    def test_reads_initial_heads(self, tmp_path: Path) -> None:
        filepath = tmp_path / "gw_main.dat"
        _write_gw_main(filepath)
        config = GWMainFileReader().read(filepath)
        assert config.initial_heads is not None
        assert config.initial_heads.shape == (2, 1)
        assert config.initial_heads[0, 0] == pytest.approx(50.0)
        assert config.initial_heads[1, 0] == pytest.approx(55.0)


# ===========================================================================
# Additional coverage tests
# ===========================================================================


from pyiwfm.components.groundwater import (  # noqa: E402
    AppGW,
    AquiferParameters,
    BoundaryCondition,
    Subsidence,
    TileDrain,
    Well,
)
from pyiwfm.io.groundwater import (  # noqa: E402
    FaceFlowSpec,
    GroundwaterWriter,
)

# ---------------------------------------------------------------------------
# GWMainFileReader._read_version edge cases
# ---------------------------------------------------------------------------


class TestReadVersionEdgeCases:
    """Tests for _read_version with missing or absent version header."""

    def test_no_version_header_returns_empty(self, tmp_path: Path) -> None:
        """If no '# X.X' line exists, version should be empty string."""
        filepath = tmp_path / "gw_main.dat"
        # File starts directly with data (no # version line)
        filepath.write_text(
            "C  GW Main\n"
            "C  No version header here\n"
            "bc_file.dat  / BCFL\n"
            "  / TDFL\n"
            "  / PUMPFL\n"
            "  / SUBSFL\n"
            "  / OVRWRTFL\n"
            "1.0  / FACTLTOU\n"
            "FT\n"
            "1.0\n"
            "TAF\n"
            "1.0\n"
            "FT/DAY\n"
            "  / VELOUTFL\n"
            "  / VFLOWOUTFL\n"
            "  / GWALLOUTFL\n"
            "  / HTPOUTFL\n"
            "  / VTPOUTFL\n"
            "  / GWBUDFL\n"
            "  / ZBUDFL\n"
            "  / FNGWFL\n"
            "1  / ITECPLOTFLAG\n"
            "0  / KDEB\n"
            "0  / NOUTH\n"
            "1.0\n"
            "  / GWHYDOUTFL\n"
            "0  / NOUTF\n"
            "  / FCHYDOUTFL\n"
        )
        config = GWMainFileReader().read(filepath)
        # When no version header is found, version is "" and
        # the first data line was consumed as the bc_file path
        assert config.version == ""


# ---------------------------------------------------------------------------
# GWMainFileReader._read_file_paths
# ---------------------------------------------------------------------------


class TestReadFilePathsDetailed:
    """Tests for reading BC, pumping, tile drain, subsidence paths."""

    def test_reads_bc_tile_drain_pumping_subsidence(self, tmp_path: Path) -> None:
        filepath = tmp_path / "gw_main.dat"
        _write_gw_main(filepath)
        config = GWMainFileReader().read(filepath)
        assert config.bc_file is not None
        assert config.bc_file.name == "bc_file.dat"
        assert config.tile_drain_file is not None
        assert config.tile_drain_file.name == "tile_drain.dat"
        assert config.pumping_file is not None
        assert config.pumping_file.name == "pumping.dat"
        assert config.subsidence_file is not None
        assert config.subsidence_file.name == "subsidence.dat"

    def test_empty_paths_become_none(self, tmp_path: Path) -> None:
        filepath = tmp_path / "gw_main.dat"
        filepath.write_text(
            "C  GW Main\n"
            "# 4.0\n"
            "  / BCFL\n"
            "  / TDFL\n"
            "  / PUMPFL\n"
            "  / SUBSFL\n"
            "  / OVRWRTFL\n"
            "1.0  / FACTLTOU\n"
            "FT\n"
            "1.0\n"
            "TAF\n"
            "1.0\n"
            "FT/DAY\n"
            "  / VELOUTFL\n"
            "  / VFLOWOUTFL\n"
            "  / GWALLOUTFL\n"
            "  / HTPOUTFL\n"
            "  / VTPOUTFL\n"
            "  / GWBUDFL\n"
            "  / ZBUDFL\n"
            "  / FNGWFL\n"
            "1  / ITECPLOTFLAG\n"
            "0  / KDEB\n"
            "0  / NOUTH\n"
            "1.0\n"
            "  / GWHYDOUTFL\n"
            "0  / NOUTF\n"
            "  / FCHYDOUTFL\n"
        )
        config = GWMainFileReader().read(filepath)
        assert config.bc_file is None
        assert config.tile_drain_file is None
        assert config.pumping_file is None
        assert config.subsidence_file is None
        assert config.overwrite_file is None

    def test_relative_path_resolved_to_base_dir(self, tmp_path: Path) -> None:
        """Relative paths are resolved against base_dir."""
        filepath = tmp_path / "gw_main.dat"
        filepath.write_text(
            "C  GW Main\n"
            "# 4.0\n"
            "subdir/bc_file.dat  / BCFL\n"
            "  / TDFL\n"
            "  / PUMPFL\n"
            "  / SUBSFL\n"
            "  / OVRWRTFL\n"
            "1.0  / FACTLTOU\n"
            "FT\n"
            "1.0\n"
            "TAF\n"
            "1.0\n"
            "FT/DAY\n"
            "  / VELOUTFL\n"
            "  / VFLOWOUTFL\n"
            "  / GWALLOUTFL\n"
            "  / HTPOUTFL\n"
            "  / VTPOUTFL\n"
            "  / GWBUDFL\n"
            "  / ZBUDFL\n"
            "  / FNGWFL\n"
            "1  / ITECPLOTFLAG\n"
            "0  / KDEB\n"
            "0  / NOUTH\n"
            "1.0\n"
            "  / GWHYDOUTFL\n"
            "0  / NOUTF\n"
            "  / FCHYDOUTFL\n"
        )
        custom_base = tmp_path / "model_base"
        custom_base.mkdir()
        config = GWMainFileReader().read(filepath, base_dir=custom_base)
        assert config.bc_file is not None
        assert config.bc_file.parent.name == "subdir"
        assert str(custom_base) in str(config.bc_file)


# ---------------------------------------------------------------------------
# GWMainFileReader._read_output_factors
# ---------------------------------------------------------------------------


class TestReadOutputFactors:
    """Tests for reading conversion factors and units."""

    def test_velocity_factor_and_unit(self, tmp_path: Path) -> None:
        filepath = tmp_path / "gw_main.dat"
        _write_gw_main(filepath)
        config = GWMainFileReader().read(filepath)
        assert config.velocity_output_factor == pytest.approx(1.0)
        assert config.velocity_output_unit == "FT/DAY"

    def test_debug_flag(self, tmp_path: Path) -> None:
        filepath = tmp_path / "gw_main.dat"
        _write_gw_main(filepath)
        config = GWMainFileReader().read(filepath)
        assert config.debug_flag == 0


# ---------------------------------------------------------------------------
# GWMainFileReader hydrograph locations — HYDTYP=0 (x-y coords)
# ---------------------------------------------------------------------------


class TestHydrographXYCoords:
    """Tests for hydrograph locations specified by x-y coordinates."""

    def test_hydtyp_0_xy_coords(self, tmp_path: Path) -> None:
        """HYDTYP=0 means x-y coordinates are provided."""
        filepath = tmp_path / "gw_main.dat"
        filepath.write_text(
            "C  GW Main\n"
            "# 4.0\n"
            "  / BCFL\n"
            "  / TDFL\n"
            "  / PUMPFL\n"
            "  / SUBSFL\n"
            "  / OVRWRTFL\n"
            "1.0  / FACTLTOU\n"
            "FT\n"
            "1.0\n"
            "TAF\n"
            "1.0\n"
            "FT/DAY\n"
            "  / VELOUTFL\n"
            "  / VFLOWOUTFL\n"
            "  / GWALLOUTFL\n"
            "  / HTPOUTFL\n"
            "  / VTPOUTFL\n"
            "  / GWBUDFL\n"
            "  / ZBUDFL\n"
            "  / FNGWFL\n"
            "1  / ITECPLOTFLAG\n"
            "0  / KDEB\n"
            "2  / NOUTH\n"
            "2.0  / FACTXY\n"
            "hydout.dat  / GWHYDOUTFL\n"
            "1  0  1  1000.0  2000.0  OBS_Point_A\n"
            "2  0  2  1500.0  2500.0  OBS_Point_B\n"
            "0  / NOUTF\n"
            "  / FCHYDOUTFL\n"
        )
        config = GWMainFileReader().read(filepath)
        assert len(config.hydrograph_locations) == 2
        # HYDTYP=0: x-y coords are multiplied by FACTXY (2.0)
        loc0 = config.hydrograph_locations[0]
        assert loc0.x == pytest.approx(2000.0)  # 1000 * 2.0
        assert loc0.y == pytest.approx(4000.0)  # 2000 * 2.0
        assert loc0.layer == 1
        assert loc0.node_id == 0  # node_id=0 for HYDTYP=0
        assert loc0.name == "OBS_Point_A"

        loc1 = config.hydrograph_locations[1]
        assert loc1.x == pytest.approx(3000.0)  # 1500 * 2.0
        assert loc1.y == pytest.approx(5000.0)  # 2500 * 2.0
        assert loc1.name == "OBS_Point_B"

    def test_hydtyp_1_node_number(self, tmp_path: Path) -> None:
        """HYDTYP=1 means node number is provided (no x-y)."""
        filepath = tmp_path / "gw_main.dat"
        _write_gw_main(filepath)
        config = GWMainFileReader().read(filepath)
        # Already node-based in _write_gw_main; confirm x=0, y=0
        loc0 = config.hydrograph_locations[0]
        assert loc0.x == pytest.approx(0.0)
        assert loc0.y == pytest.approx(0.0)
        assert loc0.node_id == 10


# ---------------------------------------------------------------------------
# GWMainFileReader._read_face_flow_specs
# ---------------------------------------------------------------------------


class TestReadFaceFlowSpecs:
    """Tests for reading element face flow data."""

    def test_reads_face_flow_specs(self, tmp_path: Path) -> None:
        filepath = tmp_path / "gw_main.dat"
        filepath.write_text(
            "C  GW Main\n"
            "# 4.0\n"
            "  / BCFL\n"
            "  / TDFL\n"
            "  / PUMPFL\n"
            "  / SUBSFL\n"
            "  / OVRWRTFL\n"
            "1.0  / FACTLTOU\n"
            "FT\n"
            "1.0\n"
            "TAF\n"
            "1.0\n"
            "FT/DAY\n"
            "  / VELOUTFL\n"
            "  / VFLOWOUTFL\n"
            "  / GWALLOUTFL\n"
            "  / HTPOUTFL\n"
            "  / VTPOUTFL\n"
            "  / GWBUDFL\n"
            "  / ZBUDFL\n"
            "  / FNGWFL\n"
            "1  / ITECPLOTFLAG\n"
            "0  / KDEB\n"
            "0  / NOUTH\n"
            "1.0\n"
            "  / GWHYDOUTFL\n"
            "2  / NOUTF\n"
            "face_flow.dat  / FCHYDOUTFL\n"
            "1  1  10  20  FaceFlow_A\n"
            "2  2  30  40  FaceFlow_B\n"
        )
        config = GWMainFileReader().read(filepath)
        assert config.n_face_flow_outputs == 2
        assert config.face_flow_output_file is not None
        assert config.face_flow_output_file.name == "face_flow.dat"
        assert len(config.face_flow_specs) == 2

        spec0 = config.face_flow_specs[0]
        assert spec0.id == 1
        assert spec0.layer == 1
        assert spec0.node_a == 10
        assert spec0.node_b == 20
        assert spec0.name == "FaceFlow_A"

        spec1 = config.face_flow_specs[1]
        assert spec1.id == 2
        assert spec1.layer == 2
        assert spec1.node_a == 30
        assert spec1.node_b == 40
        assert spec1.name == "FaceFlow_B"

    def test_face_flow_with_comments(self, tmp_path: Path) -> None:
        """Face flow specs with interspersed comment lines."""
        filepath = tmp_path / "gw_main.dat"
        filepath.write_text(
            "C  GW Main\n"
            "# 4.0\n"
            "  / BCFL\n"
            "  / TDFL\n"
            "  / PUMPFL\n"
            "  / SUBSFL\n"
            "  / OVRWRTFL\n"
            "1.0  / FACTLTOU\n"
            "FT\n"
            "1.0\n"
            "TAF\n"
            "1.0\n"
            "FT/DAY\n"
            "  / VELOUTFL\n"
            "  / VFLOWOUTFL\n"
            "  / GWALLOUTFL\n"
            "  / HTPOUTFL\n"
            "  / VTPOUTFL\n"
            "  / GWBUDFL\n"
            "  / ZBUDFL\n"
            "  / FNGWFL\n"
            "1  / ITECPLOTFLAG\n"
            "0  / KDEB\n"
            "0  / NOUTH\n"
            "1.0\n"
            "  / GWHYDOUTFL\n"
            "1  / NOUTF\n"
            "  / FCHYDOUTFL\n"
            "C  The face flow spec below\n"
            "1  1  5  6  Single_Face\n"
        )
        config = GWMainFileReader().read(filepath)
        assert len(config.face_flow_specs) == 1
        assert config.face_flow_specs[0].name == "Single_Face"


# ---------------------------------------------------------------------------
# GWMainFileReader._read_aquifer_parameters with parametric grids (NGROUP > 0)
# ---------------------------------------------------------------------------


class TestReadAquiferParamsParametricGrid:
    """Tests for aquifer params with NGROUP > 0 (parametric grids)."""

    def test_parametric_grid_single_group(self, tmp_path: Path) -> None:
        """NGROUP=1 triggers parametric grid parsing."""
        filepath = tmp_path / "gw_main.dat"
        filepath.write_text(
            "C  GW Main\n"
            "# 4.0\n"
            "  / BCFL\n"
            "  / TDFL\n"
            "  / PUMPFL\n"
            "  / SUBSFL\n"
            "  / OVRWRTFL\n"
            "1.0  / FACTLTOU\n"
            "FT\n"
            "1.0\n"
            "TAF\n"
            "1.0\n"
            "FT/DAY\n"
            "  / VELOUTFL\n"
            "  / VFLOWOUTFL\n"
            "  / GWALLOUTFL\n"
            "  / HTPOUTFL\n"
            "  / VTPOUTFL\n"
            "  / GWBUDFL\n"
            "  / ZBUDFL\n"
            "  / FNGWFL\n"
            "1  / ITECPLOTFLAG\n"
            "0  / KDEB\n"
            "0  / NOUTH\n"
            "1.0\n"
            "  / GWHYDOUTFL\n"
            "0  / NOUTF\n"
            "  / FCHYDOUTFL\n"
            # Aquifer parameters section with NGROUP=1
            "1  / NGROUP\n"
            "1.0 1.0 1.0 1.0 1.0 1.0  / FX FKH FS FN FV FL\n"
            "1DAY  / TUNITKH\n"
            "1DAY  / TUNITV\n"
            "1DAY  / TUNITL\n"
            # Parametric grid: node_range_str, NDP, NEP as separate lines
            "1-3  / node range\n"
            "3  / NDP\n"
            "1  / NEP\n"
            # Element definition: ElemID Node1 Node2 Node3 (1-based)
            "1  1  2  3\n"
            # Node data: NodeID X Y Kh Ss Sy AquitardKv Kv (5 params, 1 layer)
            "1  0.0  0.0  10.0  0.001  0.15  0.01  5.0\n"
            "2  100.0  0.0  12.0  0.002  0.18  0.02  6.0\n"
            "3  50.0  100.0  11.0  0.0015  0.16  0.015  5.5\n"
            # End of parametric node data (comment triggers end)
            "C  End of aquifer params\n"
            # Kh anomaly section (NEBK=0 still reads FACT and TUNITH)
            "0  / NEBK\n"
            "1.0  / FACT\n"
            "1DAY  / TUNITH\n"
            # Return flow flag
            "0  / IFLAGRF\n"
            # Initial heads
            "1.0  / FACTHP\n"
            "1  50.0\n"
            "2  55.0\n"
            "3  60.0\n"
        )
        config = GWMainFileReader().read(filepath)
        # Parametric grid mode => aquifer_params is None, data goes to
        # config.parametric_grids
        assert config.aquifer_params is None
        assert len(config.parametric_grids) == 1

        grid = config.parametric_grids[0]
        assert grid.n_nodes == 3
        assert grid.n_elements == 1
        assert len(grid.elements) == 1
        # Node indices are 0-based (file had 1,2,3 → 0,1,2)
        assert grid.elements[0] == (0, 1, 2)
        assert grid.node_coords.shape == (3, 2)
        assert grid.node_coords[0, 0] == pytest.approx(0.0)
        assert grid.node_coords[1, 0] == pytest.approx(100.0)
        assert grid.node_values.shape == (3, 1, 5)
        # First node, layer 0, Kh (param 0) = 10.0
        assert grid.node_values[0, 0, 0] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# GWMainFileReader._read_aquifer_parameters — multi-layer per-node
# ---------------------------------------------------------------------------


class TestReadAquiferParamsMultiLayer:
    """Tests for per-node aquifer params with multiple layers."""

    def test_two_layers_continuation_lines(self, tmp_path: Path) -> None:
        """Multi-layer per-node data uses continuation lines (5 fields)."""
        filepath = tmp_path / "gw_main.dat"
        filepath.write_text(
            "C  GW Main\n"
            "# 4.0\n"
            "  / BCFL\n"
            "  / TDFL\n"
            "  / PUMPFL\n"
            "  / SUBSFL\n"
            "  / OVRWRTFL\n"
            "1.0  / FACTLTOU\n"
            "FT\n"
            "1.0\n"
            "TAF\n"
            "1.0\n"
            "FT/DAY\n"
            "  / VELOUTFL\n"
            "  / VFLOWOUTFL\n"
            "  / GWALLOUTFL\n"
            "  / HTPOUTFL\n"
            "  / VTPOUTFL\n"
            "  / GWBUDFL\n"
            "  / ZBUDFL\n"
            "  / FNGWFL\n"
            "1  / ITECPLOTFLAG\n"
            "0  / KDEB\n"
            "0  / NOUTH\n"
            "1.0\n"
            "  / GWHYDOUTFL\n"
            "0  / NOUTF\n"
            "  / FCHYDOUTFL\n"
            # Aquifer parameters (NGROUP=0 per-node, factors all 1.0)
            "0  / NGROUP\n"
            "1.0 1.0 1.0 1.0 1.0 1.0  / FX FKH FS FN FV FL\n"
            "1DAY  / TUNITKH\n"
            "1DAY  / TUNITV\n"
            "1DAY  / TUNITL\n"
            # Node 1 (layer 1): ID PKH PS PN PV PL
            "1  10.0  0.001  0.15  0.01  5.0\n"
            # Node 1 (layer 2 -- continuation, 5 fields only)
            "20.0  0.002  0.25  0.02  8.0\n"
            # Node 2 (layer 1)
            "2  12.0  0.003  0.18  0.03  6.0\n"
            # Node 2 (layer 2 -- continuation)
            "22.0  0.004  0.28  0.04  9.0\n"
            # End marker
            "C  End of aquifer parameters\n"
            # Kh anomaly (NEBK=0 still reads FACT and TUNITH)
            "0  / NEBK\n"
            "1.0  / FACT\n"
            "1DAY  / TUNITH\n"
            # Return flow flag
            "0  / IFLAGRF\n"
            # Initial heads (2 nodes, 2 layers)
            "1.0  / FACTHP\n"
            "1  50.0  40.0\n"
            "2  55.0  45.0\n"
        )
        config = GWMainFileReader().read(filepath)
        assert config.aquifer_params is not None
        assert config.aquifer_params.n_nodes == 2
        assert config.aquifer_params.n_layers == 2
        # Node 1, layer 1 Kh
        assert config.aquifer_params.kh[0, 0] == pytest.approx(10.0)
        # Node 1, layer 2 Kh (continuation line)
        assert config.aquifer_params.kh[0, 1] == pytest.approx(20.0)
        # Node 2, layer 1 Kh
        assert config.aquifer_params.kh[1, 0] == pytest.approx(12.0)
        # Node 2, layer 2 Kh (continuation)
        assert config.aquifer_params.kh[1, 1] == pytest.approx(22.0)

        # Check specific_yield (3rd param, 'PN')
        assert config.aquifer_params.specific_yield[0, 0] == pytest.approx(0.15)
        assert config.aquifer_params.specific_yield[0, 1] == pytest.approx(0.25)

    def test_aquifer_params_with_conversion_factors(self, tmp_path: Path) -> None:
        """Conversion factors (FKH, FS, FN, FV, FL) are applied to values."""
        filepath = tmp_path / "gw_main.dat"
        filepath.write_text(
            "C  GW Main\n"
            "# 4.0\n"
            "  / BCFL\n"
            "  / TDFL\n"
            "  / PUMPFL\n"
            "  / SUBSFL\n"
            "  / OVRWRTFL\n"
            "1.0  / FACTLTOU\n"
            "FT\n"
            "1.0\n"
            "TAF\n"
            "1.0\n"
            "FT/DAY\n"
            "  / VELOUTFL\n"
            "  / VFLOWOUTFL\n"
            "  / GWALLOUTFL\n"
            "  / HTPOUTFL\n"
            "  / VTPOUTFL\n"
            "  / GWBUDFL\n"
            "  / ZBUDFL\n"
            "  / FNGWFL\n"
            "1  / ITECPLOTFLAG\n"
            "0  / KDEB\n"
            "0  / NOUTH\n"
            "1.0\n"
            "  / GWHYDOUTFL\n"
            "0  / NOUTF\n"
            "  / FCHYDOUTFL\n"
            # Factors: FX=1.0, FKH=2.0, FS=3.0, FN=4.0, FV=5.0, FL=6.0
            "0  / NGROUP\n"
            "1.0 2.0 3.0 4.0 5.0 6.0  / FX FKH FS FN FV FL\n"
            "1DAY\n"
            "1DAY\n"
            "1DAY\n"
            "1  10.0  0.001  0.15  0.01  5.0\n"
            "C  End\n"
            # Kh anomaly (NEBK=0 still reads FACT and TUNITH)
            "0  / NEBK\n"
            "1.0  / FACT\n"
            "1DAY  / TUNITH\n"
            # Return flow flag
            "0  / IFLAGRF\n"
            "1.0  / FACTHP\n"
            "1  100.0\n"
        )
        config = GWMainFileReader().read(filepath)
        assert config.aquifer_params is not None
        # kh = 10.0 * FKH(2.0) = 20.0
        assert config.aquifer_params.kh[0, 0] == pytest.approx(20.0)
        # specific_storage = 0.001 * FS(3.0) = 0.003
        assert config.aquifer_params.specific_storage[0, 0] == pytest.approx(0.003)
        # specific_yield = 0.15 * FN(4.0) = 0.60
        assert config.aquifer_params.specific_yield[0, 0] == pytest.approx(0.60)
        # aquitard_kv = 0.01 * FV(5.0) = 0.05
        assert config.aquifer_params.aquitard_kv[0, 0] == pytest.approx(0.05)
        # kv = 5.0 * FL(6.0) = 30.0
        assert config.aquifer_params.kv[0, 0] == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# GWMainFileReader._read_kh_anomalies
# ---------------------------------------------------------------------------


class TestReadKhAnomalies:
    """Tests for reading Kh anomaly entries."""

    def test_kh_anomalies_with_entries(self, tmp_path: Path) -> None:
        """NEBK > 0: read anomaly data with conversion factor."""
        filepath = tmp_path / "gw_main.dat"
        filepath.write_text(
            "C  GW Main\n"
            "# 4.0\n"
            "  / BCFL\n"
            "  / TDFL\n"
            "  / PUMPFL\n"
            "  / SUBSFL\n"
            "  / OVRWRTFL\n"
            "1.0  / FACTLTOU\n"
            "FT\n"
            "1.0\n"
            "TAF\n"
            "1.0\n"
            "FT/DAY\n"
            "  / VELOUTFL\n"
            "  / VFLOWOUTFL\n"
            "  / GWALLOUTFL\n"
            "  / HTPOUTFL\n"
            "  / VTPOUTFL\n"
            "  / GWBUDFL\n"
            "  / ZBUDFL\n"
            "  / FNGWFL\n"
            "1  / ITECPLOTFLAG\n"
            "0  / KDEB\n"
            "0  / NOUTH\n"
            "1.0\n"
            "  / GWHYDOUTFL\n"
            "0  / NOUTF\n"
            "  / FCHYDOUTFL\n"
            # Aquifer parameters (1 node, 1 layer)
            "0  / NGROUP\n"
            "1.0 1.0 1.0 1.0 1.0 1.0\n"
            "1DAY\n"
            "1DAY\n"
            "1DAY\n"
            "1  10.0  0.001  0.15  0.01  5.0\n"
            "C  End aquifer params\n"
            # Kh anomaly section: NEBK=2
            "2  / NEBK\n"
            "2.0  / FACT\n"
            "1DAY  / TUNITH\n"
            # IC  IEBK  BK[layer1]
            "1  5  0.5\n"
            "2  10  0.8\n"
            # Return flow flag
            "0  / IFLAGRF\n"
            # Initial heads
            "1.0  / FACTHP\n"
            "1  50.0\n"
        )
        config = GWMainFileReader().read(filepath)
        assert len(config.kh_anomalies) == 2
        # First entry: element 5, BK = 0.5 * FACT(2.0) = 1.0
        assert config.kh_anomalies[0].element_id == 5
        assert config.kh_anomalies[0].kh_per_layer == [pytest.approx(1.0)]
        # Second entry: element 10, BK = 0.8 * FACT(2.0) = 1.6
        assert config.kh_anomalies[1].element_id == 10
        assert config.kh_anomalies[1].kh_per_layer == [pytest.approx(1.6)]

    def test_kh_anomalies_zero_nebk(self, tmp_path: Path) -> None:
        """NEBK=0 means no anomalies."""
        filepath = tmp_path / "gw_main.dat"
        _write_gw_main(filepath)
        config = GWMainFileReader().read(filepath)
        assert config.kh_anomalies == []


# ---------------------------------------------------------------------------
# GWMainFileReader._read_initial_heads
# ---------------------------------------------------------------------------


class TestReadInitialHeadsFromMain:
    """Tests for initial heads reading from the main file."""

    def test_initial_heads_conversion_factor(self, tmp_path: Path) -> None:
        """FACTHP conversion factor is applied to head values."""
        filepath = tmp_path / "gw_main.dat"
        filepath.write_text(
            "C  GW Main\n"
            "# 4.0\n"
            "  / BCFL\n"
            "  / TDFL\n"
            "  / PUMPFL\n"
            "  / SUBSFL\n"
            "  / OVRWRTFL\n"
            "1.0  / FACTLTOU\n"
            "FT\n"
            "1.0\n"
            "TAF\n"
            "1.0\n"
            "FT/DAY\n"
            "  / VELOUTFL\n"
            "  / VFLOWOUTFL\n"
            "  / GWALLOUTFL\n"
            "  / HTPOUTFL\n"
            "  / VTPOUTFL\n"
            "  / GWBUDFL\n"
            "  / ZBUDFL\n"
            "  / FNGWFL\n"
            "1  / ITECPLOTFLAG\n"
            "0  / KDEB\n"
            "0  / NOUTH\n"
            "1.0\n"
            "  / GWHYDOUTFL\n"
            "0  / NOUTF\n"
            "  / FCHYDOUTFL\n"
            # Aquifer params (1 node, 1 layer)
            "0  / NGROUP\n"
            "1.0 1.0 1.0 1.0 1.0 1.0\n"
            "1DAY\n"
            "1DAY\n"
            "1DAY\n"
            "1  10.0  0.001  0.15  0.01  5.0\n"
            "C  End\n"
            # Anomaly (NEBK=0 still reads FACT and TUNITH)
            "0  / NEBK\n"
            "1.0  / FACT\n"
            "1DAY  / TUNITH\n"
            # Return flow flag
            "0  / IFLAGRF\n"
            # Initial heads with FACTHP = 0.3048 (feet to meters)
            "0.3048  / FACTHP\n"
            "1  100.0\n"
        )
        config = GWMainFileReader().read(filepath)
        assert config.initial_heads is not None
        assert config.initial_heads.shape == (1, 1)
        # 100.0 * 0.3048 = 30.48
        assert config.initial_heads[0, 0] == pytest.approx(30.48)

    def test_initial_heads_multi_layer(self, tmp_path: Path) -> None:
        """Initial heads with two layers."""
        filepath = tmp_path / "gw_main.dat"
        filepath.write_text(
            "C  GW Main\n"
            "# 4.0\n"
            "  / BCFL\n"
            "  / TDFL\n"
            "  / PUMPFL\n"
            "  / SUBSFL\n"
            "  / OVRWRTFL\n"
            "1.0  / FACTLTOU\n"
            "FT\n"
            "1.0\n"
            "TAF\n"
            "1.0\n"
            "FT/DAY\n"
            "  / VELOUTFL\n"
            "  / VFLOWOUTFL\n"
            "  / GWALLOUTFL\n"
            "  / HTPOUTFL\n"
            "  / VTPOUTFL\n"
            "  / GWBUDFL\n"
            "  / ZBUDFL\n"
            "  / FNGWFL\n"
            "1  / ITECPLOTFLAG\n"
            "0  / KDEB\n"
            "0  / NOUTH\n"
            "1.0\n"
            "  / GWHYDOUTFL\n"
            "0  / NOUTF\n"
            "  / FCHYDOUTFL\n"
            "0  / NGROUP\n"
            "1.0 1.0 1.0 1.0 1.0 1.0\n"
            "1DAY\n"
            "1DAY\n"
            "1DAY\n"
            "1  10.0  0.001  0.15  0.01  5.0\n"
            "20.0  0.002  0.25  0.02  8.0\n"
            "2  12.0  0.003  0.18  0.03  6.0\n"
            "22.0  0.004  0.28  0.04  9.0\n"
            "C  End\n"
            # Kh anomaly (NEBK=0 still reads FACT and TUNITH)
            "0  / NEBK\n"
            "1.0  / FACT\n"
            "1DAY  / TUNITH\n"
            # Return flow flag
            "0  / IFLAGRF\n"
            "1.0  / FACTHP\n"
            "1  50.0  40.0\n"
            "2  55.0  45.0\n"
        )
        config = GWMainFileReader().read(filepath)
        assert config.initial_heads is not None
        assert config.initial_heads.shape == (2, 2)
        assert config.initial_heads[0, 0] == pytest.approx(50.0)
        assert config.initial_heads[0, 1] == pytest.approx(40.0)
        assert config.initial_heads[1, 0] == pytest.approx(55.0)
        assert config.initial_heads[1, 1] == pytest.approx(45.0)


# ---------------------------------------------------------------------------
# GWMainFileReader.read() — full file with all sections
# ---------------------------------------------------------------------------


class TestGWMainFileReaderFullFile:
    """Test reading a complete main file end-to-end."""

    def test_full_file_all_sections(self, tmp_path: Path) -> None:
        """Read a full main file with all sections populated."""
        filepath = tmp_path / "gw_main.dat"
        _write_gw_main(filepath)
        config = GWMainFileReader().read(filepath)
        # Version
        assert config.version == "4.0"
        # File paths
        assert config.bc_file is not None
        assert config.tile_drain_file is not None
        assert config.pumping_file is not None
        assert config.subsidence_file is not None
        # Conversion factors
        assert config.head_output_factor == pytest.approx(1.0)
        assert config.head_output_unit == "FEET"
        assert config.volume_output_factor == pytest.approx(1.0)
        assert config.volume_output_unit == "TAF"
        # Debug flag
        assert config.debug_flag == 0
        # Hydrographs
        assert len(config.hydrograph_locations) == 2
        assert config.hydrograph_output_file is not None
        # Face flow
        assert config.n_face_flow_outputs == 0
        assert config.face_flow_specs == []
        # Aquifer params
        assert config.aquifer_params is not None
        assert config.aquifer_params.n_nodes == 2
        assert config.aquifer_params.n_layers == 1
        # Anomalies
        assert config.kh_anomalies == []
        # Initial heads
        assert config.initial_heads is not None
        assert config.initial_heads.shape == (2, 1)

    def test_read_with_explicit_base_dir(self, tmp_path: Path) -> None:
        """base_dir parameter overrides file parent directory."""
        filepath = tmp_path / "gw_main.dat"
        _write_gw_main(filepath)
        custom_base = tmp_path / "custom_base"
        custom_base.mkdir()
        config = GWMainFileReader().read(filepath, base_dir=custom_base)
        # BC file should be resolved relative to custom_base
        assert config.bc_file is not None
        assert config.bc_file.parent == custom_base


# ---------------------------------------------------------------------------
# GroundwaterWriter — write_wells
# ---------------------------------------------------------------------------


def _make_gw_component(
    n_nodes: int = 3,
    n_layers: int = 1,
    n_elements: int = 2,
) -> AppGW:
    """Create a minimal AppGW for writer tests."""
    return AppGW(n_nodes=n_nodes, n_layers=n_layers, n_elements=n_elements)


class TestWriteWells:
    """Tests for GroundwaterWriter.write_wells."""

    def test_write_wells_basic(self, tmp_path: Path) -> None:
        config = GWFileConfig(output_dir=tmp_path)
        writer = GroundwaterWriter(config)
        gw = _make_gw_component()
        gw.wells = {
            1: Well(
                id=1,
                x=100.0,
                y=200.0,
                element=1,
                top_screen=50.0,
                bottom_screen=10.0,
                max_pump_rate=500.0,
                name="Well_A",
            ),
            2: Well(
                id=2,
                x=300.0,
                y=400.0,
                element=2,
                top_screen=60.0,
                bottom_screen=20.0,
                max_pump_rate=750.0,
                name="Well_B",
            ),
        }
        result = writer.write_wells(gw)
        assert result.exists()
        content = result.read_text()
        assert "NWELLS" in content
        assert "Well_A" in content
        assert "Well_B" in content

    def test_write_wells_custom_header(self, tmp_path: Path) -> None:
        config = GWFileConfig(output_dir=tmp_path)
        writer = GroundwaterWriter(config)
        gw = _make_gw_component()
        gw.wells = {
            1: Well(
                id=1,
                x=100.0,
                y=200.0,
                element=1,
                top_screen=50.0,
                bottom_screen=10.0,
                max_pump_rate=500.0,
                name="W1",
            ),
        }
        result = writer.write_wells(gw, header="My Custom Header")
        content = result.read_text()
        assert "My Custom Header" in content

    def test_write_wells_roundtrip(self, tmp_path: Path) -> None:
        """Write wells then read them back and verify data integrity."""
        config = GWFileConfig(output_dir=tmp_path)
        writer = GroundwaterWriter(config)
        gw = _make_gw_component()
        gw.wells = {
            1: Well(
                id=1,
                x=1000.0,
                y=2000.0,
                element=1,
                top_screen=50.0,
                bottom_screen=10.0,
                max_pump_rate=500.0,
                name="Well_1",
            ),
        }
        filepath = writer.write_wells(gw)

        reader = GroundwaterReader()
        wells = reader.read_wells(filepath)
        assert len(wells) == 1
        assert wells[1].x == pytest.approx(1000.0)
        assert wells[1].y == pytest.approx(2000.0)
        assert wells[1].name == "Well_1"


# ---------------------------------------------------------------------------
# GroundwaterWriter — write_aquifer_params
# ---------------------------------------------------------------------------


class TestWriteAquiferParams:
    """Tests for GroundwaterWriter.write_aquifer_params."""

    def test_write_aquifer_params_basic(self, tmp_path: Path) -> None:
        config = GWFileConfig(output_dir=tmp_path)
        writer = GroundwaterWriter(config)
        gw = _make_gw_component(n_nodes=2, n_layers=1)
        gw.aquifer_params = AquiferParameters(
            n_nodes=2,
            n_layers=1,
            kh=np.array([[10.0], [12.0]]),
            kv=np.array([[5.0], [6.0]]),
            specific_storage=np.array([[0.001], [0.002]]),
            specific_yield=np.array([[0.15], [0.18]]),
        )
        result = writer.write_aquifer_params(gw)
        assert result.exists()
        content = result.read_text()
        assert "NNODES" in content
        assert "NLAYERS" in content

    def test_write_aquifer_params_no_params_raises(self, tmp_path: Path) -> None:
        config = GWFileConfig(output_dir=tmp_path)
        writer = GroundwaterWriter(config)
        gw = _make_gw_component()
        gw.aquifer_params = None
        with pytest.raises(ValueError, match="No aquifer parameters"):
            writer.write_aquifer_params(gw)

    def test_write_aquifer_params_multi_layer(self, tmp_path: Path) -> None:
        config = GWFileConfig(output_dir=tmp_path)
        writer = GroundwaterWriter(config)
        gw = _make_gw_component(n_nodes=2, n_layers=2)
        gw.aquifer_params = AquiferParameters(
            n_nodes=2,
            n_layers=2,
            kh=np.array([[10.0, 20.0], [12.0, 22.0]]),
            kv=np.array([[5.0, 8.0], [6.0, 9.0]]),
            specific_storage=np.array([[0.001, 0.002], [0.003, 0.004]]),
            specific_yield=np.array([[0.15, 0.25], [0.18, 0.28]]),
        )
        result = writer.write_aquifer_params(gw)
        content = result.read_text()
        # Both layers should be present for each node
        lines = [ln for ln in content.split("\n") if ln.strip() and not ln.startswith("C")]
        # Should contain NNODES line, NLAYERS line, then data lines
        assert any("2" in ln and "NNODES" in ln for ln in lines)


# ---------------------------------------------------------------------------
# GroundwaterWriter — write_boundary_conditions
# ---------------------------------------------------------------------------


class TestWriteBoundaryConditions:
    """Tests for GroundwaterWriter.write_boundary_conditions."""

    def test_write_boundary_conditions(self, tmp_path: Path) -> None:
        config = GWFileConfig(output_dir=tmp_path)
        writer = GroundwaterWriter(config)
        gw = _make_gw_component()
        gw.boundary_conditions = [
            BoundaryCondition(
                id=1,
                bc_type="specified_head",
                nodes=[1, 2],
                values=[100.0, 105.0],
                layer=1,
            ),
            BoundaryCondition(
                id=2,
                bc_type="specified_flow",
                nodes=[3],
                values=[-50.0],
                layer=1,
            ),
            BoundaryCondition(
                id=3,
                bc_type="general_head",
                nodes=[1],
                values=[110.0],
                layer=1,
                conductance=[0.01],
            ),
        ]
        result = writer.write_boundary_conditions(gw)
        assert result.exists()
        content = result.read_text()
        assert "SPECIFIED HEAD" in content
        assert "SPECIFIED FLOW" in content
        assert "GENERAL HEAD" in content
        assert "N_SPEC_HEAD_BC" in content
        assert "N_SPEC_FLOW_BC" in content
        assert "N_GEN_HEAD_BC" in content

    def test_write_boundary_conditions_custom_header(self, tmp_path: Path) -> None:
        config = GWFileConfig(output_dir=tmp_path)
        writer = GroundwaterWriter(config)
        gw = _make_gw_component()
        gw.boundary_conditions = []
        result = writer.write_boundary_conditions(gw, header="Custom BC Header")
        content = result.read_text()
        assert "Custom BC Header" in content


# ---------------------------------------------------------------------------
# GroundwaterWriter — write_tile_drains
# ---------------------------------------------------------------------------


class TestWriteTileDrains:
    """Tests for GroundwaterWriter.write_tile_drains."""

    def test_write_tile_drains(self, tmp_path: Path) -> None:
        config = GWFileConfig(output_dir=tmp_path)
        writer = GroundwaterWriter(config)
        gw = _make_gw_component()
        gw.tile_drains = {
            1: TileDrain(
                id=1,
                element=1,
                elevation=50.0,
                conductance=0.01,
                destination_type="stream",
                destination_id=5,
            ),
            2: TileDrain(
                id=2,
                element=2,
                elevation=45.0,
                conductance=0.02,
                destination_type="outside",
                destination_id=None,
            ),
        }
        result = writer.write_tile_drains(gw)
        assert result.exists()
        content = result.read_text()
        assert "NDRAINS" in content
        assert "stream" in content
        assert "outside" in content

    def test_write_tile_drains_custom_header(self, tmp_path: Path) -> None:
        config = GWFileConfig(output_dir=tmp_path)
        writer = GroundwaterWriter(config)
        gw = _make_gw_component()
        gw.tile_drains = {
            1: TileDrain(id=1, element=1, elevation=50.0, conductance=0.01),
        }
        result = writer.write_tile_drains(gw, header="My Tile Drains")
        content = result.read_text()
        assert "My Tile Drains" in content


# ---------------------------------------------------------------------------
# GroundwaterWriter — write_subsidence
# ---------------------------------------------------------------------------


class TestWriteSubsidence:
    """Tests for GroundwaterWriter.write_subsidence."""

    def test_write_subsidence(self, tmp_path: Path) -> None:
        config = GWFileConfig(output_dir=tmp_path)
        writer = GroundwaterWriter(config)
        gw = _make_gw_component()
        gw.subsidence = [
            Subsidence(
                element=1,
                layer=1,
                elastic_storage=1e-5,
                inelastic_storage=1e-4,
                preconsolidation_head=80.0,
            ),
            Subsidence(
                element=2,
                layer=1,
                elastic_storage=2e-5,
                inelastic_storage=2e-4,
                preconsolidation_head=75.0,
            ),
        ]
        result = writer.write_subsidence(gw)
        assert result.exists()
        content = result.read_text()
        assert "N_SUBSIDENCE" in content

    def test_write_subsidence_custom_header(self, tmp_path: Path) -> None:
        config = GWFileConfig(output_dir=tmp_path)
        writer = GroundwaterWriter(config)
        gw = _make_gw_component()
        gw.subsidence = [
            Subsidence(
                element=1,
                layer=1,
                elastic_storage=1e-5,
                inelastic_storage=1e-4,
                preconsolidation_head=80.0,
            ),
        ]
        result = writer.write_subsidence(gw, header="Subsidence Data")
        content = result.read_text()
        assert "Subsidence Data" in content


# ---------------------------------------------------------------------------
# GroundwaterWriter — write_initial_heads
# ---------------------------------------------------------------------------


class TestWriteInitialHeads:
    """Tests for GroundwaterWriter.write_initial_heads."""

    def test_write_initial_heads(self, tmp_path: Path) -> None:
        config = GWFileConfig(output_dir=tmp_path)
        writer = GroundwaterWriter(config)
        gw = _make_gw_component(n_nodes=3, n_layers=2)
        gw.heads = np.array([[100.0, 90.0], [105.0, 95.0], [110.0, 100.0]])
        result = writer.write_initial_heads(gw)
        assert result.exists()
        content = result.read_text()
        assert "NNODES" in content
        assert "NLAYERS" in content
        assert "HEAD_L01" in content
        assert "HEAD_L02" in content

    def test_write_initial_heads_no_heads_raises(self, tmp_path: Path) -> None:
        config = GWFileConfig(output_dir=tmp_path)
        writer = GroundwaterWriter(config)
        gw = _make_gw_component()
        gw.heads = None
        with pytest.raises(ValueError, match="No initial heads"):
            writer.write_initial_heads(gw)

    def test_write_initial_heads_custom_header(self, tmp_path: Path) -> None:
        config = GWFileConfig(output_dir=tmp_path)
        writer = GroundwaterWriter(config)
        gw = _make_gw_component(n_nodes=2, n_layers=1)
        gw.heads = np.array([[50.0], [55.0]])
        result = writer.write_initial_heads(gw, header="Custom Head Header")
        content = result.read_text()
        assert "Custom Head Header" in content

    def test_write_initial_heads_roundtrip(self, tmp_path: Path) -> None:
        """Write then read back to verify roundtrip integrity."""
        config = GWFileConfig(output_dir=tmp_path)
        writer = GroundwaterWriter(config)
        gw = _make_gw_component(n_nodes=3, n_layers=2)
        gw.heads = np.array([[100.0, 90.0], [105.0, 95.0], [110.0, 100.0]])
        filepath = writer.write_initial_heads(gw)

        reader = GroundwaterReader()
        n_nodes, n_layers, heads = reader.read_initial_heads(filepath)
        assert n_nodes == 3
        assert n_layers == 2
        np.testing.assert_allclose(heads, gw.heads, atol=1e-3)


# ---------------------------------------------------------------------------
# GroundwaterWriter.write() — full dispatch
# ---------------------------------------------------------------------------


class TestWriteDispatch:
    """Tests for GroundwaterWriter.write() dispatching to sub-writers."""

    def test_write_all_components(self, tmp_path: Path) -> None:
        config = GWFileConfig(output_dir=tmp_path)
        writer = GroundwaterWriter(config)
        gw = _make_gw_component(n_nodes=2, n_layers=1)
        gw.wells = {
            1: Well(
                id=1,
                x=100.0,
                y=200.0,
                element=1,
                top_screen=50.0,
                bottom_screen=10.0,
                max_pump_rate=500.0,
                name="W1",
            ),
        }
        gw.aquifer_params = AquiferParameters(
            n_nodes=2,
            n_layers=1,
            kh=np.array([[10.0], [12.0]]),
            kv=np.array([[5.0], [6.0]]),
            specific_storage=np.array([[0.001], [0.002]]),
            specific_yield=np.array([[0.15], [0.18]]),
        )
        gw.boundary_conditions = [
            BoundaryCondition(
                id=1,
                bc_type="specified_head",
                nodes=[1],
                values=[100.0],
                layer=1,
            ),
        ]
        gw.tile_drains = {
            1: TileDrain(id=1, element=1, elevation=50.0, conductance=0.01),
        }
        gw.subsidence = [
            Subsidence(
                element=1,
                layer=1,
                elastic_storage=1e-5,
                inelastic_storage=1e-4,
                preconsolidation_head=80.0,
            ),
        ]
        gw.heads = np.array([[100.0], [105.0]])

        files = writer.write(gw)
        assert "wells" in files
        assert "aquifer_params" in files
        assert "boundary_conditions" in files
        assert "tile_drains" in files
        assert "subsidence" in files
        assert "initial_heads" in files
        for path in files.values():
            assert path.exists()

    def test_write_empty_components(self, tmp_path: Path) -> None:
        """Write with no components produces no files."""
        config = GWFileConfig(output_dir=tmp_path)
        writer = GroundwaterWriter(config)
        gw = _make_gw_component()
        files = writer.write(gw)
        assert files == {}


# ---------------------------------------------------------------------------
# GWFileConfig — additional path getter tests
# ---------------------------------------------------------------------------


class TestGWFileConfigPathGettersExtended:
    """Additional tests for GWFileConfig path getter methods."""

    def test_default_filenames(self, tmp_path: Path) -> None:
        config = GWFileConfig(output_dir=tmp_path)
        assert config.get_wells_path().name == "wells.dat"
        assert config.get_pumping_path().name == "pumping.dat"
        assert config.get_aquifer_params_path().name == "aquifer_params.dat"
        assert config.get_boundary_conditions_path().name == "boundary_conditions.dat"
        assert config.get_tile_drains_path().name == "tile_drains.dat"
        assert config.get_subsidence_path().name == "subsidence.dat"
        assert config.get_initial_heads_path().name == "initial_heads.dat"

    def test_custom_filenames_all(self, tmp_path: Path) -> None:
        config = GWFileConfig(
            output_dir=tmp_path,
            wells_file="w.in",
            pumping_file="p.in",
            aquifer_params_file="aq.in",
            boundary_conditions_file="bc.in",
            tile_drains_file="td.in",
            subsidence_file="sub.in",
            initial_heads_file="ih.in",
        )
        assert config.get_wells_path() == tmp_path / "w.in"
        assert config.get_pumping_path() == tmp_path / "p.in"
        assert config.get_aquifer_params_path() == tmp_path / "aq.in"
        assert config.get_boundary_conditions_path() == tmp_path / "bc.in"
        assert config.get_tile_drains_path() == tmp_path / "td.in"
        assert config.get_subsidence_path() == tmp_path / "sub.in"
        assert config.get_initial_heads_path() == tmp_path / "ih.in"

    def test_paths_resolve_under_output_dir(self, tmp_path: Path) -> None:
        sub_dir = tmp_path / "subdir"
        sub_dir.mkdir()
        config = GWFileConfig(output_dir=sub_dir)
        for getter in [
            config.get_wells_path,
            config.get_pumping_path,
            config.get_aquifer_params_path,
            config.get_boundary_conditions_path,
            config.get_tile_drains_path,
            config.get_subsidence_path,
            config.get_initial_heads_path,
        ]:
            assert getter().parent == sub_dir


# ---------------------------------------------------------------------------
# FaceFlowSpec dataclass
# ---------------------------------------------------------------------------


class TestFaceFlowSpec:
    """Tests for FaceFlowSpec dataclass."""

    def test_construction(self) -> None:
        spec = FaceFlowSpec(id=1, layer=1, node_a=10, node_b=20, name="FF1")
        assert spec.id == 1
        assert spec.layer == 1
        assert spec.node_a == 10
        assert spec.node_b == 20
        assert spec.name == "FF1"

    def test_default_name_empty(self) -> None:
        spec = FaceFlowSpec(id=2, layer=3, node_a=5, node_b=6)
        assert spec.name == ""
