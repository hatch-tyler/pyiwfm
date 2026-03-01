"""Coverage tests for gw_subsidence.py reader module.

Tests SubsidenceNodeParams, SubsidenceConfig, helper functions,
and SubsidenceReader for both v4.0 and v5.0 IWFM file formats.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pyiwfm.io.gw_subsidence import (
    SubsidenceConfig,
    SubsidenceNodeParams,
    SubsidenceReader,
    read_gw_subsidence,
)
from pyiwfm.io.iwfm_reader import is_comment_line as _is_comment_line
from pyiwfm.io.iwfm_reader import strip_inline_comment as _strip_comment

# ---------------------------------------------------------------------------
# Dataclass construction
# ---------------------------------------------------------------------------


class TestSubsidenceNodeParams:
    """Tests for SubsidenceNodeParams dataclass."""

    def test_default_construction(self) -> None:
        """Default construction yields empty lists and node_id 0."""
        params = SubsidenceNodeParams()
        assert params.node_id == 0
        assert params.elastic_sc == []
        assert params.inelastic_sc == []
        assert params.interbed_thick == []
        assert params.interbed_thick_min == []
        assert params.precompact_head == []
        assert params.kv_sub == []
        assert params.n_eq == []

    def test_construction_with_values(self) -> None:
        """SubsidenceNodeParams stores per-layer lists correctly."""
        params = SubsidenceNodeParams(
            node_id=5,
            elastic_sc=[0.1, 0.2],
            inelastic_sc=[0.3, 0.4],
            interbed_thick=[10.0, 20.0],
            interbed_thick_min=[1.0, 2.0],
            precompact_head=[50.0, 60.0],
            kv_sub=[0.01, 0.02],
            n_eq=[3.0, 4.0],
        )
        assert params.node_id == 5
        assert len(params.elastic_sc) == 2
        assert params.kv_sub[1] == pytest.approx(0.02)


class TestSubsidenceConfig:
    """Tests for SubsidenceConfig dataclass."""

    def test_default_construction(self) -> None:
        """Default construction yields sensible defaults."""
        config = SubsidenceConfig()
        assert config.version == ""
        assert config.ic_file is None
        assert config.tecplot_file is None
        assert config.final_subs_file is None
        assert config.output_factor == pytest.approx(1.0)
        assert config.output_unit == "FEET"
        assert config.interbed_dz == pytest.approx(0.0)
        assert config.n_parametric_grids == 0
        assert config.conversion_factors == []
        assert config.node_params == []
        assert config.n_nodes == 0
        assert config.n_layers == 0
        assert config.ic_factor == pytest.approx(1.0)
        assert config.ic_interbed_thick is None
        assert config.ic_precompact_head is None


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestHelpers:
    """Tests for _is_comment_line and _strip_comment."""

    @pytest.mark.parametrize(
        "line,expected",
        [
            ("C  This is a comment", True),
            ("c  lowercase comment", True),
            ("*  asterisk comment", True),
            ("", True),
            ("   ", True),
            ("100  / some value", False),
            ("# version line", False),
        ],
    )
    def test_is_comment_line(self, line: str, expected: bool) -> None:
        assert _is_comment_line(line) is expected

    def test_strip_comment_with_slash(self) -> None:
        """Value followed by ' / description' is split correctly."""
        value, desc = _strip_comment("100.0 / Output factor")
        assert value == "100.0"
        assert desc == "Output factor"

    def test_strip_comment_hash_not_delimiter(self) -> None:
        """'#' is NOT a comment delimiter in IWFM — only '/' is."""
        value, desc = _strip_comment("4.0 # version")
        assert value == "4.0 # version"
        assert desc == ""

    def test_strip_comment_no_description(self) -> None:
        """Line with no delimiter returns empty description."""
        value, desc = _strip_comment("42")
        assert value == "42"
        assert desc == ""


# ---------------------------------------------------------------------------
# File builder helpers
# ---------------------------------------------------------------------------


def _write_subsidence_v40(
    path: Path,
    n_nodes: int = 2,
    n_layers: int = 1,
) -> None:
    """Write a minimal v4.0 subsidence parameter file."""
    lines = [
        "C  Subsidence parameter file\n",
        "# 4.0\n",
        "  / IC file path (empty)\n",
        "  / Tecplot output file (empty)\n",
        "  / Final subs output file (empty)\n",
        "1.0  / Output factor\n",
        "FT   / Output unit\n",
        "0    / NOUTS\n",
        "0    / N parametric grids\n",
        # Conversion factors: 6 values for v4.0
        "1.0 1.0 1.0 1.0 1.0 1.0  / factors\n",
    ]

    # For each node, write n_layers lines.
    # First line of each node has: NodeID Elastic Inelastic Thick ThickMin Precompact
    for node_idx in range(n_nodes):
        node_id = node_idx + 1
        for layer_idx in range(n_layers):
            if layer_idx == 0:
                lines.append(f"  {node_id}  0.001  0.01  10.0  1.0  50.0\n")
            else:
                lines.append("  0.002  0.02  20.0  2.0  60.0\n")

    path.write_text("".join(lines))


def _write_subsidence_v50(
    path: Path,
    n_nodes: int = 2,
    n_layers: int = 1,
) -> None:
    """Write a minimal v5.0 subsidence parameter file."""
    lines = [
        "C  Subsidence parameter file v5.0\n",
        "# 5.0\n",
        "  / IC file path (empty)\n",
        "  / Tecplot output file (empty)\n",
        "  / Final subs output file (empty)\n",
        "1.0  / Output factor\n",
        "FT   / Output unit\n",
        "0    / NOUTS\n",
        "5.0  / Interbed DZ\n",
        "0    / N parametric grids\n",
        # Conversion factors: 7 values for v5.0
        "1.0 1.0 1.0 1.0 1.0 1.0 1.0  / factors\n",
    ]

    for node_idx in range(n_nodes):
        node_id = node_idx + 1
        for layer_idx in range(n_layers):
            if layer_idx == 0:
                # NodeID Elastic Inelastic Thick ThickMin Precompact Kv Neq
                lines.append(f"  {node_id}  0.001  0.01  10.0  1.0  50.0  0.005  2.0\n")
            else:
                lines.append("  0.002  0.02  20.0  2.0  60.0  0.006  3.0\n")

    path.write_text("".join(lines))


def _write_ic_file(path: Path, n_nodes: int = 2, n_layers: int = 1) -> None:
    """Write a minimal initial-conditions file."""
    lines = [
        "C  Initial conditions file\n",
        "1.0  / IC factor\n",
    ]
    for i in range(n_nodes):
        node_id = i + 1
        # ID  InterbedThick(per layer)  PrecompactHead(per layer)
        thick_vals = "  ".join(f"{5.0 + i}" for _ in range(n_layers))
        head_vals = "  ".join(f"{45.0 + i}" for _ in range(n_layers))
        lines.append(f"  {node_id}  {thick_vals}  {head_vals}\n")
    path.write_text("".join(lines))


# ---------------------------------------------------------------------------
# SubsidenceReader tests
# ---------------------------------------------------------------------------


class TestSubsidenceReaderV40:
    """Tests for reading v4.0 subsidence files."""

    def test_read_version(self, tmp_path: Path) -> None:
        """Reader detects v4.0 version string."""
        filepath = tmp_path / "subs.dat"
        _write_subsidence_v40(filepath, n_nodes=2, n_layers=1)

        reader = SubsidenceReader()
        config = reader.read(filepath, n_nodes=2, n_layers=1)

        assert config.version == "4.0"

    def test_read_output_factor_and_unit(self, tmp_path: Path) -> None:
        """Reader captures output factor and unit."""
        filepath = tmp_path / "subs.dat"
        _write_subsidence_v40(filepath, n_nodes=1, n_layers=1)

        config = SubsidenceReader().read(filepath, n_nodes=1, n_layers=1)
        assert config.output_factor == pytest.approx(1.0)
        assert config.output_unit == "FT"

    def test_read_conversion_factors(self, tmp_path: Path) -> None:
        """Reader captures the 6 conversion factors."""
        filepath = tmp_path / "subs.dat"
        _write_subsidence_v40(filepath, n_nodes=1, n_layers=1)

        config = SubsidenceReader().read(filepath, n_nodes=1, n_layers=1)
        assert len(config.conversion_factors) == 6
        assert all(f == pytest.approx(1.0) for f in config.conversion_factors)

    def test_read_node_params_single_layer(self, tmp_path: Path) -> None:
        """Reader parses per-node parameters for 1-layer v4.0 file."""
        filepath = tmp_path / "subs.dat"
        _write_subsidence_v40(filepath, n_nodes=2, n_layers=1)

        config = SubsidenceReader().read(filepath, n_nodes=2, n_layers=1)
        assert len(config.node_params) == 2
        assert config.node_params[0].node_id == 1
        assert config.node_params[1].node_id == 2
        assert config.node_params[0].elastic_sc[0] == pytest.approx(0.001)
        assert config.node_params[0].inelastic_sc[0] == pytest.approx(0.01)
        assert config.node_params[0].interbed_thick[0] == pytest.approx(10.0)
        assert config.node_params[0].interbed_thick_min[0] == pytest.approx(1.0)
        assert config.node_params[0].precompact_head[0] == pytest.approx(50.0)

    def test_read_multi_layer(self, tmp_path: Path) -> None:
        """Reader handles multiple layers per node in v4.0."""
        filepath = tmp_path / "subs.dat"
        _write_subsidence_v40(filepath, n_nodes=1, n_layers=2)

        config = SubsidenceReader().read(filepath, n_nodes=1, n_layers=2)
        assert len(config.node_params) == 1
        params = config.node_params[0]
        assert len(params.elastic_sc) == 2
        assert params.elastic_sc[0] == pytest.approx(0.001)
        assert params.elastic_sc[1] == pytest.approx(0.002)


class TestSubsidenceReaderV50:
    """Tests for reading v5.0 subsidence files."""

    def test_read_version(self, tmp_path: Path) -> None:
        """Reader detects v5.0 version string."""
        filepath = tmp_path / "subs50.dat"
        _write_subsidence_v50(filepath, n_nodes=2, n_layers=1)

        config = SubsidenceReader().read(filepath, n_nodes=2, n_layers=1)
        assert config.version.startswith("5")

    def test_read_interbed_dz(self, tmp_path: Path) -> None:
        """Reader captures interbed DZ (v5.0-only field)."""
        filepath = tmp_path / "subs50.dat"
        _write_subsidence_v50(filepath, n_nodes=1, n_layers=1)

        config = SubsidenceReader().read(filepath, n_nodes=1, n_layers=1)
        assert config.interbed_dz == pytest.approx(5.0)

    def test_read_v50_node_params(self, tmp_path: Path) -> None:
        """Reader parses v5.0 extended parameters including kv_sub and n_eq."""
        filepath = tmp_path / "subs50.dat"
        _write_subsidence_v50(filepath, n_nodes=1, n_layers=1)

        config = SubsidenceReader().read(filepath, n_nodes=1, n_layers=1)
        assert len(config.node_params) == 1
        params = config.node_params[0]
        assert params.kv_sub[0] == pytest.approx(0.005)
        assert params.n_eq[0] == pytest.approx(2.0)

    def test_read_conversion_factors_v50(self, tmp_path: Path) -> None:
        """Reader captures the 7 conversion factors for v5.0."""
        filepath = tmp_path / "subs50.dat"
        _write_subsidence_v50(filepath, n_nodes=1, n_layers=1)

        config = SubsidenceReader().read(filepath, n_nodes=1, n_layers=1)
        assert len(config.conversion_factors) == 7


class TestSubsidenceReaderICFile:
    """Tests for IC file reading."""

    def test_read_ic_file(self, tmp_path: Path) -> None:
        """Reader loads IC arrays when IC file exists."""
        ic_path = tmp_path / "subs_ic.dat"
        _write_ic_file(ic_path, n_nodes=2, n_layers=1)

        # Write main file that references the IC file
        main_path = tmp_path / "subs.dat"
        lines = [
            "C  Subsidence parameter file\n",
            "# 4.0\n",
            f"{ic_path.name}  / IC file path\n",
            "  / Tecplot output file (empty)\n",
            "  / Final subs output file (empty)\n",
            "1.0  / Output factor\n",
            "FT   / Output unit\n",
            "0    / NOUTS\n",
            "0    / N parametric grids\n",
            "1.0 1.0 1.0 1.0 1.0 1.0  / factors\n",
            "  1  0.001  0.01  10.0  1.0  50.0\n",
            "  2  0.001  0.01  10.0  1.0  50.0\n",
        ]
        main_path.write_text("".join(lines))

        config = SubsidenceReader().read(main_path, n_nodes=2, n_layers=1)
        assert config.ic_interbed_thick is not None
        assert config.ic_precompact_head is not None
        assert config.ic_interbed_thick.shape == (2, 1)
        assert config.ic_precompact_head.shape == (2, 1)
        assert config.ic_interbed_thick[0, 0] == pytest.approx(5.0)
        assert config.ic_precompact_head[0, 0] == pytest.approx(45.0)


class TestConvenienceFunction:
    """Tests for the module-level read_gw_subsidence function."""

    def test_read_gw_subsidence(self, tmp_path: Path) -> None:
        """Convenience function delegates to SubsidenceReader.read()."""
        filepath = tmp_path / "subs.dat"
        _write_subsidence_v40(filepath, n_nodes=2, n_layers=1)

        config = read_gw_subsidence(filepath, n_nodes=2, n_layers=1)
        assert config.version == "4.0"
        assert len(config.node_params) == 2
