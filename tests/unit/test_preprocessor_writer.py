"""Unit tests for :class:`pyiwfm.io.config.PreProcessorFileConfig`.

The IWFM ASCII format-level tests for nodes/elements/stratigraphy live in
``test_io_mesh.py``; the orchestration tests for ``PreProcessorWriter``
live in ``test_io_preprocessor_coverage.py`` and
``test_preprocessor_writer_full.py``.
"""

from __future__ import annotations

from pathlib import Path

from pyiwfm.io.config import PreProcessorFileConfig


class TestPreProcessorFileConfig:
    """Tests for PreProcessorFileConfig dataclass."""

    def test_basic_creation(self, tmp_path: Path) -> None:
        config = PreProcessorFileConfig(output_dir=tmp_path)

        assert config.output_dir == tmp_path
        assert config.main_file == "Preprocessor.in"
        assert config.node_file == "Nodes.dat"
        assert config.element_file == "Elements.dat"
        assert config.stratigraphy_file == "Stratigraphy.dat"
        assert config.stream_config_file == "StreamConfig.dat"
        assert config.lake_config_file == "LakeConfig.dat"

    def test_custom_file_names(self, tmp_path: Path) -> None:
        config = PreProcessorFileConfig(
            output_dir=tmp_path,
            main_file="custom_pre.in",
            node_file="custom_nodes.dat",
            element_file="custom_elements.dat",
        )

        assert config.main_file == "custom_pre.in"
        assert config.node_file == "custom_nodes.dat"
        assert config.element_file == "custom_elements.dat"

    def test_version_settings(self, tmp_path: Path) -> None:
        config = PreProcessorFileConfig(
            output_dir=tmp_path,
            stream_version="4.0",
            lake_version="4.0",
        )

        assert config.stream_version == "4.0"
        assert config.lake_version == "4.0"

    def test_path_properties(self, tmp_path: Path) -> None:
        config = PreProcessorFileConfig(output_dir=tmp_path)

        assert config.main_path == tmp_path / "Preprocessor.in"
        assert config.node_path == tmp_path / "Nodes.dat"
        assert config.element_path == tmp_path / "Elements.dat"
        assert config.stratigraphy_path == tmp_path / "Stratigraphy.dat"
        assert config.stream_config_path == tmp_path / "StreamConfig.dat"
        assert config.lake_config_path == tmp_path / "LakeConfig.dat"

    def test_post_init_converts_string_to_path(self, tmp_path: Path) -> None:
        config = PreProcessorFileConfig(output_dir=str(tmp_path))

        assert isinstance(config.output_dir, Path)
        assert config.output_dir == tmp_path
