"""Additional coverage tests for io/model_loader.py (CompleteModelLoader)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pyiwfm.io.model_loader import (
    CommentAwareModelLoader,
    CompleteModelLoader,
    ModelLoadResult,
    ModelLoadResultWithComments,
    load_complete_model,
    load_model_with_comments,
)

# ---------------------------------------------------------------------------
# ModelLoadResult
# ---------------------------------------------------------------------------


class TestModelLoadResult:
    """Tests for ModelLoadResult dataclass."""

    def test_success_when_model_set(self) -> None:
        result = ModelLoadResult(model=MagicMock())
        assert result.success is True

    def test_not_success_when_model_none(self) -> None:
        result = ModelLoadResult()
        assert result.success is False

    def test_has_errors(self) -> None:
        result = ModelLoadResult(errors={"sim": "failed"})
        assert result.has_errors is True

    def test_no_errors(self) -> None:
        result = ModelLoadResult()
        assert result.has_errors is False


# ---------------------------------------------------------------------------
# ModelLoadResultWithComments
# ---------------------------------------------------------------------------


class TestModelLoadResultWithComments:
    """Tests for comment-aware load result."""

    def test_get_file_comments_found(self) -> None:
        mock_comments = MagicMock()
        result = ModelLoadResultWithComments(comment_metadata={"simulation_main": mock_comments})
        assert result.get_file_comments("simulation_main") is mock_comments

    def test_get_file_comments_not_found(self) -> None:
        result = ModelLoadResultWithComments()
        assert result.get_file_comments("nonexistent") is None


# ---------------------------------------------------------------------------
# CompleteModelLoader
# ---------------------------------------------------------------------------


class TestCompleteModelLoader:
    """Tests for CompleteModelLoader."""

    def test_init_stores_paths(self, tmp_path: Path) -> None:
        sim = tmp_path / "Sim.in"
        pp = tmp_path / "PP.in"
        loader = CompleteModelLoader(sim, pp)
        assert loader.simulation_file == sim
        assert loader.preprocessor_file == pp
        assert loader.base_dir == tmp_path

    def test_init_no_preprocessor(self, tmp_path: Path) -> None:
        sim = tmp_path / "Sim.in"
        loader = CompleteModelLoader(sim)
        assert loader.preprocessor_file is None

    @patch("pyiwfm.io.model_loader.CompleteModelLoader._read_simulation_config")
    def test_load_simulation_read_fails(self, mock_read, tmp_path: Path) -> None:
        mock_read.side_effect = RuntimeError("parse error")
        sim = tmp_path / "Sim.in"
        loader = CompleteModelLoader(sim)
        result = loader.load()
        assert not result.success
        assert "simulation" in result.errors

    @patch("pyiwfm.core.model.IWFMModel")
    @patch("pyiwfm.io.model_loader.CompleteModelLoader._read_simulation_config")
    def test_load_no_preprocessor_found(self, mock_read, mock_model, tmp_path: Path) -> None:
        mock_config = MagicMock()
        mock_config.preprocessor_file = None
        mock_read.return_value = mock_config

        sim = tmp_path / "Sim.in"
        loader = CompleteModelLoader(sim)
        result = loader.load()
        assert not result.success
        assert "preprocessor" in result.errors

    @patch("pyiwfm.core.model.IWFMModel")
    @patch("pyiwfm.io.model_loader.CompleteModelLoader._read_simulation_config")
    def test_load_success(self, mock_read, mock_model_cls, tmp_path: Path) -> None:
        mock_config = MagicMock()
        mock_config.preprocessor_file = "PP.in"
        mock_read.return_value = mock_config

        pp = tmp_path / "PP.in"
        pp.touch()

        mock_model = MagicMock()
        mock_model.metadata = {}
        mock_model_cls.from_simulation_with_preprocessor.return_value = mock_model

        sim = tmp_path / "Sim.in"
        loader = CompleteModelLoader(sim, pp)
        result = loader.load()
        assert result.success

    @patch("pyiwfm.core.model.IWFMModel")
    @patch("pyiwfm.io.model_loader.CompleteModelLoader._read_simulation_config")
    def test_load_model_raises_on_failure(self, mock_read, mock_model, tmp_path: Path) -> None:
        mock_read.side_effect = RuntimeError("fail")
        sim = tmp_path / "Sim.in"
        loader = CompleteModelLoader(sim)
        with pytest.raises(RuntimeError, match="Failed to load"):
            loader.load_model()

    @patch("pyiwfm.core.model.IWFMModel")
    @patch("pyiwfm.io.model_loader.CompleteModelLoader._read_simulation_config")
    def test_load_collects_component_errors(
        self, mock_read, mock_model_cls, tmp_path: Path
    ) -> None:
        mock_config = MagicMock()
        mock_config.preprocessor_file = None
        mock_read.return_value = mock_config

        pp = tmp_path / "PP.in"
        pp.touch()

        mock_model = MagicMock()
        mock_model.metadata = {"gw_load_error": "bad wells"}
        mock_model_cls.from_simulation_with_preprocessor.return_value = mock_model

        sim = tmp_path / "Sim.in"
        loader = CompleteModelLoader(sim, pp)
        result = loader.load()
        assert result.success
        assert "gw" in result.errors

    def test_resolve_path_relative(self, tmp_path: Path) -> None:
        sim = tmp_path / "Sim.in"
        loader = CompleteModelLoader(sim)
        result = loader._resolve_path("data/file.in")
        assert result == tmp_path / "data" / "file.in"

    def test_resolve_path_absolute(self, tmp_path: Path) -> None:
        sim = tmp_path / "Sim.in"
        loader = CompleteModelLoader(sim)
        abs_path = tmp_path / "other" / "file.in"
        assert loader._resolve_path(abs_path) == abs_path

    @patch("pyiwfm.io.model_loader.CompleteModelLoader._read_positional")
    def test_read_config_forces_positional(self, mock_pos, tmp_path: Path) -> None:
        mock_pos.return_value = MagicMock()
        sim = tmp_path / "Sim.in"
        loader = CompleteModelLoader(sim, use_positional_format=True)
        loader._read_simulation_config()
        mock_pos.assert_called_once()

    @patch("pyiwfm.io.model_loader.CompleteModelLoader._read_description_based")
    def test_read_config_forces_description(self, mock_desc, tmp_path: Path) -> None:
        mock_desc.return_value = MagicMock()
        sim = tmp_path / "Sim.in"
        loader = CompleteModelLoader(sim, use_positional_format=False)
        loader._read_simulation_config()
        mock_desc.assert_called_once()

    @patch("pyiwfm.io.model_loader.CompleteModelLoader._read_auto_detect")
    def test_read_config_auto_detect(self, mock_auto, tmp_path: Path) -> None:
        mock_auto.return_value = MagicMock()
        sim = tmp_path / "Sim.in"
        loader = CompleteModelLoader(sim, use_positional_format=None)
        loader._read_simulation_config()
        mock_auto.assert_called_once()


# ---------------------------------------------------------------------------
# load_complete_model convenience function
# ---------------------------------------------------------------------------


class TestLoadCompleteModel:
    """Tests for the load_complete_model() convenience function."""

    @patch("pyiwfm.io.model_loader.CompleteModelLoader")
    def test_calls_load_model(self, mock_loader_cls) -> None:
        mock_loader = MagicMock()
        mock_loader.load_model.return_value = MagicMock()
        mock_loader_cls.return_value = mock_loader

        load_complete_model("sim.in", "pp.in")
        mock_loader.load_model.assert_called_once()


# ---------------------------------------------------------------------------
# CommentAwareModelLoader
# ---------------------------------------------------------------------------


class TestCommentAwareModelLoader:
    """Tests for CommentAwareModelLoader."""

    def test_init(self, tmp_path: Path) -> None:
        sim = tmp_path / "Sim.in"
        loader = CommentAwareModelLoader(sim, preserve_comments=True)
        assert loader.preserve_comments is True

    @patch("pyiwfm.io.model_loader.CompleteModelLoader.load")
    def test_load_returns_with_comments(self, mock_base_load, tmp_path: Path) -> None:
        mock_result = ModelLoadResult(model=None, errors={"sim": "fail"})
        mock_base_load.return_value = mock_result

        sim = tmp_path / "Sim.in"
        loader = CommentAwareModelLoader(sim, preserve_comments=False)
        result = loader.load()
        assert isinstance(result, ModelLoadResultWithComments)


# ---------------------------------------------------------------------------
# Additional coverage tests for missing lines
# ---------------------------------------------------------------------------


class TestPreprocessorResolutionFromSimConfig:
    """Tests for lines 122-128: preprocessor file resolution from sim_config."""

    @patch("pyiwfm.io.model_loader.CompleteModelLoader._read_simulation_config")
    def test_pp_resolved_from_sim_config_exists(self, mock_read, tmp_path: Path) -> None:
        """When no pp_file given, resolve from sim_config.preprocessor_file (exists)."""
        pp = tmp_path / "PP.in"
        pp.touch()

        mock_config = MagicMock()
        mock_config.preprocessor_file = "PP.in"
        mock_read.return_value = mock_config

        mock_model = MagicMock()
        mock_model.metadata = {}

        sim = tmp_path / "Sim.in"
        loader = CompleteModelLoader(sim, preprocessor_file=None)

        with patch(
            "pyiwfm.core.model.IWFMModel.from_simulation_with_preprocessor",
            return_value=mock_model,
        ):
            result = loader.load()

        assert result.success
        assert result.simulation_config is mock_config

    @patch("pyiwfm.io.model_loader.CompleteModelLoader._read_simulation_config")
    def test_pp_resolved_from_sim_config_not_found(self, mock_read, tmp_path: Path) -> None:
        """When pp resolved from sim_config but file does not exist."""
        mock_config = MagicMock()
        mock_config.preprocessor_file = "NonExistent_PP.in"
        mock_read.return_value = mock_config

        sim = tmp_path / "Sim.in"
        loader = CompleteModelLoader(sim, preprocessor_file=None)
        result = loader.load()

        # pp_file resolved but not found -> warning + pp set to None -> error
        assert not result.success
        assert "preprocessor" in result.errors
        assert any("not found" in w for w in result.warnings)


class TestModelLoadExceptionInFromSimulation:
    """Tests for lines 154-156: exception during model construction."""

    @patch("pyiwfm.io.model_loader.CompleteModelLoader._read_simulation_config")
    def test_model_construction_raises(self, mock_read, tmp_path: Path) -> None:
        """When IWFMModel.from_simulation_with_preprocessor raises an exception."""
        mock_config = MagicMock()
        mock_config.preprocessor_file = None
        mock_read.return_value = mock_config

        pp = tmp_path / "PP.in"
        pp.touch()
        sim = tmp_path / "Sim.in"

        loader = CompleteModelLoader(sim, pp)

        with patch(
            "pyiwfm.core.model.IWFMModel.from_simulation_with_preprocessor",
            side_effect=ValueError("corrupt file"),
        ):
            result = loader.load()

        assert not result.success
        assert "model" in result.errors
        assert "corrupt file" in result.errors["model"]


class TestComponentErrorMetadataLoop:
    """Tests for line 147->146: iterating metadata for _load_error keys."""

    @patch("pyiwfm.io.model_loader.CompleteModelLoader._read_simulation_config")
    def test_multiple_load_errors_collected(self, mock_read, tmp_path: Path) -> None:
        """When metadata has multiple _load_error entries they are all collected."""
        mock_config = MagicMock()
        mock_config.preprocessor_file = None
        mock_read.return_value = mock_config

        pp = tmp_path / "PP.in"
        pp.touch()
        sim = tmp_path / "Sim.in"

        mock_model = MagicMock()
        mock_model.metadata = {
            "gw_load_error": "bad wells",
            "stream_load_error": "missing reach",
            "name": "TestModel",
        }

        loader = CompleteModelLoader(sim, pp)

        with patch(
            "pyiwfm.core.model.IWFMModel.from_simulation_with_preprocessor",
            return_value=mock_model,
        ):
            result = loader.load()

        assert result.success
        assert "gw" in result.errors
        assert "stream" in result.errors
        assert "name" not in result.errors

    @patch("pyiwfm.io.model_loader.CompleteModelLoader._read_simulation_config")
    def test_no_load_errors_in_metadata(self, mock_read, tmp_path: Path) -> None:
        """When metadata has no _load_error keys, errors dict stays empty."""
        mock_config = MagicMock()
        mock_config.preprocessor_file = None
        mock_read.return_value = mock_config

        pp = tmp_path / "PP.in"
        pp.touch()
        sim = tmp_path / "Sim.in"

        mock_model = MagicMock()
        mock_model.metadata = {"name": "TestModel", "version": "1.0"}

        loader = CompleteModelLoader(sim, pp)

        with patch(
            "pyiwfm.core.model.IWFMModel.from_simulation_with_preprocessor",
            return_value=mock_model,
        ):
            result = loader.load()

        assert result.success
        assert not result.has_errors


class TestLoadModelSuccessPath:
    """Tests for line 175: load_model() successful return."""

    @patch("pyiwfm.io.model_loader.CompleteModelLoader._read_simulation_config")
    def test_load_model_returns_model_on_success(self, mock_read, tmp_path: Path) -> None:
        """load_model() returns the IWFMModel instance when loading succeeds."""
        mock_config = MagicMock()
        mock_config.preprocessor_file = None
        mock_read.return_value = mock_config

        pp = tmp_path / "PP.in"
        pp.touch()
        sim = tmp_path / "Sim.in"

        mock_model = MagicMock()
        mock_model.metadata = {}

        loader = CompleteModelLoader(sim, pp)

        with patch(
            "pyiwfm.core.model.IWFMModel.from_simulation_with_preprocessor",
            return_value=mock_model,
        ):
            model = loader.load_model()

        assert model is mock_model


class TestAutoDetectReturnsConfig:
    """Tests for line 224: _read_auto_detect returning config with dates/files."""

    @patch("pyiwfm.io.model_loader.CompleteModelLoader._read_description_based")
    def test_auto_detect_returns_config_with_dates(self, mock_desc, tmp_path: Path) -> None:
        """Auto-detect returns config when has_dates is True."""
        from datetime import datetime

        mock_config = MagicMock()
        mock_config.start_date = datetime(2010, 1, 1)
        mock_config.end_date = datetime(2020, 12, 31)
        mock_config.groundwater_file = None
        mock_config.streams_file = None
        mock_config.rootzone_file = None
        mock_desc.return_value = mock_config

        sim = tmp_path / "Sim.in"
        loader = CompleteModelLoader(sim, use_positional_format=None)
        result = loader._read_auto_detect()

        assert result is mock_config

    @patch("pyiwfm.io.model_loader.CompleteModelLoader._read_description_based")
    def test_auto_detect_returns_config_with_files(self, mock_desc, tmp_path: Path) -> None:
        """Auto-detect returns config when has_files is True."""
        from datetime import datetime

        mock_config = MagicMock()
        mock_config.start_date = datetime(2000, 1, 1)
        mock_config.end_date = datetime(2000, 12, 31)
        mock_config.groundwater_file = "GW.in"
        mock_config.streams_file = None
        mock_config.rootzone_file = None
        mock_desc.return_value = mock_config

        sim = tmp_path / "Sim.in"
        loader = CompleteModelLoader(sim, use_positional_format=None)
        result = loader._read_auto_detect()

        assert result is mock_config

    @patch("pyiwfm.io.model_loader.CompleteModelLoader._read_positional")
    @patch("pyiwfm.io.model_loader.CompleteModelLoader._read_description_based")
    def test_auto_detect_falls_back_to_positional(
        self, mock_desc, mock_pos, tmp_path: Path
    ) -> None:
        """Auto-detect falls back to positional when no dates or files."""
        from datetime import datetime

        mock_config = MagicMock()
        mock_config.start_date = datetime(2000, 1, 1)
        mock_config.end_date = datetime(2000, 12, 31)
        mock_config.groundwater_file = None
        mock_config.streams_file = None
        mock_config.rootzone_file = None
        mock_desc.return_value = mock_config

        mock_pos_config = MagicMock()
        mock_pos.return_value = mock_pos_config

        sim = tmp_path / "Sim.in"
        loader = CompleteModelLoader(sim, use_positional_format=None)
        result = loader._read_auto_detect()

        assert result is mock_pos_config
        mock_pos.assert_called_once()

    @patch("pyiwfm.io.model_loader.CompleteModelLoader._read_positional")
    @patch("pyiwfm.io.model_loader.CompleteModelLoader._read_description_based")
    def test_auto_detect_exception_falls_back_to_positional(
        self, mock_desc, mock_pos, tmp_path: Path
    ) -> None:
        """Auto-detect falls back to positional on description reader exception."""
        mock_desc.side_effect = Exception("parse error")
        mock_pos_config = MagicMock()
        mock_pos.return_value = mock_pos_config

        sim = tmp_path / "Sim.in"
        loader = CompleteModelLoader(sim, use_positional_format=None)
        result = loader._read_auto_detect()

        assert result is mock_pos_config


class TestCommentAwareModelLoaderExtractAllComments:
    """Tests for lines 359, 374-408: _extract_all_comments in CommentAwareModelLoader."""

    @patch("pyiwfm.io.model_loader.CompleteModelLoader.load")
    @patch("pyiwfm.io.model_loader.CommentAwareModelLoader._extract_all_comments")
    def test_load_calls_extract_when_preserve_and_success(
        self, mock_extract, mock_base_load, tmp_path: Path
    ) -> None:
        """load() calls _extract_all_comments when preserve_comments=True and success."""
        mock_model = MagicMock()
        mock_result = ModelLoadResult(model=mock_model)
        mock_base_load.return_value = mock_result
        mock_extract.return_value = {"simulation_main": MagicMock()}

        sim = tmp_path / "Sim.in"
        loader = CommentAwareModelLoader(sim, preserve_comments=True)
        result = loader.load()

        mock_extract.assert_called_once_with(mock_result)
        assert "simulation_main" in result.comment_metadata

    @patch("pyiwfm.io.model_loader.CompleteModelLoader.load")
    def test_load_skips_extract_when_not_success(self, mock_base_load, tmp_path: Path) -> None:
        """load() skips comment extraction when model loading failed."""
        mock_result = ModelLoadResult(model=None, errors={"sim": "fail"})
        mock_base_load.return_value = mock_result

        sim = tmp_path / "Sim.in"
        loader = CommentAwareModelLoader(sim, preserve_comments=True)
        result = loader.load()

        assert result.comment_metadata == {}

    @patch("pyiwfm.io.model_loader.CommentAwareModelLoader._extract_component_comments")
    @patch("pyiwfm.io.comment_extractor.CommentExtractor.extract")
    def test_extract_all_comments_simulation_and_pp(
        self, mock_extract, mock_comp_extract, tmp_path: Path
    ) -> None:
        """_extract_all_comments extracts from simulation main and preprocessor."""
        sim = tmp_path / "Sim.in"
        sim.touch()
        pp = tmp_path / "PP.in"
        pp.touch()

        mock_sim_comments = MagicMock()
        mock_pp_comments = MagicMock()
        mock_extract.side_effect = [mock_sim_comments, mock_pp_comments]

        mock_config = MagicMock()
        mock_config.preprocessor_file = "PP.in"
        base_result = ModelLoadResult(model=MagicMock(), simulation_config=mock_config)

        loader = CommentAwareModelLoader(sim, pp, preserve_comments=True)
        comments = loader._extract_all_comments(base_result)

        assert comments["simulation_main"] is mock_sim_comments
        assert comments["preprocessor_main"] is mock_pp_comments
        mock_comp_extract.assert_called_once()

    @patch("pyiwfm.io.model_loader.CommentAwareModelLoader._extract_component_comments")
    @patch("pyiwfm.io.comment_extractor.CommentExtractor.extract")
    def test_extract_all_comments_sim_extract_fails(
        self, mock_extract, mock_comp_extract, tmp_path: Path
    ) -> None:
        """Simulation comment extraction failure is logged but not fatal."""
        sim = tmp_path / "Sim.in"
        sim.touch()

        mock_extract.side_effect = Exception("read error")

        # simulation_config=None avoids pp resolution path
        base_result = ModelLoadResult(model=MagicMock(), simulation_config=None)

        pp = tmp_path / "PP.in"
        # PP file does not exist, so pp extraction path is skipped
        loader = CommentAwareModelLoader(sim, pp, preserve_comments=True)
        comments = loader._extract_all_comments(base_result)

        # Sim comments extraction failed, pp does not exist
        assert "simulation_main" not in comments

    @patch("pyiwfm.io.model_loader.CommentAwareModelLoader._extract_component_comments")
    @patch("pyiwfm.io.comment_extractor.CommentExtractor.extract")
    def test_extract_all_comments_pp_from_sim_config(
        self, mock_extract, mock_comp_extract, tmp_path: Path
    ) -> None:
        """PP file resolved from sim_config when self.preprocessor_file is None."""
        sim = tmp_path / "Sim.in"
        sim.touch()
        pp = tmp_path / "PP.in"
        pp.touch()

        mock_sim_comments = MagicMock()
        mock_pp_comments = MagicMock()
        mock_extract.side_effect = [mock_sim_comments, mock_pp_comments]

        mock_config = MagicMock()
        mock_config.preprocessor_file = "PP.in"
        base_result = ModelLoadResult(model=MagicMock(), simulation_config=mock_config)

        loader = CommentAwareModelLoader(sim, preserve_comments=True)
        # self.preprocessor_file is None, so it should resolve from sim_config
        comments = loader._extract_all_comments(base_result)

        assert "preprocessor_main" in comments

    @patch("pyiwfm.io.comment_extractor.CommentExtractor.extract")
    def test_extract_all_comments_pp_extract_fails(self, mock_extract, tmp_path: Path) -> None:
        """Preprocessor comment extraction failure is logged but not fatal."""
        sim = tmp_path / "Sim.in"
        sim.touch()
        pp = tmp_path / "PP.in"
        pp.touch()

        mock_sim_comments = MagicMock()
        mock_extract.side_effect = [mock_sim_comments, Exception("pp error")]

        mock_config = MagicMock()
        mock_config.preprocessor_file = None
        # No component files to avoid _extract_component_comments calls
        base_result = ModelLoadResult(model=MagicMock(), simulation_config=None)

        loader = CommentAwareModelLoader(sim, pp, preserve_comments=True)
        comments = loader._extract_all_comments(base_result)

        assert "simulation_main" in comments
        assert "preprocessor_main" not in comments


class TestExtractComponentComments:
    """Tests for lines 424-442: _extract_component_comments."""

    @patch("pyiwfm.io.comment_extractor.CommentExtractor.extract")
    def test_extracts_comments_from_existing_component_files(
        self, mock_extract, tmp_path: Path
    ) -> None:
        """Component files that exist have their comments extracted."""
        gw_file = tmp_path / "GW.in"
        gw_file.touch()
        stream_file = tmp_path / "Stream.in"
        stream_file.touch()

        mock_gw_comments = MagicMock()
        mock_stream_comments = MagicMock()
        mock_extract.side_effect = [mock_gw_comments, mock_stream_comments]

        mock_config = MagicMock()
        mock_config.groundwater_file = "GW.in"
        mock_config.streams_file = "Stream.in"
        mock_config.lake_file = None
        mock_config.rootzone_file = None
        mock_config.small_watershed_file = None
        mock_config.unsaturated_zone_file = None

        sim = tmp_path / "Sim.in"
        loader = CommentAwareModelLoader(sim, preserve_comments=True)

        comments: dict[str, MagicMock] = {}
        extractor_instance = MagicMock()
        extractor_instance.extract.side_effect = [mock_gw_comments, mock_stream_comments]
        loader._extract_component_comments(mock_config, extractor_instance, comments)

        assert "gw_main" in comments
        assert "stream_main" in comments
        assert comments["gw_main"] is mock_gw_comments
        assert comments["stream_main"] is mock_stream_comments

    def test_skips_missing_component_files(self, tmp_path: Path) -> None:
        """Component files that do not exist are skipped."""
        mock_config = MagicMock()
        mock_config.groundwater_file = "Missing_GW.in"
        mock_config.streams_file = None
        mock_config.lake_file = None
        mock_config.rootzone_file = None
        mock_config.small_watershed_file = None
        mock_config.unsaturated_zone_file = None

        sim = tmp_path / "Sim.in"
        loader = CommentAwareModelLoader(sim, preserve_comments=True)

        comments: dict = {}
        mock_extractor = MagicMock()
        loader._extract_component_comments(mock_config, mock_extractor, comments)

        # GW file does not exist, so it should not be in comments
        assert "gw_main" not in comments
        mock_extractor.extract.assert_not_called()

    def test_component_extract_exception_handled(self, tmp_path: Path) -> None:
        """Exception extracting component comments is caught, not raised."""
        gw_file = tmp_path / "GW.in"
        gw_file.touch()

        mock_config = MagicMock()
        mock_config.groundwater_file = "GW.in"
        mock_config.streams_file = None
        mock_config.lake_file = None
        mock_config.rootzone_file = None
        mock_config.small_watershed_file = None
        mock_config.unsaturated_zone_file = None

        sim = tmp_path / "Sim.in"
        loader = CommentAwareModelLoader(sim, preserve_comments=True)

        comments: dict = {}
        mock_extractor = MagicMock()
        mock_extractor.extract.side_effect = Exception("read error")
        loader._extract_component_comments(mock_config, mock_extractor, comments)

        assert "gw_main" not in comments


class TestLoadModelWithComments:
    """Tests for lines 476-488: load_model_with_comments() convenience function."""

    @patch("pyiwfm.io.model_loader.CommentAwareModelLoader.load")
    def test_returns_model_and_comments_on_success(self, mock_load, tmp_path: Path) -> None:
        """load_model_with_comments returns (model, comments) on success."""
        mock_model = MagicMock()
        mock_comments = {"simulation_main": MagicMock()}
        mock_result = ModelLoadResultWithComments(model=mock_model, comment_metadata=mock_comments)
        mock_load.return_value = mock_result

        model, comments = load_model_with_comments(tmp_path / "Sim.in", tmp_path / "PP.in")

        assert model is mock_model
        assert comments is mock_comments

    @patch("pyiwfm.io.model_loader.CommentAwareModelLoader.load")
    def test_raises_on_failure(self, mock_load, tmp_path: Path) -> None:
        """load_model_with_comments raises RuntimeError when loading fails."""
        mock_result = ModelLoadResultWithComments(model=None, errors={"simulation": "bad file"})
        mock_load.return_value = mock_result

        with pytest.raises(RuntimeError, match="Failed to load IWFM model"):
            load_model_with_comments(tmp_path / "Sim.in")

    @patch("pyiwfm.io.model_loader.CommentAwareModelLoader.load")
    def test_passes_format_flag(self, mock_load, tmp_path: Path) -> None:
        """load_model_with_comments passes use_positional_format to loader."""
        mock_model = MagicMock()
        mock_result = ModelLoadResultWithComments(model=mock_model, comment_metadata={})
        mock_load.return_value = mock_result

        load_model_with_comments(
            tmp_path / "Sim.in",
            preprocessor_file=tmp_path / "PP.in",
            use_positional_format=True,
        )

        mock_load.assert_called_once()
