"""Coverage tests for templates/engine.py module.

Tests TemplateEngine creation, rendering methods, hybrid rendering,
comment-aware rendering, IWFMFileWriter, and filter functions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# Skip the entire module if jinja2 is not installed
jinja2 = pytest.importorskip("jinja2")

from pyiwfm.templates.engine import (
    IWFMFileWriter,
    TemplateEngine,
    _fortran_float,
    _fortran_int,
    _iwfm_comment,
    _pad_left,
    _pad_right,
)
from pyiwfm.io.comment_metadata import (
    CommentMetadata,
    PreserveMode,
    SectionComments,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine() -> TemplateEngine:
    """A default TemplateEngine instance."""
    return TemplateEngine()


@pytest.fixture
def custom_template_dir(tmp_path: Path) -> Path:
    """Create a temp dir containing a simple .j2 template."""
    tpl_dir = tmp_path / "templates"
    tpl_dir.mkdir()
    (tpl_dir / "hello.j2").write_text("Hello, {{ name }}!\n")
    (tpl_dir / "header.j2").write_text(
        "C  Test file\n{{ n_values }}  / COUNT\n"
    )
    return tpl_dir


@pytest.fixture
def comment_metadata() -> CommentMetadata:
    """CommentMetadata with a header and one section."""
    meta = CommentMetadata(
        source_file="test.in",
        preserve_mode=PreserveMode.FULL,
        header_block=[
            "C**** PRESERVED HEADER ****",
            "C  Original comment",
            "C**** END ****",
        ],
    )
    meta.sections["NODES"] = SectionComments(
        section_name="NODES",
        header_comments=["C  NODES section preserved"],
    )
    return meta


# ---------------------------------------------------------------------------
# TemplateEngine construction
# ---------------------------------------------------------------------------


class TestTemplateEngineCreation:
    """Tests for TemplateEngine constructor paths."""

    def test_default_creation(self) -> None:
        """Default creation with package templates succeeds."""
        engine = TemplateEngine()
        assert engine.env is not None

    def test_custom_template_dir(self, custom_template_dir: Path) -> None:
        """Creation with a custom template directory succeeds."""
        engine = TemplateEngine(template_dir=custom_template_dir)
        assert engine.env is not None

    def test_no_loaders(self) -> None:
        """Creation with no template dirs and no package templates."""
        engine = TemplateEngine(
            template_dir=None, use_package_templates=False
        )
        # Should still support string rendering
        result = engine.render_string("{{ x }}", x=42)
        assert result == "42"


# ---------------------------------------------------------------------------
# render_string
# ---------------------------------------------------------------------------


class TestRenderString:
    """Tests for TemplateEngine.render_string()."""

    def test_simple_template(self, engine: TemplateEngine) -> None:
        result = engine.render_string("Value is {{ val }}", val=99)
        assert result == "Value is 99"

    def test_with_context_variables(self, engine: TemplateEngine) -> None:
        result = engine.render_string(
            "{{ a }} + {{ b }} = {{ a + b }}", a=3, b=7
        )
        assert result == "3 + 7 = 10"

    def test_with_filter(self, engine: TemplateEngine) -> None:
        result = engine.render_string(
            "{{ v | fortran_float(10, 2) }}", v=3.14
        )
        assert "3.14" in result


# ---------------------------------------------------------------------------
# render_template
# ---------------------------------------------------------------------------


class TestRenderTemplate:
    """Tests for TemplateEngine.render_template() from file."""

    def test_render_from_file(self, custom_template_dir: Path) -> None:
        engine = TemplateEngine(template_dir=custom_template_dir)
        result = engine.render_template("hello.j2", name="World")
        assert "Hello, World!" in result


# ---------------------------------------------------------------------------
# render_to_file
# ---------------------------------------------------------------------------


class TestRenderToFile:
    """Tests for TemplateEngine.render_to_file()."""

    def test_creates_file(
        self, custom_template_dir: Path, tmp_path: Path
    ) -> None:
        engine = TemplateEngine(template_dir=custom_template_dir)
        output = tmp_path / "output.txt"
        engine.render_to_file("hello.j2", output, name="pyiwfm")
        assert output.exists()
        assert "Hello, pyiwfm!" in output.read_text()

    def test_creates_parent_dirs(
        self, custom_template_dir: Path, tmp_path: Path
    ) -> None:
        engine = TemplateEngine(template_dir=custom_template_dir)
        output = tmp_path / "sub" / "dir" / "out.txt"
        engine.render_to_file("hello.j2", output, name="deep")
        assert output.exists()


# ---------------------------------------------------------------------------
# render_with_comments
# ---------------------------------------------------------------------------


class TestRenderWithComments:
    """Tests for TemplateEngine.render_with_comments()."""

    def test_without_metadata(self, custom_template_dir: Path) -> None:
        """Without metadata, returns plain rendered content."""
        engine = TemplateEngine(template_dir=custom_template_dir)
        result = engine.render_with_comments(
            "hello.j2", comment_metadata=None, name="Test"
        )
        assert "Hello, Test!" in result

    def test_with_metadata_injects_header(
        self,
        custom_template_dir: Path,
        comment_metadata: CommentMetadata,
    ) -> None:
        """With metadata that has a header, the header is injected."""
        # Create a template that starts with C*****
        (custom_template_dir / "with_header.j2").write_text(
            "C***** TEMPLATE HEADER\nC  template\n100  / DATA\n"
        )
        engine = TemplateEngine(template_dir=custom_template_dir)
        result = engine.render_with_comments(
            "with_header.j2",
            comment_metadata=comment_metadata,
        )
        assert "PRESERVED HEADER" in result

    def test_with_empty_metadata(self, custom_template_dir: Path) -> None:
        """Metadata with no comments returns plain rendered content."""
        empty_meta = CommentMetadata()
        engine = TemplateEngine(template_dir=custom_template_dir)
        result = engine.render_with_comments(
            "hello.j2", comment_metadata=empty_meta, name="Test"
        )
        assert "Hello, Test!" in result


# ---------------------------------------------------------------------------
# render_hybrid
# ---------------------------------------------------------------------------


class TestRenderHybrid:
    """Tests for TemplateEngine.render_hybrid()."""

    def test_with_1d_array(self, engine: TemplateEngine, tmp_path: Path) -> None:
        output = tmp_path / "hybrid1d.dat"
        data = np.array([1.0, 2.0, 3.0])
        engine.render_hybrid(
            "C  Header\n{{ count }}  / COUNT\n",
            output,
            arrays={"vals": (data, "%10.4f")},
            count=3,
        )
        content = output.read_text()
        assert "3  / COUNT" in content
        assert "1.0000" in content
        assert "3.0000" in content

    def test_with_2d_array(self, engine: TemplateEngine, tmp_path: Path) -> None:
        output = tmp_path / "hybrid2d.dat"
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        engine.render_hybrid(
            "C  Data\n",
            output,
            arrays={"matrix": (data, "%8.2f")},
        )
        content = output.read_text()
        assert "1.00" in content
        assert "4.00" in content

    def test_no_arrays(self, engine: TemplateEngine, tmp_path: Path) -> None:
        output = tmp_path / "header_only.dat"
        engine.render_hybrid("C  Just a header\n", output)
        assert output.read_text() == "C  Just a header\n"

    def test_with_j2_file(
        self, custom_template_dir: Path, tmp_path: Path
    ) -> None:
        engine = TemplateEngine(template_dir=custom_template_dir)
        output = tmp_path / "from_j2.dat"
        data = np.array([10.0, 20.0])
        engine.render_hybrid(
            "header.j2",
            output,
            arrays={"vals": (data, "%10.2f")},
            n_values=2,
        )
        content = output.read_text()
        assert "2  / COUNT" in content
        assert "10.00" in content


# ---------------------------------------------------------------------------
# render_hybrid_with_comments
# ---------------------------------------------------------------------------


class TestRenderHybridWithComments:
    """Tests for TemplateEngine.render_hybrid_with_comments()."""

    def test_with_preserved_header(
        self,
        engine: TemplateEngine,
        comment_metadata: CommentMetadata,
        tmp_path: Path,
    ) -> None:
        output = tmp_path / "hybrid_comments.dat"
        data = np.array([1.0])
        engine.render_hybrid_with_comments(
            "C  template header\n",
            output,
            comment_metadata=comment_metadata,
            arrays={"vals": (data, "%10.2f")},
        )
        content = output.read_text()
        assert "PRESERVED HEADER" in content
        assert "1.00" in content

    def test_without_metadata(
        self, engine: TemplateEngine, tmp_path: Path
    ) -> None:
        output = tmp_path / "hybrid_no_meta.dat"
        engine.render_hybrid_with_comments(
            "C  plain header\n",
            output,
            comment_metadata=None,
        )
        assert "plain header" in output.read_text()


# ---------------------------------------------------------------------------
# Filter functions (direct calls)
# ---------------------------------------------------------------------------


class TestFilterFunctions:
    """Tests for standalone filter helper functions."""

    def test_fortran_float_defaults(self) -> None:
        result = _fortran_float(3.14)
        assert "3.140000" in result
        assert len(result) == 14

    def test_fortran_float_custom(self) -> None:
        result = _fortran_float(2.5, width=8, decimals=2)
        assert "2.50" in result
        assert len(result) == 8

    def test_fortran_int_defaults(self) -> None:
        result = _fortran_int(42)
        assert result.strip() == "42"
        assert len(result) == 10

    def test_fortran_int_custom_width(self) -> None:
        result = _fortran_int(7, width=5)
        assert result.strip() == "7"
        assert len(result) == 5

    def test_iwfm_comment(self) -> None:
        assert _iwfm_comment("Hello") == "C  Hello"

    def test_pad_right(self) -> None:
        result = _pad_right("abc", 8)
        assert result == "abc     "
        assert len(result) == 8

    def test_pad_left(self) -> None:
        result = _pad_left("abc", 8)
        assert result == "     abc"
        assert len(result) == 8


# ---------------------------------------------------------------------------
# IWFMFileWriter
# ---------------------------------------------------------------------------


class TestIWFMFileWriterCoverage:
    """Additional IWFMFileWriter tests for uncovered paths."""

    def test_write_nodes_default_ids(self, tmp_path: Path) -> None:
        """write_nodes_file generates sequential IDs when none provided."""
        writer = IWFMFileWriter()
        x = np.array([0.0, 100.0])
        y = np.array([0.0, 100.0])
        output = tmp_path / "nodes.dat"
        writer.write_nodes_file(output, x, y)
        content = output.read_text()
        assert "2" in content  # n_nodes
        assert "NNODES" in content

    def test_write_elements_default_n_subregions(self, tmp_path: Path) -> None:
        """write_elements_file infers n_subregions from max(subregions)."""
        writer = IWFMFileWriter()
        vertices = np.array([[1, 2, 3, 0]], dtype=np.int32)
        subregions = np.array([3], dtype=np.int32)
        output = tmp_path / "elems.dat"
        writer.write_elements_file(output, vertices, subregions)
        content = output.read_text()
        # n_subregions should be 3
        assert "3" in content
        assert "NSUBREGION" in content

    def test_write_stratigraphy_default_ids(self, tmp_path: Path) -> None:
        """write_stratigraphy_file generates sequential node IDs."""
        writer = IWFMFileWriter()
        gs = np.array([100.0, 95.0])
        top = np.array([[100.0], [95.0]])
        bot = np.array([[50.0], [45.0]])
        output = tmp_path / "strat.dat"
        writer.write_stratigraphy_file(output, gs, top, bot)
        content = output.read_text()
        assert "NNODES" in content
        assert "NLAYERS" in content
