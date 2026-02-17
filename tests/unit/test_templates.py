"""Unit tests for template engine."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# Skip all tests if jinja2 is not available
pytest.importorskip("jinja2")

from pyiwfm.templates.engine import IWFMFileWriter, TemplateEngine


class TestTemplateEngine:
    """Tests for TemplateEngine."""

    def test_render_string(self) -> None:
        """Test rendering a template string."""
        engine = TemplateEngine()

        result = engine.render_string("Hello, {{ name }}!", name="World")

        assert result == "Hello, World!"

    def test_render_string_with_filters(self) -> None:
        """Test rendering with custom filters."""
        engine = TemplateEngine()

        result = engine.render_string(
            "Value: {{ value | fortran_float(10, 2) }}",
            value=3.14159,
        )

        assert "3.14" in result

    def test_render_to_file(self, tmp_path: Path) -> None:
        """Test rendering to a file using inline template."""
        engine = TemplateEngine()

        # Create a simple template file
        template_dir = tmp_path / "templates"
        template_dir.mkdir()
        template_file = template_dir / "test.j2"
        template_file.write_text("Hello, {{ name }}!\nValue: {{ value }}")

        # Create engine with custom template dir
        engine = TemplateEngine(template_dir=template_dir)

        output_file = tmp_path / "output.txt"
        engine.render_to_file("test.j2", output_file, name="World", value=42)

        assert output_file.exists()
        content = output_file.read_text()
        assert "Hello, World!" in content
        assert "Value: 42" in content

    def test_render_hybrid(self, tmp_path: Path) -> None:
        """Test hybrid rendering with header and arrays."""
        engine = TemplateEngine()

        header = """C  Test file
{{ n_values }}                         / COUNT
"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        output_file = tmp_path / "hybrid.dat"
        engine.render_hybrid(
            header,
            output_file,
            arrays={"data": (data, "%10.4f")},
            n_values=len(data),
        )

        assert output_file.exists()
        content = output_file.read_text()
        assert "5" in content  # n_values
        assert "1.0000" in content
        assert "5.0000" in content


class TestIWFMFileWriter:
    """Tests for IWFMFileWriter."""

    def test_write_nodes_file(self, tmp_path: Path) -> None:
        """Test writing a nodes file - check file structure."""
        writer = IWFMFileWriter()

        x = np.array([0.0, 100.0, 200.0, 0.0, 100.0, 200.0])
        y = np.array([0.0, 0.0, 0.0, 100.0, 100.0, 100.0])

        output_file = tmp_path / "nodes.dat"
        writer.write_nodes_file(output_file, x, y)

        # Check file structure
        content = output_file.read_text()
        content.strip().split("\n")

        # Should have header + 1 count line + 6 data lines
        assert "NNODES" in content
        assert "6" in content  # n_nodes count
        # Check data is present
        assert "0.000000" in content
        assert "100.000000" in content
        assert "200.000000" in content

    def test_write_elements_file(self, tmp_path: Path) -> None:
        """Test writing an elements file - check file structure."""
        writer = IWFMFileWriter()

        vertices = np.array(
            [
                [1, 2, 5, 4],
                [2, 3, 6, 5],
            ],
            dtype=np.int32,
        )
        subregions = np.array([1, 1], dtype=np.int32)

        output_file = tmp_path / "elements.dat"
        writer.write_elements_file(output_file, vertices, subregions)

        # Check file structure
        content = output_file.read_text()

        assert "NELEM" in content
        assert "NSUBREGION" in content
        assert "2" in content  # n_elements

    def test_write_elements_with_triangles(self, tmp_path: Path) -> None:
        """Test writing elements file with triangles - check structure."""
        writer = IWFMFileWriter()

        # Triangle has 0 in 4th position
        vertices = np.array(
            [
                [1, 2, 3, 0],
                [2, 4, 3, 0],
            ],
            dtype=np.int32,
        )
        subregions = np.array([1, 1], dtype=np.int32)

        output_file = tmp_path / "tri_elements.dat"
        writer.write_elements_file(output_file, vertices, subregions)

        content = output_file.read_text()
        assert "NELEM" in content
        # Check the 0 vertex is present (for triangles)
        assert "    0" in content

    def test_write_stratigraphy_file(self, tmp_path: Path) -> None:
        """Test writing a stratigraphy file - check file structure."""
        writer = IWFMFileWriter()

        gs_elev = np.array([100.0, 100.0, 100.0, 100.0])
        top_elev = np.array([[100.0, 50.0], [100.0, 50.0], [100.0, 50.0], [100.0, 50.0]])
        bottom_elev = np.array([[50.0, 0.0], [50.0, 0.0], [50.0, 0.0], [50.0, 0.0]])

        output_file = tmp_path / "stratigraphy.dat"
        writer.write_stratigraphy_file(output_file, gs_elev, top_elev, bottom_elev)

        # Check file structure
        content = output_file.read_text()

        assert "NNODES" in content
        assert "NLAYERS" in content
        assert "4" in content  # n_nodes
        assert "2" in content  # n_layers
        assert "100.0000" in content  # gs_elev
        assert "50.0000" in content  # layer top/bottom


class TestTemplateFilters:
    """Tests for template filters."""

    def test_fortran_float(self) -> None:
        """Test fortran_float filter."""
        engine = TemplateEngine()

        result = engine.render_string("{{ 3.14159 | fortran_float(12, 4) }}")

        assert result.strip() == "3.1416"

    def test_fortran_int(self) -> None:
        """Test fortran_int filter."""
        engine = TemplateEngine()

        result = engine.render_string("{{ 42 | fortran_int(8) }}")

        assert result.strip() == "42"
        assert len(result) == 8

    def test_iwfm_comment(self) -> None:
        """Test iwfm_comment filter."""
        engine = TemplateEngine()

        result = engine.render_string("{{ 'Test comment' | iwfm_comment }}")

        assert result == "C  Test comment"

    def test_pad_right(self) -> None:
        """Test pad_right filter."""
        engine = TemplateEngine()

        result = engine.render_string("{{ 'test' | pad_right(10) }}")

        assert result == "test      "
        assert len(result) == 10

    def test_pad_left(self) -> None:
        """Test pad_left filter."""
        engine = TemplateEngine()

        result = engine.render_string("{{ 'test' | pad_left(10) }}")

        assert result == "      test"
        assert len(result) == 10


# ── Additional tests for increased coverage ──────────────────────────


from datetime import datetime as dt  # noqa: E402

from pyiwfm.templates.engine import (  # noqa: E402
    _dss_pathname,
    _fortran_float,
    _fortran_int,
    _iwfm_array_row,
    _iwfm_comment,
    _iwfm_timestamp,
    _pad_left,
    _pad_right,
    _timeseries_ref,
)


class TestFilterFunctionsDirect:
    """Direct tests for filter helper functions."""

    def test_fortran_float_default(self) -> None:
        """Test _fortran_float with default width/decimals."""
        result = _fortran_float(3.14159)
        assert "3.141590" in result
        assert len(result) == 14

    def test_fortran_float_custom(self) -> None:
        """Test _fortran_float with custom width/decimals."""
        result = _fortran_float(2.5, width=8, decimals=2)
        assert "2.50" in result
        assert len(result) == 8

    def test_fortran_int_default(self) -> None:
        """Test _fortran_int with default width."""
        result = _fortran_int(42)
        assert result.strip() == "42"
        assert len(result) == 10

    def test_fortran_int_custom(self) -> None:
        """Test _fortran_int with custom width."""
        result = _fortran_int(7, width=5)
        assert result.strip() == "7"
        assert len(result) == 5

    def test_iwfm_comment(self) -> None:
        """Test _iwfm_comment prefixes with 'C  '."""
        result = _iwfm_comment("hello")
        assert result == "C  hello"

    def test_pad_right_function(self) -> None:
        """Test _pad_right function."""
        result = _pad_right("abc", 8)
        assert result == "abc     "

    def test_pad_left_function(self) -> None:
        """Test _pad_left function."""
        result = _pad_left("abc", 8)
        assert result == "     abc"

    def test_iwfm_timestamp_string(self) -> None:
        """Test _iwfm_timestamp with a plain string input."""
        result = _iwfm_timestamp("09/30/2020_24:00")
        assert result.startswith("09/30/2020_24:00")
        assert len(result) == 16

    def test_iwfm_timestamp_datetime(self) -> None:
        """Test _iwfm_timestamp with a datetime object."""
        d = dt(2020, 10, 1, 12, 30, 0)
        result = _iwfm_timestamp(d)
        assert "10/01/2020_12:30" in result
        assert len(result) == 16

    def test_iwfm_timestamp_numpy_datetime64(self) -> None:
        """Test _iwfm_timestamp with a numpy datetime64."""
        np_dt = np.datetime64("2020-10-01T12:30:00")
        result = _iwfm_timestamp(np_dt)
        assert "10/01/2020" in result
        assert len(result) == 16

    def test_dss_pathname(self) -> None:
        """Test _dss_pathname builds correct DSS path."""
        result = _dss_pathname(
            a_part="BASIN",
            b_part="LOC1",
            c_part="FLOW",
            d_part="01JAN2020",
            e_part="1DAY",
            f_part="V1",
        )
        assert result == "/BASIN/LOC1/FLOW/01JAN2020/1DAY/V1/"

    def test_dss_pathname_empty_parts(self) -> None:
        """Test _dss_pathname with default empty parts."""
        result = _dss_pathname()
        assert result == "///////"

    def test_timeseries_ref(self) -> None:
        """Test _timeseries_ref formats file reference correctly."""
        result = _timeseries_ref("path/to/file.dat", column=2, factor=1.5)
        assert "path/to/file.dat" in result
        assert "2" in result
        assert "1.5000" in result

    def test_timeseries_ref_defaults(self) -> None:
        """Test _timeseries_ref with default column and factor."""
        result = _timeseries_ref("data.dat")
        assert "data.dat" in result
        assert "1" in result
        assert "1.0000" in result

    def test_iwfm_array_row_list(self) -> None:
        """Test _iwfm_array_row with a list of values."""
        result = _iwfm_array_row([1.0, 2.0, 3.0], fmt="%10.4f")
        assert "1.0000" in result
        assert "2.0000" in result
        assert "3.0000" in result

    def test_iwfm_array_row_numpy(self) -> None:
        """Test _iwfm_array_row with a numpy array."""
        arr = np.array([10.0, 20.0, 30.0])
        result = _iwfm_array_row(arr, fmt="%8.2f")
        assert "10.00" in result
        assert "20.00" in result
        assert "30.00" in result

    def test_iwfm_array_row_custom_separator(self) -> None:
        """Test _iwfm_array_row with a custom separator."""
        result = _iwfm_array_row([1.0, 2.0], fmt="%5.1f", sep=",")
        assert "," in result


class TestTemplateEngineExtended:
    """Extended tests for TemplateEngine covering uncovered paths."""

    def test_engine_no_loaders(self) -> None:
        """Test engine with no template dirs and use_package_templates=False."""
        engine = TemplateEngine(
            template_dir=None,
            use_package_templates=False,
        )
        # Should still work for string rendering
        result = engine.render_string("Hello {{ name }}", name="Test")
        assert result == "Hello Test"

    def test_render_hybrid_with_j2_template(self, tmp_path: Path) -> None:
        """Test render_hybrid with a .j2 file template."""
        # Create template directory and file
        template_dir = tmp_path / "templates"
        template_dir.mkdir()
        template_file = template_dir / "header.j2"
        template_file.write_text("C  Header with {{ count }} records\n")

        engine = TemplateEngine(template_dir=template_dir)

        data = np.array([1.0, 2.0, 3.0])
        output_file = tmp_path / "output.dat"

        engine.render_hybrid(
            "header.j2",
            output_file,
            arrays={"values": (data, "%10.4f")},
            count=3,
        )

        content = output_file.read_text()
        assert "Header with 3 records" in content
        assert "1.0000" in content
        assert "3.0000" in content

    def test_render_hybrid_with_2d_array(self, tmp_path: Path) -> None:
        """Test render_hybrid with a 2D numpy array."""
        engine = TemplateEngine()

        header = "C  Data\n"
        data_2d = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        output_file = tmp_path / "matrix.dat"

        engine.render_hybrid(
            header,
            output_file,
            arrays={"matrix": (data_2d, "%8.2f")},
        )

        content = output_file.read_text()
        assert "C  Data" in content
        assert "1.00" in content
        assert "6.00" in content

    def test_render_hybrid_no_arrays(self, tmp_path: Path) -> None:
        """Test render_hybrid with no arrays (header only)."""
        engine = TemplateEngine()

        output_file = tmp_path / "header_only.dat"
        engine.render_hybrid("C  Just a header\n", output_file)

        content = output_file.read_text()
        assert "Just a header" in content

    def test_render_to_file_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Test render_to_file creates parent directories."""
        template_dir = tmp_path / "tpl"
        template_dir.mkdir()
        (template_dir / "t.j2").write_text("Hello {{ name }}")

        engine = TemplateEngine(template_dir=template_dir)
        output = tmp_path / "sub" / "dir" / "output.txt"
        engine.render_to_file("t.j2", output, name="World")

        assert output.exists()
        assert output.read_text() == "Hello World"


class TestIWFMFileWriterExtended:
    """Extended tests for IWFMFileWriter."""

    def test_write_nodes_file_custom_ids(self, tmp_path: Path) -> None:
        """Test writing nodes file with custom node IDs."""
        writer = IWFMFileWriter()

        x = np.array([0.0, 100.0, 200.0])
        y = np.array([0.0, 0.0, 0.0])
        node_ids = np.array([10, 20, 30], dtype=np.int32)

        output_file = tmp_path / "nodes_custom.dat"
        writer.write_nodes_file(output_file, x, y, node_ids=node_ids)

        content = output_file.read_text()
        assert "3" in content  # n_nodes
        assert "NNODES" in content
        # Check custom IDs appear
        assert "10" in content
        assert "20" in content
        assert "30" in content

    def test_write_elements_file_custom_ids_and_subregions(self, tmp_path: Path) -> None:
        """Test writing elements file with custom element IDs and explicit n_subregions."""
        writer = IWFMFileWriter()

        vertices = np.array([[1, 2, 5, 4], [2, 3, 6, 5]], dtype=np.int32)
        subregions = np.array([1, 2], dtype=np.int32)
        element_ids = np.array([100, 200], dtype=np.int32)

        output_file = tmp_path / "elems_custom.dat"
        writer.write_elements_file(
            output_file,
            vertices,
            subregions,
            element_ids=element_ids,
            n_subregions=5,
        )

        content = output_file.read_text()
        assert "2" in content  # n_elements
        assert "5" in content  # n_subregions
        assert "NELEM" in content
        assert "NSUBREGION" in content

    def test_write_stratigraphy_file_custom_ids(self, tmp_path: Path) -> None:
        """Test writing stratigraphy file with custom node IDs."""
        writer = IWFMFileWriter()

        gs_elev = np.array([100.0, 95.0, 90.0])
        top_elev = np.array([[100.0, 50.0], [95.0, 45.0], [90.0, 40.0]])
        bottom_elev = np.array([[50.0, 0.0], [45.0, -5.0], [40.0, -10.0]])
        node_ids = np.array([10, 20, 30], dtype=np.int32)

        output_file = tmp_path / "strat_custom.dat"
        writer.write_stratigraphy_file(
            output_file, gs_elev, top_elev, bottom_elev, node_ids=node_ids
        )

        content = output_file.read_text()
        assert "3" in content  # n_nodes
        assert "2" in content  # n_layers
        assert "NNODES" in content
        assert "NLAYERS" in content
