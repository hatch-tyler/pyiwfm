"""Tests for pyiwfm.calibration.pest_files."""

from __future__ import annotations

from pathlib import Path

from pyiwfm.calibration.pest_files import (
    _safe_pest_name,
    generate_multilayer_ins,
    generate_smp_ins,
    generate_thyd_ins,
)


class TestSafePestName:
    """Test _safe_pest_name helper."""

    def test_basic(self) -> None:
        assert _safe_pest_name("WELL_01") == "WELL_01"

    def test_replaces_spaces(self) -> None:
        assert _safe_pest_name("my well") == "my_well"

    def test_replaces_percent(self) -> None:
        assert _safe_pest_name("W1%1") == "W1_1"

    def test_truncates(self) -> None:
        long_name = "A" * 60
        result = _safe_pest_name(long_name)
        assert len(result) == 50

    def test_custom_max_len(self) -> None:
        result = _safe_pest_name("ABCDEF", max_len=4)
        assert result == "ABCD"

    def test_empty_string(self) -> None:
        assert _safe_pest_name("") == ""

    def test_no_truncation_needed(self) -> None:
        assert _safe_pest_name("SHORT") == "SHORT"


class TestGenerateSmpIns:
    """Test generate_smp_ins."""

    def test_basic(self, tmp_path: Path) -> None:
        smp = tmp_path / "test.smp"
        smp.write_text(
            "BORE_A                   01/31/2020  12:00:00     105.50\n"
            "BORE_A                   02/29/2020  12:00:00     106.00\n"
            "BORE_B                   01/31/2020  12:00:00      50.00\n"
        )
        ins = tmp_path / "test.ins"
        n = generate_smp_ins(smp, ins)

        assert n == 3
        lines = ins.read_text().strip().split("\n")
        assert lines[0] == "pif #"
        assert "BORE_A_0001" in lines[1]
        assert "BORE_A_0002" in lines[2]
        assert "BORE_B_0001" in lines[3]

    def test_empty_file(self, tmp_path: Path) -> None:
        smp = tmp_path / "empty.smp"
        smp.write_text("")
        ins = tmp_path / "empty.ins"
        n = generate_smp_ins(smp, ins)
        assert n == 0

    def test_skip_blank_lines(self, tmp_path: Path) -> None:
        smp = tmp_path / "test.smp"
        smp.write_text(
            "BORE_A                   01/31/2020  12:00:00     105.50\n"
            "\n"
            "BORE_A                   02/29/2020  12:00:00     106.00\n"
        )
        ins = tmp_path / "test.ins"
        n = generate_smp_ins(smp, ins)
        assert n == 2

    def test_custom_col_range(self, tmp_path: Path) -> None:
        smp = tmp_path / "test.smp"
        smp.write_text("W1                       01/31/2020  12:00:00     100.00\n")
        ins = tmp_path / "test.ins"
        generate_smp_ins(smp, ins, col_range="40:50")
        content = ins.read_text()
        assert "40:50" in content

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        smp = tmp_path / "test.smp"
        smp.write_text("W1                       01/31/2020  12:00:00     100.00\n")
        ins = tmp_path / "sub" / "dir" / "test.ins"
        generate_smp_ins(smp, ins)
        assert ins.exists()


class TestGenerateThydIns:
    """Test generate_thyd_ins."""

    def test_basic(self, tmp_path: Path) -> None:
        out = tmp_path / "cls1.out"
        out.write_text(
            "    PEST_NAME        DATE                          sim_sub1_cls1\n"
            "TH1_1           01/15/2010            1.234000\n"
            "TH1_2           07/15/2010           -0.567000\n"
        )
        ins = tmp_path / "cls1.ins"
        n = generate_thyd_ins(out, ins)

        assert n == 2
        lines = ins.read_text().strip().split("\n")
        assert lines[0] == "pif #"
        assert lines[1] == "l1"  # skip header
        assert "TH1_1" in lines[2]
        assert "TH1_2" in lines[3]

    def test_skip_blank_lines(self, tmp_path: Path) -> None:
        out = tmp_path / "cls1.out"
        out.write_text(
            "    PEST_NAME        DATE\n"
            "TH1_1           01/15/2010            1.234000\n"
            "\n"
            "TH1_2           07/15/2010           -0.567000\n"
        )
        ins = tmp_path / "cls1.ins"
        n = generate_thyd_ins(out, ins)
        assert n == 2


class TestGenerateMultilayerIns:
    """Test generate_multilayer_ins."""

    def test_basic(self, tmp_path: Path) -> None:
        smp = tmp_path / "ml.smp"
        smp.write_text(
            "W1                       01/31/2020  12:00:00      105.50\n"
            "W1                       02/29/2020  12:00:00      106.00\n"
        )
        ins = tmp_path / "ml.ins"
        n = generate_multilayer_ins(smp, ins)
        assert n == 2

    def test_with_header(self, tmp_path: Path) -> None:
        smp = tmp_path / "ml.smp"
        smp.write_text(
            "Name                     Date        Time        Value\n"
            "W1                       01/31/2020  12:00:00      105.50\n"
        )
        ins = tmp_path / "ml.ins"
        n = generate_multilayer_ins(smp, ins)
        assert n == 1
        lines = ins.read_text().strip().split("\n")
        assert lines[1] == "l1"  # skip header instruction

    def test_seq_resets_per_well(self, tmp_path: Path) -> None:
        smp = tmp_path / "ml.smp"
        smp.write_text(
            "W1                       01/31/2020  12:00:00      100.00\n"
            "W1                       02/29/2020  12:00:00      101.00\n"
            "W2                       01/31/2020  12:00:00       90.00\n"
        )
        ins = tmp_path / "ml.ins"
        generate_multilayer_ins(smp, ins)
        lines = ins.read_text().strip().split("\n")
        assert "W1_00001" in lines[1]
        assert "W1_00002" in lines[2]
        assert "W2_00001" in lines[3]

    def test_special_chars_in_name(self, tmp_path: Path) -> None:
        smp = tmp_path / "ml.smp"
        smp.write_text("WELL A%1                 01/31/2020  12:00:00      100.00\n")
        ins = tmp_path / "ml.ins"
        generate_multilayer_ins(smp, ins)
        content = ins.read_text()
        assert "WELL_A_1" in content
