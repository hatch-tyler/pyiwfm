"""Tests for Fortran IWFM alignment fixes.

Covers: active-node filtering (2a), case-insensitive matching (2b),
excluded flag filtering (2c), head difference pairs (2d),
subsidence/tiledrain location discovery (2e), IWFM2OBS input file
parser (2f), and multi-layer subsidence summation (2g).
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pyiwfm.calibration.iwfm2obs import (
    HeadDifferencePair,
    compute_head_differences,
    interpolate_batch,
    interpolate_to_obs_times,
    read_head_difference_pairs,
    read_iwfm2obs_config,
)
from pyiwfm.calibration.results_extraction import ExtractionSpec
from pyiwfm.io.smp import SMPTimeSeries


def _make_ts(
    bore_id: str,
    dates: list[str],
    values: list[float],
    excluded: list[bool] | None = None,
) -> SMPTimeSeries:
    times = np.array(dates, dtype="datetime64[D]")
    vals = np.array(values, dtype=np.float64)
    excl = np.array(excluded if excluded else [False] * len(values), dtype=np.bool_)
    return SMPTimeSeries(bore_id=bore_id, times=times, values=vals, excluded=excl)


# ── 2a: Active-node filtering for HEAD Layer=0 ─────────────────────────


class TestActiveNodeFiltering:
    """HEAD Layer=0 should average only active layers (Fortran behavior)."""

    def test_active_layers_mask_excludes_inactive(self) -> None:
        """With active_layers mask, inactive layers are excluded from average."""
        from pyiwfm.calibration.results_extraction import ResultsExtractor

        # 3 timesteps, 3 layers.  Layer 3 is inactive (active_mask=False).
        per_layer = np.array([[10.0, 20.0, 0.0], [30.0, 40.0, 0.0], [50.0, 60.0, 0.0]])

        extractor = object.__new__(ResultsExtractor)
        extractor._data_type = "HEAD"
        extractor._incremental = False
        extractor._t_weights = {}
        # Only layers 0 and 1 are active
        extractor._active_layers = {"test": np.array([True, True, False])}

        spec = ExtractionSpec(name="test", x=0.0, y=0.0, layer=0)
        result = extractor._aggregate_layers(per_layer, spec, 3)

        # Should average only layers 0 and 1: (10+20)/2, (30+40)/2, (50+60)/2
        assert_allclose(result, [15.0, 35.0, 55.0])

    def test_no_active_mask_falls_back_to_nanmean(self) -> None:
        """Without active_layers mask, falls back to all non-NaN layers."""
        from pyiwfm.calibration.results_extraction import ResultsExtractor

        per_layer = np.array([[10.0, 20.0, 30.0]])

        extractor = object.__new__(ResultsExtractor)
        extractor._data_type = "HEAD"
        extractor._incremental = False
        extractor._t_weights = {}
        extractor._active_layers = {}

        spec = ExtractionSpec(name="test", x=0.0, y=0.0, layer=0)
        result = extractor._aggregate_layers(per_layer, spec, 3)

        assert_allclose(result, [20.0])  # mean of 10, 20, 30

    def test_active_mask_with_nan(self) -> None:
        """Active mask combined with NaN values."""
        from pyiwfm.calibration.results_extraction import ResultsExtractor

        per_layer = np.array([[10.0, np.nan, 30.0]])

        extractor = object.__new__(ResultsExtractor)
        extractor._data_type = "HEAD"
        extractor._incremental = False
        extractor._t_weights = {}
        # All 3 layers active, but layer 1 is NaN
        extractor._active_layers = {"test": np.array([True, True, True])}

        spec = ExtractionSpec(name="test", x=0.0, y=0.0, layer=0)
        result = extractor._aggregate_layers(per_layer, spec, 3)

        # NaN excluded: mean of 10, 30 = 20
        assert_allclose(result, [20.0])


# ── 2b: Case-insensitive bore ID matching ──────────────────────────────


class TestCaseInsensitiveMatching:
    def test_mixed_case_matches(self) -> None:
        obs = {"Well_01": _make_ts("Well_01", ["2020-06-15"], [0.0])}
        sim = {"WELL_01": _make_ts("WELL_01", ["2020-01-01", "2020-12-01"], [10.0, 20.0])}
        result = interpolate_batch(obs, sim)
        assert len(result) == 1
        assert "Well_01" in result

    def test_lowercase_sim_matches_uppercase_obs(self) -> None:
        obs = {"PUMP_A": _make_ts("PUMP_A", ["2020-06-15"], [0.0])}
        sim = {"pump_a": _make_ts("pump_a", ["2020-01-01", "2020-12-01"], [5.0, 15.0])}
        result = interpolate_batch(obs, sim)
        assert len(result) == 1
        assert "PUMP_A" in result


# ── 2c: Excluded flag filtering ────────────────────────────────────────


class TestExcludedFlagFiltering:
    def test_excluded_sim_values_not_used_in_interpolation(self) -> None:
        """Simulated values marked as excluded should not affect interpolation."""
        obs = _make_ts("W1", ["2020-07-01"], [0.0])
        # Middle value is excluded — should interpolate between first and last
        sim = _make_ts(
            "W1",
            ["2020-01-01", "2020-07-01", "2021-01-01"],
            [10.0, 999.0, 20.0],
            excluded=[False, True, False],
        )
        result = interpolate_to_obs_times(obs, sim)
        # Without excluded filter, result would be 999.0 (exact match)
        # With excluded filter, interpolates between 10.0 and 20.0 → ~15.0
        assert abs(result.values[0] - 15.0) < 0.1

    def test_no_excluded_values_unchanged(self) -> None:
        """With no excluded values, behavior is unchanged."""
        obs = _make_ts("W1", ["2020-07-01"], [0.0])
        sim = _make_ts("W1", ["2020-01-01", "2021-01-01"], [10.0, 20.0])
        result = interpolate_to_obs_times(obs, sim)
        assert abs(result.values[0] - 15.0) < 0.1


# ── 2d: Head difference pairs ──────────────────────────────────────────


class TestHeadDifferencePairs:
    def test_read_pairs_file(self, tmp_path: Path) -> None:
        pairs_file = tmp_path / "pairs.dat"
        pairs_file.write_text("WELL_A  WELL_B\nWELL_C  WELL_D\n")
        pairs = read_head_difference_pairs(pairs_file)
        assert len(pairs) == 2
        assert pairs[0].id1 == "WELL_A"
        assert pairs[0].id2 == "WELL_B"

    def test_read_pairs_skips_comments(self, tmp_path: Path) -> None:
        pairs_file = tmp_path / "pairs.dat"
        pairs_file.write_text("C This is a comment\nWELL_A  WELL_B\n# Another\n")
        pairs = read_head_difference_pairs(pairs_file)
        assert len(pairs) == 1

    def test_read_pairs_uppercases_ids(self, tmp_path: Path) -> None:
        pairs_file = tmp_path / "pairs.dat"
        pairs_file.write_text("well_a  Well_B\n")
        pairs = read_head_difference_pairs(pairs_file)
        assert pairs[0].id1 == "WELL_A"
        assert pairs[0].id2 == "WELL_B"

    def test_read_pairs_identical_ids_raises(self, tmp_path: Path) -> None:
        pairs_file = tmp_path / "pairs.dat"
        pairs_file.write_text("WELL_A  WELL_A\n")
        with pytest.raises(ValueError, match="identical"):
            read_head_difference_pairs(pairs_file)

    def test_read_pairs_empty_raises(self, tmp_path: Path) -> None:
        pairs_file = tmp_path / "pairs.dat"
        pairs_file.write_text("C comment only\n")
        with pytest.raises(ValueError, match="No pairs"):
            read_head_difference_pairs(pairs_file)

    def test_compute_head_differences(self) -> None:
        ts1 = _make_ts("W1", ["2020-01-01", "2020-07-01"], [100.0, 110.0])
        ts2 = _make_ts("W2", ["2020-01-01", "2020-07-01"], [90.0, 95.0])
        interp = {"W1": ts1, "W2": ts2}
        pairs = [HeadDifferencePair(id1="W1", id2="W2")]
        result = compute_head_differences(interp, pairs)
        assert "W1-W2" in result
        assert_allclose(result["W1-W2"].values, [10.0, 15.0])

    def test_compute_differences_case_insensitive(self) -> None:
        ts1 = _make_ts("well_a", ["2020-01-01"], [50.0])
        ts2 = _make_ts("Well_B", ["2020-01-01"], [30.0])
        interp = {"well_a": ts1, "Well_B": ts2}
        pairs = [HeadDifferencePair(id1="WELL_A", id2="WELL_B")]
        result = compute_head_differences(interp, pairs)
        assert "WELL_A-WELL_B" in result
        assert_allclose(result["WELL_A-WELL_B"].values, [20.0])

    def test_compute_differences_missing_id_warns(self) -> None:
        ts1 = _make_ts("W1", ["2020-01-01"], [50.0])
        interp = {"W1": ts1}
        pairs = [HeadDifferencePair(id1="W1", id2="W_MISSING")]
        result = compute_head_differences(interp, pairs)
        assert len(result) == 0


# ── 2e: Subsidence/TileDrain location discovery ────────────────────────


class TestLocationDiscoveryFields:
    def test_hydrograph_file_info_has_new_fields(self) -> None:
        from pyiwfm.calibration.model_file_discovery import HydrographFileInfo

        info = HydrographFileInfo()
        assert hasattr(info, "subsidence_locations")
        assert hasattr(info, "tiledrain_locations")
        assert info.subsidence_locations == []
        assert info.tiledrain_locations == []


# ── 2f: IWFM2OBS input file parser ─────────────────────────────────────


class TestIWFM2OBSInputFileParser:
    def test_parse_new_format(self, tmp_path: Path) -> None:
        content = textwrap.dedent("""\
            C  Comment line
            Simulation/C2VSimFG.in                          / Simulation main file
            2                                              / Date format
            C  GW block
            model_gw.smp                                   / Model SMP
            obs_gw.smp                                     / Obs SMP
            out_gw.smp                                     / Output SMP
            1.0                                            / Threshold
            gw.ins                                         / INS file
            gw.pcf                                         / PCF file
            C  Stream block
            model_str.smp                                  / Model SMP
            obs_str.smp                                    / Obs SMP
            out_str.smp                                    / Output SMP
            2.0                                            / Threshold
                                                           / INS file
                                                           / PCF file
            C  Tile drain block
                                                           / Model SMP
                                                           / Obs SMP
                                                           / Output SMP
                                                           / Threshold
                                                           / INS file
                                                           / PCF file
            C  Subsidence block
                                                           / Model SMP
                                                           / Obs SMP
                                                           / Output SMP
                                                           / Threshold
                                                           / INS file
                                                           / PCF file
            N                                              / Head differences
            N                                              / Multi-layer
        """)
        config_file = tmp_path / "iwfm2obs.in"
        config_file.write_text(content)
        cfg = read_iwfm2obs_config(config_file)

        assert cfg.simulation_main_file == "Simulation/C2VSimFG.in"
        assert cfg.date_format == 2
        assert cfg.gw.obs_smp == "obs_gw.smp"
        assert cfg.gw.output_smp == "out_gw.smp"
        assert cfg.gw.threshold == 1.0
        assert cfg.gw.ins_file == "gw.ins"
        assert cfg.stream.obs_smp == "obs_str.smp"
        assert cfg.stream.threshold == 2.0
        assert cfg.tiledrain.obs_smp == ""
        assert cfg.subsidence.obs_smp == ""
        assert cfg.head_diff_enabled is False
        assert cfg.multilayer_enabled is False

    def test_parse_old_format(self, tmp_path: Path) -> None:
        """Old format: first data line is date format integer, no sim file."""
        content = textwrap.dedent("""\
            C  Old format
            2                                              / Date format (old)
            C  GW block
            model.smp                                      / Model SMP
            obs.smp                                        / Obs SMP
            out.smp                                        / Output SMP
            1.0                                            / Threshold
                                                           / INS
                                                           / PCF
            C  Stream
                                                           / Model SMP
                                                           / Obs SMP
                                                           / Output SMP
                                                           / Threshold
                                                           / INS
                                                           / PCF
            C  TileDrain
                                                           / Model SMP
                                                           / Obs SMP
                                                           / Output SMP
                                                           / Threshold
                                                           / INS
                                                           / PCF
            C  Subsidence
                                                           / Model SMP
                                                           / Obs SMP
                                                           / Output SMP
                                                           / Threshold
                                                           / INS
                                                           / PCF
            N                                              / Head diff
            N                                              / Multi-layer
        """)
        config_file = tmp_path / "iwfm2obs_old.in"
        config_file.write_text(content)
        cfg = read_iwfm2obs_config(config_file)

        assert cfg.simulation_main_file == ""
        assert cfg.date_format == 2
        assert cfg.gw.model_smp == "model.smp"
        assert cfg.gw.obs_smp == "obs.smp"

    def test_parse_with_head_diff_enabled(self, tmp_path: Path) -> None:
        content = textwrap.dedent("""\
            C  Config
            sim.in                                         / Sim file
            2                                              / Date format
            C GW
            m.smp                                          /
            o.smp                                          /
            out.smp                                        /
            1.0                                            /
                                                           /
                                                           /
            C Stream
                                                           /
                                                           /
                                                           /
                                                           /
                                                           /
                                                           /
            C TD
                                                           /
                                                           /
                                                           /
                                                           /
                                                           /
                                                           /
            C Sub
                                                           /
                                                           /
                                                           /
                                                           /
                                                           /
                                                           /
            Y                                              / Head diff
            pairs.dat                                      / Pair file
            N                                              / Multi-layer
        """)
        config_file = tmp_path / "iwfm2obs_hd.in"
        config_file.write_text(content)
        cfg = read_iwfm2obs_config(config_file)

        assert cfg.head_diff_enabled is True
        assert cfg.head_diff_file == "pairs.dat"
        assert cfg.multilayer_enabled is False


# ── 2g: Multi-layer subsidence summation ────────────────────────────────


class TestCompositeSubsidence:
    def test_compute_composite_subsidence(self) -> None:
        from pyiwfm.calibration.iwfm2obs import compute_composite_subsidence

        layer_values = np.array([1.0, 2.0, 3.0])
        result = compute_composite_subsidence(layer_values)
        assert result == 6.0

    def test_compute_composite_subsidence_with_nan(self) -> None:
        from pyiwfm.calibration.iwfm2obs import compute_composite_subsidence

        layer_values = np.array([1.0, np.nan, 3.0])
        result = compute_composite_subsidence(layer_values)
        assert result == 4.0
