"""Tests for per-stream-node depletion and model-driven budget lookup
(Phase 2.2.a-iv).

Covers:

- ``BudgetOutputMissingError`` raised when a model didn't declare or didn't
  produce the required budget output
- ``_resolve_budget_file_from_model`` for both reach and node kinds
- ``compute_stream_node_depletion`` end-to-end with mocked BudgetReader
- ``compute_stream_depletion_from_models`` (reach-level, model-driven)
- Exact column matching at all entry points
- ``StreamNodeDepletionResult.to_dict`` / ``StreamNodeDepletionReport.to_dict``
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pyiwfm.core.model import IWFMModel
from pyiwfm.io.streams.depletion import (
    DEFAULT_SA_COLUMN,
    BudgetOutputMissingError,
    StreamNodeDepletionReport,
    StreamNodeDepletionResult,
    _resolve_budget_file_from_model,
    compute_stream_depletion_from_models,
    compute_stream_node_depletion,
)


def _mock_budget_reader(
    n_locations: int,
    flux_baseline: list[list[float]],
    flux_scenario: list[list[float]],
    *,
    sa_column: str = DEFAULT_SA_COLUMN,
    location_label: str = "Node",
) -> tuple[MagicMock, MagicMock]:
    """Build (baseline_reader, scenario_reader) mocks where each location
    has a single ``sa_column`` column with the supplied per-timestep values.
    """
    headers = ["Upstream Inflow (+)", sa_column, "Downstream Outflow (-)"]
    n_ts = len(flux_baseline[0])
    times = np.arange(n_ts, dtype=np.float64)

    def make(flux_per_loc: list[list[float]]) -> MagicMock:
        reader = MagicMock()
        reader.n_locations = n_locations
        reader.locations = [f"{location_label} {i + 1}" for i in range(n_locations)]
        reader.get_column_headers.return_value = headers

        def _get_values(loc_idx: int):
            f = flux_per_loc[loc_idx]
            # Two extra columns surrounding the sa_column so the index
            # extraction has to find the right one
            cols = [
                np.full(n_ts, 100.0),  # Upstream Inflow
                np.array(f, dtype=np.float64),  # SA flux
                np.full(n_ts, -120.0),  # Downstream Outflow
            ]
            return times, np.column_stack(cols)

        reader.get_values.side_effect = _get_values
        reader.header.timestep.start_datetime = None
        return reader

    return make(flux_baseline), make(flux_scenario)


class TestBudgetOutputMissingError:
    def test_message_includes_label_kind_and_remediation(self):
        err = BudgetOutputMissingError(
            model_label="baseline",
            kind="node",
            metadata_key="stream_node_budget_file",
            reason="declared file does not exist on disk: /tmp/missing.hdf",
        )
        msg = str(err)
        assert "baseline model" in msg
        assert "stream node budget" in msg
        assert "stream_node_budget_file" in msg
        assert "/tmp/missing.hdf" in msg
        # Mentions both possible IWFM input lines so the operator knows where to fix
        assert "STRMRCHBUDFL" in msg
        assert "STNDBUDFL" in msg

    def test_inherits_from_value_error(self):
        # ``except ValueError`` should still catch this so generic handlers work
        err = BudgetOutputMissingError("baseline", "reach", "stream_budget_file", "x")
        assert isinstance(err, ValueError)


class TestResolveBudgetFileFromModel:
    def test_reach_path_from_metadata(self, tmp_path: Path):
        budget_file = tmp_path / "BaseStrmBud.hdf"
        budget_file.write_bytes(b"")
        model = IWFMModel(name="m")
        model.metadata["stream_budget_file"] = str(budget_file)

        result = _resolve_budget_file_from_model(model, kind="reach", model_label="baseline")
        assert result == budget_file

    def test_node_path_from_metadata(self, tmp_path: Path):
        budget_file = tmp_path / "ScenStrmNodeBud.hdf"
        budget_file.write_bytes(b"")
        model = IWFMModel(name="m")
        model.metadata["stream_node_budget_file"] = str(budget_file)

        result = _resolve_budget_file_from_model(model, kind="node", model_label="scenario")
        assert result == budget_file

    def test_missing_metadata_raises(self):
        model = IWFMModel(name="m")
        with pytest.raises(BudgetOutputMissingError) as exc_info:
            _resolve_budget_file_from_model(model, kind="reach", model_label="baseline")
        assert exc_info.value.kind == "reach"
        assert "did not declare" in exc_info.value.reason

    def test_missing_file_on_disk_raises(self, tmp_path: Path):
        model = IWFMModel(name="m")
        # Path is declared but the file doesn't exist (simulation didn't run)
        model.metadata["stream_node_budget_file"] = str(tmp_path / "never_created.hdf")
        with pytest.raises(BudgetOutputMissingError) as exc_info:
            _resolve_budget_file_from_model(model, kind="node", model_label="scenario")
        assert "does not exist" in exc_info.value.reason
        assert "never_created.hdf" in str(exc_info.value)


class TestComputeStreamNodeDepletion:
    def _make_models(self, tmp_path: Path) -> tuple[IWFMModel, IWFMModel]:
        baseline_file = tmp_path / "base_node_bud.hdf"
        scenario_file = tmp_path / "scen_node_bud.hdf"
        baseline_file.write_bytes(b"")
        scenario_file.write_bytes(b"")

        baseline = IWFMModel(name="baseline")
        baseline.metadata["stream_node_budget_file"] = str(baseline_file)
        scenario = IWFMModel(name="scenario")
        scenario.metadata["stream_node_budget_file"] = str(scenario_file)
        return baseline, scenario

    def test_per_node_depletion_arithmetic(self, tmp_path: Path):
        baseline, scenario = self._make_models(tmp_path)

        # 3 stream nodes, 4 timesteps. Baseline has more SA gain than scenario,
        # so depletion = base - scen should be positive at every node.
        flux_base = [
            [10.0, 11.0, 12.0, 13.0],
            [20.0, 21.0, 22.0, 23.0],
            [5.0, 5.5, 6.0, 6.5],
        ]
        flux_scen = [
            [8.0, 9.0, 10.0, 11.0],
            [18.0, 19.0, 20.0, 21.0],
            [4.0, 4.5, 5.0, 5.5],
        ]
        base_reader, scen_reader = _mock_budget_reader(3, flux_base, flux_scen)

        with patch("pyiwfm.io.budget.BudgetReader") as MockReader:
            MockReader.side_effect = [base_reader, scen_reader]
            report = compute_stream_node_depletion(baseline, scenario)

        assert isinstance(report, StreamNodeDepletionReport)
        assert report.n_stream_nodes == 3
        assert report.n_timesteps == 4
        # Per-node depletion is base - scen at each timestep
        np.testing.assert_array_almost_equal(report.results[0].depletion, [2.0, 2.0, 2.0, 2.0])
        np.testing.assert_array_almost_equal(report.results[1].depletion, [2.0, 2.0, 2.0, 2.0])
        np.testing.assert_array_almost_equal(report.results[2].depletion, [1.0, 1.0, 1.0, 1.0])
        # Cumulative is the running sum
        np.testing.assert_array_almost_equal(
            report.results[0].cumulative_depletion, [2.0, 4.0, 6.0, 8.0]
        )
        # Per-node ID matches location index + 1
        assert report.results[0].stream_node_id == 1
        assert report.results[2].stream_node_id == 3

    def test_node_ids_filter(self, tmp_path: Path):
        baseline, scenario = self._make_models(tmp_path)
        flux = [[1.0, 2.0]] * 5
        base_reader, scen_reader = _mock_budget_reader(5, flux, flux)

        with patch("pyiwfm.io.budget.BudgetReader") as MockReader:
            MockReader.side_effect = [base_reader, scen_reader]
            report = compute_stream_node_depletion(baseline, scenario, node_ids=[2, 4])

        assert report.n_stream_nodes == 2
        assert [r.stream_node_id for r in report.results] == [2, 4]

    def test_unknown_node_id_raises(self, tmp_path: Path):
        baseline, scenario = self._make_models(tmp_path)
        flux = [[1.0]] * 3
        base_reader, scen_reader = _mock_budget_reader(3, flux, flux)

        with patch("pyiwfm.io.budget.BudgetReader") as MockReader:
            MockReader.side_effect = [base_reader, scen_reader]
            with pytest.raises(ValueError, match=r"\[99\] are not present"):
                compute_stream_node_depletion(baseline, scenario, node_ids=[1, 99])

    def test_missing_baseline_budget_raises(self, tmp_path: Path):
        # baseline doesn't declare a node budget; scenario does
        scen_file = tmp_path / "scen.hdf"
        scen_file.write_bytes(b"")
        baseline = IWFMModel(name="baseline")
        scenario = IWFMModel(name="scenario")
        scenario.metadata["stream_node_budget_file"] = str(scen_file)

        with pytest.raises(BudgetOutputMissingError) as exc_info:
            compute_stream_node_depletion(baseline, scenario)
        # The specific failure points at the baseline model
        assert exc_info.value.model_label == "baseline"
        assert exc_info.value.kind == "node"

    def test_missing_scenario_budget_raises(self, tmp_path: Path):
        base_file = tmp_path / "base.hdf"
        base_file.write_bytes(b"")
        baseline = IWFMModel(name="baseline")
        baseline.metadata["stream_node_budget_file"] = str(base_file)
        scenario = IWFMModel(name="scenario")  # no metadata

        with pytest.raises(BudgetOutputMissingError) as exc_info:
            compute_stream_node_depletion(baseline, scenario)
        assert exc_info.value.model_label == "scenario"

    def test_missing_sa_column_raises_with_helpful_message(self, tmp_path: Path):
        baseline, scenario = self._make_models(tmp_path)
        flux = [[1.0, 2.0]] * 2
        # Use a non-default sa_column name so the lookup fails
        base_reader, scen_reader = _mock_budget_reader(2, flux, flux, sa_column="My Custom Column")

        with patch("pyiwfm.io.budget.BudgetReader") as MockReader:
            MockReader.side_effect = [base_reader, scen_reader]
            with pytest.raises(KeyError, match="not found in budget headers"):
                compute_stream_node_depletion(baseline, scenario)
            # Message should suggest the override
            with pytest.raises(KeyError, match="sa_column"):
                MockReader.side_effect = [
                    *_mock_budget_reader(2, flux, flux, sa_column="My Custom Column")
                ]
                compute_stream_node_depletion(baseline, scenario)

    def test_explicit_sa_column_works(self, tmp_path: Path):
        baseline, scenario = self._make_models(tmp_path)
        flux_base = [[5.0, 5.0]] * 2
        flux_scen = [[3.0, 3.0]] * 2
        # Build readers whose SA column has a non-default name
        base_reader, scen_reader = _mock_budget_reader(
            2, flux_base, flux_scen, sa_column="Gain from GW (+)"
        )

        with patch("pyiwfm.io.budget.BudgetReader") as MockReader:
            MockReader.side_effect = [base_reader, scen_reader]
            report = compute_stream_node_depletion(baseline, scenario, sa_column="Gain from GW (+)")
        np.testing.assert_array_almost_equal(report.results[0].depletion, [2.0, 2.0])


class TestComputeStreamDepletionFromModels:
    """Same model-driven entry point but for reach-level depletion."""

    def test_reads_reach_budget_path_from_metadata(self, tmp_path: Path):
        base_file = tmp_path / "base_reach.hdf"
        scen_file = tmp_path / "scen_reach.hdf"
        base_file.write_bytes(b"")
        scen_file.write_bytes(b"")
        baseline = IWFMModel(name="baseline")
        baseline.metadata["stream_budget_file"] = str(base_file)
        scenario = IWFMModel(name="scenario")
        scenario.metadata["stream_budget_file"] = str(scen_file)

        flux_base = [[10.0, 11.0, 12.0]]
        flux_scen = [[7.0, 8.0, 9.0]]
        base_reader, scen_reader = _mock_budget_reader(
            1, flux_base, flux_scen, location_label="Reach"
        )

        with patch("pyiwfm.io.budget.BudgetReader") as MockReader:
            MockReader.side_effect = [base_reader, scen_reader]
            report = compute_stream_depletion_from_models(baseline, scenario)
        assert report.n_reaches == 1
        np.testing.assert_array_almost_equal(report.results[0].depletion, [3.0, 3.0, 3.0])

    def test_missing_reach_budget_raises(self):
        baseline = IWFMModel(name="baseline")  # no metadata
        scenario = IWFMModel(name="scenario")
        with pytest.raises(BudgetOutputMissingError) as exc_info:
            compute_stream_depletion_from_models(baseline, scenario)
        assert exc_info.value.kind == "reach"


class TestStreamNodeDepletionDataclasses:
    def test_result_to_dict_round_trip(self):
        result = StreamNodeDepletionResult(
            stream_node_id=42,
            times=["2024-01", "2024-02"],
            baseline_sa_flux=np.array([10.0, 11.0]),
            scenario_sa_flux=np.array([8.0, 9.0]),
            depletion=np.array([2.0, 2.0]),
            cumulative_depletion=np.array([2.0, 4.0]),
            max_depletion=2.0,
            max_depletion_timestep=0,
            total_depletion=4.0,
        )
        d = result.to_dict()
        assert d["stream_node_id"] == 42
        assert d["max_depletion"] == 2.0
        assert d["depletion"] == [2.0, 2.0]
        assert d["cumulative_depletion"] == [2.0, 4.0]

    def test_report_to_dict_includes_all_nodes(self):
        n1 = StreamNodeDepletionResult(
            stream_node_id=1,
            times=["t1"],
            baseline_sa_flux=np.array([5.0]),
            scenario_sa_flux=np.array([3.0]),
            depletion=np.array([2.0]),
            cumulative_depletion=np.array([2.0]),
            max_depletion=2.0,
            max_depletion_timestep=0,
            total_depletion=2.0,
        )
        report = StreamNodeDepletionReport(
            results=[n1],
            n_stream_nodes=1,
            n_timesteps=1,
            total_max_depletion=2.0,
            total_cumulative_depletion=2.0,
        )
        d = report.to_dict()
        assert d["n_stream_nodes"] == 1
        assert len(d["stream_nodes"]) == 1  # type: ignore[arg-type]
        assert d["total_max_depletion"] == 2.0
