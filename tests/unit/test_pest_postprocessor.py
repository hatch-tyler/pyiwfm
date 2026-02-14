"""Unit tests for PEST++ post-processor."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from pyiwfm.runner.pest_postprocessor import (
    PestPostProcessor,
    CalibrationResults,
    ResidualData,
    SensitivityData,
)


class TestResidualData:
    """Tests for ResidualData."""

    @pytest.fixture
    def sample_residuals(self):
        """Create sample residual data."""
        return ResidualData(
            names=["obs1", "obs2", "obs3"],
            groups=["head", "head", "flow"],
            observed=np.array([100.0, 200.0, 50.0]),
            simulated=np.array([98.0, 205.0, 48.0]),
            residuals=np.array([2.0, -5.0, 2.0]),
            weights=np.array([1.0, 1.0, 0.5]),
        )

    def test_n_observations(self, sample_residuals):
        """Test observation count."""
        assert sample_residuals.n_observations == 3

    def test_weighted_residuals(self, sample_residuals):
        """Test weighted residuals computation."""
        wr = sample_residuals.weighted_residuals
        assert wr[0] == pytest.approx(2.0)
        assert wr[1] == pytest.approx(-5.0)
        assert wr[2] == pytest.approx(1.0)

    def test_phi(self, sample_residuals):
        """Test objective function computation."""
        phi = sample_residuals.phi
        expected = 2.0**2 + 5.0**2 + 1.0**2  # 4 + 25 + 1 = 30
        assert phi == pytest.approx(expected)

    def test_group_phi(self, sample_residuals):
        """Test per-group phi."""
        group_phi = sample_residuals.group_phi()
        assert "head" in group_phi
        assert "flow" in group_phi
        assert group_phi["head"] == pytest.approx(4.0 + 25.0)  # 2^2 + 5^2
        assert group_phi["flow"] == pytest.approx(1.0)  # 1^2


class TestSensitivityData:
    """Tests for SensitivityData."""

    @pytest.fixture
    def sample_sensitivity(self):
        """Create sample sensitivity data."""
        return SensitivityData(
            parameter_names=["hk_z1", "sy_z1", "pump", "strk"],
            composite_sensitivities=np.array([10.0, 2.0, 50.0, 0.5]),
        )

    def test_n_parameters(self, sample_sensitivity):
        """Test parameter count."""
        assert sample_sensitivity.n_parameters == 4

    def test_most_sensitive(self, sample_sensitivity):
        """Test getting most sensitive parameters."""
        top = sample_sensitivity.most_sensitive(2)
        assert len(top) == 2
        assert top[0][0] == "pump"
        assert top[0][1] == pytest.approx(50.0)
        assert top[1][0] == "hk_z1"

    def test_least_sensitive(self, sample_sensitivity):
        """Test getting least sensitive parameters."""
        bottom = sample_sensitivity.least_sensitive(2)
        assert len(bottom) == 2
        assert bottom[0][0] == "strk"
        assert bottom[0][1] == pytest.approx(0.5)


class TestCalibrationResults:
    """Tests for CalibrationResults."""

    def test_basic_creation(self):
        """Test basic creation."""
        results = CalibrationResults(case_name="test")
        assert results.case_name == "test"
        assert results.n_iterations == 0
        assert results.final_phi == 0.0

    def test_fit_statistics(self):
        """Test computing fit statistics."""
        results = CalibrationResults(
            case_name="test",
            residuals=ResidualData(
                names=["o1", "o2", "o3", "o4"],
                groups=["head"] * 4,
                observed=np.array([100.0, 200.0, 150.0, 120.0]),
                simulated=np.array([102.0, 198.0, 155.0, 118.0]),
                residuals=np.array([-2.0, 2.0, -5.0, 2.0]),
                weights=np.ones(4),
            ),
        )

        stats = results.fit_statistics()
        assert "rmse" in stats
        assert "mae" in stats
        assert "r_squared" in stats
        assert "nse" in stats
        assert "bias" in stats
        assert stats["n_observations"] == 4
        assert stats["rmse"] > 0
        assert stats["r_squared"] >= 0

    def test_fit_statistics_by_group(self):
        """Test fit statistics for specific group."""
        results = CalibrationResults(
            case_name="test",
            residuals=ResidualData(
                names=["h1", "h2", "f1"],
                groups=["head", "head", "flow"],
                observed=np.array([100.0, 200.0, 50.0]),
                simulated=np.array([102.0, 198.0, 48.0]),
                residuals=np.array([-2.0, 2.0, 2.0]),
                weights=np.ones(3),
            ),
        )

        head_stats = results.fit_statistics(group="head")
        assert head_stats["n_observations"] == 2

        flow_stats = results.fit_statistics(group="flow")
        assert flow_stats["n_observations"] == 1

    def test_fit_statistics_no_residuals(self):
        """Test fit statistics with no residuals."""
        results = CalibrationResults(case_name="test")
        stats = results.fit_statistics()
        assert stats == {}


class TestPestPostProcessorInit:
    """Tests for PestPostProcessor initialization."""

    def test_basic_init(self):
        """Test basic initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pp = PestPostProcessor(pest_dir=tmpdir, case_name="test")
            assert pp.case_name == "test"
            assert pp.pest_dir == Path(tmpdir)

    def test_repr(self):
        """Test string representation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pp = PestPostProcessor(pest_dir=tmpdir, case_name="test")
            r = repr(pp)
            assert "PestPostProcessor" in r
            assert "test" in r


class TestLoadResults:
    """Tests for loading PEST++ output files."""

    def test_load_empty_results(self):
        """Test loading with no output files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pp = PestPostProcessor(pest_dir=tmpdir, case_name="test")
            results = pp.load_results()
            assert results.case_name == "test"
            assert results.residuals is None
            assert results.sensitivities is None

    def test_load_residual_file(self):
        """Test loading .rei residual file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a .rei file
            rei_content = (
                "Name Group Measured Modelled Residual Weight\n"
                "obs1 head 100.0 98.0 2.0 1.0\n"
                "obs2 head 200.0 205.0 -5.0 1.0\n"
                "obs3 flow 50.0 48.0 2.0 0.5\n"
            )
            rei_path = Path(tmpdir) / "test.rei"
            rei_path.write_text(rei_content)

            pp = PestPostProcessor(pest_dir=tmpdir, case_name="test")
            results = pp.load_results()

            assert results.residuals is not None
            assert results.residuals.n_observations == 3
            assert results.residuals.names[0] == "obs1"

    def test_load_sensitivity_file(self):
        """Test loading .sen sensitivity file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sen_content = (
                "Parameter_Name CSS\n"
                "hk_z1 10.5\n"
                "sy_z1 2.3\n"
                "pump 45.2\n"
            )
            sen_path = Path(tmpdir) / "test.sen"
            sen_path.write_text(sen_content)

            pp = PestPostProcessor(pest_dir=tmpdir, case_name="test")
            results = pp.load_results()

            assert results.sensitivities is not None
            assert results.sensitivities.n_parameters == 3
            top = results.sensitivities.most_sensitive(1)
            assert top[0][0] == "pump"

    def test_load_iteration_history(self):
        """Test loading .iobj iteration file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            iobj_content = (
                "iteration total_phi\n"
                "0 1000.0\n"
                "1 500.0\n"
                "2 250.0\n"
                "3 200.0\n"
            )
            iobj_path = Path(tmpdir) / "test.iobj"
            iobj_path.write_text(iobj_content)

            pp = PestPostProcessor(pest_dir=tmpdir, case_name="test")
            results = pp.load_results()

            assert results.n_iterations == 4
            assert results.iteration_phi[0] == pytest.approx(1000.0)
            assert results.iteration_phi[-1] == pytest.approx(200.0)

    def test_load_calibrated_parameters(self):
        """Test loading .par parameter file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            par_content = (
                "hk_z1 1.5e-04 1.0 0.0\n"
                "sy_z1 1.8e-01 1.0 0.0\n"
            )
            par_path = Path(tmpdir) / "test.par"
            par_path.write_text(par_content)

            pp = PestPostProcessor(pest_dir=tmpdir, case_name="test")
            results = pp.load_results()

            assert len(results.calibrated_values) == 2
            assert "hk_z1" in results.calibrated_values
            assert results.calibrated_values["hk_z1"] == pytest.approx(1.5e-04)


class TestExportAndAnalysis:
    """Tests for export and analysis methods."""

    def test_export_calibrated_parameters_csv(self):
        """Test exporting calibrated parameters as CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create .par file
            par_content = "hk_z1 1.5e-04 1.0 0.0\nsy_z1 0.18 1.0 0.0\n"
            (Path(tmpdir) / "test.par").write_text(par_content)

            pp = PestPostProcessor(pest_dir=tmpdir, case_name="test")
            output = Path(tmpdir) / "calibrated.csv"
            pp.export_calibrated_parameters(output, format="csv")

            assert output.exists()
            content = output.read_text()
            assert "parameter_name" in content
            assert "hk_z1" in content

    def test_export_calibrated_parameters_pest(self):
        """Test exporting calibrated parameters as PEST format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            par_content = "hk_z1 1.5e-04 1.0 0.0\n"
            (Path(tmpdir) / "test.par").write_text(par_content)

            pp = PestPostProcessor(pest_dir=tmpdir, case_name="test")
            output = Path(tmpdir) / "calibrated.par"
            pp.export_calibrated_parameters(output, format="pest")

            assert output.exists()
            content = output.read_text()
            assert "single point" in content

    def test_identifiability_no_data(self):
        """Test identifiability with no sensitivity data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pp = PestPostProcessor(pest_dir=tmpdir, case_name="test")
            result = pp.compute_identifiability()
            assert result is None

    def test_identifiability_with_data(self):
        """Test identifiability with sensitivity data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sen_content = (
                "Parameter_Name CSS\n"
                "hk_z1 10.0\n"
                "sy_z1 5.0\n"
                "pump 2.0\n"
            )
            (Path(tmpdir) / "test.sen").write_text(sen_content)

            pp = PestPostProcessor(pest_dir=tmpdir, case_name="test")
            ident = pp.compute_identifiability()

            assert ident is not None
            assert ident["hk_z1"] == pytest.approx(1.0)  # Most sensitive
            assert ident["sy_z1"] == pytest.approx(0.5)
            assert ident["pump"] == pytest.approx(0.2)

    def test_summary_report(self):
        """Test generating summary report."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create minimal output files
            rei_content = (
                "Name Group Measured Modelled Residual Weight\n"
                "obs1 head 100.0 98.0 2.0 1.0\n"
            )
            (Path(tmpdir) / "test.rei").write_text(rei_content)

            pp = PestPostProcessor(pest_dir=tmpdir, case_name="test")
            report = pp.summary_report()

            assert "test" in report
            assert "Fit Statistics" in report

    def test_summary_report_empty(self):
        """Test summary report with no data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pp = PestPostProcessor(pest_dir=tmpdir, case_name="test")
            report = pp.summary_report()
            assert "test" in report


class TestResidualDataEdgeCases:
    """Edge-case tests for ResidualData."""

    def test_single_observation(self):
        """Test with a single observation."""
        rd = ResidualData(
            names=["obs1"],
            groups=["head"],
            observed=np.array([100.0]),
            simulated=np.array([100.0]),
            residuals=np.array([0.0]),
            weights=np.array([1.0]),
        )
        assert rd.n_observations == 1
        assert rd.phi == pytest.approx(0.0)

    def test_zero_weights(self):
        """Test with zero weights produces zero phi."""
        rd = ResidualData(
            names=["obs1", "obs2"],
            groups=["head", "head"],
            observed=np.array([100.0, 200.0]),
            simulated=np.array([0.0, 0.0]),
            residuals=np.array([100.0, 200.0]),
            weights=np.array([0.0, 0.0]),
        )
        assert rd.phi == pytest.approx(0.0)
        assert np.all(rd.weighted_residuals == 0.0)

    def test_group_phi_single_group(self):
        """Test group_phi with only one group."""
        rd = ResidualData(
            names=["o1", "o2"],
            groups=["head", "head"],
            observed=np.array([10.0, 20.0]),
            simulated=np.array([11.0, 19.0]),
            residuals=np.array([-1.0, 1.0]),
            weights=np.array([1.0, 1.0]),
        )
        gp = rd.group_phi()
        assert len(gp) == 1
        assert "head" in gp
        assert gp["head"] == pytest.approx(2.0)


class TestSensitivityDataEdgeCases:
    """Edge-case tests for SensitivityData."""

    def test_most_sensitive_more_than_available(self):
        """Test most_sensitive with n larger than parameter count."""
        sd = SensitivityData(
            parameter_names=["a", "b"],
            composite_sensitivities=np.array([5.0, 3.0]),
        )
        top = sd.most_sensitive(10)
        assert len(top) == 2
        assert top[0][0] == "a"

    def test_least_sensitive_more_than_available(self):
        """Test least_sensitive with n larger than parameter count."""
        sd = SensitivityData(
            parameter_names=["a", "b"],
            composite_sensitivities=np.array([5.0, 3.0]),
        )
        bottom = sd.least_sensitive(10)
        assert len(bottom) == 2
        assert bottom[0][0] == "b"

    def test_single_parameter(self):
        """Test with a single parameter."""
        sd = SensitivityData(
            parameter_names=["only_param"],
            composite_sensitivities=np.array([42.0]),
        )
        assert sd.n_parameters == 1
        top = sd.most_sensitive(1)
        assert top[0] == ("only_param", 42.0)
        bottom = sd.least_sensitive(1)
        assert bottom[0] == ("only_param", 42.0)

    def test_equal_sensitivities(self):
        """Test with all equal sensitivities."""
        sd = SensitivityData(
            parameter_names=["a", "b", "c"],
            composite_sensitivities=np.array([1.0, 1.0, 1.0]),
        )
        top = sd.most_sensitive(3)
        assert len(top) == 3
        assert all(s == pytest.approx(1.0) for _, s in top)


class TestCalibrationResultsEdgeCases:
    """Edge-case tests for CalibrationResults."""

    def test_fit_statistics_zero_ss_tot(self):
        """Test R^2 when all observations are equal (ss_tot=0)."""
        results = CalibrationResults(
            case_name="test",
            residuals=ResidualData(
                names=["o1", "o2", "o3"],
                groups=["head"] * 3,
                observed=np.array([100.0, 100.0, 100.0]),  # all equal
                simulated=np.array([101.0, 99.0, 100.0]),
                residuals=np.array([-1.0, 1.0, 0.0]),
                weights=np.ones(3),
            ),
        )
        stats = results.fit_statistics()
        assert stats["r_squared"] == 0.0  # max(r_squared, 0.0) when ss_tot=0

    def test_fit_statistics_zero_obs_sum(self):
        """Test pbias when sum of observations is 0."""
        results = CalibrationResults(
            case_name="test",
            residuals=ResidualData(
                names=["o1", "o2"],
                groups=["head", "head"],
                observed=np.array([1.0, -1.0]),  # sum = 0
                simulated=np.array([0.5, -0.5]),
                residuals=np.array([0.5, -0.5]),
                weights=np.ones(2),
            ),
        )
        stats = results.fit_statistics()
        assert stats["pbias"] == pytest.approx(0.0)

    def test_fit_statistics_empty_group(self):
        """Test fit statistics for a group with no matching observations."""
        results = CalibrationResults(
            case_name="test",
            residuals=ResidualData(
                names=["o1"],
                groups=["head"],
                observed=np.array([100.0]),
                simulated=np.array([98.0]),
                residuals=np.array([2.0]),
                weights=np.ones(1),
            ),
        )
        stats = results.fit_statistics(group="nonexistent_group")
        assert stats == {}

    def test_fit_statistics_perfect_fit(self):
        """Test fit statistics with perfect fit (zero residuals)."""
        results = CalibrationResults(
            case_name="test",
            residuals=ResidualData(
                names=["o1", "o2"],
                groups=["head", "head"],
                observed=np.array([100.0, 200.0]),
                simulated=np.array([100.0, 200.0]),
                residuals=np.array([0.0, 0.0]),
                weights=np.ones(2),
            ),
        )
        stats = results.fit_statistics()
        assert stats["rmse"] == pytest.approx(0.0)
        assert stats["mae"] == pytest.approx(0.0)
        assert stats["bias"] == pytest.approx(0.0)
        assert stats["r_squared"] == pytest.approx(1.0)

    def test_calibrated_values_default(self):
        """Test default calibrated_values is empty dict."""
        results = CalibrationResults(case_name="test")
        assert results.calibrated_values == {}

    def test_iteration_phi_default(self):
        """Test default iteration_phi is empty list."""
        results = CalibrationResults(case_name="test")
        assert results.iteration_phi == []


class TestPestPostProcessorFileLoading:
    """Additional file loading edge-case tests."""

    def test_load_res_file_fallback(self):
        """Test loading .res file when .rei doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            res_content = (
                "Name Group Measured Modelled Residual Weight\n"
                "obs1 head 100.0 98.0 2.0 1.0\n"
            )
            (Path(tmpdir) / "test.res").write_text(res_content)
            pp = PestPostProcessor(pest_dir=tmpdir, case_name="test")
            results = pp.load_results()
            assert results.residuals is not None
            assert results.residuals.n_observations == 1

    def test_rei_takes_priority_over_res(self):
        """Test that .rei is loaded instead of .res when both exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rei_content = (
                "Name Group Measured Modelled Residual Weight\n"
                "rei_obs head 100.0 98.0 2.0 1.0\n"
                "rei_obs2 head 200.0 195.0 5.0 1.0\n"
            )
            res_content = (
                "Name Group Measured Modelled Residual Weight\n"
                "res_obs flow 50.0 48.0 2.0 0.5\n"
            )
            (Path(tmpdir) / "test.rei").write_text(rei_content)
            (Path(tmpdir) / "test.res").write_text(res_content)
            pp = PestPostProcessor(pest_dir=tmpdir, case_name="test")
            results = pp.load_results()
            assert results.residuals is not None
            assert results.residuals.n_observations == 2
            assert results.residuals.names[0] == "rei_obs"

    def test_sensitivity_file_with_name_header(self):
        """Test loading sensitivity file where header starts with 'name'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sen_content = (
                "name sensitivity\n"
                "param1 10.0\n"
                "param2 5.0\n"
            )
            (Path(tmpdir) / "test.sen").write_text(sen_content)
            pp = PestPostProcessor(pest_dir=tmpdir, case_name="test")
            results = pp.load_results()
            assert results.sensitivities is not None
            assert results.sensitivities.n_parameters == 2

    def test_sensitivity_file_empty_data(self):
        """Test loading sensitivity file with header only."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sen_content = "Parameter_Name CSS\n"
            (Path(tmpdir) / "test.sen").write_text(sen_content)
            pp = PestPostProcessor(pest_dir=tmpdir, case_name="test")
            results = pp.load_results()
            assert results.sensitivities is None

    def test_iteration_history_with_invalid_lines(self):
        """Test loading iteration file with non-numeric lines."""
        with tempfile.TemporaryDirectory() as tmpdir:
            iobj_content = (
                "iteration total_phi\n"
                "0 1000.0\n"
                "1 bad_value\n"
                "2 500.0\n"
            )
            (Path(tmpdir) / "test.iobj").write_text(iobj_content)
            pp = PestPostProcessor(pest_dir=tmpdir, case_name="test")
            results = pp.load_results()
            # Should skip the bad line
            assert results.n_iterations == 2
            assert results.iteration_phi == [1000.0, 500.0]

    def test_calibrated_params_with_invalid_lines(self):
        """Test loading .par file with non-numeric value lines."""
        with tempfile.TemporaryDirectory() as tmpdir:
            par_content = (
                "single point\n"
                "hk_z1 1.5e-04 1.0 0.0\n"
                "bad_line not_a_number\n"
                "sy_z1 0.18 1.0 0.0\n"
            )
            (Path(tmpdir) / "test.par").write_text(par_content)
            pp = PestPostProcessor(pest_dir=tmpdir, case_name="test")
            results = pp.load_results()
            # "single point" header line has non-numeric second field => skipped
            # "bad_line" has non-numeric => skipped
            assert "hk_z1" in results.calibrated_values
            assert "sy_z1" in results.calibrated_values

    def test_residual_file_with_short_lines(self):
        """Test parsing residual file with lines that have fewer than 6 fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rei_content = (
                "Name Group Measured Modelled Residual Weight\n"
                "obs1 head 100.0 98.0 2.0 1.0\n"
                "short_line head\n"
                "obs2 head 200.0 195.0 5.0 1.0\n"
            )
            (Path(tmpdir) / "test.rei").write_text(rei_content)
            pp = PestPostProcessor(pest_dir=tmpdir, case_name="test")
            results = pp.load_results()
            assert results.residuals.n_observations == 2

    def test_load_all_files_together(self):
        """Test loading all result files at once."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rei_content = (
                "Name Group Measured Modelled Residual Weight\n"
                "obs1 head 100.0 98.0 2.0 1.0\n"
                "obs2 flow 50.0 48.0 2.0 0.5\n"
            )
            sen_content = (
                "Parameter_Name CSS\n"
                "hk_z1 10.0\n"
                "sy_z1 5.0\n"
            )
            iobj_content = (
                "iteration total_phi\n"
                "0 1000.0\n"
                "1 200.0\n"
            )
            par_content = "hk_z1 1.5e-04 1.0 0.0\nsy_z1 0.18 1.0 0.0\n"

            (Path(tmpdir) / "test.rei").write_text(rei_content)
            (Path(tmpdir) / "test.sen").write_text(sen_content)
            (Path(tmpdir) / "test.iobj").write_text(iobj_content)
            (Path(tmpdir) / "test.par").write_text(par_content)

            pp = PestPostProcessor(pest_dir=tmpdir, case_name="test")
            results = pp.load_results()

            assert results.residuals is not None
            assert results.sensitivities is not None
            assert results.n_iterations == 2
            assert len(results.calibrated_values) == 2
            assert results.final_phi > 0


class TestPestPostProcessorAnalysisEdgeCases:
    """Additional analysis edge-case tests."""

    def test_identifiability_zero_max_sensitivity(self):
        """Test identifiability when all sensitivities are zero."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sen_content = (
                "Parameter_Name CSS\n"
                "param1 0.0\n"
                "param2 0.0\n"
            )
            (Path(tmpdir) / "test.sen").write_text(sen_content)
            pp = PestPostProcessor(pest_dir=tmpdir, case_name="test")
            ident = pp.compute_identifiability()
            assert ident is not None
            # max_sens should be 1.0 when max(sens) is 0, so all = 0/1 = 0
            assert ident["param1"] == pytest.approx(0.0)
            assert ident["param2"] == pytest.approx(0.0)

    def test_export_csv_no_parameters(self):
        """Test exporting CSV when there are no calibrated parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pp = PestPostProcessor(pest_dir=tmpdir, case_name="test")
            output = Path(tmpdir) / "empty.csv"
            result_path = pp.export_calibrated_parameters(output, format="csv")
            assert result_path == output
            content = output.read_text()
            assert "parameter_name" in content
            # Should have header but no data lines
            lines = content.strip().split("\n")
            assert len(lines) == 1

    def test_export_pest_no_parameters(self):
        """Test exporting PEST format when there are no parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pp = PestPostProcessor(pest_dir=tmpdir, case_name="test")
            output = Path(tmpdir) / "empty.par"
            result_path = pp.export_calibrated_parameters(output, format="pest")
            assert result_path == output
            content = output.read_text()
            assert "single point" in content

    def test_summary_report_full(self):
        """Test summary report with all data sections present."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rei_content = (
                "Name Group Measured Modelled Residual Weight\n"
                "obs1 head 100.0 98.0 2.0 1.0\n"
                "obs2 head 200.0 195.0 5.0 1.0\n"
                "obs3 flow 50.0 48.0 2.0 0.5\n"
            )
            sen_content = (
                "Parameter_Name CSS\n"
                "hk_z1 10.0\n"
                "sy_z1 5.0\n"
            )
            iobj_content = (
                "iteration total_phi\n"
                "0 1000.0\n"
                "1 500.0\n"
                "2 250.0\n"
            )
            par_content = "hk_z1 1.5e-04 1.0 0.0\nsy_z1 0.18 1.0 0.0\n"

            (Path(tmpdir) / "test.rei").write_text(rei_content)
            (Path(tmpdir) / "test.sen").write_text(sen_content)
            (Path(tmpdir) / "test.iobj").write_text(iobj_content)
            (Path(tmpdir) / "test.par").write_text(par_content)

            pp = PestPostProcessor(pest_dir=tmpdir, case_name="test")
            report = pp.summary_report()

            assert "Calibration Report" in report
            assert "Iterations completed: 3" in report
            assert "Initial phi" in report
            assert "Final phi" in report
            assert "Phi reduction" in report
            assert "Fit Statistics" in report
            assert "Fit Statistics by Group" in report
            assert "Most Sensitive Parameters" in report
            assert "Calibrated Parameters" in report
            assert "hk_z1" in report

    def test_summary_report_with_iterations_only(self):
        """Test summary report with iteration history but no residuals."""
        with tempfile.TemporaryDirectory() as tmpdir:
            iobj_content = (
                "iteration total_phi\n"
                "0 500.0\n"
                "1 250.0\n"
            )
            (Path(tmpdir) / "test.iobj").write_text(iobj_content)
            pp = PestPostProcessor(pest_dir=tmpdir, case_name="test")
            report = pp.summary_report()
            assert "Iterations completed: 2" in report
            assert "Phi reduction" in report

    def test_summary_report_iteration_zero_initial_phi(self):
        """Test summary report when initial phi is 0 (edge case for reduction)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            iobj_content = (
                "iteration total_phi\n"
                "0 0.0\n"
                "1 0.0\n"
            )
            (Path(tmpdir) / "test.iobj").write_text(iobj_content)
            pp = PestPostProcessor(pest_dir=tmpdir, case_name="test")
            report = pp.summary_report()
            assert "Phi reduction: 0.0%" in report

    def test_summary_report_many_calibrated_params(self):
        """Test summary report truncation with >20 calibrated parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lines = []
            for i in range(25):
                lines.append(f"param_{i:03d} {float(i) * 0.01:.6e} 1.0 0.0")
            (Path(tmpdir) / "test.par").write_text("\n".join(lines))
            pp = PestPostProcessor(pest_dir=tmpdir, case_name="test")
            report = pp.summary_report()
            assert "Calibrated Parameters (25)" in report
            assert "... and 5 more" in report

    def test_export_returns_path_object(self):
        """Test that export methods return Path objects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            par_content = "hk_z1 1.5e-04 1.0 0.0\n"
            (Path(tmpdir) / "test.par").write_text(par_content)
            pp = PestPostProcessor(pest_dir=tmpdir, case_name="test")

            csv_path = pp.export_calibrated_parameters(
                Path(tmpdir) / "out.csv", format="csv"
            )
            assert isinstance(csv_path, Path)

            pest_path = pp.export_calibrated_parameters(
                Path(tmpdir) / "out.par", format="pest"
            )
            assert isinstance(pest_path, Path)

    def test_string_pest_dir(self):
        """Test PestPostProcessor accepts string path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pp = PestPostProcessor(pest_dir=str(tmpdir), case_name="test")
            assert isinstance(pp.pest_dir, Path)

    def test_export_calibrated_parameters_string_path(self):
        """Test export_calibrated_parameters accepts string filepath."""
        with tempfile.TemporaryDirectory() as tmpdir:
            par_content = "hk_z1 1.5e-04 1.0 0.0\n"
            (Path(tmpdir) / "test.par").write_text(par_content)
            pp = PestPostProcessor(pest_dir=tmpdir, case_name="test")
            output = str(Path(tmpdir) / "out.csv")
            result = pp.export_calibrated_parameters(output, format="csv")
            assert isinstance(result, Path)
            assert result.exists()
