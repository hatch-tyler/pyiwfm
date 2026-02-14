"""Unit tests for PEST++ ensemble management."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from pyiwfm.runner.pest_ensemble import (
    IWFMEnsembleManager,
    EnsembleStatistics,
)
from pyiwfm.runner.pest_params import Parameter, IWFMParameterType
from pyiwfm.runner.pest_geostat import Variogram


class TestEnsembleStatistics:
    """Tests for EnsembleStatistics."""

    def test_basic(self):
        """Test basic ensemble statistics."""
        stats = EnsembleStatistics(
            mean=np.array([1.0, 2.0]),
            std=np.array([0.1, 0.2]),
            median=np.array([1.0, 2.0]),
            q05=np.array([0.8, 1.6]),
            q95=np.array([1.2, 2.4]),
            n_realizations=100,
            n_parameters=2,
            parameter_names=["p1", "p2"],
        )
        assert stats.n_realizations == 100
        assert stats.n_parameters == 2

    def test_to_dict(self):
        """Test converting to dictionary."""
        stats = EnsembleStatistics(
            mean=np.array([1.0]),
            std=np.array([0.1]),
            median=np.array([1.0]),
            q05=np.array([0.8]),
            q95=np.array([1.2]),
            n_realizations=50,
            n_parameters=1,
            parameter_names=["test_param"],
        )
        d = stats.to_dict()
        assert d["n_realizations"] == 50
        assert "test_param" in d["parameters"]
        assert d["parameters"]["test_param"]["mean"] == pytest.approx(1.0)


class TestIWFMEnsembleManagerInit:
    """Tests for IWFMEnsembleManager initialization."""

    def test_basic_init(self):
        """Test basic initialization."""
        em = IWFMEnsembleManager()
        assert em.n_parameters == 0
        assert em._prior_ensemble is None
        assert em._posterior_ensemble is None

    def test_init_with_parameters(self):
        """Test initialization with parameters."""
        params = [
            Parameter(
                name="p1",
                param_type=IWFMParameterType.HORIZONTAL_K,
                initial_value=1.0,
                lower_bound=0.1,
                upper_bound=10.0,
            ),
        ]
        em = IWFMEnsembleManager(parameters=params)
        assert em.n_parameters == 1
        assert em.parameter_names == ["p1"]

    def test_repr(self):
        """Test string representation."""
        em = IWFMEnsembleManager()
        r = repr(em)
        assert "IWFMEnsembleManager" in r
        assert "n_parameters=0" in r


class TestPriorEnsemble:
    """Tests for prior ensemble generation."""

    @pytest.fixture
    def simple_params(self):
        """Create simple parameters."""
        return [
            Parameter(
                name="p1",
                param_type=IWFMParameterType.HORIZONTAL_K,
                initial_value=1.0,
                lower_bound=0.1,
                upper_bound=10.0,
            ),
            Parameter(
                name="p2",
                param_type=IWFMParameterType.SPECIFIC_YIELD,
                initial_value=0.15,
                lower_bound=0.05,
                upper_bound=0.3,
            ),
        ]

    def test_generate_prior(self, simple_params):
        """Test generating prior ensemble."""
        em = IWFMEnsembleManager(parameters=simple_params)
        prior = em.generate_prior_ensemble(n_realizations=50, seed=42)
        assert prior.shape == (50, 2)

    def test_prior_stored(self, simple_params):
        """Test that prior is stored."""
        em = IWFMEnsembleManager(parameters=simple_params)
        prior = em.generate_prior_ensemble(n_realizations=20, seed=42)
        assert em._prior_ensemble is not None
        np.testing.assert_array_equal(em._prior_ensemble, prior)

    def test_no_parameters_raises(self):
        """Test that no parameters raises error."""
        em = IWFMEnsembleManager()
        with pytest.raises(ValueError, match="No parameters"):
            em.generate_prior_ensemble()


class TestObservationEnsemble:
    """Tests for observation noise ensemble."""

    def test_generate_observation_ensemble(self):
        """Test generating observation noise ensemble."""
        em = IWFMEnsembleManager()
        obs_values = np.array([100.0, 200.0, 50.0])
        obs_weights = np.array([1.0, 0.5, 2.0])

        ensemble = em.generate_observation_ensemble(
            obs_values, obs_weights,
            n_realizations=50, seed=42,
        )
        assert ensemble.shape == (50, 3)
        # Mean should be close to observed values
        means = np.mean(ensemble, axis=0)
        np.testing.assert_allclose(means, obs_values, atol=1.0)


class TestEnsembleIO:
    """Tests for ensemble file I/O."""

    @pytest.fixture
    def ensemble_manager(self):
        """Create manager with parameters."""
        params = [
            Parameter(
                name="hk_z1",
                param_type=IWFMParameterType.HORIZONTAL_K,
                initial_value=1.0,
                lower_bound=0.1,
                upper_bound=10.0,
            ),
            Parameter(
                name="sy_z1",
                param_type=IWFMParameterType.SPECIFIC_YIELD,
                initial_value=0.15,
                lower_bound=0.05,
                upper_bound=0.3,
            ),
        ]
        return IWFMEnsembleManager(parameters=params)

    def test_write_parameter_ensemble(self, ensemble_manager):
        """Test writing parameter ensemble to CSV."""
        ensemble = np.array([
            [1.0, 0.15],
            [2.0, 0.20],
            [0.5, 0.10],
        ])

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "prior.csv"
            result = ensemble_manager.write_parameter_ensemble(ensemble, filepath)
            assert result.exists()

            content = result.read_text()
            lines = content.strip().split("\n")
            assert len(lines) == 4  # header + 3 rows
            assert "hk_z1" in lines[0]
            assert "sy_z1" in lines[0]
            assert "r0000" in lines[1]

    def test_write_observation_ensemble(self, ensemble_manager):
        """Test writing observation ensemble to CSV."""
        ensemble = np.array([
            [100.0, 200.0],
            [101.0, 199.0],
        ])

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "obs_noise.csv"
            result = ensemble_manager.write_observation_ensemble(
                ensemble, ["obs1", "obs2"], filepath
            )
            assert result.exists()

            content = result.read_text()
            lines = content.strip().split("\n")
            assert len(lines) == 3  # header + 2 rows
            assert "obs1" in lines[0]

    def test_load_posterior_ensemble(self, ensemble_manager):
        """Test loading posterior ensemble from CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "posterior.csv"
            content = "real_name,hk_z1,sy_z1\nr0000,1.5,0.18\nr0001,2.0,0.12\n"
            filepath.write_text(content)

            posterior = ensemble_manager.load_posterior_ensemble(filepath)
            assert posterior.shape == (2, 2)
            assert posterior[0, 0] == pytest.approx(1.5)
            assert posterior[0, 1] == pytest.approx(0.18)

    def test_load_nonexistent_raises(self, ensemble_manager):
        """Test loading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            ensemble_manager.load_posterior_ensemble("/nonexistent.csv")


class TestEnsembleAnalysis:
    """Tests for ensemble analysis methods."""

    @pytest.fixture
    def manager_with_params(self):
        """Create manager with parameters."""
        params = [
            Parameter(
                name="p1",
                param_type=IWFMParameterType.HORIZONTAL_K,
                initial_value=1.0,
                lower_bound=0.1,
                upper_bound=10.0,
            ),
            Parameter(
                name="p2",
                param_type=IWFMParameterType.SPECIFIC_YIELD,
                initial_value=0.15,
                lower_bound=0.05,
                upper_bound=0.3,
            ),
        ]
        return IWFMEnsembleManager(parameters=params)

    def test_analyze_ensemble(self, manager_with_params):
        """Test computing ensemble statistics."""
        np.random.seed(42)
        ensemble = np.random.rand(100, 2)

        stats = manager_with_params.analyze_ensemble(ensemble)
        assert stats.n_realizations == 100
        assert stats.n_parameters == 2
        assert len(stats.mean) == 2
        assert len(stats.std) == 2
        assert np.all(stats.q05 < stats.median)
        assert np.all(stats.median < stats.q95)

    def test_compute_reduction_factor(self, manager_with_params):
        """Test uncertainty reduction computation."""
        np.random.seed(42)
        # Wide prior, narrow posterior
        prior = np.random.randn(100, 2) * 10
        posterior = np.random.randn(100, 2) * 1

        reduction = manager_with_params.compute_reduction_factor(prior, posterior)
        assert reduction.shape == (2,)
        # Posterior should have less variance -> positive reduction
        assert np.all(reduction > 0.5)

    def test_get_best_realization(self, manager_with_params):
        """Test getting best realization."""
        ensemble = np.array([
            [1.0, 0.15],
            [2.0, 0.20],
            [1.5, 0.18],
        ])
        objectives = np.array([10.0, 5.0, 8.0])

        idx, values = manager_with_params.get_best_realization(ensemble, objectives)
        assert idx == 1
        np.testing.assert_array_equal(values, [2.0, 0.20])
