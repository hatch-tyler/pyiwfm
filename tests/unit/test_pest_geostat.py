"""Unit tests for PEST++ geostatistics module."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from pyiwfm.runner.pest_geostat import (
    HAS_SCIPY,
    GeostatManager,
    Variogram,
    VariogramType,
    compute_empirical_variogram,
)
from pyiwfm.runner.pest_params import IWFMParameterType, Parameter


class TestVariogramType:
    """Tests for VariogramType enum."""

    def test_all_types(self):
        """Test all variogram types exist."""
        assert VariogramType.SPHERICAL.value == "spherical"
        assert VariogramType.EXPONENTIAL.value == "exponential"
        assert VariogramType.GAUSSIAN.value == "gaussian"
        assert VariogramType.MATERN.value == "matern"
        assert VariogramType.LINEAR.value == "linear"
        assert VariogramType.POWER.value == "power"
        assert VariogramType.NUGGET.value == "nugget"


class TestVariogramCreation:
    """Tests for Variogram creation and validation."""

    def test_basic_creation(self):
        """Test basic variogram creation."""
        v = Variogram("exponential", a=1000, sill=1.0, nugget=0.1)
        assert v.variogram_type == VariogramType.EXPONENTIAL
        assert v.a == 1000
        assert v.sill == 1.0
        assert v.nugget == 0.1

    def test_creation_with_enum(self):
        """Test creation with enum type."""
        v = Variogram(VariogramType.SPHERICAL, a=500, sill=2.0)
        assert v.variogram_type == VariogramType.SPHERICAL
        assert v.a == 500

    def test_invalid_range_raises(self):
        """Test that non-positive range raises error."""
        with pytest.raises(ValueError, match="Range must be positive"):
            Variogram("exponential", a=0, sill=1.0)
        with pytest.raises(ValueError, match="Range must be positive"):
            Variogram("exponential", a=-100, sill=1.0)

    def test_negative_sill_raises(self):
        """Test that negative sill raises error."""
        with pytest.raises(ValueError, match="Sill must be non-negative"):
            Variogram("exponential", a=1000, sill=-1.0)

    def test_negative_nugget_raises(self):
        """Test that negative nugget raises error."""
        with pytest.raises(ValueError, match="Nugget must be non-negative"):
            Variogram("exponential", a=1000, sill=1.0, nugget=-0.1)

    def test_invalid_anisotropy_raises(self):
        """Test that non-positive anisotropy raises error."""
        with pytest.raises(ValueError, match="Anisotropy ratio must be positive"):
            Variogram("exponential", a=1000, sill=1.0, anisotropy_ratio=0)

    def test_total_sill(self):
        """Test total_sill property."""
        v = Variogram("exponential", a=1000, sill=0.8, nugget=0.2)
        assert v.total_sill == 1.0

    def test_effective_range_exponential(self):
        """Test effective range for exponential."""
        v = Variogram("exponential", a=1000)
        assert v.effective_range == pytest.approx(3000)

    def test_effective_range_gaussian(self):
        """Test effective range for gaussian."""
        v = Variogram("gaussian", a=1000)
        assert v.effective_range == pytest.approx(np.sqrt(3) * 1000)

    def test_effective_range_spherical(self):
        """Test effective range for spherical."""
        v = Variogram("spherical", a=1000)
        assert v.effective_range == 1000


class TestVariogramModels:
    """Tests for variogram model evaluation."""

    def test_spherical_at_zero(self):
        """Test spherical model at h=0."""
        v = Variogram("spherical", a=1000, sill=1.0, nugget=0.1)
        assert v.evaluate(0) == 0.0

    def test_spherical_at_range(self):
        """Test spherical model at h=range."""
        v = Variogram("spherical", a=1000, sill=1.0, nugget=0.0)
        assert v.evaluate(1000) == pytest.approx(1.0)

    def test_spherical_beyond_range(self):
        """Test spherical model beyond range."""
        v = Variogram("spherical", a=1000, sill=1.0, nugget=0.1)
        assert v.evaluate(2000) == pytest.approx(1.1)

    def test_exponential_at_zero(self):
        """Test exponential model at h=0."""
        v = Variogram("exponential", a=1000, sill=1.0, nugget=0.1)
        assert v.evaluate(0) == 0.0

    def test_exponential_approaches_sill(self):
        """Test exponential approaches sill at large h."""
        v = Variogram("exponential", a=1000, sill=1.0, nugget=0.0)
        # At h=3*a, should be ~95% of sill
        gamma_3a = v.evaluate(3000)
        assert gamma_3a == pytest.approx(1 - np.exp(-3), rel=0.01)

    def test_gaussian_at_zero(self):
        """Test gaussian model at h=0."""
        v = Variogram("gaussian", a=1000, sill=1.0)
        assert v.evaluate(0) == 0.0

    def test_gaussian_at_range(self):
        """Test gaussian model at h=range."""
        v = Variogram("gaussian", a=1000, sill=1.0, nugget=0.0)
        gamma = v.evaluate(1000)
        expected = 1.0 * (1.0 - np.exp(-1))
        assert gamma == pytest.approx(expected)

    def test_matern_at_zero(self):
        """Test matern model at h=0."""
        v = Variogram("matern", a=1000, sill=1.0)
        assert v.evaluate(0) == 0.0

    def test_linear_proportional(self):
        """Test linear model is proportional to distance."""
        v = Variogram("linear", a=1000, sill=1.0, nugget=0.0)
        assert v.evaluate(500) == pytest.approx(0.5)
        assert v.evaluate(1000) == pytest.approx(1.0)

    def test_power_model(self):
        """Test power model."""
        v = Variogram("power", a=1000, sill=1.0, nugget=0.0, power=2)
        assert v.evaluate(500) == pytest.approx(0.25)  # (500/1000)^2

    def test_nugget_model(self):
        """Test pure nugget model."""
        v = Variogram("nugget", a=1000, sill=1.0, nugget=0.5)
        assert v.evaluate(0) == 0.0
        assert v.evaluate(100) == pytest.approx(1.5)
        assert v.evaluate(1000) == pytest.approx(1.5)

    def test_evaluate_array(self):
        """Test evaluating at multiple distances."""
        v = Variogram("exponential", a=1000, sill=1.0, nugget=0.0)
        h = np.array([0, 500, 1000, 2000])
        gamma = v.evaluate(h)
        assert gamma.shape == (4,)
        assert gamma[0] == 0.0
        assert gamma[1] > 0
        assert gamma[2] > gamma[1]

    def test_with_nugget(self):
        """Test that nugget shifts variogram up."""
        v_no_nugget = Variogram("exponential", a=1000, sill=1.0, nugget=0.0)
        v_nugget = Variogram("exponential", a=1000, sill=1.0, nugget=0.2)

        h = 500
        gamma_no = v_no_nugget.evaluate(h)
        gamma_with = v_nugget.evaluate(h)
        assert gamma_with == pytest.approx(gamma_no + 0.2)


class TestVariogramCovariance:
    """Tests for variogram covariance computation."""

    def test_covariance_at_zero(self):
        """Test covariance at h=0 equals total sill."""
        v = Variogram("exponential", a=1000, sill=0.8, nugget=0.2)
        assert v.covariance(0) == pytest.approx(1.0)

    def test_covariance_decreases(self):
        """Test covariance decreases with distance."""
        v = Variogram("exponential", a=1000, sill=1.0, nugget=0.0)
        c0 = v.covariance(0)
        c500 = v.covariance(500)
        c1000 = v.covariance(1000)
        assert c0 > c500 > c1000

    def test_covariance_array(self):
        """Test covariance with array input."""
        v = Variogram("spherical", a=1000, sill=1.0, nugget=0.0)
        h = np.array([0, 500, 1000, 2000])
        cov = v.covariance(h)
        assert cov.shape == (4,)
        assert cov[0] == pytest.approx(1.0)
        assert cov[3] == pytest.approx(0.0)  # Beyond range


class TestVariogramAnisotropy:
    """Tests for anisotropic variogram support."""

    def test_isotropic_transform(self):
        """Test that isotropic transform is identity."""
        v = Variogram("exponential", a=1000, anisotropy_ratio=1.0)
        x = np.array([0, 100, 200])
        y = np.array([0, 50, 100])
        x_t, y_t = v.transform_coordinates(x, y)
        np.testing.assert_array_equal(x_t, x)
        np.testing.assert_array_equal(y_t, y)

    def test_anisotropic_transform(self):
        """Test anisotropic coordinate transform."""
        v = Variogram("exponential", a=1000, anisotropy_ratio=2.0, anisotropy_angle=0)
        x = np.array([100, 0])
        y = np.array([0, 100])
        x_t, y_t = v.transform_coordinates(x, y)
        # Y should be scaled by ratio
        assert y_t[1] == pytest.approx(200)

    def test_distance_matrix_isotropic(self):
        """Test distance matrix for isotropic case."""
        v = Variogram("exponential", a=1000)
        x = np.array([0, 100])
        y = np.array([0, 0])
        dist = v.compute_distance_matrix(x, y)
        assert dist[0, 1] == pytest.approx(100)
        assert dist[1, 0] == pytest.approx(100)
        assert dist[0, 0] == 0.0

    def test_distance_matrix_two_sets(self):
        """Test distance matrix between two sets of points."""
        v = Variogram("exponential", a=1000)
        x1 = np.array([0, 100])
        y1 = np.array([0, 0])
        x2 = np.array([50])
        y2 = np.array([0])
        dist = v.compute_distance_matrix(x1, y1, x2, y2)
        assert dist.shape == (2, 1)
        assert dist[0, 0] == pytest.approx(50)
        assert dist[1, 0] == pytest.approx(50)


class TestVariogramSerialization:
    """Tests for variogram serialization."""

    def test_to_dict(self):
        """Test converting to dictionary."""
        v = Variogram(
            "exponential",
            a=1000,
            sill=0.8,
            nugget=0.2,
            anisotropy_ratio=1.5,
            anisotropy_angle=45,
        )
        d = v.to_dict()
        assert d["variogram_type"] == "exponential"
        assert d["a"] == 1000
        assert d["sill"] == 0.8
        assert d["nugget"] == 0.2
        assert d["anisotropy_ratio"] == 1.5
        assert d["anisotropy_angle"] == 45

    def test_from_dict(self):
        """Test creating from dictionary."""
        d = {
            "variogram_type": "spherical",
            "a": 500,
            "sill": 1.5,
            "nugget": 0.1,
        }
        v = Variogram.from_dict(d)
        assert v.variogram_type == VariogramType.SPHERICAL
        assert v.a == 500
        assert v.sill == 1.5
        assert v.nugget == 0.1

    def test_roundtrip(self):
        """Test dict roundtrip preserves values."""
        v1 = Variogram("matern", a=2000, sill=0.5, nugget=0.05)
        d = v1.to_dict()
        v2 = Variogram.from_dict(d)
        assert v2.variogram_type == v1.variogram_type
        assert v2.a == v1.a
        assert v2.sill == v1.sill
        assert v2.nugget == v1.nugget

    def test_repr(self):
        """Test string representation."""
        v = Variogram("exponential", a=1000, sill=1.0, nugget=0.1)
        r = repr(v)
        assert "exponential" in r
        assert "1000" in r


class TestGeostatManagerBasic:
    """Tests for GeostatManager basic functionality."""

    def test_init(self):
        """Test basic initialization."""
        gm = GeostatManager()
        assert gm.model is None

    def test_init_with_model(self):
        """Test initialization with model."""
        model = object()  # Mock model
        gm = GeostatManager(model=model)
        assert gm.model is model

    def test_repr(self):
        """Test string representation."""
        gm = GeostatManager()
        assert "GeostatManager" in repr(gm)
        assert "model=False" in repr(gm)


class TestCovarianceMatrix:
    """Tests for covariance matrix computation."""

    def test_covariance_matrix_shape(self):
        """Test covariance matrix has correct shape."""
        gm = GeostatManager()
        v = Variogram("exponential", a=1000, sill=1.0)
        x = np.array([0, 100, 200, 300])
        y = np.array([0, 0, 0, 0])

        cov = gm.compute_covariance_matrix(x, y, v)
        assert cov.shape == (4, 4)

    def test_covariance_matrix_symmetric(self):
        """Test covariance matrix is symmetric."""
        gm = GeostatManager()
        v = Variogram("spherical", a=500, sill=1.0)
        rng = np.random.default_rng(42)
        x = rng.random(10) * 1000
        y = rng.random(10) * 1000

        cov = gm.compute_covariance_matrix(x, y, v)
        np.testing.assert_array_almost_equal(cov, cov.T)

    def test_covariance_diagonal(self):
        """Test diagonal equals total sill."""
        gm = GeostatManager()
        v = Variogram("exponential", a=1000, sill=0.8, nugget=0.2)
        x = np.array([0, 500, 1000])
        y = np.array([0, 0, 0])

        cov = gm.compute_covariance_matrix(x, y, v)
        for i in range(3):
            assert cov[i, i] == pytest.approx(1.0)

    def test_variogram_matrix(self):
        """Test variogram matrix computation."""
        gm = GeostatManager()
        v = Variogram("exponential", a=1000, sill=1.0)
        x = np.array([0, 100])
        y = np.array([0, 0])

        gamma = gm.compute_variogram_matrix(x, y, v)
        assert gamma.shape == (2, 2)
        assert gamma[0, 0] == 0.0
        assert gamma[0, 1] == pytest.approx(v.evaluate(100))


@pytest.mark.skipif(not HAS_SCIPY, reason="scipy required")
class TestKriging:
    """Tests for kriging interpolation."""

    def test_kriging_ordinary_basic(self):
        """Test basic ordinary kriging."""
        gm = GeostatManager()
        v = Variogram("exponential", a=1000, sill=1.0)

        # Pilot points
        pp_x = np.array([0, 1000, 500])
        pp_y = np.array([0, 0, 500])
        pp_values = np.array([1.0, 2.0, 1.5])

        # Target point (at one pilot)
        target_x = np.array([0])
        target_y = np.array([0])

        result = gm.krige(pp_x, pp_y, pp_values, target_x, target_y, v)
        # Should be close to pilot value
        assert result[0] == pytest.approx(1.0, rel=0.1)

    def test_kriging_simple(self):
        """Test simple kriging."""
        gm = GeostatManager()
        v = Variogram("exponential", a=1000, sill=1.0)

        pp_x = np.array([0, 1000])
        pp_y = np.array([0, 0])
        pp_values = np.array([1.0, 2.0])

        target_x = np.array([500])
        target_y = np.array([0])

        result = gm.krige(pp_x, pp_y, pp_values, target_x, target_y, v, kriging_type="simple")
        assert len(result) == 1
        # Should be between the two values
        assert 1.0 <= result[0] <= 2.0

    def test_kriging_with_variance(self):
        """Test kriging returns variance."""
        gm = GeostatManager()
        v = Variogram("exponential", a=1000, sill=1.0)

        pp_x = np.array([0, 1000])
        pp_y = np.array([0, 0])
        pp_values = np.array([1.0, 2.0])

        target_x = np.array([500])
        target_y = np.array([0])

        result, variance = gm.krige(
            pp_x, pp_y, pp_values, target_x, target_y, v, return_variance=True
        )
        assert len(result) == 1
        assert len(variance) == 1
        assert variance[0] >= 0

    def test_kriging_multiple_targets(self):
        """Test kriging to multiple targets."""
        gm = GeostatManager()
        v = Variogram("spherical", a=500, sill=1.0)

        pp_x = np.array([0, 500, 250])
        pp_y = np.array([0, 0, 250])
        pp_values = np.array([1.0, 3.0, 2.0])

        target_x = np.array([100, 200, 300, 400])
        target_y = np.array([0, 0, 0, 0])

        result = gm.krige(pp_x, pp_y, pp_values, target_x, target_y, v)
        assert len(result) == 4


@pytest.mark.skipif(not HAS_SCIPY, reason="scipy required")
class TestKrigingFactors:
    """Tests for kriging factor computation."""

    def test_factors_shape(self):
        """Test kriging factors have correct shape."""
        gm = GeostatManager()
        v = Variogram("exponential", a=1000, sill=1.0)

        pp_x = np.array([0, 500, 1000])
        pp_y = np.array([0, 0, 0])
        target_x = np.array([100, 200, 300, 400])
        target_y = np.array([0, 0, 0, 0])

        factors = gm.compute_kriging_factors(pp_x, pp_y, target_x, target_y, v)
        assert factors.shape == (4, 3)  # 4 targets, 3 pilots

    def test_factors_sum_to_one(self):
        """Test ordinary kriging factors sum to 1."""
        gm = GeostatManager()
        v = Variogram("spherical", a=500, sill=1.0)

        pp_x = np.array([0, 500, 250])
        pp_y = np.array([0, 0, 250])
        target_x = np.array([100, 200])
        target_y = np.array([50, 100])

        factors = gm.compute_kriging_factors(
            pp_x, pp_y, target_x, target_y, v, kriging_type="ordinary"
        )
        # Each row should sum to ~1
        for i in range(len(target_x)):
            assert np.sum(factors[i, :]) == pytest.approx(1.0, rel=0.01)

    def test_factors_reproduce_kriging(self):
        """Test that factors reproduce kriging result."""
        gm = GeostatManager()
        v = Variogram("exponential", a=1000, sill=1.0)

        pp_x = np.array([0, 500, 1000])
        pp_y = np.array([0, 0, 0])
        pp_values = np.array([1.0, 2.0, 1.5])
        target_x = np.array([250])
        target_y = np.array([0])

        # Get kriging result directly
        kriged = gm.krige(pp_x, pp_y, pp_values, target_x, target_y, v)

        # Get via factors
        factors = gm.compute_kriging_factors(pp_x, pp_y, target_x, target_y, v)
        via_factors = factors @ pp_values

        np.testing.assert_array_almost_equal(kriged, via_factors)


@pytest.mark.skipif(not HAS_SCIPY, reason="scipy required")
class TestRealizations:
    """Tests for geostatistical realization generation."""

    def test_realizations_shape(self):
        """Test realizations have correct shape."""
        gm = GeostatManager()
        v = Variogram("exponential", a=500, sill=1.0)

        x = np.array([0, 100, 200, 300, 400])
        y = np.zeros(5)

        reals = gm.generate_realizations(x, y, v, n_realizations=10, seed=42)
        assert reals.shape == (10, 5)

    def test_realizations_mean(self):
        """Test realizations have correct mean."""
        gm = GeostatManager()
        v = Variogram("exponential", a=500, sill=0.5)

        x = np.linspace(0, 1000, 20)
        y = np.zeros(20)

        reals = gm.generate_realizations(x, y, v, n_realizations=100, mean=5.0, seed=42)
        # Mean of all realizations should be close to specified mean
        assert np.mean(reals) == pytest.approx(5.0, abs=0.5)

    def test_realizations_reproducible(self):
        """Test realizations are reproducible with seed."""
        gm = GeostatManager()
        v = Variogram("spherical", a=500, sill=1.0)

        x = np.array([0, 100, 200])
        y = np.zeros(3)

        reals1 = gm.generate_realizations(x, y, v, n_realizations=5, seed=123)
        reals2 = gm.generate_realizations(x, y, v, n_realizations=5, seed=123)

        np.testing.assert_array_equal(reals1, reals2)


@pytest.mark.skipif(not HAS_SCIPY, reason="scipy required")
class TestPriorEnsemble:
    """Tests for prior ensemble generation."""

    def test_prior_ensemble_shape(self):
        """Test prior ensemble has correct shape."""
        gm = GeostatManager()

        params = [
            Parameter(
                name="p1",
                param_type=IWFMParameterType.HORIZONTAL_K,
                initial_value=1e-4,
                lower_bound=1e-6,
                upper_bound=1e-2,
            ),
            Parameter(
                name="p2",
                param_type=IWFMParameterType.VERTICAL_K,
                initial_value=1e-5,
                lower_bound=1e-7,
                upper_bound=1e-3,
            ),
        ]

        ensemble = gm.generate_prior_ensemble(params, n_realizations=50, seed=42)
        assert ensemble.shape == (50, 2)

    def test_prior_ensemble_within_bounds(self):
        """Test prior ensemble values are within bounds."""
        gm = GeostatManager()

        params = [
            Parameter(
                name="p1",
                param_type=IWFMParameterType.HORIZONTAL_K,
                initial_value=0.5,
                lower_bound=0.1,
                upper_bound=0.9,
            ),
        ]

        ensemble = gm.generate_prior_ensemble(params, n_realizations=100, seed=42)
        assert np.all(ensemble >= 0.1)
        assert np.all(ensemble <= 0.9)

    def test_prior_ensemble_with_pilot_points(self):
        """Test prior ensemble with spatial parameters."""
        gm = GeostatManager()
        v = Variogram("exponential", a=500, sill=0.5)

        params = [
            Parameter(
                name="pp1",
                param_type=IWFMParameterType.HORIZONTAL_K,
                initial_value=1e-4,
                lower_bound=1e-6,
                upper_bound=1e-2,
                location=(0, 0),
            ),
            Parameter(
                name="pp2",
                param_type=IWFMParameterType.HORIZONTAL_K,
                initial_value=1e-4,
                lower_bound=1e-6,
                upper_bound=1e-2,
                location=(100, 0),
            ),
            Parameter(
                name="pp3",
                param_type=IWFMParameterType.HORIZONTAL_K,
                initial_value=1e-4,
                lower_bound=1e-6,
                upper_bound=1e-2,
                location=(50, 50),
            ),
        ]

        ensemble = gm.generate_prior_ensemble(params, n_realizations=20, variogram=v, seed=42)
        assert ensemble.shape == (20, 3)

    def test_prior_ensemble_lhs(self):
        """Test Latin Hypercube Sampling."""
        gm = GeostatManager()

        params = [
            Parameter(
                name="p1",
                param_type=IWFMParameterType.PUMPING_MULT,
                initial_value=1.0,
                lower_bound=0.5,
                upper_bound=1.5,
            ),
        ]

        ensemble = gm.generate_prior_ensemble(params, n_realizations=100, method="lhs", seed=42)
        assert ensemble.shape == (100, 1)
        # LHS should cover the range reasonably well
        assert np.min(ensemble) < 0.6
        assert np.max(ensemble) > 1.4


class TestLatinHypercube:
    """Tests for Latin Hypercube sampling."""

    def test_lhs_shape(self):
        """Test LHS has correct shape."""
        gm = GeostatManager()
        samples = gm._latin_hypercube(100, 5)
        assert samples.shape == (100, 5)

    def test_lhs_range(self):
        """Test LHS samples are in [0, 1]."""
        gm = GeostatManager()
        samples = gm._latin_hypercube(50, 3)
        assert np.all(samples >= 0)
        assert np.all(samples <= 1)

    def test_lhs_coverage(self):
        """Test LHS covers the range."""
        gm = GeostatManager()
        rng = np.random.default_rng(42)
        samples = gm._latin_hypercube(100, 1, rng=rng).flatten()
        # Check that samples cover different strata
        hist, _ = np.histogram(samples, bins=10)
        # Each bin should have approximately 10 samples
        assert np.all(hist >= 5)


@pytest.mark.skipif(not HAS_SCIPY, reason="scipy required")
class TestWriteKrigingFactors:
    """Tests for writing kriging factors to file."""

    def test_write_pest_format(self):
        """Test writing PEST format factors file."""
        gm = GeostatManager()
        v = Variogram("exponential", a=1000, sill=1.0)

        pp_x = np.array([0, 500])
        pp_y = np.array([0, 0])
        pp_names = ["pp1", "pp2"]
        target_x = np.array([250])
        target_y = np.array([0])
        target_ids = [1]

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "factors.dat"
            result = gm.write_kriging_factors(
                pp_x, pp_y, pp_names, target_x, target_y, target_ids, v, filepath, format="pest"
            )

            assert result.exists()
            content = result.read_text()
            assert "Kriging factors" in content
            assert "pp1" in content or "pp2" in content

    def test_write_csv_format(self):
        """Test writing CSV format factors file."""
        gm = GeostatManager()
        v = Variogram("exponential", a=1000, sill=1.0)

        pp_x = np.array([0, 500])
        pp_y = np.array([0, 0])
        pp_names = ["pp1", "pp2"]
        target_x = np.array([250, 750])
        target_y = np.array([0, 0])
        target_ids = [1, 2]

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "factors.csv"
            result = gm.write_kriging_factors(
                pp_x, pp_y, pp_names, target_x, target_y, target_ids, v, filepath, format="csv"
            )

            assert result.exists()
            content = result.read_text()
            lines = content.strip().split("\n")
            assert len(lines) == 3  # header + 2 rows
            assert "pp1" in lines[0]
            assert "pp2" in lines[0]


class TestWriteStructureFile:
    """Tests for writing PEST++ structure file."""

    def test_write_structure_file(self):
        """Test writing structure file."""
        gm = GeostatManager()
        v = Variogram("exponential", a=1000, sill=0.8, nugget=0.2)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "structure.dat"
            result = gm.write_structure_file(v, filepath, name="hk_struct")

            assert result.exists()
            content = result.read_text()
            assert "STRUCTURE hk_struct" in content
            assert "NUGGET 0.2" in content
            assert "A 1000" in content
            assert "SILL 0.8" in content
            assert "EXPONENTIAL" in content

    def test_write_structure_file_anisotropic(self):
        """Test writing structure file with anisotropy."""
        gm = GeostatManager()
        v = Variogram("spherical", a=500, sill=1.0, anisotropy_ratio=2.0, anisotropy_angle=45)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "structure.dat"
            result = gm.write_structure_file(v, filepath)

            content = result.read_text()
            assert "ANISOTROPY 2.0" in content
            assert "BEARING 45" in content


class TestEmpiricalVariogram:
    """Tests for empirical variogram computation."""

    def test_basic_computation(self):
        """Test basic empirical variogram computation."""
        # Create simple test data
        x = np.arange(10) * 100.0
        y = np.zeros(10)
        values = np.linspace(1, 3, 10)

        lags, gamma, counts = compute_empirical_variogram(x, y, values, n_lags=5)

        assert len(lags) == 5
        assert len(gamma) == 5
        assert len(counts) == 5
        assert np.all(counts >= 0)

    def test_with_max_lag(self):
        """Test empirical variogram with max lag."""
        rng = np.random.default_rng(42)
        x = rng.random(20) * 1000
        y = rng.random(20) * 1000
        values = rng.random(20)

        lags, gamma, counts = compute_empirical_variogram(x, y, values, n_lags=10, max_lag=500)

        assert lags[-1] < 500
        assert len(lags) == 10


@pytest.mark.skipif(not HAS_SCIPY, reason="scipy required")
class TestVariogramFitting:
    """Tests for variogram fitting from data."""

    def test_fit_from_data(self):
        """Test fitting variogram from data."""
        # Generate spatially correlated data
        rng = np.random.default_rng(42)
        n = 50
        x = rng.random(n) * 1000
        y = rng.random(n) * 1000
        # Simple spatial pattern
        values = x / 1000 + 0.1 * rng.standard_normal(n)

        v = Variogram.from_data(x, y, values, variogram_type="exponential")

        assert v.variogram_type == VariogramType.EXPONENTIAL
        assert v.a > 0
        assert v.sill > 0
        assert v.nugget >= 0

    def test_fit_different_types(self):
        """Test fitting different variogram types."""
        rng = np.random.default_rng(42)
        n = 30
        x = rng.random(n) * 500
        y = rng.random(n) * 500
        values = rng.random(n)

        for vtype in ["spherical", "exponential", "gaussian"]:
            v = Variogram.from_data(x, y, values, variogram_type=vtype)
            assert v.variogram_type.value == vtype


# ---------------------------------------------------------------------------
# Additional tests to increase coverage beyond 88%
# ---------------------------------------------------------------------------


class TestVariogramEvaluateScalar:
    """Tests for scalar input/output of variogram evaluate."""

    def test_evaluate_returns_float_for_scalar_input(self):
        """Test that evaluate returns a float when given a scalar."""
        v = Variogram("exponential", a=1000, sill=1.0, nugget=0.1)
        result = v.evaluate(500.0)
        assert isinstance(result, float)

    def test_evaluate_scalar_zero(self):
        """Evaluate at h=0 returns float 0.0."""
        v = Variogram("spherical", a=100, sill=1.0)
        result = v.evaluate(0.0)
        assert result == 0.0
        assert isinstance(result, float)

    def test_covariance_scalar(self):
        """Covariance at scalar h returns scalar."""
        v = Variogram("exponential", a=500, sill=1.0, nugget=0.0)
        c = v.covariance(250.0)
        assert isinstance(c, float)
        assert c > 0


class TestVariogramEffectiveRangeAllTypes:
    """Cover effective_range for all variogram types."""

    def test_effective_range_linear(self):
        """Linear variogram effective range equals a."""
        v = Variogram("linear", a=1000)
        assert v.effective_range == 1000

    def test_effective_range_power(self):
        """Power variogram effective range equals a."""
        v = Variogram("power", a=2000)
        assert v.effective_range == 2000

    def test_effective_range_matern(self):
        """Matern variogram effective range equals a."""
        v = Variogram("matern", a=1500)
        assert v.effective_range == 1500

    def test_effective_range_nugget(self):
        """Nugget variogram effective range equals a."""
        v = Variogram("nugget", a=800)
        assert v.effective_range == 800


class TestVariogramFromDictDefaults:
    """Test from_dict with missing optional keys to exercise defaults."""

    def test_from_dict_minimal(self):
        """From dict with only required keys uses defaults."""
        d = {"variogram_type": "exponential", "a": 500}
        v = Variogram.from_dict(d)
        assert v.sill == 1.0
        assert v.nugget == 0.0
        assert v.anisotropy_ratio == 1.0
        assert v.anisotropy_angle == 0.0
        assert v.power == 1.0


class TestAnisotropyRotation:
    """Cover anisotropy with non-zero angle."""

    def test_anisotropy_rotation_45_degrees(self):
        """Test rotation at 45 degrees with anisotropy."""
        v = Variogram("exponential", a=1000, anisotropy_ratio=2.0, anisotropy_angle=45)
        x = np.array([100.0])
        y = np.array([0.0])
        x_t, y_t = v.transform_coordinates(x, y)
        angle_rad = np.radians(45)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        expected_x = 100.0 * cos_a + 0.0 * sin_a
        expected_y = (-100.0 * sin_a + 0.0 * cos_a) * 2.0
        assert x_t[0] == pytest.approx(expected_x)
        assert y_t[0] == pytest.approx(expected_y)

    def test_anisotropic_distance_matrix(self):
        """Distance matrix differs when anisotropy is active."""
        v_iso = Variogram("exponential", a=1000, anisotropy_ratio=1.0)
        v_aniso = Variogram("exponential", a=1000, anisotropy_ratio=3.0, anisotropy_angle=30)
        x = np.array([0.0, 100.0])
        y = np.array([0.0, 100.0])
        d_iso = v_iso.compute_distance_matrix(x, y)
        d_aniso = v_aniso.compute_distance_matrix(x, y)
        # Anisotropic distances should generally differ from isotropic
        assert not np.allclose(d_iso, d_aniso)


class TestDistanceMatrixNoScipy:
    """Cover the manual distance computation fallback (no scipy)."""

    def test_manual_distance_computation(self):
        """Simulate fallback path for compute_distance_matrix."""
        import pyiwfm.runner.pest_geostat as gmod

        orig = gmod.HAS_SCIPY
        try:
            gmod.HAS_SCIPY = False
            v = Variogram("exponential", a=1000)
            x = np.array([0.0, 3.0])
            y = np.array([0.0, 4.0])
            dist = v.compute_distance_matrix(x, y)
            assert dist[0, 1] == pytest.approx(5.0)
            assert dist[1, 0] == pytest.approx(5.0)
        finally:
            gmod.HAS_SCIPY = orig


class TestEmpiricalVariogramNoScipy:
    """Cover compute_empirical_variogram fallback without scipy."""

    def test_empirical_variogram_no_scipy(self):
        """Test empirical variogram fallback distance computation."""
        import pyiwfm.runner.pest_geostat as gmod

        orig = gmod.HAS_SCIPY
        try:
            gmod.HAS_SCIPY = False
            x = np.array([0.0, 100.0, 200.0])
            y = np.array([0.0, 0.0, 0.0])
            values = np.array([1.0, 2.0, 3.0])
            lags, gamma, counts = compute_empirical_variogram(x, y, values, n_lags=3)
            assert len(lags) == 3
        finally:
            gmod.HAS_SCIPY = orig


@pytest.mark.skipif(not HAS_SCIPY, reason="scipy required")
class TestVariogramFittingEdgeCases:
    """Cover from_data edge cases including curve_fit failure."""

    def test_fit_with_explicit_max_lag(self):
        """Test fitting with an explicit max_lag parameter."""
        rng = np.random.default_rng(99)
        n = 30
        x = rng.random(n) * 500
        y = rng.random(n) * 500
        values = x / 500 + 0.05 * rng.standard_normal(n)
        v = Variogram.from_data(x, y, values, variogram_type="exponential", max_lag=200.0)
        assert v.a > 0
        assert v.sill > 0

    def test_fit_fallback_on_curve_fit_failure(self):
        """Test from_data fallback when curve_fit raises RuntimeError."""
        from unittest.mock import patch

        rng = np.random.default_rng(42)
        n = 20
        x = rng.random(n) * 100
        y = rng.random(n) * 100
        values = rng.random(n)

        with patch("pyiwfm.runner.pest_geostat.curve_fit", side_effect=RuntimeError("fail")):
            v = Variogram.from_data(x, y, values, variogram_type="exponential")
            # Should fall back to simple estimates
            assert v.nugget == 0.0
            assert v.a > 0
            assert v.sill > 0

    def test_from_data_no_scipy_raises(self):
        """Test from_data raises ImportError when scipy is missing."""
        import pyiwfm.runner.pest_geostat as gmod

        orig = gmod.HAS_SCIPY
        try:
            gmod.HAS_SCIPY = False
            with pytest.raises(ImportError, match="scipy required"):
                Variogram.from_data(np.array([0.0]), np.array([0.0]), np.array([1.0]))
        finally:
            gmod.HAS_SCIPY = orig


@pytest.mark.skipif(not HAS_SCIPY, reason="scipy required")
class TestKrigingEdgeCases:
    """Cover kriging edge cases: singular matrices, simple variance, etc."""

    def test_simple_kriging_with_variance(self):
        """Test simple kriging returns variance when requested."""
        gm = GeostatManager()
        v = Variogram("exponential", a=1000, sill=1.0)
        pp_x = np.array([0.0, 1000.0])
        pp_y = np.array([0.0, 0.0])
        pp_values = np.array([1.0, 2.0])
        target_x = np.array([500.0])
        target_y = np.array([0.0])

        result, variance = gm.krige(
            pp_x,
            pp_y,
            pp_values,
            target_x,
            target_y,
            v,
            kriging_type="simple",
            return_variance=True,
        )
        assert len(result) == 1
        assert len(variance) == 1
        assert variance[0] >= 0

    def test_ordinary_kriging_singular_matrix(self):
        """Test ordinary kriging handles singular covariance matrix."""
        from unittest.mock import patch

        from scipy import linalg

        gm = GeostatManager()
        v = Variogram("exponential", a=1000, sill=1.0)
        pp_x = np.array([0.0, 1000.0, 500.0])
        pp_y = np.array([0.0, 0.0, 500.0])
        pp_values = np.array([1.0, 2.0, 1.5])
        target_x = np.array([250.0])
        target_y = np.array([250.0])

        call_count = [0]
        original_inv = linalg.inv

        def flaky_inv(m):
            call_count[0] += 1
            if call_count[0] == 1:
                raise linalg.LinAlgError("singular")
            return original_inv(m)

        with patch("pyiwfm.runner.pest_geostat.linalg.inv", side_effect=flaky_inv):
            result = gm.krige(pp_x, pp_y, pp_values, target_x, target_y, v)
            assert len(result) == 1

    def test_simple_kriging_singular_matrix(self):
        """Test simple kriging handles singular covariance matrix."""
        from unittest.mock import patch

        from scipy import linalg

        gm = GeostatManager()
        v = Variogram("exponential", a=1000, sill=1.0)
        pp_x = np.array([0.0, 1000.0])
        pp_y = np.array([0.0, 0.0])
        pp_values = np.array([1.0, 2.0])
        target_x = np.array([500.0])
        target_y = np.array([0.0])

        call_count = [0]
        original_inv = linalg.inv

        def flaky_inv(m):
            call_count[0] += 1
            if call_count[0] == 1:
                raise linalg.LinAlgError("singular")
            return original_inv(m)

        with patch("pyiwfm.runner.pest_geostat.linalg.inv", side_effect=flaky_inv):
            result = gm.krige(
                pp_x,
                pp_y,
                pp_values,
                target_x,
                target_y,
                v,
                kriging_type="simple",
            )
            assert len(result) == 1

    def test_krige_no_scipy_raises(self):
        """Test krige raises ImportError when scipy is missing."""
        import pyiwfm.runner.pest_geostat as gmod

        orig = gmod.HAS_SCIPY
        try:
            gmod.HAS_SCIPY = False
            gm = GeostatManager()
            v = Variogram("exponential", a=1000, sill=1.0)
            with pytest.raises(ImportError, match="scipy required"):
                gm.krige(
                    np.array([0.0]),
                    np.array([0.0]),
                    np.array([1.0]),
                    np.array([1.0]),
                    np.array([0.0]),
                    v,
                )
        finally:
            gmod.HAS_SCIPY = orig


@pytest.mark.skipif(not HAS_SCIPY, reason="scipy required")
class TestKrigingFactorsEdgeCases:
    """Cover kriging factors edge cases."""

    def test_simple_kriging_factors(self):
        """Test computing kriging factors with simple kriging."""
        gm = GeostatManager()
        v = Variogram("exponential", a=1000, sill=1.0)
        pp_x = np.array([0.0, 500.0, 1000.0])
        pp_y = np.array([0.0, 0.0, 0.0])
        target_x = np.array([250.0, 750.0])
        target_y = np.array([0.0, 0.0])

        factors = gm.compute_kriging_factors(
            pp_x, pp_y, target_x, target_y, v, kriging_type="simple"
        )
        assert factors.shape == (2, 3)

    def test_factors_ordinary_singular_matrix(self):
        """Test ordinary kriging factors with singular matrix."""
        from unittest.mock import patch

        from scipy import linalg

        gm = GeostatManager()
        v = Variogram("exponential", a=1000, sill=1.0)
        pp_x = np.array([0.0, 500.0])
        pp_y = np.array([0.0, 0.0])
        target_x = np.array([250.0])
        target_y = np.array([0.0])

        call_count = [0]
        original_inv = linalg.inv

        def flaky_inv(m):
            call_count[0] += 1
            if call_count[0] == 1:
                raise linalg.LinAlgError("singular")
            return original_inv(m)

        with patch("pyiwfm.runner.pest_geostat.linalg.inv", side_effect=flaky_inv):
            factors = gm.compute_kriging_factors(
                pp_x, pp_y, target_x, target_y, v, kriging_type="ordinary"
            )
            assert factors.shape == (1, 2)

    def test_factors_simple_singular_matrix(self):
        """Test simple kriging factors with singular matrix."""
        from unittest.mock import patch

        from scipy import linalg

        gm = GeostatManager()
        v = Variogram("exponential", a=1000, sill=1.0)
        pp_x = np.array([0.0, 500.0])
        pp_y = np.array([0.0, 0.0])
        target_x = np.array([250.0])
        target_y = np.array([0.0])

        call_count = [0]
        original_inv = linalg.inv

        def flaky_inv(m):
            call_count[0] += 1
            if call_count[0] == 1:
                raise linalg.LinAlgError("singular")
            return original_inv(m)

        with patch("pyiwfm.runner.pest_geostat.linalg.inv", side_effect=flaky_inv):
            factors = gm.compute_kriging_factors(
                pp_x, pp_y, target_x, target_y, v, kriging_type="simple"
            )
            assert factors.shape == (1, 2)

    def test_kriging_factors_no_scipy_raises(self):
        """Test compute_kriging_factors raises ImportError without scipy."""
        import pyiwfm.runner.pest_geostat as gmod

        orig = gmod.HAS_SCIPY
        try:
            gmod.HAS_SCIPY = False
            gm = GeostatManager()
            v = Variogram("exponential", a=1000, sill=1.0)
            with pytest.raises(ImportError, match="scipy required"):
                gm.compute_kriging_factors(
                    np.array([0.0]),
                    np.array([0.0]),
                    np.array([1.0]),
                    np.array([0.0]),
                    v,
                )
        finally:
            gmod.HAS_SCIPY = orig


@pytest.mark.skipif(not HAS_SCIPY, reason="scipy required")
class TestRealizationsEdgeCases:
    """Cover realization edge cases: Cholesky fail, conditioning."""

    def test_realizations_cholesky_fallback(self):
        """Test eigendecomposition fallback when Cholesky fails."""
        from unittest.mock import patch

        from scipy import linalg

        gm = GeostatManager()
        v = Variogram("exponential", a=500, sill=1.0)
        x = np.array([0.0, 100.0, 200.0])
        y = np.zeros(3)

        with patch(
            "pyiwfm.runner.pest_geostat.linalg.cholesky",
            side_effect=linalg.LinAlgError("not positive definite"),
        ):
            reals = gm.generate_realizations(x, y, v, n_realizations=5, seed=42)
            assert reals.shape == (5, 3)

    def test_realizations_with_conditioning_data(self):
        """Test conditional realizations with conditioning data."""
        gm = GeostatManager()
        v = Variogram("exponential", a=500, sill=1.0)
        x = np.array([0.0, 100.0, 200.0, 300.0, 400.0])
        y = np.zeros(5)

        cond_x = np.array([50.0, 250.0])
        cond_y = np.array([0.0, 0.0])
        cond_values = np.array([5.0, 10.0])

        reals = gm.generate_realizations(
            x,
            y,
            v,
            n_realizations=3,
            mean=7.0,
            conditioning_data=(cond_x, cond_y, cond_values),
            seed=42,
        )
        assert reals.shape == (3, 5)

    def test_realizations_no_scipy_raises(self):
        """Test generate_realizations raises ImportError without scipy."""
        import pyiwfm.runner.pest_geostat as gmod

        orig = gmod.HAS_SCIPY
        try:
            gmod.HAS_SCIPY = False
            gm = GeostatManager()
            v = Variogram("exponential", a=500, sill=1.0)
            with pytest.raises(ImportError, match="scipy required"):
                gm.generate_realizations(
                    np.array([0.0]),
                    np.array([0.0]),
                    v,
                    n_realizations=1,
                )
        finally:
            gmod.HAS_SCIPY = orig


@pytest.mark.skipif(not HAS_SCIPY, reason="scipy required")
class TestPriorEnsembleEdgeCases:
    """Cover prior ensemble edge cases: log transforms, uniform method."""

    def test_prior_ensemble_log_transformed_spatial(self):
        """Test prior ensemble with log-transformed spatial parameters."""
        gm = GeostatManager()
        v = Variogram("exponential", a=500, sill=0.5)

        params = [
            Parameter(
                name="pp1",
                param_type=IWFMParameterType.HORIZONTAL_K,
                initial_value=1e-3,
                lower_bound=1e-6,
                upper_bound=1e-1,
                location=(0, 0),
                transform="log",
            ),
            Parameter(
                name="pp2",
                param_type=IWFMParameterType.HORIZONTAL_K,
                initial_value=1e-3,
                lower_bound=1e-6,
                upper_bound=1e-1,
                location=(100, 0),
                transform="log",
            ),
        ]
        ensemble = gm.generate_prior_ensemble(
            params,
            n_realizations=10,
            variogram=v,
            seed=42,
        )
        assert ensemble.shape == (10, 2)
        # Values should be clipped to bounds
        assert np.all(ensemble >= 1e-6)
        assert np.all(ensemble <= 1e-1)

    def test_prior_ensemble_nonspatial_log_transform(self):
        """Test non-spatial parameters with log transform."""
        gm = GeostatManager()
        params = [
            Parameter(
                name="p1",
                param_type=IWFMParameterType.HORIZONTAL_K,
                initial_value=1.0,
                lower_bound=0.01,
                upper_bound=100.0,
                transform="log",
            ),
        ]
        ensemble = gm.generate_prior_ensemble(
            params,
            n_realizations=50,
            seed=42,
            method="lhs",
        )
        assert ensemble.shape == (50, 1)
        assert np.all(ensemble >= 0.01)
        assert np.all(ensemble <= 100.0)

    def test_prior_ensemble_uniform_method(self):
        """Test prior ensemble with uniform sampling method."""
        gm = GeostatManager()
        params = [
            Parameter(
                name="p1",
                param_type=IWFMParameterType.PUMPING_MULT,
                initial_value=1.0,
                lower_bound=0.5,
                upper_bound=1.5,
            ),
            Parameter(
                name="p2",
                param_type=IWFMParameterType.RECHARGE_MULT,
                initial_value=1.0,
                lower_bound=0.5,
                upper_bound=1.5,
            ),
        ]
        ensemble = gm.generate_prior_ensemble(
            params,
            n_realizations=50,
            seed=42,
            method="uniform",
        )
        assert ensemble.shape == (50, 2)
        assert np.all(ensemble >= 0.5)
        assert np.all(ensemble <= 1.5)

    def test_prior_ensemble_spatial_no_variogram(self):
        """Test that spatial params without variogram are not correlated."""
        gm = GeostatManager()
        params = [
            Parameter(
                name="pp1",
                param_type=IWFMParameterType.HORIZONTAL_K,
                initial_value=1.0,
                lower_bound=0.1,
                upper_bound=10.0,
                location=(0, 0),
            ),
        ]
        # No variogram => spatial params are not generated with correlation
        ensemble = gm.generate_prior_ensemble(
            params,
            n_realizations=10,
            variogram=None,
            seed=42,
        )
        assert ensemble.shape == (10, 1)

    def test_prior_ensemble_mixed_spatial_nonspatial(self):
        """Test ensemble with both spatial and non-spatial params."""
        gm = GeostatManager()
        v = Variogram("exponential", a=500, sill=0.5)
        params = [
            Parameter(
                name="pp1",
                param_type=IWFMParameterType.HORIZONTAL_K,
                initial_value=1.0,
                lower_bound=0.1,
                upper_bound=10.0,
                location=(0, 0),
            ),
            Parameter(
                name="mult1",
                param_type=IWFMParameterType.PUMPING_MULT,
                initial_value=1.0,
                lower_bound=0.5,
                upper_bound=1.5,
            ),
        ]
        ensemble = gm.generate_prior_ensemble(
            params,
            n_realizations=20,
            variogram=v,
            seed=42,
        )
        assert ensemble.shape == (20, 2)


class TestGeostatManagerReprWithModel:
    """Cover repr with a model set."""

    def test_repr_with_model(self):
        """GeostatManager repr indicates model presence."""
        gm = GeostatManager(model="some_model")
        assert "model=True" in repr(gm)
