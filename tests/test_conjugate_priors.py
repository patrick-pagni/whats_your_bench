import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "whats_your_bench"))

import numpy as np
import pytest
import conjugate_priors as cp


RNG = np.random.default_rng(42)


class TestNormalKnownVar:
    def setup_method(self):
        self.prior = cp.NormalKnownVar(variance=1, prior_params={"mu": 0.0, "sigma": 1.0})
        self.data = RNG.normal(loc=2.0, scale=1.0, size=100)
        self.prior.find_predictive_posterior(self.data)

    def test_posterior_mu_shifts_toward_data(self):
        # With N=100 observations near 2.0, posterior mu should be close to 2.0
        assert abs(self.prior.posterior_params.mu - 2.0) < 0.3

    def test_posterior_sigma_shrinks(self):
        # Posterior variance must be smaller than prior variance
        assert self.prior.posterior_params.sigma < 1.0

    def test_predictive_params_exist(self):
        p = self.prior.posterior_predictive_params
        assert hasattr(p, "loc")
        assert hasattr(p, "scale")
        assert p.scale > 0

    def test_predictive_dist_is_norm(self):
        from scipy import stats
        assert self.prior.predictive_dist is stats.norm


class TestNormalKnownMean:
    def setup_method(self):
        self.prior = cp.NormalKnownMean(mean=0.0, prior_params={"alpha": 1.0, "beta": 1.0})
        self.data = RNG.normal(loc=0.0, scale=2.0, size=50)
        self.prior.find_predictive_posterior(self.data)

    def test_posterior_alpha_increases(self):
        assert self.prior.posterior_params.alpha > 1.0

    def test_posterior_beta_increases(self):
        assert self.prior.posterior_params.beta > 1.0

    def test_predictive_params_exist(self):
        p = self.prior.posterior_predictive_params
        assert hasattr(p, "df")
        assert hasattr(p, "loc")
        assert hasattr(p, "scale")
        assert p.df > 0
        assert p.scale > 0

    def test_predictive_loc_equals_known_mean(self):
        assert self.prior.posterior_predictive_params.loc == 0.0


class TestMvNormalKnownCov:
    def setup_method(self):
        self.d = 2
        cov = np.eye(self.d).tolist()
        self.prior = cp.MvNormalKnownCov(
            covariance=cov,
            prior_params={"mu": [0.0, 0.0], "sigma": cov},
        )
        self.data = RNG.multivariate_normal([3.0, 5.0], np.eye(self.d), size=50)
        self.prior.find_predictive_posterior(self.data)

    def test_posterior_mu_shape(self):
        assert len(self.prior.posterior_params.mu) == self.d

    def test_posterior_mu_shifts_toward_data_mean(self):
        mu = self.prior.posterior_params.mu
        assert abs(mu[0] - 3.0) < 0.5
        assert abs(mu[1] - 5.0) < 0.5

    def test_predictive_params_exist(self):
        p = self.prior.posterior_predictive_params
        assert hasattr(p, "mean")
        assert hasattr(p, "cov")
        assert np.array(p.cov).shape == (self.d, self.d)


class TestMvNormalKnownMean:
    def setup_method(self):
        self.d = 2
        self.prior = cp.MvNormalKnownMean(
            mean=[0.0, 0.0],
            prior_params={"nu": 3, "psi": np.eye(self.d).tolist()},
        )
        self.data = RNG.multivariate_normal([0.0, 0.0], np.eye(self.d), size=20)
        self.prior.find_predictive_posterior(self.data)

    def test_posterior_nu_increases(self):
        assert self.prior.posterior_params.nu > 3

    def test_predictive_params_exist(self):
        p = self.prior.posterior_predictive_params
        assert hasattr(p, "df")
        assert hasattr(p, "loc")
        assert hasattr(p, "shape")
        assert p.df > 0
