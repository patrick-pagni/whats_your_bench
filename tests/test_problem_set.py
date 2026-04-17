import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "whats_your_bench"))

import numpy as np
import pytest
import problem_set
from problem_set import PROBLEM_CONFIGS, ParameterizedProblem


class TestProblemConfigs:
    def test_exactly_21_configs(self):
        assert len(PROBLEM_CONFIGS) == 21

    def test_all_configs_have_required_fields(self):
        for cfg in PROBLEM_CONFIGS:
            assert cfg.conjugate_prior is not None
            assert cfg.ppl_priors is not None
            assert cfg.sample_size > 0
            assert cfg.random_state is not None
            assert cfg.model_fn in ("normal_variance", "normal_mean", "mvnormal_covariance")

    def test_model_fn_distribution(self):
        fns = [cfg.model_fn for cfg in PROBLEM_CONFIGS]
        assert fns.count("normal_variance") == 6
        assert fns.count("normal_mean") == 9
        assert fns.count("mvnormal_covariance") == 6


class TestParameterizedProblemInit:
    """Smoke tests: instantiate problems without running MCMC."""

    def test_problem1_initialises(self):
        p = ParameterizedProblem(PROBLEM_CONFIGS[0])
        assert p.data.shape == (10,)
        assert p.conjugate_model is not None
        # posterior_predictive_params must have been computed
        params = p.conjugate_model.posterior_predictive_params
        assert hasattr(params, "loc")

    def test_problem7_initialises(self):
        p = ParameterizedProblem(PROBLEM_CONFIGS[6])
        assert p.data.shape == (10,)
        params = p.conjugate_model.posterior_predictive_params
        assert hasattr(params, "df")

    def test_problem16_initialises(self):
        p = ParameterizedProblem(PROBLEM_CONFIGS[15])
        assert p.data.shape[0] == 50
        assert p.data.shape[1] == 2
        params = p.conjugate_model.posterior_predictive_params
        assert hasattr(params, "mean")

    def test_ppl_priors_stored(self):
        p = ParameterizedProblem(PROBLEM_CONFIGS[0])
        assert p.ppl_priors == [0, 1]

    def test_models_initialised_to_none(self):
        p = ParameterizedProblem(PROBLEM_CONFIGS[0])
        assert p.models.pymc_model is None
        assert p.models.pyro_model is None
        assert p.models.stan_model is None
