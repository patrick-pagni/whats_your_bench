from whats_your_bench import conjugate_priors as cp
from whats_your_bench._problem import Problem

import numpy as np
from scipy import stats
import time

"""
Import models
"""
from whats_your_bench import pymc_models, pyro_models, stan_models

class Problem1(Problem):

    def __init__(self):
        super().__init__(
            cp.NormalKnownVar(
                variance=1,
                prior_params={"mu": 3, "sigma": 1}
                 ),
            ppl_priors=[0, 1],
            sample_size=10,
            data_distribution=stats.norm(3, 1)
        )

        self.models.pymc_model, self.times.pymc_model = pymc_models.normal_variance(
            self.ppl_priors,
            self.conjugate_model.sigma,
            self.data
            )
        

        self.models.pyro_model, self.times.pyro_model = pyro_models.normal_variance(
            self.ppl_priors,
            self.conjugate_model.sigma,
            self.data
        )

        self.models.stan_model, self.times.stan_model = stan_models.normal_variance(
            self.ppl_priors,
            self.conjugate_model.sigma,
            self.data
        )

class Problem2(Problem):

    def __init__(self):
        super().__init__(
            cp.NormalKnownMean(
                mean = 3,
                prior_params = {"alpha": 1, "beta": 1}
            ),
            ppl_priors = [
                1,
                1
            ],
            sample_size = 10,
            data_distribution=stats.norm(3, 1)
        )

        self.models.pymc_model, self.times.pymc_model = pymc_models.normal_mean(
            self.ppl_priors,
            self.conjugate_model.mu,
            self.data
            )

        self.models.pyro_model, self.times.pyro_model = pyro_models.normal_mean(
            self.ppl_priors,
            self.conjugate_model.mu,
            self.data
        )

        self.models.stan_model, self.times.stan_model = stan_models.normal_mean(
            self.ppl_priors,
            self.conjugate_model.mu,
            self.data
        )

class Problem3(Problem):

    def __init__(self):
        super().__init__(
            cp.MvNormalKnownCov(
                covariance = np.eye(2).tolist(), 
                prior_params={"mu": [3, 5], "sigma": np.eye(2).tolist()}
                ),
            ppl_priors = [
                [0, 0],
                np.eye(2).tolist()
                ],
            sample_size = 100,
            data_distribution = stats.multivariate_normal([3, 5], np.eye(2).tolist())
        )

        self.models.pymc_model, self.times.pymc_model = pymc_models.mvnormal_covariance(
            self.ppl_priors,
            self.conjugate_model.sigma,
            self.data
        )

        self.models.pyro_model, self.times.pyro_model = pyro_models.mvnormal_covariance(
            self.ppl_priors,
            self.conjugate_model.sigma,
            self.data
        )

        self.models.stan_model, self.times.stan_model = stan_models.mvnormal_covariance(
            self.ppl_priors,
            self.conjugate_model.sigma,
            self.data
        )
    
class Problem4(Problem):

    def __init__(self):
        super().__init__(
            cp.MvNormalKnownMean(
                mean = [3, 5], 
                prior_params={"nu": 3, "psi": np.eye(2).tolist()}
                ),
            ppl_priors = [
                10,
                10,
                10
                ],
            sample_size = 100,
            data_distribution = stats.multivariate_normal([3, 5], np.eye(2).tolist())
        )

        self.models.pymc_model, self.times.pymc_model = pymc_models.mvnormal_mean(
            self.ppl_priors,
            self.conjugate_model.mu,
            self.data
        )

        self.models.pyro_model, self.times.pyro_model = pyro_models.mvnormal_mean(
            self.ppl_priors,
            self.conjugate_model.mu,
            self.data
        )

        self.models.stan_model, self.times.stan_model = stan_models.mvnormal_mean(
            self.ppl_priors,
            self.conjugate_model.mu,
            self.data
        )

class Problem5(Problem):

    def __init__(self):
        super().__init__(
            cp.MvNormalKnownCov(
                covariance = [
                    [0.83, 1.03, 0.5],
                    [1.03, 1.78, 0.55],
                    [0.5, 0.55, 0.38]
                ], 
                prior_params={"mu": [3, 5, 4], "sigma": np.eye(3).tolist()}
                ),
            ppl_priors = [
                [0, 0, 0],
                np.eye(3).tolist()
                ],
            sample_size = 100,
            data_distribution = stats.multivariate_normal([3.3, 5.1, 3.7], np.eye(3).tolist())
        )

        self.models.pymc_model, self.times.pymc_model = pymc_models.mvnormal_covariance(
            self.ppl_priors,
            self.conjugate_model.sigma,
            self.data
        )

        self.models.pyro_model, self.times.pyro_model = pyro_models.mvnormal_covariance(
            self.ppl_priors,
            self.conjugate_model.sigma,
            self.data
        )

        self.models.stan_model, self.times.stan_model = stan_models.mvnormal_covariance(
            self.ppl_priors,
            self.conjugate_model.sigma,
            self.data
        )

class Problem6(Problem):

    def __init__(self):
        super().__init__(
            cp.MvNormalKnownCov(
                covariance = [
                    [2.3689, 1.009, 1.0316, 0.534, 2.0336],
                    [1.009, 2.3008, 1.7977, 0.5032, 1.8492],
                    [1.0316, 1.7977, 1.4661, 0.4641, 1.5613],
                    [0.534, 0.5032, 0.4641, 0.2146, 0.5603],
                    [2.0336, 1.8492, 1.5613, 0.5603, 2.4378]
                ], 
                prior_params={"mu": [8.0, 4.0, 9.0, 7.0, 1.0], "sigma": np.eye(5).tolist()}
                ),
            ppl_priors = [
                [0, 0, 0, 0, 0],
                np.eye(5).tolist()
                ],
            sample_size = 100,
            data_distribution = stats.multivariate_normal([8.0, 4.0, 9.0, 7.0, 1.0], np.eye(5).tolist())
        )

        self.models.pymc_model, self.times.pymc_model = pymc_models.mvnormal_covariance(
            self.ppl_priors,
            self.conjugate_model.sigma,
            self.data
        )

        self.models.pyro_model, self.times.pyro_model = pyro_models.mvnormal_covariance(
            self.ppl_priors,
            self.conjugate_model.sigma,
            self.data
        )

        self.models.stan_model, self.times.stan_model = stan_models.mvnormal_covariance(
            self.ppl_priors,
            self.conjugate_model.sigma,
            self.data
        )