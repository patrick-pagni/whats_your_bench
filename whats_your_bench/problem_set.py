from whats_your_bench import conjugate_priors as cp
from whats_your_bench._problem import Problem

import numpy as np
from scipy import stats

"""
Import models
"""
from whats_your_bench import pymc_models, pyro_models, stan_models

class Problem01(Problem):

    def __init__(self):
        super().__init__(
            cp.NormalKnownVar(
                variance=1,
                prior_params={"mu": 3, "sigma": 1}
                 ),
            ppl_priors=[0, 1],
            sample_size=10,
            data_distribution=stats.norm(3, 1),
            random_state = 1
        )

    def run_models(self):
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

class Problem02(Problem):

    def __init__(self):
        super().__init__(
            cp.NormalKnownVar(
                variance=1,
                prior_params={"mu": 3, "sigma": 1}
                 ),
            ppl_priors=[0, 1],
            sample_size=50,
            data_distribution=stats.norm(3, 1),
            random_state = 2
        )

    def run_models(self):
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

class Problem03(Problem):

    def __init__(self):
        super().__init__(
            cp.NormalKnownVar(
                variance=1,
                prior_params={"mu": 3, "sigma": 1}
                 ),
            ppl_priors=[0, 1],
            sample_size=100,
            data_distribution=stats.norm(3, 1),
            random_state = 3
        )

    def run_models(self):
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

class Problem04(Problem):

    def __init__(self):
        super().__init__(
            cp.NormalKnownVar(
                variance=1,
                prior_params={"mu": 5, "sigma": 1}
                 ),
            ppl_priors=[0, 1],
            sample_size=10,
            data_distribution=stats.norm(5, 1),
            random_state = 1
        )

    def run_models(self):
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

class Problem05(Problem):

    def __init__(self):
        super().__init__(
            cp.NormalKnownVar(
                variance=1,
                prior_params={"mu": 5, "sigma": 1}
                 ),
            ppl_priors=[0, 1],
            sample_size=50,
            data_distribution=stats.norm(5, 1),
            random_state = 2
        )

    def run_models(self):
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

class Problem06(Problem):

    def __init__(self):
        super().__init__(
            cp.NormalKnownVar(
                variance=1,
                prior_params={"mu": 5, "sigma": 1}
                 ),
            ppl_priors=[0, 1],
            sample_size=100,
            data_distribution=stats.norm(5, 1),
            random_state = 3
        )

    def run_models(self):
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

"""
Problems designed to measure the impact of overly informative priors
"""
class Problem07(Problem):

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
            data_distribution=stats.norm(3, 1),
            random_state = 4
        )

    def run_models(self):
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

class Problem08(Problem):

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
            sample_size = 50,
            data_distribution=stats.norm(3, 1),
            random_state = 5
        )

    def run_models(self):
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

class Problem09(Problem):

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
            sample_size = 100,
            data_distribution=stats.norm(3, 1),
            random_state = 6
        )

    def run_models(self):
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

class Problem10(Problem):

    def __init__(self):
        super().__init__(
            cp.NormalKnownMean(
                mean = 3,
                prior_params = {"alpha": 1, "beta": 1}
            ),
            ppl_priors = [
                0.5,
                0.5
            ],
            sample_size = 10,
            data_distribution=stats.norm(3, 1),
            random_state = 4
        )

    def run_models(self):
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

class Problem11(Problem):

    def __init__(self):
        super().__init__(
            cp.NormalKnownMean(
                mean = 3,
                prior_params = {"alpha": 3, "beta": 5}
            ),
            ppl_priors = [
                0.5,
                0.5
            ],
            sample_size = 50,
            data_distribution=stats.norm(3, 1),
            random_state = 5
        )

    def run_models(self):
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

class Problem12(Problem):

    def __init__(self):
        super().__init__(
            cp.NormalKnownMean(
                mean = 3,
                prior_params = {"alpha": 3, "beta": 5}
            ),
            ppl_priors = [
                0.5,
                0.5
            ],
            sample_size = 100,
            data_distribution=stats.norm(3, 1),
            random_state = 6
        )

    def run_models(self):
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

class Problem13(Problem):

    def __init__(self):
        super().__init__(
            cp.NormalKnownMean(
                mean = 3,
                prior_params = {"alpha": 1, "beta": 1}
            ),
            ppl_priors = [
                0.1,
                0.1
            ],
            sample_size = 10,
            data_distribution=stats.norm(3, 1),
            random_state = 4
        )

    def run_models(self):
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

class Problem14(Problem):

    def __init__(self):
        super().__init__(
            cp.NormalKnownMean(
                mean = 3,
                prior_params = {"alpha": 3, "beta": 5}
            ),
            ppl_priors = [
                0.1,
                0.1
            ],
            sample_size = 50,
            data_distribution=stats.norm(3, 1),
            random_state = 5
        )

    def run_models(self):
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

class Problem15(Problem):

    def __init__(self):
        super().__init__(
            cp.NormalKnownMean(
                mean = 3,
                prior_params = {"alpha": 3, "beta": 5}
            ),
            ppl_priors = [
                0.1,
                0.1
            ],
            sample_size = 100,
            data_distribution=stats.norm(3, 1),
            random_state = 6
        )

    def run_models(self):
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

"""
Multivariate models

Testing how increased dimensionality increases execution times
"""

class Problem16(Problem):

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
            sample_size = 50,
            data_distribution = stats.multivariate_normal([3, 5], np.eye(2).tolist()),
            random_state = 7
        )

    def run_models(self):
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

class Problem17(Problem):

    def __init__(self):
        super().__init__(
            cp.MvNormalKnownCov(
                covariance = np.eye(2).tolist(), 
                prior_params={"mu": [10.0, 8.0], "sigma": np.eye(2).tolist()}
                ),
            ppl_priors = [
                [0, 0],
                np.eye(2).tolist()
                ],
            sample_size = 50,
            data_distribution = stats.multivariate_normal([3, 5], np.eye(2).tolist()),
            random_state = 8
        )
    
    def run_models(self):
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

class Problem18(Problem):

    def __init__(self):
        super().__init__(
            cp.MvNormalKnownCov(
                covariance = np.eye(2).tolist(), 
                prior_params={"mu": [14.0, 12.0], "sigma": np.eye(2).tolist()}
                ),
            ppl_priors = [
                [0, 0],
                np.eye(2).tolist()
                ],
            sample_size = 50,
            data_distribution = stats.multivariate_normal([3, 5], np.eye(2).tolist()),
            random_state = 9
        )

    def run_models(self):
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

class Problem19(Problem):

    def __init__(self):
        super().__init__(
            cp.MvNormalKnownCov(
                covariance = np.eye(5).tolist(), 
                prior_params={"mu": [3, 5, 4, 6, 7], "sigma": np.eye(5).tolist()}
                ),
            ppl_priors = [
                [0, 0, 0, 0, 0],
                np.eye(5).tolist()
                ],
            sample_size = 50,
            data_distribution = stats.multivariate_normal([3, 5, 3, 7, 4], np.eye(5).tolist()),
            random_state = 10
        )

    def run_models(self):
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

class Problem20(Problem):

    def __init__(self):
        super().__init__(
            cp.MvNormalKnownCov(
                covariance = np.eye(7).tolist(), 
                prior_params={"mu": [3, 5, 4, 6, 7, 8, 9], "sigma": np.eye(7).tolist()}
                ),
            ppl_priors = [
                [0, 0, 0, 0, 0, 0, 0],
                np.eye(7).tolist()
                ],
            sample_size = 50,
            data_distribution = stats.multivariate_normal([3, 5, 3, 7, 4, 8, 9], np.eye(7).tolist()),
            random_state = 11
        )

    def run_models(self):
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

class Problem21(Problem):

    def __init__(self):
        super().__init__(
            cp.MvNormalKnownCov(
                covariance = np.eye(10).tolist(), 
                prior_params={"mu": [3, 5, 4, 6, 7, 8, 9, 3, 3, 2], "sigma": np.eye(10).tolist()}
                ),
            ppl_priors = [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                np.eye(10).tolist()
                ],
            sample_size = 50,
            data_distribution = stats.multivariate_normal([3, 5, 3, 7, 4, 8, 9, 3, 2, 3], np.eye(10).tolist()),
            random_state = 12
        )

    def run_models(self):
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
    
"""
PPLs not properly implemented for multivariate normal with known mean
"""
# class Problem10(Problem):

#     def __init__(self):
#         super().__init__(
#             cp.MvNormalKnownMean(
#                 mean = [3, 5], 
#                 prior_params={"nu": 3, "psi": np.eye(2).tolist()}
#                 ),
#             ppl_priors = [
#                 100,
#                 100,
#                 100
#                 ],
#             sample_size = 10,
#             data_distribution = stats.multivariate_normal([3, 5], np.eye(2).tolist())
#         )
    
#     def run_models(self):
#         self.models.pymc_model, self.times.pymc_model = pymc_models.mvnormal_mean(
#             self.ppl_priors,
#             self.conjugate_model.mu,
#             self.data
#         )

#         self.models.pyro_model, self.times.pyro_model = pyro_models.mvnormal_mean(
#             self.ppl_priors,
#             self.conjugate_model.mu,
#             self.data
#         )

#         self.models.stan_model, self.times.stan_model = stan_models.mvnormal_mean(
#             self.ppl_priors,
#             self.conjugate_model.mu,
#             self.data
#         )

# class Problem11(Problem):

#     def __init__(self):
#         super().__init__(
#             cp.MvNormalKnownMean(
#                 mean = [3, 5], 
#                 prior_params={"nu": 3, "psi": np.eye(2).tolist()}
#                 ),
#             ppl_priors = [
#                 100,
#                 100,
#                 100
#                 ],
#             sample_size = 50,
#             data_distribution = stats.multivariate_normal([3, 5], np.eye(2).tolist())
#         )

#     def run_models(self):
#         self.models.pymc_model, self.times.pymc_model = pymc_models.mvnormal_mean(
#             self.ppl_priors,
#             self.conjugate_model.mu,
#             self.data
#         )

#         self.models.pyro_model, self.times.pyro_model = pyro_models.mvnormal_mean(
#             self.ppl_priors,
#             self.conjugate_model.mu,
#             self.data
#         )

#         self.models.stan_model, self.times.stan_model = stan_models.mvnormal_mean(
#             self.ppl_priors,
#             self.conjugate_model.mu,
#             self.data
#         )

# class Problem12(Problem):

#     def __init__(self):
#         super().__init__(
#             cp.MvNormalKnownMean(
#                 mean = [3, 5], 
#                 prior_params={"nu": 3, "psi": np.eye(2).tolist()}
#                 ),
#             ppl_priors = [
#                 100,
#                 100,
#                 100
#                 ],
#             sample_size = 100,
#             data_distribution = stats.multivariate_normal([3, 5], np.eye(2).tolist())
#         )

#     def run_models(self):
#         self.models.pymc_model, self.times.pymc_model = pymc_models.mvnormal_mean(
#             self.ppl_priors,
#             self.conjugate_model.mu,
#             self.data
#         )

#         self.models.pyro_model, self.times.pyro_model = pyro_models.mvnormal_mean(
#             self.ppl_priors,
#             self.conjugate_model.mu,
#             self.data
#         )

#         self.models.stan_model, self.times.stan_model = stan_models.mvnormal_mean(
#             self.ppl_priors,
#             self.conjugate_model.mu,
#             self.data
#         )