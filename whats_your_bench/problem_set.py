from whats_your_bench import conjugate_priors as cp
from whats_your_bench import distance

from scipy import stats
import numpy as np
from types import SimpleNamespace
import pandas as pd

"""
Import models
"""
from whats_your_bench import pymc_models, pyro_models, stan_models


class Problem():

    def __init__(
            self,
            conjugate_prior,
            ppl_priors,
            sample_size,
            data_distribution
            ):
        
        self.data = data_distribution.rvs(size = sample_size)

        self.conjugate_model = conjugate_prior
        self.conjugate_model.find_predictive_posterior(self.data)

        self.ppl_priors = ppl_priors

        self.models = SimpleNamespace(
            pymc_model = None,
            pyro_model = None,
            stan_model = None
        )

    def get_support_lim(self):
        true_params = self.conjugate_model.posterior_predictive_params

        if not true_params.mu.shape:
            ks_lim = true_params.mu + (5*true_params.sigma)
            kl_lim = [true_params.mu - (5*true_params.sigma), true_params.mu + (5*true_params.sigma)]

        else:
            ks_lim = (true_params.mu + (5*true_params.sigma.diagonal())).max()
            kl_lim = [true_params.mu - (5*true_params.sigma.diagonal()), true_params.mu + (5*true_params.sigma.diagonal())]

        self.support_lim = [ks_lim, kl_lim]

    def _model_dist(self, dist, params):

        return dist(**params)

    def get_distance(self, metric, p, q, support_lim):

        if metric == "ks_test":
            return distance.ks_test(p, q, support_lim)
        
        elif metric == "kl_divergence":
            return distance.kl_divergence(p, q, support_lim)
        
    def evaluate_models(self, dist):

        ppl = []
        ks_distances = []
        ks_scores = []
        kl_div = []

        q = self.conjugate_model.predictive_dist

        self.get_support_lim()

        for model in self.models.__dict__:
            ppl.append(model)
            p = self._model_dist(dist, getattr(self.models, model).__dict__)
            ks_distance, ks_score = self.get_distance("ks_test", p, q, self.support_lim[0])
            ks_distances.append(ks_distance)
            ks_scores.append(ks_score)
            kl_div.append(self.get_distance("kl_divergence", p, q, self.support_lim[1]))
        
        self.results = pd.DataFrame(zip(ppl, ks_distances, ks_scores, kl_div), columns = ["Language", "KS Distance", "KS Score", "KL Divergence"])
    

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

        self.models.pymc_model = pymc_models.normal_variance(
            self.ppl_priors,
            self.conjugate_model.sigma,
            self.data
            )

        self.models.pyro_model = pyro_models.normal_variance(
            self.ppl_priors,
            self.conjugate_model.sigma,
            self.data
        )

        self.models.stan_model = stan_models.normal_variance(
            self.ppl_priors,
            self.conjugate_model.sigma,
            self.data
        )

class Problem2(Problem):

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

        self.models.pymc_model = pymc_models.mvnormal_covariance(
            self.ppl_priors,
            self.conjugate_model.sigma,
            self.data
        )

        self.models.pyro_model = pyro_models.mvnormal_covariance(
            self.ppl_priors,
            self.conjugate_model.sigma,
            self.data
        )

        self.models.stan_model = stan_models.mvnormal_covariance(
            self.ppl_priors,
            self.conjugate_model.sigma,
            self.data
        )

class Problem3(Problem):

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

        self.models.pymc_model = pymc_models.mvnormal_covariance(
            self.ppl_priors,
            self.conjugate_model.sigma,
            self.data
        )

        self.models.pyro_model = pyro_models.mvnormal_covariance(
            self.ppl_priors,
            self.conjugate_model.sigma,
            self.data
        )

        self.models.stan_model = stan_models.mvnormal_covariance(
            self.ppl_priors,
            self.conjugate_model.sigma,
            self.data
        )

class Problem4(Problem):

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

        self.models.pymc_model = pymc_models.mvnormal_covariance(
            self.ppl_priors,
            self.conjugate_model.sigma,
            self.data
        )

        self.models.pyro_model = pyro_models.mvnormal_covariance(
            self.ppl_priors,
            self.conjugate_model.sigma,
            self.data
        )

        self.models.stan_model = stan_models.mvnormal_covariance(
            self.ppl_priors,
            self.conjugate_model.sigma,
            self.data
        )