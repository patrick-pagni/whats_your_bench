import conjugate_priors as cp
from _problem import Problem

import numpy as np
from scipy import stats

from dataclasses import dataclass
from typing import Any

"""
Import models
"""
import pymc_models, pyro_models, stan_models


@dataclass
class ProblemConfig:
    conjugate_prior: Any       # already-instantiated conjugate prior object
    ppl_priors: list
    data_distribution: Any     # already-instantiated scipy distribution object
    model_fn: str              # one of: "normal_variance", "normal_mean", "mvnormal_covariance"
    sample_size: int
    random_state: int


PROBLEM_CONFIGS: list[ProblemConfig] = [
    # Problem01 – normal, known variance, loc=3, n=10
    ProblemConfig(
        conjugate_prior=cp.NormalKnownVar(
            variance=1,
            prior_params={"mu": 3, "sigma": 1}
        ),
        ppl_priors=[0, 1],
        data_distribution=stats.norm(3, 1),
        model_fn="normal_variance",
        sample_size=10,
        random_state=1,
    ),
    # Problem02 – normal, known variance, loc=3, n=50
    ProblemConfig(
        conjugate_prior=cp.NormalKnownVar(
            variance=1,
            prior_params={"mu": 3, "sigma": 1}
        ),
        ppl_priors=[0, 1],
        data_distribution=stats.norm(3, 1),
        model_fn="normal_variance",
        sample_size=50,
        random_state=2,
    ),
    # Problem03 – normal, known variance, loc=3, n=100
    ProblemConfig(
        conjugate_prior=cp.NormalKnownVar(
            variance=1,
            prior_params={"mu": 3, "sigma": 1}
        ),
        ppl_priors=[0, 1],
        data_distribution=stats.norm(3, 1),
        model_fn="normal_variance",
        sample_size=100,
        random_state=3,
    ),
    # Problem04 – normal, known variance, loc=5, n=10
    ProblemConfig(
        conjugate_prior=cp.NormalKnownVar(
            variance=1,
            prior_params={"mu": 5, "sigma": 1}
        ),
        ppl_priors=[0, 1],
        data_distribution=stats.norm(5, 1),
        model_fn="normal_variance",
        sample_size=10,
        random_state=1,
    ),
    # Problem05 – normal, known variance, loc=5, n=50
    ProblemConfig(
        conjugate_prior=cp.NormalKnownVar(
            variance=1,
            prior_params={"mu": 5, "sigma": 1}
        ),
        ppl_priors=[0, 1],
        data_distribution=stats.norm(5, 1),
        model_fn="normal_variance",
        sample_size=50,
        random_state=2,
    ),
    # Problem06 – normal, known variance, loc=5, n=100
    ProblemConfig(
        conjugate_prior=cp.NormalKnownVar(
            variance=1,
            prior_params={"mu": 5, "sigma": 1}
        ),
        ppl_priors=[0, 1],
        data_distribution=stats.norm(5, 1),
        model_fn="normal_variance",
        sample_size=100,
        random_state=3,
    ),
    # Problem07 – normal, known mean, uninformative prior, n=10
    ProblemConfig(
        conjugate_prior=cp.NormalKnownMean(
            mean=3,
            prior_params={"alpha": 1, "beta": 1}
        ),
        ppl_priors=[1, 1],
        data_distribution=stats.norm(3, 1),
        model_fn="normal_mean",
        sample_size=10,
        random_state=4,
    ),
    # Problem08 – normal, known mean, uninformative prior, n=50
    ProblemConfig(
        conjugate_prior=cp.NormalKnownMean(
            mean=3,
            prior_params={"alpha": 1, "beta": 1}
        ),
        ppl_priors=[1, 1],
        data_distribution=stats.norm(3, 1),
        model_fn="normal_mean",
        sample_size=50,
        random_state=5,
    ),
    # Problem09 – normal, known mean, uninformative prior, n=100
    ProblemConfig(
        conjugate_prior=cp.NormalKnownMean(
            mean=3,
            prior_params={"alpha": 1, "beta": 1}
        ),
        ppl_priors=[1, 1],
        data_distribution=stats.norm(3, 1),
        model_fn="normal_mean",
        sample_size=100,
        random_state=6,
    ),
    # Problem10 – normal, known mean, weakly-informative prior (0.5), n=10
    ProblemConfig(
        conjugate_prior=cp.NormalKnownMean(
            mean=3,
            prior_params={"alpha": 1, "beta": 1}
        ),
        ppl_priors=[0.5, 0.5],
        data_distribution=stats.norm(3, 1),
        model_fn="normal_mean",
        sample_size=10,
        random_state=4,
    ),
    # Problem11 – normal, known mean, informative prior (alpha=3,beta=5), n=50
    ProblemConfig(
        conjugate_prior=cp.NormalKnownMean(
            mean=3,
            prior_params={"alpha": 3, "beta": 5}
        ),
        ppl_priors=[0.5, 0.5],
        data_distribution=stats.norm(3, 1),
        model_fn="normal_mean",
        sample_size=50,
        random_state=5,
    ),
    # Problem12 – normal, known mean, informative prior (alpha=3,beta=5), n=100
    ProblemConfig(
        conjugate_prior=cp.NormalKnownMean(
            mean=3,
            prior_params={"alpha": 3, "beta": 5}
        ),
        ppl_priors=[0.5, 0.5],
        data_distribution=stats.norm(3, 1),
        model_fn="normal_mean",
        sample_size=100,
        random_state=6,
    ),
    # Problem13 – normal, known mean, vague prior (0.1), n=10
    ProblemConfig(
        conjugate_prior=cp.NormalKnownMean(
            mean=3,
            prior_params={"alpha": 1, "beta": 1}
        ),
        ppl_priors=[0.1, 0.1],
        data_distribution=stats.norm(3, 1),
        model_fn="normal_mean",
        sample_size=10,
        random_state=4,
    ),
    # Problem14 – normal, known mean, informative prior (alpha=3,beta=5), vague ppl_priors, n=50
    ProblemConfig(
        conjugate_prior=cp.NormalKnownMean(
            mean=3,
            prior_params={"alpha": 3, "beta": 5}
        ),
        ppl_priors=[0.1, 0.1],
        data_distribution=stats.norm(3, 1),
        model_fn="normal_mean",
        sample_size=50,
        random_state=5,
    ),
    # Problem15 – normal, known mean, informative prior (alpha=3,beta=5), vague ppl_priors, n=100
    ProblemConfig(
        conjugate_prior=cp.NormalKnownMean(
            mean=3,
            prior_params={"alpha": 3, "beta": 5}
        ),
        ppl_priors=[0.1, 0.1],
        data_distribution=stats.norm(3, 1),
        model_fn="normal_mean",
        sample_size=100,
        random_state=6,
    ),
    # Problem16 – mvnormal, 2-D, well-specified prior
    ProblemConfig(
        conjugate_prior=cp.MvNormalKnownCov(
            covariance=np.eye(2).tolist(),
            prior_params={"mu": [3, 5], "sigma": np.eye(2).tolist()}
        ),
        ppl_priors=[[0, 0], np.eye(2).tolist()],
        data_distribution=stats.multivariate_normal([3, 5], np.eye(2).tolist()),
        model_fn="mvnormal_covariance",
        sample_size=50,
        random_state=7,
    ),
    # Problem17 – mvnormal, 2-D, misspecified prior (mu=[10,8])
    ProblemConfig(
        conjugate_prior=cp.MvNormalKnownCov(
            covariance=np.eye(2).tolist(),
            prior_params={"mu": [10.0, 8.0], "sigma": np.eye(2).tolist()}
        ),
        ppl_priors=[[0, 0], np.eye(2).tolist()],
        data_distribution=stats.multivariate_normal([3, 5], np.eye(2).tolist()),
        model_fn="mvnormal_covariance",
        sample_size=50,
        random_state=8,
    ),
    # Problem18 – mvnormal, 2-D, heavily misspecified prior (mu=[14,12])
    ProblemConfig(
        conjugate_prior=cp.MvNormalKnownCov(
            covariance=np.eye(2).tolist(),
            prior_params={"mu": [14.0, 12.0], "sigma": np.eye(2).tolist()}
        ),
        ppl_priors=[[0, 0], np.eye(2).tolist()],
        data_distribution=stats.multivariate_normal([3, 5], np.eye(2).tolist()),
        model_fn="mvnormal_covariance",
        sample_size=50,
        random_state=9,
    ),
    # Problem19 – mvnormal, 5-D
    ProblemConfig(
        conjugate_prior=cp.MvNormalKnownCov(
            covariance=np.eye(5).tolist(),
            prior_params={"mu": [3, 5, 4, 6, 7], "sigma": np.eye(5).tolist()}
        ),
        ppl_priors=[[0, 0, 0, 0, 0], np.eye(5).tolist()],
        data_distribution=stats.multivariate_normal([3, 5, 3, 7, 4], np.eye(5).tolist()),
        model_fn="mvnormal_covariance",
        sample_size=50,
        random_state=10,
    ),
    # Problem20 – mvnormal, 7-D
    ProblemConfig(
        conjugate_prior=cp.MvNormalKnownCov(
            covariance=np.eye(7).tolist(),
            prior_params={"mu": [3, 5, 4, 6, 7, 8, 9], "sigma": np.eye(7).tolist()}
        ),
        ppl_priors=[[0, 0, 0, 0, 0, 0, 0], np.eye(7).tolist()],
        data_distribution=stats.multivariate_normal([3, 5, 3, 7, 4, 8, 9], np.eye(7).tolist()),
        model_fn="mvnormal_covariance",
        sample_size=50,
        random_state=11,
    ),
    # Problem21 – mvnormal, 10-D
    ProblemConfig(
        conjugate_prior=cp.MvNormalKnownCov(
            covariance=np.eye(10).tolist(),
            prior_params={"mu": [3, 5, 4, 6, 7, 8, 9, 3, 3, 2], "sigma": np.eye(10).tolist()}
        ),
        ppl_priors=[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], np.eye(10).tolist()],
        data_distribution=stats.multivariate_normal([3, 5, 3, 7, 4, 8, 9, 3, 2, 3], np.eye(10).tolist()),
        model_fn="mvnormal_covariance",
        sample_size=50,
        random_state=12,
    ),
]


_MODEL_FN_MAP = {
    "normal_variance": (
        pymc_models.normal_variance,
        pyro_models.normal_variance,
        stan_models.normal_variance,
    ),
    "normal_mean": (
        pymc_models.normal_mean,
        pyro_models.normal_mean,
        stan_models.normal_mean,
    ),
    "mvnormal_covariance": (
        pymc_models.mvnormal_covariance,
        pyro_models.mvnormal_covariance,
        stan_models.mvnormal_covariance,
    ),
}

# Attribute name on the conjugate model that holds the posterior parameter
# forwarded to the PPL model functions.
_POSTERIOR_ATTR = {
    "normal_variance": "sigma",
    "normal_mean": "mu",
    "mvnormal_covariance": "sigma",
}


class ParameterizedProblem(Problem):
    """A single Problem subclass driven by a ProblemConfig instance."""

    def __init__(self, config: ProblemConfig):
        super().__init__(
            config.conjugate_prior,
            ppl_priors=config.ppl_priors,
            sample_size=config.sample_size,
            data_distribution=config.data_distribution,
            random_state=config.random_state,
        )
        self._model_fn = config.model_fn

    def run_models(self):
        pymc_fn, pyro_fn, stan_fn = _MODEL_FN_MAP[self._model_fn]
        posterior = getattr(self.conjugate_model, _POSTERIOR_ATTR[self._model_fn])

        self.models.pymc_model, self.times.pymc_model = pymc_fn(
            self.ppl_priors, posterior, self.data
        )
        self.models.pyro_model, self.times.pyro_model = pyro_fn(
            self.ppl_priors, posterior, self.data
        )
        self.models.stan_model, self.times.stan_model = stan_fn(
            self.ppl_priors, posterior, self.data
        )


# ---------------------------------------------------------------------------
# Backwards-compatible aliases so that code using problem_set.Problem01() etc.
# continues to work unchanged.
# ---------------------------------------------------------------------------
Problem01  = lambda: ParameterizedProblem(PROBLEM_CONFIGS[0])
Problem02  = lambda: ParameterizedProblem(PROBLEM_CONFIGS[1])
Problem03  = lambda: ParameterizedProblem(PROBLEM_CONFIGS[2])
Problem04  = lambda: ParameterizedProblem(PROBLEM_CONFIGS[3])
Problem05  = lambda: ParameterizedProblem(PROBLEM_CONFIGS[4])
Problem06  = lambda: ParameterizedProblem(PROBLEM_CONFIGS[5])
Problem07  = lambda: ParameterizedProblem(PROBLEM_CONFIGS[6])
Problem08  = lambda: ParameterizedProblem(PROBLEM_CONFIGS[7])
Problem09  = lambda: ParameterizedProblem(PROBLEM_CONFIGS[8])
Problem10  = lambda: ParameterizedProblem(PROBLEM_CONFIGS[9])
Problem11  = lambda: ParameterizedProblem(PROBLEM_CONFIGS[10])
Problem12  = lambda: ParameterizedProblem(PROBLEM_CONFIGS[11])
Problem13  = lambda: ParameterizedProblem(PROBLEM_CONFIGS[12])
Problem14  = lambda: ParameterizedProblem(PROBLEM_CONFIGS[13])
Problem15  = lambda: ParameterizedProblem(PROBLEM_CONFIGS[14])
Problem16  = lambda: ParameterizedProblem(PROBLEM_CONFIGS[15])
Problem17  = lambda: ParameterizedProblem(PROBLEM_CONFIGS[16])
Problem18  = lambda: ParameterizedProblem(PROBLEM_CONFIGS[17])
Problem19  = lambda: ParameterizedProblem(PROBLEM_CONFIGS[18])
Problem20  = lambda: ParameterizedProblem(PROBLEM_CONFIGS[19])
Problem21  = lambda: ParameterizedProblem(PROBLEM_CONFIGS[20])
