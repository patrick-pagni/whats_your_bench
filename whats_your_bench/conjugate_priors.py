import numpy as np
import numpy.typing as npt
from scipy import stats
from dataclasses import dataclass
from typing import Any, Union

@dataclass
class _NormalKnownVarPriorParams:
    mu: Any
    sigma: Any

@dataclass
class NormalKnownVarPosteriorParams:
    mu: Any
    sigma: Any

@dataclass
class NormalKnownVarPredictiveParams:
    loc: Any
    scale: Any

@dataclass
class _NormalKnownMeanPriorParams:
    alpha: Any
    beta: Any

@dataclass
class NormalKnownMeanPosteriorParams:
    alpha: Any
    beta: Any

@dataclass
class NormalKnownMeanPredictiveParams:
    df: Any
    loc: Any
    scale: Any

@dataclass
class _MvNormalKnownCovPriorParams:
    mu: Any
    sigma: Any

@dataclass
class MvNormalKnownCovPosteriorParams:
    mu: Any
    sigma: Any

@dataclass
class MvNormalKnownCovPredictiveParams:
    mean: Any
    cov: Any

@dataclass
class _MvNormalKnownMeanPriorParams:
    nu: Any
    psi: Any

@dataclass
class MvNormalKnownMeanPosteriorParams:
    nu: Any
    psi: Any

@dataclass
class MvNormalKnownMeanPredictiveParams:
    df: Any
    loc: Any
    shape: Any

class NormalKnownVar():
    def __init__(
            self,
            variance: float,
            prior_params: dict[str, float],
            ) -> None:

        self.prior_params = _NormalKnownVarPriorParams(**prior_params)
        self.sigma = variance

    def find_predictive_posterior(
            self,
            data: npt.NDArray
            ) -> None:

        N = data.shape[0]

        self.posterior_params = NormalKnownVarPosteriorParams(
            mu = np.power((1/self.prior_params.sigma)+(N/self.sigma), -1)*((self.prior_params.mu/self.prior_params.sigma) + (data.sum()/self.sigma)),
            sigma =  np.power((1 / self.prior_params.sigma) + (N/self.sigma), -1)
        )

        self.posterior_predictive_params = NormalKnownVarPredictiveParams(
            loc = self.posterior_params.mu,
            scale = np.sqrt(self.posterior_params.sigma + self.prior_params.sigma)
        )

        self.predictive_dist = stats.norm

class NormalKnownMean():
    def __init__(
            self,
            mean: float,
            prior_params: dict[str, float]
            ) -> None:

        self.prior_params = _NormalKnownMeanPriorParams(**prior_params)

        self.mu = mean

    def find_predictive_posterior(
            self,
            data: npt.NDArray
            ) -> None:

        N = data.shape[0]

        self.posterior_params = NormalKnownMeanPosteriorParams(
            alpha = self.prior_params.alpha + N/2,
            beta = self.prior_params.beta + 0.5 * np.sum(np.power(data - self.mu, 2))
        )

        self.posterior_predictive_params = NormalKnownMeanPredictiveParams(
            df = 2 * self.posterior_params.alpha,
            loc = self.mu,
            scale = np.sqrt(self.posterior_params.beta/self.posterior_params.alpha)
        )

        self.predictive_dist = stats.t

class MvNormalKnownCov():

    def __init__(
            self,
            covariance: npt.NDArray,
            prior_params: dict[str, Union[npt.NDArray, float]]
            ) -> None:

        self.prior_params = _MvNormalKnownCovPriorParams(**prior_params)

        self.sigma = covariance

    def find_predictive_posterior(
            self,
            data: npt.NDArray
        ) -> None:

        N = data.shape[0]

        self.posterior_params = MvNormalKnownCovPosteriorParams(
            mu = np.linalg.inv(np.linalg.inv(self.prior_params.sigma) + N*np.linalg.inv(self.sigma)) @ (np.linalg.inv(self.prior_params.sigma) @ self.prior_params.mu + N * (np.linalg.inv(self.sigma) @ data.mean(axis = 0))),
            sigma = np.linalg.inv(np.linalg.inv(self.prior_params.sigma) + N*np.linalg.inv(self.sigma))
        )

        self.posterior_predictive_params = MvNormalKnownCovPredictiveParams(
            mean = self.posterior_params.mu,
            cov = self.posterior_params.sigma + self.sigma
        )

        self.predictive_dist = stats.multivariate_normal

class MvNormalKnownMean():
    def __init__(
            self,
            mean: npt.NDArray,
            prior_params: dict[str, Union[npt.NDArray, float]]
            ) -> None:

        self.prior_params = _MvNormalKnownMeanPriorParams(**prior_params)

        self.mu = mean

    def find_predictive_posterior(
            self,
            data: npt.NDArray
        ) -> None:

        N = data.shape[0]
        p = data.shape[1]

        self.posterior_params = MvNormalKnownMeanPosteriorParams(
            nu = N + self.prior_params.nu,
            psi = self.prior_params.psi + np.sum((data - self.mu) @ (data - self.mu).T)
        )

        self.posterior_predictive_params = MvNormalKnownMeanPredictiveParams(
            df = self.posterior_params.nu - p + 1,
            loc = self.mu,
            shape = (1/(self.posterior_params.nu - p + 1)) * self.posterior_params.psi
        )

        self.predictive_dist = stats.multivariate_t
