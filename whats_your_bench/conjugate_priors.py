import numpy as np
from scipy import stats
from types import SimpleNamespace

class NormalKnownVar():
    def __init__(
            self,
            variance,
            prior_params,
            ):
        
        self.prior_params = SimpleNamespace(**prior_params)
        self.sigma = variance

    def find_predictive_posterior(
            self,
            data
            ):
        
        N = data.shape[0]

        self.posterior_params = SimpleNamespace(
            mu = np.power((1/self.prior_params.sigma)+(N/self.sigma), -1)*((self.prior_params.mu/self.prior_params.sigma) + (data.sum()/self.sigma)),
            sigma =  np.power((1 / self.prior_params.sigma) + (N/self.sigma), -1)
        )

        self.posterior_predictive_params = SimpleNamespace(
            mu = self.posterior_params.mu,
            sigma = self.posterior_params.sigma + self.sigma
        )

        self.predictive_dist = stats.norm(
            loc = self.posterior_predictive_params.mu,
            scale = np.sqrt(self.posterior_predictive_params.sigma)
            )

class NormalKnownMean():
    def __init__(
            self,
            mean,
            prior_params
            ):
        
        self.prior_params = SimpleNamespace(**prior_params)

        self.mu = mean

    def find_predictive_posterior(
            self,
            data
            ):
        
        N = data.shape[0]

        self.posterior_params = SimpleNamespace(
            alpha = self.prior_params.alpha + N/2,
            beta = self.prior_params.beta + 0.5 * np.sum(np.power(data - self.mu, 2))
        )

        self.posterior_predictive_params = SimpleNamespace(
            nu = 2 * self.posterior_params.alpha,
            loc = self.mu,
            scale = np.sqrt(self.posterior_params.beta/self.posterior_params.alpha)
        )

        self.predictive_dist = stats.t(
            self.posterior_predictive_params.nu,
            loc = self.posterior_predictive_params.loc,
            scale = self.posterior_predictive_params.scale
            )

class MvNormalKnownCov():

    def __init__(
            self,
            covariance,
            prior_params
            ):
        
        self.prior_params = SimpleNamespace(**prior_params)

        self.sigma = covariance

    def find_predictive_posterior(
            self,
            data
        ):

        N = data.shape[0]

        self.posterior_params = SimpleNamespace(
            mu = np.linalg.inv(np.linalg.inv(self.prior_params.sigma) + N*np.linalg.inv(self.sigma)) @ (np.linalg.inv(self.prior_params.sigma) @ self.prior_params.mu + N * (np.linalg.inv(self.sigma) @ data.mean(axis = 0))),
            sigma = np.linalg.inv(np.linalg.inv(self.prior_params.sigma) + N*np.linalg.inv(self.sigma))
        )

        self.posterior_predictive_params = SimpleNamespace(
            mu = self.posterior_params.mu,
            sigma = self.posterior_params.sigma + self.sigma
        )

        self.predictive_dist = stats.multivariate_normal(
            mean = self.posterior_predictive_params.mu,
            cov = self.posterior_predictive_params.sigma
            )

class MvNormalKnownMean():
    def __init__(
            self,
            mean,
            prior_params
            ):
        
        self.prior_params = SimpleNamespace(**prior_params)

        self.mu = mean

    def find_predictive_posterior(
            self,
            data
        ):

        N = data.shape[0]
        p = self.posterior_params.psi.shape[1]

        self.posterior_params = SimpleNamespace(
            nu = N + self.prior_params.nu,
            psi = self.prior_params.psi + np.sum((data - self.mu) @ (data - self.mu).T)
        )

        self.posterior_predictive_params = SimpleNamespace(
            nu = self.posterior_params.nu - p + 1,
            loc = self.mu,
            psi = (1/(self.posterior_params.nu - p + 1)) * self.posterior_params.psi
        )

        self.predictive_dist = stats.multivariate_t(
            self.posterior_predictive_params.nu,
            loc = self.mu,
            scale = self.posterior_predictive_params.psi
        )
