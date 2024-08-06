from whats_your_bench import conjugate_priors as cp
from whats_your_bench import distance

from scipy import stats
import os
from types import SimpleNamespace
import pandas as pd

"""
Import for PyMC
"""
import pymc as pm

"""
Imports for pyro
"""
import torch
import pyro
import pyro.distributions as pyro_dist

"""
Imports for Stan
"""
from cmdstanpy import CmdStanModel


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

    def get_distance(self, metric, p, q, support_lim):

        if metric == "ks_test":
            return distance.ks_test(p, q, support_lim)
        
        elif metric == "kl_divergence":
            return distance.kl_divergence(p, q, support_lim)
        
    def _evaluate_models(self, dist, support_lim):

        ppl = []
        ks_test = []
        kl_div = []

        q = self.conjugate_model.predictive_dist

        for model in self.models.__dict__:
            ppl.append(model)
            p = self._model_dist(dist, getattr(self.models, model).__dict__)
            ks_test.append(self.get_distance("ks_test", p, q, support_lim[0]).item())
            kl_div.append(self.get_distance("kl_divergence", p, q, support_lim[1]))
        
        self.results = pd.DataFrame(zip(ppl, ks_test, kl_div), columns = ["Language", "KS Score", "KL Divergence"])
        
    def _model_dist(self, dist, params):

        return dist(**params)
    

class Problem1(Problem):

    """
    Challenge: Given a random data set with a known variance, find the mean of the posterior.

    Parameters:
        Variance: 1
        Conjugate prior:
            Normal(mean = 3, variance = 1)
        PPL Priors [Uninformative Prior]:
            Normal(mean = 0, variance = 1)
        Data characterisitcs:
            Random sample from Normal(mean = 3, variance = 1)
            N = 10
    """

    def __init__(self):
        super().__init__(
            cp.NormalKnownVar(
                1,
                {"mu": 3, "sigma": 1}
                 ),
            [0, 1],
            10,
            stats.norm(3, 1)
        )

        self.ppl_mu, self.ppl_sigma = self.ppl_priors

        self._run_models()

    def _pymc_model(self):
        with pm.Model() as model:

            mu = pm.Normal("mu", mu = self.ppl_mu, sigma = self.ppl_sigma)
            obs = pm.Normal("obs", mu = mu, sigma = self.conjugate_model.sigma, observed = self.data)

            idata = pm.sample()

            self.models.pymc_model = SimpleNamespace(
                loc = float(idata.posterior["mu"].mean()),
                scale = self.conjugate_model.sigma
            )
        
    def _setup_pyro_model(self):
        mu = pyro.sample("mu", pyro_dist.Normal(self.ppl_mu, self.ppl_sigma))

        with pyro.plate("data", self.data.shape[0]):
            pyro.sample("obs", pyro_dist.Normal(mu, self.conjugate_model.sigma), obs = torch.Tensor(self.data))

    def _pyro_model(self):
        kernel = pyro.infer.NUTS(self._setup_pyro_model)

        mcmc = pyro.infer.MCMC(kernel, num_samples = 1000, warmup_steps = 200)
        mcmc.run()

        hmc_samples = {k: v.detach().cpu().numpy() for k, v in mcmc.get_samples().items()}

        self.models.pyro_model = SimpleNamespace(
            loc = hmc_samples["mu"].mean(),
            scale = self.conjugate_model.sigma
        )
    
    def _stan_model(self):

        stan_file = os.path.join("../whats_your_bench/stan_models/normalKnownVar.stan")

        model = CmdStanModel(stan_file = stan_file)
        
        stan_data = {
            "N": self.data.shape[0],
            "X": self.data,
            "prior_mu": self.ppl_mu,
            "prior_sigma": self.ppl_sigma,
            "obs_sigma": self.conjugate_model.sigma
        }

        fit = model.sample(data = stan_data)
        self.models.stan_model = SimpleNamespace(
            loc = fit.mu.mean(axis = 0),
            scale = self.conjugate_model.sigma
        )

    def _run_models(self):
        self._pymc_model()
        self._pyro_model()
        self._stan_model()

    def get_support_lim(self):
        true_params = self.conjugate_model.posterior_predictive_params
        
        ks_lim = true_params.mu + (5*true_params.sigma)
        kl_lim = [true_params.mu - (5*true_params.sigma), true_params.mu + (5*true_params.sigma)]

        self.support_lim = [ks_lim, kl_lim]

    def evaluate_models(self, dist):

        self.get_support_lim()
        return self._evaluate_models(dist, self.support_lim)

class Problem2(Problem):

    """
    Challenge: Given a random data set with a known covariance, find the mean of the posterior.

    Parameters:
        Variance: [[1, 3], [3, 1]]
        Conjugate prior:
            MultivariateNormal(mean = [3, 5], covariance = [[1, 3], [3, 1]])
        PPL Priors [Uninformative Prior]:
            MvNormal(mean = [0, 0], covariance = [[1, 0], [0, 1]])
        Data characterisitcs:
            Random sample from MvNormal(mean = [3, 5], variance = [[1, 2.5], [2.5, 1]])
            N = 100
    """

    def __init__(self):
        super().__init__(
            cp.MvNormalKnownCov(
                [[1, 0], [0, 1]], 
                {"mu": [3, 5], "sigma": [[1, 0], [0, 1]]}
                ),
            [
                [0, 0],
                [[1, 0], [0, 1]]
                ],
            100,
            stats.multivariate_normal([3, 5], [[1, 0], [0, 1]])
        )

        self.ppl_mu, self.ppl_sigma = self.ppl_priors

        self._run_models()

    def _pymc_model(self):
        with pm.Model() as model:

            with pm.Model() as m:
                # Priors for unknown model parameters
                mu = pm.MvNormal(
                    "mu",
                    mu = self.ppl_mu,
                    cov = self.ppl_sigma,
                    shape = 2
                    )
                
                # Likelihood
                obs = pm.MvNormal(
                    "obs",
                    mu = mu,
                    cov = self.conjugate_model.sigma,
                    observed = self.data
                    )

                idata = pm.sample(cores = 1)

            self.models.pymc_model = SimpleNamespace(
                mean = idata.posterior["mu"].mean(axis = 0).mean(axis = 0).values,
                cov = self.conjugate_model.sigma
            )
        
    def _setup_pyro_model(self):
        mu = pyro.sample(
            "mu",
            pyro_dist.MultivariateNormal(
                torch.Tensor(self.ppl_mu),
                torch.Tensor(self.ppl_sigma)
                )
            )

        with pyro.plate("data", self.data.shape[0]):
            pyro.sample(
                "obs",
                pyro_dist.MultivariateNormal(
                    torch.Tensor(mu),
                    torch.Tensor(self.conjugate_model.sigma)
                    ),
                    obs = torch.Tensor(self.data)
                    )

    def _pyro_model(self):
        kernel = pyro.infer.NUTS(self._setup_pyro_model)

        mcmc = pyro.infer.MCMC(kernel, num_samples = 1000, warmup_steps = 200)
        mcmc.run()

        hmc_samples = {k: v.detach().cpu().numpy() for k, v in mcmc.get_samples().items()}

        self.models.pyro_model = SimpleNamespace(
            mean = hmc_samples["mu"].mean(axis = 0),
            cov = self.conjugate_model.sigma
        )
    
    def _stan_model(self):

        stan_file = os.path.join("../whats_your_bench/stan_models/mvNormalKnownCov.stan")

        model = CmdStanModel(stan_file = stan_file)
        
        stan_data = {
            "N": self.data.shape[0],
            "M": self.data.shape[1],
            "X": self.data,
            "prior_mu": self.ppl_mu,
            "prior_sigma": self.ppl_sigma,
            "obs_sigma": self.conjugate_model.sigma
        }

        fit = model.sample(data = stan_data)
        self.models.stan_model = SimpleNamespace(
            mean = fit.mu.mean(axis = 0),
            cov = self.conjugate_model.sigma
        )

    def _run_models(self):
        self._pymc_model()
        self._pyro_model()
        self._stan_model()

    def get_support_lim(self):
        true_params = self.conjugate_model.posterior_predictive_params
        
        ks_lim = (true_params.mu + (5*true_params.sigma.diagonal())).max()
        kl_lim = [true_params.mu - (5*true_params.sigma.diagonal()), true_params.mu + (5*true_params.sigma.diagonal())]

        self.support_lim = [ks_lim, kl_lim]

    def evaluate_models(self, dist):

        self.get_support_lim()
        return self._evaluate_models(dist, self.support_lim)