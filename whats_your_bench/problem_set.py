from whats_your_bench import conjugate_priors as cp
from scipy import stats

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

class Problem1(Problem):

    """
    First problem in problem set.

    Challenge: Given a random data set with a known variance, find the mean of the posterior.

    Parameters:
        Variance: 3
        Conjugate prior:
            Normal(mean = 3, variance = 1)
        PPL Priors [Uninformative Prior]:
            Normal(mean = 0, variance = 1)
        Data characterisitcs:
            Random sample from Normal(mean = 3, variance = 1)
            N = 100
    """

    def __init__(self):
        super().__init__(
            cp.NormalKnownVar(3, [5, 3]),
            [0, 1],
            100,
            stats.norm(3, 4)
        )

        self.ppl_mu, self.ppl_sigma = self.ppl_priors

    def _pymc_model(self):
        with pm.Model() as model:

            mu = pm.Normal("mu", mu = self.ppl_mu, sigma = self.ppl_sigma)
            obs = pm.Normal("obs", mu = mu, sigma = self.conjugate_model.sigma, observed = self.data)

            idata = pm.sample()

            return float(idata.posterior["mu"].mean())
        
    def _build_pyro_model(self):
        mu = pyro.sample("mu", pyro_dist.Normal(self.ppl_mu, self.ppl_sigma))

        with pyro.plate("data", self.data.shape[0]):
            pyro.sample("obs", pyro_dist.Normal(mu, self.conjugate_model.sigma), obs = torch.Tensor(self.data))

    def _pyro_model(self):
        kernel = pyro.infer.NUTS(self._build_pyro_model)

        mcmc = pyro.infer.MCMC(kernel, num_samples = 1000, warmup_steps = 200)
        mcmc.run()

        hmc_samples = {k: v.detach().cpu().numpy() for k, v in mcmc.get_samples().items()}

        return hmc_samples["mu"].mean()


        


