from whats_your_bench import conjugate_priors as cp
from scipy import stats
import pymc as pm

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

    def __init__(self):
        super().__init__(
            cp.NormalKnownVar(3, [5, 3]),
            [0, 1],
            10,
            stats.norm(3, 4)
        )

        self.ppl_mu, self.ppl_sigma = self.ppl_priors

    def _pymc_model(self):
        with pm.Model() as m:

            mu = pm.Normal("mu", mu = self.ppl_mu, sigma = self.ppl_sigma)
            obs = pm.Normal("obs", mu = mu, sigma = self.conjugate_model.sigma, observed = self.data)

            idata = pm.sample_prior_predictive()
            idata.extend(pm.sample())
            pm.sample_posterior_predictive(idata, extend_inferencedata=True)

        return idata
