import pymc as pm
from types import SimpleNamespace

def normal_variance(priors, variance, data):

    prior_mu, prior_sigma = priors
    with pm.Model():

        mu = pm.Normal("mu", mu = prior_mu, sigma = prior_sigma)
        obs = pm.Normal("obs", mu = mu, sigma = variance, observed = data)

        idata = pm.sample()

        return SimpleNamespace(
            loc = float(idata.posterior["mu"].mean()),
            scale = variance
        )
    
def mvnormal_covariance(priors, covariance, data):

    prior_mu, prior_sigma = priors
    d = data.shape[1]

    with pm.Model() as m:
        # Priors for unknown model parameters
        mu = pm.MvNormal(
            "mu",
            mu = prior_mu,
            cov = prior_sigma,
            shape = d
            )
        
        # Likelihood
        obs = pm.MvNormal(
            "obs",
            mu = mu,
            cov = covariance,
            observed = data
            )

        idata = pm.sample(cores = 1)

    return SimpleNamespace(
        mean = idata.posterior["mu"].mean(axis = 0).mean(axis = 0).values,
        cov = covariance
    )