import pymc as pm
import numpy as np
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
    
def normal_mean(prior_sigma, mean, data):

    with pm.Model() as m:

        sigma = pm.HalfNormal("sigma", sigma = prior_sigma)
        obs = pm.Normal("obs", mu = 0, sigma = sigma, observed = data)

        idata = pm.sample()

        return SimpleNamespace(
            loc = mean,
            scale = float(idata.posterior["sigma"].mean())
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


def mvnormal_mean(priors, mean, data):

    prior_lambda, prior_eta = priors
    N, d = data.shape

    coords = {"axis": [f"x{i+1}" for i in range(d)], "axis_bis": [f"x{i+1}" for i in range(d)], "obs_id": np.arange(N)}

    with pm.Model(coords=coords):
        chol, corr, stds = pm.LKJCholeskyCov(
            "chol", n=d, eta=prior_eta, sd_dist=pm.Exponential.dist(prior_lambda, shape = d)
        )
        cov = pm.Deterministic("covariance", chol.dot(chol.T), dims=("axis", "axis_bis"))
        obs = pm.MvNormal("obs", mean, chol=chol, observed=data, dims=("obs_id", "axis"))

        idata = pm.sample(
            idata_kwargs={"dims": {"chol_stds": ["axis"], "chol_corr": ["axis", "axis_bis"]}},
        )
    
    return SimpleNamespace(
        mean = mean,
        cov = idata.posterior.covariance.mean(axis = 0).values
    )