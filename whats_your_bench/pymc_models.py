from whats_your_bench.utils import timer

import pymc as pm
import numpy as np
from types import SimpleNamespace

@timer
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

@timer
def normal_mean(priors, mean, data):

    prior_nu, prior_sigma = priors

    with pm.Model() as m:
        # Priors for unknown model parameters
        nu = pm.HalfNormal("nu", sigma = prior_nu)
        sigma = pm.HalfNormal("sigma", sigma = prior_sigma)
        
        # Likelihood
        obs = pm.StudentT("obs", nu = nu, mu = mean, sigma = sigma, observed = data)

        idata = pm.sample()

        return SimpleNamespace(
            df = float(idata.posterior["nu"].mean()),
            loc = mean,
            scale = float(idata.posterior["sigma"].mean())
        )


@timer
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


@timer
def mvnormal_mean(priors, mean, data):

    prior_beta, prior_eta, prior_nu = priors
    N, d = data.shape

    coords = {
        "axis": [f"x{i+1}" for i in range(d)],
        "axis_bis": [f"x{i+1}" for i in range(d)],
        "obs_id": np.arange(N)
        }

    with pm.Model(coords=coords) as model:

        nu = pm.HalfNormal("nu", prior_nu)

        chol, corr, stds = pm.LKJCholeskyCov(
            "chol", n=d, eta=prior_eta, sd_dist=pm.HalfCauchy.dist(beta = prior_beta, shape=d)
        )
        scale = pm.Deterministic("scale", chol.dot(chol.T), dims=("axis", "axis_bis"))

        obs = pm.MvStudentT("obs", nu=nu, mu = mean, scale = scale, observed=data, dims=("obs_id", "axis"))

        idata = pm.sample(
            idata_kwargs={"dims": {"chol_stds": ["axis"], "chol_corr": ["axis", "axis_bis"]}},
            cores = 1
        )
    
    return SimpleNamespace(
        df = idata.posterior.nu.mean().values,
        loc = mean,
        shape = idata.posterior.scale.mean(axis = 0).mean(axis = 0).values
    )