from utils import timer

import os
from pathlib import Path
from cmdstanpy import CmdStanModel
from types import SimpleNamespace

_STAN_DIR = Path(__file__).parent / "stan_models"

@timer
def normal_variance(priors, variance, data):
    stan_file = _STAN_DIR / "normalKnownVar.stan"

    model = CmdStanModel(stan_file = stan_file)

    prior_mu, prior_sigma = priors
        
    stan_data = {
        "N": data.shape[0],
        "X": data,
        "prior_mu": prior_mu,
        "prior_sigma": prior_sigma,
        "obs_sigma": variance
    }

    fit = model.sample(data = stan_data)

    return SimpleNamespace(
        loc = fit.mu.mean(axis = 0),
        scale = variance
    )

@timer
def normal_mean(priors, mean, data):
    stan_file = _STAN_DIR / "normalKnownMean.stan"

    model = CmdStanModel(stan_file = stan_file)

    prior_nu, prior_sigma = priors
        
    stan_data = {
        "N": data.shape[0],
        "X": data,
        "prior_nu": prior_nu,
        "prior_sigma": prior_sigma,
        "obs_mean": mean
    }

    fit = model.sample(data = stan_data)

    return SimpleNamespace(
        df = fit.nu.mean(axis = 0),
        loc = mean,
        scale = fit.sigma.mean(axis = 0)
    )

@timer
def mvnormal_covariance(priors, covariance, data):
    stan_file = _STAN_DIR / "mvNormalKnownCov.stan"

    model = CmdStanModel(stan_file = stan_file)

    prior_mu, prior_sigma = priors
    
    stan_data = {
        "N": data.shape[0],
        "M": data.shape[1],
        "X": data,
        "prior_mu": prior_mu,
        "prior_sigma": prior_sigma,
        "obs_sigma": covariance
    }

    fit = model.sample(data = stan_data)
    return SimpleNamespace(
        mean = fit.mu.mean(axis = 0),
        cov = covariance
    )

@timer
def mvnormal_mean(priors, mean, data):
    stan_file = _STAN_DIR / "mvNormalKnownMean.stan"

    model = CmdStanModel(stan_file = stan_file)

    prior_beta, prior_eta, prior_nu = priors
    
    stan_data = {
        "N": data.shape[0],
        "M": data.shape[1],
        "X": data,
        "prior_beta": prior_beta,
        "prior_eta": prior_eta,
        "prior_nu": prior_nu,
        "obs_mean": mean
    }

    fit = model.sample(data = stan_data)
    return SimpleNamespace(
        df = fit.nu.mean(),
        loc = mean,
        shape = fit.Psi.mean(axis = 0)
    )
