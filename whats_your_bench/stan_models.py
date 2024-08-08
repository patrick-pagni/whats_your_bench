import os
from cmdstanpy import CmdStanModel
from types import SimpleNamespace

def normal_variance(priors, variance, data):
    stan_file = os.path.join("../whats_your_bench/stan_models/normalKnownVar.stan")

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


def normal_variance(prior_sigma, mean, data):
    stan_file = os.path.join("../whats_your_bench/stan_models/normalKnownMean.stan")

    model = CmdStanModel(stan_file = stan_file)
        
    stan_data = {
        "N": data.shape[0],
        "X": data,
        "prior_sigma": prior_sigma,
        "obs_mean": mean
    }

    fit = model.sample(data = stan_data)

    return SimpleNamespace(
        loc = mean,
        scale = fit.sigma.mean(axis = 0)
    )

def mvnormal_covariance(priors, covariance, data):
    stan_file = os.path.join("../whats_your_bench/stan_models/mvNormalKnownCov.stan")

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

def mvnormal_mean(priors, mean, data):
    stan_file = os.path.join("../whats_your_bench/stan_models/mvNormalKnownMean.stan")

    model = CmdStanModel(stan_file = stan_file)

    prior_lambda, prior_eta = priors
    
    stan_data = {
        "N": data.shape[0],
        "M": data.shape[1],
        "X": data,
        "prior_lambda": prior_lambda,
        "prior_eta": prior_eta,
        "obs_mean": mean
    }

    fit = model.sample(data = stan_data)
    return SimpleNamespace(
        mean = mean,
        cov = fit.Sigma.mean(axis = 0)
    )