from utils import timer

import torch
import pyro
import pyro.distributions as pyro_dist
from types import SimpleNamespace

@timer
def normal_variance(priors, variance, data):

    def _setup_pyro_model():
        mu = pyro.sample("mu", pyro_dist.Normal(prior_mu, prior_sigma))

        with pyro.plate("data", data.shape[0]):
            pyro.sample("obs", pyro_dist.Normal(mu, variance), obs = torch.Tensor(data))
    
    prior_mu, prior_sigma = priors

    kernel = pyro.infer.NUTS(_setup_pyro_model)

    mcmc = pyro.infer.MCMC(kernel, num_samples = 1000, warmup_steps = 200)
    mcmc.run()

    hmc_samples = {k: v.detach().cpu().numpy() for k, v in mcmc.get_samples().items()}

    return  SimpleNamespace(
        loc = hmc_samples["mu"].mean(),
        scale = variance
    )

@timer
def normal_mean(priors, mean, data):

    prior_nu, prior_sigma = priors

    def _setup_pyro_model():

        nu = pyro.sample(
            "nu",
            pyro_dist.HalfNormal(prior_nu)
        )

        sigma = pyro.sample(
            "sigma",
            pyro_dist.HalfNormal(prior_sigma)
        )

        with pyro.plate("data", data.shape[0]):
            pyro.sample("obs", pyro_dist.StudentT(nu, mean, sigma), obs = torch.Tensor(data))

    kernel = pyro.infer.NUTS(_setup_pyro_model)

    mcmc = pyro.infer.MCMC(kernel, num_samples = 1000, warmup_steps = 200)
    mcmc.run()

    hmc_samples = {k: v.detach().cpu().numpy() for k, v in mcmc.get_samples().items()}

    return SimpleNamespace(
        df = hmc_samples["nu"].mean(),
        loc = mean,
        scale = hmc_samples["sigma"].mean().item()
    )
    
@timer
def mvnormal_covariance(priors, covariance, data):

    def _setup_pyro_model():
        mu = pyro.sample(
            "mu",
            pyro_dist.MultivariateNormal(
                torch.Tensor(prior_mu),
                torch.Tensor(prior_sigma)
                )
            )

        with pyro.plate("data", data.shape[0]):
            pyro.sample(
                "obs",
                pyro_dist.MultivariateNormal(
                    torch.Tensor(mu),
                    torch.Tensor(covariance)
                    ),
                    obs = torch.Tensor(data)
                    )

    prior_mu, prior_sigma = priors
    kernel = pyro.infer.NUTS(_setup_pyro_model)

    mcmc = pyro.infer.MCMC(kernel, num_samples = 1000, warmup_steps = 200)
    mcmc.run()

    hmc_samples = {k: v.detach().cpu().numpy() for k, v in mcmc.get_samples().items()}

    return SimpleNamespace(
        mean = hmc_samples["mu"].mean(axis = 0),
        cov = covariance
    )

@timer
def mvnormal_mean(priors, mean, data):

    def _setup_pyro_model():
        N, M = data.shape

        nu = pyro.sample("nu", pyro_dist.HalfNormal(prior_nu))

        # Sample the standard deviations with an exponential prior
        sigma = pyro.sample("sigma", pyro_dist.HalfCauchy(prior_beta).expand([M]))

        # Sample the correlation matrix using the LKJ prior
        Omega = pyro.sample("Omega", pyro_dist.LKJCholesky(M, prior_eta))

        # Construct the covariance matrix
        scale = torch.mm(torch.diag(sigma), torch.mm(Omega, torch.diag(sigma)))

        # Observe data
        with pyro.plate("observations", N):
            pyro.sample("obs", pyro_dist.MultivariateStudentT(
                df = nu,
                loc = torch.Tensor(mean),
                scale_tril=scale
                ), obs=torch.Tensor(data))

    prior_beta, prior_eta, prior_nu = priors

    kernel = pyro.infer.NUTS(_setup_pyro_model)

    mcmc = pyro.infer.MCMC(kernel, num_samples = 1000, warmup_steps = 200)
    mcmc.run()

    hmc_samples = {k: v.detach().cpu() for k, v in mcmc.get_samples().items()}

    D = torch.diag(hmc_samples["sigma"].mean(axis = 0).sqrt())
    L = hmc_samples["Omega"].mean(axis = 0)

    shape = D @ L @ L.T @ D

    return SimpleNamespace(
        df = hmc_samples["nu"].mean().item(),
        loc = mean,
        shape = shape.numpy()
    )