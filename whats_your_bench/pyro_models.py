import torch
import pyro
import pyro.distributions as pyro_dist
from types import SimpleNamespace

def normal_variance(priors, variance, data):

    def _setup_pyro_model():
        mu = pyro.sample("mu", pyro_dist.Normal(prior_mu, prior_sigma))

        with pyro.plate("data", data.shape[0]):
            pyro.sample("obs", pyro_dist.Normal(mu, variance), obs = torch.Tensor(data))
        
        return None
    
    prior_mu, prior_sigma = priors

    kernel = pyro.infer.NUTS(_setup_pyro_model)

    mcmc = pyro.infer.MCMC(kernel, num_samples = 1000, warmup_steps = 200)
    mcmc.run()

    hmc_samples = {k: v.detach().cpu().numpy() for k, v in mcmc.get_samples().items()}

    return  SimpleNamespace(
        loc = hmc_samples["mu"].mean(),
        scale = variance
    )

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