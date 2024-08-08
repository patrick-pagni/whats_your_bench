import torch
import pyro
import pyro.distributions as pyro_dist
from types import SimpleNamespace

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

def normal_mean(prior_sigma, mean, data):

    def _setup_pyro_model():

        sigma = pyro.sample(
            "sigma",
            pyro_dist.HalfNormal(prior_sigma)
        )

        with pyro.plate("data", data.shape[0]):
            pyro.sample("obs", pyro_dist.Normal(mean, sigma), obs = torch.Tensor(data))

    kernel = pyro.infer.NUTS(_setup_pyro_model)

    mcmc = pyro.infer.MCMC(kernel, num_smaples = 1000, warmup_steps = 200)
    mcmc.run()

    hmc_samples = {k: v.detach().cpu().numpy() for k, v in mcmc.get_samples().items()}

    return SimpleNamespace(
        loc = mean,
        scale = hmc_samples["sigma"].mean(),
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

def mvnormal_mean(priors, mean, data):

    def _setup_pyro_model():
        N, M = data.shape

        # Sample the standard deviations with an exponential prior
        sigma = pyro.sample("sigma", pyro_dist.Exponential(prior_lambda).expand([M]))

        # Sample the correlation matrix using the LKJ prior
        Omega = pyro.sample("Omega", pyro_dist.LKJCholesky(M, prior_eta))
        Omega = torch.mm(Omega, Omega.T)

        # Construct the covariance matrix
        Sigma = torch.mm(torch.diag(sigma), torch.mm(Omega, torch.diag(sigma)))

        # Observe data
        with pyro.plate("observations", N):
            pyro.sample("obs", pyro_dist.MultivariateNormal(torch.Tensor(mean), covariance_matrix=Sigma), obs=torch.Tensor(data))

    prior_lambda, prior_eta = priors

    kernel = pyro.infer.NUTS(_setup_pyro_model)

    mcmc = pyro.infer.MCMC(kernel, num_samples = 1000, warmup_steps = 200)
    mcmc.run()

    hmc_samples = {k: v.detach().cpu() for k, v in mcmc.get_samples().items()}

    D = torch.diag(hmc_samples["sigma"].mean(axis = 0).sqrt())
    L = hmc_samples["Omega"].mean(axis = 0)

    covariance = D @ L @ L.T @ D

    return SimpleNamespace(
        mean = mean,
        cov = covariance.numpy()
    )