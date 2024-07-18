import numpy as np
from scipy import stats

config = {
    "normal_known_variance": {
        "conjugate_prior": stats.norm,
        "prior": {
            "mu": 0,
            "sigma": 1
        },
        "posterior": {
            "mu": lambda p_mu, p_std, std, data: np.power((1/np.power(p_std, 2))+(data.shape[0]/np.power(std, 2)), -1)*((p_mu/np.power(p_std, 2)) + (data.sum()/np.power(std, 2))),
            "sigma": lambda p_std, std, data: np.power((1 / np.power(p_std, 2)) + (data.shape[0]/np.power(std, 2)), -1)
        },
        "posterior_predictive": {
            "mu": lambda mu: mu,
            "sigma": lambda p_std, std: np.power(p_std, 2) + np.power(std, 2)
        }
    }
}