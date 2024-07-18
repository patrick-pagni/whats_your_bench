import numpy as np
from scipy import stats

class NormalKnownVar():
    def __init__(
            self,
            variance,
            prior_params,
            data
            ):
        
        self.prior_params = {
            "mean": prior_params[0],
            "variance": prior_params[1]
        }

        self.posterior_params = {
            "mean": np.power((1/prior_params[1])+(data.shape[0]/variance), -1)*((prior_params[0]/prior_params[1]) + (data.sum()/variance)),
            "variance": np.power((1 / prior_params[1]) + (data.shape[0]/variance), -1)
        }

        self.posterior_predictive_params = {
            "mean": self.posterior_params["mean"],
            "variance": self.posterior_params["variance"] + variance
        }

class NormalKnownMean():
    def __init__(
            self,
            mean,
            prior_params,
            data
            ):
        
        self.prior_params = {
            "alpha": prior_params[0],
            "beta": prior_params[1]
        }   

        self.posterior_params = {
            "alpha": prior_params[0] + data.shape[0]/2,
            "beta": prior_params[1] + 0.5 * np.sum(np.power(data - mean, 2))
        }

        self.posterior_predictive_params = {
            "k": self.posterior_params["alpha"],
            "loc": mean,
            "scale": np.sqrt(self.posterior_params["beta"]/self.posterior_params["alpha"])
        }

class MvNormalKnownCov():

    def __init__(
            self,
            covariance,
            prior_params,
            data
            ):
        
        self.prior_params = {
            "mean": prior_params[0],
            "covariance": prior_params[1]
        }   

        self.posterior_params = {
            "mean": np.linalg.inv(np.linalg.inv(prior_params[1]) + data.shape[0]*np.linalg.inv(covariance)) @ (np.linalg.inv(prior_params[1]) @ prior_params[0] + data.shape[0] * np.linalg.inv(covariance) @ data.mean(axis = 0)),
            "covariance": np.linalg.inv(np.linalg.inv(prior_params[1]) + data.shape[0]*np.linalg.inv(covariance))
        }

        self.posterior_predictive_params = {
            "mean": self.posterior_params["mean"],
            "covariance": self.posterior_params["covariance"] + covariance
        }

class MvNormalKnownMean():
    def __init__(
            self,
            mean,
            prior_params,
            data
            ):
        
        self.prior_params = {
            "df": prior_params[0],
            "scale": prior_params[1]
        }   

        self.posterior_params = {
            "df": data.shape[0] + prior_params[0],
            "scale": prior_params[1] + np.sum((data - mean) @ (data - mean).T)
        }

        self.posterior_predictive_params = {
            "df": self.posterior_params["df"],
            "scale": (1/(self.posterior_params["df"] - data.shape[1] + 1)) * self.posterior_params["scale"]
        }
