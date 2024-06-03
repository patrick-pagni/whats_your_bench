# README

File to eventually serve as the readme for the project.

For now, let's just try to outline what the project is trying to achieve.

There are a class of programming languages called probabilistic programming languages. Examples of these include:

- [pymc](https://www.pymc.io/welcome.html)
- [stan](https://mc-stan.org/)

These languages/libraries enable the user to build Bayesian models easily, and fit them using Markov chain Monte Carlo methods.

## PyMC3 Notes

To introduce model definition, fitting, and posterior analysis, we first consider a simple Bayesian linear regression model with normally-distributed priors for the parameters. We are interested in predicting outcomes $Y$ as normally-distributed observations with an expected value $\mu$ that is a linear function of two predictor variables, $X_1$ and $X_2$:

$$
Y \sim \mathcal{N}(\mu, \sigma^2)\\
\mu = \alpha + \beta_1X_1 + \beta_2X_2
$$

where $\alpha$ is the intercept, and $\beta_i$ is the coefficient for covariate $X_i$, while $\sigma$ represents the observation error. Since we are constructing a Bayesian model, we must assign a prior distribution to the unknown variables in the model. We choose zero-mean normal priors with variance of 100 for both regression coefficients, which corresponds to weak information regarding the true parameter values. We choose a half-normal distribution (normal distribution bounded at zero) as the prior for $\sigma$.

$$
\alpha \sim \mathcal{N}(0, 100)\\
\beta_i \sim \mathcal{N}(0,100)\\
\sigma \sim |\mathcal{N}(0, 100)|
$$