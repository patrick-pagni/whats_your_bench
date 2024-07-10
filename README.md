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

## Conjugate Priors

### Notation

| Expression | Definition |
|-----|-----|
|$P(x_i \| \theta)$ | Probability of an observation given the model and model parameters.|
|$\theta$| Unknown parameter for the likelihood model |
|$p(\theta \| \Theta)$| This is the conjugate prior distribution. This models the probability of the model parameter given the prior hyperparameters.|
|$p(\theta \| \Theta')$| This is the conjugate posterior distribution. This models the probability of the model parameters given the posterior hyperparameters.|
|$\Theta$| The parameters for the prior distribution. The prior distribution is the expected outcome of a variable before you have any information about it. This is the initial information from the dataset, meaning the parameters are the sample statistics from the data.|
|$\Theta'$| The parameters of the posterior distribution. This is the probabilitiy of some outcome given the evidence. 

| Likelihood $P(x_i\| \theta)$ | Model parameters $\theta$ | Conjugate prior (and posterior) distribution $p(\theta \| \Theta) = p(\theta \| \Theta')$| Prior hyperparameters $\Theta$ | Posterior hyperparameters $\Theta'$ | Posterior predictive $p(\tilde{x} \| X, \Theta)$ |
|-----|-----|-----|-----|-----|-----|
|Normal with known variance $\sigma^2$| $\mu$ (mean) | Normal | $\mu_0, \sigma_0^2$ | $\frac{1}{\frac{1}{\sigma_0^2} + \frac{n}{\sigma^2}}(\frac{\mu_0}{\sigma_0^2} + \frac{\sum_{i=1}^n x_i}{\sigma^2}),\\ (\frac{1}{\sigma_0^2} + \frac{n}{\sigma^2})^{-1}$|$\mathcal{N}(\tilde{x}\|{\mu_0}', {\sigma_0^2}' + \sigma^2)$|

## Potential distance metrics

### Kullback-Leibler Divergence

This is also known as relative entropy and l-divergence.

Measures how probability distribution $P(x)$ differs from a second reference distribution $Q(x)$. Can be interpreted as how much one can expect to be surprised when using distribution $Q$ to model data which follows the actual distirbution $P$.

For continuous random variables:

$$
D_{KL}(P||Q) = \int_{-\infty}^{\infty}p(x) \log{\frac{p(x)}{q(x)}}dx
$$

### KS-distance

### Pick largest difference between two n-dimensional arrays (L-infinity distance)

### Integral of the square between two surfaces

### Calculate the area where two distributions disagree

