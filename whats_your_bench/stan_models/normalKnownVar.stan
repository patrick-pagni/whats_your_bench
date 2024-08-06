data {
int<lower=0> N;
array[N] real X;
real prior_mu;
real prior_sigma;
real obs_sigma;
}

parameters {
real mu;
}

model {
mu ~ normal(prior_mu, prior_sigma);
X ~ normal(mu, obs_sigma);
}
