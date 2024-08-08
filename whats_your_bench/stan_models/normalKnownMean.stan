data {
int<lower=0> N;
array[N] real X;
real prior_sigma;
real obs_mean;
}

parameters {
real<lower=0> sigma;
}

model {
sigma ~ normal(0, prior_sigma);
X ~ normal(obs_mean, sigma);
}
