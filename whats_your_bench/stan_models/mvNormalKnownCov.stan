data {
int<lower=0> N;
int<lower=0> M;
matrix[N, M] X;
vector[M] prior_mu;
matrix[M, M] prior_sigma;
matrix[M, M] obs_sigma;
}

parameters {
vector[M] mu;
}

model {
mu ~ multi_normal(prior_mu, prior_sigma);
for (n in 1:N)
    X[n] ~ multi_normal(mu, obs_sigma);
}