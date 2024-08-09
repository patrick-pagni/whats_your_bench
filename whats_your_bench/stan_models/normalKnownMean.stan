data {
    int<lower=0> N;
    array[N] real X;
    real prior_nu;
    real prior_sigma;
    real obs_mean;
}

parameters {
    real<lower = 0> nu;
    real<lower=0> sigma;
}

model {
    nu ~ normal(0, prior_nu);
    sigma ~ normal(0, prior_sigma);
    X ~ student_t(nu, obs_mean, sigma);
}
