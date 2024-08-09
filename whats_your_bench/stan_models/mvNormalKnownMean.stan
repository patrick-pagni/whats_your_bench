data {
int<lower=0> N;
int<lower=0> M;
matrix[N, M] X;
real prior_beta;
real prior_eta;
real prior_nu;
vector[M] obs_mean;
}

parameters {
    vector<lower=0>[M] sigma;    // Standard deviations
    cholesky_factor_corr[M] L;   // Cholesky factor of the correlation matrix
    real<lower=0> nu;            // Degrees of freedom for the t-distribution
}

transformed parameters {
    matrix[M, M] Psi;
    Psi = diag_pre_multiply(sigma, L) * diag_pre_multiply(sigma, L)'; 
}

model {
    // Priors
    sigma ~ cauchy(0, prior_beta);                  // Half-Cauchy prior on standard deviations
    L ~ lkj_corr_cholesky(prior_eta);              // LKJ prior on the correlation matrix (via its Cholesky factor)
    nu ~ normal(0, prior_nu);                      // Prior on degrees of freedom
    
    // Likelihood
    for (n in 1:N)
        X[n] ~ multi_student_t(nu, obs_mean, Psi);  // Multivariate t-distribution likelihood
}