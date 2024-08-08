data {
int<lower=0> N;
int<lower=0> M;
matrix[N, M] X;
real prior_lambda;
real prior_eta;
vector[M] obs_mean;
}

parameters {
  corr_matrix[M] Omega; 
  vector<lower=0>[M] sigma; 
}
transformed parameters {
  cov_matrix[M] Sigma; 
  Sigma = quad_form_diag(Omega, sigma); 
}
model {
  for (n in 1:N)
    X[n] ~ multi_normal(obs_mean,Sigma); // sampling distribution of the observations
    sigma ~ exponential(prior_lambda); // prior on the standard deviations
    Omega ~ lkj_corr(prior_eta); // LKJ prior on the correlation matrix 
}