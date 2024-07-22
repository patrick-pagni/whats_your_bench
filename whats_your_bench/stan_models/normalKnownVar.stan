data {
int<lower=0> N;
array[N] real X;
}

parameters {
real mu;
}

model {
mu ~ normal(0, 1);  // uniform prior on interval 0,1
X ~ normal(mu, 3);
}
