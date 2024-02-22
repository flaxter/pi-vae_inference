data {
  int p; // input dimensionality (latent dimension of VAE)
  int p1; // hidden layer 1 number of units
  int p2; // hidden layer 2 number of units
  int n; // output dimensionality
  matrix[n,n] dist;
  vector[n] y;
  int ll_len;                    // length of indices for likelihood
  array[ll_len] int ll_idxs;           // indices for likelihood
}
parameters {
  real<lower=0> sigma2;
  real<lower=0> l;
  vector[n] f;
}
model {
  matrix[n, n] K; 
  K = exp(-0.5 * dist / l^2); // squared exponential kernel
  
  sigma2 ~ normal(0,0.025);
  f ~ multi_normal_cholesky(rep_vector(0, n),cholesky_decompose(K));
  y[ll_idxs] ~ normal(f[ll_idxs],sigma2);
}
