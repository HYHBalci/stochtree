devtools::load_all()

# Generate synthetic data
set.seed(123)
n <- 200
p <- 5
X <- matrix(rnorm(n * p), n, p)
Z <- as.integer(rbinom(n, 1, 0.5))

# Simple DGP
tau <- X[,1] + X[,2]
mu <- X[,3]
Y <- mu + Z * tau + rnorm(n, 0, 0.1)
pihat <- rep(0.5, n)

# Run ranking BCF with ListNet (kl_divergence_method = "listnet")
cat("Running ranking BCF with ListNet...\n")
fit_listnet <- ranking_bcf(X_train = X, Z_train = Z, y_train = Y, propensity_train = pihat, kl_divergence_method = "listnet", num_mcmc = 50, num_burnin = 10, prognostic_forest_params = list(num_trees = 10), treatment_effect_forest_params = list(num_trees = 10))
cat("ListNet completed successfully.\n")

# Run ranking BCF with Distributional (kl_divergence_method = "distributional")
cat("Running ranking BCF with Distributional...\n")
fit_dist <- ranking_bcf(X_train = X, Z_train = Z, y_train = Y, propensity_train = pihat, kl_divergence_method = "distributional", num_mcmc = 50, num_burnin = 10, prognostic_forest_params = list(num_trees = 10), treatment_effect_forest_params = list(num_trees = 10))
cat("Distributional completed successfully.\n")

# Predictions
cat("Making predictions...\n")
pred_listnet <- predict(fit_listnet, X_test = X, Z_test = Z, propensity_test = pihat)
pred_dist <- predict(fit_dist, X_test = X, Z_test = Z, propensity_test = pihat)

cat("Predictions successful.\n")
cat("Finished test script.\n")
