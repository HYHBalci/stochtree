# Test script for Integrated BCF Chiseler with Vapnik-inspired efficiency options
devtools::load_all()

set.seed(123)
n <- 250
p <- 4

# Covariates
X <- matrix(rnorm(n * p), n, p)
colnames(X) <- paste0("x", 1:p)

# True treatment effect: tau(x) = 1 + 2*x1
tau <- 1 + 2 * X[, 1]
mu <- X[, 2] - X[, 3]
pihat <- rep(0.5, n)
Z <- rbinom(n, 1, pihat)
Y <- mu + Z * tau + rnorm(n, 0, 0.5)

tau_c <- 1.0 # Patients benefit when tau(x) >= 1.0 (i.e. x1 >= 0)

cat("--- Testing bcf_integrated_chiseler with LOGIT & ADAPTIVE KAPPA SCHEDULE ---\n")
fit_adaptive <- bcf_integrated_chiseler(
  X_train = X,
  Z_train = Z,
  y_train = Y,
  propensity_train = pihat,
  tau_c = tau_c,
  kappa_weight = 2.0,
  boundary_link = "logit",
  kappa_schedule = "adaptive",
  cost_ratio = 1.2,
  num_gfr = 10,
  num_burnin = 50,
  num_mcmc = 100,
  general_params = list(verbose = FALSE)
)
cat("Adaptive chiseler fit completed successfully!\n")
cat("Boundary link:", fit_adaptive$boundary_link, "\n")
cat("Kappa schedule:", fit_adaptive$kappa_schedule, "\n")
cat("Cost ratio:", fit_adaptive$cost_ratio, "\n")
cat("Mean boundary coefficients:\n")
print(colMeans(fit_adaptive$beta_boundary_samples))

cat("\nAll tests completed successfully!\n")
