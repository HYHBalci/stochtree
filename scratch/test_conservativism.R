devtools::load_all()

set.seed(42)
n <- 600
p <- 5

X <- matrix(runif(n * p, -2, 2), n, p)
colnames(X) <- paste0("x", 1:p)

# True CATE: tau(x) = x1 + 0.5*x2 (smooth transition through 0)
tau <- X[, 1] + 0.5 * X[, 2]
tau_c <- 0.0

# Base prognostic effect: mu(x) = 2*x3 - x4
mu <- 2 * X[, 3] - X[, 4]
pihat <- rep(0.5, n)
Z <- rbinom(n, 1, pihat)
# High noise to test conservativism
Y <- mu + Z * tau + rnorm(n, 0, 2.5) 

cat("Fitting BCF Integrated Chiseler...\n")
fit <- bcf_integrated_chiseler(
  X_train = X,
  Z_train = Z,
  y_train = Y,
  propensity_train = pihat,
  tau_c = tau_c,
  kappa_weight = 2.0,
  boundary_link = "logit",
  kappa_schedule = "adaptive",
  cost_ratio = 1.0, # symmetric costs
  num_gfr = 25,
  num_burnin = 100,
  num_mcmc = 250,
  general_params = list(verbose = FALSE)
)

probs <- fit$level_set_prob_smoothed
true_class <- tau >= tau_c

# Group patients by their true CATE (distance from boundary)
distance_bins <- cut(tau, breaks = c(-Inf, -1.5, -0.5, 0, 0.5, 1.5, Inf))

cat("\n--- Conservativism Analysis ---\n")
res <- data.frame(True_CATE = tau, Prob_Above_Boundary = probs, True_Class = true_class)
agg <- aggregate(res$Prob_Above_Boundary, by = list(Distance = distance_bins), FUN = function(x) c(mean = round(mean(x), 3), min = round(min(x), 3), max = round(max(x), 3)))
print(agg)

cat("\nMean Probability for points just BELOW boundary (True CATE in [-0.5, 0]): ", mean(probs[tau >= -0.5 & tau < 0]))
cat("\nMean Probability for points just ABOVE boundary (True CATE in [0, 0.5]): ", mean(probs[tau >= 0 & tau <= 0.5]))

cat("\n\nClassification Performance (Threshold 0.5):")
cat("\nTrue Negatives (Correctly Below): ", sum(probs <= 0.5 & !true_class))
cat("\nTrue Positives (Correctly Above): ", sum(probs > 0.5 & true_class))
cat("\nFalse Positives (Prob > 0.5 but True < 0): ", sum(probs > 0.5 & !true_class))
cat("\nFalse Negatives (Prob < 0.5 but True >= 0): ", sum(probs <= 0.5 & true_class))
cat("\n\nClassification Performance (Conservative Threshold 0.8 - Requires Strong Evidence to Treat):")
cat("\nTreated (Prob > 0.8): ", sum(probs > 0.8))
cat("\nOf those treated, truly positive: ", sum(probs > 0.8 & true_class))
cat("\nOf those treated, false positive: ", sum(probs > 0.8 & !true_class))
