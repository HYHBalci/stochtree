#' Marginal Ratio (MR) Policy Evaluation (Fully Bayesian)
#'
#' @description
#' Evaluates the expected population outcome of a deterministic treatment policy
#' using the Marginal Ratio (MR) Off-Policy Evaluation estimator. Uncertainty
#' is propagated via posterior sampling of the propensity scores (if not provided)
#' and the density ratio weights.
#'
#' @param policy_assignments A numeric vector of binary treatment assignments recommended by the policy.
#' @param Y Numeric vector of observed outcomes.
#' @param A Numeric vector of observed binary treatments.
#' @param X Dataframe or matrix of covariates (used for propensity modeling if missing).
#' @param propensity_scores Optional vector of fixed propensity scores. If NULL, an in-house Bayesian logistic regression is used.
#' @param n_samples Number of posterior samples to draw (default 1000).
#' @param df_weights Degrees of freedom for the B-spline weighting function (default 5).
#' @param prior_var_theta Prior variance for the B-spline coefficients (default 100).
#'
#' @return A list containing posterior samples of the OPE estimate and posterior mean weights.
#' @export
evaluate_policy_mr <- function(policy_assignments, Y, A, X = NULL, propensity_scores = NULL, n_samples = 1000, df_weights = 5, prior_var_theta = 100) {
  
  library(splines)
  library(MASS) # for mvrnorm
  
  n_obs <- length(Y)
  
  if (is.null(X) && is.null(propensity_scores)) {
    stop("If propensity_scores are not provided, X must be provided to estimate them.")
  }
  
  # Propensity Score Estimation via Laplace Approximation
  if (is.null(propensity_scores)) {
    message("Propensity scores not provided. Estimating via in-house Bayesian logistic regression (Laplace Approximation)...")
    glm_fit <- glm(A ~ ., data = as.data.frame(X), family = binomial)
    beta_hat <- coef(glm_fit)
    Sigma_hat <- vcov(glm_fit)
    
    # Draw propensity score parameters for each MCMC sample
    beta_samples <- mvrnorm(n_samples, mu = beta_hat, Sigma = Sigma_hat)
    X_matrix <- model.matrix(glm_fit)
  }
  
  # Basis Expansion for Y (constant across iterations)
  B_spline <- bs(Y, df = df_weights)
  prior_prec <- diag(df_weights) / prior_var_theta
  
  # Storage for posterior samples
  ope_samples <- numeric(n_samples)
  weights_mean <- rep(0, n_obs)
  
  for (j in 1:n_samples) {
    # 1. Propensity Score
    if (is.null(propensity_scores)) {
      pi_j <- plogis(as.vector(X_matrix %*% beta_samples[j, ]))
    } else {
      pi_j <- propensity_scores
    }
    pi_j <- pmax(pmin(pi_j, 0.999), 0.001)
    
    # 2. Policy Ratio
    # rho(A, X) = 1(A == policy(X)) / P(A|X)
    rho_j <- ifelse(A == policy_assignments, 1 / (A * pi_j + (1 - A) * (1 - pi_j)), 0)
    
    # 3. Conjugate Bayesian Linear Regression for Spline Weights
    # Model: rho_j ~ N(B theta, sigma2_rho I)
    sigma2_rho <- max(var(rho_j), 1e-6)
    
    prec_matrix <- crossprod(B_spline) / sigma2_rho + prior_prec
    
    # Posterior parameters
    mean_theta <- solve(prec_matrix, crossprod(B_spline, rho_j) / sigma2_rho)
    L_chol <- chol(prec_matrix)
    
    # Draw theta from posterior
    theta_j <- mean_theta + backsolve(L_chol, rnorm(df_weights))
    
    # 4. Compute Weights
    w_j <- as.vector(B_spline %*% theta_j)
    w_j <- pmax(w_j, 1e-4) # Enforce non-negativity
    weights_mean <- weights_mean + w_j / n_samples
    
    # 5. Compute OPE Draw
    ope_samples[j] <- mean(w_j * Y)
  }
  
  return(list(
    ope_posterior = ope_samples,
    weights_mean = weights_mean
  ))
}
