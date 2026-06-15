#' Augmented Marginal Ratio (AMR) ATE Estimation (Fully Bayesian)
#'
#' @description
#' This function computes a doubly robust ATE estimate using Augmented Marginal
#' Ratio (AMR) weighting. It propagates uncertainty across the entire posterior
#' by estimating the ATE for every MCMC sample of the BCF outcome model, 
#' sampling propensity scores (if not provided) using a Laplace approximation,
#' and sampling B-spline density ratio weights from a conjugate normal posterior.
#'
#' @param bcf_fit A fitted BCF model object containing posterior samples of `tau` and `mu`.
#' @param Y Numeric vector of observed outcomes.
#' @param A Numeric vector of binary treatments.
#' @param X Dataframe or matrix of covariates (used for propensity modeling if missing).
#' @param propensity_scores Optional vector of fixed propensity scores. If NULL, an in-house Bayesian logistic regression is used.
#' @param df_weights Degrees of freedom for the B-spline weighting function (default 5).
#' @param prior_var_theta Prior variance for the B-spline coefficients (default 100).
#'
#' @return A list containing posterior samples of the ATE, plugin ATE, augmentation term, and posterior mean weights.
#' @export
bcf_amr_ate <- function(bcf_fit, Y, A, X = NULL, propensity_scores = NULL, df_weights = 5, prior_var_theta = 100) {
  
  library(splines)
  library(MASS) # for mvrnorm
  
  # Number of MCMC samples in the BCF fit
  # Assuming bcf_fit$tau is a matrix with dimensions (n_samples, n_obs)
  n_samples <- nrow(bcf_fit$tau)
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
  
  # Storage for posterior samples
  ate_samples <- numeric(n_samples)
  plugin_samples <- numeric(n_samples)
  aug_samples <- numeric(n_samples)
  weights_mean <- rep(0, n_obs)
  
  for (j in 1:n_samples) {
    # 1. Propensity Score
    if (is.null(propensity_scores)) {
      pi_j <- plogis(as.vector(X_matrix %*% beta_samples[j, ]))
    } else {
      pi_j <- propensity_scores
    }
    pi_j <- pmax(pmin(pi_j, 0.999), 0.001)
    
    # 2. Clever Covariate
    h_j <- (A - pi_j) / (pi_j * (1 - pi_j))
    
    # 3. Outcome Draws
    mu_j <- bcf_fit$mu[j, ]
    tau_j <- bcf_fit$tau[j, ]
    
    mu_0_j <- mu_j
    mu_1_j <- mu_j + tau_j
    mu_A_j <- mu_j + A * tau_j
    
    # 4. AMR Transformation
    Y_star_j <- Y - (pi_j * mu_0_j + (1 - pi_j) * mu_1_j)
    
    # 5. Basis Expansion
    B_j <- bs(Y_star_j, df = df_weights)
    
    # 6. Conjugate Bayesian Linear Regression for Spline Weights
    # Model: h_j ~ N(B_j theta, sigma2_h I)
    sigma2_h <- max(var(h_j), 1e-6)
    
    # Prior: theta ~ N(0, prior_var_theta * I)
    prior_prec <- diag(df_weights) / prior_var_theta
    prec_matrix <- crossprod(B_j) / sigma2_h + prior_prec
    
    # Posterior parameters
    mean_theta <- solve(prec_matrix, crossprod(B_j, h_j) / sigma2_h)
    L_chol <- chol(prec_matrix)
    
    # Draw theta from posterior
    theta_j <- mean_theta + backsolve(L_chol, rnorm(df_weights))
    
    # 7. Compute Weights
    w_j <- as.vector(B_j %*% theta_j)
    w_j <- pmax(w_j, 1e-4) # Enforce non-negativity
    weights_mean <- weights_mean + w_j / n_samples
    
    # 8. Compute ATE Draw
    plugin_j <- mean(mu_1_j - mu_0_j)
    aug_j <- mean(w_j * (Y - mu_A_j))
    
    plugin_samples[j] <- plugin_j
    aug_samples[j] <- aug_j
    ate_samples[j] <- plugin_j + aug_j
  }
  
  return(list(
    ate_posterior = ate_samples,
    plugin_posterior = plugin_samples,
    augmentation_posterior = aug_samples,
    weights_mean = weights_mean
  ))
}
