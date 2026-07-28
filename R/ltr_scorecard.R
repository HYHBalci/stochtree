#' Two-Stage Semi-Parametric Learning-to-Rank for Causal Inference
#'
#' @description
#' Stage 2 of the Semi-Parametric LTR framework. Takes the continuous treatment proxy
#' from Stage 1 and uses a Horseshoe-regularized linear model to sort patients
#' based on a pairwise ranking objective.
#'
#' @param tau_tilde Numeric vector of continuous treatment proxies from Stage 1.
#' @param X Numeric matrix of covariates.
#' @param method String. Either "MSE" (Pairwise Mean Squared Error) or "PG" (Polya-Gamma weighted Logistic).
#' @param mini_batch_size Integer. Number of pairs to sample per MCMC iteration.
#' @param epsilon Numeric. Minimum difference between tau_tilde for a pair to be considered valid.
#' @param n_iter Integer. Number of retained MCMC iterations.
#' @param burn_in Integer. Number of burn-in MCMC iterations.
#' @param seed Integer. Random seed.
#'
#' @return A list containing MCMC samples for the triage scorecard coefficients (beta).
#' @export
ltr_scorecard <- function(tau_tilde, X, method = c("MSE", "PG"), 
                          interaction_rule = c("none", "continuous", "continuous_or_binary", "all"), 
                          unlink = FALSE,
                          mini_batch_size = 2000, epsilon = 0.01,
                          n_iter = 1000, burn_in = 500, seed = 123) {
  method <- match.arg(method)
  interaction_rule <- match.arg(interaction_rule)
  if (!is.matrix(X)) X <- as.matrix(X)
  if (length(tau_tilde) != nrow(X)) stop("Length of tau_tilde must equal the number of rows in X")
  
  set.seed(seed)
  
  # Determine which variables can interact
  are_continuous <- rep(0L, ncol(X))
  for (j in 1:ncol(X)) {
    unique_vals <- length(unique(X[,j]))
    if (interaction_rule == "continuous") {
      if (unique_vals > 2) are_continuous[j] <- 1L
    } else if (interaction_rule == "continuous_or_binary") {
      if (unique_vals >= 2) are_continuous[j] <- 1L
    } else if (interaction_rule == "all") {
      are_continuous[j] <- 1L
    }
  }
  if (interaction_rule == "none") {
    are_continuous <- rep(0L, ncol(X))
  }
  are_continuous <- as.integer(are_continuous)
  
  # Calculate number of interaction terms
  P_main <- ncol(X)
  int_pairs_count <- 0
  for (i in 1:P_main) {
    for (j in seq_len(P_main)) {
      if (j > i) {
        if (are_continuous[i] == 1 || are_continuous[j] == 1) {
          int_pairs_count <- int_pairs_count + 1
        }
      }
    }
  }
  
  # Initialization
  beta_init <- rep(0, P_main)
  beta_int_init <- if (int_pairs_count > 0) rep(0, int_pairs_count) else numeric(0)
  P_combined <- P_main + int_pairs_count
  
  tau_beta_init <- rep(1, P_combined)
  nu_init <- rep(1, P_combined)
  tau_int_init <- 1.0
  tau_glob_init <- 1.0
  xi_init <- 1.0
  sigma2_rank_init <- 1.0
  
  # Compute analytical pre-test for heterogeneity (main effects)
  valid_cols <- apply(X, 2, sd) > 1e-6
  if (any(valid_cols)) {
    X_cen <- scale(X[, valid_cols, drop = FALSE], center = TRUE, scale = TRUE)
    n_obs <- nrow(X_cen)
    p_valid <- ncol(X_cen)
    
    if (method == "MSE") {
      s_vec <- tau_tilde - mean(tau_tilde)
    } else {
      # PG method uses ranks (equivalent to multivariate Kendall's Tau / Spearman's Rho)
      s_vec <- 2 * rank(tau_tilde) - n_obs - 1
    }
    
    U_stat <- as.numeric(t(X_cen) %*% s_vec)
    var_s <- sum(s_vec^2) / (n_obs - 1)
    Var_U <- var_s * crossprod(X_cen)
    
    # Tiny ridge for numerical stability
    Var_U_ridge <- Var_U + diag(1e-8, p_valid)
    
    Var_U_inv <- tryCatch({
      solve(Var_U_ridge)
    }, error = function(e) {
      if (requireNamespace("MASS", quietly = TRUE)) {
        MASS::ginv(Var_U)
      } else {
        diag(1/diag(Var_U_ridge))
      }
    })
    
    T_quad <- as.numeric(t(U_stat) %*% Var_U_inv %*% U_stat)
    pre_test_pval <- pchisq(T_quad, df = p_valid, lower.tail = FALSE)
  } else {
    pre_test_pval <- 1.0
  }
  
  if (method == "MSE") {
    out <- run_ltr_mse_cpp(
      tau_tilde = tau_tilde,
      X = X,
      are_continuous = are_continuous,
      M = mini_batch_size,
      epsilon = epsilon,
      beta_init = beta_init,
      beta_int_init = beta_int_init,
      tau_beta_init = tau_beta_init,
      nu_init = nu_init,
      tau_int_init = tau_int_init,
      tau_glob_init = tau_glob_init,
      xi_init = xi_init,
      sigma2_rank_init = sigma2_rank_init,
      unlink = unlink,
      n_iter = n_iter,
      burn_in = burn_in
    )
  } else if (method == "PG") {
    out <- run_ltr_pg_cpp(
      tau_tilde = tau_tilde,
      X = X,
      are_continuous = are_continuous,
      M = mini_batch_size,
      epsilon = epsilon,
      beta_init = beta_init,
      beta_int_init = beta_int_init,
      tau_beta_init = tau_beta_init,
      nu_init = nu_init,
      tau_int_init = tau_int_init,
      tau_glob_init = tau_glob_init,
      xi_init = xi_init,
      unlink = unlink,
      n_iter = n_iter,
      burn_in = burn_in
    )
  }
  
  out$pre_test_main_effects_pval <- pre_test_pval
  
  return(out)
}
