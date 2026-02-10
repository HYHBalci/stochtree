# Helper function to create interaction pairs (1-based for R)
create_interaction_pairs_R <- function(p_main) {
  pairs <- list()
  if (p_main < 2) return(pairs)
  idx <- 1
  for (j in 1:(p_main - 1)) {
    for (k in (j + 1):p_main) {
      pairs[[idx]] <- c(j, k)
      idx <- idx + 1
    }
  }
  return(pairs)
}

# Helper function for inverse-gamma sampling in R
rinvgamma_R <- function(n, shape, scale) {
  if (shape <= 0 || scale <= 0) {
    stop("Shape and scale must be positive for rinvgamma.")
  }
  # If scale is scale_IG for InvGamma(shape, scale_IG)
  # then 1/X ~ Gamma(shape, rate_G = scale_IG)
  # rgamma in R takes shape and rate OR shape and scale_G (where scale_G = 1/rate_G)
  # We want rate_G = scale (from InvGamma definition)
  return(1 / rgamma(n, shape = shape, rate = scale))
}

# Helper for safe variance (to avoid 1/0 issues)
safe_var_R <- function(x) {
  return(max(x, 1e-9))
}

#' Linked Shrinkage Logistic Gibbs Sampler (Pure R Version)
#'
#' @description
#' Performs MCMC sampling for a Bayesian logistic regression model. This is a
#' pure R implementation. The model includes a prognostic part with linked
#' shrinkage priors for main effects (`beta`) and their interactions
#' (`beta_interaction`). It also includes a treatment-specific modifier part,
#' where the treatment indicator `Z_vec` multiplies an intercept term (`aleph`),
#' main covariate effects (`gamma_vec`), and two-way covariate interaction
#' effects (`gamma_int_vec`). The `gamma_vec` and `gamma_int_vec` coefficients
#' are given Horseshoe priors with a shared global shrinkage parameter
#' (`tau_hs_combined`). The `tau_int_param` for `beta_interaction` terms
#' is fixed at 1.0 in this version.
#'
#' This function relies on the `BayesLogit` package for Polya-Gamma sampling
#' and the `MASS` package for multivariate normal sampling. Ensure these are
#' installed and listed in your package's DESCRIPTION file if applicable.
#'
#' @param y_vec Integer vector. Binary response variable (must contain only 0s and 1s).
#' @param X_mat Numeric matrix. Covariate matrix for the prognostic part and for
#'    forming interactions with the treatment indicator `Z_vec`.
#'    Each row is an observation, each column is a covariate.
#' @param Z_vec Numeric vector. Treatment indicator (e.g., 0 or 1) or other key
#'    variable modifying the effect of covariates in `X_mat`. Must have the same
#'    length as `y_vec` and the number of rows in `X_mat`.
#' @param n_iter Integer. Total number of MCMC iterations.
#' @param burn_in Integer. Number of initial MCMC iterations to discard as burn-in.
#'    Must be less than `n_iter`.
#' @param alpha_prior_sd Numeric scalar. Standard deviation for the normal prior
#'    of the global intercept `alpha`. Default is 10.0.
#' @param aleph_prior_sd Numeric scalar. Standard deviation for the normal prior
#'    of the intercept `aleph` for the treatment-specific modifier part. Default is 10.0.
#' @param init_alpha Numeric scalar. Initial value for the global intercept `alpha`.
#'    Default is 0.0.
#' @param init_aleph Numeric scalar. Initial value for the intercept `aleph` of the
#'    treatment-specific modifier. Default is 0.0.
#' @param seed Integer or NULL. Seed for R's random number generator to ensure
#'    reproducibility. Default is 1848.
#'
#' @return A list containing MCMC samples (after burn-in) for the model parameters:
#' \describe{
#'   \item{alpha}{Numeric vector of samples for the global intercept `alpha`.}
#'   \item{beta}{Numeric matrix of samples for the main prognostic effects.}
#'   \item{beta_interaction}{Numeric matrix of samples for prognostic interaction effects.}
#'   \item{aleph}{Numeric vector of samples for the treatment-specific intercept.}
#'   \item{gamma}{Numeric matrix of samples for the treatment modifier coefficients.}
#'   \item{gamma_int}{Numeric matrix of samples for treatment interaction terms.}
#'   \item{tau_j}{Numeric matrix of samples for local shrinkage parameters.}
#'   \item{tau_int_fixed_value}{Numeric scalar, fixed value of 1.0.}
#'   \item{lambda_gamma}{Numeric matrix of samples for local Horseshoe parameters.}
#'   \item{lambda_g_int}{Numeric matrix of samples for interaction Horseshoe parameters.}
#'   \item{tau_hs_combined}{Numeric vector of samples for the global Horseshoe parameter.}
#' }
#'
#' @export
#' @importFrom MASS mvrnorm
#' @importFrom BayesLogit rpg

linked_shrinkage_logistic_gibbs_R <- function( # Body of the function as previously provided
  y_vec, X_mat, Z_vec,
  n_iter, burn_in,
  alpha_prior_sd = 10.0,
  aleph_prior_sd = 10.0,
  init_alpha = 0.0,
  init_aleph = 0.0,
  seed = 1848
) {
  if (!is.null(seed)) {
    set.seed(seed)
  }
  
  # Helper function to create interaction pairs (1-based for R)
  create_interaction_pairs_R <- function(p_main_local) {
    pairs_local <- list()
    if (p_main_local < 2) return(pairs_local)
    idx <- 1
    for (j_idx in 1:(p_main_local - 1)) {
      for (k_idx in (j_idx + 1):p_main_local) {
        pairs_local[[idx]] <- c(j_idx, k_idx)
        idx <- idx + 1
      }
    }
    return(pairs_local)
  }
  
  # Helper function for inverse-gamma sampling in R
  rinvgamma_R <- function(n, shape, scale) {
    if (shape <= 0 || scale <= 0) {
      # Simplified error handling for R version
      if(shape <= 0) return(rep(NaN, n))
      if(scale <= 0) return(rep(Inf, n)) # InvGamma with scale 0 means point mass at Inf or 0
    }
    return(1 / stats::rgamma(n, shape = shape, rate = scale))
  }
  
  safe_var_R <- function(x) {
    return(max(x, 1e-9))
  }
  
  N <- nrow(X_mat)
  p_main <- ncol(X_mat)
  
  if (length(y_vec) != N) stop("Length of y_vec must match number of rows in X_mat.")
  if (length(Z_vec) != N) stop("Length of Z_vec must match number of rows in X_mat.")
  
  main_interaction_indices <- create_interaction_pairs_R(p_main)
  p_interaction_main <- length(main_interaction_indices)
  
  alpha <- init_alpha
  aleph <- init_aleph
  beta_vec <- rep(0, p_main)
  beta_interaction_vec <- if (p_interaction_main > 0) rep(0, p_interaction_main) else numeric(0)
  gamma_vec <- rep(0, p_main)
  gamma_int_vec <- if (p_interaction_main > 0) rep(0, p_interaction_main) else numeric(0)
  
  tau_j_params_vec <- rep(1, p_main)
  zeta_tau_j_vec <- rep(1, p_main)
  tau_int_param <- 1.0 
  
  lambda_gamma_vec <- rep(1, p_main)
  nu_gamma_vec <- rep(1, p_main)
  lambda_g_int_vec <- if (p_interaction_main > 0) rep(1, p_interaction_main) else numeric(0)
  nu_g_int_vec <- if (p_interaction_main > 0) rep(1, p_interaction_main) else numeric(0)
  
  tau_hs_combined <- 1.0
  xi_hs_combined <- 1.0
  
  omega_vec <- rep(1, N)
  eta_vec <- rep(0, N) 
  kappa_vec <- y_vec - 0.5 
  
  num_samples_to_store <- n_iter - burn_in
  if (num_samples_to_store <= 0) {
    stop("n_iter must be greater than burn_in.")
  }
  
  alpha_samples <- numeric(num_samples_to_store)
  beta_samples <- matrix(0, nrow = num_samples_to_store, ncol = p_main)
  beta_interaction_samples <- matrix(0, nrow = num_samples_to_store, ncol = p_interaction_main)
  aleph_samples <- numeric(num_samples_to_store)
  gamma_samples <- matrix(0, nrow = num_samples_to_store, ncol = p_main)
  gamma_int_samples <- matrix(0, nrow = num_samples_to_store, ncol = p_interaction_main)
  tau_j_samples <- matrix(0, nrow = num_samples_to_store, ncol = p_main)
  lambda_gamma_samples <- matrix(0, nrow = num_samples_to_store, ncol = p_main)
  lambda_g_int_samples <- matrix(0, nrow = num_samples_to_store, ncol = p_interaction_main)
  tau_hs_combined_samples <- numeric(num_samples_to_store)
  
  current_sample_idx_R <- 0 # Use a different name to avoid conflict if debugging
  
  if (!requireNamespace("BayesLogit", quietly = TRUE)) {
    stop("Package 'BayesLogit' needed. Please install it.", call. = FALSE)
  }
  if (!requireNamespace("MASS", quietly = TRUE)) {
    stop("Package 'MASS' needed for mvrnorm. Please install it.", call. = FALSE)
  }
  
  
  for (iter in 1:n_iter) {
    if (iter %% 100 == 0) message(paste("R Version - Iteration:", iter, "/", n_iter))
    
    # 1. Update Linear Predictor eta_vec
    eta_vec <- as.vector(X_mat %*% beta_vec) + alpha 
    if (p_interaction_main > 0) {
      for (k in 1:p_interaction_main) {
        idx_pair <- main_interaction_indices[[k]] 
        eta_vec <- eta_vec + beta_interaction_vec[k] * (X_mat[, idx_pair[1]] * X_mat[, idx_pair[2]])
      }
    }
    
    treatment_modifier_part <- rep(aleph, N) + as.vector(X_mat %*% gamma_vec)
    if (p_interaction_main > 0) {
      for (k in 1:p_interaction_main) {
        idx_pair <- main_interaction_indices[[k]]
        treatment_modifier_part <- treatment_modifier_part + gamma_int_vec[k] * (X_mat[, idx_pair[1]] * X_mat[, idx_pair[2]])
      }
    }
    eta_vec <- eta_vec + Z_vec * treatment_modifier_part
    
    # 2. Sample Polya-Gamma latent variables omega_vec
    omega_vec <- BayesLogit::rpg(num = N, h = 1, z = abs(eta_vec))
    omega_vec[omega_vec < 1e-9] <- 1e-9 # Safeguard against zero omega
    
    # 3. Sample Regression Coefficients
    num_beta_int_terms = if (p_interaction_main > 0) p_interaction_main else 0
    num_gamma_int_terms = if (p_interaction_main > 0) p_interaction_main else 0
    
    total_coeffs <- 1 + p_main + num_beta_int_terms + 1 + p_main + num_gamma_int_terms
    
    X_full <- matrix(0, nrow = N, ncol = total_coeffs)
    col_counter <- 1
    
    X_full[, col_counter] <- 1; col_counter <- col_counter + 1 # alpha
    if (p_main > 0) {
      X_full[, col_counter:(col_counter + p_main - 1)] <- X_mat; col_counter <- col_counter + p_main # beta
    }
    if (p_interaction_main > 0) {
      for (k in 1:p_interaction_main) {
        idx_pair <- main_interaction_indices[[k]]
        X_full[, col_counter + k - 1] <- X_mat[, idx_pair[1]] * X_mat[, idx_pair[2]]
      }
      col_counter <- col_counter + p_interaction_main 
    }
    X_full[, col_counter] <- Z_vec; col_counter <- col_counter + 1 # aleph
    if (p_main > 0) {
      for (l in 1:p_main) {
        X_full[, col_counter + l - 1] <- Z_vec * X_mat[, l]
      }
      col_counter <- col_counter + p_main 
    }
    if (p_interaction_main > 0) {
      for (k in 1:p_interaction_main) {
        idx_pair <- main_interaction_indices[[k]]
        X_full[, col_counter + k - 1] <- Z_vec * (X_mat[, idx_pair[1]] * X_mat[, idx_pair[2]])
      }
    } 
    
    Y_star_vec <- kappa_vec / omega_vec
    Omega_diag_mat <- diag(omega_vec) 
    
    Xt_Omega_X <- t(X_full) %*% Omega_diag_mat %*% X_full
    Xt_Omega_Y <- t(X_full) %*% Omega_diag_mat %*% Y_star_vec
    
    prior_precision_diag_vec <- numeric(total_coeffs)
    col_counter <- 1
    prior_precision_diag_vec[col_counter] <- 1 / safe_var_R(alpha_prior_sd^2); col_counter <- col_counter + 1
    if (p_main > 0) {
      for (j in 1:p_main) {
        prior_precision_diag_vec[col_counter + j - 1] <- 1 / safe_var_R(tau_j_params_vec[j]^2)
      }
      col_counter <- col_counter + p_main
    }
    if (p_interaction_main > 0) {
      for (k in 1:p_interaction_main) {
        idx_pair <- main_interaction_indices[[k]]
        var_jk <- safe_var_R(tau_j_params_vec[idx_pair[1]] * tau_j_params_vec[idx_pair[2]] * tau_int_param)
        prior_precision_diag_vec[col_counter + k - 1] <- 1 / var_jk
      }
      col_counter <- col_counter + p_interaction_main
    }
    prior_precision_diag_vec[col_counter] <- 1 / safe_var_R(aleph_prior_sd^2); col_counter <- col_counter + 1
    if (p_main > 0) {
      for (l in 1:p_main) { 
        prior_precision_diag_vec[col_counter + l - 1] <- 1 / safe_var_R(lambda_gamma_vec[l]^2 * tau_hs_combined^2)
      }
      col_counter <- col_counter + p_main
    }
    if (p_interaction_main > 0) {
      for (k in 1:p_interaction_main) { 
        prior_precision_diag_vec[col_counter + k - 1] <- 1 / safe_var_R(lambda_g_int_vec[k]^2 * tau_hs_combined^2)
      }
    } 
    
    P_inv_diag_mat <- diag(prior_precision_diag_vec)
    posterior_precision_mat <- Xt_Omega_X + P_inv_diag_mat
    
    chol_attempt <- try(chol(posterior_precision_mat), silent = TRUE)
    if (inherits(chol_attempt, "try-error")) {
      jitter_val <- 1e-6 * mean(diag(posterior_precision_mat)) # Relative jitter
      posterior_precision_mat <- posterior_precision_mat + diag(jitter_val, total_coeffs)
      chol_attempt <- try(chol(posterior_precision_mat))
      if (inherits(chol_attempt, "try-error")) {
        stop(paste("Cholesky decomposition failed at iteration", iter, "even with jitter. Aborting."))
      }
    }
    posterior_covariance_mat <- chol2inv(chol_attempt)
    posterior_mean_vec <- posterior_covariance_mat %*% Xt_Omega_Y
    
    sampled_coeffs_vec <- MASS::mvrnorm(n = 1, mu = as.vector(posterior_mean_vec), Sigma = posterior_covariance_mat)
    
    col_counter <- 1
    alpha <- sampled_coeffs_vec[col_counter]; col_counter <- col_counter + 1
    if (p_main > 0) {
      beta_vec <- sampled_coeffs_vec[col_counter:(col_counter + p_main - 1)]; col_counter <- col_counter + p_main
    }
    if (p_interaction_main > 0) {
      beta_interaction_vec <- sampled_coeffs_vec[col_counter:(col_counter + p_interaction_main - 1)]
      col_counter <- col_counter + p_interaction_main
    }
    aleph <- sampled_coeffs_vec[col_counter]; col_counter <- col_counter + 1
    if (p_main > 0) {
      gamma_vec <- sampled_coeffs_vec[col_counter:(col_counter + p_main - 1)]; col_counter <- col_counter + p_main
    }
    if (p_interaction_main > 0) {
      gamma_int_vec <- sampled_coeffs_vec[col_counter:(col_counter + p_interaction_main - 1)]
    }
    
    # 4. Sample tau_j_params_vec
    if (p_main > 0) {
      # Define parameters for the C++ slice sampler call, specific to prognostic taus
      sigma_for_prognostic_taus <- 1.0 # Because variance for beta_j prior is tau_j^2
      tau_glob_for_prognostic_taus <- 1.0 # As there's no separate global term for prognostic betas here
      tau_int_for_prognostic_interactions <- tau_int_param # This is fixed at 1.0 in your R code
      unlink_status_for_prognostic <- FALSE # For linked shrinkage of prognostic interactions
      
      for (j_idx_r in 1:p_main) { # Iterate through each main effect tau_j (1-based R index)
        
        current_tau_old <- tau_j_params_vec[j_idx_r]
        current_beta_main_effect <- beta_vec[j_idx_r]
        new_tau_j <- sample_tau_j_slice( # Or sample_tau_j_slice_cpp if that's the R name
          tau_old = current_tau_old,
          beta_j = current_beta_main_effect,
          index = j_idx_r - 1, # C++ index is 0-based
          beta_int = beta_interaction_vec,    # Pass the full vector of prognostic interaction betas
          tau = tau_j_params_vec,             # Pass the full current vector of prognostic taus
          tau_int = tau_int_for_prognostic_interactions, # Scale for interaction terms
          sigma = sigma_for_prognostic_taus,    # Overall scale for priors (should be 1 for this model part)
          interaction = TRUE,                 # Assumed from C++ function default
          step_out = 0.5,
          max_steps = 50, 
          tau_glob = tau_glob_for_prognostic_taus, # Global shrinkage (1 for this model part)
          unlink = unlink_status_for_prognostic   # Linked structure for interactions
        )
        
        tau_j_params_vec[j_idx_r] <- new_tau_j
      }
    }
    
    # 6. Sample Local Horseshoe parameters for gamma_vec
    if (p_main > 0) {
      for (l in 1:p_main) {
        nu_gamma_vec[l] <- rinvgamma_R(1, shape = 1.0, scale = 1.0 + 1.0/safe_var_R(lambda_gamma_vec[l]^2))
        lambda_gamma_vec[l] <- sqrt(safe_var_R(rinvgamma_R(1, shape = 1.0, scale = 1.0/nu_gamma_vec[l] + (gamma_vec[l]^2) / safe_var_R(2.0*tau_hs_combined^2))))
      }
    }
    
    # 7. Sample Local Horseshoe parameters for gamma_int_vec
    if (p_interaction_main > 0) {
      for (k in 1:p_interaction_main) {
        nu_g_int_vec[k] <- rinvgamma_R(1, shape = 1.0, scale = 1.0 + 1.0/safe_var_R(lambda_g_int_vec[k]^2))
        lambda_g_int_vec[k] <- sqrt(safe_var_R(rinvgamma_R(1, shape = 1.0, scale = 1.0/nu_g_int_vec[k] + (gamma_int_vec[k]^2) / safe_var_R(2.0*tau_hs_combined^2))))
        
      }
    }
    
    # 8. Sample SHARED Global Horseshoe parameter tau_hs_combined
    sum_gamma_over_lambda_sq <- if (p_main > 0) sum( (gamma_vec^2) / safe_var_R(lambda_gamma_vec^2) ) else 0
    sum_g_int_over_lambda_sq <- if (p_interaction_main > 0) sum( (gamma_int_vec^2) / safe_var_R(lambda_g_int_vec^2) ) else 0
    
    total_coeffs_for_hs <- p_main + (if (p_interaction_main > 0) p_interaction_main else 0)
    
    if (total_coeffs_for_hs > 0) { 
      tau_hs_combined <- sqrt(safe_var_R(rinvgamma_R(1, shape = (total_coeffs_for_hs + 1.0)/2.0, 
                                                     scale = 1.0/xi_hs_combined + (sum_gamma_over_lambda_sq + sum_g_int_over_lambda_sq)/2.0)))
      xi_hs_combined <- rinvgamma_R(1, shape = 1.0, scale = 1.0 + 1.0/safe_var_R(tau_hs_combined^2))
    } else { 
      tau_hs_combined <- 1.0 
      xi_hs_combined <- 1.0
    }
    
    # Store Samples
    if (iter > burn_in) {
      current_sample_idx_R <- iter - burn_in # Corrected index for R storage
      alpha_samples[current_sample_idx_R] <- alpha
      aleph_samples[current_sample_idx_R] <- aleph
      tau_hs_combined_samples[current_sample_idx_R] <- tau_hs_combined
      
      if (p_main > 0) {
        beta_samples[current_sample_idx_R, ] <- beta_vec
        gamma_samples[current_sample_idx_R, ] <- gamma_vec
        tau_j_samples[current_sample_idx_R, ] <- tau_j_params_vec
        lambda_gamma_samples[current_sample_idx_R, ] <- lambda_gamma_vec
      }
      if (p_interaction_main > 0) {
        beta_interaction_samples[current_sample_idx_R, ] <- beta_interaction_vec
        gamma_int_samples[current_sample_idx_R, ] <- gamma_int_vec
        lambda_g_int_samples[current_sample_idx_R, ] <- lambda_g_int_vec
      }
    }
  }
  
  return(list(
    alpha = alpha_samples,
    beta = beta_samples,
    beta_interaction = beta_interaction_samples,
    aleph = aleph_samples,
    gamma = gamma_samples,
    gamma_int = gamma_int_samples,
    tau_j = tau_j_samples,
    lambda_gamma = lambda_gamma_samples,
    lambda_g_int = lambda_g_int_samples,
    tau_hs_combined = tau_hs_combined_samples
  ))
}