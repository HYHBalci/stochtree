#' Fit a Restricted BCF Model (Constant Treatment Effect) and Compute Score Test P-Values
#'
#' @param X_train Covariates used to split trees in the ensemble.
#' @param Z_train Vector of treatment assignments.
#' @param y_train Outcome to be modeled by the ensemble.
#' @param propensity_train (Optional) Vector of propensity scores.
#' @param rfx_group_ids_train (Optional) Group labels used for an additive random effects model.
#' @param rfx_basis_train (Optional) Basis for "random-slope" regression.
#' @param num_gfr Number of "warm-start" iterations. Default: 5.
#' @param num_burnin Number of "burn-in" iterations. Default: 100.
#' @param num_mcmc Number of "retained" iterations. Default: 500.
#' @param use_rao_blackwell Logical. If TRUE, averages the score vector and covariance matrix across MCMC draws to compute a single optimal P-value. Default: TRUE.
#' @param test_type Character string specifying the score test type: \code{"mean"} for testing the conditional mean, or \code{"distributional"} for testing the full conditional distribution via ECDF. Default: \code{"mean"}.
#' @param num_grid_points Integer specifying the number of evaluation points for the distributional ECDF test. Ignored if \code{test_type = "mean"}. Default: 50.
#' @param general_params (Optional) A list of general model parameters.
#' @param prognostic_forest_params (Optional) A list of prognostic forest model parameters.
#' @param variance_forest_params (Optional) A list of variance forest model parameters.
#' @return A list of class `bcfscoretest` containing posterior samples and score test P-values.
#' @export
bcf_restricted_score_test <- function(X_train, Z_train, y_train, propensity_train = NULL, 
                                      rfx_group_ids_train = NULL, rfx_basis_train = NULL, 
                                      num_gfr = 5, num_burnin = 100, num_mcmc = 500, 
                                      use_rao_blackwell = TRUE,
                                      test_type = c("mean", "distributional"),
                                      num_grid_points = 50,
                                      general_params = list(), prognostic_forest_params = list(), 
                                      variance_forest_params = list()) {
  
  test_type <- match.arg(test_type)
  
  if (!requireNamespace("mvtnorm", quietly = TRUE)) stop("Package 'mvtnorm' is required.")
  if (!requireNamespace("MASS", quietly = TRUE)) stop("Package 'MASS' is required.")
  
  general_params_default <- list(
    cutpoint_grid_size = 100, standardize = TRUE, sample_sigma2_global = TRUE, sigma2_global_init = NULL, 
    sigma2_global_shape = 1, sigma2_global_scale = 0.001, variable_weights = NULL, 
    propensity_covariate = "mu", adaptive_coding = FALSE, rfx_prior_var = NULL, random_seed = -1, 
    keep_burnin = FALSE, keep_gfr = FALSE, keep_every = 1, num_chains = 1, verbose = TRUE, 
    probit_outcome_model = FALSE, standardize_cov = FALSE, interaction_rule = "continuous",
    alpha_prior_var = 100.0 # Added explicit prior variance for the main effect
  )
  general_params_updated <- preprocessParams(general_params_default, general_params)
  
  if (general_params_updated$probit_outcome_model) {
    stop("The score test is currently only mathematically derived and implemented for continuous outcomes. Please ensure probit_outcome_model = FALSE.")
  }
  
  rinvgamma <- function(shape, scale) {
    if (shape <= 0.0 || scale <= 0.0) stop("Shape and scale must be positive.");
    return(1.0 / rgamma(1, shape, scale))
  }
  
  if(general_params_updated$verbose) print("Pre-Processing data for Score Test!")
  
  handled_data_list <- standardize_X_by_index(X_train, process_data = general_params_updated$standardize_cov, 
                                              interaction_rule = general_params_updated$interaction_rule, 
                                              cat_coding_method = "difference")
  X_train_standardized <- handled_data_list$X_final
  X_train_metadata_init <- handled_data_list$X_final_var_info
  
  train_cov_preprocess_list <- preprocessTrainData(X_train_standardized)
  X_train_metadata <- train_cov_preprocess_list$metadata
  X_train_raw <- X_train_standardized
  X_train_forest <- train_cov_preprocess_list$data
  original_var_indices <- X_train_metadata$original_var_indices
  feature_types <- as.integer(X_train_metadata$feature_types)
  
  n <- nrow(X_train_forest)
  num_cov_orig <- ncol(X_train_forest)
  
  internal_propensity_model <- FALSE
  if (is.null(propensity_train) && general_params_updated$propensity_covariate != "none") {
    internal_propensity_model <- TRUE
    if(general_params_updated$verbose) print("Estimating internal propensity scores...")
    bart_model_propensity <- bart(X_train = X_train_raw, y_train = as.numeric(Z_train), 
                                  num_gfr = num_gfr, num_burnin = 0, num_mcmc = 0)
    
    # Robust indexing for internal propensity scores: take the last 80% of GFR draws
    start_idx <- floor(num_gfr * 0.2) + 1
    if (start_idx > num_gfr) start_idx <- max(1, num_gfr - 5)
    
    propensity_train <- as.matrix(rowMeans(bart_model_propensity$y_hat_train[, start_idx:num_gfr, drop=FALSE]))
  }
  if(is.null(propensity_train)) propensity_train <- rep(mean(Z_train), n)
  propensity_train <- as.matrix(propensity_train)
  
  # Center treatment indicator
  Z_cen <- Z_train - as.vector(propensity_train)
  
  # =====================================================================
  # SCORE TEST DESIGN MATRIX
  # =====================================================================
  design_info <- prepare_score_test_design(X_train_raw, X_train_metadata_init, general_params_updated$interaction_rule)
  X_cen <- design_info$X_cen
  p_valid <- design_info$p_valid
  
  num_chains <- general_params_updated$num_chains
  alpha_samples <- matrix(0, nrow = num_chains, ncol = num_mcmc)
  sigma2_samples <- matrix(0, nrow = num_chains, ncol = num_mcmc)
  
  if (use_rao_blackwell) {
    if (test_type == "mean") {
      s_i_sum <- matrix(0, nrow = num_chains, ncol = n)
    } else {
      s_mat_sum <- replicate(num_chains, matrix(0, nrow = n, ncol = num_grid_points), simplify = FALSE)
    }
    pval_max_samples <- NULL
    pval_quad_samples <- NULL
  } else {
    if (test_type == "distributional") {
      stop("test_type = 'distributional' requires use_rao_blackwell = TRUE")
    }
    pval_max_samples <- matrix(0, nrow = num_chains, ncol = num_mcmc)
    pval_quad_samples <- matrix(0, nrow = num_chains, ncol = num_mcmc)
  }
  
  alpha_prior_var <- general_params_updated$alpha_prior_var
  
  prognostic_forest_params_default <- list(
    num_trees = 250, alpha = 0.95, beta = 2.0, min_samples_leaf = 5, max_depth = 10, 
    sample_sigma2_leaf = TRUE, sigma2_leaf_init = NULL, sigma2_leaf_shape = 3, 
    sigma2_leaf_scale = NULL, keep_vars = NULL, drop_vars = NULL
  )
  prognostic_forest_params_updated <- preprocessParams(prognostic_forest_params_default, prognostic_forest_params)
  
  # Variance forest params retained per user request, though not sampled in restricted model
  variance_forest_params_default <- list(
    num_trees = 0, alpha = 0.95, beta = 2.0, min_samples_leaf = 5, max_depth = 10, 
    leaf_prior_calibration_param = 1.5, variance_forest_init = NULL, var_forest_prior_shape = NULL, 
    var_forest_prior_scale = NULL, keep_vars = NULL, drop_vars = NULL
  )
  variance_forest_params_updated <- preprocessParams(variance_forest_params_default, variance_forest_params)
  
  cutpoint_grid_size <- general_params_updated$cutpoint_grid_size
  standardize <- general_params_updated$standardize
  sample_sigma2_global <- general_params_updated$sample_sigma2_global
  random_seed <- general_params_updated$random_seed
  if (is.null(random_seed) || random_seed == -1) random_seed = sample(1:1000000, 1, FALSE)
  set.seed(random_seed)
  rng <- createCppRNG(random_seed)
  verbose <- general_params_updated$verbose
  probit_outcome_model <- general_params_updated$probit_outcome_model
  
  num_trees_mu <- prognostic_forest_params_updated$num_trees
  sample_sigma2_leaf_mu <- prognostic_forest_params_updated$sample_sigma2_leaf
  a_leaf_mu <- prognostic_forest_params_updated$sigma2_leaf_shape
  b_leaf_mu <- prognostic_forest_params_updated$sigma2_leaf_scale
  
  num_trees_variance <- variance_forest_params_updated$num_trees
  include_variance_forest <- num_trees_variance > 0
  variance_forest_init <- variance_forest_params_updated$variance_forest_init
  
  variable_weights <- rep(1/ncol(X_train_raw), ncol(X_train_raw))
  variable_weights_adj <- 1/sapply(original_var_indices, function(x) sum(original_var_indices == x))
  variable_weights <- variable_weights[original_var_indices]*variable_weights_adj
  
  variable_weights_variance <- variable_weights_mu <- variable_weights
  
  if (general_params_updated$propensity_covariate %in% c("mu", "both")) {
    feature_types <- as.integer(c(feature_types, rep(0, ncol(propensity_train))))
    X_train_forest <- cbind(X_train_forest, propensity_train)
    variable_weights_mu <- c(variable_weights_mu, rep(1/num_cov_orig, ncol(propensity_train)))
    if(include_variance_forest) variable_weights_variance <- c(variable_weights_variance, rep(0, ncol(propensity_train)))
  }
  variable_weights_mu <- variable_weights_mu / sum(variable_weights_mu)
  
  if (standardize) {
    y_bar_train <- mean(y_train)
    y_std_train <- sd(y_train)
  } else {
    y_bar_train <- 0
    y_std_train <- 1
  }
  resid_train_base <- (y_train - y_bar_train) / y_std_train
  init_mu <- mean(resid_train_base)
  
  # Fix scale mismatch for user-provided initializations
  user_sigma2_init <- general_params_updated$sigma2_global_init
  if (!is.null(user_sigma2_init)) {
    if (standardize) user_sigma2_init <- user_sigma2_init / (y_std_train^2)
  }
  current_sigma2_init <- user_sigma2_init %||% (1.0 * var(resid_train_base))
  
  if (is.null(b_leaf_mu)) b_leaf_mu <- var(resid_train_base) / (num_trees_mu)
  
  user_sigma2_leaf <- prognostic_forest_params_updated$sigma2_leaf_init
  if (!is.null(user_sigma2_leaf)) {
    if (standardize) user_sigma2_leaf <- user_sigma2_leaf / (y_std_train^2)
  }
  sigma2_leaf_mu <- user_sigma2_leaf %||% (2.0 * var(resid_train_base) / num_trees_mu)
  current_leaf_scale_mu <- as.matrix(sigma2_leaf_mu)
  
  num_samples <- num_gfr + num_burnin + num_mcmc
  
  if (test_type == "distributional") {
    # Grid will be set adaptively on the first MCMC iteration,
    # after BART has warmed up and the residuals reflect the fitted model
    t_grid <- NULL
    grid_initialized <- FALSE
  }
  
  # =====================================================================
  # MAIN SAMPLER LOOP
  # =====================================================================
  for (chain_num in 1:num_chains) {
    
    alpha <- 0.0
    current_sigma2 <- current_sigma2_init
    resid_train <- resid_train_base
    
    forest_dataset_train <- createForestDataset(X_train_forest, Z_train)
    outcome_train <- createOutcome(resid_train)
    
    global_model_config <- createGlobalModelConfig(global_error_variance = current_sigma2)
    forest_model_config_mu <- createForestModelConfig(
      feature_types = feature_types, num_trees = num_trees_mu, num_features = ncol(X_train_forest), 
      num_observations = n, variable_weights = variable_weights_mu, leaf_dimension = 1, 
      alpha = prognostic_forest_params_updated$alpha, beta = prognostic_forest_params_updated$beta, 
      min_samples_leaf = prognostic_forest_params_updated$min_samples_leaf, max_depth = prognostic_forest_params_updated$max_depth, 
      leaf_model_type = 0, leaf_model_scale = current_leaf_scale_mu, cutpoint_grid_size = cutpoint_grid_size
    )
    
    forest_model_mu <- createForestModel(forest_dataset_train, forest_model_config_mu, global_model_config)
    forest_samples_mu <- createForestSamples(num_trees_mu, 1, TRUE)
    active_forest_mu <- createForest(num_trees_mu, 1, TRUE)
    active_forest_mu$prepare_for_sampler(forest_dataset_train, outcome_train, forest_model_mu, 0, init_mu)
    active_forest_mu$adjust_residual(forest_dataset_train, outcome_train, forest_model_mu, FALSE, FALSE)
    
    for (i in 1:num_samples) {
      is_mcmc <- i > (num_gfr + num_burnin)
      is_gfr <- i <= num_gfr
      
      if (verbose && (i %% 100 == 0 || i == 1)) {
        if (is_gfr) cat("Chain", chain_num, "- Sampling", i, "of", num_gfr, "GFR draws\n")
        else if (!is_mcmc) cat("Chain", chain_num, "- Sampling", i - num_gfr, "of", num_burnin, "Burn-in draws\n")
        else cat("Chain", chain_num, "- Sampling", i - num_gfr - num_burnin, "of", num_mcmc, "MCMC draws\n")
      }
      
      forest_model_mu$sample_one_iteration(
        forest_dataset = forest_dataset_train, residual = outcome_train, 
        forest_samples = forest_samples_mu, active_forest = active_forest_mu, rng = rng, 
        forest_model_config = forest_model_config_mu, global_model_config = global_model_config, 
        keep_forest = is_mcmc, gfr = is_gfr
      )
      
      if (sample_sigma2_leaf_mu) {
        leaf_scale_mu_double <- sampleLeafVarianceOneIteration(active_forest_mu, rng, a_leaf_mu, b_leaf_mu)
        forest_model_config_mu$update_leaf_model_scale(as.matrix(leaf_scale_mu_double))
      }
      
      mu_x_raw_train <- active_forest_mu$predict_raw(forest_dataset_train)
      r_alpha <- resid_train - mu_x_raw_train
      
      var_alpha <- 1.0 / (sum(Z_cen^2) / current_sigma2 + 1.0 / alpha_prior_var)
      mean_alpha <- var_alpha * sum(r_alpha * Z_cen) / current_sigma2
      alpha <- rnorm(1, mean = mean_alpha, sd = sqrt(var_alpha))
      
      resid_full <- r_alpha - alpha * Z_cen
      outcome_train$update_data(resid_full)
      
      if (sample_sigma2_global) {
        shape_sigma_post <- 0.001 + n / 2
        rate_sigma_post <- 0.001 + 0.5 * sum(outcome_train$get_data()^2)
        current_sigma2 <- max(rinvgamma(shape_sigma_post, rate_sigma_post), 1e-9)
        global_model_config$update_global_error_variance(current_sigma2)
      }
      
      # =====================================================================
      # SCORE TEST INJECTION
      # =====================================================================
      if (is_mcmc) {
        mcmc_counter <- i - num_gfr - num_burnin
        
        alpha_samples[chain_num, mcmc_counter] <- alpha
        sigma2_samples[chain_num, mcmc_counter] <- current_sigma2
        
        if (test_type == "mean") {
          # 1. Calculate Score Residuals
          s_i <- resid_full * Z_cen
          s_i_cen <- s_i - mean(s_i)
          
          if (use_rao_blackwell) {
            # Accumulate score residuals to compute variance at the posterior mean
            s_i_sum[chain_num, ] <- s_i_sum[chain_num, ] + as.numeric(s_i_cen)
          } else {
            # Legacy method: P-values at every draw using safe evaluation
            T_vec <- t(X_cen) %*% s_i_cen
            X_s <- X_cen * as.vector(s_i_cen)
            Var_T <- crossprod(X_s)
            draw_pvals <- compute_pmvnorm_safe(T_vec, Var_T, p_valid)
            pval_quad_samples[chain_num, mcmc_counter] <- draw_pvals$quad
            pval_max_samples[chain_num, mcmc_counter] <- draw_pvals$max
          }
        } else if (test_type == "distributional") {
          # On the first MCMC iteration, set the grid from post-warmup residuals
          if (!grid_initialized) {
            t_grid <- as.numeric(quantile(resid_full, probs = seq(0.01, 0.99, length.out = num_grid_points)))
            grid_initialized <- TRUE
          }
          
          # Calculate Empirical CDF Indicators and Score Matrix
          # I_mat: Indicator matrix for residuals falling below each grid point t
          # F_k_draw: Marginal ECDF at each grid point
          # e_mat: Centered indicator (I - F), which has mean 0 under the null
          # s_mat: Score contribution for treatment heterogeneity: Z_cen * (I - F)
          I_mat <- outer(as.numeric(resid_full), t_grid, "<=") * 1.0
          F_k_draw <- colMeans(I_mat)
          e_mat <- sweep(I_mat, 2, F_k_draw, "-")
          s_mat <- e_mat * as.numeric(Z_cen)
          
          if (use_rao_blackwell) {
            s_mat_sum[[chain_num]] <- s_mat_sum[[chain_num]] + s_mat
          }
        }
      }
    }
  }
  
  # =====================================================================
  # POST-PROCESSING FOR RAO-BLACKWELLIZATION
  # =====================================================================
  if (use_rao_blackwell) {
    if (test_type == "mean") {
      rb_pval_quad <- numeric(num_chains)
      rb_pval_max <- numeric(num_chains)
      
      for (c in 1:num_chains) {
        s_i_mean <- s_i_sum[c, ] / num_mcmc
        
        T_vec_mean <- t(X_cen) %*% s_i_mean
        
        # Compute the Huber-White empirical sandwich variance estimator of the score.
        # This provides robustness against heteroskedasticity in the residuals.
        X_s_mean <- X_cen * as.vector(s_i_mean)
        Var_T_mean <- crossprod(X_s_mean)
        
        rb_pvals <- compute_pmvnorm_safe(T_vec_mean, Var_T_mean, p_valid)
        rb_pval_quad[c] <- rb_pvals$quad
        rb_pval_max[c] <- rb_pvals$max
      }
    } else {
      rb_pval_wass <- numeric(num_chains)
      rb_pval_cvm <- numeric(num_chains)
      rb_pval_max_dist <- numeric(num_chains)
      
      for (c in 1:num_chains) {
        s_mat_mean <- s_mat_sum[[c]] / num_mcmc
        
        pvals_dist <- compute_distributional_pvalue_safe(s_mat_mean, t_grid, X_cen, p_valid, num_grid_points)
        rb_pval_wass[c] <- pvals_dist$wass
        rb_pval_cvm[c] <- pvals_dist$cvm
        rb_pval_max_dist[c] <- pvals_dist$max
      }
    }
  }
  
  mu_hat_train <- forest_samples_mu$predict(forest_dataset_train) * y_std_train + y_bar_train
  
  result <- list(
    "model_params" = general_params_updated,
    "alpha_samples" = alpha_samples,
    "sigma2_samples" = sigma2_samples,
    "mu_hat_train" = mu_hat_train,
    "train_set_metadata" = X_train_metadata
  )
  
  if (use_rao_blackwell) {
    if (test_type == "mean") {
      result[["rb_pval_quad"]] <- rb_pval_quad
      result[["rb_pval_max"]] <- rb_pval_max
    } else {
      result[["rb_pval_wass"]] <- rb_pval_wass
      result[["rb_pval_cvm"]] <- rb_pval_cvm
      result[["rb_pval_max"]] <- rb_pval_max_dist
    }
  } else {
    result[["pval_max_samples"]] <- pval_max_samples
    result[["pval_quad_samples"]] <- pval_quad_samples
  }
  
  class(result) <- "bcfscoretest"
  return(result)
}

# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

#' Prepare the Design Matrix for the Score Test
#'
#' Centers the candidate covariates, generates interactions if requested, and performs a 
#' QR decomposition to extract a linearly independent, full-rank basis. This makes the 
#' resulting score test statistic invariant to multicollinearity and dummy variable traps.
#'
#' @param X_train_raw Matrix of raw covariates.
#' @param X_train_metadata_init Metadata identifying continuous vs binary variables.
#' @param interaction_rule String dictating which interactions to compute ("continuous", "none", etc.).
#' @return A list containing `X_cen` (the full-rank centered design matrix) and `p_valid` (its rank).
prepare_score_test_design <- function(X_train_raw, X_train_metadata_init, interaction_rule) {
  n <- nrow(X_train_raw)
  sd_X <- apply(X_train_raw, 2, sd)
  valid_cols <- sd_X > 1e-6
  
  X_linear <- scale(X_train_raw[, valid_cols, drop = FALSE], center = TRUE, scale = TRUE)
  
  is_cont <- as.logical(as.numeric(X_train_metadata_init$is_continuous))
  is_bin <- as.logical(as.numeric(X_train_metadata_init$is_binary))
  
  if (interaction_rule == "continuous") {
    is_candidate <- is_cont
  } else if (interaction_rule == "continuous_or_binary") {
    is_candidate <- is_cont | is_bin
  } else if (interaction_rule == "none") {
    is_candidate <- rep(FALSE, ncol(X_train_raw))
  } else {
    is_candidate <- rep(TRUE, ncol(X_train_raw))
  }
  
  interaction_list <- list()
  num_covariates <- ncol(X_train_raw)
  if (num_covariates > 1) {
    for (j in 1:(num_covariates - 1)) {
      for (k in (j + 1):num_covariates) {
        if (is_candidate[j] || is_candidate[k]) {
          interaction_list[[length(interaction_list) + 1]] <- c(j, k)
        }
      }
    }
  }
  
  if (length(interaction_list) > 0) {
    ipairs <- do.call(cbind, interaction_list)
    X_interactions <- matrix(0, nrow = n, ncol = ncol(ipairs))
    for (idx in 1:ncol(ipairs)) {
      X_interactions[, idx] <- X_train_raw[, ipairs[1, idx]] * X_train_raw[, ipairs[2, idx]]
    }
    X_interactions <- scale(X_interactions, center = TRUE, scale = TRUE)
    valid_int <- apply(X_interactions, 2, function(col) {
      s <- sd(col, na.rm = TRUE)
      !is.na(s) && s > 1e-6
    })
    X_interactions <- X_interactions[, valid_int, drop = FALSE]
  } else {
    X_interactions <- NULL
  }
  
  X_cen_candidate <- cbind(X_linear, X_interactions)
  
  # Perform QR decomposition to drop linearly dependent columns and find exact rank
  qr_decomp <- qr(X_cen_candidate)
  p_valid <- qr_decomp$rank
  
  # Select only the linearly independent columns using pivot from QR
  independent_cols <- qr_decomp$pivot[1:p_valid]
  X_cen <- X_cen_candidate[, independent_cols, drop = FALSE]
  
  return(list(X_cen = X_cen, p_valid = p_valid))
}

#' Compute P-Values for the Mean Score Test
#'
#' Calculates the chi-squared (quadratic form) and maximum absolute Z-score p-values.
#' 
#' @param T_vec The score vector.
#' @param Var_T The covariance matrix of the score vector.
#' @param p_valid The degrees of freedom (rank of the design matrix).
#' @param n_sim Number of simulations used for the max test.
#' @return A list with `quad` (chi-squared p-value) and `max` (simulated max p-value).
compute_pmvnorm_safe <- function(T_vec, Var_T, p_valid, n_sim = 10000) {
  # Add a tiny ridge penalty to ensure positive definiteness in edge cases
  Var_T_ridge <- Var_T + diag(1e-8, p_valid)
  
  Var_T_inv <- tryCatch({
    solve(Var_T_ridge)
  }, error = function(e) {
    MASS::ginv(Var_T)
  })
  
  T_quad <- as.numeric(t(T_vec) %*% Var_T_inv %*% T_vec)
  pval_quad <- pchisq(T_quad, df = p_valid, lower.tail = FALSE)
  
  inv_sd_T <- 1 / sqrt(diag(Var_T_ridge))
  Cor_T <- diag(inv_sd_T) %*% Var_T_ridge %*% diag(inv_sd_T)
  
  Z_vec <- as.numeric(T_vec * inv_sd_T)
  max_Z <- max(abs(Z_vec))
  
  # Max test via simulation
  # We simulate Z_sim ~ N(0, Cor_T)
  # Then compute proportion of times max(abs(Z_sim)) >= max_Z
  pval_max <- tryCatch({
    chol_Cor <- chol(Cor_T)
    # Generate standard normals: (p_valid x n_sim)
    Z_standard <- matrix(rnorm(p_valid * n_sim), nrow = p_valid, ncol = n_sim)
    # Multiply by Cholesky factor to get correlated normals
    Z_sim <- crossprod(chol_Cor, Z_standard)
    # Find max absolute value for each simulation
    max_Z_sim <- apply(abs(Z_sim), 2, max)
    # Compute p-value
    mean(max_Z_sim >= max_Z)
  }, error = function(e) {
    # Fallback to Bonferroni bound if Cholesky fails (e.g., matrix not pos-def)
    min(1, p_valid * 2 * pnorm(-max_Z))
  })
  
  return(list(quad = pval_quad, max = pval_max))
}

#' Compute P-Values for the Distributional Score Test
#'
#' Computes the Wasserstein (integral of sqrt) and Cramer-von Mises (integral) style 
#' test statistics for the ECDF-based score test, and simulates p-values from the 
#' asymptotic Gaussian process (Brownian Bridge) under the null hypothesis.
#'
#' @param s_mat_mean The posterior mean of the score matrix (n x K).
#' @param t_grid The K evaluation points for the ECDF.
#' @param X_cen The full-rank centered design matrix.
#' @param p_valid The rank of the design matrix.
#' @param K The number of grid points.
#' @param n_sim Number of simulations to draw from the null Gaussian process.
#' @return A list with `wass` (Wasserstein-style p-value) and `cvm` (CvM-style p-value).
compute_distributional_pvalue_safe <- function(s_mat_mean, t_grid, X_cen, p_valid, K, n_sim = 10000) {
  # Compute the Rao-Blackwellized score vectors
  U_k_mean <- crossprod(X_cen, s_mat_mean)
  
  # Construct the full block-diagonal design matrix scaled by the score process
  # This enables a highly efficient vectorized computation of the empirical covariance
  n <- nrow(X_cen)
  tilde_X <- matrix(0, nrow = n, ncol = p_valid * K)
  for (k in 1:K) {
    idx_k <- ((k-1)*p_valid + 1):(k*p_valid)
    tilde_X[, idx_k] <- X_cen * s_mat_mean[, k]
  }
  
  # Compute the empirical Huber-White sandwich estimator for the entire score process
  # This inherently captures the variance reduction from Rao-Blackwellization and 
  # is robust to underlying heteroskedasticity, directly solving the over-conservatism.
  Sigma_full <- crossprod(tilde_X)
  
  # Add a tiny ridge penalty to ensure positive definiteness in edge cases
  Sigma_full <- Sigma_full + diag(1e-8, p_valid * K)
  
  T_k_obs <- numeric(K)
  V_k_inv_list <- list()
  for (k in 1:K) {
    idx_k <- ((k-1)*p_valid + 1):(k*p_valid)
    V_k <- Sigma_full[idx_k, idx_k, drop = FALSE]
    V_k_inv <- tryCatch(solve(V_k), error = function(e) MASS::ginv(V_k))
    V_k_inv_list[[k]] <- V_k_inv
    T_k_obs[k] <- as.numeric(t(U_k_mean[, k]) %*% V_k_inv %*% U_k_mean[, k])
  }
  
  t_diffs <- diff(t_grid)
  T_wass_obs <- sum(t_diffs * sqrt(pmax(T_k_obs[-K], 0))) 
  T_cvm_obs <- mean(T_k_obs)
  
  # Max Score Statistic (max Z) calculation
  U_vec <- as.numeric(U_k_mean)
  inv_sd_U <- 1 / sqrt(diag(Sigma_full))
  max_Z_obs <- max(abs(U_vec * inv_sd_U))
  
  chol_Sigma <- tryCatch(chol(Sigma_full), error = function(e) NULL)
  
  if (is.null(chol_Sigma)) {
    eig <- eigen(Sigma_full, symmetric = TRUE)
    pos_eig <- eig$values > 1e-8
    L <- sweep(eig$vectors[, pos_eig, drop = FALSE], 2, sqrt(eig$values[pos_eig]), "*")
    Z_standard <- matrix(rnorm(sum(pos_eig) * n_sim), nrow = sum(pos_eig), ncol = n_sim)
    U_sim <- L %*% Z_standard
  } else {
    Z_standard <- matrix(rnorm(p_valid * K * n_sim), nrow = p_valid * K, ncol = n_sim)
    U_sim <- t(chol_Sigma) %*% Z_standard
  }
  
  T_k_sim_matrix <- matrix(0, nrow = K, ncol = n_sim)
  for (k in 1:K) {
    idx_k <- ((k-1)*p_valid + 1):(k*p_valid)
    U_k_sim <- U_sim[idx_k, , drop = FALSE]
    T_k_sim_matrix[k, ] <- colSums(U_k_sim * (V_k_inv_list[[k]] %*% U_k_sim))
  }
  
  T_wass_sim <- as.numeric(t_diffs %*% sqrt(pmax(T_k_sim_matrix[-K, ], 0)))
  T_cvm_sim <- colMeans(T_k_sim_matrix)
  
  Z_sim <- U_sim * inv_sd_U
  max_Z_sim <- apply(abs(Z_sim), 2, max)
  
  pval_wass <- mean(T_wass_sim >= T_wass_obs)
  pval_cvm <- mean(T_cvm_sim >= T_cvm_obs)
  pval_max <- mean(max_Z_sim >= max_Z_obs)
  
  return(list(wass = pval_wass, cvm = pval_cvm, max = pval_max))
}