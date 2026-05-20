#' Fit an Integrated BCF Model + Chiseled Boundary
#'
#' @param X_train Covariates used to split trees in the ensemble.
#' @param Z_train Vector of treatment assignments.
#' @param y_train Outcome to be modeled by the ensemble.
#' @param propensity_train (Optional) Vector of propensity scores.
#' @param tau_c The clinical decision threshold for the boundary. Default: 0.0.
#' @param kappa_weight kappa_weight for the decision boundary.
#' @param rfx_group_ids_train (Optional) Group labels used for an additive random effects model.
#' @param rfx_basis_train (Optional) Basis for "random-slope" regression.
#' @param num_gfr Number of "warm-start" iterations. Default: 5.
#' @param num_burnin Number of "burn-in" iterations. Default: 100.
#' @param num_mcmc Number of "retained" iterations. Default: 500.
#' @param general_params (Optional) A list of general model parameters.
#' @param prognostic_forest_params (Optional) A list of prognostic forest model parameters.
#' @param variance_forest_params (Optional) A list of variance forest model parameters.
#' @return A list of class `bcfchiseler` containing posterior samples, inclusion probabilities, and boundary coefficients.
#' @export
bcf_integrated_chiseler <- function(X_train, Z_train, y_train, propensity_train = NULL, 
                                    tau_c = 0.0, kappa_weight = 2.0, 
                                    rfx_group_ids_train = NULL, rfx_basis_train = NULL, 
                                    num_gfr = 5, num_burnin = 100, num_mcmc = 500, 
                                    general_params = list(), prognostic_forest_params = list(), 
                                    variance_forest_params = list()) {
  
  general_params_default <- list(
    cutpoint_grid_size = 100, standardize = TRUE, sample_sigma2_global = TRUE, sigma2_global_init = NULL, 
    sigma2_global_shape = 1, sigma2_global_scale = 0.001, variable_weights = NULL, 
    propensity_covariate = "mu", adaptive_coding = FALSE, rfx_prior_var = NULL, random_seed = -1, 
    keep_burnin = FALSE, keep_gfr = FALSE, keep_every = 1, num_chains = 1, verbose = TRUE, 
    probit_outcome_model = FALSE, standardize_cov = FALSE, interaction_rule = "continuous"
  )
  general_params_updated <- preprocessParams(general_params_default, general_params)
  
  rinvgamma <- function(shape, scale) {
    if (shape <= 0.0 || scale <= 0.0) stop("Shape and scale must be positive.");
    return(1.0 / rgamma(1, shape, scale))
  }
  
  if(general_params_updated$verbose) print("Pre-Processing data for Integrated Chiseler!")
  
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
                                  num_gfr = 50, num_burnin = 0, num_mcmc = 0)
    propensity_train <- as.matrix(rowMeans(bart_model_propensity$y_hat_train[, 11:50]))
  }
  if(is.null(propensity_train)) propensity_train <- rep(mean(Z_train), n)
  propensity_train <- as.matrix(propensity_train)
  
  # Center treatment indicator
  Z_cen <- Z_train - as.vector(propensity_train)
  
  # =====================================================================
  # BOUNDARY DESIGN MATRIX SETUP
  # =====================================================================
  X_boundary <- cbind(1, as.matrix(X_train_raw))
  colnames(X_boundary)[1] <- "Intercept"
  p_bound <- ncol(X_boundary)
  
  hs_beta <- rep(0, p_bound)
  hs_lambda_sq <- rep(1, p_bound)
  hs_tau_sq <- 1
  hs_nu <- rep(1, p_bound)
  hs_xi <- 1
  # =====================================================================
  
  num_chains <- general_params_updated$num_chains
  alpha_samples <- matrix(0, nrow = num_chains, ncol = num_mcmc)
  sigma2_samples <- matrix(0, nrow = num_chains, ncol = num_mcmc)
  
  # Track Boundary and Pseudo-CATEs across MCMC
  beta_samples <- matrix(0, nrow = num_chains * num_mcmc, ncol = p_bound)
  pseudo_cate_samples <- matrix(0, nrow = num_chains * num_mcmc, ncol = n)
  
  alpha_prior_var <- 100.0
  
  prognostic_forest_params_default <- list(
    num_trees = 250, alpha = 0.95, beta = 2.0, min_samples_leaf = 5, max_depth = 10, 
    sample_sigma2_leaf = TRUE, sigma2_leaf_init = NULL, sigma2_leaf_shape = 3, 
    sigma2_leaf_scale = NULL, keep_vars = NULL, drop_vars = NULL
  )
  prognostic_forest_params_updated <- preprocessParams(prognostic_forest_params_default, prognostic_forest_params)
  
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
  set.seed(ifelse(random_seed == -1, sample(1:10000, 1), random_seed))
  rng <- createCppRNG(random_seed)
  verbose <- general_params_updated$verbose
  probit_outcome_model <- general_params_updated$probit_outcome_model
  
  num_trees_mu <- prognostic_forest_params_updated$num_trees
  sample_sigma2_leaf_mu <- prognostic_forest_params_updated$sample_sigma2_leaf
  a_leaf_mu <- prognostic_forest_params_updated$sigma2_leaf_shape
  b_leaf_mu <- prognostic_forest_params_updated$sigma2_leaf_scale
  keep_vars_mu <- prognostic_forest_params_updated$keep_vars
  drop_vars_mu <- prognostic_forest_params_updated$drop_vars
  
  num_trees_variance <- variance_forest_params_updated$num_trees
  include_variance_forest <- num_trees_variance > 0
  variance_forest_init <- variance_forest_params_updated$variance_forest_init
  
  variable_weights <- general_params_updated$variable_weights
  if (is.null(variable_weights)) variable_weights <- rep(1/ncol(X_train_raw), ncol(X_train_raw))
  
  if (!is.null(keep_vars_mu)) {
    if (is.character(keep_vars_mu)) variable_subset_mu <- unname(which(names(X_train_raw) %in% keep_vars_mu))
    else variable_subset_mu <- keep_vars_mu
  } else if (!is.null(drop_vars_mu)) {
    if (is.character(drop_vars_mu)) variable_subset_mu <- unname(which(!(names(X_train_raw) %in% drop_vars_mu)))
    else variable_subset_mu <- (1:ncol(X_train_raw))[!(1:ncol(X_train_raw) %in% drop_vars_mu)]
  } else {
    variable_subset_mu <- 1:ncol(X_train_raw)
  }
  
  variable_weights_adj <- 1/sapply(original_var_indices, function(x) sum(original_var_indices == x))
  variable_weights <- variable_weights[original_var_indices]*variable_weights_adj
  
  variable_weights_variance <- variable_weights_mu <- variable_weights
  variable_weights_mu[!(original_var_indices %in% variable_subset_mu)] <- 0
  
  if (general_params_updated$propensity_covariate %in% c("mu", "both")) {
    feature_types <- as.integer(c(feature_types, rep(0, ncol(propensity_train))))
    X_train_forest <- cbind(X_train_forest, propensity_train)
    variable_weights_mu <- c(variable_weights_mu, rep(1/num_cov_orig, ncol(propensity_train)))
    if(include_variance_forest) variable_weights_variance <- c(variable_weights_variance, rep(0, ncol(propensity_train)))
  }
  variable_weights_mu <- variable_weights_mu / sum(variable_weights_mu)
  if(include_variance_forest) variable_weights_variance <- variable_weights_variance / sum(variable_weights_variance)
  
  if (probit_outcome_model) {
    y_bar_train <- qnorm(mean(y_train))
    y_std_train <- 1
    resid_train_base <- y_train - mean(y_train)
    init_mu <- 0.0
    current_sigma2_init <- 1.0
    sigma2_leaf_mu <- 2/(num_trees_mu)
  } else {
    if (standardize) {
      y_bar_train <- mean(y_train)
      y_std_train <- sd(y_train)
    } else {
      y_bar_train <- 0
      y_std_train <- 1
    }
    resid_train_base <- (y_train - y_bar_train) / y_std_train
    init_mu <- mean(resid_train_base)
    current_sigma2_init <- general_params_updated$sigma2_global_init %||% (1.0 * var(resid_train_base))
    if (is.null(variance_forest_init)) variance_forest_init <- 1.0 * var(resid_train_base)
    if (is.null(b_leaf_mu)) b_leaf_mu <- var(resid_train_base) / (num_trees_mu)
    sigma2_leaf_mu <- prognostic_forest_params_updated$sigma2_leaf_init %||% (2.0 * var(resid_train_base) / num_trees_mu)
  }
  current_leaf_scale_mu <- as.matrix(sigma2_leaf_mu)
  
  has_rfx <- !is.null(rfx_group_ids_train)
  if (has_rfx) {
    if (is.null(rfx_basis_train)) rfx_basis_train <- matrix(1, nrow = n, ncol = 1)
    group_ids_factor <- factor(rfx_group_ids_train)
    rfx_group_ids_train <- as.integer(group_ids_factor)
    num_rfx_groups <- length(unique(rfx_group_ids_train))
    num_rfx_components <- ncol(rfx_basis_train)
    
    rfx_dataset_train <- createRandomEffectsDataset(rfx_group_ids_train, rfx_basis_train)
    rfx_tracker_train <- createRandomEffectsTracker(rfx_group_ids_train)
    rfx_model <- createRandomEffectsModel(num_rfx_components, num_rfx_groups)
    rfx_samples <- createRandomEffectSamples(num_rfx_components, num_rfx_groups, rfx_tracker_train)
    
    xi_init <- matrix(0, num_rfx_components, num_rfx_groups)
    for (i in 1:num_rfx_groups) {
      group_subset_indices <- rfx_group_ids_train == i
      rfx_group_model <- lm(resid_train_base[group_subset_indices] ~ 0 + rfx_basis_train[group_subset_indices,])
      xi_init[,i] <- unname(coef(rfx_group_model))
    }
  }
  
  num_samples <- num_gfr + num_burnin + num_mcmc
  sample_counter <- 0
  
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
    
    if (include_variance_forest) {
      forest_model_config_variance <- createForestModelConfig(
        feature_types = feature_types, num_trees = num_trees_variance, num_features = ncol(X_train_forest), 
        num_observations = n, variable_weights = variable_weights_variance, leaf_dimension = 1, 
        alpha = variance_forest_params_updated$alpha, beta = variance_forest_params_updated$beta, 
        min_samples_leaf = variance_forest_params_updated$min_samples_leaf, max_depth = variance_forest_params_updated$max_depth, 
        leaf_model_type = 3, cutpoint_grid_size = cutpoint_grid_size
      )
      forest_model_variance <- createForestModel(forest_dataset_train, forest_model_config_variance, global_model_config)
      forest_samples_variance <- createForestSamples(num_trees_variance, 1, TRUE, TRUE)
      active_forest_variance <- createForest(num_trees_variance, 1, TRUE, TRUE)
      active_forest_variance$prepare_for_sampler(forest_dataset_train, outcome_train, forest_model_variance, 3, variance_forest_init)
    }
    
    if (has_rfx) {
      rfx_model$set_working_parameter(rep(1, num_rfx_components))
      rfx_model$set_group_parameters(xi_init)
      rfx_model$set_working_parameter_cov(diag(1, num_rfx_components))
      rfx_model$set_group_parameter_cov(diag(general_params_updated$rfx_prior_var %||% 1, num_rfx_components))
    }
    
    for (i in 1:num_samples) {
      is_mcmc <- i > (num_gfr + num_burnin)
      is_gfr <- i <= num_gfr
      
      if (verbose && (i %% 100 == 0 || i == 1)) {
        if (is_gfr) cat("Chain", chain_num, "- Sampling", i, "of", num_gfr, "GFR draws\n")
        else if (!is_mcmc) cat("Chain", chain_num, "- Sampling", i - num_gfr, "of", num_burnin, "Burn-in draws\n")
        else cat("Chain", chain_num, "- Sampling", i - num_gfr - num_burnin, "of", num_mcmc, "MCMC draws\n")
      }
      
      if (probit_outcome_model) {
        mu_forest_pred <- active_forest_mu$predict(forest_dataset_train)
        forest_pred <- mu_forest_pred + alpha * Z_train
        u0 <- runif(sum(y_train == 0), 0, pnorm(0 - forest_pred[y_train == 0]))
        u1 <- runif(sum(y_train == 1), pnorm(0 - forest_pred[y_train == 1]), 1)
        resid_train[y_train == 0] <- forest_pred[y_train == 0] + qnorm(u0)
        resid_train[y_train == 1] <- forest_pred[y_train == 1] + qnorm(u1)
        outcome_train$update_data(resid_train - forest_pred)
        current_sigma2 <- 1
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
      rfx_preds <- if (has_rfx) rfx_model$predict(rfx_dataset_train, rfx_tracker_train) else 0
      
      r_alpha <- resid_train - mu_x_raw_train - rfx_preds
      
      var_alpha <- 1.0 / (sum(Z_cen^2) / current_sigma2 + 1.0 / alpha_prior_var)
      mean_alpha <- var_alpha * sum(r_alpha * Z_cen) / current_sigma2
      alpha <- rnorm(1, mean = mean_alpha, sd = sqrt(var_alpha))
      
      resid_full <- r_alpha - alpha * Z_cen
      outcome_train$update_data(resid_full)
      
      if (include_variance_forest) {
        forest_model_variance$sample_one_iteration(
          forest_dataset = forest_dataset_train, residual = outcome_train, forest_samples = forest_samples_variance, 
          active_forest = active_forest_variance, rng = rng, forest_model_config = forest_model_config_variance, 
          global_model_config = global_model_config, keep_forest = is_mcmc, gfr = is_gfr
        )
      }
      
      if (sample_sigma2_global && !probit_outcome_model) {
        shape_sigma_post <- 0.001 + n / 2
        rate_sigma_post <- 0.001 + 0.5 * sum(outcome_train$get_data()^2)
        current_sigma2 <- max(rinvgamma(shape_sigma_post, rate_sigma_post), 1e-9)
        global_model_config$update_global_error_variance(current_sigma2)
      }
      
      if (has_rfx) {
        rfx_model$sample_random_effect(rfx_dataset_train, outcome_train, rfx_tracker_train, rfx_samples, is_mcmc, current_sigma2, rng)
      }
      
      s_i <- resid_full * as.vector(Z_cen)
      
      # THE FIX: Mathematically forbid division by exactly zero
      prop_safe <- pmin(pmax(as.vector(propensity_train), 0.01), 0.99)
      
      # 2. Calculate the Pseudo-CATE securely
      alpha_tilde <- alpha + (s_i / (prop_safe * (1 - prop_safe)))
      
      # 3. Create the shifting labels and CHISELING weights
      y_star <- as.integer(alpha_tilde >= tau_c)
      
      weights <- exp(-kappa_weight * abs(alpha_tilde - tau_c)) + 1e-6
      
      # Final safeguard to ensure mean(weights) is never exactly 0
      w_mean <- mean(weights)
      if (is.na(w_mean) || w_mean < 1e-8) w_mean <- 1e-8
      weights <- weights / w_mean 
      
      hs_step <- horseshoe_probit_step_cpp(
        X = X_boundary, 
        y_star = y_star, 
        weights = weights, 
        beta_in = hs_beta, 
        lambda_sq_in = hs_lambda_sq, 
        tau_sq = hs_tau_sq, 
        nu_in = hs_nu, 
        xi = hs_xi
      )
      
      # 5. Overwrite state (allows the chain to carry forward)
      hs_beta <- hs_step$beta
      hs_lambda_sq <- hs_step$lambda_sq
      hs_tau_sq <- hs_step$tau_sq
      hs_nu <- hs_step$nu
      hs_xi <- hs_step$xi
      
      # =====================================================================
      # RETAINED SAMPLES SAVING (MCMC Only)
      # =====================================================================
      if (is_mcmc) {
        mcmc_counter <- i - num_gfr - num_burnin
        global_iter_idx <- mcmc_counter + (chain_num - 1) * num_mcmc
        
        alpha_samples[chain_num, mcmc_counter] <- alpha
        sigma2_samples[chain_num, mcmc_counter] <- current_sigma2
        pseudo_cate_samples[global_iter_idx, ] <- alpha_tilde
        beta_samples[global_iter_idx, ] <- hs_beta
      }
    }
  }
  
  if(general_params_updated$verbose) print("Calculating Level Set Posterior Probabilities...")
  
  # Calculate Smoothed Inclusion Probabilities (Using the Horseshoe Boundary)
  boundary_linear_preds <- beta_samples %*% t(X_boundary)
  level_set_prob_smoothed <- colMeans(boundary_linear_preds > 0)
  
  # Calculate Raw Inclusion Probabilities (Using just the Pseudo-CATEs)
  level_set_prob_raw <- colMeans(pseudo_cate_samples >= tau_c)
  
  mu_hat_train <- forest_samples_mu$predict(forest_dataset_train) * y_std_train + y_bar_train
  colnames(beta_samples) <- colnames(X_boundary)
  
  result <- list(
    "model_params" = general_params_updated,
    "alpha_samples" = alpha_samples,
    "sigma2_samples" = sigma2_samples,
    "beta_boundary_samples" = beta_samples,
    "pseudo_cate_samples" = pseudo_cate_samples,
    "level_set_prob_smoothed" = level_set_prob_smoothed,
    "level_set_prob_raw" = level_set_prob_raw,
    "mu_hat_train" = mu_hat_train,
    "train_set_metadata" = X_train_metadata
  )
  
  if (internal_propensity_model) result[["bart_propensity_model"]] <- bart_model_propensity
  if (include_variance_forest) result[["forests_variance"]] <- forest_samples_variance
  if (has_rfx) result[["rfx_samples"]] <- rfx_samples
  
  class(result) <- "bcfchiseler"
  return(result)
}