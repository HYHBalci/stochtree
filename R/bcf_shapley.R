#' Run the Bayesian Causal Forest (BCF) algorithm for regularized causal effect estimation. 
#'
#' @param X_train Covariates used to split trees in the ensemble. May be provided either as a dataframe or a matrix. 
#' Matrix covariates will be assumed to be all numeric. Covariates passed as a dataframe will be 
#' preprocessed based on the variable types (e.g. categorical columns stored as unordered factors will be one-hot encoded, 
#' categorical columns stored as ordered factors will passed as integers to the core algorithm, along with the metadata 
#' that the column is ordered categorical).
#' @param Z_train Vector of (continuous or binary) treatment assignments.
#' @param y_train Outcome to be modeled by the ensemble.
#' @param propensity_train (Optional) Vector of propensity scores. If not provided, this will be estimated from the data.
#' @param rfx_group_ids_train (Optional) Group labels used for an additive random effects model.
#' @param rfx_basis_train (Optional) Basis for "random-slope" regression in an additive random effects model.
#' If `rfx_group_ids_train` is provided with a regression basis, an intercept-only random effects model 
#' will be estimated.
#' @param X_test (Optional) Test set of covariates used to define "out of sample" evaluation data. 
#' May be provided either as a dataframe or a matrix, but the format of `X_test` must be consistent with 
#' that of `X_train`.
#' @param Z_test (Optional) Test set of (continuous or binary) treatment assignments.
#' @param propensity_test (Optional) Vector of propensity scores. If not provided, this will be estimated from the data.
#' @param rfx_group_ids_test (Optional) Test set group labels used for an additive random effects model. 
#' We do not currently support (but plan to in the near future), test set evaluation for group labels
#' that were not in the training set.
#' @param rfx_basis_test (Optional) Test set basis for "random-slope" regression in additive random effects model.
#' @param num_gfr Number of "warm-start" iterations run using the grow-from-root algorithm (He and Hahn, 2021). Default: 5.
#' @param num_burnin Number of "burn-in" iterations of the MCMC sampler. Default: 0.
#' @param num_mcmc Number of "retained" iterations of the MCMC sampler. Default: 100.
#' @param previous_model_json (Optional) JSON string containing a previous BCF model. This can be used to "continue" a sampler interactively after inspecting the samples or to run parallel chains "warm-started" from existing forest samples. Default: `NULL`.
#' @param previous_model_warmstart_sample_num (Optional) Sample number from `previous_model_json` that will be used to warmstart this BCF sampler. One-indexed (so that the first sample is used for warm-start by setting `previous_model_warmstart_sample_num = 1`). Default: `NULL`.
#' @param general_params (Optional) A list of general (non-forest-specific) model parameters, each of which has a default value processed internally, so this argument list is optional.
#'
#'   - `cutpoint_grid_size` Maximum size of the "grid" of potential cutpoints to consider in the GFR algorithm. Default: `100`.
#'   - `standardize` Whether or not to standardize the outcome (and store the offset / scale in the model object). Default: `TRUE`.
#'   - `sample_sigma2_global` Whether or not to update the `sigma^2` global error variance parameter based on `IG(sigma2_global_shape, sigma2_global_scale)`. Default: `TRUE`.
#'   - `sigma2_global_init` Starting value of global error variance parameter. Calibrated internally as `1.0*var((y_train-mean(y_train))/sd(y_train))` if not set.
#'   - `sigma2_global_shape` Shape parameter in the `IG(sigma2_global_shape, sigma2_global_scale)` global error variance model. Default: `0`.
#'   - `sigma2_global_scale` Scale parameter in the `IG(sigma2_global_shape, sigma2_global_scale)` global error variance model. Default: `0`.
#'   - `variable_weights` Numeric weights reflecting the relative probability of splitting on each variable. Does not need to sum to 1 but cannot be negative. Defaults to `rep(1/ncol(X_train), ncol(X_train))` if not set here. Note that if the propensity score is included as a covariate in either forest, its weight will default to `1/ncol(X_train)`. A workaround if you wish to provide a custom weight for the propensity score is to include it as a column in `X_train` and then set `propensity_covariate` to `'none'` adjust `keep_vars` accordingly for the `mu` or `tau` forests.
#'   - `propensity_covariate` Whether to include the propensity score as a covariate in either or both of the forests. Enter `"none"` for neither, `"mu"` for the prognostic forest, `"tau"` for the treatment forest, and `"both"` for both forests. If this is not `"none"` and a propensity score is not provided, it will be estimated from (`X_train`, `Z_train`) using `stochtree::bart()`. Default: `"mu"`.
#'   - `adaptive_coding` Whether or not to use an "adaptive coding" scheme in which a binary treatment variable is not coded manually as (0,1) or (-1,1) but learned via parameters `b_0` and `b_1` that attach to the outcome model `[b_0 (1-Z) + b_1 Z] tau(X)`. This is ignored when Z is not binary. Default: `TRUE`.
#'   - `control_coding_init` Initial value of the "control" group coding parameter. This is ignored when Z is not binary. Default: `-0.5`.
#'   - `treated_coding_init` Initial value of the "treatment" group coding parameter. This is ignored when Z is not binary. Default: `0.5`.
#'   - `rfx_prior_var` Prior on the (diagonals of the) covariance of the additive group-level random regression coefficients. Must be a vector of length `ncol(rfx_basis_train)`. Default: `rep(1, ncol(rfx_basis_train))`
#'   - `random_seed` Integer parameterizing the C++ random number generator. If not specified, the C++ random number generator is seeded according to `std::random_device`.
#'   - `keep_burnin` Whether or not "burnin" samples should be included in the stored samples of forests and other parameters. Default `FALSE`. Ignored if `num_mcmc = 0`.
#'   - `keep_gfr` Whether or not "grow-from-root" samples should be included in the stored samples of forests and other parameters. Default `FALSE`. Ignored if `num_mcmc = 0`.
#'   - `keep_every` How many iterations of the burned-in MCMC sampler should be run before forests and parameters are retained. Default `1`. Setting `keep_every <- k` for some `k > 1` will "thin" the MCMC samples by retaining every `k`-th sample, rather than simply every sample. This can reduce the autocorrelation of the MCMC samples.
#'   - `num_chains` How many independent MCMC chains should be sampled. If `num_mcmc = 0`, this is ignored. If `num_gfr = 0`, then each chain is run from root for `num_mcmc * keep_every + num_burnin` iterations, with `num_mcmc` samples retained. If `num_gfr > 0`, each MCMC chain will be initialized from a separate GFR ensemble, with the requirement that `num_gfr >= num_chains`. Default: `1`.
#'   - `verbose` Whether or not to print progress during the sampling loops. Default: `FALSE`.
#'   - `probit_outcome_model` Whether or not the outcome should be modeled as explicitly binary via a probit link. If `TRUE`, `y` must only contain the values `0` and `1`. Default: `FALSE`.
#'
#' @param prognostic_forest_params (Optional) A list of prognostic forest model parameters, each of which has a default value processed internally, so this argument list is optional.
#'
#'   - `num_trees` Number of trees in the ensemble for the prognostic forest. Default: `250`. Must be a positive integer.
#'   - `alpha` Prior probability of splitting for a tree of depth 0 in the prognostic forest. Tree split prior combines `alpha` and `beta` via `alpha*(1+node_depth)^-beta`. Default: `0.95`.
#'   - `beta` Exponent that decreases split probabilities for nodes of depth > 0 in the prognostic forest. Tree split prior combines `alpha` and `beta` via `alpha*(1+node_depth)^-beta`. Default: `2`.
#'   - `min_samples_leaf` Minimum allowable size of a leaf, in terms of training samples, in the prognostic forest. Default: `5`.
#'   - `max_depth` Maximum depth of any tree in the ensemble in the prognostic forest. Default: `10`. Can be overridden with ``-1`` which does not enforce any depth limits on trees.
#'   - `variable_weights` Numeric weights reflecting the relative probability of splitting on each variable in the prognostic forest. Does not need to sum to 1 but cannot be negative. Defaults to `rep(1/ncol(X_train), ncol(X_train))` if not set here.
#'   - `sample_sigma2_leaf` Whether or not to update the leaf scale variance parameter based on `IG(sigma2_leaf_shape, sigma2_leaf_scale)`.
#'   - `sigma2_leaf_init` Starting value of leaf node scale parameter. Calibrated internally as `1/num_trees` if not set here.
#'   - `sigma2_leaf_shape` Shape parameter in the `IG(sigma2_leaf_shape, sigma2_leaf_scale)` leaf node parameter variance model. Default: `3`.
#'   - `sigma2_leaf_scale` Scale parameter in the `IG(sigma2_leaf_shape, sigma2_leaf_scale)` leaf node parameter variance model. Calibrated internally as `0.5/num_trees` if not set here.
#'   - `keep_vars` Vector of variable names or column indices denoting variables that should be included in the forest. Default: `NULL`.
#'   - `drop_vars` Vector of variable names or column indices denoting variables that should be excluded from the forest. Default: `NULL`. If both `drop_vars` and `keep_vars` are set, `drop_vars` will be ignored.
#'
#' @param treatment_effect_forest_params (Optional) A list of treatment effect forest model parameters, each of which has a default value processed internally, so this argument list is optional.
#'
#'   - `num_trees` Number of trees in the ensemble for the treatment effect forest. Default: `50`. Must be a positive integer.
#'   - `alpha` Prior probability of splitting for a tree of depth 0 in the treatment effect forest. Tree split prior combines `alpha` and `beta` via `alpha*(1+node_depth)^-beta`. Default: `0.25`.
#'   - `beta` Exponent that decreases split probabilities for nodes of depth > 0 in the treatment effect forest. Tree split prior combines `alpha` and `beta` via `alpha*(1+node_depth)^-beta`. Default: `3`.
#'   - `min_samples_leaf` Minimum allowable size of a leaf, in terms of training samples, in the treatment effect forest. Default: `5`.
#'   - `max_depth` Maximum depth of any tree in the ensemble in the treatment effect forest. Default: `5`. Can be overridden with ``-1`` which does not enforce any depth limits on trees.
#'   - `variable_weights` Numeric weights reflecting the relative probability of splitting on each variable in the treatment effect forest. Does not need to sum to 1 but cannot be negative. Defaults to `rep(1/ncol(X_train), ncol(X_train))` if not set here.
#'   - `sample_sigma2_leaf` Whether or not to update the leaf scale variance parameter based on `IG(sigma2_leaf_shape, sigma2_leaf_scale)`. Cannot (currently) be set to true if `ncol(Z_train)>1`. Default: `FALSE`.
#'   - `sigma2_leaf_init` Starting value of leaf node scale parameter. Calibrated internally as `1/num_trees` if not set here.
#'   - `sigma2_leaf_shape` Shape parameter in the `IG(sigma2_leaf_shape, sigma2_leaf_scale)` leaf node parameter variance model. Default: `3`.
#'   - `sigma2_leaf_scale` Scale parameter in the `IG(sigma2_leaf_shape, sigma2_leaf_scale)` leaf node parameter variance model. Calibrated internally as `0.5/num_trees` if not set here.
#'   - `delta_max` Maximum plausible conditional distributional treatment effect (i.e. P(Y(1) = 1 | X) - P(Y(0) = 1 | X)) when the outcome is binary. Only used when the outcome is specified as a probit model in `general_params`. Must be > 0 and < 1. Default: `0.9`. Ignored if `sigma2_leaf_init` is set directly, as this parameter is used to calibrate `sigma2_leaf_init`.
#'   - `keep_vars` Vector of variable names or column indices denoting variables that should be included in the forest. Default: `NULL`.
#'   - `drop_vars` Vector of variable names or column indices denoting variables that should be excluded from the forest. Default: `NULL`. If both `drop_vars` and `keep_vars` are set, `drop_vars` will be ignored.
#'
#' @param variance_forest_params (Optional) A list of variance forest model parameters, each of which has a default value processed internally, so this argument list is optional.
#'
#'   - `num_trees` Number of trees in the ensemble for the conditional variance model. Default: `0`. Variance is only modeled using a tree / forest if `num_trees > 0`.
#'   - `alpha` Prior probability of splitting for a tree of depth 0 in the variance model. Tree split prior combines `alpha` and `beta` via `alpha*(1+node_depth)^-beta`. Default: `0.95`.
#'   - `beta` Exponent that decreases split probabilities for nodes of depth > 0 in the variance model. Tree split prior combines `alpha` and `beta` via `alpha*(1+node_depth)^-beta`. Default: `2`.
#'   - `min_samples_leaf` Minimum allowable size of a leaf, in terms of training samples, in the variance model. Default: `5`.
#'   - `max_depth` Maximum depth of any tree in the ensemble in the variance model. Default: `10`. Can be overridden with ``-1`` which does not enforce any depth limits on trees.
#'   - `leaf_prior_calibration_param` Hyperparameter used to calibrate the `IG(var_forest_prior_shape, var_forest_prior_scale)` conditional error variance model. If `var_forest_prior_shape` and `var_forest_prior_scale` are not set below, this calibration parameter is used to set these values to `num_trees / leaf_prior_calibration_param^2 + 0.5` and `num_trees / leaf_prior_calibration_param^2`, respectively. Default: `1.5`.
#'   - `variance_forest_init` Starting value of root forest prediction in conditional (heteroskedastic) error variance model. Calibrated internally as `log(0.6*var((y_train-mean(y_train))/sd(y_train)))/num_trees` if not set.
#'   - `var_forest_prior_shape` Shape parameter in the `IG(var_forest_prior_shape, var_forest_prior_scale)` conditional error variance model (which is only sampled if `num_trees > 0`). Calibrated internally as `num_trees / 1.5^2 + 0.5` if not set.
#'   - `var_forest_prior_scale` Scale parameter in the `IG(var_forest_prior_shape, var_forest_prior_scale)` conditional error variance model (which is only sampled if `num_trees > 0`). Calibrated internally as `num_trees / 1.5^2` if not set.
#'   - `keep_vars` Vector of variable names or column indices denoting variables that should be included in the forest. Default: `NULL`.
#'   - `drop_vars` Vector of variable names or column indices denoting variables that should be excluded from the forest. Default: `NULL`. If both `drop_vars` and `keep_vars` are set, `drop_vars` will be ignored.
#'
#' @return List of sampling outputs and a wrapper around the sampled forests (which can be used for in-memory prediction on new data, or serialized to JSON on disk).
#' @export
#'
#' @examples
#' n <- 500
#' p <- 5
#' X <- matrix(runif(n*p), ncol = p)
#' mu_x <- (
#'     ((0 <= X[,1]) & (0.25 > X[,1])) * (-7.5) + 
#'     ((0.25 <= X[,1]) & (0.5 > X[,1])) * (-2.5) + 
#'     ((0.5 <= X[,1]) & (0.75 > X[,1])) * (2.5) + 
#'     ((0.75 <= X[,1]) & (1 > X[,1])) * (7.5)
#' )
#' pi_x <- (
#'     ((0 <= X[,1]) & (0.25 > X[,1])) * (0.2) + 
#'     ((0.25 <= X[,1]) & (0.5 > X[,1])) * (0.4) + 
#'     ((0.5 <= X[,1]) & (0.75 > X[,1])) * (0.6) + 
#'     ((0.75 <= X[,1]) & (1 > X[,1])) * (0.8)
#' )
#' tau_x <- (
#'     ((0 <= X[,2]) & (0.25 > X[,2])) * (0.5) + 
#'     ((0.25 <= X[,2]) & (0.5 > X[,2])) * (1.0) + 
#'     ((0.5 <= X[,2]) & (0.75 > X[,2])) * (1.5) + 
#'     ((0.75 <= X[,2]) & (1 > X[,2])) * (2.0)
#' )
#' Z <- rbinom(n, 1, pi_x)
#' noise_sd <- 1
#' y <- mu_x + tau_x*Z + rnorm(n, 0, noise_sd)
#' test_set_pct <- 0.2
#' n_test <- round(test_set_pct*n)
#' n_train <- n - n_test
#' test_inds <- sort(sample(1:n, n_test, replace = FALSE))
#' train_inds <- (1:n)[!((1:n) %in% test_inds)]
#' X_test <- X[test_inds,]
#' X_train <- X[train_inds,]
#' pi_test <- pi_x[test_inds]
#' pi_train <- pi_x[train_inds]
#' Z_test <- Z[test_inds]
#' Z_train <- Z[train_inds]
#' y_test <- y[test_inds]
#' y_train <- y[train_inds]
#' mu_test <- mu_x[test_inds]
#' mu_train <- mu_x[train_inds]
#' tau_test <- tau_x[test_inds]
#' tau_train <- tau_x[train_inds]
#' bcf_model <- bcf(X_train = X_train, Z_train = Z_train, y_train = y_train, 
#'                  propensity_train = pi_train, X_test = X_test, Z_test = Z_test, 
#'                  propensity_test = pi_test, num_gfr = 10, 
#'                  num_burnin = 0, num_mcmc = 10)
bcf_linear_shapley <- function(X_train, Z_train, y_train, propensity_train = NULL, rfx_group_ids_train = NULL, 
                              rfx_basis_train = NULL, X_test = NULL, Z_test = NULL, propensity_test = NULL, 
                              rfx_group_ids_test = NULL, rfx_basis_test = NULL, 
                              num_gfr = 5, num_burnin = 0, num_mcmc = 100, 
                              previous_model_json = NULL, previous_model_warmstart_sample_num = NULL, 
                              general_params = list(), prognostic_forest_params = list(), 
                              treatment_effect_forest_params = list(), variance_forest_params = list()) {
  # Update general BCF parameters
  general_params_default <- list(
    cutpoint_grid_size = 100, standardize = TRUE, 
    sample_sigma2_global = TRUE, sigma2_global_init = NULL, 
    sigma2_global_shape = 1, sigma2_global_scale = 0.001, 
    variable_weights = NULL, propensity_covariate = "mu", 
    adaptive_coding = TRUE, control_coding_init = -0.5, 
    treated_coding_init = 0.5, rfx_prior_var = NULL, 
    random_seed = -1, keep_burnin = FALSE, keep_gfr = FALSE, 
    keep_every = 1, num_chains = 1, verbose = T, sample_global_prior = "none", unlink = F, 
    propensity_seperate = "none", step_out = 0.5, max_steps = 50, gibbs = F, save_output = F, probit_outcome_model = F , interaction_rule = "continuous", standardize_cov = F, simple_prior = F, save_partial_residual = F, regularize_ATE = F, 
    sigma_residual = 0, hn_scale = 0, n_tijn = 1, use_ncp = F, shapley = T
  )
  general_params_updated <- preprocessParams(
    general_params_default, general_params
  )
  ####
  rinvgamma <- function(shape, scale){
    if (shape <= 0.0 || scale <= 0.0) {
      stop("Shape and scale must be positive.");
    }
    
    g = rgamma(1, shape, scale)
    out = 1.0 / g
    return(out)
  }
  ####
  interaction_pairs <- function(num_covariates, boolean_vector) {
    interaction_list <- list()
    if (num_covariates > 1) {
      for (j in 1:(num_covariates - 1)) {
        for (k in (j + 1):num_covariates) {
          if (boolean_vector[j] || boolean_vector[k])
            interaction_list[[length(interaction_list) + 1]] <- c(j, k)
        }
      }
    }
    if (length(interaction_list) == 0) return(matrix(nrow=2, ncol=0))
    return(do.call(cbind, interaction_list))
  }
  ##############
  n_tijn <- general_params_updated$n_tijn
  use_ncp <- general_params_updated$use_ncp
  hn_scale <- general_params_updated$hn_scale
  sigma_residual <- general_params_updated$sigma_residual
  sample_global_prior <- general_params_updated$sample_global_prior
  unlink <- general_params_updated$unlink 
  gibbs <- general_params_updated$gibbs
  simple_prior <- general_params_updated$simple_prior
  save_output <- general_params_updated$save_output
  interaction_rule <- general_params_updated$interaction_rule
  standardize_cov <- general_params_updated$standardize_cov
  regularize_ATE <- general_params_updated$regularize_ATE
  shapley <- general_params_updated$shapley
  
  propensity_seperate <- match.arg(general_params_updated$propensity_seperate, choices = c("none", "mu", "tau"))
  # Interaction term initialization
  save_partial_residual <- general_params_updated$save_partial_residual
  
  # Data handling
  if(is.logical(sample_global_prior)){
    stop("global prior definition should be a string")
  }
  if(general_params_updated$verbose){
    print("Pre-Processing data!")
  }
  handled_data_list <- standardize_X_by_index(X_train, process_data = standardize_cov, interaction_rule = interaction_rule, cat_coding_method = "difference")
  if(general_params_updated$verbose){
    print("Pre-Processing data done!")
    print("total number of parameters to be estimated by the model:")
    print(handled_data_list$p_int + ncol(X_train))
  }
  X_train <- handled_data_list$X_final
  X_final_var_info <- handled_data_list$X_final_var_info
  X_train_raw <- X_train
  p_int <- handled_data_list$p_int
  non_continous_idx_cpp <- handled_data_list$non_continous_idx_cpp
  boolean_continuous <- as.vector(X_final_var_info$is_continuous)
  if(interaction_rule == 'continuous'){
    boolean_continuous <- as.vector(X_final_var_info$is_continuous)
  } else if(interaction_rule == 'continuous_or_binary'){
    boolean_continuous <- as.vector(X_final_var_info$is_continuous) + as.vector(X_final_var_info$is_binary)
  } else{ #This means we allow all interactions. 
    boolean_continuous <- as.vector(X_final_var_info$is_continuous) + as.vector(X_final_var_info$is_binary) + as.vector(X_final_var_info$is_categorical)
  }
  
  full_design_matrix_train <- cbind(1, X_train, generate_interaction_matrix(X_train, X_final_var_info, interaction_rule, propensity_seperate))
  beta_int <- rep(0, p_int) 
  p_mod <- ncol(X_train)
  n <- nrow(X_train)
  alpha <- 0 

  if(propensity_seperate == "tau"){
    p_mod <- p_mod + 1
  }
  beta <- rep(0, p_mod)
  
  # Generate covariate names for output
  beta_names <- colnames(X_train_raw)
  if (propensity_seperate == "tau") {
    beta_names <- c(beta_names, "propensity")
  }
  
  # Generate interaction names
  p_main <- ncol(X_train_raw)
  main_cov_names <- colnames(X_train_raw)
  if (p_int > 0) {
    int_pairs_matrix <- interaction_pairs(p_main, boolean_continuous)
    beta_int_names <- apply(int_pairs_matrix, 2, function(pair) {
      paste(main_cov_names[pair[1]], main_cov_names[pair[2]], sep = ":")
    })
  } else {
    beta_int_names <- character(0)
  }
  
  ##### end block for name generation
  xi <- 1
  if (unlink) {   
    tau_beta <- c(rep(1, p_mod + regularize_ATE), rep(1, p_int))  # Separate shrinkage
    nu <- 1 / (tau_beta^2)  # Initialize auxiliary variables
  } else {
    tau_beta <- rep(1, p_mod + regularize_ATE)  # Local shrinkage parameters
    nu <- 1 / (tau_beta^2)  # Initialize auxiliary variables
  }
  
  tau_int <- 1 
  tau_glob <- 1 # prior scale for global shrinkage 
  # Residual standard deviation
  sigma <- 1  
  sigma2_samples <- numeric(num_mcmc)
  
  # Store all MCMC samples, accross different chains. 
  num_chains <- general_params_updated$num_chains
  alpha_samples <- matrix(0, nrow = num_chains, ncol = num_mcmc)
  #gamma samples
  all_partial_residuals <- array(0, dim = c(n, num_mcmc, num_chains))
  gamma_prop <- 0
  #####
  gamma_prop_samples <- matrix(0, nrow = num_chains, ncol = num_mcmc)
  if(save_output){
    xi_samples <- matrix(0, nrow = num_chains, ncol = num_mcmc)
  }
  beta_samples <- array(0, dim = c(num_chains, num_mcmc, p_mod))
  
  if(unlink){
    tau_beta_samples <- array(0, dim = c(num_chains, num_mcmc, p_mod + p_int + regularize_ATE))
    if(save_output){
      nu_samples <- array(0, dim = c(num_chains, num_mcmc, p_mod + p_int + regularize_ATE))
    }
  } else {
    tau_beta_samples <- array(0, dim = c(num_chains, num_mcmc, p_mod + regularize_ATE))
    if(save_output){
      nu_samples <- array(0, dim = c(num_chains, num_mcmc, p_mod + regularize_ATE))
    }
  }
  if (unlink && length(tau_beta) < p_mod + p_int) {
    stop("tau_beta is too short for unlink = TRUE")
  }
  beta_int_samples <- array(0, dim = c(num_chains, num_mcmc, p_int))
  tau_int_samples <- matrix(0, nrow = num_chains, ncol = num_mcmc)
  tau_glob_samples <- matrix(0, nrow = num_chains, ncol = num_mcmc)
  
  
  # Apply dimnames to sample arrays
  dimnames(beta_samples) <- list(Chain = NULL, Iteration = NULL, Covariate = beta_names)
  dimnames(beta_int_samples) <- list(Chain = NULL, Iteration = NULL, Interaction = beta_int_names)
  
  beta_names_with_alpha <- if(regularize_ATE) c("alpha", beta_names) else beta_names
  
  if(unlink){
    tau_beta_names <- c(beta_names_with_alpha, beta_int_names)
    dimnames(tau_beta_samples) <- list(Chain = NULL, Iteration = NULL, Parameter = tau_beta_names)
    if(save_output){
      dimnames(nu_samples) <- list(Chain = NULL, Iteration = NULL, Parameter = tau_beta_names)
    }
  } else {
    tau_beta_names <- beta_names_with_alpha
    dimnames(tau_beta_samples) <- list(Chain = NULL, Iteration = NULL, Parameter = tau_beta_names)
    if(save_output){
      dimnames(nu_samples) <- list(Chain = NULL, Iteration = NULL, Parameter = tau_beta_names)
    }
  }
  
  #########
  
  # Update mu forest BCF parameters
  prognostic_forest_params_default <- list(
    num_trees = 250, alpha = 0.95, beta = 2.0, 
    min_samples_leaf = 5, max_depth = 10, 
    sample_sigma2_leaf = TRUE, sigma2_leaf_init = NULL, 
    sigma2_leaf_shape = 3, sigma2_leaf_scale = NULL, 
    keep_vars = NULL, drop_vars = NULL
  )
  prognostic_forest_params_updated <- preprocessParams(
    prognostic_forest_params_default, prognostic_forest_params
  )
  
  # Update tau forest BCF parameters
  treatment_effect_forest_params_default <- list(
    num_trees = 50, alpha = 0.25, beta = 3.0, 
    min_samples_leaf = 5, max_depth = 5, 
    sample_sigma2_leaf = FALSE, sigma2_leaf_init = NULL, 
    sigma2_leaf_shape = 3, sigma2_leaf_scale = NULL, 
    keep_vars = NULL, drop_vars = NULL, 
    delta_max = 0.9
  )
  treatment_effect_forest_params_updated <- preprocessParams(
    treatment_effect_forest_params_default, treatment_effect_forest_params
  )
  
  # Update variance forest BCF parameters
  variance_forest_params_default <- list(
    num_trees = 0, alpha = 0.95, beta = 2.0, 
    min_samples_leaf = 5, max_depth = 10, 
    leaf_prior_calibration_param = 1.5, 
    variance_forest_init = NULL, 
    var_forest_prior_shape = NULL, 
    var_forest_prior_scale = NULL, 
    keep_vars = NULL, drop_vars = NULL
  )
  variance_forest_params_updated <- preprocessParams(
    variance_forest_params_default, variance_forest_params
  )
  
  ### Unpack all parameter values
  # 1. General parameters
  cutpoint_grid_size <- general_params_updated$cutpoint_grid_size
  standardize <- general_params_updated$standardize
  sample_sigma2_global <- general_params_updated$sample_sigma2_global
  sample_sigma_global <- sample_sigma2_global
  sigma2_init <- general_params_updated$sigma2_global_init
  a_global <- general_params_updated$sigma2_global_shape
  b_global <- general_params_updated$sigma2_global_scale
  variable_weights <- general_params_updated$variable_weights
  propensity_covariate <- general_params_updated$propensity_covariate
  adaptive_coding <- general_params_updated$adaptive_coding
  b_0 <- general_params_updated$control_coding_init
  b_1 <- general_params_updated$treated_coding_init
  rfx_prior_var <- general_params_updated$rfx_prior_var
  random_seed <- general_params_updated$random_seed
  keep_burnin <- general_params_updated$keep_burnin
  keep_gfr <- general_params_updated$keep_gfr
  keep_every <- general_params_updated$keep_every
  num_chains <- general_params_updated$num_chains
  verbose <- general_params_updated$verbose
  probit_outcome_model <- general_params_updated$probit_outcome_model
  save_output <- general_params_updated$save_output
  global_shrinkage <- general_params_updated$global_shrinkage #Allow for global shrinkage in the linear part.
  ## SAMPLER SETTINGS FOR LINEAR SLICE SAMPLER.
  max_steps <- general_params_updated$max_steps
  step_out <- general_params_updated$step_out
  save_output <- general_params_updated$save_output
  
  # 2. Mu forest parameters
  num_trees_mu <- prognostic_forest_params_updated$num_trees
  alpha_mu <- prognostic_forest_params_updated$alpha
  beta_mu <- prognostic_forest_params_updated$beta
  min_samples_leaf_mu <- prognostic_forest_params_updated$min_samples_leaf
  max_depth_mu <- prognostic_forest_params_updated$max_depth
  sample_sigma2_leaf_mu <- prognostic_forest_params_updated$sample_sigma2_leaf
  sample_sigma_leaf_mu <- sample_sigma2_leaf_mu
  sigma2_leaf_mu <- prognostic_forest_params_updated$sigma2_leaf_init
  a_leaf_mu <- prognostic_forest_params_updated$sigma2_leaf_shape
  b_leaf_mu <- prognostic_forest_params_updated$sigma2_leaf_scale
  keep_vars_mu <- prognostic_forest_params_updated$keep_vars
  drop_vars_mu <- prognostic_forest_params_updated$drop_vars
  
  # 3. Tau forest parameters
  num_trees_tau <- treatment_effect_forest_params_updated$num_trees
  alpha_tau <- treatment_effect_forest_params_updated$alpha
  beta_tau <- treatment_effect_forest_params_updated$beta
  min_samples_leaf_tau <- treatment_effect_forest_params_updated$min_samples_leaf
  max_depth_tau <- treatment_effect_forest_params_updated$max_depth
  sample_sigma2_leaf_tau <- treatment_effect_forest_params_updated$sample_sigma2_leaf
  sigma2_leaf_tau <- treatment_effect_forest_params_updated$sigma2_leaf_init
  a_leaf_tau <- treatment_effect_forest_params_updated$sigma2_leaf_shape
  b_leaf_tau <- treatment_effect_forest_params_updated$sigma2_leaf_scale
  keep_vars_tau <- treatment_effect_forest_params_updated$keep_vars
  drop_vars_tau <- treatment_effect_forest_params_updated$drop_vars
  delta_max <- treatment_effect_forest_params_updated$delta_max
  
  # 4. Variance forest parameters
  num_trees_variance <- variance_forest_params_updated$num_trees
  alpha_variance <- variance_forest_params_updated$alpha
  beta_variance <- variance_forest_params_updated$beta
  min_samples_leaf_variance <- variance_forest_params_updated$min_samples_leaf
  max_depth_variance <- variance_forest_params_updated$max_depth
  a_0 <- variance_forest_params_updated$leaf_prior_calibration_param
  variance_forest_init <- variance_forest_params_updated$init_root_val
  a_forest <- variance_forest_params_updated$var_forest_prior_shape
  b_forest <- variance_forest_params_updated$var_forest_prior_scale
  keep_vars_variance <- variance_forest_params_updated$keep_vars
  drop_vars_variance <- variance_forest_params_updated$drop_vars
  
  # Check if there are enough GFR samples to seed num_chains samplers
  if (num_gfr > 0) {
    if (num_chains > num_gfr) {
      stop("num_chains > num_gfr, meaning we do not have enough GFR samples to seed num_chains distinct MCMC chains")
    }
  }
  
  # Override keep_gfr if there are no MCMC samples
  if (num_mcmc == 0) keep_gfr <- TRUE
  
  # Check if previous model JSON is provided and parse it if so
  has_prev_model <- !is.null(previous_model_json)
  if (has_prev_model) {
  } else {
    previous_y_bar <- NULL
    previous_y_scale <- NULL
    previous_global_var_samples <- NULL
    previous_leaf_var_mu_samples <- NULL
    previous_leaf_var_tau_samples <- NULL
    previous_rfx_samples <- NULL
    previous_forest_samples_mu <- NULL
    previous_forest_samples_tau <- NULL
    previous_forest_samples_variance <- NULL
    previous_b_1_samples <- NULL
    previous_b_0_samples <- NULL
  }
  
  # Determine whether conditional variance will be modeled
  if (num_trees_variance > 0) include_variance_forest = TRUE
  else include_variance_forest = FALSE
  
  # Set the variance forest priors if not set
  if (include_variance_forest) {
    if (is.null(a_forest)) a_forest <- num_trees_variance / (a_0^2) + 0.5
    if (is.null(b_forest)) b_forest <- num_trees_variance / (a_0^2)
  } else {
    a_forest <- 1.
    b_forest <- 1.
  }
  
  # Variable weight preprocessing (and initialization if necessary)
  if (is.null(variable_weights)) {
    variable_weights = rep(1/ncol(X_train), ncol(X_train))
  }
  if (any(variable_weights < 0)) {
    stop("variable_weights cannot have any negative weights")
  }
  
  # Check covariates are matrix or dataframe
  if ((!is.data.frame(X_train)) && (!is.matrix(X_train))) {
    stop("X_train must be a matrix or dataframe")
  }
  if (!is.null(X_test)){
    if ((!is.data.frame(X_test)) && (!is.matrix(X_test))) {
      stop("X_test must be a matrix or dataframe")
    }
  }
  num_cov_orig <- ncol(X_train)
  
  # Check delta_max is valid
  if ((delta_max <= 0) || (delta_max >= 1)) {
    stop("delta_max must be > 0 and < 1")
  }
  
  # Standardize the keep variable lists to numeric indices
  if (!is.null(keep_vars_mu)) {
    if (is.character(keep_vars_mu)) {
      if (!all(keep_vars_mu %in% names(X_train))) {
        stop("keep_vars_mu includes some variable names that are not in X_train")
      }
      variable_subset_mu <- unname(which(names(X_train) %in% keep_vars_mu))
    } else {
      if (any(keep_vars_mu > ncol(X_train))) {
        stop("keep_vars_mu includes some variable indices that exceed the number of columns in X_train")
      }
      if (any(keep_vars_mu < 0)) {
        stop("keep_vars_mu includes some negative variable indices")
      }
      variable_subset_mu <- keep_vars_mu
    }
  } else if ((is.null(keep_vars_mu)) && (!is.null(drop_vars_mu))) {
    if (is.character(drop_vars_mu)) {
      if (!all(drop_vars_mu %in% names(X_train))) {
        stop("drop_vars_mu includes some variable names that are not in X_train")
      }
      variable_subset_mu <- unname(which(!(names(X_train) %in% drop_vars_mu)))
    } else {
      if (any(drop_vars_mu > ncol(X_train))) {
        stop("drop_vars_mu includes some variable indices that exceed the number of columns in X_train")
      }
      if (any(drop_vars_mu < 0)) {
        stop("drop_vars_mu includes some negative variable indices")
      }
      variable_subset_mu <- (1:ncol(X_train))[!(1:ncol(X_train) %in% drop_vars_mu)]
    }
  } else {
    variable_subset_mu <- 1:ncol(X_train)
  }
  if (!is.null(keep_vars_tau)) {
    if (is.character(keep_vars_tau)) {
      if (!all(keep_vars_tau %in% names(X_train))) {
        stop("keep_vars_tau includes some variable names that are not in X_train")
      }
      variable_subset_tau <- unname(which(names(X_train) %in% keep_vars_tau))
    } else {
      if (any(keep_vars_tau > ncol(X_train))) {
        stop("keep_vars_tau includes some variable indices that exceed the number of columns in X_train")
      }
      if (any(keep_vars_tau < 0)) {
        stop("keep_vars_tau includes some negative variable indices")
      }
      variable_subset_tau <- keep_vars_tau
    }
  } else if ((is.null(keep_vars_tau)) && (!is.null(drop_vars_tau))) {
    if (is.character(drop_vars_tau)) {
      if (!all(drop_vars_tau %in% names(X_train))) {
        stop("drop_vars_tau includes some variable names that are not in X_train")
      }
      variable_subset_tau <- unname(which(!(names(X_train) %in% drop_vars_tau)))
    } else {
      if (any(drop_vars_tau > ncol(X_train))) {
        stop("drop_vars_tau includes some variable indices that exceed the number of columns in X_train")
      }
      if (any(drop_vars_tau < 0)) {
        stop("drop_vars_tau includes some negative variable indices")
      }
      variable_subset_tau <- (1:ncol(X_train))[!(1:ncol(X_train) %in% drop_vars_tau)]
    }
  } else {
    variable_subset_tau <- 1:ncol(X_train)
  }
  if (!is.null(keep_vars_variance)) {
    if (is.character(keep_vars_variance)) {
      if (!all(keep_vars_variance %in% names(X_train))) {
        stop("keep_vars_variance includes some variable names that are not in X_train")
      }
      variable_subset_variance <- unname(which(names(X_train) %in% keep_vars_variance))
    } else {
      if (any(keep_vars_variance > ncol(X_train))) {
        stop("keep_vars_variance includes some variable indices that exceed the number of columns in X_train")
      }
      if (any(keep_vars_variance < 0)) {
        stop("keep_vars_variance includes some negative variable indices")
      }
      variable_subset_variance <- keep_vars_variance
    }
  } else if ((is.null(keep_vars_variance)) && (!is.null(drop_vars_variance))) {
    if (is.character(drop_vars_variance)) {
      if (!all(drop_vars_variance %in% names(X_train))) {
        stop("drop_vars_variance includes some variable names that are not in X_train")
      }
      variable_subset_variance <- unname(which(!(names(X_train) %in% drop_vars_variance)))
    } else {
      if (any(drop_vars_variance > ncol(X_train))) {
        stop("drop_vars_variance includes some variable indices that exceed the number of columns in X_train")
      }
      if (any(drop_vars_variance < 0)) {
        stop("drop_vars_variance includes some negative variable indices")
      }
      variable_subset_variance <- (1:ncol(X_train))[!(1:ncol(X_train) %in% drop_vars_variance)]
    }
  } else {
    variable_subset_variance <- 1:ncol(X_train)
  }
  
  # Preprocess covariates
  if (ncol(X_train) != length(variable_weights)) {
    stop("length(variable_weights) must equal ncol(X_train)")
  }
  train_cov_preprocess_list <- preprocessTrainData(X_train)
  X_train_metadata <- train_cov_preprocess_list$metadata
  X_train_raw <- X_train
  X_train <- train_cov_preprocess_list$data
  original_var_indices <- X_train_metadata$original_var_indices
  feature_types <- X_train_metadata$feature_types
  X_test_raw <- X_test
  if (!is.null(X_test)) X_test <- preprocessPredictionData(X_test, X_train_metadata)
  
  # Convert all input data to matrices if not already converted
  if ((is.null(dim(Z_train))) && (!is.null(Z_train))) {
    Z_train <- as.matrix(as.numeric(Z_train))
  }
  if ((is.null(dim(propensity_train))) && (!is.null(propensity_train))) {
    propensity_train <- as.matrix(propensity_train)
  }
  if ((is.null(dim(Z_test))) && (!is.null(Z_test))) {
    Z_test <- as.matrix(as.numeric(Z_test))
  }
  if ((is.null(dim(propensity_test))) && (!is.null(propensity_test))) {
    propensity_test <- as.matrix(propensity_test)
  }
  if ((is.null(dim(rfx_basis_train))) && (!is.null(rfx_basis_train))) {
    rfx_basis_train <- as.matrix(rfx_basis_train)
  }
  if ((is.null(dim(rfx_basis_test))) && (!is.null(rfx_basis_test))) {
    rfx_basis_test <- as.matrix(rfx_basis_test)
  }
  
  # Recode group IDs to integer vector (if passed as, for example, a vector of county names, etc...)
  has_rfx <- FALSE
  has_rfx_test <- FALSE
  if (!is.null(rfx_group_ids_train)) {
    group_ids_factor <- factor(rfx_group_ids_train)
    rfx_group_ids_train <- as.integer(group_ids_factor)
    has_rfx <- TRUE
    if (!is.null(rfx_group_ids_test)) {
      group_ids_factor_test <- factor(rfx_group_ids_test, levels = levels(group_ids_factor))
      if (sum(is.na(group_ids_factor_test)) > 0) {
        stop("All random effect group labels provided in rfx_group_ids_test must be present in rfx_group_ids_train")
      }
      rfx_group_ids_test <- as.integer(group_ids_factor_test)
      has_rfx_test <- TRUE
    }
  }
  # if((!global_shrinkage) & (gibbs)){
  #   stop("You can not turn on gibss sampling without global shrinkage for now!")
  # }
  # Check that outcome and treatment are numeric
  if (!is.numeric(y_train)) stop("y_train must be numeric")
  if (!is.numeric(Z_train)) stop("Z_train must be numeric")
  if (!is.null(Z_test)) {
    if (!is.numeric(Z_test)) stop("Z_test must be numeric")
  }
  
  # Data consistency checks
  if ((!is.null(X_test)) && (ncol(X_test) != ncol(X_train))) {
    stop("X_train and X_test must have the same number of columns")
  }
  if ((!is.null(Z_test)) && (ncol(Z_test) != ncol(Z_train))) {
    stop("Z_train and Z_test must have the same number of columns")
  }
  if ((!is.null(Z_train)) && (nrow(Z_train) != nrow(X_train))) {
    stop("Z_train and X_train must have the same number of rows")
  }
  if ((!is.null(propensity_train)) && (nrow(propensity_train) != nrow(X_train))) {
    stop("propensity_train and X_train must have the same number of rows")
  }
  if ((!is.null(Z_test)) && (nrow(Z_test) != nrow(X_test))) {
    stop("Z_test and X_test must have the same number of rows")
  }
  if ((!is.null(propensity_test)) && (nrow(propensity_test) != nrow(X_test))) {
    stop("propensity_test and X_test must have the same number of rows")
  }
  if (nrow(X_train) != length(y_train)) {
    stop("X_train and y_train must have the same number of observations")
  }
  if ((!is.null(rfx_basis_test)) && (ncol(rfx_basis_test) != ncol(rfx_basis_train))) {
    stop("rfx_basis_train and rfx_basis_test must have the same number of columns")
  }
  if (!is.null(rfx_group_ids_train)) {
    if (!is.null(rfx_group_ids_test)) {
      if ((!is.null(rfx_basis_train)) && (is.null(rfx_basis_test))) {
        stop("rfx_basis_train is provided but rfx_basis_test is not provided")
      }
    }
  }
  
  # Stop if multivariate treatment is provided
  if (ncol(Z_train) > 1) stop("Multivariate treatments are not currently supported")
  
  # Random effects covariance prior
  if (has_rfx) {
    if (is.null(rfx_prior_var)) {
      rfx_prior_var <- rep(1, ncol(rfx_basis_train))
    } else {
      if ((!is.integer(rfx_prior_var)) && (!is.numeric(rfx_prior_var))) stop("rfx_prior_var must be a numeric vector")
      if (length(rfx_prior_var) != ncol(rfx_basis_train)) stop("length(rfx_prior_var) must equal ncol(rfx_basis_train)")
    }
  }
  
  # Update variable weights
  variable_weights_adj <- 1/sapply(original_var_indices, function(x) sum(original_var_indices == x))
  variable_weights <- variable_weights[original_var_indices]*variable_weights_adj
  
  # Create mu and tau (and variance) specific variable weights with weights zeroed out for excluded variables
  variable_weights_variance <- variable_weights_tau <- variable_weights_mu <- variable_weights
  variable_weights_mu[!(original_var_indices %in% variable_subset_mu)] <- 0
  variable_weights_tau[!(original_var_indices %in% variable_subset_tau)] <- 0
  if (include_variance_forest) {
    variable_weights_variance[!(original_var_indices %in% variable_subset_variance)] <- 0
  }
  
  # Fill in rfx basis as a vector of 1s (random intercept) if a basis not provided 
  has_basis_rfx <- FALSE
  num_basis_rfx <- 0
  if (has_rfx) {
    if (is.null(rfx_basis_train)) {
      rfx_basis_train <- matrix(rep(1,nrow(X_train)), nrow = nrow(X_train), ncol = 1)
    } else {
      has_basis_rfx <- TRUE
      num_basis_rfx <- ncol(rfx_basis_train)
    }
    num_rfx_groups <- length(unique(rfx_group_ids_train))
    num_rfx_components <- ncol(rfx_basis_train)
    if (num_rfx_groups == 1) warning("Only one group was provided for random effect sampling, so the 'redundant parameterization' is likely overkill")
  }
  if (has_rfx_test) {
    if (is.null(rfx_basis_test)) {
      if (!is.null(rfx_basis_train)) {
        stop("Random effects basis provided for training set, must also be provided for the test set")
      }
      rfx_basis_test <- matrix(rep(1,nrow(X_test)), nrow = nrow(X_test), ncol = 1)
    }
  }
  
  # Check that number of samples are all nonnegative
  stopifnot(num_gfr >= 0)
  stopifnot(num_burnin >= 0)
  stopifnot(num_mcmc >= 0)
  
  # Determine whether a test set is provided
  has_test = !is.null(X_test)
  
  # Convert y_train to numeric vector if not already converted
  if (!is.null(dim(y_train))) {
    y_train <- as.matrix(y_train)
  }
  
  # Check whether treatment is binary (specifically 0-1 binary)
  binary_treatment <- length(unique(Z_train)) == 2
  if (binary_treatment) {
    unique_treatments <- sort(unique(Z_train))
    if (!(all(unique_treatments == c(0,1)))) binary_treatment <- FALSE
  }
  
  # Adaptive coding will be ignored for continuous / ordered categorical treatments
  if ((!binary_treatment) && (adaptive_coding)) {
    adaptive_coding <- FALSE
  }
  
  # Check if propensity_covariate is one of the required inputs
  if (!(propensity_covariate %in% c("mu","tau","both","none"))) {
    stop("propensity_covariate must equal one of 'none', 'mu', 'tau', or 'both'")
  }
  
  # Estimate if pre-estimated propensity score is not provided
  internal_propensity_model <- FALSE
  if ((is.null(propensity_train)) && (propensity_covariate != "none")) {
    internal_propensity_model <- TRUE
    # Estimate using the last of several iterations of GFR BART
    num_burnin <- 10
    num_total <- 50
    bart_model_propensity <- bart(X_train = X_train, y_train = as.numeric(Z_train), X_test = X_test_raw, 
                                  num_gfr = num_total, num_burnin = 0, num_mcmc = 0)
    propensity_train <- rowMeans(bart_model_propensity$y_hat_train[,(num_burnin+1):num_total])
    if ((is.null(dim(propensity_train))) && (!is.null(propensity_train))) {
      propensity_train <- as.matrix(propensity_train)
    }
    if (has_test) {
      propensity_test <- rowMeans(bart_model_propensity$y_hat_test[,(num_burnin+1):num_total])
      if ((is.null(dim(propensity_test))) && (!is.null(propensity_test))) {
        propensity_test <- as.matrix(propensity_test)
      }
    }
  }
  
  if (has_test) {
    if (is.null(propensity_test)) stop("Propensity score must be provided for the test set if provided for the training set")
  }
  
  # Update feature_types and covariates
  feature_types <- as.integer(feature_types)
  if ((propensity_covariate != "none") & (propensity_seperate != "mu") ) {
    feature_types <- as.integer(c(feature_types,rep(0, ncol(propensity_train))))
    X_train <- cbind(X_train, propensity_train)
    if (propensity_covariate == "mu") {
      variable_weights_mu <- c(variable_weights_mu, rep(1./num_cov_orig, ncol(propensity_train)))
      variable_weights_tau <- c(variable_weights_tau, rep(0, ncol(propensity_train)))
      if (include_variance_forest) variable_weights_variance <- c(variable_weights_variance, rep(0, ncol(propensity_train)))
    } else if (propensity_covariate == "tau") {
      variable_weights_mu <- c(variable_weights_mu, rep(0, ncol(propensity_train)))
      variable_weights_tau <- c(variable_weights_tau, rep(1./num_cov_orig, ncol(propensity_train)))
      if (include_variance_forest) variable_weights_variance <- c(variable_weights_variance, rep(0, ncol(propensity_train)))
    } else if (propensity_covariate == "both") {
      variable_weights_mu <- c(variable_weights_mu, rep(1./num_cov_orig, ncol(propensity_train)))
      variable_weights_tau <- c(variable_weights_tau, rep(1./num_cov_orig, ncol(propensity_train)))
      if (include_variance_forest) variable_weights_variance <- c(variable_weights_variance, rep(0, ncol(propensity_train)))
    }
    if (has_test) X_test <- cbind(X_test, propensity_test)
  }
  
  p_prog <- 0
  gamma_shap <- numeric(0)
  tau_gamma <- numeric(0)
  nu_gamma <- numeric(0)
  gamma_shap_samples <- NULL
  tau_gamma_samples <- NULL
  if (shapley) {
    p_prog <- ncol(X_train) # One Shapley value per feature including the propensity value
    gamma_shap_samples <- array(0, dim = c(num_chains, num_mcmc, p_prog))
    tau_gamma_samples <- array(0, dim = c(num_chains, num_mcmc, p_prog))
    gamma_shap <- rep(0, p_prog)
    tau_gamma <- rep(1, p_prog)
    nu_gamma <- 1 / (tau_gamma^2)
    Phi_train <- matrix(0, nrow=n, ncol=p_prog)
    gamma_names <- colnames(X_train)
    dimnames(gamma_shap_samples) <- list(NULL, NULL, Feature = gamma_names)
  } else {
    Phi_train <- matrix(0, nrow=n, ncol=1)
  }
  # Renormalize variable weights
  variable_weights_mu <- variable_weights_mu / sum(variable_weights_mu)
  variable_weights_tau <- variable_weights_tau / sum(variable_weights_tau)
  if (include_variance_forest) {
    variable_weights_variance <- variable_weights_variance / sum(variable_weights_variance)
  }
  
  # Preliminary runtime checks for probit link
  if (probit_outcome_model) {
    if (!(length(unique(y_train)) == 2)) {
      stop("You specified a probit outcome model, but supplied an outcome with more than 2 unique values")
    }
    unique_outcomes <- sort(unique(y_train))
    if (!(all(unique_outcomes == c(0,1)))) {
      stop("You specified a probit outcome model, but supplied an outcome with 2 unique values other than 0 and 1")
    }
    if (include_variance_forest) {
      stop("We do not support heteroskedasticity with a probit link")
    }
    if (sample_sigma2_global) {
      warning("Global error variance will not be sampled with a probit link as it is fixed at 1")
      sample_sigma2_global <- F
    }
  }
  
  # Handle standardization, prior calibration, and initialization of forest
  # differently for binary and continuous outcomes
  if (probit_outcome_model) {
    # Compute a probit-scale offset and fix scale to 1
    y_bar_train <- qnorm(mean(y_train))
    y_std_train <- 1
    
    # Set a pseudo outcome by subtracting mean(y_train) from y_train
    resid_train <- y_train - mean(y_train)
    
    # Set initial value for the mu forest
    init_mu <- 0.0
    
    # Calibrate priors for global sigma^2 and sigma2_leaf_mu / sigma2_leaf_tau
    # Set sigma2_init to 1, ignoring any defaults provided
    sigma2_init <- 1.0
    # Skip variance_forest_init, since variance forests are not supported with probit link
    if (is.null(b_leaf_mu)) b_leaf_mu <- 1/num_trees_mu
    if (is.null(b_leaf_tau)) b_leaf_tau <- 1/(2*num_trees_tau)
    if (is.null(sigma2_leaf_mu)) {
      sigma2_leaf_mu <- 2/(num_trees_mu)
      current_leaf_scale_mu <- as.matrix(sigma2_leaf_mu)
    } else {
      if (!is.matrix(sigma2_leaf_mu)) {
        current_leaf_scale_mu <- as.matrix(sigma2_leaf_mu)
      } else {
        current_leaf_scale_mu <- sigma2_leaf_mu
      }
    }
    if (is.null(sigma2_leaf_tau)) {
      # Calibrate prior so that P(abs(tau(X)) < delta_max / dnorm(0)) = p
      # Use p = 0.9 as an internal default rather than adding another 
      # user-facing "parameter" of the binary outcome BCF prior. 
      # Can be overriden by specifying `sigma2_leaf_init` in 
      # treatment_effect_forest_params.
      p <- 0.6827
      q_quantile <- qnorm((p+1)/2)
      sigma2_leaf_tau <- ((delta_max/(q_quantile*dnorm(0)))^2)/num_trees_tau
      current_leaf_scale_tau <- as.matrix(diag(sigma2_leaf_tau, ncol(Z_train)))
    } else {
      if (!is.matrix(sigma2_leaf_tau)) {
        current_leaf_scale_tau <- as.matrix(diag(sigma2_leaf_tau, ncol(Z_train)))
      } else {
        if (ncol(sigma2_leaf_tau) != ncol(Z_train)) stop("sigma2_leaf_init for the tau forest must have the same number of columns / rows as columns in the Z_train matrix")
        if (nrow(sigma2_leaf_tau) != ncol(Z_train)) stop("sigma2_leaf_init for the tau forest must have the same number of columns / rows as columns in the Z_train matrix")
        current_leaf_scale_tau <- sigma2_leaf_tau
      }
    }
    current_sigma2 <- sigma2_init
  } else {
    # Only standardize if user requested
    if (standardize) {
      y_bar_train <- mean(y_train)
      y_std_train <- sd(y_train)
    } else {
      y_bar_train <- 0
      y_std_train <- 1
    }
    
    # Compute standardized outcome
    resid_train <- (y_train-y_bar_train)/y_std_train
    y_train_shapley <- resid_train
    # Set initial value for the mu forest
    init_mu <- mean(resid_train)
    
    # Calibrate priors for global sigma^2 and sigma2_leaf_mu / sigma2_leaf_tau
    if (is.null(sigma2_init)) sigma2_init <- 1.0*var(resid_train)
    if (is.null(variance_forest_init)) variance_forest_init <- 1.0*var(resid_train)
    if (is.null(b_leaf_mu)) b_leaf_mu <- var(resid_train)/(num_trees_mu)
    if (is.null(b_leaf_tau)) b_leaf_tau <- var(resid_train)/(2*num_trees_tau)
    if (is.null(sigma2_leaf_mu)) {
      sigma2_leaf_mu <- 2.0*var(resid_train)/(num_trees_mu)
      current_leaf_scale_mu <- as.matrix(sigma2_leaf_mu)
    } else {
      if (!is.matrix(sigma2_leaf_mu)) {
        current_leaf_scale_mu <- as.matrix(sigma2_leaf_mu)
      } else {
        current_leaf_scale_mu <- sigma2_leaf_mu
      }
    }
    if (is.null(sigma2_leaf_tau)) {
      sigma2_leaf_tau <- var(resid_train)/(num_trees_tau)
      current_leaf_scale_tau <- as.matrix(diag(sigma2_leaf_tau, ncol(Z_train)))
    } else {
      if (!is.matrix(sigma2_leaf_tau)) {
        current_leaf_scale_tau <- as.matrix(diag(sigma2_leaf_tau, ncol(Z_train)))
      } else {
        if (ncol(sigma2_leaf_tau) != ncol(Z_train)) stop("sigma2_leaf_init for the tau forest must have the same number of columns / rows as columns in the Z_train matrix")
        if (nrow(sigma2_leaf_tau) != ncol(Z_train)) stop("sigma2_leaf_init for the tau forest must have the same number of columns / rows as columns in the Z_train matrix")
        current_leaf_scale_tau <- sigma2_leaf_tau
      }
    }
    current_sigma2 <- sigma2_init
  }
  
  # Switch off leaf scale sampling for multivariate treatments
  if (ncol(Z_train) > 1) {
    if (sample_sigma2_leaf_tau) {
      warning("Sampling leaf scale not yet supported for multivariate leaf models, so the leaf scale parameter will not be sampled for the treatment forest in this model.")
      sample_sigma2_leaf_tau <- FALSE
    }
  }
  
  # Set mu and tau leaf models / dimensions
  leaf_model_mu_forest <- 0
  leaf_dimension_mu_forest <- 1
  if (ncol(Z_train) > 1) {
    leaf_model_tau_forest <- 2
    leaf_dimension_tau_forest <- ncol(Z_train)
  } else {
    leaf_model_tau_forest <- 1
    leaf_dimension_tau_forest <- 1
  }
  
  # Set variance leaf model type (currently only one option)
  leaf_model_variance_forest <- 3
  leaf_dimension_variance_forest <- 1
  
  # Random effects prior parameters
  if (has_rfx) {
    # Initialize the working parameter to 1
    if (num_rfx_components < 1) {
      stop("There must be at least 1 random effect component")
    }
    alpha_init <- rep(1,num_rfx_components)
    # Initialize each group parameter based on a regression of outcome on basis in that grou
    xi_init <- matrix(0,num_rfx_components,num_rfx_groups)
    for (i in 1:num_rfx_groups) {
      group_subset_indices <- rfx_group_ids_train == i
      basis_group <- rfx_basis_train[group_subset_indices,]
      resid_group <- resid_train[group_subset_indices]
      rfx_group_model <- lm(resid_group ~ 0+basis_group)
      xi_init[,i] <- unname(coef(rfx_group_model))
    }
    sigma_alpha_init <- diag(1,num_rfx_components,num_rfx_components)
    sigma_xi_init <- diag(rfx_prior_var)
    sigma_xi_shape <- 1
    sigma_xi_scale <- 1
  }
  
  # Random effects data structure and storage container
  if (has_rfx) {
    rfx_dataset_train <- createRandomEffectsDataset(rfx_group_ids_train, rfx_basis_train)
    rfx_tracker_train <- createRandomEffectsTracker(rfx_group_ids_train)
    rfx_model <- createRandomEffectsModel(num_rfx_components, num_rfx_groups)
    rfx_model$set_working_parameter(alpha_init)
    rfx_model$set_group_parameters(xi_init)
    rfx_model$set_working_parameter_cov(sigma_alpha_init)
    rfx_model$set_group_parameter_cov(sigma_xi_init)
    rfx_model$set_variance_prior_shape(sigma_xi_shape)
    rfx_model$set_variance_prior_scale(sigma_xi_scale)
    rfx_samples <- createRandomEffectSamples(num_rfx_components, num_rfx_groups, rfx_tracker_train)
  }
  
  # Container of variance parameter samples
  num_actual_mcmc_iter <- num_mcmc * keep_every
  num_samples <- num_gfr + num_burnin + num_actual_mcmc_iter
  # Delete GFR samples from these containers after the fact if desired
  # num_retained_samples <- ifelse(keep_gfr, num_gfr, 0) + ifelse(keep_burnin, num_burnin, 0) + num_mcmc
  num_retained_samples <- num_gfr + ifelse(keep_burnin, num_burnin, 0) + num_mcmc * num_chains
  if (sample_sigma2_global) global_var_samples <- rep(NA, num_retained_samples)
  if (sample_sigma2_leaf_mu) leaf_scale_mu_samples <- rep(NA, num_retained_samples)
  if (sample_sigma2_leaf_tau) leaf_scale_tau_samples <- rep(NA, num_retained_samples)
  sample_counter <- 0
  Phi_train <- matrix(0, nrow=n, ncol=1) # Default dummy
  # Prepare adaptive coding structure
  if ((!is.numeric(b_0)) || (!is.numeric(b_1)) || (length(b_0) > 1) || (length(b_1) > 1)) {
    stop("b_0 and b_1 must be single numeric values")
  }
  if (adaptive_coding) {
    b_0_samples <- rep(NA, num_retained_samples)
    b_1_samples <- rep(NA, num_retained_samples)
    current_b_0 <- b_0
    current_b_1 <- b_1
    tau_basis_train <- (1-Z_train)*current_b_0 + Z_train*current_b_1
    if (has_test) tau_basis_test <- (1-Z_test)*current_b_0 + Z_test*current_b_1
  } else {
    tau_basis_train <- Z_train
    if (has_test) tau_basis_test <- Z_test
  }
  
  # Data
  forest_dataset_train <- createForestDataset(X_train, tau_basis_train)
  if (has_test) forest_dataset_test <- createForestDataset(X_test, tau_basis_test)
  outcome_train <- createOutcome(resid_train)
  
  # Random number generator (std::mt19937)
  if (is.null(random_seed)) random_seed = sample(1:10000,1,FALSE)
  rng <- createCppRNG(random_seed)
  
  # Sampling data structures
  global_model_config <- createGlobalModelConfig(global_error_variance=current_sigma2)
  forest_model_config_mu <- createForestModelConfig(feature_types=feature_types, num_trees=num_trees_mu, num_features=ncol(X_train), 
                                                    num_observations=nrow(X_train), variable_weights=variable_weights_mu, leaf_dimension=leaf_dimension_mu_forest, 
                                                    alpha=alpha_mu, beta=beta_mu, min_samples_leaf=min_samples_leaf_mu, max_depth=max_depth_mu, 
                                                    leaf_model_type=leaf_model_mu_forest, leaf_model_scale=current_leaf_scale_mu, 
                                                    cutpoint_grid_size=cutpoint_grid_size)
  # forest_model_config_tau <- createForestModelConfig(feature_types=feature_types, num_trees=num_trees_tau, num_features=ncol(X_train), 
  #                                                    num_observations=nrow(X_train), variable_weights=variable_weights_tau, leaf_dimension=leaf_dimension_tau_forest, 
  #                                                    alpha=alpha_tau, beta=beta_tau, min_samples_leaf=min_samples_leaf_tau, max_depth=max_depth_tau, 
  #                                                    leaf_model_type=leaf_model_tau_forest, leaf_model_scale=current_leaf_scale_tau, 
  #                                                    cutpoint_grid_size=cutpoint_grid_size)
  forest_model_mu <- createForestModel(forest_dataset_train, forest_model_config_mu, global_model_config)
  # forest_model_tau <- createForestModel(forest_dataset_train, forest_model_config_tau, global_model_config)
  if (include_variance_forest) {
    forest_model_config_variance <- createForestModelConfig(feature_types=feature_types, num_trees=num_trees_variance, num_features=ncol(X_train), 
                                                            num_observations=nrow(X_train), variable_weights=variable_weights_variance, 
                                                            leaf_dimension=leaf_dimension_variance_forest, alpha=alpha_variance, beta=beta_variance, 
                                                            min_samples_leaf=min_samples_leaf_variance, max_depth=max_depth_variance, 
                                                            leaf_model_type=leaf_model_variance_forest, cutpoint_grid_size=cutpoint_grid_size)
    forest_model_variance <- createForestModel(forest_dataset_train, forest_model_config_variance, global_model_config)
  }
  
  # Container of forest samples
  forest_samples_mu <- createForestSamples(num_trees_mu, 1, TRUE)
  forest_samples_tau <- createForestSamples(num_trees_tau, 1, FALSE)
  active_forest_mu <- createForest(num_trees_mu, 1, TRUE)
  # active_forest_tau <- createForest(num_trees_tau, 1, FALSE)
  if (include_variance_forest) {
    forest_samples_variance <- createForestSamples(num_trees_variance, 1, TRUE, TRUE)
    active_forest_variance <- createForest(num_trees_variance, 1, TRUE, TRUE)
  }
  
  # Initialize the leaves of each tree in the prognostic forest
  active_forest_mu$prepare_for_sampler(forest_dataset_train, outcome_train, forest_model_mu, 0, init_mu)
  active_forest_mu$adjust_residual(forest_dataset_train, outcome_train, forest_model_mu, FALSE, FALSE)
  
  # Initialize the leaves of each tree in the treatment effect forest
  init_tau <- 0.
  # active_forest_tau$prepare_for_sampler(forest_dataset_train, outcome_train, forest_model_tau, 1, init_tau)
  # active_forest_tau$adjust_residual(forest_dataset_train, outcome_train, forest_model_tau, TRUE, FALSE)
  
  # Initialize the leaves of each tree in the variance forest
  if (include_variance_forest) {
    active_forest_variance$prepare_for_sampler(forest_dataset_train, outcome_train, forest_model_variance, leaf_model_variance_forest, variance_forest_init)
  }
  if (is.null(propensity_train)) {
    propensity_train_for_cpp <- numeric(0)
  } else {
    propensity_train_for_cpp <- propensity_train
  }
  
  
  # Run GFR (warm start) if specified
  if (num_gfr > 0){
    for (i in 1:num_gfr) {
      # Keep all GFR samples at this stage -- remove from ForestSamples after MCMC
      # keep_sample <- ifelse(keep_gfr, TRUE, FALSE)
      keep_sample <- TRUE
      if (keep_sample) sample_counter <- sample_counter + 1
      # Print progress
      if (verbose) {
        if ((i %% 1 == 0) || (i == num_gfr)) {
          cat("Sampling", i, "out of", num_gfr, "XBCF (grow-from-root) draws\n")
        }
      }
      # 1. Update Treatment Coding (if adaptive)
      if(adaptive_coding){
        Z_linear <- tau_basis_train
      } else {
        Z_linear <- Z_train
      }
      
      # 2. Update Prognostic Forest (Mu) - GFR Mode
      # Note: gfr = TRUE usually enables aggressive growth or specific initialization logic
      forest_model_mu$sample_one_iteration(
        forest_dataset = forest_dataset_train, residual = outcome_train, forest_samples = forest_samples_mu, 
        active_forest = active_forest_mu, rng = rng, forest_model_config = forest_model_config_mu, 
        global_model_config = global_model_config, keep_forest = keep_sample, gfr = TRUE
      )
      

      if (sample_sigma2_leaf_mu) {
        leaf_scale_mu_double <- sampleLeafVarianceOneIteration(active_forest_mu, rng, a_leaf_mu, b_leaf_mu)
        current_leaf_scale_mu <- as.matrix(leaf_scale_mu_double)
        forest_model_config_mu$update_leaf_model_scale(current_leaf_scale_mu)
        if (keep_sample) leaf_scale_mu_samples[sample_counter] <- leaf_scale_mu_double   # <<< ADD THIS
      }
      
      # ----------------------------------------------------------------------
      # START: SHAPLEY LOGIC (GFR PHASE)
      # ----------------------------------------------------------------------
      
      
      # ----------------------------------------------------------------------
      # 1. SHAPLEY REPRESENTATION LEARNING
      # ----------------------------------------------------------------------
      if (shapley) {
        # A. Extract structure from the tree (Mu)
        Phi_raw <- stochtree:::get_saabas_shapley_active_forest_cpp(
          active_forest_mu$forest_ptr, 
          forest_dataset_train$data_ptr
        )
        
        # B. Handle Edge Cases
        Phi_raw[is.na(Phi_raw)] <- 0
        
        # C. Scale (Representation Learning needs standardized features)
        Phi_train <- apply(Phi_raw, 2, function(x) {
          sigma <- sd(x)
          if (is.na(sigma) || sigma < 1e-9) return(rep(0, length(x))) 
          else return((x - mean(x)) / sigma)
        })
        Phi_train <- as.matrix(Phi_train)
      }
      
      # Main effects
      current_linear_pred <- as.matrix(X_train_raw) %*% beta 
      
      # Add interactions if they exist
      if (p_int > 0) {
        interaction_cols <- full_design_matrix_train[, (ncol(X_train_raw)+2):ncol(full_design_matrix_train)]
        current_linear_pred <- current_linear_pred + as.matrix(interaction_cols) %*% beta_int
      }
      
      # Add Shapley effects
      if (shapley) {
        current_linear_pred <- current_linear_pred + Phi_train %*% gamma
      }
      
      # Add Intercept
      if (regularize_ATE) {
        current_linear_pred <- current_linear_pred + Z_linear * alpha
      } else {
        current_linear_pred <- current_linear_pred + Z_linear * alpha
      }
      
      # This residual represents "What the linear model currently fails to explain"
      tau_residual <- resid_train - current_linear_pred
      
      # Add noise for the sampler stability if desired, though standard Gibbs 
      # usually takes the pure residual.
      if (!shapley && sigma_residual > 0) {
        tau_residual <- tau_residual + rnorm(n, 0, sigma_residual)
      }
      # 3. Linear Update (Beta, Alpha, Gamma)
      if(simple_prior){ sigma2_lin <- 1 } else { sigma2_lin <- current_sigma2 }
      if(is.null(propensity_train)){ propensity_train <- numeric(0) }
      
      
      if(use_ncp == TRUE){
        update_results <- updateLinearTreatmentCpp_NCP_cpp(
          X = if (propensity_seperate == "tau") X_train else X_train_raw,
          Phi = Phi_train,
          Z = Z_linear,
          propensity_train = propensity_train, 
          residual = tau_residual,
          are_continuous = as.vector(as.integer(boolean_continuous*1)),
          alpha_tilde = alpha, 
          gamma_prop = gamma_prop, 
          beta_tilde = beta,
          beta_int_tilde = beta_int,
          gamma_tilde = gamma_shap,
          tau_beta = tau_beta,
          tau_gamma = tau_gamma,
          nu = nu,
          nu_gamma = nu_gamma,
          xi = xi,
          tau_int = 1.0,
          sigma = sqrt(sigma2_lin),
          alpha_prior_sd = 10.0,
          tau_glob = tau_glob,
          sample_global_prior = sample_global_prior,
          unlink = unlink,
          gibbs = gibbs, 
          regularize_ATE = regularize_ATE,
          hn_scale = hn_scale,
          use_prognostic_shapley = shapley
        )
      } else {
        update_results <- updateLinearTreatmentCpp_cpp(
          X = if (propensity_seperate == "tau") X_train else X_train_raw,
          Phi = Phi_train,
          Z = Z_linear,
          propensity_train = propensity_train,
          residual = tau_residual,
          are_continuous = as.vector(as.integer(boolean_continuous*1)),
          alpha = alpha,
          gamma_prop = gamma_prop,
          beta = beta,
          beta_int = beta_int,
          gamma = gamma_shap,
          tau_beta = tau_beta,
          tau_gamma = tau_gamma,
          nu = nu,
          nu_gamma = nu_gamma,
          xi = xi,
          tau_int = tau_int,
          sigma = sqrt(sigma2_lin),
          alpha_prior_sd = 10.0,
          tau_glob = tau_glob,
          sample_global_prior = sample_global_prior,
          unlink = unlink,
          gibbs = gibbs,
          save_output = save_output,
          index = sample_counter,
          max_steps = max_steps,
          step_out = step_out,
          propensity_seperate = propensity_seperate,
          regularize_ATE = regularize_ATE,
          hn_scale = hn_scale,
          use_prognostic_shapley = shapley
        )
      }
      
      # --- UNPACKING ---
      # 1. Extract Vectors
      params_vec <- update_results$params
      
      # 2. Extract Values
      idx <- 1
      alpha <- params_vec[idx]; idx <- idx + 1
      tau_int <- params_vec[idx]; idx <- idx + 1
      tau_glob <- params_vec[idx]; idx <- idx + 1
      gamma_prop <- params_vec[idx]; idx <- idx + 1
      xi <- params_vec[idx]; idx <- idx + 1
      
      beta_end <- idx + p_mod - 1
      beta <- params_vec[idx:beta_end]
      idx <- beta_end + 1
      
      beta_int_end <- idx + p_int - 1
      beta_int <- params_vec[idx:beta_int_end]
      idx <- beta_int_end + 1
      
      if (shapley) {
        gamma_shap_end <- idx + p_prog - 1
        gamma_shap <- params_vec[idx:gamma_shap_end]
        idx <- gamma_shap_end + 1
      }
      
      tau_beta_len <- length(tau_beta)
      tau_beta_end <- idx + tau_beta_len - 1
      tau_beta <- params_vec[idx:tau_beta_end]
      idx <- tau_beta_end + 1
      
      if (shapley) {
        tau_gamma_len <- length(tau_gamma)
        tau_gamma_end <- idx + tau_gamma_len - 1
        tau_gamma <- params_vec[idx:tau_gamma_end]
        idx <- tau_gamma_end + 1
      }
      
      nu_len <- length(nu)
      nu_end <- idx + nu_len - 1
      nu <- params_vec[idx:nu_end]
      idx <- nu_end + 1
      
      if (shapley) {
        nu_gamma_len <- length(nu_gamma)
        nu_gamma_end <- idx + nu_gamma_len - 1
        nu_gamma <- params_vec[idx:nu_gamma_end]
        idx <- nu_gamma_end + 1
      }
      
      if(shapley){
      current_linear_pred <- as.matrix(X_train_raw) %*% beta 
      
      # Add interactions if they exist
        if (p_int > 0) {
        interaction_cols <- full_design_matrix_train[, (ncol(X_train_raw)+2):ncol(full_design_matrix_train)]
        current_linear_pred <- current_linear_pred + as.matrix(interaction_cols) %*% beta_int
        }
        current_linear_pred <- current_linear_pred + Z_linear * alpha
      
      
        residual <- resid_train - current_linear_pred
        outcome_train$update_data(matrix(residual, ncol=1))
        
      } else {
        residual <- update_results$residuals
        outcome_train$update_data(matrix(residual, ncol=1))
      }
      
      # Sample coding parameters (if requested)
      if (adaptive_coding) {
        # Estimate mu(X) and tau(X) and compute y - mu(X)
        mu_x_raw_train <- active_forest_mu$predict_raw(forest_dataset_train)
        tau_x_raw_train <- predict_interaction_lm(X_train_raw, c(alpha, beta, beta_int), X_final_var_info)
        partial_resid_mu_train <- resid_train - mu_x_raw_train
        if (has_rfx) {
          rfx_preds_train <- rfx_model$predict(rfx_dataset_train, rfx_tracker_train)
          partial_resid_mu_train <- partial_resid_mu_train - rfx_preds_train
        }
        
        # Compute sufficient statistics for regression of y - mu(X) on [tau(X)(1-Z), tau(X)Z]
        s_tt0 <- sum(tau_x_raw_train*tau_x_raw_train*(Z_train==0))
        s_tt1 <- sum(tau_x_raw_train*tau_x_raw_train*(Z_train==1))
        s_ty0 <- sum(tau_x_raw_train*partial_resid_mu_train*(Z_train==0))
        s_ty1 <- sum(tau_x_raw_train*partial_resid_mu_train*(Z_train==1))
        
        # Sample b0 (coefficient on tau(X)(1-Z)) and b1 (coefficient on tau(X)Z)
        current_b_0 <- rnorm(1, (s_ty0/(s_tt0 + 2*current_sigma2)), sqrt(current_sigma2/(s_tt0 + 2*current_sigma2)))
        current_b_1 <- rnorm(1, (s_ty1/(s_tt1 + 2*current_sigma2)), sqrt(current_sigma2/(s_tt1 + 2*current_sigma2)))
        
        # Update basis for the leaf regression
        tau_basis_train <- (1-Z_train)*current_b_0 + Z_train*current_b_1
        forest_dataset_train$update_basis(tau_basis_train)
        if (keep_sample) {
          b_0_samples[sample_counter] <- current_b_0
          b_1_samples[sample_counter] <- current_b_1
        }
        if (has_test) {
          tau_basis_test <- (1-Z_test)*current_b_0 + Z_test*current_b_1
          # forest_dataset_test$update_basis(tau_basis_test)
        }
      }
      
      # Sample variance parameters (if requested)
      if (include_variance_forest) {
        forest_model_variance$sample_one_iteration(
          forest_dataset = forest_dataset_train, residual = outcome_train, forest_samples = forest_samples_variance, 
          active_forest = active_forest_variance, rng = rng, forest_model_config = forest_model_config_variance, 
          global_model_config = global_model_config, keep_forest = keep_sample, gfr = TRUE
        )
      }
      if (sample_sigma2_global) {
        shape_sigma_post <- 0.001 + n / 2
        rate_sigma_post <- 0.001 + 0.5 * sum(outcome_train$get_data()^2)
        sigma_sq <- rinvgamma(shape = shape_sigma_post, scale = rate_sigma_post)
        current_sigma2 <- max(sigma_sq, 1e-9)
        global_model_config$update_global_error_variance(current_sigma2)
        if (keep_sample) global_var_samples[sample_counter] <- current_sigma2
        # current_sigma2 <- sampleGlobalErrorVarianceOneIteration(outcome_train, forest_dataset_train, rng, a_global, b_global)
        # if (keep_sample) global_var_samples[sample_counter] <- current_sigma2
        # global_model_config$update_global_error_variance(current_sigma2)
      } ##SAMPLED
      # if (sample_sigma2_leaf_tau) {
      #   leaf_scale_tau_double <- sampleLeafVarianceOneIteration(active_forest_tau, rng, a_leaf_tau, b_leaf_tau)
      #   current_leaf_scale_tau <- as.matrix(leaf_scale_tau_double)
      #   if (keep_sample) leaf_scale_tau_samples[sample_counter] <- leaf_scale_tau_double
      #   forest_model_config_mu$update_leaf_model_scale(current_leaf_scale_mu)
      # }
      
      # Sample random effects parameters (if requested)
      if (has_rfx) {
        rfx_model$sample_random_effect(rfx_dataset_train, outcome_train, rfx_tracker_train, rfx_samples, keep_sample, current_sigma2, rng)
      }
    }
  }
  if (num_burnin + num_mcmc > 0) {
    for (chain_num in 1:num_chains) {
      linear_counter <- 0 
      if (num_gfr > 0) {
        # Reset state of active_forest and forest_model based on a previous GFR sample
        forest_ind <- num_gfr - chain_num
        resetActiveForest(active_forest_mu, forest_samples_mu, forest_ind)
        resetForestModel(forest_model_mu, active_forest_mu, forest_dataset_train, outcome_train, TRUE)
        # resetActiveForest(active_forest_tau, forest_samples_tau, forest_ind)
        # resetForestModel(forest_model_tau, active_forest_tau, forest_dataset_train, outcome_train, TRUE)
        if (sample_sigma2_leaf_mu) {
          leaf_scale_mu_double <- leaf_scale_mu_samples[forest_ind + 1]
          print(leaf_scale_mu_double)
          current_leaf_scale_mu <- as.matrix(leaf_scale_mu_double)
          print(current_leaf_scale_mu)
          
          forest_model_config_mu$update_leaf_model_scale(current_leaf_scale_mu)
        }
        # if (sample_sigma2_leaf_tau) {
        #   leaf_scale_tau_double <- leaf_scale_tau_samples[forest_ind + 1]
        #   current_leaf_scale_tau <- as.matrix(leaf_scale_tau_double)
        #   forest_model_config_tau$update_leaf_model_scale(current_leaf_scale_tau)
        # }
        if (include_variance_forest) {
          resetActiveForest(active_forest_variance, forest_samples_variance, forest_ind)
          resetForestModel(forest_model_variance, active_forest_variance, forest_dataset_train, outcome_train, FALSE)
        }
        if (has_rfx) {
          resetRandomEffectsModel(rfx_model, rfx_samples, forest_ind, sigma_alpha_init)
          resetRandomEffectsTracker(rfx_tracker_train, rfx_model, rfx_dataset_train, outcome_train, rfx_samples)
        }
        if (adaptive_coding) {
          current_b_1 <- b_1_samples[forest_ind + 1]
          current_b_0 <- b_0_samples[forest_ind + 1]
          tau_basis_train <- (1-Z_train)*current_b_0 + Z_train*current_b_1
          forest_dataset_train$update_basis(tau_basis_train)
          if (has_test) {
            tau_basis_test <- (1-Z_test)*current_b_0 + Z_test*current_b_1
            forest_dataset_test$update_basis(tau_basis_test)
          }
          # forest_model_tau$propagate_basis_update(forest_dataset_train, outcome_train, active_forest_tau)
        }
      } else if (has_prev_model) {
        resetActiveForest(active_forest_mu, previous_forest_samples_mu, previous_model_warmstart_sample_num - 1)
        resetForestModel(forest_model_mu, active_forest_mu, forest_dataset_train, outcome_train, TRUE)
        # resetActiveForest(active_forest_tau, previous_forest_samples_tau, previous_model_warmstart_sample_num - 1)
        # resetForestModel(forest_model_tau, active_forest_tau, forest_dataset_train, outcome_train, TRUE)
        if (include_variance_forest) {
          resetActiveForest(active_forest_variance, previous_forest_samples_variance, previous_model_warmstart_sample_num - 1)
          resetForestModel(forest_model_variance, active_forest_variance, forest_dataset_train, outcome_train, FALSE)
        }
        if (sample_sigma2_leaf_mu && (!is.null(previous_leaf_var_mu_samples))) {
          leaf_scale_mu_double <- previous_leaf_var_mu_samples[previous_model_warmstart_sample_num]
          current_leaf_scale_mu <- as.matrix(leaf_scale_mu_double)
          forest_model_config_mu$update_leaf_model_scale(current_leaf_scale_mu)
        }
        # if (sample_sigma2_leaf_tau && (!is.null(previous_leaf_var_tau_samples))) {
        #   leaf_scale_tau_double <- previous_leaf_var_tau_samples[previous_model_warmstart_sample_num]
        #   current_leaf_scale_tau <- as.matrix(leaf_scale_tau_double)
        #   forest_model_config_tau$update_leaf_model_scale(current_leaf_scale_tau)
        # }
        if (adaptive_coding) {
          if (!is.null(previous_b_1_samples)) {
            current_b_1 <- previous_b_1_samples[previous_model_warmstart_sample_num]
          }
          if (!is.null(previous_b_0_samples)) {
            current_b_0 <- previous_b_0_samples[previous_model_warmstart_sample_num]
          }
          tau_basis_train <- (1-Z_train)*current_b_0 + Z_train*current_b_1
          forest_dataset_train$update_basis(tau_basis_train)
          if (has_test) {
            tau_basis_test <- (1-Z_test)*current_b_0 + Z_test*current_b_1
            forest_dataset_test$update_basis(tau_basis_test)
          }
          # forest_model_tau$propagate_basis_update(forest_dataset_train, outcome_train, active_forest_tau)
        }
        if (has_rfx) {
          if (is.null(previous_rfx_samples)) {
            warning("`previous_model_json` did not have any random effects samples, so the RFX sampler will be run from scratch while the forests and any other parameters are warm started")
            rootResetRandomEffectsModel(rfx_model, alpha_init, xi_init, sigma_alpha_init,
                                        sigma_xi_init, sigma_xi_shape, sigma_xi_scale)
            rootResetRandomEffectsTracker(rfx_tracker_train, rfx_model, rfx_dataset_train, outcome_train)
          } else {
            resetRandomEffectsModel(rfx_model, previous_rfx_samples, previous_model_warmstart_sample_num - 1, sigma_alpha_init)
            resetRandomEffectsTracker(rfx_tracker_train, rfx_model, rfx_dataset_train, outcome_train, rfx_samples)
          }
        }
      } else {
        resetActiveForest(active_forest_mu)
        active_forest_mu$set_root_leaves(init_mu / num_trees_mu)
        resetForestModel(forest_model_mu, active_forest_mu, forest_dataset_train, outcome_train, TRUE)
        # resetActiveForest(active_forest_tau)
        # active_forest_tau$set_root_leaves(init_tau / num_trees_tau)
        # resetForestModel(forest_model_tau, active_forest_tau, forest_dataset_train, outcome_train, TRUE)
        if (sample_sigma2_leaf_mu) {
          current_leaf_scale_mu <- as.matrix(sigma2_leaf_mu)
          forest_model_config_mu$update_leaf_model_scale(current_leaf_scale_mu)
        }
        if (sample_sigma2_leaf_tau) {
          current_leaf_scale_tau <- as.matrix(sigma2_leaf_tau)
          forest_model_config_tau$update_leaf_model_scale(current_leaf_scale_tau)
        }
        if (include_variance_forest) {
          resetActiveForest(active_forest_variance)
          active_forest_variance$set_root_leaves(log(variance_forest_init) / num_trees_variance)
          resetForestModel(forest_model_variance, active_forest_variance, forest_dataset_train, outcome_train, FALSE)
        }
        if (has_rfx) {
          rootResetRandomEffectsModel(rfx_model, alpha_init, xi_init, sigma_alpha_init,
                                      sigma_xi_init, sigma_xi_shape, sigma_xi_scale)
          rootResetRandomEffectsTracker(rfx_tracker_train, rfx_model, rfx_dataset_train, outcome_train)
        }
        if (adaptive_coding) {
          current_b_1 <- b_1
          current_b_0 <- b_0
          tau_basis_train <- (1-Z_train)*current_b_0 + Z_train*current_b_1
          forest_dataset_train$update_basis(tau_basis_train)
          if (has_test) {
            tau_basis_test <- (1-Z_test)*current_b_0 + Z_test*current_b_1
            forest_dataset_test$update_basis(tau_basis_test)
          }
          forest_model_tau$propagate_basis_update(forest_dataset_train, outcome_train, active_forest_tau)
        }
        if (sample_sigma2_global) {
          current_sigma2 <- sigma2_init
          global_model_config$update_global_error_variance(current_sigma2)
        }
      }
      for (i in (num_gfr+1):num_samples) {
        is_mcmc <- i > (num_gfr + num_burnin)
        if (is_mcmc) {
          mcmc_counter <- i - (num_gfr + num_burnin)
          if (mcmc_counter %% keep_every == 0) keep_sample <- TRUE
          else keep_sample <- FALSE
        } else {
          if (keep_burnin) keep_sample <- TRUE
          else keep_sample <- FALSE
        }
        if (keep_sample) sample_counter <- sample_counter + 1
        # Print progress
        if (verbose) {
          if (num_burnin > 0) {
            if (((i - num_gfr) %% 100 == 0) || ((i - num_gfr) == num_burnin)) {
              cat("Sampling", i - num_gfr, "out of", num_gfr, "BCF burn-in draws\n")
            }
          }
          if (num_mcmc > 0) {
            if (((i - num_gfr - num_burnin) %% 100 == 0) || (i == num_samples)) {
              cat("Sampling", i - num_burnin - num_gfr, "out of", num_mcmc, "BCF MCMC draws\n")
            }
          }
        }
        # Sample the treatment forest
        # Sample the treatment forest
        if(adaptive_coding){
          Z_linear <- tau_basis_train
        } else {
          Z_linear <- Z_train
        }
        
        if (probit_outcome_model) {
          # Sample latent probit variable, z | -
          mu_forest_pred <- active_forest_mu$predict(forest_dataset_train)
          # come here
          tau_forest_pred <- as.vector(as.matrix(full_design_matrix_train)  %*% c(alpha, beta, beta_int))
          
          forest_pred <- mu_forest_pred + Z_linear*tau_forest_pred
          mu0 <- forest_pred[y_train == 0]
          mu1 <- forest_pred[y_train == 1]
          u0 <- runif(sum(y_train == 0), 0, pnorm(0 - mu0))
          u1 <- runif(sum(y_train == 1), pnorm(0 - mu1), 1)
          resid_train[y_train==0] <- mu0 + qnorm(u0)
          resid_train[y_train==1] <- mu1 + qnorm(u1)
          
          # Update outcome
          outcome_train$update_data(resid_train - forest_pred)
          current_sigma2 <- 1
        }
        
        # Sample the prognostic forest
        forest_model_mu$sample_one_iteration(
          forest_dataset = forest_dataset_train, residual = outcome_train, forest_samples = forest_samples_mu, 
          active_forest = active_forest_mu, rng = rng, forest_model_config = forest_model_config_mu, 
          global_model_config = global_model_config, keep_forest = keep_sample, gfr = FALSE
        )
        
        if (sample_sigma2_leaf_mu) {
          leaf_scale_mu_double <- sampleLeafVarianceOneIteration(active_forest_mu, rng, a_leaf_mu, b_leaf_mu)
          current_leaf_scale_mu <- as.matrix(leaf_scale_mu_double)
          if (keep_sample) leaf_scale_mu_samples[sample_counter] <- leaf_scale_mu_double
          forest_model_config_mu$update_leaf_model_scale(current_leaf_scale_mu)
        }
        if (sample_sigma_leaf_mu) {
          leaf_scale_mu_double <- sampleLeafVarianceOneIteration(active_forest_mu, rng, a_leaf_mu, b_leaf_mu)
          current_leaf_scale_mu <- as.matrix(leaf_scale_mu_double)
          if (keep_sample) leaf_scale_mu_samples[sample_counter] <- leaf_scale_mu_double
          forest_model_config_mu$update_leaf_model_scale(current_leaf_scale_mu)
        }
        if(simple_prior){
          sigma2_lin <- 1
        } else {
          sigma2_lin <- current_sigma2
        }
        
        # ----------------------------------------------------------------------
        # 1. SHAPLEY REPRESENTATION LEARNING
        # ----------------------------------------------------------------------
        if (shapley) {
          # A. Extract structure from the tree (Mu)
          Phi_raw <- stochtree:::get_saabas_shapley_active_forest_cpp(
            active_forest_mu$forest_ptr, 
            forest_dataset_train$data_ptr
          )
          
          # B. Handle Edge Cases
          Phi_raw[is.na(Phi_raw)] <- 0
          
          # C. Scale (Representation Learning needs standardized features)
          Phi_train <- apply(Phi_raw, 2, function(x) {
            sigma <- sd(x)
            if (is.na(sigma) || sigma < 1e-9) return(rep(0, length(x))) 
            else return((x - mean(x)) / sigma)
          })
          Phi_train <- as.matrix(Phi_train)
        }
        
        # ----------------------------------------------------------------------
        # 2. CALCULATE LINEAR RESIDUAL (The Fix)
        # ----------------------------------------------------------------------
        
        # We need to pass (Y - Current_Linear_Pred) to the C++ sampler.
        # The C++ sampler will then add Current_Linear_Pred back to this
        # to reconstruct Y as the target.
        
        # Calculate current linear fit in R
        # Note: We use the *previous* iteration's beta/gamma/alpha
        
        # Main effects
        current_linear_pred <- as.matrix(X_train_raw) %*% beta 
        
        # Add interactions if they exist
        if (p_int > 0) {
          # You need a way to get the interaction columns here. 
          # Assuming 'full_design_matrix_train' has them at the end:
          # (Adjust indices based on your design matrix structure)
          interaction_cols <- full_design_matrix_train[, (ncol(X_train_raw)+2):ncol(full_design_matrix_train)]
          current_linear_pred <- current_linear_pred + as.matrix(interaction_cols) %*% beta_int
        }
        
        # Add Shapley effects
        if (shapley) {
          current_linear_pred <- current_linear_pred + Phi_train %*% gamma
        }
        
        # Add Intercept
        if (regularize_ATE) {
          current_linear_pred <- current_linear_pred + Z_linear * alpha
        } else {
          current_linear_pred <- current_linear_pred + Z_linear * alpha
        }
        
        tau_residual <- resid_train - current_linear_pred
        
        if (!shapley && sigma_residual > 0) {
          tau_residual <- tau_residual + rnorm(n, 0, sigma_residual)
        }
        if(is_mcmc){
          all_partial_residuals[, mcmc_counter, chain_num] <- outcome_train$get_data()
        }
        
        if(use_ncp == TRUE){
          update_results <- updateLinearTreatmentCpp_NCP_cpp(
            X = if (propensity_seperate == "tau") X_train else X_train_raw,
            Phi = Phi_train,
            Z = Z_linear,
            propensity_train = propensity_train, 
            residual = tau_residual,
            are_continuous = as.vector(as.integer(boolean_continuous*1)),
            alpha_tilde = alpha, 
            gamma_prop = gamma_prop, 
            beta_tilde = beta,
            beta_int_tilde = beta_int,
            gamma_tilde = gamma_shap,
            tau_beta = tau_beta,
            tau_gamma = tau_gamma,
            nu = nu,
            nu_gamma = nu_gamma,
            xi = xi,
            tau_int = 1.0,
            sigma = sqrt(sigma2_lin),
            alpha_prior_sd = 10.0,
            tau_glob = tau_glob,
            sample_global_prior = sample_global_prior,
            unlink = unlink,
            gibbs = gibbs, 
            regularize_ATE = regularize_ATE,
            hn_scale = hn_scale,
            use_prognostic_shapley = shapley
          )
        } else {
          update_results <- updateLinearTreatmentCpp_cpp(
            X = if (propensity_seperate == "tau") X_train else X_train_raw,
            Phi = Phi_train,
            Z = Z_linear,
            propensity_train = propensity_train,
            residual = tau_residual,
            are_continuous = as.vector(as.integer(boolean_continuous*1)),
            alpha = alpha,
            gamma_prop = gamma_prop,
            beta = beta,
            beta_int = beta_int,
            gamma = gamma_shap,
            tau_beta = tau_beta,
            tau_gamma = tau_gamma,
            nu = nu,
            nu_gamma = nu_gamma,
            xi = xi,
            tau_int = tau_int,
            sigma = sqrt(sigma2_lin),
            alpha_prior_sd = 10.0,
            tau_glob = tau_glob,
            sample_global_prior = sample_global_prior,
            unlink = unlink,
            gibbs = gibbs,
            save_output = save_output,
            index = sample_counter,
            max_steps = max_steps,
            step_out = step_out,
            propensity_seperate = propensity_seperate,
            regularize_ATE = regularize_ATE,
            hn_scale = hn_scale,
            use_prognostic_shapley = shapley
          )
        }
        
        # 1. Extract Vectors
        params_vec <- update_results$params
        
        # 2. Extract Values
        idx <- 1
        alpha <- params_vec[idx]; idx <- idx + 1
        tau_int <- params_vec[idx]; idx <- idx + 1
        tau_glob <- params_vec[idx]; idx <- idx + 1
        gamma_prop <- params_vec[idx]; idx <- idx + 1
        xi <- params_vec[idx]; idx <- idx + 1
        
        beta_end <- idx + p_mod - 1
        beta <- params_vec[idx:beta_end]
        idx <- beta_end + 1
        
        beta_int_end <- idx + p_int - 1
        beta_int <- params_vec[idx:beta_int_end]
        idx <- beta_int_end + 1
        
        if (shapley) {
          gamma_shap_end <- idx + p_prog - 1
          gamma_shap <- params_vec[idx:gamma_shap_end]
          idx <- gamma_shap_end + 1
        }
        
        tau_beta_len <- length(tau_beta)
        tau_beta_end <- idx + tau_beta_len - 1
        tau_beta <- params_vec[idx:tau_beta_end]
        idx <- tau_beta_end + 1
        
        if (shapley) {
          tau_gamma_len <- length(tau_gamma)
          tau_gamma_end <- idx + tau_gamma_len - 1
          tau_gamma <- params_vec[idx:tau_gamma_end]
          idx <- tau_gamma_end + 1
        }
        
        nu_len <- length(nu)
        nu_end <- idx + nu_len - 1
        nu <- params_vec[idx:nu_end]
        idx <- nu_end + 1
        
        if (shapley) {
          nu_gamma_len <- length(nu_gamma)
          nu_gamma_end <- idx + nu_gamma_len - 1
          nu_gamma <- params_vec[idx:nu_gamma_end]
          idx <- nu_gamma_end + 1
        }
        
        if(shapley){
          current_linear_pred <- as.matrix(X_train_raw) %*% beta 
          
          # Add interactions if they exist
          if (p_int > 0) {
            interaction_cols <- full_design_matrix_train[, (ncol(X_train_raw)+2):ncol(full_design_matrix_train)]
            current_linear_pred <- current_linear_pred + as.matrix(interaction_cols) %*% beta_int
          }
          current_linear_pred <- current_linear_pred + Z_linear * alpha
          
          
          residual <- resid_train - current_linear_pred
          outcome_train$update_data(matrix(residual, ncol=1))
          
        } else {
          residual <- update_results$residuals
          outcome_train$update_data(matrix(residual, ncol=1))
        }

        if(use_ncp) {
            ate_offset <- as.integer(regularize_ATE)
            tau_beta_main_indices <- (1 + ate_offset):(p_mod + ate_offset)
            
            if(regularize_ATE) {
              alpha_real <- alpha * tau_beta[1] * tau_glob
            } else {
              alpha_real <- alpha
            }
            
            beta_real <- beta * tau_beta[tau_beta_main_indices] * tau_glob
            
            if(p_int > 0) {
              if(unlink) {
                tau_beta_int_indices <- (p_mod + ate_offset + 1):length(tau_beta)
                beta_int_real <- beta_int * tau_beta[tau_beta_int_indices] * tau_glob
              } else {
                int_pairs_matrix <- interaction_pairs(p_mod, boolean_continuous)
                
                beta_int_real <- numeric(p_int)
                for (k in 1:p_int) {
                  # Pak de 1-gebaseerde indexen (i, j) van de hoofdeffecten
                  idx_i <- int_pairs_matrix[1, k]
                  idx_j <- int_pairs_matrix[2, k]
                  
                  # Vind de bijbehorende tau_beta waarden (met offset)
                  tau_i <- tau_beta[idx_i + ate_offset]
                  tau_j <- tau_beta[idx_j + ate_offset]
                  
                  # Pas de NCP-formule toe
                  beta_int_real[k] <- beta_int[k] * tau_glob * (sqrt(tau_int) * tau_i * tau_j)
                }
              }            
            } else {
              beta_int_real <- numeric(0)
            }
            
          } else {
            alpha_real <- alpha
            beta_real <- beta
            beta_int_real <- beta_int
          }

          
          if(keep_sample){
            linear_counter <- linear_counter + 1
            
            alpha_samples[chain_num, linear_counter] <- alpha_real
            beta_samples[chain_num, linear_counter, ] <- beta_real
            beta_int_samples[chain_num, linear_counter, ] <- beta_int_real
            
            gamma_prop_samples[chain_num, linear_counter] <- gamma_prop
            tau_beta_samples[chain_num, linear_counter, ] <- tau_beta
            tau_int_samples[chain_num, linear_counter] <- tau_int
            tau_glob_samples[chain_num, linear_counter] <- tau_glob
            
            if(save_output){
              xi_samples[chain_num, linear_counter] <- xi 
              nu_samples[chain_num, linear_counter, ] <- nu
            }
            if (shapley) {
              gamma_shap_samples[chain_num, linear_counter, ] <- gamma_shap
              tau_gamma_samples[chain_num, linear_counter, ] <- tau_gamma
            }
          }}
        
        
        # Sample coding parameters (if requested)
        if (adaptive_coding) {
          # Estimate mu(X) and tau(X) and compute y - mu(X)
          mu_x_raw_train <- active_forest_mu$predict_raw(forest_dataset_train)
          tau_x_raw_train <- active_forest_tau$predict_raw(forest_dataset_train)
          partial_resid_mu_train <- resid_train - mu_x_raw_train
          if (has_rfx) {
            rfx_preds_train <- rfx_model$predict(rfx_dataset_train, rfx_tracker_train)
            partial_resid_mu_train <- partial_resid_mu_train - rfx_preds_train
          }
          
          # Compute sufficient statistics for regression of y - mu(X) on [tau(X)(1-Z), tau(X)Z]
          s_tt0 <- sum(tau_x_raw_train*tau_x_raw_train*(Z_train==0))
          s_tt1 <- sum(tau_x_raw_train*tau_x_raw_train*(Z_train==1))
          s_ty0 <- sum(tau_x_raw_train*partial_resid_mu_train*(Z_train==0))
          s_ty1 <- sum(tau_x_raw_train*partial_resid_mu_train*(Z_train==1))
          
          # Sample b0 (coefficient on tau(X)(1-Z)) and b1 (coefficient on tau(X)Z)
          current_b_0 <- rnorm(1, (s_ty0/(s_tt0 + 2*current_sigma2)), sqrt(current_sigma2/(s_tt0 + 2*current_sigma2)))
          current_b_1 <- rnorm(1, (s_ty1/(s_tt1 + 2*current_sigma2)), sqrt(current_sigma2/(s_tt1 + 2*current_sigma2)))
          
          # Update basis for the leaf regression
          tau_basis_train <- (1-Z_train)*current_b_0 + Z_train*current_b_1
          forest_dataset_train$update_basis(tau_basis_train)
          if (keep_sample) {
            b_0_samples[sample_counter] <- current_b_0
            b_1_samples[sample_counter] <- current_b_1
          }
          if (has_test) {
            tau_basis_test <- (1-Z_test)*current_b_0 + Z_test*current_b_1
            forest_dataset_test$update_basis(tau_basis_test)
          }
          
          # Update leaf predictions and residual
          # forest_model_tau$propagate_basis_update(forest_dataset_train, outcome_train, active_forest_tau)
        }
        
        # Sample variance parameters (if requested)
        if (include_variance_forest) {
          forest_model_variance$sample_one_iteration(
            forest_dataset = forest_dataset_train, residual = outcome_train, forest_samples = forest_samples_variance, 
            active_forest = active_forest_variance, rng = rng, forest_model_config = forest_model_config_variance, 
            global_model_config = global_model_config, keep_forest = keep_sample, gfr = FALSE
          )
        }
        if (sample_sigma_global) {
          # if(gibbs){
          #   lambda_sq <- tau_beta^2  # assume 'lambda' is a vector of lambda_j
          #   Lambda_inv <- diag(1 / lambda_sq)
          #   if(unlink){
          #     beta_tot <- c(beta, beta_int)
          #     scale <- as.numeric(0.5 * crossprod(outcome_train$get_data()) + 0.5 * t(beta_tot) %*% Lambda_inv %*% beta_tot)
          #     shape <- (n + p_mod + p_int) / 2
          #     current_sigma2 <- rinvgamma(shape, scale)
          #   } else {
          #     beta_tot <- c(beta)
          #     scale <- as.numeric(0.5 * crossprod(outcome_train$get_data()) + 0.5 * t(beta_tot) %*% Lambda_inv %*% beta_tot)
          #     shape <- (n + p_mod) / 2
          #     current_sigma2 <- rinvgamma(shape, scale)
          #   }
          #   global_model_config$update_global_error_variance(current_sigma2)
          #   
          #   
          # # } else {
          #   current_sigma2 <- sampleGlobalErrorVarianceOneIteration(outcome_train, forest_dataset_train, rng, a_global, b_global)
          #   global_model_config$update_global_error_variance(current_sigma2)
          #   if (keep_sample) global_var_samples[sample_counter] <- current_sigma2
          # # }
          
          shape_sigma_post <- 0.001 + n / 2
          rate_sigma_post <- 0.001 + 0.5 * sum(outcome_train$get_data()^2)
          sigma_sq <- rinvgamma(shape = shape_sigma_post, scale = rate_sigma_post)
          current_sigma2 <- max(sigma_sq, 1e-9)
          global_model_config$update_global_error_variance(current_sigma2)
          if (keep_sample) global_var_samples[sample_counter] <- current_sigma2
          
        }
        # Sample random effects parameters (if requested)
        if (has_rfx) {
          rfx_model$sample_random_effect(rfx_dataset_train, outcome_train, rfx_tracker_train, rfx_samples, keep_sample, current_sigma2, rng)
        }
      }
    }
  
  # Remove GFR samples if they are not to be retained
  if ((!keep_gfr) && (num_gfr > 0)) {
    for (i in 1:num_gfr) {
      forest_samples_mu$delete_sample(0)
      # forest_samples_tau$delete_sample(0)
      if (include_variance_forest) {
        forest_samples_variance$delete_sample(0)
      }
      if (has_rfx) {
        rfx_samples$delete_sample(0)
      }
    }
    if (sample_sigma2_global) {
      global_var_samples <- global_var_samples[(num_gfr+1):length(global_var_samples)]
    }
    if (sample_sigma2_leaf_mu) {
      leaf_scale_mu_samples <- leaf_scale_mu_samples[(num_gfr+1):length(leaf_scale_mu_samples)]
    }
    if (sample_sigma2_leaf_tau) {
      leaf_scale_tau_samples <- leaf_scale_tau_samples[(num_gfr+1):length(leaf_scale_tau_samples)]
    }
    if (adaptive_coding) {
      b_1_samples <- b_1_samples[(num_gfr+1):length(b_1_samples)]
      b_0_samples <- b_0_samples[(num_gfr+1):length(b_0_samples)]
    }
    num_retained_samples <- num_retained_samples - num_gfr
  }
  
  # Forest predictions
  mu_hat_train <- forest_samples_mu$predict(forest_dataset_train)*y_std_train + y_bar_train
  if (adaptive_coding) {
    # tau_hat_train_raw <- forest_samples_tau$predict_raw(forest_dataset_train)
    # tau_hat_train <- t(t(tau_hat_train_raw) * (b_1_samples - b_0_samples))*y_std_train
  } else {
    tau_hat_train <- forest_samples_tau$predict_raw(forest_dataset_train)*y_std_train
  }
  y_hat_train <- mu_hat_train + predict_interaction_lm(if(propensity_seperate == "tau") X_train else X_train_raw, c(alpha, beta, beta_int), X_final_var_info, interaction_rule, propensity_seperate) * as.numeric(Z_train)
  if (has_test) {
    mu_hat_test <- forest_samples_mu$predict(forest_dataset_test)*y_std_train + y_bar_train
    if (adaptive_coding) {
      tau_hat_test_raw <- forest_samples_tau$predict_raw(forest_dataset_test)
      tau_hat_test <- t(t(tau_hat_test_raw) * (b_1_samples - b_0_samples))*y_std_train
    } else {
      tau_hat_test <- forest_samples_tau$predict_raw(forest_dataset_test)*y_std_train
    }
    y_hat_test <- mu_hat_test + tau_hat_test * as.numeric(Z_test)
  }
  if (include_variance_forest) {
    sigma2_x_hat_train <- forest_samples_variance$predict(forest_dataset_train)
    if (has_test) sigma2_x_hat_test <- forest_samples_variance$predict(forest_dataset_test)
  }
  
  # Random effects predictions
  if (has_rfx) {
    rfx_preds_train <- rfx_samples$predict(rfx_group_ids_train, rfx_basis_train)*y_std_train
    y_hat_train <- y_hat_train + rfx_preds_train
  }
  if ((has_rfx_test) && (has_test)) {
    rfx_preds_test <- rfx_samples$predict(rfx_group_ids_test, rfx_basis_test)*y_std_train
    y_hat_test <- y_hat_test + rfx_preds_test
  }
  
  # Global error variance
  if (sample_sigma2_global) sigma2_global_samples <- global_var_samples*(y_std_train^2)
  
  # Leaf parameter variance for prognostic forest
  if (sample_sigma2_leaf_mu) sigma2_leaf_mu_samples <- leaf_scale_mu_samples
  
  # Leaf parameter variance for treatment effect forest
  if (sample_sigma2_leaf_tau) sigma2_leaf_tau_samples <- leaf_scale_tau_samples
  
  # Rescale variance forest prediction by global sigma2 (sampled or constant)
  if (include_variance_forest) {
    if (sample_sigma2_global) {
      sigma2_x_hat_train <- sapply(1:num_retained_samples, function(i) sigma2_x_hat_train[,i]*sigma2_global_samples[i])
      if (has_test) sigma2_x_hat_test <- sapply(1:num_retained_samples, function(i) sigma2_x_hat_test[,i]*sigma2_global_samples[i])
    } else {
      sigma2_x_hat_train <- sigma2_x_hat_train*sigma2_init*y_std_train*y_std_train
      if (has_test) sigma2_x_hat_test <- sigma2_x_hat_test*sigma2_init*y_std_train*y_std_train
    }
  }
  
  # Return results as a list
  if (include_variance_forest) {
    num_variance_covariates <- sum(variable_weights_variance > 0)
  } else {
    num_variance_covariates <- 0
  }
  model_params <- list(
    "initial_sigma2" = sigma2_init, 
    "initial_sigma2_leaf_mu" = sigma2_leaf_mu,
    "initial_sigma2_leaf_tau" = sigma2_leaf_tau,
    "initial_b_0" = b_0,
    "initial_b_1" = b_1,
    "a_global" = a_global,
    "b_global" = b_global,
    "a_leaf_mu" = a_leaf_mu, 
    "b_leaf_mu" = b_leaf_mu,
    "a_leaf_tau" = a_leaf_tau, 
    "b_leaf_tau" = b_leaf_tau,
    "a_forest" = a_forest, 
    "b_forest" = b_forest,
    "outcome_mean" = y_bar_train,
    "outcome_scale" = y_std_train,
    "standardize" = standardize, 
    "num_covariates" = num_cov_orig,
    "num_prognostic_covariates" = sum(variable_weights_mu > 0),
    "num_treatment_covariates" = sum(variable_weights_tau > 0),
    "num_variance_covariates" = num_variance_covariates,
    "treatment_dim" = ncol(Z_train), 
    "propensity_covariate" = propensity_covariate, 
    "binary_treatment" = binary_treatment, 
    "adaptive_coding" = adaptive_coding, 
    "internal_propensity_model" = internal_propensity_model, 
    "num_samples" = num_retained_samples, 
    "num_gfr" = num_gfr, 
    "num_burnin" = num_burnin, 
    "num_mcmc" = num_mcmc, 
    "keep_every" = keep_every,
    "num_chains" = num_chains,
    "has_rfx" = has_rfx, 
    "has_rfx_basis" = has_basis_rfx, 
    "num_rfx_basis" = num_basis_rfx, 
    "include_variance_forest" = include_variance_forest, 
    "sample_sigma2_global" = sample_sigma2_global,
    "sample_sigma2_leaf_mu" = sample_sigma2_leaf_mu,
    "sample_sigma2_leaf_tau" = sample_sigma2_leaf_tau, 
    "probit_outcome_model" = probit_outcome_model
  )
  if(save_output){
    result <- list(
      "forests_mu" = forest_samples_mu, 
      "forests_tau" = forest_samples_tau, 
      "model_params" = model_params, 
      "mu_hat_train" = mu_hat_train, 
      "tau_hat_train" = tau_hat_train, 
      "y_hat_train" = y_hat_train, 
      "train_set_metadata" = X_train_metadata,
      "alpha" = alpha_samples,
      "Beta" = beta_samples,
      "Tau" = tau_beta_samples,
      "Xi" = xi_samples,
      "Nu" = nu_samples,
      "Beta_int" = beta_int_samples,
      "Tau_int" = tau_int_samples,
      "Tau_glob" = tau_glob_samples
    )
  } else {
    result <- list( 
      "forests_mu" = forest_samples_mu, 
      "forests_tau" = forest_samples_tau, 
      "model_params" = model_params, 
      "mu_hat_train" = mu_hat_train, 
      "tau_hat_train" = tau_hat_train, 
      "y_hat_train" = y_hat_train, 
      "train_set_metadata" = X_train_metadata,
      "alpha" = alpha_samples,
      "Beta" = beta_samples,
      "Tau" = tau_beta_samples,
      "Gamma_Shapley" = gamma_shap_samples,
      "Tau_Gamma" = tau_gamma_samples,
      "Beta_int" = beta_int_samples,
      "Tau_int" = tau_int_samples,
      "Tau_glob" = tau_glob_samples
    )}
  if (propensity_seperate == "mu"){
    result[["gamma"]] = gamma_prop_samples
  }
  if (has_test) result[["mu_hat_test"]] = mu_hat_test
  if (has_test) result[["tau_hat_test"]] = tau_hat_test
  if (has_test) result[["y_hat_test"]] = y_hat_test
  if (include_variance_forest) {
    result[["forests_variance"]] = forest_samples_variance
    result[["sigma2_x_hat_train"]] = sigma2_x_hat_train
    if (has_test) result[["sigma2_x_hat_test"]] = sigma2_x_hat_test
  }
  if (sample_sigma2_global) result[["sigma2_global_samples"]] = sigma2_global_samples
  if (sample_sigma2_leaf_mu) result[["sigma2_leaf_mu_samples"]] = sigma2_leaf_mu_samples
  if (sample_sigma2_leaf_tau) result[["sigma2_leaf_tau_samples"]] = sigma2_leaf_tau_samples
  if (adaptive_coding) {
    result[["b_0_samples"]] = b_0_samples
    result[["b_1_samples"]] = b_1_samples
  }
  if (has_rfx) {
    result[["rfx_samples"]] = rfx_samples
    result[["rfx_preds_train"]] = rfx_preds_train
    result[["rfx_unique_group_ids"]] = levels(group_ids_factor)
  }
  if ((has_rfx_test) && (has_test)) result[["rfx_preds_test"]] = rfx_preds_test
  
  if (internal_propensity_model) {
    result[["bart_propensity_model"]] = bart_model_propensity
  } 
  
  if(save_partial_residual){
    result[["partial_residuals"]] = all_partial_residuals
  }
  
  class(result) <- "bcfmodel"
  
  return(result)
}