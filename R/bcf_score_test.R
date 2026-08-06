#' Fit a Restricted BCF Model (Constant Treatment Effect) and Compute Score Test P-Values
#'
#' @param X_train Covariates used to split trees in the ensemble.
#' @param Z_train Vector of treatment assignments.
#' @param y_train Outcome to be modeled by the ensemble.
#' @param propensity_train (Optional) Vector of propensity scores.
#' @param rfx_group_ids_train (Optional) Group labels used for an additive random effects model.
#'   NOTE: random effects are not implemented in the restricted sampler; supplying this argument is an error.
#' @param rfx_basis_train (Optional) Basis for "random-slope" regression. See note on \code{rfx_group_ids_train}.
#' @param num_gfr Number of "warm-start" iterations. Default: 5.
#' @param num_burnin Number of "burn-in" iterations. Default: 100.
#' @param num_mcmc Number of "retained" iterations. Default: 500.
#' @param use_rao_blackwell Logical. If TRUE, averages the score vector and covariance matrix across MCMC draws to compute a single optimal P-value. Default: TRUE.
#' @param test_type Character string specifying the score test type:
#'   \code{"mean"} tests the conditional mean (BLP of the CATE in the score design);
#'   \code{"distributional"} tests the full conditional distribution via the ECDF score process;
#'   \code{"variance"} tests for variance (scale) heterogeneity via the squared-residual score
#'   \code{s_i = Z_cen_i (e_i^2 - mean(e^2))}, sharing the mean test's machinery and its double robustness;
#'   \code{"smooth"} is a Neyman smooth test: the score process is projected on
#'   \code{num_smooth_components} orthonormal Legendre polynomials of the residual rank, so that
#'   component 1 captures location heterogeneity, component 2 scale, components 3+ skew/tail shape.
#'   Default: \code{"mean"}.
#' @param num_grid_points Integer specifying the number of evaluation points for the distributional ECDF test. Ignored unless \code{test_type = "distributional"}. Default: 50.
#' @param num_smooth_components Integer number of Legendre components (1-6) for \code{test_type = "smooth"}. Default: 4.
#' @param dist_interaction_rule (Optional) Interaction rule for the score design used by the
#'   \code{"distributional"} and \code{"smooth"} tests only (e.g. \code{"none"} for main effects),
#'   allowing a smaller basis than the mean test. \code{NULL} (default) uses \code{general_params$interaction_rule}.
#' @param centering For the distributional test, how indicators are centered: \code{"ecdf"} (default,
#'   marginal empirical CDF — current behavior) or \code{"model"} (the fitted null model's Gaussian CDF,
#'   \code{pnorm(t / sigma)}), which restores a product-bias/double-robustness structure at the price of
#'   assuming the null error law is Gaussian.
#' @param standardize_residuals Logical. If TRUE, residuals are divided by the per-draw \code{sigma}
#'   before computing indicator/rank scores, removing scale drift across MCMC draws. Default: FALSE.
#' @param grid_pool_draws Integer >= 1. Number of trailing burn-in draws whose residuals are pooled
#'   (together with the first retained draw) to set the ECDF grid. Default: 1 (current behavior: first retained draw only).
#' @param ad_weights Logical. If TRUE, the CvM-type distributional statistic uses Anderson-Darling
#'   weights \code{1/(F(1-F))}, upweighting the tails where scale alternatives concentrate. Default: FALSE.
#' @param null_method How the null distribution of the Rao-Blackwellized statistics is computed:
#'   \code{"gaussian"} (default, current behavior: simulate \code{N(0, Sigma_hat)} via Cholesky of the full
#'   covariance), \code{"multiplier"} (Gaussian multiplier bootstrap \code{t(X_s) \%*\% G} — the same law,
#'   but never forms or factors the \code{(p*K) x (p*K)} matrix), or \code{"permutation"} (permute the
#'   \code{Z - pihat} marks against the fitted residual scores; finite-sample-motivated, recommended with
#'   \code{num_null_draws = 2000}). For \code{"mean"}/\code{"variance"}, \code{"multiplier"} is equivalent
#'   to \code{"gaussian"} and is treated as such. Ignored when \code{use_rao_blackwell = FALSE}.
#' @param num_null_draws Number of simulation draws (or permutations) for null distributions. Default: 10000.
#' @param propensity_clip (Optional) If non-NULL, propensity scores are clipped to
#'   \code{[propensity_clip, 1 - propensity_clip]}. Recommended (e.g. 0.02) whenever the propensity is
#'   estimated internally, since the internal BART fit is not constrained to (0, 1). Default: NULL (no clipping).
#' @param general_params (Optional) A list of general model parameters.
#' @param prognostic_forest_params (Optional) A list of prognostic forest model parameters.
#' @param variance_forest_params (Optional) A list of variance forest model parameters.
#' @return A list of class `bcfscoretest` containing posterior samples and score test P-values.
#'   For \code{test_type = "mean"} / \code{"variance"}: \code{rb_pval_quad}, \code{rb_pval_max}.
#'   For \code{"distributional"}: \code{rb_pval_wass}, \code{rb_pval_cvm}, \code{rb_pval_max}, \code{t_grid}.
#'   For \code{"smooth"}: \code{rb_pval_quad}, \code{rb_pval_max}, \code{rb_pval_minp}, and
#'   \code{rb_pval_components} (chains x components; component j's chi-squared p-value).
#' @export
bcf_restricted_score_test <- function(X_train, Z_train, y_train, propensity_train = NULL,
                                      rfx_group_ids_train = NULL, rfx_basis_train = NULL,
                                      num_gfr = 5, num_burnin = 100, num_mcmc = 500,
                                      use_rao_blackwell = TRUE,
                                      test_type = c("mean", "distributional", "variance", "smooth"),
                                      num_grid_points = 50,
                                      num_smooth_components = 4,
                                      dist_interaction_rule = NULL,
                                      centering = c("ecdf", "model"),
                                      standardize_residuals = FALSE,
                                      grid_pool_draws = 1,
                                      ad_weights = FALSE,
                                      null_method = c("gaussian", "multiplier", "permutation"),
                                      num_null_draws = 10000,
                                      propensity_clip = NULL,
                                      general_params = list(), prognostic_forest_params = list(),
                                      variance_forest_params = list()) {

  test_type <- match.arg(test_type)
  centering <- match.arg(centering)
  null_method <- match.arg(null_method)

  if (!requireNamespace("MASS", quietly = TRUE)) stop("Package 'MASS' is required.")

  if (!is.null(rfx_group_ids_train) || !is.null(rfx_basis_train)) {
    stop(paste0("Random effects are not implemented in the restricted score-test sampler: the rfx ",
                "arguments would be silently ignored and the i.i.d. sandwich covariance would be ",
                "invalid for clustered data. Please remove rfx_group_ids_train / rfx_basis_train."))
  }

  if (test_type %in% c("distributional", "smooth") && !use_rao_blackwell) {
    stop(paste0("test_type = '", test_type, "' requires use_rao_blackwell = TRUE"))
  }
  if (num_smooth_components < 1 || num_smooth_components > 6) {
    stop("num_smooth_components must be between 1 and 6.")
  }
  if (grid_pool_draws < 1) stop("grid_pool_draws must be >= 1.")
  # Multiplier and Gaussian simulation draw from the same law; for the finite-dimensional
  # mean/variance scores the Gaussian path is already cheap, so collapse the option.
  if (test_type %in% c("mean", "variance") && null_method == "multiplier") null_method <- "gaussian"

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

  # Optional clipping guardrail: the internal Gaussian-likelihood BART fit is not constrained
  # to (0, 1), and both tests' validity leans on the propensity (entirely so for the
  # distributional test), so overlap violations should not pass through silently.
  if (!is.null(propensity_clip)) {
    if (propensity_clip <= 0 || propensity_clip >= 0.5) stop("propensity_clip must be in (0, 0.5).")
    propensity_train <- pmin(pmax(propensity_train, propensity_clip), 1 - propensity_clip)
  }

  # Center treatment indicator
  Z_cen <- Z_train - as.vector(propensity_train)

  # =====================================================================
  # SCORE TEST DESIGN MATRIX
  # =====================================================================
  # The distributional/smooth tests may use a smaller basis than the mean test:
  # their score already spans num_grid_points / num_smooth_components coordinates
  # per design column, so a lean design concentrates power.
  score_design_rule <- general_params_updated$interaction_rule
  if (!is.null(dist_interaction_rule) && test_type %in% c("distributional", "smooth")) {
    score_design_rule <- dist_interaction_rule
  }
  design_info <- prepare_score_test_design(X_train_raw, X_train_metadata_init, score_design_rule)
  X_cen <- design_info$X_cen
  p_valid <- design_info$p_valid

  if (p_valid > n / 10) {
    warning(sprintf(paste0("Score design has p_valid = %d columns for n = %d observations; the ",
                           "chi-squared calibration of the quadratic statistic degrades when p_valid ",
                           "is not small relative to n. Consider interaction_rule = 'none' or a smaller design."),
                    p_valid, n))
  }
  if (test_type == "distributional" && p_valid * num_grid_points > n) {
    warning(sprintf(paste0("Distributional score process has p_valid * K = %d coordinates but only ",
                           "n = %d observations: the empirical covariance is rank-degenerate and power ",
                           "is diluted. Consider reducing num_grid_points, setting dist_interaction_rule ",
                           "= 'none', or using test_type = 'smooth'."),
                    p_valid * num_grid_points, n))
  }

  num_chains <- general_params_updated$num_chains
  alpha_samples <- matrix(0, nrow = num_chains, ncol = num_mcmc)
  sigma2_samples <- matrix(0, nrow = num_chains, ncol = num_mcmc)

  # Accumulators. For every test type the per-draw score factors as
  #   s_i(.) = Z_cen_i * base_i(.),
  # with base_i depending only on the residuals, so we accumulate the base and
  # multiply by Z_cen once after averaging. This is numerically equivalent to
  # accumulating the product (Z_cen is constant across draws) and is what makes
  # the mark-permutation null cheap: permutations reuse the averaged base.
  if (use_rao_blackwell) {
    if (test_type %in% c("mean", "variance")) {
      base_sum <- matrix(0, nrow = num_chains, ncol = n)
    } else if (test_type == "smooth") {
      base_mat_sum <- replicate(num_chains, matrix(0, nrow = n, ncol = num_smooth_components), simplify = FALSE)
    } else {
      base_mat_sum <- replicate(num_chains, matrix(0, nrow = n, ncol = num_grid_points), simplify = FALSE)
    }
    pval_max_samples <- NULL
    pval_quad_samples <- NULL
  } else {
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
  sigma2_global_shape <- general_params_updated$sigma2_global_shape
  sigma2_global_scale <- general_params_updated$sigma2_global_scale
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
    # Grid is set adaptively at the first MCMC iteration, after BART has warmed up
    # and the residuals reflect the fitted model. With grid_pool_draws > 1, residuals
    # from the trailing burn-in draws (of the first chain) are pooled in as well to
    # stabilize the grid against a single atypical draw.
    t_grid <- NULL
    grid_initialized <- FALSE
    grid_pool_buffer <- NULL
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
        shape_sigma_post <- sigma2_global_shape + n / 2
        rate_sigma_post <- sigma2_global_scale + 0.5 * sum(outcome_train$get_data()^2)
        current_sigma2 <- max(rinvgamma(shape_sigma_post, rate_sigma_post), 1e-9)
        global_model_config$update_global_error_variance(current_sigma2)
      }

      # Pool trailing burn-in residuals for the distributional grid (first chain only)
      if (test_type == "distributional" && chain_num == 1 && !is_mcmc && grid_pool_draws > 1 &&
          i > (num_gfr + num_burnin - (grid_pool_draws - 1))) {
        pool_resid <- if (standardize_residuals) resid_full / sqrt(current_sigma2) else resid_full
        grid_pool_buffer <- c(grid_pool_buffer, as.numeric(pool_resid))
      }

      # =====================================================================
      # SCORE TEST INJECTION
      # =====================================================================
      if (is_mcmc) {
        mcmc_counter <- i - num_gfr - num_burnin

        alpha_samples[chain_num, mcmc_counter] <- alpha
        sigma2_samples[chain_num, mcmc_counter] <- current_sigma2

        resid_score <- if (standardize_residuals) resid_full / sqrt(current_sigma2) else resid_full

        if (test_type %in% c("mean", "variance")) {
          # base_i: residual (mean test) or centered squared residual (variance test)
          if (test_type == "mean") {
            base_i <- as.numeric(resid_score)
          } else {
            r2 <- as.numeric(resid_score)^2
            base_i <- r2 - mean(r2)
          }

          if (use_rao_blackwell) {
            base_sum[chain_num, ] <- base_sum[chain_num, ] + base_i
          } else {
            # Legacy method: P-values at every draw using safe evaluation
            s_i <- base_i * Z_cen
            s_i_cen <- s_i - mean(s_i)
            T_vec <- t(X_cen) %*% s_i_cen
            X_s <- X_cen * as.vector(s_i_cen)
            Var_T <- crossprod(X_s)
            draw_pvals <- compute_pmvnorm_safe(T_vec, Var_T, p_valid, n_sim = num_null_draws)
            pval_quad_samples[chain_num, mcmc_counter] <- draw_pvals$quad
            pval_max_samples[chain_num, mcmc_counter] <- draw_pvals$max
          }
        } else if (test_type == "smooth") {
          # Neyman smooth components: orthonormal Legendre polynomials of the residual rank.
          # Component 1 ~ location, 2 ~ scale, 3+ ~ skew/tail heterogeneity.
          u_i <- (rank(as.numeric(resid_score), ties.method = "average") - 0.5) / n
          base_mat_sum[[chain_num]] <- base_mat_sum[[chain_num]] + legendre_basis(u_i, num_smooth_components)
        } else if (test_type == "distributional") {
          # On the first MCMC iteration, set the grid from post-warmup residuals
          if (!grid_initialized) {
            grid_pool <- c(grid_pool_buffer, as.numeric(resid_score))
            t_grid <- as.numeric(quantile(grid_pool, probs = seq(0.01, 0.99, length.out = num_grid_points)))
            grid_initialized <- TRUE
            grid_pool_buffer <- NULL
          }

          # Calculate Empirical CDF Indicators and centered indicator matrix
          # I_mat: Indicator matrix for residuals falling below each grid point t
          # F_k_draw: CDF at each grid point — marginal ECDF ("ecdf") or the fitted
          #           null model's Gaussian CDF ("model", restores double robustness
          #           under Gaussian errors)
          # e_mat: Centered indicator (I - F), which has mean 0 under the null
          I_mat <- outer(as.numeric(resid_score), t_grid, "<=") * 1.0
          if (centering == "ecdf") {
            F_k_draw <- colMeans(I_mat)
          } else {
            F_k_draw <- pnorm(t_grid, sd = if (standardize_residuals) 1.0 else sqrt(current_sigma2))
          }
          e_mat <- sweep(I_mat, 2, F_k_draw, "-")

          if (use_rao_blackwell) {
            base_mat_sum[[chain_num]] <- base_mat_sum[[chain_num]] + e_mat
          }
        }
      }
    }
  }

  # =====================================================================
  # POST-PROCESSING FOR RAO-BLACKWELLIZATION
  # =====================================================================
  if (use_rao_blackwell) {
    if (test_type %in% c("mean", "variance")) {
      rb_pval_quad <- numeric(num_chains)
      rb_pval_max <- numeric(num_chains)

      for (c in 1:num_chains) {
        base_mean <- base_sum[c, ] / num_mcmc
        s_i_mean <- base_mean * Z_cen
        s_i_mean <- s_i_mean - mean(s_i_mean)

        if (null_method == "permutation") {
          rb_pvals <- compute_mark_permutation_pvalue(base_mean, Z_cen, X_cen, p_valid, n_perm = num_null_draws)
        } else {
          T_vec_mean <- t(X_cen) %*% s_i_mean

          # Compute the Huber-White empirical sandwich variance estimator of the score.
          # This provides robustness against heteroskedasticity in the residuals.
          X_s_mean <- X_cen * as.vector(s_i_mean)
          Var_T_mean <- crossprod(X_s_mean)

          rb_pvals <- compute_pmvnorm_safe(T_vec_mean, Var_T_mean, p_valid, n_sim = num_null_draws)
        }
        rb_pval_quad[c] <- rb_pvals$quad
        rb_pval_max[c] <- rb_pvals$max
      }
    } else if (test_type == "smooth") {
      rb_pval_quad <- numeric(num_chains)
      rb_pval_max_smooth <- numeric(num_chains)
      rb_pval_minp <- numeric(num_chains)
      rb_pval_components <- matrix(0, nrow = num_chains, ncol = num_smooth_components)

      for (c in 1:num_chains) {
        phi_mean <- base_mat_sum[[c]] / num_mcmc
        pvals_smooth <- compute_smooth_pvalue_safe(phi_mean, Z_cen, X_cen, p_valid, num_smooth_components,
                                                   n_sim = num_null_draws, null_method = null_method)
        rb_pval_quad[c] <- pvals_smooth$quad
        rb_pval_max_smooth[c] <- pvals_smooth$max
        rb_pval_minp[c] <- pvals_smooth$minp
        rb_pval_components[c, ] <- pvals_smooth$components
      }
    } else {
      rb_pval_wass <- numeric(num_chains)
      rb_pval_cvm <- numeric(num_chains)
      rb_pval_max_dist <- numeric(num_chains)

      for (c in 1:num_chains) {
        e_mat_mean <- base_mat_sum[[c]] / num_mcmc
        s_mat_mean <- e_mat_mean * as.numeric(Z_cen)

        pvals_dist <- compute_distributional_pvalue_safe(s_mat_mean, t_grid, X_cen, p_valid, num_grid_points,
                                                         n_sim = num_null_draws, null_method = null_method,
                                                         ad_weights = ad_weights, Z_cen = Z_cen,
                                                         e_mat_mean = e_mat_mean)
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
    "train_set_metadata" = X_train_metadata,
    "test_config" = list(
      test_type = test_type, use_rao_blackwell = use_rao_blackwell, null_method = null_method,
      num_grid_points = num_grid_points, num_smooth_components = num_smooth_components,
      score_design_rule = score_design_rule, p_valid = p_valid, centering = centering,
      standardize_residuals = standardize_residuals, grid_pool_draws = grid_pool_draws,
      ad_weights = ad_weights, num_null_draws = num_null_draws, propensity_clip = propensity_clip
    )
  )

  if (use_rao_blackwell) {
    if (test_type %in% c("mean", "variance")) {
      result[["rb_pval_quad"]] <- rb_pval_quad
      result[["rb_pval_max"]] <- rb_pval_max
    } else if (test_type == "smooth") {
      result[["rb_pval_quad"]] <- rb_pval_quad
      result[["rb_pval_max"]] <- rb_pval_max_smooth
      result[["rb_pval_minp"]] <- rb_pval_minp
      result[["rb_pval_components"]] <- rb_pval_components
    } else {
      result[["rb_pval_wass"]] <- rb_pval_wass
      result[["rb_pval_cvm"]] <- rb_pval_cvm
      result[["rb_pval_max"]] <- rb_pval_max_dist
      result[["t_grid"]] <- t_grid
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

#' Orthonormal Shifted Legendre Polynomial Basis on [0, 1]
#'
#' Basis functions for the Neyman smooth score test: orthonormal with respect to the
#' uniform distribution on [0, 1] and mean zero, so that under the null (residual ranks
#' approximately uniform) each component is a valid mean-zero score direction.
#'
#' @param u Vector of values in [0, 1] (typically shifted residual ranks).
#' @param J Number of components (1 to 6).
#' @return A `length(u) x J` matrix of basis evaluations.
legendre_basis <- function(u, J) {
  P <- matrix(0, nrow = length(u), ncol = J)
  if (J >= 1) P[, 1] <- sqrt(3) * (2*u - 1)
  if (J >= 2) P[, 2] <- sqrt(5) * (6*u^2 - 6*u + 1)
  if (J >= 3) P[, 3] <- sqrt(7) * (20*u^3 - 30*u^2 + 12*u - 1)
  if (J >= 4) P[, 4] <- 3 * (70*u^4 - 140*u^3 + 90*u^2 - 20*u + 1)
  if (J >= 5) P[, 5] <- sqrt(11) * (252*u^5 - 630*u^4 + 560*u^3 - 210*u^2 + 30*u - 1)
  if (J >= 6) P[, 6] <- sqrt(13) * (924*u^6 - 2772*u^5 + 3150*u^4 - 1680*u^3 + 420*u^2 - 42*u + 1)
  P
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

#' Mark-Permutation P-Values for Vector Score Tests (Mean / Variance)
#'
#' Permutes the centered-treatment marks `Z_cen` against the (fixed) Rao-Blackwellized
#' residual base, recomputing the score vector, its sandwich covariance, and both the
#' quadratic and max statistics for each permutation. Because the fitted nuisance is held
#' fixed, this is an approximate (not exact) permutation test, but it typically has better
#' finite-sample size than the plug-in Gaussian reference.
#'
#' @param base_mean Rao-Blackwellized residual base (n-vector; residuals or centered squared residuals).
#' @param Z_cen Centered treatment vector.
#' @param X_cen The full-rank centered design matrix.
#' @param p_valid The rank of the design matrix.
#' @param n_perm Number of permutations.
#' @return A list with `quad` and `max` p-values.
compute_mark_permutation_pvalue <- function(base_mean, Z_cen, X_cen, p_valid, n_perm = 2000) {
  n <- nrow(X_cen)
  ridge <- diag(1e-8, p_valid)

  stat_pair <- function(z) {
    s <- base_mean * z
    s <- s - mean(s)
    T_vec <- crossprod(X_cen, s)
    V <- crossprod(X_cen * as.vector(s)) + ridge
    V_inv <- tryCatch(solve(V), error = function(e) MASS::ginv(V))
    quad <- as.numeric(t(T_vec) %*% V_inv %*% T_vec)
    max_z <- max(abs(as.numeric(T_vec) / sqrt(diag(V))))
    c(quad, max_z)
  }

  obs <- stat_pair(Z_cen)
  perm_stats <- matrix(0, nrow = n_perm, ncol = 2)
  for (b in 1:n_perm) {
    perm_stats[b, ] <- stat_pair(Z_cen[sample.int(n)])
  }

  return(list(quad = mean(perm_stats[, 1] >= obs[1]),
              max = mean(perm_stats[, 2] >= obs[2])))
}

#' Compute P-Values for the Neyman Smooth Score Test
#'
#' Projects the treatment-heterogeneity score process on J orthonormal Legendre components
#' of the residual rank and computes: an overall quadratic statistic across all p*J
#' coordinates, per-component chi-squared p-values, a min-p combination across components
#' (calibrated by joint simulation), and a max-|Z| statistic.
#'
#' @param phi_mean Rao-Blackwellized n x J matrix of Legendre basis evaluations.
#' @param Z_cen Centered treatment vector.
#' @param X_cen The full-rank centered design matrix.
#' @param p_valid The rank of the design matrix.
#' @param J The number of smooth components.
#' @param n_sim Number of null simulation draws (or permutations).
#' @param null_method One of "gaussian", "multiplier", "permutation".
#' @return A list with `quad`, `max`, `minp`, and `components` (J-vector of chi-squared p-values).
compute_smooth_pvalue_safe <- function(phi_mean, Z_cen, X_cen, p_valid, J, n_sim = 10000,
                                       null_method = "gaussian") {
  n <- nrow(X_cen)
  s_mat <- phi_mean * as.numeric(Z_cen)
  s_mat <- sweep(s_mat, 2, colMeans(s_mat), "-")

  U <- crossprod(X_cen, s_mat)   # p x J

  A_list <- vector("list", J)
  V_inv_list <- vector("list", J)
  sd_mat <- matrix(0, nrow = p_valid, ncol = J)
  T_j_obs <- numeric(J)
  for (j in 1:J) {
    A_j <- X_cen * s_mat[, j]
    A_list[[j]] <- A_j
    V_j <- crossprod(A_j) + diag(1e-8, p_valid)
    V_inv <- tryCatch(solve(V_j), error = function(e) MASS::ginv(V_j))
    V_inv_list[[j]] <- V_inv
    sd_mat[, j] <- sqrt(diag(V_j))
    T_j_obs[j] <- as.numeric(t(U[, j]) %*% V_inv %*% U[, j])
  }

  comp_pvals <- pchisq(T_j_obs, df = p_valid, lower.tail = FALSE)
  minp_obs <- min(comp_pvals)
  max_Z_obs <- max(abs(U / sd_mat))

  # Overall quadratic form across all p*J coordinates (pJ is small by construction)
  tilde_X <- matrix(0, nrow = n, ncol = p_valid * J)
  for (j in 1:J) tilde_X[, ((j-1)*p_valid + 1):(j*p_valid)] <- A_list[[j]]
  Sigma_full <- crossprod(tilde_X) + diag(1e-8, p_valid * J)
  Sigma_inv <- tryCatch(solve(Sigma_full), error = function(e) MASS::ginv(Sigma_full))
  U_vec <- as.numeric(U)
  quad_obs <- as.numeric(t(U_vec) %*% Sigma_inv %*% U_vec)

  # Null simulation: T_j per component, minp across components, max coordinate, overall quad
  T_j_sim <- matrix(0, nrow = J, ncol = n_sim)
  max_Z_sim <- numeric(n_sim)
  quad_sim <- numeric(n_sim)

  if (null_method == "permutation") {
    for (b in 1:n_sim) {
      z_p <- Z_cen[sample.int(n)]
      s_p <- phi_mean * z_p
      s_p <- sweep(s_p, 2, colMeans(s_p), "-")
      U_p <- crossprod(X_cen, s_p)
      for (j in 1:J) T_j_sim[j, b] <- as.numeric(t(U_p[, j]) %*% V_inv_list[[j]] %*% U_p[, j])
      max_Z_sim[b] <- max(abs(U_p / sd_mat))
      u_p_vec <- as.numeric(U_p)
      quad_sim[b] <- as.numeric(t(u_p_vec) %*% Sigma_inv %*% u_p_vec)
    }
  } else {
    # "gaussian" and "multiplier" draw from the same law N(0, Sigma_hat); the multiplier
    # form is used for both since pJ is small and it avoids a Cholesky failure mode.
    chunk <- 2000
    done <- 0
    while (done < n_sim) {
      m <- min(chunk, n_sim - done)
      G <- matrix(rnorm(n * m), nrow = n, ncol = m)
      U_sim_full <- crossprod(tilde_X, G)   # (pJ) x m
      cols <- (done + 1):(done + m)
      for (j in 1:J) {
        idx_j <- ((j-1)*p_valid + 1):(j*p_valid)
        U_j_sim <- U_sim_full[idx_j, , drop = FALSE]
        T_j_sim[j, cols] <- colSums(U_j_sim * (V_inv_list[[j]] %*% U_j_sim))
        seg <- apply(abs(U_j_sim / sd_mat[, j]), 2, max)
        max_Z_sim[cols] <- pmax(max_Z_sim[cols], seg)
      }
      quad_sim[cols] <- colSums(U_sim_full * (Sigma_inv %*% U_sim_full))
      done <- done + m
    }
  }

  comp_pvals_sim <- pchisq(T_j_sim, df = p_valid, lower.tail = FALSE)
  minp_sim <- apply(comp_pvals_sim, 2, min)

  return(list(
    quad = mean(quad_sim >= quad_obs),
    max = mean(max_Z_sim >= max_Z_obs),
    minp = mean(minp_sim <= minp_obs),
    components = comp_pvals
  ))
}

#' Compute P-Values for the Distributional Score Test
#'
#' Computes the Wasserstein-style (integral of sqrt) and Cramer-von Mises (integral)
#' test statistics for the ECDF-based score test, plus a max statistic, and obtains their
#' null distribution by one of three methods: simulation from the empirical Gaussian
#' approximation (via full-covariance Cholesky or the equivalent multiplier bootstrap),
#' or mark permutation.
#'
#' @param s_mat_mean The posterior mean of the score matrix (n x K).
#' @param t_grid The K evaluation points for the ECDF.
#' @param X_cen The full-rank centered design matrix.
#' @param p_valid The rank of the design matrix.
#' @param K The number of grid points.
#' @param n_sim Number of simulations (or permutations) to draw from the null.
#' @param null_method One of "gaussian" (default; simulate N(0, Sigma_hat) via Cholesky of the
#'   full (p*K) x (p*K) covariance — the original behavior), "multiplier" (Gaussian multiplier
#'   bootstrap; identical law, never forms the full covariance), or "permutation" (permute the
#'   Z_cen marks; requires `Z_cen` and `e_mat_mean`).
#' @param ad_weights Logical; if TRUE the CvM statistic uses Anderson-Darling weights 1/(F(1-F)).
#' @param Z_cen (Optional) Centered treatment vector; required for `null_method = "permutation"`.
#' @param e_mat_mean (Optional) Rao-Blackwellized centered-indicator matrix (n x K), i.e.
#'   `s_mat_mean` without the Z_cen factor; required for `null_method = "permutation"`.
#' @return A list with `wass`, `cvm`, and `max` p-values.
compute_distributional_pvalue_safe <- function(s_mat_mean, t_grid, X_cen, p_valid, K, n_sim = 10000,
                                               null_method = "gaussian", ad_weights = FALSE,
                                               Z_cen = NULL, e_mat_mean = NULL) {
  n <- nrow(X_cen)

  if (null_method == "permutation" && (is.null(Z_cen) || is.null(e_mat_mean))) {
    stop("null_method = 'permutation' requires Z_cen and e_mat_mean.")
  }

  # Compute the Rao-Blackwellized score vectors
  U_k_mean <- crossprod(X_cen, s_mat_mean)

  # CvM weights: uniform (default) or Anderson-Darling 1/(F(1-F)) on the quantile grid,
  # normalized to mean 1 so the statistic's scale is comparable across settings
  grid_probs <- seq(0.01, 0.99, length.out = K)
  if (ad_weights) {
    w_k <- 1 / (grid_probs * (1 - grid_probs))
    w_k <- w_k / mean(w_k)
  } else {
    w_k <- rep(1.0, K)
  }

  # Per-gridpoint empirical Huber-White covariance blocks V_k = crossprod(X_cen * s_k).
  # These are all the "gaussian" path's Sigma_full block-diagonals; the multiplier and
  # permutation paths never need the full (pK) x (pK) matrix.
  A_list <- vector("list", K)
  V_k_inv_list <- vector("list", K)
  sd_mat <- matrix(0, nrow = p_valid, ncol = K)
  T_k_obs <- numeric(K)
  for (k in 1:K) {
    A_k <- X_cen * s_mat_mean[, k]
    A_list[[k]] <- A_k
    V_k <- crossprod(A_k) + diag(1e-8, p_valid)
    V_k_inv <- tryCatch(solve(V_k), error = function(e) MASS::ginv(V_k))
    V_k_inv_list[[k]] <- V_k_inv
    sd_mat[, k] <- sqrt(diag(V_k))
    T_k_obs[k] <- as.numeric(t(U_k_mean[, k]) %*% V_k_inv %*% U_k_mean[, k])
  }

  t_diffs <- diff(t_grid)
  T_wass_obs <- sum(t_diffs * sqrt(pmax(T_k_obs[-K], 0)))
  T_cvm_obs <- mean(w_k * T_k_obs)
  max_Z_obs <- max(abs(U_k_mean / sd_mat))

  T_k_sim_matrix <- matrix(0, nrow = K, ncol = n_sim)
  max_Z_sim <- numeric(n_sim)

  if (null_method == "gaussian") {
    # Original path: build the full covariance of the score process and simulate from it.
    tilde_X <- matrix(0, nrow = n, ncol = p_valid * K)
    for (k in 1:K) {
      idx_k <- ((k-1)*p_valid + 1):(k*p_valid)
      tilde_X[, idx_k] <- A_list[[k]]
    }
    Sigma_full <- crossprod(tilde_X)
    Sigma_full <- Sigma_full + diag(1e-8, p_valid * K)

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

    for (k in 1:K) {
      idx_k <- ((k-1)*p_valid + 1):(k*p_valid)
      U_k_sim <- U_sim[idx_k, , drop = FALSE]
      T_k_sim_matrix[k, ] <- colSums(U_k_sim * (V_k_inv_list[[k]] %*% U_k_sim))
    }

    inv_sd_vec <- 1 / as.numeric(sd_mat)
    Z_sim <- U_sim * inv_sd_vec
    max_Z_sim <- apply(abs(Z_sim), 2, max)
  } else if (null_method == "multiplier") {
    # Gaussian multiplier bootstrap: U_sim = t(X_cen * s_k) %*% G, the same law as the
    # "gaussian" path but with no (pK)^2 matrix and no Cholesky. Processed in chunks.
    chunk <- 2000
    done <- 0
    while (done < n_sim) {
      m <- min(chunk, n_sim - done)
      G <- matrix(rnorm(n * m), nrow = n, ncol = m)
      cols <- (done + 1):(done + m)
      for (k in 1:K) {
        U_k_sim <- crossprod(A_list[[k]], G)   # p x m
        T_k_sim_matrix[k, cols] <- colSums(U_k_sim * (V_k_inv_list[[k]] %*% U_k_sim))
        seg <- apply(abs(U_k_sim / sd_mat[, k]), 2, max)
        max_Z_sim[cols] <- pmax(max_Z_sim[cols], seg)
      }
      done <- done + m
    }
  } else {
    # Mark permutation: permute Z_cen against the (fixed) averaged indicator scores.
    # Studentization (V_k_inv, sd_mat) is held at the observed values.
    for (b in 1:n_sim) {
      z_p <- Z_cen[sample.int(n)]
      s_p <- e_mat_mean * z_p
      U_p <- crossprod(X_cen, s_p)
      for (k in 1:K) {
        T_k_sim_matrix[k, b] <- as.numeric(t(U_p[, k]) %*% V_k_inv_list[[k]] %*% U_p[, k])
      }
      max_Z_sim[b] <- max(abs(U_p / sd_mat))
    }
  }

  T_wass_sim <- as.numeric(t_diffs %*% sqrt(pmax(T_k_sim_matrix[-K, , drop = FALSE], 0)))
  T_cvm_sim <- as.numeric(crossprod(w_k, T_k_sim_matrix)) / K

  pval_wass <- mean(T_wass_sim >= T_wass_obs)
  pval_cvm <- mean(T_cvm_sim >= T_cvm_obs)
  pval_max <- mean(max_Z_sim >= max_Z_obs)

  return(list(wass = pval_wass, cvm = pval_cvm, max = pval_max))
}
