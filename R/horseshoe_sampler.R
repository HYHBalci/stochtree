#' @title Get Initial Values for the Horseshoe Sampler
#' @description A helper function to generate reasonable, data-driven starting
#'   values for the MCMC sampler by fitting a single joint linear model.
#' @param X A numeric matrix of covariates.
#' @param y A numeric vector of outcomes.
#' @param z A numeric vector of treatment assignments.
#' @param p_int_mu Number of interaction terms in the mu model.
#' @param p_int_tau Number of interaction terms in the tau model.
#' @return A list of initial values for the sampler.
#' @keywords internal
get_initial_values <- function(X, y, z, p_int_mu, p_int_tau) {
  n <- nrow(X)
  p <- ncol(X)
  
  # Create a data frame for the joint model
  # The formula y ~ . * z expands to y ~ X + z + X:z
  # This jointly estimates the prognostic and treatment effects.
  model_data <- as.data.frame(X)
  colnames(model_data) <- paste0("X", 1:p)
  model_data$y <- y
  model_data$z <- z
  
  # Construct the formula for the joint model
  # This will estimate main effects for X (mu), a main effect for z (tau intercept),
  # and interactions X:z (tau slopes)
  formula_str <- "y ~ z * (.)"
  # We remove y from the formula predictors
  formula_str <- sub("\\.", paste(colnames(model_data)[1:p], collapse = " + "), formula_str)
  formula <- stats::as.formula(formula_str)
  
  # Fit the joint linear model
  joint_fit <- tryCatch(stats::lm(formula, data = model_data), error = function(e) NULL)
  
  if (is.null(joint_fit)) {
    # Fallback if lm fails (e.g., due to perfect collinearity)
    warning("Initial value estimation via lm() failed. Using default zeros.")
    return(list(
      sigma_init = 1.0,
      alpha_mu_init = mean(y), beta_mu_init = rep(0, p), beta_int_mu_init = rep(0, p_int_mu),
      alpha_tau_init = 0.0, beta_tau_init = rep(0, p), beta_int_tau_init = rep(0, p_int_tau),
      tau_glob_mu_init = 0.1, tau_glob_tau_init = 0.1
    ))
  }
  
  # Extract coefficients
  all_coefs <- stats::coef(joint_fit)
  
  # Handle potential NA coefficients from lm (due to collinearity)
  all_coefs[is.na(all_coefs)] <- 0
  
  # Map coefficients to parameters
  alpha_mu_init <- all_coefs["(Intercept)"]
  beta_mu_init <- all_coefs[grepl("^X", names(all_coefs)) & !grepl(":", names(all_coefs))]
  
  alpha_tau_init <- ifelse("z" %in% names(all_coefs), all_coefs["z"], 0)
  beta_tau_init <- all_coefs[grepl(":z", names(all_coefs))]
  
  # Ensure correct length
  if(length(beta_mu_init) != p) beta_mu_init <- rep(0, p)
  if(length(beta_tau_init) != p) beta_tau_init <- rep(0, p)
  
  # Initialize higher-order interactions to zero for stability
  beta_int_mu_init <- rep(0, p_int_mu)
  beta_int_tau_init <- rep(0, p_int_tau)
  
  sigma_init <- summary(joint_fit)$sigma
  
  return(list(
    sigma_init = sigma_init,
    alpha_mu_init = alpha_mu_init,
    beta_mu_init = beta_mu_init,
    beta_int_mu_init = beta_int_mu_init,
    alpha_tau_init = alpha_tau_init,
    beta_tau_init = beta_tau_init,
    beta_int_tau_init = beta_int_tau_init,
    tau_glob_mu_init = 1, # Start with strong shrinkage
    tau_glob_tau_init = 1
  ))
}


#' Fit a heteroskedastic Bayesian linear regression model with a horseshoe prior
#'
#' This function fits a model of the form y = mu(x) + z * tau(x) + epsilon,
#' where mu(x) is the prognostic effect and tau(x) is the treatment effect.
#' Both mu(x) and tau(x) are linear models with coefficients given a horseshoe
#' prior. This allows for flexible modeling of treatment effect heterogeneity.
#'
#' @param X A numeric matrix or data frame of covariates.
#' @param y A numeric vector of outcomes.
#' @param z A numeric vector of treatment assignments (e.g., 0 or 1).
#' @param num_iterations Total number of MCMC iterations.
#' @param burn_in Number of burn-in iterations to discard.
#' @param mu_model A list of options for the mu(x) model component.
#' @param tau_model A list of options for the tau(x) model component.
#' @param propensity_separate (logical) Whether to separately model the propensity score.
#' @param alpha_prior_sd (double) The standard deviation of the prior on the intercept terms.
#' @param initial_values A list of starting values for the sampler. If NULL (default), data driven initiliaziion will be made. 
#' @param standardize_cov test
#' @param interaction_rule test
#'
#' @return A list object of class `stochtree_hs` containing the posterior samples for all model parameters.
#' @export
horseshoe_sampler <- function(X, y, z,
                              num_iterations = 2000, burn_in = 1000,
                              mu_model = list(gibbs = TRUE, global_shrink = TRUE, unlink = FALSE, p_int = 0),
                              tau_model = list(gibbs = TRUE, global_shrink = TRUE, unlink = FALSE, p_int = 0),
                              propensity_separate = FALSE,
                              alpha_prior_sd = 10.0,
                              initial_values = NULL, standardize_cov = F, interaction_rule = "continuous_or_binary") {
  
  handled_data_list <- standardize_X_by_index(X, process_data = standardize_cov, interaction_rule = interaction_rule, cat_coding_method = "difference")
  
  X_train <- handled_data_list$X_final
  print(dim(X_train))
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
  are_continuous <- as.vector(as.integer(boolean_continuous*1))
  if(propensity_separate){
    num_burnin <- 10
    num_total <- 50
    bart_model_propensity <- bart(X_train = X, y_train = as.numeric(z), X_test = NULL, 
                                  num_gfr = num_total, num_burnin = 0, num_mcmc = 0)
    propensity_train <- rowMeans(bart_model_propensity$y_hat_train[,(num_burnin+1):num_total])
    print('propensity scores calculated')
  } else{
    print('Running model without propensity as a covariate')
    propensity_train <- numeric(0)
  }
  if (is.null(initial_values)) {
    initial_values <- get_initial_values(X, y, z, mu_model$p_int, tau_model$p_int)
  }
  
  # 3. Call the C++ backend function with initial values
  samples <- run_mcmc_sampler_2(
    X_r = X,
    Y_r = y,
    Z_r = z,
    are_continuous_r = are_continuous,
    propensity_scores_r = propensity_scores,
    num_iterations = num_iterations,
    burn_in = burn_in,
    mu_gibbs = mu_model$gibbs,
    mu_global_shrink = mu_model$global_shrink,
    mu_unlink = mu_model$unlink,
    mu_p_int = p_int,
    tau_gibbs = tau_model$gibbs,
    tau_global_shrink = tau_model$global_shrink,
    tau_unlink = tau_model$unlink,
    tau_p_int = p_int,
    propensity_separate = propensity_separate,
    alpha_prior_sd = alpha_prior_sd,
    sigma_init = initial_values$sigma_init,
    alpha_mu_init = initial_values$alpha_mu_init,
    beta_mu_init = initial_values$beta_mu_init,
    beta_int_mu_init = initial_values$beta_int_mu_init,
    alpha_tau_init = initial_values$alpha_tau_init,
    beta_tau_init = initial_values$beta_tau_init,
    beta_int_tau_init = initial_values$beta_int_tau_init,
    tau_glob_mu_init = initial_values$tau_glob_mu_init,
    tau_glob_tau_init = initial_values$tau_glob_tau_init
  )
  
  # 4. Format the output
  class(samples) <- "stochtree_hs"
  return(samples)
}
