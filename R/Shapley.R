#' Generate Interaction Pairs Based on a Boolean Vector
#'
#' This function creates a matrix of pairwise interaction indices, but only includes
#' pairs where at least one of the variables is marked as TRUE in the boolean vector.
#'
#' @param num_covariates The total number of covariates.
#' @param boolean_vector A logical vector of length `num_covariates`. An interaction is included if at least one of the covariates is TRUE
#' @return A 2-row matrix where each column is a pair of interacting variable indices.
interaction_pairs <- function(num_covariates, boolean_vector) {
  interaction_list <- list()
  if (num_covariates > 1) {
    for (j in 1:(num_covariates - 1)) {
      for (k in (j + 1):num_covariates) {
        if (boolean_vector[j] || boolean_vector[k]) {
          interaction_list[[length(interaction_list) + 1]] <- c(j, k)
        }
      }
    }
  }
  
  # Handle the edge case where no interactions are selected to avoid errors.
  if (length(interaction_list) == 0) {
    return(matrix(numeric(0), nrow = 2, ncol = 0))
  }
  
  # Return as a 2-row matrix, similar to the output of combn.
  return(do.call(cbind, interaction_list))
}


#' Shapley Value Calculator for x_{i}
#'
#' This function calculates Shapley values for a given observation, considering
#' main effects and specified pairwise interaction effects.
#'
#' @param index The row index in the data matrix `X` for which to compute Shapley values.
#' @param X A centered covariate matrix (n x p).
#' @param beta_post Posterior samples of main effects (chains x samples x p).
#' @param beta_int_post Posterior samples of interaction effects (chains x samples x p_int).
#'   The dimension p_int must correspond to the number of pairs generated based on `boolean_interaction`.
#' @param boolean_interaction A logical vector indicating which variables are allowed to have interactions.
#' @return A long-format tibble containing the posterior samples of the Shapley values.
#' @importFrom tibble as_tibble
#' @importFrom tidyr pivot_longer
#' @importFrom dplyr %>%
#' @export
shapley <- function(index, X, beta_post, beta_int_post, boolean_interaction) {
  num_covariates <- ncol(X)
  x_star <- X[index, ]
  cov_X <- cov(X)
  
  ipairs <- interaction_pairs(num_covariates, boolean_interaction)
  p_int <- ncol(ipairs)
  
  num_beta_int_coeffs <- if (!is.null(beta_int_post)) dim(beta_int_post)[3] else 0
  if (num_beta_int_coeffs != p_int) {
    stop(paste("Dimension mismatch: The number of interaction pairs is", p_int,
               "but beta_int_post has", num_beta_int_coeffs, "coefficients."))
  }

  beta_samples <- do.call(rbind, lapply(1:dim(beta_post)[1], function(chain) beta_post[chain, , ]))
  
  # Prepare interaction samples, handling the case of zero interactions.
  if (p_int > 0) {
    beta_int_samples <- do.call(rbind, lapply(1:dim(beta_int_post)[1], function(chain) beta_int_post[chain, , ]))
  } else {
    beta_int_samples <- matrix(0, nrow = nrow(beta_samples), ncol = 0)
  }
  
  num_samples <- nrow(beta_samples)
  shapley_vals <- matrix(0, nrow = num_samples, ncol = num_covariates)
  
  for (j in 1:num_covariates) {
    main_term <- beta_samples[, j] * x_star[j]
    interaction_sum <- numeric(num_samples)
    
    # Calculate interaction sum only if interactions exist for this variable.
    if (p_int > 0) {
      involved <- which(ipairs[1, ] == j | ipairs[2, ] == j)
      
      if (length(involved) > 0) {
        for (m in involved) {
          k <- if (ipairs[1, m] == j) ipairs[2, m] else ipairs[1, m]
          delta <- x_star[j] * x_star[k] - cov_X[j, k]
          interaction_sum <- interaction_sum + beta_int_samples[, m] * delta
        }
      }
    }
    
    shapley_vals[, j] <- main_term + 0.5 * interaction_sum
  }
  colnames(shapley_vals) <- make.names(colnames(X), unique = TRUE)
  shapley_df <- as_tibble(shapley_vals)
  shapley_df$sample <- 1:num_samples
  
  shapley_long <- pivot_longer(shapley_df, -sample, names_to = "feature", values_to = "shapley")
  shapley_long$obs <- index
  
  return(shapley_long)
}

#' Compute Shapley Values for Multiple Observations
#'
#' Computes Shapley values for a subset of observations in the dataset.
#'
#' @param X A centered covariate matrix (n x p) where each row is an observation and each column is a feature.
#' @param beta_post A 3D array of posterior main effect coefficients: (chains x samples x p).
#' @param beta_int_post A 3D array of posterior interaction effect coefficients: (chains x samples x p_int).
#' @param indices Optional vector of row indices from `X` for which to compute Shapley values. If `NULL`, 100 random indices are sampled.
#' @param boolean_interaction Logical vector indicating if variable i should be allowed to interact with a variable for which boolean_interaction[j] == FALSE
#'
#' @return A data frame with Shapley values for each selected observation and feature.
#' @export
compute_shapley_all <- function(X, beta_post, beta_int_post, indices = NULL, boolean_interaction) {
  if (is.null(indices)) indices <- sample(1:nrow(X), 100)
  all_shapleys <- purrr::map_dfr(indices, ~shapley(.x, X, beta_post, beta_int_post, boolean_interaction))
  return(all_shapleys)
}

