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


#' Shapley Value Calculator for x
#'
#' This function calculates Shapley values for a given observation, considering
#' main effects and specified pairwise interaction effects.
#'
#' @param index The row index in the data matrix `X` for which to compute Shapley values.
#' @param X A centered covariate matrix (n x p).
#' @param beta_post Posterior samples of main effects. Can be a 3D array 
#'   (chains x samples x p) or a 2D matrix (total_samples x p).
#' @param beta_int_post Posterior samples of interaction effects. Can be a 3D array
#'   (chains x samples x p_int) or a 2D matrix (total_samples x p_int).
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
  
  # Check format of posteriors (3D array or 2D matrix)
  is_3D <- length(dim(beta_post)) == 3
  
  # --- Validate Interaction Dimensions ---
  num_beta_int_coeffs <- 0
  if (!is.null(beta_int_post)) {
    int_dims <- dim(beta_int_post)
    if (is_3D) {
      # 3D format: chains x samples x p_int
      if (length(int_dims) != 3) stop("beta_int_post is not 3D, but beta_post is.")
      num_beta_int_coeffs <- int_dims[3]
    } else {
      # 2D format: samples x p_int
      if (length(int_dims) != 2) stop("beta_int_post is not 2D, but beta_post is.")
      num_beta_int_coeffs <- int_dims[2]
    }
  }
  
  # Handle case where p_int is 0
  if (p_int == 0) {
    # If no interactions are expected, check that none were provided
    if (num_beta_int_coeffs > 0) {
      stop(paste("Dimension mismatch: The number of interaction pairs is 0,",
                 "but beta_int_post has", num_beta_int_coeffs, "coefficients."))
    }
  } else if (num_beta_int_coeffs != p_int) {
    # If interactions are expected, check that the dimensions match
    stop(paste("Dimension mismatch: The number of interaction pairs is", p_int,
               "but beta_int_post has", num_beta_int_coeffs, "coefficients."))
  }
  
  # --- Prepare Posterior Samples ---
  if (is_3D) {
    # 3D: Combine chains
    beta_samples <- do.call(rbind, lapply(1:dim(beta_post)[1], function(chain) beta_post[chain, , ]))
  } else {
    # 2D: Already in correct format
    beta_samples <- beta_post
  }
  
  # Prepare interaction samples, handling the case of zero interactions.
  if (p_int > 0) {
    if (is_3D) {
      beta_int_samples <- do.call(rbind, lapply(1:dim(beta_int_post)[1], function(chain) beta_int_post[chain, , ]))
    } else {
      beta_int_samples <- beta_int_post
    }
  } else {
    # Create empty matrix with correct number of rows (samples)
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

#' Plot Shapley Value Distributions
#'
#' Creates a summary violin and boxplot of Shapley value distributions
#' for each feature, aggregated across all posterior samples and observations.
#'
#' Features are automatically ordered by their mean absolute Shapley value,
#' placing the most important features at the top of the plot.
#'
#' @param shapley_data A long-format data frame, typically the output of
#'   `compute_shapley_all()`. Must contain 'feature' and 'shapley' columns.
#' @param ... Additional arguments passed to the `aes()` call in ggplot.
#' @return A ggplot object.
#' @import ggplot2
#' @importFrom forcats fct_reorder
#' @importFrom rlang .data
#' @export
plot_shapley_summary <- function(shapley_data, ...) {
  
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Package 'ggplot2' needed for this function to work. Please install it.", call. = FALSE)
  }
  if (!requireNamespace("forcats", quietly = TRUE)) {
    stop("Package 'forcats' needed for this function to work. Please install it.", call. = FALSE)
  }
  
  p <- ggplot2::ggplot(
    shapley_data,
    ggplot2::aes(
      x = .data$shapley,
      # Reorder features by mean absolute shapley value
      y = forcats::fct_reorder(.data$feature, .data$shapley, .fun = function(x) mean(abs(x))),
      fill = .data$feature,
      ...
    )
  ) +
    ggplot2::geom_violin(trim = FALSE, alpha = 0.6, show.legend = FALSE) +
    ggplot2::geom_boxplot(
      width = 0.15,
      fill = "white",
      outlier.shape = NA,
      show.legend = FALSE
    ) +
    ggplot2::geom_vline(
      xintercept = 0,
      linetype = "dashed",
      color = "black",
      linewidth = 0.7
    ) +
    ggplot2::labs(
      title = "Shapley Value Distribution per Feature",
      subtitle = "Aggregated across posterior samples and selected observations",
      x = "Shapley Value (Contribution to Prediction)",
      y = "Feature"
    ) +
    ggplot2::theme_minimal(base_size = 14) +
    ggplot2::theme(
      panel.grid.major.y = ggplot2::element_blank(),
      panel.grid.minor.y = ggplot2::element_blank()
    )
  
  return(p)
}

