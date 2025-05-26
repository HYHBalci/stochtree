interaction_pairs <- function(num_covariates) {
  combn(1:num_covariates, 2)
}
#' Shapley calculator for x_{i}
#'
#' Shapley calculator.
#'
#' @param index: row index in X for which to compute Shapley values
#' @param X: centered covariate matrix (n x p)
#' @param beta_post: posterior samples of beta (chains x samples x p)
#' @param beta_int_post: posterior samples of interaction effects (chains x samples x p_int)
#' @return The posterior samples of the shapley values, and a dataframe with summary statistics. 
#' @export
shapley <- function(index, X, beta_post, beta_int_post) {
  num_covariates <- ncol(X)
  x_star <- X[index, ]
  cov_X <- cov(X)
  ipairs <- combn(1:num_covariates, 2)
  p_int <- ncol(ipairs)
  
  beta_samples <- do.call(rbind, lapply(1:dim(beta_post)[1], function(chain) beta_post[chain, , ]))
  beta_int_samples <- do.call(rbind, lapply(1:dim(beta_int_post)[1], function(chain) beta_int_post[chain, , ]))
  
  num_samples <- nrow(beta_samples)
  shapley_vals <- matrix(0, nrow = num_samples, ncol = num_covariates)
  
  for (j in 1:num_covariates) {
    main_term <- beta_samples[, j] * x_star[j]
    involved <- which(ipairs[1, ] == j | ipairs[2, ] == j)
    interaction_sum <- numeric(num_samples)
    
    for (m in involved) {
      k <- if (ipairs[1, m] == j) ipairs[2, m] else ipairs[1, m]
      delta <- x_star[j] * x_star[k] - cov_X[j, k]
      interaction_sum <- interaction_sum + beta_int_samples[, m] * delta
    }
    
    shapley_vals[, j] <- main_term + 0.5 * interaction_sum
  }
  
  # Convert to long-format tibble
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
#'
#' @return A data frame with Shapley values for each selected observation and feature.
#' @export
compute_shapley_all <- function(X, beta_post, beta_int_post, indices = NULL) {
  if (is.null(indices)) indices <- sample(1:nrow(X), 100)
  all_shapleys <- purrr::map_dfr(indices, ~shapley(.x, X, beta_post, beta_int_post))
  return(all_shapleys)
}

#' Plot Shapley Value Summaries for Multiple Observations
#'
#' Computes and visualizes Shapley value summaries across selected observations.
#'
#' @param X A centered covariate matrix (n x p).
#' @param beta_post A 3D array of posterior main effect coefficients: (chains x samples x p).
#' @param beta_int_post A 3D array of posterior interaction effect coefficients: (chains x samples x p_int).
#' @param indices Optional vector of row indices in `X` to include. Defaults to 100 random samples.
#'
#' @return A `ggplot2` object visualizing the distribution of Shapley values for each feature.
#' @export
plot_shapley_summary <- function(X, beta_post, beta_int_post, indices = NULL) {
  shapley_all_df <- compute_shapley_all(X, beta_post, beta_int_post, indices)
  shapley_summary <- shapley_all_df %>%
    group_by(feature, obs) %>%
    summarise(
      mean = mean(shapley),
      .groups = "drop"
    )
  
  ggplot(shapley_summary, aes(x = mean, y = feature)) +
    geom_jitter(width = 0, height = 0.2, alpha = 0.6) +
    stat_summary(fun = median, geom = "point", shape = 18, size = 3, color = "red") +
    labs(
      title = "Shapley Value Distribution Across Observations",
      x = "Shapley Value (ϕ_ij)",
      y = "Feature"
    ) +
    theme_minimal()
}
compute_shapley_all <- function(X, beta_post, beta_int_post, indices = NULL) {
  if (is.null(indices)) indices <- sample(1:nrow(X), 100)
  all_shapleys <- purrr::map_dfr(indices, ~shapley(.x, X, beta_post, beta_int_post))
  return(all_shapleys)
}

plot_shapley_summary <- function(X, beta_post, beta_int_post, indices = NULL) {
  shapley_all_df <- compute_shapley_all(X, beta_post, beta_int_post, indices)
  shapley_summary <- shapley_all_df %>%
    group_by(feature, obs) %>%
    summarise(
      mean = mean(shapley),
      .groups = "drop"
    )
  
  ggplot(shapley_summary, aes(x = mean, y = feature)) +
    geom_jitter(width = 0, height = 0.2, alpha = 0.6) +
    stat_summary(fun = median, geom = "point", shape = 18, size = 3, color = "red") +
    labs(
      title = "Shapley Value Distribution Across Observations",
      x = "Shapley Value (ϕ_ij)",
      y = "Feature"
    ) +
    theme_minimal()
}

#' Plot Shapley Values vs. Covariate Values
#'
#' Plots the relationship between covariate values and their corresponding Shapley values
#' for a subset of observations.
#'
#' @param X A centered covariate matrix (n x p).
#' @param beta_post A 3D array of posterior main effect coefficients: (chains x samples x p).
#' @param beta_int_post A 3D array of posterior interaction effect coefficients: (chains x samples x p_int).
#' @param indices Optional vector of row indices to include (default: 100 random rows).
#' @param feature Optional feature index or name to plot. If NULL, plots all features in a grid.
#'
#' @return A ggplot object or a facetted plot.
#' @export
plot_shapley_vs_covariate <- function(X, beta_post, beta_int_post, indices = NULL, feature = NULL) {
  if (is.null(indices)) indices <- sample(1:nrow(X), 100)
  shapley_list <- purrr::map_dfr(indices, ~{
    res <- shapley(.x, X, beta_post, beta_int_post)
    tibble::tibble(
      obs = .x,
      feature = colnames(X),
      x_value = X[.x, ],
      shapley = res$summary[, "mean"]
    )
  })
  
  if (!is.null(feature)) {
    # If feature is index, convert to name
    if (is.numeric(feature)) {
      feature <- colnames(X)[feature]
    }
    shapley_list <- dplyr::filter(shapley_list, feature == !!feature)
  }
  
  p <- ggplot(shapley_list, aes(x = x_value, y = shapley)) +
    geom_point(alpha = 0.6) +
    geom_smooth(method = "loess", se = FALSE, color = "red", linetype = "dashed") +
    labs(
      title = "Shapley Value vs. Covariate",
      x = "Covariate Value",
      y = "Shapley Value"
    ) +
    theme_minimal()
  
  if (is.null(feature)) {
    p <- p + facet_wrap(~feature, scales = "free_x")
  }
  
  return(p)
}
