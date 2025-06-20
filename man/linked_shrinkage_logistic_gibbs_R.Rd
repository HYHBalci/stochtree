% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/logistic_gibbs.R
\name{linked_shrinkage_logistic_gibbs_R}
\alias{linked_shrinkage_logistic_gibbs_R}
\title{Linked Shrinkage Logistic Gibbs Sampler (Pure R Version)}
\usage{
linked_shrinkage_logistic_gibbs_R(
  y_vec,
  X_mat,
  Z_vec,
  n_iter,
  burn_in,
  alpha_prior_sd = 10,
  aleph_prior_sd = 10,
  init_alpha = 0,
  init_aleph = 0,
  seed = 1848
)
}
\arguments{
\item{y_vec}{Integer vector. Binary response variable (must contain only 0s and 1s).}

\item{X_mat}{Numeric matrix. Covariate matrix for the prognostic part and for
forming interactions with the treatment indicator \code{Z_vec}.
Each row is an observation, each column is a covariate.}

\item{Z_vec}{Numeric vector. Treatment indicator (e.g., 0 or 1) or other key
variable modifying the effect of covariates in \code{X_mat}. Must have the same
length as \code{y_vec} and the number of rows in \code{X_mat}.}

\item{n_iter}{Integer. Total number of MCMC iterations.}

\item{burn_in}{Integer. Number of initial MCMC iterations to discard as burn-in.
Must be less than \code{n_iter}.}

\item{alpha_prior_sd}{Numeric scalar. Standard deviation for the normal prior
of the global intercept \code{alpha}. Default is 10.0.}

\item{aleph_prior_sd}{Numeric scalar. Standard deviation for the normal prior
of the intercept \code{aleph} for the treatment-specific modifier part. Default is 10.0.}

\item{init_alpha}{Numeric scalar. Initial value for the global intercept \code{alpha}.
Default is 0.0.}

\item{init_aleph}{Numeric scalar. Initial value for the intercept \code{aleph} of the
treatment-specific modifier. Default is 0.0.}

\item{seed}{Integer or NULL. Seed for R's random number generator to ensure
reproducibility. Default is 1848.}
}
\value{
A list containing MCMC samples (after burn-in) for the model parameters:
\item{alpha}{Numeric vector of samples for the global intercept \code{alpha}.}
\item{beta}{Numeric matrix of samples for the main prognostic effects \code{beta_vec}. Each row is an iteration, each column corresponds to a covariate in \code{X_mat}.}
\item{beta_interaction}{Numeric matrix of samples for the prognostic interaction effects \code{beta_interaction_vec} (pairwise interactions of \code{X_mat}). Each row is an iteration. Columns correspond to interaction pairs. Will have 0 columns if no interactions are possible (i.e., \code{ncol(X_mat) < 2}).}
\item{aleph}{Numeric vector of samples for the intercept \code{aleph} of the treatment-specific modifier.}
\item{gamma}{Numeric matrix of samples for the coefficients \code{gamma_vec} of Z\emph{X terms. Each row is an iteration.}
\item{gamma_int}{Numeric matrix of samples for the coefficients \code{gamma_int_vec} of Z}X*X interaction terms. Each row is an iteration. Will have 0 columns if no interactions are possible.}
\item{tau_j}{Numeric matrix of samples for the local shrinkage parameters \code{tau_j_params_vec} for \code{beta_vec} coefficients.}
\item{tau_int_fixed_value}{Numeric scalar, the fixed value of \code{tau_int_param} (which is 1.0 in this version).}
\item{lambda_gamma}{Numeric matrix of samples for the local Horseshoe shrinkage parameters for \code{gamma_vec} coefficients.}
\item{lambda_g_int}{Numeric matrix of samples for the local Horseshoe shrinkage parameters for \code{gamma_int_vec} coefficients. Will have 0 columns if no interactions are possible.}
\item{tau_hs_combined}{Numeric vector of samples for the shared global Horseshoe shrinkage parameter for \code{gamma_vec} and \code{gamma_int_vec} coefficients.}
}
\description{
Performs MCMC sampling for a Bayesian logistic regression model. This is a
pure R implementation. The model includes a prognostic part with linked
shrinkage priors for main effects (\code{beta}) and their interactions
(\code{beta_interaction}). It also includes a treatment-specific modifier part,
where the treatment indicator \code{Z_vec} multiplies an intercept term (\code{aleph}),
main covariate effects (\code{gamma_vec}), and two-way covariate interaction
effects (\code{gamma_int_vec}). The \code{gamma_vec} and \code{gamma_int_vec} coefficients
are given Horseshoe priors with a shared global shrinkage parameter
(\code{tau_hs_combined}). The \code{tau_int_param} for \code{beta_interaction} terms
is fixed at 1.0 in this version.

This function relies on the \code{BayesLogit} package for Polya-Gamma sampling
and the \code{MASS} package for multivariate normal sampling. Ensure these are
installed and listed in your package's DESCRIPTION file if applicable.
}
\examples{
\dontrun{
# Generate some dummy data
N <- 100 # Number of observations
p <- 3   # Number of covariates, keep small for R version speed
X_data <- matrix(rnorm(N * p), nrow = N, ncol = p)
colnames(X_data) <- paste0("X", 1:p)
Z_data <- sample(0:1, N, replace = TRUE)

# True parameters (example)
true_alpha <- 0.2
true_beta <- rnorm(p, 0, 0.5)
true_beta_int_effects <- if (p >= 2) {
  n_int <- ncol(combn(p, 2))
  rnorm(n_int, 0, 0.2)
} else { numeric(0) }
true_aleph <- 0.5
true_gamma <- rnorm(p, 0, 0.4)
true_gamma_int_effects <- if (p >= 2) {
 n_int <- ncol(combn(p, 2))
 rnorm(n_int, 0, 0.1)
} else { numeric(0) }

# Construct linear predictor
lin_pred <- true_alpha + X_data \%*\% true_beta
if (p >= 2) {
  pairs <- combn(1:p, 2)
  for (k_int in 1:ncol(pairs)) {
    lin_pred <- lin_pred + true_beta_int_effects[k_int] * X_data[,pairs[1,k_int]] * X_data[,pairs[2,k_int]]
  }
}
treatment_part <- true_aleph + X_data \%*\% true_gamma
if (p >= 2) {
  pairs <- combn(1:p, 2)
  for (k_int in 1:ncol(pairs)) {
    treatment_part <- treatment_part + true_gamma_int_effects[k_int] * X_data[,pairs[1,k_int]] * X_data[,pairs[2,k_int]]
  }
}
lin_pred <- lin_pred + Z_data * treatment_part

prob_true <- 1 / (1 + exp(-lin_pred))
y_data <- rbinom(N, 1, prob_true)

# Run the R version of the sampler (will be slow for many iterations)
fit_R <- linked_shrinkage_logistic_gibbs_R(
  y_vec = y_data,
  X_mat = X_data,
  Z_vec = Z_data,
  n_iter = 500,  # Keep low for example
  burn_in = 200,
  seed = 123
)

# Summarize some results
print(paste("Alpha mean:", mean(fit_R$alpha)))
if (p > 0) {
  print("Beta means:")
  print(colMeans(fit_R$beta))
}
if (length(fit_R$beta_interaction) > 0 && ncol(fit_R$beta_interaction) > 0) {
  print("Beta Interaction means:")
  print(colMeans(fit_R$beta_interaction))
}
plot(fit_R$alpha, type = "l", main = "Trace plot for alpha (R version)")
}
}
