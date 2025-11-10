#' Estimate Propensity Scores using a Single-Layer Neural Network
#'
#' @description
#' Fits a standard, single-hidden-layer feed-forward neural network to model the
#' probability of treatment assignment (propensity score), P(Z=1 | X).
#'
#' This function uses the R-native **`nnet`** package, avoiding external Python dependencies.
#' While highly stable, it is less flexible than modern deep learning frameworks like Keras,
#' as it is limited to one hidden layer.
#'
#' @param X_train A numeric matrix or data frame of predictor variables for the training set.
#' @param Z_train A numeric vector of binary treatment assignments (0 or 1) for the training set.
#' @param X_test (Optional) A numeric matrix or data frame of predictor variables for the test set.
#'   Must have the same columns in the same order as `X_train`.
#' @param size Number of neurons (units) in the single hidden layer. This is the primary parameter
#'   for controlling model complexity. Default is 10.
#' @param decay Weight decay parameter for regularization. This helps prevent overfitting by
#'   penalizing large weights. Increase for stronger regularization. Default is 0.01.
#' @param maxit Maximum number of iterations for training. Default is 200.
#' @param verbose Logical. If `TRUE`, training progress information is printed. Default is `FALSE`.
#'
#' @details
#' This function requires the `nnet` package. Input data `X_train` is automatically scaled
#' to the [0, 1] range, which is generally recommended for `nnet`. The treatment variable `Z_train`
#' is converted to a factor for classification.
#'
#' @return
#' A list containing:
#' \item{train}{A numeric vector of estimated propensity scores for the `X_train` data.}
#' \item{test}{A numeric vector of estimated propensity scores for the `X_test` data. Returns `NULL` if `X_test` is not provided.}
#'
#' @importFrom nnet nnet
#' @importFrom stats predict
#'
#' @export
#'
#' @examples
#' \dontrun{
#' # Generate synthetic data
#' set.seed(123)
#' n <- 500
#' p <- 5
#' X <- matrix(rnorm(n * p), ncol = p)
#' true_propensity <- pnorm(-0.5 + X[,1] + 0.5 * X[,2]^2)
#' Z <- rbinom(n, 1, true_propensity)
#'
#' # Split data
#' train_idx <- sample(1:n, 400)
#' X_train <- X[train_idx, ]
#' Z_train <- Z[train_idx]
#' X_test <- X[-train_idx, ]
#'
#' # Estimate propensity scores using nnet
#' nnet_scores <- estimate_propensity_nnet(
#'   X_train = X_train,
#'   Z_train = Z_train,
#'   X_test = X_test,
#'   size = 8, # Number of neurons in the hidden layer
#'   decay = 0.1
#' )
#'
#' # View a summary of the estimated scores
#' summary(nnet_scores$train)
#' }
estimate_propensity_nnet <- function(X_train, Z_train, X_test = NULL,
                                     size = 10, decay = 0.01, maxit = 200, verbose = FALSE) {
  
  
  # --- 2. Data Preprocessing: Min-Max Scaling [0, 1] ---
  # nnet often performs better with inputs scaled between 0 and 1.
  X_train_matrix <- data.matrix(X_train)
  mins <- apply(X_train_matrix, 2, min)
  maxs <- apply(X_train_matrix, 2, max)
  ranges <- maxs - mins
  # Avoid division by zero for constant columns
  ranges[ranges == 0] <- 1 
  
  X_train_scaled <- sweep(X_train_matrix, 2, mins, "-")
  X_train_scaled <- sweep(X_train_scaled, 2, ranges, "/")
  
  nnet_model <- nnet::nnet(
    x = X_train_scaled,
    y = Z_train,
    size = size,
    decay = decay,
    maxit = maxit,
    linout = FALSE, # Use logistic activation function for output
    trace = verbose # Suppress output unless requested
  )
  
  # --- 4. Generate Predictions ---
  # predict with type = "raw" gives probabilities for nnet classification models.
  propensity_train <- as.vector(stats::predict(nnet_model, newdata = X_train_scaled, type = "raw"))
  
  # Handle cases where nnet returns probabilities for both classes (if size=1 and 2 classes)
  if(is.matrix(propensity_train)) propensity_train <- propensity_train[,1]
  
  # Predictions for test data
  propensity_test <- NULL
  if (!is.null(X_test)) {
    # Apply the same scaling from the training set to the test set
    X_test_matrix <- data.matrix(X_test)
    X_test_scaled <- sweep(X_test_matrix, 2, mins, "-")
    X_test_scaled <- sweep(X_test_scaled, 2, ranges, "/")
    
    propensity_test <- as.vector(stats::predict(nnet_model, newdata = X_test_scaled, type = "raw"))
    if(is.matrix(propensity_test)) propensity_test <- propensity_test[,1]
  }
  
  return(list(train = propensity_train, test = propensity_test))
}