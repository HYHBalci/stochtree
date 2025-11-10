#' Estimate Propensity Scores using a Neural Network
#'
#' @description
#' Fits a flexible, feed-forward neural network to model the probability of
#' treatment assignment (the propensity score), P(Z=1 | X). This serves as a
#' powerful alternative to traditional models like logistic regression, capable
#' of capturing complex non-linearities and interactions among covariates.
#'
#' The function handles data scaling internally and uses early stopping to
#' prevent overfitting based on validation loss.
#'
#' @param X_train A numeric matrix or data frame of predictor variables for the training set.
#' @param Z_train A numeric vector of binary treatment assignments (0 or 1) for the training set.
#' @param X_test (Optional) A numeric matrix or data frame of predictor variables for the test set.
#'   Must have the same columns in the same order as `X_train`.
#' @param architecture A function that defines the Keras model architecture. The function
#'   must accept one argument, `input_dim`, which is the number of input features.
#'   If `NULL`, a default flexible architecture is used (see details).
#' @param learning_rate The learning rate for the Adam optimizer. Default is 0.001.
#' @param epochs Number of training epochs (passes through the entire dataset). Default is 50.
#' @param batch_size Number of samples processed before updating model weights. Default is 32.
#' @param validation_split Fraction of the training data to be used as validation data for
#'   monitoring early stopping. Default is 0.2 (i.e., 20%).
#' @param verbose Controls training verbosity. 0 = silent, 1 = progress bar. Default is 0.
#'
#' @details
#' This function requires the `keras` package and a configured TensorFlow backend.
#'
#' The default neural network architecture is a sequential model with:
#' 1. A dense layer with 128 units and ReLU activation.
#' 2. A batch normalization layer.
#' 3. A dropout layer with a rate of 0.3.
#' 4. A dense layer with 64 units and ReLU activation.
#' 5. A batch normalization layer.
#' 6. A dropout layer with a rate of 0.2.
#' 7. A final dense output layer with 1 unit and a sigmoid activation for probability output.
#'
#' The early stopping callback monitors `val_loss` and restores the best model weights.
#'
#' @return
#' A list containing:
#' \item{train}{A numeric vector of estimated propensity scores for the `X_train` data.}
#' \item{test}{A numeric vector of estimated propensity scores for the `X_test` data. Returns `NULL` if `X_test` is not provided.}
#'
#' @importFrom keras keras_model_sequential layer_dense layer_batch_normalization layer_dropout optimizer_adam compile fit callback_early_stopping
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
#' # Estimate propensity scores using the NN with explicit parameters
#' nn_scores <- estimate_propensity_nn(
#'   X_train = X_train,
#'   Z_train = Z_train,
#'   X_test = X_test,
#'   epochs = 40,
#'   batch_size = 32,
#'   verbose = 1 # Show progress bar during training
#' )
#'
#' # View a summary of the estimated scores
#' summary(nn_scores$train)
#' }
estimate_propensity_nn <- function(X_train, Z_train, X_test = NULL, architecture = NULL,
                                   learning_rate = 0.001, epochs = 50, batch_size = 32,
                                   validation_split = 0.2, verbose = 0) {
  
  X_train_matrix <- data.matrix(X_train)
  X_train_scaled <- scale(X_train_matrix)
  
  train_center <- attr(X_train_scaled, "scaled:center")
  train_scale <- attr(X_train_scaled, "scaled:scale")
  input_dim <- ncol(X_train_scaled)
  
  if (is.null(architecture)) {
      architecture <- function(input_dim) {
      keras::keras_model_sequential() %>%
        keras::layer_dense(units = 128, activation = "relu", input_shape = input_dim) %>%
        keras::layer_batch_normalization() %>%
        keras::layer_dropout(rate = 0.3) %>%
        keras::layer_dense(units = 64, activation = "relu") %>%
        keras::layer_batch_normalization() %>%
        keras::layer_dropout(rate = 0.2) %>%
        keras::layer_dense(units = 1, activation = "sigmoid")
    }
  }
  
  model <- architecture(input_dim)
  
  model %>% keras::compile(
    loss = "binary_crossentropy",
    optimizer = keras::optimizer_adam(learning_rate = learning_rate),
    metrics = c("accuracy")
  )
  
  early_stop <- keras::callback_early_stopping(monitor = "val_loss", patience = 10, restore_best_weights = TRUE)
  
  history <- model %>% keras::fit(
    x = X_train_scaled,
    y = Z_train,
    epochs = epochs,
    batch_size = batch_size,
    validation_split = validation_split,
    callbacks = list(early_stop),
    verbose = verbose
  )
  
  propensity_train <- as.vector(keras::predict(model, X_train_scaled))
  
  propensity_test <- NULL
  if (!is.null(X_test)) {
    X_test_matrix <- data.matrix(X_test)
    X_test_scaled <- scale(X_test_matrix, center = train_center, scale = train_scale)
    propensity_test <- as.vector(keras::predict(model, X_test_scaled))
  }
  
  return(list(train = propensity_train, test = propensity_test))
}