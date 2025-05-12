generate_interaction_matrix <- function(X) {
  p_mod <- ncol(X)  
  p_int <- (p_mod * (p_mod - 1)) / 2 
  interaction_matrix <- matrix(NA, nrow = nrow(X), ncol = p_int)
  
  int_pairs <- list()  
  col_idx <- 1 
  
  for (i in 1:(p_mod - 1)) {
    for (j in (i + 1):p_mod) {  
      interaction_matrix[, col_idx] <- X[, i] * X[, j]
      int_pairs[[col_idx]] <- c(i, j)  
      col_idx <- col_idx + 1
    }
  }
  
  col_names <- sapply(int_pairs, function(pair) {
    paste0("X", pair[1], "*X", pair[2])  
  })
  
  interaction_matrix <- as.data.frame(interaction_matrix)
  colnames(interaction_matrix) <- col_names
  
  return(interaction_matrix)
}

predict_interaction_lm <- function(X, coef) {
  interaction_terms <- generate_interaction_matrix(X)
  full_design_matrix <- as.matrix(cbind(1, X, interaction_terms))
  
  return(as.vector(full_design_matrix %*% coef))  
}

