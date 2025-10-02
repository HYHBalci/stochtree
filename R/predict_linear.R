generate_interaction_matrix <- function(X, X_final_var_info, interaction_rule, propensity_seperate = FALSE) {
  
  p_mod <- ncol(X)
  if(interaction_rule == 'continuous'){
    boolean_continuous <- as.vector(X_final_var_info$is_continuous)
  } else if(interaction_rule == 'continuous_or_binary'){
    boolean_continuous <- as.vector(X_final_var_info$is_continuous) + as.vector(X_final_var_info$is_binary)
  } else{ 
    boolean_continuous <- as.vector(X_final_var_info$is_continuous) + as.vector(X_final_var_info$is_binary) + as.vector(X_final_var_info$is_categorical)
  }
  is_continuous_map_final_X <- as.logical(boolean_continuous)
  
  interaction_cols_list <- list()
  int_pairs <- list()
  col_idx_counter <- 1
  
  if (p_mod > 1) {
    for (i in 1:(p_mod - 1)) {
      for (j in (i + 1):p_mod) {
        if ((propensity_seperate == "tau") && (j == p_mod)) {
          next 
        }
        if (is_continuous_map_final_X[i] || is_continuous_map_final_X[j]) {
          interaction_cols_list[[col_idx_counter]] <- X[, i] * X[, j]
          int_pairs[[col_idx_counter]] <- c(i, j) 
          col_idx_counter <- col_idx_counter + 1
        }
      }
    }
  }
  
  if (length(interaction_cols_list) > 0) {
    interaction_matrix <- as.matrix(as.data.frame(interaction_cols_list)) 
    
    col_names <- sapply(int_pairs, function(pair) {
      name_i <- X_final_var_info$col_name_final[pair[1]]
      name_j <- X_final_var_info$col_name_final[pair[2]]
      paste0(name_i, "*", name_j)
    })
    colnames(interaction_matrix) <- col_names
  } else {
    interaction_matrix <- matrix(NA, nrow = nrow(X), ncol = 0) 
  }
  
  return(interaction_matrix) # This now correctly returns a matrix
}

predict_interaction_lm <- function(X, coef, X_final_var_info, interaction_rule, propensity_seperate = "none") {
  interaction_terms <- generate_interaction_matrix(X, X_final_var_info, interaction_rule, propensity_seperate)
  full_design_matrix <- as.matrix(cbind(1, X, interaction_terms))
  
  return(as.vector(full_design_matrix %*% coef))  
}

