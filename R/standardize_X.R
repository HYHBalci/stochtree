#' @title Standardize and Transform Design Matrix (Enhanced)
#' @description Processes an input data.frame or matrix `X_initial` by
#' standardizing numeric columns and transforming categorical/binary columns
#' into a design matrix suitable for modeling. It also generates metadata about
#' the transformed columns and calculates the number of interaction terms based on a specified rule.
#' @param X_initial A data.frame or matrix containing the predictor variables.
#' @param process_data A boolean flag. If `TRUE` (default), the function standardizes numeric
#'  variables and creates dummy variables for categorical ones. If `FALSE`, it bypasses
#'  transformations and returns the original data as a matrix, but still generates the metadata.
#' @param interaction_rule A character string specifying which interactions are allowed.
#' Must be one of:
#'  \itemize{
#'  \item `"continuous"` (Default): Allows interactions where at least one variable is continuous.
#'  \item `"continuous_or_binary"`: Allows interactions where at least one variable is continuous OR binary.
#'  \item `"all"`: Allows all possible two-way interactions between valid columns.
#'}
#' @param cat_coding_method A character string for categorical variable contrast coding.
#' Must be one of `"sum"` (for sum-to-zero/deviation coding) or `"difference"` (for successive differences).
#' Default is `"sum"`.
#' @return A list containing `X_final`, `p_int`, `non_continous_idx_cpp`, and `X_final_var_info`.
#' @export
#' @importFrom stats terms model.matrix sd na.omit
# Note: Requires MASS package if using `cat_coding_method = "difference"`

standardize_X_by_index <- function(X_initial,
                                   process_data = TRUE, # New parameter
                                   interaction_rule = c("continuous", "continuous_or_binary", "all"),
                                   cat_coding_method = c("sum", "difference")) {
  
  # --- Input Validation and Setup ---
  interaction_rule <- match.arg(interaction_rule)
  cat_coding_method <- match.arg(cat_coding_method)
  
  if (cat_coding_method == "difference" && !requireNamespace("MASS", quietly = TRUE)) {
    stop("Package 'MASS' is required for difference coding (contr.sdif). Please install it.", call. = FALSE)
  }
  if (!is.data.frame(X_initial) && !is.matrix(X_initial)) {
    stop("X_initial must be a data.frame or a matrix.")
  }
  
  if (ncol(X_initial) == 0) {
    # Handle empty input gracefully
    warning("X_initial has no columns. Returning empty outputs.")
    return(list(
      X_final = matrix(nrow = nrow(X_initial), ncol = 0), p_int = 0,
      non_continous_idx_cpp = numeric(0),
      X_final_var_info = data.frame(
        original_idx = integer(0), is_continuous = logical(0),
        is_binary = logical(0), is_categorical = logical(0),
        col_name_final = character(0), stringsAsFactors = FALSE
      )
    ))
  }
  
  original_colnames <- colnames(X_initial)
  if (is.null(original_colnames)) original_colnames <- paste0("V", 1:ncol(X_initial))
  # Ensure X_initial has column names for the logic below
  if(is.matrix(X_initial)) {
    X_df <- as.data.frame(X_initial)
    colnames(X_df) <- original_colnames
  } else {
    X_df <- X_initial
    colnames(X_df) <- original_colnames
  }
  
  
  # --- Main Logic Branch ---
  if (process_data) {

    # Step 1: Classify Original Columns and Prepare Intermediate Data
    all_cols_info <- lapply(1:ncol(X_df), function(j_idx) {
      col_data <- X_df[[j_idx]]
      n_unique <- length(unique(col_data[!is.na(col_data)]))
      
      is_continuous <- FALSE; is_binary <- FALSE; is_categorical <- FALSE
      processed_col_data <- NULL; should_include <- FALSE
      
      if (n_unique > 1) {
        if (is.logical(col_data)) col_data <- as.character(col_data)
        n_unique <- length(unique(col_data[!is.na(col_data)]))
        
        if (n_unique == 2) {
          factor_col <- factor(col_data)
          processed_col_data <- ifelse(factor_col == levels(factor_col)[1], -1, 1)
          is_binary <- TRUE; should_include <- TRUE
        } else if (is.character(col_data) || is.factor(col_data) || (is.numeric(col_data) && n_unique > 2 && n_unique < 4)) {
          col_as_factor <- factor(col_data)
          if (nlevels(col_as_factor) > 1) {
            contrasts(col_as_factor) <- if (cat_coding_method == "difference") MASS::contr.sdif(nlevels(col_as_factor)) else contr.sum(nlevels(col_as_factor))
            processed_col_data <- col_as_factor
            is_categorical <- TRUE; should_include <- TRUE
          }
        } else if (is.numeric(col_data)) {
          processed_col_data <- col_data
          is_continuous <- TRUE; should_include <- TRUE
        }
      }
      list(
        processed_data = processed_col_data, is_continuous = is_continuous,
        is_binary = is_binary, is_categorical = is_categorical, 
        original_idx = j_idx, col_name = original_colnames[j_idx],
        include_in_intermediate = should_include
      )
    })
    
    # Step 2: Generate Processed Dataframes and Metadata
    valid_cols_info <- all_cols_info[sapply(all_cols_info, `[[`, "include_in_intermediate")]
    if (length(valid_cols_info) == 0) stop('No valid columns to process.')
    
    X_intermediate_list <- lapply(valid_cols_info, `[[`, "processed_data")
    names(X_intermediate_list) <- sapply(valid_cols_info, `[[`, "col_name")
    X_intermediate_df <- as.data.frame(X_intermediate_list, stringsAsFactors = FALSE)
    
    intermediate_map <- data.frame(
      col_name_intermediate = sapply(valid_cols_info, `[[`, "col_name"),
      original_idx = sapply(valid_cols_info, `[[`, "original_idx"),
      is_continuous = sapply(valid_cols_info, `[[`, "is_continuous"),
      is_binary = sapply(valid_cols_info, `[[`, "is_binary"),
      is_categorical = sapply(valid_cols_info, `[[`, "is_categorical"),
      stringsAsFactors = FALSE
    )
    
    cols_to_scale <- intermediate_map$col_name_intermediate[intermediate_map$is_continuous]
    if (length(cols_to_scale) > 0) {
      X_intermediate_df[cols_to_scale] <- lapply(X_intermediate_df[cols_to_scale], function(col) as.vector(scale(col)))
    }
    
    X_final <- model.matrix(~ . - 1, data = X_intermediate_df)
    
    attr_assign <- attr(X_final, "assign")
    matched_rows <- match(colnames(X_intermediate_df)[attr_assign], intermediate_map$col_name_intermediate)
    
    X_final_var_info <- data.frame(
      original_idx = intermediate_map$original_idx[matched_rows],
      is_continuous = intermediate_map$is_continuous[matched_rows],
      is_binary = intermediate_map$is_binary[matched_rows],
      is_categorical = intermediate_map$is_categorical[matched_rows],
      col_name_final = colnames(X_final),
      stringsAsFactors = FALSE
    )
    
    # Calculate non-continuous indices for C++ (0-based)
    original_var_is_continuous <- sapply(all_cols_info, `[[`, "is_continuous")
    non_continous_idx_cpp <- which(!original_var_is_continuous) - 1
    
  } else {

    # X_final is just the original matrix
    X_final <- as.matrix(X_df)
    
    all_cols_info <- lapply(1:ncol(X_df), function(j_idx) {
      col_data <- X_df[[j_idx]]
      n_unique <- length(unique(col_data[!is.na(col_data)]))
      
      is_continuous <- FALSE; is_binary <- FALSE; is_categorical <- FALSE
      
      if (n_unique > 1) {
        if (is.logical(col_data)) col_data <- as.character(col_data)
        n_unique <- length(unique(col_data[!is.na(col_data)]))
        
        if (n_unique == 2) {
          is_binary <- TRUE
        } else if (is.character(col_data) || is.factor(col_data) || (is.numeric(col_data) && n_unique > 2 && n_unique < min(20, 0.2 * nrow(X_df)))) {
          is_categorical <- TRUE
        } else if (is.numeric(col_data)) {
          is_continuous <- TRUE
        }
      }
      list(is_continuous = is_continuous, is_binary = is_binary, is_categorical = is_categorical)
    })
    
    # Create the metadata frame directly
    X_final_var_info <- data.frame(
      original_idx = 1:ncol(X_final),
      is_continuous = sapply(all_cols_info, `[[`, "is_continuous"),
      is_binary = sapply(all_cols_info, `[[`, "is_binary"),
      is_categorical = sapply(all_cols_info, `[[`, "is_categorical"),
      col_name_final = original_colnames,
      stringsAsFactors = FALSE
    )
    
    # Calculate non-continuous indices for C++ (0-based)
    non_continous_idx_cpp <- which(!X_final_var_info$is_continuous) - 1
  }
  
  # --- Common Steps for Both Paths: Calculate p_int ---
  p_int_calculated <- 0
  n_final_cols <- ncol(X_final)
  if (n_final_cols > 1) {
    # Determine which columns are candidates for interaction based on the rule
    is_candidate <- switch(
      interaction_rule,
      "all" = rep(TRUE, n_final_cols),
      "continuous_or_binary" = X_final_var_info$is_continuous | X_final_var_info$is_binary,
      "continuous" = X_final_var_info$is_continuous
    )
    
    # Count pairs where at least one variable is a candidate
    for (i in 1:(n_final_cols - 1)) {
      for (j in (i + 1):n_final_cols) {
        if (is_candidate[i] || is_candidate[j]) {
          p_int_calculated <- p_int_calculated + 1
        }
      }
    }
  }
  
  # --- Assemble the final result list ---
  return(list(
    X_final = X_final,
    p_int = p_int_calculated,
    non_continous_idx_cpp = non_continous_idx_cpp,
    X_final_var_info = X_final_var_info
  ))
}
