#include <cpp11.hpp>             // main cpp11 header
#include <cpp11/doubles.hpp>     // for doubles, writable::doubles
#include <cpp11/list.hpp>  
#include <cpp11/matrix.hpp>     // for writable::list
#include <cmath>                 // for std::log, std::sqrt
#include <limits>                // for std::numeric_limits<double>::infinity
#include <algorithm>             // for std::max
#include <vector>
#include <fstream>
#include <iostream> 
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <random>

// For runif, rnorm, rgamma; these are part of R's C API.
// If you'd prefer a different RNG, replace these calls accordingly.
#include <Rmath.h> // ensures R::rnorm, R::runif, R::rgamma work

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Use the cpp11 namespace
using namespace cpp11;

inline double safe_log(double x) {
  return std::log(std::max(x, 1e-6));
}

inline double safe_var(double x) {
  return std::max(x, 1e-6);
} 

inline bool is_invalid(double x) {
  return (!std::isfinite(x) || std::isnan(x));
} 

[[cpp11::register]]
double rinvgamma(double shape, double scale) {
  if (shape <= 0.0 || scale <= 0.0) {
    stop("Shape and scale must be positive.");
  }
  
  double g = Rf_rgamma(shape, 1 / scale);
  double out = 1.0 / g;
  return out;
}

void save_output_vector_labeled(
    double alpha, double tau_int, double tau_glob, double gamma, double xi, double sigma, 
    const cpp11::writable::doubles& beta,
    const cpp11::writable::doubles& beta_int,
    const cpp11::writable::doubles& tau_beta,
    const cpp11::writable::doubles& nu,
    const cpp11::writable::doubles& residual,
    int index,
    const std::string& filename = "output_log.txt"
) {
  std::ofstream outfile(filename, std::ios::app);
  if (!outfile.is_open()) {
    
    return;
  }
  
  outfile << "=== Iteration Output ===\n";
  outfile << "index: " << index << "\n";
  outfile << "alpha: " << alpha << "\n";
  outfile << "tau_int: " << tau_int << "\n";
  outfile << "tau_glob: " << tau_glob << "\n";
  outfile << "gamma: " << gamma << "\n";
  outfile << "xi: " << xi << "\n";
  outfile << "sigma: " << sigma << "\n";
  auto write_vector = [&](const std::string& name, const cpp11::writable::doubles& vec) {
    outfile << name << ": ";
    for (size_t i = 0; i < vec.size(); ++i) {
      outfile << vec[i];
      if (i < vec.size() - 1) outfile << ", ";
    }
    outfile << "\n";
  };
  
  write_vector("beta", beta);
  write_vector("beta_int", beta_int);
  write_vector("tau_beta", tau_beta);
  write_vector("nu", nu);
  write_vector("residual", residual);
  outfile << "\n";  // blank line between iterations
  outfile.close();
}


// --------------------------------------------
// 1) sample_beta_j
[[cpp11::register]]
double sample_beta_j_cpp(
    int N,
    writable::doubles& r_beta,
    writable::doubles& z,
    writable::doubles& w_j,
    double tau_j,
    double sigma,
    double tau_glob = 1
) {
  double sum_XZ2 = 0.0;
  double sum_XZr = 0.0;
  
  for(int i = 0; i < N; i++){
    double xz = z[i] * w_j[i];
    sum_XZ2 += xz * xz;
    sum_XZr += xz * r_beta[i];
  }
   
  double priorVar  = (sigma * sigma) * (tau_j * tau_j) * (tau_glob * tau_glob);
  double prec_data  = sum_XZ2 / (sigma * sigma);
  double prec_prior = 1.0 / priorVar;
  
  double postVar  = 1.0 / (prec_data + prec_prior);
  double postMean = postVar * (sum_XZr / (sigma * sigma));
  
  // draw from Normal(postMean, sqrt(postVar)) using R's RNG
  double draw = ::Rf_rnorm(postMean, std::sqrt(postVar));
  return draw;
}

//Log Posterior distribution for the global Tau, LogPosteriorTauGlob

double logPosteriorTauGlob(
    double tau_glob,                // Proposed tau_j > 0
    const std::vector<double> betas, 
    const std::vector<double> & beta_int, // Interaction terms: beta_{j,k} for all k in 'otherIdx'
    const std::vector<double> & tau, // The vector of all tau
    double tau_int,   
    double sigma,
    const std::vector<std::pair<int,int>> &int_pairs_trt = std::vector<std::pair<int,int>>(), //vector initializing the pairs made {j,k}
    bool interaction = false,
    bool unlink = false){
  // 1) check positivity
  if (tau_glob <= 0.0) {
    return -std::numeric_limits<double>::infinity();
  }  
  
  // 2) half-Cauchy(0,1) log prior: log(2/pi) - log(1 + tau_j^2)
  double logPrior = std::log(2.0 / M_PI) - safe_log(1.0 + tau_glob * tau_glob);
  
  // 3) main effect log-likelihood: Normal(0, sigma^2 * tau_j^2)
  int p_mod = betas.size();
  double logLikMain = 0;
  double log2pi  = std::log(2.0 * M_PI);
  for(size_t s=0; s < betas.size(); s++){
    double var_main= safe_var((sigma * sigma) * (tau[s] * tau[s]) * (tau_glob * tau_glob));
    double contribution = -0.5 * ( log2pi + safe_log(var_main) ) 
      - 0.5 * ( (betas[s] * betas[s]) / var_main );
    logLikMain += contribution;
  }
  // 4) interaction log-likelihood
  //    For each k in otherIdx, beta_{j,k} ~ Normal(0, sigma^2 * tau_j * tau_k * tau_int)
  double logLikInter = 0.0;
  if(interaction){
    for (size_t m=0; m < int_pairs_trt.size(); m++) {
      int iVar = int_pairs_trt[m].first;
      int jVar = int_pairs_trt[m].second;
      double beta_jk = beta_int[m];
      double var_jk; 
      if(unlink){
        var_jk = (sigma * sigma) * (tau[p_mod + m] * tau[p_mod + m]) *  (tau_glob * tau_glob);
      } else {
        var_jk  = (sigma * sigma) * (tau[iVar] * tau[jVar]) *  (tau_glob * tau_glob) * tau_int;
      } 
      double b2      = beta_jk * beta_jk;
      double ll = -0.5 * (log2pi + safe_log(var_jk)) -0.5 * (b2 / var_jk); 
      logLikInter += ll;
      }}
  
  return logPrior + logLikMain + logLikInter;
}

//Slice sampler for a global shrinkage parameter. tau_{glob}

// --------------------------------------------
// 2) sample_tau_j_slice
double logPosteriorTauJ(
    double tau_j,                // Proposed tau_j > 0
    double beta_j, //proposal for beta_{j}
    int index,  // index of {j}
    const std::vector<double> & beta_int, // Interaction terms: beta_{j,k} for all k in 'otherIdx'
    const std::vector<double> & tau, // The vector of all tau
    double tau_int,   
    double sigma,
    const std::vector<std::pair<int,int>> &int_pairs_trt = std::vector<std::pair<int,int>>(), //vector initializing the pairs made {j,k}
    bool interaction = false,
    double tau_glob = 1,
    bool unlink = false
) {
  // 1) check positivity
  if (tau_j <= 0.0) {
    return -std::numeric_limits<double>::infinity();
  }  
  
  // 2) half-Cauchy(0,1) log prior: log(2/pi) - log(1 + tau_j^2)
  double logPrior = std::log(2.0 / M_PI) - safe_log(1.0 + tau_j * tau_j);
  
  // 3) main effect log-likelihood: Normal(0, sigma^2 * tau_j^2)
  double log2pi  = std::log(2.0 * M_PI);
  double var_main= (sigma * sigma) * (tau_j * tau_j) * (tau_glob * tau_glob);
  double logLikMain = -0.5 * ( log2pi + safe_log(var_main) ) 
    - 0.5 * ( (beta_j * beta_j) / var_main ); 
  
  // 4) interaction log-likelihood
  //    For each k in otherIdx, beta_{j,k} ~ Normal(0, sigma^2 * tau_j * tau_k * tau_int)
  double logLikInter = 0.0;
  if((interaction) & (!unlink)){
    for (size_t m=0; m < int_pairs_trt.size(); m++) {
      int iVar = int_pairs_trt[m].first;
      int jVar = int_pairs_trt[m].second;
      int target_idx = -1848; // index to fetch the other tau[]
      bool sample_flag = false;
      if (iVar == index){
        target_idx = jVar;
        sample_flag = true;
      } else if (jVar == index){ 
        target_idx =iVar;
        sample_flag = true;
      }
      if(sample_flag){
        double beta_jk = beta_int[m];
        double var_jk;
        if(target_idx == index){
          var_jk  = safe_var((sigma * sigma) * tau_j * tau_j *(tau_glob * tau_glob)* tau_int);
        } else {
          var_jk  = safe_var((sigma * sigma) * tau_j * tau[target_idx] * (tau_glob * tau_glob) * tau_int);
        }
        double b2 = beta_jk * beta_jk;
        
        double ll = -0.5 * (log2pi + safe_log((var_jk))) 
          -0.5 * (b2 / var_jk); 
        logLikInter += ll;
      }}}   
  
  return logPrior + logLikMain + logLikInter;
}
//Slice sampler for a global shrinkage parameter. tau_{glob}
[[cpp11::register]]
double sample_tau_j_slice(
    double tau_old,
    double beta_j,
    int index,
    const std::vector<double> & beta_int,
    const std::vector<double> & tau,
    double tau_int,
    double sigma,
    bool interaction = true,
    double step_out = 0.5,
    int max_steps = 50,
    double tau_glob = 1,
    bool unlink = false
) {
  int p_int = beta_int.size();
  int p_mod = tau.size();
  std::vector<std::pair<int, int>> int_pairs_trt;
  int_pairs_trt.reserve(p_int);
  for (int ii = 0; ii < p_mod; ii++) {
    for (int jj = ii + 1; jj < p_mod; jj++) {
      int_pairs_trt.emplace_back(ii, jj);
    }
  } 
  
  double logP_old = logPosteriorTauJ(
    tau_old, beta_j, index, beta_int, tau, tau_int, sigma,
    int_pairs_trt, interaction, tau_glob, unlink);
   
  if (is_invalid(logP_old)) {
    return tau_old * (1.0 + 0.01 * Rf_runif(-1.0, 1.0));
  }
   
  double y_slice = logP_old + safe_log(Rf_runif(0.0, 1.0));
  double L = std::max(1e-6, tau_old - step_out);
  double R = tau_old + step_out;
  int step_count = 0;
   
  while (L > 1e-6 && step_count < max_steps) {
    double lp = logPosteriorTauJ(L, beta_j, index, beta_int, tau,
                                 tau_int, sigma, int_pairs_trt,
                                 interaction, tau_glob, unlink);
    if (!is_invalid(lp) && lp > y_slice) L = std::max(1e-6, L - step_out);
    else break;
    step_count++;
  } 
  
  step_count = 0;
  while (step_count < max_steps) {
    double lp = logPosteriorTauJ(R, beta_j, index, beta_int, tau,
                                 tau_int, sigma, int_pairs_trt,
                                 interaction, tau_glob, unlink);
    if (!is_invalid(lp) && lp > y_slice) R += step_out;
    else break;
    step_count++;
  }
   
  for (int rep = 0; rep < max_steps; rep++) {
    double prop = std::max(1e-8, Rf_runif(L, R));
    double lp = logPosteriorTauJ(prop, beta_j, index, beta_int, tau,
                                 tau_int, sigma, int_pairs_trt,
                                 interaction, tau_glob, unlink);
    if (!is_invalid(lp) && lp > y_slice) return prop;
    if (prop < tau_old) L = prop; else R = prop;
  } 

  return tau_old * (1.0 + 0.01 * Rf_runif(-1.0, 1.0));
} 

[[cpp11::register]]
double sample_tau_global_slice(
    double tau_old,
    const std::vector<double>& beta,
    const std::vector<double>& beta_int,
    const std::vector<double>& tau,
    double tau_int,
    double sigma,
    bool interaction = true,
    double step_out = 0.5,
    int max_steps = 50,
    bool unlink = false
) {
  int p_mod = tau.size();
  int p_int = beta_int.size();
  std::vector<std::pair<int, int>> int_pairs;
  int_pairs.reserve(p_int);
  for (int i = 0; i < p_mod; ++i) {
    for (int j = i + 1; j < p_mod; ++j) {
      int_pairs.emplace_back(i, j);
    }
  } 
  
  double logP_old = logPosteriorTauGlob(tau_old, beta, beta_int, tau, tau_int, sigma, int_pairs, interaction, unlink);
  if (is_invalid(logP_old)) return tau_old;
   
  double y_slice = logP_old + safe_log(Rf_runif(0.0, 1.0));
  double L = std::max(1e-6, tau_old - step_out);
  double R = tau_old + step_out;
   
  int steps = 0;
  while (steps < max_steps) {
    double lp = logPosteriorTauGlob(L, beta, beta_int, tau, tau_int, sigma, int_pairs, interaction, unlink);
    if (is_invalid(lp) || lp <= y_slice) break;
    L = std::max(1e-6, L - step_out);
    steps++;
  }
  steps = 0;
  while (steps < max_steps) {
    double lp = logPosteriorTauGlob(R, beta, beta_int, tau, tau_int, sigma, int_pairs, interaction, unlink);
    if (is_invalid(lp) || lp <= y_slice) break;
    R += step_out;
    steps++;
  } 
  
  for (int i = 0; i < max_steps; ++i) {
    double prop = std::max(1e-8, Rf_runif(L, R));
    double lp = logPosteriorTauGlob(prop, beta, beta_int, tau, tau_int, sigma, int_pairs, interaction, unlink);
    if (!is_invalid(lp) && lp > y_slice) return prop;
    if (prop < tau_old) L = prop; else R = prop;
  }
   
  return tau_old * (1.0 + 0.01 * Rf_runif(-1.0, 1.0));
} 

// --------------------------------------------
// 3) sample_alpha
[[cpp11::register]]
double sample_alpha_cpp(
    int N,
    writable::doubles r_alpha,
    writable::doubles z_,
    double sigma,
    double alpha_prior_sd = 10.0
) {
  double prior_var   = alpha_prior_sd * alpha_prior_sd;
  double prec_prior  = 1.0 / prior_var;
  
  double sum_z = 0.0;
  double sum_rz = 0.0;
  for(int i = 0; i < N; i++){
    sum_z  += z_[i] * z_[i];;
    sum_rz += z_[i] * r_alpha[i];
  } 
  
  double prec_data = sum_z / (sigma * sigma);
  double prec_post = prec_data + prec_prior;
  double var_post  = 1.0 / prec_post;
  double mean_post = var_post * (sum_rz / (sigma * sigma));
   
  double alpha_draw = ::Rf_rnorm(mean_post, std::sqrt(var_post));
  return alpha_draw;
} 

// --------------------------------------------
// 4) sample_sigma2_ig
// [[cpp11::register]]
double sample_sigma2_ig_cpp(
    int N,
    writable::doubles resid,
    double shape_prior = 1.0,
    double rate_prior  = 0.001
) {
  if(N == 0) return NA_REAL;
   
  double rss = 0.0;
  for(int i = 0; i < N; i++){
    rss += resid[i] * resid[i];
  }
  
  double shape_post = shape_prior + 0.5 * N;
  double rate_post  = rate_prior + 0.5 * rss;
  
  double gamma_draw = ::Rf_rgamma(shape_post, 1.0 / rate_post);
  return 1.0 / gamma_draw; // Inverse-Gamma sample
}

// --------------------------------------------
// 5) loglikeTauInt (not exported via cpp11::register)
static double loglikeTauInt(
    double tau_int,
    const std::vector<double> &beta_int_base,
    const std::vector<std::pair<int,int>> &int_pairs_base,
    const std::vector<double> &tau_main,
    double sigma, 
    bool include_treatment_int = false,
    const std::vector<double> &beta_int_trt = std::vector<double>(),
    const std::vector<double> &tau_trt = std::vector<double>(),
    const std::vector<std::pair<int,int>> &int_pairs_trt = std::vector<std::pair<int,int>>(),
    double tau_glob = 1
)
{ 
  if(tau_int < 0.01 || tau_int > 1.0) {
    return -std::numeric_limits<double>::infinity();
  } 
  
  double logp = 0.0;
  const double log2pi = std::log(2.0 * M_PI);
   
  // baseline interaction terms
  for(size_t k = 0; k < int_pairs_base.size(); k++){
    int iVar = int_pairs_base[k].first;
    int jVar = int_pairs_base[k].second;
    double var_ij = tau_int * tau_main[iVar] * tau_main[jVar] * (sigma * sigma) * (tau_glob*tau_glob);
    double beta2  = beta_int_base[k] * beta_int_base[k];
    
    logp += -0.5 * (log2pi + std::log(var_ij)) -
      0.5 * (beta2 / var_ij);
  }
  
  // treatment interaction terms
  if(include_treatment_int && !beta_int_trt.empty()){
    for(size_t k = 0; k < int_pairs_trt.size(); k++){
      int iVar = int_pairs_trt[k].first;
      int jVar = int_pairs_trt[k].second;
      double var_ij = tau_int * tau_main[iVar] * tau_main[jVar] * (sigma * sigma) * (tau_glob*tau_glob);
      double beta2  = beta_int_trt[k]*beta_int_trt[k];
      logp += -0.5 * (log2pi + std::log(var_ij)) -
        0.5 * (beta2 / var_ij);
    }
  }
  return logp;
}

bool is_integer_not_in_list_cpp(cpp11::integers r_int_vector, int target_value) {
  for (int value_in_vector : r_int_vector) {
    if (value_in_vector == target_value) {
      return false; 
    }
  }
  return true; 
}
 
// --------------------------------------------
// 6) updateLinearTreatmentCpp
[[cpp11::register]]
cpp11::writable::doubles updateLinearTreatmentCpp_cpp(
    cpp11::doubles_matrix<> X,
    cpp11::doubles Z,    
    cpp11::doubles propensity_train,
    cpp11::writable::doubles residual,
    cpp11::integers are_continous,
    double alpha,
    cpp11::writable::doubles beta,  
    double gamma,
    cpp11::writable::doubles beta_int,
    cpp11::writable::doubles tau_beta,  
    cpp11::writable::doubles nu,
    double xi,
    double tau_int,
    double sigma,
    double alpha_prior_sd,
    double tau_glob = 1.0,
    bool global_shrink = false,
    bool unlink = false,
    bool propensity_seperate = false,
    bool gibbs = false,
    bool save_output = true,
    int index = 1,
    int max_steps = 50,
    double step_out = 0.5
){
  int n = residual.size();
  int p_mod = X.ncol();
  int p_int = beta_int.size();
  double sigma2 = sigma * sigma; 
  
  if (unlink && (tau_beta.size() < (size_t)(p_mod + p_int))) {
    stop("tau_beta is too short for unlink=true! Needs p_mod + p_int entries.");
  }

  
  // Prepare pairs of indices for interaction terms
  std::vector<std::pair<int,int>> int_pairs;
  int_pairs.reserve(p_int);
  for(int i = 0; i < p_mod; i++){
    for(int j = i + 1; j < p_mod; j++){
      // Only allow interactions where of the variables is continous. 
      if((are_continous[i] ==1) | (are_continous[j]==1)){
          int_pairs.push_back(std::make_pair(i, j));
      }
    }
  }  
  if (unlink && (tau_beta.size() < (size_t)(p_mod + p_int))) {
    stop("tau_beta is too short for unlink=true! Needs p_mod + p_int entries.");
  }
  
  // Optionally, update propensity score adjustment (gamma)
  if(propensity_seperate){
    for(int i = 0; i < n; i++){
      residual[i] += propensity_train[i]* gamma; // Add back old gamma effect
    }  
    double gamma_new = sample_alpha_cpp(n, residual, propensity_train, sigma, alpha_prior_sd);
    for(int i = 0; i < n; i++){
      residual[i] -= propensity_train[i] * gamma_new; // Subtract new gamma effect
    }  
    gamma = gamma_new;
  }
  
  // Update overall intercept (alpha)
  for(int i = 0; i < n; i++){
    residual[i] += Z[i] * alpha; // Add back old alpha effect
  }  
  double alpha_new = sample_alpha_cpp(n, residual, Z, sigma, alpha_prior_sd);
  for(int i = 0; i < n; i++){
    residual[i] -= Z[i] * alpha_new; // Subtract new alpha effect
  }  
  alpha = alpha_new;
  
  // Standard vector copies for slice samplers (if not using Gibbs for betas)
  std::vector<double> beta_int_std(beta_int.begin(), beta_int.end()); 
  std::vector<double> tau_beta_std(tau_beta.begin(), tau_beta.end()); 
  
  if (gibbs) {
    // Block sample main (beta) and interaction (beta_int) coefficients using Gibbs
    int P_combined = p_mod + p_int; // Total number of beta coefficients
    
    Eigen::Map<const Eigen::MatrixXd> X_map(REAL(X), n, p_mod);
    Eigen::Map<const Eigen::VectorXd> Z_map(REAL(Z), n);
    Eigen::Map<Eigen::VectorXd> y_target_map(REAL(residual), n);
    // 1. Construct the target variable for this regression step.
    // This is y_target = current_residual + (fit from old betas)
    // which equals y - Z*alpha - propensity*gamma
    cpp11::writable::doubles y_target_vec_writable(n);
    for(int i=0; i<n; ++i) y_target_vec_writable[i] = residual[i];
    
    for (int j_main = 0; j_main < p_mod; ++j_main) {
      for (int i = 0; i < n; ++i) {
        y_target_vec_writable[i] += Z[i] * beta[j_main] * X(i, j_main);
      }
    }
    for (size_t k_idx = 0; k_idx < int_pairs.size(); ++k_idx) {
      int iVar = int_pairs[k_idx].first;
      int jVar = int_pairs[k_idx].second;
      for (int i = 0; i < n; ++i) {
        double x_interaction_val = X(i, iVar) * X(i, jVar);
        y_target_vec_writable[i] += Z[i] * beta_int[k_idx] * x_interaction_val;
      }
    }
    // Explicitly get the SEXP from the cpp11::writable::doubles object
    SEXP y_target_sxp = y_target_vec_writable; // writable::doubles has an operator SEXP()
    
    // Get the double* pointer using REAL() from R's C API
    double* y_target_ptr = (n > 0 ? REAL(y_target_sxp) : nullptr);
    
    // Now use this double* with Eigen::Map
    Eigen::Map<Eigen::VectorXd> y_target_eigen(y_target_ptr, n);
    // 2. Build the combined design matrix (X_combined) for all beta coefficients.
    Eigen::MatrixXd X_combined_eigen(n, P_combined);
    for (int i = 0; i < n; ++i) {
      for (int j_main = 0; j_main < p_mod; ++j_main) { // Main effect predictors
        X_combined_eigen(i, j_main) = Z[i] * X(i, j_main);
      }
      for (size_t k_idx = 0; k_idx < int_pairs.size(); ++k_idx) { // Interaction predictors
        int iVar = int_pairs[k_idx].first;
        int jVar = int_pairs[k_idx].second;
        X_combined_eigen(i, p_mod + k_idx) = Z[i] * X(i, iVar) * X(i, jVar);
      }
    }
    
    // 3. Define prior precision components (1 / V_k_star) for the betas.
    // The prior for beta_k is N(0, sigma^2 * V_k_star).
    // Lambda_star_inv has diagonal elements 1 / V_k_star.
    Eigen::VectorXd diag_Lambda_star_inv_eigen(P_combined);
    for (int j_main = 0; j_main < p_mod; ++j_main) { // Main effects
      double V_k_star = tau_beta[j_main] * tau_beta[j_main] * tau_glob * tau_glob;
      diag_Lambda_star_inv_eigen(j_main) = 1.0 / safe_var(V_k_star);
    }
    for (size_t k_idx = 0; k_idx < int_pairs.size(); ++k_idx) { // Interaction effects
      double V_k_star;
      if (unlink) { // Unlinked interactions have their own tau_beta
        V_k_star = tau_beta[p_mod + k_idx] * tau_beta[p_mod + k_idx] * tau_glob * tau_glob;
      } else { // Linked interactions depend on main effect taus and tau_int
        int iVar = int_pairs[k_idx].first;
        int jVar = int_pairs[k_idx].second;
        V_k_star = tau_int * tau_beta[iVar] * tau_beta[jVar] * tau_glob * tau_glob;
      }
      diag_Lambda_star_inv_eigen(p_mod + k_idx) = 1.0 / safe_var(V_k_star);
    }
    
    // 4. Calculate A = X_combined^T * X_combined + Lambda_star_inv
    Eigen::MatrixXd XtX = X_combined_eigen.transpose() * X_combined_eigen;
    Eigen::MatrixXd A_eigen = XtX + Eigen::MatrixXd(diag_Lambda_star_inv_eigen.asDiagonal());
    
    Eigen::LLT<Eigen::MatrixXd> lltOfA(A_eigen); // Cholesky decomposition of A
    if(lltOfA.info() != Eigen::Success) {
      cpp11::warning("Cholesky decomposition of A failed for beta Gibbs sampling. Betas not updated.");
    } else {
      // 5. Posterior mean: A^-1 * X_combined^T * y_target
      Eigen::VectorXd Xt_y_target = X_combined_eigen.transpose() * y_target_eigen;
      Eigen::VectorXd post_mean_beta_eigen = lltOfA.solve(Xt_y_target);
      
      // 6. Sample new betas from N(posterior_mean, sigma^2 * A^-1)
      //    using L_post_cov where L_post_cov * L_post_cov^T = sigma^2 * A^-1
      Eigen::MatrixXd L_A = lltOfA.matrixL(); // Lower Cholesky factor of A
      Eigen::MatrixXd Id = Eigen::MatrixXd::Identity(P_combined, P_combined);
      
      Eigen::MatrixXd L_A_inv_T = L_A.transpose().template triangularView<Eigen::Upper>().solve(Id);
      
      Eigen::MatrixXd L_post_cov = sigma * L_A_inv_T;
      
      Eigen::VectorXd std_normal_draws(P_combined);
      for(int k=0; k<P_combined; ++k) std_normal_draws(k) = Rf_rnorm(0.0, 1.0);
      
      Eigen::VectorXd beta_combined_new_eigen = post_mean_beta_eigen + L_post_cov * std_normal_draws;
      
      // 7. Update the original beta and beta_int vectors
      for(int j_main=0; j_main<p_mod; ++j_main) beta[j_main] = beta_combined_new_eigen(j_main);
      for(int k_idx=0; k_idx<p_int; ++k_idx) beta_int[k_idx] = beta_combined_new_eigen(p_mod + k_idx);
      
      // 8. Update residuals with the new beta values
      Eigen::VectorXd new_fit_eigen = X_combined_eigen * beta_combined_new_eigen;
      for(int i=0; i<n; ++i) residual[i] = y_target_eigen(i) - new_fit_eigen(i);
    } 
    
    // Sample local shrinkage parameters (tau_beta and nu) using the new betas
    for(int j = 0; j < p_mod; j++){ // For main effects
      nu[j] = rinvgamma(1.0, 1.0 + 1.0 / safe_var(tau_beta[j]*tau_beta[j]));
      double new_tau_beta_j_sq = rinvgamma(1.0, (1.0 / safe_var(nu[j])) + (beta[j] * beta[j]) / safe_var(2.0*tau_glob*tau_glob*sigma*sigma));
      tau_beta[j] = std::sqrt(safe_var(new_tau_beta_j_sq));
      tau_beta_std[j] = tau_beta[j]; 
    }
    if(unlink){ // For unlinked interaction effects
      for(size_t k_idx = 0; k_idx < int_pairs.size(); k_idx++){
        int full_idx = p_mod + k_idx; 
        nu[full_idx] = rinvgamma(1.0, 1.0 + 1.0 / safe_var(tau_beta[full_idx]*tau_beta[full_idx]));
        double new_tau_beta_int_k_sq = rinvgamma(1.0, (1.0 / safe_var(nu[full_idx])) + (beta_int[k_idx] * beta_int[k_idx]) / safe_var(2.0*tau_glob*tau_glob*sigma*sigma));
        tau_beta[full_idx] = std::sqrt(safe_var(new_tau_beta_int_k_sq));
        tau_beta_std[full_idx] = tau_beta[full_idx]; 
      }
    }
    if (global_shrink) { // This is the Gibbs update for tau_glob and xi
      xi = rinvgamma(1.0, 1.0 + 1.0 / safe_var(tau_glob*tau_glob)); 
      double sum_scaled_sq_betas = 0.0;
      int n_params_for_tau_glob = 0; // Number of parameters shrunk by tau_glob
      
      if(unlink){ 
        n_params_for_tau_glob = p_mod + p_int;
        for(int j = 0; j < p_mod; j++){ 
          sum_scaled_sq_betas += (beta[j]*beta[j]) / safe_var(tau_beta[j] * tau_beta[j]);
        }
        for(int k_idx = 0; k_idx < p_int; k_idx++){ 
          sum_scaled_sq_betas += (beta_int[k_idx] * beta_int[k_idx]) / safe_var(tau_beta[p_mod + k_idx] * tau_beta[p_mod + k_idx]);
        }
      } else{ 
        // Main effects are definitely shrunk by tau_glob
        for(int j = 0; j < p_mod; j++){ 
          sum_scaled_sq_betas += (beta[j]*beta[j]) / safe_var(tau_beta[j] * tau_beta[j]);
        }
        n_params_for_tau_glob = p_mod;
        
        // Linked interactions are also scaled by tau_glob.
        // Their "local variance part" is (tau_int * tau_beta[iVar] * tau_beta[jVar]).
        for(size_t k_idx = 0; k_idx < int_pairs.size(); ++k_idx) {
          int iVar = int_pairs[k_idx].first;
          int jVar = int_pairs[k_idx].second;
          double local_var_contribution = tau_int * tau_beta[iVar] * tau_beta[jVar];
          sum_scaled_sq_betas += (beta_int[k_idx] * beta_int[k_idx]) / safe_var(local_var_contribution);
        }
        n_params_for_tau_glob += p_int; 
      }
      
      double shape_tau_glob = (static_cast<double>(n_params_for_tau_glob) + 1.0) / 2.0;
      double rate_tau_glob = (1.0 / safe_var(xi)) + (1.0 / safe_var(2.0 * sigma2)) * sum_scaled_sq_betas;
      tau_glob = std::sqrt(safe_var(rinvgamma(shape_tau_glob, rate_tau_glob)));
    } 
    // If not global_shrink, tau_glob remains fixed, and xi isn't updated.
    
  } else { // Not Gibbs: use original slice sampling for betas and taus
    for(int j = 0; j < p_mod; j++){
      for(int i = 0; i < n; i++){ // Add back current beta_j's effect
        residual[i] += Z[i] * beta[j] * X(i,j);
      }
      cpp11::writable::doubles w_j(n); // Predictor for beta_j
      for(int i = 0; i < n; i++){
        w_j[i] = X(i, j);
      }
      double beta_j_new = sample_beta_j_cpp(n, residual, Z, w_j, tau_beta[j], sigma, tau_glob);
      for(int i = 0; i < n; i++){ // Subtract new beta_j's effect
        residual[i] -= Z[i] * beta_j_new * X(i,j);
      }  
      beta[j] = beta_j_new;
      
      // Sample local scale tau_beta[j]
      double tb_j_new = sample_tau_j_slice(
        tau_beta[j], beta[j], j, beta_int_std, tau_beta_std,
        tau_int, sigma, true, step_out, max_steps, tau_glob, unlink);
      tau_beta[j] = tb_j_new;
      tau_beta_std[j] = tb_j_new; // Keep std vector copy in sync
    }  
    
    for(size_t k = 0; k < int_pairs.size(); k++){ // For interaction terms
      int iVar = int_pairs[k].first;
      int jVar = int_pairs[k].second;
      for(int i = 0; i < n; i++){ // Add back current beta_int[k]'s effect
        double x_ij = X(i, iVar) * X(i, jVar);
        residual[i] += Z[i] * beta_int[k] * x_ij;
      }
      cpp11::writable::doubles w_ij(n); // Predictor for beta_int[k]
      for(int i = 0; i < n; i++){
        w_ij[i] = X(i, iVar) * X(i, jVar);
      }  
      double scale_int;  // Prior scale for this interaction beta
      if(unlink){
        scale_int = tau_beta[p_mod + k];
      } else{
        scale_int = std::sqrt(safe_var(tau_int * tau_beta[iVar] * tau_beta[jVar]));
      }
      double beta_int_new = sample_beta_j_cpp(n, residual, Z, w_ij, scale_int, sigma, tau_glob);
      for(int i = 0; i < n; i++){ // Subtract new beta_int[k]'s effect
        double x_ij = X(i, iVar) * X(i, jVar);
        residual[i] -= Z[i] * beta_int_new * x_ij;
      }  
      beta_int[k] = beta_int_new;
      beta_int_std[k] = beta_int_new; 
      
      if(unlink){ // If unlinked, sample its specific tau_beta
        double tb_j_new = sample_tau_j_slice(
          tau_beta[p_mod+k], beta_int[k], static_cast<int>(p_mod+k), 
          beta_int_std, tau_beta_std, tau_int, sigma,
          true, step_out, max_steps, tau_glob, unlink); 
        tau_beta[p_mod+k] = tb_j_new;
        tau_beta_std[p_mod+k] = tb_j_new;
      }
    }
    
    // Sample tau_int (interaction scale) using Metropolis-Hastings, if applicable
    if((!global_shrink) && (!unlink)){ 
      std::vector<double> beta_int_std_current(beta_int.begin(), beta_int.end());
      // Assuming loglikeTauInt uses the full tau_beta_std and picks relevant taus internally
      double currentTauInt  = tau_int;
      double proposedTauInt = ::Rf_runif(0.01, 1.0);  
      double logPosteriorCurrent = loglikeTauInt(currentTauInt, beta_int_std_current, int_pairs, tau_beta_std, sigma, false, {}, {}, {}, tau_glob);  
      double logPosteriorProposed = loglikeTauInt(proposedTauInt, beta_int_std_current, int_pairs, tau_beta_std, sigma, false, {}, {}, {}, tau_glob);
      double logAccept = logPosteriorProposed - logPosteriorCurrent;
      if(::Rf_runif(0.0, 1.0) < std::exp(logAccept)){
        tau_int = proposedTauInt;
      }
    } else if (unlink || global_shrink) { 
      // If unlinked or global_shrink is on, tau_int might be fixed (e.g., to 1) or not used.
      // This behavior should align with your model's definition.
      tau_int = 1.0; 
    }
    
    // Sample global shrinkage tau_glob using slice sampler, if applicable
    if(global_shrink){
      std::vector<double> beta_std_current(beta.begin(), beta.end());
      std::vector<double> beta_int_std_current(beta_int.begin(), beta_int.end());
      tau_glob = sample_tau_global_slice(tau_glob, beta_std_current, beta_int_std_current, tau_beta_std, tau_int, sigma, true, step_out, max_steps, unlink);
    }
  } 
  
  // Consolidate results for output
  size_t total_size = 5 + beta.size() + beta_int.size() + tau_beta.size() + nu.size() + residual.size();  
  cpp11::writable::doubles output;
  output.reserve(total_size);
  output.push_back(alpha);
  output.push_back(tau_int);
  output.push_back(tau_glob);
  output.push_back(gamma);
  output.push_back(xi); 
  for(double val : beta) output.push_back(val);
  for(double val : beta_int) output.push_back(val);
  for(double val : tau_beta) output.push_back(val);
  for(double val : nu) output.push_back(val);
  for(double val : residual) output.push_back(val);
  
  if(save_output){
    save_output_vector_labeled( // Helper function to log parameters
      alpha, tau_int, tau_glob, gamma, xi, sigma,
      beta, beta_int, tau_beta, nu, residual, index
    );
  }
  return output;
}


 /////////////////////////////////////////////////////////
/// LOGISTIC REGRESSION LINKED SHRINKAGE + HORSESHOE. ///
////////////////////////////////////////////////////////
// 
// std::vector<std::pair<int, int>> create_interaction_pairs(int p_main) {
//   std::vector<std::pair<int, int>> pairs;
//   if (p_main < 2) return pairs;
//   for (int j = 0; j < p_main; ++j) {
//     for (int k = j + 1; k < p_main; ++k) {
//       pairs.push_back({j, k});
//     }
//   }
//   return pairs;
// }
// 
// [[cpp11::register]]
// cpp11::list linked_shrinkage_logistic_gibbs(
//     cpp11::integers R_y,            
//     cpp11::doubles_matrix<> R_X,   
//     cpp11::doubles R_Z,             
//     int n_iter,
//     int burn_in,
//     double alpha_prior_sd = 10.0,
//     double aleph_prior_sd = 10.0,
//     double init_alpha = 0.0,
//     double init_aleph = 0.0,
//     unsigned int seed = 1848
// ) {
//   cpp11::function set_seed_r = cpp11::package("base")["set.seed"];
//   set_seed_r(seed);
//   
//   int N = R_X.nrow(); 
//   int p_main = R_X.ncol();
//   
//   // --- Copy data from cpp11 to Eigen (simplification to avoid Map errors) ---
//   Eigen::VectorXi y(N);
//   for (int i = 0; i < N; ++i) {
//     y(i) = R_y[i]; 
//   }
//   
//   Eigen::MatrixXd X(N, p_main);
//   for (int j = 0; j < p_main; ++j) { 
//     for (int i = 0; i < N; ++i) {
//       X(i, j) = R_X(i, j); 
//     }
//   }
//   
//   Eigen::VectorXd Z(N);
//   for (int i = 0; i < N; ++i) {
//     Z(i) = R_Z[i]; 
//   }
//   // --- End Data Copy ---
//   
//   std::vector<std::pair<int, int>> main_interaction_indices = create_interaction_pairs(p_main);
//   int p_interaction_main = main_interaction_indices.size();
//   
//   double alpha = init_alpha;
//   double aleph = init_aleph;
//   Eigen::VectorXd beta = Eigen::VectorXd::Zero(p_main);
//   Eigen::VectorXd beta_interaction = Eigen::VectorXd::Zero(p_interaction_main);
//   Eigen::VectorXd gamma = Eigen::VectorXd::Zero(p_main);
//   Eigen::VectorXd gamma_int = Eigen::VectorXd::Zero(p_interaction_main);
//   
//   Eigen::VectorXd tau_j_params = Eigen::VectorXd::Ones(p_main);
//   Eigen::VectorXd zeta_tau_j = Eigen::VectorXd::Ones(p_main);
//   const double tau_int_param = 1.0; 
//   
//   Eigen::VectorXd lambda_gamma = Eigen::VectorXd::Ones(p_main);
//   Eigen::VectorXd nu_gamma = Eigen::VectorXd::Ones(p_main);
//   Eigen::VectorXd lambda_g_int = Eigen::VectorXd::Ones(p_interaction_main);
//   Eigen::VectorXd nu_g_int = Eigen::VectorXd::Ones(p_interaction_main);
//   double tau_hs_combined = 1.0;
//   double xi_hs_combined = 1.0;
//   
//   Eigen::VectorXd omega = Eigen::VectorXd::Ones(N);
//   Eigen::VectorXd eta = Eigen::VectorXd::Zero(N);
//   Eigen::VectorXd kappa = y.cast<double>().array() - 0.5; // Uses the Eigen 'y'
//   
//   int num_samples_to_store = n_iter - burn_in;
//   if (num_samples_to_store <= 0) {
//     cpp11::stop("n_iter must be greater than burn_in.");
//   }
//   
//   // These declarations might still cause "alias template deduction" errors if there's a
//   // C++17 vs C++20 feature use issue with your cpp11 version / compiler.
//   cpp11::writable::doubles alpha_samples; alpha_samples.reserve(num_samples_to_store);
//   cpp11::writable::doubles_matrix beta_samples(num_samples_to_store, p_main);
//   cpp11::writable::doubles_matrix beta_interaction_samples(num_samples_to_store, std::max(1, p_interaction_main));
//   if (p_interaction_main == 0 && num_samples_to_store > 0) {
//     beta_interaction_samples = cpp11::writable::doubles_matrix(num_samples_to_store, 0);
//   }
//   cpp11::writable::doubles aleph_samples; aleph_samples.reserve(num_samples_to_store);
//   cpp11::writable::doubles_matrix gamma_samples(num_samples_to_store, p_main); 
//   cpp11::writable::doubles_matrix gamma_int_samples(num_samples_to_store, std::max(1,p_interaction_main)); 
//   if (p_interaction_main == 0 && num_samples_to_store > 0) {
//     gamma_int_samples = cpp11::writable::doubles_matrix(num_samples_to_store, 0);
//   }
//   cpp11::writable::doubles_matrix tau_j_samples(num_samples_to_store, p_main);
//   cpp11::writable::doubles_matrix lambda_gamma_samples(num_samples_to_store, p_main);
//   cpp11::writable::doubles_matrix lambda_g_int_samples(num_samples_to_store, std::max(1,p_interaction_main));
//   if (p_interaction_main == 0 && num_samples_to_store > 0) {
//     lambda_g_int_samples = cpp11::writable::doubles_matrix(num_samples_to_store, 0);
//   }
//   cpp11::writable::doubles tau_hs_combined_samples; tau_hs_combined_samples.reserve(num_samples_to_store);
//   
//   int current_sample_idx = 0;
//   cpp11::function rpg_cpp = cpp11::package("BayesLogit")["rpg"];
//   
//   for (int iter = 0; iter < n_iter; ++iter) {
//     // 1. Update Linear Predictor eta
//     eta = X * beta; // X is now the Eigen::MatrixXd copy
//     eta.array() += alpha;
//     if (p_interaction_main > 0) { 
//       for (int k = 0; k < p_interaction_main; ++k) {
//         eta.array() += beta_interaction(k) * (X.col(main_interaction_indices[k].first).array() * X.col(main_interaction_indices[k].second).array());
//       }
//     }
//     Eigen::VectorXd treatment_modifier_part = Eigen::VectorXd::Zero(N);
//     treatment_modifier_part.setConstant(aleph);
//     treatment_modifier_part += X * gamma;
//     if (p_interaction_main > 0) { 
//       for (int k = 0; k < p_interaction_main; ++k) {
//         treatment_modifier_part.array() += gamma_int(k) * (X.col(main_interaction_indices[k].first).array() * X.col(main_interaction_indices[k].second).array());
//       }
//     }
//     eta.array() += Z.array() * treatment_modifier_part.array(); // Z is now Eigen::VectorXd
//     
//     // 2. Sample Polya-Gamma latent variables omega_i
//     cpp11::writable::doubles eta_cpp(N);
//     for(int i=0; i<N; ++i) eta_cpp[i] = std::abs(eta[i]);
//     cpp11::writable::doubles pg_b_param(N);
//     for(int i=0; i<N; ++i) pg_b_param[i] = 1.0;
//     cpp11::doubles omega_new_cpp = cpp11::as_doubles(rpg_cpp(N, pg_b_param, eta_cpp));
//     for(int i=0; i<N; ++i) omega[i] = omega_new_cpp[i];
//     
//     // 3. Sample Regression Coefficients 
//     int current_total_coeffs = 1 + p_main + (p_interaction_main > 0 ? p_interaction_main : 0) + 
//       1 + p_main + (p_interaction_main > 0 ? p_interaction_main : 0);
//     Eigen::MatrixXd X_full(N, current_total_coeffs); 
//     int col_counter = 0;
//     X_full.col(col_counter).setOnes(); col_counter++; 
//     X_full.block(0, col_counter, N, p_main) = X; col_counter += p_main; 
//     if (p_interaction_main > 0) {
//       for (int k = 0; k < p_interaction_main; ++k) {
//         X_full.col(col_counter + k) = X.col(main_interaction_indices[k].first).array() * X.col(main_interaction_indices[k].second).array();
//       }
//       col_counter += p_interaction_main; 
//     }
//     X_full.col(col_counter) = Z; col_counter++; 
//     for (int l = 0; l < p_main; ++l) {
//       X_full.col(col_counter + l) = Z.array() * X.col(l).array();
//     }
//     col_counter += p_main; 
//     if (p_interaction_main > 0) {
//       for (int k = 0; k < p_interaction_main; ++k) {
//         X_full.col(col_counter + k) = Z.array() * (X.col(main_interaction_indices[k].first).array() * X.col(main_interaction_indices[k].second).array());
//       }
//     } 
//     
//     Eigen::VectorXd Y_star = kappa.array() / omega.array(); 
//     Eigen::MatrixXd Omega_diag_mat = omega.asDiagonal(); 
//     Eigen::MatrixXd Xt_Omega_X = X_full.transpose() * Omega_diag_mat * X_full;
//     Eigen::VectorXd Xt_Omega_Y = X_full.transpose() * Omega_diag_mat * Y_star;
//     
//     Eigen::VectorXd prior_precision_diag_vec(current_total_coeffs); 
//     col_counter = 0;
//     prior_precision_diag_vec(col_counter) = 1.0 / safe_var(alpha_prior_sd * alpha_prior_sd); col_counter++; 
//     for (int j = 0; j < p_main; ++j) {
//       prior_precision_diag_vec(col_counter + j) = 1.0 / safe_var(tau_j_params(j) * tau_j_params(j));
//     }
//     col_counter += p_main; 
//     if (p_interaction_main > 0) {
//       for (int k = 0; k < p_interaction_main; ++k) {
//         double var_jk = safe_var(tau_j_params(main_interaction_indices[k].first) *
//                                  tau_j_params(main_interaction_indices[k].second) *
//                                  tau_int_param); 
//         prior_precision_diag_vec(col_counter + k) = 1.0 / var_jk;
//       }
//       col_counter += p_interaction_main; 
//     }
//     prior_precision_diag_vec(col_counter) = 1.0 / safe_var(aleph_prior_sd * aleph_prior_sd); col_counter++; 
//     for (int l = 0; l < p_main; ++l) { 
//       prior_precision_diag_vec(col_counter + l) = 1.0 / safe_var(lambda_gamma(l) * lambda_gamma(l) * tau_hs_combined * tau_hs_combined);
//     }
//     col_counter += p_main; 
//     if (p_interaction_main > 0) {
//       for (int k = 0; k < p_interaction_main; ++k) { 
//         prior_precision_diag_vec(col_counter + k) = 1.0 / safe_var(lambda_g_int(k) * lambda_g_int(k) * tau_hs_combined * tau_hs_combined);
//       }
//     } 
//     
//     Eigen::MatrixXd P_inv = prior_precision_diag_vec.asDiagonal(); 
//     Eigen::MatrixXd posterior_precision = Xt_Omega_X + P_inv;
//     Eigen::LLT<Eigen::MatrixXd> llt(posterior_precision);
//     if(llt.info() != Eigen::Success) {
//       double jitter = 1e-6; 
//       Eigen::MatrixXd eye = Eigen::MatrixXd::Identity(current_total_coeffs, current_total_coeffs); 
//       posterior_precision += jitter * eye; 
//       llt.compute(posterior_precision); 
//       if(llt.info() != Eigen::Success) cpp11::stop("Cholesky failed even with jitter. Aborting.");
//     }
//     Eigen::VectorXd posterior_mean = llt.solve(Xt_Omega_Y);
//     Eigen::VectorXd z_norm(current_total_coeffs); 
//     for(int i=0; i<current_total_coeffs; ++i) z_norm(i) = Rf_rnorm(0,1); 
//     Eigen::MatrixXd L_matrix = llt.matrixL(); 
//     Eigen::MatrixXd L_transpose_inv = L_matrix.transpose().inverse(); 
//     Eigen::VectorXd sampled_coeffs = posterior_mean + L_transpose_inv * z_norm;
//     
//     col_counter = 0; 
//     alpha = sampled_coeffs(col_counter); col_counter++;
//     beta = sampled_coeffs.segment(col_counter, p_main); col_counter += p_main;
//     if (p_interaction_main > 0) {
//       beta_interaction = sampled_coeffs.segment(col_counter, p_interaction_main);
//       col_counter += p_interaction_main;
//     } else { beta_interaction.setZero(); }
//     aleph = sampled_coeffs(col_counter); col_counter++;
//     gamma = sampled_coeffs.segment(col_counter, p_main); col_counter += p_main;
//     if (p_interaction_main > 0) {
//       gamma_int = sampled_coeffs.segment(col_counter, p_interaction_main);
//     } else { gamma_int.setZero(); }
//     
//     // 4. Sample tau_j_params 
//     for (int j = 0; j < p_main; ++j) {
//       zeta_tau_j(j) = rinvgamma(1.0, 1.0 + 1.0 / safe_var(tau_j_params(j) * tau_j_params(j)));
//       double sum_sq_beta_terms = beta(j) * beta(j);
//       int num_dependent_coeffs = 1;
//       if (p_interaction_main > 0) {
//         for (int k_int = 0; k_int < p_interaction_main; ++k_int) {
//           int idx1 = main_interaction_indices[k_int].first;
//           int idx2 = main_interaction_indices[k_int].second;
//           if (idx1 == j) {
//             sum_sq_beta_terms += beta_interaction(k_int) * beta_interaction(k_int) /
//               safe_var(tau_j_params(idx2) * tau_j_params(idx2) * tau_int_param * tau_int_param); 
//             num_dependent_coeffs++;
//           } else if (idx2 == j) {
//             sum_sq_beta_terms += beta_interaction(k_int) * beta_interaction(k_int) /
//               safe_var(tau_j_params(idx1) * tau_j_params(idx1) * tau_int_param * tau_int_param);
//             num_dependent_coeffs++;
//           }
//         }
//       }
//       tau_j_params(j) = std::sqrt(safe_var(rinvgamma(0.5 + (double)num_dependent_coeffs / 2.0,
//                                            1.0 / zeta_tau_j(j) + sum_sq_beta_terms / 2.0)));
//     }
//     
//     // 6. Sample Local Horseshoe parameters for gamma
//     for (int l = 0; l < p_main; ++l) {
//       lambda_gamma(l) = std::sqrt(safe_var(rinvgamma(1.0, 1.0/nu_gamma(l) + (gamma(l)*gamma(l)) / safe_var(2.0*tau_hs_combined*tau_hs_combined))));
//       nu_gamma(l) = rinvgamma(1.0, 1.0 + 1.0/safe_var(lambda_gamma(l)*lambda_gamma(l)));
//     }
//     
//     // 7. Sample Local Horseshoe parameters for gamma_int
//     if (p_interaction_main > 0) {
//       for (int k = 0; k < p_interaction_main; ++k) {
//         lambda_g_int(k) = std::sqrt(safe_var(rinvgamma(1.0, 1.0/nu_g_int(k) + (gamma_int(k)*gamma_int(k)) / safe_var(2.0*tau_hs_combined*tau_hs_combined))));
//         nu_g_int(k) = rinvgamma(1.0, 1.0 + 1.0/safe_var(lambda_g_int(k)*lambda_g_int(k)));
//       }
//     }
//     
//     // 8. Sample SHARED Global Horseshoe parameter tau_hs_combined
//     double sum_gamma_over_lambda_sq = 0;
//     for (int l = 0; l < p_main; ++l) {
//       sum_gamma_over_lambda_sq += (gamma(l)*gamma(l)) / safe_var(lambda_gamma(l)*lambda_gamma(l));
//     }
//     double sum_g_int_over_lambda_sq = 0;
//     if (p_interaction_main > 0) {
//       for (int k = 0; k < p_interaction_main; ++k) {
//         sum_g_int_over_lambda_sq += (gamma_int(k)*gamma_int(k)) / safe_var(lambda_g_int(k)*lambda_g_int(k));
//       }
//     }
//     int total_coeffs_for_hs = p_main + (p_interaction_main > 0 ? p_interaction_main : 0) ; 
//     if (total_coeffs_for_hs > 0) { 
//       tau_hs_combined = std::sqrt(safe_var(rinvgamma( (double)(total_coeffs_for_hs + 1.0)/2.0, 
//                                                       1.0/xi_hs_combined + (sum_gamma_over_lambda_sq + sum_g_int_over_lambda_sq)/2.0)));
//       xi_hs_combined = rinvgamma(1.0, 1.0 + 1.0/safe_var(tau_hs_combined*tau_hs_combined));
//     } else { 
//       tau_hs_combined = 1.0; 
//       xi_hs_combined = 1.0;
//     }
//     
//     // Store Samples
//     if (iter >= burn_in) {
//       alpha_samples.push_back(alpha);
//       aleph_samples.push_back(aleph);
//       tau_hs_combined_samples.push_back(tau_hs_combined);
//       
//       for(int j=0; j<p_main; ++j) beta_samples(current_sample_idx, j) = beta(j);
//       if (p_interaction_main > 0) {
//         for(int k=0; k<p_interaction_main; ++k) beta_interaction_samples(current_sample_idx, k) = beta_interaction(k);
//       }
//       for(int j=0; j<p_main; ++j) gamma_samples(current_sample_idx, j) = gamma(j);
//       if (p_interaction_main > 0) {
//         for(int k=0; k<p_interaction_main; ++k) gamma_int_samples(current_sample_idx, k) = gamma_int(k);
//       }
//       for(int j=0; j<p_main; ++j) tau_j_samples(current_sample_idx, j) = tau_j_params(j);
//       for(int j=0; j<p_main; ++j) lambda_gamma_samples(current_sample_idx, j) = lambda_gamma(j);
//       if (p_interaction_main > 0) {
//         for(int k=0; k<p_interaction_main; ++k) lambda_g_int_samples(current_sample_idx, k) = lambda_g_int(k);
//       }
//       current_sample_idx++;
//     }
//   } // End MCMC loop
//   
//   cpp11::writable::doubles fixed_tau_int_value; 
//   fixed_tau_int_value.push_back(tau_int_param);
//   
//   // Using explicit cpp11::named_arg constructor
//   return cpp11::writable::list({
//     cpp11::named_arg("alpha", alpha_samples),
//     cpp11::named_arg("beta", beta_samples),
//     cpp11::named_arg("beta_interaction", beta_interaction_samples),
//     cpp11::named_arg("aleph", aleph_samples),
//     cpp11::named_arg("gamma", gamma_samples),
//     cpp11::named_arg("gamma_int", gamma_int_samples),
//     cpp11::named_arg("tau_j", tau_j_samples),
//     cpp11::named_arg("tau_int_fixed_value", fixed_tau_int_value), 
//     cpp11::named_arg("lambda_gamma", lambda_gamma_samples),
//     cpp11::named_arg("lambda_g_int", lambda_g_int_samples),
//     cpp11::named_arg("tau_hs_combined", tau_hs_combined_samples)
//   });
// }
