#include <cpp11.hpp>
#include <cpp11/doubles.hpp>
#include <cpp11/integers.hpp>
#include <cpp11/list.hpp>
#include <cpp11/matrix.hpp>
#include <fstream>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>

// Eigen headers for fast linear algebra
#include <Eigen/Dense>
#include <Eigen/Cholesky>

// For R's random number generators (rnorm, rgamma, etc.)
#include <Rmath.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Use the cpp11 namespace for R interoperability
using namespace cpp11;

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// SECTION 1: UTILITY AND HELPER FUNCTIONS
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

inline double safe_log(double x) {
  return std::log(std::max(x, 1e-12));
}

inline double safe_var(double x) {
  return std::max(x, 1e-12);
}

inline bool is_invalid(double x) {
  return !std::isfinite(x);
}

double rinvgamma(double shape, double scale) {
  if (shape <= 0.0 || scale <= 0.0) {
    stop("Shape and scale for rinvgamma must be positive.");
  }
  double g = Rf_rgamma(shape, 1.0 / scale);
  return (g > 1e-12) ? 1.0 / g : 1e12;
}

void save_output_vector_labeled(
    std::ofstream& outfile,
    double alpha, double tau_int, double tau_glob, double gamma, double xi, double sigma,
    const cpp11::writable::doubles& beta, const cpp11::writable::doubles& beta_int,
    const cpp11::writable::doubles& tau_beta, const cpp11::writable::doubles& nu, int index) {
  if (!outfile.is_open()) return;
  outfile << "=== Iteration " << index << " ===\n";
  outfile << "alpha: " << alpha << "\n";
  outfile << "tau_int: " << tau_int << "\n";
  outfile << "tau_glob: " << tau_glob << "\n";
  outfile << "gamma: " << gamma << "\n";
  outfile << "xi: " << xi << "\n";
  outfile << "sigma: " << sigma << "\n";
  
  auto write_vector = [&](const std::string& name, const cpp11::doubles& vec) {
    outfile << name << ": ";
    for (R_xlen_t i = 0; i < vec.size(); ++i) {
      outfile << vec[i] << (i < vec.size() - 1 ? ", " : "");
    }
    outfile << "\n";
  };
  write_vector("beta", beta);
  write_vector("beta_int", beta_int);
  write_vector("tau_beta", tau_beta);
  write_vector("nu", nu);
  outfile << "\n";
}

double sample_alpha_cpp(const cpp11::writable::doubles& r_alpha, const cpp11::doubles& z, double sigma, double alpha_prior_sd) {
  int N = r_alpha.size();
  Eigen::Map<const Eigen::VectorXd> r_alpha_map(REAL(r_alpha), N);
  Eigen::Map<const Eigen::VectorXd> z_map(REAL(z), N);
  double sigma2 = sigma * sigma;
  double prior_var = alpha_prior_sd * alpha_prior_sd;
  double sum_z_sq = z_map.squaredNorm();
  double sum_rz = r_alpha_map.dot(z_map);
  double prec_data = sum_z_sq / sigma2;
  double prec_prior = 1.0 / prior_var;
  double postVar = 1.0 / (prec_data + prec_prior);
  double mean_post = postVar * (sum_rz / sigma2);
  return Rf_rnorm(mean_post, std::sqrt(postVar));
}

double sample_beta_j_cpp(const cpp11::writable::doubles& r_beta, const cpp11::doubles& z, const cpp11::doubles& w_j, double tau_j, double sigma, double tau_glob) {
  int N = r_beta.size();
  Eigen::Map<const Eigen::VectorXd> r_beta_map(REAL(r_beta), N);
  Eigen::Map<const Eigen::VectorXd> z_map(REAL(z), N);
  Eigen::Map<const Eigen::VectorXd> w_j_map(REAL(w_j), N);
  Eigen::VectorXd xz = z_map.array() * w_j_map.array();
  double sum_XZ2 = xz.squaredNorm();
  double sum_XZr = xz.dot(r_beta_map);
  double sigma2 = sigma * sigma;
  double priorVar = sigma2 * (tau_j * tau_j) * (tau_glob * tau_glob);
  double prec_data = sum_XZ2 / sigma2;
  double prec_prior = 1.0 / safe_var(priorVar);
  double postVar = 1.0 / (prec_data + prec_prior);
  double postMean = postVar * (sum_XZr / sigma2);
  return Rf_rnorm(postMean, std::sqrt(postVar));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// SECTION 2: SLICE SAMPLER HELPER FUNCTIONS (COMPLETE & VERIFIED)
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

double logPosteriorTauJ(double tau_j, double beta_j, int index, const std::vector<double>& beta_int, const std::vector<double>& tau, const std::vector<std::pair<int, int>>& int_pairs, double tau_int, double sigma, double tau_glob, bool unlink) {
  if (tau_j <= 0.0) return -std::numeric_limits<double>::infinity();
  
  double logPrior = std::log(2.0 / M_PI) - safe_log(1.0 + tau_j * tau_j);
  double sigma2 = sigma * sigma;
  double log2pi = std::log(2.0 * M_PI);
  double var_main = sigma2 * (tau_j * tau_j) * (tau_glob * tau_glob);
  double logLikMain = -0.5 * (log2pi + safe_log(var_main)) - 0.5 * ((beta_j * beta_j) / var_main);
  
  double logLikInter = 0.0;
  if (!unlink && !int_pairs.empty()) {
    for (size_t m = 0; m < int_pairs.size(); m++) {
      int iVar = int_pairs[m].first;
      int jVar = int_pairs[m].second;
      int target_idx = -1;
      if (iVar == index) target_idx = jVar;
      else if (jVar == index) target_idx = iVar;
      
      if (target_idx != -1) {
        double beta_jk = beta_int[m];
        double var_jk = (target_idx == index) ?
        safe_var(sigma2 * tau_j * tau_j * (tau_glob * tau_glob) * tau_int) :
          safe_var(sigma2 * tau_j * tau[target_idx] * (tau_glob * tau_glob) * tau_int);
        logLikInter += -0.5 * (log2pi + safe_log(var_jk)) - 0.5 * (beta_jk * beta_jk / var_jk);
      }
    }
  }
  return logPrior + logLikMain + logLikInter;
}

double sample_tau_j_slice(double tau_old, double beta_j, int index, const std::vector<double>& beta_int, const std::vector<double>& tau, const std::vector<std::pair<int, int>>& int_pairs, double tau_int, double sigma, double tau_glob, bool unlink, double step_out, int max_steps) {
  double logP_old = logPosteriorTauJ(tau_old, beta_j, index, beta_int, tau, int_pairs, tau_int, sigma, tau_glob, unlink);
  if (is_invalid(logP_old)) return tau_old;
  
  double y_slice = logP_old - Rf_rexp(1.0);
  double L = std::max(1e-6, tau_old - step_out);
  double R = tau_old + step_out;
  
  for (int s = 0; s < max_steps; ++s) {
    if (L <= 1e-6 || logPosteriorTauJ(L, beta_j, index, beta_int, tau, int_pairs, tau_int, sigma, tau_glob, unlink) <= y_slice) break;
    L = std::max(1e-6, L - step_out);
  }
  for (int s = 0; s < max_steps; ++s) {
    if (logPosteriorTauJ(R, beta_j, index, beta_int, tau, int_pairs, tau_int, sigma, tau_glob, unlink) <= y_slice) break;
    R += step_out;
  }
  
  for (int rep = 0; rep < max_steps; rep++) {
    double prop = Rf_runif(L, R);
    if (logPosteriorTauJ(prop, beta_j, index, beta_int, tau, int_pairs, tau_int, sigma, tau_glob, unlink) > y_slice) return prop;
    if (prop < tau_old) L = prop; else R = prop;
  }
  return tau_old;
}

double logPosteriorTauGlob(double tau_glob, const std::vector<double>& betas, const std::vector<double>& beta_int, const std::vector<double>& tau, const std::vector<std::pair<int, int>>& int_pairs, double tau_int, double sigma, bool unlink) {
  if (tau_glob <= 0.0) return -std::numeric_limits<double>::infinity();
  
  double logPrior = std::log(2.0 / M_PI) - safe_log(1.0 + tau_glob * tau_glob);
  double sigma2 = sigma * sigma;
  double log2pi = std::log(2.0 * M_PI);
  double logLik = 0.0;
  
  for (size_t s = 0; s < betas.size(); s++) {
    double var_main = safe_var(sigma2 * (tau[s] * tau[s]) * (tau_glob * tau_glob));
    logLik += -0.5 * (log2pi + safe_log(var_main)) - 0.5 * (betas[s] * betas[s] / var_main);
  }
  for (size_t m = 0; m < int_pairs.size(); m++) {
    double var_jk = unlink ?
    safe_var(sigma2 * (tau[betas.size() + m] * tau[betas.size() + m]) * (tau_glob * tau_glob)) :
    safe_var(sigma2 * (tau[int_pairs[m].first] * tau[int_pairs[m].second]) * (tau_glob * tau_glob) * tau_int);
    logLik += -0.5 * (log2pi + safe_log(var_jk)) - 0.5 * (beta_int[m] * beta_int[m] / var_jk);
  }
  return logPrior + logLik;
}

double sample_tau_global_slice(double tau_old, const std::vector<double>& beta, const std::vector<double>& beta_int, const std::vector<double>& tau, const std::vector<std::pair<int, int>>& int_pairs, double tau_int, double sigma, bool unlink, double step_out, int max_steps) {
  double logP_old = logPosteriorTauGlob(tau_old, beta, beta_int, tau, int_pairs, tau_int, sigma, unlink);
  if (is_invalid(logP_old)) return tau_old;
  
  double y_slice = logP_old - Rf_rexp(1.0);
  double L = std::max(1e-6, tau_old - step_out);
  double R = tau_old + step_out;
  
  for (int s = 0; s < max_steps; ++s) {
    if (L <= 1e-6 || logPosteriorTauGlob(L, beta, beta_int, tau, int_pairs, tau_int, sigma, unlink) <= y_slice) break;
    L = std::max(1e-6, L - step_out);
  }
  for (int s = 0; s < max_steps; ++s) {
    if (logPosteriorTauGlob(R, beta, beta_int, tau, int_pairs, tau_int, sigma, unlink) <= y_slice) break;
    R += step_out;
  }
  
  for (int rep = 0; rep < max_steps; rep++) {
    double prop = Rf_runif(L, R);
    if (logPosteriorTauGlob(prop, beta, beta_int, tau, int_pairs, tau_int, sigma, unlink) > y_slice) return prop;
    if (prop < tau_old) L = prop; else R = prop;
  }
  return tau_old;
}

double loglikeTauInt(double tau_int, const std::vector<double>& beta_int, const std::vector<std::pair<int, int>>& int_pairs, const std::vector<double>& tau_main, double sigma, double tau_glob) {
  if (tau_int <= 0.0) return -std::numeric_limits<double>::infinity();
  
  double logp = 0.0;
  const double log2pi = std::log(2.0 * M_PI);
  double sigma2 = sigma * sigma;
  
  for (size_t k = 0; k < int_pairs.size(); k++) {
    double var_ij = safe_var(tau_int * tau_main[int_pairs[k].first] * tau_main[int_pairs[k].second] * sigma2 * (tau_glob * tau_glob));
    logp += -0.5 * (log2pi + std::log(var_ij)) - 0.5 * (beta_int[k] * beta_int[k] / var_ij);
  }
  return logp;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// SECTION 3: MAIN C++ WORKER FUNCTION
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

[[cpp11::register]]
cpp11::writable::doubles updateLinearTreatmentCpp_cpp(
    const cpp11::doubles_matrix<>& X,
    const cpp11::doubles& Z,
    const cpp11::doubles& propensity_train,
    cpp11::writable::doubles residual,
    const cpp11::integers& are_continuous,
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
    double step_out = 0.5) {
  
  // --- INITIAL SETUP ---
  int n = residual.size();
  int p_mod = X.ncol();
  int p_int = beta_int.size();
  double sigma2 = sigma * sigma;
  
  std::vector<std::pair<int, int>> int_pairs;
  if (p_int > 0) {
    int_pairs.reserve(p_int);
    for (int i = 0; i < p_mod; i++) {
      for (int j = i + 1; j < p_mod; j++) {
        if ((are_continuous[i] == 1) || (are_continuous[j] == 1)) {
          int_pairs.push_back(std::make_pair(i, j));
        }
      }
    }
  }
  
  Eigen::Map<Eigen::VectorXd> residual_map(REAL(residual), n);
  Eigen::Map<const Eigen::VectorXd> Z_map(REAL(Z), n);
  Eigen::Map<const Eigen::MatrixXd> X_map(REAL(X), n, p_mod);
  
  // --- GAMMA & ALPHA UPDATES (VECTORIZED) ---
  if (propensity_seperate) {
    Eigen::Map<const Eigen::VectorXd> propensity_map(REAL(propensity_train), n);
    residual_map += propensity_map * gamma;
    gamma = sample_alpha_cpp(residual, propensity_train, sigma, alpha_prior_sd);
    residual_map -= propensity_map * gamma;
  }
  residual_map += Z_map * alpha;
  alpha = sample_alpha_cpp(residual, Z, sigma, alpha_prior_sd);
  residual_map -= Z_map * alpha;
  
  // --- BETA & TAU UPDATES ---
  if (gibbs) { 
    int P_combined = p_mod + p_int;
    
    // Choose the most efficient algorithm based on problem dimensions
    bool use_bhatt_sampler = P_combined >= n;
    
    // Use Eigen::Map for efficient access to R vectors
    Eigen::Map<Eigen::VectorXd> residual_map(REAL(residual), n);
    Eigen::Map<const Eigen::MatrixXd> X_map(REAL(X), n, p_mod);
    Eigen::Map<Eigen::VectorXd> beta_map(REAL(beta), p_mod);
    Eigen::Map<Eigen::VectorXd> beta_int_map(REAL(beta_int), p_int);
    
    // 1. Construct target variable y* = residual + old_fit
    Eigen::VectorXd y_target = residual_map;
    for (int j = 0; j < p_mod; ++j) {
      y_target.array() += Z_map.array() * X_map.col(j).array() * beta_map(j);
    } 
    for (size_t k = 0; k < int_pairs.size(); ++k) {
      y_target.array() += Z_map.array() * X_map.col(int_pairs[k].first).array() * X_map.col(int_pairs[k].second).array() * beta_int_map(k);
    }
    
    // 2. Define prior variance matrix D (excluding sigma^2)
    Eigen::VectorXd D_diag(P_combined);
    for (int j = 0; j < p_mod; ++j) {
      D_diag(j) = safe_var(tau_beta[j] * tau_beta[j] * tau_glob * tau_glob);
    } 
    for (size_t k = 0; k < int_pairs.size(); ++k) {
      double V_k_star = unlink ?
      (tau_beta[p_mod + k] * tau_beta[p_mod + k] * tau_glob * tau_glob) :
      (tau_int * tau_beta[int_pairs[k].first] * tau_beta[int_pairs[k].second] * tau_glob * tau_glob);
      D_diag(p_mod + k) = safe_var(V_k_star);
    }
    Eigen::MatrixXd D_mat = D_diag.asDiagonal();
    
    // This will hold the new sample for all coefficients (beta and beta_int)
    Eigen::VectorXd beta_combined_new_eigen(P_combined);
    
    // --- ALGORITHM CHOICE ---
    if (use_bhatt_sampler) {
      // --- CORRECTED Bhattacharya et al. (2016) Algorithm for p >= n ---
      // This samples from the posterior of beta ~ N(0, sigma^2 * D)
      
      // Step A: Form X_combined matrix
      Eigen::MatrixXd X_combined(n, P_combined);
      for (int j = 0; j < p_mod; ++j) X_combined.col(j) = Z_map.array() * X_map.col(j).array();
      for (size_t k = 0; k < int_pairs.size(); ++k) X_combined.col(p_mod + k) = Z_map.array() * X_map.col(int_pairs[k].first).array() * X_map.col(int_pairs[k].second).array();
      
      // Step B: Rescale the model for a sigma=1 sampler
      Eigen::VectorXd y_target_scaled = y_target / sigma;
      
      // Step C: Draw u from N(0, D) and delta from N(0, I_n)
      Eigen::VectorXd u(P_combined);
      for (int j = 0; j < P_combined; ++j) u(j) = Rf_rnorm(0.0, sqrt(D_diag(j)));
      Eigen::VectorXd delta(n);
      for (int i = 0; i < n; ++i) delta(i) = Rf_rnorm(0.0, 1.0);
      
      // Step D: Compute v = X*u + delta
      Eigen::VectorXd v = X_combined * u + delta;
      
      // Step E: Solve the n x n system (X*D*X^T + I_n) * w = y_target_scaled - v
      Eigen::MatrixXd M_solve = X_combined * D_mat * X_combined.transpose();
      M_solve.diagonal().array() += 1.0; // Add I_n
      
      Eigen::LLT<Eigen::MatrixXd> lltOfM(M_solve);
      if (lltOfM.info() != Eigen::Success) {
        cpp11::warning("Cholesky of n x n system failed in fast sampler. Betas not updated.");
        // Keep beta_combined_new_eigen as its old values (by not updating beta/beta_int)
      } else {
        Eigen::VectorXd w = lltOfM.solve(y_target_scaled - v);
        // Step F: Compute scaled-down sample and scale back up by sigma
        Eigen::VectorXd beta_tilde_new = u + D_mat * X_combined.transpose() * w;
        beta_combined_new_eigen = sigma * beta_tilde_new;
      }
      
    } else { 
      // Step A: Calculate XtX and Xt_y on the fly
      Eigen::MatrixXd XtX = Eigen::MatrixXd::Zero(P_combined, P_combined);
      Eigen::VectorXd Xt_y = Eigen::VectorXd::Zero(P_combined);
      for (int i = 0; i < n; ++i) {
        Eigen::VectorXd x_row_combined(P_combined);
        for (int j = 0; j < p_mod; ++j) x_row_combined(j) = Z_map(i) * X_map(i, j);
        for (size_t k = 0; k < int_pairs.size(); ++k) x_row_combined(p_mod + k) = Z_map(i) * X_map(i, int_pairs[k].first) * X_map(i, int_pairs[k].second);
        XtX.selfadjointView<Eigen::Lower>().rankUpdate(x_row_combined);
        Xt_y += x_row_combined * y_target(i);
      }
      XtX = XtX.selfadjointView<Eigen::Lower>();
      
      // Step B: Form the unscaled posterior precision matrix (X'X + D^-1)
      Eigen::VectorXd D_inv_diag = D_diag.cwiseInverse();
      Eigen::MatrixXd Post_Prec_Unscaled = XtX + Eigen::MatrixXd(D_inv_diag.asDiagonal());
      
      // Step C: Calculate posterior parameters and sample
      Eigen::LLT<Eigen::MatrixXd> lltOfA(Post_Prec_Unscaled);
      if (lltOfA.info() == Eigen::Success) {
        // Posterior mean = (X'X + D^-1)^-1 * X'y
        Eigen::VectorXd post_mean_beta_eigen = lltOfA.solve(Xt_y);
        
        // Get the lower triangular Cholesky factor L, where L*L' = (X'X + D^-1)
        Eigen::MatrixXd L_chol = lltOfA.matrixL();
        Eigen::VectorXd std_normal_draws(P_combined);
        for (int k = 0; k < P_combined; ++k) std_normal_draws(k) = Rf_rnorm(0.0, 1.0);
        
        // --- THIS IS THE CORRECTED PART ---
        // To sample from N(mean, sigma^2 * (L*L')^-1), we compute: 
        // mean + sigma * (L')^-1 * z, where z ~ N(0,I)
        // We solve for x in L' * x = z, which is what L_chol.transpose().solve(...) does.
        Eigen::VectorXd solved_part = L_chol.transpose().template triangularView<Eigen::Upper>().solve(std_normal_draws);
        beta_combined_new_eigen = post_mean_beta_eigen + sigma * solved_part;
        // --- END OF CORRECTION ---
        
      } else {
        cpp11::warning("Cholesky decomposition failed in standard Gibbs sampler. Betas not updated.");
        // If Cholesky fails, we should not update the coefficients.
        // Copy old values into the new vector so the unpacking step uses the old values.
        for(int j=0; j<p_mod; ++j) beta_combined_new_eigen(j) = beta[j];
        for(int k=0; k<p_int; ++k) beta_combined_new_eigen(p_mod + k) = beta_int[k];
      }
      
      // Unpack coefficients AFTER the if/else block to handle both cases
      for(int j=0; j<p_mod; ++j) beta[j] = beta_combined_new_eigen(j);
      for(int k=0; k<p_int; ++k) beta_int[k] = beta_combined_new_eigen(p_mod + k);
    }
    // Recalculate residual based on new beta coefficients
    Eigen::VectorXd new_fit = Eigen::VectorXd::Zero(n);
    for (int j=0; j<p_mod; ++j) new_fit.array() += Z_map.array() * X_map.col(j).array() * beta[j];
    for (size_t k = 0; k < int_pairs.size(); ++k) new_fit.array() += Z_map.array() * X_map.col(int_pairs[k].first).array() * X_map.col(int_pairs[k].second).array() * beta_int[k];
    residual_map = y_target - new_fit;
    // 5. Sample local and global shrinkage parameters
    for(int j = 0; j < p_mod; j++){
      nu[j] = rinvgamma(1.0, 1.0 + 1.0 / safe_var(tau_beta[j]*tau_beta[j]));
      tau_beta[j] = std::sqrt(safe_var(rinvgamma(1.0, (1.0 / safe_var(nu[j])) + (beta[j] * beta[j]) / safe_var(2.0 * tau_glob * tau_glob * sigma2))));
    }
    if(unlink){
      for(size_t k = 0; k < int_pairs.size(); k++){
        int full_idx = p_mod + k;
        nu[full_idx] = rinvgamma(1.0, 1.0 + 1.0 / safe_var(tau_beta[full_idx]*tau_beta[full_idx]));
        tau_beta[full_idx] = std::sqrt(safe_var(rinvgamma(1.0, (1.0 / safe_var(nu[full_idx])) + (beta_int[k] * beta_int[k]) / safe_var(2.0 * tau_glob * tau_glob * sigma2))));
      }
    }
    if (global_shrink) {
      xi = rinvgamma(1.0, 1.0 + 1.0 / safe_var(tau_glob*tau_glob));
      double sum_scaled_sq_betas = 0.0;
      for(int j = 0; j < p_mod; j++) sum_scaled_sq_betas += (beta[j]*beta[j]) / safe_var(tau_beta[j] * tau_beta[j]);
      if(unlink){
        for(int k = 0; k < p_int; k++) sum_scaled_sq_betas += (beta_int[k] * beta_int[k]) / safe_var(tau_beta[p_mod + k] * tau_beta[p_mod + k]);
      } else {
        for(size_t k = 0; k < int_pairs.size(); ++k) {
          sum_scaled_sq_betas += (beta_int[k] * beta_int[k]) / safe_var(tau_int * tau_beta[int_pairs[k].first] * tau_beta[int_pairs[k].second]);
        }
      }
      double shape_tau_glob = (static_cast<double>(p_mod + p_int) + 1.0) / 2.0;
      double rate_tau_glob = (1.0 / safe_var(xi)) + (1.0 / safe_var(2.0 * sigma2)) * sum_scaled_sq_betas;
      tau_glob = std::sqrt(safe_var(rinvgamma(shape_tau_glob, rate_tau_glob)));
    }
    
  } else { 

    int P_combined = p_mod + p_int;
    Eigen::Map<Eigen::VectorXd> beta_map(REAL(beta), p_mod);
    Eigen::Map<Eigen::VectorXd> beta_int_map(REAL(beta_int), p_int);
    
    // 1. Construct target variable y* = residual + old_fit
    Eigen::VectorXd y_target = residual_map;
    for (int j = 0; j < p_mod; ++j) {
      y_target.array() += Z_map.array() * X_map.col(j).array() * beta_map(j);
    }
    for (size_t k = 0; k < int_pairs.size(); ++k) {
      y_target.array() += Z_map.array() * X_map.col(int_pairs[k].first).array() * X_map.col(int_pairs[k].second).array() * beta_int_map(k);
    } 
    
    // 2. Define prior variance D (excluding sigma^2), with tau_int = 1.0
    Eigen::VectorXd D_diag(P_combined);
    for (int j = 0; j < p_mod; ++j) {
      D_diag(j) = safe_var(tau_beta[j] * tau_beta[j] * tau_glob * tau_glob);
    } 
    for (size_t k = 0; k < int_pairs.size(); ++k) {
      // For linked interactions, tau_int is now treated as 1.0
      double V_k_star = unlink ?
      (tau_beta[p_mod + k] * tau_beta[p_mod + k] * tau_glob * tau_glob) : 
      (1.0 * tau_beta[int_pairs[k].first] * tau_beta[int_pairs[k].second] * tau_glob * tau_glob);
      D_diag(p_mod + k) = safe_var(V_k_star);
    } 
    
    // 3. Block sample all beta coefficients
    Eigen::VectorXd beta_combined_new_eigen(P_combined);
     
    // Step A: Calculate XtX and Xt_y on the fly
    Eigen::MatrixXd XtX = Eigen::MatrixXd::Zero(P_combined, P_combined);
    Eigen::VectorXd Xt_y = Eigen::VectorXd::Zero(P_combined);
    for (int i = 0; i < n; ++i) {
      Eigen::VectorXd x_row_combined(P_combined);
      for (int j = 0; j < p_mod; ++j) x_row_combined(j) = Z_map(i) * X_map(i, j);
      for (size_t k = 0; k < int_pairs.size(); ++k) x_row_combined(p_mod + k) = Z_map(i) * X_map(i, int_pairs[k].first) * X_map(i, int_pairs[k].second);
      XtX.selfadjointView<Eigen::Lower>().rankUpdate(x_row_combined);
      Xt_y += x_row_combined * y_target(i);
    } 
    XtX = XtX.selfadjointView<Eigen::Lower>();
     
    // Step B: Form the unscaled posterior precision matrix (X'X + D^-1)
    Eigen::VectorXd D_inv_diag = D_diag.cwiseInverse();
    Eigen::MatrixXd Post_Prec_Unscaled = XtX + Eigen::MatrixXd(D_inv_diag.asDiagonal());
     
    // Step C: Calculate posterior parameters and sample
    Eigen::LLT<Eigen::MatrixXd> lltOfA(Post_Prec_Unscaled);
    if (lltOfA.info() == Eigen::Success) {
      Eigen::VectorXd post_mean_beta_eigen = lltOfA.solve(Xt_y);
      Eigen::MatrixXd L_chol = lltOfA.matrixL();
      Eigen::VectorXd std_normal_draws(P_combined);
      for (int k = 0; k < P_combined; ++k) std_normal_draws(k) = Rf_rnorm(0.0, 1.0);
      beta_combined_new_eigen = post_mean_beta_eigen + sigma * L_chol.transpose().template triangularView<Eigen::Upper>().solve(std_normal_draws);
    } else { 
      cpp11::warning("Cholesky decomposition failed in beta block sampling. Betas not updated.");
      for(int j=0; j<p_mod; ++j) beta_combined_new_eigen(j) = beta[j];
      for(int k=0; k<p_int; ++k) beta_combined_new_eigen(p_mod + k) = beta_int[k];
    }
     
    // Unpack new coefficients
    for(int j=0; j<p_mod; ++j) beta[j] = beta_combined_new_eigen(j);
    for(int k=0; k<p_int; ++k) beta_int[k] = beta_combined_new_eigen(p_mod + k);
    
    // 4. Update residual map based on new coefficients
    Eigen::VectorXd new_fit = Eigen::VectorXd::Zero(n);
    for (int j=0; j<p_mod; ++j) new_fit.array() += Z_map.array() * X_map.col(j).array() * beta[j];
    for (size_t k = 0; k < int_pairs.size(); ++k) new_fit.array() += Z_map.array() * X_map.col(int_pairs[k].first).array() * X_map.col(int_pairs[k].second).array() * beta_int[k];
    residual_map = y_target - new_fit;
     
    // 5. Sample tau_beta using slice sampler, now conditional on the new block of betas
    std::vector<double> beta_int_std(beta_int.begin(), beta_int.end());
    std::vector<double> tau_beta_std(tau_beta.begin(), tau_beta.end());
    
    for (int j = 0; j < p_mod; j++) {
      double tb_j_new = sample_tau_j_slice(tau_beta[j], beta[j], j, beta_int_std, tau_beta_std, int_pairs, 1.0, sigma, tau_glob, unlink, step_out, max_steps);
      tau_beta[j] = tb_j_new;
      tau_beta_std[j] = tb_j_new;
    } 
    
    if (unlink) {
      for (size_t k = 0; k < int_pairs.size(); k++) {
        int full_idx = p_mod + k;
        double tb_k_new = sample_tau_j_slice(tau_beta[full_idx], beta_int[k], full_idx, beta_int_std, tau_beta_std, int_pairs, 1.0, sigma, tau_glob, unlink, step_out, max_steps);
        tau_beta[full_idx] = tb_k_new;
        tau_beta_std[full_idx] = tb_k_new;
      }
    } 
    
    // 6. Update global shrinkage tau_glob if applicable
    if (global_shrink) {
      std::vector<double> beta_std_current(beta.begin(), beta.end());
      tau_glob = sample_tau_global_slice(tau_glob, beta_std_current, beta_int_std, tau_beta_std, int_pairs, 1.0, sigma, unlink, step_out, max_steps);
    } 
    
    // Ensure tau_int is set to 1.0 as it's disregarded in this path. The argument is ignored but we set it here for clarity.
    tau_int = 1.0; 
  } 
  // --- CONSOLIDATE AND RETURN RESULTS ---
  size_t total_size = 5 + beta.size() + beta_int.size() + tau_beta.size() + nu.size() + residual.size();
  cpp11::writable::doubles output;
  output.reserve(total_size);
  output.push_back(alpha);
  output.push_back(tau_int);
  output.push_back(tau_glob);
  output.push_back(gamma);
  output.push_back(xi);
  for (double val : beta) output.push_back(val);
  for (double val : beta_int) output.push_back(val);
  for (double val : tau_beta) output.push_back(val);
  for (double val : nu) output.push_back(val);
  for (int i = 0; i < n; ++i) output.push_back(residual_map[i]);
  
  if (save_output) {
    std::ofstream outfile("output_log.txt", std::ios::app);
    save_output_vector_labeled(outfile, alpha, tau_int, tau_glob, gamma, xi, sigma, beta, beta_int, tau_beta, nu, index);
    outfile.close();
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
