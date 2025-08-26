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
    const cpp11::r_vector<int>& are_continuous,
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

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// SECTION 3: MCMC SAMPLER WRAPPER
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Eigen::VectorXd calculate_component_fit(
    const doubles_matrix<>& X, const Eigen::VectorXd& moderator, const doubles& beta,
    const doubles& beta_int, const cpp11::r_vector<int>& are_continuous, double alpha, bool propensity_seperate,
    double gamma, const doubles& propensity_scores_r) {
  int n = X.nrow();
  int p_mod = X.ncol();
  Eigen::Map<const Eigen::MatrixXd> X_map(REAL(X), n, p_mod);
  Eigen::Map<const Eigen::VectorXd> beta_map(REAL(beta), beta.size());
  
  Eigen::VectorXd fit = Eigen::VectorXd::Zero(n);
  for (int j = 0; j < p_mod; ++j) {
    fit.array() += moderator.array() * X_map.col(j).array() * beta_map(j);
  }
  
  if (beta_int.size() > 0) {
    Eigen::Map<const Eigen::VectorXd> beta_int_map(REAL(beta_int), beta_int.size());
    int p_int_count = 0;
    for (int i = 0; i < p_mod; i++) {
      for (int j = i + 1; j < p_mod; j++) {
        if ((are_continuous[i] == 1) || (are_continuous[j] == 1)) {
          if (p_int_count < beta_int.size()) {
            fit.array() += moderator.array() * X_map.col(i).array() * X_map.col(j).array() * beta_int_map(p_int_count);
            p_int_count++;
          }
        }
      }
    }
  }
  fit.array() += moderator.array() * alpha;
  if(propensity_seperate){
    Eigen::Map<const Eigen::VectorXd> propensity_scores(REAL(propensity_scores_r), n);
    fit.array() += moderator.array()*propensity_scores.array()*gamma;
  }
  return fit; 
}
// 
// [[cpp11::register]]
// cpp11::list run_mcmc_sampler_2(
//     const cpp11::matrix<cpp11::r_vector<double>, cpp11::by_column>& X_r,
//     const cpp11::r_vector<double>& Y_r,
//     const cpp11::r_vector<double>& Z_r,
//     const cpp11::r_vector<int>& are_continuous,
//     const cpp11::r_vector<double>& propensity_scores_r,
//     int num_iterations, int burn_in,
//     bool mu_gibbs, bool mu_global_shrink, bool mu_unlink, int mu_p_int,
//     bool tau_gibbs, bool tau_global_shrink, bool tau_unlink, int tau_p_int,
//     bool propensity_separate = false, double alpha_prior_sd = 10.0,
//     double sigma_init = 1.0,
//     double alpha_mu_init = 0.0, double alpha_tau_init = 0.0,
//     const cpp11::r_vector<double>& beta_mu_init = {}, const cpp11::r_vector<double>& beta_int_mu_init = {},
//     const cpp11::r_vector<double>& beta_tau_init = {}, const cpp11::r_vector<double>& beta_int_tau_init = {},
//     double tau_glob_mu_init = 1.0, double tau_glob_tau_init = 1.0
// ){
//   
//   int n = Y_r.size();
//   int p_mod = X_r.ncol();
//   
//   // --- DATA PREPARATION: Standardize Y ---
//   writable::doubles Y_standardized_r(n);
//   double y_mean = 0.0;
//   for(double val : Y_r) { y_mean += val; }
//   y_mean /= n;
//   
//   double y_sd = 0.0;
//   for(double val : Y_r) { y_sd += (val - y_mean) * (val - y_mean); }
//   y_sd = std::sqrt(y_sd / (n - 1));
//   if (y_sd < 1e-6) y_sd = 1.0;
//   
//   for(int i = 0; i < n; ++i) { Y_standardized_r[i] = (Y_r[i] - y_mean) / y_sd; }
//   
//   // --- INITIALIZE PARAMETERS ---
//   writable::doubles beta_mu(beta_mu_init);
//   writable::doubles beta_int_mu(beta_int_mu_init);
//   writable::doubles tau_beta_mu(p_mod + mu_p_int); std::fill(tau_beta_mu.begin(), tau_beta_mu.end(), 1.0);
//   writable::doubles nu_mu(p_mod + mu_p_int); std::fill(nu_mu.begin(), nu_mu.end(), 1.0);
//   double alpha_mu = alpha_mu_init, gamma_mu = 0.0, xi_mu = 1.0, tau_int_mu = 1.0, tau_glob_mu = tau_glob_mu_init;
//   
//   writable::doubles beta_tau(beta_tau_init);
//   writable::doubles beta_int_tau(beta_int_tau_init);
//   writable::doubles tau_beta_tau(p_mod + tau_p_int); std::fill(tau_beta_tau.begin(), tau_beta_tau.end(), 1.0);
//   writable::doubles nu_tau(p_mod + tau_p_int); std::fill(nu_tau.begin(), nu_tau.end(), 1.0);
//   double alpha_tau = alpha_tau_init, gamma_tau = 0.0, xi_tau = 1.0, tau_int_tau = 1.0, tau_glob_tau = tau_glob_tau_init;
//   
//   double sigma = sigma_init / y_sd; // Scale initial sigma
//   
//   writable::doubles ones(n); std::fill(ones.begin(), ones.end(), 1.0);
//   
//   // --- STORAGE FOR SAMPLES ---
//   std::vector<std::vector<double>> s_beta_mu, s_beta_int_mu, s_tau_beta_mu;
//   std::vector<double> s_alpha_mu, s_tau_glob_mu;
//   std::vector<std::vector<double>> s_beta_tau, s_beta_int_tau, s_tau_beta_tau;
//   std::vector<double> s_alpha_tau, s_tau_glob_tau;
//   std::vector<double> s_sigma;
//   
//   // --- MCMC LOOP ---
//   for (int iter = 0; iter < num_iterations; ++iter) {
//     if ((iter + 1) % 100 == 0) {
//       Rprintf("Iteration: %d / %d | Sigma: %.4f\n", iter + 1, num_iterations, sigma * y_sd);
//     }
//     
//     // 1. UPDATE MU(X)
//     Eigen::VectorXd tau_fit = calculate_component_fit(X_r, Eigen::Map<const Eigen::VectorXd>(REAL(Z_r), n), beta_tau, beta_int_tau, are_continuous, alpha_tau, false, gamma_tau, propensity_scores_r);
//     writable::doubles residual_for_mu(n);
//     for(int i=0; i<n; ++i) residual_for_mu[i] = Y_standardized_r[i] - tau_fit[i];
//     
//     writable::doubles mu_output = updateLinearTreatmentCpp_cpp(
//       X_r, ones, propensity_scores_r, residual_for_mu, are_continuous,
//       alpha_mu, beta_mu, gamma_mu, beta_int_mu, tau_beta_mu, nu_mu, xi_mu, tau_int_mu,
//       sigma, alpha_prior_sd, tau_glob_mu, mu_global_shrink, mu_unlink,
//       propensity_separate, mu_gibbs, false, iter, 50, 0.5);
//     
//     int offset = 0;
//     alpha_mu = mu_output[offset++];
//     tau_int_mu = mu_output[offset++];
//     tau_glob_mu = mu_output[offset++];
//     gamma_mu = mu_output[offset++];
//     xi_mu = mu_output[offset++];
//     for(int j=0; j<p_mod; ++j) { beta_mu[j] = mu_output[offset++]; }
//     for(int j=0; j<mu_p_int; ++j) { beta_int_mu[j] = mu_output[offset++]; }
//     for(int j=0; j<p_mod + mu_p_int; ++j) { tau_beta_mu[j] = mu_output[offset++]; }
//     for(int j=0; j<p_mod + mu_p_int; ++j) { nu_mu[j] = mu_output[offset++]; }
//     
//     // 2. UPDATE TAU(X)
//     Eigen::VectorXd mu_fit = calculate_component_fit(X_r, Eigen::Map<const Eigen::VectorXd>(REAL(ones), n), beta_mu, beta_int_mu, are_continuous, alpha_mu, propensity_separate, gamma_mu, propensity_scores_r);
//     writable::doubles residual_for_tau(n);
//     for(int i=0; i<n; ++i) residual_for_tau[i] = Y_standardized_r[i] - mu_fit[i];
//     
//     writable::doubles tau_output = updateLinearTreatmentCpp_cpp(
//       X_r, Z_r, propensity_scores_r, residual_for_tau, are_continuous,
//       alpha_tau, beta_tau, gamma_tau, beta_int_tau, tau_beta_tau, nu_tau, xi_tau, tau_int_tau,
//       sigma, alpha_prior_sd, tau_glob_tau, tau_global_shrink, tau_unlink,
//       false, tau_gibbs, false, iter, 50, 0.5);
//     
//     offset = 0;
//     alpha_tau = tau_output[offset++];
//     tau_int_tau = tau_output[offset++];
//     tau_glob_tau = tau_output[offset++];
//     gamma_tau = tau_output[offset++];
//     xi_tau = tau_output[offset++];
//     for(int j=0; j<p_mod; ++j) { beta_tau[j] = tau_output[offset++]; }
//     for(int j=0; j<tau_p_int; ++j) { beta_int_tau[j] = tau_output[offset++]; }
//     for(int j=0; j<p_mod + tau_p_int; ++j) { tau_beta_tau[j] = tau_output[offset++]; }
//     for(int j=0; j<p_mod + tau_p_int; ++j) { nu_tau[j] = tau_output[offset++]; }
//     
//     // 3. UPDATE SIGMA
//     Eigen::Map<const Eigen::VectorXd> Y_standardized_map_const(REAL(Y_standardized_r), n);
//     Eigen::VectorXd final_residual = Y_standardized_map_const - mu_fit - tau_fit;
//     if (final_residual.hasNaN()) stop("NaN generated in final_residual at iteration %d", iter + 1);
//     
//     double shape, rate;
//     bool any_gibbs = mu_gibbs || tau_gibbs;
//     
//     if (any_gibbs) {
//       shape = (double)n / 2.0;
//       rate = final_residual.squaredNorm() / 2.0;
//     } else {
//       double sum_scaled_sq_betas = 0.0;
//       for(int j=0; j<p_mod; ++j) {
//         sum_scaled_sq_betas += (beta_mu[j] * beta_mu[j]) / safe_var(tau_beta_mu[j] * tau_beta_mu[j] * tau_glob_mu * tau_glob_mu);
//       }
//       for(int j=0; j<p_mod; ++j) {
//         sum_scaled_sq_betas += (beta_tau[j] * beta_tau[j]) / safe_var(tau_beta_tau[j] * tau_beta_tau[j] * tau_glob_tau * tau_glob_tau);
//       }
//       int total_p = p_mod * 2; // Simplified for now
//       shape = (double)(n + total_p) / 2.0;
//       rate = 0.5 * (final_residual.squaredNorm() + sum_scaled_sq_betas);
//     }
//     
//     sigma = std::sqrt(rinvgamma(shape, rate));
//     if (is_invalid(sigma)) stop("NaN generated in sigma at iteration %d. Rate was %.4f", iter + 1, rate);
//     if (sigma < 1e-6) sigma = 1e-6;
//      
//     if (iter >= burn_in) {
//       s_alpha_mu.push_back((alpha_mu * y_sd) + y_mean);
//       std::vector<double> temp_beta_mu(p_mod);
//       for(int j=0; j<p_mod; ++j) temp_beta_mu[j] = beta_mu[j] * y_sd;
//       s_beta_mu.push_back(temp_beta_mu);
//       
//       if(mu_p_int > 0) {
//         std::vector<double> temp_beta_int_mu(mu_p_int);
//         for(int j=0; j<mu_p_int; ++j) temp_beta_int_mu[j] = beta_int_mu[j] * y_sd;
//         s_beta_int_mu.push_back(temp_beta_int_mu);
//       }
//       s_tau_beta_mu.push_back(std::vector<double>(tau_beta_mu.begin(), tau_beta_mu.end()));
//       s_tau_glob_mu.push_back(tau_glob_mu);
//       
//       s_alpha_tau.push_back(alpha_tau * y_sd);
//       std::vector<double> temp_beta_tau(p_mod);
//       for(int j=0; j<p_mod; ++j) temp_beta_tau[j] = beta_tau[j] * y_sd;
//       s_beta_tau.push_back(temp_beta_tau);
//       
//       if(tau_p_int > 0) {
//         std::vector<double> temp_beta_int_tau(tau_p_int);
//         for(int j=0; j<tau_p_int; ++j) temp_beta_int_tau[j] = beta_int_tau[j] * y_sd;
//         s_beta_int_tau.push_back(temp_beta_int_tau);
//       }
//       s_tau_beta_tau.push_back(std::vector<double>(tau_beta_tau.begin(), tau_beta_tau.end()));
//       s_tau_glob_tau.push_back(tau_glob_tau);
//       
//       s_sigma.push_back(sigma * y_sd);
//     }
//   }
//   
//   auto to_matrix = [](const std::vector<std::vector<double>>& vec) {
//     if (vec.empty() || vec[0].empty()) return writable::doubles_matrix<>(0, 0);
//     R_xlen_t n_rows = vec.size();
//     R_xlen_t n_cols = vec[0].size();
//     writable::doubles_matrix<> mat(n_rows, n_cols);
//     for (R_xlen_t i = 0; i < n_rows; ++i) {
//       for (R_xlen_t j = 0; j < n_cols; ++j) {
//         mat(i, j) = vec[i][j];
//       }
//     }
//     return mat;
//   };
//   
//   return writable::list({
//     "mu_samples"_nm = writable::list({
//       "alpha"_nm = writable::doubles(s_alpha_mu.begin(), s_alpha_mu.end()),
//         "beta"_nm = to_matrix(s_beta_mu),
//         "beta_int"_nm = to_matrix(s_beta_int_mu),
//         "tau_beta"_nm = to_matrix(s_tau_beta_mu),
//         "tau_glob"_nm = writable::doubles(s_tau_glob_mu.begin(), s_tau_glob_mu.end())
//     }),
//     "tau_samples"_nm = writable::list({
//       "alpha"_nm = writable::doubles(s_alpha_tau.begin(), s_alpha_tau.end()),
//         "beta"_nm = to_matrix(s_beta_tau),
//         "beta_int"_nm = to_matrix(s_beta_int_tau),
//         "tau_beta"_nm = to_matrix(s_tau_beta_tau),
//         "tau_glob"_nm = writable::doubles(s_tau_glob_tau.begin(), s_tau_glob_tau.end())
//     }),
//     "sigma_samples"_nm = writable::doubles(s_sigma.begin(), s_sigma.end())
//   });
// }