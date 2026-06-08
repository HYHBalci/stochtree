#include <cpp11.hpp>
#include <cpp11/doubles.hpp>
#include <cpp11/matrix.hpp>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <Rmath.h>
#include <fstream>
#include <R_ext/Random.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace cpp11;

// -----------------------------------------------------------------------------
// SECTION 1: UTILITY AND HELPER FUNCTIONS
// -----------------------------------------------------------------------------

inline double safe_log_linear(double x) {
  return std::log(std::max(x, 1e-12));
}

inline double safe_var_linear(double x) {
  return std::max(x, 1e-12);
} 

inline bool is_invalid_linear(double x) {
  return !std::isfinite(x);
} 

void save_output_vector_labeled(
    std::ofstream& outfile,
    double alpha, double tau_int, double tau_glob, double xi, double sigma,
    const cpp11::writable::doubles& beta, const cpp11::writable::doubles& beta_int,
    const cpp11::writable::doubles& tau_beta, const cpp11::writable::doubles& nu, int index) {
  if (!outfile.is_open()) return;
  outfile << "=== Iteration " << index << " ===\n";
  outfile << "alpha: " << alpha << "\n";
  outfile << "tau_int: " << tau_int << "\n";
  outfile << "tau_glob: " << tau_glob << "\n";
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


// Renamed to avoid linking conflicts if defined elsewhere
double rinvgamma_linear(double shape, double scale) {
  if (shape <= 0.0 || scale <= 0.0) {
    stop("Shape and scale for rinvgamma must be positive.");
  } 
  double g = Rf_rgamma(shape, 1.0 / scale);
  return (g > 1e-12) ? 1.0 / g : 1e12;
} 

double sample_alpha_linear(const writable::doubles& r_alpha, const doubles& z, double sigma, double alpha_prior_sd) {
  int N = r_alpha.size();
  Eigen::Map<const Eigen::VectorXd> r_alpha_map((const double*)r_alpha.data(), N);
  Eigen::Map<const Eigen::VectorXd> z_map((const double*)z.data(), N);
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

// -----------------------------------------------------------------------------
// SECTION 2: SLICE SAMPLER HELPERS (Kept for compatibility, though currently unused in Gibbs path)
// -----------------------------------------------------------------------------
// (Simplified logic included to match original structure)

double logPosteriorTauJ(double tau_j, double beta_j, int index, const std::vector<double>& beta_int, const std::vector<double>& tau, const std::vector<std::pair<int, int>>& int_pairs, double tau_int, double sigma, double tau_glob, bool unlink) {
  if (tau_j <= 0.0) return -std::numeric_limits<double>::infinity();
  
  double logPrior = std::log(2.0 / M_PI) - safe_log_linear(1.0 + tau_j * tau_j);
  double sigma2 = sigma * sigma;
  double log2pi = std::log(2.0 * M_PI);
  double var_main = sigma2 * (tau_j * tau_j) * (tau_glob * tau_glob);
  double logLikMain = -0.5 * (log2pi + safe_log_linear(var_main)) - 0.5 * ((beta_j * beta_j) / var_main);
  
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
        safe_var_linear(sigma2 * tau_j * tau_j * (tau_glob * tau_glob) * tau_int) :
          safe_var_linear(sigma2 * tau_j * tau[target_idx] * (tau_glob * tau_glob) * tau_int);
        logLikInter += -0.5 * (log2pi + safe_log_linear(var_jk)) - 0.5 * (beta_jk * beta_jk / var_jk);
      }
    }
  } 
  return logPrior + logLikMain + logLikInter;
} 

// -----------------------------------------------------------------------------
// SECTION 3: MAIN C++ WORKER FUNCTION (CENTERED)
// -----------------------------------------------------------------------------

[[cpp11::register]]
writable::list updateLinearTreatmentCpp_cpp(
    const doubles_matrix<>& X,
    const doubles_matrix<>& Phi,
    const doubles& Z,
    const doubles& propensity_train,
    writable::doubles residual,
    const integers& are_continuous,
    double alpha,
    double gamma_prop,
    writable::doubles beta,
    writable::doubles beta_int,
    writable::doubles gamma,
    writable::doubles tau_beta,
    writable::doubles tau_gamma,
    writable::doubles nu,
    writable::doubles nu_gamma,
    double xi,
    double tau_int,
    double sigma,
    double alpha_prior_sd,
    double tau_glob,
    const std::string& sample_global_prior,
    bool unlink,
    bool gibbs,
    bool save_output,
    int index,
    int max_steps,
    double step_out,
    const std::string& propensity_seperate,
    bool regularize_ATE,
    double hn_scale,
    bool use_prognostic_shapley) {
  
  GetRNGstate();
  
  int n = residual.size();
  int p_mod = X.ncol();
  int p_prog = use_prognostic_shapley ? Phi.ncol() : 0; 
  int p_int = beta_int.size();
  
  if (sigma < 1e-6) sigma = 1e-6;
  double sigma2 = sigma * sigma;
  
  std::vector<std::pair<int, int>> int_pairs;
  if (p_int > 0) {
    int_pairs.reserve(p_int);
    for (int i = 0; i < p_mod; i++) {
      for (int j = i + 1; j < p_mod; j++) {
        if(!((propensity_seperate == "tau") & ((i == (p_mod -1))| (j == (p_mod-1))))){
          if ((are_continuous[i] == 1) || (are_continuous[j] == 1)) {
            int_pairs.push_back(std::make_pair(i, j));
          }
        }
      }
    } 
  }
  
  Eigen::Map<Eigen::VectorXd> residual_map((double*)residual.data(), n);
  Eigen::Map<const Eigen::VectorXd> Z_map((const double*)Z.data(), n);
  Eigen::Map<const Eigen::MatrixXd> X_map((const double*)X.data(), n, p_mod);
  int phi_cols = (p_prog > 0) ? p_prog : 1;
  Eigen::Map<const Eigen::MatrixXd> Phi_map((const double*)Phi.data(), n, phi_cols);
   
  // --- 1. Marginal Updates (Alpha / Propensity) ---
  if (propensity_seperate == "mu") {
    Eigen::Map<const Eigen::VectorXd> prop_map((const double*)propensity_train.data(), n);
    residual_map += prop_map * gamma_prop;
    gamma_prop = sample_alpha_linear(residual, propensity_train, sigma, 10.0);
    residual_map -= prop_map * gamma_prop;
  }  
  
  if (!regularize_ATE){
    residual_map += Z_map * alpha;
    alpha = sample_alpha_linear(residual, Z, sigma, alpha_prior_sd);
    residual_map -= Z_map * alpha;
  } 
   
  // --- 2. Joint Gibbs ---
  int offset_alpha = regularize_ATE ? 1 : 0;
  int offset_beta = offset_alpha;
  int offset_beta_int = offset_beta + p_mod;
  int offset_gamma = offset_beta_int + p_int;
  int P_combined = offset_gamma + p_prog; 
  
  if (gibbs) {
    Eigen::Map<Eigen::VectorXd> beta_map((double*)beta.data(), p_mod);
    Eigen::Map<Eigen::VectorXd> beta_int_map((double*)beta_int.data(), p_int);
    Eigen::Map<Eigen::VectorXd> gamma_map((double*)gamma.data(), (p_prog > 0 ? p_prog : 1));
    
    // Construct Target Y
    Eigen::VectorXd y_target = residual_map;
    if(regularize_ATE) y_target.array() += Z_map.array() * alpha;
    for (int j = 0; j < p_mod; ++j) y_target.array() += Z_map.array() * X_map.col(j).array() * beta_map(j);
    for (size_t k = 0; k < int_pairs.size(); ++k) y_target.array() += Z_map.array() * X_map.col(int_pairs[k].first).array() * X_map.col(int_pairs[k].second).array() * beta_int_map(k);
    
    if (use_prognostic_shapley) {
      for(int l=0; l<p_prog; ++l) y_target.array() += Phi_map.col(l).array() * gamma_map(l);
    } 
    
    // Construct Diagonal Prior Variance D_diag
    Eigen::VectorXd D_diag(P_combined);
    
    // Alpha & Beta priors
    for (int j = 0; j < p_mod + regularize_ATE; ++j) {
      D_diag(j) = safe_var_linear(tau_beta[j] * tau_beta[j] * tau_glob * tau_glob);
    } 
    // Interaction priors
    for (size_t k = 0; k < int_pairs.size(); ++k) {
      double V_k_star = unlink ?
      (tau_beta[offset_beta_int + k] * tau_beta[offset_beta_int + k] * tau_glob * tau_glob) :
      (tau_int * tau_beta[offset_beta + int_pairs[k].first] * tau_beta[offset_beta + int_pairs[k].second] * tau_glob * tau_glob);
      D_diag(offset_beta_int + k) = safe_var_linear(V_k_star);
    } 
    // Gamma priors
    if (use_prognostic_shapley) {
      for(int l=0; l<p_prog; ++l) {
        D_diag(offset_gamma + l) = safe_var_linear(tau_gamma[l] * tau_gamma[l] * tau_glob * tau_glob);
      }
    }
    
    // NEW: Algorithm Selection (Bhattacharya for P > N)
    bool use_bhatt_sampler = P_combined >= n;
    Eigen::VectorXd combined_coeffs_new(P_combined);
    bool update_success = false;
     
    if (use_bhatt_sampler) {
      // --- Bhattacharya Sampler (O(N^2 P) instead of O(P^3)) ---
      
      // 1. Build X_combined
      Eigen::MatrixXd X_combined(n, P_combined);
      if(regularize_ATE) X_combined.col(0) = Z_map;
      for(int j=0; j<p_mod; ++j) X_combined.col(offset_beta + j) = Z_map.array() * X_map.col(j).array();
      for(size_t k=0; k<int_pairs.size(); ++k) X_combined.col(offset_beta_int + k) = Z_map.array() * X_map.col(int_pairs[k].first).array() * X_map.col(int_pairs[k].second).array();
      if(use_prognostic_shapley) {
        for(int l=0; l<p_prog; ++l) X_combined.col(offset_gamma + l) = Phi_map.col(l);
      }
      
      Eigen::MatrixXd D_scaled = D_diag.asDiagonal();
      D_scaled /= sigma2;
      Eigen::VectorXd D_scaled_sqrt = D_diag.cwiseSqrt() / sigma;
      
      // 2. Sample u ~ N(0, D/sigma2)
      Eigen::VectorXd u(P_combined);
      for(int j=0; j<P_combined; ++j) u(j) = Rf_rnorm(0.0, D_scaled_sqrt(j));
      
      // 3. Sample delta ~ N(0, I)
      Eigen::VectorXd delta(n);
      for(int i=0; i<n; ++i) delta(i) = Rf_rnorm(0.0, 1.0);
      
      // 4. v = X*u + delta
      Eigen::VectorXd v = X_combined * u + delta;
      
      // 5. M = X D X' + I (NxN matrix)
      Eigen::MatrixXd M = X_combined * D_scaled * X_combined.transpose();
      M.diagonal().array() += 1.0;
      
      // 6. Robust Solve
      Eigen::LLT<Eigen::MatrixXd> llt(M);
      if(llt.info() != Eigen::Success) {
        M.diagonal().array() += 1e-6; // Jitter
        llt.compute(M);
      }
      
      if(llt.info() == Eigen::Success) {
        Eigen::VectorXd y_scaled = y_target / sigma;
        Eigen::VectorXd w = llt.solve(y_scaled - v);
        combined_coeffs_new = u + D_scaled * X_combined.transpose() * w;
        combined_coeffs_new *= sigma; // Scale back
        update_success = true;
      } else {
        cpp11::warning("Bhattacharya sampler failed. Keeping old values.");
      }
      
    } else {
      // --- Standard Gibbs (Row-wise updates) ---
      
      Eigen::MatrixXd XtX = Eigen::MatrixXd::Zero(P_combined, P_combined);
      Eigen::VectorXd Xt_y = Eigen::VectorXd::Zero(P_combined);
      
      for (int i = 0; i < n; ++i) {
        Eigen::VectorXd x_row(P_combined);
        x_row.setZero();
        if(regularize_ATE) x_row(0) = Z_map(i);
        for(int j=0; j<p_mod; ++j) x_row(offset_beta + j) = Z_map(i) * X_map(i, j);
        for(size_t k=0; k<int_pairs.size(); ++k) x_row(offset_beta_int + k) = Z_map(i) * X_map(i, int_pairs[k].first) * X_map(i, int_pairs[k].second);
        if (use_prognostic_shapley) {
          for(int l=0; l<p_prog; ++l) x_row(offset_gamma + l) = Phi_map(i, l);
        }
        
        XtX.selfadjointView<Eigen::Lower>().rankUpdate(x_row);
        Xt_y += x_row * y_target(i);
      }
      XtX = XtX.selfadjointView<Eigen::Lower>();
      
      // Robust Linear Solve
      Eigen::MatrixXd Prec = XtX / sigma2;
      Eigen::VectorXd prior_prec_diag = D_diag.cwiseInverse();
      // Clamp extreme precisions
      for(int k=0; k<prior_prec_diag.size(); ++k) if(prior_prec_diag(k) > 1e12) prior_prec_diag(k) = 1e12;
      
      Prec.diagonal() += prior_prec_diag;
      
      Eigen::LLT<Eigen::MatrixXd> lltOfA(Prec);
      if (lltOfA.info() != Eigen::Success) {
        Prec.diagonal().array() += 1e-6; // Jitter
        lltOfA.compute(Prec);
      }
      
      if (lltOfA.info() == Eigen::Success) {
        Eigen::VectorXd mean = lltOfA.solve(Xt_y / sigma2);
        Eigen::VectorXd noise(P_combined);
        for(int k=0; k<P_combined; ++k) noise(k) = Rf_rnorm(0.0, 1.0);
        combined_coeffs_new = mean + lltOfA.matrixU().solve(noise);
        update_success = true;
      } else {
        cpp11::warning("Cholesky failed. Keeping old values.");
      }
    }
    
    // Unpack only if successful
    if (update_success) {
      if(regularize_ATE) alpha = combined_coeffs_new(0);
      for(int j=0; j<p_mod; ++j) beta[j] = combined_coeffs_new(offset_beta + j);
      for(int k=0; k<p_int; ++k) beta_int[k] = combined_coeffs_new(offset_beta_int + k);
      if (use_prognostic_shapley) {
        for(int l=0; l<p_prog; ++l) gamma[l] = combined_coeffs_new(offset_gamma + l);
      } 
    }
    
    // Update Residual
    Eigen::VectorXd new_fit = Eigen::VectorXd::Zero(n);
    if(regularize_ATE) new_fit.array() += Z_map.array() * alpha;
    for (int j=0; j<p_mod; ++j) new_fit.array() += Z_map.array() * X_map.col(j).array() * beta[j];
    for (size_t k=0; k<int_pairs.size(); ++k) new_fit.array() += Z_map.array() * X_map.col(int_pairs[k].first).array() * X_map.col(int_pairs[k].second).array() * beta_int[k];
    if (use_prognostic_shapley) {                                            
      for (int l=0; l<p_prog; ++l) new_fit.array() += Phi_map.col(l).array() * gamma[l];
    }
    residual_map = y_target - new_fit;
    
    // Shrinkage Updates
    for(int j = 0; j < p_mod + regularize_ATE; j++){
      double current_coeff = regularize_ATE ? ((j==0) ? alpha : beta[j-1]) : beta[j];
      nu[j] = rinvgamma_linear(1.0, 1.0 + 1.0 / safe_var_linear(tau_beta[j]*tau_beta[j]));
      tau_beta[j] = std::sqrt(safe_var_linear(rinvgamma_linear(1.0, (1.0 / safe_var_linear(nu[j])) + (current_coeff * current_coeff) / safe_var_linear(2.0 * tau_glob * tau_glob * sigma2))));
    } 
    if(unlink){
      for(size_t k=0; k<int_pairs.size(); k++) {
        int idx = offset_beta_int + k;
        nu[idx] = rinvgamma_linear(1.0, 1.0 + 1.0/safe_var_linear(tau_beta[idx]*tau_beta[idx]));
        tau_beta[idx] = std::sqrt(safe_var_linear(rinvgamma_linear(1.0, (1.0/safe_var_linear(nu[idx])) + (beta_int[k]*beta_int[k])/safe_var_linear(2.0*tau_glob*tau_glob*sigma2))));
      }
    }
    if (use_prognostic_shapley) {
      for(int l = 0; l < p_prog; l++){
        nu_gamma[l] = rinvgamma_linear(1.0, 1.0 + 1.0 / safe_var_linear(tau_gamma[l]*tau_gamma[l]));
        tau_gamma[l] = std::sqrt(safe_var_linear(rinvgamma_linear(1.0, (1.0 / safe_var_linear(nu_gamma[l])) + (gamma[l] * gamma[l]) / safe_var_linear(2.0 * tau_glob * tau_glob * sigma2))));
      }
    }
    
    // Global Tau
    if (sample_global_prior == "half-cauchy") {
      xi = rinvgamma_linear(1.0, 1.0 + 1.0 / safe_var_linear(tau_glob*tau_glob));
      double sum_scaled = 0.0;
      if(regularize_ATE) sum_scaled += (alpha*alpha) / safe_var_linear(tau_beta[0]*tau_beta[0]);
      for(int j=0; j<p_mod; j++) sum_scaled += (beta[j]*beta[j]) / safe_var_linear(tau_beta[j+offset_beta]*tau_beta[j+offset_beta]);
      
      if(unlink) {
        for(size_t k=0; k<int_pairs.size(); k++) sum_scaled += (beta_int[k]*beta_int[k]) / safe_var_linear(tau_beta[offset_beta_int+k]*tau_beta[offset_beta_int+k]);
      } else {
        for(size_t k=0; k<int_pairs.size(); k++) {
          double var_k = tau_int * tau_beta[offset_beta + int_pairs[k].first] * tau_beta[offset_beta + int_pairs[k].second];
          sum_scaled += (beta_int[k]*beta_int[k]) / safe_var_linear(var_k);
        }
      } 
      if (use_prognostic_shapley) {
        for(int l=0; l<p_prog; l++) sum_scaled += (gamma[l]*gamma[l]) / safe_var_linear(tau_gamma[l]*tau_gamma[l]);
      }
      double shape_glob = (static_cast<double>(P_combined) + 1.0) / 2.0;
      double rate_glob = (1.0 / safe_var_linear(xi)) + (1.0 / safe_var_linear(2.0 * sigma2)) * sum_scaled;
      if(rate_glob < 1e-12) rate_glob = 1e-12;
      tau_glob = std::sqrt(safe_var_linear(rinvgamma_linear(shape_glob, rate_glob)));
    }
  } else {
    cpp11::stop("Only Gibbs sampler supported for augmented prognostic model.");
  }
  
  // Output Packing
  size_t total_size = 5 + beta.size() + beta_int.size() + gamma.size() + 
    tau_beta.size() + tau_gamma.size() + 
    nu.size() + nu_gamma.size();
  
  writable::doubles params_out;
  params_out.reserve(total_size);
  params_out.push_back(alpha);
  params_out.push_back(tau_int);
  params_out.push_back(tau_glob);
  params_out.push_back(gamma_prop);
  params_out.push_back(xi);
  
  for (double val : beta) params_out.push_back(val);
  for (double val : beta_int) params_out.push_back(val);
  for (double val : gamma) params_out.push_back(val);
  for (double val : tau_beta) params_out.push_back(val);
  for (double val : tau_gamma) params_out.push_back(val); 
  for (double val : nu) params_out.push_back(val);
  for (double val : nu_gamma) params_out.push_back(val); 
  
  writable::doubles residuals_out;
  residuals_out.reserve(n);
  for (int i = 0; i < n; ++i) residuals_out.push_back(residual_map[i]);
  
  if (save_output) {
    std::ofstream outfile("output_log.txt", std::ios::app);
    save_output_vector_labeled(outfile, alpha, tau_int, tau_glob, xi, sigma, beta, beta_int, tau_beta, nu, index);
    outfile.close();
  }
  
  PutRNGstate();
  return writable::list({
    "params"_nm = params_out,
      "residuals"_nm = residuals_out
  });
}

// -----------------------------------------------------------------------------
// SECTION 4: NCP LINEAR UPDATE
// -----------------------------------------------------------------------------

[[cpp11::register]]
writable::doubles updateLinearTreatmentCpp_NCP_cpp(
    const doubles_matrix<>& X,
    const doubles_matrix<>& Phi,
    const doubles& Z,
    const doubles& propensity_train,
    writable::doubles residual,
    const integers& are_continuous,
    double alpha_tilde,
    double gamma_prop, 
    writable::doubles beta_tilde,
    writable::doubles beta_int_tilde,
    writable::doubles gamma_tilde,
    writable::doubles tau_beta,
    writable::doubles tau_gamma,
    writable::doubles nu,
    writable::doubles nu_gamma,
    double xi,
    double tau_int,
    double sigma,
    double alpha_prior_sd,
    double tau_glob,
    const std::string& sample_global_prior,
    bool unlink,
    bool gibbs,
    bool save_output,
    int index,
    int max_steps,
    double step_out,
    const std::string& propensity_seperate,
    bool regularize_ATE,
    double hn_scale,
    bool use_prognostic_shapley) {
  
  GetRNGstate();
  int n = residual.size();
  int p_mod = X.ncol();
  int p_prog = use_prognostic_shapley ? Phi.ncol() : 0; 
  int p_int = beta_int_tilde.size();
  double sigma2 = sigma * sigma;
  
  std::vector<std::pair<int, int>> int_pairs;
  if (p_int > 0) {
    int_pairs.reserve(p_int);
    for (int i = 0; i < p_mod; i++) {
      for (int j = i + 1; j < p_mod; j++) {
        if(!((propensity_seperate == "tau") & ((i == (p_mod -1))| (j == (p_mod-1))))){
          if ((are_continuous[i] == 1) || (are_continuous[j] == 1)) {
            int_pairs.push_back(std::make_pair(i, j));
          }
        }
      }
    }
  }
  
  Eigen::Map<Eigen::VectorXd> residual_map((double*)residual.data(), n);
  Eigen::Map<const Eigen::VectorXd> Z_map((const double*)Z.data(), n);
  Eigen::Map<const Eigen::MatrixXd> X_map((const double*)X.data(), n, p_mod);
  Eigen::Map<const Eigen::MatrixXd> Phi_map((const double*)Phi.data(), n, (p_prog > 0 ? p_prog : 1));
  
  Eigen::Map<Eigen::VectorXd> beta_tilde_map((double*)beta_tilde.data(), p_mod);
  Eigen::Map<Eigen::VectorXd> beta_int_tilde_map((double*)beta_int_tilde.data(), p_int);
  Eigen::Map<Eigen::VectorXd> gamma_tilde_map((double*)gamma_tilde.data(), (p_prog > 0 ? p_prog : 1));
  
  // --- Calculate Real Betas ---
  Eigen::VectorXd beta_current(p_mod);
  Eigen::VectorXd beta_int_current(p_int);
  Eigen::VectorXd gamma_current(p_prog > 0 ? p_prog : 1);
  double alpha_current = alpha_tilde;
  
  int offset_alpha = regularize_ATE ? 1 : 0;
  int offset_beta = offset_alpha;
  int offset_beta_int = offset_beta + p_mod;
  int offset_gamma = offset_beta_int + p_int;
  
  if(regularize_ATE) alpha_current = alpha_tilde * tau_beta[0] * tau_glob;
  for (int j = 0; j < p_mod; ++j) beta_current(j) = beta_tilde_map(j) * tau_beta[j + offset_beta] * tau_glob;
  for (size_t k = 0; k < int_pairs.size(); ++k) {
    double V_k = unlink ? tau_beta[offset_beta_int + k] : (std::sqrt(tau_int) * tau_beta[offset_beta + int_pairs[k].first] * tau_beta[offset_beta + int_pairs[k].second]);
    beta_int_current(k) = beta_int_tilde_map(k) * V_k * tau_glob;
  }
  
  if (use_prognostic_shapley) {
    for(int l = 0; l < p_prog; ++l) gamma_current(l) = gamma_tilde_map(l) * tau_gamma[l] * tau_glob;
  }
  
  // --- Propensity ---
  if (propensity_seperate == "mu") {
    Eigen::Map<const Eigen::VectorXd> prop_map((const double*)propensity_train.data(), n);
    residual_map += prop_map * gamma_prop;
    gamma_prop = sample_alpha_linear(residual, propensity_train, sigma, 10.0);
    residual_map -= prop_map * gamma_prop;
  }
  if (!regularize_ATE){
    residual_map += Z_map * alpha_current;
    alpha_tilde = sample_alpha_linear(residual, Z, sigma, alpha_prior_sd);
    residual_map -= Z_map * alpha_tilde;
    alpha_current = alpha_tilde;
  }
  
  // --- Joint Gibbs ---
  if (gibbs) {
    int P_combined = offset_gamma + p_prog;
    
    // Target y*
    Eigen::VectorXd y_target = residual_map;
    if(regularize_ATE) y_target.array() += Z_map.array() * alpha_current;
    for (int j = 0; j < p_mod; ++j) y_target.array() += Z_map.array() * X_map.col(j).array() * beta_current(j);
    for (size_t k = 0; k < int_pairs.size(); ++k) y_target.array() += Z_map.array() * X_map.col(int_pairs[k].first).array() * X_map.col(int_pairs[k].second).array() * beta_int_current(k);
    
    if (use_prognostic_shapley) {
      for (int l = 0; l < p_prog; ++l) y_target.array() += Phi_map.col(l).array() * gamma_current(l);
    }
    
    // X_star (NCP Design Matrix)
    Eigen::MatrixXd X_star(n, P_combined);
    if(regularize_ATE) X_star.col(0) = Z_map.array() * tau_beta[0] * tau_glob;
    for (int j = 0; j < p_mod; ++j) X_star.col(offset_beta + j) = Z_map.array() * X_map.col(j).array() * tau_beta[j + offset_beta] * tau_glob;
    for (size_t k = 0; k < int_pairs.size(); ++k) {
      double V_k = unlink ? tau_beta[offset_beta_int + k] : (std::sqrt(tau_int) * tau_beta[offset_beta + int_pairs[k].first] * tau_beta[offset_beta + int_pairs[k].second]);
      X_star.col(offset_beta_int + k) = Z_map.array() * X_map.col(int_pairs[k].first).array() * X_map.col(int_pairs[k].second).array() * V_k * tau_glob;
    }
    
    if (use_prognostic_shapley) {
      for (int l = 0; l < p_prog; ++l) X_star.col(offset_gamma + l) = Phi_map.col(l).array() * tau_gamma[l] * tau_glob;
    }
    
    Eigen::MatrixXd XtX = X_star.transpose() * X_star;
    Eigen::VectorXd Xty = X_star.transpose() * y_target;
    
    // --- FIX 1: Explicit Diagonal Update ---
    Eigen::MatrixXd Prec = XtX / sigma2;
    Prec.diagonal().array() += 1.0;
    
    Eigen::LLT<Eigen::MatrixXd> llt(Prec);
    Eigen::VectorXd combined_tilde_new(P_combined);
    
    if(llt.info() == Eigen::Success) {
      Eigen::VectorXd mean = llt.solve(Xty / sigma2);
      Eigen::VectorXd z(P_combined);
      for(int i=0; i<P_combined; ++i) z(i) = Rf_rnorm(0.0, 1.0);
      
      // --- FIX 2: Correct Upper Triangular Solve ---
      combined_tilde_new = mean + llt.matrixU().solve(z);
    }
    
    // Unpack
    if(regularize_ATE) alpha_tilde = combined_tilde_new(0);
    for(int j=0; j<p_mod; ++j) beta_tilde[j] = combined_tilde_new(offset_beta + j);
    for(int k=0; k<p_int; ++k) beta_int_tilde[k] = combined_tilde_new(offset_beta_int + k);
    
    if (use_prognostic_shapley) {
      for(int l=0; l<p_prog; ++l) gamma_tilde[l] = combined_tilde_new(offset_gamma + l);
    }
    
    // Update Real
    if(regularize_ATE) alpha_current = alpha_tilde * tau_beta[0] * tau_glob;
    for (int j = 0; j < p_mod; ++j) beta_current(j) = beta_tilde_map(j) * tau_beta[j + offset_beta] * tau_glob;
    for (size_t k = 0; k < int_pairs.size(); ++k) {
      double V_k = unlink ? tau_beta[offset_beta_int + k] : (std::sqrt(tau_int) * tau_beta[offset_beta + int_pairs[k].first] * tau_beta[offset_beta + int_pairs[k].second]);
      beta_int_current(k) = beta_int_tilde_map(k) * V_k * tau_glob;
    }
    if (use_prognostic_shapley) {
      for (int l = 0; l < p_prog; ++l) gamma_current(l) = gamma_tilde_map(l) * tau_gamma[l] * tau_glob;
    }
    
    // Residual Update
    Eigen::VectorXd new_fit = Eigen::VectorXd::Zero(n);
    if(regularize_ATE) new_fit.array() += Z_map.array() * alpha_current;
    for (int j = 0; j < p_mod; ++j) new_fit.array() += Z_map.array() * X_map.col(j).array() * beta_current(j);
    for (size_t k = 0; k < int_pairs.size(); ++k) new_fit.array() += Z_map.array() * X_map.col(int_pairs[k].first).array() * X_map.col(int_pairs[k].second).array() * beta_int_current(k);
    // if (use_prognostic_shapley) {
    //   for (int l = 0; l < p_prog; ++l) new_fit.array() += Phi_map.col(l).array() * gamma_current(l);
    // }
    residual_map = y_target - new_fit;
    
    // Shrinkage
    for(int j=0; j<p_mod+regularize_ATE; ++j) {
      double coef = regularize_ATE ? ((j==0)?alpha_tilde:beta_tilde[j-1]) : beta_tilde[j];
      nu[j] = rinvgamma_linear(1.0, 1.0 + 1.0/safe_var_linear(tau_beta[j]*tau_beta[j]));
      tau_beta[j] = std::sqrt(safe_var_linear(rinvgamma_linear(1.0, (1.0/safe_var_linear(nu[j])) + (coef*coef)/safe_var_linear(2.0*tau_glob*tau_glob))));
    }
    if(unlink){
      for(size_t k=0; k<int_pairs.size(); k++){
        int full_idx = offset_beta_int + k;
        nu[full_idx] = rinvgamma_linear(1.0, 1.0 + 1.0 / safe_var_linear(tau_beta[full_idx]*tau_beta[full_idx]));
        tau_beta[full_idx] = std::sqrt(safe_var_linear(rinvgamma_linear(1.0, (1.0 / safe_var_linear(nu[full_idx])) + (beta_int_tilde[k] * beta_int_tilde[k]) / safe_var_linear(2.0 * tau_glob * tau_glob))));
      }
    }
    
    if (use_prognostic_shapley) {
      for(int l=0; l<p_prog; ++l) {
        nu_gamma[l] = rinvgamma_linear(1.0, 1.0 + 1.0/safe_var_linear(tau_gamma[l]*tau_gamma[l]));
        tau_gamma[l] = std::sqrt(safe_var_linear(rinvgamma_linear(1.0, (1.0/safe_var_linear(nu_gamma[l])) + (gamma_tilde[l]*gamma_tilde[l])/safe_var_linear(2.0*tau_glob*tau_glob))));
      }
    }
    
    if (sample_global_prior == "half-cauchy") {
      xi = rinvgamma_linear(1.0, 1.0 + 1.0 / safe_var_linear(tau_glob*tau_glob));
      double sum_scaled = 0.0;
      
      if(regularize_ATE) sum_scaled += (alpha_tilde*alpha_tilde) / safe_var_linear(tau_beta[0]*tau_beta[0]);
      for(int j=0; j<p_mod; ++j) sum_scaled += (beta_tilde[j]*beta_tilde[j]) / safe_var_linear(tau_beta[j+offset_beta]*tau_beta[j+offset_beta]);
      
      if(unlink){
        for(int k = 0; k < p_int; k++) {
          sum_scaled += (beta_int_tilde[k] * beta_int_tilde[k]) / safe_var_linear(tau_beta[offset_beta_int + k] * tau_beta[offset_beta_int + k]);
        } 
      } else {
        for(size_t k = 0; k < int_pairs.size(); ++k) {
          double var_k = tau_int * tau_beta[offset_beta + int_pairs[k].first] * tau_beta[offset_beta + int_pairs[k].second];
          sum_scaled += (beta_int_tilde[k] * beta_int_tilde[k]) / safe_var_linear(var_k);
        }
      } 
      
      if (use_prognostic_shapley) {
        for(int l=0; l<p_prog; ++l) sum_scaled += (gamma_tilde[l]*gamma_tilde[l]) / safe_var_linear(tau_gamma[l]*tau_gamma[l]);
      }
      
      double shape = (static_cast<double>(P_combined) + 1.0) / 2.0;
      double rate = (1.0/safe_var_linear(xi)) + (1.0/2.0) * sum_scaled;
      tau_glob = std::sqrt(safe_var_linear(rinvgamma_linear(shape, rate)));
    }
  } else {
    cpp11::stop("NCP is only implemented for the `gibbs = true` path.");
  }
  
  // Return
  size_t total_size = 5 + beta_tilde.size() + beta_int_tilde.size() + gamma_tilde.size() + 
    tau_beta.size() + tau_gamma.size() + 
    nu.size() + nu_gamma.size() + residual.size();
  
  writable::doubles output;
  output.reserve(total_size);
  output.push_back(alpha_tilde);
  output.push_back(tau_int);
  output.push_back(tau_glob);
  output.push_back(gamma_prop);
  output.push_back(xi);
  PutRNGstate();
  for (double val : beta_tilde) output.push_back(val);
  for (double val : beta_int_tilde) output.push_back(val);
  for (double val : gamma_tilde) output.push_back(val);
  for (double val : tau_beta) output.push_back(val);
  for (double val : tau_gamma) output.push_back(val);
  for (double val : nu) output.push_back(val);
  for (double val : nu_gamma) output.push_back(val);
  for (int i = 0; i < n; ++i) output.push_back(residual_map[i]);
  
  return output;
}

// -----------------------------------------------------------------------------
// SECTION: LTR CAUSAL SCORECARD SAMPLERS
// -----------------------------------------------------------------------------

// Helper for PG(1, z) exact sample using truncated Gamma sum
// Optimized: truncates at k=20 and adds the exact expectation of the remaining terms
double sample_pg1_cpp(double z) {
    double z_sq_half = (z * z) / 2.0;
    double sum = 0.0;
    constexpr double PI2_2 = 2.0 * M_PI * M_PI;
    
    // Sum first 20 terms
    for (int k = 1; k <= 20; ++k) {
        double e = Rf_rexp(1.0);
        double k_half = k - 0.5;
        double d = PI2_2 * k_half * k_half + z_sq_half;
        sum += e / d;
    }
    
    // Add expected value of the tail
    double x = 20.5;
    if (z_sq_half < 1e-8) {
        sum += 1.0 / (PI2_2 * x);
    } else {
        double a = std::sqrt(PI2_2);
        double b = std::sqrt(z_sq_half);
        sum += (1.0 / (a * b)) * (M_PI / 2.0 - std::atan(a * x / b));
    }
    
    return sum;
}

[[cpp11::register]]
writable::list run_ltr_mse_cpp(
    const doubles& tau_tilde,
    const doubles_matrix<>& X,
    const integers& are_continuous,
    int M,
    double epsilon,
    const doubles& beta_init,
    const doubles& beta_int_init,
    const doubles& tau_beta_init,
    const doubles& nu_init,
    double tau_int_init,
    double tau_glob_init,
    double xi_init,
    double sigma2_rank_init,
    bool unlink,
    int n_iter,
    int burn_in
) {
    GetRNGstate();
    int N = tau_tilde.size();
    int P_main = X.ncol();
    
    std::vector<std::pair<int, int>> int_pairs;
    for (int i = 0; i < P_main; i++) {
        for (int j = i + 1; j < P_main; j++) {
            if ((are_continuous[i] == 1) || (are_continuous[j] == 1)) {
                int_pairs.push_back({i, j});
            }
        }
    }
    int P_int = int_pairs.size();
    int P_combined = P_main + P_int;
    
    // Map data
    Eigen::Map<const Eigen::VectorXd> tau_map((const double*)tau_tilde.data(), N);
    Eigen::Map<const Eigen::MatrixXd> X_map((const double*)X.data(), N, P_main);
    
    // State variables
    Eigen::VectorXd beta(P_combined);
    for(int j=0; j<P_main; ++j) beta(j) = beta_init[j];
    for(int k=0; k<P_int; ++k) beta(P_main + k) = beta_int_init[k];
    
    std::vector<double> tau_beta(P_combined);
    std::vector<double> nu(P_combined);
    for(int j=0; j<P_combined; ++j) {
        tau_beta[j] = tau_beta_init[j];
        nu[j] = nu_init[j];
    }
    double tau_int = tau_int_init;
    double tau_glob = tau_glob_init;
    double xi = xi_init;
    double sigma2_rank = sigma2_rank_init;
    
    // Output containers
    int n_save = n_iter;
    writable::doubles_matrix<> out_beta(n_save, P_combined);
    writable::doubles_matrix<> out_tau_beta(n_save, P_combined);
    writable::doubles out_sigma2(n_save);
    writable::doubles out_tau_glob(n_save);
    
    // Find valid pairs
    std::vector<std::pair<int, int>> valid_pairs;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (tau_tilde[i] > tau_tilde[j] + epsilon) {
                valid_pairs.push_back({i, j});
            }
        }
    }
    if (valid_pairs.empty()) {
        PutRNGstate();
        cpp11::stop("No valid pairs found with the given epsilon.");
    }
    
    int total_iter = burn_in + n_iter;
    for (int iter = 0; iter < total_iter; ++iter) {
        // 1. Construct mini-batch
        int actual_M = std::min(M, (int)valid_pairs.size());
        Eigen::MatrixXd X_diff(actual_M, P_combined);
        Eigen::VectorXd Omega(actual_M);
        
        for (int m = 0; m < actual_M; ++m) {
            int idx = floor(Rf_runif(0.0, valid_pairs.size()));
            if(idx == valid_pairs.size()) idx = valid_pairs.size() - 1;
            int i = valid_pairs[idx].first;
            int j = valid_pairs[idx].second;
            Omega(m) = tau_map(i) - tau_map(j);
            for(int k=0; k<P_main; ++k) X_diff(m, k) = X_map(i, k) - X_map(j, k);
            for(int k=0; k<P_int; ++k) {
                int idx1 = int_pairs[k].first;
                int idx2 = int_pairs[k].second;
                X_diff(m, P_main + k) = X_map(i, idx1)*X_map(i, idx2) - X_map(j, idx1)*X_map(j, idx2);
            }
        }
        
        // 2. Update Beta
        Eigen::MatrixXd XtX = X_diff.transpose() * X_diff;
        Eigen::VectorXd XtY = X_diff.transpose() * Omega;
        
        Eigen::MatrixXd Prec = XtX / sigma2_rank;
        for (int k = 0; k < P_main; ++k) {
            Prec(k, k) += 1.0 / safe_var_linear(tau_beta[k] * tau_beta[k] * tau_glob * tau_glob);
        }
        for (int k = 0; k < P_int; ++k) {
            double var_k = unlink ? (tau_beta[P_main + k] * tau_beta[P_main + k]) : (tau_int * tau_beta[int_pairs[k].first] * tau_beta[int_pairs[k].second]);
            Prec(P_main + k, P_main + k) += 1.0 / safe_var_linear(var_k * tau_glob * tau_glob);
        }
        
        Eigen::LLT<Eigen::MatrixXd> llt(Prec);
        if (llt.info() != Eigen::Success) {
            Prec.diagonal().array() += 1e-6;
            llt.compute(Prec);
        }
        
        if (llt.info() == Eigen::Success) {
            Eigen::VectorXd mean = llt.solve(XtY / sigma2_rank);
            Eigen::VectorXd z(P_combined);
            for(int j=0; j<P_combined; ++j) z(j) = Rf_rnorm(0.0, 1.0);
            beta = mean + llt.matrixU().solve(z);
        }
        
        // 3. Update sigma2_rank
        Eigen::VectorXd Y_hat = X_diff * beta;
        double SSE = (Omega - Y_hat).squaredNorm();
        double a_new = 0.001 + actual_M / 2.0;
        double b_new = 0.001 + SSE / 2.0;
        sigma2_rank = rinvgamma_linear(a_new, b_new);
        
        // 4. Update Horseshoe
        for(int k=0; k<P_main; ++k) {
            nu[k] = rinvgamma_linear(1.0, 1.0 + 1.0 / safe_var_linear(tau_beta[k]*tau_beta[k]));
            tau_beta[k] = std::sqrt(safe_var_linear(rinvgamma_linear(1.0, 1.0 / safe_var_linear(nu[k]) + (beta(k)*beta(k)) / safe_var_linear(2.0 * tau_glob * tau_glob * sigma2_rank))));
        }
        if (unlink) {
             for(int k=0; k<P_int; ++k) {
                  int idx = P_main + k;
                  nu[idx] = rinvgamma_linear(1.0, 1.0 + 1.0 / safe_var_linear(tau_beta[idx]*tau_beta[idx]));
                  tau_beta[idx] = std::sqrt(safe_var_linear(rinvgamma_linear(1.0, 1.0 / safe_var_linear(nu[idx]) + (beta(idx)*beta(idx)) / safe_var_linear(2.0 * tau_glob * tau_glob * sigma2_rank))));
             }
        }
        
        xi = rinvgamma_linear(1.0, 1.0 + 1.0 / safe_var_linear(tau_glob*tau_glob));
        double sum_scaled = 0.0;
        for(int k=0; k<P_main; ++k) sum_scaled += (beta(k)*beta(k)) / safe_var_linear(tau_beta[k]*tau_beta[k]);
        if (unlink) {
             for(int k=0; k<P_int; ++k) sum_scaled += (beta(P_main+k)*beta(P_main+k)) / safe_var_linear(tau_beta[P_main+k]*tau_beta[P_main+k]);
        } else {
             for(int k=0; k<P_int; ++k) {
                  double var_k = tau_int * tau_beta[int_pairs[k].first] * tau_beta[int_pairs[k].second];
                  sum_scaled += (beta(P_main+k)*beta(P_main+k)) / safe_var_linear(var_k);
             }
        }
        double shape_glob = (P_combined + 1.0) / 2.0;
        double rate_glob = 1.0 / safe_var_linear(xi) + sum_scaled / safe_var_linear(2.0 * sigma2_rank);
        tau_glob = std::sqrt(safe_var_linear(rinvgamma_linear(shape_glob, rate_glob)));
        
        // Save
        if (iter >= burn_in) {
            int save_idx = iter - burn_in;
            for(int j=0; j<P_combined; ++j) {
                out_beta(save_idx, j) = beta(j);
                out_tau_beta(save_idx, j) = tau_beta[j];
            }
            out_sigma2[save_idx] = sigma2_rank;
            out_tau_glob[save_idx] = tau_glob;
        }
    }
    PutRNGstate();
    
    return writable::list({
        "beta"_nm = out_beta,
        "tau_beta"_nm = out_tau_beta,
        "sigma2_rank"_nm = out_sigma2,
        "tau_glob"_nm = out_tau_glob
    });
}

[[cpp11::register]]
writable::list run_ltr_pg_cpp(
    const doubles& tau_tilde,
    const doubles_matrix<>& X,
    const integers& are_continuous,
    int M,
    double epsilon,
    const doubles& beta_init,
    const doubles& beta_int_init,
    const doubles& tau_beta_init,
    const doubles& nu_init,
    double tau_int_init,
    double tau_glob_init,
    double xi_init,
    bool unlink,
    int n_iter,
    int burn_in
) {
    GetRNGstate();
    int N = tau_tilde.size();
    int P_main = X.ncol();
    
    std::vector<std::pair<int, int>> int_pairs;
    for (int i = 0; i < P_main; i++) {
        for (int j = i + 1; j < P_main; j++) {
            if ((are_continuous[i] == 1) || (are_continuous[j] == 1)) {
                int_pairs.push_back({i, j});
            }
        }
    }
    int P_int = int_pairs.size();
    int P_combined = P_main + P_int;
    
    Eigen::Map<const Eigen::VectorXd> tau_map((const double*)tau_tilde.data(), N);
    Eigen::Map<const Eigen::MatrixXd> X_map((const double*)X.data(), N, P_main);
    
    Eigen::VectorXd beta(P_combined);
    for(int j=0; j<P_main; ++j) beta(j) = beta_init[j];
    for(int k=0; k<P_int; ++k) beta(P_main + k) = beta_int_init[k];
    
    std::vector<double> tau_beta(P_combined);
    std::vector<double> nu(P_combined);
    for(int j=0; j<P_combined; ++j) {
        tau_beta[j] = tau_beta_init[j];
        nu[j] = nu_init[j];
    }
    double tau_int = tau_int_init;
    double tau_glob = tau_glob_init;
    double xi = xi_init;
    
    int n_save = n_iter;
    writable::doubles_matrix<> out_beta(n_save, P_combined);
    writable::doubles_matrix<> out_tau_beta(n_save, P_combined);
    writable::doubles out_tau_glob(n_save);
    
    struct Patient_sort_helper {
        double tau;
        int id;
        bool operator<(const Patient_sort_helper& other) const {
            return tau < other.tau;
        }
    };
    
    std::vector<Patient_sort_helper> sorted_tau(N);
    for (int i = 0; i < N; ++i) {
        sorted_tau[i] = {tau_map(i), i};
    }
    std::sort(sorted_tau.begin(), sorted_tau.end());
    
    std::vector<double> S(N + 1, 0.0);
    for (int i = 0; i < N; ++i) {
        S[i + 1] = S[i] + sorted_tau[i].tau;
    }
    
    std::vector<int> k_idx(N, 0);
    std::vector<double> W(N, 0.0);
    std::vector<double> CW(N, 0.0);
    long long total_valid_pairs = 0;
    double total_weight = 0.0;
    
    for (int i = 0; i < N; ++i) {
        double target = sorted_tau[i].tau - epsilon;
        auto it = std::lower_bound(sorted_tau.begin(), sorted_tau.begin() + i, target, 
                                   [](const Patient_sort_helper& p, double val) { return p.tau < val; });
        int k = std::distance(sorted_tau.begin(), it);
        k_idx[i] = k;
        total_valid_pairs += k;
        
        if (k > 0) {
            W[i] = k * sorted_tau[i].tau - S[k];
        } else {
            W[i] = 0.0;
        }
        total_weight += W[i];
        CW[i] = total_weight;
    }
    
    if (total_valid_pairs == 0) {
        PutRNGstate();
        cpp11::stop("No valid pairs found with the given epsilon.");
    }
    
    int total_iter = burn_in + n_iter;
    for (int iter = 0; iter < total_iter; ++iter) {
        int actual_M = std::min((long long)M, total_valid_pairs);
        Eigen::MatrixXd Xt_Gamma_X = Eigen::MatrixXd::Zero(P_combined, P_combined);
        Eigen::VectorXd XtY = Eigen::VectorXd::Zero(P_combined);
        
        // Scale factor for unbiased estimation of full dataset likelihood
        double scale = total_weight / actual_M;
        
        for (int m = 0; m < actual_M; ++m) {
            // Importance sampling proportional to tau difference
            double u = Rf_runif(0.0, total_weight);
            auto it = std::lower_bound(CW.begin(), CW.end(), u);
            int idx_i = std::distance(CW.begin(), it);
            if (idx_i >= N) idx_i = N - 1;
            
            // Re-draw if we somehow hit an index with 0 valid pairs (precision boundary issues)
            while (k_idx[idx_i] == 0 && idx_i < N - 1) {
                idx_i++;
            }
            if (k_idx[idx_i] == 0) continue;
            
            double u_j = Rf_runif(0.0, W[idx_i]);
            int low = 1, high = k_idx[idx_i];
            int chosen_j = -1;
            while (low <= high) {
                int mid = low + (high - low) / 2;
                double cw_mid = mid * sorted_tau[idx_i].tau - S[mid];
                if (cw_mid < u_j) {
                    low = mid + 1;
                } else {
                    chosen_j = mid - 1;
                    high = mid - 1;
                }
            }
            if (chosen_j == -1) chosen_j = k_idx[idx_i] - 1;
            
            int i = sorted_tau[idx_i].id;
            int j = sorted_tau[chosen_j].id;
            
            Eigen::VectorXd x_row(P_combined);
            for(int k=0; k<P_main; ++k) x_row(k) = X_map(i, k) - X_map(j, k);
            for(int k=0; k<P_int; ++k) {
                int idx1 = int_pairs[k].first;
                int idx2 = int_pairs[k].second;
                x_row(P_main + k) = X_map(i, idx1)*X_map(i, idx2) - X_map(j, idx1)*X_map(j, idx2);
            }
            
            double psi_k = x_row.dot(beta);
            double pg_draw = sample_pg1_cpp(psi_k);
            
            // Notice that w_k nicely cancels out because of the importance sampling probability!
            double gamma_m = scale * pg_draw;
            
            Xt_Gamma_X.selfadjointView<Eigen::Lower>().rankUpdate(x_row, gamma_m);
            XtY += x_row * (scale / 2.0);
        }
        Xt_Gamma_X = Xt_Gamma_X.selfadjointView<Eigen::Lower>();
        
        Eigen::MatrixXd Prec = Xt_Gamma_X;
        for (int k = 0; k < P_main; ++k) {
            Prec(k, k) += 1.0 / safe_var_linear(tau_beta[k] * tau_beta[k] * tau_glob * tau_glob);
        }
        for (int k = 0; k < P_int; ++k) {
            double var_k = unlink ? (tau_beta[P_main + k] * tau_beta[P_main + k]) : (tau_int * tau_beta[int_pairs[k].first] * tau_beta[int_pairs[k].second]);
            Prec(P_main + k, P_main + k) += 1.0 / safe_var_linear(var_k * tau_glob * tau_glob);
        }
        
        Eigen::LLT<Eigen::MatrixXd> llt(Prec);
        if (llt.info() != Eigen::Success) {
            Prec.diagonal().array() += 1e-6;
            llt.compute(Prec);
        }
        
        if (llt.info() == Eigen::Success) {
            Eigen::VectorXd mean = llt.solve(XtY);
            Eigen::VectorXd z(P_combined);
            for(int j=0; j<P_combined; ++j) z(j) = Rf_rnorm(0.0, 1.0);
            beta = mean + llt.matrixU().solve(z);
        }
        
        for(int k=0; k<P_main; ++k) {
            nu[k] = rinvgamma_linear(1.0, 1.0 + 1.0 / safe_var_linear(tau_beta[k]*tau_beta[k]));
            tau_beta[k] = std::sqrt(safe_var_linear(rinvgamma_linear(1.0, 1.0 / safe_var_linear(nu[k]) + (beta(k)*beta(k)) / safe_var_linear(2.0 * tau_glob * tau_glob))));
        }
        if (unlink) {
             for(int k=0; k<P_int; ++k) {
                  int idx = P_main + k;
                  nu[idx] = rinvgamma_linear(1.0, 1.0 + 1.0 / safe_var_linear(tau_beta[idx]*tau_beta[idx]));
                  tau_beta[idx] = std::sqrt(safe_var_linear(rinvgamma_linear(1.0, 1.0 / safe_var_linear(nu[idx]) + (beta(idx)*beta(idx)) / safe_var_linear(2.0 * tau_glob * tau_glob))));
             }
        }
        
        xi = rinvgamma_linear(1.0, 1.0 + 1.0 / safe_var_linear(tau_glob*tau_glob));
        double sum_scaled = 0.0;
        for(int k=0; k<P_main; ++k) sum_scaled += (beta(k)*beta(k)) / safe_var_linear(tau_beta[k]*tau_beta[k]);
        if (unlink) {
             for(int k=0; k<P_int; ++k) sum_scaled += (beta(P_main+k)*beta(P_main+k)) / safe_var_linear(tau_beta[P_main+k]*tau_beta[P_main+k]);
        } else {
             for(int k=0; k<P_int; ++k) {
                  double var_k = tau_int * tau_beta[int_pairs[k].first] * tau_beta[int_pairs[k].second];
                  sum_scaled += (beta(P_main+k)*beta(P_main+k)) / safe_var_linear(var_k);
             }
        }
        double shape_glob = (P_combined + 1.0) / 2.0;
        double rate_glob = 1.0 / safe_var_linear(xi) + sum_scaled / 2.0;
        tau_glob = std::sqrt(safe_var_linear(rinvgamma_linear(shape_glob, rate_glob)));
        
        if (iter >= burn_in) {
            int save_idx = iter - burn_in;
            for(int j=0; j<P_combined; ++j) {
                out_beta(save_idx, j) = beta(j);
                out_tau_beta(save_idx, j) = tau_beta[j];
            }
            out_tau_glob[save_idx] = tau_glob;
        }
    }
    PutRNGstate();
    
    return writable::list({
        "beta"_nm = out_beta,
        "tau_beta"_nm = out_tau_beta,
        "tau_glob"_nm = out_tau_glob
    });
}
