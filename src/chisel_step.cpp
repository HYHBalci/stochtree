#include <cpp11.hpp>
#include <cpp11/doubles.hpp>
#include <cpp11/matrix.hpp>
#include <Eigen/Dense>
#include <Rmath.h>
#include <R_ext/Random.h> 
#include <cmath>

using namespace cpp11; 

double rtruncnorm_cpp(double mu, double sigma, bool positive) {
  double u = Rf_runif(0.0, 1.0);
  
  u = std::max(1e-15, std::min(u, 1.0 - 1e-15));
  
  if (positive) {
    double log_p_gt_0 = Rf_pnorm5(0.0, mu, sigma, 0, 1); 
    double log_p_target = std::log(1.0 - u) + log_p_gt_0;
    return Rf_qnorm5(log_p_target, mu, sigma, 0, 1); 
  } else {
    double log_p_lt_0 = Rf_pnorm5(0.0, mu, sigma, 1, 1); 
    double log_p_target = std::log(u) + log_p_lt_0;
    return Rf_qnorm5(log_p_target, mu, sigma, 1, 1);
  } 
}  

double rinvgamma_cpp(double shape, double scale) {
  double safe_scale = std::max(scale, 1e-15);
  double g = Rf_rgamma(shape, 1.0 / safe_scale);
  
  g = std::max(g, 1e-15);
  return 1.0 / g;
} 

[[cpp11::register]]
writable::list horseshoe_probit_step_cpp(
    const doubles_matrix<>& X,
    const integers& y_star,
    const doubles& weights,
    const doubles& beta_in,
    const doubles& lambda_sq_in,
    double tau_sq,
    const doubles& nu_in,
    double xi) {
  
  GetRNGstate();
  
  int n = X.nrow();
  int p = X.ncol();
  
  // FIXED: Use REAL() to safely extract the raw double pointer from the cpp11 SEXP objects
  Eigen::Map<const Eigen::MatrixXd> X_map(REAL(X), n, p);
  Eigen::Map<const Eigen::VectorXd> W_map(REAL(weights), n);
  Eigen::Map<const Eigen::VectorXd> beta_map(REAL(beta_in), p);
  Eigen::Map<const Eigen::VectorXd> lambda_sq_map(REAL(lambda_sq_in), p);
  Eigen::Map<const Eigen::VectorXd> nu_map(REAL(nu_in), p);
  
  Eigen::VectorXd beta = beta_map;
  Eigen::VectorXd lambda_sq = lambda_sq_map;
  Eigen::VectorXd nu = nu_map;
  Eigen::VectorXd z(n);
  
  // 1. Sample Latent z_i (Albert-Chib)
  Eigen::VectorXd X_beta = X_map * beta;
  for (int i = 0; i < n; ++i) {
    double w_safe = W_map(i);
    if (std::isnan(w_safe) || w_safe < 1e-8) w_safe = 1e-8;
    
    double mu_i = X_beta(i);
    if (std::isnan(mu_i)) mu_i = 0.0;
    
    double sd_i = 1.0 / std::sqrt(w_safe);
    z(i) = rtruncnorm_cpp(mu_i, sd_i, y_star[i] == 1);
  }  
  
  // 2. Sample Beta
  Eigen::MatrixXd X_T_W = X_map.transpose() * W_map.asDiagonal();
  Eigen::MatrixXd Prec = X_T_W * X_map;
  
  Eigen::VectorXd Lambda_inv_diag(p);
  Lambda_inv_diag(0) = 1e-4; 
  
  for(int j = 1; j < p; ++j) {
    double shrink_factor = lambda_sq(j) * tau_sq;
    shrink_factor = std::max(shrink_factor, 1e-12); 
    double precision_val = 1.0 / shrink_factor;
    Lambda_inv_diag(j) = std::min(precision_val, 1e8); 
  }  
  
  Prec.diagonal() += Lambda_inv_diag;
  
  Eigen::LLT<Eigen::MatrixXd> llt(Prec);
  if (llt.info() != Eigen::Success) {
    Prec.diagonal().array() += 1e-4; 
    llt.compute(Prec);
  }  
  
  if (llt.info() != Eigen::Success) {
    for(int j=0; j<p; ++j) beta(j) = Rf_rnorm(0.0, 1e-2);
  } else { 
    Eigen::VectorXd mean = llt.solve(X_T_W * z);
    Eigen::VectorXd noise(p);
    for(int j=0; j<p; ++j) noise(j) = Rf_rnorm(0.0, 1.0);
    beta = mean + llt.matrixU().solve(noise);
  }   
  
  // 3. Sample Horseshoe Variances 
  double sum_beta_sq = 0.0;
  for (int j = 1; j < p; ++j) {
    lambda_sq(j) = rinvgamma_cpp(0.5, 1.0 / std::max(nu(j), 1e-8) + (beta(j) * beta(j)) / std::max(2.0 * tau_sq, 1e-8));
    nu(j) = rinvgamma_cpp(0.5, 1.0 + 1.0 / std::max(lambda_sq(j), 1e-8));
    
    sum_beta_sq += (beta(j) * beta(j)) / std::max(lambda_sq(j), 1e-8);
  }  
  
  tau_sq = rinvgamma_cpp(((p - 1.0) + 1.0) / 2.0, 1.0 / std::max(xi, 1e-8) + sum_beta_sq / 2.0);
  xi = rinvgamma_cpp(0.5, 1.0 + 1.0 / std::max(tau_sq, 1e-8));
  
  PutRNGstate();
  
  writable::doubles beta_out(p);
  writable::doubles lambda_sq_out(p);
  writable::doubles nu_out(p);
  for(int j=0; j<p; ++j) {
    beta_out[j] = beta(j);
    lambda_sq_out[j] = lambda_sq(j);
    nu_out[j] = nu(j);
  }   
  
  return writable::list({
    "beta"_nm = beta_out,
      "lambda_sq"_nm = lambda_sq_out,
      "tau_sq"_nm = tau_sq,
      "nu"_nm = nu_out,
      "xi"_nm = xi 
  }); 
}