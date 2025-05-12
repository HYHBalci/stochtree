#include <cpp11.hpp>             // main cpp11 header
#include <cpp11/doubles.hpp>     // for doubles, writable::doubles
#include <cpp11/list.hpp>        // for writable::list
#include <cmath>                 // for std::log, std::sqrt
#include <limits>                // for std::numeric_limits<double>::infinity
#include <algorithm>             // for std::max
#include <vector>
#include <fstream>
#include <iostream> 
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
    const std::string& filename = "output_log.txt"
) {
  std::ofstream outfile(filename, std::ios::app);
  if (!outfile.is_open()) {
    
    return;
  }
  
  outfile << "=== Iteration Output ===\n";
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
    writable::doubles r_beta,
    writable::doubles z,
    writable::doubles w_j,
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
 
// --------------------------------------------
// 6) updateLinearTreatmentCpp
[[cpp11::register]]
cpp11::writable::doubles updateLinearTreatmentCpp_cpp(
    cpp11::doubles_matrix<> X,             // design matrix for beta
    cpp11::doubles Z,     
    cpp11::doubles propensity_train, // treatment indicator
    cpp11::writable::doubles residual,     // current total fit from the model
    double alpha,                          // current alpha
    cpp11::writable::doubles beta,  
    double gamma,                          // current beta vector
    cpp11::writable::doubles beta_int,     // interaction betas
    cpp11::writable::doubles tau_beta,  
    cpp11::writable::doubles nu,
    double xi,                             // local scales for each beta_j
    double tau_int,                        // scale for interactions
    double sigma,                          // current noise sd
    double alpha_prior_sd,
    double tau_glob = 1,                   // set global shrinkage parameter to 1 per default  
    bool global_shrink = false,
    bool unlink = false,
    bool propensity_seperate = false,
    bool gibbs = false,
    bool save_output = true, 
    int max_steps = 50,
    double step_out = 0.5
)
{
  int n = residual.size();
  int p_mod = X.ncol();
  // number of interaction pairs = p_mod*(p_mod+1)/2
  int p_int = (p_mod * (p_mod - 1)) / 2;
  if (unlink && (tau_beta.size() < (size_t)(p_mod + p_int))) {
    stop("tau_beta is too short for unlink=true! Need at least p_mod + p_int entries.");
  }
  
  // Build the interaction-pair index list
  std::vector<std::pair<int,int>> int_pairs;
  int_pairs.reserve(p_int);
  for(int i = 0; i < p_mod; i++){
    for(int j = i + 1; j < p_mod; j++){
      int_pairs.push_back(std::make_pair(i, j));
    }
  } 
  //Update gamma (propensity adjustment) if asked for. 
  if(propensity_seperate){
    // Add old alpha contribution back to residual
    for(int i = 0; i < n; i++){
      residual[i] += propensity_train[i]* gamma;
    } 
    // sample_alpha_cpp(...) is assumed to exist
    double gamma_new = sample_alpha_cpp(n, residual, propensity_train, sigma, alpha_prior_sd);
    
    // Remove new alpha from residual
    for(int i = 0; i < n; i++){
      residual[i] -= propensity_train[i] * gamma_new;
    } 
    gamma = gamma_new;
  }
  //------------------------------------------------
  // 1) Update alpha
  //------------------------------------------------
  // Add old alpha contribution back to residual
  for(int i = 0; i < n; i++){
    residual[i] += Z[i] * alpha;
  } 
  // sample_alpha_cpp(...) is assumed to exist
  double alpha_new = sample_alpha_cpp(n, residual, Z, sigma, alpha_prior_sd);
  
  // Remove new alpha from residual
  for(int i = 0; i < n; i++){
    residual[i] -= Z[i] * alpha_new;
  } 
  alpha = alpha_new;
   
  //------------------------------------------------
  // 2) Update each beta_j and its tau_beta[j]
  //------------------------------------------------
  // Make std::vector copies so we can pass them to sample_tau_j_slice
  std::vector<double> beta_int_std(beta_int.begin(), beta_int.end());
  std::vector<double> tau_beta_std(tau_beta.begin(), tau_beta.end());
  for(int j = 0; j < p_mod; j++){
    // 2a) Remove old beta_j from residual
    for(int i = 0; i < n; i++){
      residual[i] += Z[i] * beta[j] * X(i,j);
    }
     
    // Build local w_j
    cpp11::writable::doubles w_j(n);
    for(int i = 0; i < n; i++){
      w_j[i] = X(i, j);
    }
     
    double beta_j_new = sample_beta_j_cpp(n, residual, Z, w_j, tau_beta[j], sigma, tau_glob);
     
    // Remove the newly updated beta_j from residual
    for(int i = 0; i < n; i++){
      residual[i] -= Z[i] * beta_j_new * X(i,j);
    } 
    beta[j] = beta_j_new;
    double tb_j_new; 

    // 2b) Sample local scale tau_beta[j] via slice or via conjugate prior. 
    if(gibbs){
      nu[j] = rinvgamma(1, 1 + 1 / (tau_beta[j]*tau_beta[j]));

      tb_j_new = std::sqrt(rinvgamma(1, (1 / nu[j]) + (beta[j] * beta[j]) / (2*tau_glob*tau_glob*sigma*sigma))); // Following Makalic, Schmidt A simple sampler for the horseshoe estimator.
    } else {
      tb_j_new = sample_tau_j_slice(
        tau_beta[j],          // old tau_j
                beta[j],              // new beta_j 
                    j,                    // index
                    beta_int_std,         // entire vector of interaction betas
                    tau_beta_std,         // entire vector of tau_beta
                    tau_int,              // global interaction scale
                    sigma,                // noise sd
                    true,                 // interaction = true
                    step_out,                  // step_out
                    max_steps,
                    tau_glob,
                    unlink
      );
    }
    tau_beta[j] = tb_j_new;
    tau_beta_std[j] = tb_j_new;
  } 
  
  //------------------------------------------------
  // 3) Update pairwise interaction effects
  //------------------------------------------------
  for(size_t k = 0; k < int_pairs.size(); k++){
    int iVar = int_pairs[k].first;
    int jVar = int_pairs[k].second;
     
    // Remove old beta_int[k] from residual
    for(int i = 0; i < n; i++){
      double x_ij = X(i, iVar) * X(i, jVar);
      residual[i] += Z[i] * beta_int[k] * x_ij;
    }
     
    // Build w_ij
    cpp11::writable::doubles w_ij(n);
    for(int i = 0; i < n; i++){
      w_ij[i] = X(i, iVar) * X(i, jVar);
    } 
    
    // scale for normal prior on beta_int[k]
    double scale_int; 
    double beta_int_new;
    if(unlink){
      scale_int = tau_beta[p_mod + k];
      beta_int_new = sample_beta_j_cpp(n, residual, Z, w_ij, scale_int, sigma, tau_glob); //If unlink == TRUE set sigma equal to 1.
    } else{
      scale_int = std::sqrt(tau_int * tau_beta[iVar] * tau_beta[jVar]);
      beta_int_new = sample_beta_j_cpp(n, residual, Z, w_ij, scale_int, sigma, tau_glob);
    }
    // Remove newly updated beta_int from residual
    for(int i = 0; i < n; i++){
      double x_ij = X(i, iVar) * X(i, jVar);
      residual[i] -= Z[i] * beta_int_new * x_ij;
    } 
    beta_int[k] = beta_int_new;
    beta_int_std[k] = beta_int_new;
    if(unlink){
      double tb_j_new;
      if(gibbs){
        nu[p_mod+k] = rinvgamma(1, 1 + 1 / (tau_beta[p_mod+k]*tau_beta[p_mod+k]));
        tb_j_new = std::sqrt(rinvgamma(1, 1 / nu[p_mod+k] + (beta_int[k] * beta_int[k]) / (2*tau_glob*tau_glob*sigma*sigma)));
      } else {
      tb_j_new = sample_tau_j_slice(
      tau_beta[p_mod+k],          
              beta_int[k],              
                  k,                    // index
                  beta_int_std,         
                  tau_beta_std,         
                  tau_int,              
                  sigma,                
                  true,                
                  step_out,                 
                  max_steps,
                  tau_glob,
                  unlink
    );
    }
      tau_beta[p_mod+k] = tb_j_new;
      tau_beta_std[p_mod+k] = tb_j_new;
    }}
   
  //------------------------------------------------
  // 4) sample tau_int using a MH step. 
  //------------------------------------------------
  std::vector<double> beta_std2(beta.begin(), beta.end());
  std::vector<double> beta_int_std2(beta_int.begin(), beta_int.end());
  std::vector<double> tau_beta_std2(tau_beta.begin(), tau_beta.end());
  if((!global_shrink) && (!unlink)){
    double currentTauInt   = tau_int;
    double proposedTauInt  = ::Rf_runif(0.01, 1.0);
  
    double logPosteriorCurrent = loglikeTauInt(
      currentTauInt,
      beta_int_std2,
      int_pairs,
      tau_beta_std2,
      sigma,
      tau_glob
      ); 
    double logPosteriorProposed = loglikeTauInt(
      proposedTauInt,
      beta_int_std2,
      int_pairs,
      tau_beta_std2,
      sigma,
      tau_glob
    ); 
    double logAccept = logPosteriorProposed - logPosteriorCurrent;
  
    if(::Rf_runif(0.0, 1.0) < std::exp(logAccept)){
      tau_int = proposedTauInt;
    } else { 
      tau_int = currentTauInt;
    }} else {
      tau_int = 1;
    }
  if(global_shrink){
    if(!gibbs){
      tau_glob = sample_tau_global_slice(tau_glob, beta_std2, beta_int_std2, tau_beta_std2, tau_int, sigma);
    }
  }
  if(gibbs){
    xi = rinvgamma(1, 1 + 1 / (tau_glob*tau_glob));
    double sum_scale;
    if(unlink){
      sum_scale = 0;
      for(int i = 0; i < p_mod + p_int; i++){
        if(i < p_mod){
          sum_scale = sum_scale +  (beta[i]*beta[i]) / (tau_beta[i] * tau_beta[i]);
        } else {
          sum_scale = sum_scale + (beta_int[i-p_mod] * beta_int[i-p_mod]) / (tau_beta[i] * tau_beta[i]);
        }
      }
      tau_glob = std::sqrt(rinvgamma((p_mod + p_int + 1)/2, 1/xi + (1/(2*sigma*sigma))*sum_scale));
    } else{
      sum_scale = 0;
      for(int i = 0; i < p_mod; i++){
        sum_scale += (beta[i]*beta[i]) / (tau_beta[i] * tau_beta[i]);
      }
      tau_glob = std::sqrt(rinvgamma((p_mod + 1)/2, 1/xi + (1/(2*sigma*sigma))*sum_scale));
    }
  
  }
  
  
   
  //------------------------------------------------
  // Prepare output
  //------------------------------------------------
  size_t total_size = 5 
  + beta.size() 
    + beta_int.size() 
    + tau_beta.size() 
    + nu.size()
    + residual.size(); 
    
    cpp11::writable::doubles output;
    output.reserve(total_size);
    
    // alpha, tau_int, tau_glob, gamma 
    output.push_back(alpha);
    output.push_back(tau_int);
    output.push_back(tau_glob);
    output.push_back(gamma);
    output.push_back(xi);
    // beta
    for(double val : beta){
      output.push_back(val);
    }
    
    // beta_int
    for(double val : beta_int){
      output.push_back(val);
    }
    
    // tau_beta
    for(double val : tau_beta){
      output.push_back(val);
    }
    // nu 
    for(double val : nu){
      output.push_back(val);
    }
    // residual
    for(double val : residual){
      output.push_back(val);
    }
    
    if(save_output){
      
    
    save_output_vector_labeled(
      alpha, tau_int, tau_glob, gamma, xi, sigma,
      beta, beta_int, tau_beta, nu, residual
    );
    }
    return output;
}