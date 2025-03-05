#ifndef HORSESHOESAMPLERS_H
#define HORSESHOESAMPLERS_H

#include <Rcpp.h>
#include <cmath>
#include <limits>

// Ensure M_PI is defined if not available
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Namespace declaration to avoid repetition
using namespace Rcpp;

double sample_beta_j(
    int N,
    NumericVector r_beta,
    NumericVector z,
    NumericVector w_j,
    double tau_j,
    double sigma
);

double sample_tau_j_slice(
    double tau_old,
    double beta_j,
    double sigma,
    double step_out = 0.5,
    int max_steps = 50
);

double sample_alpha(
    int N,
    NumericVector r_alpha, // partial residual
    NumericVector z_,      // treatment indicator
    double sigma,
    double alpha_prior_sd = 10.0
);

double sample_sigma2_ig(
    int N,
    NumericVector resid,   // vector of residuals e_i
    double shape_prior = 1.0, 
    double rate_prior  = 0.001
);

double loglikeTauInt(
    double tau_int,
    const std::vector<double> &beta_int_base,
    const std::vector<std::pair<int,int>> &int_pairs_base,
    const std::vector<double> &tau_main,
    double sigma,
    
    bool include_treatment_int = false,
    const std::vector<double> &beta_int_trt = std::vector<double>(),
    const std::vector<double> &tau_trt = std::vector<double>(),
    const std::vector<std::pair<int,int>> &int_pairs_trt = std::vector<std::pair<int,int>>()
);  

#endif 