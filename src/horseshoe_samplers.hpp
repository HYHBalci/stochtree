#ifndef HORSESHOE_SAMPLERS_HPP
#define HORSESHOE_SAMPLERS_HPP

#include <cpp11.hpp>
#include <cpp11/doubles.hpp>
#include <cpp11/list.hpp>
#include <vector>
#include <cmath>
#include <limits>

// Function declarations
double sample_beta_j_cpp(int N, cpp11::writable::doubles r_beta,
                         cpp11::writable::doubles z, cpp11::writable::doubles w_j,
                         double tau_j, double sigma, double tau_glob = 1);

double sample_tau_j_slice(
    double tau_old,
    double beta_j,            // main effect beta_j
    int index,                // index of {j}!
    const std::vector<double> & beta_int,  // all interaction betas
    const std::vector<double> & tau,       // all tau
    double tau_int,
    double sigma,
    bool interaction = true,  // or default false
    double step_out = 0.5,
    int max_steps = 50, double tau_glob = 1,  
    bool global_shrink = false,
    bool unlink = false
);

double sample_alpha_cpp(int N, cpp11::writable::doubles r_alpha,
                        cpp11::writable::doubles z_, double sigma, double alpha_prior_sd);

double sample_sigma2_ig_cpp(int N, cpp11::writable::doubles resid,
                            double shape_prior, double rate_prior);

cpp11::writable::list updateLinearTreatmentCpp_cpp(
    cpp11::doubles_matrix<> X, cpp11::doubles Z, cpp11::writable::doubles residual,
    double alpha, cpp11::writable::doubles beta, cpp11::writable::doubles beta_int,
    cpp11::writable::doubles tau_beta, double tau_int, double sigma, double alpha_prior_sd, double tau_glob = 1, 
    bool global_shrink = false, bool unlink = false);

#endif 