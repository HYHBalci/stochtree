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
    double beta_j,
    int index,
    const std::vector<double>& beta_int,
    const std::vector<double>& tau,
    const std::vector<std::pair<int, int>>& int_pairs,
    double tau_int,
    double sigma,
    bool interaction = true,
    double step_out = 0.5,
    int max_steps = 50, double tau_glob = 1,  
    bool global_shrink = false,
    bool unlink = false
);


double sample_alpha_cpp(int N, cpp11::writable::doubles r_alpha,
                        cpp11::writable::doubles z_, double sigma, double alpha_prior_sd);

double sample_sigma2_ig_cpp(int N, cpp11::writable::doubles resid,
                            double shape_prior, double rate_prior);

cpp11::writable::doubles updateLinearTreatmentCpp_cpp(
    cpp11::doubles_matrix<> X,
    cpp11::doubles Z,    
    cpp11::doubles propensity_train,
    cpp11::writable::doubles residual,
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
);

// cpp11::list run_mcmc_sampler_2(
//     const cpp11::matrix<cpp11::r_vector<double>, cpp11::by_column>& X_r,
//     const cpp11::r_vector<double>& Y_r,
//     const cpp11::r_vector<double>& Z_r,
//     const cpp11::integers& are_continuous_r, // This type is already read-only
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
// );

#endif 