#include <cpp11.hpp>
#include <stochtree/dispatcher.h>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
using namespace cpp11;

[[cpp11::register]]
cpp11::external_pointer<StochTree::MCMCDispatcher> bart_sample_cpp(cpp11::doubles_matrix<> y, cpp11::doubles_matrix<> X, cpp11::doubles_matrix<> omega, cpp11::doubles_matrix<> rfx_basis, cpp11::integers rfx_groups,
                                                                   int num_rfx_groups, int num_samples, int num_burnin, int num_trees, double nu, double lambda, double a_rfx, double b_rfx, int random_seed = -1) {
    // Extract dimensions of covariate matrix X and pointer to its contiguous block of memory
    int p_covariates = X.ncol();
    int n = X.nrow();
    double* covariate_data_ptr = REAL(PROTECT(X));
    
    // Extract dimensions of basis matrix omega and pointer to its contiguous block of memory
    int p_basis = omega.ncol();
    double* basis_data_ptr = REAL(PROTECT(omega));
    
    // Extract dimensions of outcome matrix y and pointer to its contiguous block of memory
    int p_outcome = y.ncol();
    double* outcome_data_ptr = REAL(PROTECT(y));
    
    // Extract dimensions of random effects basis matrix rfx_basis and pointer to its contiguous block of memory
    int p_rfx_basis = rfx_basis.ncol();
    double* rfx_basis_data_ptr = REAL(PROTECT(rfx_basis));
    
    // Unload group indices into a std::vector
    std::vector<int32_t> rfx_group_indices(n);
    for (int i = 0; i < n; i++) {
        rfx_group_indices[i] = rfx_groups[i];
    }

    // Configure MCMCDispatcher object
    std::unique_ptr<StochTree::MCMCDispatcher> bart_ptr_;
    bart_ptr_.reset(new StochTree::MCMCDispatcher(num_samples, num_burnin, num_trees, random_seed));
    
    // Initialize model classes
    StochTree::GaussianHomoskedasticUnivariateRegressionModelWrapper model = StochTree::GaussianHomoskedasticUnivariateRegressionModelWrapper();
    model.SetGlobalParameter(1., StochTree::GlobalParamName::GlobalVariance);
    model.SetGlobalParameter(1., StochTree::GlobalParamName::LeafPriorVariance);
    StochTree::GlobalHomoskedasticVarianceModel variance_model = StochTree::GlobalHomoskedasticVarianceModel();
    StochTree::ClassicTreePrior tree_prior{0.95, 2.0, 10};

    // Run the sampler
    std::vector<StochTree::FeatureType> feature_types(p_covariates, StochTree::FeatureType::kNumeric);
    bart_ptr_->SampleModel<StochTree::GaussianHomoskedasticUnivariateRegressionModelWrapper, StochTree::ClassicTreePrior>(covariate_data_ptr, p_covariates, basis_data_ptr, p_basis, outcome_data_ptr, p_outcome, rfx_basis_data_ptr, p_rfx_basis, num_rfx_groups, rfx_group_indices, n, false, true, nu, lambda, a_rfx, b_rfx, model, tree_prior, variance_model);

    // Unprotect pointers
    UNPROTECT(3);

    return cpp11::external_pointer<StochTree::MCMCDispatcher>(bart_ptr_.release());
}

[[cpp11::register]]
cpp11::writable::doubles_matrix<> bart_predict_cpp(cpp11::external_pointer<StochTree::MCMCDispatcher> bart_ptr, cpp11::doubles_matrix<> X, cpp11::doubles_matrix<> omega, cpp11::doubles_matrix<> rfx_basis, cpp11::integers rfx_groups, int num_samples) {
    // Extract dimensions of covariate matrix X and pointer to its contiguous block of memory
    int p_covariate = X.ncol();
    int n = X.nrow();
    double* covariate_data_ptr = REAL(PROTECT(X));
    
    // Extract dimensions of basis matrix omega and pointer to its contiguous block of memory
    int p_basis = omega.ncol();
    double* basis_data_ptr = REAL(PROTECT(omega));
    
    // Extract dimensions of random effects basis matrix rfx_basis and pointer to its contiguous block of memory
    int p_rfx_basis = rfx_basis.ncol();
    double* rfx_basis_data_ptr = REAL(PROTECT(rfx_basis));
    
    // Unload group indices into a std::vector
    std::vector<int32_t> rfx_group_indices(n);
    for (int i = 0; i < n; i++) {
        rfx_group_indices[i] = rfx_groups[i];
    }

    // Predict from the sampled BART model
    // std::vector<double> output_raw = bart_ptr->PredictSamples();
    std::vector<double> output_raw = bart_ptr->PredictSamples(covariate_data_ptr, p_covariate, basis_data_ptr, p_basis, rfx_basis_data_ptr, p_rfx_basis, rfx_group_indices, n, false);

    // Convert result to a matrix
    cpp11::writable::doubles_matrix<> output(n, num_samples);
    for (size_t i = 0; i < n; i++) {
        for (int j = 0; j < num_samples; j++) {
            output(i, j) = output_raw[n*j + i];
        }
    }

    // Unprotect pointers
    UNPROTECT(2);

    return output;
}
