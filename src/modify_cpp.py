import os

cpp_file = r"C:\Users\P094412\OneDrive - Amsterdam UMC\Documenten\stochtree\src\horseshoe_samplers_amr.cpp"
with open(cpp_file, 'r') as f:
    code = f.read()

# Replace function names
code = code.replace("updateLinearTreatmentCpp_cpp_old", "updateLinearTreatmentCpp_amr")
code = code.replace("updateLinearTreatmentCpp_NCP_cpp_old", "updateLinearTreatmentCpp_NCP_amr")

# Replace sample_alpha_cpp definition
old_alpha_def = "double sample_alpha_cpp(const cpp11::writable::doubles& r_alpha, const cpp11::doubles& z, double sigma, double alpha_prior_sd) {"
new_alpha_def = """double sample_alpha_cpp(const cpp11::writable::doubles& r_alpha, const cpp11::doubles& z, const cpp11::doubles& w, double sigma, double alpha_prior_sd) {
  int N = r_alpha.size();
  Eigen::Map<const Eigen::VectorXd> r_alpha_map(REAL(r_alpha), N);
  Eigen::Map<const Eigen::VectorXd> z_map(REAL(z), N);
  Eigen::Map<const Eigen::VectorXd> w_map(REAL(w), N);
  double sigma2 = sigma * sigma;
  double prior_var = alpha_prior_sd * alpha_prior_sd;
  double sum_z_sq = (z_map.array().square() * w_map.array()).sum();
  double sum_rz = (r_alpha_map.array() * z_map.array() * w_map.array()).sum();
  double prec_data = sum_z_sq / sigma2;
  double prec_prior = 1.0 / prior_var;
  double postVar = 1.0 / (prec_data + prec_prior);
  double mean_post = postVar * (sum_rz / sigma2);
  return Rf_rnorm(mean_post, std::sqrt(postVar));
}"""

# Need to replace the whole body of sample_alpha_cpp
import re
code = re.sub(r'double sample_alpha_cpp.*?return Rf_rnorm.*?\} \n', new_alpha_def + ' \n', code, flags=re.DOTALL)

# Add obs_weights parameter to the two main functions
code = code.replace("cpp11::writable::doubles residual,\n    const cpp11::r_vector<int>& are_continuous,", 
                    "cpp11::writable::doubles residual,\n    const cpp11::doubles& obs_weights,\n    const cpp11::r_vector<int>& are_continuous,")

# Update gamma and alpha sampling
code = code.replace("gamma = sample_alpha_cpp(residual, propensity_train, sigma, 10);", 
                    "gamma = sample_alpha_cpp(residual, propensity_train, obs_weights, sigma, 10);")
code = code.replace("alpha = sample_alpha_cpp(residual, Z, sigma, alpha_prior_sd);", 
                    "alpha = sample_alpha_cpp(residual, Z, obs_weights, sigma, alpha_prior_sd);")
code = code.replace("alpha_tilde = sample_alpha_cpp(residual, Z, sigma, alpha_prior_sd);", 
                    "alpha_tilde = sample_alpha_cpp(residual, Z, obs_weights, sigma, alpha_prior_sd);")

# Update Bhattacharya sampler
code = code.replace("delta(i) = Rf_rnorm(0.0, 1.0);", 
                    "delta(i) = Rf_rnorm(0.0, 1.0 / std::sqrt(obs_weights[i]));")
code = code.replace("M_solve.diagonal().array() += 1.0;", 
                    "for (int i = 0; i < n; ++i) M_solve(i,i) += 1.0 / obs_weights[i];")

# Update Gibbs and Slice samplers
code = code.replace("XtX.selfadjointView<Eigen::Lower>().rankUpdate(x_row_combined);", 
                    "XtX.selfadjointView<Eigen::Lower>().rankUpdate(x_row_combined, obs_weights[i]);")
code = code.replace("Xt_y += x_row_combined * y_target(i);", 
                    "Xt_y += x_row_combined * (y_target(i) * obs_weights[i]);")

with open(cpp_file, 'w') as f:
    f.write(code)
print("done")
