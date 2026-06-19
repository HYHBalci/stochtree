#include <stochtree/ranking_kl_leaf_model.h>
#include <stochtree/leaf_model.h>
#include <cmath>
#include <algorithm>

namespace StochTree {

// --- ListNetKLLeafModel ---

double ListNetKLLeafModel::SplitLogMarginalLikelihood(ListNetKLSuffStat& left_stat, ListNetKLSuffStat& right_stat, double global_variance) {
  double left_log_ml = -0.5 * std::log(1.0 + tau_ * left_stat.sum_h) + (tau_ * left_stat.sum_g * left_stat.sum_g) / (2.0 * (tau_ * left_stat.sum_h + 1.0));
  double right_log_ml = -0.5 * std::log(1.0 + tau_ * right_stat.sum_h) + (tau_ * right_stat.sum_g * right_stat.sum_g) / (2.0 * (tau_ * right_stat.sum_h + 1.0));
  return left_log_ml + right_log_ml;
}

double ListNetKLLeafModel::NoSplitLogMarginalLikelihood(ListNetKLSuffStat& suff_stat, double global_variance) {
  return -0.5 * std::log(1.0 + tau_ * suff_stat.sum_h) + (tau_ * suff_stat.sum_g * suff_stat.sum_g) / (2.0 * (tau_ * suff_stat.sum_h + 1.0));
}

double ListNetKLLeafModel::PosteriorParameterMean(ListNetKLSuffStat& suff_stat, double global_variance) {
  return -(tau_ * suff_stat.sum_g) / (suff_stat.sum_h * tau_ + 1.0);
}

double ListNetKLLeafModel::PosteriorParameterVariance(ListNetKLSuffStat& suff_stat, double global_variance) {
  return tau_ / (suff_stat.sum_h * tau_ + 1.0);
}



void ListNetKLLeafModel::SampleLeafParameters(ForestDataset& dataset, ForestTracker& tracker, ColumnVector& residual, Tree* tree, int tree_num, double global_variance, std::mt19937& gen) {
  std::vector<int32_t> tree_leaves = tree->GetLeaves();
  ListNetKLSuffStat node_suff_stat = ListNetKLSuffStat();
  
  double node_mean;
  double node_variance;
  double node_mu;
  int32_t leaf_id;
  for (int i = 0; i < tree_leaves.size(); i++) {
    leaf_id = tree_leaves[i];
    node_suff_stat.ResetSuffStat();
    AccumulateSingleNodeSuffStat<ListNetKLSuffStat, false>(node_suff_stat, dataset, tracker, residual, tree_num, leaf_id);
    
    node_mean = PosteriorParameterMean(node_suff_stat, global_variance);
    node_variance = PosteriorParameterVariance(node_suff_stat, global_variance);
    node_mu = normal_sampler_.Sample(node_mean, node_variance, gen);
    tree->SetLeaf(leaf_id, node_mu);
  }
}

void ListNetKLLeafModel::SetEnsembleRootPredictedValue(ForestDataset& dataset, TreeEnsemble* ensemble, double root_pred_value) {}

// --- DistributionalKLLeafModel ---

double DistributionalKLLeafModel::ComputeKLDivergence(DistributionalKLSuffStat& stat) {
  if (stat.n_T < 2 || stat.n_C < 2) return 0.0;
  double mu_T = stat.sum_y_T / stat.n_T;
  double mu_C = stat.sum_y_C / stat.n_C;
  double var_T = std::max(1e-6, (stat.sum_y2_T - stat.n_T * mu_T * mu_T) / (stat.n_T - 1));
  double var_C = std::max(1e-6, (stat.sum_y2_C - stat.n_C * mu_C * mu_C) / (stat.n_C - 1));
  return std::log(std::sqrt(var_C)/std::sqrt(var_T)) + (var_T + (mu_T - mu_C)*(mu_T - mu_C))/(2.0*var_C) - 0.5;
}

double DistributionalKLLeafModel::SplitLogMarginalLikelihood(DistributionalKLSuffStat& left_stat, DistributionalKLSuffStat& right_stat, double global_variance) {
  return alpha_ * (left_stat.n * ComputeKLDivergence(left_stat) + right_stat.n * ComputeKLDivergence(right_stat));
}

double DistributionalKLLeafModel::NoSplitLogMarginalLikelihood(DistributionalKLSuffStat& suff_stat, double global_variance) {
  return alpha_ * suff_stat.n * ComputeKLDivergence(suff_stat);
}

double DistributionalKLLeafModel::PosteriorParameterMean(DistributionalKLSuffStat& stat, double global_variance) {
  double mu_T = stat.n_T > 0 ? stat.sum_y_T / stat.n_T : 0.0;
  double mu_C = stat.n_C > 0 ? stat.sum_y_C / stat.n_C : 0.0;
  double n_eff = (stat.n_T * stat.n_C) / (double)(std::max(1, stat.n_T + stat.n_C));
  double prec_data = n_eff / global_variance;
  double prec_prior = 1.0 / tau_;
  return (prec_data / (prec_data + prec_prior)) * (mu_T - mu_C);
}

double DistributionalKLLeafModel::PosteriorParameterVariance(DistributionalKLSuffStat& stat, double global_variance) {
  double n_eff = (stat.n_T * stat.n_C) / (double)(std::max(1, stat.n_T + stat.n_C));
  double prec_data = n_eff / global_variance;
  double prec_prior = 1.0 / tau_;
  return 1.0 / (prec_data + prec_prior);
}

void DistributionalKLLeafModel::SampleLeafParameters(ForestDataset& dataset, ForestTracker& tracker, ColumnVector& residual, Tree* tree, int tree_num, double global_variance, std::mt19937& gen) {
  std::vector<int32_t> tree_leaves = tree->GetLeaves();
  DistributionalKLSuffStat node_suff_stat = DistributionalKLSuffStat();
  
  double node_mean;
  double node_variance;
  double node_mu;
  int32_t leaf_id;
  for (int i = 0; i < tree_leaves.size(); i++) {
    leaf_id = tree_leaves[i];
    node_suff_stat.ResetSuffStat();
    
    AccumulateSingleNodeSuffStat<DistributionalKLSuffStat, false>(node_suff_stat, dataset, tracker, residual, tree_num, leaf_id);
    
    node_mean = PosteriorParameterMean(node_suff_stat, global_variance);
    node_variance = PosteriorParameterVariance(node_suff_stat, global_variance);
    node_mu = normal_sampler_.Sample(node_mean, node_variance, gen);
    tree->SetLeaf(leaf_id, node_mu);
  }
}

void DistributionalKLLeafModel::SetEnsembleRootPredictedValue(ForestDataset& dataset, TreeEnsemble* ensemble, double root_pred_value) {}

} // namespace StochTree
