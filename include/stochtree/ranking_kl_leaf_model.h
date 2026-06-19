#ifndef STOCHTREE_RANKING_KL_LEAF_MODEL_H_
#define STOCHTREE_RANKING_KL_LEAF_MODEL_H_

#include <Eigen/Dense>
#include <stochtree/data.h>
#include <stochtree/partition_tracker.h>
#include <stochtree/tree.h>
#include <stochtree/normal_sampler.h>
#include <random>
#include <cmath>
#include <algorithm>

namespace StochTree {

class ListNetKLSuffStat {
 public:
  data_size_t n;
  double sum_g;
  double sum_h;
  ListNetKLSuffStat() { n = 0; sum_g = 0.0; sum_h = 0.0; }
  void IncrementSuffStat(ForestDataset& dataset, Eigen::VectorXd& outcome, ForestTracker& tracker, data_size_t row_idx, int tree_idx) {
    if (row_idx < 0 || row_idx >= outcome.size()) {
      std::cout << "ListNetKLSuffStat::IncrementSuffStat INVALID ROW_IDX: " << row_idx << " outcome size: " << outcome.size() << std::endl;
    }
    n++;
    sum_g += outcome(row_idx, 0); // outcome stores g_i
    if (dataset.HasVarWeights()) {
      sum_h += dataset.VarWeightValue(row_idx); // var_weights stores h_i
    }
  }
  void ResetSuffStat() { n = 0; sum_g = 0.0; sum_h = 0.0; }
  void AddSuffStat(ListNetKLSuffStat& lhs, ListNetKLSuffStat& rhs) {
    n = lhs.n + rhs.n; sum_g = lhs.sum_g + rhs.sum_g; sum_h = lhs.sum_h + rhs.sum_h;
  }
  void SubtractSuffStat(ListNetKLSuffStat& lhs, ListNetKLSuffStat& rhs) {
    n = lhs.n - rhs.n; sum_g = lhs.sum_g - rhs.sum_g; sum_h = lhs.sum_h - rhs.sum_h;
  }
  bool SampleGreaterThan(data_size_t threshold) { return n > threshold; }
  bool SampleGreaterThanEqual(data_size_t threshold) { return n >= threshold; }
  data_size_t SampleSize() { return n; }
};

class ListNetKLLeafModel {
 public:
  ListNetKLLeafModel(double tau) { tau_ = tau; normal_sampler_ = UnivariateNormalSampler(); }
  ~ListNetKLLeafModel() {}
  double SplitLogMarginalLikelihood(ListNetKLSuffStat& left_stat, ListNetKLSuffStat& right_stat, double global_variance);
  double NoSplitLogMarginalLikelihood(ListNetKLSuffStat& suff_stat, double global_variance);
  double PosteriorParameterMean(ListNetKLSuffStat& suff_stat, double global_variance);
  double PosteriorParameterVariance(ListNetKLSuffStat& suff_stat, double global_variance);
  void SampleLeafParameters(ForestDataset& dataset, ForestTracker& tracker, ColumnVector& residual, Tree* tree, int tree_num, double global_variance, std::mt19937& gen);
  void SetEnsembleRootPredictedValue(ForestDataset& dataset, TreeEnsemble* ensemble, double root_pred_value);
  void SampleSplitPriorParameter(double& alpha, double& beta) {}
  bool RequiresBasis() { return false; }
 private:
  double tau_;
  UnivariateNormalSampler normal_sampler_;
};

class DistributionalKLSuffStat {
 public:
  data_size_t n;
  data_size_t n_T;
  double sum_y_T;
  double sum_y2_T;
  data_size_t n_C;
  double sum_y_C;
  double sum_y2_C;
  DistributionalKLSuffStat() {
    n = 0; n_T = 0; sum_y_T = 0.0; sum_y2_T = 0.0;
    n_C = 0; sum_y_C = 0.0; sum_y2_C = 0.0;
  }
  void IncrementSuffStat(ForestDataset& dataset, Eigen::VectorXd& outcome, ForestTracker& tracker, data_size_t row_idx, int tree_idx) {
    n++;
    UpliftForestDataset* uplift_dataset = static_cast<UpliftForestDataset*>(&dataset);
    double y = outcome(row_idx, 0); // true outcome
    if (uplift_dataset->HasTreatmentIndicator() && uplift_dataset->TreatmentIndicatorValue(row_idx) == 1) {
      n_T++;
      sum_y_T += y;
      sum_y2_T += y * y;
    } else {
      n_C++;
      sum_y_C += y;
      sum_y2_C += y * y;
    }
  }
  void ResetSuffStat() {
    n = 0; n_T = 0; sum_y_T = 0.0; sum_y2_T = 0.0;
    n_C = 0; sum_y_C = 0.0; sum_y2_C = 0.0;
  }
  void AddSuffStat(DistributionalKLSuffStat& lhs, DistributionalKLSuffStat& rhs) {
    n = lhs.n + rhs.n;
    n_T = lhs.n_T + rhs.n_T; sum_y_T = lhs.sum_y_T + rhs.sum_y_T; sum_y2_T = lhs.sum_y2_T + rhs.sum_y2_T;
    n_C = lhs.n_C + rhs.n_C; sum_y_C = lhs.sum_y_C + rhs.sum_y_C; sum_y2_C = lhs.sum_y2_C + rhs.sum_y2_C;
  }
  void SubtractSuffStat(DistributionalKLSuffStat& lhs, DistributionalKLSuffStat& rhs) {
    n = lhs.n - rhs.n;
    n_T = lhs.n_T - rhs.n_T; sum_y_T = lhs.sum_y_T - rhs.sum_y_T; sum_y2_T = lhs.sum_y2_T - rhs.sum_y2_T;
    n_C = lhs.n_C - rhs.n_C; sum_y_C = lhs.sum_y_C - rhs.sum_y_C; sum_y2_C = lhs.sum_y2_C - rhs.sum_y2_C;
  }
  bool SampleGreaterThan(data_size_t threshold) { return n > threshold; }
  bool SampleGreaterThanEqual(data_size_t threshold) { return n >= threshold; }
  data_size_t SampleSize() { return n; }
};

class DistributionalKLLeafModel {
 public:
  DistributionalKLLeafModel(double tau, double alpha) { tau_ = tau; alpha_ = alpha; normal_sampler_ = UnivariateNormalSampler(); }
  ~DistributionalKLLeafModel() {}
  double SplitLogMarginalLikelihood(DistributionalKLSuffStat& left_stat, DistributionalKLSuffStat& right_stat, double global_variance);
  double NoSplitLogMarginalLikelihood(DistributionalKLSuffStat& suff_stat, double global_variance);
  double PosteriorParameterMean(DistributionalKLSuffStat& suff_stat, double global_variance);
  double PosteriorParameterVariance(DistributionalKLSuffStat& suff_stat, double global_variance);
  void SampleLeafParameters(ForestDataset& dataset, ForestTracker& tracker, ColumnVector& residual, Tree* tree, int tree_num, double global_variance, std::mt19937& gen);
  void SetEnsembleRootPredictedValue(ForestDataset& dataset, TreeEnsemble* ensemble, double root_pred_value);
  void SampleSplitPriorParameter(double& alpha, double& beta) {}
  bool RequiresBasis() { return false; }
 private:
  double tau_;
  double alpha_;
  UnivariateNormalSampler normal_sampler_;
  double ComputeKLDivergence(DistributionalKLSuffStat& stat);
};

} // namespace StochTree

#endif // STOCHTREE_RANKING_KL_LEAF_MODEL_H_
