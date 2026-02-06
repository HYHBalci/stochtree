#include <cpp11.hpp>
#include <cpp11/doubles.hpp>
#include <cpp11/matrix.hpp>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <Eigen/Dense>

// StochTree Headers
#include <stochtree/tree.h>
#include <stochtree/data.h>
#include <stochtree/ensemble.h>

using namespace cpp11;

[[cpp11::register]]
writable::doubles_matrix<> get_saabas_shapley_active_forest_cpp(
    external_pointer<StochTree::TreeEnsemble> active_forest, 
    external_pointer<StochTree::ForestDataset> dataset) {
  
  const Eigen::MatrixXd& X = dataset->GetCovariates(); 
  int n = (int)X.rows();
  int p = (int)X.cols();
  int num_trees = active_forest->NumTrees();
  
  writable::doubles_matrix<> shapley_values(n, p);
  
  // Safe Initialization
  for(int i=0; i<n; ++i) {
    for(int j=0; j<p; ++j) {
      shapley_values(i, j) = 0.0;
    }
  }
  
  for (int t = 0; t < num_trees; t++) {
    StochTree::Tree* tree = active_forest->GetTree(t);
    std::vector<int32_t> nodes = tree->GetNodes();
    int max_node_id = 0;
    if (!nodes.empty()) {
      max_node_id = *std::max_element(nodes.begin(), nodes.end());
    } 
    // Add buffer +1 because IDs are 0-indexed
    int vec_size = max_node_id + 1;
    
    std::vector<double> node_sums(vec_size, 0.0);
    std::vector<double> node_counts(vec_size, 0.0);
    
    // PASS 1: Node Means
    for (int i = 0; i < n; i++) {
      int current = 0;
      while (!tree->IsLeaf(current)) {
        if (current >= vec_size) break; // Safety Break
        
        int split_feat = tree->SplitIndex(current);
        if (split_feat < 0 || split_feat >= p) break; // Bounds check
         
        bool go_left = false;
        if (tree->IsNumericSplitNode(current)) {
          go_left = (X(i, split_feat) <= tree->Threshold(current));
        } else { 
          uint32_t val = static_cast<uint32_t>(X(i, split_feat));
          const std::vector<uint32_t>& cats = tree->CategoryList(current);
          for(auto c : cats) { if(c == val) { go_left = true; break; } }
        } 
        
        int next = go_left ? tree->LeftChild(current) : tree->RightChild(current);
        if (next == current) break; // Prevent infinite loop
        current = next;
      } 
      
      // Accumulate at leaf
      if (current < vec_size) {
        double leaf_val = tree->LeafValue(current, 0);
        int back = current;
        int safety = 0;
        while (true) {
          if (back < vec_size) {
            node_sums[back] += leaf_val;
            node_counts[back] += 1.0;
          }
          if (back == 0) break; 
          back = tree->Parent(back);
          if (safety++ > 1000) break; // Infinite loop guard
        }
      }
    } 
    
    // PASS 2: Shapley
    double root_exp = (node_counts[0] > 0) ? (node_sums[0] / node_counts[0]) : 0.0;
    for (int i = 0; i < n; i++) {
      int current = 0;
      double curr_exp = root_exp;
      int safety = 0;
      
      while (!tree->IsLeaf(current)) {
        if (current >= vec_size) break;
        if (safety++ > 1000) break;
         
        int split_feat = tree->SplitIndex(current);
        if (split_feat < 0 || split_feat >= p) break;
         
        bool go_left = false;
        if (tree->IsNumericSplitNode(current)) {
          go_left = (X(i, split_feat) <= tree->Threshold(current));
        } else { 
          uint32_t val = static_cast<uint32_t>(X(i, split_feat));
          const std::vector<uint32_t>& cats = tree->CategoryList(current);
          for(auto c : cats) { if(c == val) { go_left = true; break; } }
        } 
        
        int next = go_left ? tree->LeftChild(current) : tree->RightChild(current);
        if (next >= vec_size) break; 
         
        double next_exp = curr_exp;
        if (node_counts[next] > 0) next_exp = node_sums[next] / node_counts[next];
        
        // Explicit read/write safe operation
        double current_shap = shapley_values(i, split_feat);
        shapley_values(i, split_feat) = current_shap + (next_exp - curr_exp);
        
        current = next;
        curr_exp = next_exp;
      }
    } 
  }
  return shapley_values;
} 