"""
Python classes wrapping C++ sampler objects
"""
import numpy as np
from .data import Dataset, Residual
from .forest import ForestContainer, Forest
from stochtree_cpp import RngCpp, ForestSamplerCpp, GlobalVarianceModelCpp, LeafVarianceModelCpp
from typing import Union

class RNG:
    """
    Wrapper around the C++ standard library random number generator. 
    Accepts an optional random seed at initialization for replicability.

    Parameters
    ----------
    random_seed : int, optional
        Random seed for replicability. If not specified, the default value of `-1` 
        triggers an initialization of the RNG based on 
        [std::random_device](https://en.cppreference.com/w/cpp/numeric/random/random_device).
    """
    def __init__(self, random_seed: int = -1) -> None:
        self.rng_cpp = RngCpp(random_seed)


class ForestSampler:
    """
    Wrapper around many of the core C++ sampling data structures and algorithms.

    Parameters
    ----------
    dataset : Dataset
        `stochtree` dataset object storing covariates / bases / weights
    feature_types : np.array
        Array of integer-coded values indicating the column type of each feature in `dataset`. 
        Integer codes map `0` to "numeric" (continuous), `1` to "ordered categorical, and `2` to 
        "unordered categorical".
    num_trees : int
        Number of trees in the forest model that this sampler class will fit.
    num_obs : int
        Number of observations / "rows" in `dataset`.
    alpha : float
        Prior probability of splitting for a tree of depth 0 in a forest model. Tree split prior combines `alpha` and `beta` via `alpha*(1+node_depth)^-beta`.
    beta : float
        Exponent that decreases split probabilities for nodes of depth > 0 in a forest model. Tree split prior combines `alpha` and `beta` via `alpha*(1+node_depth)^-beta`.
    min_samples_leaf : int
        Minimum allowable size of a leaf, in terms of training samples, in a forest model.
    max_depth : int, optional
        Maximum depth of any tree in the ensemble in a forest model.
    """
    def __init__(self, dataset: Dataset, feature_types: np.array, num_trees: int, num_obs: int, alpha: float, beta: float, min_samples_leaf: int, max_depth: int = -1) -> None:
        self.forest_sampler_cpp = ForestSamplerCpp(dataset.dataset_cpp, feature_types, num_trees, num_obs, alpha, beta, min_samples_leaf, max_depth)
    
    def reconstitute_from_forest(self, forest: Forest, dataset: Dataset, residual: Residual, is_mean_model: bool) -> None:
        """
        Re-initialize a forest sampler tracking data structures from a specific forest in a `ForestContainer`

        Parameters
        ----------
        dataset : Dataset
            `stochtree` dataset object storing covariates / bases / weights
        residual : Residual
            `stochtree` object storing continuously updated partial / full residual
        forest : Forest
            `stochtree` object storing tree ensemble
        is_mean_model : bool
            Indicator of whether the model being updated a conditional mean model (`True`) or a conditional variance model (`False`)
        """
        self.forest_sampler_cpp.ReconstituteTrackerFromForest(forest.forest_cpp, dataset.dataset_cpp, residual.residual_cpp, is_mean_model)
    
    def sample_one_iteration(self, forest_container: ForestContainer, forest: Forest, dataset: Dataset, 
                             residual: Residual, rng: RNG, feature_types: np.array, cutpoint_grid_size: int, 
                             leaf_model_scale_input: np.array, variable_weights: np.array, a_forest: float, b_forest: float, 
                             global_variance: float, leaf_model_int: int, keep_forest: bool, gfr: bool, pre_initialized: bool) -> None:
        """
        Sample one iteration of a forest using the specified model and tree sampling algorithm

        Parameters
        ----------
        forest_container : ForestContainer
            `stochtree` object storing tree ensembles
        forest : Forest
            `stochtree` object storing the "active" forest being sampled
        dataset : Dataset
            `stochtree` dataset object storing covariates / bases / weights
        residual : Residual
            `stochtree` object storing continuously updated partial / full residual
        rng : RNG
            `stochtree` object storing C++ random number generator to be used sampling algorithm
        feature_types : np.array
            Array of integer-coded feature types (0 = numeric, 1 = ordered categorical, 2 = unordered categorical)
        cutpoint_grid_size : int
            Maximum size of a grid of available cutpoints (which thins the number of possible splits, particularly useful in the grow-from-root algorithm)
        leaf_model_scale_input : np.array
            Numpy array containing leaf model scale parameter (if the leaf model is univariate, this is essentially a scalar which is used as such in the C++ source, but stored as a numpy array)
        variable_weights : np.array
            Numpy array containing sampling probabilities for each feature
        a_forest : float
            Shape parameter for the inverse gamma outcome model for a heteroskedasticity forest
        b_forest : float
            Scale parameter for the inverse gamma outcome model for a heteroskedasticity forest
        global_variance : float
            Current value of the global error variance parameter
        leaf_model_int : int
            Integer encoding the leaf model type (0 = constant Gaussian leaf mean model, 1 = univariate Gaussian leaf regression mean model, 2 = multivariate Gaussian leaf regression mean model, 3 = univariate Inverse Gamma constant leaf variance model)
        keep_forest : bool
            Whether or not the resulting forest should be retained in `forest_container` or discarded (due to burnin or thinning for example)
        gfr : bool
            Whether or not the "grow-from-root" (GFR) sampler is run (if this is `True` and `leaf_model_int=0` this is equivalent to XBART, if this is `FALSE` and `leaf_model_int=0` this is equivalent to the original BART)
        pre_initialized : bool
            Whether or not the forest being sampled has already been initialized
        """
        self.forest_sampler_cpp.SampleOneIteration(forest_container.forest_container_cpp, forest.forest_cpp, dataset.dataset_cpp, residual.residual_cpp, rng.rng_cpp, 
                                                   feature_types, cutpoint_grid_size, leaf_model_scale_input, variable_weights, 
                                                   a_forest, b_forest, global_variance, leaf_model_int, keep_forest, gfr, pre_initialized)
    
    def prepare_for_sampler(self, dataset: Dataset, residual: Residual, forest: Forest, leaf_model: int, initial_values: np.array) -> None:
        """
        Initialize forest and tracking data structures with constant root values before running a sampler

        Parameters
        ----------
        dataset : Dataset
            `stochtree` dataset object storing covariates / bases / weights
        residual : Residual
            `stochtree` object storing continuously updated partial / full residual
        forest : Forest
            `stochtree` object storing the "active" forest being sampled
        leaf_model : int
            Integer encoding the leaf model type
        initial_values : np.array
            Constant root node value(s) at which to initialize forest prediction (internally, it is divided by the number of trees and typically it is 0 for mean models and 1 for variance models).
        """
        self.forest_sampler_cpp.InitializeForestModel(dataset.dataset_cpp, residual.residual_cpp, forest.forest_cpp, leaf_model, initial_values)
    
    def adjust_residual(self, dataset: Dataset, residual: Residual, forest: Forest, requires_basis: bool, add: bool) -> None:
        """
        Method that "adjusts" the residual used for training tree ensembles by either adding or subtracting the prediction of each tree to the existing residual. 
        
        This is typically run just once at the beginning of a forest sampling algorithm --- after trees are initialized with constant root node predictions, their 
        root predictions are subtracted out of the residual.

        Parameters
        ----------
        dataset : Dataset
            `stochtree` dataset object storing covariates / bases / weights
        residual : Residual
            `stochtree` object storing continuously updated partial / full residual
        forest : Forest
            `stochtree` object storing the "active" forest being sampled
        requires_basis : bool
            Whether or not the forest requires a basis dot product when predicting
        add : bool
            Whether the predictions of each tree are added (if `add=True`) or subtracted (`add=False`) from the outcome to form the new residual
        """
        forest.forest_cpp.AdjustResidual(dataset.dataset_cpp, residual.residual_cpp, self.forest_sampler_cpp, requires_basis, add)
    
    def propagate_basis_update(self, dataset: Dataset, residual: Residual, forest: Forest) -> None:
        """
        Propagates basis update through to the (full/partial) residual by iteratively (a) adding back in the previous prediction of each tree, (b) recomputing predictions 
        for each tree (caching on the C++ side), (c) subtracting the new predictions from the residual.

        This is useful in cases where a basis (for e.g. leaf regression) is updated outside of a tree sampler (as with e.g. adaptive coding for binary treatment BCF). 
        Once a basis has been updated, the overall "function" represented by a tree model has changed and this should be reflected through to the residual before the 
        next sampling loop is run.

        Parameters
        ----------
        dataset : Dataset
            Stochtree dataset object storing covariates / bases / weights
        residual : Residual
            Stochtree object storing continuously updated partial / full residual
        forest : Forest
            Stochtree object storing the "active" forest being sampled
        """
        self.forest_sampler_cpp.PropagateBasisUpdate(dataset.dataset_cpp, residual.residual_cpp, forest.forest_cpp)
    
    def update_alpha(self, alpha: float) -> None:
        """
        Update `alpha` in the tree prior

        Parameters
        ----------
        alpha : float
            New value of `alpha` to be used
        """
        self.forest_sampler_cpp.UpdateAlpha(alpha)
    
    def update_beta(self, beta: float) -> None:
        """
        Update `beta` in the tree prior

        Parameters
        ----------
        beta : float
            New value of `beta` to be used
        """
        self.forest_sampler_cpp.UpdateBeta(beta)
    
    def update_min_samples_leaf(self, min_samples_leaf: int) -> None:
        """
        Update `min_samples_leaf` in the tree prior

        Parameters
        ----------
        min_samples_leaf : int
            New value of `min_samples_leaf` to be used
        """
        self.forest_sampler_cpp.UpdateMinSamplesLeaf(min_samples_leaf)
    
    def update_max_depth(self, max_depth: int) -> None:
        """
        Update `max_depth` in the tree prior

        Parameters
        ----------
        max_depth : int
            New value of `max_depth` to be used
        """
        self.forest_sampler_cpp.UpdateMaxDepth(max_depth)


class GlobalVarianceModel:
    """
    Wrapper around methods / functions for sampling a "global" error variance model 
    with [inverse gamma](https://en.wikipedia.org/wiki/Inverse-gamma_distribution) prior.
    """
    def __init__(self) -> None:
        self.variance_model_cpp = GlobalVarianceModelCpp()
    
    def sample_one_iteration(self, residual: Residual, rng: RNG, a: float, b: float) -> float:
        """
        Sample one iteration of a global error variance parameter

        Parameters
        ----------
        residual : Residual
            `stochtree` object storing continuously updated partial / full residual
        rng : RNG
            `stochtree` object storing C++ random number generator to be used sampling algorithm
        a : float
            Shape parameter for the inverse gamma error variance model
        b : float
            Scale parameter for the inverse gamma error variance model
        
        Returns
        -------
        float
            One draw from a Gibbs sampler for the error variance model, which depends 
            on the rest of the model only through the "full" residual stored in 
            a `Residual` object (net of predictions of any mean term such as a forest or 
            an additive parametric fixed / random effect term).
        """
        return self.variance_model_cpp.SampleOneIteration(residual.residual_cpp, rng.rng_cpp, a, b)


class LeafVarianceModel:
    """
    Wrapper around methods / functions for sampling a "leaf scale" model for the variance term of a Gaussian 
    leaf model with [inverse gamma](https://en.wikipedia.org/wiki/Inverse-gamma_distribution) prior.
    """
    def __init__(self) -> None:
        self.variance_model_cpp = LeafVarianceModelCpp()
    
    def sample_one_iteration(self, forest: Forest, rng: RNG, a: float, b: float) -> float:
        """
        Sample one iteration of a forest leaf model's variance parameter (assuming a location-scale leaf model, most commonly `N(0, tau)`)

        Parameters
        ----------
        forest : Forest
            `stochtree` object storing the "active" forest being sampled
        rng : RNG
            `stochtree` object storing C++ random number generator to be used sampling algorithm
        a : float
            Shape parameter for the inverse gamma leaf scale model
        b : float
            Scale parameter for the inverse gamma leaf scale model
        
        Returns
        -------
        float
            One draw from a Gibbs sampler for the leaf scale model, which depends 
            on the rest of the model only through its respective forest.
        """
        return self.variance_model_cpp.SampleOneIteration(forest.forest_cpp, rng.rng_cpp, a, b)
