"""
Bayesian Causal Forests (BCF) module
"""
import numpy as np
import pandas as pd
from sklearn.utils import check_scalar
from typing import Optional, Union, Dict, Any
from .bart import BARTModel
from .data import Dataset, Residual
from .forest import ForestContainer, Forest
from .preprocessing import CovariateTransformer, _preprocess_bcf_params
from .sampler import ForestSampler, RNG, GlobalVarianceModel, LeafVarianceModel
from .serialization import JSONSerializer
from .utils import NotSampledError

class BCFModel:
    r"""
    Class that handles sampling, storage, and serialization of stochastic forest models for causal effect estimation. 
    The class takes its name from Bayesian Causal Forests, an MCMC sampler originally developed in 
    Hahn, Murray, Carvalho (2020), but supports several sampling algorithms:

    * MCMC: The "classic" sampler defined in Hahn, Murray, Carvalho (2020). In order to run the MCMC sampler, 
    set `num_gfr = 0` (explained below) and then define a sampler according to several parameters:
        * `num_burnin`: the number of iterations to run before "retaining" samples for further analysis. These "burned in" samples 
        are helpful for allowing a sampler to converge before retaining samples.
        * `num_chains`: the number of independent sequences of MCMC samples to generate (typically referred to in the literature as "chains")
        * `num_mcmc`: the number of "retained" samples of the posterior distribution
        * `keep_every`: after a sampler has "burned in", we will run the sampler for `keep_every` * `num_mcmc` iterations, retaining one of each `keep_every` iteration in a chain.
    * GFR (Grow-From-Root): A fast, greedy approximation of the BART MCMC sampling algorithm introduced in Krantsevich, He, and Hahn (2023). GFR sampler iterations are 
    governed by the `num_gfr` parameter, and there are two primary ways to use this sampler:
        * Standalone: setting `num_gfr > 0` and both `num_burnin = 0` and `num_mcmc = 0` will only run and retain GFR samples of the posterior. This is typically referred to as "XBART" (accelerated BART).
        * Initializer for MCMC: setting `num_gfr > 0` and `num_mcmc > 0` will use ensembles from the GFR algorithm to initialize `num_chains` independent MCMC BART samplers, which are run for `num_mcmc` iterations. 
        This is typically referred to as "warm start BART".
    
    In addition to enabling multiple samplers, we support a broad set of models. First, note that the original BCF model of Hahn, Murray, Carvalho (2020) is

    \begin{equation*}
    \begin{aligned}
    y &= \mu(X) + b_z(X) + \epsilon\\
    b_z(X) &= (b_1 Z + b_0 (1-Z)) \tau(X)\\
    b_0, b_1 &\sim N(0, \frac{1}{2})\\\\
    \mu(X) &\sim \text{BART}(\cdot)\\
    \tau(X) &\sim \text{BART}(\cdot)\\
    \epsilon &\sim N(0, \sigma^2)\\
    \sigma^2 &\sim IG(\nu, \nu\lambda)
    \end{aligned}
    \end{equation*}

    for continuous outcome $y$, binary treatment $Z$, and covariates $X$.

    In words, there are two nonparametric mean functions -- a "prognostic" function and a treatment effect function -- governed by tree ensembles with BART priors and an additive (mean-zero) Gaussian error 
    term, whose variance is parameterized with an inverse gamma prior.

    The `BCFModel` class supports the following extensions of this model:
    
    - Continuous Treatment: If $Z$ is continuous rather than binary, we define $b_z(X) & = \tau(X, Z) = Z \tau(X)$, where the "leaf model" for the $\tau$ forest is essentially a regression on continuous $Z$.
    - Heteroskedasticity: Rather than define $\epsilon$ parameterically, we can let a forest $\sigma^2(X)$ model a conditional error variance function. This can be done by setting `num_trees_variance > 0` in the `params` dictionary passed to the `sample` method.
    """

    def __init__(self) -> None:
        # Internal flag for whether the sample() method has been run
        self.sampled = False
        self.rng = np.random.default_rng()
    
    def sample(self, X_train: Union[pd.DataFrame, np.array], Z_train: np.array, y_train: np.array, pi_train: np.array = None, 
               X_test: Union[pd.DataFrame, np.array] = None, Z_test: np.array = None, pi_test: np.array = None, 
               num_gfr: int = 5, num_burnin: int = 0, num_mcmc: int = 100, params: Optional[Dict[str, Any]] = None) -> None:
        """Runs a BCF sampler on provided training set. Outcome predictions and estimates of the prognostic and treatment effect functions 
        will be cached for the training set and (if provided) the test set.

        Parameters
        ----------
        X_train : np.array or pd.DataFrame
            Covariates used to split trees in the ensemble. Can be passed as either a matrix or dataframe.
        Z_train : np.array
            Array of (continuous or binary; univariate or multivariate) treatment assignments.
        y_train : np.array
            Outcome to be modeled by the ensemble.
        pi_train : np.array
            Optional vector of propensity scores. If not provided, this will be estimated from the data.
        X_test : np.array, optional
            Optional test set of covariates used to define "out of sample" evaluation data.
        Z_test : np.array, optional
            Optional test set of (continuous or binary) treatment assignments.
            Must be provided if `X_test` is provided.
        pi_test : np.array, optional
            Optional test set vector of propensity scores. If not provided (but `X_test` and `Z_test` are), this will be estimated from the data.
        num_gfr : int, optional
            Number of "warm-start" iterations run using the grow-from-root algorithm (He and Hahn, 2021). Defaults to `5`.
        num_burnin : int, optional
            Number of "burn-in" iterations of the MCMC sampler. Defaults to `0`. Ignored if `num_gfr > 0`.
        num_mcmc : int, optional
            Number of "retained" iterations of the MCMC sampler. Defaults to `100`. If this is set to 0, GFR (XBART) samples will be retained.
        params : dict, optional
            Dictionary of model parameters, each of which has a default value.

            * `cutpoint_grid_size` (`int`): Maximum number of cutpoints to consider for each feature. Defaults to `100`.
            * `sigma_leaf_mu` (`float`): Starting value of leaf node scale parameter for the prognostic forest. Calibrated internally as `2/num_trees_mu` if not set here.
            * `sigma_leaf_tau` (`float` or `np.array`): Starting value of leaf node scale parameter for the treatment effect forest. 
                When treatment (`Z_train`) is multivariate, this can be either a `float` or a square 2-dimensional `np.array` 
                with `sigma_leaf_tau.shape[0] == Z_train.shape[1]` and `sigma_leaf_tau.shape[1] == Z_train.shape[1]`.
                If `sigma_leaf_tau` is provided as a float for multivariate treatment, the leaf scale term will be set as a 
                diagonal matrix with `sigma_leaf_tau` on every diagonal. If not passed as an argument, this parameter is 
                calibrated internally as `1/num_trees_tau` (and propagated to a diagonal matrix if necessary).
            * `alpha_mu` (`float`): Prior probability of splitting for a tree of depth 0 for the prognostic forest. 
                Tree split prior combines `alpha` and `beta` via `alpha*(1+node_depth)^-beta`.
            * `alpha_tau` (`float`): Prior probability of splitting for a tree of depth 0 for the treatment effect forest. 
                Tree split prior combines `alpha` and `beta` via `alpha*(1+node_depth)^-beta`.
            * `alpha_variance` (`float`): Prior probability of splitting for a tree of depth 0 in the conditional variance model. Tree split prior combines `alpha_variance` and `beta_variance` via `alpha_variance*(1+node_depth)^-beta_variance`.
            * `beta_mu` (`float`): Exponent that decreases split probabilities for nodes of depth > 0 for the prognostic forest. 
                Tree split prior combines `alpha` and `beta` via `alpha*(1+node_depth)^-beta`.
            * `beta_tau` (`float`): Exponent that decreases split probabilities for nodes of depth > 0 for the treatment effect forest. 
                Tree split prior combines `alpha` and `beta` via `alpha*(1+node_depth)^-beta`.
            * `beta_variance` (`float`): Exponent that decreases split probabilities for nodes of depth > 0 in the conditional variance model. Tree split prior combines `alpha_variance` and `beta_variance` via `alpha_variance*(1+node_depth)^-beta_variance`.
            * `min_samples_leaf_mu` (`int`): Minimum allowable size of a leaf, in terms of training samples, for the prognostic forest. Defaults to `5`.
            * `min_samples_leaf_tau` (`int`): Minimum allowable size of a leaf, in terms of training samples, for the treatment effect forest. Defaults to `5`.
            * `min_samples_leaf_variance` (`int`): Minimum allowable size of a leaf, in terms of training samples in the conditional variance model. Defaults to `5`.
            * `max_depth_mu` (`int`): Maximum depth of any tree in the mu ensemble. Defaults to `10`. Can be overriden with `-1` which does not enforce any depth limits on trees.
            * `max_depth_tau` (`int`): Maximum depth of any tree in the tau ensemble. Defaults to `5`. Can be overriden with `-1` which does not enforce any depth limits on trees.
            * `max_depth_variance` (`int`): Maximum depth of any tree in the ensemble in the conditional variance model. Defaults to `10`. Can be overriden with `-1` which does not enforce any depth limits on trees.
            * `a_global` (`float`): Shape parameter in the `IG(a_global, b_global)` global error variance model. Defaults to `0`.
            * `b_global` (`float`): Component of the scale parameter in the `IG(a_global, b_global)` global error variance prior. Defaults to `0`.
            * `a_leaf_mu` (`float`): Shape parameter in the `IG(a_leaf, b_leaf)` leaf node parameter variance model for the prognostic forest. Defaults to `3`.
            * `a_leaf_tau` (`float`): Shape parameter in the `IG(a_leaf, b_leaf)` leaf node parameter variance model for the treatment effect forest. Defaults to `3`.
            * `b_leaf_mu` (`float`): Scale parameter in the `IG(a_leaf, b_leaf)` leaf node parameter variance model for the prognostic forest. Calibrated internally as `0.5/num_trees` if not set here.
            * `b_leaf_tau` (`float`): Scale parameter in the `IG(a_leaf, b_leaf)` leaf node parameter variance model for the treatment effect forest. Calibrated internally as `0.5/num_trees` if not set here.
            * `sigma2_init` (`float`): Starting value of global variance parameter. Calibrated internally as in Sparapani et al (2021) if not set here.
            * `variance_forest_leaf_init` (`float`): Starting value of root forest prediction in conditional (heteroskedastic) error variance model. Calibrated internally as `np.log(pct_var_variance_forest_init*np.var((y-np.mean(y))/np.std(y)))/num_trees_variance` if not set.
            * `pct_var_sigma2_init` (`float`): Percentage of standardized outcome variance used to initialize global error variance parameter. Superseded by `sigma2`. Defaults to `0.25`.
            * `pct_var_variance_forest_init` (`float`): Percentage of standardized outcome variance used to initialize global error variance parameter. Default: `1`. Superseded by `variance_forest_init`.
            * `variable_weights_mean` (`np.`array`): Numeric weights reflecting the relative probability of splitting on each variable in the prognostic and treatment effect forests. Does not need to sum to 1 but cannot be negative. Defaults to `np.repeat(1/X_train.shape[1], X_train.shape[1])` if not set here. Note that if the propensity score is included as a covariate in either forest, its weight will default to `1/X_train.shape[1]`. A workaround if you wish to provide a custom weight for the propensity score is to include it as a column in `X_train` and then set `propensity_covariate` to `'none'` and adjust `keep_vars_mu` and `keep_vars_tau` accordingly.
            * `variable_weights_variance` (`np.array`): Numeric weights reflecting the relative probability of splitting on each variable in the variance forest. Does not need to sum to 1 but cannot be negative. Defaults to uniform over the columns of `X_train` if not provided.
            * `keep_vars_mu` (`list` or `np.array`): Vector of variable names or column indices denoting variables that should be included in the prognostic (`mu(X)`) forest. Defaults to `None`.
            * `drop_vars_mu` (`list` or `np.array`): Vector of variable names or column indices denoting variables that should be excluded from the prognostic (`mu(X)`) forest. Defaults to `None`. If both `drop_vars_mu` and `keep_vars_mu` are set, `drop_vars_mu` will be ignored.
            * `drop_vars_variance` (`list` or `np.array`): Vector of variable names or column indices denoting variables that should be excluded from the variance (`sigma^2(X)`) forest. Defaults to `None`. If both `drop_vars_variance` and `keep_vars_variance` are set, `drop_vars_variance` will be ignored.
            * `keep_vars_tau` (`list` or `np.array`): Vector of variable names or column indices denoting variables that should be included in the treatment effect (`tau(X)`) forest. Defaults to `None`.
            * `drop_vars_tau` (`list` or `np.array`): Vector of variable names or column indices denoting variables that should be excluded from the treatment effect (`tau(X)`) forest. Defaults to `None`. If both `drop_vars_tau` and `keep_vars_tau` are set, `drop_vars_tau` will be ignored.
            * `drop_vars_variance` (`list` or `np.array`): Vector of variable names or column indices denoting variables that should be excluded from the variance (`sigma^2(X)`) forest. Defaults to `None`. If both `drop_vars_variance` and `keep_vars_variance` are set, `drop_vars_variance` will be ignored.
            * `keep_vars_variance` (`list` or `np.array`): Vector of variable names or column indices denoting variables that should be included in the variance (`sigma^2(X)`) forest. Defaults to `None`.
            * `num_trees_mu` (`int`): Number of trees in the prognostic forest. Defaults to `200`.
            * `num_trees_tau` (`int`): Number of trees in the treatment effect forest. Defaults to `50`.
            * `num_trees_variance` (`int`): Number of trees in the ensemble for the conditional variance model. Defaults to `0`. Variance is only modeled using a tree / forest if `num_trees_variance > 0`.
            * `sample_sigma_global` (`bool`): Whether or not to update the `sigma^2` global error variance parameter based on `IG(a_global, b_global)`. Defaults to `True`.
            * `sample_sigma_leaf_mu` (`bool`): Whether or not to update the `tau` leaf scale variance parameter based on `IG(a_leaf, b_leaf)` for the prognostic forest. 
                Cannot (currently) be set to true if `basis_train` has more than one column. Defaults to `True`.
            * `sample_sigma_leaf_tau` (`bool`): Whether or not to update the `tau` leaf scale variance parameter based on `IG(a_leaf, b_leaf)` for the treatment effect forest. 
                Cannot (currently) be set to true if `basis_train` has more than one column. Defaults to `True`.
            * `propensity_covariate` (`str`): Whether to include the propensity score as a covariate in either or both of the forests. Enter `"none"` for neither, `"mu"` for the prognostic forest, `"tau"` for the treatment forest, and `"both"` for both forests. 
                If this is not `"none"` and a propensity score is not provided, it will be estimated from (`X_train`, `Z_train`) using `BARTModel`. Defaults to `"mu"`.
            * `adaptive_coding` (`bool`): Whether or not to use an "adaptive coding" scheme in which a binary treatment variable is not coded manually as (0,1) or (-1,1) but learned via 
                parameters `b_0` and `b_1` that attach to the outcome model `[b_0 (1-Z) + b_1 Z] tau(X)`. This is ignored when Z is not binary. Defaults to True.
            * `b_0` (`float`): Initial value of the "control" group coding parameter. This is ignored when `Z` is not binary. Default: `-0.5`.
            * `b_1` (`float`): Initial value of the "treated" group coding parameter. This is ignored when `Z` is not binary. Default: `0.5`.
            * `random_seed` (`int`): Integer parameterizing the C++ random number generator. If not specified, the C++ random number generator is seeded according to `std::random_device`.
            * `keep_burnin` (`bool`): Whether or not "burnin" samples should be included in predictions. Defaults to `False`. Ignored if `num_mcmc == 0`.
            * `keep_gfr` (`bool`): Whether or not "warm-start" / grow-from-root samples should be included in predictions. Defaults to `False`. Ignored if `num_mcmc == 0`.
            * `keep_every` (`int`): How many iterations of the burned-in MCMC sampler should be run before forests and parameters are retained. Defaults to `1`. Setting `keep_every = k` for some `k > 1` will "thin" the MCMC samples by retaining every `k`-th sample, rather than simply every sample. This can reduce the autocorrelation of the MCMC samples.
        """
        # Unpack parameters
        bcf_params = _preprocess_bcf_params(params)
        cutpoint_grid_size = bcf_params['cutpoint_grid_size']
        sigma_leaf_mu = bcf_params['sigma_leaf_mu']
        sigma_leaf_tau = bcf_params['sigma_leaf_tau']
        alpha_mu = bcf_params['alpha_mu']
        alpha_tau = bcf_params['alpha_tau']
        alpha_variance = bcf_params['alpha_variance']
        beta_mu = bcf_params['beta_mu']
        beta_tau = bcf_params['beta_tau']
        beta_variance = bcf_params['beta_variance']
        min_samples_leaf_mu = bcf_params['min_samples_leaf_mu']
        min_samples_leaf_tau = bcf_params['min_samples_leaf_tau']
        min_samples_leaf_variance = bcf_params['min_samples_leaf_variance']
        max_depth_mu = bcf_params['max_depth_mu']
        max_depth_tau = bcf_params['max_depth_tau']
        max_depth_variance = bcf_params['max_depth_variance']
        a_global = bcf_params['a_global']
        b_global = bcf_params['b_global']
        a_forest = bcf_params['a_forest']
        b_forest = bcf_params['b_forest']
        a_leaf_mu = bcf_params['a_leaf_mu']
        a_leaf_tau = bcf_params['a_leaf_tau']
        b_leaf_mu = bcf_params['b_leaf_mu']
        b_leaf_tau = bcf_params['b_leaf_tau']
        sigma2_init = bcf_params['sigma2_init']
        variance_forest_leaf_init = bcf_params['variance_forest_leaf_init']
        pct_var_sigma2_init = bcf_params['pct_var_sigma2_init']
        pct_var_variance_forest_init = bcf_params['pct_var_variance_forest_init']
        variable_weights_mu = bcf_params['variable_weights_mu']
        variable_weights_tau = bcf_params['variable_weights_tau']
        variable_weights_variance = bcf_params['variable_weights_variance']
        keep_vars_mu = bcf_params['keep_vars_mu']
        drop_vars_mu = bcf_params['drop_vars_mu']
        keep_vars_tau = bcf_params['keep_vars_tau']
        drop_vars_tau = bcf_params['drop_vars_tau']
        keep_vars_variance = bcf_params['keep_vars_variance']
        drop_vars_variance = bcf_params['drop_vars_variance']
        num_trees_mu = bcf_params['num_trees_mu']
        num_trees_tau = bcf_params['num_trees_tau']
        num_trees_variance = bcf_params['num_trees_variance']
        sample_sigma_global = bcf_params['sample_sigma_global']
        sample_sigma_leaf_mu = bcf_params['sample_sigma_leaf_mu']
        sample_sigma_leaf_tau = bcf_params['sample_sigma_leaf_tau']
        propensity_covariate = bcf_params['propensity_covariate']
        adaptive_coding = bcf_params['adaptive_coding']
        b_0 = bcf_params['b_0']
        b_1 = bcf_params['b_1']
        random_seed = bcf_params['random_seed']
        keep_burnin = bcf_params['keep_burnin']
        keep_gfr = bcf_params['keep_gfr']
        self.standardize = bcf_params['standardize']
        keep_every = bcf_params['keep_every']
        
        # Variable weight preprocessing (and initialization if necessary)
        if variable_weights_mu is None:
            if X_train.ndim > 1:
                variable_weights_mu = np.repeat(1/X_train.shape[1], X_train.shape[1])
            else:
                variable_weights_mu = np.repeat(1., 1)
        if np.any(variable_weights_mu < 0):
            raise ValueError("variable_weights_mu cannot have any negative weights")
        if variable_weights_tau is None:
            if X_train.ndim > 1:
                variable_weights_tau = np.repeat(1/X_train.shape[1], X_train.shape[1])
            else:
                variable_weights_tau = np.repeat(1., 1)
        if np.any(variable_weights_tau < 0):
            raise ValueError("variable_weights_tau cannot have any negative weights")
        if variable_weights_variance is None:
            if X_train.ndim > 1:
                variable_weights_variance = np.repeat(1/X_train.shape[1], X_train.shape[1])
            else:
                variable_weights_variance = np.repeat(1., 1)
        if np.any(variable_weights_variance < 0):
            raise ValueError("variable_weights_variance cannot have any negative weights")
        
        # Determine whether conditional variance model will be fit
        self.include_variance_forest = True if num_trees_variance > 0 else False
        
        # Check data inputs
        if not isinstance(X_train, pd.DataFrame) and not isinstance(X_train, np.ndarray):
            raise ValueError("X_train must be a pandas dataframe or numpy array")
        if not isinstance(Z_train, np.ndarray):
            raise ValueError("Z_train must be a numpy array")
        if pi_train is not None:
            if not isinstance(pi_train, np.ndarray):
                raise ValueError("pi_train must be a numpy array")
        if not isinstance(y_train, np.ndarray):
            raise ValueError("y_train must be a numpy array")
        if X_test is not None:
            if not isinstance(X_test, pd.DataFrame) and not isinstance(X_test, np.ndarray):
                raise ValueError("X_test must be a pandas dataframe or numpy array")
        if Z_test is not None:
            if not isinstance(Z_test, np.ndarray):
                raise ValueError("Z_test must be a numpy array")
        if pi_test is not None:
            if not isinstance(pi_test, np.ndarray):
                raise ValueError("pi_test must be a numpy array")
        
        # Convert everything to standard shape (2-dimensional)
        if isinstance(X_train, np.ndarray):
            if X_train.ndim == 1:
                X_train = np.expand_dims(X_train, 1)
        if Z_train is not None:
            if Z_train.ndim == 1:
                Z_train = np.expand_dims(Z_train, 1)
        if pi_train is not None:
            if pi_train.ndim == 1:
                pi_train = np.expand_dims(pi_train, 1)
        if y_train.ndim == 1:
            y_train = np.expand_dims(y_train, 1)
        if X_test is not None:
            if isinstance(X_test, np.ndarray):
                if X_test.ndim == 1:
                    X_test = np.expand_dims(X_test, 1)
        if Z_test is not None:
            if Z_test.ndim == 1:
                Z_test = np.expand_dims(Z_test, 1)
        if pi_test is not None:
            if pi_test.ndim == 1:
                pi_test = np.expand_dims(pi_test, 1)

        # Original number of covariates
        num_cov_orig = X_train.shape[1]
        
        # Data checks
        if X_test is not None:
            if X_test.shape[1] != X_train.shape[1]:
                raise ValueError("X_train and X_test must have the same number of columns")
        if Z_test is not None:
            if Z_test.shape[1] != Z_train.shape[1]:
                raise ValueError("Z_train and Z_test must have the same number of columns")
        if Z_train.shape[0] != X_train.shape[0]:
            raise ValueError("X_train and Z_train must have the same number of rows")
        if y_train.shape[0] != X_train.shape[0]:
            raise ValueError("X_train and y_train must have the same number of rows")
        if pi_train is not None:
            if pi_train.shape[0] != X_train.shape[0]:
                raise ValueError("X_train and pi_train must have the same number of rows")
        if X_test is not None and Z_test is not None:
            if X_test.shape[0] != Z_test.shape[0]:
                raise ValueError("X_test and Z_test must have the same number of rows")
        if X_test is not None and pi_test is not None:
            if X_test.shape[0] != pi_test.shape[0]:
                raise ValueError("X_test and pi_test must have the same number of rows")
        
        # Treatment details
        self.treatment_dim = Z_train.shape[1]
        self.multivariate_treatment = True if self.treatment_dim > 1 else False
        treatment_leaf_model = 2 if self.multivariate_treatment else 1
        
        # Set variance leaf model type (currently only one option)
        leaf_model_variance_forest = 3
        self.variance_scale = 1
        
        # Check parameters
        if sigma_leaf_tau is not None:
            if not isinstance(sigma_leaf_tau, float) and not isinstance(sigma_leaf_tau, np.ndarray):
                raise ValueError("sigma_leaf_tau must be a float or numpy array")
            if self.multivariate_treatment:
                if sigma_leaf_tau is not None:
                    if isinstance(sigma_leaf_tau, np.ndarray):
                        if sigma_leaf_tau.ndim != 2:
                            raise ValueError("sigma_leaf_tau must be 2-dimensional if passed as a np.array")
                        if self.treatment_dim != sigma_leaf_tau.shape[0] or self.treatment_dim != sigma_leaf_tau.shape[1]:
                            raise ValueError("sigma_leaf_tau must have the same number of rows and columns, which must match Z_train.shape[1]")
        if sigma_leaf_mu is not None:
            sigma_leaf_mu = check_scalar(x=sigma_leaf_mu, name="sigma_leaf_mu", target_type=float, 
                                         min_val=0., max_val=None, include_boundaries="neither")
        if cutpoint_grid_size is not None:
            cutpoint_grid_size = check_scalar(x=cutpoint_grid_size, name="cutpoint_grid_size", target_type=int, 
                                              min_val=1, max_val=None, include_boundaries="left")
        if min_samples_leaf_mu is not None:
            min_samples_leaf_mu = check_scalar(x=min_samples_leaf_mu, name="min_samples_leaf_mu", target_type=int, 
                                               min_val=1, max_val=None, include_boundaries="left")
        if min_samples_leaf_tau is not None:
            min_samples_leaf_tau = check_scalar(x=min_samples_leaf_tau, name="min_samples_leaf_tau", target_type=int, 
                                                min_val=1, max_val=None, include_boundaries="left")
        if num_trees_mu is not None:
            num_trees_mu = check_scalar(x=num_trees_mu, name="num_trees_mu", target_type=int, 
                                        min_val=1, max_val=None, include_boundaries="left")
        if num_trees_tau is not None:
            num_trees_tau = check_scalar(x=num_trees_tau, name="num_trees_tau", target_type=int, 
                                         min_val=1, max_val=None, include_boundaries="left")
        num_gfr = check_scalar(x=num_gfr, name="num_gfr", target_type=int, 
                                min_val=0, max_val=None, include_boundaries="left")
        num_burnin = check_scalar(x=num_burnin, name="num_burnin", target_type=int, 
                                min_val=0, max_val=None, include_boundaries="left")
        num_mcmc = check_scalar(x=num_mcmc, name="num_mcmc", target_type=int, 
                                min_val=0, max_val=None, include_boundaries="left")
        num_samples = num_gfr + num_burnin + num_mcmc
        num_samples = check_scalar(x=num_samples, name="num_samples", target_type=int, 
                                   min_val=1, max_val=None, include_boundaries="left")
        if random_seed is not None:
            random_seed = check_scalar(x=random_seed, name="random_seed", target_type=int, 
                                       min_val=-1, max_val=None, include_boundaries="left")
        if alpha_mu is not None:
            alpha_mu = check_scalar(x=alpha_mu, name="alpha_mu", target_type=(float,int), 
                                    min_val=0, max_val=1, include_boundaries="neither")
        if alpha_tau is not None:
            alpha_tau = check_scalar(x=alpha_tau, name="alpha_tau", target_type=(float,int), 
                                     min_val=0, max_val=1, include_boundaries="neither")
        if beta_mu is not None:
            beta_mu = check_scalar(x=beta_mu, name="beta_mu", target_type=(float,int), 
                                   min_val=1, max_val=None, include_boundaries="left")
        if beta_tau is not None:
            beta_tau = check_scalar(x=beta_tau, name="beta_tau", target_type=(float,int), 
                                    min_val=1, max_val=None, include_boundaries="left")
        if a_global is not None:
            a_global = check_scalar(x=a_global, name="a_global", target_type=(float,int), 
                                    min_val=0, max_val=None, include_boundaries="left")
        if b_global is not None:
            b_global = check_scalar(x=b_global, name="b_global", target_type=(float,int), 
                                    min_val=0, max_val=None, include_boundaries="left")
        if a_leaf_mu is not None:
            a_leaf_mu = check_scalar(x=a_leaf_mu, name="a_leaf_mu", target_type=(float,int), 
                                     min_val=0, max_val=None, include_boundaries="left")
        if a_leaf_tau is not None:
            a_leaf_tau = check_scalar(x=a_leaf_tau, name="a_leaf_tau", target_type=(float,int), 
                                      min_val=0, max_val=None, include_boundaries="left")
        if b_leaf_mu is not None:
            b_leaf_mu = check_scalar(x=b_leaf_mu, name="b_leaf_mu", target_type=(float,int), 
                                     min_val=0, max_val=None, include_boundaries="left")
        if b_leaf_tau is not None:
            b_leaf_tau = check_scalar(x=b_leaf_tau, name="b_leaf_tau", target_type=(float,int), 
                                      min_val=0, max_val=None, include_boundaries="left")
        if sigma2_init is not None:
            sigma2_init = check_scalar(x=sigma2_init, name="sigma2_init", target_type=(float,int), 
                                       min_val=0, max_val=None, include_boundaries="neither")
        if sample_sigma_leaf_mu is not None:
            if not isinstance(sample_sigma_leaf_mu, bool):
                raise ValueError("sample_sigma_leaf_mu must be a bool")
        if sample_sigma_leaf_tau is not None:
            if not isinstance(sample_sigma_leaf_tau, bool):
                raise ValueError("sample_sigma_leaf_tau must be a bool")
        if propensity_covariate is not None:
            if propensity_covariate not in ["mu", "tau", "both", "none"]:
                raise ValueError("propensity_covariate must be one of 'mu', 'tau', 'both', or 'none'")
        if b_0 is not None:
            b_0 = check_scalar(x=b_0, name="b_0", target_type=(float,int), 
                            min_val=None, max_val=None, include_boundaries="neither")
        if b_1 is not None:
            b_1 = check_scalar(x=b_1, name="b_1", target_type=(float,int), 
                            min_val=None, max_val=None, include_boundaries="neither")
        if keep_burnin is not None:
            if not isinstance(keep_burnin, bool):
                raise ValueError("keep_burnin must be a bool")
        if keep_gfr is not None:
            if not isinstance(keep_gfr, bool):
                raise ValueError("keep_gfr must be a bool")
        
        # Standardize the keep variable lists to numeric indices
        if keep_vars_mu is not None:
            if isinstance(keep_vars_mu, list):
                if all(isinstance(i, str) for i in keep_vars_mu):
                    if not np.all(np.isin(keep_vars_mu, X_train.columns)):
                        raise ValueError("keep_vars_mu includes some variable names that are not in X_train")
                    variable_subset_mu = [i for i in X_train.shape[1] if keep_vars_mu.count(X_train.columns.array[i]) > 0]
                elif all(isinstance(i, int) for i in keep_vars_mu):
                    if any(i >= X_train.shape[1] for i in keep_vars_mu):
                        raise ValueError("keep_vars_mu includes some variable indices that exceed the number of columns in X_train")
                    if any(i < 0 for i in keep_vars_mu):
                        raise ValueError("keep_vars_mu includes some negative variable indices")
                    variable_subset_mu = keep_vars_mu
                else:
                    raise ValueError("keep_vars_mu must be a list of variable names (str) or column indices (int)")
            elif isinstance(keep_vars_mu, np.ndarray):
                if keep_vars_mu.dtype == np.str_:
                    if not np.all(np.isin(keep_vars_mu, X_train.columns)):
                        raise ValueError("keep_vars_mu includes some variable names that are not in X_train")
                    variable_subset_mu = [i for i in X_train.shape[1] if keep_vars_mu.count(X_train.columns.array[i]) > 0]
                else:
                    if np.any(keep_vars_mu >= X_train.shape[1]):
                        raise ValueError("keep_vars_mu includes some variable indices that exceed the number of columns in X_train")
                    if np.any(keep_vars_mu < 0):
                        raise ValueError("keep_vars_mu includes some negative variable indices")
                    variable_subset_mu = [i for i in keep_vars_mu]
            else:
                raise ValueError("keep_vars_mu must be a list or np.array")
        elif keep_vars_mu is None and drop_vars_mu is not None:
            if isinstance(drop_vars_mu, list):
                if all(isinstance(i, str) for i in drop_vars_mu):
                    if not np.all(np.isin(drop_vars_mu, X_train.columns)):
                        raise ValueError("drop_vars_mu includes some variable names that are not in X_train")
                    variable_subset_mu = [i for i in range(X_train.shape[1]) if drop_vars_mu.count(X_train.columns.array[i]) == 0]
                elif all(isinstance(i, int) for i in drop_vars_mu):
                    if any(i >= X_train.shape[1] for i in drop_vars_mu):
                        raise ValueError("drop_vars_mu includes some variable indices that exceed the number of columns in X_train")
                    if any(i < 0 for i in drop_vars_mu):
                        raise ValueError("drop_vars_mu includes some negative variable indices")
                    variable_subset_mu = [i for i in range(X_train.shape[1]) if drop_vars_mu.count(i) == 0]
                else:
                    raise ValueError("drop_vars_mu must be a list of variable names (str) or column indices (int)")
            elif isinstance(drop_vars_mu, np.ndarray):
                if drop_vars_mu.dtype == np.str_:
                    if not np.all(np.isin(drop_vars_mu, X_train.columns)):
                        raise ValueError("drop_vars_mu includes some variable names that are not in X_train")
                    keep_inds = ~np.isin(X_train.columns.array, drop_vars_mu)
                    variable_subset_mu = [i for i in keep_inds]
                else:
                    if np.any(drop_vars_mu >= X_train.shape[1]):
                        raise ValueError("drop_vars_mu includes some variable indices that exceed the number of columns in X_train")
                    if np.any(drop_vars_mu < 0):
                        raise ValueError("drop_vars_mu includes some negative variable indices")
                    keep_inds = ~np.isin(np.arange(X_train.shape[1]), drop_vars_mu)
                    variable_subset_mu = [i for i in keep_inds]
            else:
                raise ValueError("drop_vars_mu must be a list or np.array")
        else:
            variable_subset_mu = [i for i in range(X_train.shape[1])]
        if keep_vars_tau is not None:
            if isinstance(keep_vars_tau, list):
                if all(isinstance(i, str) for i in keep_vars_tau):
                    if not np.all(np.isin(keep_vars_tau, X_train.columns)):
                        raise ValueError("keep_vars_tau includes some variable names that are not in X_train")
                    variable_subset_tau = [i for i in range(X_train.shape[1]) if keep_vars_tau.count(X_train.columns.array[i]) > 0]
                elif all(isinstance(i, int) for i in keep_vars_tau):
                    if any(i >= X_train.shape[1] for i in keep_vars_tau):
                        raise ValueError("keep_vars_tau includes some variable indices that exceed the number of columns in X_train")
                    if any(i < 0 for i in keep_vars_tau):
                        raise ValueError("keep_vars_tau includes some negative variable indices")
                    variable_subset_tau = keep_vars_tau
                else:
                    raise ValueError("keep_vars_tau must be a list of variable names (str) or column indices (int)")
            elif isinstance(keep_vars_tau, np.ndarray):
                if keep_vars_tau.dtype == np.str_:
                    if not np.all(np.isin(keep_vars_tau, X_train.columns)):
                        raise ValueError("keep_vars_tau includes some variable names that are not in X_train")
                    variable_subset_tau = [i for i in range(X_train.shape[1]) if keep_vars_tau.count(X_train.columns.array[i]) > 0]
                else:
                    if np.any(keep_vars_tau >= X_train.shape[1]):
                        raise ValueError("keep_vars_tau includes some variable indices that exceed the number of columns in X_train")
                    if np.any(keep_vars_tau < 0):
                        raise ValueError("keep_vars_tau includes some negative variable indices")
                    variable_subset_tau = [i for i in keep_vars_tau]
            else:
                raise ValueError("keep_vars_tau must be a list or np.array")
        elif keep_vars_tau is None and drop_vars_tau is not None:
            if isinstance(drop_vars_tau, list):
                if all(isinstance(i, str) for i in drop_vars_tau):
                    if not np.all(np.isin(drop_vars_tau, X_train.columns)):
                        raise ValueError("drop_vars_tau includes some variable names that are not in X_train")
                    variable_subset_tau = [i for i in range(X_train.shape[1]) if drop_vars_tau.count(X_train.columns.array[i]) == 0]
                elif all(isinstance(i, int) for i in drop_vars_tau):
                    if any(i >= X_train.shape[1] for i in drop_vars_tau):
                        raise ValueError("drop_vars_tau includes some variable indices that exceed the number of columns in X_train")
                    if any(i < 0 for i in drop_vars_tau):
                        raise ValueError("drop_vars_tau includes some negative variable indices")
                    variable_subset_tau = [i for i in range(X_train.shape[1]) if drop_vars_tau.count(i) == 0]
                else:
                    raise ValueError("drop_vars_tau must be a list of variable names (str) or column indices (int)")
            elif isinstance(drop_vars_tau, np.ndarray):
                if drop_vars_tau.dtype == np.str_:
                    if not np.all(np.isin(drop_vars_tau, X_train.columns)):
                        raise ValueError("drop_vars_tau includes some variable names that are not in X_train")
                    keep_inds = ~np.isin(X_train.columns.array, drop_vars_tau)
                    variable_subset_tau = [i for i in keep_inds]
                else:
                    if np.any(drop_vars_tau >= X_train.shape[1]):
                        raise ValueError("drop_vars_tau includes some variable indices that exceed the number of columns in X_train")
                    if np.any(drop_vars_tau < 0):
                        raise ValueError("drop_vars_tau includes some negative variable indices")
                    keep_inds = ~np.isin(np.arange(X_train.shape[1]), drop_vars_tau)
                    variable_subset_tau = [i for i in keep_inds]
            else:
                raise ValueError("drop_vars_tau must be a list or np.array")
        else:
            variable_subset_tau = [i for i in range(X_train.shape[1])]
        if keep_vars_variance is not None:
            if isinstance(keep_vars_variance, list):
                if all(isinstance(i, str) for i in keep_vars_variance):
                    if not np.all(np.isin(keep_vars_variance, X_train.columns)):
                        raise ValueError("keep_vars_variance includes some variable names that are not in X_train")
                    variable_subset_variance = [i for i in range(X_train.shape[1]) if keep_vars_variance.count(X_train.columns.array[i]) > 0]
                elif all(isinstance(i, int) for i in keep_vars_variance):
                    if any(i >= X_train.shape[1] for i in keep_vars_variance):
                        raise ValueError("keep_vars_variance includes some variable indices that exceed the number of columns in X_train")
                    if any(i < 0 for i in keep_vars_variance):
                        raise ValueError("keep_vars_variance includes some negative variable indices")
                    variable_subset_variance = keep_vars_variance
                else:
                    raise ValueError("keep_vars_variance must be a list of variable names (str) or column indices (int)")
            elif isinstance(keep_vars_variance, np.ndarray):
                if keep_vars_variance.dtype == np.str_:
                    if not np.all(np.isin(keep_vars_variance, X_train.columns)):
                        raise ValueError("keep_vars_variance includes some variable names that are not in X_train")
                    variable_subset_variance = [i for i in range(X_train.shape[1]) if keep_vars_variance.count(X_train.columns.array[i]) > 0]
                else:
                    if np.any(keep_vars_variance >= X_train.shape[1]):
                        raise ValueError("keep_vars_variance includes some variable indices that exceed the number of columns in X_train")
                    if np.any(keep_vars_variance < 0):
                        raise ValueError("keep_vars_variance includes some negative variable indices")
                    variable_subset_variance = [i for i in keep_vars_variance]
            else:
                raise ValueError("keep_vars_variance must be a list or np.array")
        elif keep_vars_variance is None and drop_vars_variance is not None:
            if isinstance(drop_vars_variance, list):
                if all(isinstance(i, str) for i in drop_vars_variance):
                    if not np.all(np.isin(drop_vars_variance, X_train.columns)):
                        raise ValueError("drop_vars_variance includes some variable names that are not in X_train")
                    variable_subset_variance = [i for i in range(X_train.shape[1]) if drop_vars_variance.count(X_train.columns.array[i]) == 0]
                elif all(isinstance(i, int) for i in drop_vars_variance):
                    if any(i >= X_train.shape[1] for i in drop_vars_variance):
                        raise ValueError("drop_vars_variance includes some variable indices that exceed the number of columns in X_train")
                    if any(i < 0 for i in drop_vars_variance):
                        raise ValueError("drop_vars_variance includes some negative variable indices")
                    variable_subset_variance = [i for i in range(X_train.shape[1]) if drop_vars_variance.count(i) == 0]
                else:
                    raise ValueError("drop_vars_variance must be a list of variable names (str) or column indices (int)")
            elif isinstance(drop_vars_variance, np.ndarray):
                if drop_vars_variance.dtype == np.str_:
                    if not np.all(np.isin(drop_vars_variance, X_train.columns)):
                        raise ValueError("drop_vars_variance includes some variable names that are not in X_train")
                    keep_inds = ~np.isin(X_train.columns.array, drop_vars_variance)
                    variable_subset_variance = [i for i in keep_inds]
                else:
                    if np.any(drop_vars_variance >= X_train.shape[1]):
                        raise ValueError("drop_vars_variance includes some variable indices that exceed the number of columns in X_train")
                    if np.any(drop_vars_variance < 0):
                        raise ValueError("drop_vars_variance includes some negative variable indices")
                    keep_inds = ~np.isin(np.arange(X_train.shape[1]), drop_vars_variance)
                    variable_subset_variance = [i for i in keep_inds]
            else:
                raise ValueError("drop_vars_variance must be a list or np.array")
        else:
            variable_subset_variance = [i for i in range(X_train.shape[1])]
        
        # Covariate preprocessing
        self._covariate_transformer = CovariateTransformer()
        self._covariate_transformer.fit(X_train)
        X_train_processed = self._covariate_transformer.transform(X_train)
        if X_test is not None:
            X_test_processed = self._covariate_transformer.transform(X_test)
        feature_types = np.asarray(self._covariate_transformer._processed_feature_types)
        original_var_indices = self._covariate_transformer.fetch_original_feature_indices()

        # Determine whether a test set is provided
        self.has_test = X_test is not None
        
        # Unpack data dimensions
        self.n_train = y_train.shape[0]
        self.n_test = X_test.shape[0] if self.has_test else 0
        self.p_x = X_train_processed.shape[1]

        # Check whether treatment is binary
        self.binary_treatment = np.unique(Z_train).size == 2

        # Adaptive coding will be ignored for continuous / ordered categorical treatments
        self.adaptive_coding = adaptive_coding
        if adaptive_coding and not self.binary_treatment:
            self.adaptive_coding = False
        if adaptive_coding and self.multivariate_treatment:
            self.adaptive_coding = False

        # Sampling sigma_leaf_tau will be ignored for multivariate treatments
        if sample_sigma_leaf_tau and self.multivariate_treatment:
            sample_sigma_leaf_tau = False

        # Check if user has provided propensities that are needed in the model
        if pi_train is None and propensity_covariate != "none":
            if self.multivariate_treatment:
                raise ValueError("Propensities must be provided (via pi_train and / or pi_test parameters) or omitted by setting propensity_covariate = 'none' for multivariate treatments")
            else:
                self.bart_propensity_model = BARTModel()
                if self.has_test:
                    self.bart_propensity_model.sample(X_train=X_train_processed, y_train=Z_train, X_test=X_test_processed, num_gfr=10, num_mcmc=10)
                    pi_train = np.mean(self.bart_propensity_model.y_hat_train, axis = 1, keepdims = True)
                    pi_test = np.mean(self.bart_propensity_model.y_hat_test, axis = 1, keepdims = True)
                else:
                    self.bart_propensity_model.sample(X_train=X_train_processed, y_train=Z_train, num_gfr=10, num_mcmc=10)
                    pi_train = np.mean(self.bart_propensity_model.y_hat_train, axis = 1, keepdims = True)
                self.internal_propensity_model = True
        else:
            self.internal_propensity_model = False

        # Scale outcome
        if self.standardize:
            self.y_bar = np.squeeze(np.mean(y_train))
            self.y_std = np.squeeze(np.std(y_train))
        else:
            self.y_bar = 0
            self.y_std = 1
        resid_train = (y_train-self.y_bar)/self.y_std

        # Calibrate priors for global sigma^2 and sigma_leaf_mu / sigma_leaf_tau (don't use regression initializer for warm-start or XBART)
        if not sigma2_init:
            sigma2_init = pct_var_sigma2_init*np.var(resid_train)
        if not variance_forest_leaf_init:
            variance_forest_leaf_init = pct_var_variance_forest_init*np.var(resid_train)
        b_leaf_mu = np.squeeze(np.var(resid_train)) / num_trees_mu if b_leaf_mu is None else b_leaf_mu
        b_leaf_tau = np.squeeze(np.var(resid_train)) / (2*num_trees_tau) if b_leaf_tau is None else b_leaf_tau
        sigma_leaf_mu = np.squeeze(np.var(resid_train)) / num_trees_mu if sigma_leaf_mu is None else sigma_leaf_mu
        sigma_leaf_tau = np.squeeze(np.var(resid_train)) / (2*num_trees_tau) if sigma_leaf_tau is None else sigma_leaf_tau
        if self.multivariate_treatment:
            if not isinstance(sigma_leaf_tau, np.ndarray):
                sigma_leaf_tau = np.diagflat(np.repeat(sigma_leaf_tau, self.treatment_dim))
        current_sigma2 = sigma2_init
        self.sigma2_init = sigma2_init
        current_leaf_scale_mu = np.array([[sigma_leaf_mu]])
        if not isinstance(sigma_leaf_tau, np.ndarray):
            current_leaf_scale_tau = np.array([[sigma_leaf_tau]])
        else:
            current_leaf_scale_tau = sigma_leaf_tau
        if self.include_variance_forest:
            if not a_forest:
                a_forest = num_trees_variance / 1.5**2 + 0.5
            if not b_forest:
                b_forest = num_trees_variance / 1.5**2
        else:
            if not a_forest:
                a_forest = 1.
            if not b_forest:
                b_forest = 1.


        # Update variable weights
        variable_counts = [original_var_indices.count(i) for i in original_var_indices]
        variable_weights_mu_adj = [1/i for i in variable_counts]
        variable_weights_tau_adj = [1/i for i in variable_counts]
        variable_weights_variance_adj = [1/i for i in variable_counts]
        variable_weights_mu = variable_weights_mu[original_var_indices]*variable_weights_mu_adj
        variable_weights_tau = variable_weights_tau[original_var_indices]*variable_weights_tau_adj
        variable_weights_variance = variable_weights_variance[original_var_indices]*variable_weights_variance_adj

        # Create mu, tau, and variance specific variable weights with weights zeroed out for excluded variables
        variable_weights_mu[[variable_subset_mu.count(i) == 0 for i in original_var_indices]] = 0
        variable_weights_tau[[variable_subset_tau.count(i) == 0 for i in original_var_indices]] = 0
        variable_weights_variance[[variable_subset_variance.count(i) == 0 for i in original_var_indices]] = 0
        
        # Update covariates to include propensities if requested
        if propensity_covariate not in ["none", "mu", "tau", "both"]:
            raise ValueError("propensity_covariate must equal one of 'none', 'mu', 'tau', or 'both'")
        if propensity_covariate != "none":
            feature_types = np.append(feature_types, 0).astype('int')
            X_train_processed = np.c_[X_train_processed, pi_train]
            if self.has_test:
                X_test_processed = np.c_[X_test_processed, pi_test]
            if propensity_covariate == "mu":
                variable_weights_mu = np.append(variable_weights_mu, np.repeat(1/num_cov_orig, pi_train.shape[1]))
                variable_weights_tau = np.append(variable_weights_tau, np.repeat(0., pi_train.shape[1]))
            elif propensity_covariate == "tau":
                variable_weights_mu = np.append(variable_weights_mu, np.repeat(0., pi_train.shape[1]))
                variable_weights_tau = np.append(variable_weights_tau, np.repeat(1/num_cov_orig, pi_train.shape[1]))
            elif propensity_covariate == "both":
                variable_weights_mu = np.append(variable_weights_mu, np.repeat(1/num_cov_orig, pi_train.shape[1]))
                variable_weights_tau = np.append(variable_weights_tau, np.repeat(1/num_cov_orig, pi_train.shape[1]))
        variable_weights_variance = np.append(variable_weights_variance, np.repeat(0., pi_train.shape[1]))
        
        # Renormalize variable weights
        variable_weights_mu = variable_weights_mu / np.sum(variable_weights_mu)
        variable_weights_tau = variable_weights_tau / np.sum(variable_weights_tau)
        variable_weights_variance = variable_weights_variance / np.sum(variable_weights_variance)
        
        # Store propensity score requirements of the BCF forests
        self.propensity_covariate = propensity_covariate

        # Container of variance parameter samples
        self.num_gfr = num_gfr
        self.num_burnin = num_burnin
        self.num_mcmc = num_mcmc
        num_actual_mcmc_iter = num_mcmc * keep_every
        num_temp_samples = num_gfr + num_burnin + num_actual_mcmc_iter
        num_retained_samples = num_mcmc
        # Delete GFR samples from these containers after the fact if desired
        # if keep_gfr:
        #     num_retained_samples += num_gfr
        num_retained_samples += num_gfr
        if keep_burnin:
            num_retained_samples += num_burnin
        self.num_samples = num_retained_samples
        self.sample_sigma_global = sample_sigma_global
        self.sample_sigma_leaf_mu = sample_sigma_leaf_mu
        self.sample_sigma_leaf_tau = sample_sigma_leaf_tau
        if sample_sigma_global:
            self.global_var_samples = np.empty(self.num_samples, dtype=np.float64)
        if sample_sigma_leaf_mu:
            self.leaf_scale_mu_samples = np.empty(self.num_samples, dtype=np.float64)
        if sample_sigma_leaf_tau:
            self.leaf_scale_tau_samples = np.empty(self.num_samples, dtype=np.float64)
        sample_counter = -1
        
        # Prepare adaptive coding structure
        if self.adaptive_coding:
            if np.size(b_0) > 1 or np.size(b_1) > 1:
                raise ValueError("b_0 and b_1 must be single numeric values")
            if not (isinstance(b_0, (int, float)) or isinstance(b_1, (int, float))):
                raise ValueError("b_0 and b_1 must be numeric values")
            self.b0_samples = np.empty(self.num_samples, dtype=np.float64)
            self.b1_samples = np.empty(self.num_samples, dtype=np.float64)
            current_b_0 = b_0
            current_b_1 = b_1
            tau_basis_train = (1-Z_train)*current_b_0 + Z_train*current_b_1
            if self.has_test:
                tau_basis_test = (1-Z_test)*current_b_0 + Z_test*current_b_1
        else:
            tau_basis_train = Z_train
            if self.has_test:
                tau_basis_test = Z_test

        # Prognostic Forest Dataset (covariates)
        forest_dataset_train = Dataset()
        forest_dataset_train.add_covariates(X_train_processed)
        forest_dataset_train.add_basis(tau_basis_train)
        if self.has_test:
            forest_dataset_test = Dataset()
            forest_dataset_test.add_covariates(X_test_processed)
            forest_dataset_test.add_basis(tau_basis_test)

        # Residual
        residual_train = Residual(resid_train)

        # C++ random number generator
        if random_seed is None: 
            cpp_rng = RNG(-1)
        else:
            cpp_rng = RNG(random_seed)
        
        # Sampling data structures
        forest_sampler_mu = ForestSampler(forest_dataset_train, feature_types, num_trees_mu, self.n_train, alpha_mu, beta_mu, min_samples_leaf_mu, max_depth_mu)
        forest_sampler_tau = ForestSampler(forest_dataset_train, feature_types, num_trees_tau, self.n_train, alpha_tau, beta_tau, min_samples_leaf_tau, max_depth_tau)
        if self.include_variance_forest:
            forest_sampler_variance = ForestSampler(forest_dataset_train, feature_types, num_trees_variance, self.n_train, alpha_variance, beta_variance, min_samples_leaf_variance, max_depth_variance)

        # Container of forest samples
        self.forest_container_mu = ForestContainer(num_trees_mu, 1, True, False)
        self.forest_container_tau = ForestContainer(num_trees_tau, Z_train.shape[1], False, False)
        active_forest_mu = Forest(num_trees_mu, 1, True, False)
        active_forest_tau = Forest(num_trees_tau, Z_train.shape[1], False, False)
        if self.include_variance_forest:
            self.forest_container_variance = ForestContainer(num_trees_variance, 1, True, True)
            active_forest_variance = Forest(num_trees_variance, 1, True, True)
        
        # Variance samplers
        if self.sample_sigma_global:
            global_var_model = GlobalVarianceModel()
        if self.sample_sigma_leaf_mu:
            leaf_var_model_mu = LeafVarianceModel()
        if self.sample_sigma_leaf_tau:
            leaf_var_model_tau = LeafVarianceModel()

        # Initialize the leaves of each tree in the prognostic forest
        init_mu = np.array([np.squeeze(np.mean(resid_train))])
        forest_sampler_mu.prepare_for_sampler(forest_dataset_train, residual_train, active_forest_mu, 0, init_mu)

        # Initialize the leaves of each tree in the treatment forest
        if self.multivariate_treatment:
            init_tau = np.zeros(Z_train.shape[1])
        else:
            init_tau = np.array([0.])
        forest_sampler_tau.prepare_for_sampler(forest_dataset_train, residual_train, active_forest_tau, treatment_leaf_model, init_tau)

        # Initialize the leaves of each tree in the variance forest
        if self.include_variance_forest:
            init_val_variance = np.array([variance_forest_leaf_init])
            forest_sampler_variance.prepare_for_sampler(forest_dataset_train, residual_train, active_forest_variance, leaf_model_variance_forest, init_val_variance)

        # Run GFR (warm start) if specified
        if num_gfr > 0:
            for i in range(num_gfr):
                # Keep all GFR samples at this stage -- remove from ForestSamples after MCMC
                # keep_sample = keep_gfr
                keep_sample = True
                if keep_sample:
                    sample_counter += 1
                # Sample the prognostic forest
                forest_sampler_mu.sample_one_iteration(
                    self.forest_container_mu, active_forest_mu, forest_dataset_train, residual_train, cpp_rng, feature_types, 
                    cutpoint_grid_size, current_leaf_scale_mu, variable_weights_mu, a_forest, b_forest, 
                    current_sigma2, 0, keep_sample, True, True
                )

                # Sample variance parameters (if requested)
                if self.sample_sigma_global:
                    current_sigma2 = global_var_model.sample_one_iteration(residual_train, cpp_rng, a_global, b_global)
                if self.sample_sigma_leaf_mu:
                    current_leaf_scale_mu[0,0] = leaf_var_model_mu.sample_one_iteration(active_forest_mu, cpp_rng, a_leaf_mu, b_leaf_mu)
                    if keep_sample:
                        self.leaf_scale_mu_samples[sample_counter] = current_leaf_scale_mu[0,0]
                
                # Sample the treatment forest
                forest_sampler_tau.sample_one_iteration(
                    self.forest_container_tau, active_forest_tau, forest_dataset_train, residual_train, cpp_rng, feature_types, 
                    cutpoint_grid_size, current_leaf_scale_tau, variable_weights_tau, a_forest, b_forest, 
                    current_sigma2, treatment_leaf_model, keep_sample, True, True
                )
                
                # Sample coding parameters (if requested)
                if self.adaptive_coding:
                    mu_x = active_forest_mu.predict_raw(forest_dataset_train)
                    tau_x = np.squeeze(active_forest_tau.predict_raw(forest_dataset_train))
                    s_tt0 = np.sum(tau_x*tau_x*(np.squeeze(Z_train)==0))
                    s_tt1 = np.sum(tau_x*tau_x*(np.squeeze(Z_train)==1))
                    partial_resid_mu = np.squeeze(resid_train - mu_x)
                    s_ty0 = np.sum(tau_x*partial_resid_mu*(np.squeeze(Z_train)==0))
                    s_ty1 = np.sum(tau_x*partial_resid_mu*(np.squeeze(Z_train)==1))
                    current_b_0 = self.rng.normal(loc = (s_ty0/(s_tt0 + 2*current_sigma2)), 
                                             scale = np.sqrt(current_sigma2/(s_tt0 + 2*current_sigma2)), size = 1)
                    current_b_1 = self.rng.normal(loc = (s_ty1/(s_tt1 + 2*current_sigma2)), 
                                             scale = np.sqrt(current_sigma2/(s_tt1 + 2*current_sigma2)), size = 1)
                    tau_basis_train = (1-np.squeeze(Z_train))*current_b_0 + np.squeeze(Z_train)*current_b_1
                    forest_dataset_train.update_basis(tau_basis_train)
                    if self.has_test:
                        tau_basis_test = (1-np.squeeze(Z_test))*current_b_0 + np.squeeze(Z_test)*current_b_1
                        forest_dataset_test.update_basis(tau_basis_test)
                    if keep_sample:
                        self.b0_samples[sample_counter] = current_b_0
                        self.b1_samples[sample_counter] = current_b_1

                    # Update residual to reflect adjusted basis
                    forest_sampler_tau.propagate_basis_update(forest_dataset_train, residual_train, active_forest_tau)
                
                # Sample the variance forest
                if self.include_variance_forest:
                    forest_sampler_variance.sample_one_iteration(
                        self.forest_container_variance, active_forest_variance, forest_dataset_train, residual_train, 
                        cpp_rng, feature_types, cutpoint_grid_size, current_leaf_scale_mu, variable_weights_variance, a_forest, b_forest, 
                        current_sigma2, leaf_model_variance_forest, keep_sample, True, True
                    )
                
                # Sample variance parameters (if requested)
                if self.sample_sigma_global:
                    current_sigma2 = global_var_model.sample_one_iteration(residual_train, cpp_rng, a_global, b_global)
                    if keep_sample:
                        self.global_var_samples[sample_counter] = current_sigma2
                if self.sample_sigma_leaf_tau:
                    current_leaf_scale_tau[0,0] = leaf_var_model_tau.sample_one_iteration(active_forest_tau, cpp_rng, a_leaf_tau, b_leaf_tau)
                    if keep_sample:
                        self.leaf_scale_tau_samples[sample_counter] = current_leaf_scale_tau[0,0]
        
        # Run MCMC
        if num_burnin + num_mcmc > 0:
            for i in range(num_gfr, num_temp_samples):
                is_mcmc = i + 1 > num_gfr + num_burnin
                if is_mcmc:
                    mcmc_counter = i - num_gfr - num_burnin + 1
                    if (mcmc_counter % keep_every == 0):
                        keep_sample = True
                    else:
                        keep_sample = False
                else:
                    if keep_burnin:
                        keep_sample = True
                    else:
                        keep_sample = False
                if keep_sample:
                    sample_counter += 1
                # Sample the prognostic forest
                forest_sampler_mu.sample_one_iteration(
                    self.forest_container_mu, active_forest_mu, forest_dataset_train, residual_train, cpp_rng, feature_types, 
                    cutpoint_grid_size, current_leaf_scale_mu, variable_weights_mu, a_forest, b_forest, 
                    current_sigma2, 0, keep_sample, False, True
                )

                # Sample variance parameters (if requested)
                if self.sample_sigma_global:
                    current_sigma2 = global_var_model.sample_one_iteration(residual_train, cpp_rng, a_global, b_global)
                if self.sample_sigma_leaf_mu:
                    current_leaf_scale_mu[0,0] = leaf_var_model_mu.sample_one_iteration(active_forest_mu, cpp_rng, a_leaf_mu, b_leaf_mu)
                    if keep_sample:
                        self.leaf_scale_mu_samples[sample_counter] = current_leaf_scale_mu[0,0]
                
                # Sample the treatment forest
                forest_sampler_tau.sample_one_iteration(
                    self.forest_container_tau, active_forest_tau, forest_dataset_train, residual_train, cpp_rng, feature_types, 
                    cutpoint_grid_size, current_leaf_scale_tau, variable_weights_tau, a_forest, b_forest, 
                    current_sigma2, treatment_leaf_model, keep_sample, False, True
                )    
                
                # Sample coding parameters (if requested)
                if self.adaptive_coding:
                    mu_x = active_forest_mu.predict_raw(forest_dataset_train)
                    tau_x = np.squeeze(active_forest_tau.predict_raw(forest_dataset_train))
                    s_tt0 = np.sum(tau_x*tau_x*(np.squeeze(Z_train)==0))
                    s_tt1 = np.sum(tau_x*tau_x*(np.squeeze(Z_train)==1))
                    partial_resid_mu = np.squeeze(resid_train - mu_x)
                    s_ty0 = np.sum(tau_x*partial_resid_mu*(np.squeeze(Z_train)==0))
                    s_ty1 = np.sum(tau_x*partial_resid_mu*(np.squeeze(Z_train)==1))
                    current_b_0 = self.rng.normal(loc = (s_ty0/(s_tt0 + 2*current_sigma2)), 
                                             scale = np.sqrt(current_sigma2/(s_tt0 + 2*current_sigma2)), size = 1)
                    current_b_1 = self.rng.normal(loc = (s_ty1/(s_tt1 + 2*current_sigma2)), 
                                             scale = np.sqrt(current_sigma2/(s_tt1 + 2*current_sigma2)), size = 1)
                    tau_basis_train = (1-np.squeeze(Z_train))*current_b_0 + np.squeeze(Z_train)*current_b_1
                    forest_dataset_train.update_basis(tau_basis_train)
                    if self.has_test:
                        tau_basis_test = (1-np.squeeze(Z_test))*current_b_0 + np.squeeze(Z_test)*current_b_1
                        forest_dataset_test.update_basis(tau_basis_test)
                    if keep_sample:
                        self.b0_samples[sample_counter] = current_b_0
                        self.b1_samples[sample_counter] = current_b_1

                    # Update residual to reflect adjusted basis
                    forest_sampler_tau.propagate_basis_update(forest_dataset_train, residual_train, active_forest_tau)
                
                # Sample the variance forest
                if self.include_variance_forest:
                    forest_sampler_variance.sample_one_iteration(
                        self.forest_container_variance, active_forest_variance, forest_dataset_train, residual_train, 
                        cpp_rng, feature_types, cutpoint_grid_size, current_leaf_scale_mu, variable_weights_variance, a_forest, b_forest, 
                        current_sigma2, leaf_model_variance_forest, keep_sample, False, True
                    )
                
                # Sample variance parameters (if requested)
                if self.sample_sigma_global:
                    current_sigma2 = global_var_model.sample_one_iteration(residual_train, cpp_rng, a_global, b_global)
                    if keep_sample:
                        self.global_var_samples[sample_counter] = current_sigma2
                if self.sample_sigma_leaf_tau:
                    current_leaf_scale_tau[0,0] = leaf_var_model_tau.sample_one_iteration(active_forest_tau, cpp_rng, a_leaf_tau, b_leaf_tau)
                    if keep_sample:
                        self.leaf_scale_tau_samples[sample_counter] = current_leaf_scale_tau[0,0]       
        
        # Mark the model as sampled
        self.sampled = True

        # Remove GFR samples if they are not to be retained
        if not keep_gfr and num_gfr > 0:
            for i in range(num_gfr):
                self.forest_container_mu.delete_sample(i)
                self.forest_container_tau.delete_sample(i)
                if self.include_variance_forest:
                    self.forest_container_variance.delete_sample(i)
            if self.adaptive_coding:
                self.b1_samples = self.b1_samples[num_gfr:]
                self.b0_samples = self.b0_samples[num_gfr:]
            if self.sample_sigma_global:
                self.global_var_samples = self.global_var_samples[num_gfr:]
            if self.sample_sigma_leaf_mu:
                self.leaf_scale_mu_samples = self.leaf_scale_mu_samples[num_gfr:]
            if self.sample_sigma_leaf_tau:
                self.leaf_scale_tau_samples = self.leaf_scale_tau_samples[num_gfr:]
            self.num_samples -= num_gfr

        # Store predictions
        mu_raw = self.forest_container_mu.forest_container_cpp.Predict(forest_dataset_train.dataset_cpp)
        self.mu_hat_train = mu_raw*self.y_std + self.y_bar
        tau_raw_train = self.forest_container_tau.forest_container_cpp.PredictRaw(forest_dataset_train.dataset_cpp)
        self.tau_hat_train = tau_raw_train
        if self.adaptive_coding:
            adaptive_coding_weights = np.expand_dims(self.b1_samples - self.b0_samples, axis=(0,2))
            self.tau_hat_train = self.tau_hat_train*adaptive_coding_weights
        self.tau_hat_train = np.squeeze(self.tau_hat_train*self.y_std)
        if self.multivariate_treatment:
            treatment_term_train = np.multiply(np.atleast_3d(Z_train).swapaxes(1,2),self.tau_hat_train).sum(axis=2)
        else:
            treatment_term_train = Z_train*np.squeeze(self.tau_hat_train)
        self.y_hat_train = self.mu_hat_train + treatment_term_train
        if self.has_test:
            mu_raw_test = self.forest_container_mu.forest_container_cpp.Predict(forest_dataset_test.dataset_cpp)
            self.mu_hat_test = mu_raw_test*self.y_std + self.y_bar
            tau_raw_test = self.forest_container_tau.forest_container_cpp.PredictRaw(forest_dataset_test.dataset_cpp)
            self.tau_hat_test = tau_raw_test
            if self.adaptive_coding:
                adaptive_coding_weights_test = np.expand_dims(self.b1_samples - self.b0_samples, axis=(0,2))
                self.tau_hat_test = self.tau_hat_test*adaptive_coding_weights_test
            self.tau_hat_test = np.squeeze(self.tau_hat_test*self.y_std)
            if self.multivariate_treatment:
                treatment_term_test = np.multiply(np.atleast_3d(Z_test).swapaxes(1,2),self.tau_hat_test).sum(axis=2)
            else:
                treatment_term_test = Z_test*np.squeeze(self.tau_hat_test)
            self.y_hat_test = self.mu_hat_test + treatment_term_test
        
        if self.include_variance_forest:
            sigma2_x_train_raw = self.forest_container_variance.forest_container_cpp.Predict(forest_dataset_train.dataset_cpp)
            if self.sample_sigma_global:
                self.sigma2_x_train = sigma2_x_train_raw
                for i in range(self.num_samples):
                    self.sigma2_x_train[:,i] = sigma2_x_train_raw[:,i]*self.global_var_samples[i]
            else:
                self.sigma2_x_train = sigma2_x_train_raw*self.sigma2_init*self.y_std*self.y_std/self.variance_scale
            if self.has_test:
                sigma2_x_test_raw = self.forest_container_variance.forest_container_cpp.Predict(forest_dataset_test.dataset_cpp)
                if self.sample_sigma_global:
                    self.sigma2_x_test = sigma2_x_test_raw
                    for i in range(self.num_samples):
                        self.sigma2_x_test[:,i] = sigma2_x_test_raw[:,i]*self.global_var_samples[i]
                else:
                    self.sigma2_x_test = sigma2_x_test_raw*self.sigma2_init*self.y_std*self.y_std/self.variance_scale
    
        if self.sample_sigma_global:
            self.global_var_samples = self.global_var_samples*self.y_std*self.y_std
        
        if self.sample_sigma_leaf_mu:
            self.leaf_scale_mu_samples = self.leaf_scale_mu_samples

        if self.sample_sigma_leaf_tau:
            self.leaf_scale_tau_samples = self.leaf_scale_tau_samples
        
        if self.adaptive_coding:
            self.b0_samples = self.b0_samples
            self.b1_samples = self.b1_samples
    
    def predict_tau(self, X: np.array, Z: np.array, propensity: np.array = None) -> np.array:
        """Predict CATE function for every provided observation.

        Parameters
        ----------
        X : np.array or pd.DataFrame
            Test set covariates.
        Z : np.array
            Test set treatment indicators.
        propensity : np.array, optional
            Optional test set propensities. Must be provided if propensities were provided when the model was sampled.
        
        Returns
        -------
        np.array
            Array with as many rows as in `X` and as many columns as retained samples of the algorithm.
        """
        if not self.is_sampled():
            msg = (
                "This BCFModel instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this model."
            )
            raise NotSampledError(msg)
        
        # Convert everything to standard shape (2-dimensional)
        if X.ndim == 1:
            X = np.expand_dims(X, 1)
        if Z.ndim == 1:
            Z = np.expand_dims(Z, 1)
        else:
            if Z.ndim != 2:
                raise ValueError("treatment must have 1 or 2 dimensions")
        if propensity is not None:
            if propensity.ndim == 1:
                propensity = np.expand_dims(propensity, 1)
        
        # Data checks
        if Z.shape[0] != X.shape[0]:
            raise ValueError("X and Z must have the same number of rows")
        if propensity is not None:
            if propensity.shape[0] != X.shape[0]:
                raise ValueError("X and propensity must have the same number of rows")
        else:
            if self.propensity_covariate == "tau":
                if not self.internal_propensity_model:
                    raise ValueError("Propensity scores not provided, but no propensity model was trained during sampling")
                else:
                    propensity = np.mean(self.bart_propensity_model.predict(X), axis=1, keepdims=True)
            else:
                # Dummy propensities if not provided but also not needed
                propensity = np.ones(X.shape[0])
                propensity = np.expand_dims(propensity, 1)
        
        # Update covariates to include propensities if requested
        if self.propensity_covariate == "none":
            X_combined = X
        else:
            X_combined = np.c_[X, propensity]
        
        # Forest dataset
        forest_dataset_test = Dataset()
        forest_dataset_test.add_covariates(X_combined)
        forest_dataset_test.add_basis(Z)
        
        # Estimate treatment effect
        tau_raw = self.forest_container_tau.forest_container_cpp.PredictRaw(forest_dataset_test.dataset_cpp)
        tau_raw = tau_raw
        if self.adaptive_coding:
            adaptive_coding_weights = np.expand_dims(self.b1_samples - self.b0_samples, axis=(0,2))
            tau_raw = tau_raw*adaptive_coding_weights
        tau_x = np.squeeze(tau_raw*self.y_std)

        # Return result matrix
        return tau_x

    def predict_variance(self, covariates: np.array, propensity: np.array = None) -> np.array:
        """Predict expected conditional variance from a BART model.

        Parameters
        ----------
        covariates : np.array
            Test set covariates.
        propensity : np.array, optional
            Test set propensity scores. Optional (not currently used in variance forests).
        
        Returns
        -------
        np.array
            Array of predictions corresponding to the variance forest. Each array will contain as many rows as in `covariates` and as many columns as retained samples of the algorithm.
        """
        if not self.is_sampled():
            msg = (
                "This BARTModel instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this model."
            )
            raise NotSampledError(msg)

        if not self.include_variance_forest:
            msg = (
                "This BARTModel instance was not sampled with a variance forest. "
                "Call 'fit' with appropriate arguments before using this model."
            )
            raise NotSampledError(msg)
    
        # Convert everything to standard shape (2-dimensional)
        if covariates.ndim == 1:
            covariates = np.expand_dims(covariates, 1)
        if propensity is not None:
            if propensity.ndim == 1:
                propensity = np.expand_dims(propensity, 1)
        
        # Update covariates to include propensities if requested
        if self.propensity_covariate == "none":
            X_combined = covariates
        else:
            if propensity is not None:
                X_combined = np.c_[covariates, propensity]
            else:
                # Dummy propensities if not provided but also not needed
                propensity = np.ones(covariates.shape[0])
                propensity = np.expand_dims(propensity, 1)
                X_combined = np.c_[covariates, propensity]
        
        # Forest dataset
        pred_dataset = Dataset()
        pred_dataset.add_covariates(X_combined)
        
        variance_pred_raw = self.forest_container_variance.forest_container_cpp.Predict(pred_dataset.dataset_cpp)
        if self.sample_sigma_global:
            variance_pred = variance_pred_raw
            for i in range(self.num_samples):
                variance_pred[:,i] = variance_pred_raw[:,i]*self.global_var_samples[i]
        else:
            variance_pred = variance_pred_raw*self.sigma2_init*self.y_std*self.y_std/self.variance_scale

        return variance_pred
    
    def predict(self, X: np.array, Z: np.array, propensity: np.array = None) -> tuple:
        """Predict outcome model components (CATE function and prognostic function) as well as overall outcome for every provided observation. 
        Predicted outcomes are computed as `yhat = mu_x + Z*tau_x` where mu_x is a sample of the prognostic function and tau_x is a sample of the treatment effect (CATE) function.

        Parameters
        ----------
        X : np.array or pd.DataFrame
            Test set covariates.
        Z : np.array
            Test set treatment indicators.
        propensity : `np.array`, optional
            Optional test set propensities. Must be provided if propensities were provided when the model was sampled.
        
        Returns
        -------
        tau_x : np.array
            Conditional average treatment effect (CATE) samples for every observation provided.
        mu_x : np.array
            Prognostic effect samples for every observation provided.
        yhat_x : np.array
            Outcome prediction samples for every observation provided.
        sigma2_x : np.array, optional
            Variance forest samples for every observation provided. Only returned if the 
            model includes a heteroskedasticity forest.
        """
        if not self.is_sampled():
            msg = (
                "This BCFModel instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this model."
            )
            raise NotSampledError(msg)
        
        # Convert everything to standard shape (2-dimensional)
        if X.ndim == 1:
            X = np.expand_dims(X, 1)
        if Z.ndim == 1:
            Z = np.expand_dims(Z, 1)
        else:
            if Z.ndim != 2:
                raise ValueError("treatment must have 1 or 2 dimensions")
        if propensity is not None:
            if propensity.ndim == 1:
                propensity = np.expand_dims(propensity, 1)
        
        # Data checks
        if Z.shape[0] != X.shape[0]:
            raise ValueError("X and Z must have the same number of rows")
        if propensity is not None:
            if propensity.shape[0] != X.shape[0]:
                raise ValueError("X and propensity must have the same number of rows")
        else:
            if self.propensity_covariate != "none":
                if not self.internal_propensity_model:
                    raise ValueError("Propensity scores not provided, but no propensity model was trained during sampling")
                else:
                    propensity = np.mean(self.bart_propensity_model.predict(X), axis=1, keepdims=True)
        
        # Update covariates to include propensities if requested
        if self.propensity_covariate == "none":
            X_combined = X
        else:
            X_combined = np.c_[X, propensity]
        
        # Forest dataset
        forest_dataset_test = Dataset()
        forest_dataset_test.add_covariates(X_combined)
        forest_dataset_test.add_basis(Z)
        
        # Compute predicted outcome and decomposed outcome model terms
        mu_raw = self.forest_container_mu.forest_container_cpp.Predict(forest_dataset_test.dataset_cpp)
        mu_x = mu_raw*self.y_std/np.sqrt(self.variance_scale) + self.y_bar
        tau_raw = self.forest_container_tau.forest_container_cpp.PredictRaw(forest_dataset_test.dataset_cpp)
        if self.adaptive_coding:
            adaptive_coding_weights = np.expand_dims(self.b1_samples - self.b0_samples, axis=(0,2))
            tau_raw = tau_raw*adaptive_coding_weights
        tau_x = np.squeeze(tau_raw*self.y_std/np.sqrt(self.variance_scale))
        if Z.shape[1] > 1:
            treatment_term = np.multiply(np.atleast_3d(Z).swapaxes(1,2),tau_x).sum(axis=2)
        else:
            treatment_term = Z*np.squeeze(tau_x)
        yhat_x = mu_x + treatment_term

        # Compute predictions from the variance forest (if included)
        if self.include_variance_forest:
            sigma2_x_raw = self.forest_container_variance.forest_container_cpp.Predict(forest_dataset_test.dataset_cpp)        
            if self.sample_sigma_global:
                sigma2_x = sigma2_x_raw
                for i in range(self.num_samples):
                    sigma2_x[:,i] = sigma2_x_raw[:,i]*self.global_var_samples[i]
            else:
                sigma2_x = sigma2_x_raw*self.sigma2_init*self.y_std*self.y_std/self.variance_scale
        
        # Return result matrices as a tuple
        if self.include_variance_forest:
            return (tau_x, mu_x, yhat_x, sigma2_x)
        else:
            return (tau_x, mu_x, yhat_x)

    def to_json(self) -> str:
        """
        Converts a sampled BART model to JSON string representation (which can then be saved to a file or 
        processed using the `json` library)

        Returns
        -------
        str
            JSON string representing model metadata (hyperparameters), sampled parameters, and sampled forests
        """
        if not self.is_sampled:
            msg = (
                "This BARTModel instance has not yet been sampled. "
                "Call 'fit' with appropriate arguments before using this model."
            )
            raise NotSampledError(msg)
        
        # Initialize JSONSerializer object
        bcf_json = JSONSerializer()
        
        # Add the forests
        bcf_json.add_forest(self.forest_container_mu)
        bcf_json.add_forest(self.forest_container_tau)
        if self.include_variance_forest:
            bcf_json.add_forest(self.forest_container_variance)
        
        # Add global parameters
        bcf_json.add_scalar("variance_scale", self.variance_scale)
        bcf_json.add_scalar("outcome_scale", self.y_std)
        bcf_json.add_scalar("outcome_mean", self.y_bar)
        bcf_json.add_boolean("standardize", self.standardize)
        bcf_json.add_scalar("sigma2_init", self.sigma2_init)
        bcf_json.add_boolean("sample_sigma_global", self.sample_sigma_global)
        bcf_json.add_boolean("sample_sigma_leaf_mu", self.sample_sigma_leaf_mu)
        bcf_json.add_boolean("sample_sigma_leaf_tau", self.sample_sigma_leaf_tau)
        bcf_json.add_boolean("include_variance_forest", self.include_variance_forest)
        bcf_json.add_scalar("num_gfr", self.num_gfr)
        bcf_json.add_scalar("num_burnin", self.num_burnin)
        bcf_json.add_scalar("num_mcmc", self.num_mcmc)
        bcf_json.add_scalar("num_samples", self.num_samples)
        bcf_json.add_boolean("adaptive_coding", self.adaptive_coding)
        bcf_json.add_string("propensity_covariate", self.propensity_covariate)
        
        # Add parameter samples
        if self.sample_sigma_global:
            bcf_json.add_numeric_vector("sigma2_global_samples", self.global_var_samples, "parameters")
        if self.sample_sigma_leaf_mu:
            bcf_json.add_numeric_vector("sigma2_leaf_mu_samples", self.leaf_scale_mu_samples, "parameters")
        if self.sample_sigma_leaf_tau:
            bcf_json.add_numeric_vector("sigma2_leaf_tau_samples", self.leaf_scale_tau_samples, "parameters")
        if self.adaptive_coding:
            bcf_json.add_numeric_vector("b0_samples", self.b0_samples, "parameters")
            bcf_json.add_numeric_vector("b1_samples", self.b1_samples, "parameters")
        
        return bcf_json.return_json_string()

    def from_json(self, json_string: str) -> None:
        """
        Converts a JSON string to an in-memory BART model.

        Parameters
        ----------
        json_string : str
            JSON string representing model metadata (hyperparameters), sampled parameters, and sampled forests
        """
        # Parse string to a JSON object in C++
        bcf_json = JSONSerializer()
        bcf_json.load_from_json_string(json_string)
        
        # Unpack forests
        self.include_variance_forest = bcf_json.get_boolean("include_variance_forest")
        # TODO: don't just make this a placeholder that we overwrite
        self.forest_container_mu = ForestContainer(0, 0, False, False)
        self.forest_container_mu.forest_container_cpp.LoadFromJson(bcf_json.json_cpp, "forest_0")
        # TODO: don't just make this a placeholder that we overwrite
        self.forest_container_tau = ForestContainer(0, 0, False, False)
        self.forest_container_tau.forest_container_cpp.LoadFromJson(bcf_json.json_cpp, "forest_1")
        if self.include_variance_forest:
            # TODO: don't just make this a placeholder that we overwrite
            self.forest_container_variance = ForestContainer(0, 0, False, False)
            self.forest_container_variance.forest_container_cpp.LoadFromJson(bcf_json.json_cpp, "forest_2")
        
        # Unpack global parameters
        self.variance_scale = bcf_json.get_scalar("variance_scale")
        self.y_std = bcf_json.get_scalar("outcome_scale")
        self.y_bar = bcf_json.get_scalar("outcome_mean")
        self.standardize = bcf_json.get_boolean("standardize")
        self.sigma2_init = bcf_json.get_scalar("sigma2_init")
        self.sample_sigma_global = bcf_json.get_boolean("sample_sigma_global")
        self.sample_sigma_leaf_mu = bcf_json.get_boolean("sample_sigma_leaf_mu")
        self.sample_sigma_leaf_tau = bcf_json.get_boolean("sample_sigma_leaf_tau")
        self.num_gfr = int(bcf_json.get_scalar("num_gfr"))
        self.num_burnin = int(bcf_json.get_scalar("num_burnin"))
        self.num_mcmc = int(bcf_json.get_scalar("num_mcmc"))
        self.num_samples = int(bcf_json.get_scalar("num_samples"))
        self.adaptive_coding = bcf_json.get_boolean("adaptive_coding")
        self.propensity_covariate = bcf_json.get_string("propensity_covariate")

        # Unpack parameter samples
        if self.sample_sigma_global:
            self.global_var_samples = bcf_json.get_numeric_vector("sigma2_global_samples", "parameters")
        if self.sample_sigma_leaf_mu:
            self.leaf_scale_mu_samples = bcf_json.get_numeric_vector("sigma2_leaf_mu_samples", "parameters")
        if self.sample_sigma_leaf_tau:
            self.leaf_scale_tau_samples = bcf_json.get_numeric_vector("sigma2_leaf_tau_samples", "parameters")
        if self.adaptive_coding:
            self.b1_samples = bcf_json.get_numeric_vector("b1_samples", "parameters")
            self.b0_samples = bcf_json.get_numeric_vector("b0_samples", "parameters")
        
        # Mark the deserialized model as "sampled"
        self.sampled = True
    
    def is_sampled(self) -> bool:
        """Whether or not a BCF model has been sampled.

        Returns
        -------
        bool
            `True` if a BCF model has been sampled, `False` otherwise
        """
        return self.sampled

