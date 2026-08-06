# Audit of `bcf_restricted_score_test`: Mean and Distributional Settings

**File audited:** `R/bcf_score_test.R` (working tree, 2026-08-06)
**Scope:** statistical validity and methodological grounding of `test_type = "mean"` and `test_type = "distributional"`, the Rao-Blackwellization scheme, and the supporting helpers (`prepare_score_test_design`, `compute_pmvnorm_safe`, `compute_distributional_pvalue_safe`), with a literature review of closely related methods and a small empirical calibration study.

---

## 1. Executive summary

The function fits a *restricted* Bayesian Causal Forest — prognostic BART forest `mu(x)` plus a single scalar treatment coefficient `alpha` on the propensity-centered treatment `Z - pihat(x)` — and then tests the restriction (no treatment-effect heterogeneity) with a frequentist score test evaluated at (a Rao-Blackwellized average over) the posterior draws.

The core construction is sound and, in fact, rests on a genuinely good idea: the score for heterogeneity is a **Neyman-orthogonal, residual-on-residual (Robinson-style) moment**, `s_i = (Z_i - pihat(x_i)) * ehat_i`, whose first-order insensitivity to estimation error in `mu` and `pi` is exactly what makes a plug-in p-value at the posterior mean asymptotically calibrated (in the sense of Robins, van der Vaart & Ventura, 2000). The same orthogonality is what lets the distributional (ECDF) version sidestep the classical Durbin problem without a Khmaladze transformation. The Gaussian simulation from the empirical sandwich covariance is algebraically a Gaussian multiplier bootstrap, which has strong theoretical support for the max-type statistic even in high dimensions.

The main findings, in decreasing order of importance:

1. **[High] The distributional test is *not* doubly robust; it is single-robust in the propensity.** The mean test inherits Robinson-type product bias (valid if *either* `mu` *or* `pi` is well estimated). The distributional test centers indicators by the *marginal* ECDF, so under residual distributions that vary with `x`, its validity rests entirely on `E[Z - pihat(x) | x] = 0`, i.e., on a correctly estimated propensity. (§5.3, finding D1)
2. **[High] Random-effects arguments are silently ignored.** `rfx_group_ids_train` / `rfx_basis_train` are accepted and documented but never used; the i.i.d. sandwich covariance is then wrong for clustered data. (finding S1)
3. **[Medium] The joint covariance of the distributional score process is a `(p*K) x (p*K)` empirical outer-product with rank at most `n`.** With the defaults (`num_grid_points = 50`, all pairwise interactions in the design), `p*K` exceeds `n` in realistically sized problems, so the simulated null process lives on a degenerate subspace. The max statistic survives this (multiplier-bootstrap logic); the CvM/Wasserstein statistics, which invert `p x p` blocks, need `p << n` and deserve an explicit guard. (finding D2)
4. **[Medium] No cross-fitting; nuisances and score share the same data.** Orthogonality protects the first order, and BART's prior regularization plays the role that sample splitting plays in DML, but the residual-overfitting direction is toward conservatism/power loss and nothing in the code controls it. (finding S7)
5. **[Medium] The `alpha` (and, in observational designs, propensity-heteroskedasticity) projection is only partially removed.** Centering `s_i` by its mean handles the intercept direction; it does not remove the component of the score along the `Z - pihat` regressor when `pihat(x)(1 - pihat(x))` co-varies with the test covariates. In an RCT with constant propensity this term vanishes; in observational data it does not. (finding M2/S8)
6. Several smaller implementation bugs and documentation mismatches (hard-coded `sigma^2` prior, `mu_hat_train` from last chain only, unused `mvtnorm` dependency, `keep_every`/`keep_burnin`/`keep_gfr` ignored, "Wasserstein" naming). (§5)

An empirical calibration study (n = 250, RCT with known propensity, BART-relevant nonlinear prognostic surface; §6) corroborates the analysis: both tests are conservative under the null (no p-value below 0.08 in 40 null replicates for the quad/CvM/Wasserstein statistics), the mean test has good power against linear heterogeneity (65–90%), but the distributional test — run at its defaults, which put `p*K = 750 > n = 250` — has almost no power against the variance-only heterogeneity that motivates it (5–10% at nominal 5%).

---

## 2. What the function implements

### 2.1 Restricted model

After standardizing `y`, the sampler alternates ([bcf_score_test.R:229–266](R/bcf_score_test.R#L229-L266)):

- BART prognostic forest `mu(x)` (propensity appended as a covariate, per Hahn–Murray–Carvalho BCF practice);
- conjugate normal update for the scalar `alpha` on the centered treatment `Z_cen = Z - pihat(x)` ([bcf_score_test.R:254–256](R/bcf_score_test.R#L254-L256));
- inverse-gamma update for the global `sigma^2` ([bcf_score_test.R:261–266](R/bcf_score_test.R#L261-L266)).

So the null model is `y = mu(x) + alpha * (Z - pihat(x)) + eps`, `eps ~ N(0, sigma^2)`: a partially linear model with a *constant* treatment effect. The alternative, implicitly, is `tau(x) = alpha + X_cen' beta` (mean test) or any conditional-distribution dependence of the treatment contrast on `x` (distributional test).

### 2.2 Mean test

Per retained draw, `s_i = ehat_i * Z_cen_i` with `ehat_i` the full residual ([bcf_score_test.R:279](R/bcf_score_test.R#L279)). Rao-Blackwellization accumulates `s_i` across draws; post-processing forms

- score vector `T = X_cen' s_bar`,
- Huber–White covariance `V = sum_i x_i x_i' s_bar_i^2` ([bcf_score_test.R:330–335](R/bcf_score_test.R#L330-L335)),
- a chi-squared statistic `T' V^{-1} T ~ chi^2_p` and a simulated max-|Z| statistic ([bcf_score_test.R:469–507](R/bcf_score_test.R#L469-L507)).

`X_cen` is built by `prepare_score_test_design` ([bcf_score_test.R:399–458](R/bcf_score_test.R#L399-L458)): standardized covariates plus pairwise interactions (per `interaction_rule`), QR-reduced to a full-rank basis of dimension `p_valid`.

### 2.3 Distributional test

Per retained draw, on a fixed grid `t_1 < ... < t_K` of residual quantiles (set at the first retained draw, [bcf_score_test.R:296–299](R/bcf_score_test.R#L296-L299)):

`s_i(t_k) = Z_cen_i * ( 1{ehat_i <= t_k} - Fhat(t_k) )`,

with `Fhat` the within-draw marginal ECDF ([bcf_score_test.R:306–309](R/bcf_score_test.R#L306-L309)). After averaging over draws, `compute_distributional_pvalue_safe` forms the `p x K` score-process matrix `U(t_k) = X_cen' s_bar(., t_k)`, its full `(pK) x (pK)` empirical sandwich covariance ([bcf_score_test.R:527–541](R/bcf_score_test.R#L527-L541)), per-gridpoint quadratic forms `T_k = U_k' V_k^{-1} U_k`, and three functionals:

- `wass`: `sum_k dt_k * sqrt(T_k)` — an L1 norm in `t` of the standardized process;
- `cvm`: `mean_k T_k` — because the grid sits at residual quantiles, this approximates an integral against `dF`, i.e., a genuine Cramér–von Mises weighting;
- `max`: max standardized |U| over all `p*K` coordinates.

Null distributions are obtained by simulating `N(0, Sigma_hat)` ([bcf_score_test.R:562–586](R/bcf_score_test.R#L562-L586)).

---

## 3. Methodological underpinnings and closest relatives in the literature

### 3.1 The mean test is an orthogonalized interaction score test

The moment `E[(Z - pi(x)) * (y - mu(x) - alpha (Z - pi(x))) * g(x)] = 0` for all `g` is the Robinson (1988) partially-linear-model orthogonality condition, the same one underlying double/debiased machine learning (Chernozhukov et al., 2018) and the R-learner (Nie & Wager, 2021). Testing it against `g(x) = x_j` and pairwise products is exactly testing that the **best linear predictor of the CATE** in the chosen basis is zero — the BLP object of Chernozhukov, Demirer, Duflo & Fernández-Val's generic-ML framework. Three properties follow from this framing and are the reason the construction can work at all:

- **First-order insensitivity to `mu`-error** because `E[Z_cen | x] = 0`, and to `pi`-error because `E[e | x, Z] = 0` — the familiar product-bias (double-robustness) structure: the score's bias is `O(||mu_hat - mu|| * ||pi_hat - pi||)`.
- **The nuisance may be Bayesian.** Nothing in the frequentist analysis of the score requires `mu_hat` to be an M-estimator; a posterior mean is a fine plug-in, and orthogonality is what neutralizes its (slow, adaptive) convergence rate.
- **The implied alternative is a projection.** Power is only against alternatives whose projection on `span(X_cen)` is nonzero; a purely non-linear `tau(x)` orthogonal to the basis is invisible. This is a design choice, not a bug, but it should be documented.

**Closest published relatives.**
- Crump, Hotz, Imbens & Mitnik (2008, *REStat*) test "no heterogeneity in CATE by covariates" with quadratic forms in sieve regressions — the same null, tested by a non-orthogonalized sieve GLS statistic. The present function can be read as a Bayesian-nuisance, orthogonalized, sandwich-robust version of their second test.
- Ding, Feller & Miratrix (2016 *JRSS-B*; 2019 *JASA*) decompose treatment-effect variation into systematic and idiosyncratic parts and give randomization-based omnibus tests for systematic variation — the same estimand family under randomization inference.
- The quadratic form `T' V^{-1} T` with many interaction columns is structurally a **variance-component score test** (Lin 1997; SKAT, Wu et al. 2011). That literature deliberately avoids inverting `V` when `p` is large, using weighted quadratic forms with mixture-of-chi-squared nulls — the natural upgrade path if `p_valid` grows (see D2/S9).
- The simulated max-|Z| test corresponds to the high-dimensional Gaussian multiplier bootstrap of Chernozhukov, Chetverikov & Kato (2013), which justifies max-type statistics even with `p >> n`.

### 3.2 Rao-Blackwellization is a plug-in p-value at the posterior mean — and that is fine *because* the score is orthogonal

Because `s_i` is linear in the residual, averaging `s_i` over draws equals plugging the posterior means `mu_bar`, `alpha_bar` into the score: the "Rao-Blackwellized" p-value is precisely the **plug-in p-value** studied by Robins, van der Vaart & Ventura (2000). Their result: plug-in (and posterior-predictive) p-values are asymptotically uniform when the statistic's asymptotic distribution is insensitive to the nuisance at the truth, and conservative otherwise. Orthogonality delivers exactly that insensitivity, so the combination *(orthogonal score) + (plug-in at posterior mean) + (self-normalized sandwich)* is a coherent recipe, not an ad-hoc mash-up of paradigms. Two caveats:

- The self-normalization argument (numerator and denominator both scale with any uniform shrinkage of `s_bar`) protects the chi-squared statistic against *uniform* posterior shrinkage but not against *x-dependent* shrinkage, which BART's adaptive regularization can produce.
- In the non-RB mode the function returns a posterior *distribution* of p-values with no combination rule. The mean of per-draw p-values is a posterior-predictive-flavored object and is **not** uniform under the null (Robins et al. 2000; Meng 1994); the documentation should warn against averaging them.

For the distributional test the indicator is nonlinear in the residual, so the RB average is not a plug-in but a **posterior-smoothed ECDF score** — each indicator is convolved with the posterior of `mu, alpha`, a kernel-smoothing effect with data-driven bandwidth. Since the covariance is estimated from the *same* smoothed per-observation scores, self-normalization carries over.

### 3.3 The distributional test is a residual-marked empirical process, and the `Z - pihat` mark is what evades Khmaladze

The object `U(t) = sum_i x_i Z_cen_i (1{ehat_i <= t} - Fhat(t))` is a **marked empirical process** in the sense of Stute (1997) and Escanciano (2006), with mark `x * (Z - pihat)`; it is essentially the statistic of Delgado & González-Manteiga (2001) for testing significance of a variable (here, the `Z`-by-`x` interaction) in nonparametric regression, transplanted to the distributional scale.

The classical obstacle for such processes is the **Durbin problem** (Durbin 1973): indicators evaluated at *estimated* residuals have a different null law than the ideal process, and the standard fixes are Khmaladze's (1981) martingale transform or a (parametric/wild) bootstrap. The published tests closest to this setting — Chung & Olivares' permutation tests for treatment-effect heterogeneity via the ECDF (2021, *J. Econometrics*) and via the quantile process (2025, *J. Applied Econometrics*), and Andrews' (1997) conditional Kolmogorov test — all confront this head-on with a martingale transform or a parametric bootstrap.

The present function takes a third route that deserves to be stated explicitly (it is nowhere in the comments): **the mean-zero mark `Z - pi(x)` orthogonalizes the process against residual-estimation error to first order.** Perturbing `mu` by `dm(x)` perturbs `E[U(t)]` by `sum_i x_i Z_cen_i f_i(t) dm(x_i)`, which has zero expectation when `E[Z_cen | x] = 0`. So, with a correct propensity, the estimated-parameter drift that plagues residual ECDF tests vanishes at first order and the empirical covariance of the observed scores is a legitimate variance estimate — no Khmaladze transform needed. The flip side is finding D1: this shield is only as good as the propensity, and — unlike the mean test — there is no outcome-model backup.

Finally, simulating `N(0, Sigma_hat)` with `Sigma_hat = sum_i (x_i s_bar_i(.)) (x_i s_bar_i(.))'` is distributionally identical to the **Gaussian multiplier (wild) bootstrap** `sum_i g_i x_i s_bar_i(.)`, `g_i ~ N(0,1)` — the standard device for such processes (and the memory-light way to implement it; see D2).

### 3.4 Novelty assessment

I found no published work that runs a frequentist score test *inside* a BART/BCF Gibbs sampler with Rao-Blackwellized scores — the closest neighbors are (a) frequentist: Crump et al. (2008), Ding–Feller–Miratrix, Chung–Olivares, generic-ML BLP tests; (b) Bayesian: posterior summaries/credible intervals for `tau(x)` in BCF (Hahn, Murray & Carvalho 2020) and posterior-predictive checks, which Robins et al. (2000) show are conservative. The hybrid here appears genuinely novel, and its two load-bearing walls are the orthogonality of the score and the plug-in p-value theory — both of which hold, with the caveats below.

---

## 4. Findings — both settings

**S1. [High] `rfx_group_ids_train` / `rfx_basis_train` are accepted, documented, and silently ignored.** The signature ([bcf_score_test.R:21](R/bcf_score_test.R#L21)) and roxygen advertise random effects; the sampler never touches them, and the sandwich treats observations as independent. With clustered data both tests will be anti-conservative (within-cluster correlation inflates `Var(T)` beyond `sum_i x_i x_i' s_i^2`). Either implement the RFX block (and a cluster-robust sandwich `sum_g (sum_{i in g} x_i s_i)(...)'`), or error out when these arguments are non-NULL.

**S2. [Low, bug] The `sigma^2` Gibbs update hard-codes the prior.** [bcf_score_test.R:262–263](R/bcf_score_test.R#L262-L263) uses `0.001 + n/2` and `0.001 + SSR/2`, ignoring the documented `sigma2_global_shape = 1`, `sigma2_global_scale = 0.001` defaults in `general_params`. Diffuse either way, but the exposed knobs do nothing.

**S3. [Low] `keep_every`, `keep_burnin`, `keep_gfr`, `num_chains`-related outputs.** `keep_every`/`keep_burnin`/`keep_gfr` are accepted but unused (no thinning). `mu_hat_train` ([bcf_score_test.R:357](R/bcf_score_test.R#L357)) is computed from `forest_samples_mu` of the **last chain only** — for `num_chains > 1`, earlier chains' forests are discarded without notice.

**S4. [Cosmetic] `mvtnorm` is required ([bcf_score_test.R:31](R/bcf_score_test.R#L31)) but never called** — `compute_pmvnorm_safe` is misnamed; it simulates rather than calling `pmvnorm`. Drop the dependency or the name.

**S5. [Medium] Internal propensity estimation is fragile.** [bcf_score_test.R:71–84](R/bcf_score_test.R#L71-L84) fits a *Gaussian-likelihood* BART to the binary `Z` using only `num_gfr` (default 5) grow-from-root draws, averages the last 80%, and applies no clipping to (0,1). Predictions outside [0,1] are possible and overlap violations are unhandled; and since both tests' validity leans on the propensity (entirely so for the distributional test — D1), this is the weakest link in the default pipeline. Recommend `probit_outcome_model = TRUE` for the internal fit (or at least `pmin(pmax(., eps), 1-eps)` clipping), more draws, and a warning when `propensity_train` is internal.

**S6. [Medium] No overlap/degeneracy guards.** If `Z` is constant, or `sum(Z_cen^2) ~ 0`, the `alpha` update degenerates to the prior and the score collapses without an informative error.

**S7. [Medium] No cross-fitting.** `mu` is fit on the same observations whose residuals feed the score. Orthogonality removes the first-order effect, and heavy BART regularization is the same defense BCF itself relies on, but the own-observation overfitting direction (residuals partially absorbing `Z_cen`-correlated noise) shrinks the score under the null → conservatism and power loss, unquantified. DML resolves this with cross-fitting; a K-fold "cross-fitted posterior residual" variant would be the principled upgrade if size accuracy matters more than compute.

**S8. [Medium, observational designs only] Incomplete projection of the `alpha`-direction.** The statistic uses residuals with `alpha` estimated; to first order `T_j` picks up `-(alpha_hat - alpha) * sum_i x_ij Z_cen_i^2`, and `E[x_j Z_cen^2] = Cov(x_j, pi(x)(1 - pi(x)))`, which vanishes under constant propensity but not otherwise. This term is the same order as `T_j` itself, is *common across observations* (so the i.i.d. sandwich cannot see it), and its omission makes the naive variance an over-estimate → conservative. The clean fix is to residualize each column `x_j Z_cen` on `Z_cen` (and on 1, which the current mean-centering of `s_i` already does) before forming `T` — i.e., use the efficient score. The same term (with `f_i(t)` weights) appears in the distributional test.

**S9. [Medium] No guard on `p_valid` relative to `n`.** With all pairwise interactions, `p_valid = O(p^2)` and only the QR rank (≤ n) caps it. The chi-squared calibration of `T' V^{-1} T` needs `p_valid << n`; the sandwich is HC0 (no leverage correction), which is anti-conservative in small samples with many columns. Consider warning when `p_valid > n/10`, an HC2/HC3-style correction, or a SKAT-style weighted quadratic form that avoids inverting `V`.

## 5. Findings — per setting

### 5.1 Mean setting

**M1. [Low] Centering `s_i` before the sandwich is right, and `T` is unaffected by it** (columns of `X_cen` have mean zero) — this is correct but worth a comment: the centering only matters through `Var(T)`.

**M2. [Info] The implied alternative is the linear+pairwise-interaction projection of `tau(x)`.** Users should know the test has no power against heterogeneity orthogonal to that basis (e.g., pure threshold effects in one covariate can load mostly on the linear term, but sinusoidal heterogeneity may not). This is standard for BLP-type tests but not stated in the docs.

**M3. [Low] Non-RB mode emits per-draw p-values with no combination guidance** ([bcf_score_test.R:286–292](R/bcf_score_test.R#L286-L292)); averaging them is invalid (see §3.2). Either document a valid summary or drop the mode.

**M4. [Low] `ginv` fallback keeps `df = p_valid`** ([bcf_score_test.R:473–480](R/bcf_score_test.R#L473-L480)). If `V` is materially rank-deficient (many near-zero `s_i`), the quadratic form concentrates on a lower-dimensional space and `chi^2_{p_valid}` overstates the df → conservative. Using `df = rank(V)` (from the pivoted Cholesky/eigen already computed) would be more accurate.

### 5.2 Distributional setting

**D1. [High, conceptual] Single robustness.** The indicator scores are centered by the *marginal* ECDF `Fhat(t)`; if the error distribution varies with `x` (heteroskedasticity, skew changing with covariates — precisely the settings where a distributional test is interesting), then `E[1{e_i <= t} | x_i] != F(t)` and the mean-zero property of the score rests solely on `E[Z_cen | x] = 0`. Misestimated propensity + x-dependent error law ⇒ `E[U(t)] != 0` under the null ⇒ size distortion that no amount of data fixes. The mean test does not share this: its bias is a product of the two nuisance errors. Recommended mitigations, in increasing ambition: (i) document the assumption prominently; (ii) in the internal-propensity path, strengthen S5; (iii) center indicators by a *conditional* CDF estimate (e.g., Gaussian `Phi((t - 0)/sigma_i)` from a heteroskedastic variance forest, or distributional regression), restoring a product-bias structure.

**D2. [Medium] `(pK) x (pK)` empirical covariance with rank ≤ n.** With defaults `K = 50` and interactions on (`p_valid` = 15 already for 5 continuous covariates), `pK = 750` exceeds typical `n`. `Sigma_hat` then has rank ≤ n; the 1e-8 ridge makes `chol` succeed numerically and the simulated process lives on a degenerate subspace whose geometry is estimation noise. Consequences by statistic: the **max** statistic is fine in principle (this is exactly the Chernozhukov–Chetverikov–Kato multiplier bootstrap regime); the **CvM/Wasserstein** statistics additionally invert each `p x p` block `V_k` — fine while `p_valid << n`, but the *joint* law across `k` (which the simulation uses) is poorly estimated when `pK >> n`. Also note the memory/time cost: `tilde_X` is `n x pK` and `chol` is `O((pK)^3)`. Concrete improvements: default `K` down to ~20; simulate via the multiplier form `U_sim = t(tilde_X) %*% G` (`G` an `n x n_sim` standard normal matrix), which never forms `Sigma_hat`, is exact for the same law, and costs `O(n * pK * n_sim)`; warn when `p_valid * K > n`.

**D3. [Low] Grid from a single posterior draw.** `t_grid` is fixed from the first retained draw's residuals ([bcf_score_test.R:296–299](R/bcf_score_test.R#L296-L299)). Valid (any fixed grid is), but a single atypical draw yields an inefficient grid. Pooling residuals from, say, the last few burn-in draws would stabilize it at no conceptual cost.

**D4. [Low, naming] "Wasserstein" is a misnomer.** `sum_k dt_k sqrt(T_k)` is an L1-in-`t` norm of the *standardized* (Mahalanobis) process — closer in spirit to an Anderson–Darling-weighted L1 statistic than to a Wasserstein distance between conditional distributions (which would integrate the *unstandardized* CDF difference). The `cvm` name, by contrast, is well-earned: the quantile-placed grid makes `mean_k T_k` an integral against `dF`. Suggest renaming or documenting.

**D5. [Info] Residual-scale drift across draws.** Indicators use raw (standardized-y) residuals while `sigma^2` varies over draws; the RB average therefore mixes slightly different scales. Harmless for validity (covariance is empirical) with a small power dilution; standardizing residuals by the per-draw `sigma` before gridding would tighten this.

### 5.3 Things done right (worth keeping)

- Orthogonalized score with propensity centering — the single most important design decision (§3.1, §3.3).
- QR-pivoted rank reduction of the design ([bcf_score_test.R:449–455](R/bcf_score_test.R#L449-L455)) — makes the statistic invariant to dummy traps and collinearity.
- Self-normalized RB statistic: numerator and covariance built from the *same* averaged scores.
- Huber–White sandwich rather than the model-based `sigma^2 X'X` — robust to heteroskedasticity and to `Var(Z | x)` variation.
- Simulation from `N(0, Sigma_hat)` = Gaussian multiplier bootstrap; the max-type variant has high-dimensional guarantees.
- Quantile-placed grid giving a principled CvM weighting for free.
- Guarding `test_type = "distributional"` behind `use_rao_blackwell = TRUE` — per-draw ECDF p-values would be both expensive and statistically dubious.

---

## 6. Empirical calibration check

**Design.** Run 2026-08-06 with the working-tree code sourced over stochtree 0.1.1. DGP: `n = 250`, 5 iid U(0,1) covariates, RCT with known `pi = 0.5` passed as `propensity_train`, `mu(x) = sin(pi x1) + 2(x2 - .5)^2 + x3`, `eps ~ N(0,1)`; `num_gfr = 5`, `num_burnin = 100`, `num_mcmc = 200`, all other settings default. Note `p_valid = 15` (5 linear + 10 interaction columns), so the distributional test runs with `pK = 750 > n = 250` — deliberately exercising finding D2. Three scenarios:

- **null** — constant effect `tau = 1` (40 replicates per test type);
- **mean_alt** — linear heterogeneity `tau(x) = 1 + 2 x1` (20 reps for the matched test, 12 for the cross test);
- **dist_alt** — constant mean effect `tau = 1` but treatment doubles the error SD where `x1 > 0.5` (variance-only heterogeneity; 20 reps matched, 12 cross).

**Results** (rejection rate at nominal level, median and minimum p across replicates):

| Scenario | Test | Statistic | rej. @ .05 | rej. @ .10 | median p | min p | reps |
|---|---|---|---|---|---|---|---|
| null | mean | quad (chi-sq) | 0.000 | 0.050 | 0.43 | 0.080 | 40 |
| null | mean | max | 0.100 | 0.150 | 0.54 | 0.033 | 40 |
| null | distributional | cvm | 0.000 | 0.025 | 0.58 | 0.096 | 40 |
| null | distributional | wass | 0.000 | 0.025 | 0.51 | 0.082 | 40 |
| null | distributional | max | 0.000 | 0.025 | 0.56 | 0.098 | 40 |
| mean_alt | mean | quad | 0.650 | 0.750 | 0.02 | 0.000 | 20 |
| mean_alt | mean | max | 0.900 | 0.950 | 0.002 | 0.000 | 20 |
| mean_alt | distributional | cvm | 0.667 | 0.750 | 0.035 | 0.000 | 12 |
| mean_alt | distributional | wass | 0.333 | 0.583 | 0.093 | 0.003 | 12 |
| mean_alt | distributional | max | 0.750 | 0.750 | 0.005 | 0.000 | 12 |
| dist_alt | distributional | cvm | 0.100 | 0.100 | 0.34 | 0.010 | 20 |
| dist_alt | distributional | wass | 0.050 | 0.100 | 0.39 | 0.023 | 20 |
| dist_alt | distributional | max | 0.050 | 0.200 | 0.29 | 0.018 | 20 |
| dist_alt | mean | quad | 0.083 | 0.167 | 0.56 | 0.034 | 12 |
| dist_alt | mean | max | 0.083 | 0.083 | 0.54 | 0.008 | 12 |

**Reading of the results** (with the caveat that 12–40 replicates give Monte Carlo SEs of roughly 0.03–0.06 on these rates):

1. **Both tests are conservative under the null, as predicted.** The mean quad statistic and *all three* distributional statistics produced *no* p-value below 0.08 in 40 null replicates each (under uniformity, `P(min p > 0.08) = 0.92^40`, about 3.5%, for a single statistic — individually suggestive, and the same pattern across statistics and both test types). This matches the direction predicted by S7 (no cross-fitting: in-sample BART residuals partially absorb `Z_cen`-correlated noise and shrink the score) and, for the distributional test, D2 (rank-degenerate `pK > n` covariance). The mean max statistic is the exception (10% at nominal 5%), within Monte Carlo noise of nominal.
2. **The mean test has good power against linear heterogeneity** (`tau = 1 + 2 x1`): 90% for the max statistic, 65% for quad. The max statistic dominates here because the alternative loads on a single design column — consistent with the sparse-alternative logic of max-type tests (CCK).
3. **The distributional test also detects the mean alternative** (67–75% for cvm/max) — expected, since a location shift moves the whole conditional CDF — but it **nearly fails against the variance-only alternative it is motivated by**: 5–10% rejection at nominal 5%, i.e., barely above its own (conservative) size, despite a large effect (doubling the error SD for half the covariate space). The likely mechanism is the conservatism in (1) compounded by signal dilution across 750 score coordinates of which only the `x1`-by-tail cells carry signal. This is the single most actionable empirical finding: with default `num_grid_points = 50` and interactions enabled, the distributional test as configured has little practical power at `n = 250`. Reducing `K`, restricting the design (`interaction_rule = "none"` for the distributional test), or the block-diagonal/multiplier changes in D2 are the levers.
4. **Specificity is as designed:** the mean test stays at its size level under the variance alternative (8%) — it tests means only.
5. Caveats: single DGP family, one sample size, RCT with *known* propensity supplied (so the single-robustness concern D1 and the propensity-estimation fragility S5 are deliberately *not* stressed here), and modest replicate counts chosen for runtime (~20–30 s per fit).

---

## 7. Recommendations (ranked)

1. Error out (or implement) when `rfx_group_ids_train` is supplied (S1).
2. Document the distributional test's reliance on a correct propensity, and harden the internal propensity path: probit BART, clipping, more draws (D1 + S5).
3. Reduce default `num_grid_points` to ~20 and switch the null simulation to the multiplier form `t(tilde_X) %*% G`; warn when `p_valid * K > n` (D2). §6 shows the practical stakes: at `n = 250` with the defaults, the distributional test had 5–10% power against a large variance-heterogeneity effect.
4. Warn when `p_valid > n/10`; consider a SKAT-style weighted quadratic form or HC2/HC3 correction for large designs (S9).
5. Partial out the `Z_cen` direction from each score column (efficient score) to remove the observational-design conservatism (S8).
6. Fix the hard-coded `sigma^2` prior; honor or remove `keep_every`/`keep_burnin`/`keep_gfr`; compute `mu_hat_train` across chains; drop the `mvtnorm` requirement; rename or re-document the "Wasserstein" statistic (S2, S3, S4, D4).
7. Document that the mean test targets the BLP of the CATE in the linear+interaction basis, and that non-RB per-draw p-values must not be averaged (M2, M3).
8. Consider a cross-fitted variant for settings where exact size matters (S7).

---

## 8. Addendum (2026-08-07): implemented improvements and verification

All improvements from §7 that concern the tests themselves are now implemented in `R/bcf_score_test.R` as **opt-in options whose defaults reproduce the previous behavior exactly** (verified: old vs new code on the same seed agree to machine precision for the mean and distributional defaults). New capabilities:

- **`test_type = "variance"`** — the targeted scale-heterogeneity score `s_i = Z_cen_i (e_i^2 - mean(e^2))`, sharing the mean test's quad/max machinery and its double robustness.
- **`test_type = "smooth"`** — Neyman smooth test on `num_smooth_components` (1–6) orthonormal Legendre polynomials of the residual rank (component 1 = location, 2 = scale, 3+ = shape), returning an overall quadratic p-value, a max p-value, per-component chi-squared p-values, and a simulation-calibrated min-p combination.
- **`dist_interaction_rule`** — a lean score design (e.g. `"none"`) for the distributional/smooth tests, independent of the mean test's design.
- **`null_method = "multiplier"`** — Gaussian multiplier bootstrap (`t(X_s) %*% G`), the same law as the Cholesky path but without ever forming the `(pK)^2` covariance; **`"permutation"`** — mark-permutation null for all RB test types.
- **`centering = "model"`** (Gaussian-CDF centering, restoring a product-bias structure), **`standardize_residuals`**, **`grid_pool_draws`**, **`ad_weights`**, **`num_null_draws`**, **`propensity_clip`**, plus warnings when `p_valid > n/10` or `p_valid * K > n`.
- Bug fixes: rfx arguments now error instead of being silently ignored (S1); the `sigma^2` prior honors `sigma2_global_shape`/`scale` (S2); the unused `mvtnorm` requirement was dropped (S4).

**Mini-verification** (same DGP as §6; 10 replicates per cell, so Monte Carlo SEs are large — directional evidence only). Rejection rates at nominal 5% / 10%:

| Config | Statistic | null @.05 | null @.10 | dist_alt @.05 | dist_alt @.10 |
|---|---|---|---|---|---|
| variance | quad | 0.0 | 0.1 | 0.3 | 0.3 |
| variance | max | 0.0 | 0.0 | **0.6** | **0.8** |
| smooth (J=4, main effects, multiplier) | quad | 0.0 | 0.0 | 0.3 | 0.5 |
| smooth | max | 0.0 | 0.0 | 0.4 | 0.6 |
| smooth | min-p | 0.0 | 0.0 | 0.4 | 0.4 |
| smooth | component 2 (scale) | 0.0 | 0.0 | **0.5** | **0.6** |
| distributional, improved (K=20, main effects, multiplier, AD, standardized, pooled grid) | cvm | 0.0 | 0.1 | 0.1 | 0.1 |
| distributional, improved | wass | 0.1 | 0.1 | 0.1 | 0.4 |
| distributional, improved | max | 0.1 | 0.2 | 0.2 | 0.2 |

Compare the old defaults from §6: 5–10% power against the same variance alternative. The ordering matches the theory in §3 exactly: the **targeted variance score** recovers the most power (60–80%), the **smooth test's scale component** is close behind (50–60%) while remaining an omnibus test, and the improved omnibus ECDF test roughly doubles-to-quadruples the old power but remains the weakest — the indicator process spreads a concentrated scale signal too thinly even at `p*K = 100`. Null behavior stays conservative-to-nominal for all new configurations, and the improved distributional configuration no longer shows the empty lower tail of §6 (null p-values as low as 0.011 now occur). Practical guidance: use `"smooth"` (with min-p) as the default omnibus distributional screen, `"variance"` when scale heterogeneity is the concern, and reserve the ECDF test for K-specific diagnostics with `dist_interaction_rule = "none"` and `null_method = "multiplier"` or `"permutation"`.

Not yet implemented (deliberately): cross-fitting (S7), the efficient-score `Z_cen` projection (S8), conditional centering by a heteroskedastic variance forest (the full D1 fix — `centering = "model"` is the homoskedastic-Gaussian version), and multi-chain `mu_hat_train` (S3).

---

## 9. References

- Robinson, P. M. (1988). Root-N-consistent semiparametric regression. *Econometrica* 56(4), 931–954.
- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018). [Double/debiased machine learning for treatment and structural parameters](https://economics.mit.edu/sites/default/files/2022-08/2017.01%20Double%20DeBiased.pdf). *Econometrics Journal* 21(1), C1–C68.
- Nie, X., & Wager, S. (2021). Quasi-oracle estimation of heterogeneous treatment effects. *Biometrika* 108(2), 299–319.
- Chernozhukov, V., Demirer, M., Duflo, E., & Fernández-Val, I. (2025). [Fisher–Schultz Lecture: Generic machine learning inference on heterogeneous treatment effects in randomized experiments](https://arxiv.org/pdf/1712.04802). *Econometrica*. ([NBER w24678](https://www.nber.org/system/files/working_papers/w24678/w24678.pdf))
- Crump, R. K., Hotz, V. J., Imbens, G. W., & Mitnik, O. A. (2008). [Nonparametric tests for treatment effect heterogeneity](https://direct.mit.edu/rest/article-abstract/90/3/389/57732/Nonparametric-Tests-for-Treatment-Effect). *Review of Economics and Statistics* 90(3), 389–405. ([working paper](https://public.econ.duke.edu/~vjh3/working_papers/test_treat.pdf))
- Ding, P., Feller, A., & Miratrix, L. (2016). [Randomization inference for treatment effect variation](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/rssb.12124). *JRSS-B* 78(3), 655–671.
- Ding, P., Feller, A., & Miratrix, L. (2019). [Decomposing treatment effect variation](https://www.tandfonline.com/doi/abs/10.1080/01621459.2017.1407322). *JASA* 114(525), 304–317. ([arXiv:1605.06566](https://arxiv.org/abs/1605.06566))
- Robins, J. M., van der Vaart, A., & Ventura, V. (2000). [Asymptotic distribution of p values in composite null models](https://www.tandfonline.com/doi/abs/10.1080/01621459.2000.10474310). *JASA* 95(452), 1143–1156. ([pdf](https://statweb.rutgers.edu/ztan/material/robins-vandervaart-ventura.pdf))
- Meng, X.-L. (1994). Posterior predictive p-values. *Annals of Statistics* 22(3), 1142–1160.
- Andrews, D. W. K. (1997). [A conditional Kolmogorov test](https://www.econometricsociety.org/publications/econometrica/1997/09/01/conditional-kolmogorov-test). *Econometrica* 65(5), 1097–1128.
- Stute, W. (1997). [Nonparametric model checks for regression](https://projecteuclid.org/journals/annals-of-statistics/volume-25/issue-2/Nonparametric-model-checks-for-regression/10.1214/aos/1031833666.full). *Annals of Statistics* 25(2), 613–641.
- Delgado, M. A., & González-Manteiga, W. (2001). Significance testing in nonparametric regression based on the bootstrap. *Annals of Statistics* 29(5), 1469–1507.
- Escanciano, J. C. (2006). A consistent diagnostic test for regression models using projections. *Econometric Theory* 22(6), 1030–1051.
- Durbin, J. (1973). Weak convergence of the sample distribution function when parameters are estimated. *Annals of Statistics* 1(2), 279–290.
- Khmaladze, E. V. (1981). Martingale approach in the theory of goodness-of-fit tests. *Theory of Probability & Its Applications* 26(2), 240–257. (See also [martingale-transform GoF tests](https://arxiv.org/pdf/math/0406518).)
- Chung, E., & Olivares, M. (2021). [Permutation test for heterogeneous treatment effects with a nuisance parameter](https://www.sciencedirect.com/science/article/abs/pii/S0304407621001561). *Journal of Econometrics* 225(2), 148–174.
- Chung, E., & Olivares, M. (2025). [Quantile-based test for heterogeneous treatment effects](https://onlinelibrary.wiley.com/doi/full/10.1002/jae.3093). *Journal of Applied Econometrics*. ([pdf](https://mauolivares.github.io/files/QTE_PT.pdf))
- Chernozhukov, V., Chetverikov, D., & Kato, K. (2013). Gaussian approximations and multiplier bootstrap for maxima of sums of high-dimensional random vectors. *Annals of Statistics* 41(6), 2786–2819.
- Wu, M. C., Lee, S., Cai, T., Li, Y., Boehnke, M., & Lin, X. (2011). [Rare-variant association testing (SKAT)](https://www.biorxiv.org/content/10.1101/085639v1.full). *AJHG* 89(1), 82–93.
- Lin, X. (1997). Variance component testing in generalised linear models with random effects. *Biometrika* 84(2), 309–326.
- Hahn, P. R., Murray, J. S., & Carvalho, C. M. (2020). [Bayesian regression tree models for causal inference](https://arxiv.org/pdf/1706.09523). *Bayesian Analysis* 15(3), 965–1056.
