# Decisions

Design decisions made while working on the open issues, with the reason for each and where it stands. Newest first.

**Who decided:**
- **maintainer:** chosen in review.
- **principle:** settled by a numbered rule in `docs/Design Principles.rst`.
- **proposed:** made during implementation and not yet confirmed.

**Before asking the maintainer, check the principles.** If one settles the question, apply it and record it here.

---

## 2026-10-30

### Fixes behind the 0.26 test audit's strict xfails

| Decision | By | Reason | Status |
|---|---|---|---|
| `logrank` with nothing left to compare (no event at all) returns the documented no-test result, statistic 0 on `dof` 0 with p-value 1, rather than refusing. | principle 1 | It is the result the docstring already promises and every other "nothing to compare" case gives; a refusal would break callers (`pairwise_logrank`) that skip pairs with `dof` 0. | done |
| Knots given to `RoystonParmar.fit` and `RoystonParmarRegression` are sorted on entry; fewer than two, non-finite and repeated knots are refused. A saved model's knots are read back as stored (not re-sorted), so its coefficients keep their basis. | principle 2 | Knots in another order are the same spline only once sorted; the spline coefficients then mean the same thing whatever order a user typed. | done |
| A weighted Cox fit runs the unweighted fit's data test for a run-off (`runoff_direction`) where the Newton search did not converge, went far, or its information collapsed, and then says "No finite maximum" once (not also "did not converge") and sets `maximum = "no finite maximum"`, with `nan` standard errors for the runaway columns. | principle 13 | Positive weights change neither the risk sets nor who fails in them, so whether a direction runs off is the same as unweighted. | done |
| The penalised parametric fit writes an interval starting at the lower end of the baseline's support (`[0, r]`) as the left-censored row it is, for its objective only (the model keeps the data as given); a non-finite Hessian makes the maximum unverified rather than raising. | proposed | The likelihood is identical, and its Hessian at `F(0)` of a Weibull is `0 * log(0)**2`, nan. | done |
| A Royston-Parmar regression's `likelihood_ratio_test()` nests it in the univariate spline fit on the regression's own knots and scale; with a spline coefficient held by `fixed` it is refused (the univariate fit cannot hold one). | principle 2 | The default univariate knots are placed differently, so only the regression's give a nested null. | done |
| `fit_bayes(points=...)` takes a whole number of at least 2 (an integral float too); anything else, `bool` included, is refused naming `points`. | principle 2 | A grid needs two points a parameter to have a cell. | done |
| The accelerated-life fit polishes a verified answer by Newton's method with a Hessian from central differences of autograd's gradient, only where autograd's Hessian is not finite (a left truncation at the support's lower end, `tl = 0` for a Weibull) so that the existing Newton finish could not run. Each step must lower the objective and the answer must still be verified, so a fit already at its maximum is unchanged; the refactor snapshot is bit-identical. | maintainer (W-X1) | TNC stopped with a gradient of about 2e-4 left there, so fits that should reach one maximum (unit weights against none, counts against repeated rows) differed by 1e-5 in the parameters; after the polish they agree to about 4e-10. | done |
| `prior.around(m, between=(a, b))` refuses `m` outside the open interval `(a, b)`, naming `m` and `between` and pointing to widening `between` or `prior.between`. | maintainer (B24) | A belief "about m, almost surely between a and b" with m outside is a contradiction; it built a prior centred on the interval instead. | done |
| The Exponential-Gamma (Lomax) predictive `var()` is the closed form `b^2 a / ((a - 1)^2 (a - 2))`, infinite for a posterior shape `a <= 2` (the mean is infinite for `a <= 1`). | maintainer | As `mean()`; the quantile nodes gave a finite variance where it is infinite. | done |
| `BayesianParametric.from_draws` refuses draws at or outside a parameter's open bounds (as the PyMC helper does, whose own message, run first, still asks whether a transform is missing), counting them per parameter. | principle 2 | The model was built where the distribution is not defined. | done |

---

## 2026-10-29

### Bayesian fitting: offsets, zero inflation, power priors and data checks

| Decision | By | Reason | Status |
|---|---|---|---|
| With an offset or zero inflation, each posterior point's base `sf` and `df` are held to the base distribution's support in the Bayesian layer itself (`sf` 1 and `df` 0 below the point's offset; the zero-inflation rule for `x < 0` kept), in one shared step used by the predictive functions and `quantity_at("R")`. The plain path is untouched. | principle 1 | As `Parametric` clamps (#256). The shipped distributions already return 1 and 0 below their support, so this was latent there; the Bayesian layer no longer depends on it (a base that gives NaN, as a bare formula does, made `sf` 0 below the offset and `qf` loop forever). | done |
| `qf`'s root bracket stops at the predictive support's lower edge (the smallest offset plus the support edge, or 0 with zero inflation), is widened at most 200 times, and returns NaN with a warning that says why where it cannot be bracketed or the function is NaN. | principle 22 | A search with no end is never the right answer; NaN with a reason is. | done |
| `quantity_at(("q", p))` is 0 for `p <= f0` in a zero-inflated fit, as `Parametric.qf` (the mass sits at 0, not at the offset). NaN stays for `p` beyond the limited-failure proportion. | principle 1 | `quantile_cri` and `compare_to` turned the NaN into an infinite life. | done |
| `var()` takes the offset and zero-inflated forms: the posterior mean of each point's variance, `(1 - f0) var(Y) + f0 (1 - f0) (gamma + E[Y])^2`, plus the variance of the points' means. Infinite where the mean is (a limited failure population). `moment(n)` still takes the plain form only, and its refusal now points to `mean()` and `var()`. | proposed | `mean()` already took these forms; a variance is computable in closed form, so computing it beats refusing. Checked against integrating the predictive survival function. | done |
| `model.prior_check()` keeps the offset, with the fitted data setting its range; a flat offset is then refused as any flat parameter is. `prior.check(..., offset=True)` without data still refuses. | principle 2 | It raised or silently checked a model without the offset. | done |
| The checks on flat priors (no failures; every parameter flat needs a verified maximum) judge the new data together with the old data of every power prior (`prior.from_data`, an exact `as_prior()`), at full weight. A power-prior fit whose own data give no maximum-likelihood start starts from the maximum of the pooled data. | proposed | The old data determine the parameters as much as new data would; a zero-failure demonstration on top of earlier failures is the main use. Weight below 1 tempers the evidence but does not take it away. | done |
| An explicit `prior.flat()` counts as flat for the "every parameter flat" check and in the grid's message (`BoundPriors.improper_names`). | principle 12 | Writing the default out should not switch off its safeguard. | done |
| Bernoulli `fit_bayes` makes `Bernoulli.fit`'s checks: outcomes 0 or 1, no censoring, no truncation, no offset, `lfp` or `zi`. | principle 2 | Other values were dropped and censoring ignored silently. | done |
| `fit_bayes`, `update` and `prior.from_data` share one intake (`bayes_data`) that refuses weights (as #1008), covariates, event types, time-varying covariates and marker readings with `fit(data)`'s message; `update`'s `_combine` refuses them too and keeps the data's unit (refusing two different units). | principle 2 | Refused, not dropped, as `accepts_surpyval_data` does. | done |
| An exact `as_prior()` carries the parameters the earlier fit held with `fixed=` as `prior.fixed(v)`. The normal-summary `as_prior(param=..., widen=...)` returns a joint normal on the free parameters only, as before: give the fixed values with `fixed=` in the new fit. | proposed | The old data's likelihood was taken at those values. The summary form returns a joint prior, not a set of priors, so it cannot carry them without changing its type. | done |
| The prior interval used by the prior-data conflict warning is sorted, so an offset's (whose scale runs down from its upper bound) is not read backwards. | principle 22 | It warned of a conflict with every offset prior. | done |
| No changelog entry for these: Bayesian fitting is unreleased (v0.26); the feature's changelog line now lists what it refuses. | proposed | Fixes to an unreleased feature are not changes to released behaviour. | done |

### Weighted Cox review fixes

| Decision | By | Reason | Status |
|---|---|---|---|
| The weighted refusal of residuals, `check_ph` and the diagnostics sits in `proportional_hazards.diagnostics._require_cox` (`cox_fit_data`), the entry point every path shares: the model's methods, the module's functions and `diagnose()`. | principle 13, 22 | They take the rows as counted units; refused in one place, no path can skip it. | done |
| A weighted Cox fit moves its baseline to `Z = 0` with `baseline_at_origin`, as the unweighted fit, refusing a move that over- or underflows. Its `r` becomes the risk weight `sum n w exp(beta'Z)` over each risk set (the unweighted `r` is `sum n exp(beta'Z)`; Breslow's increment is `d / r` for both), moved with the baseline; it was the weighted number at risk, which is not moved by a change of origin. `r` is stored, not used by any prediction. | proposed | One meaning of `r` for every Cox fit; moving the number at risk by `exp(beta'centre)` would be wrong. | done |
| Aliasing in a weighted Cox fit uses `_cox_aliased` on the weighted information at 0, with the weighted counts `n w` for the columns' spread and the weighted events for its scale; the aliased columns are nan in `params`, `covariance()`, `robust_covariance()` and the summaries, and 0 in the predictions, as unweighted. | proposed | The same rule as the unweighted fit, on the information the weighted fit maximises (R's `coxph` gives NA either way). | done |
| A weighted Efron fit saves `baseline_method = "efron"` (`cox_ph._baseline_method`). No compatibility read for `"fleming-harrington"`: weights are new in v0.26, unreleased, so no saved model has it, and the label is not used in any calculation. | proposed | — | done |
| Royston-Parmar predictions sum the spline per query point with `utils.linalg.fixed_order_dot` (each row reduced alone), in the univariate model and the regressions' spline baseline, which the fits share. | principle 4 | A BLAS matrix-vector product sums a row differently depending on its place, so a permuted query changed results in the last bit on Linux x86-64 (as #1018 for the Bayesian predictive). The other models' predictions passed the same check at query lengths up to 64 on Linux. | done |

### Penalised regression

| Decision | By | Reason | Status |
|---|---|---|---|
| A new method, `fit_penalised(..., penalizer, l1_ratio, standardise)`, on `CoxPH` and the PH, AFT and PO fitters, rather than new arguments to `fit`. | proposed | `fit` keeps its signature; a penalised fit is a different estimator with different inference. | proposed |
| `penalizer` and `l1_ratio` are lifelines' names (glmnet's `lambda`, `alpha`), with glmnet's objective (mean negative log-likelihood plus the penalty); `N` is the sum of the counts. | principle 5, 16 | The established references; counts equal repetition. | proposed |
| `standardise=True` by default, dividing by the count-weighted sd with divisor `N` (glmnet's; lifelines uses `N - 1`); coefficients reported in the covariates' units. | principle 6, 16 | The penalty should not depend on units. | proposed |
| No covariance, standard errors, bounds or p-values for a penalised model: each raises, saying why and what to do; `summary()` shows `nan`. | every-bound rule, principle 17 | Penalised Wald intervals do not have their stated coverage. | proposed |
| `aic`/`bic` count the effective degrees of freedom `tr[H (H + P)^-1]` over the parameters not at 0 (Gray 1992). | proposed | The lasso's count of non-zero coefficients, and the usual ridge df. | proposed |
| `CoxPH.fit_penalised(..., strata=)`: the stratified partial likelihood, one set of coefficients and a baseline per stratum; `N` is the units over every stratum and the covariates are centred and standardised over all the rows. | proposed | glmnet's `stratifySurv`, which it matches to 1e-6; the penalty's scale should not change with how the units are split into strata. | done |
| Start-stop (delayed entry) rows in penalised Cox: `N` is the number of rows weighted by `n`, not of subjects, and the standardisation is over rows. | proposed | glmnet's convention for `Surv(start, stop, event)`, the reference it is checked against; documented, since a subject with many rows weighs more in the penalty's scale. | done |
| Penalised Cox refuses right truncation (`tr`, or a `t` with a finite upper bound), naming the parametric penalised fits, which take left, right and both. A `t` with no upper bound is read as `tl`. | principle 2 | The partial likelihood has no term for right truncation, as for `CoxPH.fit`. | done |
| Weights `w` are refused by `fit_penalised` (Cox and parametric), with a message, for now. Follow-up: support them with `N` the sum of `w * n` and `w`-weighted standardisation (so rescaling the weights changes nothing), checked against glmnet's Cox `weights`; the information criteria stay refused for weighted fits, as elsewhere. | proposed | Clean support needs the weighted likelihoods' own paths (the weighted Cox partial likelihood and its ties), a reference check and a decision on what the effective df means under weights; refusing is better than a penalty whose scale silently depends on how the weights are normalised. | open |
| A penalised model refuses the robust (sandwich) covariance and standard errors, `summary(robust=True)` and `likelihood_ratio_test`, each saying why and to refit unpenalised. | every-bound rule, principle 17 | The sandwich of a biased estimate is not its covariance, and the likelihood-ratio statistic of a penalised fit is not chi-squared. | done |
| The penalty does not depend on the time unit for the models given `fit_penalised` (Cox, PH, AFT, PO): their coefficients are log ratios. An additive hazards model's coefficients are rates (1/time), so a penalty there would; that is to be settled if it is given one. | proposed | Recorded so that extending it does not miss it. | done |
| Not built yet: a coefficient path over `penalizer` and a way to choose it. | proposed | Left for a later change. | open |

### Life tables: `life_table`

| Decision | By | Reason | Status |
|---|---|---|---|
| A function, `surpyval.life_table(x, c, n, tl, intervals)`, returning a DataFrame, not a fitted model class. | proposed | A life table is a tabulation with estimates beside it, as `at_risk_table` is; Kaplan-Meier is the model for the same data. | done |
| Intervals are `[start, end)`: an event or censoring on a boundary counts in the interval it starts; data beyond the last boundary goes in an open last interval, which has no density or hazard (as `KMsurv::lifetab`'s last interval). | proposed | Klein and Moeschberger's convention, and grouped data given at each interval's start falls in that interval. | done |
| A late entrant is counted in `entered` in the interval it enters and as half at risk there, as a withdrawal is; entry exactly on a boundary is at risk from that interval's start. | proposed | The actuarial assumption applied to entries; with no late entry the table is the classical one. | done |
| Left and interval censoring and right truncation are refused, pointing to `Turnbull`. | principle 2 | Their events are not known to fall in one interval. | done |

### Group comparisons and the global test

| Decision | By | Reason | Status |
|---|---|---|---|
| `pairwise_logrank` tests each pair on its own two groups' rows (as R's `survminer::pairwise_survdiff`), with `adjust="holm"` by default (R's `p.adjust` default); `"bonferroni"`, `"bh"` and `"none"` too. A pair with nothing to compare (dof 0) keeps p = 1 and is not counted as a test. | proposed | Holm controls the family-wise error and never rejects less than Bonferroni. | done |
| The fixed-time test is `sf_diff(model_a, model_b, x, transform="log-log")`, a dict like `rmst_diff`'s. The variance is the step function's at the last observed time at or before `x`, not interpolated between observed times. | proposed | The estimate is a step; an interpolated variance does not belong to it. | done |
| The Peto-Peto weight includes the current event time, prod(1 - d/(r+1)) (Klein and Moeschberger section 7.3); `t0` censors events after `t0` at `t0` and drops rows entering at or after it. | proposed | — | done |
| `likelihood_ratio_test()` returns a dict (`statistic`, `df`, `p_value`, both log-likelihoods). For parametric regression it is available for the built-in links of PH, AFT, PO and AH (whose coefficients at 0 leave the baseline), and refused for accelerated life and custom links. | proposed | A life model has no general "no effect" value. | done |
| `fit_tvc(strata=)` takes a label per row, so a subject may change stratum; missing labels are refused rather than dropped. | proposed | Dropping a row would cut a subject's path short. | done |

### Aalen's additive hazards model (`AalenAdditive`)

| Decision | By | Reason | Status |
|---|---|---|---|
| A model object, `AalenAdditive.fit(x, Z, c, n, tl, coef_penalty)` -> `AalenAdditiveModel`, beside the Lin-Ying `AdditiveHazards`; the diagnostics function `aalen_cumulative_coefficients` is unchanged and the unpenalised fit equals it bit for bit. | proposed | The estimator was already validated against `timereg::aalen`; the model adds prediction, bounds, tests and saving around it. | done |
| Predictions use the running maximum of `H = B_0(t) + B(t)'Z` from 0, floored at 0, as the Lin-Ying model does (#376); R's `timereg` and lifelines return the raw estimate, which is the same wherever it does not fall. | principle (valid values) | `sf` must stay in [0, 1] and never rise. | done |
| `coef_penalty` is a ridge penalty on the covariates only, scaled by each covariate's variance; the baseline is not penalised. lifelines' `coef_penalizer` also penalises its intercept, so the two differ. | proposed | A penalty that does not depend on the covariates' units keeps the covariate-scale invariance; shrinking the baseline hazard towards 0 has no meaning. | done |
| `cb` is Wald only, on `log H` with the optional-variation covariance of `B`; `method` other than `"wald"` is refused saying why (no likelihood; no bootstrap implemented). | principle (every bound) | The documented gap. | done |
| `constant_effects_test` is timereg's Kolmogorov-Smirnov and Cramér-von Mises test of time-invariant effects, its null simulated by resampling (`n_sim`, `random_state`); its own seeded study found it anti-conservative at n = 300 (size 0.08 in the study, 0.13-0.16 in a separate check; 0.05 at n = 1000), pinned as a strict xfail in `calibration/test_coverage_aalen.py` and stated in the docstring. | proposed | Reported as found, not tuned. A fix (a time range short of the last event, or a small-sample correction) is for the maintainer. | open |
| Left out: lifelines' `smoothing_penalizer`, and the refit-registry study for the case (its fixture's fit is not an additive model to draw from); the model has its own calibration study. | proposed | — | done |

### Spline-baseline (Royston-Parmar) regression

| Decision | By | Reason | Status |
|---|---|---|---|
| Built as the `PH` and `PO` families on a spline baseline distribution with fixed knots (`SplineBaseline`), not a new model class. | principle 14, 21 | The model then has every bound, `summary`, `fit_from_df`, formulas and saving as the other parametric regressions, by the same code. | proposed |
| `RoystonParmarRegression(df=3, scale="hazard", knots=None)` configures the fitter (its `fit` and `fit_from_df` keep the regression signature); `RoystonParmarPH` and `RoystonParmarPO` are the `df=3` instances. `RoystonParmar.fit` is unchanged. | principle 21 | `fit_from_df` cannot take model options. | proposed |
| Knots as the univariate fit places them (quantiles of the log event times), then held: bounds, bootstrap and likelihood-ratio searches are conditional on them, as in flexsurv. | principle 16 | — | proposed |
| On the odds scale the coefficients multiply the survival odds, as every SurPyval `PO`: flexsurv's odds-scale coefficients have the opposite sign. | principle 21 | One meaning of a PO coefficient in the package. | proposed |
| `scale="normal"` with covariates is refused, naming `LogNormalAFT` and the univariate fit. | proposed | Its covariate shift is neither PH nor PO; a probit-shift family would be new. | open |

### Robust standard errors on the parametric regressions

| Decision | By | Reason | Status |
|---|---|---|---|
| The names are the Cox model's: `robust_covariance(cluster=None)`, `robust_standard_errors(cluster=None)` and `summary(robust=True, cluster=...)`; not a `robust=` argument to `fit` (lifelines' form). | principle 21 | The same option has the same name everywhere, and the fit is the same either way. | proposed |
| HC0, no small-sample factor; counts are frequency weights (a row with `n = 3` is three units, each its own cluster without `cluster`, three members of its cluster with it). | principle 5, 16 | R's `sandwich` and lifelines use HC0; counts must equal repetition, as `CoxPH.robust_covariance` already reads them. | proposed |
| `cluster` is one label per row given to the fit, in that order; rows dropped for a missing covariate are matched and left out (the fit records which given row each fitted row is). | principle 3 | A label belongs to its row; the fit sorts and drops rows. | proposed |
| Only the standard errors and the summary change; `cb`, `param_cb` and `p_values` keep the model-based covariance. A `fit_tvc` model refuses (its likelihood is over histories, not rows), naming the bootstrap and likelihood-ratio bounds. | principle 24 | Simple by default; the robust form is an option beside it. | proposed |

### PyMC helpers (`surpyval.pymc`)

| Decision | By | Reason | Status |
|---|---|---|---|
| PyMC is never a dependency of SurPyval, not even an optional extra in `pyproject.toml`, and not a test dependency. `surpyval.pymc` holds helpers for people who already use PyMC; each imports `pymc` / `pytensor` inside the function and, where they are absent, raises an `ImportError` saying to install pymc. `import surpyval` (and `import surpyval.pymc`) never imports them (tested). Their tests skip where pytensor or PyMC is not importable. | maintainer | SurPyval stays light and its own Bayesian fitting uses its own engines; the helpers serve PyMC users without tying the package to PyMC's release cycle. | done |
| The likelihood is a `pm.Potential` built from per-family pytensor log density, log survival and log CDF in SurPyval's parameterisation and stable forms, not `pm.Censored`. Supported: Weibull, Exponential, LogNormal, Gamma, LogLogistic, Normal, Gumbel, Logistic, Rayleigh; any other is refused with the list. | proposed | `pm.Censored` covers right and left censoring only; one term covers intervals, truncation on both sides and counts, and equals `_log_likelihood` to 1e-10 (tested per family, row type and truncation). | done |
| The data go through `SurpyvalData` (built with `group_and_sort=False` from arrays) and the distribution's support check, so refusals are `fit`'s, word for word. A parameter is a scalar or one value per row in the data's row order; other shapes are refused, and a length unknown before the graph runs is checked by an assertion in it. | proposed | Per-row values are how covariates enter, and they must line up with the rows as the user gave them. | done |
| `prior_from_model(model, name=, weight=)` is a multivariate normal on the log of each positive parameter (a real-line one as is), the fit's covariance carried there by the delta method and divided by `weight` in (0, 1]; models with an offset, a limited-failure proportion or zero inflation are refused. `prior_from_spec` takes each prior as a SurPyval distribution, `(family, *params)` or `{"dist", "params"}`, always in SurPyval's parameter order. | proposed | The same scale as the fit's Wald bounds; SurPyval's order avoids PyMC's swapped Weibull names. | done |

### Distributional regression: covariates on every parameter

| Decision | By | Reason | Status |
|---|---|---|---|
| The name is `Distributional(dist)`, with prebuilt `WeibullDistributional` and the like; not `GAMLSS`, which would promise its penalised smoothers. | maintainer | — | done |
| A new family, `Distributional(dist)` (`DistributionalFitter`, `DistributionalRegressionModel`), in which every parameter of the distribution has its own linear predictor and link, `g_k(theta_k) = b_k0 + x_k' b_k`. Covariates on a shape parameter live here rather than as an option of the AFT fitter. | maintainer | One general model rather than one option per family; the AFT model is its covariates-on-the-scale special case, tested as such. | proposed |
| One covariate set per fit, `Z`; each parameter chooses columns by `covariates={"alpha": [...], "beta": [...]}`, or by `formulas={...}` per parameter in `fit_from_df` (`formula=` puts one formula on every parameter). With none named, every covariate is on every parameter; with some named, the others have an intercept only. | maintainer | The same rule as the joint models' `hazard_covariates`/`path_covariates`; predictions take one row per unit. The alternative default was the location only (gamlss's). | done |
| Default links from each parameter's range: identity for a location, log for a positive parameter; a log link may be put on a location (`links={"mu": "log"}`); an override that can leave the range (identity on a scale) is refused, naming the allowed links. No logit link yet: no supported distribution has a parameter in (0, 1). | principles 2, 12, 16 | gamlss's and flexsurv's defaults; a link that can leave the parameter space gives a likelihood that is not defined. | proposed |
| `params` are the distribution's parameters at `Z = 0` (or at the covariate means with `center=True`) on their natural scale, then each parameter's coefficients on its link scale, named `<parameter>_<column>`. | principle 21 | `params[:k_dist]` is a distribution, as in every regression; flexsurv's coefficient names. | proposed |
| Coefficients have the sign of the link (`alpha(z) = alpha e^{b z}`): a positive scale coefficient means longer life, the opposite of SurPyval's AFT sign, which the documentation shows side by side. | principle 16 | gamlss and flexsurv; in this model a coefficient moves a parameter, not time. | proposed |
| The fit's aliasing, units and no-finite-maximum checks are per parameter: a column is aliased within each parameter's block, each coefficient is judged in its own column's units, and one warning names the columns. | principles 6, 12, 13 | A coefficient per column per parameter. | proposed |
| A verified answer with a direction of near-zero curvature has its profile walked out along that direction and along each parameter in it; where the likelihood rises along one for at least 8 units (each parameter's size), never falling below the answer, the fit is "No finite maximum", naming those parameters. | principle 12, 13 | A group with no failures can have its scale and shape (or location and spread) run off together; the search stops on a plateau within 1e-9 nats of the supremum, where the one-parameter-at-a-time checks see a maximum. In the new fit only. | proposed |
| `qf`, `mean` and `random` use the distribution's closed forms at each row's parameters. | principle 9 | Exact and vectorised. | proposed |
| `parameter_cb(name, Z, method=...)` bounds a parameter at each row on its link's scale (Wald by the delta method, likelihood-ratio as the extremes over the region, as `cb` takes them, bootstrap BCa), mapped back through the link. | every-bound rule, principle 18 | A shape's bound stays above 0; at `Z = 0` it is `param_cb`'s bound on the parameter. | proposed |
| ExpoWeibull stays (covariates on any of its three parameters): it fits and verifies on simulated data with covariates on its scale. Uniform and Beta4 are refused (support depends on the parameters). | proposed | Design said "only if well behaved"; it is, at the sample sizes tried. | proposed |
| Not built: evaluation along a covariate path (`sf_tvc` and the rest; refused: with the shape moving there is no accumulated age), penalised smoothers (`bs()`/`cr()` regression splines with a fixed number of degrees of freedom work in the formulas; the maintainer put penalised splines in a later version), covariates on a limited-failure proportion (a cure-fraction regression), distributions whose support depends on their parameters. | proposed | Documented gaps; each refusal says what to use instead. | open |

### Sampling, propensity and censoring weights, `w` (stage 1)

| Decision | By | Reason | Status |
|---|---|---|---|
| Weights are a new argument, `w`, beside the counts `n`, which keep their meaning (whole numbers of identical units) and their check; a decimal `n` is still refused, and the message now points to `w`. A row may carry both (`n` identical units, each standing for `w`). Not a float `n`. | maintainer | One meaning per argument: whole-number survey weights put in `n` would silently get count semantics and intervals too narrow, and nothing in the numbers tells the two apart. Purely additive: no existing call changes. | done |
| Passing `w` switches the fit to a robust variance, with no option of the classical one: the infinitesimal jackknife for Kaplan-Meier, Nelson-Aalen and Fleming-Harrington (R's `survfit(..., weights = w, robust = TRUE)`), the sandwich `H^-1 (sum n w^2 s s') H^-1` for the parametric and regression fits (`survreg(..., robust = TRUE)`), the Lin-Wei sandwich for Cox (`coxph(..., robust = TRUE)`). | maintainer | The classical interval is never right for real weights; its coverage depends on the scale of the weights: a "95%" Greenwood interval covered about a third of the time with the weights multiplied by 10 (0.33 in 300 simulated samples of 200, against 0.87 with the weights as drawn). | done |
| `aic`, `bic`, `aic_c` and `fit_best` refuse weighted fits, saying why. `neg_ll` and `log_likelihood` stay (the pseudo-likelihood's value). | maintainer | A weighted fit maximises a pseudo-likelihood, which does not compare models. | done |
| Stage 1 is the Kaplan-Meier family, the parametric distributions (MLE), the parametric regressions (PH, AFT, PO, additive hazards, accelerated life) and Cox. Turnbull, competing risks, the log-rank tests, Cox baseline bounds and time-varying fits, and a weighted bootstrap are stage 2; every fitter without `w` refuses weighted data (`fit(data)` names those that take it). | maintainer | — | done |
| Every bound weights make invalid refuses with a `WeightedRefusal` (a `ValueError`) naming the bound to use: likelihood-ratio and parametric bootstrap bounds (Wald on the robust covariance instead), `band` and `bootstrap_cb` of the non-parametric estimates (`cb` instead), the rank-based plotting positions (`heuristic='Kaplan-Meier'`, `'Nelson-Aalen'` or `'Fleming-Harrington'`), MPP, MOM, MSE and MPS (MLE), Cox's residuals and `check_ph`, the exact tie methods and strata of a weighted Cox fit, and the record-based reshapes of weighted data (`censor_at`, `select`, `item`, ...; `data[rows]` keeps the weights). | maintainer (the every-bound rule) | A bound that looks right and is not is worse than a refusal. | done |
| With weights, `hess_inv` of a parametric model holds the robust covariance of the distribution's parameters, as `covariance()` does, not the inverse Hessian. | proposed | Every consumer of `hess_inv` (`param_cb`, the plots, packages built on SurPyval) then gets the valid variance; the plan kept `hess_inv` as the inverse Hessian, which would have left those consumers with the classical one. | done |
| Ties in a weighted Cox fit follow R's `coxph`: each of the `m` tied deaths (`m` the units, from the counts) at the deaths' mean weight in Efron's denominator. A weighted Cox fit has its own partial likelihood (`proportional_hazards._weighted_cox`); the unweighted fit is unchanged. Its `robust_covariance(cluster)` sums the weighted dfbetas within clusters. The `nan` the unweighted `robust_covariance` gave with fractional counts does not arise: weights are not counts. | proposed | R's convention is the reference (principle 16), and the unweighted likelihood's tie terms need whole-number deaths. | done |
| Rows are grouped only where their weights are equal too; `summary()` and the printouts give the effective sample size `(sum n w)^2 / sum n w^2`, and a fit warns below 20. | proposed | The sandwich needs each unit's own weight; the sandwich is poor with few effective units (principle 22). | done |
| A weighted model's dict (and weighted data's) is stamped schema 3, which older readers refuse; unweighted dicts keep schema 1 or 2. | principle 20 | A reader without weights would give the robust bounds the classical meaning. | done |
| A Kaplan-Meier `rmst` (and `mean_cb`) of weighted data takes the infinitesimal-jackknife variance of the area. R's `print(survfit(..., robust = TRUE))` reports `se(rmean)` from the classical formula, so the reference test checks it against an independent numerical jackknife instead. | proposed | — | done |
| The calibration study's seeds (26101, 26102, 26103), sizes (populations of 1,000, 1,000 replicates; 500 for Cox; IPTW samples of 600) and rules (`Z_TOL` Monte Carlo errors plus a slack of 0.02; the IPTW check one-sided) were fixed before its first run. | principle 17 | — | done |
| `fit_from_df` takes `w_col` (the univariate and parametric regression fitters); Cox's `fit_from_df` does not yet: use `CoxPH.fit(..., w=w)`. | proposed | Its frame handling drops rows before `fit` sees them. | gap |
| A weighted regression's `robust_covariance(cluster)` is the weighted sandwich: `covariance()` without `cluster`; with it the bread is the inverse weighted information and the meat sums each cluster's `n w s` (R's `survreg` with `cluster()`, `coxph` with `cluster()`). | proposed | The robust-SE code wrapped the already-robust covariance in a second sandwich and dropped the weights. | done |
| `likelihood_ratio_test()` refuses weighted fits (Wald tests on the robust covariance instead). A fit whose log-likelihood is clearly below its null model's (more than 1e-6 relative) is refused, not given a statistic of 0. | proposed (the every-bound rule) | The ratio of weighted pseudo-likelihoods is not chi-squared; a fit below a model nested in it did not reach its maximum. | done |
| `life_table` and the Bayesian and PyMC entry points refuse weighted data (stage 2). The at-risk tables count units, not weights; `sf_diff` and `rmst_diff` use the weighted estimates' robust variances. | proposed | They dropped the weights or read them as counts. | done |
| Weighted data's dict carries `"schema": 3`; unweighted data's has no `"schema"`, as before. `SurpyvalData.from_dict` checks the schema and reads `w` whatever it is. v0.25 does not check a data dict's schema, so it still drops the weights. | principle 20 | The decision above stamped models but `SurpyvalData.to_dict` was never stamped. | done |

### Bayesian fitting of the parametric distributions

| Decision | By | Reason | Status |
|---|---|---|---|
| SurPyval fits by Bayes' rule with engines of its own: closed forms where they exist (Exponential with a Gamma prior, a Normal's mean with its spread known, Bernoulli with a Beta prior), else a self-checking grid for one to three free parameters, with a normal (Laplace) approximation as an option. No probabilistic-programming library is a dependency, required or optional, and `fit_bayes` has no method that calls one; a sampler of SurPyval's own, for regression and pooling, is later work. | maintainer | Deterministic answers with no seed, no new dependency, and fast enough to run inline. | done |
| The entry points are `Distribution.fit_bayes(...)`, `fit_bayes(data)` with a `SurpyvalData`, and `fit_bayes_from_df(...)`; `fit` is not changed. All the code is in `surpyval/bayes/`; the distributions' base class has two delegating methods. | maintainer | Additive: existing fits are unaffected. | done |
| A Bayesian model's `sf`, `ff`, `df`, `hf`, `Hf`, `qf`, `mean` and `random` are the posterior predictive (a new unit's life with the parameter uncertainty averaged in); `params` are the posterior medians, and `at_params()` is the plug-in model there. | maintainer | A proper distribution, so every identity holds and anything that takes a model takes it. | done |
| A parameter with no prior is flat on its unconstrained scale (log for a positive parameter, logit for a proportion), allowed only where the data alone determine it: refused with no failure information; with every parameter flat, refused unless the maximum-likelihood fit's maximum is verified; and refused whenever the grid finds the posterior does not fall away (it has no finite mass). | maintainer | Never invents information; a reference posterior in one line. | done |
| Credible intervals are `cri(x, on=...)`, `quantile_cri(p)`, `mean_cri()` and `param_cri(name)`, with `alpha_ci` and `bound=` as everywhere; `cb`, `quantile_cb`, `mean_cb` and `param_cb` refuse and point to them. | maintainer | Different questions, different names. | done |
| Priors given on a parameter (`around`, `between`, any SurPyval distribution), on several (`MultivariateNormal`, `from_model`), on quantities (`on_quantities`: B-lives, `R_at`, the mean, the median, parameters by name) and from old data (`from_data`, a power prior). `around(m, between=(a, b))` is normal on the unconstrained scale with its ends at the 5% and 95% points (90% by `mass=`); its median is the interval's middle on that scale, and the summary says what was built. A prior's support must lie in its parameter's range (refused, or `truncate=True`). | maintainer (the worked examples) | — | done |
| `on_quantities` takes exactly one statement per free parameter and carries the quantities' density to the parameters by the change of variables (the Jacobian by central differences at every grid point). More statements than parameters (matching them as well as possible) is not built. | proposed | Exact where it is defined; matching over-determined statements is a separate piece of elicitation. | done |
| `from_model(model)` of a maximum-likelihood fit is a normal on the unconstrained scale with the fit's covariance carried there, divided by `weight` and widened by `widen` squared; refused where the fit's maximum is not verified. `from_model` of a Bayesian fit is its priors times its data's likelihood raised to `weight` (exact); with `param=` or `widen` it is a normal fitted to the posterior, and says so. | proposed | Exact where the data are at hand. | done |
| `update(x, c, ...)` refits with the original prior and the old and new data together (exact); `replace=True` takes the new records in place of the old. | maintainer | Updating in steps equals fitting at once. | done |
| `compare_to(other, on="qf", p=0.10)` is the posterior probability that this model's quantity exceeds the other's: two Bayesian fits as independent posteriors; a maximum-likelihood fit's estimate as normal on the log scale (natural scale for a probability) with its delta-method standard error, independent of this one. The result states which. | maintainer (the worked example) | — | done |
| The grid: on the unconstrained scale along the curvature's axes at the peak, stretched (`z = 3 sinh(u / 3)`) so cells are fine near the peak and coarse in long tails, widened until every face is below 1e-8 of the peak, and a second grid offset by half a cell. A summary's quantiles spread each point's mass over its cell and remove the spread's own bias by extrapolation from a two-cell spread. The error estimate is half the difference of the two grids' 2.5%, 50% and 97.5% points of every free parameter and the 10% quantile, as a fraction of the 95% interval's width; it must be at most 0.1% with one or two free parameters and 0.5% with three (refined up to 241 and 61 points a dimension; past that the fit refuses with the numbers). | proposed | 0.1% at three parameters needs grids too large to stay near a few seconds; the error is always reported. | done |
| Where the data disagree with a proper prior on one parameter (the posterior median outside the prior's 99% interval), the fit warns with both intervals and sides with neither. | proposed | Design principle 22. | done |
| `mean()` is infinite, with a warning, where parameter values with an infinite mean carry more than 1e-12 of the posterior; below that they are below what the grid resolves. | proposed | — | done |
| Conformance: registered as `Weibull[bayes]` (the grid) and `Exponential[bayes]` (the closed form, with its prior's time scale rescaled with the data). Not applicable, with reasons in the registry: `maximum` (not a maximisation), `convergence` (no optimiser to starve; the grid's own refusal is tested), and containment of the predictive hazard in `cri(on="hf")` (the predictive hazard is not the posterior mean of the hazard; checked against the posterior median instead). The intervals are not Wald (they close onto the posterior median). | proposed | Principle 18 as it applies to a posterior. | done |
| Validation (rules fixed before the first run): the vectorised likelihood equals the fitters' to 1e-10 for every continuous distribution and kind of data; the closed forms are checked in exact arithmetic and the grid matches them to 1e-4; update in steps and a power prior at weight 1 equal the fit of all the data (to the grid's tolerance where the start differs); with flat priors and 1,000 complete Weibull values the posterior median is within 0.02 standard errors of the maximum and the 95% interval within 2% of the likelihood-ratio interval; on a hard posterior (2 failures, 98 suspensions) the reported error is within a factor of 2 of the error against a 4x finer reference. | maintainer (the design) | — | done |
| Results of those rules, as found: the likelihood identity holds to 1e-16; the closed forms are exact; the grid matches them to 1e-4; update and the power prior agree; the peak is the maximum to 1e-6 and the 95% intervals are within 2% of the likelihood-ratio intervals; the reported error on the hard posterior is 2.7e-4 against an actual 4.7e-4 (within the factor of 2). One rule fails and is kept as a strict expected failure: beta's posterior median is 0.037 standard errors from the maximum at n = 1000 (the rule asks 0.02); the difference is the posterior's skewness, which shrinks as 1 / sqrt(n), so the rule was too tight for 1,000 values. | proposed | Reported, not loosened. | done |
| The calibration study's runs, as found. Run 1: 6 of 11 scenarios passed; the censored, truncated and few-failure scenarios failed (B10 coverage 0.97 to 0.978, alpha's ranks p = 0.00099, 14 truncated fits refused). Two causes. (a) The study set each replicate's censoring, inspection and truncation times from its true parameters (the 70% point, a third of the median), which tells the fit about the truth through the design and breaks the premise of the method; the times are now fixed per scenario, from the prior's central model, with the same seeds, replicates and rules. (b) The grid refused curved posteriors (late entry, wide prior): it is now laid again along the spread it finds (a fix in the fit, with a regression test). Run 3, after both: 9 of 11 pass. Two fail and are kept as strict expected failures: "Weibull complete, informative" (a scenario the design fault did not touch), B10's 95% coverage 0.978 against the rule's 0.95 +- 0.02, with p = 0.0094 for uniformity; on 1,000 fresh replicates it was 0.954, so it is taken as this seed set's excursion; and "Weibull left truncated", 1 of 500 replicates refused because the simulation drew an infinite time (a fault of the simulation). | proposed | Reported, not loosened. | done |
| Simulation-based calibration (Talts et al. 2018), fixed before its first run: 500 replicates per scenario; the true parameters drawn from the prior and the data simulated with seed 8100 + r, the 999 posterior draws with seed 8600 + r; the rank of the truth among the draws for each parameter and the B10 life, in 20 bins, must pass a chi-square test at the 0.001 level (a Bonferroni note: about 30 tests), and the 95% interval's coverage must be within 0.02 of 0.95; a replicate whose fit refuses fails its scenario. Scenarios: Weibull with 20 complete values (weakly informative and informative priors), with 30% and 80% Type I censoring, interval inspections, left truncation and about 3 failures in 40; LogNormal, Gamma and LogLogistic with 30% censoring; Exponential with a Gamma prior and 30% censoring. | proposed | Principle 17's coverage studies, for a posterior. | done (the validation pull request) |
| External reference: PyMC (NUTS, 4 x 10,000 draws) posteriors for a right-censored Weibull and an interval-, left-censored and left-truncated one, computed once in a separate environment and fixed in a test; the grid's medians and 2.5% / 97.5% points agree within 4 Monte Carlo standard errors (the rule fixed beforehand). R (brms) and Stan were not available, so they are not used. | proposed | No new dependency. | done |

### PyMC helpers (`surpyval.pymc`)

| Decision | By | Reason | Status |
|---|---|---|---|
| PyMC is never a dependency of SurPyval, not even an optional extra in `pyproject.toml`, and not a test dependency. `surpyval.pymc` holds helpers for people who already use PyMC; each imports `pymc` / `pytensor` inside the function and, where they are absent, raises an `ImportError` saying to install pymc. `import surpyval` (and `import surpyval.pymc`) never imports them (tested). Their tests skip where pytensor or PyMC is not importable. | maintainer | SurPyval stays light and its own Bayesian fitting uses its own engines; the helpers serve PyMC users without tying the package to PyMC's release cycle. | done |
| The likelihood is a `pm.Potential` built from per-family pytensor log density, log survival and log CDF in SurPyval's parameterisation and stable forms, not `pm.Censored`. Supported: Weibull, Exponential, LogNormal, Gamma, LogLogistic, Normal, Gumbel, Logistic, Rayleigh; any other is refused with the list. | proposed | `pm.Censored` covers right and left censoring only; one term covers intervals, truncation on both sides and counts, and equals `_log_likelihood` to 1e-10 (tested per family, row type and truncation). | done |
| The data go through `SurpyvalData` (built with `group_and_sort=False` from arrays) and the distribution's support check, so refusals are `fit`'s, word for word. A parameter is a scalar or one value per row in the data's row order; other shapes are refused, and a length unknown before the graph runs is checked by an assertion in it. | proposed | Per-row values are how covariates enter, and they must line up with the rows as the user gave them. | done |
| `prior_from_model(model, name=, weight=)` is a multivariate normal on the log of each positive parameter (a real-line one as is), the fit's covariance carried there by the delta method and divided by `weight` in (0, 1]; models with an offset, a limited-failure proportion or zero inflation are refused. `prior_from_spec` takes each prior as a SurPyval distribution, `(family, *params)` or `{"dist", "params"}`, always in SurPyval's parameter order. | proposed | The same scale as the fit's Wald bounds; SurPyval's order avoids PyMC's swapped Weibull names. | done |
| `to_surpyval(trace, dist, transform=, data=, priors=, **param_map)` returns `BayesianParametric.from_draws(...)`, not a class of its own. A parameter maps to a trace variable by name with integer indices (`"log_alpha[1]"`); unmapped parameters use the variable of their own name; `transform` is `"exp"`, `"log"`, `"inverse"` or a function. It reads an InferenceData/DataTree (`posterior`) or a MultiTrace, joins the chains, and refuses non-finite draws and draws outside a parameter's bounds (naming the likely missing transform). | proposed | One Bayesian model class in the package; the reference string is what a PyMC user already writes for a vector element. | done |

---

## 2026-10-17

### Joint models: separate covariates, interval data, dynamic prediction (#891)

| Decision | By | Reason | Status |
|---|---|---|---|
| Separate covariates for the hazard and the path are chosen by name from the one covariate set the records carry: `fit(..., hazard_covariates=[...], path_covariates=[...] or {path parameter: [...]})`, default every covariate in both (as v1). Not a second covariate array (`path_Z`, as the closed #966 had). This revisits v1's "no new words such as `path_covariates`": the capability was asked for, and these two arguments say what they do; the alternative was one mapping, `covariates={"site": "hazard", "load": ["b"]}`. | maintainer | An `ItemLifetime` has one `covariates`, so a second array would need a second place on the record; naming the columns keeps `fit(data)`, arrays and `fit_from_df` alike. The maintainer kept the names `hazard_covariates` and `path_covariates` (2026-10-17). | done |
| A covariate in neither part is refused; one in only some parts has no parameters elsewhere (no `nan`, no 0 entries): `b_load` without `a_load` or `load`. | proposed | A coefficient that is not estimated is not a parameter. | done |
| Saved models write `path_covariates` and `hazard_covariates` (`None` where every covariate is in both); a dict without them loads as every covariate in both. | principle 21 | Saved models keep working. | done |
| Interval and left censoring: `S(l) - S(u)` over the item's paths, the hazard integrated over `(l, u]` directly (not a difference of two cumulative hazards); a left censored item's interval starts at its entry. Right truncation divides by the population's `S(tl) - S(tr)`, as left truncation already divided by `S(tl)`. | proposed | The same conditioning v1 chose for late entry; the direct integral keeps a narrow late interval accurate. | done |
| One item's dynamic prediction is an object, `JointModel.predict(...)` -> `DynamicPrediction`, beside the array prediction `sf/ff/cb(x, Z, markers=, given=)`, which it uses and which is unchanged. Its `cb` is the model's (Wald on `log(-log sf)`); its `qf` brackets on a grid and refines by Newton steps on `log sf` (derivative `-hf`). | proposed | The object answers the one-asset questions (hazard, remaining life, smoothed marker, draws) without a second implementation. | done |
| `DynamicPrediction.to_dict(x=None, alpha_ci=0.05)` is strict JSON for a report or chart: the readings as `[age, value]` pairs, a grid from `age` to the 0.99 quantile (0.9, or ten scales on, where the survival stops short of it), `sf`/`ff`/`hf` with the two-sided Wald bounds on `sf`, remaining-life quantiles at 0.1, 0.5 and 0.9, and the marker's path with its interval from the first reading; non-finite values are `None` with the package's `"non_finite"` record. | proposed | The same convention as every other `to_dict`, so a reader can restore the values. | done |
| The calibration study's seeds (7300 + r, 7600 + r) and its rules (`check_bias` defaults, coverage slack 0.02) were fixed before its first run. | principle 17 | — | done |

## 2026-10-16

### Data classes: usability (#931–#940, #945; tracking #953)

| Decision | By | Reason | Status |
|---|---|---|---|
| `from_dataframe` names its columns `x_col`, `c_col`, `n_col`, `xl_col`, `xr_col`, `tl_col`, `tr_col`, `i_col`, `e_col`, `Z_cols` (the `fit` argument with `_col`), not `id=`, `time=`, `event=` as #936 sketched; a status column (1 = failed) is `event_col`. | principle 21 | Every DataFrame entry point names a column argument after the fit argument it fills. | done |
| `print(data)` is the repr (the size and a table of the first rows), as for an `EventHistoryData` or a DataFrame; `summary()` stays a method, not `__str__`. | proposed | `print(data)` shows the data; the docs (Conventions) rely on it. #934 suggested `__str__ = summary()`. | done |
| The likelihoods' split arrays (`x_o`, `mask_i`, `Z_t`, ...) and the recurrent likelihood pieces keep their names and stay public, left out of `dir()`; no underscore rename with deprecated aliases. | proposed | Some 50 names used across the fitters and the diagnostics; principle 21's one-release deprecation for each is more churn than the clutter costs. #934 suggested the rename. | done |
| `summary()` counts items (rows weighted by `n`) everywhere, and the heading reads "4 items, 3 rows" for both datasets. | proposed | #934: the counts disagreed between lines. | done |
| `for_event_type(["seal", "bearing"])` takes the types as one mode rather than refusing a list. | proposed | #932 allowed either; merging is what the list means. | done |
| `censor_at` / `enter_at` refuse a cut-off at or before 0 only where every time and truncation bound of the data is at or above 0; data on the real line (a Normal baseline) takes a negative cut-off. | proposed | A negative age is a silent error for ages from 0, and a valid time on the real line. | done |
| `select(ids=)` on data without ids is refused (rows are sorted and grouped, so their numbers would not be stable); `item(k)` by row number stays; `SurpyvalData(..., ids=)` gives rows ids. | proposed | #932's recommendation. | done |
| `fit(data)` takes the data's unit; `fit(data, unit=other)` is refused, as are records in different units in one `from_items`; nothing is converted (`to_unit` left for later). | #945 as written | — | done |
| `groupby(name)` refuses items without the attribute (naming them); `select(where={name: value})` leaves them out. | proposed | A group of "none" is a silent default; a selection by value is not. | done |
| A `SurpyvalData` or `EventHistoryData` dict is tagged `"class"` and read by `surpyval.from_dict` / `from_json` by that tag, as a model is by `"model"`. | #935 as written | — | done |
| `predict(..., given="age")` conditions on each running item's age by the ratio `S(t) / S(age)` from the model's own functions, for every lifetime model alike; a failed item gets `nan`; `qf` and `mean` are not conditioned. | proposed | With fixed covariates the ratio is the conditional survival under every model (the AFT subtlety of #962 is for covariates that change). | done |
| `EventHistoryData.from_dataframe` builds each item's `ItemHistory` (rather than arrays), so every check names the item; an item in `ends=` with no row cannot carry covariates and is refused with `Z_cols`. | proposed | — | done |
| `fit_from_surpyval_data` keeps accepting covariates it does not fit. | proposed | The regression fitters fit through it with the data's `Z`; refusing broke them. `fit(data)` refuses, as before. | done |

### Branches left behind

| Decision | By | Reason | Status |
|---|---|---|---|
| Test planning (#862–#865, #880), built on branch `claude/happy-einstein-9bpmln` on 14 Oct but never opened as a PR, is merged. | maintainer | Test planning is statistics, so it belongs in SurPyval. | merged (#965) |
| Leave `claude/roadmap-live-npv` (a ROADMAP section) and PR #858 (Reliafy links) unmerged. | maintainer | No code in either. | left as is |

### Diagnostics: `diagnose()` (#867–#875, #881)

| Decision | By | Reason | Status |
|---|---|---|---|
| The API is as designed in #867: `table`, `advice`, `plot()`, `ok`, `raise_if_failed()`, `to_dict()`, one registry of families. It can change later if needed. | maintainer | — | merged with the diagnostics PR |
| `check_ph()` computes the exact `cox.zph` score test (R's survival 3.0 and later). The old approximation is removed, and every place in the package that runs, quotes or tests the PH test agrees. | maintainer | "Use the better version. Make sure the package is consistent." The old values matched lifelines and pre-3.0 R, not current R. | merged with the diagnostics PR |
| The signature is `diagnose(alpha, columns, *, n_boot, random_state, **family_options)`. | principle 19, 21, 24 | 19: every method that draws takes `random_state`. 21: `n_boot` is the standard name. 24: family options (`use_stress=`, `t0=`) default to the simple behaviour. | done; recurrent events takes them too |
| Rows that choose between models, where the test rejecting is the good outcome, are screens: ok or warn, never fail. These are the recurrent trend and repair rows and the frailty variance row. | principle (framework rule: screens set ok/warn) | They are model-choice screens, not tests of an assumption. | done (Verdicts section) |
| Competing risks runs the Cox checks per cause, but not calibration. | proposed | A cause's exp(−H_k) is not the probability of being free of that cause when other causes compete, so there is nothing to calibrate it against. | merged with the diagnostics PR |
| Docs data typed from memory (Nelson's insulating fluid) must be checked against a published source, or replaced with simulated data. | principle 16, 23 | Quoted data and numbers must be checked. | done: checked against R's `Sleuth3::case0802`; one typo fixed (100.58 → 100.59) |
| The accelerated-life advice to "plan more stress levels" names `surpyval.test_planning`'s accelerated life test planning (#864), not "simulate a plan with `model.random`". | proposed | Test planning is now in SurPyval. | done |

### Partial-effects plots (#876)

| Decision | By | Reason | Status |
|---|---|---|---|
| Hold the other covariates at their means, including 0/1 indicators; a formula's categorical variable at its reference level; `reference=` overrides. | maintainer | Consistent with how the regressions centre their covariates. | merged with the partial-effects PR |

### Joint models (#891 v1)

| Decision | By | Reason | Status |
|---|---|---|---|
| Use the data classes' language: `markers={age: value}` on `ItemLifetime`, `fit(data)`, and prediction from a record, `sf(x, item)`. No new words such as "readings", `link=` or `path_covariates`. | maintainer | — | merged (joint-model PR) |
| `JointPH(dist, path_model)` takes any degradation `PathModel`. `WeibullJointPH = JointPH(Weibull, LinearPath)`. Parameter names come from the parts: `alpha`, `beta`, `marker_effect`; the path's own `a`, `b`, `c`; `a_<cov>`; `sd_a`, `corr_a_b`; `sd_noise`. | maintainer | The marker function has its own names, as a life model does. | merged (joint-model PR) |
| v1 records hold one plain value per age, and only on `ItemLifetime`. | maintainer | — | merged (joint-model PR) |
| Keep: `marker_effect` as the link's name; the array fit route; `unit=`; reshapes trim readings to the observation window; exact conditioning on entry for late entry; Wald-only `cb`; fixed quadrature settings; joining the shared conformance checks. | maintainer | — | merged (joint-model PR) |
| Centre the marker at the mean reading by default; `marker_reference=` sets a round value; `summary()` says which value `alpha` is at. | maintainer | Only `alpha`'s meaning changes. | merged (joint-model PR) |
| Array prediction `sf(x, Z=, markers=, given=)`. `sf(x, item)` builds exactly those arguments. With no readings and no `given`, it gives the new-unit (population) prediction. | maintainer | — | merged (joint-model PR) |
| Every path parameter varies between units by default; `varying=` chooses which. | maintainer | — | merged (joint-model PR) |
| Units differ on each parameter's own scale by default; `LinkedPathModel` (e.g. `{"b": "log"}`) gives a log scale. | principle 15, 24 | The same default as the degradation module; the alternative is an option. | done; two `LinkedPathModel` bugs fixed on the way |
| Keep `fit_from_df`. | principle 14, 21 | Every fitter has one, and the entry points agree. | merged (joint-model PR) |
| Run a 200-fleet recovery study per path model before merging. | principle 17 | Intervals must achieve their stated coverage; 24 fleets was too few to tell. | done: `scripts/studies/joint_recovery` |

### `fit_from_recurrent_data` refusals (#959)

| Decision | By | Reason | Status |
|---|---|---|---|
| Refuse what `fit(data)` refuses, in one shared check. What a fitter takes is read from its `fit` signature. | maintainer | One source of truth, so the two routes cannot disagree. | merged (#964) |

### Survival given an age (#962)

| Decision | By | Reason | Status |
|---|---|---|---|
| Under PH, PO, additive hazards and Cox, `given=g` needs the covariates from `g` only. AFT and accelerated life still need them from 0. | maintainer (option A) | The part before `g` divides out of S(x)/S(g) under these models, but not under AFT, whose age at `g` depends on it. This matches what fitting already allows for late entry. | merged (#964) |

---

## 2026-10-15

### Working method

| Decision | By |
|---|---|
| One issue at a time. Explain it with a minimal example, then the maintainer decides, then build. | maintainer |
| Before merging, show before/after results on the real code, and list anything done beyond what was agreed. | maintainer |
| A long plan is a map, not a commitment; re-plan after each issue. | maintainer |

### AFT on baselines defined below 0 (Normal, Gumbel, Logistic)

| Decision | By | Reason | Status |
|---|---|---|---|
| Keep these AFT models. AFT means S(t \| X) = S0(φ(X)·t), which rescales time about 0. No warning and no deprecation. | maintainer | By that definition everything follows: dependence on where 0 is, the sign flip before 0, the clock anchored at 0. | done |
| AFT and ALT are distinct. A model whose covariates shift the location is an accelerated life model, e.g. `AcceleratedLife(Normal, life_models.Linear)`. R's gaussian `survreg` "AFT" is that model. | maintainer | SurPyval's naming follows the textbook definition, not R's. | done (guide note) |
| A `GeneralLinear` life model (μ = a + b'X). | maintainer | So the location model can take several covariates. | #963 open |
| A time before 0 needs the covariate from that time to 0. A faster-ageing unit looks safer before 0: "it is what it is". | maintainer | It follows from the anchoring at 0. | done |

### Covariates must cover every time the answer depends on (#930, #952, #957)

| Decision | By | Reason | Status |
|---|---|---|---|
| Refuse a schedule or path that starts after the time origin, rather than holding its first value back. This is a breaking change. | maintainer | "This is a silent error." | merged (#951) |
| On baselines defined below 0, PH and PO need the covariates from −∞. AFT and AH are anchored at 0. In `fit_tvc`, a first `xl = 0` means the unit entered at 0, and `-inf` means followed since −∞. | maintainer | The model's maths depends on the whole stretch. | merged (#956) |
| Integrate a `CovariatePath` below 0, and from −∞ for PH and PO. | maintainer | — | merged (#958) |
