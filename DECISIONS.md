# Decisions

Design decisions made while working on the open issues, with the reason for each and where it stands. Newest first.

**Who decided:**
- **maintainer:** chosen in review.
- **principle:** settled by a numbered rule in `docs/Design Principles.rst`.
- **proposed:** made during implementation and not yet confirmed.

**Status:** `done` (built), `open` or `gap` (not built yet), `changed (date)` (the maintainer chose otherwise; the reason points to the decision that replaces it), `closed (date)` (not to be built), `superseded (date)` (replaced by a later fix).

**Before asking the maintainer, check the principles.** If one settles the question, apply it and record it here.

---

## 2026-11-05

### Restricted mean survival time of the parametric models

| Decision | By | Reason | Status |
|---|---|---|---|
| `Parametric.rmst(tau, alpha_ci)` returns the non-parametric `rmst`'s dict (`rmst`, `se`, `lower`, `upper`, `tau`); `se` is the delta method's on the RMST itself, and the interval is the Wald interval on `logit(rmst / tau)`, which `rmst_cb(method="wald")` also gives. | proposed | The same call and result as the non-parametric models; the logit keeps the bounds in (0, tau), where a plain or log interval can pass tau when the RMST is near it. Matches R's delta method on the same scale to 1e-6. | done |
| `rmst()` refuses a model with no covariance (built with `from_params`, not fitted by MLE), naming `mean(tau=tau)`, which gives the point alone, as the non-parametric `mean(tau)` does. | proposed | A dict with `nan` bounds would hide why; the non-parametric `rmst` refuses a model without a variance too. | done |
| The RMST is the area from 0 for every distribution, a whole-line one (Normal, Gumbel, Logistic) included: the time lived in [0, tau]. A `tau` past the data is allowed (the fit extrapolates); the default `tau` is the largest observed time, as for the non-parametric models. Discrete distributions are refused (their RMST is a sum of `sf` over the counts). | proposed | One definition across models; the non-parametric refusal past the data is about a curve that is not estimated there, which a parametric fit is. | done |
| The regression models' `rmst_cb` has Wald and bootstrap bounds, not likelihood-ratio ones: `method="lr"` is refused with that message. | proposed | The regression likelihood-ratio search is built per function (sf, quantiles); a search over an integral at every step is a separate piece. | gap |

### Firth's penalised Cox fit: `CoxPH.fit_firth`

| Decision | By | Reason | Status |
|---|---|---|---|
| A separate method, `CoxPH.fit_firth(x, Z, c, n, tl, strata, alpha_ci, profile, center, tol)`, in `cox_firth.py`, returning the usual Cox model with a `firth` result (`FirthResult`, in `to_dict`). | proposed | As `fit_penalised`: a different estimator with its own inference; the same model type predicts and serialises as any Cox model. | done |
| Breslow ties only, no `tie_method`. | proposed | R's `coxphf`, the reference, has only Breslow's; the penalty's derivatives are written for it. | done |
| The covariance is the inverse of the unpenalised information at the Firth estimate, `I(beta_hat)^-1`, not of the penalised curvature. | proposed | What `coxphf` reports (checked to 1e-10) and Heinze and Schemper use. | done |
| Intervals and p-values by the profile penalised likelihood by default (`coxphf`'s `pl = TRUE`), Wald with `profile=False`; `model.p_values` and `summary()` show the chosen ones, at the fitted `alpha_ci` (another level refuses, naming the refit). `z` stays Wald's. | proposed | Under separation the profile is flat on one side and the Wald interval is far too narrow there. | done |
| `likelihood_ratio_test()` of a Firth fit is the penalised test against all coefficients 0 (`coxphf`'s); `neg_ll()`, AIC and BIC stay those of the unpenalised partial likelihood at the estimate. | proposed | The penalised statistic is the one that goes with the estimate; the criteria keep one meaning across Cox fits. | done |
| A column the partial likelihood cannot determine is refused, not aliased. | proposed | The penalty, a log determinant, is not defined where the information is singular. | done |
| The monotone-likelihood warning of a Cox fit names `CoxPH.fit_firth` as the fix; the Fine-Gray and weighted Cox fits, which it does not cover, keep the general advice. | proposed | The warning should name a fix that the caller's fit can use. | done |
| No weights `w` yet. | proposed | `coxphf` has none to check against. | open |

### Brier score and time-dependent AUC for competing risks

| Decision | By | Reason | Status |
|---|---|---|---|
| Competing risks enter the existing metrics as `e=` (event type per row, `None` for censored, `c` then optional, as the competing-risks fits) and `cause=` (the one scored); the prediction matrix then holds the cumulative incidence `F_k(t \| Z)` of that cause in the argument that otherwise holds `S(t \| Z)`. No new functions. | proposed | One set of metric functions; the same data convention as every competing-risks fit; with one cause the result is the ordinary metric of `1 - S`. | done |
| A subject with a competing event by `t` is weighted `1 / G(x_i-)` and counts as "no event of the cause" in the Brier score; the AUC's controls default to everyone without the cause by `t` (`controls="cause_free"`: `Score`'s definition, `timeROC`'s `AUC_2`), with `"event_free"` (`AUC_1`) as the option. `G` is the Kaplan-Meier of the censoring of every cause. | proposed | `riskRegression::Score(cause = k)`'s definitions, so the numbers check against it; `AUC_2` is the one that uses every uncensored subject. | done |
| The AUC standard errors follow the influence function that `Score` gives when the scored cause is coded 1 (and `timeROC`), not `Score`'s output for a cause coded otherwise, which changes with the labelling. | proposed | A standard error cannot depend on what a cause is called; the relabelled `Score` call and `timeROC` agree with each other and with SurPyval. | done |

### Standard errors and comparisons of the Brier score, IBS and time-dependent AUC

| Decision | By | Reason | Status |
|---|---|---|---|
| A new function, `score_predictions(x, c, predictions, times, metrics, alpha_ci)`, returns a `PredictionScores` result (scores and comparisons tables, influence functions, `to_dict`/`from_dict`); `brier_score`, `integrated_brier_score` and `auc_td` are unchanged and still return bare numbers. | proposed | A flag that changed the plain functions' return type would break unpacking (`times, bs = ...`) and their types; one entry point taking several models mirrors `riskRegression::Score` and is where a later `cause=` fits. | done |
| The standard errors include the censoring Kaplan-Meier's own variability (`Score(conservative = FALSE)`), through its Nelson-Aalen expansion with everyone with `x >= s` at risk, as `riskRegression` and `timeROC` do; `se = sd(IF) / sqrt(n)` with the `n - 1` standard deviation, as `Score`. The censoring distribution is the scored data's own (no `x_train`): a separate one would need another variance term. | proposed | Matches `Score` to rounding (1e-13) with and without ties. | done (gap: `x_train`) |
| Intervals of the scores are clipped to [0, 1] and those of the differences are not; each model is compared with every model before it (model minus reference), as `Score`'s contrasts. | proposed | `Score`'s conventions, so the numbers can be checked against it. | done |

### Pseudo-observations and the regressions on them

| Decision | By | Reason | Status |
|---|---|---|---|
| The coefficients of `pseudo_regression` take SurPyval's sign (a positive coefficient means a shorter life) on every quantity: the survival regressions model the event probability `1 - S(t)` (cloglog: `log(-log S(t)) = a(t) + b'Z`, the proportional-hazards reading; logit: the log-odds of failing by `t`), the cumulative-incidence regressions the cause's incidence, and the RMST regression is `g(RMST) = a - b'Z`, so an identity-link coefficient is minus the RMST difference and a log-link one minus the log ratio. R's `geeglm` on `pseudomean` reports the RMST coefficients with the other sign; the reference tests flip it. | proposed | The maintainer's one sign convention (2026-11-03, "One sign convention") for new models. How it applies on a scale that is not a hazard or a time (a mean, a probability) was not settled; modelling the event probability makes it the natural sign for the survival and incidence models, and the minus sign in the RMST model is the conservative reading of the rule. The alternative is the RMST's own sign (positive = longer), as R and `survRM2`. | proposed |
| Rows with a missing covariate are dropped (with the usual warning) before the pseudo-values are computed, so the Kaplan-Meier estimate is that of the analysed units. | principle (missing values) | The package's fitting rule; R's usual practice computes them on every unit and then drops rows in `geeglm`. The reference tests use complete data. | done |
| Gaussian working variance (each pseudo-value weighs the same) and the robust sandwich `A^-1 B A^-1` with no small-sample factor; counts `n` are copies (each its own cluster). | principle 16 | `geepack::geeglm(..., family = gaussian, corstr = "independence")`, the reference, and the literature's default. | done |
| Delayed entry, left and interval censoring and times past the last observation are refused with a message; there is no `w` argument (sampling weights) yet. | proposed | Leave-one-out estimates of truncated or interval-censored samples do not give pseudo-values with the right expectation without further work. | open |

### Confidence bounds on a Cox model's predicted curve

| Decision | By | Reason | Status |
|---|---|---|---|
| `SemiParametricRegressionModel.cb(x, Z, stratum, on, alpha_ci, bound, bound_type, method)` bounds `sf`, `ff` and `Hf` with the variance of R's `survfit.coxph` (`std.err`: Tsiatis's baseline variance plus the coefficients' part, the Efron form after an Efron fit, weighted with the robust covariance for a weighted fit). `bound_type` takes R's `conf.type` names, `"log"`, `"log-log"` and `"plain"`, and defaults to `"log"`, R's default; the non-parametric estimates' `"exp"` and `"normal"` are accepted as `"log-log"` and `"plain"`. | proposed | The task is R parity, and `"log"` is what `survfit(fit, newdata)` prints. The non-parametric estimates and Aalen's model default to the log-log scale (`"exp"`), so the option's default differs between models; the maintainer may prefer one default everywhere. | done |
| Only Wald bounds (`method="wald"`); `"lr"` and `"bootstrap"` refuse, naming the fix (refit to resampled rows). `hf` and `df` (the baseline's jumps) have no bounds; nor has `qf` (no `quantile_cb` yet). | proposed | The baseline has no likelihood for a likelihood-ratio bound; R has no bound on the jumps either. | done (gaps recorded) |
| A model restored by `from_dict` and a penalised fit refuse `cb`: the variance needs the fitted rows, which a dict does not keep, and a penalised fit has no covariance. | proposed | Storing the running sums in `to_dict` is possible later; refusing keeps the saved format unchanged. | done (gap recorded) |

### Comparing two MCFs (`recurrent.mcf_diff`)

| Decision | By | Reason | Status |
|---|---|---|---|
| `mcf_diff(model_a, model_b)` takes two fitted `NonParametricCounting` estimates, as `sf_diff` takes two survival fits and `reda::mcfDiff` two `mcf` objects; a model without per-item data (`from_xrd`, `from_dict`, simulated) is refused, since neither the robust variance nor the tests can be computed from it. | proposed | The groups' fits already carry the data, late entry and windows; a group-label signature would duplicate `fit`'s arguments. | done |
| The difference is estimated only from the later origin to the end of the shorter follow-up and is NaN outside it; `reda::mcfDiff` carries the shorter group's MCF at its last value. | proposed | The MCF itself is NaN past its last observed time (no claim that no events happen after it); the reference tests compare reda's values on the shared range. | done |
| The pointwise bounds of the difference are Wald bounds only (no log scale), and there is no simultaneous band. | proposed | A difference can be negative; a band is a further feature. | done (band a gap) |

### Interval-censored log-rank and Wilcoxon tests: `interval_logrank`

| Decision | By | Reason | Status |
|---|---|---|---|
| A function of its own, `interval_logrank`, not new paths in `logrank`. `logrank` keeps refusing left and interval-censored data (and now a two-column `x`), with a message naming `interval_logrank`; its results do not change. | proposed | The tests differ in kind: `logrank`'s weightings, strata, `t0`, `w` and hypergeometric variance have no counterpart in the score tests on the pooled Turnbull estimate, whose variance is the permutation one. Switching method on the shape of the data would change a result when one interval row is added. | done |
| Scores `"log-rank"` (Sun's; the default, as R's `ictest`), `"finkelstein"` and `"wilcoxon"`; R's names `logrank1`, `logrank2`, `wmw` are accepted. | proposed | Sun's scores are the usual log-rank scores on exact and right censored data (each group's score sum is the observed minus expected events; tested). | done |
| The asymptotic p-value is the k-sample permutation chi-squared (R's `method = "pclt"`); `method="permutation"` is a Monte Carlo test of the same statistic, `(1 + count) / (1 + B)`, seeded by `random_state`. For two groups this is two-sided by the absolute statistic (R's `tsmethod = "abs"`), not R's default doubling of the smaller one-sided p-value. No exact enumeration and no score-form (`method = "scoretest"`) variance. | proposed | One statistic for any number of groups; the doubled one-sided p-value is defined for two groups only. | done |
| Truncated data (finite `tl` or `tr`) are refused, pointing to `logrank` with `tl`. | proposed | The permutation argument needs exchangeable units, which delayed entry breaks; R's `ictest` takes no truncation either. | done |
| The pooled NPMLE is the EM-ICM with `tol = 1e-12` (the Karush-Kuhn-Tucker gap) by default, tighter than `Turnbull.fit`'s 1e-10, and warns if it does not converge. | proposed | The statistic moves with the NPMLE's convergence; R's default `epsilon = 1e-6` moves it by about 1e-8 relative. | done |

## 2026-11-04

### Crow Extended projection: bias-corrected or plain shapes

| Decision | By | Reason | Status |
|---|---|---|---|
| `CrowAMSAA.projection` keeps the bias-corrected shapes by default ((N - 1) / N for the demonstrated intensity, (K - 1) / K for the rate of new BD modes) and takes `bias_corrected=False` for the plain maximum-likelihood shapes. | proposed | MIL-HDBK-00189A's example uses the corrected shapes (demonstrated 7.84708, projected 11.29418); MIL-HDBK-189C's worked example (section 6.3.3.8) uses the plain ones (7.71, 10.82), while its test-find-test example (section 6.3.2.8) gives both and recommends the corrected one. Keeping the default moves no numbers; the option lets a user reproduce either handbook. Grouped data is not taken (exact times only), so 189C's rule of always using the plain shape for it does not arise. | done |

## 2026-11-03

### Bootstrap bounds for limited-failure and zero-inflated models

| Decision | By | Reason | Status |
|---|---|---|---|
| The bootstrap of a limited-failure or zero-inflated model is the one the plain and offset models already have: the parametric (model) bootstrap with the conditional censoring of Davison & Hinkley and bias-corrected percentile intervals, refitted with the same `offset`, `lfp`, `zi` and `fixed`; not a case (row) resampling with BCa intervals. | principle 15, 24 | One bootstrap for every univariate model, so the methods agree and the plain models' bounds do not move. | done |
| A unit drawn as never failing (`inf`) whose drawn censoring time is also `inf` (a failed unit past the last censoring time) is censored at the last time in the data. | proposed | No unit was followed longer than that; a finite failure keeps its exact time, as before, so the plain models' draws are unchanged. | done |
| A refit's `lfp_p` or `f0` counts as on its edge (1, 0) where moving it there, the other parameters held, costs less than 1e-3 in deviance (the threshold the likelihood-ratio bounds use for an estimate on the edge); the model's own estimate is read the same way. Refits on the edge are kept, never dropped. | proposed | The fit stops a hair short of an edge (`lfp_p` at 1 - 1e-7), so without this an interval could never reach the edge and would miss a truth on it. | done |
| Calibration study (`test_coverage_extra_parameters.py::test_bootstrap_coverage`): n = 100 units, 200 replicates, `n_boot = 200`, data seeds 311 (Weibull lfp, `lfp_p` 0.6), 312 (Weibull, `lfp_p` 1, fitted with `lfp=True`), 313 (Weibull zi, `f0` 0.1), 314 (LogNormal lfp, `lfp_p` 0.6); the bootstrap's seed at replicate `r` is `1000 * seed + r`; two-sided 90% and 95% bounds on every parameter, on `sf` and on `qf` at three true quantiles; the usual rule (`check_coverage`, slack 0.01); on the edge the target for `lfp_p` is `1 - alpha_ci / 2`, as for the likelihood-ratio study. Fixed before the first run. | principle 17 | — | first run (3 November 2026, 33 minutes on ten busy cores): `lfp`, `lfp_edge` and `zi` pass; `lognormal_lfp` fails one check, the 90% `quantile_cb` at p = 0.1, 0.825 against the rule's lower limit 0.826. The 90% intervals run short in every study (0.83-0.89 where 0.90 is the target, e.g. `zi` `cb` 0.83); the 95% ones are 0.89-0.97. The maintainer chose BCa (below); seeds and tolerances unchanged. |
| A limited-failure or zero-inflated model's bootstrap intervals are BCa, with Efron's jackknife acceleration: the model refitted without each unit in turn, a row of count `n` standing for `n` units (weight `n`), edges read as for the refits; 0 (the percentile interval) where the jackknife has no spread, as on an edge. The plain and offset models keep the bias-corrected percentile interval, unchanged. | maintainer | The first run's 90% intervals were short; the jackknife needs no covariance, which these models lack on an edge. | done |
| The same study rerun with BCa (same seeds 311-314, n, replicates, `n_boot` and rule; 47 minutes): every check passes. 95%: `lfp` 0.915-0.965, `lfp_edge` 0.90-0.95 (`lfp_p` 0.995), `zi` 0.90-0.93, `lognormal_lfp` 0.905-0.96. 90%: `lfp` 0.87-0.925, `lfp_edge` 0.855-0.88 (`lfp_p` 0.995), `zi` 0.835-0.91, `lognormal_lfp` 0.84-0.915 (the failing check, the 90% `quantile_cb` at p = 0.1, went from 0.825 to 0.84). The 90% intervals of the edge and zi designs are still a little short of 0.90, inside the rule. | principle 17 | — | done |

### Weights: the case bootstrap

| Decision | By | Reason | Status |
|---|---|---|---|
| `method="bootstrap"` on a weighted parametric fit (`cb`, `param_cb`, `quantile_cb`, `mrl_cb`) or a weighted parametric regression (`cb`, `param_cb`, `quantile_cb`; PH, AFT, PO, AH, accelerated life) is the case bootstrap: the units are drawn i.i.d. with replacement, each with its weight, covariates, time, censoring and truncation, and the weighted model is refitted. The univariate bounds are the bias-corrected percentile intervals of the refits and the regressions' the BCa intervals, as unweighted. | proposed | The parametric bootstrap simulates from the model and knows nothing of the design the weights describe; resampling the units does. Likelihood-ratio bounds stay refused. | done |
| The regressions' BCa acceleration with weights is the skewness of `g' A^-1 S*`, `A^-1` the inverse weighted information (the bread), not the robust covariance. | proposed | The linearisation of the weighted estimate is `A^-1` times the weighted score; the robust covariance is the variance of that, not its direction. | done |
| The resamples are drawn by position, and the refits use the weights scaled to sum to the units, so a seed gives the same bounds at any scale of `w` (to the optimiser's tolerance: the conformance check uses each case's own tolerance). | proposed | The weight-scale property every weighted bound keeps. | done |
| The case bootstrap resamples left- and interval-censored data as they are (it simulates nothing); limited-failure, zero-inflated and discrete fits still refuse it, as unweighted; time-varying fits too. No fit takes a cluster label (only `robust_covariance(cluster)` does), so clusters are not resampled. `bootstrap_cb` and `band` of the non-parametric estimates still refuse weights. | proposed | — | done (gaps recorded) |
| The calibration study of the case bootstrap (informative sampling, seed 26104, 200 replicates of 200 refits, 90% intervals of the Weibull `alpha`, `beta` and `sf` at the quartiles, the study's `Z_TOL` plus slack 0.02) was fixed before its first run. | principle 17 | — | done |

### Weights: Cox from a DataFrame and with time-varying covariates

| Decision | By | Reason | Status |
|---|---|---|---|
| `CoxPH.fit_from_df(w_col=)`: the weights go through the one mask of the rows with every covariate, so the fit equals `fit(..., w=)` of the rows kept, bit for bit. `validate_coxph_df_inputs` returns the weights as a tenth element only when `w_col` is given, so its tuple is unchanged for other callers. | proposed | This closes the gap recorded on 2026-10-29 ("its frame handling drops rows before `fit` sees them"). | done |
| `CoxPH.fit_tvc(w=)` takes one weight per interval row (a weight may change between a subject's rows, as a time-varying censoring weight does; R allows it too). The covariance is the weighted sandwich clustered by subject, and `robust_covariance()` of the model clusters by subject by default; `cluster=` still overrides. Refused with `strata` (the weighted stratified fit is not built). | proposed | R's `coxph(..., id = id, robust = TRUE)`; with the subject as the cluster, splitting follow-up into episodes changes nothing (checked). | done |
| Robust bounds on a weighted Cox model's predicted survival are not built here: `CoxPH` models have no `cb` (no survival bounds at all, weighted or not), so this needs the unweighted bound (and its conformance declarations) first. | proposed | Bigger than one weighted extra; it is a new bound for the model. | gap |

### Weights: the log-rank tests

| Decision | By | Reason | Status |
|---|---|---|---|
| `logrank(..., w=)` and `pairwise_logrank(..., w=)`: the weighted numerator with the design-based variance `sum_i n_i w_i^2 s_i s_i'`, `s_i` each unit's score residual at `beta = 0` (Breslow's ties), which is R's `coxph(..., weights = w, robust = TRUE, ties = "breslow")$rscore`; checked against it to 1e-10. The robust variance is used whenever `w` is given, `w = 1` included. | principle 16 | R's robust score test is the reference; a hypergeometric variance with weighted counts depends on the scale of the weights. | done |
| The test weights (Gehan's `r`, Tarone-Ware's `sqrt(r)`, Peto-Peto's `r + 1`, the Fleming-Harrington Kaplan-Meier) use the weighted numbers with the weights scaled to a mean of 1 per unit over the whole data (all strata at once), and are treated as fixed in the variance. | proposed | The `+ 1` of Peto-Peto is one unit, so the weights must be on the units' scale for the test not to depend on the scale of `w`; one scale for every stratum keeps the strata's relative weights. R has no weighted Peto-Peto or Fleming-Harrington test to compare with. | done |

### Weights stage 2: Turnbull

| Decision | By | Reason | Status |
|---|---|---|---|
| `Turnbull.fit(..., w=)` is the weighted NPMLE: each row counts `n w` in the E-step and the M-step (and its truncation "ghosts"), the weights first scaled to sum to the units as a weighted likelihood uses them. Checked against R's `icenReg::ic_np(..., weights = w)` to 1e-8. | proposed | The weighted pseudo-likelihood's maximum; the scaling makes the iterations themselves free of the weights' scale. | done |
| The Fleming-Harrington option (the default) takes each of the `m` expected tied failures at their mean weight, `sum_{l<m} a / (r - l a)`, `a = d / m`, `m` the expected units failing (from the counts), in the EM's update and at the end, as the weighted `FlemingHarrington` does. So whole-number weights do not equal counts under it (as for `FlemingHarrington`); the Kaplan-Meier and Nelson-Aalen options do. | proposed | The only form free of the weights' scale that gives the unweighted estimate at `w = 1`. | done |
| A weighted Turnbull estimate has no variance (`greenwood` is None): `cb`, `quantile_cb`, the bounds of `rmst` and `mean`, and `sf_diff` refuse with a `WeightedRefusal` naming `bootstrap_cb`, and its bounds are `bootstrap_cb`, the weighted case bootstrap (the units drawn i.i.d. by position, each with its weight; percentile bounds; `random_state` the seed). `cb` does not silently switch to the bootstrap. `plot` draws the curve without bounds. | proposed | Greenwood's formula on the EM's weighted expected counts reads weights as units seen, and no robust (infinitesimal jackknife) variance of the weighted EM is built. A `cb` that resampled would be random with no seed argument; non-parametric models have no `method=`, and `bootstrap_cb` is already the recommended Turnbull bound. | done |
| The bootstrap's coverage was checked by a study fixed before its first run (`scripts/studies/weighted_turnbull/coverage.py`: informative sampling of Weibull lives inspected every 2, 500 replicates seeded 26105 + r, 200 refits seeded 36105 + r, 95% percentile bounds of S(4) and S(8)): 0.946 and 0.930, Monte Carlo error 0.010 and 0.011. The second is 1.8 errors below 0.95, the usual small undercoverage of percentile bounds; reported, not tuned. | principle 17 | R has no weighted Turnbull bounds. | done |
| A weighted probability plot of left or interval censored or right truncated data (and `heuristic="Turnbull"`) uses the weighted Turnbull positions; `how="MPP"` still refuses weights. | proposed | The positions are scale free; MPP's regression on them is not a weighted estimator anyone has defined. | done |

### Stratified Cox: residuals, `diagnose()`, robust errors, concordance and saving

| Decision | By | Reason | Status |
|---|---|---|---|
| Every residual is computed within its stratum's risk sets; the scaled Schoenfeld residuals and dfbeta use the whole fit's covariance and number of events; the proportional-hazards test sums the information over the strata, with the km scale of time from all the rows. | principle 16 | What R's `residuals.coxph` and `cox.zph` do; checked against them to 1e-6. | done |
| `concordance()` of a stratified fit pools the concordant and usable pairs of each stratum, and takes `strata=` with `x`, `c`, `Z` for new data (refused on an unstratified model). The pair counts come from a new `metrics.concordance.concordance_counts`, which `concordance_index` now divides. | principle 16 | R's `concordance` of a stratified `coxph`. | done |
| A stratified Cox dict stores a `"strata"` list of baselines and is schema 4; the top-level baseline is still the first stratum's. | principle 20 | A schema-3 reader would otherwise read the first stratum's baseline as the model's, silently. | done |
| A stratified fit keeps its rows (`_fit_data`, in the order given, with each row's stratum) as an unstratified one does; a penalised stratified fit still keeps none, and its residuals say so. | proposed | The penalised fit is out of this change's scope. | done |

### Concordance: Uno's C, standard errors and comparing two scores

| Decision | By | Reason | Status |
|---|---|---|---|
| `concordance_estimate` reproduces R's `survival::concordance(..., reverse = TRUE)`: Uno's weighting is `timewt = "n/G2"` (`1 / G(t-)^2`, `G` the events-first reverse Kaplan-Meier of the same data), `tau` is `ymax` (pairs whose event is *at or before* `tau`; `survC1::Est.Cval` uses *before*), and the standard error is the infinitesimal jackknife with the censoring weights held fixed. `compare_concordance` is R's two-fit covariance with the contrast `c(1, -1)`. Checked to 1e-10 (C) and a relative 1e-8 (variances). | principle 16 | R's `concordance` is the reference users compare with; `survC1` agrees with it to about 1e-8 on data without tied times. | done |
| The interval is the normal one on the C scale by default (`ci_method="normal"`), with `"logit"` as the option that stays inside (0, 1); a C of exactly 0 or 1 gives the point as its logit interval. | proposed | R reports only `var`; `C -/+ z se` is what its users compute. | done |
| A stratified Cox model refuses `weighting="uno"`, `tau` and `concordance_estimate`, naming `surpyval.metrics.concordance_estimate` per stratum. | proposed | R pools the strata's jackknife terms and estimates `G` per stratum; that needs its own reference, out of this change's scope. | done |
| A model's `concordance_estimate` with a missing covariate raises (naming the rows to drop), while `concordance` keeps its NaN. | proposed | A standard error of an undefined index has nothing to report; `concordance_index` already refuses a missing score. | done |

### `random()` for the AFT, PO, Cox and Aalen models

| Decision | By | Reason | Status |
|---|---|---|---|
| A Cox or Aalen draw beyond the last time the step curve is estimated at is `inf`, without a warning; the docstring says so and how often (`sf` at that time). | proposed | It is what the model says (the curve stops above 0 when the data end censored), as a limited-failure model's `random` gives `inf`; a warning in every simulation loop would be noise. The user censors the draws where they choose. | done |
| The Cox model's `random(size, Z, random_state=None, *, stratum=None)`: `stratum` is keyword-only, unlike `sf(x, Z, stratum)`. | principle 21 | `random_state` keeps its place as the third argument of every regression's `random`. | done |
| The frailty models, the Lin-Ying additive model, Buckley-James, the semi-parametric proportional odds model, the joint model (which draws through `predict(...).random`) and the trees exclude the new `random_sf` conformance property with the reason. | proposed | Out of the scope of this change; each is a documented gap. | done |

### Refusals that named issue numbers

| Decision | By | Reason | Status |
|---|---|---|---|
| No error or warning message names an issue number, an issue URL or "tracked in"; each says what to do instead. `surpyval/tests/test_no_issue_numbers_in_messages.py` checks every string in the library's code that is not a docstring. Docstrings and comments keep their references. | principle 2, 22 | The issues named were closed, so the message pointed to a page that did not say what to do. | done |
| `registry.Pending` carries `instead` (what to check the model with meanwhile) in place of its issue number. Aalen additive, Bayesian, distributional and Buckley-James models have `PENDING` entries; any other model without a family gets a general refusal pointing to a nonparametric comparison and the models that have `diagnose()`. `AalenAdditiveModel` and `BayesianParametric` inherit `DiagnoseMixin`, so `diagnose()` refuses rather than being missing. | principle 2 | One refusal path for every model without checks. | done |
| `fit_bayes(w=)` and `fit_bayes_from_df(w_col=)` are explicit keyword arguments that refuse a value with a `WeightedRefusal` (`ValueError`), as `takes_unit` does for the fitters without weights; the message points to the weighted `fit` only where the distribution's `fit` takes `w`. | principle 2 | `fit_bayes` is not wrapped by `takes_unit`, so `w=` was Python's `TypeError`. | done |
| A stratified Cox model's `to_dict()` stays a `NotImplementedError`, now saying to save it with `pickle` (checked to keep every stratum's baseline) or to refit with `strata=`. | proposed | Serialising the per-stratum baselines is a format change, out of scope for a message fix. | superseded: stratified Cox models now save, as schema 4 (see "Stratified Cox: residuals, `diagnose()`, robust errors, concordance and saving") |

### A limited-failure proportion fitted at its bound

| Decision | By | Reason | Status |
|---|---|---|---|
| The printout of a fitted limited-failure model notes `lfp_p` at its bound of 1 when `1 - lfp_p < 1e-5`, not only when it equals 1, and says to fit without `lfp=True`. A model built with `from_params` gets no note. | principle 22 | The search stops just short of the bound (1 - lfp_p was between 1e-10 and 1e-6 on every such fit tried), so `== 1` would almost never fire. | done |

### CI time limits and balanced test parts

| Decision | By | Reason | Status |
|---|---|---|---|
| Every job in `actions.yml` and `publish.yml` has `timeout-minutes` of about twice its time on the 0.26 release run (lint, coverage, wheel and the publish jobs 10-15, conformance and docs 20, the suite's parts and calibration 60, pymc 30 as before). | proposed | A hang fails with the job's name; generous because CI runs only on the release pull request, where a limit too tight blocks a release. | done |
| Per-test limits (pytest-timeout, thread method): 20 minutes in the suite's parts, 10 in the conformance suite, 45 in calibration as before. | proposed | The longest suite tests take a few minutes on the runners; the per-test durations of each part are now printed (`--durations=20`) and recorded, so the limit can be tightened from data. | done |
| The parts are balanced by recorded test times with the existing `--shard` option (`--shard-durations`, longest first into the least-loaded part; a test with no recorded time counts as the median), not with the `pytest-split` plugin. Each release run records the times (`--store-durations`) and the `coverage` job merges them into an artifact to commit as `.github/test_durations.json`; with no file the hash split stays. | proposed | No new dependency, and the hash split stays as the fallback. No times file is committed yet: one needs a full run's times, which the next release run produces. | open |
| actionlint is not added to the lint job yet. | proposed | It was not run over the workflows locally; with shellcheck on the runner it could report findings in the existing `run:` scripts and turn the release run red. Run it once over both workflows, fix what it finds, then add it. | open |

## 2026-10-30

### Fixes behind the second bug hunt's strict xfails

| Decision | By | Reason | Status |
|---|---|---|---|
| A row of a two-column `x` flagged interval censored (`c = 2`) whose ends are equal, `[t, t]`, is the exact (observed) time `t` on every path: `fit`, `fit_from_df` and the non-parametric fits read it as `c = 0`. A genuinely reversed interval (`xl > xr`) is still refused, and the refusal now counts the rows, names them and says to give each interval lower end first. | maintainer | `fit_from_df` already promised it ("If xl[i] == xr[i] assumes observed"), and the default flags of a two-column `x` already read such a row as exact; refusing it on one path only, with a message about "one failure time", was the inconsistency. An interval of zero width carries exactly the information of an exact time. | done |
| `w=` given to the `fit` of a fitter that takes no weights (the distributional, Aalen, Royston-Parmar, frailty, competing-risks, recurrence, degradation and other fitters), or `w_col=` to a `fit_from_df` that has no `w_col`, raises a `ValueError` (`WeightedRefusal`) naming the fitter and `w`, saying weighted fitting is not built for it yet and which models take weights, instead of Python's `TypeError`. It is raised by `takes_unit`, the one wrapper every public `fit` has. | maintainer | Principle 2: invalid input raises a `ValueError` naming the argument and the fix. Weighted data passed as a `SurpyvalData` was already refused this way. | done |
| A Cox fit with no finite maximum whose baseline cannot be moved to `Z = 0` (the runaway coefficients make `exp(beta'Z)` over- or underflow on the rows as given) keeps the baseline at the covariate means (`model.center`), as `center=True` does, with one warning saying so, instead of raising after the "No finite maximum" warning. A fit with a maximum whose move fails is still refused, pointing to `center=True`. Unweighted, stratified (every stratum kept, or every one moved) and weighted fits alike. | principle 13 | Whether a separated fit returned or raised depended on how far its search ran before the move. The model at the means is the same model, so its predictions are those of `center=True`, which the warning about the runaway estimate already qualifies. | done |
| `concordance_index` refuses a missing (NaN) risk score, naming `risk`; a model's `concordance(x, c, Z)` with a missing covariate still gives NaN (NaN in, NaN out, as the forests' `concordance`). A NaN time still gives NaN. Equal infinite scores are tied; an infinite score is tied with no finite one. | principle 2 | A silent NaN index hid the missing score; the tie test's relative tolerance was infinite at an infinite score, which counted it as tied with every finite one. | done |
| `brier_score` (and so `integrated_brier_score`) refuses survival probabilities outside [0, 1], counting them. | principle 2 | A survival probability outside [0, 1] is a hazard or a mistake, and its score is meaningless. | done |

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
| No changelog entry for these: Bayesian fitting was unreleased then (it shipped in v0.26, with these fixes); the feature's changelog line now lists what it refuses. | proposed | Fixes to a feature not yet released were not changes to released behaviour. 0.26 has shipped: fixes to it now get changelog entries. | done |

### Weighted Cox review fixes

| Decision | By | Reason | Status |
|---|---|---|---|
| The weighted refusal of residuals, `check_ph` and the diagnostics sits in `proportional_hazards.diagnostics._require_cox` (`cox_fit_data`), the entry point every path shares: the model's methods, the module's functions and `diagnose()`. | principle 13, 22 | They take the rows as counted units; refused in one place, no path can skip it. | done |
| A weighted Cox fit moves its baseline to `Z = 0` with `baseline_at_origin`, as the unweighted fit, refusing a move that over- or underflows. Its `r` becomes the risk weight `sum n w exp(beta'Z)` over each risk set (the unweighted `r` is `sum n exp(beta'Z)`; Breslow's increment is `d / r` for both), moved with the baseline; it was the weighted number at risk, which is not moved by a change of origin. `r` is stored, not used by any prediction. | proposed | One meaning of `r` for every Cox fit; moving the number at risk by `exp(beta'centre)` would be wrong. | done |
| Aliasing in a weighted Cox fit uses `_cox_aliased` on the weighted information at 0, with the weighted counts `n w` for the columns' spread and the weighted events for its scale; the aliased columns are nan in `params`, `covariance()`, `robust_covariance()` and the summaries, and 0 in the predictions, as unweighted. | proposed | The same rule as the unweighted fit, on the information the weighted fit maximises (R's `coxph` gives NA either way). | done |
| A weighted Efron fit saves `baseline_method = "efron"` (`cox_ph._baseline_method`). No compatibility read for `"fleming-harrington"`: weights were new in v0.26 and unreleased when this was decided, so no saved model had the old label (0.26 shipped with `"efron"`), and the label is not used in any calculation. | proposed | — | done |
| Royston-Parmar predictions sum the spline per query point with `utils.linalg.fixed_order_dot` (each row reduced alone), in the univariate model and the regressions' spline baseline, which the fits share. | principle 4 | A BLAS matrix-vector product sums a row differently depending on its place, so a permuted query changed results in the last bit on Linux x86-64 (as #1018 for the Bayesian predictive). The other models' predictions passed the same check at query lengths up to 64 on Linux. | done |

### Penalised regression

| Decision | By | Reason | Status |
|---|---|---|---|
| A new method, `fit_penalised(..., penalizer, l1_ratio, standardise)`, on `CoxPH` and the PH, AFT and PO fitters, rather than new arguments to `fit`. | proposed | `fit` keeps its signature; a penalised fit is a different estimator with different inference. | proposed |
| `penalizer` and `l1_ratio` are lifelines' names (glmnet's `lambda`, `alpha`), with glmnet's objective (mean negative log-likelihood plus the penalty); `N` is the sum of the counts. | maintainer (2026-11-03) | The established references; counts equal repetition. Confirmed by the maintainer (2026-11-03), names and objective. | done |
| `standardise=True` by default, dividing by the count-weighted sd with divisor `N` (glmnet's; lifelines uses `N - 1`); coefficients reported in the covariates' units. | maintainer (2026-11-03) | The penalty should not depend on units. Confirmed by the maintainer (2026-11-03): divisor `N`, glmnet's, as documented in `fit_penalised`. | done |
| No covariance, standard errors, bounds or p-values for a penalised model: each raises, saying why and what to do; `summary()` shows `nan`. | every-bound rule, principle 17 | Penalised Wald intervals do not have their stated coverage. | done |
| `aic`/`bic` count the effective degrees of freedom `tr[H (H + P)^-1]` over the parameters not at 0 (Gray 1992). | proposed | The lasso's count of non-zero coefficients, and the usual ridge df. | proposed |
| `CoxPH.fit_penalised(..., strata=)`: the stratified partial likelihood, one set of coefficients and a baseline per stratum; `N` is the units over every stratum and the covariates are centred and standardised over all the rows. | proposed | glmnet's `stratifySurv`, which it matches to 1e-6; the penalty's scale should not change with how the units are split into strata. | done |
| Start-stop (delayed entry) rows in penalised Cox: `N` is the number of rows weighted by `n`, not of subjects, and the standardisation is over rows. | maintainer (2026-11-03) | glmnet's convention for `Surv(start, stop, event)`, the reference it is checked against; documented, since a subject with many rows weighs more in the penalty's scale. Changed by the maintainer (2026-11-03): `N` counts subjects, not rows, so the same data split into more rows give the same fit; made in a separate change. | changed (2026-11-03) |
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
| `coef_penalty` is a ridge penalty on the covariates only, scaled by each covariate's variance; the baseline is not penalised. lifelines' `coef_penalizer` also penalises its intercept, so the two differ. | maintainer (2026-11-03) | A penalty that does not depend on the covariates' units keeps the covariate-scale invariance; shrinking the baseline hazard towards 0 has no meaning. The maintainer (2026-11-03) confirmed the penalty and changed its name: `coef_penalty` becomes `penalizer`, the name of `fit_penalised`, with `coef_penalty` kept, deprecated, for one release; made in a separate change. | changed (2026-11-03) |
| `cb` is Wald only, on `log H` with the optional-variation covariance of `B`; `method` other than `"wald"` is refused saying why (no likelihood; no bootstrap implemented). | principle (every bound) | The documented gap. | done |
| `constant_effects_test` is timereg's Kolmogorov-Smirnov and Cramér-von Mises test of time-invariant effects, its null simulated by resampling (`n_sim`, `random_state`); its own seeded study found it anti-conservative at n = 300 (size 0.08 in the study, 0.13-0.16 in a separate check; 0.05 at n = 1000), pinned as a strict xfail in `calibration/test_coverage_aalen.py` and stated in the docstring. | proposed | Reported as found, not tuned. A fix (a time range short of the last event, or a small-sample correction) is for the maintainer. Superseded by the fix of 2026-11-03 ("Aalen: the size of `constant_effects_test`"), which resamples the events: about 5% from n = 300. | superseded (2026-11-03) |
| Left out: lifelines' `smoothing_penalizer`, and the refit-registry study for the case (its fixture's fit is not an additive model to draw from); the model has its own calibration study. | proposed | — | done |

### Spline-baseline (Royston-Parmar) regression

| Decision | By | Reason | Status |
|---|---|---|---|
| Built as the `PH` and `PO` families on a spline baseline distribution with fixed knots (`SplineBaseline`), not a new model class. | principle 14, 21 | The model then has every bound, `summary`, `fit_from_df`, formulas and saving as the other parametric regressions, by the same code. | done |
| `RoystonParmarRegression(df=3, scale="hazard", knots=None)` configures the fitter (its `fit` and `fit_from_df` keep the regression signature); `RoystonParmarPH` and `RoystonParmarPO` are the `df=3` instances. `RoystonParmar.fit` is unchanged. | principle 21 | `fit_from_df` cannot take model options. | done |
| Knots as the univariate fit places them (quantiles of the log event times), then held: bounds, bootstrap and likelihood-ratio searches are conditional on them, as in flexsurv. | principle 16 | — | done |
| On the odds scale the coefficients multiply the survival odds, as every SurPyval `PO`: flexsurv's odds-scale coefficients have the opposite sign. | maintainer (2026-11-03) | One meaning of a PO coefficient in the package. Put under the maintainer's one sign convention (2026-11-03): "the SurPyval convention, in the way the existing models define it". The odds scale already has the sign of the existing `PO` models, so its coefficients do not move; see 2026-11-03, "One sign convention". | changed (2026-11-03) |
| `scale="normal"` with covariates is refused, naming `LogNormalAFT` and the univariate fit. | maintainer (2026-11-03) | Its covariate shift is neither PH nor PO; a probit-shift family would be new. Closed as won't-do for now by the maintainer (2026-11-03): no normal-scale Royston-Parmar model with covariates; `LogNormalAFT` covers it and the refusal message stays. | closed (2026-11-03) |

### Robust standard errors on the parametric regressions

| Decision | By | Reason | Status |
|---|---|---|---|
| The names are the Cox model's: `robust_covariance(cluster=None)`, `robust_standard_errors(cluster=None)` and `summary(robust=True, cluster=...)`; not a `robust=` argument to `fit` (lifelines' form). | principle 21 | The same option has the same name everywhere, and the fit is the same either way. | done |
| HC0, no small-sample factor; counts are frequency weights (a row with `n = 3` is three units, each its own cluster without `cluster`, three members of its cluster with it). | maintainer (2026-11-03) | R's `sandwich` and lifelines use HC0; counts must equal repetition, as `CoxPH.robust_covariance` already reads them. Confirmed by the maintainer (2026-11-03); documented in Weighted Data with the size of R `survey`'s `sqrt(n/(n-1))` (0.5% at n = 100, 5% at n = 10). | done |
| `cluster` is one label per row given to the fit, in that order; rows dropped for a missing covariate are matched and left out (the fit records which given row each fitted row is). | principle 3 | A label belongs to its row; the fit sorts and drops rows. | done |
| Only the standard errors and the summary change; `cb`, `param_cb` and `p_values` keep the model-based covariance. A `fit_tvc` model refuses (its likelihood is over histories, not rows), naming the bootstrap and likelihood-ratio bounds. | maintainer (2026-11-03) | Simple by default; the robust form is an option beside it. Changed by the maintainer (2026-11-03): robust is all or none, `summary`, `cb`, `param_cb` and the p-values together, and what cannot refuses; see 2026-11-03, "Robust all or none". | changed (2026-11-03) |

### Distributional regression: covariates on every parameter

| Decision | By | Reason | Status |
|---|---|---|---|
| The name is `Distributional(dist)`, with prebuilt `WeibullDistributional` and the like; not `GAMLSS`, which would promise its penalised smoothers. | maintainer | — | done |
| A new family, `Distributional(dist)` (`DistributionalFitter`, `DistributionalRegressionModel`), in which every parameter of the distribution has its own linear predictor and link, `g_k(theta_k) = b_k0 + x_k' b_k`. Covariates on a shape parameter live here rather than as an option of the AFT fitter. | maintainer | One general model rather than one option per family; the AFT model is its covariates-on-the-scale special case, tested as such. | done |
| One covariate set per fit, `Z`; each parameter chooses columns by `covariates={"alpha": [...], "beta": [...]}`, or by `formulas={...}` per parameter in `fit_from_df` (`formula=` puts one formula on every parameter). With none named, every covariate is on every parameter; with some named, the others have an intercept only. | maintainer | The same rule as the joint models' `hazard_covariates`/`path_covariates`; predictions take one row per unit. The alternative default was the location only (gamlss's). | done |
| Default links from each parameter's range: identity for a location, log for a positive parameter; a log link may be put on a location (`links={"mu": "log"}`); an override that can leave the range (identity on a scale) is refused, naming the allowed links. No logit link yet: no supported distribution has a parameter in (0, 1). | principles 2, 12, 16 | gamlss's and flexsurv's defaults; a link that can leave the parameter space gives a likelihood that is not defined. | done |
| `params` are the distribution's parameters at `Z = 0` (or at the covariate means with `center=True`) on their natural scale, then each parameter's coefficients on its link scale, named `<parameter>_<column>`. | principle 21 | `params[:k_dist]` is a distribution, as in every regression; flexsurv's coefficient names. | done |
| Coefficients have the sign of the link (`alpha(z) = alpha e^{b z}`): a positive scale coefficient means longer life, the opposite of SurPyval's AFT sign, which the documentation shows side by side. | maintainer (2026-11-03) | gamlss and flexsurv; in this model a coefficient moves a parameter, not time. Changed by the maintainer (2026-11-03): every model added in 0.26 takes SurPyval's own sign, that of the AFT and PH models (a positive coefficient means a shorter life); see 2026-11-03, "One sign convention". | changed (2026-11-03) |
| The fit's aliasing, units and no-finite-maximum checks are per parameter: a column is aliased within each parameter's block, each coefficient is judged in its own column's units, and one warning names the columns. | principles 6, 12, 13 | A coefficient per column per parameter. | done |
| A verified answer with a direction of near-zero curvature has its profile walked out along that direction and along each parameter in it; where the likelihood rises along one for at least 8 units (each parameter's size), never falling below the answer, the fit is "No finite maximum", naming those parameters. | principle 12, 13 | A group with no failures can have its scale and shape (or location and spread) run off together; the search stops on a plateau within 1e-9 nats of the supremum, where the one-parameter-at-a-time checks see a maximum. In the new fit only. | done |
| `qf`, `mean` and `random` use the distribution's closed forms at each row's parameters. | principle 9 | Exact and vectorised. | done |
| `parameter_cb(name, Z, method=...)` bounds a parameter at each row on its link's scale (Wald by the delta method, likelihood-ratio as the extremes over the region, as `cb` takes them, bootstrap BCa), mapped back through the link. | every-bound rule, principle 18 | A shape's bound stays above 0; at `Z = 0` it is `param_cb`'s bound on the parameter. | done |
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
| With weights, `hess_inv` of a parametric model holds the robust covariance of the distribution's parameters, as `covariance()` does, not the inverse Hessian. | maintainer (2026-11-03) | Every consumer of `hess_inv` (`param_cb`, the plots, packages built on SurPyval) then gets the valid variance; the plan kept `hess_inv` as the inverse Hessian, which would have left those consumers with the classical one. Changed by the maintainer (2026-11-03): `hess_inv` is always the inverse Hessian, and every bound reads `covariance()`; see 2026-11-03, "`hess_inv` is the inverse Hessian". | changed (2026-11-03) |
| Ties in a weighted Cox fit follow R's `coxph`: each of the `m` tied deaths (`m` the units, from the counts) at the deaths' mean weight in Efron's denominator. A weighted Cox fit has its own partial likelihood (`proportional_hazards._weighted_cox`); the unweighted fit is unchanged. Its `robust_covariance(cluster)` sums the weighted dfbetas within clusters. The `nan` the unweighted `robust_covariance` gave with fractional counts does not arise: weights are not counts. | maintainer (2026-11-03) | R's convention is the reference (principle 16), and the unweighted likelihood's tie terms need whole-number deaths. Confirmed by the maintainer (2026-11-03); documented in Weighted Data with the worked example (one subject of weight 3 against three tied copies), checked in `test_weighted_cox_ties.py`. | done |
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
| The grid: on the unconstrained scale along the curvature's axes at the peak, stretched (`z = 3 sinh(u / 3)`) so cells are fine near the peak and coarse in long tails, widened until every face is below 1e-8 of the peak, and a second grid offset by half a cell. A summary's quantiles spread each point's mass over its cell and remove the spread's own bias by extrapolation from a two-cell spread. The error estimate is half the difference of the two grids' 2.5%, 50% and 97.5% points of every free parameter and the 10% quantile, as a fraction of the 95% interval's width; it must be at most 0.1% with one or two free parameters and 0.5% with three (refined up to 241 and 61 points a dimension; past that the fit refuses with the numbers). | maintainer (2026-11-03) | 0.1% at three parameters needs grids too large to stay near a few seconds; the error is always reported. Confirmed by the maintainer (2026-11-03). | done |
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

## 2026-11-03

### The maintainer's answers on the proposed 0.26 rows

The rows of 2026-10-29 and 2026-10-30 that were still proposed were reviewed by the maintainer; each answer is recorded in its row, with the date.

| Decision | By | Reason | Status |
|---|---|---|---|
| Confirmed: penalised regression's names `penalizer` and `l1_ratio`; its standardisation with divisor `N` (glmnet's, documented); robust standard errors as HC0 with counts as frequency weights (documented with the size of R `survey`'s factor); weighted Cox Efron ties as R's `coxph` takes them (documented with a worked example); the Bayesian grid's accuracy rule. | maintainer | — | done |
| Changed, each in its own change: the sign of the distributional regression's coefficients and of Royston-Parmar's odds scale (one sign convention, SurPyval's); `hess_inv` with weights (always the inverse Hessian); robust standard errors (all or none); penalised Cox on start-stop data (counts subjects, not rows); Aalen's `coef_penalty` (renamed `penalizer`, the old name deprecated for one release). | maintainer | — | changed |
| Closed as won't-do for now: a normal-scale Royston-Parmar model with covariates (`LogNormalAFT` covers it, and the refusal message stays). | maintainer | — | closed |
| The other proposed rows stay proposed. Rows settled by a principle or by the maintainer whose status still said "proposed" now say "done" (they shipped in 0.26); the PyMC section, written twice, is kept once (the copy with `to_surpyval`); reasons that said 0.26 was unreleased now say it has shipped. | proposed (housekeeping) | — | done |

### One sign convention: distributional regression and Royston-Parmar

| Decision | By | Reason | Status |
|---|---|---|---|
| Every model added in 0.26 takes SurPyval's own coefficient sign, that of the AFT and PH models (a positive coefficient means a shorter life), defined as the existing models define it; not the references' sign. This reverses the 2026-10-29 row "Coefficients have the sign of the link". | maintainer (2026-11-03) | One meaning of a coefficient's sign across the package (principle 21) outweighs matching gamlss and flexsurv (principle 16); the reference tests flip the sign where they compare with R. | done |
| In a distributional regression each parameter's coefficients enter as `g(theta(z)) = g(theta) + s b'z`, `s` from the parameter's effect on life (`_links.EFFECT`, checked against each distribution's quantile function): `s = -1` where raising the parameter makes every quantile later (a scale, a location, a Gamma's or ExpoWeibull's ordering shape, a Beta's `alpha`), `s = +1` where it makes every quantile earlier (a rate, a Beta's `beta`), and `s = +1` (the link's sign) for a shape or spread whose quantiles cross (a Weibull's `beta`, every `sigma`). Covariates on the scale alone give the AFT model's coefficients, sign included. | proposed | "A positive coefficient shortens life" is defined only where a parameter orders lives; where it does not, the link's sign is the plain reading (a positive coefficient raises the shape) and is 0.26's, so those coefficients do not move. | done |
| A distributional model saved by 0.26 (`to_dict`/`to_json`, and `pickle`) is converted on loading: each coefficient, its fixed value, and the rows and columns of every stored covariance, centred-fit state and Hessian are multiplied by the old sign times the new, so it predicts as it did (tested on dicts and pickles written by 0.26 from PyPI). A dict now stores `"coefficient_signs"` and is stamped schema 4, which 0.26 refuses. | principle 20, 21 | A sign flip cannot keep the old form working for a release with a deprecation (principle 21): the same number cannot mean both. Saved models are the part that can be kept exact, and an older reader must refuse rather than read the new sign as the old. | done |
| Royston-Parmar regression is unchanged: its hazard scale is `PH` on the spline baseline (a positive coefficient shortens life, as flexsurv's) and its odds scale `PO` on it, with every SurPyval `PO` model's sign (a positive coefficient multiplies the survival odds and lengthens life; flexsurv's is the opposite). The 2026-10-29 odds-scale row therefore stands as it was, and `df = 1` on the odds scale is now tested to equal `PO(LogLogistic)`. | maintainer (2026-11-03) | The existing `PO` models are the definition the rule points to; making Royston-Parmar's odds coefficients "positive = shorter" would give `PO(SplineBaseline)` a different sign from `PO(Weibull)`. The maintainer decided (2026-11-03) to leave PO as it is: no PO model changes sign. PO keeps its meaning (a positive coefficient raises the survival odds, a longer life) and PH and AFT keep theirs (a positive coefficient, a shorter life); the docs show the rules side by side. The first instruction assumed every existing model read a positive coefficient as a shorter life; that was found to be wrong for PO. | done |

### Robust all or none

| Decision | By | Reason | Status |
|---|---|---|---|
| When robust is asked for, every uncertainty uses the robust covariance: `cb`, `quantile_cb` and `param_cb` take `robust=` and `cluster=` as `summary` does, and `robust_p_values(cluster)` gives the p-values (a method beside `robust_standard_errors`, since `p_values` is a property and keeps its meaning). This reverses the 2026-10-29 row "Only the standard errors and the summary change". | maintainer (2026-11-03) | One coefficient must not show two uncertainties. | done |
| What has no robust form refuses rather than mixing: `robust=True` with the likelihood-ratio or bootstrap bounds (no covariance in them), `cluster` without `robust=True`, and (already) restored, penalised and `fit_tvc` models. The default stays model-based. | principle 2, 24 | Simple by default (principle 24); a refusal says what to do (principle 2). | done |
| The robust Wald bounds are computed in the same parameterisation as the model-based ones (the centred fit's, behind a baseline reported at 0), with the sandwich in place of the inverse information (`_robust.robust_inference_state`). Cox's `summary(robust=True)` already built its whole table from the robust standard errors, and Cox has no coefficient `cb` or `param_cb`, so it is unchanged. The distributional model's `parameter_cb` takes no `robust` yet. | proposed | The delta method gives the same bound either way, and the centred one is better conditioned (#463). | done |

### `hess_inv` is the inverse Hessian

| Decision | By | Reason | Status |
|---|---|---|---|
| `hess_inv` is always the inverse Hessian of the (weighted) negative log-likelihood; with weights it is that of the weights as given (R's `survreg` `naive.var`, checked to 2e-4), not the robust covariance. The robust covariance is `covariance()`. This reverses the 2026-10-29 weights row "With weights, `hess_inv` ... holds the robust covariance". | maintainer (2026-11-03) | One meaning per attribute (principle 21). | done |
| Every variance inside SurPyval comes from `covariance()` (`Parametric._core_covariance`): `param_cb`, the Wald `cb` family, the probability plot's band and the summaries. A model restored from an old dict with `hess_inv` and no `"covariance"` uses `hess_inv` there only when it is not weighted (then they are the same matrix). The likelihood-ratio searches keep reading `hess_inv`: they want the likelihood's curvature to scale their steps, and weighted fits refuse them. | proposed | Weighted fits keep their bounds (tested against 0.26's values); unweighted ones are bit-identical. | done |
| A weighted model's dict writes the inverse Hessian as `"inverse_hessian"` and keeps the robust block under `"hess_inv"`, so a 0.26 reader (whose `param_cb` reads `"hess_inv"`) keeps robust bounds, with no new schema. A weighted dict saved by 0.26 (no `"inverse_hessian"`) loads with no `hess_inv`, its `"hess_inv"` being the robust covariance; its bounds use its `"covariance"` as before. | proposed | A schema bump would refuse in 0.26 a document 0.26 can read correctly this way; an attribute that cannot be known is left unset rather than filled with the wrong matrix. | done |

### Penalised regression: one shared helper and model mixin

| Decision | By | Reason | Status |
|---|---|---|---|
| Every `fit_penalised` goes through `_penalised.penalised_fit(objective, Z, weights, coef_index, penalizer, l1_ratio, standardise, tol)`, where `objective(scale, N)` builds the model's own likelihood on the covariates divided by `scale` (over `N`), rather than a helper that rescales the coefficients of a likelihood on the raw covariates. | proposed | Each model divides its covariates where its data are built (a Cox fit centres them first); taking the builder keeps the existing fits bit-exact and lets each model keep its own data preparation. | done |
| The fit helper takes the rows' `weights` (today the counts `n`), so sampling weights can be passed in later without a new signature. | proposed | — | done |
| A model that gives no Hessian is solved with one by central finite differences of its gradient; the solver's result says `hessian="finite differences"`. | proposed | The Newton polish, the check of the minimum and the effective degrees of freedom all need second derivatives. | done |
| The model side is `PenalisedModelMixin` (the `penalty`, the refusals, the effective df of `aic`/`bic`, the `to_dict` entry and the printout line), on both regression model classes. | proposed | — | done |

### Penalised fits on more models

| Decision | By | Reason | Status |
|---|---|---|---|
| The Cox model with time-varying covariates takes a penalty through new methods, `CoxPH.fit_tvc_penalised` and `CoxPH.fit_tvc_timeline_penalised`, rather than penalty arguments on `fit_tvc`. | proposed | The same split as `fit` / `fit_penalised`: a penalised model has a different contract (no inference), so the method name says which one the caller gets. | done |
| The cause-specific competing-risks Cox model's penalty is one `penalizer` and `l1_ratio` for every cause; `penalty.effective_df` is the causes' sum (each cause's own is on `models[cause].penalty`). The Fine-Gray model has no penalised fit. | proposed | The causes share the rows and so the scale; one penalty per cause can follow if asked. | done |
| `ProportionalIntensityHPP.fit_penalised` divides by `N`, the number of items, and standardises the covariates over the items (each item once, whatever its number of rows). | proposed | The covariates describe the items; for exact events this is glmnet's Poisson fit with one row per item and offset log(follow-up). | done |
| `ProportionalIntensityNHPP` has no penalised fit yet. | proposed | Its likelihood is written in plain numpy (as are Duane's and Crow-AMSAA's intensities), so it has no exact gradient; the penalised solver needs the gradient to its optimality tolerance, which finite differences do not reach. The likelihood needs moving to autograd first. | open |
| A distributional `fit_penalised` takes `penalizer` as one value, one per covariate column (that column on every parameter) or a dict per parameter (`{"beta": 0.5}`, one value or one per that parameter's columns); a parameter the dict does not name is not penalised. The intercepts are never penalised, and the fit is not centred (as the other penalised fits). | proposed | One penalty per parameter's coefficients was asked for; the column forms match the other fitters. | done |
| The distributional fit is checked against `WeibullAFT.fit_penalised` (covariates on the scale alone) and against its own optimality conditions, not against gamlss's `ri()`. | proposed | `ri()` chooses its own penalty scale and smoothing, so its fit is not this objective at a given penalty. | done |
| glmnet references for the new fits use a convergence threshold of 1e-30 (the older references 1e-20): at 1e-20 glmnet stops 1e-7 short on one lasso coefficient of the PBC transplant cause, where its own gradient is not yet 0. | proposed | The reference must be glmnet's answer at convergence; the test's tolerance (1e-6) was not changed. | done |

### `AalenAdditive`: `coef_penalty` is now `penalizer`

| Decision | By | Reason | Status |
|---|---|---|---|
| Rename `AalenAdditive`'s `coef_penalty` (argument, model attribute and dict key) to `penalizer`, the old names accepted with a `DeprecationWarning` until v0.28 (`REMOVED_IN_NEXT`). The maths is unchanged. | maintainer | One name for a penalty across the package (principle 21). | done |

### Penalised Cox on start-stop data counts subjects

| Decision | By | Reason | Status |
|---|---|---|---|
| `CoxPH.fit_tvc_penalised` (and the timeline form) divides by `N`, the number of subjects (distinct ids, each weighted by the `n` of its last row), and standardises over the subjects, not the rows. This replaces the rows rule 0.26 documented for start-stop data (glmnet's). | maintainer | The partial likelihood does not change when a subject's follow-up is split into more rows, so the penalty's scale should not either. | done |
| A subject's covariate for the standardisation is its time-weighted average over its follow-up: each row's covariates weighted by `xr - xl`. | proposed | Splitting a row into several with the same covariates leaves it unchanged (so the fit does not depend on the split), and it reflects every value the subject had, not only its first. | done |
| `CoxPH.fit_penalised` with `tl` (no ids) still counts rows, unchanged, and still matches glmnet's start-stop fit; the new fit is checked against glmnet with the objective rescaled to subjects (`standardize = FALSE` on `X / s`, `lambda * N_subjects / N_rows`). | proposed | Data without ids cannot be grouped into subjects. | done |

### The penalty path and choosing the penalty by AIC or BIC

| Decision | By | Reason | Status |
|---|---|---|---|
| One function, `penalty_path(fit_penalised, *args, **kwargs)`, for every model's penalised fit, rather than a `fit_penalised_path` method on each fitter. | proposed | Every penalised fit takes `penalizer` and `l1_ratio`, so one function covers them all and the path cannot drift between models. | done |
| The default path is glmnet's: from the smallest penalty that sets every penalised coefficient to 0, found by doubling and bisection to six digits (for a ridge path the one for `l1_ratio=0.001`, as glmnet), down to `min_ratio` (0.001) times it, `n_penalties` (20) on the log scale. Every fit starts afresh (no warm starts), so each is exactly `fit_penalised` at its penalty. | proposed | The same path as the reference implementation; fresh starts cost time but keep each point reproducible on its own. | done |
| The penalty is chosen by AIC or BIC with the effective degrees of freedom; ties go to the larger penalty. Cross-validation is not offered. | proposed | The information criteria are statistics of the fit and already defined for every penalised model; cross-validation is a resampling workflow outside the package's scope. | done |

### Likelihood-ratio bounds on flat profiles

| Decision | By | Reason | Status |
|---|---|---|---|
| `_lr_walk` checks each root, and each levelling off, a hair either side (`_LR_HAIR` 1e-6, `_LR_RESOLVED` 1e-3 of deviance beyond the slope). In noise it reads the walked points again for a deviance converging geometrically (each rise at most half the one before over four points, the rest of the rise short of the critical value), which gives the edge; otherwise the side is `nan`, with a warning naming how far the profile was followed. | proposed | A Gamma offset model's 80% `alpha` bound was 2e11, 7e11, 1.6e12 or infinity by platform: a crossing in the profile searches' noise. An honest edge or "not found" is the same everywhere; a number from noise is not. | done |
| The geometric reading is used only once a walk has run into noise; the walk's ordinary levelling-off test is unchanged. | proposed | Using it everywhere stopped the NegativeBinomial `r` walks earlier and doubled the slices probed (`test_609`); slowing the steps on a concave profile moved ExpoWeibull valley bounds (`test_601`). | done |
| An offset's upper side whose walk runs into noise within the corner gap of the first failure is the first failure, as when the walk reaches it. | proposed | Near the corner the searches stop at a wall; the interval ends at the first failure (`_lr_corner`). | done |
### Aalen: the size of `constant_effects_test`

| Decision | By | Reason | Status |
|---|---|---|---|
| The resampled null multiplies each event's contribution to the increments of `B`, `(X'YX)^{-1} X_i dN_i`, by a standard normal draw, not each unit's martingale residual `dN_i - Y_i X_i dB`. Each draw's variance is then the optional variation `B_var`. | proposed | The residuals of the least-squares fit nearly vanish where few units are at risk, so the draws understated the variance at the last event time, which anchors the statistics: the test rejected constant effects 44% of the time at n = 100, 21% at n = 300 and 8% at n = 1000. With the events it rejects at about 5% from n = 300 and is conservative (0.5% to 3.5%) at n = 100. The statistics themselves are unchanged. | done |
| The risk-set sums `X'YX` are summed from the last row back, not as a difference of running totals. | proposed | The difference left rounding noise where one unit was at risk, so a singular `X'YX` passed the rank check and the last steps of `B` and `B_var` were noise (a negative variance). | done |

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
| `diagnose()` is the single assumptions-checking API. `check_ph()` (the Cox model's method and `proportional_hazards.diagnostics.check_ph`) is deprecated in v0.26 and removed in v0.28: two releases, not the usual one. | maintainer (30 Oct 2026: "Can we just get rid of check_ph altogether? Leave the api for all models as just diagnose?") | One way to check a model; the extra release gives downstream code time to switch. | done |
| A longer deprecation window is listed with its own release in `utils.deprecation.SCHEDULED_REMOVALS` and warns through `scheduled_removal`; `test_scheduled_removals_are_not_overdue` fails when the package reaches that release, and the one-release guard (`REMOVED_IN`) ignores it. | proposed | `REMOVED_IN` and `REMOVED_IN_NEXT` move on each release, so they cannot hold a fixed later release. | done |
| `diagnose()` exposes the proportional-hazards test as a table: `Check.table(ctx)` gives a check's own table and the report holds them in `Diagnosis.tables` by check name (kept by `to_dict`/`from_dict`, schema unchanged since the key is optional). The Cox family's `tables["proportional_hazards"]` is exactly what `check_ph(transform)` returned; `transform` is a Cox family option (default `"km"`, as `check_ph` and R). The cause-specific competing-risks table is the causes' tables stacked, indexed by `(cause, covariate)`. | proposed | A stable, documented place for the `cox.zph` table, number for number, so `check_ph` can go without losing anything. | done |
| `diagnose(transform="identity" or "log")` adds rows on that scale to the km and rank rows; `"km"` and `"rank"` add none. | proposed | Asking for a scale should test on it; km and rank stay the default verdict. | done |
| The proportional-hazards test of a penalised Cox fit refuses with the penalised fit's own no-inference message, in `zph_score_test` (so `check_ph`, both forms, and `diagnose()` refuse alike). `compute_residuals` and `added_covariate_score_test` are unchanged. | maintainer (R-Q) | A penalised model has no p-values; one message everywhere. | done |
| `zph_score_test` (the dict) and `zph_table` (the table) stay as the functions the test is computed by, beside `compute_residuals` and `added_covariate_score_test`; nothing in the package calls `check_ph`. | proposed | `diagnose()` and the low-level tools share one implementation, with no deprecation warning inside the package. | done |

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
