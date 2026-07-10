# Development Notes

This document tracks known issues, technical debt, and improvement priorities for surpyval. Issues are grouped by theme and ordered by severity within each section. Implemented items are removed; this reflects the state of the codebase as of 2026-06-12. The full test suite passes on Python 3.11+ with numpy 2.x / scipy 1.17 / pandas 3.x.

---

## 1. Type Hints — Finish the Package

The package ships `py.typed` and `mypy` runs in CI, so the typing
contract is live and must be honoured across the whole package, not
just the parts done so far.

**Done (public APIs, all mypy-clean, no new errors over baseline):**
- Univariate **parametric**: `ParametricFitter.fit`/`fit_from_df`/
  `fit_from_surpyval_data`/`fit_from_ecdf` and the `Parametric` model
  (sf/ff/df/hf/Hf/qf/cs/random/mean/var/moment/entropy/cb/param_cb,
  the AIC/BIC family, serialization, plotting).
- Univariate **nonparametric**: `NonParametricFitter.fit`/`from_xrd`
  and the `NonParametric` model methods, plus the `FIT_FUNCS`/
  `VAR_FUNCS` dispatch dicts.
- Univariate **regression**: `fit`/`fit_from_df` on the AFT/PH/PO/
  parameter-substitution fitters and `CoxPH`, the `DataFrameRegression
  Mixin`, and the `ParametricRegressionModel`/`SemiParametricRegression
  Model` prediction methods.
- Top-level helpers: `fit_best`, `handle_xicn` (and the recurrent
  validators), `SurpyvalData`, `RecurrentEventData`.

**Still to do (in suggested order):**
- Public APIs of the remaining modules: univariate **competing risks**
  (Fine-Gray, cause-specific PH), the **recurrent** package
  (parametric intensity, renewal, regression, nonparametric MCF,
  competing risks), and the **experimental** forest.
- A second pass to annotate the *internal* helpers in every module
  (the `_cb_*` confidence-bound helpers, the `mle`/`mpp`/`mps`/`mse`/
  `mom` fitters, etc.) — public-API typing only annotates the surface.
- Once a module is fully typed (public **and** internal defs), turn on
  `disallow_untyped_defs` for it in `[tool.mypy]` so coverage ratchets
  and cannot regress. This cannot be enabled per-module until that
  module has zero unannotated defs.

Convention used so far: annotate array-like inputs as `npt.ArrayLike`
and array returns as `npt.NDArray`; declare dynamically-set instance
attributes as class-level annotations; leave the autograd-`numpy`
numeric kernels loosely typed (they buy little from precise typing).

---

## 4. Recurrent Module — Bugs and Gaps

### Fresh deep-dive — 2026-06-14 (full-package audit)

A current-state audit of the whole `recurrent/` package (parametric intensity,
renewal, regression, nonparametric, competing risks). Findings are grouped by
theme; items already resolved are marked **[done]**.

**C. Simplification**
- The multi-start MLE fit scaffolding that was copy-pasted across the four
  repair fitters (`GeneralizedRenewal`, `GeneralizedOneRenewal`, `ARA`, `ARI`)
  now lives in a shared `RenewalFitMixin` (`recurrent/renewal/fit_mixin.py`):
  `_multistart` owns the `for X_init in [...]` loop, the best-start selection
  and the two identical convergence-failure raises; `_bounds_transform` owns
  the `bounds_convert` setup; `_attach_inference` owns the
  `_neg_ll`/`_mle`/`_n_obs` storage; and `_initial_dist_params` de-duplicates
  the first-event initialisation shared by `GeneralizedRenewal`/`ARA`. Each
  fitter keeps only its own likelihood and optimiser call, so the numerics are
  unchanged (`GeneralizedOneRenewal` still uses bounded Nelder-Mead, not the
  transform, to keep its pinned results identical). **[done]**
- `ProportionalIntensityModel` now inherits `RecurrenceSimulationMixin` instead
  of carrying the stale pre-mixin simulation copy, so the regression model
  gains seeding, the `max_events` backstop and the data-returning simulators;
  the covariate vector `Z` is threaded to the sampler by the public entry
  points. **[done]**
- The parametric intensity models (`HPP`/`CrowAMSAA`/`Duane`/`CoxLewis`) and
  the proportional-intensity regression fitters now set
  `_neg_ll`/`_mle`/`_n_obs` and inherit `LikelihoodInferenceMixin`, so they
  expose `log_likelihood`/`aic`/`bic`/`standard_errors` like the repair models.
  `ParametricRecurrenceModel`/`ProportionalIntensityModel` define
  `_parameter_names` for the labelling; the inference helpers were generalised
  off the renewal-only `[q, *dist_params]` assumption. An MSE fit (or a
  `from_params` model) sets nothing, so inference still raises there. **[done]**
- `Duane`/`CrowAMSAA`/`CoxLewis` no longer repeat the `iif`/`log_iif`/`cif`/
  `inv_cif` docstring boilerplate: the contract (plus `inv_cif` and
  `parameter_initialiser`) lives once on a new `IntensityModel` ABC that
  `NHPPFitter` subclasses, and the concrete models supply only the maths (which
  is left distinct). `HPP` was *not* folded into `NHPPFitter` — its bespoke
  log-space root-finding fit and pinned outputs differ enough from the
  MSE/Nelder-Mead NHPP path that the subclassing would change behaviour for no
  real saving; it stays a `CountingProcess`. **[done]**

**D. Missing capabilities**
- Trend tests are now available as standalone functions in
  `surpyval.recurrent.tests`: `laplace` (the centroid test, asymptotically
  normal) and `mil_hdbk_189c` (the Military Handbook / total-time-on-test
  statistic, chi-squared). Both take `(x, i, T)` and an `alternative`
  direction, support single- and multi-system data and both time- and
  failure-truncation, and return a `TrendTestResult`. Still to do: a
  Cramér-von Mises NHPP goodness-of-fit test, and exposing the trend tests
  as a method on fitted NHPP models. **[done]**
- Truncation now fully wired into the parametric NHPP family: a finite right
  truncation `tr` closes the observation window directly in the likelihood
  integral, so the intensity is integrated out to `tr` without relying on an
  explicit `c=1` row. The window-close term (`cif(x_last) - cif(tr)` per item,
  exactly zero when `tr` is infinite or a `c=1` row already sits at `tr`, so
  the two ways of closing the window never double-count) is shared via
  `RecurrentEventData.get_right_truncation_close()` across `HPP`, the
  `NHPPFitter` models (`CrowAMSAA`/`Duane`/`CoxLewis`) and both
  proportional-intensity regression fitters. `t`/`tl`/`tr` are now exposed on
  `ProportionalIntensityNHPP.fit()` and `ProportionalIntensityHPP.fit()`. (The
  nonparametric MCF risk set honours `tl`, and `tl`/`tr` are exposed on
  `NonParametricCounting.fit()` and `CauseSpecificMCF.fit()` — see A.) Still to
  do: multi-window (gapped) observation per item. **[done]**
- `handle_xicn` has no `e` (event-mark) parameter, so `CauseSpecificMCF.fit`
  bypasses it and can't take truncation; no `fit_from_df` for marks.
- The nonparametric MCF has **zero tests**. (The regression submodule now has
  fit/inference and simulation tests — see C and the Test coverage note below.)
- No residual diagnostics; no parametric CI band on `plot()`.

**E. API inconsistencies**
- `ProportionalIntensityModel.cif(x, Z)` / `time_terminated_simulation(T, Z,
  ...)` require `Z` at call time vs `ParametricRecurrenceModel.cif(x)`.
- `HPP` is not an `NHPPFitter`; `how=` only on `NHPPFitter`; `__repr__`
  hardcodes "Fitted by: MLE"; string dispatch `dist.name == "CoxLewis"`.
  (The `@classmethod`-on-`iif`/`cif`-that-take-`self` inconsistency is **[done]**
  — see F, all recurrent fitters are now instance-based.)

**F. Resolved since this review began** *(this branch)*
- Renewal fitters now return a generic `RenewalModel` (fitter/model split
  matching univariate `Weibull_ → Parametric`); `ARI` folded into the same
  `RenewalModel` (its "distribution" is an intensity model). **[done]**
- AIC/BIC/standard errors for the repair models via `LikelihoodInferenceMixin`.
  **[done]**
- Left-truncation support for the NHPP family; interval-censoring validation
  fix in `handle_xicn`; shared `coerce_xcnt_x`/`format_truncation` helpers.
  **[done]**
- `kijima_type` validation; `q` bounded `≥ 0`; dead `q = q` removed;
  `RecurrenceSimulationMixin` de-duplication. **[done]**
- Fitter API standardised on instances: every recurrent fitter (`HPP`,
  `CrowAMSAA`, `Duane`, `CoxLewis`, `NonParametricCounting`, the renewal and
  proportional-intensity fitters) is now a configured singleton instance with
  an instance-method `fit()`, matching the univariate `Weibull_ → Weibull`
  pattern, via the `surpyval.utils.fitter.singleton_fitter` decorator.
  Classmethod `fit`/`fit_from_recurrent_data` and the `@classmethod` `iif`/`cif`
  are gone; public `X.fit(...)` calls are unchanged. The dead
  `ParametricRecurrenceRegressionModel` stub was removed and the
  `ProportionalIntensityHPP` "dummy dist" lambda replaced with a
  `SimpleNamespace`. **[done]**

---

### Confirmed bugs

**Typo `has_left_censoing` (missing 'r')**
**Files:** `surpyval/recurrent/parametric/nhpp_fitter.py:18`, `surpyval/recurrent/regression/nhpp_proportional_intensity.py:72`
Variable is never read back so it is silent dead code today, but it will break any future code that reads the flag.

**`log_iif()` not implemented for proportional-intensity NHPP**
**File:** `surpyval/recurrent/regression/nhpp_proportional_intensity.py:129` — marked TODO.
The MLE likelihood falls back to the non-log path; for small intensities this causes underflow and silent NaN log-likelihoods.

**`CoxLewis.inv_cif()` can return negative times**
When `ln(N) < alpha`, the expression `(ln(N) - alpha) / beta` is negative. No guard or error is raised; simulation silently produces invalid (negative) event times.

---

### Missing capabilities — high priority

**No parameter uncertainty**
All models return point estimates only. No standard errors, covariance matrix, or confidence intervals for fitted parameters. For HPP, `autograd` already computes the Hessian; use `np.linalg.inv(-H)` to get the observed Fisher information. For NHPP, `scipy.optimize.minimize` returns `result.hess_inv` (BFGS) or a numerical Hessian can be computed via `numdifftools`. This is the single highest-value missing feature across the entire recurrent sub-package.

**No goodness-of-fit or trend tests**
There is no Laplace test for trend, no Military Handbook (MIL-HDBK-189C) test, no Cramér-von Mises test for NHPP, and no AIC/BIC for model comparison. The Laplace statistic is a one-liner given the event times; it should be a standalone function and a method on fitted NHPP models.

**No residual diagnostics**
No martingale residuals, no cumulative-hazard residuals, no probability-integral-transform (PIT) check. Without these, model validation is limited to eyeballing the MCF overlay.

**Renewal models have no `plot()` method**
`GeneralizedRenewal` and `GeneralizedOneRenewal` fit and simulate but cannot plot — unlike every other model in the module. Add the same MCF-overlay `plot()` that `ParametricRecurrenceModel` already implements.

**Parametric `plot()` shows no confidence band**
`ParametricRecurrenceModel.plot()` overlays the fitted CIF on the nonparametric MCF but draws no band around the parametric curve. Once parameter covariance is available (see above), add a delta-method band using the same Greenwood logic already in `NonParametricCounting.mcf_cb`.

---

### Missing capabilities — medium priority

**Left-truncation support (partial)**
`handle_xicn` now takes the surpyval `t`/`tl`/`tr` truncation fields, and the calendar-time NHPP models (`HPP`, `CrowAMSAA`, `Duane`, `CoxLewis`) integrate each item's likelihood from its entry time `tl`, so delayed-entry (warranty-from-first-sale) data is analysed correctly there. The virtual-age / history-dependent models (Kijima/G1/ARA/ARI) reject `tl > 0` with an explanatory error, since the virtual age at entry is undefined without the pre-entry history. The nonparametric MCF now uses delayed-entry risk sets (`RecurrentEventData.to_xrd` honours each item's `tl`, exposed on `NonParametricCounting.fit()` and `CauseSpecificMCF.fit()`). The proportional-intensity regression fitters (`ProportionalIntensityNHPP`/`ProportionalIntensityHPP`) now accept `t`/`tl`/`tr` and close each item's window at `tr` in the likelihood (see D). Still to do: multi-window (gapped) observation per item.

**No input validation**
Negative times, NaN/inf values, and empty arrays all pass silently into the optimiser. Add a validation step in `handle_xicn()` with informative `ValueError`s.

**No AIC/BIC on fitted models**
Log-likelihood is computed during fitting but discarded. Store it on the returned model object and expose `aic` / `bic` properties.

---

### Missing capabilities — lower priority

**Laplace and trend-test standalone functions** — **done**
`surpyval.recurrent.tests.laplace(x, i, T)` and `mil_hdbk_189c(x, i, T)` are
implemented as module-level functions independent of any fitted model (also
re-exported from `surpyval.recurrent`). Each accepts single- or multi-system
data, scalar/array/dict observation windows `T` (failure-truncated when `T` is
omitted), and a `two-sided`/`increasing`/`decreasing` alternative, returning a
`TrendTestResult` with the statistic, p-value and suggested trend direction.

**Additional virtual-age models**
Both Doyen & Gaudoin imperfect-repair families are now implemented: `surpyval.recurrent.ARA` (Arithmetic Reduction of Age, on the lifetime-distribution machinery) and `surpyval.recurrent.ARI` (Arithmetic Reduction of Intensity, built on the NHPP baseline intensity models — `CrowAMSAA`/`Duane`/`CoxLewis`), each parameterised by repair efficiency `rho` and memory `m`. Note the equivalences already covered elsewhere: ARA₁ = Kijima-I and ARA∞ = Kijima-II (both in `GeneralizedRenewal`), the geometric process (Lam) = `GeneralizedOneRenewal` reparameterised by `a = 1/(1+q)`, and ARI at `rho = 0` is the plain NHPP of its baseline intensity (used as the correctness check). The leftover virtual-age model worth adding is the general geometric-process estimator if a first-class `a` parameterisation is wanted.

**Competing failure modes**
Multiple event types in a single recurrent process (e.g., two failure modes on the same repairable system). The nonparametric scaffold is in place: `RecurrentEventData` now carries an optional event-type (mark) column and `surpyval.recurrent.competing_risks.CauseSpecificMCF` produces per-cause MCF curves over a shared at-risk set. Still to do: parametric cause-specific intensity models (cause-specific NHPP) and proportional-intensity regression, plus `fit_from_df`/`handle_xicn` support for the mark column.

---

### Test coverage

The renewal, parametric and regression sub-packages now have tests
(`tests/recurrent/test_counting.py`, `test_ara.py`, `test_ari.py`,
`test_cox_lewis.py`, `test_counting_process.py`, `test_parametric_inference.py`,
`test_hpp_proportional_intensity.py`, `test_regression_simulation.py`, …). The
remaining gaps for minimum viable coverage:

- Round-trip fit + CIF/IIF evaluation for each parametric model (HPP,
  CrowAMSAA, Duane, CoxLewis) — likelihood/AIC/BIC/SE now covered by
  `test_parametric_inference.py`; CIF/IIF round-trips still thin.
- MCF and confidence bounds for `NonParametricCounting`.
- Simulation outputs for `GeneralizedRenewal` and `GeneralizedOneRenewal`
  (covered in `test_counting.py`).
- Proportional intensity HPP and NHPP fit + predict (fit/inference covered;
  predict round-trips still thin).

---

## 5. Code Quality

### Docstring examples are not doctested
Most distribution docstring examples now execute correctly (Rayleigh's
and GumbelLEV's were rewritten with verified outputs in June 2026),
but scalar-returning examples (`mean`, `moment`) print pre-numpy-2
style plain floats, so `pytest --doctest-modules` fails on them.
Decide a policy: either render with `np.float64(...)` reprs and run
doctests in CI, or keep the readable plain-float style and accept that
examples are unchecked.

### Duplicated simulation block
The same "simulate timelines to `T`" loop appears twice (it was three times
before `parametric_regression.py` was removed):

- `surpyval/recurrent/parametric/parametric_recurrence.py` (`time_terminated_simulation`)
- `surpyval/recurrent/regression/proportional_intensity.py` (`time_terminated_simulation`)

The broken convergence-failure handling in each copy was fixed (both now emit the same `warnings.warn`), but the duplication remains. Extract a shared helper. The `count_terminated_simulation` methods are similarly duplicated.

---

## 6. Univariate Non-Parametric Module — Remaining Work

The June 2026 confidence-bound review fixed the per-estimator variance
formulas (Greenwood for KM, Aalen for NA, tie-corrected for FH), the
`random()`/`qf`/`mean` correctness bugs, and added quantile and RMST
intervals, a bootstrap, simultaneous confidence bands, the log-rank
test, and a smoothed hazard estimator. The following engineering debt
in `surpyval/univariate/nonparametric/` remains.

### Turnbull EM performance and convergence control
**File:** `surpyval/univariate/nonparametric/turnbull.py`

The EM loop builds `dok_matrix` sparse matrices and iterates over
`.keys()`/`.values()` in Python, which dominates runtime (the bootstrap
and any simulation involving Turnbull are noticeably slow). The
convergence criterion is hardcoded — `np.allclose(p, p_prev,
rtol=1e-30, atol=1e-30)` with a silent `iters < 1000` cap — so a
non-converged fit is returned without warning, and callers cannot
trade accuracy for speed. Expose `tol` and `max_iter` as `fit`
parameters, emit a `warnings.warn` when the cap is hit without
convergence, and act on the existing in-file TODO to do row-wise
iteration on the sparse matrices.

### `random()` uses the legacy global RNG
**File:** `nonparametric.py` (`NonParametric.random`, ~line 454)

`random()` calls `np.random.choice`, drawing on the legacy global
random state, so it cannot be seeded locally and is inconsistent with
the new `bootstrap_cb`/`band` methods that already accept a
`random_state`. Add a `random_state` argument and route through
`np.random.default_rng`, matching the parametric side.

### No serialization
Parametric models expose `to_dict`/`to_json`; `NonParametric` has
nothing. A fitted non-parametric model (its `x`, `R`, `r`, `d`,
`greenwood`, and estimator metadata) cannot be persisted or
round-tripped. Add `to_dict`/`from_dict` (and the JSON wrappers) and
fold the result into the empty `test_to_dict.py` once the general
parametric path is also covered.

### `interp='cubic'` can violate monotonicity
**File:** `nonparametric.py` (`interp_function`)

The survival function is monotone non-increasing, but `interp1d(...,
kind='cubic')` can overshoot and produce a non-monotone — even
out-of-`[0, 1]` — interpolated curve, which then propagates into `Hf`,
`hf`, and the interpolated confidence bounds. Switch the smooth
interpolation to a shape-preserving monotone scheme
(`scipy.interpolate.PchipInterpolator`) so interpolated curves remain
valid survival functions.

### `dist='t'` confidence-bound heuristic
**File:** `nonparametric.py` (`R_cb`)

The `dist='t'` option is documented as an unfounded conservative
heuristic (degrees of freedom from the at-risk count, undefined at the
last point). It is retained only for backward compatibility; decide
whether to formally deprecate and remove it, or keep it and accept the
maintenance of a statistically unjustified path.

### Residual test gaps
The new behaviour is well covered, but `plot()` with non-step
`interp`, the `set_lower_limit` path in `NonParametricFitter.fit`, and
the `from_xrd` entry point still have no direct tests.

---

## 7. Semi-Parametric Regression — Future Work

Four semi-parametric models are candidates for addition, in priority order:

### Buckley-James (semi-parametric AFT) — High priority
The semi-parametric counterpart to Cox PH. Fits `log(T) = β'Z + ε` without assuming a parametric baseline distribution. Uses an iterative censoring-imputation (Buckley-James) algorithm. Completes the semi-parametric trio alongside `CoxPH`. Supported in R's `survival::survreg` with no baseline assumption.

### Additive Hazards / Lin-Ying — Medium priority
`h(x|Z) = h₀(x) + β'Z` — covariate effects are additive on the absolute hazard scale rather than multiplicative. Has a closed-form estimator (no iterative optimisation needed), making it numerically fast. Useful in epidemiology for risk-difference interpretation. Implemented in R's `timereg`.

### Semi-parametric Proportional Odds — Low priority
`O(x|Z) = O₀(x) · exp(β'Z)` with a non-parametric baseline odds step function. Requires joint NPMLE of (β, Λ₀) — profile likelihood with isotonic regression inner loop (Murphy, Rossini & van der Vaart, 1997). Significantly more complex than Cox PH to implement. Parametric PO (`WeibullPO`, `LogisticPO`, etc.) already covers most practical cases.

### Frailty Models — Low priority
`h(x|Z, u) = u · h₀(x) · exp(β'Z)` where `u` is a subject-level random effect (Gamma or log-normal frailty). Different problem class (clustered/recurrent data); significant scope increase. Defer until the other three are stable.

---

## 7a. Degradation Analysis — Future Work

`surpyval.degradation` ships the classic pseudo-failure-time approach: per-unit
least-squares path fits (linear, quadratic, exponential, offset-exponential,
power, logarithmic, Lloyd-Lipow, Gompertz, Michaelis-Menten — plus
`path="best"` AICc selection across all of them), extrapolation to a
threshold, and a lifetime-distribution fit to the resulting pseudo failure
times (units whose path never reaches the threshold are right censored at
their last observation). Natural extensions, roughly in priority order:

Shipped so far beyond the basic pipeline: the two-stage noise-corrected
population path-parameter distribution (Lu–Meeker moments: pooled
`measurement_var`, `path_param_mean`, `path_param_cov` via `S - V̄` with a PSD
clip); `population_method="reml"` fitting the same quantities by restricted
marginal likelihood of the linear mixed model (Cholesky-parameterised `Sigma`,
GLS-profiled mean; linear-in-parameter paths only, coincides with moments on
balanced designs); and `DegradationModel.predict_rul(x, y)` — the Gaussian
posterior for a new unit's path parameters (conjugate for linear-in-parameter
paths, iterated linearisation otherwise) pushed through `inv_path` by Monte
Carlo, giving shrinkage RUL predictions with credible intervals.

- **Uncertainty propagation for the life model.** The two-stage method treats
  pseudo failure times as exact, so life-model confidence bounds understate
  the true uncertainty. A bootstrap over units (refit paths + life model per
  resample) is the cheap fix; expose it as a `cb`/`bootstrap` option on the
  model.
- **Induced failure-time distribution.** With `(mu, Sigma, sigma^2)` fitted
  (especially by REML), the population failure-time distribution can be
  derived from the path model directly (Monte Carlo over `theta ~ MVN` through
  `inv_path`) instead of via noisy pseudo failures — the full Lu–Meeker
  program. Compare against the pseudo-failure Weibull as a diagnostic.
- **REML for nonlinear paths.** Exponential/power paths need FO/FOCE-style
  linearisation of the mixed model (or fitting the exponential path on the
  log scale, where it is linear). Uncertainty in `(mu, Sigma)` itself
  (hierarchical bands) is also unmodelled.
- **Stochastic-process degradation models.** Wiener process with drift (first
  passage → inverse Gaussian lifetimes) and gamma process (monotone
  degradation) — these are genuine models with proper likelihoods, and they
  handle measurement error and irregular sampling more honestly.
- **Accelerated degradation testing (ADT).** Stress covariates on the path
  parameters (Arrhenius/power-law links), the degradation counterpart of the
  existing ALT regression models.
- **Destructive degradation** (one measurement per unit) — needs a different
  data model since per-unit path fits are impossible.

---

## 8. Time-Varying Covariates and Truncation (to be confirmed)

Full support for time-varying covariates (TVCs) and left/right truncation across all regression model families needs to be designed and confirmed before implementation. Key points established so far:

- **Left and right truncation** need to be added to `AFTFitter` and `ProportionalOddsFitter` — `ProportionalHazardsFitter` already handles both via `data.tl`/`data.tr`. Double truncation (observing only if `t_L < T < t_R`) is handled by dividing the likelihood by `S(t_L|Z) - S(t_R|Z)`.
- **TVCs via start-stop format** `(t_start, t_stop, event, Z)` — requires truncation support as a prerequisite, since each interval is a left-truncated observation.
- **Difficulty varies by model family:**
  - Cox PH and parametric PH: medium — cumulative hazard is additive over intervals, segments compose cleanly
  - Additive hazards (Lin-Ying): easiest — `H(t) = H₀(t) + β'·∫Z(s)ds` accumulates linearly; interval contributions are just `β'·Z·Δt`
  - Parametric AFT: harder — must track cumulative "accelerated age" `φ(Z₁)·t₁ + φ(Z₂)·(t₂-t₁) + ...` across prior intervals
  - Parametric PO: not practical — no additive hazard structure; would require numerical integration per interval

---

## 9. Long-term: Replace `autograd` with JAX (deferred)

`autograd` (HIPS/autograd) is in low-activity maintenance mode with no GPU support. JAX is the spiritual successor and a near-drop-in replacement for `autograd.numpy` patterns. The interim steps (inlining the `autograd_gamma` gradients into `surpyval/utils/autograd_gamma_compat.py` and upgrading to `autograd` 1.8 for numpy 2.x compatibility) are done, so there is no urgency. A JAX migration can be revisited once the library is otherwise stable — it is a multi-week effort touching every gradient computation.
