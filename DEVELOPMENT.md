# Development Notes

This document tracks known issues, technical debt, and improvement priorities for surpyval. Issues are grouped by theme and ordered by severity within each section. Implemented items are removed; this reflects the state of the codebase as of 2026-06-12. The full test suite passes on Python 3.11+ with numpy 2.x / scipy 1.17 / pandas 3.x.

---

## Priority Order

1. Correctness (silent wrong results)
2. Stability (features that crash)
3. Test infrastructure
4. API consistency
5. Code quality / tooling

---

## 4. Recurrent Module — Bugs and Gaps

### Confirmed bugs

**Typo `has_left_censoing` (missing 'r')**
**Files:** `surpyval/recurrent/parametric/nhpp_fitter.py:18`, `surpyval/recurrent/regression/nhpp_proportional_intensity.py:72`
Variable is never read back so it is silent dead code today, but it will break any future code that reads the flag.

**`log_iif()` not implemented for proportional-intensity NHPP**
**File:** `surpyval/recurrent/regression/nhpp_proportional_intensity.py:129` — marked TODO.
The MLE likelihood falls back to the non-log path; for small intensities this causes underflow and silent NaN log-likelihoods.

**`ParametricRecurrenceRegressionModel` is incomplete and unexported**
**File:** `surpyval/recurrent/regression/parametric_regression.py:137` — simulation methods are marked TODO, class is defined but never imported in `__init__.py`.

**`CoxLewis.inv_cif()` can return negative times**
When `ln(N) < alpha`, the expression `(ln(N) - alpha) / beta` is negative. No guard or error is raised; simulation silently produces invalid (negative) event times.

**Renewal model restoration factor `q` is unbounded**
**File:** `surpyval/recurrent/renewal/generalized_renewal.py`
Multi-start search covers `q ∈ {0.0001, 1.0, 2.0}` but the optimizer is unconstrained, so `q` can go negative, violating the virtual-age assumption. Add a bound `q ≥ 0`.

**Redundant `q = q` assignment**
**File:** `surpyval/recurrent/renewal/generalized_renewal.py:409` — dead assignment, remove.

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

**No left-truncation support**
The univariate module handles left truncation via `tl`; the recurrent module does not. Items observed only after a delayed start (e.g., warranty data starting from first sale) cannot be correctly analysed.

**No input validation**
Negative times, NaN/inf values, and empty arrays all pass silently into the optimiser. Add a validation step in `handle_xicn()` with informative `ValueError`s.

**No AIC/BIC on fitted models**
Log-likelihood is computed during fitting but discarded. Store it on the returned model object and expose `aic` / `bic` properties.

---

### Missing capabilities — lower priority

**Laplace and trend-test standalone functions**
Implement `surpyval.recurrent.tests.laplace(x, i, T)` and `mil_hdbk_189c(x, i, T)` as module-level functions independent of any fitted model.

**Additional virtual-age models**
Beyond Kijima I/II: arithmetic reduction of intensity (ARI), arithmetic reduction of age (ARA), and the geometric process. These are well-studied and have closed-form likelihood contributions.

**Competing failure modes**
Multiple event types in a single recurrent process (e.g., two failure modes on the same repairable system). Requires extending `RecurrentEventData` with an event-type column and updating the MCF estimator to produce cause-specific curves.

---

### Test coverage

Only one test file with one test covers the entire recurrent module (`tests/recurrent/test_counting.py` — a single `GeneralizedOneRenewal` fit). The parametric, nonparametric, and regression sub-packages have no tests at all. Minimum viable coverage:

- Round-trip fit + CIF/IIF evaluation for each parametric model (HPP, Crow, CrowAMSAA, Duane, CoxLewis)
- MCF and confidence bounds for `NonParametricCounting`
- Simulation outputs for `GeneralizedRenewal` and `GeneralizedOneRenewal`
- Proportional intensity HPP and NHPP fit + predict

---

## 5. Code Quality

### Turnbull heuristic downgrade never takes effect
**File:** `surpyval/univariate/parametric/parametric_fitter.py` (`_validate_fit_inputs`, ~line 344)

The block that swaps the memory-hungry Turnbull heuristic for the
plain estimator when there is no left/interval censoring or
right-truncation assigns to a local variable and returns `True`, so the
caller never sees the downgrade — the optimization has never applied.
Fix by returning the adjusted heuristic and using it in
`fit_from_surpyval_data`. Results should be equivalent, but plotting
points and performance change, so verify against a Turnbull fixture.

### Structural refactors in the univariate parametric module
Deferred from the June 2026 clean-up (sections 1–5 of that review are
done):

- `ParametricFitter.fit_from_surpyval_data` (~200 lines) mixes
  truncation clamping, validation, initial-guess derivation, fitter
  dispatch and support assignment. Extract `_initial_guess(...)` and
  `_set_support(model)`; the LFP/ZI guess blocks are already
  self-contained.
- `Parametric.cb` (~135 lines) still builds nested closures for the
  R-based bounds. Extract the per-function bound computations. (The
  hf/df bounds now use the delta method directly rather than
  differentiating the Hf bound curve, June 2026.)
- `probability_plotting.probability_plot_data` special-cases
  distributions by name with `dist.name in ("Beta")` — string
  membership on a *string*, not a tuple, so any distribution whose name
  is a substring of "Beta" would match. Replace the name-based
  branching with a `plot_x_limits` hook on the distribution.
- Data-dependent support setting now works for the general case
  (June 2026): a distribution declares `support=(np.nan, np.nan)` and
  a `support_param_index` naming which fitted parameters supply the
  bounds, and `fit_from_surpyval_data`/`from_params` resolve the
  support from them. The new four-parameter Beta (`Beta4`) uses this
  path with `support_param_index=(2, 3)`. `Uniform` still declares
  `support=(-np.inf, np.inf)` rather than NaN (see the comment in
  `distributions/uniform.py`); it could be migrated to the NaN path
  with `support_param_index=(0, 1)` (the default) for consistency, but
  the current workaround is harmless.
- `MixtureModel` composes rather than inherits: it now shares the
  probability-plot code but still reimplements `sf/ff/df/mean/random`
  aggregation and its own `R_cb`, and sets most attributes outside
  `__init__`. Its `R_cb` is unreachable: it reads `self.res` and
  `self.hess_inv`, which `_em()` never sets, so it raises
  `AttributeError` on any fitted model. Either compute a Hessian after
  EM or remove the method.

### Type hints are vestigial
The package ships `py.typed`, but only `ParametricFitter.__init__` is
annotated. Either grow annotations outward (fitter signatures and
`fit()` are the highest-value targets) or remove the `py.typed`
marker; the current state advertises typing that doesn't exist.

### Docstring examples are not doctested
Most distribution docstring examples now execute correctly (Rayleigh's
and GumbelLEV's were rewritten with verified outputs in June 2026),
but scalar-returning examples (`mean`, `moment`) print pre-numpy-2
style plain floats, so `pytest --doctest-modules` fails on them.
Decide a policy: either render with `np.float64(...)` reprs and run
doctests in CI, or keep the readable plain-float style and accept that
examples are unchecked.

### Triplicated simulation block
The same "simulate timelines to `T`" loop appears three times:

- `surpyval/recurrent/parametric/parametric_recurrence.py` (`time_terminated_simulation`)
- `surpyval/recurrent/regression/proportional_intensity.py` (`time_terminated_simulation`)
- `surpyval/recurrent/regression/parametric_regression.py` (`time_terminated_simulation`)

The broken convergence-failure handling in each copy was fixed (all three now emit the same `warnings.warn`), but the duplication remains. Extract a shared helper. The `count_terminated_simulation` methods are similarly triplicated.

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

### Type hints and residual test gaps
The module is untyped despite the package shipping `py.typed` (see also
section 4). The new behaviour is well covered, but `plot()` with
non-step `interp`, the `set_lower_limit` path in
`NonParametricFitter.fit`, and the `from_xrd` entry point still have no
direct tests.

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
