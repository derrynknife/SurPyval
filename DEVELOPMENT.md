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

## 1. Test Coverage Gaps

### Entirely untested subsystems

| Subsystem | Files | Notes |
|-----------|-------|-------|
| `CompetingRisks`, `FineGray`, `CompetingRisksProportionalHazard` | `surpyval/competing_risks/` | No tests at all |
| `MixtureModel`, `SeriesModel`, `ParallelModel` | `surpyval/univariate/parametric/` | No tests at all |
| Recurrence/NHPP (`Crow`, `Duane`, `CoxLewis`, `CrowAMSAA`, `HPP`) | `surpyval/recurrent/parametric/` | Only `GeneralizedOneRenewal` has a test (`tests/recurrent/test_counting.py`) |
| `AcceleratedFailureTime` regression variants | `surpyval/univariate/regression/accelerated_failure_time/` | `test_regression.py` covers ALT/PH paths only |
| Serialization (`to_dict` / `from_dict` / `to_json`) | `test_to_dict.py` is 0 bytes | `to_dict` is abstract on the `Distribution` ABC but has no coverage; `Bernoulli`/`ExactEventTime` round-trips are covered in `test_regressions.py` but the general path is not |
| Probability plotting (`get_plot_data` / `plot` / `probability_plotting.py`) | `surpyval/univariate/parametric/` | No tests at all; the June 2026 plotting refactor was verified only by ad-hoc before/after output comparison |

Two crash bugs fixed in June 2026 (a `warnings.warng` typo and a `0 * inf = NaN` in `InstantlyOccurs.Hf`) lived in these untested areas — coverage here pays for itself immediately.

---

## 2. API Consistency Issues

### Weibull/Rayleigh `cs()` convention change needs a release note
Fixed June 2026: `cs()` now uniformly computes `sf(x + X) / sf(X)`
("survive a further `x` given survival to `X`"). Weibull and Rayleigh
previously computed `sf(x) / sf(X)` (absolute time), and
`Parametric.cs` shifted `x` by gamma accordingly. This changes results
for existing Weibull/Rayleigh `cs` users, so the next release's notes
must call it out.

### LFP/ZI confidence bound change needs a release note
Fixed June 2026: MLE confidence bounds now include the variance of the
LFP (`p`) and zero-inflation (`f0`) parameters via the full covariance
matrix (`cov_matrix` on the fitted model), so bounds for `lfp`/`zi`
models are wider than before and the lower `sf` bound can fall below
the fitted `1 - p` asymptote. `param_cb("p")` and `param_cb("f0")` are
now supported. `gamma` deliberately still carries no Wald variance
(threshold parameters are non-regular). User-`fixed` parameters now
correctly carry zero variance, so free-parameter bounds on such fits
are conditional (narrower than before, which treated fixed parameters
as estimated), and `fixed={"p": ...}` / `fixed={"f0": ...}` work (they
previously raised `IndexError` from an off-by-one in
`Parametric.__init__`'s `param_map`). The next release's notes must
call out the changed LFP/ZI and fixed-parameter bounds.

### Weibull is the only distribution with a closed-form `R_cb`
**File:** `surpyval/univariate/parametric/distributions/weibull.py`

Every other distribution's closed-form confidence bound was dead code
(removed June 2026) and they all use the generic autograd delta-method
bound in `Parametric.cb`. Weibull's survives because it is actually
reachable. The two methods differ by up to ~7% (the closed form
expands in log-log "u-space", the generic in logit-space; both are
valid first-order delta methods). Decide whether to delete Weibull's
closed form so all distributions are consistent — a user-visible
change to Weibull confidence bounds — or keep it and document why
Weibull is special.

### `entropy` is declared by the ABC but implemented by four distributions
`Distribution` (`distribution.py`) declares `entropy()`, and
`Parametric.entropy` delegates to the underlying distribution, but only
Bernoulli, Exponential, Rayleigh and Weibull implement it. Calling
`model.entropy()` on the other ten raises `AttributeError`. Either
implement the closed forms (all are textbook), provide a numerical
default on `ParametricFitter`, or stop declaring it on the ABC.

### `SeriesModel` / `ParallelModel` are unexported and undocumented
**Files:** `surpyval/univariate/parametric/series.py`, `parallel.py`

They support a nice composition API (`model_a | model_b`,
`model_a & model_b`) but are not exported from
`surpyval.univariate.parametric`, have no docstrings, and implement
only `sf/ff/df/hf/Hf` (no `params`, `fit`, serialization or bounds).
Either make them public properly or mark them experimental.

### `cb()` default `on` parameter differs between `Parametric` and `NonParametric`
`Parametric.cb()` (`parametric.py`) defaults to `on='R'`; `NonParametric.cb()` (`nonparametric.py`) defaults to `on='sf'`. Both strings refer to the survival function in the same conditional.

### `Distribution` ABC adoption is incomplete
`Parametric` now inherits from the `Distribution` ABC (`distribution.py`), but `NonParametric`, `MixtureModel`, `SeriesModel`, `ParallelModel`, `NeverOccurs`, and `InstantlyOccurs` expose the same interface without inheriting it. Users still cannot write polymorphic code using `isinstance(model, Distribution)` across model types.

### Parameter naming inconsistency (`alpha`/`beta`)
Weibull, LogLogistic, Gamma, Beta, and ExpoWeibull all use `param_names=['alpha', 'beta']` but the roles differ across distributions. Positional use will produce silently incorrect results.

### `FineGray.fit` raises `NotImplementedError` over ~100 lines of commented-out implementation
**File:** `surpyval/competing_risks/regression/fine_gray.py`

The class body is mostly a commented-out likelihood/jacobian/hessian implementation (originally numba-based); `fit()` raises `NotImplementedError`. Either finish the implementation (without numba) or delete the class and the dead code — exporting a model that cannot fit misleads users about the library's capabilities.

---

## 3. Code Quality

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
- `Uniform` declares `support=(-np.inf, np.inf)` as a workaround (see
  the comment in `distributions/uniform.py`), and the NaN-support
  branch in `fit_from_surpyval_data` (with its `TODO: More general
  support setting. i.e. 4 parameter Beta`) appears unreachable now.
  Design proper data-dependent support setting.
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

## 4. Semi-Parametric Regression — Future Work

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

## 5. Time-Varying Covariates and Truncation (to be confirmed)

Full support for time-varying covariates (TVCs) and left/right truncation across all regression model families needs to be designed and confirmed before implementation. Key points established so far:

- **Left and right truncation** need to be added to `AFTFitter` and `ProportionalOddsFitter` — `ProportionalHazardsFitter` already handles both via `data.tl`/`data.tr`. Double truncation (observing only if `t_L < T < t_R`) is handled by dividing the likelihood by `S(t_L|Z) - S(t_R|Z)`.
- **TVCs via start-stop format** `(t_start, t_stop, event, Z)` — requires truncation support as a prerequisite, since each interval is a left-truncated observation.
- **Difficulty varies by model family:**
  - Cox PH and parametric PH: medium — cumulative hazard is additive over intervals, segments compose cleanly
  - Additive hazards (Lin-Ying): easiest — `H(t) = H₀(t) + β'·∫Z(s)ds` accumulates linearly; interval contributions are just `β'·Z·Δt`
  - Parametric AFT: harder — must track cumulative "accelerated age" `φ(Z₁)·t₁ + φ(Z₂)·(t₂-t₁) + ...` across prior intervals
  - Parametric PO: not practical — no additive hazard structure; would require numerical integration per interval

---

## 6. Long-term: Replace `autograd` with JAX (deferred)

`autograd` (HIPS/autograd) is in low-activity maintenance mode with no GPU support. JAX is the spiritual successor and a near-drop-in replacement for `autograd.numpy` patterns. The interim steps (inlining the `autograd_gamma` gradients into `surpyval/utils/autograd_gamma_compat.py` and upgrading to `autograd` 1.8 for numpy 2.x compatibility) are done, so there is no urgency. A JAX migration can be revisited once the library is otherwise stable — it is a multi-week effort touching every gradient computation.
