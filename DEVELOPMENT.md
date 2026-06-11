# Development Notes

This document tracks known issues, technical debt, and improvement priorities for surpyval. Issues are grouped by theme and ordered by severity within each section. Implemented items are removed; this reflects the state of the codebase as of 2026-06-11. The full test suite passes on Python 3.11+ with numpy 2.x / scipy 1.17 / pandas 3.x.

---

## Priority Order

1. Correctness (silent wrong results)
2. Stability (features that crash)
3. Test infrastructure
4. API consistency
5. Code quality / tooling

---

## 1. MLE Optimizer Improvements

**File:** `surpyval/univariate/parametric/fitters/mle.py`

### Hessian computed in wrong parameterization space (lines 158-168)
Standard errors should be derived from the Hessian in the *transformed* (unbounded) parameterization used during optimization, then mapped back via the delta method. Computing in physical parameter space (`transform=False`) gives incorrect standard errors for any bounded parameter.

### Optimizer cascade does not warm-start (lines 72-90)
All five methods (Nelder-Mead → Powell → BFGS → TNC → Newton-CG) start from the same cold `init`. Gradient methods should warm-start from the best-found point so far.

---

## 2. Test Coverage Gaps

### Entirely untested subsystems

| Subsystem | Files | Notes |
|-----------|-------|-------|
| `CompetingRisks`, `FineGray`, `CompetingRisksProportionalHazard` | `surpyval/competing_risks/` | No tests at all |
| `MixtureModel`, `SeriesModel`, `ParallelModel` | `surpyval/univariate/parametric/` | No tests at all |
| Recurrence/NHPP (`Crow`, `Duane`, `CoxLewis`, `CrowAMSAA`, `HPP`) | `surpyval/recurrent/parametric/` | Only `GeneralizedOneRenewal` has a test (`tests/recurrent/test_counting.py`) |
| `AcceleratedFailureTime` regression variants | `surpyval/univariate/regression/accelerated_failure_time/` | `test_regression.py` covers ALT/PH paths only |
| Serialization (`to_dict` / `from_dict` / `to_json`) | `test_to_dict.py` is 0 bytes | `to_dict` is abstract on the `Distribution` ABC but has no coverage |

Two crash bugs fixed in June 2026 (a `warnings.warng` typo and a `0 * inf = NaN` in `InstantlyOccurs.Hf`) lived in these untested areas — coverage here pays for itself immediately.

---

## 3. API Consistency Issues

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

## 4. Code Quality

### Triplicated simulation block
The same "simulate timelines to `T`" loop appears three times:

- `surpyval/recurrent/parametric/parametric_recurrence.py` (`time_terminated_simulation`)
- `surpyval/recurrent/regression/proportional_intensity.py` (`time_terminated_simulation`)
- `surpyval/recurrent/regression/parametric_regression.py` (`time_terminated_simulation`)

The broken convergence-failure handling in each copy was fixed (all three now emit the same `warnings.warn`), but the duplication remains. Extract a shared helper. The `count_terminated_simulation` methods are similarly triplicated.

---

## 5. Semi-Parametric Regression — Future Work

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

## 6. Time-Varying Covariates and Truncation (to be confirmed)

Full support for time-varying covariates (TVCs) and left/right truncation across all regression model families needs to be designed and confirmed before implementation. Key points established so far:

- **Left and right truncation** need to be added to `AFTFitter` and `ProportionalOddsFitter` — `ProportionalHazardsFitter` already handles both via `data.tl`/`data.tr`. Double truncation (observing only if `t_L < T < t_R`) is handled by dividing the likelihood by `S(t_L|Z) - S(t_R|Z)`.
- **TVCs via start-stop format** `(t_start, t_stop, event, Z)` — requires truncation support as a prerequisite, since each interval is a left-truncated observation.
- **Difficulty varies by model family:**
  - Cox PH and parametric PH: medium — cumulative hazard is additive over intervals, segments compose cleanly
  - Additive hazards (Lin-Ying): easiest — `H(t) = H₀(t) + β'·∫Z(s)ds` accumulates linearly; interval contributions are just `β'·Z·Δt`
  - Parametric AFT: harder — must track cumulative "accelerated age" `φ(Z₁)·t₁ + φ(Z₂)·(t₂-t₁) + ...` across prior intervals
  - Parametric PO: not practical — no additive hazard structure; would require numerical integration per interval

---

## 7. Long-term: Replace `autograd` with JAX (deferred)

`autograd` (HIPS/autograd) is in low-activity maintenance mode with no GPU support. JAX is the spiritual successor and a near-drop-in replacement for `autograd.numpy` patterns. The interim steps (inlining the `autograd_gamma` gradients into `surpyval/utils/autograd_gamma_compat.py` and upgrading to `autograd` 1.8 for numpy 2.x compatibility) are done, so there is no urgency. A JAX migration can be revisited once the library is otherwise stable — it is a multi-week effort touching every gradient computation.
