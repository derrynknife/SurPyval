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

## 1. Test Coverage Gaps

### Entirely untested subsystems

| Subsystem | Files | Notes |
|-----------|-------|-------|
| `CompetingRisks`, `FineGray`, `CompetingRisksProportionalHazard` | `surpyval/competing_risks/` | No tests at all |
| `MixtureModel`, `SeriesModel`, `ParallelModel` | `surpyval/univariate/parametric/` | No tests at all |
| Recurrence/NHPP (`Crow`, `Duane`, `CoxLewis`, `CrowAMSAA`, `HPP`) | `surpyval/recurrent/parametric/` | Only `GeneralizedOneRenewal` has a test (`tests/recurrent/test_counting.py`) |
| `AcceleratedFailureTime` regression variants | `surpyval/univariate/regression/accelerated_failure_time/` | `test_regression.py` covers ALT/PH paths only |
| Serialization (`to_dict` / `from_dict` / `to_json`) | `test_to_dict.py` is 0 bytes | `to_dict` is abstract on the `Distribution` ABC but has no coverage |
| Confidence bounds | `test_confidence_bounds.py` | Basic coverage added June 2026 (hess_inv vs observed information, param/sf bounds, offset/zi/lfp/fixed); `R_cb` distribution overrides and the `NonParametric` bounds remain untested |

Two crash bugs fixed in June 2026 (a `warnings.warng` typo and a `0 * inf = NaN` in `InstantlyOccurs.Hf`) lived in these untested areas — coverage here pays for itself immediately.

---

## 2. API Consistency Issues

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

### Triplicated simulation block
The same "simulate timelines to `T`" loop appears three times:

- `surpyval/recurrent/parametric/parametric_recurrence.py` (`time_terminated_simulation`)
- `surpyval/recurrent/regression/proportional_intensity.py` (`time_terminated_simulation`)
- `surpyval/recurrent/regression/parametric_regression.py` (`time_terminated_simulation`)

The broken convergence-failure handling in each copy was fixed (all three now emit the same `warnings.warn`), but the duplication remains. Extract a shared helper. The `count_terminated_simulation` methods are similarly triplicated.

---

## 4. Long-term: Replace `autograd` with JAX (deferred)

`autograd` (HIPS/autograd) is in low-activity maintenance mode with no GPU support. JAX is the spiritual successor and a near-drop-in replacement for `autograd.numpy` patterns. The interim steps (inlining the `autograd_gamma` gradients into `surpyval/utils/autograd_gamma_compat.py` and upgrading to `autograd` 1.8 for numpy 2.x compatibility) are done, so there is no urgency. A JAX migration can be revisited once the library is otherwise stable — it is a multi-week effort touching every gradient computation.
