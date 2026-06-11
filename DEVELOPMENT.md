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

## 1. Bugs

### `warnings.warng` typo crashes recurrent simulation on convergence problems
**File:** `surpyval/recurrent/parametric/parametric_recurrence.py:168`

`warnings.warng(...)` (typo for `warnings.warn`) raises `AttributeError` whenever a simulated timeline fails to reach `T` — exactly the case the warning was meant to report. Untested, so it has never been caught. Note this is one of three near-identical copies of the simulation block (see §5, duplicated simulation code).

### `InstantlyOccurs.Hf` returns NaN
**File:** `surpyval/univariate/parametric/__init__.py:68`

`np.zeros_like(x).astype(float) * np.inf` is `0 * inf = NaN`, not the intended infinite cumulative hazard. Use `np.full_like(x, np.inf, dtype=float)`.

### `neg_ll()` guard condition is broken
**File:** `surpyval/univariate/parametric/parametric.py:855`

`from_params()` constructs `Parametric(self, "given parameters", None, ...)`, so `self.data` is always set (to `None`) and `if not hasattr(self, "data")` is always `False`. Models created via `from_params()` fall through to `return self._neg_ll`, which raises an uninformative `AttributeError` instead of the intended `ValueError("Must have been fit with data")`. Check `if self.data is None` or `if not hasattr(self, '_neg_ll')` instead.

---

## 2. MLE Optimizer Improvements

**File:** `surpyval/univariate/parametric/fitters/mle.py`

### Hessian computed in wrong parameterization space (lines 160-170)
Standard errors should be derived from the Hessian in the *transformed* (unbounded) parameterization used during optimization, then mapped back via the delta method. Computing in physical parameter space (`transform=False`) gives incorrect standard errors for any bounded parameter.

### Optimizer cascade does not warm-start (lines 72-90)
All five methods (Nelder-Mead → Powell → BFGS → TNC → Newton-CG) start from the same cold `init`. Gradient methods should warm-start from the best-found point so far.

---

## 3. Test Coverage Gaps

### `pip install -e .[tests]` cannot run the test suite
**Files:** `pyproject.toml`, `surpyval/tests/experimental/forest/`

The `[tests]` extra contains only `pytest`, but the forest tests import `sksurv` (scikit-survival) at module level, so collection fails with `ModuleNotFoundError` unless `requirements_dev.txt` was used. Either add `scikit-survival` to the `tests` extra or guard the imports with `pytest.importorskip("sksurv")`.

### Entirely untested subsystems

| Subsystem | Files | Notes |
|-----------|-------|-------|
| `CompetingRisks`, `FineGray`, `CompetingRisksProportionalHazard` | `surpyval/competing_risks/` | No tests at all |
| `MixtureModel`, `SeriesModel`, `ParallelModel` | `surpyval/univariate/parametric/` | No tests at all |
| Recurrence/NHPP (`Crow`, `Duane`, `CoxLewis`, `CrowAMSAA`, `HPP`) | `surpyval/recurrent/parametric/` | Only `GeneralizedOneRenewal` has a test (`tests/recurrent/test_counting.py`) |
| `AcceleratedFailureTime` regression variants | `surpyval/univariate/regression/accelerated_failure_time/` | `test_regression.py` covers ALT/PH paths only |
| Serialization (`to_dict` / `from_dict` / `to_json`) | `test_to_dict.py` is 0 bytes | `to_dict` is abstract on the `Distribution` ABC but has no coverage |

The §1 bugs above (the `warng` typo, `InstantlyOccurs.Hf`) live in these untested areas — coverage here pays for itself immediately.

---

## 4. API Consistency Issues

### `cb()` default `on` parameter differs between `Parametric` and `NonParametric`
`Parametric.cb()` (`parametric.py:694`) defaults to `on='R'`; `NonParametric.cb()` (`nonparametric.py:236`) defaults to `on='sf'`. Both strings refer to the survival function in the same conditional.

### `Distribution` ABC adoption is incomplete
`Parametric` now inherits from the `Distribution` ABC (`distribution.py`), but `NonParametric`, `MixtureModel`, `SeriesModel`, `ParallelModel`, `NeverOccurs`, and `InstantlyOccurs` expose the same interface without inheriting it. Users still cannot write polymorphic code using `isinstance(model, Distribution)` across model types. (Minor related nit: `InstantlyOccurs.random` is a `@classmethod` whose first parameter is named `self` instead of `cls`.)

### Parameter naming inconsistency (`alpha`/`beta`)
Weibull, LogLogistic, Gamma, Beta, and ExpoWeibull all use `param_names=['alpha', 'beta']` but the roles differ across distributions. Positional use will produce silently incorrect results.

### `FineGray.fit` raises `NotImplementedError` over ~100 lines of commented-out implementation
**File:** `surpyval/competing_risks/regression/fine_gray.py`

The class body is mostly a commented-out likelihood/jacobian/hessian implementation (originally numba-based); `fit()` raises `NotImplementedError`. Either finish the implementation (without numba) or delete the class and the dead code — exporting a model that cannot fit misleads users about the library's capabilities.

---

## 5. Code Quality

### Mutable default arguments
`init=[]`, `fixed={}`, `include=[]`, `exclude=[]` are shared across calls and risk state pollution:

- `surpyval/fit_best.py:39`
- `surpyval/univariate/regression/accelerated_failure_time/accelerated_failure_time.py:107`
- `surpyval/univariate/regression/proportional_hazards/proportional_hazards_fitter.py:148`
- `surpyval/univariate/regression/accelerated_life/parameter_substitution.py:203`

Use `None` defaults and initialize inside the function.

### Triplicated simulation block with leftover debug prints
The same "simulate timelines to `T`" loop appears three times, each with different (broken) convergence-failure handling:

- `surpyval/recurrent/parametric/parametric_recurrence.py:120-178` — `warnings.warng` typo (§1)
- `surpyval/recurrent/regression/proportional_intensity.py:200-255` — `print("Maybe...")`
- `surpyval/recurrent/regression/parametric_regression.py:50-101` — `print("Maybe...")`

Extract a shared helper and emit a real `warnings.warn` message once.

### `print()` used instead of `warnings`/`logging` for diagnostics
- `surpyval/univariate/parametric/mixture_model.py:132` — `print("Max iterations reached")`
- `surpyval/univariate/parametric/fitters/mle.py:106, 115` — fit-failure diagnostics to stderr
- `surpyval/univariate/parametric/fitters/mps.py:78` — `print("MPS FAILED: ...")`

`warnings.warn` lets users filter or escalate these; `print` cannot be silenced or caught.

### Bare `except:` swallows all exceptions
**File:** `surpyval/univariate/regression/accelerated_life/parameter_substitution.py:224`

`except:  # noqa: E722` catches `KeyboardInterrupt`/`SystemExit` and hides genuine errors in the per-stress initial fits. Use `except Exception:` at minimum.

### `type(x) == Cls` instead of `isinstance()`
Fails for subclasses (e.g. pandas subclassed Series). Occurrences include:

- `surpyval/utils/__init__.py:434, 435, 448`
- `surpyval/univariate/parametric/parametric.py:263, 303, 342, 390, 429, 466`
- `surpyval/univariate/parametric/series.py:10, 16` and `parallel.py:10, 12, 18, 20`
- `surpyval/univariate/regression/accelerated_life/parameter_substitution.py:171`

### Bitwise `&`/`|` used for logical conditions
Widespread in `surpyval/utils/__init__.py` (lines 49, 181, 189, 199, 409, 414, 417, 545, 1037, …) and in the regression fitters (`accelerated_failure_time.py:52`, `proportional_hazards_fitter.py:74`, `custom_distribution.py:105`, `parametric.py:554`). On scalars use `and`/`or` — idiomatic, short-circuiting, and avoids precedence surprises.

### Deprecated `importlib.resources.path` in dataset loaders
**File:** `surpyval/datasets/__init__.py:162` (and the other loaders in the module)

Every test run emits `DeprecationWarning: path is deprecated. Use files() instead.` The API is slated for removal; migrate to `importlib.resources.files(data_module) / "lung.csv"`.

### Loop-variable shadowing in `parameter_substitution.py`
`hf()` reassigns the function's `*params` inside its loop (line 99) and `random()` reassigns `dist_params` inside its loop (line 179). Both currently produce correct results because only the life-parameter slot changes per iteration, but the shadowing makes the code fragile to reordering. Use a distinct name (`dist_params_i`) as `Hf()` already does.

---

## 6. CI / Tooling

**File:** `.github/workflows/actions.yml`

- **`black $SRC` does not check** — without `--check`, black reformats the CI workspace and always exits 0, so formatting violations never fail CI.
- **Single-version matrix** — CI tests only Python 3.12 while `pyproject.toml` declares 3.11–3.13 support. Use a matrix over 3.11/3.12/3.13.
- **Lint runs after tests** — `flake8`/`mypy`/`black` only run if pytest passes; a test failure hides lint results. Run them as a separate job or before the (much slower) test step.
- **Stale pre-commit pins** — `.pre-commit-config.yaml` pins black 23.1.0 / flake8 6.0.0 / isort 5.12.0 (2023 era) while CI installs latest, so pre-commit and CI can disagree on formatting. The flake8 hook also duplicates `--ignore`/`--per-file-ignores` as args instead of reading the `[tool.flake8]` config (the hook env lacks `flake8-pyproject`).
- **`pytest==9.0.3` exact pin** in the `tests` extra — use a lower bound (`pytest>=9`) so downstream environments can resolve.

---

## 7. Long-term: Replace `autograd` with JAX (deferred)

`autograd` (HIPS/autograd) is in low-activity maintenance mode with no GPU support. JAX is the spiritual successor and a near-drop-in replacement for `autograd.numpy` patterns. The interim steps (inlining the `autograd_gamma` gradients into `surpyval/utils/autograd_gamma_compat.py` and upgrading to `autograd` 1.8 for numpy 2.x compatibility) are done, so there is no urgency. A JAX migration can be revisited once the library is otherwise stable — it is a multi-week effort touching every gradient computation.
