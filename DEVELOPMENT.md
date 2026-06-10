# Development Notes

This document tracks known issues, technical debt, and improvement priorities for surpyval. Issues are grouped by theme and ordered by severity within each section.

---

## Priority Order

1. Correctness (silent wrong results)
2. Stability (features that crash)
3. Test infrastructure
4. Packaging
5. Architecture

---

## 1. MLE Optimizer Bugs

**File:** `surpyval/univariate/parametric/fitters/mle.py`

### Wrong optimizer name recorded (lines 72-93, 165)
`results['optimizer'] = method` uses the loop variable, which always holds `'Newton-CG'` (the last iteration) regardless of which method produced the best result. Tag `best_result` with the winning method name inside the loop.

### Wrong confidence intervals for offset/zi/lfp models (lines 149-158)
The `numdifftools` Hessian fallback uses `Hessian(lambda x: fun(x, False, False, False, False))`, passing only 4 of the 7 required arguments. `gamma`, `f0`, and `p` silently default to `0, 0, 1`, so the Hessian is computed at a different function than the one optimized. All confidence intervals for offset, zero-inflated, and LFP models are wrong.

**Fix:** Forward all arguments: `Hessian(lambda x: fun(x, False, False, False, False, gamma, f0, p))`.

### Hessian computed in wrong parameterization space
Standard errors should be derived from the Hessian in the *transformed* (unbounded) parameterization used during optimization, then mapped back via the delta method. Computing in physical parameter space gives incorrect standard errors for any bounded parameter.

### Optimizer cascade does not warm-start (lines 72-91)
All five methods (Nelder-Mead → Powell → BFGS → TNC → Newton-CG) start from the same cold `init`. Gradient methods should warm-start from the best-found point so far.

---

## 2. Test Coverage Gaps

### No random seed in parametric tests
**Files:** `surpyval/tests/test_fit.py`, `surpyval/tests/test_np.py`

All tests generate random parameters and samples with no `np.random.seed()` call. The `FIT_SIZES = [1000, 10000, 100000]` retry loop masks intermittent failures and makes CI failures impossible to reproduce.

### Entirely untested subsystems

| Subsystem | Files |
|-----------|-------|
| `CompetingRisks`, `FineGray`, `CompetingRisksProportionalHazard` | `surpyval/competing_risks/` |
| `MixtureModel`, `SeriesModel`, `ParallelModel` | `surpyval/univariate/parametric/` |
| Recurrence/NHPP (`Crow`, `Duane`, `CoxLewis`, `CrowAMSAA`) | `surpyval/recurrence/parametric/` |
| `AcceleratedFailureTime` regression variants | `surpyval/regression/` |
| Serialization (`to_dict` / `from_dict`) | `test_to_dict.py` is 0 bytes |

### `test_mps_truncated[Beta]` consistently fails
**File:** `surpyval/tests/test_fit.py::test_mps_truncated[Beta-random_parameters]`

The MPS truncated fit for the Beta distribution consistently fails its 10 % convergence tolerance at n=2000. Root cause is unclear without a fixed seed — it may be a genuine convergence problem in the MPS fitter for Beta under truncation (CDF values are close together, spacing objective is ill-conditioned), or the tolerance is too tight for the sample size. Before investing in the fitter: add a seed, establish a baseline, and check whether the fit converges at n=100 000. If it still diverges, the MPS optimiser path for Beta needs attention; if it passes at a larger n, widen the tolerance or increase the sample size.

### Tests only check statistical convergence, not mathematical correctness
~~The current tests only check statistical convergence. Add `tests/test_distributions_math.py` with closed-form checks: verify `sf(x) + ff(x) == 1`, `qf(0.5)` matches the known median, and `random()` samples have expected mean/variance for each distribution (with a fixed seed).~~
**Done** — `surpyval/tests/test_distributions_math.py` added with all three checks.

### ~~`GumbelLEV.qf` is the inverse of `sf`, not `ff`~~ ✓ Done
**File:** `surpyval/univariate/parametric/gumbel_lev.py`

~~`GumbelLEV.qf(p)` returns the value `x` where `sf(x) = p`, so `ff(qf(p)) = 1-p` instead of `p`. All other distributions define `qf` as the inverse of `ff` (the CDF). The fix is to make `GumbelLEV.qf` return the value where `ff(x) = p`, consistent with the rest of the API. Caught by `test_qf_ff_roundtrip[GumbelLEV]` (currently `xfail`).~~
**Fixed** — `qf` now returns `mu - sigma * log(-log(p))` (inverse of `ff`); `xfail` marker removed from `test_qf_ff_roundtrip[GumbelLEV]`.

---

## 3. Packaging ✓ Complete

### ~~Incomplete numba removal~~ ✓ Done
No live numba import remains. `from numba import njit` and `@njit` decorators
are commented out in `cox_ph.py`; numba removed from `install_requires`,
`requirements.txt`, and mypy overrides in `pyproject.toml`.

### ~~Wrong `python_requires`~~ ✓ Done
Updated to `>=3.11` in `pyproject.toml`.

### ~~`requirements.txt` pins exact production dependency versions~~ ✓ Done
Production deps moved to `pyproject.toml` with loose bounds. `requirements.txt`
is now a redirect comment only. `requirements_dev.txt` uses `-e .[tests]`.

### ~~`pyproject.toml` has no `[build-system]` or `[project]` table~~ ✓ Done
Fully migrated from `setup.py` to `pyproject.toml`. `setup.py` deleted.

---

## 4. API Consistency Issues

### ~~Regression `fit()` argument order~~ ✓ Done
~~`CoxPH.fit(x, Z, ...)` takes time first, while `ProportionalHazardsFitter`, `ParameterSubstitutionFitter`, and `AcceleratedFailureTimeFitter` all take `(Z, x, ...)`. Switching between models silently swaps time and covariates.~~
**Fixed** — all three fitters now use `fit(x, Z, ...)`, matching `CoxPH`, the forest models, and the proportional intensity models. All internal call sites already used keyword arguments, so no callers needed updating.

### ~~CoxPH docstring inverts censoring convention~~ ✓ Done
Fixed: `c` parameter now correctly documented as `0=event, 1=right-censored`.

### ~~`R_cb()` returns `ValueError` objects instead of raising them~~ ✓ Done
Fixed: `return ValueError(...)` changed to `raise ValueError(...)` in
`surpyval/univariate/nonparametric/nonparametric.py`.

### `cb()` default `on` parameter differs between `Parametric` and `NonParametric`
`Parametric.cb()` defaults to `on='R'`; `NonParametric.cb()` defaults to `on='sf'`. Both strings refer to the survival function in the same conditional.

### `neg_ll()` guard condition is broken
**File:** `surpyval/univariate/parametric/parametric.py:843-847`

`from_params()` sets `self.data = None`, so `if not hasattr(self, 'data')` is always `False`. Falls through to `return self._neg_ll` which raises an uninformative `AttributeError`. Check `if self.data is None` or `if not hasattr(self, '_neg_ll')` instead.

### `ParametricFitter` does not inherit `Distribution` ABC
The `Distribution` ABC in `distribution.py` is effectively unused — `ParametricFitter`, `NonParametric`, `MixtureModel`, `NeverOccurs`, and `InstantlyOccurs` all expose the same interface but none inherit from it. Users cannot write polymorphic code using `isinstance(model, Distribution)`.

### Parameter naming inconsistency (`alpha`/`beta`)
Weibull, LogLogistic, Gamma, Beta, and ExpoWeibull all use `param_names=['alpha', 'beta']` but the roles differ across distributions. Positional use will produce silently incorrect results.

---

## 5. Long-term: Replace `autograd` with JAX (deferred)

`autograd` (HIPS/autograd) is in low-activity maintenance mode with no GPU support. JAX is the spiritual successor and a near-drop-in replacement for `autograd.numpy` patterns. However, migrating every gradient computation in the library is a multi-week effort and is not the right next step. The near-term plan (§6 Phase 5) is to inline the `autograd_gamma` gradients and update `autograd` to 1.8, which gives numpy 2.x compatibility without a large refactor. A JAX migration can be revisited once the library is otherwise stable.

---

## 6. Proposed Package Hierarchy

### Organising principle

The current structure mixes two different axes: it groups by *data type* at the top level (`univariate/`, `recurrence/`, `competing_risks/`) but then breaks the pattern by pulling all regression into a single top-level `regression/` module regardless of what data type it applies to. The proposed structure applies one consistent rule: **data type first → covariate second → estimation method third**. A user's first question is always "what kind of data do I have?" — regression, nonparametric, and parametric are sub-choices within each answer.

### Proposed structure

```
surpyval/
│
├── univariate/              # One event per subject
│   ├── nonparametric/       # KM, NA, FH, Turnbull
│   ├── parametric/          # Weibull, Gamma, Exponential, etc.
│   └── regression/          # Covariates on a single event time
│       ├── proportional_hazards/     # Cox (semi-param), WeibullPH, ExpPH
│       ├── accelerated_failure_time/ # AFT (log-linear on T)
│       ├── accelerated_life/         # ALT: parameter substitution + life models
│       │                             # (Power, Eyring, Arrhenius, DualPower, ...)
│       └── forest/                   # RandomSurvivalForest, SurvivalTree
│
├── competing_risks/         # Multiple causes; only one event wins per subject
│   ├── nonparametric/       # Cause-specific KM, empirical CIF (currently missing)
│   ├── parametric/          # CompetingRisks model
│   └── regression/          # Fine-Gray, CompetingRiskPH
│
├── recurrent/               # Multiple events per subject
│   ├── nonparametric/       # MCF (mean cumulative function)
│   ├── parametric/          # HPP, NHPP, Crow-AMSAA, Duane, Cox-Lewis
│   ├── renewal/             # Kijima virtual-age models (Kijima I & II)
│   └── regression/          # ProportionalIntensity HPP/NHPP
│
└── multivariate/            # Correlated event times across subjects/components (future)
    ├── nonparametric/       # Multivariate KM extensions, cross-ratio estimators
    ├── parametric/          # Copula-based joint survival, shared frailty models
    ├── multi_state/         # State-transition models (e.g. illness-death model)
    └── regression/          # Frailty models with covariates, joint longitudinal models
```

### Changes from current layout

| Current | Proposed | Reason |
|---------|----------|--------|
| `regression/` (top-level) | `univariate/regression/` | Cox, AFT, ALT all apply to single-event data; they belong under `univariate/` |
| `regression/lifemodels/` | `univariate/regression/accelerated_life/` | ALT is a restricted parameterisation of regression, not a separate concept |
| `recurrence/` + `renewal/` (separate top-level modules) | `recurrent/` (merged) | Both answer "repeated events per subject"; renewal is recurrence with virtual age |
| `competing_risks/` (flat) | `competing_risks/{nonparametric,parametric,regression}/` | Mirrors the sub-structure of every other module |
| (absent) | `multivariate/` | Natural extension point; frailty and copula models fit here |

### Notes on the regression sub-hierarchy

PH, AFT, and ALT are distinct enough to keep in separate sub-modules but related enough to live together under `univariate/regression/`:

- **Proportional Hazards** — covariate multiplies the hazard: `h(t|Z) = h₀(t) exp(β'Z)`. Baseline is either unspecified (Cox, semi-parametric) or fully parametric (Weibull-PH, Exponential-PH).
- **Accelerated Failure Time** — covariate scales time directly: `log T = β'Z + ε`. Note: Weibull is both PH and AFT simultaneously, worth documenting explicitly.
- **Accelerated Life Testing (ALT)** — one distribution parameter is itself a deterministic function of a physical stress variable via an engineering relationship (Arrhenius, inverse power law, Eyring, dual-stress models). This is a restricted form of AFT/PH specific to reliability and test design. Keeping it separate from general AFT avoids conflating statistical and engineering parameterisations.

### Notes on "counting processes"

"Counting processes" is a statistical *framework* (martingale theory, stochastic integrals) not a data type. The module is named `recurrent/` because users navigate by their data, not the underlying mathematics. The counting-process formulation is an implementation detail that can be noted in docstrings.

### Notes on multivariate (future work)

The natural build-out order within `multivariate/`:

1. **Copula models** (`parametric/`) — joint survival `S(t₁, t₂)` factored through a copula family (Clayton, Gumbel, Frank). Tractable starting point with few new abstractions.
2. **Shared frailty** (`parametric/`) — subjects in a cluster share a latent gamma or log-normal random effect; extends univariate PH naturally and is a mild generalisation of existing fitters.
3. **Multi-state models** (`multi_state/`) — transitions between discrete states (healthy → sick → dead; illness-death model). Requires a transition-intensity matrix rather than a single survival function, so warrants its own sub-module.
4. **Regression extensions** (`regression/`) — frailty models with fixed-effect covariates; joint models for longitudinal biomarker + event time.

---

## 7. Dependency Modernisation Plan

Original assessment date: 2026-06-10. Status updated as phases complete.
**All phases are now complete except Phase 4 (pandas 3.x).**

### Python version target — ✅ COMPLETED

`requires-python = ">=3.11"` is set in `pyproject.toml` and the
3.11/3.12/3.13 classifiers are declared. The stale `.python-version`
file has been removed.

| Version | Status as of Jun 2026 | Verdict |
|---------|----------------------|---------|
| 3.8–3.10 | EOL / EOL in months | Dropped |
| 3.11 | Security-only, EOL Oct 2027 | **Supported** — new minimum |
| 3.12 | Security-only, EOL Oct 2028 | **Supported** |
| 3.13 | Active bugfix, EOL Oct 2029 | **Supported** |

### Dependency status

| Package | Status | Notes |
|---------|--------|-------|
| numpy | ✅ 2.4.6 (`>=2.1,<3`) | Phase 3 complete |
| scipy | ✅ 1.17.1 (`>=1.17`) | Phase 2 complete |
| formulaic | ✅ 1.2.2 (`>=1.2`) | Phase 2 complete |
| autograd | ✅ 1.8.0 (`>=1.8`) | Phase 1 complete |
| matplotlib | ✅ 3.10.9 (`>=3.10`) | Phase 1 complete |
| numdifftools | ✅ 0.9.42 (`>=0.9.42`) | Phase 1 complete |
| autograd-gamma | ✅ Removed | Phase 5 complete — inlined |
| numpy-indexed | ✅ Removed | Replaced with pure-numpy `group_by` in `cox_ph.py` |
| pandas | ⏳ 2.x (unpinned) | **Phase 4 — the only remaining phase** |

### Upgrade phases

#### Phase 1 — Low-hanging fruit — ✅ COMPLETED

`numdifftools>=0.9.42`, `matplotlib>=3.10`, and `autograd>=1.8` (numpy 2.x
compat) are set in `pyproject.toml`. `numpy-indexed` was not bumped but
removed entirely — its only use (`group_by` in `cox_ph.py`) was replaced
with a pure-numpy implementation.

#### Phase 2 — scipy and formulaic — ✅ COMPLETED (June 2026)

- `scipy`: now `>=1.17` (tested against 1.17.1). All scipy usage in the
  codebase (`optimize.minimize`/`root`, `special`, `stats`,
  `interpolate.interp1d`, `sparse.dok_matrix`, `integrate.quad`) is
  stable API; no code changes were required. Smoke tests with
  `DeprecationWarning` raised as an error pass on the interp1d,
  dok_matrix, and pearsonr paths. Note `interp1d` is classed as
  *legacy* by scipy — if it is ever removed, migrate
  `univariate/nonparametric/nonparametric.py` to
  `scipy.interpolate.make_interp_spline` or `np.interp`.
- `formulaic`: now `>=1.2` (tested against 1.2.2). The only call site,
  `wrangle_and_check_form_and_Z_cols` in `surpyval/utils/__init__.py`
  (`Formula(...)` + `get_model_matrix(df, na_action="ignore")`), works
  unchanged under the 1.x API, including categorical encoding and NA
  masking. The full regression test suite (`test_regression.py`,
  including `test_formula_interface_matches_Z_cols`) passes.

#### Phase 3 — numpy 2.x (breaking) — ✅ COMPLETED (June 2026)

The codebase runs on numpy 2.4.6 with the full test suite passing.
`numpy>=2.1,<3` is pinned in `pyproject.toml`, `np.in1d` was replaced
with `np.isin`, and size-1-array-to-scalar conversions (an error since
numpy 2.x) were fixed in `Logistic.moment`, the Gamma MPP fitter, and
the test helpers.

#### Phase 4 — pandas 3.x (breaking) — ⏳ REMAINING

The last outstanding phase. pandas 3.0 introduced mandatory Copy-on-Write
(chained assignment silently fails), changed string columns from `object`
dtype to `str` dtype, changed datetime resolution to `us`, and removed
long-deprecated offset aliases.

Early signal: surpyval's own test suite (excluding lifelines-dependent
comparison tests) passes under pandas 3.0.3 — surpyval mostly consumes
DataFrames via `.values` rather than mutating them. The blockers are:

1. `lifelines` (test dependency) requires `pandas<3`, so the dev
   environment resolves to pandas 2.x. Either wait for a
   pandas-3-compatible lifelines release or move the comparison tests to
   a separate optional CI job.
2. A proper audit of `surpyval/utils/` data ingestion for chained
   assignment and `dtype == object` string checks has not been done.

Steps:
1. Upgrade dev environment to pandas 2.3 (last 2.x), fix all
   `FutureWarning` / `DeprecationWarning` output.
2. Audit `surpyval/utils/` ingestion code, then pin `pandas>=3` once the
   lifelines constraint is resolved.

#### Phase 5 — inline autograd-gamma and drop the dependency — ✅ COMPLETED

`surpyval/utils/autograd_gamma_compat.py` now provides the
`gammainc`/`gammaincc`/`betainc` primitives (and their `ln` variants)
with `defvjp` registrations, replacing the abandoned `autograd-gamma`
package. All imports across the codebase point at the compat module and
`autograd-gamma` is no longer a dependency. (It can still appear in dev
environments as a transitive dependency of `lifelines`.)

### Current `[project.dependencies]` (pyproject.toml)

```toml
dependencies = [
    "autograd>=1.8",          # or remove when JAX migration complete
    "numpy>=2.1,<3",
    "scipy>=1.17",
    "pandas",                 # pin >=3 after Phase 4
    "matplotlib>=3.10",
    "formulaic>=1.2",
    "numdifftools>=0.9.42",
    "joblib",
]
```

### CI matrix recommendation

With Python 3.11 as the minimum, test against: **3.11, 3.12, 3.13** on Linux (ubuntu-latest). All jobs now run numpy 2.x / scipy 1.17 / formulaic 1.2 by default. Add a pandas-3 job once Phase 4 lands.

---

## 8. `SurpyvalData` Class Issues

**File:** `surpyval/utils/surpyval_data.py`  
**Review date:** 2026-06-10

### Bugs

#### `to_json` crashes when `Z` is `None` (line 258)
`Z` is always set in `__init__` (to the array or `None`), so `hasattr(self, 'Z')` is always `True`. When `Z` is `None`, `None.tolist()` raises `AttributeError`. Fix: change the guard to `if self.Z is not None`.

#### `__getitem__` crashes on scalar integer index (line 201)
When `index` is an integer, `self.t[index]` has shape `(2,)`. `__init__` then does `self.t[:, 0]` on that 1D array, raising `IndexError`. The existing test only uses slice indexing (`data[0:2]`), so this is hidden. Fix: normalize scalar indices to a slice before passing to the constructor:

```python
def __getitem__(self, index):
    if isinstance(index, (int, np.integer)):
        index = slice(index, index + 1)
    return SurpyvalData(self.x[index], self.c[index], self.n[index], self.t[index], handle=False)
```

### Design Issues

#### Inconsistent types for interval split attributes (line 147)
When no interval data is present, `x_il`, `x_ir`, `n_i` are assigned Python lists `[]`. When interval data exists (lines 143–145) they are numpy arrays. Downstream code using `.shape`, `np.concatenate`, or arithmetic will behave differently depending on whether interval data was present. Replace with `np.array([], dtype=float)` / `np.array([], dtype=int)`.

#### Fragile path detection in `from_json` (line 283)
```python
if isinstance(json_str, (str, Path)) and Path(json_str).exists():
```
If the caller passes a real JSON string that coincidentally matches an existing file path, the method silently reads the file instead. For very long JSON strings, `Path(json_str)` can raise `OSError` on some platforms. Fix: branch explicitly on type — treat `Path` instances as file paths and `str` as JSON text. Add a `from_json_file` classmethod if string-as-path is needed.

#### Stateful iterator breaks nested loops (line 210)
`__iter__` returns `self` and resets `_index` to 0. A nested or restarted `for` loop over the same instance resets the index mid-iteration. Standard fix: return a generator rather than `self`:

```python
def __iter__(self):
    return zip(self.x, self.c, self.n, self.t)
```

This also removes the need for `__next__` and `_index`.

#### Missing `__len__`
`__iter__` and `__next__` are defined but `len(data)` raises `TypeError`. Add:

```python
def __len__(self) -> int:
    return len(self.x)
```

#### `x_min`/`x_max` semantics are implicit for 2D (interval) data (line 108)
`np.min(x)` and `np.max(x)` on a 2D interval array return the global min/max across both left and right bounds. This is probably the right answer but is non-obvious. Make it explicit:

```python
if x.ndim == 2:
    self.x_min, self.x_max = np.min(x[:, 0]), np.max(x[:, 1])
else:
    self.x_min, self.x_max = np.min(x), np.max(x)
```

### Style

#### Bitwise `|` used for boolean conditions in `to_xrd` (line 180)
```python
if (
    np.isfinite(self.t[:, 1]).any()
    | (2 in self.c)
    | (-1 in self.c)
):
```
`|` is bitwise OR. Use `or` for logical conditions — it is idiomatic and supports short-circuit evaluation.

#### Docstring copy-paste artifacts (lines 28–34)
"The data is stored in the x, c, n, and t attributes of the object." is repeated three times; the final sentence is cut off mid-phrase. Line 29 has the typo "convertdata" (missing space). Clean up the class docstring body.

### Related: `utils/__init__.py`

- **`type(x) == list` / `type(x) == Series`** at lines 433 and 447 — use `isinstance()` to handle subclasses correctly.
- **`&` instead of `and`** for `None` checks in `fsli_handler` (lines 179, 187) and elsewhere — bitwise AND on booleans works but is incorrect idiom and prevents short-circuit evaluation.

### Priority summary

| # | Severity | Location | Issue |
|---|----------|----------|-------|
| 1 | **Bug** | `to_json:258` | `hasattr` always True; crashes when `Z` is `None` |
| 2 | **Bug** | `__getitem__:201` | Scalar index causes crash via `self.t[:, 0]` on 1D array |
| 3 | Design | `_split_to_observation_types:147` | Python list vs numpy array inconsistency for empty interval attributes |
| 4 | Design | `from_json:283` | Fragile str-as-path detection |
| 5 | Design | `__iter__:210` | Stateful iterator; nested loops corrupt state |
| 6 | Design | — | Missing `__len__` |
| 7 | Minor | `__init__:108` | 2D `x_min`/`x_max` spans both bounds implicitly |
| 8 | Style | `to_xrd:180` | Bitwise `\|` used for boolean OR |
| 9 | Docs | `__init__:28` | Repeated/truncated docstring sentences, typo |
