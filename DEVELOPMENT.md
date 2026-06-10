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

### `GumbelLEV.qf` is the inverse of `sf`, not `ff`
**File:** `surpyval/univariate/parametric/gumbel_lev.py`

`GumbelLEV.qf(p)` returns the value `x` where `sf(x) = p`, so `ff(qf(p)) = 1-p` instead of `p`. All other distributions define `qf` as the inverse of `ff` (the CDF). The fix is to make `GumbelLEV.qf` return the value where `ff(x) = p`, consistent with the rest of the API. Caught by `test_qf_ff_roundtrip[GumbelLEV]` (currently `xfail`).

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

### Regression `fit()` argument order
`CoxPH.fit(x, Z, ...)` takes time first, while `ProportionalHazardsFitter`, `ParameterSubstitutionFitter`, and `AcceleratedFailureTimeFitter` all take `(Z, x, ...)`. Switching between models silently swaps time and covariates.

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

Assessment date: 2026-06-10. All installed versions are from the current development environment.

### Python version target

| Version | Status as of Jun 2026 | Verdict |
|---------|----------------------|---------|
| 3.8 | EOL Oct 2024 | **Drop** |
| 3.9 | EOL Oct 2025 | **Drop** |
| 3.10 | Security-only, EOL Oct 2026 | **Drop** — EOL in months |
| 3.11 | Security-only, EOL Oct 2027 | **Support** — new minimum |
| 3.12 | Security-only, EOL Oct 2028 | **Support** |
| 3.13 | Active bugfix, EOL Oct 2029 | **Support** |
| 3.14 | Active bugfix, EOL Oct 2030 | **Support** |

**Action:** change `python_requires=">=3.8"` → `">=3.11"` in `setup.py`. Update `.python-version` from 3.10.4 to 3.13.x for local development.

### Dependency audit

| Package | Installed | Latest | Risk | Notes |
|---------|-----------|--------|------|-------|
| numpy | 1.23.5 | 2.4.6 | **High** | Major ABI/API break at 2.0 |
| pandas | 1.5.3 | 3.0.3 | **High** | Two major versions behind |
| autograd-gamma | 0.5.0 | 0.5.0 | **High** | Abandoned since 2020; no numpy 2.x testing |
| autograd | 1.5 | 1.8.0 | Medium | numpy 2.x support added in 1.7; low-activity maintenance |
| formulaic | 0.5.2 | 1.2.2 | Medium | Pre-1.0 → 1.2; API redesigned at 1.0 |
| scipy | 1.10.0 | 1.17.1 | Medium | 7 minor versions behind; needs numpy ≥1.26 |
| numpy-indexed | 0.3.5 | 0.3.7 | Low | Dormant; no numpy 2.x CI |
| matplotlib | 3.7.1 | 3.10.9 | Low | Deprecation cleanups only |
| numdifftools | 0.9.41 | 0.9.42 | Low | One patch release |

### Upgrade phases

#### Phase 1 — Low-hanging fruit (no breaking changes expected)

- `numdifftools`: 0.9.41 → 0.9.42
- `matplotlib`: 3.7.1 → 3.10.9
- `autograd`: 1.5 → 1.8.0 (numpy 2.x compat patch)
- `numpy-indexed`: 0.3.5 → 0.3.7

These are safe to bump in a single PR. Run the existing test suite to confirm.

#### Phase 2 — scipy and formulaic

- `scipy`: 1.10.0 → 1.17.1. Requires numpy ≥1.26 — do this after Phase 3.
- `formulaic`: 0.5.2 → 1.2.2. The `SimpleFormula`/`StructuredFormula` redesign (v1.0) and the `ModelSpec.required_variables` change (v1.2) will affect the regression formula parsing in `surpyval/regression/`. Audit all `formulaic` call sites before bumping.

#### Phase 3 — numpy 2.x (breaking)

numpy 2.0 removed ~100 deprecated aliases (`np.bool`, `np.int`, `np.float`, `np.complex`, `np.object`, `np.str` → use Python built-ins), moved `np.core` to private `np._core`, and changed scalar type promotion rules. This will break code in surpyval and possibly in `autograd` (see §5).

Steps:
1. Run `ruff check --select NPY201 .` to auto-fix removed type aliases.
2. Audit `np.core` usage (none expected in surpyval itself, but check via `grep -r "np\.core" .`).
3. Pin `numpy>=2.0,<3` in `install_requires` once verified.
4. Re-run the full test suite under numpy 2.x — pay attention to scalar promotion differences in log-likelihood computations.

Do Phase 3 before Phase 2 (scipy 1.17 requires numpy ≥1.26 and is compatible with numpy 2.x).

#### Phase 4 — pandas 3.x (breaking)

pandas 3.0 introduced mandatory Copy-on-Write (chained assignment silently fails), changed string columns from `object` dtype to `str` dtype, changed datetime resolution to `us`, and removed long-deprecated offset aliases.

Steps:
1. First upgrade to pandas 2.3 (last 2.x), fix all `FutureWarning` / `DeprecationWarning` output.
2. Then upgrade to 3.0.3.
3. Audit all `surpyval/utils/` data ingestion code for chained assignment patterns and `dtype == object` string checks.

#### Phase 5 — inline autograd-gamma and drop the dependency (HIGH PRIORITY)

`autograd-gamma` has had no release since October 2020 and has no numpy 2.x testing. It is a small package: it registers custom VJPs for the incomplete gamma function (`scipy.special.gammainc` / `gammaincc`) with autograd's `primitive` system. The math is known in closed form.

**Plan: inline the gradients, remove the package.**

The gradient of the regularised lower incomplete gamma function `P(a, x) = gammainc(a, x)` is:

- ∂P/∂x = exp(-x) * x^(a-1) / Γ(a)  — straightforward
- ∂P/∂a = ∂/∂a of the series — this is the hard part; it requires the derivative of the regularised incomplete gamma w.r.t. the shape parameter, which can be computed via the series expansion from Batir (2008) or approximated numerically with `scipy.special.polygamma`

The simplest correct implementation registers two `autograd.primitive` wrappers (one for `gammainc`, one for `gammaincc`) with their `defvjp` calls. This is ~50–80 lines and has no external dependencies beyond `autograd` and `scipy.special`.

Steps:
1. Create `surpyval/utils/autograd_gamma_compat.py` with the inlined primitives.
2. Replace all `from autograd_gamma import ...` imports across the codebase with imports from the new module.
3. Remove `autograd_gamma` from `install_requires` in `setup.py`.
4. Pin `autograd>=1.8` — autograd 1.8 has numpy 2.x support, so this also unblocks Phase 3.

Until this is done, pin `autograd-gamma==0.5.0` explicitly and do not bump numpy past 1.x.

### Target `install_requires` after full upgrade

```python
install_requires=[
    "autograd>=1.8",          # or remove when JAX migration complete
    "numpy>=2.0,<3",
    "scipy>=1.17",
    "pandas>=3.0",
    "matplotlib>=3.10",
    "formulaic>=1.2",
    "numdifftools>=0.9.42",
    # numpy_indexed and autograd_gamma: replace or inline (see above)
],
python_requires=">=3.11",
```

### CI matrix recommendation

Once Python 3.11 is the minimum, test against: **3.11, 3.12, 3.13** on Linux (ubuntu-latest). Add a numpy-2.x job as soon as Phase 3 is underway. Drop macOS-only local dev assumption from `.python-version`.

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
