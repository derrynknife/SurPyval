---
name: surpyval
description: Use this skill for any survival, reliability, or time-to-event analysis with the SurPyval Python package (`import surpyval`). Covers fitting parametric distributions (Weibull, LogNormal, Exponential, ...), non-parametric estimators (Kaplan-Meier, Nelson-Aalen, Turnbull), regression (AFT, proportional hazards, Cox, additive hazards, Buckley-James, proportional odds), competing risks, recurrent events (NHPP/HPP, renewal/imperfect-repair, MCF), degradation/RUL, multivariate copulas, and model serialisation. Trigger whenever the user works with censored, truncated, or interval data, mentions `surpyval`/`Weibull.fit`, reliability/failure-time/hazard/survival curves, or the xcnt data model.
---

# SurPyval

SurPyval is a survival-analysis package whose defining strength is that **every
estimator accepts arbitrary combinations of observed, censored, and truncated
data** through one consistent input convention (the "xcnt" data model). Fit a
model, then call `sf`/`ff`/`hf`/`Hf`/`df`/`qf` on it.

Current version: 0.15.2. Import as `import surpyval` (commonly `import surpyval as sp`).

## The universal fit pattern

Almost everything is a fitter object with a `.fit(...)` classmethod that returns
a fitted **model** object:

```python
import surpyval as sp

model = sp.Weibull.fit(x=[10, 12, 8, 9, 11, 13])   # returns a Parametric model
model.params        # fitted parameters, e.g. array([alpha, beta])
model.sf(10)        # survival / reliability function
model.ff(10)        # CDF / failure function  (== 1 - sf)
model.hf(10); model.Hf(10)   # hazard, cumulative hazard
model.df(10)        # density
model.qf(0.5)       # quantile (median here)
model.mean(); model.var(); model.moment(1)
model.cb(10, alpha=0.05)     # confidence bounds on the function
model.aic(); model.bic(); model.neg_ll()   # fit diagnostics
model.plot()        # probability plot with the empirical overlay
```

`sp.fit_best(x, ...)` fits several distributions and returns the best by AIC.

## The xcnt data model (the thing to get right)

Every fitter's `fit()` takes the same optional arrays. Only `x` is usually needed.

- **`x`** — the observed values. For interval-censored points, an entry may be a
  `[left, right]` pair, OR pass interval bounds separately as `xl`/`xr`.
- **`c`** — censoring flag per observation, in `{-1, 0, 1, 2}`:
  - `0` = **observed** (exact event) — the default when `c` is omitted
  - `1` = **right censored** (event after `x`)
  - `-1` = **left censored** (event before `x`)
  - `2` = **interval censored** (`x` entry is `[left, right]`)
- **`n`** — integer count/weight at each `x` (repeat-observation shorthand).
- **`t`** — truncation, a two-column `[tl, tr]`; or pass **`tl`**/**`tr`** (left/right
  truncation) separately. Use `np.inf`/`-np.inf` for one-sided truncation.

```python
# observed + right + left censoring, with counts and left truncation
sp.Weibull.fit(x=[1, 2, 3, 4, 5],
               c=[0, 0, 1, -1, 0],
               n=[1, 2, 1, 1, 3],
               tl=[0, 0, 0, 1, 0])

# interval censored via xl/xr
sp.Weibull.fit(xl=[1, 2, 3], xr=[2, 4, 5])
```

**Gotcha:** a right-censored row (`c=1`) with a *finite* right-truncation `tr` is
contradictory and now emits a warning (right truncation means the event was seen
before `tr`; right censoring says it is after `x`) — such data can make the
likelihood unbounded.

Extra `fit` options on parametric distributions: `how` (`"MLE"` default, or
`"MPP"`/`"MSE"`/`"MOM"`), `offset=True` (fit a location/threshold `gamma`, e.g.
3-parameter Weibull), `zi=True` (zero-inflation), `lfp=True` (limited failure
population / cure fraction), `fixed={"beta": 2}` (hold parameters), `init=[...]`.
**Note:** `offset`/`zi`/`lfp` are univariate-distribution features only — the
**regression** fitters do not accept them (they raise `TypeError`).

### DataFrame entry
Most fitters also offer `fit_from_df(df, ...)` mapping column names to the xcnt
roles (regression fitters map a formula / covariate columns).

## What lives where

| Task | Import | Fitters |
|---|---|---|
| **Parametric distributions** | `sp.<Name>` | Weibull, Exponential, Gamma, LogNormal, Normal, Gumbel, Logistic, LogLogistic, ExpoWeibull, Rayleigh, Beta, Uniform, and discrete (Poisson, Binomial, Geometric, NegativeBinomial, DiscreteWeibull, ...) |
| **Non-parametric** | `sp.<Name>` | KaplanMeier, NelsonAalen, FlemingHarrington, **Turnbull** (NPMLE for the full data model incl. interval + truncation) |
| **Regression (parametric)** | `sp.<Dist><Kind>` or `sp.AFT/PH/PO/AH(dist)` | AFT, PH (proportional hazards), PO (proportional odds), AH (additive hazards). E.g. `sp.WeibullPH`, `sp.LogNormalAFT`. Fit with `(x, Z, c, n, t)`; predict `sf(x, Z)`. |
| **Semi-parametric** | `sp.CoxPH` | Cox proportional hazards (Breslow baseline); also `CoxPH.fit_tvc` / `fit_tvc_from_df` for time-varying covariates. `sp.BuckleyJames` (AFT), `sp.AdditiveHazards` (Lin–Ying). |
| **Competing risks** | `surpyval.univariate.competing_risks` | `CompetingRisks` (nonparametric CIF), `ParametricCompetingRisks`, `FineGray` (subdistribution regression), `CompetingRisksProportionalHazards` |
| **Recurrent events** | `surpyval.recurrent` | `NonParametricCounting` (MCF), NHPP/HPP intensity fits (`CrowAMSAA`, `Duane`, `CoxLewis`, ...), renewal / imperfect repair (`GeneralizedRenewal`, `GeneralizedOneRenewal`, `ARA`, `ARI`), proportional-intensity regression, cause-specific MCF/NHPP, trend/GoF diagnostics |
| **Degradation & RUL** | `surpyval.degradation` | `DegradationAnalysis` (path models via `PATH_MODELS`: Linear, Exponential, Power, ...), stochastic processes `WienerProcess`/`GammaProcess`, `InducedFailureDistribution` (Lu–Meeker), `ProcessRUL` |
| **Multivariate** | `surpyval.multivariate` | Copulas: `Clayton`, `Frank`, `Gumbel`, `Gaussian`, `Independence` |
| **Mixtures** | `sp.MixtureModel(dist=..., m=...)` | EM mixture of a base family |
| **ML (beta, pre-stable)** | `surpyval.beta.ml` | `SurvivalTree`, `RandomSurvivalForest` (full data model; coupled `kind="weibull"/"exponential"/"non-parametric"`) |
| **System models (alpha)** | `surpyval.alpha` | `SeriesModel`, `ParallelModel` |

> `surpyval.experimental` is a deprecated re-export of `alpha` + `beta.ml` (warns on import). Pre-stable tiers: `alpha` = exploratory, `beta` = complete but interface not yet frozen.

## Regression example

```python
import numpy as np, surpyval as sp
Z = np.random.default_rng(0).normal(0, 1, (40, 1))
x = 10 * np.exp(-0.3 * Z[:, 0]) * np.random.default_rng(0).weibull(2, 40)

model = sp.WeibullPH.fit(x=x, Z=Z, c=np.zeros(40))   # proportional hazards
model.sf(5.0, np.array([0.5]))    # survival at t=5 for covariate vector Z=[0.5]
# Cox when you don't want to assume a baseline shape:
cox = sp.CoxPH.fit(x=x, Z=Z, c=np.zeros(40))
```

Regression predictions take the covariate vector as the second argument:
`model.sf(x, Z)`, `model.ff(x, Z)`, `model.hf(x, Z)`, etc.

## Serialisation (files & MongoDB)

Every fitted model round-trips to a plain dict / JSON, and any model can be
restored **without knowing its class** via the package-level readers:

```python
blob = model.to_dict()          # JSON- and BSON-safe native types, carries "schema"
model.to_json("model.json")

m2 = sp.from_dict(blob)         # dispatches on the dict itself
m3 = sp.from_json("model.json")
```

MongoDB works directly: `collection.insert_one(model.to_dict())` then
`sp.from_dict(collection.find_one(...))` — the `_id` field is ignored, and a
document written by a newer schema than the installed SurPyval is refused with a
clear error rather than misread.

## Datasets & utilities

- `surpyval.datasets` — bundled example data: `load_lung()`, `load_rossi_static()`,
  `load_heart_transplants()`, `load_bofors_steel()`, etc. (return DataFrames).
- `surpyval.utils` — data-format handlers (`xcnt_handler`, `fsli_handler`, converters
  `fs_to_xcnt`, `xcnt_to_xrd`, ...) and `logrank` for two-sample tests.
- `SurpyvalData` — the internal container fitters build from xcnt input; you rarely
  need it directly, but `Model.fit_from_surpyval_data(data)` exists on the fitters.

## Conventions when working in this repo

- **Distributions are singletons**: `sp.Weibull` is an *instance* of `Weibull_`; you
  call `.fit()`/`.from_params()` on it, you don't instantiate it.
- **Lint/type/format** before committing: `flake8 surpyval`, `black --check surpyval`,
  `mypy surpyval` (config in `pyproject.toml`; flake8 ignores E203/W503/E704/E741).
- **Tests**: `pytest surpyval/tests` (the `surpyval/tests/alpha` tier is excluded in CI).
- **Docs** build with the pinned toolchain in `docs/requirements.txt`
  (`python -m sphinx -b html docs docs/_build/html`); the jupyter-execute examples run
  live, so keep them fast and identifiable.
- Full theory + worked examples live in `docs/` (Regression, Recurrent, Degradation,
  Non-Parametric Modelling) and the API in the module docstrings.
