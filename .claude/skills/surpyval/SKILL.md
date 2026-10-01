---
name: surpyval
description: Use this skill for any survival, reliability, or time-to-event analysis with the SurPyval Python package (`import surpyval`). Covers fitting parametric distributions (Weibull, LogNormal, Exponential, ...), non-parametric estimators (Kaplan-Meier, Nelson-Aalen, Turnbull), regression (AFT, proportional hazards, Cox, additive hazards, Buckley-James, proportional odds), competing risks, recurrent events (NHPP/HPP, renewal/imperfect-repair, MCF), degradation/RUL, multivariate copulas, and model serialisation. Trigger whenever the user works with censored, truncated, or interval data, mentions `surpyval`/`Weibull.fit`, reliability/failure-time/hazard/survival curves, or the xcnt data model.
---

# SurPyval

SurPyval is a survival-analysis package whose defining strength is that **every
estimator accepts arbitrary combinations of observed, censored, and truncated
data** through one consistent input convention (the "xcnt" data model). Fit a
model, then call `sf`/`ff`/`hf`/`Hf`/`df`/`qf` on it.

This skill describes SurPyval 0.22. Check the installed version with
`surpyval.__version__`; the notes marked **0.22** differ from 0.21 and earlier.
Import as `import surpyval` (commonly `import surpyval as sp`). Every code
block below runs as written (`surpyval/tests/test_skill.py` executes them).

## The universal fit pattern

Almost everything is a fitter object with a `.fit(...)` method that returns a
fitted **model** object:

```python
import surpyval as sp

model = sp.Weibull.fit(x=[10, 12, 8, 9, 11, 13])   # returns a Parametric model
model.params        # fitted parameters, e.g. array([alpha, beta])
model.parameter_names   # ['alpha', 'beta'], the order of params
model.sf(10)        # survival / reliability function
model.ff(10)        # CDF / failure function  (== 1 - sf)
model.hf(10); model.Hf(10)   # hazard, cumulative hazard
model.df(10)        # density
model.qf(0.5)       # quantile (median here); nan for p outside [0, 1]
model.mean(); model.var(); model.moment(1)
model.cb(10, alpha_ci=0.05)               # Wald bounds on sf at x=10
model.cb(10, alpha_ci=0.05, method="lr")  # likelihood-ratio: better for small n
model.aic(); model.bic(); model.neg_ll()  # fit diagnostics
model.plot()        # probability plot with the empirical overlay
```

The interval level is always `alpha_ci` (0.05 gives a 95% interval), seeds are
always `random_state`, bootstrap sizes `n_boot`, times `x` and probabilities
`p`. **0.22:** the old 0.21 spellings (`alpha=`, `confidence=`, `seed=`, `B=`,
`t=`, Cox `method=`, `cause=`, `id_col=`, `time_col=`) are removed and raise
`TypeError`.

`sp.fit_best(x, ...)` fits the continuous candidates and returns the best by
AIC (or `metric="bic"`, `"aic_c"`, `"neg_ll"`).

A model built from parameters needs no data: `sp.Weibull.from_params([10, 3])`.
Its `plot()` draws the CDF alone, handy for comparing a spec with a fit.

## The xcnt data model (the thing to get right)

Every fitter's `fit()` takes the same optional arrays. Only `x` is usually needed.

- **`x`** — the observed values. For interval-censored points, an entry may be a
  `[left, right]` pair, OR pass interval bounds separately as `xl`/`xr`. A
  single column, e.g. `df[["t"]].to_numpy()`, is one value per row.
- **`c`** — censoring flag per observation, in `{-1, 0, 1, 2}`:
  - `0` = **observed** (exact event) — the default when `c` is omitted
  - `1` = **right censored** (event after `x`)
  - `-1` = **left censored** (event before `x`)
  - `2` = **interval censored** (`x` entry is `[left, right]`)
- **`n`** — integer count/weight at each `x` (repeat-observation shorthand).
- **`t`** — truncation, a two-column `[tl, tr]`; or pass **`tl`**/**`tr`** (left/right
  truncation) separately. Use `np.inf`/`-np.inf` for one-sided truncation.

```python
import numpy as np
import surpyval as sp

# observed + right + left censoring, with counts and left truncation
sp.Weibull.fit(x=[1, 2, 3, 4, 5],
               c=[0, 0, 1, -1, 0],
               n=[1, 2, 1, 1, 3],
               tl=[0, 0, 0, 0.5, 0])

# interval censored via xl/xr
sp.Weibull.fit(xl=[1, 2, 3], xr=[2, 4, 5])
```

**Mind the direction of `c`:** 1 means *censored*, the opposite of an "event"
or "status" column in R or lifelines. Pass `c = 1 - event`. The bundled data
keep that coding (**0.22:** Rossi's `arrest` is 1 for an arrest and lung's
`status` 1 for a death), so it is `c = 1 - df["arrest"]` and
`c = 1 - df["status"]`.

Kaplan-Meier, Nelson-Aalen and Fleming-Harrington take observed and
right-censored data, optionally left truncated; left or interval censoring or
right truncation raise an error that points to `sp.Turnbull`.

Extra `fit` options on parametric distributions: `how` (`"MLE"` default, or
`"MPS"`/`"MPP"`/`"MSE"`/`"MOM"`, in any case), `offset=True` (fit a
location/threshold `gamma`, e.g. 3-parameter Weibull), `zi=True`
(zero-inflation), `lfp=True` (limited failure population / cure fraction),
`fixed={"beta": 2}` (hold parameters), `init=[...]`. For `how="MPP"`,
`heuristic` picks the plotting positions (`"Nelson-Aalen"` default).
**Note:** `offset`/`zi`/`lfp` are univariate-distribution features only — the
**regression** fitters do not accept them (they raise `TypeError`).

**When the data cannot identify the model.** A univariate MLE refuses data
whose likelihood has no finite maximum (e.g. every failure at one time) with a
`ValueError`. Regression, frailty, Fine-Gray, copula, mixture and degradation
fits instead warn "No finite maximum: ..." and return what they reached; treat
the reported values as meaningless and follow the advice in the warning (the
Beta4 recommends `how="MPS"`). **0.22:** the Uniform's MLE refuses censored
data; use `how="MPS"`, `"MPP"` or `"MSE"` for it.

### DataFrame entry
Most fitters also offer `fit_from_df(df, ...)`. Every DataFrame entry point
names a column argument after the `fit` argument it fills, with a `_col`
suffix (`_cols` for a list): `sp.Weibull.fit_from_df(df, x_col="t",
c_col="c")`, and `x_col`, `c_col`, `n_col`, `xl_col`, `xr_col`, `tl_col`,
`tr_col`, `i_col`, `e_col` elsewhere. The regression fitters take `x_col`,
`c_col` and `Z_cols` or a `formula=` that codes categorical columns
(`CoxPH.fit_tvc_from_df` too). The v0.21 names `x=`, `c=`, ... still work
until v0.23, with a `DeprecationWarning`.

## The xicn data model (recurrent events)

Recurrent-event fitters (`surpyval.recurrent`) use a **different, longer-format**
convention — one row per event, keyed by which item it belongs to. Do not use the
single-event `x/c/n/t` shape here; use `x/i/c/n` (handled internally by
`sp.handle_xicn`):

- **`x`** — the time of each event (or of a censoring).
- **`i`** — the **item id**: which unit/system this row belongs to (the column that
  ties repeated events on the same unit together). This is what makes it recurrent.
- **`c`** — censoring, per row (`0` event, `1` right-censored end-of-observation).
- **`n`** — count at each row (default 1).
- **`e`** — optional **event mark / cause label** per event, for cause-specific
  recurrent models (`CauseSpecificMCF`, `CauseSpecificNHPP`).
- **`tl`/`tr`** — truncation; **`windows`** — gapped/intermittent observation windows
  (periods when a unit was actually being watched).

```python
from surpyval.recurrent import NonParametricCounting, CrowAMSAA, laplace
# three systems, each with several failures then a censored end-of-watch
x = [11, 24, 40,  9, 33,  5, 18, 41]
i = [ 1,  1,  1,  2,  2,  3,  3,  3]   # item ids
c = [ 0,  0,  1,  0,  1,  0,  0,  1]
mcf = NonParametricCounting.fit(x=x, i=i, c=c)   # mean cumulative function
crow = CrowAMSAA.fit(x=x, i=i, c=c)              # NHPP intensity / reliability growth

# is the rate changing at all? (null: a homogeneous Poisson process)
result = laplace(x, i, c=c)      # c by keyword; the 3rd positional is T
result.p_value, result.direction, result.trend
crow.trend_test()                # the same test on a fitted model
```

A trend test's `trend` is the *conclusion* at `alpha_ci` (0.05): `"increasing"`
or `"decreasing"` only when `p_value < alpha_ci`, otherwise `"none"`;
`direction` is just the sign of the statistic.

Renewal / imperfect-repair models: `GeneralizedRenewal`,
`GeneralizedOneRenewal` and `ARA` take a **lifetime distribution** as `dist`
(`dist=sp.Weibull`), while `ARI` takes a **baseline intensity model**
(`dist=CrowAMSAA`, `Duane`, `CoxLewis`); each refuses the other kind with an
error naming the right fitter. Their `params` is the repair parameter (`q` or
`rho`) followed by the distribution's parameters, named by `parameter_names`.

(Sub-namespaces like `surpyval.recurrent`, `surpyval.degradation`,
`surpyval.multivariate`, `surpyval.beta.ml` are **not** auto-imported by
`import surpyval` — import them explicitly. `sp.CrowAMSAA` raises an
`AttributeError` that names the subpackage.)

## Choosing a model — what to reach for and why

**First fork: what kind of process generated the data?** Getting this wrong gives
silently wrong numbers, not errors.

- *One event per item* (a unit fails once) → a distribution, non-parametric
  estimator, or regression. The default case.
- *Several mutually-exclusive causes, and which one fired matters* → **competing
  risks**. Fitting a single-event model to one cause while censoring the others
  overstates that cause's incidence; `CompetingRisks`/`FineGray` keep the
  cumulative incidences summing correctly (`CompetingRisks.plot()` stacks them).
- *Items fail repeatedly and are repaired* → **recurrent events** (MCF / NHPP /
  renewal). A single-event fit discards the repair history and mis-estimates the
  rate of occurrence; renewal/imperfect-repair models also capture how good each
  repair was.
- *No failures yet, but a measurable signal drifting toward a threshold* →
  **degradation / RUL**. Predicts life *before* anything fails.

**Parametric vs non-parametric vs semi-parametric:**

- **Non-parametric** (`KaplanMeier`, `NelsonAalen`, `Turnbull`) — assume nothing
  about shape. Best for *describing* the data, comparing groups (`sp.logrank`),
  and sanity-checking a parametric fit. Cannot extrapolate past the last
  observation. `Turnbull` is the one that handles interval censoring and
  truncation; KM/NA need observed/right-censored (optionally left-truncated)
  data.
- **Parametric** (`Weibull`, `LogNormal`, ...) — a smooth curve you can
  *extrapolate* (B10 life, warranty tail, 1% quantile) and that summarises behaviour
  in a few parameters. Costs a shape assumption — always check with `.plot()` or
  `fit_best`.
- **Semi-parametric** (`CoxPH`, `BuckleyJames`) — covariate effects without
  committing to a baseline shape; the default when the question is "which factors
  matter and by how much", not "what's the absolute curve".

**Which distribution** (reason from the hazard shape):

- **Weibull** — the workhorse; the shape `β` reads directly: `β<1` infant mortality
  (decreasing hazard), `β=1` random/constant (= Exponential), `β>1` wear-out
  (increasing). Try it first.
- **Exponential** — memoryless, constant hazard; only when failures are genuinely
  random (no ageing).
- **LogNormal** — hazard rises then falls; fatigue, crack growth, repair-time data.
- **Gamma / ExpoWeibull / LogLogistic** — more flexible hazards when Weibull/LogNormal
  don't fit; `ExpoWeibull` can produce bathtub curves.
- **Normal / Gumbel / Logistic** — location-scale families for data on the whole real
  line (often after a log transform).
- **Bernoulli / Binomial** — pass/fail and success-count data. **0.22:** every
  discrete `sf(x)` is `P(X > x)`, Bernoulli's included, so the probability of the
  `1` outcome is `sf(0)` (or the fitted `p`), not `sf(1)`.
- Unsure → `fit_best(...)` picks by AIC, then confirm with the probability plot.
- Add `offset=True` for a failure-free threshold (3-parameter / minimum-life),
  `lfp=True` for a cure fraction (a subpopulation that never fails), `zi=True` for
  dead-on-arrival mass at zero.

**Which regression form:**

- **AFT** — covariates scale *time* ("this stress halves the life"); the natural,
  interpretable choice for **accelerated life testing**.
- **PH** — covariates scale the *hazard*; standard in biostatistics, read as hazard
  ratios.
- **Cox** — PH effects with an *unspecified* baseline; use when you care about the
  coefficients, not the absolute survival shape (and for time-varying covariates via
  `fit_tvc`).
- **Additive hazards** (Lin–Ying) — covariates *add* to the hazard rather than
  multiply; better on an absolute-risk scale.
- **Proportional odds** — effects that fade over time (converging hazards).
- **Accelerated life** (`sp.AcceleratedLife(sp.Weibull, sp.Power)`) — a life-stress
  relationship at a few controlled stress levels (at least two). The life
  parameter (the Weibull's `alpha`) is replaced by the life model: it prints as
  `L(Z)`, and its slot in `params` is a placeholder 1, not a fitted value.

## What lives where

| Task | Import | Fitters |
|---|---|---|
| **Parametric distributions** | `sp.<Name>` | Weibull, Exponential, Gamma, LogNormal, Normal, Gumbel, Logistic, LogLogistic, ExpoWeibull, Rayleigh, Beta, Beta4, Uniform, and discrete (Bernoulli, Poisson, Binomial, Geometric, NegativeBinomial, DiscreteWeibull, ...) |
| **Non-parametric** | `sp.<Name>` | KaplanMeier, NelsonAalen, FlemingHarrington, **Turnbull** (NPMLE for the full data model incl. interval + truncation) |
| **Tests** | `sp.logrank`, `sp.gray_test`, `surpyval.recurrent.laplace` / `mil_hdbk_189c` | k-sample (weighted, stratified) log-rank; Gray's test for cumulative incidences; recurrent trend tests |
| **Regression (parametric)** | `sp.<Dist><Kind>` or `sp.AFT/PH/PO/AH(dist)` | AFT, PH (proportional hazards), PO (proportional odds), AH (additive hazards). E.g. `sp.WeibullPH`, `sp.LogNormalAFT`. Fit with `(x, Z, c, n, t)`; predict `sf(x, Z)`; `params` named by `parameter_names`. |
| **Semi-parametric** | `sp.CoxPH` | Cox proportional hazards (Efron ties by default, with the matching Efron baseline); also `CoxPH.fit_tvc` / `fit_tvc_from_df` for time-varying covariates. `sp.BuckleyJames` (AFT), `sp.AdditiveHazards` (Lin–Ying). |
| **Time-varying covariates** | `sp.StepSchedule`, `sp.CovariatePath` | Evaluate a fitted regression along a covariate path with `sf_tvc` / `Hf_tvc`; `CovariatePath.from_points` / `from_callable` (**0.22**) for ramps and cycles |
| **Frailty** | `sp.WeibullFrailty`, ... | Shared gamma frailty PH; `params` = baseline, coefficients, `theta` |
| **Competing risks** | `surpyval.univariate.competing_risks` | `CompetingRisks` (nonparametric CIF), `ParametricCompetingRisks`, `FineGray` (subdistribution regression), `CompetingRisksProportionalHazards` |
| **Recurrent events** | `surpyval.recurrent` | `NonParametricCounting` (MCF), NHPP/HPP intensity fits (`CrowAMSAA`, `Duane`, `CoxLewis`, `HPP`), renewal / imperfect repair (`GeneralizedRenewal`, `GeneralizedOneRenewal`, `ARA`, `ARI`), proportional-intensity regression, cause-specific MCF/NHPP, trend/GoF diagnostics |
| **Degradation & RUL** | `surpyval.degradation` | `DegradationAnalysis` (path models via `PATH_MODELS`: linear, exponential, power, ...), stochastic processes `WienerProcess`/`GammaProcess`, `InducedFailureDistribution` (Lu–Meeker), `ProcessRUL` |
| **Multivariate** | `surpyval.multivariate` | Copulas: `Clayton`, `Frank`, `Gumbel`, `Gaussian`, `Independence` |
| **Mixtures** | `sp.MixtureModel.fit(x, dist=..., m=...)` | EM mixture of a base family |
| **Model validation** | `surpyval.metrics` | `brier_score`, `integrated_brier_score`, `auc_td`, `survival_probability` |
| **ML (beta, pre-stable)** | `surpyval.beta.ml` | `SurvivalTree`, `RandomSurvivalForest` (full data model; coupled `kind="weibull"/"exponential"/"non-parametric"`) |

> Pre-stable tier: `surpyval.beta` = complete but interface not yet frozen.
> (`surpyval.alpha` is currently empty: the old system models were removed in
> v0.17.0, and the `surpyval.experimental` alias in v0.22.0.)

## Regression example

```python
import numpy as np, surpyval as sp
rng = np.random.default_rng(0)          # one generator: Z and the noise independent
Z = rng.normal(0, 1, (1000, 1))
# Weibull PH with shape 2: hazard ratio exp(0.6) per unit of Z
x = 10 * rng.weibull(2, 1000) * np.exp(-0.6 * Z[:, 0] / 2)

model = sp.WeibullPH.fit(x=x, Z=Z)    # proportional hazards
model.params                          # alpha, beta, then beta_0 (about 0.6)
model.parameter_names                 # ['alpha', 'beta', 'beta_0']
model.sf(5.0, np.array([0.5]))        # survival at t=5 for covariate vector Z=[0.5]
# Cox when you don't want to assume a baseline shape:
cox = sp.CoxPH.fit(x=x, Z=Z)
cox.beta                              # about 0.6
```

Regression predictions take the covariate vector as the second argument:
`model.sf(x, Z)`, `model.ff(x, Z)`, `model.hf(x, Z)`, etc.

## Serialisation (files & MongoDB)

Every fitted model round-trips to a plain dict / JSON, and any model can be
restored **without knowing its class** via the package-level readers:

```python
import os, tempfile
import surpyval as sp

model = sp.Weibull.fit([10, 12, 8, 9, 11, 13])
blob = model.to_dict()          # JSON- and BSON-safe native types, carries "schema"
path = os.path.join(tempfile.mkdtemp(), "model.json")
model.to_json(path)             # write a file ...
text = model.to_json()          # ... or, with no path, get the JSON text

m2 = sp.from_dict(blob)         # dispatches on the dict itself
m3 = sp.from_json(path)
m4 = sp.from_json(text)
```

MongoDB works directly: `collection.insert_one(model.to_dict())` then
`sp.from_dict(collection.find_one(...))` — the `_id` field is ignored, and a
document written by a newer schema than the installed SurPyval is refused with a
clear error rather than misread. A restored model keeps its parameters but not
its data, so data-dependent methods (bootstrap bounds, residuals) raise.

## Datasets & utilities

- `surpyval.datasets` — bundled example data: `load_lung()`, `load_rossi_static()`,
  `load_heart_transplants()`, `load_bofors_steel()`, etc. (return DataFrames; each
  docstring says how its censoring column is coded).
- `surpyval.utils` — data-format handlers (`xcnt_handler`, `fsli_handler`,
  converters `fs_to_xcnt`, `xcnt_to_xrd`, ...). The log-rank test is `sp.logrank`.
- `SurpyvalData` — the internal container fitters build from xcnt input; you rarely
  need it directly, but `Model.fit_from_surpyval_data(data)` exists on the fitters.

**Note:** distributions are singletons — `sp.Weibull` is an *instance*, so you call
`sp.Weibull.fit()` / `sp.Weibull.from_params([10, 3])` directly; you never
instantiate it. `sp.MixtureModel` is a class: `sp.MixtureModel.fit(x,
dist=sp.Weibull, m=2)` returns the fitted model (or build
`sp.MixtureModel(dist, m)` and call its `fit`, which also returns it).
