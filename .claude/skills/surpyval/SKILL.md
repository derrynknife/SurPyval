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

## The xicn data model (recurrent events)

Recurrent-event fitters (`surpyval.recurrent`) use a **different, longer-format**
convention — one row per event, keyed by which item it belongs to. Do not use the
single-event `x/c/n/t` shape here; use `x/i/c/n` (handled internally by
`surpyval.utils.handle_xicn`):

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
from surpyval.recurrent import NonParametricCounting, CrowAMSAA
# three systems, each with several failures then a censored end-of-watch
x = [11, 24, 40,  9, 33,  5, 18, 41]
i = [ 1,  1,  1,  2,  2,  3,  3,  3]   # item ids
c = [ 0,  0,  1,  0,  1,  0,  0,  1]
mcf = NonParametricCounting.fit(x=x, i=i, c=c)   # mean cumulative function
crow = CrowAMSAA.fit(x=x, i=i, c=c)              # NHPP intensity / reliability growth
```

(Sub-namespaces like `surpyval.recurrent`, `surpyval.degradation`,
`surpyval.multivariate`, `surpyval.beta.ml` are **not** auto-imported by
`import surpyval` — import them explicitly.)

## Choosing a model — what to reach for and why

**First fork: what kind of process generated the data?** Getting this wrong gives
silently wrong numbers, not errors.

- *One event per item* (a unit fails once) → a distribution, non-parametric
  estimator, or regression. The default case.
- *Several mutually-exclusive causes, and which one fired matters* → **competing
  risks**. Fitting a single-event model to one cause while censoring the others
  overstates that cause's incidence; `CompetingRisks`/`FineGray` keep the
  cumulative incidences summing correctly.
- *Items fail repeatedly and are repaired* → **recurrent events** (MCF / NHPP /
  renewal). A single-event fit discards the repair history and mis-estimates the
  rate of occurrence; renewal/imperfect-repair models also capture how good each
  repair was.
- *No failures yet, but a measurable signal drifting toward a threshold* →
  **degradation / RUL**. Predicts life *before* anything fails.

**Parametric vs non-parametric vs semi-parametric:**

- **Non-parametric** (`KaplanMeier`, `NelsonAalen`, `Turnbull`) — assume nothing
  about shape. Best for *describing* the data, comparing groups (`logrank`), and
  sanity-checking a parametric fit. Cannot extrapolate past the last observation.
  `Turnbull` is the one that handles interval censoring and truncation; KM/NA need
  observed/right-censored (optionally left-truncated) data.
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

**Note:** distributions are singletons — `sp.Weibull` is an *instance*, so you call
`sp.Weibull.fit()` / `sp.Weibull.from_params([10, 3])` directly; you never
instantiate it.
