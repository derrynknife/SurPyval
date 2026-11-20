---
name: surpyval
description: Use this skill for any survival, reliability, or time-to-event analysis with the SurPyval Python package (`import surpyval`). Covers fitting parametric distributions (Weibull, LogNormal, Exponential, ...), Bayesian fits with priors, non-parametric estimators (Kaplan-Meier, Nelson-Aalen, Turnbull) and life tables, regression (AFT, proportional hazards, Cox, additive hazards incl. Aalen, Buckley-James, proportional odds, Royston-Parmar, distributional), sampling weights, competing risks, recurrent events (NHPP/HPP, renewal/imperfect-repair, MCF), degradation/RUL, multivariate copulas, and model serialisation. Trigger whenever the user works with censored, truncated, or interval data, mentions `surpyval`/`Weibull.fit`, reliability/failure-time/hazard/survival curves, or the xcnt data model.
---

# SurPyval

SurPyval is a survival-analysis package whose defining strength is that **every
estimator accepts arbitrary combinations of observed, censored, and truncated
data** through one consistent input convention (the "xcnt" data model). Fit a
model, then call `sf`/`ff`/`hf`/`Hf`/`df`/`qf` on it.

This skill describes SurPyval 0.28. Check the installed version with
`surpyval.__version__`; a note marked **0.22** to **0.28** says the release
whose behaviour it describes, which earlier versions do not share.
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
AIC (or `metric="bic"`, `"aic_c"`, `"neg_ll"`). **0.23:** a mixture is a
candidate when named, `include=["Weibull", sp.MixtureModel(sp.Weibull, 2)]`.
Every model with `covariance()` has `standard_errors()`, an array (`se` is
removed in 0.24; the frailty models' dict became an array).

A model built from parameters needs no data: `sp.Weibull.from_params([10, 3])`.
Its `plot()` draws the CDF alone, handy for comparing a spec with a fit.
**0.29:** `model.rescale_time(factor, unit=None)` is the same life on another
clock, `R(factor * t)`: a life in operating hours on the calendar of a unit
that runs a quarter of the time (`0.25`), or in days (`24, unit="days"`),
with the fit's covariance, data and bounds carried over (any parametric
distribution of time, `MixtureModel`; counts, `Beta`, custom refuse).

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
- **`n`** — integer count at each `x` (repeat-observation shorthand: `n=3` is
  three identical units). A decimal `n` is refused.
- **`w`** (**0.26**) — a weight per row: how many population units each unit
  stands for (sampling, propensity or censoring weights), any positive number.
  Use `w`, not `n`, for weights even when they are whole numbers.
- **`t`** — truncation, a two-column `[tl, tr]`; or pass **`tl`**/**`tr`** (left/right
  truncation) separately. Use `np.inf`/`-np.inf` for one-sided truncation.
- **`e`, `s`, `i`** (in `SurpyvalData`) — `e` the event type (failure mode) or
  state entered at `x`, `s` the state the unit was in during the row
  (multi-state data; without it every row starts from one state), `i` the unit
  (`ids` is its other name). With `s`, `c` is read from `e` (blank = still in
  `s`); non-multi-state fits refuse data in more than one state.
- **`a`** (in `SurpyvalData`) — 1 where a row's event ended the unit's history for
  good, 0 where it carried on; a unit (`i`) may have several rows in time order
  (covariate stretches or repeated events); blank `a` on an event is 1 on the
  unit's last row, 0 before. Fits of one row per unit refuse such data.
- **`m`** (in `SurpyvalData`) — readings taken on the units over their lives,
  on their own timeline (`sp.Measurements`: `{i: {x: value}}`, or
  `{name: {i: {x: value}}}` for several), linked by `i`; `data.markers` is the
  same store (one measurement), which the joint models fit.

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
`heuristic` picks the plotting positions (`"Nelson-Aalen"` default); an
MPP fit's bounds are `method="pivotal"` on `cb`/`quantile_cb`/`param_cb`
(exact for complete or type II samples; Weibull, LogNormal, LogLogistic,
Normal, Logistic, Gumbel, GumbelLEV, Exponential). `model.rba()` gives an
MLE fit's small-sample (reduced bias) adjusted Weibull shape or (log)normal
`sigma`, opt-in.
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
(`CoxPH.fit_tvc_from_df` too). The v0.21 names `x=`, `c=`, ... were removed in
v0.23 and raise a `TypeError` naming the new one.

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
  recurrent models (`CauseSpecificMCF`, `CauseSpecificNHPP`, and
  `CauseSpecificProportionalIntensity` when there are covariates too).
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

# instantaneous MTBF (1 / iif) and its 80% lower bound (delta method);
# method="crow" gives Crow's exact bound on the demonstrated MTBF
crow.mtbf(40.0), crow.mtbf_cb(40.0, alpha_ci=0.2, bound="lower")
```

A trend test's `trend` is the *conclusion* at `alpha_ci` (0.05): `"increasing"`
or `"decreasing"` only when `p_value < alpha_ci`, otherwise `"none"`;
`direction` is just the sign of the statistic. Delayed entry is fine: the
standalone tests take each system's start as `tl=`, and a fitted model's
`trend_test()` uses the `tl` it was fitted with. `iif_cb` / `mtbf_cb` bound the
intensity and the instantaneous MTBF (`mtbf_cb` flips the sides for you);
`method="crow"` needs a time-terminated test (all systems watched from 0 to the
same T) or one failure-terminated system, at x = T.

Renewal / imperfect-repair models: `GeneralizedRenewal`,
`GeneralizedOneRenewal` and `ARA` take a **lifetime distribution** as `dist`
(`dist=sp.Weibull`), while `ARI` takes a **baseline intensity model** as
`baseline` (`baseline=CrowAMSAA`, `Duane`, `CoxLewis`; its old name `dist=`
was removed in v0.23); each refuses the other kind with an error naming the
right fitter. Their `params` is the repair parameter (`q` or
`rho`) followed by the distribution's parameters, named by `parameter_names`.

(Sub-namespaces like `surpyval.recurrent`, `surpyval.degradation`,
`surpyval.multivariate`, `surpyval.ml` are **not** auto-imported by
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
- *A signal read repeatedly (vibration, wear, a biomarker) and failures that it
  drives* → a **joint model** (`sp.WeibullJointPH`, `sp.JointPH(sp.Weibull,
  path_model)`), the readings on each `ItemLifetime` as `markers={age: value}`.
  Carrying the last reading forward as a time-varying covariate biases its
  effect toward 0. Each covariate can act on the hazard, the signal's path or
  both (`hazard_covariates=`, `path_covariates=`); failures may be exact, found
  at an inspection (left or interval censored) or truncated. When the signal
  is a stress that uses up life (load, temperature), `sp.WeibullJointAFT` /
  `sp.JointAFT` makes it speed up the item's ageing instead of multiplying
  its hazard: life used at a high reading is not given back.
  `model.predict(item)` gives one asset's survival, hazard and remaining life
  given its readings so far, with Wald bands, its smoothed and projected
  signal, and draws.

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
  `1` outcome is `sf(0)` (or `params[0]`, or **0.23:** `model.p`), not `sf(1)`.
  Bound the probability with `model.param_cb("p")`: exact Clopper-Pearson by
  default (`method="wald"` or `"lr"` as options), and it works with zero
  failures (the success-run bound). **0.23:** `Binomial.fit(x, n_trials=[...])`
  takes one number of trials per row (batches of different sizes).
- Unsure → `fit_best(...)` picks by AIC, then confirm with the probability plot.
- Add `offset=True` for a failure-free threshold (3-parameter / minimum-life),
  `lfp=True` for a cure fraction (a subpopulation that never fails), `zi=True` for
  dead-on-arrival mass at zero. **0.23:** the fitted cure-model proportion that
  ever fails is `model.lfp_p` (also `fixed={"lfp_p": ...}`,
  `from_params(..., lfp_p=...)`, `param_cb("lfp_p")`); **0.24:** the old `p`
  spellings raise.

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
- **Accelerated life** (`sp.AcceleratedLife(sp.Weibull, sp.life_models.Power)`) —
  a life-stress relationship at a few controlled stress levels (at least two);
  the life models are in `sp.life_models` (Arrhenius is `life_models.Exponential`). The life
  parameter (the Weibull's `alpha`) is replaced by the life model: it prints as
  `L(Z)`, and its slot in `params` is a placeholder 1, not a fitted value.

## What lives where

| Task | Import | Fitters |
|---|---|---|
| **Parametric distributions** | `sp.<Name>` | Weibull, Exponential, Gamma, LogNormal, Normal, Gumbel, Logistic, LogLogistic, ExpoWeibull, Rayleigh, Beta, Beta4, Uniform, and discrete (Bernoulli, Poisson, Binomial, Geometric, NegativeBinomial, DiscreteWeibull, ...) |
| **Non-parametric** | `sp.<Name>` | KaplanMeier, NelsonAalen, FlemingHarrington, **Turnbull** (NPMLE for the full data model incl. interval + truncation) |
| **Non-parametric plot** | `sp.plot_nonparametric` | One or more fitted non-parametric models (a list, or `{label: model}` for groups) on one axes: `sf`/`ff`/`Hf`, `cb` or `band`, censor ticks, `quantiles=0.5` median lines and a number-at-risk table (`at_risk_rows="all"`); returns `fig`, `ax`, `counts`, `quantiles`. The table alone, no matplotlib: `sp.at_risk_table(models, times=, rows=, counts=)` |
| **Tests** | `sp.logrank`, `sp.pairwise_logrank`, `sp.interval_logrank` (interval-censored data), `sp.maxcombo_logrank` (non-proportional hazards), `sp.trend_logrank` (ordered groups), `sp.permutation_logrank` (exact / Monte Carlo p-value), `sp.maxstat_logrank` (cut-point of a continuous covariate, search-corrected p-value), `sp.sf_diff`, `model.likelihood_ratio_test()`, `sp.gray_test` (interval-censored rows too, by multiple imputation), `surpyval.recurrent.laplace` / `mil_hdbk_189c` | k-sample (weighted, stratified) log-rank; Gray's test for cumulative incidences; recurrent trend tests |
| **Regression (parametric)** | `sp.<Dist><Kind>` or `sp.AFT/PH/PO/AH(dist)` | AFT, PH (proportional hazards), PO (proportional odds), AH (additive hazards). E.g. `sp.WeibullPH`, `sp.LogNormalAFT`. Fit with `(x, Z, c, n, t)`; predict `sf(x, Z)`; `params` named by `parameter_names`. |
| **Semi-parametric** | `sp.CoxPH` | Cox proportional hazards (Efron ties by default, with the matching Efron baseline); also `CoxPH.fit_tvc` / `fit_tvc_from_df` for time-varying covariates. `CoxPH.fit` also takes left/interval-censored rows (`c=-1`, `c=2`, `[left, right]` pairs), right truncation `tr` and causes `e` (cause-specific hazards): exact/right data (with `tl`) keep the partial likelihood; anything else is the full likelihood with NPMLE baselines (`CoxNPMLEModel`, `estimator == "NPMLE"`; `cif(x, Z, event)`, `param_cb` wald/lr/bootstrap, bootstrap `cb`). `sp.BuckleyJames` (AFT), `sp.QuantileRegression` (censored quantile regression: `coefficients(tau)`, `qf(0.1, Z)` for a B10 by covariates, bootstrap `quantile_cb` / `bootstrap_ci`; `method="peng-huang"` or `"portnoy"`; coefficients on log time, positive = longer life), `sp.AdditiveHazards` (Lin–Ying), `sp.ProportionalOdds` (NPMLE; positive `beta` = longer life, as in `PO`). `sp.IntervalCoxPH` takes interval-, left- and right-censored rows (`x` as `[left, right]` pairs, `c=2`): Cox PH with an NPMLE baseline on the Turnbull intervals (icenReg's `ic_sp`); `param_cb` Wald (profile likelihood), `"lr"`, `"bootstrap"`; no `cb` on curves. |
| **Time-varying covariates** | `sp.StepSchedule`, `sp.CovariatePath` | Evaluate a fitted regression along a covariate path with `sf_tvc` / `Hf_tvc`; `CovariatePath.from_points` / `from_callable` (**0.22**) for ramps and cycles; item records with a `CovariatePath` as `time_varying_covariates` are fitted on it (cut into steps, halved until the fit stops moving: `model.path_steps`; `path.to_steps(t_max, steps)` gives the steps) |
| **Frailty** | `sp.WeibullFrailty`, ..., `sp.CoxFrailty` | Shared gamma frailty PH (`family="lognormal"` for a log-normal frailty); `params` = baseline, coefficients, `theta` |
| **Mixed-effects Cox** | `sp.CoxMixedEffects`, `sp.RandomEffect` | R's `coxme`: Gaussian random effects on the log hazard, nested/crossed groups and correlated random slopes; `fit(x, Z, c, groups=)` or `random_effects=[RandomEffect(g, slopes=[0])]`; `fit_from_df(..., random_effects="(1 + trt \| centre)")`; `variance_components()`, `variance_cb()`, `lr_test()`, `random_effects`; `sf(x, Z, group=, marginal=)` |
| **Competing risks** | `surpyval.univariate.competing_risks` | `CompetingRisks` (nonparametric CIF: Aalen-Johansen for exact times; with left/interval-censored rows the NPMLE, `estimator == "NPMLE"`, masked causes with `masked="include"`, bootstrap `cb(n_boot=, random_state=)`), `ParametricCompetingRisks` (takes interval-censored rows too), `FineGray` (subdistribution regression), `CompetingRisksProportionalHazards`, `ParametricCompetingRisksRegression` (a parametric regression per cause, `model=WeibullPH` or `{cause: fitter}`: `cif(x, Z, event)` past the data, `probability_of_cause(Z, event)`, `cb`) |
| **Recurrent events** | `surpyval.recurrent` | `NonParametricCounting` (MCF), NHPP/HPP intensity fits (`CrowAMSAA`, `Duane`, `CoxLewis`, `HPP`), software reliability growth NHPPs (`GoelOkumoto`, `YamadaDelayedS`, `InflectionS`, `MusaOkumoto`: failure times, or counts per test period as `c=2` rows `[start, end]` with `n`; data with no growth are refused, naming `HPP`/`CrowAMSAA`), the release questions on any fitted NHPP (`remaining_faults(t)` for a finite-fault model, `next_failure_sf(x, age=)` = reliability over a mission, `time_to_intensity(target)`, `time_to_reliability(target, x)`, each with `_cb` bounds; an upper time bound may be `inf`; refusals say why; `release_time(test_cost, fix_cost, field_cost, life=, reliability=, mission=)` -> dict with `time`, `cost`, `band`, and `release_time_cb(..., method="wald"|"bootstrap")`), the Jelinski-Moranda fault-count model (`JelinskiMoranda.fit(times)` or `fit(gaps, gaps=True)`, a `c=1` row for the end of testing; whole-number `N`, `remaining_faults()`, `current_rate()`, likelihood-ratio `param_cb` / `remaining_faults_cb` / `current_rate_cb` whose upper bound on `N` can be `inf`; refuses data with no growth), the Littlewood-Verrall model (`LittlewoodVerrall.fit(times)` or `fit(gaps, gaps=True, psi="linear"|"quadratic")`: `next_failure_sf(x)`, `next_failure_qf(0.5)`, `current_rate_qf(p)`, `param_cb("alpha")` by likelihood ratio; refuses gaps with no extra scatter), `compare_predictions(times, [GoelOkumoto, MusaOkumoto, CrowAMSAA, JelinskiMoranda, ...], start=10)` (one-step-ahead predictions scored: prequential likelihood, u-plot, y-plot; `.summary()`, `.verdict`, `.best`, `.plot("u")`), testing that was not steady (`EffortClock(GoelOkumoto, (period_ends, effort_per_period)).fit(...)` runs any intensity model on tester hours or test cases in place of time; `ChangePoint(GoelOkumoto, tau=t).fit(...)` multiplies its rates by `ratio` (last of `params`) from `t` on, and with no `tau` estimates it from one system's failure times: `model.change_point["tau"]`, `change_point_cb(model)`, `change_point_test(model, n_boot=99, random_state=)` -> dict with `p_value`; steady failures either side of the change are refused, naming `ChangePoint(HPP)`), defects by development phase (`RayleighDefects.fit(**phases_to_data(counts, lengths=None))`; `expected_events(start, end)` / `expected_events_cb` for a later phase, `remaining_faults(t)` for the field), renewal / imperfect repair (`GeneralizedRenewal`, `GeneralizedOneRenewal`, `ARA`, `ARI`), proportional-intensity regression (`ProportionalIntensityHPP` / `NHPP`; `fit_tvc` / `fit_tvc_timeline` for covariates that change over time, then `cif_tvc` / `iif_tvc` / `cb_tvc` along a `StepSchedule`), semi-parametric recurrent regression with robust (cluster-by-item) standard errors (`AndersenGill`, `PrenticeWilliamsPeterson(time_scale="total"|"gap")`, `WeiLinWeissfeld`; `fit(x, Z, i=, c=)` or start-stop `fit_tvc(i, xl, xr, c, Z)`; `summary()`, `cif(x, Z, stratum=)`), cause-specific MCF/NHPP, cause-specific regression (`CauseSpecificProportionalIntensity.fit(x, Z, i=, c=, e=, baseline=)`: each failure mode its own baseline and coefficients; `coefficients()` for the rate ratio of each covariate on each mode, `cif(x, Z, event)`, `total_cif(x, Z)` and their `_cb` bounds), panel counts (events only counted at each unit's own inspections, `n` the count since the last one, 0 allowed: `PanelCountMCF.fit(x, n, i=, how="MLE"|"pseudo")` for the MCF, `PanelCountRegression.fit(x, Z, n, i=, how="pseudo"|"MLE")` for proportional means, which warns "No finite maximum" and sets `converged=False` when a covariate separates the units with events from those without; bootstrap `mcf_cb` / `param_cb`), joint frailty model of recurrences and a dependent terminal event (`JointFrailty.fit(x, Z, i=, c=, terminal=, time_scale="calendar"|"gap", family="gamma"|"lognormal")`: `terminal=1` on the closing `c=1` row where the terminal event (scrapping, death) ended the history; `theta`, `alpha`, `frailty_tests()`, `frailties`, `mcf` / `sf` with `_cb` bounds, `random(size, Z, end=)`), shared frailty for a fleet whose units differ more than Poisson (`CrowAMSAA.fit(x, i, c, frailty="gamma"|"lognormal", cluster={item: cluster})`, also `Duane`, `CoxLewis`, `ProportionalIntensityNHPP`: a `SharedFrailtyModel` with `theta`, `frailty_test()`, `frailties()`, `cif`/`mcf(x, frailty=u)` with `_cb` bounds, `forecast(horizon)` whose intervals carry the frailty), trend/GoF diagnostics |
| **Degradation & RUL** | `surpyval.degradation` | `DegradationAnalysis` (path models via `PATH_MODELS`: linear, exponential, power, ...), stochastic processes `WienerProcess`/`GammaProcess`, `InducedFailureDistribution` (Lu–Meeker), `ProcessRUL` |
| **Joint models** | `sp.WeibullJointPH`, `sp.JointPH`, `sp.WeibullJointAFT`, `sp.JointAFT` | A marker read repeatedly plus a lifetime: a degradation path (any `PathModel`) with random effects, Weibull PH in its current value, its slope or the area under it (`link="value"`, `"slope"`, `"cumulative"`, or a list; `sp.WeibullJointPH.compare_links(data)` fits each link to the same data and ranks them: `.table` (with `reads_as` words and nested `lr_p_value`), `.best` (the simpler of links within 2 AIC), `.verdict` in plain words, `.likelihood_ratio_tests()`, `.to_dict()`; `lag=200.0` makes the hazard at age t follow the path at max(t - 200, 0), fixed not estimated, compare lags by `aic()`; with the cumulative link, `weight=("exponential", 0.5)` makes older exposure count for less (exp(-u/0.5) at u ago) and `weight=("window", 1.0)` counts only the last 1.0 of it, fixed not estimated, compare by `aic()`); or accelerated (`JointAFT`: the path's value or slope speeds up the item's ageing, life used not given back; `"cumulative"` refused; `model.survival` is `"PH"` or `"AFT"`); covariates per part (`hazard_covariates`, `path_covariates`); up to five path parameters varying between items (`varying=`; each item's integral on the full 7-point grid by default, or `integration="pruned"` for a slightly smaller grid: `model.integration`; a four-parameter fit takes minutes; `random_effects="diagonal"` or blocks such as `[("a", "b"), ("c",)]` leaves out the correlations between them, kept as `model.random_effects_structure`); `marker_unit=` (or a `Measurements` with `units=`) and `unit=` name what each effect is per (`model.effect_units`: `cumulative_effect` per `(mm/s)·years`); all censoring and truncation; `sf/ff/cb(x, Z, markers=, given=)` and `predict(...)` (dynamic prediction: `hf`, `remaining_life`, `marker`, `random`) |
| **Landmarking** | `sp.landmark`, `sp.LandmarkCox`, `sp.landmark_kaplan_meier` | Dynamic prediction (the chance of the event by `s + w` for a unit event-free at `s`, given `Z(s)`) without a model of the covariates' paths: `landmark(life, at, horizon)` gives the units at risk at each landmark with covariates as known then (a `SurpyvalData`, stacked over landmarks); `LandmarkCox.fit(life, at=[...], horizon=w)` a Cox model or supermodel (stratified or pooled baseline, effects polynomial in the landmark) with robust standard errors clustered on the unit; `predict(s, Z)`, `cb`, `table`, `beta(s)` |
| **Multivariate** | `surpyval.multivariate` | Bivariate copulas: `Clayton`, `Frank`, `Gumbel`, `Joe`, `AMH`, `Gaussian`, `StudentT`, `Independence` (with rotations). `MultivariateSurpyvalData`: x/c/n/t with D >= 2 columns and optional axis `names`; the model decides what the columns mean and says how many it takes (`n_axes`; copulas 2) |
| **Mixtures** | `sp.MixtureModel.fit(x, dist=..., m=...)` | EM mixture of a base family |
| **Relative survival** | `sp.RateTable`, `sp.RelativeSurvival`, `sp.ExcessHazard` | Survival beyond a population's own hazards, read from a rate table (`RateTable(rates, ages, years=)`, a dict of rates per stratum; `from_df`): net survival by Pohar Perme (default) or Ederer II (`RelativeSurvival.fit(x, c, rate_table=, age=, year=, stratum=, method=)`; `sf`/`ff`/`Hf`, `cb` with `bound_type="log"` or `"normal"`, `table`), and excess hazard regression `lambda_P(t) + lambda_0(t) exp(beta'Z)` with a piecewise-constant baseline (`ExcessHazard.fit(x, Z, c, rate_table=, age=, breaks=)`; `summary`, `param_cb`, `cb`, net `sf`/`hf`/`qf`/`random`, `observed_sf`). Exact and right-censored rows only; net survival can exceed 1 |
| **Model validation** | `surpyval.metrics` | `brier_score`, `integrated_brier_score`, `auc_td`, `score_predictions` (standard errors, model comparisons), `survival_probability`; `e=`/`cause=` score one cause's predicted CIF with competing risks; dynamic predictions made at an age (joint models, `LandmarkCox`, age-only baselines): `dynamic_auc` / `dynamic_brier` / `dynamic_calibration(model_or_dict, data, at=, horizon=, cross_validate=k)` |
| **Trees and forests** | `surpyval.ml` | `SurvivalTree`, `RandomSurvivalForest` (full data model; coupled `kind="weibull"/"exponential"/"non-parametric"`); `SurvivalTree.fit(..., e=causes)` / `RandomSurvivalForest.fit(..., e=causes)` grow competing-risks trees: `cif(x, Z, event)`, `total_cif`; the forest's `oob_log_likelihood(event=)`, `feature_importances(event=)` judge the causes |

> `surpyval.beta.ml` is the old name of `surpyval.ml` (promoted in v0.27):
> it still imports, with a `FutureWarning`, until the first release on or
> after 12 April 2027. (`surpyval.alpha` is gone: the old system models were
> removed in v0.17.0, and the `surpyval.experimental` alias in v0.22.0.)

## Regression example

```python
import numpy as np, surpyval as sp
rng = np.random.default_rng(0)          # one generator: Z and the noise independent
Z = rng.normal(0, 1, (1000, 1))
# Weibull PH with shape 2: hazard ratio exp(0.6) per unit of Z
x = 10 * rng.weibull(2, 1000) * np.exp(-0.6 * Z[:, 0] / 2)

model = sp.WeibullPH.fit(x=x, Z=Z)    # proportional hazards
model.params                          # alpha, beta, then coef_0 (about 0.6)
model.parameter_names                 # ['alpha', 'beta', 'coef_0']
model.sf(5.0, np.array([0.5]))        # survival at t=5 for covariate vector Z=[0.5]
# Cox when you don't want to assume a baseline shape:
cox = sp.CoxPH.fit(x=x, Z=Z)
cox.beta                              # about 0.6
# Check its assumptions: diagnose() is the one API (check_ph was removed in 0.28)
d = cox.diagnose()                    # transform="km" by default, as cox.zph
d.table                               # rows: test, p, verdict; d.advice remedies
d.tables["proportional_hazards"]      # cox.zph table: per covariate + GLOBAL
```

Regression predictions take the covariate vector as the second argument:
`model.sf(x, Z)`, `model.ff(x, Z)`, `model.hf(x, Z)`, etc.

## New in 0.26

```python
import numpy as np, pandas as pd, surpyval as sp
rng = np.random.default_rng(1)
x = sp.Weibull.random(80, 10, 2, random_state=1)
c = (rng.uniform(size=80) < 0.2).astype(int)
Z = rng.normal(size=(80, 1))

# Weights w: weighted estimate, robust (sandwich / jackknife) variance only
w = rng.uniform(0.5, 3, 80)
wm = sp.Weibull.fit(x, c, w=w)        # covariance() is the robust sandwich
wm.effective_n                        # printed too; a fit warns below 20
sp.KaplanMeier.fit(x, c, w=w).cb(5)   # also NelsonAalen, the regressions, CoxPH

# Bayesian fit: priors in sp.prior, credible intervals via cri, not cb
b = sp.Weibull.fit_bayes(x, c, priors={"beta": sp.prior.around(2, between=(1.5, 3))})
b.param_cri("beta"); b.cri(5); b.quantile_cri(0.1)   # on="sf" by default for cri
b.loo(); b.waic()   # compare Bayesian fits: sp.bayes.compare({"a": b, ...}), not aic

# Distributional regression: covariates on every parameter (shape too)
df = pd.DataFrame({"t": x, "c": c, "load": Z[:, 0]})
dm = sp.WeibullDistributional.fit_from_df(df, x_col="t", c_col="c", Z_cols=["load"])
dm.parameters_at(pd.DataFrame({"load": [0.0, 1.0]}))   # alpha, beta at each row

# Mixture cure: some units never fail; covariates on who fails (lfp_p) and when
rc = np.random.default_rng(0); lot = rc.binomial(1, 0.5, 400)
ever = rc.random(400) < np.where(lot == 1, 0.6, 0.3)
tc = np.where(ever, 100 * rc.weibull(2.0, 400), 500.0)   # the rest still running at 500
cm = sp.WeibullMixtureCure.fit(tc, lot, c=(~ever).astype(int))
cm.cure_probability([1]); cm.sf(100, [1], susceptible=True)   # the latency alone
cm.cure_given_survival(100, [1])       # P(never fails | still running at 100)

# Equivalent time scale: one life on a main clock (hours) and a subordinate
# second clock (starts) read at the same moments, T = hours + k * starts.
# Univariate data: x, c, n, tl on the main clock, plus x2 (an array or a
# SecondaryClock); c is the unit's (the second clock has none)
re_ = np.random.default_rng(1); rate = re_.choice([0.02, 0.1, 0.5], 300)  # starts/hour
hrs = 2000 * re_.weibull(2.0, 300) / (1 + 10 * rate)   # one start = 10 h
em = sp.WeibullEquivalence.fit(hrs, rate * hrs, clocks=("hours", "starts"))
em.k; em.param_cb("k"); em.k_lr_test()["p_value"]   # what a start is worth
em.qf_at_usage(0.1, rate=0.1); em.quantile_cb_at_usage(0.1, rate=0.1)   # B10 in hours
life = em.at_usage(0.1)   # the life at 0.1 starts/hour, a plain life model in hours
life.sf(500); life.qf(0.1); life.random(5, random_state=1); life.cb(500)
# SecondaryClock: the second clock's name and its readings at late entry
# (tl= on the fit is the main clock's); N x 2 windows for interval rows (c=2)
starts = sp.SecondaryClock(rate * hrs, tl=0.1 * rate * hrs, name="starts")
sp.WeibullEquivalence.fit(hrs, starts, tl=0.1 * hrs)   # clocks ('x1', 'starts')
# DataFrame: x_col/xl_col/xr_col, c_col, n_col, tl_col + x2_col (or
# x2l_col/x2r_col), tl2_col, x2_name. Not a MultivariateSurpyvalData (refused).

# Zero inflated: some units fail at time 0 (dead on arrival, x = 0, c = 0);
# covariates on who arrives dead (f0) and on the life of the rest
doa = rc.random(400) < np.where(lot == 1, 0.2, 0.05)
tz = np.where(doa, 0.0, 100 * rc.weibull(2.0, 400))
zm = sp.WeibullZeroInflated.fit(tz, lot)   # zm.fit_route: "two-stage" or "joint"
zm.doa_probability([1]); zm.sf(100, [1], susceptible=True)   # life of a unit that worked

# Robust SEs, the global LR test, a forest plot
ph = sp.WeibullPH.fit(x, Z, c)
ph.robust_standard_errors(); ph.summary(robust=True)  # cluster= for groups
ph.likelihood_ratio_test()["p_value"]
sp.plot_coefficients(ph)              # hazard ratios on a log axis

# Group comparisons and life tables
g = np.repeat(["A", "B", "C", "D"], 20)
sp.pairwise_logrank(x, g, c).table    # Holm-adjusted p_adjusted
sp.logrank(x, g, c, weighting="peto-peto", t0=12)
ka = sp.KaplanMeier.fit(x[:40], c[:40]); kb = sp.KaplanMeier.fit(x[40:], c[40:])
sp.sf_diff(ka, kb, 8)["p_value"]      # survival at one time, log(-log) scale
ka.cs(2, 6)                           # conditional survival from an estimate
sp.life_table(x, c, intervals=4)      # actuarial (Cutler-Ederer) table
py = sp.person_years(x, c, factors={"g": g},          # exposure table:
                     timescales={"age": (0, [0, 5, 10, np.inf])}, per=1000)
py.table; py.rate_ratio({"g": "B"}, {"g": "A"})  # rates, exact bounds
# Pooling counts from several sources that differ (two-stage, hierarchical):
pr = sp.pooled_rate([12, 4, 0, 9, 21], [8000, 5200, 1900, 4100, 9800],
                    alpha_ci=0.1)  # kind="probability" for events / trials
pr.average; pr.predictive          # the mean across sources; a new source
pr.sources; pr.heterogeneity; pr.update(2, 1000)  # shrunk; own data
# Relative survival: net survival beyond a population's own rates (a rate
# table by attained age, and calendar year and stratum where given; ages,
# years, times and rates in one unit). The cause of death is not needed.
ages = np.arange(111)
table = sp.RateTable({"F": 4e-5 * np.exp(0.09 * ages),
                      "M": 7e-5 * np.exp(0.09 * ages)}, ages=ages)
rr = np.random.default_rng(2); age = rr.uniform(50, 80, 300)
sex = rr.choice(["F", "M"], 300); trt = rr.binomial(1, 0.5, 300)
yrs = np.minimum(rr.exponential(1 / (0.2 * np.exp(-0.5 * trt))), 8.0)
ns = sp.RelativeSurvival.fit(yrs, yrs == 8.0, rate_table=table, age=age,
                             stratum=sex)   # method="ederer2" for Ederer II
ns.sf(5); ns.cb(5, alpha_ci=0.1); ns.table   # net survival is not cut at 1
eh = sp.ExcessHazard.fit(yrs, trt, c=yrs == 8.0, rate_table=table, age=age,
                         stratum=sex, breaks=2)  # lambda_P + lambda_0 e^(bZ)
eh.summary(); eh.sf(5, [1]); eh.observed_sf(5, [1], age=70, stratum="F")
```

- **Weights:** `SurpyvalData(..., w=)`, `from_dataframe(..., w_col=)` and
  `fit_from_df(..., w_col=)` (not Cox's yet). Scaling `w` changes nothing.
  Refused with weights: `method="lr"`/`"bootstrap"` bounds, `band`,
  `bootstrap_cb`, `how` other than MLE, `aic`/`bic`/`aic_c`, `fit_best`,
  and for Cox strata, exact ties, residuals and `diagnose()`.
  `Turnbull.fit(..., w=)` is the reverse: its `cb` refuses and its bounds
  are `bootstrap_cb(x, random_state=seed)` (the weighted case bootstrap).
  Weighted models save as schema 4 (0.26 and older refuse them; weighted
  data is schema 3). `hess_inv` (dict key `"hess_inv"`) is the inverse
  Hessian; the robust covariance is `covariance()` (key `"covariance"`).
  `CompetingRisksProportionalHazards.fit(..., w=)` (cause-specific Cox and
  Fine-Gray) and `FineGray.fit(..., w=)` take weights too (robust SEs; no
  `aic` or `diagnose()`; Fine-Gray with `w = 1` equals no `w`).
- **Fine-Gray standard errors** are robust by default (Fine and Gray's
  sandwich, R `cmprsk::crr`'s `var`): `covariance()`,
  `standard_errors()`, `p_values`, `summary()`. The model-based ones
  (`crr`'s `invinf`) are `covariance(robust=False)`,
  `standard_errors(robust=False)`, `summary(robust=False)`.
- **Bayesian:** priors `sp.prior.around`, `between`, `uniform`, `flat`,
  `fixed`, `on_quantities(sp.Weibull, B10=..., beta=...)`,
  `from_model(fit, weight=0.5)`, `from_data(x, c, weight=)`,
  `from_params(dist, params, within=0.2)`; check one with
  `sp.prior.check(dist, priors)`. The model's `sf`/`qf`/`mean` are the
  posterior predictive (a new unit's life); `quantile_cri(p, point=True)`
  is the posterior median of the quantile. `update(x, c)` adds data exactly.
  Deterministic grid (1 to 3 free parameters), no seed.
- **Aalen** (`sp.AalenAdditive.fit(x, Z, c)`): effects that change over time;
  read `cumulative_coefficients()` / `plot()` by slope; test with
  `constant_effects_test(random_state=)`.
- **Royston-Parmar regression:** `sp.RoystonParmarPH`, `sp.RoystonParmarPO`,
  `sp.RoystonParmarRegression(df=, scale=, knots=)`; `df=1` hazard scale is
  Weibull PH, `df=1` odds scale is `PO(LogLogistic)`. Hazard-scale signs as
  PH; odds-scale signs as every `PO` (positive = longer life; flexsurv's
  odds coefficients are ours negated). A `ParametricRegressionModel`, so
  every bound works.
- **Distributional:** `covariates={"beta": ["load"]}` or
  `formulas={"alpha": ..., "beta": ...}` per parameter; `parameter_cb`,
  `plot_parameter`; log link for positive parameters. Signs as AFT (since
  0.27): a positive scale/location coefficient = shorter life
  (`alpha(z) = alpha e^{-b'z}`, the AFT coefficient); shape/sigma
  coefficients keep the link's sign; `model.coefficient_signs`. gamlss and
  flexsurv scale coefficients are ours negated.
- **Robust SEs** are all or none (since 0.27): `summary(robust=True)`,
  `cb(..., robust=True)`, `quantile_cb(..., robust=True)`,
  `param_cb(..., robust=True)` and `robust_p_values()` (each with
  `cluster=`) use the sandwich; without `robust=True` all stay
  model-based. LR/bootstrap bounds, restored models and `fit_tvc` refuse
  `robust=True`.
- **PyMC** (`surpyval.pymc`, PyMC not a dependency): inside `with pm.Model():`
  `censored_likelihood(sp.Weibull, x=x, c=c, alpha=..., beta=...)` adds the
  exact censored/truncated likelihood; `prior_from_model`, `prior_from_spec`;
  `to_surpyval(trace, sp.Weibull, alpha="log_alpha[1]",
  transform={"alpha": "exp"}, beta="beta")` gives a `BayesianParametric`.
- **PyTorch** (`surpyval.torch as spt`, torch not a dependency): the exact
  censored/truncated likelihood as a loss for your own network.
  `y = spt.targets(x=x, c=c, n=n, tl=tl)`; `params = spt.constrain(sp.Weibull,
  net(X))` (raw `(N, k)` outputs -> `{"alpha": ..., "beta": ...}` in range);
  `loss = spt.nll(sp.Weibull, y, params)` (`reduction="mean"` per unit,
  `"sum"`, `"none"`), or `spt.NLLLoss(sp.Weibull)`. Families:
  `spt.SUPPORTED`. Use float64 to match `fit`. Mixtures (Deep Survival
  Machines): `params = spt.constrain(sp.Weibull, raw.reshape(N, K, 2))`,
  `spt.nll(sp.Weibull, y, params, mixture=logits)` with `(N, K)` logits.
  Out: `S = spt.sf(sp.Weibull, times, params, mixture=logits)` (`(N, T)`;
  `.detach().numpy()` into `score_predictions`), and
  `spt.to_surpyval(sp.Weibull, params, row=i, mixture=logits)` (a
  `MixtureModel`; without `mixture`, `dist.from_params`).

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
