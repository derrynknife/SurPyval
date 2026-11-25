# Model Atlas — the survival/reliability modelling landscape

> **Status: reference, not a roadmap.** This is an "atlas" of the space of
> models SurPyval covers or *could* cover, not a planned refactor or a
> commitment. Most cells now hold a model; the ones that do not are listed
> under "Gaps" and "Remaining gaps" below. Keep it as a map for placing new
> models consistently and for seeing what the full landscape looks like; do
> not treat a gap as work that is scheduled. Last brought up to date with
> `dev` on 25 Nov 2026 (the unreleased v0.30).

Every model is fully specified by picking one value on each of the orthogonal
axes below.

## The axes

```
1. Outcome dimension       # how many event-time series are modelled jointly
   ├── univariate          # one series per unit; units treated as independent
   │                       #   replicates (the usual case)
   └── multivariate        # several correlated series modelled jointly
                           #   (clustered / paired / parallel units); the
                           #   dependence is specified via frailty or a copula

2. Event recurrence        # how many events within one series (orthogonal to dim)
   ├── single_event        # at most one event per series
   └── recurrent           # repeated events over time within a series

3. Competing events        # branching: how many event types compete out of a state
   ├── single              # one possible event type
   └── competing           # several mutually-exclusive event types

4. State structure         # the shape of the state graph
   ├── terminal            # every event leads to an absorbing state; the
   │                       #   process ends at the first event
   │                       #   (single + terminal = "single risk",
   │                       #    competing + terminal = "competing risks")
   └── multi_state         # transient intermediate states with onward and/or
                           #   reversible transitions (illness-death,
                           #   progressive); each transient state has its own
                           #   single-vs-competing exits (axis 3)

5. Covariates
   ├── without_covariates
   └── with_covariates     # regression

6. Time scale              # nature of the time axis itself
   ├── continuous          # absolutely-continuous failure time
   └── discrete            # discrete/grouped time, or binary outcome
                           #   (revealed by Bernoulli; also success-run,
                           #   period/grouped data)

7. Estimation
   ├── parametric          # fully specified distribution / intensity
   ├── semiparametric      # parametric covariate effect + nonparametric
   │                       #   baseline (Cox PH, Fine–Gray, CRPH)
   └── nonparametric       # no distributional assumption (KM, NA, MCF, CIF)
```

## Dependencies and how existing models classify

- **`recurrent` does *not* imply `multivariate`** — the two axes are
  orthogonal. A single repairable system with repeated failures is *univariate
  recurrent* (one counting process — MCF, NHPP, …); `multivariate` is reserved
  for *several correlated series* modelled jointly. All four combinations of
  univariate/multivariate × single_event/recurrent are valid.
- `semiparametric` only co-occurs with `with_covariates` (it is the
  nonparametric-baseline-plus-covariate-effect combination).
- **Branching (axis 3) and state structure (axis 4) are independent — that is
  the split.** Classic survival is `single` + `terminal`; classic competing
  risks is `competing` + `terminal`. `multi_state` is *not* "more competing
  risks": it is a separate generalization (transient intermediate states), and
  each transient state independently has its own single-vs-competing exits. So
  multistate composes with axis 3 rather than sitting at the top of a single
  ladder.
- A `recurrent` process is the self-returning special case (a transient state
  that loops back to at-risk), so the `terminal` / `multi_state` distinction
  mainly refines `single_event` processes; recurrent rows are marked
  `recurrent` in the state column below.

Worked classifications:

| Model | dim | recurrence | events | states | covariates | time | estimation |
|-------|-----|-----------|--------|--------|------------|------|------------|
| Weibull, Exponential, … | univariate | single_event | single | terminal | none | continuous | parametric |
| KaplanMeier, NelsonAalen, FlemingHarrington, Turnbull | univariate | single_event | single | terminal | none | continuous | nonparametric |
| WeibullPH/AFT/PO/AH (every distribution), AcceleratedLife, RoystonParmar, Distributional (covariates on every parameter) | univariate | single_event | single | terminal | with | continuous | parametric |
| CoxPH, ProportionalOdds, AdditiveHazards, BuckleyJames | univariate | single_event | single | terminal | with | continuous | semiparametric |
| QuantileRegression (censored quantile regression: a coefficient vector per quantile of the log time, Peng-Huang or Portnoy) | univariate | single_event | single | terminal | with | continuous | semiparametric |
| IntervalCoxPH (Cox PH for interval-censored data: NPMLE baseline on the Turnbull intervals, fitted jointly with the coefficients) | univariate | single_event | single | terminal | with | continuous | semiparametric |
| MixtureCure (a fraction that never fails; covariates on that fraction and on the life of the rest) | univariate | single_event | single | terminal | with | continuous | parametric |
| Equivalence (one life on two clocks, `x1 + k x2`, `k` estimated) | univariate | single_event | single | terminal | none | continuous | parametric |
| RelativeSurvival (net survival beyond a rate table's population hazards: Pohar Perme, Ederer II) | univariate | single_event | single (the cause not recorded) | terminal | none | continuous | nonparametric |
| ExcessHazard (population hazard plus `lambda_0(t) exp(beta'Z)`, piecewise-constant baseline) | univariate | single_event | single (the cause not recorded) | terminal | with | continuous | parametric |
| ZeroInflated (a fraction dead on arrival, failing at time 0; covariates on that fraction and on the life of the rest) | univariate | single_event | single | terminal | with | continuous | parametric |
| Survival trees and forests (`surpyval.ml`: log-rank, deviance and conditional-inference splits; parametric or step leaves) | univariate | single_event | single | terminal | with | continuous | nonparametric |
| Survival trees and forests grown with causes (`e=`: cause-specific or Gray splits, a cumulative incidence per leaf) | univariate | single_event | competing | terminal | with | continuous | nonparametric |
| LandmarkCox (a Cox model of the units at risk at each landmark) | univariate | single_event | single | terminal | with (as known at the landmark) | continuous | semiparametric |
| JointPH, JointAFT (a reading's path with random effects, and a Weibull hazard that follows it; several readings, each with its own path and effect; a hazard per failure mode with event types) | univariate | single_event | single/competing | terminal | with | continuous | parametric |
| ParametricCompetingRisks | univariate | single_event | competing | terminal | none | continuous | parametric |
| CompetingRisks (CIF) | univariate | single_event | competing | terminal | none | continuous | nonparametric |
| ParametricCompetingRisksRegression | univariate | single_event | competing | terminal | with | continuous | parametric |
| Fine–Gray, CRPH | univariate | single_event | competing | terminal | with | continuous | semiparametric |
| NonParametricCounting (MCF) | univariate | recurrent | single | recurrent | none | continuous | nonparametric |
| HPP/NHPP (Crow-AMSAA, Duane, Cox-Lewis; software reliability growth: Goel-Okumoto, Yamada delayed S-shaped, inflection S-shaped, Musa-Okumoto; the Rayleigh defect model over development phases; any of them on testing effort in place of time, or with a change point in its rates) | univariate | recurrent | single | recurrent | none | continuous | parametric |
| Jelinski-Moranda (a whole number of faults, each found at one rate; one test campaign) | univariate | recurrent | single | recurrent | none | continuous | parametric |
| Littlewood-Verrall (exponential gaps with gamma rates that fall on average; one test campaign) | univariate | recurrent | single | recurrent | none | continuous | parametric |
| Renewal: GeneralizedRenewal, GeneralizedOneRenewal, ARA, ARI | univariate | recurrent | single | recurrent | none | continuous | parametric |
| ProportionalIntensity HPP/NHPP | univariate | recurrent | single | recurrent | with | continuous | parametric |
| AndersenGill, PrenticeWilliamsPeterson, WeiLinWeissfeld | univariate | recurrent | single | recurrent | with | continuous | semiparametric |
| CauseSpecificMCF | univariate | recurrent | competing | recurrent | none | continuous | nonparametric |
| CauseSpecificNHPP | univariate | recurrent | competing | recurrent | none | continuous | parametric |
| CauseSpecificProportionalIntensity | univariate | recurrent | competing | recurrent | with | continuous | parametric |
| PanelCountMCF (counts at inspections, no event times) | univariate | recurrent | single | recurrent | none | continuous | nonparametric |
| PanelCountRegression (proportional means, panel counts) | univariate | recurrent | single | recurrent | with | continuous | semiparametric |
| Bernoulli, Binomial, success-run | univariate | single_event | single | terminal | none | discrete | parametric |
| Discrete reliability growth: Crow-AMSAA discrete, Lloyd-Lipow, Gompertz, modified Gompertz (successes and trials by stage) | univariate | recurrent | single | recurrent | none | discrete | parametric |
| MultiState (Aalen–Johansen) | univariate | single_event | single/competing | multi_state | none | continuous | nonparametric |
| MultiStateProportionalHazards (Cox per transition) | univariate | single_event | single/competing | multi_state | with | continuous | semiparametric |
| ParametricMultiState (a distribution per transition) | univariate | single_event | single/competing | multi_state | none | continuous | parametric |
| ParametricMultiStateRegression (a parametric regression per transition) | univariate | single_event | single/competing | multi_state | with | continuous | parametric |
| ParametricMultiState / ParametricMultiStateRegression with `clock="reset"` (semi-Markov: hazards of the time in the state) | univariate | single_event | single/competing | multi_state | none/with | continuous | parametric |
| PanelMultiState, PanelMultiStateRegression (a Markov model of states seen only at visits: the moves interval censored) | univariate | single_event | single/competing | multi_state | none/with | continuous | parametric |
| (not built yet) hidden Markov (misclassified states) for panel data; semi-Markov Cox per transition | univariate | single_event | single/competing | multi_state | none/with | continuous | parametric/semiparametric |

Most models SurPyval ships are `univariate`. Four kinds of `multivariate`
model exist:

- **Copulas** (`surpyval.multivariate`: Independence, Gaussian,
  Student-t, Clayton, Frank, Gumbel, Joe and AMH, with rotations) glue
  univariate margins together with a dependence parameter, with full
  censoring/truncation support in the joint likelihood, standard errors and
  confidence bounds; the margins may be parametric or non-parametric
  (rank-based covariance, or the case bootstrap). More than two lifetimes:
  the Gaussian and Student-t copulas take a correlation matrix of three or
  more series, with any number of them censored in a row (up to 15, or 14
  for the Student-t) and with truncation; C- and D-vines of bivariate
  pair-copulas take censoring in up to two series (placed at the vine's
  ends) and no truncation.
- **Shared frailty** for clustered single-event data with covariates
  (`CoxFrailty`, and `WeibullFrailty` and the other parametric frailty
  baselines): a random effect per cluster, the random-effect dual of an
  Archimedean copula; `CoxFrailty` takes a gamma or log-normal frailty.
  `CoxMixedEffects` (R's `coxme`) generalises the log-normal one to several
  Gaussian random-effect terms: nested and crossed grouping factors and
  correlated random slopes.
- **Non-parametric joint survival** (`Dabrowska`): the joint survival of
  two right-censored lifetimes with no dependence family chosen, the
  two-dimensional Kaplan-Meier (Dabrowska, 1988); right censoring only.
- **Joint frailty of recurrent and terminal events** (`JointFrailty`):
  each item's own frailty acts on its recurrent intensity and, raised to
  a power `alpha`, on the hazard of a terminal event that ends its
  history (scrapping, death), so the two series are dependent. The frailty
  is per item, not shared by a group of items.

| Model | dim | recurrence | events | states | covariates | time | estimation |
|-------|-----|-----------|--------|--------|------------|------|------------|
| Bivariate copulas (Gaussian, Student-t, Clayton, Frank, Gumbel, Joe, AMH) | multivariate | single_event | single | terminal | none | continuous | parametric |
| Gaussian and Student-t copulas of three or more series (a correlation matrix; censored and truncated rows) | multivariate, more than two series | single_event | single | terminal | none | continuous | parametric |
| Vine copulas (C- and D-vines of bivariate pair-copulas; censoring in up to two series, at the vine's ends) | multivariate, more than two series | single_event | single | terminal | none | continuous | parametric |
| WeibullFrailty (and other parametric baselines) | multivariate (clustered) | single_event | single | terminal | with | continuous | parametric |
| CoxFrailty | multivariate (clustered) | single_event | single | terminal | with | continuous | semiparametric |
| JointFrailty (recurrent events and a dependent terminal event, a frailty per item, Weibull baselines) | multivariate (two series per item) | recurrent | single | recurrent + terminal | with | continuous | parametric |
| Shared-frailty Poisson process (`frailty=` on CrowAMSAA, Duane, CoxLewis, ProportionalIntensityNHPP; a frailty per item or cluster of items) | multivariate (clustered) | recurrent | single | recurrent | none / with | continuous | parametric |
| CoxMixedEffects (nested, crossed, random slopes) | multivariate (clustered) | single_event | single | terminal | with | continuous | semiparametric |
| Dabrowska (non-parametric joint survival of two right-censored lifetimes, no copula chosen) | multivariate (two series) | single_event | single | terminal | none | continuous | nonparametric |

One shipped capability sits deliberately *outside* the axes:
`surpyval.degradation` (pseudo-failure-time degradation analysis) is not itself
an event-time model but a **data bridge** — it converts repeated degradation
measurements into per-unit (possibly right-censored) failure times by
extrapolating a fitted degradation path to a failure threshold, then hands
those times to an ordinary univariate parametric fitter. A path is one of
the named shapes or a natural cubic spline (`SplinePath`, for a reading
that bends where no formula says it should), and a unit still running
after its last reading is right censored at the time it was last known to
be running (`censored_units=`). The life model it
produces classifies as univariate / single_event / single / terminal /
parametric; the degradation stage itself is least-squares regression on
measurements, not survival modelling. The stochastic degradation *processes*
`WienerProcess` and `GammaProcess` are genuine models: each implies a
first-passage (failure) time distribution, univariate / single_event /
single / terminal / parametric, with standard errors, bounds and AIC.
`DestructiveDegradation` fits destructive degradation tests.
The joint models (`JointPH`, `WeibullJointPH`) model the measurements and the
failure times *together*: each item's marker follows a degradation path with
random effects, and the hazard is Weibull proportional hazards in the path's
current value (or, in `JointAFT` / `WeibullJointAFT`, the path speeds up the
item's ageing: an accelerated failure time part on the Weibull). Their event-time part classifies as univariate / single_event /
single / terminal / with covariates (each in the hazard, the path or both) /
continuous / parametric, with every censoring type and left and right
truncation, and with the failures' event types a hazard for each failure
mode (competing risks: single becomes multiple); one item's dynamic prediction given its readings so far is
`JointModel.predict`, with its quantiles and mean (`qf`, `mean`). The path
may be a spline, with bends common to the fleet or varying between items;
and there may be several readings (markers) per item, each with its own path,
noise and effect, their random effects correlated across the markers
(`JointPH(Weibull, {name: path_model, ...})`). Landmarking (`landmark`, `LandmarkCox`) gives dynamic
predictions without a model of the covariates' paths: a Cox model of the
units at risk at each landmark, by their covariates known then; it
classifies as univariate / single_event / single / terminal / with
covariates (time-varying, as known at the landmark) / continuous /
semiparametric.

## Gaps

Cells of the axes with nothing built yet, or only part:

| Gap | Cell |
|-----|------|
| Recurrent events with covariates, semi-parametric: robust bounds on the cumulative baseline, and simulation (Andersen–Gill, PWP and WLW are built) | univariate / recurrent / single / recurrent / with / continuous / semiparametric |
| Recurrent events with competing failure modes and covariates, semi-parametric (the parametric one, `CauseSpecificProportionalIntensity`, is built) | univariate / recurrent / competing / recurrent / with / continuous / semiparametric |
| Multi-state: hidden Markov (misclassified states) for panel data; a semi-Markov ("clock reset") Cox model per transition (the Markov and parametric semi-Markov models are built) | `multi_state` |
| Copulas for more than two lifetimes beyond the Gaussian, Student-t and C-/D-vines: regular vines, vines with censoring in more than two series or with truncation, exchangeable Archimedean | multivariate, more than two series |
| Shared frailty and correlated series for repairable systems, beyond those built: several recurrent series per item, a frailty nested in a group's, spline baselines | multivariate / recurrent |
| Joint models of several readings per item beyond the several-marker model (built: each marker its own path and effect, correlated random effects): failure modes with several markers, a combined health index | univariate / single_event / with / parametric |
| Bayesian inference beyond the parametric distributions and the PH, AFT and accelerated-life regressions: Cox, recurrent events, degradation, competing risks, multi-state | the inference axis |

Not gaps, by choice: discrete time with covariates is logistic or binomial
regression, out of scope for the package; and discrete time without
covariates has no separate non-parametric estimator (for single trials it is
the observed proportion).

## The inference axis

- **Inference paradigm** — frequentist or Bayesian. Every model is fitted
  by a frequentist estimator (MLE/MPS/MOM/MSE/MPP). The univariate
  parametric distributions are also fitted by Bayes' rule
  (`Distribution.fit_bayes`, `surpyval.bayes`): any censoring and
  truncation, the offset, limited-failure and zero-inflated forms, priors
  on the parameters or on quantities such as a B-life. So are the
  parametric proportional hazards, accelerated failure time and
  accelerated life regressions (`fit_bayes` on the fitter: priors on the
  coefficients, the baseline and a life model's parameters).
- **The engines.** SurPyval has no sampler of its own, by decision. A
  posterior is computed in closed form where a conjugate pair exists, on
  a self-checking grid for one to three free parameters, and by the
  Laplace approximation for any number (a normal at the posterior's peak,
  checked against the exact posterior by its Pareto k and optionally
  reweighted by importance sampling). An exact posterior of a larger
  model goes through PyMC (`surpyval.pymc`): SurPyval's likelihood of a
  distribution or of a PH, AFT or accelerated-life regression as a term of
  a PyMC model, groups sharing one population (hierarchical, partial
  pooling: `pooled_likelihood`), and the draws brought back as a SurPyval
  model. A new Bayesian model is placed by asking which engine can carry
  it; one that needs a sampler belongs behind the PyMC bridge, not in a
  new engine.
- **Networks.** Likewise there are no networks or training loops:
  `surpyval.torch` gives the same censored and truncated likelihood as a
  loss, of one distribution a row or a mixture, to people who train their
  own.

## Remaining gaps (as of 25 Nov 2026)

What an analyst coming from lifelines, R's survival ecosystem or the
commercial reliability tools would look for and not find, after a check of
`dev` on that date. It adds, to the cells in "Gaps" above, the things that
sit on one axis's *values* (which distributions, which estimators, which
measures) rather than in an empty cell. A map of what is absent, in rough
order of how often it is asked for within each group; not a plan.

**Parametric, no covariates (which distributions).**

- The generalised F, inverse Gaussian and Birnbaum-Saunders are not
  built-in classes; each is a `CustomDistribution`, so none is in
  `fit_best`, the regression families or the PyMC and torch lists. (The
  generalised gamma and the Gompertz are built in: `GeneralizedGamma`,
  `Gompertz`.)
- The piecewise exponential likewise: a custom distribution, by choice.

**Recurrent, parametric (reliability growth).**

- The maturity projection models (AMPM, and AMPM-Stein and ADPM-Stein,
  MIL-HDBK-189C section 6.3.4 to 6.3.6). Built: Duane, Crow-AMSAA, Crow
  Extended, the AMSAA-Crow projection, the discrete growth models
  (Crow-AMSAA discrete, Lloyd-Lipow, Gompertz, modified Gompertz) and the
  planning curves (`surpyval.test_planning`: the idealised curve,
  PM2-Continuous and PM2-Discrete).

**Semi-parametric, with covariates.**

- Cox regression for case-cohort and nested case-control samples.
- A rank-based (Gehan) accelerated failure time fit (Buckley-James is
  built); the Cox-Aalen model.
- Penalties beyond the lasso, ridge and elastic net: group, adaptive and
  non-convex (SCAD, MCP) penalties; sure independence screening for more
  covariates than units; stability selection of the chosen covariates.
- Component-wise likelihood boosting of a Cox or AFT model (an estimator
  of the same linear predictor as a penalised fit, reached a covariate at
  a time). Boosting of trees is out of scope.

**Cure and frailty.**

- The promotion-time (non-mixture) cure model; positive-stable and
  inverse-Gaussian frailties (gamma and log-normal are built).

**Trees and forests (`surpyval.ml`).**

- Quantities of a grown forest: minimal depth, partial dependence and
  individual conditional expectation curves of the predicted survival, a
  time-dependent importance (the forest has a permutation importance by
  out-of-bag log-likelihood).
- Variants: extremely randomised (random-cut) trees and oblique splits.
- Bayesian additive regression trees need a sampler: out of scope.

**Measures of a prediction (`surpyval.metrics`, `diagnostics`).**

- D-calibration (are the predicted survival probabilities at the observed
  times uniform); Gönen and Heller's concordance for a Cox model, which
  needs no censoring weights; a measure of explained variation (Royston
  and Sauerbrei's D and R²).
- Conformal prediction: a distribution-free lower bound on one unit's
  survival time from any model's predictions.
- Decision-curve (net benefit) analysis is out of scope, as IDI and NRI
  are.

**Bridges (`surpyval.torch`, `surpyval.pymc`).**

- Losses beyond the parametric likelihood: the Cox partial likelihood and
  a discrete-time hazard likelihood, each for any censoring the package's
  own fits take.
- A bridge to a sampler other than PyMC (`log_likelihood_fn` and
  `from_draws` already work with any).

**Test planning and data.**

- Sensitivity (go/no-go) test designs, such as the up-and-down method.
- Durations from a pair of date columns (dates are refused with a
  message); DataFrames other than pandas.

Out of scope, so not gaps: tuning, pipelines and general cross-validation
tooling; boosted trees; networks and training loops; a sampler; physics of
failure; discrete-time regression (logistic or binomial regression);
stress-strength interference (a calculation on two fitted models, done in
the packages that use SurPyval's models).

## Deferred orthogonal axes (out of scope for now)
- **Effect type** — fixed vs random effects (frailty). Frailty is both the
  *clustered* kind of `multivariate` data and a random-effect modality of the
  covariate axis; the single-event frailty regressions above exist, and
  for repairable systems a frailty per item or cluster of items exists on
  the Poisson-process models (`frailty=`) and in the joint model of
  recurrent and terminal events (`JointFrailty`). (Shared-frailty models coincide with Archimedean copulas — a
  reason the multivariate-dependence machinery could be framed as
  `none / frailty / copula`.)
