# Model Atlas — the survival/reliability modelling landscape

> **Status: reference / aspirational, not a roadmap.** This is an "atlas" of the
> space of models SurPyval *could* eventually cover, not a planned refactor or a
> commitment. It is deliberately ambitious — most cells are unbuilt. Keep it as a
> map for placing new models consistently and for seeing what the full landscape
> looks like; do not treat it as work that is scheduled.

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
| WeibullPH/AFT/PO/AH (every distribution), AcceleratedLife, RoystonParmar | univariate | single_event | single | terminal | with | continuous | parametric |
| CoxPH, ProportionalOdds, AdditiveHazards, BuckleyJames | univariate | single_event | single | terminal | with | continuous | semiparametric |
| Survival trees and forests (`surpyval.beta.ml`) | univariate | single_event | single | terminal | with | continuous | nonparametric |
| ParametricCompetingRisks | univariate | single_event | competing | terminal | none | continuous | parametric |
| CompetingRisks (CIF) | univariate | single_event | competing | terminal | none | continuous | nonparametric |
| Fine–Gray, CRPH | univariate | single_event | competing | terminal | with | continuous | semiparametric |
| NonParametricCounting (MCF) | univariate | recurrent | single | recurrent | none | continuous | nonparametric |
| HPP/NHPP (Crow-AMSAA, Duane, Cox-Lewis) | univariate | recurrent | single | recurrent | none | continuous | parametric |
| Renewal: GeneralizedRenewal, GeneralizedOneRenewal, ARA, ARI | univariate | recurrent | single | recurrent | none | continuous | parametric |
| ProportionalIntensity HPP/NHPP | univariate | recurrent | single | recurrent | with | continuous | parametric |
| CauseSpecificMCF | univariate | recurrent | competing | recurrent | none | continuous | nonparametric |
| Bernoulli, Binomial, success-run | univariate | single_event | single | terminal | none | discrete | parametric |
| (future, #808) illness-death, progressive | univariate | single_event | single/competing | multi_state | none/with | continuous | any |

Most models SurPyval ships are `univariate`. Two kinds of `multivariate`
model exist:

- **Bivariate copulas** (`surpyval.multivariate`: Independence, Gaussian,
  Student-t, Clayton, Frank, Gumbel, Joe and AMH, with rotations) glue two
  univariate margins together with a dependence parameter, with full
  censoring/truncation support in the joint likelihood, standard errors and
  confidence bounds. More than two lifetimes is #809.
- **Shared frailty** for clustered single-event data with covariates
  (`CoxFrailty`, and `WeibullFrailty` and the other parametric frailty
  baselines): a random effect per cluster, the random-effect dual of an
  Archimedean copula. Frailty for repairable systems is #810.

| Model | dim | recurrence | events | states | covariates | time | estimation |
|-------|-----|-----------|--------|--------|------------|------|------------|
| Bivariate copulas (Gaussian, Student-t, Clayton, Frank, Gumbel, Joe, AMH) | multivariate | single_event | single | terminal | none | continuous | parametric |
| WeibullFrailty (and other parametric baselines) | multivariate (clustered) | single_event | single | terminal | with | continuous | parametric |
| CoxFrailty | multivariate (clustered) | single_event | single | terminal | with | continuous | semiparametric |

One shipped capability sits deliberately *outside* the axes:
`surpyval.degradation` (pseudo-failure-time degradation analysis) is not itself
an event-time model but a **data bridge** — it converts repeated degradation
measurements into per-unit (possibly right-censored) failure times by
extrapolating a fitted degradation path to a failure threshold, then hands
those times to an ordinary univariate parametric fitter. The life model it
produces classifies as univariate / single_event / single / terminal /
parametric; the degradation stage itself is least-squares regression on
measurements, not survival modelling. The stochastic degradation *processes*
`WienerProcess` and `GammaProcess` are genuine models: each implies a
first-passage (failure) time distribution, univariate / single_event /
single / terminal / parametric, with standard errors, bounds and AIC.
`DestructiveDegradation` fits destructive degradation tests.

## Gaps

Cells of the axes with nothing built yet, each filed as an issue:

| Gap | Cell | Issue |
|-----|------|-------|
| Parametric competing-risks regression | univariate / single_event / competing / terminal / with / continuous / parametric | #804 |
| Recurrent events with covariates, semi-parametric (Andersen–Gill, PWP, mean model) | univariate / recurrent / single / recurrent / with / continuous / semiparametric | #805 |
| Recurrent events with competing failure modes, parametric | univariate / recurrent / competing / recurrent / none / continuous / parametric | #806 |
| Recurrent events with competing failure modes and covariates | univariate / recurrent / competing / recurrent / with / continuous / parametric and semiparametric | #807 |
| Multi-state models (illness-death, progressive) | `multi_state` | #808 |
| Copulas for more than two lifetimes | multivariate, more than two series | #809 |
| Shared frailty and correlated series for repairable systems | multivariate / recurrent | #810 |
| Bayesian inference (a scope decision first) | the deferred inference axis | #811 |

Not gaps, by choice: discrete time with covariates is logistic or binomial
regression, out of scope for the package; and discrete time without
covariates has no separate non-parametric estimator (for single trials it is
the observed proportion).

## Deferred orthogonal axes (out of scope for now)

- **Inference paradigm** — frequentist vs Bayesian. The entire library is
  frequentist (MLE/MPS/MOM/MSE/MPP); Bayesian survival would be a genuine new
  top-level axis (#811).
- **Effect type** — fixed vs random effects (frailty). Frailty is both the
  *clustered* kind of `multivariate` data and a random-effect modality of the
  covariate axis; the single-event frailty regressions above exist, and
  frailty for repairable systems is #810. (Shared-frailty models coincide with Archimedean copulas — a
  reason the multivariate-dependence machinery could be framed as
  `none / frailty / copula`.)
