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
| KaplanMeier, NelsonAalen | univariate | single_event | single | terminal | none | continuous | nonparametric |
| CoxPH | univariate | single_event | single | terminal | with | continuous | semiparametric |
| CompetingRisks (CIF) | univariate | single_event | competing | terminal | none | continuous | nonparametric |
| Fine–Gray, CRPH | univariate | single_event | competing | terminal | with | continuous | semiparametric |
| NonParametricCounting (MCF) | univariate | recurrent | single | recurrent | none | continuous | nonparametric |
| HPP/NHPP, Crow-AMSAA, Duane | univariate | recurrent | single | recurrent | none | continuous | parametric |
| ProportionalIntensity HPP/NHPP | univariate | recurrent | single | recurrent | with | continuous | parametric |
| CauseSpecificMCF | univariate | recurrent | competing | recurrent | none | continuous | nonparametric |
| Bernoulli, success-run | univariate | single_event | single | terminal | none | discrete | parametric |
| (future) illness-death, progressive | univariate | single_event | single/competing | multi_state | none/with | continuous | any |

Almost every model SurPyval ships today is `univariate`. The first
`multivariate` models now exist: `surpyval.multivariate` provides **bivariate
copulas** (Independence, Clayton, Gumbel, Frank, Gaussian) that glue existing
univariate margins together with a dependence parameter, with full
censoring/truncation support in the joint likelihood. The frailty branch of
`multivariate` dependence (the random-effect dual of Archimedean copulas)
remains future work.

| Model | dim | recurrence | events | states | covariates | time | estimation |
|-------|-----|-----------|--------|--------|------------|------|------------|
| Clayton/Gumbel/Frank/Gaussian copula | multivariate | single_event | single | terminal | none | continuous | parametric |

One shipped capability sits deliberately *outside* the axes:
`surpyval.degradation` (pseudo-failure-time degradation analysis) is not itself
an event-time model but a **data bridge** — it converts repeated degradation
measurements into per-unit (possibly right-censored) failure times by
extrapolating a fitted degradation path to a failure threshold, then hands
those times to an ordinary univariate parametric fitter. The life model it
produces classifies as univariate / single_event / single / terminal /
parametric; the degradation stage itself is least-squares regression on
measurements, not survival modelling. (Stochastic degradation *processes* —
Wiener, gamma — would be genuine models on the atlas and remain future work.)

## Deferred orthogonal axes (out of scope for now)

- **Inference paradigm** — frequentist vs Bayesian. The entire library is
  frequentist (MLE/MPS/MOM/MSE); Bayesian survival would be a genuine new
  top-level axis.
- **Effect type** — fixed vs random effects (frailty). Frailty is both the
  *clustered* kind of `multivariate` data and a random-effect modality of the
  covariate axis. (Shared-frailty models coincide with Archimedean copulas — a
  reason the multivariate-dependence machinery could be framed as
  `none / frailty / copula`.)
