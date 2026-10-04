# SurPyval roadmap

The bigger ideas for SurPyval: features that need design, research or
several weeks of work. They live here, not in the issue tracker, so the
issues stay a list of things someone could start this week.

## How this file works

- **An idea starts here**, as a section with the problem, why it matters,
  its rough size, what it depends on, and a status:
  - *idea*: worth doing, not designed;
  - *designed*: the approach is settled (the design is linked or summarised);
  - *scheduled*: broken into issues that are being worked on.
- **When an idea is scheduled**, its design is split into a few actionable
  issues, linked from its section here.
- **When it is done**, its section moves to "Done" with a line and the
  release; the changelog is the full record.
- **Ideas we have decided against** go under "Not planned", with the reason,
  so they are not raised again.
- The issue numbers below are the issues an idea used to live in; they are
  closed and link back here.

---

## Distributional regression (GAMLSS over the xcnt data model)

*Status: designed. Size: large (several weeks). Was #218.*

Let every parameter of a distribution be its own function of covariates,
not just the scale or life parameter. For example, temperature drives a
Weibull's `alpha` through Arrhenius and its limited-failure fraction `p`
through a logit, each with its own link and covariates, fitted jointly
under any censoring and truncation. Ordinary fits, AFT, accelerated life,
cure regression, varying shape, zero-inflation regression and covariate
thresholds all become configurations of one framework. No package pairs
full distributional regression with interval censoring and truncation the
way SurPyval's data model allows.

**Starting point.** `ParameterSubstitutionFitter` already does this for one
parameter (the life parameter), with the full censored and truncated
likelihood differentiated by autograd. The work generalises it to a map
from every parameter to a spec.

**Design.**

- **Spec per parameter.** Each parameter takes one of:
  - a named life model or link (`arrhenius`, `log`, `logit`, ...);
  - a `formulaic` formula;
  - a raw `autograd.numpy` callable with a declared number of coefficients.
- **Coefficients.** An assembler flattens the coefficients of every
  parameter into one optimisation vector.
- **Likelihood.** Every parameter becomes a per-observation vector in the
  likelihood core.
- **Inference.** Standard errors come from the Hessian, prediction bounds
  from the delta method.
- **Serialisation.** A raw-callable parameter cannot be restored from a
  saved model, and is flagged as such.

**Risks.** Identifiability under heavy censoring, initialisation (fit the
constant model first and warm-start from it), per-parameter range
constraints, and an ill-conditioned Hessian.

**Phases.**

1. Generalise to a parameter map. It must reproduce `WeibullAFT` and
   `AcceleratedLife` exactly.
2. The full set of links, plus `p`, `f0` and `gamma` as targets.
3. Inference and serialisation.
4. Hardening on real data.

## Multi-state models

*Status: idea (design first). Size: large. Was #217.*

Competing risks covers the one-step "alive to cause k" case. General
multi-state models (illness-death, with or without recovery, or any
directed graph of states) are absent. They need:

- per-transition Nelson-Aalen hazards;
- the Aalen-Johansen estimator of the transition-probability matrix
  `P(s, t)` by product integration;
- state-occupation probabilities and expected time in each state;
- long-format input (start, stop, from-state, to-state), generalising the
  Cox start-stop handler.

It builds on the counting-process machinery that competing risks and
recurrent events already use. The API and input format should be designed
before anything is built.

## Copulas beyond two variables

*Status: partly designed. Size: high overall. Was #527.*

`Copula.fit` handles two variables only. In order of difficulty:

1. **Gaussian and Student-t in D dimensions**, for fully observed and
   right-censored data. Moderate work.
2. **Vine copulas** built from the bivariate families. High: structure
   selection, fitting and sampling.
3. **Nested Archimedean copulas**, if there is demand. High.

The hard part is censoring and truncation in D dimensions:

- *Elliptical copulas* need multivariate normal or t CDFs (numerical
  integration).
- *Archimedean copulas* need generator derivatives of order D.

More bivariate families (Student-t, Joe, AMH) are #157, an ordinary issue,
and step 2 builds on them.

## Continuous time-varying covariates: fitting (phase 3)

*Status: designed. Size: medium to large. Was #172.*

Phases 1 and 2 (both in 0.22) evaluate a fitted model along a known,
external covariate path, with bounds (`cb_tvc`), the mean (`mean_tvc`) and
accelerated life models. Phase 3 fits on continuous per-subject histories,
such as ramp-stress accelerated life tests where each unit had a known
ramp.

1. **Cox.** Split each subject's path at the event times and call
   `CoxPH.fit_tvc`. Exact and small.
2. **Parametric PH, AH and PO.** The path integral goes inside the
   likelihood, on a mesh frozen per subject. An interim option is to
   discretise each path into start-stop rows, refit at twice the
   resolution, and warn if the estimates move.
3. **AFT and accelerated life.** `aft_tvc_fit`, with the time
   transformation computed by the path engine.

The design is in the #172 proposal.

## Analytic two-stage bounds for accelerated degradation

*Status: idea (research). Size: medium. Was #239.*

For a plain degradation model, two-stage confidence bounds have an analytic
generated-regressor correction (`cb(method="analytic")`,
`life_parameter_covariance("analytic")`): it widens the life model's
covariance by each unit's pseudo-failure-time variance. For an accelerated
(covariate) degradation model the correction is not derived, and both
raise `NotImplementedError` pointing to `method="bootstrap"`.

The derivation has to carry each unit's pseudo-failure-time variance
through the regression life fit: the Jacobian of the fitted coefficients
with respect to the generated pseudo-failure times. The bootstrap already
gives valid bounds (it resamples units and reruns the whole pipeline,
`path="best"` reselection included), so this would add speed and a closed
form only, not coverage. Done would mean the two `NotImplementedError`
branches removed, with a test that the analytic bounds agree with the
bootstrap on a fixed seed.

## Copula uncertainty with non-parametric margins

*Status: idea (research). Size: medium. Was #623.*

Copula fits have standard errors and bounds since 0.23 (#540): the joint
Hessian for a full MLE fit, the Godambe sandwich for a two-stage (IFM) fit.
A fit with a non-parametric margin is a pseudo-likelihood fit (Genest,
Ghoudi and Rivest 1995), and the copula parameter's variance has to carry
the rank-based margins' uncertainty; the naive copula-only variance is too
narrow, so ``covariance()``, ``param_cb`` and ``cb`` refuse such fits.

Two ways forward: the rank-based asymptotic variance of Genest, Ghoudi and
Rivest (1995), or a bootstrap option for these fits. Either needs a
coverage study like ``calibration/test_coverage_copula.py`` for the
parametric margins.

## Faster and more precise likelihood-ratio bounds on edge valleys

*Status: idea (research). Size: medium. Was #609.*

Since #601, likelihood-ratio bounds reach the region's extreme even where
it lies along a long flat valley (an ExpoWeibull's beta -> inf with alpha
pinned at the largest observation, or alpha -> 0; a NegativeBinomial's
r -> inf). Following the valleys costs time: all the registry
likelihood-ratio bounds take about 92 s for ExpoWeibull and 80 s for
NegativeBinomial (one thread), against about 1 s for Weibull, after the
0.23 savings (#602, #609: skipping re-checks, probing a levelled-off valley
once). Precision is about 1e-6 relative in the beta -> inf valley.

What is left needs a different handling of the valley rather than more
tuning:

- **Reach the valley's limit directly.** ExpoWeibull's extreme is only
  approached, never reached, which is why cheaper probing (only the deepest
  point, or skipping candidates a checked answer already beats) cost up to
  6e-6 in its quantile bounds. Extrapolating to the limiting model, or
  parameterising the valley, would allow the cheaper probing
  (NegativeBinomial would go from about 80 s to 46 s).
- **Precision.** The 1e-6 limit comes from the extremality check's step and
  from the limit being approached; exact gradients of the bounded function
  alone do not remove it.
- **Brute-force checks** not yet done: the Uniform and NegativeBinomial
  bands, and ExpoWeibull's 99% ``hf`` and mean bounds.

## Competing-risks survival trees and forests

*Status: idea. Size: large (research). Was #194.*

Survival trees that take cause marks `e=`. They would:

- split on a cause-aware criterion: a cause-specific deviance, which works
  with any censoring and truncation, or a Gray's-test statistic for
  right-censored data;
- fit per-cause leaf models (`ParametricCompetingRisks` or
  `CompetingRisks`);
- expose `cif` and `total_cif` from the ensemble.

Out-of-bag evaluation with a cause-aware metric should come first, so the
forest can be validated.

## Replace autograd with JAX

*Status: idea (long-term). Size: several weeks. Was #159.*

`autograd` is in low-activity maintenance and has no GPU support. JAX is
its successor and close to a drop-in replacement for `autograd.numpy`
code. The interim compatibility work is done (inlined gamma gradients,
autograd 1.8 for numpy 2), so this is not urgent. It touches every gradient
in the package; revisit once the library is otherwise stable.

---

## Ongoing practice: hardening

Hardening is continuous, not a one-off review before 1.0 (#229 is closed in
favour of this). It is how every change is made:

- **The design principles** (`docs/Design Principles.rst`) are the rules
  every model keeps, each with the tests that enforce it. When a bug is
  fixed, ask which principle it broke, and extend that principle's check so
  it covers every model, or add a principle.
- **The conformance suite** runs every registered model through those
  properties on every pull request.
- **Review rounds.** A periodic adversarial review of one area or of the
  whole package, done as a new user would use it. Each round files concrete
  issues, which are fixed in the next round.
- **Performance sweeps.** Profile the main workloads at realistic sizes and
  fix what scales badly. The larger items become issues; this has been done
  once, in #515.
- **Calibration studies** check that intervals and tests achieve their
  stated coverage and size. They run nightly.

## Not planned

- **Internal time-varying covariates and joint longitudinal-survival
  models.** A covariate generated by the unit itself, such as a degrading
  signal, is only handled correctly by a joint model, which is a different
  class of method. SurPyval supports external covariates: measured ones
  fitted as steps, and known paths evaluated along continuous paths.
  (#172)
- **Reliability test planning** (sample sizes, demonstration tests) belongs
  in RePyability, not SurPyval. (#431, moved to RePyability)

## Done

Ideas from this file that have shipped, newest first.

- **Continuous time-varying covariates: evaluation** (phases 1 and 2 of
  #172), 0.22: `CovariatePath`, `sf_tvc` / `Hf_tvc` along a path,
  `cb_tvc`, `mean_tvc` and cumulative exposure for accelerated life
  models; likelihood-ratio `cb_tvc` in 0.23. Fitting is phase 3, above.
