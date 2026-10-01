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

*Status: designed. Size: medium to large. From #172.*

Phase 1 (shipped in 0.22) evaluates a fitted model along a known, external
covariate path; phase 2 (bounds, the mean, and accelerated life along paths)
is in progress. Phase 3 fits on
continuous per-subject histories, such as ramp-stress accelerated life tests
where each unit had a known ramp.

1. **Cox.** Split each subject's path at the event times and call
   `CoxPH.fit_tvc`. Exact and small.
2. **Parametric PH, AH and PO.** The path integral goes inside the
   likelihood, on a mesh frozen per subject. An interim option is to
   discretise each path into start-stop rows, refit at twice the
   resolution, and warn if the estimates move.
3. **AFT and accelerated life.** `aft_tvc_fit`, with the time
   transformation computed by the path engine.

The design is in the #172 proposal.

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

- *None yet.* (Before this file existed, large items were tracked in the
  issues; see the changelog.)
