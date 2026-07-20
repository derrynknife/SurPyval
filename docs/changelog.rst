Changelog
=========

v0.16.0 (unreleased)
--------------------

Diagnostics & validation
~~~~~~~~~~~~~~~~~~~~~~~~~

- **Cox model diagnostics** (#211). A fitted ``CoxPH`` model now exposes
  ``compute_residuals(kind=...)`` -- Schoenfeld, scaled Schoenfeld,
  martingale, deviance, score and dfbeta residuals -- and ``check_ph()``, the
  Grambsch-Therneau proportional-hazards test (a per-covariate and a joint
  global test against a transform of time; a small ``p``-value is evidence
  *against* proportional hazards). All residuals respect delayed entry
  (``tl``) and count weights. The residual identities are exact at the MLE
  (Schoenfeld, score and martingale residuals sum to zero) and the PH test is
  validated for both power (it detects a genuine time-varying coefficient) and
  calibration (its p-values are ~Uniform under true proportional hazards).
- **Restricted mean survival time** (#213). A fitted non-parametric model
  (e.g. ``KaplanMeier``) gains ``rmst(tau)`` -- the area under the survival
  curve to a horizon with its standard error and confidence interval -- and
  the package-level ``surpyval.rmst_diff(model_a, model_b, tau)`` compares two
  groups' RMST (difference, ratio, CI and a two-sided p-value). The
  RMST-difference is the assumption-light alternative to the hazard ratio when
  proportional hazards fails; the estimate matches its analytic value and the
  two-group test is calibrated under the null.
- **Cluster-robust standard errors** (#215). ``CoxPH`` models gain
  ``robust_covariance(cluster=...)`` and ``robust_summary(cluster=...)`` -- the
  Lin-Wei sandwich variance for clustered / correlated data (repeated events
  per subject, grouped sampling), built from the dfbeta residuals. On
  independent data it agrees with the model-based errors; on exactly
  replicated clusters it inflates by the theoretically exact
  ``sqrt(cluster size)``.
- **Gray's test** (#216). The package-level ``surpyval.gray_test`` compares
  cumulative incidence functions across groups for a specified cause in the
  presence of competing risks -- the subdistribution analogue of the log-rank
  test. Unlike a cause-specific log-rank, it keeps competing-cause failures in
  the risk set with an inverse-probability-of-censoring weight, so it tests the
  CIFs directly. Returns a chi-squared statistic, degrees of freedom and
  p-value. Validated for calibration under the null (including under heavy
  censoring, which exercises the IPCW weighting) and for power against genuine
  CIF differences.
- **Stratified Cox and stratified log-rank** (#214). ``CoxPH.fit`` /
  ``fit_from_df`` accept ``strata`` (or ``strata_col``) to fit a *stratified*
  proportional-hazards model: a separate baseline hazard per stratum with
  shared coefficients, the partial likelihood summed within strata. Prediction
  (``sf``/``Hf``/...) then takes a ``stratum`` argument to select that
  stratum's baseline. ``surpyval.logrank`` gains a ``strata`` argument for the
  stratified log-rank test (per-stratum observed-minus-expected and variance
  summed before forming the statistic). Both are the standard remedy when
  proportional hazards fails for a nuisance covariate. Validated by
  simulation: the stratified estimators recover the truth (and stay
  calibrated) in a confounded design where the pooled versions are badly
  biased / over-reject, reduce exactly to their unstratified counterparts with
  a single stratum, and the stratified Cox partial likelihood factorises into
  the per-stratum contributions.

v0.15.2 (20 Jul 2026)
---------------------

Data handling
~~~~~~~~~~~~~

- ``xcnt_handler`` now warns when right-censored observations carry a finite
  right-truncation time (#195). The combination is contradictory -- right
  truncation means the unit was only observable because its event occurred
  before ``tr``, while right censoring says the event is after the censoring
  time -- and such rows can make truncation-adjusted likelihoods unbounded.

Serialisation
~~~~~~~~~~~~~

- ``RenewalModel.from_dict`` now validates that the stored distribution name
  resolves to a genuine distribution fitter (#206), matching the guard used
  by every other reader, so an untrusted document cannot resolve arbitrary
  package attributes.

Misc
~~~~

- The bundled dataset loaders use pandas' default (C) CSV engine instead of
  ``engine="python"`` (#207) -- identical parses, faster, and one less thing
  for security scanners to worry about; the loaders are now covered by tests.
- Modernised the documentation build toolchain (``docs/requirements.txt``):
  the 2022-era pins (``sphinx 5.3``, ``jupyter-sphinx 0.4``) left ``ipykernel``
  unpinned, and against current ipykernel 7 the notebook execution hangs or
  crashes -- one of the reasons hosted docs builds kept failing. The new set
  (sphinx 8.2, sphinx-rtd-theme 3.1, jupyter-sphinx 0.5.3, ipykernel capped
  below 7) is fully pinned and validated by a complete docs build in a clean
  virtualenv.

v0.15.1 (20 Jul 2026)
---------------------

Non-parametric
~~~~~~~~~~~~~~

- **Fixed Turnbull fits with truncation hanging indefinitely** (this also hung
  the documentation builds, which is why the hosted docs went stale). The
  Fleming-Harrington tie ladder (``fh_h``/``fh_var_h``) was a per-event Python
  loop; the Turnbull EM feeds it *fractional expected* event counts which,
  under heavy truncation, can grow without bound between iterations -- the
  loop then effectively (or with an infinite count, literally) never
  returned. The ladder is now evaluated in closed form (digamma/trigamma
  harmonic sums) beyond a small exact loop, so its cost is O(1) in the event
  count: identical results for ordinary tie counts, and pathological counts
  now yield a diverging hazard (``inf``) instead of a hang. Note that the
  truncated NPMLE itself remains delicate on small or heavily truncated
  samples (it can be non-identifiable and the EM converges to a degenerate
  estimate); such fits now terminate and are flagged, and the docs note the
  caveat.

v0.15.0 (20 Jul 2026)
---------------------

Serialisation
~~~~~~~~~~~~~

- Every serialised model dictionary now carries a schema version
  (``"schema": 1``), stamped by every ``to_dict``. The version is bumped only
  when a dictionary's shape changes incompatibly, so documents stored today
  (in files or MongoDB) stay recognisable to future SurPyval versions: the
  package-level ``surpyval.from_dict`` refuses documents written by a *newer*
  schema with a clear error, and treats documents with no ``"schema"`` key
  (written before versioning) as schema 0, which remains loadable.
- MongoDB compatibility, verified for every serialisable model: BSON is
  stricter than JSON (numpy integer scalars and arrays are rejected, and
  dictionary keys must be strings), so every model's ``to_dict`` output is now
  tested through the full MongoDB path -- ``bson.encode`` (what
  ``insert_one`` does), decode, add the ``_id`` field ``find_one`` returns,
  and restore via ``surpyval.from_dict`` with predictions reproduced. The
  cause-label fields of the competing-risks containers are now normalised to
  native Python types with a new ``surpyval.serialisation.to_native`` helper
  (numpy labels passed by the caller no longer leak into the document), and
  ``pymongo`` was added to the test dependencies for the BSON round-trip
  tests.
- Added package-level readers for serialised models:
  ``surpyval.from_dict(model_dict)`` and ``surpyval.from_json(fp)`` restore a
  model of the right class from any model's ``to_dict`` dictionary /
  ``to_json`` file, so the caller no longer needs to know which class wrote
  it. Dispatch reads the serialised dictionary itself: the ``"model"`` class
  tag written by most models, or the ``"parameterization"`` marker
  (``"parametric"``, ``"non-parametric"``, ``"parametric-regression"``) of the
  core univariate families. The class-level readers are unchanged.

Package structure
~~~~~~~~~~~~~~~~~

- Pre-stable models are now tiered by maturity: ``surpyval.alpha``
  (exploratory; the interfaces may change or disappear -- currently the
  ``ParallelModel``/``SeriesModel`` system models, previously in
  ``surpyval.experimental``) and ``surpyval.beta`` (functionally complete
  and tested, interface not yet part of the release contract -- the
  survival tree and random survival forest in ``surpyval.beta.ml``).
  ``surpyval.experimental`` remains as a deprecated re-export of both and
  warns on import.

Machine learning
~~~~~~~~~~~~~~~~

- The survival tree and random survival forest graduated from
  ``surpyval.experimental`` to the beta tier:
  ``from surpyval.beta.ml import SurvivalTree, RandomSurvivalForest``. The
  old ``surpyval.experimental`` imports still work as re-exports. Their test
  suite now runs in CI, expanded with behavioural and structural tests:
  prediction coherence (``ff = 1 - sf``, ``Hf = -log(sf)``, monotone
  bounded ``sf``), ``max_depth``/``min_leaf_samples``/``min_leaf_failures``
  guarantees, seeded determinism, degenerate inputs (all-censored,
  constant covariates, tiny samples, tied times, count weights), forest
  ensemble maths (the forest ``sf`` is exactly the tree average; the
  ``"Hf"`` method averages cumulative hazards), prediction shapes,
  mortality ordering and a concordance sanity check.
- Fixed the concordance index (``surpyval.utils.score.score``, used by
  ``RandomSurvivalForest.score``): pairs were ordered by censoring flag
  instead of by time before comparison, which pushed the c-index of even a
  strongly informative forest towards 0.5. Pairs are now ordered by time
  (event first on exact ties), so ``score`` returns Harrell's c-index for
  mortality-like scores (1 = perfectly concordant). ``forest.score`` also
  now respects its ``tie_tol`` argument.

Competing risks & mixtures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Added serialisation to the competing-risks and mixture models:
  ``MixtureModel`` (EM mixture of a base family), ``FineGrayModel``
  (subdistribution-hazard regression), ``ParametricCompetingRisks`` (one
  distribution per cause) and the nonparametric ``CompetingRisks`` now have
  ``to_dict``/``from_dict`` and ``to_json``/``from_json``. The mixture stores
  its base-family name, component parameters and weights; Fine-Gray stores its
  coefficients, covariance and subdistribution-baseline step arrays; and the
  competing-risks models store their per-cause sub-models (via each cause's own
  ``to_dict``) or per-event step arrays. Every reloaded model reproduces its
  predictions exactly.

Degradation
~~~~~~~~~~~

- Added serialisation to the fitted degradation models:
  ``DegradationModel``, the stochastic-process models ``WienerProcessModel``
  and ``GammaProcessModel``, and the Monte-Carlo ``InducedFailureDistribution``
  now have ``to_dict``/``from_dict`` and ``to_json``/``from_json``. The process
  models store their few parameters; the induced distribution stores its
  samples (the ``inf`` never-fails draws are written as ``null`` so the result
  is valid JSON); and ``DegradationModel`` stores its raw data, the path model
  (by name) and per-unit fits, the population summaries, and the fitted life
  model (via its own ``to_dict`` -- plain or accelerated), so the reloaded
  model reproduces its predictions and per-unit paths and (because the data is
  kept) its bootstrap confidence bounds too.

Recurrent events
~~~~~~~~~~~~~~~~~

- Added serialisation to the renewal / imperfect-repair models
  (``RenewalModel``): the generalized-renewal (Kijima-I/II), G1 renewal, ARA
  and ARI families now have ``to_dict``/``from_dict`` and
  ``to_json``/``from_json``. These processes have no closed-form intensity
  (their MCF comes from a sampler closure that cannot be pickled), so the dict
  stores the family, the underlying distribution (by name) and its parameters,
  the restoration parameter and the family option (``kijima_type`` or memory
  ``m``); on load the family's fitter rebuilds the sampler from those, so the
  simulated MCF reproduces exactly. This completes serialisation coverage of
  every non-experimental fitted model in the package.
- Added serialisation to the fitted recurrent-event models:
  ``ParametricRecurrenceModel`` (NHPP/HPP intensity fits),
  ``NonParametricCounting`` (the MCF estimate), ``ProportionalIntensityModel``
  (proportional-intensity regression), and the competing-risks containers
  ``CauseSpecificMCF`` and ``CauseSpecificNHPP`` now have
  ``to_dict``/``from_dict`` and ``to_json``/``from_json``. The intensity model
  is stateless, so each stores its name plus the fitted parameters (or, for the
  MCF, the ``x``/``mcf_hat``/``var`` step arrays), and the reloaded model
  reproduces ``cif``/``iif``/``mcf`` exactly. Intensity models are resolved by
  name from a restricted set. The likelihood/data state is not stored, so a
  reloaded model behaves like a ``from_params`` one for confidence bounds and
  diagnostics.

Regression
~~~~~~~~~~

- Added serialisation to the **semi-parametric** regression models, each on its
  own result class: Cox proportional hazards
  (``SemiParametricRegressionModel``), the Lin-Ying additive-hazards model
  (``AdditiveHazardsModel``), and the Buckley-James AFT (``BuckleyJamesModel``)
  now have ``to_dict``/``from_dict`` and ``to_json``/``from_json``. Because the
  baseline is nonparametric, the coefficients plus the fitted baseline step
  arrays (or, for Buckley-James, the residual survival) are stored, so the
  reloaded model predicts identically -- including Cox's ``predict_tvc`` for a
  time-varying-covariate fit, the additive model's covariance / standard
  errors, and Buckley-James's ``bootstrap_ci`` (its fit data is kept).
  ``SemiParametricRegressionModel`` is now exported from
  ``surpyval.univariate.regression``.
- Added serialisation to the parametric regression models:
  ``ParametricRegressionModel`` now has ``to_dict``/``from_dict`` and
  ``to_json``/``from_json``, so a fitted Accelerated Failure Time,
  Proportional Hazards, Proportional Odds or (parametric) Additive Hazards
  model can be saved and rebuilt without the training data. The restored model
  predicts identically (``sf``/``ff``/``df``/``hf``/``Hf``/``phi``/``random``);
  if the fit's parameter covariance was computable it is stored too, so the
  reloaded model also produces confidence bounds (``cb``/``param_cb``/
  ``standard_errors``). Distribution and family are resolved by name from a
  restricted set, so an untrusted dict cannot load arbitrary objects. Models
  with a bespoke covariate link (an Accelerated Life parameter-substitution
  model) are refused with a clear error. ``ParametricRegressionModel`` is now
  exported from ``surpyval.univariate.regression``.

Experimental
~~~~~~~~~~~~

- **Breaking (experimental API):** the survival tree/forest now take a single
  ``kind`` parameter that couples the split criterion with its matching leaf
  model, replacing the independent ``split_rule`` / ``leaf_type`` /
  ``parametric`` knobs (whose free combination invited mismatched trees and
  whose defaults disagreed between entry points). ``kind="weibull"`` (the new
  default) adds the **Weibull deviance split** -- a 2-d.f. likelihood-ratio
  gain computed with the full likelihood, with power against *scale and
  shape* differences (e.g. crossing-hazards populations that the exponential
  rule and the log-rank statistic largely miss) -- paired with Weibull MLE
  leaves. ``kind="exponential"`` is the Davis-Anderson rule with Exponential
  leaves, and ``kind="non-parametric"`` is the risk-set log-rank with
  Nelson-Aalen leaves (observed/right-censored data, optionally
  left-truncated; raises otherwise). Parametric kinds now stay parametric all
  the way down: the degenerate-leaf rescue ladder is Weibull -> Exponential ->
  crude rate, never a nonparametric leaf. Split-search child fits warm-start
  from the parent's optimum, which also guarantees a non-negative split gain
  in the 2-parameter case. The internal Weibull MLE is cross-validated
  against ``Weibull.fit`` on every data configuration.
  now supports the **full SurPyval data model**: observed, left-, right- and
  interval-censored observations with optional left and/or right truncation.
  The risk-set log-rank split only exists for observed / right-censored
  (optionally left-truncated) data, so the tree gains a second split
  criterion -- the full-likelihood exponential deviance split of Davis &
  Anderson (1989) -- in which every candidate split is scored by the joint
  maximised exponential log-likelihood of its children, with each observation
  type contributing its exact likelihood term (including the
  ``S(t_l) - S(t_r)`` truncation correction). A new ``split_rule`` parameter
  (``"auto"`` default) keeps the log-rank split for data it is defined on --
  existing behaviour is unchanged -- and switches to the deviance split
  otherwise; forcing ``"log-rank"`` on incompatible data raises a clear
  error. All candidate children within a node are scored over a common
  parameter window so the criterion is monotone (a split can never score
  below its parent), and splits with no likelihood gain stop the branch.
  Nonparametric leaves now use the Turnbull NPMLE when the data has left or
  interval censoring or right truncation (Nelson-Aalen otherwise, as
  before); parametric (Weibull) leaves already supported the full data
  model. ``fit`` also accepts the ``xl``/``xr`` and ``tl``/``tr``
  conveniences.
- Fixed a crash in the experimental ``RandomSurvivalForest``: a degenerate
  bootstrap sample (e.g. heavily tied event times) could make a terminal
  node's Weibull covariance step raise, taking down the whole forest fit. A
  terminal node now falls back to progressively simpler, more robust fits
  (Exponential, then Nelson-Aalen). The experimental modules are also excluded
  from the CI test run, since they are not part of the release contract.

Degradation
~~~~~~~~~~~

- Added two-stage confidence bounds for the **accelerated-degradation
  (covariate) life fit**: ``DegradationModel.cb`` now accepts a stress vector
  ``Z`` and, with ``method="bootstrap"``, resamples units (each carrying its
  stress) and reruns the whole ADT pipeline to fold the first-stage
  path/extrapolation uncertainty into the reliability at ``Z``. Previously
  ``cb`` raised ``NotImplementedError`` for covariate models; the analytic
  (generated-regressor) correction remains underived for the regression fit, so
  bootstrap is required there. The bootstrap holds the selected path model
  fixed, so it composes cleanly with ``path="best"`` (no per-resample path
  re-selection). ``cb`` also now validates ``Z`` (required for covariate
  models, rejected for plain ones).
- Extended ``population_method="reml"`` to **nonlinear** path models
  (exponential, power, Gompertz, ...). Previously REML population estimation
  was restricted to paths linear in their parameters; nonlinear paths are now
  fitted with the Lindstrom-Bates (1990) FOCE alternating algorithm -- each
  unit's parameters are estimated at their conditional (penalised-least-
  squares) mode, the path is linearised about that mode into a working linear
  mixed model, and the linear REML step is iterated to convergence. This gives
  a positive-definite ``path_param_cov`` by construction (no PSD clipping) for
  nonlinear paths too, which is the more robust population estimate when the
  unit count is small. On a linear-in-parameters path the routine reduces
  exactly to the previous linear REML fit in a single pass.
- Added the Lu-Meeker induced failure-time distribution:
  ``DegradationModel.induced_life`` derives the population failure-time
  distribution directly from the fitted path-parameter distribution -- drawing
  path parameters ``theta ~ N(path_param_mean, path_param_cov)`` and pushing
  each through the path model's ``inv_path(threshold)`` by Monte Carlo --
  rather than via each unit's noisy pseudo failure time. It returns an
  ``InducedFailureDistribution`` exposing ``sf``/``ff``/``qf``/``mean``/
  ``median``/``random`` (with an ``inf`` "never fails" mass reported as
  ``prob_never_fails``), a diagnostic complement to the pseudo-failure-time
  life fit that the two can be overlaid to check.
- Added stochastic-process degradation models that model the degradation
  increments directly, deriving the failure-time distribution from the
  process's first passage to the threshold (rather than via pseudo failure
  times), and handling irregular measurement spacing naturally. Two
  complementary processes are provided in ``surpyval.degradation``:
  ``WienerProcess`` (Brownian motion with drift, for non-monotone / noisy
  signals; its first passage is a closed-form Inverse-Gaussian law) and
  ``GammaProcess`` (monotone increasing increments, for irreversible damage
  such as wear, corrosion or crack growth; its first-passage distribution
  comes from the incomplete gamma function). Both fit by maximum likelihood
  from ``(x, y, i)`` measurement data and expose the induced failure-time
  distribution (``sf``/``ff``/``df``/``hf``/``Hf``/``qf``/``mean``/``random``)
  plus a ``predict_rul`` remaining-useful-life summary. The degradation
  documentation gained an expansive section explaining both processes, what
  each parameter means, the first-passage failure-time derivation, worked
  runnable examples, and guidance on choosing between them.

v0.14.0 (19 Jul 2026)
---------------------

Documentation
~~~~~~~~~~~~~

- Substantially expanded the recurrent-event documentation for the release.
  The theory pages now cover the arithmetic-reduction (ARA/ARI) models, the
  geometric-process view of the G1 renewal process, the time-rescaling
  residual / trend-test / Cramer-von Mises diagnostics, marked (competing-risks)
  recurrent events, gapped multi-window observation, and truncation, each with a
  short References section. The worked-example pages gained runnable
  demonstrations of ARA/ARI, renewal-model checking, gapped observation, the
  cause-specific MCF and intensity models, and a full build-out of the
  proportional-intensity regression examples.
- Fixed and completed the recurrent-event API reference. Every model's
  autodoc page (HPP, Duane, Cox-Lewis, Crow-AMSAA, the renewal and
  proportional-intensity models) previously rendered as an empty "alias of
  object" because the fitters are exposed as singletons; the pages now
  document each model's methods. Added missing API pages for ``ARA``, ``ARI``,
  ``NonParametricCounting``, ``CauseSpecificMCF``, ``CauseSpecificNHPP`` and the
  fitted ``RenewalModel`` object.

Recurrent events
~~~~~~~~~~~~~~~~

- Added residual (``residuals``: ``cumulative_hazard`` / ``pit`` /
  ``martingale``), trend-test (``trend_test``) and Cramer-von Mises
  goodness-of-fit (``cramer_von_mises``) diagnostics to the renewal /
  virtual-age imperfect-repair models (``GeneralizedRenewal``,
  ``GeneralizedOneRenewal``, ``ARA``, ``ARI``), completing the diagnostic
  coverage of the recurrent module. These processes have no marginal
  cumulative intensity, so the time-rescaling residuals come from each one's
  *conditional* intensity -- the cumulative hazard accumulated over each
  interarrival given the model's virtual age (Kijima / ARA), time scaling
  (G1R) or intensity reduction (ARI) -- and are iid Exp(1) under the fitted
  model. The Cramer-von Mises transforms use the compensator built from those
  increments (there being no closed-form intensity), and its p-value comes
  from a parametric bootstrap that resimulates each item and refits the full
  imperfect-repair model per replicate.
- Added support for gapped (multi-window) observation: an item can be observed
  over several disjoint time windows with unobserved gaps in between (events
  may occur during a gap but are not recorded). Pass ``windows={item:
  [(start, end), ...]}`` to the intensity fitters (``HPP``, ``CrowAMSAA``,
  ``Duane``, ``CoxLewis``) and the nonparametric ``NonParametricCounting`` MCF;
  every row of ``x`` is then an observed event and the windows supply the
  end-of-window censoring. Because event counts over disjoint windows are
  independent for an NHPP, each window is fitted as its own observation period,
  so the intensity likelihood and the MCF at-risk set (an item is absent from
  the risk set during its gaps) both handle the gaps exactly. The virtual-age /
  renewal models (``GeneralizedRenewal``, ``GeneralizedOneRenewal``, ``ARA``,
  ``ARI``) reject gapped data, since the virtual age at the start of a later
  window depends on the unobserved events during the gap.
- Recurrent event marks (competing-risks recurrent events) are now first
  class. ``handle_xicn`` takes an event-type mark ``e`` per row (with
  ``None``/``NaN`` marks normalised to a single "no cause" sentinel), so marked
  data gets the same validation, sorting and truncation handling as every
  other recurrent fit. ``CauseSpecificMCF`` now routes through that handler and
  gains a ``fit_from_df``. New ``CauseSpecificNHPP`` fits a **parametric
  cause-specific intensity model** -- one NHPP (``CrowAMSAA`` by default, or any
  counting-process fitter) per event type. Because a marked Poisson process
  decomposes into independent thinned Poisson processes, each cause is fitted
  to its own events over the full observation window of every item (other-cause
  events are ignored, exactly as a censored period would be), so each
  per-cause model is an ordinary fitted recurrence model with its full
  ``cif``/``iif``, inference and diagnostics; ``total_cif`` sums them for the
  overall event intensity.

v0.13.0 (18 Jul 2026)
---------------------

Distributions
~~~~~~~~~~~~~~

- Added three Tier-2 discrete distributions: ``Poisson`` (the count
  distribution on ``{0, 1, 2, ...}``, distinct from the recurrent Poisson
  *processes*), ``BetaGeometric`` (a discrete-time frailty model — Geometric
  with a Beta-mixed failure probability, whose marginal hazard decreases with
  time), and ``Discretize(distribution)``, a factory that turns any
  non-negative continuous distribution into its integer-binned counterpart
  (``K = ceil(T)``, so ``P(K=k) = F(k) - F(k-1)`` and the discrete survival
  equals the continuous survival), fit by MLE on the underlying parameters.
- ``Beta.fit(how="MPP")`` now raises a clear ``ValueError`` (the Beta has no
  linearising probability plot) instead of a raw ``NotImplementedError``, and
  points to ``MLE`` / ``MSE`` / ``MOM``.
- ``Parametric.moment`` now works for limited-failure, zero-inflated and
  offset models (it previously raised ``NotImplementedError`` under a cure
  fraction, and silently dropped the offset). It returns the defective moment
  of the failure-time density, consistent with ``mean`` (``moment(1) ==
  mean()``): the offset shifts the failure times and the cured fraction
  contributes nothing. ``Parametric.entropy`` likewise handles the offset
  (differential entropy is translation-invariant) and now raises a clear
  ``ValueError`` for models with a probability atom (a limited-failure mass at
  infinity or a zero-inflation mass at the offset), where a single differential
  entropy does not exist -- it previously returned a wrong value for
  zero-inflated models.
- ``Parametric.qf`` now works for limited-failure, zero-inflated and offset
  models (it previously raised ``NotImplementedError`` whenever a cure fraction
  was present). It inverts the full mixture ``F(x) = f0 + (p - f0) F0(x -
  gamma)``: quantiles at or below the zero-inflation mass ``f0`` return the
  offset, and quantiles at or above the attainable proportion ``p`` are
  infinite (that cured fraction never fails, so e.g. the median of a
  majority-cured population is ``inf``). This also **fixes** the quantile of a
  zero-inflated (``p == 1``, ``f0 > 0``) model, which previously ignored
  ``f0`` and returned the wrong value.

Competing risks
~~~~~~~~~~~~~~~

- Added ``ParametricCompetingRisks``, a fully parametric competing-risks model:
  a parametric distribution is fitted to each cause's cause-specific hazard
  (the joint likelihood factorises, so each cause is fitted with the other
  causes' events treated as right-censored) and smooth, extrapolatable
  cumulative-incidence functions are assembled from them. Provides ``fit`` /
  ``fit_from_df`` (with a per-cause distribution mapping), all-cause and
  cause-specific ``hf`` / ``Hf`` / ``sf`` / ``ff``, the subdistribution density
  ``iif``, the cumulative incidence ``cif``, ``probability_of_cause``, sampling
  via ``random``, and ``aic`` / ``bic`` / ``neg_ll``. Complements the existing
  nonparametric ``CompetingRisks`` estimator and the semi-parametric
  cause-specific Cox / Fine-Gray regression models.
- ``ParametricCompetingRisks.from_fitted`` assembles a competing-risks model
  from already-fitted per-cause models, each of any family and configuration
  (e.g. a limited-failure Weibull for one cause, a LogNormal for another): pass
  a ``{cause: model}`` mapping or a sequence of models. Sampling handles cure
  fractions -- when every cause carries one, some units never fail and are
  returned with cause ``None``.
- Every competing-risks model (parametric, nonparametric, and the Fine-Gray /
  cause-specific Cox regression) now treats a *missing* event value (``None``,
  ``NaN`` or pandas ``NA``) as a censored observation with no attributed cause,
  and derives the censoring flag ``c`` from the events when it is not supplied
  -- so competing-risks data can be given as ``(x, e)`` alone, and a pandas
  cause column with ``NaN`` for censored rows works directly.

Recurrent events
~~~~~~~~~~~~~~~~

- Added residual (``residuals``: ``cumulative_hazard`` / ``pit`` /
  ``martingale``), trend-test (``trend_test``) and Cramer-von Mises
  goodness-of-fit (``cramer_von_mises``) diagnostics to the
  proportional-intensity regression models (``ProportionalIntensityHPP`` /
  ``ProportionalIntensityNHPP``), matching those already on the parametric
  recurrence models. Each item's time-rescaling residuals and conditionally-
  uniform transforms use its own covariate-scaled cumulative intensity
  ``Lambda_0(t) exp(Z'beta)``, and the Cramer-von Mises p-value comes from a
  parametric bootstrap that refits the full regression model per replicate.

Regression — Cox proportional hazards
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Added time-varying-covariate support in counting-process (start-stop)
  format: ``CoxPH.fit_tvc`` / ``fit_tvc_from_df`` take one row per interval
  ``(ident, start, stop, event, Z)``, validated by ``handle_tvc``, and
  ``SemiParametricRegressionModel.predict_tvc`` gives a subject's survival
  along a supplied covariate path.
- **Fixed** the Breslow baseline hazard to respect left-truncation / delayed
  entry (``tl``) and case weights (``n``); ``H0`` was previously wrong for any
  delayed-entry fit even though the coefficients were correct.
- ``CoxPH.fit`` gained a minimisation fallback so staggered delayed-entry data
  (e.g. the start-stop representation) converges where the root-finder stalled.
- Right / interval truncation is now rejected with a clear, Cox-specific error
  (a 2-D ``tl``), since the forward partial likelihood cannot express it.

Truncation
~~~~~~~~~~

- Verified and tested that the parametric AFT / PO / PH truncation correction
  uses each row's own covariates: a covariate-recovery test confirms the
  coefficient and scale are recovered from left-, right-, interval- and
  partially-truncated data.

Documentation
~~~~~~~~~~~~~~

- Added worked, executed examples for regression confidence bounds,
  Buckley-James AFT, competing-risks regression (Fine-Gray + cause-specific
  Cox), degradation ADT covariates and two-stage bounds, the copula module,
  and the combined data-input flexibility; wrote the Maximum Product of
  Spacings (MPS) estimation theory section.

v0.12.0 (15 Jul 2026)
---------------------

A large release consolidating the regression, recurrent-event, competing-risks,
degradation, and multivariate work accumulated since ``v0.10.1``. Requires
Python 3.11+ and NumPy 2.

Regression
~~~~~~~~~~

- Standardised every univariate regression fitter (accelerated failure time,
  proportional hazards, proportional odds, additive hazards, accelerated life)
  on a common instance-based ``fit()`` / ``fit_from_df()`` API with pandas and
  `formulaic <https://matthewwardrop.github.io/formulaic/>`_ formula support.
- ``CoxPH`` gained the Efron tie handling in addition to Breslow, and its
  analytic (Efron) information matrix is now correct, so standard errors and
  p-values are produced for tied data.
- Added delta-method confidence bounds to the parametric regression models:
  ``cb()`` on a predicted function at a covariate vector, ``param_cb()`` on a
  single coefficient, and ``covariance()`` / ``standard_errors()`` /
  ``parameter_names()`` on the fitted parameters.
- Added ``BuckleyJames``, a semi-parametric accelerated-failure-time model with
  an unspecified error distribution (the accelerated-time counterpart of Cox),
  fitted by the Buckley-James imputation iteration with percentile-bootstrap
  coefficient intervals.
- Added a parametric ``AdditiveHazards`` regression fitter.

Competing risks
~~~~~~~~~~~~~~~~

- Added a competing-risks regression module with a cause-specific Cox model and
  a Fine-Gray subdistribution-hazard model (``CompetingRisksProportionalHazards``),
  each with ``fit()`` / ``fit_from_df()`` and cumulative-incidence prediction.

Recurrent events
~~~~~~~~~~~~~~~~~

- Standardised the recurrent-model API on the same instance-based fitters the
  univariate distributions use: ``HPP``, ``CrowAMSAA``, ``Duane``,
  ``CoxLewis``, ``NonParametricCounting``, the renewal fitters
  (``GeneralizedRenewal``/``GeneralizedOneRenewal``/``ARA``/``ARI``) and the
  proportional-intensity fitters are now configured singleton instances with an
  instance-method ``fit()``. Public ``Model.fit(...)`` calls are unchanged;
  internally provided by the ``surpyval.utils.fitter.singleton_fitter``
  decorator. Removed the unused ``ParametricRecurrenceRegressionModel`` stub.
- Added parameter-uncertainty and diagnostic support to the recurrent models,
  and removed the ``dist='t'`` heuristic from the recurrent ``mcf_cb``.

Degradation
~~~~~~~~~~~

- Added the ``surpyval.degradation`` pseudo-failure-time analysis module:
  per-unit path fits over a library of path models, extrapolation to a failure
  threshold, and a fitted life distribution, with population path-parameter
  estimation (Lu-Meeker two-stage and REML) and Bayesian remaining-useful-life
  prediction (``predict_rul``).
- Added two-stage (delta-method and bootstrap) confidence bounds on the fitted
  life model that fold in the first-stage path/extrapolation uncertainty
  (``DegradationModel.cb`` / ``life_parameter_covariance``).
- Added Stage-1 accelerated degradation testing (ADT) covariates: passing
  ``Z`` to ``DegradationAnalysis.fit`` fits a regression life model on the
  pseudo failure times so life can be predicted at any stress condition.

Multivariate
~~~~~~~~~~~~~

- Added a ``surpyval.multivariate`` module with copula models over the
  univariate distributions.

Distributions and core
~~~~~~~~~~~~~~~~~~~~~~~~

- Added discrete lifetime distributions.
- Hardened input validation in the ``handle_xicn`` / ``xcnt_handler`` data
  handlers, and fixed a reserved-attribute clash.
- Simulation and ``dist='t'`` cleanups.

v0.10.1.0 (25 Mar 2022)
-----------------------

- Changed plot methods to now take 'Axis' object. This allows a user to pass in an existing axis.
- plot functions now return an Axis object instead of the Lines2D object. Allows for easy user update after plotting.
- Added fs_to_xcn as it was dropped in 10.0.1.
- Changed all imports for numpy to be done from the surpyval module. This will allow for easy maintenance in future in the event of deprecated autograd.

v0.10.0.1 (22 Nov 2021)
-----------------------

- Removed fsl_to_xcn function and replaced with fsli_to_xcn function that is able to take any combination of fsli.

v0.10.0 (9 Aug 2021)
--------------------

- Version snapshot for JOSS review

v0.9.0 (5 Aug 2021)
-------------------

- Better initial estimates in the ``_parameter_initialiser`` for the lfp data (use max F from nonp estimate...)
- `issue #13 <https://github.com/derrynknife/SurPyval/issues/13>`_ - Better failures when insufficient data provided.
- `issue #12 <https://github.com/derrynknife/SurPyval/issues/12>`_ - Created ``fsli_to_xcn`` helper function.
- Fixed bug in confidence bounds implementation for offset distributions. CBs were not using the offset and were therefore way out. Now fixed.
- Created a  ``NonParametric.cb()`` method to match ``Parametric`` API for confidence bounds.
- Cleaned up NonParametric code (removed some technical debt and duplicated code).
- Changed the ``__repr__`` function in ``NonParametric`` to be aligned to ``Parametric``
- Updated the docstring for ``fit()`` for ``NonParametric``
- Fixed bug in ``NonParametric`` that required the ``x`` input to be in order for the functions (e.g. ``df`` etc.).
- ``CoxPH`` released.
- General AL fitter in beta
- General PH fitter in beta
- Created ``Linear``, ``Power``, ``InversePower``, ``Exponential``, ``InverseExponential``, ``Eyring``, ``InverseEyring``, ``DualPower``, ``PowerExponential``, ``DualExponential`` life models.
- Created ``GeneralLogLinear`` life model for variable stress count input.
- For each combination of a SurPyval distribution and life model, there is an instance to use ``fit()``. For example there are ``WeibullDualExponential``, ``LogNormalPower``, ``ExponentialExponential`` etc.
- Docs Updates:
	- Add application examples to docs:
		- Reliability Engineering
		- Actuary / Demography
		- `Social Science/Criminology <https://link.springer.com/article/10.1007/s10940-021-09499-5>`_
		- Boston Housing
		- Medical science
		- `Economics <https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0232615>`_
		- Biology - Ware, J.H., Demets, D.L.: Reanalysis of some baboon descent data. Biometrics 459–463 (1976).

v0.8.0 (27 July 2021)
---------------------

- Made backwards incompatible changes to ``LFP`` models, these are now created with the ``lfp=True`` keyword in the ``fit()`` method
- Created ability to fit zero-inflated models. Simply pass the ``zi=True`` option to the ``fit()`` method.
- Chanages to ``utils.xcnt_handler`` to ensure ``x``, ``xl``, and ``xr`` are handled consistently.
- changed the way ``__repr__`` displays a Parametric object.
- Changed the default for plotting to be ``Fleming-Harrington``. This was a result of seeing how poorly the ``Nelson-Aalen`` method fits zero inflated models. FH therefore offers the best performance of a Non-Parametric estimate at the low values of the survival function (as KM reaches 0 for fully observed data) and at high values (KM is good but NA is poor).
- Added a Fleming-Harrington method to the Turnbull class.
- Improved stability with dedicated ``log_sf``, ``log_ff``, and ``log_df`` functions. Less chance of overflows and therefore better convergence.
- Changed interpolation method of ``NonParametric``. Allows for use of cubic interpolation
- Changed ``from_params`` to accept lfp and zi (or any combo)
- Changed ``random()`` in ``Parametric`` so that lfp or zi models can be simulated!
- Improved the way surpyval fails
- Substantial docs updates.


v0.7.0 (19 July 2021)
---------------------

- Major changes to the confidence bounds for ``Parametric`` models. Now use the ``cb()`` method for every bound.
- Removed the ``OffsetParametric`` class and made ``Parametric`` class now work with (or without) an offset.
- Minor doc updates.