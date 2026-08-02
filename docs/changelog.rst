Changelog
=========

v0.17.1 (unreleased)
--------------------

- **ExpoWeibull no longer runs a nested optimiser ladder to build its
  initial guess.** It seeds itself from a Gumbel fit to ``log(x)``, and
  that inner fit was a full maximum likelihood run -- an optimiser
  ladder, to produce a *starting point* for another optimiser. It cost
  15-30% of the fit (20 ms at n=200, 41 ms at n=5000) and the
  probability plot alone turned out to be just as good a seed: across 54
  parameter combinations, plus right-censored, left-censored and heavily
  tied data, every fit reached the same optimum to the optimiser's own
  tolerance. Ordinary fits are about 20% faster.

  The offset path keeps the refinement. There the seed reads ``log(x)``
  of the *unshifted* data, so a large shift compresses the logs into a
  narrow band and the probability plot is a poor starting point --
  dropping it moved one fit in twelve to a worse optimum (861.898 to
  861.962) and made that fit seven times slower.

- **The ARI likelihood is evaluated for the whole sample at once, making
  imperfect-repair fits 30-160x faster.** It used to walk every event in
  Python, calling the baseline ``cif`` and ``iif`` and rebuilding the
  intensity reduction from the failure history at each step -- around
  ten milliseconds per event, repeated for every one of the optimiser's
  several hundred objective evaluations. Fitting 250 items took 19
  seconds and 1000 items was impractical.

  The apparent obstacle is that the reduction depends on the failure
  history, which looks inherently sequential. It is not: summing over
  the reduction's *window offset* rather than over the failures turns it
  into a handful of whole-array passes, and the offset only ever runs to
  ``min(m, longest item)`` -- exactly one pass for ARI1. Each failure's
  ordinal within its own item bounds the window, which is what stops one
  item's history leaking into the next.

  =========================  ==========  ==========
  fit                        before      after
  =========================  ==========  ==========
  35 items x 6 events        2.11 s      0.07 s
  250 items x 8 events       19.02 s     0.12 s
  Cramer-von Mises, 10 boot  24.16 s     1.71 s
  1000 items x 10 events     ~80 s       0.53 s
  =========================  ==========  ==========

  Results are unchanged to floating point: over 312 captured values --
  the reduction helper across memory regimes, the objective on a fixed
  parameter grid, fitted parameters and the rescaled-increment
  residuals -- the largest relative difference is 1.1e-14, from
  summation order. The original per-event implementation is kept in the
  test suite as an oracle so the two cannot drift.

  One behavioural nicety: a non-positive reduced intensity is outside
  the model's support, and the scalar loop returned ``inf`` early on
  reaching one. The vectorised form tests the whole array before taking
  logs, so it returns ``inf`` rather than warning its way to a ``nan``.

- **Method of moments now matches central moments rather than raw
  ones.** The two describe the same estimator -- the binomial transform
  between them is exact and bijective, so matching the first ``k``
  central moments is matching the first ``k`` raw moments -- but raw
  moments hide the answer from the optimiser once a distribution is
  offset. ``E[X^k]`` is then dominated by ``gamma^k`` and the shape
  contributes only a fractional correction: 0.5% of ``E[X^3]`` for a
  Gamma(3, 4) shifted by 10. Fitting three parameters off the third
  decimal place of a large number degenerates, and offset fits settled
  on parameters that matched the sample moments *better than the true
  parameters did* while being nowhere near them -- a shape of 17.7
  against a true 3.0, unchanged at any sample size.

  Central moments remove the offset by construction, so the shape is
  the whole of the third moment rather than a rounding error in it. An
  offset Gamma at n=5000 goes from ``gamma=49.01, alpha=15.74,
  beta=9.07`` to ``gamma=50.01, alpha=2.74, beta=3.75`` against a true
  ``(50, 3, 4)``, and the fit drops from up to 25 s to under a second.
  Unshifted fits are unaffected -- Weibull, Gamma, Normal, LogNormal,
  Logistic, Gumbel and Exponential all agree with the previous results
  to at least four decimal places, because there the conditioning was
  never the problem.

  The terms are scaled by the sample's own ``sigma^k``, so each is
  dimensionless: the mean in units of sigma, the relative variance
  error, then the skewness difference. The mismatch warning's threshold
  moves from 1e-4 to 1e-2 to suit those units. Healthy fits land near
  1e-12 when the moment equations have an exact solution and near 1e-3
  when sampling noise means none exists and the optimiser returns the
  closest match; a fit that has actually failed sits near 0.5.

- **Offset Gamma fits no longer start from a corrupted initial guess.**
  Every offset-capable distribution returns the shift first in its
  parameter vector, because ``_initial_guess`` overwrites that slot
  with its own estimate of the shift. ``Gamma`` returned it *last*, so
  the overwrite landed on the shape parameter and destroyed it, while
  the initialiser's own copy of the shift stayed behind in the scale
  slot: the seed came back as ``(offset, shape-ish, offset)``.

  Compounding it, the shape approximation was computed on the raw
  ``x``. On offset data the constant squashes
  ``s = log(mean x) - mean(log x)`` towards zero, and since the shape
  grows like ``1 / 12s`` the estimate exploded -- 649 for a true shape
  of 3. The moments are now taken after the shift is removed.

  The consequences were silent wrong answers, not just slow ones. A
  600-point sample from ``Gamma(3, 4)`` shifted up by 10 fitted by MSE
  returned a *negative* shift of -1.35 with a shape of 63.8; another
  sample stopped after 0.03 s at the seed itself, reporting a scale
  equal to the offset. Both now recover the shift, and the fit is also
  4x faster (6.2 s to 1.6 s) because the optimiser no longer has to
  travel back from a nonsense starting point. MLE was unaffected -- it
  found its way regardless -- and non-offset fits are untouched.

  Method of moments still disagrees on offset Gamma, but that is not a
  defect: its solution matches the sample moments *better than the true
  parameters do* (first three moments 10.76 / 116 / 1253 against the
  truth's 10.75 / 115.7 / 1248). The three-parameter moment system with
  a threshold is close to non-identifiable, which is why ``MOM`` is not
  among the offset methods exercised in the test suite.

- **Fitted parameters change for censored *and* truncated data: the
  likelihood was unbounded there (#310).** A censored observation is
  only ever known to lie inside its own truncation window -- it could
  not have been observed otherwise -- so its likelihood numerator has to
  be the probability of that intersection. Every likelihood in the
  package instead used the unconditional form, ``F(x)`` for left
  censoring and ``S(x)`` for right, and divided by a separately
  accumulated window probability. That counts territory the truncation
  has already ruled out, so the contribution exceeds one, and the excess
  grows without limit as the fitted distribution's mass slides out of
  the window.

  The consequence was a silent wrong answer. On 200 left-censored
  LogNormal points with a true ``mu`` of 0 and mild left truncation, the
  fit returned ``mu = -7.81`` with ``neg_ll`` of ``-inf`` and
  ``res.success`` set to ``True``. Across eight distributions, 21 of 48
  censoring/truncation combinations returned a non-finite likelihood,
  reporting parameters such as a Weibull ``alpha`` of 1.3e81 or a Normal
  ``mu`` of -1.77e4. Regression was affected too, and failed more
  quietly: adding left truncation to a working ``WeibullAFT`` fit
  returned an entirely plausible-looking parameter vector whose
  covariate coefficient had collapsed from a true 0.5 to 0.0018.

  Rather than teach each likelihood about truncation, such rows are now
  handed over as *intervals*: a left-censored row truncated at ``tl``
  becomes ``[tl, x]``, a right-censored row truncated at ``tr`` becomes
  ``[x, tr]``. The interval term is already a difference of CDFs, so it
  computes the correct numerator with no change to any likelihood
  function -- which is also why interval-censored data never had the
  bug. One change in ``SurpyvalData`` therefore fixes the parametric,
  regression, Royston-Parmar and mixture likelihoods together.

  Only rows with a *finite* bound on the relevant side are recast, which
  is exactly where the defect lived. Untruncated fits are bit-identical:
  verified over 292 cases spanning ten distributions, seven censoring
  regimes, two sample sizes, weighted and unweighted, and three
  regression fitters, compared as exact float bit patterns. The
  restriction also keeps right censoring on the exact ``log_sf`` path,
  since expressing it as ``1 - F(x)`` loses all precision once ``F(x)``
  rounds to one -- a ``log_sf`` of -49 comes back as ``-inf``, and the
  optimiser does evaluate the likelihood that far from the data.

  The LogNormal case above now fits ``mu = -0.56``, matching the
  maximum of the correctly conditioned likelihood, and all 48
  combinations return finite results. Right censoring combined with
  finite right truncation remains contradictory data and keeps its
  existing warning; what changes is that it now yields the coherent
  conditional ``P(x < X <= tr)`` rather than an unbounded direction.

- **``group_xcnt`` no longer walks every observation in Python.** The
  step that collapses duplicate ``(x, c, t)`` rows accumulated into a
  triple-nested ``defaultdict``, one iteration per observation. That is
  linear but with a very large constant -- around 13 microseconds an
  observation -- which made it the single dominant cost of fitting once
  samples grew: 94% of a 50,000-point Normal fit, and two seconds at
  100,000 points. It is now a sort plus ``np.bincount``. Fitted values
  are bit-identical.

  Group *ordering* is preserved exactly, which matters more than it
  appears: ``xcnt_sort`` runs immediately afterwards and is a *stable*
  sort keyed on ``c``, ``t.min(axis=1)`` and ``x``, so any rows tying on
  all three keep whatever order grouping produced. Rows sharing an ``x``
  and ``c`` with different ``tr`` but equal ``t.min()`` are exactly such
  a tie, and a plain sorted ``np.unique`` would silently reorder them,
  so the original x-major nesting is reproduced instead. Integer counts
  also stay integer (``np.bincount`` returns float64), and ``nan``
  entries keep their own groups as they did under dictionary keying.

  Measured end to end: a Normal fit at n=100,000 goes from 1137 ms to
  125 ms (9.1x), Weibull at n=100,000 from 3890 ms to 926 ms (4.2x),
  and small fits improve too -- Exponential at n=1000 from 5.9 ms to
  2.4 ms. Kaplan-Meier at n=100,000 now takes 140 ms.

  On top of that, grouping is now skipped entirely when there is
  nothing to group. Continuous measurements have distinct values, so
  every row is already its own group in input order and the operation
  is the identity -- but the data handler runs it several times per
  fit regardless. Distinct values in the leading column of ``x`` are
  enough to establish this (they make whole rows distinct whatever
  ``c`` and ``t`` hold), which costs one sort of one column against
  three sorts of the full key. Only tied data -- rounded, discrete, or
  heavily weighted -- takes the grouping path now. Grouping 100,000
  distinct points falls from 74 ms to 1.2 ms, taking the Normal fit
  above to 64 ms, Weibull to 505 ms, and Kaplan-Meier to 63 ms. Repeated
  ``nan`` values deliberately fail the check and fall through to
  grouping, since ``nan != nan`` means they must stay separate.

- **Exact closed-form maximum likelihood for the Exponential, Normal and
  LogNormal, where one exists.** These have analytic MLEs -- the
  Exponential's events-over-exposure ratio, the Normal's mean and
  standard deviation -- so the fit no longer builds an initial guess or
  runs the five-optimiser ladder. Because the closed form is *exact*,
  the result is not merely faster but at least as good: verified that
  its log-likelihood is never worse than the optimiser's, and its
  parameters agree to within the optimiser's own convergence tolerance
  (~1e-8), as do its confidence bounds. At n=1000 a LogNormal fit goes
  from 135 ms to 10 ms, a Normal from 42 ms to 5 ms, and an Exponential
  from 11 ms to 5 ms; Weibull and the rest are untouched.

  The applicability conditions are exact. The Exponential admits right
  censoring and left truncation (which only moves each unit's exposure
  from ``x`` to ``x - tl``), but falls back to the optimiser for left or
  interval censoring and for right truncation, each of which makes the
  score transcendental. The Normal and LogNormal need complete,
  untruncated data: any censoring makes them the Tobit model and any
  truncation introduces a normal-CDF normaliser.
- **Fixed: closed-form fits silently ignored ``lfp``, ``zi``, ``offset``
  and fixed parameters.** The hook that short-circuited to a
  distribution's analytic MLE fired before any of these were checked, so
  ``Uniform.fit(x, lfp=True)`` returned ``p = 1.0`` and
  ``Uniform.fit(x, fixed={"a": 0.0})`` ignored the held value -- in both
  cases without warning. Requests carrying that structure now go to the
  optimiser, which estimates them.
- **Fixed: ``Uniform`` fits had no usable log-likelihood.**
  ``Uniform.fit(x).aic()`` raised ``AttributeError`` and ``cb()`` raised
  for want of a covariance. The log density is now defined directly
  rather than through the generic ``log(hf) - Hf`` identity, which is
  ``nan`` at the upper support edge (where ``sf`` is 0) -- exactly where
  the MLE puts ``b``. ``neg_ll``, ``aic``, ``bic`` and ``aic_c`` are now
  correct. No parameter covariance is offered, deliberately: the Uniform
  MLE is an order statistic sitting on the support edge rather than an
  interior stationary point, so the observed information is not positive
  definite and its inverse carries negative variances; ``cb`` refuses
  rather than returning silent ``nan`` bounds.

- **Tests: a breadth sweep over Turnbull's supported inputs.** Every
  combination of censoring type (observed, left, right, interval, and all
  four mixed), truncation form (none, left, right, both) and hazard
  estimator (Nelson-Aalen, Kaplan-Meier, Fleming-Harrington) is now fitted
  and checked for a converged, valid, monotone survival curve with a
  coherent risk-set ladder, complementing the existing tests that each pin
  one regime. The sweep also pins that the estimator choice is honoured,
  and that Fleming-Harrington coincides with Nelson-Aalen exactly when
  event times are distinct (its tied-event correction being the only
  difference). Building it surfaced #308.
- **Known issue: left censoring combined with left truncation** (#308).
  Turnbull either raises "censoring interval does not intersect its own
  truncation window" on data where the intersection is plainly non-empty,
  or fails to converge and returns a degenerate estimate. A left-censored
  row lives on the ``(-inf, x]`` bound, and the support-window
  intersection added for #273 drops that bound whenever an entry time
  sits above its lower edge, even though the event interval ``(tl, x]``
  is non-empty. Only fits with *both* left-censored observations and
  left truncation are affected. The sweep marks this regime ``xfail``
  (strict), so it will report as soon as it is fixed.

- **Fixed: ``xcnt_to_xrd`` was quadratic in time and memory, and raised
  ``MemoryError`` past roughly 50,000 observations** (#306). The at-risk
  entry count was built as an ``N x K`` comparison matrix
  (observations × distinct times): 20,000 observations needed a 3.2 GB
  intermediate and ~15 s, and 50,000 attempted an 18.6 GiB allocation and
  failed. Because this conversion feeds every nonparametric estimator —
  and the MLE initial guess, which comes from probability plotting — the
  ceiling applied to most of the package: ``Weibull.fit`` on 50,000
  points raised ``MemoryError`` even though the likelihood itself was
  fine. The entry count is now computed in two linear branches: a
  constant when nothing is left truncated (the common case, where the
  matrix was entirely ``True`` and merely recomputed ``n.sum()``), and a
  sorted ``searchsorted`` lookup otherwise. ``side="left"`` counts
  strictly-less-than exactly as the previous ``<`` did, so the
  ``(entry, exit]`` convention from #260 is unchanged, and integer counts
  make the cumulative sum exact — values are bit-identical. A
  ``Weibull.fit`` at n=10,000 goes from 1,836 ms to 182 ms; 200,000
  points now fit in 5.6 s and a 500,000-point Kaplan-Meier in 8.1 s,
  where both previously failed.

v0.17.0 (1 August 2026)
-----------------------

- **Discrete distributions are now structurally separated from the
  continuous catalogue.** A new ``DiscreteParametricFitter`` base class
  (Geometric, Poisson, DiscreteWeibull, NegativeBinomial, Binomial,
  Bernoulli/FixedEventProbability, BetaGeometric, and ``Discretize``
  wrappers) is the single home for what discreteness means when fitting:
  the ``discrete`` trait, the central ``supports_mpp = False`` (each
  class previously set its own flag), and a new clear rejection of
  ``how="MPS"`` — spacings are increments of a continuous CDF and
  repeated integers make them degenerate, so this now raises instead of
  fitting nonsense. MLE, MSE and MOM behaviour is unchanged.
- **``InstantlyOccurs`` and ``NeverOccurs`` are now first-class
  degenerate distributions** in
  ``univariate/parametric/distributions/degenerate.py`` (previously
  partial-API classes tucked into ``parametric/__init__.py``). They gain
  the missing ``df``/``hf``/``qf``/``mean`` methods and serialisation:
  ``to_dict`` stamps the schema and ``surpyval.from_dict`` restores the
  class itself (identity preserved, as the survival-tree leaves
  require). Historical import paths keep working.
- **Simplification: one shared ``fit()`` skeleton for the parametric
  regression families** (#295). PH, AFT, PO and parametric AH carried
  five copy-pasted versions of the same fit pipeline — data prep, the
  #251 param-map offset merge, ``bounds_convert``, optimisation, model
  assembly — which had already drifted (different optimiser ladders;
  only some families setting ``dist_params``/``phi_params``). The
  skeleton now lives once in ``_fit_skeleton.py`` (each family supplies
  its optimiser strategy and covariate-link object), along with a single
  ``LogLinearPhi`` for the ``exp(beta'Z)`` link previously defined
  inline in seven places. Fitted values are bit-identical; each
  family's historical optimiser ladder and serialisation name tags are
  preserved exactly.
- **Simplification: shared hazard identities and information criteria**
  (#297, #298). The six ``sf``/``ff``/``df``/``log_*`` identities
  derived from ``Hf``/``hf`` were repeated in four regression fitters;
  they now live in one ``HazardIdentitiesMixin``. PH and parametric AH
  ``ff`` now use ``-expm1(-H)`` (matching AFT/AL), which is more
  accurate in the deep left tail where ``H`` is tiny; all other values
  are bit-identical. ``neg_ll``/``aic``/``bic``/``aic_c`` were
  duplicated between ``Parametric`` and ``ParametricRegressionModel``;
  one ``InformationCriteriaMixin`` now serves both, preserving each
  class's historical ``aic_c`` parameter-count convention exactly.
- **Simplification: NHPP likelihood data split hoisted** (#296). The
  five-way censoring/interval split of recurrent-event data (the code
  that drifted into #288) was duplicated between the NHPP fitter and
  the proportional-intensity NHPP fitter; it now lives once as
  ``RecurrentEventData.split_for_nhpp_likelihood``. Fitted values are
  bit-identical.
- **Simplification: numerical dedup batch** (#299). The Aalen-Johansen
  ``S(t-)`` incidence weighting (the pattern behind #253/#278) was
  implemented three times — nonparametric ``CompetingRisks``, the
  competing-risks PH ``cif`` and Gray's pooled CIF — and now lives once
  in ``aalen_johansen_iif`` (Gray's pooled CIF is vectorised in the
  process). The Cox at-risk rule (entry-strict ``tl < tau``,
  exit-inclusive ``x >= tau``) is now documented in one
  ``cox_at_risk_mask`` helper used by the exact-tie preparation and the
  Schoenfeld risk-set means, and ``CoxPH.baseline`` replaces its
  O(K·N) Python loop with the same suffix-sum subtraction the Efron
  generator uses (values agree to ~1e-15 relative; pinned by the
  R/lifelines comparison tests). The degradation ``bootstrap_cb`` and
  ``bootstrap_cb_accelerated`` merged into one function (``Z=None``
  selects the plain path; the plain path now also drops non-finite
  refit curves instead of letting them poison the quantiles).
  ``CopulaModel`` serialisation is now round-trippable: ``to_dict``
  stamps the schema version and a new ``from_dict`` (registered with
  ``surpyval.from_dict`` under the ``"copula"`` parameterization)
  rebuilds the model, where previously the dictionary was written in a
  form nothing could read. The CB transform sharing and Cox TVC
  wrapper collapse from #299 are deferred.
- **Simplification: low-risk cleanup batch from the code-simplification
  review** (#295-#299 track the medium-risk remainder). Dead code removed:
  the unused ``surv_sksurv_transformations`` module, ``init_from_bounds``,
  ``_scale`` and ``xcn_to_fsl`` in utils, ``ParametricFitter.
  parameter_transform`` (would have crashed if called), unused
  ``mpp_inv_x_transform`` methods, commented-out blocks, the always-true
  ``hess`` flag in the Cox generator contract (generators now return
  ``(neg_ll, jac_hess)`` pairs), write-only ``fitting_info`` keys, and the
  dead constant-rate methods on the PI-NHPP fitter. Duplication collapsed:
  a shared ``SerialisableMixin`` now provides ``to_json``/``from_json``
  for ~22 model classes; the degradation package imports the delta-method
  helpers from ``recurrent.inference`` instead of carrying verbatim
  copies; ``fit`` and ``_fit_stratified`` share one solve/p-value helper
  in ``cox_ph.py``; ``NonParametricCounting.from_xrd`` is the single home
  of the MCF estimator; ``predict_tvc`` reuses ``_tvc_cumhaz``; ``mean_cb``
  delegates to ``rmst``. Plotting-position heuristics moved to dispatch
  tables. ``ParametricFitter`` gains a default conditional-survival
  ``cs`` (fixing ``AttributeError`` for the discrete distributions);
  three docstring-free identity ``cs`` copies and Weibull's less-stable
  ``log_ff`` override were deleted. Convention fixes: ``validate_tv_coxph``
  no longer double-validates (and now masks the truncation bounds
  alongside the data when covariate rows are dropped);
  ``RecurrentEventData`` iterates statelessly and orders ``items``
  deterministically; stale ``surpyval.alpha`` pointers updated.

- **Removed: the alpha-stage ``SeriesModel``/``ParallelModel``
  reliability-block composition** (#284). Nested composition produced
  incorrect survival functions (``ParallelModel | ParallelModel``
  returned a parallel model; mixed-type composition flattened blocks
  instead of nesting them), and reliability block diagrams are covered
  by the Repyability package. The ``surpyval.experimental`` shim now
  re-exports only the tree/forest models.
- **Fixed: ``NonParametricCounting.mcf_cb`` corrupted bounds for
  off-grid queries** (#285). The out-of-range masks were applied to the
  grid-length bound array before indexing by query position — zeroing
  the whole upper-bound column, wrapping out-of-range queries to the
  last grid value, and raising ``IndexError`` when queries outnumbered
  the two bound rows. Bounds are now selected per query then masked
  (below-min → 0, above-max/negative → NaN, mirroring ``mcf``), and
  two-sided output is now ordered ``[lower, upper]``, consistent with
  the parametric ``cif_cb`` (previously ``[upper, lower]``).
- **Fixed: ``CoxLewis`` constrained the log-intensity intercept to be
  non-negative** (#286), silently pinning fits at ``alpha = 0`` for any
  process with a baseline rate below one event per time unit. The
  intercept is now unbounded; a simulated ``(alpha, beta) = (-1, 0.05)``
  process is recovered to ``(-1.006, 0.050)``.
- **Fixed: recurrent-fitter batch** (#288). A typo (``x[:, 0]`` for
  ``x_prev[:, 0]``) cancelled the observed-event exposure term for 2-D
  event input without interval rows — degenerate ``[t, t]`` pairs now
  fit identically to 1-D input. The dead (and would-be-wrong) Cox-Lewis
  MCF correction in the simulator was deleted. The proportional-
  intensity HPP/NHPP fitters now honour a user-supplied ``init``
  (previously silently overwritten) and validate its length.
- **Fixed: round-2 follow-ups** (#289). MPS tie densities are evaluated
  only at genuinely tied points (untied points contributed
  ``0 * log(0) = NaN`` where a clean infinite penalty was intended);
  the additive-hazards kernel bandwidth falls back to the time scale
  when event times are (nearly) coincident instead of returning Dirac
  spikes; and ``Beta4.hf`` is 0 below and ``inf`` at/above the support
  instead of NaN above it.

- **Fixed: Cox residuals, ``check_ph`` and robust standard errors now
  apply the Efron tie correction for Efron fits** (#279). All residuals
  used plain Breslow risk-set means and increments regardless of the tie
  method, so heavily tied Efron fits (the ``fit_from_df`` default)
  disagreed with R/lifelines — ``check_ph`` km statistic 0.36 vs 0.72,
  robust SEs ~20% small. Schoenfeld residuals and ``check_ph`` (km,
  identity, log transforms) now match lifelines to 6+ figures under
  heavy ties; martingale and score residual sums vanish at the MLE for
  both tie methods; dfbeta correlates 0.999 with exact leave-one-out
  influence. The ``"rank"`` transform now uses average ranks for ties
  (R's ``cox.zph`` convention; lifelines' cumulative-count variant is
  nonstandard).
- **Fixed: the concordance index credited 0.5 to a discordant
  event/censored pair tied in time** (#276). Harrell's C treats the
  censored subject as having outlived the tied event, so the pair is
  fully comparable: 1/0.5/0 by score order. Tie-heavy data was biased
  toward 0.5; results now match lifelines up to the (documented)
  both-events-tied-time convention difference.
- **Fixed: Lin-Ying additive-hazards ``hf``/``df`` added the baseline
  *jump* to a hazard *rate*** (#277) — dimensionally incoherent, and as
  n grows the baseline vanished entirely (``hf -> beta'Z``). The
  baseline rate is now a kernel-smoothed (Ramlau-Hansen, Epanechnikov)
  estimate from the corrected cumulative-baseline increments, with a
  ``bandwidth`` argument. ``phi()`` on additive-hazards models now
  raises a clear ``NotImplementedError`` (the covariate effect is
  additive, not a multiplier) instead of an ``AttributeError``.
- **Fixed: competing-risks CIFs could exceed 1 with the default
  Nelson-Aalen method** (#278). The Aalen-Johansen increment paired the
  discrete hazard ``d/r`` with the exponential survival ``exp(-H)``;
  only the product-limit survival satisfies the telescoping identity,
  so total incidence reached 1.22-1.31 in small samples. Increments now
  always use the product-limit ``S(t-)``; the reported ``sf`` keeps the
  requested estimator.
- **Fixed: distribution edge cases** (#280). LogLogistic: ``sf``/``ff``
  are defined at ``x = 0`` (previously ``ZeroDivisionError``/NaN) and
  ``log_sf``/``log_ff`` use a ``logaddexp`` form that no longer
  overflows to ``-inf`` for large ``alpha**beta``. Beta4: ``df``/``hf``
  are 0 outside the support instead of arbitrary/negative/NaN values.
  Rayleigh, Gamma and Exponential custom probability-plotting paths now
  forward truncation bounds (previously silently dropped), and Rayleigh
  masks plotting positions at F = 1 (the ECDF heuristic returned NaN
  parameters). Uniform's closed-form MLE rejects interval-censored data
  with a clear error instead of a cryptic ``IndexError``.
- **Fixed: ``xrd_to_xcnt`` silently corrupted late-entry data** (#281).
  A risk set that grows between observation times (left truncation)
  cannot be represented in xcnt output; the ``np.abs`` of the risk-set
  differences masked the increase and returned a different study. It
  now raises an informative ``ValueError``.
- **Fixed: container and robustness batch** (#282). ``SurpyvalData``:
  scalar indexing on interval-censored data no longer flattens the
  interval row (IndexError), slicing carries covariates ``Z`` through,
  and ``to_xrd`` caches per estimator instead of returning the first
  call's result for every later estimator. Nonparametric models: scalar
  ``hf``/``df`` return the step's hazard increment instead of always
  NaN, and confidence bounds fall back to the point estimate when no
  point on the curve has a finite variance (single-observation fits
  returned NaN bounds). ``check_ph`` no longer emits a spurious
  "ignoring left truncated values" warning for models fit with a
  constant entry column (``tl = 0``), and the stale pre-#260
  ``xcnt_to_xrd`` docstring example was updated.
- **Fixed: the MPS estimator returned wrong parameters for censored,
  tied, truncated, and offset-truncated data** (#268). Four defects: the
  censored/ties block was divided by a different count than the
  spacings, making the estimator inconsistent even without truncation
  (integer-tied Weibull data fit as ``(13.7, 1.52)`` vs the true
  ``(10, 2)``); censored survivor/CDF terms were not conditioned on the
  truncation window (truncated + censored fits biased to
  ``(11.7, 4.36)``); offset fits passed unshifted truncation bounds to
  the shifted distribution (objective infinite at the true parameters);
  and interval-censored input crashed deep in ``np.hstack`` instead of a
  clear validation error. The objective is now the Cheng-Amin sum form
  (spacings + tie densities + conditional censored terms in one sum),
  bounds are shifted with the data (clamped at the support), and
  interval data raises an informative ``ValueError``. All four scenarios
  now track MLE to within ~2%.
- **Fixed: censored/truncated Gamma and Beta fits had a silently corrupted
  Wald covariance** (#270). The autograd shims for the incomplete
  gamma/beta functions stripped the derivative trace in their
  shape-parameter VJPs, zeroing every second-derivative contribution
  through a shape parameter: the stored covariance was wrong (12x the
  true sampling variance in one repro) and not even symmetric, corrupting
  ``param_cb``, ``cb``, plot bands and the serialised covariance while
  the point estimates were fine. The shape derivatives are now traced
  primitives with numerical second-derivative VJPs, so autograd Hessians
  match the true observed information (verified against numerical
  differentiation to ~1e-6 for censored Gamma and Beta, including offset
  fits); ``mle`` additionally validates Hessian symmetry and falls back
  to a numerical Hessian if a corrupted one ever reappears.
- **Fixed: Turnbull excluded the right endpoint of interval- and
  left-censored observations from their support** (#272). An interval
  ``(l, r]`` whose right endpoint coincided with an exactly observed event
  time was forbidden from having failed at ``r`` (and a left-censored
  observation from having failed at its own bound), pushing its mass onto
  earlier atoms — ``sf`` between the atoms was 0.45 where the (l, r]
  NPMLE (Turnbull 1976, lifelines, icenReg) gives 0.83. Supports now
  include the atom at the right endpoint, matching the (entry, exit]
  convention adopted in #260.
- **Fixed: Turnbull under truncation — variance ladder, support windows,
  and degenerate-interval inputs** (#273). (1) The truncated variance
  ladder redistributed right-censored mass as fractional later events and
  kept censored items at risk via conditional tail probabilities — the
  anti-conservative mechanism #260 removed for untruncated data — and at
  the last event produced huge *negative* Greenwood increments that
  passed the finiteness guard. It now uses observed counts (events at
  exact atoms, censored items leave at censoring), reducing exactly to
  the delayed-entry Kaplan-Meier Greenwood ladder for exact +
  right-censored data. (2) Each observation's support is now intersected
  with its own truncation window: mass can no longer be redistributed to
  times where the observed event provably cannot be, which previously
  drove the EM to a degenerate all-zero fixed point on valid
  left-censored + delayed-entry data — including the original #203
  reproduction, which now converges to a healthy estimate. An empty
  intersection raises an informative ``ValueError``. (3) The KM-reducible
  variance branch now recognises exact + right-censored data expressed as
  degenerate intervals (``xl == xr`` / ``xr = inf``), which previously
  fell back to the anti-conservative expected-count ladder.
- **Fixed: LFP fits with left truncation maximised an unbounded likelihood
  and returned degenerate parameters with optimiser success** (#269). The
  truncation normaliser used ``(p - f0) * (1 - F0(tl))`` — dropping the
  never-failing mass from the survival at entry — instead of the mixture
  survival ``1 - f0 - (p - f0) * F0(tl)``. ``Weibull.fit(x, c, tl=...,
  lfp=True)`` returned ``alpha ~ 1e-42`` on healthy data; it now recovers
  the true parameters. Finite-bound windows (interval censoring, double
  truncation) are algebraically unchanged.
- **Fixed: parametric PH ``random()`` sampled the wrong distribution and
  crashed with two or more covariates** (#271). The sampler inverted
  ``qf(U ** phi)`` where PH requires ``qf(1 - U ** (1 / phi))``, so every
  draw with a non-zero covariate effect came from the wrong distribution
  (empirical SF 0.67 vs model 0.89 in the repro); the covariate broadcast
  also raised ``ValueError`` for multi-covariate models. Draws now
  reproduce the model's own ``sf`` and the returned covariates have shape
  ``(size, p)``.
- **Fixed: Royston-Parmar silently returned NaN models** (#274). The BFGS
  polish replaced the finite Nelder-Mead result even when it diverged
  (e.g. on doubly-truncated data); it is now kept only when finite and
  better, and a non-finite final likelihood raises. Quantile knot
  placement over too-few or tied event times produced coincident knots
  and an all-NaN model with no warning; ``fit`` now validates that the
  data contain at least ``df + 1`` distinct event times and that knots
  are distinct, raising an informative ``ValueError``.
- **Fixed: numeric MOM fits stopped far from the moment-matching
  solution** (#275). The optimiser ran with ``tol=1e-1`` and no
  convergence check, so ``how="MOM"`` with ``offset=True`` or ``fixed``
  returned e.g. ``beta ~ 3-4`` for true ``beta = 2`` silently. The path
  now optimises tightly, polishes with Nelder-Mead when needed, and warns
  if the sample moments remain unmatched.
- **Fixed: Cox delayed-entry / start-stop (TVC) fits had corrupted scores and
  Hessians whenever any covariate value was negative** (#250). The
  left-truncation risk-set adjustment was forward-filled with
  ``np.minimum.accumulate`` — valid for the scalar (positive, non-increasing)
  sum but wrong for the signed Z-weighted score and information sums, which it
  clamped to a stale running minimum. The optimiser could "converge" to a
  spurious zero of the corrupted score (wrong coefficients with no warning),
  and even rescued fits carried garbage standard errors, p-values, ``check_ph``
  and cluster-robust covariance. The adjustment is now an exact suffix-sum
  gather (``not_yet_entered``), valid for signed quantities; the analytic score
  and information now match numerical differentiation of the partial
  log-likelihood under delayed entry. All-positive covariates were unaffected.
- **Fixed: parametric PH ``fixed={"beta_0": ...}`` silently pinned the first
  distribution parameter instead of the covariate coefficient** (#251). The
  covariate parameter map was merged without the distribution-parameter
  offset (AFT/PO/AH were unaffected), so ``WeibullPH.fit(x, Z,
  fixed={"beta_0": v})`` fixed ``alpha`` to ``v`` and left ``beta_0`` free,
  corrupting the fit and its covariance. The map is now offset like the other
  regression families.
- **Fixed: nonparametric competing-risks CIFs were systematically
  underestimated** (#253). The Aalen-Johansen incidence increment weighted
  each cause-specific hazard by the survival *after* the jump, ``S(t)``,
  instead of ``S(t-)`` — with one cause and no censoring the CIF topped out
  at ~0.72 instead of 1. Cause-specific CIFs now sum exactly to ``1 - S``
  (Kaplan-Meier weighting). The same correction applies to the Cox-path
  cause-specific ``cif``. Also fixed: query times before the first observed
  event wrapped to the *last* step value (``sf(0.1)`` on data starting at 1
  returned the final survival instead of 1) in both the nonparametric and
  Cox-path predictors, and ``CompetingRisks.fit_from_df`` stored the source
  DataFrame as ``model.df``, shadowing the density method — it is now
  ``model.source_df``.
- **Fixed: likelihood-ratio confidence bounds ignored user-fixed
  parameters** (#255). Profiling silently re-freed a parameter fixed at fit
  time, letting the profile drop below the fitted negative log-likelihood and
  inflating the interval several-fold (``Weibull.fit(x, fixed={"beta": 5})``
  gave an ``alpha`` LR interval ~5x the Wald width). Fixed parameters now stay
  pinned during both the parameter profile and the function-band constrained
  search, and requesting an LR bound *on* a fixed parameter raises a clear
  ``ValueError``.
- **Fixed: MixtureModel likelihood and EM corrections** (#254). Counts from
  grouped/tied data were applied as per-component likelihood *powers* before
  mixing (``sum w_i f_i^n != (sum w_i f_i)^n``), so any tied data (e.g.
  rounded measurements) silently skewed the mixing weights — a true 50/50
  Weibull mixture fit as 14/86. Counts now multiply the mixture
  log-likelihood; the mixing-weight update is count-weighted; the M-step now
  minimises the proper EM Q-function (responsibilities times component
  log-likelihoods) instead of an ad-hoc responsibilities-as-weights
  objective; and the interval-censored contribution was ``F(l) - F(r)``
  (negative) — now ``F(r) - F(l)``. Truncation (``tl``/``tr``/``t``) was
  accepted but silently ignored; truncated data is now fitted by direct
  maximum likelihood on the truncation-corrected observed likelihood
  (the window couples the components, so label-based EM does not apply).
  Also fixed: ``xl``/``xr``-only input crashed on ``len(None)``, and ``df``
  crashed on integer input.
- **Fixed: LFP / zero-inflated / offset parametric model conventions made
  mutually consistent** (#256). ``df``/``hf`` for combined LFP+ZI models used
  ``(1 - f0) * p`` where ``sf``/``ff`` and the likelihood use ``(p - f0)`` —
  the density did not integrate to the failure probability. ``mean()`` and
  ``moment()`` ignored ``f0`` entirely. ``qf``/``random`` placed the
  zero-inflation mass at the offset ``gamma`` while ``df``/``ff`` place it at
  0, so ``qf`` did not invert ``ff``. Offset models returned ``ff < 0`` /
  ``sf > 1`` / NaNs below ``gamma`` — now clamped to the boundary values.
  ``cb`` returned NaN where the point estimate sits on the boundary
  (``sf == 1``, e.g. ``t <= gamma``) — now the boundary. ``random()`` crashed
  for LFP models when the binomial draw produced zero failures. ``aic_c``
  penalised a different parameter count than ``aic``. The numerically stable
  left-censored likelihood branch was unreachable (inverted ``f0`` check).
- **Fixed: distribution-level defects** (#257). ``LogNormal.fit`` crashed for
  any data with geometric mean < 1 (the location ``mu`` was wrongly bounded
  positive). ``Bernoulli.fit`` was broken for essentially every input (it
  broadcast ``x`` against the literal ``[0, 1]`` and mishandled ``n=None``).
  Offset MPP fits with ``rr="x"`` mis-inverted the regression for
  Exponential and Gamma (silently wrong ``lambda``/``gamma``); the Gamma
  offset MPP also seeded its shape search from unshifted-data moments and
  now multi-starts it. Gamma's censored non-offset ``rr="x"`` crashed on a
  length mismatch. ``ExpoWeibull.sf``/``Hf``/``log_sf`` underflowed to
  0/inf/-inf in the (reachable) right tail — rewritten in a
  cancellation-free ``expm1``/``log1p`` form. Probability-plot y-axis
  inverse transforms were not inverses for Exponential, GumbelLEV and Beta
  (silently mislabelled plot axes). ``Logistic`` log-functions overflowed
  in the deep tail (now ``logaddexp``). ``ExactEventTime.fit`` without both
  censoring sides now raises an informative error.
- **Fixed: formula fits with a categorical covariate were non-identified —
  categoricals are now reference-level coded** (#252). ``fit_from_df(...,
  formula="age + sex")`` used to expand ``sex`` into a *full one-hot*
  (``sex[F]``, ``sex[M]``) whose columns sum to a constant — exactly
  collinear with the baseline distribution's scale (or the Cox baseline), so
  the likelihood was flat along a ridge and the reported coefficients and
  standard errors were optimizer-path noise (predictions were unaffected,
  which is why it went unseen). Formulas are now materialised with their
  implicit intercept, giving categoricals standard treatment coding, and the
  intercept column is dropped (the baseline provides it).

  **Migration note:** feature names and coefficient meanings change for
  formula fits with categoricals — ``['sex[F]', 'sex[M]']`` becomes
  ``['sex[T.M]']``, and the coefficient is the log-hazard-ratio (or
  equivalent) of that level versus the reference (first) level, matching R,
  lifelines and statsmodels. Predictions from refitted models are unchanged.
  An explicit ``"0 + ..."`` formula opts back into full-rank coding. This
  also fixes the ``LinAlgError`` crash in Buckley-James formula fits with
  categoricals.
- **Fixed: regression serialisation and robustness batch** (#261).
  Buckley-James and Lin-Ying additive-hazards models now persist their
  formula encoder state (the #244 treatment), so restored models predict
  from DataFrames with transforms/categoricals; repeated save/load cycles of
  a parametric regression model no longer silently drop the stored
  covariance; ``ParametricRegressionModel.random`` (broken on every path —
  it ignored ``Z``) now dispatches to the fitter's covariate-aware sampler;
  the ``AcceleratedLife`` fitter is no longer stateful across fits, keeps
  user-fixed parameters in ``model.fixed`` (SEs were reported for
  constrained parameters), and accepts 1-D stress vectors; ``WeibullPH.fit``
  accepts plain-list covariates; ``fit(init=<ndarray>)`` no longer crashes;
  deserialised univariate models support ``bic``/``aic_c``/re-serialisation
  and carry their support interval; interval/left-censored observations
  below the distribution's support are rejected at validation instead of
  producing a NaN likelihood and a silent initial-guess "fit" (whose
  reported likelihood now matches its returned parameters); and invalid
  ``cb``/``param_cb`` arguments raise ``ValueError`` instead of
  ``UnboundLocalError``.
- **Fixed: TVC prediction and alignment** (#259). Predicting along a
  covariate schedule treated intervals as ``[xl, xr)``, so a baseline-hazard
  jump exactly at a covariate-change time was weighted by the *new*
  covariate while the fitted likelihood uses ``(xl, xr]`` — predictions now
  match the fit, and a query time returns the same value regardless of the
  other query points. Cluster-robust standard errors on start-stop (TVC)
  fits now permute user-supplied per-row cluster labels into the internal
  row order (previously silently misassigned unless the input was already
  sorted), and default to clustering by subject. An exactly singular
  information matrix now degrades to the pseudo-inverse/NaN path instead of
  crashing.
- **Fixed: AFT time-varying-covariate fits now refuse delayed entry and
  observation gaps instead of silently dropping the missing exposure**
  (#258). The accumulated accelerated age ``psi(T)`` integrates the covariate
  path from time 0; a subject entering observation late (or with gaps) has
  unobserved covariates over the uncovered window, and the likelihood
  previously treated that time as contributing zero ageing — shifting every
  subject's window by +5 returned bit-identical parameters. Correct
  conditioning would require the unobserved pre-entry covariate path, so
  rather than guess it the fit raises an informative error pointing to Cox
  TVC (``CoxPH.fit_tvc``), which handles delayed entry and gaps exactly.
- **Fixed: frailty models handle the ``theta -> 0`` (no-frailty) limit**
  (#262). A frailty variance that underflows to zero — frailty-free data, or
  a restored model — gave NaN marginal predictions (division by ``theta``)
  and a NaN/crashing Wald interval; the marginal now takes the well-defined
  proportional-hazards limit ``eta * H0``, and a boundary estimate returns a
  zero-width interval instead of dividing by zero.
- **Fixed: Turnbull confidence intervals and the delayed-entry risk-set
  convention** (#260). On plain right-censored data (where Turnbull reduces
  exactly to Kaplan-Meier) the variance was computed from the EM's
  *expected*-count ladder, which redistributes censored mass as fractional
  later events and silently understated it — confidence intervals were
  anti-conservative (e.g. Var(H) 0.47 vs the correct Greenwood 0.63). The
  variance now uses the observed-count ladder in that regime and matches
  Kaplan-Meier's Greenwood intervals exactly; genuinely interval-censored
  data keeps the expected-count approximation (use ``bootstrap_cb`` for
  calibrated intervals there).

  **Convention change:** delayed-entry risk sets now follow the standard
  ``(entry, exit]`` convention (R ``survival`` / lifelines): a subject
  entering observation exactly at an event time is *not* at risk for that
  event. Kaplan-Meier/Nelson-Aalen previously counted it, disagreeing with
  Turnbull's NPMLE on identical data; the two now agree. Fits only change
  where an entry time exactly ties an event time. Consistently, a value at
  exactly its own left-truncation time (a zero-length observation window)
  is now rejected at validation instead of silently distorting the
  estimate, and ``Turnbull.fit(..., max_iter=0)`` raises instead of
  crashing. The truncated-fit degeneracy detector now inspects only the
  identifiable region, so partial collapses are reported as degenerate
  rather than as generic non-convergence.
- **Changed: the proportional-hazards test now uses the standard
  Grambsch-Therneau forms** (#262). The per-covariate statistic is
  ``d (Vu)_j^2 / (Sgc2 V_jj)`` with ``V`` the inverse information — the form
  used by R's ``cox.zph`` and lifelines — replacing the previous
  information-diagonal variant (both are valid chi-square screens, but they
  weight cross-covariate information differently, so surpyval could flag a
  different covariate than R/lifelines on the same data). The ``"km"`` time
  transform is now the true ``1 - KM(t)`` fit on the full data (censoring
  included) rather than the censoring-blind ECDF of event times.
  ``check_ph`` now matches lifelines to numerical precision (verified
  against lifelines 0.30.3); reported per-covariate statistics change for
  multi-covariate models. The global test was already the standard form and
  is unchanged.
- **Royston-Parmar flexible parametric models.** ``RoystonParmar.fit(x, c=...,
  df=..., scale=...)`` fits a flexible parametric survival model that replaces
  the straight log-cumulative-hazard-vs-log-time line of a Weibull with a
  restricted cubic spline, giving a smooth, fully parametric baseline of
  arbitrary shape -- flexible like a Cox baseline but extrapolable like a
  parametric one. Three link scales: ``"hazard"`` (proportional hazards; ``df``
  = 1 is a Weibull), ``"odds"`` (proportional odds), and ``"normal"`` (probit;
  ``df`` = 1 is a log-normal). Knots are placed at quantiles of the event
  log-times by default (or supplied explicitly), and beyond the boundary knots
  the spline is linear, so the model extrapolates with a Weibull-like tail --
  which pairs naturally with the restricted-mean survival time added in 0.16.
  The fitted ``RoystonParmarModel`` exposes ``sf`` / ``ff`` / ``hf`` / ``Hf`` /
  ``df`` / ``qf`` / ``random`` / ``mean``, a linear-predictor confidence band
  (``cb``), ``aic`` / ``bic`` for choosing ``df``, and ``to_dict`` /
  ``from_dict``. The likelihood supports the full arbitrary
  censoring/truncation surface -- observed, right-, left- and interval-censored
  observations (pass ``xl`` / ``xr`` or 2-element ``x`` rows), with left- and/or
  right-truncation (``tl`` / ``tr`` / ``t``) and observation weights (``n``).
- **Shared-frailty proportional-hazards models (Gamma frailty).** A new
  ``Frailty(distribution)`` factory (with pre-built ``WeibullFrailty``,
  ``ExponentialFrailty``, ``LogNormalFrailty``, ``GammaFrailty`` instances) fits
  a proportional-hazards model with a random hazard multiplier shared within a
  group -- ``h(t | Z, u) = u h0(t) exp(beta'Z)``, ``u`` drawn once per group
  from a Gamma of mean 1 and variance ``theta``. ``.fit(x, Z, c, groups=...)``
  and ``.fit_from_df(..., group_col=...)`` maximise the closed-form marginal
  likelihood (the Gamma frailty integrates out per group), so it captures
  unobserved between-group heterogeneity and the within-group correlation it
  induces -- the conditional/random-effects complement to the cluster-robust
  standard errors added in 0.16. The fitted ``FrailtyModel`` reports the frailty
  variance ``theta`` (with a Wald CI), the per-group posterior (empirical-Bayes)
  frailties, and predicts either **marginally** (population-averaged, the
  default -- ``S = (1 + theta e^{beta'Z} H0)^{-1/theta}``) or **conditionally**
  on an observed group or a supplied frailty value via ``sf(x, Z, group=...)`` /
  ``sf(x, Z, frailty=...)``. Omitting ``Z`` gives a pure random-effects survival
  model. Serialises with ``to_dict`` / ``from_dict``. Gamma frailty only for
  now; log-normal, Cox, and nested/hierarchical frailty are planned.
- **Fixed: formula-fit regression models now round-trip through serialisation**
  (#244). A regression model fit with ``fit_from_df(..., formula=...)`` using a
  categorical term dropped its design-matrix transformer on ``to_dict`` /
  ``from_dict``, so a restored model failed to evaluate from raw covariates
  (``['sex[F]', 'sex[M]'] not in dataframe columns``). ``to_dict`` now persists
  the categorical factor levels and numeric column names, and ``from_dict``
  rebuilds an equivalent ``formulaic`` model spec, so a restored model expands
  raw covariates identically to the original -- for the parametric families
  (PH/AFT/PO/AH) and Cox. Data-dependent transforms (``scale()`` / ``center()``)
  keep fitted statistics that cannot be restored from levels, so serialising
  such a formula now raises early at ``to_dict`` rather than round-tripping to a
  silently wrong encoding.
- **Likelihood-ratio confidence bounds on model functions.** ``cb`` gains the
  same ``method`` argument: ``method="lr"`` returns a profile-likelihood band
  on ``sf`` / ``ff`` / ``Hf`` / ``hf`` / ``df``. At each time the bound is the
  extreme value of the function over the parameter confidence region
  :math:`\{\theta : 2[\text{nll}(\theta) - \text{nll}_{\hat{}}] \le \chi^2_1\}`,
  found by constrained optimisation with a warm-started sweep over the time
  grid. Like the parameter version it is transformation-invariant and better
  behaved in small samples than the Wald/delta band, needs the original data,
  and does not yet cover offset / LFP / ZI models.
- **Likelihood-ratio confidence bounds on parameters.** A fitted parametric
  model's ``param_cb`` gains a ``method`` argument: ``method="wald"`` (the
  existing default) or ``method="lr"`` for a profile-likelihood
  (likelihood-ratio) bound. The interval is the set of parameter values whose
  profile deviance stays below the :math:`\chi^2_1` critical value, with the
  remaining parameters re-optimised at each candidate. Unlike the Wald bound it
  is transformation-invariant, respects the parameter's support boundary, and
  need not be symmetric about the estimate -- usually better small-sample
  coverage, and the reliability-engineering default. It needs the original
  data (a deserialised model raises, directing you to ``method="wald"``);
  offset / LFP / ZI models are not yet supported.

v0.16.0 (22 Jul 2026)
---------------------

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
- **Prediction-validation metrics** (#212). A new ``surpyval.metrics`` module
  scores a *predicted survival function* against right-censored outcomes with
  inverse-probability-of-censoring weighting: ``brier_score`` /
  ``integrated_brier_score`` (the time-dependent Brier score of Graf et al.
  1999 and its integral -- calibration and discrimination together, lower is
  better) and ``auc_td`` (Uno's 2007 cumulative/dynamic time-dependent AUC --
  discrimination as a function of the horizon). All are model-agnostic; the
  ``survival_probability`` helper builds the required survival matrix from any
  fitted model exposing ``sf(x, Z)`` (the parametric regression families,
  ``CoxPH`` and the ``beta.ml`` forest), giving the ML-flavoured workflow its
  first proper validation-and-comparison story. Validated against known
  answers: without censoring the Brier score is exactly the mean squared error;
  a well-specified model beats the marginal Kaplan-Meier reference (and a
  constant predictor is worse); and the AUC is ~1 for a near-perfect ordering
  and ~0.5 for a random one.

Correctness
~~~~~~~~~~~

- **Turnbull EM under truncation** (#203). Three statistical defects in the
  truncated Turnbull NPMLE are fixed. (1) The EM now iterates with the
  Kaplan-Meier self-consistency update (``p`` proportional to the expected
  counts ``d``), the canonical M-step; the ``Fleming-Harrington`` /
  ``Nelson-Aalen`` inner estimators set ``R = exp(-H)``, which violates that
  fixed point and left even healthy truncated fits reporting tol-level
  non-convergence -- they now converge, and the requested hazard-form
  estimator is applied to the *converged* ladder. (2) The expected counts are
  confined to the identifiable support each iteration, stopping the ghost
  step from migrating mass below every entry window. (3) The convergence
  check is no longer NaN-blind: a non-finite update or a total mass collapse
  is detected as a *degenerate, non-identifiable* fixed point and reported
  with an explicit warning and a ``degenerate`` flag on the model, instead of
  a silent all-zero survival curve. Untruncated fits are unchanged. Validated:
  the issue's degenerate reproduction is now flagged and warned; a
  left-truncated sample recovers ``S(median)`` to within 0.04 with all three
  inner estimators; and the documented untruncated example is byte-for-byte
  identical.

Degradation
~~~~~~~~~~~

- **Destructive degradation modelling** (#153). New
  ``surpyval.degradation.DestructiveDegradation`` for tests whose measurement
  destroys the specimen, so each unit yields a single ``(time, degradation)``
  point (material/adhesive strength, breakdown voltage, ...). With no per-unit
  paths to fit, the population degradation distribution is modelled directly as
  a location-scale regression on a time transform,
  ``Y | t ~ dist(loc = β₀ + β₁·φ(t), σ)`` (``LogNormal`` or ``Normal``;
  ``φ`` = linear / log / sqrt / reciprocal, or ``transform="best"`` by AICc),
  and the lifetime distribution is induced by crossing the failure threshold
  (``sf`` / ``ff`` / ``Hf`` / ``df``), with the increasing (wear) vs decreasing
  (strength-loss) direction inferred automatically. Censored measurements (a
  strength below the test floor, a specimen that did not break) are handled
  through the ordinary ``c`` convention; ``cb`` gives bootstrap bounds and the
  model round-trips through ``to_dict`` / ``from_dict``. This completes the
  degradation half of #153 alongside the stochastic-process models.

Regression
~~~~~~~~~~

- **Time-varying-covariate fitting for accelerated failure time** (#150).
  ``WeibullAFT`` (and every ``AFT(dist)``) gains ``fit_tvc`` /
  ``fit_tvc_timeline`` and the DataFrame variants, taking the same start-stop /
  timeline input (``i`` / ``xl`` / ``xr`` / ``c``) as the other families.
  Because AFT rescales the time axis, a subject's likelihood depends on its
  *accumulated accelerated age* ``ψ = Σ exp(β'z)(b − a)`` across intervals and
  does not factorise into independent left-truncated rows the way the
  proportional/additive-hazards families do, so it is fit with a dedicated
  accumulated-age likelihood (a within-subject scan each optimiser step) rather
  than the reshape-and-refit used for PH/AH. The shared MLE code is untouched:
  the fit binds the custom likelihood onto its own result object, so confidence
  bounds (a numerical Hessian of that likelihood) are correct, and information
  criteria are reported on the subject count rather than the episode rows. This
  closes the last open part of #150; with #170's evaluation side, AFT now has
  full time-varying-covariate support.
- **Evaluate a fitted regression model along a time-varying covariate path**
  (#170). A fitted ``WeibullPH`` (any ``PH(dist)``), ``WeibullAH`` (any
  ``AH(dist)``) or ``WeibullAFT`` (any ``AFT(dist)``) gains ``sf_tvc`` (and
  ``Hf_tvc``): given a piecewise-constant covariate schedule ``Z(t)`` it
  returns the resulting survival ``S(t)``, with an optional ``given=`` age for
  conditional survival. For proportional and additive hazards the cumulative
  hazard is additive over disjoint intervals, so the survival along a step path
  is the exact sum of the per-segment increments; for accelerated failure time
  the path instead accumulates an *accelerated age*
  ``ψ(x) = Σ exp(β'z)·(b − a)`` fed once through the baseline. Either way it
  reduces to ordinary ``sf`` for a constant covariate. The covariate path is
  described by a new
  ``StepSchedule``, built structurally (``from_changepoints`` / ``from_intervals``
  / ``cyclic`` for duty cycles) or from a step-valued expression string in
  ``t`` (``from_expression``, e.g. ``"0.9 if t % 24 < 8 else 0.3"`` or
  ``"0.3 * 2 ** floor(t / 1000)"``). Expressions are *proved* piecewise-constant
  from their syntax tree before evaluation -- ``t`` may reach the value only
  through a quantizer (``floor`` / ``ceil`` / ``//``) or a comparison -- so a
  continuously-varying covariate (``0.3 + 1e-4 * t``, ``sin(t)``) is rejected
  with ``StepValuedError`` rather than silently returning a wrong answer.
  ``sf_tvc`` may be given ``(xl, Z)`` arrays directly or a ``StepSchedule``.
  The semi-parametric ``CoxPH`` gains the same ``sf_tvc`` / ``Hf_tvc`` and
  ``StepSchedule`` convention (summing the fitted baseline-hazard jumps along
  the path); the existing interval-oriented ``predict_tvc`` is unchanged and
  ``sf_tvc`` agrees with it exactly. Only proportional odds does not yet
  expose a time-varying-covariate evaluation and raises.
- **Time-varying covariates for the parametric PH and additive-hazards
  families** (#150). ``WeibullPH`` (and every ``PH(dist)``) and ``WeibullAH``
  (every ``AH(dist)``) gain ``fit_tvc`` / ``fit_tvc_timeline`` and the
  DataFrame variants, taking the same start-stop / timeline input as
  ``CoxPH.fit_tvc`` (``i`` / ``xl`` / ``xr`` / ``c``, surpyval's censoring
  convention). For these families the cumulative hazard is additive over time
  intervals, so a time-varying-covariate subject factorises exactly into one
  left-truncated observation per constant-covariate interval; the fitter simply
  reshapes the data and reuses the ordinary parametric MLE, giving the same fit
  as the equivalent non-time-varying data. Accelerated failure time and
  proportional odds do not compose this way (they need an accumulated
  accelerated age / have no additive structure), so they do not expose
  ``fit_tvc``.
- **Timeline (xicnt-style) input for time-varying-covariate Cox.**
  ``CoxPH.fit_tvc_timeline`` / ``fit_tvc_timeline_from_df`` accept a covariate
  *timeline* -- one row per covariate change per subject (``i``, ``x``, ``Z``,
  ``c``) with the terminal event / censoring on the subject's last row -- as an
  alternative to writing explicit ``(xl, xr]`` intervals for ``fit_tvc``. Each
  covariate value holds from its time until the subject's next row, the first
  time is the (delayed-)entry time and the last is the exit; the timeline is
  expanded to start-stop intervals and fitted identically, so it gives the same
  fit as the equivalent ``fit_tvc`` data.
- **Time-varying-covariate Cox input harmonised to the surpyval convention.**
  The start-stop interface (``CoxPH.fit_tvc`` / ``fit_tvc_from_df`` /
  ``predict_tvc`` and ``handle_tvc``) is renamed to match surpyval's
  vocabulary: the subject id is ``i`` (was ``ident``), the interval bounds are
  ``xl`` / ``xr`` (were ``start`` / ``stop``), and the status is ``c`` (was
  ``event``). ``c`` now follows the standard surpyval censoring convention --
  ``0`` = event at ``xr``, ``1`` = right-censored -- which is the *inverse* of
  the old ``event`` flag (``event=1`` -> ``c=0``). The DataFrame entry point's
  columns are named ``xl_col`` / ``xr_col`` / ``c_col`` accordingly. Positional
  calls are unaffected; keyword calls and the ``event`` values need updating.
- **Accelerated Life with an Exponential distribution now fits.**
  ``AcceleratedLife(Exponential, life_model).fit(...)`` raised
  ``KeyError: 'lambda'`` because the life-parameter map named the Exponential's
  parameter ``"lambda"`` while the distribution actually calls it
  ``"failure_rate"``. The name is corrected (the ``life <-> rate`` transforms
  were already right), so Exponential accelerated-life models fit, predict and
  serialise; a guard test now checks every distribution's declared life
  parameter is a real parameter of that distribution.
- **Exact and Kalbfleisch-Prentice tie handling for Cox** (#142). ``CoxPH.fit``
  gains two further ``method`` choices beyond ``'breslow'`` and ``'efron'``:
  ``'exact'`` (the average-over-orderings exact partial likelihood, for ties
  that arise from coarse rounding of an underlying continuous time) and
  ``'kalbfleisch-prentice'`` (alias ``'kp'`` -- the exact discrete /
  conditional-logistic likelihood, for genuinely discrete time). Both honour
  delayed entry (``tl``), stratification and count weights, and reduce to
  Breslow/Efron when there are no ties. The KP denominator is the elementary
  symmetric polynomial of the risk-set scores, computed by the standard
  polynomial recursion; the exact term is summed over tied-death orderings by
  an ``O(2^d)`` subset recursion, which is guarded against oversized tie sets.
  Validated by matching a brute-force per-tie likelihood exactly, and by
  score/Hessian agreement with finite differences. These methods are niche --
  Breslow and Efron already match what R's ``survival`` and lifelines use by
  default -- and correspondingly more expensive under heavy ties.

Serialisation
~~~~~~~~~~~~~

- **Survival tree & forest serialisation** (#191). ``SurvivalTree`` and
  ``RandomSurvivalForest`` now implement ``to_dict`` / ``from_dict`` (and
  ``to_json`` / ``from_json``), completing the serialisation campaign that had
  deferred them while the forest was crash-prone. A tree serialises as its
  recursive node structure with each leaf stored as its own fitted model
  (``Parametric`` / ``NonParametric``, or a sentinel for the empty
  ``NeverOccurs`` leaf), so a restored tree predicts identically without
  re-fitting; a forest is the ensemble settings plus its trees. Both carry a
  ``"model"`` class tag and dispatch through the package-level
  ``surpyval.from_dict`` / ``surpyval.from_json``, are schema-stamped, and are
  BSON-native for MongoDB. In the course of this, a latent leak was fixed in
  ``Parametric.to_dict``: ``_neg_ll`` (always) and ``gamma`` / ``p`` / ``f0``
  (for offset / LFP / zero-inflated models) were emitted as NumPy scalars,
  which MongoDB's BSON encoder rejects; they are now native floats.
- **Accelerated Life model serialisation.** Fitted Accelerated Life
  parameter-substitution models (``AcceleratedLife(dist, life_model)``) now
  round-trip through ``to_dict`` / ``from_dict`` / ``to_json`` / ``from_json``
  and the package-level ``surpyval.from_dict``. Previously only the fixed-form
  covariate families (AFT, PH, PO, AH) serialised and any Accelerated Life
  model raised ``NotImplementedError``. The model is rebuilt from the stored
  distribution and built-in life-model names (``Power``, ``Eyring``,
  ``Linear``, the Arrhenius-style ``Exponential``, the dual-stress
  ``DualPower`` / ``DualExponential`` / ``PowerExponential``, and their
  inverses), so the restored model predicts identically and, when a covariance
  was stored, reproduces the same confidence bounds. A genuinely custom life
  model (whose parameterisation is not a fixed name map) is still refused with
  a clear error.

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