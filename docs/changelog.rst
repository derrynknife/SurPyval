Changelog
=========

v0.21.1 (unreleased)
--------------------

**Behaviour changes.** Fits accept an optimiser's answer only when it is a
verified maximum, and a fit given ``init`` is also started from the default
start, so a few fits that stopped short silently now reach a better
maximum or warn; data with no maximum raise ``ValueError``. A
non-parametric ``df`` is the probability of each step. Kaplan-Meier and
Nelson-Aalen keep the estimate over a step with no one at risk, as R's
``survfit`` does. Gray's test and the competing-risks Cox incidences now
match R. Unknown option values raise ``ValueError`` everywhere.

- **Cox models no longer break on a covariate far from zero (#459).**
  ``CoxPH`` fitted on the raw covariates, so a column such as a year or a
  date overflowed ``exp(beta'Z)``: on 200 rows, adding 2000 to a N(0, 1)
  covariate moved beta from 0.860 to 0.768, made the survival and
  p-values NaN, gave a false "monotone partial likelihood" warning and
  leaked numpy overflow warnings. The fit now centres the covariates on
  their (n-weighted) means, as R's ``coxph``, lifelines and
  scikit-survival do, and stores them as ``model.center``; beta and every
  prediction are unchanged by any shift of a column, for plain,
  stratified and time-varying fits, residuals, ``check_ph``, robust
  errors and the cause-specific Cox model. The reported baseline (``h0``,
  ``H0``) is now that of a unit at ``center`` (R's
  ``basehaz(centered=TRUE)``), and ``phi(Z)`` the hazard ratio against it,
  ``exp(beta'(Z - center))``. Saved Cox models carry ``"center"`` and are
  stamped schema 2, which 0.21.0 and earlier refuse with a request to
  upgrade; files saved earlier load with their baseline at 0, as fitted.
- **Fits no longer stop far from the maximum in silence (#427, #428,
  #429).** The maximum-likelihood ladder took the first optimiser that
  reported success, and from a poor start BFGS, TNC and Newton-CG report it
  where the likelihood first looks flat: a Weibull started at alpha 1e7
  returned beta 0.099 with a log-likelihood 40 below the maximum. A rung
  now counts only when its answer has a zero gradient and a
  positive-definite Hessian; a fit given ``init`` is also started from the
  default start and keeps the best likelihood; an answer still not a
  verified maximum warns. Accelerated-life fits (InversePower was 14.7
  below the maximum), additive-hazards fits, and NHPP, proportional
  intensity and renewal fits (ARI 227 below; Duane NaN) check their answer
  and use the default start the same way. Fits the first rung solves cost
  the same as before.
- **Data with no maximum are refused (#392).** A parametric MLE whose data
  one failure time explains completely returned a spike: an exact 0.5 with
  a left-censored 1 gave a Weibull beta of 395.7 and a Normal sigma of
  5e-324; the intervals (1, 3] and (2, 4] gave beta 57.9. These raise
  ``ValueError`` now, as tied values already did.
- **The truncated likelihood is exact in the upper tail (#412, #393).** A
  truncation or interval window where F rounds to 1 gave +inf or NaN (a
  LogNormal left-truncated at 1 with mu = -5 gave +inf, not -23.73), which
  made truncated fits depend on the data's units (Normal 8.536, 2.461 on
  the data but 48.87, 23.04 on the data x 7.3). Such windows are computed
  from the log survival function now; truncated fits are unit-free, reach
  the maximum (8.745, 2.459), and run about 40 times faster.
- **Continuous distributions are accurate in their tails (#410, #442,
  #443, #444, #447).** Functions no longer take the log or complement of a
  probability already rounded to 1, 0 or inf. ``Logistic.sf`` was NaN
  more than 709 scales below the location; ``Hf`` and ``log_sf`` were
  -0.0 deep in the left tail (true 1e-300) and ``log_ff`` 0 deep in the
  right (true -1e-30); log forms were +-inf where the probability
  underflows (Weibull ``log_ff`` at 1e-400 is -921); hazards were NaN
  where density and survival both underflow (``Normal.hf`` uses the
  asymptotic series above z = 100); LogNormal, Weibull, LogLogistic and
  Gamma were NaN at x = 0; Gamma ``df`` raised ``OverflowError`` at shape
  1000; ``Rayleigh.qf(1e-30)`` was 0 and ``Hypoexponential.qf(1e-30, 1,
  2)`` 4.3e-61 against 1.0e-15. Hypoexponential uses a power series near
  the origin and gains ``log_sf``, ``log_ff`` and ``log_df``. Every
  function is within 1e-8 of 50-digit mpmath values on the tail grid, and
  fitted parameters are unchanged.
- **Discrete and Beta-family distributions are accurate in their tails
  (#442-#447, #449, #458).** Geometric used ``log(1 - p)`` (8 digits lost
  at p = 1e-9); BetaGeometric took ``ln B(a, b + k)`` as a difference of
  ``gammaln`` values of size 2.6e13, losing up to 22% of ``ff`` at
  k = 1e12, and its ``qf(1)`` was finite; Beta4 raised ``OverflowError``
  at extreme shapes. The references generated for Beta, Binomial,
  DiscreteWeibull and NegativeBinomial (#448) exposed 102 more failing
  groups -- a Beta ``sf`` of 0 against 1e-30, a NaN density at
  alpha = beta = 1000, a Binomial hazard of 0 against 0.99999, a
  NegativeBinomial ``qf`` 2.5% short at p = 1e-6 -- and Poisson and Beta4
  lost their tails the same way. The incomplete beta gains log-scale tails
  (``betaincln`` and the new ``betainccln``, a continued fraction where a
  tail is below 1e-3), with accurate ``ln B`` and gamma ratios and exact
  discrete quantiles. Every one of the 1,404 tail groups now agrees with
  50-digit mpmath values; fitted parameters are unchanged.
- **Discretize: ``qf(ff(k))`` is k (#383).** It returned k + 1 where
  ``ceil`` rounded k + 1e-15 up.
- **Distribution functions accept lists and tuples (#424).**
  ``Gamma.sf([5, 10], 8, 3)`` returned six values (Python list repetition)
  and ``Weibull.sf([5, 10], 8, 3)`` raised ``TypeError``; every
  distribution's functions take a list or tuple as an array now.
- **A missing query gives NaN everywhere (#382).** The constant-hazard
  models (Exponential and its regressions, Geometric, the HPP ``iif``),
  Uniform, Beta4, Binomial, Bernoulli, FixedEventProbability,
  ExactEventTime, NeverOccurs, InstantlyOccurs, BetaGeometric and
  RoystonParmar ``qf``, the Gaussian copula's ``cdf`` and the
  non-parametric and renewal ``mcf`` returned a number, the last value or
  raised at a NaN; they give NaN there now.
- **Fixed: Turnbull keeps its last piece when every row is right
  truncated (#391).** The ladder assumed the last bound was +inf and
  dropped the piece ending at the largest ``tr``: one failure at 1
  observable up to 1 gave sf(1) = 1 (0 with
  ``turnbull_estimator='Kaplan-Meier'``), and a left-censored row at its
  truncation time raised ``IndexError``.
- **Changed: a non-parametric ``df`` is the step probability (#408).** It
  is the drop in ``sf`` over the step ``hf`` differences, not
  ``hf * exp(-Hf)``, which was inf x 0 = NaN (with a raw warning) where a
  Kaplan-Meier reaches zero: ``KaplanMeier.fit([1, 2, 3]).df([2.5, 3.5,
  4.5])`` was ``[inf, nan, nan]`` and is ``[1/3, 1/3, 1/3]``. Values move
  slightly elsewhere (the Nelson-Aalen example: 0.2047 to 0.1811).
- **Fixed: cubic confidence bounds close onto the estimate (#417).** A
  non-linear ``interp``'s bounds are the interpolated estimate plus the
  interpolated distance over the times with a variance, so they close onto
  ``sf`` and always contain it (at ``alpha_ci`` -> 1, 0.1873 against sf
  0.1948 before); linear bounds are unchanged.
- **Fixed: ``band`` at a large ``alpha_ci`` (#420).** The critical-value
  search climbed from far below the root: ``alpha_ci = 0.9`` took 13 s and
  ``1 - 1e-6`` tried to allocate 158 TiB or hung. Now 0.1-6 s; critical
  values at the usual levels are unchanged.
- **Changed: a step with no one at risk keeps the estimate (#425).**
  ``kaplan_meier`` and ``nelson_aalen`` took a step with no one at risk and
  no events to zero while ``fleming_harrington`` kept its value. All three,
  and the variances, now carry the estimate there, as R's ``survfit``
  does; a step with events and no one at risk raises ``ValueError``.
- **Semi-parametric fitters refuse an infinite event time (#394).**
  ``CoxPH`` (stratified too), ``AdditiveHazards``, ``BuckleyJames``,
  ``CompetingRisksProportionalHazards``, ``FineGray`` and ``CompetingRisks``
  took an observed ``x = inf`` as an event: Cox returned beta 19.4 on two
  rows and Lin-Ying died in a ``LinAlgError``. They now raise the
  univariate fitters' ``ValueError``; an infinite censoring time is still
  accepted.
- **Buckley-James takes one covariate row per time (#426).** ``sf``,
  ``ff`` and ``Hf`` took one vector only and raised numpy's bare matmul
  error for paired rows; they now pair row i with ``x[i]`` like every other
  regression model, and refuse a mismatched ``Z`` with a message.
- **Frailty models drop a row with a missing group (#388).** A ``NaN``
  label was a group of its own (9 groups instead of 8) and ``None`` raised
  ``TypeError``; such rows are dropped with one "Dropped k of n rows"
  warning, and ``group=nan`` predicts ``nan``.
- **Cox refuses a coefficient it cannot estimate (#409).** A constant
  column on separated data got a coefficient of 3.1e14, an all-NaN baseline
  and a dozen raw numpy warnings. ``CoxPH`` refuses a column that does not
  vary within any risk set at an event time, naming it, and warns of
  collinear columns.
- **Competing-risks Cox incidences add up to 1 - sf (#384).** They were
  built on the product-limit survival while ``sf`` is ``exp(-H)`` (summing
  to 1.0 against ``ff`` = 0.975 at t = 30 on the conformance fixture). Each
  step now uses the matrix-exponential transition probabilities of R's
  multi-state ``coxph``, which it matches to 7 digits.
- **Gray's test matches cmprsk (#380).** The variance was SurPyval's own
  linearisation: 6.741 against ``cuminc``'s 7.015 on tied data. The score,
  variance and rho-weight incidence now follow cmprsk's ``crst`` routine
  and agree with ``cuminc`` to about 1e-14, with ties and any rho.
- **Lin-Ying survival stays in [0, 1] (#376).** ``AdditiveHazards.sf``
  rose above 1 (1.21 inside the data, 57.8 at a row with a negative
  hazard) because the estimate falls between event times. It now predicts
  with the running maximum of its cumulative-hazard estimate from time 0;
  the fitted ``H0`` is unchanged. The parametric additive hazards models
  are not yet changed.
- **Wald bounds that do not exist say so (#411).** ``param_cb`` and ``cb``
  were a silent ``[nan, nan]`` where a variance was negative
  (GeneralizedRenewal's ``q`` = 2.7e-16 had variance -0.031), and ARI's
  ``rho = 1.0`` raised ``ZeroDivisionError``. They give NaN with one
  warning naming the parameter and the reason now.
- **Rate bounds at zero, and discrete hazards (#413, #414).**
  ``cb(on='hf'/'df')`` was ``[nan, nan]`` where the rate is 0; it is
  ``[0, 0]``. The discrete hazard bound was centred on ``df(k)/sf(k)``
  (Poisson ``hf(10)`` = 0.719 had bounds [1.73, 3.78]); it uses the
  model's ``df(k)/sf(k-1)`` on the logit scale now ([0.634, 0.791]), and
  ``hf`` of a discrete limited-failure or zero-inflated model is
  ``df(k)/sf(k-1)`` too (it gave 0.271 for 0.213).
- **Royston-Parmar one-sided bounds (#415).** ``bound='lower'`` on ``ff``
  or ``Hf`` returned the upper end, and ``bound='both'`` was taken as
  ``'upper'``; both are right now, and ``'both'`` raises.
- **Regression cumulative-hazard bounds have no ceiling (#418).** ``sf``
  was clipped at 1e-15, so ``Hf`` bounds stopped at 34.54 (GumbelPH
  ``Hf`` = 110.6 had [34.54, 34.54]); they are formed from ``Hf`` now
  ([4.4e-18, 261]).
- **Likelihood-ratio bounds (#421, partly).** A search that stopped on
  the wrong side of the estimate is retried (Rayleigh's ``df`` lower bound
  was 0.0502 against an estimate of 0.0359), one-parameter bands are the
  exact extreme over the profile interval, and the Uniform's search
  respects the data's extremes (30 s to 1.3 s).
- **Count-terminated simulation of a falling intensity is refused with the
  reason (#386).** A CoxLewis with beta < 0 expects only ``cif(inf)``
  events ever (6.04 on the conformance fixture), so a sequence can stop
  short of the count: one seed failed with "Event times 'x' must be
  finite" and others returned a sample silently.
  ``count_terminated_simulation`` (and ``_data``) raise a ``ValueError``
  for every seed now, giving ``cif(inf)``, the chance of falling short and
  the time-terminated alternative.
- **CoxLewis least-squares fit (#419).** The search started at
  alpha = beta = 1, where ``cif(60)`` is about 1e26, and BFGS stopped far
  off: on a sample of the fitted model ``cif(55)`` was 113.4 against 4.40
  by MLE. It starts from the constant rate through the MCF now, and a BFGS
  stop that did not converge is finished by Nelder-Mead: 4.32, matching a
  direct minimisation. Fits in hours rather than days agree too (both gave
  a ``cif`` of inf).
- **One rule for unknown option values (#416).** ``mcf_cb(bound='both')``
  raised ``UnboundLocalError`` and an unknown ``interp`` there returned the
  event-time bounds; an unknown ``interp`` on the non-parametric estimators
  raised scipy's ``NotImplementedError``; the cause-specific Cox model
  accepted any ``interp`` and ignored it. All raise ``ValueError: '<arg>'
  must be one of (...); got ...`` now; the cause-specific Cox model takes
  ``interp="step"`` only, and ``DestructiveDegradation.cb`` accepts
  ``on='R'`` / ``'F'`` like every other ``cb``.

v0.21.0 (28 September 2026)
---------------------------

**Upgrading to 0.21.** Most code runs unchanged, but these changes can alter
results or break code without a warning:

- **Shapes.** A scalar query now returns a numpy scalar, not a ``(1,)``
  array, and a two-sided bound ends in a ``[lower, upper]`` axis: code
  that indexed a scalar result (``km.sf(5)[0]``) raises ``IndexError`` and
  uses the result directly instead.
- **Cox ties.** ``CoxPH.fit`` defaults to Efron's tie handling (it was
  Breslow's), so fits to tied data change; pass ``tie_method="breslow"``
  for the old model.
- **Draws.** ``random()`` of a non-parametric estimate draws from the
  estimate, with ``inf`` for the probability beyond the last time; of a
  limited-failure or zero-inflated model it returns lifetimes (``inf`` for
  a unit that never fails), and the old survival-data draw is
  ``random_data()``. Seeded draws of those models give different numbers.
- **Limited-failure summaries.** ``mean()``, ``moment()`` and ``var()`` of a
  model with ``p < 1`` are ``inf``; ``defective=True`` gives the old values.
- **Criteria and extrapolation.** ``ParametricCompetingRisks.bic()`` is the
  joint criterion (larger than before), and the additive hazards model
  holds its estimate after the last observed time instead of extending it.
- **Missing values.** A missing time, covariate or probability gives NaN at
  prediction, and fitting drops rows with a missing covariate with one
  warning (or refuses them where a row is part of one unit).
- **Recurrent interval levels** are ``alpha_ci=0.05``, by keyword only: an
  old positional ``confidence`` level in ``mcf_cb`` or the recurrent
  ``plot`` methods raises ``TypeError``.

Renamed arguments keep working, with a ``DeprecationWarning`` naming the new
name, until v0.22.0, which removes them together with the
``surpyval.experimental`` alias (use ``surpyval.beta.ml``) and ``band``'s
unused ``n_sims`` and ``random_state``. To find the calls to update, run your
code or tests with ``python -W error::DeprecationWarning``. Saved models from
earlier versions still load; a model saved by 0.21 with a feature older
versions cannot read (a ``set_support`` support, a truncated band's sample
size, some formula terms) is stamped schema 2 and refused by them with a
request to upgrade.

- **Changed: one name per option (#422, principle 21).** The same option
  had different names in different parts of the package; each now has one,
  and the old name keeps working until v0.22.0 with a
  ``DeprecationWarning`` naming the new one (``surpyval.utils.deprecation``
  does this for every rename):

  - Interval level: ``alpha_ci=0.05`` everywhere (the recurrent ``mcf_cb``
    and plots took ``confidence=0.95``). Seeds: ``random_state`` everywhere
    (some recurrent, Buckley-James and degradation methods took ``seed``).
    Bootstrap size: ``n_boot`` (``B`` in ``NonParametric.bootstrap_cb``).
  - Times are ``x`` and a quantile's probability ``p`` everywhere
    (``Parametric.cb``, Royston-Parmar and the degradation models took
    ``t``; ``qf`` took ``u`` or ``q`` in a few models).
  - Regression and competing risks: ``CoxPH.fit``, ``fit_from_df`` and
    ``fit_tvc*`` take ``tie_method`` (was ``method``; ``CoxPH.baseline`` and
    the competing-risks Cox already did); the ``fit_tvc*_from_df`` methods
    take ``i_col`` (was ``id_col``: the data argument is ``i``) and
    ``fit_tvc_timeline_from_df`` takes ``x_col`` (was ``time_col``);
    ``BuckleyJamesModel.bootstrap_ci`` takes ``random_state`` (was
    ``seed``); ``CompetingRisksProportionalHazards.fit`` / ``fit_from_df``
    take ``model="Cox"`` or ``"Fine-Gray"`` (was ``how``, the estimation
    method everywhere else; the fitted ``.how`` is ``.model``); and
    ``FineGray.fit`` and ``gray_test`` take ``event`` (was ``cause``). The
    non-parametric ``CompetingRisks.fit`` / ``fit_from_df`` choose their
    survival estimator with ``how`` (was ``method``; the fitted
    ``.method`` is ``.how``).
  - Recurrent events: ``NonParametricCounting.mcf_cb`` and ``.plot``,
    ``CauseSpecificMCF.mcf_cb`` and ``.plot``, and the parametric and
    proportional-intensity ``plot`` take ``alpha_ci=0.05`` (was
    ``confidence=0.95``; ``confidence=0.9`` is read as ``alpha_ci=0.1``),
    keyword-only, so an old level passed by position raises rather than
    silently meaning its complement. The simulations, the simulated
    ``mcf`` and ``plot`` and every ``cramer_von_mises`` take
    ``random_state`` (was ``seed``), and ``CauseSpecificMCF`` /
    ``CauseSpecificNHPP`` take ``event`` (was ``cause``). Plot labels read
    "95%", not "95.0%".
  - Degradation: times are ``x``, not ``t``, in the Wiener and Gamma process
    models (``sf``, ``ff``, ``df``, ``hf``, ``Hf``) and
    ``DestructiveDegradationModel`` (``sf``, ``ff``, ``df``, ``Hf``, ``cb``,
    ``median_degradation``, ``degradation_quantile``, whose probability is
    ``p``, not ``q``). ``Z`` comes straight after the query:
    ``DegradationModel.cb(x, Z, on, ...)`` and the process models'
    ``random(size, Z, random_state)``, as ``DegradationModel.random``;
    ``DegradationModel.induced_life(n_samples, *, Z, random_state)``,
    ``DegradationModel.predict_rul(x, y, *, Z, Z_future, alpha_ci,
    n_samples, random_state)`` and the process models'
    ``predict_rul(current_degradation, *, Z, alpha_ci)`` take ``Z`` first
    and the rest by keyword, so any argument they are given by position
    after the query is read in the old order, with a warning. A
    call in the old positional order (a string second argument to ``cb``,
    or two positional arguments after ``size`` in ``random``, read as
    ``(random_state, Z)``) still works with a warning; ``random(size, v)``
    on a model fitted with stress, which raised for the missing ``Z``,
    now draws at stress ``v``.
- **Every ``random`` takes ``random_state`` (#389).** The univariate
  distributions and models (``random``, ``random_data``), mixture models
  (which accepted and ignored it), Royston-Parmar and the PH, AH and
  accelerated-life regressions now take a keyword-only ``random_state``:
  ``None`` draws from numpy's global stream exactly as before, and a seed
  gives a stream of its own (``numpy.random.default_rng(seed)``) that
  leaves the global one alone. ``conformance/test_seeds.py`` checks this
  for every registered model that draws.
- **``Binomial.random`` accepts a fitted model's parameters.** A fitted
  ``n`` is a float (5.0), and ``random`` raised ``TypeError: Cannot cast
  scalar from dtype('float64') to dtype('int64')``; a whole-number float is
  accepted now, and a fractional ``n`` raises a ``ValueError``.
- **Time-varying covariate paths start at 0 (#433).** A schedule starting
  before 0 was counted as age in the AFT ``sf_tvc`` and the degradation
  stress clock: a constant ``WeibullAFT`` path from -10 gave
  ``sf_tvc(20) = 0.8626`` against ``sf(20, Z) = 0.9362``, and the clock gave
  F(100) = 0.662 against 0.489. A schedule starting after a query time
  raised. Every schedule is now clipped to start at 0 -- the part before 0
  is ignored, a later start holds its first value back to 0 -- for every
  family and ``StressClock`` (whose ``tau(0)`` is now 0).
- **``StepSchedule.from_expression`` means what Python means (#434).**
  ``and`` / ``or`` returned a bool, so ``"(t > 50) and 2.0 or 1.0"`` was 1.0
  everywhere; they return an operand now (2.0 after t = 50). Keyword
  arguments were dropped (``round(t/10, ndigits=1)`` gave 0 at t = 1, 2)
  and ``round(t/10, 1)`` raised a ``TypeError``; both give 0.1, 0.2 now,
  and a keyword a function cannot take raises a ``ValueError`` naming it.
- **``sf_tvc`` / ``Hf_tvc`` accept any time (#435).** Time 0 raised "x must
  contain a positive time"; it gives sf 1 (or the baseline's value at 0)
  now, and negative times match ``sf`` (``NormalAFT`` gave 0.9725 at -5,
  not 0.9801). ``given`` at or below 0 now conditions as documented (a
  Logistic baseline: sf(20 | 0) = 0.9378, not the unconditional 0.8831).
  The new ``conformance/test_tvc.py`` checks every model with ``sf_tvc``
  against ``sf`` for constant paths.
- **Kaplan-Meier no longer fails when the estimate underflows (#450).**
  Once the product fell below the smallest float -- 1100 staggered entries
  with two at risk at each failure, R = 0.5^k -- ``KaplanMeier.fit``
  raised ``FloatingPointError`` from its log-space fallback and leaked
  "divide by zero in log". It gives 0 there now, quietly, and matches
  scikit-survival 0.28 exactly at all 1100 times. A step with no one at
  risk no longer leaks "invalid value" either.
- **``bootstrap_cb`` is NaN outside the data, like ``cb`` (#452).** Without
  a support it carried its step convention past the data: Kaplan-Meier of
  1..10 (last censored) gave ``bootstrap_cb([0.5, 11]) = [[1, 1], [0,
  0.548]]`` where ``cb`` is NaN, and a missing time got the last bounds.
  ``cb``, ``R_cb`` and ``bootstrap_cb`` now share one rule: NaN outside the
  data and at a missing time, or the support's values under
  ``set_support``.
- **A left-truncated estimate's band survives saving without its data
  (#451).** A restored model took ``band``'s N from the largest risk set:
  37 instead of 60 in one example, moving the band at the 20% time from
  [0.4900, 0.8917] to [0.4535, 0.9018] (a truncated Turnbull fit went the
  other way, 8 instead of 4). ``to_dict`` stores it as ``"band_n"`` where
  it differs, stamped schema 2; older dictionaries keep the old fallback
  and untruncated models are still schema 1.
- **Recurrent cause labels (#440).** ``CauseSpecificNHPP`` and
  ``CauseSpecificMCF`` handle cause labels with the same code as the
  univariate competing-risks models. A tuple label did not load back
  (``from_dict`` raised "unhashable type: 'list'"), mixed labels such as
  ``'s'`` and ``2`` raised a bare ``TypeError`` from sorting, and a tuple
  mark on every row was split into a column so the fit raised. All three
  fit, predict per cause and round-trip now. The new conformance property
  ``test_labels.py`` checks tuple and mixed labels for every model fitted
  with ``e=``.
- **Additive-hazards draws below 0 (#441).** ``random()`` of an additive
  hazards model on a Normal, Gumbel or Logistic baseline searches the
  whole support, so the share ``ff(0)`` of its mass below 0 is drawn there
  (GumbelAH: 0.0347 of the draws against ``ff(0)`` = 0.0361); it returned
  2.7e-20 for all of them.
- **Changed: ``ParametricCompetingRisks.bic()`` is the joint criterion,**
  ``2 neg_ll + K ln(n)`` with K the parameters of all causes and n the
  failures of any cause, as every other SurPyval BIC counts n. It was the
  sum of the causes' BICs, which charged each cause only ``ln`` of its own
  failures: the competing-risks guide's example goes from 2220.8 to 2223.1
  (Weibull + Exponential). ``aic()`` was already the joint AIC.
- **ExpoWeibull is accurate in both tails (#436).** ``1 - exp(-t)`` rounded
  to 0 below t = 1e-16 and ``x / alpha`` overflowed:
  ``log_df(1e-4, 10, 4, 0.5)`` was +inf (true -13.12),
  ``ff(1e-3, 10, 4, 2)`` 23% high, and at mu = 1 ``hf``, ``Hf`` and
  ``log_sf`` at x = 100 were NaN, inf and -inf (the Weibull's 30, 1000 and
  -1000). Every function is now computed on the log scale with exact
  branches for each regime and exact values at 0, ``moment`` and ``mean``
  take array parameters, and a fit that failed on data with a value at
  1e-4 (returning its start, neg_ll 124.97) now reaches 90.97.
- **``CustomDistribution`` checks its inputs (#437).** A parameter named
  after a model attribute (``k``, ``dist``, ``data``, ``method``, ...)
  overwrote it silently -- ``k`` moved the AIC from 289.50 to 290.08 -- and
  is now refused with a ``ValueError`` listing the reserved names. ``qf``
  outside [0, 1] is NaN (it was the support's lower bound), any
  ``(x, *args)`` signature is accepted, and reusing a name warns that it
  replaces the registry entry used to restore saved models.
- **``fit_from_non_parametric`` matches ``fit(how='MPP')`` (#438).** It
  plotted the censored times too (alpha 10.711 instead of 10.597 on
  censored data); it plots the failure times only now. ``fit_from_ecdf``
  raises a ``ValueError`` for an F outside [0, 1] or NaN (dropped silently
  before) and for unequal lengths (an ``IndexError`` before).
- **Probability plots with a tick at 0 (#439).** ``round_sig`` took
  ``log10(0)``, so ``Normal.fit([-1, 0.5, 2, 3, 5]).plot()`` raised
  ``OverflowError``. Zero, negative and non-finite ticks work, and
  ``round_sig(0)`` is 0.
- **Mutation testing pilot (#396).** ``scripts/mutation/run.sh`` runs
  mutmut on a module in a copy of the repository, and ``recheck.py``
  checks new tests against its survivors. On the non-parametric estimators
  593 of 2,702 mutants survived the test suite (score 78.1%); 273 were
  real gaps, now covered by ``surpyval/tests/mutation``, which raises the
  score to 88.6% (93.2% without the equivalent and dead-code mutants).
  Among the gaps no test checked: the Hall-Wellner band's width off by a
  factor of N, ``df`` ignoring ``interp``, and ``rmst_diff``'s interval
  and ratio. It found #450-#452, pinned as strict expected failures, and
  that a warning raised inside a shape-wrapped method pointed at the
  wrapper instead of the caller (fixed).
- **Changed: shape in, shape out, for every model (#381, #435).** A function
  evaluated at query points -- ``sf``, ``ff``, ``Hf``, ``hf``, ``df``,
  ``qf``, the per-cause and recurrent ``cif``, ``iif``, ``mcf``,
  ``sf_tvc``, ``Hf_tvc``, ``smoothed_hf`` and every confidence bound --
  returns the query's shape: a scalar gives a numpy scalar, 1-D and 2-D
  queries keep their shape, an empty query gives an empty array, and a
  two-sided bound adds a trailing ``[lower, upper]`` axis. The
  non-parametric estimates, Royston-Parmar, the AFT, PO and AL
  regressions, Cox, the competing-risks and recurrent models, the
  degradation models and every ``cb`` returned ``(1,)`` for a scalar
  (``(1, 2)`` for a bound); several raised on a 2-D or empty query; and
  some gave a right-looking shape with wrong values -- a Kaplan-Meier
  ``cb`` of a (2, 2) query had its axes transposed (lower 0.724 above
  upper 0.063), and a copula's (2, 2, 2) query mixed its coordinates.
  Of 18,342 surveyed calls, 5,506 changed shape and no 1-D value changed.
  Survival trees and forests keep their row-by-time grid, now
  ``(n_rows,) + x.shape``. Code that indexed a scalar query's result
  (``km.sf(5)[0]``) now uses the result directly. Parametric
  ``sf_tvc(..., given=nan)`` is now NaN.
- **Tail accuracy is checked against 50-digit references (#398).**
  ``reference/test_tails.py`` compares ``sf``, ``ff``, ``df``, ``hf``,
  ``Hf``, ``qf`` and the log forms of 17 distributions with mpmath values
  (stored in ``tails_mpmath.json``, written by
  ``scripts/reference/tails_mpmath.py``, so CI needs no mpmath) on a grid
  of extreme parameters and times, from survival 1e-300 to 1e-300 of
  failure. It needs relative accuracy 1e-8 where the value is a normal
  double, or 64 ulps of the inputs' own sensitivity where the function is
  ill-conditioned. 212 groups of values fail, pinned by cause:
  cancellation near probability 1 (#442), log-scale functions that
  under- or overflow (#443), NaN at valid arguments (#444), overflow
  errors at extreme shapes (#445), Geometric at small p (#446), ``qf`` at
  tiny probabilities (#447), BetaGeometric (#449), and the ExpoWeibull
  (#436) and Logistic (#410) forms. Beta, NegativeBinomial,
  DiscreteWeibull and Binomial await their references (#448).
- **Changed: ``random()`` of a non-parametric estimate draws from the
  estimate itself.** It drew each observed value with the estimate's
  probability there, but where the estimate does not reach zero it
  spread the remaining probability over the observed values, so the
  draws disagreed with the model's own ``sf`` (by 0.12 for one Turnbull
  fit) and an all-censored fit raised. Each draw is now ``qf(u)`` for one
  uniform ``u``, and the probability left beyond the last time is drawn
  as ``inf``, as for a parametric model's never-failing units (#403).
- **Every model is refitted to data drawn from itself (#397).** A new
  nightly study, ``calibration/test_refit_registry.py``, takes each model
  in the conformance registry that can simulate (120 of 128; the rest are
  excluded with a reason), draws a few hundred units from its fitted
  fixture 20-100 times, refits, and requires the mean estimate within
  ``3/sqrt(reps) + 0.2`` standard deviations of the truth and the mean
  curve within 3 Monte Carlo standard errors + 0.02. A likelihood that
  ignored delayed entry shows as a 1.44 sd bias against a tolerance of
  0.5. It found that ``random()`` of an additive-hazards model on a
  Normal, Gumbel or Logistic baseline never draws below 0, putting that
  mass (3.6% for one fixture) at 2.7e-20 instead (#441).
- **Added: ``set_support`` for the non-parametric estimates.** Outside the
  data a non-parametric estimate only had a convention: the step curves
  started at 1 and held their last value however far away, while the
  interpolated forms, the confidence bounds and the mean cumulative
  functions were NaN. ``KaplanMeier``, ``NelsonAalen``,
  ``FlemingHarrington`` and ``Turnbull`` models, ``CompetingRisks`` (both
  methods), ``NonParametricCounting`` and ``CauseSpecificMCF`` now take
  ``model.set_support(lower, upper)``: every function, every ``interp`` and
  the pointwise bounds are then at their start value (``sf`` 1, the rest
  0) from ``lower`` to the first observed value, hold the last value up to
  ``upper``, and are NaN outside. Negative and infinite bounds are allowed
  (the variable need not be time). The bounds are the model's ``support``,
  as for the parametric models, and are saved by ``to_dict`` (schema 2).
  Without the call nothing changes.
- **Fixed: a failing non-parametric call no longer silences numpy for the
  whole process.** ``cb``, ``R_cb``, ``band`` and the Turnbull fit turned
  numpy's floating-point warnings off with ``np.seterr`` and back on
  afterwards; a call that raised in between -- ``cb(bound_type="bogus")``,
  an unknown ``interp``, ``band(alpha_ci=2.0)`` -- left them off for every
  later computation in the session. They now use ``np.errstate``, which
  restores the state however the call ends.
- **Fixed: cubic non-parametric curves no longer dip below 0 (#417).** At
  the last time of a Kaplan-Meier estimate that falls to 0,
  ``sf(x, interp="cubic")`` was -2.3e-17, so ``Hf`` there was NaN with a
  raw warning instead of inf. The PCHIP curve is now clipped to the range
  of its knots.
- **Silent non-convergence is checked for every model (#401).** A new
  conformance property, ``test_convergence.py``, forces each iterative fit
  to fail -- an iteration limit of 1, a start a million times the answer,
  or data whose likelihood has no maximum -- and requires a warning, a
  ``ValueError``, or the true maximum. It found 62 fits that return a
  wrong model without a word, pinned as known failures: for example
  ``Weibull.fit`` from ``init=[1.03e7, 2.32]`` returns alpha 1.03e7, beta
  0.099 (log-likelihood -78.2 against -37.9; #427), and every parametric
  PH/AFT/PO model gives a group with no events a finite coefficient
  (-16.3 for WeibullPH) where ``CoxPH`` warns (#392); also #428 and #429.
- **Fixed: ``init`` with an offset is checked in the right order.** The
  check read ``[gamma, *params]`` as ``[*params, gamma]``, so it refused
  valid starts (``Exponential.fit(..., offset=True, init=[6, 10])``:
  "gamma = 10.0 lies outside its bounds") and let an offset beyond the
  first observation through to fail later as "MLE Failed".
- **Changed: ``random()`` of a limited-failure or zero-inflated model draws
  lifetimes (#403).** It returned ``(x, c, n, t)`` survival data when
  ``p < 1`` and an array otherwise, and drew zero-inflated samples by a
  binomial count and a shuffle, so a seed did not give ``qf(u)``. It now
  returns an array for every model, ``qf(u)`` from one uniform per draw:
  ``inf`` for a unit that never fails, 0 for one dead on arrival, and the
  same numbers as ``qf(np.random.random_sample(size))`` after the same
  seed. The survival-data draw is the new ``random_data()``, which censors
  the never-failing units after the last failure, ready to refit.
- **Changed: ``mean()``, ``moment(n)`` and ``var()`` of a limited-failure
  model are infinite (#404).** They returned the defective values (79.76
  for ``Weibull.from_params([100, 2], p=0.9)``), which code reading
  ``mean()`` as the mean life took at face value. A fraction ``1 - p``
  never fails, so they are now ``inf``; ``defective=True`` gives the old
  values. Models with ``p = 1`` are unchanged.
- **Added: ``df(x, continuous=True)`` (#405).** A zero-inflated model's
  ``df(0)`` is the point mass ``f0``, so integrating ``df`` on a grid from 0
  counted a spurious ``f0 * dx / 2`` (0.95 instead of 0.90 for
  ``f0 = 0.1``). ``continuous=True`` returns the continuous part alone,
  ``(p - f0)`` times the base density; the docstring says what ``df(0)``
  is. A zero-inflated ``df`` at a scalar now returns a scalar.
- **Added: ``model.with_params(params)`` and ``model.extras`` (#406).**
  ``from_params(model.params)`` silently dropped the offset, ``p`` and
  ``f0`` (``sf(50)`` 0.7788 instead of 0.7533). ``extras`` is the dict of
  those the model has (``{"gamma": 5.0, "p": 0.9, "f0": 0.1}``, empty for
  a plain model), and ``with_params`` rebuilds the same model with other
  parameters, validated as ``from_params`` validates them.
- **Fixed: loose ends of the zero-inflation mass at 0 (#407).** ``Hf``
  before time 0 was ``-0.0``; it is ``0.0``. The entropy docstring and
  error message still put the mass at the offset; they now say 0.
- **Every public item has a runnable example (#402).** 49 public classes,
  functions and fitters had a docstring but no example, and the low-level
  ``kaplan_meier``, ``nelson_aalen`` and ``fleming_harrington`` had no
  docstring. Each now has a short, seeded example that runs as a doctest
  in CI, and the conformance check allows no public item without one.
- **Changed: the additive hazards model holds its estimate past the last
  observed time (#400).** ``AdditiveHazardsModel.Hf`` kept changing after
  the last observed time, at the last interval's rate
  :math:`\beta'(Z - \bar Z)`, where there is no risk set to estimate
  anything from: on one fit, ``Hf`` was 3.52 at the last time and 23.0 at
  100 times it. It now holds its value there, as every other
  semi-parametric estimate does, so ``sf`` and ``ff`` hold and ``hf`` and
  ``df`` are 0. ``hf`` at a NaN time is now NaN rather than
  :math:`\beta' Z`.
- **Design principles (#379).** A new page, :doc:`Design Principles`, lists
  the rules every model keeps -- one data format, ``nan`` in and out, order,
  units and counts not mattering, consistent shapes and identities,
  behaviour outside the data, entry points agreeing, the same defaults and
  names everywhere, calibrated and consistent intervals, one seed rule,
  useful warnings, documented examples -- each with the tests that enforce
  it and the issues where a model does not yet comply. The README
  summarises them. Three new conformance checks fill the gaps: behaviour
  outside the data (``test_outside_data.py``: a step or semi-parametric
  estimate starts at its initial value and, past the last time, holds or is
  ``nan`` for every function alike; found #400, additive hazards
  extrapolating), the same defaults across a fitter's entry points
  (``test_defaults.py``), documentation (``test_documentation.py``: every
  public item has a docstring with an example; 49 are listed against #402
  and the list can only shrink).
- **Option sweeps in the conformance suite (#379).** The other checks call
  each model with its default options, where many past bugs lived in the
  others. ``test_options.py`` sweeps every confidence-bound method of every
  registered model over ``on=``, ``bound=``, ``alpha_ci`` and their
  variants, every ``interp=`` value and every estimation option. It checks
  that the bounds contain the estimate and stay in range, that one-sided
  and two-sided bounds agree, that intervals nest and close onto the
  estimate as ``alpha_ci`` approaches 1, the ``sf``/``ff``/``Hf``
  transforms and the output shapes, and that shared options have one name
  and default across models. It found bound failures now tracked as #411
  and #413-#419 (for example discrete ``cb(on="hf")`` centred on
  ``df/sf(k)`` instead of ``hf``, Royston-Parmar one-sided ``ff``/``Hf``
  bounds on the wrong side, and ``Hf`` bounds capped at 34.54) and ten
  naming inconsistencies (#422), each a strict expected failure.
- **Raw numerical warnings no longer leak from Kaplan-Meier, Binomial,
  Weibull and LogNormal functions.** ``KaplanMeier.Hf``/``hf`` past the
  time the estimate reaches zero, ``Binomial.Hf`` from ``x = n`` on,
  ``Weibull.df``/``hf`` at 0 with a shape below 1, and ``LogNormal.sf`` at
  0 gave numpy "divide by zero" or "invalid value" warnings; their values
  (``inf``, or 1 for ``sf(0)``) were already right and are now returned
  without a warning. The conformance and property suites now fail on any
  raw numpy, scipy or autograd warning that escapes the package, and check
  that a fit or prediction gives each deliberate warning at most once
  (#379). The check found five wrong results hidden behind warnings,
  tracked as #408-#412.
- **Changed: ``CoxPH.fit`` defaults to Efron ties, with the matching Efron
  baseline (#387).** ``fit`` defaulted to Breslow while ``fit_from_df``, the
  time-varying-covariate fits and the competing-risks Cox model defaulted to
  Efron, so the same tied data gave different models by different routes.
  Every route is now Efron. Efron is chosen on merit: with ties from
  rounding a continuous time, Breslow biases the coefficients towards zero
  (in a simulation with true :math:`\beta = 0.7`, by -0.06 to -0.21 as the
  ties grow, against -0.006 to -0.06 for Efron), at no saving worth having.
  An Efron fit's baseline hazard now takes the same tie correction as its
  likelihood: the :math:`d` deaths tied at a time leave the risk set a
  fraction at a time, and the step is
  :math:`\sum_{l<d} 1 / (R - \tfrac{l}{d} R_D)` instead of Breslow's
  :math:`d / R` -- the covariate-weighted Fleming-Harrington estimator, as
  Breslow's is the covariate-weighted Nelson-Aalen. It matches R's
  ``survfit.coxph`` after an Efron fit (checked against it in the reference
  tests) and the Efron residuals, which already used it. **Results change
  on tied data**: pass ``tie_method="breslow"`` for the old fit. Without ties
  every method gives the same model as before.
- **Property-based tests (#379).** Hypothesis generates data with mixed
  censoring, ties, counts, truncation and tiny samples, and checks the
  non-parametric estimators, parametric fits, regression, competing-risks,
  recurrent-event and serialisation paths against general properties
  (valid curves, local optimality against an independent likelihood,
  invariance to row order, units and counts, ``ValueError`` on invalid
  input), shrinking any failure to a minimal case. The default run takes
  under a minute; ``SURPYVAL_HYPOTHESIS_PROFILE=nightly`` searches
  thoroughly in the nightly workflow. ``hypothesis`` is a new test-only
  dependency. It found four bugs, pinned as strict expected failures:
  Turnbull dropping its last piece under right truncation (#391), silent
  degenerate fits where the likelihood has no maximum (#392), unit-dependent
  fits to truncated data (#393), and ``CoxPH`` accepting an infinite event
  time (#394).
- **Statistical calibration suite and nightly run (#379).** Simulation
  studies in ``surpyval/tests/calibration`` (opt in with
  ``--run-calibration``) check that results are statistically right, not
  only consistent: confidence-interval coverage for parametric,
  non-parametric, Cox and parametric regression, degradation and recurrent
  bounds; parameter recovery with truncation, interval censoring, limited
  failure, frailty and renewal models; size and power of the log-rank,
  stratified log-rank, Gray, Laplace, MIL-HDBK-189C and Cramer-von Mises
  tests; and Brier/AUC bias with tied times. Each passes within 3 Monte
  Carlo standard errors plus a stated slack, with fixed seeds, and would
  have caught the old Gray's test (size 0.20 against 0.05) and the #365
  Brier bias. The scheduled ``nightly.yml`` runs the full suite on three
  Pythons, the docs build and the calibration suite against ``develop``
  daily, once it is on ``master``. Found: the equal-precision (``nair``)
  Kaplan-Meier band covers about 0.89 for a nominal 0.95 (#390).
- **A conformance suite checks every model against the same properties
  (#379).** Bugs kept reappearing as old kinds of failure in new models
  (unsorted input, units, row routing, missing values, serialisation),
  because each fix tested only its own case. ``surpyval/tests/conformance``
  registers every public model (128 cases) and runs each through the
  identities between its functions; scalar, 2-D and empty queries; query
  and row order; units, data-row order and counts; valid values; the
  missing-value rule; seeds; the strict-JSON round trip; and agreement of
  its fit paths. A test fails when a public model is left unregistered. The
  fast set runs on every pull request (about 40 s). The 58 failures it
  found are strict expected failures, each naming its issue (#381-#388).
- **Numbers quoted in the documentation are checked (#379).** The prose
  around executed examples quoted outputs ("a shape of about 2.1", "the
  lower AIC") that nothing verified, so they went stale when outputs
  changed. Hidden cells now assert 263 such claims across 16 pages, and the
  documentation build fails when one no longer holds. The first pass found
  two stale statements in the offset section of *Parametric SurPyval
  Modelling*: the starting offset is ``min(x)`` minus the data's mean
  spacing, not ``min(x) - 1``, and the example's quoted moment-based shape
  was from a different sample. See "Checking the numbers quoted in the
  text" in :doc:`Contributing`.
- **Stored results from R and Python survival software (#379).**
  ``surpyval/tests/reference`` compares SurPyval with 82 results computed
  once on shared fixtures (lung, heart, aml, ovarian, PBC, and small sets
  with ties, left truncation, interval censoring and competing risks) by R
  survival 3.5-8, cmprsk 2.2-11, timereg 2.0.5, pec, riskRegression, npsurv
  and fitdistrplus, lifelines 0.30.3 and scikit-survival 0.28, so CI needs
  neither R nor lifelines; ``scripts/reference/regenerate.sh`` rebuilds
  them. Kaplan-Meier, Nelson-Aalen, restricted mean, log-rank, MCF,
  Aalen-Johansen, Lin-Ying, Brier score and AUC agree to rounding; Cox
  (Breslow, Efron, strata, left truncation, start-stop), survreg AFT fits,
  Fine-Gray and Turnbull to between 1e-6 and 5e-4. Deliberate differences
  are asserted and recorded with their reason. Gray's test disagrees with
  cmprsk in its variance (#380).
- **Degradation: missing values give NaN, and predictions read a DataFrame by
  name (#375, #374).** Gamma- and Wiener-process models gave sf = 1 and
  ``ff = Hf = hf = df = 0`` at a NaN time and ``qf(nan) = inf``, raised on a
  NaN stress, and ``predict_rul(current_degradation=nan)`` never returned.
  ``DegradationModel.qf`` returned inf for a NaN covariate or p and used only
  the first row of ``Z``; ``InducedFailureDistribution`` gave
  ``ff(nan) = 0``; the bootstrap ``cb`` raised on a NaN stress. A missing
  time, stress or probability now gives NaN for that element only, and
  ``qf`` pairs each p with its row of ``Z``. ``predict_rul``,
  ``predict_failure_time`` and ``induced_life`` still raise for a missing
  value (they describe one unit), and the process quantile search can no
  longer loop forever. ``DegradationAnalysis.fit_from_df`` and the new
  ``WienerProcess.fit_from_df`` / ``GammaProcess.fit_from_df`` record the
  stress columns as ``Z_cols`` (kept through ``to_dict``), so every method
  that takes ``Z`` accepts a DataFrame and selects those columns by name; a
  model fitted from arrays refuses one with an accurate message (it used to
  say "fit the model with ``fit_from_df``" to a model fitted that way).
- **One rule for missing values (#375).** Prediction: a missing covariate,
  time or probability gives NaN for exactly the outputs that depend on it,
  and a method whose input is one unit's history (``predict_rul``,
  ``induced_life``, ``sf_tvc``, ``mcf``) raises instead. Fitting: rows with
  a missing covariate are dropped with one warning where rows are
  independent observations, and refused where a row is only part of one; a
  missing time or response always raises. See :doc:`Conventions`. Fixed to
  follow it:

  - Survival trees and forests sent a missing covariate right at every
    split, so it predicted like +inf (tree sf 0.6974 for both), and kept
    such rows in the fit without a warning. They are now dropped with a
    warning, predict NaN, and ``RandomSurvivalForest.score`` is NaN when a
    score is missing.
  - Proportional-intensity ``mcf`` with a missing covariate ran every
    sequence to ``max_events`` and then reported a missing *time*; it and
    the simulation entry points now refuse a missing or mis-shaped ``Z`` by
    name before simulating.
  - ``survival_probability`` cast ``Z`` to float, so a formula fit with
    string levels could not be scored; a DataFrame is now passed to
    ``model.sf`` as it is.
  - A missing time at prediction returned the value at t = inf in
    ``CoxPH`` (every method, plain and stratified: sf 0.0102, ``hf`` and
    ``df`` 0), competing-risks Cox (cif 0.385) and Fine-Gray (0.354), and
    sf = 1 in Buckley-James; it now gives NaN. Cox ``predict_tvc`` refuses a
    covariate path with a missing value by name.
  - Fine-Gray and the competing-risks Cox array path dropped rows with a
    missing covariate silently; they now warn like every other fitter (and
    drop infinite covariates too).
  - Stratified ``CoxPH`` raised a ``TypeError`` on a missing stratum label;
    such observations are now dropped with one warning, and the array path
    warns once in total rather than once per stratum.
  - Kaplan-Meier, Nelson-Aalen, Fleming-Harrington and Turnbull (every
    function, ``cb`` and ``band``) and non-parametric ``CompetingRisks``
    returned the value at t = inf for a missing time (a NaN sorts past the
    last step), and the array ``hf`` / ``df`` copied a neighbour's increment
    into it; parametric ``sf_tvc`` / ``Hf_tvc`` raised an ``IndexError``.
    They now give NaN for that time only.
  - Covariates given as a list or object array holding ``None`` raised a
    ``TypeError`` in the parametric PH and AH families, the accelerated-life
    fit and ``AdditiveHazards`` prediction; they are now read as floats,
    so ``None`` is a missing value.
- **Competing-risks Cox pairs each time with its own covariate row.** With
  one row per time and unsorted times, ``hf`` / ``Hf`` / ``sf`` / ``ff`` /
  ``df`` read the baseline at the sorted times but used the rows in the
  given order (times [10, 1], rows [2], [-2]: ``Hf`` gave [0.139, 0.590]
  instead of [4.961, 0.017]). ``cif`` was not affected.
- **A declared category level with no fitted rows is refused at prediction
  (#377).** A level listed in ``C(g, levels=[...])``, or an unused category
  of a ``pd.Categorical`` column, got a coefficient with nothing to estimate
  it, so its predictions were made up (the reference level's for
  ``WeibullPH`` and ``CoxPH``; a drifted coefficient for ``WeibullAFT``,
  S = 0.863 against 0.803). The fit now warns once, naming the column and
  the empty levels, and keeps the column so coding stays the same across
  data splits; predicting for such a level raises the same "not fitted
  with" ``ValueError`` as an unseen level, fitted or restored.
  ``AdditiveHazards`` and Buckley-James, which reject an all-zero covariate
  column, still refuse such a fit after the warning. A saved model with an
  empty level needs schema 2.
- **Turnbull reaches the maximum-likelihood estimate with interval
  censoring and right truncation (#368).** Two index searches were one
  Turnbull piece off: a right-censored observation could not fail in the
  piece just after its censoring time, and a right-truncated window
  ``(tl, tr]`` took in the piece just after ``tr``. Exact and right-censored
  data were unaffected; otherwise the EM converged to a curve that was not
  the NPMLE (one failure in (1, 2] and one unit censored at 1.5 were fitted
  at a likelihood of 0.375 instead of 1). On 600 random small data sets the
  old fits fell up to 1.4 log-likelihood units short without right
  truncation and 25 to 68 with it; some right-truncated fits had likelihood
  zero, and a few doubly truncated ones raised ``IndexError``. Every fit
  whose NPMLE exists now matches an independent maximisation to 2e-9, and
  the ``npmle`` verdict, now built on the corrected supports, agreed with
  the EM's behaviour on all 399 of those data sets where it gave a firm
  verdict. Delayed-entry data in which a unit is censored before a later
  unit enters is now reported ``"not unique"`` (with a warning), since the
  mass between them is not determined; the Kaplan-Meier option still
  returns the delayed-entry Kaplan-Meier. The Nair interval example in the
  docs rises from -59.52 to -58.06 in log-likelihood.
- **Formula models refuse a category level they were not fitted with
  (#371).** Predicting for a level absent from the fitted data coded it
  silently as the reference level (``WeibullPH`` gave S(5) = 0.5283 for both
  ``g="a"`` and an unknown ``g="d"``), with only formulaic's
  ``DataMismatchWarning``. Every family that takes a ``formula``
  (parametric PH/AFT/PO/AH, ``AcceleratedLife``, ``CoxPH``,
  ``AdditiveHazards``, Buckley-James, frailty, competing-risks Cox and
  Fine-Gray) now raises a ``ValueError`` naming the column and the unknown
  levels, fitted or restored. A fit whose data has a level outside its
  ``C(g, levels=[...])`` list raises too; declared levels count as known. A
  missing categorical value still predicts NaN in place, as a missing
  numeric one does. Buckley-James returned survival 0 for any missing
  covariate and now returns NaN.
- **Competing-risks Cox predicts from a DataFrame (#370).**
  ``CompetingRisksProportionalHazards`` read a DataFrame by column
  position: with ``Z_cols=["z", "w"]``, passing the columns as ``[w, z]``
  changed S(5) from 0.655 to 0.914, and a ``formula`` fit could not expand
  raw covariates at all. ``sf``, ``ff``, ``Hf``, ``hf``, ``df``, ``cif``,
  ``phi`` and ``phi_e`` now select and encode the columns recorded by
  ``fit_from_df``, as ``CoxPH`` does, fitted or restored. Arrays work as
  before.
- **Proportional odds fits time-varying covariates (#372).** ``PO(dist)``
  models could be evaluated along a step covariate path but not fitted to
  one; the docs said PO lacked the structure. It does not: the PO hazard
  :math:`h_0 / (F_0 + \phi S_0)` depends only on the time and the current
  covariate, so splitting a subject into delayed-entry intervals is exact.
  ``fit_tvc``, ``fit_tvc_timeline`` and their ``_from_df`` forms now work
  for ``WeibullPO`` / ``PO(dist)``. On simulated step-path data (8 x 2,000
  subjects, truth [10, 2, 1, -0.5]) the mean estimate is
  [10.03, 1.98, 0.98, -0.50], and the fitted negative log-likelihood equals
  the path likelihood from ``sf_tvc`` / ``hf`` to about 1e-12.
- **``fit_tvc`` no longer truncates at time 0.** For PH, AH and PO models with
  a baseline defined below zero (Normal, Gumbel, Logistic), each subject's
  first interval was treated as left-truncated at 0, conditioning the fit on
  surviving to 0, so a constant covariate split into intervals did not
  reproduce ``fit`` (LogisticPO scale 5.22 against 9.33, NormalPH 5.56
  against 9.65). A first interval starting at 0 is now untruncated,
  matching ``fit`` and ``sf_tvc``. Baselines on the positive axis are
  unchanged.
- **Survival tree predictions for several subjects (#369).**
  ``SurvivalTree.sf(x, Z)`` (and ``ff``, ``df``, ``hf``, ``Hf``) routed a
  covariate matrix by a row, ``Z[split_index]``, instead of a column. With
  one covariate every subject silently got the first subject's curve
  (S(5) = 0.8811 for all rows, where row by row gives 0.2955 for half of
  them); with two or more it raised. ``survival_probability``, and so the
  Brier score and AUC, were wrong for a single tree. Each row now goes to
  its own leaf, and a 2-D ``Z`` returns an ``(n_rows, n_times)`` grid equal
  to stacking the per-row results, as ``RandomSurvivalForest`` does; a 1-D
  ``Z`` (one subject) is unchanged. The forest, already correct, now
  evaluates the whole matrix in one call per tree: identical results, about
  3x faster in ``survival_probability`` and ``score``.
- **Turnbull decides from the data whether its estimate exists (#327).** A
  fit warned "not identifiable" when more than 90% of its mass sat on pieces
  some observation gains from and none pays for, or when the EM did not
  converge: a cut-off tuned on simulated samples. On samples whose estimate
  does not exist that share ranged from 0.11 to 0.99 depending on how far
  the EM had got, so half were caught only because they had not converged;
  other non-existent estimates (a delayed-entry Kaplan-Meier that drops to
  zero before a later entry, Lynden-Bell and doubly truncated exact data)
  were reported only as not converged, and flat likelihoods not at all. The
  new ``model.npmle`` is ``"exists"``, ``"not unique"``, ``"does not
  exist"`` or ``"undetermined"``, from a structural criterion checked
  before the EM runs: Vardi and Wang's graph condition for exact data, and
  a hazard-scale gap argument for one-sided truncation with any censoring.
  It takes a few milliseconds on thousands of rows, and the warnings name
  the case and the time involved. On 240 simulated left-truncated samples,
  all 63 "does not exist" fits drifted to the boundary and none of the 176
  "exists" fits did. With censoring and truncation on both sides existence
  can depend on the counts, and such fits are reported as
  ``"undetermined"``. The fitted estimate is unchanged, and
  ``exploitable_mass`` is still reported as a diagnostic.
- **Every regression formula round-trips through serialisation (#244).** A
  model fitted with ``fit_from_df(..., formula=...)`` refused ``to_dict``
  for wrapped categoricals (``C(g)``, ``C(g, levels=...)``,
  ``C(g, contr.sum)``) and fitted transforms (``scale``, ``center``,
  ``poly``, ``bs``, ``cs``). It restored integer-level categoricals with
  string levels, so every row was coded as the reference level (sf off by
  up to 0.09), and Cox and competing-risks Cox models lost a ``0 +`` from the
  formula, so a restored model had 3 design columns for 4 coefficients and
  could not predict. ``to_dict`` now stores each factor's levels, in order
  and with their types, and each transform's fitted state as strict JSON;
  ``from_dict`` rebuilds the same design-matrix transformer, and restored
  models predict identically (rtol 1e-12) across the PH/AFT/PO/AH,
  accelerated-life, Cox, Lin-Ying, Buckley-James, frailty and
  competing-risks families. A formula is checked when saving, so anything
  that cannot be restored raises in ``to_dict``. A formula that SurPyval
  0.20 cannot rebuild is stamped schema 2, so 0.20 asks for an upgrade
  instead of failing with a formula error; plain columns and string
  categoricals stay schema 1. Old files still load, and a Cox file missing
  its ``0 +`` is repaired. Also fixed: predicting with plain integers for a
  column fitted as an integer ``Categorical`` treated it as numeric (sf
  0.372 instead of 0.083).
- **Brier score and time-dependent AUC with tied event and censoring times
  (#365, #290).** The censoring survival :math:`\hat G` behind the
  inverse-probability-of-censoring weights counted an event as still at risk
  of being censored at its own time, and weighted it by
  :math:`1/\hat G(x_i)`. The metrics now use the events-first reverse
  Kaplan-Meier (as ``prodlim`` and scikit-survival) and weight an event by
  :math:`1/\hat G(x_i-)` (as ``pec``; Gerds and Schumacher 2006). On a data
  set whose true values are known exactly, the Brier score at t = 2 was
  0.2330 against a true 0.2250 (now exact) and the AUC 0.6703 against 2/3;
  in simulation with discrete times the old Brier score was biased by
  -0.029 and is now unbiased (scikit-survival's :math:`1/\hat G(x_i)`
  weighting gives +0.011). Without such ties the results are unchanged and
  equal scikit-survival's. ``censoring_survival`` gains ``ties=``; Fine-Gray
  keeps its ``cmprsk`` convention. Also: ``integrated_brier_score`` sorts an
  unsorted grid (0.1908 became 0.1949 on one example); ``c`` must be 0 or 1
  and match ``x`` in length (a left-censored row was scored as a survivor);
  ``x_train`` needs ``c_train``; and a horizon that needs the training
  :math:`\hat G` where it has fallen to 0 scores NaN rather than being
  biased towards 0 (0.179 against a true 0.25).
- **Proportional odds along a time-varying covariate path (#236).**
  ``sf_tvc`` / ``Hf_tvc`` raised ``NotImplementedError`` for ``PO(dist)``
  models. The PO hazard :math:`h_0 / (F_0 + e^{\beta'z} S_0)` depends only on
  the time and the current covariate, so the cumulative hazard along a step
  path is exactly the sum of the constant-covariate increments, as for PH;
  PO now takes that path. It matches a numerical integral of the hazard to
  1e-9, and a constant path gives ``sf(x, Z)`` to 1e-13. PO's ``Hf`` is now
  computed as :math:`H_0 - \ln\phi + \ln(F_0 + \phi S_0)` rather than
  ``-log(sf)``: before, a change-point where the baseline survival underflows
  made every ``sf_tvc`` value NaN (``WeibullPO`` with a change at t = 1500:
  S(5) was NaN, now 0.8387). Time-varying *fitting* is still not available
  for PO.
- **``sf_tvc`` for PH and AH with a baseline defined below zero.** For a
  Normal, Gumbel or Logistic baseline a constant covariate path gave the
  survival conditional on surviving to time 0, not ``sf(x, Z)`` (at x = 5:
  PH(Normal) 0.91842 against 0.91551, PH(Gumbel) 0.90613 against 0.87800).
  The first segment now starts at the bottom of the support, so a constant
  path reproduces ``sf`` exactly.
- **Recurrent-event simulations are much faster (#362); seeded results
  change.** ``mcf``, ``plot``, ``time_terminated_simulation``,
  ``count_terminated_simulation`` (and their ``..._data`` versions) and the
  renewal models' ``cramer_von_mises`` bootstrap used to simulate one item
  and one event at a time, with two model calls per event. Every item is
  now advanced together, one event per round, with one array operation per
  round, and the renewal models' root finding is vectorised too. An
  ``mcf`` over 1000 items is 23-46x faster for the Kijima and ARA models
  (``GeneralizedRenewal`` with a Weibull lifetime and Kijima II: 2.0 s to
  0.06 s), 9-18x for ARI and the intensity models, and 4x for the already
  cheap G1 model. The simulated MCF also no longer
  computes the Lawless-Nadeau variance it then discarded, which was most of
  the time for the intensity models with many items. The draws follow the
  same processes (checked against one-sequence-at-a-time references to
  round-off), but the uniforms are assigned to events in a different order,
  so a given ``seed`` now gives different simulated values and bootstrap
  p-values than in 0.20. The unused uniform-pool helpers
  ``initialize_simulation``, ``get_uniform_random_number`` and
  ``clear_simulation`` are removed. Three examples in :doc:`Recurrent Event
  Modelling with SurPyval` use new seeds so that they still illustrate
  what the text describes.
- **One seeding rule for every random draw (#361).** With the default
  ``random_state=None`` (or ``seed=None``), the non-parametric
  ``random()`` and ``bootstrap_cb()``, ``ParametricCompetingRisks.random()``,
  the copulas' ``sample_uv()`` (and so ``random()``), the degradation
  models' ``random()``, ``induced_life()``, ``predict_rul()`` and bootstrap
  bounds, the Buckley-James bootstrap and the recurrent-event goodness-of-fit
  p-values used a fresh OS-seeded generator on every call, so
  ``np.random.seed`` had no effect on them while it did control
  ``Parametric.random`` and the recurrent simulations. ``None`` now draws
  from numpy's global RNG throughout (``surpyval.utils.rng.as_generator``).
  An explicit seed or ``Generator`` gives the same stream as before. See
  :doc:`Conventions`.
- **``import surpyval`` no longer imports matplotlib (#363).** pyplot is
  imported inside the plotting methods, which saves about 0.3 s on every
  cold start of a program that never plots. Plotting is unchanged.
- **Design changes approved after the third documentation review.**

  - **One sample size for BIC and AIC_c.** Every model that reports a BIC
    or AIC_c uses the number of observed failures (exact, left- and
    interval-censored, weighted by their counts), or the number of
    observations when there is none. Recurrent-event models count
    observed events; copulas count rows in which at least one series
    failed. Previously univariate models counted failures for BIC but all
    units for AIC_c, regression counted exact failures only (``-inf``
    without one), recurrent models counted exact events (NaN without
    one), and copula and Royston-Parmar models counted every row.
    Univariate BIC on exact and right-censored data is unchanged; AIC_c on
    censored data now uses the failures. Restored models store the sample
    size (``"ic_n"``), so ``bic()`` and ``aic_c()`` work without the data.
    ``fit_best(metric="aic_c")`` raises a clear error when no candidate
    has a finite AIC_c instead of returning ``None``.
  - **Fine-Gray follows cmprsk on tied times.** The censoring weights are
    :math:`\hat{G}(t-)/\hat{G}(x_i-)`, with the censoring Kaplan-Meier read
    just before each time, as in R's ``cmprsk::crr``. Results are
    unchanged when no censoring time equals an event time.
  - **Readings from a coarse gauge.** ``GammaProcess.fit(gauge=...)`` (with
    ``rounding``, ``exact_start`` and ``gauge_method``) maximises the
    probability that each unit's path passes through its recorded gauge
    bins. With a gauge step near the mean increment, taking rounded
    increments at face value more than doubled ``alpha`` and shrank
    stress coefficients; the gauge likelihood recovers the unrounded
    estimates. The default fit is unchanged.
  - **Scale-equivariant parametric fits.** Every continuous distribution
    and method, with or without an offset, now gives the same answer in
    any units from 1e-4 to 1e5: the search is scaled per coordinate, MLE
    normalises its objective per observation, MOM uses the scaled search,
    MPS no longer evaluates the CDF at the support edge, and offsets start
    one data spacing (not one unit) below the smallest value, with MPP
    searching the offset in the data's spacing. At a data scale of 1e-3,
    Rayleigh MOM had been 1.2% off, Uniform MPS 0.15% and Beta4 MLE 0.1%,
    and many offset fits never left their start. Fits in their own units
    move by about 1e-6 relative, towards the optimum.
  - **Strict-JSON serialisation.** ``to_dict``/``to_json`` no longer emit
    ``NaN``/``Infinity``: non-finite values are written as ``null`` and
    listed under ``"non_finite"`` (JSON Pointers by kind), and every reader
    restores them. Each file is stamped with the oldest schema version that
    reads it: 2 when it records non-finite values this way, and 1 (the
    layout SurPyval 0.20 reads, which loads it identically) otherwise;
    older dictionaries and files still load. ``to_json(path, with_data=True)`` works for ``Parametric``
    and ``NonParametric``, and every class-level ``from_dict`` applies the
    package reader's checks.
  - **Non-parametric copula margins.** Under ``how="IFM"`` a margin can be
    ``KaplanMeier`` (or any fitted non-parametric model), giving the
    semi-parametric estimator of Genest, Ghoudi and Rivest (1995); it
    used to crash.
  - **Datasets.** ``load_framingham``, ``load_pbc2`` and
    ``load_support2`` expose bundled data that had no loader; the
    undocumented ``synthetic_dataset.csv`` is removed. The G1 example data
    are credited to Kaminskiy and Krivtsov (2010).

- **Bug fixes found in the third documentation review.** This review
  probed the documented behaviour adversarially (identities, round trips,
  cross-method agreement, edge cases). Each fix has a regression test
  that fails on the old code.

  *Wrong results that are now correct.*

  - **Degradation:** the default two-sided analytic ``cb`` band was a 90%
    band (each side used the full ``alpha_ci``). Units already past the
    threshold at their first measurement were treated as survivors; they
    are left-censored there. Wiener ``sf``/``ff`` returned NaN for low
    noise; ``GammaProcessModel.mean`` could be negative; zero Gamma
    increments are censored below a ``resolution`` instead of a 1e-12
    nudge.
  - **Gray's test** is Gray's (1988) statistic with group-specific
    censoring; the pooled version rejected a true null up to 89% of the
    time when groups were censored differently.
  - **Parametric competing risks:** ``cif`` and ``probability_of_cause``
    are integrated per query instead of on a fixed grid (they could sum
    to 0.81, or 0.0005).
  - **Recurrent events:** GRP/ARA simulation was wrong at long horizons
    and NHPP simulation failed past ~745 expected events; renewal fits
    could keep a worse optimum than one they found (boundary optima such
    as ARA ``rho -> 1``); left-censored counts now cover ``(tl, x]``.
  - **Regression:** a missing or infinite covariate made PH/AH/frailty
    return their starting values (rows are now dropped with a warning in
    every fitter); stratified Cox ``predict_tvc`` used the first
    stratum's baseline; Lin-Ying predictions depended on covariate
    centring; counts were treated as clusters in robust standard errors,
    ``check_ph`` ranks and the Buckley-James bootstrap; some AFT/PO fits
    stopped short of the maximum (``PO(Weibull)`` by 5.5 nats) and are
    finished by a gradient-based optimiser.
  - **Parametric:** ``cs`` ignored ``p``, ``f0`` and the offset;
    likelihood-ratio bands collapsed onto the estimate when the inner
    search failed (Geometric coverage 0.65); the mixture EM stalled on a
    ``log(0)``; ExpoWeibull and ``CustomDistribution`` moments were wrong
    away from unit scale; MPS and MSE were not scale invariant; raw
    distribution functions were evaluated outside their support;
    ``bic()`` was ``-inf`` without exact failures; LogNormal and Gamma
    hazards overflowed in the far tail.
  - **Non-parametric:** ``qf``/``median`` had no round-off tolerance (the
    median of 1..30 was 16); ``band()`` critical values were ~1.5% low;
    log-rank counted groups never at risk in its degrees of freedom; the
    Turnbull identifiability warning fired on correct fits.
  - **Copulas:** Frank overflowed for :math:`\theta \gtrsim 37` and
    Clayton collapsed at extreme :math:`\theta`; joint MLE dropped
    pre-fitted margins' options; automatic derivatives summed over
    broadcast axes.
  - **Data layer:** NaN truncation bounds were read differently by each
    fitter; unsorted xrd input gave a wrong estimate.

  *Crashes and unclear errors.* Two-column ``x`` without intervals now
  works in every fitter; truncation rules are identical for one- and
  two-column ``x``; bad ``fixed``/``init``/``bound``/``how`` arguments,
  wrong covariate row counts, degenerate data, out-of-range parameters
  in ``from_params``/``fit_from_parameters``, non-integer data for
  discrete distributions and corrupt serialised dictionaries raise clear
  errors. Models restored without their data explain what needs it.

  *Serialisation.* Discretize, ``CustomDistribution`` (after
  re-construction), ``NeverOccurs``/``InstantlyOccurs``, destructive
  degradation models with any distribution, and competing-risks models
  with mixed or tuple labels round-trip; Cox dictionaries are strict
  JSON; likelihood-ratio bounds work after a ``with_data`` restore.

  *Behaviour changes to note.* BIC's sample size for univariate models
  counts every non-right-censored failure; ``aic_c`` is NaN when
  :math:`N \le k + 1`; recurrent covariates must be constant within an
  item; Cox ``model.phi`` is a method; stratified ``sf_tvc`` requires
  ``stratum=``; ``band()``'s ``n_sims``/``random_state`` are deprecated;
  two-column ``x`` without intervals is stored as one column.

- **Bug fixes found in the second documentation review.** Each was
  reproduced first and has a regression test that fails on the old code;
  documentation describing the old behaviour was updated.

  *Non-parametric.*

  - **Turnbull confidence bounds stepped a piece too early.** The variance
    at each value included the next piece's expected failures, so ``cb()``
    on interval-censored data gave ``[0, 1]`` where the estimate was 1. The
    ``r``, ``d`` and variance ladders now line up with ``R``; the
    Kaplan-Meier option without truncation starts the EM on Turnbull's
    innermost intervals; MPP fits with ``heuristic='Turnbull'`` pair each
    failure with the CDF after its drop (their estimates move slightly).
  - ``smoothed_hf()`` works on Turnbull models; Turnbull ``bootstrap_cb()``
    refits with the fit's ``tol`` and ``max_iter``; Greenwood's variance no
    longer blows up (about 1e14) from round-off at the last value.
  - ``'Benard'`` plotting positions use Benard's (i - 0.3)/(N + 0.4); an
    unknown ``turnbull_estimator`` raises a ``ValueError`` up front.

  *Parametric.*

  - **Distribution parameters named** ``p`` **clashed with the
    limited-failure proportion.** ``param_cb('p')`` on a Geometric or
    NegativeBinomial now bounds the distribution's own ``p``, and
    ``lfp=True`` works for them (the proportion is named ``lfp_p``).
  - **Uniform MLE with censoring was not the maximum.** ``(min, max)`` is
    used only for exact data; censored data gets a bounded search (``b =
    24.75``, not 10, in the reported example).
  - **Method of moments could return its start as the fit.**
    Beta-Geometric MOM has a closed form, non-finite moments at the start
    raise, and the search backs away from regions without moments. With
    ``fixed``, MOM matches one moment per free parameter.
  - ``CustomDistribution`` gains finite moments and ``var()``, a fast MOM,
    and the standard MPP refusal; ``offset=True`` is refused for discrete
    distributions; ``param_cb('gamma')`` and unknown names give clear
    errors; ``var()`` of LFP and zero-inflated models follows ``mean()``'s
    convention; the MSE/MPS fallback ends on Nelder-Mead as documented.
  - The gradients of the incomplete gamma/beta helpers had the wrong shape
    when a scalar argument met an array partner (NegativeBinomial fits
    that differentiate the CDF crashed).

  *Information criteria.*

  - **AIC, AICc and BIC count only estimated parameters.** Fixed
    parameters (and the accelerated-life placeholder) no longer add to
    ``k``, in parametric and regression models; restored models agree.
    A Weibull with its shape fixed now scores the same as the equivalent
    Rayleigh, and ``AcceleratedLife(Weibull, Power)`` AICs drop by 2.
  - Recurrent models use one BIC sample size, the number of exactly
    observed events (it was every row, or only failures for ARI).

  *Regression.*

  - **The exact Cox tie methods are fast.** ``'kp'`` uses the
    Gail-Lubin-Rubinstein recursion and ``'exact'`` the DeLong-Guirguis-So
    integral, with analytic score and information: a 107-way tie that took
    over nine minutes fits in 0.06 s, and the twelve-tie cap is gone.
  - **Cox silently fitted left-censored rows as right-censored** and
    failed on interval rows; both are refused with a clear error.
    ``CoxPH.fit_from_df`` gains ``tl_col``, and strata labels follow the
    missing-covariate row mask.
  - **A parametric regression fit could return its starting values** when
    the start's log-likelihood was infinite; it now falls back to the
    default start with a warning, or raises.
  - ``AH(...).random`` works with several covariate rows; the unreachable
    accelerated-life ``(low, high)`` stress option is removed and a scalar
    stress works; additive-hazards fits held at the positivity boundary
    warn; ``FrailtyModel`` gains ``neg_ll``/``aic``/``bic``/``aic_c`` and
    is warning-free near :math:`\theta = 0`.

  *Competing risks and copulas.*

  - ``gray_test`` counted a NaN cause as a competing failure (only
    ``None`` was censored); it uses the shared missing-cause rule.
  - ``CompetingRisksProportionalHazards`` (Cox and Fine-Gray) can be saved
    and restored, reproducing every prediction.
  - Copula fits start strictly inside the family's bounds and take a public
    ``init``; ``CopulaModel`` reports ``log_likelihood``, ``neg_ll()``,
    ``aic()`` and ``bic()`` from the full censored and truncated
    likelihood.

  *Recurrent events.*

  - **The cause-specific MCF's bounds were too narrow**: it used the
    per-step variance; each cause now gets the Lawless-Nadeau robust
    variance. The per-step variance itself was wrong with ties.
  - The non-parametric MCF accepts right truncation (``tr``); models
    without data raise an informative error; G1, ARI and NHPP fits are
    warning-free (NHPP searches run on an unconstrained scale, moving
    fitted values only within the optimiser tolerance); the renewal
    goodness-of-fit bootstrap resimulates each item as it was observed.

  *Degradation.*

  - Offset-exponential models could not be reloaded; every built-in path
    now round-trips. ``DestructiveDegradationModel`` gains
    ``to_json``/``from_json`` and keeps its data, so a reloaded model's
    ``cb`` matches the original.

- **Bug fixes found while rewriting the documentation.** Each was
  reproduced first, is covered by a new test, and the documentation that
  described the old behaviour (or a workaround for it) has been updated.

  *Regression.*

  - **Cox predictions before the first event were wrong.** The baseline
    lookup index was -1 there, which wrapped to the *last* baseline value,
    so ``Hf(0.1)`` returned the end-of-data cumulative hazard and ``sf``
    was near 0 where it should be 1. It is now 0 before the first event.
  - **Cox predictions paired unsorted times with the wrong covariate
    rows.** The query times were sorted for the baseline lookup but
    ``phi(Z)`` was not, so ``Hf([3, 1], Z)`` gave ``[9.69, 0.39]`` instead
    of ``[1.87, 2.03]``. Times and rows now stay paired, in the order
    given.
  - **The gamma-frailty fit broke when there was little frailty.** Its
    group log-likelihood subtracted terms of size
    :math:`(1/\theta)\log(1/\theta)` to get a difference of order
    :math:`\theta`; as :math:`\theta \to 0` round-off swamped it and the
    fit chased the noise (``neg_ll`` of -1985 against the PH fit's 755) or
    divided by an underflowed :math:`\theta`. It is now computed in a
    form that tends to the PH contribution, and the fit coincides with the
    ordinary PH fit when there is no frailty.
  - **A 1-D ``Z`` raised ``IndexError``** in the PH, AFT, PO and AH
    fitters; it is read as one covariate, as the other fitters already
    did.
  - **Accelerated life with a Gamma baseline had the life model backwards.**
    Gamma's ``beta`` is a rate, so the life now enters as ``1 / life`` (as
    for the Exponential); a longer modelled life had meant a faster rate.
  - **PH random draws returned ``inf`` for a tiny hazard multiplier**; they
    now invert through the cumulative hazard.
  - Docstring: Schoenfeld residuals follow the input order of the event
    rows, not event-time order.

  *Competing risks.*

  - **``CompetingRisksProportionalHazards`` (``how="Cox"``) incidence could
    exceed 1.** Its CIF weighted each hazard increment by ``exp(-H)`` (the
    #278 defect, fixed for the non-parametric CIF but not here): total
    incidence reached 1.07-1.18 in small samples, and 118 in one. The
    weight is the product-limit survival, and the CIFs now sum to exactly
    ``1 - S``.
  - **``CompetingRisks(method="Kaplan-Meier")`` only changed ``S``;** ``sf``,
    ``ff`` and ``Hf`` still used ``exp(-H)``. They now follow the method
    (the default is unchanged; the method is serialised).
  - The causes of a Cox competing-risks model are sorted, so the row order
    of ``betas`` no longer depends on the hash seed. Docstrings for
    ``how``/``tie_method``, FineGray's ``c``, and API entries for
    ``FineGrayModel`` and ``GrayTestResult``.

  *Parametric.*

  - **Two-sided parameter bounds other than (0, 1) were ignored** (a spline
    knot bounded to (0, 50) was fitted at 89.7). Every finite interval is
    now enforced.
  - **Poor default starts reported success at poor optima.** A
    ``CustomDistribution`` started each positive parameter at 1, where for
    data on another scale the likelihood is flat to machine precision; it
    now starts from the best of a grid of magnitudes, with the old start as
    a second try. A limited-failure-population fit could settle on the
    worse of two optima (Meeker's data: ``p = 0.116``, ``neg_ll`` 302.9,
    instead of ``p = 0.0067``, 293.0); it is also started from the failures
    alone. An MLE fit without ``init`` keeps the best of these starts.
  - **Restored models:** ``neg_ll()`` and ``aic()`` work from the stored
    likelihood; ``bic()``, ``aic_c()`` and ``plot()`` explain that the data
    were not saved instead of raising ``TypeError``.
  - Docstrings for ``from_params``' ``p``, zero-inflation in ``qf`` and
    ``random``, ``tl``/``tr``, ``MixtureModel.loglike`` and the
    ``CustomDistribution`` example.

  *Non-parametric.*

  - **``Turnbull.fit`` modified its input.** Infinite interval endpoints
    were rewritten in the caller's array, so refitting the same array gave
    a different answer.
  - **Fleming-Harrington could return ``nan``** when the Turnbull EM's risk
    and death sets carried float noise (``1 + 2e-16``); near-integer sets
    are treated as integers.
  - Docstrings for ``turnbull_estimator`` (Fleming-Harrington, the default,
    was missing), ``hf`` (it returns increments, not a rate), and the
    Turnbull EM (the estimator option acts inside the EM too).

  *Multivariate (copulas).*

  - **IFM ignored counts and truncation in the margins.** The first stage
    now fits each margin with the row counts and its own series'
    truncation window.
  - **Clayton had a spurious likelihood maximum near** :math:`\theta = 0`.
    The closed form rounded to ``C = 1`` and density ``1/(uv)`` below
    :math:`\theta \approx 10^{-16}`, which negatively dependent data drove
    the fit into; computed through ``log1p``/``expm1`` it tends to the
    independence copula.
  - ``MultivariateSurpyvalData``: interval rows without ``xl``/``xr`` are
    rejected (the check could never fire), a single row of ``D`` censoring
    codes broadcasts, and no ``RuntimeWarning`` comes from the default
    infinite truncation bounds.

  *Recurrent events.*

  - **The MCF variance ignored within-item covariance.** It is now the
    Lawless-Nadeau robust variance; the per-step variance was about eight
    times too small when items differ in their rates.
  - **``mcf_cb(bound_type="normal")`` scaled the standard error by the
    estimate a second time**, giving far too wide, negative bounds; it is
    now ``M +- z SE``.
  - **``ProportionalIntensityNHPP`` stopped short of the optimum** (Duane
    baseline: 15-30 AIC worse than the same model as Crow-AMSAA). It starts
    from the covariate-free baseline fit and searches on an unconstrained
    scale; on one-event-per-item data it now reproduces Weibull PH exactly.
    Crow-AMSAA's ``beta`` is bounded below by 0.
  - ``CauseSpecificNHPP(dist=HPP)`` works (it raised ``TypeError``) and
    ``HPP`` has ``from_params``; model summaries say how the model was
    obtained instead of always "MLE"; a covariate dict of scalars and a
    1-D ``Z`` are accepted; ``CauseSpecificMCF.plot`` draws confidence
    bounds and honours ``confidence``/``plot_bounds``.
  - Docstrings: residuals are not exactly i.i.d. Exp(1) when an item's
    window closes before an event; the Rossi example passes ``i`` and
    ``c`` by keyword.

  *Degradation.*

  - **Bootstrap bounds failed on a reloaded model.** ``from_dict`` did not
    restore the fitter the refits need, so every refit failed; it is now
    recovered from the restored life model (the distribution, or the
    regression fitter of an accelerated model).

- **REML population fits are 20-60x faster.** ``population_method="reml"``
  -- the plain and stress-dependent (``links``) populations, linear and
  nonlinear paths -- now evaluates the same REML objective through the
  Woodbury identity (a ``p x p`` computation per unit instead of an
  ``n_i x n_i`` factorisation) and searches it by BFGS with a
  Nelder-Mead fallback. The Nelder-Mead search it replaces used an
  absolute function tolerance that round-off could prevent it from
  meeting, and then ran to its 20,000-evaluation cap in every
  Lindstrom-Bates iteration: a nonlinear fit whose units' time scales
  differ several-fold ran for more than ten minutes, and now takes
  0.05 s. The estimates move only within the optimiser tolerance (at
  most ~1e-5 relative across the fingerprinted REML fits, with the REML
  objective at the new optimum equal to the old to 3e-10); moments fits
  are bit-identical. The degradation test suite runs in half the time.

- **Accelerated degradation, Stage 2: stress-conditional predictions**
  (<#155>, second half). A model fitted with ``links`` now uses its
  stress-conditional path population, ``eta ~ N(D(z) gamma, Sigma)``
  on the link scale, for prediction:

  - ``predict_rul(x, y, Z=...)`` updates a new unit's trajectory
    against the population of units at *its* stress rather than the
    pooled population that mixes every tested stress. The posterior is
    taken on the link scale, so a log-linked rate stays positive, and
    ``posterior_mean``/``posterior_cov`` are reported there.
  - ``induced_life(Z=...)`` gives the Lu-Meeker induced failure-time
    distribution at any stress -- until now it was refused for every
    accelerated model. Outside the tested range the mechanism, not a
    curve through the pseudo failure times, carries the extrapolation.
    The induced distribution records the stress it was taken at
    (``stress``, serialised and shown in its ``repr``).
  - ``path_param_link_mean(Z)`` and ``path_param_median(Z)`` expose the
    population at a stress, the latter on the natural scale (the median
    of each parameter, exactly, since each link is monotone).

  A linked model requires ``Z`` in these calls and a model without
  ``links`` refuses it, each with a message naming the fix. Everything
  without ``links`` is bit-identical to before (66 fingerprinted
  outputs across ``predict_rul`` and ``induced_life`` on plain,
  nonlinear and Stage-1 accelerated models). The degradation theory
  page gains a section on the stress-dependent model and the how-to a
  worked example.

- **Accelerated degradation, Stage 3: step-stress process models**
  (<#155>). ``WienerProcess.fit`` and ``GammaProcess.fit`` take ``Z``
  (one stress row per measurement, the stress applied over the interval
  ending there) and ``stress_ref``, so constant-stress *and*
  step-stress accelerated tests can be fitted. Stress accelerates the
  process clock, ``AF(z) = exp(gamma'(z - z_ref))``: the process runs on
  the operational time ``tau(t) = integral of AF(z(s)) ds``
  (Whitmore & Schenkelberg, 1997), the fitted process parameters are
  the reference-stress values and the new ``gamma`` coefficients how
  strongly stress speeds degradation up (``z = 1/T`` gives Arrhenius).

  - ``ff``, ``sf``, ``df``, ``hf``, ``Hf``, ``qf``, ``mean``, ``random``
    and ``predict_rul`` take ``Z``: a single stress row, or a
    ``StepSchedule`` for a stress that changes over time. The life
    under any profile is closed form, ``F(t) = F0(tau(t))``, for both
    processes; for ``predict_rul`` the schedule starts now.
  - ``acceleration_factor(Z)`` and ``is_accelerated``; ``gamma`` and
    ``stress_ref`` are serialised and shown in the ``repr``.
  - For the Wiener process stress scales the diffusion along with the
    drift, the assumption that makes the model identifiable and the
    life closed form.

  A stressed model requires ``Z`` for predictions and a stress-free one
  refuses it. Fits and predictions without ``Z`` are bit-identical to
  before (84 fingerprinted outputs). The degradation theory page gains
  a section on time-varying stress and the how-to a step-stress worked
  example.

- **Accelerated degradation, Stage 3: step-stress general-path models**
  (<#155>). ``DegradationAnalysis.fit(..., Z=Z, acceleration="clock",
  stress_ref=...)`` lets a unit's stress change *during* its test.
  Stress speeds up the clock of every unit's path,
  ``AF(z) = exp(gamma'(z - z_ref))``: the path is the ordinary path
  model on the reference-stress time the unit has aged (the
  cumulative-exposure model, Nelson 1980, which for these paths is
  also the rate-based model -- damage carries over at a step). ``Z``
  has one row per measurement, the stress over the interval ending
  there.

  - ``population_method="moments"`` estimates ``gamma`` by profile
    least squares from the units whose stress steps (and refuses data
    with no steps); ``"reml"`` fits the mixed model -- the FOCE
    profile likelihood of ``gamma`` -- which also identifies it from
    units at different constant stresses.
  - The path parameters, their population and the pseudo failure
    times are on the reference-stress clock, the life distribution is
    fitted to those reference-stress lifetimes, and ``sf``, ``ff``,
    ``df``, ``hf``, ``Hf``, ``qf``, ``mean`` and ``random`` take ``Z``
    as a stress row or a ``StepSchedule``: ``F(t) = F0(tau(t))``.
  - ``gamma``, ``stress_ref``, ``acceleration_factor(Z)``; ``path(t,
    unit)`` and ``plot`` follow each unit's own stress history.
    Serialised and shown in the ``repr``.
  - Prediction for a monitored unit on its own clock: ``predict_rul``,
    ``predict_failure_time`` and ``predict_remaining_life`` take the
    unit's stress history as ``Z`` (one row per measurement, or one row)
    and the planned stress from its last measurement as ``Z_future`` (a
    row or a ``StepSchedule`` starting now; by default the last stress
    is held). The posterior is taken against the reference-stress
    population and each draw's failure time is mapped back to calendar
    time along the history and the plan.
  - ``induced_life(Z=...)`` under a stress row or a profile, and
    two-stage bootstrap bounds ``cb(..., Z=..., method="bootstrap")``
    that resample units with their stress histories and re-estimate the
    clock on every refit. The analytic correction and
    ``life_parameter_covariance`` are not derived for a clock model
    (its pseudo failure times also depend on the estimated clock) and
    say so, pointing to the bootstrap. ``links`` and ``path="best"``
    cannot be combined with the clock.

  The clock leaves every existing fit bit-identical (118 fingerprinted
  outputs of plain, REML, nonlinear, ``path="best"``, Stage-1 and ``links`` models,
  and the 84 process-model outputs); the only change is that the error
  for a ``Z`` that varies within a unit now points to
  ``acceleration="clock"``. The process models' clock moved to a shared
  module unchanged, and the REML step gained a Woodbury-identity variant
  used by the clock fit. The theory page gains a section on the
  accelerated clock and the how-to a step-stress worked example,
  including remaining life under two stress plans.

v0.20.0 (23 September 2026)
---------------------------

- **Offset moments are exact.** ``ParametricFitter._moment`` with an
  offset -- what the method-of-moments fit and ``Parametric.var`` use
  -- computed :math:`E[(\gamma + X)^n]` by integrating the shifted
  density to infinity with ``quad``, even for distributions whose
  moments have closed forms. It now takes the binomial expansion of
  the un-offset raw moments, as ``Parametric.moment`` already did: no
  quadrature for closed-form distributions, and no ``IntegrationWarning``
  on machines where the shifted integral hit ``quad``'s roundoff limit
  (which failed the warnings-as-errors documentation build for this
  release).

- **New distribution: Hypoexponential.** The sum of independent
  Exponential stages with distinct rates (the generalised Erlang),
  which is the lifetime of a load-sharing group or a warm/hot standby
  system -- anything that passes through several memoryless stages in
  series. ``Hypoexponential.from_params([r1, r2, ...])`` takes any
  number of stage rates and returns an ordinary ``Parametric`` model
  with that many parameters (``lambda_1 ... lambda_m``), so ``sf``,
  ``ff``, ``df``, ``hf``, ``Hf``, ``qf`` (bisection between exact
  exponential brackets), ``mean``, ``var``, ``moment``, ``entropy``,
  ``random`` (one exponential draw per stage), offsets, limited-failure
  and zero-inflated variants and ``to_dict``/``from_dict`` all come with
  it. The distribution functions also take the rates directly,
  ``Hypoexponential.sf(x, r1, r2, ...)``. Rates must be strictly
  positive and distinct: the partial-fraction coefficients blow up with
  alternating signs as two rates approach, so near-equal rates raise a
  clear error pointing at ``Gamma`` (equal rates are the Erlang). There
  is no ``fit``; construct it from known stage rates.

  To make a variable-parameter-count distribution deserialisable,
  ``ParametricFitter`` gained a ``_for_params`` hook (returns ``self``
  for every fixed-arity distribution) that ``Parametric.from_dict``
  consults, so the restored model reports the right ``k``.

- **Accelerated degradation, Stage 2: stress-dependent path
  parameters** (<#155>, first half). ``DegradationAnalysis.fit`` takes
  ``links`` alongside ``Z`` to model the degradation *mechanism*
  against stress rather than only the pseudo failure times: the path
  parameters named in ``links`` depend on the unit's stress on an
  ``"identity"`` or ``"log"`` link (a log-linked rate with ``Z = 1/T``
  is the Arrhenius relationship), the others are common, and a
  per-unit random effect sits on top -- ``eta_i = D(z_i) gamma + u_i``
  with ``u_i ~ MVN(0, Sigma)``. ``gamma`` and ``Sigma`` are estimated
  by the same two-stage (Lu-Meeker) or REML route as the plain
  population and stored as ``path_param_fixed`` (labelled by
  ``path_param_fixed_names``) and ``path_param_link_cov``; the fitted
  model round-trips through ``to_dict``/``from_dict`` and shows the
  fixed effects in its ``repr``. The life model is still the Stage-1
  regression on the pseudo failure times, so every existing prediction
  method is unchanged; the stress-conditional prior for ``predict_rul``
  and ``induced_life`` is the second half.

  Under the hood a :class:`LinkedPathModel` presents any path model on
  its link scale, so the per-unit fits and the FOCE linearisation apply
  unchanged, and the REML routines take an optional fixed-effects
  design (``a_mat_list`` / ``d_mat_list``). Without ``links`` the
  pipeline is bit-identical to before (verified by fingerprinting 55
  numeric outputs across the moments, REML, nonlinear-REML,
  best-path, Stage-1 ADT and bootstrap surfaces).

- **API reference completed for the remaining public surfaces**
  (<#141>). New autodoc pages for every distribution that had none:
  the discrete lifetimes (Geometric, Poisson, Binomial, Negative
  Binomial, Beta-Geometric, discrete Weibull and ``Discretize``), the
  per-demand and degenerate models (Bernoulli, FixedEventProbability,
  ExactEventTime, InstantlyOccurs/NeverOccurs), the continuous
  stragglers (Beta, Rayleigh, GumbelLEV) and the Royston-Parmar
  flexible parametric model. The competing-risks subpackage -- absent
  from the API tree entirely -- has a page (Aalen-Johansen
  ``CompetingRisks``, ``ParametricCompetingRisks``, ``FineGray``,
  ``CompetingRisksProportionalHazards``), as do model persistence
  (``from_dict``/``from_json``) and the recurrent trend tests
  (``laplace``, ``mil_hdbk_189c``). ``fit_best`` gained its first
  docstring and joined the comparison-and-validation page, and the
  regression pages now document the fitted-model classes
  (``ParametricRegressionModel``, ``AdditiveHazardsModel``).

  Fixed along the way: every recurrent-events API page still targeted
  the ``ARA_``-style shadow classes that the ``singleton_fitter``
  refactor removed, so their method documentation had silently dropped
  out of clean builds.

- **Third duplicate-code consolidation.** Another sweep for repeated
  definitions, this time at the small end (exact duplicates the earlier
  sweeps' size thresholds skipped, plus inline fragments):

  - Every ``from_dict`` opened with the same three-line "wrong dict"
    guard, written out 21 times with hand-composed messages. They now
    call ``require_model_tag`` (``surpyval.serialisation``), which
    raises the same "Must create ... from a <Tag> dict" ``ValueError``
    with the model tag always present. One message changed wording:
    ``FrailtyModel.from_dict`` said "from its own dict" and now names
    the tag like every other model.
  - The five models that persist covariate metadata (feature names,
    formula, formula terms -- <#244>) carried the same ``to_dict``/
    ``from_dict`` blocks; they now call ``serialise_covariate_meta``/
    ``restore_covariate_meta`` in ``regression_data``.
  - The ``exp(beta'Z)`` covariate link was still restated in six places
    after <#295> introduced ``LogLinearPhi``: the AFT fitter's private
    copy, PO's methods, the AFT time-varying-covariate fit's local
    class, PH's lambdas and the deserialiser's lambda. All now use
    ``LogLinearPhi``, whose two historical serialisation names (PH's
    ``e^`` vs AFT/PO's ``exp``) are class constants. The PH
    constructor's phi signature check now compares parameter names and
    kinds instead of the signature's string form, so the annotated
    shared function passes.
  - The optimiser objective every regression ``fit`` built inline is
    now ``make_objective`` in the fit skeleton, and the shared
    ``x - gamma`` probability-plot transform lives on
    ``HazardIdentitiesMixin``.
  - Small orphans: ``_check_has_data`` moved to
    ``LikelihoodInferenceMixin``; the NHPP baselines' identical
    all-ones ``parameter_initialiser`` became the ``IntensityModel``
    default; the two competing-risks ``fit_from_df`` helpers share
    ``optional_column`` in ``utils``.

  Verified by fingerprinting 101 numeric outputs across the touched
  surfaces on both sides of the change: all identical except the one
  reworded frailty message. The heavier parameterised rewrites found in
  the same sweep are filed as <#350>, <#351> and <#352>.

- **Type-hint ratchet: finished.** Every function in the package is
  annotated -- 1750 of 1750 defs -- and the per-module ratchet list is
  gone: ``disallow_untyped_defs`` now holds package-wide, with only the
  test suite and the alpha tree exempt (they are exercised, not typed,
  but are still checked against the package's annotations). <#143> can
  close.

  The final pass covered the remaining seven areas in sequence -- the
  regression fitters and ``cox_ph``, the ``utils`` wrangling surface and
  the ``autograd_gamma_compat`` shim, competing risks, the whole
  recurrent package, degradation, the copulas and the ``beta.ml``
  forest. The conventions are the ones the earlier passes established:
  ``Numeric``/``Boxable`` wherever autograd differentiates,
  ``npt.ArrayLike`` at user entry points and never in arithmetic,
  declaration blocks for fit-populated model attributes, and
  ``TYPE_CHECKING`` contract stubs where a mixin calls methods its host
  supplies.

  Typing the bodies kept finding things, as it has all along:

  - ``SemiParametricRegressionModel.neg_ll``/``jac`` were declared as a
    float and an array; they hold the fit's closures.
  - ``Parametric.random`` is documented and annotated to return an
    array, but returns xcnt-format ``(x, c, n, t)`` arrays for
    limited-failure-population and zero-inflated models -- now
    annotated and documented honestly.
  - ``singleton_fitter`` is typed ``(cls: type[T]) -> T``, so the
    checker finally knows every fitter singleton is an instance; one
    ``type(...)``-indirection call site simplified away.
  - ``NonParametricCounting.var`` is honestly Optional (simulated MCFs
    carry no variance), and three error paths that crashed with
    unpacking/attribute errors now raise named ``ValueError``\ s:
    ``BuckleyJamesModel.bootstrap_ci`` and destructive-degradation
    bootstrap bounds on models carrying no fit data, and ``mcf_cb`` on
    a simulated MCF.
  - The recurrent intensity contract moved off the documented
    ``ArrayLike`` trap onto the honest ``Boxable`` union -- those
    functions are differentiated by autograd in the NHPP likelihoods.
  - A handful of genuine signature divergences (the covariate-extended
    simulation methods, the named-single-parameter intensity and copula
    families against their variadic base contracts) are documented at
    the definition with targeted ignores rather than silently widened.

- **Type-hint ratchet: the regression package's shared plumbing and its
  two model classes.** Coverage moves to 1041/1747 (60%), tracked in
  <#143>. Typed in dependency order -- the base layer before the
  fitters that build on it, the same sequencing the parametric package
  used -- so the coming fitter pass starts from typed call boundaries
  instead of ``Any`` flowing in.

  Eight modules join the ratchet: ``_fit_skeleton`` (the fitting spine
  every family imports -- ``LogLinearPhi``, ``HazardIdentitiesMixin``,
  the prepare/assemble pair, both optimiser ladders and
  ``mirror_distribution``), ``_likelihood`` (the shared censoring- and
  truncation-aware negative log-likelihood), ``_bounds``,
  ``tvc_schedule`` (the ``StepSchedule`` machinery), the
  ``parametric_regression_model`` and
  ``semi_parametric_regression_model`` classes, and the ``PH``/``AH``
  factory ``__init__``\ s.

  The conventions carry over from the parametric package:
  ``Numeric``/``Boxable`` for the distribution-function surface
  (``HazardIdentitiesMixin`` and ``LogLinearPhi.phi`` are
  differentiated under autograd), ``npt.ArrayLike`` at the user entry
  points, and a ``TYPE_CHECKING`` contract block on
  ``HazardIdentitiesMixin`` declaring the ``Hf``/``hf`` its identities
  call -- the host class supplies them, and one that forgets still gets
  the ``AttributeError`` that names it.

  Three annotations followed the code rather than the reverse:
  ``prepare_regression_fit``'s ``phi_bounds``/``phi_param_map`` really
  are callables *or* static values (both branches are live);
  ``_safe_eval`` in the schedule expression interpreter returns
  ``float | bool`` because comparisons are values in that grammar; and
  ``_ic_counts`` matches the ``tuple[int, int]`` its mixin supertype
  declares. No behaviour changed -- annotations are erased at runtime,
  and the only body edits are local renames where a variable was
  reused with a second type (the ``segments`` accumulation list, the
  expression interpreter's comparison operator).

- **The structural duplicates: shared bases extracted where whole class
  bodies were copied.** The body-level sweep's deeper findings, where
  the fix is a base class or driver rather than a moved function.

  ``WienerProcessModel`` and ``GammaProcessModel`` now share
  ``FirstPassageProcessModel``: both reduce their failure-time
  distribution to one hook -- the probability the process has crossed a
  distance by time ``t`` -- and everything expressible in terms of that
  CDF (``ff``/``sf``, the hazard identities, the bracket-and-``brentq``
  quantile, ``predict_rul``, serialisation) had been written out twice,
  verbatim. The density, mean, sampling and repr stay per class: those
  genuinely differ.

  ``ARA``, ``ARI`` and ``GeneralizedRenewal`` shared their entire
  fitting spine -- multi-start Nelder-Mead over ``[restoration, *dist
  params]`` in the unconstrained transform space -- as three copies that
  the near-match pass scored at 0.84-0.94 similarity: already drifting.
  It is now ``RenewalFitMixin._fit_restoration_ml``; each family
  supplies its restoration parameter's name, bounds and start grid.
  ``GeneralizedOneRenewal`` keeps its own optimiser call deliberately:
  its likelihood needs only ``q > -1``, so it runs under box bounds
  rather than a transform.

  Two five-way wrapper stacks collapsed to dispatchers:
  ``ParametricRegressionModel``'s ``sf``/``ff``/``df``/``hf``/``Hf``
  carried the same coerce-resolve-evaluate body five times (now
  ``_eval``), and ``DegradationModel``'s five carried the same
  accelerated-or-plain dispatch (now ``_life_fn``). The named methods
  and their docstrings remain. The four regression fitters' ``__init__``
  blocks mirrored the same six distribution attributes verbatim; that is
  now ``_fit_skeleton.mirror_distribution``.

  Investigated and left where they are: ``hpp.fit`` and the NHPP
  fitter's ``fit`` (identical one-call wrappers over genuinely different
  fitting routines), the forest and tree prediction methods (already
  two-line delegations to each class's dispatcher -- the end state, not
  duplication), and the renewal ``fit``/``_refit`` wrappers (two-line
  delegations whose docstrings carry the per-family defaults).

  Behaviour was checked rather than assumed: 82 fingerprints -- the 47
  from the previous sweeps plus both process models' full surface
  (fit, all distribution functions, quantiles, RUL, seeded sampling and
  a serialisation round trip), all four renewal fits, and the regression
  and degradation models' five prediction functions -- are bit-identical
  before and after, with the baseline verified to import the pre-change
  code.

- **A second duplication sweep, this time by function body.** The first
  sweep matched helper names; this one normalised every function and
  method in the package at the AST level -- identifiers abstracted,
  docstrings stripped -- and compared the 1,483 non-trivial bodies for
  exact and near matches. Three findings were acted on; the rest are
  either deliberate parallels (the distribution API restates ``hf`` and
  ``Hf`` per class, each with its own closed-form docstring) or
  structural refactors queued with their areas (the renewal family's
  triplicated fitting loop, the two process-model classes sharing
  verbatim ``predict_rul``/``qf``/``ff``).

  **The legacy AFT fitter was dead code, and two of its methods lived on
  as orphans.** ``accelerated_failure_time/accelerated_failure_time.py``
  -- the pre-skeleton ``AcceleratedFailureTimeFitter``, 220 lines --
  was imported by nothing: the package ``__init__`` re-exports from
  ``aft_fitter``, and no test touches it. It carried the only *live*
  copy of ``_parameter_initialiser_dist``; the verbatim copies on the
  proportional-hazards fitter and the accelerated-life
  parameter-substitution fitter had no callers at all. All three are
  deleted along with the module.

  **The Bernoulli / FixedEventProbability split had copied its
  estimation machinery wholesale.** The 0.20.0 split gave each class its
  own verbatim ``fit``, ``from_params``, ``entropy`` and ``random`` --
  the largest exact duplicate in the package. They now share
  ``SingleProbabilityMixin`` (``distributions/_single_probability.py``,
  ratcheted from birth): one probability in ``(0, 1)`` fitted from 0/1
  data by a weighted mean is the same estimation problem for both
  models, while everything distributional -- ``sf``, ``ff``, supports
  and each model's own convention docstrings -- stays on the classes.
  Consolidating also fixed a copy-paste artifact:
  ``FixedEventProbability.from_params``'s docstring said "Create a
  Bernoulli model".

  **The support-respecting Wald transform existed twice.** The
  four-branch core of ``param_cb`` -- generalised logit for an
  interval-bounded parameter, log distance for one-sided, natural scale
  otherwise -- was verbatim between the recurrent-event inference mixin
  and the parametric regression model, each wrapped in its own parameter
  lookup. It is now ``utils.linalg.wald_bound_on_support``; both
  ``param_cb``\ s keep their lookup and delegate.

  Two hash matches were investigated and deliberately left: ``sf_tvc``
  and ``_prepare_Z`` are five-line and one-line wrappers over machinery
  that is already shared (``Hf_tvc`` genuinely differs per family;
  ``prepare_Z`` is common), and their docstrings carry per-family
  content worth keeping.

  Behaviour was checked rather than assumed: 47 fingerprints -- the 37
  from the previous consolidation plus ``param_cb`` on both a bounded
  distribution parameter and an unbounded coefficient, and the
  Bernoulli / FixedEventProbability ``fit``/``from_params``/``entropy``
  and seeded ``random`` -- are bit-identical against ``develop``, with
  the baseline run verified to import the pre-change code.

- **The duplicated numeric helpers are consolidated into two new utils
  modules.** A sweep of every module-level helper in the package found
  the same functions written repeatedly, three of them verbatim.

  ``surpyval.utils.linalg`` now holds the single copy of each:
  ``numerical_hessian``, ``delta_method_se``, ``bound_signs`` and
  ``log_transformed_cb`` were duplicated wholesale between
  ``recurrent.inference`` and ``univariate.regression._bounds`` -- the
  drift-prone verbatim-copy pattern that produced <#288> -- with two
  *further* hand-rolled Hessians (a different step rule, ``1e-5`` against
  cube-root-of-epsilon) on ``royston_parmar`` and the frailty fitter,
  which now pass their step explicitly. The ``inv``-then-``pinv``
  fallback, written out at seven call sites, is ``safe_inv`` and
  ``safe_quadform`` (the latter the ``u'V^{-1}u`` test-statistic shape
  shared by the log-rank and Gray's tests). The eigenvalue-surgery family
  from the degradation package -- symmetrise, ``eigh``, repair the
  spectrum, reconstruct, five sites in three flavours -- is
  ``psd_project``, ``psd_floor``, ``psd_precision`` and ``psd_root``,
  with each call site's own floor convention preserved as arguments.

  ``surpyval.utils.ipcw`` holds the censoring-distribution Kaplan-Meier
  (``censoring_survival``) and the right-continuous step lookup
  (``step_at``) that Gray's test, Fine-Gray and the prediction metrics
  each carried privately -- three copies of each, under three names
  (``_G_at``/``_step``/``_g_at`` for the same four lines). The copies had
  begun to drift: the metrics copy silently ignored count weights,
  consistent with its callers today but a trap for the next reuse. The
  shared implementation is weighted, with no ``n`` as the unweighted
  case. ``utils.validate_1d`` joins the wrangling helpers for the 1-D
  float coercion the metrics module had as ``_as_1d``.

  The bodies are transplants, not rewrites, and behaviour was checked
  rather than assumed: 37 fingerprints across every touched path --
  Gray's test (both ``rho``), Fine-Gray coefficients/covariance/CIF, the
  competing-risks PH wrapper, Brier/IBS/AUC, a three-group log-rank, Cox
  ``check_ph`` and dfbeta residuals, additive-hazards fit, Royston-Parmar
  and frailty covariances, Crow-AMSAA/HPP standard errors and bounds,
  WeibullPH ``sf``/``hf`` bounds, the degradation fit with its corrected
  life covariance, REML, and the seeded induced-life sample -- are
  bit-identical before and after. One caller-facing rename:
  ``delta_method_std_errors`` (the ``recurrent.inference`` spelling) is
  now ``delta_method_se`` everywhere, matching the regression package's
  name for the identical function.

  Both new modules are fully annotated and under the mypy ratchet
  (<#143>) from birth, so the coming ``utils`` typing pass types each of
  these once instead of three times.

- **A behavioural consistency sweep across the base distributions.** The
  previous sweep compared annotations; this one compares what the
  distributions actually compute. Every identity that should hold for all
  of them -- ``sf + ff == 1``, ``Hf == -ln sf``, ``log_df == ln df``,
  ``hf == df/R(k-1)``, ``qf(ff(x)) == x``, ``mean == moment(1)`` -- was
  evaluated across all twenty-three, and the disagreements chased down.

  **Six discrete distributions returned nonsense below their support.**
  Geometric, DiscreteWeibull, BetaGeometric and NegativeBinomial live on
  :math:`\{1, 2, 3, \dots\}`; Poisson and Binomial on
  :math:`\{0, 1, 2, \dots\}`. Their closed forms are algebraic and did not
  know where the support started, so evaluating one step below it gave
  ``Geometric.df(0) == 0.43`` -- a positive probability outside the
  distribution, growing without bound as ``k`` decreases --
  ``BetaGeometric.sf(-1) == 2.0``, a survival above one that ``hf``
  divided by, ``DiscreteWeibull.df(0) == 0.0355+0.5468j``, a *complex
  number* from a negative base to a fractional power, and NaN from the
  incomplete gamma and beta forms in Poisson and NegativeBinomial. The
  pmf now sums to one whether or not the sum starts below the support;
  it did not for three of them before.

  The fitter's interior check kept these values out of a likelihood,
  which is why nothing failed, but ``df`` and ``sf`` are public: anyone
  plotting a pmf from zero got them. Each is now guarded at the first
  mass point. The guards clamp the *input*, not just the result, so the
  discarded branch of the ``np.where`` never evaluates the invalid
  expression -- otherwise it still computes the NaN and warns before
  throwing it away.

  **Three quantile functions did not invert their own CDF.**
  ``Geometric``, ``DiscreteWeibull`` and ``BetaGeometric`` answered
  ``k + 1`` for a ``u`` that came straight out of their own ``ff``.
  :math:`F(k) = 1 - R(k)` is formed by cancellation, so recovering ``k``
  from it lands a few ulp above the integer and ``ceil`` rounds away from
  it. The first two snap a near-integer before the ceiling; the third
  compares with a relative slack in its bisection.

  **``BetaGeometric.moment`` reported finite values for moments that do
  not exist.** The survival decays as :math:`k^{-a}`, so
  :math:`E[T^m]` converges only for :math:`a > m` -- the condition
  ``mean`` already applied at :math:`m = 1`. A truncated sum cannot see
  divergence; at ``a = 2, b = 3`` it returned about 25 for a second
  moment that is infinite. It now returns ``inf``, and ``moment(1)``
  uses the closed form, so it agrees with ``mean`` exactly rather than
  to three decimal places.

  **Two distributions were missing methods that are well defined.**
  ``FixedEventProbability`` had no ``Hf``, so ``log_sf`` and ``log_ff``
  -- which the base class writes in terms of it -- raised
  ``AttributeError`` instead of returning constants. Its ``df``, ``hf``,
  ``qf`` and ``mean`` remain absent deliberately: ``F`` is flat, so the
  mass is an atom rather than a density. ``Hf`` is the exception,
  exactly as for :class:`ExactEventTime`, whose ``Hf`` exists while its
  ``hf`` does not. ``ExactEventTime`` itself gained ``qf``, ``mean`` and
  ``moment``: a point mass has no density, but its quantile is ``T`` for
  every ``u``, its mean is ``T`` and its m-th moment is ``T**m``.

  **Binomial's support excluded two of its own outcomes.** ``support`` is
  a pair of *exclusive* bounds -- ``_validate_fit_inputs`` rejects
  ``x <= support[0]`` and ``x >= support[1]`` -- so a distribution
  declares them one step outside its first and last mass points, which is
  why ``Poisson`` declares ``-1`` and ``Geometric`` declares ``0``.
  ``Binomial`` had ``Geometric``'s lower bound with ``Poisson``'s first
  mass point: ``0``, saying that zero events in n trials lies outside the
  distribution when its probability is 0.168 at n = 5, p = 0.3. ``fit``
  and ``from_params`` set ``[0, n]``, excluding n events as well. The
  bounds are now ``(-1, n + 1)``.

  Nothing had observed this: the check lives on ``OptimisedFitMixin``,
  which ``Binomial`` does not inherit -- it is one of the three
  closed-form distributions that validate their own inputs -- so the
  field was inert metadata that would have become live the moment
  anything else read it. All of its values are unchanged, which was
  checked: 18 fingerprints across both constructors are bit-identical.

  Behaviour *on* the support is unchanged and was checked rather than
  assumed: 58 fingerprints -- every function over its support for all six
  discrete distributions, plus each one's fitted parameters and
  ``neg_ll`` fitted plain and right-censored -- are bit-identical before
  and after. The only intended change is ``BetaGeometric.moment``. Nine
  new tests -- 37 cases once parametrised across the distributions --
  cover the below-support behaviour, the pmf total, the quantile round
  trip, the divergence rule and the support bounds.

- **A consistency sweep across the base distributions.** With every
  distribution now annotated, the annotations themselves could be read
  as data and compared. Ten argument slots and thirteen returns
  disagreed across the twenty-two modules -- drift from having typed
  them a batch at a time rather than a deliberate difference.

  Most of it was cosmetic and is now uniform. The three ``mpp_*``
  transforms take an ``npt.NDArray``: every call site in the package
  passes one, eight of the fifteen implementations index their
  argument, and probability plotting is a least-squares regression on
  plotting positions that is never differentiated, so the input is
  never an autograd box and never a scalar. Their returns stay
  ``Boxable``, because the bodies delegate to ``qf``; narrowing them
  would mean changing code to suit a type hint, which is the wrong way
  round. ``random`` returns an ``npt.NDArray`` everywhere --
  ``Geometric`` and ``DiscreteWeibull`` returned ``self.qf(...)``
  straight through, and now wrap it, which is honest for the same
  reason in reverse: ``qf`` is ``Boxable`` because a fit differentiates
  it, and sampling never does. ``_mom`` is ``tuple[float, float]``
  throughout.

  One difference was a real error rather than an inconsistency.
  ``Numeric`` and ``Boxable`` both exclude ``list``, and ``fit`` and
  ``from_params`` were typed with them on four distributions -- yet
  every one of those accepts a list, as their own docstring examples
  show (``Binomial.from_params([5, 0.3])``). These are the entry points
  a user reaches for with whatever data they have. They are now
  ``npt.ArrayLike``, which is the correct type here precisely because
  the value is converted with ``np.asarray`` on the first line rather
  than used in arithmetic. ``Binomial.from_params`` already had it
  right; ``Bernoulli``, ``FixedEventProbability`` and
  ``ExactEventTime`` did not.

  Eight differences remain and each is deliberate:
  ``ExactEventTime``'s ``sf``, ``ff``, ``df``, ``hf`` and ``Hf`` return
  the narrower ``npt.NDArray``, which is a stronger promise rather than
  a broken one -- they are step functions built with ``np.atleast_1d``
  and provably return a real array -- and ``ExpoWeibull.unpack_rr``
  returns three values where the two-parameter distributions return
  two.

  Five tests were added to the shared-signature guard, so a future
  distribution cannot reintroduce any of this: the distribution
  functions take a ``Numeric`` and return a ``Boxable``, parameters are
  ``Boxable``, the ``mpp_*`` family takes arrays, ``random`` returns
  one, and the user entry points accept array-likes. Twenty-two tests
  in that file now. No behaviour changed -- annotations are erased at
  runtime, and the two ``np.asarray`` wraps were checked to produce
  identical samples.

- **Type-hint ratchet: ``univariate.parametric`` is finished.** Coverage
  moves from 869/1760 (49%) to 995/1771 (56%), tracked in <#143>. Every
  module in the package -- the fitters, the model, the base class and
  the mixture -- is now under ``disallow_untyped_defs``.

  Two structural additions came out of it, both of the same kind. A
  ``TYPE_CHECKING`` block on ``ParametricFitter`` now declares the
  distribution functions its own methods call -- ``cs`` divides two
  ``sf``\ s, ``log_sf`` negates ``Hf``, ``random`` inverts ``qf``, and
  the four ``ll_*`` methods are written in terms of ``hf``, ``Hf`` and
  the log densities. The class docstring already stated that contract in
  prose ("a distribution needs only ``hf`` and ``Hf``, or ``sf``, ``ff``
  and ``df``"); this is the same statement in a form the checker reads,
  and it mirrors the block ``OptimisedFitMixin`` already carried for the
  estimation machinery. Declared rather than defined, so a distribution
  that forgets one still gets the ``AttributeError`` that names it
  instead of a silently wrong inherited implementation.

  ``MixtureModel``'s fitted state -- ``data``, ``params``, ``w``, ``p``
  and ``loglike`` -- is annotated where it is initialised to ``None``.

  Three annotations had to follow the code rather than the reverse, each
  a small fact: ``probability_plot_data``'s ``ff`` is the failure
  *function*, not an array of values; ``bounds_convert`` returns five
  things, not three; and ``fallback_minimize``'s ``jac`` and ``hess`` are
  declared optional but are supplied by every caller.

  Where a value comes back from scipy or autograd and genuinely has no
  narrower type -- the confidence-bound closures, the mixture's
  prediction inputs -- it is ``Any`` rather than ``npt.ArrayLike``. That
  is the same trap the ``Numeric``/``Boxable`` comment in
  ``parametric_fitter`` already documents: ``ArrayLike`` admits ``str``
  and ``bytes``, so arithmetic on it does not type check, and the
  ``np.asarray`` that clears the error destroys an autograd box.

  Behaviour is unchanged and was checked rather than assumed: four
  distributions fitted plain, right- and left-censored, interval
  censored, truncated, with a limited-failure population and with zero
  inflation, plus ``neg_ll``, ``aic`` and a two-component mixture fit --
  bit-identical before and after.

- **Type-hint ratchet: the remaining eleven distributions.** Coverage
  moves from 665/1760 (38%) to 869/1760 (49%), tracked in <#143>. Every
  distribution module is now under ``disallow_untyped_defs`` except
  ``general_log_linear``'s counterpart concerns (<#345>).

  ``rayleigh``, ``beta``, ``beta4``, ``gamma``, ``gumbel``,
  ``gumbel_lev``, ``loglogistic``, ``exponential``, ``uniform``,
  ``degenerate`` and ``expo_weibull`` -- 202 signatures. The bulk was
  mechanical, generated from each distribution's own ``param_names`` so
  that ``x`` is a ``Numeric``, a parameter is a ``Boxable`` and the
  return follows the method. What was not mechanical were the places the
  generated guess was wrong, and each of those is a small fact about the
  code:

  - ``Rayleigh.mpp`` and ``Exponential.mpp`` treat the output of
    ``mpp_y_transform`` as an array -- indexing it, and passing it to
    ``np.polyfit`` and ``np.linalg.lstsq`` -- while the transform is
    declared to return a ``Boxable``. Wrapped at the call site rather
    than widening the transform, which is shared.
  - ``Gamma._moment_estimate`` and the two ``_mom`` helpers return
    2-tuples, not arrays.
  - ``Exponential._closed_form_mle`` and ``Uniform._closed_form_mle``
    return ``None`` when the closed form does not apply to the data, so
    they are ``npt.NDArray | None``.
  - ``ExpoWeibull.unpack_rr`` returns *three* values where every other
    distribution's returns two.
  - ``degenerate``'s classes inherit ``Distribution``, not
    ``ParametricFitter``, and its signatures have to match that
    supertype rather than the distribution convention.
  - ``ExpoWeibull._gumbel_seed`` reads ``gumb.res``, which a
    ``Parametric`` only carries after an MLE fit -- the branch that
    reads it is the one that asked for MLE, so it is annotated as
    deliberate rather than made unconditional.

  Behaviour is unchanged, and checked rather than assumed: every one of
  the eleven distributions was fitted by MLE, MPP, MSE and MOM, and its
  ``entropy`` and second moment evaluated, before and after. All 66
  results are bit-identical.

- **``Logistic`` ratcheted, and ``mgf`` made private.** ``Logistic`` was
  the only distribution with a public ``mgf``, which read as a method
  the other twenty-two were missing.

  It is not an orphan and is not removed: ``Logistic.moment``
  differentiates it ``m`` times with autograd to get the m-th raw
  moment, and the results are exact --

  .. code-block:: text

      Logistic(mu=3, sigma=2)
        moment(1) =   3.0000000000    exact  mu               = 3
        moment(2) =  22.1594725348    exact  mu^2 + s^2 pi^2/3
        moment(3) = 145.4352528131    exact  mu^3 + 3 mu s^2 pi^2/3

  The general closed form for a logistic raw moment needs Bernoulli
  numbers, so differentiating the MGF is both shorter and exact. What was
  wrong was its visibility: it is machinery for ``moment``, not part of
  the distribution surface. It is ``_mgf`` now, alongside the other
  private helpers on distributions (``_closed_form_mle``,
  ``_moment_estimate``, ``_gumbel_seed``). Nothing outside the class ever
  referenced it.

  The module is now fully annotated and added to the ratchet (#143).
  Two annotations had to follow the code rather than the other way
  round: ``mpp_y_transform`` indexes ``y``, so it takes an
  ``npt.NDArray`` rather than a ``Numeric`` that includes ``float``, and
  ``unpack_rr`` returns a *tuple* of two values, not an array -- both
  matching how ``Weibull`` already declares them.

  New tests pin the three low-order Logistic moments against the algebra
  rather than against another numerical method, check the variance comes
  out as :math:`\sigma^2\pi^2/3`, and assert that no distribution
  exposes a public ``mgf``.

- **Type-hint ratchet: the accelerated-life package, plus nine modules
  that were already complete.** Coverage across the package moves from
  611/1755 (35%) to 646/1760 (37%), tracked in
  <#143>.

  Nine modules were fully annotated but not listed under
  ``disallow_untyped_defs``, so nothing stopped them slipping back. They
  are listed now: ``fit_best``, ``utils.recurrent_utils``,
  ``utils.score``, ``recurrent.tests``,
  ``recurrent.parametric.counting_process``,
  ``univariate.regression.regression_data``,
  ``univariate.regression.tvc_fit``, ``univariate.regression.frailty``
  and ``distributions.fixed_event_probability``. Only
  ``counting_process`` needed work -- four ``*params`` that an AST scan
  counts as annotated and mypy does not.

  Eleven of the twelve accelerated-life modules follow, and locking them
  in turned up four real problems that annotations made visible:

  - **``GeneralLogLinear``'s constructor arguments were swapped.** The
    bounds lambda sat in the ``phi_param_map`` slot and the param-map
    lambda in the ``phi_bounds`` slot. Nothing consumed either, so it had
    no observable effect, but it would have bitten whoever finished the
    model. That module stays out of the ratchet: its ``phi_param_map``
    and ``phi_bounds`` are callables of the covariate dimension rather
    than the ``dict`` and ``tuple`` ``LifeModel`` declares, which is why
    it is already excluded from ``LIFE_MODELS`` (<#345>).

  - **``LifeModel.phi_bounds`` was annotated as a one-element tuple**
    while every caller passes two or three. Now variadic.

  - **Two dead branches around ``phi_init``.** The fitter chose between
    three shapes -- a ``"(Z)"``-only signature selected by comparing
    ``str(inspect.signature(...))``, the two-argument form, and a
    non-callable ``phi_init``. All ten life models are callable with
    ``(life, Z)``, so only one branch could ever run.

  - **``AcceleratedLife`` deserialisation accepted a distribution it
    cannot fit.** The guard established a ``ParametricFitter``, which
    admits ``Bernoulli``, ``Binomial`` and ``ExactEventTime`` -- none of
    them fittable. Since the dict is untrusted input, such a name got
    through and failed deep inside the fitter on a missing attribute; it
    now raises where the mistake is.

  ``hf`` is also declared in ``OptimisedFitMixin``'s ``TYPE_CHECKING``
  block, where ``sf``, ``ff``, ``df``, ``Hf`` and ``qf`` already were.
  Its absence was invisible until a typed caller reached for it.

  Behaviour is unchanged throughout: the accelerated-life fit,
  prediction, ``random`` and serialisation round-trip all produce
  bit-identical results before and after.

- **``Bernoulli.qf``.** The quantile function, added after the rest of
  the distribution::

      Bernoulli.qf([0.1, 0.7, 0.75, 0.99], 0.3)  ->  array([0., 0., 1., 1.])

  It inverts :math:`P(X \leq x)` -- the ordinary CDF -- stepping from 0
  to 1 at ``u = 1 - p``. On the open interval it matches
  ``Binomial.qf(u, 1, p)`` and ``scipy.stats.binom.ppf`` exactly. At
  ``u = 0`` those answer ``-1``, one below the support; this answers 0,
  the smallest outcome there is.

  It is deliberately *not* the inverse of this class's ``ff``, and that
  follows from the survival convention rather than being an oversight.
  ``R(x) = P(X \geq x)`` forces ``F(x) = P(X < x)`` if the two are to
  sum to one, and ``P(X < x)`` never exceeds ``1 - p`` anywhere on
  ``{0, 1}`` -- so once ``u`` passes ``1 - p`` no ``x`` in the support
  satisfies ``F(x) >= u``. The other discrete distributions, whose
  ``R(k)`` is ``P(X > k)``, do not have this split, and the usual
  ``ff(qf(u)) >= u`` check still holds for them. A test pins the
  difference in both directions so it stays a known consequence rather
  than becoming a surprise.

  What the definition does buy is the property worth having: ``qf(U)``
  for uniform ``U`` reproduces the distribution, which is how
  ``ParametricFitter.random`` samples. Tested at 200,000 draws, and at
  the degenerate ends ``p = 0`` and ``p = 1``.

- **BREAKING: ``Bernoulli`` is now a Bernoulli distribution.**
  It was not one. ``F(x)`` returned ``p`` at every ``x`` -- including
  ``x = -100`` -- which is a flat curve with no time axis, not a coin
  flip. Meanwhile ``moment``, ``entropy``, ``random`` and ``fit`` all
  described a genuine ``{0, 1}`` variable: ``E[X^m] = p``, the binary
  entropy, draws of 0 and 1, and a fit that rejects anything else. The
  class was two models at once, and ``df``, ``hf``, ``Hf`` and ``mean``
  were missing because they are the four places the contradiction
  cannot be papered over.

  ``Bernoulli`` is now the coin flip the name promises. ``x`` is the
  outcome, so 0 and 1 are the only values accepted and anything else
  raises::

      Bernoulli.sf([0, 1], 0.3)  ->  array([1. , 0.3])
      Bernoulli.df([0, 1], 0.3)  ->  array([0.7, 0.3])
      Bernoulli.hf([0, 1], 0.3)  ->  array([0.7, 1. ])
      Bernoulli.sf(37.5, 0.3)    ->  ValueError

  The survival function is :math:`R(x) = P(X \geq x)`, so ``R(0) = 1``
  and ``R(1) = p``: read as a one-shot device, ``p`` is the probability
  it works when demanded. ``df``, ``hf``, ``Hf`` and ``mean`` are added
  and every internal identity now holds -- the mass sums to one,
  ``h = f/R``, ``H = -ln R``, and ``E[X]`` from the mass equals both
  ``mean`` and ``moment(1)``. ``moment``, ``entropy``, ``random`` and
  ``fit`` are unchanged, because they already described this model.

  **``p`` has changed direction.** It was documented as the probability
  of *failure*; it is now the probability of the ``1`` outcome, which
  under the survival reading is the probability of *surviving*. Code
  that coded failures as 1 now fits the survival probability and wants
  ``1 - p``.

  ``log_df`` is defined on the class rather than inherited. Neither base
  relation fits: ``DiscreteParametricFitter`` uses
  ``f(k) = h(k) R(k - 1)``, which assumes ``R(k) = P(X > k)``, and here
  the at-risk set at ``x`` is ``R(x)`` itself.

  **The flat model is not gone.** It survives unchanged as
  :data:`FixedEventProbability`, which until now was a second instance
  of the same class and is now its own. It is the two-point mixture of
  ``InstantlyOccurs`` (weight ``p``) and ``NeverOccurs`` (weight
  ``1 - p``) -- which is why ``degenerate.py`` already described those
  two as its limits at ``p = 1`` and ``p = 0``. Its ``df``, ``hf``,
  ``qf`` and ``mean`` remain absent, correctly: a constant ``F`` has no
  density, no invertible quantile and no time to average.

  Both names serialise and round-trip under their own identities, so
  stored models keep pointing at the model they were fitted with -- but
  a stored ``Bernoulli`` fitted before 0.20.0 will now be read with the
  new semantics, and its ``p`` reinterpreted as above.

  ``binomial.py`` claimed Bernoulli was "the special case ``n = 1``".
  That was false of the old model and is now true of the mass function:
  ``Bernoulli.df`` and ``Binomial.df(..., 1, p)`` agree exactly. The
  survival functions remain offset by one by convention, and the
  docstring now says so.

- **``ExpoWeibull.moment``.** It was the only continuous distribution
  without a public ``moment``, while already having ``mean`` and
  ``entropy``.

  The exponentiated Weibull has a closed form -- an infinite series in
  :math:`\binom{\mu-1}{i}(-1)^{i}(i+1)^{-(1+m/\beta)}` -- but it only
  terminates when :math:`\mu` is a positive integer. For other
  :math:`\mu` it is alternating and slow to converge, losing
  significance to cancellation as :math:`\mu` grows. The integral is
  quadrature either way, so ``moment`` takes it directly, as ``entropy``
  already does and as ``mean`` already did. ``mean`` now delegates to
  ``moment(1)`` rather than repeating the integral.

  Checked against two references with no integration in them: at
  :math:`\mu = 1` the distribution collapses to the Weibull, whose
  m-th moment is :math:`\alpha^{m}\Gamma(1 + m/\beta)` exactly; and
  for integer :math:`\mu` the series terminates and can be summed. Both
  agree to about 1e-14. The exponentiated-exponential case
  (:math:`\alpha = \beta = 1`) is also pinned against the harmonic
  number :math:`H_{\mu}`, which is its mean.

  ``ExpoWeibull`` joins the ``moment`` comparison against quantile-bounded
  numerical integration in ``test_distributions_math.py``, which had
  excluded it by name. That check is not circular despite both sides
  integrating: the reference integrates between quantiles with
  breakpoints, ``moment`` integrates from zero to infinity.

  This does not change fitting. ``ParametricFitter._moment`` already had
  a quadrature fallback for distributions without a ``moment``, so
  ``how="MOM"`` worked for ``ExpoWeibull`` before this and still does.
  What was missing was the public method.

- **Fixed: ``Binomial.log_df`` returned the wrong mass, and
  ``ExactEventTime`` answered ``df`` and ``hf`` with ``inf``.**

  Two consequences of continuous-distribution assumptions reaching
  distributions that are not continuous.

  ``ParametricFitter.log_df`` is ``log(hf) - Hf``, which encodes the
  continuous identity :math:`f = h R(x)`. On the integers the mass at
  ``k`` is :math:`P(T = k) = h(k) R(k - 1)` -- the hazard there times
  the survival to just *before* it. The two differ by a factor
  :math:`R(k)/R(k-1)`, which is not a rounding difference::

      Binomial.log_df(3, 10, 0.3)  ->  -1.887   (was)
      log(Binomial.df(3, 10, 0.3)) ->  -1.321

  Across ``k = 1..7`` the returned mass ran from 0.88 of the truth down
  to 0.15. Five of the six discrete distributions override ``log_df``
  with a closed-form log-pmf and were unaffected; Binomial did not, and
  reached the continuous identity. ``DiscreteParametricFitter`` now
  supplies the discrete relation, so Binomial is correct and any future
  discrete distribution inherits the right one. The class already
  documented the convention -- ``hf`` is ``P(T = k) / R(k - 1)`` -- it
  simply had no ``log_df`` to match it.

  The bug was latent rather than live: ``Binomial.fit`` is analytic
  (``p`` is the observed mean over the trial count) and never evaluates
  a log-density, so no fit was affected. ``Binomial.log_df`` is public,
  though, and generic code that calls it got the wrong numbers.

  Separately, ``ExactEventTime`` is a point mass, so its density is a
  Dirac delta: zero everywhere, infinite at one point, integrating to
  one. There is no function of ``x`` that represents it. ``df`` returned
  ``inf`` at ``T`` and 0 elsewhere, which integrates to ``inf`` rather
  than 1; ``hf`` returned ``inf`` at ``T`` *and at every x after it*;
  and the inherited ``log_df`` computed ``log(inf) - inf`` and returned
  ``nan``. All three now raise ``NotImplementedError`` explaining why
  and pointing at the functions that are defined. An ``inf`` propagates
  into a plot, a likelihood or a mixture weight and surfaces far from
  its cause; a raise stops at the call site. ``Bernoulli`` already
  omitted ``df``, ``hf`` and ``Hf`` for the same reason.

  ``ExactEventTime.Hf`` is kept and is unchanged in value -- it is
  :math:`-\log R(x)`, stepping from 0 to infinity at ``T``, which is
  well defined. It had been written as an alias for ``hf``, which
  happened to take the same two values; it is now written as itself.
  ``sf``, ``ff``, ``qf`` and fitting are untouched.

  New tests cover the discrete mass identity for all six distributions,
  Binomial's log-pmf against scipy, that the discrete hazard is a
  probability (a continuous-convention hazard can exceed one, which is
  how the mix-up shows itself), that ``Hf`` accumulates as
  :math:`-\sum \log(1 - h)` rather than :math:`\sum h`, and the
  degenerate refusals alongside proof that fitting and serialisation
  still work.

- **``Beta.mpp`` and ``Beta4.mpp`` removed as unreachable.**
  Both bodies were a single ``raise NotImplementedError``, and neither
  could ever run. Refusing probability plotting is declarative --
  ``supports_mpp = False``, checked in ``fit`` before the fitter is
  dispatched -- and both distributions already set it, so the guard
  raised a ``ValueError`` naming the distribution and the alternatives
  three frames before the method was reachable.

  Deleting them changes no behaviour. ``Beta``, ``Beta4``, ``Gamma`` and
  ``ExpoWeibull`` all still refuse ``how="MPP"`` from the same guard,
  with the same message. ``mpp`` is now defined only by ``Exponential``
  and ``Rayleigh``, which is where the hook means something: absence of
  ``mpp`` sends a distribution to the *generic* plotting path, so the
  method is an override for a closed form, never a way to decline.

  Two invariants in ``test_shared_signatures.py`` keep the two
  mechanisms from drifting back together: no distribution may declare
  ``supports_mpp = False`` and define ``mpp`` as well, and every
  distribution that refuses must refuse through the shared guard rather
  than an exception of its own. The second covers nine distributions and
  is scoped to those whose ``fit`` takes a ``how`` at all -- ``Bernoulli``,
  ``Binomial`` and ``ExactEventTime`` override ``fit`` with a narrow
  signature that has none, so asking them for MPP is a ``TypeError``
  from argument binding. That is the separate ``fit`` divergence, still
  open.

- **``cs`` is inherited rather than restated on every distribution,
  and Gamma's ``cs`` documentation no longer describes the exponential.**
  Twelve distributions defined a conditional survival function. Eleven
  of the twelve had the same body as ``ParametricFitter.cs``, differing
  only in spelling the parameters out instead of taking ``*params``::

      return self.sf(x + X, alpha, beta) / self.sf(X, alpha, beta)

  The duplication had already rotted. ``Gamma.cs`` carried

  .. math::
      R(x) = e^{-\lambda x}

  which is the *exponential* survival function -- copy-pasted from
  ``exponential.py``, where both methods sat at line 136. The body
  computed the ratio correctly, so the code was right and the
  documentation above it described a different distribution. Gamma is
  not memoryless and its conditional survival is not its survival.

  The eleven pass-through overrides are removed (395 lines), and
  ``ParametricFitter.cs`` -- which had no docstring at all, so ``cs``
  was undocumented anywhere the override was absent -- now carries the
  definition, the parameter descriptions and a worked example. The
  wrong Gamma formula goes with the override it lived on, and Gamma
  inherits the correct generic statement.

  ``Exponential.cs`` is kept. The exponential is memoryless, so
  :math:`R(x, X) = R(x)`, which is one ``exp`` rather than two and a
  division, and avoids the cancellation the ratio suffers far into the
  tail.

  Ten of the removed docstrings carried doctested examples, and those
  were the only per-distribution numerical check on ``cs``. Their values
  are preserved in
  ``surpyval/tests/univariate/parametric/test_conditional_survival.py``,
  alongside tests that each distribution's ``cs`` equals the survival
  ratio (which is what checks Exponential's shortcut against the long
  way), that ``cs(0, X) == 1``, that the exponential is memoryless for
  any conditioning time, and that the discrete distributions reach a
  working inherited ``cs``.

- **BREAKING: shared methods now have one signature across every
  distribution.**
  A distribution is reached through a ``ParametricFitter`` reference
  all over the package -- ``fit_best`` iterates a list of them,
  ``Discretize`` and ``MixtureModel`` wrap one, the regression fitters
  hold one as ``self.dist`` -- so code written against that reference
  has to work for every member. Three shared methods disagreed about
  what their leading argument was called, which made a keyword call
  correct for a subset and a ``TypeError`` for the rest::

      Weibull.qf(p=0.5, alpha=10, beta=2)   worked
      Poisson.qf(p=0.5, mu=3)               TypeError
      Poisson.qf(u=0.5, mu=3)               worked
      Weibull.moment(n=2, alpha=10, beta=2) worked
      Poisson.moment(n=2, mu=3)             TypeError

  This is the defect that made the narrow ``from_params`` overrides on
  ``Bernoulli``, ``Binomial`` and ``ExactEventTime`` worth fixing
  earlier in this release, applied to the rest of the surface.

  - ``qf``'s first argument is ``u`` in all 22 implementations. It was
    ``p`` in 14, ``u`` in 7 and ``q`` in ``Binomial``. ``p`` cannot be
    the shared name because it is an actual parameter of ``Bernoulli``,
    ``Binomial``, ``Geometric`` and ``NegativeBinomial``, and ``q`` is
    one of ``DiscreteWeibull``'s -- which is why the two obvious
    choices had been avoided piecemeal in the first place.
  - ``moment``'s first argument is ``m`` in all 21. It was ``n`` in 13,
    and ``n`` is ``Binomial``'s trial count.
  - ``mpp_x_transform`` takes ``x`` alone in all 15. Eleven of them
    also took a ``gamma`` they subtracted, and the other four did not.
    No caller ever passed it: the MPP fitter subtracts the offset from
    ``x`` before calling (``fitters/mpp.py``), so a caller that did
    pass it would have subtracted the offset twice. Removed rather
    than added to the other four.

  Positional calls -- which is what every docstring example, every call
  inside the package, and every notebook uses -- are unaffected. No
  keyword call to any of the three exists in the package, its tests or
  its documentation. There is no deprecation shim: keeping the old name
  as an alias would preserve exactly the ambiguity the change removes.

  ``moment`` is also now typed ``m: int`` uniformly, and nine
  docstrings that promised "integer or numpy array of integers" are
  narrowed to "integer". Only six of the twenty implementations
  actually accepted an array of orders; the rest raised, because they
  delegate to ``scipy.stats``::

      LogNormal.moment(np.array([1, 2]), 3., 4.)  ->  [5.99e+04, 3.19e+16]
      Normal.moment(np.array([1, 2]), 3., 4.)     ->  ValueError

  ``surpyval/tests/univariate/parametric/test_shared_signatures.py``
  reads the signatures rather than asserting a list of names, so a
  distribution added later is covered without touching the test, and
  an open-ended guard fails on *any* method implemented by five or
  more distributions whose leading data argument disagrees. Parameter
  names are excluded from that guard: ``Weibull.mean(alpha, beta)``
  against ``Poisson.mean(mu)`` is not a divergence, it is what the
  distributions are.

- **API reference pages for the surfaces that only had narrative docs.**
  The multivariate copulas and the beta survival tree and forest had no
  autodoc coverage at all, and the degradation page stopped at the path
  models. Closes the second half of #141.

  New: ``surpyval.multivariate`` (the ``Copula`` base, the five copula
  classes, ``CopulaModel`` and ``MultivariateSurpyvalData``) and
  ``surpyval.beta`` (``RandomSurvivalForest``, ``SurvivalTree`` and the
  node classes a serialised tree is built from). The degradation page
  gains the Wiener and gamma stochastic-process models, ``ProcessRUL``
  and destructive degradation. ``_bounds`` and ``population`` are
  deliberately left out: neither is exported from the package's
  ``__init__``, so both are internal rather than API.

  Two surfaces the issue did not name but which fit its description
  also had no autodoc, and now have pages: shared frailty models and
  Buckley-James. And ``surpyval.regression`` had two headings --
  "Accelerated Time Models" and "Accelerated Life Models" -- with
  nothing underneath them, rendering as empty sections while the
  content sat in ``regression/parametric``; that page is reorganised
  into semi-parametric, parametric and correlated-observations.

  Three ``automethod`` directives on the NHPP regression page pointed at
  ``cif``, ``iif`` and ``inv_cif`` on the *fitter*, where they do not
  exist -- they are on the model the fit returns. Those were three of
  the 18 warnings standing between the build and ``-W``. All 138
  autodoc targets across the documentation now resolve.

- **The estimation machinery moved off the distribution base class.**
  ``ParametricFitter.fit`` takes 18 named arguments -- ``how``,
  ``offset``, ``zi``, ``lfp``, ``fixed``, the truncation and interval
  bounds -- and three distributions cannot honour any of them.
  ``Bernoulli``, ``Binomial`` and ``ExactEventTime`` estimate their
  parameters in closed form and accept only ``x`` and at most ``c``,
  ``n`` and ``t``. They overrode ``fit`` with a narrower signature,
  which is a real divergence and not a typing nicety::

      Bernoulli.fit([0, 1], c=[0, 0])              TypeError
      Bernoulli.fit([0, 1], how="MLE")             TypeError
      Binomial.fit([1, 2], n_trials=3, how="MLE")  TypeError

  Code written against a ``ParametricFitter`` therefore broke on
  exactly those three, and nothing said so until it ran.

  ``fit`` and the twelve methods it needs now live on a new
  ``OptimisedFitMixin``, which the 21 distributions that have them
  inherit alongside ``ParametricFitter``. Nothing was removed and no
  behaviour changed: every distribution is still a ``ParametricFitter``,
  which is what the ``isinstance`` checks in the model, mixture,
  regression, frailty and renewal code test, and what carries the
  distribution functions, the likelihood and ``from_params``.

  The point of the split is that the wrong thing is now unwriteable
  rather than merely undocumented. ``fit_best``'s candidate list is
  typed ``list[OptimisedFitMixin]``, so adding one of the three to it is
  a type error instead of a runtime one.

  Annotate a parameter ``OptimisedFitMixin`` when it must be fittable by
  a chosen method, and ``ParametricFitter`` when only the distribution
  functions are needed.

- **BREAKING:** ``Bernoulli.from_params`` and
  ``ExactEventTime.from_params`` **renamed their first argument to**
  ``params``. It was ``p`` and ``T`` respectively, while the base calls
  it ``params``, so positional calls worked and keyword calls raised::

      Bernoulli.from_params(0.5)          OK
      Bernoulli.from_params(params=0.5)   TypeError

  That is the shape of bug a test suite never catches, because every
  internal call and every docstring example passes positionally.

  Bernoulli's was worse than a rename. The base's ``p`` is the
  proportion that *never fails*, so ``p=0.5`` meant the never-fails
  fraction on twenty-four distributions and the event probability on
  Bernoulli -- the same keyword, sibling classes, unrelated meanings,
  and no error either way.

  ``Bernoulli.from_params(p=...)`` and ``ExactEventTime.from_params(T=...)``
  now raise ``TypeError``. Positional calls are unaffected, and no call
  changes meaning silently: ``params`` has no default, so the old
  keyword forms fail loudly rather than being reinterpreted.

  All three also accept ``gamma``, ``p`` and ``f0`` now, and reject them
  with a ``ValueError`` naming the distribution. Accepting-and-rejecting
  rather than omitting is what makes the signatures match the base, so
  these can be called through a ``ParametricFitter`` reference at all --
  and it removes the last two ``# type: ignore[override]`` markers.

- **Every distribution now exports its own type.** The 17 that read
  ``Weibull: ParametricFitter = Weibull_("Weibull")`` erased the
  concrete class, and since the base declares none of ``sf``, ``ff``,
  ``df``, ``hf``, ``Hf``, ``qf`` or ``mean``, the example in each
  distribution's own docstring did not type check for anyone whose
  checker honours ``py.typed``::

      Weibull.sf(x, 3, 4)
      error: "ParametricFitter" has no attribute "sf"

  The annotation cannot simply be dropped: the regression subpackages
  and ``fit_best`` import these names, and without an explicit type
  mypy cannot resolve them through that cycle. They name the concrete
  class instead.

- **``Normal`` and ``Gumbel`` ignored their own documented default.**
  ``ParametricFitter`` documents the initialiser signature as
  ``(self, x, c=None, n=None, t=None, offset=False)``, but ``Normal``
  tested ``2 in c`` and indexed ``x[c != -1]``, and ``Gumbel`` tested
  ``(2 in c) or (-1 in c)``, before either defaulted ``c``. Calling
  either as documented raised ``TypeError: argument of type 'NoneType'
  is not iterable``. Every caller inside the package passes ``c`` and
  ``n``, which is why it went unnoticed; ``GumbelLEV`` is unaffected
  because it forwards ``c`` to ``fit`` without inspecting it. A sweep of
  all nineteen distributions found these two and no others.

- **BREAKING: ``_parameter_initialiser`` takes a ``SurpyvalData``.**
  The signature was ``(self, x, c=None, n=None, t=None, offset=False)``,
  and every one of the 21 implementations spent its opening lines
  re-establishing conventions that had already been established --
  inconsistently, and in some cases wrongly. ``Normal`` defaulted
  ``c`` and ``n``; ``Gumbel`` guarded ``c`` with ``is not None``;
  ``Beta`` tested ``(c is not None) and (c == 0).all()``; ``Beta4``
  tested both ``c`` and ``n``; ``LogLogistic`` ran a whole
  ``xcnt_handler`` round trip in its offset branch. Two of those checks
  were absent until this release and raised ``TypeError`` for the
  documented call.

  None of it was ever needed. The one production caller,
  ``_initial_guess``, is reached from ``fit_from_surpyval_data``, which
  is *handed* a ``SurpyvalData`` -- an object whose entire purpose is to
  guarantee that ``x``, ``c``, ``n`` and ``t`` are present, validated
  and in xcnt form -- and destructured it into loose arrays on the first
  line of its body. The convention was rebuilt three layers below the
  object that had already established it.

  The signature is now ``(self, data: SurpyvalData, offset: bool =
  False)``. ``offset`` stays a separate argument because it describes
  the model being requested, not the data. ``_initial_guess`` and
  ``_fit_numerically`` take the object rather than loose arrays for the
  same reason. Seven defaulting checks are gone, along with 63 optional
  data parameters (27 of them explicitly annotated ``| None``), and the
  initialisers that used to
  round-trip their arrays back through ``fit`` (re-running
  ``xcnt_handler`` and rebuilding the object the caller already held)
  now call ``fit_from_surpyval_data`` directly.

  ``t`` is not passed to the initialisers, and never was: no caller has
  ever supplied it. ``_initial_guess`` imputes interval- and
  left-censored points to midpoints before seeding, which can put an
  observation at or before its own left-truncation bound -- data
  ``xcnt_handler`` rejects outright (#260) -- so the working copy it
  builds is deliberately untruncated. That is what every initialiser has
  always received; it is now explicit rather than accidental.

  This is a breaking change for anyone who has written their own
  distribution class. There is no shim: a bare array now fails at the
  first attribute access rather than being half-accepted. Every one of
  the 38 seeds -- each distribution, plain and offset -- is identical
  before and after.

- **Every ``_parameter_initialiser`` now returns the same thing.**
  The initial-guess seed a distribution hands the optimiser came back in
  four different containers across the 21 implementations: a tuple in
  nine, a numpy array in six, a Python list in one, a fitted model's
  ``.params`` in five -- and a bare scalar in ``Rayleigh``. Two files
  disagreed with *themselves*: ``exponential`` returned a tuple in its
  offset branch and an array in the other, ``rayleigh`` a tuple and a
  scalar.

  It worked because the one caller, ``_initial_guess``, does
  ``np.array(init)``, which flattens tuple, list and array alike. It
  stopped working at the scalar, because ``np.array`` of a scalar is
  0-dimensional rather than length-1, and the ``lfp`` and ``zi`` paths
  concatenate onto the seed.

  All 28 return statements now construct a 1-D float array explicitly,
  so the shape is decided where the values are known rather than
  inferred downstream, and a 0-dimensional seed is no longer
  expressible. No seed changed: all 38 -- every distribution, plain and
  offset -- were compared before and after and are identical.

  The seed itself is unchanged in layout, and it is flat rather than
  nested: ``[gamma]`` when an offset is requested, then the ``k``
  distribution parameters, then ``[p]`` for a limited failure population
  and ``[f0]`` for zero inflation, appended by the caller. The arity
  therefore depends on both ``k`` and the structural flags.

- **Limited-failure and zero-inflated Rayleigh models could not be fit.**
  ``Rayleigh.fit(x, lfp=True)`` and ``Rayleigh.fit(x, zi=True)`` both
  raised ``ValueError: zero-dimensional arrays cannot be concatenated``.

  Rayleigh is the only single-parameter distribution here, and its
  ``_parameter_initialiser`` returned the sigma seed as a bare scalar
  rather than a sequence. ``np.array(init)`` in ``_initial_guess`` then
  produced a 0-dimensional array instead of a length-1 one, and the
  ``lfp`` and ``zi`` paths append their ``p`` and ``f0`` seeds with
  ``np.concatenate``, which a 0-d array cannot take. The seed is now a
  one-tuple. Plain and offset fits are unchanged.

  Found by surveying every ``_parameter_initialiser`` in the library
  after the type-hint work turned up three different return shapes; a
  sweep of all fourteen continuous distributions across both paths
  confirmed Rayleigh was the only one affected.

- **Type-hint coverage is now enforced, for twenty-one modules (#143).**
  ``surpyval.distribution``, ``surpyval.serialisation``,
  ``surpyval.metrics``, ``surpyval.univariate.information_criteria``,
  ``surpyval.datasets``, the Weibull, the Normal, the LogNormal, the
  eight discrete distributions, ``CustomDistribution`` and
  ``ExactEventTime``, and
  all of ``surpyval.univariate.nonparametric``,
  ``surpyval.recurrent.nonparametric`` and
  ``surpyval.univariate.regression.frailty`` have
  ``disallow_untyped_defs`` set in ``pyproject.toml``, so an
  unannotated function in any of them is a mypy error. That covers the
  abstract base classes every model inherits from, the Kaplan-Meier,
  Nelson-Aalen, Fleming-Harrington and Turnbull estimators, the
  log-rank test, the plotting positions, the non-parametric MCF, the
  shared-frailty fitter, the bundled datasets and thirteen of the 25
  parametric distributions.

  ``LogNormal.moment`` is annotated ``n: Numeric`` where ``Normal``'s
  is ``n: int``, and the difference is real rather than an oversight.
  Both docstrings promise "integer or numpy array of integers".
  LogNormal's closed form is vectorised and delivers that;
  ``Normal``, ``GumbelLEV`` and ``LogLogistic`` delegate to
  ``scipy.stats``, which raises ``ValueError: The truth value of an
  array ... is ambiguous`` on an array of orders. The annotations now
  say which is which; the three docstrings that overpromise are not
  yet corrected.

  ``CustomDistribution`` needed restructuring rather than only
  annotating. It assigned its distribution functions onto the
  instance -- ``self.Hf = fun``, then lambdas for ``hf``, ``sf``,
  ``ff`` and ``df`` -- which stopped being possible once
  ``OptimisedFitMixin`` declared those names for its own use, because
  a subclass inherits the declarations and assigning to an inherited
  method is an error. The function is stored as ``_fun`` and the five
  are real methods delegating to it. Equivalent by construction: the
  old ``self.Hf = fun`` was an unbound instance attribute, so
  ``self.Hf(x, *params)`` called ``fun(x, *params)`` either way. The
  autograd-derived ``hf`` and ``df`` were checked numerically against
  the previous implementation, gradients included.

  Its ``_parameter_initialiser`` returns a *list*, where Weibull
  returns a tuple and the discrete distributions return an array --
  three shapes for one contract the base never pinned down. Noted in
  the signatures rather than unified, since every caller coerces.

  ``handle_xicn`` gained ``@overload`` declarations as part of this.
  Its return shape is decided by ``as_recurrent_data``, but its
  signature only said "one or the other", so all nine callers taking
  the default were handed a union to narrow themselves. The overloads
  say which argument decides, once, for all seventeen call sites.

  The package ships ``py.typed``, which tells a user's type checker
  that the annotations are there to be trusted, and mypy already ran in
  CI -- but nothing required an annotation to exist, so mypy checked
  only the ones that happened to be written. That made ``py.typed`` a
  promise the package kept unevenly.

  This is deliberately a ratchet rather than a target. A module is
  added to the list once it is clean, and from then on it cannot
  regress; the remaining ~1350 unannotated functions do not have to be
  finished first for the enforced part to start holding.

  Turning it on immediately found something. ``SerialisableMixin.to_json``
  and ``from_json`` call ``self.to_dict()`` and ``cls.from_dict()``,
  which the mixin never declares -- every class using it supplies them,
  but that contract existed only in the docstring, and mypy skips the
  bodies of unannotated functions, so the calls had never been checked.
  They are now declared under ``TYPE_CHECKING``: a real stub raising
  ``NotImplementedError`` would read better, but it would be inherited,
  and ``copula_model`` decides whether a margin is serialisable with
  ``hasattr(m, "to_dict")`` -- which an inherited stub would answer
  True for every time.

  The non-parametric package turned up a second one. Its
  ``ESTIMATOR_FUNCS`` table was built from ``nonp.nelson_aalen`` and its
  two siblings, each of which shares its name with the submodule that
  defines it -- so the attribute is the function only after the package
  ``__init__`` has bound it over the submodule, and that table is built
  while the ``__init__`` is still running. It worked because the
  ``__init__`` happens to import the estimators first, which is a load
  order rather than a guarantee. mypy resolved the names to the modules
  and reported the table as not callable. The three are now imported
  from the modules that define them; the other uses of the package
  namespace in that file are inside functions, so they resolve after
  initialisation and are left alone.

  Annotating a function makes mypy check its body, and that found a
  real inconsistency the ratchet had been hiding behind an over-wide
  annotation. ``turnbull``, ``rank_adjust`` and
  ``NonParametricCounting.from_xrd`` declare array-like parameters and
  then index, slice and divide them directly -- which array-like does
  not support, because it also covers ``str``, ``bytes`` and scalars.
  Each now takes its arguments as arrays before using them as arrays,
  so the signature and the body agree. ``_logrank_z_v`` likewise
  declared ``c`` and ``n`` as arrays while handling ``None`` for both
  internally, and ``NonParametricCounting.fit`` declared ``windows``
  as array-like when it is the ``{item: [(start, end), ...]}``
  dictionary its own docstring describes.

  ``success_run`` was the same shape of problem in its argument
  handling: it tested ``confidence`` and ``alpha`` for truthiness, so
  passing both with either set to zero skipped the "only one of" raise,
  and ``confidence=0`` fell through every branch and left ``alpha``
  unset. Both are now tested against ``None``.

  The distributions needed a vocabulary before any of them could be
  annotated, and ``parametric_fitter`` now defines it. A distribution
  function deals in two kinds of value, and only one of them can be an
  autograd box: ``Numeric`` is what the function is evaluated at (times,
  or probabilities for ``qf``), always real data; ``Boxable`` is a
  parameter, or anything computed from one. Maximum likelihood
  differentiates these functions, so during a fit autograd substitutes
  an ``ArrayBox`` for each parameter to carry the derivative. A box is
  neither a float nor an ndarray, which is why ``Boxable`` is not
  narrowed to a numpy type -- and why the "array-like in, array out"
  convention used elsewhere in the package must not be applied here.

  ``np.asarray`` on a box does not reject it. It wraps the box in a
  0-d object-dtype array, which still computes the right *value*,
  because object arrays dispatch arithmetic back to the box. Only the
  derivative is damaged, and how depends on the arithmetic: an
  operation whose backward pass needs a ufunc the box does not
  implement raises a ``TypeError``, but a plain product silently
  returns a zero gradient for that parameter. A zero gradient is not an
  error to an optimiser -- it means "this parameter does not affect the
  likelihood" -- so the fit leaves the parameter at its initial guess
  and reports success.

  ``Boxable`` names ``ArrayBox`` rather than being ``Any``, which is
  what makes the parameter positions checkable rather than merely
  annotated: under ``Any``, ``Weibull.Hf(1.0, "not a number", 4.0)`` was
  accepted in silence. autograd ships no type information, so
  ``stubs/autograd/numpy/numpy_boxes.pyi`` describes the one type of its
  that appears in surpyval's own signatures. Supplying any stub for a
  package makes mypy consider the whole package described, so the
  ``__getattr__`` stubs beside it keep the rest of autograd as untyped
  as it was.

  ``Weibull`` and the eight discrete distributions -- ``Bernoulli``,
  ``BetaGeometric``, ``Binomial``, ``DiscreteWeibull``, ``Geometric``,
  ``NegativeBinomial``, ``Poisson`` and the ``Discretize`` wrapper --
  are annotated against that vocabulary and are on the enforced list.

  Checking their bodies turned up three things about the base class.

  ``Weibull`` was exported as ``Weibull: ParametricFitter``, which
  erases the concrete type; since the base declares none of ``sf``,
  ``ff``, ``df``, ``hf``, ``Hf``, ``qf`` or ``mean``, and the package
  ships ``py.typed``, the example in ``sf``'s own docstring did not type
  check for a user::

      Weibull.sf(x, 3, 4)
      error: "ParametricFitter" has no attribute "sf"

  It is now exported as ``Weibull_``. Sixteen other distributions carry
  the same erasure and are corrected as each is annotated.
  ``Discretize`` hits the same gap from the other side: it must hold the
  distribution it wraps as ``Any``, because every delegation would
  otherwise be an error against the declared type.

  ``Bernoulli.fit`` and ``Binomial.fit`` do not honour
  ``ParametricFitter.fit``, and the divergence is real rather than a
  typing artefact -- ``Bernoulli.fit(x, c=...)`` and
  ``Binomial.fit(x, n_trials=k, how=...)`` raise ``TypeError``. Both
  have closed-form maximum likelihood estimates and support neither
  censoring nor an alternative estimation method. Generic code written
  against ``ParametricFitter`` will fail on them. This is recorded at
  each site rather than changed here.

  ``_parameter_initialiser`` is also inconsistent across the base's
  implementations: ``Weibull`` returns a tuple, the discrete
  distributions return an array. Callers coerce either, so nothing is
  broken, but the signatures now say which is which.

- **The lint job had been failing on a single over-long line.**
  The ``:class:`` cross-reference added to ``RandomSurvivalForest``'s
  docstring while clearing the documentation build warnings pushed the
  line to 80 characters. flake8 runs before mypy in the lint job, so
  from that commit on mypy was skipped rather than run, and the type
  errors described above reached CI unchecked. The line is rewrapped
  and the errors are fixed.

- **The survival tree's log-rank split statistic was wrong (#287).**
  ``kind="non-parametric"`` trees, and any ``RandomSurvivalForest``
  built from them, selected splits on a statistic off by factors of
  several -- and not merely inflated, but *reordered*: of the two cases
  in the issue, the weaker separation scored 1.856 against a true 0.276
  while the stronger scored 0.276 against a true 1.225. Trees were
  choosing the wrong split.

  The log-rank statistic sums over the *pooled* event times of both
  children, so the left child's at-risk count is needed at times where
  the left child itself has no observation. Those were filled in by
  carrying its own risk ladder forward, which was wrong twice: the
  carried value did not subtract the deaths and censorings that occurred
  *at* the time it was carried from, and the tail past the last
  observation subtracted only deaths, so a child ending in a censored
  observation kept someone at risk for ever. Both inflate the count,
  which biases the numerator and the variance.

  The at-risk count is now computed directly -- at each pooled time
  :math:`t`, the observations with :math:`t_l < t \leq x`, which is the
  ``(entry, exit]`` convention ``xcnt_to_xrd`` already uses, so the
  left child's :math:`Y_L` and the pooled :math:`Y` agree about what
  "at risk" means. There is nothing to extrapolate, so both the leading
  and trailing special cases disappear along with the forward fill.

  Tests check the statistic against a deliberately naive implementation
  of its definition: the two cases from the issue, a censored tail, left
  truncation, ties shared across children, a child starting after the
  other's first event, and 200 random partitions. One test pins
  ``at_risk_on_grid`` against ``xcnt_to_xrd`` directly, since a
  disagreement between them is what makes :math:`Y_L / Y` stop being a
  proportion.

- **The documentation build fails on any warning.** ``-W --keep-going``
  in the CI docs job, and ``fail_on_warning`` in ``.readthedocs.yaml``.
  They are set together deliberately: if only one has it, that one goes
  green while the other publishes a broken page.

  A Sphinx warning is rarely cosmetic. A broken cross-reference renders
  as plain text, a mistyped ``autoclass`` path drops the class from the
  page altogether, a page missing from every toctree is published and
  unreachable. In each case the build reports success and ships
  something wrong, and the only evidence is a line in a log nobody
  reads. That is precisely how the three broken ``ProportionalIntensityNHPP``
  autodoc targets survived -- three methods absent from the rendered
  documentation, behind a "build succeeded, 18 warnings".

  Warnings-as-errors only works from zero, so the sixteen were cleared
  first:

  - Twelve duplicate labels, from ``autosectionlabel`` minting a
    cross-reference target for every heading while the changelog
    necessarily repeats "Serialisation", "Degradation" and so on once
    per release. ``autosectionlabel_maxdepth = 1`` keeps the labels a
    ``:ref:`` between pages actually wants and stops minting the rest.
  - The ``sphinx_rtd_theme`` ``get_html_theme_path`` deprecation, whose
    own message said the call was safe to remove.
  - A title-level inconsistency in the non-parametric page, reported by
    docutils as ``CRITICAL`` rather than a warning.
  - ``RandomSurvivalForest``'s docstring, which was not valid
    reStructuredText -- and which nothing surfaced until the new API
    page began rendering it.
  - The interval-censored Turnbull example, whose EM ran out of
    iterations. It converges at ``max_iter=10000``; the example now
    passes it and the prose explains why, rather than marking the
    warning expected and hiding a usable answer.

- **``scripts/check_all_pythons.py`` runs the CI checks locally on every
  supported interpreter.** With the suite no longer running on pull
  requests into ``develop`` (below), this is the other half of the
  trade: one command runs the test suite and both doctest passes on
  3.11, 3.12 and 3.13, and refuses to say "passed" unless all of them
  did.

  It keeps its environments in a git-ignored ``.venvs/`` and reuses
  them, so only the first run pays for the installs; it uses ``uv``
  when available and falls back to ``venv`` and ``pip`` when not, and
  reports an interpreter that is not installed rather than failing on
  it. The command list is deliberately a copy of the workflow's, so
  what it runs is what CI would have run.

- **The test suite runs on the release pull request, not on every one.**
  Pull requests into ``develop`` now run lint only, about a minute
  against the nine the suite takes across three interpreters. The suite
  still runs in full on the release pull request into ``master`` and on
  pushes to ``master``.

  The reason is the edit-review loop: with a single maintainer running
  the suite locally before pushing, the pull-request run was mostly
  confirming what was already known, and it was the slowest part of
  working on the package.

  What this gives up is stated rather than glossed: a failure that
  appears on only one interpreter is now found when the release is
  prepared, with a release's worth of commits to search rather than one.
  That is not hypothetical -- the doctest numeric comparison two entries
  below landed green on 3.11 and failed on 3.12 and 3.13, and it was the
  pull-request run that caught it. ``Contributing.rst`` now says which
  jobs run on which event, and what to run locally to compensate.

- **The documentation build runs in CI on the release pull request.**
  The docs execute every ``.. jupyter-execute::`` cell as they build, so
  they are a second test suite that exercises the public API for real --
  and one that a change touching no documentation file at all can break,
  as the Gamma entry below did. Read the Docs builds only ``master`` and
  tags, so until now that break would have surfaced as a failed hosted
  build *after* a release.

  The new job is conditioned on ``github.base_ref == 'master'``, which
  is set only for pull requests, so it runs on the ``develop`` ->
  ``master`` release pull request and nowhere else. It is not run on
  pushes to ``master`` either: Read the Docs rebuilds there anyway, and
  by then the gate has nothing left to gate. It matches
  ``.readthedocs.yaml`` rather than the test jobs -- Python 3.12, the
  package installed via its own ``docs`` extra -- because its purpose is
  to reproduce the hosted build, and it uploads the rendered HTML as an
  artifact.

  It does not build with ``-W``; the build currently emits 18 warnings,
  mostly duplicate labels from ``autosectionlabel`` meeting the
  changelog's repeated section headings. Clearing those and then failing
  on warning here and in ``.readthedocs.yaml`` together is worth doing
  separately.

  The residual gap is deliberate: a documentation break introduced on a
  pull request into ``develop`` is caught when the release is prepared,
  not when it lands. Building on every pull request would cost minutes
  on each, and a path filter would not have helped here -- the change
  that broke the build was in ``gamma.py``, not under ``docs/``.

- **Fixed a documentation build broken by the Gamma MPP removal.** The
  offset-threshold section of *Parametric SurPyval Modelling* ran a
  ``jupyter-execute`` cell looping over
  ``['MPP', 'MOM', 'MSE', 'MPS', 'MLE']`` for a shifted Gamma. Since
  ``Gamma.fit(how="MPP")`` now raises, that cell raised, and because
  documentation cells are executed during the build the whole build
  failed.

  Nothing caught it: continuous integration does not build the
  documentation, and Read the Docs builds only ``master`` and tags, so
  it would have surfaced as a failed hosted build at the next release
  rather than on the change that caused it. It was found by running a
  build to validate the ``docs`` extra below.

  The prose around the cell had gone stale the same way -- it described
  the multi-start probability-plotting search that the removal deleted,
  and quoted an ``MPP`` tolerance from ``test_offset_divergence.py``
  that no longer exists. It now explains why the Gamma has no
  probability plot at all: the shape sits inside the regularised
  incomplete gamma rather than outside as an exponent, so the only
  straight-line axis is the inverse incomplete gamma, which needs the
  very shape being estimated. ``Gamma.plot()`` is unaffected, since by
  then the parameters are known.

- **The documentation toolchain is a ``docs`` extra.**
  ``pip install -e ".[docs]"`` now installs everything needed to build
  the documentation, alongside the ``tests`` extra that was already
  there. ``docs/requirements.txt`` is gone: its pins moved into
  ``pyproject.toml`` verbatim, and Read the Docs installs the extra
  directly via ``extra_requirements``. Keeping both would have meant two
  copies of the same pinned toolchain, which is the arrangement that
  drifts.

  The pins are unchanged, including the ``ipykernel==6.31.0`` cap and
  the reason for it -- jupyter-sphinx notebook execution dies against
  the ipykernel 7 line. ``matplotlib`` is not repeated in the extra; it
  is a runtime dependency of the package, which is installed alongside.

  Part of #141.

- **CI now runs the docstring examples.** ``pytest --doctest-modules``
  over the package is a new step in the deployment workflow, and every
  one of the 229 docstring examples passes. It was 59 failing tests
  when the flag was first turned on.

  A docstring example is a promise about what the library prints, and
  it is the one users and coding agents reach for first --
  ``help(Weibull.fit)`` is faster than opening the docs. Nothing was
  checking it, so it drifted: examples recorded the output of an
  optimiser two rewrites ago, of numpy 1's scalar repr, of a module
  that has since moved.

  What the run found, beyond the cosmetic drift:

  - Twelve examples could not run at all. Six regression docstrings
    (``PH``, ``AH``, ``PO``, ``AFT``, ``AcceleratedLife``, ``Frailty``)
    were sketches -- ``model = PH(Weibull).fit(x, Z=covariates, c=c)``
    with ``x``, ``covariates`` and ``c`` never defined. Four more used
    ``>>>`` on the continuation lines of a multi-line call, so pasting
    them raised ``SyntaxError``. ``plotting_positions`` imported from
    ``surpyval.nonparametric``, which moved to
    ``surpyval.univariate.nonparametric`` several releases ago.
    ``ParametricFitter.fit`` demonstrated ``how='MPP'`` on
    interval-censored input, which now (correctly) requires the Turnbull
    heuristic and raises without it. All are now runnable, with data.

  - The five ``ParametricRegressionModel`` prediction examples
    (``sf``, ``ff``, ``df``, ``hf``, ``Hf``) had been copied from the
    univariate ``Parametric`` class and never adapted: they built a
    ``Weibull.from_params([10, 3])`` and called it with no covariates at
    all, documenting a signature the method does not have. They now fit
    a ``WeibullPH`` and pass ``Z``.

  - ``Parametric.var()`` claimed 11.229 for a Weibull(10, 3). The
    variance is 10.533 (``100 Gamma(5/3) - (10 Gamma(4/3))^2``); the
    code was right.

  - Several examples fitted unseeded random data and then recorded
    specific digits, which cannot be reproducible. They now seed.

  ``Parametric.hf`` and ``Parametric.Hf`` returned a 0-d array
  (``array(0.012)``) for scalar input where ``sf``, ``ff``, ``df`` and
  ``qf`` all returned a numpy scalar, and ``cs`` did the same; their
  own ``Returns`` sections promised "the scalar value ... if a scalar
  was passed". That is now true. The 0-d array came from ``np.where``,
  which does not collapse.

  **The numbers in the examples are compared as numbers.** doctest
  compares printed output as text, which is the wrong test for a library
  whose examples end in an optimiser: the same ``Duane`` fit lands on
  ``b = 4.1995e-05`` under Python 3.11 and ``4.2032e-05`` under 3.12,
  and numpy prints eight significant digits either way. Sixteen of the
  229 examples disagree between those two Pythons somewhere in their
  digits.

  The obvious workaround -- trimming each documented number back to the
  digits that agree everywhere -- makes the docstring show something the
  reader's own session will not produce, which is precisely what these
  examples exist to avoid. So the examples record the real output, in
  full, and ``conftest.py`` installs a fallback comparison that runs
  only after the ordinary text comparison has failed. It fires when the
  two outputs are identical apart from their numeric literals -- same
  words, same brackets, same integer-versus-float shape, so ``1`` never
  matches ``1.`` and a dtype change is still a failure -- and then
  compares the numbers with ``rel_tol=1e-3``, set by the loosest genuine
  disagreement between supported Pythons with no margin beyond it, and
  ``abs_tol=1e-12`` for a restoration factor whose true value is zero
  and which surfaces as ``1e-16`` with whatever mantissa the optimiser
  stopped on.

  What that forgives is a value drifting inside the tolerance. What it
  still catches is every defect listed above: a stale value from another
  parameterisation, the wrong function being called, the wrong shape, an
  exception, a missing import. ``surpyval/tests/test_doctest_checker.py``
  pins both halves of that, using the real output pairs observed on
  different Pythons.

  The fallback only runs when an example has actually drifted, which on
  any one machine is a handful of them -- and a different handful on
  each. A gap in it is therefore invisible locally and surfaces in CI,
  on whichever Python computed a different last digit. So the doctest
  step runs twice: once normally, and once under
  ``--doctest-force-numeric``, which routes every example whose output
  contains a number through the numeric comparison. Fifteen seconds, and
  the fallback is exercised against all 229 examples rather than
  today's accidental few.

  ``NORMALIZE_WHITESPACE`` is set in ``pyproject.toml`` for the same
  reason: numpy picks its own line breaks and column padding for an
  array and both move with the width of the widest element.

  This closes #158.

- **The distribution docstring examples now show what you actually see.**
  ``pytest --doctest-modules`` over ``distributions/`` is green: 139
  examples, no failures. Previously 39 failed.

  Most were the numpy 2 scalar repr. ``Weibull.mean(3, 4)`` prints
  ``np.float64(2.7192074311664314)``, where the docstring recorded the
  bare ``2.7192074311664314`` that numpy 1 used to print. The examples
  now record the wrapper, because that is what appears at a prompt --
  the alternative was ``np.set_printoptions(legacy="1.25")`` in a test
  fixture, which would have kept the docstrings prettier by showing
  readers something their own session will not produce.

  Four ``qf`` examples printed wider than the 79-column limit once the
  real output was recorded, numpy having rewrapped the arrays. Rather
  than hand-wrap them into something numpy would not emit, those
  examples take fewer probabilities: what is shown is exactly what that
  input produces.

  Two scalar examples had drifted in the last digit, and are re-recorded.

  With the examples now true, ``--doctest-modules`` is worth running in
  CI, which is what stops this recurring; it is turned on above.

- **``Gamma`` no longer offers probability plotting as a fit method.**
  ``Gamma.fit(x, how="MPP")`` now raises, joining ``Beta`` and
  ``ExpoWeibull``, which already declined for the same reason.

  A probability plot works by rearranging the survival function so some
  transform of the data falls on a straight line. For a Weibull,
  ``log(-log S) = beta log x - beta log alpha`` — the axes do not depend
  on the answer, so you can draw them before knowing anything. The
  Gamma has no such rearrangement: its CDF is the regularised incomplete
  gamma function and the shape sits *inside* that special function
  rather than outside as an exponent. The only straight-line y-axis is
  the inverse incomplete gamma, which needs the shape. To draw the axis
  you need the answer; to get the answer you need the axis.

  The code broke the circle by guessing the shape from moments, drawing
  the plot on that guess, and regressing. When the guess was off, the
  axis was the wrong axis, the points were no longer straight on it, and
  the regression fitted a line through a curve — returning a confident
  wrong estimate rather than an error. An offset made it worse: the
  shift distorts the low-``x`` end hardest, which is exactly where the
  shape information lives.

  ``plot()`` is unaffected. It transforms with the *fitted* parameters,
  so by the time the plot is drawn the axis is the right one — the
  probability plot of an MLE-fitted Gamma remains a valid diagnostic.
  Fitting is unchanged for MLE (the default), MSE and MOM.

  The 118-line ``Gamma.mpp`` override is deleted with it, which removes
  the ``rr="x"`` mis-inversion and the censored-data ``LinAlgError`` from
  #257 by making both paths unreachable. The MPP sweep in ``test_fit.py``
  now gates on each distribution's ``supports_mpp`` flag instead of a
  hardcoded exclusion list, so it stays correct without editing.

- **Nine wrong examples in the distribution docstrings.** Running
  ``pytest --doctest-modules`` over ``distributions/`` gives 39 failures.
  Thirty are the numpy-2 scalar repr (``np.float64(2.719...)`` against a
  recorded ``2.719...``) and are cosmetic. Nine were not.

  Five documented outputs were simply wrong. ``Uniform.ff``'s example
  called ``Uniform.sf``, and ``ExpoWeibull.cs``'s called
  ``ExpoWeibull.sf`` -- in both the printed values were right for the
  function being documented and wrong for the one being called, so the
  example read as if the two were the same. ``LogLogistic.sf`` carried
  values from some other parameterisation entirely (0.622 where the
  answer is 0.988), ``LogLogistic.mean(3, 4)`` claimed ``3`` against
  ``3.3322`` (the closed form is ``alpha (pi/beta) / sin(pi/beta)``), and
  ``Exponential.qf`` had stale digits.

  The other four were the ``CustomDistribution`` example -- the Gompertz
  walkthrough -- whose multi-line ``def`` used ``>>>`` where doctest
  needs ``...``, so pasting it raised ``IndentationError``.

  In every case the code was right and the documentation was wrong, which
  is the reassuring direction, but a reader checking their understanding
  against these would have been misled. They accumulated precisely
  because the doctests were not run, which is addressed above.

- **Documented why a Turnbull fit does not equal a Kaplan-Meier fit.**
  ``Turnbull.fit`` defaults to ``turnbull_estimator="Fleming-Harrington"``
  while ``KaplanMeier.fit`` is, unsurprisingly, KM. The Turnbull EM
  recovers the same ``r`` and ``d`` either way; the three estimator
  options then differ in how those become a survival curve. Comparing the
  default against ``KaplanMeier`` and reading the gap as a defect is an
  easy mistake — it is the one #260 was filed on, and the one made again
  while checking whether #260 was still open.

  On ``x=[2,3,3,4,5,6], tl=[0,0,1,1,2,2]`` the survival at 2 is 0.750
  under KM, 0.765 under FH and 0.779 under NA. With the estimator matched,
  Turnbull agrees with ``KaplanMeier`` to around 1e-9 on both ``sf`` and
  ``cb``, across right-censored and left-truncated data.

  Only the KM option is the non-parametric MLE. Maximising the truncated
  likelihood directly over the mass vector gives 0.750; FH's 0.765 scores
  worse on that same likelihood, as an ``exp(-H)`` construction should.
  FH is the default because it behaves better in the far tails and on
  zero-inflated data (v0.8.0), not because it maximises anything. The
  docstring now says all of this, and a test pins the three figures and
  the NPMLE identity against a brute-force maximisation.

  No behaviour change.

v0.19.0 (4 August 2026)
-----------------------

- **Confidence bounds no longer turn silently to nan on data measured
  in large units, and those fits are around 17x faster.** A Weibull fit
  to the same lifetimes expressed in hours had standard errors; in
  seconds it returned ``nan`` for every one of them, with no warning and
  a perfectly good set of parameters alongside.

  The cause is the ``np.where`` trap again, this time in the parameter
  transform rather than a likelihood. Every parameter bounded on
  ``(0, inf)`` is mapped to the unbounded space the optimiser searches
  by ``adj_relu``, which chose between ``x + 1`` and ``exp(x)`` with
  ``np.where``. Autograd evaluates both branches, so ``exp(x)`` was
  taped even where ``x + 1`` was selected, and above ``x = 709.78`` it
  overflows to inf. The inf then poisoned the derivative of the branch
  that *was* chosen, so the jacobian of the transform came back nan --
  and with it ``cov_matrix``, which is that jacobian either side of the
  inverse hessian.

  The threshold is a property of the fitted parameter, not of the sample
  size or the conditioning, which is why it looked so arbitrary: a fit
  died as soon as any ``(0, inf)`` parameter exceeded about 710. A
  Weibull with ``alpha = 10`` lost its bounds once the data was scaled
  past about 70x, while a Gumbel, whose location is unbounded and so
  untransformed, survived to 350x on the same data. The Gamma failed at
  the *small* end instead, its rate parameter growing as the data
  shrinks. The Normal, LogNormal and Exponential were immune throughout
  because they have closed-form estimators and never touch the
  transform; the Uniform reports no covariance at any scale by design,
  its MLE being an order statistic rather than a stationary point.

  Clamping the dead branch's argument fixes it: the branch is
  responsible only for ``x < 0``, so restricting what it may be handed
  leaves its value and derivative untouched where it is used, and
  bounded where it is not.

  The hessian was never the problem -- the numerical fallback (#270)
  produced a finite, well conditioned matrix at every scale -- which is
  why this presented as nan bounds rather than as a warning or a
  failure. It also explains the speed: the same nan reached the
  objective's gradient, so BFGS, TNC and Newton-CG each gave up and
  Nelder-Mead finished the fit derivative free. Across twelve
  distributions at seven scales the sweep goes from 19.59s to 1.18s, and
  every fit that used to end on Nelder-Mead now ends on BFGS.

  Results that already worked are unchanged: 65 of 96 reference fits are
  bit-identical, and the other 31 are the restorations, where the
  objective agrees to fifteen significant figures and the parameters to
  nine.

  Restoring the gradient exposed a second, smaller scale problem
  underneath, now fixed with it. scipy stops BFGS when
  ``max|grad| < gtol``, and its default of 1e-5 is an absolute threshold
  on a quantity that is not scale free: a log-likelihood's gradient
  shrinks like ``1/theta``, so on data measured in tens of thousands the
  test is met well short of the optimum and BFGS reports success on its
  first check. This had been invisible because those fits used to end on
  Nelder-Mead, which is derivative free and so kept going. Three
  reference fits on real data of that magnitude landed 1e-2 away in
  relative terms, at a likelihood 2e-3 below the answer they had been
  recorded from.

  There is a second dimension to the same problem, in the opposite
  direction. A log-likelihood is a *sum* over observations, so its
  gradient grows like ``n`` as well as shrinking like ``1/theta``. At
  n = 1e5 it is five orders of magnitude larger, the same absolute
  threshold is correspondingly unreachable, and BFGS gives up on
  censored samples it should handle easily -- found by benchmarking
  against lifelines with censoring, where it was the one configuration
  in 64 where surpyval was slower.

  So the problem is rescaled in both of its dimensions, and a single
  constant then means the same thing for every fit::

      s = max(|u0|, 1)        f0 = max(|f(u0)|, 1)
      v = u / s               g(v) = f(s v) / f0

  The starting point is order 1 in every component and so is the
  objective, whatever the units and whatever the sample size. Since
  ``dg/dv = s * df/du / f0``, with ``s`` growing exactly as ``df/du``
  shrinks and ``f0`` growing like ``n``, so is the gradient. The
  ``gtol`` of 1e-6 applied there is a genuine relative tolerance rather
  than the dimensioned constant scipy's default is.

  Tuning the threshold was tried first and does not work. Three
  criteria were measured against the same reference set: an absolute
  ``gtol``, a ``gtol`` scaled by the gradient at the initial guess, and
  BFGS's step-size test ``xrtol``. Swapping the whole method for
  L-BFGS-B to reach its relative ``ftol`` was measured too. None is
  scale free in practice.

  Scaling ``gtol`` by the initial gradient in particular *looks* scale
  free and is not: the initialiser scales with the data too, so that
  gradient is itself roughly scale invariant -- a Weibull at scale 1,
  1e4 and 1e6 all came out with ``gtol = 1.86e-6``. Nor is there a
  constant that serves every case: tight enough for a Weibull at 1e6 is
  unreachable for an n=8 sample, which then drops out of BFGS into TNC.
  ``xrtol`` and ``ftol`` fail differently -- both stop on how the
  optimiser is behaving rather than on the quantity that is zero at the
  answer, so they quit early along flat directions, which is precisely
  where the standard error is largest and most needs to be right. The
  ExpoWeibull, three parameters and a flat surface, was 17% out under
  both. The full measurements are in #323.

  Rescaling beats every one of them, and is faster than the tolerance it
  replaces:

  ==============================================  ==========  ==========
  scale-equivariance of ``se`` (8 distributions)  relative    rescaled
                                                  ``gtol``    problem
  ==============================================  ==========  ==========
  worst deviation                                 1.6e-3      2.0e-5
  cases above 1e-5                                1 of 16     1 of 16
  Weibull n=1e5, 30% censored, scale 1            0.163s      0.153s
  Weibull n=1e5, 60% censored, scale 1e6          0.250s      0.119s
  ==============================================  ==========  ==========

  Rescaling the parameters alone reaches 2.8e-8 on the first row, better
  than the 2.0e-5 above, but is the version that leaves BFGS failing at
  large ``n``: those two censored fits take 0.370s and 0.590s under it.
  Normalising the objective as well trades a little of that accuracy for
  a criterion that holds across sample sizes too, which is the point.

  Both mappings are fixed before the search begins and neither can move
  the optimum: a diagonal linear change of variable relocates a minimum
  no more than dividing the objective by a positive constant does. They
  change the route taken and the units of the convergence test, nothing
  else. ``res.x`` and ``res.fun`` are both mapped back inside the
  helper, so no scaled quantity exists anywhere else in the package, not
  even transiently: the covariance step, ``cb`` and serialisation all
  receive exactly what they received before. Nothing needs to know which
  parameter is a scale and which a shape, which is what made the
  internal-rescaling proposal in #323 risky; preconditioning needs none
  of it.

  Two consequences worth noting. Fits now converge further than before
  wherever BFGS wins, so a handful of pinned numbers moved in their last
  few digits -- always towards a better likelihood. The two Monte Carlo
  simulation tests in ``test_counting.py`` also had their tolerance
  loosened from ``allclose``'s 1e-5 to 1e-3: they drive a 5000-run
  simulation from an optimiser's output, where a change in the seventh
  significant figure of a parameter moves the simulated MCF in the
  fourth. That is convergence noise, and what those tests exist to catch
  would break far more loudly.

  The second is that it flushed out a separate defect, which is fixed
  alongside it and described next.

  #323 is now rescoped. It had proposed rescaling the *model* -- fit on
  transformed data, then map the parameters and the covariance back --
  to fix both the bounds and the speed. Neither needs it. The bounds
  were the ``np.where`` overflow above, and re-measured with the
  gradient working, rescaling the data is between 4.2x faster and 6x
  *slower* depending on the distribution. What was left, the convergence
  criterion, is what preconditioning the search addresses, so the twelve
  per-distribution back-transforms are not needed for any of it.

- **Cox fits are 3x to 10x faster, with the answers unchanged to every
  digit.** None of this is a change to the maths; the coefficients still
  match ``lifelines`` to between 1e-06 and 1e-12, exactly as before.

  Four things were costing the time (#329).

  ``_GroupBy.sum`` used ``np.add.at`` for its multi-dimensional case, an
  unbuffered scatter with no fast path, called about ten times per
  ``jac_hess`` on arrays of shape ``(n, p, p)``. It is now a sorted
  ``np.add.reduceat``, with two shortcuts: nothing to permute when the
  keys already arrive grouped, and nothing to *add* when every key is
  distinct — one-element groups in order are the input array, which is
  what continuous event times give you.

  The hessian was a Python double loop over event times and tied deaths,
  with an ``np.outer`` inside it. The sum over tied deaths turns out to
  factor out of the ``p x p`` part entirely: only ``c = j / d`` depends
  on ``j``, so five scalar sums per event time carry the whole ragged
  axis and the covariate blocks are formed once. That drops the cost from
  ``O(times x ties x p^2)`` to ``O(times x ties + times x p^2)``, and it
  removes the separate no-ties case rather than special-casing it — an
  untied time is a single ``j = 0`` term with ``c = 0``.

  ``np.einsum("ij,ik->ijk", Z, Z)`` was rebuilt inside ``jac_hess`` on
  every root-finding iteration, though ``Z`` is fixed for the life of the
  fit. It is hoisted, and the two remaining weighted-outer einsums are
  plain broadcasts.

  Finally the rows are put in event-time order once when the closures are
  built, so ``_GroupBy`` never has to permute an ``(n, p, p)`` array
  again. Nothing downstream depends on row order — every quantity is
  aggregated to unique event times first — and the model still stores the
  caller's unsorted arrays for the residual and diagnostic code.

  Wall clock, Efron ties, 40% censored, against ``lifelines``:

  ===========  ======  =========  ========  ===========
  n            p       before     after     lifelines
  ===========  ======  =========  ========  ===========
  500          2       0.038s     0.012s    0.060s
  2 000        2       0.146s     0.024s    0.146s
  10 000       2       0.890s     0.105s    0.619s
  50 000       2       5.71s      0.663s    3.13s
  10 000       5       1.379s     0.339s    0.656s
  50 000       5       7.57s      2.01s     3.18s
  2 000        10      0.378s     0.107s    0.164s
  50 000       10      15.39s     8.31s     4.03s
  ===========  ======  =========  ========  ===========

  Heavily tied event times — dates, rounded durations — gain the most:
  n=20 000 with five covariates goes from 1.91s to 0.33s. Breslow, which
  shares ``_GroupBy`` and the einsum hoist, improves alongside it. The
  delayed-entry and time-varying-covariate paths, already well ahead,
  roughly halve again: 43 384 rows with delayed entry from 6.86s to
  2.62s, and 20 000 start-stop rows from 1.07s to 0.44s.

  Two cases remain slower than ``lifelines``: fifty thousand rows with
  ten covariates (8.3s against 4.0s), and heavy ties (0.33s against
  0.08s). Both are now dominated by materialising ``(n, p, p)`` arrays —
  40MB apiece at that size, several per iteration. Getting past that
  means accumulating the ``p x p`` information incrementally per event
  time instead of building per-observation outer products, which is a
  change of algorithm rather than of implementation, and is tracked
  separately as #332.

  Timings are single runs and drift by 10–20% between them; the
  ``lifelines`` column moves about as much as the surpyval one does.

- **Parametric proportional hazards fits are up to 6x faster and no
  longer degrade on data measured in large units.** A ``WeibullPH`` fit
  at data scale 1e6 settled 1.5 nats of log-likelihood short of the
  optimum, and 1e-2 away in the covariate coefficients -- a different
  fitted model, not a tolerance artefact. The same data in unit scale
  fitted correctly, so nothing about it looked wrong.

  The PH ladder was ``minimize(fun, init_t)`` followed by TNC, and three
  things were the matter with it. The objective closes over
  ``regression_neg_ll``, which is written in ``autograd.numpy`` and is
  therefore differentiable, but no ``jac`` was passed -- so scipy fell
  back to a two-point finite difference and paid ``p + 1`` extra
  objective evaluations per gradient, which is what made the fit slow
  down as the covariate count rose. The search was not preconditioned,
  so PH inherited the scale sensitivity fixed for univariate MLE
  elsewhere in this release: scipy stops BFGS on an absolute threshold
  applied to a gradient that shrinks with the data magnitude and grows
  with the sample size. And TNC's answer was returned whether or not it
  had converged, so a rung that can only ever be an improvement was free
  to be a regression -- the AFT and PO ladder, defined immediately
  below it, already guarded against exactly that.

  The ladder is now preconditioned BFGS on the analytic gradient, then
  TNC, then Nelder-Mead, stopping at the first rung that converges and
  never returning a worse point than it started from. The
  derivative-free rung stays for fits where the gradient is unusable.

  Measured against ``lifelines``, scoring both packages' answers on an
  independently written Weibull PH log-likelihood: the scale-1e6
  shortfall goes from 1.468 nats to 1.1e-08, and a 50 000 x 10 fit drops
  from 1.53s to 0.40s against ``lifelines``' 2.38s. ``ExponentialPH`` at
  the same size goes from 1.21s to 0.14s. Coefficients continue to agree
  with ``lifelines`` to around 1e-06.

  ``preconditioned_bfgs`` moved from ``fitters/mle.py`` up to
  ``fitters/__init__.py``, alongside ``bounds_convert`` and
  ``fallback_minimize``, so the univariate and regression ladders share
  one copy rather than two. Behaviour of the univariate ladder is
  unchanged.

  ``optimise_nm_tnc``, which serves AFT and PO, has the same missing
  gradient and missing preconditioning. Its first rung is Nelder-Mead,
  which is derivative free and so cannot fail the way BFGS did here, so
  it is being measured before it is changed rather than assumed to need
  the same fix — #331.

- **Turnbull no longer rejects left-censored observations under left
  truncation.** An entry time below every observation excludes nobody,
  so it should leave a fit untouched. With any left-censored row present
  it raised instead:

  .. code-block:: text

      ValueError: An observation's censoring interval does not intersect
      its own truncation window ...

  A support index ``j`` stands for the half-open interval
  ``(bounds[j], bounds[j+1]]``, so an event placed there is already
  strictly after ``bounds[j]``. The first index a row entering at ``tl``
  may use is therefore the *last* bound equal to ``tl`` -- that interval
  is ``(tl, next]``. The window construction took one index further on,
  discarding it.

  It mattered most for left censoring because such an event lies in
  ``(-inf, xr]``, which under an entry at ``tl`` is the single interval
  ``(tl, xr]`` -- frequently the only one the row has. Dropping it left
  the row with an empty support, hence the rejection.

  Neither endpoint of the search alone is correct, which is what made
  this awkward. ``side="left"`` keeps the zero-width ``(tl, tl]``
  interval that a duplicated exact event time creates, readmitting an
  event at exactly the entry time and breaking the strict
  ``(entry, exit]`` convention (#260); ``side="right"`` discards
  ``(tl, next]`` as well, one too many. ``side="right" - 1`` lands
  between them, and does so exactly, because every finite truncation
  time is itself in ``bounds``.

- **A Turnbull fit that is not identifiable now says so instead of
  returning a collapsed curve quietly.** Left-censored observations
  combined with two or more distinct entry times admit a flat direction
  in the likelihood, and where the data leans on it the estimate is
  worthless while looking ordinary.

  An interval that one observation could have failed in, but that
  precedes another observation's entry, is worth mass to the first and
  costs the second nothing. The second's contribution is conditional on
  its own entry, so mass it never had the chance to see divides out of
  both its numerator and its denominator exactly. On a six-point example
  the estimator drives 99.995% of the mass into a single such interval,
  reaching a log-likelihood of -6.14 against -9.36 for the sensible
  answer.

  So the estimator is not misbehaving. It is maximising correctly, and
  the likelihood has no interior maximum to find -- it climbs towards
  the boundary, which is why raising ``max_iter`` never helps. Both
  ingredients are needed: left censoring, the only kind whose support
  reaches back into the entry region, and two distinct entry times, so
  that such an interval exists at all. Six distinct entry times with no
  left censoring fit flawlessly; one common entry time with left
  censoring round-trips exactly.

  The fit is still returned, because meeting the condition does not mean
  the data is spoilt. Across 240 simulated samples that all met it, the
  proportion actually degenerating ran from 8% to 72%, rising with the
  share left censored -- rejecting on structure would refuse far more
  good data than bad. What separates the two is how much mass ends up on
  the flat direction: healthy fits reached at most 0.836 of it, spoilt
  ones a median of 0.994. Warning above 0.9 caught them without a single
  false alarm across those samples; 0.7 would have cost 9% and 0.5
  40%.

  The share is reported as ``model.exploitable_mass`` so a borderline
  fit can be judged rather than guessed at. Note that the structural
  condition is *not* used as a trigger on its own: ordinary
  staggered-entry data meets it routinely and estimates perfectly well,
  and pairing it with non-convergence would have mis-advised the #203
  case, which is structurally exploitable but converges given the
  iterations.

  This is the second half of #308, which closes with it; the first half,
  an off-by-one that made these same inputs raise, is above. The
  threshold is a measured cut-off standing in for a property that is
  actually decidable: Vardi (1985) and Wang (1991) give a graphical
  condition on the data that settles whether the NPMLE exists, exists
  but is not unique, or does not exist at all, with nothing to tune.
  Adopting it, and the question of what a non-identifiable fit should
  *return* rather than merely report, are #327. Worth noting alongside
  that left truncation with interval censoring is documented as yielding
  an inconsistent NPMLE, so this is a known limit of the estimator for
  this data shape rather than something particular to surpyval.

- **Truncated parametric regression fits could report a log-likelihood
  tens of thousands higher than their parameters earn, and be optimised
  towards it.** ``truncation_correction`` computed the mass in each
  observation's truncation window as a difference of CDFs, floored at
  the smallest positive float:

  .. code-block:: python

      np.log(np.maximum(right - left, _TINY))

  Under left truncation that difference is ``1 - F(tl)``, the survival
  probability at the truncation bound, which underflows to exactly zero
  as the fitted scale shrinks. The floor then capped the correction at
  ``log(tiny) = -708`` rather than letting it grow without bound -- and
  since the correction is *subtracted*, every truncated row appeared to
  contribute +708 to the log-likelihood. A region the data rules out
  entirely became the best fit on offer, and the optimiser walked
  straight into it.

  A ``WeibullPH`` fit to left-truncated data reported ``neg_ll``
  -21311.40 at parameters whose true value is +17118.30, against 927.83
  at the correct answer: wrong by 38,000, and pointing the wrong way.
  Recomputing the likelihood by hand from the model definition is what
  settled it -- at the correct parameters surpyval agrees to the digit,
  so the objective is right everywhere except where the floor engages.

  One-sided windows are now evaluated in log space through ``log_sf``
  and ``log_ff``, which stay finite where the difference cannot, so
  there is nothing to floor. Only a genuinely two-sided window still
  takes a difference, and there both bounds are finite and the mass is
  not driven to zero by the scale alone. As elsewhere, the ``np.where``
  branches are evaluated at substituted-finite arguments so that an
  infinity in an unselected branch cannot poison the gradient of the
  selected one.

  This is in ``_likelihood.py``, which serves proportional hazards,
  proportional odds, accelerated failure time and accelerated life
  alike, so any left- or right-truncated parametric regression fit was
  exposed -- it needed only the optimiser to wander far enough for the
  underflow to bite. Nothing warned when it did.

  Found by the rescaling change above, which perturbed an initial guess
  by six parts in ten million and was enough to tip one fit over. The
  first diagnosis was wrong: it looked like a genuinely unbounded
  truncated likelihood being followed legitimately, and the arithmetic
  disproved that. #326 records both. The regression test asserts the
  reported objective equals an independently computed one and that
  shrinking the scale below the truncation bounds always scores worse.

- **A truncated fit is around 60x faster, and the truncation term is
  evaluated once per distinct window rather than once per row.** Any fit
  with a truncation bound on one side only had no usable gradient. The
  window probability chose between the CDF and an analytic limit with
  ``np.where``, which picks the right *value* but evaluates both
  branches -- so ``ff(inf)`` was still recorded by autograd, and its nan
  derivative propagated through the selection whichever side won.

  Nothing warned. The objective was correct throughout; only the
  gradient was nan. So BFGS and Newton-CG each gave up after a single
  evaluation, TNC spent its whole 1000-evaluation budget discovering the
  same thing, and Nelder-Mead finished the job derivative free. A
  Weibull that fits in 0.014s took 1.36s, and a ``tl`` of 0 -- a no-op,
  since ``F(0) = 0`` -- cost exactly as much as a real truncation, which
  is what gives the cause away. Windows with *both* bounds finite were
  always fast, because no infinity ever reached the tape.

  The infinity is now substituted out of the *argument* before the CDF
  sees it, so a single vectorised call covers every row whatever its
  pattern of bounds, and the surviving ``np.where`` only ever chooses
  between two values that are already finite. The stand-in cannot be an
  arbitrary constant: zero looks natural and is wrong, because a Weibull
  with ``beta < 1`` has an unbounded density derivative at the origin,
  which would swap one nan gradient for another. Reusing a bound that is
  genuinely present keeps it inside the support and at the data's own
  magnitude; its value never reaches the result, only its derivative has
  to be finite.

  Separately, the truncation correction depends only on the observation
  *window*, not on where in it the observation fell, so it is now
  evaluated once per distinct window. Truncation is nearly always common
  to a whole sample -- one burn-in time, one study entry date -- which
  collapsed 360 CDF evaluations per likelihood call to one in the test
  case, and the likelihood is called hundreds of times per fit.

  ==============================  ==========  =========
  fit                             before      after
  ==============================  ==========  =========
  plain                           0.014s      0.015s
  left truncated                  1.398s      0.022s
  ``tl = 0`` (a no-op)            1.362s      0.041s
  right truncated                 1.344s      0.024s
  both bounds finite              0.019s      0.025s
  ==============================  ==========  =========

  Fitted results are unchanged: all 330 reference fits across thirteen
  distributions and five methods are bit-identical, and BFGS now wins
  every truncated fit where Nelder-Mead used to.

  Confidence bounds were never affected. The covariance step already
  recomputes a numerical hessian whenever the autograd one comes back
  nan or asymmetric (#270), so it caught this on every truncated fit and
  produced correct bounds by the slow route -- checked against
  ``906f0cb~1``, where the standard errors are identical to eight
  figures. That fallback was part of what made these fits slow.

- **The slow parts of the test suite are opt in, and there is a new
  invariant sweep behind the same mechanism.** ``pytest`` alone now runs
  in about two minutes rather than three: the beta survival tree and
  forest tests were 97 of the suite's 180 seconds for 85 of its 2000-odd
  tests. They run with ``--run-ml``, and continuous integration passes
  the flag, so coverage is unchanged. The ``conftest.py`` that defines
  the flags lives at the repository root: ``pytest_addoption`` is only
  honoured in *initial* conftest files, and the CI invocation selects
  with ``--ignore`` and names no path, so one under ``surpyval/tests``
  would be loaded too late to register them.

  ``--run-invariants`` adds ``test_fit_invariants.py``, a wide net over
  the fitting API. Every defect found in this release cycle slipped past
  the whole suite, and each lived at an *intersection* of dimensions the
  suite tests one at a time -- a censored observation that was also
  truncated, an offset combined with a particular method, an offset
  combined with a large shift magnitude, a sample below the 1000 floor
  of ``FIT_SIZES``. The full cross of distributions, methods, censoring,
  truncation, structural flags, sizes and scales is around 600,000
  cells, so the sweep does not attempt it. It asserts cheap invariants
  instead -- finite parameters, finite ``neg_ll``, a survival function
  that stays in [0, 1] and never increases, and maximum likelihood
  attaining the lowest negative log-likelihood of the five methods --
  over a seeded sample of that space. Four of the five defects would
  have failed the first two assertions.

  Data *scale* is included as an axis because it was previously untested
  anywhere, despite the maximum likelihood failure warning itself
  advising users to rescale towards 1. 270 cases, three and a half
  minutes.

- **Maximum likelihood fits are about 2.2x faster.** Every MLE fit ran
  five optimisers -- Nelder-Mead, Powell, BFGS, TNC and Newton-CG -- and
  kept the best result. Over 102 fits across eleven distributions, five
  data shapes and two sample sizes, all five agreed on the objective to
  1e-10. The last four were confirming what an earlier one had already
  found.

  That confirmation was not cheap. Nelder-Mead and Powell are derivative
  free, so they pay for robustness in function evaluations -- 50 and 22
  against BFGS's 21 -- and every evaluation costs O(n). On a million
  observations those two alone were 42% of the fit.

  The gradient methods now run first and the search stops at the first
  that converges, with the derivative-free pair kept as the fallback.
  Order and early exit had to change together: stopping early without
  reordering halts at Nelder-Mead, which is both the most expensive rung
  and the one with the worst objective, while reordering without
  stopping early saves nothing. Cold-start BFGS now wins 83 of 102 fits,
  TNC takes 10 and Newton-CG one; the eight that Nelder-Mead or Powell
  used to win now land on a gradient method at the same objective, so
  they were winning ties on ordering rather than finding better optima.
  The derivative-free methods still start from the cold initial guess
  when they are reached, so the multi-start behaviour survives for the
  fits that need it.

  **Fitted parameters can move in about the seventh significant digit.**
  All 102 objectives are identical to 1e-10 and one improved, so this is
  optimiser tolerance rather than a change of answer, but it is not
  bit-identical: the median shift is 3e-8 and the 90th percentile 8e-7.
  The documented ``GeneralizedOneRenewal`` example and the two tests
  that pin it have been regenerated. Those numbers were always a
  snapshot of the library's own output rather than an external
  reference, and their tolerance has deliberately been left tight, so
  that any future change to the optimiser surfaces as a decision rather
  than passing unnoticed.

- **Degenerate data is rejected with an explanation instead of an
  ``IndexError`` from inside numdifftools.** ``Weibull.fit`` on three
  tied observations died four steps from the cause: a probability plot
  has no slope through a single distinct abscissa, so ``polyfit``
  returned a nan; the nan seeded the maximum likelihood fit, which
  started at nan and produced a nan hessian; the numerical fallback then
  asked numdifftools for one, and its list of finite-difference steps
  came back empty. Neither truncation nor censoring was involved,
  despite where the symptom was first seen.

  ``Gamma`` and ``Beta`` failed the same way but in silence. Their
  moment-based initialisers divide by a variance that is exactly zero
  for tied data, giving ``(inf, inf)``, and since a failed optimiser
  reports its initial guess (#261) those infinities were returned as a
  fitted model.

  Three changes. The probability-plot regression falls back to a unit
  slope through the centroid when it is rank deficient -- zero slope
  would be the more literal reading, but every ``unpack_rr`` divides by
  the slope to recover a scale, so it only moves the nan one step later.
  ``Gamma`` and ``Beta`` seed the exponential and uniform cases rather
  than dividing by zero. And a fit now refuses to return a non-finite
  parameter whatever produced it.

  The fit is then rejected when the data cannot pin down the free
  parameters: fewer distinct non-right-censored values than free
  parameters means a flat -- for a Weibull on tied data, unbounded --
  direction in the likelihood, and the answer would be wherever the
  optimiser stopped. Three tied observations returned ``beta = 512``
  with ``success=True`` and no warning once the nan was fixed.

  The count is of *free* parameters, so fixing one buys back a degree
  of freedom: ``Weibull.fit([10.], fixed={'beta': 2})`` is well posed
  and now returns ``alpha = 10``, where before it raised. One-parameter
  distributions are unaffected -- ``Exponential`` and ``Rayleigh`` fit
  tied data exactly as they should. Probability plotting is exempt,
  being a regression rather than a likelihood maximisation, and is how
  several distributions seed themselves.

  All 330 reference fits across thirteen distributions, five methods and
  plain, right-censored and offset data are bit-identical.

v0.18.0 (2 August 2026)
-----------------------

- **Documentation caught up with the estimator changes below.** The
  most consequential correction is in :doc:`Parametric SurPyval
  Modelling`, whose section on offset unidentifiability demonstrated the
  hazard of threshold parameters by fitting ``Gamma(3, 2) + 10`` and
  reporting that ``MPP`` returned a negative ``gamma`` with a shape
  parameter inflated by two orders of magnitude. That has not been true
  since #257 and #313: every fit method now recovers the offset to three
  decimal places, at offsets from 10 to 1000 and at both small and large
  samples. The page had also drifted from the test file it cites --
  ``test_offset_divergence.py`` was tightened by #257 and #275 to assert
  parameter *recovery*, the opposite of what the prose claimed.

  The underlying theory is kept, since it is exactly what made a poor
  starting point so damaging: ``gamma`` trades off against the shape and
  scale, and the likelihood is flat along that ridge. It now reads as a
  caution for your own data rather than a demonstration of broken
  output, and names both causes of the old behaviour separately -- the
  probability-plotting search stranded by a single starting shape, and
  the moment-based initialisers taking their moments before the shift.

  The maximum likelihood notes in :doc:`Parametric Estimation` now cover the
  observation that is *both* censored and truncated -- the contribution
  it makes, why the numerator has to be capped by the truncation bound,
  and the fact that surpyval recasts such a point as interval censored
  in its internal representation rather than special-casing the
  likelihood. The user's own ``x``, ``c``, ``n`` and ``t`` are untouched,
  which is now said explicitly.

  The method of moments section explains that the optimisation matches
  scaled *central* moments rather than raw ones, and why that matters for
  offset data, where every raw moment is dominated by the offset.

  :doc:`Parametric SurPyval Modelling` gains a worked comparison of
  ``neg_ll``, ``aic`` and ``bic`` across all five fit methods, showing
  both that the criteria are available whatever the method and that MLE
  attains the lowest negative log-likelihood -- a check that could not be
  run before.

- **An offset ``ExpoWeibull`` fit now seeds itself from the shifted
  data.** ``ExpoWeibull`` starts from a Gumbel fit to ``log(x)``, since
  a Weibull's logs are Gumbel distributed. With ``offset=True`` it took
  those logs *before* removing the shift, so it read ``log(x)`` where
  the model wants ``log(x - gamma)``. A large offset compresses those
  logs into a narrow band, the Gumbel ``sigma`` collapses, and
  ``beta = 1 / sigma`` explodes: on 500 points from ``ExpoWeibull(10,
  2, 1) + 100`` the seed came back as ``alpha = 111, beta = 23.5``
  against a true 10 and 2. The maximum likelihood fit then failed
  outright, returning ``nan`` and warning its way back to the MPP
  estimate.

  This was not an edge case. Over 120 offset fits -- four offsets, five
  parameter sets, six replicates each -- **54 returned nan**, which is
  every configuration at an offset of 100 or 1000. All 54 now converge,
  none of the 66 that already worked changed for the worse, and the
  whole sweep takes 25.5 seconds against 188.3, since a hopeless
  starting point is expensive to fail from.

  The offset is now estimated first and the shape parameters read off
  ``x - gamma``. It is estimated as ``min(x) - 1``, which is what the
  fitter installs regardless of what the initialiser returns -- seeding
  against a different shift than the one being optimised under defeats
  the point.

  The nested Gumbel MLE that refines the offset seed is kept. Removing
  it was tried, on the reasoning that shifting the data correctly makes
  the probability plot alone good enough; it is not, and five of 48
  offset fits landed on a worse optimum without it.

- **``aic``, ``bic``, ``aic_c`` and ``neg_ll`` now work for every fit
  method.** They were available only after a maximum-likelihood or
  closed-form fit, because only those compute a log-likelihood on the
  way to the answer. A model fitted with ``how='MPS'``, ``'MSE'``,
  ``'MOM'`` or ``'MPP'`` raised ``AttributeError`` from all four --
  which meant the usual way of choosing between distributions was
  unavailable for four of the five methods, and failed with a message
  that did not say why.

  The log-likelihood is a property of the parameters and the data, not
  of the search that found them, so it is now evaluated after any fit.
  Methods that already reported one keep it exactly: maximum
  likelihood's is the optimiser's own final objective, which on its
  fallback path is deliberately taken at the initial guess rather than
  at the failed result (#261).

  A worked consequence, on 500 Weibull(10, 2) points -- the maximum
  likelihood estimator attaining the maximum likelihood, which was not
  checkable before:

  ===========  ==========
  method       ``neg_ll``
  ===========  ==========
  MLE          1466.3254
  MPS          1466.3865
  MOM          1466.3979
  MSE          1466.6759
  MPP          1470.7119
  ===========  ==========

- **MSE and MPS fits try BFGS before Newton-CG, and are several times
  faster for it.** Both go through a shared fallback that reached for
  Newton-CG first, which needs a hessian. Building one is
  disproportionately expensive for the distributions whose derivatives
  autograd cannot take analytically -- the incomplete gamma is
  central-differenced, so every second-order entry costs a difference of
  differences. An offset Gamma MSE fit at n=5000 spent 8.2 of its 8.3
  seconds there, and BFGS reached a marginally *better* optimum in half
  a second.

  Reversing the order was checked over 132 fits: MSE and MPS, nine
  distributions, at two sample sizes, on plain, right-censored,
  left-censored and offset data. 129 objectives came back identical,
  three improved, none got worse, for 3.9x less time overall. Newton-CG
  is still there, escalated to when BFGS fails, and Nelder-Mead behind
  it; the zero-hessian guard is kept, since a hessian of zeros makes
  Newton-CG stop at the initial guess while reporting success, so there
  is nothing to escalate to and Nelder-Mead should take over.

  ``scipy.optimize.least_squares`` was tried first, on the reasoning
  that the MSE objective is a sum of squares and Gauss-Newton should
  exploit it. It is far worse: it needs the full residual jacobian, so
  n=5000 means 5000 rows each paying that central-differenced
  derivative, and the same fit took 239 seconds against the scalar
  gradient's three numbers.

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