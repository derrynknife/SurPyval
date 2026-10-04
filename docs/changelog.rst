Changelog
=========

Unreleased
----------

**Removed.** The names 0.23 deprecated are gone; each now raises.

- The limited-failure proportion's old name ``p`` is ``lfp_p``: in
  ``fixed``, ``param_cb``, ``from_params`` and the attribute of a
  univariate or regression model (#608). ``model.p`` raises an
  ``AttributeError`` naming ``lfp_p``; on Bernoulli, Binomial, Geometric,
  NegativeBinomial and FixedEventProbability it is still the fitted
  parameter. Saved dicts keep the key ``"p"``.
- Regression coefficients' old names ``beta_0``, ``beta_1``, ... in
  ``fixed`` and ``param_cb`` are the covariate's column name, else
  ``coef_0``, ``coef_1``, ... (#614); dicts saved with them still load.
- ``se`` (Cox, Lin-Ying, proportional odds, Fine-Gray) is
  ``standard_errors()`` (#613).
- ``Parametric.cov_matrix``, ``cov`` (Fine-Gray, proportional odds,
  additive hazards) and the ``covariance`` attribute (Royston-Parmar, the
  frailty models) are ``covariance()`` (#605).
- Recurrent models' ``aic`` and ``bic`` without the call are ``aic()``
  and ``bic()`` (#572).
- ``MixtureModel.log_likelihood(params)`` is gone: ``log_likelihood`` is
  the fitted value, a plain ``float`` (#572); ``MixtureModel.loglike``
  (the *negative* log-likelihood) is ``neg_ll()``.
- ``MixtureModel``'s EM steps ``EM``, ``Q``, ``expectation``,
  ``maximisation``, ``likelihood`` and ``initialise_params`` have no
  public names; they are internal to the fit (#605).
- A Cox model's ``neg_ll(beta)`` is ``neg_ll_of(beta)``; ``neg_ll()`` is
  the fitted value (#604).
- CoxFrailty's ``loglik`` and ``loglik_no_frailty`` are
  ``log_likelihood`` and ``log_likelihood_no_frailty`` (#604).
- ``success_run``'s ``confidence=`` and ``alpha=`` are ``alpha_ci=`` (#580,
  keyword only: ``success_run(59, 0.95)`` raises a ``TypeError`` rather
  than reading 0.95 as ``alpha_ci``).
- ``surpyval.NUM``, ``TINIEST`` and ``EPS`` are ``numpy.float64``,
  ``numpy.finfo(float).tiny`` and ``numpy.sqrt(numpy.finfo(float).eps)``
  (#613); ``surpyval.utils.numeric`` no longer has them either.
- Development: ``REMOVED_IN`` is ``"0.25"`` (the names deprecated in
  0.24) and ``REMOVED_IN_NEXT`` ``"0.26"``; the removal test's scan
  checks that it sees every helper of ``surpyval.utils.deprecation``.
  Two mixture tests that counted EM steps by patching ``EM`` and ``Q``
  counted nothing since those became aliases; they now patch the
  internal methods.

**Performance**

- **Building a SurpyvalData is about a third faster.** Every fit builds
  one from its inputs first. The distinct truncation windows, and the
  distinct values the identifiability check counts, came from
  ``np.unique(..., axis=0)``, which sorts rows as structured records;
  each column is now ranked on its own and the pairs of ranks sorted as
  integers. A flat list of numbers is converted to an array once rather
  than once per check. The data are identical, dtypes included, and so
  are the errors: on 100,000 units with mixed censoring and truncation,
  from arrays 58 → 41 ms, from lists 89 → 62 ms, and the identifiability
  check 12 → 2.5 ms.
- **``fit_best`` builds its data once.** It checked the data as a
  ``SurpyvalData`` and then gave each candidate the raw inputs, so every
  family built and checked them again, and estimated the non-parametric
  start again. Every candidate is now fitted to that one ``SurpyvalData``
  (``fit_from_surpyval_data``). Results, warnings and errors are
  identical; on right-censored data 10,000 units 381 -> 346 ms and 100,000
  units 2.51 -> 2.08 s, a larger share wherever the fits themselves are
  quick.
- A fitted model's ``sf``, ``ff`` and ``df`` are about twice as fast on
  small arrays, and up to 2.4x on large ones for a model with no offset,
  limited-failure or zero-inflation part, with identical results (#642).
  Such a model now gives its distribution's own functions, skipping the
  transforms (identities there, but five passes over the query), and the
  support and missing-value checks around every distribution function use
  plain numpy rather than autograd's wrappers. A Weibull's ``sf`` on 16
  points: 58 to 28 us; an Exponential's on 20,000: 193 to 79 us.

**Added**

- ``MixtureModel`` has ``hf`` (``df / sf``), ``qf`` (``ff`` inverted
  numerically; NaN with a warning outside [0, 1]), ``covariance()`` and
  ``standard_errors()`` (the observed information, weights as softmax
  logits, carried to the parameters and weights; named by
  ``covariance_names``: ``alpha_0``, ..., ``w_0``, ...), and Wald
  ``param_cb``, ``cb`` and ``quantile_cb`` as ``Parametric``'s (#651).
- ``CompetingRisksProportionalHazards`` has the per-cause coefficients'
  inference: ``params``, ``parameter_names`` (``"a: grp"``),
  ``covariance()`` (block-diagonal, each cause's fit's),
  ``standard_errors()``, ``p_values``, ``summary()`` and a readable repr;
  ``FineGray`` has ``params``, ``parameter_names`` and ``summary()``.
  ``phi_e`` takes the cause label, as ``cif`` (the row index is
  deprecated), and ``beta`` (the causes' sum) is deprecated. A Fine-Gray
  model's ``log_likelihood`` raises ``AttributeError``, so ``hasattr``
  works (#656).
- **One API across the regression families (#662).** ``qf`` on ``CoxPH``,
  ``ProportionalOdds``, ``AdditiveHazards``, ``BuckleyJames`` and the
  frailty models (a step curve's: the first time it reaches ``p``, else
  ``nan``); ``mean(Z)`` and ``p_values`` on the parametric regressions;
  ``summary()`` tables on ``AdditiveHazards``, ``BuckleyJames`` and
  ``RoystonParmar`` (whose text is now its ``repr`` only); ``tl`` / ``tr``
  on the parametric fits; ``CoxFrailty.param_cb(method=)``. A model with
  no likelihood says why on ``neg_ll`` / ``aic``.

**Fixed**

- Removed names say what replaced them, and when they went (#653):
  ``from surpyval import Power`` raised a bare "cannot import name"; old
  attributes (``param_names``, ``se``, ``cov_matrix``, ``loglike``, ...),
  arguments (``cs(X=)``, ``ARI.fit(dist=)``, ``from_params(p=)``, ...),
  ``p`` and ``beta_j`` in ``fixed`` and ``param_cb``, and
  ``surpyval.utils.score`` now name the replacement, and a regression's
  ``fit_from_df(x=)`` names ``x_col``. ``surpyval.datasets`` works after
  ``import surpyval``; ``ProportionalIntensityNHPP`` takes ``baseline=``,
  as ``ARI`` does (``dist=`` warns until v0.25).
- Messages that sent users the wrong way (#663, items 5-9): copula margins
  given as names or as one fitter, series of unequal length and
  ``sf(50, 20)`` say what is expected; the power (and logarithmic,
  Lloyd-Lipow, Michaelis-Menten) path says to drop the t = 0 rows, and
  ``path="best"`` warns when it leaves those paths out; Wiener and Gamma
  fits of a falling signal say to negate y and the threshold; a
  zero-inflated ``quantile_cb`` below ``f0`` no longer warns of a NaN;
  ``sf("10")`` reads the number, and a dict's string or ragged entries
  are named.
- Inputs accepted that should be flagged (#664, items 2 and 3): a copula's
  ``kendall_tau``, ``spearman_rho`` and ``tail_dependence`` refuse a
  parameter outside the family's range (``Gumbel.kendall_tau(0.5)`` was
  -1.0); a Clayton, Gumbel or Joe fit that ends at its independence bound
  warns that the family cannot model the data's dependence, and
  ``param_cb`` no longer suggests profile or bootstrap bounds the copula
  models do not have. A ``WienerProcessModel`` or ``GammaProcessModel``
  pickled by 0.22 loads with ``y0 = 0``.
- ``SurvivalTree`` and ``RandomSurvivalForest`` predictions check the
  covariate count against the fitted model (#657): an extra column was
  ignored, giving plausible wrong predictions, and a missing one raised
  numpy's ``IndexError``; both now raise "The forest has 2 covariates
  (Z0, Z1); got 3 values". The count is saved in ``to_dict``.
- ``alpha_ci`` outside (0, 1) gave reversed or ``nan`` bounds in silence
  from ``cb``, ``param_cb``, ``mean_cb``, the non-parametric, regression,
  copula and degradation bounds and ``summary()`` (#647). Every method
  that takes it now refuses it with the one message ``quantile_cb`` gave
  (``surpyval.utils.validation.check_alpha_ci``), and warns, once per
  call, of a level above 0.5: ``alpha_ci=0.95`` is a 5% interval.
  ``success_run`` refuses 0 and 1 too.

- A regression prediction with a ``Z`` of the wrong width gave numpy's
  error (``shapes (3,) and (2,) not aligned``, ``operands could not be
  broadcast``) or a prediction from the wrong columns (#657). Every
  regression model's ``sf``/``ff``/``hf``/``Hf``/``df``/``qf``/``cs``,
  ``cb``, ``quantile_cb``, ``mean``, ``phi`` and ``random`` now say "The
  model has 2 covariates (coef_0, coef_1); Z gives 3 per row", and an
  accelerated life model's stresses against more times give the row-count
  message.
- A parametric regression model restored from ``to_dict`` gave ``cb``,
  ``cb(on="Hf")`` and ``quantile_cb`` up to 1e-11, 6e-11 and 2e-9 from the
  original's (#664): a fit on centred covariates (#463) computes its bounds
  in the centred parameters, the restored model at the reported ones. The
  dict now stores that state (``"inference_centring"``), and a PH model is
  rebuilt with the ``CovariateLink`` its fitter builds, not a
  ``LogLinearPhi``: the restored bounds are bit-identical.
- ``fit_from_df`` of ``CoxPH``, ``ProportionalOdds``, ``AdditiveHazards``,
  ``BuckleyJames``, ``CoxFrailty`` and the parametric frailty models names
  a missing column as the parametric families do (#663): "x_col='time' is
  not a column of the DataFrame; its columns are [...]", not a bare
  ``KeyError``. A missing ``Z_cols`` entry lists the columns too, for every
  regression ("Z_cols entry 'zz' is not a column ...").
- ``CoxPH`` and ``BuckleyJames`` refuse data with no event, with the
  message of ``ProportionalOdds`` and ``AdditiveHazards`` (#648): Cox
  returned coefficients 0 with standard errors 0 (a hazard ratio of exactly
  1, CI [1, 1]), and Buckley-James reported ``converged=True``. A Cox
  coefficient running off to infinity (separation) has a ``nan`` standard
  error, p-value and covariance row and column, not 0.
- An accelerated life model's Wald ``param_cb`` on a positive life-model
  parameter (Arrhenius's ``b``, Power's ``a``: any bounded (0, None)) is
  computed on the log scale, as a distribution's positive parameters are
  (#655); it went below zero, [-3.5e-06, 7.0e-06] for b = 1.75e-06, in
  ``summary()`` too. The regression models' ``param_cb``, ``cb`` and
  ``quantile_cb`` take ``method=None`` for the default, as the univariate
  models do.
- The Arrhenius-type life models (``Exponential``, ``InverseExponential``,
  the Eyring models, the temperature column of ``DualExponential`` and
  ``PowerExponential``) took a temperature in degrees Celsius in silence
  (#654): a 0 °C level crashed inside LAPACK, a negative one fitted a
  nonsense activation energy. A stress <= 0 there is now refused naming
  kelvin, and stresses all below 200 K warn "Did you pass degrees Celsius?
  Add 273.15" (not for ``Eyring``, also used for a non-thermal stress).
- ``fit_best`` passed over every lifetime family (support from 0) in
  silence when some times were at or below 0, and returned the best of the
  rest: with three zero ages in 50, a Normal that put 3% of the units
  failing before day 0 (#646). It now warns, naming the families passed
  over, the number of such times, what it chose, and ``zi=True`` for units
  that failed at time 0. A Beta passed over for data outside (0, 1) stays
  quiet.
- A mixture component that runs off past the data -- its failures all
  beyond the last observation, so it explains none -- was reported as a
  verified maximum (#650): a two-Weibull mixture came back with a second
  component of scale 33,561 (the largest observation 1,150) and weight
  0.09, which reads as a second failure mode. The likelihood keeps rising
  towards the other components with a limited-failure proportion, which
  no mixture reaches; the fit now warns "No finite maximum", says so, and
  points to ``lfp=True``, and ``fit_best`` sets it aside.
- A fit with a fixed parameter skipped the alternative starts every other
  fit tries, and could stop on a worse maximum and call it verified: a
  limited-failure Weibull with ``lfp_p`` fixed at 0.5 landed on the
  infant-mortality mode (shape 0.96), 11.3 log-likelihood units below the
  wear-out one the start from the failures alone reaches (#649). The
  alternative starts now run with the fixed values in place.
- An offset (3-parameter) model's Wald bounds -- ``cb``, ``quantile_cb``
  and ``param_cb`` -- hold the offset at its estimate, which the fit
  leaves out of the covariance (a threshold's likelihood is not regular),
  and said nothing: a 90% bound on a 3-parameter Weibull's B1 life from 30
  failures covered 40% of the time (#645). They now warn so, and the three
  take ``method="bootstrap"``, a parametric bootstrap whose bounds include
  the offset's uncertainty: the model is refitted to ``n_boot`` data sets
  (200 by default) simulated from it, each unit censored as it was, and
  the bound is the bias-corrected percentile interval of the refits. It
  is also the one bound on ``gamma`` itself
  (``param_cb("gamma", method="bootstrap")``). It needs the data, exact
  or right censored (truncated or not), and is not available for
  limited-failure or zero-inflated models; the refits are kept on the
  model for each ``n_boot`` and integer ``random_state``, so later bounds
  reuse them. An offset refit takes 0.1 to 0.3 s. On #645's design (shape
  1.5, 30 failures, two runs of 100 samples) the 90% bound on B1 covered
  64% and 63% (Wald: 35%), on B10 82% and 89% (69%), on B50 89% and 95%
  (85%), and on ``sf`` just above the offset 66% and 68% (53%): far
  better, but still short at the offset itself in small samples with a
  shape below 2, where the offset's estimate is biased up and no interval
  of the refits makes up for it (the percentile and basic intervals were
  tried too).
- The test suite installed with the package can be collected (#661): the
  reference results it reads (``tests/reference/data/*.json``) ship in the
  wheel; the doctest comparison's helpers and the opt-in gating live in
  the package (``surpyval/tests/_suite.py`` and ``conftest.py``), so the
  installed suite imports them and skips the ``--run-ml``,
  ``--run-invariants`` and ``--run-calibration`` groups as the repository
  does; and the modules that need an optional test dependency
  (hypothesis, scikit-survival, bson) are left out when it is not
  installed. The merge into ``master`` now collects the suite from the
  built wheel in a clean environment.
- An offset fit's ``gamma`` is capped by the smallest value that constrains
  it -- an exact failure, a left-censoring time (without zero inflation) or
  an interval's upper end -- rather than by the smallest value of any row
  (#633). On interval inspection data the offset was pinned at the first
  interval start and the bound reported as a verified maximum: 7.8
  log-likelihood units short on 500 Weibull units inspected every 3. A
  right-censored time or an interval start below the offset meets the
  survival function at 1, which the likelihood now holds there for a
  custom cumulative hazard too. A fit the cap stopped short of the first
  failure now reaches it: where the likelihood is unbounded there (a shape
  below 1) it warns "No finite maximum" instead of reporting the cap as a
  verified maximum.
- The same bound made a Rayleigh offset fit on interval data run onto the
  first interval start and warn "No finite maximum", where 0.22 found the
  interior maximum (#632, since 0.23).
- An offset fit with zero inflation and left-censored rows raised
  (``OutsideSupportError`` for a Weibull, "x cannot contain NaN" for a
  LogNormal) before the search started: the starting offset was taken
  above a left-censored row's imputed point (#631, since 0.23).
- Recurrent input (#658): a scalar ``i`` / ``c`` / ``n`` applies to every
  row, and ``fit_from_df(c="ev")`` names ``c_col`` (both were an
  ``IndexError``); an exact event given as ``[l, r]`` with l != r, which
  the fits read as ``l`` and the MCF as the midpoint, is refused; c=2 with
  1-D ``x`` is a ``ValueError``, not a bare ``AssertionError``; ``n > 1``
  on an exact event says to repeat the row; messages give the item's own
  label. An event exactly at ``tl`` is outside the window ``(tl, T]``:
  every fit, the MCF and the trend tests now refuse it with one message,
  and the MCF no longer counts an item at risk at its ``tl``.
- ``surpyval.forecast`` refuses a negative age for every model type, with
  the renewal models' message (#659): a univariate model forecast the
  window from before new, an intensity model returned NaN or a count. An
  ``n`` or ``limit`` with neither one value nor one per age says so,
  rather than raising numpy's broadcast error.
- Recurrent messages (#663): ``CrowAMSAA.projection`` on systems ending
  at different times said "Use method='wald'", which it does not take; it
  now says every system must run to the same end of test. Renewal fits
  (GeneralizedRenewal, ARA, G1) with too few distinct times between
  failures advised ``fixed=``, which they do not take; they now say so in
  recurrent terms. A model restored with ``from_dict`` says so when asked
  for bounds, rather than calling itself built from parameters.
- Renewal fits (GeneralizedRenewal, G1, ARA, ARI) warn of a negative
  ``tl`` (#664): on an age scale a negative entry age is almost always a
  data error, and it moved q from 0.28 to 0.42 in silence. It is still
  accepted, each item as new at its entry.
- The recurrent bounds refuse an ``alpha_ci`` outside (0, 1) with the
  package's one message (#647): ``cif_cb``, ``iif_cb``, ``mtbf_cb`` and
  ``param_cb`` of the intensity and proportional-intensity models,
  ``NonParametricCounting.mcf_cb``, ``CauseSpecificMCF.mcf_cb`` and the
  plots that draw them, and a renewal model's ``summary``. They returned
  reversed (1.5), equal (1) or NaN (below 0) bounds in silence.

v0.23 (4 October 2026)
----------------------

**Upgrading from 0.22.** A breaking release. Most code needs no change; the
items most likely to need one are listed here, and every entry below says
what changed. On 0.22, run your code or tests with ``python -W
error::DeprecationWarning`` first to find the calls to update.

- Names 0.22 deprecated now raise (see *Removed*): ``param_names`` is
  ``parameter_names``, ``fit_from_df(x=...)`` is ``x_col=``, ``cs(x, X=)``
  is ``given=``, ARI's ``dist=`` is ``baseline=``, and the life models are
  only in ``surpyval.life_models``.
- The limited-failure proportion is ``lfp_p``: in ``fixed``, ``param_cb``,
  ``from_params``, ``extras`` and the attribute (#608); on Bernoulli,
  Binomial, Geometric, NegativeBinomial and FixedEventProbability
  ``model.p`` is now the fitted parameter.
- Regression coefficients are named by their covariate's column, else
  ``coef_0``, ``coef_1``, ... (#614): code that looks up ``beta_0``, ...
  in ``parameter_names``, ``summary()`` or ``to_dict()`` must use the new
  names (``beta_j`` in ``fixed`` and ``param_cb`` warns until v0.24).
- ``standard_errors()`` gives an array on every model (#613): use it for
  ``se`` (Cox, Lin-Ying, proportional odds, Fine-Gray; deprecated until
  v0.24); the frailty models' ``standard_errors()`` is an array, not a
  dict, with theta's last.
- A censored time below the support (e.g. ``-1``) is refused by the
  univariate fits (#611) and the regressions (#565), as an observed one
  already was; drop or correct such rows.
- Offset fits refuse an observed value at ``±inf`` with
  ``OutsideSupportError`` (#622); remove those rows.
- Turnbull on untruncated data fits by the EM-ICM (#620), so its
  Fleming-Harrington (default) and Nelson-Aalen curves move; pass
  ``turnbull_algorithm="EM"`` for the old result.
- Renewal fits (``GeneralizedRenewal``, ``GeneralizedOneRenewal``,
  ``ARA``, ``ARI``) take ``tl``, each item as new at entry (#615); a
  negative ``tl`` used to be ignored and now changes the fit: drop it to
  keep the old one.
- ``GammaProcess`` and ``WienerProcess`` lives start at the fitted
  ``y0`` (#574); pass ``y0=0.0`` for the old life from zero.
- ``fit_best`` raises on malformed data, or when every candidate fails
  alike, rather than returning ``None`` (#570); catch the ``ValueError``.
- ``Parametric.param_cb``'s ``method`` defaults to ``None`` (#580): Wald,
  except exact Clopper-Pearson for Bernoulli, FixedEventProbability and
  Binomial; a wrapper that passes the old default ``"wald"`` on should
  pass ``None``.
- Model dicts store ``"_neg_ll"`` and ``"covariance"`` (#605): old dicts
  load, but a 0.23 dict loses its likelihood or covariance in an older
  SurPyval; code that reads the dict's keys must use the new ones.
- Comparison values are spelt alike (#572, #604): recurrent ``aic`` and
  ``bic`` are methods (``aic()``), ``MixtureModel.log_likelihood`` is the
  fitted value (``loglike`` is deprecated), and a Cox model's ``neg_ll()``
  is the fitted value (the function is ``neg_ll_of(beta)``).
- Survival trees and forests refuse times outside (0, inf) for the
  parametric kinds and an infinite failure for the non-parametric kind
  (#618), and zero-inflated fits refuse times outside the support (#610).
- The non-parametric ``qf`` gives NaN with a warning outside [0, 1]
  rather than raising (#611); test the result rather than catching
  ``ValueError``.

**Removed.** The names 0.22 deprecated are gone; each now raises.

- ``param_names`` (every distribution, fitter and model) is
  ``parameter_names``; calling ``parameter_names()`` on a regression model
  is ``parameter_names`` (a plain ``list`` now); ``CustomDistribution``'s
  ``param_names=`` keyword is ``parameter_names=``; a ``param_names``
  class attribute on your own ``PathModel``, ``Copula`` or
  ``CountingProcess`` subclass is no longer read (define
  ``parameter_names``); ``ProportionalIntensityModel.param_names`` (the
  base-rate names) is ``parameter_names[:len(params)]``. Saved files keep
  the key ``"param_names"``.
- ``fit_from_df``'s v0.21 column names (``x``, ``c``, ``n``, ``xl``,
  ``xr``, ``tl``, ``tr``, and ``x``, ``y``, ``i`` of
  ``DegradationAnalysis``, ``WienerProcess`` and ``GammaProcess``) are
  ``x_col``, ``c_col``, ...; the old ones raise a ``TypeError`` naming the
  new one rather than reaching ``fit``.
- ``cs(x, X=)`` is ``cs(x, given=)`` (by position, nothing changes).
- ARI's ``dist=`` is ``baseline=``, and ``fit_from_parameters``'s
  ``dist_params=`` is ``baseline_params=`` (#507).
- The life models' top-level names (``surpyval.Power``,
  ``surpyval.ExponentialLifeModel``, ...) are
  ``surpyval.life_models.Power``, ``life_models.Exponential``, ...;
  asking for an old name raises an ``AttributeError`` that says where it
  is.
- ``surpyval.utils.score.score`` is
  ``surpyval.metrics.concordance_index(x, c, risk, ties="harrell")``
  (Harrell's tie convention, which ``score`` had).
- Development: ``surpyval.utils.deprecation`` drops ``CallableList`` and
  ``renamed_class_attribute``, which only these names used.
  ``REMOVED_IN`` is ``"0.24"`` (the names deprecated in 0.23) and
  ``REMOVED_IN_NEXT`` ``"0.25"``; the removal test's scan now sees every
  deprecation helper.

**Changes.**

- **Regression fits with no finite maximum are recognised however far
  the search runs (#628).** With all the failures in one cell of a
  two-stress test there is no finite maximum, yet a ``WeibullAFT`` reached
  a scale of 9.1e134 and coefficients -52674 and 70 and reported
  ``maximum='verified'`` without a warning: the point is within 1e-5 of
  the supremum, so the gradient test passes, and the no-maximum check,
  made in the search's own units, could not see the run-off. The check now
  takes each positive parameter on the log scale and every parameter in
  units of its size, tests coefficients that run off together one at a
  time with the others held, and calls a profile flat to rounding a
  run-off. On 300 data sets of #583's design (27% without a finite
  maximum) ``WeibullAFT`` flags all of them (90% before), ``WeibullPH``
  all (22%), and ``AcceleratedLife(Weibull, PowerExponential)`` 79 of 81,
  whose 10 false "No finite maximum" warnings on fits that had stopped
  short are gone. Fits with a finite maximum are unchanged to the bit.
- **Parametric bootstrap bounds for the parametric regression models
  (#617).** ``cb``, ``param_cb``, ``quantile_cb`` and ``cb_tvc`` take
  ``method="bootstrap"`` with ``n_boot=`` (default 200) and
  ``random_state=``; Wald stays the default. Each resample simulates every
  unit from the fitted model at its covariates and truncation window,
  censored as it was (the conditional bootstrap), and refits; the bound is
  the BCa interval, its acceleration from the resamples' scores. On #583's
  test with 46 failures the 90% bound on R(5 y) at use covered 0.903,
  against 0.880 (Wald), 0.875 (likelihood ratio) and 0.866 (percentile),
  over 1000 repetitions of 1000 refits. With 11 failures no method holds
  (bootstrap 0.71, Wald 0.87, likelihood ratio 0.85): 28% of those data
  sets have no finite estimate, and the bootstrap warns when the model has
  none. Refits that reach no verified maximum are kept and counted (a
  warning above 2%); refits are shared per ``n_boot`` and integer seed and
  are not pickled. Refused for left- or interval-censored data and
  time-varying-covariate fits.
- **Copula standard errors and confidence bounds (#540).** A fitted
  ``CopulaModel`` reported point estimates only. It has
  ``covariance()``, ``standard_errors()`` and ``param_cb(name)`` for the
  copula parameters (``margins=True`` adds the margins'), and ``cb(x,
  on="sf"/"ff")`` on the joint survival and joint CDF. ``how="MLE"``
  uses the inverse Hessian of the joint likelihood; ``how="IFM"`` the
  Godambe sandwich of the two stages (Joe 2005) -- treating the margins
  as known understates the copula's standard error by about a fifth
  (Clayton tau = 0.5, 300 rows: 0.173 against 0.223; the bootstrap gives
  0.231 at 200 rows). ``param_cb`` works on the log or Fisher-z scale of
  the parameter's space; ``cb`` is the delta method on the logit with
  every parameter, the margins' included. Coverage of 95% intervals over
  200 samples of every family under either fit: 91.5-99%. ``to_dict``
  stores the covariance, so a restored model keeps its bounds; a dict
  saved before v0.23 raises "refit". A non-parametric margin raises
  (#623).
- **Reliability growth projection (#607).** ``CrowAMSAA.projection(x,
  modes, fef, i=, c=, bc=)`` projects the MTBF after delayed fixes: the
  AMSAA-Crow projection model (MIL-HDBK-189C 6.2) and, with modes fixed
  during the test (``bc=``), Crow's extended model. Each failure carries
  a mode label; the BD modes and their fix-effectiveness factors are a
  dict, the rest A modes. It returns the demonstrated, projected and
  growth-potential intensities and MTBFs, a per-mode table and h(T), the
  rate of new BD modes at the end of the test (unbiased beta-bar). In a
  simulation of its definition the projection is within 3% of the true
  post-fix intensity (without the unseen modes' term, 58% low).
  Time-terminated tests only.
- **Renewal fits take delayed entry (#615).** Breaking:
  ``GeneralizedRenewal``, ``GeneralizedOneRenewal``, ``ARA`` and ``ARI``
  take ``tl`` (and ``fit_from_df``'s ``tl_col``), each item as new at
  its entry (virtual age 0, as after an overhaul; for ARI no intensity
  reduction, the baseline's clock restarting), its times counted from
  there. They refused ``tl > 0`` and silently ignored a negative ``tl``,
  which now changes the fit. The fitted model's ``data`` hold the times
  since entry.
- **Per-unit prediction for renewal models (#615).**
  ``RenewalModel.unit_states()`` gives each fitted unit's state at the
  end of its history (virtual age now, G1 gap scale, or ARI intensity
  reduction), and ``next_failure_sf(x)`` / ``next_failure_hf(x)`` its
  next failure from there in closed form, for all four families;
  ``age=`` gives units with no failure yet. Fixed on the way: ARA/ARI
  memory for units with fewer failures than ``m``.
- **``surpyval.forecast`` for repairable systems (#615).** It takes the
  Poisson-process models (HPP, CrowAMSAA, Duane, CoxLewis, proportional
  intensity), counting every failure: each unit's expected count is
  Lambda(a+h) - Lambda(a), with Poisson prediction intervals; and the
  renewal models, simulating each unit's future from its own state
  (``items=``, ``random_state=``). ``age`` may be left out for a model
  fitted to data (its items are the units). ``Forecast`` gains
  ``per_unit``, ``units`` and ``simulations``.
- **Every ``qf`` follows one rule (#611).** A probability outside [0, 1]
  gives NaN with one warning. This now applies to the non-parametric
  ``qf`` (it raised a ``ValueError``; ``qf(0)`` is now accepted), the
  distributions' own ``qf`` (``Exponential.qf(-0.5, 0.2)`` was -2.36,
  ``Uniform.qf(1.5, 1, 4)`` was 14.45), the point masses
  (``NeverOccurs.qf(2)`` was inf), the process models (0 and inf in
  silence) and the accelerated degradation and RUL models (which
  raised). A new conformance property checks every model.
- **Breaking: univariate fits refuse a censored time below the support
  (#611),** as the regressions do since #565, with the same error.
  ``Weibull.fit([-1, 2, ...], c=[1, 0, ...])`` used to fit and ignore
  the row. A right-censored 0 and offset fits are unaffected.
- **``fit_best`` takes mixtures as opt-in candidates (#613):**
  ``include=["Weibull", MixtureModel(Weibull, 2)]``, ranked on the same
  criterion. Default candidates are unchanged.
- **Deprecated (#613):** ``surpyval.NUM``, ``TINIEST`` and ``EPS`` warn
  until v0.24. Use the numpy equivalents. ``surpyval.np`` stays.
- **Cox has ``covariance()`` (#613):** the inverse observed information,
  R's ``vcov(coxph)``. It is saved in ``to_dict``.
- **``standard_errors()`` is an array on every model with
  ``covariance()`` (#613).** It is new on ``Parametric`` and
  ``RoystonParmarModel``.

  - **Breaking:** the frailty models' ``standard_errors()`` returns an
    array, not a dict; theta's is ``[-1]``.
  - ``se`` on the Cox, Lin-Ying, proportional-odds and Fine-Gray models
    is deprecated until v0.24.
- **Binomial takes a number of trials per row (#608):** ``n_trials=[20,
  50, 80]``. ``p`` comes from all the trials, and the exact bounds hold
  for unequal batch sizes. With unequal trials there is no single ``n``
  (NaN); the functions of the event count say so and point to
  ``with_params``. Empty data is refused.
- **Breaking: the limited-failure proportion is ``lfp_p`` (#608)** in
  the attribute, ``fixed``, ``param_cb``, ``from_params``, ``extras``
  and the repr. ``p`` still works with a ``DeprecationWarning`` until
  v0.24, except on distributions that have a parameter called ``p``
  (Bernoulli, Binomial, FixedEventProbability, Geometric,
  NegativeBinomial). There ``model.p`` is now the fitted parameter; it
  used to read 1. ``extras`` uses the key ``"lfp_p"``. Saved dicts keep
  the key ``"p"``, and old dicts and pickles load.
- **Regression coefficients are named by their covariate (#614).
  Breaking.** Coefficients were ``beta_0``, ``beta_1``, ..., beside the
  Weibull shape ``beta``. They are now named by the covariate's column
  (a formula, ``fit_from_df``, or a DataFrame ``Z``, which every
  regression ``fit`` now accepts as ``fit_from_df`` does), otherwise
  ``coef_0``, ``coef_1``, ..., in every regression family. A clashing
  name gets ``.1``, ``.2`` (R's ``make.unique``). ``parameter_names``,
  ``summary()``, reprs and ``to_dict()`` change. ``beta_j`` in
  ``fixed=`` and ``param_cb`` still works with a ``DeprecationWarning``
  until v0.24, and dictionaries saved with ``beta_j`` load with the new
  names.
- **Fitters print what they are (#614).** ``Weibull`` printed as
  ``<...Weibull_ object at 0x...>``; it now prints ``Weibull: parametric
  fitter``, and ``WeibullAFT`` prints ``WeibullAFT: accelerated failure
  time fitter (Weibull baseline)``. Every registered fitter is checked
  by the conformance suite.
- **Large-scale covariates reach the maximum (#612).** A coefficient was
  searched in units of 1 whatever its covariate's range, so with
  covariates in units of 1e4, 14 regression fits stopped up to 0.023
  short in log-likelihood and warned "unverified". A covariate whose
  range exceeds 100 is now searched and judged in units of 1/range (as
  below 1 since #577), including ``GeneralLogLinear`` accelerated life
  fits. All 14 now verify, within 1e-8 of the unscaled maximum. Fits on
  covariates with ranges between 1 and 100 are unchanged to the last
  digit.
- **Offset fits start from one distribution at its own offset (#622).**
  An offset maximum-likelihood start is now the starting offset with the
  family's own seeds for the data shifted by it. The Weibull kept its
  probability plot's shape and scale, fitted with the plot's own offset,
  after replacing that offset. On a left-skewed sample the start's
  negative log-likelihood was 1.3e7, and the fit returned a point at
  4.8e6 labelled "No finite maximum"; it now ends at the Gumbel limit
  (30.17). The Exponential seeded its rate from the unshifted data, and
  interval data seeded every family against the wrong offset. Ordinary
  fits reach the same maximum (log-likelihood within 1e-10 to 2e-7
  relative, from a different search path).
- **An offset run onto the first failure ends quickly (#622).** With a
  shape below 1 the likelihood is unbounded as gamma approaches the
  first failure. The ladder used to follow it there in 5,000-15,000
  likelihood evaluations (2-8 s) and end "unverified", "MLE Failed" or
  "No finite maximum" depending on where the last rung stopped. The
  search now ends at the first rung that reaches it, or that stops on
  its way there, with one "No finite maximum ... how='MPS'" warning, in
  13-750 evaluations. Over 276 simulated offset fits the right status
  rose from 262 to 272, and total evaluations fell from 493,516 to
  33,698. Results now agree between CPU paths: the #584 offset Weibull
  was "no finite maximum" with AVX-512 disabled.
- **Breaking: offset fits refuse an observed value at ±inf** with
  ``OutsideSupportError``, as fits without an offset do. Previously the
  Exponential returned a rate, the Weibull warned "MLE Failed" and the
  Gamma leaked RuntimeWarnings.
- **Fixed:** an offset interval-censored fit with windows in a
  distribution's lower tail raised ``TypeError`` (#622).
- **Zero-inflated fits check the support (#610).** A ``zi=True`` fit
  skipped the support check, so a negative (or infinite) time returned the
  optimiser's start with "MLE Failed"; it raises ``OutsideSupportError``. A
  failure or left-censored time at 0 is the point mass and passes.
- **Survival trees and forests check their times (#618).** An infinite
  time reached the leaf fits (a Weibull tree gave 216 numpy warnings, an
  Exponential tree a scipy error). Breaking: the parametric kinds refuse a
  time outside (0, inf) whatever its censoring, as the parametric
  regressions do; the non-parametric kind refuses an infinite failure, as
  Kaplan-Meier and Cox do.
- **Offset fits that run to their family's limit (#616).** A runaway found
  only by a later optimiser rung, or seen only at the end of the ladder,
  ends "No finite maximum" rather than "unverified" (an offset Weibull at
  gamma = -177 and LogLogistic at -6.9e4 on a smallest-extreme-value
  sample).
- **Offset maximum product of spacings (#616).** An untruncated offset MPS
  fit was truncated at time 0 in the data's units once gamma went below 0,
  so on data containing -1 the objective was NaN everywhere (the LogNormal
  took 32 s to warn "MPS FAILED", the Gamma leaked autograd warnings, the
  LogLogistic raised ``TypeError``). That is fixed, a degenerate start is
  re-seeded, and an MPS fit running to the family's limit stops with one
  "No finite maximum" warning naming the limit. Changes results of offset
  MPS fits whose gamma went below 0: they are now unchanged by a shift of
  the data.
- **Degradation offset-exponential path (#621).** The start bending
  against nearly straight measurements is stopped at the straight-line
  limit, with identical results: ``path="best"`` on 200 straight units
  takes 2.8 s rather than 5.1 s.
- **AcceleratedLife start on a continuous stress (#621).** A stress level
  that could not be fitted alone took the mean time as its life parameter
  rather than through the parameter's transform:
  ``AcceleratedLife(LogNormal, GeneralLogLinear)`` with a continuous column
  raised "The log-likelihood is not finite at the initial parameter
  values"; it fits, and matches ``LogNormalAFT``.
- **Copula censored rows keep their digits (#619).** The likelihood of a
  row censored in both series was ``1 - u - v + C(u, v)``, and of an
  observed/right-censored pair ``1 - dC/du``; under strong negative
  dependence these were noise or the 1e-300 floor (one ulp in C moved a
  10,000-row log-likelihood by 654). Each family evaluates its quadrants
  and h-function complements directly (``C(1-u, 1-v)`` for the radially
  symmetric families, closed forms for the others, rotations by
  reflection) in the likelihood, left-truncation windows and
  ``CopulaModel.sf``, matching mpmath to about 1e-13; the Gaussian
  copula's small CDF values are integrated rather than 0. Fits on ordinary
  data are unchanged.
- **Turnbull fits untruncated data by the EM-ICM (#620).** Breaking:
  ``turnbull_algorithm`` (new) defaults to ``"auto"``, Wellner and Zhan's
  EM-ICM (as R's Icens and icenReg) for data without truncation and
  Turnbull's EM with it; ``"EM"`` and ``"EMICM"`` choose. The EM-ICM stops
  on the Karush-Kuhn-Tucker conditions of the maximum: on 1,000 random
  intervals it takes 45 iterations (9 ms) where the EM stops at
  ``max_iter`` 7.5e-3 from the maximum with a warning (it needs 54,000),
  and ``bootstrap_cb(n_boot=50)`` 0.5 s rather than 13.4 s. With the
  ``"Kaplan-Meier"`` estimator the answer is the EM's converged one. With
  the Fleming-Harrington (the default) or Nelson-Aalen estimator, the
  estimator is applied to the non-parametric MLE's expected counts, as it
  already was under truncation, rather than iterated inside the EM, so
  untruncated interval-censored curves move: on the docstring's five
  intervals ``R`` at 5 is 0.277 rather than 0.295. ``turnbull_algorithm=
  "EM"`` gives the old result. A model keeps the algorithm it ran, and
  ``to_dict`` stores it when it is not the EM (schema 2).
- **Analytic shape derivatives of the incomplete beta (#621).** The
  censored Beta fit at 3,000 rows takes 1.2 s rather than 2.5 s; the
  derivatives are accurate to about 1e-15 against mpmath, where the finite
  differences were up to 2e-4 off. NegativeBinomial and Binomial
  covariances move by up to 3e-6 relative, towards the exact Hessian.
- **Likelihood-ratio bounds along a covariate path and for the frailty
  and proportional-odds models (#617).** ``cb_tvc`` takes
  ``method="lr"``, and ``FrailtyModel.param_cb`` and
  ``ProportionalOddsModel.param_cb`` take ``method="lr"`` (the
  profile-likelihood interval; the proportional-odds model profiles out
  its baseline, Murphy and van der Vaart 2000). With no frailty in the
  data the Wald interval on ``theta`` is [0, inf] and the profile one
  finite. Wald stays the default. A regression likelihood-ratio interval
  reaching the edge of its space returns the edge itself.
- **Likelihood-ratio caches are not pickled (#617).** A model drops its
  searches' caches when pickled and rebuilds them on demand: a Weibull
  model after a 50-time band pickled at 302 kB, now 7 kB.
- **Wald bounds of a tiny life-model constant (#617).** ``cb`` and
  ``quantile_cb`` returned [nan, nan] without a warning where an
  accelerated-life constant (6e-9) was smaller than the delta method's
  difference step; the difference is retaken with a relative step where
  the first is not finite. No other result changes.
- **Log-normal frailty likelihood at an extreme cumulative hazard** raised
  ``OverflowError`` past 1e154; it is finite.
- **Log-normal frailty likelihood at a huge theta.** The mode of each
  group's integrand, ``theta D - omega``, cancelled once ``theta D`` was
  large, so at ``theta = e^50`` the log-integral of a group with 10 events
  was 30 too low and the likelihood rose spuriously far out. The
  likelihood-ratio searches followed it: with no frailty in the data the
  95% and 99% ``param_cb("theta", method="lr")`` upper bounds were 7e275
  and 6.7e275 (the 99% inside the 95%); they are 0.374 and 0.837. The mode
  is taken in its logarithmic form above ``theta D = 1e4``; ordinary fits
  are unchanged.
- **Cheaper likelihood-ratio bounds on a levelled-off edge valley (#609).**
  Where a parameter's profile has reached its limit, only the deepest slice
  is searched: NegativeBinomial's registry bounds take 80 s rather than
  98 s, equal to 1e-14.
- **Fits evaluate the likelihood once per optimiser step (#593).** The
  univariate MLE ladder, MPS, the verification polish and the parametric
  regression searches gave scipy the likelihood and its gradient
  separately, so each step evaluated the likelihood twice; they take both
  from one autograd pass. Results are identical to the bit; a Weibull fit
  makes 1 plain likelihood evaluation rather than 10, a WeibullPH fit 3
  rather than 17.
- **Accelerated-life fits are fast on continuous stresses (#592).** The
  distribution was evaluated over every row once per distinct stress, so
  the cost grew with rows x stress levels: a GeneralLogLinear fit to 120
  distinct stresses took 8.8 s and a Gamma-Eyring fit 16 s; both take
  0.2 s. Likelihoods are unchanged; covariances and bounds agree to 1e-12.
  New ``LifeModel.phi_takes_rows`` (True for the built-in models): set it
  on a custom life model whose ``phi`` takes a matrix of stress rows.
- **Large fits spend less time checking their data (#552).** Count arrays
  were checked for missing values one Python object at a time: a Weibull
  fit to a million rows takes 1.9 s rather than 2.9 s.
- **``import surpyval`` no longer loads scipy.stats or scipy.integrate
  (#470).** They came in through ``autograd.scipy``'s package import and
  module-level imports in the core; the core imports scipy.stats,
  scipy.integrate, scipy.interpolate and numdifftools where it uses them.
  Import takes 0.57 s rather than 0.79 s.
- **Faster random survival forests (#549).** A non-parametric split
  scanned a rows x event-times matrix per feature; large nodes are scored
  by sorted cumulative sums in O(N log N) (2 trees at n = 10^4: 43 s to
  2.6 s). The Weibull split sums each candidate child over its own rows and
  finds the node and every feature's children in one pass (1.5x); leaves
  are fitted together when the tree is grown (first prediction 5.6x
  faster); ``to_dict`` does not re-walk finished leaves (the same bytes,
  1.6x faster). Splits are the same; leaf parameters move in the last
  digits.
- **StudentT copula CDF without a t quantile per node (#550).** The
  bivariate t CDF integrates over the closed-form CDF of the t with 2
  degrees of freedom (the Cauchy below nu = 2), so a censored fit no longer
  evaluates a t quantile at each of 210 nodes: at n = 10^4 IFM takes 2.9 s
  rather than 4.7 s and MLE 17 s rather than 31 s, with the same estimates.
  Values agree to 1e-16 at the Genz and mpmath references and are closer
  to a converged integral at strong dependence; a split point rounding to
  1 no longer makes the CDF, and the likelihood, NaN.
- **CoxFrailty standard errors in linear time (#551).** The covariance
  formed the full information with one column per group and inverted it
  (5.3 s at 2,000 groups, 55 s at 10,000, n = 10^4); it is the inverse of
  the Schur complement, from an operator form of the Cox information (the
  new ``CoxInformation``) solved by conjugate gradients: 0.06-0.09 s, the
  same values to 1e-14.
- **Faster likelihood-ratio bounds on interval-censored and truncated data
  (#602).** The searches' lean likelihood evaluates interval and truncated
  windows without the distribution functions' input guards, which those
  data cannot trip: one evaluation on six interval-censored units takes
  123 us rather than 243 us, a Weibull probability plot's band 4.7 s rather
  than 8.2 s, identical to the bit.
- **Fewer repeated checks in likelihood-ratio searches (#609, #519).** An
  answer found again within the extremality check's tolerance of one
  already checked is not checked again, and per-point overhead is plain
  numpy: all registry likelihood-ratio bounds take 92 s rather than 121 s
  for ExpoWeibull and 79 s rather than 102 s for NegativeBinomial, and a
  Weibull band at 20 times on 1,000 units 0.70 s rather than 0.93 s, with
  identical results.
- **Likelihood-ratio bounds for the parametric regression models (#583).**
  ``cb``, ``param_cb`` and the new ``quantile_cb`` of every parametric
  regression model take ``method="lr"``: the extreme of the function or
  parameter over the likelihood region of all the parameters, by the
  univariate models' search (the region's boundary is traced once per
  model and level; a bound takes about a second, 0.7 s on the issue's
  72-unit accelerated life test). Wald stays the default: in 1000
  repetitions of the issue's test, extrapolated 40 degrees C below the
  coolest cell, the 90% bounds on the five-year reliability covered 0.897
  (Wald) and 0.893 (likelihood ratio), and with 11 failures on average
  0.877 and 0.866 (the issue's 0.86 was 300 repetitions).
- **``quantile_cb(p, Z)`` for the parametric regression models (#583),**
  bounds on the B-life ``qf(p, Z)``: Wald by the delta method on log t_p,
  ``method="lr"`` the extreme of t_p over the likelihood region.
- **Royston-Parmar ``qf`` and ``random`` are vectorised (#595).** ``qf``
  ran one root search per probability (``random(2000)`` took 0.8-3 s on a
  200-row fit); it solves every probability at once on the link scale,
  exactly beyond the boundary knots, in 1-5 ms. ``qf(0)`` is 0 and
  ``qf(1)`` inf (they were 1.1e-9 and 1.7e10, the search bracket's ends),
  and small probabilities keep their precision (3e-5 relative at
  p = 1e-12, now 2e-15); other quantiles agree with the old ones to about
  1e-12.
- **``CustomDistribution.qf`` inverts every probability at once (#596)**
  when the cumulative hazard broadcasts (gives each point's own value on an
  array, checked before and after the solve): 2000 quantiles take 5-10 ms
  rather than 0.5-1.2 s, agreeing to 1e-14. A hazard that indexes or
  reduces its argument is still inverted one probability at a time. A tiny
  probability on a hazard that rounds to 0 there no longer raises scipy's
  ``RuntimeError``.
- **The ExpoWeibull likelihood is about 3x faster with its gradient
  (#598).** Its density, CDF and survival logs have derivatives written out
  rather than traced by autograd (about 1,300 operations per evaluation).
  Values are unchanged to the bit and gradients and Hessians agree to
  1e-13: on 60 interval-censored rows value and gradient take 2.2-2.6 ms
  rather than 5-9 ms, the #584 fit 3.7 s rather than 7.2 s, and
  likelihood-ratio bounds about 20% less.
- **An offset fit running to its family's limit warns "No finite maximum"
  (#599).** On data skewed further left than any member of the family the
  offset runs to -inf while the shape compensates, approaching the limit
  (Normal for LogNormal and Gamma, Gumbel for Weibull, Logistic for
  LogLogistic) only as ``1/|gamma|``, so it never looked flat: such fits ran
  the whole optimiser ladder and ended "unverified" (6 s for LogNormal,
  17 s for Gamma, with one failure at -1 below the rest). They stop after
  the first rung in 0.3 s with one warning recommending the limit
  distribution, and ``maximum = "no finite maximum"``.
- **Regression ``fit_from_df`` reads interval columns (#571).** The
  parametric regression families' ``fit_from_df`` takes ``xl_col`` /
  ``xr_col`` in place of ``x_col`` (``fit`` already took interval data as
  a two-column ``x``) and gives the same model as ``fit`` (to 1e-10). Its
  columns are read as every other ``fit_from_df`` reads them: a missing
  column is a ``ValueError`` naming the columns there are (it was a
  ``KeyError``), and a duration or date column is refused. A new
  conformance check makes every DataFrame entry point fill each data
  argument of its ``fit``; 28 fitters failed it.
- **``ParametricCompetingRisks`` takes left truncation (#571).**
  ``fit(..., tl=)`` and ``fit_from_df(..., tl_col=)`` fit delayed-entry
  data exactly (the truncated likelihood factorises by cause). On data seen
  only from a records start date, ignoring entry gave shapes 2.59 and 1.56
  against true 2.5 and 1.3; with ``tl``, 2.45 and 1.32. Right truncation
  and interval censoring do not factorise and are still not taken.
- **Regression models have ``qf`` (#571).** ``model.qf(p, Z, grid=False)``
  is the time by which a proportion ``p`` of units with covariates ``Z``
  have failed (the B10 life is ``qf(0.1, Z)``), inverted from each family's
  own cumulative hazard to a relative 1e-12, with NaN and a warning outside
  [0, 1].
- **Regression models have ``cs`` (#581).** ``cs(x, given, Z)``, the
  conditional survival of a unit with covariates ``Z`` that has survived
  to ``given``, on the parametric, Cox, proportional-odds, additive-hazards,
  Buckley-James and frailty models (with each one's ``grid``, ``stratum``
  or ``group``), computed from the cumulative hazard, so it stays exact
  where ``sf(given)`` underflows.
- **New: ``surpyval.forecast`` (#581).** The expected failures of units in
  service at their current ages over one or more horizons, from any
  univariate or regression model (``Z`` per unit), with cohort counts
  ``n`` and a warranty ``limit``: expected counts and variance, exact
  Poisson-binomial prediction intervals (a refined normal approximation
  for very large fleets), per-period counts, and each unit's probability
  of failing (``Forecast.probability``, ``unit_expected``). The intervals
  treat the model as known.
- **Cox models report their maximised partial likelihood (#604).** A Cox
  model's ``neg_ll`` was the fit's function of the coefficients, and it had
  no AIC or BIC. ``neg_ll()`` is now the fitted value and
  ``log_likelihood`` its negative; ``aic()``, ``aic_c()`` and ``bic()``
  follow R's ``logLik.coxph`` (k the non-aliased coefficients, BIC's n the
  events): on Rossi, AIC 1327.714 and BIC 1335.923, as R. The function is
  ``neg_ll_of(beta)``; ``neg_ll(beta)`` is deprecated until v0.24. The same
  values are on FineGray (the weighted partial likelihood, ``crr``'s
  ``loglik``, n the events of the cause), CompetingRisksProportionalHazards
  with ``model="Cox"`` (summed over causes, as R's multi-state ``coxph``),
  ProportionalOdds (the profile likelihood) and CoxFrailty (the integrated
  likelihood, k including theta); a Fine-Gray
  CompetingRisksProportionalHazards raises for them. CoxFrailty's
  ``loglik`` and ``loglik_no_frailty`` are ``log_likelihood`` and
  ``log_likelihood_no_frailty`` (the old names deprecated until v0.24).
  These criteria compare models of one kind on the same data, not a Cox
  model with a parametric one.
- **Fine-Gray with large-scale covariates (#606).** Covariates of order 1e4
  raised "SVD did not converge": ``exp(beta'Z)`` overflowed at BFGS's
  first step. The fit is Newton-Raphson with step-halving, as
  ``cmprsk::crr`` and CoxPH (BFGS where Newton gives up), with the linear
  predictor shifted inside the risk-set sums. Ordinary fits move by up to
  1e-6 standard errors, the distance BFGS stopped short of the maximum, and
  now match ``crr``'s coefficients to 1e-8 rather than 2.5e-7.
- **One spelling for model comparison (#605).** Every model's covariance
  is ``covariance()``: ``Parametric.cov_matrix``, the ``covariance``
  attribute of Royston-Parmar and the frailty models, and ``cov`` on
  Fine-Gray, proportional odds and additive hazards work with a
  DeprecationWarning until v0.24. ``aic_c()`` is added on the recurrent,
  copula, competing-risks and Royston-Parmar models, and
  ``log_likelihood`` wherever there is an AIC. The mixture's EM steps
  (``EM``, ``Q``, ``expectation``, ``maximisation``, ``likelihood``,
  ``initialise_params``) are internal; their public names warn until
  v0.24. Breaking: model dicts store ``"_neg_ll"`` and ``"covariance"``;
  old keys still load, but a 0.23 dict loses its likelihood or covariance
  in an older SurPyval, which cannot read a 0.23 FineGray dict.
- **Fitted models pickle (#573).** Every fitted model -- univariate,
  PH, Cox (stratified too), competing-risks and every recurrence model --
  pickles, so models can go to ``multiprocessing``, ``joblib``,
  ``concurrent.futures``, Dask or Ray, or be cached with ``pickle`` /
  ``joblib.dump``; 92 of the 137 registry models failed with "Can't pickle
  local object". The recurrence fitters unpickle as themselves. New
  conformance properties ``pickle`` and ``pickle_paths``. A model built on
  a user's own lambda (a ``phi``, life model or ``CustomDistribution``)
  pickles only if that function does.
- **``CompetingRisksProportionalHazards.phi`` and ``phi_e`` are methods
  (#573);** they were lambda attributes, and calls are unchanged.
- **``fit_best`` raises input errors and takes ``tl``, ``tr``, ``xl``,
  ``xr`` (#570).** Malformed data raise ``fit``'s ``ValueError`` (it
  returned ``None`` with a warning quoting the data once per candidate); a
  skipped candidate is named with a one-line reason. Breaking: an error
  every candidate gives alike, or data outside every candidate's support,
  raises rather than returning ``None``.
- **Parametric regressions refuse a censored time outside the baseline's
  support (#565).** Breaking: a Weibull, Gamma, Exponential or LogNormal
  AFT/PH/PO/AH, accelerated-life or frailty fit given a time censored at
  -1 was fitted, with nan derivatives; it now raises the univariate fit's
  support error, from every entry point. A unit censored at 0 is still
  accepted, as in R ``survreg`` and lifelines.
- **``logrank(..., tl=)`` (#576):** delayed entry, each unit at risk from
  its entry; it matches R's Cox score test (3.365711 on AML with entries).
- **``qf`` warns of a probability outside [0, 1] (#576),** such as
  ``qf(10)`` meant as the B10 life, and still returns ``NaN``;
  Royston-Parmar's ``qf`` returns ``NaN`` with the warning rather than a
  scipy error.
- **Missing censoring flags and counts are named (#576):** ``NaN``,
  ``None`` or pandas ``NA`` in ``c`` or ``n`` give "Variable 'c' cannot
  contain NaN values", not "Censoring value must only be one of ...".
- **The straddling-interval truncation error explains inspection data
  (#576):** ``tl`` is the last good inspection, so ``tl <= xl``.
- **ExpoWeibull likelihood-ratio bounds reach the region's extreme,
  whatever the CPU (#601).** On long, flat valleys of the likelihood region
  (beta -> inf with alpha at the largest observation; alpha -> 0) the
  searches ran out of iterations and returned points inside the region:
  the 95% ``cb(8)`` upper bound was 0.7758 with multi-threaded BLAS (true
  0.79698), the 99% ``qf(0.95)`` upper bound ranged over 80.6-86.2 with the
  thread count and CPU (true 87.022), and some quantile bounds were too
  narrow everywhere (the 95% ``qf(0.2)`` band was [2.72, 7.63], true
  [2.2208, 7.9437]; the one-sided ``qf(0.05)`` lower bound 0.849, true
  0.5517). Searches that stop short are continued, with exact deviance
  gradients where they stall; each parameter's profile is followed down
  valleys to the edge of its space and the extreme sought over slices
  there. Bounds agree with brute-force extremes to 1e-6 on one or more
  threads and without AVX-512. A bound whose search is still moving out
  when it stops is returned with a warning, as is the fallback when the
  last-resort search fails (it returned nan). ExpoWeibull and
  NegativeBinomial likelihood-ratio bounds take 2.4x and 2.8x as long
  (#609).
- **``ExpoWeibull.qf`` at extreme parameters (#601):**
  ``qf(0.95, 2.2e-308, 0.0076, 3e95)`` was ``inf``; it is 19.4, computed in
  logs where ``alpha * exp(log t / beta)`` overflows or underflows.
- **Recurrent trend tests take delayed entry (#575).** ``trend_test()``
  refused any data with ``tl`` ("trend tests assume observation from time
  0"). ``laplace`` and ``mil_hdbk_189c`` take each system's start as
  ``tl`` and test each system on its own window: Laplace in the pooled
  form (Ascher and Feingold; Kvaloy and Lindqvist), MIL-HDBK-189C with time
  measured from each start, still exactly chi-squared on 2N degrees of
  freedom. A fitted model's ``trend_test()`` uses the ``tl`` it was fitted
  with, and gapped data is tested window by window. Data observed from 0
  gives the same results to the bit. Size with delayed entry (30,000
  replicates): Laplace 0.0496, MIL-HDBK 0.0480.
- **Bounds on the intensity and the demonstrated MTBF (#578).** New
  ``iif_cb`` on the parametric and proportional-intensity recurrence models
  (the delta method on the log intensity, as ``cif_cb``), and ``mtbf(x)``
  (= 1/iif) with ``mtbf_cb`` on the parametric ones (``bound="lower"`` is a
  lower bound on the MTBF). ``method="crow"`` gives Crow's (1982) exact
  bounds on the demonstrated MTBF at the end of a time-terminated test, or
  of a failure-terminated test of one system, as MIL-HDBK-189C tabulates
  them; the failure-terminated bounds hold their level exactly, the
  time-terminated ones at least their level.
- **Bernoulli, FixedEventProbability and Binomial: bounds on p (#580).**
  ``param_cb('p')`` raised a misleading "Hessian was singular" or "need the
  original data" error, even with zero failures. It now bounds p from the
  counts of events and trials: exact Clopper-Pearson by default (3 in 1200
  gives a 90% interval of 0.00068-0.00645), ``method="wald"`` (logit) and
  ``"lr"`` as options; with no failures the upper bound is
  1 - alpha^(1/n). Binomial ``param_cb('n')`` is the known ``n_trials``;
  ``cb``, ``quantile_cb`` and ``mean_cb`` on these models raise one message
  pointing to ``param_cb``. Breaking: ``Parametric.param_cb``'s ``method``
  default is ``None`` ("wald" everywhere else, "exact" for these three).
  Models saved before 0.23 have no counts and must be refitted for these
  bounds.
- **``success_run`` takes ``alpha_ci`` (#580),** the one bound without it;
  ``confidence=`` and ``alpha=`` are deprecated until v0.24.
- **A covariate's units no longer change a regression fit (#577).** Each
  coefficient is searched and verified in its covariate's units
  (``1/range(Z_j)``, at least 1). Before, a coefficient's gradient at its
  start of 0 was proportional to the covariate's spread, so a small-scale
  covariate met both tolerances at once: ``WeibullPH.fit_tvc`` with an
  Arrhenius ``1/T`` stopped after no iterations at coefficient 0 and
  reported a verified maximum 0.41 below the one reached with ``1000/T``.
  With covariates scaled by 1e-6, 50 fits stopped 0.06-10 short while
  reporting "verified": every PH, AFT and PO family (``fit`` and
  ``fit_tvc``), frailty, Fine-Gray and ``ProportionalIntensityHPP``
  (additive hazards at 1e-9). All now match the fit in the original units.
  ``ProportionalIntensityHPP`` polishes an unverified answer, as the NHPP
  fit did, and numerical derivatives step in each component's unit. Fits
  whose covariates have a range of 1 or more are unchanged to the bit.
- **A limited-failure ``p`` (or zero-inflation ``f0``) on its bound is
  judged there (#579).** A Weibull with ``lfp=True`` on monthly
  interval-censored return counts ran ``p`` to 1, where its search
  coordinate no longer moves the likelihood, and reported a verified
  maximum 3.8 below the one at ``p = 0.059`` (wrong in 19 of 20 simulated
  warranty data sets, by 0.13-5.8). Such a parameter is now held out of the
  gradient and Hessian test and counts as a maximum only if the likelihood
  does not rise off the bound; where it does, the fit restarts from
  mid-range and keeps the better answer. Interval-censored data also get
  the failures-alone LFP start. A genuine maximum at ``p = 1`` is reported
  as verified (it sometimes warned "unverified").
- **Process-model lives start where the units start (#574).** Breaking:
  ``GammaProcess`` and ``WienerProcess`` fitted the increments only but
  measured the life from degradation 0, so readings that start at a
  baseline gave an optimistic life (16-17% too long for a vibration signal
  starting at 1.0 mm/s with an alarm at 7). The fitted model now has
  ``y0``, the level at time zero, estimated from the readings or passed to
  ``fit``; the life is the first passage over ``threshold - y0``, and every
  life function takes ``y0=`` for a unit that starts elsewhere
  (``predict_rul`` was already right). Fits whose readings start at 0 are
  unchanged; ``y0=0.0`` gives the old result. ``y0`` is saved; older
  dictionaries load with 0.
- **Degradation fits say what they reached (#564).** ``WienerProcess``,
  ``GammaProcess`` and ``DestructiveDegradation`` record and save
  ``maximum`` as every other maximum-likelihood fit does: each searched fit
  is checked for a zero gradient and a positive-definite Hessian and warns
  when it is not a verified maximum; the noise-free fits that already
  warned record "no finite maximum".
- **Faster step-stress REML (#588).** The FOCE iteration finds every
  unit's conditional mode at once rather than one at a time in Python: a
  bootstrap ``cb`` of a step-stress REML model takes 6.3 s rather than
  11.2 s. Results agree to 5e-12 on the issue's case, and nonlinear REML
  fits within their solver's tolerance.
- **Mixture EM reaches the maximum on staggered interval counts (#582).**
  On Nevada-chart warranty counts the fit ended 101 log-likelihood units
  short (26% defectives instead of 3%): a ``nan`` gradient at interval
  rows starting at 0 (Weibull ``beta < 1``) stopped the polish after one
  evaluation. That CDF is now exactly 0 there, and EM also runs from a
  second start (the failures cut by count, the survivors to the last
  component), keeping the better. The example verifies at the maximum in
  1.2 s instead of 8.6 s; ordinary fits are unchanged.
- **Faster mixture EM, and SQUAREM as an option (#589).** Each M-step takes
  ``Q`` and its gradient from one pass (14% faster per iteration, identical
  results). ``MixtureModel.fit(..., em="squarem")`` accelerates the
  iterations after an unverified first polish: 81 iterations in 2.8 s
  instead of 1,020 in 7.7 s, ending at the maximum rather than 1.3e-4
  short.
- **Changed (breaking): model-comparison values are spelt alike (#572).**
  ``MixtureModel.log_likelihood`` is the fitted log-likelihood (calling it
  with ``params`` works until v0.24 with a ``DeprecationWarning``); the
  mixture gains ``neg_ll()``, ``aic()``, ``aic_c()`` and ``bic()``, kept by
  ``to_dict``; ``MixtureModel.loglike``, which was the *negative*
  log-likelihood, is deprecated until v0.24. Recurrent models' ``aic`` and
  ``bic`` are methods, as everywhere else, with ``neg_ll()`` added (the old
  ``model.aic`` value works until v0.24 with a ``DeprecationWarning``). A
  conformance property checks the spelling on every model (Cox is #604).
- **A univariate fit whose likelihood has no finite maximum stops and says
  so (#584).** A search running off used to run every rung of the
  optimiser ladder and end "unverified": 23 s for an ExpoWeibull on 60
  interval-censored rows, returning ``mu = 153`` at a log-likelihood of
  -74.38 though BFGS had reached -73.45. Maximum likelihood now applies the
  regression fits' Newton test (shared, in ``fitters.runaway``) after the
  first rung that stops short, during long BFGS searches, and on verified
  answers; where a parameter's profile is flat and heading for an infinite
  end of its range, the search stops, warns "No finite maximum" naming the
  parameter, and sets ``maximum = "no finite maximum"``. ExpoWeibull names
  its limits (Fréchet as ``mu`` grows; a power law ending at the largest
  observation as ``beta`` grows). That fit takes 5.8 s; an ExpoWeibull on
  50 Weibull draws 0.53 s instead of 3.65 s. **Behaviour change:** fits
  that ended "unverified" on such data now report "no finite maximum", and
  ``fit_best`` sets them aside with that reason. Ordinary fits are
  unchanged.
- **A Beta4 whose end reaches the data stops there (#584).** Its likelihood
  is unbounded where an end meets the extreme observation with a shape
  below 1, and the search used to grind against that wall through every
  rung: about 4 s for the conformance fixture. It now stops when an end
  reaches its extreme observation, warns "No finite maximum" once and sets
  ``maximum = "no finite maximum"``: 0.11 s. The conformance checks of a
  fit's derivatives and of its Wald bounds containing the estimate skip a
  fit that reports no finite maximum, whose inference it has already
  called meaningless.
- **Truncated and interval windows in the far lower tail keep their digits
  (#594).** A window probability below the smallest normal float (about
  2e-308) is taken in log space, as #412 did for the upper tail: a Normal
  far above its truncation windows had a log-likelihood of -9.67 that was
  rounding (the supremum is -10.03), or a ``nan`` truncation term. With
  #584, Normal fits on such data take 0.4 s instead of 11-12 s.
- **ExpoWeibull moments without quad (#586).** ``moment`` and ``mean``
  integrated each moment with scipy ``quad`` over a Python integrand: 96% of
  an offset method-of-moments fit, off by up to 1.4e-9 at ``mu = 0.01``,
  and an ``OverflowError`` near ``m / beta = 80``. They now use a fixed
  tanh-sinh rule over the probability, vectorised over parameters, within
  2.2e-14 of 25-digit references: an offset MOM fit 6.7 s → 1.2 s, ``mean``
  about 10x faster.
- **The simultaneous band's critical value is cached (#590).** ``band()``
  recomputed it on every call; it is cached on what it depends on, and its
  root search no longer re-evaluates its bracket's ends. Values are
  unchanged to the bit: a fresh band 150 → 68 ms, a repeated one 0.07 ms.
- **Discrete quantile_cb evaluates the band in blocks (#591).** The Wald
  bound on a discrete quantile called ``cb`` once per candidate count, each
  with its own autograd gradient; the same search now evaluates blocks of
  counts with one gradient pass per block (a custom distribution keeps the
  gradient a point at a time). Bounds are unchanged: BetaGeometric
  ``quantile_cb`` about 6x faster.
- **Likelihood-ratio bands reuse what each time's search learns (#587).**
  A band's times are searched in order, each starting from where its
  neighbour's bound was found; the likelihood region is found once per
  level and shared by every band, quantile and mean bound at that level;
  the searches run in coordinates scaled by the Wald standard errors; and
  the check that an answer is the extreme starts beside it rather than
  crawling to it. A two-parameter band takes 135-222 likelihood
  evaluations per time instead of 283-854: the probability plot's LR band
  19.1 s → 7.7 s, a Weibull band at 50 times on 1,000 units 2.8 s → 1.7 s.
  Bounds agree with the previous ones to 1e-11 on the registry.
- **Faster accelerated-degradation sampling (#585).**
  ``DegradationModel.qf`` and ``random`` on an accelerated model searched
  for each probability on its own, calling the regression model's ``sf``
  about 35 times per draw: ``random(5000)`` took 13 s. All probabilities
  are now bisected together, with the same brackets and tolerance: 8 ms,
  and the draws are identical.
- **Faster process-model quantiles (#585).** The Wiener and gamma process
  models' ``qf``, the gamma process's ``random`` and ``predict_rul`` ran
  one ``brentq`` per probability (Wiener ``qf`` of 5,000 probabilities:
  3.2 s); they are now solved together to ``brentq``'s tolerance in 11 ms,
  agreeing to 1e-11. A gamma process's ``qf`` of a probability below its
  bracket (e.g. ``1e-300``) raised ``ValueError``; it now returns the
  quantile.
- **Faster PH sampling at tiny hazard multipliers (#585).** Draws whose
  quantile rounds to ``inf`` were each solved by their own ``brentq``
  (``GammaPH.random(2000)`` at ``z = -60``: 20.7 s); they are solved
  together in 0.03 s. A draw beyond the largest float is now ``inf``, as
  ``qf`` gives, instead of raising ``ValueError``.

v0.22 (3 October 2026)
----------------------

**Upgrading from 0.21.** A breaking release. Most code needs no change; the
items most likely to need one are listed here, and every entry below says
what changed.

- Arguments 0.21 deprecated now raise ``TypeError`` (see *Removed*). Run
  your code on 0.21 with ``python -W error::DeprecationWarning`` first.
- The life models moved to ``surpyval.life_models`` (``life_models.Power``,
  ``life_models.Exponential``, ...); the old top-level names warn until
  v0.23.
- Error and warning messages have one wording per condition: code that
  matches message text (an unknown option, ``alpha_ci``, a cause, a
  covariance, column lengths, "Monotone partial likelihood", which is now
  "No finite maximum: the partial likelihood keeps increasing ...") must
  update its patterns.
- ``model.formula`` is the ``str`` you gave, for every model (Cox,
  Buckley-James and competing-risks PH kept a parsed ``Formula``).
- Some results move, each towards a better answer: fits that stopped
  short now reach a verified maximum or warn (every likelihood fit has
  ``maximum``); accelerated-life and AFT time-varying standard errors use
  the exact information (an example's ``se(a)`` was 317, correctly 574);
  Gaussian and Student-t copulas no longer clip rho at 0.9999.
- The bundled lung and Rossi data's event columns are 1 for an event, as
  in R and lifelines.
- Importing surpyval corrects autograd's derivative of ``np.where`` for the
  whole process (#562).

**Versioning.** From this release, versions have two parts,
``MAJOR.MINOR`` (``0.22``, tagged ``v0.22``); every release takes the
next minor number. pip compares ``0.22`` and ``0.22.0`` as equal.

**Removed.** The names 0.21 deprecated (#422) are gone. An old argument
name (``seed``, ``confidence``, ``B``, ``t``, ``q``, ``u``, CoxPH's
``method``, ``id_col``, ``time_col``, the competing-risks ``how`` and
``cause``, ``CompetingRisks``' ``method``) is now an unknown argument and
raises ``TypeError``. The degradation calls in the old positional order
(``Z`` last) are no longer recognised and raise, except that a process
model fitted with stress now reads ``random(size, a, b)`` as ``Z=a,
random_state=b``. Also removed: the fitted ``CompetingRisks.method`` and
``CompetingRisksProportionalHazards.how`` aliases (use ``.how`` and
``.model``), the ``surpyval.experimental`` alias (use ``surpyval.beta.ml``)
and ``band``'s unused ``n_sims`` and ``random_state``. Saved models still
load. On 0.21, run your code or tests with ``python -W
error::DeprecationWarning`` first to find the calls to update.

**Behaviour changes.** Fits accept an optimiser's answer only when it is a
verified maximum, and a fit given ``init`` is also started from the default
start, so a few fits that stopped short silently now reach a better
maximum or warn; data with no maximum raise ``ValueError``. A
non-parametric ``df`` is the probability of each step. Kaplan-Meier and
Nelson-Aalen keep the estimate over a step with no one at risk, as R's
``survfit`` does. Gray's test and the competing-risks Cox incidences now
match R. Unknown option values raise ``ValueError`` everywhere. The
Uniform's MLE refuses censored data again. Bernoulli's ``sf`` is
``P(X > x)``, as for every other discrete distribution. Fits whose data
have no finite maximum warn "No finite maximum". A covariate column
the others already account for gets a ``nan`` coefficient and a warning,
as in R. ``FrailtyModel.summary()`` returns a ``DataFrame``. Covariate rows
that cannot be paired with the times raise ``ValueError``. The bundled
Rossi data's ``arrest`` is 1 for an arrest. Trend tests report a trend
only when it is significant. ``qf`` outside [0, 1] is ``nan``.
``param_names`` is deprecated in favour of ``parameter_names``. The
bundled lung data's ``status`` is 1 for a death. ``CoxPH.check_ph()``
returns a ``DataFrame``. Durations
and dates are refused. Probability plots draw failures only. ``fit_best``
no longer considers the Uniform and Beta4 by default. Small-sample Wald
bands change (#477).

- **Changed: load_lung()'s status means a death (#509).** It was stored as
  SurPyval's censoring flag (0 = death), the opposite of lifelines, so
  ``c = 1 - status`` fitted the complement (a Kaplan-Meier median of 588
  days instead of 310). ``status`` is now 1 for a death, as in lifelines
  and R; pass ``c = 1 - status``.
- **A fitted model's printout shows its data (#508).** For example ``Data
  : 60 units: 9 events at 9 unique times, 51 right censored``, with left,
  interval and truncated counts when present, counted in units (weighted
  by ``n``), and the number of distinct event times.
  Parametric, mixture, non-parametric, parametric regression, Cox and
  Buckley-James models print it, and keep it through ``to_dict``. A "1 =
  failed" column passed as ``c`` is now visible at a glance.
- **Changed: probability plots take label= and color= (#510).**
  ``Parametric.plot`` and ``MixtureModel.plot`` draw the points, fitted
  line and bounds in one colour (by default the axes' next colour), with
  ``label=`` on the fitted line, so fits overlaid on one plot can be told
  apart. The fitted line is solid and the bounds dashed; they were a black
  dashed line and red bounds. Overlaid plots keep both ranges in view.
- **Changed: CoxPH.check_ph() returns a table (#514).** A ``DataFrame`` as
  R's ``cox.zph`` prints it: a row per covariate and a ``GLOBAL`` row, with
  ``statistic``, ``df`` and ``p``. The old dictionary is
  ``proportional_hazards.diagnostics.check_ph(model)``.
- **Deprecated: cs(x, X) is cs(x, given) (#514).** The time already
  survived is named ``given``, as in the regression models' ``sf_tvc(...,
  given=)``; ``X=`` works until v0.23 with a ``DeprecationWarning``. Called
  by position, nothing changes.
- **Plots label their axes (#514).** The time axis is "Time" unless it is
  already labelled; non-parametric plots say "Survival probability" and
  "Kaplan-Meier estimate" (etc.), not "R" and "Model Survival Plot". A
  failure at exactly 0 now points to ``zi=True``.
- **Imperfect-repair fits are 10-270 times faster (#515).** The ARA,
  Kijima-II and G1 renewal likelihoods took a Python step per item and
  per event on every evaluation; they now step through event positions
  across all items at once, with the same virtual ages bit for bit (G1's
  likelihood to the last digit). ``ARA.fit`` on 100 items went from
  10-16 s to 0.45-0.7 s and on 1000 items from 73 s to 2.4 s;
  ``GeneralizedRenewal(kijima="ii")`` on 1000 items from 16.5 s to 1.8 s;
  ``GeneralizedOneRenewal`` on 100 items from 44 s to 0.16 s.
- **Faster saving and loading (#515).** ``to_dict`` and ``from_dict``
  visited every number of a model's arrays one at a time; number arrays
  now go through in one pass. The saved documents are byte-identical. A
  Kaplan-Meier model with 100,000 rows of data loads in 0.82 s (was
  1.22 s).
- **Changed: renewal models test the repair against perfect and minimal
  repair (#513).** ``repair_test(alpha_ci=0.05)`` gives two
  likelihood-ratio tests: against perfect repair (Kijima q = 0, ARA rho =
  1; ARI's rho = 1 is maximal repair) and against minimal repair (q = 1,
  rho = 0; G1 has none). Each refits the model with the restoration
  parameter fixed and reports the statistic and p-value, halved where the
  tested value is on the edge of the parameter's range (Self and Liang
  1987). The printed model shows the conclusion, for example "consistent
  with minimal repair; perfect repair rejected" on the issue's eight haul
  trucks (against q = 1: LR 0.40, p 0.53; against q = 0: LR 34.9, p
  2e-9), rather than a fixed rule on the width of the interval. The
  refits run the first time the model is printed or tested, and are
  cached.
- **REML warns when its between-unit covariance is on the boundary.**
  ``DegradationAnalysis(population_method="reml")`` could return a
  singular covariance (an intercept-slope correlation of 0.999998 on six
  units) without comment, while the default method warned on the same
  data. It now warns when the covariance with its smallest eigenvalue
  removed fits the data as well (to ``sqrt(eps)``): the estimate is on the
  boundary of the valid covariances, where standard errors and intervals
  are unreliable. On 144 simulated datasets it flagged all 18 boundary
  fits and none of the others. The default method's warning now says that
  REML may land on the boundary too.
- **Conditional sf_tvc is 1 at and before given (#523).** ``sf_tvc(x,
  schedule, given=g)`` returned :math:`S(x)/S(g)` for x < g as well,
  above 1 (1.75 for WeibullPO and 1.64 for CoxPH on the conformance
  fixtures). It is now 1 for x <= g, for the parametric regressions and
  Cox, along step schedules and covariate paths; a conformance property
  checks it.
- **Changed / deprecated: DataFrame columns are named with _col
  (principle 21).** Every ``fit_from_df`` (and ``fit_tvc_from_df``,
  ``fit_tvc_timeline_from_df``) names a column argument after the ``fit``
  argument it fills, with ``_col`` (``_cols`` for a list).
  ``Weibull.fit_from_df(df, x=, c=, n=, xl=, xr=, tl=, tr=)`` is now
  ``x_col=, c_col=, n_col=, xl_col=, xr_col=, tl_col=, tr_col=``
  (``tl_col`` and ``tr_col`` also take a number shared by every row), and
  ``DegradationAnalysis``, ``WienerProcess`` and ``GammaProcess`` take
  ``x_col=, y_col=, i_col=``, as the regression, recurrent and
  competing-risks fitters already did. The 0.21 names work until v0.23
  with a ``DeprecationWarning``. A conformance test checks the rule on
  every DataFrame entry point.
- **Changed: the concordance index leaves out tied event times by default.**
  ``sp.metrics.concordance_index``, the regression models' ``concordance()``
  and ``RandomSurvivalForest.score`` take ``ties="therneau"`` (the
  default: two events at the same time are not a usable pair, as in R's
  ``survival::concordance`` and lifelines) or ``ties="harrell"`` (Harrell's
  original definition, which counted them). On the lung Cox model (age,
  sex, ph.ecog), with 28 pairs of tied deaths, C is 0.637135, as R and
  lifelines give, instead of 0.636942. The deprecated
  ``surpyval.utils.score.score`` keeps Harrell's convention.
- **fit_from_df on every fitter (#511).** Kaplan-Meier, Nelson-Aalen,
  Fleming-Harrington, Turnbull, RoystonParmar, MixtureModel, the
  closed-form distributions, the copulas, FineGray,
  DestructiveDegradation, the survival trees and forest, and the recurrent
  fitters had no ``fit_from_df``. They now take a ``DataFrame``, naming the
  columns as their family already did (``x=``, ``c=``, ``xl=``, ``tl=``
  for one lifetime per row; ``x_col=``, ``i_col=``, ``c_col=``,
  ``Z_cols=`` for recurrent and regression data), and give the model
  ``fit`` gives on the same arrays, which the conformance suite checks for
  every registered model. ``Weibull.fit_from_df`` no longer casts ``c`` to
  an integer, which turned a missing flag into -9.2e18.
- **The concordance index is fast, and a metric (#512).** Harrell's C was
  a pairwise Python loop: 2.9 s at 5,000 subjects and about 5 minutes at
  50,000. A merge sort over the ranked scores gives the same value, with
  the same tie rules, in 0.01 s and 0.15 s. It is
  ``sp.metrics.concordance_index(x, c, risk)``, and every regression model
  has ``concordance()``, scoring its training data by default.
  ``surpyval.utils.score.score`` is deprecated until v0.23. With tied
  event times the value differs slightly from R and lifelines (0.6369 vs
  0.6371 on the lung Cox model), which do not count two events at the same
  time as a usable pair.
- **Changed: renewal models print the restoration factor's uncertainty
  (#513).** A generalized renewal fit to minimal-repair data printed q =
  2.63 with no sign that its 95% interval was [0.094, 73.4]. The printout
  now gives each parameter's standard error and Wald interval (also
  ``summary()``) and says when ``q`` or ``rho`` is not determined by the
  data or sits at the edge of its range. The docstrings say what ``q`` and
  ``rho`` mean. ``repair_test()`` tests the fit against minimal repair (on
  that data LR = 0.40, p = 0.53).
- **Bounds, mean life and accelerated life along a covariate path (#172,
  phase 2).** ``cb_tvc(x, Z, xl=None, given=None, on="sf", ...)`` bounds
  ``sf``, ``ff`` and ``Hf`` along a step schedule or a ``CovariatePath``,
  by the delta method on the same scale as ``cb``, with the quadrature
  mesh held at the fitted parameters; a constant path gives ``cb``, and
  in 1,000 simulated fits the 95% bounds covered the truth 94.6-96.2% of
  the time. ``mean_tvc(Z, xl=None, given=None)`` gives the mean (or with
  ``given`` the mean residual life) in one cumulative pass, accurate to
  about 1e-15, and ``inf`` with a warning where survival levels off.
  ``AcceleratedLife`` models, which refused every path, now follow
  Nelson's cumulative exposure along steps and paths for the Weibull,
  Exponential, Gamma and LogNormal (location families still refuse,
  saying why). AFT and accelerated life integrate one period of a
  periodic path, so 10 million cycles take about 1 ms.
- **Fixed: likelihood-ratio bounds widen with the confidence level
  (#535).** An ExpoWeibull 99% likelihood-ratio interval did not contain
  the 95% one, because the search stopped on a local extreme of the long,
  curved likelihood region. The 99% ``hf(13)`` lower bound was 0.1046,
  above the 95% one of 0.1017, and the ``qf(0.95)`` upper bound was 36.44,
  below the 95% one of 40.37. The search now follows each bound outward
  through the regions at 1/4, 1/2, 3/4 and all of the critical value, and
  gives 0.0714 and 80.6; the region approaches 0.0708 and 87.0 as alpha
  goes to 0. Every other registered model's likelihood-ratio bounds are
  unchanged to the bit.
- **Breaking: ARI takes its baseline intensity as baseline= (#507).** In
  ``ARA``, ``GeneralizedRenewal`` and ``GeneralizedOneRenewal``, ``dist`` is
  a lifetime distribution, but in ``ARI`` it was the baseline intensity
  model (``CrowAMSAA``, ``Duane``, ``CoxLewis``), so ``dist=sp.Weibull`` was
  an easy mistake. ``fit``, ``fit_from_recurrent_data``, ``fit_from_df``
  and ``fit_from_parameters`` now take ``baseline=`` (and
  ``fit_from_parameters``'s ``dist_params`` is ``baseline_params``); the
  old names work until v0.23 with a ``DeprecationWarning``. Saved models
  are unchanged, and old files load as before.
- **New: every likelihood fit says whether it reached a verified
  maximum.** The regression fits (PH, AFT, PO, AH, accelerated life and
  their time-varying forms), Cox, parametric and Cox frailty, proportional
  odds, Fine-Gray and competing-risks PH, mixture, Royston-Parmar, HPP,
  NHPP, proportional-intensity, renewal and copula models have
  ``maximum``, as ``Parametric`` does: ``"verified"``, ``"unverified"`` or
  ``"no finite maximum"``, agreeing with the fit's warnings, and saved by
  ``to_dict`` (an older dict reads ``"unknown"``). Lin-Ying and
  Buckley-James, which solve estimating equations, are ``"not
  applicable"``. A parameter on its bound where the likelihood is highest
  (a frailty variance of 0, an AMH copula at ``theta = 1``) is a verified
  boundary maximum. A new conformance property, ``maximum``, checks every
  likelihood fit in the registry and verifies each answer independently;
  the degradation process and destructive fits are its known gap (#564).
- **Fixed: fits that kept an unverified answer now polish it or say so.**
  The Nelder-Mead fits (NHPP, proportional intensity, renewal, copulas),
  Fine-Gray's BFGS, Cox's fallback root-finder, the HPP and Royston-Parmar
  took their optimiser's answer unchecked. Where results move, they move
  towards the maximum: Cox-Lewis by 1.7e-5 relative (log-likelihood up
  7e-9), a Gaussian copula's ``rho`` by 1.9e-6, a G1 renewal's ``q`` by
  1e-4.
- **Fixed: the truncated mixture fit kept L-BFGS-B's answer unverified
  (#560).** It is now polished and verified, as the EM fit is; two
  equivalent forms of the same data agree to 5e-7 (they differed by
  1.8e-5).
- **Changed: one wording for no finite maximum.** Cox's, Fine-Gray's,
  competing-risks PH's and Cox frailty's "Monotone partial likelihood: ..."
  now reads "No finite maximum: the partial likelihood keeps increasing
  ..."; update code that matches the old text. Cox frailty's EM warning
  says it "did not reach a verified maximum", and a mixture with a
  point-mass component warns once, not twice.
- **Fixed: rows censored with a finite truncation bound are read as the
  intervals they are in the fit checks (#559).** Data whose every row is
  right censored with a finite ``tr`` (or left censored with a finite
  ``tl``) was refused as having no failure, from the raw censoring codes.
  The checks and the start guess now read each row as the likelihood does.
  Such data still bounds no failure from one side, so its likelihood has
  no finite maximum unless a parameter is fixed: a free two-parameter fit
  is refused, as the same rows written as intervals are; a fit with a
  parameter fixed equals the interval fit; and the Exponential, Rayleigh,
  Poisson, Geometric, NegativeBinomial and DiscreteWeibull fits, which
  stopped at ``failure_rate = 3.6e-7`` or ``sigma = 313.5`` and reported
  it verified, warn "No finite maximum". MPS still refuses data with no
  exact value.
- **Fixed: distribution functions are quiet in the far tail and right at
  infinity (#561).** ``Weibull.sf`` far in the tail warned "overflow
  encountered in power" though its 0 was right; an overflow or division by
  zero inside a distribution function is no longer warned about (an
  invalid operation still is). A sweep of every registered distribution at
  extreme ``x`` and parameters also found 218 wrong ``nan`` values, each
  now the right value: a Gamma's or Poisson's ``sf(inf)``, a Weibull's
  ``df(1e300)``, Gumbel's ``df(inf)``, a NegativeBinomial past 1.3e154
  trials, a discretised Weibull's hazard from k = 1e6, a zero-inflated
  model's hazard at large ``x``. Hazards take their limits at infinity (a
  Gamma's rate, a NegativeBinomial's p). Fitted results are bit-identical.
- **Fixed: derivatives through ``np.where`` with a broadcast argument
  (#562).** autograd's rule for ``np.where(c, x, y)`` did not reduce the
  gradient to the shape of a broadcast ``x`` or ``y``: the gradient came
  back the wrong shape or raised, and where a later step summed it away
  the second derivative was silently wrong (7.52 instead of 4.46 in the old
  accelerated-life substitution). SurPyval registers a broadcast-aware rule
  when imported (``surpyval/utils/autograd_where_compat.py``), so every
  model and every custom distribution written with ``surpyval.np`` gets
  the right derivatives; no fitted value changes. **Behaviour change:**
  importing surpyval changes ``autograd.numpy.where``'s derivative for the
  whole process. A test notices when autograd fixes this itself, so the
  patch can go.
- **Fixed: ``Discretize`` has an exact Hessian at the first bin (#562).**
  The probability of ``k = 1`` took ``R(0)`` through the continuous
  formula, whose Hessian is not finite there (a Weibull's ``(0/α)^β``), so
  the covariance silently fell back to a numerical Hessian. ``R`` is now
  exactly 1 at the start of the support; values are unchanged.
- **A ``derivatives`` conformance property (#562).** For every registered
  model whose fit or inference differentiates its likelihood, the gradient
  and Hessian it takes at the fit, in its search space, agree with
  Richardson finite differences to 1e-6 of the standard-error scale (1e-5
  through the numerical incomplete gamma and beta shape derivatives); so do
  a parametric ``cb``'s delta-method gradients, the copulas' h-functions
  and density, and the degradation paths' Jacobians.
- **Fixed: accelerated life and AFT time-varying fits warn of no finite
  maximum and use the exact information (#555).** Stress levels without
  failures, or a covariate level with no events, let parameters run off
  silently; these fits now warn "No finite maximum" once, as the other
  regressions do. The covariance is the exact observed information
  instead of a numerical Hessian, which was up to 1e-3 of the standard
  errors off and much worse on ill-conditioned fits: the Arrhenius
  example's ``se(a)`` was 317 where the correct value is 574, so AL
  standard errors and bounds change. The AL likelihood's second
  derivatives were also wrong (autograd's ``np.where`` with a broadcast
  argument: a LogNormal ``Hf`` was 13% off; #562 audits the rest). AFT
  time-varying fits now search with their exact gradient and reach a
  slightly better maximum (parameters move by up to 6e-5 relative), in
  half the time.
- **Fixed: HPP proportional intensity restarts and warns (#554).** From a
  poor ``init`` the single BFGS search returned a rate of 0 with ``nan``
  coefficients, or its starting point, without a word. A user's ``init``
  is now followed by the default start, the better answer kept, and an
  answer that is not a verified maximum warns. Default fits are unchanged.
- **AFT models fit a covariate timeline from a DataFrame (#553):**
  ``WeibullAFT.fit_tvc_timeline_from_df`` (every AFT model), as PH, AH, PO
  and Cox have.
- **Fixed (breaking): the Gaussian and Student-t copulas no longer clip
  rho to ±0.9999 (#541).** Every formula silently evaluated ``rho =
  0.9999`` for any larger value: ``from_params([0.99995])`` gave a density
  of 81.11 at (0.3, 0.3001) where the true value is 114.68. They are now
  accurate for any ``|rho| < 1`` (within 1e-14 of a 40-digit integration);
  ``rho = ±1`` is still refused. A fit whose rho runs to ±1, which stopped
  silently near 0.99999, gives the standard no-finite-maximum warning
  recommending the comonotone or countermonotone model. Ordinary fits
  change only in the last digits.
- **Changed: ``RandomSurvivalForest.fit`` is quiet and takes ``n_jobs``
  (#546).** It printed joblib's progress on every fit. ``n_jobs`` (default
  1, the old sequential behaviour; -1 for every core) grows the trees in
  worker processes: 20 trees on 400 rows in 3.9 s with 2 jobs against 6.6
  s with 1. A seeded forest is identical whatever ``n_jobs`` is.
- **Fixed: a non-parametric survival tree or forest crashed on data mixing
  exact, right- and interval-censored rows (#543).** A node holding only
  exact and right-censored rows kept its parent's two-column times; it is
  now split and leafed as its rows would be on their own.
- **Fixed: mixture fits with a censored row and finite truncation (#544).**
  A right-censored row with a finite ``tr`` (or left-censored with a
  finite ``tl``) is the interval [x, tr] (or [tl, x]) since #310, but the
  mixture likelihood grouped rows by censoring code and raised
  ``IndexError``. It now uses the same observation masks as every other
  fitter, and equals a fit to the rows written as intervals.
- **Fixed: zero-inflated fits with a left truncation below 0 (#548).** A
  ``tl`` below 0 counted the mass at 0 as already excluded, so a no-op
  ``tl=-1`` sent ``f0`` to 1 with an unbounded likelihood. The mass now
  enters a window only from 0 on, as in the model's ``ff``, so ``tl < 0``
  gives exactly the untruncated fit (with ``lfp`` and ``offset`` too);
  ``tl = 0`` still excludes it, as for the discrete distributions.
- **Performance sweep.** Results are unchanged to the bit except where
  noted, checked by the new equivalence harness:

  - Cox residuals, ``check_ph`` and robust standard errors are linear time.
    They looped over the event times with a mask of every row: dfbeta on
    10,000 rows took 4.3 s and now takes 0.01 s; ``check_ph`` with four
    residual kinds on 100,000 rows 6.5 s, now 0.3 s.
  - A truncation time at or below the support's edge truncates nothing
    and is no longer evaluated. A ``tl`` of 0 sent the covariance to the
    numerical Hessian: a left-truncated Weibull at 100,000 rows fits in
    0.37 s instead of 0.80 s, and the covariance is now the analytic one
    (the registry's mixed-censoring Weibull covariance was 0.09% off; the
    Gamma's 4e-6). With an offset, rows truncated below the threshold
    gave a nan gradient, so the fit fell back to Nelder-Mead and stopped
    1.23 log-likelihood units short of the maximum with a "not verified"
    warning; it now converges, in 0.15 s instead of 4.6 s.
  - Likelihood-ratio bounds keep each likelihood their searches evaluate
    (18% were repeats): 10-60% faster.
  - Censored Gamma, Beta and Negative Binomial fits compute each
    incomplete function's complement only where it is used: a censored
    Gamma at 100,000 rows takes 0.77 s instead of 1.25 s.
  - Kaplan-Meier, Nelson-Aalen and Fleming-Harrington sort and validate
    their data once: 1,000,000 rows in 0.49 s instead of 0.72 s.
  - ``metrics.auc_td`` counts pairs by binary search: 100,000 rows and 20
    horizons in 0.48 s instead of 15.5 s.
  - Saving and reading models: the schema check walks the document once,
    and Cox models are read into plain arrays. A Cox JSON round trip at
    100,000 rows takes 0.41 s instead of 2.35 s.

- **Removed: the empty surpyval.alpha package**, which has held no models
  since v0.17.0, and the unused ``surpyval.utils.validate_tv_coxph_df_inputs``.
- **Development: large modules split (maintainability sweep, phase 1).**
  Code was moved only, checked bit-exact with the equivalence harness.
  The likelihood-ratio bounds are in
  ``univariate/parametric/_likelihood_ratio.py``, the optimised fits in
  ``optimised_fit.py`` and their input checks and starts in
  ``_fit_inputs.py``; the non-parametric support helpers in ``_support.py``
  and its bands in ``_bands.py``; ``surpyval/utils/__init__.py`` is split
  into ``data_formats``, ``validation``, ``covariates``, ``numeric`` and
  ``warnings``; and the remaining-useful-life classes are in
  ``degradation/rul.py``. Old import paths keep working.
  ``surpyval.utils`` now has an ``__all__`` of its 18 documented handlers,
  converters and helpers; everything else it exports is internal. A test
  stops new imports of another package's private names. In the same way,
  the parametric regression model's time-varying evaluation is in
  ``univariate/regression/_tvc_evaluation.py`` and its covariance and Wald
  bounds in ``_inference.py``; the Cox partial likelihood (tie terms,
  likelihood generators and the Newton-Raphson solver) is in
  ``proportional_hazards/cox_likelihood.py``; the fitted
  ``DegradationModel`` is in ``degradation/degradation_model.py``; and
  ``bootstrap_cb`` is in ``nonparametric/_bands.py`` beside ``band``.
- **A model's formula is the str you gave.** Cox, Buckley-James and the
  competing-risks PH model kept a parsed ``formulaic.Formula`` in
  ``model.formula``, every other model the str; now all of them keep the
  str, and a saved Cox model's formula no longer changes on a round trip
  (a time-varying Cox model saved ``"1 + dose"`` for ``"dose"``). Code
  that read ``.formula`` as a ``Formula`` should parse the str with
  ``formulaic.Formula(model.formula)``.
- **The semi-parametric fits share one input check.** Proportional odds,
  Lin-Ying additive hazards, Buckley-James, Fine-Gray and the
  competing-risks PH model now drop a row with a missing covariate (with
  the usual warning) before checking that the observed times are finite,
  as Cox did; they raised "must be finite" on such a row. The degradation
  models' input errors share one wording and name the offending input
  ("y must contain only finite values"; "x, y, and i must have the same
  length; got 55, 54, and 55").
- **Error and warning wording unified (messages only; no numbers change).**
  Code that matches the old texts must update its patterns.

  - An unknown option value (``bound``, ``on``, ``how``, ``method``,
    ``tie_method``, ``kind`` and the other enumerated arguments) raises
    ``'<name>' must be one of 'a', 'b' or 'c'; got <value>``.
  - ``alpha_ci`` outside (0, 1): ``'alpha_ci' must be strictly between 0
    and 1; got <value>``.
  - An unknown cause: ``Unknown cause 'x'; the causes are [...]``
    (``CauseSpecificMCF.mcf`` and ``mcf_cb`` raised ``KeyError``; they now
    raise ``ValueError``). A missing cause: ``<what> is of one cause at a
    time; pass `event`.``
  - No covariance, a singular information matrix, unknown parameter names
    in ``param_cb`` and ``fixed`` (the univariate fit now lists every
    unknown name), and ``c``, ``n``, ``tl`` or ``tr`` of the wrong length
    (``'c' must be the same length as 'x'``) each have one wording.
  - ``fit_from_df`` given neither ``Z_cols`` nor ``formula``: ``One of
    'Z_cols' or 'formula' must be provided``.
  - Every fit that stops short of a verified maximum gives one warning,
    "... did not reach a verified maximum of the likelihood (<reason>)",
    at the caller's line. The univariate "Precision was lost" and the
    regression and NHPP "did not converge" warnings are this warning now,
    and ``quiet_maximum_warnings`` holds it back for every fit, not only
    the univariate MLE.
  - ``RandomSurvivalForest.sf`` refuses an unknown ``ensemble_method``; it
    used ``'sf'`` for anything other than ``'Hf'``.
- **Development: duplicated code merged (consolidation sweep, phases 1-3).**
  Fitted numbers are bit-identical, checked with the equivalence harness.
  The PH, AFT and PO fits share ``fit_log_linear`` and
  ``split_log_linear``; the accelerated-life and AFT time-varying fits are
  assembled by ``assemble_regression_model``; the time-varying DataFrame
  fits share ``fit_tvc_df``; the Efron and Breslow Cox generators share
  one risk-set setup; the semi-parametric models share one covariate
  centring and one ``LinearPredictorMixin``; the plain and
  proportional-intensity NHPP fitters, and HPP, share one log-likelihood
  (#350); the bootstrap tails share ``percentile_bounds`` (#351); and the
  degradation inputs are checked by one ``validate_xy`` (#352); option
  checks go through ``utils.validation.check_option`` and unverified-maximum
  warnings through ``utils.no_maximum.warn_unverified``; and the
  ``fit_from_df`` design matrices are built by ``design_matrix_from_df``
  alone (``wrangle_and_check_form_and_Z_cols`` is removed). The
  accelerated-life fitter now has the deprecated ``param_names`` alias
  the other fitters have.
- **Development: fitted regression models declare their attributes
  (maintainability sweep, phase 2).** ``ParametricRegressionModel`` and the
  Cox, ProportionalOdds, Lin-Ying, Buckley-James and Fine-Gray model classes
  declare every attribute their builders set, with its type and meaning.
  The covariate links (``Phi``, the additive link and ``from_dict``'s
  namespace) are one ``CovariateLink`` (``name``, ``phi_param_map``,
  ``phi``; ``Phi`` remains as an alias so old pickles load). ``fit`` now
  also sets ``dist``, and ``from_dict`` sets ``distribution_param_map`` and
  ``phi_param_map``, which only the fits set before. A new conformance
  property, ``attributes``, checks that ``fit``, ``fit_from_df``, a formula,
  ``fit_tvc`` and ``from_dict`` give a model the same declared attributes
  (``from_dict`` less the data and what was computed from it) and nothing
  undeclared; before this change it failed 33 cases. The regression family
  names are constants in ``regression/_kinds.py``, and the model branches on
  ``_is_accelerated_life()`` and ``_is_additive()`` instead of comparing
  ``kind`` with string literals; ``kind`` and its values are unchanged.
- **Development: long functions split into named steps; flake8
  ``max-complexity`` lowered from 70 to 25 (maintainability sweep, phase
  2).** ``handle_xicn``, the tvc-schedule expression evaluator, the MLE
  fitter, ``xcnt_handler``, ``turnbull``, ``DegradationAnalysis.fit``, the
  fit-input validation and the harness's ``diff`` are split into named
  steps, and the likelihood-ratio search behind ``cb(method="lr")`` is a
  ``_PsiBoundSearch`` class rather than eleven closures. No result
  changes: the equivalence harness is bit-identical, and old-against-new
  runs over thousands of inputs per function give identical values,
  warnings and error messages. Comments in these modules state current
  behaviour rather than its history, and the copula and renewal
  tolerances are named for what they are (``_U_CLIP``, ``_LOG_FLOOR``,
  ``_MACHINE_EPS``).
- **Development: tests organised by feature (maintainability sweep,
  phase 3).** The 59 test modules named after fix rounds
  (``*_fixesN.py``, ``*_roundN.py``, ``test_tvc_phase2.py``, ...) are
  renamed as, or merged into, feature modules; the 17,023 collected tests
  are unchanged apart from their paths. Helpers copied between modules
  live once in ``surpyval/tests/_helpers.py``, the conformance registry is
  split into fixtures, family helpers, cases and known failures (still
  imported through ``registry.py``), and :doc:`Contributing` says where a
  fix's regression test goes.
- **Development: refactors are proven bit-identical.**
  ``scripts/refactor/snapshot.py`` records what every registered model
  computes and says (fits, predictions, every bound, ``to_dict``,
  printouts, warnings, and errors on invalid input), plus time-varying,
  bootstrap and recurrent fits, the public API, the import set and the
  test IDs, so ``compare`` shows a change altered nothing (see
  :doc:`Contributing`). CI lints with isort as well, and covers
  ``conftest.py`` and ``scripts/``; flake8 caps function complexity (now at 25);
  mypy reports unused ``type: ignore`` comments, redundant casts and
  impossible comparisons; a test fails once the version reaches
  ``REMOVED_IN`` (0.23) while deprecated names are still accepted; and
  the nightly refit study covers the seven models added in 0.22 (#545).
- **Added: Joe, Ali-Mikhail-Haq and Student-t copulas, and rotations
  (#157).** ``surpyval.multivariate.Joe``, ``AMH`` and ``StudentT`` have
  censoring- and truncation-aware likelihoods, Kendall's tau, Spearman's
  rho and tail dependence, conditional-inversion sampling and
  serialisation. The Student-t CDF is a deterministic integral within
  1e-11 of Genz's exact algorithm (scipy's ``multivariate_t.cdf`` is
  randomised and about 1e-4 off). A Student-t fit to data with no tail
  dependence warns that ``nu`` has no finite maximum and recommends the
  Gaussian copula. ``rotation=`` (90, 180, 270) for the Clayton, Gumbel
  and Joe copulas gives dependence in the other tail; 180 is the survival
  copula. Parameter recovery was checked over 200 replications with
  censoring for every family.
- **Fixed: copula accuracy (#291).** A numerical review of every copula
  against reference values (pyvinecopulib, R's mvtnorm, 50-digit mpmath):

  - Spearman's rho for Clayton and Gumbel was estimated from 50,000
    simulated pairs, up to 5e-3 off (Clayton at theta 0.5: 0.2901 for
    0.2949). The default Kendall's tau and Spearman's rho of any copula are
    now integrals, accurate to about 1e-11.
  - Gumbel's h-function and density underflowed near the upper corner:
    the density was inf at (0.98, 0.98) for theta 100. They are now closed
    forms in log space, within 1e-9 of 50-digit references.
  - **Breaking:** with margins passed already fitted, the data were not
    checked: a NaN returned the starting value with a NaN likelihood, and
    negative counts, ``xl > xr`` and values outside the truncation window
    were used. Each series is now checked like univariate data, and the
    error names the series.

- **Added: log-normal shared frailty (#343).** ``Frailty(dist,
  family="lognormal")`` fits :math:`u = e^w`, :math:`w \sim N(0, \theta)`,
  as frailtypack, coxme and ``survival::frailty(dist="gaussian")`` define
  it, so ``theta`` is the variance of :math:`\log u` and the median
  frailty is 1. Each group's likelihood is integrated by 30-node adaptive
  Gauss-Hermite quadrature, centred on the group's mode: within 3e-10 of
  scipy's adaptive quadrature for :math:`\theta \le 1`. On the kidney
  data it matches R's lme4 (``nAGQ = 25``) to 6e-13 in the log-likelihood.
  Gamma stays the default. ``frailty_variance`` (Var(u)/E(u)^2) and the new
  ``kendall_tau`` compare the two families on one scale. **Breaking:** an
  unknown ``family`` raises ``ValueError`` (it was ``NotImplementedError``).
- **Added: shared frailty with a Cox baseline (#342).** ``CoxFrailty``
  fits a gamma frailty with an unspecified baseline, by EM over the
  frailties with ``CoxPH``'s partial likelihood as the M-step (accelerated
  by SQUAREM), and ``theta`` maximising the profile likelihood. It
  reproduces R's ``coxph(... + frailty(id, dist="gamma"))`` on the kidney
  data: theta 0.40777, the ``female`` coefficient -1.5832 with standard
  error 0.4484, and I-likelihood -181.6386, with Efron ties (Breslow too).
  It predicts marginally or for an observed group, and saves and loads.
  In 200 simulated fits its coefficient and theta intervals covered 94% and
  97%. ``load_kidney()`` adds the kidney catheter data (McGilchrist and
  Aisbett 1991; R's ``survival::kidney``).
- **Added: a semi-parametric proportional odds model (#341).**
  ``surpyval.ProportionalOdds`` is the proportional-odds counterpart of
  ``CoxPH``: the covariates multiply the survival odds of a baseline left
  to the data, :math:`S(x|Z)/F(x|Z) = e^{\beta'Z} S_0(x)/F_0(x)`. It is
  fitted by nonparametric maximum likelihood (Murphy, Rossini and van der
  Vaart 1997), with an exact solve for the baseline at each step, and its
  standard errors come from the profile likelihood. A positive coefficient
  means a longer life, as in ``LogisticPO``; R's ``timereg::prop.odds``
  reports the negatives. It takes observed and right-censored data with
  left truncation, and refuses other censoring with a ``ValueError``.
  The coefficients agree with R's ``survival::coxph`` with a unit gamma
  frailty per subject (the same model) to 1e-6 on the lung data and 1e-7
  on the Rossi data, and the intervals covered 94-96% in 1,000-replication
  studies.
- **Changed: the life models are in surpyval.life_models.** ``Power``,
  ``InversePower``, ``Eyring``, ``InverseEyring``, ``Linear``,
  ``InverseExponential``, ``DualPower``, ``DualExponential``,
  ``PowerExponential``, ``GeneralLogLinear`` and the ``LifeModel`` base
  class are used as ``AcceleratedLife(Weibull, life_models.Power)``. The
  exponential (Arrhenius) life model is ``life_models.Exponential``: at the
  top level that name is the distribution, so it was
  ``ExponentialLifeModel``. The old top-level names still work until
  v0.23, with a ``DeprecationWarning`` naming the new one. Every life model's
  docstring now gives its formula, what each parameter means (``a`` of
  the Arrhenius model is :math:`E_a / k_B`, for example), its constraints
  and an example.
- **Breaking: GeneralLogLinear fixed, with a constant term, and exported
  (#530, #345).** ``AcceleratedLife(dist, GeneralLogLinear).fit`` raised
  an autograd broadcast ``ValueError`` on any data. The life model is now
  L(Z) = c exp(sum of beta_j Z_j), with one coefficient per column of
  ``Z``; with no constant, L(0) was 1 in whatever unit the times were in
  (principle 6). With the Weibull it reaches the Weibull AFT maximum
  (log-likelihood -422.414831 in both). It is importable as
  ``sp.life_models.GeneralLogLinear``, round-trips through JSON (a model saved now
  cannot be read by 0.21), and is in the conformance registry.
  ``LifeModel.resolve(n_stresses)`` builds the model for a number of
  columns, so its parameter map and bounds are the types ``LifeModel``
  declares.
- **Forest importance no longer silently NaN (#533).** When an out-of-bag
  row had zero ensemble probability, ``feature_importances`` returned NaN
  for every feature without a warning (3 trees, 1 row of 46). Each drop is
  now over the rows that are finite before and after the shuffle, the same
  as before when every row is, and one warning gives the counts and
  recommends more trees or ``kind="exponential"``. ``oob_log_likelihood``
  warns the same way when it returns -inf.
- **Normal and LogNormal 2-4 times faster (#469).** ``sf``, ``ff``, ``df``,
  ``qf`` and their logs use ``scipy.special`` directly instead of
  ``scipy.stats.norm``, with bit-identical values: ``Normal.sf`` on 1,024
  values takes 41 µs instead of 96 µs, ``qf`` 33 µs instead of 110 µs,
  and a censored fit 30 ms instead of 42 ms. The density no longer emits
  numpy's overflow warning at \|x\| near 1e300.
- **import surpyval is 0.3 s faster (#470).** The regression,
  competing-risks, recurrent and degradation models and the metrics load
  on first use, and pandas and formulaic are imported only where used
  (1.16 s to 0.86 s). Every name, ``dir(surpyval)`` and attribute access
  such as ``surpyval.recurrent.laplace`` work as before.
- **Faster no-maximum check (#501).** A coefficient at or near 0, which
  Newton's step leaves unchecked, had its profile read with a third
  derivative and two full Hessians: 65% of a 100,000-row LogNormal AFT
  fit. It now uses Hessian-vector products. The criterion and its verdicts
  are unchanged, and its curvature is exact where the old one lost
  digits. The fit takes 2.7 s instead of 4.8 s.
- **Faster likelihood-ratio bounds (#519).** The searches evaluate the
  distribution's formulas without the guards the data cannot trip
  (bit-identical values), and for two-parameter models the region's
  boundary is traced once and each search starts from it. A Weibull
  ``cb(method="lr")`` at 20 points on 1,000 units takes 2.4 s instead of
  9.4 s, and at one point 0.56 s instead of 1.1 s. Bounds are unchanged
  to 2.3e-7. A steep LogNormal hazard bound that ended 5e-8 outside the
  region now sits on its boundary.
- **Incomplete beta tails (#520).** Where a small tail's ``x`` rounds
  towards 1, as with a NegativeBinomial ``r`` near 1e-172, the continued
  fraction ran to 100,000 terms (4.6 s per call) and was 0.5 nats off.
  That tail now comes from the other side's power series, to 1e-15 of
  mpmath, in milliseconds, and an unconverged continued fraction gives
  nan instead of a wrong value.
- **Faster degradation bootstrap bounds (#522).** Each refit reuses the
  units' path fits and warm-starts the life fit from the full-data
  estimate, and every refit still reaches a verified maximum. 200 units
  with 200 resamples take 5.4 s instead of 9.7 s, and the bounds change by
  at most 4e-7.
- **MCF variance in linear time (#521).** The Lawless-Nadeau variance and
  each item's observation window took a pass over all times or rows per
  item. At 5,000 items the fit takes 0.27 s instead of 1.8 s, with the
  same variance to 2e-13.
- **Design principle 24: simple by default, more as an option.** When a
  method reaches its limit, the new approach is added as an option beside
  it. The default stays the simple, standard method, and changes only
  when it is wrong for the usual case.
- **Fixed: survival along a continuous path at far-out times.**

  - **Hazard shut off by the path.** ``sf_tvc`` along a ``CovariatePath``
    started with one quadrature panel from the last earlier edge to the
    query. When the path shuts the hazard off, all of the hazard sits near
    the model's time scale, and that panel never sampled it. For example,
    with a Weibull(10, 2) PH model and ``Z = -0.2t``, ``sf_tvc(1e21)``
    gave 1.0, or sf(1) when queried with t = 1, against the exact
    exp(-0.5) = 0.6065. The starting mesh now has edges a factor of 2
    apart from below the model's time scale up to the query, about 110
    panels for 1e21.
  - **Hazard that overflows.** A hazard that grows without bound along
    the path overflowed to inf - inf and gave NaN with a raw numpy
    warning. It now gives survival 0.
- **Fixed: an additive hazards model's cb below 0.** It gave a band where
  ``sf`` is 1 (WeibullAH ``cb(-1, z)`` was [0.81, 0.97]); it is now [1, 1].
- **Survival forests grow 8-90 times faster (#190, #518).** The Weibull
  split ran a Nelder-Mead fit for every candidate child (2 trees at n = 300:
  21 s). On observed and right-censored data each child's maximum is now
  found directly, every candidate of a feature at once: the exponential
  rate and Weibull scale in closed form, the Weibull shape from its
  profile likelihood (0.23 s). The log-rank split sorts each feature once
  and scores every threshold from cumulative counts (10 trees at n = 1000:
  2.2 s, was 16.5 s). The chosen splits are unchanged, so seeded forests
  predict exactly as before. Data with left or interval censoring or
  truncation still runs an optimiser for each candidate.
- **A forest's first prediction is about 20 times faster.** Each Weibull
  or exponential leaf was fitted by a full ``Weibull.fit`` the first time a
  prediction reached it: 93 s for a 20-tree forest on 1,000 rows, five
  times as long as growing it. On observed and right-censored data a leaf
  is now the maximum found as the split search finds a child's (the
  exponential rate in closed form, the Weibull from its profile
  likelihood, with the shape searched as widely as ``Weibull.fit`` does),
  built from its parameters: 4.3 s. The parameters agree with
  ``Weibull.fit`` to its tolerance (about 1e-5), so predictions move in
  their last digits. Such a leaf has no ``cb()`` of its own.
- **Changed: trees and forests know their covariate names (#192).**
  ``fit_from_df`` takes a ``formula`` as well as ``Z_cols``, and ``fit``
  takes a ``DataFrame`` ``Z``; the fitted model keeps ``feature_names`` and
  serialises them, ``print(tree)`` shows the splits by name (``temp <=
  42``) and each leaf's model, and predictions read a ``DataFrame`` by name.
  ``RandomSurvivalForest.feature_importances`` is a ``pandas.Series`` keyed
  by feature name (it was an array).
- **min_split_gain for the likelihood trees (#189).** A ``"weibull"`` or
  ``"exponential"`` node splits only if its best cut raises the maximised
  log-likelihood by more than ``min_split_gain``: a number, ``"aic"`` (the
  kind's parameters, 1 or 2) or ``"bic"``. The default, 0, keeps the old
  behaviour, which suits a forest; ``"aic"`` is the setting for a single
  tree (on no-effect data, 3.4 leaves on average instead of 32).
- **Split housekeeping (#193).** ``log_rank_split`` called directly on left-
  or interval-censored or right-truncated data raised ``IndexError`` or
  returned a wrong split; it now raises ``ValueError``. ``min_leaf_failures``
  counts failures weighted by ``n`` in every split, so a row with count n
  and n identical rows give the same tree.
- **Non-parametric trees on truncated data (#188).** ``SurvivalTree`` and
  ``RandomSurvivalForest`` with ``kind="non-parametric"`` refused
  right-truncated data, and truncated data with left or interval
  censoring. They now take the Turnbull-score split with Turnbull leaves:
  a truncated row's log-rank score is that of its likelihood given its
  truncation window (the event's score less the window's, under the
  pooled Turnbull estimate fitted with the truncation). On left-truncated
  right-censored data these are the delayed-entry martingale residuals.
  Without the window term, a covariate that changed only the truncation
  was found significant in 39-49% of data sets at the 5% level; with it,
  0.5-3%. Every tree kind now accepts the full data model.
- **Breaking: Kaplan-Meier bands hold their level (#390).** The
  equal-precision band covered 0.87-0.89 for a nominal 0.95 (0.83
  untransformed), most misses at the first events, where its boundary is
  unbounded and the estimate rests on a few failures. ``band()`` now forms
  its bands on the arcsine-square-root scale by default
  (``bound_type="arcsine"``; Borgan & Liestøl 1990), and the
  equal-precision band covers 0.1 <= a <= 0.9 by default, NaN outside;
  ``x_range=(t_L, t_U)`` sets any range. Coverage is now 0.94-0.96 at n =
  40-400; Hall-Wellner covers about 0.95. Pass ``bound_type="exp"`` for
  the old scale. ``cb()`` is unchanged.
- **Breaking: parametric regression Wald bands on the baseline family's
  scale (#504).** As for the univariate (#477) and degradation models, the
  ``sf``/``ff``/``Hf`` band is now formed on ln H (Weibull, Exponential,
  Rayleigh, Gumbel), the normal quantile of F (Normal, LogNormal) or the
  logit (the rest), from the cumulative hazard, and ``cb_tvc`` follows. A
  model with its coefficient fixed at 0 now gives the univariate band (it
  was up to 120% away). On ten-point samples the Weibull PH and AFT bands
  turned back in a tail in 200 of 200 fits and now never do. Small-sample
  bands change; at n = 2000 they move by at most 1.4% of their width. The
  three families of model share one helper in ``surpyval.utils.linalg``.
- **Proportional-intensity regressions alias (#502).** A repeated column
  was split (-0.231 as -0.116 / -0.116) and a constant column took part of
  the baseline (rate 0.0807 became 0.0764), silently. Both now give a NaN
  coefficient, ``model.aliased``, one warning, and the fit without the
  column. A constant is aliased where the baseline has a scale
  (``has_scale``: HPP, Duane, Crow-AMSAA, Cox-Lewis).
- **Dual-stress life models alias an undetermined stress effect (#503).**
  With equal (or collinear) stresses, DualPower and DualExponential split
  one effect between two parameters (-1.174 as -0.568 / -0.605), silently.
  The later stress's parameter is now NaN, with one warning naming the
  column, and the fit is the single-stress one. PowerExponential is
  unaffected.
- **Renewal models: intervals at a boundary (#461).** A ``q`` driven to 0,
  or a ``rho`` to 1 or 0, gave NaN ``param_cb`` for it and for ``alpha``
  (whose variance was -10.6). The restoration parameter now gets its
  one-sided profile-likelihood interval (e.g. q in [0, 0.069]); the others
  get Wald intervals of the model held at the edge, and the printed table
  says so.
- **MixtureModel: no false "max iterations" warning, and faster (#506).**
  EM on a censored mixture crawled for 1000 iterations and warned at the
  maximum. After at most 20 EM iterations the fit is polished by direct
  maximum likelihood with autograd gradients, and a verified maximum is
  accepted. The issue's case takes about 1 s (7.4 s before on the same
  machine), with no warning and a slightly better maximum (738.046941
  against 738.047053).
- **Proportional-odds cumulative hazard is accurate where it is small
  (#528).** ``Hf``, ``log_sf`` and ``Hf_tvc`` of every PO model computed
  H0 - ln(phi) + ln(F0 + phi S0), whose terms cancel to about 1e-16 in
  absolute terms: 20% wrong at H = 4e-16, and ``log_sf`` was -inf where S0
  underflows. They now use ln(1 + F0/(phi S0)). Checked against 50-digit
  values for all seven baselines; fitted parameters are unchanged.
- **ExpoWeibull likelihood is accurate at extreme shapes (#472).** The
  log-density and hazard added terms of size beta ln(x/alpha) that cancel,
  an error of about 1e4 per point at beta = 7e19, so the likelihood came
  out up to 3.7e6 in deviance above the fitted maximum and a
  likelihood-ratio search met -2e139. They are now computed without the
  cancellation, and the likelihood-ratio band's walk starts from the
  points its direct searches reached.
- **Every parametric fit searches in the units of its start (#366).** Only
  offset fits did; the others searched a scale as a log below 1 and
  linearly above it, a different search in every set of units. Fits to
  data in millionths and in millions now agree to rounding (1e-16 to
  1e-11; up to 2e-6 before, and 3% for an ExpoWeibull MSE fit). Fits move
  only in their last digits. A Beta4 fit on data where its likelihood is
  unbounded can now warn "No finite maximum" instead of "did not reach a
  verified maximum".
- **Fine-Gray fits in linear time (#517).** ``FineGray.fit`` (and
  ``CompetingRisksProportionalHazards(model="Fine-Gray")``) built a dense
  events-by-rows matrix of censoring weights and used it in every
  likelihood, gradient and Hessian evaluation: 1.3 s and 476 MiB at 10,000
  rows, and impossible at 100,000 (22 GB for the matrix alone). The risk
  sets are now cumulative sums over the rows in time order, and the
  censoring Kaplan-Meier (which the prediction metrics use too) is a
  suffix sum, bit-identical. A fit takes 0.036 s at 10,000 rows and
  0.63 s at 100,000, with memory linear in the rows (34 MiB). Results
  agree to about 1e-14, except in a few percent of large fits where the
  optimiser stops one iteration apart (coefficients within its
  tolerance, up to 3.4e-6 relative); it now stops on "precision loss"
  less often.
- **CoxPH fits are 4-10 times faster at 100,000 rows (#516).** The
  information matrix was built as one p x p matrix per event time, with
  truncation terms computed even with no truncation, and the solver built
  it about 14 times per fit. It is now one product over the rows, the
  truncation terms are skipped when nothing is truncated, and the
  coefficients come from Newton-Raphson with step-halving (as in R's
  ``coxph``), about 5 steps, falling back to the previous solver where it
  fails, as where the likelihood has no finite maximum (that warning is
  unchanged). At 100,000 rows and 5 covariates, Efron takes 0.28 s with
  ties (was 1.6 s) and 0.66 s without (was 5.6 s); time-varying fits on
  40,000 intervals take 0.24 s (was 0.85 s). The score is now solved to
  rounding, so results change only in the last digits (at most 4e-13
  relative). ``tol`` is the largest step, in standard errors, at which the
  fit stops.
- **Faster Efron Cox fits with tied times (#515).** The Efron score was
  computed on a masked (times x largest tie x covariates) array: one
  51-way tie among 30,000 rows took 10.3 s instead of 1.1 s. The sum over
  tied deaths is now factored and stored per death: 1.4 s. Time-varying
  Cox fits on 40,000 rows take 0.53 s (was 8.3 s). Untied and Breslow
  fits are bit-identical; tied Efron fits agree to the last digits.
- **Frailty and parametric additive hazards fits use the gradient
  (#515).** They began with thousands of Nelder-Mead evaluations. They now
  run a gradient search first and keep its answer when it is a verified
  maximum, falling back to the old search otherwise (the #376 and #392
  warnings are unchanged). At 10,000 rows: WeibullFrailty 2.2 s to 0.13 s,
  GammaFrailty 23 s to 1.4 s, WeibullAH 1.0 s to 0.21 s. On some data the
  old frailty search stopped with the variance at 0, up to 0.1
  log-likelihood units short of an interior maximum the new one finds.
- **Faster log-rank test, Turnbull, Fleming-Harrington and
  competing-risks fits (#515).** Results are bit-identical. ``logrank``
  built an at-risk array of rows by event times: 13 s and 2 GB for 30,000
  rows in three groups, and out of memory at 100,000; with running totals
  it takes 0.02 s and 0.06 s. The Fleming-Harrington estimator,
  ``Turnbull``'s default, summed each step's tied events in a Python loop
  on every EM iteration: a Turnbull fit to 1,000 random intervals takes
  0.22 s (was 7.9 s) and ``FlemingHarrington.fit`` on 100,000 rows 0.05 s
  (was 0.74 s). ``CompetingRisks.fit`` searched the distinct times for
  every row: 0.07 s at 100,000 rows (was 1.65 s); the new
  ``surpyval.utils.missing_events`` finds the censored rows in one pass.
  Grouping tied rows sorts the data once instead of three times, halving
  a tied 100,000-row Weibull fit.
- **Faster Kaplan-Meier, Nelson-Aalen, AFT and proportional-odds fits
  (#498, #499).** Greenwood's and the Nelson-Aalen variance snapped each
  ``d / r`` to a whole number in a Python loop, most of a large fit;
  vectorised, a 100,000-row Kaplan-Meier fit takes 50 ms instead of 207
  ms (Nelson-Aalen 53 ms, was 228), with bit-identical variances and
  bounds. AFT and PO fits started with Nelder-Mead, hundreds of
  derivative-free evaluations; with a differentiable likelihood they now
  take the gradient ladder the PH models use first, and fall back to the
  old ladder only when that cannot verify its optimum. At 100,000 rows and
  5 covariates a WeibullAFT fit takes 0.48 s (was 2.0 s) and WeibullPO
  0.64 s (3.9 s); at 1,000 rows AFT and PO fits are 4-6 times faster. They
  reach the same maximum to 1e-6 in the log-likelihood, or a slightly
  higher one. The time-varying AFT likelihood is not differentiable and
  keeps Nelder-Mead.
- **Likelihood-ratio bounds at the edge, and along valleys (#421).**
  Profiles are now searched on the log or logit scale of each parameter,
  each point starting from the ones already solved. The old search on raw
  parameters, restarted from the fit every time, overstated the profiles:
  ExpoWeibull 42.4 for 0.29 at mu = 1e-4, NegativeBinomial 3.05 for 2.35
  at p = 0.999999. Where a profile levels off below the critical value,
  the bound is the edge of the parameter's space. A NegativeBinomial
  ``r`` tends to a shifted Poisson with deviance 2.345, so its 95% upper
  bound is ``inf``, not 1.7e16, and its 80% bound (36.2) is finite.
  ExpoWeibull ``beta`` is now [0.053, inf] at 95% and [0.486, inf] at 80%;
  the old [0.171, 367.5] and [0.486, 372.0] were not nested. A band is
  where the function's own profile reaches the critical value, and
  ``sf``, ``ff`` and ``Hf`` share one band. NegativeBinomial ``hf(2)`` at
  95% is now 0.305 (was 0.207), and the ExpoWeibull bands that were
  ``nan`` are found. The NegativeBinomial sweep no longer leaks over
  14,000 raw numpy warnings. Likelihood-ratio bounds are slower for the
  simple families (a Weibull ``cb`` went from about 0.06 to 0.2-0.5 s).
- **Continuously varying covariates: CovariatePath (#172, phase 1).**
  ``sf_tvc`` and ``Hf_tvc`` accepted only step schedules, so a ramp-stress
  profile or a thermal cycle had to be cut into steps. That was slow (150
  ms for 1000 steps), and only accurate to the square of the step width,
  with no error reported. ``CovariatePath.from_points(times, values,
  period=None)`` (straight lines; a repeated time is a jump) and
  ``CovariatePath.from_callable(func, p=1, breakpoints=None, period=None)``
  now describe the path. PH, AH, PO and AFT integrate the hazard (AFT the
  accelerated age, Nelson's cumulative exposure) by adaptive Gauss-Kronrod
  quadrature to 1e-10 relative error on H, with one ``RuntimeWarning`` if
  that is missed. Cox sums its baseline jumps along the path exactly. The
  error against closed forms is 5e-16 to 9e-14, in about 2 ms for 200
  times. ``given=`` integrates from the conditioning age. A step schedule
  is still summed exactly, and a flat path gives the same result to
  rounding. A path evaluates a fitted model only: fitting still uses steps
  (``fit_tvc``).
- **Out-of-bag log-likelihood and permutation importance for the random
  survival forest (#186).** The forest could only be scored by
  concordance, which needs right-censored data.
  ``RandomSurvivalForest.oob_log_likelihood()`` now scores every training
  row by its full likelihood (density, S, F or interval probability, over
  the truncation probability) under the trees that did not see it, for
  every censoring type and truncation. ``feature_importances(n_repeats=5,
  random_state=None)`` reports how much that score drops when a feature is
  shuffled among each tree's out-of-bag rows. For this score a
  non-parametric leaf is read as a continuous distribution (linear between
  its drops, exponential after the last). On the docs example the score
  rises from -2.644 without splits to -2.484 with them.
- **Non-parametric survival trees on left- and interval-censored data
  (#188, stage 1).** ``kind="non-parametric"`` raised on such data. It now
  splits on the log-rank scores of the node's pooled Turnbull estimate
  (the standardised left-child sum, with its permutation variance), with
  Turnbull leaves. On right-censored data the scores are exactly the
  classic log-rank scores, so the split chooses as the log-rank split does.
  Truncation combined with left or interval censoring, and right
  truncation, still raise (stage 2).
- **Added: random_state for SurvivalTree and RandomSurvivalForest
  (#471).** The bootstrap samples and each split's candidate features
  were drawn from numpy's global stream, so a forest could only be
  reproduced by seeding numpy globally, and fitting one disturbed the
  global stream. ``random_state=None`` still draws from the global
  stream exactly as before, so forests under ``np.random.seed`` are
  identical. An int or ``Generator`` gives the forest its own stream,
  with a child stream per tree, and leaves the global one alone.
- **Added: conditional-inference trees, selection="ctree" (#188).**
  Greedy search prefers covariates with many values and always splits:
  in 60 simulated data sets it chose a noise covariate 47% of the time
  over a two-valued covariate with a real effect. With
  ``selection="ctree"`` each node chooses its covariate by the
  Bonferroni-adjusted p-value of its maximally selected score statistic,
  and splits only if that p is below ``alpha_split`` (0.05). The scores
  are log-rank for ``"non-parametric"``, and the working model's score
  contributions for ``"exponential"`` and ``"weibull"``. The p-value is
  exact for the statistic's asymptotic chain. Noise is then chosen 10% of
  the time, and on null data 96% of trees stay a single leaf, where
  greedy search always splits. It works for every kind and every
  censoring type; the default is unchanged.
- **Beta4: a fit with no maximum says so, and MPS is recommended (#385).**
  The four-parameter Beta's likelihood is unbounded: a shape below 1 makes
  the density infinite at a support end. So a maximum-likelihood fit could
  run an end onto the smallest or largest observation and stop wherever
  its search gave up. The answer depended on the data's units: shapes
  1.00, 1.19 on a fixture, and 0.18, 0.18 on the same data times 7.3.
  Such a fit now warns "No finite maximum" and recommends
  ``how="MPS"``. Maximum product of spacings scores an end gap of zero as
  minus infinity, so its estimates are finite and the same in any units.
  MLE stays the default; the class docstring explains when to prefer MPS.
- **Changed: Bernoulli's survival function is P(X > x) (#344).** It was
  ``P(X >= x)``, so ``sf`` was [1, p] at the outcomes 0 and 1 and ``ff``
  was ``P(X < x)``, which never reaches 1, so ``qf`` could not invert it.
  ``sf`` is now [p, 0] and ``ff`` [1 - p, 1], as for ``Binomial`` with
  ``n = 1``, ``scipy.stats.bernoulli`` and every other discrete
  distribution in the package. ``Hf`` is [-log p, inf]; ``df``, ``hf``,
  ``qf``, ``mean``, ``random`` and the fitted ``p`` are unchanged. Code
  that read the probability of the ``1`` outcome (a one-shot device
  working on demand) as ``sf(1)`` wants ``sf(0)``, or ``p`` itself.
- **The Uniform's MLE refuses censored data again (#460).** 0.21.0 fitted
  right- and left-censored data by maximum likelihood. The estimates were
  right, but they sit on a wall of the likelihood (the smallest or largest
  observation), where its curvature says nothing about their uncertainty.
  The covariance it reported was not positive definite, so Wald bounds
  were NaN or silently several times too wide: an ``sf`` bound of [0.25,
  0.98] where the likelihood-ratio bound is [0.72, 0.85]. ``Uniform.fit``
  now raises ``ValueError`` on any censored value, as it always did for
  interval-censored ones, and names the methods that take censored data
  (``how="MPS"``, ``"MPP"``, ``"MSE"``). Exactly observed data, truncated
  or not, fit as before, still with no covariance.
- **Fits with no finite maximum warn instead of returning silently
  (#392).** Some data leave the likelihood with no maximum: a covariate
  level with no events, perfectly dependent pairs, a mixture component on
  a point mass, noise-free degradation readings. These fits returned
  wherever their optimiser stopped, silently:

  - a WeibullPH coefficient of -14.7 (+32 to +36 for the PO models, about
    -32 for most frailty models);
  - Fine-Gray -12.9, with BFGS reporting success;
  - copula dependence of Clayton theta 3.2e6, Frank 1.2e7, Gumbel 105.5
    (log-likelihood inf) and Gaussian rho at its 0.9999 cap;
  - a mixture component with beta 9100;
  - GammaProcess alpha at the end of its search range (1e6), and
    DestructiveDegradation sigma 9.9e-16;
  - BetaGeometric in its Geometric limit (a, b about 1e5, 3.5e5).

  Each now gives one ``UserWarning`` starting "No finite maximum" at the
  caller's line. It names the parameter that runs away and what to do
  instead (for example "use Geometric"), and the fit still returns the
  model it reached. Fine-Gray and ``CompetingRisksProportionalHazards``
  (Fine-Gray) use CoxPH's "Monotone partial likelihood" warning.

  The regression test is Newton's. Along each coefficient's profile,
  Kantorovich's ``h = |f'''| |f'| / f''^2`` is 1 on the way to a supremum,
  however far the optimiser went. At the maximum of an ordinary fit it is
  at most 2e-4, over the test registry and 1360 calibration refits. It
  costs one Hessian at the fitted values, and a coefficient's profile is
  read only when its Newton step there exceeds 1/709.8 of its value (the
  log of the largest double): every coefficient running off to infinity
  exceeds it and a converged one does not. That Hessian is kept:
  ``covariance()``, ``standard_errors()``, ``cb()`` and ``param_cb()`` of
  the PH, AFT, PO, AH, frailty and Fine-Gray fits invert it instead of
  computing a numerical Hessian on every call, so a PH fit followed by
  its standard errors and a band is 8-42% faster than before, and the
  standard errors move by at most 5e-4 relative (rounding in the
  numerical Hessian). The numerical Hessian is still used for reloaded
  models, accelerated-life and AFT ``fit_tvc`` fits, fits with no finite
  maximum, and Hessians that are not positive definite. The
  additive-hazards models, whose likelihood rises without bound on such
  data, now say so instead of reporting a positivity boundary (all except
  GammaAH). The Gumbel copula no longer leaks about 230 raw numpy overflow
  warnings. Univariate MLE now also refuses a point mass at the edge of a
  truncation window (Weibull beta 455.6, Normal sigma 0.037, silently).
- **Parametric additive hazards warn where their hazard is negative
  (#376).** ``h_0(x) + beta'Z`` has nothing keeping it positive away from
  the observed failures, so for a protective covariate row the cumulative
  hazard falls: ``sf`` exceeded 1 (1.03 for a ``WeibullAH``) and ``ff``
  and ``df`` went negative, silently. The values are still the model's,
  but ``sf``, ``ff``, ``df``, ``hf``, ``Hf``, ``cb``, ``sf_tvc`` and
  ``Hf_tvc`` now give one ``RuntimeWarning`` per call naming how many
  queried points have a negative hazard and how far ``sf`` exceeds 1. (The
  semi-parametric ``AdditiveHazards`` already predicts with the running
  maximum of its cumulative hazard, since #462.)
- **Additive hazards bounds no longer leak a numpy overflow warning
  (#465).** Where the fitted cumulative hazard is negative, the
  logit-scale ``sf`` bound computed ``1 / (1 + exp(-t))`` with ``t``
  hugely negative, and numpy warned "overflow encountered in exp" on the
  way to the right answer, 0. It now uses ``scipy.special.expit``; the
  bounds are unchanged.
- **Cox models no longer break on a covariate far from zero (#459).**
  ``CoxPH`` fitted on the raw covariates, so a column such as a year or a
  date overflowed ``exp(beta'Z)``: on 200 rows, adding 2000 to a N(0, 1)
  covariate moved beta from 0.860 to 0.768, made the survival and
  p-values NaN, gave a false "monotone partial likelihood" warning and
  leaked numpy overflow warnings. The fit now always runs on the
  covariates centred on their (n-weighted) means, as R's ``coxph``,
  lifelines and scikit-survival do; beta and every prediction are
  unchanged by any shift of a column, for plain, stratified and
  time-varying fits, residuals, ``check_ph``, robust errors and the
  cause-specific Cox model. The reported baseline (``h0``, ``H0``) stays
  at ``Z = 0`` (R's ``basehaz(fit, centered = FALSE)``), as before, and
  ``phi(Z)`` is ``exp(beta'Z)``; predictions combine the two on the log
  scale, so a tiny baseline and a huge multiplier lose nothing. Where the
  baseline at ``Z = 0`` cannot be represented (it over- or underflows),
  the fit raises ``ValueError`` and says to pass the new ``center=True``
  (on ``fit``, ``fit_from_df`` and the ``fit_tvc`` variants), which keeps
  the baseline at the means, ``model.center`` (R's ``basehaz(fit)``), with
  ``phi(Z) = exp(beta'(Z - center))``, and saves ``"center"`` (schema 2).
  A default fit's file has the layout of before (schema 1), and files
  saved by 0.21.0 and earlier load as fitted.
- **Parametric regressions and Fine-Gray no longer break on a covariate
  far from zero (#463).** ``exp(beta'Z)`` overflowed in these fits too:
  adding 2000 to a N(0, 1) covariate turned a ``WeibullPH`` coefficient of
  0.707 into 0.0247 with a scale of 2.4e18, silently, and ``FineGray.fit``
  and ``CompetingRisksProportionalHazards(model="Fine-Gray")`` raised
  ``LinAlgError: SVD did not converge``. ``FineGray`` and the competing-
  risks model now centre as Cox does, and take ``center=`` with the same
  meaning. The parametric families (``PH``, ``AFT``, ``PO``, ``AH`` and
  their ``fit_from_df`` / ``fit_tvc`` variants) take ``center=False``: by
  default the baseline is reported at ``Z = 0``, as before. Where the
  family and link have an exact map between the covariate means and
  ``Z = 0`` (a Weibull, Rayleigh, Exponential or Gumbel PH; every AFT
  baseline SurPyval has; a LogLogistic or Logistic PO) the fit runs on
  centred covariates and maps back, so beta, the predictions and the
  bounds are the same whatever the covariates' origin, and it raises
  ``ValueError``, pointing to ``center=True``, when the baseline at 0 over-
  or underflows. For the other pairs the default fit is at ``Z = 0``,
  unchanged and unchecked, so on covariates far from zero it can still
  fail or stop at a poor answer (a LogNormal PH coefficient of 0.006
  against 0.64 at a shift of 300). ``center=True`` fits
  any family with its baseline at the means (``model.center``, shown in the
  summary and saved, schema 2), where a shift of a column changes nothing.
  For additive hazards, ``center=True`` is a different model,
  ``h0 + beta'(Z - center)``.
  On ordinary data the reported parameters are those of before to
  optimiser tolerance (log-likelihoods within 2e-8 on 48 of 49 test fits;
  one Logistic PO fit on 34 tires stops 1.3e-5 nats short of the old
  point), and the fits are no slower.
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
- **Coefficients the data cannot separate are NaN, as in R (#476, #409).**
  A covariate column that adds nothing to the others -- a constant column
  where the baseline already has a scale, a duplicated column, dummy
  columns that add up to another column -- has no estimate of its own:
  shifting weight between it and the columns it repeats fits the data
  equally well. The fit used to return whichever split its search stopped
  at. A constant column moved WeibullAFT's ``alpha`` from 54.06 to 51.02,
  a repeated column split a Cox effect into 0.148 and -0.264, and
  a constant column on separated data got 3.1e14 with an all-NaN baseline.
  Now the fit leaves such columns out, as R's ``coxph`` and ``survreg``
  do: their coefficients are ``nan``, with ``nan`` standard error and
  p-value, ``model.aliased`` lists them, and one warning names them. The
  other coefficients and the predictions are those of the fit without the
  column. This covers ``CoxPH`` (with strata and time-varying
  covariates), the parametric proportional hazards, AFT, proportional odds
  and additive hazards models and their ``fit_tvc`` (the AFT one split a
  repeated column's 0.338 into 1.685 and -1.346), Fine-Gray, the
  competing-risks Cox model, the frailty models, and the Lin-Ying
  ``AdditiveHazards`` and ``BuckleyJames`` models. **Changed:** those two
  raised ``ValueError`` for a constant column, a single observation or
  collinear columns; they now fit the other columns and warn. Fits without
  such a column are unchanged. The proportional-intensity recurrent
  regressions (#502) and the dual-stress life models (#503) do not alias
  yet. ``conformance/test_aliasing.py`` checks every registered model
  (principle 12).
- **Regression models print a coefficient table: summary() (#484).** The
  Cox and parametric regression models' ``summary()`` returns a
  ``DataFrame`` in the layout of R's ``summary(coxph)`` and lifelines:
  ``coef``, ``exp(coef)``, ``se(coef)``, Wald intervals for both, ``z``
  and ``p``, one row per covariate, named from the ``DataFrame`` columns
  for ``fit_from_df``. On the Rossi data it matches R (``fin``: coef
  -0.37942, se 0.19138, p 0.04742). The parametric models add the
  baseline's parameters, with the intervals of ``param_cb``, so the
  Weibull shape ``beta`` is no longer printed among the coefficients
  ``beta_0``, ``beta_1``, .... The models' ``repr`` prints the table.
- **Changed: FrailtyModel.summary() returns the table (#484).** It
  returned the text its ``repr`` prints; it now returns the same
  ``DataFrame`` as the parametric models, with a ``frailty`` row for
  ``theta``. ``print(model)`` gives the text.
- **A cause label 0 with no censoring warns (#486).** lifelines,
  scikit-survival and R's ``cmprsk`` code a censored row as cause 0;
  SurPyval codes it as a missing cause (``None`` or ``NaN``), or with
  ``c``. Data coded the other way fitted without complaint as a model with
  an extra cause "0" and no censoring. Numeric cause labels that include
  0, with none missing and no ``c``, now warn and give the one-line
  conversion; passing ``c`` says 0 really is a cause and silences it.
- **Covariate rows and times that cannot be paired raise (#488).** A
  regression model pairs row ``i`` of ``Z`` with time ``x[i]`` (one row
  for every time, or one time for every row). Any other count was a raw
  numpy broadcast error; it is now a ``ValueError`` saying so. For a
  survival curve per subject, the Cox and parametric regression
  functions take ``grid=True``, which gives every row at every time,
  shape ``(len(Z),) + x.shape``, as the survival tree and forest do.
- **Changed: load_rossi_static()'s arrest means an arrest (#479).** It
  stored the censoring flag (1 = not arrested) as a float under the name
  ``arrest``, so ``c = 1 - df["arrest"]``, the natural call coming from R
  or lifelines, fitted the complement without complaint (a Cox
  coefficient of -0.001 for age instead of -0.057). ``arrest`` is now an
  integer, 1 for an arrest, as in R's ``carData::Rossi`` and lifelines.
  Code that passed ``c=df["arrest"]`` must pass ``c=1 - df["arrest"]``.
  The bundled rossi, heart and lung data also lose their saved row-index
  columns (``Unnamed: 0``).
- **Changed: trend tests name a trend only when it is significant (#481).**
  ``laplace`` and ``mil_hdbk_189c`` printed "Suggested trend: increasing"
  at p = 0.25. ``trend`` is now the conclusion at a keyword-only
  ``alpha_ci=0.05`` (``"none"`` unless p < ``alpha_ci``) and the new
  ``direction`` is the sign of the statistic. The models' ``trend_test()``
  take ``alpha_ci``, and the tests take ``c=`` by keyword only.
- **MixtureModel.fit returns the model (#482).** It returned ``None``.
  ``MixtureModel.fit(x, dist=Weibull, m=2)`` also builds and fits in one
  call. An unfitted model's ``repr`` names it, and a truncated fit reports
  "Fitted by: MLE", which it is, not "EM".
- **params and param_names on FrailtyModel and RenewalModel (#483).**
  Frailty: the baseline, the coefficients, then ``theta``. Renewal: ``q``
  or ``rho``, then the distribution's parameters, the order of
  ``standard_errors``.
- **AcceleratedLife reports its life parameter as the life model's
  (#489).** It printed "alpha: 1.0", a placeholder, as if fitted; it now
  prints ``alpha: L(Z) of the Power life model``, ``param_cb`` refuses
  that parameter, and the model has ``life_parameter``. Every parametric
  regression model has ``param_names``. Data at one stress level raise
  "needs at least two distinct stress levels", and too few levels with
  ``init`` warn that the life-stress relationship cannot be identified.
- **The repair models say which kind of dist they take (#495).** A
  lifetime distribution passed to ``ARI`` (an ``AttributeError``) and an
  intensity model passed to ``ARA``, ``GeneralizedRenewal`` or
  ``GeneralizedOneRenewal`` ("more than one right censored time") raise a
  ``ValueError`` naming the right fitter.
- **API papercuts (#485).** ``to_json()`` without a path returns the
  JSON text, and ``from_json`` reads it. A model with no data (from
  ``from_params``, or loaded) plots its CDF. ``how`` is case-insensitive.
  ``x``, ``c`` and ``n`` given as (n, 1) columns are read as one value per
  row. **Changed:** ``qf`` outside [0, 1] is ``nan`` (it was ``inf`` or
  0), a continuous distribution's ``qf(0)`` is the start of its support (a
  Normal's ``-inf``, not 0), and a bounded one's ``qf(1)`` its end
  (``Uniform(2, 5)``: 5, not ``inf``). Kaplan-Meier, Nelson-Aalen and
  Fleming-Harrington given left- or interval-censored or right-truncated
  data point to Turnbull. ``logrank`` warns when most groups have one
  member. ``fit_best`` passes over candidates whose support excludes the
  data quietly and reports other failures once. The MPP heuristic error
  lists the valid names, and ``sp.CrowAMSAA`` and the like say which
  subpackage to import from. ``CoxPH.fit_tvc_from_df`` and
  ``fit_tvc_timeline_from_df`` take ``formula=``, and non-numeric
  ``Z_cols`` suggest it. ``CompetingRisks.plot()`` is new. Signatures
  print readably: ``Weibull.fit``'s is 593 characters, was 3,218.
- **Changed / deprecated: one name for parameter names, parameter_names
  (principle 21).** A model's parameter names were spelt three ways:
  ``param_names``, the regression models' ``parameter_names()`` method
  and the recurrent models' ``parameter_names`` property (which a model
  built with ``from_params`` refused). Every distribution and every model
  with ``params`` now has ``parameter_names``, a list naming ``params``
  entry by entry (``Weibull.fit(x).parameter_names`` is ``['alpha',
  'beta']``, ``WeibullPH``'s ``['alpha', 'beta', 'beta_0']``), including
  models that had none (``CoxPH``, ``AdditiveHazards``, ``BuckleyJames``,
  ``RoystonParmar``, ``MixtureModel``, ``CopulaModel``). A
  ``ProportionalIntensityModel`` names ``params`` then ``coeffs``, the
  order of its ``covariance``. Until v0.23 the old spellings work with a
  ``DeprecationWarning``: the ``param_names`` attribute, calling
  ``parameter_names()``, the ``param_names=`` keyword of
  ``CustomDistribution``, and a ``param_names`` class attribute on your
  own ``PathModel``, ``Copula`` or ``CountingProcess`` subclass. ``params``
  is unchanged, and saved files keep the key ``"param_names"``, so they
  move both ways between 0.21 and 0.22.
- **Fits record whether they reached a maximum.** A parametric model has
  ``maximum``: ``"verified"`` (zero gradient and a positive-definite
  Hessian, or an exact estimator), ``"unverified"``, ``"no finite
  maximum"``, ``"not applicable"`` (not a maximum-likelihood fit, or
  ``from_params``) or ``"unknown"`` (loaded from an older save). It
  matches the fit's warnings and is saved by ``to_dict``. ``fit_best``
  sets candidates aside by it rather than by the text of their warnings
  (#492).
- **Changed: two-parameter fits no longer hide an unverified maximum.** A
  maximum-likelihood fit that did not reach a verified maximum warned
  only for families with more than two parameters: the exemption meant
  for the Uniform, whose support ends are parameters, matched every
  two-parameter family (Weibull, Gamma, LogNormal, ...). Such fits now
  warn.
- **formula= for the parametric time-varying fits.** ``fit_tvc_from_df``
  (PH, AH, PO, AFT) and ``fit_tvc_timeline_from_df`` (PH, AH, PO) take a
  formula instead of ``Z_cols``, as ``CoxPH``'s do (#485): categorical
  columns are coded, the model predicts from a ``DataFrame`` with the
  same coding, and an aliased column is named. ``Z_cols`` now defaults to
  ``None``.
- **The recurrent-event, competing-risks and degradation models are
  importable from surpyval.** ``sp.CrowAMSAA``, ``sp.ARA``,
  ``sp.FineGray``, ``sp.CompetingRisks``, ``sp.DegradationAnalysis`` and
  the other model classes are now at the top level, as the regression
  models already were, and stay in their packages too. Helper functions
  and result types (``laplace``, ``mil_hdbk_189c``,
  ``TrendTestResult``) and the generically named copulas (``Gaussian``,
  ``Frank``, ...) stay in their packages; asking for one at the top level
  says where it is.
- **The bundled Claude Code skill matches 0.22, and its code is tested
  (#491).** Every code block in it runs as a test.
- **Offset fits with no maximum warn (#487).** With a shape below 1 the
  density is infinite at the offset, so the likelihood grows without
  bound as ``gamma`` runs onto the first failure. Such a fit returned a
  degenerate model silently or behind "MLE Failed": the 3-parameter
  Weibull on [55, ..., 140] reached a shape of 0.09. Every offset family
  now warns "No finite maximum" and recommends ``how="MPS"``, and returns
  the point its search reached. The Exponential's genuine maximum at the
  first failure is unchanged.
- **Changed: durations and dates are refused (#480).** ``timedelta64``
  and ``datetime64`` input was fitted in its storage ticks (a scale of
  5.0e5 for durations of days held in seconds, 6.9e14 in nanoseconds),
  and predictions raised numpy's ``TypeError``. SurPyval has no time
  unit, so such values now raise a ``ValueError`` wherever a time is
  accepted, with the conversion to use (``x / pd.Timedelta(days=1)``).
- **Changed: suspensions are not drawn as points (#478).** A probability
  plot drew each suspension at the ``F`` of the failure before it, where
  it looked like one more failure. ``plot()`` now draws the failures
  only, as in Abernethy's *New Weibull Handbook* and Weibull++, and
  ``plot(show_censored=True)`` (also on ``MixtureModel.plot``) marks the
  suspension times with ticks on the time axis, which still spans every
  time. ``get_plot_data()`` keeps its meaning: ``x_`` and ``F`` hold
  every row as before, and a new boolean ``failed`` mask selects the
  failures that are drawn; ``x_censored`` holds the suspension times. The
  non-parametric ``get_plot_data`` returns ``failed`` too.
- **weibayes (#493).** ``surpyval.weibayes(x, c, n, beta)`` gives the
  Weibayes lower confidence bound on a Weibull's scale of known shape
  from few or no failures (Nelson 1985; Abernethy), as a Weibull model:
  ten units run 500 hours with no failure and a shape of 2 give a 95%
  lower bound on the scale of 913.5 hours. ``fit`` still refuses data
  with no failures, and now points to it.
- **Changed: fit_best ranks regular maxima only (#492).** AIC and BIC
  assume a regular maximum. The Uniform and Beta4, whose support ends are
  parameters, are no longer default candidates (``include=`` still
  tries them), and a fit with no finite maximum or an unverified one is
  ranked only when no regular candidate fits, with one warning naming it.
  The Beta4 no longer "wins" on [1, ..., 7], nor the Uniform on 50
  Weibull draws.
- **Changed: Wald bands are monotone (#477).** Wald bounds on ``sf``,
  ``ff`` and ``Hf`` were formed on the logit of ``sf`` for every family,
  and on small samples turned back: the issue's lower bound on ``F`` fell
  from 0.39 to 0.00004 as time went on. They are now formed on each
  family's probability-plot scale (``log(-log S)`` for the Weibull, the
  normal quantile for the Normal and LogNormal), where they are monotone
  whenever the shape's own interval excludes 0; that bound is now 0.83.
  Large samples are unchanged to 2e-4. The degradation models' two-stage
  band is formed on the same scale, so it still contains the life model's
  own. ``plot`` and ``get_plot_data`` take ``method=`` to draw the
  likelihood-ratio band.
- **quantile_cb and mean_cb (#494).** Parametric models give confidence
  bounds on a quantile (a B-life) and on the mean, by Wald (matching R's
  ``survreg`` to 1e-7) or likelihood ratio (``method="lr"``, better on
  small samples), with ``bound=`` and ``alpha_ci`` as on ``cb``.
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