Rank-based AFT (Gehan and log-rank)
===================================

A second semi-parametric accelerated-failure-time estimator, beside
Buckley-James. Both fit the same model: each covariate stretches or
shrinks the time to failure by a fixed factor, and the shape of the
failure-time distribution is left to the data. They differ in how they
find the coefficients.

**The idea in words.** Once the coefficients are right, the "residual"
log times (each unit's log time with its covariates' effect taken off)
all come from one distribution, so no covariate should help to say which
unit fails first. The rank estimator compares each failure with the units
still at risk at its residual, exactly as a log-rank test compares
groups, and chooses the coefficients at which the covariates of the units
that fail look, on average, like those of the units at risk. It uses only
the *order* of the residuals, so a few extreme times do not drag the fit,
and censoring is handled as the log-rank test handles it. Unlike
Buckley-James it has standard errors without resampling.

Two weightings are offered:

- ``rank_weights="gehan"`` (the default) weights each failure by the
  number at risk. The estimating function is then the gradient of a
  convex loss, so every solution is a global one and it is found
  reliably.
- ``rank_weights="logrank"`` gives each failure the same weight, which is
  more efficient when the errors have an extreme-value (Weibull-like)
  shape. Its estimating function is not monotone and has several exact
  roots, within a small fraction of a standard error of each other (up
  to 0.06 of one in the tests). The estimate reported is the smoothed
  root reached by refitting weighted Gehan fits from the Gehan estimate
  (Jin, Lin, Wei and Ying 2003).

**Standard errors.** The estimating functions are step functions of the
coefficients, which have no derivative. Induced smoothing (Brown and Wang
2005, 2007) replaces each "is this unit's residual at least that one's"
indicator by a normal probability whose spread is the estimate's own
standard error, which gives a smooth function with a derivative. The
standard errors are the sandwich of that derivative and the variance of
the unsmoothed function's per-unit contributions; ``summary()``,
``covariance()``, ``standard_errors()`` and ``param_cb()`` (Wald, or the
percentile bootstrap with ``method="bootstrap"``) report them. The
estimate reported is the root of the smoothed function, which differs
from the root of the step function (an exact linear programme for Gehan)
by a fraction of a standard error that shrinks as the data grow: on 100
units, a tenth of one or less for Gehan and up to about 0.3 for log-rank
(median about 0.15).

**The model and the formulas.** With :math:`Y_i = \log x_i`, event flag
:math:`\delta_i`, counts :math:`w_i` and residuals
:math:`e_i = Y_i - \gamma' Z_i`,

.. math::

    U_\phi(\gamma) = \sum_i w_i \delta_i\, \phi(e_i)
        \Big\{ Z_i - \frac{\sum_j w_j Z_j 1(e_j \ge e_i)}
                          {\sum_j w_j 1(e_j \ge e_i)} \Big\} = 0,

with :math:`\phi = 1` (log-rank) or :math:`\phi(e_i) = \sum_j w_j
1(e_j \ge e_i)` (Gehan). The Gehan function is the gradient of
:math:`\sum_{i,j} w_i w_j \delta_i \{e_j - e_i\}^+`. Smoothing replaces
:math:`1(e_j \ge e_i)` by :math:`\Phi\{(e_j - e_i)/r_{ij}\}` with
:math:`r_{ij}^2 = (Z_i - Z_j)'\Gamma(Z_i - Z_j)`, :math:`\Gamma` the
estimate's covariance (started at the inverse of the covariates'
covariance over the number of observations and updated until its standard
deviations settle to 1%). The full derivation, including the sandwich's
middle, is in the module docstring below.

The fitted ``beta`` uses the package's AFT sign: a positive coefficient
*shortens* life (it is the negative of the slope :math:`\gamma` of
:math:`\log x` on :math:`Z`). The intercept is not identified by ranks
(adding a constant to every residual leaves their order alone), so, as
for Buckley-James, a constant column is aliased and the location lives in
the Kaplan-Meier estimate of the residuals, which the predictions read:
:math:`S(t \mid Z) = \hat S_\varepsilon(\log t + \beta' Z)`, a step
function. There is no likelihood (no AIC, no likelihood-ratio bound) and
no bound on ``sf``; for those fit a parametric AFT model of the same data
(``WeibullAFT``, ``LogNormalAFT``, ...). Only observed and right-censored
data with positive times are supported.

**How it was checked.** No other package was used as a reference. The
Gehan estimate was checked against the exact linear programme
(``scipy.optimize.linprog``) on small data with censoring and tied
times, and the smoothed estimate shown to tend to it as the smoothing
shrinks, inside a bound proved from the loss; the log-rank estimate
against the same iteration run on exact linear programmes; the step
functions change sign across the reported estimates; with no censoring
and symmetric errors the Gehan estimate agrees in expectation with least
squares on the log times; on one large censored data set it agrees with
``BuckleyJames``; and a seeded simulation (Weibull lives, a quarter
censored, n = 200) checks the bias, the standard errors against the
estimates' spread and the coverage of the 95% intervals.

Usage::

    from surpyval import RankAFT
    model = RankAFT.fit(x, Z=Z, c=c)                       # Gehan
    model = RankAFT.fit(x, Z=Z, c=c, rank_weights="logrank")
    model.summary()

.. autoclass:: surpyval.univariate.regression.rank_aft.rank_aft.RankAFT_
    :members:

.. autoclass:: surpyval.univariate.regression.rank_aft.rank_aft.RankAFTModel
    :members:
