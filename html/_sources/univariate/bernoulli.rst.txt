Bernoulli Distribution
======================

A single weighted coin flip: for pass/fail demands with no time
dimension.

.. _probability-uncertainty:

How sure is the estimate of p?
------------------------------

A fitted ``p`` is a proportion: 3 failures in 1200 demands gives
``p = 0.0025``. With few events that number is rough, and code that uses
it, such as a probability-on-demand component in a system model, needs
to know how rough. A fitted ``Bernoulli``, ``FixedEventProbability`` or
``Binomial`` says so in two ways.

**A confidence interval,** ``param_cb("p")``. Read a 90% interval as: had
the demands been repeated many times, intervals built this way would
contain the true ``p`` at least about 90% of the time. Three methods
are built from the counts:

- ``"exact"`` (the default), the Clopper-Pearson interval. It holds *at
  least* its level for every ``p`` and every number of demands, so it is
  the safe one to report; the price is that it is a little wide. With no
  failures its upper bound is the zero-failure (success-run) bound.
- ``"jeffreys"``, the Jeffreys interval: narrower, with coverage close
  to the level on average, but below it for some ``p`` (see the study
  below). Use it when a typical rather than a guaranteed level is wanted.
- ``"wald"`` (on the logit scale) and ``"lr"`` (likelihood ratio), as for
  the other models.

**A variance,** ``covariance()``, for code that carries the uncertainty
further by the delta method or a normal approximation. It is the
textbook ``p (1 - p) / N``.

Why not just use ``p`` plus or minus two standard errors? Near 0 or 1,
where reliability data usually are, the normal approximation fails: the
interval runs below 0, and with no failures the standard error is 0,
which would say ``p`` is known exactly. That is why the variance is
``nan`` (undefined) when there are no events, or only events, and why
the interval to quote is the exact or the Jeffreys one.

The formulas
~~~~~~~~~~~~

With :math:`k` events in :math:`N` trials, :math:`\hat p = k / N` and,
for a two-sided interval at level :math:`1 - \alpha` (one-sided bounds
use :math:`\alpha` in place of :math:`\alpha / 2`):

- Clopper-Pearson: :math:`[B_{\alpha/2}(k, N - k + 1),\;
  B_{1-\alpha/2}(k + 1, N - k)]`, with :math:`B_q(a, b)` the
  :math:`q` quantile of a Beta(:math:`a, b`) distribution; the lower
  bound is 0 when :math:`k = 0` and the upper 1 when :math:`k = N`.
- Jeffreys: :math:`[B_{\alpha/2}(k + \tfrac12, N - k + \tfrac12),\;
  B_{1-\alpha/2}(k + \tfrac12, N - k + \tfrac12)]`, the equal-tailed
  interval of the posterior under Jeffreys' prior Beta(1/2, 1/2), with
  the same convention at :math:`k = 0` and :math:`k = N` (Brown, Cai and
  DasGupta, 2001). R's ``binom::binom.confint(..., methods = "bayes")``
  moves the whole of :math:`\alpha` into the open tail at those two
  counts; SurPyval keeps :math:`\alpha / 2` there, as Clopper-Pearson
  does.
- The variance: :math:`\widehat{\mathrm{var}}(\hat p) = \hat p (1 -
  \hat p) / N`, the inverse of the observed information
  :math:`N / (\hat p (1 - \hat p))`; ``nan`` at :math:`k = 0` or
  :math:`k = N`.

For draws of ``p`` itself (a Monte Carlo propagation), fit the Bayesian
model with Jeffreys' prior,
``Bernoulli.fit_bayes(x, priors={"p": Beta.from_params([0.5, 0.5])})``,
whose posterior is Beta(:math:`k + \tfrac12, N - k + \tfrac12`) at
every :math:`k`, zero failures included.

.. code:: python

    import surpyval as sp

    # 3 failures (coded 1) in 1200 demands
    model = sp.Bernoulli.fit([1, 0], n=[3, 1197])
    model.param_cb("p", alpha_ci=0.1)                     # [0.00068, 0.00645]
    model.param_cb("p", alpha_ci=0.1, method="jeffreys")  # [0.00090, 0.00585]
    model.covariance()                                    # [[2.08e-06]]
    model.standard_errors()                               # [0.00144]

    # no failures in 1200 demands
    clean = sp.Bernoulli.fit([0], n=[1200])
    clean.param_cb("p", alpha_ci=0.1, bound="upper")      # [0.00192]
    clean.covariance()                                    # [[nan]]

How well each interval holds its level
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The share of 95% intervals that contain the true ``p``
(``scripts/studies/probability_coverage.py``, 4000 replicates per row,
fixed seeds; Monte Carlo standard error 0.001 to 0.008; the exact
coverage, summed over the binomial outcomes, agrees within it):

=====  ===  =====  ========  ========================
p      N    exact  Jeffreys  p ± 1.96 standard errors
=====  ===  =====  ========  ========================
0.01   10   0.995  0.904     0.095
0.01   50   0.987  0.987     0.395
0.05   10   0.990  0.990     0.407
0.05   50   0.991  0.883     0.915
0.3    10   0.990  0.924     0.843
0.3    50   0.968  0.955     0.930
=====  ===  =====  ========  ========================

The exact interval never falls below 95%. The Jeffreys interval is
narrower and averages near 95%, but drops to about 88-90% at some ``p``
(one failure in a small sample puts its lower bound just above a small
true ``p``). The interval built from the variance alone is far below its
level for small ``p``, which is why it is not offered as a bound.

.. autoclass:: surpyval.univariate.parametric.distributions.bernoulli.Bernoulli_
   :members:
   :inherited-members:
