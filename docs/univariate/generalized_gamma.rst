Generalised Gamma Distribution
==============================

**In plain words.** The generalised gamma is one family that contains
three of the most used lifetime distributions: the Weibull, the
lognormal and the gamma (and with them the exponential). Its third
parameter, ``Q``, says which of them the data look most like, so one fit
answers "Weibull or lognormal?" with a number and a standard error rather
than a choice between two separate fits. Its hazard can rise, fall, rise
then fall, or (for some ``Q``) fall then rise, which makes it a common
default in medical and economic survival analysis, where it is the
standard flexible parametric model (Cox, Chu, Schneider and Muñoz, 2007).

``GeneralizedGamma`` is an instance of the class below: call its ``fit``
to estimate the parameters from data (exact, censored and truncated, with
an offset if wanted), or ``from_params`` for known parameters; both
return a :doc:`Parametric <parametric_class>` model.

Parameters
----------

The parameterisation is Prentice's (1974), the one R's ``flexsurv`` calls
``gengamma``, so results compare one to one with ``flexsurvreg``:

- ``mu``, any real number: the location of the log time;
- ``sigma``, positive: the scale of the log time;
- ``Q``, any real number: the shape.

With the standardised log time :math:`w = (\ln x - \mu)/\sigma` and
:math:`q = Q^{-2}`, a gamma variable :math:`G` of shape :math:`q` gives
the log time as :math:`w = \ln(G/q)/Q`, so

.. math::

    S(x) = \begin{cases}
    1 - P\left(q,\, q e^{Q w}\right) & Q > 0, \\
    1 - \Phi(w) & Q = 0, \\
    P\left(q,\, q e^{Q w}\right) & Q < 0,
    \end{cases}
    \qquad
    f(x) = \frac{|Q|\, q^{q}}{\sigma x\, \Gamma(q)}
    \exp\left\{q\left(Q w - e^{Q w}\right)\right\},

where :math:`P` is the regularised lower incomplete gamma function and
:math:`\Phi` the standard normal distribution function.

Special cases
-------------

Each is the generalised gamma with one or two parameters fixed, so a
likelihood-ratio test of it is a fit with ``fixed=`` (one degree of
freedom per fixed parameter):

=======================  =====================================  ===============================
Fixed                    Is                                     In SurPyval
=======================  =====================================  ===============================
``Q = 0``                lognormal, ``mu``, ``sigma``            ``LogNormal(mu, sigma)``
``Q = 1``                Weibull, scale :math:`e^{\mu}`,         ``Weibull(exp(mu), 1/sigma)``
                         shape :math:`1/\sigma`
``Q = sigma``            gamma, shape :math:`1/\sigma^2`,        ``Gamma(1/sigma**2,``
                         rate :math:`e^{-\mu}/\sigma^{2}`        ``exp(-mu)/sigma**2)``
``Q = sigma = 1``        exponential, rate :math:`e^{-\mu}`      ``Exponential(exp(-mu))``
=======================  =====================================  ===============================

The gamma case ties ``Q`` to ``sigma`` rather than fixing it, so it is
tested by comparing with ``Gamma.fit`` on the same data (the
log-likelihoods are the same function there).

.. code:: python

    import surpyval as surv
    from scipy import stats

    gg = surv.GeneralizedGamma.fit(x, c=c)
    weibull = surv.GeneralizedGamma.fit(x, c=c, fixed={"Q": 1})
    lr = 2 * (weibull.neg_ll() - gg.neg_ll())
    p = stats.chi2(1).sf(lr)   # small p: the data are not Weibull

How it is computed
------------------

At ``Q = 0`` the gamma shape :math:`q` is infinite, and near it the
incomplete gamma functions are asked for a shape of :math:`10^{8}` at a
point within a fraction of a percent of it, where the usual algorithms
lose digits (scipy's log cdf was -15.49 where it is -15.06 at
:math:`Q = 10^{-4}`, :math:`w = -5`), and a formula with :math:`q \ln q`
and :math:`\ln\Gamma(q)` in it cancels (R's ``dgengamma`` loses 4e-7 of
the density at :math:`Q = 10^{-4}`). SurPyval instead:

- writes the log density as :math:`-\ln\sqrt{2\pi} -
  \operatorname{stirlerr}(q) - w^2 h(Qw) - \ln(\sigma x)`, with
  :math:`h(y) = (e^y - 1 - y)/y^2` and Stirling's remainder as a series in
  :math:`Q^2`, so nothing cancels and it is exactly the lognormal's at
  ``Q = 0``;
- takes the survival and distribution functions, for :math:`|Q| < 0.1`,
  from Temme's uniform expansion of the incomplete gamma function written
  in :math:`(w, Q)`, which is smooth through ``Q = 0`` (and is the normal
  there); elsewhere from the incomplete gamma functions, with the log of
  each tail computed directly so neither rounds to 0 or 1;
- has derivatives in every parameter, including ``Q`` at and across 0, so
  the fit, its standard errors and every bound work wherever the
  lognormal is the answer.

Against R's ``flexsurv`` 2.3.2 the density, distribution function, log
survival function and quantiles agree to 1e-10 or better at parameters
with both signs of ``Q``, and the fits to survival's ``lung`` data (right
censored, and left truncated) agree in log-likelihood to 1e-6 and in the
estimates to a hundredth of a standard error
(``surpyval/tests/reference/test_generalized_gamma.py``).

Moments
-------

:math:`E[X^m] = e^{m\mu}\, q^{-m\sigma/Q}\, \Gamma(q + m\sigma/Q) /
\Gamma(q)` (:math:`e^{m\mu + m^2\sigma^2/2}` at ``Q = 0``). For a
negative ``Q`` the upper tail is heavy: the ``m``-th moment is infinite
once :math:`m\sigma|Q| \geq 1`, and ``mean`` and ``moment`` say ``inf``.

What it does not do
-------------------

- **Probability-plot fitting** (``how="MPP"``) is refused: the straight-line
  axis is the quantile of the standardised log time, which depends on
  ``Q``, the parameter being estimated (as for the gamma). ``plot()``
  draws on the fitted ``Q``, where the axis is the right one; fit by
  ``"MLE"`` (the default), ``"MPS"``, ``"MSE"`` or ``"MOM"``.
- **Covariates** go on ``mu`` through ``AFT(GeneralizedGamma)`` (as
  flexsurv's ``gengamma`` with covariates); the proportional hazards and
  odds families take it as any baseline, but are not closed-form
  reparameterisations of it, so they fit at the covariates as given (see
  :doc:`../regression analysis`).

.. autoclass:: surpyval.univariate.parametric.distributions.generalized_gamma.GeneralizedGamma_
   :members:
   :inherited-members:
