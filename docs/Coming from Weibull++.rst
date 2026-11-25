Coming from Weibull++ and other reliability tools
=================================================

This page is for reliability engineers who know Weibull++, Minitab's
reliability menus, or R's ``WeibullR`` (and its predecessor ``abrem``), and
want to do the same analyses in SurPyval. It maps the vocabulary, shows six
common analyses, and lists the settings that make two tools' answers differ.

The SurPyval code runs when this page is built. Its numbers were checked
against R (``survival`` 3.8.12 and ``WeibullR`` 1.2.4) and against closed
forms; hidden checks stop the build if they drift. The examples use the
Lieblein-Zelen ball bearing data, a classic public data set: 23 bearings
tested to failure, in millions of revolutions (Lieblein and Zelen, 1956,
*Journal of Research of the National Bureau of Standards*, 57, 273-316).

The vocabulary
--------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Weibull++ / Minitab / WeibullR
     - SurPyval
   * - failure, time to failure
     - a row with ``c = 0`` (the default), at ``x``
   * - suspension, survivor, right censored
     - ``c = 1``
   * - interval data, inspection data (failed between two inspections)
     - ``xl`` / ``xr``, or ``x`` as ``[left, right]`` with ``c = 2``
   * - left censored (failed before the first inspection)
     - ``c = -1``
   * - grouped data, "number in state", free-form data with quantities
     - ``n`` (a count per row)
   * - eta, scale; beta, shape (Weibull)
     - ``alpha``, ``beta`` (same meaning: :math:`R(t) = e^{-(t/\alpha)^\beta}`)
   * - gamma, location, failure-free life (three-parameter Weibull)
     - ``offset=True``, fitted as ``model.gamma``
   * - log-mean, log-std (lognormal)
     - ``mu``, ``sigma``
   * - lambda, failure rate (exponential); MTTF
     - ``failure_rate``; ``model.mean()``
   * - reliability, unreliability, failure rate at a time
     - ``model.sf(t)``, ``model.ff(t)``, ``model.hf(t)``
   * - conditional reliability (survive a mission, given an age)
     - ``model.cs(mission, age)``
   * - B-life, B10, time at a given unreliability
     - ``model.qf(0.1)``, bounds ``model.quantile_cb(0.1, ...)``
   * - MLE
     - ``how="MLE"`` (the default)
   * - rank regression on X (RRX, Minitab's "least squares, X on Y",
       WeibullR's ``"rr-xony"``)
     - ``how="MPP", rr="x"``
   * - rank regression on Y (RRY)
     - ``how="MPP", rr="y"``
   * - median ranks, Benard's approximation
     - ``heuristic="Benard"`` (with ``how="MPP"``)
   * - Kaplan-Meier, Johnson's adjusted ranks for suspensions
     - ``heuristic="Kaplan-Meier"``; the rank heuristics adjust ranks by
       Johnson's method when there are suspensions
   * - Fisher-matrix bounds
     - Wald bounds, ``model.cb(t)`` and ``quantile_cb`` (the default,
       ``method="wald"``)
   * - likelihood-ratio bounds
     - ``method="lr"``
   * - bootstrap, Monte Carlo bounds
     - ``method="bootstrap"``; for a rank regression, ``method="pivotal"``
   * - confidence level 90%, two-sided
     - ``alpha_ci=0.1`` (``bound="two-sided"``, the default)
   * - one-sided lower bound at 90%
     - ``alpha_ci=0.1, bound="lower"``
   * - distribution wizard, Minitab's distribution ID plot
     - ``sp.fit_best(x, c)`` (by AIC), then ``model.plot()``
   * - probability plot
     - ``model.plot()``
   * - mixed Weibull
     - ``sp.MixtureModel.fit(x, c, dist=sp.Weibull, m=2)``
   * - competing failure modes
     - ``surpyval.univariate.competing_risks`` (:doc:`Competing Risks
       SurPyval Modelling`)
   * - ALTA life-stress models (Arrhenius, Eyring, inverse power law,
       temperature-humidity, general log-linear)
     - ``sp.AcceleratedLife(sp.Weibull, sp.life_models.InversePower)`` and
       the other ``sp.life_models``
   * - RGA: Crow-AMSAA (NHPP), Duane
     - ``surpyval.recurrent.CrowAMSAA``, ``Duane``
   * - RGA: repairable systems, general renewal
     - ``surpyval.recurrent`` (MCF, ``GeneralizedRenewal``, ``ARA``,
       ``ARI``); see :doc:`Recurrent Event Modelling with SurPyval`
   * - degradation analysis
     - ``surpyval.degradation`` (:doc:`Degradation Modelling with SurPyval`)
   * - test design, demonstration tests
     - ``surpyval.test_planning`` (:doc:`Test Planning with SurPyval`)

A Weibull by maximum likelihood
-------------------------------

.. jupyter-execute::

    import numpy as np
    import surpyval as sp

    bearings = np.array([
        17.88, 28.92, 33.00, 41.52, 42.12, 45.60, 48.48, 51.84, 51.96, 54.12,
        55.56, 67.80, 68.64, 68.64, 68.88, 84.12, 93.12, 98.64, 105.12,
        105.84, 127.92, 128.04, 173.40,
    ])
    mle = sp.Weibull.fit(bearings)
    mle

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert np.allclose(mle.params, [81.87834, 2.102059], rtol=1e-6), mle.params

R's ``survreg`` gives :math:`\eta = 81.8783` and :math:`\beta = 2.10206`,
the same to six digits. ``WeibullR::MLEw2p`` reports 81.8485 and 2.10462:
its optimiser stops a little short of the maximum (its log-likelihood is
-113.69134, against -113.69129 here and in ``survreg``). Differences in
the fourth digit between tools are usually this, not a different method.

B10 life with bounds
--------------------

The B10 life is the time by which 10% have failed, ``qf(0.1)``. Its 90%
two-sided bounds, Fisher-matrix (Wald) and likelihood ratio:

.. jupyter-execute::

    print("B10:", mle.qf(0.1))
    print("Fisher matrix, 90% two-sided:", mle.quantile_cb(0.1, alpha_ci=0.1))
    print("Likelihood ratio, 90% two-sided:", mle.quantile_cb(0.1, alpha_ci=0.1, method="lr"))
    print("Fisher matrix, 90% lower:", mle.quantile_cb(0.1, alpha_ci=0.1, bound="lower"))

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert round(float(mle.qf(0.1)), 3) == 28.069, mle.qf(0.1)
    assert np.allclose(mle.quantile_cb(0.1, alpha_ci=0.1), [19.38699, 40.64024], atol=1e-4)
    assert np.allclose(mle.quantile_cb(0.1, alpha_ci=0.1, method="lr"), [18.0783, 38.6900], atol=1e-3)
    assert round(float(mle.quantile_cb(0.1, alpha_ci=0.1, bound="lower")), 2) == 21.04

The Fisher-matrix interval, 19.39 to 40.64 million revolutions, is the one
R gets from ``survreg`` with the delta method on log time (the same
construction as Weibull++'s Fisher-matrix bounds), to five digits.
``WeibullR`` gives 19.42 to 40.65 (Fisher matrix) and 18.09 to 38.68
(likelihood ratio), around its slightly different estimate. The bounds on
reliability work the same way: ``mle.cb(50, alpha_ci=0.1)`` gives 0.5519 to
0.8093 around :math:`R(50) = 0.7015`, the Fisher-matrix bounds on
:math:`u = \beta(\ln t - \ln\eta)`.

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert np.allclose(mle.cb(50, alpha_ci=0.1), [0.5519328, 0.8093122], atol=1e-6)

**Which bound to quote.** A two-sided 90% interval has 5% outside each
end; a one-sided 90% lower bound has 10% below it, so it is higher (21.04
here, against 19.39). A B-life is a "no worse than" claim, so it is often
quoted as a one-sided lower bound. Reliafy, the app built on SurPyval,
reports life bounds that way: 90% one-sided (lower on a life, upper on a
probability of failure). SurPyval's own default is 95% two-sided
(``alpha_ci=0.05``), so state ``alpha_ci`` and ``bound`` when you compare
with another tool.

Rank regression
---------------

.. jupyter-execute::

    rrx = sp.Weibull.fit(bearings, how="MPP", rr="x", heuristic="Benard")
    rry = sp.Weibull.fit(bearings, how="MPP", rr="y", heuristic="Benard")
    print("RRX:", rrx.params)
    print("RRY:", rry.params)

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert np.allclose(rrx.params, [80.97235, 2.247893], rtol=1e-5), rrx.params
    assert np.allclose(rry.params, [81.57757, 2.181229], rtol=1e-5), rry.params

These are the regressions on Benard's median ranks, :math:`(i - 0.3) / (n
+ 0.4)`. ``WeibullR::MRRw2p`` gives 80.947 and 2.2550: it regresses on the
*exact* median ranks (the median of each order statistic's beta
distribution) rather than Benard's approximation of them, and the two
differ by up to a few tenths of a percent for small samples. SurPyval has
no exact-median-rank heuristic; Benard's is what most tools mean by
"median ranks". With RRX the time is regressed on the rank (the
probability plot's line is fitted horizontally), and with RRY the other
way; the two give different answers whenever the points are not on a
straight line.

Rank regression comes from the probability plot; with censored data MLE
is usually the better estimate, and it is SurPyval's default. Some
practitioners prefer rank regression for very small complete samples,
because the MLE of :math:`\beta` is biased upwards there; ``mle.rba()``
gives the MLE with the reduced-bias adjustment of the shape instead (2.020
here, from 2.102). A rank regression's bounds are ``method="pivotal"``,
found by simulation.

Suspensions
-----------

Suppose the test had been stopped at 110 million revolutions, leaving five
bearings unfailed. They are suspensions, ``c = 1`` at 110:

.. jupyter-execute::

    c = (bearings > 110).astype(int)
    x = np.where(c == 1, 110.0, bearings)
    suspended = sp.Weibull.fit(x, c)
    print(suspended.params)

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert np.allclose(suspended.params, [79.71094, 2.288984], rtol=1e-5), suspended.params

``survreg`` gives 79.7109 and 2.28898 (``WeibullR``: 79.670 and 2.2909, the
same small shortfall as before). Five identical suspensions could equally
be one row with ``n = 5``.

Accelerated life: inverse power law
-----------------------------------

A voltage endurance test at 20, 30 and 40 V, twelve units each, stopped at
2,000 hours (simulated here). The inverse power law makes the Weibull's
scale :math:`L(V) = 1 / (a V^n)`, with a common shape.

.. jupyter-execute::

    rng = np.random.default_rng(7)
    volts = np.repeat([20.0, 30.0, 40.0], 12)
    life = 5e6 * volts ** -2.5 * rng.weibull(1.8, volts.size)
    t = np.minimum(life, 2000.0)
    ct = (life > 2000.0).astype(int)

    ipl = sp.AcceleratedLife(sp.Weibull, sp.life_models.InversePower).fit(t, volts, ct)
    print(dict(zip(ipl.parameter_names, ipl.params)))
    print("B10 at 10 V:", ipl.qf(0.1, [10.0]))

.. jupyter-execute::
    :hide-code:
    :hide-output:

    _p = dict(zip(ipl.parameter_names, ipl.params))
    assert round(_p["n"], 5) == 2.74600, _p
    assert round(_p["beta"], 5) == 1.84149, _p
    assert round(float(ipl.qf(0.1, [10.0])), 1) == 5767.8, ipl.qf(0.1, [10.0])

The inverse power law Weibull is a Weibull regression on :math:`\log V`,
so R fits it as ``survreg(Surv(t, 1 - ct) ~ log(volts), dist =
"weibull")``: the slope is :math:`-n = -2.74600`, ``1 / scale`` is
:math:`\beta = 1.84149`, and the B10 life at 10 V is 5,767.8 hours, the
same as here. The ``alpha`` printed as 1 is a placeholder: the life model
takes its place. Arrhenius (temperature in kelvin), Eyring, exponential,
dual (temperature-humidity) and general log-linear models are in
``sp.life_models``.

Reliability growth: Crow-AMSAA
------------------------------

Twelve failures of one system in a test stopped at 800 hours
(time-terminated):

.. jupyter-execute::

    from surpyval.recurrent import CrowAMSAA

    failures = np.array([4.3, 10.6, 30.5, 50.9, 63.2, 101.4, 141.0, 212.8,
                         301.6, 410.0, 538.5, 720.3])
    T = 800.0
    xg = np.append(failures, T)
    cg = np.append(np.zeros(12), 1)     # the end of the test, a censored row
    crow = CrowAMSAA.fit(x=xg, i=np.ones(13), c=cg)
    alpha, beta = crow.params
    print("beta:", beta, " lambda:", alpha ** -beta)
    print("instantaneous MTBF at 800 h:", crow.mtbf(T))

.. jupyter-execute::
    :hide-code:
    :hide-output:

    _b = 12 / np.sum(np.log(T / failures))
    assert np.isclose(beta, _b, rtol=1e-8), (beta, _b)
    assert np.isclose(alpha ** -beta, 12 / T ** _b, rtol=1e-6)
    assert np.isclose(crow.mtbf(T), T / (12 * _b), rtol=1e-6)

These are the closed-form maximum likelihood estimates of MIL-HDBK-189:
:math:`\hat\beta = n / \sum \ln(T / t_i) = 0.4728` and :math:`\hat\lambda =
n / T^{\hat\beta} = 0.5088`, and the instantaneous MTBF :math:`T / (n
\hat\beta) = 141.0` hours. SurPyval writes the expected number of failures
as :math:`(t/\alpha)^\beta` rather than :math:`\lambda t^\beta`, so
:math:`\lambda = \alpha^{-\beta}`. The handbook's bias-corrected
:math:`\bar\beta = (n - 1)\hat\beta / n` for a time-terminated test is not
applied; multiply by hand if your procedure calls for it.
``crow.mtbf_cb(T, alpha_ci=0.2, bound="lower", method="crow")`` gives
Crow's exact bound on the demonstrated MTBF.

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert round(beta, 4) == 0.4728 and round(alpha ** -beta, 4) == 0.5088
    assert round(float(crow.mtbf(T)), 1) == 141.0

Mixed Weibull and the three-parameter Weibull
---------------------------------------------

.. jupyter-execute::

    mixed = sp.MixtureModel.fit(bearings, dist=sp.Weibull, m=2)
    print("weights:", mixed.w)
    print("parameters:", mixed.params)

    w3 = sp.Weibull.fit(bearings, offset=True)
    print("3-parameter: gamma", w3.gamma, "eta, beta", w3.params)

A mixture is fitted by the EM algorithm; with 23 points and no sign of two
populations on the probability plot it is over-fitted here, and
``mixed.aic()`` against ``mle.aic()`` says so. The three-parameter MLE can
run off to an unbounded likelihood when :math:`\beta < 1`;
``how="MPS"`` (maximum product of spacings) is the robust choice for
offsets.

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert mixed.aic() > mle.aic(), (mixed.aic(), mle.aic())
    assert round(mle.rba().params[1], 3) == 2.020, mle.rba().params

Differences that catch people out
---------------------------------

- **Confidence level and sidedness.** Weibull++, Minitab and WeibullR each
  let you set the level, and a "90%" can be one- or two-sided. SurPyval
  takes ``alpha_ci`` (0.1 for 90%) and ``bound``; its default is 95%
  two-sided.
- **The estimation method.** Rank regression and MLE give different
  answers on the same data, and RRX and RRY differ too. Tools differ in
  their default; SurPyval's is MLE (``how="MLE"``). Match the method
  before comparing numbers.
- **Plotting positions.** Benard's approximation, exact median ranks,
  Kaplan-Meier and Nelson-Aalen give slightly different points, and so
  slightly different regression lines. SurPyval's default for
  ``how="MPP"`` is Nelson-Aalen; pass ``heuristic="Benard"`` for median
  ranks.
- **Ties and suspensions in rank regression.** Tools differ in how they
  rank tied failures and adjust ranks for suspensions (Johnson's method,
  Kaplan-Meier); with heavy censoring, rank regression results vary between
  tools more than MLE results do.
- **The censoring flag.** ``c = 1`` is a suspension. Data from a tool that
  marks failures with ``F`` and suspensions with ``S`` becomes
  ``c = (status == "S").astype(int)``.

What SurPyval adds
------------------

SurPyval is free and open source, and runs as a Python library, so an
analysis is a script you can rerun, review and test. Beyond the reliability
tools' standard menus, it has the methods of biostatistics on the same data
model: Kaplan-Meier, Nelson-Aalen and Turnbull estimates, Cox and other
regression models with covariates, competing risks regression, frailty,
joint models of condition readings and failures, and Bayesian fits with
priors. Any mix of failures, suspensions, interval and left-censored data
and truncation (units that came into the data set late, or only because
they had failed) goes into the same ``fit``.

The commercial tools have strengths of their own: point-and-click data
entry and reports, validated workflows for regulated industries, and
support contracts. Use SurPyval where a scriptable, testable analysis is
what you need, and check it against your tool on a shared data set, as
this page does, when you move work across.
