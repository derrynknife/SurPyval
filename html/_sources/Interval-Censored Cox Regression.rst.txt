Interval-Censored Cox Regression
================================

Often a failure is not seen when it happens, only found at the next
look:

- a crack found at an inspection was not there at the one before;
- a patient's test is negative at one visit and positive at the next;
- a buried pipe is found leaking at a survey, years after it was laid.

The time of the event is then known only to lie *between* two
inspections. These are interval-censored data (``c = 2``, with ``x``
given as ``[left, right]`` pairs; see :doc:`Conventions`). An event
before the first look is left censored (``c = -1``), and a unit still
fine at its last look is right censored (``c = 1``).

The Cox model (:doc:`regression/cox_ph`) cannot use such rows: its
partial likelihood compares, at each event time, the unit that failed
with the units still at risk, and with intervals nobody knows when the
event happened or who was at risk then. ``IntervalCoxPH`` keeps what
makes the Cox model attractive -- the covariates multiply the hazard,
and the shape of the baseline is left to the data -- and fits the full
likelihood instead.

In plain words
--------------

- **What it estimates.** One hazard ratio per covariate, as in Cox
  regression: ``exp(coef) = 2`` means the event comes at twice the rate,
  at every age.
- **The baseline.** The survival curve of a unit with all covariates at
  0 is estimated without a formula, the way the Turnbull estimator
  estimates one group's curve from intervals: it can only drop inside
  the *Turnbull intervals*, the stretches of time where the data say
  failures happened. Where exactly inside such an interval the drop
  happens, the data cannot say; the curve is shown dropping at the
  interval's end.
- **How sure it is.** The standard errors come from the *profile
  likelihood*: how fast the fit's likelihood falls as a coefficient moves
  away from its estimate, with the baseline re-fitted at each value.
  ``param_cb`` gives Wald intervals from them, profile likelihood-ratio
  intervals (``method="lr"``) or bootstrap intervals
  (``method="bootstrap"``).

The model
---------

.. math::

    S(x \mid Z) = S_0(x)^{\exp(\beta' Z)}

Row :math:`i` adds the log of the probability of what was seen: for an
interval :math:`(l_i, r_i]`, :math:`S(l_i \mid Z_i) - S(r_i \mid Z_i)`;
for a left-censored row :math:`1 - S(r_i \mid Z_i)`; for a right-censored
row :math:`S(l_i \mid Z_i)`; for an exact time the drop of the curve
there. With an entry time :math:`t_{l,i}` (left truncation) the
probability is divided by :math:`S(t_{l,i} \mid Z_i)`. :math:`S_0` puts
mass only on the Turnbull intervals of all the rows pooled, and is
maximised together with :math:`\beta` (Finkelstein 1986; Huang 1996). It
is the model of R's ``icenReg::ic_sp(model = "ph")``, and SurPyval's fits
agree with it.

The standard errors are the profile likelihood's (Murphy and van der
Vaart 2000): the curvature of
:math:`p\ell(\beta) = \max_{S_0} \ell(\beta, S_0)` by second
differences, with a step of :math:`1/\sqrt{N}` standard deviations of
each covariate. With interval censoring the baseline is estimated more
slowly than the coefficients, so the curvature of the likelihood in the
baseline's masses is not a valid variance; the profile likelihood's is.

An example: a condition found at check-ups
------------------------------------------

Two hundred units, half of them treated (``treated``), with an age at
the start (``age``, centred), are checked every one to three months;
each check says only whether a condition has appeared yet. The data are
simulated, so the truth is known: treatment halves the hazard
(:math:`\beta = \log 0.5 \approx -0.69`) and each year of age raises it
by 5% (:math:`\beta \approx 0.049`).

.. jupyter-execute::

    import numpy as np
    import surpyval as sp

    rng = np.random.default_rng(11)
    n = 200
    treated = rng.binomial(1, 0.5, n)
    age = rng.normal(0, 10, n)
    life = 12 * (rng.exponential(size=n)
                 / np.exp(np.log(0.5) * treated + 0.049 * age)) ** (1 / 1.3)
    x, c = np.empty((n, 2)), np.empty(n, dtype=int)
    for i in range(n):
        visits = np.cumsum(np.r_[rng.uniform(0, 2), rng.uniform(1, 3, 40)])
        visits = visits[visits <= 36]
        if life[i] <= visits[0]:
            x[i], c[i] = visits[0], -1        # before the first check
        elif life[i] > visits[-1]:
            x[i], c[i] = visits[-1], 1        # still fine at the last
        else:
            k = np.searchsorted(visits, life[i]) - 1
            x[i], c[i] = (visits[k], visits[k + 1]), 2
    Z = np.column_stack([treated, age])
    print(np.bincount(c + 1))  # left, (exact), right, interval

.. jupyter-execute::

    model = sp.IntervalCoxPH.fit(x, Z, c=c)
    model

The hazard ratios with their 95% intervals, by each method:

.. jupyter-execute::

    print(np.exp(model.param_cb("coef_0")))
    print(np.exp(model.param_cb("coef_0", method="lr")))
    print(np.exp(model.param_cb("coef_0", method="bootstrap",
                                n_boot=200, random_state=1)))

The predicted survival of an untreated and a treated unit of average
age, at one and two years:

.. jupyter-execute::

    model.sf([12, 24], [[0, 0], [1, 0]], grid=True).round(3)

A parametric PH model of the same data assumes a Weibull baseline. Where
it fits, its coefficients are close, and it has bounds on the curves and
``diagnose()``; the semi-parametric model makes no assumption about the
baseline's shape.

.. jupyter-execute::

    weibull = sp.WeibullPH.fit(x, Z, c=c)
    print(weibull.params[-2:].round(3), model.beta.round(3))

Exact and right-censored data
-----------------------------

The model takes exact times too (``c = 0``). With only exact and
right-censored rows, it is the grouped-data proportional hazards model
(Prentice and Gloeckler 1978): each event time is a drop of a discrete
survival curve, where ``CoxPH``'s partial likelihood treats the baseline
as a continuous hazard. The coefficients are close and draw together as
the data grow, but are not equal; for such data ``CoxPH`` is the usual
choice, with its residuals, strata, time-varying covariates and
diagnostics.

What it does not do
-------------------

- **Bounds on the curves.** ``cb`` refuses: no variance of the
  interval-censored baseline is offered. Bound the coefficients with
  ``param_cb``, or use a parametric PH model (``WeibullPH``), whose
  ``cb`` bounds the curves.
- **Right truncation.** Refused; a parametric PH model takes
  ``t=[tl, tr]``.
- **Proportional odds.** The semi-parametric proportional odds model
  (``ProportionalOdds``) takes exact and right-censored data only; for
  interval-censored data use a parametric PO model (``PO(LogLogistic)``,
  ``WeibullPO``).
- **Diagnostics.** ``diagnose()`` refuses and says what to check
  instead: a parametric PH fit's ``diagnose()``, and each group's
  predicted curve against its Turnbull estimate.

References
----------

Finkelstein, D. M. (1986), "A proportional hazards model for
interval-censored failure time data", *Biometrics* 42, 845-854.

Huang, J. (1996), "Efficient estimation for the proportional hazards
model with interval censoring", *Annals of Statistics* 24, 540-568.

Murphy, S. A. and van der Vaart, A. W. (2000), "On profile likelihood",
*Journal of the American Statistical Association* 95, 449-465.

Prentice, R. L. and Gloeckler, L. A. (1978), "Regression analysis of
grouped survival data with application to breast cancer data",
*Biometrics* 34, 57-67.

Anderson-Bergman, C. (2017), "icenReg: regression models for interval
censored data in R", *Journal of Statistical Software* 81(12).
