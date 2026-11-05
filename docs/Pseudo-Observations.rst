Pseudo-Observations
===================

Most survival regressions describe how a covariate changes the *hazard*
(Cox, PH) or the *time scale* (AFT). Often the question is simpler and
asked directly on a probability or a time: "how much more likely is a
patient in group B to be alive at one year?", "how many more days, on
average, does a pump last in its first year if it is run cooler?", "is
the chance of failing from corrosion by 5 years higher at the coast?".
Pseudo-observations answer such questions without a model of the hazard.

The idea in plain words
-----------------------

If no time were censored, the questions above would be ordinary
regressions: regress "alive at one year" (1 or 0), or "time lived in the
first year", ``min(x, 365)``, on the covariates. Censoring stops that,
because a unit censored at 200 days has no known one-year outcome.

A pseudo-observation fills that gap. Take the quantity the whole sample
estimates, say the Kaplan-Meier survival at one year. Recompute it with
one unit left out. How much the estimate moves when that unit is left
out says what the unit contributed: the unit's *pseudo-value* is
``N`` times the full estimate minus ``N - 1`` times the leave-one-out
estimate (a jackknife). For a unit followed past the time it is close to
its observed outcome (1 if it survived, 0 if it failed), and for a unit
censored before the time it is the sample's best guess (a value in
between). The pseudo-values then go into an ordinary regression as if
they were complete responses.

Three quantities are offered:

- **survival probability** at one or more times
  (:func:`~surpyval.pseudo_survival`, R's ``pseudo::pseudosurv``);
- **restricted mean survival time** (RMST) to a horizon ``tau``, the
  average time lived in ``[0, tau]``, the area under the Kaplan-Meier
  curve (:func:`~surpyval.pseudo_rmst`, R's ``pseudo::pseudomean``);
- **cumulative incidence** of one cause when there are competing causes
  of failure (:func:`~surpyval.pseudo_cif`, R's ``pseudo::pseudoci``).

The regression on them is :func:`~surpyval.pseudo_regression` (or
:func:`~surpyval.pseudo_regression_from_df`). It fits a generalised linear
model to the pseudo-values by estimating equations (GEE) with an
independence working model, and reports robust (sandwich) standard
errors, which allow for the pseudo-values not being independent of each
other. This is what R's ``geepack::geeglm(..., corstr =
"independence")`` gives on R's pseudo-values; SurPyval matches both to
within ``1e-8`` (the pseudo-values) and ``1e-6`` (coefficients and
standard errors).

The formulas
------------

For an estimator :math:`\hat\theta` of the quantity from all :math:`N`
units and :math:`\hat\theta_{(-i)}` from all but unit :math:`i`,

.. math::

    \hat\theta_i = N \hat\theta - (N - 1) \hat\theta_{(-i)}.

With :math:`\hat\theta = \hat S(t)` (Kaplan-Meier), :math:`\int_0^\tau
\hat S(u)\,du` (RMST) or :math:`\hat F_k(t)` (Aalen-Johansen), and
censoring independent of the covariates, :math:`E[\hat\theta_i \mid Z_i]
\approx \theta(Z_i)` (Andersen, Klein and Rosthøj 2003; Graw, Gerds and
Schumacher 2009). The regression solves

.. math::

    \sum_i D_i^\top (y_i - \mu_i) = 0, \qquad
    g(\mu_{ij}) = a_j + s\,\beta^\top Z_i,

over each unit's times :math:`j`, with :math:`D_i = \partial\mu_i /
\partial(a, \beta)`, and the robust covariance :math:`A^{-1} B A^{-1}`,
:math:`A = \sum_i D_i^\top D_i` and :math:`B = \sum_i D_i^\top r_i
r_i^\top D_i`. A unit with count ``n`` counts ``n`` times in both.

The sign of the coefficients
----------------------------

SurPyval reads every coefficient one way: **a positive coefficient means
a shorter life** (as its PH and AFT models). So:

- the **survival** regressions model the chance of the event by
  :math:`t`, :math:`F(t) = 1 - S(t)`. With the default complementary
  log-log link, :math:`\log(-\log S(t)) = a(t) + \beta^\top Z`, a
  proportional-hazards model at the chosen times: :math:`e^\beta` is a
  ratio of cumulative hazards, close to a Cox hazard ratio. With
  ``link="logit"``, :math:`e^\beta` is an odds ratio of failing by
  :math:`t`. (To compare with R, fit ``geeglm`` to ``1 - pseudosurv``.)
- the **cumulative incidence** regressions model the incidence of the
  cause, so a positive coefficient means more failures from it, as in
  Fine-Gray.
- the **RMST** regression is :math:`g(\text{RMST}) = a - \beta^\top Z`.
  With the identity link (the default) a covariate's coefficient is
  *minus* its difference in RMST: :math:`\beta = 20` means 20 time units
  *less* of the first :math:`\tau`. With ``link="log"``,
  :math:`e^{-\beta}` is the ratio of RMSTs. R's ``geeglm`` on
  ``pseudomean`` reports the same numbers with the other sign.

An example: lung cancer
-----------------------

.. jupyter-execute::

    import numpy as np
    import surpyval as surv

    lung = surv.datasets.load_lung()
    lung["c"] = 1 - lung["status"]          # 1 = censored
    lung["female"] = (lung["sex"] == 2).astype(int)

The average number of days lived in the first year, by sex and age:

.. jupyter-execute::

    rmst = surv.pseudo_regression_from_df(
        lung, "time", ["female", "age"], c_col="c", quantity="rmst", tau=365
    )
    rmst

Women live on average about 53 more days of the first year than men of
the same age (the coefficient is -52.6, with SurPyval's sign), with a
95% interval of 23 to 82 days.

The chance of being alive at 6, 12 and 18 months, with one intercept per
time and the covariates' effects shared:

.. jupyter-execute::

    surv_model = surv.pseudo_regression_from_df(
        lung, "time", ["female", "age"], c_col="c", t=[180, 365, 540]
    )
    surv_model.summary()[["coef", "exp(coef)", "se(coef)", "p"]]

``exp(coef) = 0.58`` for women: their cumulative hazard at each of these
times is about 0.58 times men's. Predictions, with bounds, at one of the
model's times:

.. jupyter-execute::

    Z = [[0, 60], [1, 60]]              # a man and a woman aged 60
    surv_model.predict(Z, t=365), surv_model.predict_cb(Z, t=365)

Competing risks
---------------

.. jupyter-execute::

    rng = np.random.default_rng(3)
    z = rng.integers(0, 2, 300)
    t1 = rng.exponential(10 * np.exp(-0.5 * z))   # cause "wear"
    t2 = rng.exponential(15, 300)                  # cause "impact"
    cens = rng.uniform(0, 25, 300)
    x = np.minimum.reduce([t1, t2, cens])
    e = np.where(x == cens, None, np.where(t1 < t2, "wear", "impact"))

    cif = surv.pseudo_regression(
        x, z, quantity="cif", e=e, cause="wear", t=[5, 10], names=["z"]
    )
    cif.summary()[["coef", "exp(coef)", "se(coef)", "p"]]

What is not covered
-------------------

- Delayed entry (``tl``) is refused: leave-one-out estimates of a
  left-truncated sample do not give pseudo-values with the right
  expectation. Use :class:`~surpyval.CoxPH` or a parametric model with
  ``tl``.
- Left and interval censoring are refused.
- Times past the largest time in the data are refused (the estimate is
  not defined there), and so is a time with no failures before it.
- Censoring must not depend on the covariates for the pseudo-values to be
  unbiased; where it does, the regression is biased (Binder, Gerds and
  Andersen 2014 give an inverse-probability weighted version).

Cost
----

Each leave-one-out estimate is a product over the distinct event times,
computed once for each distinct row, in blocks: the work is proportional
to (distinct rows) x (distinct event times). Pseudo-values at two times
and an RMST for 5,000 units with distinct times take about half a second.

References
----------

- Andersen, P. K., Klein, J. P. and Rosthøj, S. (2003). Generalised
  linear models for correlated pseudo-observations, with applications to
  multi-state models. *Biometrika* 90, 15-27.
- Andersen, P. K. and Perme, M. P. (2010). Pseudo-observations in survival
  analysis. *Statistical Methods in Medical Research* 19, 71-99.
- Klein, J. P., Gerster, M., Andersen, P. K., Tarima, S. and Perme, M. P.
  (2008). SAS and R functions to compute pseudo-values for censored data
  regression. *Computer Methods and Programs in Biomedicine* 89, 289-300.
- Graw, F., Gerds, T. A. and Schumacher, M. (2009). On pseudo-values for
  regression analysis in competing risks models. *Lifetime Data Analysis*
  15, 241-255.
