Cox-Aalen
=========

A model with two kinds of covariate. Some scale the hazard by a constant
factor, as in Cox's model, and are summarised by a hazard ratio. Others
*add* to the hazard an amount that may change with time, as in Aalen's
additive model, and are summarised by a cumulative effect over time. The
Cox-Aalen model (Scheike and Zhang 2002) has both: a unit's hazard is its
additive part (a baseline plus each additive covariate's time-varying
excess) multiplied by ``exp(beta'Z)`` for the multiplicative covariates.

**When to use it.** When a covariate's effect is better described as an
excess rate that comes and goes (an exposure that adds risk early and not
later), while the others are well described by hazard ratios. With only
the baseline additive it is Cox's model, and with every covariate
additive it is Aalen's, so it sits between ``CoxPH`` and
``AalenAdditive``; a covariate that fails Cox's proportional-hazards check
can be moved into the additive part and the others kept as hazard ratios.

**Choosing the parts.** ``additive=`` names the columns of ``Z`` (by
position, or by name with ``fit_from_df`` or a DataFrame) that act
additively; the rest act multiplicatively. The default, no additive
columns, is Cox's model.

.. code-block:: python

    from surpyval import CoxAalen
    model = CoxAalen.fit(x, Z, c=c, additive=[1])     # column 1 additive
    model.summary()                                    # hazard ratios
    model.cumulative_coefficients([1, 2, 5])           # additive effects
    model.sf(5, [1, 0.5]); model.cb(5, [1, 0.5])

**What it reports.** ``beta`` (with ``summary()``, ``covariance()``,
``robust_covariance()``, ``standard_errors()`` and Wald ``param_cb``) for
the multiplicative covariates; ``cumulative_coefficients(t)`` and its
Wald bounds ``cumulative_coefficients_cb`` for the additive part, for a
unit with every multiplicative covariate at 0; and the predictions
``sf``, ``ff``, ``Hf``, ``qf``, ``random`` with Wald bounds ``cb`` on the
scales ``CoxPH.cb`` uses. There is no hazard rate ``hf`` (the additive
part is a step function; ``AalenAdditive.smoothed_coefficients`` smooths
one), no likelihood (no AIC, no likelihood-ratio bound; fit ``CoxPH`` for
one when the additive part is the baseline alone) and no bootstrap (refit
to resampled rows for one). Only observed and right-censored data, with
left truncation (``tl``), are taken.

**A negative additive hazard.** Nothing keeps the additive part positive:
a protective additive covariate can make the estimated additive hazard of
some units negative, so their estimated cumulative hazard falls. As
``AalenAdditive`` does, the prediction is then the running maximum of the
estimate from time 0, floored at 0, so survival stays in :math:`[0, 1]`
and never rises; ``cumulative_coefficients`` reports the estimate as it
is.

**Where the fit stops.** At each event time the additive part is a
least-squares fit over the units at risk, so it needs enough of them to
separate the additive covariates. Where they run out before the last
event (for example, every unit with a binary additive covariate at 1 has
left), the fit stops at the last time it can estimate, warns with that
time and the number of events left out, and records it in
``stopped_at``; every prediction after it (``sf``, ``ff``, ``Hf``,
``qf``, ``cb``, ``cumulative_coefficients`` and their bounds) is ``nan``
rather than a curve held flat, which would understate the risk. A fit
that does not stop has ``stopped_at`` ``nan`` and holds its curves after
the last event time, as the other semi-parametric models do.

**How far to trust the bounds.** The bounds are large-sample. Late in
follow-up, with only a handful of units at risk, the pointwise bounds on
the curves and the cumulative coefficients cover less often than stated
(about 92% for a nominal 95% with 4 units at risk, in one simulation).
Where the additive estimate falls and the prediction is held at its
running maximum, ``cb`` places the raw estimate's standard error about
the held value; it does not allow for the holding.

**The estimating equations.** With additive covariates :math:`X_i` (a
leading 1) and multiplicative ones :math:`Z_i`,
:math:`h_i(t) = \{X_i'\alpha(t)\} e^{\beta' Z_i}`. At each event time the
increments of :math:`A(t) = \int_0^t \alpha` are the weighted
least-squares fit :math:`d\hat A = (X'WX)^{-1} X' dN`, with
:math:`W` the at-risk counts times :math:`e^{\beta' Z}`, and
:math:`\beta` solves the likelihood score with that :math:`d\hat A` put
in, :math:`U(\beta) = \sum_t Z'\{I - WX(X'WX)^{-1}X'\} dN(t) = 0`, by
Newton's method. The variances, and how they reduce to Cox's and Aalen's,
are in the module docstring below.

**How it was checked.** No other package was used as a reference. With
the baseline alone additive the fit equals ``CoxPH`` with Breslow ties
(coefficients, model-based and robust standard errors, predicted
cumulative hazards and their bounds) to rounding, with tied times and
with delayed entry; with every covariate additive it equals
``AalenAdditive`` (cumulative coefficients and their variances); the
score written out again from the paper with explicit matrices is zero at
the estimate, and the derivative Newton's method uses matches its
numerical derivative; and a seeded simulation (n = 400) recovers the
hazard ratio and both cumulative coefficients with the stated coverage.

.. automodule:: surpyval.univariate.regression.cox_aalen.cox_aalen
    :no-members:

.. autoclass:: surpyval.univariate.regression.cox_aalen.cox_aalen.CoxAalen_
    :members:

.. autoclass:: surpyval.univariate.regression.cox_aalen.cox_aalen.CoxAalenModel
    :members:
