Weighted Data
=============

Some data come with a number on each row that says how much that row
*counts*: a **weight**. SurPyval takes it as ``w``, beside the counts ``n``,
in the Kaplan-Meier, Nelson-Aalen and Fleming-Harrington estimators.
This page says what weights are, when to use ``w`` rather than ``n``, why a
weighted fit's intervals are computed differently, and what a weighted fit
will not do.

What a weight is
----------------

A count ``n`` says how many *identical units* a row records: ``n = 3`` is
three units, each seen, with the same time and the same censoring. Writing
the row out three times gives the same answer.

A weight ``w`` says how many units of the *population* each unit stands
for, or how much it should be up- or down-weighted to undo a bias. Nobody
saw ``w`` units. Three common sources:

- **Sampling weights.** You inspected one pump in every 40 at the small
  sites and one in every 5 at the big site. Each pump inspected at a small
  site stands for 40 pumps, each at the big site for 5. Without the weights
  the big site dominates the answer; with them the answer describes the
  whole fleet. Survey and registry data work the same way.
- **Inverse probability of treatment weights.** In observational data the
  units that got a treatment (an overhaul, a new supplier, a drug) differ
  from those that did not. Weighting each unit by one over the probability
  that it got what it got makes the groups comparable, as in a randomised
  trial.
- **Inverse probability of censoring weights.** If units drop out of
  follow-up for reasons tied to their risk, each unit still followed is
  weighted by one over its probability of still being followed.

A weight is any positive number (0.37, 12.5); a count is a whole number. A
row may have both: ``n = 3`` identical units, each standing for ``w = 12.5``.

When to use ``w`` and when ``n``
--------------------------------

.. list-table::
   :header-rows: 1
   :widths: 28 36 36

   * -
     - Count ``n``
     - Weight ``w``
   * - Meaning
     - ``n`` identical units were seen
     - one unit seen, standing for ``w`` units
   * - Values
     - 1, 2, 3, ...
     - any positive number
   * - Every value multiplied by 10
     - ten times the data: the intervals shrink
     - the same data: **nothing changes**
   * - Variance
     - the classical formulas
     - a robust (sandwich) variance

Use ``n`` when the numbers count units you saw. Use ``w`` when they say
how much each unit stands for, **even if they are whole numbers**: survey
weights of exactly 40 put in ``n`` would be read as forty units seen, and
the intervals would be more than six times too narrow. SurPyval refuses a
decimal ``n`` and says to use ``w``, but it cannot tell whole-number weights
from counts.

Why the intervals change
------------------------

The estimate itself is easy: each unit's contribution is multiplied by its
weight (to the numbers at risk and of failures, or to the likelihood).

The uncertainty is not. The classical formulas -- Greenwood's for a
Kaplan-Meier curve, Aalen's for a Nelson-Aalen one -- assume that the information in the data grows with the number of units, and
with weights they would take the *total of the weights* for that number.
Multiply every weight by 10 and they would claim ten times the information
from the same data. The honest amount of information depends on how uneven
the weights are: a few units with huge weights give a noisy answer however
large the total. A useful summary is the **effective sample size**,
:math:`(\sum n w)^2 / \sum n w^2`, which equals the number of units when
the weights are equal and is smaller the more uneven they are. A weighted
fit prints it, and warns when it is below 20.

So every weighted fit uses a **robust** variance, one that is valid
whatever the weights' scale, and there is no option of the classical one.
The intervals are then built from it exactly as before (log(-log) bounds
for a survival curve), so ``cb``, ``quantile_cb``, ``rmst`` and the rest
work as usual.

A worked example
----------------

Suppose a study samples long-lived units more often than short-lived ones:
of a population of lifetimes, a unit above the median is kept with
probability 0.8 and one below it with probability 0.2. The sample then
overstates how long units last. Each kept unit stands for one over its
probability of being kept:

.. jupyter-execute::

    import numpy as np
    import surpyval as surv

    rng = np.random.default_rng(1)
    truth = surv.Weibull.from_params([10.0, 1.5])
    t = truth.random(2000, random_state=rng)
    censor = rng.uniform(0, 30, 2000)
    keep_probability = np.where(t > truth.qf(0.5), 0.8, 0.2)
    kept = rng.uniform(size=2000) < keep_probability
    x = np.minimum(t, censor)[kept]
    c = (censor < t)[kept].astype(int)
    w = 1 / keep_probability[kept]

    print(surv.KaplanMeier.fit(x, c).sf([5, 10, 15]))  # too long-lived
    km = surv.KaplanMeier.fit(x, c, w=w)
    print(km.sf([5, 10, 15]))                          # close to the truth
    print(truth.sf([5, 10, 15]))
    print(km)

The weighted estimate recovers the population's survival, and its bounds
use the robust variance:

.. jupyter-execute::

    print(km.cb([5, 10, 15]))
    print(km.quantile_cb(0.5))

The same weights scaled by any constant give the same estimate and the
same intervals:

.. jupyter-execute::

    scaled = surv.KaplanMeier.fit(x, c, w=10 * w)
    print(np.allclose(scaled.cb([5, 10, 15]), km.cb([5, 10, 15])))

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert np.allclose(scaled.cb([5, 10, 15]), km.cb([5, 10, 15]), rtol=1e-12)
    assert np.all(np.abs(km.sf([5, 10, 15]) - truth.sf([5, 10, 15])) < 0.05)
    _plain = surv.KaplanMeier.fit(x, c).sf([5, 10, 15])
    assert np.all(_plain > truth.sf([5, 10, 15]))

``SurpyvalData`` holds the weights too (``SurpyvalData(x, c, n, w=w)``,
``SurpyvalData.from_dataframe(df, ..., w_col="weight")``), and
``fit_from_df`` takes ``w_col``.

What a weighted fit will not do
-------------------------------

Some bounds and methods are wrong with weights, and refuse with a message
saying what to use instead:

- **Simultaneous bands and** ``bootstrap_cb`` of a non-parametric estimate:
  use ``cb``.
- **Turnbull** and the fitters that do not take ``w`` (the parametric
  and regression fitters for now, competing risks, recurrent events,
  degradation, mixtures, ...), which refuse weighted data.
- **Reshaping weighted data** with ``censor_at``, ``select`` and the other
  record-based methods: index the rows, ``data[rows]``, which keeps the
  weights, or build a new ``SurpyvalData`` from the arrays.

Two cautions:

- **Estimated weights.** When the weights are themselves estimated (inverse
  probability of treatment weights from a logistic regression), the robust
  variance ignores that estimation and is usually a little conservative:
  intervals slightly too wide. This is what R and lifelines do too.
- **Complex survey designs.** With ``w`` alone, units are independent draws
  with unequal weights. Strata, clusters of units sampled together and
  finite-population corrections are out of scope; R's ``survey`` package
  handles full designs.

The formulas
------------

Kaplan-Meier, Nelson-Aalen and Fleming-Harrington use weighted numbers at
risk and of deaths, :math:`r_k = \sum_i n_i w_i Y_i(t_k)` and
:math:`d_k = \sum_i n_i w_i dN_i(t_k)`. Their variance is the
infinitesimal jackknife (R's ``survfit(..., weights = w, robust = TRUE)``):
with :math:`U_i(t)` the derivative of :math:`\log \hat S(t)` in unit
:math:`i`'s weight,

.. math::

    \widehat{Var}\,\log \hat S(t) = \sum_i n_i w_i^2 U_i(t)^2,
    \qquad
    U_i^{KM}(t) = -\sum_{t_k \le t}
        \frac{dN_i(t_k) - Y_i(t_k)\, d_k / r_k}{r_k - d_k}

(with :math:`r_k` in place of :math:`r_k - d_k` for Nelson-Aalen).
Fleming-Harrington's increment with :math:`m_k` tied deaths takes each at
the deaths' mean weight, :math:`\sum_{l < m_k} a_k / (r_k - l a_k)`,
:math:`a_k = d_k / m_k`, as R's ``ctype = 2`` does. The restricted mean's
variance is the same jackknife of the area under the curve.

Translating from lifelines and R
--------------------------------

- lifelines ``KaplanMeierFitter().fit(T, E, weights=w)`` is
  ``KaplanMeier.fit(T, 1 - E, w=w)``.
- R ``survfit(Surv(T, E) ~ 1, weights = w, robust = TRUE)`` is
  ``KaplanMeier.fit(T, 1 - E, w=w)``.

The reference tests (``surpyval/tests/reference/test_weights.py``) compare
these with R ``survival`` and with lifelines, and the
calibration study (``surpyval/tests/calibration/test_coverage_weights.py``)
checks the intervals' coverage under informative sampling, at any scale of
the weights, and with estimated treatment weights.
