Degradation Analysis
====================

Sometimes failure data is scarce or takes too long to collect: items are
highly reliable, test time is limited, and few (or no) units fail during
the observation window. But failure is often the end point of a gradual,
measurable process — a crack grows, a resistance drifts, a lumen output
fades, a material wears. Degradation analysis exploits this: instead of
waiting for units to fail, we track a *degradation measurement* over time
on each unit, define failure as the measurement crossing a *threshold*,
and extrapolate each unit's degradation trend to that threshold to obtain
a failure time — even for units that never actually failed on test.

SurPyval implements the classic *pseudo-failure-time* approach:

1. A degradation *path model* (e.g. linear) is fitted, by least squares,
   to each unit's measurements.
2. Each unit's fitted path is extrapolated to the failure threshold. The
   crossing time is that unit's *pseudo failure time*.
3. A lifetime distribution (Weibull by default) is fitted to the pseudo
   failure times, and can then be used like any other SurPyval parametric
   model.

If a unit's fitted path never reaches the threshold (for example, the
unit is not degrading, or is trending away from the threshold), the unit
is treated as right censored at its last observed time, and the censoring
is passed through to the lifetime distribution fit.

Degradation path models
-----------------------

The path models available, and the pseudo failure time each implies for a
threshold :math:`y_{t}`, are:

.. list-table::
    :header-rows: 1

    * - Name
      - Path
      - Threshold crossing time
    * - ``"linear"``
      - :math:`y = a + b x`
      - :math:`(y_{t} - a) / b`
    * - ``"exponential"``
      - :math:`y = a e^{b x}`
      - :math:`\ln(y_{t} / a) / b`
    * - ``"power"``
      - :math:`y = a x^{b}`
      - :math:`(y_{t} / a)^{1/b}`
    * - ``"logarithmic"``
      - :math:`y = a + b \ln(x)`
      - :math:`e^{(y_{t} - a) / b}`
    * - ``"lloyd-lipow"``
      - :math:`y = a - b / x`
      - :math:`b / (a - y_{t})`

Degradation can be increasing (crack length) or decreasing (luminous
flux); the direction is captured by the sign of the fitted parameters and
needs no configuration. Models that are linear in their parameters
(linear, logarithmic, Lloyd-Lipow) are fitted in closed form; the others
(exponential, power) are fitted by nonlinear least squares started from
the log-linearised fit.

Example
-------

Consider four units whose degradation is measured every 100 hours, with
failure defined as the measurement reaching 150:

.. code:: python

    import numpy as np
    from surpyval.degradation import DegradationAnalysis

    x = np.tile(np.arange(100, 1100, 100), 4)
    i = np.repeat([1, 2, 3, 4], 10)
    slopes = np.repeat([0.31, 0.28, 0.44, 0.37], 10)
    y = 10 + slopes * x

    model = DegradationAnalysis.fit(x, y, i, threshold=150)
    print(model)

.. code:: text

    Degradation Analysis SurPyval Model
    ===================================
    Path Model          : Linear
    Threshold           : 150.0
    Number of Units     : 4
    Censored Units      : 0
    Life Distribution   : Weibull
    Parameters          :
         alpha: 441.4780882117898
          beta: 6.987078993008337

The fitted model exposes the per-unit results and forwards the usual
lifetime functions to the fitted life model:

.. code:: python

    model.pseudo_failure_times
    # array([451.61290323, 500.        , 318.18181818, 378.37837838])

    model.sf([300, 400, 500])
    # array([0.93496716, 0.60538471, 0.09196813])

    model.life_model    # the underlying Parametric Weibull model
    model.plot()        # data, fitted paths, and the threshold

Data can also come straight from a DataFrame with
``DegradationAnalysis.fit_from_df(df, x="time", y="measurement",
i="unit", threshold=150)``, the life distribution and fitting method can
be changed with the ``distribution`` and ``how`` arguments (e.g.
``distribution=LogNormal, how="MPP"``), and a custom path shape can be
used by passing a ``surpyval.degradation.PathModel`` subclass instance as
``path``.

Predicting a new unit's failure time
------------------------------------

A fitted model can estimate the failure time of a *new*, partially
observed unit from its degradation trajectory. The model fits its path
shape to the new measurements and extrapolates to the same threshold:

.. code:: python

    # a new unit observed for only 300 hours, degrading at ~0.35/hour
    x_new = [100, 200, 300]
    y_new = [45, 80, 115]

    model.predict_failure_time(x_new, y_new)   # ~400: when y reaches 150
    model.predict_remaining_life(x_new, y_new) # ~100: minus its age (300)

If the trajectory has already crossed the threshold, the predicted
failure time is in the past and the remaining life is negative. If the
new unit's fitted path never reaches the threshold (it is not
degrading), both return ``nan`` with a warning. For a population-level
view instead of a per-unit extrapolation, the fitted life model can be
used directly — e.g. the survival of a unit that has already survived
to time ``a``: ``model.life_model.cs(t, a)``.

The population path-parameter distribution
------------------------------------------

The fitted model also estimates the *population* distribution of the
path parameters, which is what a random-effects treatment (and any
Bayesian blending of a new unit's trajectory with the population) needs
as its prior:

.. code:: python

    model.path_param_mean        # mean path parameters across units
    model.path_param_cov         # between-unit covariance (corrected)
    model.path_param_sample_cov  # raw sample covariance (uncorrected)
    model.measurement_var        # pooled measurement-error variance

Because each unit's fitted parameters are least-squares *estimates*,
their scatter across units mixes two sources: real unit-to-unit
variability and per-unit estimation noise
(:math:`\mathrm{Cov}(\hat{\theta}_i) = \Sigma + V_i`). The raw sample
covariance therefore overstates the between-unit variability.
``path_param_cov`` applies the Lu-Meeker two-stage correction: the
measurement variance is pooled from the per-unit residuals, each
unit's estimation covariance :math:`V_i = \sigma^2 (J_i^T J_i)^{-1}`
is computed from the path Jacobian, and the average is subtracted from
the sample covariance. The result is projected onto the positive
semi-definite cone; if material clipping was needed (estimation noise
comparable to the between-unit scatter — few units or few measurements
per unit), a warning is raised and the corrected covariance should be
treated as unreliable. When every unit has only as many measurements
as path parameters, the measurement variance cannot be estimated and
no correction is applied.

A note on uncertainty
---------------------

The pseudo-failure-time approach is a two-stage method: the life
distribution is fitted to *estimated* failure times as if they had been
observed exactly, so the reported confidence bounds on the life model do
not include the path-fitting or extrapolation uncertainty. This is the
standard, pragmatic form of degradation analysis; random-effects
(Lu-Meeker, fitted by REML) and stochastic-process (Wiener, gamma
process) degradation models, which propagate that uncertainty, are
candidates for future work.
